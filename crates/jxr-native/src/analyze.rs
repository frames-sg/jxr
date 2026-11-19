//! Forward transform and quantization: the exact inverse of [`crate::reconstruct`].
//!
//! Every stage undoes the matching reconstruction stage in reverse order, on
//! the same block and tile geometry, so a quantization step of one reproduces
//! the input samples exactly. The plane traversals, overlap window schedules,
//! and low-pass plane layout are reconstruction's own functions run with the
//! forward operators, so both directions share one vector and thread path.
//! This holds for full-resolution components and for the half-width and
//! half-size macroblocks of YUV 4:2:2 and 4:2:0 chroma alike.
//!
//! A component is analyzed as a stream down its plane, [`AnalysisStrip`],
//! which is reconstruction's
//! [`ComponentStrip`](crate::reconstruct::ComponentStrip) run backwards
//! through the same strip entries: rows of samples go in, and each
//! macroblock row comes out quantized once no row still to come can change
//! it. The whole plane is the strip given every row at once.

use core::ops::Range;

use jxr_core::{BandPresence, OverlapMode, QuantizerSet};
use jxr_math::{
    quantization::Quantizer,
    transform::{forward_chroma_420, forward_chroma_422, forward_core_transform},
};

use crate::{
    CpuCapabilities,
    cpu::I32Vectors,
    executor::Executor,
    reconstruct::{
        ForwardTransform, NOT_BEGUN, PARALLEL_MACROBLOCKS, PARALLEL_SAMPLES, Prefilter,
        QuantizedMacroblock, ReconstructionConfig, ReconstructionError, RowWindow, StripPlan,
        quantize_coefficients, transform_rows,
    },
};

/// Samples across and down one transform block.
const BLOCK_SAMPLES: usize = 4;

/// Transform and quantize one component plane.
///
/// `config` gives the component's macroblock shape exactly as reconstruction
/// takes it: four blocks square at full resolution, two across and four down
/// for YUV 4:2:2 chroma, two square for YUV 4:2:0 chroma. `samples` is the
/// row-major extended plane, a whole number of such macroblocks in each
/// direction, holding signed values after input bias and scaling.
///
/// The result is in raster macroblock order with prediction not yet applied.
/// High-pass element zero of every block is unused and zero, as is every
/// DC/low-pass entry and high-pass block beyond the component's block count.
///
/// This is an [`AnalysisStrip`] given the whole plane at once, which runs
/// each stage over the plane with `cpu`'s vectors and executor. The encoder
/// streams its planes; the whole plane is what the reduced-decode scale
/// probe inverts.
pub(crate) fn analyze_component(
    samples: Vec<i32>,
    config: &ReconstructionConfig,
    quantizers: QuantizerSet,
    bands: BandPresence,
    cpu: CpuCapabilities,
) -> Result<Vec<QuantizedMacroblock>, ReconstructionError> {
    let rows = usize::try_from(config.macroblocks_y)
        .map_err(|_| ReconstructionError::ArithmeticOverflow("macroblock height conversion"))?;
    let mut strip = AnalysisStrip::default();
    strip.begin(config, quantizers, bands, 0, cpu)?;
    let plan = strip.plan.as_ref().ok_or(NOT_BEGUN)?;
    let expected = rows
        .checked_mul(plan.strip.macroblock_height())
        .and_then(|height| height.checked_mul(plan.strip.width()))
        .ok_or(ReconstructionError::ArithmeticOverflow("plane size"))?;
    if samples.len() != expected {
        return Err(ReconstructionError::InvalidPlaneGeometry(
            "sample count differs from the macroblock grid",
        ));
    }
    let mut macroblocks = Vec::with_capacity(rows * plan.strip.macroblocks_x);
    strip.put_back(samples);
    if strip.advance()? != rows {
        return Err(ReconstructionError::InvalidPlaneGeometry(
            "plane not analyzed to its end",
        ));
    }
    strip.quantize_rows(0..rows, &mut macroblocks)?;
    Ok(macroblocks)
}

/// Macroblock rows an [`AnalysisStrip`] under `overlap` keeps below the
/// rows it has ready, at most, once it has taken in whole macroblock rows:
/// one for each level of overlap filtering that reaches across them.
pub(crate) const fn waiting_rows(overlap: OverlapMode) -> usize {
    match overlap {
        OverlapMode::None => 0,
        OverlapMode::One => 1,
        OverlapMode::Two => 2,
    }
}

/// What an [`AnalysisStrip`] runs over: its plane, and how it quantizes.
struct AnalysisPlan {
    strip: StripPlan,
    steps: Steps,
    /// A macroblock with the component's quantizers and bands, and no
    /// coefficients.
    template: QuantizedMacroblock,
    /// Where each stage runs the rows it is given.
    executor: Executor,
}

/// One component analyzed down its plane, a few rows at a time:
/// reconstruction's [`ComponentStrip`](crate::reconstruct::ComponentStrip)
/// run backwards, through the same strip entries.
///
/// Sample rows are taken in at the bottom. The overlap prefilter of the
/// samples runs every window wholly within the rows taken in and leaves the
/// rows above its last window final ([`StripPlan::filter_samples`]). A
/// macroblock row wholly above them is transformed, which waits for two
/// sample rows of the row below. Under second-level overlap the block DC
/// coefficients of each transformed row join a plane of them, which the
/// prefilter of the low-pass level runs down in the same way
/// ([`StripPlan::filter_low`]), so that a row waits for the row below it to
/// be transformed. A macroblock row whose block DCs are final is ready: the
/// caller quantizes it and then releases it.
///
/// Every window and block is run once, when the rows it reads have arrived,
/// and a strip holds only the rows taken in that are not yet released: a
/// macroblock row and four sample rows more, at most, than the caller lets
/// build up before it quantizes what is ready. Rows are counted from the top
/// of the plane, macroblock rows going out and sample rows coming in.
#[derive(Default)]
pub(crate) struct AnalysisStrip {
    plan: Option<AnalysisPlan>,
    /// Samples taken in: prefiltered, and then transformed, in place.
    samples: RowWindow,
    /// Sample rows above this are as the overlap prefilter leaves them.
    prefiltered: usize,
    /// Macroblock rows whose blocks have been transformed.
    transformed: usize,
    /// Under second-level overlap, the block DC coefficients of the
    /// macroblock rows transformed and not yet released.
    low: RowWindow,
    /// Rows of `low` above this are as the low-pass-level prefilter leaves
    /// them.
    low_finished: usize,
}

impl AnalysisStrip {
    /// Begins a strip at the top of the plane `config` describes, which
    /// quantizes with the decoder's steps `quantizers` and
    /// zeroes the bands `bands` leaves out. Storage is retained from the
    /// last strip.
    ///
    /// `rows` is the most macroblock rows the caller takes in at a time.
    /// Between them a strip that quantizes what is ready keeps at most the
    /// rows its filters wait on, a macroblock row for each overlap level,
    /// so storage for both is made once, and does not grow and move as rows
    /// arrive. Each stage runs the rows it is given on `cpu`'s vectors and
    /// through its executor.
    pub(crate) fn begin(
        &mut self,
        config: &ReconstructionConfig,
        quantizers: QuantizerSet,
        bands: BandPresence,
        rows: usize,
        cpu: CpuCapabilities,
    ) -> Result<(), ReconstructionError> {
        let strip = StripPlan::new(config, Some(cpu))?;
        let steps = Steps::new(quantizers, config.scale_after_first_transform, cpu)?;
        let height = strip.macroblock_height();
        let waiting = waiting_rows(config.overlap);
        self.samples.begin(strip.width(), 0, waiting * height);
        if rows > 0 {
            self.samples.reserve_rows((rows + waiting) * height);
        }
        self.prefiltered = 0;
        self.transformed = 0;
        self.low.begin(strip.low_width(), 0, 4 * strip.block_rows);
        self.low_finished = 0;
        self.plan = Some(AnalysisPlan {
            strip,
            steps,
            template: QuantizedMacroblock {
                dc_low_pass: [0; 16],
                high_pass: [0; 256],
                quantizers,
                bands,
            },
            executor: cpu.executor(),
        });
        Ok(())
    }

    /// Sample rows of the plane taken in and not yet released.
    pub(crate) const fn held(&self) -> Range<usize> {
        self.samples.held()
    }

    /// Samples across the plane.
    #[cfg(test)]
    pub(crate) const fn width(&self) -> usize {
        self.samples.width()
    }

    /// Macroblocks across the plane.
    pub(crate) fn macroblocks_x(&self) -> usize {
        self.plan
            .as_ref()
            .map_or(0, |plan| plan.strip.macroblocks_x)
    }

    /// Takes in the next `rows` sample rows, which the caller writes every
    /// sample of in the slice returned.
    pub(crate) fn append(&mut self, rows: usize) -> &mut [i32] {
        self.samples.append(rows)
    }

    /// The storage of the sample rows held, which hold its start, so that
    /// the caller can write the rows it takes in next below them in place:
    /// it gives the storage back with [`Self::put_back`], with nothing else
    /// done to the strip in between.
    pub(crate) fn take_storage(&mut self) -> Vec<i32> {
        self.samples.take()
    }

    /// Gives back the storage [`Self::take_storage`] took, every whole row
    /// of which the strip now holds: the rows it held, then those the
    /// caller wrote below them.
    pub(crate) fn put_back(&mut self, samples: Vec<i32>) {
        self.samples.put_back(samples);
    }

    /// Runs every stage as far as the rows taken in allow, and returns how
    /// many macroblock rows from the top of the plane are ready to quantize.
    ///
    /// At the bottom of the plane and of a hard tile every window ends, so
    /// a strip that has taken in the plane's last row has every row ready.
    pub(crate) fn advance(&mut self) -> Result<usize, ReconstructionError> {
        let Self {
            plan,
            samples,
            prefiltered,
            transformed,
            low,
            low_finished,
        } = self;
        let AnalysisPlan {
            strip, executor, ..
        } = plan.as_ref().ok_or(NOT_BEGUN)?;
        let (overlap, executor) = (strip.config.overlap, *executor);
        let height = strip.macroblock_height();
        let end = samples.end();
        if overlap == OverlapMode::None {
            *prefiltered = end;
        } else if end > *prefiltered {
            let held = *prefiltered..end;
            let rows = samples.rows_mut(held.clone());
            *prefiltered = strip.filter_samples::<Prefilter>(rows, held, executor, None)?;
        }
        let transformable = *prefiltered / height;
        if transformable > *transformed {
            let rows = *transformed * height..transformable * height;
            let width = strip.width();
            transform_rows::<ForwardTransform>(
                strip.level,
                samples.rows_mut(rows.clone()),
                width,
                executor,
                None,
            )?;
            if overlap == OverlapMode::Two {
                let dcs = low.append((transformable - *transformed) * strip.block_rows);
                // The DC coefficient of each block is the first sample of
                // the first of its rows.
                let block_rows = samples.rows(rows).chunks_exact(width * BLOCK_SAMPLES);
                for (row, block_row) in dcs.chunks_exact_mut(strip.low_width()).zip(block_rows) {
                    for (dc, &value) in row.iter_mut().zip(block_row.iter().step_by(BLOCK_SAMPLES))
                    {
                        *dc = value;
                    }
                }
            }
            *transformed = transformable;
        }
        let low_end = *transformed * strip.block_rows;
        if overlap != OverlapMode::Two {
            *low_finished = low_end;
        } else if low_end > *low_finished {
            let held = *low_finished..low_end;
            let rows = low.rows_mut(held.clone());
            *low_finished = strip.filter_low::<Prefilter>(rows, held, executor, None)?;
        }
        Ok(*low_finished / strip.block_rows)
    }

    /// Quantizes the macroblock rows `rows` of the plane, which must be
    /// ready and not released, and appends their macroblocks to `output`
    /// in raster order.
    pub(crate) fn quantize_rows(
        &self,
        rows: Range<usize>,
        output: &mut Vec<QuantizedMacroblock>,
    ) -> Result<(), ReconstructionError> {
        let plan = self.plan.as_ref().ok_or(NOT_BEGUN)?;
        let strip = &plan.strip;
        let (height, block_rows) = (strip.macroblock_height(), strip.block_rows);
        if rows.start > rows.end
            || rows.start * height < self.samples.held().start
            || rows.end * block_rows > self.low_finished
        {
            return Err(ReconstructionError::InvalidPlaneGeometry(
                "macroblock rows that are not ready",
            ));
        }
        if rows.is_empty() {
            return Ok(());
        }
        let samples = self.samples.rows(rows.start * height..rows.end * height);
        let low = (strip.config.overlap == OverlapMode::Two).then(|| {
            self.low
                .rows(rows.start * block_rows..rows.end * block_rows)
        });
        let first = output.len();
        let rows = Gathered {
            samples,
            low,
            count: rows.len(),
        };
        gather_macroblocks(&rows, plan, output);
        let (steps, bands) = (&plan.steps, plan.template.bands);
        let second_level = |macroblock: &mut QuantizedMacroblock| {
            forward_first_level(&mut macroblock.dc_low_pass, strip)?;
            steps.quantize(macroblock, bands);
            Ok(())
        };
        let quantized = &mut output[first..];
        plan.executor
            .if_worth(quantized.len() >= PARALLEL_MACROBLOCKS)
            .try_for_each(quantized, |_, macroblock| second_level(macroblock))
    }

    /// Lets go of the macroblock rows above `row`, which have been
    /// quantized; rows that are not ready are kept.
    pub(crate) fn release(&mut self, row: usize) {
        let Some(plan) = &self.plan else {
            return;
        };
        let block_rows = plan.strip.block_rows;
        let row = row.min(self.low_finished / block_rows);
        self.samples
            .discard_before(row * plan.strip.macroblock_height());
        self.low.discard_before(row * block_rows);
    }
}

/// Transformed macroblock rows of a strip, as [`gather_macroblocks`] reads
/// them.
struct Gathered<'a> {
    /// The transformed samples of the rows.
    samples: &'a [i32],
    /// The block DC coefficients of the rows after the low-pass-level
    /// prefilter, under second-level overlap.
    low: Option<&'a [i32]>,
    /// Macroblock rows.
    count: usize,
}

/// Collects transformed macroblock rows into raster-ordered macroblocks
/// appended to `output`, each block's DC with its other coefficients or,
/// where the low-pass level is filtered, as that filter leaves it.
///
/// Each macroblock is made where it is filled, so on a large band the
/// macroblocks, which are larger than the samples they come from, are
/// written once and by every thread rather than cleared by one first.
fn gather_macroblocks(
    rows: &Gathered<'_>,
    plan: &AnalysisPlan,
    output: &mut Vec<QuantizedMacroblock>,
) {
    let strip = &plan.strip;
    let row_samples = strip.width() * strip.macroblock_height();
    let (block_columns, block_rows) = (strip.block_columns, strip.block_rows);
    let low_width = strip.low_width();
    let gather = |index: usize| {
        let row = index / strip.macroblocks_x;
        let column = index % strip.macroblocks_x;
        let mut macroblock = plan.template.clone();
        gather_blocks(
            &rows.samples[row * row_samples..(row + 1) * row_samples],
            strip,
            column * block_columns * BLOCK_SAMPLES,
            &mut macroblock,
        );
        if let Some(low) = rows.low {
            let low = &low[row * block_rows * low_width..];
            for (block_row, dcs) in macroblock.dc_low_pass[..block_columns * block_rows]
                .chunks_exact_mut(block_columns)
                .enumerate()
            {
                let start = block_row * low_width + column * block_columns;
                dcs.copy_from_slice(&low[start..start + block_columns]);
            }
        }
        macroblock
    };
    plan.executor
        .if_worth(rows.samples.len() >= PARALLEL_SAMPLES)
        .extend_with(
            output,
            strip.macroblocks_x * rows.count,
            strip.macroblocks_x,
            gather,
        );
}

/// Splits the transformed blocks of one macroblock: each block's DC into the
/// low-pass array and its other fifteen coefficients into high-pass. This is
/// the block placement of reconstruction's second level, reversed; blocks the
/// component's macroblock does not have are left untouched.
fn gather_blocks(
    band: &[i32],
    strip: &StripPlan,
    origin_x: usize,
    macroblock: &mut QuantizedMacroblock,
) {
    let width = strip.width();
    for block in 0..strip.block_columns * strip.block_rows {
        let x = origin_x + block % strip.block_columns * BLOCK_SAMPLES;
        let y = block / strip.block_columns * BLOCK_SAMPLES;
        let coefficients = &mut macroblock.high_pass[block * 16..block * 16 + 16];
        for row in 0..BLOCK_SAMPLES {
            let start = (y + row) * width + x;
            coefficients[row * BLOCK_SAMPLES..(row + 1) * BLOCK_SAMPLES]
                .copy_from_slice(&band[start..start + BLOCK_SAMPLES]);
        }
        macroblock.dc_low_pass[block] = coefficients[0];
        coefficients[0] = 0;
    }
}

/// Transforms the block DCs of one macroblock into its DC and low-pass
/// coefficients with the operator reconstruction inverts for the same shape.
fn forward_first_level(low: &mut [i32; 16], strip: &StripPlan) -> Result<(), ReconstructionError> {
    match (strip.block_columns, strip.block_rows) {
        (4, 4) => forward_core_transform(low)
            .map_err(|_| ReconstructionError::ArithmeticOverflow("forward core transform")),
        (2, 4) => {
            let mut samples = [0; 8];
            samples.copy_from_slice(&low[..8]);
            forward_chroma_422(&mut samples).map_err(|_| {
                ReconstructionError::ArithmeticOverflow("YUV422 first forward transform")
            })?;
            low[..8].copy_from_slice(&samples);
            Ok(())
        }
        (2, 2) => {
            let mut samples = [0; 4];
            samples.copy_from_slice(&low[..4]);
            forward_chroma_420(&mut samples).map_err(|_| {
                ReconstructionError::ArithmeticOverflow("YUV420 first forward transform")
            })?;
            low[..4].copy_from_slice(&samples);
            Ok(())
        }
        _ => Err(ReconstructionError::InvalidPlaneGeometry(
            "unsupported first-level transform shape",
        )),
    }
}

/// The quantizer of one band with the rounding offset the band uses.
#[derive(Clone, Copy)]
struct BandQuantizer {
    quantizer: Quantizer,
    /// Added to a magnitude before the division.
    offset: u32,
}

impl BandQuantizer {
    const fn new(quantizer: Quantizer, offset: u32) -> Self {
        Self { quantizer, offset }
    }

    /// Quantizes every coefficient in place, on `vectors` where it can.
    ///
    /// A step of one divides nothing, so a lossless band is left as it is
    /// without a division for each of its coefficients.
    #[inline]
    fn quantize_all(self, vectors: Option<I32Vectors>, coefficients: &mut [i32]) {
        if self.quantizer.step() != 1 {
            quantize_coefficients(vectors, self.quantizer, self.offset, coefficients);
        }
    }
}

/// The quantizer of each band.
struct Steps {
    dc: BandQuantizer,
    low_pass: BandQuantizer,
    high_pass: BandQuantizer,
    /// The vector set the bands are quantized on, where there is one.
    vectors: Option<I32Vectors>,
}

impl Steps {
    /// Resolve the divisors for the decoder's `quantizers`.
    ///
    /// When reconstruction doubles the result of its DC/low-pass transform,
    /// those two bands are divided by twice their step instead: the decoder
    /// then recovers the same magnitude, and no value is halved and rounded
    /// before the transform that follows. The bands are quantized on the
    /// vectors of `cpu`, where it has them.
    fn new(
        quantizers: QuantizerSet,
        doubled_after_transform: bool,
        cpu: CpuCapabilities,
    ) -> Result<Self, ReconstructionError> {
        let low_band = |step: u32, band: &'static str| {
            let divisor = if doubled_after_transform {
                step.checked_mul(2)
                    .ok_or(ReconstructionError::ArithmeticOverflow("quantizer scaling"))?
            } else {
                step
            };
            let quantizer =
                Quantizer::new(divisor).ok_or(ReconstructionError::ZeroQuantizer(band))?;
            // DC and low-pass round to nearest.
            Ok(BandQuantizer::new(quantizer, divisor / 2))
        };
        let high_pass = Quantizer::new(quantizers.high_pass)
            .ok_or(ReconstructionError::ZeroQuantizer("high-pass"))?;
        Ok(Self {
            dc: low_band(quantizers.dc, "DC")?,
            low_pass: low_band(quantizers.low_pass, "low-pass")?,
            // High-pass uses a 3/8 offset: its coefficients cluster near
            // zero, so a wider zero bin costs little distortion and saves
            // the bits of many isolated ones.
            high_pass: BandQuantizer::new(high_pass, quantizers.high_pass / 8 * 3),
            vectors: I32Vectors::of(cpu.level()),
        })
    }

    /// Quantizes in place and zeroes the bands the codestream will not carry.
    fn quantize(&self, macroblock: &mut QuantizedMacroblock, bands: BandPresence) {
        let (dc, low) = macroblock.dc_low_pass.split_at_mut(1);
        self.dc.quantize_all(self.vectors, dc);
        if bands.has_low_pass() {
            self.low_pass.quantize_all(self.vectors, low);
        } else {
            low.fill(0);
        }
        if bands.has_high_pass() {
            // Element zero of each block is not a high-pass coefficient and
            // is already zero.
            self.high_pass
                .quantize_all(self.vectors, &mut macroblock.high_pass);
        } else {
            macroblock.high_pass.fill(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::encode::test_support::Lcg;
    use crate::reconstruct::{TilePartition, reconstruct_luma};

    fn cpu() -> CpuCapabilities {
        CpuCapabilities::detect()
    }

    /// Blocks across and down one macroblock: full resolution, YUV 4:2:2
    /// chroma, and YUV 4:2:0 chroma.
    const SHAPES: [(u8, u8); 3] = [(4, 4), (2, 4), (2, 2)];
    const LUMA: (u8, u8) = (4, 4);

    fn plane(seed: u64, width: usize, height: usize) -> Vec<i32> {
        let mut state = seed;
        (0..width * height)
            .map(|index| {
                state = Lcg::advance(state);
                // A smooth ramp plus noise exercises DC, LP, and HP together.
                let ramp = i32::try_from((index % width) * 3 + index / width * 5).unwrap();
                ramp + ((state >> 40) % 512) as i32 - 256
            })
            .collect()
    }

    /// A noisy plane filling the macroblock grid of `config`.
    fn plane_for(seed: u64, config: &ReconstructionConfig) -> Vec<i32> {
        let width = usize::try_from(config.macroblocks_x).unwrap()
            * usize::from(config.block_columns)
            * BLOCK_SAMPLES;
        let height = usize::try_from(config.macroblocks_y).unwrap()
            * usize::from(config.block_rows)
            * BLOCK_SAMPLES;
        plane(seed, width, height)
    }

    fn shaped_config(
        (block_columns, block_rows): (u8, u8),
        overlap: OverlapMode,
        tiles: TilePartition,
    ) -> ReconstructionConfig {
        ReconstructionConfig {
            macroblock_origin_x: 0,
            macroblock_origin_y: 0,
            macroblocks_x: *tiles.column_boundaries.last().unwrap(),
            macroblocks_y: *tiles.row_boundaries.last().unwrap(),
            block_columns,
            block_rows,
            scale_after_first_transform: false,
            overlap,
            tiles,
        }
    }

    fn config(overlap: OverlapMode, tiles: TilePartition) -> ReconstructionConfig {
        shaped_config(LUMA, overlap, tiles)
    }

    fn worst_error(reconstructed: &[i32], samples: &[i32]) -> u32 {
        assert_eq!(reconstructed.len(), samples.len());
        reconstructed
            .iter()
            .zip(samples)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap()
    }

    const UNIT: QuantizerSet = QuantizerSet {
        dc: 1,
        low_pass: 1,
        high_pass: 1,
    };

    /// One tile, then soft and hard multi-tile layouts of a 5-by-3 macroblock
    /// plane. The last has tiles one macroblock wide and tall, the smallest
    /// region either overlap schedule filters.
    fn partitions() -> [TilePartition; 5] {
        let tiled = |columns: &[u32], rows: &[u32], hard_boundaries| TilePartition {
            column_boundaries: columns.to_vec(),
            row_boundaries: rows.to_vec(),
            hard_boundaries,
        };
        [
            TilePartition::single(5, 3),
            tiled(&[0, 2, 5], &[0, 1, 3], false),
            tiled(&[0, 2, 5], &[0, 1, 3], true),
            tiled(&[0, 1, 4, 5], &[0, 2, 3], false),
            tiled(&[0, 1, 4, 5], &[0, 2, 3], true),
        ]
    }

    #[test]
    fn unit_quantizers_reconstruct_the_input_exactly() {
        for shape in SHAPES {
            for overlap in [OverlapMode::None, OverlapMode::One, OverlapMode::Two] {
                for tiles in partitions() {
                    let config = shaped_config(shape, overlap, tiles);
                    let samples = plane_for(11, &config);
                    let macroblocks =
                        analyze_component(samples.clone(), &config, UNIT, BandPresence::All, cpu())
                            .unwrap();
                    let reconstructed = reconstruct_luma(&macroblocks, &config).unwrap();
                    assert_eq!(
                        reconstructed.samples, samples,
                        "{shape:?} {overlap:?} {:?}",
                        config.tiles
                    );
                }
            }
        }
    }

    #[test]
    fn planes_large_enough_for_the_parallel_stages_reconstruct_exactly() {
        let tiles = TilePartition {
            column_boundaries: vec![0, 1, 7, 16],
            row_boundaries: vec![0, 9, 16],
            hard_boundaries: true,
        };
        for shape in SHAPES {
            let config = shaped_config(shape, OverlapMode::Two, tiles.clone());
            let samples = plane_for(15, &config);
            assert!(samples.len() >= PARALLEL_SAMPLES);
            let macroblocks =
                analyze_component(samples.clone(), &config, UNIT, BandPresence::All, cpu())
                    .unwrap();
            assert!(macroblocks.len() >= PARALLEL_MACROBLOCKS);
            let reconstructed = reconstruct_luma(&macroblocks, &config).unwrap();
            assert_eq!(reconstructed.samples, samples, "{shape:?}");
        }
    }

    /// The plane of `config` streamed through an analysis strip `step`
    /// sample rows at a time, every row quantized on its own as soon as it
    /// is ready and then released. Rows go in through the strip's own
    /// storage and through storage it lends, by turns. Returns the
    /// macroblocks and the most sample rows the strip held after quantizing
    /// what was ready.
    fn streamed(
        samples: &[i32],
        config: &ReconstructionConfig,
        step: usize,
        quantizers: QuantizerSet,
    ) -> (Vec<QuantizedMacroblock>, usize) {
        let rows = usize::try_from(config.macroblocks_y).unwrap();
        let mut strip = AnalysisStrip::default();
        strip
            .begin(
                config,
                quantizers,
                BandPresence::All,
                1,
                cpu().on_calling_thread(),
            )
            .unwrap();
        let width = strip.width();
        let height = samples.len() / width;
        let (mut all, mut quantized, mut most) = (Vec::new(), 0, 0);
        for (index, first) in (0..height).step_by(step).enumerate() {
            let band = &samples[first * width..(first + step).min(height) * width];
            if index % 2 == 0 {
                strip.append(band.len() / width).copy_from_slice(band);
            } else {
                let mut storage = strip.take_storage();
                storage.extend_from_slice(band);
                strip.put_back(storage);
            }
            let ready = strip.advance().unwrap();
            assert!(
                strip
                    .quantize_rows(ready..ready + 1, &mut Vec::new())
                    .is_err()
            );
            for row in quantized..ready {
                strip.quantize_rows(row..row + 1, &mut all).unwrap();
                strip.release(row + 1);
                // A released row is gone.
                assert!(strip.quantize_rows(row..row + 1, &mut Vec::new()).is_err());
            }
            quantized = ready;
            most = most.max(strip.held().len());
        }
        assert_eq!(quantized, rows);
        (all, most)
    }

    /// Tile layouts of a 5-by-7 macroblock plane, soft and hard: one tile,
    /// tiles of several sizes, and tiles one macroblock wide and tall.
    fn tall_partitions() -> Vec<TilePartition> {
        let tiled = |columns: &[u32], rows: &[u32], hard_boundaries| TilePartition {
            column_boundaries: columns.to_vec(),
            row_boundaries: rows.to_vec(),
            hard_boundaries,
        };
        let mut partitions = vec![TilePartition::single(5, 7)];
        for hard in [false, true] {
            partitions.push(tiled(&[0, 2, 5], &[0, 3, 7], hard));
            partitions.push(tiled(&[0, 1, 4, 5], &[0, 1, 2, 5, 6, 7], hard));
        }
        partitions
    }

    /// A plane streamed in bands of any height, whole macroblock rows or
    /// not, is the plane analyzed whole: for every overlap mode, macroblock
    /// shape and tile layout.
    #[test]
    fn bands_of_every_height_analyze_as_the_whole_plane() {
        let quantizers = QuantizerSet {
            dc: 3,
            low_pass: 5,
            high_pass: 7,
        };
        for shape in SHAPES {
            let height = usize::from(shape.1) * BLOCK_SAMPLES;
            for overlap in [OverlapMode::None, OverlapMode::One, OverlapMode::Two] {
                for tiles in tall_partitions() {
                    for scaled in [false, true] {
                        let config = ReconstructionConfig {
                            scale_after_first_transform: scaled,
                            ..shaped_config(shape, overlap, tiles.clone())
                        };
                        let samples = plane_for(23, &config);
                        for quantizers in [UNIT, quantizers] {
                            let whole = analyze_component(
                                samples.clone(),
                                &config,
                                quantizers,
                                BandPresence::All,
                                cpu(),
                            )
                            .unwrap();
                            for step in [1, 3, height, 2 * height, 4 * height, 7 * height] {
                                let (streamed, _) = streamed(&samples, &config, step, quantizers);
                                assert!(
                                    streamed == whole,
                                    "{shape:?} {overlap:?} scaled {scaled} {quantizers:?} \
                                     {tiles:?} steps of {step} rows"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Taken in a macroblock row or more at a time and quantized as rows
    /// become ready, a strip holds at most the rows its filters wait on: two
    /// macroblock rows under second-level overlap, one under first-level
    /// overlap and none without, whatever the height of a band.
    #[test]
    fn a_strip_holds_only_the_rows_its_filters_wait_on() {
        for shape in SHAPES {
            let height = usize::from(shape.1) * BLOCK_SAMPLES;
            for (overlap, waiting) in [
                (OverlapMode::None, 0),
                (OverlapMode::One, height),
                (OverlapMode::Two, 2 * height),
            ] {
                for tiles in tall_partitions() {
                    let config = shaped_config(shape, overlap, tiles);
                    let samples = plane_for(29, &config);
                    for rows in [1, 2, 3] {
                        let (_, most) = streamed(&samples, &config, rows * height, UNIT);
                        assert!(
                            most <= waiting,
                            "{shape:?} {overlap:?} {:?}: {most} rows held",
                            config.tiles
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rows_are_quantized_only_once_ready_and_until_released() {
        let config = config(OverlapMode::Two, TilePartition::single(5, 7));
        let samples = plane_for(31, &config);
        let mut strip = AnalysisStrip::default();
        assert_eq!(strip.advance(), Err(NOT_BEGUN));
        strip
            .begin(
                &config,
                UNIT,
                BandPresence::All,
                2,
                cpu().on_calling_thread(),
            )
            .unwrap();
        let mut output = Vec::new();
        // Nothing is ready before any row arrives, nor a row whose
        // neighbour below has not.
        assert!(strip.quantize_rows(0..1, &mut output).is_err());
        strip.append(32).copy_from_slice(&samples[..32 * 80]);
        assert_eq!(strip.advance(), Ok(0));
        strip.append(16).copy_from_slice(&samples[32 * 80..48 * 80]);
        assert_eq!(strip.advance(), Ok(1));
        assert!(strip.quantize_rows(0..2, &mut output).is_err());
        strip.quantize_rows(0..1, &mut output).unwrap();
        assert_eq!(output.len(), 5);
        // Quantizing appends.
        strip.quantize_rows(0..1, &mut output).unwrap();
        assert_eq!(output[..5], output[5..]);
        strip.release(1);
        assert!(strip.quantize_rows(0..1, &mut output).is_err());
        // Releasing what is not ready keeps it.
        strip.release(7);
        assert_eq!(strip.held(), 16..48);
    }

    #[test]
    fn a_band_is_quantized_as_its_quantizer_divides_and_a_step_of_one_changes_nothing() {
        let extremes = [0, 1, -1, 7, -8, 1_000, -1_001, i32::MAX, i32::MIN];
        // Within the vector domain, and a band one coefficient leaves.
        let inside: [i32; 16] = core::array::from_fn(|index| {
            let value = i32::try_from(index * index * 997).unwrap();
            if index % 2 == 0 { value } else { -value }
        });
        let vectors = I32Vectors::of(cpu().level());
        for coefficients in [&extremes[..], &inside[..]] {
            for step in [1, 2, 3, 16, 255, 4_099] {
                let quantizer = Quantizer::new(step).unwrap();
                for offset in [0, step / 8 * 3, step / 2] {
                    for vectors in [vectors, None] {
                        let mut band = coefficients.to_vec();
                        BandQuantizer::new(quantizer, offset).quantize_all(vectors, &mut band);
                        let expected: Vec<i32> = coefficients
                            .iter()
                            .map(|&coefficient| quantizer.quantize(coefficient, offset))
                            .collect();
                        assert_eq!(band, expected, "step {step} offset {offset}");
                        assert_eq!(band == coefficients, step == 1);
                    }
                }
            }
        }
    }

    #[test]
    fn coarser_quantizers_bound_the_reconstruction_error() {
        let quantizers = QuantizerSet {
            dc: 8,
            low_pass: 8,
            high_pass: 8,
        };
        for shape in SHAPES {
            for overlap in [OverlapMode::None, OverlapMode::One, OverlapMode::Two] {
                let config = shaped_config(shape, overlap, TilePartition::single(5, 3));
                let samples = plane_for(12, &config);
                let macroblocks = analyze_component(
                    samples.clone(),
                    &config,
                    quantizers,
                    BandPresence::All,
                    cpu(),
                )
                .unwrap();
                let reconstructed = reconstruct_luma(&macroblocks, &config).unwrap();
                let worst = worst_error(&reconstructed.samples, &samples);
                assert!(worst > 0, "a step of eight is lossy for noisy input");
                assert!(
                    worst <= 16,
                    "{shape:?} {overlap:?}: worst sample error {worst}"
                );
            }
        }
    }

    #[test]
    fn doubled_low_bands_reconstruct_within_the_quantization_error() {
        let quantizers = QuantizerSet {
            dc: 2,
            low_pass: 2,
            high_pass: 2,
        };
        for shape in SHAPES {
            for tiles in partitions() {
                let config = ReconstructionConfig {
                    scale_after_first_transform: true,
                    ..shaped_config(shape, OverlapMode::Two, tiles)
                };
                let samples = plane_for(14, &config);
                let macroblocks = analyze_component(
                    samples.clone(),
                    &config,
                    quantizers,
                    BandPresence::All,
                    cpu(),
                )
                .unwrap();
                // The recorded steps are the decoder's, not the doubled divisors.
                assert_eq!(macroblocks[0].quantizers, quantizers);
                let reconstructed = reconstruct_luma(&macroblocks, &config).unwrap();
                let worst = worst_error(&reconstructed.samples, &samples);
                assert!(worst <= 8, "{shape:?} {:?}: worst {worst}", config.tiles);
            }
        }
    }

    #[test]
    fn doubled_low_bands_halve_the_dc_and_low_pass_coefficients() {
        // A unit step doubled, rounding to nearest.
        let two = Quantizer::new(2).unwrap();
        for shape in SHAPES {
            let plain = shaped_config(shape, OverlapMode::Two, TilePartition::single(5, 3));
            let doubled = ReconstructionConfig {
                scale_after_first_transform: true,
                ..plain.clone()
            };
            let samples = plane_for(16, &plain);
            let reference =
                analyze_component(samples.clone(), &plain, UNIT, BandPresence::All, cpu()).unwrap();
            let halved =
                analyze_component(samples, &doubled, UNIT, BandPresence::All, cpu()).unwrap();
            for (reference, halved) in reference.iter().zip(&halved) {
                assert_eq!(halved.high_pass, reference.high_pass, "{shape:?}");
                for (&full, &half) in reference.dc_low_pass.iter().zip(&halved.dc_low_pass) {
                    assert_eq!(half, two.quantize(full, 1), "{shape:?}");
                }
            }
        }
    }

    #[test]
    fn a_flat_plane_has_only_dc_for_every_shape() {
        for shape in SHAPES {
            let config = shaped_config(shape, OverlapMode::None, TilePartition::single(5, 3));
            let samples = vec![40; plane_for(0, &config).len()];
            let macroblocks =
                analyze_component(samples, &config, UNIT, BandPresence::All, cpu()).unwrap();
            assert_eq!(macroblocks.len(), 15);
            for macroblock in &macroblocks {
                assert_ne!(macroblock.dc_low_pass[0], 0, "{shape:?}");
                assert_eq!(macroblock.dc_low_pass, macroblocks[0].dc_low_pass);
                assert!(macroblock.dc_low_pass[1..].iter().all(|&value| value == 0));
                assert!(macroblock.high_pass.iter().all(|&value| value == 0));
            }
        }
    }

    #[test]
    fn entries_beyond_the_component_block_count_are_zero() {
        for shape in [(2, 4), (2, 2)] {
            let blocks = usize::from(shape.0) * usize::from(shape.1);
            for overlap in [OverlapMode::None, OverlapMode::Two] {
                let config = shaped_config(shape, overlap, TilePartition::single(5, 3));
                let samples = plane_for(17, &config);
                let macroblocks =
                    analyze_component(samples, &config, UNIT, BandPresence::All, cpu()).unwrap();
                for macroblock in &macroblocks {
                    assert!(macroblock.dc_low_pass[blocks..].iter().all(|&v| v == 0));
                    assert!(macroblock.high_pass[blocks * 16..].iter().all(|&v| v == 0));
                    // Element zero of every block belongs to the DC/LP array.
                    assert!(macroblock.high_pass.iter().step_by(16).all(|&v| v == 0));
                }
                // The present entries do carry the noisy plane.
                assert!(macroblocks.iter().any(|macroblock| {
                    macroblock.dc_low_pass[blocks - 1] != 0
                        && macroblock.high_pass[(blocks - 1) * 16 + 1..blocks * 16]
                            .iter()
                            .any(|&v| v != 0)
                }));
            }
        }
    }

    #[test]
    fn absent_bands_are_zeroed() {
        for shape in SHAPES {
            let config = shaped_config(shape, OverlapMode::None, TilePartition::single(5, 3));
            let samples = plane_for(13, &config);
            let dc_only =
                analyze_component(samples.clone(), &config, UNIT, BandPresence::DcOnly, cpu())
                    .unwrap();
            assert!(dc_only.iter().all(|macroblock| {
                macroblock.dc_low_pass[1..].iter().all(|&value| value == 0)
                    && macroblock.high_pass.iter().all(|&value| value == 0)
            }));
            let no_high =
                analyze_component(samples, &config, UNIT, BandPresence::NoHighPass, cpu()).unwrap();
            assert!(
                no_high.iter().any(|macroblock| {
                    macroblock.dc_low_pass[1..].iter().any(|&value| value != 0)
                })
            );
            assert!(
                no_high
                    .iter()
                    .all(|macroblock| macroblock.high_pass.iter().all(|&value| value == 0))
            );
        }
    }

    #[test]
    fn mismatched_plane_size_is_rejected() {
        let config = config(OverlapMode::None, TilePartition::single(5, 3));
        let result = analyze_component(vec![0; 80 * 47], &config, UNIT, BandPresence::All, cpu());
        assert!(matches!(
            result,
            Err(ReconstructionError::InvalidPlaneGeometry(_))
        ));
        // A full-resolution plane is not a 4:2:0 chroma plane of the same grid.
        let chroma = shaped_config((2, 2), OverlapMode::None, TilePartition::single(5, 3));
        let result = analyze_component(vec![0; 80 * 48], &chroma, UNIT, BandPresence::All, cpu());
        assert!(matches!(
            result,
            Err(ReconstructionError::InvalidPlaneGeometry(_))
        ));
    }

    #[test]
    fn unsupported_macroblock_shapes_are_rejected() {
        for shape in [(4, 2), (1, 1), (0, 4), (2, 3)] {
            let config = shaped_config(shape, OverlapMode::None, TilePartition::single(5, 3));
            let samples = vec![0; usize::from(shape.0) * usize::from(shape.1) * 16 * 15];
            assert!(
                matches!(
                    analyze_component(samples, &config, UNIT, BandPresence::All, cpu()),
                    Err(ReconstructionError::InvalidPlaneGeometry(_))
                ),
                "{shape:?}"
            );
        }
    }

    #[test]
    fn invalid_tile_partitions_are_rejected() {
        let tiles = TilePartition {
            column_boundaries: vec![0, 3, 2, 5],
            row_boundaries: vec![0, 3],
            hard_boundaries: true,
        };
        for shape in SHAPES {
            let config = shaped_config(shape, OverlapMode::Two, tiles.clone());
            let samples = plane_for(18, &config);
            assert_eq!(
                analyze_component(samples, &config, UNIT, BandPresence::All, cpu()),
                Err(ReconstructionError::InvalidTilePartition("column")),
                "{shape:?}"
            );
        }
    }
}
