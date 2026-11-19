//! Ordered CPU execution of the T.832 clause 9.9 reconstruction stages.

use jxr_core::{DecodeScale, OverlapMode, QuantizerSet};
use jxr_math::quantization::Quantizer;

use crate::CpuCapabilities;

use super::{
    PlanarSamples, QuantizedMacroblock, ReconstructionConfig, ReconstructionError,
    lanes::{
        InverseTransform, PARALLEL_MACROBLOCKS, Postfilter, SimdReporter, apply_block, simd_used,
    },
    sample_scale::{BLOCK_DC_GAIN_BITS, ReducedPicture, SampleScale, macroblock_dc_gain_bits},
    strip::{ComponentStrip, RasterMacroblocks, ReconstructionSimd, StripPlan},
    transform::{inverse_chroma_420, inverse_chroma_422},
};

/// Retained per-component transform buffers and recyclable output plane.
#[derive(Debug, Default)]
pub(crate) struct ReconstructionPipelineWorkspace {
    /// Block DC coefficients of a reduced reconstruction.
    dc_low_pass: Vec<[i32; 16]>,
    /// The strip a full reconstruction runs down the plane.
    strip: ComponentStrip,
    output_samples: Vec<i32>,
    /// One geometry's calibrated reduced scale, reused with different coefficients.
    sample_scale: Option<(ReconstructionConfig, DecodeScale, SampleScale)>,
    simd: ReconstructionSimd,
    reuses: u64,
    output_reuses: u64,
}

impl ReconstructionPipelineWorkspace {
    pub(crate) const fn reuses(&self) -> u64 {
        self.reuses.saturating_add(self.output_reuses)
    }

    pub(crate) const fn simd(&self) -> ReconstructionSimd {
        self.simd
    }

    #[cfg(test)]
    pub(crate) const fn output_reuses(&self) -> u64 {
        self.output_reuses
    }

    pub(crate) fn retained_output_bytes(&self) -> usize {
        self.output_samples
            .capacity()
            .saturating_mul(core::mem::size_of::<i32>())
    }

    pub(crate) fn recycle_samples(&mut self, mut samples: Vec<i32>) {
        samples.clear();
        if samples.capacity() > self.output_samples.capacity() {
            self.output_samples = samples;
        }
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.dc_low_pass
            .capacity()
            .saturating_mul(core::mem::size_of::<[i32; 16]>())
            .saturating_add(self.strip.retained_bytes())
            .saturating_add(self.retained_output_bytes())
            .saturating_add(self.sample_scale.as_ref().map_or(0, |(config, _, scale)| {
                config
                    .tiles
                    .column_boundaries
                    .capacity()
                    .saturating_add(config.tiles.row_boundaries.capacity())
                    .saturating_mul(core::mem::size_of::<u32>())
                    .saturating_add(scale.retained_bytes())
            }))
    }

    fn take_output_samples(&mut self) -> Vec<i32> {
        core::mem::take(&mut self.output_samples)
    }

    fn scale_reduced_samples(
        &mut self,
        config: &ReconstructionConfig,
        resolution: DecodeScale,
        cpu: CpuCapabilities,
        picture: &ReducedPicture<'_>,
    ) -> Result<(), ReconstructionError> {
        let (gain_bits, units) = match resolution {
            DecodeScale::Quarter => (BLOCK_DC_GAIN_BITS, component_block_geometry(config)?),
            DecodeScale::Sixteenth => (
                macroblock_dc_gain_bits(config)? + BLOCK_DC_GAIN_BITS,
                (1, 1),
            ),
            DecodeScale::Full => {
                return Err(ReconstructionError::InvalidPlaneGeometry(
                    "full reconstruction has no reduced sample scale",
                ));
            }
        };
        if config.overlap == OverlapMode::None {
            return SampleScale::Shift(gain_bits).apply(cpu.executor(), &mut self.output_samples);
        }
        let matches = self
            .sample_scale
            .as_ref()
            .is_some_and(|(held, scale, _)| held == config && *scale == resolution);
        if !matches {
            let scale = SampleScale::new(config, gain_bits, units, cpu, picture)?;
            self.sample_scale = Some((config.clone(), resolution, scale));
        }
        self.sample_scale
            .as_ref()
            .expect("the reduced sample scale is initialized")
            .2
            .apply(cpu.executor(), &mut self.output_samples)
    }
}

/// Reconstruct one full-resolution component from quantized coefficients.
///
/// The returned samples are signed transform-domain results. Output bias,
/// post-scaling, and colour conversion must be completed before packing.
pub fn reconstruct_luma(
    macroblocks: &[QuantizedMacroblock],
    config: &ReconstructionConfig,
) -> Result<PlanarSamples, ReconstructionError> {
    reconstruct_luma_impl(
        macroblocks,
        config,
        None,
        &mut ReconstructionPipelineWorkspace::default(),
    )
}

#[cfg(test)]
pub(crate) fn reconstruct_luma_scaled(
    macroblocks: &[QuantizedMacroblock],
    config: &ReconstructionConfig,
    scale: DecodeScale,
) -> Result<PlanarSamples, ReconstructionError> {
    reconstruct_luma_scaled_impl(
        macroblocks,
        config,
        scale,
        None,
        &mut ReconstructionPipelineWorkspace::default(),
    )
}

pub(crate) fn reconstruct_luma_scaled_with_cpu(
    macroblocks: &[QuantizedMacroblock],
    config: &ReconstructionConfig,
    scale: DecodeScale,
    cpu: CpuCapabilities,
    workspace: &mut ReconstructionPipelineWorkspace,
) -> Result<PlanarSamples, ReconstructionError> {
    reconstruct_luma_scaled_impl(macroblocks, config, scale, Some(cpu), workspace)
}

fn reconstruct_luma_scaled_impl(
    macroblocks: &[QuantizedMacroblock],
    config: &ReconstructionConfig,
    scale: DecodeScale,
    cpu: Option<CpuCapabilities>,
    workspace: &mut ReconstructionPipelineWorkspace,
) -> Result<PlanarSamples, ReconstructionError> {
    workspace.simd = ReconstructionSimd::default();
    match scale {
        DecodeScale::Full => reconstruct_luma_impl(macroblocks, config, cpu, workspace),
        DecodeScale::Quarter => reconstruct_quarter(macroblocks, config, cpu, workspace),
        DecodeScale::Sixteenth => reconstruct_sixteenth(macroblocks, config, cpu, workspace),
    }
}

fn reconstruct_quarter(
    macroblocks: &[QuantizedMacroblock],
    config: &ReconstructionConfig,
    cpu: Option<CpuCapabilities>,
    workspace: &mut ReconstructionPipelineWorkspace,
) -> Result<PlanarSamples, ReconstructionError> {
    let cpu = cpu.unwrap_or_else(CpuCapabilities::detect);
    let reused = workspace.dc_low_pass.capacity() >= macroblocks.len();
    let output_capacity = workspace.output_samples.capacity();
    let bands = LowBands::DcAndLowPass;
    let (width, height) = block_dc_plane(macroblocks, config, bands, cpu, workspace)?;
    let output_reused = output_capacity >= width.saturating_mul(height);
    // One sample stands for a block: its DC, less the gain of the block
    // transform and of the overlap filtering that follows it, neither of
    // which was run.
    workspace.scale_reduced_samples(config, DecodeScale::Quarter, cpu, &|flat, flat_config| {
        let mut workspace = ReconstructionPipelineWorkspace::default();
        block_dc_plane(flat, flat_config, bands, cpu, &mut workspace)?;
        Ok(workspace.take_output_samples())
    })?;
    let samples = workspace.take_output_samples();
    workspace.reuses = workspace.reuses.saturating_add(u64::from(reused));
    workspace.output_reuses = workspace
        .output_reuses
        .saturating_add(u64::from(output_reused));
    Ok(PlanarSamples {
        origin_x: config
            .macroblock_origin_x
            .checked_mul(u32::from(config.block_columns))
            .ok_or(ReconstructionError::ArithmeticOverflow(
                "quarter-resolution x origin",
            ))?,
        origin_y: config
            .macroblock_origin_y
            .checked_mul(u32::from(config.block_rows))
            .ok_or(ReconstructionError::ArithmeticOverflow(
                "quarter-resolution y origin",
            ))?,
        width: u32::try_from(width).map_err(|_| {
            ReconstructionError::ArithmeticOverflow("quarter-resolution width conversion")
        })?,
        height: u32::try_from(height).map_err(|_| {
            ReconstructionError::ArithmeticOverflow("quarter-resolution height conversion")
        })?,
        samples,
    })
}

fn reconstruct_sixteenth(
    macroblocks: &[QuantizedMacroblock],
    config: &ReconstructionConfig,
    cpu: Option<CpuCapabilities>,
    workspace: &mut ReconstructionPipelineWorkspace,
) -> Result<PlanarSamples, ReconstructionError> {
    let cpu = cpu.unwrap_or_else(CpuCapabilities::detect);
    let output_reused = workspace.output_samples.capacity() >= macroblocks.len();
    let (width, height) = macroblock_dc_plane(macroblocks, config, cpu, workspace)?;
    // One sample stands for a macroblock: its DC, less the gain of both
    // transform stages and of the overlap filtering after the second,
    // none of which was run.
    workspace.scale_reduced_samples(
        config,
        DecodeScale::Sixteenth,
        cpu,
        &|flat, flat_config| {
            let mut workspace = ReconstructionPipelineWorkspace::default();
            macroblock_dc_plane(flat, flat_config, cpu, &mut workspace)?;
            Ok(workspace.take_output_samples())
        },
    )?;
    let samples = workspace.take_output_samples();
    workspace.output_reuses = workspace
        .output_reuses
        .saturating_add(u64::from(output_reused));
    Ok(PlanarSamples {
        origin_x: config.macroblock_origin_x,
        origin_y: config.macroblock_origin_y,
        width: u32::try_from(width).map_err(|_| {
            ReconstructionError::ArithmeticOverflow("sixteenth-resolution width conversion")
        })?,
        height: u32::try_from(height).map_err(|_| {
            ReconstructionError::ArithmeticOverflow("sixteenth-resolution height conversion")
        })?,
        samples,
    })
}

fn validated_geometry(
    macroblocks: &[QuantizedMacroblock],
    config: &ReconstructionConfig,
) -> Result<(usize, usize, usize, usize), ReconstructionError> {
    validated_geometry_for(macroblocks.len(), config)
}

fn validated_geometry_for(
    count: usize,
    config: &ReconstructionConfig,
) -> Result<(usize, usize, usize, usize), ReconstructionError> {
    let macroblocks_x = usize::try_from(config.macroblocks_x)
        .map_err(|_| ReconstructionError::ArithmeticOverflow("macroblock width conversion"))?;
    let macroblocks_y = usize::try_from(config.macroblocks_y)
        .map_err(|_| ReconstructionError::ArithmeticOverflow("macroblock height conversion"))?;
    let expected = macroblocks_x
        .checked_mul(macroblocks_y)
        .ok_or(ReconstructionError::ArithmeticOverflow("macroblock count"))?;
    if count != expected {
        return Err(ReconstructionError::MacroblockCount {
            expected,
            actual: count,
        });
    }
    let (block_columns, block_rows) = validated_block_geometry(config)?;
    Ok((macroblocks_x, macroblocks_y, block_columns, block_rows))
}

/// Checks the tile partition of the plane `config` describes, and returns
/// the blocks across and down one of its macroblocks.
pub(crate) fn validated_block_geometry(
    config: &ReconstructionConfig,
) -> Result<(usize, usize), ReconstructionError> {
    validate_partition(
        &config.tiles.column_boundaries,
        config.macroblocks_x,
        "column",
    )?;
    validate_partition(&config.tiles.row_boundaries, config.macroblocks_y, "row")?;
    component_block_geometry(config)
}

/// Full-resolution reconstruction: the strip that covers the plane and is
/// never released, so that its window ends as the plane itself.
fn reconstruct_luma_impl(
    macroblocks: &[QuantizedMacroblock],
    config: &ReconstructionConfig,
    cpu: Option<CpuCapabilities>,
    workspace: &mut ReconstructionPipelineWorkspace,
) -> Result<PlanarSamples, ReconstructionError> {
    workspace.simd = ReconstructionSimd::default();
    let (macroblocks_x, macroblocks_y, block_columns, block_rows) =
        validated_geometry(macroblocks, config)?;
    let width = macroblocks_x.checked_mul(block_columns * 4).ok_or(
        ReconstructionError::ArithmeticOverflow("reconstructed width"),
    )?;
    let height = macroblocks_y.checked_mul(block_rows * 4).ok_or(
        ReconstructionError::ArithmeticOverflow("reconstructed height"),
    )?;
    let output_length =
        width
            .checked_mul(height)
            .ok_or(ReconstructionError::ArithmeticOverflow(
                "reconstructed plane size",
            ))?;
    let reused = workspace.strip.retained_bytes() > 0;
    let output_reused = workspace.output_samples.capacity() >= output_length;
    let strip = &mut workspace.strip;
    strip.begin(config, cpu, 0, macroblocks_y)?;
    strip.reserve_plane(
        core::mem::take(&mut workspace.output_samples),
        macroblocks_y,
    );
    let source = RasterMacroblocks {
        macroblocks,
        width: macroblocks_x,
    };
    strip.push_rows(&source, 0..macroblocks_y)?;
    let simd = strip.simd();
    let samples = strip.take_samples();
    workspace.simd = simd;
    workspace.reuses = workspace.reuses.saturating_add(u64::from(reused));
    workspace.output_reuses = workspace
        .output_reuses
        .saturating_add(u64::from(output_reused));
    Ok(PlanarSamples {
        origin_x: config
            .macroblock_origin_x
            .checked_mul(u32::from(config.block_columns) * 4)
            .ok_or(ReconstructionError::ArithmeticOverflow("output x origin"))?,
        origin_y: config
            .macroblock_origin_y
            .checked_mul(u32::from(config.block_rows) * 4)
            .ok_or(ReconstructionError::ArithmeticOverflow("output y origin"))?,
        width: u32::try_from(width)
            .map_err(|_| ReconstructionError::ArithmeticOverflow("output width conversion"))?,
        height: u32::try_from(height)
            .map_err(|_| ReconstructionError::ArithmeticOverflow("output height conversion"))?,
        samples,
    })
}

/// The coefficients of a macroblock that a plane of block DC coefficients
/// is made from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LowBands {
    /// Both bands, as full reconstruction takes them.
    DcAndLowPass,
    /// The DC band alone, as if the low-pass band were zero: all that a
    /// sixteenth-size decode has read.
    Dc,
}

/// Fills the output plane of `workspace` with one value for each
/// macroblock, made of its DC coefficient alone, and returns the size of
/// the plane. A sixteenth-size picture is this plane, scaled.
///
/// The low-pass-level overlap filter, which only [`OverlapMode::Two`] has,
/// reaches half a macroblock into each neighbour, so it is not left to the
/// scale: the DC coefficients are spread over their blocks, the filter is
/// run over the plane as [`block_dc_plane`] runs it, and each macroblock
/// takes the sum of its blocks. What the plane holds there is the
/// quarter-size picture of the DC band, a macroblock at a time. Elsewhere it
/// holds the DC coefficient itself, at the weight the first-level transform
/// would give it.
fn macroblock_dc_plane(
    macroblocks: &[QuantizedMacroblock],
    config: &ReconstructionConfig,
    cpu: CpuCapabilities,
    workspace: &mut ReconstructionPipelineWorkspace,
) -> Result<(usize, usize), ReconstructionError> {
    if config.overlap != OverlapMode::Two {
        let (macroblocks_x, macroblocks_y, _, _) = validated_geometry(macroblocks, config)?;
        workspace.output_samples.clear();
        for macroblock in macroblocks {
            let dc = dequantize_dc(macroblock)?;
            let dc = if config.scale_after_first_transform {
                dc.checked_mul(2)
                    .ok_or(ReconstructionError::ArithmeticOverflow(
                        "sixteenth-resolution chroma normalization",
                    ))?
            } else {
                dc
            };
            workspace.output_samples.push(dc);
        }
        return Ok((macroblocks_x, macroblocks_y));
    }
    let (width, height) = block_dc_plane(macroblocks, config, LowBands::Dc, cpu, workspace)?;
    let (block_columns, block_rows) = component_block_geometry(config)?;
    sum_macroblocks(
        &mut workspace.output_samples,
        width,
        block_columns,
        block_rows,
    )?;
    Ok((width / block_columns, height / block_rows))
}

/// Replaces a plane of one value for each block by the sum of the blocks of
/// each macroblock, in macroblock raster order.
fn sum_macroblocks(
    plane: &mut Vec<i32>,
    width: usize,
    block_columns: usize,
    block_rows: usize,
) -> Result<(), ReconstructionError> {
    let macroblocks_x = width / block_columns;
    let macroblocks_y = plane.len() / (width * block_rows);
    // A macroblock is written at or before the first of its own blocks, so
    // no block is overwritten before it is summed.
    for macroblock_y in 0..macroblocks_y {
        for macroblock_x in 0..macroblocks_x {
            let mut sum = 0_i32;
            for row in 0..block_rows {
                let start =
                    (macroblock_y * block_rows + row) * width + macroblock_x * block_columns;
                for &block in &plane[start..start + block_columns] {
                    sum = sum
                        .checked_add(block)
                        .ok_or(ReconstructionError::ArithmeticOverflow(
                            "sixteenth-resolution macroblock sum",
                        ))?;
                }
            }
            plane[macroblock_y * macroblocks_x + macroblock_x] = sum;
        }
    }
    plane.truncate(macroblocks_x * macroblocks_y);
    Ok(())
}

/// Fills the output plane of `workspace` with the DC coefficient of every
/// block, as full reconstruction of the `bands` of each macroblock holds
/// them before its block transform: the first-level inverse transform of
/// each macroblock, and the overlap filter of that level over the whole
/// plane. Returns the size of the plane.
fn block_dc_plane(
    macroblocks: &[QuantizedMacroblock],
    config: &ReconstructionConfig,
    bands: LowBands,
    cpu: CpuCapabilities,
    workspace: &mut ReconstructionPipelineWorkspace,
) -> Result<(usize, usize), ReconstructionError> {
    let (macroblocks_x, macroblocks_y, block_columns, block_rows) =
        validated_geometry(macroblocks, config)?;
    let block_count = block_columns * block_rows;
    workspace.dc_low_pass.resize(macroblocks.len(), [0; 16]);
    cpu.executor()
        .if_worth(macroblocks.len() >= PARALLEL_MACROBLOCKS)
        .try_for_each(&mut workspace.dc_low_pass, |index, low| {
            block_dc_into(&macroblocks[index], block_count, bands, config, low)
        })?;
    let width =
        macroblocks_x
            .checked_mul(block_columns)
            .ok_or(ReconstructionError::ArithmeticOverflow(
                "quarter-resolution width",
            ))?;
    let height =
        macroblocks_y
            .checked_mul(block_rows)
            .ok_or(ReconstructionError::ArithmeticOverflow(
                "quarter-resolution height",
            ))?;
    combine_low_pass_into(
        &workspace.dc_low_pass,
        macroblocks_x,
        macroblocks_y,
        block_columns,
        block_rows,
        &mut workspace.output_samples,
    )?;
    if config.overlap == OverlapMode::Two {
        // The overlap filter of the low-pass level, which only second-level
        // overlap has: the strip that covers the plane.
        let plan = StripPlan::new(config, Some(cpu))?;
        let used_simd = SimdReporter::new(false);
        plan.filter_low::<Postfilter>(
            &mut workspace.output_samples,
            0..height,
            cpu.executor(),
            Some(&used_simd),
        )?;
        workspace.simd.first_overlap |= simd_used(&used_simd);
    }
    Ok((width, height))
}

/// The DC coefficient of each block of one macroblock: the first-level
/// inverse transform of its DC coefficient and, where `bands` has them, its
/// low-pass coefficients.
fn block_dc_into(
    macroblock: &QuantizedMacroblock,
    block_count: usize,
    bands: LowBands,
    config: &ReconstructionConfig,
    low: &mut [i32; 16],
) -> Result<(), ReconstructionError> {
    *low = match bands {
        LowBands::DcAndLowPass => dequantize_low(macroblock, block_count)?,
        LowBands::Dc => {
            let mut low = [0; 16];
            low[0] = dequantize_dc(macroblock)?;
            low
        }
    };
    inverse_first_level(low, config)
}

/// Dequantizes the high-pass blocks of `macroblock`, `block_count` blocks
/// of which `block_columns` are across, straight into their places in
/// `rows`, the sample rows of its macroblock row, `width` samples across,
/// with its first block in block column `column`: zero where it codes no
/// high-pass band. Element zero of each block, which its block DC
/// coefficient replaces, is placed as zero.
///
/// The products wrap, and a coefficient that the checked
/// [`Quantizer::dequantize`] cannot scale fails the macroblock with its
/// error once every block is placed: the macroblock's samples are then
/// never read.
pub(super) fn dequantize_high_pass_into(
    macroblock: &QuantizedMacroblock,
    block_columns: usize,
    block_count: usize,
    rows: &mut [i32],
    width: usize,
    column: usize,
) -> Result<(), ReconstructionError> {
    let quantizer = quantizer(macroblock.quantizers.high_pass, "high-pass")?;
    let blocks = &macroblock.high_pass.as_chunks::<16>().0[..block_count];
    let coded = macroblock.bands.has_high_pass();
    // The product of a step beyond `i32` wraps to the exact product where
    // that is within `i32`, as for any other step.
    let multiplier = i32::from_ne_bytes(quantizer.step().to_ne_bytes());
    // Each value with its bits inverted when negative is its magnitude,
    // less one for a negative value: their OR bounds every magnitude.
    let mut magnitude = 0_i32;
    for (block, coefficients) in blocks.iter().enumerate() {
        let origin = block / block_columns * 4 * width + (column + block % block_columns) * 4;
        let mut values = if coded { *coefficients } else { [0; 16] };
        values[0] = 0;
        for (line, values) in values.as_chunks::<4>().0.iter().enumerate() {
            let start = origin + line * width;
            let samples: &mut [i32; 4] = (&mut rows[start..start + 4])
                .try_into()
                .expect("a block line spans four samples");
            for (sample, &value) in samples.iter_mut().zip(values) {
                magnitude |= value ^ (value >> 31);
                *sample = value.wrapping_mul(multiplier);
            }
        }
    }
    // Below `i32::MAX / step`, every magnitude scales within `i32`.
    if multiplier > 0 && magnitude < i32::MAX / multiplier {
        return Ok(());
    }
    // A coefficient near the bound, or a step beyond `i32`: the oracle.
    for coefficients in blocks.iter().filter(|_| coded) {
        for &coefficient in &coefficients[1..] {
            scale(quantizer, coefficient)?;
        }
    }
    Ok(())
}

/// Places the block DC coefficients `low` of one macroblock, `COLUMNS`
/// blocks across, in `low_rows`, `low_width` samples across, from column
/// `column`. A copy of a fixed size is a few loads and stores where a copy
/// of a size known only when it runs is a call in WebAssembly without bulk
/// memory.
pub(super) fn place_low<const COLUMNS: usize>(
    low: &[i32],
    low_rows: &mut [i32],
    low_width: usize,
    column: usize,
) {
    for (block_row, values) in low.as_chunks::<COLUMNS>().0.iter().enumerate() {
        let start = block_row * low_width + column;
        low_rows[start..start + COLUMNS].copy_from_slice(values);
    }
}

pub(super) fn dequantize_low(
    macroblock: &QuantizedMacroblock,
    block_count: usize,
) -> Result<[i32; 16], ReconstructionError> {
    let QuantizerSet { dc, low_pass, .. } = macroblock.quantizers;
    let dc = quantizer(dc, "DC")?;
    let low_pass = quantizer(low_pass, "low-pass")?;
    let mut low = [0; 16];
    low[0] = scale(dc, macroblock.dc_low_pass[0])?;
    if macroblock.bands.has_low_pass() {
        for (output, &coefficient) in low[1..block_count]
            .iter_mut()
            .zip(&macroblock.dc_low_pass[1..block_count])
        {
            *output = scale(low_pass, coefficient)?;
        }
    }
    Ok(low)
}

/// The DC coefficient of a macroblock alone, as [`dequantize_low`] scales it.
fn dequantize_dc(macroblock: &QuantizedMacroblock) -> Result<i32, ReconstructionError> {
    let quantizer = quantizer(macroblock.quantizers.dc, "DC")?;
    scale(quantizer, macroblock.dc_low_pass[0])
}

pub(crate) fn component_block_geometry(
    config: &ReconstructionConfig,
) -> Result<(usize, usize), ReconstructionError> {
    let columns = usize::from(config.block_columns);
    let rows = usize::from(config.block_rows);
    match (columns, rows) {
        (4 | 2, 4) | (2, 2) => Ok((columns, rows)),
        _ => Err(ReconstructionError::InvalidPlaneGeometry(
            "unsupported component macroblock shape",
        )),
    }
}

pub(super) fn inverse_first_level(
    low: &mut [i32; 16],
    config: &ReconstructionConfig,
) -> Result<(), ReconstructionError> {
    match (config.block_columns, config.block_rows) {
        (4, 4) => apply_block::<InverseTransform>(low)?,
        (2, 4) => {
            let mut coefficients = [0; 8];
            coefficients.copy_from_slice(&low[..8]);
            inverse_chroma_422(&mut coefficients)?;
            low[..8].copy_from_slice(&coefficients);
        }
        (2, 2) => {
            let mut coefficients = [0; 4];
            coefficients.copy_from_slice(&low[..4]);
            inverse_chroma_420(&mut coefficients)?;
            low[..4].copy_from_slice(&coefficients);
        }
        _ => {
            return Err(ReconstructionError::InvalidPlaneGeometry(
                "unsupported first-level transform shape",
            ));
        }
    }
    if config.scale_after_first_transform {
        let count = usize::from(config.block_columns) * usize::from(config.block_rows);
        for coefficient in &mut low[..count] {
            *coefficient =
                coefficient
                    .checked_mul(2)
                    .ok_or(ReconstructionError::ArithmeticOverflow(
                        "scaled chroma normalization",
                    ))?;
        }
    }
    Ok(())
}

fn quantizer(step: u32, band: &'static str) -> Result<Quantizer, ReconstructionError> {
    Quantizer::new(step).ok_or(ReconstructionError::ZeroQuantizer(band))
}

fn scale(quantizer: Quantizer, coefficient: i32) -> Result<i32, ReconstructionError> {
    quantizer
        .dequantize(coefficient)
        .map_err(|_| ReconstructionError::ArithmeticOverflow("coefficient dequantization"))
}

pub(crate) fn combine_low_pass_into(
    macroblocks: &[[i32; 16]],
    macroblocks_x: usize,
    macroblocks_y: usize,
    block_columns: usize,
    block_rows: usize,
    plane: &mut Vec<i32>,
) -> Result<(), ReconstructionError> {
    let width = macroblocks_x
        .checked_mul(block_columns)
        .ok_or(ReconstructionError::ArithmeticOverflow("low-pass width"))?;
    let length = width
        .checked_mul(macroblocks_y)
        .and_then(|value| value.checked_mul(block_rows))
        .ok_or(ReconstructionError::ArithmeticOverflow(
            "low-pass plane size",
        ))?;
    plane.clear();
    plane.resize(length, 0);
    for macroblock_y in 0..macroblocks_y {
        for macroblock_x in 0..macroblocks_x {
            let block = &macroblocks[macroblock_y * macroblocks_x + macroblock_x];
            for row in 0..block_rows {
                let destination =
                    (macroblock_y * block_rows + row) * width + macroblock_x * block_columns;
                plane[destination..destination + block_columns].copy_from_slice(
                    &block[row * block_columns..row * block_columns + block_columns],
                );
            }
        }
    }
    Ok(())
}

pub(crate) fn scale_boundaries(
    boundaries: &[u32],
    scale: usize,
) -> Result<Vec<usize>, ReconstructionError> {
    boundaries
        .iter()
        .map(|&value| {
            usize::try_from(value)
                .ok()
                .and_then(|value| value.checked_mul(scale))
                .ok_or(ReconstructionError::ArithmeticOverflow(
                    "tile boundary scaling",
                ))
        })
        .collect()
}

fn validate_partition(
    boundaries: &[u32],
    extent: u32,
    axis: &'static str,
) -> Result<(), ReconstructionError> {
    if boundaries.len() < 2
        || boundaries.first() != Some(&0)
        || boundaries.last() != Some(&extent)
        || boundaries.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(ReconstructionError::InvalidTilePartition(axis));
    }
    Ok(())
}
