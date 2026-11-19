//! One component reconstructed a macroblock row at a time.
//!
//! [`ComponentStrip`] is the T.832 clause 9.9 pipeline as a stream. It is
//! given the quantized macroblocks of a component one macroblock row after
//! another, and holds only the rows whose samples are not yet settled: the
//! overlap filters of both levels reach across macroblock rows, so the
//! samples of a row are finished a little after the row arrives. A caller
//! reads the finished rows, formats or copies them, and releases them.
//!
//! The whole plane is the strip that is never released, and that is how
//! [`reconstruct_luma`](super::reconstruct_luma) runs. A strip may equally
//! begin and end part way down a plane, one macroblock row beyond the rows
//! its caller is after on either side: the windows that reach outside the
//! strip are left out, and what they would have changed lies outside the
//! rows the caller is after. Rows of a plane can therefore be reconstructed
//! by several strips at once and come out as the plane-wide sweep leaves
//! them.

use core::ops::Range;

use fearless_simd::Level;
use jxr_core::OverlapMode;

use crate::{CpuCapabilities, executor::Executor};

use super::{
    QuantizedMacroblock, ReconstructionConfig, ReconstructionError,
    lanes::{InverseTransform, Postfilter, SimdReporter, simd_used, transform_rows},
    overlap::{OverlapKernel, OverlapPlane, apply_overlap_rows},
    pipeline::{
        dequantize_high_pass_into, dequantize_low, inverse_first_level, place_low,
        scale_boundaries, validated_block_geometry,
    },
    subsampled_overlap::{SubsampledOverlapKernel, apply_subsampled_first_overlap_rows},
};

/// Quantized macroblocks of one component, read by position in its plane.
///
/// Slices serve stored macroblocks; the CPU decoder unpacks each one from
/// packed coefficient storage on demand, into storage its reader reuses
/// from one macroblock to the next, so that no copy of a row in this form
/// is retained and none is made per macroblock.
pub(crate) trait MacroblockSource {
    /// The macroblock in column `x` of macroblock row `y`: one the source
    /// holds, or one it unpacks into `scratch`.
    ///
    /// A reader passes the same `scratch` for every macroblock of a
    /// component, so a source unpacking into it writes every coefficient
    /// the macroblock's bands and block count place, as unpacking into
    /// zeroed storage would; the coefficients of blocks beyond its block
    /// count it may leave as they are.
    fn macroblock<'a>(
        &'a self,
        x: usize,
        y: usize,
        scratch: &'a mut QuantizedMacroblock,
    ) -> Result<&'a QuantizedMacroblock, ReconstructionError>;
}

/// Raster-ordered macroblocks of a plane `width` macroblocks wide.
pub(crate) struct RasterMacroblocks<'a> {
    pub(crate) macroblocks: &'a [QuantizedMacroblock],
    pub(crate) width: usize,
}

impl MacroblockSource for RasterMacroblocks<'_> {
    #[inline]
    fn macroblock<'a>(
        &'a self,
        x: usize,
        y: usize,
        _scratch: &'a mut QuantizedMacroblock,
    ) -> Result<&'a QuantizedMacroblock, ReconstructionError> {
        Ok(&self.macroblocks[y * self.width + x])
    }
}

/// A run of consecutive rows of a plane that slides down it.
///
/// Rows are appended at the bottom and discarded from the top. Discarded
/// rows are reclaimed only when the storage has no room left below, and
/// then the rows still held are moved to its start, so storage with a few
/// times the rows that are held at once is moved a fraction of a row for
/// each row appended.
#[derive(Debug, Default)]
pub(crate) struct RowWindow {
    samples: Vec<i32>,
    width: usize,
    /// Row of the plane the first held row is.
    first: usize,
    /// Rows held.
    rows: usize,
    /// Discarded rows of storage above the first held row.
    discarded: usize,
    /// Rows of room to add whenever the storage has to grow.
    slack: usize,
}

impl RowWindow {
    /// Empties the window and places it at row `first` of a plane `width`
    /// samples wide. Storage is retained.
    pub(crate) fn begin(&mut self, width: usize, first: usize, slack: usize) {
        self.width = width;
        self.first = first;
        self.rows = 0;
        self.discarded = 0;
        self.slack = slack;
    }

    /// Row of the plane after the last held row.
    pub(crate) const fn end(&self) -> usize {
        self.first + self.rows
    }

    /// Samples in a row.
    pub(crate) const fn width(&self) -> usize {
        self.width
    }

    /// Adds `rows` rows below those held and returns them. Their contents
    /// are unspecified: the caller writes every sample.
    pub(crate) fn append(&mut self, rows: usize) -> &mut [i32] {
        let held = self.rows * self.width;
        let added = rows * self.width;
        if (self.discarded + self.rows + rows) * self.width > self.samples.len() {
            if self.discarded > 0 {
                let start = self.discarded * self.width;
                self.samples.copy_within(start..start + held, 0);
                self.discarded = 0;
            }
            if held + added > self.samples.len() {
                self.samples
                    .resize(held + added + self.slack * self.width, 0);
            }
        }
        let start = self.discarded * self.width + held;
        self.rows += rows;
        &mut self.samples[start..start + added]
    }

    /// The rows `rows` of the plane, which must be held.
    pub(crate) fn rows(&self, rows: Range<usize>) -> &[i32] {
        let start = (self.discarded + rows.start - self.first) * self.width;
        &self.samples[start..start + rows.len() * self.width]
    }

    /// The rows `rows` of the plane, which must be held.
    pub(crate) fn rows_mut(&mut self, rows: Range<usize>) -> &mut [i32] {
        let start = (self.discarded + rows.start - self.first) * self.width;
        &mut self.samples[start..start + rows.len() * self.width]
    }

    /// Discards the rows above `row` of the plane.
    pub(crate) fn discard_before(&mut self, row: usize) {
        let row = row.clamp(self.first, self.end());
        let discarded = row - self.first;
        self.first = row;
        self.rows -= discarded;
        self.discarded = if self.rows == 0 {
            0
        } else {
            self.discarded + discarded
        };
    }

    /// Makes room for `rows` rows in all, so that appending them does not
    /// move those already held.
    pub(crate) fn reserve_rows(&mut self, rows: usize) {
        let length = rows * self.width;
        if self.samples.len() < length {
            self.samples.resize(length, 0);
        }
    }

    /// Replaces the storage with `storage`, whose contents are not kept.
    pub(crate) fn adopt(&mut self, storage: Vec<i32>) {
        self.samples = storage;
        self.rows = 0;
        self.discarded = 0;
    }

    /// Takes the held rows out of the window, which is left without storage.
    pub(crate) fn take(&mut self) -> Vec<i32> {
        let held = self.rows * self.width;
        if self.discarded > 0 {
            let start = self.discarded * self.width;
            self.samples.copy_within(start..start + held, 0);
        }
        self.samples.truncate(held);
        self.rows = 0;
        self.discarded = 0;
        core::mem::take(&mut self.samples)
    }

    /// Gives back storage that [`Self::take`] took, with rows written below
    /// those it held: the window holds every whole row of it, from the row
    /// it held first. A caller that fills rows in storage of its own, as
    /// the encoder's source reader does, appends them so without a copy.
    pub(crate) fn put_back(&mut self, samples: Vec<i32>) {
        self.rows = samples.len().checked_div(self.width).unwrap_or(0);
        self.discarded = 0;
        self.samples = samples;
    }

    /// Rows of the plane held.
    pub(crate) const fn held(&self) -> Range<usize> {
        self.first..self.first + self.rows
    }

    /// Bytes of storage retained.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.samples
            .capacity()
            .saturating_mul(core::mem::size_of::<i32>())
    }
}

/// The geometry of the plane a strip runs down, resolved once: what the
/// strips of both directions, reconstruction's and analysis's, filter and
/// transform over.
#[derive(Debug)]
pub(crate) struct StripPlan {
    pub(crate) config: ReconstructionConfig,
    pub(crate) level: Level,
    pub(crate) macroblocks_x: usize,
    pub(crate) block_columns: usize,
    pub(crate) block_rows: usize,
    /// Tile boundaries of the plane of block DC coefficients.
    low_columns: Vec<usize>,
    low_rows: Vec<usize>,
    /// Tile boundaries of the plane of samples.
    sample_columns: Vec<usize>,
    sample_rows: Vec<usize>,
}

impl StripPlan {
    /// The plan of the plane `config` describes, whose filters run their
    /// vector forms at the level `cpu` selects, or the level detected here.
    pub(crate) fn new(
        config: &ReconstructionConfig,
        cpu: Option<CpuCapabilities>,
    ) -> Result<Self, ReconstructionError> {
        let (block_columns, block_rows) = validated_block_geometry(config)?;
        let macroblocks_x = usize::try_from(config.macroblocks_x)
            .map_err(|_| ReconstructionError::ArithmeticOverflow("macroblock width conversion"))?;
        macroblocks_x
            .checked_mul(block_columns)
            .and_then(|low_width| low_width.checked_mul(4))
            .ok_or(ReconstructionError::ArithmeticOverflow(
                "reconstructed width",
            ))?;
        Ok(Self {
            level: cpu.map_or_else(Level::new, CpuCapabilities::level),
            macroblocks_x,
            block_columns,
            block_rows,
            low_columns: scale_boundaries(&config.tiles.column_boundaries, block_columns)?,
            low_rows: scale_boundaries(&config.tiles.row_boundaries, block_rows)?,
            sample_columns: scale_boundaries(&config.tiles.column_boundaries, block_columns * 4)?,
            sample_rows: scale_boundaries(&config.tiles.row_boundaries, block_rows * 4)?,
            config: config.clone(),
        })
    }

    /// Samples across the plane of block DC coefficients.
    pub(crate) const fn low_width(&self) -> usize {
        self.macroblocks_x * self.block_columns
    }

    /// Samples across the plane.
    pub(crate) const fn width(&self) -> usize {
        self.low_width() * 4
    }

    /// Sample rows of one macroblock row.
    pub(crate) const fn macroblock_height(&self) -> usize {
        self.block_rows * 4
    }

    /// The overlap filter of the low-pass level over the rows `held` of the
    /// plane of block DC coefficients, which `low` holds: the four-by-four
    /// windows of full-resolution components or the two-by-two ones of
    /// subsampled chroma, run with the operators of `K`, the former through
    /// `executor`. Returns the row above which the plane is filtered, as the
    /// strip entries do.
    pub(crate) fn filter_low<K>(
        &self,
        low: &mut [i32],
        held: Range<usize>,
        executor: Executor,
        used_simd: Option<&SimdReporter>,
    ) -> Result<usize, ReconstructionError>
    where
        K: OverlapKernel + SubsampledOverlapKernel,
    {
        if self.block_columns == 4 {
            apply_overlap_rows::<K>(
                self.level,
                low,
                held,
                &self.low_plane(),
                executor,
                used_simd,
            )
        } else {
            apply_subsampled_first_overlap_rows::<K>(low, held, &self.low_plane())
        }
    }

    /// The overlap filter of the samples over the rows `held` of the plane,
    /// which `samples` holds, run with the operators of `K` through
    /// `executor`. Returns the row above which the plane is filtered.
    pub(crate) fn filter_samples<K: OverlapKernel>(
        &self,
        samples: &mut [i32],
        held: Range<usize>,
        executor: Executor,
        used_simd: Option<&SimdReporter>,
    ) -> Result<usize, ReconstructionError> {
        apply_overlap_rows::<K>(
            self.level,
            samples,
            held,
            &self.sample_plane(),
            executor,
            used_simd,
        )
    }

    fn low_plane(&self) -> OverlapPlane<'_> {
        OverlapPlane {
            width: self.low_width(),
            height: self.low_rows.last().copied().unwrap_or(0),
            column_boundaries: &self.low_columns,
            row_boundaries: &self.low_rows,
            hard_boundaries: self.config.tiles.hard_boundaries,
        }
    }

    fn sample_plane(&self) -> OverlapPlane<'_> {
        OverlapPlane {
            width: self.low_width() * 4,
            height: self.sample_rows.last().copied().unwrap_or(0),
            column_boundaries: &self.sample_columns,
            row_boundaries: &self.sample_rows,
            hard_boundaries: self.config.tiles.hard_boundaries,
        }
    }
}

/// Reconstruction stages that used at least one native vector group.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReconstructionSimd {
    pub(crate) first_overlap: bool,
    pub(crate) second_inverse: bool,
    pub(crate) second_overlap: bool,
}

impl ReconstructionSimd {
    pub(crate) fn include(&mut self, other: Self) {
        self.first_overlap |= other.first_overlap;
        self.second_inverse |= other.second_inverse;
        self.second_overlap |= other.second_overlap;
    }
}

/// The reconstruction of one component, streamed down its plane.
///
/// Rows are counted from the top of the plane the strip was begun with:
/// macroblock rows going in, sample rows coming out.
#[derive(Debug, Default)]
pub(crate) struct ComponentStrip {
    plan: Option<StripPlan>,
    /// Block DC coefficients of the macroblock rows not yet transformed.
    low: RowWindow,
    /// Rows of `low` above this are as the low-pass-level overlap filter
    /// leaves them.
    low_finished: usize,
    /// Samples: coefficients in block layout until a row is transformed.
    samples: RowWindow,
    /// Rows of `samples` above this are finished.
    finished: usize,
    /// Macroblock rows whose coefficients have been placed.
    placed: usize,
    /// Macroblock rows whose blocks have been transformed.
    transformed: usize,
    simd: ReconstructionSimd,
}

impl ComponentStrip {
    /// Begins a strip at macroblock row `first_row` of the plane `config`
    /// describes. Storage is retained from the last strip.
    ///
    /// `held_rows` is how many macroblock rows of samples the caller lets
    /// build up before it releases any, which sizes the storage so that
    /// sliding the window moves few samples.
    pub(crate) fn begin(
        &mut self,
        config: &ReconstructionConfig,
        cpu: Option<CpuCapabilities>,
        first_row: usize,
        held_rows: usize,
    ) -> Result<(), ReconstructionError> {
        let plan = StripPlan::new(config, cpu)?;
        let block_rows = plan.block_rows;
        // A macroblock row waits a row for the low-pass-level filter, and
        // the caller may hold rows back for its other components.
        let slack = (held_rows + 3) * plan.macroblock_height();
        self.low
            .begin(plan.low_width(), first_row * block_rows, 4 * block_rows);
        self.low_finished = first_row * block_rows;
        self.samples
            .begin(plan.width(), first_row * plan.macroblock_height(), slack);
        self.finished = first_row * plan.macroblock_height();
        self.placed = first_row;
        self.transformed = first_row;
        self.simd = ReconstructionSimd::default();
        self.plan = Some(plan);
        Ok(())
    }

    /// Takes in the macroblock rows `rows`, which follow those already
    /// taken in, and finishes every sample row that no longer depends on a
    /// macroblock row still to come.
    pub(crate) fn push_rows(
        &mut self,
        source: &impl MacroblockSource,
        rows: Range<usize>,
    ) -> Result<(), ReconstructionError> {
        if rows.start != self.placed {
            return Err(ReconstructionError::InvalidPlaneGeometry(
                "macroblock rows out of order",
            ));
        }
        for row in rows {
            self.place_row(source, row)?;
            self.advance(false)?;
        }
        Ok(())
    }

    /// Finishes what the rows taken in allow when no more follow.
    ///
    /// At the bottom of the plane or of a hard tile nothing is left to do.
    /// Part way down, the last macroblock row is transformed although the
    /// filter of its lower blocks has not run, which leaves its first two
    /// sample rows, and with them every row above, as the plane-wide sweep
    /// has them.
    pub(crate) fn flush(&mut self) -> Result<(), ReconstructionError> {
        self.advance(true)
    }

    /// Sample rows from the top of the plane that are finished.
    pub(crate) const fn finished(&self) -> usize {
        self.finished
    }

    /// Samples across the plane.
    pub(crate) const fn width(&self) -> usize {
        self.samples.width()
    }

    /// The sample rows `rows`, which must be finished and not released.
    pub(crate) fn rows(&self, rows: Range<usize>) -> &[i32] {
        self.samples.rows(rows)
    }

    /// Lets go of the finished sample rows above `row`.
    pub(crate) fn release(&mut self, row: usize) {
        self.samples.discard_before(row.min(self.finished));
    }

    /// Makes room for the whole plane, for a strip that is never released.
    pub(crate) fn reserve_plane(&mut self, storage: Vec<i32>, macroblock_rows: usize) {
        let height = self
            .plan
            .as_ref()
            .map_or(0, |plan| macroblock_rows * plan.macroblock_height());
        self.samples.adopt(storage);
        self.samples.reserve_rows(height);
    }

    /// Takes the sample rows held out of the strip.
    pub(crate) fn take_samples(&mut self) -> Vec<i32> {
        self.samples.take()
    }

    /// Bytes of storage retained between strips.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.low
            .retained_bytes()
            .saturating_add(self.samples.retained_bytes())
    }

    /// Reconstruction stages that used native vector groups in this strip.
    pub(crate) const fn simd(&self) -> ReconstructionSimd {
        self.simd
    }

    /// Dequantizes macroblock row `row` and runs its first-level transform:
    /// the block DC coefficients go to `low`, and the coefficients of every
    /// block straight to its place in `samples`.
    fn place_row(
        &mut self,
        source: &impl MacroblockSource,
        row: usize,
    ) -> Result<(), ReconstructionError> {
        let plan = self.plan.as_ref().ok_or(NOT_BEGUN)?;
        let (block_columns, block_rows) = (plan.block_columns, plan.block_rows);
        let block_count = block_columns * block_rows;
        let low_width = plan.low_width();
        let width = low_width * 4;
        let low_rows = self.low.append(block_rows);
        let rows = self.samples.append(plan.macroblock_height());
        let mut scratch = QuantizedMacroblock::ZERO;
        for x in 0..plan.macroblocks_x {
            let macroblock = source.macroblock(x, row, &mut scratch)?;
            let mut low = dequantize_low(macroblock, block_count)?;
            inverse_first_level(&mut low, &plan.config)?;
            let column = x * block_columns;
            if block_columns == 4 {
                place_low::<4>(&low[..block_count], low_rows, low_width, column);
            } else {
                place_low::<2>(&low[..block_count], low_rows, low_width, column);
            }
            dequantize_high_pass_into(macroblock, block_columns, block_count, rows, width, column)?;
        }
        self.placed = row + 1;
        Ok(())
    }

    /// Runs every stage as far as the rows placed allow, or to their end
    /// when `to_end` says no more rows follow.
    fn advance(&mut self, to_end: bool) -> Result<(), ReconstructionError> {
        let plan = self.plan.as_ref().ok_or(NOT_BEGUN)?;
        let low_end = self.placed * plan.block_rows;
        if plan.config.overlap != OverlapMode::Two {
            self.low_finished = low_end;
        } else if low_end > self.low_finished {
            let held = self.low_finished..low_end;
            let low = self.low.rows_mut(held.clone());
            let used_simd = SimdReporter::new(false);
            self.low_finished =
                plan.filter_low::<Postfilter>(low, held, Executor::Sequential, Some(&used_simd))?;
            self.simd.first_overlap |= simd_used(&used_simd);
        }
        let height = plan.macroblock_height();
        while self.transformed < self.placed
            && (to_end || (self.transformed + 1) * plan.block_rows <= self.low_finished)
        {
            let row = self.transformed;
            let low = self
                .low
                .rows(row * plan.block_rows..(row + 1) * plan.block_rows);
            let samples = self.samples.rows_mut(row * height..(row + 1) * height);
            let used_simd = SimdReporter::new(false);
            transform_macroblock_row(plan, low, samples, Some(&used_simd))?;
            self.simd.second_inverse |= simd_used(&used_simd);
            self.transformed += 1;
        }
        self.low.discard_before(self.transformed * plan.block_rows);
        let end = self.transformed * height;
        if plan.config.overlap == OverlapMode::None {
            self.finished = end;
        } else if end > self.finished {
            let held = self.finished..end;
            let samples = self.samples.rows_mut(held.clone());
            let used_simd = SimdReporter::new(false);
            self.finished = plan.filter_samples::<Postfilter>(
                samples,
                held,
                Executor::Sequential,
                Some(&used_simd),
            )?;
            self.simd.second_overlap |= simd_used(&used_simd);
        }
        Ok(())
    }
}

/// What a strip that has not begun, of either direction, returns.
pub(crate) const NOT_BEGUN: ReconstructionError =
    ReconstructionError::InvalidPlaneGeometry("strip has no plane");

/// Gives every block of one macroblock row its DC coefficient from `low`
/// and runs the second-level transform over the row in place.
fn transform_macroblock_row(
    plan: &StripPlan,
    low: &[i32],
    samples: &mut [i32],
    used_simd: Option<&SimdReporter>,
) -> Result<(), ReconstructionError> {
    let low_width = plan.low_width();
    let width = low_width * 4;
    if width == 0 {
        return Ok(());
    }
    for (values, block_row) in low
        .chunks_exact(low_width)
        .zip(samples.chunks_exact_mut(width * 4))
    {
        for (slot, &value) in block_row[..width].iter_mut().step_by(4).zip(values) {
            *slot = value;
        }
    }
    transform_rows::<InverseTransform>(plan.level, samples, width, Executor::Sequential, used_simd)
}

#[cfg(test)]
mod tests {
    use jxr_core::{BandPresence, OverlapMode, QuantizerSet};

    use super::*;
    use crate::entropy::encode::test_support::Lcg;
    use crate::reconstruct::{TilePartition, reconstruct_luma};

    fn noise(state: &mut u64, range: i32) -> i32 {
        *state = Lcg::advance(*state);
        i32::try_from((*state >> 40) % u64::try_from(2 * range).unwrap()).unwrap() - range
    }

    fn macroblocks(count: usize, seed: u64) -> Vec<QuantizedMacroblock> {
        let mut state = seed;
        (0..count)
            .map(|_| QuantizedMacroblock {
                dc_low_pass: core::array::from_fn(|_| noise(&mut state, 400)),
                high_pass: core::array::from_fn(|_| noise(&mut state, 60)),
                quantizers: QuantizerSet {
                    dc: 3,
                    low_pass: 2,
                    high_pass: 5,
                },
                bands: BandPresence::All,
            })
            .collect()
    }

    /// The rows of the plane a strip of the macroblock rows `owned` must
    /// take in: one more on each side, but for the edges of the plane and
    /// of hard tiles, which no window crosses.
    fn with_halo(owned: &Range<usize>, config: &ReconstructionConfig) -> Range<usize> {
        let edge = |row: usize| {
            row == 0
                || row == config.macroblocks_y as usize
                || (config.tiles.hard_boundaries
                    && config
                        .tiles
                        .row_boundaries
                        .contains(&u32::try_from(row).unwrap()))
        };
        owned.start - usize::from(!edge(owned.start))..owned.end + usize::from(!edge(owned.end))
    }

    /// Every run of macroblock rows of a plane, reconstructed by a strip of
    /// its own that takes in one macroblock row more on each side, has the
    /// samples the plane-wide sweep gives it: for every overlap mode, block
    /// geometry and tile layout, in bands of any size, with the rows let go
    /// as they are read.
    #[test]
    fn a_strip_with_a_halo_row_on_each_side_matches_the_plane_wide_sweep() {
        let (columns, rows) = (5_u32, 7_u32);
        let layouts = [
            TilePartition::single(columns, rows),
            TilePartition {
                column_boundaries: vec![0, 2, 5],
                row_boundaries: vec![0, 1, 4, 7],
                hard_boundaries: false,
            },
            TilePartition {
                column_boundaries: vec![0, 1, 5],
                row_boundaries: vec![0, 3, 4, 7],
                hard_boundaries: true,
            },
        ];
        for (block_columns, block_rows) in [(4_u8, 4_u8), (2, 4), (2, 2)] {
            for overlap in [OverlapMode::None, OverlapMode::One, OverlapMode::Two] {
                for tiles in &layouts {
                    let config = ReconstructionConfig {
                        macroblock_origin_x: 0,
                        macroblock_origin_y: 0,
                        macroblocks_x: columns,
                        macroblocks_y: rows,
                        block_columns,
                        block_rows,
                        scale_after_first_transform: block_columns == 2,
                        overlap,
                        tiles: tiles.clone(),
                    };
                    let source = macroblocks(35, u64::from(block_columns) * 7 + 3);
                    let plane = reconstruct_luma(&source, &config).unwrap();
                    let width = plane.width as usize;
                    let height = usize::from(block_rows) * 4;
                    let source = RasterMacroblocks {
                        macroblocks: &source,
                        width: columns as usize,
                    };
                    let mut strip = ComponentStrip::default();
                    for first in 0..rows as usize {
                        for end in first + 1..=rows as usize {
                            let owned = first..end;
                            let taken = with_halo(&owned, &config);
                            strip.begin(&config, None, taken.start, 1).unwrap();
                            // A row either side of the rows the strip is
                            // after is right as well, where it took one in:
                            // the interpolation of chroma reads it.
                            let margin = |halo: bool| usize::from(halo);
                            let wanted = owned.start * height - margin(taken.start < owned.start)
                                ..owned.end * height + margin(taken.end > owned.end);
                            let mut read = wanted.start;
                            // Bands of one and two macroblock rows by turns.
                            let mut next = taken.start;
                            while next < taken.end {
                                let band = next..(next + 1 + next % 2).min(taken.end);
                                strip.push_rows(&source, band.clone()).unwrap();
                                next = band.end;
                                if next == taken.end {
                                    strip.flush().unwrap();
                                }
                                let ready = strip.finished().min(wanted.end);
                                if ready > read {
                                    assert_eq!(
                                        strip.rows(read..ready),
                                        &plane.samples[read * width..ready * width],
                                        "{block_columns}x{block_rows} {overlap:?} {tiles:?} \
                                         rows {owned:?} at {read}..{ready}",
                                    );
                                    read = ready;
                                    strip.release(read);
                                }
                            }
                            assert_eq!(read, wanted.end, "{overlap:?} rows {owned:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_window_slides_without_losing_the_rows_it_holds() {
        let mut window = RowWindow::default();
        window.begin(3, 10, 2);
        let mut next = 0;
        for step in 0..40 {
            for value in window.append(2) {
                *value = next;
                next += 1;
            }
            // Keep between one and three rows back.
            window.discard_before(window.end().saturating_sub(1 + step % 3));
            let first = window.end() - (1 + step % 3).min(window.end() - 10);
            let held = window.rows(first..window.end());
            let expected = i32::try_from((first - 10) * 3).unwrap();
            assert_eq!(held[0], expected, "step {step}");
            assert_eq!(*held.last().unwrap(), next - 1, "step {step}");
        }
        let taken = window.take();
        assert_eq!(*taken.last().unwrap(), next - 1);
    }
}
