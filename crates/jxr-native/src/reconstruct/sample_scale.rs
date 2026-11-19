//! The scale of the coefficients a reduced-resolution decode returns.

use jxr_core::{BandPresence, OverlapMode, QuantizerSet};

use super::{
    QuantizedMacroblock, ReconstructionConfig, ReconstructionError, TilePartition,
    lanes::PARALLEL_SAMPLES,
    overlap::filter_regions,
    pipeline::{component_block_geometry, scale_boundaries},
};
use crate::{CpuCapabilities, analyze::analyze_component, executor::Executor};

/// Binary logarithm of the factor by which the DC coefficient of a block
/// exceeds the mean of the block's samples: the core transform sums four
/// samples along each axis and halves once.
pub(super) const BLOCK_DC_GAIN_BITS: u32 = 2;

/// Binary logarithm of the factor by which the DC coefficient of a
/// macroblock exceeds the mean of its blocks' DC coefficients. The transform
/// of the sixteen blocks of a full-resolution component has the gain of the
/// block transform. Subsampled chroma has two blocks along an axis it is
/// subsampled on, and its transform sums and halves them to a gain of two.
pub(super) fn macroblock_dc_gain_bits(
    config: &ReconstructionConfig,
) -> Result<u32, ReconstructionError> {
    match component_block_geometry(config)? {
        (4, 4) => Ok(BLOCK_DC_GAIN_BITS),
        _ => Ok(1),
    }
}

/// Levels of the flat images whose reduced pictures give the scale, each
/// probed at both signs.
///
/// They are large enough for the rounding of every stage to vanish beside
/// the slope they give, and small enough for every stage to keep to its
/// vector path. What the stages round away differs from level to level, so
/// the offset is taken over a run of neighbouring levels, as an image of
/// many levels meets it.
const PROBE_LEVELS: [i32; 4] = [0x1_0001, 0x1_0002, 0x1_0003, 0x1_0004];

/// Macroblocks along the longest region of the probe image.
///
/// An overlap operator reaches from a macroblock into the ones beside it
/// and no further. What a macroblock of a flat image holds therefore depends
/// on where the edges of its region lie among itself and its neighbours, and
/// what a reduced decode makes of it on the same for those neighbours.
/// Regions of one to five macroblocks have every such arrangement: in a
/// longer one, a macroblock further than one from both ends is placed like
/// the middle one of five.
const PROBE_EXTENT: usize = 5;

/// What turns the coefficients a reduced decode stops at into the samples
/// they stand for.
///
/// A reduced decode leaves out the block transform and the overlap filtering
/// that follows it. Both scale a flat area: the transform divides by a power
/// of two, and the inverse overlap filter raises what its forward filter
/// lowered, by a factor that is no power of two. A coefficient left at its
/// own scale would give a picture of the wrong contrast.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum SampleScale {
    /// The gain of the transform stages alone, as its binary logarithm:
    /// all there is to undo in an image coded without overlap filtering.
    Shift(u32),
    /// The reduced pictures of flat images of opposite levels, between
    /// which the reduced picture of the component lies as its samples lie
    /// between those levels.
    Flat(FlatPictures),
}

/// The unscaled reduced pictures of the flat probe image at every one of
/// [`PROBE_LEVELS`] and at its negative, as the terms of the line through
/// them, and the sample of them that is placed like each sample of a
/// component's reduced picture.
///
/// The forward overlap filters round upwards, by half a level and more on
/// average, and the inverse filters take that back. A reduced decode does
/// not run them, so the coefficient of a flat area is not in proportion to
/// its level: it is offset by what the rounding left. Levels of opposite
/// sign give both the proportion and the offset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct FlatPictures {
    /// What a coefficient is multiplied by: the number of pictures, times
    /// the positive levels less the negative ones.
    gain: i64,
    /// The terms for each sample of the pictures, row-major.
    terms: Vec<FlatTerms>,
    /// Samples across the pictures.
    stride: usize,
    /// For each column of the component's picture, its column here.
    columns: Vec<usize>,
    /// For each row of the component's picture, its row here.
    rows: Vec<usize>,
}

/// What the pictures of the probe hold at one sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FlatTerms {
    /// What a coefficient times the gain is divided by, positive: the
    /// number of pictures, times the pictures of the positive levels less
    /// those of the negative ones.
    divisor: i64,
    /// What is taken from a coefficient times the gain first: the sum of
    /// the pictures, which is what rounding added to all of them, times the
    /// positive levels less the negative ones; less half the divisor, for
    /// the quotient to round to nearest.
    bias: i64,
}

/// What a reduced decode makes of the quantized macroblocks of a component
/// reconstructed as the configuration says: its unscaled reduced picture.
pub(super) type ReducedPicture<'a> = dyn Fn(&[QuantizedMacroblock], &ReconstructionConfig) -> Result<Vec<i32>, ReconstructionError>
    + 'a;

impl SampleScale {
    pub(super) fn retained_bytes(&self) -> usize {
        match self {
            Self::Shift(_) => 0,
            Self::Flat(flat) => flat
                .terms
                .capacity()
                .saturating_mul(core::mem::size_of::<FlatTerms>())
                .saturating_add(
                    flat.columns
                        .capacity()
                        .saturating_add(flat.rows.capacity())
                        .saturating_mul(core::mem::size_of::<usize>()),
                ),
        }
    }

    /// The scale of the reduced picture of a component reconstructed as
    /// `config` says, which has `units` samples across and down each
    /// macroblock, and where the transform stages left out have a gain of
    /// two to the power `transform_gain_bits`.
    ///
    /// Without overlap filtering that gain is the scale. With it the scale
    /// is measured on the stages themselves: flat images are run through
    /// [`analyze_component`], the exact inverse of full reconstruction, with
    /// the decode's `cpu`, and `picture` returns the unscaled reduced
    /// picture of their macroblocks, given their geometry. Whatever the filters do to a flat area at the
    /// position of a sample, and whatever the reduced decode leaves undone
    /// of it, is in those pictures, so every flat area of the component
    /// comes out at its own level, but for what the forward filters round
    /// away at that one level and not on average.
    ///
    /// The probe image has no more than [`PROBE_EXTENT`] macroblocks for
    /// each distinct region extent up to that along each axis, however
    /// large the component: see [`ProbeAxis`].
    ///
    /// `picture` is taken as a trait object so that the quarter and the
    /// sixteenth scale share one copy of this. The reconstruction workspace
    /// retains the result while its geometry and reduced resolution match.
    pub(super) fn new(
        config: &ReconstructionConfig,
        transform_gain_bits: u32,
        units: (usize, usize),
        cpu: CpuCapabilities,
        picture: &ReducedPicture<'_>,
    ) -> Result<Self, ReconstructionError> {
        if config.overlap == OverlapMode::None {
            return Ok(Self::Shift(transform_gain_bits));
        }
        let tiles = &config.tiles;
        let hard = tiles.hard_boundaries;
        let columns = ProbeAxis::new(
            filter_regions(&tiles.column_boundaries, &[0, config.macroblocks_x], hard),
            "column",
        )?;
        let rows = ProbeAxis::new(
            filter_regions(&tiles.row_boundaries, &[0, config.macroblocks_y], hard),
            "row",
        )?;
        let probe_macroblocks = (columns.macroblocks, rows.macroblocks);
        let positions = (
            columns.sample_positions(units.0),
            rows.sample_positions(units.1),
        );
        let probe_config = ReconstructionConfig {
            macroblock_origin_x: 0,
            macroblock_origin_y: 0,
            macroblocks_x: columns.extent(),
            macroblocks_y: rows.extent(),
            tiles: TilePartition {
                column_boundaries: columns.boundaries,
                row_boundaries: rows.boundaries,
                hard_boundaries: true,
            },
            ..config.clone()
        };
        let (block_columns, block_rows) = component_block_geometry(config)?;
        let sample_count =
            probe_macroblocks.0 * block_columns * 4 * probe_macroblocks.1 * block_rows * 4;
        let stride = probe_macroblocks.0 * units.0;
        let length = stride * probe_macroblocks.1 * units.1;
        let flat_picture = |level: i32| {
            let unit = QuantizerSet {
                dc: 1,
                low_pass: 1,
                high_pass: 1,
            };
            let flat = vec![level; sample_count];
            let flat = analyze_component(flat, &probe_config, unit, BandPresence::All, cpu)?;
            let samples = picture(&flat, &probe_config)?;
            if samples.len() == length {
                Ok(samples)
            } else {
                Err(ReconstructionError::InvalidPlaneGeometry(
                    "reduced picture of the sample scale probe",
                ))
            }
        };
        let (mut rise, mut offset) = (vec![0_i64; length], vec![0_i64; length]);
        let (mut span, mut count) = (0_i64, 0_i64);
        for level in PROBE_LEVELS {
            let raised = flat_picture(level)?;
            let lowered = flat_picture(-level)?;
            for (index, (raised, lowered)) in raised.into_iter().zip(lowered).enumerate() {
                rise[index] += i64::from(raised) - i64::from(lowered);
                offset[index] += i64::from(raised) + i64::from(lowered);
            }
            span += 2 * i64::from(level);
            count += 2;
        }
        // The coefficient of a flat image of `level` is its share of the
        // offset and `level` parts in `span` of its share of the rise.
        let terms = rise
            .into_iter()
            .zip(offset)
            .map(|(rise, offset)| {
                let divisor = count.checked_mul(rise).filter(|&divisor| divisor > 0)?;
                let bias = offset.checked_mul(span)?.checked_sub(divisor / 2)?;
                Some(FlatTerms { divisor, bias })
            })
            .collect::<Option<_>>()
            .ok_or(ReconstructionError::ArithmeticOverflow(
                "reduced-resolution sample scale",
            ))?;
        Ok(Self::Flat(FlatPictures {
            gain: count * span,
            terms,
            stride,
            columns: positions.0,
            rows: positions.1,
        }))
    }

    /// Scales the unscaled reduced picture of the component, row-major in
    /// `samples`, to the samples it stands for, a band of rows at a time
    /// through `executor`.
    pub(super) fn apply(
        &self,
        executor: Executor,
        samples: &mut [i32],
    ) -> Result<(), ReconstructionError> {
        let flat = match self {
            Self::Shift(bits) => {
                for sample in samples {
                    *sample >>= bits;
                }
                return Ok(());
            }
            Self::Flat(flat) => flat,
        };
        let width = flat.columns.len();
        if samples.len() != width * flat.rows.len() {
            return Err(ReconstructionError::InvalidPlaneGeometry(
                "reduced picture differs from its sample scale",
            ));
        }
        if samples.is_empty() {
            return Ok(());
        }
        let scale_row = |(row, &flat_row): (&mut [i32], &usize)| {
            let terms = &flat.terms[flat_row * flat.stride..][..flat.stride];
            row.iter_mut()
                .zip(&flat.columns)
                .try_for_each(|(sample, &column)| {
                    *sample = terms[column].level(*sample, flat.gain)?;
                    Ok(())
                })
        };
        // Rows are scaled a band of about `PARALLEL_SAMPLES` to a thread.
        let band_rows = (PARALLEL_SAMPLES / width).max(1);
        executor
            .if_worth(samples.len() > PARALLEL_SAMPLES)
            .try_for_each_chunk(samples, width * band_rows, |index, band| {
                band.chunks_exact_mut(width)
                    .zip(&flat.rows[index * band_rows..])
                    .try_for_each(scale_row)
            })
    }
}

impl FlatTerms {
    /// The level of the flat image that has `coefficient` at this sample,
    /// rounded to nearest, given the gain of the pictures.
    ///
    /// An image of zeros has coefficients of zero: nothing is rounded in
    /// it, so it has none of the offset the other levels have, and zero
    /// stands for it.
    #[inline]
    fn level(self, coefficient: i32, gain: i64) -> Result<i32, ReconstructionError> {
        if coefficient == 0 {
            return Ok(0);
        }
        // The gain is the sum of a few probe levels, and the bias that of
        // as few 32-bit pictures times as much: neither term nears 64 bits.
        let level = (i64::from(coefficient) * gain - self.bias).div_euclid(self.divisor);
        i32::try_from(level)
            .map_err(|_| ReconstructionError::ArithmeticOverflow("reduced-resolution sample"))
    }
}

/// One axis of the probe image, and of the way from a component to it.
///
/// The probe has one region for each extent a region of the component has
/// along the axis, counting every extent of [`PROBE_EXTENT`] or more as
/// that. Its regions are hard tiles: no overlap operator crosses from one
/// into the next, as none crosses the edge of a region of the component.
struct ProbeAxis {
    /// Boundaries of the probe's regions, in macroblocks.
    boundaries: Vec<u32>,
    /// Macroblocks along the probe.
    macroblocks: usize,
    /// For each macroblock along the component, the probe macroblock that
    /// lies in its region as it lies in its own.
    positions: Vec<usize>,
}

impl ProbeAxis {
    /// The probe axis for a component whose regions along the axis have the
    /// boundaries `regions`, in macroblocks.
    fn new(regions: &[u32], axis: &'static str) -> Result<Self, ReconstructionError> {
        let regions = scale_boundaries(regions, 1)?;
        let mut boundaries = vec![0];
        let mut macroblocks = 0;
        let mut positions = Vec::with_capacity(regions.last().copied().unwrap_or(0));
        // Where the probe region of each extent starts, once there is one.
        let mut starts = [None; PROBE_EXTENT + 1];
        for region in regions.windows(2) {
            let extent = region[1]
                .checked_sub(region[0])
                .filter(|&extent| extent > 0)
                .ok_or(ReconstructionError::InvalidTilePartition(axis))?;
            let probe_extent = extent.min(PROBE_EXTENT);
            let start = *starts[probe_extent].get_or_insert_with(|| {
                let start = macroblocks;
                macroblocks += probe_extent;
                boundaries.push(macroblocks);
                start
            });
            positions.extend((0..extent).map(|offset| start + probe_offset(offset, extent)));
        }
        if positions.is_empty() {
            return Err(ReconstructionError::InvalidTilePartition(axis));
        }
        let boundaries = boundaries
            .into_iter()
            .map(|boundary| {
                u32::try_from(boundary)
                    .map_err(|_| ReconstructionError::ArithmeticOverflow("probe tile boundary"))
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            boundaries,
            macroblocks,
            positions,
        })
    }

    /// Macroblocks along the probe, as its geometry counts them.
    fn extent(&self) -> u32 {
        self.boundaries[self.boundaries.len() - 1]
    }

    /// For each sample along a reduced picture of the component that has
    /// `units` samples to the macroblock, the sample of the probe's reduced
    /// picture placed like it.
    fn sample_positions(&self, units: usize) -> Vec<usize> {
        self.positions
            .iter()
            .flat_map(|&position| (0..units).map(move |unit| position * units + unit))
            .collect()
    }
}

/// The macroblock of a probe region that lies in it as macroblock `offset`
/// lies in a region of `extent` macroblocks: at the same distance from the
/// nearer end where that is less than two, in the middle otherwise.
const fn probe_offset(offset: usize, extent: usize) -> usize {
    let ends = PROBE_EXTENT / 2;
    if extent <= PROBE_EXTENT || offset < ends {
        offset
    } else if extent - offset <= ends {
        PROBE_EXTENT - (extent - offset)
    } else {
        ends
    }
}

#[cfg(test)]
mod tests {
    use super::{PROBE_EXTENT, ProbeAxis, probe_offset};

    #[test]
    fn a_region_keeps_its_ends_and_folds_its_middle_onto_the_probe() {
        assert_eq!(PROBE_EXTENT, 5);
        for extent in 1..=5 {
            for offset in 0..extent {
                assert_eq!(probe_offset(offset, extent), offset, "{offset} of {extent}");
            }
        }
        let folded = |extent| -> Vec<usize> {
            (0..extent)
                .map(|offset| probe_offset(offset, extent))
                .collect()
        };
        assert_eq!(folded(6), [0, 1, 2, 2, 3, 4]);
        assert_eq!(folded(9), [0, 1, 2, 2, 2, 2, 2, 3, 4]);
    }

    #[test]
    fn the_probe_has_one_region_for_each_extent_of_the_component() {
        // Regions of 2, 1, 7, 2 and 30 macroblocks: the probe has one of
        // two, one of one, and one of five for both long ones.
        let axis = ProbeAxis::new(&[0, 2, 3, 10, 12, 42], "column").unwrap();
        assert_eq!(axis.boundaries, [0, 2, 3, 8]);
        assert_eq!(axis.macroblocks, 8);
        assert_eq!(axis.extent(), 8);
        assert_eq!(axis.positions.len(), 42);
        assert_eq!(axis.positions[..12], [0, 1, 2, 3, 4, 5, 5, 5, 6, 7, 0, 1]);
        assert_eq!(axis.positions[12..16], [3, 4, 5, 5]);
        assert_eq!(axis.positions[38..], [5, 5, 6, 7]);
        // Four samples to the macroblock keep their place within it.
        assert_eq!(
            axis.sample_positions(4)[..12],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        );
        assert_eq!(axis.sample_positions(4)[12..16], [12, 13, 14, 15]);
        assert_eq!(axis.sample_positions(2)[20..24], [0, 1, 2, 3]);
    }

    #[test]
    fn a_partition_that_does_not_advance_is_refused() {
        assert!(ProbeAxis::new(&[0, 3, 3], "row").is_err());
        assert!(ProbeAxis::new(&[0, 3, 2], "row").is_err());
        assert!(ProbeAxis::new(&[0], "row").is_err());
    }
}
