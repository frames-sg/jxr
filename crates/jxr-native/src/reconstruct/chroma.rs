//! Full-resolution chroma reconstruction for T.832 4:2:0 and 4:2:2 planes,
//! and the subsampling an encoder forms those planes with, both as streams
//! of rows.

use jxr_core::ChromaSampling;
use jxr_math::sampling::{ChromaCentering, downsample_chroma, upsample_chroma_pair};

use core::ops::Range;

use crate::ImagePlaneHeader;

use super::{
    PlanarSamples, ReconstructionError,
    strip::{ComponentStrip, RowWindow},
};

/// Geometry and parsed chroma-grid positioning for one reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChromaReconstructionConfig {
    /// Coded chroma subsampling mode.
    pub sampling: ChromaSampling,
    /// Full-resolution luma width in samples, including coded margins.
    pub full_width: u32,
    /// Full-resolution luma height in samples, including coded margins.
    pub full_height: u32,
    /// Horizontal quarter-sample chroma-grid offset.
    pub centering_x: u8,
    /// Vertical quarter-sample chroma-grid offset.
    pub centering_y: u8,
}

impl ChromaReconstructionConfig {
    /// Build reconstruction geometry directly from a parsed image-plane header.
    pub fn from_header(
        header: &ImagePlaneHeader,
        full_width: u32,
        full_height: u32,
    ) -> Result<Self, ReconstructionError> {
        let sampling = match header.internal_color_format {
            1 => ChromaSampling::Cs420,
            2 => ChromaSampling::Cs422,
            _ => {
                return Err(ReconstructionError::Unsupported(
                    "chroma reconstruction requires YUV420 or YUV422",
                ));
            }
        };
        Ok(Self {
            sampling,
            full_width,
            full_height,
            centering_x: header.chroma_centering_x,
            centering_y: header.chroma_centering_y,
        })
    }
}

/// Reconstruct U and V to full resolution using the T.832 example filter.
///
/// The returned planes are YUV444 geometry and can be passed directly to the
/// output formatter together with a full-resolution luma plane.
pub fn reconstruct_chroma_444(
    u: &PlanarSamples,
    v: &PlanarSamples,
    config: ChromaReconstructionConfig,
) -> Result<[PlanarSamples; 2], ReconstructionError> {
    validate_matching_chroma(u, v)?;
    Ok([reconstruct_plane(u, config)?, reconstruct_plane(v, config)?])
}

/// One chroma plane at full resolution: the [`ChromaStrip`] of the whole
/// plane, every row of which is finished.
fn reconstruct_plane(
    input: &PlanarSamples,
    config: ChromaReconstructionConfig,
) -> Result<PlanarSamples, ReconstructionError> {
    validate_plane(input)?;
    let width = usize::try_from(input.width)
        .map_err(|_| ReconstructionError::ArithmeticOverflow("chroma width conversion"))?;
    let height = usize::try_from(input.height)
        .map_err(|_| ReconstructionError::ArithmeticOverflow("chroma height conversion"))?;
    let mut strip = ChromaStrip::default();
    // The strip checks the centering codes and the sampling.
    strip.begin(config, [width, height], 0, 0)?;
    let rows: u32 = if config.sampling == ChromaSampling::Cs420 {
        2
    } else {
        1
    };
    validate_geometry(
        input,
        config.full_width,
        config.full_height,
        true,
        rows == 2,
    )?;
    strip.reserve_rows(height * rows as usize);
    strip.advance(&mut WholePlane {
        samples: &input.samples,
        width,
        height,
    })?;
    Ok(PlanarSamples {
        origin_x: input
            .origin_x
            .checked_mul(2)
            .ok_or(ReconstructionError::ArithmeticOverflow(
                "upsampled x origin",
            ))?,
        origin_y: input.origin_y.checked_mul(rows).ok_or(
            ReconstructionError::ArithmeticOverflow("upsampled y origin"),
        )?,
        width: config.full_width,
        height: config.full_height,
        samples: strip.window.take(),
    })
}

/// The rows of a chroma plane a [`ChromaStrip`] interpolates.
pub(crate) trait ChromaRows {
    /// Samples across the plane.
    fn width(&self) -> usize;
    /// Rows from the top of the plane that are finished.
    fn finished(&self) -> usize;
    /// The rows `rows`, which must be finished and held.
    fn rows(&self, rows: Range<usize>) -> &[i32];
    /// Lets go of the finished rows above `row`.
    fn release(&mut self, row: usize);
}

impl ChromaRows for ComponentStrip {
    fn width(&self) -> usize {
        Self::width(self)
    }

    fn finished(&self) -> usize {
        Self::finished(self)
    }

    fn rows(&self, rows: Range<usize>) -> &[i32] {
        Self::rows(self, rows)
    }

    fn release(&mut self, row: usize) {
        Self::release(self, row);
    }
}

/// A whole chroma plane, every row of it finished and held.
struct WholePlane<'a> {
    samples: &'a [i32],
    width: usize,
    height: usize,
}

impl ChromaRows for WholePlane<'_> {
    fn width(&self) -> usize {
        self.width
    }

    fn finished(&self) -> usize {
        self.height
    }

    fn rows(&self, rows: Range<usize>) -> &[i32] {
        &self.samples[rows.start * self.width..rows.end * self.width]
    }

    fn release(&mut self, _: usize) {}
}

/// Interpolates one chroma row to `output`, twice as wide. The row's end
/// samples stand in for the neighbours beyond them.
fn widen_row(
    row: &[i32],
    centering: ChromaCentering,
    output: &mut [i32],
) -> Result<(), ReconstructionError> {
    let pairs = output.as_chunks_mut::<2>().0;
    debug_assert_eq!(pairs.len(), row.len());
    let Some(last) = row.len().min(pairs.len()).checked_sub(1) else {
        return Ok(());
    };
    let interpolate = |previous, current, next| {
        upsample_chroma_pair(previous, current, next, centering)
            .map_err(|_| ReconstructionError::ArithmeticOverflow("horizontal chroma interpolation"))
    };
    pairs[0] = interpolate(row[0], row[0], row[1.min(last)])?;
    // Between the ends every sample has both neighbours, and the loop is
    // one the compiler writes with vectors.
    for (pair, window) in pairs[1..].iter_mut().zip(row.windows(3)) {
        *pair = interpolate(window[0], window[1], window[2])?;
    }
    if last > 0 {
        pairs[last] = interpolate(row[last - 1], row[last], row[last])?;
    }
    Ok(())
}

/// Interpolates the chroma row `current` to the two rows `even` and `odd`
/// that take its place, from the rows above and below it.
fn double_row(
    [previous, current, next]: [&[i32]; 3],
    centering: ChromaCentering,
    even: &mut [i32],
    odd: &mut [i32],
) -> Result<(), ReconstructionError> {
    let sources = previous.iter().zip(current).zip(next);
    for ((even, odd), ((&previous, &current), &next)) in even.iter_mut().zip(odd).zip(sources) {
        [*even, *odd] = upsample_chroma_pair(previous, current, next, centering).map_err(|_| {
            ReconstructionError::ArithmeticOverflow("vertical chroma interpolation")
        })?;
    }
    Ok(())
}

/// One chroma component interpolated to full resolution as its rows are
/// finished: [`reconstruct_chroma_444`] as a stream.
///
/// Rows are counted from the top of the plane the component's strip runs
/// down. A 4:2:0 row is interpolated from the rows above and below it, so
/// it waits for the row below to be finished, and the row above is held
/// back from release until then. At the top and bottom of the plane the
/// end row stands in for the missing neighbour, as it does plane-wide.
#[derive(Debug, Default)]
pub(crate) struct ChromaStrip {
    /// Full-resolution rows not yet released.
    window: RowWindow,
    /// A 4:2:0 row doubled vertically, before it is widened.
    doubled: Vec<i32>,
    geometry: Option<ChromaStripGeometry>,
    /// Chroma rows interpolated.
    interpolated: usize,
}

#[derive(Debug, Clone, Copy)]
struct ChromaStripGeometry {
    centering_x: ChromaCentering,
    /// Vertical centering of 4:2:0, or `None` for 4:2:2.
    centering_y: Option<ChromaCentering>,
    /// Rows of the chroma plane.
    height: usize,
}

impl ChromaStrip {
    /// Begins at chroma row `first_row` of a chroma plane `width` samples
    /// wide and `height` rows tall, interpolated as `config` says.
    pub(crate) fn begin(
        &mut self,
        config: ChromaReconstructionConfig,
        [width, height]: [usize; 2],
        first_row: usize,
        slack: usize,
    ) -> Result<(), ReconstructionError> {
        let centering_x = centering(config.centering_x, "horizontal chroma centering")?;
        let centering_y = match config.sampling {
            ChromaSampling::Cs422 => None,
            ChromaSampling::Cs420 => {
                Some(centering(config.centering_y, "vertical chroma centering")?)
            }
            ChromaSampling::Cs444 => {
                return Err(ReconstructionError::Unsupported(
                    "YUV444 chroma does not require upsampling",
                ));
            }
        };
        self.geometry = Some(ChromaStripGeometry {
            centering_x,
            centering_y,
            height,
        });
        let rows = if centering_y.is_some() { 2 } else { 1 };
        self.window.begin(width * 2, first_row * rows, slack);
        self.doubled.clear();
        self.doubled.resize(width * 2, 0);
        self.interpolated = first_row;
        Ok(())
    }

    /// Interpolates every chroma row of `source` that is finished and, for
    /// 4:2:0, has a finished row below it or is the last of the plane.
    /// Rows of `source` that no later row needs are released.
    pub(crate) fn advance(
        &mut self,
        source: &mut impl ChromaRows,
    ) -> Result<(), ReconstructionError> {
        let Some(geometry) = self.geometry else {
            return Err(ReconstructionError::InvalidPlaneGeometry(
                "chroma strip has no plane",
            ));
        };
        let width = source.width();
        let finished = source.finished().min(geometry.height);
        let Some(centering_y) = geometry.centering_y else {
            for row in self.interpolated..finished {
                let output = self.window.append(1);
                widen_row(source.rows(row..row + 1), geometry.centering_x, output)?;
            }
            self.interpolated = self.interpolated.max(finished);
            source.release(self.interpolated);
            return Ok(());
        };
        while self.interpolated < finished
            && (self.interpolated + 1 < finished || self.interpolated + 1 == geometry.height)
        {
            let row = self.interpolated;
            let above = row.saturating_sub(1);
            let below = (row + 1).min(geometry.height - 1);
            let line = |row: usize| source.rows(row..row + 1);
            let (even, odd) = self.doubled.split_at_mut(width);
            double_row(
                [line(above), line(row), line(below)],
                centering_y,
                even,
                odd,
            )?;
            let output = self.window.append(2);
            let (first, second) = output.split_at_mut(width * 2);
            widen_row(even, geometry.centering_x, first)?;
            widen_row(odd, geometry.centering_x, second)?;
            self.interpolated += 1;
        }
        source.release(self.interpolated.saturating_sub(1));
        Ok(())
    }

    /// Makes room for `rows` full-resolution rows, so that taking them in
    /// does not move those already held.
    fn reserve_rows(&mut self, rows: usize) {
        self.window.reserve_rows(rows);
    }

    /// Full-resolution rows from the top of the plane that are finished.
    pub(crate) const fn finished(&self) -> usize {
        self.window.end()
    }

    /// Samples across the full-resolution plane.
    pub(crate) const fn width(&self) -> usize {
        self.window.width()
    }

    /// The full-resolution rows `rows`, which must be finished and held.
    pub(crate) fn rows(&self, rows: Range<usize>) -> &[i32] {
        self.window.rows(rows)
    }

    /// Lets go of the full-resolution rows above `row`.
    pub(crate) fn release(&mut self, row: usize) {
        self.window.discard_before(row);
    }

    /// Bytes of storage retained between strips.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.window.retained_bytes().saturating_add(
            self.doubled
                .capacity()
                .saturating_mul(core::mem::size_of::<i32>()),
        )
    }
}

/// One chroma component subsampled as its full-resolution rows arrive: the
/// direction [`ChromaStrip`] reverses.
///
/// The width is halved first and, for 4:2:0, the height second, each with
/// [`downsample_chroma`], so the result sits on the grid
/// [`jxr_math::sampling::DOWNSAMPLED_CENTERING`] declares. Each row is
/// halved across once, as it is taken in. A 4:2:0 row is filtered from the
/// halved rows up to two above and two below the one it keeps, mirrored at
/// the plane's first and last rows, so it waits for the rows below it; the
/// halved rows no later row reads are then let go. Every row is filtered
/// once, and rows are counted from the top of the plane.
#[derive(Debug, Default)]
pub(crate) struct ChromaDecimation {
    /// Rows halved across and not yet let go.
    narrow: RowWindow,
    /// Whether rows are halved as well as columns, for 4:2:0.
    halve_rows: bool,
    /// Full-resolution rows of the plane.
    height: usize,
    /// Subsampled rows written out.
    written: usize,
}

impl ChromaDecimation {
    /// Begins at the top of a full-resolution chroma plane `width` samples
    /// wide and `height` rows tall, subsampled to `sampling`, whose
    /// dimensions must be even along every halved axis.
    pub(crate) fn begin(
        &mut self,
        [width, height]: [usize; 2],
        sampling: ChromaSampling,
    ) -> Result<(), ReconstructionError> {
        let halve_rows = match sampling {
            ChromaSampling::Cs420 => true,
            ChromaSampling::Cs422 => false,
            ChromaSampling::Cs444 => {
                return Err(ReconstructionError::Unsupported(
                    "YUV444 chroma does not require subsampling",
                ));
            }
        };
        if width == 0 || height == 0 {
            return Err(ReconstructionError::InvalidPlaneGeometry(
                "chroma plane is not a whole number of rows",
            ));
        }
        if !width.is_multiple_of(2) || (halve_rows && !height.is_multiple_of(2)) {
            return Err(ReconstructionError::InvalidPlaneGeometry(
                "chroma plane has an odd dimension to halve",
            ));
        }
        // Between the rows it takes in, it keeps at most the four halved
        // rows the next 4:2:0 row reads above the first it has not: storage
        // grown by those beside the rows taken in holds as many again.
        self.narrow.begin(width / 2, 0, 4);
        self.halve_rows = halve_rows;
        self.height = height;
        self.written = 0;
        Ok(())
    }

    /// Takes in the next full-resolution rows of the plane, `rows`, a whole
    /// number of rows twice as wide as the subsampled plane.
    pub(crate) fn push_rows(&mut self, rows: &[i32]) -> Result<(), ReconstructionError> {
        let half = self.narrow.width();
        let width = half * 2;
        if width == 0
            || !rows.len().is_multiple_of(width)
            || self.narrow.end() + rows.len() / width > self.height
        {
            return Err(ReconstructionError::InvalidPlaneGeometry(
                "chroma rows outside the plane",
            ));
        }
        let overflow = |_| ReconstructionError::ArithmeticOverflow("chroma decimation");
        let narrow = self.narrow.append(rows.len() / width);
        for (row, output) in rows.chunks_exact(width).zip(narrow.chunks_exact_mut(half)) {
            for (index, sample) in output.iter_mut().enumerate() {
                *sample = downsample_chroma(|x| row[x], width, index).map_err(overflow)?;
            }
        }
        Ok(())
    }

    /// Subsampled rows from the top of the plane that the rows taken in
    /// finish: for 4:2:0 each one whose lowest tap has arrived, or every one
    /// once the plane's last row has.
    pub(crate) const fn finished(&self) -> usize {
        let end = self.narrow.end();
        if !self.halve_rows {
            end
        } else if end == self.height {
            end / 2
        } else {
            // Row `c` reads down to row `2c + 2`.
            end.saturating_sub(1) / 2
        }
    }

    /// Subsampled rows written out so far.
    pub(crate) const fn written(&self) -> usize {
        self.written
    }

    /// Writes the next subsampled rows into `output`, which holds a whole
    /// number of them, no more than are finished and not yet written, and
    /// lets go of the halved rows no later row reads.
    pub(crate) fn write_rows(&mut self, output: &mut [i32]) -> Result<(), ReconstructionError> {
        let half = self.narrow.width();
        let count = output.len().checked_div(half).unwrap_or(0);
        let rows = self.written..self.written + count;
        if !output.len().is_multiple_of(half.max(1)) || rows.end > self.finished() {
            return Err(ReconstructionError::InvalidPlaneGeometry(
                "chroma rows that are not finished",
            ));
        }
        if !self.halve_rows {
            output.copy_from_slice(self.narrow.rows(rows.clone()));
            self.narrow.discard_before(rows.end);
            self.written = rows.end;
            return Ok(());
        }
        let held = self.narrow.held();
        let narrow = self.narrow.rows(held.clone());
        let overflow = |_| ReconstructionError::ArithmeticOverflow("chroma decimation");
        for (row, output) in rows.clone().zip(output.chunks_exact_mut(half)) {
            for (column, sample) in output.iter_mut().enumerate() {
                let line = |y: usize| narrow[(y - held.start) * half + column];
                *sample = downsample_chroma(line, self.height, row).map_err(overflow)?;
            }
        }
        // The next row reads from two rows above the one it keeps.
        self.narrow.discard_before((2 * rows.end).saturating_sub(2));
        self.written = rows.end;
        Ok(())
    }
}

/// Subsample a full-resolution chroma plane to `sampling`: a
/// [`ChromaDecimation`] given the whole plane at once.
///
/// `samples` is a row-major plane `width` samples wide whose dimensions are
/// even along every halved axis. Returns the subsampled plane, `width / 2`
/// samples wide.
#[cfg(test)]
pub(crate) fn subsample_chroma(
    samples: &[i32],
    width: usize,
    sampling: ChromaSampling,
) -> Result<Vec<i32>, ReconstructionError> {
    if width == 0 || samples.is_empty() || !samples.len().is_multiple_of(width) {
        return Err(ReconstructionError::InvalidPlaneGeometry(
            "chroma plane is not a whole number of rows",
        ));
    }
    let mut decimation = ChromaDecimation::default();
    decimation.begin([width, samples.len() / width], sampling)?;
    decimation.push_rows(samples)?;
    let mut output = vec![0; decimation.finished() * (width / 2)];
    decimation.write_rows(&mut output)?;
    Ok(output)
}

fn validate_matching_chroma(
    u: &PlanarSamples,
    v: &PlanarSamples,
) -> Result<(), ReconstructionError> {
    if u.width != v.width
        || u.height != v.height
        || u.origin_x != v.origin_x
        || u.origin_y != v.origin_y
    {
        return Err(ReconstructionError::InvalidPlaneGeometry(
            "U and V dimensions differ",
        ));
    }
    Ok(())
}

fn validate_plane(plane: &PlanarSamples) -> Result<(), ReconstructionError> {
    if plane.width == 0 || plane.height == 0 {
        return Err(ReconstructionError::InvalidPlaneGeometry(
            "chroma plane is empty",
        ));
    }
    let expected = usize::try_from(plane.width)
        .ok()
        .and_then(|width| {
            usize::try_from(plane.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or(ReconstructionError::ArithmeticOverflow(
            "chroma plane extent",
        ))?;
    if plane.samples.len() < expected {
        return Err(ReconstructionError::BufferTooSmall {
            required: expected,
            available: plane.samples.len(),
        });
    }
    if plane.samples.len() > expected {
        return Err(ReconstructionError::InvalidPlaneGeometry(
            "chroma plane has trailing samples",
        ));
    }
    Ok(())
}

fn validate_geometry(
    input: &PlanarSamples,
    full_width: u32,
    full_height: u32,
    double_width: bool,
    double_height: bool,
) -> Result<(), ReconstructionError> {
    let width_factor = if double_width { 2 } else { 1 };
    let height_factor = if double_height { 2 } else { 1 };
    let expected_width = input
        .width
        .checked_mul(width_factor)
        .ok_or(ReconstructionError::ArithmeticOverflow("full chroma width"))?;
    let expected_height =
        input
            .height
            .checked_mul(height_factor)
            .ok_or(ReconstructionError::ArithmeticOverflow(
                "full chroma height",
            ))?;
    if expected_width != full_width || expected_height != full_height {
        return Err(ReconstructionError::InvalidPlaneGeometry(
            "subsampled chroma dimensions do not match luma geometry",
        ));
    }
    Ok(())
}

fn centering(value: u8, operation: &'static str) -> Result<ChromaCentering, ReconstructionError> {
    ChromaCentering::new(value).ok_or(ReconstructionError::Unsupported(operation))
}

#[cfg(test)]
mod tests {
    use jxr_core::ChromaSampling;

    use super::{
        ChromaReconstructionConfig, PlanarSamples, reconstruct_chroma_444, subsample_chroma,
    };

    fn plane(width: u32, height: u32, samples: &[i32]) -> PlanarSamples {
        PlanarSamples {
            origin_x: 0,
            origin_y: 0,
            width,
            height,
            samples: samples.to_vec(),
        }
    }

    #[test]
    fn reconstructs_422_horizontally_with_clamped_edges() {
        let input = plane(2, 1, &[8, 16]);
        let [u, v] = reconstruct_chroma_444(
            &input,
            &input,
            ChromaReconstructionConfig {
                sampling: ChromaSampling::Cs422,
                full_width: 4,
                full_height: 1,
                centering_x: 0,
                centering_y: 0,
            },
        )
        .unwrap();
        assert_eq!(u.samples, [8, 12, 16, 16]);
        assert_eq!(v, u);
    }

    #[test]
    fn reconstructs_420_vertically_then_horizontally() {
        let input = plane(1, 2, &[0, 16]);
        let [u, _] = reconstruct_chroma_444(
            &input,
            &input,
            ChromaReconstructionConfig {
                sampling: ChromaSampling::Cs420,
                full_width: 2,
                full_height: 4,
                centering_x: 0,
                centering_y: 0,
            },
        )
        .unwrap();
        assert_eq!(u.samples, [0, 0, 8, 8, 16, 16, 16, 16]);
    }

    #[test]
    fn rejects_unknown_centering_and_mismatched_geometry() {
        let input = plane(1, 1, &[0]);
        let mut config = ChromaReconstructionConfig {
            sampling: ChromaSampling::Cs422,
            full_width: 2,
            full_height: 1,
            centering_x: 7,
            centering_y: 0,
        };
        assert!(reconstruct_chroma_444(&input, &input, config).is_err());
        config.centering_x = 0;
        config.full_width = 3;
        assert!(reconstruct_chroma_444(&input, &input, config).is_err());
    }

    #[test]
    fn subsampling_halves_the_axes_of_the_sampling_format() {
        let (width, height) = (16_usize, 32_usize);
        let flat = vec![-37_i32; width * height];
        assert_eq!(
            subsample_chroma(&flat, width, ChromaSampling::Cs422).unwrap(),
            vec![-37; 8 * 32]
        );
        assert_eq!(
            subsample_chroma(&flat, width, ChromaSampling::Cs420).unwrap(),
            vec![-37; 8 * 16]
        );
    }

    #[test]
    fn subsampling_filters_rows_then_columns() {
        let (width, height) = (16_usize, 16_usize);
        // One sample of a row leaves 6/16 of itself in that row; one sample
        // of the column that keeps then leaves 6/16 of that below it.
        let mut plane = vec![0_i32; width * height];
        plane[4 * width + 6] = 256;
        let half = subsample_chroma(&plane, width, ChromaSampling::Cs422).unwrap();
        assert_eq!(half[4 * 8 + 3], 96);
        assert_eq!(half[4 * 8 + 2], 16);
        assert_eq!(half.iter().filter(|&&value| value != 0).count(), 3);
        let quarter = subsample_chroma(&plane, width, ChromaSampling::Cs420).unwrap();
        assert_eq!(quarter[2 * 8 + 3], 36);
        assert_eq!(quarter[8 + 3], 6);
        assert_eq!(quarter[2 * 8 + 2], 6);
        assert_eq!(quarter.iter().filter(|&&value| value != 0).count(), 9);
    }

    #[test]
    fn subsampling_then_reconstruction_restores_a_smooth_plane() {
        let (width, height) = (32_u32, 32_u32);
        let plane: Vec<i32> = (0..height)
            .flat_map(|y| (0..width).map(move |x| i32::try_from(3 * x + 5 * y).unwrap() - 100))
            .collect();
        for sampling in [ChromaSampling::Cs422, ChromaSampling::Cs420] {
            let half = subsample_chroma(&plane, width as usize, sampling).unwrap();
            let input = PlanarSamples {
                origin_x: 0,
                origin_y: 0,
                width: width / 2,
                height: if sampling == ChromaSampling::Cs420 {
                    height / 2
                } else {
                    height
                },
                samples: half,
            };
            let [restored, _] = reconstruct_chroma_444(
                &input,
                &input,
                ChromaReconstructionConfig {
                    sampling,
                    full_width: width,
                    full_height: height,
                    centering_x: jxr_math::sampling::DOWNSAMPLED_CENTERING.get(),
                    centering_y: jxr_math::sampling::DOWNSAMPLED_CENTERING.get(),
                },
            )
            .unwrap();
            // Away from the plane's edges a ramp survives exactly.
            for y in 4..(height as usize - 4) {
                for x in 4..(width as usize - 4) {
                    let index = y * width as usize + x;
                    assert_eq!(
                        restored.samples[index], plane[index],
                        "{sampling:?} {x},{y}"
                    );
                }
            }
            let worst = restored
                .samples
                .iter()
                .zip(&plane)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            assert!(worst <= 16, "{sampling:?}: worst edge error {worst}");
        }
    }

    #[test]
    fn subsampling_rejects_planes_it_cannot_halve() {
        assert!(subsample_chroma(&[0; 15], 5, ChromaSampling::Cs422).is_err());
        assert!(subsample_chroma(&[0; 12], 4, ChromaSampling::Cs420).is_err());
        assert!(subsample_chroma(&[0; 16], 4, ChromaSampling::Cs444).is_err());
        assert!(subsample_chroma(&[0; 15], 4, ChromaSampling::Cs422).is_err());
        assert!(subsample_chroma(&[], 4, ChromaSampling::Cs422).is_err());
    }
}
