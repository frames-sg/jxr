//! T.832 clause 9.9.8 overlap operators and full-resolution edge geometry.

use core::ops::Range;

use fearless_simd::Level;
use jxr_math::overlap::{forward_overlap_filter_4, inverse_overlap_filter_4};

use super::{
    ReconstructionError,
    lanes::{BlockKernel, Postfilter, Prefilter, SimdReporter, filter_window_rows},
};
use crate::executor::Executor;

/// The operators applied over one overlap window schedule.
///
/// Decoding and encoding visit identical, disjoint windows; only the operator
/// differs. Sharing the schedule keeps the two directions exact inverses for
/// every tile layout, and gives both the same vector and thread paths.
pub(crate) trait OverlapKernel {
    /// Operator for an interior 4-by-4 window straddling four blocks.
    type Window: BlockKernel;

    /// Operator for a four-sample edge strip or two-by-two corner.
    fn edge(values: &mut [i32; 4]) -> Result<(), ReconstructionError>;
}

/// Decoder postfilter of T.832 clause 9.9.8.
impl OverlapKernel for Postfilter {
    type Window = Self;

    #[inline]
    fn edge(values: &mut [i32; 4]) -> Result<(), ReconstructionError> {
        inverse_overlap_filter_4(values)
            .map_err(|_| ReconstructionError::ArithmeticOverflow("overlap postfilter"))
    }
}

/// Encoder prefilter: the exact inverse of the postfilter.
impl OverlapKernel for Prefilter {
    type Window = Self;

    #[inline]
    fn edge(values: &mut [i32; 4]) -> Result<(), ReconstructionError> {
        forward_overlap_filter_4(values)
            .map_err(|_| ReconstructionError::ArithmeticOverflow("overlap prefilter"))
    }
}

/// The boundaries along one axis of the regions an overlap schedule filters
/// on its own, none of whose operators reaches into the next: the tiles
/// where their boundaries are hard, the whole plane where they are soft.
pub(crate) fn filter_regions<'a, T>(
    tile_boundaries: &'a [T],
    plane: &'a [T; 2],
    hard_boundaries: bool,
) -> &'a [T] {
    if hard_boundaries {
        tile_boundaries
    } else {
        plane
    }
}

/// Apply the overlap operators of `K` over a plane: [`apply_overlap_rows`]
/// over the strip that covers it, which the tests compare strips with.
///
/// With hard boundaries each tile is filtered on its own; otherwise the whole
/// plane is one region. Every window is visited once and the windows are
/// disjoint, so running the inverse kernel over the same schedule undoes this
/// exactly, whatever order the windows are taken in.
#[cfg(test)]
pub(crate) fn apply_overlap<K: OverlapKernel>(
    level: Level,
    samples: &mut [i32],
    width: usize,
    height: usize,
    column_boundaries: &[usize],
    row_boundaries: &[usize],
    hard_boundaries: bool,
) -> Result<(), ReconstructionError> {
    let plane = OverlapPlane {
        width,
        height,
        column_boundaries,
        row_boundaries,
        hard_boundaries,
    };
    apply_overlap_rows::<K>(level, samples, 0..height, &plane, Executor::DEFAULT, None).map(|_| ())
}

/// The geometry an overlap schedule is laid out over: a plane and its tiles.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OverlapPlane<'a> {
    /// Plane width in samples, which is also the stride of a strip of it.
    pub(crate) width: usize,
    /// Plane height in samples.
    pub(crate) height: usize,
    /// Tile column boundaries, including zero and the width.
    pub(crate) column_boundaries: &'a [usize],
    /// Tile row boundaries, including zero and the height.
    pub(crate) row_boundaries: &'a [usize],
    /// Whether each tile is filtered on its own.
    pub(crate) hard_boundaries: bool,
}

/// Apply the overlap operators of `K` over a strip of a plane.
///
/// `samples` holds the rows `held` of the plane and nothing above them. The
/// windows of a region stack down it without gaps: the edge strip of its
/// first two rows, the windows four rows tall that follow, and the edge
/// strip of its last two rows. This runs each of them that lies wholly
/// within `held`, and returns the row the last one ends at, or the first
/// row of `held` when it holds none. No window straddles that row, so the
/// rows above it are finished, and a caller streaming down the plane
/// passes it as the first row of its next strip, with the rows that have
/// arrived since. A strip that covers the plane is the whole schedule.
///
/// A strip may also begin within a window, as a halo above the rows a
/// caller is after does. That window is left out, with whatever it would
/// have changed above it.
///
/// The interior windows of a strip large enough to share out run through
/// `executor`, by rows of windows.
pub(crate) fn apply_overlap_rows<K: OverlapKernel>(
    level: Level,
    samples: &mut [i32],
    held: Range<usize>,
    plane: &OverlapPlane<'_>,
    executor: Executor,
    used_simd: Option<&SimdReporter>,
) -> Result<usize, ReconstructionError> {
    let width = plane.width;
    let (full_columns, full_rows) = ([0, width], [0, plane.height]);
    let columns = filter_regions(
        plane.column_boundaries,
        &full_columns,
        plane.hard_boundaries,
    );
    let rows = filter_regions(plane.row_boundaries, &full_rows, plane.hard_boundaries);
    // Interior windows of block-aligned regions use the four-lane kernel.
    let aligned = columns.iter().chain(rows).all(|boundary| boundary % 4 == 0);
    let mut finished = held.start;
    for vertical in rows.windows(2) {
        let (top, bottom) = (vertical[0], vertical[1]);
        if bottom <= held.start || top >= held.end {
            continue;
        }
        let strip = RegionStrip::new(top, bottom, &held);
        if aligned && !strip.windows.is_empty() {
            let windows = &mut samples[strip.windows.start * width..strip.windows.end * width];
            filter_window_rows::<K::Window>(level, windows, width, columns, executor, used_simd)?;
        }
        for horizontal in columns.windows(2) {
            let region = [horizontal[0], horizontal[1]];
            if !aligned {
                filter_interior::<K>(samples, width, region, &strip.windows)?;
            }
            filter_region_edges::<K>(samples, width, region, &strip)?;
        }
        finished = finished.max(strip.finished);
    }
    Ok(finished)
}

/// The windows of one region that a strip holds whole, by row of the strip.
#[derive(Debug, Clone)]
struct RegionStrip {
    /// First row of the edge strip of the region's first two rows, if held.
    top_edge: Option<usize>,
    /// Rows of the held windows four rows tall, which begin two rows below
    /// the top of the region.
    windows: Range<usize>,
    /// First row of the edge strip of the region's last two rows, if held.
    bottom_edge: Option<usize>,
    /// Row of the plane the last held window ends at, or zero for none.
    finished: usize,
}

impl RegionStrip {
    /// The windows of the region `top..bottom` within the rows `held`.
    fn new(top: usize, bottom: usize, held: &Range<usize>) -> Self {
        // A window starts at `top + 2 + 4k` and, as the whole-plane schedule
        // has it, only while two rows of the region remain below its start.
        let first = top + 2 + (held.start.saturating_sub(top + 2)).div_ceil(4) * 4;
        let mut end = first;
        while end + 2 < bottom && end + 4 <= held.end {
            end += 4;
        }
        let top_edge = (top >= held.start && top + 2 <= held.end).then(|| top - held.start);
        let bottom_edge =
            (bottom >= held.start + 2 && bottom <= held.end).then(|| bottom - 2 - held.start);
        let finished = if bottom_edge.is_some() {
            bottom
        } else if end > first {
            end
        } else if top_edge.is_some() {
            top + 2
        } else {
            0
        };
        let windows = if end > first {
            first - held.start..end - held.start
        } else {
            0..0
        };
        Self {
            top_edge,
            windows,
            bottom_edge,
            finished,
        }
    }
}

/// Filters the interior windows of the columns `left..right` that start in
/// the rows `windows`.
fn filter_interior<K: OverlapKernel>(
    samples: &mut [i32],
    stride: usize,
    [left, right]: [usize; 2],
    windows: &Range<usize>,
) -> Result<(), ReconstructionError> {
    for y in windows.clone().step_by(4) {
        let mut x = left + 2;
        while x + 2 < right {
            filter_window_4x4::<K>(samples, stride, x, y)?;
            x += 4;
        }
    }
    Ok(())
}

/// Filters the edge strips and corners of the columns `left..right` of a
/// region that `strip` holds.
fn filter_region_edges<K: OverlapKernel>(
    samples: &mut [i32],
    stride: usize,
    [left, right]: [usize; 2],
    strip: &RegionStrip,
) -> Result<(), ReconstructionError> {
    filter_vertical_edges::<K>(samples, stride, [left, right], &strip.windows)?;
    for y in strip.top_edge.into_iter().chain(strip.bottom_edge) {
        filter_horizontal_edge::<K>(samples, stride, [left, right], y)?;
        filter_corner::<K>(samples, stride, left, y)?;
        filter_corner::<K>(samples, stride, right - 2, y)?;
    }
    Ok(())
}

fn filter_vertical_edges<K: OverlapKernel>(
    samples: &mut [i32],
    stride: usize,
    [left, right]: [usize; 2],
    windows: &Range<usize>,
) -> Result<(), ReconstructionError> {
    for y in windows.clone().step_by(4) {
        for x in [left, left + 1, right - 2, right - 1] {
            let indices = [
                x + y * stride,
                x + (y + 1) * stride,
                x + (y + 2) * stride,
                x + (y + 3) * stride,
            ];
            filter_indices(samples, indices, K::edge)?;
        }
    }
    Ok(())
}

/// Filters the edge strip of the two rows from `y` of the columns `left..right`.
fn filter_horizontal_edge<K: OverlapKernel>(
    samples: &mut [i32],
    stride: usize,
    [left, right]: [usize; 2],
    y: usize,
) -> Result<(), ReconstructionError> {
    let mut x = left + 2;
    while x + 2 < right {
        for y in [y, y + 1] {
            let start = x + y * stride;
            filter_indices(samples, [start, start + 1, start + 2, start + 3], K::edge)?;
        }
        x += 4;
    }
    Ok(())
}

fn filter_corner<K: OverlapKernel>(
    samples: &mut [i32],
    stride: usize,
    x: usize,
    y: usize,
) -> Result<(), ReconstructionError> {
    filter_indices(
        samples,
        [
            x + y * stride,
            x + 1 + y * stride,
            x + (y + 1) * stride,
            x + 1 + (y + 1) * stride,
        ],
        K::edge,
    )
}

fn filter_window_4x4<K: OverlapKernel>(
    samples: &mut [i32],
    stride: usize,
    x: usize,
    y: usize,
) -> Result<(), ReconstructionError> {
    let mut values = [0; 16];
    for row in 0..4 {
        values[row * 4..row * 4 + 4]
            .copy_from_slice(&samples[x + (y + row) * stride..x + (y + row) * stride + 4]);
    }
    K::Window::checked(&mut values)?;
    for row in 0..4 {
        samples[x + (y + row) * stride..x + (y + row) * stride + 4]
            .copy_from_slice(&values[row * 4..row * 4 + 4]);
    }
    Ok(())
}

/// Applies `operation` to the samples `indices` of `samples`, gathered in
/// that order and written back to them: an edge strip, corner or pair of
/// the overlap filters.
pub(super) fn filter_indices<const N: usize>(
    samples: &mut [i32],
    indices: [usize; N],
    operation: impl FnOnce(&mut [i32; N]) -> Result<(), ReconstructionError>,
) -> Result<(), ReconstructionError> {
    let mut values = indices.map(|index| samples[index]);
    operation(&mut values)?;
    for (index, value) in indices.into_iter().zip(values) {
        samples[index] = value;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use fearless_simd::Level;

    use super::{
        BlockKernel, Executor, OverlapKernel, OverlapPlane, Postfilter, Prefilter,
        ReconstructionError, apply_overlap, apply_overlap_rows,
    };
    use crate::entropy::encode::test_support::Lcg;

    fn overlap_post_filter_4(values: &mut [i32; 4]) -> Result<(), ReconstructionError> {
        Postfilter::edge(values)
    }

    fn overlap_post_filter_4x4(values: &mut [i32; 16]) -> Result<(), ReconstructionError> {
        Postfilter::checked(values)
    }

    #[test]
    fn forward_overlap_undoes_the_postfilter_for_every_tile_layout() {
        let (width, height) = (48_usize, 32_usize);
        let mut state = 7_u64;
        let source: Vec<i32> = (0..width * height)
            .map(|_| {
                state = Lcg::advance(state);
                ((state >> 40) % 4096) as i32 - 2048
            })
            .collect();
        let layouts: [(&[usize], &[usize]); 3] = [
            (&[0, 48], &[0, 32]),
            (&[0, 16, 48], &[0, 16, 32]),
            // A low-pass plane has four samples per macroblock, so its tile
            // boundaries are multiples of four rather than sixteen.
            (&[0, 4, 48], &[0, 12, 32]),
        ];
        for (columns, rows) in layouts {
            for hard in [false, true] {
                let mut samples = source.clone();
                apply_overlap::<Prefilter>(
                    Level::new(),
                    &mut samples,
                    width,
                    height,
                    columns,
                    rows,
                    hard,
                )
                .unwrap();
                assert_ne!(samples, source, "the prefilter changes the plane");
                apply_overlap::<Postfilter>(
                    Level::new(),
                    &mut samples,
                    width,
                    height,
                    columns,
                    rows,
                    hard,
                )
                .unwrap();
                assert_eq!(samples, source, "columns {columns:?} hard {hard}");
            }
        }
    }

    /// A plane filtered in strips, each begun at the row the last one
    /// finished and ended wherever its rows ran out, is the plane filtered
    /// whole, in both directions and for every tile layout.
    #[test]
    fn strips_of_any_height_filter_as_the_whole_plane_does() {
        let (width, height) = (48_usize, 32_usize);
        let mut state = 11_u64;
        let source: Vec<i32> = (0..width * height)
            .map(|_| {
                state = Lcg::advance(state);
                ((state >> 40) % 4096) as i32 - 2048
            })
            .collect();
        let layouts: [(&[usize], &[usize]); 3] = [
            (&[0, 48], &[0, 32]),
            (&[0, 16, 48], &[0, 16, 32]),
            (&[0, 4, 48], &[0, 4, 12, 32]),
        ];
        for (columns, rows) in layouts {
            for hard in [false, true] {
                for step in [1_usize, 3, 4, 7, 16, 32] {
                    let plane = OverlapPlane {
                        width,
                        height,
                        column_boundaries: columns,
                        row_boundaries: rows,
                        hard_boundaries: hard,
                    };
                    let mut whole = source.clone();
                    apply_overlap::<Postfilter>(
                        Level::new(),
                        &mut whole,
                        width,
                        height,
                        columns,
                        rows,
                        hard,
                    )
                    .unwrap();
                    let mut strips = source.clone();
                    let mut finished = 0;
                    let mut end = 0;
                    while end < height {
                        end = (end + step).min(height);
                        finished = apply_overlap_rows::<Postfilter>(
                            Level::new(),
                            &mut strips[finished * width..end * width],
                            finished..end,
                            &plane,
                            Executor::Sequential,
                            None,
                        )
                        .unwrap();
                        assert_eq!(
                            strips[..finished * width],
                            whole[..finished * width],
                            "rows above {finished}, {rows:?} hard {hard} step {step}"
                        );
                    }
                    assert_eq!(finished, height);
                    // The prefilter runs the same windows, so it undoes them
                    // strip by strip as well.
                    let (mut finished, mut end) = (0, 0);
                    while end < height {
                        end = (end + step).min(height);
                        finished = apply_overlap_rows::<Prefilter>(
                            Level::new(),
                            &mut strips[finished * width..end * width],
                            finished..end,
                            &plane,
                            Executor::Sequential,
                            None,
                        )
                        .unwrap();
                    }
                    assert_eq!(strips, source, "{rows:?} hard {hard} step {step}");
                }
            }
        }
    }

    #[test]
    fn overlap_operators_preserve_zero() {
        let mut four = [0; 4];
        overlap_post_filter_4(&mut four).unwrap();
        assert_eq!(four, [0; 4]);
        let mut sixteen = [0; 16];
        overlap_post_filter_4x4(&mut sixteen).unwrap();
        assert_eq!(sixteen, [0; 16]);
    }

    #[test]
    fn four_point_operator_has_stable_exact_result() {
        let mut values = [1, 2, 3, 4];
        overlap_post_filter_4(&mut values).unwrap();
        assert_eq!(values, [2, 1, 4, 5]);
    }

    #[test]
    fn hard_boundary_uses_independent_edge_operators() {
        let source = [
            1, 1, 1, 1, 9, 9, 9, 9, 1, 1, 1, 1, 9, 9, 9, 9, 1, 1, 1, 1, 9, 9, 9, 9, 1, 1, 1, 1, 9,
            9, 9, 9,
        ];
        let mut soft = source;
        apply_overlap::<Postfilter>(Level::new(), &mut soft, 8, 4, &[0, 4, 8], &[0, 4], false)
            .unwrap();
        let mut hard = source;
        apply_overlap::<Postfilter>(Level::new(), &mut hard, 8, 4, &[0, 4, 8], &[0, 4], true)
            .unwrap();
        assert_eq!(
            soft,
            [
                1, 1, 3, 6, 8, 11, 11, 12, 1, 1, 3, 6, 8, 11, 13, 14, 1, 1, 3, 6, 8, 11, 11, 12, 1,
                1, 3, 6, 8, 11, 13, 14,
            ]
        );
        assert_eq!(
            hard,
            [
                1, 1, 1, 1, 11, 12, 11, 12, 1, 1, 1, 1, 13, 14, 13, 14, 1, 1, 1, 1, 11, 12, 11, 12,
                1, 1, 1, 1, 13, 14, 13, 14,
            ]
        );
    }

    #[test]
    fn full_plane_operators_cover_the_two_sample_image_edges() {
        const ROWS: [[i32; 16]; 4] = [
            [
                -88, -88, -88, -88, -87, -87, -88, -88, -87, -87, -88, -88, -87, -87, -88, -88,
            ],
            [
                -87, -87, -88, -88, -87, -87, -88, -88, -87, -87, -88, -88, -87, -87, -87, -87,
            ],
            [
                -88, -88, -87, -87, -88, -88, -87, -87, -88, -88, -87, -87, -88, -88, -88, -88,
            ],
            [
                -87, -87, -88, -88, -88, -88, -88, -88, -88, -88, -88, -88, -88, -88, -87, -87,
            ],
        ];
        const ROW_ORDER: [usize; 16] = [0, 1, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 0, 1];
        let mut samples = Vec::with_capacity(16 * 16);
        for row in ROW_ORDER {
            samples.extend_from_slice(&ROWS[row]);
        }
        apply_overlap::<Postfilter>(
            Level::new(),
            &mut samples,
            16,
            16,
            &[0, 16],
            &[0, 16],
            false,
        )
        .unwrap();
        assert_eq!(samples, vec![-128; 16 * 16]);
    }
}
