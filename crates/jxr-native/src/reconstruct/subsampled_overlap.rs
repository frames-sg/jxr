//! First-level overlap geometry for YUV420 and YUV422 chroma planes, in both
//! directions.

use jxr_math::{
    arithmetic::{checked_add, checked_sub},
    overlap::{
        forward_overlap_filter_2, forward_overlap_filter_2x2, inverse_overlap_filter_2,
        inverse_overlap_filter_2x2,
    },
};

use core::ops::Range;

use super::{
    ReconstructionError,
    lanes::{Postfilter, Prefilter},
    overlap::{OverlapPlane, filter_indices, filter_regions},
};

/// The operators applied over the subsampled first-level overlap schedule.
///
/// Unlike the full-resolution schedule, the steps of a region are not
/// independent. The corner residuals read samples that the boundary windows
/// rewrite, and in a region two samples wide they read each other. A region
/// is therefore an ordered list of steps: decoding runs it first to last, and
/// encoding runs it last to first with every step inverted.
pub(crate) trait SubsampledOverlapKernel {
    /// Whether this kernel undoes the decoder's schedule.
    const UNDOES_DECODER: bool;

    /// Operator for an interior two-by-two window.
    fn square(values: &mut [i32; 4]) -> Result<(), ReconstructionError>;

    /// Operator for a two-sample window along a region boundary.
    fn pair(values: &mut [i32; 2]) -> Result<(), ReconstructionError>;
}

/// Decoder postfilter of T.832 clause 9.9.8 for subsampled chroma.
impl SubsampledOverlapKernel for Postfilter {
    const UNDOES_DECODER: bool = false;

    #[inline]
    fn square(values: &mut [i32; 4]) -> Result<(), ReconstructionError> {
        inverse_overlap_filter_2x2(values)
            .map_err(|_| ReconstructionError::ArithmeticOverflow("subsampled overlap postfilter"))
    }

    #[inline]
    fn pair(values: &mut [i32; 2]) -> Result<(), ReconstructionError> {
        inverse_overlap_filter_2(values)
            .map_err(|_| ReconstructionError::ArithmeticOverflow("subsampled overlap postfilter"))
    }
}

/// Encoder prefilter: the exact inverse of the postfilter.
impl SubsampledOverlapKernel for Prefilter {
    const UNDOES_DECODER: bool = true;

    #[inline]
    fn square(values: &mut [i32; 4]) -> Result<(), ReconstructionError> {
        forward_overlap_filter_2x2(values)
            .map_err(|_| ReconstructionError::ArithmeticOverflow("subsampled overlap prefilter"))
    }

    #[inline]
    fn pair(values: &mut [i32; 2]) -> Result<(), ReconstructionError> {
        forward_overlap_filter_2(values)
            .map_err(|_| ReconstructionError::ArithmeticOverflow("subsampled overlap prefilter"))
    }
}

/// Apply the first-level overlap operators of `K` over a subsampled chroma
/// low-pass plane: [`apply_subsampled_first_overlap_rows`] over the strip
/// that covers it, which the tests compare strips with.
///
/// With hard boundaries each tile is filtered on its own; otherwise the whole
/// plane is one region. Regions are disjoint, and within a region `K` takes
/// the steps in the order that makes the two kernels exact inverses.
#[cfg(test)]
pub(crate) fn apply_subsampled_first_overlap<K: SubsampledOverlapKernel>(
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
    apply_subsampled_first_overlap_rows::<K>(samples, 0..height, &plane).map(|_| ())
}

/// Apply the first-level overlap operators of `K` over a strip of a
/// subsampled chroma low-pass plane.
///
/// `samples` holds the rows `held` of the plane and nothing above them. The
/// steps of a region fall into groups that stack down it without gaps and
/// share no sample: those of its first row, those of each pair of rows
/// after it, and those of its last row. This runs each group that lies
/// wholly within `held` and returns the row the last one ends at, or the
/// first row of `held` when it holds none, as
/// [`apply_overlap_rows`](super::overlap::apply_overlap_rows) does for the
/// full-resolution schedule. A strip that covers the plane is the whole
/// schedule.
pub(crate) fn apply_subsampled_first_overlap_rows<K: SubsampledOverlapKernel>(
    samples: &mut [i32],
    held: Range<usize>,
    plane: &OverlapPlane<'_>,
) -> Result<usize, ReconstructionError> {
    let (full_columns, full_rows) = ([0, plane.width], [0, plane.height]);
    let columns = filter_regions(
        plane.column_boundaries,
        &full_columns,
        plane.hard_boundaries,
    );
    let rows = filter_regions(plane.row_boundaries, &full_rows, plane.hard_boundaries);
    let mut finished = held.start;
    for vertical in rows.windows(2) {
        let (top, bottom) = (vertical[0], vertical[1]);
        if bottom <= held.start || top >= held.end {
            continue;
        }
        for horizontal in columns.windows(2) {
            let region = [horizontal[0], horizontal[1], top, bottom];
            let mut steps = strip_steps(plane.width, region, held.clone());
            if K::UNDOES_DECODER {
                steps.rev().try_for_each(|step| step.run::<K>(samples))?;
            } else {
                steps.try_for_each(|step| step.run::<K>(samples))?;
            }
        }
        finished = finished.max(groups(top, bottom, &held).finished);
    }
    Ok(finished)
}

/// One step of the decoder's schedule, addressing samples by plane index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    /// Interior two-by-two window.
    Square([usize; 4]),
    /// Two-sample window along a region boundary.
    Pair([usize; 2]),
    /// Adds `neighbour` to the region corner `corner`, or subtracts it.
    Corner {
        corner: usize,
        neighbour: usize,
        add: bool,
    },
}

impl Step {
    /// Performs this step for the decoder, or its inverse for a kernel that
    /// undoes the decoder.
    fn run<K: SubsampledOverlapKernel>(
        self,
        samples: &mut [i32],
    ) -> Result<(), ReconstructionError> {
        match self {
            Self::Square(indices) => filter_indices(samples, indices, K::square),
            Self::Pair(indices) => filter_indices(samples, indices, K::pair),
            Self::Corner {
                corner,
                neighbour,
                add,
            } => {
                samples[corner] = if add == K::UNDOES_DECODER {
                    checked_sub(samples[corner], samples[neighbour])
                } else {
                    checked_add(samples[corner], samples[neighbour])
                }
                .map_err(|_| {
                    ReconstructionError::ArithmeticOverflow("subsampled corner residual")
                })?;
                Ok(())
            }
        }
    }
}

/// The groups of one region's steps that a strip holds whole.
struct Groups {
    /// The region's first row, if held.
    first_row: Option<usize>,
    /// The lower row of each held pair of rows after the first.
    pairs: core::iter::StepBy<Range<usize>>,
    /// The region's last row, if held.
    last_row: Option<usize>,
    /// Row of the plane the last held group ends at, or zero for none.
    finished: usize,
}

/// The groups of the region of rows `top..bottom` within the rows `held`.
fn groups(top: usize, bottom: usize, held: &Range<usize>) -> Groups {
    // The pairs of rows are `(y - 1, y)` for `y` two rows apart from
    // `top + 2`, as the whole-plane schedule has them.
    let first = top + 2 + (held.start + 1).saturating_sub(top + 2).div_ceil(2) * 2;
    let end = bottom.min(held.end);
    let pairs = if first < end { first..end } else { 0..0 };
    let first_row = (top >= held.start && top < held.end).then_some(top);
    let last_row = (bottom > held.start && bottom <= held.end).then(|| bottom - 1);
    let finished = if last_row.is_some() {
        bottom
    } else if let Some(last) = pairs.clone().step_by(2).next_back() {
        last + 1
    } else if first_row.is_some() {
        top + 1
    } else {
        0
    };
    Groups {
        first_row,
        pairs: pairs.step_by(2),
        last_row,
        finished,
    }
}

/// The decoder's steps for the region `[left, right, top, bottom]`, in order.
#[cfg(test)]
fn region_steps(stride: usize, region: [usize; 4]) -> impl DoubleEndedIterator<Item = Step> {
    strip_steps(stride, region, 0..region[3])
}

/// The decoder's steps for the part of the region `[left, right, top, bottom]`
/// that a strip of the rows `held` holds whole, in order, addressing samples
/// from the first row of the strip.
///
/// The steps of a region fall into groups by row. In its first row, and
/// again in its last, each corner first loses its row neighbour, the windows
/// along the boundary are filtered, and each corner then regains the
/// neighbour's filtered value; a region two samples wide makes each corner
/// of a row the other's neighbour. Between them every pair of rows has its
/// interior windows and the windows along its two sides. The windows are
/// mutually disjoint and never contain a corner, and no group touches a
/// sample of another, so only the order within a group matters.
fn strip_steps(
    stride: usize,
    [left, right, top, bottom]: [usize; 4],
    held: Range<usize>,
) -> impl DoubleEndedIterator<Item = Step> {
    let origin = held.start;
    let at = move |x: usize, y: usize| x + (y - origin) * stride;
    let window_columns = move || (left + 2..right).step_by(2);
    let boundary_row = move |row: usize| {
        let corners = [
            (at(left, row), at(left + 1, row)),
            (at(right - 1, row), at(right - 2, row)),
        ];
        let residuals = move |add| {
            corners
                .into_iter()
                .map(move |(corner, neighbour)| Step::Corner {
                    corner,
                    neighbour,
                    add,
                })
        };
        let end_pairs = window_columns().map(move |x| Step::Pair([at(x - 1, row), at(x, row)]));
        residuals(false).chain(end_pairs).chain(residuals(true))
    };
    let row_pair = move |y: usize| {
        let squares = window_columns()
            .map(move |x| Step::Square([at(x - 1, y - 1), at(x, y - 1), at(x - 1, y), at(x, y)]));
        let side_pairs = [
            Step::Pair([at(left, y - 1), at(left, y)]),
            Step::Pair([at(right - 1, y - 1), at(right - 1, y)]),
        ];
        squares.chain(side_pairs)
    };
    let groups = groups(top, bottom, &held);
    groups
        .first_row
        .into_iter()
        .flat_map(boundary_row)
        .chain(groups.pairs.flat_map(row_pair))
        .chain(groups.last_row.into_iter().flat_map(boundary_row))
}

#[cfg(test)]
mod tests {
    use super::{
        OverlapPlane, Postfilter, Prefilter, ReconstructionError, Step,
        apply_subsampled_first_overlap, apply_subsampled_first_overlap_rows, region_steps,
    };
    use crate::entropy::encode::test_support::Lcg;

    /// Tile layouts of a 12-by-16 plane: 6-by-8 macroblocks of 4:2:0 chroma
    /// or 6-by-4 of 4:2:2, including tiles one macroblock wide and tall.
    const LAYOUTS: [(&[usize], &[usize]); 4] = [
        (&[0, 12], &[0, 16]),
        (&[0, 4, 12], &[0, 8, 16]),
        (&[0, 2, 4, 12], &[0, 2, 4, 16]),
        (&[0, 2, 10, 12], &[0, 4, 12, 16]),
    ];

    fn noise(seed: u64, length: usize) -> Vec<i32> {
        let mut state = seed;
        (0..length)
            .map(|_| {
                state = Lcg::advance(state);
                i32::try_from((state >> 40) % 8192).unwrap() - 4096
            })
            .collect()
    }

    #[test]
    fn zero_plane_is_stable_for_both_subsampled_geometries() {
        let mut yuv420 = [0; 16];
        apply_subsampled_first_overlap::<Postfilter>(&mut yuv420, 4, 4, &[0, 4], &[0, 4], false)
            .unwrap();
        assert_eq!(yuv420, [0; 16]);

        let mut yuv422 = [0; 32];
        apply_subsampled_first_overlap::<Postfilter>(&mut yuv422, 4, 8, &[0, 4], &[0, 8], false)
            .unwrap();
        assert_eq!(yuv422, [0; 32]);
    }

    #[test]
    fn postfilter_has_stable_exact_results() {
        let source: Vec<i32> = (0..48).map(|index| index * 37 % 101 - 50).collect();
        let filtered = |width, height, columns: &[usize], rows: &[usize], hard| {
            let mut plane = source[..width * height].to_vec();
            apply_subsampled_first_overlap::<Postfilter>(
                &mut plane, width, height, columns, rows, hard,
            )
            .unwrap();
            plane
        };
        assert_eq!(
            filtered(4, 4, &[0, 4], &[0, 4], false),
            [
                -39, -2, 21, -43, 20, 23, -20, -19, 48, -10, 6, -50, -28, 9, -28, 9
            ]
        );
        assert_eq!(
            filtered(6, 8, &[0, 6], &[0, 8], false),
            [
                -39, -2, 21, -49, -25, 12, -41, 24, 41, -14, 23, -70, -27, 24, -20, 5, 42, -56, 28,
                35, -7, 17, -46, -1, 40, -24, -8, 39, -26, 11, 38, -4, 33, -50, -12, 9, -8, 15, 52,
                -30, 7, -38, -29, 8, -29, 18, 41, -23
            ]
        );
        assert_eq!(
            filtered(6, 8, &[0, 2, 6], &[0, 4, 8], true),
            [
                -13, 11, 15, -49, -25, 12, -41, 22, 28, -14, 23, -70, -27, 35, -19, 5, 42, -56, 47,
                131, -40, -3, -40, -3, -34, -132, -7, 30, -6, 31, 38, -12, 49, -50, -12, 9, -8, 0,
                61, -30, 7, -38, 26, 89, -19, 18, 41, -23
            ]
        );
        // One 4:2:0 macroblock: each top corner is the other's neighbour.
        assert_eq!(
            filtered(2, 2, &[0, 2], &[0, 2], false),
            [-13, 11, -40, -144]
        );
    }

    #[test]
    fn prefilter_and_postfilter_are_exact_inverses_for_every_tile_layout() {
        let (width, height) = (12, 16);
        let source = noise(21, width * height);
        for (columns, rows) in LAYOUTS {
            for hard in [false, true] {
                let mut samples = source.clone();
                apply_subsampled_first_overlap::<Prefilter>(
                    &mut samples,
                    width,
                    height,
                    columns,
                    rows,
                    hard,
                )
                .unwrap();
                assert_ne!(samples, source, "the prefilter changes the plane");
                apply_subsampled_first_overlap::<Postfilter>(
                    &mut samples,
                    width,
                    height,
                    columns,
                    rows,
                    hard,
                )
                .unwrap();
                assert_eq!(samples, source, "post of pre, {columns:?} {rows:?} {hard}");

                apply_subsampled_first_overlap::<Postfilter>(
                    &mut samples,
                    width,
                    height,
                    columns,
                    rows,
                    hard,
                )
                .unwrap();
                apply_subsampled_first_overlap::<Prefilter>(
                    &mut samples,
                    width,
                    height,
                    columns,
                    rows,
                    hard,
                )
                .unwrap();
                assert_eq!(samples, source, "pre of post, {columns:?} {rows:?} {hard}");
            }
        }
    }

    /// A plane filtered in strips, each begun at the row the last one
    /// finished, is the plane filtered whole, in both directions.
    #[test]
    fn strips_of_any_height_filter_as_the_whole_plane_does() {
        let (width, height) = (12, 16);
        let source = noise(23, width * height);
        for (columns, rows) in LAYOUTS {
            for hard in [false, true] {
                for step in [1_usize, 2, 3, 5, 16] {
                    let plane = OverlapPlane {
                        width,
                        height,
                        column_boundaries: columns,
                        row_boundaries: rows,
                        hard_boundaries: hard,
                    };
                    let mut whole = source.clone();
                    apply_subsampled_first_overlap::<Postfilter>(
                        &mut whole, width, height, columns, rows, hard,
                    )
                    .unwrap();
                    let mut strips = source.clone();
                    let (mut finished, mut end) = (0, 0);
                    while end < height {
                        end = (end + step).min(height);
                        finished = apply_subsampled_first_overlap_rows::<Postfilter>(
                            &mut strips[finished * width..end * width],
                            finished..end,
                            &plane,
                        )
                        .unwrap();
                        assert_eq!(
                            strips[..finished * width],
                            whole[..finished * width],
                            "rows above {finished}, {rows:?} hard {hard} step {step}"
                        );
                    }
                    assert_eq!(finished, height);
                    let (mut finished, mut end) = (0, 0);
                    while end < height {
                        end = (end + step).min(height);
                        finished = apply_subsampled_first_overlap_rows::<Prefilter>(
                            &mut strips[finished * width..end * width],
                            finished..end,
                            &plane,
                        )
                        .unwrap();
                    }
                    assert_eq!(strips, source, "{rows:?} hard {hard} step {step}");
                }
            }
        }
    }

    #[test]
    fn single_macroblock_regions_invert_exactly() {
        for (width, height) in [(2, 2), (2, 4)] {
            let source = noise(22, width * height);
            let mut samples = source.clone();
            apply_subsampled_first_overlap::<Prefilter>(
                &mut samples,
                width,
                height,
                &[0, width],
                &[0, height],
                false,
            )
            .unwrap();
            assert_ne!(samples, source);
            apply_subsampled_first_overlap::<Postfilter>(
                &mut samples,
                width,
                height,
                &[0, width],
                &[0, height],
                false,
            )
            .unwrap();
            assert_eq!(samples, source, "{width}x{height}");
        }
    }

    /// Samples a step writes and samples it reads without writing.
    fn footprint(step: Step) -> (Vec<usize>, Vec<usize>) {
        match step {
            Step::Square(indices) => (indices.to_vec(), Vec::new()),
            Step::Pair(indices) => (indices.to_vec(), Vec::new()),
            Step::Corner {
                corner, neighbour, ..
            } => (vec![corner], vec![neighbour]),
        }
    }

    #[test]
    fn windows_are_disjoint_but_corner_residuals_depend_on_step_order() {
        for region in [[0, 8, 0, 8], [2, 8, 4, 12], [0, 2, 0, 4], [4, 6, 2, 4]] {
            let steps: Vec<Step> = region_steps(16, region).collect();
            let mut written = vec![0_u32; 16 * 16];
            for &step in &steps {
                if !matches!(step, Step::Corner { .. }) {
                    for index in footprint(step).0 {
                        written[index] += 1;
                    }
                }
            }
            assert!(
                written.iter().all(|&count| count <= 1),
                "windows overlap in {region:?}"
            );

            // Every corner neighbour is rewritten between the two residual
            // steps, by a boundary window or by the other corner's residual.
            let residuals: Vec<Step> = steps
                .iter()
                .copied()
                .filter(|step| matches!(step, Step::Corner { .. }))
                .collect();
            assert_eq!(residuals.len(), 8, "{region:?}");
            for &residual in &residuals {
                let neighbour = footprint(residual).1[0];
                let rewritten = steps
                    .iter()
                    .filter(|&&step| step != residual)
                    .any(|&step| footprint(step).0.contains(&neighbour));
                assert!(rewritten, "{residual:?} in {region:?}");
            }
        }
    }

    #[test]
    fn region_steps_reverse_to_the_same_schedule() {
        let forward: Vec<Step> = region_steps(8, [0, 6, 0, 8]).collect();
        let mut backward: Vec<Step> = region_steps(8, [0, 6, 0, 8]).rev().collect();
        backward.reverse();
        assert_eq!(forward, backward);
        // 4 + 4 residuals, 2 * 3 squares, 2 * 3 side pairs, 2 * 2 end pairs.
        assert_eq!(forward.len(), 24);
    }

    #[test]
    fn overflow_is_reported_in_both_directions() {
        let mut samples = [i32::MAX, i32::MIN, 0, 0];
        assert_eq!(
            apply_subsampled_first_overlap::<Postfilter>(
                &mut samples,
                2,
                2,
                &[0, 2],
                &[0, 2],
                false
            ),
            Err(ReconstructionError::ArithmeticOverflow(
                "subsampled corner residual"
            ))
        );
        let mut samples = vec![i32::MAX; 16];
        assert_eq!(
            apply_subsampled_first_overlap::<Prefilter>(
                &mut samples,
                4,
                4,
                &[0, 4],
                &[0, 4],
                false
            ),
            Err(ReconstructionError::ArithmeticOverflow(
                "subsampled overlap prefilter"
            ))
        );
    }
}
