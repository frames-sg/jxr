//! T.832 clause 9.9.8 overlap operators and full-resolution edge geometry.

use jxr_math::overlap::{inverse_overlap_filter_4, inverse_overlap_filter_4x4};

use super::ReconstructionError;

pub(super) fn apply_overlap(
    samples: &mut [i32],
    width: usize,
    height: usize,
    column_boundaries: &[usize],
    row_boundaries: &[usize],
    hard_boundaries: bool,
) -> Result<(), ReconstructionError> {
    let full_columns = [0, width];
    let full_rows = [0, height];
    let columns = if hard_boundaries {
        column_boundaries
    } else {
        &full_columns
    };
    let rows = if hard_boundaries {
        row_boundaries
    } else {
        &full_rows
    };
    for vertical in rows.windows(2) {
        for horizontal in columns.windows(2) {
            filter_region(
                samples,
                width,
                horizontal[0],
                horizontal[1],
                vertical[0],
                vertical[1],
            )?;
        }
    }
    Ok(())
}

fn filter_region(
    samples: &mut [i32],
    stride: usize,
    left: usize,
    right: usize,
    top: usize,
    bottom: usize,
) -> Result<(), ReconstructionError> {
    let mut y = top + 2;
    while y + 2 < bottom {
        let mut x = left + 2;
        while x + 2 < right {
            filter_window_4x4(samples, stride, x, y)?;
            x += 4;
        }
        y += 4;
    }
    filter_vertical_edges(samples, stride, left, right, top, bottom)?;
    filter_horizontal_edges(samples, stride, left, right, top, bottom)?;
    filter_corner(samples, stride, left, top)?;
    filter_corner(samples, stride, right - 2, top)?;
    filter_corner(samples, stride, left, bottom - 2)?;
    filter_corner(samples, stride, right - 2, bottom - 2)
}

fn filter_vertical_edges(
    samples: &mut [i32],
    stride: usize,
    left: usize,
    right: usize,
    top: usize,
    bottom: usize,
) -> Result<(), ReconstructionError> {
    let mut y = top + 2;
    while y + 2 < bottom {
        for x in [left, left + 1, right - 2, right - 1] {
            let indices = [
                x + y * stride,
                x + (y + 1) * stride,
                x + (y + 2) * stride,
                x + (y + 3) * stride,
            ];
            filter_indices_4(samples, indices)?;
        }
        y += 4;
    }
    Ok(())
}

fn filter_horizontal_edges(
    samples: &mut [i32],
    stride: usize,
    left: usize,
    right: usize,
    top: usize,
    bottom: usize,
) -> Result<(), ReconstructionError> {
    let mut x = left + 2;
    while x + 2 < right {
        for y in [top, top + 1, bottom - 2, bottom - 1] {
            let start = x + y * stride;
            filter_indices_4(samples, [start, start + 1, start + 2, start + 3])?;
        }
        x += 4;
    }
    Ok(())
}

fn filter_corner(
    samples: &mut [i32],
    stride: usize,
    x: usize,
    y: usize,
) -> Result<(), ReconstructionError> {
    filter_indices_4(
        samples,
        [
            x + y * stride,
            x + 1 + y * stride,
            x + (y + 1) * stride,
            x + 1 + (y + 1) * stride,
        ],
    )
}

fn filter_window_4x4(
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
    overlap_post_filter_4x4(&mut values)?;
    for row in 0..4 {
        samples[x + (y + row) * stride..x + (y + row) * stride + 4]
            .copy_from_slice(&values[row * 4..row * 4 + 4]);
    }
    Ok(())
}

fn filter_indices_4(samples: &mut [i32], indices: [usize; 4]) -> Result<(), ReconstructionError> {
    let mut values = indices.map(|index| samples[index]);
    overlap_post_filter_4(&mut values)?;
    for (index, value) in indices.into_iter().zip(values) {
        samples[index] = value;
    }
    Ok(())
}

fn overlap_post_filter_4x4(values: &mut [i32; 16]) -> Result<(), ReconstructionError> {
    inverse_overlap_filter_4x4(values)
        .map_err(|_| ReconstructionError::ArithmeticOverflow("overlap postfilter"))
}

fn overlap_post_filter_4(values: &mut [i32; 4]) -> Result<(), ReconstructionError> {
    inverse_overlap_filter_4(values)
        .map_err(|_| ReconstructionError::ArithmeticOverflow("overlap postfilter"))
}

#[cfg(test)]
mod tests {
    use super::{apply_overlap, overlap_post_filter_4, overlap_post_filter_4x4};

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
        apply_overlap(&mut soft, 8, 4, &[0, 4, 8], &[0, 4], false).unwrap();
        let mut hard = source;
        apply_overlap(&mut hard, 8, 4, &[0, 4, 8], &[0, 4], true).unwrap();
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
        apply_overlap(&mut samples, 16, 16, &[0, 16], &[0, 16], false).unwrap();
        assert_eq!(samples, vec![-128; 16 * 16]);
    }
}
