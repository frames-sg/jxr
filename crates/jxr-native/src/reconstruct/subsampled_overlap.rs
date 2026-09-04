//! First-level overlap geometry for YUV420 and YUV422 chroma planes.

use jxr_math::{
    arithmetic::{checked_add, checked_sub},
    overlap::{inverse_overlap_filter_2, inverse_overlap_filter_2x2},
};

use super::ReconstructionError;

pub(super) fn apply_subsampled_first_overlap(
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
    subtract_corner_residuals(samples, stride, left, right, top, bottom)?;
    for y in (top + 2..bottom).step_by(2) {
        for x in (left + 2..right).step_by(2) {
            filter_square(samples, stride, x, y)?;
        }
    }
    for y in (top + 2..bottom).step_by(2) {
        filter_pair(samples, [left + (y - 1) * stride, left + y * stride])?;
        filter_pair(
            samples,
            [right - 1 + (y - 1) * stride, right - 1 + y * stride],
        )?;
    }
    for x in (left + 2..right).step_by(2) {
        filter_pair(samples, [x - 1 + top * stride, x + top * stride])?;
        filter_pair(
            samples,
            [x - 1 + (bottom - 1) * stride, x + (bottom - 1) * stride],
        )?;
    }
    add_corner_residuals(samples, stride, left, right, top, bottom)
}

fn subtract_corner_residuals(
    samples: &mut [i32],
    stride: usize,
    left: usize,
    right: usize,
    top: usize,
    bottom: usize,
) -> Result<(), ReconstructionError> {
    update_corner(samples, left + top * stride, left + 1 + top * stride, false)?;
    update_corner(
        samples,
        right - 1 + top * stride,
        right - 2 + top * stride,
        false,
    )?;
    update_corner(
        samples,
        left + (bottom - 1) * stride,
        left + 1 + (bottom - 1) * stride,
        false,
    )?;
    update_corner(
        samples,
        right - 1 + (bottom - 1) * stride,
        right - 2 + (bottom - 1) * stride,
        false,
    )
}

fn add_corner_residuals(
    samples: &mut [i32],
    stride: usize,
    left: usize,
    right: usize,
    top: usize,
    bottom: usize,
) -> Result<(), ReconstructionError> {
    update_corner(samples, left + top * stride, left + 1 + top * stride, true)?;
    update_corner(
        samples,
        right - 1 + top * stride,
        right - 2 + top * stride,
        true,
    )?;
    update_corner(
        samples,
        left + (bottom - 1) * stride,
        left + 1 + (bottom - 1) * stride,
        true,
    )?;
    update_corner(
        samples,
        right - 1 + (bottom - 1) * stride,
        right - 2 + (bottom - 1) * stride,
        true,
    )
}

fn update_corner(
    samples: &mut [i32],
    destination: usize,
    source: usize,
    add: bool,
) -> Result<(), ReconstructionError> {
    samples[destination] = if add {
        checked_add(samples[destination], samples[source])
    } else {
        checked_sub(samples[destination], samples[source])
    }
    .map_err(|_| ReconstructionError::ArithmeticOverflow("subsampled corner residual"))?;
    Ok(())
}

fn filter_square(
    samples: &mut [i32],
    stride: usize,
    x: usize,
    y: usize,
) -> Result<(), ReconstructionError> {
    let indices = [
        x - 1 + (y - 1) * stride,
        x + (y - 1) * stride,
        x - 1 + y * stride,
        x + y * stride,
    ];
    let mut values = indices.map(|index| samples[index]);
    inverse_overlap_filter_2x2(&mut values)
        .map_err(|_| ReconstructionError::ArithmeticOverflow("subsampled overlap postfilter"))?;
    for (index, value) in indices.into_iter().zip(values) {
        samples[index] = value;
    }
    Ok(())
}

fn filter_pair(samples: &mut [i32], indices: [usize; 2]) -> Result<(), ReconstructionError> {
    let mut values = indices.map(|index| samples[index]);
    inverse_overlap_filter_2(&mut values)
        .map_err(|_| ReconstructionError::ArithmeticOverflow("subsampled overlap postfilter"))?;
    for (index, value) in indices.into_iter().zip(values) {
        samples[index] = value;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::apply_subsampled_first_overlap;

    #[test]
    fn zero_plane_is_stable_for_both_subsampled_geometries() {
        let mut yuv420 = [0; 16];
        apply_subsampled_first_overlap(&mut yuv420, 4, 4, &[0, 4], &[0, 4], false).unwrap();
        assert_eq!(yuv420, [0; 16]);

        let mut yuv422 = [0; 32];
        apply_subsampled_first_overlap(&mut yuv422, 4, 8, &[0, 4], &[0, 8], false).unwrap();
        assert_eq!(yuv422, [0; 32]);
    }
}
