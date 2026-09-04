//! Exact inverse overlap-filter arithmetic.

use crate::{
    arithmetic::{MathError, checked_add, checked_mul, checked_sub},
    transform::inverse_hadamard_2x2,
};

/// Apply the exact four-sample JPEG XR inverse overlap postfilter in place.
pub fn inverse_overlap_filter_4(values: &mut [i32; 4]) -> Result<(), MathError> {
    values[0] = checked_add(values[0], values[3])?;
    values[1] = checked_add(values[1], values[2])?;
    values[3] = checked_sub(values[3], checked_add(values[0], 1)? >> 1)?;
    values[2] = checked_sub(values[2], checked_add(values[1], 1)? >> 1)?;
    let mut pair = [values[0], values[3]];
    inverse_scale(&mut pair)?;
    [values[0], values[3]] = pair;
    pair = [values[1], values[2]];
    inverse_scale(&mut pair)?;
    [values[1], values[2]] = pair;
    values[0] = checked_add(values[0], checked_add(checked_mul(values[3], 3)?, 4)? >> 3)?;
    values[1] = checked_add(values[1], checked_add(checked_mul(values[2], 3)?, 4)? >> 3)?;
    values[3] = checked_sub(values[3], values[0] >> 1)?;
    values[2] = checked_sub(values[2], values[1] >> 1)?;
    values[0] = checked_add(values[0], values[3])?;
    values[1] = checked_add(values[1], values[2])?;
    values[3] = checked_neg(values[3])?;
    values[2] = checked_neg(values[2])?;
    pair = [values[2], values[3]];
    inverse_rotate(&mut pair)?;
    [values[2], values[3]] = pair;
    values[3] = checked_add(values[3], checked_add(values[0], 1)? >> 1)?;
    values[2] = checked_add(values[2], checked_add(values[1], 1)? >> 1)?;
    values[0] = checked_sub(values[0], values[3])?;
    values[1] = checked_sub(values[1], values[2])?;
    Ok(())
}

/// Apply the exact 4-by-4 JPEG XR inverse overlap postfilter in place.
pub fn inverse_overlap_filter_4x4(values: &mut [i32; 16]) -> Result<(), MathError> {
    for indices in [[0, 3, 12, 15], [1, 2, 13, 14], [4, 7, 8, 11], [5, 6, 9, 10]] {
        apply_group(values, indices, |group| inverse_hadamard_2x2(group, 0))?;
    }
    for indices in [[13, 12], [9, 8], [7, 3], [6, 2]] {
        apply_pair(values, indices, inverse_rotate)?;
    }
    apply_group(values, [10, 11, 14, 15], inverse_todd_odd_post)?;
    for indices in [[0, 15], [1, 14], [4, 11], [5, 10]] {
        apply_pair(values, indices, inverse_scale)?;
    }
    for indices in [[0, 3, 12, 15], [1, 2, 13, 14], [4, 7, 8, 11], [5, 6, 9, 10]] {
        apply_group(values, indices, inverse_hadamard_post)?;
    }
    Ok(())
}

/// Apply the exact 2-by-2 JPEG XR inverse overlap postfilter in place.
///
/// This is the first-level interior operator for subsampled chroma.
pub fn inverse_overlap_filter_2x2(values: &mut [i32; 4]) -> Result<(), MathError> {
    values[0] = checked_add(values[0], values[3])?;
    values[1] = checked_add(values[1], values[2])?;
    values[3] = checked_sub(values[3], checked_add(values[0], 1)? >> 1)?;
    values[2] = checked_sub(values[2], checked_add(values[1], 1)? >> 1)?;
    values[1] = checked_add(values[1], checked_add(values[0], 2)? >> 2)?;
    values[0] = checked_add(values[0], checked_add(values[1], 1)? >> 1)?;
    values[0] = checked_add(values[0], values[1] >> 5)?;
    values[0] = checked_add(values[0], values[1] >> 9)?;
    values[0] = checked_add(values[0], values[1] >> 13)?;
    values[1] = checked_add(values[1], checked_add(values[0], 2)? >> 2)?;
    values[3] = checked_add(values[3], checked_add(values[0], 1)? >> 1)?;
    values[2] = checked_add(values[2], checked_add(values[1], 1)? >> 1)?;
    values[0] = checked_sub(values[0], values[3])?;
    values[1] = checked_sub(values[1], values[2])?;
    Ok(())
}

/// Apply the exact two-sample JPEG XR inverse overlap postfilter in place.
///
/// This is the first-level boundary operator for subsampled chroma.
pub fn inverse_overlap_filter_2(values: &mut [i32; 2]) -> Result<(), MathError> {
    values[1] = checked_add(values[1], checked_add(values[0], 2)? >> 2)?;
    values[0] = checked_add(values[0], checked_add(values[1], 1)? >> 1)?;
    values[0] = checked_add(values[0], values[1] >> 5)?;
    values[0] = checked_add(values[0], values[1] >> 9)?;
    values[0] = checked_add(values[0], values[1] >> 13)?;
    values[1] = checked_add(values[1], checked_add(values[0], 2)? >> 2)?;
    Ok(())
}

fn inverse_rotate(values: &mut [i32; 2]) -> Result<(), MathError> {
    values[0] = checked_sub(values[0], checked_add(values[1], 1)? >> 1)?;
    values[1] = checked_add(values[1], checked_add(values[0], 1)? >> 1)?;
    Ok(())
}

fn inverse_scale(values: &mut [i32; 2]) -> Result<(), MathError> {
    values[0] = checked_add(values[0], values[1])?;
    values[1] = checked_sub(values[0] >> 1, values[1])?;
    values[0] = checked_add(values[0], checked_mul(values[1], 3)? >> 3)?;
    values[1] = checked_add(values[1], checked_mul(values[0], 3)? >> 4)?;
    values[1] = checked_add(values[1], values[0] >> 7)?;
    values[1] = checked_sub(values[1], values[0] >> 10)?;
    Ok(())
}

fn inverse_hadamard_post(values: &mut [i32; 4]) -> Result<(), MathError> {
    values[1] = checked_sub(values[1], values[2])?;
    values[0] = checked_add(values[0], checked_add(checked_mul(values[3], 3)?, 4)? >> 3)?;
    values[3] = checked_sub(values[3], values[1] >> 1)?;
    values[2] = checked_sub(checked_sub(values[0], values[1])? >> 1, values[2])?;
    values.swap(2, 3);
    values[0] = checked_sub(values[0], values[3])?;
    values[1] = checked_add(values[1], values[2])?;
    Ok(())
}

fn inverse_todd_odd_post(values: &mut [i32; 4]) -> Result<(), MathError> {
    values[3] = checked_add(values[3], values[0])?;
    values[2] = checked_sub(values[2], values[1])?;
    let first = values[3] >> 1;
    let second = values[2] >> 1;
    values[0] = checked_sub(values[0], first)?;
    values[1] = checked_add(values[1], second)?;
    values[0] = checked_sub(values[0], checked_add(checked_mul(values[1], 3)?, 6)? >> 3)?;
    values[1] = checked_add(values[1], checked_add(checked_mul(values[0], 3)?, 2)? >> 2)?;
    values[0] = checked_sub(values[0], checked_add(checked_mul(values[1], 3)?, 4)? >> 3)?;
    values[1] = checked_sub(values[1], second)?;
    values[0] = checked_add(values[0], first)?;
    values[2] = checked_add(values[2], values[1])?;
    values[3] = checked_sub(values[3], values[0])?;
    Ok(())
}

fn apply_group(
    values: &mut [i32; 16],
    indices: [usize; 4],
    operation: impl FnOnce(&mut [i32; 4]) -> Result<(), MathError>,
) -> Result<(), MathError> {
    let mut group = indices.map(|index| values[index]);
    operation(&mut group)?;
    for (index, value) in indices.into_iter().zip(group) {
        values[index] = value;
    }
    Ok(())
}

fn apply_pair(
    values: &mut [i32; 16],
    indices: [usize; 2],
    operation: impl FnOnce(&mut [i32; 2]) -> Result<(), MathError>,
) -> Result<(), MathError> {
    let mut pair = indices.map(|index| values[index]);
    operation(&mut pair)?;
    for (index, value) in indices.into_iter().zip(pair) {
        values[index] = value;
    }
    Ok(())
}

fn checked_neg(value: i32) -> Result<i32, MathError> {
    value
        .checked_neg()
        .ok_or_else(|| MathError::overflow("overlap negation"))
}

#[cfg(test)]
mod tests {
    use super::{
        inverse_overlap_filter_2, inverse_overlap_filter_2x2, inverse_overlap_filter_4,
        inverse_overlap_filter_4x4,
    };

    #[test]
    fn inverse_filters_preserve_zero() {
        let mut four = [0; 4];
        inverse_overlap_filter_4(&mut four).unwrap();
        assert_eq!(four, [0; 4]);
        let mut sixteen = [0; 16];
        inverse_overlap_filter_4x4(&mut sixteen).unwrap();
        assert_eq!(sixteen, [0; 16]);
    }

    #[test]
    fn subsampled_filters_have_exact_results() {
        let mut pair = [1, 2];
        inverse_overlap_filter_2(&mut pair).unwrap();
        assert_eq!(pair, [2, 3]);

        let mut square = [1, 2, 3, 4];
        inverse_overlap_filter_2x2(&mut square).unwrap();
        assert_eq!(square, [3, 4, 4, 5]);
    }
}
