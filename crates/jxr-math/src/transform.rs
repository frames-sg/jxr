//! Exact JPEG XR inverse core-transform lifting operators.

use crate::arithmetic::{MathError, checked_add, checked_mul, checked_sub};
use crate::tables::INVERSE_PERMUTATION;

/// Apply the reversible JPEG XR 4-by-4 inverse core transform in place.
pub fn inverse_core_transform(coefficients: &mut [i32; 16]) -> Result<(), MathError> {
    inverse_permute(coefficients);
    transform_group(coefficients, [0, 1, 4, 5], |values| t2x2h(values, 1))?;
    transform_group(coefficients, [2, 3, 6, 7], inverse_todd)?;
    transform_group(coefficients, [8, 12, 9, 13], inverse_todd)?;
    transform_group(coefficients, [10, 11, 14, 15], inverse_todd_odd)?;
    transform_group(coefficients, [0, 3, 12, 15], |values| t2x2h(values, 0))?;
    transform_group(coefficients, [5, 6, 9, 10], |values| t2x2h(values, 0))?;
    transform_group(coefficients, [1, 2, 13, 14], |values| t2x2h(values, 0))?;
    transform_group(coefficients, [4, 7, 8, 11], |values| t2x2h(values, 0))
}

/// Apply the four-value inverse Hadamard lifting operator used by transforms and postfilters.
pub fn inverse_hadamard_2x2(values: &mut [i32; 4], rounding: i32) -> Result<(), MathError> {
    t2x2h(values, rounding)
}

/// Apply the first-level inverse transform for one YUV 4:2:0 chroma macroblock.
pub fn inverse_chroma_420(coefficients: &mut [i32; 4]) -> Result<(), MathError> {
    t2x2h(coefficients, 0)?;
    coefficients.swap(1, 2);
    Ok(())
}

/// Apply the first-level inverse transform for one YUV 4:2:2 chroma macroblock.
pub fn inverse_chroma_422(coefficients: &mut [i32; 8]) -> Result<(), MathError> {
    let mut pair = [coefficients[0], coefficients[4]];
    inverse_t2pt(&mut pair)?;
    coefficients[0] = pair[0];
    coefficients[4] = pair[1];

    let mut first = [
        coefficients[0],
        coefficients[1],
        coefficients[2],
        coefficients[3],
    ];
    t2x2h(&mut first, 0)?;
    coefficients[..4].copy_from_slice(&first);
    coefficients.swap(1, 2);

    let mut second = [
        coefficients[4],
        coefficients[6],
        coefficients[5],
        coefficients[7],
    ];
    t2x2h(&mut second, 0)?;
    coefficients[4] = second[0];
    coefficients[6] = second[1];
    coefficients[5] = second[2];
    coefficients[7] = second[3];
    coefficients.swap(5, 6);
    Ok(())
}

fn inverse_t2pt(values: &mut [i32; 2]) -> Result<(), MathError> {
    values[0] = checked_sub(values[0], checked_add(values[1], 1)? >> 1)?;
    values[1] = checked_add(values[1], values[0])?;
    Ok(())
}

fn transform_group(
    coefficients: &mut [i32; 16],
    indices: [usize; 4],
    operation: impl FnOnce(&mut [i32; 4]) -> Result<(), MathError>,
) -> Result<(), MathError> {
    let mut values = indices.map(|index| coefficients[index]);
    operation(&mut values)?;
    for (index, value) in indices.into_iter().zip(values) {
        coefficients[index] = value;
    }
    Ok(())
}

fn inverse_permute(coefficients: &mut [i32; 16]) {
    let input = *coefficients;
    for (source, destination) in INVERSE_PERMUTATION.into_iter().enumerate() {
        coefficients[destination] = input[source];
    }
}

fn t2x2h(values: &mut [i32; 4], rounding: i32) -> Result<(), MathError> {
    values[0] = checked_add(values[0], values[3])?;
    values[1] = checked_sub(values[1], values[2])?;
    let difference = checked_sub(values[0], values[1])?;
    let first = checked_add(difference, rounding)? >> 1;
    let second = values[2];
    values[2] = checked_sub(first, values[3])?;
    values[3] = checked_sub(first, second)?;
    values[0] = checked_sub(values[0], values[3])?;
    values[1] = checked_add(values[1], values[2])?;
    Ok(())
}

fn inverse_todd(values: &mut [i32; 4]) -> Result<(), MathError> {
    values[1] = checked_add(values[1], values[3])?;
    values[0] = checked_sub(values[0], values[2])?;
    values[3] = checked_sub(values[3], values[1] >> 1)?;
    values[2] = checked_add(values[2], checked_add(values[0], 1)? >> 1)?;
    values[0] = checked_sub(values[0], checked_add(checked_mul(values[1], 3)?, 4)? >> 3)?;
    values[1] = checked_add(values[1], checked_add(checked_mul(values[0], 3)?, 4)? >> 3)?;
    values[2] = checked_sub(values[2], checked_add(checked_mul(values[3], 3)?, 4)? >> 3)?;
    values[3] = checked_add(values[3], checked_add(checked_mul(values[2], 3)?, 4)? >> 3)?;
    values[2] = checked_sub(values[2], checked_add(values[1], 1)? >> 1)?;
    values[3] = checked_sub(checked_add(values[0], 1)? >> 1, values[3])?;
    values[1] = checked_add(values[1], values[2])?;
    values[0] = checked_sub(values[0], values[3])?;
    Ok(())
}

fn inverse_todd_odd(values: &mut [i32; 4]) -> Result<(), MathError> {
    values[3] = checked_add(values[3], values[0])?;
    values[2] = checked_sub(values[2], values[1])?;
    let first = values[3] >> 1;
    let second = values[2] >> 1;
    values[0] = checked_sub(values[0], first)?;
    values[1] = checked_add(values[1], second)?;
    values[0] = checked_sub(values[0], checked_add(checked_mul(values[1], 3)?, 3)? >> 3)?;
    values[1] = checked_add(values[1], checked_add(checked_mul(values[0], 3)?, 3)? >> 2)?;
    values[0] = checked_sub(values[0], checked_add(checked_mul(values[1], 3)?, 4)? >> 3)?;
    values[1] = checked_sub(values[1], second)?;
    values[0] = checked_add(values[0], first)?;
    values[2] = checked_add(values[2], values[1])?;
    values[3] = checked_sub(values[3], values[0])?;
    values[1] = values[1]
        .checked_neg()
        .ok_or_else(|| MathError::overflow("inverse Todd odd negation"))?;
    values[2] = values[2]
        .checked_neg()
        .ok_or_else(|| MathError::overflow("inverse Todd odd negation"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{inverse_chroma_422, inverse_core_transform};

    #[test]
    fn dc_only_block_is_uniform() {
        let mut coefficients = [0; 16];
        coefficients[0] = 16;
        inverse_core_transform(&mut coefficients).unwrap();
        assert_eq!(coefficients, [4; 16]);
    }

    #[test]
    fn inverse_transform_matches_t835_signed_rounding_case() {
        let mut coefficients = [
            50, -42, 9, 21, 126, 33, -6, -18, -24, 21, -3, -9, -6, -33, 6, 15,
        ];
        inverse_core_transform(&mut coefficients).unwrap();
        assert_eq!(
            coefficients,
            [
                50, 50, 47, 49, 39, 38, 41, 41, -30, -44, 27, 36, -47, -55, -23, -20,
            ]
        );
    }

    #[test]
    fn inverse_chroma_422_transforms_each_row_before_inverse_permutation() {
        let mut coefficients = [1, 2, 3, 4, 5, 6, 7, 8];
        inverse_chroma_422(&mut coefficients).unwrap();
        assert_eq!(coefficients, [4, -3, -4, -2, 12, -2, -3, -1]);
    }
}
