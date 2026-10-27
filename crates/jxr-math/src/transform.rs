//! Exact JPEG XR inverse core-transform lifting operators.

use crate::arithmetic::{MathError, checked_add, checked_mul, checked_sub};
use crate::tables::INVERSE_PERMUTATION;

/// Apply the reversible JPEG XR 4-by-4 inverse core transform in place.
#[inline]
pub fn inverse_core_transform(coefficients: &mut [i32; 16]) -> Result<(), MathError> {
    inverse_permute(coefficients);
    math_try!(transform_group(coefficients, [0, 1, 4, 5], t2x2h_round));
    math_try!(transform_group(coefficients, [2, 3, 6, 7], inverse_todd));
    math_try!(transform_group(coefficients, [8, 12, 9, 13], inverse_todd));
    math_try!(transform_group(
        coefficients,
        [10, 11, 14, 15],
        inverse_todd_odd
    ));
    math_try!(transform_group(coefficients, [0, 3, 12, 15], t2x2h_zero));
    math_try!(transform_group(coefficients, [5, 6, 9, 10], t2x2h_zero));
    math_try!(transform_group(coefficients, [1, 2, 13, 14], t2x2h_zero));
    transform_group(coefficients, [4, 7, 8, 11], t2x2h_zero)
}

/// Apply the four-value inverse Hadamard lifting operator used by transforms and postfilters.
#[inline]
pub fn inverse_hadamard_2x2(values: &mut [i32; 4], rounding: i32) -> Result<(), MathError> {
    t2x2h(values, rounding)
}

/// Apply the first-level inverse transform for one YUV 4:2:0 chroma macroblock.
#[inline]
pub fn inverse_chroma_420(coefficients: &mut [i32; 4]) -> Result<(), MathError> {
    math_try!(t2x2h(coefficients, 0));
    [coefficients[1], coefficients[2]] = [coefficients[2], coefficients[1]];
    Ok(())
}

/// Apply the first-level inverse transform for one YUV 4:2:2 chroma macroblock.
#[inline]
pub fn inverse_chroma_422(coefficients: &mut [i32; 8]) -> Result<(), MathError> {
    let mut pair = [coefficients[0], coefficients[4]];
    math_try!(inverse_t2pt(&mut pair));
    coefficients[0] = pair[0];
    coefficients[4] = pair[1];

    let mut first = [
        coefficients[0],
        coefficients[1],
        coefficients[2],
        coefficients[3],
    ];
    math_try!(t2x2h(&mut first, 0));
    coefficients[..4].copy_from_slice(&first);
    [coefficients[1], coefficients[2]] = [coefficients[2], coefficients[1]];

    let mut second = [
        coefficients[4],
        coefficients[6],
        coefficients[5],
        coefficients[7],
    ];
    math_try!(t2x2h(&mut second, 0));
    coefficients[4] = second[0];
    coefficients[6] = second[1];
    coefficients[5] = second[2];
    coefficients[7] = second[3];
    [coefficients[5], coefficients[6]] = [coefficients[6], coefficients[5]];
    Ok(())
}

#[inline]
fn inverse_t2pt(values: &mut [i32; 2]) -> Result<(), MathError> {
    values[0] = math_try!(checked_sub(
        values[0],
        math_try!(checked_add(values[1], 1)) >> 1
    ));
    values[1] = math_try!(checked_add(values[1], values[0]));
    Ok(())
}

#[inline]
fn transform_group(
    coefficients: &mut [i32; 16],
    indices: [usize; 4],
    operation: impl FnOnce(&mut [i32; 4]) -> Result<(), MathError>,
) -> Result<(), MathError> {
    let mut values = [
        coefficients[indices[0]],
        coefficients[indices[1]],
        coefficients[indices[2]],
        coefficients[indices[3]],
    ];
    math_try!(operation(&mut values));
    for i in 0..4 {
        coefficients[indices[i]] = values[i];
    }
    Ok(())
}

#[inline]
fn inverse_permute(coefficients: &mut [i32; 16]) {
    let input = *coefficients;
    // Constant subscripts let CUDA Oxide keep the permutation and temporary
    // blocks in registers instead of materializing per-thread stack arrays.
    coefficients[INVERSE_PERMUTATION[0]] = input[0];
    coefficients[INVERSE_PERMUTATION[1]] = input[1];
    coefficients[INVERSE_PERMUTATION[2]] = input[2];
    coefficients[INVERSE_PERMUTATION[3]] = input[3];
    coefficients[INVERSE_PERMUTATION[4]] = input[4];
    coefficients[INVERSE_PERMUTATION[5]] = input[5];
    coefficients[INVERSE_PERMUTATION[6]] = input[6];
    coefficients[INVERSE_PERMUTATION[7]] = input[7];
    coefficients[INVERSE_PERMUTATION[8]] = input[8];
    coefficients[INVERSE_PERMUTATION[9]] = input[9];
    coefficients[INVERSE_PERMUTATION[10]] = input[10];
    coefficients[INVERSE_PERMUTATION[11]] = input[11];
    coefficients[INVERSE_PERMUTATION[12]] = input[12];
    coefficients[INVERSE_PERMUTATION[13]] = input[13];
    coefficients[INVERSE_PERMUTATION[14]] = input[14];
    coefficients[INVERSE_PERMUTATION[15]] = input[15];
}

#[inline]
fn t2x2h_round(values: &mut [i32; 4]) -> Result<(), MathError> {
    t2x2h(values, 1)
}

#[inline]
fn t2x2h_zero(values: &mut [i32; 4]) -> Result<(), MathError> {
    t2x2h(values, 0)
}

#[inline]
fn t2x2h(values: &mut [i32; 4], rounding: i32) -> Result<(), MathError> {
    values[0] = math_try!(checked_add(values[0], values[3]));
    values[1] = math_try!(checked_sub(values[1], values[2]));
    let difference = math_try!(checked_sub(values[0], values[1]));
    let first = math_try!(checked_add(difference, rounding)) >> 1;
    let second = values[2];
    values[2] = math_try!(checked_sub(first, values[3]));
    values[3] = math_try!(checked_sub(first, second));
    values[0] = math_try!(checked_sub(values[0], values[3]));
    values[1] = math_try!(checked_add(values[1], values[2]));
    Ok(())
}

#[inline]
fn inverse_todd(values: &mut [i32; 4]) -> Result<(), MathError> {
    values[1] = math_try!(checked_add(values[1], values[3]));
    values[0] = math_try!(checked_sub(values[0], values[2]));
    values[3] = math_try!(checked_sub(values[3], values[1] >> 1));
    values[2] = math_try!(checked_add(
        values[2],
        math_try!(checked_add(values[0], 1)) >> 1
    ));
    values[0] = math_try!(checked_sub(
        values[0],
        math_try!(checked_add(math_try!(checked_mul(values[1], 3)), 4)) >> 3
    ));
    values[1] = math_try!(checked_add(
        values[1],
        math_try!(checked_add(math_try!(checked_mul(values[0], 3)), 4)) >> 3
    ));
    values[2] = math_try!(checked_sub(
        values[2],
        math_try!(checked_add(math_try!(checked_mul(values[3], 3)), 4)) >> 3
    ));
    values[3] = math_try!(checked_add(
        values[3],
        math_try!(checked_add(math_try!(checked_mul(values[2], 3)), 4)) >> 3
    ));
    values[2] = math_try!(checked_sub(
        values[2],
        math_try!(checked_add(values[1], 1)) >> 1
    ));
    values[3] = math_try!(checked_sub(
        math_try!(checked_add(values[0], 1)) >> 1,
        values[3]
    ));
    values[1] = math_try!(checked_add(values[1], values[2]));
    values[0] = math_try!(checked_sub(values[0], values[3]));
    Ok(())
}

#[inline]
fn inverse_todd_odd(values: &mut [i32; 4]) -> Result<(), MathError> {
    values[3] = math_try!(checked_add(values[3], values[0]));
    values[2] = math_try!(checked_sub(values[2], values[1]));
    let first = values[3] >> 1;
    let second = values[2] >> 1;
    values[0] = math_try!(checked_sub(values[0], first));
    values[1] = math_try!(checked_add(values[1], second));
    values[0] = math_try!(checked_sub(
        values[0],
        math_try!(checked_add(math_try!(checked_mul(values[1], 3)), 3)) >> 3
    ));
    values[1] = math_try!(checked_add(
        values[1],
        math_try!(checked_add(math_try!(checked_mul(values[0], 3)), 3)) >> 2
    ));
    values[0] = math_try!(checked_sub(
        values[0],
        math_try!(checked_add(math_try!(checked_mul(values[1], 3)), 4)) >> 3
    ));
    values[1] = math_try!(checked_sub(values[1], second));
    values[0] = math_try!(checked_add(values[0], first));
    values[2] = math_try!(checked_add(values[2], values[1]));
    values[3] = math_try!(checked_sub(values[3], values[0]));
    values[1] = math_try!(
        values[1]
            .checked_neg()
            .ok_or(crate::arithmetic::overflow("inverse Todd odd negation"))
    );
    values[2] = math_try!(
        values[2]
            .checked_neg()
            .ok_or(crate::arithmetic::overflow("inverse Todd odd negation"))
    );
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
