//! Exact JPEG XR core-transform lifting operators in both directions.

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

/// Apply the reversible JPEG XR 4-by-4 forward core transform in place.
///
/// This is the exact inverse of [`inverse_core_transform`]: every lifting
/// stage is undone in reverse order, so the pair is lossless for any input
/// whose intermediates fit in `i32`.
#[inline]
pub fn forward_core_transform(samples: &mut [i32; 16]) -> Result<(), MathError> {
    // The Hadamard lifting operator is its own inverse for either rounding.
    math_try!(transform_group(samples, [4, 7, 8, 11], t2x2h_zero));
    math_try!(transform_group(samples, [1, 2, 13, 14], t2x2h_zero));
    math_try!(transform_group(samples, [5, 6, 9, 10], t2x2h_zero));
    math_try!(transform_group(samples, [0, 3, 12, 15], t2x2h_zero));
    math_try!(transform_group(samples, [10, 11, 14, 15], forward_todd_odd));
    math_try!(transform_group(samples, [8, 12, 9, 13], forward_todd));
    math_try!(transform_group(samples, [2, 3, 6, 7], forward_todd));
    math_try!(transform_group(samples, [0, 1, 4, 5], t2x2h_round));
    forward_permute(samples);
    Ok(())
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

/// Apply the first-level forward transform for one YUV 4:2:0 chroma macroblock.
///
/// This is the exact inverse of [`inverse_chroma_420`].
#[inline]
pub fn forward_chroma_420(samples: &mut [i32; 4]) -> Result<(), MathError> {
    [samples[1], samples[2]] = [samples[2], samples[1]];
    // The zero-rounding Hadamard operator is its own inverse.
    t2x2h(samples, 0)
}

/// Apply the first-level forward transform for one YUV 4:2:2 chroma macroblock.
///
/// This is the exact inverse of [`inverse_chroma_422`]: the two row
/// transforms are undone before the lifting step that joins their DC terms.
#[inline]
pub fn forward_chroma_422(samples: &mut [i32; 8]) -> Result<(), MathError> {
    [samples[5], samples[6]] = [samples[6], samples[5]];
    let mut second = [samples[4], samples[6], samples[5], samples[7]];
    math_try!(t2x2h(&mut second, 0));
    samples[4] = second[0];
    samples[6] = second[1];
    samples[5] = second[2];
    samples[7] = second[3];

    [samples[1], samples[2]] = [samples[2], samples[1]];
    let mut first = [samples[0], samples[1], samples[2], samples[3]];
    math_try!(t2x2h(&mut first, 0));
    samples[..4].copy_from_slice(&first);

    let mut pair = [samples[0], samples[4]];
    math_try!(forward_t2pt(&mut pair));
    samples[0] = pair[0];
    samples[4] = pair[1];
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

/// Exact inverse of [`inverse_t2pt`].
#[inline]
fn forward_t2pt(values: &mut [i32; 2]) -> Result<(), MathError> {
    values[1] = math_try!(checked_sub(values[1], values[0]));
    values[0] = math_try!(checked_add(
        values[0],
        math_try!(checked_add(values[1], 1)) >> 1
    ));
    Ok(())
}

/// Applies `operation` to the elements `indices` of a block's
/// `coefficients`, gathered in that order and written back to them: the
/// transform and overlap operators each work on a group of two or four.
#[inline]
pub(crate) fn transform_group<const N: usize>(
    coefficients: &mut [i32; 16],
    indices: [usize; N],
    operation: impl FnOnce(&mut [i32; N]) -> Result<(), MathError>,
) -> Result<(), MathError> {
    // CUDA kernels reach this function, and CUDA Oxide's pinned compiler
    // fails on a closure here (`indices.map(|index| ...)`), so the gather
    // is a loop like the write-back below.
    let mut values = [0; N];
    for i in 0..N {
        values[i] = coefficients[indices[i]];
    }
    math_try!(operation(&mut values));
    for i in 0..N {
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
fn forward_permute(coefficients: &mut [i32; 16]) {
    let input = *coefficients;
    coefficients[0] = input[INVERSE_PERMUTATION[0]];
    coefficients[1] = input[INVERSE_PERMUTATION[1]];
    coefficients[2] = input[INVERSE_PERMUTATION[2]];
    coefficients[3] = input[INVERSE_PERMUTATION[3]];
    coefficients[4] = input[INVERSE_PERMUTATION[4]];
    coefficients[5] = input[INVERSE_PERMUTATION[5]];
    coefficients[6] = input[INVERSE_PERMUTATION[6]];
    coefficients[7] = input[INVERSE_PERMUTATION[7]];
    coefficients[8] = input[INVERSE_PERMUTATION[8]];
    coefficients[9] = input[INVERSE_PERMUTATION[9]];
    coefficients[10] = input[INVERSE_PERMUTATION[10]];
    coefficients[11] = input[INVERSE_PERMUTATION[11]];
    coefficients[12] = input[INVERSE_PERMUTATION[12]];
    coefficients[13] = input[INVERSE_PERMUTATION[13]];
    coefficients[14] = input[INVERSE_PERMUTATION[14]];
    coefficients[15] = input[INVERSE_PERMUTATION[15]];
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

/// Exact inverse of [`inverse_todd`], undoing its lifting steps in reverse.
#[inline]
fn forward_todd(values: &mut [i32; 4]) -> Result<(), MathError> {
    values[0] = math_try!(checked_add(values[0], values[3]));
    values[1] = math_try!(checked_sub(values[1], values[2]));
    values[3] = math_try!(checked_sub(
        math_try!(checked_add(values[0], 1)) >> 1,
        values[3]
    ));
    values[2] = math_try!(checked_add(
        values[2],
        math_try!(checked_add(values[1], 1)) >> 1
    ));
    values[3] = math_try!(checked_sub(
        values[3],
        math_try!(checked_add(math_try!(checked_mul(values[2], 3)), 4)) >> 3
    ));
    values[2] = math_try!(checked_add(
        values[2],
        math_try!(checked_add(math_try!(checked_mul(values[3], 3)), 4)) >> 3
    ));
    values[1] = math_try!(checked_sub(
        values[1],
        math_try!(checked_add(math_try!(checked_mul(values[0], 3)), 4)) >> 3
    ));
    values[0] = math_try!(checked_add(
        values[0],
        math_try!(checked_add(math_try!(checked_mul(values[1], 3)), 4)) >> 3
    ));
    values[2] = math_try!(checked_sub(
        values[2],
        math_try!(checked_add(values[0], 1)) >> 1
    ));
    values[3] = math_try!(checked_add(values[3], values[1] >> 1));
    values[0] = math_try!(checked_add(values[0], values[2]));
    values[1] = math_try!(checked_sub(values[1], values[3]));
    Ok(())
}

/// Exact inverse of [`inverse_todd_odd`], undoing its lifting steps in reverse.
#[inline]
fn forward_todd_odd(values: &mut [i32; 4]) -> Result<(), MathError> {
    values[1] = math_try!(
        values[1]
            .checked_neg()
            .ok_or(crate::arithmetic::overflow("forward Todd odd negation"))
    );
    values[2] = math_try!(
        values[2]
            .checked_neg()
            .ok_or(crate::arithmetic::overflow("forward Todd odd negation"))
    );
    values[3] = math_try!(checked_add(values[3], values[0]));
    values[2] = math_try!(checked_sub(values[2], values[1]));
    // The inverse derives these halves from the same two values, which none
    // of the steps between here and their use modify.
    let first = values[3] >> 1;
    let second = values[2] >> 1;
    values[0] = math_try!(checked_sub(values[0], first));
    values[1] = math_try!(checked_add(values[1], second));
    values[0] = math_try!(checked_add(
        values[0],
        math_try!(checked_add(math_try!(checked_mul(values[1], 3)), 4)) >> 3
    ));
    values[1] = math_try!(checked_sub(
        values[1],
        math_try!(checked_add(math_try!(checked_mul(values[0], 3)), 3)) >> 2
    ));
    values[0] = math_try!(checked_add(
        values[0],
        math_try!(checked_add(math_try!(checked_mul(values[1], 3)), 3)) >> 3
    ));
    values[1] = math_try!(checked_sub(values[1], second));
    values[0] = math_try!(checked_add(values[0], first));
    values[2] = math_try!(checked_add(values[2], values[1]));
    values[3] = math_try!(checked_sub(values[3], values[0]));
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{
        forward_chroma_420, forward_chroma_422, forward_core_transform, inverse_chroma_420,
        inverse_chroma_422, inverse_core_transform,
    };

    /// Deterministic values spanning signs and magnitudes up to `bits` bits,
    /// which the transform and overlap tests draw.
    pub(crate) fn samples(seed: u64, bits: u32) -> impl FnMut() -> i32 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let magnitude = ((state >> 33) % (1 << bits)) as i32;
            if state >> 63 == 0 {
                magnitude
            } else {
                -magnitude
            }
        }
    }

    #[test]
    fn forward_and_inverse_core_transforms_are_exact_inverses() {
        for bits in [1, 4, 9, 17, 24] {
            let mut next = samples(u64::from(bits), bits);
            for _ in 0..2_000 {
                let original: [i32; 16] = core::array::from_fn(|_| next());
                let mut block = original;
                forward_core_transform(&mut block).unwrap();
                inverse_core_transform(&mut block).unwrap();
                assert_eq!(block, original, "inverse of forward");
                inverse_core_transform(&mut block).unwrap();
                forward_core_transform(&mut block).unwrap();
                assert_eq!(block, original, "forward of inverse");
            }
        }
    }

    #[test]
    fn forward_and_inverse_chroma_transforms_are_exact_inverses() {
        for bits in [1, 4, 9, 17, 24] {
            let mut next = samples(u64::from(bits) + 200, bits);
            for _ in 0..2_000 {
                let original: [i32; 4] = core::array::from_fn(|_| next());
                let mut four = original;
                forward_chroma_420(&mut four).unwrap();
                inverse_chroma_420(&mut four).unwrap();
                assert_eq!(four, original, "4:2:0 inverse of forward");
                inverse_chroma_420(&mut four).unwrap();
                forward_chroma_420(&mut four).unwrap();
                assert_eq!(four, original, "4:2:0 forward of inverse");

                let original: [i32; 8] = core::array::from_fn(|_| next());
                let mut eight = original;
                forward_chroma_422(&mut eight).unwrap();
                inverse_chroma_422(&mut eight).unwrap();
                assert_eq!(eight, original, "4:2:2 inverse of forward");
                inverse_chroma_422(&mut eight).unwrap();
                forward_chroma_422(&mut eight).unwrap();
                assert_eq!(eight, original, "4:2:2 forward of inverse");
            }
        }
    }

    #[test]
    fn forward_chroma_transforms_concentrate_a_flat_macroblock_in_dc() {
        let mut four = [6; 4];
        forward_chroma_420(&mut four).unwrap();
        assert_eq!(four, [12, 0, 0, 0]);

        let mut eight = [6; 8];
        forward_chroma_422(&mut eight).unwrap();
        assert_ne!(eight[0], 0);
        assert!(eight[1..].iter().all(|&coefficient| coefficient == 0));
        inverse_chroma_422(&mut eight).unwrap();
        assert_eq!(eight, [6; 8]);
    }

    #[test]
    fn forward_chroma_transforms_report_overflow() {
        assert!(forward_chroma_420(&mut [i32::MAX; 4]).is_err());
        assert!(forward_chroma_422(&mut [i32::MAX; 8]).is_err());
    }

    #[test]
    fn forward_transform_concentrates_a_flat_block_in_dc() {
        let mut block = [7; 16];
        forward_core_transform(&mut block).unwrap();
        assert!(block[1..].iter().all(|&coefficient| coefficient == 0));
        inverse_core_transform(&mut block).unwrap();
        assert_eq!(block, [7; 16]);
    }

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
