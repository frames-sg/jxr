//! Capability-token SIMD for the encoder's exact quantization, which
//! dequantization inverts.

use fearless_simd::{Bytes, Select, Simd, SimdBase, SimdCvtFloat, SimdCvtTruncate};
use jxr_math::quantization::Quantizer;

use crate::cpu::{I32Vectors, with_i32_vectors};

/// Rounded magnitudes and steps below this bound are exact in `f32`, and
/// their quotient estimated in `f32` is within one of the true quotient.
const VECTOR_QUANTIZATION_DOMAIN: u32 = 1 << 24;

/// Quantizes `coefficients` in place exactly as [`Quantizer::quantize`]
/// does with `offset`: on `vectors` where they are given and every rounded
/// magnitude and the step lie in the vector form's domain, and through
/// [`Quantizer::quantize`], the oracle, otherwise. The inverse of
/// [`Quantizer::dequantize`].
pub(crate) fn quantize_coefficients(
    vectors: Option<I32Vectors>,
    quantizer: Quantizer,
    offset: u32,
    coefficients: &mut [i32],
) {
    let on_vectors = vectors.is_some_and(|vectors| {
        with_i32_vectors!(vectors, simd => quantize_vectorized(simd, quantizer, offset, coefficients))
    });
    if !on_vectors {
        for coefficient in coefficients {
            *coefficient = quantizer.quantize(*coefficient, offset);
        }
    }
}

/// Quantizes with `f32` estimates of each quotient, corrected exactly, and
/// returns whether it did; it does nothing when the slice is shorter than a
/// vector or a rounded magnitude or the step reaches
/// [`VECTOR_QUANTIZATION_DOMAIN`].
///
/// A rounded magnitude `n` and a step `d` below `2^24` convert to `f32`
/// exactly, and the reciprocal and the product are each rounded once, by
/// a relative error of at most `2^-24`: the estimate differs from `n / d` by
/// less than `(n / d) * 2^-23 < 2 / d`, which for `d >= 3` is below one, and
/// for a power of two, whose reciprocal is exact, is below `2^-24 * n / d`.
/// Its truncation is therefore the quotient `q` or one either side, the
/// remainder `n - t * d` then lies in `-d..2 * d` without overflow, and one
/// step towards it gives `q`.
///
/// The domain is checked once for the slice, in a pass of its own: on
/// WebAssembly a test of each vector costs more than the pass.
#[inline]
fn quantize_vectorized<S: Simd>(
    simd: S,
    quantizer: Quantizer,
    offset: u32,
    coefficients: &mut [i32],
) -> bool {
    let step = quantizer.step();
    // `Quantizer::quantize` clamps the offset below one step.
    let offset = offset.min(step - 1);
    if step >= VECTOR_QUANTIZATION_DOMAIN || coefficients.len() < S::i32s::LEN {
        return false;
    }
    simd.vectorize(|| {
        let lanes = S::i32s::LEN;
        let magnitude = coefficients
            .chunks_exact(lanes)
            .fold(S::i32s::splat(simd, 0), |bits, chunk| {
                bits | S::i32s::from_slice(simd, chunk).abs()
            });
        let tail = coefficients
            .chunks_exact(lanes)
            .remainder()
            .iter()
            .fold(0, |bits, coefficient| bits | coefficient.unsigned_abs());
        // The OR of the magnitudes bounds each of them; the wrapped
        // magnitude of `i32::MIN` is out of the domain as an unsigned value.
        let largest = magnitude.bitcast::<S::u32s>().reduce_max() | tail;
        if largest >= VECTOR_QUANTIZATION_DOMAIN - offset {
            return false;
        }
        let signed = |value: u32| i32::try_from(value).expect("below the vector domain");
        let divisor = S::i32s::splat(simd, signed(step));
        let rounding = S::i32s::splat(simd, signed(offset));
        let inverse = S::f32s::splat(simd, 1.0) / S::f32s::float_from(divisor);
        let zero = S::i32s::splat(simd, 0);
        let one = S::i32s::splat(simd, 1);
        let mut chunks = coefficients.chunks_exact_mut(lanes);
        for chunk in &mut chunks {
            let coefficient = S::i32s::from_slice(simd, chunk);
            let rounded = coefficient.abs() + rounding;
            let estimate = S::i32s::truncate_from(S::f32s::float_from(rounded) * inverse);
            let remainder = rounded - estimate * divisor;
            let estimate = remainder.simd_ge(divisor).select(estimate + one, estimate);
            let quotient = remainder.simd_lt(zero).select(estimate - one, estimate);
            coefficient
                .simd_lt(zero)
                .select(-quotient, quotient)
                .store_slice(chunk);
        }
        for coefficient in chunks.into_remainder() {
            *coefficient = quantizer.quantize(*coefficient, offset);
        }
        true
    })
}

#[cfg(test)]
mod tests {
    use jxr_math::quantization::Quantizer;

    use super::{VECTOR_QUANTIZATION_DOMAIN, quantize_coefficients};
    use crate::{CpuCapabilities, cpu::I32Vectors, tile_decode::quantizer::component_steps};

    /// Every step the encoder divides by: each step the decoder's mapping
    /// gives a parameter of a component, scaled or not, and its double, by
    /// which DC and low-pass bands are divided where reconstruction doubles
    /// them.
    fn codec_steps() -> Vec<u32> {
        let mut steps: Vec<u32> = (0..=u8::MAX)
            .flat_map(|parameter| {
                [(0, false), (0, true), (1, false), (1, true)]
                    .map(|(component, scaled)| component_steps(component, [parameter; 3], scaled))
            })
            .map(Result::unwrap)
            .flat_map(|steps| [steps.dc, steps.low_pass, steps.high_pass])
            .flat_map(|step| [step, step * 2])
            .collect();
        steps.sort_unstable();
        steps.dedup();
        steps
    }

    fn vectors() -> Option<I32Vectors> {
        I32Vectors::of(CpuCapabilities::detect().level())
    }

    /// Quantizes `coefficients` both ways and compares with the division.
    fn assert_quantizes(step: u32, offset: u32, coefficients: &[i32]) {
        let quantizer = Quantizer::new(step).unwrap();
        let expected: Vec<i32> = coefficients
            .iter()
            .map(|&value| quantizer.quantize(value, offset))
            .collect();
        for vectors in [vectors(), None] {
            let mut actual = coefficients.to_vec();
            quantize_coefficients(vectors, quantizer, offset, &mut actual);
            assert_eq!(actual, expected, "step {step} offset {offset}");
        }
    }

    #[test]
    fn vector_quantization_matches_the_division_at_quotient_boundaries() {
        for step in codec_steps() {
            for offset in [step / 2, step / 8 * 3, 0, step - 1] {
                let top = u64::from(VECTOR_QUANTIZATION_DOMAIN - 1 - offset.min(step - 1));
                let step = u64::from(step);
                // Each side of the first quotient boundaries, of the last
                // ones in the vector domain, and of the domain's edge.
                let magnitudes = [0, 1, step - 1, step, step + 1, 2 * step - 1, 2 * step]
                    .into_iter()
                    .chain([top / step * step, top - 1, top, top + 1, top + 2])
                    .chain((0..8).map(|index| top * index / 8 / step * step + step / 2));
                let coefficients: Vec<i32> = magnitudes
                    .filter_map(|magnitude| i32::try_from(magnitude).ok())
                    .flat_map(|magnitude| [magnitude, -magnitude])
                    .chain([i32::MIN, i32::MAX])
                    .collect();
                let step = u32::try_from(step).unwrap();
                // Inside the domain, then one coefficient out of it.
                let inside: Vec<i32> = coefficients
                    .iter()
                    .copied()
                    .filter(|value| value.unsigned_abs() <= u32::try_from(top).unwrap())
                    .collect();
                assert_quantizes(step, offset, &inside);
                assert_quantizes(step, offset, &coefficients);
            }
        }
    }

    /// Every coefficient of the vector domain at every step the codec uses,
    /// with the offsets it uses. Run with
    /// `cargo test --release -p jxr-native -- --ignored`.
    #[test]
    #[ignore = "exhaustive over the vector domain; minutes in a release build"]
    fn vector_quantization_matches_the_division_over_its_whole_domain() {
        let Some(vectors) = vectors() else {
            return;
        };
        for step in codec_steps()
            .into_iter()
            .filter(|&step| step < VECTOR_QUANTIZATION_DOMAIN)
        {
            let quantizer = Quantizer::new(step).unwrap();
            for offset in [step / 2, step / 8 * 3] {
                let limit =
                    i32::try_from(VECTOR_QUANTIZATION_DOMAIN - offset.min(step - 1)).unwrap();
                let all: Vec<i32> = (-limit + 1..limit).collect();
                for chunk in all.chunks(1 << 16) {
                    let mut actual = chunk.to_vec();
                    quantize_coefficients(Some(vectors), quantizer, offset, &mut actual);
                    for (&value, &quantized) in chunk.iter().zip(&actual) {
                        assert_eq!(
                            quantized,
                            quantizer.quantize(value, offset),
                            "step {step} offset {offset} coefficient {value}"
                        );
                    }
                }
            }
        }
    }
}
