//! Checked capability-token SIMD for contiguous coefficient scaling.

use fearless_simd::{Simd, SimdBase};
use jxr_math::quantization::Quantizer;

use crate::CpuCapabilities;

use super::ReconstructionError;

const MIN_VECTOR_COEFFICIENTS: usize = 64;

pub(super) fn scale_coefficients(
    cpu: CpuCapabilities,
    quantizer: Quantizer,
    input: &[i32],
    output: &mut [i32],
) -> Result<bool, ReconstructionError> {
    if input.len() != output.len() {
        return Err(ReconstructionError::InvalidPlaneGeometry(
            "SIMD dequantization slice lengths differ",
        ));
    }
    let step = quantizer.step();
    let (used_simd, [minimum, maximum]) = scale_with_range(cpu, input, output, step);
    // Validation is fused into the scaling pass. Wrapped products written for
    // an out-of-range coefficient are never observed because the call fails.
    let [lower, upper] = representable_coefficients(step);
    if minimum < lower || maximum > upper {
        return Err(ReconstructionError::ArithmeticOverflow(
            "coefficient dequantization",
        ));
    }
    Ok(used_simd)
}

/// Returns the inclusive coefficient range whose product with `step` fits `i32`.
fn representable_coefficients(step: u32) -> [i32; 2] {
    let step = i64::from(step);
    // Truncating division rounds the negative bound toward zero, which is the
    // ceiling required for an inclusive lower bound.
    [i64::from(i32::MIN) / step, i64::from(i32::MAX) / step]
        .map(|bound| i32::try_from(bound).expect("a nonzero step narrows the i32 range"))
}

fn scale_with_range(
    cpu: CpuCapabilities,
    input: &[i32],
    output: &mut [i32],
    step: u32,
) -> (bool, [i32; 2]) {
    if input.len() >= MIN_VECTOR_COEFFICIENTS {
        let level = cpu.level();
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(avx2) = level.as_avx2() {
            return (true, scale_vectorized(avx2, input, output, step));
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(neon) = level.as_neon() {
            return (true, scale_vectorized(neon, input, output, step));
        }
    }
    (false, scale_scalar(input, output, step))
}

#[inline]
fn scale_vectorized<S: Simd>(simd: S, input: &[i32], output: &mut [i32], step: u32) -> [i32; 2] {
    let multiplier = i32::from_ne_bytes(step.to_ne_bytes());
    simd.vectorize(|| {
        let lanes = S::i32s::LEN;
        let vector_multiplier = S::i32s::splat(simd, multiplier);
        let mut minimum = S::i32s::splat(simd, i32::MAX);
        let mut maximum = S::i32s::splat(simd, i32::MIN);
        let mut input_chunks = input.chunks_exact(lanes);
        let mut output_chunks = output.chunks_exact_mut(lanes);
        for (source, destination) in input_chunks.by_ref().zip(output_chunks.by_ref()) {
            let coefficients = S::i32s::from_slice(simd, source);
            minimum = minimum.min(coefficients);
            maximum = maximum.max(coefficients);
            (coefficients * vector_multiplier).store_slice(destination);
        }
        let [tail_minimum, tail_maximum] = scale_scalar(
            input_chunks.remainder(),
            output_chunks.into_remainder(),
            step,
        );
        [
            minimum.reduce_min().min(tail_minimum),
            maximum.reduce_max().max(tail_maximum),
        ]
    })
}

/// Scales with wrapping products and returns the observed coefficient range.
#[inline]
fn scale_scalar(input: &[i32], output: &mut [i32], step: u32) -> [i32; 2] {
    let multiplier = i32::from_ne_bytes(step.to_ne_bytes());
    let mut range = [i32::MAX, i32::MIN];
    for (output, &coefficient) in output.iter_mut().zip(input) {
        range = [range[0].min(coefficient), range[1].max(coefficient)];
        *output = coefficient.wrapping_mul(multiplier);
    }
    range
}

#[cfg(test)]
mod tests {
    use jxr_math::quantization::Quantizer;

    use super::scale_coefficients;
    use crate::CpuCapabilities;

    #[test]
    fn selected_level_matches_exact_scalar_dequantization() {
        let input: Vec<_> = (-128..128).collect();
        let quantizer = Quantizer::new(37).unwrap();
        let mut output = vec![0; input.len()];
        scale_coefficients(CpuCapabilities::detect(), quantizer, &input, &mut output).unwrap();
        let expected: Vec<_> = input
            .iter()
            .map(|&value| quantizer.dequantize(value).unwrap())
            .collect();
        assert_eq!(output, expected);
    }

    #[test]
    fn validation_rejects_overflow_before_vector_arithmetic() {
        let input = vec![i32::MAX; 64];
        let mut output = vec![0; input.len()];
        assert!(
            scale_coefficients(
                CpuCapabilities::detect(),
                Quantizer::new(2).unwrap(),
                &input,
                &mut output,
            )
            .is_err()
        );
    }

    #[test]
    fn fused_range_check_matches_widened_products_at_the_boundaries() {
        for step in [
            1_u32,
            2,
            3,
            37,
            113,
            65_536,
            i32::MAX.unsigned_abs(),
            1 << 31,
            u32::MAX,
        ] {
            let quantizer = Quantizer::new(step).unwrap();
            let [lower, upper] = super::representable_coefficients(step);
            let candidates = [
                i32::MIN,
                lower.saturating_sub(1),
                lower,
                -1,
                0,
                1,
                upper,
                upper.saturating_add(1),
                i32::MAX,
            ];
            for candidate in candidates {
                // Place the candidate in the vector body and in the scalar tail.
                for position in [5, 66] {
                    let mut input = vec![0; 67];
                    input[position] = candidate;
                    let mut output = vec![0; input.len()];
                    let result = scale_coefficients(
                        CpuCapabilities::detect(),
                        quantizer,
                        &input,
                        &mut output,
                    );
                    match quantizer.dequantize(candidate) {
                        Ok(expected) => {
                            result.unwrap();
                            assert_eq!(output[position], expected, "step {step} value {candidate}");
                        }
                        Err(_) => assert!(result.is_err(), "step {step} value {candidate}"),
                    }
                }
            }
        }
    }

    #[test]
    fn coefficient_scaling_preserves_negative_values_and_tail() {
        let input: Vec<_> = (-34..35).collect();
        let quantizer = Quantizer::new(113).unwrap();
        let mut output = vec![0; input.len()];
        scale_coefficients(CpuCapabilities::detect(), quantizer, &input, &mut output).unwrap();
        let expected: Vec<_> = input
            .iter()
            .map(|&value| quantizer.dequantize(value).unwrap())
            .collect();
        assert_eq!(output, expected);
    }
}
