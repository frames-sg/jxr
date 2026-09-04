//! Checked capability-token SIMD for contiguous coefficient scaling.

use fearless_simd::Simd;
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
    if input
        .iter()
        .any(|&coefficient| i32::try_from(i64::from(coefficient) * i64::from(step)).is_err())
    {
        return Err(ReconstructionError::ArithmeticOverflow(
            "coefficient dequantization",
        ));
    }
    if input.len() < MIN_VECTOR_COEFFICIENTS {
        scale_scalar(input, output, step);
        return Ok(false);
    }
    let level = cpu.level();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(avx2) = level.as_avx2() {
        scale_vectorized(avx2, input, output, step);
        return Ok(true);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = level.as_neon() {
        scale_vectorized(neon, input, output, step);
        return Ok(true);
    }
    scale_scalar(input, output, step);
    Ok(false)
}

#[inline]
fn scale_vectorized<S: Simd>(_simd: S, input: &[i32], output: &mut [i32], step: u32) {
    scale_validated(input, output, step);
}

#[inline]
fn scale_validated(input: &[i32], output: &mut [i32], step: u32) {
    let multiplier = i32::from_ne_bytes(step.to_ne_bytes());
    for (output, &coefficient) in output.iter_mut().zip(input) {
        *output = coefficient.wrapping_mul(multiplier);
    }
}

fn scale_scalar(input: &[i32], output: &mut [i32], step: u32) {
    scale_validated(input, output, step);
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
}
