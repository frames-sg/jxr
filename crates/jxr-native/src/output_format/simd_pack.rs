//! Capability-token dispatch for common unsigned planar packing.

use fearless_simd::{Bytes, Level, Simd, SimdBase, SimdNarrow};

use super::OutputFormatError;

const MIN_VECTOR_SAMPLES: usize = 64;

pub(super) fn append_u8(
    level: Level,
    input: &[i32],
    stride: usize,
    start: [usize; 2],
    dimensions: [usize; 2],
    scaled: bool,
    output: &mut Vec<u8>,
) -> Result<bool, OutputFormatError> {
    let [width, height] = dimensions;
    let count = width
        .checked_mul(height)
        .ok_or_else(|| OutputFormatError::arithmetic("calculating SIMD output length"))?;
    let output_start = output.len();
    let output_end = output_start
        .checked_add(count)
        .ok_or_else(|| OutputFormatError::arithmetic("growing SIMD output"))?;
    output.resize(output_end, 0);
    pack_u8_into(
        level,
        input,
        stride,
        start,
        dimensions,
        scaled,
        &mut output[output_start..output_end],
    )
}

pub(super) fn pack_u8_into(
    level: Level,
    input: &[i32],
    stride: usize,
    start: [usize; 2],
    dimensions: [usize; 2],
    scaled: bool,
    output: &mut [u8],
) -> Result<bool, OutputFormatError> {
    let [width, height] = dimensions;
    let count = width
        .checked_mul(height)
        .ok_or_else(|| OutputFormatError::arithmetic("calculating SIMD output length"))?;
    if output.len() != count {
        return Err(OutputFormatError::UnsupportedCombination {
            combination: "SIMD U8 destination length differs from the output contract",
        });
    }
    if height > 0 {
        // Row starts increase with `y`, so bounding the final row bounds every row.
        let last_row_end = start[1]
            .checked_add(height - 1)
            .and_then(|row| row.checked_mul(stride))
            .and_then(|row| row.checked_add(start[0]))
            .and_then(|row| row.checked_add(width))
            .ok_or_else(|| OutputFormatError::arithmetic("calculating SIMD input row"))?;
        if last_row_end > input.len() {
            return Err(OutputFormatError::InvalidPlane {
                component: None,
                reason: "SIMD input row exceeds component plane",
            });
        }
    }
    let bias = PackBias::new(scaled);
    let (used_simd, maximum) =
        match pack_accelerated(level, input, stride, start, dimensions, bias, output) {
            Some(maximum) => (true, maximum),
            None => (
                false,
                pack_scalar(input, stride, start, dimensions, bias, output),
            ),
        };
    // Validation is fused into the packing pass; output written for an
    // overflowing sample is never observed because the call fails.
    if maximum > i32::MAX - bias.addition {
        return Err(OutputFormatError::arithmetic("adding output bias"));
    }
    Ok(used_simd)
}

#[derive(Clone, Copy)]
struct PackBias {
    /// Output bias plus rounding, added before the down-shift.
    addition: i32,
    shift: u32,
}

impl PackBias {
    const fn new(scaled: bool) -> Self {
        if scaled {
            Self {
                addition: 1_024 + 3,
                shift: 3,
            }
        } else {
            Self {
                addition: 128,
                shift: 0,
            }
        }
    }

    /// Packs one sample. Saturation only differs from the checked result for
    /// samples that the caller rejects after the pass.
    #[inline]
    fn pack(self, sample: i32) -> u8 {
        u8::try_from((sample.saturating_add(self.addition) >> self.shift).clamp(0, 255))
            .expect("sample is clipped to u8")
    }
}

/// Returns the maximum input sample when a vector implementation packed the output.
fn pack_accelerated(
    level: Level,
    input: &[i32],
    stride: usize,
    start: [usize; 2],
    dimensions: [usize; 2],
    bias: PackBias,
    output: &mut [u8],
) -> Option<i32> {
    if output.len() < MIN_VECTOR_SAMPLES {
        return None;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(avx2) = level.as_avx2() {
        return pack_vectorized(avx2, input, stride, start, dimensions, bias, output);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = level.as_neon() {
        return pack_vectorized(neon, input, stride, start, dimensions, bias, output);
    }
    None
}

#[inline]
fn pack_vectorized<S: Simd>(
    simd: S,
    input: &[i32],
    stride: usize,
    start: [usize; 2],
    dimensions: [usize; 2],
    bias: PackBias,
    output: &mut [u8],
) -> Option<i32> {
    let lanes = S::i32s::LEN;
    let packed_lanes = S::u8s::LEN;
    let [width, height] = dimensions;
    if width < packed_lanes {
        return None;
    }

    Some(simd.vectorize(|| {
        let addition = S::i32s::splat(simd, bias.addition);
        let zero = S::i32s::splat(simd, 0);
        let max = S::i32s::splat(simd, 255);
        let mut peak = S::i32s::splat(simd, i32::MIN);
        let mut tail_peak = i32::MIN;
        let vector_width = width / packed_lanes * packed_lanes;
        for y in 0..height {
            let source_start = (start[1] + y) * stride + start[0];
            let source = &input[source_start..source_start + width];
            let destination = &mut output[y * width..(y + 1) * width];
            for (source, destination) in source[..vector_width]
                .chunks_exact(packed_lanes)
                .zip(destination[..vector_width].chunks_exact_mut(packed_lanes))
            {
                let mut quarter = |index: usize| {
                    let samples =
                        S::i32s::from_slice(simd, &source[index * lanes..(index + 1) * lanes]);
                    peak = peak.max(samples);
                    ((samples + addition) >> bias.shift)
                        .max(zero)
                        .min(max)
                        .bitcast::<S::u32s>()
                };
                let first = quarter(0);
                let second = quarter(1);
                let third = quarter(2);
                let fourth = quarter(3);
                first
                    .saturating_narrow(second)
                    .saturating_narrow(third.saturating_narrow(fourth))
                    .store_slice(destination);
            }
            tail_peak = tail_peak.max(pack_row(
                &source[vector_width..],
                &mut destination[vector_width..],
                bias,
            ));
        }
        peak.reduce_max().max(tail_peak)
    }))
}

/// Packs one row and returns its maximum input sample.
#[inline]
fn pack_row(source: &[i32], destination: &mut [u8], bias: PackBias) -> i32 {
    let mut peak = i32::MIN;
    for (destination, &sample) in destination.iter_mut().zip(source) {
        peak = peak.max(sample);
        *destination = bias.pack(sample);
    }
    peak
}

fn pack_scalar(
    input: &[i32],
    stride: usize,
    start: [usize; 2],
    dimensions: [usize; 2],
    bias: PackBias,
    output: &mut [u8],
) -> i32 {
    let [width, height] = dimensions;
    let mut peak = i32::MIN;
    for y in 0..height {
        let source_start = (start[1] + y) * stride + start[0];
        peak = peak.max(pack_row(
            &input[source_start..source_start + width],
            &mut output[y * width..(y + 1) * width],
            bias,
        ));
    }
    peak
}

#[cfg(test)]
mod tests {
    use fearless_simd::Level;

    use super::append_u8;

    #[test]
    fn selected_simd_level_matches_scalar_u8_semantics() {
        let input: Vec<_> = (-96..96).map(|value| value * 7).collect();
        for scaled in [false, true] {
            let mut output = Vec::new();
            append_u8(
                Level::new(),
                &input,
                24,
                [2, 1],
                [20, 6],
                scaled,
                &mut output,
            )
            .unwrap();
            let bias = if scaled { 1_024 } else { 128 };
            let rounding = if scaled { 3 } else { 0 };
            let shift = if scaled { 3 } else { 0 };
            let expected: Vec<_> = (0..6)
                .flat_map(|y| &input[(y + 1) * 24 + 2..(y + 1) * 24 + 22])
                .map(|&sample| {
                    u8::try_from(((sample + bias + rounding) >> shift).clamp(0, 255))
                        .expect("sample is clipped to u8")
                })
                .collect();
            assert_eq!(output, expected);
        }
    }

    #[test]
    fn fused_bias_check_rejects_overflow_on_vector_and_scalar_paths() {
        for (width, height) in [(64, 2), (7, 3)] {
            for scaled in [false, true] {
                let addition = if scaled { 1_027 } else { 128 };
                for (sample, accepted) in [
                    (i32::MAX - addition, true),
                    (i32::MAX - addition + 1, false),
                ] {
                    for position in [0, width * height - 1] {
                        let mut input = vec![0; width * height];
                        input[position] = sample;
                        let mut output = Vec::new();
                        let result = append_u8(
                            Level::new(),
                            &input,
                            width,
                            [0, 0],
                            [width, height],
                            scaled,
                            &mut output,
                        );
                        assert_eq!(
                            result.is_ok(),
                            accepted,
                            "{width}x{height} {scaled} {sample}"
                        );
                        if accepted {
                            assert_eq!(output[position], 255);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_rows_beyond_the_component_plane() {
        let input = vec![0; 99];
        let mut output = Vec::new();
        assert!(
            append_u8(
                Level::new(),
                &input,
                10,
                [0, 0],
                [10, 10],
                false,
                &mut output
            )
            .is_err()
        );
    }

    #[test]
    fn packing_matches_scalar_at_row_tails_and_clamp_boundaries() {
        let width = 37;
        let height = 3;
        let stride = 41;
        let input: Vec<i32> = (0..stride * (height + 1))
            .map(|index| match index % 5 {
                0 => -2_000,
                1 => -128,
                2 => 0,
                3 => 1_024,
                _ => i32::MAX - 1_027,
            })
            .collect();
        for scaled in [false, true] {
            let mut output = Vec::new();
            append_u8(
                Level::new(),
                &input,
                stride,
                [2, 1],
                [width, height],
                scaled,
                &mut output,
            )
            .unwrap();
            let bias = if scaled { 1_024 } else { 128 };
            let rounding = if scaled { 3 } else { 0 };
            let shift = if scaled { 3 } else { 0 };
            let expected: Vec<_> = (1..=height)
                .flat_map(|y| &input[y * stride + 2..y * stride + 2 + width])
                .map(|&sample| {
                    u8::try_from(((sample + bias + rounding) >> shift).clamp(0, 255))
                        .expect("sample is clipped to u8")
                })
                .collect();
            assert_eq!(output, expected);
        }
    }
}
