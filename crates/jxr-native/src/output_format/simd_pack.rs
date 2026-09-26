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
    let addition = if scaled { 1_027 } else { 128 };
    for y in 0..height {
        let row_start = (start[1] + y)
            .checked_mul(stride)
            .and_then(|row| row.checked_add(start[0]))
            .ok_or_else(|| OutputFormatError::arithmetic("calculating SIMD input row"))?;
        let row =
            input
                .get(row_start..row_start + width)
                .ok_or(OutputFormatError::InvalidPlane {
                    component: None,
                    reason: "SIMD input row exceeds component plane",
                })?;
        if row.iter().any(|&sample| sample > i32::MAX - addition) {
            return Err(OutputFormatError::arithmetic("adding output bias"));
        }
    }
    if count >= MIN_VECTOR_SAMPLES
        && pack_accelerated(level, input, stride, start, dimensions, scaled, output)
    {
        return Ok(true);
    }
    pack_scalar(input, stride, start, dimensions, scaled, output);
    Ok(false)
}

fn pack_accelerated(
    level: Level,
    input: &[i32],
    stride: usize,
    start: [usize; 2],
    dimensions: [usize; 2],
    scaled: bool,
    output: &mut [u8],
) -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(avx2) = level.as_avx2() {
        return pack_vectorized(avx2, input, stride, start, dimensions, scaled, output);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = level.as_neon() {
        return pack_vectorized(neon, input, stride, start, dimensions, scaled, output);
    }
    false
}

#[inline]
fn pack_vectorized<S: Simd>(
    simd: S,
    input: &[i32],
    stride: usize,
    start: [usize; 2],
    dimensions: [usize; 2],
    scaled: bool,
    output: &mut [u8],
) -> bool {
    let lanes = S::i32s::LEN;
    let packed_lanes = S::u8s::LEN;
    let [width, height] = dimensions;
    if width < packed_lanes {
        return false;
    }

    simd.vectorize(|| {
        let bias = if scaled { 1_024 } else { 128 };
        let rounding = if scaled { 3 } else { 0 };
        let addition = bias + rounding;
        let shift = if scaled { 3 } else { 0 };
        let zero = S::i32s::splat(simd, 0);
        let max = S::i32s::splat(simd, 255);
        let vector_width = width / packed_lanes * packed_lanes;
        for y in 0..height {
            let source_start = (start[1] + y) * stride + start[0];
            let source = &input[source_start..source_start + width];
            let destination = &mut output[y * width..(y + 1) * width];
            for (source, destination) in source[..vector_width]
                .chunks_exact(packed_lanes)
                .zip(destination[..vector_width].chunks_exact_mut(packed_lanes))
            {
                let first = pack_lanes(simd, &source[..lanes], addition, shift, zero, max);
                let second =
                    pack_lanes(simd, &source[lanes..2 * lanes], addition, shift, zero, max);
                let third = pack_lanes(
                    simd,
                    &source[2 * lanes..3 * lanes],
                    addition,
                    shift,
                    zero,
                    max,
                );
                let fourth = pack_lanes(simd, &source[3 * lanes..], addition, shift, zero, max);
                first
                    .saturating_narrow(second)
                    .saturating_narrow(third.saturating_narrow(fourth))
                    .store_slice(destination);
            }
            for (destination, &sample) in destination[vector_width..]
                .iter_mut()
                .zip(&source[vector_width..])
            {
                *destination = u8::try_from(((sample + bias + rounding) >> shift).clamp(0, 255))
                    .expect("sample is clipped to u8");
            }
        }
    });
    true
}

#[expect(
    clippy::inline_always,
    reason = "SIMD helper must inline into the target-feature vectorize context"
)]
#[inline(always)]
fn pack_lanes<S: Simd>(
    simd: S,
    source: &[i32],
    addition: i32,
    shift: u32,
    zero: S::i32s,
    max: S::i32s,
) -> S::u32s {
    ((S::i32s::from_slice(simd, source) + addition) >> shift)
        .max(zero)
        .min(max)
        .bitcast()
}

#[inline]
fn pack_rows(
    input: &[i32],
    stride: usize,
    start: [usize; 2],
    dimensions: [usize; 2],
    scaled: bool,
    output: &mut [u8],
) {
    let [width, height] = dimensions;
    let bias = if scaled { 1_024 } else { 128 };
    let rounding = if scaled { 3 } else { 0 };
    let shift = if scaled { 3 } else { 0 };
    for y in 0..height {
        let source_start = (start[1] + y) * stride + start[0];
        let source = &input[source_start..source_start + width];
        let destination = &mut output[y * width..(y + 1) * width];
        for (destination, &sample) in destination.iter_mut().zip(source) {
            *destination = u8::try_from(((sample + bias + rounding) >> shift).clamp(0, 255))
                .expect("sample is clipped to u8");
        }
    }
}

fn pack_scalar(
    input: &[i32],
    stride: usize,
    start: [usize; 2],
    dimensions: [usize; 2],
    scaled: bool,
    output: &mut [u8],
) {
    pack_rows(input, stride, start, dimensions, scaled, output);
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
