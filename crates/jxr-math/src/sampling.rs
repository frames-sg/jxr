//! Exact integer interpolation used by the T.832 example chroma upsampler,
//! and the decimation an encoder pairs with it.

use crate::arithmetic::MathError;

/// A known chroma-grid offset in quarter-luma-sample units.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChromaCentering(u8);

impl ChromaCentering {
    /// Validate a T.832 chroma-centering value supported by the example filter.
    ///
    /// Values zero through four have normative example coefficients. Value
    /// seven means unknown positioning and therefore has no exact filter.
    #[must_use]
    pub const fn new(value: u8) -> Option<Self> {
        if value <= 4 { Some(Self(value)) } else { None }
    }

    /// Quarter-sample offset represented by this value.
    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }

    /// Coefficients `[h0, h1, h2, h3]` from T.832 Table 181.
    #[inline]
    #[must_use]
    pub const fn coefficients(self) -> [i32; 4] {
        match self.0 {
            0 => [4, 4, 0, 8],
            1 => [5, 3, 1, 7],
            2 => [6, 2, 2, 6],
            3 => [7, 1, 3, 5],
            4 => [8, 0, 4, 4],
            _ => unreachable!(),
        }
    }
}

/// Produce the two output samples associated with one subsampled chroma value.
///
/// `previous` and `next` must already reflect the caller's edge-extension
/// policy. Arithmetic and `+4` rounding follow T.832 Table 180 exactly, for
/// samples of any size: each output lies between the two samples it weighs.
#[inline]
pub fn upsample_chroma_pair(
    previous: i32,
    current: i32,
    next: i32,
    centering: ChromaCentering,
) -> Result<[i32; 2], MathError> {
    let [h0, h1, h2, h3] = centering.coefficients();
    Ok([
        weighted_pair(previous, h2, current, h3),
        weighted_pair(current, h0, next, h1),
    ])
}

/// Chroma centering of the samples [`downsample_chroma`] produces: each one
/// is co-sited with the even full-resolution sample it replaces, so the grid
/// has no offset from the luma grid in either direction.
pub const DOWNSAMPLED_CENTERING: ChromaCentering = ChromaCentering(0);

/// Halve the chroma resolution along one line of samples.
///
/// `sample` reads a full-resolution line of `length` samples, which must not
/// be empty, and the result is subsampled sample `index`, co-sited with
/// full-resolution sample `2 * index`. T.832 leaves decimation to the
/// encoder. This is the filter of the T.835 reference encoder: the binomial
/// low-pass 1, 4, 6, 4, 1 over 16, rounded to nearest, with positions beyond
/// either end of the line mirrored about its end sample. The filter is
/// symmetric about the sample it keeps, which is what
/// [`DOWNSAMPLED_CENTERING`] declares.
///
/// The weights sum to sixteen, so the result lies within the range of the
/// samples read.
pub fn downsample_chroma(
    sample: impl Fn(usize) -> i32,
    length: usize,
    index: usize,
) -> Result<i32, MathError> {
    const WEIGHTS: [i64; 5] = [1, 4, 6, 4, 1];
    let last = length.saturating_sub(1);
    let centre = 2 * index;
    let mut sum = 8_i64;
    for (tap, weight) in WEIGHTS.into_iter().enumerate() {
        // Taps run from two samples before the centre to two after it.
        let position = match (centre + tap).checked_sub(2) {
            // Mirror about the first sample: -1 reads 1 and -2 reads 2.
            None => 2 - (centre + tap),
            // Mirror about the last sample: one beyond it reads one before.
            Some(position) if position > last => last.saturating_sub(position - last),
            Some(position) => position,
        };
        // A line shorter than the filter clamps what mirroring cannot reach.
        sum += weight * i64::from(sample(position.min(last)));
    }
    i32::try_from(sum >> 4).map_err(|_| crate::arithmetic::overflow("chroma decimation"))
}

/// `(first * first_weight + second * second_weight + 4) >> 3` for weights
/// that are not negative and sum to eight.
#[inline]
const fn weighted_pair(first: i32, first_weight: i32, second: i32, second_weight: i32) -> i32 {
    // The rounded mean lies between the two samples whatever their size;
    // only the sum before the division could need more than 32 bits. So
    // each sample is split into its multiple of eight and the remainder
    // 0..=7: the eighths weigh to at most an eighth of the range times
    // eight, and the remainders to at most 7 * 8 + 4, which carries at most
    // seven into them. A row of these is plain 32-bit lane arithmetic.
    (first >> 3) * first_weight
        + (second >> 3) * second_weight
        + (((first & 7) * first_weight + (second & 7) * second_weight + 4) >> 3)
}

#[cfg(test)]
mod tests {
    use super::{ChromaCentering, DOWNSAMPLED_CENTERING, downsample_chroma, upsample_chroma_pair};

    #[test]
    fn interpolation_preserves_constant_extremes_and_rounds_their_midpoint() {
        for value in 0..=4 {
            let centering = ChromaCentering::new(value).unwrap();
            for sample in [i32::MAX, i32::MIN] {
                assert_eq!(
                    upsample_chroma_pair(sample, sample, sample, centering).unwrap(),
                    [sample, sample]
                );
            }
        }
        assert_eq!(
            upsample_chroma_pair(
                i32::MIN,
                i32::MAX,
                i32::MIN,
                ChromaCentering::new(0).unwrap()
            )
            .unwrap(),
            [i32::MAX, 0]
        );
    }

    #[test]
    fn interpolation_equals_the_table_180_sum_computed_in_64_bits() {
        // Samples around every multiple of eight near zero and at both ends
        // of the range, where a remainder carries or a sum would overflow.
        let samples: [i32; 49] = core::array::from_fn(|index| {
            let offset = i32::try_from(index % 7).unwrap() - 3;
            match index / 7 {
                0 => i32::MIN + 3 + offset,
                1 => i32::MAX - 3 + offset,
                2 => offset,
                3 => 8 + offset,
                4 => -8 + offset,
                5 => 0x1234_5678 + offset,
                _ => -0x1234_5678 + offset,
            }
        });
        let exact = |first: i32, first_weight: i32, second: i32, second_weight: i32| {
            let sum = i64::from(first) * i64::from(first_weight)
                + i64::from(second) * i64::from(second_weight)
                + 4;
            i32::try_from(sum >> 3).unwrap()
        };
        for value in 0..=4 {
            let centering = ChromaCentering::new(value).unwrap();
            let [h0, h1, h2, h3] = centering.coefficients();
            for previous in samples {
                for current in samples {
                    for next in [previous, current, samples[usize::from(value) * 9]] {
                        assert_eq!(
                            upsample_chroma_pair(previous, current, next, centering).unwrap(),
                            [
                                exact(previous, h2, current, h3),
                                exact(current, h0, next, h1)
                            ],
                            "{previous} {current} {next} at centering {value}"
                        );
                    }
                }
            }
        }
    }

    /// Every subsampled sample of `line`.
    fn downsampled(line: &[i32]) -> [i32; 8] {
        assert_eq!(line.len(), 16);
        core::array::from_fn(|index| {
            downsample_chroma(|position| line[position], line.len(), index).unwrap()
        })
    }

    #[test]
    fn decimation_keeps_a_constant_line_constant() {
        for value in [0, 1, -1, 127, -128, 32_767, -32_768, i32::MAX, i32::MIN] {
            assert_eq!(downsampled(&[value; 16]), [value; 8], "{value}");
        }
    }

    #[test]
    fn decimation_weights_the_five_samples_around_each_even_position() {
        // An impulse at an even position is kept with weight 6 and reaches
        // both neighbouring outputs with weight 1; one at an odd position
        // reaches the outputs on either side with weight 4.
        let mut even = [0_i32; 16];
        even[6] = 160;
        assert_eq!(downsampled(&even), [0, 0, 10, 60, 10, 0, 0, 0]);
        let mut odd = [0_i32; 16];
        odd[7] = 160;
        assert_eq!(downsampled(&odd), [0, 0, 0, 40, 40, 0, 0, 0]);
    }

    #[test]
    fn decimation_mirrors_about_the_end_samples() {
        // The first output reads positions -2 and -1 as 2 and 1.
        let mut line = [0_i32; 16];
        line[1] = 16;
        assert_eq!(downsampled(&line)[0], 8);
        line = [0; 16];
        line[2] = 16;
        assert_eq!(downsampled(&line)[0], 2);
        // The last output is centred on position 14 and reads 16 as 14, so
        // the last sample counts once and position 14 seven times.
        line = [0; 16];
        line[15] = 16;
        assert_eq!(downsampled(&line)[7], 4);
        line = [0; 16];
        line[14] = 16;
        assert_eq!(downsampled(&line)[7], 7);
    }

    #[test]
    fn decimation_rounds_to_nearest_with_a_floor_shift() {
        // 6/16 and 10/16 of one unit round to 0 and 1; their negatives to
        // 0 and -1, the arithmetic shift of the T.835 filter.
        let mut line = [0_i32; 16];
        line[6] = 1;
        assert_eq!(downsampled(&line)[3], 0);
        line[5] = 1;
        assert_eq!(downsampled(&line)[3], 1);
        line[5] = -1;
        line[6] = -1;
        assert_eq!(downsampled(&line)[3], -1);
        line[5] = 0;
        assert_eq!(downsampled(&line)[3], 0);
    }

    #[test]
    fn a_line_shorter_than_the_filter_is_clamped() {
        assert_eq!(downsample_chroma(|_| 9, 1, 0).unwrap(), 9);
        let pair = [16, 32];
        // Positions -2 to 2 read samples 1, 1, 0, 1 and 0 of the pair.
        assert_eq!(
            downsample_chroma(|position| pair[position], 2, 0).unwrap(),
            (16 * 7 + 32 * 9 + 8) >> 4
        );
    }

    #[test]
    fn decimation_then_interpolation_reproduces_a_ramp() {
        // A ramp is its own low-pass, so the samples kept are the even ones
        // and interpolating at the declared centering restores the odd ones.
        // Only the ends differ, where decimation mirrors the ramp back on
        // itself and interpolation repeats the last sample.
        for slope in [-7, -1, 3, 10] {
            let line: [i32; 16] =
                core::array::from_fn(|index| slope * i32::try_from(index).unwrap() - 20);
            let half = downsampled(&line);
            for index in 1..7 {
                assert_eq!(half[index], line[2 * index], "slope {slope}");
            }
            for index in 0..8_usize {
                let [even, odd] = upsample_chroma_pair(
                    half[index.saturating_sub(1)],
                    half[index],
                    half[(index + 1).min(7)],
                    DOWNSAMPLED_CENTERING,
                )
                .unwrap();
                let bound = if (1..6).contains(&index) {
                    0
                } else {
                    2 * slope.abs()
                };
                assert!((even - line[2 * index]).abs() <= bound, "slope {slope}");
                assert!((odd - line[2 * index + 1]).abs() <= bound, "slope {slope}");
            }
        }
    }

    #[test]
    fn the_declared_centering_is_the_co_sited_one() {
        // At this centering the interpolator returns each subsampled sample
        // unchanged at the even position, where decimation took it from.
        assert_eq!(DOWNSAMPLED_CENTERING, ChromaCentering::new(0).unwrap());
        assert_eq!(
            upsample_chroma_pair(-50, 8, 90, DOWNSAMPLED_CENTERING).unwrap()[0],
            8
        );
    }

    #[test]
    fn centering_zero_aligns_even_output_with_current_sample() {
        assert_eq!(
            upsample_chroma_pair(0, 8, 16, ChromaCentering::new(0).unwrap()).unwrap(),
            [8, 12]
        );
    }

    #[test]
    fn centering_four_aligns_odd_output_with_current_sample() {
        assert_eq!(
            upsample_chroma_pair(0, 8, 16, ChromaCentering::new(4).unwrap()).unwrap(),
            [4, 8]
        );
    }

    #[test]
    fn negative_interpolation_uses_normative_signed_shift() {
        assert_eq!(
            upsample_chroma_pair(-9, -4, 3, ChromaCentering::new(2).unwrap()).unwrap(),
            [-5, -2]
        );
    }

    #[test]
    fn unknown_centering_has_no_exact_filter() {
        assert!(ChromaCentering::new(7).is_none());
    }
}
