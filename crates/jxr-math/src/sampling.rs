//! Exact integer interpolation used by the T.832 example chroma upsampler.

use crate::arithmetic::{MathError, checked_add, checked_mul};

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
/// policy. Arithmetic and `+4` rounding follow T.832 Table 180 exactly.
pub fn upsample_chroma_pair(
    previous: i32,
    current: i32,
    next: i32,
    centering: ChromaCentering,
) -> Result<[i32; 2], MathError> {
    let [h0, h1, h2, h3] = centering.coefficients();
    let even = weighted_pair(previous, h2, current, h3)?;
    let odd = weighted_pair(current, h0, next, h1)?;
    Ok([even, odd])
}

fn weighted_pair(
    first: i32,
    first_weight: i32,
    second: i32,
    second_weight: i32,
) -> Result<i32, MathError> {
    checked_mul(first, first_weight)
        .and_then(|first| {
            checked_mul(second, second_weight).and_then(|second| checked_add(first, second))
        })
        .and_then(|sum| checked_add(sum, 4))
        .map(|sum| sum >> 3)
}

#[cfg(test)]
mod tests {
    use super::{ChromaCentering, upsample_chroma_pair};

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
