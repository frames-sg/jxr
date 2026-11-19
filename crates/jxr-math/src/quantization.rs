//! Quantization arithmetic with explicit signed rounding.

use core::num::NonZeroU32;

use crate::arithmetic::{MathError, checked_i64_to_i32};

/// A nonzero scalar quantization step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Quantizer(NonZeroU32);

impl Quantizer {
    /// Construct a quantizer. Zero is rejected because it has no inverse scale.
    #[must_use]
    pub const fn new(step: u32) -> Option<Self> {
        match NonZeroU32::new(step) {
            Some(step) => Some(Self(step)),
            None => None,
        }
    }

    /// Return the nonzero quantization step.
    #[must_use]
    pub const fn step(self) -> u32 {
        self.0.get()
    }

    /// Divide a coefficient by the step, rounding its magnitude toward zero
    /// after adding `offset`.
    ///
    /// An offset of half the step rounds to nearest; a smaller one widens the
    /// zero bin. Offsets of at least one step are clamped to `step - 1` so the
    /// result never exceeds the coefficient in magnitude, which also makes a
    /// step of one the identity.
    #[must_use]
    pub const fn quantize(self, coefficient: i32, offset: u32) -> i32 {
        let step = self.step();
        let offset = if offset >= step { step - 1 } else { offset };
        let magnitude = coefficient.unsigned_abs();
        // `(magnitude + offset) / step` without widening: the offset is below
        // one step, so it can only carry the quotient up by one.
        let carry = (magnitude % step >= step - offset) as u32;
        let quotient = magnitude / step + carry;
        // The quotient is at most the magnitude, so only `i32::MIN` at a step
        // of one reaches 2^31, and wrapping negation restores it.
        if coefficient < 0 {
            quotient.cast_signed().wrapping_neg()
        } else {
            quotient.cast_signed()
        }
    }

    /// Scale a coefficient, returning an error when the result exceeds `i32`.
    pub fn dequantize(self, coefficient: i32) -> Result<i32, MathError> {
        let widened = i64::from(coefficient) * i64::from(self.step());
        checked_i64_to_i32(widened)
    }
}
