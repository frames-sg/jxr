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

    /// Scale a coefficient, returning an error when the result exceeds `i32`.
    pub fn dequantize(self, coefficient: i32) -> Result<i32, MathError> {
        let widened = i64::from(coefficient) * i64::from(self.step());
        checked_i64_to_i32(widened)
    }
}
