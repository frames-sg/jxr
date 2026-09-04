//! Checked and saturating integer arithmetic with explicit rounding semantics.

use core::fmt;

/// An integer operation could not be represented in the requested type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MathError {
    operation: &'static str,
}

impl MathError {
    /// Create an overflow error for `operation`.
    #[must_use]
    pub const fn overflow(operation: &'static str) -> Self {
        Self { operation }
    }

    /// Return the operation which overflowed.
    #[must_use]
    pub const fn operation(self) -> &'static str {
        self.operation
    }
}

impl fmt::Display for MathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "integer overflow during {}", self.operation)
    }
}

/// Add two signed coefficients, returning an error on overflow.
pub fn checked_add(left: i32, right: i32) -> Result<i32, MathError> {
    left.checked_add(right)
        .ok_or_else(|| MathError::overflow("addition"))
}

/// Subtract two signed coefficients, returning an error on overflow.
pub fn checked_sub(left: i32, right: i32) -> Result<i32, MathError> {
    left.checked_sub(right)
        .ok_or_else(|| MathError::overflow("subtraction"))
}

/// Multiply two signed coefficients, returning an error on overflow.
pub fn checked_mul(left: i32, right: i32) -> Result<i32, MathError> {
    left.checked_mul(right)
        .ok_or_else(|| MathError::overflow("multiplication"))
}

/// Convert an intermediate to a coefficient, returning an error on overflow.
pub fn checked_i64_to_i32(value: i64) -> Result<i32, MathError> {
    i32::try_from(value).map_err(|_| MathError::overflow("i64 to i32 conversion"))
}

#[cfg(test)]
mod tests {
    use super::{checked_add, checked_i64_to_i32, checked_mul, checked_sub};

    #[test]
    fn checked_operations_reject_overflow() {
        assert!(checked_add(i32::MAX, 1).is_err());
        assert!(checked_sub(i32::MIN, 1).is_err());
        assert!(checked_mul(i32::MAX, 2).is_err());
        assert!(checked_i64_to_i32(i64::MAX).is_err());
    }
}
