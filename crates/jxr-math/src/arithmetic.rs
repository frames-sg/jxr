//! Checked and saturating integer arithmetic with explicit rounding semantics.

#[cfg(not(jxr_math_device))]
use core::fmt;

/// An integer operation could not be represented in the requested type.
#[cfg(not(jxr_math_device))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MathError {
    operation: &'static str,
}

// CUDA Oxide currently requires primitive enum payloads. Device callers record
// the failed stage in their status word; host errors retain operation strings.
/// Numeric arithmetic error used by the standalone CUDA Oxide device build.
#[cfg(jxr_math_device)]
pub type MathError = u32;

#[cfg(not(jxr_math_device))]
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

#[cfg(not(jxr_math_device))]
impl fmt::Display for MathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "integer overflow during {}", self.operation)
    }
}

/// Construct an arithmetic error for host or device code.
#[must_use]
#[inline]
pub(crate) const fn overflow(operation: &'static str) -> MathError {
    #[cfg(not(jxr_math_device))]
    {
        MathError::overflow(operation)
    }
    #[cfg(jxr_math_device)]
    {
        let _ = operation;
        1
    }
}

/// Add two signed coefficients, returning an error on overflow.
#[inline]
pub fn checked_add(left: i32, right: i32) -> Result<i32, MathError> {
    left.checked_add(right)
        .ok_or(crate::arithmetic::overflow("addition"))
}

/// Subtract two signed coefficients, returning an error on overflow.
#[inline]
pub fn checked_sub(left: i32, right: i32) -> Result<i32, MathError> {
    left.checked_sub(right)
        .ok_or(crate::arithmetic::overflow("subtraction"))
}

/// Multiply two signed coefficients, returning an error on overflow.
#[inline]
pub fn checked_mul(left: i32, right: i32) -> Result<i32, MathError> {
    left.checked_mul(right)
        .ok_or(crate::arithmetic::overflow("multiplication"))
}

/// Convert an intermediate to a coefficient, returning an error on overflow.
#[inline]
pub fn checked_i64_to_i32(value: i64) -> Result<i32, MathError> {
    match i32::try_from(value) {
        Ok(value) => Ok(value),
        Err(_) => Err(crate::arithmetic::overflow("i64 to i32 conversion")),
    }
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
