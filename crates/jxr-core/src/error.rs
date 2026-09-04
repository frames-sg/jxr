//! Structured decoder errors.

use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LimitKind {
    Width,
    Height,
    Pixels,
    Components,
    Tiles,
    CompressedBytes,
    CoefficientBytes,
    HostAllocationBytes,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JxrErrorKind {
    Truncated,
    InvalidSyntax,
    Unsupported,
    LimitExceeded {
        limit: LimitKind,
        requested: u64,
        maximum: u64,
    },
    ArithmeticOverflow,
    BufferTooSmall {
        required: usize,
        available: usize,
    },
    BackendUnavailable,
    DeviceFailure,
    InvalidRequest,
    InternalInvariant,
}

/// A decode failure with stable classification and operation context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JxrError {
    pub kind: JxrErrorKind,
    pub operation: &'static str,
    pub byte_offset: Option<u64>,
}

impl JxrError {
    #[must_use]
    pub const fn new(kind: JxrErrorKind, operation: &'static str) -> Self {
        Self {
            kind,
            operation,
            byte_offset: None,
        }
    }

    #[must_use]
    pub const fn at(mut self, byte_offset: u64) -> Self {
        self.byte_offset = Some(byte_offset);
        self
    }

    #[must_use]
    pub const fn arithmetic(operation: &'static str) -> Self {
        Self::new(JxrErrorKind::ArithmeticOverflow, operation)
    }

    #[must_use]
    pub const fn limit(
        operation: &'static str,
        limit: LimitKind,
        requested: u64,
        maximum: u64,
    ) -> Self {
        Self::new(
            JxrErrorKind::LimitExceeded {
                limit,
                requested,
                maximum,
            },
            operation,
        )
    }
}

impl fmt::Display for JxrError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "JPEG XR {:?} while {}",
            self.kind, self.operation
        )?;
        if let Some(offset) = self.byte_offset {
            write!(formatter, " at byte {offset}")?;
        }
        Ok(())
    }
}

impl core::error::Error for JxrError {}
