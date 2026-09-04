//! Mapping from parser-specific failures into the stable facade error taxonomy.

use jxr_core::{JxrError, JxrErrorKind};

pub(crate) fn map_native_error(error: jxr_native::NativeError) -> JxrError {
    use jxr_native::NativeError;

    let (kind, operation) = match error {
        NativeError::Truncated { .. } | NativeError::RangeOutsideInput { .. } => {
            (JxrErrorKind::Truncated, "parse compressed input")
        }
        NativeError::Unsupported { .. } => (JxrErrorKind::Unsupported, "parse compressed input"),
        NativeError::IntegerOverflow { .. } => {
            (JxrErrorKind::ArithmeticOverflow, "parse compressed input")
        }
        NativeError::InvalidSignature
        | NativeError::InvalidSyntax { .. }
        | NativeError::ReservedValue { .. }
        | NativeError::MissingAnnexAField { .. }
        | NativeError::InvalidAnnexAEntry { .. }
        | NativeError::UnsortedAnnexATags { .. }
        | NativeError::NonZeroAlignmentBit { .. } => {
            (JxrErrorKind::InvalidSyntax, "parse compressed input")
        }
        _ => (JxrErrorKind::InvalidSyntax, "parse compressed input"),
    };
    JxrError::new(kind, operation)
}
