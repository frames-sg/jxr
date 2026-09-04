//! Native parser and decoder errors.

use core::fmt;

/// Error returned while parsing or decoding JPEG XR data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativeError {
    /// The input does not begin with a supported JPEG XR signature.
    InvalidSignature,
    /// A parsed syntax combination violates a normative invariant.
    InvalidSyntax {
        /// Stable name of the violated invariant.
        field: &'static str,
    },
    /// The input ended before a complete syntax element was available.
    Truncated {
        /// Bit position at which the read was attempted.
        bit_position: usize,
        /// Number of bits requested.
        requested_bits: u8,
    },
    /// A syntax field used a reserved value that changes decoding semantics.
    ReservedValue {
        /// Normative field name.
        field: &'static str,
        /// Parsed value.
        value: u64,
    },
    /// Checked integer arithmetic overflowed.
    IntegerOverflow {
        /// Operation being evaluated.
        operation: &'static str,
    },
    /// A byte range points outside the supplied input.
    RangeOutsideInput {
        /// Field that supplied the range.
        field: &'static str,
        /// Requested byte offset.
        offset: usize,
        /// Requested byte length.
        length: usize,
        /// Available input length.
        input_length: usize,
    },
    /// A required Annex-A directory entry was absent.
    MissingAnnexAField {
        /// Missing field tag.
        tag: u16,
    },
    /// An Annex-A entry has an invalid element type or element count.
    InvalidAnnexAEntry {
        /// Entry tag.
        tag: u16,
        /// Element type.
        element_type: u16,
        /// Element count.
        count: u32,
    },
    /// Annex-A directory tags were not strictly increasing.
    UnsortedAnnexATags {
        /// Previous tag.
        previous: u16,
        /// Current tag.
        current: u16,
    },
    /// A byte-alignment padding bit was non-zero.
    NonZeroAlignmentBit {
        /// Position of the invalid padding bit.
        bit_position: usize,
    },
    /// The stream uses syntax not implemented by the current decoder stage.
    Unsupported {
        /// Stable description of the unsupported syntax.
        feature: &'static str,
    },
}

impl fmt::Display for NativeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSignature => formatter.write_str("invalid JPEG XR signature"),
            Self::InvalidSyntax { field } => {
                write!(formatter, "invalid JPEG XR syntax for {field}")
            }
            Self::Truncated {
                bit_position,
                requested_bits,
            } => write!(
                formatter,
                "truncated JPEG XR input at bit {bit_position} while reading {requested_bits} bits"
            ),
            Self::ReservedValue { field, value } => {
                write!(
                    formatter,
                    "reserved value {value} for JPEG XR field {field}"
                )
            }
            Self::IntegerOverflow { operation } => {
                write!(formatter, "integer overflow while {operation}")
            }
            Self::RangeOutsideInput {
                field,
                offset,
                length,
                input_length,
            } => write!(
                formatter,
                "{field} range {offset}..{} exceeds input length {input_length}",
                offset.saturating_add(*length)
            ),
            Self::MissingAnnexAField { tag } => {
                write!(formatter, "missing required Annex-A tag 0x{tag:04X}")
            }
            Self::InvalidAnnexAEntry {
                tag,
                element_type,
                count,
            } => write!(
                formatter,
                "invalid Annex-A tag 0x{tag:04X} type {element_type} count {count}"
            ),
            Self::UnsortedAnnexATags { previous, current } => write!(
                formatter,
                "Annex-A tags are not strictly increasing: 0x{previous:04X}, 0x{current:04X}"
            ),
            Self::NonZeroAlignmentBit { bit_position } => {
                write!(
                    formatter,
                    "non-zero alignment bit at position {bit_position}"
                )
            }
            Self::Unsupported { feature } => {
                write!(formatter, "unsupported JPEG XR feature: {feature}")
            }
        }
    }
}

impl std::error::Error for NativeError {}
