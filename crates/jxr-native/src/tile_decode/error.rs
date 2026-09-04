//! Tile-packet decoding failures.

use core::fmt;

use crate::entropy::EntropyError;

/// Failure while turning coded tile packets into reconstruction coefficients.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TileDecodeError {
    /// A planned tile byte range is outside the supplied source.
    PacketRangeOutsideInput {
        /// Start of the requested range.
        offset: usize,
        /// Length of the requested range.
        length: usize,
        /// Available source length.
        input_length: usize,
    },
    /// A tile packet did not begin with the normative 24-bit start code.
    InvalidStartCode {
        /// Value found in the packet.
        value: u32,
    },
    /// The prepared plan and parsed tile geometry disagree.
    InvalidPlan(&'static str),
    /// A reserved quantizer-table index was present.
    InvalidQuantizerIndex {
        /// Number of entries in the table.
        table_length: u8,
        /// Decoded index.
        index: u8,
    },
    /// Checked coefficient or allocation arithmetic overflowed.
    ArithmeticOverflow(&'static str),
    /// The current vertical slice deliberately does not decode this syntax.
    Unsupported(&'static str),
    /// A bounded entropy primitive rejected tile data.
    Entropy(EntropyError),
}

impl From<EntropyError> for TileDecodeError {
    fn from(error: EntropyError) -> Self {
        Self::Entropy(error)
    }
}

impl TileDecodeError {
    pub(crate) const fn operation(&self) -> &'static str {
        match self {
            Self::PacketRangeOutsideInput { .. } => "slice tile packet range",
            Self::InvalidStartCode { .. } => "validate tile packet start code",
            Self::InvalidPlan(field)
            | Self::ArithmeticOverflow(field)
            | Self::Unsupported(field) => field,
            Self::InvalidQuantizerIndex { .. } => "select tile quantizer",
            Self::Entropy(_) => "decode tile packet entropy",
        }
    }
}

impl fmt::Display for TileDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PacketRangeOutsideInput {
                offset,
                length,
                input_length,
            } => write!(
                formatter,
                "tile packet range {offset}..{} exceeds input length {input_length}",
                offset.saturating_add(*length)
            ),
            Self::InvalidStartCode { value } => {
                write!(formatter, "invalid tile start code 0x{value:06X}")
            }
            Self::InvalidPlan(field) => write!(formatter, "invalid prepared tile plan: {field}"),
            Self::InvalidQuantizerIndex {
                table_length,
                index,
            } => write!(
                formatter,
                "quantizer index {index} is outside table of length {table_length}"
            ),
            Self::ArithmeticOverflow(operation) => write!(formatter, "overflow during {operation}"),
            Self::Unsupported(feature) => write!(formatter, "unsupported tile syntax: {feature}"),
            Self::Entropy(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for TileDecodeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operation_preserves_specific_plan_and_feature_context() {
        assert_eq!(
            TileDecodeError::Unsupported("subsampled frequency HP").operation(),
            "subsampled frequency HP"
        );
        assert_eq!(
            TileDecodeError::InvalidPlan("frequency quantizer index count").operation(),
            "frequency quantizer index count"
        );
    }
}
