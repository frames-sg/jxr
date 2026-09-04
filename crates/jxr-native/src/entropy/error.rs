//! Errors produced while decoding a tile packet.

use core::fmt;

/// Error returned by an entropy decoding primitive.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum EntropyError {
    /// The packet ended before the requested bits were available.
    UnexpectedEnd {
        /// Bit offset of the failed read.
        bit_position: usize,
        /// Number of bits requested.
        requested_bits: u8,
        /// Number of bits in the bounded packet view.
        bit_length: usize,
    },
    /// A packet bit length exceeds its backing byte slice.
    InvalidBitLength {
        /// Requested packet length in bits.
        bit_length: usize,
        /// Available backing length in bits.
        available_bits: usize,
    },
    /// A bit prefix does not occur in the selected normative VLC table.
    InvalidVlc {
        /// Name of the syntax element being decoded.
        syntax: &'static str,
        /// Bit offset at which the prefix began.
        bit_position: usize,
    },
    /// A caller supplied a value outside a syntax primitive's domain.
    InvalidParameter {
        /// Parameter name.
        parameter: &'static str,
        /// Supplied value.
        value: i64,
    },
    /// Coefficient arithmetic exceeded the supported signed representation.
    CoefficientOverflow,
}

impl fmt::Display for EntropyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEnd {
                bit_position,
                requested_bits,
                bit_length,
            } => write!(
                formatter,
                "tile packet ended at bit {bit_position} while reading {requested_bits} bits (packet has {bit_length} bits)"
            ),
            Self::InvalidBitLength {
                bit_length,
                available_bits,
            } => write!(
                formatter,
                "tile packet bit length {bit_length} exceeds its {available_bits}-bit backing slice"
            ),
            Self::InvalidVlc {
                syntax,
                bit_position,
            } => write!(
                formatter,
                "invalid {syntax} VLC prefix at tile packet bit {bit_position}"
            ),
            Self::InvalidParameter { parameter, value } => {
                write!(formatter, "invalid {parameter} value {value}")
            }
            Self::CoefficientOverflow => formatter.write_str("decoded coefficient overflow"),
        }
    }
}

impl std::error::Error for EntropyError {}
