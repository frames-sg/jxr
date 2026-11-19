//! Tile-packet encoding failures.

use core::fmt;

use crate::{entropy::EntropyError, tile_decode::TileDecodeError};

/// Failure while writing one tile packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TileEncodeError {
    /// The macroblocks handed to the encoder do not describe a valid tile.
    InvalidInput(&'static str),
    /// A prediction or size computation left the supported integer range.
    ArithmeticOverflow(&'static str),
    /// The requested syntax has no implemented writer.
    Unsupported(&'static str),
    /// A coefficient or symbol could not be represented by the entropy syntax.
    Entropy(EntropyError),
}

impl From<EntropyError> for TileEncodeError {
    fn from(error: EntropyError) -> Self {
        Self::Entropy(error)
    }
}

/// Shared prediction and pattern helpers report through the decoder's error.
impl From<TileDecodeError> for TileEncodeError {
    fn from(error: TileDecodeError) -> Self {
        match error {
            TileDecodeError::ArithmeticOverflow(operation) => Self::ArithmeticOverflow(operation),
            TileDecodeError::Unsupported(feature) => Self::Unsupported(feature),
            TileDecodeError::Entropy(error) => Self::Entropy(error),
            TileDecodeError::InvalidPlan(field) => Self::InvalidInput(field),
            _ => Self::InvalidInput("tile syntax"),
        }
    }
}

impl fmt::Display for TileEncodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(field) => write!(formatter, "invalid tile encoder input: {field}"),
            Self::ArithmeticOverflow(operation) => write!(formatter, "overflow during {operation}"),
            Self::Unsupported(feature) => write!(formatter, "unsupported tile encoding: {feature}"),
            Self::Entropy(error) => write!(formatter, "tile entropy encoding failed: {error}"),
        }
    }
}

impl std::error::Error for TileEncodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Entropy(error) => Some(error),
            _ => None,
        }
    }
}
