//! CPU entropy handoff for scalar and accelerator reconstruction routes.

use jxr_core::{
    CoefficientArena, CoefficientArenaDescriptor, JxrError, JxrErrorKind, PreparedPlan,
};

use crate::{ParsedCodestream, tile_decode::decode_tiles};

/// Decode entropy, scans, remapping, and prediction into the shared coefficient arena.
pub fn decode_coefficients(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
) -> Result<CoefficientArena, JxrError> {
    decode_tiles(source, parsed, plan).map_err(|error| map_tile_error(&error))
}

pub(crate) fn decode_coefficients_reusing(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    arena: &mut CoefficientArena,
) -> Result<bool, JxrError> {
    crate::tile_decode::decode_tiles_reusing(source, parsed, plan, arena)
        .map_err(|error| map_tile_error(&error))
}

/// Exact coefficient element count required by caller-owned entropy storage.
pub fn coefficient_count(
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
) -> Result<usize, JxrError> {
    crate::tile_decode::coefficient_count(parsed, plan).map_err(|error| map_tile_error(&error))
}

/// Decode entropy directly into exact-size caller-owned coefficient storage.
pub fn decode_coefficients_into(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    destination: &mut [i32],
) -> Result<CoefficientArenaDescriptor, JxrError> {
    crate::tile_decode::decode_tiles_into(source, parsed, plan, destination)
        .map_err(|error| map_tile_error(&error))
}

fn map_tile_error(error: &crate::tile_decode::TileDecodeError) -> JxrError {
    use crate::tile_decode::TileDecodeError;

    let operation = error.operation();
    let kind = match error {
        TileDecodeError::PacketRangeOutsideInput { .. } => JxrErrorKind::Truncated,
        TileDecodeError::ArithmeticOverflow(_) => JxrErrorKind::ArithmeticOverflow,
        TileDecodeError::Unsupported(_) => JxrErrorKind::Unsupported,
        TileDecodeError::InvalidPlan(_) => JxrErrorKind::InternalInvariant,
        TileDecodeError::InvalidStartCode { .. }
        | TileDecodeError::InvalidQuantizerIndex { .. }
        | TileDecodeError::Entropy(_) => JxrErrorKind::InvalidSyntax,
    };
    JxrError::new(kind, operation)
}
