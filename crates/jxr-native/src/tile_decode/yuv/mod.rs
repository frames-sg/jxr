//! Entropy syntax shared by full-resolution and subsampled YUV planes.

use jxr_core::{BandPresence, ChromaSampling};

use crate::ParsedCodestream;

use super::{DecodedTile, TileDecodeError, frequency::FrequencyPacketRanges};

pub(in crate::tile_decode) mod frequency;
pub(in crate::tile_decode) mod spatial;
pub(in crate::tile_decode) mod syntax;

pub(in crate::tile_decode) fn decode_spatial_tile(
    packet: &[u8],
    parsed: &ParsedCodestream,
    bands: BandPresence,
    tile_width: u32,
    tile_height: u32,
) -> Result<DecodedTile, TileDecodeError> {
    spatial::decode(
        packet,
        &parsed.headers.primary,
        bands,
        tile_width,
        tile_height,
        parsed.headers.image.flags.trim_flexbits(),
        sampling(parsed.headers.primary.internal_color_format)?,
    )
}

pub(in crate::tile_decode) fn decode_frequency_tile(
    source: &[u8],
    parsed: &ParsedCodestream,
    bands: BandPresence,
    ranges: FrequencyPacketRanges,
    tile_width: u32,
    tile_height: u32,
) -> Result<DecodedTile, TileDecodeError> {
    frequency::decode(
        source,
        parsed,
        bands,
        ranges,
        tile_width,
        tile_height,
        sampling(parsed.headers.primary.internal_color_format)?,
    )
}

pub(super) const fn sampling(code: u8) -> Result<ChromaSampling, TileDecodeError> {
    match code {
        1 => Ok(ChromaSampling::Cs420),
        2 => Ok(ChromaSampling::Cs422),
        3 => Ok(ChromaSampling::Cs444),
        _ => Err(TileDecodeError::InvalidPlan("YUV sampling code")),
    }
}
