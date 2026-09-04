//! Tile packet routing and parallel tile dispatch.

use jxr_core::{BitstreamMode, PreparedPlan};
use rayon::prelude::*;

use crate::{ImagePlaneHeader, ParsedCodestream};

use super::{
    DecodedTile, TileDecodeError, TileLocation, frequency, integrated_alpha, multicomponent,
    multicomponent_frequency, packet_slice, spatial, yuv,
};

pub(super) fn decode_packets(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    locations: &[TileLocation],
) -> Result<Vec<(DecodedTile, TileLocation)>, TileDecodeError> {
    let frequency_ranges = if plan.info.primary.bitstream_mode == BitstreamMode::Frequency {
        Some(frequency::packet_ranges(
            source.len(),
            parsed,
            plan.info.primary.bands,
            locations.len(),
        )?)
    } else {
        None
    };
    let selected: Vec<_> = plan
        .tiles
        .iter()
        .enumerate()
        .filter_map(|(index, tile)| tile.required_for_reconstruction.then_some(index))
        .collect();
    let decode = |index: usize| {
        decode_packet(
            source,
            parsed,
            plan,
            frequency_ranges.as_deref(),
            locations[index],
            index,
        )
    };
    match selected.len() {
        0 => Err(TileDecodeError::InvalidPlan("empty tile grid")),
        1 => Ok(vec![decode(selected[0])?]),
        _ => selected.into_par_iter().map(decode).collect(),
    }
}

fn decode_packet(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    frequency_ranges: Option<&[frequency::FrequencyPacketRanges]>,
    location: TileLocation,
    index: usize,
) -> Result<(DecodedTile, TileLocation), TileDecodeError> {
    let tile = &plan.tiles[index];
    let expected_macroblocks = location
        .width
        .checked_mul(location.height)
        .ok_or(TileDecodeError::ArithmeticOverflow("tile macroblock count"))?;
    if tile.macroblock_count != expected_macroblocks {
        return Err(TileDecodeError::InvalidPlan("tile macroblock count"));
    }
    let decoded = if let Some(alpha) = &parsed.headers.alpha {
        decode_integrated_packet(
            source,
            parsed,
            plan,
            tile,
            location,
            alpha,
            frequency_ranges.map(|ranges| ranges[index]),
        )?
    } else {
        decode_primary_packet(source, parsed, plan, frequency_ranges, location, index)?
    };
    Ok((decoded, location))
}

fn decode_primary_packet(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    frequency_ranges: Option<&[frequency::FrequencyPacketRanges]>,
    location: TileLocation,
    index: usize,
) -> Result<DecodedTile, TileDecodeError> {
    let tile = &plan.tiles[index];
    let bands = plan.scale.retained_bands(plan.info.primary.bands);
    match (
        parsed.headers.primary.internal_color_format,
        frequency_ranges,
    ) {
        (1..=3, Some(ranges)) => yuv::decode_frequency_tile(
            source,
            parsed,
            bands,
            ranges[index],
            location.width,
            location.height,
        ),
        (1..=3, None) => yuv::decode_spatial_tile(
            packet_slice(source, tile.packet_range)?,
            parsed,
            bands,
            location.width,
            location.height,
        ),
        (4 | 6, None) => multicomponent::decode_spatial_tile(
            packet_slice(source, tile.packet_range)?,
            parsed,
            bands,
            location.width,
            location.height,
        ),
        (4 | 6, Some(ranges)) => multicomponent_frequency::decode_tile(
            source,
            parsed,
            bands,
            ranges[index],
            location.width,
            location.height,
        ),
        (_, Some(ranges)) => frequency::decode_tile(
            source,
            parsed,
            bands,
            ranges[index],
            location.width,
            location.height,
        )
        .map(DecodedTile::single),
        (_, None) => spatial::decode_spatial_packet(
            packet_slice(source, tile.packet_range)?,
            &parsed.headers.primary,
            bands,
            location.width,
            location.height,
            parsed.headers.image.flags.trim_flexbits(),
        )
        .map(DecodedTile::single),
    }
}

fn decode_integrated_packet(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    tile: &jxr_core::TilePlan,
    location: TileLocation,
    alpha: &ImagePlaneHeader,
    frequency_ranges: Option<frequency::FrequencyPacketRanges>,
) -> Result<DecodedTile, TileDecodeError> {
    let alpha_bands = plan
        .info
        .alpha
        .as_ref()
        .ok_or(TileDecodeError::InvalidPlan("integrated alpha plane"))?
        .bands;
    let primary_bands = plan.scale.retained_bands(plan.info.primary.bands);
    let alpha_bands = plan.scale.retained_bands(alpha_bands);
    if let Some(ranges) = frequency_ranges {
        let width = usize::try_from(location.width)
            .map_err(|_| TileDecodeError::ArithmeticOverflow("frequency tile width"))?;
        let height = usize::try_from(location.height)
            .map_err(|_| TileDecodeError::ArithmeticOverflow("frequency tile height"))?;
        return super::integrated_alpha_frequency::decode(
            source,
            super::integrated_alpha_frequency::PlaneDescriptor {
                header: &parsed.headers.primary,
                bands: primary_bands,
            },
            super::integrated_alpha_frequency::PlaneDescriptor {
                header: alpha,
                bands: alpha_bands,
            },
            ranges,
            width,
            height,
            parsed.headers.image.flags.trim_flexbits(),
        );
    }
    integrated_alpha::decode_spatial(
        packet_slice(source, tile.packet_range)?,
        integrated_alpha::IntegratedPlanes {
            primary: &parsed.headers.primary,
            alpha,
            primary_bands,
            alpha_bands,
            trim_present: parsed.headers.image.flags.trim_flexbits(),
        },
        location.width,
        location.height,
    )
}
