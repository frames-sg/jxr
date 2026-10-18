//! Tile packet routing and parallel tile dispatch.

use jxr_core::{BitstreamMode, PreparedPlan};
use rayon::prelude::*;

use crate::{ImagePlaneHeader, ParsedCodestream};

use super::{
    DecodedTile, TileDecodeError, TileLocation, frequency, integrated_alpha, multicomponent,
    multicomponent_frequency,
    output::{MacroblockRecord, PlaneWriter, TileWriters},
    packet_slice, spatial, yuv,
};

/// Macroblock records of one decoded tile, per component.
pub(super) type DecodedRecords = (Vec<Vec<MacroblockRecord>>, TileLocation);

pub(super) fn decode_packets(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    locations: &[TileLocation],
    mut tiles: Vec<TileWriters<'_>>,
) -> Result<Vec<DecodedRecords>, TileDecodeError> {
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
    let decode = |(index, mut writers): TileWriters<'_>| {
        let location = locations[index];
        decode_packet(
            source,
            parsed,
            plan,
            frequency_ranges.as_deref(),
            location,
            index,
            &mut writers,
        )?;
        let records = writers
            .into_iter()
            .map(PlaneWriter::finish)
            .collect::<Result<_, _>>()?;
        Ok((records, location))
    };
    match tiles.len() {
        0 => Err(TileDecodeError::InvalidPlan("empty tile grid")),
        1 => Ok(vec![decode(tiles.remove(0))?]),
        _ => tiles.into_par_iter().map(decode).collect(),
    }
}

fn decode_packet(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    frequency_ranges: Option<&[frequency::FrequencyPacketRanges]>,
    location: TileLocation,
    index: usize,
    writers: &mut [PlaneWriter<'_>],
) -> Result<(), TileDecodeError> {
    let tile = &plan.tiles[index];
    let expected_macroblocks = location
        .width
        .checked_mul(location.height)
        .ok_or(TileDecodeError::ArithmeticOverflow("tile macroblock count"))?;
    if tile.macroblock_count != expected_macroblocks {
        return Err(TileDecodeError::InvalidPlan("tile macroblock count"));
    }
    if let Some(alpha) = &parsed.headers.alpha {
        decode_integrated_packet(
            source,
            parsed,
            plan,
            tile,
            location,
            alpha,
            frequency_ranges.map(|ranges| ranges[index]),
            writers,
        )
    } else {
        decode_primary_packet(
            source,
            parsed,
            plan,
            frequency_ranges,
            location,
            index,
            writers,
        )
    }
}

fn decode_primary_packet(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    frequency_ranges: Option<&[frequency::FrequencyPacketRanges]>,
    location: TileLocation,
    index: usize,
    writers: &mut [PlaneWriter<'_>],
) -> Result<(), TileDecodeError> {
    let tile = &plan.tiles[index];
    let bands = plan.scale.retained_bands(plan.info.primary.bands);
    let decoded = match (
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
        )?,
        (1..=3, None) => {
            return yuv::decode_spatial_tile(
                packet_slice(source, tile.packet_range)?,
                parsed,
                bands,
                location.width,
                location.height,
                writers,
            );
        }
        (4 | 6, None) => {
            return multicomponent::decode_spatial_tile(
                packet_slice(source, tile.packet_range)?,
                parsed,
                bands,
                location.width,
                location.height,
                writers,
            );
        }
        (4 | 6, Some(ranges)) => multicomponent_frequency::decode_tile(
            source,
            parsed,
            bands,
            ranges[index],
            location.width,
            location.height,
        )?,
        (_, Some(ranges)) => DecodedTile::single(frequency::decode_tile(
            source,
            parsed,
            bands,
            ranges[index],
            location.width,
            location.height,
        )?),
        (_, None) => {
            return spatial::decode_spatial_packet(
                packet_slice(source, tile.packet_range)?,
                &parsed.headers.primary,
                bands,
                location.width,
                location.height,
                parsed.headers.image.flags.trim_flexbits(),
                single_writer(writers)?,
            );
        }
    };
    store_decoded(&decoded, writers)
}

/// Store a completed frequency-mode tile into its coefficient windows.
fn store_decoded(
    decoded: &DecodedTile,
    writers: &mut [PlaneWriter<'_>],
) -> Result<(), TileDecodeError> {
    if decoded.components.len() != writers.len() {
        return Err(TileDecodeError::InvalidPlan("decoded component count"));
    }
    for (macroblocks, writer) in decoded.components.iter().zip(writers) {
        for macroblock in macroblocks {
            writer.push_macroblock(macroblock)?;
        }
    }
    Ok(())
}

fn single_writer<'w, 'a>(
    writers: &'w mut [PlaneWriter<'a>],
) -> Result<&'w mut PlaneWriter<'a>, TileDecodeError> {
    match writers {
        [writer] => Ok(writer),
        _ => Err(TileDecodeError::InvalidPlan("decoded component count")),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "mirrors decode_primary_packet with the alpha plane header"
)]
fn decode_integrated_packet(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    tile: &jxr_core::TilePlan,
    location: TileLocation,
    alpha: &ImagePlaneHeader,
    frequency_ranges: Option<frequency::FrequencyPacketRanges>,
    writers: &mut [PlaneWriter<'_>],
) -> Result<(), TileDecodeError> {
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
        let decoded = super::integrated_alpha_frequency::decode(
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
        )?;
        return store_decoded(&decoded, writers);
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
        writers,
    )
}
