//! Frequency-mode Y-only packet orchestration.

mod dc;
mod high_pass;
mod low_pass;
mod ranges;

use jxr_core::BandPresence;

use crate::ParsedCodestream;

use super::{TileDecodeError, packet_slice, spatial::SpatialMacroblock};

pub(super) use ranges::{FrequencyPacketRanges, packet_ranges};

pub(super) fn decode_tile(
    source: &[u8],
    parsed: &ParsedCodestream,
    bands: BandPresence,
    packet_ranges: FrequencyPacketRanges,
    tile_width: u32,
    tile_height: u32,
) -> Result<Vec<SpatialMacroblock>, TileDecodeError> {
    let width = usize::try_from(tile_width)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("frequency tile width"))?;
    let height = usize::try_from(tile_height)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("frequency tile height"))?;
    let dc_packet = packet_slice(source, packet_ranges.dc)?;
    let (mut decoded, mut quantizers) =
        dc::decode(dc_packet, &parsed.headers.primary, bands, width, height)?;
    if bands == BandPresence::DcOnly {
        dc::finalize_dc_only(&mut decoded, &quantizers, parsed.headers.primary.scaled)?;
        return Ok(decoded);
    }

    let low_range = packet_ranges
        .low_pass
        .ok_or(TileDecodeError::InvalidPlan("missing LP frequency packet"))?;
    let low_indices = low_pass::decode(
        packet_slice(source, low_range)?,
        &parsed.headers.primary,
        &mut quantizers,
        &mut decoded,
        width,
        height,
    )?;
    if bands == BandPresence::NoHighPass {
        let high_indices = vec![0; low_indices.len()];
        assign_quantizers(
            &mut decoded,
            &quantizers,
            &low_indices,
            &high_indices,
            parsed.headers.primary.scaled,
        )?;
        return Ok(decoded);
    }

    let high_range = packet_ranges
        .high_pass
        .ok_or(TileDecodeError::InvalidPlan("missing HP frequency packet"))?;
    let high_state = high_pass::decode(
        packet_slice(source, high_range)?,
        &parsed.headers.primary,
        &mut quantizers,
        &mut decoded,
        &low_indices,
        width,
        height,
    )?;
    if let Some(flex_range) = packet_ranges.flexbits {
        high_pass::decode_flexbits(
            packet_slice(source, flex_range)?,
            parsed.headers.image.flags.trim_flexbits(),
            &mut decoded,
            &high_state,
        )?;
    } else {
        high_pass::finish_without_flexbits(&mut decoded, &high_state)?;
    }
    assign_quantizers(
        &mut decoded,
        &quantizers,
        &low_indices,
        &high_state.qp_indices,
        parsed.headers.primary.scaled,
    )?;
    Ok(decoded)
}

fn assign_quantizers(
    decoded: &mut [SpatialMacroblock],
    quantizers: &super::quantizer::TileQuantizers,
    low_pass_indices: &[u8],
    high_pass_indices: &[u8],
    scaled: bool,
) -> Result<(), TileDecodeError> {
    if decoded.len() != low_pass_indices.len() || decoded.len() != high_pass_indices.len() {
        return Err(TileDecodeError::InvalidPlan(
            "frequency quantizer index count",
        ));
    }
    for (index, macroblock) in decoded.iter_mut().enumerate() {
        macroblock.coefficients.quantizers = quantizers.reconstruction_steps(
            super::quantizer::QuantizerIndices {
                lp: low_pass_indices[index],
                hp: high_pass_indices[index],
            },
            scaled,
        )?;
    }
    Ok(())
}
