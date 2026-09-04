//! Frequency-mode YUVK and Main-profile N-component tile traversal.

use jxr_core::{BandPresence, PredictionMode, QuantizerSet};

use crate::{
    ParsedCodestream,
    entropy::{PacketBitReader, TileEntropyState, decode_flex_block},
    reconstruct::QuantizedMacroblock,
};

use super::{
    DecodedTile, TileDecodeError,
    cbphp::CbphpState,
    frequency::FrequencyPacketRanges,
    high_pass::{self, HIERARCHICAL_BLOCK_ORDER, HighpassPayload},
    multicomponent, packet_slice,
    quantizer::{QuantizerIndices, TileQuantizers},
    spatial::{
        MacroblockPosition, SpatialMacroblock, consume_byte_alignment, parse_packet_prefix, read_u8,
    },
};

pub(super) fn decode_tile(
    source: &[u8],
    parsed: &ParsedCodestream,
    bands: BandPresence,
    ranges: FrequencyPacketRanges,
    tile_width: u32,
    tile_height: u32,
) -> Result<DecodedTile, TileDecodeError> {
    let width = usize::try_from(tile_width)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("multi-component frequency width"))?;
    let height = usize::try_from(tile_height)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("multi-component frequency height"))?;
    let (mut decoded, mut quantizers) = decode_dc(
        packet_slice(source, ranges.dc)?,
        parsed,
        bands,
        width,
        height,
    )?;
    if bands == BandPresence::DcOnly {
        assign_quantizers(
            &mut decoded,
            &quantizers,
            &[],
            &[],
            parsed.headers.primary.scaled,
        )?;
        return Ok(DecodedTile {
            components: decoded,
        });
    }
    let low_range = ranges.low_pass.ok_or(TileDecodeError::InvalidPlan(
        "missing multi-component LP packet",
    ))?;
    let low_indices = decode_low_pass(
        packet_slice(source, low_range)?,
        parsed,
        &mut quantizers,
        &mut decoded,
        width,
        height,
    )?;
    if bands == BandPresence::NoHighPass {
        assign_quantizers(
            &mut decoded,
            &quantizers,
            &low_indices,
            &vec![0; low_indices.len()],
            parsed.headers.primary.scaled,
        )?;
        return Ok(DecodedTile {
            components: decoded,
        });
    }
    let high_range = ranges.high_pass.ok_or(TileDecodeError::InvalidPlan(
        "missing multi-component HP packet",
    ))?;
    let high_state = decode_high_pass(
        packet_slice(source, high_range)?,
        parsed,
        &mut quantizers,
        &mut decoded,
        &low_indices,
        width,
        height,
    )?;
    if let Some(flex_range) = ranges.flexbits {
        decode_flexbits(
            packet_slice(source, flex_range)?,
            parsed.headers.image.flags.trim_flexbits(),
            &mut decoded,
            &high_state,
        )?;
    } else {
        finish_without_flexbits(&mut decoded, &high_state)?;
    }
    assign_quantizers(
        &mut decoded,
        &quantizers,
        &low_indices,
        &high_state.qp_indices,
        parsed.headers.primary.scaled,
    )?;
    Ok(DecodedTile {
        components: decoded,
    })
}

fn decode_dc(
    packet: &[u8],
    parsed: &ParsedCodestream,
    bands: BandPresence,
    width: usize,
    height: usize,
) -> Result<(Vec<Vec<SpatialMacroblock>>, TileQuantizers), TileDecodeError> {
    let plane = &parsed.headers.primary;
    let components = usize::from(plane.components);
    let mut reader = PacketBitReader::new(packet);
    parse_packet_prefix(&mut reader)?;
    let quantizers = TileQuantizers::parse_dc_packet(&mut reader, plane)?;
    let count = width
        .checked_mul(height)
        .ok_or(TileDecodeError::ArithmeticOverflow(
            "multi-component frequency macroblock count",
        ))?;
    let mut decoded: Vec<Vec<SpatialMacroblock>> =
        (0..components).map(|_| Vec::with_capacity(count)).collect();
    let mut entropy = TileEntropyState::new();
    entropy.reset_tile();
    for y in 0..height {
        for x in 0..width {
            let position = MacroblockPosition { width, x, y };
            let (dc, mode) =
                multicomponent::decode_dc(&mut reader, &mut entropy, &decoded, position, plane)?;
            for (component, component_plane) in decoded.iter_mut().enumerate() {
                let mut low = [0_i32; 16];
                low[0] = dc[component];
                component_plane.push(SpatialMacroblock {
                    coefficients: QuantizedMacroblock {
                        dc_low_pass: low,
                        high_pass: [0_i32; 256],
                        quantizers: unit_quantizers(),
                        bands,
                    },
                    prediction: mode,
                    hp_prediction: PredictionMode::None,
                    lp_qp_index: 0,
                });
            }
            if x + 1 == width || x.is_multiple_of(16) {
                entropy.dc_vlc.adapt();
            }
        }
    }
    consume_byte_alignment(&mut reader)?;
    Ok((decoded, quantizers))
}

fn decode_low_pass(
    packet: &[u8],
    parsed: &ParsedCodestream,
    quantizers: &mut TileQuantizers,
    decoded: &mut [Vec<SpatialMacroblock>],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, TileDecodeError> {
    let mut reader = PacketBitReader::new(packet);
    parse_packet_prefix(&mut reader)?;
    quantizers.parse_low_pass_packet(&mut reader, &parsed.headers.primary)?;
    let mut indices = Vec::with_capacity(width * height);
    let mut entropy = TileEntropyState::new();
    entropy.reset_tile();
    for y in 0..height {
        for x in 0..width {
            if x.is_multiple_of(16) {
                entropy.reset_scan_totals();
            }
            let index = y * width + x;
            let position = MacroblockPosition { width, x, y };
            let qp = quantizers.low_pass_index(&mut reader)?;
            let mode = decoded[0][index].prediction;
            let mut low: Vec<[i32; 16]> = decoded
                .iter()
                .map(|component| component[index].coefficients.dc_low_pass)
                .collect();
            multicomponent::decode_low_pass(
                &mut reader,
                &mut entropy,
                decoded,
                position,
                qp,
                mode,
                &mut low,
            )?;
            for (component, coefficients) in decoded.iter_mut().zip(low) {
                component[index].coefficients.dc_low_pass = coefficients;
                component[index].lp_qp_index = qp;
            }
            indices.push(qp);
            if x + 1 == width || x.is_multiple_of(16) {
                entropy.lp_vlc.adapt();
            }
        }
    }
    consume_byte_alignment(&mut reader)?;
    Ok(indices)
}

struct HighPassState {
    qp_indices: Vec<u8>,
    model_bits: Vec<[u8; 2]>,
}

fn decode_high_pass(
    packet: &[u8],
    parsed: &ParsedCodestream,
    quantizers: &mut TileQuantizers,
    decoded: &mut [Vec<SpatialMacroblock>],
    low_indices: &[u8],
    width: usize,
    height: usize,
) -> Result<HighPassState, TileDecodeError> {
    let mut reader = PacketBitReader::new(packet);
    parse_packet_prefix(&mut reader)?;
    quantizers.parse_high_pass_packet(&mut reader, &parsed.headers.primary)?;
    let mut state = HighPassState {
        qp_indices: Vec::with_capacity(width * height),
        model_bits: Vec::with_capacity(width * height),
    };
    let mut entropy = TileEntropyState::new();
    entropy.reset_tile();
    let mut cbphp = CbphpState::new_components(width, decoded.len());
    for y in 0..height {
        for x in 0..width {
            if x.is_multiple_of(16) {
                entropy.reset_scan_totals();
            }
            let index = y * width + x;
            let position = MacroblockPosition { width, x, y };
            let qp = quantizers.high_pass_index(&mut reader, low_indices[index])?;
            let low: Vec<[i32; 16]> = decoded
                .iter()
                .map(|component| component[index].coefficients.dc_low_pass)
                .collect();
            let mode = if parsed.headers.primary.internal_color_format == 4 {
                high_pass::prediction_mode_yuvk(&low)?
            } else {
                high_pass::prediction_mode(&low[0])
            };
            let high = high_pass::decode_components(
                &mut reader,
                &mut entropy,
                &mut cbphp,
                position,
                decoded.len(),
                mode,
                HighpassPayload::VlcOnly,
            )?;
            for (component, coefficients) in decoded.iter_mut().zip(high.coefficients) {
                component[index].coefficients.high_pass = coefficients;
                component[index].hp_prediction = mode;
            }
            state.qp_indices.push(qp);
            state.model_bits.push(high.model_bits);
            if x + 1 == width || x.is_multiple_of(16) {
                entropy.hp_vlc.adapt();
                cbphp.adapt();
            }
        }
    }
    consume_byte_alignment(&mut reader)?;
    Ok(state)
}

fn decode_flexbits(
    packet: &[u8],
    trim_present: bool,
    decoded: &mut [Vec<SpatialMacroblock>],
    state: &HighPassState,
) -> Result<(), TileDecodeError> {
    let mut reader = PacketBitReader::new(packet);
    parse_packet_prefix(&mut reader)?;
    let trim = if trim_present {
        read_u8(&mut reader, 4)?
    } else {
        0
    };
    for index in 0..state.model_bits.len() {
        for (component, plane) in decoded.iter_mut().enumerate() {
            combine_flexbits(
                &mut reader,
                &mut plane[index].coefficients.high_pass,
                state.model_bits[index][usize::from(component != 0)],
                trim,
            )?;
        }
    }
    consume_byte_alignment(&mut reader)
}

fn combine_flexbits(
    reader: &mut PacketBitReader<'_>,
    coefficients: &mut [i32; 256],
    model_bits: u8,
    trim: u8,
) -> Result<(), TileDecodeError> {
    for block in HIERARCHICAL_BLOCK_ORDER {
        let destination = &mut coefficients[block * 16..(block + 1) * 16];
        let mut vlc = [0_i32; 16];
        vlc.copy_from_slice(destination);
        let flex = decode_flex_block(reader, &vlc, model_bits, trim)?;
        for coefficient in 1..16 {
            destination[coefficient] = vlc[coefficient]
                .checked_shl(u32::from(model_bits))
                .and_then(|value| value.checked_add(flex[coefficient]))
                .ok_or(TileDecodeError::ArithmeticOverflow(
                    "multi-component HP flexbits",
                ))?;
        }
    }
    Ok(())
}

fn finish_without_flexbits(
    decoded: &mut [Vec<SpatialMacroblock>],
    state: &HighPassState,
) -> Result<(), TileDecodeError> {
    for (component, plane) in decoded.iter_mut().enumerate() {
        for (index, macroblock) in plane.iter_mut().enumerate() {
            let bits = u32::from(state.model_bits[index][usize::from(component != 0)]);
            for block in 0..16 {
                for coefficient in 1..16 {
                    let position = block * 16 + coefficient;
                    macroblock.coefficients.high_pass[position] = macroblock.coefficients.high_pass
                        [position]
                        .checked_shl(bits)
                        .ok_or(TileDecodeError::ArithmeticOverflow(
                            "multi-component HP normalization",
                        ))?;
                }
            }
        }
    }
    Ok(())
}

fn assign_quantizers(
    decoded: &mut [Vec<SpatialMacroblock>],
    quantizers: &TileQuantizers,
    low_indices: &[u8],
    high_indices: &[u8],
    scaled: bool,
) -> Result<(), TileDecodeError> {
    let count = decoded.first().map_or(0, Vec::len);
    let dc_only = low_indices.is_empty() && high_indices.is_empty();
    if !dc_only && (low_indices.len() != count || high_indices.len() != count) {
        return Err(TileDecodeError::InvalidPlan(
            "multi-component quantizer index count",
        ));
    }
    for (component, plane) in decoded.iter_mut().enumerate() {
        for (index, macroblock) in plane.iter_mut().enumerate() {
            let indices = if dc_only {
                QuantizerIndices { lp: 0, hp: 0 }
            } else {
                QuantizerIndices {
                    lp: low_indices[index],
                    hp: high_indices[index],
                }
            };
            macroblock.coefficients.quantizers =
                quantizers.reconstruction_steps_for(component, indices, scaled)?;
        }
    }
    Ok(())
}

const fn unit_quantizers() -> QuantizerSet {
    QuantizerSet {
        dc: 1,
        low_pass: 1,
        high_pass: 1,
    }
}
