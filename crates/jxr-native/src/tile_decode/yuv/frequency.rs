//! Frequency-mode YUV tile traversal for all sampling geometries.

use jxr_core::{BandPresence, ChromaSampling, PredictionMode, QuantizerSet};

use crate::{
    ParsedCodestream,
    entropy::{PacketBitReader, TileEntropyState, decode_flex_block},
    reconstruct::QuantizedMacroblock,
};

use super::super::{
    DecodedTile, TileDecodeError,
    cbphp::CbphpState,
    frequency::FrequencyPacketRanges,
    high_pass::{self, HIERARCHICAL_BLOCK_ORDER, HighpassPayload},
    packet_slice,
    quantizer::{QuantizerIndices, TileQuantizers},
    spatial::{
        MacroblockPosition, SpatialMacroblock, consume_byte_alignment, parse_packet_prefix, read_u8,
    },
};
use super::syntax::{self, CbplpState};

pub(in crate::tile_decode) fn decode(
    source: &[u8],
    parsed: &ParsedCodestream,
    bands: BandPresence,
    ranges: FrequencyPacketRanges,
    tile_width: u32,
    tile_height: u32,
    sampling: ChromaSampling,
) -> Result<DecodedTile, TileDecodeError> {
    let width = usize::try_from(tile_width)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("YUV444 tile width"))?;
    let height = usize::try_from(tile_height)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("YUV444 tile height"))?;
    let (mut components, mut quantizers) = decode_dc_packet(
        packet_slice(source, ranges.dc)?,
        &parsed.headers.primary,
        bands,
        width,
        height,
        sampling,
    )?;
    if bands == BandPresence::DcOnly {
        assign_quantizers(
            &mut components,
            &quantizers,
            &[],
            &[],
            parsed.headers.primary.scaled,
        )?;
        return Ok(decoded_tile(components));
    }
    let lp_range = ranges
        .low_pass
        .ok_or(TileDecodeError::InvalidPlan("missing YUV444 LP packet"))?;
    let lp_indices = decode_lp_packet(
        packet_slice(source, lp_range)?,
        &parsed.headers.primary,
        &mut quantizers,
        &mut components,
        width,
        height,
        sampling,
    )?;
    if bands == BandPresence::NoHighPass {
        assign_quantizers(
            &mut components,
            &quantizers,
            &lp_indices,
            &[],
            parsed.headers.primary.scaled,
        )?;
        return Ok(decoded_tile(components));
    }
    let hp_range = ranges
        .high_pass
        .ok_or(TileDecodeError::InvalidPlan("missing YUV444 HP packet"))?;
    let hp_state = decode_hp_packet(
        packet_slice(source, hp_range)?,
        &mut quantizers,
        HpPacketContext {
            plane: &parsed.headers.primary,
            components: &mut components,
            lp_indices: &lp_indices,
            width,
            height,
            sampling,
        },
    )?;
    if let Some(flex_range) = ranges.flexbits {
        decode_flex_packet(
            packet_slice(source, flex_range)?,
            parsed.headers.image.flags.trim_flexbits(),
            &mut components,
            &hp_state,
            sampling,
        )?;
    } else {
        finish_hp_without_flex(&mut components, &hp_state, sampling)?;
    }
    assign_quantizers(
        &mut components,
        &quantizers,
        &lp_indices,
        &hp_state.qp_indices,
        parsed.headers.primary.scaled,
    )?;
    Ok(decoded_tile(components))
}

fn decode_dc_packet(
    packet: &[u8],
    plane: &crate::ImagePlaneHeader,
    bands: BandPresence,
    width: usize,
    height: usize,
    sampling: ChromaSampling,
) -> Result<([Vec<SpatialMacroblock>; 3], TileQuantizers), TileDecodeError> {
    let mut reader = PacketBitReader::new(packet);
    parse_packet_prefix(&mut reader)?;
    let quantizers = TileQuantizers::parse_dc_packet(&mut reader, plane)?;
    let count = width * height;
    let mut components: [Vec<SpatialMacroblock>; 3] =
        core::array::from_fn(|_| Vec::with_capacity(count));
    let mut entropy = TileEntropyState::new();
    for y in 0..height {
        for x in 0..width {
            let position = MacroblockPosition { width, x, y };
            let dc = syntax::decode_dc(&mut reader, &mut entropy, &components, position, sampling)?;
            for component in 0..3 {
                let mut low = [0_i32; 16];
                low[0] = dc[component].0;
                components[component].push(SpatialMacroblock {
                    coefficients: QuantizedMacroblock {
                        dc_low_pass: low,
                        high_pass: [0; 256],
                        quantizers: QuantizerSet {
                            dc: 1,
                            low_pass: 1,
                            high_pass: 1,
                        },
                        bands,
                    },
                    prediction: dc[component].1,
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
    Ok((components, quantizers))
}

fn decode_lp_packet(
    packet: &[u8],
    plane: &crate::ImagePlaneHeader,
    quantizers: &mut TileQuantizers,
    components: &mut [Vec<SpatialMacroblock>; 3],
    width: usize,
    height: usize,
    sampling: ChromaSampling,
) -> Result<Vec<u8>, TileDecodeError> {
    let mut reader = PacketBitReader::new(packet);
    parse_packet_prefix(&mut reader)?;
    quantizers.parse_low_pass_packet(&mut reader, plane)?;
    let mut indices = Vec::with_capacity(width * height);
    let mut entropy = TileEntropyState::new();
    let mut cbplp = CbplpState::new(sampling);
    for y in 0..height {
        for x in 0..width {
            if x.is_multiple_of(16) {
                entropy.reset_scan_totals();
            }
            let position = MacroblockPosition { width, x, y };
            let index = y * width + x;
            let qp = quantizers.low_pass_index(&mut reader)?;
            let mut low = core::array::from_fn(|component| {
                components[component][index].coefficients.dc_low_pass
            });
            let predictions =
                core::array::from_fn(|component| components[component][index].prediction);
            let context = syntax::LowPassContext {
                decoded: components,
                position,
                qp_index: qp,
                predictions,
            };
            if sampling == ChromaSampling::Cs444 {
                syntax::decode_low_pass_444(
                    &mut reader,
                    &mut entropy,
                    &mut cbplp,
                    &mut low,
                    context,
                )?;
            } else {
                syntax::decode_low_pass_subsampled(
                    &mut reader,
                    &mut entropy,
                    &mut cbplp,
                    &mut low,
                    context,
                    sampling,
                )?;
            }
            for component in 0..3 {
                components[component][index].coefficients.dc_low_pass = low[component];
                components[component][index].lp_qp_index = qp;
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

struct HpState {
    qp_indices: Vec<u8>,
    model_bits: Vec<[u8; 2]>,
}

struct HpPacketContext<'a> {
    plane: &'a crate::ImagePlaneHeader,
    components: &'a mut [Vec<SpatialMacroblock>; 3],
    lp_indices: &'a [u8],
    width: usize,
    height: usize,
    sampling: ChromaSampling,
}

fn decode_hp_packet(
    packet: &[u8],
    quantizers: &mut TileQuantizers,
    context: HpPacketContext<'_>,
) -> Result<HpState, TileDecodeError> {
    let HpPacketContext {
        plane,
        components,
        lp_indices,
        width,
        height,
        sampling,
    } = context;
    let mut reader = PacketBitReader::new(packet);
    parse_packet_prefix(&mut reader)?;
    quantizers.parse_high_pass_packet(&mut reader, plane)?;
    let mut state = HpState {
        qp_indices: Vec::with_capacity(lp_indices.len()),
        model_bits: Vec::with_capacity(lp_indices.len()),
    };
    let mut entropy = TileEntropyState::new();
    let mut cbphp = CbphpState::new(width);
    for y in 0..height {
        for x in 0..width {
            if x.is_multiple_of(16) {
                entropy.reset_scan_totals();
            }
            let index = y * width + x;
            let low = core::array::from_fn(|component| {
                components[component][index].coefficients.dc_low_pass
            });
            let mode = high_pass::prediction_mode_yuv(&low, sampling);
            let qp = quantizers.high_pass_index(&mut reader, lp_indices[index])?;
            let high = high_pass::decode_yuv(
                &mut reader,
                &mut entropy,
                &mut cbphp,
                MacroblockPosition { width, x, y },
                sampling,
                mode,
                HighpassPayload::VlcOnly,
            )?;
            for (component, plane) in components.iter_mut().enumerate() {
                plane[index].coefficients.high_pass = high.coefficients[component];
                plane[index].hp_prediction = mode;
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

fn decode_flex_packet(
    packet: &[u8],
    trim_present: bool,
    components: &mut [Vec<SpatialMacroblock>; 3],
    state: &HpState,
    sampling: ChromaSampling,
) -> Result<(), TileDecodeError> {
    let mut reader = PacketBitReader::new(packet);
    parse_packet_prefix(&mut reader)?;
    let trim = if trim_present {
        read_u8(&mut reader, 4)?
    } else {
        0
    };
    for (index, model_bits) in state.model_bits.iter().enumerate() {
        for (component, plane) in components.iter_mut().enumerate() {
            combine_flex(
                &mut reader,
                &mut plane[index].coefficients.high_pass,
                model_bits[usize::from(component != 0)],
                trim,
                component_block_count(component, sampling),
            )?;
        }
    }
    consume_byte_alignment(&mut reader)
}

fn combine_flex(
    reader: &mut PacketBitReader<'_>,
    coefficients: &mut [i32; 256],
    model_bits: u8,
    trim: u8,
    block_count: usize,
) -> Result<(), TileDecodeError> {
    for (block_index, &hierarchical_block) in HIERARCHICAL_BLOCK_ORDER
        .iter()
        .enumerate()
        .take(block_count)
    {
        let block = if block_count == 16 {
            hierarchical_block
        } else {
            block_index
        };
        let destination = &mut coefficients[block * 16..(block + 1) * 16];
        let mut vlc = [0_i32; 16];
        vlc.copy_from_slice(destination);
        let flex = decode_flex_block(reader, &vlc, model_bits, trim)?;
        for coefficient in 1..16 {
            destination[coefficient] = vlc[coefficient]
                .checked_shl(u32::from(model_bits))
                .and_then(|value| value.checked_add(flex[coefficient]))
                .ok_or(TileDecodeError::ArithmeticOverflow("YUV444 flexbits"))?;
        }
    }
    Ok(())
}

fn finish_hp_without_flex(
    components: &mut [Vec<SpatialMacroblock>; 3],
    state: &HpState,
    sampling: ChromaSampling,
) -> Result<(), TileDecodeError> {
    for (index, model_bits) in state.model_bits.iter().enumerate() {
        for (component, plane) in components.iter_mut().enumerate() {
            let bits = u32::from(model_bits[usize::from(component != 0)]);
            let coefficients = &mut plane[index].coefficients.high_pass;
            for block in 0..component_block_count(component, sampling) {
                for coefficient in 1..16 {
                    let position = block * 16 + coefficient;
                    coefficients[position] = coefficients[position].checked_shl(bits).ok_or(
                        TileDecodeError::ArithmeticOverflow("YUV444 HP normalization"),
                    )?;
                }
            }
        }
    }
    Ok(())
}

const fn component_block_count(component: usize, sampling: ChromaSampling) -> usize {
    match (component, sampling) {
        (0, _) | (_, ChromaSampling::Cs444) => 16,
        (_, ChromaSampling::Cs422) => 8,
        (_, ChromaSampling::Cs420) => 4,
    }
}

fn assign_quantizers(
    components: &mut [Vec<SpatialMacroblock>; 3],
    quantizers: &TileQuantizers,
    lp: &[u8],
    hp: &[u8],
    scaled: bool,
) -> Result<(), TileDecodeError> {
    for (component, plane) in components.iter_mut().enumerate() {
        for (index, macroblock) in plane.iter_mut().enumerate() {
            let indices = QuantizerIndices {
                lp: lp.get(index).copied().unwrap_or(0),
                hp: hp.get(index).copied().unwrap_or(0),
            };
            macroblock.coefficients.quantizers =
                quantizers.reconstruction_steps_for(component, indices, scaled)?;
        }
    }
    Ok(())
}

fn decoded_tile(components: [Vec<SpatialMacroblock>; 3]) -> DecodedTile {
    DecodedTile {
        components: components.into_iter().collect(),
    }
}
