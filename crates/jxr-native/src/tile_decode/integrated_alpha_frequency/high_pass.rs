//! Integrated primary/alpha high-pass and flexbits frequency packets.

use jxr_core::{ChromaSampling, PredictionMode};

use crate::entropy::{PacketBitReader, TileEntropyState, decode_flex_block};

use super::{PlaneState, SpatialMacroblock, TileDecodeError};
use crate::tile_decode::{
    cbphp::CbphpState,
    high_pass::{self, HIERARCHICAL_BLOCK_ORDER, HighpassPayload},
    packet_slice,
    spatial::{MacroblockPosition, consume_byte_alignment, parse_packet_prefix, read_u8},
    yuv,
};

pub(super) fn decode_packet(
    source: &[u8],
    range: jxr_core::ByteRange,
    primary: &mut PlaneState<'_>,
    alpha: &mut PlaneState<'_>,
    width: usize,
    height: usize,
) -> Result<(), TileDecodeError> {
    let mut reader = PacketBitReader::new(packet_slice(source, range)?);
    parse_packet_prefix(&mut reader)?;
    parse_header(&mut reader, primary)?;
    parse_header(&mut reader, alpha)?;
    let mut primary_entropy = TileEntropyState::new();
    let mut alpha_entropy = TileEntropyState::new();
    primary_entropy.reset_tile();
    alpha_entropy.reset_tile();
    let mut primary_pattern = CbphpState::new_components(width, primary.components.len());
    let mut alpha_pattern = CbphpState::new_components(width, alpha.components.len());
    for y in 0..height {
        for x in 0..width {
            let position = MacroblockPosition { width, x, y };
            decode_if_present(
                &mut reader,
                primary,
                &mut primary_entropy,
                &mut primary_pattern,
                position,
            )?;
            decode_if_present(
                &mut reader,
                alpha,
                &mut alpha_entropy,
                &mut alpha_pattern,
                position,
            )?;
        }
    }
    consume_byte_alignment(&mut reader)
}

fn parse_header(
    reader: &mut PacketBitReader<'_>,
    state: &mut PlaneState<'_>,
) -> Result<(), TileDecodeError> {
    if state.bands.has_high_pass() {
        let header = state.header;
        state
            .quantizers_mut()?
            .parse_high_pass_packet(reader, header)?;
    }
    Ok(())
}

fn decode_if_present(
    reader: &mut PacketBitReader<'_>,
    state: &mut PlaneState<'_>,
    entropy: &mut TileEntropyState,
    pattern: &mut CbphpState,
    position: MacroblockPosition,
) -> Result<(), TileDecodeError> {
    if !state.bands.has_high_pass() {
        return Ok(());
    }
    if position.x.is_multiple_of(16) {
        entropy.reset_scan_totals();
    }
    let index = position.y * position.width + position.x;
    let low_index = *state
        .low_pass_indices
        .get(index)
        .ok_or(TileDecodeError::InvalidPlan("frequency LP quantizer index"))?;
    let qp = state.quantizers()?.high_pass_index(reader, low_index)?;
    let mode = prediction_mode(state, index)?;
    let model_bits = match state.header.internal_color_format {
        0 => decode_luma(reader, state, entropy, pattern, position, index, mode)?,
        1..=3 => decode_yuv(reader, state, entropy, pattern, position, index, mode)?,
        4 | 6 => decode_multi(reader, state, entropy, pattern, position, index, mode)?,
        _ => {
            return Err(TileDecodeError::Unsupported(
                "integrated alpha frequency HP primary format",
            ));
        }
    };
    state.high_pass_indices.push(qp);
    state.high_pass_model_bits.push(model_bits);
    if position.x + 1 == position.width || position.x.is_multiple_of(16) {
        entropy.hp_vlc.adapt();
        pattern.adapt();
    }
    Ok(())
}

fn prediction_mode(
    state: &PlaneState<'_>,
    index: usize,
) -> Result<PredictionMode, TileDecodeError> {
    match state.header.internal_color_format {
        0 | 6 => Ok(high_pass::prediction_mode(
            &state.components[0][index].coefficients.dc_low_pass,
        )),
        1..=3 => {
            let low = core::array::from_fn(|component| {
                state.components[component][index].coefficients.dc_low_pass
            });
            Ok(high_pass::prediction_mode_yuv(
                &low,
                yuv::sampling(state.header.internal_color_format)?,
            ))
        }
        4 => {
            let low: Vec<_> = state
                .components
                .iter()
                .map(|component| component[index].coefficients.dc_low_pass)
                .collect();
            high_pass::prediction_mode_yuvk(&low)
        }
        _ => Err(TileDecodeError::InvalidPlan(
            "integrated alpha HP prediction format",
        )),
    }
}

fn decode_luma(
    reader: &mut PacketBitReader<'_>,
    state: &mut PlaneState<'_>,
    entropy: &mut TileEntropyState,
    pattern: &mut CbphpState,
    position: MacroblockPosition,
    index: usize,
    mode: PredictionMode,
) -> Result<[u8; 2], TileDecodeError> {
    let high = high_pass::decode_vlc(
        reader,
        entropy,
        pattern,
        position,
        mode,
        HighpassPayload::VlcOnly,
    )?;
    state.components[0][index].coefficients.high_pass = high.coefficients;
    state.components[0][index].hp_prediction = mode;
    Ok([high.model_bits; 2])
}

fn decode_yuv(
    reader: &mut PacketBitReader<'_>,
    state: &mut PlaneState<'_>,
    entropy: &mut TileEntropyState,
    pattern: &mut CbphpState,
    position: MacroblockPosition,
    index: usize,
    mode: PredictionMode,
) -> Result<[u8; 2], TileDecodeError> {
    let sampling = yuv::sampling(state.header.internal_color_format)?;
    let high = high_pass::decode_yuv(
        reader,
        entropy,
        pattern,
        position,
        sampling,
        mode,
        HighpassPayload::VlcOnly,
    )?;
    let components: &mut [Vec<SpatialMacroblock>; 3] =
        state
            .components
            .as_mut_slice()
            .try_into()
            .map_err(|_| TileDecodeError::InvalidPlan("integrated YUV component count"))?;
    for (component, plane) in components.iter_mut().enumerate() {
        plane[index].coefficients.high_pass = high.coefficients[component];
        plane[index].hp_prediction = if component == 0 || sampling == ChromaSampling::Cs444 {
            mode
        } else {
            PredictionMode::None
        };
    }
    Ok(high.model_bits)
}

fn decode_multi(
    reader: &mut PacketBitReader<'_>,
    state: &mut PlaneState<'_>,
    entropy: &mut TileEntropyState,
    pattern: &mut CbphpState,
    position: MacroblockPosition,
    index: usize,
    mode: PredictionMode,
) -> Result<[u8; 2], TileDecodeError> {
    let high = high_pass::decode_components(
        reader,
        entropy,
        pattern,
        position,
        state.components.len(),
        mode,
        HighpassPayload::VlcOnly,
    )?;
    for (component, coefficients) in state.components.iter_mut().zip(high.coefficients) {
        component[index].coefficients.high_pass = coefficients;
        component[index].hp_prediction = mode;
    }
    Ok(high.model_bits)
}

pub(super) fn decode_flex_packet(
    source: &[u8],
    range: jxr_core::ByteRange,
    primary: &mut PlaneState<'_>,
    alpha: &mut PlaneState<'_>,
    trim_present: bool,
) -> Result<(), TileDecodeError> {
    let mut reader = PacketBitReader::new(packet_slice(source, range)?);
    parse_packet_prefix(&mut reader)?;
    let trim = if trim_present {
        read_u8(&mut reader, 4)?
    } else {
        0
    };
    let count = primary.high_pass_model_bits.len();
    for index in 0..count {
        decode_flex_if_present(&mut reader, primary, index, trim)?;
        decode_flex_if_present(&mut reader, alpha, index, trim)?;
    }
    consume_byte_alignment(&mut reader)
}

fn decode_flex_if_present(
    reader: &mut PacketBitReader<'_>,
    state: &mut PlaneState<'_>,
    index: usize,
    trim: u8,
) -> Result<(), TileDecodeError> {
    if !state.bands.has_flexbits() {
        return Ok(());
    }
    let model_bits = state.high_pass_model_bits[index];
    let format = state.header.internal_color_format;
    for (component, plane) in state.components.iter_mut().enumerate() {
        combine_flexbits(
            reader,
            &mut plane[index].coefficients.high_pass,
            model_bits[usize::from(component != 0)],
            trim,
            component_block_count(format, component)?,
        )?;
    }
    Ok(())
}

fn combine_flexbits(
    reader: &mut PacketBitReader<'_>,
    coefficients: &mut [i32; 256],
    model_bits: u8,
    trim: u8,
    block_count: usize,
) -> Result<(), TileDecodeError> {
    for (block_index, &hierarchical) in HIERARCHICAL_BLOCK_ORDER
        .iter()
        .enumerate()
        .take(block_count)
    {
        let block = if block_count == 16 {
            hierarchical
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
                .ok_or(TileDecodeError::ArithmeticOverflow(
                    "integrated alpha HP flexbits",
                ))?;
        }
    }
    Ok(())
}

pub(super) fn finish_without_flexbits(
    state: &mut PlaneState<'_>,
    flexbits_escaped: bool,
) -> Result<(), TileDecodeError> {
    if !state.bands.has_high_pass() || (state.bands.has_flexbits() && !flexbits_escaped) {
        return Ok(());
    }
    let format = state.header.internal_color_format;
    for (component, plane) in state.components.iter_mut().enumerate() {
        let block_count = component_block_count(format, component)?;
        for (index, macroblock) in plane.iter_mut().enumerate() {
            let bits = u32::from(state.high_pass_model_bits[index][usize::from(component != 0)]);
            for block in 0..block_count {
                for coefficient in 1..16 {
                    let position = block * 16 + coefficient;
                    macroblock.coefficients.high_pass[position] = macroblock.coefficients.high_pass
                        [position]
                        .checked_shl(bits)
                        .ok_or(TileDecodeError::ArithmeticOverflow(
                            "integrated alpha HP normalization",
                        ))?;
                }
            }
        }
    }
    Ok(())
}

fn component_block_count(format: u8, component: usize) -> Result<usize, TileDecodeError> {
    match (format, component) {
        (1, 1..) => Ok(4),
        (2, 1..) => Ok(8),
        (0 | 1..=4 | 6, _) => Ok(16),
        _ => Err(TileDecodeError::InvalidPlan(
            "integrated alpha component block geometry",
        )),
    }
}
