use jxr_core::PredictionMode;

use crate::{
    ImagePlaneHeader,
    entropy::{PacketBitReader, TileEntropyState, decode_flex_block},
};

use super::super::{
    TileDecodeError,
    cbphp::CbphpState,
    high_pass::{self, HIERARCHICAL_BLOCK_ORDER, HighpassPayload},
    quantizer::TileQuantizers,
    spatial::{
        MacroblockPosition, SpatialMacroblock, consume_byte_alignment, parse_packet_prefix, read_u8,
    },
};

pub(super) struct HighPassState {
    pub(super) qp_indices: Vec<u8>,
    model_bits: Vec<u8>,
    modes: Vec<PredictionMode>,
}

pub(super) fn decode(
    packet: &[u8],
    plane: &ImagePlaneHeader,
    quantizers: &mut TileQuantizers,
    decoded: &mut [SpatialMacroblock],
    low_pass_indices: &[u8],
    width: usize,
    height: usize,
) -> Result<HighPassState, TileDecodeError> {
    let mut reader = PacketBitReader::new(packet);
    parse_packet_prefix(&mut reader)?;
    quantizers.parse_high_pass_packet(&mut reader, plane)?;
    let mut state = HighPassState {
        qp_indices: Vec::with_capacity(decoded.len()),
        model_bits: Vec::with_capacity(decoded.len()),
        modes: Vec::with_capacity(decoded.len()),
    };
    let mut entropy = TileEntropyState::new();
    entropy.reset_tile();
    let mut cbphp = CbphpState::new(width);
    for y in 0..height {
        for x in 0..width {
            if x.is_multiple_of(16) {
                entropy.reset_scan_totals();
            }
            let index = y * width + x;
            let qp_index = quantizers.high_pass_index(&mut reader, low_pass_indices[index])?;
            let mode = high_pass::prediction_mode(&decoded[index].coefficients.dc_low_pass);
            let high = high_pass::decode_vlc(
                &mut reader,
                &mut entropy,
                &mut cbphp,
                MacroblockPosition { width, x, y },
                mode,
                HighpassPayload::VlcOnly,
            )?;
            decoded[index].coefficients.high_pass = high.coefficients;
            decoded[index].hp_prediction = mode;
            state.qp_indices.push(qp_index);
            state.model_bits.push(high.model_bits);
            state.modes.push(mode);
            if x + 1 == width || x.is_multiple_of(16) {
                entropy.hp_vlc.adapt();
                cbphp.adapt();
            }
        }
    }
    consume_byte_alignment(&mut reader)?;
    Ok(state)
}

pub(super) fn decode_flexbits(
    packet: &[u8],
    trim_flexbits_present: bool,
    decoded: &mut [SpatialMacroblock],
    state: &HighPassState,
) -> Result<(), TileDecodeError> {
    let mut reader = PacketBitReader::new(packet);
    parse_packet_prefix(&mut reader)?;
    let trim = if trim_flexbits_present {
        read_u8(&mut reader, 4)?
    } else {
        0
    };
    for (index, macroblock) in decoded.iter_mut().enumerate() {
        combine_flexbits(
            &mut reader,
            &mut macroblock.coefficients.high_pass,
            state.model_bits[index],
            trim,
        )?;
    }
    consume_byte_alignment(&mut reader)
}

pub(super) fn finish_without_flexbits(
    decoded: &mut [SpatialMacroblock],
    state: &HighPassState,
) -> Result<(), TileDecodeError> {
    for (index, macroblock) in decoded.iter_mut().enumerate() {
        let bits = u32::from(state.model_bits[index]);
        for block in 0..16 {
            for coefficient in 1..16 {
                let position = block * 16 + coefficient;
                macroblock.coefficients.high_pass[position] = macroblock.coefficients.high_pass
                    [position]
                    .checked_shl(bits)
                    .ok_or(TileDecodeError::ArithmeticOverflow("HP normalization"))?;
            }
        }
    }
    Ok(())
}

fn combine_flexbits(
    reader: &mut PacketBitReader<'_>,
    high_pass: &mut [i32; 256],
    model_bits: u8,
    trim: u8,
) -> Result<(), TileDecodeError> {
    for block_map in HIERARCHICAL_BLOCK_ORDER {
        let destination = &mut high_pass[block_map * 16..(block_map + 1) * 16];
        let mut vlc = [0_i32; 16];
        vlc.copy_from_slice(destination);
        let flex = decode_flex_block(reader, &vlc, model_bits, trim)?;
        for coefficient in 1..16 {
            destination[coefficient] = vlc[coefficient]
                .checked_shl(u32::from(model_bits))
                .and_then(|value| value.checked_add(flex[coefficient]))
                .ok_or(TileDecodeError::ArithmeticOverflow("HP flexbits"))?;
        }
    }
    Ok(())
}
