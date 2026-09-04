//! Spatial-mode Y-only tile traversal.

use jxr_core::{BandPresence, PredictionMode};

use crate::{
    ImagePlaneHeader,
    entropy::{
        ColourModel, ComponentClass, FrequencyBand, PacketBitReader, TileEntropyState,
        decode_ac_block, decode_dc_coefficient, decode_lp_refinement,
    },
    reconstruct::QuantizedMacroblock,
};

use super::{
    TileDecodeError,
    cbphp::CbphpState,
    high_pass::{self, HighpassPayload},
    quantizer::TileQuantizers,
};

const TILE_START_CODE: u32 = 1;
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SpatialMacroblock {
    pub(super) coefficients: QuantizedMacroblock,
    pub(super) prediction: PredictionMode,
    pub(super) hp_prediction: PredictionMode,
    pub(super) lp_qp_index: u8,
}

pub(super) fn decode_spatial_packet(
    packet: &[u8],
    plane: &ImagePlaneHeader,
    bands: BandPresence,
    tile_width: u32,
    tile_height: u32,
    trim_flexbits_present: bool,
) -> Result<Vec<SpatialMacroblock>, TileDecodeError> {
    validate_y_only(plane)?;
    let mut reader = PacketBitReader::new(packet);
    parse_packet_prefix(&mut reader)?;
    let trim_flexbits = if trim_flexbits_present {
        read_u8(&mut reader, 4)?
    } else {
        0
    };
    let quantizers = TileQuantizers::parse(&mut reader, plane)?;
    let width = usize::try_from(tile_width)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("tile width conversion"))?;
    let height = usize::try_from(tile_height)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("tile height conversion"))?;
    let count = width
        .checked_mul(height)
        .ok_or(TileDecodeError::ArithmeticOverflow("tile macroblock count"))?;
    let mut decoded = Vec::with_capacity(count);
    let mut entropy = TileEntropyState::new();
    entropy.reset_tile();
    let mut cbphp = CbphpState::new(width);

    for y in 0..height {
        for x in 0..width {
            if x.is_multiple_of(16) {
                entropy.reset_scan_totals();
            }
            let qp_indices = quantizers.indices(&mut reader)?;
            let (low, prediction) = decode_low_bands(
                &mut reader,
                &mut entropy,
                &decoded,
                MacroblockPosition { width, x, y },
                qp_indices.lp,
                bands,
            )?;
            let hp_mode = high_pass::prediction_mode(&low);
            let high = decode_high_band(
                &mut reader,
                &mut entropy,
                &mut cbphp,
                MacroblockPosition { width, x, y },
                hp_mode,
                bands,
                trim_flexbits,
            )?;
            let steps = quantizers.reconstruction_steps(qp_indices, plane.scaled)?;
            decoded.push(SpatialMacroblock {
                coefficients: QuantizedMacroblock {
                    dc_low_pass: low,
                    high_pass: high,
                    quantizers: steps,
                    bands,
                },
                prediction,
                hp_prediction: hp_mode,
                lp_qp_index: qp_indices.lp,
            });
            adapt_at_boundary(&mut entropy, &mut cbphp, x, width);
        }
    }
    consume_byte_alignment(&mut reader)?;
    Ok(decoded)
}

pub(super) fn validate_y_only(plane: &ImagePlaneHeader) -> Result<(), TileDecodeError> {
    if plane.internal_color_format == 0 && plane.components == 1 {
        Ok(())
    } else {
        Err(TileDecodeError::Unsupported(
            "only single-component Y-only tile decoding is implemented",
        ))
    }
}

pub(super) fn parse_packet_prefix(reader: &mut PacketBitReader<'_>) -> Result<(), TileDecodeError> {
    let start = u32::try_from(reader.read_bits(24)?)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("tile start code"))?;
    if start != TILE_START_CODE {
        return Err(TileDecodeError::InvalidStartCode { value: start });
    }
    let _arbitrary_byte = reader.read_bits(8)?;
    Ok(())
}

pub(super) fn decode_low_bands(
    reader: &mut PacketBitReader<'_>,
    entropy: &mut TileEntropyState,
    decoded: &[SpatialMacroblock],
    position: MacroblockPosition,
    lp_qp_index: u8,
    bands: BandPresence,
) -> Result<([i32; 16], PredictionMode), TileDecodeError> {
    let prediction = decode_dc_band(reader, entropy, decoded, position)?;
    let mut low = [0_i32; 16];
    low[0] = prediction.value;
    if bands.has_low_pass() {
        decode_low_pass(reader, entropy, &mut low)?;
        predict_low_pass(&mut low, decoded, position, prediction.mode, lp_qp_index)?;
    }
    Ok((low, prediction.mode))
}

pub(super) fn decode_dc_band(
    reader: &mut PacketBitReader<'_>,
    entropy: &mut TileEntropyState,
    decoded: &[SpatialMacroblock],
    position: MacroblockPosition,
) -> Result<DcPrediction, TileDecodeError> {
    let has_abs_level = reader.read_bit()?;
    let dc = decode_dc_coefficient(
        reader,
        entropy.dc_model.bits(false),
        has_abs_level,
        ComponentClass::Luma,
        &mut entropy.dc_vlc,
    )?;
    entropy
        .dc_model
        .update([i32::from(has_abs_level), 0], ColourModel::LumaOnly)?;
    predict_dc(dc, decoded, position.width, position.x, position.y)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DcPrediction {
    pub(super) value: i32,
    pub(super) mode: PredictionMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MacroblockPosition {
    pub(super) width: usize,
    pub(super) x: usize,
    pub(super) y: usize,
}

pub(super) fn predict_dc(
    value: i32,
    decoded: &[SpatialMacroblock],
    width: usize,
    x: usize,
    y: usize,
) -> Result<DcPrediction, TileDecodeError> {
    let mode = match (x == 0, y == 0) {
        (true, true) => PredictionMode::None,
        (true, false) => PredictionMode::FromTop,
        (false, true) => PredictionMode::FromLeft,
        (false, false) => {
            let left = macroblock(decoded, width, x - 1, y)?.dc_low_pass[0];
            let top = macroblock(decoded, width, x, y - 1)?.dc_low_pass[0];
            let top_left = macroblock(decoded, width, x - 1, y - 1)?.dc_low_pass[0];
            let horizontal = top_left.abs_diff(left);
            let vertical = top_left.abs_diff(top);
            if horizontal.saturating_mul(4) < vertical {
                PredictionMode::FromTop
            } else if vertical.saturating_mul(4) < horizontal {
                PredictionMode::FromLeft
            } else {
                PredictionMode::FromTopLeft
            }
        }
    };
    let predicted = match mode {
        PredictionMode::None => value,
        PredictionMode::FromLeft => checked_add(
            value,
            macroblock(decoded, width, x - 1, y)?.dc_low_pass[0],
            "left DC prediction",
        )?,
        PredictionMode::FromTop => checked_add(
            value,
            macroblock(decoded, width, x, y - 1)?.dc_low_pass[0],
            "top DC prediction",
        )?,
        PredictionMode::FromTopLeft => {
            let left = macroblock(decoded, width, x - 1, y)?.dc_low_pass[0];
            let top = macroblock(decoded, width, x, y - 1)?.dc_low_pass[0];
            let average = (i64::from(left) + i64::from(top)) >> 1;
            let average = i32::try_from(average)
                .map_err(|_| TileDecodeError::ArithmeticOverflow("DC prediction average"))?;
            checked_add(value, average, "two-neighbour DC prediction")?
        }
    };
    Ok(DcPrediction {
        value: predicted,
        mode,
    })
}

pub(super) fn decode_low_pass(
    reader: &mut PacketBitReader<'_>,
    entropy: &mut TileEntropyState,
    output: &mut [i32; 16],
) -> Result<(), TileDecodeError> {
    let coded = reader.read_bit()?;
    let mut non_zero = 0_i32;
    if coded {
        let block = decode_ac_block(
            reader,
            FrequencyBand::Lowpass,
            ComponentClass::Luma,
            1,
            &mut entropy.lp_vlc,
        )?;
        non_zero = i32::from(block.non_zero_count());
        block.inverse_scan_lp(&mut entropy.lp_scan, output)?;
    }
    let model_bits = entropy.lp_model.bits(false);
    decode_lp_refinement(reader, output, model_bits)?;
    entropy
        .lp_model
        .update([non_zero, 0], ColourModel::LumaOnly)?;
    Ok(())
}

pub(super) fn predict_low_pass(
    current: &mut [i32; 16],
    decoded: &[SpatialMacroblock],
    position: MacroblockPosition,
    dc_mode: PredictionMode,
    lp_qp_index: u8,
) -> Result<(), TileDecodeError> {
    let mode = low_pass_prediction_mode(decoded, position, dc_mode, lp_qp_index)?;
    let MacroblockPosition { width, x, y } = position;
    match mode {
        PredictionMode::FromLeft => {
            let left = &macroblock(decoded, width, x - 1, y)?.dc_low_pass;
            for index in [4, 8, 12] {
                current[index] = checked_add(current[index], left[index], "left LP prediction")?;
            }
        }
        PredictionMode::FromTop => {
            let top = &macroblock(decoded, width, x, y - 1)?.dc_low_pass;
            for index in 1..=3 {
                current[index] = checked_add(current[index], top[index], "top LP prediction")?;
            }
        }
        _ => {}
    }
    Ok(())
}

pub(in crate::tile_decode) fn low_pass_prediction_mode(
    decoded: &[SpatialMacroblock],
    position: MacroblockPosition,
    dc_mode: PredictionMode,
    lp_qp_index: u8,
) -> Result<PredictionMode, TileDecodeError> {
    let MacroblockPosition { width, x, y } = position;
    Ok(match dc_mode {
        PredictionMode::FromLeft
            if spatial_macroblock(decoded, width, x - 1, y)?.lp_qp_index == lp_qp_index =>
        {
            PredictionMode::FromLeft
        }
        PredictionMode::FromTop
            if spatial_macroblock(decoded, width, x, y - 1)?.lp_qp_index == lp_qp_index =>
        {
            PredictionMode::FromTop
        }
        _ => PredictionMode::None,
    })
}

pub(super) fn decode_high_band(
    reader: &mut PacketBitReader<'_>,
    entropy: &mut TileEntropyState,
    cbphp: &mut CbphpState,
    position: MacroblockPosition,
    mode: PredictionMode,
    bands: BandPresence,
    trim_flexbits: u8,
) -> Result<[i32; 256], TileDecodeError> {
    if !bands.has_high_pass() {
        return Ok([0_i32; 256]);
    }
    let decoded = high_pass::decode_vlc(
        reader,
        entropy,
        cbphp,
        position,
        mode,
        HighpassPayload::Combined {
            flexbits_present: bands.has_flexbits(),
            trim_flexbits,
        },
    )?;
    Ok(decoded.coefficients)
}

pub(super) fn adapt_at_boundary(
    entropy: &mut TileEntropyState,
    cbphp: &mut CbphpState,
    x: usize,
    width: usize,
) {
    if x + 1 == width || x.is_multiple_of(16) {
        entropy.dc_vlc.adapt();
        entropy.lp_vlc.adapt();
        entropy.hp_vlc.adapt();
        cbphp.adapt();
    }
}

pub(super) fn consume_byte_alignment(
    reader: &mut PacketBitReader<'_>,
) -> Result<(), TileDecodeError> {
    while !reader.bit_position().is_multiple_of(8) {
        let _ = reader.read_bit()?;
    }
    Ok(())
}

fn macroblock(
    decoded: &[SpatialMacroblock],
    width: usize,
    x: usize,
    y: usize,
) -> Result<&QuantizedMacroblock, TileDecodeError> {
    decoded
        .get(y.saturating_mul(width).saturating_add(x))
        .map(|entry| &entry.coefficients)
        .ok_or(TileDecodeError::InvalidPlan("prediction neighbour"))
}

pub(in crate::tile_decode) fn spatial_macroblock(
    decoded: &[SpatialMacroblock],
    width: usize,
    x: usize,
    y: usize,
) -> Result<&SpatialMacroblock, TileDecodeError> {
    decoded
        .get(y.saturating_mul(width).saturating_add(x))
        .ok_or(TileDecodeError::InvalidPlan("prediction neighbour"))
}

pub(in crate::tile_decode) fn checked_add(
    left: i32,
    right: i32,
    operation: &'static str,
) -> Result<i32, TileDecodeError> {
    left.checked_add(right)
        .ok_or(TileDecodeError::ArithmeticOverflow(operation))
}

pub(super) fn read_u8(reader: &mut PacketBitReader<'_>, bits: u8) -> Result<u8, TileDecodeError> {
    u8::try_from(reader.read_bits(bits)?)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("tile field conversion"))
}

#[cfg(test)]
mod alignment_tests {
    use super::*;

    #[test]
    fn byte_alignment_discards_unspecified_trailing_bits() {
        let mut reader = PacketBitReader::new(&[0xff]);
        assert!(reader.read_bit().unwrap());
        consume_byte_alignment(&mut reader).unwrap();
        assert_eq!(reader.bit_position(), 8);
    }
}
