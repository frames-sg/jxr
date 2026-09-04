//! Shared YUV DC and low-pass syntax.

use jxr_core::{ChromaSampling, PredictionMode};

use crate::entropy::{
    ColourModel, ComponentClass, FrequencyBand, PacketBitReader, TileEntropyState, decode_ac_block,
    decode_dc_coefficient, decode_lp_refinement, decode_lp_refinement_at,
};

use super::super::{
    TileDecodeError,
    spatial::{
        MacroblockPosition, SpatialMacroblock, checked_add, low_pass_prediction_mode,
        predict_low_pass, spatial_macroblock,
    },
};

const DC_PATTERN: [(u16, u8, u8); 8] = [
    (0b10, 2, 0),
    (0b001, 3, 1),
    (0b00001, 5, 2),
    (0b0001, 4, 3),
    (0b11, 2, 4),
    (0b010, 3, 5),
    (0b00000, 5, 6),
    (0b011, 3, 7),
];
const LP_PATTERN: [(u16, u8, u8); 8] = [
    (0b0, 1, 0),
    (0b100, 3, 1),
    (0b1010, 4, 2),
    (0b1011, 4, 3),
    (0b1100, 4, 4),
    (0b1101, 4, 5),
    (0b1110, 4, 6),
    (0b1111, 4, 7),
];
const LP_PATTERN_SUBSAMPLED: [(u16, u8, u8); 4] =
    [(0b0, 1, 0), (0b10, 2, 1), (0b110, 3, 2), (0b111, 3, 3)];

pub(in crate::tile_decode) struct CbplpState {
    sampling: ChromaSampling,
    zero: i8,
    maximum: i8,
}

impl CbplpState {
    pub(in crate::tile_decode) const fn new(sampling: ChromaSampling) -> Self {
        Self {
            sampling,
            zero: 1,
            maximum: 1,
        }
    }

    fn decode(&mut self, reader: &mut PacketBitReader<'_>) -> Result<u8, TileDecodeError> {
        let maximum_pattern = if self.sampling == ChromaSampling::Cs444 {
            7
        } else {
            3
        };
        let pattern = if self.zero <= 0 || self.maximum < 0 {
            let mut pattern = if self.sampling == ChromaSampling::Cs444 {
                read_code(reader, &LP_PATTERN, "CBPLP_YUV1")?
            } else {
                read_code(reader, &LP_PATTERN_SUBSAMPLED, "CBPLP_YUV1")?
            };
            if self.maximum < self.zero {
                pattern = maximum_pattern - pattern;
            }
            pattern
        } else {
            let bits = if self.sampling == ChromaSampling::Cs444 {
                3
            } else {
                2
            };
            u8::try_from(reader.read_bits(bits)?)
                .map_err(|_| TileDecodeError::ArithmeticOverflow("CBPLP_YUV2"))?
        };
        let zero_delta = if pattern == 0 { 4 } else { 0 };
        let maximum_delta = if pattern == maximum_pattern { 4 } else { 0 };
        self.zero = (self.zero + 1 - zero_delta).clamp(-8, 7);
        self.maximum = (self.maximum + 1 - maximum_delta).clamp(-8, 7);
        Ok(pattern)
    }
}

pub(in crate::tile_decode) fn decode_dc(
    reader: &mut PacketBitReader<'_>,
    entropy: &mut TileEntropyState,
    decoded: &[Vec<SpatialMacroblock>; 3],
    position: MacroblockPosition,
    sampling: ChromaSampling,
) -> Result<[(i32, PredictionMode); 3], TileDecodeError> {
    let pattern = read_code(reader, &DC_PATTERN, "VAL_DC_YUV")?;
    let mut residual = [0_i32; 3];
    let mut lap_mean = [0_i32; 2];
    for component in 0..3 {
        let chroma = component != 0;
        let class = if chroma {
            ComponentClass::Chroma
        } else {
            ComponentClass::Luma
        };
        let present = pattern & (4 >> component) != 0;
        lap_mean[usize::from(chroma)] += i32::from(present);
        residual[component] = decode_dc_coefficient(
            reader,
            entropy.dc_model.bits(chroma),
            present,
            class,
            &mut entropy.dc_vlc,
        )?;
    }
    entropy
        .dc_model
        .update(lap_mean, ColourModel::Other { components: 3 })?;
    predict_dc_yuv(residual, decoded, position, sampling)
}

fn predict_dc_yuv(
    residual: [i32; 3],
    decoded: &[Vec<SpatialMacroblock>; 3],
    position: MacroblockPosition,
    sampling: ChromaSampling,
) -> Result<[(i32, PredictionMode); 3], TileDecodeError> {
    let MacroblockPosition { width, x, y } = position;
    let mode = match (x == 0, y == 0) {
        (true, true) => PredictionMode::None,
        (true, false) => PredictionMode::FromTop,
        (false, true) => PredictionMode::FromLeft,
        (false, false) => {
            let mut horizontal = 0_u32;
            let mut vertical = 0_u32;
            let luma_weight = match sampling {
                ChromaSampling::Cs420 => 8,
                ChromaSampling::Cs422 => 4,
                ChromaSampling::Cs444 => 2,
            };
            for (component, weight) in [luma_weight, 1, 1].into_iter().enumerate() {
                let left = dc_at(&decoded[component], width, x - 1, y)?;
                let top = dc_at(&decoded[component], width, x, y - 1)?;
                let top_left = dc_at(&decoded[component], width, x - 1, y - 1)?;
                horizontal =
                    horizontal.saturating_add(top_left.abs_diff(left).saturating_mul(weight));
                vertical = vertical.saturating_add(top_left.abs_diff(top).saturating_mul(weight));
            }
            if horizontal.saturating_mul(4) < vertical {
                PredictionMode::FromTop
            } else if vertical.saturating_mul(4) < horizontal {
                PredictionMode::FromLeft
            } else {
                PredictionMode::FromTopLeft
            }
        }
    };
    let mut output = [(0_i32, mode); 3];
    for component in 0..3 {
        let predicted = match mode {
            PredictionMode::None => Some(residual[component]),
            PredictionMode::FromLeft => {
                residual[component].checked_add(dc_at(&decoded[component], width, x - 1, y)?)
            }
            PredictionMode::FromTop => {
                residual[component].checked_add(dc_at(&decoded[component], width, x, y - 1)?)
            }
            PredictionMode::FromTopLeft => {
                let left = dc_at(&decoded[component], width, x - 1, y)?;
                let top = dc_at(&decoded[component], width, x, y - 1)?;
                let rounding = i64::from(component != 0 && sampling != ChromaSampling::Cs444);
                let average = i32::try_from((i64::from(left) + i64::from(top) + rounding) >> 1)
                    .map_err(|_| TileDecodeError::ArithmeticOverflow("YUV DC average"))?;
                residual[component].checked_add(average)
            }
        }
        .ok_or(TileDecodeError::ArithmeticOverflow("YUV DC prediction"))?;
        output[component].0 = predicted;
    }
    Ok(output)
}

fn dc_at(
    decoded: &[SpatialMacroblock],
    width: usize,
    x: usize,
    y: usize,
) -> Result<i32, TileDecodeError> {
    decoded
        .get(y.saturating_mul(width).saturating_add(x))
        .map(|macroblock| macroblock.coefficients.dc_low_pass[0])
        .ok_or(TileDecodeError::InvalidPlan("YUV DC prediction neighbour"))
}

#[derive(Clone, Copy)]
pub(in crate::tile_decode) struct LowPassContext<'a> {
    pub(in crate::tile_decode) decoded: &'a [Vec<SpatialMacroblock>; 3],
    pub(in crate::tile_decode) position: MacroblockPosition,
    pub(in crate::tile_decode) qp_index: u8,
    pub(in crate::tile_decode) predictions: [PredictionMode; 3],
}

pub(in crate::tile_decode) fn decode_low_pass_444(
    reader: &mut PacketBitReader<'_>,
    entropy: &mut TileEntropyState,
    cbplp: &mut CbplpState,
    low: &mut [[i32; 16]; 3],
    context: LowPassContext<'_>,
) -> Result<(), TileDecodeError> {
    let pattern = cbplp.decode(reader)?;
    let mut lap_mean = [0_i32; 2];
    for (component, coefficients) in low.iter_mut().enumerate() {
        let chroma = component != 0;
        if pattern & (1 << component) != 0 {
            let block = decode_ac_block(
                reader,
                FrequencyBand::Lowpass,
                if chroma {
                    ComponentClass::Chroma
                } else {
                    ComponentClass::Luma
                },
                1,
                &mut entropy.lp_vlc,
            )?;
            lap_mean[usize::from(chroma)] += i32::from(block.non_zero_count());
            block.inverse_scan_lp(&mut entropy.lp_scan, coefficients)?;
        }
        decode_lp_refinement(reader, coefficients, entropy.lp_model.bits(chroma))?;
        predict_low_pass(
            coefficients,
            &context.decoded[component],
            context.position,
            context.predictions[component],
            context.qp_index,
        )?;
    }
    entropy
        .lp_model
        .update(lap_mean, ColourModel::Other { components: 3 })?;
    Ok(())
}

pub(in crate::tile_decode) fn decode_low_pass_subsampled(
    reader: &mut PacketBitReader<'_>,
    entropy: &mut TileEntropyState,
    cbplp: &mut CbplpState,
    low: &mut [[i32; 16]; 3],
    context: LowPassContext<'_>,
    sampling: ChromaSampling,
) -> Result<(), TileDecodeError> {
    if sampling == ChromaSampling::Cs444 {
        return Err(TileDecodeError::InvalidPlan(
            "subsampled LP decoder received YUV444",
        ));
    }
    let pattern = cbplp.decode(reader)?;
    let mut lap_mean = [0_i32; 2];
    if pattern & 1 != 0 {
        let block = decode_ac_block(
            reader,
            FrequencyBand::Lowpass,
            ComponentClass::Luma,
            1,
            &mut entropy.lp_vlc,
        )?;
        lap_mean[0] = i32::from(block.non_zero_count());
        block.inverse_scan_lp(&mut entropy.lp_scan, &mut low[0])?;
    }
    decode_lp_refinement(reader, &mut low[0], entropy.lp_model.bits(false))?;
    if pattern & 2 != 0 {
        let start = if sampling == ChromaSampling::Cs420 {
            10
        } else {
            2
        };
        let block = decode_ac_block(
            reader,
            FrequencyBand::Lowpass,
            ComponentClass::Chroma,
            start,
            &mut entropy.lp_vlc,
        )?;
        lap_mean[1] = i32::from(block.non_zero_count());
        remap_joint_chroma(&block, low, sampling)?;
    }
    let positions: &[usize] = if sampling == ChromaSampling::Cs420 {
        &[2, 1, 3]
    } else {
        &[2, 1, 3, 4, 6, 5, 7]
    };
    decode_subsampled_chroma_refinement(reader, low, positions, entropy.lp_model.bits(true))?;
    predict_low_pass(
        &mut low[0],
        &context.decoded[0],
        context.position,
        context.predictions[0],
        context.qp_index,
    )?;
    for (component, coefficients) in low[1..].iter_mut().enumerate() {
        predict_subsampled_chroma(
            coefficients,
            &context.decoded[component + 1],
            context.position,
            context.predictions[component + 1],
            context.qp_index,
            sampling,
        )?;
    }
    entropy.lp_model.update(
        lap_mean,
        if sampling == ChromaSampling::Cs420 {
            ColourModel::Yuv420
        } else {
            ColourModel::Yuv422
        },
    )?;
    Ok(())
}

fn decode_subsampled_chroma_refinement(
    reader: &mut PacketBitReader<'_>,
    low: &mut [[i32; 16]; 3],
    positions: &[usize],
    model_bits: u8,
) -> Result<(), TileDecodeError> {
    for position in positions {
        for coefficients in &mut low[1..] {
            decode_lp_refinement_at(
                reader,
                coefficients,
                core::slice::from_ref(position),
                model_bits,
            )?;
        }
    }
    Ok(())
}

fn remap_joint_chroma(
    block: &crate::entropy::DecodedBlock,
    low: &mut [[i32; 16]; 3],
    sampling: ChromaSampling,
) -> Result<(), TileDecodeError> {
    const REMAP: [usize; 7] = [4, 1, 2, 3, 5, 6, 7];
    const TRANSPOSE_420: [usize; 4] = [0, 2, 1, 3];
    const TRANSPOSE_422: [usize; 8] = [0, 2, 1, 3, 4, 6, 5, 7];
    let count = if sampling == ChromaSampling::Cs420 {
        6
    } else {
        14
    };
    let mut temporary = [0_i32; 14];
    let mut cursor = 0_usize;
    for entry in block.entries() {
        cursor = cursor
            .checked_add(usize::from(entry.run))
            .filter(|&index| index < count)
            .ok_or(TileDecodeError::InvalidPlan("subsampled chroma LP run"))?;
        temporary[cursor] = entry.level;
        cursor += 1;
    }
    for (index, &value) in temporary[..count].iter().enumerate() {
        let remap_index = (index >> 1) + usize::from(sampling == ChromaSampling::Cs420);
        let remapped = REMAP[remap_index];
        let position = if sampling == ChromaSampling::Cs420 {
            TRANSPOSE_420[remapped]
        } else {
            TRANSPOSE_422[remapped]
        };
        low[(index & 1) + 1][position] = value;
    }
    Ok(())
}

fn predict_subsampled_chroma(
    current: &mut [i32; 16],
    decoded: &[SpatialMacroblock],
    position: MacroblockPosition,
    dc_mode: PredictionMode,
    lp_qp_index: u8,
    sampling: ChromaSampling,
) -> Result<(), TileDecodeError> {
    let mode = low_pass_prediction_mode(decoded, position, dc_mode, lp_qp_index)?;
    let MacroblockPosition { width, x, y } = position;
    match (sampling, mode) {
        (ChromaSampling::Cs420, PredictionMode::FromLeft) => {
            let left = &spatial_macroblock(decoded, width, x - 1, y)?
                .coefficients
                .dc_low_pass;
            current[2] = checked_add(current[2], left[2], "YUV420 left LP prediction")?;
        }
        (ChromaSampling::Cs420, PredictionMode::FromTop) => {
            let top = &spatial_macroblock(decoded, width, x, y - 1)?
                .coefficients
                .dc_low_pass;
            current[1] = checked_add(current[1], top[1], "YUV420 top LP prediction")?;
        }
        (ChromaSampling::Cs422, PredictionMode::FromLeft) => {
            let left = &spatial_macroblock(decoded, width, x - 1, y)?
                .coefficients
                .dc_low_pass;
            for index in [4, 2, 6] {
                current[index] =
                    checked_add(current[index], left[index], "YUV422 left LP prediction")?;
            }
        }
        (ChromaSampling::Cs422, PredictionMode::FromTop) => {
            let top = &spatial_macroblock(decoded, width, x, y - 1)?
                .coefficients
                .dc_low_pass;
            current[4] = checked_add(current[4], top[4], "YUV422 top LP prediction")?;
            current[1] = checked_add(current[1], top[5], "YUV422 top LP prediction")?;
            current[5] = checked_add(current[5], current[1], "YUV422 internal LP prediction")?;
        }
        (ChromaSampling::Cs422, PredictionMode::None) if dc_mode == PredictionMode::FromTop => {
            current[5] = checked_add(current[5], current[1], "YUV422 internal LP prediction")?;
        }
        _ => {}
    }
    Ok(())
}

fn read_code<const N: usize>(
    reader: &mut PacketBitReader<'_>,
    table: &[(u16, u8, u8); N],
    syntax: &'static str,
) -> Result<u8, TileDecodeError> {
    let start = reader.bit_position();
    let max = table.iter().map(|entry| entry.1).max().unwrap_or(0);
    let mut bits = 0_u16;
    for length in 1..=max {
        bits = (bits << 1) | u16::from(reader.read_bit()?);
        if let Some(entry) = table
            .iter()
            .find(|entry| entry.1 == length && entry.0 == bits)
        {
            return Ok(entry.2);
        }
    }
    Err(crate::entropy::EntropyError::InvalidVlc {
        syntax,
        bit_position: start,
    }
    .into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_length_cbplp_does_not_apply_vlc_inversion() {
        let mut reader = PacketBitReader::with_bit_length(&[0b1110_0000], 3).unwrap();
        let mut state = CbplpState {
            sampling: ChromaSampling::Cs444,
            zero: 4,
            maximum: 0,
        };

        assert_eq!(state.decode(&mut reader).unwrap(), 7);
        assert_eq!((state.zero, state.maximum), (5, -3));
        assert_eq!(reader.bits_remaining(), 0);
    }

    #[test]
    fn subsampled_chroma_refinement_interleaves_u_and_v_per_position() {
        let mut reader = PacketBitReader::with_bit_length(&[0b0011_0000], 4).unwrap();
        let mut low = [[0_i32; 16]; 3];
        for component in &mut low[1..] {
            component[1] = 1;
            component[2] = 1;
        }

        decode_subsampled_chroma_refinement(&mut reader, &mut low, &[2, 1], 1).unwrap();

        assert_eq!((low[1][2], low[2][2]), (2, 2));
        assert_eq!((low[1][1], low[2][1]), (3, 3));
        assert_eq!(reader.bits_remaining(), 0);
    }
}
