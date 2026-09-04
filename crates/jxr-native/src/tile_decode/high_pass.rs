//! Shared HP parsing and prediction-mode selection for tile packets.

use jxr_core::{ChromaSampling, PredictionMode};

use crate::entropy::{
    ColourModel, ComponentClass, FrequencyBand, HpScanDirection, PacketBitReader, TileEntropyState,
    decode_ac_block, decode_flex_block,
};

use super::{TileDecodeError, cbphp::CbphpState, spatial::MacroblockPosition};

pub(super) const HIERARCHICAL_BLOCK_ORDER: [usize; 16] =
    [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];

pub(super) fn prediction_mode(low: &[i32; 16]) -> PredictionMode {
    let horizontal = low[1]
        .unsigned_abs()
        .saturating_add(low[2].unsigned_abs())
        .saturating_add(low[3].unsigned_abs());
    let vertical = low[4]
        .unsigned_abs()
        .saturating_add(low[8].unsigned_abs())
        .saturating_add(low[12].unsigned_abs());
    select_prediction(horizontal, vertical)
}

/// Select the one macroblock HP direction shared by all YUV components.
pub(super) fn prediction_mode_yuv(
    low: &[[i32; 16]; 3],
    sampling: ChromaSampling,
) -> PredictionMode {
    let mut horizontal = low[0][1]
        .unsigned_abs()
        .saturating_add(low[0][2].unsigned_abs())
        .saturating_add(low[0][3].unsigned_abs());
    let mut vertical = low[0][4]
        .unsigned_abs()
        .saturating_add(low[0][8].unsigned_abs())
        .saturating_add(low[0][12].unsigned_abs());
    for chroma in &low[1..] {
        horizontal = horizontal.saturating_add(chroma[1].unsigned_abs());
        match sampling {
            ChromaSampling::Cs420 => {
                vertical = vertical.saturating_add(chroma[2].unsigned_abs());
            }
            ChromaSampling::Cs422 => {
                horizontal = horizontal.saturating_add(chroma[5].unsigned_abs());
                vertical = vertical
                    .saturating_add(chroma[2].unsigned_abs())
                    .saturating_add(chroma[6].unsigned_abs());
            }
            ChromaSampling::Cs444 => {
                vertical = vertical.saturating_add(chroma[4].unsigned_abs());
            }
        }
    }
    select_prediction(horizontal, vertical)
}

pub(super) fn prediction_mode_yuvk(low: &[[i32; 16]]) -> Result<PredictionMode, TileDecodeError> {
    let yuv: [[i32; 16]; 3] = low
        .get(..3)
        .ok_or(TileDecodeError::InvalidPlan("YUVK prediction components"))?
        .try_into()
        .map_err(|_| TileDecodeError::InvalidPlan("YUVK prediction components"))?;
    Ok(prediction_mode_yuv(&yuv, ChromaSampling::Cs444))
}

const fn select_prediction(horizontal: u32, vertical: u32) -> PredictionMode {
    if horizontal.saturating_mul(4) < vertical {
        PredictionMode::FromLeft
    } else if vertical.saturating_mul(4) < horizontal {
        PredictionMode::FromTop
    } else {
        PredictionMode::None
    }
}

pub(super) struct DecodedHighVlc {
    pub(super) coefficients: [i32; 256],
    pub(super) model_bits: u8,
}

pub(super) struct DecodedHighComponents {
    pub(super) coefficients: Box<[[i32; 256]; 3]>,
    pub(super) model_bits: [u8; 2],
}

pub(super) struct DecodedHighMulti {
    pub(super) coefficients: Vec<[i32; 256]>,
    pub(super) model_bits: [u8; 2],
}

#[derive(Debug, Clone, Copy)]
pub(super) enum HighpassPayload {
    VlcOnly,
    Combined {
        flexbits_present: bool,
        trim_flexbits: u8,
    },
}

pub(super) fn decode_vlc(
    reader: &mut PacketBitReader<'_>,
    entropy: &mut TileEntropyState,
    cbphp: &mut CbphpState,
    position: MacroblockPosition,
    mode: PredictionMode,
    payload: HighpassPayload,
) -> Result<DecodedHighVlc, TileDecodeError> {
    let mut output = [0_i32; 256];
    let mut coded_blocks = cbphp.decode(reader, position.x, position.y)?;
    let model_bits = entropy.hp_model.bits(false);
    let direction = if mode == PredictionMode::FromTop {
        HpScanDirection::Vertical
    } else {
        HpScanDirection::Horizontal
    };
    let mut non_zero = 0_i32;
    for block_map in HIERARCHICAL_BLOCK_ORDER {
        let mut vlc = [0_i32; 16];
        if coded_blocks & 1 != 0 {
            let block = decode_ac_block(
                reader,
                FrequencyBand::Highpass,
                ComponentClass::Luma,
                1,
                &mut entropy.hp_vlc,
            )?;
            non_zero += i32::from(block.non_zero_count());
            block.inverse_scan_hp(&mut entropy.hp_scan, direction, &mut vlc)?;
        }
        finish_block(
            reader,
            &mut output[block_map * 16..(block_map + 1) * 16],
            &vlc,
            model_bits,
            payload,
        )?;
        coded_blocks >>= 1;
    }
    entropy
        .hp_model
        .update([non_zero, 0], ColourModel::LumaOnly)?;
    Ok(DecodedHighVlc {
        coefficients: output,
        model_bits,
    })
}

pub(super) fn decode_yuv(
    reader: &mut PacketBitReader<'_>,
    entropy: &mut TileEntropyState,
    cbphp: &mut CbphpState,
    position: MacroblockPosition,
    sampling: ChromaSampling,
    mode: PredictionMode,
    payload: HighpassPayload,
) -> Result<DecodedHighComponents, TileDecodeError> {
    let mut output = Box::new([[0_i32; 256]; 3]);
    let mut coded_blocks = cbphp.decode_yuv(reader, position.x, position.y, sampling)?;
    let model_bits = [entropy.hp_model.bits(false), entropy.hp_model.bits(true)];
    let mut lap_mean = [0_i32; 2];
    for component in 0..3 {
        let class_index = usize::from(component != 0);
        let class = if component == 0 {
            ComponentClass::Luma
        } else {
            ComponentClass::Chroma
        };
        let direction = if mode == PredictionMode::FromTop {
            HpScanDirection::Vertical
        } else {
            HpScanDirection::Horizontal
        };
        let block_count = match (component, sampling) {
            (0, _) | (_, ChromaSampling::Cs444) => 16,
            (_, ChromaSampling::Cs422) => 8,
            (_, ChromaSampling::Cs420) => 4,
        };
        for (block, &hierarchical_block) in HIERARCHICAL_BLOCK_ORDER
            .iter()
            .enumerate()
            .take(block_count)
        {
            let block_map = if block_count == 16 {
                hierarchical_block
            } else {
                block
            };
            let mut vlc = [0_i32; 16];
            if coded_blocks[component] & 1 != 0 {
                let block = decode_ac_block(
                    reader,
                    FrequencyBand::Highpass,
                    class,
                    1,
                    &mut entropy.hp_vlc,
                )?;
                lap_mean[class_index] += i32::from(block.non_zero_count());
                block.inverse_scan_hp(&mut entropy.hp_scan, direction, &mut vlc)?;
            }
            finish_block(
                reader,
                &mut output[component][block_map * 16..(block_map + 1) * 16],
                &vlc,
                model_bits[class_index],
                payload,
            )?;
            coded_blocks[component] >>= 1;
        }
    }
    let colour = match sampling {
        ChromaSampling::Cs420 => ColourModel::Yuv420,
        ChromaSampling::Cs422 => ColourModel::Yuv422,
        ChromaSampling::Cs444 => ColourModel::Other { components: 3 },
    };
    entropy.hp_model.update(lap_mean, colour)?;
    Ok(DecodedHighComponents {
        coefficients: output,
        model_bits,
    })
}

pub(super) fn decode_components(
    reader: &mut PacketBitReader<'_>,
    entropy: &mut TileEntropyState,
    cbphp: &mut CbphpState,
    position: MacroblockPosition,
    components: usize,
    mode: PredictionMode,
    payload: HighpassPayload,
) -> Result<DecodedHighMulti, TileDecodeError> {
    let component_count = u8::try_from(components)
        .ok()
        .filter(|count| (2..=16).contains(count))
        .ok_or(TileDecodeError::InvalidPlan(
            "multi-component HP component count",
        ))?;
    let model_bits = [entropy.hp_model.bits(false), entropy.hp_model.bits(true)];
    let direction = if mode == PredictionMode::FromTop {
        HpScanDirection::Vertical
    } else {
        HpScanDirection::Horizontal
    };
    let coded_blocks = (0..components)
        .map(|component| cbphp.decode_component(reader, position.x, position.y, component))
        .collect::<Result<Vec<_>, _>>()?;
    let mut output = Vec::with_capacity(components);
    let mut lap_mean = [0_i32; 2];
    for (component, mut coded_blocks) in coded_blocks.into_iter().enumerate() {
        let class_index = usize::from(component != 0);
        let class = if component == 0 {
            ComponentClass::Luma
        } else {
            ComponentClass::Chroma
        };
        let mut coefficients = [0_i32; 256];
        for block_map in HIERARCHICAL_BLOCK_ORDER {
            let mut vlc = [0_i32; 16];
            if coded_blocks & 1 != 0 {
                let block = decode_ac_block(
                    reader,
                    FrequencyBand::Highpass,
                    class,
                    1,
                    &mut entropy.hp_vlc,
                )?;
                lap_mean[class_index] += i32::from(block.non_zero_count());
                block.inverse_scan_hp(&mut entropy.hp_scan, direction, &mut vlc)?;
            }
            finish_block(
                reader,
                &mut coefficients[block_map * 16..(block_map + 1) * 16],
                &vlc,
                model_bits[class_index],
                payload,
            )?;
            coded_blocks >>= 1;
        }
        output.push(coefficients);
    }
    entropy.hp_model.update(
        lap_mean,
        ColourModel::Other {
            components: component_count,
        },
    )?;
    Ok(DecodedHighMulti {
        coefficients: output,
        model_bits,
    })
}

fn finish_block(
    reader: &mut PacketBitReader<'_>,
    destination: &mut [i32],
    vlc: &[i32; 16],
    model_bits: u8,
    payload: HighpassPayload,
) -> Result<(), TileDecodeError> {
    let flex = match payload {
        HighpassPayload::VlcOnly => None,
        HighpassPayload::Combined {
            flexbits_present,
            trim_flexbits,
        } => Some(if flexbits_present {
            decode_flex_block(reader, vlc, model_bits, trim_flexbits)?
        } else {
            [0; 16]
        }),
    };
    for index in 1..16 {
        destination[index] = if let Some(flex) = flex {
            vlc[index]
                .checked_shl(u32::from(model_bits))
                .and_then(|value| value.checked_add(flex[index]))
                .ok_or(TileDecodeError::ArithmeticOverflow("HP refinement"))?
        } else {
            vlc[index]
        };
    }
    Ok(())
}
