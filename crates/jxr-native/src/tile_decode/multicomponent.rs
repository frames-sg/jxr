//! Spatial-mode YUVK and Main-profile N-component tile traversal.

use jxr_core::{BandPresence, PredictionMode};

use crate::{
    ImagePlaneHeader, ParsedCodestream,
    entropy::{
        ColourModel, ComponentClass, FrequencyBand, PacketBitReader, TileEntropyState,
        decode_ac_block, decode_dc_coefficient, decode_lp_refinement,
    },
    reconstruct::QuantizedMacroblock,
};

use super::{
    DecodedTile, TileDecodeError,
    cbphp::CbphpState,
    high_pass::{self, HighpassPayload},
    quantizer::TileQuantizers,
    spatial::{
        MacroblockPosition, SpatialMacroblock, checked_add, consume_byte_alignment,
        parse_packet_prefix, predict_low_pass, read_u8, spatial_macroblock,
    },
};

pub(super) fn decode_spatial_tile(
    packet: &[u8],
    parsed: &ParsedCodestream,
    bands: BandPresence,
    tile_width: u32,
    tile_height: u32,
) -> Result<DecodedTile, TileDecodeError> {
    let plane = &parsed.headers.primary;
    let (width, height, count) = tile_geometry(tile_width, tile_height)?;
    let mut reader = PacketBitReader::new(packet);
    parse_packet_prefix(&mut reader)?;
    let trim = if parsed.headers.image.flags.trim_flexbits() {
        read_u8(&mut reader, 4)?
    } else {
        0
    };
    let mut decoder = SpatialDecoder::new(
        TileQuantizers::parse(&mut reader, plane)?,
        plane,
        bands,
        trim,
        width,
        count,
    )?;
    for y in 0..height {
        for x in 0..width {
            decoder.decode_macroblock(&mut reader, MacroblockPosition { width, x, y })?;
        }
    }
    consume_byte_alignment(&mut reader)?;
    Ok(decoder.finish())
}

struct TileSyntax<'a> {
    quantizers: TileQuantizers,
    plane: &'a ImagePlaneHeader,
    bands: BandPresence,
    trim: u8,
}

pub(super) struct SpatialDecoder<'a> {
    syntax: TileSyntax<'a>,
    decoded: Vec<Vec<SpatialMacroblock>>,
    entropy: TileEntropyState,
    cbphp: CbphpState,
}

impl<'a> SpatialDecoder<'a> {
    pub(super) fn new(
        quantizers: TileQuantizers,
        plane: &'a ImagePlaneHeader,
        bands: BandPresence,
        trim: u8,
        width: usize,
        capacity: usize,
    ) -> Result<Self, TileDecodeError> {
        let components = validate_plane(plane)?;
        let mut entropy = TileEntropyState::new();
        entropy.reset_tile();
        Ok(Self {
            syntax: TileSyntax {
                quantizers,
                plane,
                bands,
                trim,
            },
            decoded: (0..components)
                .map(|_| Vec::with_capacity(capacity))
                .collect(),
            entropy,
            cbphp: CbphpState::new_components(width, components),
        })
    }

    pub(super) fn decode_macroblock(
        &mut self,
        reader: &mut PacketBitReader<'_>,
        position: MacroblockPosition,
    ) -> Result<(), TileDecodeError> {
        if position.x.is_multiple_of(16) {
            self.entropy.reset_scan_totals();
        }
        decode_macroblock(
            reader,
            &mut self.entropy,
            &mut self.cbphp,
            &self.syntax,
            &mut self.decoded,
            position,
        )?;
        if position.x + 1 == position.width || position.x.is_multiple_of(16) {
            self.entropy.dc_vlc.adapt();
            self.entropy.lp_vlc.adapt();
            self.entropy.hp_vlc.adapt();
            self.cbphp.adapt();
        }
        Ok(())
    }

    pub(super) fn finish(self) -> DecodedTile {
        DecodedTile {
            components: self.decoded,
        }
    }
}

fn decode_macroblock(
    reader: &mut PacketBitReader<'_>,
    entropy: &mut TileEntropyState,
    cbphp: &mut CbphpState,
    syntax: &TileSyntax<'_>,
    decoded: &mut [Vec<SpatialMacroblock>],
    position: MacroblockPosition,
) -> Result<(), TileDecodeError> {
    let qp_indices = syntax.quantizers.indices(reader)?;
    let (dc, dc_mode) = decode_dc(reader, entropy, decoded, position, syntax.plane)?;
    let mut low: Vec<[i32; 16]> = dc
        .into_iter()
        .map(|value| {
            let mut coefficients = [0_i32; 16];
            coefficients[0] = value;
            coefficients
        })
        .collect();
    if syntax.bands.has_low_pass() {
        decode_low_pass(
            reader,
            entropy,
            decoded,
            position,
            qp_indices.lp,
            dc_mode,
            &mut low,
        )?;
    }
    let hp_mode = if syntax.plane.internal_color_format == 4 {
        high_pass::prediction_mode_yuvk(&low)?
    } else {
        high_pass::prediction_mode(&low[0])
    };
    let high = if syntax.bands.has_high_pass() {
        high_pass::decode_components(
            reader,
            entropy,
            cbphp,
            position,
            decoded.len(),
            hp_mode,
            HighpassPayload::Combined {
                flexbits_present: syntax.bands.has_flexbits(),
                trim_flexbits: syntax.trim,
            },
        )?
        .coefficients
    } else {
        vec![[0_i32; 256]; decoded.len()]
    };
    for component in 0..decoded.len() {
        decoded[component].push(SpatialMacroblock {
            coefficients: QuantizedMacroblock {
                dc_low_pass: low[component],
                high_pass: high[component],
                quantizers: syntax.quantizers.reconstruction_steps_for(
                    component,
                    qp_indices,
                    syntax.plane.scaled,
                )?,
                bands: syntax.bands,
            },
            prediction: dc_mode,
            hp_prediction: hp_mode,
            lp_qp_index: qp_indices.lp,
        });
    }
    Ok(())
}

pub(super) fn decode_dc(
    reader: &mut PacketBitReader<'_>,
    entropy: &mut TileEntropyState,
    decoded: &[Vec<SpatialMacroblock>],
    position: MacroblockPosition,
    plane: &ImagePlaneHeader,
) -> Result<(Vec<i32>, PredictionMode), TileDecodeError> {
    let mut residuals = Vec::with_capacity(decoded.len());
    let mut lap_mean = [0_i32; 2];
    for component in 0..decoded.len() {
        let class = usize::from(component != 0);
        let present = reader.read_bit()?;
        lap_mean[class] += i32::from(present);
        residuals.push(decode_dc_coefficient(
            reader,
            entropy.dc_model.bits(component != 0),
            present,
            ComponentClass::Luma,
            &mut entropy.dc_vlc,
        )?);
    }
    entropy.dc_model.update(
        lap_mean,
        ColourModel::Other {
            components: component_count(decoded.len())?,
        },
    )?;
    let mode = dc_prediction_mode(decoded, position, plane.internal_color_format == 4)?;
    for (component, residual) in residuals.iter_mut().enumerate() {
        *residual = predict_dc_value(*residual, &decoded[component], position, mode)?;
    }
    Ok((residuals, mode))
}

pub(super) fn decode_low_pass(
    reader: &mut PacketBitReader<'_>,
    entropy: &mut TileEntropyState,
    decoded: &[Vec<SpatialMacroblock>],
    position: MacroblockPosition,
    qp_index: u8,
    dc_mode: PredictionMode,
    low: &mut [[i32; 16]],
) -> Result<(), TileDecodeError> {
    let mut pattern = Vec::with_capacity(low.len());
    for _ in 0..low.len() {
        pattern.push(reader.read_bit()?);
    }
    let mut lap_mean = [0_i32; 2];
    for (component, coefficients) in low.iter_mut().enumerate() {
        let class_index = usize::from(component != 0);
        if pattern[component] {
            let block = decode_ac_block(
                reader,
                FrequencyBand::Lowpass,
                if component == 0 {
                    ComponentClass::Luma
                } else {
                    ComponentClass::Chroma
                },
                1,
                &mut entropy.lp_vlc,
            )?;
            lap_mean[class_index] += i32::from(block.non_zero_count());
            block.inverse_scan_lp(&mut entropy.lp_scan, coefficients)?;
        }
        decode_lp_refinement(reader, coefficients, entropy.lp_model.bits(component != 0))?;
        predict_low_pass(
            coefficients,
            &decoded[component],
            position,
            dc_mode,
            qp_index,
        )?;
    }
    entropy.lp_model.update(
        lap_mean,
        ColourModel::Other {
            components: component_count(low.len())?,
        },
    )?;
    Ok(())
}

fn dc_prediction_mode(
    decoded: &[Vec<SpatialMacroblock>],
    position: MacroblockPosition,
    yuvk: bool,
) -> Result<PredictionMode, TileDecodeError> {
    let MacroblockPosition { width, x, y } = position;
    if x == 0 {
        return Ok(if y == 0 {
            PredictionMode::None
        } else {
            PredictionMode::FromTop
        });
    }
    if y == 0 {
        return Ok(PredictionMode::FromLeft);
    }
    let components = if yuvk { 3 } else { 1 };
    let mut horizontal = 0_u32;
    let mut vertical = 0_u32;
    for (component, plane) in decoded.iter().take(components).enumerate() {
        let weight = if yuvk && component == 0 { 2 } else { 1 };
        let left = dc_at(plane, width, x - 1, y)?;
        let top = dc_at(plane, width, x, y - 1)?;
        let top_left = dc_at(plane, width, x - 1, y - 1)?;
        horizontal = horizontal.saturating_add(top_left.abs_diff(left).saturating_mul(weight));
        vertical = vertical.saturating_add(top_left.abs_diff(top).saturating_mul(weight));
    }
    Ok(if horizontal.saturating_mul(4) < vertical {
        PredictionMode::FromTop
    } else if vertical.saturating_mul(4) < horizontal {
        PredictionMode::FromLeft
    } else {
        PredictionMode::FromTopLeft
    })
}

fn predict_dc_value(
    residual: i32,
    decoded: &[SpatialMacroblock],
    position: MacroblockPosition,
    mode: PredictionMode,
) -> Result<i32, TileDecodeError> {
    let MacroblockPosition { width, x, y } = position;
    match mode {
        PredictionMode::None => Ok(residual),
        PredictionMode::FromLeft => checked_add(
            residual,
            dc_at(decoded, width, x - 1, y)?,
            "multi-component left DC prediction",
        ),
        PredictionMode::FromTop => checked_add(
            residual,
            dc_at(decoded, width, x, y - 1)?,
            "multi-component top DC prediction",
        ),
        PredictionMode::FromTopLeft => {
            let left = i64::from(dc_at(decoded, width, x - 1, y)?);
            let top = i64::from(dc_at(decoded, width, x, y - 1)?);
            let average = i32::try_from((left + top) >> 1)
                .map_err(|_| TileDecodeError::ArithmeticOverflow("multi-component DC average"))?;
            checked_add(residual, average, "multi-component averaged DC prediction")
        }
    }
}

fn dc_at(
    decoded: &[SpatialMacroblock],
    width: usize,
    x: usize,
    y: usize,
) -> Result<i32, TileDecodeError> {
    Ok(spatial_macroblock(decoded, width, x, y)?
        .coefficients
        .dc_low_pass[0])
}

fn validate_plane(plane: &ImagePlaneHeader) -> Result<usize, TileDecodeError> {
    let components = usize::from(plane.components);
    if (plane.internal_color_format == 4 && components == 4)
        || (plane.internal_color_format == 6 && (2..=16).contains(&components))
    {
        Ok(components)
    } else {
        Err(TileDecodeError::Unsupported(
            "invalid YUVK or Main-profile N-component layout",
        ))
    }
}

fn tile_geometry(
    tile_width: u32,
    tile_height: u32,
) -> Result<(usize, usize, usize), TileDecodeError> {
    let width = usize::try_from(tile_width)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("multi-component tile width"))?;
    let height = usize::try_from(tile_height)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("multi-component tile height"))?;
    let count = width
        .checked_mul(height)
        .ok_or(TileDecodeError::ArithmeticOverflow(
            "multi-component tile macroblock count",
        ))?;
    Ok((width, height, count))
}

fn component_count(components: usize) -> Result<u8, TileDecodeError> {
    u8::try_from(components)
        .ok()
        .filter(|count| (2..=16).contains(count))
        .ok_or(TileDecodeError::InvalidPlan(
            "multi-component model component count",
        ))
}
