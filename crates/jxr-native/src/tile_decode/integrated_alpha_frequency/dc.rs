//! Integrated primary/alpha DC frequency packet.

use jxr_core::{PredictionMode, QuantizerSet};

use crate::{
    entropy::{PacketBitReader, TileEntropyState},
    reconstruct::QuantizedMacroblock,
};

use super::{PlaneState, SpatialMacroblock, TileDecodeError};
use crate::tile_decode::{
    multicomponent, packet_slice,
    quantizer::TileQuantizers,
    spatial::{MacroblockPosition, consume_byte_alignment, decode_dc_band, parse_packet_prefix},
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
    primary.quantizers = Some(TileQuantizers::parse_dc_packet(
        &mut reader,
        primary.header,
    )?);
    alpha.quantizers = Some(TileQuantizers::parse_dc_packet(&mut reader, alpha.header)?);
    let mut primary_entropy = TileEntropyState::new();
    let mut alpha_entropy = TileEntropyState::new();
    primary_entropy.reset_tile();
    alpha_entropy.reset_tile();
    for y in 0..height {
        for x in 0..width {
            let position = MacroblockPosition { width, x, y };
            decode_macroblock(&mut reader, primary, &mut primary_entropy, position)?;
            decode_macroblock(&mut reader, alpha, &mut alpha_entropy, position)?;
            if x + 1 == width || x.is_multiple_of(16) {
                primary_entropy.dc_vlc.adapt();
                alpha_entropy.dc_vlc.adapt();
            }
        }
    }
    consume_byte_alignment(&mut reader)
}

fn decode_macroblock(
    reader: &mut PacketBitReader<'_>,
    state: &mut PlaneState<'_>,
    entropy: &mut TileEntropyState,
    position: MacroblockPosition,
) -> Result<(), TileDecodeError> {
    match state.header.internal_color_format {
        0 => decode_luma(reader, state, entropy, position),
        1..=3 => decode_yuv(reader, state, entropy, position),
        4 | 6 => decode_multi(reader, state, entropy, position),
        _ => Err(TileDecodeError::Unsupported(
            "integrated alpha frequency DC primary format",
        )),
    }
}

fn decode_luma(
    reader: &mut PacketBitReader<'_>,
    state: &mut PlaneState<'_>,
    entropy: &mut TileEntropyState,
    position: MacroblockPosition,
) -> Result<(), TileDecodeError> {
    let prediction = decode_dc_band(reader, entropy, &state.components[0], position)?;
    let mut low = [0_i32; 16];
    low[0] = prediction.value;
    state.components[0].push(macroblock(state.bands, low, prediction.mode));
    Ok(())
}

fn decode_yuv(
    reader: &mut PacketBitReader<'_>,
    state: &mut PlaneState<'_>,
    entropy: &mut TileEntropyState,
    position: MacroblockPosition,
) -> Result<(), TileDecodeError> {
    let sampling = yuv::sampling(state.header.internal_color_format)?;
    let components: &mut [Vec<SpatialMacroblock>; 3] =
        state
            .components
            .as_mut_slice()
            .try_into()
            .map_err(|_| TileDecodeError::InvalidPlan("integrated YUV component count"))?;
    let dc = yuv::syntax::decode_dc(reader, entropy, components, position, sampling)?;
    for (plane, (value, prediction)) in components.iter_mut().zip(dc) {
        let mut low = [0_i32; 16];
        low[0] = value;
        plane.push(macroblock(state.bands, low, prediction));
    }
    Ok(())
}

fn decode_multi(
    reader: &mut PacketBitReader<'_>,
    state: &mut PlaneState<'_>,
    entropy: &mut TileEntropyState,
    position: MacroblockPosition,
) -> Result<(), TileDecodeError> {
    let (dc, prediction) =
        multicomponent::decode_dc(reader, entropy, &state.components, position, state.header)?;
    for (plane, value) in state.components.iter_mut().zip(dc) {
        let mut low = [0_i32; 16];
        low[0] = value;
        plane.push(macroblock(state.bands, low, prediction));
    }
    Ok(())
}

fn macroblock(
    bands: jxr_core::BandPresence,
    dc_low_pass: [i32; 16],
    prediction: PredictionMode,
) -> SpatialMacroblock {
    SpatialMacroblock {
        coefficients: QuantizedMacroblock {
            dc_low_pass,
            high_pass: [0_i32; 256],
            quantizers: QuantizerSet {
                dc: 1,
                low_pass: 1,
                high_pass: 1,
            },
            bands,
        },
        prediction,
        hp_prediction: PredictionMode::None,
        lp_qp_index: 0,
    }
}
