//! Integrated primary/alpha low-pass frequency packet.

use crate::entropy::{PacketBitReader, TileEntropyState};

use super::{PlaneState, SpatialMacroblock, TileDecodeError};
use crate::tile_decode::{
    multicomponent, packet_slice,
    spatial::{
        MacroblockPosition, consume_byte_alignment, decode_low_pass, parse_packet_prefix,
        predict_low_pass,
    },
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
    let mut primary_pattern = yuv_pattern(primary)?;
    let mut alpha_pattern = yuv_pattern(alpha)?;
    for y in 0..height {
        for x in 0..width {
            let position = MacroblockPosition { width, x, y };
            decode_if_present(
                &mut reader,
                primary,
                &mut primary_entropy,
                primary_pattern.as_mut(),
                position,
            )?;
            decode_if_present(
                &mut reader,
                alpha,
                &mut alpha_entropy,
                alpha_pattern.as_mut(),
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
    if state.bands.has_low_pass() {
        let header = state.header;
        state
            .quantizers_mut()?
            .parse_low_pass_packet(reader, header)?;
    }
    Ok(())
}

fn yuv_pattern(state: &PlaneState<'_>) -> Result<Option<yuv::syntax::CbplpState>, TileDecodeError> {
    if matches!(state.header.internal_color_format, 1..=3) && state.bands.has_low_pass() {
        Ok(Some(yuv::syntax::CbplpState::new(yuv::sampling(
            state.header.internal_color_format,
        )?)))
    } else {
        Ok(None)
    }
}

fn decode_if_present(
    reader: &mut PacketBitReader<'_>,
    state: &mut PlaneState<'_>,
    entropy: &mut TileEntropyState,
    pattern: Option<&mut yuv::syntax::CbplpState>,
    position: MacroblockPosition,
) -> Result<(), TileDecodeError> {
    if !state.bands.has_low_pass() {
        return Ok(());
    }
    if position.x.is_multiple_of(16) {
        entropy.reset_scan_totals();
    }
    let qp = state.quantizers()?.low_pass_index(reader)?;
    match state.header.internal_color_format {
        0 => decode_luma(reader, state, entropy, position, qp)?,
        1..=3 => decode_yuv(
            reader,
            state,
            entropy,
            pattern.ok_or(TileDecodeError::InvalidPlan("YUV LP pattern state"))?,
            position,
            qp,
        )?,
        4 | 6 => decode_multi(reader, state, entropy, position, qp)?,
        _ => {
            return Err(TileDecodeError::Unsupported(
                "integrated alpha frequency LP primary format",
            ));
        }
    }
    state.low_pass_indices.push(qp);
    if position.x + 1 == position.width || position.x.is_multiple_of(16) {
        entropy.lp_vlc.adapt();
    }
    Ok(())
}

fn decode_luma(
    reader: &mut PacketBitReader<'_>,
    state: &mut PlaneState<'_>,
    entropy: &mut TileEntropyState,
    position: MacroblockPosition,
    qp: u8,
) -> Result<(), TileDecodeError> {
    let index = position.y * position.width + position.x;
    let mut low = state.components[0][index].coefficients.dc_low_pass;
    decode_low_pass(reader, entropy, &mut low)?;
    let prediction = state.components[0][index].prediction;
    predict_low_pass(&mut low, &state.components[0], position, prediction, qp)?;
    state.components[0][index].coefficients.dc_low_pass = low;
    state.components[0][index].lp_qp_index = qp;
    Ok(())
}

fn decode_yuv(
    reader: &mut PacketBitReader<'_>,
    state: &mut PlaneState<'_>,
    entropy: &mut TileEntropyState,
    pattern: &mut yuv::syntax::CbplpState,
    position: MacroblockPosition,
    qp: u8,
) -> Result<(), TileDecodeError> {
    let sampling = yuv::sampling(state.header.internal_color_format)?;
    let index = position.y * position.width + position.x;
    let components: &mut [Vec<SpatialMacroblock>; 3] =
        state
            .components
            .as_mut_slice()
            .try_into()
            .map_err(|_| TileDecodeError::InvalidPlan("integrated YUV component count"))?;
    let mut low =
        core::array::from_fn(|component| components[component][index].coefficients.dc_low_pass);
    let predictions = core::array::from_fn(|component| components[component][index].prediction);
    let context = yuv::syntax::LowPassContext {
        decoded: components,
        position,
        qp_index: qp,
        predictions,
    };
    if sampling == jxr_core::ChromaSampling::Cs444 {
        yuv::syntax::decode_low_pass_444(reader, entropy, pattern, &mut low, context)?;
    } else {
        yuv::syntax::decode_low_pass_subsampled(
            reader, entropy, pattern, &mut low, context, sampling,
        )?;
    }
    for (component, coefficients) in components.iter_mut().zip(low) {
        component[index].coefficients.dc_low_pass = coefficients;
        component[index].lp_qp_index = qp;
    }
    Ok(())
}

fn decode_multi(
    reader: &mut PacketBitReader<'_>,
    state: &mut PlaneState<'_>,
    entropy: &mut TileEntropyState,
    position: MacroblockPosition,
    qp: u8,
) -> Result<(), TileDecodeError> {
    let index = position.y * position.width + position.x;
    let mut low: Vec<[i32; 16]> = state
        .components
        .iter()
        .map(|component| component[index].coefficients.dc_low_pass)
        .collect();
    let prediction = state.components[0][index].prediction;
    multicomponent::decode_low_pass(
        reader,
        entropy,
        &state.components,
        position,
        qp,
        prediction,
        &mut low,
    )?;
    for (component, coefficients) in state.components.iter_mut().zip(low) {
        component[index].coefficients.dc_low_pass = coefficients;
        component[index].lp_qp_index = qp;
    }
    Ok(())
}
