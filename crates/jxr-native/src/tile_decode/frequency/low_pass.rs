use crate::{
    ImagePlaneHeader,
    entropy::{PacketBitReader, TileEntropyState},
};

use super::super::{
    TileDecodeError,
    quantizer::TileQuantizers,
    spatial::{
        MacroblockPosition, SpatialMacroblock, consume_byte_alignment, decode_low_pass,
        parse_packet_prefix, predict_low_pass,
    },
};

pub(super) fn decode(
    packet: &[u8],
    plane: &ImagePlaneHeader,
    quantizers: &mut TileQuantizers,
    decoded: &mut [SpatialMacroblock],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, TileDecodeError> {
    let mut reader = PacketBitReader::new(packet);
    parse_packet_prefix(&mut reader)?;
    quantizers.parse_low_pass_packet(&mut reader, plane)?;
    let mut indices = Vec::with_capacity(decoded.len());
    let mut entropy = TileEntropyState::new();
    entropy.reset_tile();
    for y in 0..height {
        for x in 0..width {
            if x.is_multiple_of(16) {
                entropy.reset_scan_totals();
            }
            let index = y * width + x;
            let qp_index = quantizers.low_pass_index(&mut reader)?;
            let mut low = decoded[index].coefficients.dc_low_pass;
            decode_low_pass(&mut reader, &mut entropy, &mut low)?;
            predict_low_pass(
                &mut low,
                decoded,
                MacroblockPosition { width, x, y },
                decoded[index].prediction,
                qp_index,
            )?;
            decoded[index].coefficients.dc_low_pass = low;
            decoded[index].lp_qp_index = qp_index;
            indices.push(qp_index);
            if x + 1 == width || x.is_multiple_of(16) {
                entropy.lp_vlc.adapt();
            }
        }
    }
    consume_byte_alignment(&mut reader)?;
    Ok(indices)
}
