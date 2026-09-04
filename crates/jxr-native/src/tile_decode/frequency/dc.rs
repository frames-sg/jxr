use jxr_core::{BandPresence, PredictionMode, QuantizerSet};

use crate::{
    ImagePlaneHeader,
    entropy::{PacketBitReader, TileEntropyState},
    reconstruct::QuantizedMacroblock,
};

use super::super::{
    TileDecodeError,
    quantizer::TileQuantizers,
    spatial::{
        MacroblockPosition, SpatialMacroblock, consume_byte_alignment, decode_dc_band,
        parse_packet_prefix,
    },
};

pub(super) fn decode(
    packet: &[u8],
    plane: &ImagePlaneHeader,
    bands: BandPresence,
    width: usize,
    height: usize,
) -> Result<(Vec<SpatialMacroblock>, TileQuantizers), TileDecodeError> {
    let mut reader = PacketBitReader::new(packet);
    parse_packet_prefix(&mut reader)?;
    let quantizers = TileQuantizers::parse_dc_packet(&mut reader, plane)?;
    let count = width
        .checked_mul(height)
        .ok_or(TileDecodeError::ArithmeticOverflow(
            "frequency DC macroblock count",
        ))?;
    let mut decoded = Vec::with_capacity(count);
    let mut entropy = TileEntropyState::new();
    entropy.reset_tile();
    for y in 0..height {
        for x in 0..width {
            let prediction = decode_dc_band(
                &mut reader,
                &mut entropy,
                &decoded,
                MacroblockPosition { width, x, y },
            )?;
            let mut low = [0_i32; 16];
            low[0] = prediction.value;
            decoded.push(SpatialMacroblock {
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
                prediction: prediction.mode,
                hp_prediction: PredictionMode::None,
                lp_qp_index: 0,
            });
            if x + 1 == width || x.is_multiple_of(16) {
                entropy.dc_vlc.adapt();
            }
        }
    }
    consume_byte_alignment(&mut reader)?;
    Ok((decoded, quantizers))
}

pub(super) fn finalize_dc_only(
    decoded: &mut [SpatialMacroblock],
    quantizers: &TileQuantizers,
    scaled: bool,
) -> Result<(), TileDecodeError> {
    for macroblock in decoded {
        macroblock.coefficients.quantizers = quantizers.reconstruction_steps(
            super::super::quantizer::QuantizerIndices { lp: 0, hp: 0 },
            scaled,
        )?;
    }
    Ok(())
}
