//! Spatial-mode YUV tile traversal for all sampling geometries.

use jxr_core::{BandPresence, ChromaSampling, PredictionMode};

use crate::{
    ImagePlaneHeader,
    entropy::{PacketBitReader, TileEntropyState},
    reconstruct::QuantizedMacroblock,
};

use super::super::{
    DecodedTile, TileDecodeError,
    cbphp::CbphpState,
    high_pass::{self, HighpassPayload},
    quantizer::TileQuantizers,
    spatial::{
        MacroblockPosition, SpatialMacroblock, consume_byte_alignment, parse_packet_prefix, read_u8,
    },
};
use super::syntax::{self, CbplpState};

pub(in crate::tile_decode) fn decode(
    packet: &[u8],
    plane: &ImagePlaneHeader,
    bands: BandPresence,
    tile_width: u32,
    tile_height: u32,
    trim_flexbits_present: bool,
    sampling: ChromaSampling,
) -> Result<DecodedTile, TileDecodeError> {
    let (width, height, count) = tile_geometry(tile_width, tile_height)?;
    let mut reader = PacketBitReader::new(packet);
    parse_packet_prefix(&mut reader)?;
    let trim = read_trim(&mut reader, trim_flexbits_present)?;
    let mut decoder = SpatialDecoder::new(
        TileQuantizers::parse(&mut reader, plane)?,
        plane,
        bands,
        trim,
        sampling,
        width,
        count,
    );
    for y in 0..height {
        for x in 0..width {
            decoder.decode_macroblock(&mut reader, MacroblockPosition { width, x, y })?;
        }
    }
    consume_byte_alignment(&mut reader)?;
    Ok(decoder.finish())
}

pub(in crate::tile_decode) struct SpatialDecoder<'a> {
    components: [Vec<SpatialMacroblock>; 3],
    entropy: TileEntropyState,
    low_pass_pattern: CbplpState,
    high_pass_pattern: CbphpState,
    quantizers: TileQuantizers,
    plane: &'a ImagePlaneHeader,
    bands: BandPresence,
    trim: u8,
    sampling: ChromaSampling,
}

impl<'a> SpatialDecoder<'a> {
    pub(in crate::tile_decode) fn new(
        quantizers: TileQuantizers,
        plane: &'a ImagePlaneHeader,
        bands: BandPresence,
        trim: u8,
        sampling: ChromaSampling,
        width: usize,
        capacity: usize,
    ) -> Self {
        let mut entropy = TileEntropyState::new();
        entropy.reset_tile();
        Self {
            components: core::array::from_fn(|_| Vec::with_capacity(capacity)),
            entropy,
            low_pass_pattern: CbplpState::new(sampling),
            high_pass_pattern: CbphpState::new(width),
            quantizers,
            plane,
            bands,
            trim,
            sampling,
        }
    }

    pub(in crate::tile_decode) fn decode_macroblock(
        &mut self,
        reader: &mut PacketBitReader<'_>,
        position: MacroblockPosition,
    ) -> Result<(), TileDecodeError> {
        if position.x.is_multiple_of(16) {
            self.entropy.reset_scan_totals();
        }
        let qp_indices = self.quantizers.indices(reader)?;
        let dc = syntax::decode_dc(
            reader,
            &mut self.entropy,
            &self.components,
            position,
            self.sampling,
        )?;
        let mut low = core::array::from_fn(|component| {
            let mut values = [0_i32; 16];
            values[0] = dc[component].0;
            values
        });
        let predictions = dc.map(|value| value.1);
        if self.bands.has_low_pass() {
            self.decode_low_pass(reader, &mut low, position, qp_indices.lp, predictions)?;
        }
        let hp_mode = high_pass::prediction_mode_yuv(&low, self.sampling);
        let high = if self.bands.has_high_pass() {
            high_pass::decode_yuv(
                reader,
                &mut self.entropy,
                &mut self.high_pass_pattern,
                position,
                self.sampling,
                hp_mode,
                HighpassPayload::Combined {
                    flexbits_present: self.bands.has_flexbits(),
                    trim_flexbits: self.trim,
                },
            )?
            .coefficients
        } else {
            Box::new([[0_i32; 256]; 3])
        };
        for component in 0..3 {
            self.components[component].push(SpatialMacroblock {
                coefficients: QuantizedMacroblock {
                    dc_low_pass: low[component],
                    high_pass: high[component],
                    quantizers: self.quantizers.reconstruction_steps_for(
                        component,
                        qp_indices,
                        self.plane.scaled,
                    )?,
                    bands: self.bands,
                },
                prediction: predictions[component],
                hp_prediction: hp_mode,
                lp_qp_index: qp_indices.lp,
            });
        }
        adapt(
            &mut self.entropy,
            &mut self.high_pass_pattern,
            position.x,
            position.width,
        );
        Ok(())
    }

    fn decode_low_pass(
        &mut self,
        reader: &mut PacketBitReader<'_>,
        low: &mut [[i32; 16]; 3],
        position: MacroblockPosition,
        qp_index: u8,
        predictions: [PredictionMode; 3],
    ) -> Result<(), TileDecodeError> {
        let context = syntax::LowPassContext {
            decoded: &self.components,
            position,
            qp_index,
            predictions,
        };
        if self.sampling == ChromaSampling::Cs444 {
            syntax::decode_low_pass_444(
                reader,
                &mut self.entropy,
                &mut self.low_pass_pattern,
                low,
                context,
            )
        } else {
            syntax::decode_low_pass_subsampled(
                reader,
                &mut self.entropy,
                &mut self.low_pass_pattern,
                low,
                context,
                self.sampling,
            )
        }
    }

    pub(in crate::tile_decode) fn finish(self) -> DecodedTile {
        DecodedTile {
            components: self.components.into_iter().collect(),
        }
    }
}

fn tile_geometry(
    tile_width: u32,
    tile_height: u32,
) -> Result<(usize, usize, usize), TileDecodeError> {
    let width = usize::try_from(tile_width)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("YUV tile width"))?;
    let height = usize::try_from(tile_height)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("YUV tile height"))?;
    let count = width
        .checked_mul(height)
        .ok_or(TileDecodeError::ArithmeticOverflow(
            "YUV tile macroblock count",
        ))?;
    Ok((width, height, count))
}

fn read_trim(
    reader: &mut PacketBitReader<'_>,
    trim_flexbits_present: bool,
) -> Result<u8, TileDecodeError> {
    if trim_flexbits_present {
        read_u8(reader, 4)
    } else {
        Ok(0)
    }
}

fn adapt(entropy: &mut TileEntropyState, cbphp: &mut CbphpState, x: usize, width: usize) {
    if x + 1 == width || x.is_multiple_of(16) {
        entropy.dc_vlc.adapt();
        entropy.lp_vlc.adapt();
        entropy.hp_vlc.adapt();
        cbphp.adapt();
    }
}
