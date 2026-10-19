//! Integrated-alpha packet traversal for spatial primary planes.

use jxr_core::{BandPresence, PredictionMode};

use crate::{
    ImagePlaneHeader,
    entropy::{PacketBitReader, TileEntropyState},
};

use super::{
    TileDecodeError,
    cbphp::CbphpState,
    multicomponent,
    output::{MacroblockRecord, PlaneWriter},
    quantizer::TileQuantizers,
    spatial::{
        MacroblockPosition, PredictionRecord, adapt_at_boundary, consume_byte_alignment,
        decode_high_band, decode_low_bands, parse_packet_prefix, read_u8,
    },
    yuv,
};

pub(super) fn decode_spatial(
    packet: &[u8],
    planes: IntegratedPlanes<'_>,
    tile_width: u32,
    tile_height: u32,
    output: &mut [PlaneWriter<'_>],
) -> Result<(), TileDecodeError> {
    let Some((alpha_output, primary_output)) = output.split_last_mut() else {
        return Err(TileDecodeError::InvalidPlan("decoded component count"));
    };
    let width = usize::try_from(tile_width)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("integrated alpha tile width"))?;
    let height = usize::try_from(tile_height)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("integrated alpha tile height"))?;
    let count = width
        .checked_mul(height)
        .ok_or(TileDecodeError::ArithmeticOverflow(
            "integrated alpha macroblock count",
        ))?;
    let mut reader = PacketBitReader::new(packet);
    parse_packet_prefix(&mut reader)?;
    let trim = if planes.trim_present {
        read_u8(&mut reader, 4)?
    } else {
        0
    };
    let primary_quantizers = TileQuantizers::parse(&mut reader, planes.primary)?;
    let mut primary_decoder = match planes.primary.internal_color_format {
        0 => PrimaryDecoder::Luma(PlaneDecoder::new(
            primary_quantizers,
            planes.primary,
            planes.primary_bands,
            trim,
            width,
            count,
        )),
        1..=3 => PrimaryDecoder::Yuv(yuv::spatial::SpatialDecoder::new(
            primary_quantizers,
            planes.primary,
            planes.primary_bands,
            trim,
            yuv::sampling(planes.primary.internal_color_format)?,
            width,
            count,
        )),
        4 | 6 => PrimaryDecoder::Multi(multicomponent::SpatialDecoder::new(
            primary_quantizers,
            planes.primary,
            planes.primary_bands,
            trim,
            width,
            count,
        )?),
        _ => {
            return Err(TileDecodeError::Unsupported(
                "integrated alpha primary component layout",
            ));
        }
    };
    let mut alpha_decoder = PlaneDecoder::new(
        TileQuantizers::parse(&mut reader, planes.alpha)?,
        planes.alpha,
        planes.alpha_bands,
        trim,
        width,
        count,
    );
    for y in 0..height {
        for x in 0..width {
            let position = MacroblockPosition { width, x, y };
            primary_decoder.decode_macroblock(&mut reader, position, primary_output)?;
            alpha_decoder.decode_macroblock(&mut reader, position, alpha_output)?;
        }
    }
    consume_byte_alignment(&mut reader)
}

enum PrimaryDecoder<'a> {
    Luma(PlaneDecoder<'a>),
    Yuv(yuv::spatial::SpatialDecoder<'a>),
    Multi(multicomponent::SpatialDecoder<'a>),
}

impl PrimaryDecoder<'_> {
    fn decode_macroblock(
        &mut self,
        reader: &mut PacketBitReader<'_>,
        position: MacroblockPosition,
        output: &mut [PlaneWriter<'_>],
    ) -> Result<(), TileDecodeError> {
        match self {
            Self::Luma(decoder) => match output {
                [output] => decoder.decode_macroblock(reader, position, output),
                _ => Err(TileDecodeError::InvalidPlan("decoded component count")),
            },
            Self::Yuv(decoder) => decoder.decode_macroblock(reader, position, output),
            Self::Multi(decoder) => decoder.decode_macroblock(reader, position, output),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct IntegratedPlanes<'a> {
    pub(super) primary: &'a ImagePlaneHeader,
    pub(super) alpha: &'a ImagePlaneHeader,
    pub(super) primary_bands: BandPresence,
    pub(super) alpha_bands: BandPresence,
    pub(super) trim_present: bool,
}

struct PlaneDecoder<'a> {
    entropy: TileEntropyState,
    cbphp: CbphpState,
    quantizers: TileQuantizers,
    plane: &'a ImagePlaneHeader,
    bands: BandPresence,
    trim: u8,
    history: Vec<PredictionRecord>,
}

impl<'a> PlaneDecoder<'a> {
    fn new(
        quantizers: TileQuantizers,
        plane: &'a ImagePlaneHeader,
        bands: BandPresence,
        trim: u8,
        width: usize,
        capacity: usize,
    ) -> Self {
        let mut entropy = TileEntropyState::new();
        entropy.reset_tile();
        Self {
            entropy,
            cbphp: CbphpState::new(width),
            quantizers,
            plane,
            bands,
            trim,
            history: Vec::with_capacity(capacity),
        }
    }

    fn decode_macroblock(
        &mut self,
        reader: &mut PacketBitReader<'_>,
        position: MacroblockPosition,
        output: &mut PlaneWriter<'_>,
    ) -> Result<(), TileDecodeError> {
        if position.x.is_multiple_of(16) {
            self.entropy.reset_scan_totals();
        }
        let qp = self.quantizers.indices(reader)?;
        let (low, prediction) = decode_low_bands(
            reader,
            &mut self.entropy,
            &self.history,
            position,
            qp.lp,
            self.bands,
        )?;
        let hp_mode = super::high_pass::prediction_mode(&low);
        let high = decode_high_band(
            reader,
            &mut self.entropy,
            &mut self.cbphp,
            position,
            hp_mode,
            self.bands,
            self.trim,
        )?;
        output.push(
            &low,
            &high,
            MacroblockRecord {
                quantizers: self
                    .quantizers
                    .reconstruction_steps(qp, self.plane.scaled)?,
                bands: self.bands,
                prediction,
                hp_prediction: if self.bands.has_high_pass() {
                    hp_mode
                } else {
                    PredictionMode::None
                },
            },
        )?;
        self.history.push(PredictionRecord {
            dc_low_pass: low,
            lp_qp_index: qp.lp,
        });
        adapt_at_boundary(
            &mut self.entropy,
            &mut self.cbphp,
            position.x,
            position.width,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use jxr_core::BandPresence;

    use super::{IntegratedPlanes, decode_spatial};
    use crate::{ImagePlaneHeader, QuantizerSet, tile_decode::output::PlaneWriter};

    #[test]
    fn spatial_yuv444_interleaves_primary_then_alpha() {
        let packet = [0, 0, 1, 0x5a, 0x80, 0, 0, 0, 0];
        let primary = header(3, 3);
        let alpha = header(0, 1);
        let mut storage = [-1_i32; 4];
        let mut writers: Vec<_> = storage
            .chunks_mut(1)
            .map(|window| PlaneWriter::new(window, 16, 1))
            .collect();
        decode_spatial(
            &packet,
            IntegratedPlanes {
                primary: &primary,
                alpha: &alpha,
                primary_bands: BandPresence::DcOnly,
                alpha_bands: BandPresence::DcOnly,
                trim_present: false,
            },
            1,
            1,
            &mut writers,
        )
        .unwrap();
        let records: Vec<_> = writers
            .into_iter()
            .map(|writer| writer.finish().unwrap())
            .collect();
        assert_eq!(records.len(), 4);
        assert!(
            records
                .iter()
                .all(|plane| plane.len() == 1 && plane[0].bands == BandPresence::DcOnly)
        );
        assert_eq!(storage, [0; 4]);
    }

    fn header(internal_color_format: u8, components: u16) -> ImagePlaneHeader {
        ImagePlaneHeader {
            internal_color_format,
            scaled: false,
            bands_present: 3,
            components,
            chroma_centering_x: 0,
            chroma_centering_y: 0,
            shift_bits: 0,
            mantissa_length: 0,
            exponent_bias: 0,
            dc_quantizers: Some(QuantizerSet {
                components: vec![0; usize::from(components)],
            }),
            lp_quantizers: None,
            hp_quantizers: None,
        }
    }
}
