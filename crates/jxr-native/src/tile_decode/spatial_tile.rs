//! Spatial-mode tile packets, read a macroblock row at a time.
//!
//! A spatial-mode tile is one packet with every band of each macroblock
//! interleaved. [`SpatialRows`] reads it for every primary colour format,
//! and reads the macroblocks of an integrated alpha plane after those of
//! the primary plane at each position, each plane with its
//! [`PlaneReader`].

use jxr_core::{BandPresence, ColorFormat};

use crate::{ImagePlaneHeader, entropy::PacketBitReader};

use super::{
    TileDecodeError,
    output::PlaneWriter,
    plane::PlaneReader,
    quantizer::TileQuantizers,
    spatial::{MacroblockPosition, consume_byte_alignment, parse_packet_prefix, read_trim},
};

/// The planes a tile's packets code, and the bands of each that are read.
#[derive(Clone, Copy)]
pub(super) struct TilePlanes<'a> {
    pub(super) primary: &'a ImagePlaneHeader,
    /// The internal colour format the primary plane is coded in.
    pub(super) primary_colour: ColorFormat,
    pub(super) primary_bands: BandPresence,
    /// An integrated alpha plane, which is Y-only, and its bands.
    pub(super) alpha: Option<(&'a ImagePlaneHeader, BandPresence)>,
    /// Whether packets carry a flexbits trim.
    pub(super) trim_present: bool,
}

/// One spatial-mode tile packet and the state of reading it.
pub(super) struct SpatialRows<'a> {
    reader: PacketBitReader<'a>,
    primary: PlaneReader<'a>,
    alpha: Option<PlaneReader<'a>>,
    /// The flexbits trim the planes share.
    trim: u8,
    width: usize,
}

impl<'a> SpatialRows<'a> {
    /// Reads the headers of `packet`, a tile `width` macroblocks wide: the
    /// packet prefix, the flexbits trim the planes share, and the tile
    /// quantizers of the primary plane and then of the alpha plane.
    pub(super) fn open(
        packet: &'a [u8],
        planes: TilePlanes<'a>,
        width: usize,
    ) -> Result<Self, TileDecodeError> {
        let mut reader = PacketBitReader::new(packet);
        parse_packet_prefix(&mut reader)?;
        let trim = read_trim(&mut reader, planes.trim_present)?;
        // A spatial packet stores each macroblock as soon as it is read, so
        // a plane holds one macroblock at a time.
        let mut plane = |header, colour, bands| {
            let quantizers = TileQuantizers::parse(&mut reader, header)?;
            PlaneReader::new(header, colour, bands, quantizers, width, 1)
        };
        let primary = plane(planes.primary, planes.primary_colour, planes.primary_bands)?;
        let alpha = planes
            .alpha
            .map(|(header, bands)| plane(header, ColorFormat::Luma, bands))
            .transpose()?;
        Ok(Self {
            reader,
            primary,
            alpha,
            trim,
            width,
        })
    }

    /// Reads macroblock row `y` of the tile, which follows the rows already
    /// read, into `output`: the primary components' writers, then the alpha
    /// plane's.
    pub(super) fn decode_row(
        &mut self,
        y: usize,
        output: &mut [PlaneWriter<'_>],
    ) -> Result<(), TileDecodeError> {
        let alpha_writers = usize::from(self.alpha.is_some());
        if output.len() != self.primary.components() + alpha_writers {
            return Err(TileDecodeError::InvalidPlan("decoded component count"));
        }
        let (primary_output, alpha_output) = output.split_at_mut(self.primary.components());
        let width = self.width;
        for x in 0..width {
            let position = MacroblockPosition { width, x, y }.in_two_rows();
            self.primary
                .read_macroblock(&mut self.reader, position, self.trim, primary_output)?;
            if let Some(alpha) = &mut self.alpha {
                alpha.read_macroblock(&mut self.reader, position, self.trim, alpha_output)?;
            }
        }
        self.primary.end_row(width);
        if let Some(alpha) = &mut self.alpha {
            alpha.end_row(width);
        }
        Ok(())
    }

    /// Reads the padding that ends the packet after its last row.
    pub(super) fn finish(&mut self) -> Result<(), TileDecodeError> {
        consume_byte_alignment(&mut self.reader)
    }
}

#[cfg(test)]
mod tests {
    use jxr_core::{BandPresence, BitstreamMode, ChromaSampling, ColorFormat, PredictionMode};

    use super::super::{
        output::PlaneWriter,
        tests::{
            BAND_SETS, DecodedWindow, TILE_HEIGHT, TILE_WIDTH, macroblocks, plane_macroblocks,
            read_windows, unit_quantizer_plane,
        },
    };
    use super::{SpatialRows, TilePlanes};
    use crate::{
        ImagePlaneHeader, QuantizerSet,
        reconstruct::QuantizedMacroblock,
        tile_encode::{EncodedTile, PacketOptions, TileRegion, encode_tile_with_state},
    };

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
        let planes = TilePlanes {
            primary: &primary,
            primary_colour: ColorFormat::Yuv(ChromaSampling::Cs444),
            primary_bands: BandPresence::DcOnly,
            alpha: Some((&alpha, BandPresence::DcOnly)),
            trim_present: false,
        };
        let mut rows = SpatialRows::open(&packet, planes, 1).unwrap();
        rows.decode_row(0, &mut writers).unwrap();
        rows.finish().unwrap();
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

    /// Every primary syntax family, alone and with an alpha plane beside
    /// it, in every band set, with and without a flexbits trim: the reader
    /// must consume the packet exactly, recover every plane's coefficients,
    /// end with each plane's writer state, and record a high-pass direction,
    /// the one all components of a plane share, only for macroblocks with a
    /// high-pass band.
    #[test]
    fn encoded_packets_decode_to_every_planes_coefficients_and_final_state() {
        let formats = [
            (0_u8, 1_u16, ColorFormat::Luma),
            (1, 3, ColorFormat::Yuv(ChromaSampling::Cs420)),
            (2, 3, ColorFormat::Yuv(ChromaSampling::Cs422)),
            (3, 3, ColorFormat::Yuv(ChromaSampling::Cs444)),
            (4, 4, ColorFormat::YuvK),
            (6, 2, ColorFormat::NComponent(2)),
            (6, 5, ColorFormat::NComponent(5)),
            (6, 16, ColorFormat::NComponent(16)),
        ];
        for (internal_color_format, component_count, colour) in formats {
            for (bands, trim_flexbits) in BAND_SETS {
                for with_alpha in [false, true] {
                    let case =
                        format!("{colour:?} {bands:?} trim {trim_flexbits:?} alpha {with_alpha}");
                    let primary_header =
                        unit_quantizer_plane(internal_color_format, component_count, bands);
                    let alpha_header = unit_quantizer_plane(0, 1, bands);
                    let planes = TilePlanes {
                        primary: &primary_header,
                        primary_colour: colour,
                        primary_bands: bands,
                        alpha: with_alpha.then_some((&alpha_header, bands)),
                        trim_present: trim_flexbits.is_some(),
                    };
                    check_tile(planes, trim_flexbits, &case);
                }
            }
        }
    }

    /// Writes one tile of the planes of `planes` and reads it back.
    fn check_tile(planes: TilePlanes<'_>, trim_flexbits: Option<u8>, case: &str) {
        let (colour, bands) = (planes.primary_colour, planes.primary_bands);
        let components: Vec<Vec<QuantizedMacroblock>> = plane_macroblocks(colour, bands);
        // Alpha coefficients unrelated to the primary plane's.
        let alpha = planes.alpha.map(|_| macroblocks(4_242, bands));
        let tile = TileRegion {
            stride: TILE_WIDTH,
            x: 0,
            y: 0,
            width: TILE_WIDTH,
            height: TILE_HEIGHT,
        };
        let options = PacketOptions {
            colour,
            bands,
            trim_flexbits,
            mode: BitstreamMode::Spatial,
        };
        let EncodedTile {
            packets,
            primary: primary_state,
            alpha: alpha_state,
        } = encode_tile_with_state(&components, alpha.as_deref(), tile, options).unwrap();
        // A spatial-mode tile is one packet.
        let [packet] = <[Vec<u8>; 1]>::try_from(packets).unwrap();

        let mut rows = SpatialRows::open(&packet, planes, TILE_WIDTH).unwrap();
        assert_eq!(rows.trim, trim_flexbits.unwrap_or(0), "{case}");
        let windows = read_windows(planes, TILE_WIDTH * TILE_HEIGHT, |writers| {
            for y in 0..TILE_HEIGHT {
                rows.decode_row(y, writers)?;
            }
            rows.finish()
        })
        .unwrap();
        assert_eq!(rows.reader.bits_remaining(), 0, "{case}");
        let (entropy, cbphp) = rows.primary.state();
        assert_eq!(
            (entropy, cbphp),
            (&primary_state.0, &primary_state.1),
            "{case}"
        );
        let alpha_end = rows.alpha.as_ref().map(super::PlaneReader::state);
        let written_end = alpha_state
            .as_ref()
            .map(|(entropy, cbphp)| (entropy, cbphp));
        assert_eq!(alpha_end, written_end, "{case}");

        let decoded: Vec<_> = windows.iter().map(DecodedWindow::macroblocks).collect();
        let sources = components.iter().chain(&alpha);
        for (plane, source) in sources.enumerate() {
            // Every component of the primary plane shares its first
            // component's direction; the alpha plane has its own.
            let leader = if plane < components.len() { 0 } else { plane };
            for (index, ((low, high, record), source)) in
                decoded[plane].iter().zip(source).enumerate()
            {
                let at = format!("{case} plane {plane} macroblock {index}");
                assert_eq!(
                    record.hp_prediction, decoded[leader][index].2.hp_prediction,
                    "{at}"
                );
                if !bands.has_high_pass() {
                    assert_eq!(record.hp_prediction, PredictionMode::None, "{at}");
                }
                assert_eq!(low, &source.dc_low_pass, "{at}");
                // Trimmed or absent flexbits drop refinement bits.
                if bands == BandPresence::All && trim_flexbits.is_none() {
                    assert_eq!(high, &source.high_pass, "{at}");
                }
            }
        }
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
