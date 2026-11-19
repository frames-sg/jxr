//! Frequency-mode tile packets, read a macroblock row at a time.
//!
//! A frequency-mode tile is a packet for each band it codes: DC, low-pass,
//! high-pass and flexbits. No band's syntax reads another band's adaptive
//! state, so [`FrequencyRows`] advances the packets together, a macroblock
//! row of each in turn, and has every band of a row before it moves to the
//! next. It reads every primary colour format, and the macroblocks of an
//! integrated alpha plane after those of the primary plane at each position
//! of each packet, each plane with its [`PlaneReader`].

mod ranges;

use jxr_core::{BandPresence, ColorFormat};

use crate::entropy::PacketBitReader;

use super::{
    TileDecodeError,
    high_pass::HighpassPayload,
    output::PlaneWriter,
    packet_slice,
    plane::PlaneReader,
    quantizer::TileQuantizers,
    spatial::{MacroblockPosition, consume_byte_alignment, parse_packet_prefix, read_trim},
    spatial_tile::TilePlanes,
};

use crate::tail::PacketBand;

pub(super) use ranges::{FrequencyPacketRanges, packet_ranges};

#[cfg(test)]
pub(in crate::tile_decode) mod trace {
    use std::cell::RefCell;

    use crate::{
        entropy::{PacketBitReader, TileEntropyState},
        tail::PacketBand,
        tile_decode::cbphp::CbphpState,
    };

    /// The adaptive state of one image plane: its entropy state, and its
    /// coded-block-pattern state in a packet that has one.
    pub(in crate::tile_decode) type PlaneEnd = (TileEntropyState, Option<CbphpState>);

    /// The end of one band packet.
    #[derive(Debug, Clone)]
    pub(in crate::tile_decode) struct PacketEnd {
        pub(in crate::tile_decode) band: PacketBand,
        /// Bits of the packet left unread after its alignment padding.
        pub(in crate::tile_decode) bits_remaining: usize,
        /// The state of each plane with adaptive syntax in the packet, the
        /// primary plane first. A flexbits packet has none.
        pub(in crate::tile_decode) planes: Vec<PlaneEnd>,
    }

    thread_local! {
        static ENDS: RefCell<Vec<PacketEnd>> = const { RefCell::new(Vec::new()) };
    }

    /// Reports the end of a `band` packet read through `reader`.
    pub(in crate::tile_decode) fn packet_end(
        band: PacketBand,
        reader: &PacketBitReader<'_>,
        planes: &[(&TileEntropyState, Option<&CbphpState>)],
    ) {
        let end = PacketEnd {
            band,
            bits_remaining: reader.bits_remaining(),
            planes: planes
                .iter()
                .map(|&(entropy, pattern)| (entropy.clone(), pattern.cloned()))
                .collect(),
        };
        ENDS.with_borrow_mut(|ends| ends.push(end));
    }

    /// The packets read on this thread since the last call, in the order
    /// they were read.
    pub(in crate::tile_decode) fn take() -> Vec<PacketEnd> {
        ENDS.take()
    }
}

/// The stages a macroblock row passes through, in the order a reader that
/// took each packet whole would reach them: the four band packets, in the
/// order of a tile's packets, then the storing of the coefficients they add
/// up to.
const DC: usize = PacketBand::Dc.index();
const LOW_PASS: usize = PacketBand::LowPass.index();
const HIGH_PASS: usize = PacketBand::HighPass.index();
const FLEXBITS: usize = PacketBand::Flexbits.index();
const STORE: usize = FLEXBITS + 1;

/// The band packets of one frequency-mode tile and the state of reading
/// them together.
pub(super) struct FrequencyRows<'a> {
    /// The primary plane, then an integrated alpha plane.
    planes: Vec<PlaneReader<'a>>,
    width: usize,
    height: usize,
    dc: PacketBitReader<'a>,
    low_pass: Option<PacketBitReader<'a>>,
    high_pass: Option<PacketBitReader<'a>>,
    /// The flexbits packet, unless the tile codes none or its index marks
    /// the packet as left out.
    flexbits: Option<PacketBitReader<'a>>,
    /// Flexbits trim of the tile.
    trim: u8,
    /// The stage that failed and its error, held while the stages before
    /// it are read to the end of the tile.
    failed: Option<(usize, TileDecodeError)>,
}

impl<'a> FrequencyRows<'a> {
    /// Reads the headers of the band packets of a tile `width` by `height`
    /// macroblocks, which `ranges` locates in `source`.
    ///
    /// A packet after the DC packet that cannot be opened does not fail
    /// here. A reader that took each packet whole would have read every
    /// packet before it first, and reported what it found there; so the
    /// failure is held until [`Self::decode_row`] has done the same.
    pub(super) fn open(
        source: &'a [u8],
        planes: TilePlanes<'a>,
        ranges: FrequencyPacketRanges,
        width: usize,
        height: usize,
    ) -> Result<Self, TileDecodeError> {
        if let Some((_, alpha_bands)) = planes.alpha {
            validate_band_subset(planes.primary_bands, alpha_bands)?;
        }
        let bands = planes.primary_bands;
        let mut dc = PacketBitReader::new(packet_slice(source, ranges.dc)?);
        parse_packet_prefix(&mut dc)?;
        // A row is stored once every band packet has been read for it, so
        // a plane holds the macroblocks of a whole row.
        let mut plane = |header, colour, bands| {
            let quantizers = TileQuantizers::parse_dc_packet(&mut dc, header)?;
            PlaneReader::new(header, colour, bands, quantizers, width, width)
        };
        let mut states = vec![plane(planes.primary, planes.primary_colour, bands)?];
        if let Some((header, bands)) = planes.alpha {
            states.push(plane(header, ColorFormat::Luma, bands)?);
        }
        let mut rows = Self {
            planes: states,
            width,
            height,
            dc,
            low_pass: None,
            high_pass: None,
            flexbits: None,
            trim: 0,
            failed: None,
        };
        for band in [LOW_PASS, HIGH_PASS, FLEXBITS] {
            if let Err(error) = rows.open_band(source, ranges, band, planes.trim_present) {
                rows.failed = Some((band, error));
                break;
            }
        }
        Ok(rows)
    }

    /// Opens the packet of `band`, if the tile codes one, and reads its
    /// header.
    fn open_band(
        &mut self,
        source: &'a [u8],
        ranges: FrequencyPacketRanges,
        band: usize,
        trim_present: bool,
    ) -> Result<(), TileDecodeError> {
        let bands = self.planes[0].bands();
        let (coded, range, missing) = match band {
            LOW_PASS => (
                bands.has_low_pass(),
                ranges.low_pass,
                "missing LP frequency packet",
            ),
            HIGH_PASS => (
                bands.has_high_pass(),
                ranges.high_pass,
                "missing HP frequency packet",
            ),
            // An escaped flexbits packet is not an error: the high-pass
            // coefficients are normalized without it.
            _ => (
                bands.has_flexbits() && ranges.flexbits.is_some(),
                ranges.flexbits,
                "",
            ),
        };
        if !coded {
            return Ok(());
        }
        let range = range.ok_or(TileDecodeError::InvalidPlan(missing))?;
        let mut reader = PacketBitReader::new(packet_slice(source, range)?);
        parse_packet_prefix(&mut reader)?;
        match band {
            LOW_PASS => {
                for plane in &mut self.planes {
                    plane.read_low_pass_header(&mut reader)?;
                }
                self.low_pass = Some(reader);
            }
            HIGH_PASS => {
                for plane in &mut self.planes {
                    plane.read_high_pass_header(&mut reader)?;
                }
                self.high_pass = Some(reader);
            }
            _ => {
                self.trim = read_trim(&mut reader, trim_present)?;
                self.flexbits = Some(reader);
            }
        }
        Ok(())
    }

    /// Reads macroblock row `y` of the tile, which follows the rows already
    /// read, from every band packet and stores it in `output`: the primary
    /// components' writers, then the alpha plane's.
    ///
    /// When a stage fails, the stages before it are read on to the end of
    /// the tile, and the first of them to fail is the error returned: the
    /// error of a reader that took each packet whole, in band order.
    pub(super) fn decode_row(
        &mut self,
        y: usize,
        output: &mut [PlaneWriter<'_>],
    ) -> Result<(), TileDecodeError> {
        if self.failed.is_some() {
            // A packet that could not be opened: nothing of this row is read.
            return Err(self.read_on(y));
        }
        for stage in DC..=STORE {
            let result = if stage == STORE {
                self.store_row(y == 0, output)
            } else {
                self.band_row(stage, y == 0)
            };
            if let Err(error) = result {
                self.failed = Some((stage, error));
                self.end_row();
                return Err(self.read_on(y + 1));
            }
        }
        self.end_row();
        Ok(())
    }

    /// Reads the stages before the one that failed from macroblock row
    /// `row` to the end of the tile, and returns the error to report: the
    /// first of them to fail, or else the failure that was held.
    fn read_on(&mut self, row: usize) -> TileDecodeError {
        for row in row..self.height {
            let limit = self.failed.as_ref().map_or(DC, |(stage, _)| *stage);
            for stage in DC..limit {
                if let Err(error) = self.band_row(stage, row == 0) {
                    self.failed = Some((stage, error));
                    break;
                }
            }
            self.end_row();
        }
        let limit = self.failed.as_ref().map_or(DC, |(stage, _)| *stage);
        match (self.finish_bands(limit), self.failed.take()) {
            (Err(error), _) | (Ok(()), Some((_, error))) => error,
            (Ok(()), None) => TileDecodeError::InvalidPlan("frequency tile state"),
        }
    }

    /// Reads one macroblock row of the packet of `band` for every plane
    /// that has the band, each macroblock in the slot of its column.
    ///
    /// The flexbits stage completes the high-pass blocks of the row whether
    /// or not there is a flexbits packet to read them from.
    fn band_row(&mut self, band: usize, first_row: bool) -> Result<(), TileDecodeError> {
        // Each band's loop is compiled for that band alone.
        match band {
            DC => self.read_row::<DC>(first_row),
            LOW_PASS => self.read_row::<LOW_PASS>(first_row),
            HIGH_PASS => self.read_row::<HIGH_PASS>(first_row),
            _ => self.read_row::<FLEXBITS>(first_row),
        }
    }

    /// [`Self::band_row`] of the band `BAND`.
    fn read_row<const BAND: usize>(&mut self, first_row: bool) -> Result<(), TileDecodeError> {
        let width = self.width;
        let trim = self.trim;
        let mut reader = match BAND {
            DC => Some(&mut self.dc),
            LOW_PASS => self.low_pass.as_mut(),
            HIGH_PASS => self.high_pass.as_mut(),
            _ => self.flexbits.as_mut(),
        };
        if reader.is_none() && BAND != FLEXBITS {
            return Ok(());
        }
        for x in 0..width {
            let position = MacroblockPosition {
                width,
                x,
                y: usize::from(!first_row),
            };
            for plane in &mut self.planes {
                match reader.as_deref_mut() {
                    Some(reader) => read_band::<BAND>(plane, reader, position, trim)?,
                    // Without a flexbits packet, the high-pass blocks are
                    // completed without flexbits.
                    None if BAND == FLEXBITS && plane.bands().has_high_pass() => {
                        plane.scale_high_pass(x)?;
                    }
                    None => {}
                }
            }
        }
        Ok(())
    }

    /// Stores the row every band has been read for.
    fn store_row(
        &self,
        first_row: bool,
        output: &mut [PlaneWriter<'_>],
    ) -> Result<(), TileDecodeError> {
        let mut rest = output;
        for plane in &self.planes {
            let (writers, after) = rest
                .split_at_mut_checked(plane.components())
                .ok_or(TileDecodeError::InvalidPlan("decoded component count"))?;
            for x in 0..self.width {
                let position = MacroblockPosition {
                    width: self.width,
                    x,
                    y: usize::from(!first_row),
                };
                plane.store(position, x, writers)?;
            }
            rest = after;
        }
        if !rest.is_empty() {
            return Err(TileDecodeError::InvalidPlan("decoded component count"));
        }
        Ok(())
    }

    /// Makes the row just read the row above the next.
    fn end_row(&mut self) {
        for plane in &mut self.planes {
            plane.end_row(self.width);
        }
    }

    /// Reads the padding that ends each packet after its last row.
    pub(super) fn finish(&mut self) -> Result<(), TileDecodeError> {
        if let Some((_, error)) = self.failed.take() {
            // A packet that could not be opened, in a tile without rows.
            return Err(error);
        }
        self.finish_bands(STORE)
    }

    /// Reads the padding that ends the packets of the bands before `limit`.
    fn finish_bands(&mut self, limit: usize) -> Result<(), TileDecodeError> {
        if limit > DC {
            consume_byte_alignment(&mut self.dc)?;
            #[cfg(test)]
            trace_end(PacketBand::Dc, &self.dc, &self.planes, DC);
        }
        if limit > LOW_PASS
            && let Some(reader) = &mut self.low_pass
        {
            consume_byte_alignment(reader)?;
            #[cfg(test)]
            trace_end(PacketBand::LowPass, reader, &self.planes, LOW_PASS);
        }
        if limit > HIGH_PASS
            && let Some(reader) = &mut self.high_pass
        {
            consume_byte_alignment(reader)?;
            #[cfg(test)]
            trace_end(PacketBand::HighPass, reader, &self.planes, HIGH_PASS);
        }
        if limit > FLEXBITS
            && let Some(reader) = &mut self.flexbits
        {
            consume_byte_alignment(reader)?;
            #[cfg(test)]
            trace::packet_end(PacketBand::Flexbits, reader, &[]);
        }
        Ok(())
    }
}

/// Reads stage `BAND` of the macroblock at `position` of `plane`, which
/// keeps it in the slot of its column, from `reader`, the packet of the
/// band. `trim_flexbits` is the tile's flexbits trim.
fn read_band<const BAND: usize>(
    plane: &mut PlaneReader<'_>,
    reader: &mut PacketBitReader<'_>,
    position: MacroblockPosition,
    trim_flexbits: u8,
) -> Result<(), TileDecodeError> {
    let slot = position.x;
    let bands = plane.bands();
    match BAND {
        DC => plane.read_dc(reader, position, slot),
        LOW_PASS if bands.has_low_pass() => {
            plane.read_low_pass_index(reader, slot)?;
            plane.read_low_pass(reader, position, slot)
        }
        HIGH_PASS if bands.has_high_pass() => {
            plane.read_high_pass_index(reader, slot)?;
            plane.read_high_pass(reader, position, slot, HighpassPayload::VlcOnly)
        }
        // An alpha plane whose band set leaves flexbits out has none in the
        // primary plane's packet.
        FLEXBITS if bands.has_flexbits() => plane.refine_high_pass(slot, reader, trim_flexbits),
        FLEXBITS if bands.has_high_pass() => plane.scale_high_pass(slot),
        _ => Ok(()),
    }
}

/// Reports the end of the packet of `band` to the tests' trace.
#[cfg(test)]
fn trace_end(
    packet: PacketBand,
    reader: &PacketBitReader<'_>,
    planes: &[PlaneReader<'_>],
    band: usize,
) {
    let planes: Vec<_> = planes
        .iter()
        .map(|plane| {
            let (entropy, pattern) = plane.state();
            (entropy, (band == HIGH_PASS).then_some(pattern))
        })
        .collect();
    trace::packet_end(packet, reader, &planes);
}

fn validate_band_subset(primary: BandPresence, alpha: BandPresence) -> Result<(), TileDecodeError> {
    if (alpha.has_low_pass() && !primary.has_low_pass())
        || (alpha.has_high_pass() && !primary.has_high_pass())
        || (alpha.has_flexbits() && !primary.has_flexbits())
    {
        Err(TileDecodeError::Unsupported(
            "integrated alpha bands exceed primary bands",
        ))
    } else {
        Ok(())
    }
}

/// Reads a whole frequency-mode tile into a window for each component of
/// its planes, the alpha plane's last.
#[cfg(test)]
pub(in crate::tile_decode) fn decode_tile(
    source: &[u8],
    planes: TilePlanes<'_>,
    ranges: FrequencyPacketRanges,
    width: usize,
    height: usize,
) -> Result<Vec<super::tests::DecodedWindow>, TileDecodeError> {
    super::tests::read_windows(planes, width * height, |writers| {
        let mut rows = FrequencyRows::open(source, planes, ranges, width, height)?;
        for y in 0..height {
            rows.decode_row(y, writers)?;
        }
        rows.finish()
    })
}

#[cfg(test)]
mod tests {
    use jxr_core::{BandPresence, ByteRange, ChromaSampling, ColorFormat};

    use super::{FrequencyPacketRanges, TilePlanes, decode_tile};
    use crate::{
        ImagePlaneHeader, QuantizerSet,
        bit_writer::BitWriter,
        tile_decode::tests::{
            DecodedWindow, FrequencyTile, TILE_HEIGHT, TILE_WIDTH, check_frequency_tiles,
        },
    };

    /// The planes of `tile`, each read in the bands `bands`.
    fn planes(tile: &FrequencyTile, bands: BandPresence) -> TilePlanes<'_> {
        let headers = &tile.parsed.headers;
        TilePlanes {
            primary: &headers.primary,
            primary_colour: tile.colour,
            primary_bands: bands,
            alpha: headers.alpha.as_ref().map(|alpha| (alpha, bands)),
            trim_present: headers.image.flags.trim_flexbits(),
        }
    }

    fn decode(tile: &FrequencyTile, bands: BandPresence) -> Vec<DecodedWindow> {
        decode_tile(
            &tile.source,
            planes(tile, bands),
            tile.ranges,
            TILE_WIDTH,
            TILE_HEIGHT,
        )
        .unwrap()
    }

    #[test]
    fn encoded_band_packets_decode_to_their_coefficients_and_final_state() {
        check_frequency_tiles(ColorFormat::Luma, 0, false, decode);
    }

    #[test]
    fn encoded_yuv_band_packets_decode_to_their_coefficients_and_final_state() {
        for (internal_color_format, sampling) in [
            (1_u8, ChromaSampling::Cs420),
            (2, ChromaSampling::Cs422),
            (3, ChromaSampling::Cs444),
        ] {
            let colour = ColorFormat::Yuv(sampling);
            check_frequency_tiles(colour, internal_color_format, false, decode);
        }
    }

    #[test]
    fn encoded_multicomponent_band_packets_decode_to_their_coefficients_and_final_state() {
        for (internal_color_format, colour) in [
            (4_u8, ColorFormat::YuvK),
            (6, ColorFormat::NComponent(2)),
            (6, ColorFormat::NComponent(5)),
            (6, ColorFormat::NComponent(16)),
        ] {
            check_frequency_tiles(colour, internal_color_format, false, decode);
        }
    }

    /// Every primary syntax family with an alpha plane beside it. Subsampled
    /// chroma is the case the reader once got wrong: it left the chroma
    /// blocks of a predicted macroblock unpredicted, which T.835 does not.
    #[test]
    fn encoded_band_packets_decode_to_both_planes_coefficients_and_final_state() {
        for (internal_color_format, colour) in [
            (0_u8, ColorFormat::Luma),
            (1, ColorFormat::Yuv(ChromaSampling::Cs420)),
            (2, ColorFormat::Yuv(ChromaSampling::Cs422)),
            (3, ColorFormat::Yuv(ChromaSampling::Cs444)),
            (4, ColorFormat::YuvK),
            (6, ColorFormat::NComponent(5)),
        ] {
            check_frequency_tiles(colour, internal_color_format, true, decode);
        }
    }

    #[test]
    fn dc_packet_interleaves_primary_then_alpha() {
        let packet = [0, 0, 1, 0x5a, 0, 0, 0];
        let range = ByteRange::new(0, packet.len(), packet.len()).unwrap();
        let header = luma_header();
        let decoded = decode_tile(
            &packet,
            TilePlanes {
                primary: &header,
                primary_colour: ColorFormat::Luma,
                primary_bands: BandPresence::DcOnly,
                alpha: Some((&header, BandPresence::DcOnly)),
                trim_present: false,
            },
            FrequencyPacketRanges {
                dc: range,
                low_pass: None,
                high_pass: None,
                flexbits: None,
            },
            1,
            1,
        )
        .unwrap();
        assert_eq!(decoded.len(), 2);
        for window in &decoded {
            assert_eq!(window.stored, [0]);
            assert_eq!(window.records.len(), 1);
        }
    }

    #[test]
    fn hp_packet_keeps_primary_and_alpha_entropy_independent() {
        let mut source = Vec::new();
        let dc = append(&mut source, &dc_packet());
        let low_pass = append(&mut source, &low_pass_packet());
        let high_pass = append(&mut source, &high_pass_packet());
        let header = luma_header_with_bands(1);
        let decoded = decode_tile(
            &source,
            TilePlanes {
                primary: &header,
                primary_colour: ColorFormat::Luma,
                primary_bands: BandPresence::NoFlexbits,
                alpha: Some((&header, BandPresence::NoFlexbits)),
                trim_present: false,
            },
            FrequencyPacketRanges {
                dc,
                low_pass: Some(low_pass),
                high_pass: Some(high_pass),
                flexbits: None,
            },
            1,
            1,
        )
        .unwrap();
        for window in &decoded {
            let [(_, high, _)] = <[_; 1]>::try_from(window.macroblocks()).unwrap();
            assert_eq!(high.iter().filter(|&&value| value != 0).count(), 16);
        }
    }

    /// A band that fails leaves the bands before it to be read to the end
    /// of the tile, and the first of those to fail names the error, as it
    /// did when each packet was read whole: here the DC packet is cut short
    /// in its second row, after a high-pass packet that fails in its first.
    #[test]
    fn an_earlier_band_that_fails_further_down_is_the_error_reported() {
        let header = luma_header_with_bands(1);
        let planes = TilePlanes {
            primary: &header,
            primary_colour: ColorFormat::Luma,
            primary_bands: BandPresence::NoFlexbits,
            alpha: None,
            trim_present: false,
        };
        let mut source = Vec::new();
        // One macroblock row of DC where the tile has two.
        let dc = append(&mut source, &dc_packet()[..5]);
        let low_pass = append(&mut source, &low_pass_packet());
        // A high-pass packet with a start code that is not one.
        let high_pass = append(&mut source, &[0, 0, 2, 0x5a, 0, 0]);
        let ranges = FrequencyPacketRanges {
            dc,
            low_pass: Some(low_pass),
            high_pass: Some(high_pass),
            flexbits: None,
        };
        let error = decode_tile(&source, planes, ranges, 1, 2).unwrap_err();
        let dc_alone = decode_tile(
            &source,
            TilePlanes {
                primary_bands: BandPresence::DcOnly,
                ..planes
            },
            ranges,
            1,
            2,
        )
        .unwrap_err();
        assert_eq!(error, dc_alone);
        assert!(!matches!(
            error,
            crate::tile_decode::TileDecodeError::InvalidStartCode { .. }
        ));
    }

    fn luma_header() -> ImagePlaneHeader {
        luma_header_with_bands(3)
    }

    fn luma_header_with_bands(bands_present: u8) -> ImagePlaneHeader {
        ImagePlaneHeader {
            internal_color_format: 0,
            scaled: false,
            bands_present,
            components: 1,
            chroma_centering_x: 0,
            chroma_centering_y: 0,
            shift_bits: 0,
            mantissa_length: 0,
            exponent_bias: 0,
            dc_quantizers: Some(QuantizerSet {
                components: vec![0],
            }),
            lp_quantizers: (bands_present < 3).then(|| QuantizerSet {
                components: vec![0],
            }),
            hp_quantizers: (bands_present < 2).then(|| QuantizerSet {
                components: vec![0],
            }),
        }
    }

    fn append(source: &mut Vec<u8>, packet: &[u8]) -> ByteRange {
        let offset = source.len();
        source.extend_from_slice(packet);
        ByteRange::new(offset, packet.len(), source.len()).unwrap()
    }

    /// A packet whose start code and arbitrary byte `write` follows.
    fn packet(write: impl FnOnce(&mut BitWriter)) -> Vec<u8> {
        let mut bits = BitWriter::new();
        bits.write_bits(0x0000_015a, 32);
        write(&mut bits);
        bits.into_bytes()
    }

    /// Two macroblocks of DC: no level, and eight zero refinement bits.
    fn dc_packet() -> Vec<u8> {
        packet(|bits| {
            for _ in 0..2 {
                bits.write_bits(0, 1);
                bits.write_bits(0, 8);
            }
        })
    }

    /// Two macroblocks of low-pass: no block, and fifteen zero refinements
    /// of four bits.
    fn low_pass_packet() -> Vec<u8> {
        packet(|bits| {
            for _ in 0..2 {
                bits.write_bits(0, 1);
                for _ in 1..16 {
                    bits.write_bits(0, 4);
                }
            }
        })
    }

    /// Two macroblocks of high-pass: every block coded with one positive
    /// coefficient.
    fn high_pass_packet() -> Vec<u8> {
        packet(|bits| {
            for _ in 0..2 {
                bits.write_bits(1, 1);
                for _ in 0..16 {
                    bits.write_bits(0b00010, 5);
                    bits.write_bits(0, 1);
                }
            }
        })
    }
}
