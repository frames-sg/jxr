//! Tile index and profile-level syntax between image headers and tile packets.

use jxr_core::BandPresence;

use crate::{NativeError, ParsedHeaders, metadata::band_presence};

/// One declared JPEG XR profile and level pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProfileLevel {
    /// T.832 profile identifier.
    pub profile_idc: u8,
    /// T.832 level identifier.
    pub level_idc: u8,
}

/// Parsed syntax locating coded tile packets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodestreamDirectory {
    /// Tile packet offsets in normative index order.
    pub tile_offsets: Vec<u64>,
    /// Declared profile-level pairs; empty means Advanced/255 is inferred.
    pub profiles: Vec<ProfileLevel>,
    /// Byte offset of the first coded tile packet.
    pub tile_data_offset: usize,
}

impl CodestreamDirectory {
    /// Parse the tile index, subsequent-data length, and profile-level records.
    pub fn parse(bytes: &[u8], headers: &ParsedHeaders) -> Result<Self, NativeError> {
        let mut cursor = ByteCursor::new(bytes, headers.bytes_consumed)?;
        let tile_count = tile_count(headers)?;
        if headers.image.requires_index() && !headers.image.flags.index_table_present() {
            return Err(NativeError::ReservedValue {
                field: "INDEX_TABLE_PRESENT_FLAG",
                value: 0,
            });
        }
        let tile_offsets = if headers.image.flags.index_table_present() {
            parse_index(&mut cursor, headers, tile_count)?
        } else {
            Vec::new()
        };
        let subsequent_bytes = cursor.read_vlw()?;
        let subsequent_bytes =
            usize::try_from(subsequent_bytes).map_err(|_| NativeError::IntegerOverflow {
                operation: "converting SubsequentBytes",
            })?;
        let profiles = parse_profile_area(&mut cursor, subsequent_bytes)?;
        Ok(Self {
            tile_offsets,
            profiles,
            tile_data_offset: cursor.position(),
        })
    }
}

/// The band a tile packet of a frequency-mode codestream carries.
///
/// A spatial-mode tile is one packet and has one index entry. A
/// frequency-mode tile is one packet per band its image codes, and the tile
/// index lists them tile by tile in the order of the variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PacketBand {
    Dc,
    LowPass,
    HighPass,
    Flexbits,
}

impl PacketBand {
    /// The packets each tile of a frequency-mode image coding `bands` has,
    /// in tile index order.
    pub(crate) const fn coded(bands: BandPresence) -> &'static [Self] {
        const ORDER: [PacketBand; 4] = [
            PacketBand::Dc,
            PacketBand::LowPass,
            PacketBand::HighPass,
            PacketBand::Flexbits,
        ];
        let count = match bands {
            BandPresence::DcOnly => 1,
            BandPresence::NoHighPass => 2,
            BandPresence::NoFlexbits => 3,
            BandPresence::All => 4,
        };
        ORDER.split_at(count).0
    }

    /// Packets, and index entries, of each tile of an image: one in spatial
    /// mode, and one for each band it codes, [`Self::coded`], in frequency
    /// mode.
    pub(crate) const fn per_tile(frequency_mode: bool, bands: BandPresence) -> usize {
        if frequency_mode {
            Self::coded(bands).len()
        } else {
            1
        }
    }

    /// Position of this band's packet among those of its tile, which is
    /// also its position among the tile's index entries.
    pub(crate) const fn index(self) -> usize {
        self as usize
    }
}

/// Appends the tile index (when the image has one), `SubsequentBytes`, and a
/// single profile-level record: the syntax [`CodestreamDirectory::parse`] reads.
///
/// `tile_offsets` are relative to the first byte after this directory, one
/// for each tile of a spatial-mode image and one for each
/// [`PacketBand::coded`] packet of each tile of a frequency-mode image.
pub(crate) fn write_directory(
    output: &mut Vec<u8>,
    tile_offsets: Option<&[u64]>,
    profile: ProfileLevel,
) {
    if let Some(offsets) = tile_offsets {
        output.extend_from_slice(&INDEX_TABLE_START_CODE.to_be_bytes());
        for &offset in offsets {
            write_vlw(output, offset);
        }
    }
    write_vlw(output, PROFILE_LEVEL_BYTES as u64);
    // RESERVED_L is fifteen zero bits; the final bit marks the last record.
    output.extend_from_slice(&[profile.profile_idc, profile.level_idc, 0, 1]);
}

const INDEX_TABLE_START_CODE: u16 = 1;
/// Length of one `PROFILE_LEVEL_INFO` record.
const PROFILE_LEVEL_BYTES: usize = 4;

/// Writes a variable-length word in its shortest form.
fn write_vlw(output: &mut Vec<u8>, value: u64) {
    if let Ok(short) = u16::try_from(value)
        && short < 0xfb00
    {
        output.extend_from_slice(&short.to_be_bytes());
    } else if let Ok(medium) = u32::try_from(value) {
        output.push(0xfb);
        output.extend_from_slice(&medium.to_be_bytes());
    } else {
        output.push(0xfc);
        output.extend_from_slice(&value.to_be_bytes());
    }
}

fn tile_count(headers: &ParsedHeaders) -> Result<usize, NativeError> {
    headers
        .image
        .tile_widths_mb
        .len()
        .checked_add(1)
        .and_then(|columns| {
            headers
                .image
                .tile_heights_mb
                .len()
                .checked_add(1)
                .and_then(|rows| columns.checked_mul(rows))
        })
        .ok_or(NativeError::IntegerOverflow {
            operation: "computing tile count",
        })
}

fn parse_index(
    cursor: &mut ByteCursor<'_>,
    headers: &ParsedHeaders,
    tile_count: usize,
) -> Result<Vec<u64>, NativeError> {
    let start_code = cursor.read_u16()?;
    if start_code != INDEX_TABLE_START_CODE {
        return Err(NativeError::ReservedValue {
            field: "INDEX_TABLE_STARTCODE",
            value: u64::from(start_code),
        });
    }
    let frequency_mode = headers.image.flags.frequency_mode();
    let bands = if frequency_mode {
        band_presence(headers.primary.bands_present)?
    } else {
        BandPresence::All
    };
    let count = tile_count
        .checked_mul(PacketBand::per_tile(frequency_mode, bands))
        .ok_or(NativeError::IntegerOverflow {
            operation: "computing tile index entry count",
        })?;
    let mut offsets = Vec::with_capacity(count);
    for _ in 0..count {
        offsets.push(cursor.read_vlw()?);
    }
    Ok(offsets)
}

fn parse_profile_area(
    cursor: &mut ByteCursor<'_>,
    subsequent_bytes: usize,
) -> Result<Vec<ProfileLevel>, NativeError> {
    if subsequent_bytes == 0 {
        return Ok(Vec::new());
    }
    if subsequent_bytes < PROFILE_LEVEL_BYTES {
        return Err(NativeError::ReservedValue {
            field: "SubsequentBytes",
            value: subsequent_bytes as u64,
        });
    }
    let end =
        cursor
            .position()
            .checked_add(subsequent_bytes)
            .ok_or(NativeError::IntegerOverflow {
                operation: "locating profile-level data",
            })?;
    cursor.require_position(end)?;
    let mut profiles = Vec::new();
    loop {
        if cursor.position().saturating_add(PROFILE_LEVEL_BYTES) > end {
            return Err(NativeError::Truncated {
                bit_position: cursor.position().saturating_mul(8),
                requested_bits: 32,
            });
        }
        let profile_idc = cursor.read_u8()?;
        let level_idc = cursor.read_u8()?;
        let reserved_and_last = cursor.read_u16()?;
        if reserved_and_last & !1 != 0 {
            return Err(NativeError::ReservedValue {
                field: "PROFILE_LEVEL_INFO reserved bits",
                value: u64::from(reserved_and_last & !1),
            });
        }
        profiles.push(ProfileLevel {
            profile_idc,
            level_idc,
        });
        if reserved_and_last & 1 != 0 {
            break;
        }
    }
    cursor.set_position(end);
    Ok(profiles)
}

struct ByteCursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> ByteCursor<'a> {
    fn new(bytes: &'a [u8], position: usize) -> Result<Self, NativeError> {
        let cursor = Self { bytes, position };
        cursor.require_position(position)?;
        Ok(cursor)
    }

    const fn position(&self) -> usize {
        self.position
    }

    fn set_position(&mut self, position: usize) {
        self.position = position;
    }

    fn require_position(&self, position: usize) -> Result<(), NativeError> {
        if position <= self.bytes.len() {
            Ok(())
        } else {
            Err(NativeError::RangeOutsideInput {
                field: "codestream syntax",
                offset: position,
                length: 0,
                input_length: self.bytes.len(),
            })
        }
    }

    fn read_u8(&mut self) -> Result<u8, NativeError> {
        let value = *self
            .bytes
            .get(self.position)
            .ok_or(NativeError::Truncated {
                bit_position: self.position.saturating_mul(8),
                requested_bits: 8,
            })?;
        self.position += 1;
        Ok(value)
    }

    fn read_u16(&mut self) -> Result<u16, NativeError> {
        let high = self.read_u8()?;
        let low = self.read_u8()?;
        Ok(u16::from_be_bytes([high, low]))
    }

    fn read_u32(&mut self) -> Result<u32, NativeError> {
        let mut bytes = [0_u8; 4];
        for byte in &mut bytes {
            *byte = self.read_u8()?;
        }
        Ok(u32::from_be_bytes(bytes))
    }

    fn read_u64(&mut self) -> Result<u64, NativeError> {
        let mut bytes = [0_u8; 8];
        for byte in &mut bytes {
            *byte = self.read_u8()?;
        }
        Ok(u64::from_be_bytes(bytes))
    }

    fn read_vlw(&mut self) -> Result<u64, NativeError> {
        let first = self.read_u8()?;
        match first {
            0x00..=0xFA => Ok(u64::from(first) * 256 + u64::from(self.read_u8()?)),
            0xFB => self.read_u32().map(u64::from),
            0xFC => self.read_u64(),
            0xFD..=0xFF => Ok(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use jxr_core::BandPresence;

    use super::{
        ByteCursor, CodestreamDirectory, PacketBand, ProfileLevel, write_directory, write_vlw,
    };
    use crate::{
        CodestreamHeader, HeaderFlags, ImagePlaneHeader, QuantizerSet,
        bit_writer::BitWriter,
        header::{parse_codestream_headers, write_codestream_headers},
    };

    #[test]
    fn a_tile_has_a_packet_for_each_band_its_image_codes_in_band_order() {
        use PacketBand::{Dc, Flexbits, HighPass, LowPass};

        assert_eq!(PacketBand::coded(BandPresence::DcOnly), [Dc]);
        assert_eq!(PacketBand::coded(BandPresence::NoHighPass), [Dc, LowPass]);
        assert_eq!(
            PacketBand::coded(BandPresence::NoFlexbits),
            [Dc, LowPass, HighPass]
        );
        let all = PacketBand::coded(BandPresence::All);
        assert_eq!(all, [Dc, LowPass, HighPass, Flexbits]);
        // A packet's place among its tile's packets is its band's.
        for (place, band) in all.iter().enumerate() {
            assert_eq!(band.index(), place);
        }
    }

    /// The directory of a two-tile image in `frequency_mode` whose planes
    /// declare the `bands_present` code, written with `offsets` and read
    /// back.
    fn written_directory(frequency_mode: bool, bands_present: u8, offsets: &[u64]) -> Vec<u64> {
        let set = |present: bool| {
            present.then(|| QuantizerSet {
                components: vec![1],
            })
        };
        let image = CodestreamHeader {
            width: 32,
            height: 16,
            flags: HeaderFlags::from_parsed([
                false,
                frequency_mode,
                true,
                true,
                true,
                false,
                false,
                false,
                false,
            ]),
            orientation: 0,
            overlap_mode: 0,
            output_color_format: 0,
            output_bit_depth: 1,
            tile_widths_mb: vec![1],
            tile_heights_mb: Vec::new(),
            margins: [0; 4],
        };
        let plane = ImagePlaneHeader {
            internal_color_format: 0,
            scaled: false,
            bands_present,
            components: 1,
            chroma_centering_x: 0,
            chroma_centering_y: 0,
            shift_bits: 0,
            mantissa_length: 0,
            exponent_bias: 0,
            dc_quantizers: set(true),
            lp_quantizers: set(bands_present < 3),
            hp_quantizers: set(bands_present < 2),
        };
        let mut writer = BitWriter::new();
        write_codestream_headers(&mut writer, &image, &plane, None).unwrap();
        let mut bytes = writer.into_bytes();
        let profile = ProfileLevel {
            profile_idc: 111,
            level_idc: 255,
        };
        write_directory(&mut bytes, Some(offsets), profile);
        let headers = parse_codestream_headers(&bytes).unwrap();
        let directory = CodestreamDirectory::parse(&bytes, &headers).unwrap();
        assert_eq!(directory.profiles, [profile]);
        // Every entry was read: the tile data starts where the bytes end.
        assert_eq!(directory.tile_data_offset, bytes.len());
        directory.tile_offsets
    }

    #[test]
    fn a_frequency_mode_index_lists_every_band_packet_of_every_tile() {
        // Offsets past the two-byte and five-byte word forms.
        let offsets: Vec<u64> = (0..8).map(|entry| entry * 0x9000).collect();
        for (bands_present, bands) in [
            (0, BandPresence::All),
            (1, BandPresence::NoFlexbits),
            (2, BandPresence::NoHighPass),
            (3, BandPresence::DcOnly),
        ] {
            let entries = 2 * PacketBand::coded(bands).len();
            assert_eq!(
                written_directory(true, bands_present, &offsets[..entries]),
                offsets[..entries],
                "{bands:?}"
            );
            // A spatial-mode tile is one packet whatever bands it codes.
            assert_eq!(
                written_directory(false, bands_present, &offsets[..2]),
                offsets[..2],
                "{bands:?}"
            );
        }
    }

    #[test]
    fn written_vlw_values_read_back_at_every_width_boundary() {
        let values = [
            0,
            1,
            0xfaff,
            0xfb00,
            0xffff,
            u64::from(u32::MAX),
            u64::from(u32::MAX) + 1,
            u64::MAX,
        ];
        let mut bytes = Vec::new();
        for value in values {
            write_vlw(&mut bytes, value);
        }
        // 2 + 2 + 2 + 5 + 5 + 5 + 9 + 9 bytes: each value in its shortest form.
        assert_eq!(bytes.len(), 39);
        let mut cursor = ByteCursor::new(&bytes, 0).unwrap();
        for value in values {
            assert_eq!(cursor.read_vlw().unwrap(), value);
        }
        assert_eq!(cursor.position(), bytes.len());
    }

    #[test]
    fn reads_all_vlw_encodings() {
        let bytes = [
            0x01, 0x02, 0xFB, 0, 0, 1, 2, 0xFC, 0, 0, 0, 0, 0, 0, 1, 3, 0xFD,
        ];
        let mut cursor = ByteCursor::new(&bytes, 0).unwrap();
        assert_eq!(cursor.read_vlw().unwrap(), 0x0102);
        assert_eq!(cursor.read_vlw().unwrap(), 0x0102);
        assert_eq!(cursor.read_vlw().unwrap(), 0x0103);
        assert_eq!(cursor.read_vlw().unwrap(), 0);
    }
}
