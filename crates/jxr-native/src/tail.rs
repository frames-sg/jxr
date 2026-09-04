//! Tile index and profile-level syntax between image headers and tile packets.

use crate::{NativeError, ParsedHeaders};

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
        let index_required = headers.image.flags.frequency_mode() || tile_count > 1;
        if index_required && !headers.image.flags.index_table_present() {
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
    if start_code != 1 {
        return Err(NativeError::ReservedValue {
            field: "INDEX_TABLE_STARTCODE",
            value: u64::from(start_code),
        });
    }
    let bands = if headers.image.flags.frequency_mode() {
        4_usize.saturating_sub(usize::from(headers.primary.bands_present))
    } else {
        1
    };
    let count = tile_count
        .checked_mul(bands)
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
    if subsequent_bytes < 4 {
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
        if cursor.position().saturating_add(4) > end {
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
    use super::ByteCursor;

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
