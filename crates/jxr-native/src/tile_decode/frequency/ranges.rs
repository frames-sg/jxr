use jxr_core::{BandPresence, ByteRange};

use crate::ParsedCodestream;

use super::super::TileDecodeError;

#[derive(Debug, Clone, Copy)]
pub(in crate::tile_decode) struct FrequencyPacketRanges {
    pub(in crate::tile_decode) dc: ByteRange,
    pub(in crate::tile_decode) low_pass: Option<ByteRange>,
    pub(in crate::tile_decode) high_pass: Option<ByteRange>,
    pub(in crate::tile_decode) flexbits: Option<ByteRange>,
}

pub(in crate::tile_decode) fn packet_ranges(
    source_len: usize,
    parsed: &ParsedCodestream,
    bands: BandPresence,
    tile_count: usize,
) -> Result<Vec<FrequencyPacketRanges>, TileDecodeError> {
    let bands_per_tile = band_count(bands);
    let expected =
        tile_count
            .checked_mul(bands_per_tile)
            .ok_or(TileDecodeError::ArithmeticOverflow(
                "frequency index entry count",
            ))?;
    if parsed.directory.tile_offsets.len() != expected {
        return Err(TileDecodeError::InvalidPlan(
            "frequency tile index entry count",
        ));
    }
    let base = parsed
        .codestream_range
        .start
        .checked_add(parsed.directory.tile_data_offset)
        .ok_or(TileDecodeError::ArithmeticOverflow("frequency packet base"))?;
    let mut starts = Vec::with_capacity(expected);
    for (index, &relative) in parsed.directory.tile_offsets.iter().enumerate() {
        if is_flexbits_escape(index, bands_per_tile, relative) {
            starts.push(None);
            continue;
        }
        let relative = usize::try_from(relative).map_err(|_| {
            TileDecodeError::ArithmeticOverflow("frequency packet offset conversion")
        })?;
        let start = base
            .checked_add(relative)
            .ok_or(TileDecodeError::ArithmeticOverflow(
                "frequency packet offset",
            ))?;
        if start < base || start >= parsed.codestream_range.end || start >= source_len {
            return Err(TileDecodeError::PacketRangeOutsideInput {
                offset: start,
                length: 1,
                input_length: source_len,
            });
        }
        starts.push(Some(start));
    }
    let mut physical = starts.iter().flatten().copied().collect::<Vec<_>>();
    physical.sort_unstable();
    if physical.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(TileDecodeError::InvalidPlan(
            "duplicate frequency packet offsets",
        ));
    }

    let mut ranges = Vec::with_capacity(tile_count);
    for tile in 0..tile_count {
        let first = tile * bands_per_tile;
        let dc = range_at(first, &starts, &physical, parsed.codestream_range.end)?;
        let low_pass = (bands_per_tile > 1)
            .then(|| range_at(first + 1, &starts, &physical, parsed.codestream_range.end))
            .transpose()?;
        let high_pass = (bands_per_tile > 2)
            .then(|| range_at(first + 2, &starts, &physical, parsed.codestream_range.end))
            .transpose()?;
        let flexbits = if bands_per_tile > 3 && starts[first + 3].is_some() {
            Some(range_at(
                first + 3,
                &starts,
                &physical,
                parsed.codestream_range.end,
            )?)
        } else {
            None
        };
        ranges.push(FrequencyPacketRanges {
            dc,
            low_pass,
            high_pass,
            flexbits,
        });
    }
    Ok(ranges)
}

const fn is_flexbits_escape(index: usize, bands_per_tile: usize, relative: u64) -> bool {
    bands_per_tile == 4 && index % bands_per_tile == 3 && relative == 0
}

const fn band_count(bands: BandPresence) -> usize {
    match bands {
        BandPresence::DcOnly => 1,
        BandPresence::NoHighPass => 2,
        BandPresence::NoFlexbits => 3,
        BandPresence::All => 4,
    }
}

fn range_at(
    index: usize,
    starts: &[Option<usize>],
    physical: &[usize],
    codestream_end: usize,
) -> Result<ByteRange, TileDecodeError> {
    let start = starts[index].ok_or(TileDecodeError::InvalidPlan(
        "escaped required frequency packet",
    ))?;
    let position = physical
        .binary_search(&start)
        .map_err(|_| TileDecodeError::InvalidPlan("frequency packet ordering"))?;
    let end = physical
        .get(position + 1)
        .copied()
        .unwrap_or(codestream_end);
    let length = end
        .checked_sub(start)
        .ok_or(TileDecodeError::InvalidPlan("frequency packet extent"))?;
    ByteRange::new(start, length, codestream_end)
        .map_err(|_| TileDecodeError::InvalidPlan("frequency packet range"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_vlw_is_a_flexbits_escape_only_in_the_fourth_band_slot() {
        assert!(!is_flexbits_escape(0, 4, 0));
        assert!(is_flexbits_escape(3, 4, 0));
        assert!(is_flexbits_escape(7, 4, 0));
        assert!(!is_flexbits_escape(2, 3, 0));
        assert!(!is_flexbits_escape(3, 4, 8));
    }
}
