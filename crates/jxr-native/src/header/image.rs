//! T.832 `IMAGE_HEADER` parsing.

use crate::{NativeError, bit_reader::BitReader};

use super::{
    fields::{read_dimension, read_u8, read_u16, require_value},
    types::{CodestreamHeader, HeaderFlags},
};

const GDI_SIGNATURE: u64 = 0x574D_5048_4F54_4F00;

pub(super) fn parse_image_header(
    reader: &mut BitReader<'_>,
) -> Result<CodestreamHeader, NativeError> {
    if reader.read_bits(64)? != GDI_SIGNATURE {
        return Err(NativeError::InvalidSignature);
    }
    require_value(reader.read_bits(4)?, 1, "RESERVED_B")?;
    let hard_tiling = reader.read_flag()?;
    let _reserved_c = reader.read_bits(3)?;
    let tiling = reader.read_flag()?;
    let frequency_mode = reader.read_flag()?;
    let orientation = read_u8(reader, 3)?;
    let index_table_present = reader.read_flag()?;
    let overlap_mode = read_overlap_mode(reader)?;
    let short_header = reader.read_flag()?;
    let long_word = reader.read_flag()?;
    let windowing = reader.read_flag()?;
    let trim_flexbits = reader.read_flag()?;
    let _reserved_d = reader.read_flag()?;
    let red_blue_not_swapped = reader.read_flag()?;
    let premultiplied_alpha = reader.read_flag()?;
    let alpha_plane = reader.read_flag()?;
    let output_color_format = read_u8(reader, 4)?;
    let output_bit_depth = read_u8(reader, 4)?;
    validate_output_codes(output_color_format, output_bit_depth)?;
    let dimension_bits = if short_header { 16 } else { 32 };
    let width = read_dimension(reader, dimension_bits, "WIDTH_MINUS1")?;
    let height = read_dimension(reader, dimension_bits, "HEIGHT_MINUS1")?;
    let (tile_widths_mb, tile_heights_mb) = parse_tiles(reader, tiling, short_header)?;
    let margins = parse_margins(reader, windowing, width, height)?;
    let flags = HeaderFlags::from_parsed([
        hard_tiling,
        frequency_mode,
        index_table_present,
        long_word,
        short_header,
        trim_flexbits,
        red_blue_not_swapped,
        premultiplied_alpha,
        alpha_plane,
    ]);
    Ok(CodestreamHeader {
        width,
        height,
        flags,
        orientation,
        overlap_mode,
        output_color_format,
        output_bit_depth,
        tile_widths_mb,
        tile_heights_mb,
        margins,
    })
}

fn read_overlap_mode(reader: &mut BitReader<'_>) -> Result<u8, NativeError> {
    let value = read_u8(reader, 2)?;
    if value == 3 {
        Err(NativeError::ReservedValue {
            field: "OVERLAP_MODE",
            value: 3,
        })
    } else {
        Ok(value)
    }
}

fn parse_margins(
    reader: &mut BitReader<'_>,
    windowing: bool,
    width: u32,
    height: u32,
) -> Result<[u8; 4], NativeError> {
    if windowing {
        Ok([
            read_u8(reader, 6)?,
            read_u8(reader, 6)?,
            read_u8(reader, 6)?,
            read_u8(reader, 6)?,
        ])
    } else {
        Ok([0, 0, inferred_margin(height), inferred_margin(width)])
    }
}

fn inferred_margin(dimension: u32) -> u8 {
    let remainder = dimension % 16;
    if remainder == 0 {
        0
    } else {
        u8::try_from(16 - remainder).unwrap_or_default()
    }
}

fn parse_tiles(
    reader: &mut BitReader<'_>,
    tiling: bool,
    short_header: bool,
) -> Result<(Vec<u16>, Vec<u16>), NativeError> {
    if !tiling {
        return Ok((Vec::new(), Vec::new()));
    }
    let columns_minus_one = usize::from(read_u16(reader, 12)?);
    let rows_minus_one = usize::from(read_u16(reader, 12)?);
    let tile_bits = if short_header { 8 } else { 16 };
    let widths = read_tile_sizes(reader, columns_minus_one, tile_bits)?;
    let heights = read_tile_sizes(reader, rows_minus_one, tile_bits)?;
    Ok((widths, heights))
}

fn read_tile_sizes(
    reader: &mut BitReader<'_>,
    count: usize,
    bits: u8,
) -> Result<Vec<u16>, NativeError> {
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(read_u16(reader, bits)?);
    }
    Ok(values)
}

fn validate_output_codes(color: u8, depth: u8) -> Result<(), NativeError> {
    if color > 8 {
        return Err(NativeError::ReservedValue {
            field: "OUTPUT_CLR_FMT",
            value: u64::from(color),
        });
    }
    if matches!(depth, 5 | 11..=14) {
        return Err(NativeError::ReservedValue {
            field: "OUTPUT_BITDEPTH",
            value: u64::from(depth),
        });
    }
    Ok(())
}
