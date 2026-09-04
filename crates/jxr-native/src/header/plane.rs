//! T.832 `IMAGE_PLANE_HEADER` parsing.

use crate::{NativeError, bit_reader::BitReader};

use super::{
    fields::{read_u8, read_u16},
    types::{ImagePlaneHeader, QuantizerSet},
};

pub(super) fn parse_plane_header(
    reader: &mut BitReader<'_>,
    output_bit_depth: u8,
) -> Result<ImagePlaneHeader, NativeError> {
    let internal_color_format = read_u8(reader, 3)?;
    validate_internal_color(internal_color_format)?;
    let scaled = reader.read_flag()?;
    let bands_present = read_u8(reader, 4)?;
    validate_bands(bands_present)?;
    let (components, chroma_centering_x, chroma_centering_y) =
        parse_component_layout(reader, internal_color_format)?;
    let shift_bits = if matches!(output_bit_depth, 2 | 3 | 6) {
        read_u8(reader, 8)?
    } else {
        0
    };
    let (mantissa_length, exponent_bias) = parse_float_parameters(reader, output_bit_depth)?;
    let dc_quantizers = parse_optional_quantizers(reader, components)?;
    let (lp_quantizers, hp_quantizers) =
        parse_high_band_quantizers(reader, components, bands_present)?;
    reader.align_zero()?;
    Ok(ImagePlaneHeader {
        internal_color_format,
        scaled,
        bands_present,
        components,
        chroma_centering_x,
        chroma_centering_y,
        shift_bits,
        mantissa_length,
        exponent_bias,
        dc_quantizers,
        lp_quantizers,
        hp_quantizers,
    })
}

fn validate_internal_color(value: u8) -> Result<(), NativeError> {
    if matches!(value, 5 | 7) {
        Err(NativeError::ReservedValue {
            field: "INTERNAL_CLR_FMT",
            value: u64::from(value),
        })
    } else {
        Ok(())
    }
}

fn validate_bands(value: u8) -> Result<(), NativeError> {
    if value > 3 {
        Err(NativeError::ReservedValue {
            field: "BANDS_PRESENT",
            value: u64::from(value),
        })
    } else {
        Ok(())
    }
}

fn parse_float_parameters(
    reader: &mut BitReader<'_>,
    output_bit_depth: u8,
) -> Result<(u8, i8), NativeError> {
    if output_bit_depth == 7 {
        Ok((read_u8(reader, 8)?, read_u8(reader, 8)?.cast_signed()))
    } else {
        Ok((0, 0))
    }
}

fn parse_component_layout(
    reader: &mut BitReader<'_>,
    color: u8,
) -> Result<(u16, u8, u8), NativeError> {
    match color {
        0 => Ok((1, 0, 0)),
        1 => parse_yuv420_layout(reader),
        2 => parse_yuv422_layout(reader),
        3 => {
            let _reserved_f = reader.read_bits(4)?;
            let _reserved_h = reader.read_bits(4)?;
            Ok((3, 0, 0))
        }
        4 => Ok((4, 0, 0)),
        6 => parse_ncomponent_layout(reader),
        _ => Err(NativeError::ReservedValue {
            field: "INTERNAL_CLR_FMT",
            value: u64::from(color),
        }),
    }
}

fn parse_yuv420_layout(reader: &mut BitReader<'_>) -> Result<(u16, u8, u8), NativeError> {
    let _reserved_e = reader.read_flag()?;
    let x = read_chroma_centering(reader)?;
    let _reserved_g = reader.read_flag()?;
    let y = read_chroma_centering(reader)?;
    Ok((3, x, y))
}

fn parse_yuv422_layout(reader: &mut BitReader<'_>) -> Result<(u16, u8, u8), NativeError> {
    let _reserved_e = reader.read_flag()?;
    let x = read_chroma_centering(reader)?;
    let _reserved_h = reader.read_bits(4)?;
    Ok((3, x, 0))
}

fn parse_ncomponent_layout(reader: &mut BitReader<'_>) -> Result<(u16, u8, u8), NativeError> {
    let short_count = read_u16(reader, 4)?;
    let components = if short_count == 15 {
        read_u16(reader, 12)?
            .checked_add(16)
            .ok_or(NativeError::IntegerOverflow {
                operation: "computing component count",
            })?
    } else {
        let _reserved_h = reader.read_bits(4)?;
        short_count + 1
    };
    Ok((components, 0, 0))
}

fn parse_high_band_quantizers(
    reader: &mut BitReader<'_>,
    components: u16,
    bands_present: u8,
) -> Result<(Option<QuantizerSet>, Option<QuantizerSet>), NativeError> {
    if bands_present == 3 {
        return Ok((None, None));
    }
    let _reserved_i = reader.read_flag()?;
    let low_pass = parse_optional_quantizers(reader, components)?;
    if bands_present == 2 {
        return Ok((low_pass, None));
    }
    let _reserved_j = reader.read_flag()?;
    let high_pass = parse_optional_quantizers(reader, components)?;
    Ok((low_pass, high_pass))
}

fn parse_optional_quantizers(
    reader: &mut BitReader<'_>,
    components: u16,
) -> Result<Option<QuantizerSet>, NativeError> {
    reader
        .read_flag()?
        .then(|| parse_quantizer_set(reader, components))
        .transpose()
}

fn parse_quantizer_set(
    reader: &mut BitReader<'_>,
    components: u16,
) -> Result<QuantizerSet, NativeError> {
    let mode = if components == 1 {
        0
    } else {
        read_u8(reader, 2)?
    };
    let values = match mode {
        0 => vec![read_u8(reader, 8)?; usize::from(components)],
        1 => separate_quantizers(reader, components)?,
        2 => independent_quantizers(reader, components)?,
        3 => {
            return Err(NativeError::ReservedValue {
                field: "COMPONENT_MODE",
                value: 3,
            });
        }
        _ => unreachable!(),
    };
    Ok(QuantizerSet { components: values })
}

fn separate_quantizers(
    reader: &mut BitReader<'_>,
    components: u16,
) -> Result<Vec<u8>, NativeError> {
    let luma = read_u8(reader, 8)?;
    let chroma = read_u8(reader, 8)?;
    let mut values = vec![chroma; usize::from(components)];
    values[0] = luma;
    Ok(values)
}

fn independent_quantizers(
    reader: &mut BitReader<'_>,
    components: u16,
) -> Result<Vec<u8>, NativeError> {
    let mut values = Vec::with_capacity(usize::from(components));
    for _ in 0..components {
        values.push(read_u8(reader, 8)?);
    }
    Ok(values)
}

fn read_chroma_centering(reader: &mut BitReader<'_>) -> Result<u8, NativeError> {
    let value = read_u8(reader, 3)?;
    Ok(if matches!(value, 5 | 6) { 7 } else { value })
}
