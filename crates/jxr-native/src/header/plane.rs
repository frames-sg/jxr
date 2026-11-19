//! T.832 `IMAGE_PLANE_HEADER` parsing.

use jxr_core::{BandPresence, ChromaSampling, ColorFormat};

use crate::{
    NativeError,
    bit_reader::{BitReader, header_error},
    metadata::{band_presence, internal_color},
    tile_decode::quantizer::read_quantizer_set,
};

use super::{
    fields::{carries_float_parameters, carries_shift_bits, read_u8, read_u16},
    types::{ImagePlaneHeader, QuantizerSet},
};

pub(super) fn parse_plane_header(
    reader: &mut BitReader<'_>,
    output_bit_depth: u8,
) -> Result<ImagePlaneHeader, NativeError> {
    let internal_color_format = read_u8(reader, 3)?;
    let color = internal_color(internal_color_format, 0)?;
    let scaled = reader.read_flag()?;
    let bands_present = read_u8(reader, 4)?;
    let bands = band_presence(bands_present)?;
    let (components, chroma_centering_x, chroma_centering_y) =
        parse_component_layout(reader, color)?;
    let shift_bits = if carries_shift_bits(output_bit_depth) {
        read_u8(reader, 8)?
    } else {
        0
    };
    let (mantissa_length, exponent_bias) = parse_float_parameters(reader, output_bit_depth)?;
    let dc_quantizers = parse_optional_quantizers(reader, components)?;
    let (lp_quantizers, hp_quantizers) = parse_high_band_quantizers(reader, components, bands)?;
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

fn parse_float_parameters(
    reader: &mut BitReader<'_>,
    output_bit_depth: u8,
) -> Result<(u8, i8), NativeError> {
    if carries_float_parameters(output_bit_depth) {
        Ok((read_u8(reader, 8)?, read_u8(reader, 8)?.cast_signed()))
    } else {
        Ok((0, 0))
    }
}

/// The components of a plane coded in `color`, and its chroma centering.
fn parse_component_layout(
    reader: &mut BitReader<'_>,
    color: ColorFormat,
) -> Result<(u16, u8, u8), NativeError> {
    let (x, y) = match color {
        ColorFormat::NComponent(_) => return Ok((parse_component_count(reader)?, 0, 0)),
        ColorFormat::Yuv(ChromaSampling::Cs420) => {
            let _reserved_e = reader.read_flag()?;
            let x = read_chroma_centering(reader)?;
            let _reserved_g = reader.read_flag()?;
            (x, read_chroma_centering(reader)?)
        }
        ColorFormat::Yuv(ChromaSampling::Cs422) => {
            let _reserved_e = reader.read_flag()?;
            let x = read_chroma_centering(reader)?;
            let _reserved_h = reader.read_bits(4)?;
            (x, 0)
        }
        ColorFormat::Yuv(ChromaSampling::Cs444) => {
            let _reserved_f = reader.read_bits(4)?;
            let _reserved_h = reader.read_bits(4)?;
            (0, 0)
        }
        _ => (0, 0),
    };
    let components = color
        .component_count()
        .expect("only an N-component plane counts its components");
    Ok((components, x, y))
}

/// `NUM_COMPONENTS_MINUS1` of an N-component plane, plus one.
fn parse_component_count(reader: &mut BitReader<'_>) -> Result<u16, NativeError> {
    let short_count = read_u16(reader, 4)?;
    if short_count == 15 {
        read_u16(reader, 12)?
            .checked_add(16)
            .ok_or(NativeError::IntegerOverflow {
                operation: "computing component count",
            })
    } else {
        let _reserved_h = reader.read_bits(4)?;
        Ok(short_count + 1)
    }
}

fn parse_high_band_quantizers(
    reader: &mut BitReader<'_>,
    components: u16,
    bands: BandPresence,
) -> Result<(Option<QuantizerSet>, Option<QuantizerSet>), NativeError> {
    if !bands.has_low_pass() {
        return Ok((None, None));
    }
    let _reserved_i = reader.read_flag()?;
    let low_pass = parse_optional_quantizers(reader, components)?;
    if !bands.has_high_pass() {
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
    let components = read_quantizer_set(reader.packet(), components)
        .map_err(|error| header_error(&error))?
        .ok_or(NativeError::ReservedValue {
            field: "COMPONENT_MODE",
            value: 3,
        })?;
    Ok(QuantizerSet { components })
}

fn read_chroma_centering(reader: &mut BitReader<'_>) -> Result<u8, NativeError> {
    let value = read_u8(reader, 3)?;
    Ok(if matches!(value, 5 | 6) { 7 } else { value })
}
