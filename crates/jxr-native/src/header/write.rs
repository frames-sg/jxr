//! T.832 `IMAGE_HEADER` and `IMAGE_PLANE_HEADER` writing.

use jxr_core::{ChromaSampling, ColorFormat};

use crate::{
    NativeError,
    bit_writer::BitWriter,
    metadata::{band_presence, internal_color, output_color, overlap_mode},
};

use super::{
    fields::{carries_float_parameters, carries_shift_bits, reserved_bit_depth},
    image::{GDI_SIGNATURE, inferred_margin},
    types::{CodestreamHeader, ImagePlaneHeader, QuantizerSet},
};

/// Largest dimension a short header's 16-bit `*_MINUS1` fields can carry.
const SHORT_HEADER_MAX_DIMENSION: u32 = 1 << 16;
/// Largest tile count per axis: `NUM_*_TILES_MINUS1` is twelve bits.
const MAX_TILES_PER_AXIS: usize = 1 << 12;

/// Writes the image header followed by its primary and optional alpha plane
/// headers, exactly as [`super::parse_codestream_headers`] reads them.
pub(crate) fn write_codestream_headers(
    writer: &mut BitWriter,
    image: &CodestreamHeader,
    primary: &ImagePlaneHeader,
    alpha: Option<&ImagePlaneHeader>,
) -> Result<(), NativeError> {
    if image.flags.alpha_plane() != alpha.is_some() {
        return Err(invalid("ALPHA_IMAGE_PLANE_FLAG"));
    }
    write_image_header(writer, image)?;
    write_plane_header(writer, primary, image.output_bit_depth)?;
    if let Some(alpha) = alpha {
        write_plane_header(writer, alpha, image.output_bit_depth)?;
    }
    Ok(())
}

fn write_image_header(writer: &mut BitWriter, image: &CodestreamHeader) -> Result<(), NativeError> {
    let short_header = image.flags.short_header();
    let dimension_bits = if short_header { 16 } else { 32 };
    let tile_bits = if short_header { 8 } else { 16 };
    for (dimension, field) in [
        (image.width, "WIDTH_MINUS1"),
        (image.height, "HEIGHT_MINUS1"),
    ] {
        if dimension == 0 || (short_header && dimension > SHORT_HEADER_MAX_DIMENSION) {
            return Err(invalid(field));
        }
    }
    overlap_mode(image.overlap_mode).map_err(|_| invalid("OVERLAP_MODE"))?;
    if image.orientation > 7 {
        return Err(invalid("SPATIAL_XFRM_SUBORDINATE"));
    }
    if output_color(image.output_color_format, 0).is_err()
        || reserved_bit_depth(image.output_bit_depth)
    {
        return Err(invalid("OUTPUT_CLR_FMT or OUTPUT_BITDEPTH"));
    }
    if image.requires_index() && !image.flags.index_table_present() {
        return Err(invalid("INDEX_TABLE_PRESENT_FLAG"));
    }
    for sizes in [&image.tile_widths_mb, &image.tile_heights_mb] {
        let oversized = sizes
            .iter()
            .any(|&size| size == 0 || (short_header && size > 0xff));
        if sizes.len() >= MAX_TILES_PER_AXIS || oversized {
            return Err(invalid("tile partition"));
        }
    }
    let inferred = [
        0,
        0,
        inferred_margin(image.height),
        inferred_margin(image.width),
    ];
    let windowing = image.margins != inferred;
    if image.margins.iter().any(|&margin| margin > 0x3f) {
        return Err(invalid("windowing margins"));
    }

    writer.write_bits(GDI_SIGNATURE, 64);
    writer.write_bits(1, 4); // RESERVED_B
    writer.write_flag(image.flags.hard_tiling());
    writer.write_bits(1, 3); // RESERVED_C
    writer.write_flag(image.tiled());
    writer.write_flag(image.flags.frequency_mode());
    writer.write_bits(u64::from(image.orientation), 3);
    writer.write_flag(image.flags.index_table_present());
    writer.write_bits(u64::from(image.overlap_mode), 2);
    writer.write_flag(short_header);
    writer.write_flag(image.flags.long_word());
    writer.write_flag(windowing);
    writer.write_flag(image.flags.trim_flexbits());
    writer.write_flag(false); // RESERVED_D
    writer.write_flag(image.flags.red_blue_not_swapped());
    writer.write_flag(image.flags.premultiplied_alpha());
    writer.write_flag(image.flags.alpha_plane());
    writer.write_bits(u64::from(image.output_color_format), 4);
    writer.write_bits(u64::from(image.output_bit_depth), 4);
    writer.write_bits(u64::from(image.width - 1), dimension_bits);
    writer.write_bits(u64::from(image.height - 1), dimension_bits);
    if image.tiled() {
        writer.write_bits(image.tile_widths_mb.len() as u64, 12);
        writer.write_bits(image.tile_heights_mb.len() as u64, 12);
        for &size in image.tile_widths_mb.iter().chain(&image.tile_heights_mb) {
            writer.write_bits(u64::from(size), tile_bits);
        }
    }
    if windowing {
        for margin in image.margins {
            writer.write_bits(u64::from(margin), 6);
        }
    }
    Ok(())
}

fn write_plane_header(
    writer: &mut BitWriter,
    plane: &ImagePlaneHeader,
    output_bit_depth: u8,
) -> Result<(), NativeError> {
    let bands = band_presence(plane.bands_present).map_err(|_| invalid("BANDS_PRESENT"))?;
    writer.write_bits(u64::from(plane.internal_color_format), 3);
    writer.write_flag(plane.scaled);
    writer.write_bits(u64::from(plane.bands_present), 4);
    write_component_layout(writer, plane)?;
    if carries_shift_bits(output_bit_depth) {
        writer.write_bits(u64::from(plane.shift_bits), 8);
    }
    if carries_float_parameters(output_bit_depth) {
        writer.write_bits(u64::from(plane.mantissa_length), 8);
        writer.write_bits(u64::from(plane.exponent_bias.cast_unsigned()), 8);
    }
    write_optional_quantizers(writer, plane.dc_quantizers.as_ref(), plane.components)?;
    if bands.has_low_pass() {
        writer.write_flag(false); // RESERVED_I
        write_optional_quantizers(writer, plane.lp_quantizers.as_ref(), plane.components)?;
        if bands.has_high_pass() {
            writer.write_flag(false); // RESERVED_J
            write_optional_quantizers(writer, plane.hp_quantizers.as_ref(), plane.components)?;
        }
    }
    writer.align_zero();
    Ok(())
}

fn write_component_layout(
    writer: &mut BitWriter,
    plane: &ImagePlaneHeader,
) -> Result<(), NativeError> {
    let color = internal_color(plane.internal_color_format, plane.components)
        .map_err(|_| invalid("INTERNAL_CLR_FMT"))?;
    if color.component_count() != Some(plane.components) {
        return Err(invalid("component count"));
    }
    match color {
        ColorFormat::Yuv(ChromaSampling::Cs420) => {
            writer.write_flag(false); // RESERVED_E
            writer.write_bits(u64::from(plane.chroma_centering_x), 3);
            writer.write_flag(false); // RESERVED_G
            writer.write_bits(u64::from(plane.chroma_centering_y), 3);
        }
        ColorFormat::Yuv(ChromaSampling::Cs422) => {
            writer.write_flag(false); // RESERVED_E
            writer.write_bits(u64::from(plane.chroma_centering_x), 3);
            writer.write_bits(0, 4); // RESERVED_H
        }
        ColorFormat::Yuv(ChromaSampling::Cs444) => {
            writer.write_bits(0, 4); // RESERVED_F
            writer.write_bits(0, 4); // RESERVED_H
        }
        ColorFormat::NComponent(_) => match plane.components {
            1..=15 => {
                writer.write_bits(u64::from(plane.components - 1), 4);
                writer.write_bits(0, 4); // RESERVED_H
            }
            16..=4111 => {
                writer.write_bits(15, 4);
                writer.write_bits(u64::from(plane.components - 16), 12);
            }
            _ => return Err(invalid("component count")),
        },
        _ => {}
    }
    Ok(())
}

/// Writes a `*_IMAGE_PLANE_UNIFORM_FLAG` and, when set, its quantizer set.
fn write_optional_quantizers(
    writer: &mut BitWriter,
    quantizers: Option<&QuantizerSet>,
    components: u16,
) -> Result<(), NativeError> {
    writer.write_flag(quantizers.is_some());
    let Some(quantizers) = quantizers else {
        return Ok(());
    };
    let values = &quantizers.components;
    if values.len() != usize::from(components) {
        return Err(invalid("quantizer component count"));
    }
    if components == 1 {
        writer.write_bits(u64::from(values[0]), 8);
        return Ok(());
    }
    // Pick the shortest COMPONENT_MODE that represents the set.
    let chroma_uniform = values[1..].iter().all(|&value| value == values[1]);
    if chroma_uniform && values[0] == values[1] {
        writer.write_bits(0, 2);
        writer.write_bits(u64::from(values[0]), 8);
    } else if chroma_uniform {
        writer.write_bits(1, 2);
        writer.write_bits(u64::from(values[0]), 8);
        writer.write_bits(u64::from(values[1]), 8);
    } else {
        writer.write_bits(2, 2);
        for &value in values {
            writer.write_bits(u64::from(value), 8);
        }
    }
    Ok(())
}

const fn invalid(field: &'static str) -> NativeError {
    NativeError::InvalidSyntax { field }
}

#[cfg(test)]
mod tests {
    use super::super::{HeaderFlags, ParsedHeaders, parse_codestream_headers};
    use super::*;

    fn flags(values: [bool; 9]) -> HeaderFlags {
        HeaderFlags::from_parsed(values)
    }

    #[expect(
        clippy::unnecessary_wraps,
        reason = "every use fills an optional header field"
    )]
    fn set(values: &[u8]) -> Option<QuantizerSet> {
        Some(QuantizerSet {
            components: values.to_vec(),
        })
    }

    fn round_trip(
        image: &CodestreamHeader,
        primary: &ImagePlaneHeader,
        alpha: Option<&ImagePlaneHeader>,
    ) {
        let mut writer = BitWriter::new();
        write_codestream_headers(&mut writer, image, primary, alpha).unwrap();
        let bytes = writer.into_bytes();
        assert_eq!(
            parse_codestream_headers(&bytes).unwrap(),
            ParsedHeaders {
                image: image.clone(),
                primary: primary.clone(),
                alpha: alpha.cloned(),
                bytes_consumed: bytes.len(),
            }
        );
    }

    fn gray_plane() -> ImagePlaneHeader {
        ImagePlaneHeader {
            internal_color_format: 0,
            scaled: false,
            bands_present: 0,
            components: 1,
            chroma_centering_x: 0,
            chroma_centering_y: 0,
            shift_bits: 0,
            mantissa_length: 0,
            exponent_bias: 0,
            dc_quantizers: set(&[1]),
            lp_quantizers: set(&[1]),
            hp_quantizers: set(&[1]),
        }
    }

    fn gray_image() -> CodestreamHeader {
        CodestreamHeader {
            width: 37,
            height: 21,
            flags: flags([false, false, false, true, true, false, false, false, false]),
            orientation: 0,
            overlap_mode: 1,
            output_color_format: 0,
            output_bit_depth: 1,
            tile_widths_mb: Vec::new(),
            tile_heights_mb: Vec::new(),
            margins: [0, 0, 11, 11],
        }
    }

    #[test]
    fn single_tile_gray_headers_round_trip() {
        round_trip(&gray_image(), &gray_plane(), None);
    }

    #[test]
    fn long_header_tiles_windowing_and_alpha_round_trip() {
        let image = CodestreamHeader {
            width: 70_000,
            height: 4_100,
            flags: flags([true, true, true, true, false, true, true, true, true]),
            orientation: 5,
            overlap_mode: 2,
            output_color_format: 7,
            output_bit_depth: 2,
            tile_widths_mb: vec![300, 1000, 7],
            tile_heights_mb: vec![64],
            margins: [3, 0, 17, 63],
        };
        let primary = ImagePlaneHeader {
            internal_color_format: 3,
            scaled: true,
            bands_present: 1,
            components: 3,
            shift_bits: 4,
            dc_quantizers: set(&[9, 9, 9]),
            lp_quantizers: set(&[9, 12, 12]),
            hp_quantizers: None,
            ..gray_plane()
        };
        let alpha = ImagePlaneHeader {
            shift_bits: 2,
            bands_present: 3,
            dc_quantizers: None,
            lp_quantizers: None,
            hp_quantizers: None,
            ..gray_plane()
        };
        round_trip(&image, &primary, Some(&alpha));
    }

    #[test]
    fn every_internal_colour_layout_round_trips() {
        let image = CodestreamHeader {
            output_bit_depth: 7,
            ..gray_image()
        };
        let layouts: [(u8, u16, u8, u8); 7] = [
            (1, 3, 3, 4),
            (2, 3, 2, 0),
            (3, 3, 0, 0),
            (4, 4, 0, 0),
            (6, 2, 0, 0),
            (6, 15, 0, 0),
            (6, 16, 0, 0),
        ];
        for (color, components, x, y) in layouts {
            let independent: Vec<u8> = (20..).take(usize::from(components)).collect();
            let plane = ImagePlaneHeader {
                internal_color_format: color,
                components,
                chroma_centering_x: x,
                chroma_centering_y: y,
                mantissa_length: 13,
                exponent_bias: -4,
                bands_present: 2,
                dc_quantizers: set(&independent),
                lp_quantizers: None,
                hp_quantizers: None,
                ..gray_plane()
            };
            round_trip(&image, &plane, None);
        }
    }

    #[test]
    fn inconsistent_headers_are_rejected() {
        let mut writer = BitWriter::new();
        let oversized = CodestreamHeader {
            width: 70_000,
            ..gray_image()
        };
        assert_eq!(
            write_codestream_headers(&mut writer, &oversized, &gray_plane(), None),
            Err(invalid("WIDTH_MINUS1"))
        );
        let wrong_components = ImagePlaneHeader {
            components: 3,
            ..gray_plane()
        };
        assert_eq!(
            write_codestream_headers(&mut writer, &gray_image(), &wrong_components, None),
            Err(invalid("component count"))
        );
        assert_eq!(
            write_codestream_headers(
                &mut writer,
                &gray_image(),
                &gray_plane(),
                Some(&gray_plane())
            ),
            Err(invalid("ALPHA_IMAGE_PLANE_FLAG"))
        );
    }

    #[test]
    fn frequency_mode_and_tiling_need_the_tile_index() {
        let frequency = |index| CodestreamHeader {
            flags: flags([false, true, index, true, true, false, false, false, false]),
            ..gray_image()
        };
        let tiled = |index| CodestreamHeader {
            flags: flags([false, false, index, true, true, false, false, false, false]),
            tile_widths_mb: vec![1],
            ..gray_image()
        };
        for image in [frequency, tiled] {
            let mut writer = BitWriter::new();
            assert_eq!(
                write_codestream_headers(&mut writer, &image(false), &gray_plane(), None),
                Err(invalid("INDEX_TABLE_PRESENT_FLAG"))
            );
            round_trip(&image(true), &gray_plane(), None);
        }
    }
}
