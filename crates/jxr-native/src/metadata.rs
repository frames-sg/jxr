//! Conversion from parsed T.832 syntax to public image metadata.

use jxr_core::{
    AlphaMode, BandPresence, BitstreamMode, ByteRange, ChromaSampling, ColorFormat, ImageInfo,
    ImageMetadata, Level, Orientation, OverlapMode, PlaneInfo, Profile, SampleFormat, TileGrid,
};

use crate::classify_annex_a_pixel_format;
use crate::{CodestreamHeader, ImagePlaneHeader, NativeError, ParsedCodestream};

/// Build validated public metadata from a parsed codestream.
pub fn image_info(parsed: &ParsedCodestream) -> Result<ImageInfo, NativeError> {
    let image = &parsed.headers.image;
    let primary = plane_info(image, &parsed.headers.primary)?;
    let (alpha_mode, alpha) = alpha_info(parsed, image)?;
    if alpha_mode == AlphaMode::Integrated
        && alpha
            .as_ref()
            .is_some_and(|alpha| !bands_are_subset(primary.bands, alpha.bands))
    {
        return Err(NativeError::InvalidSyntax {
            field: "integrated alpha bands exceed primary bands",
        });
    }
    let (profile, level) = profile_and_level(parsed);
    let info = ImageInfo {
        width: image.width,
        height: image.height,
        profile,
        level,
        primary,
        alpha_mode,
        premultiplied_alpha: crate::pixel_format::source_is_premultiplied(parsed),
        alpha,
        tiles: tile_grid(image)?,
        metadata: metadata(parsed)?,
    };
    info.validate_consistency()
        .map_err(|_| NativeError::InvalidSyntax {
            field: "public image metadata",
        })?;
    Ok(info)
}

const fn bands_are_subset(primary: BandPresence, alpha: BandPresence) -> bool {
    (!alpha.has_low_pass() || primary.has_low_pass())
        && (!alpha.has_high_pass() || primary.has_high_pass())
        && (!alpha.has_flexbits() || primary.has_flexbits())
}

fn alpha_info(
    parsed: &ParsedCodestream,
    image: &CodestreamHeader,
) -> Result<(AlphaMode, Option<PlaneInfo>), NativeError> {
    if let Some(headers) = &parsed.separate_alpha_headers {
        let plane = headers.alpha.as_ref().unwrap_or(&headers.primary);
        return Ok((
            AlphaMode::Separate,
            Some(plane_info(&headers.image, plane)?),
        ));
    }
    if let Some(alpha) = &parsed.headers.alpha {
        return Ok((AlphaMode::Integrated, Some(plane_info(image, alpha)?)));
    }
    Ok((AlphaMode::None, None))
}

fn plane_info(
    image: &CodestreamHeader,
    plane: &ImagePlaneHeader,
) -> Result<PlaneInfo, NativeError> {
    Ok(PlaneInfo {
        color_format: internal_color(plane.internal_color_format, plane.components)?,
        sample_format: sample_format(image.output_bit_depth)?,
        bands: band_presence(plane.bands_present)?,
        bitstream_mode: if image.flags.frequency_mode() {
            BitstreamMode::Frequency
        } else {
            BitstreamMode::Spatial
        },
        overlap: overlap_mode(image.overlap_mode)?,
        short_header: image.flags.short_header(),
        long_word: image.flags.long_word(),
        scaled: plane.scaled,
        chroma_centering: [plane.chroma_centering_x, plane.chroma_centering_y],
        shift_bits: plane.shift_bits,
        mantissa_length: plane.mantissa_length,
        exponent_bias: plane.exponent_bias,
        width: image.width,
        height: image.height,
    })
}

fn tile_grid(image: &CodestreamHeader) -> Result<TileGrid, NativeError> {
    let extended_width = extended_dimension(image.width, image.margins[1], image.margins[3])?;
    let extended_height = extended_dimension(image.height, image.margins[0], image.margins[2])?;
    let columns = complete_tiles(&image.tile_widths_mb, extended_width / 16, "tile columns")?;
    let rows = complete_tiles(&image.tile_heights_mb, extended_height / 16, "tile rows")?;
    Ok(TileGrid {
        column_widths: columns,
        row_heights: rows,
        hard_tiles: image.flags.hard_tiling(),
    })
}

fn extended_dimension(dimension: u32, first: u8, second: u8) -> Result<u32, NativeError> {
    let extended = dimension
        .checked_add(u32::from(first))
        .and_then(|value| value.checked_add(u32::from(second)))
        .ok_or(NativeError::IntegerOverflow {
            operation: "computing extended image dimension",
        })?;
    if !extended.is_multiple_of(16) {
        return Err(NativeError::InvalidSyntax {
            field: "extended image dimension alignment",
        });
    }
    Ok(extended)
}

fn complete_tiles(
    explicit: &[u16],
    total: u32,
    field: &'static str,
) -> Result<Vec<u32>, NativeError> {
    let mut result = Vec::with_capacity(explicit.len() + 1);
    let mut used = 0_u32;
    for &size in explicit {
        if size == 0 {
            return Err(NativeError::InvalidSyntax { field });
        }
        used = used
            .checked_add(u32::from(size))
            .ok_or(NativeError::IntegerOverflow {
                operation: "summing tile dimensions",
            })?;
        result.push(u32::from(size));
    }
    let final_size = total
        .checked_sub(used)
        .ok_or(NativeError::InvalidSyntax { field })?;
    if final_size == 0 {
        return Err(NativeError::InvalidSyntax { field });
    }
    result.push(final_size);
    Ok(result)
}

fn metadata(parsed: &ParsedCodestream) -> Result<ImageMetadata, NativeError> {
    let icc_profile = parsed
        .annex_a
        .as_ref()
        .and_then(|annex| annex.metadata.icc_profile_range.clone())
        .map(|range| ByteRange::new(range.start, range.len(), usize::MAX))
        .transpose()
        .map_err(|_| NativeError::IntegerOverflow {
            operation: "representing ICC profile range",
        })?;
    let orientation_value = parsed
        .annex_a
        .as_ref()
        .and_then(|annex| annex.metadata.transformation)
        .unwrap_or(u32::from(parsed.headers.image.orientation));
    let orientation_value = u8::try_from(orientation_value)
        .ok()
        .filter(|&value| value < 8)
        .ok_or(NativeError::ReservedValue {
            field: "Annex-A spatial transformation",
            value: u64::from(orientation_value),
        })?;
    let container_pixel_format = parsed.annex_a.as_ref().map(|annex| annex.pixel_format_guid);
    Ok(ImageMetadata {
        orientation: orientation(orientation_value),
        icc_profile,
        container_pixel_format,
        annex_a_pixel_format: container_pixel_format.map(classify_annex_a_pixel_format),
    })
}

fn profile_and_level(parsed: &ParsedCodestream) -> (Option<Profile>, Option<Level>) {
    let Some(declaration) = parsed.directory.profiles.first() else {
        return (Some(Profile::Advanced), Some(Level(255)));
    };
    let profile = match declaration.profile_idc {
        44 => Profile::SubBaseline,
        55 => Profile::Baseline,
        66 => Profile::Main,
        111 => Profile::Advanced,
        value => Profile::Unknown(value),
    };
    (Some(profile), Some(Level(declaration.level_idc)))
}

const fn orientation(value: u8) -> Orientation {
    match value {
        0 => Orientation::Identity,
        1 => Orientation::MirrorVertical,
        2 => Orientation::MirrorHorizontal,
        3 => Orientation::Rotate180,
        4 => Orientation::Rotate90,
        5 => Orientation::Transverse,
        6 => Orientation::Transpose,
        _ => Orientation::Rotate270,
    }
}

fn internal_color(code: u8, components: u16) -> Result<ColorFormat, NativeError> {
    match code {
        0 => Ok(ColorFormat::Luma),
        1 => Ok(ColorFormat::Yuv(ChromaSampling::Cs420)),
        2 => Ok(ColorFormat::Yuv(ChromaSampling::Cs422)),
        3 => Ok(ColorFormat::Yuv(ChromaSampling::Cs444)),
        4 => Ok(ColorFormat::YuvK),
        6 => Ok(ColorFormat::NComponent(components)),
        value => Err(NativeError::ReservedValue {
            field: "INTERNAL_CLR_FMT",
            value: u64::from(value),
        }),
    }
}

fn sample_format(code: u8) -> Result<SampleFormat, NativeError> {
    match code {
        0 | 15 => Ok(SampleFormat::Bit1),
        1 => Ok(SampleFormat::Unsigned { bits: 8 }),
        2 => Ok(SampleFormat::Unsigned { bits: 16 }),
        3 => Ok(SampleFormat::Signed { bits: 16 }),
        4 => Ok(SampleFormat::Float16),
        6 => Ok(SampleFormat::Signed { bits: 32 }),
        7 => Ok(SampleFormat::Float32),
        8 => Ok(SampleFormat::Unsigned { bits: 5 }),
        9 => Ok(SampleFormat::Unsigned { bits: 10 }),
        10 => Ok(SampleFormat::Unsigned { bits: 6 }),
        value => Err(NativeError::ReservedValue {
            field: "OUTPUT_BITDEPTH",
            value: u64::from(value),
        }),
    }
}

fn band_presence(code: u8) -> Result<BandPresence, NativeError> {
    match code {
        0 => Ok(BandPresence::All),
        1 => Ok(BandPresence::NoFlexbits),
        2 => Ok(BandPresence::NoHighPass),
        3 => Ok(BandPresence::DcOnly),
        value => Err(NativeError::ReservedValue {
            field: "BANDS_PRESENT",
            value: u64::from(value),
        }),
    }
}

fn overlap_mode(code: u8) -> Result<OverlapMode, NativeError> {
    match code {
        0 => Ok(OverlapMode::None),
        1 => Ok(OverlapMode::One),
        2 => Ok(OverlapMode::Two),
        value => Err(NativeError::ReservedValue {
            field: "OVERLAP_MODE",
            value: u64::from(value),
        }),
    }
}
