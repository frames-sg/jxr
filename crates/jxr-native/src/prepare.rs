//! Device-neutral decode-plan construction from parsed syntax.

use jxr_core::{
    AlphaMode, BandPresence, ByteRange, ChromaSampling, ColorFormat, DecodeRequest, DecodeScale,
    JxrError, JxrErrorKind, PixelFormat, PlanePlan, PreparedPlan, Rect, TilePlan,
};

use crate::{CodestreamDirectory, NativeError, ParsedCodestream, image_info};

/// Validate a request and construct immutable tile/reconstruction geometry.
pub fn prepare_plan(
    input_len: usize,
    parsed: &ParsedCodestream,
    request: &DecodeRequest,
) -> Result<PreparedPlan, JxrError> {
    let info = image_info(parsed).map_err(|error| native_plan_error(&error))?;
    validate_request(input_len, parsed, &info, request)?;
    let output_region = request
        .region
        .unwrap_or_else(|| Rect::full(info.dimensions()));
    if output_region.w == 0 || output_region.h == 0 || !output_region.is_within(info.dimensions()) {
        return Err(JxrError::new(JxrErrorKind::InvalidRequest, "decode region"));
    }
    let decoded_region = native_output_region(output_region, request.scale)?;
    let host_output_bytes = host_output_bytes(
        request.output,
        parsed.headers.image.output_color_format,
        info.alpha.is_some() && request.alpha != jxr_core::AlphaHandling::Drop,
        decoded_region,
    )?;
    request
        .limits
        .check_host_allocation_bytes(host_output_bytes)?;
    let reconstruction_region =
        reconstruction_region(output_region, info.dimensions(), info.primary.overlap);
    let primary = plane_plan(&info, 0);
    let alpha = should_prepare_alpha(info.alpha_mode, request.alpha)
        .then(|| {
            info.alpha
                .as_ref()
                .map(|_| plane_plan(&info, alpha_coefficient_plane(&info)))
        })
        .flatten();
    let tiles = tile_plans(parsed, &info, output_region)?;
    let selected_macroblocks = tiles
        .iter()
        .filter(|tile| tile.required_for_reconstruction)
        .try_fold(0_u32, |total, tile| {
            total
                .checked_add(tile.macroblock_count)
                .ok_or_else(|| JxrError::arithmetic("selected macroblock count"))
        })?;
    let coefficient_bytes = coefficient_bytes(
        &info,
        alpha.is_some() || info.alpha_mode == AlphaMode::Integrated,
        selected_macroblocks,
        request.scale,
    )?;
    request.limits.check_coefficient_bytes(coefficient_bytes)?;
    let codestream_range = ByteRange::new(
        parsed.codestream_range.start,
        parsed.codestream_range.len(),
        input_len,
    )?;
    Ok(PreparedPlan {
        info,
        codestream_range,
        primary,
        alpha,
        tiles,
        reconstruction_region,
        output_region,
        decoded_region,
        scale: request.scale,
        coefficient_bytes,
    })
}

const fn should_prepare_alpha(mode: AlphaMode, handling: jxr_core::AlphaHandling) -> bool {
    matches!(mode, AlphaMode::Integrated)
        || (matches!(mode, AlphaMode::Separate)
            && !matches!(handling, jxr_core::AlphaHandling::Drop))
}

fn host_output_bytes(
    format: PixelFormat,
    output_color_format: u8,
    include_alpha: bool,
    region: Rect,
) -> Result<usize, JxrError> {
    let rows = usize::try_from(region.h).map_err(|_| JxrError::arithmetic("output row count"))?;
    if !matches!(output_color_format, 1 | 2) {
        return format
            .row_bytes(region.w)?
            .checked_mul(rows)
            .ok_or_else(|| JxrError::arithmetic("host output allocation"));
    }
    let luma = format
        .row_bytes_for_channels(region.w, 1)?
        .checked_mul(rows)
        .ok_or_else(|| JxrError::arithmetic("planar luma allocation"))?;
    let chroma_height = if output_color_format == 1 {
        region.h / 2
    } else {
        region.h
    };
    let chroma_rows =
        usize::try_from(chroma_height).map_err(|_| JxrError::arithmetic("chroma row count"))?;
    let one_chroma = format
        .row_bytes_for_channels(region.w / 2, 1)?
        .checked_mul(chroma_rows)
        .ok_or_else(|| JxrError::arithmetic("planar chroma allocation"))?;
    let alpha = if include_alpha { luma } else { 0 };
    luma.checked_add(
        one_chroma
            .checked_mul(2)
            .ok_or_else(|| JxrError::arithmetic("planar chroma allocation"))?,
    )
    .and_then(|bytes| bytes.checked_add(alpha))
    .ok_or_else(|| JxrError::arithmetic("planar host output allocation"))
}

fn alpha_coefficient_plane(info: &jxr_core::ImageInfo) -> usize {
    match info.alpha_mode {
        AlphaMode::Integrated => info
            .primary
            .color_format
            .component_count()
            .map_or(0, usize::from),
        AlphaMode::Separate | AlphaMode::None => 0,
    }
}

fn validate_request(
    input_len: usize,
    parsed: &ParsedCodestream,
    info: &jxr_core::ImageInfo,
    request: &DecodeRequest,
) -> Result<(), JxrError> {
    request.limits.check_compressed_bytes(input_len)?;
    request.limits.check_dimensions(info.width, info.height)?;
    request
        .limits
        .check_components(info.primary.color_format.component_count().unwrap_or(0))?;
    request.limits.check_tiles(info.tiles.tile_count()?)?;
    validate_native_scale(parsed, info, request)
}

fn validate_native_scale(
    parsed: &ParsedCodestream,
    info: &jxr_core::ImageInfo,
    request: &DecodeRequest,
) -> Result<(), JxrError> {
    if request.scale == DecodeScale::Full {
        return Ok(());
    }
    if info.primary.bitstream_mode != jxr_core::BitstreamMode::Frequency {
        return Err(JxrError::new(
            JxrErrorKind::Unsupported,
            "native reduced resolution requires frequency-mode packets",
        ));
    }
    if request.scale == DecodeScale::Quarter && !info.primary.bands.has_low_pass() {
        return Err(JxrError::new(
            JxrErrorKind::Unsupported,
            "quarter-resolution decode requires an LP band",
        ));
    }
    if request.scale == DecodeScale::Quarter
        && request.alpha != jxr_core::AlphaHandling::Drop
        && info
            .alpha
            .as_ref()
            .is_some_and(|alpha| !alpha.bands.has_low_pass())
    {
        return Err(JxrError::new(
            JxrErrorKind::Unsupported,
            "quarter-resolution alpha decode requires an LP band",
        ));
    }
    if request.scale == DecodeScale::Sixteenth
        && matches!(parsed.headers.image.output_color_format, 1 | 2)
    {
        return Err(JxrError::new(
            JxrErrorKind::Unsupported,
            "sixteenth-resolution subsampled planar YUV output",
        ));
    }
    Ok(())
}

fn native_output_region(region: Rect, scale: DecodeScale) -> Result<Rect, JxrError> {
    let denominator = scale.denominator();
    let right = region
        .x
        .checked_add(region.w)
        .ok_or_else(|| JxrError::arithmetic("scaled output region right edge"))?;
    let bottom = region
        .y
        .checked_add(region.h)
        .ok_or_else(|| JxrError::arithmetic("scaled output region bottom edge"))?;
    let x = region.x / denominator;
    let y = region.y / denominator;
    let right = right.div_ceil(denominator);
    let bottom = bottom.div_ceil(denominator);
    Ok(Rect {
        x,
        y,
        w: right - x,
        h: bottom - y,
    })
}

fn plane_plan(info: &jxr_core::ImageInfo, coefficient_plane: usize) -> PlanePlan {
    let macroblocks_x = info.tiles.column_widths.iter().sum();
    let macroblocks_y = info.tiles.row_heights.iter().sum();
    PlanePlan {
        width: info.width,
        height: info.height,
        macroblocks_x,
        macroblocks_y,
        overlap: info.primary.overlap,
        coefficient_plane,
    }
}

fn reconstruction_region(
    region: Rect,
    dimensions: (u32, u32),
    overlap: jxr_core::OverlapMode,
) -> Rect {
    let halo = if matches!(overlap, jxr_core::OverlapMode::None) {
        0
    } else {
        16
    };
    let x0 = (region.x / 16) * 16;
    let y0 = (region.y / 16) * 16;
    let right = region
        .x
        .saturating_add(region.w)
        .div_ceil(16)
        .saturating_mul(16);
    let bottom = region
        .y
        .saturating_add(region.h)
        .div_ceil(16)
        .saturating_mul(16);
    let x = x0.saturating_sub(halo);
    let y = y0.saturating_sub(halo);
    let right = right.saturating_add(halo).min(dimensions.0);
    let bottom = bottom.saturating_add(halo).min(dimensions.1);
    Rect {
        x,
        y,
        w: right.saturating_sub(x),
        h: bottom.saturating_sub(y),
    }
}

fn tile_plans(
    parsed: &ParsedCodestream,
    info: &jxr_core::ImageInfo,
    output_region: Rect,
) -> Result<Vec<TilePlan>, JxrError> {
    let ranges = tile_packet_ranges(parsed, &parsed.directory)?;
    let bounds = coded_reconstruction_bounds(parsed, info, output_region)?;
    let mut plans = Vec::with_capacity(ranges.len());
    let mut macroblock_start = 0_u32;
    let mut y_mb = 0_u32;
    let mut tile_index = 0_usize;
    for &height_mb in &info.tiles.row_heights {
        let mut x_mb = 0_u32;
        for &width_mb in &info.tiles.column_widths {
            let count = width_mb
                .checked_mul(height_mb)
                .ok_or_else(|| JxrError::arithmetic("tile macroblock count"))?;
            plans.push(TilePlan {
                packet_range: ranges[tile_index],
                output_region: tile_output_region(parsed, x_mb, y_mb, width_mb, height_mb),
                macroblock_start,
                macroblock_count: count,
                hard_boundaries: info.tiles.hard_tiles,
                required_for_reconstruction: x_mb < bounds[2]
                    && x_mb.saturating_add(width_mb) > bounds[0]
                    && y_mb < bounds[3]
                    && y_mb.saturating_add(height_mb) > bounds[1],
            });
            macroblock_start = macroblock_start
                .checked_add(count)
                .ok_or_else(|| JxrError::arithmetic("tile macroblock offsets"))?;
            x_mb = x_mb.saturating_add(width_mb);
            tile_index += 1;
        }
        y_mb = y_mb.saturating_add(height_mb);
    }
    Ok(plans)
}

fn coded_reconstruction_bounds(
    parsed: &ParsedCodestream,
    info: &jxr_core::ImageInfo,
    output_region: Rect,
) -> Result<[u32; 4], JxrError> {
    reconstruction_bounds(
        [
            u32::from(parsed.headers.image.margins[1]),
            u32::from(parsed.headers.image.margins[0]),
        ],
        [
            info.tiles.column_widths.iter().sum::<u32>(),
            info.tiles.row_heights.iter().sum::<u32>(),
        ],
        info.primary.overlap,
        parsed.headers.primary.internal_color_format,
        parsed.headers.image.output_color_format,
        output_region,
    )
}

fn reconstruction_bounds(
    margins: [u32; 2],
    macroblocks: [u32; 2],
    overlap: jxr_core::OverlapMode,
    internal_color_format: u8,
    output_color_format: u8,
    output_region: Rect,
) -> Result<[u32; 4], JxrError> {
    let start_x = margins[0]
        .checked_add(output_region.x)
        .ok_or_else(|| JxrError::arithmetic("region coded x"))?;
    let start_y = margins[1]
        .checked_add(output_region.y)
        .ok_or_else(|| JxrError::arithmetic("region coded y"))?;
    let end_x = start_x
        .checked_add(output_region.w)
        .ok_or_else(|| JxrError::arithmetic("region coded width"))?;
    let end_y = start_y
        .checked_add(output_region.h)
        .ok_or_else(|| JxrError::arithmetic("region coded height"))?;
    let chroma_upsampling =
        matches!(internal_color_format, 1 | 2) && !matches!(output_color_format, 1 | 2);
    let halo = u32::from(overlap != jxr_core::OverlapMode::None || chroma_upsampling);
    Ok([
        (start_x / 16).saturating_sub(halo),
        (start_y / 16).saturating_sub(halo),
        end_x.div_ceil(16).saturating_add(halo).min(macroblocks[0]),
        end_y.div_ceil(16).saturating_add(halo).min(macroblocks[1]),
    ])
}

fn tile_packet_ranges(
    parsed: &ParsedCodestream,
    directory: &CodestreamDirectory,
) -> Result<Vec<ByteRange>, JxrError> {
    let tile_count = (parsed.headers.image.tile_widths_mb.len() + 1)
        .checked_mul(parsed.headers.image.tile_heights_mb.len() + 1)
        .ok_or_else(|| JxrError::arithmetic("tile count"))?;
    if directory.tile_offsets.is_empty() {
        let offset = parsed.codestream_range.start + directory.tile_data_offset;
        return Ok(vec![ByteRange::new(
            offset,
            parsed.codestream_range.end - offset,
            parsed.codestream_range.end,
        )?]);
    }
    let base = parsed.codestream_range.start + directory.tile_data_offset;
    indexed_tile_packet_ranges(
        base,
        parsed.codestream_range.end,
        &directory.tile_offsets,
        tile_count,
        parsed.headers.image.flags.frequency_mode(),
    )
}

fn indexed_tile_packet_ranges(
    base: usize,
    codestream_end: usize,
    offsets: &[u64],
    tile_count: usize,
    frequency_mode: bool,
) -> Result<Vec<ByteRange>, JxrError> {
    if tile_count == 0 {
        return Err(JxrError::new(
            JxrErrorKind::InternalInvariant,
            "empty tile index",
        ));
    }
    let bands = offsets.len() / tile_count;
    if bands == 0 || bands * tile_count != offsets.len() {
        return Err(JxrError::new(
            JxrErrorKind::InvalidSyntax,
            "tile index shape",
        ));
    }
    if frequency_mode {
        let range = ByteRange::new(base, codestream_end.saturating_sub(base), codestream_end)?;
        return Ok(vec![range; tile_count]);
    }
    let mut starts = Vec::with_capacity(tile_count);
    for tile in 0..tile_count {
        let relative = usize::try_from(offsets[tile * bands])
            .map_err(|_| JxrError::arithmetic("tile packet offset"))?;
        starts.push(
            base.checked_add(relative)
                .ok_or_else(|| JxrError::arithmetic("tile packet position"))?,
        );
    }
    let mut physical = starts.clone();
    physical.sort_unstable();
    physical.dedup();
    starts
        .iter()
        .map(|&start| {
            let position = physical
                .binary_search(&start)
                .map_err(|_| JxrError::new(JxrErrorKind::InternalInvariant, "tile index lookup"))?;
            let end = physical
                .get(position + 1)
                .copied()
                .unwrap_or(codestream_end);
            let length = end
                .checked_sub(start)
                .ok_or_else(|| JxrError::new(JxrErrorKind::InvalidSyntax, "tile index ordering"))?;
            ByteRange::new(start, length, codestream_end)
        })
        .collect()
}

fn tile_output_region(
    parsed: &ParsedCodestream,
    x_mb: u32,
    y_mb: u32,
    width_mb: u32,
    height_mb: u32,
) -> Rect {
    let image = &parsed.headers.image;
    let left = x_mb.saturating_mul(16);
    let top = y_mb.saturating_mul(16);
    let right = left.saturating_add(width_mb.saturating_mul(16));
    let bottom = top.saturating_add(height_mb.saturating_mul(16));
    let margin_left = u32::from(image.margins[1]);
    let margin_top = u32::from(image.margins[0]);
    let x = left.saturating_sub(margin_left).min(image.width);
    let y = top.saturating_sub(margin_top).min(image.height);
    let right = right.saturating_sub(margin_left).min(image.width);
    let bottom = bottom.saturating_sub(margin_top).min(image.height);
    Rect {
        x,
        y,
        w: right.saturating_sub(x),
        h: bottom.saturating_sub(y),
    }
}

fn coefficient_bytes(
    info: &jxr_core::ImageInfo,
    include_alpha: bool,
    macroblocks: u32,
    scale: DecodeScale,
) -> Result<usize, JxrError> {
    let primary_per_mb = coefficients_per_macroblock(
        info.primary.color_format,
        scale.retained_bands(info.primary.bands),
    )?;
    let alpha_per_mb = if include_alpha {
        info.alpha
            .as_ref()
            .map(|alpha| {
                coefficients_per_macroblock(alpha.color_format, scale.retained_bands(alpha.bands))
            })
            .transpose()?
            .unwrap_or(0)
    } else {
        0
    };
    usize::try_from(macroblocks)
        .ok()
        .and_then(|count| count.checked_mul(primary_per_mb + alpha_per_mb))
        .and_then(|count| count.checked_mul(core::mem::size_of::<i32>()))
        .ok_or_else(|| JxrError::arithmetic("coefficient allocation size"))
}

fn coefficients_per_macroblock(color: ColorFormat, bands: BandPresence) -> Result<usize, JxrError> {
    let full = band_coefficient_count(bands, 1, 16, 256);
    let count = match color {
        ColorFormat::Luma => full,
        ColorFormat::Yuv(ChromaSampling::Cs420) => {
            full + 2 * band_coefficient_count(bands, 1, 4, 64)
        }
        ColorFormat::Yuv(ChromaSampling::Cs422) => {
            full + 2 * band_coefficient_count(bands, 1, 8, 128)
        }
        ColorFormat::Yuv(ChromaSampling::Cs444) | ColorFormat::Rgb => 3 * full,
        ColorFormat::Cmyk | ColorFormat::CmykDirect | ColorFormat::YuvK | ColorFormat::Rgbe => {
            4 * full
        }
        ColorFormat::NComponent(components) => usize::from(components)
            .checked_mul(full)
            .ok_or_else(|| JxrError::arithmetic("N-component coefficient count"))?,
    };
    Ok(count)
}

const fn band_coefficient_count(
    bands: BandPresence,
    dc: usize,
    low_pass: usize,
    all: usize,
) -> usize {
    match bands {
        BandPresence::DcOnly => dc,
        BandPresence::NoHighPass => low_pass,
        BandPresence::NoFlexbits | BandPresence::All => all,
    }
}

fn native_plan_error(error: &NativeError) -> JxrError {
    match error {
        NativeError::IntegerOverflow { .. } => {
            JxrError::new(JxrErrorKind::ArithmeticOverflow, "prepare decode plan")
        }
        NativeError::Unsupported { .. } => {
            JxrError::new(JxrErrorKind::Unsupported, "prepare decode plan")
        }
        _ => JxrError::new(JxrErrorKind::InvalidSyntax, "prepare decode plan"),
    }
}

#[cfg(test)]
mod tests;
