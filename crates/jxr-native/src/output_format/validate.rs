//! Validation of output-format matrix and borrowed component extents.

use jxr_core::{ChannelLayout, ChromaSampling, ColorFormat, PixelFormat};

use super::{
    ComponentPlane, OutputBitDepth, OutputFormatError, OutputFormatRequest, packing::layout,
};

pub(crate) fn validate_request(
    planes: &[ComponentPlane<'_>],
    alpha: Option<ComponentPlane<'_>>,
    request: OutputFormatRequest,
) -> Result<usize, OutputFormatError> {
    let component_count = usize::from(request.internal_color.component_count().ok_or(
        OutputFormatError::UnsupportedCombination {
            combination: "zero internal components",
        },
    )?);
    if planes.len() != component_count {
        return Err(OutputFormatError::ComponentCount {
            expected: component_count,
            actual: planes.len(),
        });
    }
    validate_matrix(request, alpha.is_some())?;
    for (component, plane) in planes.iter().copied().enumerate() {
        validate_plane(plane, Some(component), request.crop)?;
        if plane.width != planes[0].width
            || plane.height != planes[0].height
            || plane.origin_x != planes[0].origin_x
            || plane.origin_y != planes[0].origin_y
        {
            return Err(OutputFormatError::InvalidPlane {
                component: Some(component),
                reason: "full-resolution component dimensions differ",
            });
        }
    }
    if let Some(alpha) = alpha {
        validate_plane(alpha, None, request.crop)?;
    }
    Ok(component_count)
}

/// Validate color, depth, storage, channel-order, and alpha policy without samples.
pub fn validate_output_policy(
    request: OutputFormatRequest,
    has_alpha: bool,
) -> Result<(), OutputFormatError> {
    validate_matrix(request, has_alpha)
}

pub(super) fn validate_plane(
    plane: ComponentPlane<'_>,
    component: Option<usize>,
    crop: crate::reconstruct::CropWindow,
) -> Result<(), OutputFormatError> {
    let width = usize::try_from(plane.width)
        .map_err(|_| OutputFormatError::arithmetic("converting plane width"))?;
    let height = usize::try_from(plane.height)
        .map_err(|_| OutputFormatError::arithmetic("converting plane height"))?;
    if plane.stride < width {
        return Err(OutputFormatError::InvalidPlane {
            component,
            reason: "stride is shorter than width",
        });
    }
    let required = if height == 0 {
        0
    } else {
        plane
            .stride
            .checked_mul(height - 1)
            .and_then(|value| value.checked_add(width))
            .ok_or_else(|| OutputFormatError::arithmetic("calculating plane extent"))?
    };
    if plane.samples.len() < required {
        return Err(OutputFormatError::InvalidPlane {
            component,
            reason: "backing slice is shorter than its extent",
        });
    }
    let plane_right = plane
        .origin_x
        .checked_add(plane.width)
        .ok_or_else(|| OutputFormatError::arithmetic("calculating plane right edge"))?;
    let plane_bottom = plane
        .origin_y
        .checked_add(plane.height)
        .ok_or_else(|| OutputFormatError::arithmetic("calculating plane bottom edge"))?;
    let right = crop
        .x
        .checked_add(crop.width)
        .ok_or(OutputFormatError::CropOutsidePlane { component })?;
    let bottom = crop
        .y
        .checked_add(crop.height)
        .ok_or(OutputFormatError::CropOutsidePlane { component })?;
    if crop.x < plane.origin_x
        || crop.y < plane.origin_y
        || right > plane_right
        || bottom > plane_bottom
    {
        return Err(OutputFormatError::CropOutsidePlane { component });
    }
    Ok(())
}

pub(super) fn validate_matrix(
    request: OutputFormatRequest,
    has_alpha: bool,
) -> Result<(), OutputFormatError> {
    validate_alpha(layout(request.pixel_format), has_alpha)?;
    if request.premultiply_alpha && !has_alpha {
        return Err(OutputFormatError::AlphaMismatch);
    }
    if request.premultiply_alpha && matches!(request.output_color, ColorFormat::Yuv(_)) {
        return Err(OutputFormatError::UnsupportedCombination {
            combination: "premultiplied YUV output",
        });
    }
    validate_depth_storage(request)?;
    validate_color_layout(request)?;
    validate_conversion(request)?;
    validate_normative_depth(request)?;
    Ok(())
}

fn validate_alpha(layout: ChannelLayout, has_alpha: bool) -> Result<(), OutputFormatError> {
    let layout_has_alpha = matches!(
        layout,
        ChannelLayout::LumaAlpha
            | ChannelLayout::Yuva(_)
            | ChannelLayout::Rgba
            | ChannelLayout::Bgra
            | ChannelLayout::Cmyka
            | ChannelLayout::NComponentAlpha(_)
    );
    if layout_has_alpha == has_alpha {
        Ok(())
    } else {
        Err(OutputFormatError::AlphaMismatch)
    }
}

fn validate_depth_storage(request: OutputFormatRequest) -> Result<(), OutputFormatError> {
    let matches = matches!(
        (request.bit_depth, request.pixel_format),
        (
            OutputBitDepth::Bit1White | OutputBitDepth::Bit1Black,
            PixelFormat::BitPacked(ChannelLayout::Luma)
        ) | (OutputBitDepth::U8, PixelFormat::U8(_) | PixelFormat::Rgbe)
            | (
                OutputBitDepth::U10 | OutputBitDepth::U16 { .. },
                PixelFormat::U16(_)
            )
            | (OutputBitDepth::I16 { .. }, PixelFormat::I16(_))
            | (OutputBitDepth::I32 { .. }, PixelFormat::I32(_))
            | (OutputBitDepth::F16, PixelFormat::F16(_))
            | (OutputBitDepth::F32 { .. }, PixelFormat::F32(_))
            | (OutputBitDepth::Rgb555, PixelFormat::Rgb555)
            | (OutputBitDepth::Rgb565, PixelFormat::Rgb565)
            | (OutputBitDepth::Rgb101010, PixelFormat::Rgb101010)
    );
    if matches {
        Ok(())
    } else {
        Err(OutputFormatError::UnsupportedCombination {
            combination: "bit depth and typed storage",
        })
    }
}

fn validate_color_layout(request: OutputFormatRequest) -> Result<(), OutputFormatError> {
    let layout = layout(request.pixel_format);
    let matches = match request.output_color {
        ColorFormat::Luma => matches!(layout, ChannelLayout::Luma | ChannelLayout::LumaAlpha),
        ColorFormat::Rgb => {
            matches!(
                layout,
                ChannelLayout::Rgb
                    | ChannelLayout::Rgbx
                    | ChannelLayout::Rgba
                    | ChannelLayout::Bgr
                    | ChannelLayout::Bgrx
                    | ChannelLayout::Bgra
            ) || matches!(
                request.pixel_format,
                PixelFormat::Rgb555 | PixelFormat::Rgb565 | PixelFormat::Rgb101010
            )
        }
        ColorFormat::Yuv(sampling) => {
            layout == ChannelLayout::Yuv(sampling) || layout == ChannelLayout::Yuva(sampling)
        }
        ColorFormat::Cmyk | ColorFormat::CmykDirect => {
            matches!(layout, ChannelLayout::Cmyk | ChannelLayout::Cmyka)
        }
        ColorFormat::Rgbe => request.pixel_format == PixelFormat::Rgbe,
        ColorFormat::NComponent(count) => {
            layout == ChannelLayout::NComponent(count)
                || layout == ChannelLayout::NComponentAlpha(count)
        }
        ColorFormat::YuvK => false,
    };
    if matches {
        Ok(())
    } else {
        Err(OutputFormatError::UnsupportedCombination {
            combination: "output colour and channel layout",
        })
    }
}

fn validate_conversion(request: OutputFormatRequest) -> Result<(), OutputFormatError> {
    let integer_rgb = !matches!(
        request.bit_depth,
        OutputBitDepth::F16 | OutputBitDepth::F32 { .. }
    );
    let matches = match (request.internal_color, request.output_color) {
        (ColorFormat::Luma, ColorFormat::Rgb) => integer_rgb,
        (ColorFormat::Luma, ColorFormat::Luma)
        | (ColorFormat::Yuv(ChromaSampling::Cs444), ColorFormat::Rgb | ColorFormat::Rgbe)
        | (ColorFormat::YuvK, ColorFormat::Cmyk | ColorFormat::CmykDirect) => true,
        (ColorFormat::Yuv(input), ColorFormat::Yuv(output)) => input == output,
        (ColorFormat::NComponent(input), ColorFormat::NComponent(output)) => input == output,
        _ => false,
    };
    if matches {
        Ok(())
    } else {
        Err(OutputFormatError::UnsupportedCombination {
            combination: "internal and output colour formats",
        })
    }
}

fn validate_normative_depth(request: OutputFormatRequest) -> Result<(), OutputFormatError> {
    let matches = match request.output_color {
        ColorFormat::Luma => !matches!(
            request.bit_depth,
            OutputBitDepth::Rgb555 | OutputBitDepth::Rgb565 | OutputBitDepth::Rgb101010
        ),
        ColorFormat::Rgb => true,
        ColorFormat::Yuv(_) => matches!(
            request.bit_depth,
            OutputBitDepth::U8
                | OutputBitDepth::U10
                | OutputBitDepth::U16 { .. }
                | OutputBitDepth::I16 { .. }
        ),
        ColorFormat::Rgbe => matches!(request.bit_depth, OutputBitDepth::U8),
        ColorFormat::Cmyk | ColorFormat::CmykDirect | ColorFormat::NComponent(_) => {
            matches!(
                request.bit_depth,
                OutputBitDepth::U8 | OutputBitDepth::U16 { .. }
            )
        }
        ColorFormat::YuvK => false,
    };
    if matches {
        Ok(())
    } else {
        Err(OutputFormatError::UnsupportedCombination {
            combination: "T.832 output colour and bit depth matrix",
        })
    }
}
