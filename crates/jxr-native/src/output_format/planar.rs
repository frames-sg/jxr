//! Native-resolution planar YUV clipping and typed packing.

use jxr_core::{ChromaSampling, ColorFormat, DecodedSamplesMut, PixelFormat};

use crate::{CpuCapabilities, reconstruct::CropWindow};

use super::{
    ComponentPlane, OutputBitDepth, OutputFormatError, OutputFormatRequest,
    packing::split_samples,
    scaling::scale_integer_component,
    simd_pack::pack_u8_into,
    validate::{validate_matrix, validate_plane},
};

#[derive(Clone, Copy)]
struct PlaneWindow<'a> {
    plane: ComponentPlane<'a>,
    crop: CropWindow,
    component: usize,
    alpha: bool,
}

/// Stores the native planar YUV output `request` describes in
/// `destination`, which must have its type and length: each plane of
/// [`planar_planes`] after the last. Returns whether a vector kernel stored
/// any of them.
pub(crate) fn format_planar_yuv_into(
    planes: &[ComponentPlane<'_>],
    alpha: Option<ComponentPlane<'_>>,
    request: OutputFormatRequest,
    cpu: CpuCapabilities,
    destination: DecodedSamplesMut<'_>,
) -> Result<bool, OutputFormatError> {
    let windows = validated_windows(planes, alpha, request)?;
    if !destination.matches_format(request.pixel_format) {
        return Err(OutputFormatError::UnsupportedCombination {
            combination: "direct planar destination type differs from the output format",
        });
    }
    if planar_length(&windows) != Some(destination.len()) {
        return Err(OutputFormatError::UnsupportedCombination {
            combination: "direct planar destination length differs from the output contract",
        });
    }
    fill_windows(&windows, request, cpu, destination)
}

/// Samples of every window of a planar output together.
fn planar_length(windows: &[PlaneWindow<'_>]) -> Option<usize> {
    windows.iter().try_fold(0_usize, |total, window| {
        usize::try_from(window.crop.width)
            .ok()
            .and_then(|width| {
                usize::try_from(window.crop.height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .and_then(|count| total.checked_add(count))
    })
}

/// One plane of a native planar YUV output: the component it stores and
/// the part of that component's plane it takes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlanarPlane {
    /// Component the plane stores; zero for the alpha plane.
    pub(crate) component: usize,
    /// Whether the plane stores alpha rather than a colour component.
    pub(crate) alpha: bool,
    /// The stored part, in the coordinates of the component's own plane.
    pub(crate) crop: CropWindow,
}

/// The planes of the native planar YUV output `request` asks for, in the
/// order they are stored: luma, both chroma planes, and alpha if present.
pub(crate) fn planar_planes(
    request: OutputFormatRequest,
    has_alpha: bool,
) -> Result<Vec<PlanarPlane>, OutputFormatError> {
    let sampling = match request.output_color {
        ColorFormat::Yuv(ChromaSampling::Cs420) => ChromaSampling::Cs420,
        ColorFormat::Yuv(ChromaSampling::Cs422) => ChromaSampling::Cs422,
        _ => {
            return Err(OutputFormatError::UnsupportedCombination {
                combination: "native planar output colour",
            });
        }
    };
    let chroma_crop = CropWindow {
        x: request.crop.x / 2,
        y: if sampling == ChromaSampling::Cs420 {
            request.crop.y / 2
        } else {
            request.crop.y
        },
        width: request.crop.width / 2,
        height: if sampling == ChromaSampling::Cs420 {
            request.crop.height / 2
        } else {
            request.crop.height
        },
    };
    let mut planes = Vec::with_capacity(3 + usize::from(has_alpha));
    planes.push(PlanarPlane {
        component: 0,
        alpha: false,
        crop: request.crop,
    });
    for component in 1..3 {
        planes.push(PlanarPlane {
            component,
            alpha: false,
            crop: chroma_crop,
        });
    }
    if has_alpha {
        planes.push(PlanarPlane {
            component: 0,
            alpha: true,
            crop: request.crop,
        });
    }
    Ok(planes)
}

/// Stores the part `crop` of one plane of a native planar YUV output.
///
/// `plane` holds samples of the component `stored` names, or of alpha, and
/// `crop` is any part of what `stored` takes of it: the whole, or a run of
/// its rows. `destination` receives exactly the samples of `crop`. Returns
/// whether a vector kernel stored them.
pub(crate) fn format_planar_plane_into(
    plane: ComponentPlane<'_>,
    stored: PlanarPlane,
    crop: CropWindow,
    request: OutputFormatRequest,
    cpu: CpuCapabilities,
    destination: DecodedSamplesMut<'_>,
) -> Result<bool, OutputFormatError> {
    validate_plane(plane, Some(stored.component), crop)?;
    let window = PlaneWindow {
        plane,
        crop,
        component: stored.component,
        alpha: stored.alpha,
    };
    match (request.bit_depth, request.pixel_format, destination) {
        (OutputBitDepth::U8, PixelFormat::U8(_), DecodedSamplesMut::U8(output)) => {
            let width = usize::try_from(crop.width)
                .map_err(|_| OutputFormatError::arithmetic("converting planar crop width"))?;
            let height = usize::try_from(crop.height)
                .map_err(|_| OutputFormatError::arithmetic("converting planar crop height"))?;
            let start_x = usize::try_from(crop.x - plane.origin_x)
                .map_err(|_| OutputFormatError::arithmetic("converting planar crop x"))?;
            let start_y = usize::try_from(crop.y - plane.origin_y)
                .map_err(|_| OutputFormatError::arithmetic("converting planar crop y"))?;
            let scaled = if stored.alpha {
                request
                    .alpha_format
                    .map_or(request.scaled, |format| format.scaled)
            } else {
                request.scaled
            };
            pack_u8_into(
                cpu.level(),
                plane.samples,
                plane.stride,
                [start_x, start_y],
                [width, height],
                scaled,
                output,
            )
        }
        (
            OutputBitDepth::U10 | OutputBitDepth::U16 { .. },
            PixelFormat::U16(_),
            DecodedSamplesMut::U16(output),
        ) => {
            let maximum = i32::from(request.bit_depth.maximum().unwrap_or(u16::MAX));
            fill_window(&window, request, output, |value| {
                u16::try_from(value.clamp(0, maximum)).expect("sample is clipped to u16")
            })?;
            Ok(false)
        }
        (OutputBitDepth::I16 { .. }, PixelFormat::I16(_), DecodedSamplesMut::I16(output)) => {
            fill_window(&window, request, output, |value| {
                i16::try_from(value.clamp(-32_768, 32_767)).expect("sample is clipped to i16")
            })?;
            Ok(false)
        }
        _ => Err(OutputFormatError::UnsupportedCombination {
            combination: "native planar YUV direct typed storage",
        }),
    }
}

fn validated_windows<'a>(
    planes: &'a [ComponentPlane<'a>],
    alpha: Option<ComponentPlane<'a>>,
    request: OutputFormatRequest,
) -> Result<Vec<PlaneWindow<'a>>, OutputFormatError> {
    let stored = planar_planes(request, alpha.is_some())?;
    if planes.len() != 3 {
        return Err(OutputFormatError::ComponentCount {
            expected: 3,
            actual: planes.len(),
        });
    }
    validate_matrix(request, alpha.is_some())?;
    let windows: Vec<_> = stored
        .into_iter()
        .map(|stored| PlaneWindow {
            plane: match (stored.alpha, alpha) {
                (true, Some(alpha)) => alpha,
                _ => planes[stored.component],
            },
            crop: stored.crop,
            component: stored.component,
            alpha: stored.alpha,
        })
        .collect();
    for window in &windows {
        validate_plane(window.plane, Some(window.component), window.crop)?;
    }
    Ok(windows)
}

/// Stores every window of a planar output, one after another in
/// `destination`.
fn fill_windows(
    windows: &[PlaneWindow<'_>],
    request: OutputFormatRequest,
    cpu: CpuCapabilities,
    mut destination: DecodedSamplesMut<'_>,
) -> Result<bool, OutputFormatError> {
    let mut used_simd = false;
    for window in windows {
        let width = usize::try_from(window.crop.width)
            .map_err(|_| OutputFormatError::arithmetic("converting planar crop width"))?;
        let height = usize::try_from(window.crop.height)
            .map_err(|_| OutputFormatError::arithmetic("converting planar crop height"))?;
        let count = width
            .checked_mul(height)
            .ok_or_else(|| OutputFormatError::arithmetic("calculating planar sample count"))?;
        let (stored, rest) = split_samples(destination, count);
        let plane = PlanarPlane {
            component: window.component,
            alpha: window.alpha,
            crop: window.crop,
        };
        used_simd |=
            format_planar_plane_into(window.plane, plane, window.crop, request, cpu, stored)?;
        destination = rest;
    }
    Ok(used_simd)
}

/// Scales and stores the samples of one window in raster order.
fn fill_window<T>(
    window: &PlaneWindow<'_>,
    request: OutputFormatRequest,
    output: &mut [T],
    mut convert: impl FnMut(i32) -> T,
) -> Result<(), OutputFormatError> {
    let width = usize::try_from(window.crop.width)
        .map_err(|_| OutputFormatError::arithmetic("converting planar crop width"))?;
    let height = usize::try_from(window.crop.height)
        .map_err(|_| OutputFormatError::arithmetic("converting planar crop height"))?;
    let start_x = usize::try_from(window.crop.x)
        .map_err(|_| OutputFormatError::arithmetic("converting planar crop x"))?;
    let start_y = usize::try_from(window.crop.y)
        .map_err(|_| OutputFormatError::arithmetic("converting planar crop y"))?;
    let (depth, scaled, color) = if window.alpha {
        let format = request.alpha_format.unwrap_or(super::AlphaFormatRequest {
            bit_depth: request.bit_depth,
            scaled: request.scaled,
        });
        (format.bit_depth, format.scaled, ColorFormat::Luma)
    } else {
        (request.bit_depth, request.scaled, request.output_color)
    };
    if output.len() != width.saturating_mul(height) {
        return Err(OutputFormatError::UnsupportedCombination {
            combination: "direct planar destination length differs from the output contract",
        });
    }
    let mut index = 0;
    for y in 0..height {
        for x in 0..width {
            let sample = window.plane.sample(start_x + x, start_y + y);
            let scaled = scale_integer_component(sample, window.component, color, depth, scaled)?;
            output[index] = convert(scaled);
            index += 1;
        }
    }
    Ok(())
}
