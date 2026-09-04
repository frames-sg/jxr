//! Native-resolution planar YUV clipping and typed packing.

use jxr_core::{
    ChromaSampling, ColorFormat, DecodedSamples, DecodedSamplesMut, PixelFormat, PlaneDescriptor,
};

use crate::{CpuCapabilities, reconstruct::CropWindow};

use super::{
    ComponentPlane, OutputBitDepth, OutputFormatError, OutputFormatRequest,
    scaling::scale_integer_component,
    simd_pack::append_u8,
    validate::{validate_matrix, validate_plane},
};

pub(crate) struct PlanarFormattedOutput {
    pub(crate) samples: DecodedSamples,
    pub(crate) planes: Vec<PlaneDescriptor>,
    pub(crate) used_simd: bool,
}

pub(crate) struct PlanarDecodeIntoOutput {
    pub(crate) planes: Vec<PlaneDescriptor>,
    pub(crate) used_simd: bool,
}

#[derive(Clone, Copy)]
struct PlaneWindow<'a> {
    plane: ComponentPlane<'a>,
    crop: CropWindow,
    component: usize,
    alpha: bool,
}

pub(crate) fn format_planar_yuv(
    planes: &[ComponentPlane<'_>],
    alpha: Option<ComponentPlane<'_>>,
    request: OutputFormatRequest,
    cpu: CpuCapabilities,
) -> Result<PlanarFormattedOutput, OutputFormatError> {
    let windows = validated_windows(planes, alpha, request)?;

    let (samples, descriptors, used_simd) = match (request.bit_depth, request.pixel_format) {
        (OutputBitDepth::U8, PixelFormat::U8(_)) => {
            let (samples, descriptors, used_simd) = pack_u8_windows(&windows, request, cpu)?;
            (DecodedSamples::U8(samples), descriptors, used_simd)
        }
        (OutputBitDepth::U10 | OutputBitDepth::U16 { .. }, PixelFormat::U16(_)) => {
            let (samples, descriptors) =
                pack_windows(&windows, request, 2, |value, output: &mut Vec<u16>| {
                    let maximum = if matches!(request.bit_depth, OutputBitDepth::U10) {
                        1_023
                    } else {
                        65_535
                    };
                    output.push(
                        u16::try_from(value.clamp(0, maximum)).expect("sample is clipped to u16"),
                    );
                })?;
            (samples, descriptors, false)
        }
        (OutputBitDepth::I16 { .. }, PixelFormat::I16(_)) => {
            let (samples, descriptors) =
                pack_windows(&windows, request, 2, |value, output: &mut Vec<i16>| {
                    output.push(
                        i16::try_from(value.clamp(-32_768, 32_767))
                            .expect("sample is clipped to i16"),
                    );
                })?;
            (samples, descriptors, false)
        }
        _ => {
            return Err(OutputFormatError::UnsupportedCombination {
                combination: "native planar YUV typed storage",
            });
        }
    };
    Ok(PlanarFormattedOutput {
        samples,
        planes: descriptors,
        used_simd,
    })
}

pub(crate) fn format_planar_yuv_into(
    planes: &[ComponentPlane<'_>],
    alpha: Option<ComponentPlane<'_>>,
    request: OutputFormatRequest,
    _cpu: CpuCapabilities,
    destination: DecodedSamplesMut<'_>,
) -> Result<PlanarDecodeIntoOutput, OutputFormatError> {
    let windows = validated_windows(planes, alpha, request)?;
    if !destination.matches_format(request.pixel_format) {
        return Err(OutputFormatError::UnsupportedCombination {
            combination: "direct planar destination type differs from the output format",
        });
    }
    let expected = windows.iter().try_fold(0_usize, |total, window| {
        usize::try_from(window.crop.width)
            .ok()
            .and_then(|width| {
                usize::try_from(window.crop.height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .and_then(|count| total.checked_add(count))
    });
    if expected != Some(destination.len()) {
        return Err(OutputFormatError::UnsupportedCombination {
            combination: "direct planar destination length differs from the output contract",
        });
    }
    let planes = match (request.bit_depth, request.pixel_format, destination) {
        (OutputBitDepth::U8, PixelFormat::U8(_), DecodedSamplesMut::U8(output)) => {
            fill_windows(&windows, request, 1, output, |value| {
                u8::try_from(value.clamp(0, 255)).expect("sample is clipped to u8")
            })?
        }
        (
            OutputBitDepth::U10 | OutputBitDepth::U16 { .. },
            PixelFormat::U16(_),
            DecodedSamplesMut::U16(output),
        ) => {
            let maximum = if matches!(request.bit_depth, OutputBitDepth::U10) {
                1_023
            } else {
                65_535
            };
            fill_windows(&windows, request, 2, output, |value| {
                u16::try_from(value.clamp(0, maximum)).expect("sample is clipped to u16")
            })?
        }
        (OutputBitDepth::I16 { .. }, PixelFormat::I16(_), DecodedSamplesMut::I16(output)) => {
            fill_windows(&windows, request, 2, output, |value| {
                i16::try_from(value.clamp(-32_768, 32_767)).expect("sample is clipped to i16")
            })?
        }
        _ => {
            return Err(OutputFormatError::UnsupportedCombination {
                combination: "native planar YUV direct typed storage",
            });
        }
    };
    Ok(PlanarDecodeIntoOutput {
        planes,
        used_simd: false,
    })
}

fn validated_windows<'a>(
    planes: &'a [ComponentPlane<'a>],
    alpha: Option<ComponentPlane<'a>>,
    request: OutputFormatRequest,
) -> Result<Vec<PlaneWindow<'a>>, OutputFormatError> {
    let sampling = match request.output_color {
        ColorFormat::Yuv(ChromaSampling::Cs420) => ChromaSampling::Cs420,
        ColorFormat::Yuv(ChromaSampling::Cs422) => ChromaSampling::Cs422,
        _ => {
            return Err(OutputFormatError::UnsupportedCombination {
                combination: "native planar output colour",
            });
        }
    };
    if planes.len() != 3 {
        return Err(OutputFormatError::ComponentCount {
            expected: 3,
            actual: planes.len(),
        });
    }
    validate_matrix(request, alpha.is_some())?;
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
    let mut windows = Vec::with_capacity(3 + usize::from(alpha.is_some()));
    windows.push(PlaneWindow {
        plane: planes[0],
        crop: request.crop,
        component: 0,
        alpha: false,
    });
    for (component, plane) in planes[1..].iter().copied().enumerate() {
        windows.push(PlaneWindow {
            plane,
            crop: chroma_crop,
            component: component + 1,
            alpha: false,
        });
    }
    if let Some(plane) = alpha {
        windows.push(PlaneWindow {
            plane,
            crop: request.crop,
            component: 0,
            alpha: true,
        });
    }
    for window in &windows {
        validate_plane(window.plane, Some(window.component), window.crop)?;
    }
    Ok(windows)
}

fn fill_windows<T>(
    windows: &[PlaneWindow<'_>],
    request: OutputFormatRequest,
    bytes_per_sample: usize,
    output: &mut [T],
    mut convert: impl FnMut(i32) -> T,
) -> Result<Vec<PlaneDescriptor>, OutputFormatError> {
    let mut descriptors = Vec::with_capacity(windows.len());
    let mut offset = 0_usize;
    for window in windows {
        let width = usize::try_from(window.crop.width)
            .map_err(|_| OutputFormatError::arithmetic("converting planar crop width"))?;
        let height = usize::try_from(window.crop.height)
            .map_err(|_| OutputFormatError::arithmetic("converting planar crop height"))?;
        let count = width
            .checked_mul(height)
            .ok_or_else(|| OutputFormatError::arithmetic("calculating planar sample count"))?;
        let end = offset
            .checked_add(count)
            .ok_or_else(|| OutputFormatError::arithmetic("calculating planar output range"))?;
        descriptors.push(PlaneDescriptor {
            byte_offset: offset
                .checked_mul(bytes_per_sample)
                .ok_or_else(|| OutputFormatError::arithmetic("calculating planar byte offset"))?,
            row_stride_bytes: width
                .checked_mul(bytes_per_sample)
                .ok_or_else(|| OutputFormatError::arithmetic("calculating planar row stride"))?,
            width: window.crop.width,
            height: window.crop.height,
            channels: 1,
        });
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
        let mut index = offset;
        for y in 0..height {
            for x in 0..width {
                let sample = window.plane.sample(start_x + x, start_y + y);
                let scaled =
                    scale_integer_component(sample, window.component, color, depth, scaled)?;
                output[index] = convert(scaled);
                index += 1;
            }
        }
        offset = end;
    }
    Ok(descriptors)
}

fn pack_u8_windows(
    windows: &[PlaneWindow<'_>],
    request: OutputFormatRequest,
    cpu: CpuCapabilities,
) -> Result<(Vec<u8>, Vec<PlaneDescriptor>, bool), OutputFormatError> {
    let mut output = Vec::new();
    let mut descriptors = Vec::with_capacity(windows.len());
    let mut used_simd = false;
    for window in windows {
        let width = usize::try_from(window.crop.width)
            .map_err(|_| OutputFormatError::arithmetic("converting planar crop width"))?;
        let height = usize::try_from(window.crop.height)
            .map_err(|_| OutputFormatError::arithmetic("converting planar crop height"))?;
        descriptors.push(PlaneDescriptor {
            byte_offset: output.len(),
            row_stride_bytes: width,
            width: window.crop.width,
            height: window.crop.height,
            channels: 1,
        });
        let start_x = usize::try_from(window.crop.x - window.plane.origin_x)
            .map_err(|_| OutputFormatError::arithmetic("converting planar crop x"))?;
        let start_y = usize::try_from(window.crop.y - window.plane.origin_y)
            .map_err(|_| OutputFormatError::arithmetic("converting planar crop y"))?;
        let scaled = if window.alpha {
            request
                .alpha_format
                .map_or(request.scaled, |format| format.scaled)
        } else {
            request.scaled
        };
        used_simd |= append_u8(
            cpu.level(),
            window.plane.samples,
            window.plane.stride,
            [start_x, start_y],
            [width, height],
            scaled,
            &mut output,
        )?;
    }
    Ok((output, descriptors, used_simd))
}

fn pack_windows<T>(
    windows: &[PlaneWindow<'_>],
    request: OutputFormatRequest,
    bytes_per_sample: usize,
    mut push: impl FnMut(i32, &mut Vec<T>),
) -> Result<(DecodedSamples, Vec<PlaneDescriptor>), OutputFormatError>
where
    Vec<T>: IntoDecodedSamples,
{
    let mut output = Vec::new();
    let mut descriptors = Vec::with_capacity(windows.len());
    for window in windows {
        let width = usize::try_from(window.crop.width)
            .map_err(|_| OutputFormatError::arithmetic("converting planar crop width"))?;
        let height = usize::try_from(window.crop.height)
            .map_err(|_| OutputFormatError::arithmetic("converting planar crop height"))?;
        let byte_offset = output
            .len()
            .checked_mul(bytes_per_sample)
            .ok_or_else(|| OutputFormatError::arithmetic("calculating planar byte offset"))?;
        let row_stride_bytes = width
            .checked_mul(bytes_per_sample)
            .ok_or_else(|| OutputFormatError::arithmetic("calculating planar row stride"))?;
        descriptors.push(PlaneDescriptor {
            byte_offset,
            row_stride_bytes,
            width: window.crop.width,
            height: window.crop.height,
            channels: 1,
        });
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
        for y in 0..height {
            for x in 0..width {
                let sample = window.plane.sample(start_x + x, start_y + y);
                let scaled =
                    scale_integer_component(sample, window.component, color, depth, scaled)?;
                push(scaled, &mut output);
            }
        }
    }
    Ok((output.into_decoded_samples(), descriptors))
}

trait IntoDecodedSamples {
    fn into_decoded_samples(self) -> DecodedSamples;
}

impl IntoDecodedSamples for Vec<u8> {
    fn into_decoded_samples(self) -> DecodedSamples {
        DecodedSamples::U8(self)
    }
}

impl IntoDecodedSamples for Vec<u16> {
    fn into_decoded_samples(self) -> DecodedSamples {
        DecodedSamples::U16(self)
    }
}

impl IntoDecodedSamples for Vec<i16> {
    fn into_decoded_samples(self) -> DecodedSamples {
        DecodedSamples::I16(self)
    }
}
