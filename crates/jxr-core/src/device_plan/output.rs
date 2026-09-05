// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use crate::{
    AlphaFormatRequest, ChannelLayout, ChromaSampling, ColorFormat, ImageInfo, OutputBitDepth,
    OutputFormatRequest, PixelFormat, StorageKind, SurfaceLayout,
};

use super::{
    OutputParameterWords, PlanError, ReconstructionPlane, SAMPLE_ALPHA, SamplePlaneWords,
    SurfacePlaneWords,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorePipeline {
    Bits,
    U8,
    U16,
    I16,
    I32,
    F16,
    F32,
    Packed16,
    Packed32,
}

impl StorePipeline {
    pub const fn entrypoint(self) -> &'static str {
        match self {
            Self::Bits => "jxr_output_bits",
            Self::U8 => "jxr_output_u8",
            Self::U16 => "jxr_output_u16",
            Self::I16 => "jxr_output_i16",
            Self::I32 => "jxr_output_i32",
            Self::F16 => "jxr_output_f16",
            Self::F32 => "jxr_output_f32",
            Self::Packed16 => "jxr_output_packed16",
            Self::Packed32 => "jxr_output_packed32",
        }
    }
}

pub struct OutputDispatchPlan {
    pub samples: Vec<SamplePlaneWords>,
    pub surfaces: Vec<SurfacePlaneWords>,
    pub params: OutputParameterWords,
    pub pipeline: StorePipeline,
    pub planar: bool,
}

pub fn build_output_dispatch(
    planes: &[ReconstructionPlane],
    output: &SurfaceLayout,
    policy: OutputFormatRequest,
    info: &ImageInfo,
) -> Result<OutputDispatchPlan, PlanError> {
    let samples = sample_descriptors(planes)?;
    let surfaces = surface_descriptors(output)?;
    let params = output_parameters(output, policy, info, &samples, &surfaces)?;
    let pipeline = store_pipeline(policy.pixel_format);
    let planar = matches!(
        policy.output_color,
        ColorFormat::Yuv(ChromaSampling::Cs420 | ChromaSampling::Cs422)
    );
    Ok(OutputDispatchPlan {
        samples,
        surfaces,
        params,
        pipeline,
        planar,
    })
}

fn sample_descriptors(planes: &[ReconstructionPlane]) -> Result<Vec<SamplePlaneWords>, PlanError> {
    planes
        .iter()
        .map(|plane| {
            Ok([
                u32::try_from(plane.sample_offset)
                    .map_err(|_| invalid("sample plane offset exceeds the device ABI"))?,
                plane.sample_origin_x,
                plane.sample_origin_y,
                plane.sample_width,
                plane.sample_height,
                u32::from(plane.alpha),
            ])
        })
        .collect()
}

fn surface_descriptors(output: &SurfaceLayout) -> Result<Vec<SurfacePlaneWords>, PlanError> {
    output
        .planes
        .iter()
        .map(|plane| {
            Ok([
                u32::try_from(plane.byte_offset)
                    .map_err(|_| invalid("output plane offset exceeds the device ABI"))?,
                u32::try_from(plane.row_stride_bytes)
                    .map_err(|_| invalid("output row stride exceeds the device ABI"))?,
                plane.width,
                plane.height,
                u32::from(plane.channels),
                0,
            ])
        })
        .collect()
}

fn output_parameters(
    output: &SurfaceLayout,
    policy: OutputFormatRequest,
    info: &ImageInfo,
    samples: &[SamplePlaneWords],
    surfaces: &[SurfacePlaneWords],
) -> Result<OutputParameterWords, PlanError> {
    let component_count = info
        .primary
        .color_format
        .component_count()
        .ok_or_else(|| invalid("primary component count is zero"))?;
    let channel_layout = layout_code(policy.pixel_format);
    let alpha_plane = alpha_plane(samples, channel_layout)?;
    let alpha_format = policy.alpha_format.unwrap_or(AlphaFormatRequest {
        bit_depth: policy.bit_depth,
        scaled: policy.scaled,
    });
    validate_math_policy(info, policy, alpha_format)?;
    let (shift_bits, mantissa_length, exponent_bias) = depth_fields(policy.bit_depth);
    let (alpha_shift_bits, alpha_mantissa_length, alpha_exponent_bias) =
        depth_fields(alpha_format.bit_depth);
    let (internal_color, chroma_sampling) = color_code(info.primary.color_format);
    let (output_color, _) = color_code(policy.output_color);
    let output_plane_count = u32::try_from(surfaces.len())
        .map_err(|_| invalid("output plane count exceeds the device ABI"))?;
    Ok([
        output.width,
        output.height,
        policy.crop.x,
        policy.crop.y,
        u32::from(component_count),
        alpha_plane,
        internal_color,
        output_color,
        chroma_sampling,
        u32::from(info.primary.chroma_centering[0]),
        u32::from(info.primary.chroma_centering[1]),
        depth_code(policy.bit_depth),
        channel_layout,
        u32::from(policy.pixel_format.channel_count()),
        u32::from(policy.scaled),
        u32::from(alpha_format.scaled),
        u32::from(policy.premultiply_alpha),
        u32::from(policy.red_blue_not_swapped),
        shift_bits,
        alpha_shift_bits,
        mantissa_length,
        alpha_mantissa_length,
        u32::from_ne_bytes(i32::from(exponent_bias).to_ne_bytes()),
        u32::from_ne_bytes(i32::from(alpha_exponent_bias).to_ne_bytes()),
        0,
        output_plane_count,
        u32::from(policy.bit_depth == OutputBitDepth::Bit1Black),
        0,
    ])
}

fn alpha_plane(samples: &[SamplePlaneWords], channel_layout: u32) -> Result<u32, PlanError> {
    let layout_has_alpha = matches!(channel_layout, 1 | 3 | 5 | 7 | 9 | 11);
    let alpha_plane = layout_has_alpha
        .then(|| samples.iter().position(|plane| plane[SAMPLE_ALPHA] != 0))
        .flatten()
        .and_then(|index| u32::try_from(index).ok())
        .unwrap_or(u32::MAX);
    if layout_has_alpha && alpha_plane == u32::MAX {
        Err(invalid("alpha output layout has no prepared alpha plane"))
    } else {
        Ok(alpha_plane)
    }
}

fn validate_math_policy(
    info: &ImageInfo,
    policy: OutputFormatRequest,
    alpha_format: AlphaFormatRequest,
) -> Result<(), PlanError> {
    if matches!(
        info.primary.color_format,
        ColorFormat::Yuv(ChromaSampling::Cs420 | ChromaSampling::Cs422)
    ) && !matches!(policy.output_color, ColorFormat::Yuv(_))
        && info.primary.chroma_centering.iter().any(|&value| value > 4)
    {
        return Err(PlanError::UnsupportedOutputFormat {
            reason: "unknown chroma centering has no exact reconstruction filter",
        });
    }
    if matches!(policy.bit_depth, OutputBitDepth::F32 { mantissa_length, .. } if mantissa_length > 23)
        || matches!(alpha_format.bit_depth, OutputBitDepth::F32 { mantissa_length, .. } if mantissa_length > 23)
    {
        return Err(PlanError::UnsupportedOutputFormat {
            reason: "BD32F mantissa length exceeds 23 bits",
        });
    }
    Ok(())
}

const fn store_pipeline(pixel_format: PixelFormat) -> StorePipeline {
    match pixel_format.storage_kind() {
        StorageKind::BitPacked => StorePipeline::Bits,
        StorageKind::U8 => StorePipeline::U8,
        StorageKind::U16 => StorePipeline::U16,
        StorageKind::I16 => StorePipeline::I16,
        StorageKind::I32 => StorePipeline::I32,
        StorageKind::F16Bits => StorePipeline::F16,
        StorageKind::F32 => StorePipeline::F32,
        StorageKind::PackedU16 => StorePipeline::Packed16,
        StorageKind::PackedU32 => StorePipeline::Packed32,
    }
}

const fn depth_fields(depth: OutputBitDepth) -> (u32, u32, i8) {
    match depth {
        OutputBitDepth::U16 { shift_bits }
        | OutputBitDepth::I16 { shift_bits }
        | OutputBitDepth::I32 { shift_bits } => (shift_bits as u32, 0, 0),
        OutputBitDepth::F32 {
            mantissa_length,
            exponent_bias,
        } => (0, mantissa_length as u32, exponent_bias),
        _ => (0, 0, 0),
    }
}

const fn depth_code(depth: OutputBitDepth) -> u32 {
    match depth {
        OutputBitDepth::Bit1White | OutputBitDepth::Bit1Black => 0,
        OutputBitDepth::U8 => 1,
        OutputBitDepth::U10 => 2,
        OutputBitDepth::U16 { .. } => 3,
        OutputBitDepth::I16 { .. } => 4,
        OutputBitDepth::F16 => 5,
        OutputBitDepth::I32 { .. } => 6,
        OutputBitDepth::F32 { .. } => 7,
        OutputBitDepth::Rgb555 => 8,
        OutputBitDepth::Rgb101010 => 9,
        OutputBitDepth::Rgb565 => 10,
    }
}

const fn color_code(color: ColorFormat) -> (u32, u32) {
    match color {
        ColorFormat::Luma => (0, 0),
        ColorFormat::Yuv(ChromaSampling::Cs420) => (1, 1),
        ColorFormat::Yuv(ChromaSampling::Cs422) => (1, 2),
        ColorFormat::Yuv(ChromaSampling::Cs444) => (1, 3),
        ColorFormat::Rgb => (2, 0),
        ColorFormat::Cmyk => (3, 0),
        ColorFormat::CmykDirect => (4, 0),
        ColorFormat::YuvK => (5, 0),
        ColorFormat::Rgbe => (6, 0),
        ColorFormat::NComponent(_) => (7, 0),
    }
}

const fn layout_code(format: PixelFormat) -> u32 {
    let layout = match format {
        PixelFormat::BitPacked(layout)
        | PixelFormat::U8(layout)
        | PixelFormat::U16(layout)
        | PixelFormat::I16(layout)
        | PixelFormat::I32(layout)
        | PixelFormat::F16(layout)
        | PixelFormat::F32(layout) => layout,
        PixelFormat::Rgb555 | PixelFormat::Rgb565 | PixelFormat::Rgb101010 | PixelFormat::Rgbe => {
            ChannelLayout::Rgb
        }
    };
    match layout {
        ChannelLayout::Luma => 0,
        ChannelLayout::LumaAlpha => 1,
        ChannelLayout::Yuv(_) => 2,
        ChannelLayout::Yuva(_) => 3,
        ChannelLayout::Rgb => 4,
        ChannelLayout::Rgbx => 12,
        ChannelLayout::Rgba => 5,
        ChannelLayout::Bgr => 6,
        ChannelLayout::Bgrx => 13,
        ChannelLayout::Bgra => 7,
        ChannelLayout::Cmyk => 8,
        ChannelLayout::Cmyka => 9,
        ChannelLayout::NComponent(_) => 10,
        ChannelLayout::NComponentAlpha(_) => 11,
    }
}

const fn invalid(reason: &'static str) -> PlanError {
    PlanError::InvalidPlan { reason }
}
