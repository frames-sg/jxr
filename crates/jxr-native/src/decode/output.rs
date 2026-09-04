//! Final scalar color/alpha conversion, packing, and route reporting.

use jxr_core::{
    AlphaHandling, BackendKind, BackendRequest, ColorFormat, DecodeReport, DecodeRequest,
    DecodeStage, DecodedImage, DecodedSamplesMut, FallbackReason, JxrError, JxrErrorKind,
    PlaneDescriptor, PreparedPlan, StageExecutor, StageReport,
};

use crate::{
    CpuCapabilities, ParsedCodestream,
    output_format::{
        AlphaFormatRequest, ComponentPlane, OutputBitDepth, OutputFormatError, OutputFormatRequest,
        format_components_into_with_cpu, format_components_with_cpu, format_planar_yuv,
        format_planar_yuv_into,
    },
    reconstruct::{CropWindow, PlanarSamples},
};

pub(super) fn format_image_into(
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    request: &DecodeRequest,
    reconstruction: &[PlanarSamples],
    alpha_reconstruction: Option<&PlanarSamples>,
    cpu: CpuCapabilities,
    destination: DecodedSamplesMut<'_>,
) -> Result<super::CpuDecodeIntoOutput, JxrError> {
    let planes: Vec<_> = reconstruction
        .iter()
        .map(|plane| {
            ComponentPlane::positioned(
                plane.origin_x,
                plane.origin_y,
                plane.width,
                plane.height,
                &plane.samples,
            )
        })
        .collect();
    let alpha_plane = alpha_reconstruction.map(|alpha| {
        ComponentPlane::positioned(
            alpha.origin_x,
            alpha.origin_y,
            alpha.width,
            alpha.height,
            &alpha.samples,
        )
    });
    let output_request = prepare_output_format(parsed, plan, request)?;
    let planar = matches!(
        output_request.output_color,
        ColorFormat::Yuv(jxr_core::ChromaSampling::Cs420 | jxr_core::ChromaSampling::Cs422)
    );
    let (planes, used_simd) = if planar {
        let output = format_planar_yuv_into(&planes, alpha_plane, output_request, cpu, destination)
            .map_err(|error| map_output_error(&error))?;
        (output.planes, output.used_simd)
    } else {
        let used_simd =
            format_components_into_with_cpu(&planes, alpha_plane, output_request, cpu, destination)
                .map_err(|error| map_output_error(&error))?;
        let plane = PlaneDescriptor {
            byte_offset: 0,
            row_stride_bytes: request.output.row_bytes(plan.decoded_region.w)?,
            width: plan.decoded_region.w,
            height: plan.decoded_region.h,
            channels: request.output.channel_count(),
        };
        (vec![plane], used_simd)
    };
    Ok(super::CpuDecodeIntoOutput {
        info: plan.info.clone(),
        decoded_region: plan.decoded_region,
        format: request.output,
        planes,
        report: cpu_report(
            request.backend,
            used_simd,
            cpu.accelerates_i32() && plan.info.primary.bands.has_high_pass(),
        ),
    })
}

pub(super) fn format_image(
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    request: &DecodeRequest,
    reconstruction: &[PlanarSamples],
    alpha_reconstruction: Option<&PlanarSamples>,
    cpu: CpuCapabilities,
) -> Result<DecodedImage, JxrError> {
    let planes: Vec<_> = reconstruction
        .iter()
        .map(|plane| {
            ComponentPlane::positioned(
                plane.origin_x,
                plane.origin_y,
                plane.width,
                plane.height,
                &plane.samples,
            )
        })
        .collect();
    let alpha_plane = alpha_reconstruction.map(|alpha| {
        ComponentPlane::positioned(
            alpha.origin_x,
            alpha.origin_y,
            alpha.width,
            alpha.height,
            &alpha.samples,
        )
    });
    let output_request = prepare_output_format(parsed, plan, request)?;
    let output_color = output_request.output_color;
    let (samples, plane_descriptors, used_simd) = if matches!(
        output_color,
        ColorFormat::Yuv(jxr_core::ChromaSampling::Cs420 | jxr_core::ChromaSampling::Cs422)
    ) {
        let output = format_planar_yuv(&planes, alpha_plane, output_request, cpu)
            .map_err(|error| map_output_error(&error))?;
        (output.samples, output.planes, output.used_simd)
    } else {
        let (samples, used_simd) =
            format_components_with_cpu(&planes, alpha_plane, output_request, cpu)
                .map_err(|error| map_output_error(&error))?;
        let descriptor = PlaneDescriptor {
            byte_offset: 0,
            row_stride_bytes: request.output.row_bytes(plan.decoded_region.w)?,
            width: plan.decoded_region.w,
            height: plan.decoded_region.h,
            channels: request.output.channel_count(),
        };
        (samples, vec![descriptor], used_simd)
    };
    request
        .limits
        .check_host_allocation_bytes(samples.byte_len())?;
    let image = DecodedImage {
        info: plan.info.clone(),
        decoded_region: plan.decoded_region,
        format: request.output,
        planes: plane_descriptors,
        samples,
        report: cpu_report(
            request.backend,
            used_simd,
            cpu.accelerates_i32() && plan.info.primary.bands.has_high_pass(),
        ),
    };
    image.validate_layout()?;
    Ok(image)
}

/// Build the exact device-neutral output policy used by CPU and accelerator stores.
pub fn prepare_output_format(
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    request: &DecodeRequest,
) -> Result<OutputFormatRequest, JxrError> {
    let output_color = output_color(parsed)?;
    let bit_depth = OutputBitDepth::from_header_fields(
        parsed.headers.image.output_bit_depth,
        parsed.headers.primary.shift_bits,
        parsed.headers.primary.mantissa_length,
        parsed.headers.primary.exponent_bias,
    )
    .ok_or_else(|| JxrError::new(JxrErrorKind::Unsupported, "reserved output bit-depth code"))?;
    let bit_depth = if bit_depth == OutputBitDepth::U10 && output_color == ColorFormat::Rgb {
        OutputBitDepth::Rgb101010
    } else {
        bit_depth
    };
    let output = OutputFormatRequest {
        internal_color: reconstructed_color(
            plan.info.primary.color_format,
            output_color,
            plan.scale,
        ),
        output_color,
        bit_depth,
        pixel_format: request.output,
        scaled: parsed.headers.primary.scaled,
        alpha_format: alpha_format(parsed)?,
        red_blue_not_swapped: parsed.headers.image.flags.red_blue_not_swapped(),
        premultiply_alpha: matches!(request.alpha, AlphaHandling::Premultiply)
            && !plan.info.premultiplied_alpha,
        crop: output_crop(parsed, plan)?,
    };
    crate::output_format::validate_output_policy(
        output,
        plan.info.alpha_mode != jxr_core::AlphaMode::None && request.alpha != AlphaHandling::Drop,
    )
    .map_err(|error| map_output_error(&error))?;
    Ok(output)
}

fn alpha_format(parsed: &ParsedCodestream) -> Result<Option<AlphaFormatRequest>, JxrError> {
    let integrated = parsed
        .headers
        .alpha
        .as_ref()
        .map(|plane| (&parsed.headers.image, plane));
    let separate = parsed.separate_alpha_headers.as_ref().map(|headers| {
        let plane = headers.alpha.as_ref().unwrap_or(&headers.primary);
        (&headers.image, plane)
    });
    integrated
        .or(separate)
        .map(|(image, plane)| {
            Ok(AlphaFormatRequest {
                bit_depth: OutputBitDepth::from_header_fields(
                    image.output_bit_depth,
                    plane.shift_bits,
                    plane.mantissa_length,
                    plane.exponent_bias,
                )
                .ok_or_else(|| {
                    JxrError::new(
                        JxrErrorKind::Unsupported,
                        "reserved alpha output bit-depth code",
                    )
                })?,
                scaled: plane.scaled,
            })
        })
        .transpose()
}

const fn reconstructed_color(
    color: ColorFormat,
    output: ColorFormat,
    scale: jxr_core::DecodeScale,
) -> ColorFormat {
    match (color, output, scale) {
        (ColorFormat::Yuv(_), _, jxr_core::DecodeScale::Sixteenth) => {
            ColorFormat::Yuv(jxr_core::ChromaSampling::Cs444)
        }
        (ColorFormat::Yuv(sampling), ColorFormat::Yuv(output_sampling), _)
            if sampling as u8 == output_sampling as u8 =>
        {
            color
        }
        (ColorFormat::Yuv(_), _, _) => ColorFormat::Yuv(jxr_core::ChromaSampling::Cs444),
        (color, _, _) => color,
    }
}

fn output_crop(parsed: &ParsedCodestream, plan: &PreparedPlan) -> Result<CropWindow, JxrError> {
    let denominator = plan.scale.denominator();
    let x = u32::from(parsed.headers.image.margins[1])
        .checked_add(plan.output_region.x)
        .ok_or_else(|| JxrError::arithmetic("output crop x"))?
        / denominator;
    let y = u32::from(parsed.headers.image.margins[0])
        .checked_add(plan.output_region.y)
        .ok_or_else(|| JxrError::arithmetic("output crop y"))?
        / denominator;
    Ok(CropWindow {
        x,
        y,
        width: plan.decoded_region.w,
        height: plan.decoded_region.h,
    })
}

fn output_color(parsed: &ParsedCodestream) -> Result<ColorFormat, JxrError> {
    let components = parsed.headers.primary.components;
    match parsed.headers.image.output_color_format {
        0 => Ok(ColorFormat::Luma),
        1 => Ok(ColorFormat::Yuv(jxr_core::ChromaSampling::Cs420)),
        2 => Ok(ColorFormat::Yuv(jxr_core::ChromaSampling::Cs422)),
        3 => Ok(ColorFormat::Yuv(jxr_core::ChromaSampling::Cs444)),
        4 => Ok(ColorFormat::Cmyk),
        5 => Ok(ColorFormat::CmykDirect),
        6 => Ok(ColorFormat::NComponent(components)),
        7 => Ok(ColorFormat::Rgb),
        8 => Ok(ColorFormat::Rgbe),
        _ => Err(JxrError::new(
            JxrErrorKind::InvalidSyntax,
            "output color format",
        )),
    }
}

fn cpu_report(
    requested: BackendRequest,
    simd_output: bool,
    simd_reconstruction: bool,
) -> DecodeReport {
    const STAGES: [DecodeStage; 14] = [
        DecodeStage::Parse,
        DecodeStage::EntropyDecode,
        DecodeStage::InverseScan,
        DecodeStage::CoefficientRemap,
        DecodeStage::DcLowPassPrediction,
        DecodeStage::DequantizeAndFirstInverseTransform,
        DecodeStage::FirstOverlap,
        DecodeStage::HighPassPrediction,
        DecodeStage::SecondInverseTransform,
        DecodeStage::SecondOverlap,
        DecodeStage::ChromaReconstruction,
        DecodeStage::ColorAndAlphaConversion,
        DecodeStage::CropClipAndPack,
        DecodeStage::HostReadback,
    ];
    DecodeReport {
        requested,
        selected: BackendKind::Cpu,
        fallback: matches!(requested, BackendRequest::Auto)
            .then_some(FallbackReason::PipelineIncomplete),
        stages: STAGES
            .into_iter()
            .map(|stage| StageReport {
                stage,
                executor: if (simd_output && stage == DecodeStage::CropClipAndPack)
                    || (simd_reconstruction
                        && stage == DecodeStage::DequantizeAndFirstInverseTransform)
                {
                    StageExecutor::CpuSimd
                } else {
                    StageExecutor::CpuScalar
                },
            })
            .collect(),
    }
}

fn map_output_error(error: &OutputFormatError) -> JxrError {
    let kind = match error {
        OutputFormatError::UnsupportedCombination { .. } => JxrErrorKind::Unsupported,
        OutputFormatError::ArithmeticOverflow { .. } => JxrErrorKind::ArithmeticOverflow,
        OutputFormatError::InvalidPlane { .. }
        | OutputFormatError::ComponentCount { .. }
        | OutputFormatError::CropOutsidePlane { .. } => JxrErrorKind::InternalInvariant,
        OutputFormatError::AlphaMismatch => JxrErrorKind::InvalidRequest,
        OutputFormatError::InvalidFloatingPointSample => JxrErrorKind::InvalidSyntax,
    };
    JxrError::new(kind, "format decoded samples")
}

#[cfg(test)]
mod tests {
    use jxr_core::{BackendRequest, DecodeStage, StageExecutor};

    use super::cpu_report;

    #[test]
    fn report_marks_only_the_stages_that_used_simd() {
        let report = cpu_report(BackendRequest::Cpu, true, true);
        assert_eq!(
            report.executor_for(DecodeStage::DequantizeAndFirstInverseTransform),
            Some(StageExecutor::CpuSimd)
        );
        assert_eq!(
            report.executor_for(DecodeStage::CropClipAndPack),
            Some(StageExecutor::CpuSimd)
        );
        assert_eq!(
            report.executor_for(DecodeStage::SecondInverseTransform),
            Some(StageExecutor::CpuScalar)
        );
    }
}
