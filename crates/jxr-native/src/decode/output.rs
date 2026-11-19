//! Final scalar color/alpha conversion, packing, and route reporting.

use core::ops::Range;

use jxr_core::{
    AlphaHandling, BackendKind, BackendRequest, ColorFormat, DecodeReport, DecodeRequest,
    DecodeStage, DecodedImage, DecodedSamples, DecodedSamplesMut, FallbackReason, JxrError,
    JxrErrorKind, PlaneDescriptor, PreparedPlan, StageExecutor, StageReport,
};

use crate::{
    CpuCapabilities, ParsedCodestream,
    output_format::{
        AlphaFormatRequest, ComponentPlane, OutputBitDepth, OutputFormatError, OutputFormatRequest,
        PlanarPlane, borrow_samples, format_components_into_with_cpu, format_planar_plane_into,
        format_planar_yuv_into, planar_planes, split_samples, stores_with_vectors, zeroed_samples,
    },
    reconstruct::{CropWindow, PlanarSamples, ReconstructionSimd},
};

/// Stores the output of a reduced-resolution decode, whose planes are
/// reconstructed whole, in `destination`, which must have its type and
/// length.
pub(super) fn format_image_into(
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    request: &DecodeRequest,
    reconstruction: &[PlanarSamples],
    alpha_reconstruction: Option<&PlanarSamples>,
    cpu: CpuCapabilities,
    destination: DecodedSamplesMut<'_>,
) -> Result<super::CpuDecodeIntoOutput, JxrError> {
    fn positioned(plane: &PlanarSamples) -> ComponentPlane<'_> {
        ComponentPlane::positioned(
            plane.origin_x,
            plane.origin_y,
            plane.width,
            plane.height,
            &plane.samples,
        )
    }
    let output_request = prepare_output_format(parsed, plan, request)?;
    let planes: Vec<_> = reconstruction.iter().map(positioned).collect();
    let alpha_plane = alpha_reconstruction.map(positioned);
    let used_simd = if is_planar(output_request.output_color) {
        format_planar_yuv_into(&planes, alpha_plane, output_request, cpu, destination)
    } else {
        format_components_into_with_cpu(&planes, alpha_plane, output_request, cpu, destination)
    }
    .map_err(|error| map_output_error(&error))?;
    Ok(super::CpuDecodeIntoOutput {
        info: plan.info.clone(),
        decoded_region: plan.decoded_region,
        format: request.output,
        planes: plane_descriptors(output_request, alpha_plane.is_some())?,
        report: cpu_report(request.backend, used_simd, ReconstructionSimd::default()),
    })
}

/// [`format_image_into`] storage allocated for the output.
pub(super) fn format_image(
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    request: &DecodeRequest,
    reconstruction: &[PlanarSamples],
    alpha_reconstruction: Option<&PlanarSamples>,
    cpu: CpuCapabilities,
) -> Result<DecodedImage, JxrError> {
    let output_request = prepare_output_format(parsed, plan, request)?;
    let mut samples = allocate_output(output_request, alpha_reconstruction.is_some())?;
    let stored = format_image_into(
        parsed,
        plan,
        request,
        reconstruction,
        alpha_reconstruction,
        cpu,
        borrow_samples(&mut samples),
    )?;
    request
        .limits
        .check_host_allocation_bytes(samples.byte_len())?;
    let image = DecodedImage {
        info: stored.info,
        decoded_region: stored.decoded_region,
        format: stored.format,
        planes: stored.planes,
        samples,
        report: stored.report,
    };
    image.validate_layout()?;
    Ok(image)
}

/// The output of a decode, stored a run of rows at a time as its component
/// rows are finished.
///
/// An interleaved output is one plane whose every row takes a row of each
/// component. A native planar YUV output is a plane for each component,
/// whose rows are those of that component alone, so that its chroma planes
/// advance at their own rate.
pub(super) struct OutputRows<'d> {
    request: OutputFormatRequest,
    cpu: CpuCapabilities,
    planes: Vec<OutputPlane<'d>>,
}

struct OutputPlane<'d> {
    /// The component a plane of a planar output stores, or `None` for the
    /// one plane of an interleaved output.
    stored: Option<PlanarPlane>,
    /// The part of the source plane stored, in its own coordinates.
    crop: CropWindow,
    /// Storage elements in one row.
    row_elements: usize,
    /// Next row of the source plane to store.
    next: u32,
    /// Storage of the rows from `next` on.
    remaining: DecodedSamplesMut<'d>,
}

impl<'d> OutputRows<'d> {
    /// Lays the planes of the output `request` describes over `destination`,
    /// which must be exactly as long as the output, and divides them among
    /// strips of the image.
    ///
    /// `cuts` are the rows of the luma plane, each a whole number of
    /// macroblock rows down the extended plane, where one strip's rows end
    /// and the next one's begin. The outputs returned, one more than the
    /// cuts, each store the rows of every plane that fall in their strip,
    /// and between them hold all of `destination`.
    pub(super) fn split(
        request: OutputFormatRequest,
        has_alpha: bool,
        cpu: CpuCapabilities,
        destination: DecodedSamplesMut<'d>,
        cuts: &[u32],
    ) -> Result<Vec<Self>, JxrError> {
        if !destination.matches_format(request.pixel_format) {
            return Err(map_output_error(
                &OutputFormatError::UnsupportedCombination {
                    combination: "direct destination type differs from the output format",
                },
            ));
        }
        let shapes = plane_shapes(request, has_alpha)?;
        let total = shapes.iter().try_fold(0_usize, |total, shape| {
            shape
                .row_elements
                .checked_mul(shape.crop.height as usize)
                .and_then(|elements| total.checked_add(elements))
        });
        if total != Some(destination.len()) {
            return Err(map_output_error(
                &OutputFormatError::UnsupportedCombination {
                    combination: "direct destination length differs from the output contract",
                },
            ));
        }
        let mut strips: Vec<Self> = (0..=cuts.len())
            .map(|_| Self {
                request,
                cpu,
                planes: Vec::with_capacity(shapes.len()),
            })
            .collect();
        let mut rest = destination;
        for shape in shapes {
            let bottom = shape.crop.y + shape.crop.height;
            let mut next = shape.crop.y;
            for (strip, cut) in strips
                .iter_mut()
                .zip(cuts.iter().copied().chain([u32::MAX]))
            {
                // A cut is a row of the luma plane; a plane of half its
                // height is cut at half the row.
                let end = (cut >> shape.vertical_shift).clamp(next, bottom);
                let elements = shape.row_elements * (end - next) as usize;
                let (remaining, tail) = split_samples(rest, elements);
                rest = tail;
                strip.planes.push(OutputPlane {
                    stored: shape.stored,
                    crop: CropWindow {
                        y: next,
                        height: end - next,
                        ..shape.crop
                    },
                    row_elements: shape.row_elements,
                    next,
                    remaining,
                });
                next = end;
            }
        }
        Ok(strips)
    }

    /// Number of planes stored: one unless the output is planar.
    pub(super) fn plane_count(&self) -> usize {
        self.planes.len()
    }

    /// The component plane `plane` stores, or `None` for an interleaved
    /// output, which stores them all.
    pub(super) fn stored(&self, plane: usize) -> Option<PlanarPlane> {
        self.planes[plane].stored
    }

    /// Rows of its source plane that `plane` has still to store.
    pub(super) fn pending(&self, plane: usize) -> Range<u32> {
        let plane = &self.planes[plane];
        plane.next..plane.crop.y + plane.crop.height
    }

    /// Stores the rows `rows` of `plane`, which must begin at the first
    /// pending row, from component planes that hold them.
    pub(super) fn store(
        &mut self,
        plane: usize,
        components: &[ComponentPlane<'_>],
        alpha: Option<ComponentPlane<'_>>,
        rows: Range<u32>,
    ) -> Result<(), JxrError> {
        let request = self.request;
        let cpu = self.cpu;
        let plane = &mut self.planes[plane];
        if rows.start != plane.next || rows.end > plane.crop.y + plane.crop.height {
            return Err(JxrError::new(
                JxrErrorKind::InternalInvariant,
                "output rows out of order",
            ));
        }
        let crop = CropWindow {
            y: rows.start,
            height: rows.end - rows.start,
            ..plane.crop
        };
        let elements = plane.row_elements * crop.height as usize;
        let empty = DecodedSamplesMut::U8(&mut []);
        let (destination, remaining) =
            split_samples(core::mem::replace(&mut plane.remaining, empty), elements);
        plane.remaining = remaining;
        plane.next = rows.end;
        let result = match plane.stored {
            Some(stored) => {
                let source = if stored.alpha {
                    alpha
                } else {
                    components.get(stored.component).copied()
                }
                .ok_or_else(|| JxrError::new(JxrErrorKind::InternalInvariant, "planar plane"))?;
                format_planar_plane_into(source, stored, crop, request, cpu, destination)
                    .map(|_| ())
            }
            None => format_components_into_with_cpu(
                components,
                alpha,
                OutputFormatRequest { crop, ..request },
                cpu,
                destination,
            )
            .map(|_| ()),
        };
        result.map_err(|error| map_output_error(&error))
    }
}

/// The geometry of one plane of an output.
struct PlaneShape {
    stored: Option<PlanarPlane>,
    crop: CropWindow,
    row_elements: usize,
    /// How many times the plane's rows are halved from the luma plane's.
    vertical_shift: u32,
}

/// The planes of the output `request` describes, in storage order.
fn plane_shapes(
    request: OutputFormatRequest,
    has_alpha: bool,
) -> Result<Vec<PlaneShape>, JxrError> {
    if is_planar(request.output_color) {
        return Ok(planar_planes(request, has_alpha)
            .map_err(|error| map_output_error(&error))?
            .into_iter()
            .map(|stored| PlaneShape {
                stored: Some(stored),
                crop: stored.crop,
                row_elements: stored.crop.width as usize,
                vertical_shift: u32::from(
                    !stored.alpha
                        && stored.component != 0
                        && request.output_color
                            == ColorFormat::Yuv(jxr_core::ChromaSampling::Cs420),
                ),
            })
            .collect());
    }
    let row_elements = request
        .pixel_format
        .row_elements(request.crop.width)
        .ok_or_else(|| JxrError::arithmetic("output row elements"))?;
    Ok(vec![PlaneShape {
        stored: None,
        crop: request.crop,
        row_elements,
        vertical_shift: 0,
    }])
}

/// Descriptors of the planes of the output `request` describes.
pub(super) fn plane_descriptors(
    request: OutputFormatRequest,
    has_alpha: bool,
) -> Result<Vec<PlaneDescriptor>, JxrError> {
    if !is_planar(request.output_color) {
        return Ok(vec![PlaneDescriptor {
            byte_offset: 0,
            row_stride_bytes: request.pixel_format.row_bytes(request.crop.width)?,
            width: request.crop.width,
            height: request.crop.height,
            channels: request.pixel_format.channel_count(),
        }]);
    }
    let mut byte_offset = 0_usize;
    plane_shapes(request, has_alpha)?
        .into_iter()
        .map(|shape| {
            let row_stride_bytes = request
                .pixel_format
                .row_bytes_for_channels(shape.crop.width, 1)?;
            let descriptor = PlaneDescriptor {
                byte_offset,
                row_stride_bytes,
                width: shape.crop.width,
                height: shape.crop.height,
                channels: 1,
            };
            byte_offset = row_stride_bytes
                .checked_mul(shape.crop.height as usize)
                .and_then(|bytes| byte_offset.checked_add(bytes))
                .ok_or_else(|| JxrError::arithmetic("planar byte offset"))?;
            Ok(descriptor)
        })
        .collect()
}

/// Zeroed host storage for the whole output `request` describes.
pub(super) fn allocate_output(
    request: OutputFormatRequest,
    has_alpha: bool,
) -> Result<DecodedSamples, JxrError> {
    let elements = plane_shapes(request, has_alpha)?
        .iter()
        .try_fold(0_usize, |total, shape| {
            shape
                .row_elements
                .checked_mul(shape.crop.height as usize)
                .and_then(|elements| total.checked_add(elements))
        })
        .ok_or_else(|| JxrError::arithmetic("output element count"))?;
    Ok(zeroed_samples(request.pixel_format, elements))
}

/// Whether an output colour format is stored as native subsampled planes.
pub(super) const fn is_planar(output_color: ColorFormat) -> bool {
    matches!(
        output_color,
        ColorFormat::Yuv(jxr_core::ChromaSampling::Cs420 | jxr_core::ChromaSampling::Cs422)
    )
}

/// The report of a CPU decode that stored `output` with `cpu`.
pub(super) fn stream_report(
    request: &DecodeRequest,
    output: OutputFormatRequest,
    has_alpha: bool,
    reconstruction_simd: ReconstructionSimd,
    cpu: CpuCapabilities,
) -> DecodeReport {
    cpu_report(
        request.backend,
        stores_with_vectors(output, has_alpha, cpu),
        reconstruction_simd,
    )
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
    crate::metadata::output_color(
        parsed.headers.image.output_color_format,
        parsed.headers.primary.components,
    )
    .map_err(|_| JxrError::new(JxrErrorKind::InvalidSyntax, "output color format"))
}

/// The report of a CPU decode and whether each stage used at least one native
/// vector group.
fn cpu_report(
    requested: BackendRequest,
    simd_output: bool,
    reconstruction_simd: ReconstructionSimd,
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
                executor: if (simd_output && stage == DecodeStage::CropClipAndPack)
                    || reconstruction_uses(stage, reconstruction_simd)
                {
                    StageExecutor::CpuSimd
                } else {
                    StageExecutor::CpuScalar
                },
                stage,
            })
            .collect(),
    }
}

/// Adds the reconstruction vector groups observed while producing a decoded
/// image to its stage report.
pub(super) fn report_reconstruction_simd(
    report: &mut DecodeReport,
    reconstruction_simd: ReconstructionSimd,
) {
    for entry in &mut report.stages {
        if reconstruction_uses(entry.stage, reconstruction_simd) {
            entry.executor = StageExecutor::CpuSimd;
        }
    }
}

const fn reconstruction_uses(stage: DecodeStage, simd: ReconstructionSimd) -> bool {
    match stage {
        DecodeStage::FirstOverlap => simd.first_overlap,
        DecodeStage::SecondInverseTransform => simd.second_inverse,
        DecodeStage::SecondOverlap => simd.second_overlap,
        _ => false,
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
    use jxr_core::{
        BackendRequest, ChannelLayout, ChromaSampling, ColorFormat, DecodeStage, PixelFormat,
        StageExecutor,
    };

    use super::{OutputBitDepth, OutputFormatRequest, cpu_report, plane_descriptors};
    use crate::reconstruct::{CropWindow, ReconstructionSimd};

    fn planar_request(
        sampling: ChromaSampling,
        bit_depth: OutputBitDepth,
        pixel_format: PixelFormat,
        [width, height]: [u32; 2],
    ) -> OutputFormatRequest {
        OutputFormatRequest {
            internal_color: ColorFormat::Yuv(sampling),
            output_color: ColorFormat::Yuv(sampling),
            bit_depth,
            pixel_format,
            scaled: false,
            alpha_format: None,
            red_blue_not_swapped: true,
            premultiply_alpha: false,
            crop: CropWindow {
                x: 0,
                y: 0,
                width,
                height,
            },
        }
    }

    /// A native planar output is a plane for luma, each chroma component at
    /// its own sampling, and alpha, one after another, of one sample a
    /// channel.
    #[test]
    fn planar_outputs_describe_each_plane_where_it_is_stored() {
        let shapes = |request, alpha| {
            plane_descriptors(request, alpha)
                .unwrap()
                .iter()
                .map(|plane| {
                    assert_eq!(plane.channels, 1);
                    (
                        plane.byte_offset,
                        plane.row_stride_bytes,
                        plane.width,
                        plane.height,
                    )
                })
                .collect::<Vec<_>>()
        };
        let yuv420 = planar_request(
            ChromaSampling::Cs420,
            OutputBitDepth::U8,
            PixelFormat::U8(ChannelLayout::Yuv(ChromaSampling::Cs420)),
            [4, 2],
        );
        assert_eq!(
            shapes(yuv420, false),
            [(0, 4, 4, 2), (8, 2, 2, 1), (10, 2, 2, 1)]
        );
        let yuva422 = planar_request(
            ChromaSampling::Cs422,
            OutputBitDepth::U8,
            PixelFormat::U8(ChannelLayout::Yuva(ChromaSampling::Cs422)),
            [4, 2],
        );
        assert_eq!(
            shapes(yuva422, true),
            [(0, 4, 4, 2), (8, 2, 2, 2), (12, 2, 2, 2), (16, 4, 4, 2)]
        );
        let yuv422_u10 = planar_request(
            ChromaSampling::Cs422,
            OutputBitDepth::U10,
            PixelFormat::U16(ChannelLayout::Yuv(ChromaSampling::Cs422)),
            [2, 1],
        );
        assert_eq!(
            shapes(yuv422_u10, false),
            [(0, 4, 2, 1), (4, 2, 1, 1), (6, 2, 1, 1)]
        );
    }

    #[test]
    fn report_marks_vectorized_reconstruction_and_output_stages() {
        let report = cpu_report(
            BackendRequest::Cpu,
            true,
            ReconstructionSimd {
                first_overlap: true,
                second_inverse: true,
                second_overlap: true,
            },
        );
        assert_eq!(
            report.executor_for(DecodeStage::DequantizeAndFirstInverseTransform),
            Some(StageExecutor::CpuScalar)
        );
        assert_eq!(
            report.executor_for(DecodeStage::CropClipAndPack),
            Some(StageExecutor::CpuSimd)
        );
        assert_eq!(
            report.executor_for(DecodeStage::FirstOverlap),
            Some(StageExecutor::CpuSimd)
        );
        assert_eq!(
            report.executor_for(DecodeStage::SecondInverseTransform),
            Some(StageExecutor::CpuSimd)
        );
        assert_eq!(
            report.executor_for(DecodeStage::SecondOverlap),
            Some(StageExecutor::CpuSimd)
        );
    }
}
