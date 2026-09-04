//! Reusable JPEG XR decode facade.

use jxr_core::{
    DecodeRequest, DecodeScale, DecodedImage, ImageInfo, JxrError, JxrErrorKind, PreparedPlan, Rect,
};
use jxr_native::{
    CpuCapabilities, CpuDecodeWorkspace, ParsedCodestream, decode_cpu,
    decode_cpu_into_with_workspace, prepare_accelerator_coefficients, prepare_output_format,
    prepare_plan,
};

use crate::PreparedReconstruction;

mod orientation;
mod typed;

use orientation::apply_orientation;
pub use typed::{DecodeIntoResult, DecodeIntoSample};

/// Decoder borrowing parsed input and retaining no hidden global state.
#[derive(Debug)]
pub struct JxrDecoder<'a> {
    bytes: &'a [u8],
    parsed: &'a ParsedCodestream,
    info: &'a ImageInfo,
    cpu: CpuCapabilities,
    #[cfg(feature = "metal")]
    metal: Option<&'a jxr_metal::MetalDecoderSession>,
    #[cfg(feature = "cuda")]
    cuda: Option<&'a jxr_cuda::CudaDecoderSession>,
}

impl<'a> JxrDecoder<'a> {
    pub(crate) fn new(bytes: &'a [u8], parsed: &'a ParsedCodestream, info: &'a ImageInfo) -> Self {
        Self {
            bytes,
            parsed,
            info,
            cpu: CpuCapabilities::detect(),
            #[cfg(feature = "metal")]
            metal: None,
            #[cfg(feature = "cuda")]
            cuda: None,
        }
    }

    /// Attach a reusable Metal session for `Auto` or strict Metal requests.
    #[cfg(feature = "metal")]
    #[must_use]
    pub fn with_metal_session(mut self, session: &'a jxr_metal::MetalDecoderSession) -> Self {
        self.metal = Some(session);
        self
    }

    /// Attach a reusable CUDA session for `Auto` or strict CUDA requests.
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn with_cuda_session(mut self, session: &'a jxr_cuda::CudaDecoderSession) -> Self {
        self.cuda = Some(session);
        self
    }

    /// Parsed image information.
    #[must_use]
    pub const fn info(&self) -> &ImageInfo {
        self.info
    }

    /// Validate and retain the exact geometry and compressed ranges for a request.
    pub fn prepare(&self, request: &DecodeRequest) -> Result<PreparedPlan, JxrError> {
        prepare_plan(self.bytes.len(), self.parsed, request)
    }

    pub(crate) fn prepare_batch_contract(
        &self,
        request: &DecodeRequest,
    ) -> Result<(PreparedPlan, jxr_core::SurfaceLayout), JxrError> {
        let plan = self.prepare(request)?;
        let output = prepare_output_format(self.parsed, &plan, request)?;
        let layout = jxr_core::SurfaceLayout::for_output(output, 1)?;
        Ok((plan, layout))
    }

    pub(crate) fn decode_prepared_cpu_into_with_workspace(
        &self,
        plan: &PreparedPlan,
        request: &DecodeRequest,
        workspace: &mut CpuDecodeWorkspace,
        destination: jxr_core::DecodedSamplesMut<'_>,
    ) -> Result<jxr_native::CpuDecodeIntoOutput, JxrError> {
        decode_cpu_into_with_workspace(
            self.bytes,
            self.parsed,
            plan,
            request,
            self.cpu,
            workspace,
            destination,
        )
    }

    /// Run CPU-owned parsing and entropy stages once for a later reconstruction route.
    pub fn prepare_reconstruction(
        &self,
        request: &DecodeRequest,
    ) -> Result<PreparedReconstruction, JxrError> {
        let plan = self.prepare(request)?;
        self.prepare_reconstruction_from_plan(request, plan)
    }

    pub(crate) fn prepare_reconstruction_from_plan(
        &self,
        request: &DecodeRequest,
        plan: PreparedPlan,
    ) -> Result<PreparedReconstruction, JxrError> {
        validate_accelerator_request(request)?;
        let coefficients =
            prepare_accelerator_coefficients(self.bytes, self.parsed, &plan, request)?;
        let output_policy = prepare_output_format(self.parsed, &plan, request)?;
        let output_layout = jxr_core::SurfaceLayout::for_output(output_policy, 1)?;
        let coded_origin = [
            u32::from(self.parsed.headers.image.margins[1]),
            u32::from(self.parsed.headers.image.margins[0]),
        ];
        Ok(PreparedReconstruction::new(
            plan,
            coefficients,
            coded_origin,
            output_policy,
            output_layout,
            request.backend,
        ))
    }

    /// Prepare an executable CUDA plan from CPU-produced entropy coefficients.
    #[cfg(feature = "cuda")]
    pub fn prepare_cuda(
        &self,
        request: &DecodeRequest,
    ) -> Result<jxr_cuda::CudaDecodePlan, JxrError> {
        self.prepare_reconstruction(request)?
            .cuda_plan()
            .map_err(|error| map_cuda_error(&error))
    }

    /// Prepare a Metal plan by decoding entropy directly into shared Metal storage.
    #[cfg(feature = "metal")]
    pub fn prepare_metal(
        &self,
        request: &DecodeRequest,
        session: &jxr_metal::MetalDecoderSession,
    ) -> Result<jxr_metal::MetalDecodePlan, JxrError> {
        let plan = self.prepare(request)?;
        let coefficient_count = self.metal_coefficient_count_for_plan(&plan)?;
        let staging = session
            .coefficient_staging(coefficient_count)
            .map_err(|error| map_metal_error(&error))?;
        self.prepare_metal_plan_with_staging(request, &plan, staging)
    }

    /// Return the exact primary coefficient count for direct Metal staging.
    #[cfg(feature = "metal")]
    pub fn metal_coefficient_count(&self, request: &DecodeRequest) -> Result<usize, JxrError> {
        validate_accelerator_request(request)?;
        let plan = self.prepare(request)?;
        self.metal_coefficient_count_for_plan(&plan)
    }

    #[cfg(feature = "metal")]
    pub(crate) fn metal_coefficient_count_for_plan(
        &self,
        plan: &PreparedPlan,
    ) -> Result<usize, JxrError> {
        if plan.scale != DecodeScale::Full {
            return Err(native_reduction_is_cpu_only());
        }
        jxr_native::coefficient_count(self.parsed, plan)
    }

    /// Prepare Metal reconstruction into caller-allocated shared staging.
    #[cfg(feature = "metal")]
    pub fn prepare_metal_with_staging(
        &self,
        request: &DecodeRequest,
        staging: jxr_metal::MetalCoefficientStaging,
    ) -> Result<jxr_metal::MetalDecodePlan, JxrError> {
        let plan = self.prepare(request)?;
        self.prepare_metal_plan_with_staging(request, &plan, staging)
    }

    #[cfg(feature = "metal")]
    pub(crate) fn prepare_metal_plan_with_staging(
        &self,
        request: &DecodeRequest,
        plan: &PreparedPlan,
        mut staging: jxr_metal::MetalCoefficientStaging,
    ) -> Result<jxr_metal::MetalDecodePlan, JxrError> {
        validate_accelerator_request(request)?;
        if plan.info.alpha_mode == jxr_core::AlphaMode::Separate {
            return self
                .prepare_reconstruction_from_plan(request, plan.clone())?
                .metal_plan()
                .map_err(|error| map_metal_error(&error));
        }
        let descriptor = staging
            .with_coefficients_mut(|coefficients| {
                jxr_native::decode_coefficients_into(self.bytes, self.parsed, plan, coefficients)
            })
            .map_err(|error| map_metal_error(&error))??;
        let coefficients = std::sync::Arc::new(
            staging
                .seal(descriptor)
                .map_err(|error| map_metal_error(&error))?,
        );
        let output_policy = prepare_output_format(self.parsed, plan, request)?;
        let output = jxr_core::SurfaceLayout::for_output(output_policy, 1)?;
        let coded_origin = [
            u32::from(self.parsed.headers.image.margins[1]),
            u32::from(self.parsed.headers.image.margins[0]),
        ];
        jxr_metal::MetalDecodePlan::from_staged_primary(
            coefficients,
            plan,
            output_policy,
            output,
            coded_origin,
            request.backend,
        )
        .map_err(|error| map_metal_error(&error))
    }

    /// Decode the requested image region and output representation.
    pub fn decode(&mut self, request: &DecodeRequest) -> Result<DecodedImage, JxrError> {
        if request.backend == jxr_core::BackendRequest::Metal {
            validate_accelerator_request(request)?;
        }
        if request.backend == jxr_core::BackendRequest::Cuda {
            validate_accelerator_request(request)?;
            #[cfg(not(feature = "cuda"))]
            return Err(JxrError::new(
                JxrErrorKind::BackendUnavailable,
                "select CUDA decoder session",
            ));
            #[cfg(feature = "cuda")]
            if self.cuda.is_none() {
                return Err(JxrError::new(
                    JxrErrorKind::BackendUnavailable,
                    "select CUDA decoder session",
                ));
            }
        }
        #[cfg(feature = "metal")]
        if request.scale == DecodeScale::Full
            && matches!(
                request.backend,
                jxr_core::BackendRequest::Auto | jxr_core::BackendRequest::Metal
            )
        {
            if let Some(session) = self.metal {
                let prepared = self.prepare(request)?;
                let work = prepared.reconstructed_coefficients()?;
                match jxr_metal::plan_metal_route(request.backend, work, session.is_usable(), false)
                {
                    Ok(jxr_metal::MetalRouteDecision::Metal) => {
                        let metal_plan = self.prepare_metal(request, session)?;
                        return session
                            .decode_to_host(&metal_plan)
                            .map_err(|error| map_metal_error(&error));
                    }
                    Ok(jxr_metal::MetalRouteDecision::Cpu) => {}
                    Err(error) => return Err(map_metal_error(&error)),
                }
            } else if request.backend == jxr_core::BackendRequest::Metal {
                return Err(JxrError::new(
                    JxrErrorKind::BackendUnavailable,
                    "select Metal decoder session",
                ));
            }
        }
        #[cfg(feature = "cuda")]
        if request.scale == DecodeScale::Full
            && matches!(
                request.backend,
                jxr_core::BackendRequest::Auto | jxr_core::BackendRequest::Cuda
            )
        {
            if let Some(session) = self.cuda {
                let prepared = self.prepare(request)?;
                let work = prepared.reconstructed_coefficients()?;
                match jxr_cuda::plan_cuda_route(request.backend, work, session.is_usable(), false) {
                    Ok(jxr_cuda::CudaRouteDecision::Cuda) => {
                        let cuda_plan = self.prepare_cuda(request)?;
                        return session
                            .decode_to_host(&cuda_plan)
                            .map_err(|error| map_cuda_error(&error));
                    }
                    Ok(jxr_cuda::CudaRouteDecision::Cpu) => {}
                    Err(error) => return Err(map_cuda_error(&error)),
                }
            } else if request.backend == jxr_core::BackendRequest::Cuda {
                return Err(JxrError::new(
                    JxrErrorKind::BackendUnavailable,
                    "select CUDA decoder session",
                ));
            }
        }
        let plan = self.prepare(request)?;
        decode_cpu(self.bytes, self.parsed, &plan, request, self.cpu)
    }

    /// Decode host samples and apply the image's presentation orientation.
    ///
    /// Non-identity orientation currently requires a full-image, single-plane
    /// output; one-bit luma rows are repacked after transformation.
    /// [`DecodedImage::info`] continues to describe the encoded source; the
    /// returned region and plane describe oriented pixels.
    pub fn decode_oriented(&mut self, request: &DecodeRequest) -> Result<DecodedImage, JxrError> {
        let orientation = self.info.metadata.orientation;
        let image = self.decode(request)?;
        apply_orientation(image, orientation, request.region.is_some())
    }

    /// Decode an explicit display-space region with the rest of an existing request.
    pub fn decode_region(
        &mut self,
        region: Rect,
        request: &DecodeRequest,
    ) -> Result<DecodedImage, JxrError> {
        let mut request = request.clone();
        request.region = Some(region);
        self.decode(&request)
    }

    /// Decode and copy typed host samples into caller-owned storage.
    ///
    /// The destination may be larger than required. The returned layout reports
    /// the exact decoded region, byte strides, and route used. A destination
    /// whose Rust element type does not match the requested [`jxr_core::PixelFormat`] is
    /// rejected.
    pub fn decode_into<T: DecodeIntoSample>(
        &mut self,
        request: &DecodeRequest,
        destination: &mut [T],
    ) -> Result<DecodeIntoResult, JxrError> {
        if self.can_decode_directly_to_cpu(request) {
            let plan = self.prepare(request)?;
            let output_policy = prepare_output_format(self.parsed, &plan, request)?;
            let layout = jxr_core::SurfaceLayout::for_output(output_policy, 1)?;
            let element_bytes = core::mem::size_of::<T>();
            let required = layout.byte_len.checked_div(element_bytes).ok_or_else(|| {
                JxrError::new(JxrErrorKind::InternalInvariant, "typed output element size")
            })?;
            if required.saturating_mul(element_bytes) != layout.byte_len {
                return Err(JxrError::new(
                    JxrErrorKind::InternalInvariant,
                    "typed output byte alignment",
                ));
            }
            let destination = T::direct_destination(request.output, destination, required)?;
            let mut workspace = CpuDecodeWorkspace::default();
            let decoded = self.decode_prepared_cpu_into_with_workspace(
                &plan,
                request,
                &mut workspace,
                destination,
            )?;
            return Ok(DecodeIntoResult {
                info: decoded.info,
                decoded_region: decoded.decoded_region,
                format: decoded.format,
                planes: decoded.planes,
                report: decoded.report,
            });
        }
        let image = self.decode(request)?;
        T::copy_decoded(&image.samples, destination)?;
        Ok(DecodeIntoResult {
            info: image.info,
            decoded_region: image.decoded_region,
            format: image.format,
            planes: image.planes,
            report: image.report,
        })
    }

    fn can_decode_directly_to_cpu(&self, request: &DecodeRequest) -> bool {
        #[cfg(not(any(feature = "metal", feature = "cuda")))]
        let _ = self;
        if !matches!(
            request.backend,
            jxr_core::BackendRequest::Auto | jxr_core::BackendRequest::Cpu
        ) {
            return false;
        }
        #[cfg(feature = "metal")]
        if self.metal.is_some()
            && request.backend == jxr_core::BackendRequest::Auto
            && request.scale == DecodeScale::Full
        {
            return false;
        }
        #[cfg(feature = "cuda")]
        if self.cuda.is_some()
            && request.backend == jxr_core::BackendRequest::Auto
            && request.scale == DecodeScale::Full
        {
            return false;
        }
        true
    }
}

fn validate_accelerator_request(request: &DecodeRequest) -> Result<(), JxrError> {
    if request.scale == DecodeScale::Full {
        Ok(())
    } else {
        Err(native_reduction_is_cpu_only())
    }
}

const fn native_reduction_is_cpu_only() -> JxrError {
    JxrError::new(
        JxrErrorKind::Unsupported,
        "accelerator reconstruction of native reduced output",
    )
}

#[cfg(feature = "metal")]
fn map_metal_error(error: &jxr_metal::MetalError) -> JxrError {
    let kind = match error {
        jxr_metal::MetalError::Unavailable => JxrErrorKind::BackendUnavailable,
        jxr_metal::MetalError::UnsupportedBackend { .. }
        | jxr_metal::MetalError::ResidentOutputRequiresMetal
        | jxr_metal::MetalError::UnsupportedOutputFormat { .. } => JxrErrorKind::Unsupported,
        jxr_metal::MetalError::KernelArithmetic { .. } => JxrErrorKind::ArithmeticOverflow,
        jxr_metal::MetalError::InvalidPlan { .. }
        | jxr_metal::MetalError::InvalidDestination { .. }
        | jxr_metal::MetalError::InvalidSubmissionState { .. } => JxrErrorKind::InternalInvariant,
        jxr_metal::MetalError::RuntimeInitialization { .. } | jxr_metal::MetalError::Runtime(_) => {
            JxrErrorKind::DeviceFailure
        }
        _ => JxrErrorKind::DeviceFailure,
    };
    JxrError::new(kind, "Metal reconstruction")
}

#[cfg(feature = "cuda")]
pub(crate) fn map_cuda_error(error: &jxr_cuda::CudaError) -> JxrError {
    let kind = match error {
        jxr_cuda::CudaError::Unavailable => JxrErrorKind::BackendUnavailable,
        jxr_cuda::CudaError::UnsupportedBackend { .. }
        | jxr_cuda::CudaError::ResidentOutputRequiresCuda
        | jxr_cuda::CudaError::UnsupportedOutputFormat { .. }
        | jxr_cuda::CudaError::UnsupportedDevice { .. } => JxrErrorKind::Unsupported,
        jxr_cuda::CudaError::KernelArithmetic { .. } => JxrErrorKind::ArithmeticOverflow,
        jxr_cuda::CudaError::InvalidPlan { .. }
        | jxr_cuda::CudaError::InvalidDestination { .. }
        | jxr_cuda::CudaError::InvalidSubmissionState { .. }
        | jxr_cuda::CudaError::StateInvariant { .. } => JxrErrorKind::InternalInvariant,
        _ => JxrErrorKind::DeviceFailure,
    };
    JxrError::new(kind, "CUDA reconstruction")
}

#[cfg(test)]
mod tests {
    use jxr_core::{BackendRequest, DecodeRequest, DecodeScale, PixelFormat};

    use super::validate_accelerator_request;

    #[test]
    fn accelerator_boundary_rejects_native_reduced_decode() {
        let request = DecodeRequest::new(PixelFormat::Rgb101010)
            .with_scale(DecodeScale::Quarter)
            .with_backend(BackendRequest::Auto);
        let error = validate_accelerator_request(&request).unwrap_err();
        assert_eq!(error.kind, jxr_core::JxrErrorKind::Unsupported);

        let full = DecodeRequest::new(PixelFormat::Rgb101010);
        assert!(validate_accelerator_request(&full).is_ok());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_device_pool_limits_are_not_reported_as_host_allocations() {
        let error = super::map_cuda_error(&jxr_cuda::CudaError::ResourceLimit {
            reason: "test device budget",
            requested: 2,
            maximum: 1,
        });
        assert_eq!(error.kind, jxr_core::JxrErrorKind::DeviceFailure);
    }
}
