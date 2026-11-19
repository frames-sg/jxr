//! Connected scalar decode pipeline from prepared tile packets to typed output.

mod alpha;
mod output;
mod reconstruction;
mod stream;
mod strips;

pub use output::prepare_output_format;

use jxr_core::{
    AlphaMode, BackendRequest, DecodeRequest, DecodeScale, DecodedImage, DecodedSamples,
    DecodedSamplesMut, JxrError, JxrErrorKind, PlaneDescriptor, PreparedPlan,
};

use crate::{
    CpuCapabilities, ParsedCodestream,
    output_format::borrow_samples,
    reconstruct::{PlanarSamples, ReconstructionSimd},
};

use self::{
    alpha::{
        prepare_separate_alpha, reconstruct_separate_alpha_with_workspace, validate_alpha_policy,
    },
    output::{
        allocate_output, format_image, format_image_into, plane_descriptors,
        report_reconstruction_simd, stream_report,
    },
    reconstruction::reconstruct_reduced,
    stream::{Codestream, Stage, StreamWorkspace},
    strips::{Images, StripWorkspace},
};

pub(crate) use alpha::prepare_separate_alpha_coefficients;
pub(crate) use reconstruction::clipped_boundaries;

/// Run the portable scalar reconstruction route for a prepared request.
///
/// Explicit device requests are strict and are rejected here. Device sessions
/// consume the same prepared plan and coefficient arena through their adapter
/// crates rather than silently retrying through this function.
pub fn decode_cpu(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    request: &DecodeRequest,
    cpu: CpuCapabilities,
) -> Result<DecodedImage, JxrError> {
    decode_cpu_with_workspace(
        source,
        parsed,
        plan,
        request,
        cpu,
        &mut CpuDecodeWorkspace::default(),
    )
}

/// Retained native CPU entropy and reconstruction workspace for repeated decode.
#[derive(Debug, Default)]
pub struct CpuDecodeWorkspace {
    /// Storage of each strip a decode is divided into. A reduced-resolution
    /// decode, which is not divided, uses the first.
    strips: Vec<StripWorkspace>,
    coefficient_reuses: u64,
}

/// Metadata returned after native CPU decode into caller-owned typed storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuDecodeIntoOutput {
    /// Parsed source metadata.
    pub info: jxr_core::ImageInfo,
    /// Actual decoded display-space region.
    pub decoded_region: jxr_core::Rect,
    /// Exact pixel format written.
    pub format: jxr_core::PixelFormat,
    /// Plane layout within one image destination.
    pub planes: Vec<jxr_core::PlaneDescriptor>,
    /// CPU route and stage report.
    pub report: jxr_core::DecodeReport,
}

impl CpuDecodeWorkspace {
    /// Number of coefficient workspaces reused with sufficient retained
    /// capacity. A full-resolution stream's alternating arenas count as one;
    /// a decode with separate alpha may reuse one workspace per codestream.
    #[must_use]
    pub const fn coefficient_reuses(&self) -> u64 {
        self.coefficient_reuses
    }

    /// The storage of every image stream of every strip.
    fn streams(&self) -> impl Iterator<Item = &StreamWorkspace> {
        self.strips
            .iter()
            .flat_map(|strip| [&strip.primary, &strip.alpha])
    }

    /// Bytes retained by the coefficient vector capacity.
    ///
    /// A full-resolution decode keeps the coefficients of a band of
    /// macroblock rows, not of the image, so this does not grow with the
    /// height of what was decoded.
    #[must_use]
    pub fn retained_coefficient_bytes(&self) -> usize {
        self.streams()
            .flat_map(|stream| [&stream.arena, &stream.ahead])
            .fold(0_usize, |total, arena| {
                total.saturating_add(
                    arena
                        .coefficients
                        .capacity()
                        .saturating_mul(core::mem::size_of::<i32>()),
                )
            })
    }

    /// Number of reconstruction scratch allocations reused across decodes.
    #[must_use]
    pub fn reconstruction_reuses(&self) -> u64 {
        self.streams()
            .map(|stream| stream.reconstruction.reuses())
            .fold(0_u64, u64::saturating_add)
    }

    /// Bytes retained by component raster, transform, and output-plane capacities.
    #[must_use]
    pub fn retained_reconstruction_bytes(&self) -> usize {
        self.streams()
            .map(|stream| stream.reconstruction.retained_bytes())
            .fold(0_usize, usize::saturating_add)
    }

    fn reduced_simd(&self, separate_alpha: bool) -> ReconstructionSimd {
        let Some(strip) = self.strips.first() else {
            return ReconstructionSimd::default();
        };
        let mut simd = strip.primary.reconstruction.reduced_simd();
        if separate_alpha {
            simd.include(strip.alpha.reconstruction.reduced_simd());
        }
        simd
    }

    /// The storage of the one strip of a decode that is not divided.
    fn whole(&mut self) -> &mut StripWorkspace {
        if self.strips.is_empty() {
            self.strips.push(StripWorkspace::default());
        }
        &mut self.strips[0]
    }

    fn recycle_reconstruction(
        &mut self,
        alpha_mode: jxr_core::AlphaMode,
        components: Vec<PlanarSamples>,
        alpha: Option<PlanarSamples>,
    ) {
        let StripWorkspace {
            primary,
            alpha: separate,
        } = self.whole();
        let primary_count = components.len();
        primary.reconstruction.recycle_components(components);
        if let Some(alpha) = alpha {
            match alpha_mode {
                jxr_core::AlphaMode::Integrated => primary
                    .reconstruction
                    .recycle_component(primary_count, alpha),
                jxr_core::AlphaMode::Separate => {
                    separate.reconstruction.recycle_component(0, alpha);
                }
                jxr_core::AlphaMode::None => {}
            }
        }
    }
}

/// Run CPU decode while retaining coefficient and reconstruction allocations.
pub fn decode_cpu_with_workspace(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    request: &DecodeRequest,
    cpu: CpuCapabilities,
    workspace: &mut CpuDecodeWorkspace,
) -> Result<DecodedImage, JxrError> {
    validate_cpu_route(request.backend)?;
    let decode_alpha = validate_alpha_policy(parsed, plan, request.alpha)?;
    if plan.scale != DecodeScale::Full {
        let (reconstruction, alpha_reconstruction) = reconstruct_reduced_with_workspace(
            source,
            parsed,
            plan,
            request,
            cpu,
            decode_alpha,
            workspace,
        )?;
        let reconstruction_simd = workspace.reduced_simd(
            alpha_reconstruction.is_some() && plan.info.alpha_mode == AlphaMode::Separate,
        );
        let mut result = format_image(
            parsed,
            plan,
            request,
            &reconstruction,
            alpha_reconstruction.as_ref(),
            cpu,
        );
        if let Ok(image) = &mut result {
            report_reconstruction_simd(&mut image.report, reconstruction_simd);
        }
        workspace.recycle_reconstruction(
            plan.info.alpha_mode,
            reconstruction,
            alpha_reconstruction,
        );
        return result;
    }
    let mut samples = None;
    let stored = decode_rows(
        source,
        parsed,
        plan,
        request,
        cpu,
        decode_alpha,
        workspace,
        Target::Allocated(&mut samples),
    )?;
    let samples = samples
        .ok_or_else(|| JxrError::new(JxrErrorKind::InternalInvariant, "allocated output"))?;
    request
        .limits
        .check_host_allocation_bytes(samples.byte_len())?;
    let image = DecodedImage {
        info: plan.info.clone(),
        decoded_region: plan.decoded_region,
        format: request.output,
        planes: stored.planes,
        samples,
        report: stored.report,
    };
    image.validate_layout()?;
    Ok(image)
}

/// Decode directly into an exact-size caller-owned U8 image slice.
pub fn decode_cpu_u8_into_with_workspace(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    request: &DecodeRequest,
    cpu: CpuCapabilities,
    workspace: &mut CpuDecodeWorkspace,
    destination: &mut [u8],
) -> Result<CpuDecodeIntoOutput, JxrError> {
    decode_cpu_into_with_workspace(
        source,
        parsed,
        plan,
        request,
        cpu,
        workspace,
        DecodedSamplesMut::U8(destination),
    )
}

/// Decode directly into an exact-size caller-owned typed image slice.
///
/// A full-resolution decode stores rows as it finishes them, so a decode
/// that fails part way down the image leaves the rows above the failure in
/// `destination`.
pub fn decode_cpu_into_with_workspace(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    request: &DecodeRequest,
    cpu: CpuCapabilities,
    workspace: &mut CpuDecodeWorkspace,
    destination: DecodedSamplesMut<'_>,
) -> Result<CpuDecodeIntoOutput, JxrError> {
    validate_cpu_route(request.backend)?;
    let decode_alpha = validate_alpha_policy(parsed, plan, request.alpha)?;
    if plan.scale != DecodeScale::Full {
        let (reconstruction, alpha_reconstruction) = reconstruct_reduced_with_workspace(
            source,
            parsed,
            plan,
            request,
            cpu,
            decode_alpha,
            workspace,
        )?;
        let reconstruction_simd = workspace.reduced_simd(
            alpha_reconstruction.is_some() && plan.info.alpha_mode == AlphaMode::Separate,
        );
        let mut result = format_image_into(
            parsed,
            plan,
            request,
            &reconstruction,
            alpha_reconstruction.as_ref(),
            cpu,
            destination,
        );
        if let Ok(output) = &mut result {
            report_reconstruction_simd(&mut output.report, reconstruction_simd);
        }
        workspace.recycle_reconstruction(
            plan.info.alpha_mode,
            reconstruction,
            alpha_reconstruction,
        );
        return result;
    }
    let stored = decode_rows(
        source,
        parsed,
        plan,
        request,
        cpu,
        decode_alpha,
        workspace,
        Target::Borrowed(destination),
    )?;
    Ok(CpuDecodeIntoOutput {
        info: plan.info.clone(),
        decoded_region: plan.decoded_region,
        format: request.output,
        planes: stored.planes,
        report: stored.report,
    })
}

/// Where a full-resolution decode stores its rows.
enum Target<'d> {
    /// Storage allocated once the output is known to be well formed.
    Allocated(&'d mut Option<DecodedSamples>),
    /// Caller-owned storage of exactly the output's size.
    Borrowed(DecodedSamplesMut<'d>),
}

/// What a full-resolution decode reports of the output it stored.
struct StoredOutput {
    planes: Vec<PlaneDescriptor>,
    report: jxr_core::DecodeReport,
}

/// Full-resolution decode: entropy decoding, then reconstruction and output
/// a band of macroblock rows at a time.
///
/// A whole-image decode ran each stage over the image before the next, so
/// the stage that failed first in that order named the error. Stages that
/// are prepared here before an earlier one has run to the end of the image
/// therefore hold their error until it has: see [`Stage`].
#[expect(
    clippy::too_many_arguments,
    reason = "the decode entry points' arguments and the output target"
)]
fn decode_rows(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    request: &DecodeRequest,
    cpu: CpuCapabilities,
    decode_alpha: bool,
    workspace: &mut CpuDecodeWorkspace,
    target: Target<'_>,
) -> Result<StoredOutput, JxrError> {
    let integrated = plan.info.alpha_mode == AlphaMode::Integrated;
    let mut failure = None;
    let separate = if decode_alpha && plan.info.alpha_mode == AlphaMode::Separate {
        match prepare_separate_alpha(source, parsed, plan, request) {
            Ok(prepared) => Some(prepared),
            Err(error) => {
                failure = Some((Stage::AlphaEntropy, error));
                None
            }
        }
    } else {
        None
    };
    let images = Images {
        primary: Codestream {
            source,
            parsed,
            plan,
            integrated_alpha: integrated,
        },
        alpha: separate.as_ref().map(|(parsed, plan)| Codestream {
            source,
            parsed,
            plan,
            integrated_alpha: false,
        }),
        integrated_alpha: decode_alpha && integrated,
    };
    let output_request = match prepare_output_format(parsed, plan, request) {
        Ok(output_request) => Some(output_request),
        Err(error) => {
            failure = failure.or(Some((Stage::Output, error)));
            None
        }
    };
    let mut output = None;
    if let (Some(output_request), None) = (output_request, &failure) {
        let destination = match target {
            Target::Borrowed(destination) => Ok(destination),
            Target::Allocated(slot) => allocate_output(output_request, decode_alpha)
                .map(|samples| borrow_samples(slot.insert(samples))),
        };
        match destination {
            Ok(destination) => output = Some((output_request, destination)),
            Err(error) => failure = Some((Stage::Output, error)),
        }
    }
    // Bands are read into arenas retained from the last decode where their
    // complete coefficient and metadata capacity is sufficient.
    let decoded = strips::decode(images, request, cpu, output, &mut workspace.strips, failure);
    workspace.coefficient_reuses = workspace
        .coefficient_reuses
        .saturating_add(decoded.coefficient_reuses);
    if let Some((_, error)) = decoded.failure {
        return Err(error);
    }
    let output_request = output_request
        .ok_or_else(|| JxrError::new(JxrErrorKind::InternalInvariant, "output policy"))?;
    Ok(StoredOutput {
        planes: plane_descriptors(output_request, decode_alpha)?,
        report: stream_report(request, output_request, decode_alpha, decoded.simd, cpu),
    })
}

/// Reduced-resolution reconstruction of the whole image at once.
fn reconstruct_reduced_with_workspace(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    request: &DecodeRequest,
    cpu: CpuCapabilities,
    decode_alpha: bool,
    workspace: &mut CpuDecodeWorkspace,
) -> Result<(Vec<PlanarSamples>, Option<PlanarSamples>), JxrError> {
    let StripWorkspace {
        primary,
        alpha: separate,
    } = workspace.whole();
    let mut reuses = u64::from(crate::coefficient::decode_coefficients_reusing(
        source,
        parsed,
        plan,
        &mut primary.arena,
        cpu.executor(),
    )?);
    if plan.info.alpha_mode == AlphaMode::Integrated {
        let (components, alpha) = reconstruct_reduced(
            &primary.arena,
            parsed,
            plan,
            true,
            cpu,
            &mut primary.reconstruction,
        )?;
        let alpha = alpha.ok_or_else(|| {
            JxrError::new(
                JxrErrorKind::InternalInvariant,
                "integrated alpha reconstruction plane",
            )
        })?;
        workspace.coefficient_reuses = workspace.coefficient_reuses.saturating_add(reuses);
        if decode_alpha {
            Ok((components, Some(alpha)))
        } else {
            workspace
                .whole()
                .primary
                .reconstruction
                .recycle_component(components.len(), alpha);
            Ok((components, None))
        }
    } else {
        let (components, _) = reconstruct_reduced(
            &primary.arena,
            parsed,
            plan,
            false,
            cpu,
            &mut primary.reconstruction,
        )?;
        let alpha = if decode_alpha {
            let (alpha, alpha_reused) = reconstruct_separate_alpha_with_workspace(
                source,
                parsed,
                plan,
                request,
                cpu,
                &mut separate.arena,
                &mut separate.reconstruction,
            )?;
            reuses += u64::from(alpha_reused);
            Some(alpha)
        } else {
            None
        };
        workspace.coefficient_reuses = workspace.coefficient_reuses.saturating_add(reuses);
        Ok((components, alpha))
    }
}

fn validate_cpu_route(request: BackendRequest) -> Result<(), JxrError> {
    match request {
        BackendRequest::Auto | BackendRequest::Cpu => Ok(()),
        BackendRequest::Metal | BackendRequest::Cuda => Err(JxrError::new(
            JxrErrorKind::BackendUnavailable,
            "select explicit accelerator session",
        )),
    }
}
