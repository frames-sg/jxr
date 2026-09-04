//! Connected scalar decode pipeline from prepared tile packets to typed output.

mod alpha;
mod output;
mod reconstruction;

pub use output::prepare_output_format;

use jxr_core::{
    BackendRequest, DecodeRequest, DecodedImage, DecodedSamplesMut, JxrError, JxrErrorKind,
    PreparedPlan,
};

use crate::{CpuCapabilities, ParsedCodestream, reconstruct::PlanarSamples};

use self::{
    alpha::{
        reconstruct_separate_alpha, reconstruct_separate_alpha_with_workspace,
        validate_alpha_policy,
    },
    output::{format_image, format_image_into},
    reconstruction::{
        ReconstructionWorkspace, reconstruct_components,
        reconstruct_components_and_integrated_alpha,
        reconstruct_components_and_integrated_alpha_from_arena, reconstruct_components_from_arena,
    },
};

pub(crate) use alpha::prepare_separate_alpha_coefficients;

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
    validate_cpu_route(request.backend)?;
    let decode_alpha = validate_alpha_policy(parsed, plan, request.alpha)?;
    let (reconstruction, alpha_reconstruction) =
        if plan.info.alpha_mode == jxr_core::AlphaMode::Integrated {
            let (components, alpha) =
                reconstruct_components_and_integrated_alpha(source, parsed, plan, cpu)?;
            (components, decode_alpha.then_some(alpha))
        } else {
            let mut primary_plan = plan.clone();
            primary_plan.alpha = None;
            let components = reconstruct_components(source, parsed, &primary_plan, cpu)?;
            let alpha = decode_alpha
                .then(|| reconstruct_separate_alpha(source, parsed, plan, request, cpu))
                .transpose()?;
            (components, alpha)
        };
    format_image(
        parsed,
        plan,
        request,
        &reconstruction,
        alpha_reconstruction.as_ref(),
        cpu,
    )
}

/// Retained native CPU entropy and reconstruction workspace for repeated decode.
#[derive(Debug, Default)]
pub struct CpuDecodeWorkspace {
    primary_arena: jxr_core::CoefficientArena,
    primary_reconstruction: ReconstructionWorkspace,
    separate_alpha_arena: jxr_core::CoefficientArena,
    separate_alpha_reconstruction: ReconstructionWorkspace,
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
    /// Number of decodes that reused an already-sufficient coefficient arena.
    #[must_use]
    pub const fn coefficient_reuses(&self) -> u64 {
        self.coefficient_reuses
    }

    /// Bytes retained by the coefficient vector capacity.
    #[must_use]
    pub fn retained_coefficient_bytes(&self) -> usize {
        let primary = self
            .primary_arena
            .coefficients
            .capacity()
            .saturating_mul(core::mem::size_of::<i32>());
        let alpha = self
            .separate_alpha_arena
            .coefficients
            .capacity()
            .saturating_mul(core::mem::size_of::<i32>());
        primary.saturating_add(alpha)
    }

    /// Number of reconstruction scratch allocations reused across decodes.
    #[must_use]
    pub fn reconstruction_reuses(&self) -> u64 {
        self.primary_reconstruction
            .reuses()
            .saturating_add(self.separate_alpha_reconstruction.reuses())
    }

    /// Bytes retained by component raster, transform, and output-plane capacities.
    #[must_use]
    pub fn retained_reconstruction_bytes(&self) -> usize {
        self.primary_reconstruction
            .retained_bytes()
            .saturating_add(self.separate_alpha_reconstruction.retained_bytes())
    }

    fn recycle_reconstruction(
        &mut self,
        alpha_mode: jxr_core::AlphaMode,
        components: Vec<PlanarSamples>,
        alpha: Option<PlanarSamples>,
    ) {
        let primary_count = components.len();
        self.primary_reconstruction.recycle_components(components);
        if let Some(alpha) = alpha {
            match alpha_mode {
                jxr_core::AlphaMode::Integrated => self
                    .primary_reconstruction
                    .recycle_component(primary_count, alpha),
                jxr_core::AlphaMode::Separate => self
                    .separate_alpha_reconstruction
                    .recycle_component(0, alpha),
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
    let (reconstruction, alpha_reconstruction) =
        reconstruct_with_workspace(source, parsed, plan, request, cpu, decode_alpha, workspace)?;
    let result = format_image(
        parsed,
        plan,
        request,
        &reconstruction,
        alpha_reconstruction.as_ref(),
        cpu,
    );
    workspace.recycle_reconstruction(plan.info.alpha_mode, reconstruction, alpha_reconstruction);
    result
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
    let (reconstruction, alpha_reconstruction) =
        reconstruct_with_workspace(source, parsed, plan, request, cpu, decode_alpha, workspace)?;
    let result = format_image_into(
        parsed,
        plan,
        request,
        &reconstruction,
        alpha_reconstruction.as_ref(),
        cpu,
        destination,
    );
    workspace.recycle_reconstruction(plan.info.alpha_mode, reconstruction, alpha_reconstruction);
    result
}

fn reconstruct_with_workspace(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    request: &DecodeRequest,
    cpu: CpuCapabilities,
    decode_alpha: bool,
    workspace: &mut CpuDecodeWorkspace,
) -> Result<(Vec<PlanarSamples>, Option<PlanarSamples>), JxrError> {
    let reused = crate::coefficient::decode_coefficients_reusing(
        source,
        parsed,
        plan,
        &mut workspace.primary_arena,
    )?;
    workspace.coefficient_reuses = workspace
        .coefficient_reuses
        .saturating_add(u64::from(reused));
    if plan.info.alpha_mode == jxr_core::AlphaMode::Integrated {
        let (components, alpha) = reconstruct_components_and_integrated_alpha_from_arena(
            &workspace.primary_arena,
            parsed,
            plan,
            cpu,
            &mut workspace.primary_reconstruction,
        )?;
        if decode_alpha {
            Ok((components, Some(alpha)))
        } else {
            workspace
                .primary_reconstruction
                .recycle_component(components.len(), alpha);
            Ok((components, None))
        }
    } else {
        let components = reconstruct_components_from_arena(
            &workspace.primary_arena,
            parsed,
            plan,
            cpu,
            &mut workspace.primary_reconstruction,
        )?;
        let alpha = if decode_alpha {
            let (alpha, reused) = reconstruct_separate_alpha_with_workspace(
                source,
                parsed,
                plan,
                request,
                cpu,
                &mut workspace.separate_alpha_arena,
                &mut workspace.separate_alpha_reconstruction,
            )?;
            workspace.coefficient_reuses = workspace
                .coefficient_reuses
                .saturating_add(u64::from(reused));
            Some(alpha)
        } else {
            None
        };
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
