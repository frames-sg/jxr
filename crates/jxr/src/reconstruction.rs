//! CPU-prepared coefficient ownership shared by accelerator plans.

use std::sync::Arc;

use jxr_core::{
    BackendRequest, CoefficientArena, OutputFormatRequest, PreparedPlan, SurfaceLayout,
};

/// Independently encoded Annex-A alpha retained for accelerator reconstruction.
#[derive(Debug, Clone)]
pub struct PreparedAlphaReconstruction {
    plan: PreparedPlan,
    coefficients: Arc<CoefficientArena>,
}

impl PreparedAlphaReconstruction {
    /// Validated reconstruction plan for the independent alpha codestream.
    #[must_use]
    pub const fn plan(&self) -> &PreparedPlan {
        &self.plan
    }

    /// Shared luma coefficient arena for the alpha codestream.
    #[must_use]
    pub const fn coefficients(&self) -> &Arc<CoefficientArena> {
        &self.coefficients
    }
}

/// Immutable CPU entropy output and geometry ready for accelerator reconstruction.
#[derive(Debug, Clone)]
pub struct PreparedReconstruction {
    plan: PreparedPlan,
    coefficients: Arc<CoefficientArena>,
    separate_alpha: Option<PreparedAlphaReconstruction>,
    coded_origin: [u32; 2],
    output_policy: OutputFormatRequest,
    output_layout: SurfaceLayout,
    requested_backend: BackendRequest,
}

impl PreparedReconstruction {
    pub(crate) fn new(
        plan: PreparedPlan,
        coefficients: jxr_native::AcceleratorCoefficients,
        coded_origin: [u32; 2],
        output_policy: OutputFormatRequest,
        output_layout: SurfaceLayout,
        requested_backend: BackendRequest,
    ) -> Self {
        let separate_alpha = coefficients
            .separate_alpha
            .map(|alpha| PreparedAlphaReconstruction {
                plan: alpha.plan,
                coefficients: Arc::new(alpha.coefficients),
            });
        Self {
            plan,
            coefficients: Arc::new(coefficients.primary),
            separate_alpha,
            coded_origin,
            output_policy,
            output_layout,
            requested_backend,
        }
    }

    /// Validated device-neutral reconstruction plan.
    #[must_use]
    pub const fn plan(&self) -> &PreparedPlan {
        &self.plan
    }

    /// Shared macroblock-major coefficient arena produced on the CPU.
    #[must_use]
    pub const fn coefficients(&self) -> &Arc<CoefficientArena> {
        &self.coefficients
    }

    /// Left and top coded margins in reconstructed sample coordinates.
    #[must_use]
    pub const fn coded_origin(&self) -> [u32; 2] {
        self.coded_origin
    }

    /// Separate alpha entropy output, when Annex A stores it independently.
    #[must_use]
    pub const fn separate_alpha(&self) -> Option<&PreparedAlphaReconstruction> {
        self.separate_alpha.as_ref()
    }

    /// Exact output conversion and packing policy shared with the CPU path.
    #[must_use]
    pub const fn output_policy(&self) -> OutputFormatRequest {
        self.output_policy
    }

    /// Default tightly packed output surface for the original request.
    #[must_use]
    pub const fn output_layout(&self) -> &SurfaceLayout {
        &self.output_layout
    }

    /// Backend policy carried by the original decode request.
    #[must_use]
    pub const fn requested_backend(&self) -> BackendRequest {
        self.requested_backend
    }

    /// Build a Metal plan retaining this handoff's coefficient storage.
    #[cfg(feature = "metal")]
    pub fn metal_plan(&self) -> Result<jxr_metal::MetalDecodePlan, jxr_metal::MetalError> {
        self.metal_plan_with_layout(self.output_layout.clone())
    }

    /// Build a Metal plan using a caller-selected pitched or planar layout.
    #[cfg(feature = "metal")]
    pub fn metal_plan_with_layout(
        &self,
        output: SurfaceLayout,
    ) -> Result<jxr_metal::MetalDecodePlan, jxr_metal::MetalError> {
        jxr_metal::MetalDecodePlan::from_prepared(
            self.coefficients.clone(),
            self.separate_alpha
                .as_ref()
                .map(|alpha| (alpha.coefficients.clone(), alpha.plan.clone())),
            &self.plan,
            self.output_policy,
            output,
            self.coded_origin,
            self.requested_backend,
        )
    }

    /// Build a CUDA plan retaining this handoff's coefficient storage.
    #[cfg(feature = "cuda")]
    pub fn cuda_plan(&self) -> Result<jxr_cuda::CudaDecodePlan, jxr_cuda::CudaError> {
        self.cuda_plan_with_layout(self.output_layout.clone())
    }

    /// Build a CUDA plan using a caller-selected pitched or planar layout.
    #[cfg(feature = "cuda")]
    pub fn cuda_plan_with_layout(
        &self,
        output: SurfaceLayout,
    ) -> Result<jxr_cuda::CudaDecodePlan, jxr_cuda::CudaError> {
        jxr_cuda::CudaDecodePlan::from_prepared(
            self.coefficients.clone(),
            self.separate_alpha
                .as_ref()
                .map(|alpha| (alpha.coefficients.clone(), alpha.plan.clone())),
            &self.plan,
            self.output_policy,
            output,
            self.coded_origin,
            self.requested_backend,
        )
    }
}
