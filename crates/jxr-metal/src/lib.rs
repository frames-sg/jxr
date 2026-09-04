// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal planning and resource-lifecycle boundary for JPEG XR reconstruction.
//!
//! CPU entropy output is retained in a five-phase Metal plan. Submission is
//! exposed only because every packaged phase covers the accepted Main-profile
//! matrix; the portable scalar implementation remains the arithmetic oracle.

#[cfg(target_os = "macos")]
mod abi;
#[cfg(target_os = "macos")]
mod buffer_pool;
mod coefficient_staging;
mod destination;
#[cfg(target_os = "macos")]
mod encode;
mod error;
mod kernels;
#[cfg(target_os = "macos")]
mod metal_types;
#[cfg(target_os = "macos")]
mod output_plan;
#[cfg(target_os = "macos")]
mod overlap_plan;
mod plan;
mod resident;
mod route;
#[cfg(target_os = "macos")]
mod runtime;
mod session;
mod shared_image;
mod submission;
#[cfg(target_os = "macos")]
mod upload_cache;

#[cfg(target_os = "macos")]
pub use buffer_pool::{MetalBufferPoolDiagnostics, MetalBufferPoolsDiagnostics};
pub use coefficient_staging::{MetalCoefficientArena, MetalCoefficientStaging};
pub use destination::{DenseMetalBatchLayout, MetalBatchDestination, MetalDestination};
pub use error::MetalError;
pub use jxr_core::SurfaceLayout;
pub use kernels::{KernelStage, MetalKernelManifest, RECONSTRUCTION_KERNELS};
pub use plan::MetalDecodePlan;
pub use resident::{MetalResidentBatch, ResidentMetalImage};
pub use route::{METAL_AUTO_THRESHOLD, MetalRouteDecision, plan_metal_route};
pub use session::MetalDecoderSession;
pub use shared_image::SharedMetalImage;
#[cfg(target_os = "macos")]
pub use submission::MetalConsumerWait;
pub use submission::{
    MetalBatchDestinationCompletion, MetalBatchDestinationSubmission, MetalBatchSubmission,
    MetalDestinationCompletion, MetalDestinationSubmission, MetalResidentBatchSubmission,
    MetalSubmission,
};
#[cfg(target_os = "macos")]
pub use upload_cache::MetalUploadCacheDiagnostics;
