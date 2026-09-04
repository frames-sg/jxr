// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA reconstruction and resource-lifecycle boundary for JPEG XR.
//!
//! Parsing and entropy decoding stay in `jxr-native`; this crate consumes only
//! the validated macroblock-major reconstruction contract from `jxr-core`.

#![deny(missing_docs)]

mod abi;
mod buffer_pool;
mod destination;
mod encode;
mod error;
mod kernels;
mod output_plan;
mod overlap_plan;
mod plan;
mod resident;
mod route;
mod runtime;
mod session;
mod submission;
mod upload_cache;

pub use buffer_pool::CudaBufferPoolDiagnostics;
pub use destination::{CudaBatchDestination, CudaDestination, DenseCudaBatchLayout};
pub use error::CudaError;
pub use kernels::{CudaKernelManifest, KernelStage, RECONSTRUCTION_KERNELS};
pub use plan::CudaDecodePlan;
pub use resident::{CudaResidentBatch, ResidentCudaImage};
pub use route::{CUDA_AUTO_THRESHOLD, CudaRouteDecision, plan_cuda_route};
pub use session::CudaDecoderSession;
pub use submission::{
    CudaBatchDestinationCompletion, CudaBatchDestinationSubmission, CudaBatchSubmission,
    CudaDestinationCompletion, CudaDestinationSubmission, CudaResidentBatchSubmission,
    CudaSubmission,
};
pub use upload_cache::CudaUploadCacheDiagnostics;
