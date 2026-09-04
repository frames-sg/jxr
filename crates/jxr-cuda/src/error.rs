// SPDX-License-Identifier: MIT OR Apache-2.0

use jxr_core::BackendRequest;

/// Failure at the JPEG XR CUDA backend boundary.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CudaError {
    /// The CUDA driver or NVRTC runtime is unavailable on this host.
    #[error("CUDA is unavailable on this host")]
    Unavailable,
    /// The selected NVIDIA device does not meet the backend baseline.
    #[error("unsupported CUDA device: {reason}")]
    UnsupportedDevice {
        /// Stable capability rejection.
        reason: &'static str,
    },
    /// A backend request cannot be served by this adapter.
    #[error("the CUDA adapter does not support backend request {request:?}")]
    UnsupportedBackend {
        /// Rejected request.
        request: BackendRequest,
    },
    /// A CPU request cannot produce a CUDA-resident output.
    #[error("CUDA-resident output requires the CUDA backend")]
    ResidentOutputRequiresCuda,
    /// Decode plan metadata is inconsistent or overflows.
    #[error("invalid CUDA decode plan: {reason}")]
    InvalidPlan {
        /// Stable validation failure.
        reason: &'static str,
    },
    /// The destination cannot contain the planned output.
    #[error("invalid CUDA destination: {reason}")]
    InvalidDestination {
        /// Stable validation failure.
        reason: &'static str,
    },
    /// A requested output has no exact CUDA implementation.
    #[error("unsupported CUDA output format: {reason}")]
    UnsupportedOutputFormat {
        /// Stable description of the rejected representation.
        reason: &'static str,
    },
    /// Runtime kernel compilation or module creation failed.
    #[error("CUDA pipeline initialization failed: {message}")]
    RuntimeInitialization {
        /// NVRTC or driver diagnostic.
        message: String,
    },
    /// Checked device arithmetic exceeded the scalar contract.
    #[error("CUDA reconstruction arithmetic failed in status stage {status}")]
    KernelArithmetic {
        /// First nonzero stage code reported by a kernel.
        status: u32,
    },
    /// A submission was used in the wrong lifecycle state.
    #[error("invalid CUDA submission state: expected {expected}, found {actual}")]
    InvalidSubmissionState {
        /// Required state.
        expected: &'static str,
        /// Actual state.
        actual: &'static str,
    },
    /// A bounded CUDA resource budget was exceeded.
    #[error("CUDA resource limit exceeded: {reason} ({requested} > {maximum} bytes)")]
    ResourceLimit {
        /// Stable budget rejection.
        reason: &'static str,
        /// Requested bytes.
        requested: usize,
        /// Maximum permitted bytes.
        maximum: usize,
    },
    /// A reusable runtime ledger was poisoned by a panic while mutating it.
    #[error("CUDA runtime state is poisoned: {state}")]
    StatePoisoned {
        /// Affected runtime resource.
        state: &'static str,
    },
    /// A reusable runtime ledger violated its checked accounting invariants.
    #[error("invalid {state} state: {reason}")]
    StateInvariant {
        /// Affected runtime resource.
        state: &'static str,
        /// Stable invariant failure.
        reason: &'static str,
    },
    /// CUDA driver operation failed after runtime discovery.
    #[error("CUDA driver operation failed: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
}
