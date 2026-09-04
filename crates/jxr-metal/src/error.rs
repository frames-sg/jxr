// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::BackendRequest;

/// Failure at the JXR Metal adapter boundary.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MetalError {
    /// Metal cannot be used on this host.
    #[error("Metal is unavailable on this host")]
    Unavailable,
    /// The selected GPU is outside the Apple-silicon Metal portability boundary.
    #[error("unsupported Metal device: {reason}")]
    UnsupportedDevice {
        /// Stable capability rejection.
        reason: &'static str,
    },
    /// A backend request cannot be served by this adapter.
    #[error("the Metal adapter does not support backend request {request:?}")]
    UnsupportedBackend {
        /// Rejected request.
        request: BackendRequest,
    },
    /// A CPU request cannot produce a Metal-resident output.
    #[error("Metal-resident output requires the Metal backend")]
    ResidentOutputRequiresMetal,
    /// Decode plan metadata is inconsistent or overflows.
    #[error("invalid Metal decode plan: {reason}")]
    InvalidPlan {
        /// Stable validation failure.
        reason: &'static str,
    },
    /// The destination cannot contain the planned output.
    #[error("invalid Metal destination: {reason}")]
    InvalidDestination {
        /// Stable validation failure.
        reason: &'static str,
    },
    /// A requested JXR output cannot currently be represented by the resident
    /// image contract supplied by the shared Metal runtime.
    #[error("unsupported Metal output format: {reason}")]
    UnsupportedOutputFormat {
        /// Stable description of the rejected representation.
        reason: &'static str,
    },
    /// Lazy Metal pipeline construction failed and the failure is cached by the session.
    #[error("Metal pipeline initialization failed: {message}")]
    RuntimeInitialization {
        /// Metal compiler or pipeline diagnostic.
        message: String,
    },
    /// A checked shader arithmetic operation exceeded the scalar contract.
    #[error("Metal reconstruction arithmetic failed in status stage {status}")]
    KernelArithmetic {
        /// Nonzero stage bits written by the first failing shader phase.
        status: u32,
    },
    /// A submission was used in the wrong lifecycle state.
    #[error("invalid Metal submission state: expected {expected}, found {actual}")]
    InvalidSubmissionState {
        /// Required state.
        expected: &'static str,
        /// Actual state.
        actual: &'static str,
    },
    /// A reusable runtime ledger was poisoned by a panic while mutating it.
    #[error("Metal runtime state is poisoned: {state}")]
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
    /// Shared Metal support rejected a runtime operation.
    #[error("Metal runtime operation failed: {0}")]
    Runtime(#[from] j2k_metal_support::MetalSupportError),
}
