// SPDX-License-Identifier: MIT OR Apache-2.0

/// Failure at the JPEG XR codec-to-MPSGraph boundary.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The current target is not Apple Silicon macOS.
    #[error("MPSGraph integration requires Apple Silicon macOS 11 or newer")]
    UnsupportedPlatform,
    /// A requested or derived tensor contract cannot be represented.
    #[error("invalid MPSGraph tensor contract: {reason}")]
    InvalidTensorContract { reason: &'static str },
    /// Tensor shape or byte-length arithmetic overflowed `usize`.
    #[error("MPSGraph tensor shape arithmetic overflow")]
    TensorShapeOverflow,
    /// `MPSGraph` reported an asynchronous execution error.
    #[error("MPSGraph execution failed ({domain}, code {code}): {description}")]
    GraphExecution {
        domain: String,
        code: isize,
        description: String,
    },
    /// `MPSGraph` completed without returning a requested target.
    #[error("MPSGraph did not return target result {index}")]
    MissingGraphOutput { index: usize },
    /// JPEG XR parsing, validation, or CPU entropy preparation failed.
    #[error("JPEG XR preparation failed: {0}")]
    Jxr(#[from] jxr_core::JxrError),
    /// Shared owned-batch infrastructure failed before graph submission.
    #[error("JPEG XR batch preparation failed: {0}")]
    Batch(#[from] jxr::BatchInfrastructureError),
    /// The Metal codec layer failed.
    #[error("Metal codec operation failed: {0}")]
    Metal(#[from] jxr_metal::MetalError),
    /// The shared Metal runtime layer failed.
    #[error("Metal runtime operation failed: {0}")]
    MetalRuntime(#[from] j2k_metal_support::MetalSupportError),
}
