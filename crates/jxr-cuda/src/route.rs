// SPDX-License-Identifier: MIT OR Apache-2.0

use jxr_core::BackendRequest;

use crate::CudaError;

/// Provisional automatic-routing threshold in reconstructed coefficients.
pub const CUDA_AUTO_THRESHOLD: u64 = 16_384;

/// Concrete route selected by the CUDA planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaRouteDecision {
    /// Keep reconstruction on the portable CPU.
    Cpu,
    /// Submit exact reconstruction to CUDA.
    Cuda,
}

/// Resolve a CUDA route without probing or mutating runtime state.
///
/// Explicit CUDA requests are strict. Automatic routing may select CPU only
/// before submission, including when CUDA is not usable or work is too small.
pub fn plan_cuda_route(
    request: BackendRequest,
    reconstructed_coefficients: u64,
    cuda_usable: bool,
    resident_output: bool,
) -> Result<CudaRouteDecision, CudaError> {
    if resident_output {
        return match request {
            BackendRequest::Cpu => Err(CudaError::ResidentOutputRequiresCuda),
            BackendRequest::Metal => Err(CudaError::UnsupportedBackend { request }),
            BackendRequest::Auto | BackendRequest::Cuda if !cuda_usable => {
                Err(CudaError::Unavailable)
            }
            BackendRequest::Auto | BackendRequest::Cuda => Ok(CudaRouteDecision::Cuda),
        };
    }
    match request {
        BackendRequest::Metal => Err(CudaError::UnsupportedBackend { request }),
        BackendRequest::Cuda if !cuda_usable => Err(CudaError::Unavailable),
        BackendRequest::Cuda => Ok(CudaRouteDecision::Cuda),
        BackendRequest::Auto
            if cuda_usable && reconstructed_coefficients >= CUDA_AUTO_THRESHOLD =>
        {
            Ok(CudaRouteDecision::Cuda)
        }
        BackendRequest::Cpu | BackendRequest::Auto => Ok(CudaRouteDecision::Cpu),
    }
}
