// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::BackendRequest;

use crate::MetalError;

/// Provisional auto-routing threshold, measured in reconstructed coefficients.
pub const METAL_AUTO_THRESHOLD: u64 = 16_384;

/// Concrete route selected by the Metal adapter planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalRouteDecision {
    /// Keep reconstruction on the CPU.
    Cpu,
    /// Execute reconstruction using Metal.
    Metal,
}

/// Resolve a request without probing or mutating runtime state.
///
/// Resident output forces the device route. Explicit Metal is strict, while an
/// automatic request may select CPU when Metal is unavailable or the workload
/// is below the provisional threshold. `metal_usable` must include both device
/// availability and complete-pipeline readiness.
pub fn plan_metal_route(
    request: BackendRequest,
    reconstructed_coefficients: u64,
    metal_usable: bool,
    resident_output: bool,
) -> Result<MetalRouteDecision, MetalError> {
    if resident_output {
        return match request {
            BackendRequest::Cpu => Err(MetalError::ResidentOutputRequiresMetal),
            BackendRequest::Cuda => Err(MetalError::UnsupportedBackend { request }),
            BackendRequest::Auto | BackendRequest::Metal if !metal_usable => {
                Err(MetalError::Unavailable)
            }
            BackendRequest::Auto | BackendRequest::Metal => Ok(MetalRouteDecision::Metal),
        };
    }
    match request {
        BackendRequest::Cuda => Err(MetalError::UnsupportedBackend { request }),
        BackendRequest::Metal if !metal_usable => Err(MetalError::Unavailable),
        BackendRequest::Metal => Ok(MetalRouteDecision::Metal),
        BackendRequest::Auto
            if metal_usable && reconstructed_coefficients >= METAL_AUTO_THRESHOLD =>
        {
            Ok(MetalRouteDecision::Metal)
        }
        BackendRequest::Cpu | BackendRequest::Auto => Ok(MetalRouteDecision::Cpu),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_promotes_at_threshold() {
        assert_eq!(
            plan_metal_route(BackendRequest::Auto, METAL_AUTO_THRESHOLD, true, false).unwrap(),
            MetalRouteDecision::Metal
        );
        assert_eq!(
            plan_metal_route(BackendRequest::Auto, METAL_AUTO_THRESHOLD - 1, true, false).unwrap(),
            MetalRouteDecision::Cpu
        );
    }

    #[test]
    fn auto_falls_back_before_submission() {
        assert_eq!(
            plan_metal_route(BackendRequest::Auto, u64::MAX, false, false).unwrap(),
            MetalRouteDecision::Cpu
        );
    }

    #[test]
    fn explicit_metal_is_strict() {
        assert!(matches!(
            plan_metal_route(BackendRequest::Metal, 1, false, false),
            Err(MetalError::Unavailable)
        ));
    }

    #[test]
    fn resident_output_forces_available_metal() {
        assert_eq!(
            plan_metal_route(BackendRequest::Auto, 1, true, true).unwrap(),
            MetalRouteDecision::Metal
        );
        assert!(matches!(
            plan_metal_route(BackendRequest::Auto, 1, false, true),
            Err(MetalError::Unavailable)
        ));
        assert!(matches!(
            plan_metal_route(BackendRequest::Cpu, 1, true, true),
            Err(MetalError::ResidentOutputRequiresMetal)
        ));
    }
}
