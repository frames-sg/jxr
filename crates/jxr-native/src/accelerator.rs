//! CPU entropy output prepared for accelerator reconstruction.

use jxr_core::{CoefficientArena, DecodeRequest, JxrError, PreparedPlan};

use crate::{ParsedCodestream, decode::prepare_separate_alpha_coefficients};

/// Coefficient arenas needed by one accelerator reconstruction request.
#[derive(Debug)]
pub struct AcceleratorCoefficients {
    /// Primary components and integrated alpha, when present.
    pub primary: CoefficientArena,
    /// Separately encoded Annex-A alpha and its independent plan.
    pub separate_alpha: Option<PreparedAlphaCoefficients>,
}

/// Prepared coefficient arena for a separate Annex-A alpha codestream.
#[derive(Debug)]
pub struct PreparedAlphaCoefficients {
    /// Alpha codestream reconstruction plan.
    pub plan: PreparedPlan,
    /// One luma coefficient plane.
    pub coefficients: CoefficientArena,
}

/// Decode entropy and prediction-owned stages once for an accelerator backend.
pub fn prepare_accelerator_coefficients(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    request: &DecodeRequest,
) -> Result<AcceleratorCoefficients, JxrError> {
    let primary = crate::decode_coefficients(source, parsed, plan)?;
    let separate_alpha = prepare_separate_alpha_coefficients(source, parsed, plan, request)?
        .map(|(plan, coefficients)| PreparedAlphaCoefficients { plan, coefficients });
    Ok(AcceleratorCoefficients {
        primary,
        separate_alpha,
    })
}
