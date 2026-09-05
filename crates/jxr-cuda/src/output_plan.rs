// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{CudaDecodePlan, CudaError};
pub(crate) use jxr_core::device_plan::{OutputDispatchPlan, StorePipeline};

pub(crate) fn build_output_dispatch(
    plan: &CudaDecodePlan,
) -> Result<OutputDispatchPlan, CudaError> {
    let input = plan.reconstruction()?;
    let policy = plan.output_policy().ok_or(CudaError::InvalidPlan {
        reason: "output policy is absent",
    })?;
    let info = plan.info().ok_or(CudaError::InvalidPlan {
        reason: "image metadata is absent",
    })?;
    jxr_core::device_plan::build_output_dispatch(&input.planes, plan.output(), policy, info)
        .map_err(Into::into)
}
