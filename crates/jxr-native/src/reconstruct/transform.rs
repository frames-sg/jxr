//! Native error adapters for canonical `jxr-math` transform operators.

use jxr_math::transform::{
    inverse_chroma_420 as inverse_420, inverse_chroma_422 as inverse_422,
    inverse_core_transform as inverse_transform,
};

use super::ReconstructionError;

pub(super) fn inverse_core_transform(
    coefficients: &mut [i32; 16],
) -> Result<(), ReconstructionError> {
    inverse_transform(coefficients).map_err(|_| overflow("inverse core transform"))
}

pub(super) fn inverse_chroma_420(coefficients: &mut [i32; 4]) -> Result<(), ReconstructionError> {
    inverse_420(coefficients).map_err(|_| overflow("YUV420 first inverse transform"))
}

pub(super) fn inverse_chroma_422(coefficients: &mut [i32; 8]) -> Result<(), ReconstructionError> {
    inverse_422(coefficients).map_err(|_| overflow("YUV422 first inverse transform"))
}

const fn overflow(operation: &'static str) -> ReconstructionError {
    ReconstructionError::ArithmeticOverflow(operation)
}
