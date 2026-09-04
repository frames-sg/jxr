//! Normative scalar output formatting from reconstructed signed component planes.
//!
//! The implementation follows T.832 clauses 9.10.4 through 9.10.8. Chroma
//! upsampling is intentionally outside this module: callers must supply full-
//! resolution YUV444 planes when requesting RGB output.

mod color;
mod error;
mod packed_color;
mod packing;
mod planar;
mod premultiply;
mod request;
mod scaling;
mod simd_pack;
mod validate;

#[cfg(test)]
mod tests;

pub use error::OutputFormatError;
pub use packing::format_components;
pub(crate) use packing::{format_components_into_with_cpu, format_components_with_cpu};
pub(crate) use planar::{format_planar_yuv, format_planar_yuv_into};
pub use request::{AlphaFormatRequest, ComponentPlane, OutputBitDepth, OutputFormatRequest};
pub use validate::validate_output_policy;
