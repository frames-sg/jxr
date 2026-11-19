//! Normative CPU output formatting from reconstructed signed component planes.
//!
//! The implementation follows T.832 clauses 9.10.4 through 9.10.8. Chroma
//! upsampling is intentionally outside this module: callers must supply full-
//! resolution YUV444 planes when requesting RGB output.

pub(crate) mod color;
mod error;
mod interleave;
mod packed_color;
mod packing;
mod planar;
mod premultiply;
mod request;
pub(crate) mod scaling;
mod simd_pack;
mod validate;

#[cfg(test)]
mod tests;

pub use error::OutputFormatError;
pub(crate) use interleave::{InterleavedLoad, InterleavedSample};
pub(crate) use packed_color::{bit_position, unpack_rgb555, unpack_rgb565, unpack_rgb101010};
pub use packing::format_components;
pub(crate) use packing::{
    borrow_samples, format_components_into_with_cpu, is_direct, split_samples, stores_with_vectors,
    zeroed_samples,
};
pub(crate) use planar::{
    PlanarPlane, format_planar_plane_into, format_planar_yuv_into, planar_planes,
};
pub use request::{AlphaFormatRequest, ComponentPlane, OutputBitDepth, OutputFormatRequest};
pub use validate::validate_output_policy;
