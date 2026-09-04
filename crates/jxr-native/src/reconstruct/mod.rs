//! Scalar JPEG XR component reconstruction.
//!
//! This module owns the normative transform pipeline after entropy decoding and
//! DC/LP prediction. Each full-resolution component uses the same exact scalar
//! operators; chroma interpolation and output colour conversion remain focused
//! stages at the edge of the pipeline.

mod chroma;
mod contract;
mod overlap;
mod packing;
mod pipeline;
#[cfg(test)]
mod pipeline_tests;
mod simd_dequant;
mod subsampled_overlap;
mod transform;

pub use chroma::{ChromaReconstructionConfig, reconstruct_chroma_444};
pub use contract::{
    CropWindow, PlanarSamples, QuantizedMacroblock, ReconstructionConfig, ReconstructionError,
    TilePartition,
};
pub use packing::{pack_luma_u8, pack_luma_u16};
pub use pipeline::reconstruct_luma;
#[cfg(test)]
pub(crate) use pipeline::reconstruct_luma_scaled;
pub(crate) use pipeline::{ReconstructionPipelineWorkspace, reconstruct_luma_scaled_with_cpu};
