//! Host-side reconstruction planning shared by device adapters.
//!
//! Word arrays match the canonical reconstruction ABI without unsafe casts,
//! padding, or a repacking allocation. Adapters own upload and completion.
//! Callers pass the bounded plane geometry and output layouts validated by
//! their decode-plan constructors; this layer does not allocate coefficient
//! storage or select resource budgets.

mod output;
mod overlap;

pub use output::{OutputDispatchPlan, StorePipeline, build_output_dispatch};
pub use overlap::{OverlapSchedule, first_overlap_schedule, second_overlap_schedule};

/// One overlap work record: first index, second index/stride, kind, reserved.
pub type OverlapWork = [u32; 4];
/// One sample plane: offset, origin x/y, width/height, alpha flag.
pub type SamplePlaneWords = [u32; 6];
/// One output surface: byte offset, row stride, width/height, channels, reserved.
pub type SurfacePlaneWords = [u32; 6];
/// The 28 words of the canonical `JxrOutputAbi` record.
pub type OutputParameterWords = [u32; 28];

/// Mutable descriptor positions checked against the generated ABI by adapters.
pub const SAMPLE_OFFSET: usize = 0;
/// Alpha flag in a sample-plane descriptor.
pub const SAMPLE_ALPHA: usize = 5;
/// Byte offset in an output-surface descriptor.
pub const SURFACE_OFFSET: usize = 0;
/// Logical width in an output-surface descriptor.
pub const SURFACE_WIDTH: usize = 2;
/// Logical height in an output-surface descriptor.
pub const SURFACE_HEIGHT: usize = 3;
/// Selected output plane in `JxrOutputAbi`.
pub const OUTPUT_PLANE: usize = 24;

/// Checked geometry of one reconstruction plane, independent of coefficient storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReconstructionPlane {
    pub arena_index: u32,
    pub macroblock_offset: usize,
    pub macroblock_count: usize,
    pub block_columns: u8,
    pub block_rows: u8,
    pub macroblock_origin_x: u32,
    pub macroblock_origin_y: u32,
    pub macroblocks_x: u32,
    pub macroblocks_y: u32,
    pub sample_origin_x: u32,
    pub sample_origin_y: u32,
    pub sample_width: u32,
    pub sample_height: u32,
    pub low_offset: usize,
    pub sample_offset: usize,
    pub scale_after_first_transform: bool,
    pub alpha: bool,
}

/// Planning failures preserve invalid-contract versus unsupported-format categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanError {
    /// Input cannot be represented by the reconstruction contract.
    InvalidPlan { reason: &'static str },
    /// Output math or format is unsupported by the shared device pipeline.
    UnsupportedOutputFormat { reason: &'static str },
}

impl core::fmt::Display for PlanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidPlan { reason } | Self::UnsupportedOutputFormat { reason } => {
                f.write_str(reason)
            }
        }
    }
}

impl core::error::Error for PlanError {}
