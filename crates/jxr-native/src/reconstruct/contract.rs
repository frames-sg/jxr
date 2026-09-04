//! Input, output, and failure contracts for scalar reconstruction.

use core::fmt;

use jxr_core::{BandPresence, OverlapMode, QuantizerSet};

pub use jxr_core::CropWindow;

/// Quantized coefficients for one full-resolution component macroblock.
///
/// `dc_low_pass` is the 4-by-4 DC/LP array in raster order. `high_pass` contains
/// sixteen consecutive 4-by-4 coefficient blocks in raster block order; element
/// zero of each block is ignored because it is supplied by `dc_low_pass`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedMacroblock {
    /// Prediction-resolved DC and low-pass coefficients.
    pub dc_low_pass: [i32; 16],
    /// High-pass coefficients. The coefficient arena retains residuals; the
    /// scalar handoff resolves its recorded prediction before reconstruction.
    pub high_pass: [i32; 256],
    /// Quantization steps selected for this macroblock.
    pub quantizers: QuantizerSet,
    /// Bands present in the codestream.
    pub bands: BandPresence,
}

/// Macroblock tile partition used to select soft- or hard-edge overlap behavior.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TilePartition {
    /// Monotonic macroblock x coordinates, including zero and the plane width.
    pub column_boundaries: Vec<u32>,
    /// Monotonic macroblock y coordinates, including zero and the plane height.
    pub row_boundaries: Vec<u32>,
    /// Whether tile boundaries use independent boundary operators.
    pub hard_boundaries: bool,
}

impl TilePartition {
    /// Construct a single tile covering the plane.
    #[must_use]
    pub fn single(macroblocks_x: u32, macroblocks_y: u32) -> Self {
        Self {
            column_boundaries: vec![0, macroblocks_x],
            row_boundaries: vec![0, macroblocks_y],
            hard_boundaries: false,
        }
    }
}

/// Reconstruction geometry and overlap policy for a full-resolution luma plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconstructionConfig {
    /// First macroblock column represented by the input slice.
    pub macroblock_origin_x: u32,
    /// First macroblock row represented by the input slice.
    pub macroblock_origin_y: u32,
    /// Plane width in macroblocks, including coded margins.
    pub macroblocks_x: u32,
    /// Plane height in macroblocks, including coded margins.
    pub macroblocks_y: u32,
    /// Number of 4-sample transform blocks across each component macroblock.
    pub block_columns: u8,
    /// Number of 4-sample transform blocks down each component macroblock.
    pub block_rows: u8,
    /// Apply the normative post-first-transform chroma factor for scaled streams.
    pub scale_after_first_transform: bool,
    /// Normative overlap-filtering mode.
    pub overlap: OverlapMode,
    /// Tile boundaries and hard/soft policy.
    pub tiles: TilePartition,
}

/// Reconstructed signed samples before bias, post-scaling, clipping, and packing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanarSamples {
    /// Horizontal origin in the extended plane's sample coordinates.
    pub origin_x: u32,
    /// Vertical origin in the extended plane's sample coordinates.
    pub origin_y: u32,
    /// Plane width in samples.
    pub width: u32,
    /// Plane height in samples.
    pub height: u32,
    /// Row-major signed samples.
    pub samples: Vec<i32>,
}

/// Failure returned by the scalar reconstruction stages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconstructionError {
    /// A checked size or coefficient operation overflowed.
    ArithmeticOverflow(&'static str),
    /// The requested reconstruction syntax has no implemented exact operator.
    Unsupported(&'static str),
    /// Plane dimensions do not match the declared sampling geometry.
    InvalidPlaneGeometry(&'static str),
    /// Coefficient count does not match reconstruction geometry.
    MacroblockCount {
        /// Count implied by plane geometry.
        expected: usize,
        /// Count supplied by the entropy/prediction stage.
        actual: usize,
    },
    /// A quantization step was zero.
    ZeroQuantizer(&'static str),
    /// Tile coordinates did not form a complete, strictly increasing partition.
    InvalidTilePartition(&'static str),
    /// A crop extends beyond the reconstructed plane.
    CropOutsidePlane,
    /// A destination has insufficient elements for the requested crop.
    BufferTooSmall {
        /// Minimum number of required elements.
        required: usize,
        /// Number of available elements.
        available: usize,
    },
}

impl fmt::Display for ReconstructionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArithmeticOverflow(operation) => write!(formatter, "overflow during {operation}"),
            Self::Unsupported(feature) => {
                write!(formatter, "unsupported reconstruction: {feature}")
            }
            Self::InvalidPlaneGeometry(reason) => {
                write!(formatter, "invalid plane geometry: {reason}")
            }
            Self::MacroblockCount { expected, actual } => {
                write!(
                    formatter,
                    "expected {expected} macroblocks, received {actual}"
                )
            }
            Self::ZeroQuantizer(band) => write!(formatter, "zero {band} quantization step"),
            Self::InvalidTilePartition(axis) => write!(formatter, "invalid {axis} tile partition"),
            Self::CropOutsidePlane => {
                formatter.write_str("crop extends outside reconstructed plane")
            }
            Self::BufferTooSmall {
                required,
                available,
            } => {
                write!(
                    formatter,
                    "destination needs {required} elements, has {available}"
                )
            }
        }
    }
}

impl std::error::Error for ReconstructionError {}
