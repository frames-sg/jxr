//! Parsed reconstruction plans and macroblock-major coefficient storage.

use alloc::vec::Vec;

use crate::{
    AlphaMode, BandPresence, ByteRange, ChromaSampling, ColorFormat, DecodeScale, ImageInfo,
    JxrError, JxrErrorKind, OverlapMode, Rect,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PredictionMode {
    None,
    FromLeft,
    FromTop,
    FromTopLeft,
}

/// Quantization steps selected for one macroblock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuantizerSet {
    pub dc: u32,
    pub low_pass: u32,
    pub high_pass: u32,
}

/// Tile and image boundary state consumed by overlap kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TileEdgeFlags(u8);

impl TileEdgeFlags {
    pub const LEFT: Self = Self(1 << 0);
    pub const TOP: Self = Self(1 << 1);
    pub const RIGHT: Self = Self(1 << 2);
    pub const BOTTOM: Self = Self(1 << 3);
    pub const HARD_TILE: Self = Self(1 << 4);

    #[must_use]
    pub const fn from_bits(bits: u8) -> Option<Self> {
        if bits & !0x1f == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    #[must_use]
    pub const fn contains(self, flag: Self) -> bool {
        self.0 & flag.0 == flag.0
    }

    #[must_use]
    pub const fn union(self, flag: Self) -> Self {
        Self(self.0 | flag.0)
    }
}

/// Structure-of-arrays metadata for GPU and CPU reconstruction.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MacroblockMetadata {
    pub coefficient_offsets: Vec<u32>,
    pub quantizers: Vec<QuantizerSet>,
    pub bands: Vec<BandPresence>,
    pub predictions: Vec<PredictionMode>,
    /// HP prediction direction selected from reconstructed DC/LP coefficients.
    pub hp_predictions: Vec<PredictionMode>,
    pub tile_edges: Vec<TileEdgeFlags>,
    /// Macroblock column in the extended coded plane.
    pub coded_x: Vec<u32>,
    /// Macroblock row in the extended coded plane.
    pub coded_y: Vec<u32>,
    pub output_x: Vec<u32>,
    pub output_y: Vec<u32>,
}

impl MacroblockMetadata {
    #[must_use]
    pub fn len(&self) -> usize {
        self.coefficient_offsets.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.coefficient_offsets.is_empty()
    }

    pub fn validate(&self, coefficient_count: usize) -> Result<(), JxrError> {
        let expected = self.len();
        let lengths = [
            self.quantizers.len(),
            self.bands.len(),
            self.predictions.len(),
            self.hp_predictions.len(),
            self.tile_edges.len(),
            self.coded_x.len(),
            self.coded_y.len(),
            self.output_x.len(),
            self.output_y.len(),
        ];
        if lengths.iter().any(|&length| length != expected) {
            return Err(JxrError::new(
                JxrErrorKind::InternalInvariant,
                "macroblock metadata lengths",
            ));
        }
        if self.coefficient_offsets.iter().any(|&offset| {
            usize::try_from(offset).map_or(true, |offset| offset > coefficient_count)
        }) {
            return Err(JxrError::new(
                JxrErrorKind::InvalidSyntax,
                "coefficient offset",
            ));
        }
        Ok(())
    }
}

/// Location of a plane within a coefficient arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoefficientPlane {
    pub coefficient_offset: usize,
    pub coefficient_count: usize,
    pub macroblock_offset: usize,
    pub macroblock_count: usize,
    /// Number of 4-sample blocks across one component macroblock.
    pub block_columns: u8,
    /// Number of 4-sample blocks down one component macroblock.
    pub block_rows: u8,
}

/// Macroblock-major signed coefficients plus their reconstruction metadata.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CoefficientArena {
    pub coefficients: Vec<i32>,
    pub macroblocks: MacroblockMetadata,
    pub planes: Vec<CoefficientPlane>,
}

/// Coefficient geometry and metadata for externally owned coefficient storage.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CoefficientArenaDescriptor {
    pub coefficient_count: usize,
    pub macroblocks: MacroblockMetadata,
    pub planes: Vec<CoefficientPlane>,
}

impl CoefficientArenaDescriptor {
    pub fn validate(&self) -> Result<(), JxrError> {
        validate_coefficient_contract(self.coefficient_count, &self.macroblocks, &self.planes)
    }
}

impl CoefficientArena {
    pub fn validate(&self) -> Result<(), JxrError> {
        validate_coefficient_contract(self.coefficients.len(), &self.macroblocks, &self.planes)
    }
}

fn validate_coefficient_contract(
    coefficient_count: usize,
    macroblocks: &MacroblockMetadata,
    planes: &[CoefficientPlane],
) -> Result<(), JxrError> {
    macroblocks.validate(coefficient_count)?;
    for plane in planes {
        if !(1..=4).contains(&plane.block_columns) || !(1..=4).contains(&plane.block_rows) {
            return Err(JxrError::new(
                JxrErrorKind::InternalInvariant,
                "coefficient plane block geometry",
            ));
        }
        check_range(
            plane.coefficient_offset,
            plane.coefficient_count,
            coefficient_count,
            "coefficient plane range",
        )?;
        check_range(
            plane.macroblock_offset,
            plane.macroblock_count,
            macroblocks.len(),
            "macroblock plane range",
        )?;
    }
    Ok(())
}

fn check_range(
    offset: usize,
    length: usize,
    available: usize,
    operation: &'static str,
) -> Result<(), JxrError> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| JxrError::arithmetic(operation))?;
    if end > available {
        Err(JxrError::new(JxrErrorKind::InternalInvariant, operation))
    } else {
        Ok(())
    }
}

/// Per-plane reconstruction geometry and boundary policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanePlan {
    pub width: u32,
    pub height: u32,
    pub macroblocks_x: u32,
    pub macroblocks_y: u32,
    pub overlap: OverlapMode,
    pub coefficient_plane: usize,
}

/// Independently located compressed tile and its affected output region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TilePlan {
    pub packet_range: ByteRange,
    pub output_region: Rect,
    pub macroblock_start: u32,
    pub macroblock_count: u32,
    pub hard_boundaries: bool,
    /// Whether this tile intersects the requested reconstruction window.
    pub required_for_reconstruction: bool,
}

/// Device-neutral result of parse and resource planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedPlan {
    pub info: ImageInfo,
    pub codestream_range: ByteRange,
    pub primary: PlanePlan,
    pub alpha: Option<PlanePlan>,
    pub tiles: Vec<TilePlan>,
    pub reconstruction_region: Rect,
    /// Requested region in full-resolution source coordinates.
    pub output_region: Rect,
    /// Actual output region in the selected native-resolution coordinate grid.
    pub decoded_region: Rect,
    pub scale: DecodeScale,
    pub coefficient_bytes: usize,
}

impl PreparedPlan {
    /// Algorithmic reconstruction work after sparse entropy coefficients have
    /// been expanded into component sample blocks.
    pub fn reconstructed_coefficients(&self) -> Result<u64, JxrError> {
        let macroblocks = self
            .tiles
            .iter()
            .filter(|tile| tile.required_for_reconstruction)
            .try_fold(0_u64, |total, tile| {
                total
                    .checked_add(u64::from(tile.macroblock_count))
                    .ok_or_else(|| JxrError::arithmetic("reconstruction macroblock count"))
            })?;
        let primary = match self.info.primary.color_format {
            ColorFormat::Luma => 256,
            ColorFormat::Yuv(ChromaSampling::Cs420) => 384,
            ColorFormat::Yuv(ChromaSampling::Cs422) => 512,
            ColorFormat::Yuv(ChromaSampling::Cs444) | ColorFormat::Rgb | ColorFormat::Rgbe => 768,
            ColorFormat::Cmyk | ColorFormat::CmykDirect | ColorFormat::YuvK => 1_024,
            ColorFormat::NComponent(components) => u64::from(components)
                .checked_mul(256)
                .ok_or_else(|| JxrError::arithmetic("N-component reconstruction work"))?,
        };
        let alpha = match self.info.alpha_mode {
            AlphaMode::Integrated => 256,
            AlphaMode::Separate if self.alpha.is_some() => 256,
            AlphaMode::None | AlphaMode::Separate => 0,
        };
        macroblocks
            .checked_mul(primary + alpha)
            .ok_or_else(|| JxrError::arithmetic("reconstructed coefficient count"))
    }
}
