//! Parsed image metadata and byte-range descriptors.

use alloc::vec::Vec;

use crate::{
    AlphaMode, AnnexAPixelFormat, BandPresence, BitstreamMode, ColorFormat, JxrError, JxrErrorKind,
    Level, Orientation, OverlapMode, Profile, SampleFormat,
};

/// A checked half-open range in retained compressed input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ByteRange {
    pub offset: usize,
    pub length: usize,
}

impl ByteRange {
    pub fn new(offset: usize, length: usize, input_len: usize) -> Result<Self, JxrError> {
        let end = offset
            .checked_add(length)
            .ok_or_else(|| JxrError::arithmetic("compressed byte range"))?;
        if end > input_len {
            return Err(JxrError::new(
                JxrErrorKind::Truncated,
                "compressed byte range",
            ));
        }
        Ok(Self { offset, length })
    }

    #[must_use]
    pub const fn end(self) -> Option<usize> {
        self.offset.checked_add(self.length)
    }
}

/// Explicit JPEG XR tile partitioning in macroblock units.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileGrid {
    pub column_widths: Vec<u32>,
    pub row_heights: Vec<u32>,
    pub hard_tiles: bool,
}

impl TileGrid {
    pub fn tile_count(&self) -> Result<u32, JxrError> {
        let columns = u32::try_from(self.column_widths.len())
            .map_err(|_| JxrError::arithmetic("tile columns"))?;
        let rows =
            u32::try_from(self.row_heights.len()).map_err(|_| JxrError::arithmetic("tile rows"))?;
        columns
            .checked_mul(rows)
            .ok_or_else(|| JxrError::arithmetic("tile count"))
    }
}

/// Metadata for one coded primary or alpha plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaneInfo {
    pub color_format: ColorFormat,
    pub sample_format: SampleFormat,
    pub bands: BandPresence,
    pub bitstream_mode: BitstreamMode,
    pub overlap: OverlapMode,
    pub short_header: bool,
    pub long_word: bool,
    pub scaled: bool,
    /// Horizontal and vertical T.832 chroma-centering codes.
    pub chroma_centering: [u8; 2],
    pub shift_bits: u8,
    pub mantissa_length: u8,
    pub exponent_bias: i8,
    pub width: u32,
    pub height: u32,
}

/// Metadata reported without applying presentation policy.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ImageMetadata {
    pub orientation: Orientation,
    pub icc_profile: Option<ByteRange>,
    /// Raw Annex-A pixel-format identifier.
    pub container_pixel_format: Option<[u8; 16]>,
    /// Typed Annex-A pixel-format classification.
    pub annex_a_pixel_format: Option<AnnexAPixelFormat>,
}

/// Parsed image information shared by inspection and decode results.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub profile: Option<Profile>,
    pub level: Option<Level>,
    pub primary: PlaneInfo,
    pub alpha_mode: AlphaMode,
    /// Whether decoded color samples are already premultiplied by alpha.
    pub premultiplied_alpha: bool,
    pub alpha: Option<PlaneInfo>,
    pub tiles: TileGrid,
    pub metadata: ImageMetadata,
}

impl ImageInfo {
    #[must_use]
    pub const fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn validate_consistency(&self) -> Result<(), JxrError> {
        if self.width == 0 || self.height == 0 {
            return Err(JxrError::new(
                JxrErrorKind::InvalidSyntax,
                "image dimensions",
            ));
        }
        if self.primary.width != self.width || self.primary.height != self.height {
            return Err(JxrError::new(
                JxrErrorKind::InvalidSyntax,
                "primary plane dimensions",
            ));
        }
        let alpha_present = self.alpha.is_some();
        if alpha_present == matches!(self.alpha_mode, AlphaMode::None) {
            return Err(JxrError::new(
                JxrErrorKind::InvalidSyntax,
                "alpha declaration",
            ));
        }
        if self.premultiplied_alpha && !alpha_present {
            return Err(JxrError::new(
                JxrErrorKind::InvalidSyntax,
                "premultiplied alpha declaration",
            ));
        }
        if let Some(alpha) = &self.alpha
            && (alpha.width != self.width || alpha.height != self.height)
        {
            return Err(JxrError::new(
                JxrErrorKind::InvalidSyntax,
                "alpha plane dimensions",
            ));
        }
        Ok(())
    }
}
