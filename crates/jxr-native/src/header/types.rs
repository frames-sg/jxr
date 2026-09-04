//! Parsed image and plane header value types.

/// Parsed image-level T.832 syntax.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodestreamHeader {
    /// Output width before orientation is applied.
    pub width: u32,
    /// Output height before orientation is applied.
    pub height: u32,
    /// Boolean header syntax packed behind named accessors.
    pub flags: HeaderFlags,
    /// Preferred T.832 spatial transformation value.
    pub orientation: u8,
    /// Overlap mode in the range 0 through 2.
    pub overlap_mode: u8,
    /// T.832 output colour format code.
    pub output_color_format: u8,
    /// T.832 output bit-depth code.
    pub output_bit_depth: u8,
    /// Explicit tile widths except for the inferred final column.
    pub tile_widths_mb: Vec<u16>,
    /// Explicit tile heights except for the inferred final row.
    pub tile_heights_mb: Vec<u16>,
    /// Top, left, bottom, and right coded margins.
    pub margins: [u8; 4],
}

/// Boolean `IMAGE_HEADER` syntax represented without a many-boolean public struct.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeaderFlags(u16);

impl HeaderFlags {
    const HARD_TILING: u16 = 1 << 0;
    const FREQUENCY_MODE: u16 = 1 << 1;
    const INDEX_TABLE: u16 = 1 << 2;
    const LONG_WORD: u16 = 1 << 3;
    const SHORT_HEADER: u16 = 1 << 4;
    const TRIM_FLEXBITS: u16 = 1 << 5;
    const RED_BLUE_NOT_SWAPPED: u16 = 1 << 6;
    const PREMULTIPLIED_ALPHA: u16 = 1 << 7;
    const ALPHA_PLANE: u16 = 1 << 8;

    pub(super) fn from_parsed(values: [bool; 9]) -> Self {
        let masks = [
            Self::HARD_TILING,
            Self::FREQUENCY_MODE,
            Self::INDEX_TABLE,
            Self::LONG_WORD,
            Self::SHORT_HEADER,
            Self::TRIM_FLEXBITS,
            Self::RED_BLUE_NOT_SWAPPED,
            Self::PREMULTIPLIED_ALPHA,
            Self::ALPHA_PLANE,
        ];
        Self(values.into_iter().zip(masks).fold(
            0,
            |bits, (enabled, mask)| {
                if enabled { bits | mask } else { bits }
            },
        ))
    }

    /// Whether overlap filtering stops at tile boundaries.
    #[must_use]
    pub const fn hard_tiling(self) -> bool {
        self.0 & Self::HARD_TILING != 0
    }
    /// Whether packets use frequency mode rather than spatial mode.
    #[must_use]
    pub const fn frequency_mode(self) -> bool {
        self.0 & Self::FREQUENCY_MODE != 0
    }
    /// Whether a tile index is present.
    #[must_use]
    pub const fn index_table_present(self) -> bool {
        self.0 & Self::INDEX_TABLE != 0
    }
    /// Whether reconstruction values may require long-word storage.
    #[must_use]
    pub const fn long_word(self) -> bool {
        self.0 & Self::LONG_WORD != 0
    }
    /// Whether compact dimension and tile syntax is used.
    #[must_use]
    pub const fn short_header(self) -> bool {
        self.0 & Self::SHORT_HEADER != 0
    }
    /// Whether tile packets can trim flexbits.
    #[must_use]
    pub const fn trim_flexbits(self) -> bool {
        self.0 & Self::TRIM_FLEXBITS != 0
    }
    /// Whether packed RGB keeps red and blue unswapped.
    #[must_use]
    pub const fn red_blue_not_swapped(self) -> bool {
        self.0 & Self::RED_BLUE_NOT_SWAPPED != 0
    }
    /// Whether primary samples are already premultiplied by alpha.
    #[must_use]
    pub const fn premultiplied_alpha(self) -> bool {
        self.0 & Self::PREMULTIPLIED_ALPHA != 0
    }
    /// Whether an integrated alpha plane follows the primary plane.
    #[must_use]
    pub const fn alpha_plane(self) -> bool {
        self.0 & Self::ALPHA_PLANE != 0
    }
}

/// Quantizer values expanded to one value per component.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantizerSet {
    /// Component quantizer values.
    pub components: Vec<u8>,
}

/// Parsed T.832 image-plane header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImagePlaneHeader {
    /// T.832 internal colour format code.
    pub internal_color_format: u8,
    /// Whether output scaling is enabled.
    pub scaled: bool,
    /// T.832 bands-present code.
    pub bands_present: u8,
    /// Number of coded components.
    pub components: u16,
    /// Horizontal chroma centering code.
    pub chroma_centering_x: u8,
    /// Vertical chroma centering code.
    pub chroma_centering_y: u8,
    /// Fixed-point output shift.
    pub shift_bits: u8,
    /// Floating-point mantissa length.
    pub mantissa_length: u8,
    /// Floating-point exponent bias.
    pub exponent_bias: i8,
    /// Image-uniform DC quantizers, when present.
    pub dc_quantizers: Option<QuantizerSet>,
    /// Image-uniform LP quantizers, when present.
    pub lp_quantizers: Option<QuantizerSet>,
    /// Image-uniform HP quantizers, when present.
    pub hp_quantizers: Option<QuantizerSet>,
}

/// Image and plane headers plus the first byte following them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedHeaders {
    /// Image-level syntax.
    pub image: CodestreamHeader,
    /// Primary image plane.
    pub primary: ImagePlaneHeader,
    /// Interleaved alpha image plane, when present.
    pub alpha: Option<ImagePlaneHeader>,
    /// Byte offset of the tile index or subsequent-data field.
    pub bytes_consumed: usize,
}
