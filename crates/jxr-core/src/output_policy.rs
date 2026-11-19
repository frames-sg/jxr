//! Device-neutral reconstruction output policy shared by CPU and accelerators.

use crate::{AnnexABitDepth, ColorFormat, PixelFormat, SampleFormat};

/// A checked crop window within reconstructed-plane coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CropWindow {
    /// Left sample coordinate.
    pub x: u32,
    /// Top sample coordinate.
    pub y: u32,
    /// Number of samples per output row.
    pub width: u32,
    /// Number of output rows.
    pub height: u32,
}

/// Normative T.832 output sample representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OutputBitDepth {
    /// One bit per luma sample, where one represents white.
    Bit1White,
    /// One bit per luma sample, where one represents black.
    Bit1Black,
    /// Unsigned 8-bit components.
    U8,
    /// Unsigned 10-bit components stored in typed 16-bit samples.
    U10,
    /// Unsigned 16-bit components with integer postshift syntax.
    U16 { shift_bits: u8 },
    /// Signed 16-bit components with integer postshift syntax.
    I16 { shift_bits: u8 },
    /// T.832 sign-plus-15-bit-magnitude floating representation.
    F16,
    /// Signed 32-bit components with integer postshift syntax.
    I32 { shift_bits: u8 },
    /// IEEE binary32 reconstructed from integer float syntax.
    F32 {
        mantissa_length: u8,
        exponent_bias: i8,
    },
    /// Packed 5:5:5 RGB.
    Rgb555,
    /// Packed 10:10:10 RGB.
    Rgb101010,
    /// Packed 5:6:5 RGB.
    Rgb565,
}

impl OutputBitDepth {
    /// The `OUTPUT_BITDEPTH` code and the plane-header fields that qualify
    /// it: `(code, SHIFT_BITS, mantissa length, exponent bias)`.
    ///
    /// This is the inverse of [`Self::from_header_fields`]; `Rgb101010` is
    /// the RGB reading of the ten-bit code.
    #[must_use]
    pub const fn header_fields(self) -> (u8, u8, u8, i8) {
        match self {
            Self::Bit1White => (0, 0, 0, 0),
            Self::U8 => (1, 0, 0, 0),
            Self::U16 { shift_bits } => (2, shift_bits, 0, 0),
            Self::I16 { shift_bits } => (3, shift_bits, 0, 0),
            Self::F16 => (4, 0, 0, 0),
            Self::I32 { shift_bits } => (6, shift_bits, 0, 0),
            Self::F32 {
                mantissa_length,
                exponent_bias,
            } => (7, 0, mantissa_length, exponent_bias),
            Self::Rgb555 => (8, 0, 0, 0),
            Self::U10 | Self::Rgb101010 => (9, 0, 0, 0),
            Self::Rgb565 => (10, 0, 0, 0),
            Self::Bit1Black => (15, 0, 0, 0),
        }
    }

    /// The sample representation of this depth: for packed RGB, that of
    /// its widest field, the six-bit green of RGB565, whose red and blue
    /// fields are five bits.
    #[must_use]
    pub const fn sample_format(self) -> SampleFormat {
        match self {
            Self::Bit1White | Self::Bit1Black => SampleFormat::Bit1,
            Self::U8 => SampleFormat::Unsigned { bits: 8 },
            Self::U10 | Self::Rgb101010 => SampleFormat::Unsigned { bits: 10 },
            Self::U16 { .. } => SampleFormat::Unsigned { bits: 16 },
            Self::Rgb555 => SampleFormat::Unsigned { bits: 5 },
            Self::Rgb565 => SampleFormat::Unsigned { bits: 6 },
            Self::I16 { .. } => SampleFormat::Signed { bits: 16 },
            Self::I32 { .. } => SampleFormat::Signed { bits: 32 },
            Self::F16 => SampleFormat::Float16,
            Self::F32 { .. } => SampleFormat::Float32,
        }
    }

    /// The largest value an unsigned integer sample of this depth stores,
    /// that of its [`Self::sample_format`]. `None` for signed and
    /// floating-point depths.
    #[must_use]
    pub const fn maximum(self) -> Option<u16> {
        match self.sample_format() {
            SampleFormat::Bit1 => Some(1),
            SampleFormat::Unsigned { bits } => Some(u16::MAX >> (16 - bits)),
            SampleFormat::Signed { .. }
            | SampleFormat::FixedPoint { .. }
            | SampleFormat::Float16
            | SampleFormat::Float32 => None,
        }
    }

    /// The depth samples stored at the Annex-A depth `depth` are coded at,
    /// qualified by the plane-header fields as [`Self::from_header_fields`]
    /// takes them. A stored one-bit sample is the sample itself: one is the
    /// brighter value.
    #[must_use]
    pub const fn from_annex_a(
        depth: AnnexABitDepth,
        shift_bits: u8,
        mantissa_length: u8,
        exponent_bias: i8,
    ) -> Self {
        match depth {
            AnnexABitDepth::Bit1 => Self::Bit1White,
            AnnexABitDepth::U8 => Self::U8,
            AnnexABitDepth::U10 => Self::U10,
            AnnexABitDepth::U16 => Self::U16 { shift_bits },
            AnnexABitDepth::I16 => Self::I16 { shift_bits },
            AnnexABitDepth::F16 => Self::F16,
            AnnexABitDepth::I32 => Self::I32 { shift_bits },
            AnnexABitDepth::F32 => Self::F32 {
                mantissa_length,
                exponent_bias,
            },
            AnnexABitDepth::Rgb555 => Self::Rgb555,
            AnnexABitDepth::Rgb565 => Self::Rgb565,
            AnnexABitDepth::Rgb101010 => Self::Rgb101010,
        }
    }

    /// Convert parsed T.832 header fields to a known output-depth policy.
    #[must_use]
    pub const fn from_header_fields(
        code: u8,
        shift_bits: u8,
        mantissa_length: u8,
        exponent_bias: i8,
    ) -> Option<Self> {
        match code {
            0 => Some(Self::Bit1White),
            1 => Some(Self::U8),
            2 => Some(Self::U16 { shift_bits }),
            3 => Some(Self::I16 { shift_bits }),
            4 => Some(Self::F16),
            6 => Some(Self::I32 { shift_bits }),
            7 => Some(Self::F32 {
                mantissa_length,
                exponent_bias,
            }),
            8 => Some(Self::Rgb555),
            9 => Some(Self::U10),
            10 => Some(Self::Rgb565),
            15 => Some(Self::Bit1Black),
            _ => None,
        }
    }
}

/// Output-depth parameters owned by an integrated or separate alpha plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AlphaFormatRequest {
    pub bit_depth: OutputBitDepth,
    pub scaled: bool,
}

/// Complete device-neutral output formatting policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputFormatRequest {
    pub internal_color: ColorFormat,
    pub output_color: ColorFormat,
    pub bit_depth: OutputBitDepth,
    pub pixel_format: PixelFormat,
    pub scaled: bool,
    pub alpha_format: Option<AlphaFormatRequest>,
    pub red_blue_not_swapped: bool,
    pub premultiply_alpha: bool,
    pub crop: CropWindow,
}
