//! Device-neutral reconstruction output policy shared by CPU and accelerators.

use crate::{ColorFormat, PixelFormat};

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
