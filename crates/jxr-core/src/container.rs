//! Typed Annex-A container pixel-format metadata.

use crate::ChromaSampling;

/// Classified Annex-A `PIXEL_FORMAT` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AnnexAPixelFormat {
    /// A format defined by T.832 Table A.6.
    Known(AnnexAPixelFormatDescriptor),
    /// An unrecognized identifier, retained byte-for-byte.
    Unknown([u8; 16]),
}

/// Semantic family of a known Annex-A pixel format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AnnexAPixelFamily {
    Luma,
    Rgb,
    Rgbe,
    Cmyk { direct: bool },
    Yuv(ChromaSampling),
    NComponent,
}

/// Numeric interpretation declared by the Annex-A format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AnnexANumericKind {
    Unsigned,
    FixedPoint,
    Float,
}

/// Per-component output representation required by the Annex-A format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AnnexABitDepth {
    Bit1,
    U8,
    U10,
    U16,
    I16,
    F16,
    I32,
    F32,
    Rgb555,
    Rgb565,
    Rgb101010,
}

/// Reference byte/channel ordering associated with a known format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AnnexAChannelOrder {
    Luma,
    Rgb,
    /// RGB followed by one zero-valued storage padding component.
    Rgbx,
    Bgr,
    /// BGR followed by one zero-valued storage padding component.
    Bgrx,
    Rgba,
    Bgra,
    Cmyk,
    Yuv,
    Rgbe,
    Components,
    PackedBgr,
}

/// Table A.6 properties needed for inspection and codestream validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AnnexAPixelFormatDescriptor {
    pub family: AnnexAPixelFamily,
    /// Total decoded channels, including alpha when present.
    pub channels: u8,
    pub alpha: bool,
    pub premultiplied_alpha: bool,
    pub bit_depth: AnnexABitDepth,
    pub numeric: AnnexANumericKind,
    pub order: AnnexAChannelOrder,
}
