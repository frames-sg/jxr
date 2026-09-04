//! JPEG XR source and decoded-output format descriptors.

use crate::{JxrError, JxrErrorKind};

/// Chroma sampling relative to the luma plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChromaSampling {
    Cs420,
    Cs422,
    Cs444,
}

/// Color organization represented by a JPEG XR image plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorFormat {
    Luma,
    Yuv(ChromaSampling),
    Rgb,
    Cmyk,
    CmykDirect,
    YuvK,
    Rgbe,
    NComponent(u16),
}

impl ColorFormat {
    /// Return the number of components, if the declaration is valid.
    #[must_use]
    pub const fn component_count(self) -> Option<u16> {
        match self {
            Self::Luma => Some(1),
            Self::Yuv(_) | Self::Rgb | Self::Rgbe => Some(3),
            Self::Cmyk | Self::CmykDirect | Self::YuvK => Some(4),
            Self::NComponent(0) => None,
            Self::NComponent(count) => Some(count),
        }
    }
}

/// Source sample representation before output packing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleFormat {
    Bit1,
    Unsigned { bits: u8 },
    Signed { bits: u8 },
    FixedPoint { bits: u8, fractional_bits: u8 },
    Float16,
    Float32,
}

/// Logical ordering of output channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelLayout {
    Luma,
    LumaAlpha,
    /// Interleaved Y, U, and V channels at the declared sampling geometry.
    Yuv(ChromaSampling),
    /// Interleaved Y, U, V, and alpha channels.
    Yuva(ChromaSampling),
    Rgb,
    /// RGB followed by one zero-valued storage padding component.
    Rgbx,
    Rgba,
    Bgr,
    /// BGR followed by one zero-valued storage padding component.
    Bgrx,
    Bgra,
    Cmyk,
    Cmyka,
    NComponent(u16),
    NComponentAlpha(u16),
}

impl ChannelLayout {
    #[must_use]
    pub const fn channel_count(self) -> u16 {
        match self {
            Self::Luma => 1,
            Self::LumaAlpha => 2,
            Self::Yuv(_) | Self::Rgb | Self::Bgr => 3,
            Self::Yuva(_) | Self::Rgbx | Self::Rgba | Self::Bgrx | Self::Bgra | Self::Cmyk => 4,
            Self::Cmyka => 5,
            Self::NComponent(count) => count,
            Self::NComponentAlpha(count) => count.saturating_add(1),
        }
    }
}

/// Type of host storage backing decoded samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageKind {
    BitPacked,
    U8,
    U16,
    I16,
    I32,
    F16Bits,
    F32,
    PackedU16,
    PackedU32,
}

/// Requested decoded pixel representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    BitPacked(ChannelLayout),
    U8(ChannelLayout),
    U16(ChannelLayout),
    I16(ChannelLayout),
    I32(ChannelLayout),
    F16(ChannelLayout),
    F32(ChannelLayout),
    Rgb555,
    Rgb565,
    Rgb101010,
    Rgbe,
}

impl PixelFormat {
    #[must_use]
    pub const fn channel_count(self) -> u16 {
        match self {
            Self::BitPacked(layout)
            | Self::U8(layout)
            | Self::U16(layout)
            | Self::I16(layout)
            | Self::I32(layout)
            | Self::F16(layout)
            | Self::F32(layout) => layout.channel_count(),
            Self::Rgb555 | Self::Rgb565 | Self::Rgb101010 | Self::Rgbe => 3,
        }
    }

    #[must_use]
    pub const fn storage_kind(self) -> StorageKind {
        match self {
            Self::BitPacked(_) => StorageKind::BitPacked,
            Self::U8(_) => StorageKind::U8,
            Self::U16(_) => StorageKind::U16,
            Self::I16(_) => StorageKind::I16,
            Self::I32(_) => StorageKind::I32,
            Self::F16(_) => StorageKind::F16Bits,
            Self::F32(_) => StorageKind::F32,
            Self::Rgb555 | Self::Rgb565 => StorageKind::PackedU16,
            Self::Rgb101010 | Self::Rgbe => StorageKind::PackedU32,
        }
    }

    /// Return the tightly packed byte count for one row.
    pub fn row_bytes(self, width: u32) -> Result<usize, JxrError> {
        self.row_bytes_for_channels(width, self.channel_count())
    }

    /// Return the tightly packed byte count for one plane row.
    ///
    /// Planar formats pass the number of interleaved channels stored in that
    /// plane rather than the channel count of the complete pixel format.
    pub fn row_bytes_for_channels(self, width: u32, channels: u16) -> Result<usize, JxrError> {
        let channels = usize::from(channels);
        if channels == 0 {
            return Err(JxrError::new(
                JxrErrorKind::InvalidRequest,
                "output channel layout",
            ));
        }
        let width = usize::try_from(width).map_err(|_| JxrError::arithmetic("output row width"))?;
        let elements = width
            .checked_mul(channels)
            .ok_or_else(|| JxrError::arithmetic("output row elements"))?;
        match self.storage_kind() {
            StorageKind::BitPacked => elements
                .checked_add(7)
                .map(|bits| bits / 8)
                .ok_or_else(|| JxrError::arithmetic("bit-packed row size")),
            StorageKind::U8 => Ok(elements),
            StorageKind::U16 | StorageKind::I16 | StorageKind::F16Bits => elements
                .checked_mul(2)
                .ok_or_else(|| JxrError::arithmetic("16-bit row size")),
            StorageKind::I32 | StorageKind::F32 => elements
                .checked_mul(4)
                .ok_or_else(|| JxrError::arithmetic("32-bit row size")),
            StorageKind::PackedU16 => width
                .checked_mul(2)
                .ok_or_else(|| JxrError::arithmetic("packed 16-bit row size")),
            StorageKind::PackedU32 => width
                .checked_mul(4)
                .ok_or_else(|| JxrError::arithmetic("packed 32-bit row size")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ChannelLayout, ChromaSampling, PixelFormat};

    #[test]
    fn planar_row_size_uses_plane_channel_count() {
        let format = PixelFormat::U16(ChannelLayout::Yuv(ChromaSampling::Cs420));
        assert_eq!(format.row_bytes(7).unwrap(), 42);
        assert_eq!(format.row_bytes_for_channels(4, 1).unwrap(), 8);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlphaMode {
    None,
    Integrated,
    Separate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Profile {
    SubBaseline,
    Baseline,
    Main,
    Advanced,
    Unknown(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Level(pub u8);

/// Display orientation metadata. Decoders report but do not apply it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Orientation {
    #[default]
    Identity,
    MirrorHorizontal,
    Rotate180,
    MirrorVertical,
    Transpose,
    Rotate90,
    Transverse,
    Rotate270,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BitstreamMode {
    Spatial,
    Frequency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BandPresence {
    DcOnly,
    NoHighPass,
    NoFlexbits,
    All,
}

impl BandPresence {
    #[must_use]
    pub const fn has_low_pass(self) -> bool {
        !matches!(self, Self::DcOnly)
    }

    #[must_use]
    pub const fn has_high_pass(self) -> bool {
        matches!(self, Self::NoFlexbits | Self::All)
    }

    #[must_use]
    pub const fn has_flexbits(self) -> bool {
        matches!(self, Self::All)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum OverlapMode {
    None = 0,
    One = 1,
    Two = 2,
}
