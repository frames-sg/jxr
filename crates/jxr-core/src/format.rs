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

    /// Whether the last channel is alpha.
    #[must_use]
    pub const fn has_alpha(self) -> bool {
        matches!(
            self,
            Self::LumaAlpha
                | Self::Yuva(_)
                | Self::Rgba
                | Self::Bgra
                | Self::Cmyka
                | Self::NComponentAlpha(_)
        )
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

impl StorageKind {
    /// Bytes of one storage element: a byte of eight one-bit samples, one
    /// sample, or one packed pixel.
    #[must_use]
    pub const fn element_bytes(self) -> usize {
        match self {
            Self::BitPacked | Self::U8 => 1,
            Self::U16 | Self::I16 | Self::F16Bits | Self::PackedU16 => 2,
            Self::I32 | Self::F32 | Self::PackedU32 => 4,
        }
    }
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
    /// The channels of each pixel, in storage order, or `None` for a packed
    /// format: RGB, one word to a pixel, without alpha.
    #[must_use]
    pub const fn channel_layout(self) -> Option<ChannelLayout> {
        match self {
            Self::BitPacked(layout)
            | Self::U8(layout)
            | Self::U16(layout)
            | Self::I16(layout)
            | Self::I32(layout)
            | Self::F16(layout)
            | Self::F32(layout) => Some(layout),
            Self::Rgb555 | Self::Rgb565 | Self::Rgb101010 | Self::Rgbe => None,
        }
    }

    /// The chroma sampling of a planar format, one whose YUV 4:2:0 or
    /// 4:2:2 chroma is stored as planes of its own, smaller than the
    /// image's; `None` for an interleaved format.
    #[must_use]
    pub const fn planar_sampling(self) -> Option<ChromaSampling> {
        match self.channel_layout() {
            Some(
                ChannelLayout::Yuv(sampling @ (ChromaSampling::Cs420 | ChromaSampling::Cs422))
                | ChannelLayout::Yuva(sampling @ (ChromaSampling::Cs420 | ChromaSampling::Cs422)),
            ) => Some(sampling),
            _ => None,
        }
    }

    #[must_use]
    pub const fn channel_count(self) -> u16 {
        match self.channel_layout() {
            Some(layout) => layout.channel_count(),
            None => 3,
        }
    }

    /// Whether the last channel of each pixel is alpha.
    #[must_use]
    pub const fn has_alpha(self) -> bool {
        match self.channel_layout() {
            Some(layout) => layout.has_alpha(),
            None => false,
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

    /// Storage elements one row of `width` pixels takes: whole bytes of
    /// one-bit samples, one element per pixel of a packed format, and one
    /// per channel otherwise; `None` where that does not fit `usize`.
    #[must_use]
    pub fn row_elements(self, width: u32) -> Option<usize> {
        let width = usize::try_from(width).ok()?;
        match self {
            Self::BitPacked(_) => Some(width.div_ceil(8)),
            Self::Rgb555 | Self::Rgb565 | Self::Rgb101010 | Self::Rgbe => Some(width),
            _ => width.checked_mul(usize::from(self.channel_count())),
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
        let samples = width
            .checked_mul(channels)
            .ok_or_else(|| JxrError::arithmetic("output row elements"))?;
        // Every one-bit sample of the row takes a bit, where a packed pixel
        // is one element whatever its channels.
        let elements = match self.storage_kind() {
            StorageKind::BitPacked => samples.div_ceil(8),
            StorageKind::PackedU16 | StorageKind::PackedU32 => width,
            _ => samples,
        };
        elements
            .checked_mul(self.storage_kind().element_bytes())
            .ok_or_else(|| JxrError::arithmetic("output row size"))
    }
}

#[cfg(test)]
mod tests {
    use super::{ChannelLayout, ChromaSampling, PixelFormat, Profile};

    #[test]
    fn planar_row_size_uses_plane_channel_count() {
        let format = PixelFormat::U16(ChannelLayout::Yuv(ChromaSampling::Cs420));
        assert_eq!(format.row_bytes(7).unwrap(), 42);
        assert_eq!(format.row_bytes_for_channels(4, 1).unwrap(), 8);
    }

    #[test]
    fn a_row_takes_its_elements_times_their_bytes() {
        for format in [
            PixelFormat::BitPacked(ChannelLayout::Luma),
            PixelFormat::U8(ChannelLayout::Rgb),
            PixelFormat::U16(ChannelLayout::Rgba),
            PixelFormat::F16(ChannelLayout::Rgb),
            PixelFormat::I32(ChannelLayout::Luma),
            PixelFormat::F32(ChannelLayout::Rgba),
            PixelFormat::Rgb565,
            PixelFormat::Rgbe,
        ] {
            let elements = format.row_elements(13).unwrap();
            let bytes = elements * format.storage_kind().element_bytes();
            assert_eq!(format.row_bytes(13).unwrap(), bytes, "{format:?}");
        }
    }

    #[test]
    fn packed_formats_have_no_channel_layout_and_three_channels() {
        assert_eq!(
            PixelFormat::U8(ChannelLayout::Bgra).channel_layout(),
            Some(ChannelLayout::Bgra)
        );
        for format in [
            PixelFormat::Rgb555,
            PixelFormat::Rgb565,
            PixelFormat::Rgb101010,
            PixelFormat::Rgbe,
        ] {
            assert_eq!(format.channel_layout(), None);
            assert_eq!(format.channel_count(), 3);
            assert!(!format.has_alpha());
        }
    }

    #[test]
    fn profile_codes_invert_their_reading() {
        for idc in [44, 55, 66, 111, 0, 200] {
            assert_eq!(Profile::from_idc(idc).idc(), idc);
        }
        assert_eq!(Profile::from_idc(55), Profile::Baseline);
        assert_eq!(Profile::from_idc(12), Profile::Unknown(12));
    }

    #[test]
    fn annex_a_depths_are_coded_at_the_output_depth_of_their_samples() {
        use crate::{AnnexABitDepth, OutputBitDepth, SampleFormat};
        for (depth, code, sample) in [
            (AnnexABitDepth::Bit1, 0, SampleFormat::Bit1),
            (AnnexABitDepth::U8, 1, SampleFormat::Unsigned { bits: 8 }),
            (AnnexABitDepth::U16, 2, SampleFormat::Unsigned { bits: 16 }),
            (AnnexABitDepth::I16, 3, SampleFormat::Signed { bits: 16 }),
            (AnnexABitDepth::F16, 4, SampleFormat::Float16),
            (AnnexABitDepth::I32, 6, SampleFormat::Signed { bits: 32 }),
            (AnnexABitDepth::F32, 7, SampleFormat::Float32),
            (
                AnnexABitDepth::Rgb555,
                8,
                SampleFormat::Unsigned { bits: 5 },
            ),
            (AnnexABitDepth::U10, 9, SampleFormat::Unsigned { bits: 10 }),
            (
                AnnexABitDepth::Rgb101010,
                9,
                SampleFormat::Unsigned { bits: 10 },
            ),
            (
                AnnexABitDepth::Rgb565,
                10,
                SampleFormat::Unsigned { bits: 6 },
            ),
        ] {
            let coded = OutputBitDepth::from_annex_a(depth, 3, 12, -2);
            assert_eq!(coded.header_fields().0, code, "{depth:?}");
            assert_eq!(coded.sample_format(), sample, "{depth:?}");
        }
        assert_eq!(
            OutputBitDepth::from_annex_a(AnnexABitDepth::I32, 3, 12, -2),
            OutputBitDepth::I32 { shift_bits: 3 }
        );
        assert_eq!(
            OutputBitDepth::from_annex_a(AnnexABitDepth::F32, 3, 12, -2),
            OutputBitDepth::F32 {
                mantissa_length: 12,
                exponent_bias: -2
            }
        );
        // The black-is-one depth is a one-bit sample as well.
        assert_eq!(
            OutputBitDepth::Bit1Black.sample_format(),
            SampleFormat::Bit1
        );
    }

    #[test]
    fn planar_formats_are_the_subsampled_yuv_layouts() {
        use super::ChromaSampling::{Cs420, Cs422, Cs444};
        assert_eq!(
            PixelFormat::U8(ChannelLayout::Yuv(Cs420)).planar_sampling(),
            Some(Cs420)
        );
        assert_eq!(
            PixelFormat::U16(ChannelLayout::Yuva(Cs422)).planar_sampling(),
            Some(Cs422)
        );
        for interleaved in [
            PixelFormat::U8(ChannelLayout::Yuv(Cs444)),
            PixelFormat::U8(ChannelLayout::Rgb),
            PixelFormat::Rgb565,
        ] {
            assert_eq!(interleaved.planar_sampling(), None, "{interleaved:?}");
        }
    }

    #[test]
    fn every_sample_storage_reports_one_kind_and_size() {
        use crate::{DecodedSamples, DecodedSamplesMut, SourceSamples, StorageKind};
        let mut owned = DecodedSamples::Rgb565(alloc::vec![1, 2, 3]);
        assert_eq!(owned.storage_kind(), StorageKind::PackedU16);
        assert_eq!((owned.len(), owned.byte_len()), (3, 6));
        let source = SourceSamples::from(&owned);
        assert_eq!(source.storage_kind(), StorageKind::PackedU16);
        assert_eq!((source.len(), source.byte_len()), (3, 6));
        let DecodedSamples::Rgb565(values) = &mut owned else {
            unreachable!()
        };
        let destination = DecodedSamplesMut::Rgb565(values);
        assert_eq!(destination.storage_kind(), StorageKind::PackedU16);
        assert_eq!((destination.len(), destination.byte_len()), (3, 6));
        assert!(DecodedSamples::BitPacked(alloc::vec::Vec::new()).is_empty());
    }

    #[test]
    fn output_depths_bound_their_unsigned_samples() {
        use crate::OutputBitDepth;
        assert_eq!(OutputBitDepth::Bit1Black.maximum(), Some(1));
        assert_eq!(OutputBitDepth::U8.maximum(), Some(255));
        assert_eq!(OutputBitDepth::U10.maximum(), Some(1023));
        assert_eq!(
            OutputBitDepth::U16 { shift_bits: 3 }.maximum(),
            Some(u16::MAX)
        );
        assert_eq!(OutputBitDepth::Rgb565.maximum(), Some(63));
        assert_eq!(OutputBitDepth::I16 { shift_bits: 0 }.maximum(), None);
        assert_eq!(OutputBitDepth::F16.maximum(), None);
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

impl Profile {
    /// The profile a T.832 `PROFILE_IDC` declares.
    #[must_use]
    pub const fn from_idc(idc: u8) -> Self {
        match idc {
            44 => Self::SubBaseline,
            55 => Self::Baseline,
            66 => Self::Main,
            111 => Self::Advanced,
            idc => Self::Unknown(idc),
        }
    }

    /// The `PROFILE_IDC` that declares this profile: the inverse of
    /// [`Self::from_idc`].
    #[must_use]
    pub const fn idc(self) -> u8 {
        match self {
            Self::SubBaseline => 44,
            Self::Baseline => 55,
            Self::Main => 66,
            Self::Advanced => 111,
            Self::Unknown(idc) => idc,
        }
    }
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
