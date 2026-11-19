//! Typed decoded host storage.

use alloc::vec::Vec;

use crate::{DecodeReport, ImageInfo, JxrError, JxrErrorKind, PixelFormat, Rect};

/// Implements `storage_kind`, `matches_format`, `len`, `is_empty` and
/// `byte_len` for typed sample storage, an enum whose variants are those of
/// [`DecodedSamples`],
/// each holding a slice or vector of its storage elements, so that the
/// borrowed, owned, source and batch forms of the storage share one
/// definition of its kind and size.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_sample_storage {
    ($samples:ty) => {
        impl $samples {
            /// Native storage kind of these samples.
            #[must_use]
            pub const fn storage_kind(&self) -> $crate::StorageKind {
                match self {
                    Self::BitPacked(_) => $crate::StorageKind::BitPacked,
                    Self::U8(_) => $crate::StorageKind::U8,
                    Self::U16(_) => $crate::StorageKind::U16,
                    Self::I16(_) => $crate::StorageKind::I16,
                    Self::I32(_) => $crate::StorageKind::I32,
                    Self::F16(_) => $crate::StorageKind::F16Bits,
                    Self::F32(_) => $crate::StorageKind::F32,
                    Self::Rgb555(_) | Self::Rgb565(_) => $crate::StorageKind::PackedU16,
                    Self::Rgb101010(_) | Self::Rgbe(_) => $crate::StorageKind::PackedU32,
                }
            }

            /// Whether this storage variant exactly represents `format`.
            #[must_use]
            pub const fn matches_format(&self, format: $crate::PixelFormat) -> bool {
                matches!(
                    (self, format),
                    (Self::BitPacked(_), $crate::PixelFormat::BitPacked(_))
                        | (Self::U8(_), $crate::PixelFormat::U8(_))
                        | (Self::U16(_), $crate::PixelFormat::U16(_))
                        | (Self::I16(_), $crate::PixelFormat::I16(_))
                        | (Self::I32(_), $crate::PixelFormat::I32(_))
                        | (Self::F16(_), $crate::PixelFormat::F16(_))
                        | (Self::F32(_), $crate::PixelFormat::F32(_))
                        | (Self::Rgb555(_), $crate::PixelFormat::Rgb555)
                        | (Self::Rgb565(_), $crate::PixelFormat::Rgb565)
                        | (Self::Rgb101010(_), $crate::PixelFormat::Rgb101010)
                        | (Self::Rgbe(_), $crate::PixelFormat::Rgbe)
                )
            }

            /// Number of native storage elements: bytes of one-bit
            /// samples, samples, or packed pixels.
            #[must_use]
            pub const fn len(&self) -> usize {
                match self {
                    Self::BitPacked(values) | Self::U8(values) => values.len(),
                    Self::U16(values)
                    | Self::F16(values)
                    | Self::Rgb555(values)
                    | Self::Rgb565(values) => values.len(),
                    Self::I16(values) => values.len(),
                    Self::I32(values) => values.len(),
                    Self::F32(values) => values.len(),
                    Self::Rgb101010(values) | Self::Rgbe(values) => values.len(),
                }
            }

            /// Whether there are no storage elements.
            #[must_use]
            pub const fn is_empty(&self) -> bool {
                self.len() == 0
            }

            /// Bytes the storage elements take.
            #[must_use]
            pub const fn byte_len(&self) -> usize {
                self.len()
                    .saturating_mul(self.storage_kind().element_bytes())
            }
        }
    };
}

/// Mutable caller-owned storage for every supported output representation.
pub enum DecodedSamplesMut<'a> {
    /// Byte-packed one-bit samples.
    BitPacked(&'a mut [u8]),
    /// Unsigned eight-bit samples.
    U8(&'a mut [u8]),
    /// Unsigned sixteen-bit samples.
    U16(&'a mut [u16]),
    /// Signed sixteen-bit samples.
    I16(&'a mut [i16]),
    /// Signed thirty-two-bit samples.
    I32(&'a mut [i32]),
    /// IEEE binary16 bit patterns.
    F16(&'a mut [u16]),
    /// IEEE binary32 samples.
    F32(&'a mut [f32]),
    /// Packed RGB 5:5:5 words.
    Rgb555(&'a mut [u16]),
    /// Packed RGB 5:6:5 words.
    Rgb565(&'a mut [u16]),
    /// Packed RGB 10:10:10 words.
    Rgb101010(&'a mut [u32]),
    /// Packed shared-exponent RGB words.
    Rgbe(&'a mut [u32]),
}

impl core::fmt::Debug for DecodedSamplesMut<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("DecodedSamplesMut")
            .field("storage_kind", &self.storage_kind())
            .field("elements", &self.len())
            .finish_non_exhaustive()
    }
}

impl DecodedSamplesMut<'_> {
    /// Borrow this destination again without changing its storage variant.
    ///
    /// This permits a layered writer to pass the storage to one stage and then
    /// continue operating on the same caller-owned allocation after that borrow
    /// ends.
    pub fn reborrow(&mut self) -> DecodedSamplesMut<'_> {
        match self {
            Self::BitPacked(values) => DecodedSamplesMut::BitPacked(values),
            Self::U8(values) => DecodedSamplesMut::U8(values),
            Self::U16(values) => DecodedSamplesMut::U16(values),
            Self::I16(values) => DecodedSamplesMut::I16(values),
            Self::I32(values) => DecodedSamplesMut::I32(values),
            Self::F16(values) => DecodedSamplesMut::F16(values),
            Self::F32(values) => DecodedSamplesMut::F32(values),
            Self::Rgb555(values) => DecodedSamplesMut::Rgb555(values),
            Self::Rgb565(values) => DecodedSamplesMut::Rgb565(values),
            Self::Rgb101010(values) => DecodedSamplesMut::Rgb101010(values),
            Self::Rgbe(values) => DecodedSamplesMut::Rgbe(values),
        }
    }
}

/// Typed storage for every supported output representation.
#[derive(Debug, Clone, PartialEq)]
pub enum DecodedSamples {
    BitPacked(Vec<u8>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    F16(Vec<u16>),
    F32(Vec<f32>),
    Rgb555(Vec<u16>),
    Rgb565(Vec<u16>),
    Rgb101010(Vec<u32>),
    Rgbe(Vec<u32>),
}

impl_sample_storage!(DecodedSamples);
impl_sample_storage!(DecodedSamplesMut<'_>);

impl DecodedSamples {
    /// Samples held: eight to each byte of one-bit samples.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        let per_element = if matches!(self, Self::BitPacked(_)) {
            8
        } else {
            1
        };
        self.len().saturating_mul(per_element)
    }
}

/// Byte-based description of one output plane within [`DecodedSamples`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneDescriptor {
    pub byte_offset: usize,
    pub row_stride_bytes: usize,
    pub width: u32,
    pub height: u32,
    /// Number of channels interleaved within this plane.
    pub channels: u16,
}

impl PlaneDescriptor {
    pub fn validate(self, format: PixelFormat, available: usize) -> Result<(), JxrError> {
        let row_bytes = format.row_bytes_for_channels(self.width, self.channels)?;
        if self.row_stride_bytes < row_bytes {
            return Err(JxrError::new(
                JxrErrorKind::InvalidRequest,
                "decoded plane stride",
            ));
        }
        let body = if self.width == 0 || self.height == 0 {
            0
        } else {
            let preceding_rows = usize::try_from(self.height - 1)
                .map_err(|_| JxrError::arithmetic("decoded plane height"))?;
            self.row_stride_bytes
                .checked_mul(preceding_rows)
                .and_then(|bytes| bytes.checked_add(row_bytes))
                .ok_or_else(|| JxrError::arithmetic("decoded plane extent"))?
        };
        let end = self
            .byte_offset
            .checked_add(body)
            .ok_or_else(|| JxrError::arithmetic("decoded plane range"))?;
        if end > available {
            return Err(JxrError::new(
                JxrErrorKind::BufferTooSmall {
                    required: end,
                    available,
                },
                "decoded plane range",
            ));
        }
        Ok(())
    }
}

/// A complete host decode result.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedImage {
    pub info: ImageInfo,
    pub decoded_region: Rect,
    pub format: PixelFormat,
    pub planes: Vec<PlaneDescriptor>,
    pub samples: DecodedSamples,
    pub report: DecodeReport,
}

impl DecodedImage {
    pub fn validate_layout(&self) -> Result<(), JxrError> {
        if self.samples.storage_kind() != self.format.storage_kind() {
            return Err(JxrError::new(
                JxrErrorKind::InternalInvariant,
                "decoded storage type",
            ));
        }
        for plane in &self.planes {
            plane.validate(self.format, self.samples.byte_len())?;
        }
        Ok(())
    }
}
