use jxr_core::{
    DecodeReport, DecodedSamples, DecodedSamplesMut, ImageInfo, JxrError, JxrErrorKind,
    PixelFormat, PlaneDescriptor, Rect,
};

/// Metadata returned after decoding into caller-owned typed storage.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeIntoResult {
    /// Parsed image information.
    pub info: ImageInfo,
    /// Display-space region written to the destination.
    pub decoded_region: Rect,
    /// Pixel representation written to the destination.
    pub format: PixelFormat,
    /// Byte strides and extents relative to the destination's first element.
    pub planes: Vec<PlaneDescriptor>,
    /// Selected route and ownership of every decode stage.
    pub report: DecodeReport,
}

mod sealed {
    pub trait Sealed {}

    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for u32 {}
    impl Sealed for f32 {}
}

/// Host element types accepted by [`super::JxrDecoder::decode_into`].
pub trait DecodeIntoSample: sealed::Sealed + Copy {
    #[doc(hidden)]
    fn copy_decoded(samples: &DecodedSamples, destination: &mut [Self]) -> Result<(), JxrError>;

    #[doc(hidden)]
    fn direct_destination(
        format: PixelFormat,
        destination: &mut [Self],
        required: usize,
    ) -> Result<DecodedSamplesMut<'_>, JxrError>;
}

fn checked_prefix<T>(destination: &mut [T], required: usize) -> Result<&mut [T], JxrError> {
    if destination.len() < required {
        return Err(JxrError::new(
            JxrErrorKind::BufferTooSmall {
                required,
                available: destination.len(),
            },
            "decode directly into typed samples",
        ));
    }
    Ok(&mut destination[..required])
}

fn copy_samples<T: Copy>(source: &[T], destination: &mut [T]) -> Result<(), JxrError> {
    if destination.len() < source.len() {
        return Err(JxrError::new(
            JxrErrorKind::BufferTooSmall {
                required: source.len(),
                available: destination.len(),
            },
            "copy decoded samples",
        ));
    }
    destination[..source.len()].copy_from_slice(source);
    Ok(())
}

impl DecodeIntoSample for u8 {
    fn copy_decoded(samples: &DecodedSamples, destination: &mut [Self]) -> Result<(), JxrError> {
        match samples {
            DecodedSamples::BitPacked(values) | DecodedSamples::U8(values) => {
                copy_samples(values, destination)
            }
            _ => Err(sample_type_error()),
        }
    }

    fn direct_destination(
        format: PixelFormat,
        destination: &mut [Self],
        required: usize,
    ) -> Result<DecodedSamplesMut<'_>, JxrError> {
        let destination = checked_prefix(destination, required)?;
        match format {
            PixelFormat::BitPacked(_) => Ok(DecodedSamplesMut::BitPacked(destination)),
            PixelFormat::U8(_) => Ok(DecodedSamplesMut::U8(destination)),
            _ => Err(sample_type_error()),
        }
    }
}

impl DecodeIntoSample for u16 {
    fn copy_decoded(samples: &DecodedSamples, destination: &mut [Self]) -> Result<(), JxrError> {
        match samples {
            DecodedSamples::U16(values)
            | DecodedSamples::F16(values)
            | DecodedSamples::Rgb555(values)
            | DecodedSamples::Rgb565(values) => copy_samples(values, destination),
            _ => Err(sample_type_error()),
        }
    }

    fn direct_destination(
        format: PixelFormat,
        destination: &mut [Self],
        required: usize,
    ) -> Result<DecodedSamplesMut<'_>, JxrError> {
        let destination = checked_prefix(destination, required)?;
        match format {
            PixelFormat::U16(_) => Ok(DecodedSamplesMut::U16(destination)),
            PixelFormat::F16(_) => Ok(DecodedSamplesMut::F16(destination)),
            PixelFormat::Rgb555 => Ok(DecodedSamplesMut::Rgb555(destination)),
            PixelFormat::Rgb565 => Ok(DecodedSamplesMut::Rgb565(destination)),
            _ => Err(sample_type_error()),
        }
    }
}

macro_rules! impl_decode_into_sample {
    ($type:ty, $variant:ident, $format:pat) => {
        impl DecodeIntoSample for $type {
            fn copy_decoded(
                samples: &DecodedSamples,
                destination: &mut [Self],
            ) -> Result<(), JxrError> {
                match samples {
                    DecodedSamples::$variant(values) => copy_samples(values, destination),
                    _ => Err(sample_type_error()),
                }
            }

            fn direct_destination(
                format: PixelFormat,
                destination: &mut [Self],
                required: usize,
            ) -> Result<DecodedSamplesMut<'_>, JxrError> {
                let destination = checked_prefix(destination, required)?;
                match format {
                    $format => Ok(DecodedSamplesMut::$variant(destination)),
                    _ => Err(sample_type_error()),
                }
            }
        }
    };
}

impl_decode_into_sample!(i16, I16, PixelFormat::I16(_));
impl_decode_into_sample!(i32, I32, PixelFormat::I32(_));
impl_decode_into_sample!(f32, F32, PixelFormat::F32(_));

impl DecodeIntoSample for u32 {
    fn copy_decoded(samples: &DecodedSamples, destination: &mut [Self]) -> Result<(), JxrError> {
        match samples {
            DecodedSamples::Rgb101010(values) | DecodedSamples::Rgbe(values) => {
                copy_samples(values, destination)
            }
            _ => Err(sample_type_error()),
        }
    }

    fn direct_destination(
        format: PixelFormat,
        destination: &mut [Self],
        required: usize,
    ) -> Result<DecodedSamplesMut<'_>, JxrError> {
        let destination = checked_prefix(destination, required)?;
        match format {
            PixelFormat::Rgb101010 => Ok(DecodedSamplesMut::Rgb101010(destination)),
            PixelFormat::Rgbe => Ok(DecodedSamplesMut::Rgbe(destination)),
            _ => Err(sample_type_error()),
        }
    }
}

const fn sample_type_error() -> JxrError {
    JxrError::new(JxrErrorKind::InvalidRequest, "typed decode destination")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_destination_reports_the_exact_required_length() {
        let error = u16::copy_decoded(&DecodedSamples::U16(vec![1, 2]), &mut [0]).unwrap_err();
        assert_eq!(
            error.kind,
            JxrErrorKind::BufferTooSmall {
                required: 2,
                available: 1,
            }
        );
    }
}
