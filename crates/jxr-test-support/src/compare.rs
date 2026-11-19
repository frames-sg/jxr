// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::Path;

use jxr::{BackendRequest, DecodeRequest, DecodedSamples, JxrView, PixelFormat};

use crate::{OracleError, OracleFormat, T835Oracle, oracle_format};

/// Successful byte-for-byte differential comparison metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DifferentialResult {
    /// Compared output representation.
    pub format: PixelFormat,
    /// Decoded image width.
    pub width: u32,
    /// Decoded image height.
    pub height: u32,
    /// Number of bytes compared.
    pub byte_len: usize,
}

/// Raw bytes produced by the portable Rust decoder for oracle comparison.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RustRawOutput {
    /// Compared output representation.
    pub format: PixelFormat,
    /// Decoded image width.
    pub width: u32,
    /// Decoded image height.
    pub height: u32,
    /// Native-endian bytes matching the reference program's host
    /// representation, plane by plane where it writes planes.
    pub bytes: Vec<u8>,
}

/// Decode one Annex-A image with scalar Rust and T.835, then compare raw bytes.
pub fn compare_file(oracle: &T835Oracle, input: &Path) -> Result<DifferentialResult, OracleError> {
    let rust = decode_rust_file(input)?;
    compare_output(oracle, input, &rust)
}

/// Decode one Annex-A image with the portable Rust route into comparable bytes.
pub fn decode_rust_file(input: &Path) -> Result<RustRawOutput, OracleError> {
    decode_file(input, DecodeBackend::Cpu)
}

#[derive(Clone, Copy)]
enum DecodeBackend<'a> {
    Cpu,
    #[cfg(feature = "metal")]
    Metal(&'a jxr::metal::MetalDecoderSession),
    #[cfg(feature = "cuda")]
    Cuda(&'a jxr::cuda::CudaDecoderSession),
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    #[expect(
        dead_code,
        reason = "retains one decode implementation across optional Metal builds"
    )]
    Marker(core::marker::PhantomData<&'a ()>),
}

fn decode_file(input: &Path, backend: DecodeBackend<'_>) -> Result<RustRawOutput, OracleError> {
    let source = std::fs::read(input).map_err(|error| OracleError::Io {
        operation: "read JPEG XR input",
        path: input.to_owned(),
        source: error,
    })?;
    let view = JxrView::parse(&source).map_err(|error| rust_error(&error))?;
    let format = oracle_format(view.info())?;
    let mut request = DecodeRequest::new(format.pixel_format);
    request.alpha = format.alpha;
    let decoded = match backend {
        DecodeBackend::Cpu => {
            request.backend = BackendRequest::Cpu;
            view.decoder().decode(&request)
        }
        #[cfg(feature = "metal")]
        DecodeBackend::Metal(session) => {
            request.backend = BackendRequest::Metal;
            view.decoder().with_metal_session(session).decode(&request)
        }
        #[cfg(feature = "cuda")]
        DecodeBackend::Cuda(session) => {
            request.backend = BackendRequest::Cuda;
            view.decoder().with_cuda_session(session).decode(&request)
        }
        #[cfg(not(any(feature = "metal", feature = "cuda")))]
        DecodeBackend::Marker(_) => unreachable!("marker backend is never constructed"),
    }
    .map_err(|error| rust_error(&error))?;
    Ok(RustRawOutput {
        format: format.pixel_format,
        width: decoded.info.width,
        height: decoded.info.height,
        bytes: reference_bytes(&sample_bytes(&decoded.samples), format),
    })
}

/// Rearrange the Rust decoder's tightly packed samples of an image into the
/// raw layout the reference program writes an image of `format` in.
///
/// The decoder returns interleaved samples, or for YUV 4:2:0 and 4:2:2 the
/// planes themselves, which then only gain the planes the reference appends.
#[must_use]
pub fn reference_bytes(decoded: &[u8], format: OracleFormat) -> Vec<u8> {
    if !format.planar {
        return decoded.to_vec();
    }
    let mut bytes = if format.subsampling().is_some() {
        decoded.to_vec()
    } else {
        planar_bytes(decoded, format.pixel_format)
    };
    // One storage element of the fill: its low byte for eight-bit samples.
    let wide = format.trailing_fill.to_ne_bytes();
    let narrow = [format.trailing_fill.to_le_bytes()[0]];
    let fill: &[u8] = if format.pixel_format.storage_kind().element_bytes() == 1 {
        &narrow
    } else {
        &wide
    };
    for plane in 1..=usize::from(format.trailing_planes) {
        for _ in 0..format.plane_samples(plane) {
            bytes.extend_from_slice(fill);
        }
    }
    bytes
}

/// The alpha samples of raw bytes in the reference layout of `format`: the
/// last sample of every pixel, or the plane after the primary channels'
/// planes.
#[must_use]
pub fn reference_alpha_bytes(bytes: &[u8], format: OracleFormat) -> Vec<u8> {
    let element = format.pixel_format.storage_kind().element_bytes();
    let channels = usize::from(format.pixel_format.channel_count()).max(1);
    if format.planar {
        let start: usize = (0..channels - 1)
            .map(|channel| format.plane_samples(channel))
            .sum();
        let end = start + format.plane_samples(channels - 1);
        return bytes[start * element..end * element].to_vec();
    }
    bytes
        .chunks_exact(element * channels)
        .flat_map(|pixel| &pixel[element * (channels - 1)..])
        .copied()
        .collect()
}

/// Rearrange tightly packed interleaved samples of `format` into whole
/// planes, one channel after another: the order the reference program writes
/// the formats [`crate::OracleFormat::planar`] marks.
#[must_use]
pub fn planar_bytes(interleaved: &[u8], format: PixelFormat) -> Vec<u8> {
    let element = format.storage_kind().element_bytes();
    let channels = usize::from(format.channel_count());
    let mut planar = Vec::with_capacity(interleaved.len());
    for channel in 0..channels {
        for pixel in interleaved.chunks_exact(element * channels) {
            planar.extend_from_slice(&pixel[channel * element..(channel + 1) * element]);
        }
    }
    planar
}

/// Decode one Annex-A image through strict Metal and T.835, then compare raw bytes.
#[cfg(feature = "metal")]
pub fn compare_file_metal(
    oracle: &T835Oracle,
    input: &Path,
    session: &jxr::metal::MetalDecoderSession,
) -> Result<DifferentialResult, OracleError> {
    let rust = decode_metal_file(input, session)?;
    compare_output(oracle, input, &rust)
}

/// Decode one Annex-A image through a strict Metal session into comparable bytes.
#[cfg(feature = "metal")]
pub fn decode_metal_file(
    input: &Path,
    session: &jxr::metal::MetalDecoderSession,
) -> Result<RustRawOutput, OracleError> {
    decode_file(input, DecodeBackend::Metal(session))
}

/// Decode one Annex-A image through strict CUDA and T.835, then compare raw bytes.
#[cfg(feature = "cuda")]
pub fn compare_file_cuda(
    oracle: &T835Oracle,
    input: &Path,
    session: &jxr::cuda::CudaDecoderSession,
) -> Result<DifferentialResult, OracleError> {
    let rust = decode_cuda_file(input, session)?;
    compare_output(oracle, input, &rust)
}

/// Decode one Annex-A image through a strict CUDA session into comparable bytes.
#[cfg(feature = "cuda")]
pub fn decode_cuda_file(
    input: &Path,
    session: &jxr::cuda::CudaDecoderSession,
) -> Result<RustRawOutput, OracleError> {
    decode_file(input, DecodeBackend::Cuda(session))
}

fn compare_output(
    oracle: &T835Oracle,
    input: &Path,
    rust: &RustRawOutput,
) -> Result<DifferentialResult, OracleError> {
    let reference = oracle.decode_raw(input)?;
    compare_bytes(&reference.bytes, &rust.bytes)?;
    Ok(DifferentialResult {
        format: rust.format,
        width: rust.width,
        height: rust.height,
        byte_len: rust.bytes.len(),
    })
}

/// Native-endian bytes of decoded samples in their own storage order.
#[must_use]
pub fn sample_bytes(samples: &DecodedSamples) -> Vec<u8> {
    match samples {
        DecodedSamples::BitPacked(values) | DecodedSamples::U8(values) => values.clone(),
        DecodedSamples::U16(values)
        | DecodedSamples::F16(values)
        | DecodedSamples::Rgb555(values)
        | DecodedSamples::Rgb565(values) => flatten(values, u16::to_ne_bytes),
        DecodedSamples::I16(values) => flatten(values, i16::to_ne_bytes),
        DecodedSamples::I32(values) => flatten(values, i32::to_ne_bytes),
        DecodedSamples::F32(values) => flatten(values, f32::to_ne_bytes),
        DecodedSamples::Rgb101010(values) | DecodedSamples::Rgbe(values) => {
            flatten(values, u32::to_ne_bytes)
        }
    }
}

fn flatten<T, const N: usize>(values: &[T], bytes: impl Fn(T) -> [u8; N]) -> Vec<u8>
where
    T: Copy,
{
    values.iter().copied().flat_map(bytes).collect::<Vec<_>>()
}

fn compare_bytes(oracle: &[u8], rust: &[u8]) -> Result<(), OracleError> {
    let common = oracle.len().min(rust.len());
    let mismatch = oracle[..common]
        .iter()
        .zip(&rust[..common])
        .position(|(oracle, rust)| oracle != rust)
        .unwrap_or(common);
    if mismatch == common && oracle.len() == rust.len() {
        return Ok(());
    }
    Err(OracleError::Mismatch {
        offset: mismatch,
        oracle: oracle.get(mismatch).copied(),
        rust: rust.get(mismatch).copied(),
        oracle_len: oracle.len(),
        rust_len: rust.len(),
    })
}

pub(crate) fn rust_error(error: &jxr::JxrError) -> OracleError {
    OracleError::RustDecode {
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use jxr::{ChannelLayout, PixelFormat};

    use super::{compare_bytes, planar_bytes, reference_alpha_bytes, reference_bytes};
    use crate::{OracleError, OracleFormat};

    #[test]
    fn reference_bytes_follow_the_declared_layout() {
        let interleaved: Vec<u8> = (1..=10).collect();
        let mut format = OracleFormat {
            pixel_format: PixelFormat::U8(ChannelLayout::Cmyka),
            alpha: jxr::AlphaHandling::Preserve,
            planar: false,
            trailing_planes: 0,
            trailing_fill: 0,
            width: 2,
            height: 1,
        };
        assert_eq!(reference_bytes(&interleaved, format), interleaved);
        format.planar = true;
        assert_eq!(
            reference_bytes(&interleaved, format),
            [1, 6, 2, 7, 3, 8, 4, 9, 5, 10]
        );
        // A separately decoded alpha image leaves three planes unfilled.
        format.trailing_planes = 3;
        assert_eq!(
            reference_bytes(&interleaved, format),
            [1, 6, 2, 7, 3, 8, 4, 9, 5, 10, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn alpha_is_the_last_sample_or_the_plane_after_the_primary_ones() {
        let mut format = OracleFormat {
            pixel_format: PixelFormat::U16(ChannelLayout::Rgba),
            alpha: jxr::AlphaHandling::Preserve,
            planar: false,
            trailing_planes: 0,
            trailing_fill: 0,
            width: 2,
            height: 1,
        };
        let interleaved: Vec<u8> = (0..16).collect();
        assert_eq!(reference_alpha_bytes(&interleaved, format), [6, 7, 14, 15]);
        format.pixel_format = PixelFormat::U8(ChannelLayout::Cmyka);
        format.planar = true;
        format.trailing_planes = 3;
        let planar = [1, 6, 2, 7, 3, 8, 4, 9, 5, 10, 0, 0, 0, 0, 0, 0];
        assert_eq!(reference_alpha_bytes(&planar, format), [5, 10]);
    }

    #[test]
    fn subsampled_yuv_keeps_its_planes_and_gains_neutral_chroma_after_alpha() {
        use jxr::ChromaSampling;

        // A 4 x 2 image: eight Y, two U, two V, and eight alpha samples.
        let decoded: Vec<u8> = (1..=20).collect();
        let mut format = OracleFormat {
            pixel_format: PixelFormat::U8(ChannelLayout::Yuva(ChromaSampling::Cs420)),
            alpha: jxr::AlphaHandling::Preserve,
            planar: true,
            trailing_planes: 0,
            trailing_fill: 0x80,
            width: 4,
            height: 2,
        };
        assert_eq!(reference_bytes(&decoded, format), decoded);
        assert_eq!(
            reference_alpha_bytes(&decoded, format),
            (13..=20).collect::<Vec<u8>>()
        );
        // Separate alpha is followed by a plane for each chroma channel.
        format.trailing_planes = 2;
        let mut expected = decoded.clone();
        expected.extend([0x80; 4]);
        assert_eq!(reference_bytes(&decoded, format), expected);
        assert_eq!(
            reference_alpha_bytes(&expected, format),
            (13..=20).collect::<Vec<u8>>()
        );
        // 4:2:2 chroma keeps every row: four samples a plane.
        format.pixel_format = PixelFormat::U8(ChannelLayout::Yuva(ChromaSampling::Cs422));
        let decoded: Vec<u8> = (1..=24).collect();
        assert_eq!(reference_bytes(&decoded, format).len(), 24 + 8);
        assert_eq!(
            reference_alpha_bytes(&decoded, format),
            (17..=24).collect::<Vec<u8>>()
        );
    }

    #[test]
    fn full_resolution_yuv_is_gathered_into_planes_with_sixteen_bit_fill() {
        use jxr::ChromaSampling;

        // Two pixels of Y, U, V, alpha in sixteen-bit elements.
        let interleaved: Vec<u8> = (0..16).collect();
        let format = OracleFormat {
            pixel_format: PixelFormat::U16(ChannelLayout::Yuva(ChromaSampling::Cs444)),
            alpha: jxr::AlphaHandling::Preserve,
            planar: true,
            trailing_planes: 2,
            trailing_fill: 0x200,
            width: 2,
            height: 1,
        };
        let bytes = reference_bytes(&interleaved, format);
        assert_eq!(
            bytes[..16],
            [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]
        );
        let fill = 0x200_u16.to_ne_bytes();
        assert_eq!(bytes[16..], [fill, fill, fill, fill].concat());
        assert_eq!(reference_alpha_bytes(&bytes, format), [6, 7, 14, 15]);
    }

    #[test]
    fn planar_bytes_gather_each_channel_in_pixel_order() {
        assert_eq!(
            planar_bytes(
                &[1, 2, 3, 4, 11, 12, 13, 14],
                PixelFormat::U8(ChannelLayout::Cmyk)
            ),
            [1, 11, 2, 12, 3, 13, 4, 14]
        );
        // Sixteen-bit samples move as whole elements.
        let interleaved: Vec<u8> = (0..16).collect();
        assert_eq!(
            planar_bytes(&interleaved, PixelFormat::U16(ChannelLayout::Cmyk)),
            [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]
        );
    }

    #[test]
    fn reports_first_value_mismatch() {
        let error = compare_bytes(&[1, 2, 3], &[1, 9, 3]).unwrap_err();
        assert!(matches!(
            error,
            OracleError::Mismatch {
                offset: 1,
                oracle: Some(2),
                rust: Some(9),
                ..
            }
        ));
    }

    #[test]
    fn reports_length_mismatch_at_common_end() {
        let error = compare_bytes(&[1], &[1, 2]).unwrap_err();
        assert!(matches!(
            error,
            OracleError::Mismatch {
                offset: 1,
                oracle: None,
                rust: Some(2),
                ..
            }
        ));
    }
}
