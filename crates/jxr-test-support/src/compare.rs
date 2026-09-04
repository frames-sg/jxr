// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::Path;

use jxr::{BackendRequest, DecodeRequest, DecodedSamples, JxrView, PixelFormat};

use crate::{OracleError, T835Oracle, oracle_format};

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
    /// Native-endian bytes matching the reference program's host representation.
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
        bytes: sample_bytes(&decoded.samples),
    })
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

fn sample_bytes(samples: &DecodedSamples) -> Vec<u8> {
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

fn rust_error(error: &jxr::JxrError) -> OracleError {
    OracleError::RustDecode {
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::compare_bytes;
    use crate::OracleError;

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
