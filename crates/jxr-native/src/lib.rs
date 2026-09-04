//! Safe Rust JPEG XR parsing, CPU decoding, and Annex-A writing.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod accelerator;
mod annex_a;
mod bit_reader;
mod codestream;
mod coefficient;
mod cpu;
mod decode;
pub mod entropy;
mod error;
mod header;
mod metadata;
pub mod output_format;
mod pixel_format;
mod prepare;
pub mod reconstruct;
mod tail;
pub mod tile_decode;

pub use accelerator::{
    AcceleratorCoefficients, PreparedAlphaCoefficients, prepare_accelerator_coefficients,
};
pub use annex_a::{AnnexAImage, AnnexAMetadata, AnnexAWriteOptions, parse_annex_a, write_annex_a};
pub use codestream::{ParsedCodestream, parse_codestream};
pub use coefficient::{coefficient_count, decode_coefficients, decode_coefficients_into};
pub use cpu::CpuCapabilities;
pub use decode::{
    CpuDecodeIntoOutput, CpuDecodeWorkspace, decode_cpu, decode_cpu_into_with_workspace,
    decode_cpu_u8_into_with_workspace, decode_cpu_with_workspace, prepare_output_format,
};
pub use error::NativeError;
pub use header::{
    CodestreamHeader, HeaderFlags, ImagePlaneHeader, ParsedHeaders, QuantizerSet,
    parse_codestream_headers,
};
pub use metadata::image_info;
pub use pixel_format::classify_annex_a_pixel_format;
pub use prepare::prepare_plan;
pub use tail::{CodestreamDirectory, ProfileLevel};
