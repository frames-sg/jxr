// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unpublished differential-validation support for JPEG XR implementations.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod benchmark;
mod compare;
mod conformance;
mod error;
mod format;
mod process;
mod quality;

pub use benchmark::{
    BenchmarkFile, TimingSummary, check_checksum, checksum_bytes, checksum_cpu_batch_image,
    checksum_samples, milliseconds, summarize_timings, synthetic_samples,
};
pub use compare::{
    DifferentialResult, RustRawOutput, compare_file, decode_rust_file, planar_bytes,
    reference_alpha_bytes, reference_bytes, sample_bytes,
};
#[cfg(feature = "cuda")]
pub use compare::{compare_file_cuda, decode_cuda_file};
#[cfg(feature = "metal")]
pub use compare::{compare_file_metal, decode_metal_file};
#[cfg(feature = "cuda")]
pub use conformance::run_t834_cuda_case;
#[cfg(feature = "metal")]
pub use conformance::run_t834_metal_case;
pub use conformance::{
    T834Case, T834CaseExpectation, T834CaseOutcome, T834CaseResult, T834Summary,
    discover_t834_cases, run_t834_cpu_case, t834_suite, workspace_root,
};
pub use error::{OracleError, exit_on_error, exit_with_failures};
pub use format::{OracleFormat, decimates_chroma, oracle_format, subsamples_chroma};
pub use process::{OracleRawOutput, T835Oracle, T835ProfileLimit, TempFile};
pub use quality::{QualityClass, QualityMeasurement, QualityPicture, QualitySource, split_mix64};
