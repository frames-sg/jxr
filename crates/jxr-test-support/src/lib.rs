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

pub use benchmark::{
    TimingSummary, checksum_bytes, checksum_cpu_batch_image, checksum_samples, summarize_timings,
};
pub use compare::{DifferentialResult, RustRawOutput, compare_file, decode_rust_file};
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
    discover_t834_cases, run_t834_cpu_case,
};
pub use error::OracleError;
pub use format::{OracleFormat, oracle_format};
pub use process::{OracleRawOutput, T835Oracle, T835ProfileLimit};
