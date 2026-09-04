// SPDX-License-Identifier: MIT OR Apache-2.0

/// Failure while invoking or interpreting an external JPEG XR oracle.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum OracleError {
    /// The Annex-A format is not mapped by the differential harness.
    #[error("unsupported oracle pixel format: {reason}")]
    UnsupportedFormat {
        /// Stable explanation of the unsupported representation.
        reason: &'static str,
    },
    /// An external executable or file operation failed.
    #[error("{operation} failed for {path:?}: {source}")]
    Io {
        /// Operation being attempted.
        operation: &'static str,
        /// Affected path.
        path: std::path::PathBuf,
        /// Operating-system failure.
        #[source]
        source: std::io::Error,
    },
    /// The reference decoder rejected the input or failed internally.
    #[error("T.835 decoder failed with status {status:?}: {stderr}")]
    ProcessFailed {
        /// Process exit code, or `None` when terminated by a signal.
        status: Option<i32>,
        /// Captured reference-software diagnostics.
        stderr: String,
    },
    /// The reference decoder returned success without its promised raw output.
    #[error("T.835 decoder did not create {path:?}")]
    MissingOutput {
        /// Expected raw output path.
        path: std::path::PathBuf,
    },
    /// Rust parsing or decoding rejected the comparison input.
    #[error("Rust JPEG XR decode failed: {message}")]
    RustDecode {
        /// Stable display form of the codec error.
        message: String,
    },
    /// Rust output and the reference bytes differ.
    #[error(
        "T.835 mismatch at byte {offset}: oracle={oracle:?}, rust={rust:?} (oracle {oracle_len} bytes, rust {rust_len} bytes)"
    )]
    Mismatch {
        /// First differing byte, or the common length for a length-only mismatch.
        offset: usize,
        /// Reference byte at `offset`, when present.
        oracle: Option<u8>,
        /// Rust byte at `offset`, when present.
        rust: Option<u8>,
        /// Reference output byte count.
        oracle_len: usize,
        /// Rust output byte count.
        rust_len: usize,
    },
}
