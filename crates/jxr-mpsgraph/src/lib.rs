// SPDX-License-Identifier: MIT OR Apache-2.0

//! Direct `MPSGraph` integration for Metal-resident JPEG XR batches.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]

mod contract;
mod error;
mod options;
mod prepared;
mod reference;

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod platform;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod program;
#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
mod unsupported;

pub use contract::{MpsGraphElementType, MpsGraphTensorSpec};
pub use error::Error;
pub use options::{MpsGraphDecodeInput, MpsGraphDecodeOptions};
pub use prepared::{
    IndexedGroupError, IndexedPreparationError, MpsGraphPreparedBatch, MpsGraphPreparedGroup,
};
pub use reference::{RGB8_REFERENCE_CHANNEL_WEIGHTS, rgb8_nhwc_reference_cpu};

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub use platform::{MpsGraphBatchDecode, MpsGraphBatchDecoder, MpsGraphInputGroup};
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub use program::{MpsGraphProgram, MpsGraphRunOutput, SubmittedMpsGraphRun};
#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
pub use unsupported::{
    MpsGraphBatchDecode, MpsGraphBatchDecoder, MpsGraphInputGroup, MpsGraphProgram,
    MpsGraphRunOutput, SubmittedMpsGraphRun,
};
