//! JPEG XR decode and Annex-A container facade.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod batch;
mod container;
mod decoder;
mod error;
mod prepared;
mod reconstruction;
mod view;

pub use batch::{
    BatchDecodeOptions, BatchDecoder, BatchErrorStage, BatchGroupInfo, BatchInfrastructureError,
    BatchLayout, CpuBatchDecodeResult, CpuBatchDecoder, CpuBatchDestination, CpuBatchDiagnostics,
    CpuBatchGroup, CpuBatchIntoResult, CpuBatchSamples, EncodedImage, IndexedBatchError,
    PreparedBatch, PreparedBatchGroup, PreparedImage, prepare_batch, prepare_batch_from_images,
};
#[cfg(feature = "cuda")]
pub use batch::{
    CudaBatchDecodeResult, CudaBatchDecoder, CudaBatchError, CudaBatchGroup, CudaBatchGroupError,
    SubmittedCudaPreparedBatch,
};
#[cfg(feature = "metal")]
pub use batch::{
    MetalBatchDecodeResult, MetalBatchDecoder, MetalBatchError, MetalBatchGroup,
    MetalBatchGroupError, MetalDenseBatchDecodeResult, MetalDenseBatchGroup,
    MetalPreparedGroupIntoCompletion, SubmittedMetalDenseBatch, SubmittedMetalPreparedBatch,
    SubmittedMetalPreparedGroupInto,
};
pub use container::{AnnexAWriteOptions, write_annex_a};
pub use decoder::{DecodeIntoResult, DecodeIntoSample, JxrDecoder};
pub use prepared::PreparedJxr;
pub use reconstruction::{PreparedAlphaReconstruction, PreparedReconstruction};
pub use view::JxrView;

pub(crate) use error::map_native_error;
pub use jxr_core::*;

#[cfg(feature = "cuda")]
pub use jxr_cuda as cuda;
#[cfg(feature = "metal")]
pub use jxr_metal as metal;
