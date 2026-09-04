//! Owned native batch preparation and CPU decoding.

mod contracts;
mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
mod error;
#[cfg(feature = "metal")]
mod metal;
mod prepare;

pub use contracts::{
    BatchDecodeOptions, BatchDecoder, BatchGroupInfo, BatchLayout, CpuBatchDecodeResult,
    CpuBatchDestination, CpuBatchDiagnostics, CpuBatchGroup, CpuBatchIntoResult, CpuBatchSamples,
    EncodedImage, PreparedBatch, PreparedBatchGroup, PreparedImage,
};
pub use cpu::CpuBatchDecoder;
#[cfg(feature = "cuda")]
pub use cuda::{
    CudaBatchDecodeResult, CudaBatchDecoder, CudaBatchError, CudaBatchGroup, CudaBatchGroupError,
    SubmittedCudaPreparedBatch,
};
pub use error::{BatchErrorStage, BatchInfrastructureError, IndexedBatchError};
#[cfg(feature = "metal")]
pub use metal::{
    MetalBatchDecodeResult, MetalBatchDecoder, MetalBatchError, MetalBatchGroup,
    MetalBatchGroupError, MetalDenseBatchDecodeResult, MetalDenseBatchGroup,
    MetalPreparedGroupIntoCompletion, SubmittedMetalDenseBatch, SubmittedMetalPreparedBatch,
    SubmittedMetalPreparedGroupInto,
};
pub use prepare::{prepare_batch, prepare_batch_from_images};

const MAX_BATCH_WORKERS: usize = 64;
