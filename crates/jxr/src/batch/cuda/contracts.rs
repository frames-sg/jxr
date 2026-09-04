// SPDX-License-Identifier: MIT OR Apache-2.0

use jxr_core::{DecodeReport, ImageInfo, Rect};

use super::super::{BatchGroupInfo, BatchInfrastructureError, IndexedBatchError};

/// Infrastructure failure from a persistent CUDA batch session.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CudaBatchError {
    /// Shared batch preparation or allocation failed.
    #[error(transparent)]
    Infrastructure(#[from] BatchInfrastructureError),
    /// The retained CUDA session rejected an operation.
    #[error(transparent)]
    Cuda(#[from] jxr_cuda::CudaError),
}

/// Failure of one homogeneous CUDA group after input-local preparation.
#[derive(Debug, thiserror::Error)]
#[error("CUDA JPEG XR batch group {source_indices:?} failed: {source}")]
pub struct CudaBatchGroupError {
    source_indices: Vec<usize>,
    #[source]
    pub(super) source: jxr_cuda::CudaError,
}

impl CudaBatchGroupError {
    pub(super) fn new(source_indices: Vec<usize>, source: jxr_cuda::CudaError) -> Self {
        Self {
            source_indices,
            source,
        }
    }

    /// Original input positions affected by this group failure.
    #[must_use]
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }

    /// CUDA execution failure.
    #[must_use]
    pub const fn source(&self) -> &jxr_cuda::CudaError {
        &self.source
    }
}

/// One successful homogeneous CUDA-resident output group.
pub struct CudaBatchGroup {
    pub(super) info: BatchGroupInfo,
    pub(super) source_indices: Vec<usize>,
    pub(super) image_infos: Vec<ImageInfo>,
    pub(super) decoded_regions: Vec<Rect>,
    pub(super) reports: Vec<DecodeReport>,
    pub(super) images: Vec<jxr_cuda::ResidentCudaImage>,
}

impl core::fmt::Debug for CudaBatchGroup {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("CudaBatchGroup")
            .field("info", &self.info)
            .field("source_indices", &self.source_indices)
            .field("image_count", &self.images.len())
            .finish_non_exhaustive()
    }
}

impl CudaBatchGroup {
    /// Shared native output contract.
    #[must_use]
    pub const fn info(&self) -> &BatchGroupInfo {
        &self.info
    }

    /// Original input positions in resident image order.
    #[must_use]
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }

    /// Parsed source metadata in resident image order.
    #[must_use]
    pub fn image_infos(&self) -> &[ImageInfo] {
        &self.image_infos
    }

    /// Actual decoded regions in resident image order.
    #[must_use]
    pub fn decoded_regions(&self) -> &[Rect] {
        &self.decoded_regions
    }

    /// CUDA route and stage reports in resident image order.
    #[must_use]
    pub fn reports(&self) -> &[DecodeReport] {
        &self.reports
    }

    /// Completed immutable CUDA images.
    #[must_use]
    pub fn images(&self) -> &[jxr_cuda::ResidentCudaImage] {
        &self.images
    }
}

/// Successful resident groups plus indexed and group-level failures.
#[derive(Debug)]
pub struct CudaBatchDecodeResult {
    pub(super) groups: Vec<CudaBatchGroup>,
    pub(super) errors: Vec<IndexedBatchError>,
    pub(super) group_errors: Vec<CudaBatchGroupError>,
}

impl CudaBatchDecodeResult {
    /// Successful homogeneous resident groups.
    #[must_use]
    pub fn groups(&self) -> &[CudaBatchGroup] {
        &self.groups
    }

    /// Input-local preparation failures in original order.
    #[must_use]
    pub fn errors(&self) -> &[IndexedBatchError] {
        &self.errors
    }

    /// Homogeneous groups that failed during CUDA allocation or execution.
    #[must_use]
    pub fn group_errors(&self) -> &[CudaBatchGroupError] {
        &self.group_errors
    }
}

pub(super) struct SubmittedCudaGroup {
    pub(super) info: BatchGroupInfo,
    pub(super) source_indices: Vec<usize>,
    pub(super) image_infos: Vec<ImageInfo>,
    pub(super) decoded_regions: Vec<Rect>,
    pub(super) submission: jxr_cuda::CudaBatchSubmission,
}

/// Nonblocking high-level CUDA batch retaining every pending group.
pub struct SubmittedCudaPreparedBatch {
    pub(super) groups: Vec<SubmittedCudaGroup>,
    pub(super) errors: Vec<IndexedBatchError>,
    pub(super) group_errors: Vec<CudaBatchGroupError>,
}

impl core::fmt::Debug for SubmittedCudaPreparedBatch {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SubmittedCudaPreparedBatch")
            .field("pending_groups", &self.groups.len())
            .field("errors", &self.errors)
            .field("group_errors", &self.group_errors)
            .finish_non_exhaustive()
    }
}

impl SubmittedCudaPreparedBatch {
    /// Number of successfully submitted homogeneous groups.
    #[must_use]
    pub fn pending_group_count(&self) -> usize {
        self.groups.len()
    }

    /// Wait for all submitted groups and preserve group-local failures.
    pub fn wait(mut self) -> CudaBatchDecodeResult {
        let mut groups = Vec::with_capacity(self.groups.len());
        for pending in self.groups {
            match pending.submission.wait() {
                Ok(images) => {
                    let reports = images.iter().map(|image| image.report().clone()).collect();
                    groups.push(CudaBatchGroup {
                        info: pending.info,
                        source_indices: pending.source_indices,
                        image_infos: pending.image_infos,
                        decoded_regions: pending.decoded_regions,
                        reports,
                        images,
                    });
                }
                Err(source) => self
                    .group_errors
                    .push(CudaBatchGroupError::new(pending.source_indices, source)),
            }
        }
        CudaBatchDecodeResult {
            groups,
            errors: self.errors,
            group_errors: self.group_errors,
        }
    }
}
