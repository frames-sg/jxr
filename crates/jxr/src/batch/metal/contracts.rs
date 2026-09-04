use jxr_core::{DecodeReport, ImageInfo, Rect};

use super::super::{BatchGroupInfo, BatchInfrastructureError, IndexedBatchError};

/// Infrastructure failure from a persistent Metal batch session.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MetalBatchError {
    /// Shared batch preparation or allocation failed.
    #[error(transparent)]
    Infrastructure(#[from] BatchInfrastructureError),
    /// The retained Metal session could not be created.
    #[error(transparent)]
    Metal(#[from] jxr_metal::MetalError),
    /// One or more inputs could not be converted into Metal plans.
    #[error("{count} JPEG XR inputs failed Metal plan preparation")]
    IndexedPreparation {
        /// Number of failed inputs.
        count: usize,
        /// Indexed input-local failures.
        errors: Vec<IndexedBatchError>,
    },
}

impl MetalBatchError {
    /// Indexed failures when caller-owned submission required an all-success group.
    #[must_use]
    pub fn indexed_errors(&self) -> Option<&[IndexedBatchError]> {
        match self {
            Self::IndexedPreparation { errors, .. } => Some(errors),
            Self::Infrastructure(_) | Self::Metal(_) => None,
        }
    }
}

/// One successful homogeneous Metal-resident output group.
pub struct MetalBatchGroup {
    info: BatchGroupInfo,
    source_indices: Vec<usize>,
    image_infos: Vec<ImageInfo>,
    decoded_regions: Vec<Rect>,
    reports: Vec<DecodeReport>,
    images: Vec<jxr_metal::ResidentMetalImage>,
}

impl core::fmt::Debug for MetalBatchGroup {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MetalBatchGroup")
            .field("info", &self.info)
            .field("source_indices", &self.source_indices)
            .field("image_count", &self.images.len())
            .finish_non_exhaustive()
    }
}

impl MetalBatchGroup {
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

    /// Metal route and stage reports in resident image order.
    #[must_use]
    pub fn reports(&self) -> &[DecodeReport] {
        &self.reports
    }

    /// Completed immutable Metal images.
    #[must_use]
    pub fn images(&self) -> &[jxr_metal::ResidentMetalImage] {
        &self.images
    }
}

/// Failure of one homogeneous Metal group after input-local preparation.
#[derive(Debug, thiserror::Error)]
#[error("Metal JPEG XR batch group {source_indices:?} failed: {source}")]
pub struct MetalBatchGroupError {
    source_indices: Vec<usize>,
    #[source]
    pub(super) source: jxr_metal::MetalError,
}

impl MetalBatchGroupError {
    pub(super) fn new(source_indices: Vec<usize>, source: jxr_metal::MetalError) -> Self {
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

    /// Metal execution failure.
    #[must_use]
    pub const fn source(&self) -> &jxr_metal::MetalError {
        &self.source
    }
}

/// Successful resident groups plus indexed and group-level failures.
#[derive(Debug)]
pub struct MetalBatchDecodeResult {
    groups: Vec<MetalBatchGroup>,
    errors: Vec<IndexedBatchError>,
    group_errors: Vec<MetalBatchGroupError>,
}

pub(super) struct SubmittedMetalGroup {
    pub(super) info: BatchGroupInfo,
    pub(super) source_indices: Vec<usize>,
    pub(super) image_infos: Vec<ImageInfo>,
    pub(super) decoded_regions: Vec<Rect>,
    pub(super) submission: jxr_metal::MetalBatchSubmission,
}

/// Nonblocking high-level Metal batch retaining every pending group.
pub struct SubmittedMetalPreparedBatch {
    pub(super) groups: Vec<SubmittedMetalGroup>,
    pub(super) errors: Vec<IndexedBatchError>,
    pub(super) group_errors: Vec<MetalBatchGroupError>,
}

impl core::fmt::Debug for SubmittedMetalPreparedBatch {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SubmittedMetalPreparedBatch")
            .field("pending_groups", &self.groups.len())
            .field("errors", &self.errors)
            .field("group_errors", &self.group_errors)
            .finish_non_exhaustive()
    }
}

impl SubmittedMetalPreparedBatch {
    /// Number of successfully submitted homogeneous groups.
    #[must_use]
    pub fn pending_group_count(&self) -> usize {
        self.groups.len()
    }

    /// Wait for all submitted groups and preserve group-local failures.
    pub fn wait(mut self) -> MetalBatchDecodeResult {
        let mut groups = Vec::with_capacity(self.groups.len());
        for pending in self.groups {
            match pending.submission.wait() {
                Ok(images) => {
                    let reports = images.iter().map(|image| image.report().clone()).collect();
                    groups.push(MetalBatchGroup {
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
                    .push(MetalBatchGroupError::new(pending.source_indices, source)),
            }
        }
        MetalBatchDecodeResult {
            groups,
            errors: self.errors,
            group_errors: self.group_errors,
        }
    }
}

/// One completed homogeneous group stored in one private Metal allocation.
pub struct MetalDenseBatchGroup {
    info: BatchGroupInfo,
    source_indices: Vec<usize>,
    image_infos: Vec<ImageInfo>,
    decoded_regions: Vec<Rect>,
    batch: jxr_metal::MetalResidentBatch,
}

impl core::fmt::Debug for MetalDenseBatchGroup {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MetalDenseBatchGroup")
            .field("info", &self.info)
            .field("source_indices", &self.source_indices)
            .field("layout", self.batch.layout())
            .finish_non_exhaustive()
    }
}

impl MetalDenseBatchGroup {
    /// Shared native output contract.
    #[must_use]
    pub const fn info(&self) -> &BatchGroupInfo {
        &self.info
    }

    /// Original input positions in dense allocation order.
    #[must_use]
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }

    /// Source metadata in dense allocation order.
    #[must_use]
    pub fn image_infos(&self) -> &[ImageInfo] {
        &self.image_infos
    }

    /// Decoded regions in dense allocation order.
    #[must_use]
    pub fn decoded_regions(&self) -> &[Rect] {
        &self.decoded_regions
    }

    /// Completed single-allocation Metal batch.
    #[must_use]
    pub const fn batch(&self) -> &jxr_metal::MetalResidentBatch {
        &self.batch
    }
}

/// Completed dense Metal groups plus indexed and group-level failures.
#[derive(Debug)]
pub struct MetalDenseBatchDecodeResult {
    groups: Vec<MetalDenseBatchGroup>,
    errors: Vec<IndexedBatchError>,
    group_errors: Vec<MetalBatchGroupError>,
}

impl MetalDenseBatchDecodeResult {
    /// Completed single-allocation homogeneous groups.
    #[must_use]
    pub fn groups(&self) -> &[MetalDenseBatchGroup] {
        &self.groups
    }

    /// Input-local preparation failures.
    #[must_use]
    pub fn errors(&self) -> &[IndexedBatchError] {
        &self.errors
    }

    /// Group allocation, submission, or completion failures.
    #[must_use]
    pub fn group_errors(&self) -> &[MetalBatchGroupError] {
        &self.group_errors
    }
}

pub(super) struct SubmittedDenseGroup {
    pub(super) info: BatchGroupInfo,
    pub(super) source_indices: Vec<usize>,
    pub(super) image_infos: Vec<ImageInfo>,
    pub(super) decoded_regions: Vec<Rect>,
    pub(super) submission: jxr_metal::MetalResidentBatchSubmission,
}

/// Nonblocking dense Metal batch retaining each single-allocation group.
pub struct SubmittedMetalDenseBatch {
    pub(super) groups: Vec<SubmittedDenseGroup>,
    pub(super) errors: Vec<IndexedBatchError>,
    pub(super) group_errors: Vec<MetalBatchGroupError>,
}

impl core::fmt::Debug for SubmittedMetalDenseBatch {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SubmittedMetalDenseBatch")
            .field("pending_groups", &self.groups.len())
            .field("errors", &self.errors)
            .field("group_errors", &self.group_errors)
            .finish_non_exhaustive()
    }
}

impl SubmittedMetalDenseBatch {
    /// Number of successfully submitted dense groups.
    #[must_use]
    pub fn pending_group_count(&self) -> usize {
        self.groups.len()
    }

    /// Wait for all dense groups.
    pub fn wait(mut self) -> MetalDenseBatchDecodeResult {
        let mut groups = Vec::with_capacity(self.groups.len());
        for pending in self.groups {
            match pending.submission.wait() {
                Ok(batch) => groups.push(MetalDenseBatchGroup {
                    info: pending.info,
                    source_indices: pending.source_indices,
                    image_infos: pending.image_infos,
                    decoded_regions: pending.decoded_regions,
                    batch,
                }),
                Err(source) => self
                    .group_errors
                    .push(MetalBatchGroupError::new(pending.source_indices, source)),
            }
        }
        MetalDenseBatchDecodeResult {
            groups,
            errors: self.errors,
            group_errors: self.group_errors,
        }
    }
}

/// Pending exact-queue write into a caller-owned dense Metal destination.
pub struct SubmittedMetalPreparedGroupInto {
    pub(super) source_indices: Vec<usize>,
    pub(super) image_infos: Vec<ImageInfo>,
    pub(super) decoded_regions: Vec<Rect>,
    pub(super) submission: jxr_metal::MetalBatchDestinationSubmission,
}

impl core::fmt::Debug for SubmittedMetalPreparedGroupInto {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SubmittedMetalPreparedGroupInto")
            .field("source_indices", &self.source_indices)
            .finish_non_exhaustive()
    }
}

/// Completion metadata for one caller-owned high-level group.
#[derive(Debug)]
pub struct MetalPreparedGroupIntoCompletion {
    source_indices: Vec<usize>,
    image_infos: Vec<ImageInfo>,
    decoded_regions: Vec<Rect>,
    destination: jxr_metal::MetalBatchDestinationCompletion,
}

impl SubmittedMetalPreparedGroupInto {
    /// Wait for exact-queue completion and recover the retained destination.
    pub fn wait(self) -> Result<MetalPreparedGroupIntoCompletion, jxr_metal::MetalError> {
        Ok(MetalPreparedGroupIntoCompletion {
            source_indices: self.source_indices,
            image_infos: self.image_infos,
            decoded_regions: self.decoded_regions,
            destination: self.submission.wait()?,
        })
    }
}

impl MetalPreparedGroupIntoCompletion {
    /// Original input positions in destination image order.
    #[must_use]
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }

    /// Source metadata in destination image order.
    #[must_use]
    pub fn image_infos(&self) -> &[ImageInfo] {
        &self.image_infos
    }

    /// Decoded regions in destination image order.
    #[must_use]
    pub fn decoded_regions(&self) -> &[Rect] {
        &self.decoded_regions
    }

    /// Low-level completion retaining the caller-owned allocation.
    #[must_use]
    pub const fn destination(&self) -> &jxr_metal::MetalBatchDestinationCompletion {
        &self.destination
    }

    /// Consume into metadata and the retained destination completion.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        Vec<usize>,
        Vec<ImageInfo>,
        Vec<Rect>,
        jxr_metal::MetalBatchDestinationCompletion,
    ) {
        (
            self.source_indices,
            self.image_infos,
            self.decoded_regions,
            self.destination,
        )
    }
}

impl MetalBatchDecodeResult {
    /// Successful homogeneous resident groups.
    #[must_use]
    pub fn groups(&self) -> &[MetalBatchGroup] {
        &self.groups
    }

    /// Input-local preparation failures in original order.
    #[must_use]
    pub fn errors(&self) -> &[IndexedBatchError] {
        &self.errors
    }

    /// Homogeneous groups that failed during Metal allocation or execution.
    #[must_use]
    pub fn group_errors(&self) -> &[MetalBatchGroupError] {
        &self.group_errors
    }
}
