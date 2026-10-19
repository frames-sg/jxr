use rayon::prelude::*;

use super::{
    BatchDecodeOptions, BatchDecoder, BatchErrorStage, BatchInfrastructureError, CpuBatchDecoder,
    EncodedImage, IndexedBatchError, PreparedBatch, PreparedBatchGroup, PreparedImage,
    prepare::try_vec,
};

mod contracts;

pub use contracts::{
    MetalBatchDecodeResult, MetalBatchError, MetalBatchGroup, MetalBatchGroupError,
    MetalDenseBatchDecodeResult, MetalDenseBatchGroup, MetalPreparedGroupIntoCompletion,
    SubmittedMetalDenseBatch, SubmittedMetalPreparedBatch, SubmittedMetalPreparedGroupInto,
};
use contracts::{SubmittedDenseGroup, SubmittedMetalGroup};

type CountedCandidate<'a> = (usize, usize, &'a PreparedImage);
type PlannedItem<'a> = (usize, &'a PreparedImage, jxr_metal::MetalDecodePlan);

type StagedPlan<'a> = (
    usize,
    &'a PreparedImage,
    Result<jxr_metal::MetalDecodePlan, jxr_core::JxrError>,
);

/// Images prepared and submitted per pipeline step of a resident group.
///
/// Each chunk is submitted as soon as its CPU entropy finishes, so the GPU
/// reconstructs it while the worker pool prepares the next chunk; only the
/// last chunk's reconstruction remains on the critical path. Groups no larger
/// than one chunk keep the single prepare-then-submit order.
const PIPELINE_CHUNK_IMAGES: usize = 32;

/// Persistent high-level Metal decoder for owned and prepared batches.
pub struct MetalBatchDecoder {
    session: jxr_metal::MetalDecoderSession,
    preparer: CpuBatchDecoder,
}

impl core::fmt::Debug for MetalBatchDecoder {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MetalBatchDecoder")
            .field("session", &self.session)
            .field("options", &self.options())
            .finish_non_exhaustive()
    }
}

impl MetalBatchDecoder {
    /// Create a persistent decoder on the default Metal device.
    pub fn system_default(options: BatchDecodeOptions) -> Result<Self, MetalBatchError> {
        Self::with_session(jxr_metal::MetalDecoderSession::system_default()?, options)
    }

    /// Wrap an existing Metal session and retain a preparation worker pool.
    pub fn with_session(
        session: jxr_metal::MetalDecoderSession,
        options: BatchDecodeOptions,
    ) -> Result<Self, MetalBatchError> {
        if options.layout != super::BatchLayout::Native {
            return Err(BatchInfrastructureError::UnsupportedBatchLayout {
                backend: "Metal",
                layout: options.layout,
            }
            .into());
        }
        Ok(Self {
            session,
            preparer: CpuBatchDecoder::new(options)?,
        })
    }

    /// Retained shared batch policy.
    #[must_use]
    pub const fn options(&self) -> BatchDecodeOptions {
        self.preparer.options()
    }

    /// Low-level Metal session retained across calls.
    #[must_use]
    pub const fn session(&self) -> &jxr_metal::MetalDecoderSession {
        &self.session
    }

    /// Parse and group owned inputs using the retained worker pool.
    pub fn prepare(&self, inputs: Vec<EncodedImage>) -> Result<PreparedBatch, MetalBatchError> {
        Ok(self.preparer.prepare(inputs)?)
    }

    /// Regroup prepared images without reparsing them.
    pub fn prepare_prepared_images(
        &self,
        images: Vec<PreparedImage>,
    ) -> Result<PreparedBatch, MetalBatchError> {
        Ok(self.preparer.prepare_prepared_images(images)?)
    }

    /// Prepare and decode owned inputs to immutable Metal images.
    pub fn decode(
        &self,
        inputs: Vec<EncodedImage>,
    ) -> Result<MetalBatchDecodeResult, MetalBatchError> {
        let prepared = self.prepare(inputs)?;
        self.decode_prepared(&prepared)
    }

    /// Decode a reusable shared prepared batch to immutable Metal images.
    pub fn decode_prepared(
        &self,
        prepared: &PreparedBatch,
    ) -> Result<MetalBatchDecodeResult, MetalBatchError> {
        Ok(self.submit_prepared(prepared)?.wait())
    }

    /// Prepare and submit a shared batch without waiting for GPU completion.
    pub fn submit_prepared(
        &self,
        prepared: &PreparedBatch,
    ) -> Result<SubmittedMetalPreparedBatch, MetalBatchError> {
        let input_count = prepared.input_count();
        if input_count > self.options().max_inputs {
            return Err(BatchInfrastructureError::TooManyInputs {
                requested: input_count,
                maximum: self.options().max_inputs,
            }
            .into());
        }
        let mut groups = try_vec(prepared.groups().len(), "JPEG XR Metal pending groups")?;
        let mut errors = try_vec(input_count, "JPEG XR Metal batch indexed errors")?;
        let mut group_errors =
            try_vec(prepared.groups().len(), "JPEG XR Metal batch group errors")?;
        errors.extend_from_slice(prepared.errors());
        for group in prepared.groups() {
            match self.submit_group(group, &mut errors) {
                Ok(Some(decoded)) => groups.push(decoded),
                Ok(None) => {}
                Err(error) => group_errors.push(error),
            }
        }
        errors.sort_by_key(IndexedBatchError::index);
        Ok(SubmittedMetalPreparedBatch {
            groups,
            errors,
            group_errors,
        })
    }

    /// Submit each homogeneous group into one internally owned allocation.
    pub fn submit_prepared_dense(
        &self,
        prepared: &PreparedBatch,
    ) -> Result<SubmittedMetalDenseBatch, MetalBatchError> {
        let input_count = prepared.input_count();
        if input_count > self.options().max_inputs {
            return Err(BatchInfrastructureError::TooManyInputs {
                requested: input_count,
                maximum: self.options().max_inputs,
            }
            .into());
        }
        let mut groups = try_vec(prepared.groups().len(), "JPEG XR Metal dense groups")?;
        let mut errors = try_vec(input_count, "JPEG XR Metal batch indexed errors")?;
        let mut group_errors =
            try_vec(prepared.groups().len(), "JPEG XR Metal batch group errors")?;
        errors.extend_from_slice(prepared.errors());
        for group in prepared.groups() {
            match self.submit_dense_group(group, &mut errors) {
                Ok(Some(submitted)) => groups.push(submitted),
                Ok(None) => {}
                Err(error) => group_errors.push(error),
            }
        }
        errors.sort_by_key(IndexedBatchError::index);
        Ok(SubmittedMetalDenseBatch {
            groups,
            errors,
            group_errors,
        })
    }

    fn submit_group(
        &self,
        group: &PreparedBatchGroup,
        errors: &mut Vec<IndexedBatchError>,
    ) -> Result<Option<SubmittedMetalGroup>, MetalBatchGroupError> {
        let mut candidates = self.count_candidates(group, errors);
        let mut pending = PendingGroup::with_capacity(candidates.len());
        let mut prepared: Option<Vec<PlannedItem<'_>>> = None;
        while !candidates.is_empty() || prepared.is_some() {
            let rest = candidates.split_off(candidates.len().min(PIPELINE_CHUNK_IMAGES));
            let chunk = core::mem::replace(&mut candidates, rest);
            let staged = self.stage_chunk(chunk, &mut pending);
            let mut planned = None;
            self.preparer.in_place_scope(|scope| {
                if let Some((chunk, staging)) = staged {
                    scope.spawn(|_| planned = Some(entropy_into_staging(chunk, staging)));
                }
                // Encode the previous chunk on this thread while the pool
                // decodes the next chunk's entropy.
                if let Some(items) = prepared.take() {
                    pending.submit(&self.session, &items);
                }
            });
            prepared = planned.map(|planned| classify_planned(planned, errors));
        }
        pending.finish(group)
    }

    /// Allocates staging for one chunk on the session thread.
    ///
    /// Returns `None` for an empty chunk or once the group has failed; a group
    /// failure covers every image without its own decode error.
    fn stage_chunk<'a>(
        &self,
        chunk: Vec<CountedCandidate<'a>>,
        pending: &mut PendingGroup,
    ) -> Option<(
        Vec<CountedCandidate<'a>>,
        Vec<jxr_metal::MetalCoefficientStaging>,
    )> {
        if chunk.is_empty() {
            return None;
        }
        if pending.failure.is_some() {
            pending
                .source_indices
                .extend(chunk.iter().map(|(source_index, _, _)| *source_index));
            return None;
        }
        match self.stage_candidates(&chunk) {
            Ok(staging) => Some((chunk, staging)),
            Err(error) => {
                pending
                    .source_indices
                    .extend(error.source_indices().iter().copied());
                pending.failure = Some(error.source);
                None
            }
        }
    }

    fn submit_dense_group(
        &self,
        group: &PreparedBatchGroup,
        errors: &mut Vec<IndexedBatchError>,
    ) -> Result<Option<SubmittedDenseGroup>, MetalBatchGroupError> {
        let candidates = self.count_candidates(group, errors);
        if candidates.is_empty() {
            return Ok(None);
        }
        let items = self.prepare_candidates(candidates, errors)?;
        if items.is_empty() {
            return Ok(None);
        }
        let source_indices = items
            .iter()
            .map(|(source_index, _, _)| *source_index)
            .collect::<Vec<_>>();
        let plans = items
            .iter()
            .map(|(_, _, plan)| plan.clone())
            .collect::<Vec<_>>();
        let submission = self
            .session
            .submit_dense_batch(&plans)
            .map_err(|source| MetalBatchGroupError::new(source_indices.clone(), source))?;
        let image_infos = items
            .iter()
            .map(|(_, image, _)| image.plan().info.clone())
            .collect();
        let decoded_regions = items
            .iter()
            .map(|(_, image, _)| image.plan().output_region)
            .collect();
        Ok(Some(SubmittedDenseGroup {
            info: group.info().clone(),
            source_indices,
            image_infos,
            decoded_regions,
            submission,
        }))
    }

    fn count_candidates<'a>(
        &self,
        group: &'a PreparedBatchGroup,
        errors: &mut Vec<IndexedBatchError>,
    ) -> Vec<CountedCandidate<'a>> {
        let counted = self.preparer.install(|| {
            group
                .images()
                .par_iter()
                .zip(group.source_indices())
                .map(|(image, &source_index)| {
                    (
                        source_index,
                        validate_metal_request(image.request()).and_then(|()| {
                            image
                                .image()
                                .decoder()
                                .metal_coefficient_count_for_plan(image.plan())
                        }),
                        image,
                    )
                })
                .collect::<Vec<_>>()
        });
        let mut candidates = Vec::with_capacity(counted.len());
        for (source_index, count, image) in counted {
            match count {
                Ok(count) => candidates.push((source_index, count, image)),
                Err(source) => errors.push(IndexedBatchError::new(
                    source_index,
                    BatchErrorStage::Decode,
                    source,
                )),
            }
        }
        candidates
    }

    fn prepare_candidates<'a>(
        &self,
        candidates: Vec<CountedCandidate<'a>>,
        errors: &mut Vec<IndexedBatchError>,
    ) -> Result<Vec<PlannedItem<'a>>, MetalBatchGroupError> {
        let staging = self.stage_candidates(&candidates)?;
        let planned = self
            .preparer
            .install(|| entropy_into_staging(candidates, staging));
        Ok(classify_planned(planned, errors))
    }

    /// Allocates one shared staging slice per candidate.
    fn stage_candidates(
        &self,
        candidates: &[CountedCandidate<'_>],
    ) -> Result<Vec<jxr_metal::MetalCoefficientStaging>, MetalBatchGroupError> {
        let counts = candidates
            .iter()
            .map(|(_, count, _)| *count)
            .collect::<Vec<_>>();
        self.session
            .coefficient_staging_slices(&counts)
            .map_err(|source| {
                MetalBatchGroupError::new(
                    candidates
                        .iter()
                        .map(|(source_index, _, _)| *source_index)
                        .collect(),
                    source,
                )
            })
    }

    /// Submit one all-success prepared group into a caller-owned dense destination.
    pub fn submit_prepared_group_into(
        &self,
        group: &PreparedBatchGroup,
        destination: jxr_metal::MetalBatchDestination,
    ) -> Result<SubmittedMetalPreparedGroupInto, MetalBatchError> {
        let mut errors = Vec::new();
        let candidates = self.count_candidates(group, &mut errors);
        let items = self
            .prepare_candidates(candidates, &mut errors)
            .map_err(|error| MetalBatchError::Metal(error.source))?;
        if !errors.is_empty() || items.len() != group.images().len() {
            return Err(MetalBatchError::IndexedPreparation {
                count: errors.len(),
                errors,
            });
        }
        let source_indices = items
            .iter()
            .map(|(source_index, _, _)| *source_index)
            .collect::<Vec<_>>();
        let image_infos = items
            .iter()
            .map(|(_, image, _)| image.plan().info.clone())
            .collect();
        let decoded_regions = items
            .iter()
            .map(|(_, image, _)| image.plan().output_region)
            .collect();
        let plans = items
            .iter()
            .map(|(_, _, plan)| plan.clone())
            .collect::<Vec<_>>();
        let submission = self.session.submit_batch_into(&plans, destination)?;
        Ok(SubmittedMetalPreparedGroupInto {
            source_indices,
            image_infos,
            decoded_regions,
            submission,
        })
    }
}

/// Decodes CPU entropy for every candidate into its staging slice, in parallel
/// on the current pool.
fn entropy_into_staging(
    candidates: Vec<CountedCandidate<'_>>,
    staging: Vec<jxr_metal::MetalCoefficientStaging>,
) -> Vec<StagedPlan<'_>> {
    candidates
        .into_par_iter()
        .zip(staging)
        .map(|((source_index, _, image), staging)| {
            let plan = image.image().decoder().prepare_metal_plan_with_staging(
                image.request(),
                image.plan(),
                staging,
            );
            (source_index, image, plan)
        })
        .collect()
}

/// Separates successful plans from indexed decode errors.
fn classify_planned<'a>(
    planned: Vec<StagedPlan<'a>>,
    errors: &mut Vec<IndexedBatchError>,
) -> Vec<PlannedItem<'a>> {
    let mut items = Vec::with_capacity(planned.len());
    for (source_index, image, plan) in planned {
        match plan {
            Ok(plan) => items.push((source_index, image, plan)),
            Err(source) => errors.push(IndexedBatchError::new(
                source_index,
                BatchErrorStage::Decode,
                source,
            )),
        }
    }
    items
}

/// Chunk submissions and every image they account for, in source order.
struct PendingGroup {
    source_indices: Vec<usize>,
    image_infos: Vec<jxr_core::ImageInfo>,
    decoded_regions: Vec<jxr_core::Rect>,
    submissions: Vec<jxr_metal::MetalBatchSubmission>,
    failure: Option<jxr_metal::MetalError>,
}

impl PendingGroup {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            source_indices: Vec::with_capacity(capacity),
            image_infos: Vec::with_capacity(capacity),
            decoded_regions: Vec::with_capacity(capacity),
            submissions: Vec::new(),
            failure: None,
        }
    }

    /// Submits one prepared chunk unless the group has already failed.
    fn submit(&mut self, session: &jxr_metal::MetalDecoderSession, items: &[PlannedItem<'_>]) {
        if items.is_empty() {
            return;
        }
        if self.failure.is_none() {
            let plans = items
                .iter()
                .map(|(_, _, plan)| plan.clone())
                .collect::<Vec<_>>();
            match session.submit_batch(&plans) {
                Ok(submission) => self.submissions.push(submission),
                Err(source) => self.failure = Some(source),
            }
        }
        for (source_index, image, _) in items {
            self.source_indices.push(*source_index);
            self.image_infos.push(image.plan().info.clone());
            self.decoded_regions.push(image.plan().output_region);
        }
    }

    fn finish(
        self,
        group: &PreparedBatchGroup,
    ) -> Result<Option<SubmittedMetalGroup>, MetalBatchGroupError> {
        if let Some(source) = self.failure {
            return Err(MetalBatchGroupError::new(self.source_indices, source));
        }
        if self.source_indices.is_empty() {
            return Ok(None);
        }
        Ok(Some(SubmittedMetalGroup {
            info: group.info().clone(),
            source_indices: self.source_indices,
            image_infos: self.image_infos,
            decoded_regions: self.decoded_regions,
            submissions: self.submissions,
        }))
    }
}

fn validate_metal_request(request: &jxr_core::DecodeRequest) -> Result<(), jxr_core::JxrError> {
    if request.scale != jxr_core::DecodeScale::Full {
        return Err(jxr_core::JxrError::new(
            jxr_core::JxrErrorKind::Unsupported,
            "Metal batch reconstruction of native reduced output",
        ));
    }
    if matches!(
        request.backend,
        jxr_core::BackendRequest::Auto | jxr_core::BackendRequest::Metal
    ) {
        return Ok(());
    }
    Err(jxr_core::JxrError::new(
        jxr_core::JxrErrorKind::BackendUnavailable,
        "select Metal batch decoder",
    ))
}

impl BatchDecoder for MetalBatchDecoder {
    type Output = MetalBatchDecodeResult;
    type Error = MetalBatchError;

    fn options(&self) -> BatchDecodeOptions {
        self.options()
    }

    fn prepare_batch(&self, inputs: Vec<EncodedImage>) -> Result<PreparedBatch, Self::Error> {
        self.prepare(inputs)
    }

    fn prepare_prepared_images(
        &self,
        images: Vec<PreparedImage>,
    ) -> Result<PreparedBatch, Self::Error> {
        self.prepare_prepared_images(images)
    }

    fn decode_prepared(&mut self, prepared: &PreparedBatch) -> Result<Self::Output, Self::Error> {
        MetalBatchDecoder::decode_prepared(self, prepared)
    }
}

#[cfg(test)]
mod tests {
    use jxr_core::{ChannelLayout, DecodeRequest, DecodeScale, PixelFormat};

    use super::validate_metal_request;

    #[test]
    fn metal_batches_reject_native_reduced_requests_before_preparation() {
        let request = DecodeRequest::new(PixelFormat::U8(ChannelLayout::Rgb))
            .with_scale(DecodeScale::Sixteenth);
        let error = validate_metal_request(&request).unwrap_err();
        assert_eq!(error.kind, jxr_core::JxrErrorKind::Unsupported);
    }
}
