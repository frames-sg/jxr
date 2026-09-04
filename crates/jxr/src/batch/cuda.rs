// SPDX-License-Identifier: MIT OR Apache-2.0

use rayon::prelude::*;

use super::{
    BatchDecodeOptions, BatchDecoder, BatchErrorStage, BatchInfrastructureError, CpuBatchDecoder,
    EncodedImage, IndexedBatchError, PreparedBatch, PreparedBatchGroup, PreparedImage,
    prepare::try_vec,
};

mod contracts;

use contracts::SubmittedCudaGroup;
pub use contracts::{
    CudaBatchDecodeResult, CudaBatchError, CudaBatchGroup, CudaBatchGroupError,
    SubmittedCudaPreparedBatch,
};

type PlannedItem<'a> = (usize, &'a PreparedImage, jxr_cuda::CudaDecodePlan);

/// Persistent high-level CUDA decoder for owned and prepared native batches.
pub struct CudaBatchDecoder {
    session: jxr_cuda::CudaDecoderSession,
    preparer: CpuBatchDecoder,
}

impl core::fmt::Debug for CudaBatchDecoder {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("CudaBatchDecoder")
            .field("session", &self.session)
            .field("options", &self.options())
            .finish_non_exhaustive()
    }
}

impl CudaBatchDecoder {
    /// Create a persistent decoder on the default CUDA device.
    pub fn system_default(options: BatchDecodeOptions) -> Result<Self, CudaBatchError> {
        Self::with_session(jxr_cuda::CudaDecoderSession::system_default()?, options)
    }

    /// Wrap an existing CUDA session and retain a preparation worker pool.
    pub fn with_session(
        session: jxr_cuda::CudaDecoderSession,
        options: BatchDecodeOptions,
    ) -> Result<Self, CudaBatchError> {
        if options.layout != super::BatchLayout::Native {
            return Err(BatchInfrastructureError::UnsupportedBatchLayout {
                backend: "CUDA",
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

    /// Low-level CUDA session retained across calls.
    #[must_use]
    pub const fn session(&self) -> &jxr_cuda::CudaDecoderSession {
        &self.session
    }

    /// Parse and group owned inputs using the retained worker pool.
    pub fn prepare(&self, inputs: Vec<EncodedImage>) -> Result<PreparedBatch, CudaBatchError> {
        Ok(self.preparer.prepare(inputs)?)
    }

    /// Regroup prepared images without reparsing them.
    pub fn prepare_prepared_images(
        &self,
        images: Vec<PreparedImage>,
    ) -> Result<PreparedBatch, CudaBatchError> {
        Ok(self.preparer.prepare_prepared_images(images)?)
    }

    /// Prepare and decode owned inputs to immutable CUDA images.
    pub fn decode(
        &self,
        inputs: Vec<EncodedImage>,
    ) -> Result<CudaBatchDecodeResult, CudaBatchError> {
        let prepared = self.prepare(inputs)?;
        self.decode_prepared(&prepared)
    }

    /// Decode a reusable shared prepared batch to immutable CUDA images.
    pub fn decode_prepared(
        &self,
        prepared: &PreparedBatch,
    ) -> Result<CudaBatchDecodeResult, CudaBatchError> {
        Ok(self.submit_prepared(prepared)?.wait())
    }

    /// Prepare coefficients and submit every valid homogeneous group.
    pub fn submit_prepared(
        &self,
        prepared: &PreparedBatch,
    ) -> Result<SubmittedCudaPreparedBatch, CudaBatchError> {
        let input_count = prepared.input_count();
        if input_count > self.options().max_inputs {
            return Err(BatchInfrastructureError::TooManyInputs {
                requested: input_count,
                maximum: self.options().max_inputs,
            }
            .into());
        }
        let mut groups = try_vec(prepared.groups().len(), "JPEG XR CUDA pending groups")?;
        let mut errors = try_vec(input_count, "JPEG XR CUDA batch indexed errors")?;
        let mut group_errors = try_vec(prepared.groups().len(), "JPEG XR CUDA batch group errors")?;
        errors.extend_from_slice(prepared.errors());
        for group in prepared.groups() {
            let items = self.prepare_group(group, &mut errors);
            if items.is_empty() {
                continue;
            }
            let source_indices = items
                .iter()
                .map(|(source_index, _, _)| *source_index)
                .collect::<Vec<_>>();
            let plans = items
                .iter()
                .map(|(_, _, plan)| plan.clone())
                .collect::<Vec<_>>();
            match self.session.submit_batch(&plans) {
                Ok(submission) => groups.push(SubmittedCudaGroup {
                    info: group.info().clone(),
                    source_indices,
                    image_infos: items
                        .iter()
                        .map(|(_, image, _)| image.plan().info.clone())
                        .collect(),
                    decoded_regions: items
                        .iter()
                        .map(|(_, image, _)| image.plan().output_region)
                        .collect(),
                    submission,
                }),
                Err(source) => group_errors.push(CudaBatchGroupError::new(source_indices, source)),
            }
        }
        errors.sort_by_key(IndexedBatchError::index);
        Ok(SubmittedCudaPreparedBatch {
            groups,
            errors,
            group_errors,
        })
    }

    fn prepare_group<'a>(
        &self,
        group: &'a PreparedBatchGroup,
        errors: &mut Vec<IndexedBatchError>,
    ) -> Vec<PlannedItem<'a>> {
        let planned = self.preparer.install(|| {
            group
                .images()
                .par_iter()
                .zip(group.source_indices())
                .map(|(image, &source_index)| {
                    let result = validate_cuda_request(image.request())
                        .and_then(|()| image.prepare_reconstruction())
                        .and_then(|prepared| {
                            prepared
                                .cuda_plan()
                                .map_err(|error| crate::decoder::map_cuda_error(&error))
                        });
                    (source_index, image, result)
                })
                .collect::<Vec<_>>()
        });
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
}

fn validate_cuda_request(request: &jxr_core::DecodeRequest) -> Result<(), jxr_core::JxrError> {
    if request.scale != jxr_core::DecodeScale::Full {
        return Err(jxr_core::JxrError::new(
            jxr_core::JxrErrorKind::Unsupported,
            "CUDA batch reconstruction of native reduced output",
        ));
    }
    if matches!(
        request.backend,
        jxr_core::BackendRequest::Auto | jxr_core::BackendRequest::Cuda
    ) {
        return Ok(());
    }
    Err(jxr_core::JxrError::new(
        jxr_core::JxrErrorKind::BackendUnavailable,
        "select CUDA batch decoder",
    ))
}

impl BatchDecoder for CudaBatchDecoder {
    type Output = CudaBatchDecodeResult;
    type Error = CudaBatchError;

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
        CudaBatchDecoder::decode_prepared(self, prepared)
    }
}
