use std::sync::{
    Mutex,
    atomic::{AtomicU64, Ordering},
};

use rayon::prelude::*;

use jxr_core::JxrError;

use super::{
    BatchDecodeOptions, BatchDecoder, BatchErrorStage, BatchInfrastructureError,
    CpuBatchDecodeResult, CpuBatchDestination, CpuBatchDiagnostics, CpuBatchGroup,
    CpuBatchIntoResult, CpuBatchSamples, EncodedImage, IndexedBatchError, PreparedBatch,
    PreparedBatchGroup, PreparedImage,
    prepare::{PreparationCache, build_worker_pool, prepare_batch_in_pool_cached, try_vec},
    prepare_batch_from_images,
};

mod output;

use output::{
    CpuLayoutWorkspace, validate_group_destination, validate_output_budget, validate_prepared_count,
};

/// Persistent native CPU batch decoder with a retained worker pool.
pub struct CpuBatchDecoder {
    options: BatchDecodeOptions,
    workers: rayon::ThreadPool,
    workspaces: Mutex<Vec<CpuWorkerWorkspace>>,
    max_retained_workspaces: usize,
    preparation_cache: Mutex<PreparationCache>,
    diagnostics: DiagnosticCounters,
}

#[derive(Debug, Default)]
struct CpuWorkerWorkspace {
    native: jxr_native::CpuDecodeWorkspace,
    layout: CpuLayoutWorkspace,
}

#[derive(Debug, Default)]
struct DiagnosticCounters {
    preparation_calls: AtomicU64,
    prepared_inputs: AtomicU64,
    preparation_cache_hits: AtomicU64,
    preparation_cache_misses: AtomicU64,
    decode_calls: AtomicU64,
    direct_dense_images: AtomicU64,
    fallback_materialized_images: AtomicU64,
    coefficient_workspace_reuses: AtomicU64,
    reconstruction_workspace_reuses: AtomicU64,
    layout_workspace_reuses: AtomicU64,
    output_compaction_copied_samples: AtomicU64,
}

impl core::fmt::Debug for CpuBatchDecoder {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("CpuBatchDecoder")
            .field("options", &self.options)
            .field("worker_count", &self.workers.current_num_threads())
            .finish_non_exhaustive()
    }
}

impl CpuBatchDecoder {
    /// Create a persistent native batch decoder.
    pub fn new(options: BatchDecodeOptions) -> Result<Self, BatchInfrastructureError> {
        let workers = build_worker_pool(options)?;
        let worker_count = workers.current_num_threads();
        let workspaces = (0..worker_count)
            .map(|_| CpuWorkerWorkspace::default())
            .collect();
        Ok(Self {
            options,
            workers,
            workspaces: Mutex::new(workspaces),
            max_retained_workspaces: worker_count,
            preparation_cache: Mutex::new(PreparationCache::default()),
            diagnostics: DiagnosticCounters::default(),
        })
    }

    /// Retained batch policy.
    #[must_use]
    pub const fn options(&self) -> BatchDecodeOptions {
        self.options
    }

    /// Number of image-level workers retained by this session.
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.workers.current_num_threads()
    }

    #[cfg(any(feature = "metal", feature = "cuda"))]
    pub(crate) fn install<Output: Send>(
        &self,
        operation: impl FnOnce() -> Output + Send,
    ) -> Output {
        self.workers.install(operation)
    }

    /// Parse and group owned compressed inputs concurrently.
    pub fn prepare(
        &self,
        inputs: Vec<EncodedImage>,
    ) -> Result<PreparedBatch, BatchInfrastructureError> {
        let input_count = inputs.len();
        let (prepared, stats) = prepare_batch_in_pool_cached(
            &self.workers,
            inputs,
            self.options,
            &self.preparation_cache,
        )?;
        self.diagnostics
            .preparation_calls
            .fetch_add(1, Ordering::Relaxed);
        self.diagnostics.prepared_inputs.fetch_add(
            u64::try_from(input_count).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.diagnostics
            .preparation_cache_hits
            .fetch_add(stats.hits, Ordering::Relaxed);
        self.diagnostics
            .preparation_cache_misses
            .fetch_add(stats.misses, Ordering::Relaxed);
        Ok(prepared)
    }

    /// Regroup prepared images without reparsing them.
    pub fn prepare_prepared_images(
        &self,
        images: Vec<PreparedImage>,
    ) -> Result<PreparedBatch, BatchInfrastructureError> {
        prepare_batch_from_images(images, self.options)
    }

    /// Prepare and decode one owned batch.
    pub fn decode(
        &self,
        inputs: Vec<EncodedImage>,
    ) -> Result<CpuBatchDecodeResult, BatchInfrastructureError> {
        let prepared = self.prepare(inputs)?;
        self.decode_prepared(&prepared)
    }

    /// Decode a reusable prepared batch without reparsing or replanning inputs.
    pub fn decode_prepared(
        &self,
        prepared: &PreparedBatch,
    ) -> Result<CpuBatchDecodeResult, BatchInfrastructureError> {
        validate_prepared_count(prepared, self.options)?;
        validate_output_budget(prepared, self.options)?;
        self.diagnostics
            .decode_calls
            .fetch_add(1, Ordering::Relaxed);
        let input_count = prepared.input_count();
        let mut groups = try_vec(prepared.groups().len(), "JPEG XR CPU batch groups")?;
        let mut errors = try_vec(input_count, "JPEG XR CPU batch indexed errors")?;
        errors.extend_from_slice(prepared.errors());
        for group in prepared.groups() {
            if let Some(decoded) = self.decode_group(group, &mut errors)? {
                groups.push(decoded);
            }
        }
        errors.sort_by_key(IndexedBatchError::index);
        Ok(CpuBatchDecodeResult { groups, errors })
    }

    /// Monotonic session diagnostics and current retained workspace capacity.
    #[must_use]
    pub fn diagnostics(&self) -> CpuBatchDiagnostics {
        let workspaces = self
            .workspaces
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let retained_coefficient_bytes = workspaces
            .iter()
            .map(|workspace| {
                u64::try_from(workspace.native.retained_coefficient_bytes()).unwrap_or(u64::MAX)
            })
            .fold(0_u64, u64::saturating_add);
        let retained_reconstruction_bytes = workspaces
            .iter()
            .map(|workspace| {
                u64::try_from(workspace.native.retained_reconstruction_bytes()).unwrap_or(u64::MAX)
            })
            .fold(0_u64, u64::saturating_add);
        let retained_layout_bytes = workspaces
            .iter()
            .map(|workspace| u64::try_from(workspace.layout.retained_bytes()).unwrap_or(u64::MAX))
            .fold(0_u64, u64::saturating_add);
        CpuBatchDiagnostics {
            preparation_calls: self.diagnostics.preparation_calls.load(Ordering::Relaxed),
            prepared_inputs: self.diagnostics.prepared_inputs.load(Ordering::Relaxed),
            preparation_cache_hits: self
                .diagnostics
                .preparation_cache_hits
                .load(Ordering::Relaxed),
            preparation_cache_misses: self
                .diagnostics
                .preparation_cache_misses
                .load(Ordering::Relaxed),
            decode_calls: self.diagnostics.decode_calls.load(Ordering::Relaxed),
            direct_dense_images: self.diagnostics.direct_dense_images.load(Ordering::Relaxed),
            fallback_materialized_images: self
                .diagnostics
                .fallback_materialized_images
                .load(Ordering::Relaxed),
            coefficient_workspace_reuses: self
                .diagnostics
                .coefficient_workspace_reuses
                .load(Ordering::Relaxed),
            retained_coefficient_bytes,
            reconstruction_workspace_reuses: self
                .diagnostics
                .reconstruction_workspace_reuses
                .load(Ordering::Relaxed),
            retained_reconstruction_bytes,
            layout_workspace_reuses: self
                .diagnostics
                .layout_workspace_reuses
                .load(Ordering::Relaxed),
            retained_layout_bytes,
            output_compaction_copied_samples: self
                .diagnostics
                .output_compaction_copied_samples
                .load(Ordering::Relaxed),
        }
    }

    fn decode_group(
        &self,
        group: &PreparedBatchGroup,
        errors: &mut Vec<IndexedBatchError>,
    ) -> Result<Option<CpuBatchGroup>, BatchInfrastructureError> {
        let stride = group.info().image_stride_elements();
        let total = stride.checked_mul(group.images().len()).ok_or(
            BatchInfrastructureError::OutputAllocationTooLarge {
                requested: u64::MAX,
                maximum: self.options.max_host_allocation_bytes,
            },
        )?;
        let mut samples = CpuBatchSamples::zeroed(
            group.info().format(),
            total,
            "JPEG XR dense CPU batch output",
        )?;
        let mut decoded = self.decode_prepared_group_into(group, samples.destination_mut())?;
        errors.append(&mut decoded.errors);
        let mut successful_slot = 0_usize;
        for (input_slot, source_index) in group.source_indices().iter().copied().enumerate() {
            if decoded.source_indices.get(successful_slot) != Some(&source_index) {
                continue;
            }
            if successful_slot != input_slot {
                samples.copy_within(
                    input_slot * stride..(input_slot + 1) * stride,
                    successful_slot * stride,
                );
                self.diagnostics
                    .output_compaction_copied_samples
                    .fetch_add(u64::try_from(stride).unwrap_or(u64::MAX), Ordering::Relaxed);
            }
            successful_slot += 1;
        }
        if successful_slot == 0 {
            return Ok(None);
        }
        samples.truncate(successful_slot * stride);
        Ok(Some(CpuBatchGroup {
            info: group.info().clone(),
            source_indices: decoded.source_indices,
            image_infos: decoded.image_infos,
            decoded_regions: decoded.decoded_regions,
            reports: decoded.reports,
            samples,
        }))
    }

    fn decode_image_into_with_workspace(
        &self,
        image: &PreparedImage,
        mut destination: jxr_core::DecodedSamplesMut<'_>,
    ) -> Result<jxr_native::CpuDecodeIntoOutput, JxrError> {
        self.with_workspace(|workspace| {
            let decoded = image
                .image()
                .decoder()
                .decode_prepared_cpu_into_with_workspace(
                    image.plan(),
                    image.request(),
                    &mut workspace.native,
                    destination.reborrow(),
                )?;
            if image.info().batch_layout() == super::BatchLayout::Nchw {
                workspace.layout.reorder_nchw(
                    destination,
                    image.info().dimensions(),
                    usize::from(image.info().format().channel_count()),
                )?;
            }
            Ok(decoded)
        })
    }

    fn with_workspace<Output>(
        &self,
        operation: impl FnOnce(&mut CpuWorkerWorkspace) -> Output,
    ) -> Output {
        let mut workspace = self
            .workspaces
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop()
            .unwrap_or_default();
        let reuses_before = workspace.native.coefficient_reuses();
        let reconstruction_reuses_before = workspace.native.reconstruction_reuses();
        let layout_reuses_before = workspace.layout.reuses();
        let output = operation(&mut workspace);
        let reuse_delta = workspace
            .native
            .coefficient_reuses()
            .saturating_sub(reuses_before);
        self.diagnostics
            .coefficient_workspace_reuses
            .fetch_add(reuse_delta, Ordering::Relaxed);
        let reconstruction_reuse_delta = workspace
            .native
            .reconstruction_reuses()
            .saturating_sub(reconstruction_reuses_before);
        self.diagnostics
            .reconstruction_workspace_reuses
            .fetch_add(reconstruction_reuse_delta, Ordering::Relaxed);
        let layout_reuse_delta = workspace
            .layout
            .reuses()
            .saturating_sub(layout_reuses_before);
        self.diagnostics
            .layout_workspace_reuses
            .fetch_add(layout_reuse_delta, Ordering::Relaxed);
        let mut retained = self
            .workspaces
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if retained.len() < self.max_retained_workspaces {
            retained.push(workspace);
        }
        output
    }

    /// Decode one prepared homogeneous group into exact caller-owned storage.
    ///
    /// Each image retains its prepared-group slot even when another image fails.
    pub fn decode_prepared_group_into(
        &self,
        group: &PreparedBatchGroup,
        destination: CpuBatchDestination<'_>,
    ) -> Result<CpuBatchIntoResult, BatchInfrastructureError> {
        validate_group_destination(group, &destination)?;
        self.decode_direct_group_into(group, destination)
    }

    fn decode_direct_group_into(
        &self,
        group: &PreparedBatchGroup,
        mut destination: CpuBatchDestination<'_>,
    ) -> Result<CpuBatchIntoResult, BatchInfrastructureError> {
        let stride = group.info().image_stride_elements();
        macro_rules! decode_slots {
            ($output:expr, $variant:ident) => {
                self.workers.install(|| {
                    $output
                        .par_chunks_mut(stride)
                        .zip(group.images().par_iter())
                        .zip(group.source_indices().par_iter().copied())
                        .map(|((slot, image), source_index)| {
                            (
                                source_index,
                                self.decode_image_into_with_workspace(
                                    image,
                                    jxr_core::DecodedSamplesMut::$variant(slot),
                                ),
                            )
                        })
                        .collect::<Vec<_>>()
                })
            };
        }
        let decoded = match &mut destination {
            CpuBatchDestination::BitPacked(output) => decode_slots!(output, BitPacked),
            CpuBatchDestination::U8(output) => decode_slots!(output, U8),
            CpuBatchDestination::U16(output) => decode_slots!(output, U16),
            CpuBatchDestination::I16(output) => decode_slots!(output, I16),
            CpuBatchDestination::I32(output) => decode_slots!(output, I32),
            CpuBatchDestination::F16(output) => decode_slots!(output, F16),
            CpuBatchDestination::F32(output) => decode_slots!(output, F32),
            CpuBatchDestination::Rgb555(output) => decode_slots!(output, Rgb555),
            CpuBatchDestination::Rgb565(output) => decode_slots!(output, Rgb565),
            CpuBatchDestination::Rgb101010(output) => decode_slots!(output, Rgb101010),
            CpuBatchDestination::Rgbe(output) => decode_slots!(output, Rgbe),
        };
        let mut result = empty_into_result(group.images().len())?;
        for (source_index, decoded) in decoded {
            match decoded {
                Ok(decoded) => {
                    result.source_indices.push(source_index);
                    result.image_infos.push(decoded.info);
                    result.decoded_regions.push(decoded.decoded_region);
                    result.reports.push(decoded.report);
                    self.diagnostics
                        .direct_dense_images
                        .fetch_add(1, Ordering::Relaxed);
                }
                Err(source) => result.errors.push(IndexedBatchError::new(
                    source_index,
                    BatchErrorStage::Decode,
                    source,
                )),
            }
        }
        Ok(result)
    }
}

fn empty_into_result(capacity: usize) -> Result<CpuBatchIntoResult, BatchInfrastructureError> {
    Ok(CpuBatchIntoResult {
        source_indices: try_vec(capacity, "JPEG XR CPU source indices")?,
        image_infos: try_vec(capacity, "JPEG XR CPU image metadata")?,
        decoded_regions: try_vec(capacity, "JPEG XR CPU decoded regions")?,
        reports: try_vec(capacity, "JPEG XR CPU decode reports")?,
        errors: try_vec(capacity, "JPEG XR CPU indexed errors")?,
    })
}

impl BatchDecoder for CpuBatchDecoder {
    type Output = CpuBatchDecodeResult;
    type Error = BatchInfrastructureError;

    fn options(&self) -> BatchDecodeOptions {
        self.options
    }

    fn prepare_batch(
        &self,
        inputs: Vec<EncodedImage>,
    ) -> Result<PreparedBatch, BatchInfrastructureError> {
        self.prepare(inputs)
    }

    fn prepare_prepared_images(
        &self,
        images: Vec<PreparedImage>,
    ) -> Result<PreparedBatch, BatchInfrastructureError> {
        self.prepare_prepared_images(images)
    }

    fn decode_prepared(
        &mut self,
        prepared: &PreparedBatch,
    ) -> Result<Self::Output, BatchInfrastructureError> {
        CpuBatchDecoder::decode_prepared(self, prepared)
    }
}
