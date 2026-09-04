use std::{collections::VecDeque, mem::size_of, sync::Arc};

use rayon::prelude::*;

use super::{
    BatchDecodeOptions, BatchErrorStage, BatchInfrastructureError, EncodedImage, IndexedBatchError,
    MAX_BATCH_WORKERS, PreparedBatch, PreparedBatchGroup, PreparedImage,
    contracts::{PreparedImageContract, PreparedImageInner},
};
use crate::PreparedJxr;

/// Parse, validate, and group owned JPEG XR inputs concurrently.
pub fn prepare_batch(
    inputs: Vec<EncodedImage>,
    options: BatchDecodeOptions,
) -> Result<PreparedBatch, BatchInfrastructureError> {
    let pool = build_worker_pool(options)?;
    prepare_batch_in_pool(&pool, inputs, options)
}

/// Regroup prepared images without parsing or copying their compressed bytes.
///
/// Returned group indices refer to positions in `images`; each
/// [`PreparedImage::original_source_index`] remains unchanged.
pub fn prepare_batch_from_images(
    images: Vec<PreparedImage>,
    options: BatchDecodeOptions,
) -> Result<PreparedBatch, BatchInfrastructureError> {
    validate_input_count(images.len(), options)?;
    let mut groups = try_vec(images.len(), "JPEG XR prepared groups")?;
    for (source_index, image) in images.into_iter().enumerate() {
        let image = retarget_batch_layout(image, options.layout)?;
        push_prepared(&mut groups, image, source_index)?;
    }
    Ok(PreparedBatch {
        groups: Arc::from(groups),
        errors: Arc::from([]),
        options,
    })
}

pub(crate) fn build_worker_pool(
    options: BatchDecodeOptions,
) -> Result<rayon::ThreadPool, BatchInfrastructureError> {
    let available = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let workers = options
        .workers
        .map_or(available, std::num::NonZeroUsize::get)
        .clamp(1, MAX_BATCH_WORKERS);
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|index| format!("jxr-batch-{index}"))
        .build()
        .map_err(|error| BatchInfrastructureError::WorkerInitialization {
            message: error.to_string(),
        })
}

pub(crate) fn prepare_batch_in_pool(
    pool: &rayon::ThreadPool,
    inputs: Vec<EncodedImage>,
    options: BatchDecodeOptions,
) -> Result<PreparedBatch, BatchInfrastructureError> {
    validate_input_count(inputs.len(), options)?;
    let prepared = pool.install(|| {
        inputs
            .into_par_iter()
            .enumerate()
            .map(|(source_index, input)| prepare_image(input, source_index, options.layout))
            .collect::<Vec<_>>()
    });
    let mut groups = try_vec(prepared.len(), "JPEG XR prepared groups")?;
    let mut errors = try_vec(prepared.len(), "JPEG XR preparation errors")?;
    for (source_index, result) in prepared.into_iter().enumerate() {
        match result {
            Ok(image) => push_prepared(&mut groups, image, source_index)?,
            Err(source) => errors.push(IndexedBatchError::new(
                source_index,
                BatchErrorStage::Preparation,
                source,
            )),
        }
    }
    Ok(PreparedBatch {
        groups: Arc::from(groups),
        errors: Arc::from(errors),
        options,
    })
}

#[derive(Debug, Default)]
pub(crate) struct PreparationCache {
    entries: VecDeque<PreparationCacheEntry>,
}

#[derive(Debug)]
struct PreparationCacheEntry {
    contract: Arc<PreparedImageContract>,
}

impl PreparationCache {
    fn lookup(&mut self, input: &EncodedImage) -> Option<Arc<PreparedImageContract>> {
        let position = self.entries.iter().position(|entry| {
            Arc::ptr_eq(entry.contract.image.bytes(), &input.bytes)
                && entry.contract.request == input.request
        })?;
        let entry = self.entries.remove(position)?;
        let contract = Arc::clone(&entry.contract);
        self.entries.push_back(entry);
        Some(contract)
    }

    fn insert(&mut self, contract: Arc<PreparedImageContract>, maximum: usize) {
        if maximum == 0 {
            return;
        }
        while self.entries.len() >= maximum {
            self.entries.pop_front();
        }
        self.entries.push_back(PreparationCacheEntry { contract });
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PreparationStats {
    pub(crate) hits: u64,
    pub(crate) misses: u64,
}

struct UniquePrepareJob {
    input: EncodedImage,
    source_indices: Vec<usize>,
}

pub(crate) fn prepare_batch_in_pool_cached(
    pool: &rayon::ThreadPool,
    inputs: Vec<EncodedImage>,
    options: BatchDecodeOptions,
    cache: &std::sync::Mutex<PreparationCache>,
) -> Result<(PreparedBatch, PreparationStats), BatchInfrastructureError> {
    validate_input_count(inputs.len(), options)?;
    let input_count = inputs.len();
    let mut resolved: Vec<Option<Result<Arc<PreparedImageContract>, jxr_core::JxrError>>> =
        (0..input_count).map(|_| None).collect();
    let mut jobs: Vec<UniquePrepareJob> = Vec::new();
    let mut stats = PreparationStats::default();
    for (source_index, input) in inputs.into_iter().enumerate() {
        let cached = cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .lookup(&input);
        if let Some(contract) = cached {
            resolved[source_index] = Some(Ok(contract));
            stats.hits = stats.hits.saturating_add(1);
            continue;
        }
        if let Some(job) = jobs.iter_mut().find(|job| {
            Arc::ptr_eq(&job.input.bytes, &input.bytes) && job.input.request == input.request
        }) {
            job.source_indices.push(source_index);
            stats.hits = stats.hits.saturating_add(1);
            continue;
        }
        jobs.push(UniquePrepareJob {
            input,
            source_indices: vec![source_index],
        });
        stats.misses = stats.misses.saturating_add(1);
    }
    let prepared = pool.install(|| {
        jobs.into_par_iter()
            .map(|job| {
                let result = prepare_contract(job.input, options.layout);
                (job.source_indices, result)
            })
            .collect::<Vec<_>>()
    });
    for (indices, result) in prepared {
        if let Ok(contract) = &result {
            cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(Arc::clone(contract), options.preparation_cache_entries);
        }
        for source_index in indices {
            resolved[source_index] = Some(result.clone());
        }
    }
    let mut groups = try_vec(input_count, "JPEG XR prepared groups")?;
    let mut errors = try_vec(input_count, "JPEG XR preparation errors")?;
    for (source_index, result) in resolved.into_iter().enumerate() {
        match result.expect("every batch input is resolved by cache or preparation") {
            Ok(contract) => push_prepared(
                &mut groups,
                PreparedImage {
                    inner: Arc::new(PreparedImageInner {
                        contract,
                        original_source_index: source_index,
                    }),
                },
                source_index,
            )?,
            Err(source) => errors.push(IndexedBatchError::new(
                source_index,
                BatchErrorStage::Preparation,
                source,
            )),
        }
    }
    Ok((
        PreparedBatch {
            groups: Arc::from(groups),
            errors: Arc::from(errors),
            options,
        },
        stats,
    ))
}

fn prepare_image(
    input: EncodedImage,
    source_index: usize,
    batch_layout: super::BatchLayout,
) -> Result<PreparedImage, jxr_core::JxrError> {
    let contract = prepare_contract(input, batch_layout)?;
    Ok(PreparedImage {
        inner: Arc::new(PreparedImageInner {
            contract,
            original_source_index: source_index,
        }),
    })
}

fn prepare_contract(
    input: EncodedImage,
    batch_layout: super::BatchLayout,
) -> Result<Arc<PreparedImageContract>, jxr_core::JxrError> {
    let image = PreparedJxr::from_arc(input.bytes)?;
    let (plan, layout) = image.decoder().prepare_batch_contract(&input.request)?;
    validate_batch_layout(&layout, batch_layout)?;
    Ok(Arc::new(PreparedImageContract {
        image,
        request: input.request,
        plan,
        info: super::BatchGroupInfo::new(layout, batch_layout),
        reconstruction: std::sync::OnceLock::new(),
    }))
}

fn validate_batch_layout(
    layout: &jxr_core::SurfaceLayout,
    batch_layout: super::BatchLayout,
) -> Result<(), jxr_core::JxrError> {
    if batch_layout == super::BatchLayout::Native {
        return Ok(());
    }
    if layout.planes.len() != 1
        || matches!(
            layout.format,
            jxr_core::PixelFormat::BitPacked(_)
                | jxr_core::PixelFormat::Rgb555
                | jxr_core::PixelFormat::Rgb565
                | jxr_core::PixelFormat::Rgb101010
                | jxr_core::PixelFormat::Rgbe
        )
    {
        return Err(jxr_core::JxrError::new(
            jxr_core::JxrErrorKind::Unsupported,
            "NCHW batch layout requires one unpacked image plane",
        ));
    }
    Ok(())
}

fn retarget_batch_layout(
    image: PreparedImage,
    batch_layout: super::BatchLayout,
) -> Result<PreparedImage, BatchInfrastructureError> {
    if image.info().batch_layout() == batch_layout {
        return Ok(image);
    }
    validate_batch_layout(image.info().image_layout(), batch_layout).map_err(|_| {
        BatchInfrastructureError::UnsupportedBatchLayout {
            backend: "CPU",
            layout: batch_layout,
        }
    })?;
    let contract = Arc::new(PreparedImageContract {
        image: image.image().clone(),
        request: image.request().clone(),
        plan: image.plan().clone(),
        info: super::BatchGroupInfo::new(image.info().image_layout().clone(), batch_layout),
        reconstruction: std::sync::OnceLock::new(),
    });
    Ok(PreparedImage {
        inner: Arc::new(PreparedImageInner {
            contract,
            original_source_index: image.original_source_index(),
        }),
    })
}

fn push_prepared(
    groups: &mut Vec<PreparedBatchGroup>,
    image: PreparedImage,
    source_index: usize,
) -> Result<(), BatchInfrastructureError> {
    if let Some(group) = groups.iter_mut().find(|group| group.info == *image.info()) {
        try_reserve_one(&mut group.images, "JPEG XR prepared group images")?;
        try_reserve_one(
            &mut group.source_indices,
            "JPEG XR prepared group source indices",
        )?;
        group.images.push(image);
        group.source_indices.push(source_index);
        return Ok(());
    }
    let mut images = try_vec(1, "JPEG XR prepared group images")?;
    let mut source_indices = try_vec(1, "JPEG XR prepared group source indices")?;
    let info = image.info().clone();
    images.push(image);
    source_indices.push(source_index);
    groups.push(PreparedBatchGroup {
        info,
        images,
        source_indices,
    });
    Ok(())
}

fn validate_input_count(
    count: usize,
    options: BatchDecodeOptions,
) -> Result<(), BatchInfrastructureError> {
    if count > options.max_inputs {
        return Err(BatchInfrastructureError::TooManyInputs {
            requested: count,
            maximum: options.max_inputs,
        });
    }
    Ok(())
}

pub(crate) fn try_vec<T>(
    capacity: usize,
    what: &'static str,
) -> Result<Vec<T>, BatchInfrastructureError> {
    let requested = capacity.checked_mul(size_of::<T>()).ok_or(
        BatchInfrastructureError::HostAllocationFailed {
            what,
            requested: usize::MAX,
        },
    )?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| BatchInfrastructureError::HostAllocationFailed { what, requested })?;
    Ok(values)
}

fn try_reserve_one<T>(
    values: &mut Vec<T>,
    what: &'static str,
) -> Result<(), BatchInfrastructureError> {
    if values.len() < values.capacity() {
        return Ok(());
    }
    let requested = values
        .len()
        .checked_add(1)
        .and_then(|length| length.checked_mul(size_of::<T>()))
        .unwrap_or(usize::MAX);
    values
        .try_reserve(1)
        .map_err(|_| BatchInfrastructureError::HostAllocationFailed { what, requested })
}
