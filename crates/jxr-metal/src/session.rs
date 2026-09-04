// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    MetalBatchDestination, MetalBatchDestinationSubmission, MetalBatchSubmission, MetalDecodePlan,
    MetalDestination, MetalDestinationSubmission, MetalError, MetalResidentBatch,
    MetalResidentBatchSubmission, MetalSubmission, ResidentMetalImage, SharedMetalImage,
};
use jxr_core::DecodedImage;
#[cfg(target_os = "macos")]
use jxr_core::{DecodedSamples, PlaneDescriptor, StorageKind};

#[cfg(target_os = "macos")]
use j2k_metal_support::{
    checked_blit_command_encoder, checked_buffer_read_vec, checked_command_buffer,
    checked_command_queue, checked_private_buffer, checked_shared_buffer, commit_and_wait,
};
#[cfg(target_os = "macos")]
use objc2::{rc::Retained, runtime::ProtocolObject};
#[cfg(target_os = "macos")]
use objc2_metal::MTLGPUFamily;
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBlitCommandEncoder, MTLCommandEncoder, MTLCommandQueue, MTLDevice};

#[cfg(target_os = "macos")]
use crate::runtime::MetalRuntime;

#[cfg(target_os = "macos")]
const BATCH_SCRATCH_BUDGET: usize = 256 * 1024 * 1024;

/// Reusable Metal runtime session.
#[derive(Clone)]
pub struct MetalDecoderSession {
    #[cfg(target_os = "macos")]
    runtime: j2k_metal_support::MetalRuntimeSession<MetalRuntime, String>,
    #[cfg(target_os = "macos")]
    command_queue: Option<Retained<ProtocolObject<dyn MTLCommandQueue>>>,
}

impl core::fmt::Debug for MetalDecoderSession {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MetalDecoderSession")
            .finish_non_exhaustive()
    }
}

impl MetalDecoderSession {
    #[cfg(target_os = "macos")]
    fn initialized_runtime(&self) -> Result<&MetalRuntime, MetalError> {
        let queue = self.command_queue.clone();
        self.runtime
            .get_or_init_runtime(|device| {
                MetalRuntime::build(device, queue).map_err(|error| error.to_string())
            })
            .as_ref()
            .map_err(|message| MetalError::RuntimeInitialization {
                message: message.clone(),
            })
    }

    /// Bind a reusable session to the system default Metal device.
    pub fn system_default() -> Result<Self, MetalError> {
        #[cfg(target_os = "macos")]
        {
            let runtime = j2k_metal_support::MetalRuntimeSession::system_default()?;
            Ok(Self {
                runtime,
                command_queue: None,
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(MetalError::Unavailable)
        }
    }

    /// Bind a reusable session to one ordered queue on the default device.
    ///
    /// Use this when dense caller-owned destinations must share a single queue
    /// ordering domain and the application does not need to supply the queue.
    pub fn system_default_ordered() -> Result<Self, MetalError> {
        #[cfg(target_os = "macos")]
        {
            let device = j2k_metal_support::system_default_device()?;
            let queue = checked_command_queue(&device)?;
            Self::with_command_queue(device, queue)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(MetalError::Unavailable)
        }
    }

    /// Bind a reusable session to a caller-selected Metal device.
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            runtime: j2k_metal_support::MetalRuntimeSession::new(device),
            command_queue: None,
        }
    }

    /// Bind a reusable session to an exact queue owned by `device`.
    #[cfg(target_os = "macos")]
    pub fn with_command_queue(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    ) -> Result<Self, MetalError> {
        if Retained::as_ptr(&command_queue.device()) != Retained::as_ptr(&device) {
            return Err(MetalError::InvalidDestination {
                reason: "command queue belongs to a different Metal device",
            });
        }
        Ok(Self {
            runtime: j2k_metal_support::MetalRuntimeSession::new(device),
            command_queue: Some(command_queue),
        })
    }

    /// Whether this session can accept work without a pre-submit fallback.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            let device = self.runtime.device();
            device.hasUnifiedMemory() && device.supportsFamily(MTLGPUFamily::Apple7)
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    /// Return exact-size scratch retention and high-water counters.
    #[cfg(target_os = "macos")]
    pub fn buffer_pool_diagnostics(
        &self,
    ) -> Result<crate::MetalBufferPoolsDiagnostics, MetalError> {
        self.initialized_runtime()?.buffer_pools.diagnostics()
    }

    /// Return immutable coefficient-upload cache reuse counters.
    #[cfg(target_os = "macos")]
    pub fn upload_cache_diagnostics(
        &self,
    ) -> Result<crate::MetalUploadCacheDiagnostics, MetalError> {
        self.initialized_runtime()?.upload_cache.diagnostics()
    }

    /// Number of command queues available to independent batch images.
    ///
    /// Default-device sessions use a measured four-queue scheduler. Sessions
    /// constructed with an exact caller queue report one and preserve that
    /// queue's ordering contract.
    #[cfg(target_os = "macos")]
    pub fn batch_queue_count(&self) -> Result<usize, MetalError> {
        Ok(self.initialized_runtime()?.batch_queues.len())
    }

    /// Allocate exact shared storage for direct CPU entropy output.
    pub fn coefficient_staging(
        &self,
        element_count: usize,
    ) -> Result<crate::MetalCoefficientStaging, MetalError> {
        if element_count == 0 {
            return Err(MetalError::InvalidPlan {
                reason: "coefficient staging cannot be empty",
            });
        }
        #[cfg(target_os = "macos")]
        {
            let bytes = element_count
                .checked_mul(core::mem::size_of::<i32>())
                .ok_or(MetalError::InvalidPlan {
                    reason: "coefficient staging byte count overflows usize",
                })?;
            let buffer = checked_shared_buffer(self.runtime.device(), bytes)?;
            Ok(crate::MetalCoefficientStaging::new(
                buffer,
                element_count,
                0,
            ))
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(MetalError::Unavailable)
        }
    }

    /// Allocate a private dense destination owned by the caller after return.
    // The ownership contract is identical on supported and unavailable platforms.
    #[cfg_attr(not(target_os = "macos"), allow(clippy::needless_pass_by_value))]
    pub fn allocate_batch_destination(
        &self,
        layout: crate::DenseMetalBatchLayout,
    ) -> Result<MetalBatchDestination, MetalError> {
        #[cfg(target_os = "macos")]
        {
            let buffer = checked_private_buffer(self.runtime.device(), layout.byte_len())?;
            // SAFETY: This method creates a fresh private allocation with no
            // aliases and transfers its sole writable owner into the destination.
            unsafe { MetalBatchDestination::from_exclusive_buffer(buffer, layout) }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = layout;
            Err(MetalError::Unavailable)
        }
    }

    /// Allocate one shared coefficient arena split into exact per-image slices.
    pub fn coefficient_staging_batch(
        &self,
        element_count: usize,
        image_count: usize,
    ) -> Result<Vec<crate::MetalCoefficientStaging>, MetalError> {
        self.coefficient_staging_slices(&vec![element_count; image_count])
    }

    /// Allocate one shared coefficient arena split into variable checked slices.
    pub fn coefficient_staging_slices(
        &self,
        element_counts: &[usize],
    ) -> Result<Vec<crate::MetalCoefficientStaging>, MetalError> {
        if element_counts.is_empty() || element_counts.contains(&0) {
            return Err(MetalError::InvalidPlan {
                reason: "coefficient staging slices cannot be empty",
            });
        }
        let total_elements = element_counts.iter().try_fold(0_usize, |total, &count| {
            total.checked_add(count).ok_or(MetalError::InvalidPlan {
                reason: "coefficient staging slice length overflows usize",
            })
        })?;
        #[cfg(target_os = "macos")]
        {
            let bytes = total_elements
                .checked_mul(core::mem::size_of::<i32>())
                .ok_or(MetalError::InvalidPlan {
                    reason: "coefficient staging slice byte count overflows usize",
                })?;
            let buffer = checked_shared_buffer(self.runtime.device(), bytes)?;
            let mut offset = 0_usize;
            let mut slices = Vec::with_capacity(element_counts.len());
            for &count in element_counts {
                slices.push(crate::MetalCoefficientStaging::new(
                    buffer.clone(),
                    count,
                    offset,
                ));
                offset = offset.checked_add(count).ok_or(MetalError::InvalidPlan {
                    reason: "coefficient staging slice offset overflows usize",
                })?;
            }
            Ok(slices)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = total_elements;
            Err(MetalError::Unavailable)
        }
    }

    /// Submit a validated plan.
    ///
    /// Pipeline compilation is lazy and cached for the lifetime of every clone
    /// of this session. Errors happen before command-buffer submission.
    pub fn submit(&self, plan: &MetalDecodePlan) -> Result<MetalSubmission, MetalError> {
        #[cfg(target_os = "macos")]
        {
            let runtime = self.initialized_runtime()?;
            let encoded = crate::encode::encode(runtime, plan)?;
            Ok(MetalSubmission::submitted(
                encoded,
                plan.decode_report(false),
            ))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = plan;
            Err(MetalError::Unavailable)
        }
    }

    /// Submit a batch in caller order without waiting between images.
    pub fn submit_batch(
        &self,
        plans: &[MetalDecodePlan],
    ) -> Result<MetalBatchSubmission, MetalError> {
        #[cfg(target_os = "macos")]
        {
            let runtime = self.initialized_runtime()?;
            let mut submissions = Vec::with_capacity(plans.len());
            for group in batch_groups(plans)? {
                append_batch_group(runtime, &plans[group], &mut submissions)?;
            }
            Ok(MetalBatchSubmission::new(submissions))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = plans;
            Err(MetalError::Unavailable)
        }
    }

    /// Submit a homogeneous batch into one internally owned private allocation.
    pub fn submit_dense_batch(
        &self,
        plans: &[MetalDecodePlan],
    ) -> Result<MetalResidentBatchSubmission, MetalError> {
        let first = plans.first().ok_or(MetalError::InvalidPlan {
            reason: "dense Metal batch cannot be empty",
        })?;
        let layout = crate::DenseMetalBatchLayout::new(first.output().clone(), plans.len())?;
        if plans.iter().any(|plan| plan.output() != first.output()) {
            return Err(MetalError::InvalidPlan {
                reason: "dense Metal batch plans are not homogeneous",
            });
        }
        #[cfg(target_os = "macos")]
        {
            let runtime = self.initialized_runtime()?;
            let output = checked_private_buffer(self.runtime.device(), layout.byte_len())?;
            let mut submissions = Vec::with_capacity(plans.len());
            let mut reports = Vec::with_capacity(plans.len());
            for (image, plan) in plans.iter().enumerate() {
                let offset = layout.image_offset(image)?;
                let encoded = crate::encode::encode_into_at(runtime, plan, output.clone(), offset)?;
                let report = plan.decode_report(false);
                submissions.push(MetalSubmission::submitted(encoded, report.clone()));
                reports.push(report);
            }
            Ok(MetalResidentBatchSubmission::new(
                submissions,
                output,
                layout,
                reports,
            ))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = layout;
            Err(MetalError::Unavailable)
        }
    }

    /// Submit directly into an exclusively retained caller-owned allocation.
    // The ownership contract is identical on supported and unavailable platforms.
    #[cfg_attr(not(target_os = "macos"), allow(clippy::needless_pass_by_value))]
    pub fn submit_into(
        &self,
        plan: &MetalDecodePlan,
        destination: MetalDestination,
    ) -> Result<MetalDestinationSubmission, MetalError> {
        destination.validate_plan(plan)?;
        #[cfg(target_os = "macos")]
        {
            destination.validate_device(self.runtime.device())?;
            let runtime = self.initialized_runtime()?;
            let encoded = crate::encode::encode_into(runtime, plan, destination.buffer_handle())?;
            let submission = MetalSubmission::submitted(encoded, plan.decode_report(false));
            Ok(MetalDestinationSubmission::new(
                submission,
                destination,
                plan.decode_report(false),
            ))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = destination;
            Err(MetalError::Unavailable)
        }
    }

    /// Submit a dense batch into one private allocation on the exact caller queue.
    ///
    /// This path intentionally rejects default sessions because their normal
    /// batch scheduler spans four queues, which cannot safely share one tracked
    /// writable allocation without additional synchronization.
    // The ownership contract is identical on supported and unavailable platforms.
    #[cfg_attr(not(target_os = "macos"), allow(clippy::needless_pass_by_value))]
    pub fn submit_batch_into(
        &self,
        plans: &[MetalDecodePlan],
        destination: MetalBatchDestination,
    ) -> Result<MetalBatchDestinationSubmission, MetalError> {
        destination.validate_plans(plans)?;
        #[cfg(target_os = "macos")]
        {
            if self.command_queue.is_none() {
                return Err(MetalError::InvalidDestination {
                    reason: "dense batch submission requires an exact caller command queue",
                });
            }
            destination.validate_device(self.runtime.device())?;
            let runtime = self.initialized_runtime()?;
            let output = destination.buffer_handle();
            let mut submissions = Vec::with_capacity(plans.len());
            let mut reports = Vec::with_capacity(plans.len());
            for (image, plan) in plans.iter().enumerate() {
                let offset = destination.layout().image_offset(image)?;
                let encoded = crate::encode::encode_into_at(runtime, plan, output.clone(), offset)?;
                let report = plan.decode_report(false);
                submissions.push(MetalSubmission::submitted(encoded, report.clone()));
                reports.push(report);
            }
            Ok(MetalBatchDestinationSubmission::new(
                submissions,
                destination,
                reports,
            ))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = destination;
            Err(MetalError::Unavailable)
        }
    }

    /// Submit, wait, and return typed host samples.
    pub fn decode_to_host(&self, plan: &MetalDecodePlan) -> Result<DecodedImage, MetalError> {
        self.decode_batch_to_host(core::slice::from_ref(plan))?
            .pop()
            .ok_or(MetalError::InvalidPlan {
                reason: "single-image Metal batch returned no output",
            })
    }

    /// Decode a batch directly into shared host-visible outputs.
    ///
    /// Unlike resident submission followed by [`Self::readback`], this path
    /// never creates private final outputs and does not encode blit command
    /// buffers. CPU entropy preparation must already have produced `plans`.
    pub fn decode_batch_to_host(
        &self,
        plans: &[MetalDecodePlan],
    ) -> Result<Vec<DecodedImage>, MetalError> {
        #[cfg(target_os = "macos")]
        {
            let runtime = self.initialized_runtime()?;
            let mut decoded = Vec::with_capacity(plans.len());
            for group in batch_groups(plans)? {
                append_host_batch_group(runtime, &plans[group.clone()], &mut decoded)?;
            }
            Ok(decoded)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = plans;
            Err(MetalError::Unavailable)
        }
    }

    /// Decode a batch into completed shared Metal buffers without copying into `Vec`s.
    ///
    /// The returned images recycle their allocations to this session when
    /// dropped. Use [`SharedMetalImage::with_bytes`] for safe zero-copy host
    /// inspection, or keep using [`Self::decode_batch_to_host`] when owned
    /// typed samples are required.
    pub fn decode_batch_to_shared(
        &self,
        plans: &[MetalDecodePlan],
    ) -> Result<Vec<SharedMetalImage>, MetalError> {
        #[cfg(target_os = "macos")]
        {
            let runtime = self.initialized_runtime()?;
            let mut images = Vec::with_capacity(plans.len());
            for group in batch_groups(plans)? {
                append_shared_batch_group(runtime, &plans[group], &mut images)?;
            }
            Ok(images)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = plans;
            Err(MetalError::Unavailable)
        }
    }

    /// Copy a completed resident image to host-visible bytes.
    pub fn readback(&self, image: &ResidentMetalImage) -> Result<Vec<u8>, MetalError> {
        #[cfg(target_os = "macos")]
        {
            image.validate_device(self.runtime.device())?;
            let byte_len = image.layout().byte_len;
            let shared = checked_shared_buffer(self.runtime.device(), byte_len)?;
            let queue = checked_command_queue(self.runtime.device())?;
            let command = checked_command_buffer(&queue)?;
            let blit = checked_blit_command_encoder(&command)?;
            // SAFETY: `ResidentMetalImage` exposes an immutable, completed
            // allocation. The blit only reads it, and this command retains the
            // source and destination until `commit_and_wait` completes.
            unsafe {
                let source = image.raw_buffer();
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    source, 0, &shared, 0, byte_len,
                );
            }
            blit.endEncoding();
            commit_and_wait(&command)?;
            // SAFETY: `shared` is CPU-visible, the blit has completed, and the
            // checked range exactly matches its allocation.
            unsafe { checked_buffer_read_vec::<u8>(&shared, 0, byte_len) }.map_err(MetalError::from)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = image;
            Err(MetalError::Unavailable)
        }
    }

    /// Copy one image from a completed dense resident batch to host bytes.
    pub fn readback_batch_image(
        &self,
        batch: &MetalResidentBatch,
        image: usize,
    ) -> Result<Vec<u8>, MetalError> {
        #[cfg(target_os = "macos")]
        {
            batch.validate_device(self.runtime.device())?;
            let byte_len = batch.layout().image_stride();
            let source_offset = batch.layout().image_offset(image)?;
            let shared = checked_shared_buffer(self.runtime.device(), byte_len)?;
            let queue = checked_command_queue(self.runtime.device())?;
            let command = checked_command_buffer(&queue)?;
            let blit = checked_blit_command_encoder(&command)?;
            // SAFETY: `batch` is completed and immutable, the checked source
            // range identifies one image, and the command retains both buffers.
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    batch.raw_buffer(),
                    source_offset,
                    &shared,
                    0,
                    byte_len,
                );
            }
            blit.endEncoding();
            commit_and_wait(&command)?;
            // SAFETY: The blit completed and `shared` contains exactly one image.
            unsafe { checked_buffer_read_vec::<u8>(&shared, 0, byte_len) }.map_err(MetalError::from)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (batch, image);
            Err(MetalError::Unavailable)
        }
    }
}

#[cfg(target_os = "macos")]
fn append_batch_group(
    runtime: &MetalRuntime,
    plans: &[MetalDecodePlan],
    submissions: &mut Vec<MetalSubmission>,
) -> Result<(), MetalError> {
    let encoded = crate::encode::encode_batch(runtime, plans)?;
    submissions.extend(
        encoded
            .into_iter()
            .zip(plans)
            .map(|(encoded, plan)| MetalSubmission::submitted(encoded, plan.decode_report(false))),
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn append_host_batch_group(
    runtime: &MetalRuntime,
    plans: &[MetalDecodePlan],
    decoded: &mut Vec<DecodedImage>,
) -> Result<(), MetalError> {
    let mut shared = Vec::with_capacity(plans.len());
    append_shared_batch_group(runtime, plans, &mut shared)?;
    for (plan, image) in plans.iter().zip(shared) {
        let bytes = image.with_bytes(<[u8]>::to_vec)?;
        decoded.push(decoded_image(plan, bytes)?);
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn append_shared_batch_group(
    runtime: &MetalRuntime,
    plans: &[MetalDecodePlan],
    images: &mut Vec<SharedMetalImage>,
) -> Result<(), MetalError> {
    let outputs = plans
        .iter()
        .map(|plan| {
            runtime
                .buffer_pools
                .take_shared(&runtime.queue.device(), plan.output().byte_len)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let handles = outputs
        .iter()
        .map(crate::buffer_pool::PooledBuffer::handle)
        .collect::<Vec<_>>();
    let encoded = crate::encode::encode_batch_into(runtime, plans, &handles)?;
    let submissions = encoded
        .into_iter()
        .zip(plans)
        .map(|(encoded, plan)| MetalSubmission::submitted(encoded, plan.decode_report(false)))
        .collect();
    let resident = MetalBatchSubmission::new(submissions).wait()?;
    for ((plan, image), output) in plans.iter().zip(resident).zip(outputs) {
        drop(image);
        let info = plan.info().ok_or(MetalError::InvalidPlan {
            reason: "metadata-only plan cannot produce a shared image",
        })?;
        let decoded_region = plan.output_region().ok_or(MetalError::InvalidPlan {
            reason: "metadata-only plan omits its shared output region",
        })?;
        images.push(SharedMetalImage::from_pooled(
            output,
            runtime.buffer_pools.clone(),
            plan.output().clone(),
            info.clone(),
            decoded_region,
            plan.output().format,
            plan.decode_report(false),
        ));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn batch_groups(plans: &[MetalDecodePlan]) -> Result<Vec<core::ops::Range<usize>>, MetalError> {
    let mut groups = Vec::new();
    let mut group_start = 0;
    let mut group_bytes = 0_usize;
    for (index, plan) in plans.iter().enumerate() {
        let bytes = plan.scratch_bytes()?;
        if bytes > BATCH_SCRATCH_BUDGET {
            return Err(MetalError::InvalidPlan {
                reason: "one image exceeds the Metal batch scratch budget",
            });
        }
        if group_bytes
            .checked_add(bytes)
            .is_none_or(|sum| sum > BATCH_SCRATCH_BUDGET)
        {
            groups.push(group_start..index);
            group_start = index;
            group_bytes = 0;
        }
        group_bytes = group_bytes
            .checked_add(bytes)
            .ok_or(MetalError::InvalidPlan {
                reason: "Metal batch scratch byte count overflows usize",
            })?;
    }
    if group_start < plans.len() {
        groups.push(group_start..plans.len());
    }
    Ok(groups)
}

#[cfg(target_os = "macos")]
fn decoded_image(plan: &MetalDecodePlan, bytes: Vec<u8>) -> Result<DecodedImage, MetalError> {
    let info = plan.info().ok_or(MetalError::InvalidPlan {
        reason: "metadata-only plan cannot produce a decoded image",
    })?;
    let decoded_region = plan.output_region().ok_or(MetalError::InvalidPlan {
        reason: "metadata-only plan omits its output region",
    })?;
    let samples = decoded_samples(plan.output().format, bytes)?;
    let decoded = DecodedImage {
        info: info.clone(),
        decoded_region,
        format: plan.output().format,
        planes: plan
            .output()
            .planes
            .iter()
            .map(|plane| PlaneDescriptor {
                byte_offset: plane.byte_offset,
                row_stride_bytes: plane.row_stride_bytes,
                width: plane.width,
                height: plane.height,
                channels: plane.channels,
            })
            .collect(),
        samples,
        report: plan.decode_report(true),
    };
    decoded
        .validate_layout()
        .map_err(|_| MetalError::InvalidPlan {
            reason: "Metal host result violates the decoded-image layout",
        })?;
    Ok(decoded)
}

#[cfg(target_os = "macos")]
fn decoded_samples(
    format: jxr_core::PixelFormat,
    bytes: Vec<u8>,
) -> Result<DecodedSamples, MetalError> {
    match format.storage_kind() {
        StorageKind::BitPacked => Ok(DecodedSamples::BitPacked(bytes)),
        StorageKind::U8 => Ok(DecodedSamples::U8(bytes)),
        StorageKind::U16 => Ok(DecodedSamples::U16(read_u16(&bytes)?)),
        StorageKind::I16 => Ok(DecodedSamples::I16(
            read_u16(&bytes)?
                .into_iter()
                .map(|value| i16::from_ne_bytes(value.to_ne_bytes()))
                .collect(),
        )),
        StorageKind::I32 => Ok(DecodedSamples::I32(
            read_u32(&bytes)?
                .into_iter()
                .map(|value| i32::from_ne_bytes(value.to_ne_bytes()))
                .collect(),
        )),
        StorageKind::F16Bits => Ok(DecodedSamples::F16(read_u16(&bytes)?)),
        StorageKind::F32 => Ok(DecodedSamples::F32(
            read_u32(&bytes)?.into_iter().map(f32::from_bits).collect(),
        )),
        StorageKind::PackedU16 => match format {
            jxr_core::PixelFormat::Rgb555 => Ok(DecodedSamples::Rgb555(read_u16(&bytes)?)),
            jxr_core::PixelFormat::Rgb565 => Ok(DecodedSamples::Rgb565(read_u16(&bytes)?)),
            _ => Err(MetalError::InvalidPlan {
                reason: "unknown packed 16-bit output",
            }),
        },
        StorageKind::PackedU32 => match format {
            jxr_core::PixelFormat::Rgb101010 => Ok(DecodedSamples::Rgb101010(read_u32(&bytes)?)),
            jxr_core::PixelFormat::Rgbe => Ok(DecodedSamples::Rgbe(read_u32(&bytes)?)),
            _ => Err(MetalError::InvalidPlan {
                reason: "unknown packed 32-bit output",
            }),
        },
    }
}

#[cfg(target_os = "macos")]
fn read_u16(bytes: &[u8]) -> Result<Vec<u16>, MetalError> {
    let chunks = bytes.chunks_exact(2);
    if !chunks.remainder().is_empty() {
        return Err(MetalError::InvalidPlan {
            reason: "16-bit host readback has a trailing byte",
        });
    }
    Ok(chunks
        .map(|chunk| u16::from_ne_bytes([chunk[0], chunk[1]]))
        .collect())
}

#[cfg(target_os = "macos")]
fn read_u32(bytes: &[u8]) -> Result<Vec<u32>, MetalError> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(MetalError::InvalidPlan {
            reason: "32-bit host readback has trailing bytes",
        });
    }
    Ok(chunks
        .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}
