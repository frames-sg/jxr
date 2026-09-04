// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use jxr_core::{DecodedImage, DecodedSamples, PlaneDescriptor, StorageKind};

use crate::{
    CudaBatchDestination, CudaBatchDestinationSubmission, CudaBatchSubmission, CudaDecodePlan,
    CudaDestination, CudaDestinationSubmission, CudaError, CudaResidentBatch,
    CudaResidentBatchSubmission, CudaSubmission, DenseCudaBatchLayout, ResidentCudaImage,
    runtime::{BATCH_SCRATCH_BUDGET, CudaRuntime},
};

/// Reusable CUDA runtime session with cached kernels, uploads, streams, and scratch.
#[derive(Clone)]
pub struct CudaDecoderSession {
    runtime: Arc<CudaRuntime>,
}

impl core::fmt::Debug for CudaDecoderSession {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("CudaDecoderSession")
            .field("device_ordinal", &self.runtime.context.ordinal())
            .field("stream_count", &self.runtime.streams.len())
            .finish_non_exhaustive()
    }
}

impl CudaDecoderSession {
    /// Probe for both dynamically loaded CUDA driver and NVRTC libraries.
    ///
    /// This performs no CUDA initialization and is safe on CPU-only hosts.
    #[must_use]
    pub fn is_available() -> bool {
        // SAFETY: These cudarc probes only attempt to open candidate shared
        // libraries and do not retain or dereference foreign handles.
        unsafe { cudarc::driver::sys::is_culib_present() && cudarc::nvrtc::sys::is_culib_present() }
    }

    /// Create a reusable session on device ordinal zero.
    pub fn system_default() -> Result<Self, CudaError> {
        Self::new(0)
    }

    /// Create a reusable session on a caller-selected CUDA device ordinal.
    pub fn new(device_ordinal: usize) -> Result<Self, CudaError> {
        if !Self::is_available() {
            return Err(CudaError::Unavailable);
        }
        Ok(Self {
            runtime: CudaRuntime::build(device_ordinal)?,
        })
    }

    /// Whether this initialized session can accept reconstruction work.
    #[must_use]
    pub const fn is_usable(&self) -> bool {
        true
    }

    /// Device ordinal retained by this session.
    #[must_use]
    pub fn device_ordinal(&self) -> usize {
        self.runtime.context.ordinal()
    }

    /// Number of reusable nonblocking streams available to batch work.
    #[must_use]
    pub fn stream_count(&self) -> usize {
        self.runtime.streams.len()
    }

    /// Return retained exact-size scratch allocation counters.
    pub fn buffer_pool_diagnostics(&self) -> Result<crate::CudaBufferPoolDiagnostics, CudaError> {
        self.runtime.buffer_pool.diagnostics()
    }

    /// Return immutable coefficient upload-cache counters.
    pub fn upload_cache_diagnostics(&self) -> Result<crate::CudaUploadCacheDiagnostics, CudaError> {
        self.runtime.upload_cache.diagnostics()
    }

    /// Allocate an exact device output owned by the caller.
    pub fn allocate_destination(
        &self,
        layout: jxr_core::SurfaceLayout,
    ) -> Result<CudaDestination, CudaError> {
        let buffer = self.runtime.stream(0).alloc_zeros(layout.byte_len)?;
        CudaDestination::from_device_buffer(buffer, layout)
    }

    /// Allocate a dense device batch output owned by the caller.
    pub fn allocate_batch_destination(
        &self,
        layout: DenseCudaBatchLayout,
    ) -> Result<CudaBatchDestination, CudaError> {
        let buffer = self.runtime.stream(0).alloc_zeros(layout.byte_len())?;
        CudaBatchDestination::from_device_buffer(buffer, layout)
    }

    /// Submit one validated plan without waiting for completion.
    pub fn submit(&self, plan: &CudaDecodePlan) -> Result<CudaSubmission, CudaError> {
        let encoded = crate::encode::encode_owned(&self.runtime, plan, 0)?;
        Ok(CudaSubmission::submitted(
            encoded,
            plan.decode_report(false),
        ))
    }

    /// Submit a bounded batch across the session's reusable streams.
    pub fn submit_batch(&self, plans: &[CudaDecodePlan]) -> Result<CudaBatchSubmission, CudaError> {
        check_batch_scratch(plans)?;
        let mut submissions = Vec::with_capacity(plans.len());
        for (index, plan) in plans.iter().enumerate() {
            let encoded = crate::encode::encode_owned(&self.runtime, plan, index)?;
            submissions.push(CudaSubmission::submitted(
                encoded,
                plan.decode_report(false),
            ));
        }
        Ok(CudaBatchSubmission::new(submissions))
    }

    /// Submit a homogeneous batch into one internally owned device allocation.
    pub fn submit_dense_batch(
        &self,
        plans: &[CudaDecodePlan],
    ) -> Result<CudaResidentBatchSubmission, CudaError> {
        let first = plans.first().ok_or(CudaError::InvalidPlan {
            reason: "dense CUDA batch cannot be empty",
        })?;
        if plans.iter().any(|plan| plan.output() != first.output()) {
            return Err(CudaError::InvalidPlan {
                reason: "dense CUDA batch plans are not homogeneous",
            });
        }
        check_batch_scratch(plans)?;
        let layout = DenseCudaBatchLayout::new(first.output().clone(), plans.len())?;
        let stream = self.runtime.stream(0);
        let mut output = stream.alloc_zeros::<u8>(layout.byte_len())?;
        let mut submissions = Vec::with_capacity(plans.len());
        let mut reports = Vec::with_capacity(plans.len());
        for (image, plan) in plans.iter().enumerate() {
            let encoded = crate::encode::encode_into(
                &self.runtime,
                plan,
                &mut output,
                layout.image_offset(image)?,
                0,
            )?;
            let report = plan.decode_report(false);
            submissions.push(CudaSubmission::submitted(encoded, report.clone()));
            reports.push(report);
        }
        Ok(CudaResidentBatchSubmission::new(
            submissions,
            output,
            layout,
            reports,
        ))
    }

    /// Submit into an exclusively retained caller-owned device allocation.
    pub fn submit_into(
        &self,
        plan: &CudaDecodePlan,
        mut destination: CudaDestination,
    ) -> Result<CudaDestinationSubmission, CudaError> {
        destination.validate_plan(plan)?;
        destination.validate_context(&self.runtime.context)?;
        let encoded =
            crate::encode::encode_into(&self.runtime, plan, destination.buffer_mut(), 0, 0)?;
        let report = plan.decode_report(false);
        Ok(CudaDestinationSubmission::new(
            CudaSubmission::submitted(encoded, report.clone()),
            destination,
            report,
        ))
    }

    /// Submit a homogeneous batch into caller-owned dense device storage.
    pub fn submit_batch_into(
        &self,
        plans: &[CudaDecodePlan],
        mut destination: CudaBatchDestination,
    ) -> Result<CudaBatchDestinationSubmission, CudaError> {
        destination.validate_plans(plans)?;
        destination.validate_context(&self.runtime.context)?;
        check_batch_scratch(plans)?;
        let mut submissions = Vec::with_capacity(plans.len());
        let mut reports = Vec::with_capacity(plans.len());
        for (image, plan) in plans.iter().enumerate() {
            let offset = destination.layout().image_offset(image)?;
            let encoded = crate::encode::encode_into(
                &self.runtime,
                plan,
                destination.buffer_mut(),
                offset,
                0,
            )?;
            let report = plan.decode_report(false);
            submissions.push(CudaSubmission::submitted(encoded, report.clone()));
            reports.push(report);
        }
        Ok(CudaBatchDestinationSubmission::new(
            submissions,
            destination,
            reports,
        ))
    }

    /// Decode one plan and copy its exact native bytes to typed host storage.
    pub fn decode_to_host(&self, plan: &CudaDecodePlan) -> Result<DecodedImage, CudaError> {
        let resident = self.submit(plan)?.wait()?;
        let bytes = self.readback(&resident)?;
        decoded_image(plan, bytes)
    }

    /// Decode a batch to host storage while preserving caller order.
    pub fn decode_batch_to_host(
        &self,
        plans: &[CudaDecodePlan],
    ) -> Result<Vec<DecodedImage>, CudaError> {
        self.submit_batch(plans)?
            .wait()?
            .into_iter()
            .zip(plans)
            .map(|(resident, plan)| {
                let bytes = self.readback(&resident)?;
                decoded_image(plan, bytes)
            })
            .collect()
    }

    /// Copy one completed resident image to host bytes.
    pub fn readback(&self, image: &ResidentCudaImage) -> Result<Vec<u8>, CudaError> {
        validate_image_context(image, &self.runtime)?;
        let stream = image.buffer.stream();
        let bytes = stream.clone_dtoh(&image.buffer)?;
        stream.synchronize()?;
        Ok(bytes)
    }

    /// Copy one image from a completed dense batch to host bytes.
    pub fn readback_batch_image(
        &self,
        batch: &CudaResidentBatch,
        image: usize,
    ) -> Result<Vec<u8>, CudaError> {
        if batch.buffer.context() != &self.runtime.context {
            return Err(CudaError::InvalidDestination {
                reason: "resident CUDA batch belongs to a different context",
            });
        }
        let start = batch.layout().image_offset(image)?;
        let end = start
            .checked_add(batch.layout().image_stride_bytes())
            .ok_or(CudaError::InvalidDestination {
                reason: "resident CUDA batch image range overflows usize",
            })?;
        let view = batch.buffer.slice(start..end);
        let stream = batch.buffer.stream();
        let bytes = stream.clone_dtoh(&view)?;
        stream.synchronize()?;
        Ok(bytes)
    }
}

fn validate_image_context(
    image: &ResidentCudaImage,
    runtime: &CudaRuntime,
) -> Result<(), CudaError> {
    if image.buffer.context() == &runtime.context {
        Ok(())
    } else {
        Err(CudaError::InvalidDestination {
            reason: "resident CUDA image belongs to a different context",
        })
    }
}

fn check_batch_scratch(plans: &[CudaDecodePlan]) -> Result<(), CudaError> {
    let total = plans.iter().try_fold(0_usize, |total, plan| {
        total
            .checked_add(plan.scratch_bytes()?)
            .ok_or(CudaError::ResourceLimit {
                reason: "aggregate CUDA batch scratch overflows usize",
                requested: usize::MAX,
                maximum: BATCH_SCRATCH_BUDGET,
            })
    })?;
    if total > BATCH_SCRATCH_BUDGET {
        return Err(CudaError::ResourceLimit {
            reason: "aggregate CUDA batch scratch exceeds the bounded session budget",
            requested: total,
            maximum: BATCH_SCRATCH_BUDGET,
        });
    }
    Ok(())
}

fn decoded_image(plan: &CudaDecodePlan, bytes: Vec<u8>) -> Result<DecodedImage, CudaError> {
    let info = plan.info().ok_or(CudaError::InvalidPlan {
        reason: "metadata-only plan cannot produce a decoded image",
    })?;
    let decoded_region = plan.output_region().ok_or(CudaError::InvalidPlan {
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
        .map_err(|_| CudaError::InvalidPlan {
            reason: "CUDA host result violates the decoded-image layout",
        })?;
    Ok(decoded)
}

fn decoded_samples(
    format: jxr_core::PixelFormat,
    bytes: Vec<u8>,
) -> Result<DecodedSamples, CudaError> {
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
            _ => Err(CudaError::InvalidPlan {
                reason: "unknown packed 16-bit output",
            }),
        },
        StorageKind::PackedU32 => match format {
            jxr_core::PixelFormat::Rgb101010 => Ok(DecodedSamples::Rgb101010(read_u32(&bytes)?)),
            jxr_core::PixelFormat::Rgbe => Ok(DecodedSamples::Rgbe(read_u32(&bytes)?)),
            _ => Err(CudaError::InvalidPlan {
                reason: "unknown packed 32-bit output",
            }),
        },
    }
}

fn read_u16(bytes: &[u8]) -> Result<Vec<u16>, CudaError> {
    let chunks = bytes.chunks_exact(2);
    if !chunks.remainder().is_empty() {
        return Err(CudaError::InvalidPlan {
            reason: "16-bit CUDA host readback has a trailing byte",
        });
    }
    Ok(chunks
        .map(|chunk| u16::from_ne_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn read_u32(bytes: &[u8]) -> Result<Vec<u32>, CudaError> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(CudaError::InvalidPlan {
            reason: "32-bit CUDA host readback has trailing bytes",
        });
    }
    Ok(chunks
        .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}
