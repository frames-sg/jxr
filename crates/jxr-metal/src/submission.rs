// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    DenseMetalBatchLayout, MetalBatchDestination, MetalDestination, MetalError, MetalResidentBatch,
    ResidentMetalImage,
};
#[cfg(target_os = "macos")]
use objc2::{rc::Retained, runtime::ProtocolObject};
#[cfg(target_os = "macos")]
use objc2_foundation::NSString;
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLDevice, MTLEvent, MTLResource};

/// Pending Metal reconstruction retaining its command buffer until completion.
#[derive(Debug)]
pub struct MetalSubmission {
    #[cfg(target_os = "macos")]
    pending: Option<PendingMetal>,
}

#[cfg(target_os = "macos")]
struct PendingMetal {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    output: Retained<ProtocolObject<dyn MTLBuffer>>,
    status: Retained<ProtocolObject<dyn MTLBuffer>>,
    status_offset: usize,
    layout: jxr_core::SurfaceLayout,
    report: jxr_core::DecodeReport,
    uploads: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    private_scratch: Vec<crate::buffer_pool::PooledBuffer>,
    shared_scratch: Vec<crate::buffer_pool::PooledBuffer>,
    buffer_pools: std::rc::Rc<crate::buffer_pool::MetalBufferPools>,
    completion_event: Retained<ProtocolObject<dyn MTLEvent>>,
}

#[cfg(target_os = "macos")]
impl core::fmt::Debug for PendingMetal {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("PendingMetal")
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

/// Pending ordered collection of Metal reconstruction submissions.
#[derive(Debug)]
pub struct MetalBatchSubmission {
    submissions: Vec<MetalSubmission>,
}

/// Pending homogeneous reconstruction into one internally owned allocation.
#[derive(Debug)]
pub struct MetalResidentBatchSubmission {
    submissions: Vec<MetalSubmission>,
    #[cfg(target_os = "macos")]
    output: Retained<ProtocolObject<dyn MTLBuffer>>,
    layout: DenseMetalBatchLayout,
    reports: Vec<jxr_core::DecodeReport>,
}

impl MetalResidentBatchSubmission {
    #[cfg(target_os = "macos")]
    pub(crate) fn new(
        submissions: Vec<MetalSubmission>,
        output: Retained<ProtocolObject<dyn MTLBuffer>>,
        layout: DenseMetalBatchLayout,
        reports: Vec<jxr_core::DecodeReport>,
    ) -> Self {
        Self {
            submissions,
            output,
            layout,
            reports,
        }
    }

    /// Dense layout available before completion.
    #[must_use]
    pub const fn layout(&self) -> &DenseMetalBatchLayout {
        &self.layout
    }

    /// Wait for every image and return the single completed allocation.
    pub fn wait(self) -> Result<MetalResidentBatch, MetalError> {
        for submission in self.submissions {
            drop(submission.wait()?);
        }
        #[cfg(target_os = "macos")]
        {
            Ok(MetalResidentBatch::from_buffer(
                self.output,
                self.layout,
                self.reports,
            ))
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(MetalError::Unavailable)
        }
    }
}

/// Committed GPU-side wait bridging a reconstruction to another queue.
#[cfg(target_os = "macos")]
pub struct MetalConsumerWait {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

#[cfg(target_os = "macos")]
impl core::fmt::Debug for MetalConsumerWait {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MetalConsumerWait")
            .finish_non_exhaustive()
    }
}

#[cfg(target_os = "macos")]
impl MetalConsumerWait {
    /// Wait on the host until the queue bridge itself has completed.
    pub fn wait(self) -> Result<(), MetalError> {
        j2k_metal_support::wait_for_completion(&self.command).map_err(MetalError::from)
    }
}

impl MetalBatchSubmission {
    pub(crate) fn new(submissions: Vec<MetalSubmission>) -> Self {
        Self { submissions }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.submissions.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.submissions.is_empty()
    }

    /// Wait in caller order and return every completed resident image.
    pub fn wait(self) -> Result<Vec<ResidentMetalImage>, MetalError> {
        let mut outputs = Vec::with_capacity(self.submissions.len());
        for submission in self.submissions {
            outputs.push(submission.wait()?);
        }
        Ok(outputs)
    }
}

/// Completion metadata for a caller-owned Metal destination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalDestinationCompletion {
    pub layout: jxr_core::SurfaceLayout,
    pub report: jxr_core::DecodeReport,
}

/// Pending decode retaining exclusive destination access through completion.
#[derive(Debug)]
pub struct MetalDestinationSubmission {
    submission: MetalSubmission,
    destination: MetalDestination,
    report: jxr_core::DecodeReport,
}

/// Completion metadata and retained storage for a dense caller-owned batch.
#[derive(Debug)]
pub struct MetalBatchDestinationCompletion {
    destination: MetalBatchDestination,
    reports: Vec<jxr_core::DecodeReport>,
}

impl MetalBatchDestinationCompletion {
    /// Validated dense batch layout.
    #[must_use]
    pub const fn layout(&self) -> &crate::DenseMetalBatchLayout {
        self.destination.layout()
    }

    /// Decode reports in tensor batch order.
    #[must_use]
    pub fn reports(&self) -> &[jxr_core::DecodeReport] {
        &self.reports
    }

    /// Retained destination device registry identifier.
    #[must_use]
    pub const fn device_registry_id(&self) -> u64 {
        self.destination.device_registry_id()
    }

    /// Consume the completion into its retained allocation and ordered reports.
    #[must_use]
    pub fn into_parts(self) -> (MetalBatchDestination, Vec<jxr_core::DecodeReport>) {
        (self.destination, self.reports)
    }
}

/// Pending dense decode retaining every resource and exclusive destination.
#[derive(Debug)]
pub struct MetalBatchDestinationSubmission {
    submissions: Vec<MetalSubmission>,
    destination: MetalBatchDestination,
    reports: Vec<jxr_core::DecodeReport>,
}

impl MetalBatchDestinationSubmission {
    pub(crate) fn new(
        submissions: Vec<MetalSubmission>,
        destination: MetalBatchDestination,
        reports: Vec<jxr_core::DecodeReport>,
    ) -> Self {
        Self {
            submissions,
            destination,
            reports,
        }
    }

    /// Wait for every image and return the still-retained dense allocation.
    pub fn wait(self) -> Result<MetalBatchDestinationCompletion, MetalError> {
        for submission in self.submissions {
            drop(submission.wait()?);
        }
        Ok(MetalBatchDestinationCompletion {
            destination: self.destination,
            reports: self.reports,
        })
    }
}

impl MetalDestinationSubmission {
    pub(crate) fn new(
        submission: MetalSubmission,
        destination: MetalDestination,
        report: jxr_core::DecodeReport,
    ) -> Self {
        Self {
            submission,
            destination,
            report,
        }
    }

    /// Wait for GPU completion and release exclusive destination ownership.
    pub fn wait(self) -> Result<MetalDestinationCompletion, MetalError> {
        drop(self.submission.wait()?);
        Ok(MetalDestinationCompletion {
            layout: self.destination.layout().clone(),
            report: self.report,
        })
    }
}

impl MetalSubmission {
    #[cfg(target_os = "macos")]
    pub(crate) fn submitted(
        encoded: crate::encode::EncodedMetalSubmission,
        report: jxr_core::DecodeReport,
    ) -> Self {
        Self {
            pending: Some(PendingMetal {
                command: encoded.command,
                output: encoded.output,
                status: encoded.status,
                status_offset: encoded.status_offset,
                layout: encoded.layout,
                report,
                uploads: encoded.uploads,
                private_scratch: encoded.private_scratch,
                shared_scratch: encoded.shared_scratch,
                buffer_pools: encoded.buffer_pools,
                completion_event: encoded.completion_event,
            }),
        }
    }

    /// Enqueue a GPU-side event wait on a consumer queue from the same device.
    ///
    /// Commands submitted later to `consumer_queue` may safely read the pending
    /// output. This establishes resource ordering only; [`Self::wait`] remains
    /// required to observe a shader arithmetic or command-buffer failure.
    #[cfg(target_os = "macos")]
    pub fn enqueue_consumer_wait(
        &self,
        consumer_queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<MetalConsumerWait, MetalError> {
        let pending = self
            .pending
            .as_ref()
            .ok_or(MetalError::InvalidSubmissionState {
                expected: "submitted",
                actual: "completed",
            })?;
        if pending.output.device().registryID() != consumer_queue.device().registryID() {
            return Err(MetalError::InvalidDestination {
                reason: "consumer queue belongs to a different Metal device",
            });
        }
        let command = j2k_metal_support::checked_command_buffer(consumer_queue)?;
        command.setLabel(Some(&NSString::from_str("JXR cross-queue consumer wait")));
        command.encodeWaitForEvent_value(&pending.completion_event, 1);
        command.commit();
        Ok(MetalConsumerWait { command })
    }

    /// Planned layout of the pending device output.
    #[cfg(target_os = "macos")]
    pub fn output_layout(&self) -> Result<&jxr_core::SurfaceLayout, MetalError> {
        self.pending.as_ref().map(|pending| &pending.layout).ok_or(
            MetalError::InvalidSubmissionState {
                expected: "submitted",
                actual: "completed",
            },
        )
    }

    /// Borrow the pending output for an event-ordered Metal consumer.
    ///
    /// # Safety
    ///
    /// The consumer must call [`Self::enqueue_consumer_wait`] on its exact
    /// queue before submitting any command that reads this buffer. It must not
    /// write the buffer, and the submission must remain alive until the
    /// consumer command has retained the resource.
    #[cfg(target_os = "macos")]
    pub unsafe fn raw_output_buffer(&self) -> Result<&ProtocolObject<dyn MTLBuffer>, MetalError> {
        self.pending
            .as_ref()
            .map(|pending| pending.output.as_ref())
            .ok_or(MetalError::InvalidSubmissionState {
                expected: "submitted",
                actual: "completed",
            })
    }

    /// Wait for completion and return the immutable device output.
    pub fn wait(mut self) -> Result<ResidentMetalImage, MetalError> {
        #[cfg(target_os = "macos")]
        {
            let mut pending = self
                .pending
                .take()
                .ok_or(MetalError::InvalidSubmissionState {
                    expected: "submitted",
                    actual: "completed",
                })?;
            j2k_metal_support::wait_for_completion(&pending.command)?;
            // SAFETY: Waiting the owning submission establishes completion of
            // every shader write to this shared status allocation.
            let status = unsafe {
                j2k_metal_support::checked_buffer_read::<u32>(
                    &pending.status,
                    pending.status_offset,
                )
            }?;
            recycle_scratch(&mut pending)?;
            if status != 0 {
                return Err(MetalError::KernelArithmetic { status });
            }
            Ok(ResidentMetalImage::from_buffer(
                pending.output,
                pending.layout,
                pending.report,
            ))
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(MetalError::Unavailable)
        }
    }
}

impl Drop for MetalSubmission {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        if let Some(mut pending) = self.pending.take() {
            let _ = j2k_metal_support::wait_for_completion(&pending.command);
            let _ = recycle_scratch(&mut pending);
        }
    }
}

#[cfg(target_os = "macos")]
fn recycle_scratch(pending: &mut PendingMetal) -> Result<(), MetalError> {
    // Keep all uploaded coefficient buffers alive through the completion wait.
    let _ = pending.uploads.len();
    for buffer in pending.private_scratch.drain(..) {
        pending.buffer_pools.recycle_private(buffer)?;
    }
    for buffer in pending.shared_scratch.drain(..) {
        pending.buffer_pools.recycle_shared(buffer)?;
    }
    Ok(())
}
