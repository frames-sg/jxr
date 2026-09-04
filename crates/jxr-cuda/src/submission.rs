// SPDX-License-Identifier: MIT OR Apache-2.0

use cudarc::driver::{CudaSlice, CudaStream};

use crate::{
    CudaBatchDestination, CudaDestination, CudaError, CudaResidentBatch, DenseCudaBatchLayout,
    ResidentCudaImage,
};

/// Pending CUDA reconstruction retaining all device resources through completion.
pub struct CudaSubmission {
    pending: Option<PendingCuda>,
    report: jxr_core::DecodeReport,
}

struct PendingCuda {
    encoded: crate::encode::EncodedCudaSubmission,
}

impl core::fmt::Debug for CudaSubmission {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("CudaSubmission")
            .field("submitted", &self.pending.is_some())
            .field("report", &self.report)
            .finish_non_exhaustive()
    }
}

impl CudaSubmission {
    pub(crate) fn submitted(
        encoded: crate::encode::EncodedCudaSubmission,
        report: jxr_core::DecodeReport,
    ) -> Self {
        Self {
            pending: Some(PendingCuda { encoded }),
            report,
        }
    }

    /// Whether the recorded completion event has fired.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.pending
            .as_ref()
            .is_none_or(|pending| pending.encoded.completion.is_complete())
    }

    /// Enqueue a device-side wait on another stream in the same CUDA context.
    pub fn enqueue_consumer_wait(&self, stream: &CudaStream) -> Result<(), CudaError> {
        let pending = self
            .pending
            .as_ref()
            .ok_or(CudaError::InvalidSubmissionState {
                expected: "submitted",
                actual: "completed",
            })?;
        if pending.encoded.stream.context() != stream.context() {
            return Err(CudaError::InvalidDestination {
                reason: "consumer stream belongs to a different CUDA context",
            });
        }
        stream.wait(&pending.encoded.completion)?;
        Ok(())
    }

    /// Planned layout of a pending output.
    pub fn output_layout(&self) -> Result<&jxr_core::SurfaceLayout, CudaError> {
        self.pending
            .as_ref()
            .map(|pending| &pending.encoded.layout)
            .ok_or(CudaError::InvalidSubmissionState {
                expected: "submitted",
                actual: "completed",
            })
    }

    /// Borrow the pending output for an event-ordered CUDA consumer.
    ///
    /// # Safety
    ///
    /// The consumer stream must first call [`Self::enqueue_consumer_wait`], may
    /// only read this allocation, and must retain its own use before this
    /// submission is dropped.
    pub unsafe fn pending_device_buffer(&self) -> Result<&CudaSlice<u8>, CudaError> {
        self.pending
            .as_ref()
            .and_then(|pending| pending.encoded.output.as_ref())
            .ok_or(CudaError::InvalidSubmissionState {
                expected: "submitted owned output",
                actual: "completed or caller-owned output",
            })
    }

    /// Wait for completion and return immutable device-resident output.
    pub fn wait(mut self) -> Result<ResidentCudaImage, CudaError> {
        let mut pending = self
            .pending
            .take()
            .ok_or(CudaError::InvalidSubmissionState {
                expected: "submitted",
                actual: "completed",
            })?;
        complete(&mut pending)?;
        let output = pending
            .encoded
            .output
            .take()
            .ok_or(CudaError::InvalidSubmissionState {
                expected: "owned output",
                actual: "caller-owned output",
            })?;
        Ok(ResidentCudaImage::from_buffer(
            output,
            pending.encoded.layout.clone(),
            self.report.clone(),
        ))
    }

    pub(crate) fn wait_completion(mut self) -> Result<(), CudaError> {
        let mut pending = self
            .pending
            .take()
            .ok_or(CudaError::InvalidSubmissionState {
                expected: "submitted",
                actual: "completed",
            })?;
        complete(&mut pending)
    }
}

impl Drop for CudaSubmission {
    fn drop(&mut self) {
        if let Some(mut pending) = self.pending.take() {
            let _ = pending.encoded.completion.synchronize();
            let _ = recycle_scratch(&mut pending);
        }
    }
}

fn complete(pending: &mut PendingCuda) -> Result<(), CudaError> {
    pending.encoded.completion.synchronize()?;
    let mut status = [0_u32; 1];
    pending
        .encoded
        .stream
        .memcpy_dtoh(&pending.encoded.status, &mut status)?;
    pending.encoded.stream.synchronize()?;
    recycle_scratch(pending)?;
    if status[0] != 0 {
        return Err(CudaError::KernelArithmetic { status: status[0] });
    }
    Ok(())
}

fn recycle_scratch(pending: &mut PendingCuda) -> Result<(), CudaError> {
    for buffer in pending.encoded.scratch.drain(..) {
        pending.encoded.runtime.buffer_pool.recycle(buffer)?;
    }
    Ok(())
}

/// Pending ordered collection of CUDA reconstruction submissions.
#[derive(Debug)]
pub struct CudaBatchSubmission {
    submissions: Vec<CudaSubmission>,
}

impl CudaBatchSubmission {
    pub(crate) fn new(submissions: Vec<CudaSubmission>) -> Self {
        Self { submissions }
    }

    /// Number of submitted images.
    #[must_use]
    pub fn len(&self) -> usize {
        self.submissions.len()
    }

    /// Whether no images were submitted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.submissions.is_empty()
    }

    /// Wait in caller order and return every completed resident image.
    pub fn wait(self) -> Result<Vec<ResidentCudaImage>, CudaError> {
        self.submissions
            .into_iter()
            .map(CudaSubmission::wait)
            .collect()
    }
}

/// Pending homogeneous reconstruction into one internally owned allocation.
#[derive(Debug)]
pub struct CudaResidentBatchSubmission {
    submissions: Vec<CudaSubmission>,
    output: CudaSlice<u8>,
    layout: DenseCudaBatchLayout,
    reports: Vec<jxr_core::DecodeReport>,
}

impl CudaResidentBatchSubmission {
    pub(crate) fn new(
        submissions: Vec<CudaSubmission>,
        output: CudaSlice<u8>,
        layout: DenseCudaBatchLayout,
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
    pub const fn layout(&self) -> &DenseCudaBatchLayout {
        &self.layout
    }

    /// Wait for every image and return the single completed allocation.
    pub fn wait(self) -> Result<CudaResidentBatch, CudaError> {
        for submission in self.submissions {
            submission.wait_completion()?;
        }
        Ok(CudaResidentBatch::from_buffer(
            self.output,
            self.layout,
            self.reports,
        ))
    }
}

/// Completion metadata for a caller-owned CUDA destination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaDestinationCompletion {
    /// Validated output layout.
    pub layout: jxr_core::SurfaceLayout,
    /// Route and stage report.
    pub report: jxr_core::DecodeReport,
}

/// Pending decode retaining exclusive destination access through completion.
#[derive(Debug)]
pub struct CudaDestinationSubmission {
    submission: CudaSubmission,
    destination: CudaDestination,
    report: jxr_core::DecodeReport,
}

impl CudaDestinationSubmission {
    pub(crate) fn new(
        submission: CudaSubmission,
        destination: CudaDestination,
        report: jxr_core::DecodeReport,
    ) -> Self {
        Self {
            submission,
            destination,
            report,
        }
    }

    /// Wait for GPU completion and release exclusive destination ownership.
    pub fn wait(self) -> Result<(CudaDestinationCompletion, CudaDestination), CudaError> {
        self.submission.wait_completion()?;
        let completion = CudaDestinationCompletion {
            layout: self.destination.layout().clone(),
            report: self.report,
        };
        Ok((completion, self.destination))
    }
}

/// Completion metadata and retained storage for a caller-owned dense batch.
#[derive(Debug)]
pub struct CudaBatchDestinationCompletion {
    destination: CudaBatchDestination,
    reports: Vec<jxr_core::DecodeReport>,
}

impl CudaBatchDestinationCompletion {
    /// Validated dense batch layout.
    #[must_use]
    pub const fn layout(&self) -> &DenseCudaBatchLayout {
        self.destination.layout()
    }

    /// Decode reports in dense batch order.
    #[must_use]
    pub fn reports(&self) -> &[jxr_core::DecodeReport] {
        &self.reports
    }

    /// Consume the completion into its allocation and ordered reports.
    #[must_use]
    pub fn into_parts(self) -> (CudaBatchDestination, Vec<jxr_core::DecodeReport>) {
        (self.destination, self.reports)
    }
}

/// Pending dense decode retaining every resource and exclusive destination.
#[derive(Debug)]
pub struct CudaBatchDestinationSubmission {
    submissions: Vec<CudaSubmission>,
    destination: CudaBatchDestination,
    reports: Vec<jxr_core::DecodeReport>,
}

impl CudaBatchDestinationSubmission {
    pub(crate) fn new(
        submissions: Vec<CudaSubmission>,
        destination: CudaBatchDestination,
        reports: Vec<jxr_core::DecodeReport>,
    ) -> Self {
        Self {
            submissions,
            destination,
            reports,
        }
    }

    /// Wait for every image and return the still-retained dense allocation.
    pub fn wait(self) -> Result<CudaBatchDestinationCompletion, CudaError> {
        for submission in self.submissions {
            submission.wait_completion()?;
        }
        Ok(CudaBatchDestinationCompletion {
            destination: self.destination,
            reports: self.reports,
        })
    }
}
