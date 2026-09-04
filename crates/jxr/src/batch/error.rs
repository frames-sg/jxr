use jxr_core::JxrError;

/// Decode phase associated with one input-local batch failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BatchErrorStage {
    /// Parsing, request planning, or output-contract validation.
    Preparation,
    /// Entropy decoding, reconstruction, or output packing.
    Decode,
}

/// Failure affecting batch infrastructure rather than one encoded image.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum BatchInfrastructureError {
    /// The caller submitted more inputs than the configured batch ceiling.
    #[error("JPEG XR batch contains {requested} inputs; configured maximum is {maximum}")]
    TooManyInputs {
        /// Submitted input count.
        requested: usize,
        /// Configured input ceiling.
        maximum: usize,
    },
    /// Aggregate dense output exceeds the configured batch allocation ceiling.
    #[error("JPEG XR batch output requires {requested} bytes; configured maximum is {maximum}")]
    OutputAllocationTooLarge {
        /// Required dense output bytes.
        requested: u64,
        /// Configured aggregate ceiling.
        maximum: u64,
    },
    /// A checked host allocation could not be reserved.
    #[error("failed to reserve {requested} bytes for {what}")]
    HostAllocationFailed {
        /// Affected batch owner.
        what: &'static str,
        /// Requested allocation size.
        requested: usize,
    },
    /// The retained Rayon worker pool could not be created.
    #[error("failed to initialize JPEG XR batch workers: {message}")]
    WorkerInitialization {
        /// Rayon initialization diagnostic.
        message: String,
    },
    /// A backend cannot represent the requested dense tensor layout.
    #[error("JPEG XR {backend} batch does not support {layout:?} output layout")]
    UnsupportedBatchLayout {
        /// Backend rejecting the layout.
        backend: &'static str,
        /// Requested layout.
        layout: super::BatchLayout,
    },
    /// Caller-owned destination type or length differs from the prepared group.
    #[error("invalid JPEG XR CPU batch destination: {reason}")]
    InvalidDestination {
        /// Stable validation diagnostic.
        reason: &'static str,
    },
}

/// One input-local batch failure preserving original caller order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedBatchError {
    index: usize,
    stage: BatchErrorStage,
    source: JxrError,
}

impl IndexedBatchError {
    pub(crate) const fn new(index: usize, stage: BatchErrorStage, source: JxrError) -> Self {
        Self {
            index,
            stage,
            source,
        }
    }

    /// Original position in the submitted input collection.
    #[must_use]
    pub const fn index(&self) -> usize {
        self.index
    }

    /// Batch phase that rejected this input.
    #[must_use]
    pub const fn stage(&self) -> BatchErrorStage {
        self.stage
    }

    /// Stable JPEG XR failure for this input.
    #[must_use]
    pub const fn source(&self) -> &JxrError {
        &self.source
    }
}
