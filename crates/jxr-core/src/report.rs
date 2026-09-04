//! Observable routing decisions for hybrid decode.

use alloc::vec::Vec;

use crate::{BackendKind, BackendRequest};

/// Ordered decode and reconstruction phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecodeStage {
    Parse,
    EntropyDecode,
    InverseScan,
    CoefficientRemap,
    DcLowPassPrediction,
    DequantizeAndFirstInverseTransform,
    FirstOverlap,
    HighPassPrediction,
    SecondInverseTransform,
    SecondOverlap,
    ChromaReconstruction,
    ColorAndAlphaConversion,
    CropClipAndPack,
    HostReadback,
}

/// Concrete implementation which owned a stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StageExecutor {
    CpuScalar,
    CpuSimd,
    Metal,
    Cuda,
}

impl StageExecutor {
    #[must_use]
    pub const fn backend(self) -> BackendKind {
        match self {
            Self::CpuScalar | Self::CpuSimd => BackendKind::Cpu,
            Self::Metal => BackendKind::Metal,
            Self::Cuda => BackendKind::Cuda,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StageReport {
    pub stage: DecodeStage,
    pub executor: StageExecutor,
}

/// Why an automatic request selected CPU before device submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FallbackReason {
    WorkloadBelowThreshold,
    BackendNotCompiled,
    DeviceUnavailable,
    UnsupportedFormat,
    PipelineIncomplete,
    ResourceLimit,
}

/// Requested route, selected route, and per-stage ownership.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeReport {
    pub requested: BackendRequest,
    pub selected: BackendKind,
    pub fallback: Option<FallbackReason>,
    pub stages: Vec<StageReport>,
}

impl DecodeReport {
    #[must_use]
    pub fn cpu(requested: BackendRequest) -> Self {
        Self {
            requested,
            selected: BackendKind::Cpu,
            fallback: None,
            stages: Vec::new(),
        }
    }

    #[must_use]
    pub fn executor_for(&self, stage: DecodeStage) -> Option<StageExecutor> {
        self.stages
            .iter()
            .find(|entry| entry.stage == stage)
            .map(|entry| entry.executor)
    }
}
