// SPDX-License-Identifier: MIT OR Apache-2.0

use jxr::{PreparedReconstruction, Rect};

use crate::{Error, MpsGraphTensorSpec};

/// One input-local preparation failure preserving its original source index.
#[derive(Debug, Clone)]
pub struct IndexedPreparationError {
    source_index: usize,
    error: std::sync::Arc<Error>,
}

impl IndexedPreparationError {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    pub(crate) fn new(source_index: usize, error: Error) -> Self {
        Self {
            source_index,
            error: std::sync::Arc::new(error),
        }
    }

    #[must_use]
    pub const fn source_index(&self) -> usize {
        self.source_index
    }

    #[must_use]
    pub fn error(&self) -> &Error {
        self.error.as_ref()
    }
}

/// One homogeneous group execution failure preserving all affected sources.
#[derive(Debug)]
pub struct IndexedGroupError {
    source_indices: Vec<usize>,
    error: Error,
}

impl IndexedGroupError {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    pub(crate) const fn new(source_indices: Vec<usize>, error: Error) -> Self {
        Self {
            source_indices,
            error,
        }
    }

    #[must_use]
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }

    #[must_use]
    pub const fn error(&self) -> &Error {
        &self.error
    }
}

#[cfg_attr(
    not(all(target_arch = "aarch64", target_os = "macos")),
    expect(
        dead_code,
        reason = "opaque prepared storage is unavailable on this target"
    )
)]
pub(crate) struct PreparedImage {
    pub(crate) source_index: usize,
    pub(crate) reconstruction: PreparedReconstruction,
    pub(crate) decoded_region: Rect,
}

/// One deterministic homogeneous `[H, W, C, element type]` group.
pub struct MpsGraphPreparedGroup {
    pub(crate) images: Vec<PreparedImage>,
    pub(crate) spec: MpsGraphTensorSpec,
}

impl core::fmt::Debug for MpsGraphPreparedGroup {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MpsGraphPreparedGroup")
            .field("spec", &self.spec)
            .field("source_indices", &self.source_indices())
            .finish_non_exhaustive()
    }
}

impl MpsGraphPreparedGroup {
    #[must_use]
    pub const fn spec(&self) -> MpsGraphTensorSpec {
        self.spec
    }

    #[must_use]
    pub fn source_indices(&self) -> Vec<usize> {
        self.images.iter().map(|image| image.source_index).collect()
    }
}

/// Reusable homogeneous groups plus indexed failures from valid sibling inputs.
#[derive(Debug)]
pub struct MpsGraphPreparedBatch {
    pub(crate) groups: Vec<MpsGraphPreparedGroup>,
    pub(crate) errors: Vec<IndexedPreparationError>,
}

impl MpsGraphPreparedBatch {
    #[must_use]
    pub fn groups(&self) -> &[MpsGraphPreparedGroup] {
        &self.groups
    }

    #[must_use]
    pub fn errors(&self) -> &[IndexedPreparationError] {
        &self.errors
    }
}
