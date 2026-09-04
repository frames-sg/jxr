// SPDX-License-Identifier: MIT OR Apache-2.0

use cudarc::driver::CudaSlice;

use crate::DenseCudaBatchLayout;

/// Immutable completed CUDA-resident JPEG XR output.
pub struct ResidentCudaImage {
    pub(crate) buffer: CudaSlice<u8>,
    layout: jxr_core::SurfaceLayout,
    report: jxr_core::DecodeReport,
}

impl core::fmt::Debug for ResidentCudaImage {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ResidentCudaImage")
            .field("device_ordinal", &self.buffer.ordinal())
            .field("layout", &self.layout)
            .field("report", &self.report)
            .finish_non_exhaustive()
    }
}

impl ResidentCudaImage {
    pub(crate) fn from_buffer(
        buffer: CudaSlice<u8>,
        layout: jxr_core::SurfaceLayout,
        report: jxr_core::DecodeReport,
    ) -> Self {
        Self {
            buffer,
            layout,
            report,
        }
    }

    /// Validated device surface layout.
    #[must_use]
    pub const fn layout(&self) -> &jxr_core::SurfaceLayout {
        &self.layout
    }

    /// Route report for the completed decode.
    #[must_use]
    pub const fn report(&self) -> &jxr_core::DecodeReport {
        &self.report
    }

    /// CUDA device ordinal owning the allocation.
    #[must_use]
    pub fn device_ordinal(&self) -> usize {
        self.buffer.ordinal()
    }

    /// Read-only access to the completed device allocation.
    #[must_use]
    pub const fn device_buffer(&self) -> &CudaSlice<u8> {
        &self.buffer
    }
}

/// Completed homogeneous batch in one CUDA allocation.
pub struct CudaResidentBatch {
    pub(crate) buffer: CudaSlice<u8>,
    layout: DenseCudaBatchLayout,
    reports: Vec<jxr_core::DecodeReport>,
}

impl core::fmt::Debug for CudaResidentBatch {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("CudaResidentBatch")
            .field("device_ordinal", &self.buffer.ordinal())
            .field("layout", &self.layout)
            .field("report_count", &self.reports.len())
            .finish_non_exhaustive()
    }
}

impl CudaResidentBatch {
    pub(crate) fn from_buffer(
        buffer: CudaSlice<u8>,
        layout: DenseCudaBatchLayout,
        reports: Vec<jxr_core::DecodeReport>,
    ) -> Self {
        Self {
            buffer,
            layout,
            reports,
        }
    }

    /// Validated dense layout.
    #[must_use]
    pub const fn layout(&self) -> &DenseCudaBatchLayout {
        &self.layout
    }

    /// Ordered decode reports.
    #[must_use]
    pub fn reports(&self) -> &[jxr_core::DecodeReport] {
        &self.reports
    }

    /// CUDA device ordinal owning the allocation.
    #[must_use]
    pub fn device_ordinal(&self) -> usize {
        self.buffer.ordinal()
    }

    /// Read-only access to the completed dense allocation.
    #[must_use]
    pub const fn device_buffer(&self) -> &CudaSlice<u8> {
        &self.buffer
    }
}
