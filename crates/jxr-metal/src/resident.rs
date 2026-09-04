// SPDX-License-Identifier: MIT OR Apache-2.0

use jxr_core::{DecodeReport, SurfaceLayout};

#[cfg(target_os = "macos")]
use objc2::{rc::Retained, runtime::ProtocolObject};
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBuffer, MTLDevice, MTLResource};

/// Completed immutable Metal-resident JPEG XR output.
pub struct ResidentMetalImage {
    #[cfg(target_os = "macos")]
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    layout: SurfaceLayout,
    report: DecodeReport,
    device_registry_id: u64,
}

/// Completed homogeneous JPEG XR batch retained in one private Metal allocation.
pub struct MetalResidentBatch {
    #[cfg(target_os = "macos")]
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    layout: crate::DenseMetalBatchLayout,
    reports: Vec<DecodeReport>,
    device_registry_id: u64,
}

impl core::fmt::Debug for MetalResidentBatch {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MetalResidentBatch")
            .field("layout", &self.layout)
            .field("device_registry_id", &self.device_registry_id)
            .finish_non_exhaustive()
    }
}

impl MetalResidentBatch {
    #[cfg(target_os = "macos")]
    pub(crate) fn from_buffer(
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        layout: crate::DenseMetalBatchLayout,
        reports: Vec<DecodeReport>,
    ) -> Self {
        let device_registry_id = buffer.device().registryID();
        Self {
            buffer,
            layout,
            reports,
            device_registry_id,
        }
    }

    /// Dense batch and per-image layout.
    #[must_use]
    pub const fn layout(&self) -> &crate::DenseMetalBatchLayout {
        &self.layout
    }

    /// Decode reports in dense image order.
    #[must_use]
    pub fn reports(&self) -> &[DecodeReport] {
        &self.reports
    }

    /// Registry identifier of the allocation's Metal device.
    #[must_use]
    pub const fn device_registry_id(&self) -> u64 {
        self.device_registry_id
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn validate_device(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<(), crate::MetalError> {
        if self.device_registry_id != device.registryID() {
            return Err(crate::MetalError::InvalidDestination {
                reason: "resident batch belongs to a different Metal device",
            });
        }
        Ok(())
    }

    /// Borrow the completed private batch for an audited consumer.
    ///
    /// # Safety
    ///
    /// The caller must not mutate the allocation while another consumer may
    /// read it and must establish Metal ordering before cross-queue use.
    #[cfg(target_os = "macos")]
    pub unsafe fn raw_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }
}

impl core::fmt::Debug for ResidentMetalImage {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ResidentMetalImage")
            .field("layout", &self.layout)
            .field("device_registry_id", &self.device_registry_id)
            .finish_non_exhaustive()
    }
}

impl ResidentMetalImage {
    #[cfg(target_os = "macos")]
    pub(crate) fn from_buffer(
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        layout: SurfaceLayout,
        report: DecodeReport,
    ) -> Self {
        let device_registry_id = buffer.device().registryID();
        Self {
            buffer,
            layout,
            report,
            device_registry_id,
        }
    }

    #[must_use]
    pub const fn layout(&self) -> &SurfaceLayout {
        &self.layout
    }

    #[must_use]
    pub const fn report(&self) -> &DecodeReport {
        &self.report
    }

    #[must_use]
    pub const fn device_registry_id(&self) -> u64 {
        self.device_registry_id
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn validate_device(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<(), crate::MetalError> {
        if self.device_registry_id != device.registryID() {
            return Err(crate::MetalError::InvalidDestination {
                reason: "resident image belongs to a different Metal device",
            });
        }
        Ok(())
    }

    /// Borrow the completed private allocation for an audited consumer.
    ///
    /// # Safety
    ///
    /// The caller must not mutate the allocation while any other consumer may
    /// read it and must establish Metal ordering before cross-queue use.
    #[cfg(target_os = "macos")]
    pub unsafe fn raw_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }
}
