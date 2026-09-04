// SPDX-License-Identifier: MIT OR Apache-2.0

use jxr_core::{DecodeReport, ImageInfo, PixelFormat, Rect, SurfaceLayout};

#[cfg(target_os = "macos")]
use std::rc::Rc;

#[cfg(target_os = "macos")]
use objc2_metal::{MTLBuffer, MTLDevice, MTLResource, MTLStorageMode};

/// Completed host-visible Metal output without a copy into Rust-owned samples.
///
/// Byte access is closure-scoped so the backing pooled allocation cannot be
/// recycled while borrowed. The image is immutable after GPU completion.
pub struct SharedMetalImage {
    #[cfg(target_os = "macos")]
    buffer: Option<crate::buffer_pool::PooledBuffer>,
    #[cfg(target_os = "macos")]
    pools: Rc<crate::buffer_pool::MetalBufferPools>,
    layout: SurfaceLayout,
    info: ImageInfo,
    decoded_region: Rect,
    format: PixelFormat,
    report: DecodeReport,
    device_registry_id: u64,
}

impl core::fmt::Debug for SharedMetalImage {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SharedMetalImage")
            .field("layout", &self.layout)
            .field("format", &self.format)
            .field("device_registry_id", &self.device_registry_id)
            .finish_non_exhaustive()
    }
}

impl SharedMetalImage {
    #[cfg(target_os = "macos")]
    pub(crate) fn from_pooled(
        buffer: crate::buffer_pool::PooledBuffer,
        pools: Rc<crate::buffer_pool::MetalBufferPools>,
        layout: SurfaceLayout,
        info: ImageInfo,
        decoded_region: Rect,
        format: PixelFormat,
        report: DecodeReport,
    ) -> Self {
        let device_registry_id = buffer.buffer().device().registryID();
        Self {
            buffer: Some(buffer),
            pools,
            layout,
            info,
            decoded_region,
            format,
            report,
            device_registry_id,
        }
    }

    /// Execute `read` with the completed output bytes without allocating or copying.
    pub fn with_bytes<T>(&self, read: impl FnOnce(&[u8]) -> T) -> Result<T, crate::MetalError> {
        #[cfg(target_os = "macos")]
        {
            let buffer = self
                .buffer
                .as_ref()
                .ok_or(crate::MetalError::InvalidSubmissionState {
                    expected: "completed shared image",
                    actual: "recycled shared image",
                })?
                .buffer();
            if buffer.storageMode() != MTLStorageMode::Shared {
                return Err(crate::MetalError::InvalidDestination {
                    reason: "completed host image is not in shared storage",
                });
            }
            if self.layout.byte_len > buffer.length() {
                return Err(crate::MetalError::InvalidDestination {
                    reason: "completed host image exceeds its shared allocation",
                });
            }
            let pointer = buffer.contents().as_ptr().cast::<u8>();
            // SAFETY: The owning command completed before construction, the
            // shared allocation is retained and immutable for `self`, and the
            // checked range does not exceed the Metal allocation.
            let bytes = unsafe { core::slice::from_raw_parts(pointer, self.layout.byte_len) };
            Ok(read(bytes))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = read;
            Err(crate::MetalError::Unavailable)
        }
    }

    #[must_use]
    pub const fn layout(&self) -> &SurfaceLayout {
        &self.layout
    }

    #[must_use]
    pub const fn info(&self) -> &ImageInfo {
        &self.info
    }

    #[must_use]
    pub const fn decoded_region(&self) -> Rect {
        self.decoded_region
    }

    #[must_use]
    pub const fn format(&self) -> PixelFormat {
        self.format
    }

    #[must_use]
    pub const fn report(&self) -> &DecodeReport {
        &self.report
    }

    #[must_use]
    pub const fn device_registry_id(&self) -> u64 {
        self.device_registry_id
    }
}

impl Drop for SharedMetalImage {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        if let Some(buffer) = self.buffer.take() {
            let _ = self.pools.recycle_shared(buffer);
        }
    }
}
