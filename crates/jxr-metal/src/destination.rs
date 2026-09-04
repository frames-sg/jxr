// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{MetalDecodePlan, MetalError};

use jxr_core::SurfaceLayout;
#[cfg(target_os = "macos")]
use objc2::{rc::Retained, runtime::ProtocolObject};
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBuffer, MTLDevice, MTLResource, MTLStorageMode};

/// One tightly packed image repeated contiguously in a Metal allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseMetalBatchLayout {
    image_layout: SurfaceLayout,
    image_count: usize,
    image_stride: usize,
    byte_len: usize,
}

impl DenseMetalBatchLayout {
    /// Validate a dense, single-plane NHWC batch layout.
    pub fn new(image_layout: SurfaceLayout, image_count: usize) -> Result<Self, MetalError> {
        image_layout
            .validate()
            .map_err(|_| MetalError::InvalidDestination {
                reason: "dense batch image layout is invalid",
            })?;
        if image_count == 0 {
            return Err(MetalError::InvalidDestination {
                reason: "dense batch image count must be nonzero",
            });
        }
        let [plane] = image_layout.planes.as_slice() else {
            return Err(MetalError::InvalidDestination {
                reason: "dense batch images require one interleaved plane",
            });
        };
        let row_bytes = image_layout
            .format
            .row_bytes(image_layout.width)
            .map_err(|_| MetalError::InvalidDestination {
                reason: "dense batch row length is invalid",
            })?;
        let height =
            usize::try_from(image_layout.height).map_err(|_| MetalError::InvalidDestination {
                reason: "dense batch image height exceeds usize",
            })?;
        let image_stride = row_bytes
            .checked_mul(height)
            .ok_or(MetalError::InvalidDestination {
                reason: "dense batch image length overflows usize",
            })?;
        if plane.byte_offset != 0
            || plane.row_stride_bytes != row_bytes
            || plane.width != image_layout.width
            || plane.height != image_layout.height
            || plane.channels != image_layout.format.channel_count()
            || image_layout.byte_len != image_stride
        {
            return Err(MetalError::InvalidDestination {
                reason: "dense batch image layout is not tightly packed NHWC",
            });
        }
        if !image_stride.is_multiple_of(image_layout.required_alignment) {
            return Err(MetalError::InvalidDestination {
                reason: "dense batch image stride violates its alignment",
            });
        }
        let byte_len =
            image_stride
                .checked_mul(image_count)
                .ok_or(MetalError::InvalidDestination {
                    reason: "dense batch allocation length overflows usize",
                })?;
        if u32::try_from(byte_len).is_err() {
            return Err(MetalError::InvalidDestination {
                reason: "dense batch allocation exceeds the shader address ABI",
            });
        }
        Ok(Self {
            image_layout,
            image_count,
            image_stride,
            byte_len,
        })
    }

    /// Validated layout for one image in the batch.
    #[must_use]
    pub const fn image_layout(&self) -> &SurfaceLayout {
        &self.image_layout
    }

    /// Number of images in the batch dimension.
    #[must_use]
    pub const fn image_count(&self) -> usize {
        self.image_count
    }

    /// Byte distance between consecutive images.
    #[must_use]
    pub const fn image_stride(&self) -> usize {
        self.image_stride
    }

    /// Complete checked allocation length.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Byte offset of one image in the batch allocation.
    pub fn image_offset(&self, image: usize) -> Result<usize, MetalError> {
        if image >= self.image_count {
            return Err(MetalError::InvalidDestination {
                reason: "dense batch image index is out of range",
            });
        }
        image
            .checked_mul(self.image_stride)
            .ok_or(MetalError::InvalidDestination {
                reason: "dense batch image offset overflows usize",
            })
    }
}

/// Exclusively writable dense JPEG XR batch in one private Metal buffer.
pub struct MetalBatchDestination {
    #[cfg(target_os = "macos")]
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    device_registry_id: u64,
    layout: DenseMetalBatchLayout,
}

impl core::fmt::Debug for MetalBatchDestination {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MetalBatchDestination")
            .field("device_registry_id", &self.device_registry_id)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

impl MetalBatchDestination {
    /// Validate and retain an exclusively writable private Metal batch.
    ///
    /// # Safety
    ///
    /// Until the returned destination submission completes, the caller must
    /// prevent every CPU access and GPU read or write that overlaps `layout`.
    #[cfg(target_os = "macos")]
    pub unsafe fn from_exclusive_buffer(
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        layout: DenseMetalBatchLayout,
    ) -> Result<Self, MetalError> {
        if buffer.storageMode() != MTLStorageMode::Private {
            return Err(MetalError::InvalidDestination {
                reason: "dense batch destination must use private Metal storage",
            });
        }
        if buffer.length() < layout.byte_len() {
            return Err(MetalError::InvalidDestination {
                reason: "dense batch destination exceeds the Metal allocation",
            });
        }
        Ok(Self {
            device_registry_id: buffer.device().registryID(),
            buffer,
            layout,
        })
    }

    /// Validated dense batch layout.
    #[must_use]
    pub const fn layout(&self) -> &DenseMetalBatchLayout {
        &self.layout
    }

    /// Registry identifier of the allocation's Metal device.
    #[must_use]
    pub const fn device_registry_id(&self) -> u64 {
        self.device_registry_id
    }

    pub(crate) fn validate_plans(&self, plans: &[MetalDecodePlan]) -> Result<(), MetalError> {
        if plans.len() != self.layout.image_count() {
            return Err(MetalError::InvalidDestination {
                reason: "dense batch destination image count differs from plan count",
            });
        }
        if plans
            .iter()
            .any(|plan| plan.output() != self.layout.image_layout())
        {
            return Err(MetalError::InvalidDestination {
                reason: "dense batch image layout differs from a planned output",
            });
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn validate_device(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<(), MetalError> {
        if self.device_registry_id != device.registryID() {
            return Err(MetalError::InvalidDestination {
                reason: "dense batch destination belongs to a different Metal device",
            });
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn buffer_handle(&self) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        self.buffer.clone()
    }
}

/// Exclusively writable JPEG XR surface in a caller-owned Metal buffer.
pub struct MetalDestination {
    #[cfg(target_os = "macos")]
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    device_registry_id: u64,
    layout: SurfaceLayout,
}

impl core::fmt::Debug for MetalDestination {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MetalDestination")
            .field("device_registry_id", &self.device_registry_id)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

impl MetalDestination {
    /// Validate and retain an exclusively writable Metal surface.
    ///
    /// # Safety
    ///
    /// Until the returned destination submission completes, the caller must
    /// prevent every CPU access and GPU read or write that overlaps `layout`.
    #[cfg(target_os = "macos")]
    pub unsafe fn from_exclusive_buffer(
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        layout: SurfaceLayout,
    ) -> Result<Self, MetalError> {
        layout
            .validate()
            .map_err(|_| MetalError::InvalidDestination {
                reason: "destination surface layout is invalid",
            })?;
        if layout.byte_len == 0 || layout.byte_len > buffer.length() {
            return Err(MetalError::InvalidDestination {
                reason: "destination surface exceeds the Metal allocation",
            });
        }
        Ok(Self {
            device_registry_id: buffer.device().registryID(),
            buffer,
            layout,
        })
    }

    /// Validated JXR layout inside the retained allocation.
    #[must_use]
    pub const fn layout(&self) -> &SurfaceLayout {
        &self.layout
    }

    /// Registry identifier of the allocation's Metal device.
    #[must_use]
    pub const fn device_registry_id(&self) -> u64 {
        self.device_registry_id
    }

    pub(crate) fn validate_plan(&self, plan: &MetalDecodePlan) -> Result<(), MetalError> {
        if &self.layout != plan.output() {
            return Err(MetalError::InvalidDestination {
                reason: "destination layout differs from the planned output",
            });
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn validate_device(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<(), MetalError> {
        if self.device_registry_id != device.registryID() {
            return Err(MetalError::InvalidDestination {
                reason: "destination belongs to a different Metal device",
            });
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn buffer_handle(&self) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        self.buffer.clone()
    }
}

#[cfg(test)]
mod tests {
    use jxr_core::{ChannelLayout, PixelFormat, SurfaceLayout};

    use super::DenseMetalBatchLayout;

    #[test]
    fn dense_batch_layout_checks_exact_nhwc_extent() {
        let image = SurfaceLayout::tightly_packed(17, 11, PixelFormat::U16(ChannelLayout::Rgba), 1)
            .unwrap();
        let batch = DenseMetalBatchLayout::new(image.clone(), 8).unwrap();

        assert_eq!(batch.image_layout(), &image);
        assert_eq!(batch.image_count(), 8);
        assert_eq!(batch.image_stride(), 17 * 11 * 4 * 2);
        assert_eq!(batch.byte_len(), 8 * 17 * 11 * 4 * 2);
        assert_eq!(batch.image_offset(7).unwrap(), 7 * 17 * 11 * 4 * 2);
        assert!(batch.image_offset(8).is_err());
    }

    #[test]
    fn dense_batch_layout_rejects_empty_and_overflowing_batches() {
        let image =
            SurfaceLayout::tightly_packed(2, 1, PixelFormat::U8(ChannelLayout::Luma), 1).unwrap();
        assert!(DenseMetalBatchLayout::new(image.clone(), 0).is_err());
        assert!(DenseMetalBatchLayout::new(image, usize::MAX).is_err());

        let one_byte =
            SurfaceLayout::tightly_packed(1, 1, PixelFormat::U8(ChannelLayout::Luma), 1).unwrap();
        assert!(
            DenseMetalBatchLayout::new(one_byte, u32::MAX as usize + 1).is_err(),
            "the shader ABI cannot address a dense batch beyond u32 bytes"
        );
    }
}
