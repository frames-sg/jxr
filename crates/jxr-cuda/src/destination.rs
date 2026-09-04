// SPDX-License-Identifier: MIT OR Apache-2.0

use cudarc::driver::CudaSlice;
use jxr_core::SurfaceLayout;

use crate::CudaError;

/// One tightly packed image repeated contiguously in a CUDA allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseCudaBatchLayout {
    image: SurfaceLayout,
    image_count: usize,
    image_stride_bytes: usize,
    byte_len: usize,
}

/// Exclusively writable JPEG XR surface in a caller-owned CUDA allocation.
pub struct CudaDestination {
    buffer: CudaSlice<u8>,
    layout: SurfaceLayout,
}

impl core::fmt::Debug for CudaDestination {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("CudaDestination")
            .field("device_ordinal", &self.buffer.ordinal())
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

impl CudaDestination {
    /// Validate and take exclusive ownership of a device allocation.
    pub fn from_device_buffer(
        buffer: CudaSlice<u8>,
        layout: SurfaceLayout,
    ) -> Result<Self, CudaError> {
        layout
            .validate()
            .map_err(|_| CudaError::InvalidDestination {
                reason: "CUDA destination surface layout is invalid",
            })?;
        if buffer.len() < layout.byte_len {
            return Err(CudaError::InvalidDestination {
                reason: "CUDA destination allocation is too small",
            });
        }
        Ok(Self { buffer, layout })
    }

    /// Validated surface layout.
    #[must_use]
    pub const fn layout(&self) -> &SurfaceLayout {
        &self.layout
    }

    /// CUDA device ordinal that owns the allocation.
    #[must_use]
    pub fn device_ordinal(&self) -> usize {
        self.buffer.ordinal()
    }

    /// Immutable device allocation, suitable for a consumer after completion.
    #[must_use]
    pub const fn device_buffer(&self) -> &CudaSlice<u8> {
        &self.buffer
    }

    pub(crate) fn validate_plan(&self, plan: &crate::CudaDecodePlan) -> Result<(), CudaError> {
        if plan.output() != &self.layout {
            return Err(CudaError::InvalidDestination {
                reason: "CUDA destination layout differs from the decode plan",
            });
        }
        Ok(())
    }

    pub(crate) fn validate_context(
        &self,
        context: &std::sync::Arc<cudarc::driver::CudaContext>,
    ) -> Result<(), CudaError> {
        if self.buffer.context() != context {
            return Err(CudaError::InvalidDestination {
                reason: "CUDA destination belongs to a different context",
            });
        }
        Ok(())
    }

    pub(crate) fn buffer_mut(&mut self) -> &mut CudaSlice<u8> {
        &mut self.buffer
    }
}

/// Exclusively writable homogeneous batch in one caller-owned CUDA allocation.
pub struct CudaBatchDestination {
    buffer: CudaSlice<u8>,
    layout: DenseCudaBatchLayout,
}

impl core::fmt::Debug for CudaBatchDestination {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("CudaBatchDestination")
            .field("device_ordinal", &self.buffer.ordinal())
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

impl CudaBatchDestination {
    /// Validate and take exclusive ownership of a dense device allocation.
    pub fn from_device_buffer(
        buffer: CudaSlice<u8>,
        layout: DenseCudaBatchLayout,
    ) -> Result<Self, CudaError> {
        if buffer.len() < layout.byte_len() {
            return Err(CudaError::InvalidDestination {
                reason: "dense CUDA destination allocation is too small",
            });
        }
        Ok(Self { buffer, layout })
    }

    /// Validated dense layout.
    #[must_use]
    pub const fn layout(&self) -> &DenseCudaBatchLayout {
        &self.layout
    }

    /// CUDA device ordinal that owns the allocation.
    #[must_use]
    pub fn device_ordinal(&self) -> usize {
        self.buffer.ordinal()
    }

    /// Immutable device allocation, suitable for a consumer after completion.
    #[must_use]
    pub const fn device_buffer(&self) -> &CudaSlice<u8> {
        &self.buffer
    }

    pub(crate) fn validate_plans(&self, plans: &[crate::CudaDecodePlan]) -> Result<(), CudaError> {
        if plans.len() != self.layout.image_count() {
            return Err(CudaError::InvalidDestination {
                reason: "dense CUDA destination image count differs from the plans",
            });
        }
        if plans
            .iter()
            .any(|plan| plan.output() != self.layout.image_layout())
        {
            return Err(CudaError::InvalidDestination {
                reason: "dense CUDA destination layout differs from a decode plan",
            });
        }
        Ok(())
    }

    pub(crate) fn validate_context(
        &self,
        context: &std::sync::Arc<cudarc::driver::CudaContext>,
    ) -> Result<(), CudaError> {
        if self.buffer.context() != context {
            return Err(CudaError::InvalidDestination {
                reason: "dense CUDA destination belongs to a different context",
            });
        }
        Ok(())
    }

    pub(crate) fn buffer_mut(&mut self) -> &mut CudaSlice<u8> {
        &mut self.buffer
    }
}

impl DenseCudaBatchLayout {
    /// Validate a homogeneous dense batch layout.
    pub fn new(image: SurfaceLayout, image_count: usize) -> Result<Self, CudaError> {
        image
            .validate()
            .map_err(|_| CudaError::InvalidDestination {
                reason: "dense batch image layout is invalid",
            })?;
        if image_count == 0 {
            return Err(CudaError::InvalidDestination {
                reason: "dense batch cannot be empty",
            });
        }
        let image_count_u32 =
            u32::try_from(image_count).map_err(|_| CudaError::InvalidDestination {
                reason: "dense batch image count exceeds the CUDA ABI",
            })?;
        let _ = image_count_u32;
        let image_stride_bytes = image.byte_len;
        let byte_len =
            image_stride_bytes
                .checked_mul(image_count)
                .ok_or(CudaError::InvalidDestination {
                    reason: "dense batch byte length overflows usize",
                })?;
        u32::try_from(byte_len).map_err(|_| CudaError::InvalidDestination {
            reason: "dense batch byte length exceeds the CUDA ABI",
        })?;
        Ok(Self {
            image,
            image_count,
            image_stride_bytes,
            byte_len,
        })
    }

    /// Per-image surface layout.
    #[must_use]
    pub const fn image_layout(&self) -> &SurfaceLayout {
        &self.image
    }

    /// Number of images in the dense batch.
    #[must_use]
    pub const fn image_count(&self) -> usize {
        self.image_count
    }

    /// Byte distance between consecutive images.
    #[must_use]
    pub const fn image_stride_bytes(&self) -> usize {
        self.image_stride_bytes
    }

    /// Total required allocation size.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Checked byte offset for one image.
    pub fn image_offset(&self, image: usize) -> Result<usize, CudaError> {
        if image >= self.image_count {
            return Err(CudaError::InvalidDestination {
                reason: "dense batch image index is out of range",
            });
        }
        self.image_stride_bytes
            .checked_mul(image)
            .ok_or(CudaError::InvalidDestination {
                reason: "dense batch image offset overflows usize",
            })
    }
}
