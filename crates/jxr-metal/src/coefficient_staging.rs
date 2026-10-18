// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use jxr_core::CoefficientArenaDescriptor;

#[cfg(target_os = "macos")]
use objc2::{rc::Retained, runtime::ProtocolObject};
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBuffer, MTLResource, MTLStorageMode};

/// One pooled shared allocation backing every staging slice cut from it.
///
/// The buffer returns to its session's staging pool when the last slice,
/// sealed arena, or in-flight submission referencing it is dropped, so a
/// later preparation can never overwrite coefficients the GPU still reads.
#[cfg(target_os = "macos")]
pub(crate) struct StagingAllocation {
    buffer: Option<crate::buffer_pool::PooledBuffer>,
    pool: Arc<crate::buffer_pool::StagingPool>,
}

// SAFETY: The allocation is only retained, released, and queried for its
// mapped base address. Mutable CPU access is mediated by disjoint
// `MetalCoefficientStaging` slices behind `&mut self`, sealed arenas are
// immutable, and Metal buffers may be retained and released on any thread.
#[cfg(target_os = "macos")]
unsafe impl Send for StagingAllocation {}
// SAFETY: Shared references expose only the immutable buffer handle; see the
// `Send` justification for how writes to the mapped contents are partitioned.
#[cfg(target_os = "macos")]
unsafe impl Sync for StagingAllocation {}

#[cfg(target_os = "macos")]
impl StagingAllocation {
    pub(crate) fn new(
        buffer: crate::buffer_pool::PooledBuffer,
        pool: Arc<crate::buffer_pool::StagingPool>,
    ) -> Arc<Self> {
        Arc::new(Self {
            buffer: Some(buffer),
            pool,
        })
    }

    fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        self.buffer
            .as_ref()
            .expect("staging allocation owns its buffer until drop")
            .buffer()
    }
}

#[cfg(target_os = "macos")]
impl Drop for StagingAllocation {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            let _ = self.pool.recycle(buffer);
        }
    }
}

/// Exclusively writable shared Metal storage for CPU entropy output.
///
/// Storage may be recycled from an earlier decode, so its contents are
/// unspecified until written: callers must initialize every element before
/// [`Self::seal`].
pub struct MetalCoefficientStaging {
    #[cfg(target_os = "macos")]
    allocation: Arc<StagingAllocation>,
    element_count: usize,
    element_offset: usize,
}

impl MetalCoefficientStaging {
    #[cfg(target_os = "macos")]
    pub(crate) fn new(
        allocation: Arc<StagingAllocation>,
        element_count: usize,
        element_offset: usize,
    ) -> Self {
        Self {
            allocation,
            element_count,
            element_offset,
        }
    }

    /// Number of writable coefficient elements in this slice.
    #[must_use]
    pub const fn element_count(&self) -> usize {
        self.element_count
    }

    /// Element offset of this slice in its shared Metal allocation.
    #[must_use]
    pub const fn element_offset(&self) -> usize {
        self.element_offset
    }

    /// Execute `write` with exclusive direct access to the shared coefficient words.
    pub fn with_coefficients_mut<T>(
        &mut self,
        write: impl FnOnce(&mut [i32]) -> T,
    ) -> Result<T, crate::MetalError> {
        #[cfg(target_os = "macos")]
        {
            let buffer = self.allocation.buffer();
            if buffer.storageMode() != MTLStorageMode::Shared {
                return Err(crate::MetalError::InvalidDestination {
                    reason: "coefficient staging is not in shared storage",
                });
            }
            let byte_len = self
                .element_count
                .checked_mul(core::mem::size_of::<i32>())
                .ok_or(crate::MetalError::InvalidPlan {
                    reason: "coefficient staging byte count overflows usize",
                })?;
            let byte_offset = self
                .element_offset
                .checked_mul(core::mem::size_of::<i32>())
                .ok_or(crate::MetalError::InvalidPlan {
                    reason: "coefficient staging offset overflows usize",
                })?;
            if byte_offset
                .checked_add(byte_len)
                .is_none_or(|end| end > buffer.length())
            {
                return Err(crate::MetalError::InvalidDestination {
                    reason: "coefficient staging exceeds its Metal allocation",
                });
            }
            // SAFETY: The checked element offset lies within the shared allocation.
            let pointer = unsafe {
                buffer
                    .contents()
                    .as_ptr()
                    .cast::<i32>()
                    .add(self.element_offset)
            };
            // SAFETY: `&mut self` provides exclusive CPU access to this slice,
            // sibling slices cover disjoint checked ranges, the allocation is
            // not recycled while referenced, and it has not been submitted.
            let coefficients =
                unsafe { core::slice::from_raw_parts_mut(pointer, self.element_count) };
            Ok(write(coefficients))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = write;
            Err(crate::MetalError::Unavailable)
        }
    }

    /// Seal the storage with validated reconstruction metadata.
    pub fn seal(
        self,
        descriptor: CoefficientArenaDescriptor,
    ) -> Result<MetalCoefficientArena, crate::MetalError> {
        descriptor
            .validate()
            .map_err(|_| crate::MetalError::InvalidPlan {
                reason: "shared coefficient descriptor is invalid",
            })?;
        if descriptor.coefficient_count != self.element_count {
            return Err(crate::MetalError::InvalidPlan {
                reason: "shared coefficient descriptor length differs from staging",
            });
        }
        Ok(MetalCoefficientArena {
            #[cfg(target_os = "macos")]
            allocation: self.allocation,
            #[cfg(target_os = "macos")]
            element_offset: self.element_offset,
            descriptor: Arc::new(descriptor),
        })
    }
}

/// Immutable CPU-produced coefficients already resident in shared Metal storage.
pub struct MetalCoefficientArena {
    #[cfg(target_os = "macos")]
    allocation: Arc<StagingAllocation>,
    #[cfg(target_os = "macos")]
    element_offset: usize,
    descriptor: Arc<CoefficientArenaDescriptor>,
}

impl core::fmt::Debug for MetalCoefficientArena {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("MetalCoefficientArena")
            .field("coefficient_count", &self.descriptor.coefficient_count)
            .field("planes", &self.descriptor.planes.len())
            .finish_non_exhaustive()
    }
}

impl MetalCoefficientArena {
    pub(crate) fn descriptor(&self) -> &CoefficientArenaDescriptor {
        &self.descriptor
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn buffer_handle(&self) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        self.allocation
            .buffer
            .as_ref()
            .expect("staging allocation owns its buffer until drop")
            .handle()
    }

    /// Keeps the pooled allocation out of reuse until a submission completes.
    #[cfg(target_os = "macos")]
    pub(crate) fn lease(&self) -> Arc<StagingAllocation> {
        self.allocation.clone()
    }

    #[cfg(target_os = "macos")]
    pub(crate) const fn element_offset(&self) -> usize {
        self.element_offset
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn allocation_address(&self) -> usize {
        core::ptr::from_ref(self.allocation.buffer()).cast::<()>() as usize
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    #[test]
    fn variable_staging_slices_are_contiguous_and_checked() {
        let session = crate::MetalDecoderSession::system_default().unwrap();
        let slices = session.coefficient_staging_slices(&[3, 5, 2]).unwrap();
        assert_eq!(
            slices
                .iter()
                .map(|slice| (slice.element_offset(), slice.element_count()))
                .collect::<Vec<_>>(),
            [(0, 3), (3, 5), (8, 2)]
        );
        assert!(session.coefficient_staging_slices(&[]).is_err());
        assert!(session.coefficient_staging_slices(&[1, 0]).is_err());
        assert!(
            session
                .coefficient_staging_slices(&[usize::MAX, 1])
                .is_err()
        );
    }
}
