// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use jxr_core::CoefficientArenaDescriptor;

#[cfg(target_os = "macos")]
use objc2::{rc::Retained, runtime::ProtocolObject};
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBuffer, MTLResource, MTLStorageMode};

/// Exclusively writable shared Metal storage for CPU entropy output.
pub struct MetalCoefficientStaging {
    #[cfg(target_os = "macos")]
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    element_count: usize,
    element_offset: usize,
}

// SAFETY: Staging has unique ownership, exposes mapped memory only through
// `&mut self`, and is never submitted before it is consumed by `seal`.
#[cfg(target_os = "macos")]
unsafe impl Send for MetalCoefficientStaging {}

impl MetalCoefficientStaging {
    #[cfg(target_os = "macos")]
    pub(crate) fn new(
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        element_count: usize,
        element_offset: usize,
    ) -> Self {
        Self {
            buffer,
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
            if self.buffer.storageMode() != MTLStorageMode::Shared {
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
                .is_none_or(|end| end > self.buffer.length())
            {
                return Err(crate::MetalError::InvalidDestination {
                    reason: "coefficient staging exceeds its Metal allocation",
                });
            }
            // SAFETY: The checked element offset lies within the shared allocation.
            let pointer = unsafe {
                self.buffer
                    .contents()
                    .as_ptr()
                    .cast::<i32>()
                    .add(self.element_offset)
            };
            // SAFETY: `&mut self` provides exclusive CPU access, the buffer has
            // not been sealed or submitted, and the checked typed range fits.
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
            buffer: self.buffer,
            #[cfg(target_os = "macos")]
            element_offset: self.element_offset,
            descriptor: Arc::new(descriptor),
        })
    }
}

/// Immutable CPU-produced coefficients already resident in shared Metal storage.
pub struct MetalCoefficientArena {
    #[cfg(target_os = "macos")]
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    #[cfg(target_os = "macos")]
    element_offset: usize,
    descriptor: Arc<CoefficientArenaDescriptor>,
}

// SAFETY: Sealing removes mutable CPU access. Metal buffers and immutable
// resource descriptors may be retained and bound from different host threads;
// command ordering remains the submitting session's responsibility.
#[cfg(target_os = "macos")]
unsafe impl Send for MetalCoefficientArena {}
// SAFETY: Every field is immutable after sealing, and Metal permits concurrent
// read-only resource references during command encoding.
#[cfg(target_os = "macos")]
unsafe impl Sync for MetalCoefficientArena {}

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
        self.buffer.clone()
    }

    #[cfg(target_os = "macos")]
    pub(crate) const fn element_offset(&self) -> usize {
        self.element_offset
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn allocation_address(&self) -> usize {
        Retained::as_ptr(&self.buffer).cast::<()>() as usize
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
