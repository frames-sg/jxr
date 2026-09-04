use core::{ffi::c_void, ptr::NonNull};

use j2k_core::accelerator::GpuAbi;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLResource};

use crate::MetalError;

pub(crate) trait JxrComputeEncoderExt {
    fn bind_buffer(
        &self,
        index: usize,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
    ) -> Result<(), MetalError>;

    fn bind_bytes<T: GpuAbi>(&self, index: usize, value: &T) -> Result<(), MetalError>;

    fn memory_barrier(&self, buffers: &[&ProtocolObject<dyn MTLBuffer>]);
}

impl JxrComputeEncoderExt for ProtocolObject<dyn MTLComputeCommandEncoder> {
    fn bind_buffer(
        &self,
        index: usize,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
    ) -> Result<(), MetalError> {
        if index >= 31 {
            return Err(MetalError::InvalidPlan {
                reason: "Metal buffer binding index exceeds the compute table",
            });
        }
        if offset > buffer.length() {
            return Err(MetalError::InvalidPlan {
                reason: "Metal buffer binding offset exceeds the allocation",
            });
        }
        // SAFETY: The slot and byte offset were checked above. All buffers are
        // retained by the retaining command buffer until completion, and each
        // call site binds the statically matched JXR shader ABI.
        unsafe { self.setBuffer_offset_atIndex(Some(buffer), offset, index) };
        Ok(())
    }

    fn bind_bytes<T: GpuAbi>(&self, index: usize, value: &T) -> Result<(), MetalError> {
        if index >= 31 {
            return Err(MetalError::InvalidPlan {
                reason: "Metal byte binding index exceeds the compute table",
            });
        }
        let bytes = T::as_bytes(value);
        let pointer = NonNull::from(bytes).cast::<c_void>();
        // SAFETY: `GpuAbi` proves that every byte is initialized and has the
        // shader-visible layout. Metal copies `setBytes` data during encoding,
        // so the borrowed value need not outlive this call.
        unsafe { self.setBytes_length_atIndex(pointer, bytes.len(), index) };
        Ok(())
    }

    fn memory_barrier(&self, buffers: &[&ProtocolObject<dyn MTLBuffer>]) {
        let mut resources: Vec<NonNull<ProtocolObject<dyn MTLResource>>> = buffers
            .iter()
            .map(|buffer| {
                let resource: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(*buffer);
                NonNull::from(resource)
            })
            .collect();
        let pointer = NonNull::new(resources.as_mut_ptr())
            .expect("Metal resource barrier requires at least one buffer");
        // SAFETY: `resources` contains one valid protocol-object pointer per
        // retained buffer for this synchronous encoding call. MTLBuffer
        // declares MTLResource conformance and the slice remains alive here.
        unsafe { self.memoryBarrierWithResources_count(pointer, resources.len()) };
    }
}
