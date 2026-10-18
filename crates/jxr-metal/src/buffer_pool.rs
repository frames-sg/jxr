// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Mutex;

use j2k_metal_support::{checked_private_buffer, checked_shared_buffer};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use crate::MetalError;

mod diagnostics;
mod state;
#[cfg(test)]
mod tests;

pub use diagnostics::{MetalBufferPoolDiagnostics, MetalBufferPoolsDiagnostics};
pub(crate) use state::PooledBuffer;
use state::{PoolLimits, PoolState};

pub(crate) struct MetalBufferPools {
    private: Mutex<PoolState>,
    shared: Mutex<PoolState>,
}

impl MetalBufferPools {
    pub(crate) fn new(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        let limits = PoolLimits::for_device(device.maxBufferLength());
        Self {
            private: Mutex::new(PoolState::new(limits)),
            shared: Mutex::new(PoolState::new(limits)),
        }
    }

    pub(crate) fn take_private(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        bytes: usize,
    ) -> Result<PooledBuffer, MetalError> {
        Self::take_or_allocate(&self.private, bytes, |bytes| {
            checked_private_buffer(device, bytes).map_err(MetalError::from)
        })
    }

    pub(crate) fn take_shared(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        bytes: usize,
    ) -> Result<PooledBuffer, MetalError> {
        Self::take_or_allocate(&self.shared, bytes, |bytes| {
            checked_shared_buffer(device, bytes).map_err(MetalError::from)
        })
    }

    fn take_or_allocate(
        state: &Mutex<PoolState>,
        bytes: usize,
        allocate: impl FnOnce(
            usize,
        ) -> Result<
            objc2::rc::Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
            MetalError,
        >,
    ) -> Result<PooledBuffer, MetalError> {
        let bytes = bytes.max(1);
        if let Some(buffer) = state
            .lock()
            .map_err(|_| MetalError::StatePoisoned {
                state: "Metal buffer pool",
            })?
            .take(bytes)
            .map_err(pool_invariant)?
        {
            return Ok(buffer);
        }
        PooledBuffer::new(bytes, allocate(bytes)?).map_err(pool_invariant)
    }

    pub(crate) fn recycle_private(&self, buffer: PooledBuffer) -> Result<(), MetalError> {
        recycle(&self.private, buffer)
    }

    pub(crate) fn recycle_shared(&self, buffer: PooledBuffer) -> Result<(), MetalError> {
        recycle(&self.shared, buffer)
    }

    pub(crate) fn diagnostics(&self) -> Result<MetalBufferPoolsDiagnostics, MetalError> {
        Ok(MetalBufferPoolsDiagnostics {
            private: self
                .private
                .lock()
                .map_err(|_| MetalError::StatePoisoned {
                    state: "private Metal buffer pool",
                })?
                .diagnostics(),
            shared: self
                .shared
                .lock()
                .map_err(|_| MetalError::StatePoisoned {
                    state: "shared Metal buffer pool",
                })?
                .diagnostics(),
            staging: MetalBufferPoolDiagnostics::default(),
        })
    }
}

/// Exact-size retention for shared CPU entropy staging.
///
/// Fresh shared allocations fault in every page on first CPU write, which
/// costs more than the entropy copy itself for multi-megabyte images. Unlike
/// the runtime scratch pools, staging is released from whichever thread drops
/// the last plan or submission referencing it, so the pool is shared by `Arc`.
pub(crate) struct StagingPool {
    state: Mutex<PoolState>,
}

// SAFETY: The pool only retains, releases, and transfers whole Metal buffers
// under its mutex and never touches their contents. Metal buffer objects may
// be retained, released, and handed between threads; CPU access to staging
// contents is partitioned by `MetalCoefficientStaging` slices.
unsafe impl Send for StagingPool {}
// SAFETY: Every shared-reference operation locks the mutex; see `Send`.
unsafe impl Sync for StagingPool {}

impl StagingPool {
    pub(crate) fn new(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        Self {
            state: Mutex::new(PoolState::new(PoolLimits::for_device(
                device.maxBufferLength(),
            ))),
        }
    }

    pub(crate) fn take(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        bytes: usize,
    ) -> Result<PooledBuffer, MetalError> {
        MetalBufferPools::take_or_allocate(&self.state, bytes, |bytes| {
            checked_shared_buffer(device, bytes).map_err(MetalError::from)
        })
    }

    pub(crate) fn recycle(&self, buffer: PooledBuffer) -> Result<(), MetalError> {
        recycle(&self.state, buffer)
    }

    pub(crate) fn diagnostics(&self) -> Result<MetalBufferPoolDiagnostics, MetalError> {
        Ok(self
            .state
            .lock()
            .map_err(|_| MetalError::StatePoisoned {
                state: "Metal staging pool",
            })?
            .diagnostics())
    }
}

fn recycle(state: &Mutex<PoolState>, buffer: PooledBuffer) -> Result<(), MetalError> {
    state
        .lock()
        .map_err(|_| MetalError::StatePoisoned {
            state: "Metal buffer pool",
        })?
        .recycle(buffer)
        .map_err(pool_invariant)
}

fn pool_invariant(reason: &'static str) -> MetalError {
    MetalError::StateInvariant {
        state: "Metal buffer pool",
        reason,
    }
}
