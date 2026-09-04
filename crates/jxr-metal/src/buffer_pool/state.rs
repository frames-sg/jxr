// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::VecDeque;

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLBuffer;

use super::MetalBufferPoolDiagnostics;

pub(super) const DEFAULT_RETAINED_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_RETAINED_BUFFERS: usize = 64;

#[derive(Clone, Copy, Debug)]
pub(super) struct PoolLimits {
    retained_bytes: usize,
    retained_buffers: usize,
}

impl PoolLimits {
    pub(super) fn for_device(max_buffer_length: usize) -> Self {
        Self {
            retained_bytes: max_buffer_length.min(DEFAULT_RETAINED_BYTES),
            retained_buffers: DEFAULT_RETAINED_BUFFERS,
        }
    }

    #[cfg(test)]
    pub(super) const fn new(retained_bytes: usize, retained_buffers: usize) -> Self {
        Self {
            retained_bytes,
            retained_buffers,
        }
    }
}

pub(crate) struct PooledBuffer {
    bytes: usize,
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl PooledBuffer {
    pub(super) fn new(
        bytes: usize,
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    ) -> Result<Self, &'static str> {
        if bytes != buffer.length() {
            return Err("recorded buffer size differs from its Metal allocation");
        }
        Ok(Self { bytes, buffer })
    }

    #[cfg(test)]
    pub(crate) const fn bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    pub(crate) fn handle(&self) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        self.buffer.clone()
    }
}

#[derive(Default)]
struct PoolCounters {
    peak_cached_bytes: usize,
    peak_cached_buffers: usize,
    evictions: usize,
    rejections: usize,
}

pub(super) struct PoolState {
    entries: VecDeque<PooledBuffer>,
    retained_bytes: usize,
    limits: PoolLimits,
    counters: PoolCounters,
}

impl PoolState {
    pub(super) fn new(limits: PoolLimits) -> Self {
        Self {
            entries: VecDeque::new(),
            retained_bytes: 0,
            limits,
            counters: PoolCounters::default(),
        }
    }

    pub(super) fn take(&mut self, bytes: usize) -> Result<Option<PooledBuffer>, &'static str> {
        let Some(index) = self.entries.iter().position(|entry| entry.bytes == bytes) else {
            return Ok(None);
        };
        let entry = self
            .entries
            .remove(index)
            .ok_or("matched pool entry disappeared")?;
        self.retained_bytes = self
            .retained_bytes
            .checked_sub(entry.bytes)
            .ok_or("retained byte count underflow")?;
        Ok(Some(entry))
    }

    pub(super) fn recycle(&mut self, buffer: PooledBuffer) -> Result<(), &'static str> {
        if buffer.bytes > self.limits.retained_bytes || self.limits.retained_buffers == 0 {
            self.counters.rejections = self
                .counters
                .rejections
                .checked_add(1)
                .ok_or("pool rejection counter overflow")?;
            return Ok(());
        }

        while self.entries.len() >= self.limits.retained_buffers
            || self
                .retained_bytes
                .checked_add(buffer.bytes)
                .ok_or("retained byte count overflow")?
                > self.limits.retained_bytes
        {
            let oldest = self
                .entries
                .pop_front()
                .ok_or("pool is empty while eviction is required")?;
            self.retained_bytes = self
                .retained_bytes
                .checked_sub(oldest.bytes)
                .ok_or("retained byte count underflow during eviction")?;
            self.counters.evictions = self
                .counters
                .evictions
                .checked_add(1)
                .ok_or("pool eviction counter overflow")?;
        }

        self.retained_bytes = self
            .retained_bytes
            .checked_add(buffer.bytes)
            .ok_or("retained byte count overflow")?;
        self.entries.push_back(buffer);
        self.counters.peak_cached_bytes = self.counters.peak_cached_bytes.max(self.retained_bytes);
        self.counters.peak_cached_buffers =
            self.counters.peak_cached_buffers.max(self.entries.len());
        Ok(())
    }

    pub(super) fn diagnostics(&self) -> MetalBufferPoolDiagnostics {
        MetalBufferPoolDiagnostics {
            cached_bytes: self.retained_bytes,
            cached_buffers: self.entries.len(),
            peak_cached_bytes: self.counters.peak_cached_bytes,
            peak_cached_buffers: self.counters.peak_cached_buffers,
            evictions: self.counters.evictions,
            rejections: self.counters.rejections,
        }
    }
}
