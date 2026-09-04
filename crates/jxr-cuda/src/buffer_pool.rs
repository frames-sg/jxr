// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::VecDeque, sync::Mutex};

use cudarc::driver::{CudaSlice, CudaStream};

use crate::CudaError;

const MAX_RETAINED_BYTES: usize = 256 * 1024 * 1024;
const MAX_BUFFER_BYTES: usize = 128 * 1024 * 1024;

/// Exact-size CUDA scratch reuse counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaBufferPoolDiagnostics {
    /// Scratch buffers currently retained by the session.
    pub retained_buffers: usize,
    /// Bytes currently retained by the session.
    pub retained_bytes: usize,
    /// Highest retained byte count observed.
    pub high_water_bytes: usize,
    /// Exact-size pool hits.
    pub hits: u64,
    /// New device allocations.
    pub misses: u64,
    /// Buffers evicted or rejected by the bound.
    pub evictions: u64,
}

#[derive(Debug)]
struct PoolState {
    buffers: VecDeque<CudaSlice<i32>>,
    diagnostics: CudaBufferPoolDiagnostics,
}

#[derive(Debug)]
pub(crate) struct CudaBufferPool {
    state: Mutex<PoolState>,
}

impl CudaBufferPool {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(PoolState {
                buffers: VecDeque::new(),
                diagnostics: CudaBufferPoolDiagnostics::default(),
            }),
        }
    }

    pub(crate) fn take(
        &self,
        stream: &std::sync::Arc<CudaStream>,
        elements: usize,
    ) -> Result<CudaSlice<i32>, CudaError> {
        let mut state = self.state.lock().map_err(|_| CudaError::StatePoisoned {
            state: "CUDA scratch pool",
        })?;
        if let Some(index) = state
            .buffers
            .iter()
            .position(|buffer| buffer.len() == elements && buffer.context() == stream.context())
        {
            let buffer = state
                .buffers
                .remove(index)
                .ok_or(CudaError::StateInvariant {
                    state: "CUDA scratch pool",
                    reason: "located buffer disappeared",
                })?;
            state.diagnostics.retained_buffers =
                state.diagnostics.retained_buffers.saturating_sub(1);
            state.diagnostics.retained_bytes = state
                .diagnostics
                .retained_bytes
                .checked_sub(buffer.num_bytes())
                .ok_or(CudaError::StateInvariant {
                    state: "CUDA scratch pool",
                    reason: "retained byte count underflowed",
                })?;
            state.diagnostics.hits = state.diagnostics.hits.saturating_add(1);
            return Ok(buffer);
        }
        state.diagnostics.misses = state.diagnostics.misses.saturating_add(1);
        drop(state);
        Ok(stream.alloc_zeros(elements)?)
    }

    pub(crate) fn recycle(&self, buffer: CudaSlice<i32>) -> Result<(), CudaError> {
        let bytes = buffer.num_bytes();
        let mut state = self.state.lock().map_err(|_| CudaError::StatePoisoned {
            state: "CUDA scratch pool",
        })?;
        if bytes > MAX_BUFFER_BYTES || bytes > MAX_RETAINED_BYTES {
            state.diagnostics.evictions = state.diagnostics.evictions.saturating_add(1);
            return Ok(());
        }
        while state
            .diagnostics
            .retained_bytes
            .checked_add(bytes)
            .is_none_or(|sum| sum > MAX_RETAINED_BYTES)
        {
            let Some(evicted) = state.buffers.pop_front() else {
                return Err(CudaError::StateInvariant {
                    state: "CUDA scratch pool",
                    reason: "retention bound exceeded without an eviction candidate",
                });
            };
            state.diagnostics.retained_buffers =
                state.diagnostics.retained_buffers.saturating_sub(1);
            state.diagnostics.retained_bytes = state
                .diagnostics
                .retained_bytes
                .checked_sub(evicted.num_bytes())
                .ok_or(CudaError::StateInvariant {
                    state: "CUDA scratch pool",
                    reason: "retained byte count underflowed during eviction",
                })?;
            state.diagnostics.evictions = state.diagnostics.evictions.saturating_add(1);
        }
        state.diagnostics.retained_bytes += bytes;
        state.diagnostics.retained_buffers += 1;
        state.diagnostics.high_water_bytes = state
            .diagnostics
            .high_water_bytes
            .max(state.diagnostics.retained_bytes);
        state.buffers.push_back(buffer);
        Ok(())
    }

    pub(crate) fn diagnostics(&self) -> Result<CudaBufferPoolDiagnostics, CudaError> {
        self.state
            .lock()
            .map(|state| state.diagnostics)
            .map_err(|_| CudaError::StatePoisoned {
                state: "CUDA scratch pool",
            })
    }
}
