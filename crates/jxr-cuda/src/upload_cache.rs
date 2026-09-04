// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use cudarc::driver::{CudaSlice, CudaStream};

use crate::{
    CudaError,
    abi::{JxrMacroblockAbi, macroblock_abi},
};

const MAX_CACHE_ENTRIES: usize = 256;
const MAX_CACHE_BYTES: usize = 512 * 1024 * 1024;

/// Immutable coefficient upload-cache counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaUploadCacheDiagnostics {
    /// Host-arena/stream entries currently retained.
    pub retained_entries: usize,
    /// Device bytes currently retained.
    pub retained_bytes: usize,
    /// Identity cache hits.
    pub hits: u64,
    /// Device uploads performed.
    pub misses: u64,
    /// Aggregate coefficient and metadata bytes uploaded to the device.
    pub uploaded_bytes: u64,
    /// Entries evicted to enforce bounds.
    pub evictions: u64,
}

#[derive(Debug)]
pub(crate) struct DeviceArena {
    pub(crate) coefficients: CudaSlice<i32>,
    pub(crate) macroblocks: CudaSlice<JxrMacroblockAbi>,
    host: Arc<jxr_core::CoefficientArena>,
    stream_index: usize,
    bytes: usize,
}

#[derive(Debug)]
struct CacheState {
    entries: VecDeque<Arc<DeviceArena>>,
    diagnostics: CudaUploadCacheDiagnostics,
}

#[derive(Debug)]
pub(crate) struct CudaUploadCache {
    state: Mutex<CacheState>,
}

impl CudaUploadCache {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(CacheState {
                entries: VecDeque::new(),
                diagnostics: CudaUploadCacheDiagnostics::default(),
            }),
        }
    }

    pub(crate) fn get_or_upload(
        &self,
        stream_index: usize,
        stream: &Arc<CudaStream>,
        host: Arc<jxr_core::CoefficientArena>,
    ) -> Result<Arc<DeviceArena>, CudaError> {
        {
            let mut state = self.state.lock().map_err(|_| CudaError::StatePoisoned {
                state: "CUDA upload cache",
            })?;
            if let Some(index) = state.entries.iter().position(|entry| {
                entry.stream_index == stream_index && Arc::ptr_eq(&entry.host, &host)
            }) {
                let entry = state
                    .entries
                    .remove(index)
                    .ok_or(CudaError::StateInvariant {
                        state: "CUDA upload cache",
                        reason: "located entry disappeared",
                    })?;
                state.diagnostics.hits = state.diagnostics.hits.saturating_add(1);
                state.entries.push_back(entry.clone());
                return Ok(entry);
            }
            state.diagnostics.misses = state.diagnostics.misses.saturating_add(1);
        }

        let metadata = macroblock_abi(&host.macroblocks)?;
        let coefficients = stream.clone_htod(&host.coefficients)?;
        let macroblocks = stream.clone_htod(&metadata)?;
        let bytes = coefficients
            .num_bytes()
            .checked_add(macroblocks.num_bytes())
            .ok_or(CudaError::ResourceLimit {
                reason: "immutable upload byte count overflows usize",
                requested: usize::MAX,
                maximum: MAX_CACHE_BYTES,
            })?;
        let uploaded = Arc::new(DeviceArena {
            coefficients,
            macroblocks,
            host,
            stream_index,
            bytes,
        });
        {
            let mut state = self.state.lock().map_err(|_| CudaError::StatePoisoned {
                state: "CUDA upload cache",
            })?;
            state.diagnostics.uploaded_bytes = state
                .diagnostics
                .uploaded_bytes
                .saturating_add(u64::try_from(bytes).unwrap_or(u64::MAX));
        }
        if bytes > MAX_CACHE_BYTES {
            return Ok(uploaded);
        }

        let mut state = self.state.lock().map_err(|_| CudaError::StatePoisoned {
            state: "CUDA upload cache",
        })?;
        while state.entries.len() >= MAX_CACHE_ENTRIES
            || state
                .diagnostics
                .retained_bytes
                .checked_add(bytes)
                .is_none_or(|sum| sum > MAX_CACHE_BYTES)
        {
            let Some(evicted) = state.entries.pop_front() else {
                return Err(CudaError::StateInvariant {
                    state: "CUDA upload cache",
                    reason: "cache bound exceeded without an eviction candidate",
                });
            };
            state.diagnostics.retained_bytes = state
                .diagnostics
                .retained_bytes
                .checked_sub(evicted.bytes)
                .ok_or(CudaError::StateInvariant {
                    state: "CUDA upload cache",
                    reason: "retained byte count underflowed",
                })?;
            state.diagnostics.evictions = state.diagnostics.evictions.saturating_add(1);
        }
        state.diagnostics.retained_bytes += bytes;
        state.entries.push_back(uploaded.clone());
        state.diagnostics.retained_entries = state.entries.len();
        Ok(uploaded)
    }

    pub(crate) fn diagnostics(&self) -> Result<CudaUploadCacheDiagnostics, CudaError> {
        self.state
            .lock()
            .map(|state| state.diagnostics)
            .map_err(|_| CudaError::StatePoisoned {
                state: "CUDA upload cache",
            })
    }
}
