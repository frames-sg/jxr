// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use j2k_metal_support::checked_shared_buffer_with_slice;
use jxr_core::CoefficientArena;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLDevice};

use crate::{MetalError, abi::macroblock_abi};

const DEFAULT_RETAINED_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone)]
pub(crate) struct CoefficientUpload {
    pub(crate) packed: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) macroblocks: Retained<ProtocolObject<dyn MTLBuffer>>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct MetalUploadCacheDiagnostics {
    pub retained_bytes: usize,
    pub retained_images: usize,
    pub peak_retained_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub rejections: u64,
}

struct CacheEntry {
    arena: Arc<CoefficientArena>,
    upload: CoefficientUpload,
    bytes: usize,
}

struct CacheState {
    entries: VecDeque<CacheEntry>,
    diagnostics: MetalUploadCacheDiagnostics,
}

pub(crate) struct CoefficientUploadCache {
    state: Mutex<CacheState>,
    retained_limit: usize,
}

impl CoefficientUploadCache {
    pub(crate) fn new(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        Self {
            state: Mutex::new(CacheState {
                entries: VecDeque::new(),
                diagnostics: MetalUploadCacheDiagnostics::default(),
            }),
            retained_limit: device.maxBufferLength().min(DEFAULT_RETAINED_BYTES),
        }
    }

    pub(crate) fn get_or_upload(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        arena: &Arc<CoefficientArena>,
    ) -> Result<CoefficientUpload, MetalError> {
        let mut state = self.lock()?;
        if let Some(entry) = state
            .entries
            .iter()
            .find(|entry| Arc::ptr_eq(&entry.arena, arena))
        {
            let upload = entry.upload.clone();
            state.diagnostics.hits = state.diagnostics.hits.saturating_add(1);
            return Ok(upload);
        }

        state.diagnostics.misses = state.diagnostics.misses.saturating_add(1);
        let metadata = macroblock_abi(arena)?;
        let upload = CoefficientUpload {
            packed: checked_shared_buffer_with_slice(device, &arena.coefficients)?,
            macroblocks: checked_shared_buffer_with_slice(device, &metadata)?,
        };
        let bytes = upload
            .packed
            .length()
            .checked_add(upload.macroblocks.length())
            .ok_or(MetalError::StateInvariant {
                state: "Metal coefficient upload cache",
                reason: "upload byte count overflows usize",
            })?;
        if bytes > self.retained_limit {
            state.diagnostics.rejections = state.diagnostics.rejections.saturating_add(1);
            return Ok(upload);
        }
        while state.diagnostics.retained_bytes.checked_add(bytes).ok_or(
            MetalError::StateInvariant {
                state: "Metal coefficient upload cache",
                reason: "retained byte count overflows usize",
            },
        )? > self.retained_limit
        {
            let evicted = state
                .entries
                .pop_front()
                .ok_or(MetalError::StateInvariant {
                    state: "Metal coefficient upload cache",
                    reason: "cache is empty while eviction is required",
                })?;
            state.diagnostics.retained_bytes = state
                .diagnostics
                .retained_bytes
                .checked_sub(evicted.bytes)
                .ok_or(MetalError::StateInvariant {
                    state: "Metal coefficient upload cache",
                    reason: "retained byte count underflow",
                })?;
            state.diagnostics.evictions = state.diagnostics.evictions.saturating_add(1);
        }
        state.diagnostics.retained_bytes += bytes;
        state.diagnostics.peak_retained_bytes = state
            .diagnostics
            .peak_retained_bytes
            .max(state.diagnostics.retained_bytes);
        state.entries.push_back(CacheEntry {
            arena: arena.clone(),
            upload: upload.clone(),
            bytes,
        });
        state.diagnostics.retained_images = state.entries.len();
        Ok(upload)
    }

    pub(crate) fn diagnostics(&self) -> Result<MetalUploadCacheDiagnostics, MetalError> {
        Ok(self.lock()?.diagnostics)
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, CacheState>, MetalError> {
        self.state.lock().map_err(|_| MetalError::StatePoisoned {
            state: "Metal coefficient upload cache",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jxr_core::{
        BandPresence, CoefficientPlane, MacroblockMetadata, PredictionMode, QuantizerSet,
        TileEdgeFlags,
    };

    #[test]
    fn retained_arena_identity_reuses_upload_buffers() {
        let device = j2k_metal_support::system_default_device().unwrap();
        let cache = CoefficientUploadCache::new(&device);
        let arena = Arc::new(CoefficientArena {
            coefficients: vec![1],
            macroblocks: MacroblockMetadata {
                coefficient_offsets: vec![0],
                quantizers: vec![QuantizerSet {
                    dc: 1,
                    low_pass: 1,
                    high_pass: 1,
                }],
                bands: vec![BandPresence::DcOnly],
                predictions: vec![PredictionMode::None],
                hp_predictions: vec![PredictionMode::None],
                tile_edges: vec![TileEdgeFlags::default()],
                coded_x: vec![0],
                coded_y: vec![0],
                output_x: vec![0],
                output_y: vec![0],
            },
            planes: vec![CoefficientPlane {
                coefficient_offset: 0,
                coefficient_count: 1,
                macroblock_offset: 0,
                macroblock_count: 1,
                block_columns: 4,
                block_rows: 4,
            }],
        });
        let first = cache.get_or_upload(&device, &arena).unwrap();
        let second = cache.get_or_upload(&device, &arena).unwrap();
        assert_eq!(
            Retained::as_ptr(&first.packed),
            Retained::as_ptr(&second.packed)
        );
        assert_eq!(cache.diagnostics().unwrap().hits, 1);
        assert_eq!(cache.diagnostics().unwrap().misses, 1);
    }
}
