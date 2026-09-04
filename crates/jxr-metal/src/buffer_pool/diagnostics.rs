// SPDX-License-Identifier: MIT OR Apache-2.0

/// Retention and high-water counters for one Metal scratch pool.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct MetalBufferPoolDiagnostics {
    pub cached_bytes: usize,
    pub cached_buffers: usize,
    pub peak_cached_bytes: usize,
    pub peak_cached_buffers: usize,
    pub evictions: usize,
    pub rejections: usize,
}

/// Separate diagnostics for private and shared Metal scratch storage.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct MetalBufferPoolsDiagnostics {
    pub private: MetalBufferPoolDiagnostics,
    pub shared: MetalBufferPoolDiagnostics,
}
