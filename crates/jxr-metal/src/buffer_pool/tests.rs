// SPDX-License-Identifier: MIT OR Apache-2.0

use super::state::{PoolLimits, PoolState};
use j2k_metal_support::checked_private_buffer;

fn buffer(bytes: usize) -> super::state::PooledBuffer {
    let device = j2k_metal_support::system_default_device().expect("Metal device");
    super::state::PooledBuffer::new(
        bytes,
        checked_private_buffer(&device, bytes).expect("private allocation"),
    )
    .expect("matching allocation")
}

#[test]
fn exact_size_take_and_fifo_eviction_preserve_accounting() {
    let mut state = PoolState::new(PoolLimits::new(12, 2));
    state.recycle(buffer(4)).unwrap();
    state.recycle(buffer(8)).unwrap();
    state.recycle(buffer(6)).unwrap();

    assert_eq!(state.diagnostics().cached_bytes, 6);
    assert_eq!(state.diagnostics().evictions, 2);
    assert!(state.take(8).unwrap().is_none());
    assert_eq!(state.take(6).unwrap().unwrap().bytes(), 6);
    assert_eq!(state.diagnostics().cached_bytes, 0);
}

#[test]
fn allocation_larger_than_limit_is_rejected_without_retention() {
    let mut state = PoolState::new(PoolLimits::new(4, 2));
    state.recycle(buffer(8)).unwrap();
    assert_eq!(state.diagnostics().cached_buffers, 0);
    assert_eq!(state.diagnostics().rejections, 1);
}
