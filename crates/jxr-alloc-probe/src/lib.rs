// SPDX-License-Identifier: MIT OR Apache-2.0

//! A counting global allocator for memory budgets of codec calls.
//!
//! A test binary installs [`CountingAllocator`] as its global allocator and
//! wraps the call it budgets in [`measure`], which reports the most heap
//! bytes that were live at once during the call, above what was live when it
//! began, and how many allocations the call made. The counts cover the whole
//! process, every thread of a thread pool included, which is what a codec
//! that spreads its work over a pool needs counted.
//!
//! The workspace forbids `unsafe`. Implementing `GlobalAlloc` is the one
//! thing this crate needs it for, so the probe lives in a crate of its own
//! that nothing but tests depends on.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::{
        Mutex, PoisonError,
        atomic::{AtomicUsize, Ordering},
    },
};

/// Working bytes a decode or an encode may hold in a thread pool, whatever
/// the image, beside its output: the 256 MiB its strips or bands are
/// admitted against, and room beside it for what each thread holds while it
/// reads or writes a tile.
pub const POOL_BYTES: usize = 320 << 20;

/// Heap bytes live now.
static LIVE: AtomicUsize = AtomicUsize::new(0);
/// Most heap bytes live at once since [`measure`] last reset it.
static PEAK: AtomicUsize = AtomicUsize::new(0);
/// Allocations and reallocations made since the process started.
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
/// Held for the length of one measurement: the counters are the process's.
static MEASURING: Mutex<()> = Mutex::new(());

/// The system allocator, counting what passes through it.
///
/// Install it with `#[global_allocator]` in the test binary that calls
/// [`measure`].
pub struct CountingAllocator;

fn allocated(size: usize) {
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    let live = LIVE.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

fn released(size: usize) {
    LIVE.fetch_sub(size, Ordering::Relaxed);
}

// SAFETY: every method forwards its arguments unchanged to `System`, whose
// contract is the one the caller upholds, and only updates counters beside
// the call.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract for `alloc`, passed through.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract for `alloc_zeroed`, passed through.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        released(layout.size());
        // SAFETY: the caller's contract for `dealloc`, passed through.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller's contract for `realloc`, passed through.
        let replacement = unsafe { System.realloc(pointer, layout, new_size) };
        if !replacement.is_null() {
            // Bytes retained at the same address are already counted. For
            // a move, conservatively allow both blocks to be live at once.
            let retained = if pointer == replacement {
                layout.size().min(new_size)
            } else {
                0
            };
            allocated(new_size - retained);
            released(layout.size() - retained);
        }
        replacement
    }
}

/// What one measured call asked of the heap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Usage {
    /// Most bytes live at once during the call, above those live when it
    /// began. Memory the call returns to its caller counts while the call
    /// runs. For a reallocation that moves, the peak includes both blocks
    /// as a conservative estimate of the allocator's transient usage.
    pub peak_bytes: usize,
    /// Allocations and reallocations the call made.
    pub allocations: usize,
}

/// Run `work` and report what it asked of the heap.
///
/// Measurements do not overlap: a second caller waits for the first. Other
/// threads that allocate during the call are counted with it, so a test
/// binary should do nothing else meanwhile.
///
/// # Panics
///
/// Panics when [`CountingAllocator`] is not the global allocator, since
/// every budget would then pass unmeasured.
pub fn measure<T>(work: impl FnOnce() -> T) -> (T, Usage) {
    let _measuring = MEASURING.lock().unwrap_or_else(PoisonError::into_inner);
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    drop(std::hint::black_box(Box::new(0_u64)));
    assert!(
        ALLOCATIONS.load(Ordering::Relaxed) > before,
        "CountingAllocator is not the global allocator of this binary"
    );
    let start = LIVE.load(Ordering::Relaxed);
    PEAK.store(start, Ordering::Relaxed);
    let calls = ALLOCATIONS.load(Ordering::Relaxed);
    let result = work();
    let usage = Usage {
        peak_bytes: PEAK.load(Ordering::Relaxed).saturating_sub(start),
        allocations: ALLOCATIONS.load(Ordering::Relaxed) - calls,
    };
    (result, usage)
}
