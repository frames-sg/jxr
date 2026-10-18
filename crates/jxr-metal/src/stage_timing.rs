// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use core::cell::Cell;
use core::time::Duration;
#[cfg(target_os = "macos")]
use std::time::Instant;

#[cfg(target_os = "macos")]
use objc2_metal::MTLCommandBuffer;

/// Cumulative host and GPU time spent in each Metal submission stage.
///
/// Counters cover every reconstruction submitted through one session and its
/// clones since runtime initialization. Subtract an earlier snapshot with
/// [`Self::since`] to attribute one call.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct MetalStageDiagnostics {
    /// Host time building descriptors, binding resources, encoding, and committing.
    pub encode: Duration,
    /// Host time blocked on reconstruction command completion.
    pub wait: Duration,
    /// Summed GPU execution time of completed reconstruction command buffers.
    ///
    /// Commands on concurrent batch queues overlap, so a batch sum can exceed
    /// its wall-clock GPU span.
    pub gpu: Duration,
    /// Host time copying completed outputs into caller-owned memory.
    pub readback: Duration,
    /// Completed reconstruction command buffers.
    pub commands: u64,
}

impl MetalStageDiagnostics {
    /// Counter growth since `earlier`, saturating if the snapshots are reversed.
    #[must_use]
    pub fn since(self, earlier: Self) -> Self {
        Self {
            encode: self.encode.saturating_sub(earlier.encode),
            wait: self.wait.saturating_sub(earlier.wait),
            gpu: self.gpu.saturating_sub(earlier.gpu),
            readback: self.readback.saturating_sub(earlier.readback),
            commands: self.commands.saturating_sub(earlier.commands),
        }
    }
}

#[cfg(target_os = "macos")]
#[derive(Default)]
pub(crate) struct StageTimings {
    encode: Cell<Duration>,
    wait: Cell<Duration>,
    gpu: Cell<Duration>,
    readback: Cell<Duration>,
    commands: Cell<u64>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(crate) enum HostStage {
    Encode,
    Readback,
}

#[cfg(target_os = "macos")]
impl StageTimings {
    pub(crate) fn time<T>(&self, stage: HostStage, run: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let result = run();
        let cell = match stage {
            HostStage::Encode => &self.encode,
            HostStage::Readback => &self.readback,
        };
        add(cell, start.elapsed());
        result
    }

    /// Records one completion wait and, for the command's owner, its GPU span.
    pub(crate) fn record_wait(
        &self,
        waited: Duration,
        command: Option<&objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>>,
    ) {
        add(&self.wait, waited);
        if let Some(command) = command {
            let seconds = command.GPUEndTime() - command.GPUStartTime();
            if seconds.is_finite() && seconds > 0.0 {
                add(&self.gpu, Duration::from_secs_f64(seconds));
            }
            self.commands.set(self.commands.get().saturating_add(1));
        }
    }

    pub(crate) fn snapshot(&self) -> MetalStageDiagnostics {
        MetalStageDiagnostics {
            encode: self.encode.get(),
            wait: self.wait.get(),
            gpu: self.gpu.get(),
            readback: self.readback.get(),
            commands: self.commands.get(),
        }
    }
}

#[cfg(target_os = "macos")]
fn add(cell: &Cell<Duration>, elapsed: Duration) {
    cell.set(cell.get().saturating_add(elapsed));
}
