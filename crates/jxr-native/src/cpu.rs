//! CPU feature selection retained by a decoder session.

use fearless_simd::Level;

/// Runtime CPU capabilities detected once when a decoder session is created.
#[derive(Clone, Copy, Debug)]
pub struct CpuCapabilities {
    level: Level,
}

impl CpuCapabilities {
    /// Detect the strongest instruction set supported by the current process.
    #[must_use]
    pub fn detect() -> Self {
        Self {
            level: Level::new(),
        }
    }

    pub(crate) const fn level(self) -> Level {
        self.level
    }

    pub(crate) fn accelerates_i32(self) -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if self.level.as_avx2().is_some() {
            return true;
        }
        #[cfg(target_arch = "aarch64")]
        if self.level.as_neon().is_some() {
            return true;
        }
        false
    }
}
