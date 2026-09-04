// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::OracleError;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Raw bytes and diagnostics produced by one successful T.835 decode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OracleRawOutput {
    /// Combined raw primary/alpha representation selected by the Annex-A GUID.
    pub bytes: Vec<u8>,
    /// Non-fatal reference-software diagnostics.
    pub stderr: String,
}

/// Highest profile declaration accepted by the T.835 reference process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum T835ProfileLimit {
    /// Reject streams declaring a profile above Main.
    Main,
    /// Permit an Advanced declaration so Main syntax can still be compared.
    Advanced,
}

impl T835ProfileLimit {
    const fn argument(self) -> &'static str {
        match self {
            Self::Main => "66",
            Self::Advanced => "111",
        }
    }
}

/// Pinned external T.835 reference executable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct T835Oracle {
    executable: PathBuf,
    profile_limit: T835ProfileLimit,
}

impl T835Oracle {
    /// Bind an explicitly selected reference executable.
    #[must_use]
    pub fn new(executable: impl Into<PathBuf>) -> Self {
        Self {
            executable: executable.into(),
            profile_limit: T835ProfileLimit::Main,
        }
    }

    /// Set the declaration ceiling enforced by the reference process.
    ///
    /// This does not change the syntax accepted by the Rust decoder. T.834
    /// validation uses [`T835ProfileLimit::Advanced`] because some official
    /// streams declare Advanced while exercising syntax already in project scope.
    #[must_use]
    pub const fn with_profile_limit(mut self, profile_limit: T835ProfileLimit) -> Self {
        self.profile_limit = profile_limit;
        self
    }

    /// Select `JXR_T835_ORACLE`, or the repository build-script output.
    #[must_use]
    pub fn for_workspace(workspace: &Path) -> Self {
        std::env::var_os("JXR_T835_ORACLE").map_or_else(
            || Self::new(workspace.join("target/t835-oracle/t835-201201/Software/jpegxr")),
            Self::new,
        )
    }

    /// Reference executable path.
    #[must_use]
    pub fn executable(&self) -> &Path {
        &self.executable
    }

    /// Decode one Annex-A `.jxr` image into the reference raw representation.
    pub fn decode_raw(&self, input: &Path) -> Result<OracleRawOutput, OracleError> {
        let input = std::fs::canonicalize(input)
            .map_err(|source| io("canonicalize input", input, source))?;
        let output_dir = OracleTempDir::create()?;
        let raw_path = output_dir.path.join("decoded.raw");
        let result = Command::new(&self.executable)
            .arg("-P")
            .arg(self.profile_limit.argument())
            .arg("-L")
            .arg("255")
            .arg("-o")
            .arg(&raw_path)
            .arg(input)
            .output()
            .map_err(|source| io("execute T.835 decoder", &self.executable, source))?;
        let stderr = String::from_utf8_lossy(&result.stderr).into_owned();
        if !result.status.success() {
            return Err(OracleError::ProcessFailed {
                status: result.status.code(),
                stderr,
            });
        }
        let bytes = std::fs::read(&raw_path).map_err(|source| {
            if source.kind() == std::io::ErrorKind::NotFound {
                OracleError::MissingOutput {
                    path: raw_path.clone(),
                }
            } else {
                io("read T.835 raw output", &raw_path, source)
            }
        })?;
        Ok(OracleRawOutput { bytes, stderr })
    }
}

struct OracleTempDir {
    path: PathBuf,
}

impl OracleTempDir {
    fn create() -> Result<Self, OracleError> {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("jxr-t835-{}-{sequence}", std::process::id()));
        std::fs::create_dir(&path)
            .map_err(|source| io("create oracle output directory", &path, source))?;
        Ok(Self { path })
    }
}

impl Drop for OracleTempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn io(operation: &'static str, path: &Path, source: std::io::Error) -> OracleError {
    OracleError::Io {
        operation,
        path: path.to_owned(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_limit_uses_t835_profile_identifiers() {
        assert_eq!(T835ProfileLimit::Main.argument(), "66");
        assert_eq!(T835ProfileLimit::Advanced.argument(), "111");
    }

    #[test]
    fn oracle_defaults_to_main_and_can_raise_only_its_reference_ceiling() {
        let oracle = T835Oracle::new("jpegxr");
        assert_eq!(oracle.profile_limit, T835ProfileLimit::Main);
        assert_eq!(
            oracle
                .with_profile_limit(T835ProfileLimit::Advanced)
                .profile_limit,
            T835ProfileLimit::Advanced
        );
    }
}
