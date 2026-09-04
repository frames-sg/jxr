// SPDX-License-Identifier: MIT OR Apache-2.0

//! T.834 suite discovery, scope classification, and per-case execution.

use std::path::{Path, PathBuf};

use jxr::{JxrView, Profile};

use crate::{DifferentialResult, OracleError, T835Oracle, compare_file};

/// Why one official T.834 file is compared or excluded from the Main-profile run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum T834CaseExpectation {
    /// Annex-A JPEG XR file expected to use syntax in the project scope.
    CompareMainSyntax,
    /// Advanced output-format coverage outside the Main-profile target.
    SkipAdvancedSyntax,
    /// JPEG 2000 boxed wrapping, which is outside the Annex-A decoder.
    SkipJpeg2000Wrapper,
}

/// One file discovered in the official T.834 package.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct T834Case {
    /// Absolute or caller-rooted input path.
    pub input: PathBuf,
    /// Stable suite-relative path used in reports.
    pub relative_path: PathBuf,
    /// Scope decision made from the official suite category and extension.
    pub expectation: T834CaseExpectation,
}

/// Result of comparing one T.834 case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum T834CaseOutcome {
    /// Rust and T.835 produced byte-identical output.
    Passed(DifferentialResult),
    /// The case is explicitly outside the selected decoder scope.
    Skipped {
        /// Stable scope explanation.
        reason: &'static str,
    },
    /// The stream is in scope, but the comparison harness cannot represent its output yet.
    HarnessUnsupported {
        /// Stable harness limitation.
        reason: &'static str,
    },
    /// Parsing, decoding, reference execution, or byte comparison failed.
    Failed {
        /// Diagnostic retained for the report.
        message: String,
    },
}

/// Complete result for one official test file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct T834CaseResult {
    /// Discovered suite case.
    pub case: T834Case,
    /// Declared or inferred JPEG XR profile when parsing succeeded.
    pub profile: Option<Profile>,
    /// Comparison disposition.
    pub outcome: T834CaseOutcome,
}

/// Aggregate counts for one conformance run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct T834Summary {
    /// Byte-identical comparisons.
    pub passed: usize,
    /// Explicitly out-of-scope files.
    pub skipped: usize,
    /// In-scope files blocked by a comparison-format limitation.
    pub harness_unsupported: usize,
    /// In-scope failures.
    pub failed: usize,
}

impl T834Summary {
    /// Add one case outcome to the aggregate.
    pub fn observe(&mut self, outcome: &T834CaseOutcome) {
        match outcome {
            T834CaseOutcome::Passed(_) => self.passed += 1,
            T834CaseOutcome::Skipped { .. } => self.skipped += 1,
            T834CaseOutcome::HarnessUnsupported { .. } => self.harness_unsupported += 1,
            T834CaseOutcome::Failed { .. } => self.failed += 1,
        }
    }

    /// Total number of classified files.
    #[must_use]
    pub const fn total(self) -> usize {
        self.passed + self.skipped + self.harness_unsupported + self.failed
    }
}

/// Discover every `.jxr` and `.jpx` file below an extracted T.834 suite root.
pub fn discover_t834_cases(root: &Path) -> Result<Vec<T834Case>, OracleError> {
    let mut pending = vec![root.to_owned()];
    let mut cases = Vec::new();
    while let Some(directory) = pending.pop() {
        let entries = std::fs::read_dir(&directory).map_err(|source| OracleError::Io {
            operation: "read T.834 suite directory",
            path: directory.clone(),
            source,
        })?;
        for entry in entries {
            let entry = entry.map_err(|source| OracleError::Io {
                operation: "read T.834 suite entry",
                path: directory.clone(),
                source,
            })?;
            let path = entry.path();
            let file_type = entry.file_type().map_err(|source| OracleError::Io {
                operation: "inspect T.834 suite entry",
                path: path.clone(),
                source,
            })?;
            if file_type.is_dir() {
                pending.push(path);
                continue;
            }
            let relative_path = path
                .strip_prefix(root)
                .expect("discovered path remains below the suite root")
                .to_owned();
            if let Some(expectation) = classify_case(&relative_path) {
                cases.push(T834Case {
                    input: path,
                    relative_path,
                    expectation,
                });
            }
        }
    }
    cases.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    Ok(cases)
}

/// Compare one in-scope case with the portable CPU decoder.
#[must_use]
pub fn run_t834_cpu_case(oracle: &T835Oracle, case: T834Case) -> T834CaseResult {
    run_case(case, |input| compare_file(oracle, input))
}

/// Compare one in-scope case with a strict Metal session.
#[cfg(feature = "metal")]
#[must_use]
pub fn run_t834_metal_case(
    oracle: &T835Oracle,
    session: &jxr::metal::MetalDecoderSession,
    case: T834Case,
) -> T834CaseResult {
    run_case(case, |input| {
        crate::compare_file_metal(oracle, input, session)
    })
}

/// Compare one in-scope case with a strict CUDA session.
#[cfg(feature = "cuda")]
#[must_use]
pub fn run_t834_cuda_case(
    oracle: &T835Oracle,
    session: &jxr::cuda::CudaDecoderSession,
    case: T834Case,
) -> T834CaseResult {
    run_case(case, |input| {
        crate::compare_file_cuda(oracle, input, session)
    })
}

fn run_case(
    case: T834Case,
    compare: impl FnOnce(&Path) -> Result<DifferentialResult, OracleError>,
) -> T834CaseResult {
    let skipped = match case.expectation {
        T834CaseExpectation::CompareMainSyntax => None,
        T834CaseExpectation::SkipAdvancedSyntax => {
            Some("T.834 Advanced output-format category is outside the Main-profile target")
        }
        T834CaseExpectation::SkipJpeg2000Wrapper => {
            Some("JPEG 2000 boxed wrapping is outside the Annex-A decoder")
        }
    };
    if let Some(reason) = skipped {
        return T834CaseResult {
            case,
            profile: None,
            outcome: T834CaseOutcome::Skipped { reason },
        };
    }
    let profile = inspect_profile(&case.input);
    let outcome = match compare(&case.input) {
        Ok(result) => T834CaseOutcome::Passed(result),
        Err(OracleError::UnsupportedFormat { reason }) => {
            T834CaseOutcome::HarnessUnsupported { reason }
        }
        Err(error) => T834CaseOutcome::Failed {
            message: error.to_string(),
        },
    };
    T834CaseResult {
        case,
        profile,
        outcome,
    }
}

fn inspect_profile(input: &Path) -> Option<Profile> {
    let source = std::fs::read(input).ok()?;
    JxrView::parse(&source).ok()?.info().profile
}

fn classify_case(relative_path: &Path) -> Option<T834CaseExpectation> {
    let extension = relative_path.extension()?.to_str()?;
    if extension.eq_ignore_ascii_case("jpx") {
        return Some(T834CaseExpectation::SkipJpeg2000Wrapper);
    }
    if !extension.eq_ignore_ascii_case("jxr") {
        return None;
    }
    let category = relative_path.components().next()?.as_os_str().to_str()?;
    if category == "Output_Color_Format_Advanced" {
        Some(T834CaseExpectation::SkipAdvancedSyntax)
    } else {
        Some(T834CaseExpectation::CompareMainSyntax)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_scope_classification_is_explicit() {
        assert_eq!(
            classify_case(Path::new("BasicAndOverlap_1x1Tile/a.jxr")),
            Some(T834CaseExpectation::CompareMainSyntax)
        );
        assert_eq!(
            classify_case(Path::new("Output_Color_Format_Advanced/a.jxr")),
            Some(T834CaseExpectation::SkipAdvancedSyntax)
        );
        assert_eq!(
            classify_case(Path::new("BoxedBased_Format/a.jpx")),
            Some(T834CaseExpectation::SkipJpeg2000Wrapper)
        );
        assert_eq!(
            classify_case(Path::new("BasicAndOverlap_1x1Tile/Thumbs.db")),
            None
        );
    }

    #[test]
    fn summary_keeps_failures_separate_from_scope_and_harness_gaps() {
        let mut summary = T834Summary::default();
        summary.observe(&T834CaseOutcome::Passed(DifferentialResult {
            format: jxr::PixelFormat::U8(jxr::ChannelLayout::Luma),
            width: 1,
            height: 1,
            byte_len: 1,
        }));
        summary.observe(&T834CaseOutcome::Skipped { reason: "scope" });
        summary.observe(&T834CaseOutcome::HarnessUnsupported { reason: "format" });
        summary.observe(&T834CaseOutcome::Failed {
            message: "decode".into(),
        });
        assert_eq!(summary.total(), 4);
        assert_eq!(
            (
                summary.passed,
                summary.skipped,
                summary.harness_unsupported,
                summary.failed
            ),
            (1, 1, 1, 1)
        );
    }
}
