// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    ffi::OsString,
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use jxr_test_support::{
    T834Case, T834CaseExpectation, T834CaseOutcome, T834CaseResult, T834Summary, T835Oracle,
    T835ProfileLimit, discover_t834_cases, run_t834_cpu_case,
};

static REWRAP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Cpu,
    Metal,
    Cuda,
}

struct BackendSessions<'a> {
    #[cfg(feature = "metal")]
    metal: Option<&'a jxr::metal::MetalDecoderSession>,
    #[cfg(feature = "cuda")]
    cuda: Option<&'a jxr::cuda::CudaDecoderSession>,
    marker: core::marker::PhantomData<&'a ()>,
}

struct Options {
    backend: Backend,
    root: PathBuf,
    report: PathBuf,
    category: Option<OsString>,
    limit: Option<usize>,
    rewrap: bool,
    verbose: bool,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("jxr-t834: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("test-support crate is outside its workspace")?;
    let options = parse_options(workspace)?;
    let oracle =
        T835Oracle::for_workspace(workspace).with_profile_limit(T835ProfileLimit::Advanced);
    let mut cases = discover_t834_cases(&options.root)?;
    if let Some(category) = &options.category {
        cases.retain(|case| {
            case.relative_path
                .components()
                .next()
                .is_some_and(|component| component.as_os_str() == category)
        });
    }
    if let Some(limit) = options.limit {
        cases.truncate(limit);
    }
    if cases.is_empty() {
        return Err("no T.834 cases matched the selected root and category".into());
    }
    if let Some(parent) = options.report.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut report = BufWriter::new(File::create(&options.report)?);
    writeln!(report, "status\tprofile\tbytes\tpath\tdiagnostic")?;
    let mut summary = T834Summary::default();

    #[cfg(feature = "metal")]
    let metal_session = (options.backend == Backend::Metal)
        .then(jxr::metal::MetalDecoderSession::system_default)
        .transpose()?;
    #[cfg(not(feature = "metal"))]
    if options.backend == Backend::Metal {
        return Err("Metal requires `--features metal`".into());
    }
    #[cfg(feature = "cuda")]
    let cuda_session = (options.backend == Backend::Cuda)
        .then(jxr::cuda::CudaDecoderSession::system_default)
        .transpose()?;
    #[cfg(not(feature = "cuda"))]
    if options.backend == Backend::Cuda {
        return Err("CUDA requires `--features cuda`".into());
    }
    let sessions = BackendSessions {
        #[cfg(feature = "metal")]
        metal: metal_session.as_ref(),
        #[cfg(feature = "cuda")]
        cuda: cuda_session.as_ref(),
        marker: core::marker::PhantomData,
    };

    for case in cases {
        let result = execute_case(&oracle, case, options.backend, &sessions, options.rewrap)?;
        summary.observe(&result.outcome);
        write_result(&mut report, &result)?;
        print_result(&result, options.verbose);
    }
    report.flush()?;
    println!(
        "summary: {} passed, {} skipped, {} harness-unsupported, {} failed; report {}",
        summary.passed,
        summary.skipped,
        summary.harness_unsupported,
        summary.failed,
        options.report.display()
    );
    if summary.failed != 0 {
        return Err(format!("{} in-scope T.834 cases failed", summary.failed).into());
    }
    Ok(())
}

fn execute_case(
    oracle: &T835Oracle,
    case: T834Case,
    backend: Backend,
    sessions: &BackendSessions<'_>,
    rewrap: bool,
) -> Result<T834CaseResult, Box<dyn std::error::Error>> {
    execute_case_with_input(case, rewrap, |case| match backend {
        Backend::Cpu => Ok(run_t834_cpu_case(oracle, case)),
        Backend::Metal => execute_metal_case(oracle, sessions, case),
        Backend::Cuda => execute_cuda_case(oracle, sessions, case),
    })
}

#[cfg(feature = "metal")]
fn execute_metal_case(
    oracle: &T835Oracle,
    sessions: &BackendSessions<'_>,
    case: T834Case,
) -> Result<T834CaseResult, Box<dyn std::error::Error>> {
    Ok(jxr_test_support::run_t834_metal_case(
        oracle,
        sessions.metal.ok_or("missing strict Metal session")?,
        case,
    ))
}

#[cfg(not(feature = "metal"))]
fn execute_metal_case(
    _: &T835Oracle,
    _: &BackendSessions<'_>,
    _: T834Case,
) -> Result<T834CaseResult, Box<dyn std::error::Error>> {
    Err("Metal requires `--features metal`".into())
}

#[cfg(feature = "cuda")]
fn execute_cuda_case(
    oracle: &T835Oracle,
    sessions: &BackendSessions<'_>,
    case: T834Case,
) -> Result<T834CaseResult, Box<dyn std::error::Error>> {
    Ok(jxr_test_support::run_t834_cuda_case(
        oracle,
        sessions.cuda.ok_or("missing strict CUDA session")?,
        case,
    ))
}

#[cfg(not(feature = "cuda"))]
fn execute_cuda_case(
    _: &T835Oracle,
    _: &BackendSessions<'_>,
    _: T834Case,
) -> Result<T834CaseResult, Box<dyn std::error::Error>> {
    Err("CUDA requires `--features cuda`".into())
}

fn execute_case_with_input(
    case: T834Case,
    rewrap: bool,
    execute: impl FnOnce(T834Case) -> Result<T834CaseResult, Box<dyn std::error::Error>>,
) -> Result<T834CaseResult, Box<dyn std::error::Error>> {
    if !rewrap || case.expectation != T834CaseExpectation::CompareMainSyntax {
        return execute(case);
    }
    let original = case.clone();
    let input = match RewrappedInput::create(&case.input) {
        Ok(input) => input,
        Err(error) => {
            return Ok(T834CaseResult {
                case,
                profile: None,
                outcome: T834CaseOutcome::Failed {
                    message: format!("Annex-A rewrap failed: {error}"),
                },
            });
        }
    };
    let mut rewritten = case;
    rewritten.input.clone_from(&input.path);
    let mut result = execute(rewritten)?;
    result.case = original;
    Ok(result)
}

struct RewrappedInput {
    path: PathBuf,
}

impl RewrappedInput {
    fn create(input: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let source = std::fs::read(input)?;
        let parsed = jxr_native::parse_codestream(&source)?;
        let annex = parsed
            .annex_a
            .as_ref()
            .ok_or("T.834 rewrap input is not Annex-A")?;
        let info = jxr_native::image_info(&parsed)?;
        let primary = &source[annex.codestream_range.clone()];
        let mut options =
            jxr::AnnexAWriteOptions::new(annex.width, annex.height, annex.pixel_format_guid)
                .with_orientation(info.metadata.orientation);
        if let Some([horizontal, vertical]) = annex.metadata.resolution_dpi_bits {
            options =
                options.with_resolution_dpi(f32::from_bits(horizontal), f32::from_bits(vertical));
        }
        if let Some(range) = annex.metadata.icc_profile_range.clone() {
            options = options.with_icc_profile(&source[range]);
        }
        if let Some(range) = annex.alpha_range.clone() {
            options = options.with_separate_alpha(&source[range]);
        }
        let output = jxr::write_annex_a(primary, &options)?;
        let sequence = REWRAP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "jxr-t834-rewrap-{}-{sequence}.jxr",
            std::process::id()
        ));
        let input = Self { path };
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&input.path)?;
        file.write_all(&output)?;
        Ok(input)
    }
}

impl Drop for RewrappedInput {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn parse_options(workspace: &Path) -> Result<Options, Box<dyn std::error::Error>> {
    let mut backend = Backend::Cpu;
    let mut root = workspace.join("target/t834-conformance/suite-2014");
    let mut report = None;
    let mut category = None;
    let mut limit = None;
    let mut rewrap = false;
    let mut verbose = false;
    let mut arguments = std::env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--backend") => {
                backend = match arguments
                    .next()
                    .and_then(|value| value.into_string().ok())
                    .as_deref()
                {
                    Some("cpu") => Backend::Cpu,
                    Some("metal") => Backend::Metal,
                    Some("cuda") => Backend::Cuda,
                    _ => return Err("--backend must be `cpu`, `metal`, or `cuda`".into()),
                };
            }
            Some("--root") => root = PathBuf::from(arguments.next().ok_or("--root needs a path")?),
            Some("--report") => {
                report = Some(PathBuf::from(
                    arguments.next().ok_or("--report needs a path")?,
                ));
            }
            Some("--category") => {
                category = Some(arguments.next().ok_or("--category needs a name")?);
            }
            Some("--limit") => {
                let value = arguments
                    .next()
                    .and_then(|value| value.into_string().ok())
                    .ok_or("--limit needs an integer")?;
                limit = Some(value.parse()?);
            }
            Some("--rewrap") => rewrap = true,
            Some("--verbose") => verbose = true,
            Some("--help") => {
                println!(
                    "usage: jxr-t834 [--backend cpu|metal|cuda] [--root PATH] [--category NAME] [--limit N] [--report PATH] [--rewrap] [--verbose]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {}", argument.to_string_lossy()).into()),
        }
    }
    let route = match backend {
        Backend::Cpu => "cpu",
        Backend::Metal => "metal",
        Backend::Cuda => "cuda",
    };
    Ok(Options {
        backend,
        root,
        report: report.unwrap_or_else(|| {
            workspace.join(format!("target/t834-conformance/reports/{route}.tsv"))
        }),
        category,
        limit,
        rewrap,
        verbose,
    })
}

fn write_result(report: &mut impl Write, result: &T834CaseResult) -> Result<(), std::io::Error> {
    let profile = result
        .profile
        .map_or_else(|| "-".to_owned(), |profile| format!("{profile:?}"));
    let (status, bytes, diagnostic) = outcome_fields(&result.outcome);
    writeln!(
        report,
        "{status}\t{profile}\t{bytes}\t{}\t{}",
        result.case.relative_path.display(),
        sanitize(diagnostic)
    )
}

fn outcome_fields(outcome: &T834CaseOutcome) -> (&'static str, usize, &str) {
    match outcome {
        T834CaseOutcome::Passed(result) => ("pass", result.byte_len, ""),
        T834CaseOutcome::Skipped { reason } => ("skip", 0, reason),
        T834CaseOutcome::HarnessUnsupported { reason } => ("harness-unsupported", 0, reason),
        T834CaseOutcome::Failed { message } => ("fail", 0, message),
    }
}

fn print_result(result: &T834CaseResult, verbose: bool) {
    let (status, _, diagnostic) = outcome_fields(&result.outcome);
    if verbose || matches!(result.outcome, T834CaseOutcome::Failed { .. }) {
        if diagnostic.is_empty() {
            println!("{status}: {}", result.case.relative_path.display());
        } else {
            println!(
                "{status}: {}: {diagnostic}",
                result.case.relative_path.display()
            );
        }
    }
}

fn sanitize(value: &str) -> String {
    value.replace(['\t', '\r', '\n'], " ")
}
