// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    hint::black_box,
    path::PathBuf,
    time::{Duration, Instant},
};

use jxr::{
    BackendRequest, DecodeRequest, DecodedImage, JxrView,
    metal::{MetalDecoderSession, MetalStageDiagnostics},
};
use jxr_test_support::{
    BenchmarkFile, TimingSummary, checksum_samples, exit_on_error, milliseconds, summarize_timings,
    t834_suite, workspace_root,
};

type Workload = BenchmarkFile;

const DEFAULT_WARMUPS: usize = 3;
const DEFAULT_ITERATIONS: usize = 20;

struct Options {
    warmups: usize,
    iterations: usize,
    inputs: Vec<PathBuf>,
}

struct RouteMeasurements {
    first: Duration,
    warm: TimingSummary,
}

/// One Metal decode split into CPU preparation and the session's stage counters.
#[derive(Clone, Copy)]
struct MetalStageSample {
    total: Duration,
    stages: MetalStageDiagnostics,
}

impl MetalStageSample {
    /// Parse, planning, and CPU entropy: everything outside the session's stages.
    fn prepare(self) -> Duration {
        self.total
            .saturating_sub(self.stages.encode)
            .saturating_sub(self.stages.wait)
            .saturating_sub(self.stages.readback)
    }
}

fn main() {
    exit_on_error("jxr-load-bench", run());
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_options()?;
    let workloads = options
        .inputs
        .iter()
        .map(|path| BenchmarkFile::load(path))
        .collect::<Result<Vec<_>, _>>()?;

    let session_start = Instant::now();
    let session = MetalDecoderSession::system_default()?;
    let session_elapsed = session_start.elapsed();
    if !session.is_usable() {
        return Err("the default Metal device is not usable for JXR reconstruction".into());
    }

    println!("JXR full-decode CPU vs Metal host-readback benchmark");
    println!(
        "session startup: {:.3} ms; warmups: {}; measured iterations: {}",
        milliseconds(session_elapsed),
        options.warmups,
        options.iterations
    );
    println!("timed scope: in-memory parse + entropy + reconstruction + packing/readback");

    for workload in &workloads {
        benchmark_workload(workload, &session, &options)?;
    }
    Ok(())
}

fn benchmark_workload(
    workload: &Workload,
    session: &MetalDecoderSession,
    options: &Options,
) -> Result<(), Box<dyn std::error::Error>> {
    let cpu_request = workload.request(BackendRequest::Cpu);
    let metal_request = workload.request(BackendRequest::Metal);

    let (cpu_first, cpu_first_elapsed) = time_decode_cpu(&workload.source, &cpu_request)?;
    let (metal_first, metal_first_elapsed) =
        time_decode_metal(&workload.source, &metal_request, session)?;
    verify_equivalent(&cpu_first, &metal_first)?;
    let expected_checksum = checksum_samples(&cpu_first.samples);
    black_box(expected_checksum);

    for warmup in 0..options.warmups {
        if warmup % 2 == 0 {
            verify_checksum(
                &decode_cpu(&workload.source, &cpu_request)?,
                expected_checksum,
            )?;
            verify_checksum(
                &decode_metal(&workload.source, &metal_request, session)?,
                expected_checksum,
            )?;
        } else {
            verify_checksum(
                &decode_metal(&workload.source, &metal_request, session)?,
                expected_checksum,
            )?;
            verify_checksum(
                &decode_cpu(&workload.source, &cpu_request)?,
                expected_checksum,
            )?;
        }
    }

    let mut cpu_samples = Vec::with_capacity(options.iterations);
    let mut metal_samples = Vec::with_capacity(options.iterations);
    let mut metal_stages = Vec::with_capacity(options.iterations);
    for iteration in 0..options.iterations {
        if iteration % 2 == 0 {
            measure_cpu(workload, &cpu_request, expected_checksum, &mut cpu_samples)?;
            measure_metal(
                workload,
                &metal_request,
                session,
                expected_checksum,
                &mut metal_samples,
                &mut metal_stages,
            )?;
        } else {
            measure_metal(
                workload,
                &metal_request,
                session,
                expected_checksum,
                &mut metal_samples,
                &mut metal_stages,
            )?;
            measure_cpu(workload, &cpu_request, expected_checksum, &mut cpu_samples)?;
        }
    }

    let cpu = RouteMeasurements {
        first: cpu_first_elapsed,
        warm: summarize_timings(&cpu_samples).ok_or("CPU timing sample was empty")?,
    };
    let metal = RouteMeasurements {
        first: metal_first_elapsed,
        warm: summarize_timings(&metal_samples).ok_or("Metal timing sample was empty")?,
    };
    print_result(workload, &cpu_first, expected_checksum, &cpu, &metal);
    print_metal_stages(&metal_stages)?;
    Ok(())
}

fn measure_cpu(
    workload: &Workload,
    request: &DecodeRequest,
    expected_checksum: u64,
    samples: &mut Vec<Duration>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (decoded, elapsed) = time_decode_cpu(&workload.source, request)?;
    samples.push(elapsed);
    verify_checksum(&decoded, expected_checksum)
}

fn measure_metal(
    workload: &Workload,
    request: &DecodeRequest,
    session: &MetalDecoderSession,
    expected_checksum: u64,
    samples: &mut Vec<Duration>,
    stages: &mut Vec<MetalStageSample>,
) -> Result<(), Box<dyn std::error::Error>> {
    let before = session.stage_diagnostics()?;
    let (decoded, elapsed) = time_decode_metal(&workload.source, request, session)?;
    stages.push(MetalStageSample {
        total: elapsed,
        stages: session.stage_diagnostics()?.since(before),
    });
    samples.push(elapsed);
    verify_checksum(&decoded, expected_checksum)
}

fn time_decode_cpu(
    source: &[u8],
    request: &DecodeRequest,
) -> Result<(DecodedImage, Duration), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let decoded = decode_cpu(source, request)?;
    Ok((decoded, start.elapsed()))
}

fn time_decode_metal(
    source: &[u8],
    request: &DecodeRequest,
    session: &MetalDecoderSession,
) -> Result<(DecodedImage, Duration), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let decoded = decode_metal(source, request, session)?;
    Ok((decoded, start.elapsed()))
}

fn decode_cpu(
    source: &[u8],
    request: &DecodeRequest,
) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    let view = JxrView::parse(source)?;
    Ok(view.decoder().decode(request)?)
}

fn decode_metal(
    source: &[u8],
    request: &DecodeRequest,
    session: &MetalDecoderSession,
) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    let view = JxrView::parse(source)?;
    Ok(view.decoder().with_metal_session(session).decode(request)?)
}

fn verify_equivalent(cpu: &DecodedImage, metal: &DecodedImage) -> Result<(), String> {
    if cpu.info != metal.info
        || cpu.decoded_region != metal.decoded_region
        || cpu.format != metal.format
        || cpu.planes != metal.planes
        || cpu.samples != metal.samples
    {
        return Err("CPU and Metal decoded outputs are not exactly equivalent".to_owned());
    }
    Ok(())
}

fn verify_checksum(
    decoded: &DecodedImage,
    expected_checksum: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let actual = checksum_samples(&decoded.samples);
    black_box(actual);
    if actual != expected_checksum {
        return Err(format!(
            "decoded output checksum changed: expected {expected_checksum:016x}, got {actual:016x}"
        )
        .into());
    }
    Ok(())
}

fn print_result(
    workload: &Workload,
    decoded: &DecodedImage,
    checksum: u64,
    cpu: &RouteMeasurements,
    metal: &RouteMeasurements,
) {
    let speedup = cpu.warm.median.as_secs_f64() / metal.warm.median.as_secs_f64();
    let winner = if speedup > 1.0 { "Metal" } else { "CPU" };
    println!();
    println!(
        "{} — {}x{}, {} bytes, checksum {checksum:016x}",
        workload.path.display(),
        workload.width,
        workload.height,
        decoded.samples.byte_len()
    );
    print_route("CPU", cpu, workload.width, workload.height);
    print_route("Metal", metal, workload.width, workload.height);
    println!("winner: {winner}; Metal speedup at median: {speedup:.2}x");
}

fn print_metal_stages(samples: &[MetalStageSample]) -> Result<(), Box<dyn std::error::Error>> {
    let median = |stage: fn(&MetalStageSample) -> Duration| {
        summarize_timings(&samples.iter().map(stage).collect::<Vec<_>>())
            .map(|summary| milliseconds(summary.median))
            .ok_or("Metal stage sample was empty")
    };
    println!(
        "  Metal stage medians: prepare {:.3} | encode/submit {:.3} | wait {:.3} (GPU {:.3}) | readback {:.3} ms",
        median(|sample| sample.prepare())?,
        median(|sample| sample.stages.encode)?,
        median(|sample| sample.stages.wait)?,
        median(|sample| sample.stages.gpu)?,
        median(|sample| sample.stages.readback)?,
    );
    Ok(())
}

fn print_route(label: &str, measurements: &RouteMeasurements, width: u32, height: u32) {
    let megapixels = f64::from(width) * f64::from(height) / 1_000_000.0;
    let throughput = megapixels / measurements.warm.median.as_secs_f64();
    println!(
        "  {label:5} first {:8.3} ms | warm min {:8.3} | median {:8.3} | p95 {:8.3} | mean {:8.3} ms | {:8.2} MP/s",
        milliseconds(measurements.first),
        milliseconds(measurements.warm.minimum),
        milliseconds(measurements.warm.median),
        milliseconds(measurements.warm.p95),
        milliseconds(measurements.warm.mean),
        throughput
    );
}

fn parse_options() -> Result<Options, Box<dyn std::error::Error>> {
    let workspace = workspace_root()?;
    let mut warmups = DEFAULT_WARMUPS;
    let mut iterations = DEFAULT_ITERATIONS;
    let mut inputs = Vec::new();
    let mut arguments = std::env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--warmups") => {
                warmups = parse_count(arguments.next(), "--warmups")?;
            }
            Some("--iterations") => {
                iterations = parse_count(arguments.next(), "--iterations")?;
                if iterations == 0 {
                    return Err("--iterations must be greater than zero".into());
                }
            }
            Some("--help" | "-h") => {
                println!("usage: jxr-load-bench [--warmups N] [--iterations N] [image.jxr ...]");
                std::process::exit(0);
            }
            Some(value) if value.starts_with('-') => {
                return Err(format!("unknown option: {value}").into());
            }
            _ => inputs.push(PathBuf::from(argument)),
        }
    }
    if inputs.is_empty() {
        let suite = t834_suite(workspace);
        inputs = vec![
            suite.join("BasicAndOverlap_1x1Tile/Seattle_Spat_Ov2_1x1_YUV444_QP10.jxr"),
            suite.join("Entropy_Table_Coverage/Skyscraper_YONLY.jxr"),
            suite.join("Output_Color_Format_Baseline/Maui-16bppGray.jxr"),
        ];
    }
    Ok(Options {
        warmups,
        iterations,
        inputs,
    })
}

fn parse_count(
    value: Option<std::ffi::OsString>,
    option: &str,
) -> Result<usize, Box<dyn std::error::Error>> {
    let value = value.ok_or_else(|| format!("{option} requires a value"))?;
    let value = value
        .to_str()
        .ok_or_else(|| format!("{option} requires UTF-8 digits"))?;
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid {option} value {value:?}: {error}").into())
}
