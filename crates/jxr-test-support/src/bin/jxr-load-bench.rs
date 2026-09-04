// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    hint::black_box,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use jxr::{
    AlphaHandling, BackendRequest, DecodeRequest, DecodedImage, JxrView, PixelFormat,
    metal::MetalDecoderSession,
};
use jxr_test_support::{TimingSummary, checksum_samples, oracle_format, summarize_timings};

const DEFAULT_WARMUPS: usize = 3;
const DEFAULT_ITERATIONS: usize = 20;

struct Options {
    warmups: usize,
    iterations: usize,
    inputs: Vec<PathBuf>,
}

struct Workload {
    path: PathBuf,
    source: Vec<u8>,
    format: PixelFormat,
    alpha: AlphaHandling,
    width: u32,
    height: u32,
}

struct RouteMeasurements {
    first: Duration,
    warm: TimingSummary,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("jxr-load-bench: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_options()?;
    let workloads = options
        .inputs
        .iter()
        .map(|path| load_workload(path))
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
    let cpu_request = request(workload, BackendRequest::Cpu);
    let metal_request = request(workload, BackendRequest::Metal);

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
    for iteration in 0..options.iterations {
        if iteration % 2 == 0 {
            measure_cpu(workload, &cpu_request, expected_checksum, &mut cpu_samples)?;
            measure_metal(
                workload,
                &metal_request,
                session,
                expected_checksum,
                &mut metal_samples,
            )?;
        } else {
            measure_metal(
                workload,
                &metal_request,
                session,
                expected_checksum,
                &mut metal_samples,
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
) -> Result<(), Box<dyn std::error::Error>> {
    let (decoded, elapsed) = time_decode_metal(&workload.source, request, session)?;
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

fn request(workload: &Workload, backend: BackendRequest) -> DecodeRequest {
    DecodeRequest::new(workload.format)
        .with_alpha(workload.alpha)
        .with_backend(backend)
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

fn load_workload(path: &Path) -> Result<Workload, Box<dyn std::error::Error>> {
    let source = std::fs::read(path)?;
    let view = JxrView::parse(&source)?;
    let format = oracle_format(view.info())?;
    let width = view.info().width;
    let height = view.info().height;
    Ok(Workload {
        path: path.to_owned(),
        source,
        format: format.pixel_format,
        alpha: format.alpha,
        width,
        height,
    })
}

fn parse_options() -> Result<Options, Box<dyn std::error::Error>> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("test-support crate is outside its workspace")?;
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
        let suite = workspace.join("target/t834-conformance/suite-2014");
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

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
