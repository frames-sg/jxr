// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    hint::black_box,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use jxr::{
    AlphaHandling, BackendRequest, BatchDecodeOptions, CpuBatchDecoder, DecodeRequest,
    DecodedImage, EncodedImage, JxrView, PixelFormat,
    metal::{MetalDecodePlan, MetalDecoderSession},
};
use jxr_test_support::{
    checksum_bytes, checksum_cpu_batch_image, checksum_samples, oracle_format, summarize_timings,
};
use rayon::prelude::*;

const BATCH_SIZES: [usize; 5] = [1, 8, 32, 64, 128];
const WARMUPS: usize = 2;
const ITERATIONS: usize = 10;
const PIPELINE_BATCHES: usize = 4;

type BenchmarkError = Box<dyn std::error::Error + Send + Sync>;

struct Workload {
    path: PathBuf,
    source: Arc<[u8]>,
    format: PixelFormat,
    alpha: AlphaHandling,
    width: u32,
    height: u32,
}

#[derive(Clone, Copy)]
enum MetalOutput {
    Resident,
    Dense,
    Shared,
    Host,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("jxr-pathology-bench: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), BenchmarkError> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("test-support crate is outside its workspace")?;
    let path = std::env::args_os().nth(1).map_or_else(
        || workspace.join("target/t834-conformance/suite-2014/MBLevel_QP_Coverage/Boat_MBQP7.jxr"),
        PathBuf::from,
    );
    let workload = load_workload(&path)?;
    let cpu = CpuBatchDecoder::new(BatchDecodeOptions::default())?;
    let session = MetalDecoderSession::system_default()?;
    if !session.is_usable() {
        return Err("the default Metal device is not usable for JXR reconstruction".into());
    }

    let reference = decode_cpu(&workload)?;
    let checksum = checksum_samples(&reference.samples);
    validate_metal_batch(
        &workload,
        &session,
        BATCH_SIZES[BATCH_SIZES.len() - 1],
        checksum,
    )?;
    warm_pipeline(&workload, &session)?;

    println!("JXR pathology-style 256px tile batch benchmark");
    println!(
        "{} — {}x{}, {} output bytes/tile, checksum {checksum:016x}",
        workload.path.display(),
        workload.width,
        workload.height,
        reference.samples.byte_len()
    );
    #[cfg(target_os = "macos")]
    println!(
        "{} Rayon workers; {} Metal queues; {WARMUPS} warmups; {ITERATIONS} alternating measured iterations",
        cpu.worker_count(),
        session.batch_queue_count()?
    );
    println!("timed scope includes parse + CPU entropy + reconstruction; file I/O is excluded");
    println!("resident ends in private Metal output; host writes directly to pooled shared output");

    for batch_size in BATCH_SIZES {
        let result = measure_batch(&workload, &cpu, &session, batch_size, checksum)?;
        print_result(&workload, batch_size, &result);
    }
    let diagnostics = cpu.diagnostics();
    println!();
    println!(
        "CPU batch diagnostics: {} prepared inputs, {} cache hits, {} misses, {} direct dense images, {} coefficient workspace reuses, {} retained coefficient bytes, {} reconstruction workspace reuses, {} retained reconstruction bytes, {} layout workspace reuses, {} retained layout bytes",
        diagnostics.prepared_inputs,
        diagnostics.preparation_cache_hits,
        diagnostics.preparation_cache_misses,
        diagnostics.direct_dense_images,
        diagnostics.coefficient_workspace_reuses,
        diagnostics.retained_coefficient_bytes,
        diagnostics.reconstruction_workspace_reuses,
        diagnostics.retained_reconstruction_bytes,
        diagnostics.layout_workspace_reuses,
        diagnostics.retained_layout_bytes,
    );
    Ok(())
}

struct BatchResult {
    cpu: jxr_test_support::TimingSummary,
    resident: MetalSummary,
    dense: MetalSummary,
    shared: MetalSummary,
    host: MetalSummary,
    pipelined_resident: jxr_test_support::TimingSummary,
}

struct MetalMeasurement {
    total: Duration,
    prepare: Duration,
    execute: Duration,
}

struct MetalSummary {
    total: jxr_test_support::TimingSummary,
    prepare: jxr_test_support::TimingSummary,
    execute: jxr_test_support::TimingSummary,
}

fn measure_batch(
    workload: &Workload,
    cpu_decoder: &CpuBatchDecoder,
    session: &MetalDecoderSession,
    batch_size: usize,
    expected_checksum: u64,
) -> Result<BatchResult, BenchmarkError> {
    for _ in 0..WARMUPS {
        warm_batch_routes(
            workload,
            cpu_decoder,
            session,
            batch_size,
            expected_checksum,
        )?;
    }

    let mut cpu = Vec::with_capacity(ITERATIONS);
    let mut resident = Vec::with_capacity(ITERATIONS);
    let mut dense = Vec::with_capacity(ITERATIONS);
    let mut shared = Vec::with_capacity(ITERATIONS);
    let mut host = Vec::with_capacity(ITERATIONS);
    let mut pipelined_resident = Vec::with_capacity(ITERATIONS);
    for iteration in 0..ITERATIONS {
        measure_routes(
            workload,
            cpu_decoder,
            session,
            batch_size,
            expected_checksum,
            iteration % 2 == 0,
            &mut cpu,
            &mut resident,
            &mut dense,
            &mut shared,
            &mut host,
        )?;
    }
    for _ in 0..WARMUPS {
        let _ = run_pipelined_resident(workload, session, batch_size)?;
    }
    for _ in 0..ITERATIONS {
        pipelined_resident.push(run_pipelined_resident(workload, session, batch_size)?);
    }

    Ok(BatchResult {
        cpu: summarize_timings(&cpu).ok_or("CPU batch timing sample was empty")?,
        resident: summarize_metal(&resident)?,
        dense: summarize_metal(&dense)?,
        shared: summarize_metal(&shared)?,
        host: summarize_metal(&host)?,
        pipelined_resident: summarize_timings(&pipelined_resident)
            .ok_or("pipelined resident Metal timing sample was empty")?,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "the benchmark keeps one explicit timing vector per compared route"
)]
fn measure_routes(
    workload: &Workload,
    cpu_decoder: &CpuBatchDecoder,
    session: &MetalDecoderSession,
    batch_size: usize,
    expected_checksum: u64,
    cpu_first: bool,
    cpu: &mut Vec<Duration>,
    resident: &mut Vec<MetalMeasurement>,
    dense: &mut Vec<MetalMeasurement>,
    shared: &mut Vec<MetalMeasurement>,
    host: &mut Vec<MetalMeasurement>,
) -> Result<(), BenchmarkError> {
    if cpu_first {
        measure_cpu(workload, cpu_decoder, batch_size, expected_checksum, cpu)?;
    }
    let order = if cpu_first {
        [
            MetalOutput::Resident,
            MetalOutput::Dense,
            MetalOutput::Shared,
            MetalOutput::Host,
        ]
    } else {
        [
            MetalOutput::Host,
            MetalOutput::Shared,
            MetalOutput::Dense,
            MetalOutput::Resident,
        ]
    };
    for output in order {
        let measurement =
            run_metal_batch(workload, session, batch_size, expected_checksum, output)?;
        match output {
            MetalOutput::Resident => resident.push(measurement),
            MetalOutput::Dense => dense.push(measurement),
            MetalOutput::Shared => shared.push(measurement),
            MetalOutput::Host => host.push(measurement),
        }
    }
    if !cpu_first {
        measure_cpu(workload, cpu_decoder, batch_size, expected_checksum, cpu)?;
    }
    Ok(())
}

fn warm_batch_routes(
    workload: &Workload,
    cpu: &CpuBatchDecoder,
    session: &MetalDecoderSession,
    batch_size: usize,
    expected_checksum: u64,
) -> Result<(), BenchmarkError> {
    let _ = run_cpu_batch(workload, cpu, batch_size, expected_checksum)?;
    for output in [
        MetalOutput::Resident,
        MetalOutput::Dense,
        MetalOutput::Shared,
        MetalOutput::Host,
    ] {
        let _ = run_metal_batch(workload, session, batch_size, expected_checksum, output)?;
    }
    Ok(())
}

fn run_pipelined_resident(
    workload: &Workload,
    session: &MetalDecoderSession,
    batch_size: usize,
) -> Result<Duration, BenchmarkError> {
    let start = Instant::now();
    let first = prepare_metal_plans(workload, session, batch_size)?;
    let mut pending = session.submit_batch(&first)?;
    for _ in 1..PIPELINE_BATCHES {
        let next = prepare_metal_plans(workload, session, batch_size)?;
        let completed = pending.wait()?;
        if completed.len() != batch_size {
            return Err("pipelined Metal batch returned the wrong output count".into());
        }
        black_box(completed);
        pending = session.submit_batch(&next)?;
    }
    let completed = pending.wait()?;
    if completed.len() != batch_size {
        return Err("final pipelined Metal batch returned the wrong output count".into());
    }
    black_box(completed);
    Ok(start.elapsed())
}

fn summarize_metal(samples: &[MetalMeasurement]) -> Result<MetalSummary, BenchmarkError> {
    let totals = samples
        .iter()
        .map(|sample| sample.total)
        .collect::<Vec<_>>();
    let preparation = samples
        .iter()
        .map(|sample| sample.prepare)
        .collect::<Vec<_>>();
    let execution = samples
        .iter()
        .map(|sample| sample.execute)
        .collect::<Vec<_>>();
    Ok(MetalSummary {
        total: summarize_timings(&totals).ok_or("Metal total timing sample was empty")?,
        prepare: summarize_timings(&preparation)
            .ok_or("Metal preparation timing sample was empty")?,
        execute: summarize_timings(&execution).ok_or("Metal execution timing sample was empty")?,
    })
}

fn measure_cpu(
    workload: &Workload,
    decoder: &CpuBatchDecoder,
    batch_size: usize,
    expected_checksum: u64,
    samples: &mut Vec<Duration>,
) -> Result<(), BenchmarkError> {
    samples.push(run_cpu_batch(
        workload,
        decoder,
        batch_size,
        expected_checksum,
    )?);
    Ok(())
}

fn run_cpu_batch(
    workload: &Workload,
    decoder: &CpuBatchDecoder,
    batch_size: usize,
    expected_checksum: u64,
) -> Result<Duration, BenchmarkError> {
    let start = Instant::now();
    let inputs = (0..batch_size)
        .map(|_| {
            EncodedImage::new(
                Arc::clone(&workload.source),
                request(workload, BackendRequest::Cpu),
            )
        })
        .collect();
    let batch = decoder.decode(inputs)?;
    let elapsed = start.elapsed();
    if !batch.errors().is_empty() || batch.groups().len() != 1 {
        return Err(format!(
            "native CPU batch returned {} groups and {} errors",
            batch.groups().len(),
            batch.errors().len()
        )
        .into());
    }
    let group = &batch.groups()[0];
    if group.source_indices().len() != batch_size {
        return Err("native CPU batch returned the wrong output count".into());
    }
    for image in 0..batch_size {
        let checksum =
            checksum_cpu_batch_image(group.samples(), image, group.image_stride_elements())
                .ok_or("native CPU batch image range is invalid")?;
        check_checksum(checksum, expected_checksum)?;
    }
    Ok(elapsed)
}

fn run_metal_batch(
    workload: &Workload,
    session: &MetalDecoderSession,
    batch_size: usize,
    expected_checksum: u64,
    output: MetalOutput,
) -> Result<MetalMeasurement, BenchmarkError> {
    let start = Instant::now();
    let plans = prepare_metal_plans(workload, session, batch_size)?;
    let prepare = start.elapsed();
    let execute_start = Instant::now();
    let (resident_count, dense, shared, host) = match output {
        MetalOutput::Resident => (
            session.submit_batch(&plans)?.wait()?.len(),
            None,
            Vec::new(),
            Vec::new(),
        ),
        MetalOutput::Dense => {
            let batch = session.submit_dense_batch(&plans)?.wait()?;
            let count = batch.layout().image_count();
            (count, Some(batch), Vec::new(), Vec::new())
        }
        MetalOutput::Shared => {
            let images = session.decode_batch_to_shared(&plans)?;
            let count = images.len();
            (count, None, images, Vec::new())
        }
        MetalOutput::Host => {
            let decoded = session.decode_batch_to_host(&plans)?;
            let count = decoded.len();
            (count, None, Vec::new(), decoded)
        }
    };
    let execute = execute_start.elapsed();
    let total = start.elapsed();

    if resident_count != batch_size {
        return Err(format!(
            "Metal batch returned {resident_count} images for batch size {batch_size}"
        )
        .into());
    }
    match output {
        MetalOutput::Resident => {
            black_box(resident_count);
        }
        MetalOutput::Dense => {
            let batch = dense.ok_or("dense Metal route returned no batch owner")?;
            black_box(batch.layout().byte_len());
        }
        MetalOutput::Shared => {
            for image in shared {
                image.with_bytes(|bytes| {
                    check_checksum(checksum_bytes(bytes), expected_checksum)
                })??;
            }
        }
        MetalOutput::Host => {
            for image in host {
                check_checksum(checksum_samples(&image.samples), expected_checksum)?;
            }
        }
    }
    Ok(MetalMeasurement {
        total,
        prepare,
        execute,
    })
}

fn validate_metal_batch(
    workload: &Workload,
    session: &MetalDecoderSession,
    batch_size: usize,
    expected_checksum: u64,
) -> Result<(), BenchmarkError> {
    let plans = prepare_metal_plans(workload, session, batch_size)?;
    let resident = session.submit_batch(&plans)?.wait()?;
    if resident.len() != batch_size {
        return Err("Metal validation batch returned the wrong output count".into());
    }
    for image in &resident {
        check_checksum(checksum_bytes(&session.readback(image)?), expected_checksum)?;
    }
    let dense = session.submit_dense_batch(&plans)?.wait()?;
    if dense.layout().image_count() != batch_size {
        return Err("dense Metal validation batch returned the wrong output count".into());
    }
    for image in 0..batch_size {
        check_checksum(
            checksum_bytes(&session.readback_batch_image(&dense, image)?),
            expected_checksum,
        )?;
    }
    Ok(())
}

fn warm_pipeline(workload: &Workload, session: &MetalDecoderSession) -> Result<(), BenchmarkError> {
    let plan = prepare_metal_plans(workload, session, 1)?;
    black_box(session.submit_batch(&plan)?.wait()?);
    Ok(())
}

fn prepare_metal_plans(
    workload: &Workload,
    session: &MetalDecoderSession,
    batch_size: usize,
) -> Result<Vec<MetalDecodePlan>, BenchmarkError> {
    let view = JxrView::parse(&workload.source)?;
    let prepared_request = request(workload, BackendRequest::Metal);
    let coefficient_count = view.decoder().metal_coefficient_count(&prepared_request)?;
    let staging = session.coefficient_staging_batch(coefficient_count, batch_size)?;
    let plans = staging
        .into_par_iter()
        .map(|staging| {
            let view = JxrView::parse(&workload.source)?;
            let request = request(workload, BackendRequest::Metal);
            Ok(view
                .decoder()
                .prepare_metal_with_staging(&request, staging)?)
        })
        .collect::<Result<Vec<_>, BenchmarkError>>()?;
    Ok(plans)
}

fn decode_cpu(workload: &Workload) -> Result<DecodedImage, BenchmarkError> {
    let view = JxrView::parse(&workload.source)?;
    Ok(view
        .decoder()
        .decode(&request(workload, BackendRequest::Cpu))?)
}

fn request(workload: &Workload, backend: BackendRequest) -> DecodeRequest {
    DecodeRequest::new(workload.format)
        .with_alpha(workload.alpha)
        .with_backend(backend)
}

fn check_checksum(actual: u64, expected: u64) -> Result<(), BenchmarkError> {
    black_box(actual);
    if actual != expected {
        return Err(format!(
            "batch output checksum changed: expected {expected:016x}, got {actual:016x}"
        )
        .into());
    }
    Ok(())
}

fn load_workload(path: &Path) -> Result<Workload, BenchmarkError> {
    let source: Arc<[u8]> = std::fs::read(path)?.into();
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

fn print_result(workload: &Workload, batch_size: usize, result: &BatchResult) {
    let batch_size_f64 = f64::from(u32::try_from(batch_size).expect("batch size fits u32"));
    let pixels = f64::from(workload.width) * f64::from(workload.height) * batch_size_f64;
    let cpu_rate = pixels / result.cpu.median.as_secs_f64() / 1_000_000.0;
    let resident_rate = pixels / result.resident.total.median.as_secs_f64() / 1_000_000.0;
    let shared_rate = pixels / result.shared.total.median.as_secs_f64() / 1_000_000.0;
    let host_rate = pixels / result.host.total.median.as_secs_f64() / 1_000_000.0;
    let pipeline_depth = u32::try_from(PIPELINE_BATCHES).expect("pipeline depth fits u32");
    let pipeline_pixels = pixels * f64::from(pipeline_depth);
    let pipeline_rate =
        pipeline_pixels / result.pipelined_resident.median.as_secs_f64() / 1_000_000.0;
    println!();
    println!("batch {batch_size}");
    print_route("CPU host", result.cpu, cpu_rate);
    print_metal_route("Metal resident", &result.resident, resident_rate);
    let dense_rate = pixels / result.dense.total.median.as_secs_f64() / 1_000_000.0;
    print_metal_route("Metal dense", &result.dense, dense_rate);
    print_metal_route("Metal shared", &result.shared, shared_rate);
    print_metal_route("Metal host", &result.host, host_rate);
    println!(
        "  Metal pipeline median {:8.3} ms / {PIPELINE_BATCHES} batches | p95 {:8.3} ms | {:8.1} MP/s",
        milliseconds(result.pipelined_resident.median),
        milliseconds(result.pipelined_resident.p95),
        pipeline_rate
    );
    println!(
        "  host-output speedup {:.2}x; resident throughput ceiling {:.2}x CPU host",
        result.cpu.median.as_secs_f64() / result.host.total.median.as_secs_f64(),
        resident_rate / cpu_rate
    );
}

fn print_metal_route(label: &str, summary: &MetalSummary, rate: f64) {
    println!(
        "  {label:14} median {:8.3} ms | prep {:7.3} + Metal {:7.3} ms | p95 {:8.3} ms | {:8.1} MP/s",
        milliseconds(summary.total.median),
        milliseconds(summary.prepare.median),
        milliseconds(summary.execute.median),
        milliseconds(summary.total.p95),
        rate
    );
}

fn print_route(label: &str, summary: jxr_test_support::TimingSummary, rate: f64) {
    println!(
        "  {label:14} median {:8.3} ms | p95 {:8.3} ms | {:8.1} MP/s",
        milliseconds(summary.median),
        milliseconds(summary.p95),
        rate
    );
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
