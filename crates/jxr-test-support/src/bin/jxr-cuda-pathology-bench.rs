// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    hint::black_box,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use jxr::{
    AlphaHandling, BackendRequest, DecodeRequest, JxrView, PixelFormat, Rect,
    cuda::{CudaDecodePlan, CudaDecoderSession},
};
use jxr_test_support::{checksum_bytes, checksum_samples, oracle_format, summarize_timings};
use rayon::prelude::*;

const BATCH_SIZES: [usize; 5] = [1, 8, 32, 64, 128];
const DEFAULT_ITERATIONS: usize = 10;

type BenchmarkError = Box<dyn std::error::Error + Send + Sync>;

struct SourceWorkload {
    label: &'static str,
    path: PathBuf,
    source: Arc<[u8]>,
    format: PixelFormat,
    alpha: AlphaHandling,
    width: u32,
    height: u32,
}

#[derive(Clone, Copy)]
struct Workload<'a> {
    source: &'a SourceWorkload,
    region: Option<Rect>,
}

#[derive(Default)]
struct Measurements {
    prepare: Vec<Duration>,
    submit: Vec<Duration>,
    synchronize: Vec<Duration>,
    readback: Vec<Duration>,
    total: Vec<Duration>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("jxr-cuda-pathology-bench: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), BenchmarkError> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("test-support crate is outside its workspace")?;
    let mut arguments = std::env::args_os().skip(1);
    let small_path = arguments.next().map_or_else(
        || workspace.join("target/t834-conformance/suite-2014/BasicAndOverlap_2x2Tile/Small_Freq_Ov2_2x2_YUV420_QP10.jxr"),
        PathBuf::from,
    );
    let large_path = arguments.next().map_or_else(
        || workspace.join("target/t834-conformance/suite-2014/Windowing/Windowed8.jxr"),
        PathBuf::from,
    );
    if arguments.next().is_some() {
        return Err("usage: jxr-cuda-pathology-bench [SMALL.jxr] [LARGE.jxr]".into());
    }
    let iterations = std::env::var("JXR_BENCH_ITERATIONS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(DEFAULT_ITERATIONS);
    if iterations == 0 {
        return Err("JXR_BENCH_ITERATIONS must be nonzero".into());
    }
    let sources = [
        load_source("small", &small_path)?,
        load_source("large", &large_path)?,
    ];
    let session = CudaDecoderSession::system_default()?;
    println!("JXR CUDA pathology reconstruction benchmark");
    println!(
        "iterations={iterations}\tstreams={}",
        session.stream_count()
    );
    for source in &sources {
        println!("source.{}={}", source.label, source.path.display());
    }
    println!(
        "size\tpath\tbatch\twidth\theight\tprepare_median_ms\tsubmit_median_ms\tsync_median_ms\treadback_median_ms\ttotal_median_ms\ttotal_p95_ms\timages_per_s\tpool_allocations\th2d_bytes\td2h_bytes"
    );
    for source in &sources {
        for region in [None, Some(roi(source.width, source.height)?)] {
            let workload = Workload { source, region };
            for batch_size in BATCH_SIZES {
                benchmark(&session, workload, batch_size, iterations)?;
            }
        }
    }
    Ok(())
}

fn benchmark(
    session: &CudaDecoderSession,
    workload: Workload<'_>,
    batch_size: usize,
    iterations: usize,
) -> Result<(), BenchmarkError> {
    let expected = cpu_checksum(workload)?;
    let warm = prepare_plans(workload, batch_size)?;
    let output_bytes = warm
        .first()
        .ok_or("benchmark batch unexpectedly produced no plans")?
        .output()
        .byte_len;
    let warm_resident = session.submit_batch(&warm)?.wait()?;
    for image in &warm_resident {
        check_checksum(checksum_bytes(&session.readback(image)?), expected)?;
    }
    let pool_before = session.buffer_pool_diagnostics()?;
    let upload_before = session.upload_cache_diagnostics()?;
    let mut measurements = Measurements::default();
    for _ in 0..iterations {
        let total_start = Instant::now();
        let prepare_start = Instant::now();
        let plans = prepare_plans(workload, batch_size)?;
        measurements.prepare.push(prepare_start.elapsed());
        let submit_start = Instant::now();
        let pending = session.submit_batch(&plans)?;
        measurements.submit.push(submit_start.elapsed());
        let sync_start = Instant::now();
        let resident = pending.wait()?;
        measurements.synchronize.push(sync_start.elapsed());
        let readback_start = Instant::now();
        for image in &resident {
            let bytes = session.readback(image)?;
            check_checksum(checksum_bytes(&bytes), expected)?;
            black_box(bytes);
        }
        measurements.readback.push(readback_start.elapsed());
        measurements.total.push(total_start.elapsed());
    }
    let pool_after = session.buffer_pool_diagnostics()?;
    let upload_after = session.upload_cache_diagnostics()?;
    let prepare = summarize_timings(&measurements.prepare).ok_or("empty preparation sample")?;
    let submit = summarize_timings(&measurements.submit).ok_or("empty submission sample")?;
    let synchronize =
        summarize_timings(&measurements.synchronize).ok_or("empty synchronization sample")?;
    let readback = summarize_timings(&measurements.readback).ok_or("empty readback sample")?;
    let total = summarize_timings(&measurements.total).ok_or("empty total sample")?;
    let dimensions = workload
        .region
        .map_or((workload.source.width, workload.source.height), |region| {
            (region.w, region.h)
        });
    let batch_count = u32::try_from(batch_size)?;
    let images_per_second = f64::from(batch_count) / total.median.as_secs_f64();
    let d2h_bytes = output_bytes
        .checked_mul(batch_size)
        .and_then(|bytes| bytes.checked_mul(iterations))
        .ok_or("device-to-host byte count overflow")?;
    println!(
        "{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.1}\t{}\t{}\t{}",
        workload.source.label,
        if workload.region.is_some() {
            "roi"
        } else {
            "full"
        },
        batch_size,
        dimensions.0,
        dimensions.1,
        ms(prepare.median),
        ms(submit.median),
        ms(synchronize.median),
        ms(readback.median),
        ms(total.median),
        ms(total.p95),
        images_per_second,
        pool_after.misses.saturating_sub(pool_before.misses),
        upload_after
            .uploaded_bytes
            .saturating_sub(upload_before.uploaded_bytes),
        d2h_bytes,
    );
    Ok(())
}

fn prepare_plans(
    workload: Workload<'_>,
    batch_size: usize,
) -> Result<Vec<CudaDecodePlan>, BenchmarkError> {
    (0..batch_size)
        .into_par_iter()
        .map(|_| {
            let view = JxrView::parse(&workload.source.source)?;
            let request = request(workload, BackendRequest::Cuda);
            Ok(view
                .decoder()
                .prepare_reconstruction(&request)?
                .cuda_plan()?)
        })
        .collect()
}

fn cpu_checksum(workload: Workload<'_>) -> Result<u64, BenchmarkError> {
    let view = JxrView::parse(&workload.source.source)?;
    let image = view
        .decoder()
        .decode(&request(workload, BackendRequest::Cpu))?;
    Ok(checksum_samples(&image.samples))
}

fn request(workload: Workload<'_>, backend: BackendRequest) -> DecodeRequest {
    let mut request = DecodeRequest::new(workload.source.format)
        .with_alpha(workload.source.alpha)
        .with_backend(backend);
    request.region = workload.region;
    request
}

fn load_source(label: &'static str, path: &Path) -> Result<SourceWorkload, BenchmarkError> {
    let source: Arc<[u8]> = std::fs::read(path)?.into();
    let view = JxrView::parse(&source)?;
    let format = oracle_format(view.info())?;
    let width = view.info().width;
    let height = view.info().height;
    Ok(SourceWorkload {
        label,
        path: path.to_owned(),
        source,
        format: format.pixel_format,
        alpha: format.alpha,
        width,
        height,
    })
}

fn roi(width: u32, height: u32) -> Result<Rect, BenchmarkError> {
    if width < 4 || height < 4 {
        return Err("benchmark source is too small for an ROI workload".into());
    }
    let x = (width / 4) & !1;
    let y = (height / 4) & !1;
    let w = ((width / 2) & !1).max(2);
    let h = ((height / 2) & !1).max(2);
    Ok(Rect { x, y, w, h })
}

fn check_checksum(actual: u64, expected: u64) -> Result<(), BenchmarkError> {
    black_box(actual);
    if actual == expected {
        Ok(())
    } else {
        Err(format!("CUDA checksum changed: expected {expected:016x}, got {actual:016x}").into())
    }
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
