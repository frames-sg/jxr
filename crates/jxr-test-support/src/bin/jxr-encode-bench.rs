// SPDX-License-Identifier: MIT OR Apache-2.0

//! Encoder throughput on synthetic images of representative pixel formats.
//!
//! Each workload is one image, twelve megapixels unless `--size` says
//! otherwise, encoded to an Annex-A file in memory losslessly and at quality
//! 75. The timed scope is one whole `jxr::encode` call: source samples to
//! component planes, transform and quantization, entropy coding, and the
//! container. The images are smooth fields under light noise, so the
//! entropy coder has photograph-like work at either setting.
//!
//! `--tile N` sets the largest tile edge in macroblocks instead of the
//! encoder's default; 255 codes these images as one tile, which times the
//! serial tile writer. `--setting lossless` or `--setting 75` codes each
//! image at one setting only, so that a process's peak resident size is
//! that setting's. `--decode` also times decoding each encoded file back
//! to the source's pixel format, the whole `decode` call. The thread count
//! is Rayon's: set `RAYON_NUM_THREADS` to change it.
//!
//! Two workloads code premultiplied 8-bit BGRA in the two shapes Flash
//! content is stored in: a frequency-mode file with first-level overlap and
//! separate alpha, and a spatial-mode file without overlap and with
//! integrated alpha.

use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use jxr::{
    AlphaMode, BitstreamMode, ChannelLayout, ChromaSampling, DecodeRequest, EncodeRequest, JxrView,
    OverlapMode, PixelFormat, SourceImage, SourceSamples,
};
use jxr_test_support::{
    TimingSummary, exit_on_error, milliseconds, summarize_timings, synthetic_samples,
};

const DEFAULT_WARMUPS: usize = 2;
const DEFAULT_ITERATIONS: usize = 10;
const DEFAULT_SIZE: (u32, u32) = (4000, 3000);

/// How a workload is coded beyond the lossless or quality-75 request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// The encoder's defaults for the source.
    Default,
    /// Premultiplied colour in a frequency-mode file with first-level
    /// overlap and separate alpha.
    FlashFrequency,
    /// Premultiplied colour in a spatial-mode file without overlap and with
    /// integrated alpha.
    FlashSpatial,
}

impl Shape {
    fn request(self, request: EncodeRequest) -> EncodeRequest {
        match self {
            Self::Default => request,
            Self::FlashFrequency => request
                .with_bitstream_mode(BitstreamMode::Frequency)
                .with_overlap(OverlapMode::One)
                .with_alpha_mode(AlphaMode::Separate),
            Self::FlashSpatial => request
                .with_bitstream_mode(BitstreamMode::Spatial)
                .with_overlap(OverlapMode::None)
                .with_alpha_mode(AlphaMode::Integrated),
        }
    }
}

/// `(name, pixel format, shape)` of every workload: the common 8-bit
/// layouts, the deeper integer and float depths, planar subsampled YUV, and
/// premultiplied BGRA in the shapes Flash content takes.
const WORKLOADS: [(&str, PixelFormat, Shape); 8] = [
    (
        "gray8",
        PixelFormat::U8(ChannelLayout::Luma),
        Shape::Default,
    ),
    ("rgb8", PixelFormat::U8(ChannelLayout::Rgb), Shape::Default),
    (
        "rgba8",
        PixelFormat::U8(ChannelLayout::Rgba),
        Shape::Default,
    ),
    (
        "rgb16",
        PixelFormat::U16(ChannelLayout::Rgb),
        Shape::Default,
    ),
    (
        "rgb32f",
        PixelFormat::F32(ChannelLayout::Rgb),
        Shape::Default,
    ),
    (
        "yuv420p8",
        PixelFormat::U8(ChannelLayout::Yuv(ChromaSampling::Cs420)),
        Shape::Default,
    ),
    (
        "pbgra8-frequency",
        PixelFormat::U8(ChannelLayout::Bgra),
        Shape::FlashFrequency,
    ),
    (
        "pbgra8-spatial",
        PixelFormat::U8(ChannelLayout::Bgra),
        Shape::FlashSpatial,
    ),
];

struct Options {
    warmups: usize,
    iterations: usize,
    width: u32,
    height: u32,
    /// Substring a workload's name must contain to run.
    only: Option<String>,
    /// Substring a setting's label must contain to run.
    setting: Option<String>,
    /// Largest tile edge in macroblocks, when not the encoder's default.
    tile: Option<u16>,
    /// Whether decoding each encoded file is timed as well.
    decode: bool,
}

fn main() {
    exit_on_error("jxr-encode-bench", run());
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_options()?;
    let megapixels = f64::from(options.width) * f64::from(options.height) / 1_000_000.0;
    println!("JXR CPU encode benchmark");
    println!(
        "image: {}x{} ({megapixels:.2} MP); warmups: {}; measured iterations: {}; threads: {}",
        options.width,
        options.height,
        options.warmups,
        options.iterations,
        rayon::current_num_threads()
    );
    println!("timed scope: source samples to an in-memory Annex-A file");
    let tile = options
        .tile
        .unwrap_or(EncodeRequest::DEFAULT_MAX_TILE_MACROBLOCKS);
    println!(
        "largest tile edge: {tile} macroblocks{}",
        if options.tile.is_some() {
            ""
        } else {
            " (the encoder's default)"
        }
    );
    let requests = [
        ("lossless", EncodeRequest::lossless()),
        ("quality 75", EncodeRequest::quality(75)),
    ]
    .map(|(label, request)| (label, request.with_max_tile_macroblocks(tile)));
    for (name, format, shape) in WORKLOADS {
        if options
            .only
            .as_ref()
            .is_some_and(|only| !name.contains(only.as_str()))
        {
            continue;
        }
        let samples = synthetic_samples(format, options.width, options.height)?;
        let image = SourceImage::new(
            options.width,
            options.height,
            format,
            SourceSamples::from(&samples),
        )?
        .with_premultiplied_alpha(shape != Shape::Default)?;
        println!();
        println!("{name} — {format:?} {shape:?}");
        let settings = requests.iter().filter(|(label, _)| {
            options
                .setting
                .as_ref()
                .is_none_or(|setting| label.contains(setting.as_str()))
        });
        for (label, request) in settings {
            let request = &shape.request(request.clone());
            let (file, first) = time_encode(&image, request)?;
            let bytes = file.len();
            // Kept only to be decoded, so that an encode-only run holds
            // nothing of an earlier encode while it times the next.
            let file = options.decode.then_some(file);
            for _ in 0..options.warmups {
                time_encode(&image, request)?;
            }
            let mut timings = Vec::with_capacity(options.iterations);
            for _ in 0..options.iterations {
                let (again, elapsed) = time_encode(&image, request)?;
                if again.len() != bytes {
                    return Err(format!("{name} {label}: encoded size changed between runs").into());
                }
                timings.push(elapsed);
            }
            let warm = summarize_timings(&timings).ok_or("timing sample was empty")?;
            print_result(label, bytes, first, warm, megapixels);
            if let Some(file) = file {
                time_decodes(&file, format, &options, megapixels)?;
            }
        }
    }
    Ok(())
}

/// Encode once, returning the file and the time the call took.
fn time_encode(
    image: &SourceImage<'_>,
    request: &EncodeRequest,
) -> Result<(Vec<u8>, Duration), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let file = jxr::encode(black_box(image), request)?;
    let elapsed = start.elapsed();
    Ok((black_box(file), elapsed))
}

/// Time decoding `file` to `format`, under the warm-up and iteration counts
/// of the encodes, and print the result beneath the encode's.
fn time_decodes(
    file: &[u8],
    format: PixelFormat,
    options: &Options,
    megapixels: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let view = JxrView::parse(file)?;
    let request = DecodeRequest::new(format);
    let decode = || -> Result<Duration, Box<dyn std::error::Error>> {
        let start = Instant::now();
        let image = view.decoder().decode(black_box(&request))?;
        let elapsed = start.elapsed();
        black_box(image);
        Ok(elapsed)
    };
    let first = decode()?;
    for _ in 0..options.warmups {
        decode()?;
    }
    let mut timings = Vec::with_capacity(options.iterations);
    for _ in 0..options.iterations {
        timings.push(decode()?);
    }
    let warm = summarize_timings(&timings).ok_or("timing sample was empty")?;
    print_result("  decode", file.len(), first, warm, megapixels);
    Ok(())
}

fn print_result(label: &str, bytes: usize, first: Duration, warm: TimingSummary, megapixels: f64) {
    println!(
        "  {label:10} {bytes:>10} bytes | first {:8.3} ms | warm min {:8.3} | median {:8.3} | p95 {:8.3} | mean {:8.3} ms | {:8.2} MP/s",
        milliseconds(first),
        milliseconds(warm.minimum),
        milliseconds(warm.median),
        milliseconds(warm.p95),
        milliseconds(warm.mean),
        megapixels / warm.median.as_secs_f64()
    );
}

fn parse_options() -> Result<Options, Box<dyn std::error::Error>> {
    let mut options = Options {
        warmups: DEFAULT_WARMUPS,
        iterations: DEFAULT_ITERATIONS,
        width: DEFAULT_SIZE.0,
        height: DEFAULT_SIZE.1,
        only: None,
        setting: None,
        tile: None,
        decode: false,
    };
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let mut value = || {
            arguments
                .next()
                .ok_or_else(|| format!("{argument} requires a value"))
        };
        match argument.as_str() {
            "--warmups" => options.warmups = value()?.parse()?,
            "--iterations" => {
                options.iterations = value()?.parse()?;
                if options.iterations == 0 {
                    return Err("--iterations must be greater than zero".into());
                }
            }
            "--size" => {
                let size = value()?;
                let (width, height) = size
                    .split_once('x')
                    .ok_or("--size takes WIDTHxHEIGHT, for example 4000x3000")?;
                (options.width, options.height) = (width.parse()?, height.parse()?);
                // Planar 4:2:0 halves both axes.
                if options.width < 2
                    || options.height < 2
                    || (options.width | options.height) & 1 != 0
                {
                    return Err("--size needs even, non-zero dimensions".into());
                }
            }
            "--only" => options.only = Some(value()?),
            "--setting" => options.setting = Some(value()?),
            "--tile" => {
                let tile: u16 = value()?.parse()?;
                if tile == 0 {
                    return Err("--tile must be greater than zero".into());
                }
                options.tile = Some(tile);
            }
            "--decode" => options.decode = true,
            "--help" | "-h" => {
                println!(
                    "usage: jxr-encode-bench [--warmups N] [--iterations N] [--size WxH] [--only NAME]\n\
                     \x20                       [--setting LABEL] [--tile MACROBLOCKS] [--decode]\n\
                     --setting runs only the settings whose label contains LABEL (lossless, 75);\n\
                     --tile sets the largest tile edge (255: one tile for these images);\n\
                     --decode also times decoding each encoded file.\n\
                     Workloads: {}",
                    WORKLOADS.map(|(name, ..)| name).join(", ")
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown option: {other}").into()),
        }
    }
    Ok(options)
}
