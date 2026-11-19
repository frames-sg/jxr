// SPDX-License-Identifier: MIT OR Apache-2.0

//! Peak live heap of one encode of each workload in the encode-throughput
//! documentation tables, printed rather than checked. Source samples come
//! from the same generator as `jxr-encode-bench`.
//!
//! Run it alone and on the thread count to report, with nothing else in
//! the process allocating:
//!
//! ```sh
//! RAYON_NUM_THREADS=1 cargo test --release -p jxr-alloc-probe \
//!     --test encode_memory_report -- --ignored --nocapture
//! ```
//!
//! Working memory is the peak less the returned file, which the call holds
//! when it returns; the source is allocated before the measurement starts
//! and is not counted.

use jxr::{
    AlphaMode, BitstreamMode, ChannelLayout, EncodeRequest, OverlapMode, PixelFormat, SourceImage,
    SourceSamples,
};
use jxr_alloc_probe::{CountingAllocator, measure};
use jxr_test_support::synthetic_samples;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// One measured workload.
struct Case {
    name: &'static str,
    format: PixelFormat,
    size: (u32, u32),
    premultiplied: bool,
    request: EncodeRequest,
}

fn flash(name: &'static str, tile: Option<u16>, frequency: bool, lossless: bool) -> Case {
    let request = if lossless {
        EncodeRequest::lossless()
    } else {
        EncodeRequest::quality(75)
    };
    let request = if frequency {
        request
            .with_bitstream_mode(BitstreamMode::Frequency)
            .with_overlap(OverlapMode::One)
            .with_alpha_mode(AlphaMode::Separate)
    } else {
        request
            .with_overlap(OverlapMode::None)
            .with_alpha_mode(AlphaMode::Integrated)
    };
    Case {
        name,
        format: PixelFormat::U8(ChannelLayout::Bgra),
        size: (4096, 4096),
        premultiplied: true,
        request: match tile {
            Some(tile) => request.with_max_tile_macroblocks(tile),
            None => request,
        },
    }
}

fn cases() -> Vec<Case> {
    let mut cases = vec![
        flash("pbgra8 4096 q75 frequency one tile", Some(256), true, false),
        flash(
            "pbgra8 4096 lossless frequency one tile",
            Some(256),
            true,
            true,
        ),
        flash("pbgra8 4096 q75 frequency default tiles", None, true, false),
        flash(
            "pbgra8 4096 lossless frequency default tiles",
            None,
            true,
            true,
        ),
        flash("pbgra8 4096 q75 spatial one tile", Some(256), false, false),
        flash(
            "pbgra8 4096 lossless spatial one tile",
            Some(256),
            false,
            true,
        ),
        flash("pbgra8 4096 q75 spatial default tiles", None, false, false),
        flash(
            "pbgra8 4096 lossless spatial default tiles",
            None,
            false,
            true,
        ),
    ];
    for (name, format) in [
        ("gray8", PixelFormat::U8(ChannelLayout::Luma)),
        ("rgb8", PixelFormat::U8(ChannelLayout::Rgb)),
        ("rgba8", PixelFormat::U8(ChannelLayout::Rgba)),
        ("rgb16", PixelFormat::U16(ChannelLayout::Rgb)),
        ("rgb32f", PixelFormat::F32(ChannelLayout::Rgb)),
    ] {
        for (setting, request) in [
            ("lossless", EncodeRequest::lossless()),
            ("q75", EncodeRequest::quality(75)),
        ] {
            cases.push(Case {
                name: Box::leak(format!("{name} 4000x3000 {setting}").into_boxed_str()),
                format,
                size: (4000, 3000),
                premultiplied: false,
                request,
            });
        }
    }
    for (setting, request) in [
        ("lossless", EncodeRequest::lossless()),
        ("q75", EncodeRequest::quality(75)),
    ] {
        cases.push(Case {
            name: Box::leak(format!("rgb8 8000x6000 {setting}").into_boxed_str()),
            format: PixelFormat::U8(ChannelLayout::Rgb),
            size: (8000, 6000),
            premultiplied: false,
            request,
        });
    }
    cases
}

#[test]
#[ignore = "a report for the documentation; run it alone with --nocapture"]
fn report_encode_memory() {
    let only = std::env::var("JXR_MEMORY_REPORT_ONLY").ok();
    println!(
        "RAYON_NUM_THREADS: {}",
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".into())
    );
    println!("Peak heap counts both blocks when a reallocation moves.");
    println!("case | peak heap bytes | file bytes | working bytes | working bytes/pixel");
    for case in cases() {
        if only
            .as_ref()
            .is_some_and(|only| !case.name.contains(only.as_str()))
        {
            continue;
        }
        let (width, height) = case.size;
        let samples = synthetic_samples(case.format, width, height).unwrap();
        let image = SourceImage::new(width, height, case.format, SourceSamples::from(&samples))
            .unwrap()
            .with_premultiplied_alpha(case.premultiplied)
            .unwrap();
        // Once to start the thread pool, which is not the encoder's.
        drop(jxr::encode(&image, &case.request).unwrap());
        let (file, usage) = measure(|| jxr::encode(&image, &case.request).unwrap());
        let working = usage.peak_bytes.saturating_sub(file.len());
        let pixels = u64::from(width) * u64::from(height);
        // Hundredths of a byte per pixel.
        let per_pixel = u64::try_from(working).unwrap() * 100 / pixels;
        println!(
            "{} | {} | {} | {} | {}.{:02}",
            case.name,
            usage.peak_bytes,
            file.len(),
            working,
            per_pixel / 100,
            per_pixel % 100
        );
    }
}
