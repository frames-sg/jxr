// SPDX-License-Identifier: MIT OR Apache-2.0

//! Peak-memory budgets of the full-resolution CPU decoder.
//!
//! A decode reads a band of macroblock rows, reconstructs it, stores the
//! sample rows that finishes in the caller's output and lets them go. What
//! it holds beside the output is therefore a few rows of the image, however
//! tall the image is: its working memory goes with the width. The budgets
//! here are a regression gate. They sit above what a decode holds today, and
//! far below one whole-image buffer, so a change that keeps the coefficients
//! or the samples of a whole component alive fails here instead of passing
//! unnoticed. Everything is one test because the counters are the
//! process's: nothing else may allocate while a call is measured.

mod common;

use jxr::{
    AlphaMode, BitstreamMode, ChannelLayout, ChromaSampling, DecodeRequest, EncodeQuantizers,
    EncodeRequest, JxrView, OverlapMode, PixelFormat, SourceImage, SourceSamples,
};
use jxr_alloc_probe::{CountingAllocator, POOL_BYTES, Usage, measure};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const RGBA8: PixelFormat = PixelFormat::U8(ChannelLayout::Rgba);

/// Working bytes a decode on one thread may hold for each pixel of the
/// width of an eight-bit RGBA image, whatever its height.
///
/// A macroblock row of one component is 16 rows of 32-bit samples and as
/// many coefficients: 128 bytes for each pixel of width, 512 for the four
/// components. A decode holds the coefficients of one band and the samples
/// of that band and of the rows that wait on the next, which the overlap
/// filter of the low-pass level makes a whole macroblock row: some 2,000
/// bytes at one macroblock row to a band, as images 2,048 pixels wide and
/// wider have, and some 2,500 at the two rows of an image 1,024 wide.
const BYTES_PER_PIXEL_OF_WIDTH: usize = 3_072;

fn encode(width: u32, height: u32, request: &EncodeRequest) -> Vec<u8> {
    let samples = common::pixels(4, width, height);
    let image = SourceImage::new(width, height, RGBA8, SourceSamples::U8(&samples)).unwrap();
    jxr::encode(&image, request).unwrap()
}

/// What decoding `file` into storage the caller already owns asks of the
/// heap, in a pool of `threads` threads.
fn decode_usage(file: &[u8], threads: usize) -> Usage {
    let view = JxrView::parse(file).unwrap();
    let (width, height) = view.info().dimensions();
    let mut destination = vec![0_u8; width as usize * height as usize * 4];
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let request = DecodeRequest::new(RGBA8);
    let mut decode = || {
        pool.install(|| view.decoder().decode_into(&request, &mut destination))
            .unwrap();
    };
    // Once to start the threads of the pool, which are not the decoder's.
    decode();
    measure(decode).1
}

#[test]
fn decoder_memory_goes_with_the_width_of_the_image() {
    // The probe sees an allocation of a known size.
    let (block, usage) = measure(|| vec![0_u8; 1 << 20]);
    assert!(usage.peak_bytes >= block.len() && usage.peak_bytes < block.len() + 4096);
    drop(block);

    let one_tile = |request: EncodeRequest| request.with_max_tile_macroblocks(u16::MAX);
    let lossy = || EncodeRequest::lossy(EncodeQuantizers::uniform(20));
    // What Flash content looks like, what texture mips look like, and the
    // layout that holds the most: both overlap levels and chroma that is
    // interpolated from the rows above and below.
    let layouts = [
        (
            "frequency mode, overlap one, separate alpha",
            one_tile(lossy())
                .with_overlap(OverlapMode::One)
                .with_bitstream_mode(BitstreamMode::Frequency)
                .with_alpha_mode(AlphaMode::Separate),
        ),
        (
            "spatial mode, no overlap, integrated alpha",
            one_tile(lossy())
                .with_overlap(OverlapMode::None)
                .with_alpha_mode(AlphaMode::Integrated),
        ),
        (
            "overlap two, 4:2:0, integrated alpha, lossless",
            one_tile(EncodeRequest::lossless())
                .with_overlap(OverlapMode::Two)
                .with_chroma_sampling(ChromaSampling::Cs420)
                .with_alpha_mode(AlphaMode::Integrated),
        ),
    ];
    for (name, request) in &layouts {
        for width in [1024_u32, 2048] {
            let budget = width as usize * BYTES_PER_PIXEL_OF_WIDTH;
            let short = decode_usage(&encode(width, 192, request), 1);
            let tall = decode_usage(&encode(width, 768, request), 1);
            for (height, usage) in [(192, short), (768, tall)] {
                assert!(
                    usage.peak_bytes <= budget,
                    "{name}, {width}x{height}: peak {} bytes ({} for each pixel of width) is \
                     over the budget of {budget}",
                    usage.peak_bytes,
                    usage.peak_bytes / width as usize,
                );
            }
            // Four times the rows hold what a quarter of them do: the rows
            // of a band, not of the image.
            assert!(
                tall.peak_bytes <= short.peak_bytes + short.peak_bytes / 8,
                "{name}, {width} wide: {} bytes at 768 rows, {} at 192",
                tall.peak_bytes,
                short.peak_bytes,
            );
            // The peak is real: more than one macroblock row of samples of
            // every component, which no decode can do without.
            assert!(short.peak_bytes > width as usize * 16 * 4 * 4, "{name}");
            // Bands allocate, macroblocks do not.
            let macroblocks = width as usize / 16 * 48;
            assert!(
                tall.allocations < macroblocks / 2,
                "{name}, {width} wide: {} allocations for {macroblocks} macroblocks",
                tall.allocations
            );
        }
    }

    // In a thread pool the tile rows of an image are shared out among
    // strips, as many as the working bound admits, however many tiles the
    // image has.
    let tiled = EncodeRequest::lossless()
        .with_overlap(OverlapMode::Two)
        .with_max_tile_macroblocks(4);
    let usage = decode_usage(&encode(1024, 1024, &tiled), 8);
    assert!(
        usage.peak_bytes <= POOL_BYTES,
        "tiled image in a pool: peak {} bytes is over the bound of {POOL_BYTES}",
        usage.peak_bytes
    );
}
