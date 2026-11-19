// SPDX-License-Identifier: MIT OR Apache-2.0

//! Peak-memory and allocation budgets of the CPU encoder.
//!
//! An encode reads a few macroblock rows of the source at a time into a
//! strip for each component, which filters and transforms them as they
//! arrive and keeps only the rows its filters wait on; each macroblock row
//! is quantized and coded as it is ready, and the tile rows it finishes are
//! appended to the file. What it holds beside the source and the file is
//! therefore those rows and the coding state of the tile rows it is in,
//! however tall the image is: its working memory goes with the width. In a
//! thread pool a band of macroblock rows is analyzed while the band above
//! it is coded, and the bands are admitted against a working bound. The
//! budgets here are a regression gate. They sit above what an encode holds
//! today, and far below one whole-image buffer, so a change that keeps the
//! samples or the coefficients of a whole component alive fails here
//! instead of passing unnoticed. Everything is one test because the
//! counters are the process's: nothing else may allocate while a call is
//! measured.
//!
//! The encoder is measured on one thread, as it runs where memory matters
//! most, in a browser's WebAssembly, and once in a pool against its bound.
//! Working memory is the peak less the file the call returns.

mod common;

use jxr::{
    AlphaMode, BitstreamMode, ChannelLayout, ChromaSampling, EncodeQuantizers, EncodeRequest,
    OverlapMode, PixelFormat, SourceImage, SourceSamples,
};
use jxr_alloc_probe::{CountingAllocator, POOL_BYTES, Usage, measure};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Working bytes an encode on one thread may hold for each pixel of the
/// width of an image, whatever its height, beside the file and its growth.
///
/// A macroblock row of one component is 16 rows of 32-bit samples, 64 bytes
/// for each pixel of width, and as many coefficients, 69 bytes. On one
/// thread an encode reads seven macroblock rows of samples of every plane
/// at a time, below the rows its filters wait on, two macroblock rows at
/// most, and holds the coefficients of one macroblock row, and the coding
/// state and packets of the tile rows it is in. That is some 450 bytes for
/// gray, 1,600 to 2,300 for colour and alpha, and 3,600 for alpha and
/// colour read at full resolution and filtered down to 4:2:0 with
/// second-level overlap, the most rows a strip waits on. The tile writer
/// retains up to two macroblock rows of prediction history per component.
const BYTES_PER_PIXEL_OF_WIDTH: usize = 4_096;

/// What the file may take beside the working memory: itself, and the room
/// reserved for it beyond its length. The file grows once to the length
/// its first rows predict and an eighth more, which for these evenly
/// textured images is within an eighth of its length.
fn file_allowance(file: usize) -> usize {
    file + file / 8
}

/// How much more working memory, in eighths, an image four times as tall
/// may take beyond the growth of its file's allowance.
///
/// What a band holds is the same at either height. The file is the one
/// buffer that grows with the image.
const TALLER_MARGIN_EIGHTHS: usize = 1;

/// Largest tile edge of the images, in macroblocks.
const TILE: u16 = 4;

/// Rows of the shorter image of each width: more than three reads of
/// seven macroblock rows on one thread, so that one read has rows of the
/// image on both sides, as every read but the first and last of a taller
/// image does.
const SHORT: u32 = 384;
/// Rows of the taller, four times as many.
const TALL: u32 = 4 * SHORT;

/// The file length and heap use of one encode of a `width` by `height`
/// image in the pool the call is made from. Alpha is premultiplied, as
/// Flash's is.
fn encode_usage(
    layout: ChannelLayout,
    (width, height): (u32, u32),
    request: &EncodeRequest,
) -> (usize, Usage) {
    let format = PixelFormat::U8(layout);
    let samples = common::pixels(usize::from(format.channel_count()), width, height);
    let image = SourceImage::new(width, height, format, SourceSamples::U8(&samples))
        .unwrap()
        .with_premultiplied_alpha(format.has_alpha())
        .unwrap();
    let (file, usage) = measure(|| jxr::encode(&image, request).unwrap());
    (file.len(), usage)
}

/// What Flash content looks like, the layout that holds the most rows
/// around a band, and the encoder's defaults for gray and colour, each
/// with its tiles four macroblocks tall.
fn layouts() -> [(&'static str, ChannelLayout, EncodeRequest); 5] {
    let tiled = |request: EncodeRequest| request.with_max_tile_macroblocks(TILE);
    let lossy = || EncodeRequest::lossy(EncodeQuantizers::uniform(20));
    [
        (
            "bgra8, frequency mode, overlap one, separate alpha",
            ChannelLayout::Bgra,
            tiled(lossy())
                .with_overlap(OverlapMode::One)
                .with_bitstream_mode(BitstreamMode::Frequency)
                .with_alpha_mode(AlphaMode::Separate),
        ),
        (
            "bgra8, spatial mode, no overlap, integrated alpha",
            ChannelLayout::Bgra,
            tiled(lossy())
                .with_overlap(OverlapMode::None)
                .with_alpha_mode(AlphaMode::Integrated),
        ),
        (
            "rgba8, overlap two, 4:2:0, integrated alpha, lossless",
            ChannelLayout::Rgba,
            tiled(EncodeRequest::lossless())
                .with_overlap(OverlapMode::Two)
                .with_chroma_sampling(ChromaSampling::Cs420)
                .with_alpha_mode(AlphaMode::Integrated),
        ),
        (
            "gray8, lossless",
            ChannelLayout::Luma,
            tiled(EncodeRequest::lossless()),
        ),
        (
            "rgb8, quality 75",
            ChannelLayout::Rgb,
            tiled(EncodeRequest::quality(75)),
        ),
    ]
}

#[test]
fn encoder_memory_goes_with_the_width_of_the_image() {
    // The probe sees an allocation of a known size, and its release.
    let (mut block, usage) = measure(|| vec![0_u8; 1 << 20]);
    assert!(usage.peak_bytes >= block.len() && usage.peak_bytes < block.len() + 4096);
    assert!(usage.allocations >= 1);
    let original = block.as_ptr();
    block.truncate(block.len() - 16);
    let ((), shrink) = measure(|| block.shrink_to_fit());
    if block.as_ptr() == original {
        assert!(shrink.peak_bytes < 4096, "in-place shrink: {shrink:?}");
    } else {
        assert!(shrink.peak_bytes >= block.len(), "moved block: {shrink:?}");
    }
    drop(block);
    let ((), usage) = measure(|| drop(vec![0_u8; 1 << 20]));
    assert!(usage.peak_bytes >= 1 << 20);

    let layouts = layouts();
    // One thread, as a browser runs the encoder; the pool is built before
    // anything is measured.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        for (name, layout, request) in &layouts {
            for width in [1024_u32, 2048] {
                let row_budget = width as usize * BYTES_PER_PIXEL_OF_WIDTH;
                let (short_file, short) = encode_usage(*layout, (width, SHORT), request);
                let (tall_file, tall) = encode_usage(*layout, (width, TALL), request);
                let working = |file: usize, usage: &Usage| usage.peak_bytes - file;
                for (height, file, usage) in [(SHORT, short_file, &short), (TALL, tall_file, &tall)]
                {
                    let working = working(file, usage);
                    let budget = row_budget + file_allowance(file);
                    assert!(
                        working <= budget,
                        "{name}, {width}x{height}: working memory {working} bytes ({} for each \
                         pixel of width beside the file) is over the budget of {budget} ({file} \
                         byte file)",
                        working.saturating_sub(file_allowance(file)) / width as usize,
                    );
                }
                // Four times the rows hold what a quarter of them do beside
                // the larger file: the rows of a band, not of the image.
                let (short_working, tall_working) =
                    (working(short_file, &short), working(tall_file, &tall));
                let allowed = short_working
                    + short_working * TALLER_MARGIN_EIGHTHS / 8
                    + file_allowance(tall_file - short_file);
                assert!(
                    tall_working <= allowed,
                    "{name}, {width} wide: working memory {tall_working} bytes at {TALL} rows, \
                     {short_working} at {SHORT} ({tall_file} and {short_file} byte files)"
                );
                // The peak is real: the file, and one macroblock row of one
                // component as samples and as coefficients, which no band
                // is without.
                let row = width as usize * 16;
                assert!(
                    short.peak_bytes > short_file + row * 4 + row / 256 * 1104,
                    "{name}"
                );
            }
            // Tiles and bands allocate, macroblocks do not: one allocation
            // for each macroblock of one component would be as many as
            // there are macroblocks. Tiles of the default size, so that
            // what each tile allocates is spread over many macroblocks.
            let request = request
                .clone()
                .with_max_tile_macroblocks(EncodeRequest::DEFAULT_MAX_TILE_MACROBLOCKS);
            let (file, usage) = encode_usage(*layout, (1024, 768), &request);
            let macroblocks = 1024 / 16 * 48;
            let budget = 1024 * BYTES_PER_PIXEL_OF_WIDTH + file_allowance(file);
            assert!(
                usage.peak_bytes - file <= budget,
                "{name}, default tiles: working memory {} bytes is over the budget of {budget} \
                 ({file} byte file)",
                usage.peak_bytes - file
            );
            assert!(
                usage.allocations < macroblocks / 4,
                "{name}, default tiles: {} allocations for {macroblocks} macroblocks",
                usage.allocations
            );
        }
    });

    // In a thread pool the bands of an image are admitted against the
    // working bound, however tall: one tile of 4096 by 4096, whose samples
    // and coefficients alone come to some 550 MB, is coded in bands of the
    // rows the bound admits.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();
    let request = EncodeRequest::lossless().with_max_tile_macroblocks(256);
    pool.install(|| {
        // Once to start the threads of the pool, which are not the
        // encoder's.
        encode_usage(ChannelLayout::Rgba, (64, 64), &request);
        let (file, usage) = encode_usage(ChannelLayout::Rgba, (4096, 4096), &request);
        let working = usage.peak_bytes - file;
        let budget = POOL_BYTES + file_allowance(file) - file;
        assert!(
            working <= budget,
            "one tile in a pool: working memory {working} bytes is over the budget of \
             {budget} ({file} byte file)"
        );
    });
}
