//! A decode or an encode through the thread pool and one on the calling
//! thread alone return the same result.
//!
//! A target without threads (WebAssembly without the `atomics` feature)
//! only ever decodes and encodes on the calling thread, and that target is
//! not one the tests run on: this runs the same sequential executor here,
//! over images that send every stage that shares out work through the pool
//! when it is given one, and compares the two in full.

use jxr_core::{
    AlphaHandling, AlphaMode, BitstreamMode, ChannelLayout, ChromaSampling, DecodeRequest,
    DecodeScale, EncodeQuantizers, EncodeRequest, OverlapMode, PixelFormat, Rect, SourceImage,
    SourceSamples,
};
use jxr_native::{CpuCapabilities, decode_cpu, encode_annex_a, parse_codestream, prepare_plan};

/// Large enough for every stage's threshold for sharing out work: 270
/// macroblocks, and more than 65,536 pixels to store.
const WIDTH: u32 = 288;
const HEIGHT: u32 = 240;

/// Interleaved RGBA pixels with gradients, texture and noise.
fn pixels() -> Vec<u8> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut samples = Vec::new();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            for channel in 0..4 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let ramp = x * (2 + channel) + y * 3;
                let texture = if (x / 5 + y / 7 + channel).is_multiple_of(2) {
                    24
                } else {
                    0
                };
                let noise = u32::try_from((state >> 40) % 16).unwrap();
                samples.push(u8::try_from((ramp / 8 + texture + noise) % 256).unwrap());
            }
        }
    }
    samples
}

/// Encodings that reach every executor call of the decoder and of the
/// encoder: several tiles in a row (tiles read and coded side by side),
/// several tile rows (strips and the joins between them), every overlap
/// mode, both packet layouts, both alpha placements and subsampled chroma.
fn requests() -> Vec<EncodeRequest> {
    let lossy = EncodeRequest::lossy(EncodeQuantizers::uniform(12));
    vec![
        lossy
            .clone()
            .with_max_tile_macroblocks(4)
            .with_overlap(OverlapMode::One)
            .with_alpha_mode(AlphaMode::Integrated),
        lossy
            .clone()
            .with_max_tile_macroblocks(6)
            .with_overlap(OverlapMode::Two)
            .with_bitstream_mode(BitstreamMode::Frequency)
            .with_alpha_mode(AlphaMode::Separate)
            .with_chroma_sampling(ChromaSampling::Cs420),
        lossy
            .clone()
            .with_max_tile_macroblocks(5)
            .with_hard_tiles(true)
            .with_overlap(OverlapMode::One)
            .with_bitstream_mode(BitstreamMode::Frequency)
            .with_chroma_sampling(ChromaSampling::Cs422),
        EncodeRequest::lossless()
            .with_max_tile_macroblocks(u16::MAX)
            .with_overlap(OverlapMode::None)
            .with_alpha_mode(AlphaMode::Integrated),
    ]
}

/// What a caller may ask of each image: full images in two formats, a
/// region, alpha handled two ways, and both reduced scales, which only
/// frequency-mode images support (the others' errors are compared too).
fn decode_requests() -> Vec<DecodeRequest> {
    let rgba = DecodeRequest::new(PixelFormat::U8(ChannelLayout::Rgba));
    vec![
        rgba.clone(),
        DecodeRequest::new(PixelFormat::U16(ChannelLayout::Rgb)).with_alpha(AlphaHandling::Drop),
        rgba.clone().with_alpha(AlphaHandling::Premultiply),
        rgba.clone().with_region(Rect {
            x: 37,
            y: 51,
            w: 190,
            h: 133,
        }),
        rgba.clone().with_scale(DecodeScale::Quarter),
        rgba.with_scale(DecodeScale::Sixteenth),
    ]
}

/// A pool of `threads` threads, whatever the machine running the test.
fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

fn image(samples: &[u8]) -> SourceImage<'_> {
    SourceImage::new(
        WIDTH,
        HEIGHT,
        PixelFormat::U8(ChannelLayout::Rgba),
        SourceSamples::U8(samples),
    )
    .unwrap()
}

/// The file the calling thread encodes is the one a pool of several
/// threads encodes, and the one a pool of one does, which codes a band a
/// macroblock row at a time as the calling thread does but codes the tiles
/// of a row through the pool.
#[test]
fn an_encode_on_the_calling_thread_equals_one_through_the_pool() {
    let samples = pixels();
    let image = image(&samples);
    let (pool, one_thread) = (pool(4), pool(1));
    let cpu = CpuCapabilities::detect();
    for encode in requests() {
        let sequential = encode_annex_a(&image, &encode, cpu.on_calling_thread()).unwrap();
        let pooled = pool
            .install(|| encode_annex_a(&image, &encode, cpu))
            .unwrap();
        assert!(sequential == pooled, "{encode:?} through the pool");
        let single = one_thread
            .install(|| encode_annex_a(&image, &encode, cpu))
            .unwrap();
        assert!(sequential == single, "{encode:?} through a pool of one");
    }
}

#[test]
fn a_decode_on_the_calling_thread_equals_one_through_the_pool() {
    let samples = pixels();
    let image = image(&samples);
    let pool = pool(4);
    let cpu = CpuCapabilities::detect();
    for encode in requests() {
        let file = encode_annex_a(&image, &encode, cpu.on_calling_thread()).unwrap();
        let parsed = parse_codestream(&file).unwrap();
        for request in decode_requests() {
            let decode = |cpu| {
                prepare_plan(file.len(), &parsed, &request)
                    .and_then(|plan| decode_cpu(&file, &parsed, &plan, &request, cpu))
            };
            let pooled = pool.install(|| decode(cpu));
            let sequential = decode(cpu.on_calling_thread());
            assert_eq!(sequential, pooled, "{encode:?} {request:?}");
        }
    }
}
