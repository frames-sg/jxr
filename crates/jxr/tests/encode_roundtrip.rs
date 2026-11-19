//! Encoded codestreams decoded back through the public CPU decoder.

use jxr::{
    AlphaMode, AnnexABitDepth, AnnexAPixelFamily, AnnexAPixelFormat, AnnexAPixelFormatDescriptor,
    AnnexAWriteOptions, BandPresence, BitstreamMode, ChannelLayout, ChromaSampling, ColorFormat,
    DecodeRequest, DecodeScale, DecodedSamples, EncodeQuantizers, EncodeRequest, JxrErrorKind,
    JxrView, OverlapMode, PixelFormat, Profile, SampleFormat, SourceImage, SourcePlane,
    SourceSamples, encode, encode_codestream, encode_codestreams, write_annex_a,
};

const GRAY8: PixelFormat = PixelFormat::U8(ChannelLayout::Luma);

fn size(value: u32) -> usize {
    usize::try_from(value).unwrap()
}

/// A gradient with texture and noise, so every band carries energy.
fn gray_image(width: u32, height: u32, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let mut pixels = Vec::with_capacity(size(width) * size(height));
    for y in 0..height {
        for x in 0..width {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let ramp = x * 2 + y * 3;
            let texture = if (x / 5 + y / 7).is_multiple_of(2) {
                24
            } else {
                0
            };
            let noise = u32::try_from((state >> 40) % 32).unwrap();
            pixels.push(u8::try_from((ramp + texture + noise) % 256).unwrap());
        }
    }
    pixels
}

fn decode_gray(encoded: &[u8]) -> (u32, u32, Vec<u8>) {
    let view = JxrView::parse(encoded).unwrap();
    let (width, height) = view.info().dimensions();
    (width, height, decode_as(encoded, GRAY8))
}

fn encode_gray(pixels: &[u8], width: u32, height: u32, request: &EncodeRequest) -> Vec<u8> {
    let image = SourceImage::new(width, height, GRAY8, SourceSamples::U8(pixels)).unwrap();
    encode_codestream(&image, request).unwrap()
}

#[test]
fn lossless_gray_round_trips_for_every_overlap_mode_and_size() {
    // Sizes cover a single partial macroblock, exact multiples, and margins.
    for (width, height) in [(1, 1), (16, 16), (37, 21), (64, 48), (130, 75)] {
        let pixels = gray_image(width, height, u64::from(width * 31 + height));
        for overlap in [OverlapMode::None, OverlapMode::One, OverlapMode::Two] {
            let request = EncodeRequest::lossless().with_overlap(overlap);
            let codestream = encode_gray(&pixels, width, height, &request);
            let (decoded_width, decoded_height, decoded) = decode_gray(&codestream);
            assert_eq!((decoded_width, decoded_height), (width, height));
            assert_eq!(decoded, pixels, "{width}x{height} {overlap:?}");
        }
    }
}

/// Whether two images differ anywhere by more than `tolerance`.
fn differs_beyond(first: &[u8], second: &[u8], tolerance: u8) -> bool {
    first.len() != second.len()
        || first
            .iter()
            .zip(second)
            .any(|(a, b)| a.abs_diff(*b) > tolerance)
}

#[test]
fn soft_tile_size_changes_the_coding_but_not_the_decoded_samples() {
    // Soft tiles restart prediction and entropy adaptation, but the
    // transforms and overlap filters run across their edges, so every tile
    // size reconstructs the same samples, lossy or not.
    let (width, height) = (150, 100);
    let gray = gray_image(width, height, 41);
    let rgb = colour_image(width, height, 3, 43);
    let source = SourceImage::new(width, height, RGB8, SourceSamples::U8(&rgb)).unwrap();
    for overlap in OVERLAPS {
        for mode in [BitstreamMode::Spatial, BitstreamMode::Frequency] {
            for (lossless, base) in [
                (true, EncodeRequest::lossless()),
                (false, EncodeRequest::lossy(EncodeQuantizers::uniform(40))),
                (false, EncodeRequest::quality(75)),
            ] {
                let request = |max_tile| {
                    base.clone()
                        .with_overlap(overlap)
                        .with_bitstream_mode(mode)
                        .with_max_tile_macroblocks(max_tile)
                };
                let case = format!("{overlap:?} {mode:?} {:?}", base.quantization);
                let one_gray = encode_gray(&gray, width, height, &request(255));
                let one_rgb = encode(&source, &request(255)).unwrap();
                let (gray_samples, rgb_samples) =
                    (decode_gray(&one_gray).2, decode_as(&one_rgb, RGB8));
                assert_eq!(
                    gray_samples == gray && rgb_samples == rgb,
                    lossless,
                    "{case}"
                );
                // A lossy image is still close to its source.
                assert!(!differs_beyond(&gray_samples, &gray, 40), "{case}");
                for max_tile in [1, 2, 3, 5] {
                    let tiled_gray = encode_gray(&gray, width, height, &request(max_tile));
                    let tiled_rgb = encode(&source, &request(max_tile)).unwrap();
                    assert_ne!(tiled_gray, one_gray, "{case} tiles {max_tile}");
                    assert_ne!(tiled_rgb, one_rgb, "{case} tiles {max_tile}");
                    assert_eq!(
                        decode_gray(&tiled_gray).2,
                        gray_samples,
                        "{case} tiles {max_tile}"
                    );
                    assert_eq!(
                        decode_as(&tiled_rgb, RGB8),
                        rgb_samples,
                        "{case} tiles {max_tile}"
                    );
                }
            }
        }
    }
}

/// A decode in a thread pool shares the tile rows of the image out among
/// strips, each streamed down its own rows, and stores the rows either side
/// of every boundary between two strips from a strip of their own. A decode
/// on one thread is a single strip. Both must store the same samples: for
/// every overlap mode, hard and soft tiles, either packet layout, chroma at
/// every sampling, both alpha modes, and a region as well as the image.
#[test]
fn a_decode_through_a_thread_pool_stores_what_a_decode_on_one_thread_does() {
    let (width, height) = (70, 400);
    let rgba_format = PixelFormat::U8(ChannelLayout::Rgba);
    let rgba = colour_image(width, height, 4, 47);
    let source = SourceImage::new(width, height, rgba_format, SourceSamples::U8(&rgba)).unwrap();
    let pool = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    };
    let (one, several) = (pool(1), pool(5));
    let region = jxr::Rect {
        x: 9,
        y: 43,
        w: 50,
        h: 301,
    };
    let samplings = [
        ChromaSampling::Cs444,
        ChromaSampling::Cs422,
        ChromaSampling::Cs420,
    ];
    for overlap in OVERLAPS {
        for mode in [BitstreamMode::Spatial, BitstreamMode::Frequency] {
            for (max_tile, hard) in [(1, false), (1, true), (3, false), (3, true), (255, false)] {
                for sampling in samplings {
                    // The one layout the encoder refuses: the low-pass-level
                    // filter of subsampled chroma has no room in a hard tile
                    // one macroblock tall.
                    if overlap == OverlapMode::Two
                        && hard
                        && max_tile == 1
                        && sampling != ChromaSampling::Cs444
                    {
                        continue;
                    }
                    for alpha_mode in ALPHA_MODES {
                        let request = EncodeRequest::lossy(EncodeQuantizers::uniform(24))
                            .with_overlap(overlap)
                            .with_bitstream_mode(mode)
                            .with_max_tile_macroblocks(max_tile)
                            .with_hard_tiles(hard)
                            .with_chroma_sampling(sampling)
                            .with_alpha_mode(alpha_mode);
                        let case = format!(
                            "{overlap:?} {mode:?} tiles {max_tile} hard {hard} {sampling:?} \
                             {alpha_mode:?}"
                        );
                        let file = encode(&source, &request).unwrap();
                        let view = JxrView::parse(&file).unwrap();
                        let whole = DecodeRequest::new(rgba_format);
                        for request in [whole.clone(), whole.with_region(region)] {
                            let decode = || view.decoder().decode(&request).unwrap();
                            let expected = one.install(decode);
                            let stored = several.install(decode);
                            assert_eq!(stored.samples, expected.samples, "{case}");
                            assert_eq!(stored.planes, expected.planes, "{case}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn the_default_tile_size_splits_an_image_larger_than_one_tile() {
    // 560 samples are 35 macroblocks: two default tiles across, one down.
    let (width, height) = (560, 40);
    let rgb = colour_image(width, height, 3, 47);
    let source = SourceImage::new(width, height, RGB8, SourceSamples::U8(&rgb)).unwrap();
    for base in [EncodeRequest::lossless(), EncodeRequest::quality(75)] {
        assert_eq!(
            base.max_tile_macroblocks,
            EncodeRequest::DEFAULT_MAX_TILE_MACROBLOCKS
        );
        let file = encode(&source, &base).unwrap();
        let view = JxrView::parse(&file).unwrap();
        assert_eq!(view.info().tiles.column_widths.len(), 2);
        assert_eq!(view.info().tiles.row_heights.len(), 1);
        assert!(!view.info().tiles.hard_tiles);
        // The largest tiles a request can ask for keep it one tile, with
        // the same decoded samples and no more bytes.
        let one_tile = encode(&source, &base.clone().with_max_tile_macroblocks(255)).unwrap();
        let view = JxrView::parse(&one_tile).unwrap();
        assert_eq!(view.info().tiles.column_widths.len(), 1);
        assert_eq!(decode_as(&file, RGB8), decode_as(&one_tile, RGB8));
        assert!(one_tile.len() < file.len());
    }
}

/// A lossless request that codes a row of forty macroblocks as one tile,
/// which the default tile size would split in two.
fn one_wide_tile() -> EncodeRequest {
    EncodeRequest::lossless().with_max_tile_macroblocks(255)
}

#[test]
fn a_wide_tile_row_exercises_every_adaptation_boundary() {
    // Forty macroblocks per row cross two 16-macroblock context resets.
    let (width, height) = (640, 40);
    let pixels = gray_image(width, height, 9);
    let codestream = encode_gray(&pixels, width, height, &one_wide_tile());
    let view = JxrView::parse(&codestream).unwrap();
    assert_eq!(view.info().tiles.column_widths.len(), 1);
    assert_eq!(decode_gray(&codestream).2, pixels);
}

#[test]
fn row_stride_padding_is_not_encoded() {
    let (width, height, stride) = (20_u32, 9_u32, 27_usize);
    let pixels = gray_image(width, height, 3);
    let row = size(width);
    let mut padded = vec![0xee_u8; stride * (size(height) - 1) + row];
    for (y, source) in pixels.chunks_exact(row).enumerate() {
        padded[y * stride..y * stride + row].copy_from_slice(source);
    }
    let image = SourceImage::new(width, height, GRAY8, SourceSamples::U8(&padded))
        .unwrap()
        .with_row_stride(stride)
        .unwrap();
    let codestream = encode_codestream(&image, &EncodeRequest::lossless()).unwrap();
    assert_eq!(decode_gray(&codestream).2, pixels);
}

#[test]
fn annex_a_file_declares_the_source_format_and_decodes_identically() {
    let (width, height) = (45, 30);
    let pixels = gray_image(width, height, 41);
    let image = SourceImage::new(width, height, GRAY8, SourceSamples::U8(&pixels)).unwrap();
    let request = EncodeRequest::lossless();
    let file = encode(&image, &request).unwrap();
    let view = JxrView::parse(&file).unwrap();
    // The container wraps exactly the raw codestream the encoder produces.
    assert_eq!(
        view.codestream(),
        encode_codestream(&image, &request).unwrap()
    );
    assert!(view.info().metadata.annex_a_pixel_format.is_some());
    assert_eq!(decode_gray(&file).2, pixels);
}

#[test]
fn lossy_error_grows_with_the_quantizer_and_size_shrinks() {
    let (width, height) = (128, 96);
    let pixels = gray_image(width, height, 17);
    let mut previous_size = usize::MAX;
    let mut previous_error = 0_u64;
    for parameter in [8, 24, 48, 80] {
        let request = EncodeRequest::lossy(EncodeQuantizers::uniform(parameter));
        let codestream = encode_gray(&pixels, width, height, &request);
        let (_, _, decoded) = decode_gray(&codestream);
        let squared_error: u64 = decoded
            .iter()
            .zip(&pixels)
            .map(|(&a, &b)| u64::from(a.abs_diff(b)).pow(2))
            .sum();
        assert!(squared_error > 0, "parameter {parameter} is lossy");
        assert!(
            codestream.len() < previous_size,
            "parameter {parameter}: {} bytes after {previous_size}",
            codestream.len()
        );
        assert!(
            squared_error > previous_error,
            "parameter {parameter}: error {squared_error} after {previous_error}"
        );
        previous_size = codestream.len();
        previous_error = squared_error;
    }
}

#[test]
fn reduced_band_sets_decode_and_shrink_the_codestream() {
    let (width, height) = (96, 64);
    let pixels = gray_image(width, height, 23);
    let mut previous_size = usize::MAX;
    for bands in [
        BandPresence::All,
        BandPresence::NoFlexbits,
        BandPresence::NoHighPass,
        BandPresence::DcOnly,
    ] {
        let request = EncodeRequest::lossy(EncodeQuantizers::uniform(20)).with_bands(bands);
        let codestream = encode_gray(&pixels, width, height, &request);
        let view = JxrView::parse(&codestream).unwrap();
        assert_eq!(view.info().primary.bands, bands);
        let (_, _, decoded) = decode_gray(&codestream);
        assert_eq!(decoded.len(), pixels.len());
        assert!(codestream.len() <= previous_size, "{bands:?}");
        previous_size = codestream.len();
    }
}

#[test]
fn trimmed_flexbits_decode_with_bounded_extra_error() {
    let (width, height) = (96, 64);
    let pixels = gray_image(width, height, 29);
    let base = EncodeRequest::lossless();
    let untrimmed = encode_gray(&pixels, width, height, &base);
    for trim in [1, 3, 6, 15] {
        let request = base.clone().with_trim_flexbits(trim);
        let codestream = encode_gray(&pixels, width, height, &request);
        assert!(codestream.len() < untrimmed.len(), "trim {trim}");
        let (_, _, decoded) = decode_gray(&codestream);
        let worst = largest_error(&pixels, &decoded);
        // Only high-pass refinement bits are dropped, so DC and LP stay exact
        // and the sample error stays far below the full range.
        assert!(worst < 128, "trim {trim}: worst sample error {worst}");
    }
}

/// Interleaved colour with independent structure in every channel.
fn colour_image(width: u32, height: u32, channels: usize, seed: u64) -> Vec<u8> {
    let planes: Vec<Vec<u8>> = (0..channels)
        .map(|channel| {
            let offset = u32::try_from(channel).unwrap();
            gray_image(width + offset * 3, height, seed + u64::from(offset) * 101)
        })
        .collect();
    let mut pixels = Vec::with_capacity(size(width) * size(height) * channels);
    for y in 0..size(height) {
        for x in 0..size(width) {
            for (channel, plane) in planes.iter().enumerate() {
                let stride = size(width) + channel * 3;
                pixels.push(plane[y * stride + x + channel]);
            }
        }
    }
    pixels
}

fn decode_as(encoded: &[u8], format: PixelFormat) -> Vec<u8> {
    let view = JxrView::parse(encoded).unwrap();
    tight_rows(&view.decoder().decode(&DecodeRequest::new(format)).unwrap())
}

#[test]
fn lossless_colour_round_trips_in_every_channel_order() {
    for layout in [ChannelLayout::Rgb, ChannelLayout::Bgr] {
        let format = PixelFormat::U8(layout);
        for (width, height) in [(1, 1), (37, 21), (130, 75)] {
            let pixels = colour_image(width, height, 3, u64::from(width + height));
            let image =
                SourceImage::new(width, height, format, SourceSamples::U8(&pixels)).unwrap();
            for overlap in [OverlapMode::None, OverlapMode::One, OverlapMode::Two] {
                let request = EncodeRequest::lossless()
                    .with_overlap(overlap)
                    .with_max_tile_macroblocks(3);
                let file = encode(&image, &request).unwrap();
                assert_eq!(
                    decode_as(&file, format),
                    pixels,
                    "{layout:?} {width}x{height} {overlap:?}"
                );
            }
        }
    }
}

#[test]
fn channel_order_is_carried_by_the_container_not_the_samples() {
    let (width, height) = (40, 24);
    let rgb = colour_image(width, height, 3, 77);
    let bgr: Vec<u8> = rgb
        .as_chunks::<3>()
        .0
        .iter()
        .flat_map(|&[red, green, blue]| [blue, green, red])
        .collect();
    let rgb_image = SourceImage::new(
        width,
        height,
        PixelFormat::U8(ChannelLayout::Rgb),
        SourceSamples::U8(&rgb),
    )
    .unwrap();
    let bgr_image = SourceImage::new(
        width,
        height,
        PixelFormat::U8(ChannelLayout::Bgr),
        SourceSamples::U8(&bgr),
    )
    .unwrap();
    let request = EncodeRequest::lossless();
    // The same picture codes to the same codestream whichever order it was
    // supplied in, and either file decodes to either order.
    assert_eq!(
        encode_codestream(&rgb_image, &request).unwrap(),
        encode_codestream(&bgr_image, &request).unwrap()
    );
    let from_bgr = encode(&bgr_image, &request).unwrap();
    assert_eq!(
        decode_as(&from_bgr, PixelFormat::U8(ChannelLayout::Rgb)),
        rgb
    );
    assert_eq!(
        decode_as(&from_bgr, PixelFormat::U8(ChannelLayout::Bgr)),
        bgr
    );
}

#[test]
fn padded_colour_storage_ignores_its_padding_channel() {
    let (width, height) = (33, 18);
    let bgr = colour_image(width, height, 3, 55);
    let bgrx: Vec<u8> = bgr
        .as_chunks::<3>()
        .0
        .iter()
        .flat_map(|&[blue, green, red]| [blue, green, red, 0xa5])
        .collect();
    let image = SourceImage::new(
        width,
        height,
        PixelFormat::U8(ChannelLayout::Bgrx),
        SourceSamples::U8(&bgrx),
    )
    .unwrap();
    let file = encode(&image, &EncodeRequest::lossless()).unwrap();
    assert_eq!(decode_as(&file, PixelFormat::U8(ChannelLayout::Bgr)), bgr);
}

#[test]
fn lossy_colour_honours_separate_chroma_quantizers() {
    let (width, height) = (96, 64);
    let format = PixelFormat::U8(ChannelLayout::Rgb);
    let pixels = colour_image(width, height, 3, 31);
    let image = SourceImage::new(width, height, format, SourceSamples::U8(&pixels)).unwrap();
    let error = |file: &[u8]| -> u64 {
        decode_as(file, format)
            .iter()
            .zip(&pixels)
            .map(|(&a, &b)| u64::from(a.abs_diff(b)).pow(2))
            .sum()
    };
    let uniform = EncodeRequest::lossy(EncodeQuantizers::uniform(24));
    let coarse_chroma = uniform
        .clone()
        .with_chroma_quantizers(EncodeQuantizers::uniform(72));
    assert!(!coarse_chroma.is_lossless());
    let uniform_file = encode(&image, &uniform).unwrap();
    let coarse_file = encode(&image, &coarse_chroma).unwrap();
    assert!(error(&uniform_file) > 0);
    assert!(coarse_file.len() < uniform_file.len());
    assert!(error(&coarse_file) > error(&uniform_file));
    // A lossless luma request is still lossy once chroma is quantized.
    let mixed = EncodeRequest::lossless().with_chroma_quantizers(EncodeQuantizers::uniform(40));
    assert!(!mixed.is_lossless());
    assert!(error(&encode(&image, &mixed).unwrap()) > 0);
}

#[test]
fn unsupported_source_formats_are_rejected_not_approximated() {
    let pixels = [0_u8; 4 * 4 * 10];
    for layout in [
        // The container format table has no two- or nine-channel format,
        // with or without alpha, and no gray format with alpha.
        ChannelLayout::NComponent(2),
        ChannelLayout::NComponent(9),
        ChannelLayout::NComponentAlpha(2),
        ChannelLayout::NComponentAlpha(9),
        ChannelLayout::LumaAlpha,
    ] {
        let format = PixelFormat::U8(layout);
        let image = SourceImage::new(4, 4, format, SourceSamples::U8(&pixels)).unwrap();
        for request in [
            EncodeRequest::lossless(),
            EncodeRequest::lossless().with_cmyk_direct(true),
        ] {
            let error = encode_codestream(&image, &request).unwrap_err();
            assert_eq!(error.kind, JxrErrorKind::Unsupported, "{layout:?}");
            assert_eq!(
                encode(&image, &request).unwrap_err().kind,
                JxrErrorKind::Unsupported,
                "{layout:?}"
            );
        }
    }
}

/// Deterministic 32-bit values for building samples of any type.
fn values(count: usize, seed: u64) -> Vec<u32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            u32::try_from(state >> 32).unwrap()
        })
        .collect()
}

/// Encode losslessly and decode back in the same pixel format.
fn round_trip(
    width: u32,
    height: u32,
    format: PixelFormat,
    source: &DecodedSamples,
) -> DecodedSamples {
    recode(width, height, format, source, &EncodeRequest::lossless()).1
}

#[test]
fn every_integer_depth_round_trips_losslessly() {
    let (width, height) = (40_u32, 24_u32);
    let pixels = size(width) * size(height);
    let low16 = |value: u32| u16::try_from(value & 0xffff).unwrap();
    let cases = [
        (
            PixelFormat::U16(ChannelLayout::Luma),
            DecodedSamples::U16(values(pixels, 1).into_iter().map(low16).collect()),
        ),
        (
            PixelFormat::U16(ChannelLayout::Rgb),
            DecodedSamples::U16(values(pixels * 3, 2).into_iter().map(low16).collect()),
        ),
        (
            PixelFormat::I16(ChannelLayout::Luma),
            DecodedSamples::I16(
                values(pixels, 3)
                    .into_iter()
                    .map(|value| low16(value).cast_signed())
                    .collect(),
            ),
        ),
        (
            PixelFormat::I16(ChannelLayout::Rgb),
            DecodedSamples::I16(
                values(pixels * 3, 4)
                    .into_iter()
                    .map(|value| low16(value).cast_signed())
                    .collect(),
            ),
        ),
        (
            // The default shift keeps 22 of 32 bits: values are multiples of
            // 1024 within the range the transforms have headroom for.
            PixelFormat::I32(ChannelLayout::Luma),
            DecodedSamples::I32(
                values(pixels, 5)
                    .into_iter()
                    .map(|value| (value.cast_signed() >> 6) & !0x3ff)
                    .collect(),
            ),
        ),
        (
            PixelFormat::BitPacked(ChannelLayout::Luma),
            DecodedSamples::BitPacked(
                values(size(width).div_ceil(8) * size(height), 6)
                    .into_iter()
                    .map(|value| value.to_le_bytes()[0])
                    .collect(),
            ),
        ),
        (
            PixelFormat::Rgb555,
            DecodedSamples::Rgb555(
                values(pixels, 7)
                    .into_iter()
                    .map(|value| low16(value) & 0x7fff)
                    .collect(),
            ),
        ),
        (
            PixelFormat::Rgb565,
            DecodedSamples::Rgb565(values(pixels, 8).into_iter().map(low16).collect()),
        ),
        (
            PixelFormat::Rgb101010,
            DecodedSamples::Rgb101010(
                values(pixels, 9)
                    .into_iter()
                    .map(|value| value & 0x3fff_ffff)
                    .collect(),
            ),
        ),
    ];
    for (format, source) in cases {
        assert_eq!(
            round_trip(width, height, format, &source),
            source,
            "{format:?}"
        );
    }
}

#[test]
fn half_float_bit_patterns_round_trip_losslessly() {
    let (width, height) = (40_u32, 24_u32);
    for layout in [ChannelLayout::Luma, ChannelLayout::Rgb] {
        let format = PixelFormat::F16(layout);
        let count = size(width) * size(height) * usize::from(format.channel_count());
        // Any pattern but negative zero, which has no distinct coded sample.
        let source = DecodedSamples::F16(
            values(count, 11)
                .into_iter()
                .map(|value| u16::try_from(value & 0xffff).unwrap())
                .map(|bits| if bits == 0x8000 { 0 } else { bits })
                .collect(),
        );
        assert_eq!(
            round_trip(width, height, format, &source),
            source,
            "{layout:?}"
        );
    }
}

#[test]
fn floats_keep_the_configured_mantissa_and_recode_exactly() {
    let (width, height) = (40_u32, 24_u32);
    let format = PixelFormat::F32(ChannelLayout::Luma);
    let source: Vec<f32> = values(size(width) * size(height), 13)
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            // Magnitudes from 2^-10 to 2^9, both signs, plus exact zeros.
            let mantissa = 1.0 + f32::from(u16::try_from(value & 0xffff).unwrap()) / 65_536.0;
            let exponent = i32::try_from(value >> 16 & 0x1f).unwrap() % 20 - 10;
            let magnitude = mantissa * 2.0_f32.powi(exponent);
            match index % 7 {
                0 => 0.0,
                1 | 2 => -magnitude,
                _ => magnitude,
            }
        })
        .collect();
    let DecodedSamples::F32(decoded) =
        round_trip(width, height, format, &DecodedSamples::F32(source.clone()))
    else {
        panic!("float decode returned another storage");
    };
    for (&original, &value) in source.iter().zip(&decoded) {
        // Thirteen mantissa bits above 2^-3; an absolute step of 2^-16 below.
        let tolerance = (original.abs() * 2.0_f32.powi(-13)).max(2.0_f32.powi(-17));
        assert!(
            (value - original).abs() <= tolerance,
            "{original} decoded as {value}"
        );
    }
    // What the decoder returns is exactly representable, so coding it again
    // changes nothing.
    let decoded = DecodedSamples::F32(decoded);
    assert_eq!(round_trip(width, height, format, &decoded), decoded);
}

#[test]
fn shared_exponent_pixels_round_trip_losslessly() {
    let (width, height) = (40_u32, 24_u32);
    let source: Vec<u32> = values(size(width) * size(height), 17)
        .into_iter()
        .map(|value| {
            let [red, green, blue, exponent] = value.to_le_bytes();
            // Canonical pixels: black is all zero, and otherwise the
            // brightest mantissa has its leading bit set.
            if exponent == 0 {
                0
            } else {
                u32::from_le_bytes([red | 0x80, green, blue, exponent])
            }
        })
        .collect();
    let source = DecodedSamples::Rgbe(source);
    assert_eq!(
        round_trip(width, height, PixelFormat::Rgbe, &source),
        source
    );
}

#[test]
fn mismatched_storage_and_format_are_rejected() {
    let samples = [0_u16; 16];
    let error = SourceImage::new(
        4,
        4,
        PixelFormat::U8(ChannelLayout::Luma),
        SourceSamples::U16(&samples),
    )
    .unwrap_err();
    assert_eq!(error.kind, JxrErrorKind::InvalidRequest);
}

/// Interleaved samples of `channels` unrelated channels at eight or sixteen
/// bits: full-range noise over a per-channel gradient, so every channel
/// carries energy in every band and none predicts another.
fn multichannel_samples(
    width: u32,
    height: u32,
    channels: u16,
    sixteen_bit: bool,
    seed: u64,
) -> DecodedSamples {
    let noise = values(size(width) * size(height) * usize::from(channels), seed);
    let mut samples = Vec::with_capacity(noise.len());
    let mut noise = noise.into_iter();
    for y in 0..height {
        for x in 0..width {
            for channel in 0..u32::from(channels) {
                let ramp = x * (channel + 1) * 257 + y * (7 - channel % 7) * 131;
                // Every third pixel is unconstrained noise, extremes included.
                let value = noise.next().unwrap();
                let sample = if (x + y + channel).is_multiple_of(3) {
                    value
                } else {
                    ramp + (value & 0x3ff)
                };
                samples.push(u16::try_from(sample & 0xffff).unwrap());
            }
        }
    }
    if sixteen_bit {
        DecodedSamples::U16(samples)
    } else {
        DecodedSamples::U8(
            samples
                .into_iter()
                .map(|sample| sample.to_be_bytes()[0])
                .collect(),
        )
    }
}

fn multichannel_format(layout: ChannelLayout, sixteen_bit: bool) -> PixelFormat {
    if sixteen_bit {
        PixelFormat::U16(layout)
    } else {
        PixelFormat::U8(layout)
    }
}

/// Encode with `request` and decode back in the same pixel format.
fn recode(
    width: u32,
    height: u32,
    format: PixelFormat,
    source: &DecodedSamples,
    request: &EncodeRequest,
) -> (Vec<u8>, DecodedSamples) {
    let image = SourceImage::new(width, height, format, SourceSamples::from(source)).unwrap();
    let file = encode(&image, request).unwrap();
    let view = JxrView::parse(&file).unwrap();
    assert_eq!(view.info().dimensions(), (width, height));
    let decoded = view
        .decoder()
        .decode(&DecodeRequest::new(format))
        .unwrap()
        .samples;
    (file, decoded)
}

/// Sample values widened for error measurement.
fn widened(samples: &DecodedSamples) -> Vec<u32> {
    match samples {
        DecodedSamples::U8(samples) => samples.iter().map(|&sample| u32::from(sample)).collect(),
        DecodedSamples::U16(samples) => samples.iter().map(|&sample| u32::from(sample)).collect(),
        other => panic!("unexpected storage {other:?}"),
    }
}

/// Mean absolute sample error as a fraction of the sample range.
fn relative_error(source: &DecodedSamples, decoded: &DecodedSamples) -> f64 {
    let range = match source {
        DecodedSamples::U8(_) => 255.0,
        _ => 65_535.0,
    };
    quality::error_against(source, decoded, range)
}

#[test]
fn cmyk_round_trips_losslessly_with_and_without_the_colour_transform() {
    for sixteen_bit in [false, true] {
        let format = multichannel_format(ChannelLayout::Cmyk, sixteen_bit);
        for (width, height) in [(1, 1), (37, 21), (130, 75)] {
            let source = multichannel_samples(width, height, 4, sixteen_bit, 61);
            for direct in [false, true] {
                for overlap in [OverlapMode::None, OverlapMode::One, OverlapMode::Two] {
                    let request = EncodeRequest::lossless()
                        .with_overlap(overlap)
                        .with_max_tile_macroblocks(3)
                        .with_cmyk_direct(direct);
                    let (file, decoded) = recode(width, height, format, &source, &request);
                    assert_eq!(
                        decoded, source,
                        "{format:?} {width}x{height} direct {direct} {overlap:?}"
                    );
                    assert_eq!(
                        declared_format(&file).family,
                        AnnexAPixelFamily::Cmyk { direct }
                    );
                    let view = JxrView::parse(&file).unwrap();
                    assert_eq!(view.info().primary.color_format, ColorFormat::YuvK);
                    // Only the transformed form is within the Main profile.
                    let profile = if direct {
                        Profile::Advanced
                    } else {
                        Profile::Main
                    };
                    assert_eq!(view.info().profile, Some(profile));
                }
            }
        }
    }
}

#[test]
fn direct_cmyk_codes_the_channels_untransformed() {
    // Correlated channels cost fewer bits through the colour transform, and
    // unrelated ones do not, so the two forms are different codestreams of
    // the same picture.
    let (width, height) = (64, 48);
    let gray = gray_image(width, height, 71);
    let correlated: Vec<u8> = gray.iter().flat_map(|&value| [value; 4]).collect();
    let source = DecodedSamples::U8(correlated);
    let format = PixelFormat::U8(ChannelLayout::Cmyk);
    let lossless = EncodeRequest::lossless();
    let (transformed, from_transformed) = recode(width, height, format, &source, &lossless);
    let (direct, from_direct) = recode(
        width,
        height,
        format,
        &source,
        &lossless.with_cmyk_direct(true),
    );
    assert_eq!(from_transformed, source);
    assert_eq!(from_direct, source);
    assert!(
        transformed.len() < direct.len(),
        "transformed {} bytes, direct {} bytes",
        transformed.len(),
        direct.len()
    );
}

#[test]
fn n_component_images_round_trip_losslessly_at_every_channel_count() {
    for channels in 3..=8_u16 {
        for sixteen_bit in [false, true] {
            let format = multichannel_format(ChannelLayout::NComponent(channels), sixteen_bit);
            for (width, height) in [(1, 1), (37, 21), (130, 75)] {
                let source =
                    multichannel_samples(width, height, channels, sixteen_bit, u64::from(channels));
                for overlap in [OverlapMode::None, OverlapMode::One, OverlapMode::Two] {
                    let request = EncodeRequest::lossless()
                        .with_overlap(overlap)
                        .with_max_tile_macroblocks(3);
                    let (file, decoded) = recode(width, height, format, &source, &request);
                    assert_eq!(decoded, source, "{format:?} {width}x{height} {overlap:?}");
                    assert_eq!(declared_format(&file).family, AnnexAPixelFamily::NComponent);
                    let view = JxrView::parse(&file).unwrap();
                    assert_eq!(
                        view.info().primary.color_format,
                        ColorFormat::NComponent(channels)
                    );
                    assert_eq!(view.info().profile, Some(Profile::Main));
                }
            }
        }
    }
}

#[test]
fn a_wide_multichannel_tile_row_exercises_every_adaptation_boundary() {
    // Forty macroblocks per row cross two 16-macroblock context resets.
    let (width, height) = (640, 40);
    for (layout, channels) in [
        (ChannelLayout::Cmyk, 4),
        (ChannelLayout::NComponent(5), 5),
        (ChannelLayout::NComponent(8), 8),
    ] {
        let format = PixelFormat::U8(layout);
        let source = multichannel_samples(width, height, channels, false, 83);
        let (_, decoded) = recode(width, height, format, &source, &one_wide_tile());
        assert_eq!(decoded, source, "{layout:?}");
    }
}

#[test]
fn lossy_cmyk_and_n_component_stay_close_and_shrink() {
    let (width, height) = (96, 64);
    let cases = [
        (ChannelLayout::Cmyk, false),
        (ChannelLayout::Cmyk, true),
        (ChannelLayout::NComponent(4), false),
        (ChannelLayout::NComponent(5), true),
        (ChannelLayout::NComponent(8), false),
    ];
    for (layout, sixteen_bit) in cases {
        let format = multichannel_format(layout, sixteen_bit);
        // A smooth picture: quantization error stays near the step size.
        let smooth = colour_image(width, height, usize::from(layout.channel_count()), 97);
        let source = if sixteen_bit {
            DecodedSamples::U16(
                smooth
                    .into_iter()
                    .map(|value| u16::from(value) * 257)
                    .collect(),
            )
        } else {
            DecodedSamples::U8(smooth)
        };
        let (lossless, _) = recode(width, height, format, &source, &EncodeRequest::lossless());
        // Sixteen-bit samples carry eight more bits under the same step.
        let parameters = if sixteen_bit { [80, 112] } else { [24, 56] };
        let mut previous = (lossless.len(), 0.0_f64);
        for parameter in parameters {
            for direct in [false, true] {
                if direct && layout != ChannelLayout::Cmyk {
                    continue;
                }
                let request = EncodeRequest::lossy(EncodeQuantizers::uniform(parameter))
                    .with_cmyk_direct(direct);
                let (file, decoded) = recode(width, height, format, &source, &request);
                let error = relative_error(&source, &decoded);
                assert!(error > 0.0, "{format:?} qp {parameter} is lossy");
                assert!(error < 0.05, "{format:?} qp {parameter}: error {error}");
                assert!(file.len() < lossless.len(), "{format:?} qp {parameter}");
                if !direct {
                    assert!(file.len() < previous.0, "{format:?} qp {parameter}");
                    assert!(error > previous.1, "{format:?} qp {parameter}");
                    previous = (file.len(), error);
                }
            }
        }
        // Coarser quantizers on every component but the first cost quality
        // and save bytes.
        let uniform = EncodeRequest::lossy(EncodeQuantizers::uniform(parameters[0]));
        let coarse = uniform
            .clone()
            .with_chroma_quantizers(EncodeQuantizers::uniform(parameters[1]));
        let (uniform_file, uniform_decoded) = recode(width, height, format, &source, &uniform);
        let (coarse_file, coarse_decoded) = recode(width, height, format, &source, &coarse);
        assert!(coarse_file.len() < uniform_file.len(), "{format:?}");
        assert!(
            relative_error(&source, &coarse_decoded) > relative_error(&source, &uniform_decoded),
            "{format:?}"
        );
    }
}

const ALPHA_MODES: [AlphaMode; 2] = [AlphaMode::Separate, AlphaMode::Integrated];
const OVERLAPS: [OverlapMode; 3] = [OverlapMode::None, OverlapMode::One, OverlapMode::Two];
const RGB8: PixelFormat = PixelFormat::U8(ChannelLayout::Rgb);

fn declared_format(file: &[u8]) -> AnnexAPixelFormatDescriptor {
    let view = JxrView::parse(file).unwrap();
    let Some(AnnexAPixelFormat::Known(descriptor)) = view.info().metadata.annex_a_pixel_format
    else {
        panic!("encoded file declares no known pixel format");
    };
    descriptor
}

fn decode_at(file: &[u8], format: PixelFormat, scale: DecodeScale) -> jxr::DecodedImage {
    let view = JxrView::parse(file).unwrap();
    view.decoder()
        .decode(&DecodeRequest::new(format).with_scale(scale))
        .unwrap()
}

/// The packet layout a file declares for its primary image and, when it has
/// one, for its alpha plane.
fn declared_layouts(file: &[u8]) -> (BitstreamMode, Option<BitstreamMode>) {
    let view = JxrView::parse(file).unwrap();
    let info = view.info();
    (
        info.primary.bitstream_mode,
        info.alpha.as_ref().map(|alpha| alpha.bitstream_mode),
    )
}

fn largest_error(source: &[u8], decoded: &[u8]) -> u8 {
    assert_eq!(source.len(), decoded.len());
    source
        .iter()
        .zip(decoded)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap()
}

/// The samples of a decoded image's first plane without row padding.
fn tight_rows(image: &jxr::DecodedImage) -> Vec<u8> {
    let DecodedSamples::U8(samples) = &image.samples else {
        panic!("decode returned {:?}", image.format);
    };
    let plane = image.planes[0];
    let row = size(plane.width) * usize::from(plane.channels);
    let mut pixels = Vec::new();
    for y in 0..size(plane.height) {
        let start = plane.byte_offset + y * plane.row_stride_bytes;
        pixels.extend_from_slice(&samples[start..start + row]);
    }
    pixels
}

// Sources at the limits of the coded arithmetic, found by the
// `encode_roundtrip` fuzz target: every file the encoder returns decodes.

#[test]
fn a_coefficient_of_i32_min_is_refused_or_round_trips() {
    let format = PixelFormat::I32(ChannelLayout::Luma);
    // The DC coefficient of a flat image is sixteen times its sample.
    for sample in [-(1_i32 << 27), -(1 << 27) + 1, 1 << 27] {
        let samples = [sample];
        let image = SourceImage::new(1, 1, format, SourceSamples::I32(&samples)).unwrap();
        match encode(&image, &EncodeRequest::lossless().with_shift_bits(0)) {
            Err(error) => assert_eq!(error.kind, JxrErrorKind::ArithmeticOverflow, "{sample}"),
            Ok(file) => {
                let decoded = decode_at(&file, format, DecodeScale::Full);
                assert_eq!(decoded.samples, DecodedSamples::I32(samples.to_vec()));
            }
        }
    }
}

#[test]
fn lossy_fixed_point_at_the_top_of_its_range_decodes_clipped_to_it() {
    let format = PixelFormat::I32(ChannelLayout::Luma);
    for top in [i32::MAX & !0x3ff, i32::MIN] {
        let samples = [top; 16 * 16];
        let image = SourceImage::new(16, 16, format, SourceSamples::I32(&samples)).unwrap();
        let file = encode(&image, &EncodeRequest::quality(75)).unwrap();
        let decoded = decode_at(&file, format, DecodeScale::Full);
        let DecodedSamples::I32(decoded) = decoded.samples else {
            panic!("fixed-point output");
        };
        for value in decoded {
            assert!(
                i64::from(value).abs_diff(i64::from(top)) < 1 << 20,
                "{value} from {top}"
            );
        }
    }
}

#[test]
fn lossy_floats_at_the_top_of_their_range_decode() {
    let format = PixelFormat::F32(ChannelLayout::Luma);
    let samples: Vec<f32> = (0..16 * 16)
        .map(|index| {
            if (index / 16 + index % 16) % 2 == 0 {
                f32::MAX
            } else {
                0.0
            }
        })
        .collect();
    let image = SourceImage::new(16, 16, format, SourceSamples::F32(&samples)).unwrap();
    let file = encode(
        &image,
        &EncodeRequest::lossy(EncodeQuantizers::uniform(200)),
    )
    .unwrap();
    // Quantization carries the largest samples past the largest exponent;
    // the decoder clips them instead of refusing the image.
    let decoded = decode_at(&file, format, DecodeScale::Full);
    let DecodedSamples::F32(decoded) = decoded.samples else {
        panic!("float output");
    };
    assert_eq!(decoded.len(), samples.len());
}

#[test]
fn subsampled_chroma_near_the_arithmetic_limit_is_refused_or_decodes() {
    let format = PixelFormat::I32(ChannelLayout::Rgb);
    let m = 3_i32 << 26;
    let samples = [0, 0, 0, m, -m, m, -m, m, -m, 0, 0, 0];
    let image = SourceImage::new(2, 2, format, SourceSamples::I32(&samples)).unwrap();
    for sampling in [ChromaSampling::Cs422, ChromaSampling::Cs420] {
        let request = EncodeRequest::lossless()
            .with_shift_bits(0)
            .with_chroma_sampling(sampling);
        match encode(&image, &request) {
            Err(error) => assert_eq!(error.kind, JxrErrorKind::ArithmeticOverflow),
            Ok(file) => {
                decode_at(&file, format, DecodeScale::Full);
            }
        }
    }
}

#[path = "encode_roundtrip/alpha.rs"]
mod alpha;
#[path = "encode_roundtrip/chroma.rs"]
mod chroma;
#[path = "encode_roundtrip/packet_layout.rs"]
mod packet_layout;
#[path = "encode_roundtrip/quality.rs"]
mod quality;
#[path = "encode_roundtrip/thumbnails.rs"]
mod thumbnails;
