use super::chroma::{SUBSAMPLINGS, YUV420, overlap_needs_wider_tiles, smooth_colour_image};
use super::packet_layout::{TILE_LAYOUTS, encode_file, frequency, layout_sources, with_settings};
use super::*;

/// Interleaved 8-bit pixels of `channels` channels that are constant over
/// every `block` x `block` square of the image, differently in each
/// channel, with the picture of the image at one sample for each square.
fn block_image(width: u32, height: u32, channels: u32, block: u32) -> (Vec<u8>, Vec<u8>) {
    let value = |x: u32, y: u32, channel: u32| {
        u8::try_from(((x / block) * 37 + (y / block) * 101 + channel * 59 + 13) % 256).unwrap()
    };
    let pixels = |step: u32| {
        let mut pixels = Vec::new();
        for y in (0..height).step_by(size(step)) {
            for x in (0..width).step_by(size(step)) {
                pixels.extend((0..channels).map(|channel| value(x, y, channel)));
            }
        }
        pixels
    };
    (pixels(1), pixels(block))
}

/// A reduced decode reads the DC packets, and the low-pass packets for a
/// quarter of the size, and nothing else. An image that is constant over
/// every block a reduced sample stands for has an exact reduced picture,
/// one sample for each block, and a lossless file of it must decode to that
/// picture, in every colour family and both alpha modes.
#[test]
fn reduced_decodes_of_a_frequency_mode_file_reproduce_block_constant_images() {
    for (width, height) in [(64, 48), (70, 50)] {
        for (scale, block) in [(DecodeScale::Quarter, 4), (DecodeScale::Sixteenth, 16)] {
            let mut cases = Vec::new();
            for (name, layout, channels) in [
                ("gray", ChannelLayout::Luma, 1),
                ("rgb", ChannelLayout::Rgb, 3),
                ("cmyk", ChannelLayout::Cmyk, 4),
                ("six components", ChannelLayout::NComponent(6), 6),
            ] {
                cases.push((name.to_owned(), layout, channels, EncodeRequest::lossless()));
            }
            for mode in ALPHA_MODES {
                let request = EncodeRequest::lossless().with_alpha_mode(mode);
                cases.push((format!("bgra {mode:?}"), ChannelLayout::Bgra, 4, request));
            }
            for (name, layout, channels, request) in cases {
                let format = PixelFormat::U8(layout);
                let (pixels, picture) = block_image(width, height, channels, block);
                let source = DecodedSamples::U8(pixels);
                for max_tile in [255, 2] {
                    let request = frequency(request.clone().with_max_tile_macroblocks(max_tile));
                    let file = encode_file((width, height), format, &source, &request);
                    let reduced = decode_at(&file, format, scale);
                    let case = format!("{name} {width}x{height} {scale:?} tiles {max_tile}");
                    let plane = reduced.planes[0];
                    assert_eq!(
                        (plane.width, plane.height),
                        (width.div_ceil(block), height.div_ceil(block)),
                        "{case}"
                    );
                    assert_eq!(tight_rows(&reduced), picture, "{case}");
                }
            }
        }
    }
}

/// The same for chroma the encoder subsamples: constant chroma loses
/// nothing to the decimation, so the luma blocks decide every sample.
#[test]
fn reduced_decodes_reproduce_block_constant_images_with_subsampled_chroma() {
    let (width, height) = (70, 50);
    for (scale, block) in [(DecodeScale::Quarter, 4), (DecodeScale::Sixteenth, 16)] {
        let (gray, gray_picture) = block_image(width, height, 1, block);
        // Red and blue each differ from green by a constant.
        let rgb = |gray: &[u8]| -> Vec<u8> {
            gray.iter()
                .flat_map(|&gray| {
                    let green = 30 + gray / 2;
                    [green + 17, green, green - 9]
                })
                .collect()
        };
        let source = DecodedSamples::U8(rgb(&gray));
        for sampling in SUBSAMPLINGS {
            for max_tile in [255, 2] {
                let request = frequency(
                    EncodeRequest::lossless()
                        .with_chroma_sampling(sampling)
                        .with_max_tile_macroblocks(max_tile),
                );
                let file = encode_file((width, height), RGB8, &source, &request);
                let reduced = decode_at(&file, RGB8, scale);
                let case = format!("{sampling:?} {scale:?} tiles {max_tile}");
                assert_eq!(tight_rows(&reduced), rgb(&gray_picture), "{case}");
            }
        }
    }
}

/// Whatever a frequency-mode file holds beyond the bands a reduced decode
/// needs does not reach the reduced picture: it equals the reduced decode of
/// the same image coded with those bands alone, lossless or lossy, tiled or
/// not, at every overlap mode.
#[test]
fn reduced_decodes_read_only_the_bands_their_resolution_needs() {
    let (width, height) = (70, 50);
    let lossy = |parameter| EncodeRequest::lossy(EncodeQuantizers::uniform(parameter));
    let settings = [
        ("lossless", EncodeRequest::lossless()),
        (
            "lossless overlap 2 tiles",
            EncodeRequest::lossless()
                .with_overlap(OverlapMode::Two)
                .with_max_tile_macroblocks(2),
        ),
        ("qp 20", lossy(20)),
        (
            "qp 40 hard tiles",
            lossy(40).with_max_tile_macroblocks(3).with_hard_tiles(true),
        ),
        ("quality 60", EncodeRequest::quality(60)),
        (
            "lossless trim 2",
            EncodeRequest::lossless().with_trim_flexbits(2),
        ),
    ];
    for (name, format, source, options) in layout_sources(width, height) {
        for (setting, request) in &settings {
            let request = frequency(with_settings(&options, request));
            if request.alpha_mode == AlphaMode::Integrated
                && format.has_alpha()
                && request.trim_flexbits != 0
            {
                continue;
            }
            let case = format!("{name}, {setting}");
            let coded = |bands| {
                let request = request.clone().with_bands(bands).with_trim_flexbits(
                    if bands == BandPresence::All {
                        request.trim_flexbits
                    } else {
                        0
                    },
                );
                encode_file((width, height), format, &source, &request)
            };
            let all = coded(BandPresence::All);
            let no_flexbits = coded(BandPresence::NoFlexbits);
            let no_high_pass = coded(BandPresence::NoHighPass);
            let dc_only = coded(BandPresence::DcOnly);
            assert!(all.len() > no_high_pass.len() && no_high_pass.len() > dc_only.len());

            let quarter = decode_at(&no_high_pass, format, DecodeScale::Quarter);
            let plane = quarter.planes[0];
            assert_eq!(
                (plane.width, plane.height),
                (width.div_ceil(4), height.div_ceil(4)),
                "{case}"
            );
            for file in [&all, &no_flexbits] {
                let reduced = decode_at(file, format, DecodeScale::Quarter);
                assert_eq!(reduced.samples, quarter.samples, "{case}");
                assert_eq!(reduced.planes, quarter.planes, "{case}");
            }
            // The decoder does not offer a sixteenth of subsampled planar
            // YUV output.
            if format == YUV420 {
                continue;
            }
            let sixteenth = decode_at(&dc_only, format, DecodeScale::Sixteenth);
            let plane = sixteenth.planes[0];
            assert_eq!(
                (plane.width, plane.height),
                (width.div_ceil(16), height.div_ceil(16)),
                "{case}"
            );
            for file in [&all, &no_flexbits, &no_high_pass] {
                let reduced = decode_at(file, format, DecodeScale::Sixteenth);
                assert_eq!(reduced.samples, sixteenth.samples, "{case}");
            }
        }
    }
}

/// For each sample along one axis of a reduced picture with `units`
/// samples to the macroblock, whether it is the first or the last of a
/// region the overlap filters treat on its own: a tile of the `extents`
/// given in macroblocks if tile boundaries are `hard`, the whole axis
/// otherwise.
fn region_ends(extents: &[u32], hard: bool, units: u32) -> Vec<bool> {
    let whole = [extents.iter().sum::<u32>()];
    let regions: &[u32] = if hard { extents } else { &whole };
    regions
        .iter()
        .flat_map(|&extent| {
            let samples = extent * units;
            (0..samples).map(move |sample| sample == 0 || sample + 1 == samples)
        })
        .collect()
}

/// A flat image to code and decode reduced: its name, format, pixel, and
/// coding, and whether to try it at every size or at the first alone.
type FlatCase = (&'static str, PixelFormat, Vec<u8>, EncodeRequest, bool);

fn flat_cases() -> Vec<FlatCase> {
    let rgba = PixelFormat::U8(ChannelLayout::Rgba);
    let cmyk = PixelFormat::U8(ChannelLayout::Cmyk);
    let six = PixelFormat::U8(ChannelLayout::NComponent(6));
    let lossless = EncodeRequest::lossless();
    let lossy = EncodeRequest::lossy(EncodeQuantizers::uniform(16));
    let colour = vec![200, 50, 120];
    // At every size: the format whose channel is the coded plane, at
    // levels that round both ways, and each chroma sampling.
    let mut cases = vec![
        ("gray", GRAY8, vec![50], lossless.clone(), true),
        ("gray", GRAY8, vec![128], lossless.clone(), true),
        ("gray", GRAY8, vec![190], lossless.clone(), true),
        ("rgb", RGB8, colour.clone(), lossless.clone(), true),
    ];
    for sampling in SUBSAMPLINGS {
        let request = lossless.clone().with_chroma_sampling(sampling);
        cases.push(("subsampled rgb", RGB8, colour.clone(), request, true));
        let request = lossy.clone().with_chroma_sampling(sampling);
        cases.push(("lossy subsampled rgb", RGB8, colour.clone(), request, false));
    }
    let six_pixel = vec![10, 240, 99, 180, 1, 255];
    cases.extend([
        ("gray", GRAY8, vec![37], lossless.clone(), false),
        ("six components", six, six_pixel, lossless.clone(), false),
        ("rgb", RGB8, vec![31, 157, 222], lossless.clone(), false),
        (
            "neutral rgb",
            RGB8,
            vec![77, 77, 77],
            lossless.clone(),
            false,
        ),
        ("lossy rgb", RGB8, colour.clone(), lossy, false),
        (
            "cmyk",
            cmyk,
            vec![10, 240, 99, 180],
            lossless.clone(),
            false,
        ),
    ]);
    for mode in ALPHA_MODES {
        let request = lossless.clone().with_alpha_mode(mode);
        cases.push(("rgba", rgba, vec![200, 50, 120, 77], request, false));
    }
    cases
}

/// Checks both reduced decodes of `file`, a flat image of `pixel` in
/// `format`, at every sample.
///
/// A plane coded without overlap filtering is exact. With it, a reduced
/// sample is off by what the forward filters rounded away, which differs
/// from level to level and which only the bands a reduced decode leaves
/// unread could return: one level in each coded plane, and two at the
/// corners of the image and of hard tiles, where a filter of its own runs.
/// An inverse colour transform adds up the roundings of its planes.
fn assert_reduced_decodes_are_flat(file: &[u8], format: PixelFormat, pixel: &[u8], case: &str) {
    let view = JxrView::parse(file).unwrap();
    let info = view.info();
    let tiles = &info.tiles;
    let planar = matches!(
        format,
        PixelFormat::U8(ChannelLayout::Luma | ChannelLayout::NComponent(_))
    );
    for (scale, units) in [(DecodeScale::Quarter, 4), (DecodeScale::Sixteenth, 1)] {
        let image = decode_at(file, format, scale);
        let reduced = tight_rows(&image);
        let columns = size(image.planes[0].width);
        assert_eq!(
            (image.planes[0].width, image.planes[0].height),
            (
                info.width.div_ceil(16 / units),
                info.height.div_ceil(16 / units)
            ),
            "{case} {scale:?}"
        );
        let column_ends = region_ends(&tiles.column_widths, tiles.hard_tiles, units);
        let row_ends = region_ends(&tiles.row_heights, tiles.hard_tiles, units);
        for (index, samples) in reduced.chunks_exact(pixel.len()).enumerate() {
            let (x, y) = (index % columns, index / columns);
            let worst = samples.iter().zip(pixel).map(|(a, b)| a.abs_diff(*b)).max();
            let tolerance = if info.primary.overlap == OverlapMode::None {
                0
            } else if planar {
                1 + u8::from(column_ends[x] && row_ends[y])
            } else {
                2
            };
            assert!(
                worst <= Some(tolerance),
                "{case} {scale:?} sample ({x}, {y}): {samples:?} for {pixel:?}"
            );
        }
    }
}

/// A reduced decode does not run the overlap filter of the samples, which
/// changes the level of everything it passes, and at a sixteenth of the
/// size it has only the DC band to run the low-pass-level filter over. Its
/// samples must still be at the level full decoding gives them. A flat
/// image shows it best, because its reduced picture is the same flat image
/// whatever the overlap mode, the tiling, the size, the colour coding, and
/// the alpha mode: at every position, at image and tile edges as inside
/// them, but for rounding.
#[test]
fn reduced_decodes_of_flat_images_keep_their_level_in_every_overlap_mode() {
    // The first is decoded in every case. The others are one and two
    // macroblocks wide, no multiple of sixteen, and wide enough for tiles
    // of one, two, and three macroblocks.
    let sizes = [(70_u32, 50_u32), (16, 20), (29, 17), (100, 36)];
    // Soft and hard tiles down to one macroblock; the tile grid of every
    // size but the third differs across and down.
    let tilings = [
        (255, false),
        (3, false),
        (3, true),
        (2, true),
        (1, false),
        (1, true),
    ];
    for (name, format, pixel, request, every_size) in flat_cases() {
        for (width, height) in &sizes[..if every_size { sizes.len() } else { 1 }] {
            let (width, height) = (*width, *height);
            let source = DecodedSamples::U8(pixel.repeat(size(width) * size(height)));
            for overlap in OVERLAPS {
                for (max_tile, hard_tiles) in tilings {
                    let request = frequency(
                        request
                            .clone()
                            .with_overlap(overlap)
                            .with_max_tile_macroblocks(max_tile)
                            .with_hard_tiles(hard_tiles),
                    );
                    if request.chroma_sampling != ChromaSampling::Cs444
                        && overlap_needs_wider_tiles(width, &request)
                    {
                        continue;
                    }
                    let file = encode_file((width, height), format, &source, &request);
                    let case = format!(
                        "{name} {width}x{height} {overlap:?} tiles {max_tile} {hard_tiles}"
                    );
                    assert_reduced_decodes_are_flat(&file, format, &pixel, &case);
                }
            }
        }
    }
}

/// A gray image of smooth waves three macroblocks long across and four
/// down, so that neighbouring macroblocks differ widely without any edge.
fn wave_image(width: u32, height: u32) -> Vec<u8> {
    // A parabola arch up and one down: a sine wave in integers.
    let wave = |position: u32, period: u32| {
        let half = i64::from(period / 2);
        let phase = i64::from(position % period);
        let (sign, phase) = if phase < half {
            (1, phase)
        } else {
            (-1, phase - half)
        };
        sign * 4 * phase * (half - phase) * 100 / (half * half)
    };
    let mut pixels = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let level = 128 + (wave(x, 48) * 60 + wave(y, 64) * 50) / 100;
            pixels.push(u8::try_from(level).unwrap());
        }
    }
    pixels
}

/// The overlap filters reach across macroblock boundaries, the
/// low-pass-level one of [`OverlapMode::Two`] by half a macroblock. Where
/// neighbouring macroblocks differ, a sixteenth-size decode that only
/// scaled its DC coefficients would exaggerate the difference by a large
/// part of it: by 27 levels in this image. The reduced pictures must stay
/// close to the image averaged over what each of their samples stands
/// for, in every overlap mode and tiling.
#[test]
fn reduced_decodes_stay_close_to_the_image_where_macroblocks_differ() {
    let (width, height) = (144_u32, 128_u32);
    let pixels = wave_image(width, height);
    let source = DecodedSamples::U8(pixels.clone());
    let mean = |x: u32, y: u32, block: u32| {
        let mut sum = 0;
        for row in y * block..(y + 1) * block {
            for column in x * block..(x + 1) * block {
                sum += u32::from(pixels[size(row * width + column)]);
            }
        }
        (sum + block * block / 2) / (block * block)
    };
    for overlap in OVERLAPS {
        for (max_tile, hard_tiles) in TILE_LAYOUTS {
            let request = frequency(
                EncodeRequest::lossless()
                    .with_overlap(overlap)
                    .with_max_tile_macroblocks(max_tile)
                    .with_hard_tiles(hard_tiles),
            );
            let file = encode_file((width, height), GRAY8, &source, &request);
            for (scale, block) in [(DecodeScale::Quarter, 4), (DecodeScale::Sixteenth, 16)] {
                // Without overlap filtering a reduced sample is the mean
                // itself, rounded down. The filter of the samples, which no
                // reduced decode runs, moves part of the difference between
                // neighbouring samples across: more of it between the
                // single blocks of a quarter-size picture than between
                // macroblocks.
                let tolerance = match (overlap, scale) {
                    (OverlapMode::None, _) => 1,
                    (_, DecodeScale::Sixteenth) => 4,
                    _ => 8,
                };
                let reduced = tight_rows(&decode_at(&file, GRAY8, scale));
                let columns = width / block;
                assert_eq!(size(columns * (height / block)), reduced.len());
                for (index, &sample) in reduced.iter().enumerate() {
                    let index = u32::try_from(index).unwrap();
                    let (x, y) = (index % columns, index / columns);
                    let expected = mean(x, y, block);
                    assert!(
                        u32::from(sample).abs_diff(expected) <= tolerance,
                        "{overlap:?} tiles {max_tile} {hard_tiles} {scale:?} ({x}, {y}): \
                         {sample} for {expected}"
                    );
                }
            }
        }
    }
}

/// A reduced picture is the image seen from further away: where the image
/// is smooth, each reduced sample lies within the range of the samples it
/// stands for, lossless or lossy, with or without overlap filtering, at
/// tile edges as inside tiles.
#[test]
fn reduced_decodes_of_smooth_images_stay_within_the_samples_they_stand_for() {
    let (width, height) = (96_u32, 64_u32);
    let smooth = smooth_colour_image(width, height);
    let source = DecodedSamples::U8(smooth.clone());
    let sample = |x: u32, y: u32, channel: usize| {
        i32::from(smooth[(size(y) * size(width) + size(x)) * 3 + channel])
    };
    for (name, request) in [
        ("lossless", EncodeRequest::lossless()),
        (
            "lossless overlap 1",
            EncodeRequest::lossless().with_overlap(OverlapMode::One),
        ),
        (
            "lossless overlap 2 tiles",
            EncodeRequest::lossless()
                .with_overlap(OverlapMode::Two)
                .with_max_tile_macroblocks(2),
        ),
        ("qp 16", EncodeRequest::lossy(EncodeQuantizers::uniform(16))),
        ("quality 85", EncodeRequest::quality(85)),
        (
            "quality 85 4:2:0",
            EncodeRequest::quality(85).with_chroma_sampling(ChromaSampling::Cs420),
        ),
        (
            "quality 85 overlap 2 hard tiles",
            EncodeRequest::quality(85)
                .with_overlap(OverlapMode::Two)
                .with_max_tile_macroblocks(3)
                .with_hard_tiles(true),
        ),
    ] {
        // Rounding alone when every plane is coded exactly; quantization
        // error on top of it otherwise.
        let tolerance = if request.is_lossless() { 3 } else { 6 };
        let file = encode_file((width, height), RGB8, &source, &frequency(request));
        for (scale, block) in [(DecodeScale::Quarter, 4), (DecodeScale::Sixteenth, 16)] {
            let reduced = decode_at(&file, RGB8, scale);
            let pixels = tight_rows(&reduced);
            let columns = width / block;
            for (index, pixel) in pixels.as_chunks::<3>().0.iter().enumerate() {
                let index = u32::try_from(index).unwrap();
                let (x, y) = (index % columns * block, index / columns * block);
                for (channel, &value) in pixel.iter().enumerate() {
                    // The ramps are monotonic, so a block's corners bound it.
                    let corners = [
                        sample(x, y, channel),
                        sample(x + block - 1, y + block - 1, channel),
                    ];
                    let (low, high) = (corners[0].min(corners[1]), corners[0].max(corners[1]));
                    let value = i32::from(value);
                    assert!(
                        (low - tolerance..=high + tolerance).contains(&value),
                        "{name} {scale:?} ({x}, {y}) channel {channel}: {value} outside \
                         {low}..={high}"
                    );
                }
            }
        }
    }
}

#[test]
fn reduced_decodes_need_the_frequency_layout_and_the_bands_of_their_resolution() {
    let (width, height) = (40, 24);
    let pixels = gray_image(width, height, 43);
    let reduced = |codestream: &[u8], scale| {
        let view = JxrView::parse(codestream).unwrap();
        view.decoder()
            .decode(&DecodeRequest::new(GRAY8).with_scale(scale))
            .map(|image| (image.planes[0].width, image.planes[0].height))
            .map_err(|error| error.kind)
    };
    let spatial = encode_gray(&pixels, width, height, &EncodeRequest::lossless());
    for scale in [DecodeScale::Quarter, DecodeScale::Sixteenth] {
        assert_eq!(reduced(&spatial, scale), Err(JxrErrorKind::Unsupported));
    }
    let request = frequency(EncodeRequest::lossless());
    let all = encode_gray(&pixels, width, height, &request);
    assert_eq!(reduced(&all, DecodeScale::Quarter), Ok((10, 6)));
    assert_eq!(reduced(&all, DecodeScale::Sixteenth), Ok((3, 2)));
    // A quarter of the size takes the low-pass band.
    let request = request.with_bands(BandPresence::DcOnly);
    let dc_only = encode_gray(&pixels, width, height, &request);
    assert_eq!(
        reduced(&dc_only, DecodeScale::Quarter),
        Err(JxrErrorKind::Unsupported)
    );
    assert_eq!(reduced(&dc_only, DecodeScale::Sixteenth), Ok((3, 2)));
}
