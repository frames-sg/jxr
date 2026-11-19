use super::*;

pub(super) const SUBSAMPLINGS: [ChromaSampling; 2] = [ChromaSampling::Cs422, ChromaSampling::Cs420];

/// RGB pixels whose chroma is the same everywhere while their luma is
/// noise: red and blue each differ from green by a constant.
pub(super) fn flat_chroma_image(width: u32, height: u32, seed: u64) -> Vec<u8> {
    gray_image(width, height, seed)
        .into_iter()
        .flat_map(|gray| {
            let green = 30 + gray / 2;
            [green + 17, green, green - 9]
        })
        .collect()
}

/// RGB pixels whose channels are ramps, so chroma is as smooth as it gets.
pub(super) fn smooth_colour_image(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::new();
    for y in 0..height {
        for x in 0..width {
            for sample in [20 + x + y / 2, 200 - x / 2 - y, 40 + y + x / 3] {
                pixels.push(u8::try_from(sample).unwrap());
            }
        }
    }
    pixels
}

/// Whether the encoder refuses an image of this width coded with
/// subsampled chroma under `request`: second-level overlap needs the image,
/// and under hard tiling every tile, two macroblocks wide.
pub(super) fn overlap_needs_wider_tiles(width: u32, request: &EncodeRequest) -> bool {
    let macroblocks = width.div_ceil(16);
    let tiles = macroblocks.div_ceil(u32::from(request.max_tile_macroblocks));
    request.overlap == Some(OverlapMode::Two)
        && (macroblocks < 2 || (request.hard_tiles && macroblocks / tiles < 2))
}

#[test]
fn subsampled_chroma_reproduces_flat_chroma_and_stays_close_to_smooth_chroma() {
    let rgb = PixelFormat::U8(ChannelLayout::Rgb);
    for (width, height) in [(1, 1), (37, 21), (96, 70)] {
        let flat = flat_chroma_image(width, height, u64::from(width));
        let smooth = smooth_colour_image(width, height);
        for sampling in SUBSAMPLINGS {
            for overlap in OVERLAPS {
                for (max_tile, hard_tiles) in [(255, false), (3, false), (3, true)] {
                    let request = EncodeRequest::lossless()
                        .with_chroma_sampling(sampling)
                        .with_overlap(overlap)
                        .with_max_tile_macroblocks(max_tile)
                        .with_hard_tiles(hard_tiles);
                    let case = format!(
                        "{width}x{height} {sampling:?} {overlap:?} tiles {max_tile} hard {hard_tiles}"
                    );
                    let image = |pixels| {
                        SourceImage::new(width, height, rgb, SourceSamples::U8(pixels)).unwrap()
                    };
                    if overlap_needs_wider_tiles(width, &request) {
                        assert_eq!(
                            encode(&image(&flat), &request).unwrap_err().kind,
                            JxrErrorKind::InvalidRequest,
                            "{case}"
                        );
                        continue;
                    }
                    // With constant chroma nothing is filtered away, and
                    // lossless quantizers keep every coded plane exact.
                    let file = encode(&image(&flat), &request).unwrap();
                    assert_eq!(decode_as(&file, rgb), flat, "{case}");
                    let info = JxrView::parse(&file).unwrap().info().clone();
                    assert_eq!(
                        info.primary.color_format,
                        ColorFormat::Yuv(sampling),
                        "{case}"
                    );
                    // Lossless quantizers keep unscaled arithmetic.
                    assert!(!info.primary.scaled, "{case}");
                    assert_eq!(info.primary.chroma_centering, [0, 0], "{case}");
                    assert_eq!(
                        declared_format(&file).family,
                        AnnexAPixelFamily::Rgb,
                        "{case}"
                    );
                    // Ramps survive decimation and interpolation but for
                    // rounding and the mirrored image edges.
                    let file = encode(&image(&smooth), &request).unwrap();
                    let worst = largest_error(&smooth, &decode_as(&file, rgb));
                    assert!(worst <= 3, "{case}: worst error {worst}");
                }
            }
        }
    }
}

#[test]
fn every_rgb_storage_can_be_coded_with_subsampled_chroma() {
    let (width, height) = (40_u32, 24_u32);
    let pixels = size(width) * size(height);
    let low16 = |value: u32| u16::try_from(value & 0xffff).unwrap();
    let noise = || values(pixels, 61).into_iter();
    // Gray pixels of unrelated brightness in every colour storage: with no
    // chroma to filter away, lossless quantizers reproduce them exactly.
    let cases = [
        (
            PixelFormat::U16(ChannelLayout::Rgb),
            DecodedSamples::U16(noise().flat_map(|value| [low16(value); 3]).collect()),
        ),
        (
            PixelFormat::I16(ChannelLayout::Rgb),
            DecodedSamples::I16(
                noise()
                    .flat_map(|value| [low16(value).cast_signed() >> 1; 3])
                    .collect(),
            ),
        ),
        (
            PixelFormat::F16(ChannelLayout::Rgb),
            DecodedSamples::F16(
                noise()
                    .flat_map(|value| [low16(value) & 0x7fff; 3])
                    .collect(),
            ),
        ),
        (
            PixelFormat::Rgb555,
            DecodedSamples::Rgb555(
                noise()
                    .map(|value| low16(value) & 0x1f)
                    .map(|gray| gray << 10 | gray << 5 | gray)
                    .collect(),
            ),
        ),
        (
            // Green has one more bit than red and blue.
            PixelFormat::Rgb565,
            DecodedSamples::Rgb565(
                noise()
                    .map(|value| low16(value) & 0x1f)
                    .map(|gray| gray << 11 | gray << 6 | gray)
                    .collect(),
            ),
        ),
        (
            PixelFormat::Rgb101010,
            DecodedSamples::Rgb101010(
                noise()
                    .map(|value| value & 0x3ff)
                    .map(|gray| gray << 20 | gray << 10 | gray)
                    .collect(),
            ),
        ),
        (
            PixelFormat::Rgbe,
            DecodedSamples::Rgbe(
                noise()
                    .map(|value| {
                        let [mantissa, _, _, exponent] = value.to_le_bytes();
                        let mantissa = mantissa | 0x80;
                        if exponent == 0 {
                            0
                        } else {
                            u32::from_le_bytes([mantissa, mantissa, mantissa, exponent])
                        }
                    })
                    .collect(),
            ),
        ),
        (
            // Alpha is a plane of its own, untouched by the chroma sampling.
            PixelFormat::U8(ChannelLayout::Bgra),
            DecodedSamples::U8(
                noise()
                    .flat_map(|value| {
                        let [gray, alpha, ..] = value.to_le_bytes();
                        [gray, gray, gray, alpha]
                    })
                    .collect(),
            ),
        ),
    ];
    for (format, source) in cases {
        for sampling in SUBSAMPLINGS {
            for mode in ALPHA_MODES {
                let request = EncodeRequest::lossless()
                    .with_chroma_sampling(sampling)
                    .with_alpha_mode(mode);
                let (file, decoded) = recode(width, height, format, &source, &request);
                assert_eq!(decoded, source, "{format:?} {sampling:?} {mode:?}");
                let info = JxrView::parse(&file).unwrap().info().clone();
                assert_eq!(
                    info.primary.color_format,
                    ColorFormat::Yuv(sampling),
                    "{format:?}"
                );
                assert!(!info.primary.scaled, "{format:?}");
            }
        }
    }
}

#[test]
fn subsampled_chroma_shrinks_a_lossy_colour_file_and_keeps_luma_detail() {
    let (width, height) = (160_u32, 112_u32);
    let rgb = PixelFormat::U8(ChannelLayout::Rgb);
    let pixels = colour_image(width, height, 3, 211);
    let image = SourceImage::new(width, height, rgb, SourceSamples::U8(&pixels)).unwrap();
    let request = EncodeRequest::lossy(EncodeQuantizers::uniform(40));
    let full = encode(&image, &request).unwrap();
    let mut previous = full.len();
    // 4:2:2 codes half the chroma samples and 4:2:0 a quarter.
    for sampling in SUBSAMPLINGS {
        let file = encode(&image, &request.clone().with_chroma_sampling(sampling)).unwrap();
        assert!(file.len() < previous, "{sampling:?}: {} bytes", file.len());
        previous = file.len();
        let decoded = decode_as(&file, rgb);
        // Green is mostly luma, which is coded as before.
        let green = |pixels: &[u8]| {
            pixels
                .iter()
                .skip(1)
                .step_by(3)
                .copied()
                .collect::<Vec<_>>()
        };
        let reference = decode_as(&full, rgb);
        let mean = |a: &[u8], b: &[u8]| {
            let total: u32 = a
                .iter()
                .zip(b)
                .map(|(a, b)| u32::from(a.abs_diff(*b)))
                .sum();
            f64::from(total) / f64::from(u32::try_from(a.len()).unwrap())
        };
        let full_error = mean(&green(&pixels), &green(&reference));
        let error = mean(&green(&pixels), &green(&decoded));
        assert!(
            error < full_error + 12.0,
            "{sampling:?}: green error {error}"
        );
    }
}

#[test]
fn sources_without_full_resolution_chroma_ignore_the_chroma_sampling() {
    let (width, height) = (40_u32, 24_u32);
    let gray = DecodedSamples::U8(gray_image(width, height, 5));
    let planar = yuv_samples(width, height, YUV420, 8, 6);
    let cases = [
        (GRAY8, &gray),
        (
            PixelFormat::U8(ChannelLayout::Cmyk),
            &multichannel_samples(width, height, 4, false, 7),
        ),
        (
            PixelFormat::U8(ChannelLayout::NComponent(3)),
            &multichannel_samples(width, height, 3, false, 8),
        ),
        (YUV420, &planar),
    ];
    for (format, source) in cases {
        let image = SourceImage::new(width, height, format, SourceSamples::from(source)).unwrap();
        let plain = encode(&image, &EncodeRequest::lossless()).unwrap();
        for sampling in [
            ChromaSampling::Cs422,
            ChromaSampling::Cs420,
            ChromaSampling::Cs444,
        ] {
            let request = EncodeRequest::lossless().with_chroma_sampling(sampling);
            assert_eq!(
                encode(&image, &request).unwrap(),
                plain,
                "{format:?} {sampling:?}"
            );
        }
    }
}

const YUV444: PixelFormat = PixelFormat::U8(ChannelLayout::Yuv(ChromaSampling::Cs444));
pub(super) const YUV422: PixelFormat = PixelFormat::U8(ChannelLayout::Yuv(ChromaSampling::Cs422));
pub(super) const YUV420: PixelFormat = PixelFormat::U8(ChannelLayout::Yuv(ChromaSampling::Cs420));

/// The format of `layout` in the storage samples of `bits` bits take: eight
/// in bytes, ten and sixteen in 16-bit elements.
fn yuv_format(layout: ChannelLayout, bits: u8) -> PixelFormat {
    if bits == 8 {
        PixelFormat::U8(layout)
    } else {
        PixelFormat::U16(layout)
    }
}

/// Unrelated noise in every sample a YUV image of `format` stores, planar
/// or interleaved, within `bits` bits.
pub(super) fn yuv_samples(
    width: u32,
    height: u32,
    format: PixelFormat,
    bits: u8,
    seed: u64,
) -> DecodedSamples {
    let pixels = size(width) * size(height);
    let (PixelFormat::U8(layout) | PixelFormat::U16(layout) | PixelFormat::I16(layout)) = format
    else {
        panic!("not a YUV storage: {format:?}");
    };
    let chroma =
        match layout {
            ChannelLayout::Yuv(ChromaSampling::Cs420)
            | ChannelLayout::Yuva(ChromaSampling::Cs420) => pixels / 4,
            ChannelLayout::Yuv(ChromaSampling::Cs422)
            | ChannelLayout::Yuva(ChromaSampling::Cs422) => pixels / 2,
            _ => pixels,
        };
    let count = pixels + 2 * chroma + if layout.has_alpha() { pixels } else { 0 };
    let noise = values(count, seed).into_iter();
    let mask = (1_u32 << bits) - 1;
    match format {
        PixelFormat::U8(_) => DecodedSamples::U8(
            noise
                .map(|value| u8::try_from(value & mask).unwrap())
                .collect(),
        ),
        PixelFormat::I16(_) => DecodedSamples::I16(
            noise
                .map(|value| u16::try_from(value & mask).unwrap().cast_signed())
                .collect(),
        ),
        _ => DecodedSamples::U16(
            noise
                .map(|value| u16::try_from(value & mask).unwrap())
                .collect(),
        ),
    }
}

/// Encode a YUV image of `bits`-bit samples with `request` and decode it
/// back in the same pixel format.
pub(super) fn recode_yuv(
    (width, height): (u32, u32),
    format: PixelFormat,
    bits: u8,
    source: &DecodedSamples,
    request: &EncodeRequest,
) -> (Vec<u8>, jxr::DecodedImage) {
    let image = SourceImage::new(width, height, format, SourceSamples::from(source)).unwrap();
    let image = if bits == 10 {
        image.with_sample_bits(10).unwrap()
    } else {
        image
    };
    let file = encode(&image, request).unwrap();
    let view = JxrView::parse(&file).unwrap();
    assert_eq!(view.info().dimensions(), (width, height));
    let decoded = view.decoder().decode(&DecodeRequest::new(format)).unwrap();
    (file, decoded)
}

/// Every YUV format the container table declares, as `(layout, bits)`, with
/// the depth the table gives it; sixteen-bit fixed point is listed apart.
fn yuv_formats() -> Vec<(ChannelLayout, u8)> {
    let mut formats = Vec::new();
    for alpha in [false, true] {
        let layout = |sampling| {
            if alpha {
                ChannelLayout::Yuva(sampling)
            } else {
                ChannelLayout::Yuv(sampling)
            }
        };
        formats.push((layout(ChromaSampling::Cs420), 8));
        for bits in [8, 10, 16] {
            formats.push((layout(ChromaSampling::Cs422), bits));
            formats.push((layout(ChromaSampling::Cs444), bits));
        }
    }
    formats
}

/// Every YUV format at its own sampling, alone and with alpha in both alpha
/// modes. Subsampled chroma beside an integrated alpha plane is the layout
/// the frequency-mode reader once reconstructed with unpredicted chroma.
#[test]
fn yuv_formats_round_trip_losslessly_in_both_packet_layouts() {
    for (width, height) in [(2, 2), (38, 22), (96, 70)] {
        for (layout, bits) in yuv_formats() {
            let format = yuv_format(layout, bits);
            let (ChannelLayout::Yuv(sampling) | ChannelLayout::Yuva(sampling)) = layout else {
                unreachable!()
            };
            let modes: &[AlphaMode] = if layout.has_alpha() {
                &ALPHA_MODES
            } else {
                &[AlphaMode::Separate]
            };
            for &mode in modes {
                for packet_mode in [BitstreamMode::Spatial, BitstreamMode::Frequency] {
                    let seed = match packet_mode {
                        BitstreamMode::Spatial => 300,
                        BitstreamMode::Frequency => 900,
                    };
                    let source = yuv_samples(width, height, format, bits, u64::from(width) + seed);
                    for overlap in OVERLAPS {
                        for (max_tile, hard_tiles) in [(255, false), (3, true)] {
                            let request = EncodeRequest::lossless()
                                .with_overlap(overlap)
                                .with_max_tile_macroblocks(max_tile)
                                .with_hard_tiles(hard_tiles)
                                .with_alpha_mode(mode)
                                .with_bitstream_mode(packet_mode);
                            let case = format!(
                                "{format:?} {bits} bits {width}x{height} {mode:?} {packet_mode:?} \
                             {overlap:?} tiles {max_tile}"
                            );
                            let subsampled = sampling != ChromaSampling::Cs444;
                            if subsampled && overlap_needs_wider_tiles(width, &request) {
                                continue;
                            }
                            let (file, decoded) =
                                recode_yuv((width, height), format, bits, &source, &request);
                            assert_eq!(decoded.samples, source, "{case}");
                            // Planes are returned as they were given: tight and
                            // one after another.
                            let planes = if subsampled {
                                usize::from(layout.channel_count())
                            } else {
                                1
                            };
                            assert_eq!(decoded.planes.len(), planes, "{case}");
                            let info = &decoded.info;
                            assert_eq!(
                                info.primary.color_format,
                                ColorFormat::Yuv(sampling),
                                "{case}"
                            );
                            assert!(!info.primary.scaled, "{case}");
                            assert_eq!(
                                info.primary.sample_format,
                                SampleFormat::Unsigned { bits },
                                "{case}"
                            );
                            assert_eq!(info.profile, Some(Profile::Advanced), "{case}");
                            if layout.has_alpha() {
                                assert_eq!(info.alpha_mode, mode, "{case}");
                            }
                            let declared = declared_format(&file);
                            assert_eq!(declared.family, AnnexAPixelFamily::Yuv(sampling), "{case}");
                            assert_eq!(declared.alpha, layout.has_alpha(), "{case}");
                            assert_eq!(
                                declared.bit_depth == AnnexABitDepth::U10,
                                bits == 10,
                                "{case}"
                            );
                            assert_eq!(declared_layouts(&file).0, packet_mode, "{case}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn fixed_point_yuv_round_trips_losslessly() {
    let (width, height) = (38_u32, 22_u32);
    for layout in [
        ChannelLayout::Yuv(ChromaSampling::Cs444),
        ChannelLayout::Yuva(ChromaSampling::Cs444),
    ] {
        let format = PixelFormat::I16(layout);
        let source = yuv_samples(width, height, format, 16, 340);
        for mode in ALPHA_MODES {
            let request = EncodeRequest::lossless().with_alpha_mode(mode);
            let (file, decoded) = recode_yuv((width, height), format, 16, &source, &request);
            assert_eq!(decoded.samples, source, "{layout:?} {mode:?}");
            assert_eq!(
                declared_format(&file).bit_depth,
                AnnexABitDepth::I16,
                "{layout:?}"
            );
        }
    }
}

#[test]
fn lossy_yuv_stays_close_and_shrinks_at_every_sampling() {
    let (width, height) = (96_u32, 64_u32);
    for (layout, bits) in yuv_formats() {
        let format = yuv_format(layout, bits);
        // A smooth picture in every plane the format stores, stretched to
        // the sample depth: quantization error stays near the step size.
        let count = widened(&yuv_samples(width, height, format, bits, 1)).len();
        let smooth = gray_image(width, u32::try_from(count).unwrap() / width, 71);
        let range = (1_u32 << bits) - 1;
        let source = match bits {
            8 => DecodedSamples::U8(smooth),
            10 => DecodedSamples::U16(
                smooth
                    .into_iter()
                    .map(|value| u16::from(value) * 4)
                    .collect(),
            ),
            _ => DecodedSamples::U16(
                smooth
                    .into_iter()
                    .map(|value| u16::from(value) * 257)
                    .collect(),
            ),
        };
        let lossless = recode_yuv(
            (width, height),
            format,
            bits,
            &source,
            &EncodeRequest::lossless(),
        );
        assert_eq!(lossless.1.samples, source, "{format:?} {bits} bits");
        // Deeper samples carry more bits under the same step.
        let parameter = match bits {
            8 => 40,
            10 => 72,
            _ => 112,
        };
        let request = EncodeRequest::lossy(EncodeQuantizers::uniform(parameter));
        let (file, decoded) = recode_yuv((width, height), format, bits, &source, &request);
        assert!(file.len() < lossless.0.len(), "{format:?} {bits} bits");
        assert_ne!(decoded.samples, source, "{format:?} {bits} bits");
        let (a, b) = (widened(&source), widened(&decoded.samples));
        let total: f64 = a
            .iter()
            .zip(&b)
            .map(|(&a, &b)| f64::from(a.abs_diff(b)))
            .sum();
        let error = total / f64::from(range) / f64::from(u32::try_from(a.len()).unwrap());
        assert!(error < 0.05, "{format:?} {bits} bits: error {error}");
        // Ten-bit samples stay within their range.
        assert!(
            b.iter().all(|&sample| sample <= range),
            "{format:?} {bits} bits"
        );
    }
}

#[test]
fn full_resolution_yuv_can_be_coded_with_subsampled_chroma() {
    let (width, height) = (96_u32, 70_u32);
    for sampling in SUBSAMPLINGS {
        // Noise luma with constant chroma, which decimation keeps exactly.
        let flat: Vec<u8> = gray_image(width, height, 12)
            .into_iter()
            .flat_map(|luma| [luma, 90, 200])
            .collect();
        let source = DecodedSamples::U8(flat);
        for overlap in OVERLAPS {
            let request = EncodeRequest::lossless()
                .with_chroma_sampling(sampling)
                .with_overlap(overlap);
            let (file, decoded) = recode_yuv((width, height), YUV444, 8, &source, &request);
            assert_eq!(decoded.samples, source, "{sampling:?} {overlap:?}");
            // The image is still YUV 4:4:4 to its reader: one interleaved
            // plane, declared as such, with coarser chroma inside.
            assert_eq!(decoded.planes.len(), 1);
            assert_eq!(
                declared_format(&file).family,
                AnnexAPixelFamily::Yuv(ChromaSampling::Cs444)
            );
            let info = &decoded.info;
            assert_eq!(info.primary.color_format, ColorFormat::Yuv(sampling));
            assert!(!info.primary.scaled);
        }
    }
}

#[test]
fn planes_may_be_separated_and_their_rows_padded() {
    let (width, height) = (38_usize, 22_usize);
    for (format, chroma_height) in [(YUV420, height / 2), (YUV422, height)] {
        let tight = yuv_samples(38, 22, format, 8, 55);
        let DecodedSamples::U8(samples) = &tight else {
            unreachable!()
        };
        let chroma_width = width / 2;
        let luma = width * height;
        let chroma = chroma_width * chroma_height;
        // V first, then U with padded rows, then Y with padded rows.
        let (u_stride, y_stride) = (chroma_width + 5, width + 3);
        let u_offset = chroma + 7;
        let y_offset = u_offset + u_stride * chroma_height + 11;
        let mut scattered = vec![0xee_u8; y_offset + y_stride * height];
        scattered[..chroma].copy_from_slice(&samples[luma + chroma..]);
        for row in 0..chroma_height {
            let source = luma + row * chroma_width;
            scattered[u_offset + row * u_stride..][..chroma_width]
                .copy_from_slice(&samples[source..source + chroma_width]);
        }
        for row in 0..height {
            scattered[y_offset + row * y_stride..][..width]
                .copy_from_slice(&samples[row * width..(row + 1) * width]);
        }
        let planes = [
            SourcePlane {
                offset: y_offset,
                row_stride: y_stride,
            },
            SourcePlane {
                offset: u_offset,
                row_stride: u_stride,
            },
            SourcePlane {
                offset: 0,
                row_stride: chroma_width,
            },
        ];
        let image = SourceImage::new(38, 22, format, SourceSamples::U8(&scattered))
            .unwrap()
            .with_planes(&planes)
            .unwrap();
        let reference = SourceImage::new(38, 22, format, SourceSamples::U8(samples)).unwrap();
        let request = EncodeRequest::lossless();
        assert_eq!(
            encode(&image, &request).unwrap(),
            encode(&reference, &request).unwrap(),
            "{format:?}"
        );
        // A plane that runs past the samples is rejected, as is a plane
        // list of the wrong length or one given for an interleaved image.
        let mut beyond = planes;
        beyond[0].offset += 4;
        assert!(matches!(
            image.with_planes(&beyond).unwrap_err().kind,
            JxrErrorKind::BufferTooSmall { .. }
        ));
        assert_eq!(
            image.with_planes(&planes[..2]).unwrap_err().kind,
            JxrErrorKind::InvalidRequest
        );
        assert_eq!(
            image.with_row_stride(64).unwrap_err().kind,
            JxrErrorKind::InvalidRequest
        );
    }
    let interleaved = [0_u8; 38 * 22 * 3];
    let image = SourceImage::new(38, 22, YUV444, SourceSamples::U8(&interleaved)).unwrap();
    assert_eq!(
        image
            .with_planes(&[SourcePlane::default(); 3])
            .unwrap_err()
            .kind,
        JxrErrorKind::InvalidRequest
    );
}

#[test]
fn subsampled_yuv_sources_need_even_dimensions() {
    let samples = [0_u8; 64];
    for (format, width, height, valid) in [
        (YUV420, 4, 4, true),
        (YUV420, 5, 4, false),
        (YUV420, 4, 3, false),
        (YUV422, 4, 3, true),
        (YUV422, 3, 4, false),
        (YUV444, 3, 3, true),
    ] {
        let image = SourceImage::new(width, height, format, SourceSamples::U8(&samples));
        assert_eq!(image.is_ok(), valid, "{format:?} {width}x{height}");
        if let Err(error) = image {
            assert_eq!(error.kind, JxrErrorKind::InvalidRequest);
        }
    }
}

#[test]
fn ten_bit_samples_are_declared_and_must_fit_their_depth() {
    let (width, height) = (38_u32, 22_u32);
    let format = PixelFormat::U16(ChannelLayout::Yuv(ChromaSampling::Cs422));
    let source = yuv_samples(width, height, format, 10, 91);
    let DecodedSamples::U16(samples) = &source else {
        unreachable!()
    };
    let image = SourceImage::new(width, height, format, SourceSamples::U16(samples)).unwrap();
    // The same storage is a sixteen-bit image unless it says otherwise.
    let wide = encode(&image, &EncodeRequest::lossless()).unwrap();
    assert_eq!(declared_format(&wide).bit_depth, AnnexABitDepth::U16);
    let narrow = encode(
        &image.with_sample_bits(10).unwrap(),
        &EncodeRequest::lossless(),
    )
    .unwrap();
    assert_eq!(declared_format(&narrow).bit_depth, AnnexABitDepth::U10);
    // One sample beyond ten bits is refused, not clipped.
    let mut beyond = samples.clone();
    beyond[size(width) * size(height) + 3] = 1024;
    let image = SourceImage::new(width, height, format, SourceSamples::U16(&beyond))
        .unwrap()
        .with_sample_bits(10)
        .unwrap();
    assert_eq!(
        encode(&image, &EncodeRequest::lossless()).unwrap_err().kind,
        JxrErrorKind::InvalidRequest
    );
    // No container format declares ten-bit RGB in 16-bit elements, or
    // 4:2:0 at ten bits, or any other narrow depth.
    let rgb = PixelFormat::U16(ChannelLayout::Rgb);
    let rgb_samples = vec![0_u16; size(width) * size(height) * 3];
    for (format, samples, bits) in [
        (rgb, &rgb_samples, 10),
        (
            PixelFormat::U16(ChannelLayout::Yuv(ChromaSampling::Cs420)),
            &rgb_samples,
            10,
        ),
        (format, &rgb_samples, 12),
    ] {
        let image = SourceImage::new(width, height, format, SourceSamples::U16(samples))
            .unwrap()
            .with_sample_bits(bits)
            .unwrap();
        assert_eq!(
            encode(&image, &EncodeRequest::lossless()).unwrap_err().kind,
            JxrErrorKind::Unsupported,
            "{format:?} at {bits} bits"
        );
    }
    // Only 16-bit unsigned storage holds narrower samples.
    let bytes = [0_u8; 64];
    let image = SourceImage::new(4, 4, GRAY8, SourceSamples::U8(&bytes)).unwrap();
    assert_eq!(
        image.with_sample_bits(8).unwrap_err().kind,
        JxrErrorKind::InvalidRequest
    );
}

#[test]
fn chroma_centering_of_a_subsampled_source_is_declared_as_given() {
    let (width, height) = (38_u32, 22_u32);
    for (format, centering, valid) in [
        (YUV420, [0, 0], true),
        (YUV420, [2, 3], true),
        (YUV420, [4, 7], true),
        (YUV420, [5, 0], false),
        (YUV422, [3, 0], true),
        // 4:2:2 has no vertical chroma offset to declare.
        (YUV422, [3, 1], false),
        (YUV444, [1, 0], false),
    ] {
        let source = yuv_samples(width, height, format, 8, 47);
        let image = SourceImage::new(width, height, format, SourceSamples::from(&source))
            .unwrap()
            .with_chroma_centering(centering[0], centering[1]);
        assert_eq!(image.is_ok(), valid, "{format:?} {centering:?}");
        let Ok(image) = image else {
            assert_eq!(image.unwrap_err().kind, JxrErrorKind::InvalidRequest);
            continue;
        };
        let file = encode(&image, &EncodeRequest::lossless()).unwrap();
        let view = JxrView::parse(&file).unwrap();
        assert_eq!(
            view.info().primary.chroma_centering,
            centering,
            "{format:?}"
        );
        // The codes describe the samples and do not change them.
        let decoded = view.decoder().decode(&DecodeRequest::new(format)).unwrap();
        assert_eq!(decoded.samples, source, "{format:?} {centering:?}");
    }
}

// Frequency-mode codestreams: one packet for each band of a tile instead of
// one packet for the tile.
