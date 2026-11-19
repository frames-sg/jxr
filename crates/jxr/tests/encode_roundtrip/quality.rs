use super::*;

/// Qualities from lossless to the floor, close enough to follow the scale
/// and far enough apart that every step coarsens all the quantizers: a single
/// parameter step can move a small synthetic image by a byte either way.
const QUALITY_LADDER: [u8; 22] = [
    100, 99, 97, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 0,
];

/// Check that each step down [`QUALITY_LADDER`] shrinks the file and adds
/// error, from an exact image to a small fraction of the lossless size.
fn assert_quality_ladder_is_monotonic(format: PixelFormat, pixels: &[u8], width: u32, height: u32) {
    let image = SourceImage::new(width, height, format, SourceSamples::U8(pixels)).unwrap();
    let mut previous: Option<(u8, usize, u64)> = None;
    let mut lossless_size = 0;
    for quality in QUALITY_LADDER {
        let file = encode(&image, &EncodeRequest::quality(quality)).unwrap();
        let squared_error: u64 = decode_as(&file, format)
            .iter()
            .zip(pixels)
            .map(|(&a, &b)| u64::from(a.abs_diff(b)).pow(2))
            .sum();
        match previous {
            None => {
                assert_eq!(squared_error, 0, "{format:?}: quality 100 is exact");
                lossless_size = file.len();
            }
            Some((above, size_above, error_above)) => {
                assert!(
                    file.len() < size_above,
                    "{format:?} quality {quality}: {} bytes after {size_above} at {above}",
                    file.len()
                );
                assert!(
                    squared_error > error_above,
                    "{format:?} quality {quality}: error {squared_error} after {error_above} at {above}"
                );
            }
        }
        previous = Some((quality, file.len(), squared_error));
    }
    let (_, smallest, _) = previous.unwrap();
    assert!(
        smallest * 8 < lossless_size,
        "{format:?}: quality 0 is {smallest} bytes of {lossless_size}"
    );
}

#[test]
fn quality_scale_shrinks_files_and_grows_error_down_the_ladder() {
    let (width, height) = (160, 112);
    assert_quality_ladder_is_monotonic(GRAY8, &gray_image(width, height, 61), width, height);
    assert_quality_ladder_is_monotonic(
        PixelFormat::U8(ChannelLayout::Rgb),
        &colour_image(width, height, 3, 67),
        width,
        height,
    );
    // Colour formats the scale gives other steps: inks through the colour
    // transform, and channels coded as they are stored.
    for layout in [ChannelLayout::Cmyk, ChannelLayout::NComponent(4)] {
        assert_quality_ladder_is_monotonic(
            PixelFormat::U8(layout),
            &colour_image(width, height, 4, 67),
            width,
            height,
        );
    }
}

#[test]
fn neighbouring_qualities_code_a_colour_image_differently() {
    // Luma and chroma parameters move on alternate qualities where the scale
    // is flattest, so no two neighbours share an output.
    let (width, height) = (96, 64);
    let format = PixelFormat::U8(ChannelLayout::Rgb);
    let pixels = colour_image(width, height, 3, 67);
    let image = SourceImage::new(width, height, format, SourceSamples::U8(&pixels)).unwrap();
    let files: Vec<Vec<u8>> = (0..=100)
        .map(|quality| encode(&image, &EncodeRequest::quality(quality)).unwrap())
        .collect();
    for (quality, pair) in files.windows(2).enumerate() {
        assert_ne!(pair[0], pair[1], "qualities {quality} and {}", quality + 1);
    }
}

#[test]
fn quality_above_100_is_the_lossless_request() {
    let (width, height) = (40, 24);
    let pixels = gray_image(width, height, 71);
    let lossless = encode_gray(&pixels, width, height, &EncodeRequest::lossless());
    for quality in [100, 101, u8::MAX] {
        let request = EncodeRequest::quality(quality);
        assert_eq!(encode_gray(&pixels, width, height, &request), lossless);
    }
    assert_eq!(decode_gray(&lossless).2, pixels);
}

/// Mean absolute error of a decoded image as a fraction of `peak`.
pub(super) fn error_against(source: &DecodedSamples, decoded: &DecodedSamples, peak: f64) -> f64 {
    let values = |samples: &DecodedSamples| -> Vec<f64> {
        match samples {
            DecodedSamples::U8(samples) => samples.iter().map(|&s| f64::from(s)).collect(),
            DecodedSamples::U16(samples) => samples.iter().map(|&s| f64::from(s)).collect(),
            DecodedSamples::I16(samples) => samples.iter().map(|&s| f64::from(s)).collect(),
            other => panic!("unexpected storage {other:?}"),
        }
    };
    let (source, decoded) = (values(source), values(decoded));
    assert_eq!(source.len(), decoded.len());
    let total: f64 = source
        .iter()
        .zip(&decoded)
        .map(|(a, b)| (a - b).abs())
        .sum();
    total / peak / f64::from(u32::try_from(source.len()).unwrap())
}

#[test]
fn a_quality_means_the_same_at_every_sample_depth() {
    // One picture in 8 bits, in 16 bits, and in s2.13 fixed point, where
    // 255 becomes 65535 and 8160 of the 8192 that stand for one.
    let (width, height) = (160, 112);
    let pixels = colour_image(width, height, 3, 67);
    let eight = DecodedSamples::U8(pixels.clone());
    let deeper = [
        (
            PixelFormat::U16(ChannelLayout::Rgb),
            DecodedSamples::U16(pixels.iter().map(|&s| u16::from(s) * 257).collect()),
            65_535.0,
        ),
        (
            PixelFormat::I16(ChannelLayout::Rgb),
            DecodedSamples::I16(pixels.iter().map(|&s| i16::from(s) * 32).collect()),
            8192.0,
        ),
    ];
    let ratio = |a: usize, b: usize| {
        f64::from(u32::try_from(a).unwrap()) / f64::from(u32::try_from(b).unwrap())
    };
    for quality in [95, 75, 40, 10] {
        let request = EncodeRequest::quality(quality);
        let (reference, decoded) = recode(width, height, RGB8, &eight, &request);
        let reference_error = error_against(&eight, &decoded, 255.0);
        for (format, source, peak) in &deeper {
            let (file, decoded) = recode(width, height, *format, source, &request);
            // The same share of the range is lost, in a file of the same
            // size: the deeper samples add no picture, only range.
            let error = error_against(source, &decoded, *peak);
            assert!(
                (0.9..1.1).contains(&(error / reference_error)),
                "{format:?} quality {quality}: error {error} against {reference_error}"
            );
            assert!(
                (0.95..1.05).contains(&ratio(file.len(), reference.len())),
                "{format:?} quality {quality}: {} bytes against {}",
                file.len(),
                reference.len()
            );
        }
    }
}

#[test]
fn a_source_with_too_few_sample_values_is_coded_at_the_nearest_step_it_has() {
    let (width, height) = (96_u32, 64_u32);
    let gray = gray_image(width, height, 9);
    // Black and white has no step but the exact one: every quality is the
    // lossless file.
    let bits: Vec<u8> = gray
        .as_chunks::<8>()
        .0
        .iter()
        .map(|run| {
            run.iter()
                .fold(0, |byte, &sample| byte << 1 | u8::from(sample > 127))
        })
        .collect();
    let format = PixelFormat::BitPacked(ChannelLayout::Luma);
    let image = SourceImage::new(width, height, format, SourceSamples::BitPacked(&bits)).unwrap();
    let lossless = encode(&image, &EncodeRequest::lossless()).unwrap();
    for quality in [99, 75, 40, 6, 0] {
        let file = encode(&image, &EncodeRequest::quality(quality)).unwrap();
        assert_eq!(file, lossless, "quality {quality}");
    }

    // Five-bit samples have none near the top of the scale, and from there
    // down files that shrink or stay as they are.
    let words: Vec<u16> = colour_image(width, height, 3, 9)
        .as_chunks::<3>()
        .0
        .iter()
        .map(|pixel| {
            let [red, green, blue] = pixel.map(|sample| u16::from(sample >> 3));
            red << 10 | green << 5 | blue
        })
        .collect();
    let image = SourceImage::new(
        width,
        height,
        PixelFormat::Rgb555,
        SourceSamples::Rgb555(&words),
    )
    .unwrap();
    let lossless = encode(&image, &EncodeRequest::lossless()).unwrap();
    assert_eq!(
        encode(&image, &EncodeRequest::quality(96)).unwrap(),
        lossless
    );
    let mut above = lossless.len();
    let mut distinct = 0;
    for quality in (0..96).rev() {
        let size = encode(&image, &EncodeRequest::quality(quality))
            .unwrap()
            .len();
        assert!(
            size <= above,
            "quality {quality}: {size} bytes after {above}"
        );
        distinct += usize::from(size < above);
        above = size;
    }
    assert!(above * 8 < lossless.len(), "quality 0 is {above} bytes");
    assert!((10..60).contains(&distinct), "{distinct} distinct sizes");
}

#[test]
fn an_image_with_one_parameter_repeats_where_only_differences_would_move() {
    // From 75 down to 10 the scale moves by half a parameter: chroma
    // differences on odd qualities, everything else on even ones. A gray
    // image, or channels coded as they are stored, have nothing to move on
    // the odd ones, and a colour image moves on both.
    let (width, height) = (96, 64);
    let gray = DecodedSamples::U8(gray_image(width, height, 5));
    let channels = DecodedSamples::U8(colour_image(width, height, 4, 5));
    let colour = DecodedSamples::U8(colour_image(width, height, 3, 5));
    let files = |format, source: &DecodedSamples, quality| {
        recode(
            width,
            height,
            format,
            source,
            &EncodeRequest::quality(quality),
        )
        .0
    };
    let separate = PixelFormat::U8(ChannelLayout::NComponent(4));
    for quality in [72, 40, 12] {
        for (format, source) in [(GRAY8, &gray), (separate, &channels)] {
            let even = files(format, source, quality);
            assert_eq!(
                files(format, source, quality - 1),
                even,
                "{format:?} {quality}"
            );
            assert_ne!(
                files(format, source, quality + 1),
                even,
                "{format:?} {quality}"
            );
        }
        let even = files(RGB8, &colour, quality);
        assert_ne!(files(RGB8, &colour, quality - 1), even, "quality {quality}");
        assert_ne!(files(RGB8, &colour, quality + 1), even, "quality {quality}");
    }
}

#[test]
fn the_quality_scale_picks_an_overlap_mode_the_image_can_take() {
    let overlap = |width: u32, request: &EncodeRequest| {
        let pixels = colour_image(width, 24, 3, 5);
        let image = SourceImage::new(width, 24, RGB8, SourceSamples::U8(&pixels)).unwrap();
        encode(&image, request).map(|file| JxrView::parse(&file).unwrap().info().primary.overlap)
    };
    // None at the top of the scale, second level at the bottom.
    for (quality, mode) in [
        (100, OverlapMode::None),
        (99, OverlapMode::None),
        (98, OverlapMode::One),
        (7, OverlapMode::One),
        (6, OverlapMode::Two),
        (0, OverlapMode::Two),
    ] {
        let request = EncodeRequest::quality(quality);
        assert_eq!(overlap(48, &request).unwrap(), mode, "quality {quality}");
    }
    // Subsampled chroma takes second-level overlap only across two
    // macroblocks. A narrower image is coded with the first level instead,
    // where a request for the second by name is refused.
    let subsampled = EncodeRequest::quality(3).with_chroma_sampling(ChromaSampling::Cs420);
    assert_eq!(overlap(48, &subsampled).unwrap(), OverlapMode::Two);
    assert_eq!(overlap(16, &subsampled).unwrap(), OverlapMode::One);
    let named = subsampled.with_overlap(OverlapMode::Two);
    assert_eq!(
        overlap(16, &named).unwrap_err().kind,
        JxrErrorKind::InvalidRequest
    );
    // A mode the request names is the mode coded.
    let named = EncodeRequest::quality(50).with_overlap(OverlapMode::None);
    assert_eq!(overlap(48, &named).unwrap(), OverlapMode::None);
    // Left open beside quantizers given outright: none where nothing is
    // quantized, and the first level otherwise.
    let open = |request: EncodeRequest| EncodeRequest {
        overlap: None,
        ..request
    };
    let lossy = EncodeRequest::lossy(EncodeQuantizers::uniform(30));
    assert_eq!(overlap(48, &open(lossy)).unwrap(), OverlapMode::One);
    assert_eq!(
        overlap(48, &open(EncodeRequest::lossless())).unwrap(),
        OverlapMode::None
    );
}

/// DC, low-pass and high-pass parameters of each component of the primary
/// image of `file`, and then of its alpha image.
fn declared_quantizers(file: &[u8]) -> Vec<[u8; 3]> {
    let container = jxr_native::parse_annex_a(file).unwrap();
    let plane = |range: std::ops::Range<usize>| {
        let plane = jxr_native::parse_codestream_headers(&file[range])
            .unwrap()
            .primary;
        let band = |set: &Option<jxr_native::QuantizerSet>| set.clone().unwrap().components;
        let (dc, low_pass, high_pass) = (
            band(&plane.dc_quantizers),
            band(&plane.lp_quantizers),
            band(&plane.hp_quantizers),
        );
        (0..dc.len())
            .map(|component| [dc[component], low_pass[component], high_pass[component]])
            .collect::<Vec<_>>()
    };
    let mut quantizers = plane(container.codestream_range.clone());
    quantizers.extend(container.alpha_range.clone().map(plane).unwrap_or_default());
    quantizers
}

#[test]
fn quantizers_given_outright_are_coded_as_given_whatever_the_source() {
    let (width, height) = (48, 32);
    let pixels = colour_image(width, height, 4, 31);
    let coded = |format, source: &DecodedSamples, request: &EncodeRequest| {
        declared_quantizers(&recode(width, height, format, source, request).0)
    };
    let eight = DecodedSamples::U8(pixels.clone());
    let sixteen = DecodedSamples::U16(pixels.iter().map(|&s| u16::from(s) * 257).collect());
    let luma = EncodeQuantizers {
        dc: 20,
        low_pass: 40,
        high_pass: 60,
    };
    let chroma = EncodeQuantizers::uniform(90);
    let alpha = EncodeQuantizers::uniform(7);
    for (format, source) in [
        (PixelFormat::U8(ChannelLayout::Rgba), &eight),
        (PixelFormat::U16(ChannelLayout::Rgba), &sixteen),
    ] {
        // Nothing about the source moves a parameter the request gives.
        let request = EncodeRequest::lossy(luma);
        assert_eq!(
            coded(format, source, &request),
            [[20, 40, 60]; 4],
            "{format:?}"
        );
        let request = request
            .with_chroma_quantizers(chroma)
            .with_alpha_quantizers(alpha);
        assert_eq!(
            coded(format, source, &request),
            [[20, 40, 60], [90; 3], [90; 3], [7; 3]],
            "{format:?}"
        );
        // Beside a quality, the planes the request names keep theirs.
        let request = EncodeRequest::quality(60)
            .with_chroma_quantizers(chroma)
            .with_alpha_quantizers(alpha);
        let scale = coded(format, source, &EncodeRequest::quality(60));
        assert_eq!(
            coded(format, source, &request),
            [scale[0], [90; 3], [90; 3], [7; 3]],
            "{format:?}"
        );
    }
    // The scale itself: calibrated for 8 bits, and the same steps against
    // the range at 16, which are 128 parameters on. Alpha is a gray image.
    assert_eq!(
        coded(
            PixelFormat::U8(ChannelLayout::Rgba),
            &eight,
            &EncodeRequest::quality(60)
        ),
        [[58; 3], [73; 3], [73; 3], [58; 3]]
    );
    assert_eq!(
        coded(
            PixelFormat::U16(ChannelLayout::Rgba),
            &sixteen,
            &EncodeRequest::quality(60)
        ),
        [[186; 3], [201; 3], [201; 3], [186; 3]]
    );
}
