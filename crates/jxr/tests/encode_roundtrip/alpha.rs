use super::*;

/// Samples of every depth an alpha format is stored in, each of which a
/// lossless encode reproduces exactly: full-range noise for the integer
/// depths, and values the coded form represents for the others.
fn alpha_sources(width: u32, height: u32) -> Vec<(PixelFormat, DecodedSamples)> {
    let pixels = size(width) * size(height);
    let low16 = |value: u32| u16::try_from(value & 0xffff).unwrap();
    let rgba = ChannelLayout::Rgba;
    let mut sources = vec![
        (
            PixelFormat::U8(rgba),
            multichannel_samples(width, height, 4, false, 101),
        ),
        (
            PixelFormat::U8(ChannelLayout::Bgra),
            multichannel_samples(width, height, 4, false, 102),
        ),
        (
            PixelFormat::U16(rgba),
            multichannel_samples(width, height, 4, true, 103),
        ),
        (
            PixelFormat::I16(rgba),
            DecodedSamples::I16(
                values(pixels * 4, 104)
                    .into_iter()
                    .map(|value| low16(value).cast_signed())
                    .collect(),
            ),
        ),
        (
            // The default shift keeps 22 of 32 bits.
            PixelFormat::I32(rgba),
            DecodedSamples::I32(
                values(pixels * 4, 105)
                    .into_iter()
                    .map(|value| (value.cast_signed() >> 6) & !0x3ff)
                    .collect(),
            ),
        ),
        (
            // Any pattern but negative zero, which has no distinct coded sample.
            PixelFormat::F16(rgba),
            DecodedSamples::F16(
                values(pixels * 4, 106)
                    .into_iter()
                    .map(low16)
                    .map(|bits| if bits == 0x8000 { 0 } else { bits })
                    .collect(),
            ),
        ),
        (
            // Thirteen mantissa bits of values at or above 2^-3 are coded.
            PixelFormat::F32(rgba),
            DecodedSamples::F32(
                values(pixels * 4, 107)
                    .into_iter()
                    .map(|value| {
                        let mantissa = f32::from(low16(value) >> 3) / 8192.0;
                        let exponent = i32::try_from(value >> 16 & 7).unwrap() - 3;
                        let magnitude = (1.0 + mantissa) * 2.0_f32.powi(exponent);
                        match value >> 24 & 7 {
                            0 => 0.0,
                            1 | 2 => -magnitude,
                            _ => magnitude,
                        }
                    })
                    .collect(),
            ),
        ),
    ];
    for sixteen_bit in [false, true] {
        sources.push((
            multichannel_format(ChannelLayout::Cmyka, sixteen_bit),
            multichannel_samples(width, height, 5, sixteen_bit, 108),
        ));
        for channels in 3..=8_u16 {
            sources.push((
                multichannel_format(ChannelLayout::NComponentAlpha(channels), sixteen_bit),
                multichannel_samples(
                    width,
                    height,
                    channels + 1,
                    sixteen_bit,
                    110 + u64::from(channels),
                ),
            ));
        }
    }
    sources
}

#[test]
fn alpha_formats_round_trip_losslessly_in_both_packet_and_alpha_modes() {
    for (width, height) in [(1, 1), (37, 21), (70, 50)] {
        for (format, source) in alpha_sources(width, height) {
            for mode in ALPHA_MODES {
                for packet_mode in [BitstreamMode::Spatial, BitstreamMode::Frequency] {
                    let settings: &[(OverlapMode, u16)] = match packet_mode {
                        BitstreamMode::Spatial => &[
                            (OverlapMode::None, 3),
                            (OverlapMode::One, 3),
                            (OverlapMode::Two, 3),
                        ],
                        BitstreamMode::Frequency => &[
                            (OverlapMode::None, 255),
                            (OverlapMode::One, 3),
                            (OverlapMode::Two, 3),
                        ],
                    };
                    for &(overlap, max_tile) in settings {
                        let request = EncodeRequest::lossless()
                            .with_overlap(overlap)
                            .with_max_tile_macroblocks(max_tile)
                            .with_alpha_mode(mode)
                            .with_bitstream_mode(packet_mode);
                        let (file, decoded) = recode(width, height, format, &source, &request);
                        let case = format!(
                            "{format:?} {width}x{height} {mode:?} {packet_mode:?} {overlap:?}"
                        );
                        assert_eq!(decoded, source, "{case}");
                        let view = JxrView::parse(&file).unwrap();
                        assert_eq!(view.info().alpha_mode, mode, "{case}");
                        assert!(!view.info().premultiplied_alpha, "{case}");
                        let alpha = view.info().alpha.as_ref().unwrap();
                        assert_eq!(alpha.color_format, ColorFormat::Luma, "{case}");
                        assert!(!alpha.scaled && !view.info().primary.scaled, "{case}");
                        let declared = declared_format(&file);
                        assert!(declared.alpha && !declared.premultiplied_alpha, "{case}");
                        // A separate alpha image is laid out as its colour image is.
                        assert_eq!(
                            declared_layouts(&file),
                            (packet_mode, Some(packet_mode)),
                            "{case}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn direct_cmyk_with_alpha_round_trips_in_both_alpha_modes() {
    let (width, height) = (37, 21);
    for sixteen_bit in [false, true] {
        let format = multichannel_format(ChannelLayout::Cmyka, sixteen_bit);
        let source = multichannel_samples(width, height, 5, sixteen_bit, 131);
        for mode in ALPHA_MODES {
            let request = EncodeRequest::lossless()
                .with_cmyk_direct(true)
                .with_alpha_mode(mode);
            let (file, decoded) = recode(width, height, format, &source, &request);
            assert_eq!(decoded, source, "{format:?} {mode:?}");
            assert_eq!(
                declared_format(&file).family,
                AnnexAPixelFamily::Cmyk { direct: true }
            );
            assert_eq!(JxrView::parse(&file).unwrap().info().alpha_mode, mode);
        }
    }
}

#[test]
fn a_wide_alpha_tile_row_exercises_every_adaptation_boundary() {
    // Forty macroblocks per row cross two 16-macroblock context resets in
    // the primary and the alpha plane alike.
    let (width, height) = (640, 40);
    for (layout, channels) in [
        (ChannelLayout::Bgra, 4),
        (ChannelLayout::Cmyka, 5),
        (ChannelLayout::NComponentAlpha(5), 6),
    ] {
        let format = PixelFormat::U8(layout);
        let source = multichannel_samples(width, height, channels, false, 137);
        for mode in ALPHA_MODES {
            let request = one_wide_tile().with_alpha_mode(mode);
            let (_, decoded) = recode(width, height, format, &source, &request);
            assert_eq!(decoded, source, "{layout:?} {mode:?}");
        }
    }
}

#[test]
fn the_alpha_mode_decides_how_many_codestreams_an_image_takes() {
    let (width, height) = (40_u32, 24_u32);
    let format = PixelFormat::U8(ChannelLayout::Bgra);
    let source = multichannel_samples(width, height, 4, false, 139);
    let image = SourceImage::new(width, height, format, SourceSamples::from(&source)).unwrap();

    // Separate alpha, the default, is a second Y-only codestream.
    let separate = EncodeRequest::lossless();
    assert_eq!(separate.alpha_mode, AlphaMode::Separate);
    let encoded = encode_codestreams(&image, &separate).unwrap();
    let alpha = encoded.alpha.as_deref().unwrap();
    let alpha_info = JxrView::parse(alpha).unwrap().info().clone();
    assert_eq!(alpha_info.primary.color_format, ColorFormat::Luma);
    assert_eq!(alpha_info.alpha_mode, AlphaMode::None);
    // Without an alpha plane the colour image keeps the profile its colour
    // channels alone need, and its alpha image is an ordinary gray one.
    let primary_info = JxrView::parse(&encoded.primary).unwrap().info().clone();
    assert_eq!(primary_info.alpha_mode, AlphaMode::None);
    assert_eq!(primary_info.profile, Some(Profile::Baseline));
    assert_eq!(alpha_info.profile, Some(Profile::Baseline));
    // The file is the container written from exactly those codestreams.
    let options =
        AnnexAWriteOptions::new(width, height, encoded.pixel_format).with_separate_alpha(alpha);
    assert_eq!(
        write_annex_a(&encoded.primary, &options).unwrap(),
        encode(&image, &separate).unwrap()
    );
    // One codestream cannot hold it.
    assert_eq!(
        encode_codestream(&image, &separate).unwrap_err().kind,
        JxrErrorKind::InvalidRequest
    );

    // Integrated alpha is one codestream, which needs the Main profile.
    let integrated = EncodeRequest::lossless().with_alpha_mode(AlphaMode::Integrated);
    let encoded = encode_codestreams(&image, &integrated).unwrap();
    assert!(encoded.alpha.is_none());
    assert_eq!(
        encoded.primary,
        encode_codestream(&image, &integrated).unwrap()
    );
    let info = JxrView::parse(&encoded.primary).unwrap().info().clone();
    assert_eq!(info.alpha_mode, AlphaMode::Integrated);
    assert_eq!(info.profile, Some(Profile::Main));
    // The raw codestream decodes with its alpha, without a container.
    let decoded = JxrView::parse(&encoded.primary)
        .unwrap()
        .decoder()
        .decode(&DecodeRequest::new(format))
        .unwrap();
    assert_eq!(decoded.samples, source);

    // The alpha channel is part of the pixel format and is never dropped.
    let none = EncodeRequest::lossless().with_alpha_mode(AlphaMode::None);
    for error in [
        encode(&image, &none).unwrap_err(),
        encode_codestream(&image, &none).unwrap_err(),
        encode_codestreams(&image, &none).unwrap_err(),
    ] {
        assert_eq!(error.kind, JxrErrorKind::InvalidRequest);
    }
}

#[test]
fn sources_without_alpha_ignore_the_alpha_options() {
    let (width, height) = (40_u32, 24_u32);
    let format = PixelFormat::U8(ChannelLayout::Rgb);
    let source = DecodedSamples::U8(colour_image(width, height, 3, 149));
    let image = SourceImage::new(width, height, format, SourceSamples::from(&source)).unwrap();
    let plain = encode(&image, &EncodeRequest::lossless()).unwrap();
    for mode in [AlphaMode::None, AlphaMode::Integrated, AlphaMode::Separate] {
        let request = EncodeRequest::lossless()
            .with_alpha_mode(mode)
            .with_alpha_quantizers(EncodeQuantizers::uniform(90));
        assert_eq!(encode(&image, &request).unwrap(), plain, "{mode:?}");
        assert!(
            encode_codestreams(&image, &request)
                .unwrap()
                .alpha
                .is_none()
        );
        // The colour components stay unscaled whatever alpha would use.
        assert!(request.colour_is_lossless() && !request.is_lossless());
    }
}

#[test]
fn premultiplied_sources_are_declared_and_reported_premultiplied() {
    let (width, height) = (37_u32, 21_u32);
    let float = alpha_sources(width, height)
        .into_iter()
        .find(|(format, _)| *format == PixelFormat::F32(ChannelLayout::Rgba))
        .unwrap();
    let cases = [
        (
            PixelFormat::U8(ChannelLayout::Bgra),
            multichannel_samples(width, height, 4, false, 151),
        ),
        (
            PixelFormat::U8(ChannelLayout::Rgba),
            multichannel_samples(width, height, 4, false, 152),
        ),
        (
            PixelFormat::U16(ChannelLayout::Rgba),
            multichannel_samples(width, height, 4, true, 153),
        ),
        float,
    ];
    for (format, source) in cases {
        for premultiplied in [false, true] {
            for mode in ALPHA_MODES {
                let image = SourceImage::new(width, height, format, SourceSamples::from(&source))
                    .unwrap()
                    .with_premultiplied_alpha(premultiplied)
                    .unwrap();
                let request = EncodeRequest::lossless().with_alpha_mode(mode);
                let file = encode(&image, &request).unwrap();
                let case = format!("{format:?} premultiplied {premultiplied} {mode:?}");
                let view = JxrView::parse(&file).unwrap();
                assert_eq!(view.info().premultiplied_alpha, premultiplied, "{case}");
                assert_eq!(
                    declared_format(&file).premultiplied_alpha,
                    premultiplied,
                    "{case}"
                );
                // The samples are coded as given: nothing is multiplied or
                // divided on the way in or out.
                let decoded = view.decoder().decode(&DecodeRequest::new(format)).unwrap();
                assert_eq!(decoded.samples, source, "{case}");
                assert_eq!(decoded.info.premultiplied_alpha, premultiplied, "{case}");
            }
        }
    }
}

#[test]
fn premultiplied_declarations_need_alpha_and_a_container_format() {
    let pixels = [0_u8; 4 * 4 * 5];
    // Only a format with an alpha channel can be premultiplied.
    let rgb = PixelFormat::U8(ChannelLayout::Rgb);
    let error = SourceImage::new(4, 4, rgb, SourceSamples::U8(&pixels))
        .unwrap()
        .with_premultiplied_alpha(true)
        .unwrap_err();
    assert_eq!(error.kind, JxrErrorKind::InvalidRequest);
    // No container format declares premultiplied CMYK or half-float samples,
    // so they are rejected rather than declared straight.
    let halves = [0_u16; 4 * 4 * 4];
    let undeclared = [
        SourceImage::new(
            4,
            4,
            PixelFormat::U8(ChannelLayout::Cmyka),
            SourceSamples::U8(&pixels),
        ),
        SourceImage::new(
            4,
            4,
            PixelFormat::F16(ChannelLayout::Rgba),
            SourceSamples::F16(&halves),
        ),
    ];
    for image in undeclared {
        let image = image.unwrap().with_premultiplied_alpha(true).unwrap();
        for mode in ALPHA_MODES {
            let request = EncodeRequest::lossless().with_alpha_mode(mode);
            assert_eq!(
                encode(&image, &request).unwrap_err().kind,
                JxrErrorKind::Unsupported,
                "{:?} {mode:?}",
                image.format
            );
        }
    }
}

/// The colour channels and the alpha channel of interleaved samples whose
/// pixels end with alpha.
fn split_alpha(samples: &DecodedSamples, channels: usize) -> (Vec<u32>, Vec<u32>) {
    let mut colour = Vec::new();
    let mut alpha = Vec::new();
    for pixel in widened(samples).chunks_exact(channels) {
        colour.extend_from_slice(&pixel[..channels - 1]);
        alpha.push(pixel[channels - 1]);
    }
    (colour, alpha)
}

#[test]
fn alpha_quantizers_are_independent_of_the_colour_quantizers() {
    let (width, height) = (96_u32, 64_u32);
    for (layout, sixteen_bit) in [
        (ChannelLayout::Bgra, false),
        (ChannelLayout::Rgba, true),
        (ChannelLayout::Cmyka, false),
        (ChannelLayout::NComponentAlpha(4), true),
    ] {
        let format = multichannel_format(layout, sixteen_bit);
        let channels = usize::from(layout.channel_count());
        let smooth = colour_image(width, height, channels, 157);
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
        let (source_colour, source_alpha) = split_alpha(&source, channels);
        let lossy = EncodeQuantizers::uniform(if sixteen_bit { 96 } else { 40 });
        for mode in ALPHA_MODES {
            let case = format!("{format:?} {mode:?}");
            // Lossless alpha under lossy colour: only the alpha plane is
            // unscaled, and it alone is exact.
            let request = EncodeRequest::lossy(lossy)
                .with_alpha_quantizers(EncodeQuantizers::LOSSLESS)
                .with_alpha_mode(mode);
            assert!(!request.is_lossless() && !request.colour_is_lossless());
            let (file, decoded) = recode(width, height, format, &source, &request);
            let (colour, alpha) = split_alpha(&decoded, channels);
            assert_eq!(alpha, source_alpha, "{case}");
            assert_ne!(colour, source_colour, "{case}");
            let info = JxrView::parse(&file).unwrap().info().clone();
            assert!(info.primary.scaled, "{case}");
            assert!(!info.alpha.unwrap().scaled, "{case}");

            // The reverse: exact colour beside quantized alpha.
            let request = EncodeRequest::lossless()
                .with_alpha_quantizers(lossy)
                .with_alpha_mode(mode);
            assert!(!request.is_lossless() && request.colour_is_lossless());
            let (file, decoded) = recode(width, height, format, &source, &request);
            let (colour, alpha) = split_alpha(&decoded, channels);
            assert_eq!(colour, source_colour, "{case}");
            assert_ne!(alpha, source_alpha, "{case}");
            let info = JxrView::parse(&file).unwrap().info().clone();
            assert!(!info.primary.scaled, "{case}");
            assert!(info.alpha.unwrap().scaled, "{case}");

            // By default alpha takes the luma quantizers.
            let request = EncodeRequest::lossy(lossy).with_alpha_mode(mode);
            let (_, decoded) = recode(width, height, format, &source, &request);
            let (_, alpha) = split_alpha(&decoded, channels);
            assert_ne!(alpha, source_alpha, "{case}");
        }
    }
}

#[test]
fn lossy_alpha_images_stay_close_and_shrink_in_both_alpha_modes() {
    let (width, height) = (96_u32, 64_u32);
    for (layout, sixteen_bit) in [
        (ChannelLayout::Bgra, false),
        (ChannelLayout::Rgba, false),
        (ChannelLayout::Rgba, true),
        (ChannelLayout::Cmyka, false),
        (ChannelLayout::Cmyka, true),
        (ChannelLayout::NComponentAlpha(3), false),
        (ChannelLayout::NComponentAlpha(8), true),
    ] {
        let format = multichannel_format(layout, sixteen_bit);
        let smooth = colour_image(width, height, usize::from(layout.channel_count()), 163);
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
        let parameters = if sixteen_bit { [80, 112] } else { [24, 56] };
        for mode in ALPHA_MODES {
            let lossless = EncodeRequest::lossless().with_alpha_mode(mode);
            let (lossless, _) = recode(width, height, format, &source, &lossless);
            let mut previous = (lossless.len(), 0.0_f64);
            for parameter in parameters {
                for bands in [BandPresence::All, BandPresence::NoFlexbits] {
                    let request = EncodeRequest::lossy(EncodeQuantizers::uniform(parameter))
                        .with_bands(bands)
                        .with_max_tile_macroblocks(4)
                        .with_alpha_mode(mode);
                    let (file, decoded) = recode(width, height, format, &source, &request);
                    let error = relative_error(&source, &decoded);
                    let case = format!("{format:?} {mode:?} qp {parameter} {bands:?}");
                    assert!(error > 0.0, "{case} is lossy");
                    assert!(error < 0.05, "{case}: error {error}");
                    if bands == BandPresence::All {
                        assert!(file.len() < previous.0, "{case}");
                        assert!(error > previous.1, "{case}");
                        previous = (file.len(), error);
                    }
                }
            }
        }
    }
}

#[test]
fn float_rgb_without_padding_is_declared_by_its_wic_format() {
    let (width, height) = (37_u32, 21_u32);
    let format = PixelFormat::F32(ChannelLayout::Rgb);
    let (_, rgba) = alpha_sources(width, height)
        .into_iter()
        .find(|(format, _)| *format == PixelFormat::F32(ChannelLayout::Rgba))
        .unwrap();
    let DecodedSamples::F32(rgba) = rgba else {
        panic!("float source has another storage");
    };
    let source = DecodedSamples::F32(
        rgba.as_chunks::<4>()
            .0
            .iter()
            .flat_map(|pixel| pixel[..3].to_vec())
            .collect(),
    );
    let (file, decoded) = recode(width, height, format, &source, &EncodeRequest::lossless());
    assert_eq!(decoded, source);
    let declared = declared_format(&file);
    assert_eq!(
        (declared.family, declared.channels, declared.alpha),
        (AnnexAPixelFamily::Rgb, 3, false)
    );
}

#[test]
fn trimmed_flexbits_reach_alpha_only_as_a_separate_image() {
    let (width, height) = (70_u32, 50_u32);
    let format = PixelFormat::U8(ChannelLayout::Bgra);
    let source = multichannel_samples(width, height, 4, false, 167);
    let image = SourceImage::new(width, height, format, SourceSamples::from(&source)).unwrap();
    let trimmed = EncodeRequest::lossless().with_trim_flexbits(3);
    // Each image of a separate-alpha file trims its own flexbits.
    let (file, decoded) = recode(width, height, format, &source, &trimmed);
    let error = relative_error(&source, &decoded);
    assert!(error > 0.0 && error < 0.02, "error {error}");
    let (lossless, _) = recode(width, height, format, &source, &EncodeRequest::lossless());
    assert!(file.len() < lossless.len());
    // The reference decoders misread a trimmed integrated alpha plane, so
    // the combination is refused rather than written for this decoder alone.
    let integrated = trimmed.clone().with_alpha_mode(AlphaMode::Integrated);
    assert_eq!(
        encode(&image, &integrated).unwrap_err().kind,
        JxrErrorKind::Unsupported
    );
    assert_eq!(
        encode_codestream(&image, &integrated).unwrap_err().kind,
        JxrErrorKind::Unsupported
    );
    // A trim the band set never applies is no trim at all.
    let without_flexbits = integrated.with_bands(BandPresence::NoFlexbits);
    assert!(encode(&image, &without_flexbits).is_ok());
    // Sources without alpha are unaffected by the alpha mode.
    let rgb = PixelFormat::U8(ChannelLayout::Rgb);
    let pixels = colour_image(width, height, 3, 173);
    let image = SourceImage::new(width, height, rgb, SourceSamples::U8(&pixels)).unwrap();
    let request = EncodeRequest::lossless()
        .with_trim_flexbits(3)
        .with_alpha_mode(AlphaMode::Integrated);
    assert!(encode(&image, &request).is_ok());
}
