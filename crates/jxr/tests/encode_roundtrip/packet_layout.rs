use super::chroma::{
    SUBSAMPLINGS, YUV420, YUV422, flat_chroma_image, overlap_needs_wider_tiles, recode_yuv,
    yuv_samples,
};
use super::*;

pub(super) fn frequency(request: EncodeRequest) -> EncodeRequest {
    request.with_bitstream_mode(BitstreamMode::Frequency)
}

pub(super) fn encode_file(
    (width, height): (u32, u32),
    format: PixelFormat,
    source: &DecodedSamples,
    request: &EncodeRequest,
) -> Vec<u8> {
    let image = SourceImage::new(width, height, format, SourceSamples::from(source)).unwrap();
    encode(&image, request).unwrap()
}

/// The tile layouts a lossless round trip is checked in: one tile, and
/// several with soft and with hard boundaries.
pub(super) const TILE_LAYOUTS: [(u16, bool); 3] = [(255, false), (3, false), (3, true)];

#[test]
fn gray_and_rgb_round_trip_losslessly_across_packet_and_tile_layouts() {
    let mut cases = Vec::new();
    for max_tile in [1, 2, 3, 7] {
        for hard_tiles in [false, true] {
            cases.push((BitstreamMode::Spatial, (150, 100), (max_tile, hard_tiles)));
        }
    }
    for (width, height) in [(1, 1), (37, 21), (150, 100)] {
        for tile in TILE_LAYOUTS {
            cases.push((BitstreamMode::Frequency, (width, height), tile));
        }
    }
    for (packet_mode, (width, height), (max_tile, hard_tiles)) in cases {
        let (gray_seed, rgb) = match packet_mode {
            BitstreamMode::Spatial => (5, None),
            BitstreamMode::Frequency => (
                u64::from(width) + 500,
                Some(colour_image(width, height, 3, u64::from(width) + 600)),
            ),
        };
        let gray = gray_image(width, height, gray_seed);
        for overlap in OVERLAPS {
            let request = EncodeRequest::lossless()
                .with_overlap(overlap)
                .with_max_tile_macroblocks(max_tile)
                .with_hard_tiles(hard_tiles)
                .with_bitstream_mode(packet_mode);
            let case = format!(
                "{packet_mode:?} {width}x{height} {overlap:?} tiles {max_tile} {hard_tiles}"
            );
            let codestream = encode_gray(&gray, width, height, &request);
            assert_eq!(decode_gray(&codestream).2, gray, "{case}");
            let view = JxrView::parse(&codestream).unwrap();
            let info = view.info();
            assert_eq!(info.primary.bitstream_mode, packet_mode, "{case}");
            assert_eq!(info.tiles.hard_tiles, hard_tiles, "{case}");
            if packet_mode == BitstreamMode::Spatial {
                assert!(info.tiles.column_widths.len() > 1, "{case}");
            }

            if let Some(rgb) = rgb.as_deref() {
                let image = SourceImage::new(width, height, RGB8, SourceSamples::U8(rgb)).unwrap();
                let file = encode(&image, &request).unwrap();
                assert_eq!(decode_as(&file, RGB8), rgb, "{case}");
                assert_eq!(declared_layouts(&file), (packet_mode, None), "{case}");
            }
        }
    }
}

#[test]
fn frequency_mode_round_trips_subsampled_chroma_losslessly() {
    for (width, height) in [(37, 21), (96, 70)] {
        let flat = flat_chroma_image(width, height, u64::from(width) + 700);
        for sampling in SUBSAMPLINGS {
            for overlap in OVERLAPS {
                for (max_tile, hard_tiles) in TILE_LAYOUTS {
                    let request = frequency(
                        EncodeRequest::lossless()
                            .with_chroma_sampling(sampling)
                            .with_overlap(overlap)
                            .with_max_tile_macroblocks(max_tile)
                            .with_hard_tiles(hard_tiles),
                    );
                    if overlap_needs_wider_tiles(width, &request) {
                        continue;
                    }
                    let image =
                        SourceImage::new(width, height, RGB8, SourceSamples::U8(&flat)).unwrap();
                    let file = encode(&image, &request).unwrap();
                    let case = format!(
                        "{width}x{height} {sampling:?} {overlap:?} tiles {max_tile} {hard_tiles}"
                    );
                    // Constant chroma loses nothing to the decimation.
                    assert_eq!(decode_as(&file, RGB8), flat, "{case}");
                    let view = JxrView::parse(&file).unwrap();
                    let primary = &view.info().primary;
                    assert_eq!(primary.color_format, ColorFormat::Yuv(sampling), "{case}");
                    assert_eq!(primary.bitstream_mode, BitstreamMode::Frequency, "{case}");
                }
            }
        }
    }
}

#[test]
fn frequency_mode_round_trips_cmyk_and_n_component_losslessly() {
    let (width, height) = (70, 50);
    for sixteen_bit in [false, true] {
        let layouts = [
            (ChannelLayout::Cmyk, 4, false),
            (ChannelLayout::Cmyk, 4, true),
            (ChannelLayout::NComponent(3), 3, false),
            (ChannelLayout::NComponent(5), 5, false),
            (ChannelLayout::NComponent(8), 8, false),
        ];
        for (layout, channels, direct) in layouts {
            let format = multichannel_format(layout, sixteen_bit);
            let source = multichannel_samples(width, height, channels, sixteen_bit, 800);
            for overlap in OVERLAPS {
                for (max_tile, hard_tiles) in TILE_LAYOUTS {
                    let request = frequency(
                        EncodeRequest::lossless()
                            .with_cmyk_direct(direct)
                            .with_overlap(overlap)
                            .with_max_tile_macroblocks(max_tile)
                            .with_hard_tiles(hard_tiles),
                    );
                    let (file, decoded) = recode(width, height, format, &source, &request);
                    let case = format!("{format:?} direct {direct} {overlap:?} tiles {max_tile}");
                    assert_eq!(decoded, source, "{case}");
                    assert_eq!(declared_layouts(&file), (BitstreamMode::Frequency, None));
                }
            }
        }
    }
}

#[test]
fn a_wide_frequency_tile_row_adapts_at_every_boundary_in_every_band_packet() {
    // Forty macroblocks per row cross two 16-macroblock context resets in
    // each of a tile's four packets.
    let (width, height) = (640, 40);
    let request = frequency(one_wide_tile());
    let gray = gray_image(width, height, 19);
    assert_eq!(
        decode_gray(&encode_gray(&gray, width, height, &request)).2,
        gray
    );
    for (layout, channels) in [
        (ChannelLayout::Rgb, 3),
        (ChannelLayout::Cmyk, 4),
        (ChannelLayout::NComponent(6), 6),
    ] {
        let format = PixelFormat::U8(layout);
        let source = multichannel_samples(width, height, channels, false, 21);
        let (_, decoded) = recode(width, height, format, &source, &request);
        assert_eq!(decoded, source, "{layout:?}");
    }
    let source = multichannel_samples(width, height, 4, false, 23);
    for mode in ALPHA_MODES {
        let request = request.clone().with_alpha_mode(mode);
        let format = PixelFormat::U8(ChannelLayout::Bgra);
        let (_, decoded) = recode(width, height, format, &source, &request);
        assert_eq!(decoded, source, "{mode:?}");
    }
    for format in [YUV420, YUV422] {
        let source = yuv_samples(width, height, format, 8, 25);
        let (_, decoded) = recode_yuv((width, height), format, 8, &source, &request);
        assert_eq!(decoded.samples, source, "{format:?}");
    }
}

/// Sources of every coded family, with the options that select the family.
pub(super) fn layout_sources(
    width: u32,
    height: u32,
) -> Vec<(String, PixelFormat, DecodedSamples, EncodeRequest)> {
    let base = EncodeRequest::lossless();
    let rgb = DecodedSamples::U8(colour_image(width, height, 3, 31));
    let rgba = multichannel_samples(width, height, 4, false, 33);
    let rgba_format = PixelFormat::U8(ChannelLayout::Rgba);
    let mut sources = vec![
        (
            "gray".to_owned(),
            GRAY8,
            DecodedSamples::U8(gray_image(width, height, 35)),
            base.clone(),
        ),
        ("rgb".to_owned(), RGB8, rgb.clone(), base.clone()),
        (
            "cmyk".to_owned(),
            PixelFormat::U8(ChannelLayout::Cmyk),
            multichannel_samples(width, height, 4, false, 37),
            base.clone(),
        ),
        (
            "five components".to_owned(),
            PixelFormat::U16(ChannelLayout::NComponent(5)),
            multichannel_samples(width, height, 5, true, 39),
            base.clone(),
        ),
        (
            "yuv 4:2:0".to_owned(),
            YUV420,
            yuv_samples(width, height, YUV420, 8, 41),
            base.clone(),
        ),
    ];
    for sampling in SUBSAMPLINGS {
        sources.push((
            format!("rgb {sampling:?}"),
            RGB8,
            rgb.clone(),
            base.clone().with_chroma_sampling(sampling),
        ));
    }
    for mode in ALPHA_MODES {
        sources.push((
            format!("rgba {mode:?}"),
            rgba_format,
            rgba.clone(),
            base.clone().with_alpha_mode(mode),
        ));
    }
    sources
}

/// `options` with the quantizers, overlap, bands, trim, and tiles of
/// `settings`: the request `options` selects a coded family for, coded as
/// `settings` says.
pub(super) fn with_settings(options: &EncodeRequest, settings: &EncodeRequest) -> EncodeRequest {
    EncodeRequest {
        cmyk_direct: options.cmyk_direct,
        alpha_mode: options.alpha_mode,
        chroma_sampling: options.chroma_sampling,
        ..settings.clone()
    }
}

/// The two layouts carry the same coefficients, so whatever is coded, at
/// whatever loss, both files decode to the same samples and declare the
/// same profile and level.
#[test]
fn frequency_and_spatial_layouts_decode_to_identical_samples() {
    let (width, height) = (70, 50);
    let lossy = |parameter| EncodeRequest::lossy(EncodeQuantizers::uniform(parameter));
    let settings = [
        ("lossless", EncodeRequest::lossless()),
        ("qp 12", lossy(12)),
        (
            "qp 40 overlap 2 hard tiles",
            lossy(40)
                .with_overlap(OverlapMode::Two)
                .with_max_tile_macroblocks(3)
                .with_hard_tiles(true),
        ),
        ("quality 75", EncodeRequest::quality(75)),
        (
            "quality 40 tiles",
            EncodeRequest::quality(40).with_max_tile_macroblocks(2),
        ),
        (
            "qp 30 no flexbits",
            lossy(30).with_bands(BandPresence::NoFlexbits),
        ),
        (
            "qp 30 no high-pass tiles",
            lossy(30)
                .with_bands(BandPresence::NoHighPass)
                .with_max_tile_macroblocks(2),
        ),
        ("qp 30 DC only", lossy(30).with_bands(BandPresence::DcOnly)),
        (
            "lossless trim 3",
            EncodeRequest::lossless().with_trim_flexbits(3),
        ),
        (
            "qp 16 trim 2 tiles",
            lossy(16).with_trim_flexbits(2).with_max_tile_macroblocks(2),
        ),
    ];
    for (name, format, source, options) in layout_sources(width, height) {
        for (setting, request) in &settings {
            let request = with_settings(&options, request);
            // Trimmed flexbits are refused beside an integrated alpha plane.
            if request.alpha_mode == AlphaMode::Integrated
                && format.has_alpha()
                && request.trim_flexbits != 0
            {
                continue;
            }
            let case = format!("{name}, {setting}");
            let spatial = encode_file((width, height), format, &source, &request);
            let frequency = encode_file((width, height), format, &source, &frequency(request));
            assert_eq!(
                declared_layouts(&spatial).0,
                BitstreamMode::Spatial,
                "{case}"
            );
            assert_eq!(
                declared_layouts(&frequency).0,
                BitstreamMode::Frequency,
                "{case}"
            );
            // The same bits, in more packets and behind a longer index.
            assert!(frequency.len() > spatial.len(), "{case}");
            let spatial_image = decode_at(&spatial, format, DecodeScale::Full);
            let frequency_image = decode_at(&frequency, format, DecodeScale::Full);
            assert_eq!(frequency_image.samples, spatial_image.samples, "{case}");
            let (spatial, frequency) = (&spatial_image.info, &frequency_image.info);
            assert_eq!(frequency.profile, spatial.profile, "{case}");
            assert_eq!(frequency.level, spatial.level, "{case}");
        }
    }
}
