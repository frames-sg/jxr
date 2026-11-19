use jxr_core::{
    BackendKind, BackendRequest, BandPresence, ChannelLayout, ColorFormat, CropWindow,
    DecodeLimits, DecodeRequest, DecodeScale, DecodeStage, DecodedSamples, DecodedSamplesMut,
    OutputBitDepth, OutputFormatRequest, PixelFormat, Profile, Rect, StageExecutor, SurfaceLayout,
    SurfacePlaneLayout,
};

#[test]
fn decode_request_defaults_to_auto_and_full_alpha() {
    let request = DecodeRequest::new(PixelFormat::U8(ChannelLayout::Rgba));
    assert_eq!(request.backend, BackendRequest::Auto);
    assert!(request.region.is_none());
    assert_eq!(request.output.channel_count(), 4);
    assert_eq!(request.scale, DecodeScale::Full);
    assert_eq!(DecodeScale::Quarter.denominator(), 4);
    assert_eq!(DecodeScale::Sixteenth.denominator(), 16);
    assert_eq!(
        DecodeScale::Quarter.retained_bands(BandPresence::All),
        BandPresence::NoHighPass
    );
    assert_eq!(
        DecodeScale::Sixteenth.retained_bands(BandPresence::All),
        BandPresence::DcOnly
    );
    assert_eq!(
        DecodeScale::Quarter.retained_bands(BandPresence::DcOnly),
        BandPresence::DcOnly
    );
    assert_eq!(
        request.with_scale(DecodeScale::Quarter).scale,
        DecodeScale::Quarter
    );
}

#[test]
fn limits_reject_products_before_allocation() {
    let limits = DecodeLimits {
        max_pixels: 99,
        ..DecodeLimits::default()
    };
    assert!(limits.check_dimensions(10, 10).is_err());
    assert!(limits.check_dimensions(u32::MAX, u32::MAX).is_err());
}

#[test]
fn typed_samples_report_storage_in_bytes() {
    assert_eq!(DecodedSamples::U16(vec![1, 2, 3]).byte_len(), 6);
    assert_eq!(DecodedSamples::F32(vec![0.0, 1.0]).byte_len(), 8);
    assert_eq!(DecodedSamples::BitPacked(vec![0xff]).sample_count(), 8);
}

#[test]
fn mutable_typed_samples_preserve_the_exact_destination_contract() {
    let mut values = [0_u16; 3];
    let mut destination = DecodedSamplesMut::U16(&mut values);
    assert_eq!(destination.storage_kind(), jxr_core::StorageKind::U16);
    assert_eq!(destination.len(), 3);
    assert_eq!(destination.byte_len(), 6);
    assert!(destination.matches_format(PixelFormat::U16(ChannelLayout::Rgb)));
    assert!(!destination.matches_format(PixelFormat::F16(ChannelLayout::Rgb)));
    let DecodedSamplesMut::U16(reborrowed) = destination.reborrow() else {
        panic!("reborrow changed the destination variant");
    };
    reborrowed.copy_from_slice(&[1, 2, 3]);
    assert_eq!(values, [1, 2, 3]);
}

#[test]
fn surface_layout_checks_strides_ranges_and_format() {
    let layout = SurfaceLayout {
        width: 4,
        height: 2,
        format: PixelFormat::U8(ChannelLayout::Rgba),
        planes: vec![SurfacePlaneLayout {
            byte_offset: 0,
            row_stride_bytes: 16,
            width: 4,
            height: 2,
            channels: 4,
        }],
        byte_len: 32,
        required_alignment: 16,
    };
    assert!(layout.validate().is_ok());

    let mut bad = layout;
    bad.byte_len = 31;
    assert!(bad.validate().is_err());
}

#[test]
fn surface_layout_rejects_overlapping_planar_destinations() {
    let mut layout = SurfaceLayout::for_output(
        OutputFormatRequest {
            internal_color: ColorFormat::Yuv(jxr_core::ChromaSampling::Cs420),
            output_color: ColorFormat::Yuv(jxr_core::ChromaSampling::Cs420),
            bit_depth: OutputBitDepth::U8,
            pixel_format: PixelFormat::U8(ChannelLayout::Yuv(jxr_core::ChromaSampling::Cs420)),
            scaled: false,
            alpha_format: None,
            red_blue_not_swapped: true,
            premultiply_alpha: false,
            crop: CropWindow {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
            },
        },
        1,
    )
    .unwrap();
    layout.planes[1].byte_offset = layout.planes[0].byte_offset;
    assert!(layout.validate().is_err());
}

#[test]
fn shared_rect_accepts_contained_region() {
    assert!(
        Rect {
            x: 1,
            y: 1,
            w: 2,
            h: 2
        }
        .is_within((4, 4))
    );
}

#[test]
fn reconstruction_stages_are_explicit() {
    assert!(BandPresence::All.has_high_pass());
    let stage = jxr_core::StageReport {
        stage: DecodeStage::SecondInverseTransform,
        executor: StageExecutor::Cuda,
    };
    assert_eq!(stage.executor.backend(), BackendKind::Cuda);
}

#[test]
fn main_color_contract_keeps_distinct_wire_semantics() {
    assert_eq!(ColorFormat::CmykDirect.component_count(), Some(4));
    assert_eq!(ColorFormat::YuvK.component_count(), Some(4));
    assert_eq!(ColorFormat::Rgbe.component_count(), Some(3));
    assert_ne!(ColorFormat::Cmyk, ColorFormat::CmykDirect);
    assert_ne!(Profile::SubBaseline, Profile::Baseline);
    assert_ne!(Profile::Baseline, Profile::Main);
}

#[test]
fn output_policy_builds_native_planar_surface_geometry() {
    let policy = OutputFormatRequest {
        internal_color: ColorFormat::Yuv(jxr_core::ChromaSampling::Cs420),
        output_color: ColorFormat::Yuv(jxr_core::ChromaSampling::Cs420),
        bit_depth: OutputBitDepth::U8,
        pixel_format: PixelFormat::U8(ChannelLayout::Yuva(jxr_core::ChromaSampling::Cs420)),
        scaled: false,
        alpha_format: None,
        red_blue_not_swapped: true,
        premultiply_alpha: false,
        crop: CropWindow {
            x: 0,
            y: 0,
            width: 8,
            height: 6,
        },
    };
    let layout = SurfaceLayout::for_output(policy, 1).unwrap();
    assert_eq!(layout.planes.len(), 4);
    assert_eq!((layout.planes[0].width, layout.planes[0].height), (8, 6));
    assert_eq!((layout.planes[1].width, layout.planes[1].height), (4, 3));
    assert_eq!((layout.planes[3].width, layout.planes[3].height), (8, 6));
    assert_eq!(layout.byte_len, 120);
}

#[test]
fn source_image_requires_every_row_inside_its_samples() {
    use jxr_core::{JxrErrorKind, SourceImage, SourceSamples};

    let samples = [0_u8; 3 * 4 * 2 + 5];
    let format = PixelFormat::U8(ChannelLayout::Rgb);
    let tight = SourceImage::new(4, 2, format, SourceSamples::U8(&samples)).unwrap();
    assert_eq!(tight.row_stride, 12);
    // The last row needs only its own samples, not a whole stride.
    let padded = tight.with_row_stride(17).unwrap();
    assert_eq!(padded.row_samples().unwrap(), 12);

    assert_eq!(
        tight.with_row_stride(18).unwrap_err().kind,
        JxrErrorKind::BufferTooSmall {
            required: 30,
            available: 29,
        }
    );
    assert_eq!(
        tight.with_row_stride(11).unwrap_err().kind,
        JxrErrorKind::InvalidRequest
    );
    assert_eq!(
        SourceImage::new(0, 2, format, SourceSamples::U8(&samples))
            .unwrap_err()
            .kind,
        JxrErrorKind::InvalidRequest
    );
}

#[test]
fn source_image_requires_the_exact_packed_storage_variant() {
    use jxr_core::{JxrErrorKind, SourceImage, SourceSamples};

    let words16 = [0_u16; 16];
    let words32 = [0_u32; 16];
    for (format, samples) in [
        (PixelFormat::Rgb555, SourceSamples::Rgb565(&words16)),
        (PixelFormat::Rgb565, SourceSamples::Rgb555(&words16)),
        (PixelFormat::Rgb101010, SourceSamples::Rgbe(&words32)),
        (PixelFormat::Rgbe, SourceSamples::Rgb101010(&words32)),
    ] {
        assert_eq!(
            SourceImage::new(4, 4, format, samples).unwrap_err().kind,
            JxrErrorKind::InvalidRequest,
            "{format:?} with {samples:?}",
        );
    }
}

#[test]
fn planar_source_images_locate_every_plane_inside_their_samples() {
    use jxr_core::{ChromaSampling, JxrErrorKind, SourceImage, SourcePlane, SourceSamples};

    let samples = [0_u8; 8 * 6 * 3];
    // `(sampling, chroma size, tight sample count without and with alpha)`.
    for (sampling, chroma, counts) in [
        (ChromaSampling::Cs420, (4, 3), [72, 120]),
        (ChromaSampling::Cs422, (4, 6), [96, 144]),
    ] {
        for (layout, count) in [ChannelLayout::Yuv(sampling), ChannelLayout::Yuva(sampling)]
            .into_iter()
            .zip(counts)
        {
            let format = PixelFormat::U8(layout);
            let image = SourceImage::new(8, 6, format, SourceSamples::U8(&samples[..count]));
            let image = image.unwrap();
            assert_eq!(image.planar_sampling(), Some(sampling));
            let planes = usize::from(layout.channel_count());
            assert_eq!(image.plane_count(), Some(planes));
            assert_eq!(image.plane_dimensions(0), (8, 6));
            assert_eq!(image.plane_dimensions(1), chroma);
            assert_eq!(image.plane_dimensions(2), chroma);
            assert_eq!(image.plane_dimensions(3), (8, 6));
            // Tight planes follow one another: Y, U, V, alpha.
            let chroma_samples = (chroma.0 * chroma.1) as usize;
            let located = image.planes.unwrap();
            let offsets = [0, 48, 48 + chroma_samples, 48 + 2 * chroma_samples];
            for plane in 0..planes {
                let width = image.plane_dimensions(plane).0 as usize;
                assert_eq!(
                    located[plane],
                    SourcePlane {
                        offset: offsets[plane],
                        row_stride: width
                    },
                    "{layout:?} plane {plane}"
                );
            }
            // One sample short of the last plane is too few.
            assert_eq!(
                SourceImage::new(8, 6, format, SourceSamples::U8(&samples[..count - 1]))
                    .unwrap_err()
                    .kind,
                JxrErrorKind::BufferTooSmall {
                    required: count,
                    available: count - 1,
                }
            );
            // A stride shorter than its plane's rows is rejected.
            let mut narrow = located;
            narrow[1].row_stride = 3;
            assert_eq!(
                image.with_planes(&narrow[..planes]).unwrap_err().kind,
                JxrErrorKind::InvalidRequest
            );
            // Planes are located one by one, not by a common row stride.
            assert_eq!(
                image.with_row_stride(24).unwrap_err().kind,
                JxrErrorKind::InvalidRequest
            );
        }
    }
    // Full-resolution chroma is interleaved like any other format.
    let format = PixelFormat::U8(ChannelLayout::Yuv(ChromaSampling::Cs444));
    let image = SourceImage::new(8, 6, format, SourceSamples::U8(&samples)).unwrap();
    assert_eq!((image.planar_sampling(), image.plane_count()), (None, None));
    assert_eq!((image.planes, image.row_stride), (None, 24));
    assert_eq!(image.plane_dimensions(1), (8, 6));
    assert_eq!(
        image
            .with_planes(&[SourcePlane::default(); 3])
            .unwrap_err()
            .kind,
        JxrErrorKind::InvalidRequest
    );
    // A literal that contradicts its format does not validate.
    let mut contradictory = image;
    contradictory.planes = Some([SourcePlane::default(); 4]);
    assert_eq!(
        contradictory.validate().unwrap_err().kind,
        JxrErrorKind::InvalidRequest
    );
}

#[test]
fn subsampled_source_images_have_even_dimensions_along_halved_axes() {
    use jxr_core::{ChromaSampling, JxrErrorKind, SourceImage, SourceSamples};

    let samples = [0_u8; 256];
    for (sampling, width, height, valid) in [
        (ChromaSampling::Cs420, 6, 4, true),
        (ChromaSampling::Cs420, 5, 4, false),
        (ChromaSampling::Cs420, 6, 3, false),
        (ChromaSampling::Cs422, 6, 3, true),
        (ChromaSampling::Cs422, 5, 3, false),
        (ChromaSampling::Cs444, 5, 3, true),
    ] {
        let format = PixelFormat::U8(ChannelLayout::Yuv(sampling));
        let image = SourceImage::new(width, height, format, SourceSamples::U8(&samples));
        assert_eq!(image.is_ok(), valid, "{sampling:?} {width}x{height}");
        if let Err(error) = image {
            assert_eq!(error.kind, JxrErrorKind::InvalidRequest);
        }
    }
}

#[test]
fn source_sample_depth_and_chroma_centering_fit_the_format() {
    use jxr_core::{ChromaSampling, JxrErrorKind, SourceImage, SourceSamples};

    let wide = [0_u16; 64];
    let bytes = [0_u8; 64];
    let yuv = |sampling| ChannelLayout::Yuv(sampling);
    let deep = SourceImage::new(
        4,
        4,
        PixelFormat::U16(yuv(ChromaSampling::Cs444)),
        SourceSamples::U16(&wide),
    )
    .unwrap();
    assert_eq!(deep.sample_bits, None);
    assert_eq!(deep.with_sample_bits(10).unwrap().sample_bits, Some(10));
    for bits in [0, 17] {
        assert_eq!(
            deep.with_sample_bits(bits).unwrap_err().kind,
            JxrErrorKind::InvalidRequest
        );
    }
    // Bytes hold eight bits and nothing narrower is declared for them.
    let shallow = |sampling| {
        SourceImage::new(
            4,
            4,
            PixelFormat::U8(yuv(sampling)),
            SourceSamples::U8(&bytes),
        )
        .unwrap()
    };
    assert_eq!(
        shallow(ChromaSampling::Cs444)
            .with_sample_bits(8)
            .unwrap_err()
            .kind,
        JxrErrorKind::InvalidRequest
    );

    // Centering codes are zero to four, or seven for an unknown position,
    // on the axes a planar format halves.
    assert_eq!(shallow(ChromaSampling::Cs420).chroma_centering, [0, 0]);
    for (sampling, x, y, valid) in [
        (ChromaSampling::Cs420, 4, 4, true),
        (ChromaSampling::Cs420, 7, 0, true),
        (ChromaSampling::Cs420, 0, 5, false),
        (ChromaSampling::Cs420, 6, 0, false),
        (ChromaSampling::Cs420, 8, 0, false),
        (ChromaSampling::Cs422, 2, 0, true),
        (ChromaSampling::Cs422, 2, 2, false),
        (ChromaSampling::Cs444, 0, 0, true),
        (ChromaSampling::Cs444, 1, 0, false),
    ] {
        let image = shallow(sampling).with_chroma_centering(x, y);
        assert_eq!(image.is_ok(), valid, "{sampling:?} {x},{y}");
        match image {
            Ok(image) => assert_eq!(image.chroma_centering, [x, y]),
            Err(error) => assert_eq!(error.kind, JxrErrorKind::InvalidRequest),
        }
    }
}

#[test]
fn encode_request_codes_chroma_at_full_resolution_unless_asked() {
    use jxr_core::{ChromaSampling, EncodeQuantizers, EncodeRequest};

    for request in [
        EncodeRequest::lossless(),
        EncodeRequest::lossy(EncodeQuantizers::uniform(40)),
        EncodeRequest::quality(75),
        EncodeRequest::quality(0),
    ] {
        assert_eq!(request.chroma_sampling, ChromaSampling::Cs444);
        for sampling in [ChromaSampling::Cs422, ChromaSampling::Cs420] {
            let subsampled = request.clone().with_chroma_sampling(sampling);
            assert_eq!(subsampled.chroma_sampling, sampling);
            assert!(subsampled.validate().is_ok());
            // The quantizers, and what they say of losslessness, are as
            // they were.
            assert_eq!(subsampled.quantization, request.quantization);
            assert_eq!(subsampled.chroma_quantizers, request.chroma_quantizers);
            assert_eq!(subsampled.is_lossless(), request.is_lossless());
        }
    }
}

#[test]
fn encode_request_defaults_and_validation() {
    use jxr_core::{EncodeQuantization, EncodeQuantizers, EncodeRequest, OverlapMode};

    let lossless = EncodeRequest::lossless();
    assert_eq!(lossless.quantization, EncodeQuantization::LOSSLESS);
    assert!(lossless.quantization.is_lossless());
    assert_eq!(lossless.overlap, Some(OverlapMode::None));
    assert!(lossless.validate().is_ok());
    // CMYK goes through the Main-profile colour transform unless asked.
    assert!(!lossless.cmyk_direct);
    let direct = lossless.clone().with_cmyk_direct(true);
    assert!(direct.cmyk_direct && direct.is_lossless());
    assert!(direct.validate().is_ok());

    let lossy = EncodeRequest::lossy(EncodeQuantizers::uniform(40));
    assert_eq!(
        lossy.quantization,
        EncodeQuantization::Quantizers(EncodeQuantizers::uniform(40))
    );
    assert!(!lossy.quantization.is_lossless());
    assert_eq!(lossy.overlap, Some(OverlapMode::One));
    // Quantizers given outright apply to every component and to alpha until
    // those are given their own.
    for component in [0, 1, 7] {
        assert_eq!(lossy.component_quantization(component), lossy.quantization);
    }
    assert_eq!(lossy.alpha_quantization(), lossy.quantization);
    assert!(lossy.clone().with_trim_flexbits(16).validate().is_err());
    assert!(lossy.with_max_tile_macroblocks(0).validate().is_err());
}

#[test]
fn encode_request_packet_layout_is_spatial_unless_asked() {
    use jxr_core::{BitstreamMode, EncodeQuantizers, EncodeRequest};

    for request in [
        EncodeRequest::lossless(),
        EncodeRequest::lossy(EncodeQuantizers::uniform(40)),
        EncodeRequest::quality(75),
    ] {
        assert_eq!(request.bitstream_mode, BitstreamMode::Spatial);
        let frequency = request
            .clone()
            .with_bitstream_mode(BitstreamMode::Frequency);
        assert_eq!(frequency.bitstream_mode, BitstreamMode::Frequency);
        assert!(frequency.validate().is_ok());
        // The layout changes where coefficients go, not which are coded.
        assert_eq!(frequency.quantization, request.quantization);
        assert_eq!(frequency.bands, request.bands);
        assert_eq!(frequency.is_lossless(), request.is_lossless());
        assert_eq!(
            frequency.with_bitstream_mode(BitstreamMode::Spatial),
            request
        );
    }
}

#[test]
fn encode_request_quantizes_alpha_apart_from_colour() {
    use jxr_core::{AlphaMode, EncodeQuantization, EncodeQuantizers, EncodeRequest};

    // Separate alpha is the default, and alpha follows the luma quantizers.
    let lossless = EncodeRequest::lossless();
    assert_eq!(lossless.alpha_mode, AlphaMode::Separate);
    assert_eq!(lossless.alpha_quantizers, None);
    assert_eq!(lossless.alpha_quantization(), EncodeQuantization::LOSSLESS);
    let coarse = EncodeQuantizers::uniform(40);
    let lossy = EncodeRequest::lossy(coarse);
    assert_eq!(
        lossy.alpha_quantization(),
        EncodeQuantization::Quantizers(coarse)
    );
    // Chroma quantizers do not reach alpha.
    let chroma = EncodeRequest::lossless().with_chroma_quantizers(coarse);
    assert_eq!(chroma.alpha_quantization(), EncodeQuantization::LOSSLESS);

    // Each plane is lossless by its own quantizers; the image by both.
    let exact_alpha = lossy
        .clone()
        .with_alpha_quantizers(EncodeQuantizers::LOSSLESS);
    assert!(exact_alpha.alpha_quantization().is_lossless());
    assert!(!exact_alpha.colour_is_lossless() && !exact_alpha.is_lossless());
    let exact_colour = EncodeRequest::lossless().with_alpha_quantizers(coarse);
    assert!(exact_colour.colour_is_lossless() && !exact_colour.is_lossless());
    assert!(lossless.colour_is_lossless() && lossless.is_lossless());

    let integrated = lossless.with_alpha_mode(AlphaMode::Integrated);
    assert_eq!(integrated.alpha_mode, AlphaMode::Integrated);
    assert!(integrated.validate().is_ok());
}

#[test]
fn only_a_source_with_alpha_can_be_premultiplied() {
    use jxr_core::{JxrErrorKind, SourceImage, SourceSamples};

    let samples = [0_u8; 4 * 4 * 2];
    let rgba = PixelFormat::U8(ChannelLayout::Rgba);
    let straight = SourceImage::new(4, 2, rgba, SourceSamples::U8(&samples)).unwrap();
    assert!(!straight.premultiplied_alpha);
    let premultiplied = straight.with_premultiplied_alpha(true).unwrap();
    assert!(premultiplied.premultiplied_alpha);
    assert!(
        !premultiplied
            .with_premultiplied_alpha(false)
            .unwrap()
            .premultiplied_alpha
    );

    // A padding channel is not alpha.
    for layout in [
        ChannelLayout::Rgbx,
        ChannelLayout::Cmyk,
        ChannelLayout::Luma,
    ] {
        let format = PixelFormat::U8(layout);
        assert!(!format.has_alpha(), "{layout:?}");
        let image = SourceImage::new(4, 2, format, SourceSamples::U8(&samples)).unwrap();
        assert_eq!(
            image.with_premultiplied_alpha(true).unwrap_err().kind,
            JxrErrorKind::InvalidRequest,
            "{layout:?}"
        );
    }
    for layout in [
        ChannelLayout::LumaAlpha,
        ChannelLayout::Rgba,
        ChannelLayout::Bgra,
        ChannelLayout::Cmyka,
        ChannelLayout::NComponentAlpha(3),
    ] {
        assert!(PixelFormat::U16(layout).has_alpha(), "{layout:?}");
    }
    assert!(!PixelFormat::Rgb565.has_alpha());
}

#[test]
fn quality_100_and_above_is_lossless() {
    use jxr_core::EncodeRequest;

    for quality in [100, 101, 200, u8::MAX] {
        assert_eq!(
            EncodeRequest::quality(quality),
            EncodeRequest::lossless(),
            "quality {quality}"
        );
    }
    for quality in 0..100 {
        let request = EncodeRequest::quality(quality);
        assert!(!request.is_lossless(), "quality {quality}");
        assert!(!request.colour_is_lossless(), "quality {quality}");
        assert!(request.validate().is_ok(), "quality {quality}");
    }
}

#[test]
fn quality_below_100_is_stored_for_the_encoder_to_resolve() {
    use jxr_core::{EncodeQuantization, EncodeRequest};

    for quality in 0..100 {
        let request = EncodeRequest::quality(quality);
        // The request names the quality and nothing a source would decide:
        // no quantizers of any plane, and no overlap mode.
        assert_eq!(
            request.quantization,
            EncodeQuantization::Quality(quality),
            "quality {quality}"
        );
        assert_eq!(request.chroma_quantizers, None, "quality {quality}");
        assert_eq!(request.alpha_quantizers, None, "quality {quality}");
        assert_eq!(request.overlap, None, "quality {quality}");
        for component in [0, 1, 3] {
            assert_eq!(
                request.component_quantization(component),
                EncodeQuantization::Quality(quality)
            );
        }
        assert_eq!(
            request.alpha_quantization(),
            EncodeQuantization::Quality(quality)
        );
        // Every other option is the lossless request's.
        assert_eq!(
            EncodeRequest {
                quantization: EncodeQuantization::LOSSLESS,
                overlap: EncodeRequest::lossless().overlap,
                ..request.clone()
            },
            EncodeRequest::lossless()
        );
        // No two neighbours ask for the same coding.
        assert_ne!(request, EncodeRequest::quality(quality + 1));
    }
}

#[test]
fn quality_gives_way_to_what_the_request_names() {
    use jxr_core::{EncodeQuantization, EncodeQuantizers, EncodeRequest, OverlapMode};

    let chroma = EncodeQuantizers::uniform(40);
    let alpha = EncodeQuantizers {
        dc: 3,
        low_pass: 9,
        high_pass: 27,
    };
    let request = EncodeRequest::quality(60)
        .with_chroma_quantizers(chroma)
        .with_alpha_quantizers(alpha)
        .with_overlap(OverlapMode::Two);
    // The first component is still the scale's; the rest is as given.
    assert_eq!(
        request.component_quantization(0),
        EncodeQuantization::Quality(60)
    );
    for component in [1, 2, 5] {
        assert_eq!(
            request.component_quantization(component),
            EncodeQuantization::Quantizers(chroma)
        );
    }
    assert_eq!(
        request.alpha_quantization(),
        EncodeQuantization::Quantizers(alpha)
    );
    assert_eq!(request.overlap, Some(OverlapMode::Two));
    assert!(request.validate().is_ok());

    // Lossless quantizers for part of the image do not make a quality
    // lossless, and a lossy part keeps quality 100 from being so.
    let exact_rest = EncodeRequest::quality(60)
        .with_chroma_quantizers(EncodeQuantizers::LOSSLESS)
        .with_alpha_quantizers(EncodeQuantizers::LOSSLESS);
    assert!(!exact_rest.colour_is_lossless() && !exact_rest.is_lossless());
    assert!(exact_rest.alpha_quantization().is_lossless());
    let coarse_chroma = EncodeRequest::quality(100).with_chroma_quantizers(chroma);
    assert!(!coarse_chroma.colour_is_lossless() && !coarse_chroma.is_lossless());
    let coarse_alpha = EncodeRequest::quality(100).with_alpha_quantizers(alpha);
    assert!(coarse_alpha.colour_is_lossless() && !coarse_alpha.is_lossless());
}

#[test]
fn a_quality_of_100_or_more_set_directly_is_lossless() {
    use jxr_core::{EncodeQuantization, EncodeQuantizers, EncodeRequest};

    // The constructor never stores such a quality, but the field is public.
    for quality in [100, 101, u8::MAX] {
        let quantization = EncodeQuantization::Quality(quality);
        assert!(quantization.is_lossless(), "quality {quality}");
        let request = EncodeRequest {
            quantization,
            ..EncodeRequest::lossless()
        };
        assert!(request.is_lossless() && request.colour_is_lossless());
        assert!(request.validate().is_ok());
    }
    assert!(!EncodeQuantization::Quality(99).is_lossless());
    assert!(EncodeQuantization::Quantizers(EncodeQuantizers::uniform(0)).is_lossless());
    assert!(!EncodeQuantization::Quantizers(EncodeQuantizers::uniform(2)).is_lossless());
}
