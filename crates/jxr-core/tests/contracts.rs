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
fn core_uses_shared_backend_and_rect_contracts() {
    assert_eq!(BackendKind::Cpu, BackendKind::Cpu);
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
