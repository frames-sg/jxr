use jxr::{
    AlphaHandling, AlphaMode, BackendRequest, BandPresence, BitstreamMode, ChannelLayout,
    ColorFormat, DecodeReport, DecodedImage, DecodedSamples, ImageInfo, ImageMetadata, OverlapMode,
    PixelFormat, PlaneDescriptor, PlaneInfo, Rect, SampleFormat, TileGrid,
};
use jxr_image::{AlphaRepresentation, ImageAdapterError, into_image_frame};
use std::sync::Arc;

fn minimal_raw_codestream() -> Vec<u8> {
    let mut bytes = b"WMPHOTO\0".to_vec();
    bytes.extend_from_slice(&[0x11, 0x00, 0x80, 0x01]);
    bytes.extend_from_slice(&15_u16.to_be_bytes());
    bytes.extend_from_slice(&15_u16.to_be_bytes());
    let mut bit_position = bytes.len() * 8;
    for (value, count) in [(0, 3), (0, 1), (3, 4), (1, 1), (1, 8)] {
        for shift in (0..count).rev() {
            if bit_position / 8 == bytes.len() {
                bytes.push(0);
            }
            let bit = u8::from(((value >> shift) & 1) != 0);
            bytes[bit_position / 8] |= bit << (7 - (bit_position % 8));
            bit_position += 1;
        }
    }
    bytes.extend_from_slice(&[0xFD, 0, 0, 1, 0x5a, 0, 0, 0]);
    bytes
}

fn decoded(format: PixelFormat, samples: DecodedSamples) -> DecodedImage {
    let channels = format.channel_count();
    let width = 2;
    let height = 1;
    DecodedImage {
        info: ImageInfo {
            width,
            height,
            profile: None,
            level: None,
            primary: PlaneInfo {
                color_format: ColorFormat::Rgb,
                sample_format: SampleFormat::Unsigned { bits: 8 },
                bands: BandPresence::All,
                bitstream_mode: BitstreamMode::Spatial,
                overlap: OverlapMode::One,
                short_header: true,
                long_word: false,
                scaled: false,
                chroma_centering: [0, 0],
                shift_bits: 0,
                mantissa_length: 0,
                exponent_bias: 0,
                width,
                height,
            },
            alpha_mode: AlphaMode::None,
            premultiplied_alpha: false,
            alpha: None,
            tiles: TileGrid {
                column_widths: vec![1],
                row_heights: vec![1],
                hard_tiles: false,
            },
            metadata: ImageMetadata::default(),
        },
        decoded_region: Rect {
            x: 0,
            y: 0,
            w: width,
            h: height,
        },
        format,
        planes: vec![PlaneDescriptor {
            byte_offset: 0,
            row_stride_bytes: format.row_bytes(width).unwrap(),
            width,
            height,
            channels,
        }],
        samples,
        report: DecodeReport::cpu(BackendRequest::Cpu),
    }
}

#[test]
fn rgb8_conversion_preserves_pixels_profile_and_metadata() {
    let pixels = vec![1, 2, 3, 4, 5, 6];
    let original_allocation = pixels.as_ptr();
    let decoded = decoded(
        PixelFormat::U8(ChannelLayout::Rgb),
        DecodedSamples::U8(pixels),
    );

    let frame = into_image_frame(decoded, Some(&[7, 8, 9]), AlphaHandling::Drop).unwrap();

    assert_eq!(
        frame.decoded_region(),
        Rect {
            x: 0,
            y: 0,
            w: 2,
            h: 1
        }
    );
    assert_eq!(frame.icc_profile(), Some(&[7, 8, 9][..]));
    assert_eq!(frame.alpha_representation(), AlphaRepresentation::None);
    assert_eq!(
        frame.image().color_space().primaries,
        image::metadata::CicpColorPrimaries::Unspecified
    );
    assert_eq!(
        frame.image().color_space().transfer,
        image::metadata::CicpTransferCharacteristics::Unspecified
    );
    let image::DynamicImage::ImageRgb8(image) = frame.into_image() else {
        panic!("expected RGB8 image");
    };
    assert_eq!(image.as_raw().as_ptr(), original_allocation);
    assert_eq!(image.into_raw(), [1, 2, 3, 4, 5, 6]);
}

#[test]
fn premultiplied_alpha_is_explicit_in_the_adapter_contract() {
    let decoded = decoded(
        PixelFormat::U8(ChannelLayout::Rgba),
        DecodedSamples::U8(vec![1, 2, 3, 4, 5, 6, 7, 8]),
    );

    let frame = into_image_frame(decoded, None, AlphaHandling::Premultiply).unwrap();

    assert_eq!(
        frame.alpha_representation(),
        AlphaRepresentation::Premultiplied
    );
}

#[test]
fn unsupported_channel_order_is_rejected_without_conversion() {
    let decoded = decoded(
        PixelFormat::U8(ChannelLayout::Bgr),
        DecodedSamples::U8(vec![1, 2, 3, 4, 5, 6]),
    );

    assert!(matches!(
        into_image_frame(decoded, None, AlphaHandling::Drop),
        Err(ImageAdapterError::UnsupportedFormat {
            format: PixelFormat::U8(ChannelLayout::Bgr)
        })
    ));
}

#[test]
fn rgb16_and_rgb32f_keep_their_native_sample_types() {
    let rgb16 = into_image_frame(
        decoded(
            PixelFormat::U16(ChannelLayout::Rgb),
            DecodedSamples::U16(vec![1, 2, 3, 4, 5, 6]),
        ),
        None,
        AlphaHandling::Drop,
    )
    .unwrap()
    .into_image();
    let image::DynamicImage::ImageRgb16(rgb16) = rgb16 else {
        panic!("expected RGB16 image");
    };
    assert_eq!(rgb16.into_raw(), [1, 2, 3, 4, 5, 6]);

    let rgb32f = into_image_frame(
        decoded(
            PixelFormat::F32(ChannelLayout::Rgb),
            DecodedSamples::F32(vec![0.0, 0.25, 0.5, 0.75, 1.0, -1.0]),
        ),
        None,
        AlphaHandling::Drop,
    )
    .unwrap()
    .into_image();
    let image::DynamicImage::ImageRgb32F(rgb32f) = rgb32f else {
        panic!("expected RGB32F image");
    };
    assert_eq!(rgb32f.into_raw(), [0.0, 0.25, 0.5, 0.75, 1.0, -1.0]);
}

#[test]
fn unrepresentable_alpha_and_plane_metadata_are_rejected() {
    let mut premultiplied_without_alpha = decoded(
        PixelFormat::U8(ChannelLayout::Rgb),
        DecodedSamples::U8(vec![1, 2, 3, 4, 5, 6]),
    );
    premultiplied_without_alpha.info.premultiplied_alpha = true;
    assert!(matches!(
        into_image_frame(premultiplied_without_alpha, None, AlphaHandling::Drop),
        Err(ImageAdapterError::InvalidLayout { .. })
    ));

    let mut wrong_channels = decoded(
        PixelFormat::U8(ChannelLayout::Rgb),
        DecodedSamples::U8(vec![1, 2, 3, 4, 5, 6]),
    );
    wrong_channels.planes[0].channels = 1;
    assert!(matches!(
        into_image_frame(wrong_channels, None, AlphaHandling::Drop),
        Err(ImageAdapterError::InvalidLayout { .. })
    ));
}

#[test]
fn borrowed_and_prepared_decode_entry_points_match() {
    let bytes = minimal_raw_codestream();
    let request = jxr::DecodeRequest::new(PixelFormat::U8(ChannelLayout::Luma))
        .with_alpha(AlphaHandling::Drop)
        .with_backend(BackendRequest::Cpu);
    let view = jxr::JxrView::parse(&bytes).unwrap();
    let borrowed = jxr_image::decode_view(&view, &request).unwrap();
    let prepared = jxr::PreparedJxr::from_arc(Arc::from(bytes)).unwrap();
    let owned = jxr_image::decode_prepared(&prepared, &request).unwrap();

    assert_eq!(borrowed.image().width(), 16);
    assert_eq!(borrowed.image().height(), 16);
    assert_eq!(borrowed.format(), request.output);
    assert_eq!(borrowed.image().as_bytes(), owned.image().as_bytes());
}
