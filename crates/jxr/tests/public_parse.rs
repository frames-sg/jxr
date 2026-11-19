mod common;

use std::sync::Arc;

use jxr::{
    AlphaHandling, AnnexAWriteOptions, ChannelLayout, DecodeRequest, DecodeScale, JxrErrorKind,
    JxrView, Orientation, PixelFormat, PreparedJxr, Profile, write_annex_a,
};

fn minimal_raw_codestream() -> Vec<u8> {
    // Y-only, all bands, uniform DC, LP and HP quantizers of one; the VLW
    // escape mode encodes zero subsequent bytes, followed by one tile packet.
    common::codestream(
        0x01,
        &[
            (0, 3),
            (0, 1),
            (0, 4),
            (1, 1),
            (1, 8),
            (0, 1),
            (1, 1),
            (1, 8),
            (0, 1),
            (1, 1),
            (1, 8),
        ],
        &[0xFD, 0, 0, 1, 0],
    )
}

/// Macroblock payload whose DC decodes to different samples under scaled and
/// unscaled arithmetic (127..=128 versus 126..=127), so a plane decoded with the
/// wrong flag is observable.
const NONZERO_DC_MACROBLOCK: [u8; 3] = [0x0c, 0xc0, 0x00];

#[test]
fn borrowed_view_reports_inferred_advanced_profile() {
    let bytes = minimal_raw_codestream();
    let view = JxrView::parse(&bytes).expect("valid synthetic codestream");

    assert_eq!(view.info().dimensions(), (16, 16));
    assert_eq!(view.info().profile, Some(Profile::Advanced));
    assert_eq!(view.info().level.map(|level| level.0), Some(255));
}

#[test]
fn prepared_image_retains_arc_and_builds_exact_coefficient_budget() {
    let bytes: Arc<[u8]> = minimal_raw_codestream().into();
    let prepared = PreparedJxr::from_arc(Arc::clone(&bytes)).expect("valid synthetic codestream");
    assert!(Arc::ptr_eq(prepared.bytes(), &bytes));

    let decoder = prepared.decoder();
    let request = DecodeRequest::new(PixelFormat::U8(ChannelLayout::Luma));
    let plan = decoder.prepare(&request).expect("valid plan");
    assert_eq!(plan.coefficient_bytes, 256 * size_of::<i32>());
    assert_eq!(plan.tiles.len(), 1);
}

#[test]
fn native_reduction_rejects_spatially_interleaved_packets() {
    let bytes = minimal_raw_codestream();
    let view = JxrView::parse(&bytes).unwrap();
    let request =
        DecodeRequest::new(PixelFormat::U8(ChannelLayout::Luma)).with_scale(DecodeScale::Sixteenth);
    let error = view.decoder().prepare(&request).unwrap_err();
    assert_eq!(error.kind, JxrErrorKind::Unsupported);
}

fn annex_a_with_icc(codestream: &[u8]) -> Vec<u8> {
    const ICC: &[u8] = b"icc!";
    const ENTRY_COUNT: u16 = 6;
    const GUID_OFFSET: u32 = 96;
    const ICC_OFFSET: u32 = 112;
    const IMAGE_OFFSET: u32 = 128;
    let mut bytes = vec![0_u8; IMAGE_OFFSET as usize + codestream.len()];
    bytes[0..4].copy_from_slice(&[b'I', b'I', 0xBC, 1]);
    bytes[4..8].copy_from_slice(&8_u32.to_le_bytes());
    bytes[8..10].copy_from_slice(&ENTRY_COUNT.to_le_bytes());
    let entries = [
        (
            0x8773_u16,
            1_u16,
            u32::try_from(ICC.len()).unwrap(),
            ICC_OFFSET,
        ),
        (0xBC01, 1, 16, GUID_OFFSET),
        (0xBC80, 4, 1, 16),
        (0xBC81, 4, 1, 16),
        (0xBCC0, 4, 1, IMAGE_OFFSET),
        (0xBCC1, 4, 1, u32::try_from(codestream.len()).unwrap()),
    ];
    for (index, (tag, kind, count, value)) in entries.into_iter().enumerate() {
        let offset = 10 + index * 12;
        bytes[offset..offset + 2].copy_from_slice(&tag.to_le_bytes());
        bytes[offset + 2..offset + 4].copy_from_slice(&kind.to_le_bytes());
        bytes[offset + 4..offset + 8].copy_from_slice(&count.to_le_bytes());
        bytes[offset + 8..offset + 12].copy_from_slice(&value.to_le_bytes());
    }
    bytes[ICC_OFFSET as usize..ICC_OFFSET as usize + ICC.len()].copy_from_slice(ICC);
    bytes[IMAGE_OFFSET as usize..].copy_from_slice(codestream);
    bytes
}

#[test]
fn borrowed_and_owned_views_expose_embedded_icc_bytes() {
    let bytes = annex_a_with_icc(&minimal_raw_codestream());
    let view = JxrView::parse(&bytes).unwrap();
    assert_eq!(view.icc_profile(), Some(b"icc!".as_slice()));

    let prepared = PreparedJxr::from_arc(Arc::from(bytes)).unwrap();
    assert_eq!(prepared.icc_profile(), Some(b"icc!".as_slice()));
}

#[test]
fn public_writer_produces_a_decodable_oriented_container() {
    let codestream = common::dc_only_luma_codestream(false, [0; 3]);
    let alpha = common::dc_only_luma_codestream(false, [0; 3]);
    let options = AnnexAWriteOptions::new(16, 16, [0xa5; 16])
        .with_orientation(Orientation::Rotate270)
        .with_icc_profile(b"profile")
        .with_separate_alpha(&alpha);

    let bytes = write_annex_a(&codestream, &options).expect("valid container");
    let view = JxrView::parse(&bytes).expect("writer output must parse through the facade");

    assert_eq!(view.info().metadata.orientation, Orientation::Rotate270);
    assert_eq!(view.icc_profile(), Some(b"profile".as_slice()));
    assert_eq!(view.codestream(), codestream);
    assert_eq!(view.separate_alpha_codestream(), Some(alpha.as_slice()));
    let image = view
        .decoder()
        .decode(
            &DecodeRequest::new(PixelFormat::U8(ChannelLayout::Luma))
                .with_alpha(AlphaHandling::Drop),
        )
        .expect("writer output must decode");
    assert_eq!(image.decoded_region.w, 16);
    assert_eq!(image.decoded_region.h, 16);
}

#[test]
fn reference_encoder_alpha_end_offset_decodes_like_a_standard_count() {
    let primary = common::dc_only_luma_codestream(false, [0; 3]);
    let alpha = common::dc_only_luma_codestream(false, [0; 3]);
    let options = AnnexAWriteOptions::new(16, 16, [0xa5; 16]).with_separate_alpha(&alpha);
    let mut bytes = write_annex_a(&primary, &options).unwrap();
    let request = DecodeRequest::new(PixelFormat::U8(ChannelLayout::LumaAlpha));
    let expected = JxrView::parse(&bytes)
        .unwrap()
        .decoder()
        .decode(&request)
        .unwrap();

    let root = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let count = u16::from_le_bytes(bytes[root..root + 2].try_into().unwrap()) as usize;
    let alpha_count = (0..count)
        .map(|index| root + 2 + index * 12)
        .find(|&offset| bytes[offset..offset + 2] == 0xBCC3_u16.to_le_bytes())
        .unwrap();
    let end = u32::try_from(bytes.len()).unwrap();
    bytes[alpha_count + 8..alpha_count + 12].copy_from_slice(&end.to_le_bytes());

    let view = JxrView::parse(&bytes).unwrap();
    assert_eq!(view.separate_alpha_codestream(), Some(alpha.as_slice()));
    let actual = view.decoder().decode(&request).unwrap();
    assert_eq!(actual.samples, expected.samples);
    assert_eq!(actual.planes, expected.planes);
}

fn decode_u8(bytes: &[u8], layout: ChannelLayout, alpha: AlphaHandling) -> Vec<u8> {
    let request = DecodeRequest::new(PixelFormat::U8(layout)).with_alpha(alpha);
    let image = JxrView::parse(bytes)
        .expect("codestream must parse")
        .decoder()
        .decode(&request)
        .expect("codestream must decode");
    let jxr::DecodedSamples::U8(samples) = image.samples else {
        panic!("expected U8 output");
    };
    samples
}

// Regression for frames-sg/jxr#29: JXRLib writers (e.g. ImageMagick) emit an
// unscaled lossless alpha plane beside a scaled primary plane. SCALED_FLAG is a
// per-plane field, so each plane must decode with its own value, in either
// direction.
#[test]
fn separate_alpha_plane_decodes_with_its_own_scaled_flag() {
    for (primary_scaled, alpha_scaled) in [(true, false), (false, true)] {
        let primary = common::dc_only_luma_codestream(primary_scaled, NONZERO_DC_MACROBLOCK);
        let alpha = common::dc_only_luma_codestream(alpha_scaled, NONZERO_DC_MACROBLOCK);
        let standalone_primary = decode_u8(&primary, ChannelLayout::Luma, AlphaHandling::Drop);
        let standalone_alpha = decode_u8(&alpha, ChannelLayout::Luma, AlphaHandling::Drop);
        assert_ne!(
            standalone_primary, standalone_alpha,
            "fixture must make the scaled flag observable in decoded samples"
        );

        let options = AnnexAWriteOptions::new(16, 16, [0xa5; 16]).with_separate_alpha(&alpha);
        let bytes = write_annex_a(&primary, &options).expect("valid container");
        let samples = decode_u8(&bytes, ChannelLayout::LumaAlpha, AlphaHandling::Preserve);

        let (pixels, remainder) = samples.as_chunks::<2>();
        assert!(
            remainder.is_empty(),
            "LumaAlpha output must be whole pixels"
        );
        let (luma, alpha): (Vec<u8>, Vec<u8>) = pixels.iter().map(|&[y, a]| (y, a)).unzip();
        assert_eq!(
            luma, standalone_primary,
            "primary scaled={primary_scaled}, alpha scaled={alpha_scaled}"
        );
        assert_eq!(
            alpha, standalone_alpha,
            "primary scaled={primary_scaled}, alpha scaled={alpha_scaled}"
        );
    }
}

/// A 16-by-16 three-component image of constant `[200, 60, 100]` pixels,
/// coded by the T.835 reference encoder with quantization parameter 40, so
/// with `SCALED_FLAG` set.
const SCALED_THREE_COMPONENT_IMAGE: [u8; 152] = [
    0x49, 0x49, 0xbc, 0x01, 0x08, 0x00, 0x00, 0x00, 0x05, 0x00, 0x01, 0xbc, 0x01, 0x00, 0x10, 0x00,
    0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x80, 0xbc, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00,
    0x00, 0x00, 0x81, 0xbc, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0xc0, 0xbc,
    0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x5a, 0x00, 0x00, 0x00, 0xc1, 0xbc, 0x04, 0x00, 0x01, 0x00,
    0x00, 0x00, 0x3e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0xc3, 0xdd, 0x6f, 0x03, 0x4e,
    0xfe, 0x4b, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9, 0x20, 0x57, 0x4d, 0x50, 0x48, 0x4f, 0x54,
    0x4f, 0x00, 0x11, 0x01, 0xc0, 0x61, 0x00, 0x0f, 0x00, 0x0f, 0xd0, 0x20, 0x85, 0x08, 0x50, 0x85,
    0x00, 0x00, 0x04, 0x42, 0xff, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x21, 0x07, 0xd0, 0xd4, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x44, 0x90,
];

// A scaled plane carries every component after the first at half weight
// through its DC and low-pass bands, whatever its colour format; the T.834
// suite only exercises that for YUV. The expected pixels are what the T.835
// reference decoder returns for this image.
#[test]
fn scaled_n_component_planes_double_every_component_after_the_first() {
    let view = JxrView::parse(&SCALED_THREE_COMPONENT_IMAGE).expect("reference image must parse");
    assert!(view.info().primary.scaled);
    assert_eq!(
        view.info().primary.color_format,
        jxr::ColorFormat::NComponent(3)
    );
    let samples = decode_u8(
        &SCALED_THREE_COMPONENT_IMAGE,
        ChannelLayout::NComponent(3),
        AlphaHandling::Drop,
    );
    assert_eq!(samples.len(), 16 * 16 * 3);
    for pixel in samples.as_chunks::<3>().0 {
        assert_eq!(*pixel, [200, 60, 99]);
    }
}
