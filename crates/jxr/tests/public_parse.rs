use std::sync::Arc;

use jxr::{
    AlphaHandling, AnnexAWriteOptions, ChannelLayout, DecodeRequest, DecodeScale, JxrErrorKind,
    JxrView, Orientation, PixelFormat, PreparedJxr, Profile, write_annex_a,
};

fn minimal_raw_codestream() -> Vec<u8> {
    let mut bytes = b"WMPHOTO\0".to_vec();
    bytes.extend_from_slice(&[0x11, 0x00, 0x80, 0x01]);
    bytes.extend_from_slice(&15_u16.to_be_bytes());
    bytes.extend_from_slice(&15_u16.to_be_bytes());
    let mut bit_position = bytes.len() * 8;
    for (value, count) in [
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
    ] {
        push_bits(&mut bytes, &mut bit_position, value, count);
    }
    // VLW escape mode encodes zero subsequent bytes, followed by one tile packet.
    bytes.extend_from_slice(&[0xFD, 0, 0, 1, 0]);
    bytes
}

fn decodable_minimal_raw_codestream() -> Vec<u8> {
    dc_only_luma_codestream(false, [0, 0, 0])
}

/// Macroblock payload whose DC decodes to different samples under scaled and
/// unscaled arithmetic (127..=128 versus 126..=127), so a plane decoded with the
/// wrong flag is observable.
const NONZERO_DC_MACROBLOCK: [u8; 3] = [0x0c, 0xc0, 0x00];

/// 16x16 DC-only luma codestream. `scaled` sets the plane header's
/// scaled-arithmetic bit; `macroblock` is the tile's entropy-coded payload.
fn dc_only_luma_codestream(scaled: bool, macroblock: [u8; 3]) -> Vec<u8> {
    let mut bytes = b"WMPHOTO\0".to_vec();
    bytes.extend_from_slice(&[0x11, 0x00, 0x80, 0x01]);
    bytes.extend_from_slice(&15_u16.to_be_bytes());
    bytes.extend_from_slice(&15_u16.to_be_bytes());
    let mut bit_position = bytes.len() * 8;
    for (value, count) in [(0, 3), (u64::from(scaled), 1), (3, 4), (1, 1), (1, 8)] {
        push_bits(&mut bytes, &mut bit_position, value, count);
    }
    bytes.extend_from_slice(&[0xFD, 0, 0, 1, 0x5a]);
    bytes.extend_from_slice(&macroblock);
    bytes
}

fn push_bits(bytes: &mut Vec<u8>, bit_position: &mut usize, value: u64, count: u8) {
    for shift in (0..count).rev() {
        if *bit_position / 8 == bytes.len() {
            bytes.push(0);
        }
        let bit = u8::from(((value >> shift) & 1) != 0);
        bytes[*bit_position / 8] |= bit << (7 - (*bit_position % 8));
        *bit_position += 1;
    }
}

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
    let codestream = decodable_minimal_raw_codestream();
    let alpha = decodable_minimal_raw_codestream();
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
    let primary = decodable_minimal_raw_codestream();
    let alpha = decodable_minimal_raw_codestream();
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
        let primary = dc_only_luma_codestream(primary_scaled, NONZERO_DC_MACROBLOCK);
        let alpha = dc_only_luma_codestream(alpha_scaled, NONZERO_DC_MACROBLOCK);
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
