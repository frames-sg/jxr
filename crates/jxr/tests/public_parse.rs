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
    let mut bytes = b"WMPHOTO\0".to_vec();
    bytes.extend_from_slice(&[0x11, 0x00, 0x80, 0x01]);
    bytes.extend_from_slice(&15_u16.to_be_bytes());
    bytes.extend_from_slice(&15_u16.to_be_bytes());
    let mut bit_position = bytes.len() * 8;
    for (value, count) in [(0, 3), (0, 1), (3, 4), (1, 1), (1, 8)] {
        push_bits(&mut bytes, &mut bit_position, value, count);
    }
    bytes.extend_from_slice(&[0xFD, 0, 0, 1, 0x5a, 0, 0, 0]);
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
