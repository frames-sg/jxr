use jxr_core::Orientation;
use jxr_native::{AnnexAWriteOptions, NativeError, parse_annex_a, parse_codestream, write_annex_a};

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

fn annex_a_file(image_offset: u32, image_len: u32) -> Vec<u8> {
    const ENTRY_COUNT: u16 = 5;
    let mut bytes = vec![0_u8; 128];
    bytes[0..4].copy_from_slice(&[b'I', b'I', 0xBC, 1]);
    bytes[4..8].copy_from_slice(&8_u32.to_le_bytes());
    bytes[8..10].copy_from_slice(&ENTRY_COUNT.to_le_bytes());

    let entries = [
        (0xBC01_u16, 1_u16, 16_u32, 80_u32),
        (0xBC80, 4, 1, 16),
        (0xBC81, 4, 1, 16),
        (0xBCC0, 4, 1, image_offset),
        (0xBCC1, 4, 1, image_len),
    ];
    for (index, (tag, kind, count, value)) in entries.into_iter().enumerate() {
        let offset = 10 + index * 12;
        bytes[offset..offset + 2].copy_from_slice(&tag.to_le_bytes());
        bytes[offset + 2..offset + 4].copy_from_slice(&kind.to_le_bytes());
        bytes[offset + 4..offset + 8].copy_from_slice(&count.to_le_bytes());
        bytes[offset + 8..offset + 12].copy_from_slice(&value.to_le_bytes());
    }
    bytes[70..74].copy_from_slice(&0_u32.to_le_bytes());
    bytes
}

#[test]
fn parses_required_annex_a_entries() {
    let bytes = annex_a_file(96, 16);
    let image = parse_annex_a(&bytes).expect("valid Annex-A file");

    assert_eq!(image.width, 16);
    assert_eq!(image.height, 16);
    assert_eq!(image.codestream_range, 96..112);
    assert_eq!(image.pixel_format_guid, [0; 16]);
}

#[test]
fn rejects_codestream_range_outside_file() {
    let bytes = annex_a_file(120, 16);

    assert!(matches!(
        parse_annex_a(&bytes),
        Err(NativeError::RangeOutsideInput { .. })
    ));
}

#[test]
fn accepts_missing_final_pad_byte_in_even_rounded_image_count() {
    let bytes = annex_a_file(113, 16);
    let image = parse_annex_a(&bytes).expect("rounded count is bounded to the available bytes");
    assert_eq!(image.codestream_range, 113..128);
}

#[test]
fn zero_alpha_offset_and_count_mean_integrated_alpha() {
    let mut bytes = annex_a_file(112, 16);
    bytes[8..10].copy_from_slice(&7_u16.to_le_bytes());
    bytes[18..22].copy_from_slice(&96_u32.to_le_bytes());
    for (index, tag) in [0xBCC2_u16, 0xBCC3].into_iter().enumerate() {
        let offset = 70 + index * 12;
        bytes[offset..offset + 2].copy_from_slice(&tag.to_le_bytes());
        bytes[offset + 2..offset + 4].copy_from_slice(&4_u16.to_le_bytes());
        bytes[offset + 4..offset + 8].copy_from_slice(&1_u32.to_le_bytes());
        bytes[offset + 8..offset + 12].copy_from_slice(&0_u32.to_le_bytes());
    }

    let image = parse_annex_a(&bytes).expect("zero separate-alpha range is absent");
    assert_eq!(image.alpha_range, None);
}

#[test]
fn writes_parseable_annex_a_with_exact_payloads() {
    let primary = minimal_raw_codestream();
    let alpha = minimal_raw_codestream();
    let profile = [1, 3, 5, 7, 9];
    let guid = [0xa5; 16];
    let options = AnnexAWriteOptions::new(16, 16, guid)
        .with_orientation(Orientation::Rotate90)
        .with_resolution_dpi(300.0, 150.0)
        .with_icc_profile(&profile)
        .with_separate_alpha(&alpha);

    let bytes = write_annex_a(&primary, &options).expect("valid Annex-A output");
    let image = parse_annex_a(&bytes).expect("writer output must parse");

    assert_eq!(image.width, 16);
    assert_eq!(image.height, 16);
    assert_eq!(image.pixel_format_guid, guid);
    assert_eq!(image.metadata.transformation, Some(4));
    assert_eq!(
        image.metadata.resolution_dpi_bits,
        Some([300.0_f32.to_bits(), 150.0_f32.to_bits()])
    );
    assert_eq!(&bytes[image.codestream_range.clone()], primary);
    assert_eq!(&bytes[image.alpha_range.clone().unwrap()], alpha);
    assert_eq!(
        &bytes[image.metadata.icc_profile_range.clone().unwrap()],
        profile
    );
    assert_eq!(image.codestream_range.start % 4, 0);
    assert_eq!(image.alpha_range.unwrap().start % 4, 0);

    let parsed = parse_codestream(&bytes).expect("written container must pass full validation");
    assert!(parsed.separate_alpha_headers.is_some());
}

#[test]
fn writer_rejects_nested_or_inconsistent_codestreams() {
    let primary = minimal_raw_codestream();
    let options = AnnexAWriteOptions::new(15, 16, [0xa5; 16]);
    assert!(matches!(
        write_annex_a(&primary, &options),
        Err(NativeError::InvalidSyntax {
            field: "Annex-A/codestream dimension mismatch"
        })
    ));

    let options = AnnexAWriteOptions::new(16, 16, [0xa5; 16]);
    let container = write_annex_a(&primary, &options).unwrap();
    assert!(matches!(
        write_annex_a(&container, &options),
        Err(NativeError::InvalidSyntax {
            field: "Annex-A writer requires a raw codestream"
        })
    ));
}

#[test]
fn parser_rejects_non_float_resolution_entries() {
    let primary = minimal_raw_codestream();
    let options = AnnexAWriteOptions::new(16, 16, [0xa5; 16]);
    let mut bytes = write_annex_a(&primary, &options).unwrap();
    // With no ICC entry, vertical resolution is the sixth sorted IFD entry.
    bytes[72..74].copy_from_slice(&4_u16.to_le_bytes());

    assert!(matches!(
        parse_annex_a(&bytes),
        Err(NativeError::InvalidAnnexAEntry {
            tag: 0xBC83,
            element_type: 4,
            count: 1
        })
    ));
}
