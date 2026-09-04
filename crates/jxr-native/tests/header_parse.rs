use jxr_native::{NativeError, parse_codestream_headers};

fn minimal_header() -> Vec<u8> {
    let mut bytes = b"WMPHOTO\0".to_vec();
    // RESERVED_B=1, HARD_TILING=0, RESERVED_C=1
    bytes.push(0x11);
    // No tiling, spatial mode, orientation 0, no index, overlap mode 0.
    bytes.push(0x00);
    // Short header, all other flags clear.
    bytes.push(0x80);
    // Y-only, 8-bit.
    bytes.push(0x01);
    bytes.extend_from_slice(&15_u16.to_be_bytes());
    bytes.extend_from_slice(&15_u16.to_be_bytes());
    let mut bit_position = bytes.len() * 8;
    // Y-only, unscaled, all bands.
    push_bits(&mut bytes, &mut bit_position, 0, 3);
    push_bits(&mut bytes, &mut bit_position, 0, 1);
    push_bits(&mut bytes, &mut bit_position, 0, 4);
    // Uniform DC, LP, and HP quantizers equal to one.
    push_bits(&mut bytes, &mut bit_position, 1, 1);
    push_bits(&mut bytes, &mut bit_position, 1, 8);
    push_bits(&mut bytes, &mut bit_position, 0, 1);
    push_bits(&mut bytes, &mut bit_position, 1, 1);
    push_bits(&mut bytes, &mut bit_position, 1, 8);
    push_bits(&mut bytes, &mut bit_position, 0, 1);
    push_bits(&mut bytes, &mut bit_position, 1, 1);
    push_bits(&mut bytes, &mut bit_position, 1, 8);
    bytes
}

fn push_bits(bytes: &mut Vec<u8>, bit_position: &mut usize, value: u64, count: u8) {
    for shift in (0..count).rev() {
        if *bit_position / 8 == bytes.len() {
            bytes.push(0);
        }
        let bit = ((value >> shift) & 1) as u8;
        bytes[*bit_position / 8] |= bit << (7 - (*bit_position % 8));
        *bit_position += 1;
    }
}

#[test]
fn parses_minimal_image_and_plane_headers() {
    let bytes = minimal_header();
    let parsed = parse_codestream_headers(&bytes).expect("valid header");

    assert_eq!(parsed.image.width, 16);
    assert_eq!(parsed.image.height, 16);
    assert_eq!(parsed.primary.components, 1);
    assert!(!parsed.image.flags.alpha_plane());
}

#[test]
fn rejects_bad_signature() {
    let mut bytes = minimal_header();
    bytes[0] = b'X';

    assert!(matches!(
        parse_codestream_headers(&bytes),
        Err(NativeError::InvalidSignature)
    ));
}

#[test]
fn rejects_reserved_overlap_mode() {
    let mut bytes = minimal_header();
    bytes[9] |= 0b0000_0011;

    assert!(matches!(
        parse_codestream_headers(&bytes),
        Err(NativeError::ReservedValue {
            field: "OVERLAP_MODE",
            value: 3
        })
    ));
}

#[test]
fn truncation_reports_bit_position() {
    let bytes = &minimal_header()[..10];
    let error = parse_codestream_headers(bytes).expect_err("truncated header");

    assert!(matches!(error, NativeError::Truncated { .. }));
}
