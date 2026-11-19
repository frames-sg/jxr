mod common;

use jxr_native::{NativeError, parse_codestream_headers};

fn minimal_header() -> Vec<u8> {
    // No tiling, spatial mode, orientation 0, no index, overlap mode 0, a
    // short header, Y-only 8-bit output; a Y-only, unscaled plane of all
    // bands with uniform DC, LP and HP quantizers equal to one.
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
        &[],
    )
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
