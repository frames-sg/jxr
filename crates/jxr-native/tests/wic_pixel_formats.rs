use jxr_core::{
    AlphaMode, AnnexABitDepth, AnnexAChannelOrder, AnnexANumericKind, AnnexAPixelFamily,
    AnnexAPixelFormat,
};
use jxr_native::{
    AnnexAWriteOptions, classify_annex_a_pixel_format, image_info, parse_codestream, write_annex_a,
};

// Little-endian GUID bytes from the Windows SDK's `wincodec.h`.
const RGBA: [u8; 16] = [
    0x2d, 0xad, 0xc7, 0xf5, 0x8d, 0x6a, 0xdd, 0x43, 0xa7, 0xa8, 0xa2, 0x99, 0x35, 0x26, 0x1a, 0xe9,
];
const PRGBA: [u8; 16] = [
    0x50, 0xa6, 0xc4, 0x3c, 0x27, 0xa5, 0x37, 0x4d, 0xa9, 0x16, 0x31, 0x42, 0xc7, 0xeb, 0xed, 0xba,
];
const RGB_FLOAT: [u8; 16] = [
    0x8f, 0xd7, 0xfe, 0xe3, 0xdb, 0xe8, 0xcf, 0x4a, 0x84, 0xc1, 0xe9, 0x7f, 0x61, 0x36, 0xb3, 0x27,
];

#[test]
fn classifies_wic_formats_outside_table_a6() {
    for (guid, alpha, premultiplied) in [
        (RGBA, true, false),
        (PRGBA, true, true),
        (RGB_FLOAT, false, false),
    ] {
        let AnnexAPixelFormat::Known(format) = classify_annex_a_pixel_format(guid) else {
            panic!("WIC format must classify: {guid:02x?}");
        };
        assert_eq!(format.family, AnnexAPixelFamily::Rgb);
        assert_eq!(format.channels, if alpha { 4 } else { 3 });
        assert_eq!(format.alpha, alpha);
        assert_eq!(format.premultiplied_alpha, premultiplied);
        assert_eq!(
            format.bit_depth,
            if alpha {
                AnnexABitDepth::U8
            } else {
                AnnexABitDepth::F32
            }
        );
        assert_eq!(
            format.numeric,
            if alpha {
                AnnexANumericKind::Unsigned
            } else {
                AnnexANumericKind::Float
            }
        );
        assert_eq!(
            format.order,
            if alpha {
                AnnexAChannelOrder::Rgba
            } else {
                AnnexAChannelOrder::Rgb
            }
        );

        let mut unknown = guid;
        unknown[0] ^= 1;
        assert_eq!(
            classify_annex_a_pixel_format(unknown),
            AnnexAPixelFormat::Unknown(unknown)
        );
    }
}

// One 16x16 DC-only tile. RGB uses YUV444 internally; alpha uses Y-only.
fn minimal_codestream(rgb: bool) -> Vec<u8> {
    let mut bytes = b"WMPHOTO\0".to_vec();
    bytes.extend_from_slice(&[0x11, 0x00, 0x80, if rgb { 0x71 } else { 0x01 }]);
    bytes.extend_from_slice(&15_u16.to_be_bytes());
    bytes.extend_from_slice(&15_u16.to_be_bytes());
    let fields = if rgb {
        vec![(3, 3), (0, 1), (3, 4), (0, 8), (1, 1), (0, 2), (1, 8)]
    } else {
        vec![(0, 3), (0, 1), (3, 4), (1, 1), (1, 8)]
    };
    let mut bit_position = bytes.len() * 8;
    for (value, count) in fields {
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

#[test]
fn separate_alpha_metadata_uses_the_wic_premultiplication_declaration() {
    let primary = minimal_codestream(true);
    let alpha = minimal_codestream(false);
    for (guid, premultiplied) in [(RGBA, false), (PRGBA, true)] {
        let options = AnnexAWriteOptions::new(16, 16, guid).with_separate_alpha(&alpha);
        let bytes = write_annex_a(&primary, &options).unwrap();
        let parsed = parse_codestream(&bytes).unwrap();
        assert!(!parsed.headers.image.flags.premultiplied_alpha());
        let info = image_info(&parsed).unwrap();
        assert_eq!(info.alpha_mode, AlphaMode::Separate);
        assert_eq!(info.premultiplied_alpha, premultiplied);
    }
}
