//! Annex-A pixel-format GUID classification and codestream consistency checks.

use jxr_core::{AnnexABitDepth, AnnexAPixelFamily, AnnexAPixelFormat, ChromaSampling};

use crate::{NativeError, ParsedCodestream};

mod table;

use table::known_descriptor;

const GUID_PREFIX: [u8; 15] = [
    0x24, 0xc3, 0xdd, 0x6f, 0x03, 0x4e, 0xfe, 0x4b, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9,
];

/// Classify a raw Annex-A pixel-format identifier using T.832 Table A.6.
#[must_use]
pub fn classify_annex_a_pixel_format(guid: [u8; 16]) -> AnnexAPixelFormat {
    if guid[..15] != GUID_PREFIX {
        return AnnexAPixelFormat::Unknown(guid);
    }
    known_descriptor(guid[15]).map_or(AnnexAPixelFormat::Unknown(guid), AnnexAPixelFormat::Known)
}

pub(crate) fn validate_annex_a_pixel_format(parsed: &ParsedCodestream) -> Result<(), NativeError> {
    let Some(annex) = &parsed.annex_a else {
        return Ok(());
    };
    let AnnexAPixelFormat::Known(format) = classify_annex_a_pixel_format(annex.pixel_format_guid)
    else {
        return Ok(());
    };
    let image = &parsed.headers.image;
    if image.output_color_format != output_color_code(format.family) {
        return Err(invalid("Annex-A pixel format colour declaration"));
    }
    if !bit_depth_matches(format.bit_depth, image.output_bit_depth) {
        return Err(invalid("Annex-A pixel format bit depth"));
    }
    let primary_components = u16::from(format.channels - u8::from(format.alpha));
    if parsed.headers.primary.components != primary_components {
        return Err(invalid("Annex-A pixel format component count"));
    }
    let integrated_alpha = parsed.headers.alpha.as_ref();
    let separate_alpha = parsed
        .separate_alpha_headers
        .as_ref()
        .map(|headers| headers.alpha.as_ref().unwrap_or(&headers.primary));
    let actual_alpha = integrated_alpha.is_some() || separate_alpha.is_some();
    if actual_alpha != format.alpha {
        return Err(invalid("Annex-A pixel format alpha declaration"));
    }
    if !premultiplication_is_consistent(
        parsed.separate_alpha_headers.is_some(),
        image.flags.premultiplied_alpha(),
        format.premultiplied_alpha,
    ) {
        return Err(invalid("Annex-A pixel format premultiplied alpha"));
    }
    if integrated_alpha
        .into_iter()
        .chain(separate_alpha)
        .any(|plane| plane.components != 1)
    {
        return Err(invalid("Annex-A alpha component count"));
    }
    if let Some(headers) = &parsed.separate_alpha_headers
        && !bit_depth_matches(format.bit_depth, headers.image.output_bit_depth)
    {
        return Err(invalid("Annex-A separate alpha bit depth"));
    }
    Ok(())
}

const fn premultiplication_is_consistent(
    separate_alpha: bool,
    codestream_premultiplied: bool,
    annex_a_premultiplied: bool,
) -> bool {
    separate_alpha || codestream_premultiplied == annex_a_premultiplied
}

pub(crate) fn source_is_premultiplied(parsed: &ParsedCodestream) -> bool {
    if parsed.separate_alpha_headers.is_some()
        && let Some(annex) = &parsed.annex_a
        && let AnnexAPixelFormat::Known(format) =
            classify_annex_a_pixel_format(annex.pixel_format_guid)
    {
        return format.premultiplied_alpha;
    }
    parsed.headers.image.flags.premultiplied_alpha()
}

const fn invalid(field: &'static str) -> NativeError {
    NativeError::InvalidSyntax { field }
}

const fn output_color_code(family: AnnexAPixelFamily) -> u8 {
    match family {
        AnnexAPixelFamily::Luma => 0,
        AnnexAPixelFamily::Yuv(ChromaSampling::Cs420) => 1,
        AnnexAPixelFamily::Yuv(ChromaSampling::Cs422) => 2,
        AnnexAPixelFamily::Yuv(ChromaSampling::Cs444) => 3,
        AnnexAPixelFamily::Cmyk { direct: false } => 4,
        AnnexAPixelFamily::Cmyk { direct: true } => 5,
        AnnexAPixelFamily::NComponent => 6,
        AnnexAPixelFamily::Rgb => 7,
        AnnexAPixelFamily::Rgbe => 8,
    }
}

const fn bit_depth_matches(depth: AnnexABitDepth, code: u8) -> bool {
    match depth {
        AnnexABitDepth::Bit1 => code == 0 || code == 15,
        AnnexABitDepth::U8 => code == 1,
        AnnexABitDepth::U10 | AnnexABitDepth::Rgb101010 => code == 9,
        AnnexABitDepth::U16 => code == 2,
        AnnexABitDepth::I16 => code == 3,
        AnnexABitDepth::F16 => code == 4,
        AnnexABitDepth::I32 => code == 6,
        AnnexABitDepth::F32 => code == 7,
        AnnexABitDepth::Rgb555 => code == 8,
        AnnexABitDepth::Rgb565 => code == 10,
    }
}

#[cfg(test)]
mod tests {
    use jxr_core::{
        AnnexABitDepth, AnnexAChannelOrder, AnnexANumericKind, AnnexAPixelFamily, AnnexAPixelFormat,
    };

    use super::{GUID_PREFIX, classify_annex_a_pixel_format, premultiplication_is_consistent};

    fn guid(code: u8) -> [u8; 16] {
        let mut value = [0; 16];
        value[..15].copy_from_slice(&GUID_PREFIX);
        value[15] = code;
        value
    }

    #[test]
    fn classifies_representative_standard_families() {
        let cases = [
            (0x08, AnnexAPixelFamily::Luma),
            (0x0c, AnnexAPixelFamily::Rgb),
            (0x3d, AnnexAPixelFamily::Rgbe),
            (0x1c, AnnexAPixelFamily::Cmyk { direct: false }),
            (0x54, AnnexAPixelFamily::Cmyk { direct: true }),
            (0x20, AnnexAPixelFamily::NComponent),
        ];
        for (code, family) in cases {
            let AnnexAPixelFormat::Known(format) = classify_annex_a_pixel_format(guid(code)) else {
                panic!("standard format must classify");
            };
            assert_eq!(format.family, family);
        }
    }

    #[test]
    fn classifies_fixed_float_packed_and_alpha_properties() {
        let AnnexAPixelFormat::Known(fixed) = classify_annex_a_pixel_format(guid(0x12)) else {
            panic!()
        };
        assert_eq!(fixed.numeric, AnnexANumericKind::FixedPoint);
        let AnnexAPixelFormat::Known(float) = classify_annex_a_pixel_format(guid(0x1b)) else {
            panic!()
        };
        assert_eq!(float.numeric, AnnexANumericKind::Float);
        let AnnexAPixelFormat::Known(packed) = classify_annex_a_pixel_format(guid(0x0a)) else {
            panic!()
        };
        assert_eq!(packed.bit_depth, AnnexABitDepth::Rgb565);
        let AnnexAPixelFormat::Known(alpha) = classify_annex_a_pixel_format(guid(0x10)) else {
            panic!()
        };
        assert!(alpha.alpha && alpha.premultiplied_alpha);
    }

    #[test]
    fn classifies_annex_a_rgb_padding_explicitly() {
        for code in [0x1b, 0x40, 0x41, 0x42] {
            let AnnexAPixelFormat::Known(format) = classify_annex_a_pixel_format(guid(code)) else {
                panic!()
            };
            assert_eq!(format.order, AnnexAChannelOrder::Rgbx);
        }
        for code in [0x12, 0x18, 0x3b] {
            let AnnexAPixelFormat::Known(format) = classify_annex_a_pixel_format(guid(code)) else {
                panic!()
            };
            assert_eq!(format.order, AnnexAChannelOrder::Rgb);
        }
        let AnnexAPixelFormat::Known(padded_bgr) = classify_annex_a_pixel_format(guid(0x0e)) else {
            panic!()
        };
        assert_eq!(padded_bgr.order, AnnexAChannelOrder::Bgrx);
        let AnnexAPixelFormat::Known(bgr) = classify_annex_a_pixel_format(guid(0x0c)) else {
            panic!()
        };
        assert_eq!(bgr.order, AnnexAChannelOrder::Bgr);
    }

    #[test]
    fn retains_unknown_guid_bytes() {
        let unknown = [0xa5; 16];
        assert_eq!(
            classify_annex_a_pixel_format(unknown),
            AnnexAPixelFormat::Unknown(unknown)
        );
        assert_eq!(
            classify_annex_a_pixel_format(guid(0xff)),
            AnnexAPixelFormat::Unknown(guid(0xff))
        );
    }

    #[test]
    fn classifies_every_table_a6_discriminator() {
        for code in core::iter::once(0x05).chain(0x08..=0x3b).chain(0x3d..=0x56) {
            assert!(
                matches!(
                    classify_annex_a_pixel_format(guid(code)),
                    AnnexAPixelFormat::Known(_)
                ),
                "missing Table A.6 discriminator {code:#04x}"
            );
        }
    }

    #[test]
    fn separate_alpha_uses_the_combined_annex_a_premultiplication_declaration() {
        assert!(premultiplication_is_consistent(true, false, true));
        assert!(premultiplication_is_consistent(true, true, false));
        assert!(!premultiplication_is_consistent(false, false, true));
    }
}
