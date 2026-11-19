//! Annex-A pixel-format GUID classification and codestream consistency checks.

use jxr_core::{
    AnnexABitDepth, AnnexAChannelOrder, AnnexAPixelFamily, AnnexAPixelFormat,
    AnnexAPixelFormatDescriptor, ChannelLayout, ChromaSampling, OutputBitDepth, PixelFormat,
};

use crate::{NativeError, ParsedCodestream};

mod table;

use table::{WIC_GUIDS, known_descriptor, wic_descriptor};

pub(crate) const GUID_PREFIX: [u8; 15] = [
    0x24, 0xc3, 0xdd, 0x6f, 0x03, 0x4e, 0xfe, 0x4b, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9,
];

/// Classify a raw Annex-A pixel-format identifier from T.832 Table A.6 or WIC.
#[must_use]
pub fn classify_annex_a_pixel_format(guid: [u8; 16]) -> AnnexAPixelFormat {
    let descriptor = if guid[..15] == GUID_PREFIX {
        known_descriptor(guid[15])
    } else {
        wic_descriptor(guid)
    };
    descriptor.map_or(AnnexAPixelFormat::Unknown(guid), AnnexAPixelFormat::Known)
}

/// The pixel format a container format's samples are stored in: the layout
/// a decoder returns when asked for the image as its container declares it.
///
/// Returns `None` for descriptors whose channel count cannot form a layout.
#[must_use]
pub fn native_pixel_format(descriptor: &AnnexAPixelFormatDescriptor) -> Option<PixelFormat> {
    let alpha = descriptor.alpha;
    let layout = match descriptor.family {
        AnnexAPixelFamily::Luma if alpha => ChannelLayout::LumaAlpha,
        AnnexAPixelFamily::Luma => ChannelLayout::Luma,
        AnnexAPixelFamily::Rgb | AnnexAPixelFamily::Rgbe => match (descriptor.order, alpha) {
            (AnnexAChannelOrder::Bgra, true) => ChannelLayout::Bgra,
            (_, true) => ChannelLayout::Rgba,
            (AnnexAChannelOrder::Rgbx, false) => ChannelLayout::Rgbx,
            (AnnexAChannelOrder::Bgrx, false) => ChannelLayout::Bgrx,
            (AnnexAChannelOrder::Bgr, false) => ChannelLayout::Bgr,
            (_, false) => ChannelLayout::Rgb,
        },
        AnnexAPixelFamily::Cmyk { .. } if alpha => ChannelLayout::Cmyka,
        AnnexAPixelFamily::Cmyk { .. } => ChannelLayout::Cmyk,
        AnnexAPixelFamily::Yuv(sampling) if alpha => ChannelLayout::Yuva(sampling),
        AnnexAPixelFamily::Yuv(sampling) => ChannelLayout::Yuv(sampling),
        AnnexAPixelFamily::NComponent if alpha => {
            ChannelLayout::NComponentAlpha(u16::from(descriptor.channels.checked_sub(1)?))
        }
        AnnexAPixelFamily::NComponent => ChannelLayout::NComponent(u16::from(descriptor.channels)),
    };
    Some(match descriptor.bit_depth {
        AnnexABitDepth::Bit1 => PixelFormat::BitPacked(layout),
        AnnexABitDepth::U8 if descriptor.family == AnnexAPixelFamily::Rgbe => PixelFormat::Rgbe,
        AnnexABitDepth::U8 => PixelFormat::U8(layout),
        AnnexABitDepth::U10 | AnnexABitDepth::U16 => PixelFormat::U16(layout),
        AnnexABitDepth::I16 => PixelFormat::I16(layout),
        AnnexABitDepth::F16 => PixelFormat::F16(layout),
        AnnexABitDepth::I32 => PixelFormat::I32(layout),
        AnnexABitDepth::F32 => PixelFormat::F32(layout),
        AnnexABitDepth::Rgb555 => PixelFormat::Rgb555,
        AnnexABitDepth::Rgb565 => PixelFormat::Rgb565,
        AnnexABitDepth::Rgb101010 => PixelFormat::Rgb101010,
    })
}

/// What a container format declares beyond the storage of its samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct FormatDeclaration {
    /// Declare a CMYK storage `CMYKDIRECT`: both declarations exist for each.
    pub(crate) cmyk_direct: bool,
    /// Declare the colour samples premultiplied by alpha.
    pub(crate) premultiplied_alpha: bool,
    /// Significant bits of samples that do not fill their storage: `None`
    /// declares the storage's own depth. Ten is the one narrower depth the
    /// table has, for samples in 16-bit storage.
    pub(crate) sample_bits: Option<u8>,
}

impl FormatDeclaration {
    /// Whether a format of `depth` declares samples as wide as this says.
    fn declares(self, depth: AnnexABitDepth) -> bool {
        match self.sample_bits {
            None => depth != AnnexABitDepth::U10,
            Some(10) => depth == AnnexABitDepth::U10,
            Some(_) => false,
        }
    }
}

/// The container format whose samples are stored as `format` and declared
/// as `declaration`, with its GUID: the inverse of [`native_pixel_format`].
///
/// Table A.6 is searched first, where the lowest code wins among formats
/// sharing a storage; the WIC formats follow, so a file uses one only when
/// the standard has no code for its samples.
pub(crate) fn annex_a_format_for(
    format: PixelFormat,
    declaration: FormatDeclaration,
) -> Option<([u8; 16], AnnexAPixelFormatDescriptor)> {
    let standard = (0..=u8::MAX).map(|code| {
        let mut guid = [0_u8; 16];
        guid[..15].copy_from_slice(&GUID_PREFIX);
        guid[15] = code;
        (guid, known_descriptor(code))
    });
    let wic = WIC_GUIDS
        .into_iter()
        .map(|guid| (guid, wic_descriptor(guid)));
    standard.chain(wic).find_map(|(guid, descriptor)| {
        let descriptor = descriptor?;
        let other_cmyk = matches!(
            descriptor.family,
            AnnexAPixelFamily::Cmyk { direct } if direct != declaration.cmyk_direct
        );
        (descriptor.premultiplied_alpha == declaration.premultiplied_alpha
            && !other_cmyk
            // Ten-bit samples share 16-bit storage but are a different depth.
            && declaration.declares(descriptor.bit_depth)
            && native_pixel_format(&descriptor) == Some(format))
        .then_some((guid, descriptor))
    })
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

pub(crate) const fn output_color_code(family: AnnexAPixelFamily) -> u8 {
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

/// Whether `OUTPUT_BITDEPTH` code `code` codes samples of the Annex-A depth
/// `depth`: whether both are of one sample format, which either one-bit
/// depth and either reading of the ten-bit code share.
fn bit_depth_matches(depth: AnnexABitDepth, code: u8) -> bool {
    crate::header::output_bit_depth(code).is_some_and(|coded| {
        coded.sample_format() == OutputBitDepth::from_annex_a(depth, 0, 0, 0).sample_format()
    })
}

#[cfg(test)]
mod tests {
    use jxr_core::{
        AnnexABitDepth, AnnexAChannelOrder, AnnexANumericKind, AnnexAPixelFamily, AnnexAPixelFormat,
    };

    use super::{
        FormatDeclaration, GUID_PREFIX, annex_a_format_for, classify_annex_a_pixel_format,
        native_pixel_format, premultiplication_is_consistent,
    };

    const STRAIGHT: FormatDeclaration = FormatDeclaration {
        cmyk_direct: false,
        premultiplied_alpha: false,
        sample_bits: None,
    };
    const PREMULTIPLIED: FormatDeclaration = FormatDeclaration {
        cmyk_direct: false,
        premultiplied_alpha: true,
        sample_bits: None,
    };

    #[test]
    fn container_depths_match_exactly_the_codes_that_code_them() {
        use super::bit_depth_matches;
        // Every depth and the `OUTPUT_BITDEPTH` codes T.832 Table A.6 pairs
        // with it: both one-bit codes, and the ten-bit code for both readings.
        let pairs: [(AnnexABitDepth, &[u8]); 11] = [
            (AnnexABitDepth::Bit1, &[0, 15]),
            (AnnexABitDepth::U8, &[1]),
            (AnnexABitDepth::U16, &[2]),
            (AnnexABitDepth::I16, &[3]),
            (AnnexABitDepth::F16, &[4]),
            (AnnexABitDepth::I32, &[6]),
            (AnnexABitDepth::F32, &[7]),
            (AnnexABitDepth::Rgb555, &[8]),
            (AnnexABitDepth::U10, &[9]),
            (AnnexABitDepth::Rgb101010, &[9]),
            (AnnexABitDepth::Rgb565, &[10]),
        ];
        for (depth, codes) in pairs {
            for code in 0..16 {
                assert_eq!(
                    bit_depth_matches(depth, code),
                    codes.contains(&code),
                    "{depth:?} code {code}"
                );
            }
        }
    }

    #[test]
    fn container_format_lookup_inverts_the_native_pixel_format() {
        use jxr_core::{ChannelLayout, PixelFormat};

        for (format, code) in [
            (PixelFormat::BitPacked(ChannelLayout::Luma), 0x05),
            (PixelFormat::U8(ChannelLayout::Luma), 0x08),
            (PixelFormat::U8(ChannelLayout::Bgr), 0x0c),
            (PixelFormat::U8(ChannelLayout::Rgb), 0x0d),
            (PixelFormat::U8(ChannelLayout::Bgra), 0x0f),
            (PixelFormat::U16(ChannelLayout::Luma), 0x0b),
            (PixelFormat::U16(ChannelLayout::Rgb), 0x15),
            (PixelFormat::I16(ChannelLayout::Luma), 0x13),
            (PixelFormat::F16(ChannelLayout::Luma), 0x3e),
            (PixelFormat::I32(ChannelLayout::Luma), 0x3f),
            (PixelFormat::F32(ChannelLayout::Luma), 0x11),
            (PixelFormat::Rgb565, 0x0a),
            (PixelFormat::Rgbe, 0x3d),
            (PixelFormat::U8(ChannelLayout::Cmyk), 0x1c),
            (PixelFormat::U16(ChannelLayout::Cmyk), 0x1f),
            (PixelFormat::U8(ChannelLayout::NComponent(3)), 0x20),
            (PixelFormat::U8(ChannelLayout::NComponent(8)), 0x25),
            (PixelFormat::U16(ChannelLayout::NComponent(5)), 0x28),
            (PixelFormat::U16(ChannelLayout::NComponentAlpha(3)), 0x34),
        ] {
            let (guid, descriptor) = annex_a_format_for(format, STRAIGHT).unwrap();
            assert_eq!(guid[15], code, "{format:?}");
            assert_eq!(native_pixel_format(&descriptor), Some(format));
            assert_eq!(
                classify_annex_a_pixel_format(guid),
                AnnexAPixelFormat::Known(descriptor)
            );
        }
        for absent in [
            PixelFormat::I16(ChannelLayout::Bgr),
            PixelFormat::U8(ChannelLayout::NComponent(2)),
            PixelFormat::U8(ChannelLayout::NComponent(9)),
            PixelFormat::U8(ChannelLayout::LumaAlpha),
        ] {
            assert!(annex_a_format_for(absent, STRAIGHT).is_none(), "{absent:?}");
        }
    }

    #[test]
    fn premultiplied_storage_resolves_to_its_own_declaration() {
        use jxr_core::{ChannelLayout, PixelFormat};

        for (format, straight, premultiplied) in [
            (PixelFormat::U8(ChannelLayout::Bgra), 0x0f, Some(0x10)),
            (PixelFormat::U16(ChannelLayout::Rgba), 0x16, Some(0x17)),
            (PixelFormat::F32(ChannelLayout::Rgba), 0x19, Some(0x1a)),
            (PixelFormat::I16(ChannelLayout::Rgba), 0x1d, None),
            (PixelFormat::F16(ChannelLayout::Rgba), 0x3a, None),
            (PixelFormat::I32(ChannelLayout::Rgba), 0x1e, None),
            (PixelFormat::U8(ChannelLayout::Cmyka), 0x2c, None),
        ] {
            let (guid, descriptor) = annex_a_format_for(format, STRAIGHT).unwrap();
            assert_eq!(guid[15], straight, "{format:?}");
            assert!(descriptor.alpha && !descriptor.premultiplied_alpha);
            let declared = annex_a_format_for(format, PREMULTIPLIED);
            assert_eq!(
                declared.map(|(guid, _)| guid[15]),
                premultiplied,
                "{format:?}"
            );
            if let Some((guid, descriptor)) = declared {
                assert!(descriptor.premultiplied_alpha);
                assert_eq!(native_pixel_format(&descriptor), Some(format));
                assert_eq!(
                    classify_annex_a_pixel_format(guid),
                    AnnexAPixelFormat::Known(descriptor)
                );
            }
        }
        // A format without alpha has no premultiplied declaration.
        assert!(annex_a_format_for(PixelFormat::U8(ChannelLayout::Rgb), PREMULTIPLIED).is_none());
    }

    #[test]
    fn formats_table_a6_lacks_resolve_to_their_wic_identifiers() {
        use jxr_core::{ChannelLayout, PixelFormat};

        for (format, declaration, first_byte) in [
            (PixelFormat::U8(ChannelLayout::Rgba), STRAIGHT, 0x2d),
            (PixelFormat::U8(ChannelLayout::Rgba), PREMULTIPLIED, 0x50),
            (PixelFormat::F32(ChannelLayout::Rgb), STRAIGHT, 0x8f),
        ] {
            let (guid, descriptor) = annex_a_format_for(format, declaration).unwrap();
            assert_ne!(guid[..15], GUID_PREFIX, "{format:?}");
            assert_eq!(guid[0], first_byte, "{format:?}");
            assert_eq!(
                descriptor.premultiplied_alpha,
                declaration.premultiplied_alpha
            );
            assert_eq!(native_pixel_format(&descriptor), Some(format));
            assert_eq!(
                classify_annex_a_pixel_format(guid),
                AnnexAPixelFormat::Known(descriptor)
            );
        }
    }

    #[test]
    fn cmyk_storage_resolves_to_the_requested_declaration() {
        use jxr_core::{ChannelLayout, PixelFormat};

        for (format, direct, code) in [
            (PixelFormat::U8(ChannelLayout::Cmyk), false, 0x1c),
            (PixelFormat::U8(ChannelLayout::Cmyk), true, 0x54),
            (PixelFormat::U16(ChannelLayout::Cmyk), false, 0x1f),
            (PixelFormat::U16(ChannelLayout::Cmyk), true, 0x55),
            (PixelFormat::U8(ChannelLayout::Cmyka), true, 0x56),
        ] {
            let declaration = FormatDeclaration {
                cmyk_direct: direct,
                ..STRAIGHT
            };
            let (guid, descriptor) = annex_a_format_for(format, declaration).unwrap();
            assert_eq!(guid[15], code, "{format:?} direct {direct}");
            assert_eq!(descriptor.family, AnnexAPixelFamily::Cmyk { direct });
            assert_eq!(native_pixel_format(&descriptor), Some(format));
        }
        // The choice affects CMYK alone.
        let rgb = PixelFormat::U8(ChannelLayout::Rgb);
        let direct = FormatDeclaration {
            cmyk_direct: true,
            ..STRAIGHT
        };
        assert_eq!(
            annex_a_format_for(rgb, direct),
            annex_a_format_for(rgb, STRAIGHT)
        );
    }

    #[test]
    fn yuv_storage_resolves_to_the_format_of_its_sampling_and_depth() {
        use jxr_core::{ChannelLayout, ChromaSampling, PixelFormat};

        let ten_bit = FormatDeclaration {
            sample_bits: Some(10),
            ..STRAIGHT
        };
        let yuv = |sampling| ChannelLayout::Yuv(sampling);
        let yuva = |sampling| ChannelLayout::Yuva(sampling);
        // `(storage, declaration, Table A.6 code)` of every YUV format.
        for (format, declaration, code) in [
            (PixelFormat::U8(yuv(ChromaSampling::Cs420)), STRAIGHT, 0x44),
            (PixelFormat::U8(yuv(ChromaSampling::Cs422)), STRAIGHT, 0x45),
            (PixelFormat::U16(yuv(ChromaSampling::Cs422)), ten_bit, 0x46),
            (PixelFormat::U16(yuv(ChromaSampling::Cs422)), STRAIGHT, 0x47),
            (PixelFormat::U8(yuv(ChromaSampling::Cs444)), STRAIGHT, 0x48),
            (PixelFormat::U16(yuv(ChromaSampling::Cs444)), ten_bit, 0x49),
            (PixelFormat::U16(yuv(ChromaSampling::Cs444)), STRAIGHT, 0x4a),
            (PixelFormat::I16(yuv(ChromaSampling::Cs444)), STRAIGHT, 0x4b),
            (PixelFormat::U8(yuva(ChromaSampling::Cs420)), STRAIGHT, 0x4c),
            (PixelFormat::U8(yuva(ChromaSampling::Cs422)), STRAIGHT, 0x4d),
            (PixelFormat::U16(yuva(ChromaSampling::Cs422)), ten_bit, 0x4e),
            (
                PixelFormat::U16(yuva(ChromaSampling::Cs422)),
                STRAIGHT,
                0x4f,
            ),
            (PixelFormat::U8(yuva(ChromaSampling::Cs444)), STRAIGHT, 0x50),
            (PixelFormat::U16(yuva(ChromaSampling::Cs444)), ten_bit, 0x51),
            (
                PixelFormat::U16(yuva(ChromaSampling::Cs444)),
                STRAIGHT,
                0x52,
            ),
            (
                PixelFormat::I16(yuva(ChromaSampling::Cs444)),
                STRAIGHT,
                0x53,
            ),
        ] {
            let (guid, descriptor) = annex_a_format_for(format, declaration).unwrap();
            assert_eq!(guid[15], code, "{format:?} {declaration:?}");
            assert_eq!(native_pixel_format(&descriptor), Some(format));
            assert_eq!(
                descriptor.bit_depth == AnnexABitDepth::U10,
                declaration.sample_bits == Some(10)
            );
        }
        // Ten-bit samples are declared only where the table has such a
        // format, and no other narrow depth is declared at all.
        for (format, bits) in [
            (PixelFormat::U16(yuv(ChromaSampling::Cs420)), 10),
            (PixelFormat::U16(ChannelLayout::Rgb), 10),
            (PixelFormat::U16(ChannelLayout::Luma), 10),
            (PixelFormat::U16(yuv(ChromaSampling::Cs444)), 12),
            (PixelFormat::U16(yuv(ChromaSampling::Cs444)), 16),
        ] {
            let declaration = FormatDeclaration {
                sample_bits: Some(bits),
                ..STRAIGHT
            };
            assert!(
                annex_a_format_for(format, declaration).is_none(),
                "{format:?} at {bits} bits"
            );
        }
    }

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
