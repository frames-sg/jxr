use jxr_core::{
    ChannelLayout, ChromaSampling, ColorFormat, DecodedSamples, DecodedSamplesMut, PixelFormat,
};

use super::{
    AlphaFormatRequest, ComponentPlane, OutputBitDepth, OutputFormatError, OutputFormatRequest,
    format_components, format_components_into_with_cpu, format_components_with_cpu,
    format_planar_yuv, format_planar_yuv_into,
};
use crate::CpuCapabilities;
use crate::reconstruct::CropWindow;

fn request(
    internal_color: ColorFormat,
    output_color: ColorFormat,
    bit_depth: OutputBitDepth,
    pixel_format: PixelFormat,
) -> OutputFormatRequest {
    OutputFormatRequest {
        internal_color,
        output_color,
        bit_depth,
        pixel_format,
        scaled: false,
        alpha_format: None,
        red_blue_not_swapped: true,
        premultiply_alpha: false,
        crop: CropWindow {
            x: 0,
            y: 0,
            width: 2,
            height: 1,
        },
    }
}

#[test]
fn yuv420_output_keeps_native_plane_geometry() {
    let y = [-128, -127, -126, -125, -124, -123, -122, -121];
    let u = [0, 1];
    let v = [2, 3];
    let planes = [
        ComponentPlane::tightly_packed(4, 2, &y),
        ComponentPlane::tightly_packed(2, 1, &u),
        ComponentPlane::tightly_packed(2, 1, &v),
    ];
    let mut output = request(
        ColorFormat::Yuv(ChromaSampling::Cs420),
        ColorFormat::Yuv(ChromaSampling::Cs420),
        OutputBitDepth::U8,
        PixelFormat::U8(ChannelLayout::Yuv(ChromaSampling::Cs420)),
    );
    output.crop.width = 4;
    output.crop.height = 2;
    let decoded = format_planar_yuv(&planes, None, output, CpuCapabilities::detect()).unwrap();
    assert_eq!(
        decoded.samples,
        DecodedSamples::U8(vec![0, 1, 2, 3, 4, 5, 6, 7, 128, 129, 130, 131])
    );
    assert_eq!(decoded.planes.len(), 3);
    assert_eq!(
        (decoded.planes[0].byte_offset, decoded.planes[0].width),
        (0, 4)
    );
    assert_eq!(
        (decoded.planes[1].byte_offset, decoded.planes[1].width),
        (8, 2)
    );
    assert_eq!(
        (decoded.planes[2].byte_offset, decoded.planes[2].width),
        (10, 2)
    );
    assert!(decoded.planes.iter().all(|plane| plane.channels == 1));
}

#[test]
fn yuv422_alpha_is_a_full_resolution_fourth_plane() {
    let y = [0; 8];
    let u = [0; 4];
    let v = [0; 4];
    let alpha = [-128, -96, -64, -32, 0, 32, 64, 127];
    let planes = [
        ComponentPlane::tightly_packed(4, 2, &y),
        ComponentPlane::tightly_packed(2, 2, &u),
        ComponentPlane::tightly_packed(2, 2, &v),
    ];
    let mut output = request(
        ColorFormat::Yuv(ChromaSampling::Cs422),
        ColorFormat::Yuv(ChromaSampling::Cs422),
        OutputBitDepth::U8,
        PixelFormat::U8(ChannelLayout::Yuva(ChromaSampling::Cs422)),
    );
    output.crop.width = 4;
    output.crop.height = 2;
    let decoded = format_planar_yuv(
        &planes,
        Some(ComponentPlane::tightly_packed(4, 2, &alpha)),
        output,
        CpuCapabilities::detect(),
    )
    .unwrap();
    assert_eq!(decoded.planes.len(), 4);
    assert_eq!(
        decoded
            .planes
            .iter()
            .map(|plane| (plane.byte_offset, plane.width, plane.height))
            .collect::<Vec<_>>(),
        [(0, 4, 2), (8, 2, 2), (12, 2, 2), (16, 4, 2)]
    );
    let DecodedSamples::U8(samples) = decoded.samples else {
        panic!("expected u8 planar output")
    };
    assert_eq!(&samples[16..], &[0, 32, 64, 96, 128, 160, 192, 255]);
}

#[test]
fn yuv422_u10_uses_one_typed_word_per_planar_sample() {
    let y = [-512, 511];
    let u = [0];
    let v = [600];
    let planes = [
        ComponentPlane::tightly_packed(2, 1, &y),
        ComponentPlane::tightly_packed(1, 1, &u),
        ComponentPlane::tightly_packed(1, 1, &v),
    ];
    let output = request(
        ColorFormat::Yuv(ChromaSampling::Cs422),
        ColorFormat::Yuv(ChromaSampling::Cs422),
        OutputBitDepth::U10,
        PixelFormat::U16(ChannelLayout::Yuv(ChromaSampling::Cs422)),
    );
    let decoded = format_planar_yuv(&planes, None, output, CpuCapabilities::detect()).unwrap();
    assert_eq!(
        decoded.samples,
        DecodedSamples::U16(vec![0, 1023, 512, 1023])
    );
    assert_eq!(decoded.planes[1].byte_offset, 4);
    assert_eq!(decoded.planes[2].byte_offset, 6);
}

#[test]
fn direct_planar_destination_matches_owned_typed_packing() {
    let y = [-512, 511];
    let u = [0];
    let v = [600];
    let planes = [
        ComponentPlane::tightly_packed(2, 1, &y),
        ComponentPlane::tightly_packed(1, 1, &u),
        ComponentPlane::tightly_packed(1, 1, &v),
    ];
    let output = request(
        ColorFormat::Yuv(ChromaSampling::Cs422),
        ColorFormat::Yuv(ChromaSampling::Cs422),
        OutputBitDepth::U10,
        PixelFormat::U16(ChannelLayout::Yuv(ChromaSampling::Cs422)),
    );
    let expected = format_planar_yuv(&planes, None, output, CpuCapabilities::detect()).unwrap();
    let mut direct = [0_u16; 4];
    let actual = format_planar_yuv_into(
        &planes,
        None,
        output,
        CpuCapabilities::detect(),
        DecodedSamplesMut::U16(&mut direct),
    )
    .unwrap();
    assert_eq!(expected.samples, DecodedSamples::U16(direct.to_vec()));
    assert_eq!(expected.planes, actual.planes);
}

#[test]
fn luma_u8_biases_and_clips() {
    let samples = [-129, 127];
    let planes = [ComponentPlane::tightly_packed(2, 1, &samples)];
    let decoded = format_components(
        &planes,
        None,
        request(
            ColorFormat::Luma,
            ColorFormat::Luma,
            OutputBitDepth::U8,
            PixelFormat::U8(ChannelLayout::Luma),
        ),
    )
    .unwrap();
    assert_eq!(decoded, DecodedSamples::U8(vec![0, 255]));
}

#[test]
fn session_selected_u8_path_is_bit_identical_to_scalar_formula() {
    let samples: Vec<_> = (-80..80).map(|value| value * 11).collect();
    let planes = [ComponentPlane::tightly_packed(160, 1, &samples)];
    let mut output = request(
        ColorFormat::Luma,
        ColorFormat::Luma,
        OutputBitDepth::U8,
        PixelFormat::U8(ChannelLayout::Luma),
    );
    output.crop.width = 160;
    let (decoded, _) =
        format_components_with_cpu(&planes, None, output, CpuCapabilities::detect()).unwrap();
    assert_eq!(
        decoded,
        DecodedSamples::U8(
            samples
                .iter()
                .map(|&sample| {
                    u8::try_from((sample + 128).clamp(0, 255)).expect("sample is clipped to u8")
                })
                .collect()
        )
    );
}

#[test]
fn yuv444_to_rgba_formats_alpha_independently() {
    let y = [0, 1];
    let u = [0, 0];
    let v = [0, 0];
    let a = [-128, 127];
    let planes = [
        ComponentPlane::tightly_packed(2, 1, &y),
        ComponentPlane::tightly_packed(2, 1, &u),
        ComponentPlane::tightly_packed(2, 1, &v),
    ];
    let decoded = format_components(
        &planes,
        Some(ComponentPlane::tightly_packed(2, 1, &a)),
        request(
            ColorFormat::Yuv(ChromaSampling::Cs444),
            ColorFormat::Rgb,
            OutputBitDepth::U8,
            PixelFormat::U8(ChannelLayout::Rgba),
        ),
    )
    .unwrap();
    assert_eq!(
        decoded,
        DecodedSamples::U8(vec![128, 128, 128, 0, 129, 129, 129, 255])
    );
}

#[test]
fn alpha_uses_its_plane_local_scaled_flag() {
    let color = [0];
    let alpha = [8];
    let planes = [ComponentPlane::tightly_packed(1, 1, &color)];
    let mut output = request(
        ColorFormat::Luma,
        ColorFormat::Rgb,
        OutputBitDepth::U8,
        PixelFormat::U8(ChannelLayout::Rgba),
    );
    output.crop.width = 1;
    output.alpha_format = Some(AlphaFormatRequest {
        bit_depth: OutputBitDepth::U8,
        scaled: true,
    });
    let decoded = format_components(
        &planes,
        Some(ComponentPlane::tightly_packed(1, 1, &alpha)),
        output,
    )
    .unwrap();
    assert_eq!(decoded, DecodedSamples::U8(vec![128, 128, 128, 129]));
}

#[test]
fn packed_rgb_matches_normative_bit_positions() {
    let y = [-17, 15];
    let planes = [ComponentPlane::tightly_packed(2, 1, &y)];
    let decoded = format_components(
        &planes,
        None,
        request(
            ColorFormat::Luma,
            ColorFormat::Rgb,
            OutputBitDepth::Rgb555,
            PixelFormat::Rgb555,
        ),
    )
    .unwrap();
    assert_eq!(decoded, DecodedSamples::Rgb555(vec![0, 0x7fff]));
}

#[test]
fn packed_rgb_places_blue_in_the_low_field_and_red_in_the_high_field() {
    let y = [10, 10];
    let u = [3, 3];
    let v = [5, 5];
    let planes = [
        ComponentPlane::tightly_packed(2, 1, &y),
        ComponentPlane::tightly_packed(2, 1, &u),
        ComponentPlane::tightly_packed(2, 1, &v),
    ];
    let decoded = format_components(
        &planes,
        None,
        request(
            ColorFormat::Yuv(ChromaSampling::Cs444),
            ColorFormat::Rgb,
            OutputBitDepth::Rgb555,
            PixelFormat::Rgb555,
        ),
    )
    .unwrap();
    assert_eq!(decoded, DecodedSamples::Rgb555(vec![0x5b9b; 2]));
}

#[test]
fn luma_typed_outputs_apply_exact_postscaling() {
    let samples = [-1, 1];
    let planes = [ComponentPlane::tightly_packed(2, 1, &samples)];
    let cases = [
        (
            OutputBitDepth::U16 { shift_bits: 1 },
            PixelFormat::U16(ChannelLayout::Luma),
            DecodedSamples::U16(vec![32_766, 32_770]),
        ),
        (
            OutputBitDepth::I16 { shift_bits: 2 },
            PixelFormat::I16(ChannelLayout::Luma),
            DecodedSamples::I16(vec![-4, 4]),
        ),
        (
            OutputBitDepth::I32 { shift_bits: 3 },
            PixelFormat::I32(ChannelLayout::Luma),
            DecodedSamples::I32(vec![-8, 8]),
        ),
        (
            OutputBitDepth::F16,
            PixelFormat::F16(ChannelLayout::Luma),
            DecodedSamples::F16(vec![0x8001, 1]),
        ),
    ];
    for (depth, format, expected) in cases {
        let decoded = format_components(
            &planes,
            None,
            request(ColorFormat::Luma, ColorFormat::Luma, depth, format),
        )
        .unwrap();
        assert_eq!(decoded, expected);
    }

    let float_samples = [-(1 << 4), 1 << 4];
    let float_planes = [ComponentPlane::tightly_packed(2, 1, &float_samples)];
    let decoded = format_components(
        &float_planes,
        None,
        request(
            ColorFormat::Luma,
            ColorFormat::Luma,
            OutputBitDepth::F32 {
                mantissa_length: 4,
                exponent_bias: 1,
            },
            PixelFormat::F32(ChannelLayout::Luma),
        ),
    )
    .unwrap();
    let DecodedSamples::F32(values) = decoded else {
        panic!("expected f32 output")
    };
    assert_eq!(
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [(-1.0_f32).to_bits(), 1.0_f32.to_bits()]
    );
}

#[test]
fn yuv444_to_rgb_u16_preserves_inverse_transform_values() {
    let y = [0, 10];
    let u = [0, 3];
    let v = [0, 5];
    let planes = [
        ComponentPlane::tightly_packed(2, 1, &y),
        ComponentPlane::tightly_packed(2, 1, &u),
        ComponentPlane::tightly_packed(2, 1, &v),
    ];
    let decoded = format_components(
        &planes,
        None,
        request(
            ColorFormat::Yuv(ChromaSampling::Cs444),
            ColorFormat::Rgb,
            OutputBitDepth::U16 { shift_bits: 0 },
            PixelFormat::U16(ChannelLayout::Rgb),
        ),
    )
    .unwrap();
    assert_eq!(
        decoded,
        DecodedSamples::U16(vec![32_768, 32_768, 32_768, 32_774, 32_780, 32_779])
    );
}

#[test]
fn direct_typed_destinations_match_owned_component_packing() {
    let values = [-17, 23];
    let planes = [ComponentPlane::tightly_packed(2, 1, &values)];

    let u16_request = request(
        ColorFormat::Luma,
        ColorFormat::Luma,
        OutputBitDepth::U16 { shift_bits: 0 },
        PixelFormat::U16(ChannelLayout::Luma),
    );
    let expected = format_components(&planes, None, u16_request).unwrap();
    let mut direct = [0_u16; 2];
    format_components_into_with_cpu(
        &planes,
        None,
        u16_request,
        CpuCapabilities::detect(),
        DecodedSamplesMut::U16(&mut direct),
    )
    .unwrap();
    assert_eq!(expected, DecodedSamples::U16(direct.to_vec()));

    let f32_request = request(
        ColorFormat::Luma,
        ColorFormat::Luma,
        OutputBitDepth::F32 {
            mantissa_length: 4,
            exponent_bias: 1,
        },
        PixelFormat::F32(ChannelLayout::Luma),
    );
    let expected = format_components(&planes, None, f32_request).unwrap();
    let mut direct = [0.0_f32; 2];
    format_components_into_with_cpu(
        &planes,
        None,
        f32_request,
        CpuCapabilities::detect(),
        DecodedSamplesMut::F32(&mut direct),
    )
    .unwrap();
    assert_eq!(expected, DecodedSamples::F32(direct.to_vec()));
}

#[test]
fn direct_packed_destinations_match_owned_component_packing() {
    let y = [176, 176];
    let u = [160, 160];
    let v = [-64, -64];
    let planes = [
        ComponentPlane::tightly_packed(2, 1, &y),
        ComponentPlane::tightly_packed(2, 1, &u),
        ComponentPlane::tightly_packed(2, 1, &v),
    ];
    let output = request(
        ColorFormat::Yuv(ChromaSampling::Cs444),
        ColorFormat::Rgbe,
        OutputBitDepth::U8,
        PixelFormat::Rgbe,
    );
    let expected = format_components(&planes, None, output).unwrap();
    let mut direct = [0_u32; 2];
    format_components_into_with_cpu(
        &planes,
        None,
        output,
        CpuCapabilities::detect(),
        DecodedSamplesMut::Rgbe(&mut direct),
    )
    .unwrap();
    assert_eq!(expected, DecodedSamples::Rgbe(direct.to_vec()));
}

#[test]
fn rgbx_typed_output_writes_a_zero_padding_component() {
    let y = [0, 10];
    let u = [0, 3];
    let v = [0, 5];
    let planes = [
        ComponentPlane::tightly_packed(2, 1, &y),
        ComponentPlane::tightly_packed(2, 1, &u),
        ComponentPlane::tightly_packed(2, 1, &v),
    ];
    let decoded = format_components(
        &planes,
        None,
        request(
            ColorFormat::Yuv(ChromaSampling::Cs444),
            ColorFormat::Rgb,
            OutputBitDepth::U16 { shift_bits: 0 },
            PixelFormat::U16(ChannelLayout::Rgbx),
        ),
    )
    .unwrap();
    assert_eq!(
        decoded,
        DecodedSamples::U16(vec![32_768, 32_768, 32_768, 0, 32_774, 32_780, 32_779, 0,])
    );
}

#[test]
fn bgrx_typed_output_reorders_color_and_writes_zero_padding() {
    let y = [10];
    let u = [3];
    let v = [5];
    let planes = [
        ComponentPlane::tightly_packed(1, 1, &y),
        ComponentPlane::tightly_packed(1, 1, &u),
        ComponentPlane::tightly_packed(1, 1, &v),
    ];
    let mut output = request(
        ColorFormat::Yuv(ChromaSampling::Cs444),
        ColorFormat::Rgb,
        OutputBitDepth::U16 { shift_bits: 0 },
        PixelFormat::U16(ChannelLayout::Bgrx),
    );
    output.crop.width = 1;
    let decoded = format_components(&planes, None, output).unwrap();
    assert_eq!(
        decoded,
        DecodedSamples::U16(vec![32_779, 32_780, 32_774, 0])
    );
}

#[test]
fn bit_packing_is_msb_first_and_black_mode_inverts() {
    let samples = [0, 1];
    let planes = [ComponentPlane::tightly_packed(2, 1, &samples)];
    let mut req = request(
        ColorFormat::Luma,
        ColorFormat::Luma,
        OutputBitDepth::Bit1Black,
        PixelFormat::BitPacked(ChannelLayout::Luma),
    );
    let decoded = format_components(&planes, None, req).unwrap();
    assert_eq!(decoded, DecodedSamples::BitPacked(vec![0x80]));
    req.bit_depth = OutputBitDepth::Bit1White;
    let decoded = format_components(&planes, None, req).unwrap();
    assert_eq!(decoded, DecodedSamples::BitPacked(vec![0x40]));
}

#[test]
fn cmyk_bias_uses_opposite_black_direction() {
    let y = [0, 0];
    let u = [0, 0];
    let v = [0, 0];
    let k = [0, 0];
    let planes = [
        ComponentPlane::tightly_packed(2, 1, &y),
        ComponentPlane::tightly_packed(2, 1, &u),
        ComponentPlane::tightly_packed(2, 1, &v),
        ComponentPlane::tightly_packed(2, 1, &k),
    ];
    let decoded = format_components(
        &planes,
        None,
        request(
            ColorFormat::YuvK,
            ColorFormat::Cmyk,
            OutputBitDepth::U8,
            PixelFormat::U8(ChannelLayout::Cmyk),
        ),
    )
    .unwrap();
    assert_eq!(
        decoded,
        DecodedSamples::U8(vec![64, 64, 64, 0, 64, 64, 64, 0])
    );
}

#[test]
fn ncomponent_output_interleaves_more_than_four_planes() {
    let values = [[-128, -127], [-1, 0], [1, 2], [126, 127], [200, -200]];
    let planes = values
        .iter()
        .map(|samples| ComponentPlane::tightly_packed(2, 1, samples))
        .collect::<Vec<_>>();
    let decoded = format_components(
        &planes,
        None,
        request(
            ColorFormat::NComponent(5),
            ColorFormat::NComponent(5),
            OutputBitDepth::U8,
            PixelFormat::U8(ChannelLayout::NComponent(5)),
        ),
    )
    .unwrap();
    assert_eq!(
        decoded,
        DecodedSamples::U8(vec![0, 127, 129, 254, 255, 1, 128, 130, 255, 0])
    );
}

#[test]
fn cmyk_direct_alpha_uses_a_five_channel_layout() {
    let component = [0];
    let alpha = [-128];
    let planes = [
        ComponentPlane::tightly_packed(1, 1, &component),
        ComponentPlane::tightly_packed(1, 1, &component),
        ComponentPlane::tightly_packed(1, 1, &component),
        ComponentPlane::tightly_packed(1, 1, &component),
    ];
    let mut output = request(
        ColorFormat::YuvK,
        ColorFormat::CmykDirect,
        OutputBitDepth::U8,
        PixelFormat::U8(ChannelLayout::Cmyka),
    );
    output.crop.width = 1;
    let decoded = format_components(
        &planes,
        Some(ComponentPlane::tightly_packed(1, 1, &alpha)),
        output,
    )
    .unwrap();
    assert_eq!(decoded, DecodedSamples::U8(vec![128, 128, 128, 128, 0]));
}

#[test]
fn ncomponent_alpha_appends_after_every_primary_component() {
    let values = [[-3], [-2], [-1], [0], [1]];
    let planes: Vec<_> = values
        .iter()
        .map(|samples| ComponentPlane::tightly_packed(1, 1, samples))
        .collect();
    let alpha = [127];
    let mut output = request(
        ColorFormat::NComponent(5),
        ColorFormat::NComponent(5),
        OutputBitDepth::U8,
        PixelFormat::U8(ChannelLayout::NComponentAlpha(5)),
    );
    output.crop.width = 1;
    let decoded = format_components(
        &planes,
        Some(ComponentPlane::tightly_packed(1, 1, &alpha)),
        output,
    )
    .unwrap();
    assert_eq!(
        decoded,
        DecodedSamples::U8(vec![125, 126, 127, 128, 129, 255])
    );
}

#[test]
fn invalid_layout_is_a_typed_error() {
    let samples = [0, 0];
    let planes = [ComponentPlane::tightly_packed(2, 1, &samples)];
    let error = format_components(
        &planes,
        None,
        request(
            ColorFormat::Luma,
            ColorFormat::Luma,
            OutputBitDepth::U8,
            PixelFormat::U8(ChannelLayout::Rgb),
        ),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        OutputFormatError::UnsupportedCombination { .. }
    ));
}

#[test]
fn rgbe_output_applies_color_conversion_and_shared_exponent_packing() {
    let y = [176, 176];
    let u = [160, 160];
    let v = [-64, -64];
    let planes = [
        ComponentPlane::tightly_packed(2, 1, &y),
        ComponentPlane::tightly_packed(2, 1, &u),
        ComponentPlane::tightly_packed(2, 1, &v),
    ];
    let decoded = format_components(
        &planes,
        None,
        request(
            ColorFormat::Yuv(ChromaSampling::Cs444),
            ColorFormat::Rgbe,
            OutputBitDepth::U8,
            PixelFormat::Rgbe,
        ),
    )
    .unwrap();
    assert_eq!(decoded, DecodedSamples::Rgbe(vec![0x0220_8040; 2]));
}

#[test]
fn rgba_u8_premultiplication_uses_formatted_alpha() {
    let y = [0, 0];
    let u = [0, 0];
    let v = [0, 0];
    let alpha = [-64, 127];
    let planes = [
        ComponentPlane::tightly_packed(2, 1, &y),
        ComponentPlane::tightly_packed(2, 1, &u),
        ComponentPlane::tightly_packed(2, 1, &v),
    ];
    let mut req = request(
        ColorFormat::Yuv(ChromaSampling::Cs444),
        ColorFormat::Rgb,
        OutputBitDepth::U8,
        PixelFormat::U8(ChannelLayout::Rgba),
    );
    req.premultiply_alpha = true;
    let decoded = format_components(
        &planes,
        Some(ComponentPlane::tightly_packed(2, 1, &alpha)),
        req,
    )
    .unwrap();
    assert_eq!(
        decoded,
        DecodedSamples::U8(vec![32, 32, 32, 64, 128, 128, 128, 255])
    );
}
