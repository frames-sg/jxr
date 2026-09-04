// SPDX-License-Identifier: MIT OR Apache-2.0

use jxr::{ChannelLayout, PixelFormat, SurfaceLayout};
use jxr_mpsgraph::{MpsGraphElementType, MpsGraphTensorSpec, rgb8_nhwc_reference_cpu};

#[test]
fn gray_rgb_rgba_integer_layouts_map_to_static_nhwc() {
    for (layout, channels) in [
        (ChannelLayout::Luma, 1),
        (ChannelLayout::Rgb, 3),
        (ChannelLayout::Rgba, 4),
    ] {
        for (format, element) in [
            (PixelFormat::U8(layout), MpsGraphElementType::U8),
            (PixelFormat::U16(layout), MpsGraphElementType::U16),
            (PixelFormat::I16(layout), MpsGraphElementType::I16),
        ] {
            let layout = SurfaceLayout::tightly_packed(17, 11, format, 1).unwrap();
            let spec = MpsGraphTensorSpec::from_image_layout(&layout, 8).unwrap();
            assert_eq!(spec.shape(), [8, 11, 17, channels]);
            assert_eq!(spec.element_type(), element);
        }
    }
}

#[test]
fn tensor_contract_rejects_zero_overflow_and_unsupported_formats() {
    assert!(MpsGraphTensorSpec::new([0, 1, 1, 1], MpsGraphElementType::U8).is_err());
    assert!(MpsGraphTensorSpec::new([usize::MAX, 2, 1, 1], MpsGraphElementType::U16).is_err());
    let bgr = SurfaceLayout::tightly_packed(1, 1, PixelFormat::U8(ChannelLayout::Bgr), 1).unwrap();
    assert!(MpsGraphTensorSpec::from_image_layout(&bgr, 1).is_err());
}

#[test]
fn rgb8_reference_oracle_reduces_each_image() {
    let values = rgb8_nhwc_reference_cpu(&[255, 0, 0, 0, 255, 0], 2, 1, 1).unwrap();
    assert_eq!(values, [0.2126, 0.7152]);
}

#[test]
fn production_adapter_has_no_decoded_pixel_readback_or_upload_calls() {
    let source_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    for entry in std::fs::read_dir(source_root).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(std::ffi::OsStr::to_str) != Some("rs") {
            continue;
        }
        let source = std::fs::read_to_string(&path).unwrap();
        for forbidden in [
            "checked_buffer_read",
            "checked_buffer_write",
            "readBytes_strideBytes",
            "newBufferWithBytes",
        ] {
            assert!(
                !source.contains(forbidden),
                "forbidden {forbidden} in {}",
                path.display()
            );
        }
    }
}
