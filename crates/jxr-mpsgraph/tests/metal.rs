// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

use jxr::{ChannelLayout, DecodeRequest, EncodedImage, PixelFormat, PreparedJxr, Rect};
use jxr_mpsgraph::{
    MpsGraphBatchDecoder, MpsGraphDecodeInput, MpsGraphDecodeOptions, MpsGraphProgram,
};

fn conformance_image() -> PreparedJxr {
    corpus_image("Output_Color_Format_Baseline/Maui-8bppGray_64x64.jxr")
}

fn corpus_image(relative: &str) -> PreparedJxr {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/t834-conformance/suite-2014")
        .join(relative);
    let bytes = std::fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "download the T.834 corpus before running this ignored device test ({}): {error}",
            path.display()
        )
    });
    PreparedJxr::from_arc(bytes.into()).unwrap()
}

#[test]
#[ignore = "requires the downloaded T.834 corpus and an Apple GPU"]
fn device_identity_covers_gray_rgb_rgba_integer_matrix() {
    let cases = [
        (
            "Output_Color_Format_Baseline/Maui-8bppGray_64x64.jxr",
            PixelFormat::U8(ChannelLayout::Luma),
        ),
        (
            "Output_Color_Format_Baseline/Maui-24bppRGB_64x64.jxr",
            PixelFormat::U8(ChannelLayout::Rgb),
        ),
        (
            "Alpha_Interleaved/Maui-32bppBGRA_64x64_Interleaved.jxr",
            PixelFormat::U8(ChannelLayout::Rgba),
        ),
        (
            "Output_Color_Format_Baseline/Maui-16bppGray.jxr",
            PixelFormat::U16(ChannelLayout::Luma),
        ),
        (
            "Output_Color_Format_Baseline/Maui-48bppRGB_64x64.jxr",
            PixelFormat::U16(ChannelLayout::Rgb),
        ),
        (
            "Alpha_Interleaved/Maui-64bppRGBA_64x64_Interleaved.jxr",
            PixelFormat::U16(ChannelLayout::Rgba),
        ),
        (
            "Output_Color_Format_Baseline/Maui-16bppGrayFixedPoint_64x64.jxr",
            PixelFormat::I16(ChannelLayout::Luma),
        ),
        (
            "Output_Color_Format_Baseline/Maui-48bppRGBFixedPoint_64x64.jxr",
            PixelFormat::I16(ChannelLayout::Rgb),
        ),
        (
            "Alpha_Interleaved/Maui-64bppRGBAFixedPoint_64x64_Interleaved.jxr",
            PixelFormat::I16(ChannelLayout::Rgba),
        ),
    ];
    let mut decoder = MpsGraphBatchDecoder::system_default().unwrap();
    for (relative, format) in cases {
        let image = corpus_image(relative);
        let prepared = decoder.prepare(vec![input(&image, format)]).unwrap();
        assert!(
            prepared.errors().is_empty(),
            "{relative}: {:?}",
            prepared.errors()
        );
        let group = &prepared.groups()[0];
        let program = MpsGraphProgram::identity(group.spec()).unwrap();
        let output = decoder
            .run_prepared_group(&program, group)
            .unwrap_or_else(|error| panic!("{relative}: {error}"));
        assert_eq!(output.results().len(), 1, "{relative}");
    }
}

#[test]
#[ignore = "requires the downloaded T.834 corpus and an Apple GPU"]
fn device_identity_covers_required_batch_sizes() {
    let image = conformance_image();
    let mut decoder = MpsGraphBatchDecoder::system_default().unwrap();
    for batch in [1, 2, 8, 32] {
        let prepared = decoder
            .prepare(
                (0..batch)
                    .map(|_| input(&image, PixelFormat::U8(ChannelLayout::Luma)))
                    .collect(),
            )
            .unwrap();
        let group = &prepared.groups()[0];
        assert_eq!(group.spec().shape()[0], batch);
        let program = MpsGraphProgram::identity(group.spec()).unwrap();
        let output = decoder.run_prepared_group(&program, group).unwrap();
        assert_eq!(output.source_indices().len(), batch);
        assert_eq!(output.results().len(), 1);
    }
}

#[test]
#[ignore = "requires the downloaded T.834 corpus and an Apple GPU"]
fn shared_native_batch_contract_feeds_dense_mpsgraph_decode() {
    let image = conformance_image();
    let mut decoder = MpsGraphBatchDecoder::system_default().unwrap();
    let inputs = (0..2)
        .map(|_| {
            EncodedImage::new(
                image.bytes().clone(),
                DecodeRequest::new(PixelFormat::U8(ChannelLayout::Luma)),
            )
        })
        .chain(std::iter::once(EncodedImage::full(
            std::sync::Arc::from([0_u8, 1, 2]),
            PixelFormat::U8(ChannelLayout::Luma),
        )))
        .collect();
    let prepared = decoder.prepare_batch(inputs).unwrap();
    let output = decoder.decode_batch(&prepared).unwrap();

    assert_eq!(output.errors().len(), 1);
    assert_eq!(output.errors()[0].source_index(), 2);
    assert!(output.group_errors().is_empty());
    assert_eq!(output.groups().len(), 1);
    assert_eq!(output.groups()[0].source_indices(), [0, 1]);
    assert_eq!(output.groups()[0].reports().len(), 2);
}

fn input(image: &PreparedJxr, format: PixelFormat) -> MpsGraphDecodeInput {
    MpsGraphDecodeInput {
        image: image.clone(),
        options: MpsGraphDecodeOptions::new(format),
    }
}

#[test]
#[ignore = "requires the downloaded T.834 corpus and an Apple GPU"]
fn prepare_partitions_by_tensor_shape_and_preserves_indexed_errors() {
    let image = conformance_image();
    let (width, height) = image.info().dimensions();
    let region_width = width / 2;
    let region_height = height / 2;
    let mut region = MpsGraphDecodeOptions::new(PixelFormat::U8(ChannelLayout::Luma));
    region.region = Some(Rect {
        x: 0,
        y: 0,
        w: region_width,
        h: region_height,
    });
    let decoder = MpsGraphBatchDecoder::system_default().unwrap();
    let prepared = decoder
        .prepare(vec![
            input(&image, PixelFormat::U8(ChannelLayout::Luma)),
            MpsGraphDecodeInput {
                image: image.clone(),
                options: region,
            },
            input(&image, PixelFormat::U8(ChannelLayout::Luma)),
            input(&image, PixelFormat::U8(ChannelLayout::Bgr)),
        ])
        .unwrap();

    assert_eq!(
        prepared.groups().len(),
        2,
        "errors: {:?}",
        prepared.errors()
    );
    assert_eq!(prepared.groups()[0].source_indices(), [0, 2]);
    assert_eq!(
        prepared.groups()[0].spec().shape(),
        [2, height as usize, width as usize, 1]
    );
    assert_eq!(prepared.groups()[1].source_indices(), [1]);
    assert_eq!(
        prepared.groups()[1].spec().shape(),
        [1, region_height as usize, region_width as usize, 1]
    );
    assert_eq!(prepared.errors().len(), 1);
    assert_eq!(prepared.errors()[0].source_index(), 3);
}

#[test]
#[ignore = "requires the downloaded T.834 corpus and an Apple GPU"]
fn direct_identity_run_and_completed_handoff_share_the_codec_queue() {
    let image = conformance_image();
    let mut decoder = MpsGraphBatchDecoder::system_default().unwrap();
    let prepared = decoder
        .prepare(vec![
            input(&image, PixelFormat::U8(ChannelLayout::Luma)),
            input(&image, PixelFormat::U8(ChannelLayout::Luma)),
        ])
        .unwrap();
    assert_eq!(
        prepared.groups().len(),
        1,
        "errors: {:?}",
        prepared.errors()
    );
    let group = &prepared.groups()[0];
    let program = MpsGraphProgram::identity(group.spec()).unwrap();

    drop(decoder.submit_prepared_group(&program, group).unwrap());
    let output = decoder.run_prepared_group(&program, group).unwrap();
    assert_eq!(output.results().len(), 1);
    assert_eq!(output.source_indices(), [0, 1]);
    assert_eq!(output.reports().len(), 2);

    let completed = decoder.decode_prepared(&prepared).unwrap();
    assert!(completed.group_errors().is_empty());
    assert_eq!(completed.groups()[0].source_indices(), [0, 1]);
    assert_eq!(completed.groups()[0].reports().len(), 2);
}

#[test]
#[ignore = "requires the downloaded T.834 corpus and an Apple GPU"]
fn odd_dense_stride_uses_surface_offsets_not_buffer_binding_offsets() {
    use core::{ffi::c_void, ptr::NonNull};

    let image = conformance_image();
    let mut options = MpsGraphDecodeOptions::new(PixelFormat::U8(ChannelLayout::Luma));
    options.region = Some(Rect {
        x: 0,
        y: 0,
        w: 17,
        h: 11,
    });
    let mut decoder = MpsGraphBatchDecoder::system_default().unwrap();
    let prepared = decoder
        .prepare(vec![
            MpsGraphDecodeInput {
                image: image.clone(),
                options: options.clone(),
            },
            MpsGraphDecodeInput { image, options },
        ])
        .unwrap();
    let group = &prepared.groups()[0];
    assert_eq!(group.spec().shape(), [2, 11, 17, 1]);
    let program = MpsGraphProgram::identity(group.spec()).unwrap();
    let output = decoder.run_prepared_group(&program, group).unwrap();
    assert_eq!(output.results().len(), 1);
    let image_len = 17 * 11;
    let mut pixels = vec![0_u8; 2 * image_len];
    // SAFETY: graph completion has established an identity U8 output with the
    // exact static shape `[2, 11, 17, 1]` covered by `pixels`.
    unsafe {
        output.results()[0].mpsndarray().readBytes_strideBytes(
            NonNull::new(pixels.as_mut_ptr().cast::<c_void>()).unwrap(),
            core::ptr::null_mut(),
        );
    }
    assert_eq!(&pixels[..image_len], &pixels[image_len..]);
}

#[test]
#[ignore = "1,000-run Apple GPU lifecycle soak"]
fn direct_run_lifecycle_soak() {
    let image = conformance_image();
    let mut decoder = MpsGraphBatchDecoder::system_default().unwrap();
    let prepared = decoder
        .prepare(vec![input(&image, PixelFormat::U8(ChannelLayout::Luma))])
        .unwrap();
    let group = &prepared.groups()[0];
    let program = MpsGraphProgram::identity(group.spec()).unwrap();
    for _ in 0..1_000 {
        let output = decoder.run_prepared_group(&program, group).unwrap();
        assert_eq!(output.results().len(), 1);
    }
}
