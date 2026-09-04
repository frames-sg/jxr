// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "cuda")]

use std::{collections::HashSet, path::Path};

use jxr::{
    AlphaMode, BackendRequest, DecodeRequest, JxrView, OverlapMode, PixelFormat, Rect,
    cuda::CudaDecoderSession,
};
use jxr_test_support::{T834CaseExpectation, discover_t834_cases, oracle_format};

#[test]
#[ignore = "requires the downloaded T.834 corpus; no NVIDIA device is needed"]
fn every_in_scope_t834_plan_passes_cuda_preflight_without_a_device() {
    let root = t834_root();
    let cases = discover_t834_cases(&root).expect("downloaded T.834 corpus");
    let mut prepared = 0_usize;
    let mut format_classes = HashSet::new();
    let mut alpha_modes = HashSet::new();
    let mut overlap_modes = HashSet::new();
    let mut saw_hard_tiles = false;
    let mut saw_soft_tiles = false;
    let mut saw_multi_tile = false;
    for (index, case) in cases.into_iter().enumerate() {
        if case.expectation != T834CaseExpectation::CompareMainSyntax {
            continue;
        }
        let source = std::fs::read(&case.input).expect("read T.834 input");
        let view = JxrView::parse(&source).expect("parse in-scope T.834 input");
        let format = oracle_format(view.info()).expect("representable T.834 output");
        format_classes.insert(pixel_format_class(format.pixel_format));
        alpha_modes.insert(view.info().alpha_mode);
        overlap_modes.insert(view.info().primary.overlap);
        saw_hard_tiles |= view.info().tiles.hard_tiles;
        saw_soft_tiles |= !view.info().tiles.hard_tiles;
        saw_multi_tile |=
            view.info().tiles.column_widths.len() * view.info().tiles.row_heights.len() > 1;
        preflight_request(
            &view,
            DecodeRequest::new(format.pixel_format).with_alpha(format.alpha),
            &case.relative_path,
        );
        if let Some(region) = boundary_region(view.info().width, view.info().height, index) {
            preflight_request(
                &view,
                DecodeRequest::new(format.pixel_format)
                    .with_alpha(format.alpha)
                    .with_region(region),
                &case.relative_path,
            );
        }
        prepared += 1;
    }
    assert_eq!(prepared, 517, "the in-scope CUDA corpus changed");
    assert_eq!(
        format_classes,
        HashSet::from([
            "bit-packed",
            "u8",
            "u16",
            "i16",
            "i32",
            "f16",
            "f32",
            "rgb555",
            "rgb565",
            "rgb101010",
            "rgbe",
        ]),
        "the CUDA preflight no longer covers every PixelFormat storage path"
    );
    assert_eq!(
        alpha_modes,
        HashSet::from([AlphaMode::None, AlphaMode::Integrated, AlphaMode::Separate]),
        "the CUDA preflight no longer covers every alpha mode"
    );
    assert_eq!(
        overlap_modes,
        HashSet::from([OverlapMode::None, OverlapMode::One, OverlapMode::Two]),
        "the CUDA preflight no longer covers every overlap mode"
    );
    assert!(saw_hard_tiles && saw_soft_tiles && saw_multi_tile);
}

#[test]
#[ignore = "requires the downloaded T.834 corpus and compatible NVIDIA CUDA hardware"]
fn every_in_scope_t834_output_and_roi_matches_the_cpu_oracle() {
    let root = t834_root();
    let session = CudaDecoderSession::system_default().expect("usable CUDA reconstruction session");
    let cases = discover_t834_cases(&root).expect("downloaded T.834 corpus");
    let mut formats = HashSet::new();
    let mut compared = 0_usize;
    for (index, case) in cases.into_iter().enumerate() {
        if case.expectation != T834CaseExpectation::CompareMainSyntax {
            continue;
        }
        let source = std::fs::read(&case.input).expect("read T.834 input");
        let view = JxrView::parse(&source).expect("parse in-scope T.834 input");
        let format = oracle_format(view.info()).expect("representable T.834 output");
        formats.insert(format.pixel_format);
        compare_request(
            &view,
            &session,
            DecodeRequest::new(format.pixel_format).with_alpha(format.alpha),
            &case.relative_path,
        );
        if let Some(region) = boundary_region(view.info().width, view.info().height, index) {
            compare_request(
                &view,
                &session,
                DecodeRequest::new(format.pixel_format)
                    .with_alpha(format.alpha)
                    .with_region(region),
                &case.relative_path,
            );
        }
        compared += 1;
    }
    assert_eq!(compared, 517, "the in-scope CUDA corpus changed");
    assert!(
        formats.len() >= 10,
        "T.834 output-format coverage regressed: {formats:?}"
    );
}

fn t834_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("test-support crate is inside the workspace")
        .join("target/t834-conformance/suite-2014")
}

fn preflight_request(view: &JxrView<'_>, mut request: DecodeRequest, path: &Path) {
    request.backend = BackendRequest::Cuda;
    let prepared = view
        .decoder()
        .prepare_reconstruction(&request)
        .unwrap_or_else(|error| panic!("CUDA preparation failed for {}: {error}", path.display()));
    prepared
        .cuda_plan()
        .unwrap_or_else(|error| panic!("CUDA preflight failed for {}: {error}", path.display()));
}

fn compare_request(
    view: &JxrView<'_>,
    session: &CudaDecoderSession,
    request: DecodeRequest,
    path: &Path,
) {
    let mut cpu_request = request.clone();
    cpu_request.backend = BackendRequest::Cpu;
    let cpu = view
        .decoder()
        .decode(&cpu_request)
        .unwrap_or_else(|error| panic!("CPU oracle failed for {}: {error}", path.display()));
    let mut cuda_request = request;
    cuda_request.backend = BackendRequest::Cuda;
    let cuda = view
        .decoder()
        .with_cuda_session(session)
        .decode(&cuda_request)
        .unwrap_or_else(|error| panic!("CUDA failed for {}: {error}", path.display()));
    assert_eq!(
        cuda.decoded_region,
        cpu.decoded_region,
        "{}",
        path.display()
    );
    assert_eq!(cuda.planes, cpu.planes, "{}", path.display());
    assert_eq!(cuda.samples, cpu.samples, "{}", path.display());
}

fn boundary_region(width: u32, height: u32, sequence: usize) -> Option<Rect> {
    if width < 4 || height < 4 {
        return None;
    }
    let w = ((width / 2) & !1).max(2);
    let h = ((height / 2) & !1).max(2);
    let (x, y) = match sequence % 3 {
        0 => (0, 0),
        1 => (width - w, height - h),
        _ => (
            (width.saturating_sub(w) / 2) & !1,
            (height.saturating_sub(h) / 2) & !1,
        ),
    };
    Some(Rect { x, y, w, h })
}

const fn pixel_format_class(format: PixelFormat) -> &'static str {
    match format {
        PixelFormat::BitPacked(_) => "bit-packed",
        PixelFormat::U8(_) => "u8",
        PixelFormat::U16(_) => "u16",
        PixelFormat::I16(_) => "i16",
        PixelFormat::I32(_) => "i32",
        PixelFormat::F16(_) => "f16",
        PixelFormat::F32(_) => "f32",
        PixelFormat::Rgb555 => "rgb555",
        PixelFormat::Rgb565 => "rgb565",
        PixelFormat::Rgb101010 => "rgb101010",
        PixelFormat::Rgbe => "rgbe",
    }
}
