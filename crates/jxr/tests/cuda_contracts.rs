#![cfg(feature = "cuda")]

use jxr::{
    BackendKind, BackendRequest, BatchDecodeOptions, ChannelLayout, CudaBatchDecoder,
    DecodeRequest, EncodedImage, JxrErrorKind, JxrView, PixelFormat,
};
use std::sync::Arc;

fn decodable_minimal_raw_codestream() -> Vec<u8> {
    let mut bytes = b"WMPHOTO\0".to_vec();
    bytes.extend_from_slice(&[0x11, 0x00, 0x80, 0x01]);
    bytes.extend_from_slice(&15_u16.to_be_bytes());
    bytes.extend_from_slice(&15_u16.to_be_bytes());
    let mut bit_position = bytes.len() * 8;
    for (value, count) in [(0, 3), (0, 1), (3, 4), (1, 1), (1, 8)] {
        push_bits(&mut bytes, &mut bit_position, value, count);
    }
    bytes.extend_from_slice(&[0xFD, 0, 0, 1, 0x5a, 0, 0, 0]);
    bytes
}

fn push_bits(bytes: &mut Vec<u8>, bit_position: &mut usize, value: u64, count: u8) {
    for shift in (0..count).rev() {
        if *bit_position / 8 == bytes.len() {
            bytes.push(0);
        }
        let bit = u8::from(((value >> shift) & 1) != 0);
        bytes[*bit_position / 8] |= bit << (7 - (*bit_position % 8));
        *bit_position += 1;
    }
}

fn request(backend: BackendRequest) -> DecodeRequest {
    DecodeRequest::new(PixelFormat::U8(ChannelLayout::Luma)).with_backend(backend)
}

#[test]
fn prepared_reconstruction_builds_an_executable_cuda_plan() {
    let bytes = decodable_minimal_raw_codestream();
    let view = JxrView::parse(&bytes).unwrap();
    let prepared = view
        .decoder()
        .prepare_reconstruction(&request(BackendRequest::Cuda))
        .unwrap();
    let plan = prepared.cuda_plan().unwrap();
    assert!(plan.is_executable());
    assert_eq!(plan.output().width, 16);
    assert_eq!(plan.requested_backend(), BackendRequest::Cuda);
}

#[test]
fn strict_cuda_without_an_attached_session_never_falls_back() {
    let bytes = decodable_minimal_raw_codestream();
    let view = JxrView::parse(&bytes).unwrap();
    let error = view
        .decoder()
        .decode(&request(BackendRequest::Cuda))
        .unwrap_err();
    assert_eq!(error.kind, JxrErrorKind::BackendUnavailable);
}

#[test]
fn auto_without_an_attached_cuda_session_remains_on_cpu() {
    let bytes = decodable_minimal_raw_codestream();
    let view = JxrView::parse(&bytes).unwrap();
    let image = view
        .decoder()
        .decode(&request(BackendRequest::Auto))
        .unwrap();
    assert_eq!(image.report.selected, BackendKind::Cpu);
}

#[test]
#[ignore = "requires a compatible NVIDIA GPU, CUDA driver, and NVRTC"]
fn cuda_submission_resident_readback_and_host_decode_match_cpu() {
    let bytes = decodable_minimal_raw_codestream();
    let view = JxrView::parse(&bytes).unwrap();
    let cpu = view
        .decoder()
        .decode(&request(BackendRequest::Cpu))
        .unwrap();
    let session = jxr::cuda::CudaDecoderSession::system_default().unwrap();
    let prepared = view
        .decoder()
        .prepare_reconstruction(&request(BackendRequest::Cuda))
        .unwrap();
    let plan = prepared.cuda_plan().unwrap();

    let pool_before = session.buffer_pool_diagnostics().unwrap();
    let cache_before = session.upload_cache_diagnostics().unwrap();
    let context_probe = session.allocate_destination(plan.output().clone()).unwrap();
    let consumer = context_probe
        .device_buffer()
        .context()
        .new_stream()
        .unwrap();
    drop(context_probe);
    let pending = session.submit(&plan).unwrap();
    pending.enqueue_consumer_wait(&consumer).unwrap();
    let resident = pending.wait().unwrap();
    let bytes = session.readback(&resident).unwrap();
    let jxr::DecodedSamples::U8(expected) = &cpu.samples else {
        panic!("expected U8 CPU output");
    };
    assert_eq!(bytes, *expected);

    let decoded = session.decode_to_host(&plan).unwrap();
    assert_eq!(decoded.samples, cpu.samples);
    assert_eq!(decoded.report.selected, BackendKind::Cuda);
    let pool_after = session.buffer_pool_diagnostics().unwrap();
    let cache_after = session.upload_cache_diagnostics().unwrap();
    assert!(pool_after.hits >= pool_before.hits + 2);
    assert!(cache_after.hits > cache_before.hits);

    let destination = session.allocate_destination(plan.output().clone()).unwrap();
    let (completion, destination) = session
        .submit_into(&plan, destination)
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(&completion.layout, plan.output());
    assert_eq!(destination.layout(), plan.output());

    let routed = view
        .decoder()
        .with_cuda_session(&session)
        .decode(&request(BackendRequest::Cuda))
        .unwrap();
    assert_eq!(routed.samples, cpu.samples);
    assert_eq!(routed.report.selected, BackendKind::Cuda);
}

#[test]
#[ignore = "requires a compatible NVIDIA GPU, CUDA driver, and NVRTC"]
fn cuda_dense_batch_and_checked_destination_match_individual_output() {
    let bytes = decodable_minimal_raw_codestream();
    let view = JxrView::parse(&bytes).unwrap();
    let session = jxr::cuda::CudaDecoderSession::system_default().unwrap();
    let prepared = view
        .decoder()
        .prepare_reconstruction(&request(BackendRequest::Cuda))
        .unwrap();
    let plan = prepared.cuda_plan().unwrap();
    let plans = vec![plan.clone(); 8];

    let dense = session.submit_dense_batch(&plans).unwrap().wait().unwrap();
    assert_eq!(dense.layout().image_count(), 8);
    let first = session.readback_batch_image(&dense, 0).unwrap();
    for image in 1..8 {
        assert_eq!(session.readback_batch_image(&dense, image).unwrap(), first);
    }

    let destination = session
        .allocate_batch_destination(dense.layout().clone())
        .unwrap();
    let completed = session
        .submit_batch_into(&plans, destination)
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(completed.layout().image_count(), 8);
}

#[test]
#[ignore = "requires a compatible NVIDIA GPU, CUDA driver, and NVRTC"]
fn high_level_cuda_batch_preserves_native_groups_and_resident_outputs() {
    let source: Arc<[u8]> = decodable_minimal_raw_codestream().into();
    let session = jxr::cuda::CudaDecoderSession::system_default().unwrap();
    let decoder = CudaBatchDecoder::with_session(session, BatchDecodeOptions::default()).unwrap();
    let inputs = (0..8)
        .map(|_| EncodedImage::new(Arc::clone(&source), request(BackendRequest::Cuda)))
        .collect();
    let submitted = decoder
        .submit_prepared(&decoder.prepare(inputs).unwrap())
        .unwrap();
    let completed = submitted.wait();
    assert!(completed.errors().is_empty());
    assert!(completed.group_errors().is_empty());
    assert_eq!(completed.groups().len(), 1);
    assert_eq!(
        completed.groups()[0].source_indices(),
        &[0, 1, 2, 3, 4, 5, 6, 7]
    );
    assert_eq!(completed.groups()[0].images().len(), 8);
}
