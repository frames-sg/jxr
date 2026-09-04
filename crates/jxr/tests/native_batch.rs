use std::{num::NonZeroUsize, sync::Arc};

use jxr::{
    AlphaHandling, BackendRequest, BatchDecodeOptions, BatchErrorStage, BatchInfrastructureError,
    BatchLayout, ChannelLayout, CpuBatchDecoder, CpuBatchDestination, CpuBatchSamples,
    DecodeRequest, EncodedImage, PixelFormat, PreparedJxr, Rect,
};

fn minimal_raw_codestream() -> Vec<u8> {
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

fn request(format: PixelFormat) -> DecodeRequest {
    DecodeRequest::new(format).with_backend(BackendRequest::Cpu)
}

#[test]
fn preparation_groups_native_contracts_and_isolates_bad_inputs() {
    let encoded: Arc<[u8]> = minimal_raw_codestream().into();
    let inputs = vec![
        EncodedImage::new(
            Arc::clone(&encoded),
            request(PixelFormat::U8(ChannelLayout::Luma)),
        ),
        EncodedImage::full(
            Arc::from([0_u8, 1, 2]),
            PixelFormat::U8(ChannelLayout::Luma),
        ),
        EncodedImage::new(
            Arc::clone(&encoded),
            request(PixelFormat::U8(ChannelLayout::Luma)).with_region(Rect {
                x: 0,
                y: 0,
                w: 8,
                h: 8,
            }),
        ),
        EncodedImage::new(
            Arc::clone(&encoded),
            request(PixelFormat::U8(ChannelLayout::Luma)),
        ),
    ];
    let decoder = CpuBatchDecoder::new(BatchDecodeOptions::default()).unwrap();
    let prepared = decoder.prepare(inputs).unwrap();

    assert_eq!(prepared.errors().len(), 1, "{:?}", prepared.errors());
    assert_eq!(prepared.errors()[0].index(), 1);
    assert_eq!(prepared.groups().len(), 2);
    assert_eq!(prepared.groups()[0].source_indices(), &[0, 3]);
    assert_eq!(prepared.groups()[1].source_indices(), &[2]);
    assert_eq!(
        prepared.groups()[0].info().format(),
        PixelFormat::U8(ChannelLayout::Luma)
    );
    assert_eq!(prepared.groups()[0].info().dimensions(), (16, 16));

    let batch = decoder.decode_prepared(&prepared).unwrap();
    assert_eq!(batch.errors().len(), 1);
    assert_eq!(batch.errors()[0].index(), 1);
    assert_eq!(batch.groups().len(), 2);
    assert_eq!(batch.groups()[0].samples().byte_len(), 2 * 16 * 16);
    assert_eq!(batch.groups()[1].samples().byte_len(), 8 * 8);
}

#[test]
fn prepared_cpu_batch_is_reusable_and_matches_individual_decode() {
    let encoded: Arc<[u8]> = minimal_raw_codestream().into();
    let decode_request = request(PixelFormat::U8(ChannelLayout::Luma));
    let prepared_image = PreparedJxr::from_arc(Arc::clone(&encoded)).unwrap();
    let expected = prepared_image.decoder().decode(&decode_request).unwrap();
    let inputs = (0..2)
        .map(|_| EncodedImage::new(Arc::clone(&encoded), decode_request.clone()))
        .collect();
    let decoder = CpuBatchDecoder::new(BatchDecodeOptions::default()).unwrap();
    let prepared = decoder.prepare(inputs).unwrap();

    for _ in 0..2 {
        let batch = decoder.decode_prepared(&prepared).unwrap();
        assert!(batch.errors().is_empty());
        assert_eq!(batch.groups().len(), 1);
        let group = &batch.groups()[0];
        assert_eq!(group.source_indices(), &[0, 1]);
        assert_eq!(group.image_stride_bytes(), expected.samples.byte_len());
        assert_eq!(
            group.image_stride_elements(),
            expected.samples.sample_count()
        );
        let CpuBatchSamples::U8(samples) = group.samples() else {
            panic!("expected native U8 batch samples");
        };
        let jxr::DecodedSamples::U8(expected_samples) = &expected.samples else {
            panic!("expected individual U8 samples");
        };
        assert_eq!(&samples[..expected_samples.len()], expected_samples);
        assert_eq!(&samples[expected_samples.len()..], expected_samples);
    }

    let images = prepared.groups()[0].images().to_vec();
    let regrouped = decoder.prepare_prepared_images(images).unwrap();
    assert_eq!(regrouped.groups().len(), 1);
    assert_eq!(regrouped.groups()[0].source_indices(), &[0, 1]);
}

#[test]
fn cpu_batch_keeps_successful_siblings_when_one_route_is_invalid() {
    let encoded: Arc<[u8]> = minimal_raw_codestream().into();
    let cpu = request(PixelFormat::U8(ChannelLayout::Luma));
    let metal = DecodeRequest::new(PixelFormat::U8(ChannelLayout::Luma))
        .with_backend(BackendRequest::Metal);
    let decoder = CpuBatchDecoder::new(BatchDecodeOptions::default()).unwrap();
    let batch = decoder
        .decode(vec![
            EncodedImage::new(Arc::clone(&encoded), cpu),
            EncodedImage::new(encoded, metal),
        ])
        .unwrap();

    assert_eq!(batch.groups().len(), 1);
    assert_eq!(batch.groups()[0].source_indices(), &[0]);
    assert_eq!(batch.errors().len(), 1);
    assert_eq!(batch.errors()[0].index(), 1);
    assert_eq!(batch.errors()[0].stage(), BatchErrorStage::Decode);
}

#[test]
fn batch_limits_fail_before_dense_output_allocation() {
    let encoded: Arc<[u8]> = minimal_raw_codestream().into();
    let input = EncodedImage::new(encoded, request(PixelFormat::U8(ChannelLayout::Luma)));
    let options = BatchDecodeOptions {
        max_inputs: 1,
        max_host_allocation_bytes: 255,
        ..BatchDecodeOptions::default()
    };
    let decoder = CpuBatchDecoder::new(options).unwrap();
    let prepared = decoder.prepare(vec![input.clone()]).unwrap();
    assert!(matches!(
        decoder.decode_prepared(&prepared),
        Err(BatchInfrastructureError::OutputAllocationTooLarge {
            requested: 256,
            maximum: 255,
        })
    ));
    assert!(matches!(
        decoder.prepare(vec![input.clone(), input]),
        Err(BatchInfrastructureError::TooManyInputs {
            requested: 2,
            maximum: 1,
        })
    ));

    let permissive = CpuBatchDecoder::new(BatchDecodeOptions::default()).unwrap();
    let prepared = permissive
        .prepare(vec![
            EncodedImage::new(
                Arc::from(minimal_raw_codestream()),
                request(PixelFormat::U8(ChannelLayout::Luma)),
            ),
            EncodedImage::new(
                Arc::from(minimal_raw_codestream()),
                request(PixelFormat::U8(ChannelLayout::Luma)),
            ),
        ])
        .unwrap();
    assert!(matches!(
        decoder.decode_prepared(&prepared),
        Err(BatchInfrastructureError::TooManyInputs {
            requested: 2,
            maximum: 1,
        })
    ));
}

#[test]
fn repeated_identity_requests_hit_the_preparation_cache() {
    let encoded: Arc<[u8]> = minimal_raw_codestream().into();
    let input = EncodedImage::new(
        Arc::clone(&encoded),
        request(PixelFormat::U8(ChannelLayout::Luma)),
    );
    let decoder = CpuBatchDecoder::new(BatchDecodeOptions::default()).unwrap();

    let first = decoder.prepare(vec![input.clone(), input.clone()]).unwrap();
    let second = decoder.prepare(vec![input]).unwrap();

    assert_eq!(first.groups()[0].images().len(), 2);
    assert_eq!(second.groups()[0].images().len(), 1);
    let diagnostics = decoder.diagnostics();
    assert_eq!(diagnostics.preparation_calls, 2);
    assert_eq!(diagnostics.preparation_cache_misses, 1);
    assert_eq!(diagnostics.preparation_cache_hits, 2);
    assert_eq!(diagnostics.prepared_inputs, 3);

    let images = first.groups()[0].images();
    assert!(!images[0].reconstruction_is_cached());
    let first_coefficients = images[0].prepare_reconstruction().unwrap();
    assert!(images[1].reconstruction_is_cached());
    let second_coefficients = images[1].prepare_reconstruction().unwrap();
    assert!(Arc::ptr_eq(
        first_coefficients.coefficients(),
        second_coefficients.coefficients()
    ));
}

#[test]
fn cpu_batch_reuses_native_workspace_across_calls() {
    let encoded: Arc<[u8]> = minimal_raw_codestream().into();
    let input = EncodedImage::new(encoded, request(PixelFormat::U8(ChannelLayout::Luma)));
    let decoder = CpuBatchDecoder::new(BatchDecodeOptions {
        workers: NonZeroUsize::new(1),
        ..BatchDecodeOptions::default()
    })
    .unwrap();
    let prepared = decoder.prepare(vec![input]).unwrap();

    decoder.decode_prepared(&prepared).unwrap();
    let first = decoder.diagnostics();
    decoder.decode_prepared(&prepared).unwrap();
    let second = decoder.diagnostics();

    assert_eq!(second.decode_calls, first.decode_calls + 1);
    assert!(second.coefficient_workspace_reuses > first.coefficient_workspace_reuses);
    assert!(second.retained_coefficient_bytes > 0);
    assert!(second.reconstruction_workspace_reuses > first.reconstruction_workspace_reuses);
    assert!(second.retained_reconstruction_bytes > 0);
    assert_eq!(second.direct_dense_images, 2);
}

#[test]
fn caller_owned_cpu_destination_receives_dense_images() {
    let encoded: Arc<[u8]> = minimal_raw_codestream().into();
    let inputs = (0..2)
        .map(|_| {
            EncodedImage::new(
                Arc::clone(&encoded),
                request(PixelFormat::U8(ChannelLayout::Luma)),
            )
        })
        .collect();
    let decoder = CpuBatchDecoder::new(BatchDecodeOptions::default()).unwrap();
    let prepared = decoder.prepare(inputs).unwrap();
    let mut output = vec![0xa5; 2 * 16 * 16];

    let result = decoder
        .decode_prepared_group_into(&prepared.groups()[0], CpuBatchDestination::U8(&mut output))
        .unwrap();

    assert!(result.errors().is_empty());
    assert_eq!(result.source_indices(), &[0, 1]);
    assert_eq!(&output[..256], &output[256..]);
    assert_ne!(output, vec![0xa5; 512]);

    let mut wrong_type = vec![0_u16; 512];
    assert!(matches!(
        decoder.decode_prepared_group_into(
            &prepared.groups()[0],
            CpuBatchDestination::U16(&mut wrong_type),
        ),
        Err(BatchInfrastructureError::InvalidDestination { .. })
    ));
}

#[test]
fn separate_alpha_batch_writes_directly_into_the_dense_destination() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../target/t834-conformance/suite-2014/Output_Color_Format_Main/3channel_noprof_alpha.jxr",
    );
    let Ok(bytes) = std::fs::read(path) else {
        return;
    };
    let encoded: Arc<[u8]> = bytes.into();
    let decode_request = request(PixelFormat::U8(ChannelLayout::NComponentAlpha(3)))
        .with_alpha(AlphaHandling::Preserve);
    let prepared_image = PreparedJxr::from_arc(Arc::clone(&encoded)).unwrap();
    let mut individual_decoder = prepared_image.decoder();
    let expected = individual_decoder.decode(&decode_request).unwrap();
    let mut direct_output = vec![0_u8; expected.samples.sample_count()];
    let direct = individual_decoder
        .decode_into(&decode_request, &mut direct_output)
        .unwrap();
    let jxr::DecodedSamples::U8(expected_samples) = &expected.samples else {
        panic!("expected U8 individual output");
    };
    assert_eq!(direct.format, decode_request.output);
    assert_eq!(direct_output, *expected_samples);
    let decoder = CpuBatchDecoder::new(BatchDecodeOptions {
        workers: NonZeroUsize::new(1),
        ..BatchDecodeOptions::default()
    })
    .unwrap();
    let prepared = decoder
        .prepare(vec![EncodedImage::new(encoded, decode_request)])
        .unwrap();

    for _ in 0..2 {
        let batch = decoder.decode_prepared(&prepared).unwrap();
        assert!(batch.errors().is_empty());
        assert_eq!(batch.groups().len(), 1);
        let CpuBatchSamples::U8(actual) = batch.groups()[0].samples() else {
            panic!("expected U8 batch output");
        };
        assert_eq!(actual, expected_samples);
    }
    let diagnostics = decoder.diagnostics();
    assert_eq!(diagnostics.direct_dense_images, 2);
    assert_eq!(diagnostics.fallback_materialized_images, 0);
    assert!(diagnostics.coefficient_workspace_reuses > 0);
    assert!(diagnostics.reconstruction_workspace_reuses > 0);
}

#[test]
fn nchw_policy_is_retained_in_the_dense_contract() {
    let encoded: Arc<[u8]> = minimal_raw_codestream().into();
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nchw,
        ..BatchDecodeOptions::default()
    };
    let decoder = CpuBatchDecoder::new(options).unwrap();
    let batch = decoder
        .decode(vec![EncodedImage::new(
            encoded,
            request(PixelFormat::U8(ChannelLayout::Luma)),
        )])
        .unwrap();

    assert_eq!(batch.groups()[0].info().batch_layout(), BatchLayout::Nchw);
}

#[test]
fn nchw_u16_batch_writes_directly_into_the_dense_destination() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../target/t834-conformance/suite-2014/Output_Color_Format_Baseline/Maui-48bppRGB_64x64.jxr",
    );
    let Ok(bytes) = std::fs::read(path) else {
        return;
    };
    let encoded: Arc<[u8]> = bytes.into();
    let decode_request = request(PixelFormat::U16(ChannelLayout::Rgb));
    let expected_image = PreparedJxr::from_arc(Arc::clone(&encoded))
        .unwrap()
        .decoder()
        .decode(&decode_request)
        .unwrap();
    let pixels =
        usize::try_from(expected_image.decoded_region.w * expected_image.decoded_region.h).unwrap();
    let jxr::DecodedSamples::U16(expected) = expected_image.samples else {
        panic!("expected U16 individual output");
    };
    let mut expected_nchw = vec![0_u16; expected.len()];
    for pixel in 0..pixels {
        for channel in 0..3 {
            expected_nchw[channel * pixels + pixel] = expected[pixel * 3 + channel];
        }
    }
    let decoder = CpuBatchDecoder::new(BatchDecodeOptions {
        layout: BatchLayout::Nchw,
        workers: NonZeroUsize::new(1),
        ..BatchDecodeOptions::default()
    })
    .unwrap();
    let prepared = decoder
        .prepare(vec![EncodedImage::new(encoded, decode_request)])
        .unwrap();

    for _ in 0..2 {
        let batch = decoder.decode_prepared(&prepared).unwrap();
        assert!(batch.errors().is_empty());
        let CpuBatchSamples::U16(actual) = batch.groups()[0].samples() else {
            panic!("expected U16 batch output");
        };
        assert_eq!(actual, &expected_nchw);
    }
    let diagnostics = decoder.diagnostics();
    assert_eq!(diagnostics.direct_dense_images, 2);
    assert_eq!(diagnostics.fallback_materialized_images, 0);
    assert!(diagnostics.layout_workspace_reuses > 0);
    assert!(diagnostics.retained_layout_bytes >= u64::try_from(expected_nchw.len() * 2).unwrap());
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn assert_unavailable_metal_contract(
    decoder: &jxr::MetalBatchDecoder,
    prepared: &jxr::PreparedBatch,
) -> bool {
    if decoder.session().is_usable() {
        return false;
    }
    // Hosted macOS runners can expose a Metal device without the Apple GPU
    // features required by reconstruction. Exercise strict failure there.
    let batch = decoder.decode_prepared(prepared).unwrap();
    assert!(batch.groups().is_empty());
    assert_eq!(batch.group_errors().len(), 1);
    assert!(format!("{:?}", batch.group_errors()).contains("requires an M1-or-newer Apple GPU"));
    true
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_batch_consumes_shared_preparation_and_keeps_resident_outputs() {
    use jxr::MetalBatchDecoder;

    let encoded: Arc<[u8]> = minimal_raw_codestream().into();
    let strict_metal = DecodeRequest::new(PixelFormat::U8(ChannelLayout::Luma))
        .with_backend(BackendRequest::Metal);
    let inputs = (0..2)
        .map(|_| EncodedImage::new(Arc::clone(&encoded), strict_metal.clone()))
        .collect();
    let decoder = MetalBatchDecoder::system_default(BatchDecodeOptions::default()).unwrap();
    let prepared = decoder.prepare(inputs).unwrap();
    if assert_unavailable_metal_contract(&decoder, &prepared) {
        return;
    }
    let batch = decoder.decode_prepared(&prepared).unwrap();

    assert!(batch.errors().is_empty());
    assert!(batch.group_errors().is_empty());
    assert_eq!(batch.groups().len(), 1);
    let group = &batch.groups()[0];
    assert_eq!(group.source_indices(), &[0, 1]);
    assert_eq!(group.images().len(), 2);
    for image in group.images() {
        assert_eq!(image.layout(), group.info().image_layout());
        assert_eq!(decoder.session().readback(image).unwrap().len(), 256);
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_batch_exposes_nonblocking_and_single_allocation_groups() {
    use jxr::MetalBatchDecoder;

    let encoded: Arc<[u8]> = minimal_raw_codestream().into();
    let strict_metal = DecodeRequest::new(PixelFormat::U8(ChannelLayout::Luma))
        .with_backend(BackendRequest::Metal);
    let inputs = (0..2)
        .map(|_| EncodedImage::new(Arc::clone(&encoded), strict_metal.clone()))
        .collect();
    let decoder = MetalBatchDecoder::system_default(BatchDecodeOptions::default()).unwrap();
    let prepared = decoder.prepare(inputs).unwrap();
    if assert_unavailable_metal_contract(&decoder, &prepared) {
        return;
    }

    let submitted = decoder.submit_prepared(&prepared).unwrap();
    assert_eq!(submitted.pending_group_count(), 1);
    let resident = submitted.wait();
    assert!(resident.errors().is_empty());
    assert!(resident.group_errors().is_empty());
    assert_eq!(resident.groups()[0].images().len(), 2);

    let submitted = decoder.submit_prepared_dense(&prepared).unwrap();
    assert_eq!(submitted.pending_group_count(), 1);
    let dense = submitted.wait();
    assert!(dense.errors().is_empty());
    assert!(dense.group_errors().is_empty());
    let group = &dense.groups()[0];
    assert_eq!(group.batch().layout().image_count(), 2);
    assert_eq!(group.batch().layout().byte_len(), 512);
    assert_eq!(
        decoder
            .session()
            .readback_batch_image(group.batch(), 0)
            .unwrap(),
        decoder
            .session()
            .readback_batch_image(group.batch(), 1)
            .unwrap()
    );
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn high_level_metal_batch_writes_an_ordered_caller_owned_destination() {
    use jxr::{MetalBatchDecoder, metal::DenseMetalBatchLayout};

    let encoded: Arc<[u8]> = minimal_raw_codestream().into();
    let strict_metal = DecodeRequest::new(PixelFormat::U8(ChannelLayout::Luma))
        .with_backend(BackendRequest::Metal);
    let inputs = (0..2)
        .map(|_| EncodedImage::new(Arc::clone(&encoded), strict_metal.clone()))
        .collect();
    let session = jxr::metal::MetalDecoderSession::system_default_ordered().unwrap();
    let decoder = MetalBatchDecoder::with_session(session, BatchDecodeOptions::default()).unwrap();
    let prepared = decoder.prepare(inputs).unwrap();
    if assert_unavailable_metal_contract(&decoder, &prepared) {
        return;
    }
    let layout =
        DenseMetalBatchLayout::new(prepared.groups()[0].info().image_layout().clone(), 2).unwrap();
    let destination = decoder
        .session()
        .allocate_batch_destination(layout)
        .unwrap();

    let completed = decoder
        .submit_prepared_group_into(&prepared.groups()[0], destination)
        .unwrap()
        .wait()
        .unwrap();

    assert_eq!(completed.source_indices(), &[0, 1]);
    assert_eq!(completed.destination().layout().image_count(), 2);
    assert_eq!(completed.destination().reports().len(), 2);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_batch_isolates_an_explicit_cpu_request() {
    use jxr::MetalBatchDecoder;

    let encoded: Arc<[u8]> = minimal_raw_codestream().into();
    let metal = DecodeRequest::new(PixelFormat::U8(ChannelLayout::Luma))
        .with_backend(BackendRequest::Metal);
    let cpu = request(PixelFormat::U8(ChannelLayout::Luma));
    let decoder = MetalBatchDecoder::system_default(BatchDecodeOptions::default()).unwrap();
    let batch = decoder
        .decode(vec![
            EncodedImage::new(Arc::clone(&encoded), metal),
            EncodedImage::new(encoded, cpu),
        ])
        .unwrap();

    if decoder.session().is_usable() {
        assert!(batch.group_errors().is_empty());
        assert_eq!(batch.groups().len(), 1);
        assert_eq!(batch.groups()[0].source_indices(), &[0]);
    } else {
        assert!(batch.groups().is_empty());
        assert_eq!(batch.group_errors().len(), 1);
        assert!(
            format!("{:?}", batch.group_errors()).contains("requires an M1-or-newer Apple GPU")
        );
    }
    assert_eq!(batch.errors().len(), 1);
    assert_eq!(batch.errors()[0].index(), 1);
    assert_eq!(batch.errors()[0].stage(), BatchErrorStage::Decode);
}
