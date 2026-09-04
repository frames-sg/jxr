#![no_main]

use std::{
    num::NonZeroUsize,
    sync::{Arc, OnceLock},
};

use jxr::{
    BackendRequest, BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples,
    DecodeLimits, DecodeRequest, DecodeScale, DecodedImage, DecodedSamples, EncodedImage, JxrView,
    PixelFormat,
};
use jxr_test_support::oracle_format;
use libfuzzer_sys::fuzz_target;

const MAX_BYTES: u64 = 128 * 1024 * 1024;
const MAX_BATCH_BYTES: u64 = MAX_BYTES * 2;
static BATCH_DECODER: OnceLock<CpuBatchDecoder> = OnceLock::new();

fuzz_target!(|bytes: &[u8]| {
    let Ok(view) = JxrView::parse(bytes) else {
        return;
    };
    let output = oracle_format(view.info())
        .map(|format| (format.pixel_format, format.alpha))
        .unwrap_or((
            PixelFormat::U8(jxr::ChannelLayout::Luma),
            jxr::AlphaHandling::Drop,
        ));
    let scale = match bytes.len() % 3 {
        0 => DecodeScale::Full,
        1 => DecodeScale::Quarter,
        _ => DecodeScale::Sixteenth,
    };
    let request = DecodeRequest::new(output.0)
        .with_alpha(output.1)
        .with_scale(scale)
        .with_backend(BackendRequest::Cpu)
        .with_limits(fuzz_limits());
    let Ok(expected) = view.decoder().decode(&request) else {
        return;
    };

    assert_direct_output(&view, &request, &expected);
    assert_native_batch(bytes, request, &expected);
});

const fn fuzz_limits() -> DecodeLimits {
    DecodeLimits {
        max_width: 2048,
        max_height: 2048,
        max_pixels: 4 * 1024 * 1024,
        max_components: 16,
        max_tiles: 4096,
        max_compressed_bytes: 16 * 1024 * 1024,
        max_coefficient_bytes: MAX_BYTES,
        max_host_allocation_bytes: MAX_BYTES,
    }
}

fn assert_direct_output(view: &JxrView<'_>, request: &DecodeRequest, expected: &DecodedImage) {
    macro_rules! decode_and_compare {
        ($expected:expr, $value:expr) => {{
            let mut output = vec![$value; $expected.len()];
            let actual = view
                .decoder()
                .decode_into(request, &mut output)
                .expect("owned decode succeeded but typed decode failed");
            assert_eq!(actual.decoded_region, expected.decoded_region);
            assert_eq!(actual.format, expected.format);
            assert_eq!(actual.planes, expected.planes);
            assert_eq!(output, *$expected);
        }};
    }
    match &expected.samples {
        DecodedSamples::BitPacked(values) | DecodedSamples::U8(values) => {
            decode_and_compare!(values, 0_u8);
        }
        DecodedSamples::U16(values)
        | DecodedSamples::F16(values)
        | DecodedSamples::Rgb555(values)
        | DecodedSamples::Rgb565(values) => decode_and_compare!(values, 0_u16),
        DecodedSamples::I16(values) => decode_and_compare!(values, 0_i16),
        DecodedSamples::I32(values) => decode_and_compare!(values, 0_i32),
        DecodedSamples::F32(values) => {
            let mut output = vec![0.0_f32; values.len()];
            let actual = view
                .decoder()
                .decode_into(request, &mut output)
                .expect("owned decode succeeded but typed decode failed");
            assert_eq!(actual.decoded_region, expected.decoded_region);
            assert_eq!(actual.format, expected.format);
            assert_eq!(actual.planes, expected.planes);
            assert_f32_bits(&output, values);
        }
        DecodedSamples::Rgb101010(values) | DecodedSamples::Rgbe(values) => {
            decode_and_compare!(values, 0_u32);
        }
    }
}

fn assert_native_batch(bytes: &[u8], request: DecodeRequest, expected: &DecodedImage) {
    let decoder = BATCH_DECODER.get_or_init(|| {
        CpuBatchDecoder::new(BatchDecodeOptions {
            layout: BatchLayout::Native,
            workers: NonZeroUsize::new(1),
            max_inputs: 2,
            max_host_allocation_bytes: MAX_BATCH_BYTES,
            preparation_cache_entries: 0,
        })
        .expect("one-worker fuzz batch pool")
    });
    let encoded: Arc<[u8]> = bytes.into();
    let input = EncodedImage::new(encoded, request);
    let batch = decoder
        .decode(vec![input.clone(), input])
        .expect("owned decode succeeded but batch infrastructure failed");
    assert!(batch.errors().is_empty());
    assert_eq!(batch.groups().len(), 1);
    let group = &batch.groups()[0];
    assert_eq!(group.source_indices(), [0, 1]);
    assert_eq!(
        group.image_stride_elements(),
        storage_elements(&expected.samples)
    );
    macro_rules! compare_repeated {
        ($actual:expr, $expected:expr) => {{
            assert_eq!($actual.len(), $expected.len() * 2);
            assert_eq!(&$actual[..$expected.len()], $expected);
            assert_eq!(&$actual[$expected.len()..], $expected);
        }};
    }
    match (group.samples(), &expected.samples) {
        (CpuBatchSamples::BitPacked(actual), DecodedSamples::BitPacked(expected))
        | (CpuBatchSamples::U8(actual), DecodedSamples::U8(expected)) => {
            compare_repeated!(actual, expected);
        }
        (CpuBatchSamples::U16(actual), DecodedSamples::U16(expected))
        | (CpuBatchSamples::F16(actual), DecodedSamples::F16(expected))
        | (CpuBatchSamples::Rgb555(actual), DecodedSamples::Rgb555(expected))
        | (CpuBatchSamples::Rgb565(actual), DecodedSamples::Rgb565(expected)) => {
            compare_repeated!(actual, expected);
        }
        (CpuBatchSamples::I16(actual), DecodedSamples::I16(expected)) => {
            compare_repeated!(actual, expected);
        }
        (CpuBatchSamples::I32(actual), DecodedSamples::I32(expected)) => {
            compare_repeated!(actual, expected);
        }
        (CpuBatchSamples::F32(actual), DecodedSamples::F32(expected)) => {
            assert_eq!(actual.len(), expected.len() * 2);
            assert_f32_bits(&actual[..expected.len()], expected);
            assert_f32_bits(&actual[expected.len()..], expected);
        }
        (CpuBatchSamples::Rgb101010(actual), DecodedSamples::Rgb101010(expected))
        | (CpuBatchSamples::Rgbe(actual), DecodedSamples::Rgbe(expected)) => {
            compare_repeated!(actual, expected);
        }
        _ => panic!("batch storage differs from successful owned decode"),
    }
}

fn storage_elements(samples: &DecodedSamples) -> usize {
    match samples {
        DecodedSamples::BitPacked(values) | DecodedSamples::U8(values) => values.len(),
        DecodedSamples::U16(values)
        | DecodedSamples::F16(values)
        | DecodedSamples::Rgb555(values)
        | DecodedSamples::Rgb565(values) => values.len(),
        DecodedSamples::I16(values) => values.len(),
        DecodedSamples::I32(values) => values.len(),
        DecodedSamples::F32(values) => values.len(),
        DecodedSamples::Rgb101010(values) | DecodedSamples::Rgbe(values) => values.len(),
    }
}

fn assert_f32_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}
