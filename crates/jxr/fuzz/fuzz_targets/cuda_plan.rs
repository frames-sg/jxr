#![no_main]

use jxr::{BackendRequest, DecodeLimits, DecodeRequest, JxrView, PixelFormat};
use jxr_test_support::oracle_format;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|bytes: &[u8]| {
    let Ok(view) = JxrView::parse(bytes) else {
        return;
    };
    let (pixel_format, alpha) = oracle_format(view.info())
        .map(|format| (format.pixel_format, format.alpha))
        .unwrap_or((
            PixelFormat::U8(jxr::ChannelLayout::Luma),
            jxr::AlphaHandling::Drop,
        ));
    let request = DecodeRequest::new(pixel_format)
        .with_alpha(alpha)
        .with_backend(BackendRequest::Cuda)
        .with_limits(DecodeLimits {
            max_width: 2048,
            max_height: 2048,
            max_pixels: 4 * 1024 * 1024,
            max_components: 16,
            max_tiles: 4096,
            max_compressed_bytes: 16 * 1024 * 1024,
            max_coefficient_bytes: 128 * 1024 * 1024,
            max_host_allocation_bytes: 128 * 1024 * 1024,
        });
    if let Ok(prepared) = view.decoder().prepare_reconstruction(&request) {
        let _ = prepared.cuda_plan();
    }
});
