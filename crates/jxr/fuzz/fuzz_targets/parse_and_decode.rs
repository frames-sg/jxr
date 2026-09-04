#![no_main]

use jxr::{
    BackendRequest, ChannelLayout, DecodeLimits, DecodeRequest, DecodeScale, JxrView, PixelFormat,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|bytes: &[u8]| {
    let Ok(view) = JxrView::parse(bytes) else {
        return;
    };
    let limits = DecodeLimits {
        max_width: 4096,
        max_height: 4096,
        max_pixels: 16 * 1024 * 1024,
        max_components: 16,
        max_tiles: 4096,
        max_compressed_bytes: 16 * 1024 * 1024,
        max_coefficient_bytes: 256 * 1024 * 1024,
        max_host_allocation_bytes: 256 * 1024 * 1024,
    };
    let scale = match bytes.len() % 3 {
        0 => DecodeScale::Full,
        1 => DecodeScale::Quarter,
        _ => DecodeScale::Sixteenth,
    };
    let request = DecodeRequest::new(PixelFormat::U8(ChannelLayout::Luma))
        .with_scale(scale)
        .with_backend(BackendRequest::Cpu)
        .with_limits(limits);
    let _ = view.decoder().decode(&request);
});
