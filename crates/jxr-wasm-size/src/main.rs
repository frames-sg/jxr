//! The decoder as a Flash player links it into WebAssembly: what
//! `tools/wasm_size_check.py` builds and measures.
//!
//! It decodes a file it cannot see at compile time, the way Ruffle decodes
//! JPEG XR images: parsed, then decoded to the eight- or sixteen-bit grey,
//! grey and alpha, RGB or RGBA layout the header calls for. Everything a
//! decode of any such file can reach is linked, and nothing else: no
//! encoder, and no thread pool, which WebAssembly without the `atomics`
//! feature cannot start. Run natively it decodes nothing and exits.

use std::hint::black_box;

use jxr::{
    AlphaMode, ChannelLayout, ColorFormat, DecodeRequest, JxrView, PixelFormat, SampleFormat,
};

fn main() {
    let file: Vec<u8> = black_box(Vec::new());
    let Ok(view) = JxrView::parse(&file) else {
        return;
    };
    let info = view.info();
    let grey = info.primary.color_format == ColorFormat::Luma;
    let alpha = info.alpha_mode != AlphaMode::None;
    let layout = match (grey, alpha) {
        (true, false) => ChannelLayout::Luma,
        (true, true) => ChannelLayout::LumaAlpha,
        (false, false) => ChannelLayout::Rgb,
        (false, true) => ChannelLayout::Rgba,
    };
    let format = match info.primary.sample_format {
        SampleFormat::Unsigned { bits } if bits > 8 => PixelFormat::U16(layout),
        _ => PixelFormat::U8(layout),
    };
    // What is measured is what a decode links; whether this one succeeds
    // does not change that.
    let decoded = view.decoder().decode(&DecodeRequest::new(format));
    black_box(&decoded);
}
