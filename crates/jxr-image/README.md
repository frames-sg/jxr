# JXR image adapter

`jxr-image` transfers supported owned JPEG XR decode buffers into
`image::DynamicImage` without copying pixel samples. It targets `image` 0.25.10
with default codec features disabled, so using the adapter does not enable
another image decoder stack.

```rust,no_run
use jxr::{AlphaHandling, ChannelLayout, DecodeRequest, JxrView, PixelFormat};

# fn decode(bytes: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
let view = JxrView::parse(bytes)?;
let request = DecodeRequest::new(PixelFormat::U16(ChannelLayout::Rgb))
    .with_alpha(AlphaHandling::Drop);
let frame = jxr_image::decode_view(&view, &request)?;
assert_eq!(frame.image().width(), frame.decoded_region().w);
# Ok(())
# }
```

The exact zero-copy mappings are Luma/LumaA/RGB/RGBA at `U8` and `U16`, plus
RGB/RGBA at `F32`. Planar YUV, BGR/BGRA, padded RGBX/BGRX, signed and fixed-point
samples, half floats, and packed RGB/RGBE are rejected because `DynamicImage`
cannot represent them without a pixel conversion.

`ImageFrame` retains the JPEG XR `ImageInfo`, decoded region, route report, ICC
profile bytes, exact source output format, and whether alpha is straight or
premultiplied. The adapter performs no ICC transform. Because JPEG XR does not
provide a CICP declaration, the `DynamicImage` primaries and transfer function
are set to `Unspecified` instead of silently claiming sRGB.
