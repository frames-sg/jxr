# JXR: JPEG XR for Rust

[![crates.io](https://img.shields.io/crates/v/jxr.svg)](https://crates.io/crates/jxr)
[![docs.rs](https://img.shields.io/docsrs/jxr)](https://docs.rs/jxr)
[![CI](https://github.com/frames-sg/jxr/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/frames-sg/jxr/actions/workflows/ci.yml?query=branch%3Amain+event%3Apush)
[![downloads](https://img.shields.io/crates/d/jxr.svg)](https://crates.io/crates/jxr)
[![license](https://img.shields.io/crates/l/jxr.svg)](#license)

JXR decodes JPEG XR images in Rust, with optional Apple GPU (Metal) and
NVIDIA GPU (CUDA) reconstruction. The default build runs on the CPU and needs
no GPU runtime or reference codec.

It reads raw JPEG XR codestreams and Annex-A still-image files, decodes full
images or regions, and preserves high-bit-depth samples, alpha, and embedded
ICC profiles. It can also wrap an existing codestream in an Annex-A file;
pixel-to-codestream encoding is not implemented.

```sh
cargo add jxr
```

Requires **Rust 1.99.0 or newer**. The current release is **0.4.0**; see the
[release notes](docs/releasing.md#040-release-line) for migration details.

## What it does

- **Inspect without decoding pixels.** Read dimensions, color format, bit
  depth, tiling, orientation, alpha, and container metadata through `JxrView`.
- **Decode a whole image or a region.** Region requests limit reconstruction
  to the required tiles and overlap halo.
- **Decode thumbnails at native resolution.** Frequency-mode images support
  quarter-width/height and sixteenth-width/height CPU decode, skipping the
  higher-frequency bands that the output does not need.
- **Keep the sample representation you need.** Supported outputs include
  grayscale, RGB/BGR, CMYK, multichannel, planar YUV, integer and floating-point
  samples, packed RGB/RGBE, and integrated or separate alpha.
- **Decode batches.** Prepare inputs once, group compatible output shapes,
  and reuse CPU workers or GPU sessions. CPU batches support native and NCHW
  layouts; GPU output can remain in device memory.
- **Connect to other libraries.** `jxr-image` transfers supported buffers into
  `image::DynamicImage`; `jxr-mpsgraph` hands Metal batches to Apple's MPSGraph.
- **Write Annex-A containers.** `write_annex_a` packages an existing raw
  codestream with validated metadata and optional separate alpha.

The supported format boundary is **Main-profile syntax**. Advanced-only output
syntax, JPEG 2000 `.jpx` wrappers, Annex-F/HEIF sequences, and transcoding are
outside the scope. See the [full support guide](docs/support.md).

## Quick start

Decode a compatible 8-bit color image to RGB, explicitly dropping any alpha:

```rust
use jxr::{AlphaHandling, ChannelLayout, DecodeRequest, JxrView, PixelFormat};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read("image.jxr")?;
    let view = JxrView::parse(&bytes)?;
    let request = DecodeRequest::new(PixelFormat::U8(ChannelLayout::Rgb))
        .with_alpha(AlphaHandling::Drop);
    let mut decoder = view.decoder();
    let image = decoder.decode(&request)?;

    println!(
        "{}x{}, {:?}, {} bytes",
        image.decoded_region.w,
        image.decoded_region.h,
        image.format,
        image.samples.byte_len(),
    );
    Ok(())
}
```

`image.samples` holds typed pixel storage, and `image.planes` describes its
layout. ICC bytes are available through `view.icc_profile()`; applying the
profile is the application's responsibility. Use `decode_oriented` to apply
the file's presentation orientation for supported full-image host outputs.

### Choose an output format

For an unfamiliar file, inspect `view.info()` before choosing a request. Replace
the request and decode portion of `main` above with this to select grayscale or
RGB output for common unsigned 8- and 16-bit sources, retaining alpha:

```rust
use jxr::{AlphaMode, ColorFormat, SampleFormat};

let info = view.info();
println!("Source: {:?}, {:?}, alpha: {:?}",
    info.primary.color_format, info.primary.sample_format, info.alpha_mode);

let has_alpha = info.alpha_mode != AlphaMode::None;
let layout = match (info.primary.color_format, has_alpha) {
    (ColorFormat::Luma, false) => ChannelLayout::Luma,
    (ColorFormat::Luma, true) => ChannelLayout::LumaAlpha,
    (ColorFormat::Rgb | ColorFormat::Yuv(_), false) => ChannelLayout::Rgb,
    (ColorFormat::Rgb | ColorFormat::Yuv(_), true) => ChannelLayout::Rgba,
    _ => return Err("Choose an explicit output layout for this color format".into()),
};
let output = match info.primary.sample_format {
    SampleFormat::Unsigned { bits: 8 } => PixelFormat::U8(layout),
    SampleFormat::Unsigned { bits: 16 } => PixelFormat::U16(layout),
    _ => return Err("Choose an explicit output format for this sample depth".into()),
};
let request = DecodeRequest::new(output); // Preserve alpha by default.
let image = view.decoder().decode(&request)?;
```

`primary.color_format` describes the coded planes: an RGB image commonly uses
YUV internally, which is why both branches request RGB output. The container's
declared pixel representation, when present, is available separately through
`info.metadata.annex_a_pixel_format`.

This example handles a common subset, not every JXR output. CMYK, multichannel,
signed, floating-point, and packed formats need an explicit `PixelFormat`;
see the [supported formats](docs/support.md). Decoding validates the full
source/output combination and returns an error if it is unsupported.

### Use with `image::DynamicImage`

Add the adapter alongside `jxr`:

```sh
cargo add jxr-image
```

Using the `view` and format-aware `request` above, decode into the `image` crate
and create a preview that fits within 256 × 256 pixels:

```rust
let frame = jxr_image::decode_view(&view, &request)?;
let image: &jxr_image::image::DynamicImage = frame.image();
let preview = image.thumbnail(256, 256);
println!("Preview: {}x{}", preview.width(), preview.height());
```

Use this in place of the preceding `decoder.decode` call to avoid decoding
twice. The adapter transfers the decoded pixel allocation without copying it;
`thumbnail` then creates a resized image after a full decode. For frequency-mode
inputs, the native thumbnail route below can skip higher-frequency bands.

Keep `frame` when you need its ICC profile or straight/premultiplied alpha
metadata; `frame.into_image()` returns only the `DynamicImage` and discards that
extra metadata. The adapter does not apply ICC transforms or presentation
orientation. It supports Luma/LumaA/RGB/RGBA at `U8` and `U16`, plus RGB/RGBA at
`F32`. See the [adapter guide](crates/jxr-image/README.md) for details.

### Regions and thumbnails

With the same `view` and `request`, decode a 256 × 256 region within the image:

```rust
use jxr::Rect;

let region = Rect { x: 0, y: 0, w: 256, h: 256 };
let crop = view.decoder().decode(&request.clone().with_region(region))?;
```

For a frequency-mode image, decode a thumbnail at one sixteenth of the width
and height:

```rust
use jxr::{BackendRequest, DecodeScale};

let thumbnail_request = request.clone()
    .with_scale(DecodeScale::Sixteenth)
    .with_backend(BackendRequest::Cpu);
let thumbnail = view.decoder().decode(&thumbnail_request)?;
```

`DecodeScale::Quarter` is also available. Region coordinates remain in
full-resolution pixels. Native reduced decode currently requires frequency-mode
input and the CPU backend; spatial-mode and GPU reduction are not implemented.

### Inspect a file

From a repository checkout:

```sh
cargo run -p jxr --example inspect -- image.jxr
```

This prints parsed metadata without decoding pixels. JXR currently provides a
library and this inspection example, rather than a standalone conversion CLI.

## GPU support

The CPU parses the container and decodes entropy; Metal or CUDA can run the
reconstruction and output stages. Both support host output, resident device
buffers, caller-owned destinations, and batches.

| Backend | Enable | Runtime requirements |
| --- | --- | --- |
| CPU | Default; no features needed | No GPU runtime |
| Metal | `cargo add jxr --features metal` | Apple-silicon macOS (M1 or newer) |
| CUDA | `cargo add jxr --features cuda` | Compatible NVIDIA GPU and driver; CUDA Oxide kernels built on Linux (see guide) |

For single-image GPU decoding, attach a reusable session to the decoder;
enabling a feature alone does not attach one. See the
[Metal and CUDA session example](docs/architecture.md#attach-a-gpu-session).
Building the CUDA feature does not require CUDA libraries or an NVIDIA driver.

`BackendRequest::Cpu` keeps work on the CPU. The default `Auto` may use an
attached GPU session or choose the CPU before submission. Explicit `Metal` and
`Cuda` requests return an error when unavailable or unsupported. GPU failures
after submission are returned without a CPU retry.

Automatic routing uses workload thresholds (measured for Metal, provisional for
CUDA). **No general CPU-versus-GPU speedup is claimed.** See the [routing model](docs/architecture.md#route-model),
[CUDA backend guide](docs/cuda-backend.md), and hardware reports for
[release 0.3.0](https://github.com/frames-sg/jxr/releases/tag/v0.3.0).

## Which crate do I need?

Most applications only need `jxr`.

| You want to | Use |
| --- | --- |
| Inspect or decode JPEG XR; wrap existing codestreams | [`jxr`](https://docs.rs/jxr) |
| Decode with an Apple or NVIDIA GPU | `jxr` with `metal` or `cuda` |
| Transfer decoded pixels to `image::DynamicImage` | [`jxr-image`](crates/jxr-image/README.md) |
| Feed Metal-resident batches to MPSGraph | [`jxr-mpsgraph`](crates/jxr-mpsgraph/README.md) |
| Manage device sessions and buffers directly | [`jxr-metal`](https://docs.rs/jxr-metal) or [`jxr-cuda`](https://docs.rs/jxr-cuda) |

`jxr-core`, `jxr-native`, and `jxr-math` provide the shared types, CPU engine,
and exact arithmetic behind those APIs. `jxr-test-support` is an unpublished
validation package. See [architecture](docs/architecture.md) for how they fit
together.

## Correctness and limits

**Reference comparisons.** For release 0.3.0, CPU, Metal, and CUDA output each
byte-matched the T.835 reference software on all **517 in-scope T.834 cases**.
Annex-A writer comparisons passed the same cases. Another 179 entries were
excluded for Advanced-only output syntax or `.jpx` wrappers. This is scoped
differential evidence, not a claim of complete JPEG XR conformance. The
[release notes](docs/releasing.md#030-release-line) link to the validation runs.

**Input handling.** The public facade forbids unsafe Rust. Decode requests
include configurable `DecodeLimits` for dimensions, compressed input,
coefficients, and host allocations. Parser, decode, and output/batch fuzz targets
live in the repository; ordinary CI checks that the fuzz package builds.

**Color and metadata.** JXR exposes ICC profiles without applying color
transforms. It preserves explicit alpha and output-format semantics, and rejects
unsupported combinations instead of approximating another format.

**API status.** JXR is under active development on the `0.x` release line.
The 0.4 line depends on J2K 0.13.1 or a later 0.13 patch release; applications
using the shared J2K types must update those dependencies together. Check the release notes
before upgrading.

## Documentation and contributing

- [API reference](https://docs.rs/jxr)
- [Supported formats and known limitations](docs/support.md)
- [Architecture and batch decoding](docs/architecture.md)
- [CUDA backend and validation](docs/cuda-backend.md)
- [`image` integration](crates/jxr-image/README.md)
- [MPSGraph integration](crates/jxr-mpsgraph/README.md)
- [Release history and migration notes](docs/releasing.md)
- [Contributing, local checks, and bug reports](CONTRIBUTING.md)
- [Issues](https://github.com/frames-sg/jxr/issues)

## License

Licensed under either [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your
option. See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and
reference-software boundaries.
