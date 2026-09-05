# JXR

JXR is a JPEG XR implementation with a safe Rust CPU decoder, optional Metal or
CUDA reconstruction, and a validated Annex-A still-image container writer.

The project is under active development. Its CPU route byte-matches T.835 for
all 517 T.834 cases in the repository's Main-syntax scope; Advanced-only output
syntax and JPEG 2000 `.jpx` wrappers remain outside that scope. Measured CUDA results and their limits are recorded in the
[2026-09-05 hardware report](docs/cuda-validation-2026-09-05.md); no general
CPU-versus-GPU speedup is claimed.

## Architecture

Container parsing, entropy decoding, adaptive scans, coefficient remapping,
and DC/LP prediction run on the CPU. Regular reconstruction stages—dequantization,
inverse transforms, overlap filtering, color conversion, and packing—have
backend-local Metal phase code and resource lifecycles. Metal submission is
enabled for the complete five-phase pipeline.

The default `jxr` build is CPU-only. The `metal` and `cuda` features add
accelerator integration without changing decoded image semantics or requiring
those runtimes for a CPU-only build.

The CUDA backend consumes the same CPU-produced coefficient contract as Metal
and includes reusable sessions, four streams, bounded scratch and immutable
upload caches, asynchronous and resident output, checked caller-owned device
destinations, homogeneous dense batches, and all typed/packed store paths. It is
compiled through dynamically loaded Driver API/NVRTC bindings, so an all-feature
build does not need a CUDA toolkit or NVIDIA driver. The self-hosted NVIDIA
workflow passed on 2026-09-05, including all 517 in-scope T.834/T.835 comparisons,
CPU/ROI differential tests, and checksum-checked benchmarks. See the
[hardware report](docs/cuda-validation-2026-09-05.md) and
[CUDA backend decision and validation boundary](docs/cuda-backend.md).

## Current usable slice

The connected CPU path decodes spatial- and frequency-mode Y-only, YUV420,
YUV422, YUV444, YUVK, and Main-profile 2–16 channel N-component syntax across
all band-presence modes. It carries compact, sampling-aware per-component
quantizers and prediction metadata through a macroblock-major arena,
prunes region requests to the required tile/halo window, reconstructs
independent color planes and macroblock transform phases in parallel, and feeds exact scalar
chroma reconstruction, color, alpha, crop, and typed packing stages. Hard and
soft tiles, overlap modes 0–2, supported integer/float depths, and matching
separate Annex-A alpha are represented by the current pipeline. Native planar
YUV420/YUV422, BD10 YUV, CMYK/N-component alpha, and session-selected safe
AVX2/NEON HP dequantization and common U8 packing are connected.
Frequency-mode inputs also support true native quarter-width/height DC+LP decode
and sixteenth-width/height DC-only decode; unneeded packet bands are not entropy
decoded. These reduced routes are currently CPU-only.

Annex-A inspection retains the raw pixel-format GUID and also reports a typed
classification for every T.832 Table A.6 value, with known formats checked
against the codestream's color, depth, component, alpha, premultiplication, and
RGB/BGR padding declarations.

```rust,no_run
use jxr::{ChannelLayout, DecodeRequest, DecodeScale, JxrView, PixelFormat};

# fn decode(bytes: &[u8]) -> Result<(), jxr::JxrError> {
let view = JxrView::parse(bytes)?;
let request = DecodeRequest::new(PixelFormat::U8(ChannelLayout::Luma));
let mut decoder = view.decoder();
let image = decoder.decode(&request)?;
assert_eq!(image.decoded_region.w, view.info().width);

let thumbnail = decoder.decode(&request.clone().with_scale(DecodeScale::Sixteenth))?;
assert_eq!(thumbnail.decoded_region.w, view.info().width.div_ceil(16));
# Ok(())
# }
```

Integrated alpha is connected for every implemented primary layout in spatial
and frequency modes. On M1-or-newer Apple GPUs, Metal reconstruction supports
multi-plane transforms, hard/soft overlap, chroma and color conversion, integrated
or separate alpha, native planar and typed/packed output, host readback, resident
output, checked external destinations, and bounded batch submission. Default-device
batch submission schedules independent images over four Metal queues; caller-supplied
queues remain exact and ordered. CPU entropy output
can be written directly into per-image slices of one shared Metal allocation.
Compatible image groups reuse that allocation, concatenate descriptors, and run
macroblock × image × component transform grids without a coefficient repack.
Batch host output writes directly into pooled shared buffers without
private-output blits, and `decode_batch_to_shared` exposes completed bytes without
copying them into Rust-owned samples. The current CPU T.834/T.835 differential
run passes all 517 in-scope cases. Metal byte-matches every affected baseline and
Main output-format case, including padded RGB/BGR and RGBE storage.

The standalone `jxr-mpsgraph` crate adds an Apple-silicon NHWC tensor route for
homogeneous Gray, RGB, and RGBA `U8`, `U16`, and `I16` batches. It keeps JPEG XR
parsing and entropy on the CPU, exact reconstruction in the existing Metal
kernels, and codec plus graph work on one caller-owned command queue. The
application does not read decoded pixels back or upload them again on this path;
framework-internal zero-copy is deliberately not claimed without Metal capture.

The standalone `jxr-image` crate transfers exact Luma/LumaA/RGB/RGBA `U8` and
`U16` output, plus RGB/RGBA `F32`, into `image::DynamicImage` without copying
pixel storage. Its `ImageFrame` keeps ICC bytes, decode metadata, and straight
versus premultiplied alpha outside `DynamicImage`; unsupported layouts are
rejected instead of being silently converted.

`write_annex_a` wraps an existing raw T.832 codestream in a deterministic
Annex-A file. It validates the primary and optional separate-alpha codestreams,
their declared dimensions, display resolution, orientation, ICC payload, and
known pixel-format GUID consistency before returning output. This is container
serialization, not pixel-to-codestream encoding.

## High-throughput native batches

The shared owned-batch API accepts `EncodedImage` values containing `Arc<[u8]>`
storage and a complete `DecodeRequest`. Preparation parses and plans inputs in a
retained worker pool, groups matching native output contracts without padding,
and preserves input-local failures by original index. `PreparedBatch` is cheaply
cloneable and can be decoded repeatedly without reparsing or replanning.

`CpuBatchDecoder` returns one contiguous typed `CpuBatchSamples` owner per
homogeneous group. Native-layout bit-packed, integer, floating-point, packed RGB,
RGBE, and planar YUV outputs are reconstructed and packed directly into that
final allocation, including independently encoded alpha. Every validated unpacked
NCHW format uses the same direct store followed by a typed transpose in
worker-retained scratch. Each CPU worker also keeps primary and separate-alpha
coefficient arenas, component-raster buffers, inverse-transform scratch, and
recyclable signed component planes across calls, while preparation keeps a
bounded identity-and-request cache so repeated `Arc<[u8]>` inputs share parsing,
planning, and optional coefficient-ready state.
`CpuBatchDiagnostics` reports cache, direct-store, compaction, and retained
workspace counters. CPU groups can also write exact caller-owned typed
destinations.

`MetalBatchDecoder` consumes the same prepared groups. It supports nonblocking
high-level submission, completed per-image resident output, or one private
`MetalResidentBatch` allocation per homogeneous group. Exact-queue sessions can
submit a prepared group into a caller-owned dense Metal destination while the
submission retains exclusive access through completion. The MPSGraph adapter
continues to use native NHWC contracts on its exact command queue.

Annex-A ICC bytes are available without copying through `JxrView::icc_profile`
and `PreparedJxr::icc_profile`. `JxrDecoder::decode_oriented` applies declared
presentation orientation for supported full-image single-plane host layouts,
including bit-packed luma. Color-profile application remains the responsibility
of a color-management system; the codec does not approximate ICC transforms.

## License

Licensed under either Apache-2.0 or MIT at your option. Third-party table data,
when introduced, is tracked in `THIRD_PARTY_NOTICES.md`.
