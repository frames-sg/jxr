# JPEG XR support boundary

The intended boundary is raw T.832 codestream and Annex-A still-image decode for
Main-profile syntax, validated Annex-A serialization around existing raw
codestreams. Annex-F/HEIF sequence storage, T.832 codestream encoding, and
transcoding are out of scope.

The current CPU route byte-matches T.835 for all 517 T.834 cases classified as
in-scope Main syntax. The remaining 179 corpus entries are explicitly skipped
because they use Advanced-only output syntax or a JPEG 2000 `.jpx` wrapper.

## Implemented now

- Borrowed raw-codestream and Annex-A inspection, checked ranges, tile index,
  profile/level, orientation, zero-copy ICC bytes, primary plane, and separate alpha.
- Complete typed classification of the Annex-A Table A.6 pixel GUIDs, including
  RGB/BGR storage padding and premultiplied separate alpha, lossless retention
  of unknown GUIDs, and consistency checks against parsed syntax.
- Spatial and indexed frequency-mode Y-only, YUV420, YUV422, YUV444, YUVK, and
  Main-profile 2–16 channel N-component entropy decoding for DC-only, no-HP,
  no-flexbits, and all bands.
- Tile quantizers, adaptive resets, scans, coefficient prediction, flexbits,
  compact subsampled coefficient storage, exact sampling-specific inverse
  transforms, hard/soft tile reconstruction, overlap modes 0–2, chroma
  reconstruction, crop halos, and typed output.
- Component-separated coefficient arenas, reconstruction-phase HP prediction,
  tile/halo-pruned region decode, and parallel CPU reconstruction for independent
  planes and macroblock transform phases.
- Frequency-mode native quarter-width/height DC+LP reconstruction and
  sixteenth-width/height DC-only reconstruction, with higher packet bands
  excluded from entropy decode and coefficient allocation.
- Typed output packing for bit-packed, integer, F16/F32, packed RGB, RGBE,
  native planar YUV420/YUV422, BD10 YUV, and alpha-bearing CMYK/N-component
  families.
- Per-session safe AVX2/NEON capability tokens for HP dequantization and common
  U8 packing, with the exact scalar implementation retained as the oracle and
  portable fallback.
- Separate-alpha preservation and typed scalar premultiplication for unpacked
  integer and floating-point channel layouts.
- Integrated-alpha spatial and frequency packet interleaving for every implemented
  primary layout, with independent alpha entropy state, band presence, quantizers,
  coefficient storage, reconstruction, plane-local output scaling, preservation,
  and premultiplication.
- Direct CPU entropy writes into shared Metal coefficient storage, complete
  Apple-silicon Metal reconstruction, precise MSL compilation, checked external
  destinations, resident and zero-copy shared host output, bounded aggregate
  descriptor batches, scratch pools, and immutable fallback upload caching.
- Shared owned batch preparation, stable homogeneous grouping, indexed sibling
  failures, identity/request plan caching, optional coefficient-ready caching,
  direct final-allocation typed CPU stores, caller-owned typed CPU output, CPU
  NCHW layout, direct independently encoded alpha output, retained coefficient,
  component-raster, inverse-transform, signed component-plane, and typed layout
  workspaces, nonblocking Metal submission, single-allocation Metal resident
  groups, exact-queue caller-owned Metal output, and shared-batch MPSGraph
  handoff.
- Explicit full-image host orientation application for supported single-plane
  typed, packed, and one-bit luma outputs, plus bounded parser/decode and
  typed-output/native-batch differential fuzz targets.
- A standalone zero-copy `image::DynamicImage` ownership adapter for exact
  Luma/LumaA/RGB/RGBA `U8` and `U16` plus RGB/RGBA `F32`, retaining ICC and alpha
  semantics outside the upstream image type.
- Deterministic Annex-A still-image writing for validated raw primary and
  separate-alpha codestreams, including typed orientation, finite positive
  display resolution, exact ICC bytes, four-byte-aligned payloads, and a final
  full parser/format-consistency validation pass.
- CPU CI on Linux/macOS and an opt-in, serialized self-hosted Apple-silicon
  workflow covering strict workspace checks, Metal, MPSGraph, T.834/T.835
  differential runs, the pathology benchmark, and retained diagnostic artifacts.
- Optional dynamically loaded CUDA reconstruction with reusable contexts and
  streams, bounded scratch and immutable uploads, asynchronous/resident output,
  checked caller-owned destinations, native and dense batches, an ignored full
  CPU differential suite, a serialized self-hosted NVIDIA workflow, and a
  phase-separated pathology benchmark. CUDA hardware results are not yet
  available from the current development Mac.

## Explicitly incomplete

- A wider device performance corpus. CPU passes all 517 in-scope T.834/T.835
  differential cases; the affected Metal output-format categories also pass
  byte-for-byte. The hardware commands pass locally on Apple silicon; GitHub
  dispatch remains unverified until a matching labeled runner is provisioned.
- Native reduced-resolution decode for spatially interleaved packets and for
  Metal or CUDA reconstruction. The implemented frequency-mode CPU route is genuinely
  band-limited and does not label full decode plus resampling as native reduction.
- CUDA runtime, NVRTC kernel compilation, full conformance, and performance have
  not been validated on compatible NVIDIA hardware yet. No CUDA correctness or
  speed claim is made until the self-hosted workflow passes and retains reports.
- ICC color transforms. Profiles are exposed byte-for-byte for a caller-selected
  color-management system and are never approximated inside the codec.
Unsupported combinations return errors; they are not approximated by a nearby
color format or silently retried after device submission.
