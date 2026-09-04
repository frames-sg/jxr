# Architecture

`jxr` is the public facade. `jxr-core` owns stable contracts, `jxr-math` owns
portable exact arithmetic, and `jxr-native` owns parsing and CPU decoding.
`jxr-metal` depends inward on those crates and owns device-specific sessions,
storage, and kernels.

CPU parsing and entropy state never live in an accelerator crate. Device crates
never implement a second parser. Explicit device requests are strict; automatic
routing may choose CPU only before a device submission begins.

Crate roots contain declarations and re-exports. Modules are split by codec
stage or resource lifetime, not by arbitrary helper categories.

## Decode flow

1. `jxr-native` discovers raw or Annex-A ranges, parses headers and the tile
   directory, applies request limits, and builds a `PreparedPlan`.
2. Independently located tiles decode in parallel. Frequency-mode native reduced
   requests omit higher packet bands before entropy decoding and allocation.
   Adaptive entropy, scans, and DC/LP prediction remain raster-ordered inside each tile. HP direction is
   selected there, while the coefficient operation is deferred to reconstruction.
3. Coefficients are written once into packed macroblock-major storage with
   structure-of-arrays metadata and one contiguous range per component.
   `PreparedReconstruction` retains this handoff for an accelerator without
   reparsing or re-running entropy work. Supported integrated alpha is appended
   as a distinct plane after the primary components while sharing the packet walk.
4. CPU reconstruction expands only the requested tile/halo window, restores
   local raster order with coded-plane origins, applies HP prediction through
   `jxr-math`, and runs sampling-specific first-level transforms and overlap
   operators followed by the common second level. Independent planes and large
   macroblock phases run in parallel. A session-retained capability token selects
   safe AVX2/NEON HP dequantization and common U8 stores; scalar math remains the
   oracle and fallback.
5. Metal and CUDA plans retain the same coefficient contract. Their sessions own device
   state, allocation pools, submissions, completion, resident
   output, and host readback. Metal preparation can write entropy results directly
   into shared allocation slices; compatible batches bind those slices without
   repacking and use macroblock × image × component transform grids. CUDA keeps
   one immutable upload per coefficient identity and stream, submits the
   equivalent five-phase pipeline, and retains bounded device scratch and
   resident outputs without requiring CUDA at CPU-only build time.

## Route model

The provisional promotion point is 16,384 reconstructed coefficients for
Metal and CUDA. CPU remains the route below the threshold or when a compatible device
is unavailable. Explicit device requests are strict. Automatic fallback is
legal only before submission; an error after a command is submitted is returned
to the caller. Native reduced requests currently stay on CPU; explicit Metal or
CUDA requests for those scales fail before accelerator preparation.

The thresholds are algorithmic starting points, not benchmark claims. They stay
isolated in the backend route modules so later end-to-end measurements can
replace them without changing entropy or reconstruction code.

## Owned batch flow

`PreparedJxr` owns parsed compressed metadata behind a shared allocation.
`PreparedImage` adds one validated request plan and native output contract, while
`PreparedBatch` groups compatible images in stable first-occurrence order. This
layer is device-neutral: input-local preparation errors remain indexed rather
than failing valid siblings.

`CpuBatchDecoder` retains an image-level worker pool, independent primary and
separate-alpha coefficient arenas, component-raster and inverse-transform
scratch, and recyclable signed component planes per worker. A bounded LRU-style
preparation cache is keyed by compressed `Arc` identity plus the complete
request; duplicate prepared images share an optional coefficient-ready
`PreparedReconstruction`. Native-layout typed output stores write directly into
the final dense group allocation, including independently encoded alpha. NCHW is
a CPU-only tensor policy implemented by a type-matched transpose in retained
worker scratch after the same direct store. Exact typed caller destinations
retain one fixed slot per prepared image so a sibling error does not move
another image's slot.

`MetalBatchDecoder` prepares coefficient slices concurrently from the same
groups. High-level submission may return immediately, complete to per-image
resident allocations, or retain one private `MetalResidentBatch` allocation per
homogeneous group. Caller-owned dense destinations require a one-queue session;
the pending owner retains exclusive destination access through completion.
`jxr-mpsgraph` accepts the shared native-layout prepared contract, then owns the
additional coefficient and dense-allocation lifetime required by its exact-queue
NHWC graph handoff.

`CudaBatchDecoder` consumes the same groups and coefficient-ready cache. It
schedules independent images over four streams and can retain per-image outputs,
one homogeneous dense device allocation, or an exclusive caller-owned
destination without host pixel staging.

`jxr-image` is an outward ownership adapter. It depends on `jxr`, transfers only
single-plane formats that `image::DynamicImage` represents exactly, and retains
JPEG XR ICC, alpha, and route semantics in a wrapper. Neither `jxr-core` nor the
decoder crates depend on `image`.

## Container write flow

The public `jxr::write_annex_a` facade delegates to the Annex-A module beside
the native reader. The writer accepts existing raw T.832 primary and optional
separate-alpha codestreams, fully parses them, verifies their dimensions, emits
sorted typed directory entries and aligned payloads, then parses the completed
file again. Pixel-to-codestream encoding remains a separate boundary; the
container writer does not accept decoded pixels or synthesize entropy syntax.

## Canonical math and ABI

`jxr-math` is `no_std` and owns checked integer transforms, overlap operators,
dequantization, HP prediction, interpolation, color, alpha, and RGBE arithmetic.
Metal uses backend-local MSL and CUDA uses backend-local CUDA C compiled through
NVRTC; their current
HP phase consumes compact CPU-selected directions and resolves prediction with
race-free per-block prefix sums before dequantization. Transform constants and
parameter indices are generated from `data/reconstruction.abi`, which also
generates the Rust bindings used by both device encoders. All accelerator
manifests describe the complete packaged Main-profile pipeline.
