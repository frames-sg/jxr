# CUDA reconstruction backend

## Repository-grounded architecture audit

The implementation boundary follows the ownership already established by the
workspace:

- `jxr-core` owns `DecodeRequest`, backend request/result reporting,
  `PreparedPlan`, macroblock-major `CoefficientArena`, typed output contracts,
  and checked `SurfaceLayout`. Those types are the device-neutral ABI; CUDA
  adds no parser or codec data model.
- `jxr-native` owns raw T.832 and Annex-A parsing, entropy decoding, DC/LP
  prediction, sparse ROI coefficient production, the scalar reconstruction
  oracle, output-policy validation, and typed host packing. CUDA begins only
  after `PreparedReconstruction` has obtained these validated coefficients.
- `jxr` owns public routing and native batch contracts. Its prepared-image
  cache shares one coefficient handoff among duplicate inputs; homogeneous
  groups preserve original indices, per-image metadata, ROI regions, and
  group-local errors. Explicit CUDA selection is rejected when the feature or
  an attached session is absent, while `Auto` can remain on CPU before a
  submission exists.
- `jxr-metal` established the five reconstruction phases, resident/caller-owned
  outputs, asynchronous completion, reusable allocation and upload caches, and
  dense batches. The CUDA crate mirrors those responsibility and lifecycle
  boundaries, while using CUDA contexts, nonblocking streams, events, and
  device allocations rather than exposing Metal types or duplicating its
  platform adapter.
- `jxr-math/data/reconstruction.abi` is the canonical transform permutation and
  descriptor schema. Its build script emits MSL and CUDA Oxide Rust descriptors;
  Rust layout assertions keep the host descriptors tied to the same source.
- Existing tests place scalar math in `jxr-math`, syntax/reconstruction/output
  behavior in `jxr-native`, public routing and batches in `jxr`, and official
  T.834/T.835 discovery/comparison in `jxr-test-support`. CUDA therefore adds
  device-free contract/preflight tests beside the adapter and ignored device
  differential tests beside the existing conformance harness.
- Regular CI already exercises formatting, workspace Clippy/tests, minimal
  features, strict rustdoc, and fuzz-package compilation on Linux and macOS.
  Metal hardware validation is a serialized self-hosted workflow. CUDA follows
  that split: dynamic-loading builds stay in regular CI, while CUDA Oxide compilation, device execution, the 517-case differential suite, and benchmarks are
  confined to the serialized NVIDIA runner.
- The existing pathology benchmarks separate preparation, submission,
  synchronization, and readback while checking output checksums. The CUDA
  benchmark retains that measurement model and adds transfer and device-pool
  counters across the requested batch/ROI matrix.

The Annex-A writer remains outside every accelerator path. It still accepts an
already encoded codestream and packages metadata and optional encoded alpha; no
pixel-to-codestream encoder is introduced.

## CUDA Oxide build and runtime

All reconstruction kernels are Rust compiled to PTX by
[NVIDIA CUDA Oxide](https://github.com/NVlabs/cuda-oxide). The device project
pins the same CUDA Oxide revision and `nightly-2026-04-03` toolchain used by
this organization's `j2k` backend. The main workspace retains its existing
Rust toolchain. `jxr-math` supports the device compiler's Rust 1.96 baseline.
Transforms and overlap filters call the allocation-free
`jxr-math` implementation directly; device descriptors are generated from
`jxr-math/data/reconstruction.abi`. The isolated device build uses numeric
arithmetic errors, explicit error propagation, and indexed array operations to
accommodate current CUDA Oxide code-generation limits. Host errors retain their
operation descriptions.

On Linux, install the pinned CUDA Oxide compiler and its documented build
prerequisites before building an executable CUDA backend. This includes a
working libclang and its builtin headers; custom installations may require
`LIBCLANG_PATH`, `LD_LIBRARY_PATH`, and `BINDGEN_EXTRA_CLANG_ARGS`.
The build script
stages its standalone device project under `OUT_DIR`, invokes
`cargo oxide build --arch sm_50 --no-fmad`, and embeds the resulting PTX.
`JXR_CUDA_OXIDE_ARCH` selects another target architecture. Compiler failures
are build errors; no CUDA C implementation is retained.

```sh
JXR_REQUIRE_CUDA_OXIDE=1 cargo test -p jxr --features cuda --test cuda_contracts -- --ignored --test-threads=1
JXR_REQUIRE_CUDA_OXIDE=1 cargo test -p jxr-test-support --features cuda --test cuda_hardware -- --ignored --test-threads=1
```

For builds without the compiler, including cross-platform CI, the Rust
adapter and its device-free contracts still compile, but
`CudaDecoderSession::is_available()` returns false. Set
`JXR_REQUIRE_CUDA_OXIDE=1` in hardware CI to make missing compiler support an
error. Alternatively, `JXR_CUDA_OXIDE_PTX=/absolute/path/jxr_reconstruction.ptx`
embeds a previously compiled module, without invoking the device compiler.
The PTX target must be compatible with the deployment GPU.

`cudarc` 0.19.10 supplies dynamically loaded Driver API contexts, streams,
events, allocations, and launches. Its `nvrtc` Cargo feature remains enabled
because this version gates its `Ptx` container and `load_module` method behind
that feature. JXR never invokes NVRTC or loads its library. Session creation
loads the embedded Rust-generated PTX through the CUDA driver; it performs no
source compilation. Deployment requires the NVIDIA driver, not NVRTC or the
CUDA Oxide compiler. The default target requires compute capability 5.0.

## Architecture and semantics

The CPU remains the parser, entropy decoder, predictor, portable reconstruction
oracle, and automatic fallback. `jxr-cuda` accepts the same
`PreparedReconstruction` coefficient arenas used by Metal. It does not parse a
codestream, walk entropy packets, or encode JPEG XR.

An executable CUDA plan validates the coefficient arena, sparse ROI window,
component geometry, tile partition, crop, output policy, and every 32-bit device
ABI offset before allocation or submission. Five ordered phases implement:

1. DC/LP dequantization and the first inverse transform;
2. first-level full or subsampled overlap for overlap mode two;
3. CPU-selected HP prediction, HP dequantization, and the second transform;
4. second-level overlap for modes one and two; and
5. chroma reconstruction, color conversion, alpha handling, crop, clipping, and
   typed or packed output storage.

CUDA Oxide kernels and host descriptors follow the same canonical ABI
schema used by Metal. Checked 64-bit intermediates set a device status word on
any operation that would exceed the scalar `i32` contract. That status is read
only after the completion event; it is never converted into a CPU retry.
Separately encoded alpha retains its own overlap mode and hard/soft tile
partition rather than inheriting the primary codestream's boundary policy.

`CudaDecoderSession` owns one primary context, four nonblocking streams, a lazily
reused exact-size scratch pool capped at 256 MiB, and an immutable coefficient
upload cache capped at 256 entries/512 MiB. The cache is keyed by coefficient
`Arc` identity and stream so coefficients stay contiguous and are never repacked.
Submissions retain every upload, descriptor, scratch allocation, status word,
output, and completion event. Dropping a pending submission waits before
recycling resources, preventing device use-after-free.

Single images, caller-owned outputs, independent native batches, and homogeneous
dense batches are supported. Dense batches keep all pixels in one device
allocation. ROI plans upload only the CPU-produced selected coefficient arena and
reconstruct its required overlap/chroma halo. Resident paths perform no host
pixel readback; host conversion copies the final native bytes exactly once.

Explicit `BackendRequest::Cuda` is strict. `Auto` chooses CPU for an absent or
uncompiled backend, native reduced output, or work below the provisional
threshold, all before CUDA submission. Unsupported CUDA plan combinations and
resource rejections are explicit errors rather than silent semantic changes. A
driver, launch, kernel-arithmetic, synchronization, or transfer failure after
submission is likewise returned to the caller and is never retried on CPU.

## Validation and benchmark boundary

The regular CI matrix compiles and lints the CUDA feature on CPU-only Linux and
macOS hosts through dynamic loading. `.github/workflows/cuda-hardware.yml` is a
serialized, manually dispatched self-hosted NVIDIA workflow. It records the GPU
and CUDA Oxide version, runs ignored lifecycle/destination tests, differentially compares
all 517 in-scope T.834 cases and ROI boundary variants with CPU, compares the
complete suite with T.835, and retains reports.

The reproducible pathology command is:

```text
JXR_BENCH_ITERATIONS=10 cargo run --release -p jxr-test-support --features cuda --bin jxr-cuda-pathology-bench -- SMALL.jxr LARGE.jxr
```

It covers full-image and centered ROI paths for batch sizes 1, 8, 32, 64, and
128, reporting preparation, asynchronous submission, synchronization, and
device-to-host median latency plus total p95 latency; image throughput; scratch
allocation misses; immutable host-to-device bytes; and output transfer bytes.
Paths, iteration count, and stream count are printed with the result.

Release 0.2.0 passed on the self-hosted `Cuda` runner. All 517 in-scope
T.834/T.835 cases passed, as did the CPU/ROI differential suite, lifecycle
tests, and configured benchmark. The
[workflow run](https://github.com/frames-sg/jxr/actions/runs/36360508404)
records the tested commit and hardware. The benchmark used two small conformance
fixtures, 32×32 and 145×130 pixels. This validates that NVIDIA configuration;
it does not establish performance on other devices, representative pathology
images, or a general speedup over CPU reconstruction.
