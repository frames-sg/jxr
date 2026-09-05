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
  descriptor schema. Its build script now emits both MSL and CUDA C declarations;
  Rust layout assertions keep the host descriptors tied to the same source.
- Existing tests place scalar math in `jxr-math`, syntax/reconstruction/output
  behavior in `jxr-native`, public routing and batches in `jxr`, and official
  T.834/T.835 discovery/comparison in `jxr-test-support`. CUDA therefore adds
  device-free contract/preflight tests beside the adapter and ignored device
  differential tests beside the existing conformance harness.
- Regular CI already exercises formatting, workspace Clippy/tests, minimal
  features, strict rustdoc, and fuzz-package compilation on Linux and macOS.
  Metal hardware validation is a serialized self-hosted workflow. CUDA follows
  that split: dynamic-loading builds stay in regular CI, while compilation by
  NVRTC, device execution, the 517-case differential suite, and benchmarks are
  confined to the serialized NVIDIA runner.
- The existing pathology benchmarks separate preparation, submission,
  synchronization, and readback while checking output checksums. The CUDA
  benchmark retains that measurement model and adds transfer and device-pool
  counters across the requested batch/ROI matrix.

The Annex-A writer remains outside every accelerator path. It still accepts an
already encoded codestream and packages metadata and optional encoded alpha; no
pixel-to-codestream encoder is introduced.

## Dependency decision (September 2026)

`jxr-cuda` pins `cudarc` 0.19.9 with only `std`, `driver`, `nvrtc`,
`dynamic-loading`, and the CUDA 11.4 ABI baseline enabled. The decision is based
on the following primary project and vendor documentation:

- [`cudarc` 0.19.9 API documentation](https://docs.rs/cudarc/0.19.9/cudarc/)
  documents safe context, stream, event, slice, Driver API, and NVRTC wrappers;
  its dynamic-loading mode requires no CUDA libraries at build time. The project
  is MIT OR Apache-2.0 and its
  [release history](https://github.com/chelsea0x3b/cudarc/releases) shows current
  CUDA 13.x maintenance.
- [`cust` 0.3.2 documentation](https://docs.rs/cust/0.3.2/cust/) describes a
  capable Driver API wrapper, but requires CUDA development libraries on the
  build system. The upstream
  [Rust-CUDA README](https://github.com/Rust-GPU/rust-cuda/blob/main/README.md)
  also characterizes the rebooted project as early development with expected
  bugs and safety issues. Its Rust-device toolchain would add nightly/LLVM and
  build-time CUDA requirements that conflict with this workspace's portable
  all-feature checks.
- The original
  [RustaCUDA repository](https://github.com/bheisler/RustaCUDA) is no longer the
  maintained direction; its ecosystem moved toward `cust`/Rust-CUDA.
- NVIDIA documents NVRTC as the supported in-process CUDA C++ to PTX boundary
  and states that its PTX can be loaded through the Driver API
  ([NVRTC guide](https://docs.nvidia.com/cuda/nvrtc/)). NVIDIA also defines
  streams and events as the core asynchronous execution primitives
  ([CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html))
  and documents stream-ordered allocation and memory-pool capability discovery
  in the
  [Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html).

The selected dependency adds no codec, image, neural-network, BLAS, or CUDA
Runtime API layer. Disabling default `cudarc` features avoids cuBLAS, cuDNN,
cuRAND, and their transitive/runtime surfaces. The lockfile adds `cudarc`,
ISC-licensed `libloading`, `cfg-if`, and the Windows loader shim; every license
is compatible with distribution under this workspace's terms. This keeps
compile time and binary surface substantially smaller than a general GPU
framework.

Maintenance risk remains concentrated in one FFI wrapper and NVIDIA's Driver
API/NVRTC compatibility. The version is pinned so an upstream ABI or safety
change cannot silently alter builds. The dependency's dual license matches the
workspace. Runtime loader failures and missing symbols become explicit session
initialization errors; CPU-only builds and ordinary `jxr` builds do not link,
load, or probe CUDA.

Enabling the feature embeds CUDA C source and compiles the small Rust/FFI
adapter, but runs neither `nvcc` nor NVRTC at build time. Creating a session
does incur one NVRTC compile and one Driver API module load for that device;
cloning and reusing the session amortizes that startup cost. Linux and Windows
use the NVIDIA libraries found by the platform loader, while macOS can compile
the Rust feature but reports CUDA unavailable at runtime. Dynamic loading keeps
the toolkit out of CPU deployments, but inherits the operating system's library
search-path trust boundary: production processes must not search user-writable
directories for `cuda`/`nvcuda` or `nvrtc`. Unsafe FFI representation is kept in
the focused backend and `cudarc`; checked Rust constructors validate buffer
extents, contexts, ABI widths, and kernel indexing metadata before submission.

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

CUDA C kernels and Rust descriptors are generated from the same canonical ABI
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
and toolkit, runs ignored lifecycle/destination tests, differentially compares
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

The workflow passed on the self-hosted `Cuda` runner on 2026-09-05. All 517
in-scope T.834/T.835 cases passed, as did the CPU/ROI differential suite,
lifecycle tests, and checksum-checked benchmark. The
[hardware report](cuda-validation-2026-09-05.md) records the tested commit,
driver/toolkit, measurements, and corpus limits. This validates that NVIDIA
configuration; it does not establish performance on other devices or a general
speedup over CPU reconstruction.
