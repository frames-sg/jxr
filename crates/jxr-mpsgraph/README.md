# JXR MPSGraph

`jxr-mpsgraph` connects CPU-prepared JPEG XR reconstruction to `MPSGraph` on
Apple-silicon macOS. Exact reconstruction runs in `jxr-metal`; the adapter writes
homogeneous Gray, RGB, or RGBA `U8`, `U16`, and `I16` batches directly into one
dense private NHWC `MTLBuffer`, then presents that allocation as
`MPSGraphTensorData` on the same command queue.

The application performs no decoded-pixel readback or upload in this path.
This is an application-level lifetime and queue-ordering guarantee, not a claim
about framework-internal copies; use Xcode Metal capture to characterize those.

Parsing and entropy decoding remain on the CPU. Floating-point conversion,
normalization, transposition, reductions, and model operations belong in the
graph. NCHW codec stores, Core ML model loading, MTLTensor/Metal 4, and using MPS
for JPEG XR reconstruction are outside this crate's v1 scope.

## Shared submission owner and release prerequisite

The adapter uses `j2k-mpsgraph-support` for graph execution, callback/error ownership,
and input lifetime. JXR preparation, queue/device validation, codec completion and
report assembly remain here. Version 0.10.0 is pinned to immutable Git revision
`5a0e238307079e6381095bf91b15c156569796d2` in the J2K repository, so standalone
checkout builds need no sibling source overlay.

Publishing this crate to crates.io still requires publication of the shared owner
and a reviewed migration of this dependency and lockfile to its registry source.
