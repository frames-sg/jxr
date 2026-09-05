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
report assembly remain here. The shared package must be released from the J2K
repository before this dependency resolves from crates.io; coordinate its pinned
version with that release and refresh Cargo.lock from the registry.

Until then, local validation can use an explicit source overlay from the JXR root:

```console
cargo test --config 'patch.crates-io.j2k-mpsgraph-support.path="../j2k/crates/j2k-mpsgraph-support"' -p jxr-mpsgraph --locked
```

This development overlay does not establish standalone registry buildability.
