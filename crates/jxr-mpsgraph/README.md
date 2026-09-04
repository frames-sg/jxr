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
