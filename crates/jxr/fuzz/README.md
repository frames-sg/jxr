# JPEG XR fuzzing

The `parse_and_decode` target exercises the raw T.832 and Annex-A parser plus
bounded full, quarter, and sixteenth-scale CPU decode requests. The heavier
`typed_batch` target maps known Annex-A formats to their exact native storage,
differentially checks owned output against caller-owned typed output, and checks
a duplicate native batch against the same result. Expected syntax, truncation,
resource-limit, and unsupported-format errors are ignored; panics, aborts,
sanitizer findings, and output disagreements are failures.

The `cuda_plan` target stops before runtime discovery or submission. It fuzzes
the CPU-produced coefficient handoff plus CUDA ABI, geometry, crop, overlap,
and resource preflight on hosts with no CUDA toolkit or NVIDIA driver.

The `encode_roundtrip` target fuzzes the encoder. Each input selects a pixel
format, among them every one the container table declares, an image of up to
64 by 64 samples in tight or padded storage, its samples, and an encode
request: quantizers or a quality, overlap, bands, trimmed flexbits, tiles,
alpha mode, chroma sampling, direct CMYK, shift bits, and float precision.
Encoding must not panic, must give the same result twice, and must return a
file or a classified error, never an internal invariant; a plain lossless
request for a declared format must succeed. Every file must parse, report the
source dimensions, and decode on the CPU in the source's pixel format. When
the request is lossless for the format (lossless quantizers, all bands, no
trim, no chroma subsampled by the encoder, and samples the coded form
represents) the decoded samples must be the source's, colour and alpha each
by its own quantizers; a float image must instead survive being coded a
second time.

Run from this directory with a nightly toolchain and `cargo-fuzz`:

```sh
cargo fuzz run parse_and_decode
cargo fuzz run typed_batch
cargo fuzz run cuda_plan
cargo fuzz run encode_roundtrip -- -max_len=256
```

The encoder harness is the library in `src/`, so it also runs without a
fuzzing engine. `cargo test` checks every declared format, flat images at
every power of two, and a fixed sequence of pseudo-random inputs, and lists
each distinct finding with the first case that showed it:

```sh
cargo test --release
JXR_FUZZ_CASES=300000 JXR_FUZZ_SEED=7 cargo test --release pseudo_random
JXR_FUZZ_INPUTS=artifacts/encode_roundtrip cargo test --release saved_inputs
```

The last form replays saved inputs, a corpus or the artifacts of a fuzzing
run, and groups what they show.

Seed each target's corpus with licensed T.834 inputs locally. Official conformance
vectors are intentionally not vendored.
