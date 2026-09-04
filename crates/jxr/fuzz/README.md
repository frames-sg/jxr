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

Run from this directory with a nightly toolchain and `cargo-fuzz`:

```sh
cargo fuzz run parse_and_decode
cargo fuzz run typed_batch
cargo fuzz run cuda_plan
```

Seed each target's corpus with licensed T.834 inputs locally. Official conformance
vectors are intentionally not vendored.
