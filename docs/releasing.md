# Releasing JXR

The public repository is https://github.com/frames-sg/jxr. Crates use exact versions for dependencies within JXR. The workspace version
is the initial baseline; package-level patch versions are used for focused fixes.
`jxr-test-support` and downloaded reference corpora are not published.

Run the CI checks from `.github/workflows/ci.yml`, then the external CPU oracle:

```sh
cargo run -p jxr-test-support --bin jxr-t834 -- --backend cpu
```

Publish in dependency order, first using `cargo publish -p NAME --dry-run` and
then `cargo publish -p NAME` after reviewing the package contents:

1. jxr-math
2. jxr-core
3. jxr-native
4. jxr-metal
5. jxr-cuda
6. jxr
7. jxr-image
8. jxr-mpsgraph

Optional dependencies must also exist in the registry before publishing the
facade. The default `jxr` feature set remains CPU-only. CUDA hardware correctness
requires the separate hardware workflow and is not implied by a successful
all-feature compilation or tests on a Mac.

For the initial 0.1.0 release on 2026-09-04, local formatting, all-feature Clippy,
all-feature workspace tests and documentation passed on aarch64 macOS with Rust
1.96. The T.834/T.835 CPU comparison passed 517 in-scope cases; 179 cases were
outside the declared scope, with zero failures and zero harness-unsupported
cases. This is differential evidence, not a claim of complete conformance.

The 0.1.1 patches for `jxr-metal` and `jxr` make non-macOS Metal stubs pass
all-target Clippy without changing the ownership contract or decode behavior.
The patch was checked locally on macOS and with Clippy targeting x86_64 Linux.
