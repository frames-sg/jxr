# Releasing JXR

The public repository is https://github.com/frames-sg/jxr. Crates use exact versions for dependencies within JXR. The workspace version
is the initial baseline; package-level patch versions are used for focused fixes.
`jxr-test-support` and downloaded reference corpora are not published.

Run the CI checks from `.github/workflows/ci.yml`, then the external CPU oracle:

```sh
cargo run -p jxr-test-support --bin jxr-t834 -- --backend cpu
```

Publish only changed packages and their exact-version dependents. Do not
republish unchanged package versions that already exist on crates.io. Use this
dependency order, first running `cargo publish -p NAME --dry-run` and then
`cargo publish -p NAME` after reviewing the package contents:

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

## 0.1.2 patch release

This patch set publishes the shared CPU-side device reconstruction plans
used by Metal and CUDA, and the MPSGraph submission lifetime owner. Public decode
contracts and the default CPU-only feature set are unchanged.

Publish the changed packages and their exact-version dependents in this order:

1. `jxr-core` 0.1.1
2. `jxr-native` 0.1.1
3. `jxr-metal` 0.1.2
4. `jxr-cuda` 0.1.1
5. `jxr` 0.1.2
6. `jxr-image` 0.1.1
7. `jxr-mpsgraph` 0.1.1

`jxr-math` remains at the already-published 0.1.0 version. The graph adapter
uses the published `j2k-mpsgraph-support` 0.11.0 registry dependency. Its source
matches the previous Git pin. The other J2K dependencies remain at 0.10.0;
the graph owner does not depend on them.

## 0.2.0 release

[Version 0.2.0](https://github.com/frames-sg/jxr/releases/tag/v0.2.0) is published
on crates.io for all seven packages below. The
[hosted checks](https://github.com/frames-sg/jxr/actions/runs/36360496727),
[Metal hardware validation](https://github.com/frames-sg/jxr/actions/runs/36359668987),
and [CUDA hardware validation](https://github.com/frames-sg/jxr/actions/runs/36360508404)
passed. CPU, Annex-A writer, Metal, and CUDA reference comparisons each passed
517 in-scope cases, with 179 declared out-of-scope cases and no failures. The
reports are attached to the release. Metal tested the reviewed PR source; its
tree is identical to the release merge used by CUDA.

This release moves the J2K dependencies to the 0.11 line, `jxr-native` to
`fearless_simd` 1.0.0, and `jxr-cuda` to `cudarc` 0.19.10. `jxr-core` re-exports
`j2k_core::{BackendKind, BackendRequest, Rect}`, so the J2K upgrade changes public
types and every published crate that exposes them moves to 0.2.0. The
`fearless_simd` upgrade is private to `jxr-native`.

`j2k-core`, `j2k-metal-support`, and `j2k-mpsgraph-support` are pinned to
0.11.2, the J2K release used by the downstream WSI crates. J2K crates pin each
other exactly, so one dependency graph cannot hold two J2K 0.11 patch releases.

The packages were published in this order:

1. `jxr-core` 0.2.0
2. `jxr-native` 0.2.0
3. `jxr-metal` 0.2.0
4. `jxr-cuda` 0.2.0
5. `jxr` 0.2.0
6. `jxr-image` 0.2.0
7. `jxr-mpsgraph` 0.2.0

`jxr-math` remains at the already-published 0.1.0 version.

## 0.2.1 release

This dependency-only patch aligns the exact J2K dependencies with 0.11.3 so
applications can use the native JPEG 2000 tag-tree fix alongside JPEG XR.
The published 0.2.0 packages pin J2K 0.11.2 and cannot resolve with J2K 0.11.3
in one dependency graph. Decoder source and public signatures are unchanged.

Published `jxr-core`, `jxr-native`, `jxr-metal`, `jxr-cuda`, `jxr`, `jxr-image`,
and `jxr-mpsgraph` at 0.2.1 in the dependency order above.
`jxr-math` remains at 0.1.0.

The [hosted checks](https://github.com/frames-sg/jxr/actions/runs/36435652621),
[Metal hardware validation](https://github.com/frames-sg/jxr/actions/runs/36435651913),
and [CUDA hardware validation](https://github.com/frames-sg/jxr/actions/runs/36435655712)
passed for the tagged source. CPU, Annex-A writer, Metal, and CUDA reference
comparisons each passed 517 in-scope cases, with 179 declared out of scope and
no failures. The reports are attached to the
[0.2.1 release](https://github.com/frames-sg/jxr/releases/tag/v0.2.1).

## 0.3.0 release line

This release aligns JXR with J2K 0.12.0 and requires Rust 1.99.0.
Public J2K types now come from the 0.12 line, so downstream applications
must update their direct J2K dependencies at the same time. The entropy
decoder adds inlining hints and places highpass coefficients directly into
their adaptive scan destinations, avoiding a temporary run/level block.
Malformed input still returns a typed error and discards the failed tile.

Published `jxr-math` 0.1.1 and the other seven libraries at 0.3.0 in the
order above. The [hosted checks](https://github.com/frames-sg/jxr/actions/runs/37444721162),
[Metal hardware validation](https://github.com/frames-sg/jxr/actions/runs/37444774439),
and [CUDA hardware validation](https://github.com/frames-sg/jxr/actions/runs/37444777485)
passed for the tagged source. CPU, Annex-A writer, Metal, and CUDA reference
comparisons each passed 517 in-scope cases, with 179 declared scope exclusions
and no failures. The reports and hardware benchmark results are attached to
the [0.3.0 release](https://github.com/frames-sg/jxr/releases/tag/v0.3.0).
