# Contributing to JXR

Bug reports, documentation improvements, and focused fixes are welcome. Start
with the [support boundary](docs/support.md) and [architecture](docs/architecture.md)
to see which formats and stages the project implements.

## Local development

Use the Rust toolchain pinned in `rust-toolchain.toml` (currently 1.99.0).
The default `jxr` build is CPU-only; GPU runtimes and the external reference
software are not needed for ordinary CPU development.

From the repository root, run the relevant package tests. For a change to the
public CPU facade:

```sh
cargo test -p jxr --no-default-features
```

For changes to another crate, select that package and, where useful, the affected
integration test. Add regression coverage for changed behavior. Documentation-only
changes do not need application builds or codec test runs; check examples against
the public API and verify links instead.

Keep changes focused and format Rust edits with the pinned toolchain. The
[CI workflow](.github/workflows/ci.yml) defines the full pull-request gates,
including formatting, Clippy, workspace tests, documentation, and fuzz-package
build checks. A successful all-feature build does not establish GPU correctness;
Metal and CUDA execution use separate hardware workflows.

## Reference comparisons

When a change affects format parsing, reconstruction, or container writing, use
the official T.834 corpus and T.835 reference implementation as appropriate.
Setup and focused-run options are documented in:

- [T.834 conformance runner](tools/t834-conformance/README.md)
- [T.835 reference oracle](tools/t835-oracle/README.md)

The setup scripts download reference material into `target/`. Do not commit
reference software, downloaded corpora, or private application images. Reports
must distinguish passing in-scope cases, scope exclusions, and failures.

## Reporting a problem

Open an [issue](https://github.com/frames-sg/jxr/issues) with:

- The JXR version, Rust version, operating system, and enabled features.
- The requested output format, region, scale, and backend.
- The expected result and the actual error or output difference.
- A small reproducer and, when shareable, a minimal input image. Remove private
  or identifying content before sharing files.
- For GPU problems, the device and driver/runtime details, and whether the same
  input and request succeed with `BackendRequest::Cpu`.

For performance reports, include the workload, image dimensions and formats,
batch size, hardware, and what the timing includes (preparation, submission,
readback). Conformance results alone do not establish a performance advantage.

## Pull requests and releases

Explain the user-visible problem, the change, and the checks actually run. Call
out any hardware or corpus validation that remains untested. Keep public format
support, examples, and migration notes consistent with API changes.

Publishing instructions and release history are in [releasing](docs/releasing.md).
