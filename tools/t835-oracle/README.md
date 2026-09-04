# T.835 differential oracle

This directory builds the official ITU-T T.835 (2012) JPEG XR reference
software 1.32 as an external validation executable. It is not a production
dependency and its source is never copied into this repository.

Build it with:

```sh
tools/t835-oracle/build.sh
```

The script downloads the official software archive into `target/t835-oracle`,
checks SHA-256
`22526f45c09d5f7c77793aba68b3fbe480f0e1d58315868fc8fa2d60db6db79b`,
and builds the `jpegxr` reference program using the included Makefile. Set
`JXR_T835_ORACLE` to select a different compatible executable.

The official source carries its own ITU/ISO/IEC/Microsoft research and
conformance notice in `Software/COPYRIGHT.txt`. It is deliberately downloaded
on demand rather than vendored or linked into any Rust crate.

Compare one or more Annex-A images after building:

```sh
cargo run -p jxr-test-support --bin jxr-diff -- image.jxr
```

The comparison requests the Annex-A pixel representation from the Rust CPU
decoder, invokes T.835 with Main-profile limits, and compares the resulting raw
bytes. Unsupported or reference-padded layouts fail explicitly.

On macOS, run the same comparison through a strict Metal session:

```sh
cargo run -p jxr-test-support --features metal --bin jxr-diff -- --backend metal image.jxr
```

The Metal command does not permit CPU fallback. A device, pipeline, submission,
or arithmetic failure is returned instead of being hidden by the scalar route.
