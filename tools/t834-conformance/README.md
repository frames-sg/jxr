# T.834 conformance suite

This directory downloads the in-force October 2014 ITU-T T.834 JPEG XR
conformance package into `target/t834-conformance`. The 61,448,825-byte archive
is checksum-pinned to SHA-256
`c066c5e24a212f3bb09eaf235cf21359754ffbb747d9fedc620e09629ca2a55d`.
No conformance vectors are vendored or published with the Rust crates.

Setup requires `curl`, a SHA-256 utility, and either `unzip` or Python 3.
The archive checksum is verified before either extractor runs.

Download and extract the suite:

```sh
tools/t834-conformance/build.sh
```

Build the T.835 oracle, then run every Main-scope Annex-A vector through the
portable decoder:

```sh
tools/t835-oracle/build.sh
cargo run -p jxr-test-support --bin jxr-t834 -- --backend cpu
```

Run strict Metal comparisons on macOS:

```sh
cargo run -p jxr-test-support --features metal --bin jxr-t834 -- --backend metal
```

Reports are written to `target/t834-conformance/reports`. Use `--category NAME`,
`--limit N`, `--report PATH`, or `--verbose` for focused diagnosis. Pass
`--rewrap` to extract each in-scope raw codestream and its metadata, serialize a
new Annex-A file with `jxr::write_annex_a`, and compare that generated file with
T.835. This checks the writer against the same official corpus as the decoder.

The runner explicitly excludes the `Output_Color_Format_Advanced` category and
`.jpx` JPEG 2000 boxed wrappers from the Main-profile Annex-A target. A format
the harness cannot serialize is reported separately from both scope exclusions
and decoder failures. T.835 is invoked with an Advanced declaration ceiling so
official files that declare Advanced while using in-scope syntax can still be
compared; this does not expand the syntax accepted by the Rust decoder. Byte
identity against T.835 is strong differential evidence, but no conformance
claim is made until every in-scope vector passes and the T.834 procedures are
reviewed as a whole.
