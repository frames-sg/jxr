# T.835 differential oracle

This directory builds the official ITU-T T.835 (2012) JPEG XR reference
software 1.32 as an external validation executable. It is not a production
dependency and its source is never copied into this repository.

Setup requires `curl`, a SHA-256 utility, `make`, a C compiler, and either
`unzip` or Python 3. The archive checksum is verified before extraction.

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
bytes. Unsupported or reference-padded layouts fail explicitly. `--advanced`
before the image names raises the reference's limit to the Advanced profile,
which images with a YUV or CMYKDIRECT output colour declare. T.835 writes
those plane by plane and the comparison arranges the Rust decoder's samples
the same way.

On macOS, run the same comparison through a strict Metal session:

```sh
cargo run -p jxr-test-support --features metal --bin jxr-diff -- --backend metal image.jxr
```

The Metal command does not permit CPU fallback. A device, pipeline, submission,
or arithmetic failure is returned instead of being hidden by the scalar route.

## Checking the encoder

`jxr-encode-check` encodes source images under a matrix of settings (lossless
and lossy quantizers, every overlap mode, tiling, band sets, flexbits
trimming, points on the quality scale, both packet layouts) and checks each
result three ways: the Rust decoder and T.835 must
produce identical bytes, T.835 must not warn about the declared profile or
level, and a lossless result must equal its source. Every CMYK source is
also coded as `CMYKDIRECT`, whose raw output T.835 writes one plane after
another; the comparison arranges the Rust decoder's samples the same way.

Eleven of the settings are frequency mode: every band set, trimmed flexbits,
soft and hard tiles, every overlap mode, lossless and lossy. They are part
of the matrix, so each runs in every coded form of a source described below.
A frequency-mode file must also declare that layout and decode in the Rust
decoder at a quarter and a sixteenth of its size, to an image of the reduced
dimensions; T.835 has no reduced output to compare those samples with. The
decoder documents the reduced requests it does not serve, a quarter of an
image without a low-pass band and a sixteenth of subsampled planar YUV
output, and only those may fail.

Every RGB, RGBE and YUV 4:4:4 source runs each setting at 4:2:2 and at 4:2:0
chroma as well as at 4:4:4. The encoder discards chroma resolution there, so
those results are compared between the decoders but not with the source.
The T.835 software asserts that second-level overlap of subsampled chroma is
not used in an image, or under hard tiling in a tile, one macroblock wide;
for such a setting the check requires the encoder to refuse the request and
counts the refusal as a pass. Only a request that names the overlap mode can
ask for that: a quality leaves the mode to the encoder, which codes such an
image with first-level overlap, and one setting, quality 3 in hard tiles one
macroblock wide, is there to show it. YUV sources are coded at their own sampling from the planes the Rust
decoder returns, with the bit depth and chroma centering their file declares.
T.835 writes YUV output plane by plane, like `CMYKDIRECT`.

Every source with alpha runs each setting with separate and with integrated
alpha, and seven more settings in each mode, two of them frequency mode, that
quantize the alpha plane apart from the colour; a lossless alpha plane must
equal its source whatever the colour beside it uses. Settings that trim flexbits run with separate alpha
only: T.835 reads an integrated alpha plane as untrimmed and fails on such a
tile, its own encoder's included, so the encoder refuses the combination. For
a `CMYKDIRECT` or YUV image with separate alpha T.835 appends the alpha
image's own output, the alpha plane followed by a plane for each further
colour channel, and the comparison expects those: three empty planes after
`CMYKDIRECT`, and after YUV two planes of the chroma planes' size holding
the mid-range value of the sample depth.

`--keep DIR` saves each source's files in a subdirectory named for the
source. If that name is already in use, the checker appends `-2`, `-3`,
and so on. Each file inside is named for its encode setting.

`--quality-scale` runs another check in place of the comparison, without
T.835: it codes every source at each quality from 100 down to 0, in each of
its coded forms, and lists the steps on which a file grew. Those are
counted, not failed: an image that is exact in a few coefficients of
unscaled arithmetic is cheapest coded losslessly, as described under
[Quality scale](../../docs/support.md#quality-scale).

With no arguments it uses a built-in set of synthetic images, among them
planar YUV 4:2:2 and 4:2:0 with and without alpha:

```sh
cargo run --release -p jxr-test-support --bin jxr-encode-check
```

Given `.jxr` files, it decodes each one and re-encodes its pixels, skipping
images whose pixel format the encoder does not support. To use every image of
the T.834 suite as a source:

```sh
find target/t834-conformance/suite-2014 -name '*.jxr' -print0 \
  | xargs -0 cargo run --release -p jxr-test-support --bin jxr-encode-check --
```

The exit status is non-zero if any case fails. Agreement between the two
decoders shows an encoded codestream is valid; it does not measure how well
it is compressed. `jxr-quality-curve` measures that for the quality scale:
size, PSNR, SSIM and largest error at every quality, for one picture stored
at each sample depth and in each colour format the scale is calibrated for,
as described under [Quality scale](../../docs/support.md#quality-scale).
