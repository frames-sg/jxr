# JPEG XR support boundary

JXR supports raw T.832 codestream and Annex-A still-image decode for
Main-profile syntax, CPU encoding of grayscale, RGB and YUV images at every
sample depth their container formats declare and of CMYK and N-component
images, and validated Annex-A serialization around existing raw codestreams.
Annex-F/HEIF sequence storage and transcoding are out of scope.

For published [JXR 0.3.0](https://github.com/frames-sg/jxr/releases/tag/v0.3.0),
the CPU, Metal, and CUDA routes byte-match T.835 for all 517 T.834 cases
classified as in-scope Main syntax. The Annex-A writer comparison also passes
those 517 cases. The remaining 179 corpus entries are explicitly skipped because
they use Advanced-only output syntax or a JPEG 2000 `.jpx` wrapper.

## Implemented now

- Borrowed raw-codestream and Annex-A inspection, checked ranges, tile index,
  profile/level, orientation, zero-copy ICC bytes, primary plane, and separate alpha.
- Complete typed classification of the Annex-A Table A.6 pixel GUIDs, including
  RGB/BGR storage padding and premultiplied separate alpha, lossless retention
  of unknown GUIDs, and consistency checks against parsed syntax.
- WIC's additional 32-bit RGBA/PRGBA and 96-bit RGB float pixel GUIDs,
  including premultiplied separate-alpha metadata.
- Spatial and indexed frequency-mode Y-only, YUV420, YUV422, YUV444, YUVK, and
  Main-profile 2–16 channel N-component entropy decoding for DC-only, no-HP,
  no-flexbits, and all bands.
- Tile quantizers, adaptive resets, scans, coefficient prediction, flexbits,
  compact subsampled coefficient storage, exact sampling-specific inverse
  transforms, hard/soft tile reconstruction, overlap modes 0–2, chroma
  reconstruction, crop halos, and typed output.
- Component-separated coefficient arenas, reconstruction-phase HP prediction,
  and tile/halo-pruned region decode.
- Full-resolution CPU decode as a stream of macroblock rows: packets are read
  a band of rows at a time in both packet layouts, each band is
  reconstructed and its finished rows stored in the output before the next
  is read, and the working memory goes with the width of the image, not its
  area. One thread needs no thread pool; in a pool the tile rows are shared
  out among strips under a byte bound. See [Decode memory](#decode-memory).
- Frequency-mode native quarter-width/height DC+LP reconstruction and
  sixteenth-width/height DC-only reconstruction, with higher packet bands
  excluded from entropy decode and coefficient allocation. Each reduced
  sample is the DC of a block or macroblock at the level full reconstruction
  gives its samples: the transform gain is removed, at each chroma sampling,
  and under overlap filtering so are the level change and the rounding
  offset of the sample-level filter, which is not run, at every position
  among the edges of the image and of hard tiles. The low-pass-level overlap
  filter works at a quarter of the size and is run by both: a sixteenth-size
  decode runs it over the DC band alone and averages each macroblock. An
  image that is constant over every block a reduced sample stands for
  decodes exactly when coded losslessly without overlap. With overlap a flat
  image keeps its level at every position to within what the forward
  filters round away, which differs from level to level: one sample value
  in each coded plane, two at the corners of the image and of hard tiles,
  and two after a colour transform in the tests. Detail at the scale of the
  sample-level filter is approximate, and a coded window whose margins are
  no multiple of the reduction is cut at block or macroblock boundaries, so
  its reduced picture is shifted by the remainder.
- Typed output packing for bit-packed, integer, F16/F32, packed RGB, RGBE,
  native planar YUV420/YUV422, BD10 YUV, and alpha-bearing CMYK/N-component
  families.
- Per-session safe AVX2/NEON capability tokens for HP dequantization and common
  U8 packing, with the exact scalar implementation retained as the oracle and
  portable fallback.
- Separate-alpha preservation and typed scalar premultiplication for unpacked
  integer and floating-point channel layouts.
- Integrated-alpha spatial and frequency packet interleaving for every implemented
  primary layout, with independent alpha entropy state, band presence, quantizers,
  coefficient storage, reconstruction, plane-local output scaling, preservation,
  and premultiplication.
- Direct CPU entropy writes into shared Metal coefficient storage, complete
  Apple-silicon Metal reconstruction, precise MSL compilation, checked external
  destinations, resident and zero-copy shared host output, bounded aggregate
  descriptor batches, scratch pools, and immutable fallback upload caching.
- Shared owned batch preparation, stable homogeneous grouping, indexed sibling
  failures, identity/request plan caching, optional coefficient-ready caching,
  direct final-allocation typed CPU stores, caller-owned typed CPU output, CPU
  NCHW layout, direct independently encoded alpha output, retained coefficient,
  component-raster, inverse-transform, signed component-plane, and typed layout
  workspaces, nonblocking Metal submission, single-allocation Metal resident
  groups, exact-queue caller-owned Metal output, and shared-batch MPSGraph
  handoff.
- Explicit full-image host orientation application for supported single-plane
  typed, packed, and one-bit luma outputs, plus bounded parser/decode,
  typed-output/native-batch differential, and encode round-trip fuzz targets.
- A standalone zero-copy `image::DynamicImage` ownership adapter for exact
  Luma/LumaA/RGB/RGBA `U8` and `U16` plus RGB/RGBA `F32`, retaining ICC and alpha
  semantics outside the upstream image type.
- Deterministic Annex-A still-image writing for validated raw primary and
  separate-alpha codestreams, including typed orientation, finite positive
  display resolution, exact ICC bytes, four-byte-aligned payloads, and a final
  full parser/format-consistency validation pass.
- Hosted CPU CI on Linux and an opt-in, serialized self-hosted Apple-silicon
  workflow covering strict workspace checks, Metal, MPSGraph, T.834/T.835
  differential runs, the pathology benchmark, and retained diagnostic artifacts.
- Optional dynamically loaded CUDA reconstruction with reusable contexts and
  streams, bounded scratch and immutable uploads, asynchronous/resident output,
  checked caller-owned destinations, native and dense batches, an ignored full
  CPU differential suite, a serialized self-hosted NVIDIA workflow, and a
  phase-separated pathology benchmark. The NVIDIA workflow passed all 517
  in-scope T.834/T.835 cases, CPU/ROI comparisons, lifecycle tests, and benchmark
  checks for release 0.3.0; see the [CUDA workflow](https://github.com/frames-sg/jxr/actions/runs/37444777485).

## Encoding

Implemented:

- Source formats: grayscale, RGB, BGR, and RGB or BGR with a storage padding
  channel, coded as Y-only or YUV through the reversible colour transform.
  Tightly packed or strided rows.
- Chroma subsampling: `EncodeRequest::with_chroma_sampling` codes the chroma
  of an RGB, RGBE or YUV 4:4:4 source at 4:2:2 or 4:2:0, the internal formats
  a decoder interpolates back to full resolution. The encoder halves chroma
  with the filter of the T.835 reference encoder, taps 1, 4, 6, 4, 1 over 16
  mirrored at the image edges, after padding to whole macroblocks, and
  declares chroma centering zero, the grid that filter keeps. Planes are
  scaled by the quantizers as at 4:4:4, so lossless quantizers keep every
  coded plane exact, luma included, and lose only the chroma that was
  filtered away; an image whose chroma is constant is reproduced exactly.
  The official suite codes 4:2:0 and 4:2:2 unscaled throughout; the T.835
  encoder scales such planes whatever their quantizers, which this encoder
  does not copy, because scaled lossless planes coded `Dog.jxr` of the suite
  larger at 4:2:0 than lossless 4:4:4 does, and unscaled ones 41 percent
  smaller. At uniform parameters 12 and 40, 4:2:0 coded it 24 and 33 percent
  smaller than 4:4:4.
  Second-level overlap filters subsampled chroma across pairs of
  macroblocks, and the T.835 reference software asserts that an image coded
  with both is at least two macroblocks wide, as is every tile under hard
  tiling; the encoder returns `InvalidRequest` for narrower ones.
- YUV sources: every YUV container format, coded directly with no colour
  transform and declared at the Advanced profile. 4:4:4 is interleaved, at 8,
  10 and 16 bits and in 16-bit fixed point. 4:2:2 at 8, 10 and 16 bits and
  4:2:0 at 8 bits are planar, as the decoder returns them: Y, U and V planes
  and then alpha, located by `SourceImage::planes` when they are not packed
  one after another. Dimensions must be even along each halved axis, which
  the T.835 reference software asserts of such an image.
  Ten-bit samples in 16-bit storage are declared with
  `SourceImage::with_sample_bits`, and a sample above 1023 is rejected
  rather than clipped. `SourceImage::with_chroma_centering` declares where a
  subsampled source's chroma lies. A 4:4:4 source takes the chroma sampling
  option like RGB; a subsampled one is coded at its own sampling.
- CMYK at 8 and 16 bits, coded as YUVK through the reversible colour
  transform, or with `EncodeRequest::with_cmyk_direct` as untransformed
  `CMYKDIRECT`, which takes an Advanced-profile decoder. N-component images
  of three to eight channels at 8 and 16 bits, the counts the container
  format table declares, coded as independent components.
- Alpha in every format the container format table declares with it: RGBA at
  8-bit (BGRA in Table A.6, RGBA through the WIC identifier jxrlib and Windows
  write), 16-bit, fixed point, half and 32-bit float; CMYK and `CMYKDIRECT`
  with alpha; YUV with alpha at each sampling and depth; and three to eight
  channels with alpha, at 8 and 16 bits.
  `EncodeRequest::alpha_mode` selects separate alpha, the default, which
  codes alpha as a second Y-only codestream in the Annex-A file, or integrated
  alpha, a second plane interleaved into each tile packet of one codestream.
  Separate alpha is the default because the T.835 reference encoder and
  jxrlib write it, it leaves the colour image at the profile it needs without
  alpha, and jxrlib does not read integrated alpha beside N-component colour.
  The alpha plane takes the luma quantizers unless
  `EncodeRequest::with_alpha_quantizers` sets its own, and is scaled or
  unscaled by its own losslessness in either mode, so alpha can be exact
  under lossy colour and the reverse.
- Premultiplied sources, marked by `SourceImage::with_premultiplied_alpha`:
  the samples are coded as given and declared premultiplied, by the container
  format in both modes and by the codestream header as well with integrated
  alpha. The formats with a premultiplied declaration are 8-bit BGRA and
  RGBA, 16-bit RGBA, and 32-bit float RGBA.
- Sample depths: 1-bit, 8-bit, 10-bit and 16-bit unsigned, 16-bit and 32-bit
  signed fixed point with a chosen `SHIFT_BITS`, half float, 32-bit float
  with a chosen mantissa length and exponent bias, packed RGB 555, 565 and
  10-10-10, and RGBE. Each depth uses the decoder's own bias, scaling and float mapping
  run backwards, so a format encodes exactly as it decodes. 32-bit float keeps
  the configured mantissa bits; 32-bit integers drop `SHIFT_BITS` low bits.
- Lossless coding with unscaled arithmetic, and lossy coding with scaled
  arithmetic and separate DC, low-pass, and high-pass quantizers for luma and
  for chroma.
- A quality scale from 0 to 100 that the encoder resolves against the
  sample range and colour format of each source into those quantizers and
  an overlap mode, described [below](#quality-scale). It leaves chroma at
  full resolution unless the request selects another sampling.
- Overlap modes 0-2, soft and hard tiles, a configurable tile size, all four
  band sets, and flexbits trimming with prediction closed around the trimmed
  values. Tiles are at most 32 macroblocks, 512 samples, a side unless
  `EncodeRequest::with_max_tile_macroblocks` says otherwise, so an image
  larger than that is entropy coded, and later decoded, in parallel, and a
  smaller one is a single tile. The size is fixed, not chosen by the machine,
  so a request gives the same bytes everywhere; 255 keeps most images one
  tile and gives the smallest file. [Encode throughput](#encode-throughput)
  has the measurements behind the default.
- Both packet layouts, from one writer for every internal colour format.
  Spatial mode, the default, codes a tile as one packet. Frequency mode,
  `EncodeRequest::with_bitstream_mode(BitstreamMode::Frequency)`, codes it as
  a DC, a low-pass, a high-pass and a flexbits packet, as far as the band set
  goes, which is what a decoder needs to stop early: such a file decodes at
  `DecodeScale::Sixteenth` and, when it codes a low-pass band, at
  `DecodeScale::Quarter`. Every format and option is available in both
  layouts, and the two carry the same coefficients, so they decode to the
  same samples and declare the same profile and level. A frequency-mode file
  is larger by a few bytes for each extra packet: its start code, its tile
  index entry, and its padding to a whole byte.
- A tile index for multi-tile and for frequency-mode images, and the lowest
  profile and level declaration the codestream meets.
- An Annex-A file declaring the source pixel format, a single raw codestream
  (`encode_codestream`, which separate alpha does not fit), or the raw
  codestreams and pixel-format identifier a container is written from
  (`encode_codestreams`). The codestreams of an Annex-A file are written
  where the file holds them, so none is held beside the file.
- Memory that follows the image width and not its height. The encoder
  streams each component down its plane, reading and converting a few
  macroblock rows at a time and filtering, transforming, quantizing and
  entropy coding every row once, holding only the rows its filters wait on:
  seven macroblock rows are read at a time on one thread, and with threads
  bands of tile rows are admitted against the 256 MiB bound a decode's
  strips are, an image that fits being one band. Every band height and
  thread count codes the same bytes. [Bands](#bands) and [Memory](#memory) have the
  measurements.

Not implemented; these return `Unsupported` or have no request option:

- Gray with alpha, which no container format declares, and YUV at a depth
  or sampling the table lacks, such as 4:2:0 above 8 bits.
- A YUV 4:2:2 source coded as 4:2:0. Neither this decoder nor T.835
  interpolates chroma up to a subsampled output, so a subsampled source is
  coded at its own sampling whatever the request selects.
- Flexbits trimming with integrated alpha. A tile carries one
  `TRIM_FLEXBITS`, which this decoder and the T.835 encoder apply to both of
  its planes, but the T.835 and jxrlib decoders read an integrated alpha
  plane as untrimmed: T.835 rejects such a tile and jxrlib decodes noise,
  from the T.835 encoder's own output as well. The encoder does not write
  what only this decoder reads; separate alpha trims each image on its own.
- Premultiplied sources in a format without a premultiplied declaration:
  fixed point, half float, CMYK, and N-component with alpha.
- Quantizers that vary by tile or macroblock: every tile takes the
  quantizers of the image plane header.
- Rate control: a quality or explicit quantizers fix the coding, and the size
  follows from the image. There is no target size or bit rate.
- GPU encoding. The encoder is CPU-only; tiles are coded in parallel.

Encoded files are compared with T.835 by `jxr-encode-check`, described in the
[T.835 oracle guide](../tools/t835-oracle/README.md). Its lossy CMYK and
N-component cases are also the differential evidence for decoding scaled YUVK
and N-component planes, which the T.834 suite does not contain, and its alpha
cases for decoding an alpha plane whose scaled flag differs from the colour
planes' and integrated alpha beside N-component colour. T.835 does not know
the WIC identifiers, so 8-bit RGBA and unpadded 32-bit float RGB are checked
against jxrlib instead.

jxrlib reads a hard-tiled file with overlap filtering correctly only when the
tile columns start at the same macroblock positions as the tile rows, as in
every hard-tiled image of the T.834 suite. Its hard-tile inverse transform
looks up column boundaries in the row table and row boundaries in the column
table (`invTransformMacroblock_alteredOperators_hard`), so on any other grid
it filters across some hard boundaries. This decoder and T.835 agree on such
files, in both packet layouts; the encoder writes them as T.832 specifies.

The check runs every RGB, RGBE and YUV 4:4:4 source at each chroma sampling
as well. The lossy ones among those cases are the differential evidence for
decoding scaled YUV 4:2:0 and 4:2:2 planes: all 103 images of the suite with
subsampled chroma are unscaled. Its sources include the 16 YUV images of the suite's
`Output_Color_Format_Advanced` category, which the decoder's own conformance
run leaves out: `jxr-diff --advanced` decodes each of them identically in JXR
and T.835, at every sampling and depth, with and without alpha.

Eleven of the check's settings are frequency mode, in every coded form of a
source. They are the differential evidence for decoding frequency-mode
packets of subsampled chroma beside an integrated alpha plane, which the
suite does not contain. T.835 has no reduced-size output, so each
frequency-mode file is only required to decode at a quarter and a sixteenth
of its size to the reduced dimensions; the round-trip tests compare the
reduced samples with images whose reduced picture is known, and with the
reduced decode of the same image coded with only the bands that picture
needs.

### Quality scale

`EncodeRequest::quality` takes a number from 0 to 100. 100, and anything
above it, is the lossless request. A lower value is stored in the request as
it is, and the encoder resolves it against each source into quantizers for
every component and band and an overlap mode. Every lower value is lossy
wherever the source has a lossy step to offer, and never quantizes more
finely than the value above it.

A quality is meant to say the same of a picture however the picture is
stored: the same error against the range of its samples, in a file of
comparable size. The scale is therefore defined in three layers. The first
is the scale itself, calibrated on 8-bit pictures as parameters of luma and
of full-resolution chroma. The second gives each coded component the step of
its part in the colour format. The third exchanges every parameter for the
one whose step is as large against the source's sample range as the
calibrated step is against 255. All three live in one module of the encoder,
`encode::quality`. Quantizers given by `EncodeRequest::lossy`,
`with_chroma_quantizers` and `with_alpha_quantizers`, and an overlap mode
given by `with_overlap`, are coded as given whatever the source: they
replace the scale for the planes they name and nothing else.

**The 8-bit scale** has four parts, each chosen from measurements on 8-bit
photographs:

- **Level.** A quality selects a level in half quantization parameters. Luma
  takes the level rounded up and chroma the level rounded down, so where the
  scale moves by half a parameter the two change on alternate qualities. The
  level follows straight segments: luma parameter 5 at quality 99; 8, 12, ...
  32 from 98 to 92; then 1.5 parameters per quality to 90, one to 75, a half
  to 10, and three below 10. Parameter 4 is a step of exactly one sample
  value and saved 3 percent over lossless on gray images, so the scale starts
  at 5. Below 32 only multiples of four quantize by whole sample values; the
  parameter just under each multiple coded larger with more luma error.
- **Chroma.** The chroma step is twice the luma step. At equal RGB PSNR that
  coded 7 to 13 percent smaller than equal parameters, within about one
  percent of the best offset at every quality.
- **Bands.** DC, low-pass and high-pass share a parameter up to level 64,
  where offsets of a few parameters between them changed the size at equal
  PSNR by under one percent. Above it DC keeps half of the excess, at most 16 parameters, which
  coded up to 7 percent smaller at equal PSNR.
- **Overlap.** First-level overlap from 98 to 7: coding without overlap cost
  4 to 16 percent more bits at equal PSNR, and second-level overlap up to 10
  percent more. Quality 99 codes without overlap, because at its luma step
  overlap coded synthetic noise larger than lossless. Quality 6 and below
  use second-level overlap: first-level overlap leaves a visible grid of
  macroblock edges there, which second-level overlap removes at a cost of
  about 0.25 dB. Subsampled chroma takes second-level overlap only in an
  image, and under hard tiling in tiles, wider than one macroblock; a
  narrower one is coded with first-level overlap at those qualities. An
  image the scale codes losslessly is coded without overlap.

**Components.** A component is either at the sample range of the source or
a difference the encoder's colour transform forms between source channels,
which spans twice that range:

| Source | Luma parameters | Chroma parameters |
| --- | --- | --- |
| Gray, and any alpha plane | the one component | |
| RGB and RGBE, coded as YUV | Y | U and V, at twice the luma step |
| RGB with `chroma_sampling` 4:2:2 or 4:2:0 | Y | U and V, eight or four parameters on: about 1.4 or 1.2 times the luma step |
| CMYK, coded as YUVK | the fourth component | the first three, at twice the luma step |
| CMYK coded as `CMYKDIRECT`, N-component | every channel | |
| YUV 4:4:4, 4:2:2 and 4:2:0 sources | Y, U and V | |

- The transform of CMYK leaves the mean of black and the other inks in its
  last component, and its first three are differences. Taking the first for
  luma, as a request's explicit quantizers do, coded the calibration set
  with 1.1 to 2.3 dB less PSNR over the four inks at equal size.
- A subsampled difference stands for two or four pixels, so an error in it
  weighs that much more. The offsets are close to those the T.835 reference
  encoder derives chroma quantizers with. At 4:2:0 they gained up to 0.2 dB
  of RGB PSNR at equal size over the doubled step, which also coded one
  calibration image larger on one step down the scale; at 4:2:2 the gain
  was under 0.1 dB.
- The chroma of a YUV source is stored at the range of its luma, not twice
  it, and an error in it weighs as much in RGB as an error in luma does.
  Coded at the luma step it kept 0.2 to 0.6 dB more PSNR over the stored
  samples at equal size than at twice that, at every sampling.
- Alpha is coded as a gray image of the alpha samples at the same quality.
  On an 8-bit matte its PSNR stays within 1.1 dB of the luma PSNR of the
  colour beside it down to quality 10.

**Sample range.** Steps double every sixteen parameters from 16 up, so a
range of 255 times a power of two adds sixteen parameters for each doubling;
the parameter is looked up in the decoder's table of steps, which also
serves the linear part below 16. The range is the distance between the coded
values of black and of nominal white:

| Samples | Range |
| --- | ---: |
| 1-bit | 1 |
| Packed RGB 555, 565 | 31, 63 |
| 8-bit | 255 |
| 10-bit, packed RGB 10-10-10 | 1023 |
| 16-bit unsigned, less `SHIFT_BITS` | 65535 |
| 16-bit fixed point (s2.13), zero to one, less `SHIFT_BITS` | 8192 |
| 32-bit fixed point (s7.24), zero to one, less the default ten `SHIFT_BITS` | 16384 |
| Half float | 8192 |
| 32-bit float at the default 13 mantissa bits and exponent bias 4 | 16384 |
| RGBE | 1024 |

- A float is coded by its bit pattern, which rises by a fixed count, two to
  the mantissa bits, for each doubling of the value. Eight octaves of
  patterns stand for the 8-bit range. That count was measured on linear
  light with the error taken after the sRGB transfer function: it kept the
  PSNR of half-float and RGBE pictures within 1.3 dB of the 8-bit picture's
  at every quality down to 10, and a whole number of octaves keeps every
  parameter on the table's own grid.
- A 32-bit float is coded linearly below `2^(1 - exponent bias)`, which at
  the default bias of four is most of a picture, and linear coding of linear
  light needs finer steps in the shadows. The range is eight octaves from a
  bias of six up and halves with each step of bias below that: two octaves
  at the default. With 13 mantissa bits the PSNR at quality 75 was then 1.2
  dB above the 8-bit picture's at biases 0 to 3, 0.9 dB above at 4, equal at
  5, 2.1 dB below at 6 and 0.9 dB below from 8 up. The range follows the
  mantissa bits: 16 of them at bias 4 coded to the same size and error as
  13.
- A float alpha sample takes two octaves, or the patterns up to one where
  the format codes those linearly and they are fewer. Coverage counts by its
  differences rather than its ratios, and its top octave is half of its
  range. On a half-float matte that kept alpha PSNR between 5 dB above and
  1 dB below the 8-bit matte's; four octaves fell 8 dB below it at quality
  0.
- A range below 255 has few steps. It is held to the parameters the 8-bit
  luma scale uses below 32, which are 5 and the multiples of four, and every
  band and component follows the luma parameter alone. A quality whose step
  comes down to one sample value is coded losslessly, without overlap:
  scaled arithmetic at steps that fine codes larger than exact coding does.
- Parameters end at 255. 16-bit chroma reaches it at quality 0. A range
  above 2^17 reaches it higher up the scale and is coded the same from there
  down: 32-bit fixed point with fewer than seven `SHIFT_BITS`, or a 32-bit
  float with more than 16 mantissa bits at the default exponent bias.

**Where neighbouring qualities code the same file.** Quantization parameters
are whole numbers, and the repeats below follow from that; `jxr-quality-curve`
counts them:

- From 75 down to 10 chroma differences move on odd qualities and everything
  else on even ones. A source without differences has one parameter to move
  and repeats its output on 32 of the 100 steps: every odd quality from 73
  down to 11 codes as the even quality above it. That is gray, alpha, YUV
  sources, direct CMYK and N-component images. Moving one band of a gray
  image on the odd qualities instead was measured and rejected: on 16
  pictures, 7 of 528 such pairs of steps had a step that did not shrink the
  file when the high-pass band moved first, and 9 when it moved second.
- Float sources repeat on the same 32 steps. A step of the differences alone
  moved the size of a float picture by less than the noise of coding its bit
  patterns, and grew one RGBE file by 1.2 percent, so their differences move
  with luma.
- RGBE is coded at quality 10 from there down. An RGBE pattern steps from
  zero straight to the exponent of the smallest value the format holds, and
  the steep segment below 10 rings on that cliff by whole exponents: PSNR
  fell to 9 dB at quality 0, where the 8-bit picture keeps 21.
- Packed RGB 565 is lossless from 96 up and has 37 distinct codings below
  that; 555 is lossless from 91 up and has 20. A 1-bit image is lossless at
  every quality.

#### Measurements

Measured with `jxr-quality-curve` on images from the T.834 suite. The colour
set is `Seattle_IFD_Tags.jxr`, which is lossless, and `Dog.jxr`, `Boats.jxr`,
`Skyscraper.jxr` and `Bridge_MBQP2.jxr` halved; the gray set is the four
`*_YONLY.jxr` images of `Entropy_Table_Coverage` halved. Halving removes the
block grid of a source that was itself coded lossily. Sizes are geometric
means in bits per pixel, PSNR and SSIM are means, and the error is the
largest over the set. Quantizers are DC/low-pass/high-pass parameters.

The 8-bit scale, on the colour set as RGB and the gray set:

| Quality | Luma | Chroma | Overlap | Colour bpp | RGB PSNR | Luma PSNR | Luma SSIM | Max error | Gray bpp | PSNR | SSIM | Max error |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 1/1/1 | 1/1/1 | none | 12.063 | exact | exact | 1.0000 | 0 | 4.808 | exact | 1.0000 | 0 |
| 99 | 5/5/5 | 10/10/10 | none | 8.271 | 51.86 | 55.13 | 0.9991 | 4 | 4.336 | 55.06 | 0.9991 | 2 |
| 98 | 8/8/8 | 16/16/16 | one | 6.366 | 48.97 | 51.87 | 0.9983 | 5 | 3.550 | 51.73 | 0.9981 | 3 |
| 97 | 12/12/12 | 24/24/24 | one | 5.138 | 46.08 | 48.81 | 0.9969 | 9 | 3.029 | 48.83 | 0.9967 | 5 |
| 96 | 16/16/16 | 32/32/32 | one | 4.325 | 44.02 | 46.59 | 0.9953 | 13 | 2.663 | 46.69 | 0.9951 | 6 |
| 95 | 20/20/20 | 36/36/36 | one | 3.768 | 42.45 | 44.84 | 0.9934 | 16 | 2.396 | 44.97 | 0.9932 | 8 |
| 90 | 35/35/35 | 51/51/51 | one | 2.394 | 37.94 | 39.79 | 0.9829 | 27 | 1.664 | 39.93 | 0.9827 | 17 |
| 85 | 40/40/40 | 56/56/56 | one | 2.013 | 36.47 | 38.21 | 0.9771 | 46 | 1.446 | 38.34 | 0.9771 | 19 |
| 80 | 45/45/45 | 61/61/61 | one | 1.718 | 35.18 | 36.75 | 0.9702 | 39 | 1.260 | 36.86 | 0.9702 | 24 |
| 75 | 50/50/50 | 66/66/66 | one | 1.438 | 33.85 | 35.32 | 0.9611 | 45 | 1.080 | 35.42 | 0.9614 | 30 |
| 70 | 53/53/53 | 68/68/68 | one | 1.270 | 32.94 | 34.24 | 0.9524 | 48 | 0.950 | 34.35 | 0.9531 | 32 |
| 60 | 58/58/58 | 73/73/73 | one | 1.028 | 31.60 | 32.77 | 0.9376 | 56 | 0.792 | 32.90 | 0.9393 | 46 |
| 50 | 63/63/63 | 78/78/78 | one | 0.860 | 30.52 | 31.59 | 0.9222 | 82 | 0.676 | 31.76 | 0.9257 | 50 |
| 40 | 66/68/68 | 82/83/83 | one | 0.656 | 29.06 | 29.96 | 0.8942 | 111 | 0.525 | 30.14 | 0.9003 | 53 |
| 30 | 69/73/73 | 84/88/88 | one | 0.506 | 27.84 | 28.63 | 0.8640 | 100 | 0.413 | 28.82 | 0.8729 | 69 |
| 25 | 70/75/75 | 86/91/91 | one | 0.455 | 27.42 | 28.19 | 0.8521 | 104 | 0.379 | 28.40 | 0.8630 | 81 |
| 20 | 71/78/78 | 87/93/93 | one | 0.405 | 26.91 | 27.62 | 0.8349 | 120 | 0.335 | 27.82 | 0.8465 | 86 |
| 10 | 74/83/83 | 89/98/98 | one | 0.301 | 25.83 | 26.43 | 0.7929 | 116 | 0.252 | 26.63 | 0.8086 | 96 |
| 5 | 82/98/98 | 97/113/113 | two | 0.131 | 23.16 | 23.62 | 0.6641 | 156 | 0.112 | 23.78 | 0.6887 | 135 |
| 0 | 97/113/113 | 112/128/128 | two | 0.062 | 21.25 | 21.69 | 0.5484 | 197 | 0.052 | 21.72 | 0.5729 | 151 |

Every other kind of source is measured as a class of the tool: the same
pictures, stored another way. A class codes the picture, decodes it, and
takes the error on one scale for all, where 255 is the class's peak.

| Class | The picture stored as | Peak |
| --- | --- | --- |
| `gray8`, `rgb8` | itself | 255 |
| `rgb8-422`, `rgb8-420` | itself, chroma subsampled by the encoder | 255, on the RGB samples |
| `rgba8` | RGB with an alpha matte made from its luma: `(luma - 64) * 2`, clipped | 255; alpha apart from colour |
| `gray16`, `rgb16` | each sample times 257, plus low-order detail | 65535 |
| `rgb-fixed16`, `rgb-fixed32` | the 16-bit samples scaled so that 65535 is one | one: 8192, 16777216 |
| `rgb-half`, `rgb-float`, `rgbe` | linear light of the 16-bit samples read as sRGB | one, after the sRGB transfer function |
| `rgba-half` | that linear light with the `rgba8` matte as a float | as `rgb-half`; alpha against one |
| `rgb101010`, `rgb565`, `rgb555` | the 16-bit samples rounded to the fields | 1023; 31 and 63; 31 |
| `bw1` | the gray picture dithered to black and white | 1 |
| `cmyk8`, `cmyk8-direct`, `cmyk16` | inks `255 - R`, `255 - G`, `255 - B` with their minimum moved to black | 255 or 65535, on the four inks |
| `channels8` | red, green, blue and Rec. 601 luma as four channels | 255 |
| `yuv444`, `yuv422`, `yuv420` | full-range Rec. 601 YCbCr, chroma averaged where subsampled | 255, on the stored Y, Cb and Cr |

The low-order detail of the deep classes is pseudo-random and uniform, up to
half an 8-bit step either way: what an 8-bit picture has rounded off and the
low byte of a 16-bit photograph mostly holds. It makes the deep classes 1.5
to 3 times as large as `rgb8` when lossless and leaves the picture what the
8-bit classes code. The float classes hold linear light because that is
what float formats are for, and their error is taken after the sRGB transfer
function so that it stands on the scale of the 8-bit picture's.

The target is a PSNR within 1 dB of the 8-bit class that holds the same
picture, `rgb8` for the RGB classes and `gray8` or the luma of `rgb8` for a
luma plane, wherever the source has the steps for it, in a file of
comparable size. Size in bits per pixel:

| Class | 100 | 99 | 95 | 90 | 75 | 50 | 25 | 10 | 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gray8` | 4.808 | 4.336 | 2.396 | 1.664 | 1.080 | 0.676 | 0.379 | 0.252 | 0.052 |
| `rgb8` | 12.063 | 8.271 | 3.768 | 2.394 | 1.438 | 0.860 | 0.455 | 0.301 | 0.062 |
| `rgb8-422` | 8.331 | 6.549 | 3.193 | 2.103 | 1.313 | 0.809 | 0.445 | 0.295 | 0.059 |
| `rgb8-420` | 6.591 | 5.537 | 2.813 | 1.897 | 1.211 | 0.758 | 0.421 | 0.280 | 0.058 |
| `rgba8` | 17.326 | 13.141 | 6.652 | 4.523 | 2.936 | 1.890 | 1.106 | 0.762 | 0.145 |
| `gray16` | 13.155 | 4.407 | 2.394 | 1.680 | 1.080 | 0.677 | 0.380 | 0.253 | 0.052 |
| `rgb16` | 36.890 | 8.488 | 3.762 | 2.409 | 1.438 | 0.861 | 0.457 | 0.302 | 0.063 |
| `rgb-fixed16` | 27.573 | 8.491 | 3.763 | 2.410 | 1.438 | 0.861 | 0.457 | 0.302 | 0.063 |
| `rgb-fixed32` | 30.680 | 8.490 | 3.763 | 2.409 | 1.438 | 0.861 | 0.457 | 0.302 | 0.063 |
| `rgb-half` | 28.711 | 9.785 | 4.534 | 2.917 | 1.738 | 1.023 | 0.552 | 0.360 | 0.070 |
| `rgba-half` | 39.694 | 17.471 | 10.481 | 7.972 | 5.946 | 4.541 | 3.420 | 2.850 | 1.364 |
| `rgb-float` | 34.753 | 12.666 | 6.493 | 4.533 | 2.979 | 1.959 | 1.205 | 0.859 | 0.182 |
| `rgbe` | 19.764 | 10.087 | 4.941 | 3.297 | 2.067 | 1.304 | 0.767 | 0.545 | 0.545 |
| `rgb101010` | 18.314 | 8.517 | 3.765 | 2.409 | 1.439 | 0.861 | 0.456 | 0.302 | 0.063 |
| `rgb565` | 8.173 | 8.173 | 4.139 | 2.947 | 1.616 | 0.819 | 0.439 | 0.288 | 0.057 |
| `rgb555` | 6.205 | 6.205 | 6.205 | 2.346 | 1.644 | 0.824 | 0.427 | 0.310 | 0.057 |
| `bw1` | 1.529 | 1.529 | 1.529 | 1.529 | 1.529 | 1.529 | 1.529 | 1.529 | 1.529 |
| `cmyk8` | 16.444 | 11.123 | 4.773 | 2.882 | 1.593 | 0.880 | 0.422 | 0.282 | 0.086 |
| `cmyk8-direct` | 14.160 | 12.009 | 5.469 | 3.450 | 2.055 | 1.208 | 0.644 | 0.424 | 0.104 |
| `cmyk16` | 49.463 | 11.418 | 4.765 | 2.899 | 1.596 | 0.882 | 0.423 | 0.283 | 0.087 |
| `channels8` | 19.788 | 17.752 | 9.610 | 6.630 | 4.295 | 2.694 | 1.480 | 0.961 | 0.160 |
| `yuv444` | 10.714 | 8.821 | 3.843 | 2.414 | 1.440 | 0.851 | 0.455 | 0.298 | 0.064 |
| `yuv422` | 7.921 | 6.741 | 3.189 | 2.076 | 1.282 | 0.778 | 0.425 | 0.281 | 0.058 |
| `yuv420` | 6.493 | 5.666 | 2.851 | 1.902 | 1.199 | 0.740 | 0.409 | 0.273 | 0.058 |

PSNR over all colour samples against the class's peak, in dB:

| Class | 100 | 99 | 95 | 90 | 75 | 50 | 25 | 10 | 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gray8` | exact | 55.06 | 44.97 | 39.93 | 35.42 | 31.76 | 28.40 | 26.63 | 21.72 |
| `rgb8` | exact | 51.86 | 42.45 | 37.94 | 33.85 | 30.52 | 27.42 | 25.83 | 21.25 |
| `rgb8-422` | 39.86 | 39.64 | 38.03 | 35.88 | 33.04 | 30.16 | 27.29 | 25.74 | 20.97 |
| `rgb8-420` | 37.46 | 37.36 | 36.35 | 34.79 | 32.49 | 29.95 | 27.23 | 25.71 | 21.11 |
| `rgba8` | exact | 51.86 | 42.45 | 37.94 | 33.85 | 30.52 | 27.42 | 25.83 | 21.25 |
| `gray16` | exact | 56.56 | 45.03 | 40.10 | 35.43 | 31.78 | 28.42 | 26.64 | 21.72 |
| `rgb16` | exact | 52.87 | 42.47 | 38.05 | 33.86 | 30.54 | 27.43 | 25.85 | 21.27 |
| `rgb-fixed16` | exact | 52.87 | 42.47 | 38.04 | 33.85 | 30.53 | 27.42 | 25.83 | 21.24 |
| `rgb-fixed32` | 89.43 | 52.87 | 42.46 | 38.04 | 33.85 | 30.53 | 27.42 | 25.83 | 21.24 |
| `rgb-half` | exact | 51.80 | 41.35 | 36.99 | 32.95 | 29.74 | 26.81 | 25.16 | 20.69 |
| `rgba-half` | exact | 51.80 | 41.35 | 36.99 | 32.95 | 29.74 | 26.81 | 25.16 | 20.69 |
| `rgb-float` | 96.53 | 54.49 | 43.62 | 39.05 | 34.79 | 31.43 | 28.39 | 26.81 | 21.78 |
| `rgbe` | exact | 51.18 | 41.29 | 36.95 | 32.89 | 29.61 | 26.46 | 24.58 | 24.58 |
| `rgb101010` | exact | 52.83 | 42.47 | 38.04 | 33.85 | 30.54 | 27.43 | 25.84 | 21.26 |
| `rgb565` | exact | exact | 37.33 | 35.86 | 32.92 | 29.48 | 26.81 | 25.28 | 20.32 |
| `rgb555` | exact | exact | exact | 35.02 | 33.08 | 29.53 | 26.67 | 25.53 | 20.31 |
| `bw1` | exact | exact | exact | exact | exact | exact | exact | exact | exact |
| `cmyk8` | exact | 51.90 | 42.38 | 38.07 | 34.15 | 31.31 | 28.80 | 27.70 | 24.13 |
| `cmyk8-direct` | exact | 55.74 | 46.55 | 41.94 | 37.91 | 34.59 | 31.63 | 30.05 | 25.55 |
| `cmyk16` | exact | 52.74 | 42.35 | 38.09 | 34.14 | 31.30 | 28.80 | 27.70 | 24.15 |
| `channels8` | exact | 54.66 | 44.88 | 39.91 | 35.44 | 31.70 | 28.30 | 26.51 | 21.69 |
| `yuv444` | exact | 55.10 | 46.25 | 41.92 | 37.99 | 34.74 | 31.79 | 30.21 | 25.67 |
| `yuv422` | exact | 54.99 | 45.73 | 41.19 | 37.07 | 33.62 | 30.52 | 28.85 | 23.73 |
| `yuv420` | exact | 54.96 | 45.36 | 40.66 | 36.42 | 32.86 | 29.63 | 27.92 | 23.00 |

Luma PSNR, for the classes that have luma: Rec. 601 luma of RGB, the Y plane
of YUV, or the one component:

| Class | 100 | 99 | 95 | 90 | 75 | 50 | 25 | 10 | 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gray8` | exact | 55.06 | 44.97 | 39.93 | 35.42 | 31.76 | 28.40 | 26.63 | 21.72 |
| `rgb8` | exact | 55.13 | 44.84 | 39.79 | 35.32 | 31.59 | 28.19 | 26.43 | 21.69 |
| `rgb8-422` | 51.12 | 50.19 | 43.94 | 39.46 | 35.20 | 31.54 | 28.18 | 26.42 | 21.67 |
| `rgb8-420` | 49.11 | 48.57 | 43.43 | 39.26 | 35.13 | 31.51 | 28.18 | 26.42 | 21.68 |
| `rgba8` | exact | 55.13 | 44.84 | 39.79 | 35.32 | 31.59 | 28.19 | 26.43 | 21.69 |
| `gray16` | exact | 56.56 | 45.03 | 40.10 | 35.43 | 31.78 | 28.42 | 26.64 | 21.72 |
| `rgb16` | exact | 56.32 | 44.85 | 39.95 | 35.32 | 31.60 | 28.21 | 26.44 | 21.70 |
| `rgb-fixed16` | exact | 56.33 | 44.85 | 39.95 | 35.32 | 31.59 | 28.20 | 26.43 | 21.68 |
| `rgb-fixed32` | 90.11 | 56.32 | 44.85 | 39.95 | 35.32 | 31.59 | 28.20 | 26.43 | 21.68 |
| `rgb-half` | exact | 55.42 | 44.22 | 39.32 | 34.73 | 31.04 | 27.72 | 25.91 | 21.23 |
| `rgba-half` | exact | 55.42 | 44.22 | 39.32 | 34.73 | 31.04 | 27.72 | 25.91 | 21.23 |
| `rgb-float` | 100.36 | 58.38 | 46.92 | 42.08 | 37.45 | 33.74 | 30.32 | 28.48 | 22.64 |
| `rgbe` | exact | 54.87 | 44.18 | 39.33 | 34.72 | 31.03 | 27.64 | 25.74 | 25.74 |
| `rgb101010` | exact | 56.33 | 44.86 | 39.95 | 35.32 | 31.60 | 28.21 | 26.44 | 21.70 |
| `rgb565` | exact | exact | 41.71 | 39.61 | 35.46 | 31.04 | 27.87 | 26.18 | 21.15 |
| `rgb555` | exact | exact | exact | 37.65 | 35.25 | 30.92 | 27.65 | 26.37 | 21.10 |
| `bw1` | exact | exact | exact | exact | exact | exact | exact | exact | exact |
| `yuv444` | exact | 54.73 | 44.90 | 39.92 | 35.45 | 31.70 | 28.29 | 26.50 | 21.73 |
| `yuv422` | exact | 54.73 | 44.90 | 39.92 | 35.45 | 31.70 | 28.29 | 26.50 | 21.73 |
| `yuv420` | exact | 54.73 | 44.90 | 39.91 | 35.44 | 31.70 | 28.28 | 26.50 | 21.72 |

Alpha PSNR, for the classes with alpha:

| Class | 100 | 99 | 95 | 90 | 75 | 50 | 25 | 10 | 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rgba8` | exact | 55.74 | 45.55 | 40.35 | 35.60 | 31.53 | 27.64 | 25.37 | 18.95 |
| `rgba-half` | exact | 60.88 | 49.02 | 44.03 | 38.97 | 34.74 | 30.46 | 27.96 | 18.05 |

What the measurements show:

- No class coded any picture larger, or with less error, at one quality than
  at the quality above it: 0 of 100 steps on each of the 4 or 5 pictures of
  each of the 24 classes. The steps that repeat the output above are the ones
  listed earlier: 32 for each picture of `gray8`, `gray16`, the YUV classes,
  `cmyk8-direct`, `channels8` and the float classes, 42 for `rgbe`, 63 for
  `rgb565`, 80 for `rgb555`, all 100 for `bw1`, and none elsewhere.
- On every picture of `rgb8` each of the 100 steps down the scale gave a
  smaller file and a larger squared error. A step shrinks the colour set by
  0.7 to 23 percent below quality 99, and by 31 percent from lossless to 99.
- Deeper integer and fixed-point samples code as the 8-bit picture does.
  From 95 down, `rgb16`, `rgb-fixed16`, `rgb-fixed32` and `rgb101010` are
  within 0.11 dB and 2 percent in size of `rgb8`, and `gray16` within 0.2 dB
  of `gray8`. At 99 they keep 1 to 1.5 dB more, because the 8-bit picture
  also rounds its output to whole sample values at a step of 1.25. Before
  the range was taken into account a 16-bit RGB image of the suite kept 48
  percent of its lossless size at quality 75; `rgb16` now codes at 1.44 bits
  per pixel there, as `rgb8` does, which is 4 percent of its lossless size.
- Float classes follow within about 1 dB. `rgb-half` is 0.6 to 1.1 dB below
  `rgb8` from 95 down in files a fifth larger, and `rgbe` 0.7 to 1.25 dB
  below down to 10 in files 1.2 to 1.8 times as large. `rgb-float` is 0.5 to
  1.2 dB above from 95 down and 2.6 dB above at 99, in files 1.7 to 2.9
  times as large: the linear part of its default coding spends its patterns
  evenly on light that the eye, and the sRGB function, weigh towards the
  shadows.
- `cmyk8` and `cmyk16` are within 0.8 dB of `rgb8` down to quality 50 and
  above it below that; the inks of this separation carry less than the RGB
  they were made from. Channels coded as they are stored keep the luma step
  on every channel: `channels8` is within 0.2 dB of `gray8` from 95 down,
  and the Y plane of the YUV classes within 0.2 dB of the luma of `rgb8`
  below 99.
- Subsampled chroma keeps luma: from 90 down the luma PSNR of `rgb8-420` is
  within 0.55 dB of `rgb8`, in files 6 to 21 percent smaller. Its RGB PSNR is
  bounded by what decimation removes, 37.5 dB on this set.
- Packed samples are within 1.1 dB of `rgb8` from 75 down, and coarser than
  it near the top, where the first lossy step of a 5-bit sample is a step of
  ten 8-bit values.
- Luma PSNR and SSIM do not move on a chroma-only step, apart from noise of
  at most 0.003 dB and 0.0003.
- The largest sample error grows down the scale but not at every step.
- The numbers track baseline JPEG at the same quality. The reference was a
  model of it on the same pixels: the standard tables scaled as libjpeg
  scales them, 4:2:0 chroma for colour, ideal entropy coding. Luma PSNR was
  within 1.3 dB of the model at every quality checked from 99 to 1 and
  within 0.5 dB at 95, 90, 50 and 10. At the same number the colour files
  were 3 to 30 percent smaller than the model's from 90 down to 10, and
  larger above 90, where they carry 4:4:4 chroma that 4:2:0 JPEG does not.

Limits:

- Sizes follow from image content. An image that is mostly noise gains
  little from the top of the scale: on synthetic noise quality 99 was as
  little as 4 percent smaller than lossless.
- An image that is exact in a few coefficients of unscaled arithmetic is
  cheapest coded losslessly, and codes larger at some lower quality. The
  suite has 16 such sources, decoded from codestreams without overlap and
  without a high-pass band. They are constant over whole blocks of the
  4-sample or 16-sample grid but for a ripple of one sample value that the
  inverse transform's rounding leaves, which lossless coding reproduces from
  the same few coefficients and scaled arithmetic codes as texture; the
  overlap filter of the lossy qualities then spreads their block edges into
  high-pass coefficients. They coded 3.3 to 5.8 times their lossless size at
  98, the first quality with overlap, and stayed above it down to somewhere
  between 75 and 0. The encoder does not look for such images: finding one
  would take a second transform of every image.
- Small files move by bytes. `jxr-encode-check --quality-scale` codes each
  of the 537 images of the suite in its own pixel format at every quality,
  in each of its coded forms: 137,500 steps down the scale in 1375 forms.
  1044 of them grew the file, 883 of those on images of 64 by 64 samples,
  whose files are a few hundred bytes:
  - 86 at 99 or 98, the first step of scaled arithmetic and the first with
    overlap filtering: on the 16 sources above, on 30 N-component images,
    which grew 0.2 to 8 percent at 98, and on four other small images.
  - 46 at quality 6, where second-level overlap starts, all on images of 64
    by 64 samples: 0.3 to 16 percent.
  - 912 elsewhere, none by more than 2.2 percent, 692 by less than half a
    percent and 725 by one or two bytes. 19 of them are on files over 5000
    bytes, the largest by 73 bytes of 13,527.
  The 52 synthetic sources of the check, flat fields, single pixels and
  noise among them, grew on 44 steps in 30 of their 90 forms: 21 at 99 or
  98, by up to 89 percent, and 23 elsewhere, by at most 2.3 percent.
- At the seven qualities `jxr-encode-check` codes every image of the suite
  at in its own pixel format, which are 100, 99, 92, 75, 40, 5 and 0, 510
  of the 537 sources shrank at every step and only the 16 above grew. The
  other 11 repeat a size where the rules above say they do: six packed 5-bit
  and 6-bit images at the top of the scale, the RGBE image below 10, two
  1-bit images throughout, and two images of a single row or column of
  sixteen samples that reach 193 bytes by quality 75 and stay there. Before
  the range was taken into account, fourteen 16-bit N-component images also
  coded larger at 92 than at 99.
- A float picture with exact zeros beside small values codes badly at the
  bottom of the scale in any format whose patterns step from zero to a
  distant exponent. RGBE is held at quality 10 for that reason. A 32-bit
  float coded with an exponent bias far above the default has the same
  cliff and is not held; the default bias of four does not have it.
- Float alpha is costly. The `rgba-half` matte took 4.2 bits per pixel at
  quality 75 where the 8-bit matte took 1.5, because every step of coverage
  near zero is a large step of the bit pattern.
- Float alpha in 32-bit float images follows the rule measured on half
  float and was not measured itself.

Regenerate the tables after any change to quantization, prediction or
entropy coding, with the suite in `target/t834-conformance/suite-2014`. One
run measures every class the pictures can be stored as, the colour pictures
as the colour classes and the gray ones as the gray classes:

```sh
S=target/t834-conformance/suite-2014
E=$S/Entropy_Table_Coverage
cargo run --release -p jxr-test-support --bin jxr-quality-curve -- \
  $S/Tag_Based_Container/Seattle_IFD_Tags.jxr \
  --halve $E/Dog.jxr $E/Boats.jxr $E/Skyscraper.jxr $S/MBLevel_QP_Coverage/Bridge_MBQP2.jxr \
  $E/Dog_YONLY.jxr $E/Boats_YONLY.jxr $E/Skyscraper_YONLY.jxr $E/Seattle_YONLY.jxr
```

`--class rgb16,rgb-half` limits the run to the named classes, `--rows
99,75,50` limits each table to those qualities, `--float-precision 13,6`
codes `rgb-float` with those mantissa bits and that exponent bias, and
`--help` lists the classes with their peaks. Each class prints its table, with the quantizers
its files declare and the number of pictures that grew, lost error or
repeated on each step, and then a summary line. The tool exits with an error
if any picture of any class grows, or loses error, from one quality to the
next lower one. `jxr-encode-check --quality-scale` reports the steps that
grow, for sizes alone, on any image in its own pixel format.

### Encode throughput

`jxr-encode-bench` times `jxr::encode` end to end, from source samples to an
Annex-A file in memory, on synthetic images of six pixel formats, each coded
losslessly and at quality 75. The images are twelve megapixels unless
`--size` says otherwise: triangle waves at every scale under light noise,
which code at about 14 bits per pixel losslessly and 0.9 at quality 75 as
8-bit RGB. `--decode` also times decoding each encoded file back to the
source's pixel format, and `--tile N` sets the largest tile edge in
macroblocks instead of the default; 255 codes these images as one tile.
`--setting lossless` or `--setting 75` codes at one setting only, so that the
process's peak resident size is that setting's. Two workloads code
premultiplied 8-bit BGRA in the two shapes Flash content is stored in.

```sh
cargo run --release -p jxr-test-support --bin jxr-encode-bench
cargo run --release -p jxr-test-support --bin jxr-encode-bench -- --size 1024x1024 --only rgb
RAYON_NUM_THREADS=1 cargo run --release -p jxr-test-support --bin jxr-encode-bench -- --decode --tile 255
```

The tables below are from one machine, an Apple M5 Max, and are measurements
to compare changes against, not guarantees. Each figure is the median of ten
calls after one cold call and two warm-ups, the timed scope being the whole
`encode` or `decode` call. "Before" and "after" are the encoder before and
after its tile writer and tile default were reworked, timed by the same
benchmark source built against each. Other work was running on the machine,
so a difference of a few percent between two figures is noise: the decoder,
for one, did not change, and timed in alternation on the same file it took
247 ms before and 244 ms after.

One thread (`RAYON_NUM_THREADS=1`), milliseconds for the twelve-megapixel
image. This is the serial path: every stage in turn on one thread, as on a
target without threads. The tile size makes no difference to it (8-bit RGB
lossless is 237 ms as one tile and 236 ms as 48):

| Source                 | Lossless encode before | after | decode | Quality 75 encode before | after | decode |
| ---------------------- | ---------------------: | ----: | -----: | -----------------------: | ----: | -----: |
| 8-bit gray             |                    132 |    68 |     80 |                       56 |    44 |     40 |
| 8-bit RGB              |                    469 |   236 |    247 |                      139 |   119 |    101 |
| 8-bit RGBA             |                    605 |   305 |    329 |                      198 |   165 |    143 |
| 16-bit RGB             |                    415 |   242 |    259 |                      445 |   237 |    251 |
| 32-bit float RGB       |                    537 |   294 |    382 |                      547 |   297 |    379 |
| 8-bit planar YUV 4:2:0 |                    232 |   107 |    128 |                      102 |    75 |     67 |

The single-thread tests also covered a 4096 x 4096 8-bit RGBA image coded
as one tile in two Flash layouts. The frequency-mode file used first-level
overlap and separate alpha. The spatial-mode file used no overlap and
integrated alpha. The peak-heap figures below used
the allocation report's former four-octave input. The timings used six
octaves. Both tools now share a generator as described under [Memory](#memory).

| 4096 x 4096 RGBA, one tile, one thread   | Encode before | after | Decode | Peak heap before | after  |
| ---------------------------------------- | ------------: | ----: | -----: | ---------------: | -----: |
| Quality 75, frequency, separate alpha    |        306 ms |   238 |    234 |           340 MB | 271 MB |
| Quality 75, spatial, integrated alpha    |        332 ms |   237 |    200 |           340 MB | 340 MB |
| Lossless, frequency, separate alpha      |        878 ms |   442 |    478 |           382 MB | 271 MB |
| Lossless, spatial, integrated alpha      |        928 ms |   439 |    466 |           416 MB | 340 MB |

Eighteen threads, milliseconds for the twelve-megapixel image. "One tile" is
the reworked encoder with `--tile 255`, which shows the serial tile writer;
the default tiles add the parallel entropy coding, in the encoder and in the
decoder of its files:

| Lossless               | Encode before | one tile | default tiles | Decode before | default tiles |
| ---------------------- | ------------: | -------: | ------------: | ------------: | ------------: |
| 8-bit gray             |           129 |       54 |           6.9 |            70 |           9.9 |
| 8-bit RGB              |           445 |      178 |          21.7 |           211 |          22.6 |
| 8-bit RGBA             |           576 |      235 |          29.2 |           280 |          52.8 |
| 16-bit RGB             |           390 |      187 |          38.3 |           218 |          28.8 |
| 32-bit float RGB       |           435 |      185 |          33.5 |           348 |           207 |
| 8-bit planar YUV 4:2:0 |           189 |       88 |          12.8 |           100 |          15.7 |

| Quality 75             | Encode before | one tile | default tiles | Decode before | default tiles |
| ---------------------- | ------------: | -------: | ------------: | ------------: | ------------: |
| 8-bit gray             |          44.2 |     26.6 |           5.1 |          28.9 |           7.2 |
| 8-bit RGB              |          81.5 |     50.9 |          12.9 |          54.5 |          12.5 |
| 8-bit RGBA             |           125 |     77.9 |          28.6 |          81.8 |          25.5 |
| 16-bit RGB             |           375 |      169 |          34.1 |           202 |          25.9 |
| 32-bit float RGB       |           440 |      189 |          57.5 |           332 |           153 |
| 8-bit planar YUV 4:2:0 |          62.6 |     41.3 |          10.3 |          38.2 |          11.5 |

#### Tile size

A tile is the only part of a JPEG XR image that is entropy coded
independently, so the largest tile edge decides how far coding and decoding
spread over threads, and each tile costs bytes: its prediction and its
adaptive code tables start over. The 8-bit RGB benchmark image on eighteen
threads, by largest tile edge in macroblocks:

| Tile edge      | Lossless bytes | encode ms | decode ms | Quality 75 bytes | encode ms | decode ms |
| -------------- | -------------: | --------: | --------: | ---------------: | --------: | --------: |
| 255 (one tile) |     20,985,504 |       179 |       208 |        1,381,423 |      51.4 |      53.4 |
| 128            |         +0.01% |      53.2 |      59.4 |           +0.08% |      21.1 |      20.9 |
| 64             |         +0.05% |      28.0 |      27.9 |           +0.27% |      15.7 |      14.3 |
| 48             |         +0.09% |      25.6 |      27.1 |           +0.48% |      14.1 |      14.4 |
| 32 (default)   |         +0.15% |      23.6 |      24.0 |           +0.86% |      14.3 |      13.8 |
| 24             |         +0.25% |      22.7 |      23.8 |           +1.47% |      14.2 |      13.6 |
| 16             |         +0.48% |      22.9 |      23.5 |           +2.79% |      14.1 |      13.9 |
| 8              |         +1.35% |      23.3 |      23.9 |           +7.84% |      14.5 |      14.0 |

Every image of the T.834 suite re-encoded under every setting of
`jxr-encode-check` that leaves the tile size to the default, counting only
the files a tile size splits, grows by 0.7 percent losslessly and 1.2 percent
lossily at 64 macroblocks, 1.0 and 0.9 percent at 32, 2.0 and 1.7 percent at
16, and 4.4 and 3.5 percent at 8. Its photographs, 601 to 1,019 samples a
side, grow by 0.7 and 0.5 percent at 32 and by 1.2 and 1.0 percent at 16;
its strips one macroblock high or wide, which have no second direction to
predict from, by 1 to 4 percent at 32 and 4 to 7 percent at 16. Thirty-two
macroblocks is where the time stops falling: 16 is within five percent of
it on images of two megapixels and more, at about twice the cost in bytes.
A megapixel image is the exception, four tiles at 32 and sixteen at 16,
and codes in about half the time at 16.

With soft tiles the tile size changes only how coefficients are predicted
and coded: the transforms and overlap filters run across tile edges, so
every tile size decodes to the same samples. The tile count never raises
the declared level.

#### Bands

The encoder streams each component down its plane instead of holding the
whole image. On one thread it reads seven macroblock rows at a time and
quantizes and codes a row at a time; with threads it codes the tiles of a
band of tile rows in parallel while the samples of the next band are read,
filtered and transformed, the bands admitted against the 256 MiB bound a
decode's strips are, the whole image when it fits. Every band height and
thread count writes the same bytes. "Whole image" is the encoder before
bands, "banded" the one that recomputed a halo around bands of 8 MiB for
each thread, and "strips" this one, timed by the same benchmark source built
against each; each figure is the fastest of the runs, with the load average
between 2.1 and 6.6 on 18 cores.

One thread, milliseconds, the fastest of two runs of three warm calls:

| Image                                               | Lossless banded | strips | Quality 75 banded | strips |
| --------------------------------------------------- | --------------: | -----: | ----------------: | -----: |
| 4096 x 4096 BGRA, frequency, separate alpha, 1 tile |             381 |    377 |               216 |    213 |
| the same, default tiles                             |             380 |    376 |               217 |    214 |
| 4096 x 4096 BGRA, spatial, integrated alpha, 1 tile |             373 |    374 |               211 |    211 |
| the same, default tiles                             |             373 |    373 |               210 |    212 |
| 4000 x 3000 8-bit gray                              |            58.8 |   58.5 |              41.2 |   40.9 |
| 4000 x 3000 8-bit RGB                               |             204 |    203 |               112 |    110 |
| 4000 x 3000 8-bit RGBA                              |             264 |    264 |               155 |    152 |
| 4000 x 3000 16-bit RGB                              |             212 |    211 |               113 |    111 |
| 4000 x 3000 32-bit float RGB                        |             251 |    249 |               186 |    184 |
| 8000 x 6000 8-bit RGB                               |             817 |    820 |               443 |    443 |

Eighteen threads, milliseconds, the fastest of three runs of five warm calls.
An image of one tile is coded on one thread either way, beside its bands'
analysis; with default tiles a band ends with its slowest tile, which bands of
at least two tile rows keep to the whole-image encoder's time:

| Image                                               | Lossless whole image | banded | strips | Quality 75 whole image | banded | strips |
| --------------------------------------------------- | -------------------: | -----: | -----: | ---------------------: | -----: | -----: |
| 4096 x 4096 BGRA, frequency, separate alpha, 1 tile |                  297 |    290 |    293 |                   99.1 |   92.7 |   96.0 |
| the same, default tiles                             |                 40.4 |   42.7 |   39.2 |                   23.8 |   26.0 |   23.8 |
| 4096 x 4096 BGRA, spatial, integrated alpha, 1 tile |                  307 |    299 |    302 |                    112 |    104 |    109 |
| the same, default tiles                             |                 38.7 |   40.9 |   36.7 |                   22.7 |   23.2 |   21.1 |
| 4000 x 3000 8-bit gray                              |                  6.8 |    7.0 |    6.9 |                    4.9 |    4.7 |    4.9 |
| 4000 x 3000 8-bit RGB                               |                 21.5 |   22.4 |   20.2 |                   12.9 |   13.2 |   12.0 |
| 4000 x 3000 8-bit RGBA                              |                 28.3 |   29.7 |   27.3 |                   17.9 |   18.7 |   17.2 |
| 4000 x 3000 16-bit RGB                              |                 23.1 |   26.3 |   21.8 |                   12.7 |   13.8 |   12.0 |
| 4000 x 3000 32-bit float RGB                        |                 25.9 |   28.4 |   24.9 |                   18.9 |   20.0 |   18.2 |
| 8000 x 6000 8-bit RGB                               |                 77.8 |   86.5 |   79.7 |                   43.8 |   52.9 |   43.1 |

#### Memory

The measurements in this section predate the two-row prediction history
and the allocation-probe correction. The old probe counted both the old
and new sizes on every reallocation, which could overstate the heap peak
when a buffer grew or shrank in place. The heap probe used a
four-octave image generator, while the resident-size benchmark used six
octaves, so the heap and resident columns are not measurements of the same
input. Both tools now use the six-octave
`jxr_test_support::synthetic_samples` generator. Rerun both tools to measure
the current encoder on matching inputs.

Peak live heap of one `encode` call beside the file it returns, counted by an
allocator that tracks the high-water mark, with the source excluded
(`cargo test --release -p jxr-alloc-probe --test encode_memory_report --
--ignored --nocapture`), and the peak resident size of the whole benchmark
process (`/usr/bin/time -l jxr-encode-bench --warmups 0 --iterations 1
--setting ...`), which adds the source image, the program, and what the
allocator has not returned to the system. "Before" is the banded encoder,
"after" the strips. MB is 10^6 bytes:

| One thread                                                    | Heap before | after | Resident before | after |
| ------------------------------------------------------------- | ----------: | ----: | --------------: | ----: |
| 4096 x 4096 BGRA, frequency, separate alpha, 1 tile, lossless |        69.2 |  69.2 |             170 |   167 |
| the same, quality 75                                          |        10.4 |   9.6 |              89 |    86 |
| the same, default tiles, lossless                             |        69.3 |  64.1 |             169 |   130 |
| the same, default tiles, quality 75                           |        10.1 |   8.7 |              85 |    83 |
| 4096 x 4096 BGRA, spatial, integrated alpha, 1 tile, lossless |        72.5 |  49.5 |             163 |   160 |
| the same, quality 75                                          |        13.0 |  10.8 |              87 |    84 |
| the same, default tiles, lossless                             |        79.3 |  49.7 |             168 |   131 |
| the same, default tiles, quality 75                           |        13.5 |  11.0 |              90 |    84 |
| 4000 x 3000 8-bit gray, lossless                              |        17.8 |  12.0 |              37 |    28 |
| 4000 x 3000 8-bit gray, quality 75                            |         3.9 |   3.2 |              19 |    19 |
| 4000 x 3000 8-bit RGB, lossless                               |        43.8 |  29.4 |              93 |    72 |
| 4000 x 3000 8-bit RGB, quality 75                             |         9.3 |   8.1 |              49 |    51 |
| 4000 x 3000 8-bit RGBA, lossless                              |        50.2 |  46.9 |             121 |    92 |
| 4000 x 3000 8-bit RGBA, quality 75                            |         9.3 |   8.1 |              64 |    66 |
| 4000 x 3000 16-bit RGB, lossless                              |         124 |  80.3 |             219 |   158 |
| 4000 x 3000 16-bit RGB, quality 75                            |         9.3 |   8.1 |              85 |    87 |
| 4000 x 3000 32-bit float RGB, lossless                        |         124 |  80.5 |             284 |   226 |
| 4000 x 3000 32-bit float RGB, quality 75                      |        15.2 |  11.9 |             165 |   160 |
| 8000 x 6000 8-bit RGB, lossless                               |         155 |   101 |             339 |   338 |
| 8000 x 6000 8-bit RGB, quality 75                             |        21.2 |  17.7 |             179 |   169 |

On one thread an encode holds, for each plane it reads, the seven macroblock
rows read last and the rows its filters wait on, two macroblock rows at most;
one macroblock row of coefficients; the coding state and packets of the tile
rows it is in; and the file. The file grows once, early, to the length the
rows coded so far predict for it and an eighth more for the share of rows
still to code, an exact length once every row is coded, so that it is not
held twice over at the end, and the first packet of a spatial-mode tile row is
written onto its end. A frequency-mode tile keeps its four packets apart until
it is finished, so a lossless image of one tile holds its file twice at the
end. The tile writer keeps at most two macroblock rows of prediction history,
reusing storage sized for the widest tile its worker has coded. At 68 bytes
per macroblock per component, a single tile 4096 pixels wide needs at most
102 KiB for three components or 136 KiB for four. A tile one macroblock
row high needs half as much.

In WebAssembly, where the linear memory never shrinks and every move of a
growing buffer leaves its old storage behind, the file's single early growth
keeps the memory's growth to the heap's peak. Encoding the harness's
4096 x 4096 RGBA image with default tiles grows the linear memory by 30.6 MiB
losslessly (heap peak 31.1 MB), where growing the file by eighths grew it by
62.9 MiB (heap peak 28.5 MB), and by 10.9 MiB at quality 75 (12.1 MiB
before); of one tile, by 28.1 MiB losslessly (36.9 MiB before) and 10.7 MiB at
quality 75 (10.1 MiB before).

| Eighteen threads                                              | Heap before | after | Resident before | after |
| ------------------------------------------------------------- | ----------: | ----: | --------------: | ----: |
| 4096 x 4096 BGRA, frequency, separate alpha, 1 tile, lossless |         341 |   271 |             584 |   515 |
| the same, quality 75                                          |         285 |   214 |             506 |   434 |
| the same, default tiles, lossless                             |         352 |   212 |             616 |   391 |
| the same, default tiles, quality 75                           |         286 |   144 |             507 |   330 |
| 4096 x 4096 BGRA, spatial, integrated alpha, 1 tile, lossless |         378 |   269 |             433 |   345 |
| the same, quality 75                                          |         283 |   194 |             362 |   269 |
| the same, default tiles, lossless                             |         349 |   187 |             454 |   277 |
| the same, default tiles, quality 75                           |         283 |   145 |             362 |   221 |
| 4000 x 3000 8-bit gray, lossless                              |         125 |   120 |             150 |   138 |
| 4000 x 3000 8-bit gray, quality 75                            |         102 |   102 |             121 |   121 |
| 4000 x 3000 8-bit RGB, lossless                               |         348 |   131 |             412 |   180 |
| 4000 x 3000 8-bit RGB, quality 75                             |         303 |   106 |             347 |   149 |
| 4000 x 3000 8-bit RGBA, lossless                              |         348 |   154 |             540 |   301 |
| 4000 x 3000 8-bit RGBA, quality 75                            |         303 |   106 |             461 |   268 |
| 4000 x 3000 16-bit RGB, lossless                              |         414 |   194 |             570 |   297 |
| 4000 x 3000 16-bit RGB, quality 75                            |         303 |   106 |             382 |   186 |
| 4000 x 3000 32-bit float RGB, lossless                        |         414 |   194 |             636 |   363 |
| 4000 x 3000 32-bit float RGB, quality 75                      |         311 |   111 |             464 |   262 |
| 8000 x 6000 8-bit RGB, lossless                               |         477 |   299 |             652 |   464 |
| 8000 x 6000 8-bit RGB, quality 75                             |         310 |   214 |             469 |   367 |

With threads the samples of one band are analyzed while the band above it is
coded, and are then quantized into the coefficients it leaves, so an encode
holds one band of samples and one of coefficients, and the bands are admitted
against 256 MiB by what both take, whatever the image's height. A 4000 x 3000
colour image in the historical run held 106 to 154 MB, and 194 MB with the
52 MB lossless files of 16-bit and float RGB, against the whole-image encoder's
180 to 208 MB and the banded encoder's 303 to 414 MB. An image that fits in
one band is coded as one; the historical 4000 x 3000 8-bit gray run held
120 MB losslessly.

`cargo test -p jxr-alloc-probe` encodes on one thread images 1024 and 2048
pixels wide, at 384 and four times as many rows, in the Flash shapes, with
4:2:0 chroma and second-level overlap, and at the defaults for gray and
colour, and fails if an encode holds more than 4,096 bytes for each pixel of
the width beside its file and the room reserved for it, if the taller image
holds more than an eighth more beside its larger file, or if an encode at the
default tile size allocates as often as once for every four macroblocks. In a
pool of eight threads it encodes a 4096 x 4096 RGBA image of one tile, whose
samples and coefficients come to some 550 MB, and fails if the encode holds
more than 320 MiB beside its file.

## Decode memory

A full-resolution CPU decode reads a band of macroblock rows, reconstructs
it, stores the sample rows that finishes in the output, and lets them go. It
holds the coefficients of one band, a few sample rows of every component,
and two rows of prediction history for each tile being read. None of that
grows with the height of the image.

Peak resident size of a whole process that parses one file and decodes it to
8-bit RGBA on one thread (`/usr/bin/time -l`, `RAYON_NUM_THREADS=1`), and
the estimated peak heap of the `decode` call above the image it returns.
The heap column used the same pre-correction allocation probe described
under [Memory](#memory). The files are one tile, in
the two shapes Flash content has: a frequency-mode file with overlap 1 and
separate alpha, and a spatial-mode file with no overlap and integrated
alpha. "Before" held the coefficients of the whole image, a second copy of
them dequantized, and every component as 32-bit samples, all at once.
jxrlib is Microsoft's C codec decoding the same file to the same format:

| One tile, one thread             | Output | Resident before | now     | jxrlib  | Heap above output before | now     |
| -------------------------------- | -----: | --------------: | ------: | ------: | -----------------------: | ------: |
| 1024 x 1024 RGBA, frequency      |   4 MB |         42.6 MB |  9.8 MB |  6.8 MB |                  34.1 MB |  2.4 MB |
| 2048 x 2048 RGBA, frequency      |  16 MB |        169.3 MB | 26.2 MB | 19.2 MB |                 136.3 MB |  3.8 MB |
| 4096 x 4096 RGBA, frequency      |  64 MB |        646.7 MB | 89.0 MB | 68.0 MB |                 545.3 MB |  7.5 MB |
| 1024 x 1024 RGBA, spatial        |   4 MB |         54.2 MB |  9.4 MB |  6.8 MB |                  46.8 MB |  2.2 MB |
| 2048 x 2048 RGBA, spatial        |  16 MB |        210.3 MB | 25.4 MB | 19.4 MB |                 187.0 MB |  3.3 MB |
| 4096 x 4096 RGBA, spatial        |  64 MB |        830.5 MB | 88.5 MB | 68.5 MB |                 748.0 MB |  6.5 MB |
| 4000 x 3000 RGB, lossless        |  36 MB |        459.7 MB | 62.0 MB | 38.2 MB |                 402.9 MB |  4.8 MB |

The resident size counts the output, the file (0.4 to 18 MB here, read into
memory before it is parsed), and the process itself. Less the output and
the file it is 5 to 11 MB, where it was 38 to 753 MB. The heap above the
output is 1,200 to 2,400 bytes for each pixel of the image's width, the same
for an image of 192 rows as for one of 768.

In a thread pool a decode reads one band ahead of the one it reconstructs
and stores, in larger bands, and shares the tile rows of a tiled image out
among strips, as many as there are threads and as a bound of 256 MiB of
working storage admits (the request's `max_coefficient_bytes` if that is
lower). On eighteen threads the files above hold 7 to 30 MB above their
output; an RGBA image of 1024 by 1024 in tiles of four macroblocks, on
eight threads, holds 36 MB.

Decode time on the same machine, the best of several runs, before and now.
The one-tile files are those of the table above; the tiled ones are a
lossless 8-bit RGB image of 4000 by 3000 with tiles of at most 64, 32, 16
and 8 macroblocks a side:

| Decode, milliseconds             | One thread before | now | 18 threads before | now |
| -------------------------------- | ----------------: | --: | ----------------: | --: |
| 1024 x 1024 RGBA, frequency      |              14.8 |  13.7 |            11.1 |   8.1 |
| 2048 x 2048 RGBA, frequency      |              72.3 |  66.4 |            53.2 |  41.7 |
| 4096 x 4096 RGBA, frequency      |             307.3 | 280.4 |           217.0 | 166.1 |
| 1024 x 1024 RGBA, spatial        |              12.3 |  11.3 |             9.1 |   7.7 |
| 2048 x 2048 RGBA, spatial        |              62.2 |  57.1 |            46.0 |  41.5 |
| 4096 x 4096 RGBA, spatial        |             260.1 | 247.1 |           183.6 | 166.4 |
| 4000 x 3000 RGB, one tile        |             161.4 | 151.9 |           126.9 | 118.4 |
| 4000 x 3000 RGB, 64-macroblock tiles |                |       |            26.8 |  27.5 |
| 4000 x 3000 RGB, 32-macroblock tiles |                |       |            19.5 |  16.1 |
| 4000 x 3000 RGB, 16-macroblock tiles |                |       |            18.0 |  14.7 |
| 4000 x 3000 RGB, 8-macroblock tiles  |                |       |            18.6 |  14.6 |

One thread is faster because a band is reconstructed and stored while it is
still in the cache. A one-tile file on many threads is bounded by reading
its packets, which is serial within a tile; reconstruction and storing run
beside it.

What the limits of a request check is unchanged. `max_coefficient_bytes` is
checked against the coefficients of every tile the request selects, on
every route: an accelerator route, a prepared reconstruction and a
reduced-resolution decode hold all of them, and the CPU route is held to the
same figure although it needs far less, so that a request does not pass on
one route and fail on another. `max_host_allocation_bytes` is checked
against the output.

A decode that fails reports the error a decode that ran each stage over the
whole image in turn would have reported, and a file that decodes gives the
same samples as before on one thread and on many. A decode into caller-owned
storage that fails part way down the image leaves the rows above the
failure in that storage.

`cargo test -p jxr-alloc-probe` fails if a decode on one thread of a
one-tile 8-bit RGBA image 1,024 or 2,048 pixels wide holds more than 3,072
bytes for each pixel of width above its output, in either shape above or
with both overlap levels and 4:2:0 chroma; if it holds an eighth more at 768
rows than at 192; if it allocates as often as once for every two
macroblocks; or if a tiled image decoded on eight threads holds more than
320 MiB.

Reduced-resolution decode and the Metal and CUDA routes still hold the
coefficients of the whole image: a reduced decode reads a sixteenth of them
or less, and an accelerator takes them as one upload.


## Explicitly incomplete

- A wider device performance corpus. CPU, Metal, and CUDA pass all 517
  in-scope T.834/T.835 differential cases in the published release reports.
  The [Metal hardware workflow](https://github.com/frames-sg/jxr/actions/runs/37444774439)
  and NVIDIA workflow are both operational, but these conformance fixtures do
  not establish performance across a representative application corpus.
- Native reduced-resolution decode for spatially interleaved packets and for
  Metal or CUDA reconstruction. The implemented frequency-mode CPU route is genuinely
  band-limited and does not label full decode plus resampling as native reduction.
- CUDA validation on additional NVIDIA devices, drivers, and operating systems.
  The passing Linux runner covers one configuration. The benchmark uses two
  small conformance images and establishes no general speedup over CPU.
- ICC color transforms. Profiles are exposed byte-for-byte for a caller-selected
  color-management system and are never approximated inside the codec.

Unsupported combinations return errors; they are not approximated by a nearby
color format or silently retried after device submission.
