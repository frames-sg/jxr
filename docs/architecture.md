# Architecture

`jxr` is the public facade. `jxr-core` owns stable contracts and shared device
reconstruction planning, `jxr-math` owns portable exact arithmetic, and
`jxr-native` owns parsing, CPU decoding, and CPU encoding. `jxr-metal` and `jxr-cuda` depend
inward on those crates and own device-specific sessions, storage, and kernels.

CPU parsing and entropy state never live in an accelerator crate. Device crates
never implement a second parser. Explicit device requests are strict; automatic
routing may choose CPU only before a device submission begins.

Crate roots contain declarations and re-exports. Modules are split by codec
stage or resource lifetime, not by arbitrary helper categories.

## Decode flow

1. `jxr-native` discovers raw or Annex-A ranges, parses headers and the tile
   directory, applies request limits, and builds a `PreparedPlan`.
2. One reader, `tile_decode::dispatch::TileRows`, reads a tile a macroblock
   row at a time in either packet layout and for every colour format. A
   spatial-mode tile is one packet. A frequency-mode tile is a packet for
   each band, and no band's syntax reads another band's adaptive state, so
   the reader advances the DC, low-pass, high-pass and flexbits packets
   together, a macroblock row of each in turn. An integrated alpha plane is
   read after the primary plane at each position of each packet. Adaptive
   entropy, scans, and DC/LP prediction remain raster-ordered inside each
   tile; prediction reads the row being read and the row above it, so a
   reader keeps a history of those two rows
   (`MacroblockPosition::in_two_rows`, `end_history_row`) and nothing of the
   rows before. HP direction is selected there, while the coefficient
   operation is deferred to reconstruction.
3. Coefficients are written once into packed macroblock-major storage with
   structure-of-arrays metadata and one contiguous range per component.
   Supported integrated alpha is appended as a distinct plane after the
   primary components while sharing the packet walk. The storage holds
   whatever rows its filler reads into it. `decode_tiles` runs every tile to
   its end, in parallel, and fills an arena that holds the image:
   `PreparedReconstruction` retains that handoff for an accelerator without
   reparsing or re-running entropy work, and a reduced-resolution CPU decode
   reconstructs from it. `BandDecoder` takes the tiles of each tile row down
   the image together and fills the same layout with a band of a few
   macroblock rows, for a decode that reconstructs each band before it reads
   the next.
4. A full-resolution CPU decode is a stream. `reconstruct::ComponentStrip`
   takes the macroblock rows of one component in order: it applies HP
   prediction through `jxr-math`, dequantizes, runs the first-level
   transform and the overlap filter of that level over the block DC
   coefficients, then the second-level transform and the overlap filter of
   the samples. The filters reach across macroblock rows, so a row's
   samples are finished a little after it arrives: two sample rows later
   under overlap one, and a macroblock row and two sample rows later under
   overlap two, whose low-pass-level filter reaches half a macroblock (one
   row of block DC coefficients for subsampled chroma). At the bottom of
   the image and of a hard tile nothing is held back. The plane traversals
   are written for a strip of rows: `transform_rows`, `apply_overlap_rows`
   and `apply_subsampled_first_overlap_rows` run the blocks and the windows
   that lie wholly within the rows they are given, and a whole plane is the
   strip that covers it: `reconstruct_luma` is the `ComponentStrip` given
   every row and never released, a reduced decode filters its plane of
   block DC coefficients as one strip, and the encoder's whole-plane
   analysis is the `AnalysisStrip` given every row. `StripPlan` holds what a
   plane's strip filters over, its filter regions and both levels' entries,
   for the decoder's strips and the encoder's alike. Subsampled chroma is interpolated to full
   resolution as its rows are finished (`ChromaStrip`), a 4:2:0 row once the
   row below it is. Finished rows are colour-converted and packed straight
   into the caller's output (`decode::output::OutputRows`) and then let go,
   so a decode holds the coefficients of one band and a few sample rows of
   each component, whatever the height of the image. A separate alpha image
   is a second stream kept level with the first. A session-retained
   capability token selects safe AVX2/NEON HP dequantization and common U8
   stores, and on WebAssembly built with `simd128` the U8 stores alone
   (`cpu::I32Vectors`): there the scalar dequantization, compiled for
   `simd128`, is as fast as the vector form. Scalar math remains the oracle
   and fallback. The transform and overlap lane kernels are written once
   over a `Lane`: four blocks to a vector where the target has vectors, and
   on a target without them, such as WebAssembly built without `simd128`,
   one block at a time on one-sample lanes, which keep its samples in
   registers where emulated vectors would pass them through memory.

   Every stage that shares its work out, in the decoder and the encoder,
   hands the tasks to the executor (`executor::Executor`) of the
   `CpuCapabilities` the session runs with, which also hold its vector
   level: the rayon pool, or the calling thread, which runs the same tasks
   to the same result. A stage decides only whether its work is large
   enough to share out (`Executor::if_worth`). WebAssembly without the
   `atomics` feature cannot start a thread, so it has no pool and none of
   the pool's code is compiled into it. `CpuCapabilities::on_calling_thread`
   selects the calling thread on any target, for a decode (`decode_cpu`)
   and an encode (`encode_annex_a` and the other encode entries) alike;
   `jxr::encode` and `JxrDecoder` run with `CpuCapabilities::detect`. The
   coefficient entries that take no capabilities, `decode_coefficients`
   and `decode_coefficients_into` and the accelerator handoff
   (`prepare_accelerator_coefficients`) built on them, read their tiles
   through the default executor, the one `detect` selects.
   `crates/jxr-native/tests/executors.rs` decodes and encodes through both
   and compares, and a unit test in `cpu.rs` counts that no stage of either
   on the calling thread hands a task to the pool. `tools/wasm_size_check.py`, run by
   CI, holds the decode-only WebAssembly module (`crates/jxr-wasm-size`) to
   a size budget.

   On one thread the image is one strip and nothing enters a thread pool.
   In a pool, a stream reads its next band while the last one is
   reconstructed and stored, reads the tiles of a band on several threads,
   and reconstructs the components of a band on several. The
   tile rows of the image are shared out among strips (`decode::strips`),
   each the same stream over its own rows storing into its own part of the
   output. A strip takes in one macroblock row beyond the rows it stores
   wherever a neighbour's rows reach into them, which is all the rows it
   stores can depend on: the windows that reach outside the strip are left
   out, and what they would have changed lies outside the rows it stores.
   Tile rows that share nothing, hard tiles or no overlap without vertical
   chroma interpolation, are strips with no such row. Otherwise each strip
   leaves the row next to a boundary unstored and keeps the coefficients of
   its two rows nearest it, and a joining strip of those four rows stores
   the two in the middle. The number of strips is admitted against a byte
   bound, 256 MiB or the request's coefficient limit if that is lower, by
   what one strip holds: two bands of coefficients, the four rows it keeps,
   and the sample rows of a band. The rule is `executor::admitted`, which
   the encoder's bands are admitted by too. Every strip stores exactly what the
   plane-wide sweep stores.

   A decode that fails reports what a decode that finished each stage over
   the whole image before starting the next would have: when a stage fails
   in one band, the stages that come before it are carried to the end of
   the image without it, in the order DC, low-pass, high-pass and flexbits
   packets, reconstruction, the same of a separate alpha image, and output.
   Rows above the failure have by then been stored in a caller-owned
   destination.

   Frequency-mode native reduced requests omit higher packet bands before
   entropy decoding and allocation. Reconstruction then stops at the DC of
   each block or macroblock and scales it to the sample it stands for: by
   the gain of the transform stages not run and, under overlap filtering,
   by the level and the rounding offset that the overlap filter of the
   samples, which is not run, would have restored. Both are measured on
   small flat images taken through the forward transform and through the
   reduced decode itself, at every position a sample can have among the
   edges of the image and of hard tiles, so a flat area keeps its level
   wherever it lies. Each component of a retained native decode workspace
   keeps one `SampleScale` for overlap filtering. It reuses that scale when
   the `ReconstructionConfig` and `DecodeScale` match, even if the
   coefficients change. A different overlapped configuration or resolution
   replaces the entry. Ordinary `JxrDecoder` calls start with a fresh
   workspace. With second-level overlap, both reduced resolutions run the
   low-pass overlap filter at quarter size. A quarter-size decode uses the
   DC and low-pass bands. A sixteenth-size decode uses the DC band alone
   and then averages each macroblock.
   A reduced decode reconstructs the whole window at once from an arena
   that holds it, which is a sixteenth or less of the full coefficients.
5. `jxr-core::device_plan` owns the shared plane geometry, overlap schedules, and
   output plans. Metal and CUDA adapt those plans to their device ABI and retain
   the same coefficient contract. Their sessions own device
   state, allocation pools, submissions, completion, resident
   output, and host readback. Metal preparation can write entropy results directly
   into shared allocation slices; compatible batches bind those slices without
   repacking and use macroblock × image × component transform grids. CUDA keeps
   one immutable upload per coefficient identity and stream, submits the
   equivalent five-phase pipeline, and retains bounded device scratch and
   resident outputs without requiring CUDA at CPU-only build time.

## Route model

Metal promotes at 2^21 (2,097,152) reconstructed coefficients, measured with
interleaved single-image CPU and Metal host decodes on M4, M4 Pro, and M5 Max:
smaller images mostly ran at 0.29-0.98x on Metal because a fixed submission
latency outweighs the CPU reconstruction it replaces, while larger ones ran at
0.92-1.35x. CUDA keeps the provisional 16,384-coefficient promotion point.
CPU remains the route below the threshold or when a compatible device
is unavailable. Explicit device requests are strict. Automatic fallback is
legal only before submission; an error after a command is submitted is returned
to the caller. Native reduced requests currently stay on CPU; explicit Metal or
CUDA requests for those scales fail before accelerator preparation.

The thresholds are routing heuristics, not benchmark claims. They stay
isolated in the backend route modules so later end-to-end measurements can
replace them without changing entropy or reconstruction code.

## Attach a GPU session

Enable the `metal` or `cuda` feature on `jxr`, then create a session and reuse it
across images. For a compatible 8-bit RGB source on Apple-silicon macOS:

```rust
use jxr::{
    AlphaHandling, BackendRequest, ChannelLayout, DecodeRequest, JxrView,
    PixelFormat, metal::MetalDecoderSession,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read("image.jxr")?;
    let view = JxrView::parse(&bytes)?;
    let request = DecodeRequest::new(PixelFormat::U8(ChannelLayout::Rgb))
        .with_alpha(AlphaHandling::Drop)
        .with_backend(BackendRequest::Metal);
    let session = MetalDecoderSession::system_default()?;
    let mut decoder = view.decoder().with_metal_session(&session);
    let image = decoder.decode(&request)?;
    println!("Decoded {} bytes", image.samples.byte_len());
    Ok(())
}
```

For CUDA, replace the session with `jxr::cuda::CudaDecoderSession::new(0)?`
(device 0), attach it with `with_cuda_session(&session)`, and request
`BackendRequest::Cuda`. The CUDA driver loads embedded CUDA Oxide PTX at runtime.
Enabling a feature alone does not attach a session. Keep the session alive for
as long as the decoder borrows it.

## Owned batch flow

`PreparedJxr` owns parsed compressed metadata behind a shared allocation.
`PreparedImage` adds one validated request plan and native output contract, while
`PreparedBatch` groups compatible images in stable first-occurrence order. This
layer is device-neutral: input-local preparation errors remain indexed rather
than failing valid siblings.

`CpuBatchDecoder` retains an image-level worker pool, independent primary and
separate-alpha coefficient arenas, component-raster and inverse-transform
scratch, and recyclable signed component planes per worker. A bounded LRU-style
preparation cache is keyed by compressed `Arc` identity plus the complete
request; duplicate prepared images share an optional coefficient-ready
`PreparedReconstruction`. Native-layout typed output stores write directly into
the final dense group allocation, including independently encoded alpha. NCHW is
a CPU-only tensor policy implemented by a type-matched transpose in retained
worker scratch after the same direct store. Exact typed caller destinations
retain one fixed slot per prepared image so a sibling error does not move
another image's slot.

`MetalBatchDecoder` prepares coefficient slices concurrently from the same
groups. High-level submission may return immediately, complete to per-image
resident allocations, or retain one private `MetalResidentBatch` allocation per
homogeneous group. Caller-owned dense destinations require a one-queue session;
the pending owner retains exclusive destination access through completion.
`jxr-mpsgraph` accepts the shared native-layout prepared contract, then owns the
additional coefficient and dense-allocation lifetime required by its exact-queue
NHWC graph handoff. The shared `j2k-mpsgraph-support` submission guard retains
those input owners and graph resources through completion; JXR validation and
result assembly remain in the adapter. See the
[adapter's shared submission owner](../crates/jxr-mpsgraph/README.md#shared-submission-owner).

`CudaBatchDecoder` consumes the same groups and coefficient-ready cache. It
schedules independent images over four streams and can retain per-image outputs,
one homogeneous dense device allocation, or an exclusive caller-owned
destination without host pixel staging.

`jxr-image` is an outward ownership adapter. It depends on `jxr`, transfers only
single-plane formats that `image::DynamicImage` represents exactly, and retains
JPEG XR ICC, alpha, and route semantics in a wrapper. Neither `jxr-core` nor the
decoder crates depend on `image`.

## Encode flow

Encoding is the decode flow in reverse, and each stage is written as the exact
inverse of the decoding stage it mirrors rather than as a second
implementation of the format.

The image passes through the first three stages as a stream down each
component's plane (`encode::bands`), the decode stream run backwards: the
source is read a few macroblock rows at a time straight into the rows each
component's `AnalysisStrip` holds, each stage runs as far as the rows taken
in allow, and a macroblock row is quantized and entropy coded once the
filters that reach across macroblock rows have finished it. A strip then
lets the row go and keeps only the rows its filters wait on, so every
sample is read, converted, filtered and transformed once, no stage holds a
whole-image plane of samples or of coefficients, and an encode's working
memory follows the image's width and not its height. On one thread the
source is read seven macroblock rows at a time and each macroblock row is
quantized and coded on its own. With threads the rows go in bands: the
tiles a band reaches are coded in parallel while the samples of the next
band are read, filtered and transformed, and those are then quantized into
the coefficients the coded band leaves. Bands are admitted by the rule a
decode's strips are (`executor::admitted`), against the same 256 MiB, at
what a band's samples and coefficients take: an image that fits is one
band, which is then the whole-image encoder exactly, and a larger one is
cut at tile rows, at least two to a band. Bands change nothing in the
output: every band height and thread count codes the same coefficients,
and the same packets.

1. `encode::input` is the output stage run backwards and defines no format
   of its own. The container pixel-format table gives a source format's
   declaration through its inverse lookup; `ChannelScale::unapply` inverts
   each depth's bias, scaling and post-scaling; and the half-float, 32-bit
   float, RGBE and colour conversions, and the unpacking of RGB555, RGB565
   and RGB101010 words and of one-bit samples
   (`output_format::packed_color`), are inverses that live beside the
   decoder's functions. It then pads each component plane to whole macroblocks
   by repeating its edge. `SourceReader` reads any window of rows of the
   padded planes, a row below the image being its last row read again.
   Interleaved 8- and 16-bit luma and RGB pixels, the layouts the decoder's
   interleaved store writes, are read through that store run backwards
   (`output_format::interleave::InterleavedLoad`), with its layout, its
   scale and the wrapping form of the colour transform beside the checked
   one; a pixel of any other storage is converted one sample at a time
   through the checked inverses.
   Planar YUV 4:2:0 and 4:2:2 sources are read plane by plane, as the
   decoder returns them. When the request codes full-resolution chroma at a
   coarser sampling, the full-resolution chroma rows are halved as they are
   read by `reconstruct::ChromaDecimation`, the direction `ChromaStrip`
   reverses, which sits beside the decoder's interpolation and uses the
   `jxr-math` decimation filter whose grid the plane header's chroma
   centering declares. Each row is halved across as it arrives, and a 4:2:0
   row is filtered once the two rows below the pair it keeps have arrived,
   mirrored only at the plane's first and last rows, so every chroma row is
   filtered once.
   `encode::quality` then settles what the request leaves to the encoder.
   A request names quantization parameters or a quality. A quality is
   resolved here and nowhere else, against the format `encode::input`
   resolved: each coded component takes the parameters of its part in the
   colour format, and each parameter is exchanged for the one whose step is
   as large against the range of the coded samples as the calibrated step
   is against 8-bit samples, looked up in the decoder's own table of steps
   (`tile_decode::quantizer::component_steps`). The overlap mode, when the
   request leaves it open, is chosen with them.
2. `analyze` runs reconstruction's own plane code with the forward operators.
   A `BlockKernel` pairs each 4-by-4 operator's vector form with its checked
   `jxr-math` oracle; `transform_rows`, `apply_overlap_rows` and
   `apply_subsampled_first_overlap_rows` are generic over it, so the forward
   and inverse transforms and overlap filters share one traversal, window
   schedule, SIMD dispatch and thread split. A speedup to that plumbing
   applies to decoding and encoding alike. Each traversal is written for a
   strip of rows of a plane, and analysis streams through them as the
   decoder does, over the regions its `StripPlan` gives; the output type is
   the `QuantizedMacroblock` reconstruction consumes.
   Each component is analyzed in the macroblock shape the tile decoder gives
   it: four blocks square, or the two-by-four and two-by-two of subsampled
   chroma, with reconstruction's own chroma transforms and overlap operators.
   Quantization is the inverse of dequantization and sits beside its vector
   form: on AVX2, NEON and `simd128` each quotient is estimated in `f32` and
   corrected to the checked division's quotient, and the checked division
   remains for bands beyond the range `f32` holds exactly and for targets
   without vectors.

   `AnalysisStrip` takes sample rows in at the bottom and runs the
   prefilter, the forward transform of each macroblock row the prefilter
   has finished, and under second-level overlap the low-pass-level
   prefilter over the block DC coefficients of the rows transformed, each
   as far as the rows taken in allow, as `ComponentStrip` runs the inverse
   stages. A macroblock row is ready to quantize two sample rows into the
   next under first-level overlap, and a macroblock row and two sample rows
   later under second-level overlap; at the bottom of the plane and of a
   hard tile every window ends and nothing waits. The rows a filter waits
   on stay in the strip until the rows below them arrive, so no row is
   filtered twice and a band has no halo. `analyze_component`, the whole
   plane taken in at once, is the one-band case.
3. `tile_encode` codes each tile independently and in parallel with one
   packet writer for every internal colour format. A `TileWriter` begins a
   tile, codes its macroblock rows as each band brings them, and finishes it
   by appending its packets to the output, in the tile index's order; it
   keeps between bands only the state the syntax carries from row to row:
   each plane's adaptive entropy and coded-block-pattern state, the DC and
   low-pass values its prediction reads, and the packets so far. The
   prediction history holds only the row being coded and the row above it,
   using the decoder's two-row indexing and reusing the storage for later
   tiles. A band is coded across every tile it reaches at once, the tile
   columns of one tile row or every tile of a band that spans several. A
   spatial-mode tile that begins when every earlier tile is finished is
   lent the output and writes its packet onto the output's end, so the first
   packet of a tile row, the whole of a
   one-tile spatial image, is never copied. The output grows to the length
   the rows coded so far predict for the codestream, and an eighth more for
   the share of rows still to code, so that it moves about once, early, and
   is not held twice over at the end; in WebAssembly, whose memory never
   shrinks, each move of a growing output leaves its old storage behind. Y-only, YUV at each chroma
   sampling, YUVK, and N-component planes share its macroblock loop and
   differ only in how component presence is signalled, which components
   select the prediction modes, and how many blocks a chroma component has.
   The tables that are particular to subsampled chroma, the order of its
   jointly coded low-pass block, its low-pass prediction steps, its block
   order and its coded-block-pattern prediction, are the decoder's, each with
   its inverse beside it. It forms DC, low-pass, and high-pass residuals with the
   prediction-mode functions the tile decoder uses, and drives the decoder's
   adaptive entropy state: VLC table selection, scan order, coefficient model
   bits, and coded-block pattern prediction. State and tables live once, beside the decoder; only
   the bit writing is separate. High-pass prediction is closed around the
   values a decoder will reconstruct, so bits dropped by flexbits trimming or
   an absent flexbits band do not accumulate from block to block.

   The same loop writes both packet layouts. It forms every element of a
   macroblock once and hands it to `TilePackets` under the band it belongs
   to. In spatial mode every band is the same bit writer, so a tile is one
   packet with the bands of each macroblock interleaved. In frequency mode
   each band the image codes has a writer of its own, so a tile is a DC, a
   low-pass, a high-pass and a flexbits packet, each with its own start code
   and the flexbits packet with the flexbits trim. No band's syntax reads
   another band's adaptive state, and a frequency-mode reader restarts
   adaptation in every band packet by the rules the spatial reader follows
   for all bands at once (`adapts_after`, `resets_scan_totals`), so one set
   of state per plane serves both layouts. The order of a tile's packets is
   `PacketBand::coded`, which the tile index parser and the decoder's packet
   lookup use as well.
4. `header::write` and the tile directory writer emit the image header, plane
   headers, tile index, and one profile-level record around the packets. The
   index has one entry for each packet: one a tile, or in frequency mode one
   for each band of each tile, and a frequency-mode image has an index even
   when it is a single tile. The headers are written before the first packet
   and the index after the last, into room left for the longest index, which
   is then closed up, so the packets are appended once, to the buffer that
   is the output.

An alpha channel is one more plane through the same stages. `encode::input`
reads it with the scale the decoder packs alpha with, `analyze` transforms it
as a Y-only plane with its own quantizers and scaled flag, and the packet
writer codes it with the Y-only syntax. Separate alpha puts that plane in a
codestream of its own, a gray image the Annex-A writer places beside the
primary one. Integrated alpha keeps it in the primary codestream: the packet
writer holds one set of adaptive state per plane and writes the alpha
macroblock after the primary one at every position, in the one packet of a
spatial-mode tile or in each band packet of a frequency-mode one, the order
the tile decoder's integrated-alpha readers consume them in, and the alpha
plane gets a second plane header.

`jxr::encode_codestreams` returns the raw codestreams, one or two, with the
pixel-format identifier that declares the source; `jxr::encode` wraps them in
an Annex-A container, and `jxr::encode_codestream` returns the codestream of
an image that needs only one. `jxr::encode` writes each codestream into the
file where the container places it, after room for the directory, and the
Annex-A writer then completes the file around them, so no codestream is
held beside the file it ends up in.

## Container write flow

The public `jxr::write_annex_a` facade delegates to the Annex-A module beside
the native reader. The writer accepts existing raw T.832 primary and optional
separate-alpha codestreams, fully parses them, verifies their dimensions, emits
sorted typed directory entries and aligned payloads, then parses the completed
file again. The container writer never accepts pixels or synthesizes entropy
syntax; the encoder above produces the codestreams it wraps. The encoder uses
the same writer to complete a file whose codestreams it wrote in place, with
the same checks.

## Canonical math and ABI

`jxr-math` is `no_std` and owns checked integer transforms, overlap operators,
dequantization, HP prediction, interpolation, color, alpha, and RGBE arithmetic.
Metal uses backend-local MSL. CUDA uses Rust kernels compiled by CUDA Oxide,
calling the shared Rust transforms and overlap filters directly. The HP phase
consumes compact CPU-selected directions and resolves prediction within each
macroblock before dequantization. Transform constants and
parameter indices are generated from `data/reconstruction.abi`, which also
generates the Rust bindings used by both device encoders. All accelerator
manifests describe the complete packaged Main-profile pipeline.
