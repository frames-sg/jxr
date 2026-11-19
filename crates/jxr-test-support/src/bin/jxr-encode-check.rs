// SPDX-License-Identifier: MIT OR Apache-2.0

//! Differential check of the encoder against the T.835 reference decoder.
//!
//! Every source image is encoded under a matrix of settings. Each result must
//! decode identically in the Rust decoder and in T.835, which establishes that
//! the codestream is conformant rather than merely self-consistent, and every
//! lossless result must reproduce its source exactly.
//!
//! `--quality-scale` runs another check in place of that one, without T.835:
//! every source is coded at each quality from 100 down to 0, in each of its
//! coded forms, and the steps on which a file grew are reported. Those are
//! counted, not failed: an image that is exact in a few coefficients of
//! unscaled arithmetic is cheapest coded losslessly.

use std::path::{Path, PathBuf};

use jxr::{
    AlphaMode, BandPresence, BitstreamMode, ChannelLayout, ChromaSampling, DecodeRequest,
    DecodeScale, DecodedSamples, EncodeQuantizers, EncodeRequest, JxrError, JxrErrorKind, JxrView,
    OverlapMode, PixelFormat, SampleFormat, SourceImage, SourcePlane, SourceSamples,
};
use rayon::prelude::*;

use jxr_test_support::{
    T835Oracle, T835ProfileLimit, TempFile, decimates_chroma, decode_rust_file, exit_with_failures,
    oracle_format, reference_alpha_bytes, reference_bytes, sample_bytes, subsamples_chroma,
    workspace_root,
};

fn main() {
    exit_with_failures("jxr-encode-check", run());
}

/// One source image in the storage its pixel format uses.
struct Source {
    name: String,
    width: u32,
    height: u32,
    format: PixelFormat,
    samples: DecodedSamples,
    /// Storage elements between the starts of consecutive rows of an
    /// interleaved image.
    row_stride: usize,
    /// Where each plane of a planar image lies; empty for an interleaved one.
    planes: Vec<SourcePlane>,
    /// Significant bits of samples narrower than their storage.
    sample_bits: Option<u8>,
    /// Chroma-centering codes of a planar image, which the encoded file
    /// must declare as the source did.
    chroma_centering: [u8; 2],
    /// Whether the colour samples are premultiplied by alpha, which the
    /// encoded file must declare as the source did.
    premultiplied: bool,
}

fn run() -> Result<u64, Box<dyn std::error::Error>> {
    let mut arguments: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if arguments.iter().any(|argument| argument == "--help") {
        println!(
            "usage: jxr-encode-check [--keep DIR] [--quality-scale] [image.jxr ...]\n\
             With no images, a built-in set of synthetic images is used.\n\
             Images whose pixel format the encoder does not support are skipped.\n\
             --keep DIR saves each source's encoded files in a new subdirectory of DIR.\n\
             --quality-scale codes every source at each quality from 100 to 0\n\
             instead, and reports each step down the scale on which a file grew."
        );
        return Ok(0);
    }
    let quality_scale = match arguments
        .iter()
        .position(|argument| argument == "--quality-scale")
    {
        Some(index) => {
            arguments.remove(index);
            true
        }
        None => false,
    };
    let keep = match arguments.iter().position(|argument| argument == "--keep") {
        Some(index) if index + 1 < arguments.len() => {
            let directory = arguments.remove(index + 1);
            arguments.remove(index);
            std::fs::create_dir_all(&directory)?;
            Some(directory)
        }
        Some(_) => return Err("--keep needs a directory".into()),
        None => None,
    };
    let workspace = workspace_root()?;
    // The encoder declares the lowest profile its output meets; the ceiling
    // only keeps the reference from rejecting a correct declaration.
    let oracle =
        T835Oracle::for_workspace(workspace).with_profile_limit(T835ProfileLimit::Advanced);

    if quality_scale {
        report_quality_scale(&arguments)?;
        return Ok(0);
    }
    let requests = request_matrix();
    let (mut passed, mut failed) = (0_u64, 0_u64);
    let (source_count, skipped) = visit_sources(&arguments, |source| {
        let output = keep
            .as_ref()
            .map(|dir| kept_directory(dir, &source.name))
            .transpose()?;
        let source_bytes = sample_bytes(&source.samples);
        for (label, request) in source_requests(&requests, source.format) {
            let kept = output.as_ref().map(|directory| {
                let setting = label.replace(|character: char| !character.is_alphanumeric(), "_");
                directory.join(format!("{setting}.jxr"))
            });
            match check(&oracle, source, &source_bytes, &request, kept.as_deref()) {
                Ok(Outcome::Decoded(bytes)) => {
                    passed += 1;
                    println!(
                        "pass\t{}\t{}x{} {:?}\t{label}\t{bytes} bytes",
                        source.name, source.width, source.height, source.format
                    );
                }
                Ok(Outcome::Refused) => {
                    passed += 1;
                    println!(
                        "pass\t{}\t{}x{} {:?}\t{label}\trefused, as T.835 would",
                        source.name, source.width, source.height, source.format
                    );
                }
                Err(error) => {
                    failed += 1;
                    println!(
                        "FAIL\t{}\t{}x{} {:?}\t{label}\t{error}",
                        source.name, source.width, source.height, source.format
                    );
                }
            }
        }
        Ok(())
    })?;
    println!(
        "summary: {passed} passed, {failed} failed, {skipped} sources skipped as unsupported \
         ({} sources x {} settings, {} of them frequency mode, each CMYK source also coded as \
         CMYKDIRECT, each RGB and YUV 4:4:4 source also with 4:2:2 and 4:2:0 chroma, each \
         source with alpha also with integrated alpha and with alpha quantized apart from \
         colour)",
        source_count,
        requests.len(),
        frequency_requests().len()
    );
    Ok(failed)
}

/// Finish each file's checks while its decoded samples are in memory.
fn visit_sources(
    paths: &[PathBuf],
    mut visit: impl FnMut(&Source) -> Result<(), Box<dyn std::error::Error>>,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    if paths.is_empty() {
        let sources = synthetic_sources();
        for source in &sources {
            visit(source)?;
        }
        return Ok((sources.len(), 0));
    }
    let mut skipped = 0;
    for path in paths {
        if let Some(source) = load_source(path)? {
            visit(&source)?;
        } else {
            skipped += 1;
            println!("skip\t{}", path.display());
        }
    }
    Ok((paths.len() - skipped, skipped))
}

/// Reserve a fresh directory even when another input has the same file name.
fn kept_directory(parent: &Path, name: &str) -> std::io::Result<PathBuf> {
    let mut path = parent.join(name);
    let mut copy = 1;
    loop {
        match std::fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                copy += 1;
                path = parent.join(format!("{name}-{copy}"));
            }
            Err(error) => return Err(error),
        }
    }
}

/// Code every source at each quality of the scale, in each of its coded
/// forms, and report the steps down the scale on which a file grew.
fn report_quality_scale(paths: &[PathBuf]) -> Result<(), Box<dyn std::error::Error>> {
    let scale: Vec<(String, EncodeRequest)> = (0..=100_u8)
        .rev()
        .map(|quality| {
            (
                format!("quality={quality}"),
                EncodeRequest::quality(quality),
            )
        })
        .collect();
    // Coded forms differ in one option; alpha is coded apart either way.
    let forms = |format| {
        source_requests(&scale[..1], format)
            .into_iter()
            .filter(|(_, request)| request.alpha_mode == AlphaMode::Separate)
            .filter(|(_, request)| request.alpha_quantizers.is_none())
    };
    let (mut grown, mut steps, mut coded) = (0_u64, 0_u64, 0_u64);
    let (source_count, _) = visit_sources(paths, |source| {
        let image = source.image()?;
        for (label, form) in forms(source.format) {
            let form_label = label.trim_start_matches("quality=100").trim();
            let sizes = scale
                .par_iter()
                .map(|(_, request)| {
                    let request = EncodeRequest {
                        quantization: request.quantization,
                        overlap: request.overlap,
                        ..form.clone()
                    };
                    jxr::encode(&image, &request).map(|file| file.len())
                })
                .collect::<Result<Vec<_>, _>>()?;
            coded += 1;
            let growth: Vec<usize> = (1..sizes.len())
                .filter(|&index| sizes[index] > sizes[index - 1])
                .collect();
            grown += u64::from(!growth.is_empty());
            for index in growth {
                steps += 1;
                println!(
                    "grow\t{}\t{}x{} {:?}\t{form_label}\tquality {}: {} bytes after {}",
                    source.name,
                    source.width,
                    source.height,
                    source.format,
                    100 - index,
                    sizes[index],
                    sizes[index - 1]
                );
            }
        }
        Ok(())
    })?;
    println!(
        "quality scale: {steps} steps to a larger file, in {grown} of {coded} coded forms of \
         {source_count} sources"
    );
    Ok(())
}

/// Every coded form of a source of `format` under the settings `requests`.
///
/// CMYK has a second coded form, so its sources run every setting twice,
/// and a source whose chroma the encoder can subsample runs every setting
/// at each of the three chroma samplings. A source with alpha runs each of
/// those with separate alpha, which is what the settings select as they
/// stand, and again with integrated alpha, and then with the alpha plane
/// quantized apart from the colour in each mode.
fn source_requests(
    requests: &[(String, EncodeRequest)],
    format: PixelFormat,
) -> Vec<(String, EncodeRequest)> {
    let mut all = requests.to_vec();
    if is_cmyk(format) {
        all.extend(requests.iter().map(|(label, request)| {
            (
                format!("{label} cmykdirect"),
                request.clone().with_cmyk_direct(true),
            )
        }));
    }
    if subsamples_chroma(format) {
        for (name, sampling) in [
            ("4:2:2", ChromaSampling::Cs422),
            ("4:2:0", ChromaSampling::Cs420),
        ] {
            all.extend(requests.iter().map(|(label, request)| {
                (
                    format!("{label} chroma={name}"),
                    request.clone().with_chroma_sampling(sampling),
                )
            }));
        }
    }
    if !format.has_alpha() {
        return all;
    }
    all.extend(alpha_quantizer_requests());
    // The encoder refuses trimmed flexbits with integrated alpha: the
    // reference decoder reads such an alpha plane as untrimmed and fails,
    // on its own encoder's output as well.
    let integrated: Vec<_> = all
        .iter()
        .filter(|(_, request)| request.trim_flexbits == 0)
        .map(|(label, request)| {
            (
                format!("{label} alpha=integrated"),
                request.clone().with_alpha_mode(AlphaMode::Integrated),
            )
        })
        .collect();
    all.extend(integrated);
    all
}

/// Settings whose alpha plane is quantized, and so scaled, differently from
/// the colour planes.
fn alpha_quantizer_requests() -> Vec<(String, EncodeRequest)> {
    let lossy = EncodeQuantizers::uniform(30);
    vec![
        (
            "qp=30 alpha lossless".to_owned(),
            EncodeRequest::lossy(lossy).with_alpha_quantizers(EncodeQuantizers::LOSSLESS),
        ),
        (
            "lossless alpha qp=30".to_owned(),
            EncodeRequest::lossless().with_alpha_quantizers(lossy),
        ),
        (
            "qp=20 alpha qp=60 overlap=2".to_owned(),
            EncodeRequest::lossy(EncodeQuantizers::uniform(20))
                .with_alpha_quantizers(EncodeQuantizers::uniform(60))
                .with_overlap(OverlapMode::Two),
        ),
        (
            "qp=40 alpha lossless tiles=2mb bands=NoFlexbits".to_owned(),
            EncodeRequest::lossy(EncodeQuantizers::uniform(40))
                .with_alpha_quantizers(EncodeQuantizers::LOSSLESS)
                .with_max_tile_macroblocks(2)
                .with_bands(BandPresence::NoFlexbits),
        ),
        (
            "qp=30 alpha lossless tiles=2mb frequency".to_owned(),
            EncodeRequest::lossy(lossy)
                .with_alpha_quantizers(EncodeQuantizers::LOSSLESS)
                .with_max_tile_macroblocks(2)
                .with_bitstream_mode(BitstreamMode::Frequency),
        ),
        (
            "lossless alpha qp=30 frequency".to_owned(),
            EncodeRequest::lossless()
                .with_alpha_quantizers(lossy)
                .with_bitstream_mode(BitstreamMode::Frequency),
        ),
        // A quality resolves the colour planes and leaves the alpha plane
        // its own quantizers.
        (
            "quality=60 alpha lossless".to_owned(),
            EncodeRequest::quality(60).with_alpha_quantizers(EncodeQuantizers::LOSSLESS),
        ),
    ]
}

/// Whether `format` is one the container can declare as CMYK or CMYKDIRECT.
const fn is_cmyk(format: PixelFormat) -> bool {
    matches!(
        format,
        PixelFormat::U8(ChannelLayout::Cmyk | ChannelLayout::Cmyka)
            | PixelFormat::U16(ChannelLayout::Cmyk | ChannelLayout::Cmyka)
    )
}

/// Settings covering every implemented syntax choice at least once.
fn request_matrix() -> Vec<(String, EncodeRequest)> {
    let mut requests = Vec::new();
    for overlap in [OverlapMode::None, OverlapMode::One, OverlapMode::Two] {
        requests.push((
            format!("lossless overlap={}", overlap as u8),
            EncodeRequest::lossless().with_overlap(overlap),
        ));
        for parameter in [12, 40, 90] {
            requests.push((
                format!("qp={parameter} overlap={}", overlap as u8),
                EncodeRequest::lossy(EncodeQuantizers::uniform(parameter)).with_overlap(overlap),
            ));
        }
        for hard_tiles in [false, true] {
            requests.push((
                format!(
                    "lossless tiles=2mb hard={hard_tiles} overlap={}",
                    overlap as u8
                ),
                EncodeRequest::lossless()
                    .with_overlap(overlap)
                    .with_max_tile_macroblocks(2)
                    .with_hard_tiles(hard_tiles),
            ));
        }
    }
    for bands in [
        BandPresence::NoFlexbits,
        BandPresence::NoHighPass,
        BandPresence::DcOnly,
    ] {
        requests.push((
            format!("qp=30 bands={bands:?}"),
            EncodeRequest::lossy(EncodeQuantizers::uniform(30)).with_bands(bands),
        ));
    }
    for trim in [2, 7] {
        requests.push((
            format!("lossless trim={trim}"),
            EncodeRequest::lossless().with_trim_flexbits(trim),
        ));
    }
    // The ends of the quality scale and one point on each of its segments:
    // every overlap mode, relieved DC, and quantizers up to 128.
    for quality in [99, 92, 75, 40, 5, 0] {
        requests.push((
            format!("quality={quality}"),
            EncodeRequest::quality(quality),
        ));
    }
    // What a quality leaves to the request: quantizers given for part of
    // the image, and an overlap mode the image may not be able to take.
    requests.push((
        "quality=60 chroma qp=40".to_owned(),
        EncodeRequest::quality(60).with_chroma_quantizers(EncodeQuantizers::uniform(40)),
    ));
    requests.push((
        "quality=3 tiles=1mb hard=true".to_owned(),
        EncodeRequest::quality(3)
            .with_max_tile_macroblocks(1)
            .with_hard_tiles(true),
    ));
    requests.push((
        "qp=30 overlap=auto".to_owned(),
        EncodeRequest {
            overlap: None,
            ..EncodeRequest::lossy(EncodeQuantizers::uniform(30))
        },
    ));
    requests.push((
        "qp=20 chroma qp=60".to_owned(),
        EncodeRequest::lossy(EncodeQuantizers::uniform(20))
            .with_chroma_quantizers(EncodeQuantizers::uniform(60)),
    ));
    requests.push((
        "qp dc=10 lp=30 hp=60".to_owned(),
        EncodeRequest::lossy(EncodeQuantizers {
            dc: 10,
            low_pass: 30,
            high_pass: 60,
        }),
    ));
    requests.extend(frequency_requests());
    requests
}

/// Frequency-mode settings: every band set, trimmed flexbits, soft and hard
/// tiles, every overlap mode, lossless and lossy. They are a part of the
/// spatial-mode matrix, because the two layouts differ only in where a
/// tile's bits go, and run in every coded form of a source as the rest do.
fn frequency_requests() -> Vec<(String, EncodeRequest)> {
    let lossy = |parameter| EncodeRequest::lossy(EncodeQuantizers::uniform(parameter));
    let tiled = |request: EncodeRequest| request.with_max_tile_macroblocks(2);
    [
        ("lossless overlap=0", EncodeRequest::lossless()),
        (
            "lossless tiles=2mb hard=false overlap=2",
            tiled(EncodeRequest::lossless()).with_overlap(OverlapMode::Two),
        ),
        (
            "lossless tiles=2mb hard=true overlap=1",
            tiled(EncodeRequest::lossless())
                .with_overlap(OverlapMode::One)
                .with_hard_tiles(true),
        ),
        ("qp=40 overlap=1", lossy(40)),
        (
            "qp=12 overlap=2 tiles=2mb hard=true",
            tiled(lossy(12))
                .with_overlap(OverlapMode::Two)
                .with_hard_tiles(true),
        ),
        (
            "qp=30 bands=NoFlexbits tiles=2mb",
            tiled(lossy(30)).with_bands(BandPresence::NoFlexbits),
        ),
        (
            "qp=30 bands=NoHighPass",
            lossy(30).with_bands(BandPresence::NoHighPass),
        ),
        (
            "qp=30 bands=DcOnly tiles=2mb",
            tiled(lossy(30)).with_bands(BandPresence::DcOnly),
        ),
        (
            "lossless trim=2",
            EncodeRequest::lossless().with_trim_flexbits(2),
        ),
        (
            "qp=20 trim=7 tiles=2mb",
            tiled(lossy(20)).with_trim_flexbits(7),
        ),
        ("quality=75", EncodeRequest::quality(75)),
    ]
    .into_iter()
    .map(|(label, request)| {
        (
            format!("{label} frequency"),
            request.with_bitstream_mode(BitstreamMode::Frequency),
        )
    })
    .collect()
}

/// What checking one setting of one source established.
enum Outcome {
    /// Both decoders agree on the encoded file of this many bytes.
    Decoded(usize),
    /// The encoder refused a setting the T.835 reference software does
    /// not accept for the image.
    Refused,
}

impl Source {
    /// The image as the encoder borrows it.
    fn image(&self) -> Result<SourceImage<'_>, JxrError> {
        let image = SourceImage::new(
            self.width,
            self.height,
            self.format,
            SourceSamples::from(&self.samples),
        )?;
        let image = if self.planes.is_empty() {
            image.with_row_stride(self.row_stride)?
        } else {
            image.with_planes(&self.planes)?
        };
        let image = match self.sample_bits {
            Some(bits) => image.with_sample_bits(bits)?,
            None => image,
        };
        image
            .with_chroma_centering(self.chroma_centering[0], self.chroma_centering[1])?
            .with_premultiplied_alpha(self.premultiplied)
    }

    /// Whether the T.835 reference software accepts no image of this size
    /// coded as `request` asks: second-level overlap of subsampled chroma
    /// needs an image, and under hard tiling every tile, at least two
    /// macroblocks wide. Only a request that names the mode can ask for
    /// that; the encoder never chooses it for such an image.
    fn refuses(&self, request: &EncodeRequest) -> bool {
        let subsampled = decimates_chroma(self.format, request)
            || matches!(
                self.format,
                PixelFormat::U8(
                    ChannelLayout::Yuv(ChromaSampling::Cs420 | ChromaSampling::Cs422)
                        | ChannelLayout::Yuva(ChromaSampling::Cs420 | ChromaSampling::Cs422)
                ) | PixelFormat::U16(
                    ChannelLayout::Yuv(ChromaSampling::Cs420 | ChromaSampling::Cs422)
                        | ChannelLayout::Yuva(ChromaSampling::Cs420 | ChromaSampling::Cs422)
                )
            );
        if !subsampled || request.overlap != Some(OverlapMode::Two) {
            return false;
        }
        // The narrowest tile of the near-equal split the encoder makes.
        let macroblocks = self.width.div_ceil(16);
        let tiles = macroblocks.div_ceil(u32::from(request.max_tile_macroblocks));
        macroblocks < 2 || (request.hard_tiles && macroblocks / tiles < 2)
    }
}

/// Encode `source`, compare both decoders on the result, and check a lossless
/// result against the source. The encoded file is also written to `keep`
/// when given.
fn check(
    oracle: &T835Oracle,
    source: &Source,
    source_bytes: &[u8],
    request: &EncodeRequest,
    keep: Option<&Path>,
) -> Result<Outcome, Box<dyn std::error::Error>> {
    let image = source.image()?;
    let encoded = match jxr::encode(&image, request) {
        Err(error) if error.kind == JxrErrorKind::InvalidRequest && source.refuses(request) => {
            return Ok(Outcome::Refused);
        }
        Ok(_) if source.refuses(request) => {
            return Err("encoded an image T.835 does not accept".into());
        }
        encoded => encoded?,
    };
    if let Some(path) = keep {
        std::fs::write(path, &encoded)?;
    }
    let file = TempFile::write("jxr-encode-check", &encoded)?;
    let decoded = decode_rust_file(file.path())?;
    let reference = oracle.decode_raw(file.path())?;
    if reference.bytes != decoded.bytes {
        return Err("T.835 and the Rust decoder disagree on the encoded image".into());
    }
    // The reference reports a profile or level declaration the codestream
    // does not meet as a warning rather than a failure.
    if let Some(warning) = reference
        .stderr
        .lines()
        .find(|line| line.contains("WARNING"))
    {
        return Err(format!("T.835: {}", warning.trim()).into());
    }
    let every_bit = request.bands == BandPresence::All && request.trim_flexbits == 0;
    // Both decoders' bytes are in the reference layout of the encoded file,
    // which is planar for an image declared CMYKDIRECT or YUV.
    let view = JxrView::parse(&encoded)?;
    let layout = oracle_format(view.info())?;
    let expected = reference_bytes(source_bytes, layout);
    // Chroma the encoder subsamples is not reproduced, whatever the
    // quantizers.
    let exact = every_bit && request.is_lossless() && !decimates_chroma(source.format, request);
    if exact && decoded.bytes != expected {
        return Err("lossless encode did not reproduce its source".into());
    }
    // An alpha plane is lossless by its own quantizers, whatever the colour
    // planes beside it use.
    if every_bit
        && source.format.has_alpha()
        && request.alpha_quantization().is_lossless()
        && reference_alpha_bytes(&decoded.bytes, layout) != reference_alpha_bytes(&expected, layout)
    {
        return Err("lossless alpha plane did not reproduce its source".into());
    }
    let info = view.info();
    let alpha_plane = info.alpha.as_ref();
    if info.primary.bitstream_mode != request.bitstream_mode
        || alpha_plane.is_some_and(|alpha| alpha.bitstream_mode != request.bitstream_mode)
    {
        return Err("encoded packet layout differs from the request".into());
    }
    if request.bitstream_mode == BitstreamMode::Frequency {
        check_reduced_decodes(&view, source)?;
    }
    if info.alpha_mode != expected_alpha_mode(source.format, request)
        || info.premultiplied_alpha != source.premultiplied
    {
        return Err("encoded alpha declaration differs from the request".into());
    }
    // The encoder's own decimation keeps the co-sited chroma grid.
    let centering = if decimates_chroma(source.format, request) {
        [0, 0]
    } else {
        source.chroma_centering
    };
    if info.primary.chroma_centering != centering {
        return Err("encoded chroma centering differs from the source".into());
    }
    if (decoded.width, decoded.height) != (source.width, source.height) {
        return Err("encoded dimensions differ from the source".into());
    }
    Ok(Outcome::Decoded(encoded.len()))
}

/// A frequency-mode file must decode at a quarter and a sixteenth of its
/// size from its low-pass and DC packets alone, to an image of the reduced
/// dimensions. T.835 has no reduced output to compare the samples with.
///
/// The decoder documents the requests it does not serve, and only those may
/// fail: a quarter of an image, or of its alpha plane, that codes no
/// low-pass band, and a sixteenth in a subsampled planar YUV layout.
fn check_reduced_decodes(
    view: &JxrView<'_>,
    source: &Source,
) -> Result<(), Box<dyn std::error::Error>> {
    let info = view.info();
    let layout = oracle_format(info)?;
    let low_pass = info.primary.bands.has_low_pass()
        && info
            .alpha
            .as_ref()
            .is_none_or(|alpha| alpha.bands.has_low_pass());
    for (scale, served) in [
        (DecodeScale::Quarter, low_pass),
        (DecodeScale::Sixteenth, layout.subsampling().is_none()),
    ] {
        let mut request = DecodeRequest::new(layout.pixel_format).with_scale(scale);
        request.alpha = layout.alpha;
        match view.decoder().decode(&request) {
            Ok(decoded) if served => {
                let denominator = scale.denominator();
                let expected = (
                    source.width.div_ceil(denominator),
                    source.height.div_ceil(denominator),
                );
                let plane = decoded.planes[0];
                if (plane.width, plane.height) != expected {
                    return Err(format!("{scale:?} decode has the wrong dimensions").into());
                }
            }
            Ok(_) => return Err(format!("{scale:?} decode served an undocumented case").into()),
            Err(error) if !served && error.kind == JxrErrorKind::Unsupported => {}
            Err(error) => return Err(format!("{scale:?} decode: {error}").into()),
        }
    }
    Ok(())
}

/// How a file encoded from `format` under `request` carries alpha.
fn expected_alpha_mode(format: PixelFormat, request: &EncodeRequest) -> AlphaMode {
    if format.has_alpha() {
        request.alpha_mode
    } else {
        AlphaMode::None
    }
}

/// Decode an existing image into the storage its container declares, or
/// `None` when the encoder does not support that pixel format.
fn load_source(path: &Path) -> Result<Option<Source>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let view = JxrView::parse(&bytes)?;
    let (width, height) = view.info().dimensions();
    // The container's own pixel format, so the re-encoded file declares the
    // layout the reference decoder writes its raw output in.
    let Ok(oracle) = oracle_format(view.info()) else {
        return Ok(None);
    };
    let mut request = DecodeRequest::new(oracle.pixel_format);
    request.alpha = oracle.alpha;
    let decoded = view.decoder().decode(&request)?;
    let element_bytes = decoded.samples.byte_len() / decoded.samples.sample_count().max(1);
    let element_bytes = if matches!(decoded.samples, DecodedSamples::BitPacked(_)) {
        1
    } else {
        element_bytes
    };
    if decoded.planes.iter().any(|plane| {
        !plane.byte_offset.is_multiple_of(element_bytes)
            || !plane.row_stride_bytes.is_multiple_of(element_bytes)
    }) {
        return Ok(None);
    }
    let planar = oracle.subsampling().is_some();
    let planes = match decoded.planes.as_slice() {
        [plane] if !planar && plane.byte_offset == 0 => Vec::new(),
        planes if planar => planes
            .iter()
            .map(|plane| SourcePlane {
                offset: plane.byte_offset / element_bytes,
                row_stride: plane.row_stride_bytes / element_bytes,
            })
            .collect(),
        _ => return Ok(None),
    };
    // Ten-bit samples in 16-bit elements; packed 10-bit RGB has its own
    // storage.
    let ten_bit = decoded.info.primary.sample_format == SampleFormat::Unsigned { bits: 10 }
        && matches!(decoded.samples, DecodedSamples::U16(_));
    let source = Source {
        name: path.file_name().map_or_else(
            || path.display().to_string(),
            |name| name.to_string_lossy().into_owned(),
        ),
        width,
        height,
        format: oracle.pixel_format,
        row_stride: decoded.planes[0].row_stride_bytes / element_bytes,
        planes,
        sample_bits: ten_bit.then_some(10),
        chroma_centering: if planar {
            decoded.info.primary.chroma_centering
        } else {
            [0, 0]
        },
        premultiplied: decoded.info.premultiplied_alpha,
        samples: decoded.samples,
    };
    // Probe with the image itself: support depends on the pixel format alone.
    match jxr::encode(&source.image()?, &EncodeRequest::lossless()) {
        Ok(_) => Ok(Some(source)),
        Err(error) if error.kind == JxrErrorKind::Unsupported => Ok(None),
        Err(error) => Err(error.into()),
    }
}

/// Images chosen to reach syntax that natural images rarely do: flat fields,
/// saturated extremes, full-range noise, partial macroblocks, and colour
/// whose channels are unrelated to one another, in every component syntax.
fn synthetic_sources() -> Vec<Source> {
    let mut set = SyntheticSet {
        sources: Vec::new(),
    };
    let mut noise = Noise(0x243f_6a88_85a3_08d3);
    gray_and_rgb_sources(&mut set, &mut noise);
    multichannel_sources(&mut set, &mut noise);
    alpha_sources(&mut set, &mut noise);
    yuv_sources(&mut set, &mut noise);
    set.sources
}

/// Deterministic full-range bytes.
struct Noise(u64);

impl Noise {
    fn next(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0.to_be_bytes()[0]
    }
}

/// Eight-bit images built sample by sample.
struct SyntheticSet {
    sources: Vec<Source>,
}

impl SyntheticSet {
    /// Add an image whose samples `sample` gives from `(x, y, channel)`.
    fn push(
        &mut self,
        name: &str,
        format: PixelFormat,
        (width, height): (u32, u32),
        mut sample: impl FnMut(u32, u32, u32) -> u8,
    ) {
        let channels = u32::from(format.channel_count());
        let mut samples = Vec::new();
        for y in 0..height {
            for x in 0..width {
                for channel in 0..channels {
                    samples.push(sample(x, y, channel));
                }
            }
        }
        self.sources.push(Source {
            name: name.to_owned(),
            width,
            height,
            format,
            row_stride: (width * channels) as usize,
            planes: Vec::new(),
            sample_bits: None,
            chroma_centering: [0, 0],
            premultiplied: false,
            samples: DecodedSamples::U8(samples),
        });
    }
}

impl SyntheticSet {
    /// Add a planar YUV image whose samples `sample` gives from
    /// `(x, y, plane)`: Y, U, V and then alpha planes, tightly packed one
    /// after another, the chroma planes at the sampling of `format`.
    fn push_planar(
        &mut self,
        name: &str,
        format: PixelFormat,
        (width, height): (u32, u32),
        mut sample: impl FnMut(u32, u32, u32) -> u8,
    ) {
        let PixelFormat::U8(ChannelLayout::Yuv(sampling) | ChannelLayout::Yuva(sampling)) = format
        else {
            panic!("{format:?} is not an eight-bit YUV format");
        };
        let chroma_height = if sampling == ChromaSampling::Cs420 {
            height / 2
        } else {
            height
        };
        let mut samples = Vec::new();
        let mut planes = Vec::new();
        for plane in 0..u32::from(format.channel_count()) {
            let (plane_width, plane_height) = if matches!(plane, 1 | 2) {
                (width / 2, chroma_height)
            } else {
                (width, height)
            };
            planes.push(SourcePlane {
                offset: samples.len(),
                row_stride: plane_width as usize,
            });
            for y in 0..plane_height {
                for x in 0..plane_width {
                    samples.push(sample(x, y, plane));
                }
            }
        }
        self.sources.push(Source {
            name: name.to_owned(),
            width,
            height,
            format,
            row_stride: width as usize,
            planes,
            sample_bits: None,
            chroma_centering: [0, 0],
            premultiplied: false,
            samples: DecodedSamples::U8(samples),
        });
    }

    /// Declare the colour samples of the image last added premultiplied by
    /// its alpha.
    fn mark_premultiplied(&mut self) {
        if let Some(source) = self.sources.last_mut() {
            source.premultiplied = true;
        }
    }
}

fn gray_and_rgb_sources(set: &mut SyntheticSet, noise: &mut Noise) {
    let gray = PixelFormat::U8(ChannelLayout::Luma);
    let rgb = PixelFormat::U8(ChannelLayout::Rgb);
    let bgr = PixelFormat::U8(ChannelLayout::Bgr);
    set.push("flat-black", gray, (33, 17), |_, _, _| 0);
    set.push("flat-white", gray, (48, 48), |_, _, _| 255);
    set.push("one-pixel", gray, (1, 1), |_, _, _| 77);
    set.push("gradient", gray, (200, 120), |x, y, _| low_byte(x + 2 * y));
    set.push("checker", gray, (64, 64), |x, y, _| {
        if (x + y).is_multiple_of(2) { 0 } else { 255 }
    });
    set.push("noise", gray, (97, 61), |_, _, _| noise.next());
    set.push("wide-strip", gray, (700, 20), |x, y, _| {
        low_byte((x * 7) ^ (y * 13))
    });
    set.push("tall-strip", gray, (20, 700), |x, y, _| {
        low_byte((x * 11) ^ (y * 5))
    });
    set.push("rgb-one-pixel", rgb, (1, 1), |_, _, channel| {
        [255, 0, 128][channel as usize]
    });
    set.push("rgb-primaries", rgb, (50, 34), |x, y, channel| {
        if (x / 10 + y / 10) % 3 == channel {
            255
        } else {
            0
        }
    });
    set.push("rgb-gradient", rgb, (160, 90), |x, y, channel| {
        low_byte(x * (channel + 1) + y * (3 - channel))
    });
    set.push("bgr-gradient", bgr, (90, 160), |x, y, channel| {
        low_byte(x * (channel + 2) + y * (4 - channel))
    });
    set.push("rgb-noise", rgb, (61, 97), |_, _, _| noise.next());
    set.push("bgr-wide-strip", bgr, (700, 20), |x, y, channel| {
        low_byte((x * 7) ^ (y * 13) ^ (channel * 85))
    });
}

/// CMYK, which has a colour transform, and N-component, which has none.
fn multichannel_sources(set: &mut SyntheticSet, noise: &mut Noise) {
    let cmyk = PixelFormat::U8(ChannelLayout::Cmyk);
    let channels = |count: u16| PixelFormat::U8(ChannelLayout::NComponent(count));
    set.push("cmyk-one-pixel", cmyk, (1, 1), |_, _, channel| {
        [255, 0, 128, 1][channel as usize]
    });
    set.push("cmyk-inks", cmyk, (50, 34), |x, y, channel| {
        if (x / 10 + y / 10) % 4 == channel {
            255
        } else {
            0
        }
    });
    set.push("cmyk-gradient", cmyk, (160, 90), |x, y, channel| {
        low_byte(x * (channel + 1) + y * (4 - channel))
    });
    set.push("cmyk-noise", cmyk, (61, 97), |_, _, _| noise.next());
    set.push("cmyk-wide-strip", cmyk, (700, 20), |x, y, channel| {
        low_byte((x * 7) ^ (y * 13) ^ (channel * 60))
    });
    set.push("3channel-noise", channels(3), (61, 97), |_, _, _| {
        noise.next()
    });
    set.push(
        "4channel-one-pixel",
        channels(4),
        (1, 1),
        |_, _, channel| [255, 0, 128, 1][channel as usize],
    );
    set.push(
        "5channel-gradient",
        channels(5),
        (160, 90),
        |x, y, channel| low_byte(x * (channel + 1) + y * (5 - channel)),
    );
    set.push(
        "6channel-wide-strip",
        channels(6),
        (700, 20),
        |x, y, channel| low_byte((x * 7) ^ (y * 13) ^ (channel * 40)),
    );
    set.push("8channel-bands", channels(8), (50, 34), |x, y, channel| {
        if (x / 10 + y / 10) % 8 == channel {
            255
        } else {
            0
        }
    });
    set.push("8channel-noise", channels(8), (97, 61), |_, _, _| {
        noise.next()
    });
}

/// Every family with an alpha channel: alpha that follows the colour, that
/// is unrelated to it, that is flat, and that the colour is multiplied by.
fn alpha_sources(set: &mut SyntheticSet, noise: &mut Noise) {
    let bgra = PixelFormat::U8(ChannelLayout::Bgra);
    let cmyka = PixelFormat::U8(ChannelLayout::Cmyka);
    let channels = |count: u16| PixelFormat::U8(ChannelLayout::NComponentAlpha(count));
    set.push("bgra-one-pixel", bgra, (1, 1), |_, _, channel| {
        [255, 0, 128, 77][channel as usize]
    });
    set.push("bgra-opaque", bgra, (50, 34), |x, y, channel| {
        if channel == 3 {
            255
        } else {
            low_byte(x * (channel + 2) + y * 3)
        }
    });
    set.push("bgra-gradient", bgra, (160, 90), |x, y, channel| {
        low_byte(x * (channel + 1) + y * (4 - channel))
    });
    set.push("bgra-noise", bgra, (61, 97), |_, _, _| noise.next());
    set.push("bgra-wide-strip", bgra, (700, 20), |x, y, channel| {
        low_byte((x * 7) ^ (y * 13) ^ (channel * 60))
    });
    // Colour no greater than the alpha it is multiplied by.
    set.push("bgra-premultiplied", bgra, (90, 70), |x, y, channel| {
        let alpha = low_byte(x * 2 + y);
        if channel == 3 {
            alpha
        } else {
            low_byte(u32::from(alpha) * ((x + y * (channel + 1)) % 256) / 255)
        }
    });
    set.mark_premultiplied();
    set.push("cmyka-one-pixel", cmyka, (1, 1), |_, _, channel| {
        [255, 0, 128, 1, 200][channel as usize]
    });
    set.push("cmyka-gradient", cmyka, (160, 90), |x, y, channel| {
        low_byte(x * (channel + 1) + y * (5 - channel))
    });
    set.push("cmyka-noise", cmyka, (61, 97), |_, _, _| noise.next());
    set.push("3channel-alpha-noise", channels(3), (61, 97), |_, _, _| {
        noise.next()
    });
    set.push(
        "5channel-alpha-wide-strip",
        channels(5),
        (700, 20),
        |x, y, channel| low_byte((x * 7) ^ (y * 13) ^ (channel * 40)),
    );
    set.push(
        "8channel-alpha-gradient",
        channels(8),
        (97, 61),
        |x, y, channel| low_byte(x * (channel + 1) + y * (9 - channel)),
    );
}

/// YUV at each chroma sampling, with and without alpha: interleaved at full
/// resolution and planar when subsampled, down to one chroma sample.
fn yuv_sources(set: &mut SyntheticSet, noise: &mut Noise) {
    let yuv = |sampling| PixelFormat::U8(ChannelLayout::Yuv(sampling));
    let yuva = |sampling| PixelFormat::U8(ChannelLayout::Yuva(sampling));
    let (full, half, quarter) = (
        ChromaSampling::Cs444,
        ChromaSampling::Cs422,
        ChromaSampling::Cs420,
    );
    set.push("yuv444-noise", yuv(full), (61, 97), |_, _, _| noise.next());
    set.push(
        "yuv444-wide-strip",
        yuv(full),
        (700, 20),
        |x, y, channel| low_byte((x * 7) ^ (y * 13) ^ (channel * 85)),
    );
    set.push(
        "yuva444-gradient",
        yuva(full),
        (160, 90),
        |x, y, channel| low_byte(x * (channel + 1) + y * (4 - channel)),
    );
    set.push_planar("yuv422-two-pixels", yuv(half), (2, 1), |x, _, plane| {
        [[255, 3], [128, 0], [77, 0]][plane as usize][x as usize]
    });
    set.push_planar("yuv422-noise", yuv(half), (98, 61), |_, _, _| noise.next());
    set.push_planar("yuv422-gradient", yuv(half), (160, 90), |x, y, plane| {
        low_byte(x * (plane + 2) + y * (3 - plane))
    });
    set.push_planar("yuv422-wide-strip", yuv(half), (700, 20), |x, y, plane| {
        low_byte((x * 7) ^ (y * 13) ^ (plane * 85))
    });
    set.push_planar("yuva422-noise", yuva(half), (62, 97), |_, _, _| {
        noise.next()
    });
    set.push_planar("yuv420-four-pixels", yuv(quarter), (2, 2), |x, y, plane| {
        [[255, 3, 0, 200], [128; 4], [77; 4]][plane as usize][(y * 2 + x) as usize]
    });
    set.push_planar("yuv420-noise", yuv(quarter), (62, 98), |_, _, _| {
        noise.next()
    });
    set.push_planar("yuv420-gradient", yuv(quarter), (160, 90), |x, y, plane| {
        low_byte(x * (plane + 2) + y * (3 - plane))
    });
    set.push_planar(
        "yuv420-wide-strip",
        yuv(quarter),
        (700, 20),
        |x, y, plane| low_byte((x * 7) ^ (y * 13) ^ (plane * 85)),
    );
    set.push_planar(
        "yuv420-tall-strip",
        yuv(quarter),
        (20, 700),
        |x, y, plane| low_byte((x * 11) ^ (y * 5) ^ (plane * 60)),
    );
    set.push_planar("yuva420-noise", yuva(quarter), (98, 62), |_, _, _| {
        noise.next()
    });
    set.push_planar(
        "yuva420-wide-strip",
        yuva(quarter),
        (700, 20),
        |x, y, plane| low_byte((x * 7) ^ (y * 13) ^ (plane * 60)),
    );
}

fn low_byte(value: u32) -> u8 {
    value.to_le_bytes()[0]
}
