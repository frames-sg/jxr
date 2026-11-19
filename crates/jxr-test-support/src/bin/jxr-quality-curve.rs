// SPDX-License-Identifier: MIT OR Apache-2.0

//! Size and error of the encoder's quality scale, measured on real images.
//!
//! Each source picture is stored as every class the scale is calibrated
//! for: 8 and 16 bits, fixed and floating point, packed, gray, RGB, CMYK,
//! separate channels, YUV, with subsampled chroma and with alpha. Every
//! class is encoded at each quality from 100 down to 0 and decoded again.
//! The table of a class gives, per quality, its size, the PSNR over all
//! samples against the class's peak, the PSNR and SSIM of luma, the largest
//! sample error, the quantizers the file declares, and how many sources
//! reversed or repeated the step from the quality above. This is how the
//! tables in `docs/support.md` are produced, and how to recalibrate the
//! scale after a change to the encoder.
//!
//! The exit status is non-zero if any source codes larger, or with less
//! error, at one quality than at the quality above it.

use std::path::{Path, PathBuf};

use jxr::{ChannelLayout, DecodeRequest, DecodedSamples, JxrView, PixelFormat};
use jxr_test_support::{
    QualityClass, QualityMeasurement, QualityPicture, QualitySource, exit_with_failures,
};
use rayon::prelude::*;

const USAGE: &str = "\
usage: jxr-quality-curve [--class NAME,...] [--rows QUALITY,...] [--float-precision MANTISSA,BIAS]
                         [image.jxr ...] [--halve image.jxr ...]
Sources must be 8-bit gray, RGB or BGR. Sources after --halve are reduced to
half size first: a source that is itself a lossy JPEG XR image carries a
block grid the encoder would otherwise realign with.
--class limits the run to the named classes; the default is every class the
sources can be stored as. --rows limits the table to the named qualities;
every quality is measured either way. --float-precision codes 32-bit floats
with that many mantissa bits and that exponent bias instead of the default.";

fn main() {
    exit_with_failures("jxr-quality-curve", run());
}

/// What the command line asks for.
struct Options {
    pictures: Vec<QualityPicture>,
    classes: Vec<QualityClass>,
    /// Qualities the tables list.
    rows: Vec<u8>,
    /// Mantissa bits and exponent bias 32-bit floats are coded with, when
    /// not the default.
    float_precision: Option<(u8, i8)>,
}

fn options() -> Result<Option<Options>, Box<dyn std::error::Error>> {
    let mut options = Options {
        pictures: Vec::new(),
        classes: QualityClass::ALL.to_vec(),
        rows: (0..=100).rev().collect(),
        float_precision: None,
    };
    let mut halve = false;
    let mut arguments = std::env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        let mut value = |name: &str| {
            arguments
                .next()
                .and_then(|value| value.into_string().ok())
                .ok_or_else(|| format!("{name} needs a value"))
        };
        match argument.to_str() {
            Some("--help") => {
                println!("{USAGE}");
                println!("\nclasses:");
                for class in QualityClass::ALL {
                    println!("  {}: {}", class.name(), class.description());
                }
                return Ok(None);
            }
            Some("--halve") => halve = true,
            Some("--class") => {
                options.classes = value("--class")?
                    .split(',')
                    .map(|name| {
                        QualityClass::from_name(name).ok_or_else(|| format!("no class {name}"))
                    })
                    .collect::<Result<_, _>>()?;
            }
            Some("--rows") => {
                options.rows = value("--rows")?
                    .split(',')
                    .map(|quality| quality.parse::<u8>().ok().filter(|quality| *quality <= 100))
                    .collect::<Option<_>>()
                    .ok_or("--rows takes qualities from 0 to 100")?;
            }
            Some("--float-precision") => {
                let value = value("--float-precision")?;
                options.float_precision = value.split_once(',').and_then(|(mantissa, bias)| {
                    Some((mantissa.parse().ok()?, bias.parse().ok()?))
                });
                if options.float_precision.is_none() {
                    return Err("--float-precision takes mantissa bits and a bias, as 13,4".into());
                }
            }
            _ => options
                .pictures
                .push(load_picture(&PathBuf::from(argument), halve)?),
        }
    }
    if options.pictures.is_empty() {
        return Err("no source images given; see --help".into());
    }
    Ok(Some(options))
}

fn run() -> Result<u64, Box<dyn std::error::Error>> {
    let Some(options) = options()? else {
        return Ok(0);
    };
    let mut reversals = 0;
    for class in options.classes {
        let sources: Vec<QualitySource> = options
            .pictures
            .iter()
            .filter_map(|picture| class.source(picture))
            .collect();
        if sources.is_empty() {
            continue;
        }
        // Row `index` of each curve is quality `100 - index`.
        let curves = sources
            .iter()
            .map(|source| {
                (0..=100_u8)
                    .into_par_iter()
                    .map(|index| {
                        let request = class.request(100 - index);
                        source.measure(&match options.float_precision {
                            Some((mantissa, bias)) => request.with_float_precision(mantissa, bias),
                            None => request,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, String>>()
            .map_err(|error| format!("{}: {error}", class.name()))?;
        println!("class\t{}\t{}", class.name(), class.description());
        print_table(&sources, &curves, &options.rows);
        reversals += report_steps(class, &sources, &curves);
        println!();
    }
    Ok(reversals)
}

/// How one source's step from the quality above `index` came out.
#[derive(PartialEq, Eq)]
enum Step {
    Smaller,
    /// The same output as the quality above.
    Repeated,
    /// A larger file than the quality above.
    Larger,
    /// Less error than the quality above, in a file no larger.
    Closer,
}

fn step(curve: &[QualityMeasurement], index: usize) -> Step {
    let (Some(above), below) = (
        index.checked_sub(1).map(|above| &curve[above]),
        &curve[index],
    ) else {
        return Step::Smaller;
    };
    if above == below {
        Step::Repeated
    } else if below.bytes > above.bytes {
        Step::Larger
    } else if below.squared_error < above.squared_error {
        Step::Closer
    } else {
        Step::Smaller
    }
}

fn print_table(sources: &[QualitySource], curves: &[Vec<QualityMeasurement>], rows: &[u8]) {
    println!(
        "quality\tbytes\tbits_per_pixel\tsize_step_percent\tpsnr_db\tluma_psnr_db\tluma_ssim\t\
         alpha_psnr_db\tmax_error\tquantizers\toverlap\tsteps"
    );
    let bits = |index: usize| {
        mean(
            sources
                .iter()
                .zip(curves)
                .map(|(source, curve)| bits_per_pixel(source, &curve[index]).ln()),
        )
        .exp()
    };
    let optional = |value: Option<f64>, digits: usize| {
        value.map_or_else(|| "-".to_owned(), |value| format!("{value:.digits$}"))
    };
    for &quality in rows {
        let index = usize::from(100 - quality);
        let row: Vec<(&QualitySource, &QualityMeasurement)> = sources
            .iter()
            .zip(curves)
            .map(|(source, curve)| (source, &curve[index]))
            .collect();
        // Mean over the sources of a measure only some classes have.
        let over = |measure: &dyn Fn(&QualitySource, &QualityMeasurement) -> Option<f64>| {
            row.iter()
                .map(|(source, m)| measure(source, m))
                .collect::<Option<Vec<f64>>>()
                .map(|values| mean(values.into_iter()))
        };
        let step_percent = index
            .checked_sub(1)
            .map_or(0.0, |above| 100.0 * (bits(index) / bits(above) - 1.0));
        let count = |kind: Step| {
            curves
                .iter()
                .filter(|curve| step(curve, index) == kind)
                .count()
        };
        let steps = [
            (count(Step::Larger), "larger"),
            (count(Step::Closer), "closer"),
            (count(Step::Repeated), "repeated"),
        ]
        .into_iter()
        .filter(|(count, _)| *count != 0)
        .map(|(count, name)| format!("{count} {name}"))
        .collect::<Vec<_>>();
        let first = row[0].1;
        println!(
            "{quality}\t{}\t{:.3}\t{step_percent:.1}\t{:.2}\t{}\t{}\t{}\t{:.1}\t{}\t{}\t{}",
            row.iter().map(|(_, m)| m.bytes).sum::<usize>(),
            bits(index),
            mean(row.iter().map(|(_, m)| m.psnr())),
            optional(over(&|source, m| m.luma_psnr(source.pixels())), 2),
            optional(over(&|_, m| m.luma.map(|(_, ssim)| ssim)), 4),
            optional(over(&|source, m| m.alpha_psnr(source.pixels())), 2),
            row.iter().map(|(_, m)| m.max_error).fold(0.0, f64::max),
            first
                .parameters
                .iter()
                .map(|[dc, low_pass, high_pass]| format!("{dc}/{low_pass}/{high_pass}"))
                .collect::<Vec<_>>()
                .join(" "),
            first.overlap,
            if steps.is_empty() {
                "-".to_owned()
            } else {
                steps.join(", ")
            },
        );
    }
}

/// Report every step that reverses and count the steps that repeat their
/// neighbour's output. Returns the number of reversals.
fn report_steps(
    class: QualityClass,
    sources: &[QualitySource],
    curves: &[Vec<QualityMeasurement>],
) -> u64 {
    let (mut larger, mut closer, mut repeats) = (0_u64, 0_u64, 0_u64);
    for (source, curve) in sources.iter().zip(curves) {
        for index in 1..curve.len() {
            let (above, below) = (&curve[index - 1], &curve[index]);
            let kind = step(curve, index);
            match kind {
                Step::Smaller => continue,
                Step::Repeated => {
                    repeats += 1;
                    continue;
                }
                Step::Larger => larger += 1,
                Step::Closer => closer += 1,
            }
            println!(
                "reversal\t{}\t{}\tquality {}: {} bytes after {}, squared error {} after {}",
                class.name(),
                source.name,
                100 - index,
                below.bytes,
                above.bytes,
                below.squared_error,
                above.squared_error
            );
        }
    }
    println!(
        "summary\t{}\t{} sources, {larger} steps to a larger file, {closer} steps to less \
         error, {repeats} steps repeating the output above",
        class.name(),
        sources.len()
    );
    larger + closer
}

fn bits_per_pixel(source: &QualitySource, measurement: &QualityMeasurement) -> f64 {
    // File sizes stay far below the 2^53 bytes an f64 counts exactly.
    let bytes = u32::try_from(measurement.bytes).map_or(f64::INFINITY, f64::from);
    bytes * 8.0 / (f64::from(source.width) * f64::from(source.height))
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, count) = values.fold((0.0, 0.0), |(sum, count), value| (sum + value, count + 1.0));
    sum / count
}

/// Decode `path` as 8-bit gray or RGB, optionally at half size.
fn load_picture(path: &Path, halve: bool) -> Result<QualityPicture, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let view = JxrView::parse(&bytes)?;
    let format = match jxr_test_support::oracle_format(view.info())?.pixel_format {
        PixelFormat::U8(ChannelLayout::Luma) => PixelFormat::U8(ChannelLayout::Luma),
        PixelFormat::U8(ChannelLayout::Rgb | ChannelLayout::Bgr) => {
            PixelFormat::U8(ChannelLayout::Rgb)
        }
        other => {
            return Err(format!(
                "{}: the quality scale is measured on 8-bit gray or RGB, not {other:?}",
                path.display()
            )
            .into());
        }
    };
    let (mut width, mut height, mut pixels) = decode_packed(&bytes, format)?;
    let mut name = path.file_name().map_or_else(
        || path.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    );
    if halve {
        (width, height, pixels) = halved(width, height, &pixels, format);
        name.push_str(" (halved)");
    }
    if width == 0 || height == 0 {
        return Err(format!("{}: too small to halve", path.display()).into());
    }
    Ok(QualityPicture {
        name,
        width,
        height,
        channels: usize::from(format.channel_count()),
        pixels,
    })
}

/// Decode an image to `format` with rows packed end to end.
fn decode_packed(bytes: &[u8], format: PixelFormat) -> Result<(u32, u32, Vec<u8>), String> {
    let view = JxrView::parse(bytes).map_err(|error| error.to_string())?;
    let (width, height) = view.info().dimensions();
    let image = view
        .decoder()
        .decode(&DecodeRequest::new(format))
        .map_err(|error| error.to_string())?;
    let (DecodedSamples::U8(samples), [plane]) = (&image.samples, image.planes.as_slice()) else {
        return Err(format!("decode returned {:?}", image.format));
    };
    let row = width as usize * usize::from(format.channel_count());
    let mut pixels = Vec::with_capacity(row * height as usize);
    for y in 0..height as usize {
        let start = plane.byte_offset + y * plane.row_stride_bytes;
        pixels.extend_from_slice(&samples[start..start + row]);
    }
    Ok((width, height, pixels))
}

/// Average each 2-by-2 block of samples, dropping an odd last row or column.
fn halved(width: u32, height: u32, pixels: &[u8], format: PixelFormat) -> (u32, u32, Vec<u8>) {
    let channels = usize::from(format.channel_count());
    let stride = width as usize * channels;
    let (half_width, half_height) = (width / 2, height / 2);
    let mut out = Vec::with_capacity(half_width as usize * half_height as usize * channels);
    for y in 0..half_height as usize {
        for x in 0..half_width as usize {
            for channel in 0..channels {
                let at = |dx: usize, dy: usize| {
                    u16::from(pixels[(2 * y + dy) * stride + (2 * x + dx) * channels + channel])
                };
                let sum = at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1);
                // Four 8-bit samples average to at most 255.
                out.push(u8::try_from((sum + 2) / 4).unwrap_or(u8::MAX));
            }
        }
    }
    (half_width, half_height, out)
}
