// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    hint::black_box,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use jxr::{
    AlphaHandling, BackendRequest, ChannelLayout, ChromaSampling, CpuBatchSamples, DecodeRequest,
    DecodedSamples, JxrView, PixelFormat,
};

use crate::{OracleError, compare::rust_error, oracle_format};

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// A duration in milliseconds, as the tools print timings.
#[must_use]
pub fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

/// A file a decode benchmark reads, held whole, with the format and alpha
/// handling the reference decoder writes it in, which the benchmark asks
/// every backend for.
#[derive(Clone, Debug)]
pub struct BenchmarkFile {
    /// Where the file was read from.
    pub path: PathBuf,
    /// The whole file.
    pub source: Arc<[u8]>,
    /// The pixel format of the reference decoder's output.
    pub format: PixelFormat,
    /// The alpha handling of the reference decoder's output.
    pub alpha: AlphaHandling,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl BenchmarkFile {
    /// Reads and parses the file at `path`.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or parsed, or has no
    /// format the reference decoder writes.
    pub fn load(path: &Path) -> Result<Self, OracleError> {
        let source: Arc<[u8]> = std::fs::read(path)
            .map_err(|error| OracleError::Io {
                operation: "read benchmark input",
                path: path.to_owned(),
                source: error,
            })?
            .into();
        let view = JxrView::parse(&source).map_err(|error| rust_error(&error))?;
        let format = oracle_format(view.info())?;
        let (width, height) = (view.info().width, view.info().height);
        Ok(Self {
            path: path.to_owned(),
            source,
            format: format.pixel_format,
            alpha: format.alpha,
            width,
            height,
        })
    }

    /// A whole-image decode of the file in its benchmark format on `backend`.
    #[must_use]
    pub fn request(&self, backend: BackendRequest) -> DecodeRequest {
        DecodeRequest::new(self.format)
            .with_alpha(self.alpha)
            .with_backend(backend)
    }
}

/// Checks that the `output` checksum of a benchmark iteration is the one
/// its first decode set, keeping the value observed so the work behind it
/// is not optimized away.
///
/// # Errors
///
/// Returns a message naming both values when they differ.
pub fn check_checksum(output: &str, actual: u64, expected: u64) -> Result<(), String> {
    black_box(actual);
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "{output} checksum changed: expected {expected:016x}, got {actual:016x}"
        ))
    }
}

/// Samples of a `width` by `height` image of `format` in its own storage:
/// interleaved pixels, or for planar YUV a Y plane and then the U and V
/// planes at their own sampling. Shared by the encode-throughput benchmark
/// and the allocation report so both measure the same input image.
///
/// The field uses six triangle-wave octaves under light noise. Supported
/// benchmark formats are interleaved U8, U16 and F32, and planar U8 YUV420;
/// planar dimensions must be even.
///
/// # Errors
///
/// Returns an error for an unsupported sample format, odd planar dimensions,
/// or dimensions that do not fit in `usize`.
pub fn synthetic_samples(
    format: PixelFormat,
    width: u32,
    height: u32,
) -> Result<DecodedSamples, Box<dyn std::error::Error>> {
    let (width, height) = (usize::try_from(width)?, usize::try_from(height)?);
    let channels = usize::from(format.channel_count());
    let planar = format.planar_sampling().is_some();
    if planar {
        if format != PixelFormat::U8(ChannelLayout::Yuv(ChromaSampling::Cs420)) {
            return Err(format!("no synthetic samples for {format:?}").into());
        }
        if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err("planar YUV420 benchmark images need even dimensions".into());
        }
    }
    let size = (width, height);
    Ok(match format {
        PixelFormat::U8(_) => DecodedSamples::U8(planes(planar, size, channels, |value| {
            value.to_be_bytes()[0]
        })),
        PixelFormat::U16(_) => DecodedSamples::U16(planes(planar, size, channels, |value| value)),
        // Linear light from zero to one.
        PixelFormat::F32(_) => DecodedSamples::F32(planes(planar, size, channels, |value| {
            f32::from(value) / 65_535.0
        })),
        _ => return Err(format!("no synthetic samples for {format:?}").into()),
    })
}

/// The samples of an image of `channels` interleaved channels, or of a
/// planar 4:2:0 one, each converted by `convert` as it is made, so that the
/// process never holds more than the source itself.
fn planes<T>(
    planar: bool,
    (width, height): (usize, usize),
    channels: usize,
    convert: impl Fn(u16) -> T + Copy,
) -> Vec<T> {
    let mut values = Vec::new();
    if planar {
        field(&mut values, (width, height), 1, 0, convert);
        field(&mut values, (width / 2, height / 2), 1, 1, convert);
        field(&mut values, (width / 2, height / 2), 1, 2, convert);
    } else {
        field(&mut values, (width, height), channels, 0, convert);
    }
    values
}

/// Appends interleaved 16-bit samples of `channels` channels to `values`,
/// each converted by `convert`. Each channel is a sum
/// of triangle waves whose amplitude falls with their period, along the
/// axes and both diagonals, under noise of about one percent of the range:
/// detail at every scale, as a photograph has, and channels that correlate
/// without repeating one another.
fn field<T>(
    values: &mut Vec<T>,
    (width, height): (usize, usize),
    channels: usize,
    first_channel: usize,
    convert: impl Fn(u16) -> T,
) {
    // `(period in samples, peak amplitude)` of each octave.
    const OCTAVES: [(usize, usize); 6] = [
        (1531, 0x4000),
        (389, 0x2800),
        (97, 0x1400),
        (29, 0x1800),
        (11, 0x1400),
        (5, 0x1000),
    ];
    let triangle = |position: usize, (period, amplitude): (usize, usize)| {
        let phase = position % period;
        phase.min(period - phase) * 2 * amplitude / period
    };
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    values.reserve_exact(width * height * channels);
    for y in 0..height {
        for x in 0..width {
            for channel in first_channel..first_channel + channels {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let noise = usize::try_from(state >> 54).unwrap_or(0);
                // Each octave runs along its own direction, shifted per
                // channel so no channel is a copy of another.
                let positions = [
                    x + 97 * channel,
                    y + 53 * channel,
                    x + y,
                    x + 2 * height - y,
                    2 * x + y + 31 * channel,
                    x + 3 * y,
                ];
                let value = OCTAVES
                    .into_iter()
                    .zip(positions)
                    .map(|(octave, position)| triangle(position, octave))
                    .sum::<usize>()
                    + noise;
                values.push(convert(u16::try_from(value).unwrap_or(u16::MAX)));
            }
        }
    }
}

/// Distribution summary for one set of decode timings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimingSummary {
    /// Fastest observed sample.
    pub minimum: Duration,
    /// Nearest-rank 50th percentile.
    pub median: Duration,
    /// Nearest-rank 95th percentile.
    pub p95: Duration,
    /// Arithmetic mean, rounded down to the nearest nanosecond.
    pub mean: Duration,
}

/// Summarize a non-empty timing sample without modifying caller-owned values.
#[must_use]
pub fn summarize_timings(samples: &[Duration]) -> Option<TimingSummary> {
    if samples.is_empty() {
        return None;
    }

    let mut ordered = samples.to_vec();
    ordered.sort_unstable();
    let total_nanos = ordered.iter().map(Duration::as_nanos).sum::<u128>();
    let mean_nanos = total_nanos / ordered.len() as u128;
    let mean_seconds = u64::try_from(mean_nanos / 1_000_000_000).ok()?;
    let mean_subseconds = u32::try_from(mean_nanos % 1_000_000_000).ok()?;

    Some(TimingSummary {
        minimum: ordered[0],
        median: ordered[nearest_rank_index(ordered.len(), 50)],
        p95: ordered[nearest_rank_index(ordered.len(), 95)],
        mean: Duration::new(mean_seconds, mean_subseconds),
    })
}

const fn nearest_rank_index(sample_count: usize, percentile: usize) -> usize {
    percentile
        .saturating_mul(sample_count)
        .saturating_add(99)
        .saturating_div(100)
        .saturating_sub(1)
}

/// Compute a stable FNV-1a checksum used to keep benchmark output observable.
#[must_use]
pub fn checksum_bytes(bytes: &[u8]) -> u64 {
    checksum_chunk(FNV_OFFSET_BASIS, bytes)
}

/// Checksum typed output in the same native-endian representation as host readback.
#[must_use]
pub fn checksum_samples(samples: &DecodedSamples) -> u64 {
    match samples {
        DecodedSamples::BitPacked(values) | DecodedSamples::U8(values) => checksum_bytes(values),
        DecodedSamples::U16(values)
        | DecodedSamples::F16(values)
        | DecodedSamples::Rgb555(values)
        | DecodedSamples::Rgb565(values) => checksum_values(values, u16::to_ne_bytes),
        DecodedSamples::I16(values) => checksum_values(values, i16::to_ne_bytes),
        DecodedSamples::I32(values) => checksum_values(values, i32::to_ne_bytes),
        DecodedSamples::F32(values) => checksum_values(values, f32::to_ne_bytes),
        DecodedSamples::Rgb101010(values) | DecodedSamples::Rgbe(values) => {
            checksum_values(values, u32::to_ne_bytes)
        }
    }
}

/// Checksum one image inside a dense native CPU batch owner.
#[must_use]
pub fn checksum_cpu_batch_image(
    samples: &CpuBatchSamples,
    image: usize,
    stride_elements: usize,
) -> Option<u64> {
    let start = image.checked_mul(stride_elements)?;
    let end = start.checked_add(stride_elements)?;
    Some(match samples {
        CpuBatchSamples::BitPacked(values) | CpuBatchSamples::U8(values) => {
            checksum_bytes(values.get(start..end)?)
        }
        CpuBatchSamples::U16(values)
        | CpuBatchSamples::F16(values)
        | CpuBatchSamples::Rgb555(values)
        | CpuBatchSamples::Rgb565(values) => {
            checksum_values(values.get(start..end)?, u16::to_ne_bytes)
        }
        CpuBatchSamples::I16(values) => checksum_values(values.get(start..end)?, i16::to_ne_bytes),
        CpuBatchSamples::I32(values) => checksum_values(values.get(start..end)?, i32::to_ne_bytes),
        CpuBatchSamples::F32(values) => checksum_values(values.get(start..end)?, f32::to_ne_bytes),
        CpuBatchSamples::Rgb101010(values) | CpuBatchSamples::Rgbe(values) => {
            checksum_values(values.get(start..end)?, u32::to_ne_bytes)
        }
        _ => return None,
    })
}

fn checksum_values<T: Copy, const N: usize>(values: &[T], bytes: impl Fn(T) -> [u8; N]) -> u64 {
    values
        .iter()
        .copied()
        .fold(FNV_OFFSET_BASIS, |hash, value| {
            checksum_chunk(hash, &bytes(value))
        })
}

fn checksum_chunk(initial: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(initial, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use jxr::{ChannelLayout, ChromaSampling, DecodedSamples, PixelFormat};

    use super::{
        BenchmarkFile, check_checksum, checksum_bytes, checksum_cpu_batch_image, checksum_samples,
        summarize_timings, synthetic_samples,
    };
    use crate::OracleError;

    #[test]
    fn synthetic_planar_samples_require_the_supported_layout_and_dimensions() {
        let yuv420 = PixelFormat::U8(ChannelLayout::Yuv(ChromaSampling::Cs420));
        assert_eq!(synthetic_samples(yuv420, 4, 2).unwrap().len(), 12);
        assert!(synthetic_samples(yuv420, 3, 2).is_err());
        assert!(synthetic_samples(yuv420, 4, 1).is_err());
        let yuv422 = PixelFormat::U8(ChannelLayout::Yuv(ChromaSampling::Cs422));
        assert!(synthetic_samples(yuv422, 4, 2).is_err());
    }

    #[test]
    fn a_changed_checksum_names_the_output_and_both_values() {
        assert_eq!(check_checksum("CUDA", 7, 7), Ok(()));
        assert_eq!(
            check_checksum("CUDA", 0xab, 0xcd),
            Err("CUDA checksum changed: expected 00000000000000cd, got 00000000000000ab".into())
        );
    }

    #[test]
    fn a_benchmark_file_that_cannot_be_read_names_its_path() {
        let path = std::path::Path::new("/nonexistent/jxr-benchmark-input.jxr");
        let Err(OracleError::Io {
            operation,
            path: reported,
            ..
        }) = BenchmarkFile::load(path)
        else {
            panic!("a missing file is an I/O error");
        };
        assert_eq!(
            (operation, reported.as_path()),
            ("read benchmark input", path)
        );
    }

    #[test]
    fn timing_summary_uses_sorted_nearest_rank_percentiles() {
        let samples = [
            Duration::from_millis(5),
            Duration::from_millis(1),
            Duration::from_millis(4),
            Duration::from_millis(2),
            Duration::from_millis(3),
        ];

        let summary = summarize_timings(&samples).unwrap();

        assert_eq!(summary.minimum, Duration::from_millis(1));
        assert_eq!(summary.median, Duration::from_millis(3));
        assert_eq!(summary.p95, Duration::from_millis(5));
        assert_eq!(summary.mean, Duration::from_millis(3));
    }

    #[test]
    fn timing_summary_rejects_an_empty_sample() {
        assert!(summarize_timings(&[]).is_none());
    }

    #[test]
    fn checksum_is_stable_and_order_sensitive() {
        assert_eq!(checksum_bytes(b"JPEG XR"), 0x3c90_4314_7685_6f13);
        assert_ne!(checksum_bytes(b"JPEG XR"), checksum_bytes(b"RX GEPJ"));
    }

    #[test]
    fn typed_sample_checksum_matches_native_byte_representation() {
        let values = [0x1234_u16, 0xabcd];
        let bytes = values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();

        assert_eq!(
            checksum_samples(&DecodedSamples::U16(values.to_vec())),
            checksum_bytes(&bytes)
        );
    }

    #[test]
    fn native_batch_checksum_selects_one_dense_image() {
        let samples = jxr::CpuBatchSamples::U16(vec![1, 2, 3, 4]);
        assert_eq!(
            checksum_cpu_batch_image(&samples, 1, 2),
            Some(checksum_samples(&DecodedSamples::U16(vec![3, 4])))
        );
        assert_eq!(checksum_cpu_batch_image(&samples, 2, 2), None);
    }
}
