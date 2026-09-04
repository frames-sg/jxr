// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Duration;

use jxr::{CpuBatchSamples, DecodedSamples};

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

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

    use jxr::DecodedSamples;

    use super::{checksum_bytes, checksum_cpu_batch_image, checksum_samples, summarize_timings};

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
