//! Four-lane wrapping forms of the inverse core transform and 4-by-4 overlap
//! postfilter, applied in place to horizontally adjacent plane blocks.
//!
//! Each kernel repeats the operation order of the checked `jxr-math` oracle with
//! wrapping arithmetic. A group of four blocks uses a kernel only when every
//! input magnitude is below that kernel's bound, which the tests derive by
//! interval arithmetic so that no intermediate can leave `i32`; wrapping and
//! checked results are then identical. Other groups run the checked scalar
//! operators, so every value and error matches the oracle.

use fearless_simd::{Level, Simd, SimdBase, dispatch, i32x4};
use jxr_math::tables::INVERSE_PERMUTATION;
use rayon::prelude::*;

use super::{ReconstructionError, transform::inverse_core_transform};

/// Exclusive input magnitude bound for [`inverse_core_transform_lanes`].
const TRANSFORM_BOUND: u32 = 1 << 25;
/// Exclusive input magnitude bound for [`inverse_overlap_filter_4x4_lanes`].
const OVERLAP_BOUND: u32 = 1 << 23;
/// Samples spanned by one group of four horizontally adjacent 4-by-4 blocks.
const GROUP_WIDTH: usize = 16;
/// Planes below 64 macroblocks of samples stay on the calling thread, matching
/// the threshold of the other parallel reconstruction stages.
pub(super) const PARALLEL_SAMPLES: usize = 64 * 256;

/// Runs `operation` over consecutive `band`-sized chunks, in parallel only for large planes.
pub(super) fn try_for_each_band(
    samples: &mut [i32],
    band: usize,
    operation: impl Fn(&mut [i32]) -> Result<(), ReconstructionError> + Sync + Send,
) -> Result<(), ReconstructionError> {
    if samples.len() < PARALLEL_SAMPLES {
        samples.chunks_mut(band).try_for_each(operation)
    } else {
        samples.par_chunks_mut(band).try_for_each(operation)
    }
}

/// Lane-wise integer operations used by the lifting kernels.
pub(super) trait Lane: Copy {
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn add_const(self, value: i32) -> Self;
    fn shr(self, bits: u32) -> Self;
    fn mul3(self) -> Self;
    fn neg(self) -> Self;
}

impl<S: Simd> Lane for i32x4<S> {
    #[inline]
    fn add(self, other: Self) -> Self {
        self + other
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        self - other
    }

    #[inline]
    fn add_const(self, value: i32) -> Self {
        self + value
    }

    #[inline]
    fn shr(self, bits: u32) -> Self {
        self >> bits
    }

    #[inline]
    fn mul3(self) -> Self {
        self * 3
    }

    #[inline]
    fn neg(self) -> Self {
        -self
    }
}

#[inline]
fn group<L: Lane>(values: &mut [L; 16], indices: [usize; 4], operation: impl FnOnce(&mut [L; 4])) {
    let mut lanes = indices.map(|index| values[index]);
    operation(&mut lanes);
    for (index, value) in indices.into_iter().zip(lanes) {
        values[index] = value;
    }
}

#[inline]
fn pair<L: Lane>(values: &mut [L; 16], indices: [usize; 2], operation: impl FnOnce(&mut [L; 2])) {
    let mut lanes = indices.map(|index| values[index]);
    operation(&mut lanes);
    for (index, value) in indices.into_iter().zip(lanes) {
        values[index] = value;
    }
}

/// Mirrors `jxr_math::transform::inverse_core_transform`.
#[inline]
pub(super) fn inverse_core_transform_lanes<L: Lane>(values: &mut [L; 16]) {
    let input = *values;
    for (source, destination) in INVERSE_PERMUTATION.into_iter().enumerate() {
        values[destination] = input[source];
    }
    group(values, [0, 1, 4, 5], |lanes| t2x2h(lanes, 1));
    group(values, [2, 3, 6, 7], inverse_todd);
    group(values, [8, 12, 9, 13], inverse_todd);
    group(values, [10, 11, 14, 15], inverse_todd_odd);
    for indices in [[0, 3, 12, 15], [5, 6, 9, 10], [1, 2, 13, 14], [4, 7, 8, 11]] {
        group(values, indices, |lanes| t2x2h(lanes, 0));
    }
}

/// Mirrors `jxr_math::overlap::inverse_overlap_filter_4x4`.
#[inline]
pub(super) fn inverse_overlap_filter_4x4_lanes<L: Lane>(values: &mut [L; 16]) {
    for indices in [[0, 3, 12, 15], [1, 2, 13, 14], [4, 7, 8, 11], [5, 6, 9, 10]] {
        group(values, indices, |lanes| t2x2h(lanes, 0));
    }
    for indices in [[13, 12], [9, 8], [7, 3], [6, 2]] {
        pair(values, indices, inverse_rotate);
    }
    group(values, [10, 11, 14, 15], inverse_todd_odd_post);
    for indices in [[0, 15], [1, 14], [4, 11], [5, 10]] {
        pair(values, indices, inverse_scale);
    }
    for indices in [[0, 3, 12, 15], [1, 2, 13, 14], [4, 7, 8, 11], [5, 6, 9, 10]] {
        group(values, indices, inverse_hadamard_post);
    }
}

#[inline]
fn t2x2h<L: Lane>(v: &mut [L; 4], rounding: i32) {
    v[0] = v[0].add(v[3]);
    v[1] = v[1].sub(v[2]);
    let first = v[0].sub(v[1]).add_const(rounding).shr(1);
    let second = v[2];
    v[2] = first.sub(v[3]);
    v[3] = first.sub(second);
    v[0] = v[0].sub(v[3]);
    v[1] = v[1].add(v[2]);
}

#[inline]
fn inverse_todd<L: Lane>(v: &mut [L; 4]) {
    v[1] = v[1].add(v[3]);
    v[0] = v[0].sub(v[2]);
    v[3] = v[3].sub(v[1].shr(1));
    v[2] = v[2].add(v[0].add_const(1).shr(1));
    v[0] = v[0].sub(v[1].mul3().add_const(4).shr(3));
    v[1] = v[1].add(v[0].mul3().add_const(4).shr(3));
    v[2] = v[2].sub(v[3].mul3().add_const(4).shr(3));
    v[3] = v[3].add(v[2].mul3().add_const(4).shr(3));
    v[2] = v[2].sub(v[1].add_const(1).shr(1));
    v[3] = v[0].add_const(1).shr(1).sub(v[3]);
    v[1] = v[1].add(v[2]);
    v[0] = v[0].sub(v[3]);
}

#[inline]
fn inverse_todd_odd<L: Lane>(v: &mut [L; 4]) {
    v[3] = v[3].add(v[0]);
    v[2] = v[2].sub(v[1]);
    let first = v[3].shr(1);
    let second = v[2].shr(1);
    v[0] = v[0].sub(first);
    v[1] = v[1].add(second);
    v[0] = v[0].sub(v[1].mul3().add_const(3).shr(3));
    v[1] = v[1].add(v[0].mul3().add_const(3).shr(2));
    v[0] = v[0].sub(v[1].mul3().add_const(4).shr(3));
    v[1] = v[1].sub(second);
    v[0] = v[0].add(first);
    v[2] = v[2].add(v[1]);
    v[3] = v[3].sub(v[0]);
    v[1] = v[1].neg();
    v[2] = v[2].neg();
}

#[inline]
fn inverse_rotate<L: Lane>(v: &mut [L; 2]) {
    v[0] = v[0].sub(v[1].add_const(1).shr(1));
    v[1] = v[1].add(v[0].add_const(1).shr(1));
}

#[inline]
fn inverse_scale<L: Lane>(v: &mut [L; 2]) {
    v[0] = v[0].add(v[1]);
    v[1] = v[0].shr(1).sub(v[1]);
    v[0] = v[0].add(v[1].mul3().shr(3));
    v[1] = v[1].add(v[0].mul3().shr(4));
    v[1] = v[1].add(v[0].shr(7));
    v[1] = v[1].sub(v[0].shr(10));
}

#[inline]
fn inverse_hadamard_post<L: Lane>(v: &mut [L; 4]) {
    v[1] = v[1].sub(v[2]);
    v[0] = v[0].add(v[3].mul3().add_const(4).shr(3));
    v[3] = v[3].sub(v[1].shr(1));
    v[2] = v[0].sub(v[1]).shr(1).sub(v[2]);
    v.swap(2, 3);
    v[0] = v[0].sub(v[3]);
    v[1] = v[1].add(v[2]);
}

#[inline]
fn inverse_todd_odd_post<L: Lane>(v: &mut [L; 4]) {
    v[3] = v[3].add(v[0]);
    v[2] = v[2].sub(v[1]);
    let first = v[3].shr(1);
    let second = v[2].shr(1);
    v[0] = v[0].sub(first);
    v[1] = v[1].add(second);
    v[0] = v[0].sub(v[1].mul3().add_const(6).shr(3));
    v[1] = v[1].add(v[0].mul3().add_const(2).shr(2));
    v[0] = v[0].sub(v[1].mul3().add_const(4).shr(3));
    v[1] = v[1].sub(second);
    v[0] = v[0].add(first);
    v[2] = v[2].add(v[1]);
    v[3] = v[3].sub(v[0]);
}

/// Loads four horizontally adjacent 4-by-4 blocks with one lane per block and
/// returns whether every input magnitude is below `bound`.
#[inline]
fn load_group<S: Simd>(simd: S, rows: [&[i32]; 4], x: usize, bound: u32) -> ([i32x4<S>; 16], bool) {
    let mut values = [i32x4::splat(simd, 0); 16];
    let mut magnitude = i32x4::splat(simd, 0);
    for (row, samples) in rows.into_iter().enumerate() {
        let samples: &[i32; GROUP_WIDTH] = samples[x..x + GROUP_WIDTH]
            .try_into()
            .expect("group spans sixteen samples");
        // Four-way deinterleaving places column `c` of every block in vector `c`.
        let columns = simd.load_four_interleaved_i32x4(samples);
        for (column, vector) in columns.into_iter().enumerate() {
            magnitude |= vector.abs();
            values[row * 4 + column] = vector;
        }
    }
    // A power-of-two bound is exceeded exactly when the OR reaches it; the
    // wrapped magnitude of `i32::MIN` sets the sign bit.
    let bound = i32::try_from(bound).expect("bound fits i32");
    let within = magnitude.reduce_min() >= 0 && magnitude.reduce_max() < bound;
    (values, within)
}

#[inline]
fn store_group<S: Simd>(simd: S, rows: &mut [&mut [i32]; 4], x: usize, values: &[i32x4<S>; 16]) {
    for (row, samples) in rows.iter_mut().enumerate() {
        let samples: &mut [i32; GROUP_WIDTH] = (&mut samples[x..x + GROUP_WIDTH])
            .try_into()
            .expect("group spans sixteen samples");
        let columns = core::array::from_fn(|column| values[row * 4 + column]);
        simd.store_four_interleaved_i32x4(columns, samples);
    }
}

/// Applies a checked scalar operator to one 4-by-4 window at column `x`.
fn checked_window(
    rows: &mut [&mut [i32]; 4],
    x: usize,
    operation: impl Fn(&mut [i32; 16]) -> Result<(), ReconstructionError>,
) -> Result<(), ReconstructionError> {
    let mut values = [0; 16];
    for (row, samples) in rows.iter().enumerate() {
        values[row * 4..row * 4 + 4].copy_from_slice(&samples[x..x + 4]);
    }
    operation(&mut values)?;
    for (row, samples) in rows.iter_mut().enumerate() {
        samples[x..x + 4].copy_from_slice(&values[row * 4..row * 4 + 4]);
    }
    Ok(())
}

/// Splits one four-row band into its rows.
fn band_rows(band: &mut [i32], stride: usize) -> [&mut [i32]; 4] {
    let (first, rest) = band.split_at_mut(stride);
    let (second, rest) = rest.split_at_mut(stride);
    let (third, fourth) = rest.split_at_mut(stride);
    [first, second, third, &mut fourth[..stride]]
}

/// Applies one 4-by-4 operator to windows `left, left + 4, ...` below `right`
/// in one band, grouping four windows per kernel call.
#[inline]
#[expect(clippy::too_many_arguments, reason = "one band of one plane operation")]
fn filter_band<S: Simd>(
    simd: S,
    band: &mut [i32],
    stride: usize,
    left: usize,
    right: usize,
    bound: u32,
    kernel: impl Fn(&mut [i32x4<S>; 16]),
    checked: impl Fn(&mut [i32; 16]) -> Result<(), ReconstructionError>,
) -> Result<(), ReconstructionError> {
    let mut rows = band_rows(band, stride);
    let mut x = left;
    while x + GROUP_WIDTH <= right {
        let (mut values, within) =
            load_group(simd, [&*rows[0], &*rows[1], &*rows[2], &*rows[3]], x, bound);
        if within {
            kernel(&mut values);
            store_group(simd, &mut rows, x, &values);
        } else {
            for window in 0..4 {
                checked_window(&mut rows, x + window * 4, &checked)?;
            }
        }
        x += GROUP_WIDTH;
    }
    while x + 4 <= right {
        checked_window(&mut rows, x, &checked)?;
        x += 4;
    }
    Ok(())
}

/// Inverse-transforms every 4-by-4 block of a plane in place, in parallel by block row.
pub(super) fn transform_plane(
    level: Level,
    samples: &mut [i32],
    width: usize,
) -> Result<(), ReconstructionError> {
    if width == 0 {
        return Ok(());
    }
    if !samples.len().is_multiple_of(width * 4) {
        return Err(ReconstructionError::InvalidPlaneGeometry(
            "plane height is not a whole number of blocks",
        ));
    }
    try_for_each_band(samples, width * 4, |band| {
        dispatch!(level, simd => filter_band(
            simd,
            band,
            width,
            0,
            width,
            TRANSFORM_BOUND,
            inverse_core_transform_lanes,
            inverse_core_transform,
        ))
    })
}

/// Applies the 4-by-4 overlap postfilter to every interior window of the
/// regions bounded by `columns` within rows `top..bottom`, in parallel by window row.
///
/// Region boundaries must be multiples of four. Windows start two samples
/// inside each region and are mutually disjoint, so their order does not
/// affect the result.
pub(super) fn filter_interior_windows(
    level: Level,
    samples: &mut [i32],
    stride: usize,
    columns: &[usize],
    rows: [usize; 2],
    checked: impl Fn(&mut [i32; 16]) -> Result<(), ReconstructionError> + Sync,
) -> Result<(), ReconstructionError> {
    let [top, bottom] = rows;
    if bottom < top + 8 {
        return Ok(());
    }
    // Window rows start at `top + 2` and the final two rows are edge samples.
    try_for_each_band(
        &mut samples[(top + 2) * stride..(bottom - 2) * stride],
        stride * 4,
        |band| {
            columns.windows(2).try_for_each(|region| {
                dispatch!(level, simd => filter_band(
                    simd,
                    band,
                    stride,
                    region[0] + 2,
                    region[1] - 2,
                    OVERLAP_BOUND,
                    inverse_overlap_filter_4x4_lanes,
                    &checked,
                ))
            })
        },
    )
}

#[cfg(test)]
mod tests {
    use jxr_math::{overlap::inverse_overlap_filter_4x4, transform};

    use super::*;

    /// Conservative magnitude of a value and of every intermediate it depends on.
    #[derive(Clone, Copy)]
    struct Bound {
        value: u64,
        peak: u64,
    }

    impl Bound {
        fn derive(self, other: Self, value: u64) -> Self {
            Self {
                value,
                peak: self.peak.max(other.peak).max(value),
            }
        }
    }

    impl Lane for Bound {
        fn add(self, other: Self) -> Self {
            self.derive(other, self.value + other.value)
        }
        fn sub(self, other: Self) -> Self {
            self.derive(other, self.value + other.value)
        }
        fn add_const(self, value: i32) -> Self {
            self.derive(self, self.value + u64::from(value.unsigned_abs()))
        }
        // Flooring a negative value can raise its magnitude by one.
        fn shr(self, bits: u32) -> Self {
            self.derive(self, (self.value >> bits) + 1)
        }
        fn mul3(self) -> Self {
            self.derive(self, self.value * 3)
        }
        fn neg(self) -> Self {
            self
        }
    }

    fn peak(kernel: impl Fn(&mut [Bound; 16]), bound: u32) -> u64 {
        let input = u64::from(bound - 1);
        let mut values = [Bound {
            value: input,
            peak: input,
        }; 16];
        kernel(&mut values);
        values.iter().map(|value| value.peak).max().unwrap()
    }

    #[test]
    fn kernel_bounds_keep_every_intermediate_in_i32() {
        let limit = i32::MAX.unsigned_abs().into();
        assert!(peak(inverse_core_transform_lanes, TRANSFORM_BOUND) <= limit);
        assert!(peak(inverse_overlap_filter_4x4_lanes, OVERLAP_BOUND) <= limit);
    }

    fn random_values(state: &mut u64, bound: u32) -> [i32; 16] {
        core::array::from_fn(|_| {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            let magnitude = i32::try_from((*state >> 8) % u64::from(bound)).unwrap();
            if *state & 1 == 0 {
                magnitude
            } else {
                -magnitude
            }
        })
    }

    fn assert_lanes_match<S: Simd>(
        simd: S,
        kernel: impl Fn(&mut [i32x4<S>; 16]),
        oracle: impl Fn(&mut [i32; 16]),
        bound: u32,
    ) {
        let mut state = 0x9e37_79b9_7f4a_7c15;
        for round in 0..4096 {
            // Alternate full-range and small inputs to cover both regimes.
            let limit = if round % 2 == 0 { bound } else { 4096 };
            let blocks: [[i32; 16]; 4] = core::array::from_fn(|_| random_values(&mut state, limit));
            let mut lanes: [i32x4<S>; 16] = core::array::from_fn(|index| {
                i32x4::from_slice(
                    simd,
                    &core::array::from_fn::<_, 4, _>(|lane| blocks[lane][index]),
                )
            });
            kernel(&mut lanes);
            for (lane, block) in blocks.iter().enumerate() {
                let mut expected = *block;
                oracle(&mut expected);
                let actual: [i32; 16] = core::array::from_fn(|index| lanes[index][lane]);
                assert_eq!(actual, expected, "round {round}, lane {lane}");
            }
        }
    }

    fn for_each_level(test: impl Fn(Level)) {
        test(Level::new());
        test(Level::baseline());
    }

    #[test]
    fn transform_lanes_match_checked_oracle() {
        for_each_level(|level| {
            dispatch!(level, simd => assert_lanes_match(
                simd,
                inverse_core_transform_lanes,
                |values| transform::inverse_core_transform(values).unwrap(),
                TRANSFORM_BOUND,
            ));
        });
    }

    #[test]
    fn overlap_lanes_match_checked_oracle() {
        for_each_level(|level| {
            dispatch!(level, simd => assert_lanes_match(
                simd,
                inverse_overlap_filter_4x4_lanes,
                |values| inverse_overlap_filter_4x4(values).unwrap(),
                OVERLAP_BOUND,
            ));
        });
    }

    #[test]
    fn plane_transform_matches_blockwise_oracle_and_rejects_overflow() {
        let width = 40;
        let mut state = 0x2545_f491_4f6c_dd1d;
        let mut samples: Vec<i32> = (0..width * 8)
            .flat_map(|_| random_values(&mut state, 1 << 12))
            .take(width * 8)
            .collect();
        // One out-of-bound block exercises the checked path without overflowing.
        samples[width * 4 + 20] = 1 << 25;
        let mut expected = samples.clone();
        for block_y in 0..2 {
            for block_x in 0..width / 4 {
                let mut values = [0; 16];
                for row in 0..4 {
                    let start = (block_y * 4 + row) * width + block_x * 4;
                    values[row * 4..row * 4 + 4].copy_from_slice(&expected[start..start + 4]);
                }
                transform::inverse_core_transform(&mut values).unwrap();
                for row in 0..4 {
                    let start = (block_y * 4 + row) * width + block_x * 4;
                    expected[start..start + 4].copy_from_slice(&values[row * 4..row * 4 + 4]);
                }
            }
        }
        transform_plane(Level::new(), &mut samples, width).unwrap();
        assert_eq!(samples, expected);

        let mut overflowing = vec![i32::MAX; width * 4];
        assert_eq!(
            transform_plane(Level::new(), &mut overflowing, width),
            Err(ReconstructionError::ArithmeticOverflow(
                "inverse core transform"
            ))
        );
    }
}
