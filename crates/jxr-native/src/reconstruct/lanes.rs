//! Four-lane wrapping forms of the core transform and 4-by-4 overlap filter,
//! in both directions, applied in place to horizontally adjacent plane blocks.
//!
//! Decoding and encoding run the same plane traversal, grouping, threading,
//! and SIMD dispatch; a [`BlockKernel`] supplies only the operator.
//!
//! Each kernel repeats the operation order of the checked `jxr-math` oracle with
//! wrapping arithmetic. A group of four blocks uses a kernel only when every
//! input magnitude is below that kernel's bound, which the tests derive by
//! interval arithmetic so that no intermediate can leave `i32`; wrapping and
//! checked results are then identical. Other groups run the checked scalar
//! operators, so every value and error matches the oracle.

#![expect(
    clippy::inline_always,
    reason = "a kernel's lifting steps must be inlined into it for its values to stay in \
              registers, on one-sample lanes as on vectors; LLVM keeps steps used twice \
              apart, and passes their values through memory"
)]

use core::sync::atomic::{AtomicBool, Ordering};

use fearless_simd::{Level, Simd, SimdBase, dispatch, i32x4};
use jxr_math::tables::INVERSE_PERMUTATION;

use jxr_math::{
    overlap::{forward_overlap_filter_4x4, inverse_overlap_filter_4x4},
    transform::forward_core_transform,
};

use super::{ReconstructionError, transform::inverse_core_transform};
use crate::executor::Executor;

/// Exclusive input magnitude bound for the core-transform lane kernels.
const TRANSFORM_BOUND: u32 = 1 << 25;
/// Exclusive input magnitude bound for the overlap-filter lane kernels.
const OVERLAP_BOUND: u32 = 1 << 23;

/// One 4-by-4 block operator: its lane form, the checked scalar oracle it
/// mirrors, and the input bound below which the two agree.
///
/// Groups whose inputs reach the bound run the oracle instead, so every
/// value and error is the oracle's.
pub(crate) trait BlockKernel {
    /// Exclusive magnitude bound keeping every lane intermediate in `i32`.
    const BOUND: u32;

    fn lanes<L: Lane>(values: &mut [L; 16]);

    fn checked(values: &mut [i32; 16]) -> Result<(), ReconstructionError>;
}

/// Inverse core transform of reconstruction.
pub(crate) struct InverseTransform;
/// Forward core transform of analysis.
pub(crate) struct ForwardTransform;
/// Overlap postfilter of reconstruction.
pub(crate) struct Postfilter;
/// Overlap prefilter of analysis.
pub(crate) struct Prefilter;

impl BlockKernel for InverseTransform {
    const BOUND: u32 = TRANSFORM_BOUND;

    #[inline]
    fn lanes<L: Lane>(values: &mut [L; 16]) {
        inverse_core_transform_lanes(values);
    }

    #[inline]
    fn checked(values: &mut [i32; 16]) -> Result<(), ReconstructionError> {
        inverse_core_transform(values)
    }
}

impl BlockKernel for ForwardTransform {
    const BOUND: u32 = TRANSFORM_BOUND;

    #[inline]
    fn lanes<L: Lane>(values: &mut [L; 16]) {
        forward_core_transform_lanes(values);
    }

    #[inline]
    fn checked(values: &mut [i32; 16]) -> Result<(), ReconstructionError> {
        forward_core_transform(values)
            .map_err(|_| ReconstructionError::ArithmeticOverflow("forward core transform"))
    }
}

impl BlockKernel for Postfilter {
    const BOUND: u32 = OVERLAP_BOUND;

    #[inline]
    fn lanes<L: Lane>(values: &mut [L; 16]) {
        inverse_overlap_filter_4x4_lanes(values);
    }

    #[inline]
    fn checked(values: &mut [i32; 16]) -> Result<(), ReconstructionError> {
        inverse_overlap_filter_4x4(values)
            .map_err(|_| ReconstructionError::ArithmeticOverflow("overlap postfilter"))
    }
}

impl BlockKernel for Prefilter {
    const BOUND: u32 = OVERLAP_BOUND;

    #[inline]
    fn lanes<L: Lane>(values: &mut [L; 16]) {
        forward_overlap_filter_4x4_lanes(values);
    }

    #[inline]
    fn checked(values: &mut [i32; 16]) -> Result<(), ReconstructionError> {
        forward_overlap_filter_4x4(values)
            .map_err(|_| ReconstructionError::ArithmeticOverflow("overlap prefilter"))
    }
}
/// Samples spanned by one group of four horizontally adjacent 4-by-4 blocks.
const GROUP_WIDTH: usize = 16;
/// Macroblocks from which the per-macroblock stages of reconstruction and
/// analysis share their work out; fewer stay on the calling thread.
pub(crate) const PARALLEL_MACROBLOCKS: usize = 64;
/// Samples from which the plane traversals share their work out: those of
/// [`PARALLEL_MACROBLOCKS`] full-resolution macroblocks.
pub(crate) const PARALLEL_SAMPLES: usize = PARALLEL_MACROBLOCKS * 256;

pub(crate) type SimdReporter = AtomicBool;

#[inline]
pub(crate) fn simd_used(reporter: &SimdReporter) -> bool {
    reporter.load(Ordering::Relaxed)
}

/// Runs `operation` over consecutive `band`-sized chunks, through `executor`
/// for planes large enough to share out.
///
/// A traversal is written once, for a strip of block rows; a whole plane is
/// the strip that covers it. A strip a decoder streams through is already
/// the unit one thread runs, and is given the calling thread.
fn try_for_each_band(
    executor: Executor,
    samples: &mut [i32],
    band: usize,
    operation: impl Fn(&mut [i32]) -> Result<(), ReconstructionError> + Sync + Send,
) -> Result<(), ReconstructionError> {
    executor
        .if_worth(samples.len() >= PARALLEL_SAMPLES)
        .try_for_each_chunk(samples, band, |_, band| operation(band))
}

/// Lane-wise integer operations used by the lifting kernels.
pub(crate) trait Lane: Copy {
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

/// One sample: the lane of a target without vectors, with the wrapping
/// arithmetic of a vector lane.
impl Lane for i32 {
    #[inline]
    fn add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        self.wrapping_sub(other)
    }

    #[inline]
    fn add_const(self, value: i32) -> Self {
        self.wrapping_add(value)
    }

    #[inline]
    fn shr(self, bits: u32) -> Self {
        self >> bits
    }

    #[inline]
    fn mul3(self) -> Self {
        self.wrapping_mul(3)
    }

    #[inline]
    fn neg(self) -> Self {
        self.wrapping_neg()
    }
}

/// Applies `operation` to the four elements `indices` of `values`, each
/// read and written by its own index so that every one keeps a place of
/// its own wherever the compiler does not unroll a loop over them.
#[inline(always)]
fn group<L: Lane>(
    values: &mut [L; 16],
    [a, b, c, d]: [usize; 4],
    operation: impl FnOnce(&mut [L; 4]),
) {
    let mut lanes = [values[a], values[b], values[c], values[d]];
    operation(&mut lanes);
    [values[a], values[b], values[c], values[d]] = lanes;
}

/// [`group`] for two elements.
#[inline(always)]
fn pair<L: Lane>(values: &mut [L; 16], [a, b]: [usize; 2], operation: impl FnOnce(&mut [L; 2])) {
    let mut lanes = [values[a], values[b]];
    operation(&mut lanes);
    [values[a], values[b]] = lanes;
}

/// Mirrors `jxr_math::transform::inverse_core_transform`.
#[inline(always)]
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

/// Mirrors `jxr_math::transform::forward_core_transform`.
#[inline(always)]
pub(super) fn forward_core_transform_lanes<L: Lane>(values: &mut [L; 16]) {
    for indices in [[4, 7, 8, 11], [1, 2, 13, 14], [5, 6, 9, 10], [0, 3, 12, 15]] {
        group(values, indices, |lanes| t2x2h(lanes, 0));
    }
    group(values, [10, 11, 14, 15], forward_todd_odd);
    group(values, [8, 12, 9, 13], forward_todd);
    group(values, [2, 3, 6, 7], forward_todd);
    group(values, [0, 1, 4, 5], |lanes| t2x2h(lanes, 1));
    let input = *values;
    for (destination, source) in INVERSE_PERMUTATION.into_iter().enumerate() {
        values[destination] = input[source];
    }
}

/// Mirrors `jxr_math::overlap::forward_overlap_filter_4x4`.
#[inline(always)]
pub(super) fn forward_overlap_filter_4x4_lanes<L: Lane>(values: &mut [L; 16]) {
    for indices in [[0, 3, 12, 15], [1, 2, 13, 14], [4, 7, 8, 11], [5, 6, 9, 10]] {
        group(values, indices, forward_hadamard_pre);
    }
    for indices in [[0, 15], [1, 14], [4, 11], [5, 10]] {
        pair(values, indices, forward_scale);
    }
    group(values, [10, 11, 14, 15], forward_todd_odd_pre);
    for indices in [[13, 12], [9, 8], [7, 3], [6, 2]] {
        pair(values, indices, forward_rotate);
    }
    for indices in [[0, 3, 12, 15], [1, 2, 13, 14], [4, 7, 8, 11], [5, 6, 9, 10]] {
        group(values, indices, |lanes| t2x2h(lanes, 0));
    }
}

/// Mirrors `jxr_math::overlap::inverse_overlap_filter_4x4`.
#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
fn forward_todd<L: Lane>(v: &mut [L; 4]) {
    v[0] = v[0].add(v[3]);
    v[1] = v[1].sub(v[2]);
    v[3] = v[0].add_const(1).shr(1).sub(v[3]);
    v[2] = v[2].add(v[1].add_const(1).shr(1));
    v[3] = v[3].sub(v[2].mul3().add_const(4).shr(3));
    v[2] = v[2].add(v[3].mul3().add_const(4).shr(3));
    v[1] = v[1].sub(v[0].mul3().add_const(4).shr(3));
    v[0] = v[0].add(v[1].mul3().add_const(4).shr(3));
    v[2] = v[2].sub(v[0].add_const(1).shr(1));
    v[3] = v[3].add(v[1].shr(1));
    v[0] = v[0].add(v[2]);
    v[1] = v[1].sub(v[3]);
}

#[inline(always)]
fn forward_todd_odd<L: Lane>(v: &mut [L; 4]) {
    v[1] = v[1].neg();
    v[2] = v[2].neg();
    v[3] = v[3].add(v[0]);
    v[2] = v[2].sub(v[1]);
    let first = v[3].shr(1);
    let second = v[2].shr(1);
    v[0] = v[0].sub(first);
    v[1] = v[1].add(second);
    v[0] = v[0].add(v[1].mul3().add_const(4).shr(3));
    v[1] = v[1].sub(v[0].mul3().add_const(3).shr(2));
    v[0] = v[0].add(v[1].mul3().add_const(3).shr(3));
    v[1] = v[1].sub(second);
    v[0] = v[0].add(first);
    v[2] = v[2].add(v[1]);
    v[3] = v[3].sub(v[0]);
}

#[inline(always)]
fn forward_rotate<L: Lane>(v: &mut [L; 2]) {
    v[1] = v[1].sub(v[0].add_const(1).shr(1));
    v[0] = v[0].add(v[1].add_const(1).shr(1));
}

#[inline(always)]
fn forward_scale<L: Lane>(v: &mut [L; 2]) {
    v[1] = v[1].add(v[0].shr(10));
    v[1] = v[1].sub(v[0].shr(7));
    v[1] = v[1].sub(v[0].mul3().shr(4));
    v[0] = v[0].sub(v[1].mul3().shr(3));
    v[1] = v[0].shr(1).sub(v[1]);
    v[0] = v[0].sub(v[1]);
}

#[inline(always)]
fn forward_hadamard_pre<L: Lane>(v: &mut [L; 4]) {
    v[1] = v[1].sub(v[2]);
    v[0] = v[0].add(v[3]);
    v.swap(2, 3);
    v[2] = v[0].sub(v[1]).shr(1).sub(v[2]);
    v[3] = v[3].add(v[1].shr(1));
    v[0] = v[0].sub(v[3].mul3().add_const(4).shr(3));
    v[1] = v[1].add(v[2]);
}

#[inline(always)]
fn forward_todd_odd_pre<L: Lane>(v: &mut [L; 4]) {
    v[3] = v[3].add(v[0]);
    v[2] = v[2].sub(v[1]);
    let first = v[3].shr(1);
    let second = v[2].shr(1);
    v[0] = v[0].sub(first);
    v[1] = v[1].add(second);
    v[0] = v[0].add(v[1].mul3().add_const(4).shr(3));
    v[1] = v[1].sub(v[0].mul3().add_const(2).shr(2));
    v[0] = v[0].add(v[1].mul3().add_const(6).shr(3));
    v[1] = v[1].sub(second);
    v[0] = v[0].add(first);
    v[2] = v[2].add(v[1]);
    v[3] = v[3].sub(v[0]);
}

#[inline(always)]
fn inverse_rotate<L: Lane>(v: &mut [L; 2]) {
    v[0] = v[0].sub(v[1].add_const(1).shr(1));
    v[1] = v[1].add(v[0].add_const(1).shr(1));
}

#[inline(always)]
fn inverse_scale<L: Lane>(v: &mut [L; 2]) {
    v[0] = v[0].add(v[1]);
    v[1] = v[0].shr(1).sub(v[1]);
    v[0] = v[0].add(v[1].mul3().shr(3));
    v[1] = v[1].add(v[0].mul3().shr(4));
    v[1] = v[1].add(v[0].shr(7));
    v[1] = v[1].sub(v[0].shr(10));
}

#[inline(always)]
fn inverse_hadamard_post<L: Lane>(v: &mut [L; 4]) {
    v[1] = v[1].sub(v[2]);
    v[0] = v[0].add(v[3].mul3().add_const(4).shr(3));
    v[3] = v[3].sub(v[1].shr(1));
    v[2] = v[0].sub(v[1]).shr(1).sub(v[2]);
    v.swap(2, 3);
    v[0] = v[0].sub(v[3]);
    v[1] = v[1].add(v[2]);
}

#[inline(always)]
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
///
/// The rows are written out rather than looped over: a vector placed in the
/// group by a loop index keeps the whole group in memory wherever the
/// compiler does not unroll the loop, as it does not for WebAssembly, whose
/// deinterleaving loads take many shuffles where NEON has one instruction.
/// With every vector in a fixed place the group stays in registers.
#[inline]
fn load_group<S: Simd>(simd: S, rows: [&[i32]; 4], x: usize, bound: u32) -> ([i32x4<S>; 16], bool) {
    let mut magnitude = i32x4::splat(simd, 0);
    // Four-way deinterleaving places column `c` of every block in vector `c`.
    let mut load = |samples: &[i32]| {
        let samples: &[i32; GROUP_WIDTH] = samples[x..x + GROUP_WIDTH]
            .try_into()
            .expect("group spans sixteen samples");
        let columns = simd.load_four_interleaved_i32x4(samples);
        for column in columns {
            magnitude |= column.abs();
        }
        columns
    };
    let [first, second, third, fourth] =
        [load(rows[0]), load(rows[1]), load(rows[2]), load(rows[3])];
    let values = [
        first[0], first[1], first[2], first[3], second[0], second[1], second[2], second[3],
        third[0], third[1], third[2], third[3], fourth[0], fourth[1], fourth[2], fourth[3],
    ];
    // A power-of-two bound is exceeded exactly when the OR reaches it; the
    // wrapped magnitude of `i32::MIN` sets the sign bit.
    let bound = i32::try_from(bound).expect("bound fits i32");
    let within = magnitude.reduce_min() >= 0 && magnitude.reduce_max() < bound;
    (values, within)
}

/// Stores a group [`load_group`] loaded, its rows written out as there.
#[inline]
fn store_group<S: Simd>(simd: S, rows: &mut [&mut [i32]; 4], x: usize, values: &[i32x4<S>; 16]) {
    let store = |samples: &mut [i32], columns: [i32x4<S>; 4]| {
        let samples: &mut [i32; GROUP_WIDTH] = (&mut samples[x..x + GROUP_WIDTH])
            .try_into()
            .expect("group spans sixteen samples");
        simd.store_four_interleaved_i32x4(columns, samples);
    };
    let [first, second, third, fourth] = rows;
    store(first, [values[0], values[1], values[2], values[3]]);
    store(second, [values[4], values[5], values[6], values[7]]);
    store(third, [values[8], values[9], values[10], values[11]]);
    store(fourth, [values[12], values[13], values[14], values[15]]);
}

/// Applies a scalar operator to one 4-by-4 window at column `x`.
#[inline(always)]
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
/// in one band, grouping four windows per kernel call. The windows no group
/// takes, and those of a group with an input at the kernel's bound, run the
/// checked oracle.
#[inline]
fn filter_band<S: Simd, K: BlockKernel>(
    simd: S,
    band: &mut [i32],
    stride: usize,
    left: usize,
    right: usize,
    used_simd: Option<&SimdReporter>,
) -> Result<(), ReconstructionError> {
    let mut rows = band_rows(band, stride);
    let mut x = left;
    let mut ran_vectors = false;
    while x + GROUP_WIDTH <= right {
        let (mut values, within) = load_group(
            simd,
            [&*rows[0], &*rows[1], &*rows[2], &*rows[3]],
            x,
            K::BOUND,
        );
        if within {
            K::lanes(&mut values);
            store_group(simd, &mut rows, x, &values);
            ran_vectors = true;
        } else {
            for window in 0..4 {
                checked_window(&mut rows, x + window * 4, K::checked)?;
            }
        }
        x += GROUP_WIDTH;
    }
    while x + 4 <= right {
        checked_window(&mut rows, x, K::checked)?;
        x += 4;
    }
    if ran_vectors
        && let Some(flag) = used_simd
        && !simd_used(flag)
    {
        flag.store(true, Ordering::Relaxed);
    }
    Ok(())
}

/// [`filter_band`] where there are no vectors: applies one 4-by-4 operator
/// to the windows `left, left + 4, ...` below `right` of a band's rows one
/// at a time, through the lane kernel on one-sample lanes where a window's
/// inputs are below the kernel's bound, and through the checked oracle
/// otherwise. Grouping them into lanes of emulated vectors would only move
/// their samples through memory, which the target's registers spare.
fn filter_windows<K: BlockKernel>(
    rows: &mut [&mut [i32]; 4],
    left: usize,
    right: usize,
) -> Result<(), ReconstructionError> {
    let mut x = left;
    while x + 4 <= right {
        checked_window(rows, x, apply_block::<K>)?;
        x += 4;
    }
    Ok(())
}

/// Applies `K` to one 4-by-4 block `values`: through the lane kernel on
/// one-sample lanes where its inputs are below the kernel's bound, and
/// through the checked oracle otherwise.
#[inline]
pub(super) fn apply_block<K: BlockKernel>(
    values: &mut [i32; 16],
) -> Result<(), ReconstructionError> {
    // Each value with its bits inverted when negative is its magnitude,
    // less one for a negative value; their OR is below the power-of-two
    // bound exactly when every magnitude is at most the bound, which the
    // kernels hold (see the bound test). `i32::MIN` inverts to `i32::MAX`.
    let magnitude = values
        .iter()
        .fold(0, |bits, &value| bits | (value ^ (value >> 31)));
    if magnitude.cast_unsigned() < K::BOUND {
        K::lanes(values);
        Ok(())
    } else {
        K::checked(values)
    }
}

/// Applies `K` to the windows `left, left + 4, ...` below `right` in one
/// band: on the vectors of `level`, or, on a target that has none, on
/// one-sample lanes. `level` is the same on every call on such a target,
/// which therefore compiles one of the two.
#[inline]
fn filter_band_at<K: BlockKernel>(
    level: Level,
    band: &mut [i32],
    stride: usize,
    left: usize,
    right: usize,
    used_simd: Option<&SimdReporter>,
) -> Result<(), ReconstructionError> {
    if level.is_fallback() {
        filter_windows::<K>(&mut band_rows(band, stride), left, right)
    } else {
        dispatch!(level, simd => filter_band::<_, K>(simd, band, stride, left, right, used_simd))
    }
}

/// Applies `K` to every 4-by-4 block of a strip of whole block rows in place.
///
/// The blocks of a plane are independent, so any run of its block rows is
/// transformed as the plane is, and a plane is the strip of all of them,
/// which `executor` may run in parallel by block row.
pub(crate) fn transform_rows<K: BlockKernel>(
    level: Level,
    samples: &mut [i32],
    width: usize,
    executor: Executor,
    used_simd: Option<&SimdReporter>,
) -> Result<(), ReconstructionError> {
    if width == 0 {
        return Ok(());
    }
    if !samples.len().is_multiple_of(width * 4) {
        return Err(ReconstructionError::InvalidPlaneGeometry(
            "plane height is not a whole number of blocks",
        ));
    }
    try_for_each_band(executor, samples, width * 4, |band| {
        filter_band_at::<K>(level, band, width, 0, width, used_simd)
    })
}

/// Applies `K` to every interior 4-by-4 window of the regions bounded by
/// `columns` in a strip of whole window rows, four sample rows each.
///
/// Region boundaries must be multiples of four. Windows start two samples
/// inside each region and are mutually disjoint, so their order does not
/// affect the result.
pub(crate) fn filter_window_rows<K: BlockKernel>(
    level: Level,
    samples: &mut [i32],
    stride: usize,
    columns: &[usize],
    executor: Executor,
    used_simd: Option<&SimdReporter>,
) -> Result<(), ReconstructionError> {
    try_for_each_band(executor, samples, stride * 4, |band| {
        columns.windows(2).try_for_each(|region| {
            filter_band_at::<K>(level, band, stride, region[0] + 2, region[1] - 2, used_simd)
        })
    })
}

#[cfg(test)]
mod tests {
    use jxr_math::transform;

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
        assert!(peak(InverseTransform::lanes, InverseTransform::BOUND) <= limit);
        assert!(peak(ForwardTransform::lanes, ForwardTransform::BOUND) <= limit);
        assert!(peak(Postfilter::lanes, Postfilter::BOUND) <= limit);
        assert!(peak(Prefilter::lanes, Prefilter::BOUND) <= limit);
        // Inputs of the bound's own magnitude, which the one-sample check
        // admits for negative values.
        assert!(peak(InverseTransform::lanes, InverseTransform::BOUND + 1) <= limit);
        assert!(peak(ForwardTransform::lanes, ForwardTransform::BOUND + 1) <= limit);
        assert!(peak(Postfilter::lanes, Postfilter::BOUND + 1) <= limit);
        assert!(peak(Prefilter::lanes, Prefilter::BOUND + 1) <= limit);
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

    fn assert_kernel_matches_oracle<K: BlockKernel>() {
        for_each_level(|level| {
            dispatch!(level, simd => assert_lanes_match(
                simd,
                K::lanes,
                |values| K::checked(values).unwrap(),
                K::BOUND,
            ));
        });
    }

    #[test]
    fn every_lane_kernel_matches_its_checked_oracle() {
        assert_kernel_matches_oracle::<InverseTransform>();
        assert_kernel_matches_oracle::<ForwardTransform>();
        assert_kernel_matches_oracle::<Postfilter>();
        assert_kernel_matches_oracle::<Prefilter>();
    }

    /// The one-sample instance of `K`'s kernel against its oracle.
    fn assert_one_sample_kernel_matches_oracle<K: BlockKernel>() {
        let mut state = 0x9e37_79b9_7f4a_7c15;
        for round in 0..16_384 {
            let limit = if round % 2 == 0 { K::BOUND } else { 4096 };
            let mut values = random_values(&mut state, limit);
            let mut expected = values;
            K::checked(&mut expected).unwrap();
            K::lanes::<i32>(&mut values);
            assert_eq!(values, expected, "round {round}");
        }
    }

    #[test]
    fn every_one_sample_kernel_matches_its_checked_oracle() {
        assert_one_sample_kernel_matches_oracle::<InverseTransform>();
        assert_one_sample_kernel_matches_oracle::<ForwardTransform>();
        assert_one_sample_kernel_matches_oracle::<Postfilter>();
        assert_one_sample_kernel_matches_oracle::<Prefilter>();
    }

    /// A band of windows filtered one at a time on one-sample lanes, as a
    /// target without vectors filters it, against the vectors of this one,
    /// with some windows at and beyond the kernel's bound.
    fn assert_one_sample_band_matches_vectors<K: BlockKernel>() {
        let stride = 70;
        let mut state = 0x2545_f491_4f6c_dd1d;
        for round in 0..64 {
            let mut band: Vec<i32> = (0..=stride * 4 / 16)
                .flat_map(|_| {
                    random_values(&mut state, if round % 2 == 0 { K::BOUND } else { 1 << 12 })
                })
                .take(stride * 4)
                .collect();
            let bound = i32::try_from(K::BOUND).unwrap();
            for (position, value) in [
                (round, bound),
                (stride + 20, -bound + 1),
                (2 * stride + 9, -bound),
                (3 * stride + 37, i32::MIN),
            ] {
                band[position % (stride * 4)] = value;
            }
            let (left, right) = (2 + round % 3 * 4, stride - 2 - round % 2 * 4);
            let mut windows = band.clone();
            let one_sample = filter_windows::<K>(&mut band_rows(&mut windows, stride), left, right);
            let vectors = dispatch!(Level::new(), simd => filter_band::<_, K>(
                simd,
                &mut band,
                stride,
                left,
                right,
                None,
            ));
            assert_eq!(one_sample, vectors, "round {round}");
            if vectors.is_ok() {
                assert_eq!(windows, band, "round {round}");
            }
        }
    }

    #[test]
    fn one_sample_bands_filter_as_vector_bands_do() {
        assert_one_sample_band_matches_vectors::<InverseTransform>();
        assert_one_sample_band_matches_vectors::<ForwardTransform>();
        assert_one_sample_band_matches_vectors::<Postfilter>();
        assert_one_sample_band_matches_vectors::<Prefilter>();
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
        transform_rows::<InverseTransform>(
            Level::new(),
            &mut samples,
            width,
            Executor::DEFAULT,
            None,
        )
        .unwrap();
        assert_eq!(samples, expected);

        let mut checked = vec![0; 16 * 4];
        checked[20] = 1 << 25;
        let used_simd = SimdReporter::new(false);
        transform_rows::<ForwardTransform>(
            Level::new(),
            &mut checked,
            16,
            Executor::Sequential,
            Some(&used_simd),
        )
        .unwrap();
        assert!(!simd_used(&used_simd));

        let mut overflowing = vec![i32::MAX; width * 4];
        assert_eq!(
            transform_rows::<InverseTransform>(
                Level::new(),
                &mut overflowing,
                width,
                Executor::DEFAULT,
                None,
            ),
            Err(ReconstructionError::ArithmeticOverflow(
                "inverse core transform"
            ))
        );
    }
}
