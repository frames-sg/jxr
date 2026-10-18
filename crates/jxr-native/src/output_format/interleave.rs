//! Row-parallel interleaved integer stores for the common luma and RGB layouts.
//!
//! The kernels use wrapping arithmetic and accumulate an upper bound of every
//! input magnitude. Inputs below [`SAFE_MAGNITUDE`] cannot overflow any checked
//! step of the normative conversion, bias, rounding, or postscaling, so wrapping
//! and checked results are identical. Callers rerun the checked scalar packer
//! when a kernel declines, which keeps every error and value bit-exact.

use jxr_core::{ChannelLayout, ColorFormat, PixelFormat};
use rayon::prelude::*;

use super::{
    OutputBitDepth,
    packing::{FormatContext, layout},
    scaling::ChannelScale,
};

/// Exclusive input magnitude bound for which wrapping arithmetic is exact.
const SAFE_MAGNITUDE: u32 = 1 << 26;
/// Rows are grouped so that each parallel task stores at least this many pixels.
const PIXELS_PER_TASK: usize = 16 * 1024;

/// Integer output storage written by the interleaved kernels.
pub(super) trait InterleavedSample: Copy + Default + Send + Sync {
    fn from_clamped(value: i32) -> Self;
}

impl InterleavedSample for u8 {
    #[inline]
    fn from_clamped(value: i32) -> Self {
        Self::try_from(value).unwrap_or_default()
    }
}

impl InterleavedSample for u16 {
    #[inline]
    fn from_clamped(value: i32) -> Self {
        Self::try_from(value).unwrap_or_default()
    }
}

/// Bias, rounding, shift, postscale, and clip for one channel.
#[derive(Clone, Copy)]
struct IntegerScale {
    addition: i32,
    shift: u32,
    post_shift: u32,
    maximum: i32,
}

impl IntegerScale {
    fn resolve(scale: &ChannelScale, maximum: i32) -> Option<Self> {
        let ChannelScale::Integer {
            shifted_bias,
            rounding,
            shift,
            post_shift,
        } = *scale
        else {
            return None;
        };
        let scale = Self {
            addition: shifted_bias.checked_add(rounding)?,
            shift,
            post_shift: post_shift.unwrap_or(0),
            maximum,
        };
        // Bound every intermediate for the largest admitted input, including
        // the fourfold growth of the inverse YUV transform.
        let largest = i64::from(SAFE_MAGNITUDE) * 4 + 2 + i64::from(scale.addition.unsigned_abs());
        let shifted = (largest >> scale.shift.min(63)) + 1;
        (scale.shift < 32
            && scale.post_shift < 32
            && shifted << scale.post_shift <= i64::from(i32::MAX))
        .then_some(scale)
    }

    #[inline]
    fn apply<T: InterleavedSample>(self, sample: i32) -> T {
        let scaled = (sample.wrapping_add(self.addition) >> self.shift) << self.post_shift;
        T::from_clamped(scaled.clamp(0, self.maximum))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Conversion {
    Luma,
    LumaToRgb,
    YuvToRgb,
}

#[derive(Clone, Copy)]
struct Plan<'a> {
    conversion: Conversion,
    color: IntegerScale,
    alpha: Option<IntegerScale>,
    planes: [Rows<'a>; 3],
    alpha_rows: Option<Rows<'a>>,
    width: usize,
}

/// Cropped row access into one borrowed component plane.
#[derive(Clone, Copy)]
struct Rows<'a> {
    samples: &'a [i32],
    stride: usize,
    first: usize,
}

impl<'a> Rows<'a> {
    fn new(plane: super::ComponentPlane<'a>, crop_x: u32, crop_y: u32) -> Option<Self> {
        let x = usize::try_from(crop_x.checked_sub(plane.origin_x)?).ok()?;
        let y = usize::try_from(crop_y.checked_sub(plane.origin_y)?).ok()?;
        Some(Self {
            samples: plane.samples,
            stride: plane.stride,
            first: y.checked_mul(plane.stride)?.checked_add(x)?,
        })
    }

    #[inline]
    fn row(self, y: usize, width: usize) -> &'a [i32] {
        let start = self.first + y * self.stride;
        &self.samples[start..start + width]
    }
}

/// Stores the cropped image when its layout has an interleaved kernel.
///
/// Returns `false` without a meaningful partial result when the request is not
/// covered or an input sample is outside the exact wrapping range; the caller
/// must then run the checked scalar packer over the whole output.
pub(super) fn try_store<T: InterleavedSample>(
    context: FormatContext<'_>,
    output: &mut [T],
) -> bool {
    let Some((plan, shape)) = plan(context) else {
        return false;
    };
    let channels = shape.channels();
    if output.len() != plan.width * context.height * channels {
        return false;
    }
    if plan.width == 0 || context.height == 0 {
        return true;
    }
    let rows_per_task = (PIXELS_PER_TASK / plan.width).max(1);
    let row_length = plan.width * channels;
    let store_task = |(task, rows): (usize, &mut [T])| {
        rows.chunks_exact_mut(row_length)
            .enumerate()
            .fold(0, |bits, (row, destination)| {
                bits | store_row(plan, shape, task * rows_per_task + row, destination)
            })
    };
    // Small images stay on the calling thread rather than entering the pool.
    let magnitude = if plan.width * context.height < PIXELS_PER_TASK * 4 {
        output
            .chunks_mut(rows_per_task * row_length)
            .enumerate()
            .map(store_task)
            .fold(0, |left, right| left | right)
    } else {
        output
            .par_chunks_mut(rows_per_task * row_length)
            .enumerate()
            .map(store_task)
            .reduce(|| 0, |left, right| left | right)
    };
    magnitude < SAFE_MAGNITUDE
}

#[derive(Clone, Copy)]
enum Shape {
    Luma,
    LumaAlpha,
    Rgb { reverse: bool },
    RgbPadding { reverse: bool },
    RgbAlpha { reverse: bool },
}

impl Shape {
    const fn channels(self) -> usize {
        match self {
            Self::Luma => 1,
            Self::LumaAlpha => 2,
            Self::Rgb { .. } => 3,
            Self::RgbPadding { .. } | Self::RgbAlpha { .. } => 4,
        }
    }
}

fn plan(context: FormatContext<'_>) -> Option<(Plan<'_>, Shape)> {
    let request = context.request;
    let maximum = match (request.bit_depth, request.pixel_format) {
        (OutputBitDepth::U8, PixelFormat::U8(_)) => 255,
        (OutputBitDepth::U10, PixelFormat::U16(_)) => 1_023,
        (OutputBitDepth::U16 { .. }, PixelFormat::U16(_)) => 65_535,
        _ => return None,
    };
    let conversion = match (request.internal_color, request.output_color) {
        (ColorFormat::Luma, ColorFormat::Luma) => Conversion::Luma,
        (ColorFormat::Luma, ColorFormat::Rgb) => Conversion::LumaToRgb,
        (ColorFormat::Yuv(jxr_core::ChromaSampling::Cs444), ColorFormat::Rgb) => {
            Conversion::YuvToRgb
        }
        _ => return None,
    };
    let shape = match (conversion, layout(request.pixel_format)) {
        (Conversion::Luma, ChannelLayout::Luma) => Shape::Luma,
        (Conversion::Luma, ChannelLayout::LumaAlpha) => Shape::LumaAlpha,
        (Conversion::Luma, _) => return None,
        (_, ChannelLayout::Rgb) => Shape::Rgb { reverse: false },
        (_, ChannelLayout::Bgr) => Shape::Rgb { reverse: true },
        (_, ChannelLayout::Rgbx) => Shape::RgbPadding { reverse: false },
        (_, ChannelLayout::Bgrx) => Shape::RgbPadding { reverse: true },
        (_, ChannelLayout::Rgba) => Shape::RgbAlpha { reverse: false },
        (_, ChannelLayout::Bgra) => Shape::RgbAlpha { reverse: true },
        _ => return None,
    };
    let has_alpha = matches!(shape, Shape::LumaAlpha | Shape::RgbAlpha { .. });
    if has_alpha != context.alpha.is_some() || context.planes.is_empty() {
        return None;
    }
    let primaries = shape.channels() - usize::from(has_alpha);
    // Every color channel of these layouts resolves to one shared scale.
    let color = IntegerScale::resolve(&context.channel_scale(0, false), maximum)?;
    let alpha = if has_alpha {
        Some(IntegerScale::resolve(
            &context.channel_scale(primaries, true),
            maximum,
        )?)
    } else {
        None
    };
    let crop = request.crop;
    let first = Rows::new(context.planes[0], crop.x, crop.y)?;
    let mut planes = [first; 3];
    if conversion == Conversion::YuvToRgb {
        for (rows, &plane) in planes.iter_mut().zip(context.planes).skip(1) {
            *rows = Rows::new(plane, crop.x, crop.y)?;
        }
    }
    let alpha_rows = context
        .alpha
        .map(|plane| Rows::new(plane, crop.x, crop.y))
        .map_or(Some(None), |rows| rows.map(Some))?;
    Some((
        Plan {
            conversion,
            color,
            alpha,
            planes,
            alpha_rows,
            width: context.width,
        },
        shape,
    ))
}

/// Stores one output row and returns the OR of its input magnitudes.
fn store_row<T: InterleavedSample>(
    plan: Plan<'_>,
    shape: Shape,
    y: usize,
    output: &mut [T],
) -> u32 {
    let width = plan.width;
    let alpha = plan.alpha_rows.map(|rows| rows.row(y, width));
    match (plan.conversion, shape) {
        (Conversion::Luma, Shape::Luma) => {
            store_luma(plan.planes[0].row(y, width), plan.color, output)
        }
        (Conversion::Luma, Shape::LumaAlpha) => store_luma_alpha(
            plan.planes[0].row(y, width),
            alpha.unwrap_or_default(),
            plan.color,
            plan.alpha.unwrap_or(plan.color),
            output,
        ),
        (conversion, shape) => {
            let rows = [
                plan.planes[0].row(y, width),
                plan.planes[1].row(y, width),
                plan.planes[2].row(y, width),
            ];
            let gray = conversion == Conversion::LumaToRgb;
            match shape {
                Shape::Rgb { reverse } => {
                    store_rgb::<T, 3>(rows, gray, reverse, None, plan.color, output)
                }
                Shape::RgbPadding { reverse } => {
                    store_rgb::<T, 4>(rows, gray, reverse, None, plan.color, output)
                }
                Shape::RgbAlpha { reverse } => store_rgb::<T, 4>(
                    rows,
                    gray,
                    reverse,
                    alpha.zip(plan.alpha),
                    plan.color,
                    output,
                ),
                Shape::Luma | Shape::LumaAlpha => unreachable!("luma shapes use luma conversion"),
            }
        }
    }
}

#[inline]
fn store_luma<T: InterleavedSample>(source: &[i32], scale: IntegerScale, output: &mut [T]) -> u32 {
    let mut bits = 0;
    for (destination, &sample) in output.iter_mut().zip(source) {
        bits |= sample.unsigned_abs();
        *destination = scale.apply(sample);
    }
    bits
}

#[inline]
fn store_luma_alpha<T: InterleavedSample>(
    source: &[i32],
    alpha: &[i32],
    color: IntegerScale,
    alpha_scale: IntegerScale,
    output: &mut [T],
) -> u32 {
    let mut bits = 0;
    for ((destination, &sample), &opacity) in output
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(source)
        .zip(alpha)
    {
        bits |= sample.unsigned_abs() | opacity.unsigned_abs();
        destination[0] = color.apply(sample);
        destination[1] = alpha_scale.apply(opacity);
    }
    bits
}

/// Inverse T.832 YUV444 transform with wrapping arithmetic.
#[inline]
const fn inverse_yuv(y: i32, u: i32, v: i32) -> [i32; 3] {
    let temporary = u.wrapping_neg();
    let green = y.wrapping_sub(temporary >> 1);
    let red = temporary
        .wrapping_add(green)
        .wrapping_sub((v >> 1) + (v & 1));
    let blue = v.wrapping_add(red);
    [red, green, blue]
}

#[inline]
fn store_rgb<T: InterleavedSample, const CHANNELS: usize>(
    rows: [&[i32]; 3],
    gray: bool,
    reverse: bool,
    alpha: Option<(&[i32], IntegerScale)>,
    scale: IntegerScale,
    output: &mut [T],
) -> u32 {
    let mut bits = 0;
    let pixels = output.as_chunks_mut::<CHANNELS>().0.iter_mut();
    if gray {
        for (destination, &luma) in pixels.zip(rows[0]) {
            bits |= luma.unsigned_abs();
            let value = scale.apply(luma);
            destination[..3].fill(value);
            if CHANNELS == 4 {
                destination[3] = T::default();
            }
        }
    } else {
        for (((destination, &y), &u), &v) in pixels.zip(rows[0]).zip(rows[1]).zip(rows[2]) {
            bits |= y.unsigned_abs() | u.unsigned_abs() | v.unsigned_abs();
            let [red, green, blue] = inverse_yuv(y, u, v);
            let (first, last) = if reverse { (blue, red) } else { (red, blue) };
            destination[0] = scale.apply(first);
            destination[1] = scale.apply(green);
            destination[2] = scale.apply(last);
            if CHANNELS == 4 {
                destination[3] = T::default();
            }
        }
    }
    if let Some((alpha, alpha_scale)) = alpha {
        for (destination, &opacity) in output.as_chunks_mut::<CHANNELS>().0.iter_mut().zip(alpha) {
            bits |= opacity.unsigned_abs();
            destination[CHANNELS - 1] = alpha_scale.apply(opacity);
        }
    }
    bits
}
