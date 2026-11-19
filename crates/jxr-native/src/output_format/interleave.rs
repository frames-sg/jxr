//! Row-parallel interleaved integer stores for the common luma and RGB
//! layouts, and the loads that run them backwards for the encoder.
//!
//! The store kernels use wrapping arithmetic and accumulate an upper bound of
//! every input magnitude. Inputs below [`SAFE_MAGNITUDE`] cannot overflow any
//! checked step of the normative conversion, bias, rounding, or postscaling,
//! so wrapping and checked results are identical. Callers rerun the checked
//! scalar packer when a kernel declines, which keeps every error and value
//! bit-exact. A load reads stored integers, whose range bounds every value
//! it forms, so it is resolved only where wrapping is exact for all of them.

use jxr_core::{ChannelLayout, ColorFormat, PixelFormat};

use super::{
    OutputBitDepth,
    color::{FORWARD_YUV_BOUND, forward_yuv_lanes, inverse_yuv_lanes},
    packing::{FormatContext, layout},
    scaling::ChannelScale,
};

/// Exclusive input magnitude bound for which wrapping arithmetic is exact.
const SAFE_MAGNITUDE: u32 = 1 << 26;
/// Rows are grouped so that each parallel task stores at least this many pixels.
const PIXELS_PER_TASK: usize = 16 * 1024;

/// Integer storage written by the interleaved stores and read by the loads.
pub(crate) trait InterleavedSample: Copy + Default + Send + Sync + Into<i32> {
    /// The largest stored value.
    const MAXIMUM: i32;

    fn from_clamped(value: i32) -> Self;
}

impl InterleavedSample for u8 {
    const MAXIMUM: i32 = 0xff;

    #[inline]
    fn from_clamped(value: i32) -> Self {
        Self::try_from(value).unwrap_or_default()
    }
}

impl InterleavedSample for u16 {
    const MAXIMUM: i32 = 0xffff;

    #[inline]
    fn from_clamped(value: i32) -> Self {
        Self::try_from(value).unwrap_or_default()
    }
}

/// Bias, rounding, shift, postscale, and clip for one channel.
#[derive(Clone, Copy)]
struct IntegerScale {
    /// The bias in the units of the reconstructed sample.
    bias: i32,
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
            bias: shifted_bias,
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

    /// A bound on the magnitude of every value [`Self::unapply`] forms from
    /// a stored value of `0..=stored`, and of its shifted value.
    fn unapplied_magnitude(self, stored: i32) -> Option<u32> {
        let shifted = i64::from(stored >> self.post_shift) << self.shift.min(32);
        u32::try_from(shifted + i64::from(self.bias).abs()).ok()
    }

    /// [`ChannelScale::unapply`] with wrapping arithmetic: the same value
    /// for every stored value [`Self::unapplied_magnitude`] covers when it
    /// is within `i32`.
    #[inline]
    fn unapply(self, value: i32) -> i32 {
        ((value >> self.post_shift) << self.shift).wrapping_sub(self.bias)
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
    let store_task = |task: usize, rows: &mut [T]| {
        rows.chunks_exact_mut(row_length)
            .enumerate()
            .fold(0, |bits, (row, destination)| {
                bits | store_row(plan, shape, task * rows_per_task + row, destination)
            })
    };
    // Small images stay on the calling thread rather than entering the pool.
    let magnitude = context
        .executor
        .if_worth(plan.width * context.height >= PIXELS_PER_TASK * 4)
        .fold_chunks(
            output,
            rows_per_task * row_length,
            0,
            store_task,
            |left, right| left | right,
        );
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

impl Shape {
    /// The layout of `format`'s pixels whose colour channels convert to and
    /// from `internal` components through the interleaved kernels, with the
    /// largest value a channel of `bit_depth` stores in it.
    fn resolve(
        (internal, output): (ColorFormat, ColorFormat),
        format: PixelFormat,
        bit_depth: OutputBitDepth,
    ) -> Option<(Conversion, Self, i32)> {
        let maximum = match (bit_depth, format) {
            (OutputBitDepth::U8, PixelFormat::U8(_))
            | (OutputBitDepth::U10 | OutputBitDepth::U16 { .. }, PixelFormat::U16(_)) => {
                i32::from(bit_depth.maximum()?)
            }
            _ => return None,
        };
        let conversion = match (internal, output) {
            (ColorFormat::Luma, ColorFormat::Luma) => Conversion::Luma,
            (ColorFormat::Luma, ColorFormat::Rgb) => Conversion::LumaToRgb,
            (ColorFormat::Yuv(jxr_core::ChromaSampling::Cs444), ColorFormat::Rgb) => {
                Conversion::YuvToRgb
            }
            _ => return None,
        };
        let shape = match (conversion, layout(format)) {
            (Conversion::Luma, ChannelLayout::Luma) => Self::Luma,
            (Conversion::Luma, ChannelLayout::LumaAlpha) => Self::LumaAlpha,
            (Conversion::Luma, _) => return None,
            (_, ChannelLayout::Rgb) => Self::Rgb { reverse: false },
            (_, ChannelLayout::Bgr) => Self::Rgb { reverse: true },
            (_, ChannelLayout::Rgbx) => Self::RgbPadding { reverse: false },
            (_, ChannelLayout::Bgrx) => Self::RgbPadding { reverse: true },
            (_, ChannelLayout::Rgba) => Self::RgbAlpha { reverse: false },
            (_, ChannelLayout::Bgra) => Self::RgbAlpha { reverse: true },
            _ => return None,
        };
        Some((conversion, shape, maximum))
    }

    const fn has_alpha(self) -> bool {
        matches!(self, Self::LumaAlpha | Self::RgbAlpha { .. })
    }
}

fn plan(context: FormatContext<'_>) -> Option<(Plan<'_>, Shape)> {
    let request = context.request;
    let (conversion, shape, maximum) = Shape::resolve(
        (request.internal_color, request.output_color),
        request.pixel_format,
        request.bit_depth,
    )?;
    let has_alpha = shape.has_alpha();
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
            let [red, green, blue] = inverse_yuv_lanes([y, u, v]);
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

/// The decoder's interleaved store run backwards: reads rows of interleaved
/// integer pixels of the layouts and depths it stores into the coded
/// component rows and the alpha row, through the same layout and scale.
#[derive(Clone, Copy)]
pub(crate) struct InterleavedLoad {
    shape: Shape,
    color: IntegerScale,
    alpha: Option<IntegerScale>,
}

impl InterleavedLoad {
    /// The load of pixels of `format` stored as `T`, at `bit_depth`, whose
    /// colour channels, of `colors.1`, are coded as `colors.0` components
    /// at their own resolution through `color`, and whose alpha channel,
    /// where it has one, through `alpha`; `None` where the store has no
    /// kernel for the format, or wrapping arithmetic is not exact for every
    /// stored value.
    pub(crate) fn resolve<T: InterleavedSample>(
        colors: (ColorFormat, ColorFormat),
        format: PixelFormat,
        bit_depth: OutputBitDepth,
        color: &ChannelScale,
        alpha: &ChannelScale,
    ) -> Option<Self> {
        let (conversion, shape, maximum) = Shape::resolve(colors, format, bit_depth)?;
        // A luma image stored as RGB has no inverse.
        if conversion == Conversion::LumaToRgb {
            return None;
        }
        let exact = |scale: &ChannelScale| {
            let scale = IntegerScale::resolve(scale, maximum)?;
            let magnitude = scale.unapplied_magnitude(T::MAXIMUM)?;
            (magnitude < FORWARD_YUV_BOUND).then_some(scale)
        };
        Some(Self {
            shape,
            color: exact(color)?,
            alpha: if shape.has_alpha() {
                Some(exact(alpha)?)
            } else {
                None
            },
        })
    }

    /// Whether each pixel has an alpha channel, which [`Self::load_row`]
    /// reads into an alpha row.
    pub(crate) const fn has_alpha(self) -> bool {
        self.shape.has_alpha()
    }

    /// Reads one row of `source` pixels into the component rows `colour`,
    /// every component's or, for a reading of alpha alone, none, and into
    /// `alpha`, where the reading takes it: the reconstructed samples a
    /// decoder scales back into the stored values, converted to the coded
    /// components. Rows longer than the pixels are left as they are beyond
    /// them.
    pub(crate) fn load_row<T: InterleavedSample>(
        self,
        source: &[T],
        colour: &mut [&mut [i32]],
        alpha: Option<&mut [i32]>,
    ) {
        match (self.shape, colour) {
            (Shape::Luma, [luma]) => load_channel::<T, 1>(source, 0, self.color, luma),
            (Shape::LumaAlpha, [luma]) => load_channel::<T, 2>(source, 0, self.color, luma),
            (Shape::Rgb { reverse }, [first, second, third]) => {
                load_rgb::<T, 3>(source, reverse, self.color, [first, second, third]);
            }
            (
                Shape::RgbPadding { reverse } | Shape::RgbAlpha { reverse },
                [first, second, third],
            ) => load_rgb::<T, 4>(source, reverse, self.color, [first, second, third]),
            _ => {}
        }
        if let (Some(row), Some(scale)) = (alpha, self.alpha) {
            match self.shape {
                Shape::LumaAlpha => load_channel::<T, 2>(source, 1, scale, row),
                _ => load_channel::<T, 4>(source, 3, scale, row),
            }
        }
    }
}

/// Reads channel `channel` of pixels of `CHANNELS` samples into `row`.
#[inline]
fn load_channel<T: InterleavedSample, const CHANNELS: usize>(
    source: &[T],
    channel: usize,
    scale: IntegerScale,
    row: &mut [i32],
) {
    for (sample, pixel) in row.iter_mut().zip(source.as_chunks::<CHANNELS>().0) {
        *sample = scale.unapply(pixel[channel].into());
    }
}

/// Reads the colour channels of pixels of `CHANNELS` samples, red first or,
/// when `reverse`, blue first, into YUV component rows.
#[inline]
fn load_rgb<T: InterleavedSample, const CHANNELS: usize>(
    source: &[T],
    reverse: bool,
    scale: IntegerScale,
    [y, u, v]: [&mut [i32]; 3],
) {
    let (first, last) = if reverse { (2, 0) } else { (0, 2) };
    let rows = y.iter_mut().zip(u.iter_mut()).zip(v.iter_mut());
    for (((y, u), v), pixel) in rows.zip(source.as_chunks::<CHANNELS>().0) {
        let red = scale.unapply(pixel[first].into());
        let green = scale.unapply(pixel[1].into());
        let blue = scale.unapply(pixel[last].into());
        [*y, *u, *v] = forward_yuv_lanes([red, green, blue]);
    }
}
