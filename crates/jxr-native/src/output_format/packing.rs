//! Checked crop, clipping, interleaving, and packed output from T.832 clause 9.10.8.

use jxr_core::{ChannelLayout, ColorFormat, DecodedSamples, DecodedSamplesMut, PixelFormat};
use jxr_math::color::clamp_sample;

use super::{
    ComponentPlane, OutputBitDepth, OutputFormatError, OutputFormatRequest,
    color::convert,
    interleave,
    packed_color::{pack_rgb555_into, pack_rgb565_into, pack_rgb101010_into, pack_rgbe_into},
    premultiply::premultiply_output_mut,
    scaling::{ChannelScale, float16_bits, float32_bits, scale_integer_component},
    simd_pack::pack_u8_into,
    validate::validate_request,
};
use crate::executor::Executor;

/// Format reconstructed signed planes into typed host samples.
///
/// YUV input must already be 4:4:4. `alpha` is a reconstructed luma plane and
/// is formatted with the same depth, scaling flag, and crop as the primary image.
pub fn format_components(
    planes: &[ComponentPlane<'_>],
    alpha: Option<ComponentPlane<'_>>,
    request: OutputFormatRequest,
) -> Result<DecodedSamples, OutputFormatError> {
    // Checked before the output is allocated, as the store checks it.
    validate_request(planes, alpha, request)?;
    let (_, _, length) = output_extent(request)?;
    let mut samples = zeroed_samples(request.pixel_format, length);
    format_components_into_with_cpu(
        planes,
        alpha,
        request,
        crate::CpuCapabilities::detect(),
        borrow_samples(&mut samples),
    )?;
    Ok(samples)
}

/// Stores the output `request` describes of reconstructed signed planes in
/// `destination`, which must have its type and length, as
/// [`format_components`] does. Returns whether a vector kernel stored it.
pub(crate) fn format_components_into_with_cpu(
    planes: &[ComponentPlane<'_>],
    alpha: Option<ComponentPlane<'_>>,
    request: OutputFormatRequest,
    cpu: crate::CpuCapabilities,
    destination: DecodedSamplesMut<'_>,
) -> Result<bool, OutputFormatError> {
    let component_count = validate_request(planes, alpha, request)?;
    if !destination.matches_format(request.pixel_format) {
        return Err(OutputFormatError::UnsupportedCombination {
            combination: "direct destination type differs from the output format",
        });
    }
    let (width, height, length) = output_extent(request)?;
    if destination.len() != length {
        return Err(OutputFormatError::UnsupportedCombination {
            combination: "direct destination length differs from the output contract",
        });
    }
    let context = FormatContext {
        planes,
        alpha,
        request,
        component_count,
        width,
        height,
        executor: cpu.executor(),
    };
    fill_destination(context, cpu, destination)
}

/// The width and height of the output `request` describes, and the
/// elements of storage it takes: a byte for every eight pixels of a row of
/// one-bit samples, a word for every pixel of a packed format, and a sample
/// for every channel of every pixel otherwise.
fn output_extent(request: OutputFormatRequest) -> Result<(usize, usize, usize), OutputFormatError> {
    let width = usize::try_from(request.crop.width)
        .map_err(|_| OutputFormatError::arithmetic("converting crop width"))?;
    let height = usize::try_from(request.crop.height)
        .map_err(|_| OutputFormatError::arithmetic("converting crop height"))?;
    let length = request
        .pixel_format
        .row_elements(request.crop.width)
        .and_then(|row| row.checked_mul(height))
        .ok_or_else(|| OutputFormatError::arithmetic("calculating output element count"))?;
    Ok((width, height, length))
}

/// Zeroed storage of `length` elements of the type `format` stores.
pub(crate) fn zeroed_samples(format: PixelFormat, length: usize) -> DecodedSamples {
    match format {
        PixelFormat::BitPacked(_) => DecodedSamples::BitPacked(vec![0; length]),
        PixelFormat::U8(_) => DecodedSamples::U8(vec![0; length]),
        PixelFormat::U16(_) => DecodedSamples::U16(vec![0; length]),
        PixelFormat::I16(_) => DecodedSamples::I16(vec![0; length]),
        PixelFormat::I32(_) => DecodedSamples::I32(vec![0; length]),
        PixelFormat::F16(_) => DecodedSamples::F16(vec![0; length]),
        PixelFormat::F32(_) => DecodedSamples::F32(vec![0.0; length]),
        PixelFormat::Rgb555 => DecodedSamples::Rgb555(vec![0; length]),
        PixelFormat::Rgb565 => DecodedSamples::Rgb565(vec![0; length]),
        PixelFormat::Rgb101010 => DecodedSamples::Rgb101010(vec![0; length]),
        PixelFormat::Rgbe => DecodedSamples::Rgbe(vec![0; length]),
    }
}

/// Borrows owned storage as a destination.
pub(crate) fn borrow_samples(samples: &mut DecodedSamples) -> DecodedSamplesMut<'_> {
    match samples {
        DecodedSamples::BitPacked(values) => DecodedSamplesMut::BitPacked(values),
        DecodedSamples::U8(values) => DecodedSamplesMut::U8(values),
        DecodedSamples::U16(values) => DecodedSamplesMut::U16(values),
        DecodedSamples::I16(values) => DecodedSamplesMut::I16(values),
        DecodedSamples::I32(values) => DecodedSamplesMut::I32(values),
        DecodedSamples::F16(values) => DecodedSamplesMut::F16(values),
        DecodedSamples::F32(values) => DecodedSamplesMut::F32(values),
        DecodedSamples::Rgb555(values) => DecodedSamplesMut::Rgb555(values),
        DecodedSamples::Rgb565(values) => DecodedSamplesMut::Rgb565(values),
        DecodedSamples::Rgb101010(values) => DecodedSamplesMut::Rgb101010(values),
        DecodedSamples::Rgbe(values) => DecodedSamplesMut::Rgbe(values),
    }
}

/// Splits a destination into its first `mid` elements and the rest.
pub(crate) fn split_samples(
    samples: DecodedSamplesMut<'_>,
    mid: usize,
) -> (DecodedSamplesMut<'_>, DecodedSamplesMut<'_>) {
    macro_rules! split {
        ($variant:ident, $values:expr) => {{
            let (head, tail) = $values.split_at_mut(mid);
            (
                DecodedSamplesMut::$variant(head),
                DecodedSamplesMut::$variant(tail),
            )
        }};
    }
    match samples {
        DecodedSamplesMut::BitPacked(values) => split!(BitPacked, values),
        DecodedSamplesMut::U8(values) => split!(U8, values),
        DecodedSamplesMut::U16(values) => split!(U16, values),
        DecodedSamplesMut::I16(values) => split!(I16, values),
        DecodedSamplesMut::I32(values) => split!(I32, values),
        DecodedSamplesMut::F16(values) => split!(F16, values),
        DecodedSamplesMut::F32(values) => split!(F32, values),
        DecodedSamplesMut::Rgb555(values) => split!(Rgb555, values),
        DecodedSamplesMut::Rgb565(values) => split!(Rgb565, values),
        DecodedSamplesMut::Rgb101010(values) => split!(Rgb101010, values),
        DecodedSamplesMut::Rgbe(values) => split!(Rgbe, values),
    }
}

/// Stores the output `context` describes into `destination`, which has its
/// type and length, and premultiplies it if the request asks.
fn fill_destination(
    context: FormatContext<'_>,
    cpu: crate::CpuCapabilities,
    mut destination: DecodedSamplesMut<'_>,
) -> Result<bool, OutputFormatError> {
    let used_simd = fill_direct_destination(context, cpu, &mut destination)?;
    if context.request.premultiply_alpha {
        premultiply_output_mut(destination, context.request.pixel_format)?;
    }
    Ok(used_simd)
}

/// Whether formatting the whole output of `request` stores it with a vector
/// kernel, as the decode report states it.
///
/// Only the plain eight-bit plane store reports itself: a luma image without
/// alpha, and the planes of a native planar YUV image.
pub(crate) fn stores_with_vectors(
    request: OutputFormatRequest,
    has_alpha: bool,
    cpu: crate::CpuCapabilities,
) -> bool {
    if request.bit_depth != OutputBitDepth::U8
        || !matches!(request.pixel_format, PixelFormat::U8(_))
    {
        return false;
    }
    let vectors = |crop: crate::reconstruct::CropWindow| {
        super::simd_pack::packs_with_vectors(
            cpu.level(),
            [
                usize::try_from(crop.width).unwrap_or(0),
                usize::try_from(crop.height).unwrap_or(0),
            ],
        )
    };
    if let Ok(planes) = super::planar::planar_planes(request, has_alpha) {
        return planes.iter().any(|plane| vectors(plane.crop));
    }
    is_luma_plane(request, has_alpha) && vectors(request.crop)
}

/// Whether an eight-bit output is a luma plane stored as it is, without
/// alpha or colour conversion: the plane store [`pack_u8_into`] takes.
fn is_luma_plane(request: OutputFormatRequest, has_alpha: bool) -> bool {
    request.internal_color == ColorFormat::Luma
        && request.output_color == ColorFormat::Luma
        && !has_alpha
        && request.pixel_format == PixelFormat::U8(ChannelLayout::Luma)
}

fn fill_direct_destination(
    context: FormatContext<'_>,
    cpu: crate::CpuCapabilities,
    destination: &mut DecodedSamplesMut<'_>,
) -> Result<bool, OutputFormatError> {
    match destination {
        DecodedSamplesMut::BitPacked(output) => {
            fill_bits(context, output)?;
            Ok(false)
        }
        DecodedSamplesMut::U8(output) => fill_u8(context, cpu, output),
        DecodedSamplesMut::U16(output) => {
            fill_u16(context, output)?;
            Ok(false)
        }
        DecodedSamplesMut::I16(output) => {
            fill_i16(context, output)?;
            Ok(false)
        }
        DecodedSamplesMut::I32(output) => {
            let scaling = context.channel_scales();
            fill_ordered(context, output, |sample, component, alpha| {
                scaling.apply(sample, component, alpha)
            })?;
            Ok(false)
        }
        DecodedSamplesMut::F16(output) => {
            fill_f16(context, output)?;
            Ok(false)
        }
        DecodedSamplesMut::F32(output) => {
            fill_f32(context, output)?;
            Ok(false)
        }
        DecodedSamplesMut::Rgb555(output) => {
            pack_rgb555_into(context, output)?;
            Ok(false)
        }
        DecodedSamplesMut::Rgb565(output) => {
            pack_rgb565_into(context, output)?;
            Ok(false)
        }
        DecodedSamplesMut::Rgb101010(output) => {
            pack_rgb101010_into(context, output)?;
            Ok(false)
        }
        DecodedSamplesMut::Rgbe(output) => {
            pack_rgbe_into(context, output)?;
            Ok(false)
        }
    }
}

fn fill_u16(context: FormatContext<'_>, output: &mut [u16]) -> Result<(), OutputFormatError> {
    // A 16-bit store is validated to be of a 10- or 16-bit depth.
    let maximum = i32::from(context.request.bit_depth.maximum().unwrap_or(u16::MAX));
    if interleave::try_store(context, output) {
        return Ok(());
    }
    let scaling = context.channel_scales();
    fill_ordered(context, output, |sample, component, alpha| {
        let scaled = scaling.apply(sample, component, alpha)?;
        Ok(u16::try_from(clamp_sample(scaled, 0, maximum)).expect("sample is clipped to u16"))
    })
}

fn fill_i16(context: FormatContext<'_>, output: &mut [i16]) -> Result<(), OutputFormatError> {
    let scaling = context.channel_scales();
    fill_ordered(context, output, |sample, component, alpha| {
        let scaled = scaling.apply(sample, component, alpha)?;
        Ok(i16::try_from(clamp_sample(scaled, -32_768, 32_767)).expect("sample is clipped to i16"))
    })
}

fn fill_f16(context: FormatContext<'_>, output: &mut [u16]) -> Result<(), OutputFormatError> {
    fill_ordered(context, output, |sample, component, alpha| {
        if context.is_padding(component, alpha) {
            return Ok(0);
        }
        let scaled = if alpha {
            context.alpha_format().scaled
        } else {
            context.request.scaled
        };
        float16_bits(sample, scaled)
    })
}

fn fill_f32(context: FormatContext<'_>, output: &mut [f32]) -> Result<(), OutputFormatError> {
    fill_ordered(context, output, |sample, component, alpha| {
        if context.is_padding(component, alpha) {
            return Ok(0.0);
        }
        let format = if alpha {
            context.alpha_format()
        } else {
            super::AlphaFormatRequest {
                bit_depth: context.request.bit_depth,
                scaled: context.request.scaled,
            }
        };
        let OutputBitDepth::F32 {
            mantissa_length,
            exponent_bias,
        } = format.bit_depth
        else {
            return Err(OutputFormatError::UnsupportedCombination {
                combination: "alpha and primary floating-point depth",
            });
        };
        Ok(f32::from_bits(float32_bits(
            sample,
            format.scaled,
            mantissa_length,
            exponent_bias,
        )?))
    })
}

/// Stores every sample of the output in order, as `format` makes it of
/// the sample, its component and whether it is alpha.
///
/// A row of samples is gathered and colour-converted at a time by
/// [`ordered_row`], which every output type shares; only the typed store of
/// each row is particular to the type. An error is the first in output
/// order, whether gathering or `format` met it.
fn fill_ordered<T>(
    context: FormatContext<'_>,
    output: &mut [T],
    mut format: impl FnMut(i32, usize, bool) -> Result<T, OutputFormatError>,
) -> Result<(), OutputFormatError> {
    if context.width == 0 || context.height == 0 {
        return Ok(());
    }
    let channels = ordered_channels(context);
    let row_length = context.width * channels.len();
    let mut row = vec![0; row_length];
    for (y, destination) in output.chunks_exact_mut(row_length).enumerate() {
        let gathered = ordered_row(context, y, channels.len(), &mut row);
        let complete = gathered
            .as_ref()
            .map_or_else(|(samples, _)| *samples, |()| row_length);
        for (index, (destination, &sample)) in destination[..complete]
            .iter_mut()
            .zip(&row[..complete])
            .enumerate()
        {
            let (component, alpha) = channels[index % channels.len()];
            *destination = format(sample, component, alpha)?;
        }
        gathered.map_err(|(_, error)| error)?;
    }
    Ok(())
}

fn fill_bits(context: FormatContext<'_>, output: &mut [u8]) -> Result<(), OutputFormatError> {
    output.fill(0);
    let row_bytes = context.width.div_ceil(8);
    for y in 0..context.height {
        for x in 0..context.width {
            let sample = context.color_components(x, y)?[0];
            let scaled = scale_integer_component(
                sample,
                0,
                ColorFormat::Luma,
                context.request.bit_depth,
                context.request.scaled,
            )?;
            let clipped = clamp_sample(scaled, 0, 1);
            let bit = match context.request.bit_depth {
                OutputBitDepth::Bit1White => clipped,
                OutputBitDepth::Bit1Black => 1 - clipped,
                _ => unreachable!("bit packer selected only for 1-bit output"),
            };
            let (byte, shift) = super::bit_position(x);
            output[y * row_bytes + byte] |= u8::try_from(bit).expect("bit is zero or one") << shift;
        }
    }
    Ok(())
}

/// Stores an eight-bit output, which the request's validation has made of
/// eight-bit depth, into `output`, which has its length: a luma plane
/// through the plane store, others through the interleaved store or, where
/// that does not cover them, in order.
fn fill_u8(
    context: FormatContext<'_>,
    cpu: crate::CpuCapabilities,
    output: &mut [u8],
) -> Result<bool, OutputFormatError> {
    let FormatContext {
        planes,
        request,
        width,
        height,
        ..
    } = context;
    if is_luma_plane(request, context.alpha.is_some()) {
        let plane = planes[0];
        let start_x = usize::try_from(request.crop.x - plane.origin_x)
            .map_err(|_| OutputFormatError::arithmetic("converting SIMD luma crop x"))?;
        let start_y = usize::try_from(request.crop.y - plane.origin_y)
            .map_err(|_| OutputFormatError::arithmetic("converting SIMD luma crop y"))?;
        return pack_u8_into(
            cpu.level(),
            plane.samples,
            plane.stride,
            [start_x, start_y],
            [width, height],
            request.scaled,
            output,
        );
    }
    if !interleave::try_store(context, output) {
        let scaling = context.channel_scales();
        fill_ordered(context, output, |sample, component, alpha| {
            let scaled = scaling.apply(sample, component, alpha)?;
            Ok(u8::try_from(clamp_sample(scaled, 0, 255)).expect("sample is clipped to u8"))
        })?;
    }
    // The caller premultiplies, as it does for every other storage type.
    Ok(false)
}

#[derive(Clone, Copy)]
pub(super) struct FormatContext<'a> {
    pub(super) planes: &'a [ComponentPlane<'a>],
    pub(super) alpha: Option<ComponentPlane<'a>>,
    pub(super) request: OutputFormatRequest,
    pub(super) component_count: usize,
    pub(super) width: usize,
    pub(super) height: usize,
    /// Where the rows of a large output are stored.
    pub(super) executor: Executor,
}

impl FormatContext<'_> {
    fn alpha_format(self) -> super::AlphaFormatRequest {
        self.request
            .alpha_format
            .unwrap_or(super::AlphaFormatRequest {
                bit_depth: self.request.bit_depth,
                scaled: self.request.scaled,
            })
    }

    pub(super) fn channel_scale(self, component: usize, alpha: bool) -> ChannelScale {
        if self.is_padding(component, alpha) {
            return ChannelScale::Padding;
        }
        if alpha {
            let format = self.alpha_format();
            return ChannelScale::alpha(format.bit_depth, format.scaled);
        }
        ChannelScale::resolve(
            component,
            self.request.output_color,
            self.request.bit_depth,
            self.request.scaled,
        )
    }

    fn is_padding(self, component: usize, alpha: bool) -> bool {
        !alpha
            && matches!(
                layout(self.request.pixel_format),
                ChannelLayout::Rgbx | ChannelLayout::Bgrx
            )
            && component == 3
    }

    pub(super) fn color_components(
        self,
        x: usize,
        y: usize,
    ) -> Result<[i32; 4], OutputFormatError> {
        let source_x = usize::try_from(self.request.crop.x)
            .map_err(|_| OutputFormatError::arithmetic("converting crop x"))?
            + x;
        let source_y = usize::try_from(self.request.crop.y)
            .map_err(|_| OutputFormatError::arithmetic("converting crop y"))?
            + y;
        self.color_components_at(source_x, source_y, self.color_shape())
    }

    /// Visits every cropped pixel's converted color components in raster order.
    pub(super) fn for_each_color(
        self,
        mut visit: impl FnMut([i32; 4]) -> Result<(), OutputFormatError>,
    ) -> Result<(), OutputFormatError> {
        if self.width == 0 || self.height == 0 {
            return Ok(());
        }
        let crop_x = usize::try_from(self.request.crop.x)
            .map_err(|_| OutputFormatError::arithmetic("converting crop x"))?;
        let crop_y = usize::try_from(self.request.crop.y)
            .map_err(|_| OutputFormatError::arithmetic("converting crop y"))?;
        let shape = self.color_shape();
        for source_y in crop_y..crop_y + self.height {
            for source_x in crop_x..crop_x + self.width {
                visit(self.color_components_at(source_x, source_y, shape)?)?;
            }
        }
        Ok(())
    }

    /// Converts one pixel at a crop-adjusted source position.
    #[inline]
    fn color_components_at(
        self,
        source_x: usize,
        source_y: usize,
        shape: ColorShape,
    ) -> Result<[i32; 4], OutputFormatError> {
        let mut input = [0; 4];
        for (component, plane) in self.planes.iter().take(4).enumerate() {
            input[component] = plane.sample(source_x, source_y);
        }
        if shape.direct {
            return Ok(input);
        }
        convert(
            self.request.internal_color,
            self.request.output_color,
            &input,
            shape.packed_rgb,
            self.request.red_blue_not_swapped,
        )
    }

    /// Resolves the per-image color-conversion decisions.
    fn color_shape(self) -> ColorShape {
        ColorShape {
            direct: is_direct(self.request.internal_color, self.request.output_color),
            packed_rgb: matches!(
                self.request.bit_depth,
                OutputBitDepth::Rgb555 | OutputBitDepth::Rgb565 | OutputBitDepth::Rgb101010
            ),
        }
    }

    /// Resolves the per-image layout decisions made for every ordered pixel.
    fn pixel_shape(self) -> PixelShape {
        let layout = layout(self.request.pixel_format);
        PixelShape {
            color: self.color_shape(),
            reverse_primaries: matches!(
                layout,
                ChannelLayout::Bgr | ChannelLayout::Bgrx | ChannelLayout::Bgra
            ),
            primary_count: match layout {
                ChannelLayout::Luma => 1,
                ChannelLayout::Yuv(_) | ChannelLayout::Rgb | ChannelLayout::Bgr => 3,
                ChannelLayout::Rgbx
                | ChannelLayout::Bgrx
                | ChannelLayout::Cmyk
                | ChannelLayout::Cmyka => 4,
                ChannelLayout::NComponent(_) | ChannelLayout::NComponentAlpha(_) => {
                    self.component_count
                }
                ChannelLayout::LumaAlpha
                | ChannelLayout::Yuva(_)
                | ChannelLayout::Rgba
                | ChannelLayout::Bgra => layout.channel_count() as usize - 1,
            },
        }
    }

    #[inline]
    fn ordered_components_at(
        self,
        source_x: usize,
        source_y: usize,
        shape: PixelShape,
    ) -> Result<([i32; 5], usize), OutputFormatError> {
        let converted = self.color_components_at(source_x, source_y, shape.color)?;
        let primary = if shape.reverse_primaries {
            [converted[2], converted[1], converted[0], 0]
        } else {
            converted
        };
        // Only the first `primary_count` entries (plus alpha) are consumed.
        let mut ordered = [primary[0], primary[1], primary[2], primary[3], 0];
        match self.alpha {
            Some(alpha) => {
                ordered[shape.primary_count] = alpha.sample(source_x, source_y);
                Ok((ordered, shape.primary_count + 1))
            }
            None => Ok((ordered, shape.primary_count)),
        }
    }
}

impl<'a> FormatContext<'a> {
    /// Resolves every channel's integer scaling once per image.
    fn channel_scales(self) -> ChannelScales<'a> {
        // Ordered pixels carry at most four primaries plus alpha; N-component
        // output carries one entry per plane plus alpha.
        let count = self.planes.len().max(4) + 1;
        ChannelScales {
            primary: (0..count)
                .map(|component| self.channel_scale(component, false))
                .collect(),
            alpha: (0..count)
                .map(|component| self.channel_scale(component, true))
                .collect(),
            context: self,
        }
    }
}

/// Color-conversion decisions shared by every pixel of one output.
#[derive(Clone, Copy)]
struct ColorShape {
    direct: bool,
    packed_rgb: bool,
}

/// Layout decisions shared by every pixel of one ordered output.
#[derive(Clone, Copy)]
struct PixelShape {
    color: ColorShape,
    reverse_primaries: bool,
    primary_count: usize,
}

/// Per-image channel scaling resolved before the pixel loop.
struct ChannelScales<'a> {
    primary: Vec<ChannelScale>,
    alpha: Vec<ChannelScale>,
    context: FormatContext<'a>,
}

impl ChannelScales<'_> {
    #[inline]
    fn apply(&self, sample: i32, component: usize, alpha: bool) -> Result<i32, OutputFormatError> {
        let table = if alpha { &self.alpha } else { &self.primary };
        match table.get(component) {
            Some(scale) => scale.apply(sample),
            None => self.context.channel_scale(component, alpha).apply(sample),
        }
    }
}

/// The component each sample of an ordered output pixel stores, in order,
/// and whether it is alpha: the primaries in the layout's order, or every
/// plane of an N-component image, then alpha if the output has it.
fn ordered_channels(context: FormatContext<'_>) -> Vec<(usize, bool)> {
    let primaries = if matches!(context.request.output_color, ColorFormat::NComponent(_)) {
        context.planes.len()
    } else {
        context.pixel_shape().primary_count
    };
    let mut channels: Vec<_> = (0..primaries).map(|component| (component, false)).collect();
    if context.alpha.is_some() {
        channels.push((primaries, true));
    }
    channels
}

/// Gathers row `y` of the output into `samples`, `count` samples to a
/// pixel in the order the output stores them: colour-converted, ordered
/// and unscaled. On an error, returns how many samples were gathered before
/// the pixel that failed.
#[inline(never)]
fn ordered_row(
    context: FormatContext<'_>,
    y: usize,
    count: usize,
    samples: &mut [i32],
) -> Result<(), (usize, OutputFormatError)> {
    let crop = |value: u32, what| {
        usize::try_from(value).map_err(|_| (0, OutputFormatError::arithmetic(what)))
    };
    let source_x = crop(context.request.crop.x, "converting crop x")?;
    let source_y = crop(context.request.crop.y, "converting crop y")? + y;
    let pixels = samples.chunks_exact_mut(count).enumerate();
    if matches!(context.request.output_color, ColorFormat::NComponent(_)) {
        for (x, pixel) in pixels {
            for (sample, plane) in pixel.iter_mut().zip(context.planes) {
                *sample = plane.sample(source_x + x, source_y);
            }
            if let Some(alpha) = context.alpha {
                pixel[count - 1] = alpha.sample(source_x + x, source_y);
            }
        }
        return Ok(());
    }
    let shape = context.pixel_shape();
    for (x, pixel) in pixels {
        let (ordered, _) = context
            .ordered_components_at(source_x + x, source_y, shape)
            .map_err(|error| (x * count, error))?;
        pixel.copy_from_slice(&ordered[..count]);
    }
    Ok(())
}

/// Whether the internal components are the output channels themselves, so
/// that no colour conversion runs between them in either direction.
pub(crate) const fn is_direct(internal: ColorFormat, output: ColorFormat) -> bool {
    matches!((internal, output), (ColorFormat::Luma, ColorFormat::Luma))
        || matches!((internal, output), (ColorFormat::Yuv(a), ColorFormat::Yuv(b)) if a as u8 == b as u8)
        || matches!((internal, output), (ColorFormat::NComponent(a), ColorFormat::NComponent(b)) if a == b)
}

/// The channels of each pixel of `format`: those of a packed format are
/// RGB.
pub(super) const fn layout(format: PixelFormat) -> ChannelLayout {
    match format.channel_layout() {
        Some(layout) => layout,
        None => ChannelLayout::Rgb,
    }
}
