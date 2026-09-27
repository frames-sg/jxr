//! Checked crop, clipping, interleaving, and packed output from T.832 clause 9.10.8.

use jxr_core::{ChannelLayout, ColorFormat, DecodedSamples, DecodedSamplesMut, PixelFormat};
use jxr_math::color::clamp_sample;

use super::{
    ComponentPlane, OutputBitDepth, OutputFormatError, OutputFormatRequest,
    color::convert,
    packed_color::{
        pack_rgb555, pack_rgb555_into, pack_rgb565, pack_rgb565_into, pack_rgb101010,
        pack_rgb101010_into, pack_rgbe, pack_rgbe_into,
    },
    premultiply::{premultiply_output, premultiply_output_mut},
    scaling::{ChannelScale, float16_bits, float32_bits, scale_integer_component},
    simd_pack::{append_u8, pack_u8_into},
    validate::validate_request,
};

/// Format reconstructed signed planes into typed host samples.
///
/// YUV input must already be 4:4:4. `alpha` is a reconstructed luma plane and
/// is formatted with the same depth, scaling flag, and crop as the primary image.
pub fn format_components(
    planes: &[ComponentPlane<'_>],
    alpha: Option<ComponentPlane<'_>>,
    request: OutputFormatRequest,
) -> Result<DecodedSamples, OutputFormatError> {
    format_components_with_cpu(planes, alpha, request, crate::CpuCapabilities::detect())
        .map(|(samples, _)| samples)
}

pub(crate) fn format_components_with_cpu(
    planes: &[ComponentPlane<'_>],
    alpha: Option<ComponentPlane<'_>>,
    request: OutputFormatRequest,
    cpu: crate::CpuCapabilities,
) -> Result<(DecodedSamples, bool), OutputFormatError> {
    let component_count = validate_request(planes, alpha, request)?;
    let width = usize::try_from(request.crop.width)
        .map_err(|_| OutputFormatError::arithmetic("converting crop width"))?;
    let height = usize::try_from(request.crop.height)
        .map_err(|_| OutputFormatError::arithmetic("converting crop height"))?;
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| OutputFormatError::arithmetic("calculating output pixel count"))?;
    let channels = usize::from(request.pixel_format.channel_count());
    let elements = pixels
        .checked_mul(channels)
        .ok_or_else(|| OutputFormatError::arithmetic("calculating output element count"))?;
    let context = FormatContext {
        planes,
        alpha,
        request,
        component_count,
        width,
        height,
    };
    let (mut samples, used_simd) = if request.pixel_format == PixelFormat::Rgbe {
        (pack_rgbe(context, pixels)?, false)
    } else {
        match request.bit_depth {
            OutputBitDepth::U8 => pack_u8(context, elements, cpu)?,
            OutputBitDepth::Bit1White | OutputBitDepth::Bit1Black => (pack_bits(context)?, false),
            OutputBitDepth::U10 | OutputBitDepth::U16 { .. } => {
                (pack_u16(context, elements)?, false)
            }
            OutputBitDepth::I16 { .. } => (pack_i16(context, elements)?, false),
            OutputBitDepth::I32 { .. } => (pack_i32(context, elements)?, false),
            OutputBitDepth::F16 => (pack_f16(context, elements)?, false),
            OutputBitDepth::F32 { .. } => (pack_f32(context, elements)?, false),
            OutputBitDepth::Rgb555 => (pack_rgb555(context, pixels)?, false),
            OutputBitDepth::Rgb565 => (pack_rgb565(context, pixels)?, false),
            OutputBitDepth::Rgb101010 => (pack_rgb101010(context, pixels)?, false),
        }
    };
    if request.premultiply_alpha {
        premultiply_output(&mut samples, request.pixel_format)?;
    }
    Ok((samples, used_simd))
}

pub(crate) fn format_components_into_with_cpu(
    planes: &[ComponentPlane<'_>],
    alpha: Option<ComponentPlane<'_>>,
    request: OutputFormatRequest,
    cpu: crate::CpuCapabilities,
    mut destination: DecodedSamplesMut<'_>,
) -> Result<bool, OutputFormatError> {
    let component_count = validate_request(planes, alpha, request)?;
    if !destination.matches_format(request.pixel_format) {
        return Err(OutputFormatError::UnsupportedCombination {
            combination: "direct destination type differs from the output format",
        });
    }
    let width = usize::try_from(request.crop.width)
        .map_err(|_| OutputFormatError::arithmetic("converting crop width"))?;
    let height = usize::try_from(request.crop.height)
        .map_err(|_| OutputFormatError::arithmetic("converting crop height"))?;
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| OutputFormatError::arithmetic("calculating output pixel count"))?;
    let elements = pixels
        .checked_mul(usize::from(request.pixel_format.channel_count()))
        .ok_or_else(|| OutputFormatError::arithmetic("calculating output element count"))?;
    let expected = match request.pixel_format {
        PixelFormat::BitPacked(_) => width
            .div_ceil(8)
            .checked_mul(height)
            .ok_or_else(|| OutputFormatError::arithmetic("calculating bit-packed output size"))?,
        PixelFormat::Rgb555 | PixelFormat::Rgb565 | PixelFormat::Rgb101010 | PixelFormat::Rgbe => {
            pixels
        }
        _ => elements,
    };
    if destination.len() != expected {
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
    };
    let used_simd = fill_direct_destination(context, cpu, &mut destination)?;
    if request.premultiply_alpha {
        premultiply_output_mut(destination, request.pixel_format)?;
    }
    Ok(used_simd)
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
        DecodedSamplesMut::U8(output) => format_components_u8_into_with_cpu(
            context.planes,
            context.alpha,
            context.request,
            cpu,
            output,
        ),
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
    let maximum = if matches!(context.request.bit_depth, OutputBitDepth::U10) {
        1_023
    } else {
        65_535
    };
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

fn fill_ordered<T>(
    context: FormatContext<'_>,
    output: &mut [T],
    mut format: impl FnMut(i32, usize, bool) -> Result<T, OutputFormatError>,
) -> Result<(), OutputFormatError> {
    let mut index = 0;
    for_each_ordered(context, |sample, component, alpha| {
        output[index] = format(sample, component, alpha)?;
        index += 1;
        Ok(())
    })
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
            output[y * row_bytes + x / 8] |=
                u8::try_from(bit).expect("bit is zero or one") << (7 - x % 8);
        }
    }
    Ok(())
}

pub(crate) fn format_components_u8_into_with_cpu(
    planes: &[ComponentPlane<'_>],
    alpha: Option<ComponentPlane<'_>>,
    request: OutputFormatRequest,
    cpu: crate::CpuCapabilities,
    output: &mut [u8],
) -> Result<bool, OutputFormatError> {
    let component_count = validate_request(planes, alpha, request)?;
    if !matches!(request.pixel_format, PixelFormat::U8(_))
        || request.bit_depth != OutputBitDepth::U8
    {
        return Err(OutputFormatError::UnsupportedCombination {
            combination: "direct U8 output requires an eight-bit U8 pixel format",
        });
    }
    let width = usize::try_from(request.crop.width)
        .map_err(|_| OutputFormatError::arithmetic("converting crop width"))?;
    let height = usize::try_from(request.crop.height)
        .map_err(|_| OutputFormatError::arithmetic("converting crop height"))?;
    let expected = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(usize::from(request.pixel_format.channel_count())))
        .ok_or_else(|| OutputFormatError::arithmetic("calculating direct U8 output size"))?;
    if output.len() != expected {
        return Err(OutputFormatError::UnsupportedCombination {
            combination: "direct U8 destination length differs from the output contract",
        });
    }
    let context = FormatContext {
        planes,
        alpha,
        request,
        component_count,
        width,
        height,
    };
    if request.internal_color == ColorFormat::Luma
        && request.output_color == ColorFormat::Luma
        && alpha.is_none()
        && request.pixel_format == PixelFormat::U8(ChannelLayout::Luma)
    {
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
    let mut index = 0;
    let scaling = context.channel_scales();
    for_each_ordered(context, |sample, component, alpha| {
        let scaled = scaling.apply(sample, component, alpha)?;
        output[index] =
            u8::try_from(clamp_sample(scaled, 0, 255)).expect("sample is clipped to u8");
        index += 1;
        Ok(())
    })?;
    if request.premultiply_alpha {
        super::premultiply::premultiply_u8(
            output,
            usize::from(request.pixel_format.channel_count()),
        )?;
    }
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

    fn channel_scale(self, component: usize, alpha: bool) -> ChannelScale {
        if self.is_padding(component, alpha) {
            return ChannelScale::Padding;
        }
        let (color, depth, scaled) = if alpha {
            let format = self.alpha_format();
            (ColorFormat::Luma, format.bit_depth, format.scaled)
        } else {
            (
                self.request.output_color,
                self.request.bit_depth,
                self.request.scaled,
            )
        };
        ChannelScale::resolve(component, color, depth, scaled)
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

fn pack_bits(context: FormatContext<'_>) -> Result<DecodedSamples, OutputFormatError> {
    let row_bytes = context.width.div_ceil(8);
    let length = row_bytes
        .checked_mul(context.height)
        .ok_or_else(|| OutputFormatError::arithmetic("calculating bit-packed output size"))?;
    let mut output = vec![0_u8; length];
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
            output[y * row_bytes + x / 8] |=
                u8::try_from(bit).expect("bit is zero or one") << (7 - x % 8);
        }
    }
    Ok(DecodedSamples::BitPacked(output))
}

fn pack_u8(
    context: FormatContext<'_>,
    elements: usize,
    cpu: crate::CpuCapabilities,
) -> Result<(DecodedSamples, bool), OutputFormatError> {
    if context.request.internal_color == ColorFormat::Luma
        && context.request.output_color == ColorFormat::Luma
        && context.alpha.is_none()
        && context.request.pixel_format == PixelFormat::U8(ChannelLayout::Luma)
    {
        let plane = context.planes[0];
        let start_x = usize::try_from(context.request.crop.x - plane.origin_x)
            .map_err(|_| OutputFormatError::arithmetic("converting SIMD luma crop x"))?;
        let start_y = usize::try_from(context.request.crop.y - plane.origin_y)
            .map_err(|_| OutputFormatError::arithmetic("converting SIMD luma crop y"))?;
        let mut output = Vec::new();
        let used_simd = append_u8(
            cpu.level(),
            plane.samples,
            plane.stride,
            [start_x, start_y],
            [context.width, context.height],
            context.request.scaled,
            &mut output,
        )?;
        return Ok((DecodedSamples::U8(output), used_simd));
    }
    let mut output = Vec::with_capacity(elements);
    let scaling = context.channel_scales();
    for_each_ordered(context, |sample, component, alpha| {
        let scaled = scaling.apply(sample, component, alpha)?;
        output.push(u8::try_from(clamp_sample(scaled, 0, 255)).expect("sample is clipped to u8"));
        Ok(())
    })?;
    Ok((DecodedSamples::U8(output), false))
}

fn pack_u16(
    context: FormatContext<'_>,
    elements: usize,
) -> Result<DecodedSamples, OutputFormatError> {
    let mut output = Vec::with_capacity(elements);
    let scaling = context.channel_scales();
    let maximum = if matches!(context.request.bit_depth, OutputBitDepth::U10) {
        1_023
    } else {
        65_535
    };
    for_each_ordered(context, |sample, component, alpha| {
        let scaled = scaling.apply(sample, component, alpha)?;
        output.push(
            u16::try_from(clamp_sample(scaled, 0, maximum)).expect("sample is clipped to u16"),
        );
        Ok(())
    })?;
    Ok(DecodedSamples::U16(output))
}

fn pack_i16(
    context: FormatContext<'_>,
    elements: usize,
) -> Result<DecodedSamples, OutputFormatError> {
    let mut output = Vec::with_capacity(elements);
    let scaling = context.channel_scales();
    for_each_ordered(context, |sample, component, alpha| {
        let scaled = scaling.apply(sample, component, alpha)?;
        output.push(
            i16::try_from(clamp_sample(scaled, -32_768, 32_767)).expect("sample is clipped to i16"),
        );
        Ok(())
    })?;
    Ok(DecodedSamples::I16(output))
}

fn pack_i32(
    context: FormatContext<'_>,
    elements: usize,
) -> Result<DecodedSamples, OutputFormatError> {
    let mut output = Vec::with_capacity(elements);
    let scaling = context.channel_scales();
    for_each_ordered(context, |sample, component, alpha| {
        output.push(scaling.apply(sample, component, alpha)?);
        Ok(())
    })?;
    Ok(DecodedSamples::I32(output))
}

fn pack_f16(
    context: FormatContext<'_>,
    elements: usize,
) -> Result<DecodedSamples, OutputFormatError> {
    let mut output = Vec::with_capacity(elements);
    for_each_ordered(context, |sample, component, alpha| {
        if context.is_padding(component, alpha) {
            output.push(0);
            return Ok(());
        }
        let scaled = if alpha {
            context.alpha_format().scaled
        } else {
            context.request.scaled
        };
        output.push(float16_bits(sample, scaled)?);
        Ok(())
    })?;
    Ok(DecodedSamples::F16(output))
}

fn pack_f32(
    context: FormatContext<'_>,
    elements: usize,
) -> Result<DecodedSamples, OutputFormatError> {
    let mut output = Vec::with_capacity(elements);
    for_each_ordered(context, |sample, component, alpha| {
        if context.is_padding(component, alpha) {
            output.push(0.0);
            return Ok(());
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
        output.push(f32::from_bits(float32_bits(
            sample,
            format.scaled,
            mantissa_length,
            exponent_bias,
        )?));
        Ok(())
    })?;
    Ok(DecodedSamples::F32(output))
}

fn for_each_ordered(
    context: FormatContext<'_>,
    mut format: impl FnMut(i32, usize, bool) -> Result<(), OutputFormatError>,
) -> Result<(), OutputFormatError> {
    if context.width == 0 || context.height == 0 {
        return Ok(());
    }
    if matches!(context.request.output_color, ColorFormat::NComponent(_)) {
        let crop_x = usize::try_from(context.request.crop.x)
            .map_err(|_| OutputFormatError::arithmetic("converting N-component crop x"))?;
        let crop_y = usize::try_from(context.request.crop.y)
            .map_err(|_| OutputFormatError::arithmetic("converting N-component crop y"))?;
        for source_y in crop_y..crop_y + context.height {
            for source_x in crop_x..crop_x + context.width {
                for (component, plane) in context.planes.iter().enumerate() {
                    format(plane.sample(source_x, source_y), component, false)?;
                }
                if let Some(alpha) = context.alpha {
                    format(alpha.sample(source_x, source_y), context.planes.len(), true)?;
                }
            }
        }
        return Ok(());
    }
    let crop_x = usize::try_from(context.request.crop.x)
        .map_err(|_| OutputFormatError::arithmetic("converting crop x"))?;
    let crop_y = usize::try_from(context.request.crop.y)
        .map_err(|_| OutputFormatError::arithmetic("converting crop y"))?;
    let shape = context.pixel_shape();
    let has_alpha = context.alpha.is_some();
    for source_y in crop_y..crop_y + context.height {
        for source_x in crop_x..crop_x + context.width {
            let (samples, count) = context.ordered_components_at(source_x, source_y, shape)?;
            for (component, &sample) in samples[..count].iter().enumerate() {
                format(sample, component, has_alpha && component + 1 == count)?;
            }
        }
    }
    Ok(())
}

const fn is_direct(internal: ColorFormat, output: ColorFormat) -> bool {
    matches!((internal, output), (ColorFormat::Luma, ColorFormat::Luma))
        || matches!((internal, output), (ColorFormat::Yuv(a), ColorFormat::Yuv(b)) if a as u8 == b as u8)
        || matches!((internal, output), (ColorFormat::NComponent(a), ColorFormat::NComponent(b)) if a == b)
}

pub(super) const fn layout(format: PixelFormat) -> ChannelLayout {
    match format {
        PixelFormat::BitPacked(layout)
        | PixelFormat::U8(layout)
        | PixelFormat::U16(layout)
        | PixelFormat::I16(layout)
        | PixelFormat::I32(layout)
        | PixelFormat::F16(layout)
        | PixelFormat::F32(layout) => layout,
        PixelFormat::Rgb555 | PixelFormat::Rgb565 | PixelFormat::Rgb101010 | PixelFormat::Rgbe => {
            ChannelLayout::Rgb
        }
    }
}
