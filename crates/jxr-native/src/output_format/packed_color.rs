//! Packed RGB and shared-exponent RGBE output words.

use jxr_core::{ColorFormat, DecodedSamples};
use jxr_math::rgbe::{pack_rgbe as pack_rgbe_word, postscale_rgbe};

use super::{OutputBitDepth, OutputFormatError, packing::FormatContext, scaling::ChannelScale};

pub(super) fn pack_rgb555(
    context: FormatContext<'_>,
    pixels: usize,
) -> Result<DecodedSamples, OutputFormatError> {
    let mut output = vec![0; pixels];
    pack_rgb555_into(context, &mut output)?;
    Ok(DecodedSamples::Rgb555(output))
}

pub(super) fn pack_rgb555_into(
    context: FormatContext<'_>,
    output: &mut [u16],
) -> Result<(), OutputFormatError> {
    let scaling = rgb_scales(context);
    fill_pixels(context, output, |components| {
        let values = scale_rgb(context, &scaling, components)?;
        Ok(
            u16::try_from(values[2] | (values[1] << 5) | (values[0] << 10))
                .expect("RGB555 occupies 15 bits"),
        )
    })
}

pub(super) fn pack_rgb565(
    context: FormatContext<'_>,
    pixels: usize,
) -> Result<DecodedSamples, OutputFormatError> {
    let mut output = vec![0; pixels];
    pack_rgb565_into(context, &mut output)?;
    Ok(DecodedSamples::Rgb565(output))
}

pub(super) fn pack_rgb565_into(
    context: FormatContext<'_>,
    output: &mut [u16],
) -> Result<(), OutputFormatError> {
    let scaling = rgb_scales(context);
    fill_pixels(context, output, |components| {
        let values = scale_rgb(context, &scaling, components)?;
        Ok(
            u16::try_from(values[2] | (values[1] << 5) | (values[0] << 11))
                .expect("RGB565 occupies 16 bits"),
        )
    })
}

pub(super) fn pack_rgb101010(
    context: FormatContext<'_>,
    pixels: usize,
) -> Result<DecodedSamples, OutputFormatError> {
    let mut output = vec![0; pixels];
    pack_rgb101010_into(context, &mut output)?;
    Ok(DecodedSamples::Rgb101010(output))
}

pub(super) fn pack_rgb101010_into(
    context: FormatContext<'_>,
    output: &mut [u32],
) -> Result<(), OutputFormatError> {
    let scaling = rgb_scales(context);
    fill_pixels(context, output, |components| {
        let values = scale_rgb(context, &scaling, components)?;
        Ok(values[2] | (values[1] << 10) | (values[0] << 20))
    })
}

pub(super) fn pack_rgbe(
    context: FormatContext<'_>,
    pixels: usize,
) -> Result<DecodedSamples, OutputFormatError> {
    let mut output = vec![0; pixels];
    pack_rgbe_into(context, &mut output)?;
    Ok(DecodedSamples::Rgbe(output))
}

pub(super) fn pack_rgbe_into(
    context: FormatContext<'_>,
    output: &mut [u32],
) -> Result<(), OutputFormatError> {
    let scaling: [ChannelScale; 3] = core::array::from_fn(|component| {
        ChannelScale::resolve(
            component,
            ColorFormat::Rgbe,
            OutputBitDepth::U8,
            context.request.scaled,
        )
    });
    fill_pixels(context, output, |components| {
        let mut scaled = [0; 3];
        for component in 0..3 {
            scaled[component] = scaling[component].apply(components[component])?;
        }
        Ok(pack_rgbe_word(postscale_rgbe(scaled)))
    })
}

/// Resolves the three packed-RGB channel scales once per image.
fn rgb_scales(context: FormatContext<'_>) -> [ChannelScale; 3] {
    core::array::from_fn(|component| {
        ChannelScale::resolve(
            component,
            ColorFormat::Rgb,
            context.request.bit_depth,
            context.request.scaled,
        )
    })
}

fn scale_rgb(
    context: FormatContext<'_>,
    scaling: &[ChannelScale; 3],
    components: [i32; 4],
) -> Result<[u32; 3], OutputFormatError> {
    let mut values = [0; 3];
    for component in 0..3 {
        let scaled = scaling[component].apply(components[component])?;
        values[component] = clip(
            scaled,
            match context.request.bit_depth {
                OutputBitDepth::Rgb565 if component == 1 => 63,
                OutputBitDepth::Rgb555 | OutputBitDepth::Rgb565 => 31,
                OutputBitDepth::Rgb101010 => 1023,
                _ => unreachable!("RGB scaler selected only for packed output"),
            },
        );
    }
    Ok(values)
}

fn clip(value: i32, maximum: u32) -> u32 {
    u32::try_from(value).unwrap_or(0).min(maximum)
}

fn fill_pixels<T>(
    context: FormatContext<'_>,
    output: &mut [T],
    mut format: impl FnMut([i32; 4]) -> Result<T, OutputFormatError>,
) -> Result<(), OutputFormatError> {
    let mut index = 0;
    context.for_each_color(|components| {
        output[index] = format(components)?;
        index += 1;
        Ok(())
    })
}
