//! Packed RGB and shared-exponent RGBE output words.

use jxr_core::ColorFormat;
use jxr_math::rgbe::{pack_rgbe as pack_rgbe_word, postscale_rgbe};

use super::{OutputBitDepth, OutputFormatError, packing::FormatContext, scaling::ChannelScale};

/// An RGB555 word of red, green and blue, each of five bits: red highest.
fn pack_rgb555([red, green, blue]: [u32; 3]) -> u16 {
    u16::try_from(blue | (green << 5) | (red << 10)).expect("RGB555 occupies 15 bits")
}

/// Red, green and blue of an RGB555 word: [`pack_rgb555`] reversed.
pub(crate) fn unpack_rgb555(word: u16) -> [i32; 3] {
    let word = i32::from(word);
    [word >> 10 & 0x1f, word >> 5 & 0x1f, word & 0x1f]
}

/// An RGB565 word of five-bit red and blue and six-bit green: red highest.
fn pack_rgb565([red, green, blue]: [u32; 3]) -> u16 {
    u16::try_from(blue | (green << 5) | (red << 11)).expect("RGB565 occupies 16 bits")
}

/// Red, green and blue of an RGB565 word: [`pack_rgb565`] reversed.
pub(crate) fn unpack_rgb565(word: u16) -> [i32; 3] {
    let word = i32::from(word);
    [word >> 11 & 0x1f, word >> 5 & 0x3f, word & 0x1f]
}

/// An RGB101010 word of red, green and blue, each of ten bits: red highest.
const fn pack_rgb101010([red, green, blue]: [u32; 3]) -> u32 {
    blue | (green << 10) | (red << 20)
}

/// Red, green and blue of an RGB101010 word: [`pack_rgb101010`] reversed.
pub(crate) fn unpack_rgb101010(word: u32) -> [i32; 3] {
    let field = |shift: u32| (word >> shift & 0x3ff).cast_signed();
    [field(20), field(10), field(0)]
}

/// The byte of a row of one-bit samples that holds the sample of column
/// `x`, and the shift of its bit there: the first sample is the highest
/// bit of the first byte.
pub(crate) const fn bit_position(x: usize) -> (usize, usize) {
    (x / 8, 7 - x % 8)
}

pub(super) fn pack_rgb555_into(
    context: FormatContext<'_>,
    output: &mut [u16],
) -> Result<(), OutputFormatError> {
    let scaling = rgb_scales(context);
    fill_pixels(context, output, |components| {
        Ok(pack_rgb555(scale_rgb(context, &scaling, components)?))
    })
}

pub(super) fn pack_rgb565_into(
    context: FormatContext<'_>,
    output: &mut [u16],
) -> Result<(), OutputFormatError> {
    let scaling = rgb_scales(context);
    fill_pixels(context, output, |components| {
        Ok(pack_rgb565(scale_rgb(context, &scaling, components)?))
    })
}

pub(super) fn pack_rgb101010_into(
    context: FormatContext<'_>,
    output: &mut [u32],
) -> Result<(), OutputFormatError> {
    let scaling = rgb_scales(context);
    fill_pixels(context, output, |components| {
        Ok(pack_rgb101010(scale_rgb(context, &scaling, components)?))
    })
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
        let maximum = match (context.request.bit_depth, component) {
            // Red and blue are a bit narrower than green.
            (OutputBitDepth::Rgb565, 0 | 2) => 31,
            (depth, _) => u32::from(
                depth
                    .maximum()
                    .expect("RGB scaler selected only for packed output"),
            ),
        };
        values[component] = clip(scaled, maximum);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_rgb_words_unpack_to_the_fields_packed() {
        for [red, green, blue] in [[0, 0, 0], [31, 0, 17], [1, 2, 3], [31, 31, 31]] {
            let fields = [red, green, blue].map(|value: u32| i32::try_from(value).unwrap());
            assert_eq!(unpack_rgb555(pack_rgb555([red, green, blue])), fields);
            let wide = [red, green * 2 + 1, blue];
            let fields = wide.map(|value| i32::try_from(value).unwrap());
            assert_eq!(unpack_rgb565(pack_rgb565(wide)), fields);
            let ten = [red * 33, green * 33, blue * 33];
            let fields = ten.map(|value| i32::try_from(value).unwrap());
            assert_eq!(unpack_rgb101010(pack_rgb101010(ten)), fields);
        }
        // Red is the highest field of every word.
        assert_eq!(pack_rgb555([1, 0, 0]), 1 << 10);
        assert_eq!(pack_rgb565([1, 0, 0]), 1 << 11);
        assert_eq!(pack_rgb101010([1, 0, 0]), 1 << 20);
        assert_eq!(bit_position(0), (0, 7));
        assert_eq!(bit_position(13), (1, 2));
    }
}
