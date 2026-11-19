//! Pixel formats a case can have, and what the harness needs to know of
//! their storage.

use jxr::{ChannelLayout, ChromaSampling, PixelFormat};

use super::{Input, samples::size};

/// Every pixel format the container table declares, with the sample depth of
/// those narrower than their storage.
pub(super) fn declared_formats() -> Vec<(PixelFormat, Option<u8>)> {
    use ChannelLayout::{
        Bgr, Bgra, Bgrx, Cmyk, Cmyka, Luma, NComponent, NComponentAlpha, Rgb, Rgba, Rgbx, Yuv, Yuva,
    };
    use ChromaSampling::{Cs420, Cs422, Cs444};
    let mut formats = vec![
        (PixelFormat::BitPacked(Luma), None),
        (PixelFormat::Rgb555, None),
        (PixelFormat::Rgb565, None),
        (PixelFormat::Rgb101010, None),
        (PixelFormat::Rgbe, None),
    ];
    let channels = (3..=8).flat_map(|count| [NComponent(count), NComponentAlpha(count)]);
    let eight_bit = [
        Luma,
        Rgb,
        Bgr,
        Bgrx,
        Rgba,
        Bgra,
        Cmyk,
        Cmyka,
        Yuv(Cs444),
        Yuv(Cs422),
        Yuv(Cs420),
        Yuva(Cs444),
        Yuva(Cs422),
        Yuva(Cs420),
    ];
    formats.extend(
        eight_bit
            .into_iter()
            .chain(channels.clone())
            .map(|layout| (PixelFormat::U8(layout), None)),
    );
    formats.extend(
        [Luma, Rgb, Rgba, Cmyk, Cmyka]
            .into_iter()
            .chain(channels)
            .map(|layout| (PixelFormat::U16(layout), None)),
    );
    // YUV in 16-bit storage holds sixteen-bit or ten-bit samples.
    for layout in [Yuv(Cs444), Yuv(Cs422), Yuva(Cs444), Yuva(Cs422)] {
        formats.push((PixelFormat::U16(layout), None));
        formats.push((PixelFormat::U16(layout), Some(10)));
    }
    formats.extend(
        [Luma, Rgb, Rgbx, Rgba, Yuv(Cs444), Yuva(Cs444)]
            .map(|layout| (PixelFormat::I16(layout), None)),
    );
    let deep: [fn(ChannelLayout) -> PixelFormat; 3] =
        [PixelFormat::I32, PixelFormat::F16, PixelFormat::F32];
    for storage in deep {
        formats.extend([Luma, Rgb, Rgbx, Rgba].map(|layout| (storage(layout), None)));
    }
    formats
}

/// Any storage with any channel layout, most of which nothing declares.
pub(super) fn undeclared_format(input: &mut Input<'_>) -> (PixelFormat, Option<u8>) {
    let count = u16::from(input.below(11));
    let sampling = match input.below(3) {
        0 => ChromaSampling::Cs444,
        1 => ChromaSampling::Cs422,
        _ => ChromaSampling::Cs420,
    };
    let layout = match input.below(16) {
        0 => ChannelLayout::Luma,
        1 => ChannelLayout::LumaAlpha,
        2 => ChannelLayout::Rgb,
        3 => ChannelLayout::Rgbx,
        4 => ChannelLayout::Rgba,
        5 => ChannelLayout::Bgr,
        6 => ChannelLayout::Bgrx,
        7 => ChannelLayout::Bgra,
        8 => ChannelLayout::Cmyk,
        9 => ChannelLayout::Cmyka,
        10 => ChannelLayout::Yuv(sampling),
        11 => ChannelLayout::Yuva(sampling),
        12 | 13 => ChannelLayout::NComponent(count),
        _ => ChannelLayout::NComponentAlpha(count),
    };
    let format = match input.below(11) {
        0 => PixelFormat::BitPacked(layout),
        1 => PixelFormat::U8(layout),
        2 => PixelFormat::U16(layout),
        3 => PixelFormat::I16(layout),
        4 => PixelFormat::I32(layout),
        5 => PixelFormat::F16(layout),
        6 => PixelFormat::F32(layout),
        7 => PixelFormat::Rgb555,
        8 => PixelFormat::Rgb565,
        9 => PixelFormat::Rgb101010,
        _ => PixelFormat::Rgbe,
    };
    let sample_bits = input.rarely().then(|| input.below(18));
    (format, sample_bits)
}

/// Storage elements of a tightly packed image.
pub(super) fn sample_count(format: PixelFormat, width: u32, height: u32) -> usize {
    let Some(sampling) = format.planar_sampling() else {
        let row = format.row_elements(width).expect("a fuzzed row fits usize");
        return row * size(height);
    };
    let (width, height) = (size(width), size(height));
    let chroma_height = if sampling == ChromaSampling::Cs420 {
        height / 2
    } else {
        height
    };
    let full = 1 + usize::from(format.has_alpha());
    width * height * full + 2 * (width / 2) * chroma_height
}
