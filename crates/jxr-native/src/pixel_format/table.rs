//! Canonical Annex-A Table A.6 descriptors.

use jxr_core::{
    AnnexABitDepth, AnnexAChannelOrder, AnnexANumericKind, AnnexAPixelFamily,
    AnnexAPixelFormatDescriptor, ChromaSampling,
};

pub(super) fn known_descriptor(code: u8) -> Option<AnnexAPixelFormatDescriptor> {
    ncomponent_descriptor(code)
        .or_else(|| basic_descriptor(code))
        .or_else(|| rgb_descriptor(code))
        .or_else(|| rgba_descriptor(code))
        .or_else(|| cmyk_yuv_descriptor(code))
}

fn basic_descriptor(code: u8) -> Option<AnnexAPixelFormatDescriptor> {
    match code {
        0x05 => Some(luma(AnnexABitDepth::Bit1, AnnexANumericKind::Unsigned)),
        0x08 => Some(luma(AnnexABitDepth::U8, AnnexANumericKind::Unsigned)),
        0x0b => Some(luma(AnnexABitDepth::U16, AnnexANumericKind::Unsigned)),
        0x13 => Some(luma(AnnexABitDepth::I16, AnnexANumericKind::FixedPoint)),
        0x3e => Some(luma(AnnexABitDepth::F16, AnnexANumericKind::Float)),
        0x3f => Some(luma(AnnexABitDepth::I32, AnnexANumericKind::FixedPoint)),
        0x11 => Some(luma(AnnexABitDepth::F32, AnnexANumericKind::Float)),
        0x09 => Some(packed_rgb(AnnexABitDepth::Rgb555)),
        0x0a => Some(packed_rgb(AnnexABitDepth::Rgb565)),
        0x14 => Some(packed_rgb(AnnexABitDepth::Rgb101010)),
        0x3d => Some(descriptor(
            AnnexAPixelFamily::Rgbe,
            3,
            false,
            false,
            AnnexABitDepth::U8,
            AnnexANumericKind::Float,
            AnnexAChannelOrder::Rgbe,
        )),
        _ => None,
    }
}

fn rgb_descriptor(code: u8) -> Option<AnnexAPixelFormatDescriptor> {
    match code {
        0x0d => Some(rgb(
            3,
            false,
            false,
            AnnexABitDepth::U8,
            AnnexANumericKind::Unsigned,
            AnnexAChannelOrder::Rgb,
        )),
        0x0c => Some(rgb(
            3,
            false,
            false,
            AnnexABitDepth::U8,
            AnnexANumericKind::Unsigned,
            AnnexAChannelOrder::Bgr,
        )),
        0x0e => Some(rgb(
            3,
            false,
            false,
            AnnexABitDepth::U8,
            AnnexANumericKind::Unsigned,
            AnnexAChannelOrder::Bgrx,
        )),
        0x15 => Some(rgb(
            3,
            false,
            false,
            AnnexABitDepth::U16,
            AnnexANumericKind::Unsigned,
            AnnexAChannelOrder::Rgb,
        )),
        0x12 => Some(rgb(
            3,
            false,
            false,
            AnnexABitDepth::I16,
            AnnexANumericKind::FixedPoint,
            AnnexAChannelOrder::Rgb,
        )),
        0x40 => Some(rgb(
            3,
            false,
            false,
            AnnexABitDepth::I16,
            AnnexANumericKind::FixedPoint,
            AnnexAChannelOrder::Rgbx,
        )),
        0x3b => Some(rgb(
            3,
            false,
            false,
            AnnexABitDepth::F16,
            AnnexANumericKind::Float,
            AnnexAChannelOrder::Rgb,
        )),
        0x42 => Some(rgb(
            3,
            false,
            false,
            AnnexABitDepth::F16,
            AnnexANumericKind::Float,
            AnnexAChannelOrder::Rgbx,
        )),
        0x18 => Some(rgb(
            3,
            false,
            false,
            AnnexABitDepth::I32,
            AnnexANumericKind::FixedPoint,
            AnnexAChannelOrder::Rgb,
        )),
        0x41 => Some(rgb(
            3,
            false,
            false,
            AnnexABitDepth::I32,
            AnnexANumericKind::FixedPoint,
            AnnexAChannelOrder::Rgbx,
        )),
        0x1b => Some(rgb(
            3,
            false,
            false,
            AnnexABitDepth::F32,
            AnnexANumericKind::Float,
            AnnexAChannelOrder::Rgbx,
        )),
        _ => None,
    }
}

fn rgba_descriptor(code: u8) -> Option<AnnexAPixelFormatDescriptor> {
    let (premultiplied, depth, numeric, order) = match code {
        0x0f => (
            false,
            AnnexABitDepth::U8,
            AnnexANumericKind::Unsigned,
            AnnexAChannelOrder::Bgra,
        ),
        0x16 => (
            false,
            AnnexABitDepth::U16,
            AnnexANumericKind::Unsigned,
            AnnexAChannelOrder::Rgba,
        ),
        0x1d => (
            false,
            AnnexABitDepth::I16,
            AnnexANumericKind::FixedPoint,
            AnnexAChannelOrder::Rgba,
        ),
        0x3a => (
            false,
            AnnexABitDepth::F16,
            AnnexANumericKind::Float,
            AnnexAChannelOrder::Rgba,
        ),
        0x1e => (
            false,
            AnnexABitDepth::I32,
            AnnexANumericKind::FixedPoint,
            AnnexAChannelOrder::Rgba,
        ),
        0x19 => (
            false,
            AnnexABitDepth::F32,
            AnnexANumericKind::Float,
            AnnexAChannelOrder::Rgba,
        ),
        0x10 => (
            true,
            AnnexABitDepth::U8,
            AnnexANumericKind::Unsigned,
            AnnexAChannelOrder::Bgra,
        ),
        0x17 => (
            true,
            AnnexABitDepth::U16,
            AnnexANumericKind::Unsigned,
            AnnexAChannelOrder::Rgba,
        ),
        0x1a => (
            true,
            AnnexABitDepth::F32,
            AnnexANumericKind::Float,
            AnnexAChannelOrder::Rgba,
        ),
        _ => return None,
    };
    Some(rgb(4, true, premultiplied, depth, numeric, order))
}

fn cmyk_yuv_descriptor(code: u8) -> Option<AnnexAPixelFormatDescriptor> {
    match code {
        0x1c => Some(cmyk(false, false, AnnexABitDepth::U8)),
        0x2c => Some(cmyk(false, true, AnnexABitDepth::U8)),
        0x1f => Some(cmyk(false, false, AnnexABitDepth::U16)),
        0x2d => Some(cmyk(false, true, AnnexABitDepth::U16)),
        0x54 => Some(cmyk(true, false, AnnexABitDepth::U8)),
        0x56 => Some(cmyk(true, true, AnnexABitDepth::U8)),
        0x55 => Some(cmyk(true, false, AnnexABitDepth::U16)),
        0x43 => Some(cmyk(true, true, AnnexABitDepth::U16)),
        0x44 => Some(yuv(ChromaSampling::Cs420, false, AnnexABitDepth::U8)),
        0x45 => Some(yuv(ChromaSampling::Cs422, false, AnnexABitDepth::U8)),
        0x46 => Some(yuv(ChromaSampling::Cs422, false, AnnexABitDepth::U10)),
        0x47 => Some(yuv(ChromaSampling::Cs422, false, AnnexABitDepth::U16)),
        0x48 => Some(yuv(ChromaSampling::Cs444, false, AnnexABitDepth::U8)),
        0x49 => Some(yuv(ChromaSampling::Cs444, false, AnnexABitDepth::U10)),
        0x4a => Some(yuv(ChromaSampling::Cs444, false, AnnexABitDepth::U16)),
        0x4b => Some(yuv_fixed(ChromaSampling::Cs444, false)),
        0x4c => Some(yuv(ChromaSampling::Cs420, true, AnnexABitDepth::U8)),
        0x4d => Some(yuv(ChromaSampling::Cs422, true, AnnexABitDepth::U8)),
        0x4e => Some(yuv(ChromaSampling::Cs422, true, AnnexABitDepth::U10)),
        0x4f => Some(yuv(ChromaSampling::Cs422, true, AnnexABitDepth::U16)),
        0x50 => Some(yuv(ChromaSampling::Cs444, true, AnnexABitDepth::U8)),
        0x51 => Some(yuv(ChromaSampling::Cs444, true, AnnexABitDepth::U10)),
        0x52 => Some(yuv(ChromaSampling::Cs444, true, AnnexABitDepth::U16)),
        0x53 => Some(yuv_fixed(ChromaSampling::Cs444, true)),
        _ => None,
    }
}

fn ncomponent_descriptor(code: u8) -> Option<AnnexAPixelFormatDescriptor> {
    let (channels, alpha, depth) = match code {
        0x20..=0x25 => (code - 0x20 + 3, false, AnnexABitDepth::U8),
        0x2e..=0x33 => (code - 0x2e + 4, true, AnnexABitDepth::U8),
        0x26..=0x2b => (code - 0x26 + 3, false, AnnexABitDepth::U16),
        0x34..=0x39 => (code - 0x34 + 4, true, AnnexABitDepth::U16),
        _ => return None,
    };
    Some(descriptor(
        AnnexAPixelFamily::NComponent,
        channels,
        alpha,
        false,
        depth,
        AnnexANumericKind::Unsigned,
        AnnexAChannelOrder::Components,
    ))
}

const fn luma(depth: AnnexABitDepth, numeric: AnnexANumericKind) -> AnnexAPixelFormatDescriptor {
    descriptor(
        AnnexAPixelFamily::Luma,
        1,
        false,
        false,
        depth,
        numeric,
        AnnexAChannelOrder::Luma,
    )
}

const fn packed_rgb(depth: AnnexABitDepth) -> AnnexAPixelFormatDescriptor {
    descriptor(
        AnnexAPixelFamily::Rgb,
        3,
        false,
        false,
        depth,
        AnnexANumericKind::Unsigned,
        AnnexAChannelOrder::PackedBgr,
    )
}

const fn rgb(
    channels: u8,
    alpha: bool,
    premultiplied: bool,
    depth: AnnexABitDepth,
    numeric: AnnexANumericKind,
    order: AnnexAChannelOrder,
) -> AnnexAPixelFormatDescriptor {
    descriptor(
        AnnexAPixelFamily::Rgb,
        channels,
        alpha,
        premultiplied,
        depth,
        numeric,
        order,
    )
}

const fn cmyk(direct: bool, alpha: bool, depth: AnnexABitDepth) -> AnnexAPixelFormatDescriptor {
    descriptor(
        AnnexAPixelFamily::Cmyk { direct },
        4 + alpha as u8,
        alpha,
        false,
        depth,
        AnnexANumericKind::Unsigned,
        AnnexAChannelOrder::Cmyk,
    )
}

const fn yuv(
    sampling: ChromaSampling,
    alpha: bool,
    depth: AnnexABitDepth,
) -> AnnexAPixelFormatDescriptor {
    descriptor(
        AnnexAPixelFamily::Yuv(sampling),
        3 + alpha as u8,
        alpha,
        false,
        depth,
        AnnexANumericKind::Unsigned,
        AnnexAChannelOrder::Yuv,
    )
}

const fn yuv_fixed(sampling: ChromaSampling, alpha: bool) -> AnnexAPixelFormatDescriptor {
    descriptor(
        AnnexAPixelFamily::Yuv(sampling),
        3 + alpha as u8,
        alpha,
        false,
        AnnexABitDepth::I16,
        AnnexANumericKind::FixedPoint,
        AnnexAChannelOrder::Yuv,
    )
}

const fn descriptor(
    family: AnnexAPixelFamily,
    channels: u8,
    alpha: bool,
    premultiplied_alpha: bool,
    bit_depth: AnnexABitDepth,
    numeric: AnnexANumericKind,
    order: AnnexAChannelOrder,
) -> AnnexAPixelFormatDescriptor {
    AnnexAPixelFormatDescriptor {
        family,
        channels,
        alpha,
        premultiplied_alpha,
        bit_depth,
        numeric,
        order,
    }
}
