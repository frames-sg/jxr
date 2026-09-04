// SPDX-License-Identifier: MIT OR Apache-2.0

use jxr::{
    AlphaHandling, AnnexABitDepth, AnnexAChannelOrder, AnnexAPixelFamily, AnnexAPixelFormat,
    ChannelLayout, ImageInfo, PixelFormat,
};

use crate::OracleError;

/// Rust decode policy matching the raw representation selected by T.835.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OracleFormat {
    /// Typed Rust output requested for byte-for-byte comparison.
    pub pixel_format: PixelFormat,
    /// Alpha policy matching the reference program's combined raw output.
    pub alpha: AlphaHandling,
}

/// Map parsed Annex-A metadata to the matching T.835 raw representation.
pub fn oracle_format(info: &ImageInfo) -> Result<OracleFormat, OracleError> {
    let Some(AnnexAPixelFormat::Known(descriptor)) = info.metadata.annex_a_pixel_format else {
        return Err(unsupported("known Annex-A pixel format is required"));
    };
    let alpha = if descriptor.alpha {
        AlphaHandling::Preserve
    } else {
        AlphaHandling::Drop
    };
    let layout = channel_layout(
        descriptor.family,
        descriptor.order,
        descriptor.channels,
        descriptor.alpha,
    )?;
    let pixel_format = match descriptor.bit_depth {
        AnnexABitDepth::Bit1 => PixelFormat::BitPacked(layout),
        AnnexABitDepth::U8 => match descriptor.family {
            AnnexAPixelFamily::Rgbe => PixelFormat::Rgbe,
            _ => PixelFormat::U8(layout),
        },
        AnnexABitDepth::U10 | AnnexABitDepth::U16 => PixelFormat::U16(layout),
        AnnexABitDepth::I16 => PixelFormat::I16(layout),
        AnnexABitDepth::F16 => PixelFormat::F16(layout),
        AnnexABitDepth::I32 => PixelFormat::I32(layout),
        AnnexABitDepth::F32 => PixelFormat::F32(layout),
        AnnexABitDepth::Rgb555 => PixelFormat::Rgb555,
        AnnexABitDepth::Rgb565 => PixelFormat::Rgb565,
        AnnexABitDepth::Rgb101010 => PixelFormat::Rgb101010,
    };
    Ok(OracleFormat {
        pixel_format,
        alpha,
    })
}

fn channel_layout(
    family: AnnexAPixelFamily,
    order: AnnexAChannelOrder,
    channels: u8,
    alpha: bool,
) -> Result<ChannelLayout, OracleError> {
    Ok(match family {
        AnnexAPixelFamily::Luma if alpha => ChannelLayout::LumaAlpha,
        AnnexAPixelFamily::Luma => ChannelLayout::Luma,
        AnnexAPixelFamily::Rgb if alpha && order == AnnexAChannelOrder::Bgra => ChannelLayout::Bgra,
        AnnexAPixelFamily::Rgb if alpha => ChannelLayout::Rgba,
        AnnexAPixelFamily::Rgb if order == AnnexAChannelOrder::Rgbx => ChannelLayout::Rgbx,
        AnnexAPixelFamily::Rgb if order == AnnexAChannelOrder::Bgrx => ChannelLayout::Bgrx,
        AnnexAPixelFamily::Rgb if order == AnnexAChannelOrder::Bgr => ChannelLayout::Bgr,
        AnnexAPixelFamily::Rgb | AnnexAPixelFamily::Rgbe => ChannelLayout::Rgb,
        AnnexAPixelFamily::Cmyk { .. } if alpha => ChannelLayout::Cmyka,
        AnnexAPixelFamily::Cmyk { .. } => ChannelLayout::Cmyk,
        AnnexAPixelFamily::Yuv(sampling) if alpha => ChannelLayout::Yuva(sampling),
        AnnexAPixelFamily::Yuv(sampling) => ChannelLayout::Yuv(sampling),
        AnnexAPixelFamily::NComponent if alpha => {
            let primary = channels
                .checked_sub(1)
                .ok_or_else(|| unsupported("alpha channel count"))?;
            ChannelLayout::NComponentAlpha(u16::from(primary))
        }
        AnnexAPixelFamily::NComponent => ChannelLayout::NComponent(u16::from(channels)),
    })
}

const fn unsupported(reason: &'static str) -> OracleError {
    OracleError::UnsupportedFormat { reason }
}

#[cfg(test)]
mod tests {
    use jxr::{
        AlphaMode, AnnexABitDepth, AnnexAChannelOrder, AnnexANumericKind, AnnexAPixelFamily,
        AnnexAPixelFormatDescriptor, BandPresence, BitstreamMode, ColorFormat, ImageMetadata,
        OverlapMode, PlaneInfo, SampleFormat, TileGrid,
    };

    use super::*;

    #[test]
    fn maps_eight_bit_bgr_without_alpha() {
        let mut info = image_info();
        info.metadata.annex_a_pixel_format =
            Some(AnnexAPixelFormat::Known(AnnexAPixelFormatDescriptor {
                family: AnnexAPixelFamily::Rgb,
                channels: 3,
                alpha: false,
                premultiplied_alpha: false,
                bit_depth: AnnexABitDepth::U8,
                numeric: AnnexANumericKind::Unsigned,
                order: AnnexAChannelOrder::Bgr,
            }));
        assert_eq!(
            oracle_format(&info).unwrap(),
            OracleFormat {
                pixel_format: PixelFormat::U8(jxr::ChannelLayout::Bgr),
                alpha: AlphaHandling::Drop,
            }
        );
    }

    fn image_info() -> ImageInfo {
        ImageInfo {
            width: 1,
            height: 1,
            profile: None,
            level: None,
            primary: PlaneInfo {
                color_format: ColorFormat::Rgb,
                sample_format: SampleFormat::Unsigned { bits: 8 },
                bands: BandPresence::DcOnly,
                bitstream_mode: BitstreamMode::Spatial,
                overlap: OverlapMode::None,
                short_header: false,
                long_word: false,
                scaled: false,
                chroma_centering: [0, 0],
                shift_bits: 0,
                mantissa_length: 0,
                exponent_bias: 0,
                width: 1,
                height: 1,
            },
            alpha_mode: AlphaMode::None,
            premultiplied_alpha: false,
            alpha: None,
            tiles: TileGrid {
                column_widths: vec![1],
                row_heights: vec![1],
                hard_tiles: false,
            },
            metadata: ImageMetadata::default(),
        }
    }
}
