// SPDX-License-Identifier: MIT OR Apache-2.0

use jxr::{
    AlphaHandling, AlphaMode, AnnexABitDepth, AnnexAPixelFamily, AnnexAPixelFormat, ChannelLayout,
    ChromaSampling, EncodeRequest, ImageInfo, PixelFormat,
};

use crate::OracleError;

/// Rust decode policy matching the raw representation selected by T.835.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OracleFormat {
    /// Typed Rust output requested for byte-for-byte comparison.
    pub pixel_format: PixelFormat,
    /// Alpha policy matching the reference program's combined raw output.
    pub alpha: AlphaHandling,
    /// Whether the reference program writes each channel as a whole plane,
    /// one after another, instead of interleaving them. It does so for
    /// CMYKDIRECT and YUV images.
    pub planar: bool,
    /// Planes the reference program writes after the alpha plane of a planar
    /// image. It decodes separate alpha as an image of its own and appends
    /// that image's output, which it lays out with one plane per primary
    /// channel, each the size of that channel's plane, and decodes only the
    /// first into.
    pub trailing_planes: u16,
    /// The sample every trailing plane holds: zero beside CMYKDIRECT, and
    /// the neutral chroma value of the sample depth beside YUV.
    pub trailing_fill: u16,
    /// Image width in samples, which with the height gives each plane's
    /// size.
    pub width: u32,
    /// Image height in samples.
    pub height: u32,
}

impl OracleFormat {
    /// The chroma sampling of a YUV 4:2:0 or 4:2:2 format, whose samples
    /// the Rust decoder already returns plane by plane.
    #[must_use]
    pub const fn subsampling(self) -> Option<ChromaSampling> {
        self.pixel_format.planar_sampling()
    }

    /// Samples in the plane of channel `channel`: the image's own count,
    /// or that of a subsampled U or V plane.
    #[must_use]
    pub const fn plane_samples(self, channel: usize) -> usize {
        let (width, height) = (self.width as usize, self.height as usize);
        match (self.subsampling(), channel) {
            (Some(ChromaSampling::Cs420), 1 | 2) => (width / 2) * (height / 2),
            (Some(_), 1 | 2) => (width / 2) * height,
            _ => width * height,
        }
    }
}

/// Whether the encoder can code a source of `format` with subsampled
/// chroma: full-resolution RGB chroma at every depth, packed RGB and RGBE
/// among them, and YUV 4:4:4.
#[must_use]
pub const fn subsamples_chroma(format: PixelFormat) -> bool {
    use ChannelLayout::{Bgr, Bgra, Bgrx, Rgb, Rgba, Rgbx, Yuv, Yuva};
    match (format, format.channel_layout()) {
        (PixelFormat::BitPacked(_), _) => false,
        (_, None) => true,
        (_, Some(layout)) => matches!(
            layout,
            Rgb | Rgbx
                | Rgba
                | Bgr
                | Bgrx
                | Bgra
                | Yuv(ChromaSampling::Cs444)
                | Yuva(ChromaSampling::Cs444)
        ),
    }
}

/// Whether a file coded from `format` under `request` has its chroma
/// subsampled by the encoder, which loses information whatever the
/// quantizers, so that it is not reproduced exactly.
#[must_use]
pub const fn decimates_chroma(format: PixelFormat, request: &EncodeRequest) -> bool {
    subsamples_chroma(format) && !matches!(request.chroma_sampling, ChromaSampling::Cs444)
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
    let pixel_format = jxr_native::native_pixel_format(&descriptor)
        .ok_or_else(|| unsupported("alpha channel count"))?;
    let yuv = matches!(descriptor.family, AnnexAPixelFamily::Yuv(_));
    let planar = yuv || descriptor.family == AnnexAPixelFamily::Cmyk { direct: true };
    let trailing_planes = if planar && info.alpha_mode == AlphaMode::Separate {
        pixel_format.channel_count().saturating_sub(2)
    } else {
        0
    };
    // The alpha image has no chroma, so the reference fills in mid-range.
    let trailing_fill = match (yuv, descriptor.bit_depth) {
        (false, _) => 0,
        (true, AnnexABitDepth::U8) => 1 << 7,
        (true, AnnexABitDepth::U10) => 1 << 9,
        (true, _) => 1 << 15,
    };
    Ok(OracleFormat {
        pixel_format,
        alpha,
        planar,
        trailing_planes,
        trailing_fill,
        width: info.width,
        height: info.height,
    })
}

const fn unsupported(reason: &'static str) -> OracleError {
    OracleError::UnsupportedFormat { reason }
}

#[cfg(test)]
mod tests {
    use jxr::{
        AnnexABitDepth, AnnexAChannelOrder, AnnexANumericKind, AnnexAPixelFormatDescriptor,
        BandPresence, BitstreamMode, ColorFormat, ImageMetadata, OverlapMode, PlaneInfo,
        SampleFormat, TileGrid,
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
                planar: false,
                trailing_planes: 0,
                trailing_fill: 0,
                width: 1,
                height: 1,
            }
        );
    }

    #[test]
    fn only_direct_cmyk_is_written_plane_by_plane() {
        for (direct, bit_depth, pixel_format) in [
            (
                false,
                AnnexABitDepth::U8,
                PixelFormat::U8(jxr::ChannelLayout::Cmyk),
            ),
            (
                true,
                AnnexABitDepth::U16,
                PixelFormat::U16(jxr::ChannelLayout::Cmyk),
            ),
        ] {
            let mut info = image_info();
            info.metadata.annex_a_pixel_format =
                Some(AnnexAPixelFormat::Known(AnnexAPixelFormatDescriptor {
                    family: AnnexAPixelFamily::Cmyk { direct },
                    channels: 4,
                    alpha: false,
                    premultiplied_alpha: false,
                    bit_depth,
                    numeric: AnnexANumericKind::Unsigned,
                    order: AnnexAChannelOrder::Cmyk,
                }));
            assert_eq!(
                oracle_format(&info).unwrap(),
                OracleFormat {
                    pixel_format,
                    alpha: AlphaHandling::Drop,
                    planar: direct,
                    trailing_planes: 0,
                    trailing_fill: 0,
                    width: 1,
                    height: 1,
                }
            );
        }
    }

    #[test]
    fn separate_alpha_of_a_planar_image_is_followed_by_empty_planes() {
        for (direct, alpha_mode, trailing_planes) in [
            (true, AlphaMode::Separate, 3),
            (true, AlphaMode::Integrated, 0),
            (false, AlphaMode::Separate, 0),
        ] {
            let mut info = image_info();
            info.alpha_mode = alpha_mode;
            info.metadata.annex_a_pixel_format =
                Some(AnnexAPixelFormat::Known(AnnexAPixelFormatDescriptor {
                    family: AnnexAPixelFamily::Cmyk { direct },
                    channels: 5,
                    alpha: true,
                    premultiplied_alpha: false,
                    bit_depth: AnnexABitDepth::U8,
                    numeric: AnnexANumericKind::Unsigned,
                    order: AnnexAChannelOrder::Cmyk,
                }));
            assert_eq!(
                oracle_format(&info).unwrap(),
                OracleFormat {
                    pixel_format: PixelFormat::U8(jxr::ChannelLayout::Cmyka),
                    alpha: AlphaHandling::Preserve,
                    planar: direct,
                    trailing_planes,
                    trailing_fill: 0,
                    width: 1,
                    height: 1,
                }
            );
        }
    }

    #[test]
    fn yuv_is_written_plane_by_plane_with_neutral_planes_after_separate_alpha() {
        use jxr::{ChannelLayout, ChromaSampling};

        // `(sampling, depth, storage, neutral chroma, chroma plane samples)`
        // of an 8 x 6 image.
        for (sampling, bit_depth, pixel_format, trailing_fill, chroma) in [
            (
                ChromaSampling::Cs420,
                AnnexABitDepth::U8,
                PixelFormat::U8(ChannelLayout::Yuva(ChromaSampling::Cs420)),
                0x80,
                12,
            ),
            (
                ChromaSampling::Cs422,
                AnnexABitDepth::U10,
                PixelFormat::U16(ChannelLayout::Yuva(ChromaSampling::Cs422)),
                0x200,
                24,
            ),
            (
                ChromaSampling::Cs444,
                AnnexABitDepth::U16,
                PixelFormat::U16(ChannelLayout::Yuva(ChromaSampling::Cs444)),
                0x8000,
                48,
            ),
            (
                ChromaSampling::Cs444,
                AnnexABitDepth::I16,
                PixelFormat::I16(ChannelLayout::Yuva(ChromaSampling::Cs444)),
                0x8000,
                48,
            ),
        ] {
            for (alpha_mode, trailing_planes) in
                [(AlphaMode::Separate, 2), (AlphaMode::Integrated, 0)]
            {
                let mut info = image_info();
                (info.width, info.height) = (8, 6);
                info.alpha_mode = alpha_mode;
                info.metadata.annex_a_pixel_format =
                    Some(AnnexAPixelFormat::Known(AnnexAPixelFormatDescriptor {
                        family: AnnexAPixelFamily::Yuv(sampling),
                        channels: 4,
                        alpha: true,
                        premultiplied_alpha: false,
                        bit_depth,
                        numeric: AnnexANumericKind::Unsigned,
                        order: AnnexAChannelOrder::Yuv,
                    }));
                let format = oracle_format(&info).unwrap();
                assert_eq!(
                    format,
                    OracleFormat {
                        pixel_format,
                        alpha: AlphaHandling::Preserve,
                        planar: true,
                        trailing_planes,
                        trailing_fill,
                        width: 8,
                        height: 6,
                    }
                );
                assert_eq!(
                    [0, 1, 2, 3].map(|channel| format.plane_samples(channel)),
                    [48, chroma, chroma, 48]
                );
                assert_eq!(
                    format.subsampling(),
                    (sampling != ChromaSampling::Cs444).then_some(sampling)
                );
            }
        }
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
