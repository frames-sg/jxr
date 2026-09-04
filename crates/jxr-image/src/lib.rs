#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

/// The pinned upstream image API used by this adapter.
pub use image;

use image::{
    DynamicImage, ImageBuffer, Luma, LumaA, Rgb, Rgba,
    metadata::{CicpColorPrimaries, CicpTransferCharacteristics},
};
use jxr::{
    AlphaHandling, ChannelLayout, DecodeReport, DecodeRequest, DecodedImage, DecodedSamples,
    ImageInfo, JxrError, JxrView, PixelFormat, PreparedJxr, Rect,
};

/// Whether the adapted image has straight, premultiplied, or no alpha.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlphaRepresentation {
    /// The image has no alpha channel.
    None,
    /// Color values are independent of the stored alpha value.
    Straight,
    /// Color values have already been multiplied by the stored alpha value.
    Premultiplied,
}

/// Failure at the lossless `jxr` to `image` ownership boundary.
#[derive(Debug, thiserror::Error)]
pub enum ImageAdapterError {
    /// Parsing, planning, decoding, or core layout validation failed.
    #[error(transparent)]
    Codec(#[from] JxrError),
    /// `image::DynamicImage` has no exact representation for this format.
    #[error("image::DynamicImage cannot represent {format:?} without conversion")]
    UnsupportedFormat {
        /// Exact JPEG XR output format that was rejected.
        format: PixelFormat,
    },
    /// The decoded owner is not one tightly packed image plane.
    #[error("invalid decoded layout for image::DynamicImage: {reason}")]
    InvalidLayout {
        /// Violated ownership or extent invariant.
        reason: &'static str,
    },
}

/// An `image::DynamicImage` plus JPEG XR metadata not represented by that type.
#[derive(Debug)]
pub struct ImageFrame {
    image: DynamicImage,
    info: ImageInfo,
    decoded_region: Rect,
    format: PixelFormat,
    report: DecodeReport,
    icc_profile: Option<Vec<u8>>,
    alpha: AlphaRepresentation,
}

impl ImageFrame {
    /// Adapted image pixels.
    #[must_use]
    pub const fn image(&self) -> &DynamicImage {
        &self.image
    }

    /// Consume the frame and return only its `image` owner.
    ///
    /// This discards the separate ICC, alpha-representation, and decode-route
    /// metadata. Retain the frame when those semantics matter.
    #[must_use]
    pub fn into_image(self) -> DynamicImage {
        self.image
    }

    /// Parsed JPEG XR source information.
    #[must_use]
    pub const fn info(&self) -> &ImageInfo {
        &self.info
    }

    /// Decoded source-space region represented by the image.
    #[must_use]
    pub const fn decoded_region(&self) -> Rect {
        self.decoded_region
    }

    /// Exact JPEG XR output format transferred into the image owner.
    #[must_use]
    pub const fn format(&self) -> PixelFormat {
        self.format
    }

    /// CPU or accelerator route used to produce the pixels.
    #[must_use]
    pub const fn report(&self) -> &DecodeReport {
        &self.report
    }

    /// Copied Annex-A ICC profile bytes, when present.
    #[must_use]
    pub fn icc_profile(&self) -> Option<&[u8]> {
        self.icc_profile.as_deref()
    }

    /// Alpha representation that `DynamicImage` itself cannot record.
    #[must_use]
    pub const fn alpha_representation(&self) -> AlphaRepresentation {
        self.alpha
    }
}

/// Decode a borrowed JPEG XR view and transfer supported pixels to `image`.
pub fn decode_view(
    view: &JxrView<'_>,
    request: &DecodeRequest,
) -> Result<ImageFrame, ImageAdapterError> {
    let icc_profile = view.icc_profile().map(<[u8]>::to_vec);
    let decoded = view.decoder().decode(request)?;
    into_image_frame(decoded, icc_profile.as_deref(), request.alpha)
}

/// Decode a prepared JPEG XR owner and transfer supported pixels to `image`.
pub fn decode_prepared(
    prepared: &PreparedJxr,
    request: &DecodeRequest,
) -> Result<ImageFrame, ImageAdapterError> {
    let icc_profile = prepared.icc_profile().map(<[u8]>::to_vec);
    let decoded = prepared.decoder().decode(request)?;
    into_image_frame(decoded, icc_profile.as_deref(), request.alpha)
}

/// Transfer a supported decoded allocation into `image::DynamicImage`.
///
/// Pixel data is not copied. ICC bytes are copied because [`DecodedImage`]
/// does not retain its compressed source owner.
pub fn into_image_frame(
    decoded: DecodedImage,
    icc_profile: Option<&[u8]>,
    alpha_handling: AlphaHandling,
) -> Result<ImageFrame, ImageAdapterError> {
    decoded.validate_layout()?;
    let [plane] = decoded.planes.as_slice() else {
        return Err(ImageAdapterError::UnsupportedFormat {
            format: decoded.format,
        });
    };
    let expected_stride = decoded.format.row_bytes(decoded.decoded_region.w)?;
    let expected_bytes = expected_stride
        .checked_mul(
            usize::try_from(decoded.decoded_region.h)
                .map_err(|_| invalid_layout("decoded height exceeds usize"))?,
        )
        .ok_or_else(|| invalid_layout("decoded byte extent overflows usize"))?;
    if plane.byte_offset != 0
        || plane.width != decoded.decoded_region.w
        || plane.height != decoded.decoded_region.h
        || plane.channels != decoded.format.channel_count()
        || plane.row_stride_bytes != expected_stride
        || decoded.samples.byte_len() != expected_bytes
    {
        return Err(invalid_layout("image plane is not tightly packed"));
    }
    let alpha = alpha_representation(decoded.format, &decoded.info, alpha_handling)?;
    let DecodedImage {
        info,
        decoded_region,
        format,
        samples,
        report,
        ..
    } = decoded;
    let width = decoded_region.w;
    let height = decoded_region.h;
    macro_rules! dynamic_image {
        ($variant:ident, $pixel:ty, $values:expr) => {
            ImageBuffer::<$pixel, _>::from_raw(width, height, $values)
                .map(DynamicImage::$variant)
                .ok_or_else(|| invalid_layout("sample count does not match image dimensions"))?
        };
    }
    let mut image = match (format, samples) {
        (PixelFormat::U8(ChannelLayout::Luma), DecodedSamples::U8(values)) => {
            dynamic_image!(ImageLuma8, Luma<u8>, values)
        }
        (PixelFormat::U8(ChannelLayout::LumaAlpha), DecodedSamples::U8(values)) => {
            dynamic_image!(ImageLumaA8, LumaA<u8>, values)
        }
        (PixelFormat::U8(ChannelLayout::Rgb), DecodedSamples::U8(values)) => {
            dynamic_image!(ImageRgb8, Rgb<u8>, values)
        }
        (PixelFormat::U8(ChannelLayout::Rgba), DecodedSamples::U8(values)) => {
            dynamic_image!(ImageRgba8, Rgba<u8>, values)
        }
        (PixelFormat::U16(ChannelLayout::Luma), DecodedSamples::U16(values)) => {
            dynamic_image!(ImageLuma16, Luma<u16>, values)
        }
        (PixelFormat::U16(ChannelLayout::LumaAlpha), DecodedSamples::U16(values)) => {
            dynamic_image!(ImageLumaA16, LumaA<u16>, values)
        }
        (PixelFormat::U16(ChannelLayout::Rgb), DecodedSamples::U16(values)) => {
            dynamic_image!(ImageRgb16, Rgb<u16>, values)
        }
        (PixelFormat::U16(ChannelLayout::Rgba), DecodedSamples::U16(values)) => {
            dynamic_image!(ImageRgba16, Rgba<u16>, values)
        }
        (PixelFormat::F32(ChannelLayout::Rgb), DecodedSamples::F32(values)) => {
            dynamic_image!(ImageRgb32F, Rgb<f32>, values)
        }
        (PixelFormat::F32(ChannelLayout::Rgba), DecodedSamples::F32(values)) => {
            dynamic_image!(ImageRgba32F, Rgba<f32>, values)
        }
        _ => return Err(ImageAdapterError::UnsupportedFormat { format }),
    };
    image.set_rgb_primaries(CicpColorPrimaries::Unspecified);
    image.set_transfer_function(CicpTransferCharacteristics::Unspecified);
    Ok(ImageFrame {
        image,
        info,
        decoded_region,
        format,
        report,
        icc_profile: icc_profile.map(<[u8]>::to_vec),
        alpha,
    })
}

fn alpha_representation(
    format: PixelFormat,
    info: &ImageInfo,
    handling: AlphaHandling,
) -> Result<AlphaRepresentation, ImageAdapterError> {
    let has_alpha = matches!(
        format,
        PixelFormat::U8(ChannelLayout::LumaAlpha | ChannelLayout::Rgba)
            | PixelFormat::U16(ChannelLayout::LumaAlpha | ChannelLayout::Rgba)
            | PixelFormat::F32(ChannelLayout::Rgba)
    );
    if !has_alpha {
        if info.premultiplied_alpha {
            return Err(invalid_layout(
                "premultiplied color has no alpha channel to describe it",
            ));
        }
        return Ok(AlphaRepresentation::None);
    }
    if handling == AlphaHandling::Drop {
        return Err(invalid_layout(
            "alpha-bearing output was paired with drop-alpha policy",
        ));
    }
    if handling == AlphaHandling::Premultiply || info.premultiplied_alpha {
        Ok(AlphaRepresentation::Premultiplied)
    } else {
        Ok(AlphaRepresentation::Straight)
    }
}

const fn invalid_layout(reason: &'static str) -> ImageAdapterError {
    ImageAdapterError::InvalidLayout { reason }
}
