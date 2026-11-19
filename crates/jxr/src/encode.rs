//! Public CPU encode entry points.

use jxr_core::{EncodeRequest, EncodedCodestreams, JxrError, SourceImage};
use jxr_native::CpuCapabilities;

/// Encode one image as an Annex-A still-image file, the usual `.jxr` form.
///
/// The container declares the source pixel format, so a decoder returns the
/// same channel layout and sample type that was encoded. An alpha channel is
/// coded as [`EncodeRequest::alpha_mode`] selects. As a decode does, the
/// encode runs on the vector instructions this process has and through the
/// thread pool where the target has threads.
///
/// ```
/// use jxr::{
///     ChannelLayout, DecodeRequest, DecodedSamples, EncodeRequest, JxrView, PixelFormat,
///     SourceImage, SourceSamples,
/// };
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let format = PixelFormat::U8(ChannelLayout::Rgb);
/// let rgb: Vec<u8> = (0..16 * 16 * 3).map(|value| (value % 251) as u8).collect();
/// let image = SourceImage::new(16, 16, format, SourceSamples::U8(&rgb))?;
/// let file = jxr::encode(&image, &EncodeRequest::lossless())?;
///
/// let decoded = JxrView::parse(&file)?.decoder().decode(&DecodeRequest::new(format))?;
/// assert_eq!(decoded.samples, DecodedSamples::U8(rgb));
/// # Ok(())
/// # }
/// ```
///
/// [`EncodeRequest::with_chroma_sampling`] codes the chroma of an RGB or
/// YUV 4:4:4 image at 4:2:2 or 4:2:0. An image that is YUV 4:2:2 or 4:2:0
/// already is given plane by plane, as the decoder returns it, and coded at
/// its own sampling:
///
/// ```
/// use jxr::{
///     ChannelLayout, ChromaSampling, DecodeRequest, DecodedSamples, EncodeRequest, JxrView,
///     PixelFormat, SourceImage, SourceSamples,
/// };
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let format = PixelFormat::U8(ChannelLayout::Yuv(ChromaSampling::Cs420));
/// // A 16 x 16 Y plane, then the 8 x 8 U and V planes.
/// let yuv: Vec<u8> = (0..16 * 16 + 2 * 8 * 8).map(|value| (value % 251) as u8).collect();
/// let image = SourceImage::new(16, 16, format, SourceSamples::U8(&yuv))?;
/// let file = jxr::encode(&image, &EncodeRequest::lossless())?;
///
/// let decoded = JxrView::parse(&file)?.decoder().decode(&DecodeRequest::new(format))?;
/// assert_eq!(decoded.planes.len(), 3);
/// assert_eq!(decoded.samples, DecodedSamples::U8(yuv));
/// # Ok(())
/// # }
/// ```
///
/// [`EncodeRequest::with_bitstream_mode`] codes each tile as one packet for
/// each frequency band, so that a decoder can reconstruct a reduced picture
/// from the leading packets and skip the rest:
///
/// ```
/// use jxr::{
///     BitstreamMode, ChannelLayout, DecodeRequest, DecodeScale, DecodedSamples, EncodeRequest,
///     JxrView, PixelFormat, SourceImage, SourceSamples,
/// };
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let format = PixelFormat::U8(ChannelLayout::Luma);
/// let gray = vec![90_u8; 64 * 48];
/// let image = SourceImage::new(64, 48, format, SourceSamples::U8(&gray))?;
/// let request = EncodeRequest::lossless().with_bitstream_mode(BitstreamMode::Frequency);
/// let file = jxr::encode(&image, &request)?;
///
/// let thumbnail = DecodeRequest::new(format).with_scale(DecodeScale::Sixteenth);
/// let decoded = JxrView::parse(&file)?.decoder().decode(&thumbnail)?;
/// assert_eq!((decoded.planes[0].width, decoded.planes[0].height), (4, 3));
/// assert_eq!(decoded.samples, DecodedSamples::U8(vec![90; 4 * 3]));
/// # Ok(())
/// # }
/// ```
pub fn encode(image: &SourceImage<'_>, request: &EncodeRequest) -> Result<Vec<u8>, JxrError> {
    jxr_native::encode_annex_a(image, request, CpuCapabilities::detect())
}

/// Encode one image as a single raw T.832 codestream.
///
/// The codestream can be decoded directly or placed in an Annex-A container
/// with [`crate::write_annex_a`]; [`encode`] does both. Source pixel formats
/// without an implemented forward conversion return
/// [`crate::JxrErrorKind::Unsupported`]; they are never approximated by a
/// nearby format.
///
/// A source with an alpha channel fits one codestream only as
/// [`crate::AlphaMode::Integrated`] alpha. The default,
/// [`crate::AlphaMode::Separate`], is a second codestream, so it returns
/// [`crate::JxrErrorKind::InvalidRequest`] here; [`encode_codestreams`]
/// returns both.
pub fn encode_codestream(
    image: &SourceImage<'_>,
    request: &EncodeRequest,
) -> Result<Vec<u8>, JxrError> {
    jxr_native::encode_codestream(image, request, CpuCapabilities::detect())
}

/// Encode one image as the raw T.832 codestreams an Annex-A container holds,
/// with the pixel-format identifier that declares its samples.
///
/// This is [`encode`] without the container, for callers that write their
/// own with [`crate::write_annex_a`], for example to attach an ICC profile.
///
/// ```
/// use jxr::{
///     AlphaMode, AnnexAWriteOptions, ChannelLayout, EncodeRequest, JxrView, PixelFormat,
///     SourceImage, SourceSamples,
/// };
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let format = PixelFormat::U8(ChannelLayout::Bgra);
/// let bgra: Vec<u8> = (0..16 * 16 * 4).map(|value| (value % 251) as u8).collect();
/// let image = SourceImage::new(16, 16, format, SourceSamples::U8(&bgra))?;
/// let encoded = jxr::encode_codestreams(&image, &EncodeRequest::lossless())?;
///
/// // Alpha is a codestream of its own unless the request integrates it.
/// let alpha = encoded.alpha.as_deref().expect("separate alpha by default");
/// let options = AnnexAWriteOptions::new(16, 16, encoded.pixel_format).with_separate_alpha(alpha);
/// let file = jxr::write_annex_a(&encoded.primary, &options)?;
/// assert_eq!(JxrView::parse(&file)?.info().alpha_mode, AlphaMode::Separate);
/// # Ok(())
/// # }
/// ```
pub fn encode_codestreams(
    image: &SourceImage<'_>,
    request: &EncodeRequest,
) -> Result<EncodedCodestreams, JxrError> {
    jxr_native::encode_codestreams(image, request, CpuCapabilities::detect())
}
