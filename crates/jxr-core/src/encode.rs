//! Encode request and borrowed source-image contracts.

use alloc::vec::Vec;

use crate::{
    AlphaMode, BandPresence, BitstreamMode, ChromaSampling, DecodedSamples, JxrError, JxrErrorKind,
    OverlapMode, PixelFormat, StorageKind,
};

/// T.832 quantization parameters for the three frequency bands.
///
/// A parameter of zero or one selects a quantization step of one. When every
/// band does, the encoder uses unscaled arithmetic and the image is coded
/// losslessly. Larger parameters quantize more coarsely; the step grows
/// linearly up to fifteen and then by a factor of two every sixteen values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EncodeQuantizers {
    pub dc: u8,
    pub low_pass: u8,
    pub high_pass: u8,
}

impl EncodeQuantizers {
    /// Step one in every band: mathematically lossless coding.
    pub const LOSSLESS: Self = Self::uniform(1);

    /// The same quantization parameter in every band.
    #[must_use]
    pub const fn uniform(parameter: u8) -> Self {
        Self {
            dc: parameter,
            low_pass: parameter,
            high_pass: parameter,
        }
    }

    /// Whether every band uses a quantization step of one.
    #[must_use]
    pub const fn is_lossless(self) -> bool {
        self.dc <= 1 && self.low_pass <= 1 && self.high_pass <= 1
    }
}

/// What sets the quantizers of an image: parameters given outright, or a
/// quality the encoder turns into parameters once it knows the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EncodeQuantization {
    /// T.832 quantization parameters, coded as given whatever the source.
    Quantizers(EncodeQuantizers),
    /// A quality on the scale of [`EncodeRequest::quality`]. 100, and
    /// anything above it, is lossless. A lower value has no parameters of
    /// its own: the encoder resolves it against the sample range and the
    /// colour format of each source it is given.
    Quality(u8),
}

impl EncodeQuantization {
    /// Step one in every band: mathematically lossless coding.
    pub const LOSSLESS: Self = Self::Quantizers(EncodeQuantizers::LOSSLESS);

    /// Whether this asks for a quantization step of one in every band,
    /// whatever the source: lossless parameters, or a quality of 100 or
    /// more.
    #[must_use]
    pub const fn is_lossless(self) -> bool {
        match self {
            Self::Quantizers(quantizers) => quantizers.is_lossless(),
            Self::Quality(quality) => quality >= 100,
        }
    }
}

/// Host encode request.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EncodeRequest {
    /// What sets the quantizers of the first component, which is luma in an
    /// RGB or YUV image, and of the other components and the alpha plane
    /// unless they are given their own.
    pub quantization: EncodeQuantization,
    /// Quantizers of every component but the first. `None` leaves them to
    /// [`Self::quantization`]: the same parameters as the first component,
    /// or what the quality scale gives each of them.
    pub chroma_quantizers: Option<EncodeQuantizers>,
    /// Overlap filtering applied before the core transform. `None` leaves
    /// the mode to the encoder: what the quality scale selects for
    /// [`EncodeQuantization::Quality`], and otherwise no overlap where every
    /// colour component is lossless and first-level overlap where one is
    /// not.
    pub overlap: Option<OverlapMode>,
    /// Frequency bands written to the codestream.
    pub bands: BandPresence,
    /// Low-order flexbits planes to discard, at most fifteen.
    pub trim_flexbits: u8,
    /// Largest tile edge in macroblocks. Each axis is split into the fewest
    /// near-equal tiles that respect it, so an image no larger than this in
    /// either direction is one tile.
    ///
    /// A tile is the only part of a JPEG XR image that is entropy coded, and
    /// decoded, independently of the rest, so the tile size decides how much
    /// of either can run in parallel. The default is
    /// [`Self::DEFAULT_MAX_TILE_MACROBLOCKS`]; 255 gives the largest tiles
    /// every level allows, and the smallest file.
    pub max_tile_macroblocks: u16,
    /// Stop overlap filtering at tile boundaries so tiles decode independently.
    pub hard_tiles: bool,
    /// `SHIFT_BITS` for 16-bit and 32-bit integer samples: low-order bits
    /// dropped before coding. `None` selects zero for 16-bit samples and
    /// [`Self::DEFAULT_I32_SHIFT_BITS`] for 32-bit samples.
    pub shift_bits: Option<u8>,
    /// Mantissa bits kept from 32-bit floating-point samples, at most 23.
    pub float_mantissa_bits: u8,
    /// Exponent bias of 32-bit floating-point samples: magnitudes below
    /// `2^(1 - bias)` are coded with fewer significant bits.
    pub float_exponent_bias: i8,
    /// Code the channels of a CMYK source as they are, without the colour
    /// transform, and declare the image `CMYKDIRECT`. Decoding such an image
    /// takes the Advanced profile rather than Main. Other sources ignore it.
    pub cmyk_direct: bool,
    /// How the alpha channel of a source that has one is coded. Sources
    /// without an alpha channel ignore it.
    ///
    /// [`AlphaMode::Separate`], the default, codes alpha as an image of its
    /// own beside the primary one, which an Annex-A file holds but a single
    /// codestream cannot. It is the form the T.835 reference encoder and
    /// jxrlib write, and it leaves the primary image decodable by readers
    /// that ignore alpha, at the profile the colour channels alone need.
    /// [`AlphaMode::Integrated`] codes alpha as a second plane of one
    /// codestream, which takes the Main profile or higher. It cannot be
    /// combined with [`Self::trim_flexbits`]: the reference decoders do not
    /// apply the trim to an integrated alpha plane, so such an image would
    /// decode nowhere else, and the encoder returns
    /// [`JxrErrorKind::Unsupported`] instead of writing it.
    /// [`AlphaMode::None`] is rejected for a source with an alpha channel:
    /// the channel is part of the declared pixel format and is never dropped.
    pub alpha_mode: AlphaMode,
    /// Quantizers of the alpha plane. `None` leaves them to
    /// [`Self::quantization`]: the same parameters as the first component,
    /// or what the quality scale gives a gray image of the alpha samples.
    pub alpha_quantizers: Option<EncodeQuantizers>,
    /// Chroma resolution of the coded image, for a source whose colour is
    /// coded as luma and full-resolution chroma: RGB at any sample depth,
    /// RGBE, and YUV 4:4:4.
    ///
    /// [`ChromaSampling::Cs444`], the default, codes chroma at full
    /// resolution. [`ChromaSampling::Cs422`] halves it horizontally and
    /// [`ChromaSampling::Cs420`] in both directions, with the low-pass filter
    /// of the T.835 reference encoder, and a decoder interpolates it back.
    /// The decimation loses information, so such an image is not reproduced
    /// exactly whatever its quantizers: with lossless ones each coded plane
    /// is exact, luma included, and only the chroma that was filtered away
    /// is lost.
    ///
    /// Second-level overlap filters subsampled chroma across pairs of
    /// macroblocks. With [`OverlapMode::Two`] the image must therefore be
    /// wider than one macroblock and, under [`Self::hard_tiles`], so must
    /// every tile, as the T.835 reference software asserts; otherwise
    /// encoding returns [`JxrErrorKind::InvalidRequest`]. An overlap mode
    /// left to the encoder is never one the image cannot take.
    ///
    /// Sources without chroma of that kind ignore the option: gray, CMYK
    /// and N-component images, which are not coded as luma and two chroma
    /// components, and YUV 4:2:0 and 4:2:2 images, which are coded at their
    /// own sampling.
    pub chroma_sampling: ChromaSampling,
    /// How the coefficients of each tile are laid out in the codestream.
    ///
    /// [`BitstreamMode::Spatial`], the default, codes a tile as one packet
    /// that interleaves the bands of each macroblock. It is the layout most
    /// files use and a decoder reads front to back.
    /// [`BitstreamMode::Frequency`] codes a tile as one packet per band
    /// instead: DC, low-pass, high-pass, then flexbits, as far as
    /// [`Self::bands`] goes. A decoder can then stop after the DC or the
    /// low-pass packets and reconstruct the image at a sixteenth or a
    /// quarter of its size without reading the rest, which is what
    /// [`DecodeScale::Sixteenth`](crate::DecodeScale::Sixteenth) and
    /// [`DecodeScale::Quarter`](crate::DecodeScale::Quarter) need.
    ///
    /// Both layouts carry the same coefficients, so the choice changes
    /// neither the decoded samples nor the profile and level. A
    /// frequency-mode codestream always has a tile index, with one entry
    /// per band packet, and is a few bytes per tile and band larger.
    pub bitstream_mode: BitstreamMode,
}

impl EncodeRequest {
    /// Default largest tile edge: 32 macroblocks, 512 samples.
    ///
    /// The value is fixed, whatever machine encodes, so the same request
    /// gives the same bytes everywhere. It trades file size for parallel
    /// entropy coding in the encoder and in every decoder of the file. With
    /// soft tiles the transforms still run across tile edges, so the tile
    /// size changes how the coefficients are predicted and coded but not the
    /// decoded samples. A twelve-megapixel image is 48 tiles instead of one
    /// and both encodes and decodes several times faster on a multi-core
    /// machine; photographs grow by about 0.7 percent coded losslessly and
    /// 0.5 percent at quality 75. Halving the tile edge again gains a few
    /// percent of speed at twice that cost, and images of up to 512 samples
    /// a side stay one tile. `docs/support.md` has the measurements.
    ///
    /// No tile count this produces raises the level an image declares: every
    /// T.832 level allows a tile for each 64 samples of its largest image
    /// edge. Pass 255 to [`Self::with_max_tile_macroblocks`] for the largest
    /// tiles every level from 16 up allows, which must be narrower and
    /// shorter than 4096 samples: one tile for most images, and the smallest
    /// file.
    pub const DEFAULT_MAX_TILE_MACROBLOCKS: u16 = 32;

    /// Bits dropped from 32-bit integer samples by default. Thirty-two
    /// significant bits leave the transforms no headroom in 32-bit
    /// arithmetic; ten is the value the T.835 reference software uses.
    pub const DEFAULT_I32_SHIFT_BITS: u8 = 10;
    /// Default mantissa bits kept from 32-bit floating-point samples.
    pub const DEFAULT_FLOAT_MANTISSA_BITS: u8 = 13;
    /// Default exponent bias of 32-bit floating-point samples.
    pub const DEFAULT_FLOAT_EXPONENT_BIAS: i8 = 4;

    /// Lossless coding of every band without overlap filtering.
    #[must_use]
    pub const fn lossless() -> Self {
        Self {
            quantization: EncodeQuantization::LOSSLESS,
            chroma_quantizers: None,
            overlap: Some(OverlapMode::None),
            bands: BandPresence::All,
            trim_flexbits: 0,
            max_tile_macroblocks: Self::DEFAULT_MAX_TILE_MACROBLOCKS,
            hard_tiles: false,
            shift_bits: None,
            float_mantissa_bits: Self::DEFAULT_FLOAT_MANTISSA_BITS,
            float_exponent_bias: Self::DEFAULT_FLOAT_EXPONENT_BIAS,
            cmyk_direct: false,
            alpha_mode: AlphaMode::Separate,
            alpha_quantizers: None,
            chroma_sampling: ChromaSampling::Cs444,
            bitstream_mode: BitstreamMode::Spatial,
        }
    }

    /// Lossy coding with the given quantizers and first-level overlap
    /// filtering, which suppresses blocking at coarse quantization.
    #[must_use]
    pub const fn lossy(quantizers: EncodeQuantizers) -> Self {
        Self {
            quantization: EncodeQuantization::Quantizers(quantizers),
            overlap: Some(OverlapMode::One),
            ..Self::lossless()
        }
    }

    /// Coding at a quality from 0 to 100, with the feel of JPEG's scale.
    ///
    /// 100, and anything above it, is [`Self::lossless`]. Every lower value
    /// is lossy, and never asks for finer quantization than the value above
    /// it: files shrink and error grows as the quality falls. As a guide, 90
    /// stays close to the source, 75 is a sound default, 50 is visibly
    /// compressed, and values below 10 trade most detail for size.
    ///
    /// A quality is a statement about the picture, not about quantization
    /// parameters, so the request stores the number and the encoder resolves
    /// it against each source. The scale is calibrated on 8-bit pictures so
    /// that luma PSNR tracks baseline JPEG at the same number. For another
    /// source its quantization steps are scaled to the range of the samples,
    /// and each coded component takes the step its part in the colour format
    /// calls for. The same picture then loses about the same share of its
    /// sample range, in a file of comparable size, whether it is stored in 8
    /// or 16 bits, in fixed or floating point, as gray, RGB, CMYK or separate
    /// channels, and with or without [`Self::chroma_sampling`]. The alpha
    /// plane of a source that has one is coded as a gray image at the same
    /// quality. `docs/support.md` gives the rule and the sizes and errors
    /// measured for each kind of source.
    ///
    /// The scale selects the quantizers of every component and band and the
    /// overlap mode; other options keep their [`Self::lossless`] defaults
    /// and can be changed afterwards. [`Self::with_chroma_quantizers`],
    /// [`Self::with_alpha_quantizers`] and [`Self::with_overlap`] replace
    /// the scale's choice for what they name, and [`Self::lossy`] gives
    /// every quantizer directly.
    ///
    /// Not every source can tell every quality apart, because quantization
    /// parameters are whole numbers:
    ///
    /// - A source with fewer sample values than a quality's step needs is
    ///   coded at the nearest step it has, which near the top of the scale
    ///   is the lossless one. Packed 5-bit and 6-bit samples are lossless
    ///   down to about 91 and 96, and black-and-white images at every
    ///   quality.
    /// - Where the scale is flattest it moves the chroma of an RGB or CMYK
    ///   image and everything else on alternate qualities. A gray image, an
    ///   image coded channel by channel, and a float image, whose chroma
    ///   moves with its luma, are coded identically on those pairs.
    /// - RGBE images are coded at quality 10 from there down.
    #[must_use]
    pub const fn quality(quality: u8) -> Self {
        if quality >= 100 {
            return Self::lossless();
        }
        Self {
            quantization: EncodeQuantization::Quality(quality),
            overlap: None,
            ..Self::lossless()
        }
    }

    /// Give every component but the first its own quantizers: the chroma
    /// components of a colour image, and every further channel of an image
    /// coded channel by channel.
    #[must_use]
    pub const fn with_chroma_quantizers(mut self, quantizers: EncodeQuantizers) -> Self {
        self.chroma_quantizers = Some(quantizers);
        self
    }

    /// What sets the quantizers of component `component` of the internal
    /// colour format: [`Self::chroma_quantizers`] for every component but
    /// the first when they are given, and [`Self::quantization`] otherwise.
    #[must_use]
    pub const fn component_quantization(&self, component: usize) -> EncodeQuantization {
        match self.chroma_quantizers {
            Some(chroma) if component != 0 => EncodeQuantization::Quantizers(chroma),
            _ => self.quantization,
        }
    }

    /// Whether the request asks for a quantization step of one in every
    /// colour component and in the alpha plane, whatever the source. A
    /// source without an alpha channel codes no alpha plane, so for it
    /// [`Self::colour_is_lossless`] alone decides. The image is then
    /// reproduced exactly, unless [`Self::chroma_sampling`] has the encoder
    /// discard chroma resolution first.
    ///
    /// A quality below 100 is not lossless by this measure, although the
    /// encoder codes a source exactly where its samples have no coarser
    /// step to offer; see [`Self::quality`].
    #[must_use]
    pub const fn is_lossless(&self) -> bool {
        self.colour_is_lossless() && self.alpha_quantization().is_lossless()
    }

    /// Whether the request asks for a quantization step of one in every
    /// colour component, whatever the alpha plane uses. The colour
    /// components then use unscaled arithmetic.
    #[must_use]
    pub const fn colour_is_lossless(&self) -> bool {
        self.quantization.is_lossless()
            && match self.chroma_quantizers {
                Some(chroma) => chroma.is_lossless(),
                None => true,
            }
    }

    /// What sets the quantizers of the alpha plane, which is scaled or
    /// unscaled by its own losslessness: [`Self::alpha_quantizers`] when
    /// they are given, and [`Self::quantization`] otherwise.
    #[must_use]
    pub const fn alpha_quantization(&self) -> EncodeQuantization {
        match self.alpha_quantizers {
            Some(alpha) => EncodeQuantization::Quantizers(alpha),
            None => self.quantization,
        }
    }

    /// Select the overlap filtering mode.
    #[must_use]
    pub const fn with_overlap(mut self, overlap: OverlapMode) -> Self {
        self.overlap = Some(overlap);
        self
    }

    /// Select the frequency bands written to the codestream.
    #[must_use]
    pub const fn with_bands(mut self, bands: BandPresence) -> Self {
        self.bands = bands;
        self
    }

    /// Discard low-order flexbits planes.
    #[must_use]
    pub const fn with_trim_flexbits(mut self, trim_flexbits: u8) -> Self {
        self.trim_flexbits = trim_flexbits;
        self
    }

    /// Set the largest tile edge in macroblocks; see
    /// [`Self::max_tile_macroblocks`]. 255 restores the single tile of most
    /// images that was the default before tiles were sized for parallel
    /// coding.
    #[must_use]
    pub const fn with_max_tile_macroblocks(mut self, macroblocks: u16) -> Self {
        self.max_tile_macroblocks = macroblocks;
        self
    }

    /// Stop or continue overlap filtering across tile boundaries.
    #[must_use]
    pub const fn with_hard_tiles(mut self, hard_tiles: bool) -> Self {
        self.hard_tiles = hard_tiles;
        self
    }

    /// Set `SHIFT_BITS` for 16-bit and 32-bit integer samples.
    #[must_use]
    pub const fn with_shift_bits(mut self, shift_bits: u8) -> Self {
        self.shift_bits = Some(shift_bits);
        self
    }

    /// Set the mantissa bits and exponent bias of 32-bit floating-point samples.
    #[must_use]
    pub const fn with_float_precision(mut self, mantissa_bits: u8, exponent_bias: i8) -> Self {
        self.float_mantissa_bits = mantissa_bits;
        self.float_exponent_bias = exponent_bias;
        self
    }

    /// Code CMYK channels directly or through the colour transform.
    #[must_use]
    pub const fn with_cmyk_direct(mut self, cmyk_direct: bool) -> Self {
        self.cmyk_direct = cmyk_direct;
        self
    }

    /// Select how the alpha channel of a source that has one is coded.
    #[must_use]
    pub const fn with_alpha_mode(mut self, alpha_mode: AlphaMode) -> Self {
        self.alpha_mode = alpha_mode;
        self
    }

    /// Give the alpha plane its own quantizers, so alpha can stay lossless
    /// under lossy colour or the reverse.
    #[must_use]
    pub const fn with_alpha_quantizers(mut self, quantizers: EncodeQuantizers) -> Self {
        self.alpha_quantizers = Some(quantizers);
        self
    }

    /// Select the chroma resolution RGB, RGBE and YUV 4:4:4 sources are
    /// coded at.
    #[must_use]
    pub const fn with_chroma_sampling(mut self, chroma_sampling: ChromaSampling) -> Self {
        self.chroma_sampling = chroma_sampling;
        self
    }

    /// Select the packet layout of each tile.
    #[must_use]
    pub const fn with_bitstream_mode(mut self, bitstream_mode: BitstreamMode) -> Self {
        self.bitstream_mode = bitstream_mode;
        self
    }

    /// Reject parameter values the codestream syntax cannot carry.
    pub fn validate(&self) -> Result<(), JxrError> {
        if self.trim_flexbits > 15 {
            return Err(JxrError::new(
                JxrErrorKind::InvalidRequest,
                "encode trim flexbits",
            ));
        }
        if matches!(self.shift_bits, Some(bits) if bits > 31) {
            return Err(JxrError::new(
                JxrErrorKind::InvalidRequest,
                "encode shift bits",
            ));
        }
        if self.float_mantissa_bits > 23 {
            return Err(JxrError::new(
                JxrErrorKind::InvalidRequest,
                "encode float mantissa bits",
            ));
        }
        if self.max_tile_macroblocks == 0 {
            return Err(JxrError::new(
                JxrErrorKind::InvalidRequest,
                "encode tile size",
            ));
        }
        Ok(())
    }
}

/// Borrowed samples of a source image, typed by their storage.
///
/// The variants mirror [`DecodedSamples`], so anything the decoder returns can
/// be handed back to the encoder.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum SourceSamples<'a> {
    /// One bit per pixel, most significant bit first, rows padded to bytes.
    BitPacked(&'a [u8]),
    U8(&'a [u8]),
    U16(&'a [u16]),
    I16(&'a [i16]),
    I32(&'a [i32]),
    /// IEEE half-precision bit patterns.
    F16(&'a [u16]),
    F32(&'a [f32]),
    Rgb555(&'a [u16]),
    Rgb565(&'a [u16]),
    Rgb101010(&'a [u32]),
    Rgbe(&'a [u32]),
}

crate::impl_sample_storage!(SourceSamples<'_>);

impl<'a> From<&'a DecodedSamples> for SourceSamples<'a> {
    fn from(samples: &'a DecodedSamples) -> Self {
        match samples {
            DecodedSamples::BitPacked(samples) => Self::BitPacked(samples),
            DecodedSamples::U8(samples) => Self::U8(samples),
            DecodedSamples::U16(samples) => Self::U16(samples),
            DecodedSamples::I16(samples) => Self::I16(samples),
            DecodedSamples::I32(samples) => Self::I32(samples),
            DecodedSamples::F16(samples) => Self::F16(samples),
            DecodedSamples::F32(samples) => Self::F32(samples),
            DecodedSamples::Rgb555(samples) => Self::Rgb555(samples),
            DecodedSamples::Rgb565(samples) => Self::Rgb565(samples),
            DecodedSamples::Rgb101010(samples) => Self::Rgb101010(samples),
            DecodedSamples::Rgbe(samples) => Self::Rgbe(samples),
        }
    }
}

/// Where one plane of a planar source image lies within its samples: the
/// encoder's counterpart of [`crate::PlaneDescriptor`], in storage elements
/// instead of bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SourcePlane {
    /// Storage elements before the first sample of the plane.
    pub offset: usize,
    /// Distance between the starts of consecutive rows of the plane, in
    /// storage elements.
    pub row_stride: usize,
}

/// One source image borrowed by the encoder.
///
/// The samples of a pixel are interleaved in every format but YUV 4:2:0 and
/// 4:2:2, with or without alpha. Those are planar, as the decoder returns
/// them: a Y plane, a U plane and a V plane half as wide, and for 4:2:0 half
/// as tall, as the image, and then an alpha plane when the format has one.
/// Their dimensions must be even along each halved axis, which the T.835
/// reference software asserts of an image whose output colour is subsampled.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceImage<'a> {
    pub width: u32,
    pub height: u32,
    /// Sample type and channel order of `samples`.
    pub format: PixelFormat,
    pub samples: SourceSamples<'a>,
    /// Distance between the starts of consecutive rows of an interleaved
    /// image, in storage elements. A planar image gives each plane's in
    /// [`Self::planes`] instead.
    pub row_stride: usize,
    /// Location of each plane of a planar image, Y, U, V and then alpha;
    /// entries beyond the format's planes are unused. `None` for an
    /// interleaved image.
    pub planes: Option<[SourcePlane; 4]>,
    /// Significant bits of each sample when fewer than its storage holds.
    /// `None` uses the full storage width. Ten is the one narrower depth
    /// container formats declare, for YUV in 16-bit storage; a sample above
    /// the depth's range is then rejected.
    pub sample_bits: Option<u8>,
    /// Horizontal and vertical T.832 chroma-centering codes of a planar
    /// image: the offset of its chroma grid from its luma grid in quarter
    /// samples, zero to four, or seven when it is not known. Zero, the
    /// default, has each chroma sample co-sited with the first luma sample
    /// it covers. The codes are declared as given, as
    /// [`crate::PlaneInfo::chroma_centering`] reports them for a decoded
    /// image, and do not change how the samples are coded. An interleaved
    /// image has none, and neither has the vertical axis of 4:2:2.
    pub chroma_centering: [u8; 2],
    /// Whether the colour samples are already multiplied by alpha. The
    /// samples are coded as given and the image is declared premultiplied,
    /// as [`crate::ImageInfo::premultiplied_alpha`] reports for a decoded
    /// one. Only a format with an alpha channel can be premultiplied.
    pub premultiplied_alpha: bool,
}

impl<'a> SourceImage<'a> {
    /// Describe tightly packed samples: rows of `width * channels` samples
    /// for an interleaved format, and for a planar one each plane's rows of
    /// its own width, one plane directly after another.
    pub fn new(
        width: u32,
        height: u32,
        format: PixelFormat,
        samples: SourceSamples<'a>,
    ) -> Result<Self, JxrError> {
        let mut image = Self {
            width,
            height,
            format,
            samples,
            row_stride: row_samples(width, format)?,
            planes: None,
            sample_bits: None,
            chroma_centering: [0, 0],
            premultiplied_alpha: false,
        };
        if let Some(count) = image.plane_count() {
            let mut planes = [SourcePlane::default(); 4];
            let mut offset = 0_usize;
            for (index, plane) in planes.iter_mut().enumerate().take(count) {
                let (plane_width, plane_height) = image.plane_dimensions(index);
                let plane_width = usize::try_from(plane_width)
                    .map_err(|_| JxrError::arithmetic("source image width"))?;
                let plane_height = usize::try_from(plane_height)
                    .map_err(|_| JxrError::arithmetic("source image height"))?;
                *plane = SourcePlane {
                    offset,
                    row_stride: plane_width,
                };
                offset = plane_width
                    .checked_mul(plane_height)
                    .and_then(|samples| samples.checked_add(offset))
                    .ok_or_else(|| JxrError::arithmetic("source image extent"))?;
            }
            image.planes = Some(planes);
        }
        image.validate()?;
        Ok(image)
    }

    /// Declare whether the colour samples are already multiplied by alpha.
    pub fn with_premultiplied_alpha(mut self, premultiplied_alpha: bool) -> Result<Self, JxrError> {
        self.premultiplied_alpha = premultiplied_alpha;
        self.validate()?;
        Ok(self)
    }

    /// Describe rows of an interleaved image separated by `row_stride`
    /// samples, which may exceed the samples each row uses.
    pub fn with_row_stride(mut self, row_stride: usize) -> Result<Self, JxrError> {
        if self.planes.is_some() {
            return Err(JxrError::new(
                JxrErrorKind::InvalidRequest,
                "row stride of a planar source image",
            ));
        }
        self.row_stride = row_stride;
        self.validate()?;
        Ok(self)
    }

    /// Describe where each plane of a planar image lies within the samples:
    /// one entry per plane, Y, U, V and then alpha. Planes may be separated
    /// and their rows padded.
    pub fn with_planes(mut self, planes: &[SourcePlane]) -> Result<Self, JxrError> {
        if self.plane_count() != Some(planes.len()) {
            return Err(JxrError::new(
                JxrErrorKind::InvalidRequest,
                "source image plane count",
            ));
        }
        let mut located = [SourcePlane::default(); 4];
        located[..planes.len()].copy_from_slice(planes);
        self.planes = Some(located);
        self.validate()?;
        Ok(self)
    }

    /// Declare the significant bits of each sample.
    pub fn with_sample_bits(mut self, sample_bits: u8) -> Result<Self, JxrError> {
        self.sample_bits = Some(sample_bits);
        self.validate()?;
        Ok(self)
    }

    /// Declare the chroma-centering codes of a planar image.
    pub fn with_chroma_centering(mut self, x: u8, y: u8) -> Result<Self, JxrError> {
        self.chroma_centering = [x, y];
        self.validate()?;
        Ok(self)
    }

    /// Samples used by each row of an interleaved image: `width * channels`.
    pub fn row_samples(&self) -> Result<usize, JxrError> {
        row_samples(self.width, self.format)
    }

    /// The chroma sampling of a planar format, and `None` for an interleaved
    /// one.
    #[must_use]
    pub const fn planar_sampling(&self) -> Option<ChromaSampling> {
        self.format.planar_sampling()
    }

    /// Number of planes of a planar format: three, or four with alpha.
    /// `None` for an interleaved format.
    #[must_use]
    pub const fn plane_count(&self) -> Option<usize> {
        match self.planar_sampling() {
            Some(_) => Some(self.format.channel_count() as usize),
            None => None,
        }
    }

    /// Width and height in samples of plane `plane` of a planar image: the
    /// image's own for Y and alpha, and for U and V half its width and, for
    /// 4:2:0, half its height. An interleaved image has one plane of its own
    /// size.
    #[must_use]
    pub const fn plane_dimensions(&self, plane: usize) -> (u32, u32) {
        match (self.planar_sampling(), plane) {
            (Some(ChromaSampling::Cs420), 1 | 2) => (self.width / 2, self.height / 2),
            (Some(ChromaSampling::Cs422), 1 | 2) => (self.width / 2, self.height),
            _ => (self.width, self.height),
        }
    }

    /// Check that the exact storage variant matches the format, the
    /// dimensions are non-zero and even where a planar format halves them,
    /// every row of every plane lies inside `samples`, and the declarations
    /// fit the format: only a format with an alpha channel is premultiplied,
    /// only 16-bit unsigned storage holds narrower samples, and only a
    /// planar format has chroma centering.
    pub fn validate(&self) -> Result<(), JxrError> {
        if self.premultiplied_alpha && !self.format.has_alpha() {
            return Err(JxrError::new(
                JxrErrorKind::InvalidRequest,
                "premultiplied source without an alpha channel",
            ));
        }
        if !self.samples.matches_format(self.format) {
            return Err(JxrError::new(
                JxrErrorKind::InvalidRequest,
                "source sample storage differs from its pixel format",
            ));
        }
        if self.width == 0 || self.height == 0 {
            return Err(JxrError::new(
                JxrErrorKind::InvalidRequest,
                "source image dimensions",
            ));
        }
        if self.sample_bits.is_some_and(|bits| {
            !(1..=16).contains(&bits) || self.format.storage_kind() != StorageKind::U16
        }) {
            return Err(JxrError::new(
                JxrErrorKind::InvalidRequest,
                "source sample depth",
            ));
        }
        self.validate_chroma_centering()?;
        let available = self.samples.len();
        let within = |offset: usize, row_stride: usize, row_samples: usize, rows: u32| {
            if row_stride < row_samples {
                return Err(JxrError::new(
                    JxrErrorKind::InvalidRequest,
                    "source image row stride",
                ));
            }
            let preceding_rows = usize::try_from(rows - 1)
                .map_err(|_| JxrError::arithmetic("source image height"))?;
            let required = row_stride
                .checked_mul(preceding_rows)
                .and_then(|samples| samples.checked_add(row_samples))
                .and_then(|samples| samples.checked_add(offset))
                .ok_or_else(|| JxrError::arithmetic("source image extent"))?;
            if required > available {
                return Err(JxrError::new(
                    JxrErrorKind::BufferTooSmall {
                        required,
                        available,
                    },
                    "source image samples",
                ));
            }
            Ok(())
        };
        match (self.planar_sampling(), &self.planes) {
            (None, None) => within(0, self.row_stride, self.row_samples()?, self.height),
            (Some(sampling), Some(planes)) => {
                let odd_height =
                    sampling == ChromaSampling::Cs420 && !self.height.is_multiple_of(2);
                if !self.width.is_multiple_of(2) || odd_height {
                    return Err(JxrError::new(
                        JxrErrorKind::InvalidRequest,
                        "odd dimension of a subsampled source image",
                    ));
                }
                let count = usize::from(self.format.channel_count());
                for (index, plane) in planes.iter().enumerate().take(count) {
                    let (width, height) = self.plane_dimensions(index);
                    let width = usize::try_from(width)
                        .map_err(|_| JxrError::arithmetic("source image width"))?;
                    within(plane.offset, plane.row_stride, width, height)?;
                }
                Ok(())
            }
            _ => Err(JxrError::new(
                JxrErrorKind::InvalidRequest,
                "source image plane layout differs from its pixel format",
            )),
        }
    }

    /// Chroma centering is a code from zero to four, or seven, on each axis
    /// a planar format subsamples, and zero elsewhere.
    fn validate_chroma_centering(&self) -> Result<(), JxrError> {
        let [x, y] = self.chroma_centering;
        let code = |value: u8| value <= 4 || value == 7;
        let valid = match self.planar_sampling() {
            Some(ChromaSampling::Cs420) => code(x) && code(y),
            Some(_) => code(x) && y == 0,
            None => x == 0 && y == 0,
        };
        if valid {
            Ok(())
        } else {
            Err(JxrError::new(
                JxrErrorKind::InvalidRequest,
                "source chroma centering",
            ))
        }
    }
}

/// The raw codestreams of one encoded image and the container pixel format
/// that declares its samples: everything an Annex-A container is written
/// from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedCodestreams {
    /// The colour image, with the alpha plane inside it when the request
    /// selected [`AlphaMode::Integrated`].
    pub primary: Vec<u8>,
    /// The alpha plane as a Y-only image of its own, present when the source
    /// has an alpha channel and the request selected [`AlphaMode::Separate`].
    pub alpha: Option<Vec<u8>>,
    /// Annex-A pixel-format identifier of the source samples. With separate
    /// alpha it alone declares the alpha channel and whether the colour
    /// samples are premultiplied by it.
    pub pixel_format: [u8; 16],
}

/// Storage elements one row occupies ([`PixelFormat::row_elements`]).
fn row_samples(width: u32, format: PixelFormat) -> Result<usize, JxrError> {
    format
        .row_elements(width)
        .ok_or_else(|| JxrError::arithmetic("source image row length"))
}
