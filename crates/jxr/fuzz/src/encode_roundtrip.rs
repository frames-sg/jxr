//! Harness of the `encode_roundtrip` target: an image and an encode request
//! derived from fuzz input, and the properties every encode of them has.
//!
//! - Encoding never panics, is deterministic, and never reports an internal
//!   invariant: it returns a file or an error that names what was refused.
//! - A plain lossless encode of a format the container table declares
//!   succeeds.
//! - Every file parses, reports the source dimensions, and decodes on the
//!   CPU in the source's own pixel format.
//! - When the request is lossless for the format, the decoded samples are
//!   the source's: all bands, no trimmed flexbits, lossless quantizers, no
//!   chroma the encoder subsamples, and samples the coded form represents.
//!   Colour and alpha are each held to it by their own quantizers.

mod formats;
mod samples;

use std::fmt;

use jxr::{
    AlphaMode, BackendRequest, BandPresence, BitstreamMode, ChannelLayout, ChromaSampling,
    DecodeRequest, DecodedSamples, EncodeQuantizers, EncodeRequest, JxrError, JxrErrorKind,
    JxrView, OverlapMode, PixelFormat, SourceImage, SourcePlane, SourceSamples,
};

use formats::{declared_formats, undeclared_format};
use jxr_test_support::decimates_chroma;
use samples::{Pattern, loosen, pattern_samples, size};

/// Largest width and height of an image: four macroblocks, enough for
/// several tiles along each axis while a case stays a few hundred kilobytes.
const MAX_EDGE: u8 = 64;

/// Run one fuzz input: build the case it describes and check every property.
pub fn encode_roundtrip(bytes: &[u8]) {
    check(&Case::from_bytes(bytes));
}

/// Fuzz input read as a stream of small decisions. An exhausted input reads
/// as zeros, which select the plainest choice everywhere, so every input is
/// a case and a short input is a simple one.
struct Input<'a> {
    bytes: &'a [u8],
}

impl Input<'_> {
    fn byte(&mut self) -> u8 {
        let Some((&first, rest)) = self.bytes.split_first() else {
            return 0;
        };
        self.bytes = rest;
        first
    }

    /// A value below `bound`.
    fn below(&mut self, bound: u8) -> u8 {
        self.byte() % bound
    }

    /// True for one byte value in sixteen, for choices that mostly end in a
    /// refusal.
    fn rarely(&mut self) -> bool {
        self.byte() % 16 == 15
    }

    fn word(&mut self) -> u64 {
        u64::from_le_bytes(std::array::from_fn(|_| self.byte()))
    }
}

/// Where the rows of an image lie in its storage.
#[derive(Debug, Clone)]
enum Placement {
    /// Rows, and the planes of a planar image, directly after one another.
    Tight,
    /// Rows of an interleaved image this many elements apart.
    Rows(usize),
    /// Planes of a planar image anywhere, with rows apart.
    Planes(Vec<SourcePlane>),
}

/// One image and the request to encode it with.
#[derive(Clone)]
struct Case {
    format: PixelFormat,
    sample_bits: Option<u8>,
    width: u32,
    height: u32,
    request: EncodeRequest,
    /// The image with its rows and planes directly after one another, which
    /// is how a decoder returns it.
    samples: DecodedSamples,
    /// The storage the encoder reads: `samples`, laid out by `placement`.
    stored: DecodedSamples,
    placement: Placement,
    premultiplied: bool,
    chroma_centering: [u8; 2],
    /// Whether every sample is one the coded form of the format represents
    /// under the request, so that lossless coding reproduces it.
    representable: bool,
    /// Whether the format is declared, the samples representable, and both
    /// image and request plain: nothing may refuse such a case.
    must_encode: bool,
}

impl fmt::Display for Case {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:?} ({:?} bits) {}x{}, {:?}, premultiplied {}, centering {:?}, representable {}, \
             {:?}",
            self.format,
            self.sample_bits,
            self.width,
            self.height,
            self.placement,
            self.premultiplied,
            self.chroma_centering,
            self.representable,
            self.request
        )
    }
}

/// What a case is assembled from.
struct Recipe<'a> {
    format: (PixelFormat, Option<u8>),
    dimensions: (u32, u32),
    request: EncodeRequest,
    representable: bool,
    /// Unused elements after each row and before each plane; none stores
    /// the image tight.
    loose: u8,
    premultiplied: bool,
    chroma_centering: [u8; 2],
    pattern: Pattern<'a>,
    must_encode: bool,
}

/// Any quantizer of each band.
fn quantizers(input: &mut Input<'_>) -> EncodeQuantizers {
    EncodeQuantizers {
        dc: input.byte(),
        low_pass: input.byte(),
        high_pass: input.byte(),
    }
}

/// An encode request, and whether it is the plain lossless one.
fn request(input: &mut Input<'_>) -> (EncodeRequest, bool) {
    let mode = input.below(4);
    // One bit per option that is set apart from the constructor's choice.
    let options = u16::from_le_bytes([input.byte(), input.byte()]);
    let mut request = match mode {
        0 => EncodeRequest::lossless(),
        1 => EncodeRequest::quality(input.byte()),
        2 => EncodeRequest::lossy(EncodeQuantizers::uniform(input.byte())),
        _ => EncodeRequest::lossy(quantizers(input)),
    };
    let chosen = |bit: u32| options >> bit & 1 == 1;
    if chosen(0) {
        request.chroma_quantizers = Some(quantizers(input));
    }
    if chosen(1) {
        request.alpha_quantizers = Some(quantizers(input));
    }
    // The last choice leaves the mode to the encoder, whatever sets the
    // quantizers.
    if chosen(2) {
        request.overlap = match input.below(4) {
            0 => Some(OverlapMode::None),
            1 => Some(OverlapMode::One),
            2 => Some(OverlapMode::Two),
            _ => None,
        };
    }
    if chosen(3) {
        request.bands = match input.below(4) {
            0 => BandPresence::All,
            1 => BandPresence::NoFlexbits,
            2 => BandPresence::NoHighPass,
            _ => BandPresence::DcOnly,
        };
    }
    // One past the largest trim, and no tile at all, are refused.
    if chosen(4) {
        request.trim_flexbits = input.below(17);
    }
    if chosen(5) {
        request.max_tile_macroblocks = u16::from(input.below(5));
    }
    request.hard_tiles = chosen(6);
    if chosen(7) {
        request.alpha_mode = match input.below(3) {
            0 => AlphaMode::Separate,
            1 => AlphaMode::Integrated,
            _ => AlphaMode::None,
        };
    }
    if chosen(8) {
        request.chroma_sampling = match input.below(3) {
            0 => ChromaSampling::Cs444,
            1 => ChromaSampling::Cs422,
            _ => ChromaSampling::Cs420,
        };
    }
    request.cmyk_direct = chosen(9);
    // Shifts and mantissas one past the largest are refused.
    if chosen(10) {
        request.shift_bits = Some(input.below(33));
    }
    if chosen(11) {
        request.float_mantissa_bits = input.below(25);
        request.float_exponent_bias = input.byte().cast_signed();
    }
    if chosen(12) {
        request.bitstream_mode = BitstreamMode::Frequency;
    }
    (request, mode == 0 && options == 0)
}

impl Case {
    /// The case `bytes` describe. Each decision reads its own bytes in a
    /// fixed order, so a mutation of one byte changes one thing about the
    /// case.
    fn from_bytes(bytes: &[u8]) -> Self {
        let mut input = Input { bytes };
        let declared = declared_formats();
        let selector = usize::from(input.byte());
        let undeclared = input.rarely();
        let (format, sample_bits) = if undeclared {
            undeclared_format(&mut input)
        } else {
            declared[selector % declared.len()]
        };
        let mut width = 1 + u32::from(input.below(MAX_EDGE));
        let mut height = 1 + u32::from(input.below(MAX_EDGE));
        // A subsampled plane needs an even dimension to halve; an odd one
        // is refused.
        let odd = input.rarely();
        if !odd {
            match format.planar_sampling() {
                Some(ChromaSampling::Cs420) => {
                    width = width.next_multiple_of(2);
                    height = height.next_multiple_of(2);
                }
                Some(_) => width = width.next_multiple_of(2),
                None => {}
            }
        }
        let (request, plain) = request(&mut input);
        let representable = !input.rarely();
        let loose = input.below(4);
        let premultiplied = input.rarely();
        let chroma_centering = if input.rarely() {
            [input.below(9), input.below(9)]
        } else {
            [0, 0]
        };
        let pattern = Pattern {
            kind: input.below(7),
            state: input.word(),
            step: u32::from_le_bytes(std::array::from_fn(|_| input.byte())),
            raw: input.bytes,
            position: 0,
        };
        Self::assemble(Recipe {
            format: (format, sample_bits),
            dimensions: (width, height),
            request,
            representable,
            loose,
            premultiplied,
            chroma_centering,
            pattern,
            must_encode: !undeclared
                && !odd
                && plain
                && representable
                && !premultiplied
                && chroma_centering == [0, 0],
        })
    }

    fn assemble(mut recipe: Recipe<'_>) -> Self {
        let (format, sample_bits) = recipe.format;
        let (width, height) = recipe.dimensions;
        let samples = pattern_samples(
            recipe.format,
            recipe.dimensions,
            &recipe.request,
            recipe.representable,
            &mut recipe.pattern,
        );
        let tight = SourceImage::new(width, height, format, SourceSamples::from(&samples)).ok();
        let (stored, placement) = (recipe.loose != 0)
            .then(|| loosen(&samples, tight, usize::from(recipe.loose)))
            .flatten()
            .unwrap_or_else(|| (samples.clone(), Placement::Tight));
        Self {
            format,
            sample_bits,
            width,
            height,
            request: recipe.request,
            samples,
            stored,
            placement,
            premultiplied: recipe.premultiplied,
            chroma_centering: recipe.chroma_centering,
            representable: recipe.representable,
            must_encode: recipe.must_encode,
        }
    }

    /// The image as the encoder borrows it.
    fn image(&self) -> Result<SourceImage<'_>, JxrError> {
        let samples = SourceSamples::from(&self.stored);
        let image = SourceImage::new(self.width, self.height, self.format, samples)?;
        let image = match &self.placement {
            Placement::Tight => image,
            Placement::Rows(row_stride) => image.with_row_stride(*row_stride)?,
            Placement::Planes(planes) => image.with_planes(planes)?,
        };
        let image = match self.sample_bits {
            Some(bits) => image.with_sample_bits(bits)?,
            None => image,
        };
        image
            .with_chroma_centering(self.chroma_centering[0], self.chroma_centering[1])?
            .with_premultiplied_alpha(self.premultiplied)
    }

    fn encode(&self) -> Result<Vec<u8>, JxrError> {
        jxr::encode(&self.image()?, &self.request)
    }

    /// Encode the image and decode the file, or `None` when the encoder
    /// refused the case.
    fn round_trip(&self) -> Option<DecodedSamples> {
        let encoded = self.encode();
        if encoded != self.encode() {
            self.fail(format_args!("encoding is not deterministic"));
        }
        let file = match encoded {
            Ok(file) => file,
            Err(error) if error.kind == JxrErrorKind::InternalInvariant => self.fail(format_args!(
                "the encoder broke its own invariant: {error:?}"
            )),
            Err(error) if self.must_encode => self.fail(format_args!(
                "a plain lossless encode was refused: {error:?}"
            )),
            Err(_) => return None,
        };
        let view = JxrView::parse(&file).unwrap_or_else(|error| {
            self.fail(format_args!("the encoded file does not parse: {error:?}"))
        });
        if view.info().dimensions() != (self.width, self.height) {
            self.fail(format_args!(
                "the encoded file is {:?} samples",
                view.info().dimensions()
            ));
        }
        let request = DecodeRequest::new(self.format).with_backend(BackendRequest::Cpu);
        let decoded = view.decoder().decode(&request).unwrap_or_else(|error| {
            self.fail(format_args!("the encoded file does not decode: {error:?}"))
        });
        let count = |samples| SourceSamples::from(samples).len();
        if decoded.format != self.format || count(&decoded.samples) != count(&self.samples) {
            self.fail(format_args!("the decoded image is not the source's shape"));
        }
        Some(decoded.samples)
    }

    /// Fail with `finding` and the case it was found in.
    fn fail(&self, finding: fmt::Arguments<'_>) -> ! {
        panic!("{finding}\n  in {self}")
    }

    /// Whether the colour channels, and whether the alpha channel, are coded
    /// losslessly: each by its own quantizers, with every band and bit, and
    /// for colour without chroma the encoder subsamples.
    fn lossless(&self) -> (bool, bool) {
        let request = &self.request;
        let every_bit = request.bands == BandPresence::All && request.trim_flexbits == 0;
        let colour = request.colour_is_lossless() && !decimates_chroma(self.format, request);
        let alpha = self.format.has_alpha() && request.alpha_quantization().is_lossless();
        (
            self.representable && every_bit && colour,
            self.representable && every_bit && alpha,
        )
    }

    /// The colour samples and the alpha samples of an image of the case's
    /// format, as bit patterns.
    fn channels(&self, samples: &DecodedSamples) -> (Vec<u32>, Vec<u32>) {
        let values: Vec<u32> = match samples {
            DecodedSamples::BitPacked(values) | DecodedSamples::U8(values) => {
                values.iter().map(|&value| u32::from(value)).collect()
            }
            DecodedSamples::U16(values)
            | DecodedSamples::F16(values)
            | DecodedSamples::Rgb555(values)
            | DecodedSamples::Rgb565(values) => {
                values.iter().map(|&value| u32::from(value)).collect()
            }
            DecodedSamples::I16(values) => values
                .iter()
                .map(|&value| u32::from(value.cast_unsigned()))
                .collect(),
            DecodedSamples::I32(values) => {
                values.iter().map(|&value| value.cast_unsigned()).collect()
            }
            DecodedSamples::F32(values) => values.iter().map(|value| value.to_bits()).collect(),
            DecodedSamples::Rgb101010(values) | DecodedSamples::Rgbe(values) => values.clone(),
        };
        if self.format.planar_sampling().is_some() && self.format.has_alpha() {
            // The alpha plane is the last, at the image's own size.
            let alpha = size(self.width) * size(self.height);
            let colour = values.len().saturating_sub(alpha);
            return (values[..colour].to_vec(), values[colour..].to_vec());
        }
        // The last sample of a pixel may be alpha, or padding that is
        // storage and not part of the image.
        let padded = matches!(
            self.format.channel_layout(),
            Some(ChannelLayout::Rgbx | ChannelLayout::Bgrx)
        );
        if !self.format.has_alpha() && !padded {
            return (values, Vec::new());
        }
        let channels = usize::from(self.format.channel_count()).max(1);
        let (mut colour, mut alpha) = (Vec::new(), Vec::new());
        for pixel in values.chunks(channels) {
            let (last, rest) = pixel.split_last().expect("chunks are never empty");
            colour.extend_from_slice(rest);
            if !padded {
                alpha.push(*last);
            }
        }
        (colour, alpha)
    }

    /// Hold `decoded` to the samples it was coded from, channel by channel.
    fn assert_reproduces(&self, source: &DecodedSamples, decoded: &DecodedSamples) {
        let (colour, alpha) = self.lossless();
        let (source_colour, source_alpha) = self.channels(source);
        let (decoded_colour, decoded_alpha) = self.channels(decoded);
        if colour && source_colour != decoded_colour {
            self.fail(format_args!("lossless colour was not reproduced"));
        }
        if alpha && source_alpha != decoded_alpha {
            self.fail(format_args!("lossless alpha was not reproduced"));
        }
    }
}

/// Check every property of one case.
fn check(case: &Case) {
    let Some(decoded) = case.round_trip() else {
        return;
    };
    let (colour, alpha) = case.lossless();
    if !colour && !alpha {
        return;
    }
    if !matches!(case.format, PixelFormat::F32(_)) {
        case.assert_reproduces(&case.samples, &decoded);
        return;
    }
    // A float is coded to the precision the request keeps, so the source
    // itself need not come back. What a decoder returns is representable,
    // and coding that again must reproduce it.
    let recoded = Case {
        samples: decoded.clone(),
        stored: decoded.clone(),
        placement: Placement::Tight,
        must_encode: false,
        ..case.clone()
    };
    let Some(again) = recoded.round_trip() else {
        // Colour that was coded lossily may have left the range the encoder
        // accepts; colour that was not is the source's own coded values.
        if colour {
            case.fail(format_args!("decoded floats were refused when coded again"));
        }
        return;
    };
    case.assert_reproduces(&decoded, &again);
}

#[cfg(test)]
mod tests;
