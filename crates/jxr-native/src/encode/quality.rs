//! Quantizers and overlap mode of one image: the request's own, or the
//! quality scale resolved against the source.
//!
//! A quality names a picture fidelity, not quantization parameters. The
//! scale was calibrated on 8-bit pictures, as parameters of luma and of the
//! chroma differences of a full-resolution colour transform. Resolving it
//! for a source takes two more steps, both defined here and nowhere else:
//!
//! - Each coded component takes the parameters of its [`Role`] in the
//!   source's colour format.
//! - Each parameter is exchanged for the one whose step is as large against
//!   the source's [`SampleScale`] as the calibrated step is against 255,
//!   found in the decoder's own table of steps.
//!
//! `docs/support.md` records the measurements behind each choice, and
//! `jxr-quality-curve` reproduces them.

use jxr_core::{
    ChromaSampling, ColorFormat, EncodeQuantization, EncodeQuantizers, EncodeRequest, JxrError,
    OutputBitDepth, OverlapMode,
};

use crate::{output_format::scaling::float32_sample, tile_decode::quantizer::component_steps};

use super::input::SourceFormat;

/// Sample range the scale's parameters are calibrated for: 8-bit samples.
const REFERENCE_RANGE: u32 = 255;

/// The parameter whose step in scaled arithmetic is one sample value. It
/// codes no smaller than lossless coding does, so it stands for lossless
/// coding where a source's range brings a quality down to it.
const ONE_SAMPLE: u8 = 4;

/// The finest lossy parameter of the scale, a step of 1.25 sample values.
const FINEST_LOSSY: u8 = 5;

/// Parameters below this one quantize by quarters of a sample value.
const WHOLE_STEPS_BELOW: u8 = 32;

/// Binary logarithm of the octaves of a float's bit patterns that stand for
/// the 8-bit range in a colour sample: eight octaves.
///
/// A float is coded by its bit pattern, which rises by a fixed count for
/// each doubling of the value. Measured on linear-light pictures, with the
/// error taken after the sRGB transfer function, eight octaves of patterns
/// in place of the 8-bit range kept the PSNR of half-float and RGBE images
/// within 1.3 dB of the 8-bit picture's at every quality down to 10. A whole
/// number of octaves also keeps every exchanged parameter a whole number of
/// octaves from its original, where the table of steps repeats exactly.
const FLOAT_OCTAVES_LOG2: i32 = 3;

/// The exponent bias from which a 32-bit float counts as coded
/// logarithmically throughout a picture.
///
/// Values below `2^(1 - bias)` are coded linearly, in as many patterns as
/// one octave above them takes. At the default bias of four that is most of
/// a picture, and linear coding of linear light needs finer steps in the
/// shadows: the range that kept the PSNR halved with each step of bias
/// below this one.
const LOGARITHMIC_FROM_BIAS: i32 = 6;

/// Binary logarithm of the octaves that stand for the range of a float
/// alpha sample: two octaves.
///
/// Coverage counts by its differences, not its ratios, and half of its
/// range is its top octave. Measured on a half-float matte, two octaves
/// kept the PSNR of alpha between 5 dB above and 1 dB below that of the
/// same matte in 8 bits. Four octaves ran 2 to 2.5 dB below it down to
/// quality 10 and 8 dB below at 0, where steps of several octaves ring on
/// the cliff between no coverage and the least there is.
const ALPHA_OCTAVES_LOG2: i32 = 1;

/// The lowest quality RGBE samples are coded at; lower ones code as it does.
///
/// An RGBE pattern steps from zero straight to the exponent of the smallest
/// value the format holds, a hundred octaves up. The steep segment of the
/// scale below 10 rings on that cliff by whole exponents: the PSNR of the
/// calibration pictures fell to 9 dB at quality 0, where 8-bit samples keep
/// 21 dB, while down to 10 it stays within 1.3 dB of theirs.
const RGBE_LOWEST_QUALITY: u8 = 10;

/// The value one in 16-bit fixed-point samples, which are s2.13.
const FIXED16_ONE: u32 = 1 << 13;
/// The value one in 32-bit fixed-point samples, which are s7.24.
const FIXED32_ONE: u32 = 1 << 24;

/// What the planes of one image are coded with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ResolvedQuantization {
    /// Quantizers of each coded colour component.
    colour: Vec<EncodeQuantizers>,
    /// Quantizers of the alpha plane; lossless for a source without one,
    /// which codes none.
    pub(super) alpha: EncodeQuantizers,
    pub(super) overlap: OverlapMode,
}

impl ResolvedQuantization {
    /// Resolve `request` for a source of `format`. `second_level_fits` says
    /// whether the image can take second-level overlap; an overlap mode the
    /// request leaves open is never one it cannot.
    pub(super) fn resolve(
        request: &EncodeRequest,
        format: &SourceFormat,
        second_level_fits: bool,
    ) -> Result<Self, JxrError> {
        let plane = |quantization, role, scale| match quantization {
            EncodeQuantization::Quantizers(quantizers) => Ok(quantizers),
            EncodeQuantization::Quality(quality) => quality_quantizers(quality, role, scale),
        };
        let scale = SampleScale::of_colour(format);
        let colour = (0..usize::from(format.components))
            .map(|component| {
                plane(
                    request.component_quantization(component),
                    Role::of(format, component),
                    scale,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        // An alpha plane is a gray image of the alpha samples.
        let alpha = if format.alpha {
            plane(
                request.alpha_quantization(),
                Role::Primary,
                SampleScale::of_alpha(format)?,
            )?
        } else {
            EncodeQuantizers::LOSSLESS
        };
        let lossless = colour.iter().all(|quantizers| quantizers.is_lossless());
        let overlap = match (request.overlap, request.quantization) {
            (Some(overlap), _) => overlap,
            // Overlap filtering only costs bits where nothing is quantized.
            (None, _) if lossless => OverlapMode::None,
            (None, EncodeQuantization::Quality(quality)) => {
                match quality_overlap(quality.max(scale.lowest_quality)) {
                    OverlapMode::Two if !second_level_fits => OverlapMode::One,
                    overlap => overlap,
                }
            }
            (None, EncodeQuantization::Quantizers(_)) => OverlapMode::One,
        };
        Ok(Self {
            colour,
            alpha,
            overlap,
        })
    }

    /// Quantizers of colour component `component`.
    pub(super) fn component(&self, component: usize) -> EncodeQuantizers {
        self.colour[component]
    }

    /// Whether every colour component is coded with a step of one, so that
    /// the colour plane uses unscaled arithmetic.
    pub(super) fn colour_is_lossless(&self) -> bool {
        self.colour
            .iter()
            .all(|quantizers| quantizers.is_lossless())
    }
}

/// The part a coded component plays in the picture, which decides the step
/// the scale gives it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    /// A component at the sample range of the source: luma, a gray or alpha
    /// plane, every channel coded as it is stored, and the sum the CMYK
    /// transform forms.
    Primary,
    /// A difference the colour transform forms between source channels,
    /// spanning twice their range, coded at the given chroma sampling.
    Difference(ChromaSampling),
}

impl Role {
    fn of(format: &SourceFormat, component: usize) -> Self {
        let sampling = format.decimation().unwrap_or(ChromaSampling::Cs444);
        match (format.output_color, component) {
            // Y is (R + 2G + B) / 4, and U and V are differences.
            (ColorFormat::Rgb | ColorFormat::Rgbe, 1 | 2) => Self::Difference(sampling),
            // The transform of CMYK leaves the mean of black and the other
            // inks in its last component; the first three are differences.
            (ColorFormat::Cmyk, 0..=2) => Self::Difference(ChromaSampling::Cs444),
            // N-component and direct CMYK channels, and the samples of a
            // YUV source, chroma included, are coded as they are stored.
            _ => Self::Primary,
        }
    }
}

/// How the coded samples of an image plane compare with 8-bit samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SampleScale {
    /// The distance between coded values that stands for the 8-bit range:
    /// from black to nominal white where the samples are coded linearly.
    range: u32,
    /// Whether the samples are coded by float bit patterns.
    float: bool,
    /// The quality that every lower one is coded as.
    lowest_quality: u8,
}

impl SampleScale {
    /// The scale of the colour samples of `format`.
    ///
    /// An unsigned integer spans its whole range and a fixed-point sample
    /// the values zero to one, each less the low-order bits `SHIFT_BITS`
    /// drops. A float takes [`FLOAT_OCTAVES_LOG2`] octaves of its patterns,
    /// and fewer with an exponent bias below [`LOGARITHMIC_FROM_BIAS`].
    fn of_colour(format: &SourceFormat) -> Self {
        match format.bit_depth {
            // An RGBE channel is coded as an exponent above a 7-bit mantissa.
            OutputBitDepth::U8 if format.output_color == ColorFormat::Rgbe => Self {
                lowest_quality: RGBE_LOWEST_QUALITY,
                ..Self::of_patterns(7, FLOAT_OCTAVES_LOG2)
            },
            // A half float has ten mantissa bits.
            OutputBitDepth::F16 => Self::of_patterns(10, FLOAT_OCTAVES_LOG2),
            OutputBitDepth::F32 {
                mantissa_length,
                exponent_bias,
            } => {
                let linear = (LOGARITHMIC_FROM_BIAS - i32::from(exponent_bias)).max(0);
                Self::of_patterns(mantissa_length, FLOAT_OCTAVES_LOG2 - linear)
            }
            depth => Self::of_integer(depth),
        }
    }

    /// The scale of the alpha samples of `format`. An integer alpha sample
    /// is scaled as a colour sample is. A float one takes
    /// [`ALPHA_OCTAVES_LOG2`] octaves of its patterns, or the patterns up
    /// to one where the format codes those linearly and they are fewer.
    fn of_alpha(format: &SourceFormat) -> Result<Self, JxrError> {
        let (mantissa_bits, one) = match format.bit_depth {
            // One is the pattern 0x3c00.
            OutputBitDepth::F16 => (10, 0x3c00),
            OutputBitDepth::F32 {
                mantissa_length,
                exponent_bias,
            } => {
                let one = float32_sample(1.0_f32.to_bits(), false, mantissa_length, exponent_bias)
                    .map_err(|_| JxrError::arithmetic("encode float sample range"))?;
                (mantissa_length, u32::try_from(one).unwrap_or(0))
            }
            depth => return Ok(Self::of_integer(depth)),
        };
        let octaves = Self::of_patterns(mantissa_bits, ALPHA_OCTAVES_LOG2);
        Ok(Self {
            range: octaves.range.min(one).max(1),
            ..octaves
        })
    }

    /// The scale of float patterns with `mantissa_bits` mantissa bits, of
    /// which `2^octaves_log2` octaves stand for the 8-bit range.
    fn of_patterns(mantissa_bits: u8, octaves_log2: i32) -> Self {
        let magnitude = octaves_log2.unsigned_abs();
        let octave = 1_u32 << mantissa_bits.min(23);
        let range = if octaves_log2 >= 0 {
            octave.checked_shl(magnitude)
        } else {
            octave.checked_shr(magnitude)
        };
        Self {
            range: range.unwrap_or(0).max(1),
            float: true,
            lowest_quality: 0,
        }
    }

    /// The scale of samples that are not floats.
    fn of_integer(depth: OutputBitDepth) -> Self {
        let range = match depth {
            OutputBitDepth::U16 { shift_bits } => u32::from(u16::MAX) >> shift_bits,
            OutputBitDepth::I16 { shift_bits } => FIXED16_ONE >> shift_bits,
            OutputBitDepth::I32 { shift_bits } => FIXED32_ONE >> shift_bits,
            // Red and blue of RGB565 are coded at twice their five-bit
            // values, to the range of its green.
            depth => depth.maximum().map_or(REFERENCE_RANGE, u32::from),
        };
        Self {
            range: range.max(1),
            float: false,
            lowest_quality: 0,
        }
    }
}

/// Quantizers of a component of `role` at `quality`, for samples of `scale`.
fn quality_quantizers(
    quality: u8,
    role: Role,
    scale: SampleScale,
) -> Result<EncodeQuantizers, JxrError> {
    if quality >= 100 {
        return Ok(EncodeQuantizers::LOSSLESS);
    }
    let level = quality_level(quality.max(scale.lowest_quality));
    // The level counts half parameters. Primary components round it up and
    // differences round it down, so a one-level step moves one kind: the
    // primary alone on leaving a whole level, then the differences to catch
    // up.
    let primary = level.div_ceil(2);
    if scale.range < REFERENCE_RANGE {
        return reduced_quantizers(primary, role, scale.range);
    }
    // A step of the differences alone moved the size of float images by
    // less than the noise of coding their bit patterns, and one RGBE image
    // grew on it. Their differences keep to the primary's whole levels.
    let difference = if scale.float { primary } else { level / 2 };
    let reference = match role {
        Role::Primary => band_quantizers(primary, primary),
        Role::Difference(sampling) => {
            band_quantizers(difference, difference_parameter(difference, sampling))
        }
    };
    Ok(EncodeQuantizers {
        dc: scaled_parameter(reference.dc, scale.range)?,
        low_pass: scaled_parameter(reference.low_pass, scale.range)?,
        high_pass: scaled_parameter(reference.high_pass, scale.range)?,
    })
}

/// Quantizers of a component of `role` for samples of a `range` below 255,
/// where the scale's primary parameter is `primary`.
///
/// Such samples have few steps to choose from. Below
/// [`WHOLE_STEPS_BELOW`] the 8-bit scale uses 5 and the multiples of four,
/// which quantize by whole sample values and coded smaller with less error
/// than the parameters between them; a smaller range is held to those. A
/// quality whose step comes down to one sample value is lossless: scaled
/// arithmetic at steps that fine codes larger than exact coding does. And
/// every band and component follows the primary parameter alone, so that
/// nothing moves on a quality where the primary cannot.
fn reduced_quantizers(primary: u8, role: Role, range: u32) -> Result<EncodeQuantizers, JxrError> {
    // Steps are compared as `step * 255` against `primary step * range`.
    let reference = u128::from(REFERENCE_RANGE);
    let target = step(primary)? * u128::from(range);
    let mut nearest = (ONE_SAMPLE, step(ONE_SAMPLE)?);
    for parameter in (FINEST_LOSSY..=u8::MAX).filter(|&parameter| {
        parameter == FINEST_LOSSY || parameter >= WHOLE_STEPS_BELOW || parameter.is_multiple_of(4)
    }) {
        let candidate = step(parameter)?;
        if is_past(candidate * reference, nearest.1 * reference, target) {
            break;
        }
        nearest = (parameter, candidate);
    }
    Ok(match (nearest.0, role) {
        (ONE_SAMPLE, _) => EncodeQuantizers::LOSSLESS,
        (parameter, Role::Primary) => EncodeQuantizers::uniform(parameter),
        (parameter, Role::Difference(sampling)) => {
            EncodeQuantizers::uniform(difference_parameter(parameter, sampling))
        }
    })
}

/// The parameter that quantizes samples of `range`, at least 255, as
/// `reference` quantizes 8-bit samples: the one whose step is nearest
/// `range / 255` times as large.
///
/// Every range above 255 is within half a percent of 255 times a power of
/// two, and steps double every sixteen parameters, so from 16 up this adds
/// sixteen parameters for each doubling of the range.
fn scaled_parameter(reference: u8, range: u32) -> Result<u8, JxrError> {
    // Steps are compared as `step * 255` against `reference step * range`.
    let scale = u128::from(REFERENCE_RANGE);
    let target = step(reference)? * u128::from(range);
    let mut nearest = (reference, step(reference)?);
    for parameter in reference..=u8::MAX {
        let candidate = step(parameter)?;
        if is_past(candidate * scale, nearest.1 * scale, target) {
            break;
        }
        nearest = (parameter, candidate);
    }
    Ok(nearest.0)
}

/// Whether `candidate`, the next step up from `nearest`, is farther from
/// `target` than `nearest` is: whether `target` lies below their geometric
/// mean. Steps rise with the parameter, so a search for the nearest one
/// ends here.
const fn is_past(candidate: u128, nearest: u128, target: u128) -> bool {
    candidate * nearest > target * target
}

/// Step of the first component's high-pass band at `parameter` in scaled
/// arithmetic, from the decoder's table. Only ratios of steps are used.
fn step(parameter: u8) -> Result<u128, JxrError> {
    component_steps(0, [parameter; 3], true)
        .map(|steps| u128::from(steps.high_pass))
        .map_err(|_| JxrError::arithmetic("encode quantizer step"))
}

/// Quantization level of a lossy quality, in half parameters of luma.
///
/// Each arm is a straight segment between anchors calibrated on 8-bit
/// photographs, so every step is a whole number of half parameters. Luma
/// PSNR stays within 1.3 dB of baseline JPEG's at the same quality.
///
/// - 99 is luma parameter 5. Scaled arithmetic carries three fractional
///   bits, which makes parameter 4 a step of exactly one sample value: gray
///   photographs coded only 3 percent smaller than lossless there, and
///   smaller parameters coded larger still.
/// - 98 down to 92 are the luma parameters 8, 12, ... 32. Below 32 only
///   multiples of four quantize by whole sample values, and the parameter
///   just under each multiple coded larger, with more luma error, than the
///   multiple did on every calibration image.
/// - 92 down to 75 ease from one and a half parameters per quality to one.
/// - 75 down to 10 fall by half a parameter, the finest step there is:
///   primary components and differences move on alternate qualities.
/// - Below 10 the scale drops three parameters per quality, as steeply as
///   JPEG's does, to about 0.05 bits per pixel.
const fn quality_level(quality: u8) -> u8 {
    match quality {
        99.. => 10,
        93..=98 => 16 + 8 * (98 - quality),
        91..=92 => 64 + 3 * (92 - quality),
        76..=90 => 70 + 2 * (90 - quality),
        11..=75 => 100 + (75 - quality),
        0..=10 => 165 + 6 * (10 - quality),
    }
}

/// Parameter of a colour-transform difference whose channels would take
/// `parameter`.
///
/// At full resolution the step is doubled, as the range of a difference is.
/// The step is linear in the parameter up to 16 and doubles every sixteen
/// parameters after that. On RGB photographs the doubled step coded 7 to 13
/// percent smaller than equal parameters at the same RGB PSNR, within about
/// one percent of the best offset at every quality.
///
/// A subsampled difference stands for two or four pixels, so an error in it
/// weighs that much more: it takes eight parameters more than the channels
/// at 4:2:2 and four at 4:2:0, about 1.4 and 1.2 times their step, close to
/// the offsets the T.835 reference encoder derives chroma quantizers with.
/// Against the doubled step that gained up to 0.2 dB of RGB PSNR at equal
/// size at 4:2:0, where the doubled step also coded one calibration image
/// larger on one step down the scale.
const fn difference_parameter(parameter: u8, sampling: ChromaSampling) -> u8 {
    let offset = match sampling {
        ChromaSampling::Cs444 => 16,
        ChromaSampling::Cs422 => 8,
        ChromaSampling::Cs420 => 4,
    };
    // The same factor in the linear part of the table, below 16.
    let offset = if parameter < 16 {
        parameter * offset / 16
    } else {
        offset
    };
    parameter.saturating_add(offset)
}

/// Quantizers of one component at `level`, whose AC bands use `parameter`.
///
/// The bands share a parameter up to level 64, where offsets of a few
/// parameters moved the size at a given PSNR by under one percent. Past it
/// DC keeps half of the excess, at most sixteen parameters: that coded up to
/// 7 percent smaller at equal PSNR, and steadies macroblock means where
/// little else is coded.
const fn band_quantizers(level: u8, parameter: u8) -> EncodeQuantizers {
    let excess = level.saturating_sub(64) / 2;
    let relief = if excess < 16 { excess } else { 16 };
    EncodeQuantizers {
        dc: parameter - relief,
        low_pass: parameter,
        high_pass: parameter,
    }
}

/// Overlap mode of a lossy quality.
///
/// First-level overlap is the rule: coding without overlap cost 4 to 16
/// percent more bits at equal PSNR across the scale, and second-level
/// overlap up to 10 percent more and never less. The two ends differ:
///
/// - 99 codes without overlap. At a luma step of 1.25 sample values overlap
///   stops paying on noisy content, and coded synthetic noise larger than
///   lossless coding did.
/// - 6 and below use second-level overlap. First-level overlap leaves a grid
///   of macroblock edges there, their steps over twice as large against
///   steps inside blocks as in the source, which second-level overlap levels
///   with ordinary block edges.
const fn quality_overlap(quality: u8) -> OverlapMode {
    match quality {
        99.. => OverlapMode::None,
        7..=98 => OverlapMode::One,
        0..=6 => OverlapMode::Two,
    }
}

#[cfg(test)]
mod tests {
    use jxr_core::{ChannelLayout, PixelFormat};

    use super::*;

    const RGB8: PixelFormat = PixelFormat::U8(ChannelLayout::Rgb);
    const RGB16: PixelFormat = PixelFormat::U16(ChannelLayout::Rgb);
    const CMYK8: PixelFormat = PixelFormat::U8(ChannelLayout::Cmyk);
    const BLACK_WHITE: PixelFormat = PixelFormat::BitPacked(ChannelLayout::Luma);

    fn source(format: PixelFormat, request: &EncodeRequest) -> SourceFormat {
        SourceFormat::resolve(format, false, None, request).unwrap()
    }

    fn resolve(format: PixelFormat, request: &EncodeRequest) -> ResolvedQuantization {
        ResolvedQuantization::resolve(request, &source(format, request), true).unwrap()
    }

    fn at_quality(format: PixelFormat, quality: u8) -> ResolvedQuantization {
        resolve(format, &EncodeRequest::quality(quality))
    }

    fn bands(quantizers: EncodeQuantizers) -> [u8; 3] {
        [quantizers.dc, quantizers.low_pass, quantizers.high_pass]
    }

    /// The parameters of every component and of alpha, each as DC,
    /// low-pass, high-pass.
    fn parameters(quantization: &ResolvedQuantization) -> Vec<[u8; 3]> {
        quantization
            .colour
            .iter()
            .chain([&quantization.alpha])
            .map(|&quantizers| bands(quantizers))
            .collect()
    }

    /// The six quantization parameters of 8-bit RGB at a quality: luma then
    /// chroma, each as DC, low-pass, high-pass.
    fn quality_parameters(quality: u8) -> [u8; 6] {
        let quantization = at_quality(RGB8, quality);
        let [luma, chroma] = [0, 1].map(|component| bands(quantization.component(component)));
        [luma[0], luma[1], luma[2], chroma[0], chroma[1], chroma[2]]
    }

    /// Formats of every kind the scale resolves differently.
    fn formats() -> Vec<PixelFormat> {
        use ChannelLayout::{Cmyk, Luma, NComponent, Rgb, Rgba, Yuv};
        vec![
            PixelFormat::U8(Luma),
            RGB8,
            PixelFormat::U8(Rgba),
            RGB16,
            PixelFormat::I16(Rgb),
            PixelFormat::I32(Rgb),
            PixelFormat::F16(Rgba),
            PixelFormat::F32(Rgb),
            PixelFormat::Rgbe,
            PixelFormat::Rgb101010,
            PixelFormat::Rgb565,
            PixelFormat::Rgb555,
            BLACK_WHITE,
            CMYK8,
            PixelFormat::U16(Cmyk),
            PixelFormat::U8(NComponent(5)),
            PixelFormat::U8(Yuv(ChromaSampling::Cs444)),
            PixelFormat::U8(Yuv(ChromaSampling::Cs420)),
        ]
    }

    #[test]
    fn quality_never_quantizes_more_finely_as_it_falls() {
        for quality in 0..100 {
            let coarser = at_quality(RGB8, quality);
            let finer = at_quality(RGB8, quality + 1);
            for (band, (coarse, fine)) in quality_parameters(quality)
                .into_iter()
                .zip(quality_parameters(quality + 1))
                .enumerate()
            {
                assert!(
                    coarse >= fine,
                    "quality {quality} band {band}: {coarse} is finer than {fine} above it"
                );
            }
            // No two neighbours ask for the same coding.
            assert_ne!(coarser, finer, "quality {quality}");
            // Overlap filtering only strengthens towards the bottom of the scale.
            assert!(
                coarser.overlap as u8 >= finer.overlap as u8,
                "quality {quality}"
            );
        }
    }

    #[test]
    fn no_format_is_quantized_more_finely_as_the_quality_falls() {
        for format in formats() {
            for sampling in [ChromaSampling::Cs444, ChromaSampling::Cs420] {
                for quality in 0..100 {
                    let request =
                        |quality| EncodeRequest::quality(quality).with_chroma_sampling(sampling);
                    let coarser = resolve(format, &request(quality));
                    let finer = resolve(format, &request(quality + 1));
                    for (coarse, fine) in parameters(&coarser).into_iter().zip(parameters(&finer)) {
                        for band in 0..3 {
                            assert!(
                                coarse[band] >= fine[band],
                                "{format:?} {sampling:?} quality {quality}: {coarse:?} after \
                                 {fine:?}"
                            );
                        }
                        // DC is never the coarsest band of a component.
                        assert!(coarse[0] <= coarse[1] && coarse[1] <= coarse[2]);
                    }
                    assert!(coarser.overlap as u8 >= finer.overlap as u8);
                }
            }
        }
    }

    #[test]
    fn quality_keeps_chroma_no_finer_than_luma() {
        for quality in 0..=100 {
            let quantization = at_quality(RGB8, quality);
            let luma = quantization.component(0);
            let chroma = quantization.component(1);
            assert!(chroma.dc >= luma.dc, "quality {quality} DC");
            assert!(
                chroma.low_pass >= luma.low_pass,
                "quality {quality} low-pass"
            );
            assert!(
                chroma.high_pass >= luma.high_pass,
                "quality {quality} high-pass"
            );
            // DC is never the coarsest band of a component.
            for component in [luma, chroma] {
                assert!(component.dc <= component.low_pass, "quality {quality}");
            }
        }
    }

    #[test]
    fn quality_anchors_keep_their_calibrated_settings() {
        // Changing these changes every file the scale produces; the measured
        // curve in docs/support.md has to be regenerated with them.
        for (quality, parameters, overlap) in [
            (99, [5, 5, 5, 10, 10, 10], OverlapMode::None),
            (90, [35, 35, 35, 51, 51, 51], OverlapMode::One),
            (75, [50, 50, 50, 66, 66, 66], OverlapMode::One),
            (50, [63, 63, 63, 78, 78, 78], OverlapMode::One),
            (25, [70, 75, 75, 86, 91, 91], OverlapMode::One),
            (0, [97, 113, 113, 112, 128, 128], OverlapMode::Two),
        ] {
            assert_eq!(quality_parameters(quality), parameters, "quality {quality}");
            assert_eq!(
                at_quality(RGB8, quality).overlap,
                overlap,
                "quality {quality}"
            );
        }
        assert_eq!(
            at_quality(RGB8, 100),
            resolve(RGB8, &EncodeRequest::lossless())
        );
    }

    #[test]
    fn eight_bit_samples_take_the_scale_as_it_was_calibrated() {
        for parameter in FINEST_LOSSY..=u8::MAX {
            assert_eq!(
                scaled_parameter(parameter, REFERENCE_RANGE).unwrap(),
                parameter
            );
        }
        // Luma, alpha and every channel coded as stored share the primary's
        // parameters; both chroma differences share theirs.
        for quality in [99, 75, 40, 3] {
            let rgba = at_quality(PixelFormat::U8(ChannelLayout::Rgba), quality);
            assert_eq!(rgba.alpha, rgba.component(0), "quality {quality}");
            assert_eq!(rgba.component(1), rgba.component(2), "quality {quality}");
            let gray = at_quality(PixelFormat::U8(ChannelLayout::Luma), quality);
            assert_eq!(gray.component(0), rgba.component(0), "quality {quality}");
        }
    }

    #[test]
    fn a_deeper_range_scales_every_step_with_it() {
        // Sixteen parameters double a step, from 16 up.
        for parameter in 16..=127 {
            assert_eq!(
                scaled_parameter(parameter, 65_535).unwrap(),
                parameter + 128
            );
            assert_eq!(scaled_parameter(parameter, 1023).unwrap(), parameter + 32);
        }
        for range in [511, 1023, 4095, 8192, 16_384, 65_535] {
            for reference in FINEST_LOSSY..=113 {
                let scaled = scaled_parameter(reference, range).unwrap();
                // Within the table's spacing of the exact step: one part in
                // sixteen at its widest.
                let exact = step(reference).unwrap() * u128::from(range);
                let found = step(scaled).unwrap() * u128::from(REFERENCE_RANGE);
                assert!(
                    found.abs_diff(exact) * 32 <= exact,
                    "range {range} parameter {reference} became {scaled}"
                );
            }
        }
        // The table ends at 255, where the deepest ranges arrive early.
        assert_eq!(scaled_parameter(128, 65_535).unwrap(), u8::MAX);
        let deep = EncodeRequest::quality(50).with_shift_bits(0);
        let quantization = resolve(PixelFormat::I32(ChannelLayout::Rgb), &deep);
        assert_eq!(bands(quantization.component(0)), [u8::MAX; 3]);
        // 16-bit samples are the 8-bit scale, eight octaves up. Below 16
        // the table is linear, and eight octaves up from 5 is 116.
        assert_eq!(bands(at_quality(RGB16, 99).component(0)), [116; 3]);
        for quality in [96, 90, 75, 40, 10] {
            let (eight, sixteen) = (at_quality(RGB8, quality), at_quality(RGB16, quality));
            for component in 0..3 {
                assert_eq!(
                    bands(sixteen.component(component)),
                    bands(eight.component(component)).map(|parameter| parameter + 128)
                );
            }
        }
    }

    #[test]
    fn each_component_takes_the_step_of_its_part_in_the_colour_format() {
        use ChannelLayout::{Luma, NComponent, Yuv};
        let lossless = EncodeRequest::lossless();
        let roles = |format, request: &EncodeRequest| {
            let source = source(format, request);
            (0..usize::from(source.components))
                .map(|component| Role::of(&source, component))
                .collect::<Vec<_>>()
        };
        let full = Role::Difference(ChromaSampling::Cs444);
        assert_eq!(roles(RGB8, &lossless), [Role::Primary, full, full]);
        assert_eq!(
            roles(PixelFormat::Rgbe, &lossless),
            [Role::Primary, full, full]
        );
        for sampling in [ChromaSampling::Cs422, ChromaSampling::Cs420] {
            let subsampled = lossless.clone().with_chroma_sampling(sampling);
            let difference = Role::Difference(sampling);
            assert_eq!(
                roles(RGB16, &subsampled),
                [Role::Primary, difference, difference]
            );
            // Chroma a YUV source stores is no difference of the encoder's,
            // whether it subsamples that chroma or the source did.
            for format in [Yuv(ChromaSampling::Cs444), Yuv(sampling)] {
                assert_eq!(
                    roles(PixelFormat::U8(format), &subsampled),
                    [Role::Primary; 3]
                );
            }
        }
        // The CMYK transform leaves its sum in the last component.
        assert_eq!(roles(CMYK8, &lossless), [full, full, full, Role::Primary]);
        let direct = lossless.clone().with_cmyk_direct(true);
        assert_eq!(roles(CMYK8, &direct), [Role::Primary; 4]);
        assert_eq!(
            roles(PixelFormat::U8(NComponent(6)), &lossless),
            [Role::Primary; 6]
        );
        assert_eq!(roles(PixelFormat::U8(Luma), &lossless), [Role::Primary]);

        // A difference takes twice the primary's step at full resolution,
        // and less of it the more pixels it stands for.
        for (sampling, parameters) in [
            (ChromaSampling::Cs444, [10, 32, 66, 144]),
            (ChromaSampling::Cs422, [7, 24, 58, 136]),
            (ChromaSampling::Cs420, [6, 20, 54, 132]),
        ] {
            assert_eq!(
                [5, 16, 50, 128].map(|parameter| difference_parameter(parameter, sampling)),
                parameters,
                "{sampling:?}"
            );
        }
        let quantization = at_quality(CMYK8, 75);
        assert_eq!(bands(quantization.component(3)), [50; 3]);
        for component in 0..3 {
            assert_eq!(bands(quantization.component(component)), [66; 3]);
        }
    }

    #[test]
    fn sample_ranges_follow_the_coded_samples() {
        use ChannelLayout::{Luma, Rgb, Rgba};
        let lossless = EncodeRequest::lossless();
        let colour = |format, request: &EncodeRequest| {
            let scale = SampleScale::of_colour(&source(format, request));
            (scale.range, scale.float, scale.lowest_quality)
        };
        let alpha = |format, request: &EncodeRequest| {
            SampleScale::of_alpha(&source(format, request))
                .unwrap()
                .range
        };
        for (format, range) in [
            (BLACK_WHITE, 1),
            (PixelFormat::Rgb555, 31),
            (PixelFormat::Rgb565, 63),
            (PixelFormat::U8(Luma), 255),
            (PixelFormat::Rgb101010, 1023),
            (PixelFormat::U16(Rgb), 65_535),
            // Fixed point counts from zero to one, less the bits it drops.
            (PixelFormat::I16(Rgb), 8192),
            (PixelFormat::I32(Rgb), 16_384),
        ] {
            assert_eq!(colour(format, &lossless), (range, false, 0), "{format:?}");
            assert_eq!(alpha(format, &lossless), range, "{format:?}");
        }
        let shifted = lossless.clone().with_shift_bits(4);
        assert_eq!(colour(PixelFormat::U16(Rgb), &shifted).0, 4095);
        assert_eq!(colour(PixelFormat::I16(Rgb), &shifted).0, 512);
        assert_eq!(colour(PixelFormat::I32(Rgb), &shifted).0, 1 << 20);

        // Floats: eight octaves of patterns, two for alpha.
        assert_eq!(
            colour(PixelFormat::F16(Rgba), &lossless),
            (8 << 10, true, 0)
        );
        assert_eq!(alpha(PixelFormat::F16(Rgba), &lossless), 2 << 10);
        assert_eq!(
            colour(PixelFormat::Rgbe, &lossless),
            (8 << 7, true, RGBE_LOWEST_QUALITY)
        );
        // A 32-bit float halves its range with each step of bias below six.
        for (bias, range, alpha_range) in [
            (15, 8 << 13, 2 << 13),
            (6, 8 << 13, 2 << 13),
            (5, 4 << 13, 2 << 13),
            (4, 2 << 13, 2 << 13),
            // Coverage coded linearly up to one takes the patterns it has.
            (1, 1 << 11, 1 << 13),
            (0, 1 << 10, 1 << 12),
            (-40, 1, 1),
        ] {
            let request = lossless.clone().with_float_precision(13, bias);
            let format = PixelFormat::F32(Rgba);
            assert_eq!(colour(format, &request), (range, true, 0), "bias {bias}");
            assert_eq!(alpha(format, &request), alpha_range, "bias {bias}");
        }
        let precise = lossless.clone().with_float_precision(16, 4);
        assert_eq!(colour(PixelFormat::F32(Rgb), &precise).0, 2 << 16);
    }

    #[test]
    fn float_patterns_are_linear_below_the_bias_and_an_octave_wide_above() {
        // What `SampleScale` assumes of the decoder's float mapping.
        let pattern = |value: f32, mantissa, bias| {
            float32_sample(value.to_bits(), false, mantissa, bias).unwrap()
        };
        for (mantissa, bias) in [(13_u8, 4_i8), (13, 1), (10, 6), (16, 4)] {
            let octave = 1 << mantissa;
            let linear_below = 2.0_f32.powi(1 - i32::from(bias));
            assert_eq!(pattern(linear_below, mantissa, bias), octave);
            assert_eq!(pattern(linear_below / 2.0, mantissa, bias), octave / 2);
            assert_eq!(pattern(linear_below / 4.0, mantissa, bias), octave / 4);
            assert_eq!(pattern(linear_below * 2.0, mantissa, bias), 2 * octave);
            assert_eq!(pattern(linear_below * 4.0, mantissa, bias), 3 * octave);
        }
    }

    #[test]
    fn a_range_below_eight_bits_keeps_to_whole_steps_and_to_lossless_coding() {
        // One bit has no lossy step at any quality, nor a use for overlap.
        for quality in 0..=100 {
            assert_eq!(
                at_quality(BLACK_WHITE, quality),
                resolve(BLACK_WHITE, &EncodeRequest::lossless()),
                "quality {quality}"
            );
        }
        for format in [PixelFormat::Rgb555, PixelFormat::Rgb565] {
            let mut lossless = 0;
            for quality in 0..100 {
                let quantization = at_quality(format, quality);
                let [luma, chroma] = [0, 1].map(|component| quantization.component(component));
                if luma.is_lossless() {
                    lossless += 1;
                    assert!(chroma.is_lossless(), "{format:?} quality {quality}");
                    assert_eq!(quantization.overlap, OverlapMode::None);
                    continue;
                }
                // One parameter in every band, from the steps the scale
                // uses, and chroma at twice its step.
                let parameter = luma.high_pass;
                assert_eq!(luma, EncodeQuantizers::uniform(parameter));
                assert!(
                    parameter == FINEST_LOSSY || parameter >= 32 || parameter % 4 == 0,
                    "{format:?} quality {quality}: {parameter}"
                );
                assert_eq!(
                    chroma,
                    EncodeQuantizers::uniform(difference_parameter(
                        parameter,
                        ChromaSampling::Cs444
                    ))
                );
            }
            // The top of the scale has no step to offer but the exact one.
            assert!((3..40).contains(&lossless), "{format:?}: {lossless}");
            assert!(at_quality(format, 99).colour_is_lossless());
            assert!(!at_quality(format, 50).colour_is_lossless());
        }
    }

    #[test]
    fn float_differences_move_with_the_primary_and_rgbe_stops_at_its_lowest_quality() {
        for format in [
            PixelFormat::F16(ChannelLayout::Rgb),
            PixelFormat::F32(ChannelLayout::Rgb),
            PixelFormat::Rgbe,
        ] {
            // Between 75 and 10 the primary moves on even qualities only.
            for quality in (12..74).step_by(2) {
                assert_eq!(
                    at_quality(format, quality + 1),
                    at_quality(format, quality + 2),
                    "{format:?} quality {quality}"
                );
                assert_ne!(at_quality(format, quality), at_quality(format, quality + 1));
            }
        }
        // Integer differences move on the qualities between.
        assert_ne!(at_quality(RGB16, 41), at_quality(RGB16, 42));
        for quality in 0..RGBE_LOWEST_QUALITY {
            assert_eq!(
                at_quality(PixelFormat::Rgbe, quality),
                at_quality(PixelFormat::Rgbe, RGBE_LOWEST_QUALITY),
                "quality {quality}"
            );
        }
        assert_ne!(
            at_quality(PixelFormat::F16(ChannelLayout::Rgb), 0),
            at_quality(PixelFormat::F16(ChannelLayout::Rgb), RGBE_LOWEST_QUALITY)
        );
    }

    #[test]
    fn quantizers_given_outright_are_kept_for_every_format() {
        let luma = EncodeQuantizers {
            dc: 3,
            low_pass: 77,
            high_pass: 201,
        };
        let chroma = EncodeQuantizers::uniform(9);
        let alpha = EncodeQuantizers::uniform(250);
        for format in formats() {
            let components = usize::from(source(format, &EncodeRequest::lossless()).components);
            let has_alpha = source(format, &EncodeRequest::lossless()).alpha;
            let plain = resolve(format, &EncodeRequest::lossy(luma));
            assert_eq!(plain.colour, vec![luma; components], "{format:?}");
            // A source without an alpha channel codes no alpha plane.
            let given = |quantizers| {
                if has_alpha {
                    quantizers
                } else {
                    EncodeQuantizers::LOSSLESS
                }
            };
            assert_eq!(plain.alpha, given(luma), "{format:?}");
            assert_eq!(plain.overlap, OverlapMode::One, "{format:?}");

            // Beside a quality, each plane named keeps what it was given
            // and the first component alone follows the scale.
            let request = EncodeRequest::quality(60)
                .with_chroma_quantizers(chroma)
                .with_alpha_quantizers(alpha)
                .with_overlap(OverlapMode::Two);
            let mixed = resolve(format, &request);
            let scale = at_quality(format, 60);
            assert_eq!(mixed.component(0), scale.component(0), "{format:?}");
            assert_eq!(
                mixed.colour[1..],
                vec![chroma; components - 1],
                "{format:?}"
            );
            assert_eq!(mixed.alpha, given(alpha), "{format:?}");
            assert_eq!(mixed.overlap, OverlapMode::Two, "{format:?}");
        }
    }

    #[test]
    fn an_overlap_mode_left_open_is_one_the_image_can_take() {
        let open = |request: EncodeRequest| EncodeRequest {
            overlap: None,
            ..request
        };
        let overlap = |request: &EncodeRequest, second_level_fits| {
            ResolvedQuantization::resolve(request, &source(RGB8, request), second_level_fits)
                .unwrap()
                .overlap
        };
        // The scale's own choice, less second-level overlap where it does
        // not fit.
        for (quality, mode) in [
            (99, OverlapMode::None),
            (98, OverlapMode::One),
            (7, OverlapMode::One),
            (6, OverlapMode::Two),
            (0, OverlapMode::Two),
        ] {
            let request = EncodeRequest::quality(quality);
            assert_eq!(overlap(&request, true), mode, "quality {quality}");
            let narrow = if mode == OverlapMode::Two {
                OverlapMode::One
            } else {
                mode
            };
            assert_eq!(overlap(&request, false), narrow, "quality {quality}");
        }
        // A mode the request names is kept, whether or not it fits: the
        // encoder refuses it rather than change it.
        let named = EncodeRequest::quality(50).with_overlap(OverlapMode::Two);
        assert_eq!(overlap(&named, false), OverlapMode::Two);
        // Without a quality: none where nothing is quantized, else first
        // level.
        assert_eq!(
            overlap(&open(EncodeRequest::lossless()), true),
            OverlapMode::None
        );
        let lossy = EncodeRequest::lossy(EncodeQuantizers::uniform(30));
        assert_eq!(overlap(&open(lossy), true), OverlapMode::One);
        let chroma = EncodeRequest::lossless().with_chroma_quantizers(EncodeQuantizers::uniform(9));
        assert_eq!(overlap(&open(chroma), true), OverlapMode::One);
        // A quality of 100 set directly is the lossless request.
        let direct = EncodeRequest {
            quantization: EncodeQuantization::Quality(100),
            ..EncodeRequest::quality(50)
        };
        assert_eq!(
            resolve(RGB8, &direct),
            resolve(RGB8, &EncodeRequest::lossless())
        );
    }
}
