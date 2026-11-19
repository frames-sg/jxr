//! Bias, scaled-stream rounding, and postscaling from T.832 clauses 9.10.5-9.10.7.

use jxr_core::ColorFormat;
use jxr_math::arithmetic::{checked_add, checked_sub};

use super::{OutputBitDepth, OutputFormatError};

#[inline]
pub(crate) fn scale_integer_component(
    sample: i32,
    component: usize,
    output_color: ColorFormat,
    depth: OutputBitDepth,
    scaled: bool,
) -> Result<i32, OutputFormatError> {
    ChannelScale::resolve(component, output_color, depth, scaled).apply(sample)
}

/// Bias, rounding, and postscaling for one output channel, resolved once so
/// per-sample packing performs only the checked arithmetic.
///
/// Resolution failures are retained and reported by the first `apply`, which
/// keeps error behavior identical to resolving per sample.
#[derive(Clone, Debug)]
pub(crate) enum ChannelScale {
    /// RGBX/BGRX padding, which always stores zero.
    Padding,
    Integer {
        shifted_bias: i32,
        rounding: i32,
        shift: u32,
        post_shift: Option<u32>,
    },
    Invalid(OutputFormatError),
}

impl ChannelScale {
    #[inline]
    pub(crate) fn resolve(
        component: usize,
        output_color: ColorFormat,
        depth: OutputBitDepth,
        scaled: bool,
    ) -> Self {
        let bias = match bias(depth) {
            Ok(bias) => bias,
            Err(error) => return Self::Invalid(error),
        };
        let component_bias = match output_color {
            ColorFormat::Cmyk if component < 3 => bias >> 1,
            ColorFormat::Cmyk => -(bias >> 1),
            ColorFormat::Rgbe => 0,
            _ => bias,
        };
        let scale = u32::from(scaled) * 3;
        let shifted_bias = match checked_shift_left(component_bias, scale, "scaling sample bias") {
            Ok(shifted_bias) => shifted_bias,
            Err(error) => return Self::Invalid(error),
        };
        let rounding = if scaled {
            match depth {
                OutputBitDepth::Bit1White
                | OutputBitDepth::Bit1Black
                | OutputBitDepth::U16 { .. } => 4,
                _ => 3,
            }
        } else {
            0
        };
        let shift = if matches!(depth, OutputBitDepth::Rgb565) && component != 1 {
            scale + 1
        } else {
            scale
        };
        let post_shift = match depth {
            OutputBitDepth::U16 { shift_bits }
            | OutputBitDepth::I16 { shift_bits }
            | OutputBitDepth::I32 { shift_bits } => Some(u32::from(shift_bits)),
            _ => None,
        };
        Self::Integer {
            shifted_bias,
            rounding,
            shift,
            post_shift,
        }
    }

    /// The scale of an alpha channel: a luma sample of the alpha plane's own
    /// depth and scaled flag, whatever the colour channels beside it use.
    #[inline]
    pub(crate) fn alpha(depth: OutputBitDepth, scaled: bool) -> Self {
        Self::resolve(0, ColorFormat::Luma, depth, scaled)
    }

    #[inline]
    pub(crate) fn apply(&self, sample: i32) -> Result<i32, OutputFormatError> {
        match *self {
            Self::Padding => Ok(0),
            Self::Invalid(ref error) => Err(error.clone()),
            Self::Integer {
                shifted_bias,
                rounding,
                shift,
                post_shift,
            } => {
                let biased = checked_add(sample, shifted_bias)
                    .map_err(|_| OutputFormatError::arithmetic("adding output bias"))?;
                let scaled_sample = checked_add(biased, rounding)
                    .map_err(|_| OutputFormatError::arithmetic("adding scaled-stream rounding"))?
                    >> shift;
                Ok(match post_shift {
                    Some(bits) => saturating_shift_left(scaled_sample, bits),
                    None => scaled_sample,
                })
            }
        }
    }
}

impl ChannelScale {
    /// Inverse of [`Self::apply`]: the reconstructed sample an output value
    /// comes from.
    ///
    /// Applying the scale to the result returns `value` whenever a decoder
    /// can produce it, that is, when its post-scaled low bits are zero.
    #[inline]
    pub(crate) fn unapply(&self, value: i32) -> Result<i32, OutputFormatError> {
        match *self {
            Self::Padding => Ok(0),
            Self::Invalid(ref error) => Err(error.clone()),
            Self::Integer {
                shifted_bias,
                shift,
                post_shift,
                ..
            } => {
                let unscaled = match post_shift {
                    Some(bits) => value >> bits,
                    None => value,
                };
                let shifted = checked_shift_left(unscaled, shift, "scaling input sample")?;
                checked_sub(shifted, shifted_bias)
                    .map_err(|_| OutputFormatError::arithmetic("removing output bias"))
            }
        }
    }
}

/// Inverse of [`float16_bits`]: the reconstructed sample of a half-precision
/// bit pattern, which the format carries as sign and 15-bit magnitude.
pub(crate) fn float16_sample(bits: u16, scaled: bool) -> Result<i32, OutputFormatError> {
    let magnitude = i32::from(bits & 0x7fff);
    let value = if bits & 0x8000 == 0 {
        magnitude
    } else {
        -magnitude
    };
    ChannelScale::resolve(0, ColorFormat::Luma, OutputBitDepth::F16, scaled).unapply(value)
}

/// Inverse of [`float32_bits`]: the reconstructed sample whose decoded value
/// is nearest `bits`.
///
/// The format keeps `mantissa_length` mantissa bits and a biased exponent in
/// one integer. Magnitudes below `2^(1 - exponent_bias)` have no implicit
/// leading one and lose significant bits; rounding may carry into the
/// exponent, which the integer form represents correctly.
pub(crate) fn float32_sample(
    bits: u32,
    scaled: bool,
    mantissa_length: u8,
    exponent_bias: i8,
) -> Result<i32, OutputFormatError> {
    if mantissa_length > 23 {
        return Err(OutputFormatError::UnsupportedCombination {
            combination: "BD32F mantissa length greater than 23",
        });
    }
    let length = u32::from(mantissa_length);
    let mut exponent = i64::from((bits >> 23) & 0xff);
    let mut mantissa = u64::from(bits & 0x7f_ffff);
    let magnitude = if exponent == 0 && mantissa == 0 {
        0
    } else {
        if exponent == 0 {
            // An IEEE denormal: restore the leading one a normal value implies.
            exponent = 1;
            while mantissa < 1 << 23 {
                mantissa <<= 1;
                exponent -= 1;
            }
        } else {
            mantissa |= 1 << 23;
        }
        let coded_exponent = exponent - 127 + i64::from(exponent_bias);
        if coded_exponent >= 1 {
            let fraction = round_shift(mantissa & 0x7f_ffff, 23 - length);
            (coded_exponent << length) + i64::try_from(fraction).unwrap_or(i64::MAX)
        } else {
            // Below the smallest normal coded value: keep the leading one in
            // the mantissa and shift it down to coded exponent zero.
            let shift =
                u32::try_from(i64::from(23 - length) + 1 - coded_exponent).unwrap_or(u32::MAX);
            i64::try_from(round_shift(mantissa, shift)).unwrap_or(i64::MAX)
        }
    };
    let signed = if bits >> 31 == 0 {
        magnitude
    } else {
        -magnitude
    };
    let value = i32::try_from(signed).map_err(|_| OutputFormatError::InvalidFloatingPointSample)?;
    ChannelScale::resolve(
        0,
        ColorFormat::Luma,
        OutputBitDepth::F32 {
            mantissa_length,
            exponent_bias,
        },
        scaled,
    )
    .unapply(value)
}

/// `value / 2^shift` rounded to nearest, ties upward.
const fn round_shift(value: u64, shift: u32) -> u64 {
    match shift {
        0 => value,
        1..=63 => (value + (1 << (shift - 1))) >> shift,
        _ => 0,
    }
}

pub(crate) fn float16_bits(sample: i32, scaled: bool) -> Result<u16, OutputFormatError> {
    let scaled =
        scale_integer_component(sample, 0, ColorFormat::Luma, OutputBitDepth::F16, scaled)?;
    let sign = u16::from(scaled < 0) << 15;
    let magnitude = i64::from(scaled).unsigned_abs().min(32_767);
    Ok(sign | u16::try_from(magnitude).expect("magnitude is clamped to 15 bits"))
}

pub(crate) fn float32_bits(
    sample: i32,
    scaled: bool,
    mantissa_length: u8,
    exponent_bias: i8,
) -> Result<u32, OutputFormatError> {
    if mantissa_length > 23 {
        return Err(OutputFormatError::UnsupportedCombination {
            combination: "BD32F mantissa length greater than 23",
        });
    }
    let scaled = scale_integer_component(
        sample,
        0,
        ColorFormat::Luma,
        OutputBitDepth::F32 {
            mantissa_length,
            exponent_bias,
        },
        scaled,
    )?;
    let sign = u32::from(scaled < 0) << 31;
    let magnitude = i64::from(scaled).unsigned_abs();
    let length = u32::from(mantissa_length);
    let implicit = 1_u64 << length;
    let mut exponent = i64::try_from(magnitude >> length)
        .map_err(|_| OutputFormatError::InvalidFloatingPointSample)?;
    let mut mantissa = (magnitude & (implicit - 1)) | implicit;
    if exponent == 0 {
        mantissa ^= implicit;
        exponent = 1;
    }
    exponent = exponent - i64::from(exponent_bias) + 127;
    while mantissa < implicit && exponent > 1 && mantissa > 0 {
        exponent -= 1;
        mantissa <<= 1;
    }
    if mantissa < implicit {
        exponent = 0;
    } else {
        mantissa ^= implicit;
    }
    mantissa <<= 23 - length;
    let Some(exponent) = u32::try_from(exponent).ok().filter(|&value| value <= 0xff) else {
        // Lossy coding can carry a sample at the top of the range past the
        // largest exponent a float holds. Like an integer sample, it is
        // clipped to the range of its output: the largest finite value.
        return Ok(sign | f32::MAX.to_bits());
    };
    let mantissa = u32::try_from(mantissa)
        .ok()
        .filter(|&value| value <= 0x7f_ffff)
        .ok_or(OutputFormatError::InvalidFloatingPointSample)?;
    Ok(sign | (exponent << 23) | mantissa)
}

fn bias(depth: OutputBitDepth) -> Result<i32, OutputFormatError> {
    let (base, shift) = match depth {
        OutputBitDepth::Rgb555 => (1 << 4, 0),
        OutputBitDepth::Rgb565 => (1 << 5, 0),
        OutputBitDepth::U8 => (1 << 7, 0),
        OutputBitDepth::U10 | OutputBitDepth::Rgb101010 => (1 << 9, 0),
        OutputBitDepth::U16 { shift_bits } => (1 << 15, shift_bits),
        OutputBitDepth::I16 { shift_bits } | OutputBitDepth::I32 { shift_bits } => (0, shift_bits),
        OutputBitDepth::Bit1White
        | OutputBitDepth::Bit1Black
        | OutputBitDepth::F16
        | OutputBitDepth::F32 { .. } => (0, 0),
    };
    if shift >= 32 {
        return Err(OutputFormatError::UnsupportedCombination {
            combination: "integer SHIFT_BITS greater than 31",
        });
    }
    Ok(base >> shift)
}

/// `value << bits`, held at the 32-bit limit when it passes it.
///
/// T.832 postscales a fixed-point sample and then clips it to its output
/// range. No output is wider than 32 bits, so a sample that lossy coding
/// carried past the limit clips to the same value from here.
fn saturating_shift_left(value: i32, bits: u32) -> i32 {
    let limit = if value < 0 { i32::MIN } else { i32::MAX };
    i32::try_from(i64::from(value) << bits.min(31)).unwrap_or(limit)
}

fn checked_shift_left(
    value: i32,
    bits: u32,
    operation: &'static str,
) -> Result<i32, OutputFormatError> {
    value
        .checked_shl(bits)
        .and_then(|shifted| (shifted >> bits == value).then_some(shifted))
        .ok_or_else(|| OutputFormatError::arithmetic(operation))
}

#[cfg(test)]
mod tests {
    use jxr_core::ColorFormat;

    use super::{
        ChannelScale, float16_bits, float16_sample, float32_bits, float32_sample,
        scale_integer_component,
    };
    use crate::output_format::OutputBitDepth;

    #[test]
    fn unapply_inverts_apply_for_every_depth_and_arithmetic() {
        let depths = [
            (OutputBitDepth::U8, 0, 255),
            (OutputBitDepth::U16 { shift_bits: 0 }, 0, 65_535),
            (OutputBitDepth::U16 { shift_bits: 3 }, 0, 65_535),
            (OutputBitDepth::I16 { shift_bits: 0 }, -32_768, 32_767),
            (OutputBitDepth::I32 { shift_bits: 10 }, -(1 << 28), 1 << 28),
            (OutputBitDepth::Rgb555, 0, 31),
            (OutputBitDepth::Rgb565, 0, 31),
            (OutputBitDepth::Rgb101010, 0, 1023),
            (OutputBitDepth::Bit1White, 0, 1),
        ];
        for (depth, low, high) in depths {
            let post = match depth {
                OutputBitDepth::U16 { shift_bits }
                | OutputBitDepth::I16 { shift_bits }
                | OutputBitDepth::I32 { shift_bits } => u32::from(shift_bits),
                _ => 0,
            };
            for scaled in [false, true] {
                for component in 0..3 {
                    let scale = ChannelScale::resolve(component, ColorFormat::Rgb, depth, scaled);
                    for value in [low, low + 1, i32::midpoint(low, high), high - 1, high] {
                        // Only multiples of the post-scale are decoder outputs.
                        let value = (value >> post) << post;
                        let sample = scale.unapply(value).unwrap();
                        assert_eq!(
                            scale.apply(sample).unwrap(),
                            value,
                            "{depth:?} scaled {scaled} component {component}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn half_float_samples_round_trip_every_bit_pattern() {
        for scaled in [false, true] {
            for bits in 0..=u16::MAX {
                let sample = float16_sample(bits, scaled).unwrap();
                // Negative zero has no distinct sign-magnitude sample.
                let expected = if bits == 0x8000 { 0 } else { bits };
                assert_eq!(float16_bits(sample, scaled).unwrap(), expected);
            }
        }
    }

    #[test]
    fn float_samples_invert_the_decoder_mapping_exactly() {
        for (length, bias) in [(13_u8, 4_i8), (12, 3), (10, 0), (23, 4), (0, 4)] {
            let limit = 60_i32 << length.min(20);
            let step = (limit / 4099).max(1);
            let mut sample = -limit;
            while sample <= limit {
                let bits = float32_bits(sample, false, length, bias).unwrap();
                assert_eq!(
                    float32_sample(bits, false, length, bias).unwrap(),
                    sample,
                    "mantissa {length} bias {bias} bits {bits:#x}"
                );
                sample += step;
            }
        }
    }

    #[test]
    fn arbitrary_floats_decode_within_the_kept_mantissa_precision() {
        let (length, bias) = (13_u8, 4_i8);
        for value in [
            1.0_f32,
            -1.0,
            0.5,
            0.123_456_79,
            3.999_999,
            1234.567,
            -0.000_976,
            1.0e-6,
            65_504.0,
        ] {
            let sample = float32_sample(value.to_bits(), false, length, bias).unwrap();
            let decoded = f32::from_bits(float32_bits(sample, false, length, bias).unwrap());
            // Normal coded values keep 13 mantissa bits; smaller magnitudes
            // keep an absolute step of 2^(1 - bias - length).
            let tolerance = (value.abs() * 2.0_f32.powi(-13)).max(2.0_f32.powi(-17));
            assert!(
                (decoded - value).abs() <= tolerance,
                "{value} decoded as {decoded}"
            );
        }
        assert_eq!(float32_sample(0, false, length, bias).unwrap(), 0);
        assert_eq!(float32_sample(0x8000_0000, false, length, bias).unwrap(), 0);
    }

    #[test]
    fn scaled_u8_adds_bias_then_rounds_down() {
        assert_eq!(
            scale_integer_component(-1, 0, ColorFormat::Luma, OutputBitDepth::U8, true).unwrap(),
            128
        );
        assert_eq!(
            scale_integer_component(-5, 0, ColorFormat::Luma, OutputBitDepth::U8, true).unwrap(),
            127
        );
    }

    #[test]
    fn integer_postshift_follows_scaling() {
        assert_eq!(
            scale_integer_component(
                7,
                0,
                ColorFormat::Luma,
                OutputBitDepth::I16 { shift_bits: 3 },
                true,
            )
            .unwrap(),
            8
        );
    }

    #[test]
    fn a_postscaled_sample_past_the_32_bit_limit_is_clipped_to_it() {
        let scale = |sample| {
            scale_integer_component(
                sample,
                0,
                ColorFormat::Luma,
                OutputBitDepth::I32 { shift_bits: 10 },
                false,
            )
            .unwrap()
        };
        assert_eq!(scale((1 << 21) - 1), i32::MAX & !0x3ff);
        assert_eq!(scale(1 << 21), i32::MAX);
        assert_eq!(scale(i32::MAX), i32::MAX);
        assert_eq!(scale(-(1 << 21)), i32::MIN);
        assert_eq!(scale(-(1 << 21) - 1), i32::MIN);
    }

    #[test]
    fn an_f32_exponent_past_the_format_is_clipped_to_the_largest_finite_value() {
        // With four mantissa bits and a bias of one, a coded exponent of 129
        // is the last a float holds.
        assert_eq!(
            float32_bits(129 << 4, false, 4, 1).unwrap(),
            f32::INFINITY.to_bits()
        );
        assert_eq!(
            float32_bits(130 << 4, false, 4, 1).unwrap(),
            f32::MAX.to_bits()
        );
        assert_eq!(
            float32_bits(-(130 << 4), false, 4, 1).unwrap(),
            (-f32::MAX).to_bits()
        );
        assert_eq!(
            float32_bits(i32::MAX, false, 4, 1).unwrap(),
            f32::MAX.to_bits()
        );
    }

    #[test]
    fn f16_is_sign_and_clamped_magnitude() {
        assert_eq!(float16_bits(-40_000, false).unwrap(), 0xffff);
        assert_eq!(float16_bits(17, false).unwrap(), 17);
    }

    #[test]
    fn f32_reconstructs_ieee_fields() {
        assert_eq!(
            float32_bits(1 << 4, false, 4, 1).unwrap(),
            1.0_f32.to_bits()
        );
        assert_eq!(
            float32_bits(-(1 << 4), false, 4, 1).unwrap(),
            (-1.0_f32).to_bits()
        );
    }
}
