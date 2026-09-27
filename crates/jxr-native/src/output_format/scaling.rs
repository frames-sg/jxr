//! Bias, scaled-stream rounding, and postscaling from T.832 clauses 9.10.5-9.10.7.

use jxr_core::ColorFormat;
use jxr_math::arithmetic::checked_add;

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
                match post_shift {
                    Some(bits) => checked_shift_left(scaled_sample, bits, "integer postscaling"),
                    None => Ok(scaled_sample),
                }
            }
        }
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
    let exponent = u32::try_from(exponent)
        .ok()
        .filter(|&value| value <= 0xff)
        .ok_or(OutputFormatError::InvalidFloatingPointSample)?;
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

    use super::{float16_bits, float32_bits, scale_integer_component};
    use crate::output_format::OutputBitDepth;

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
