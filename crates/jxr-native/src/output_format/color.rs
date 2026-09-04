//! Reversible internal-to-output colour conversions from T.832 clause 9.10.4.

use jxr_core::ColorFormat;
use jxr_math::arithmetic::{checked_add, checked_sub};

use super::OutputFormatError;

pub(crate) fn convert(
    internal: ColorFormat,
    output: ColorFormat,
    input: &[i32],
    packed_rgb: bool,
    red_blue_not_swapped: bool,
) -> Result<[i32; 4], OutputFormatError> {
    match (internal, output) {
        (ColorFormat::Luma, ColorFormat::Luma) => Ok([input[0], 0, 0, 0]),
        (ColorFormat::Luma, ColorFormat::Rgb) => Ok([input[0], input[0], input[0], 0]),
        (
            ColorFormat::Yuv(jxr_core::ChromaSampling::Cs444),
            ColorFormat::Rgb | ColorFormat::Rgbe,
        ) => {
            let mut rgb = inverse_yuv(input[0], input[1], input[2])?;
            if packed_rgb && !red_blue_not_swapped {
                rgb.swap(0, 2);
            }
            Ok([rgb[0], rgb[1], rgb[2], 0])
        }
        (ColorFormat::YuvK, ColorFormat::Cmyk) => inverse_yuvk(input),
        (ColorFormat::YuvK, ColorFormat::CmykDirect) => {
            Ok([input[1], input[2], input[3], input[0]])
        }
        (ColorFormat::NComponent(input_count), ColorFormat::NComponent(output_count))
            if input_count == output_count =>
        {
            Err(OutputFormatError::UnsupportedCombination {
                combination: "N-component conversion requires direct component handling",
            })
        }
        _ if internal == output => Err(OutputFormatError::UnsupportedCombination {
            combination: "direct conversion requires component handling",
        }),
        _ => Err(OutputFormatError::UnsupportedCombination {
            combination: "internal-to-output colour conversion",
        }),
    }
}

fn inverse_yuv(y: i32, u: i32, v: i32) -> Result<[i32; 3], OutputFormatError> {
    let temporary = checked_sub(0, u).map_err(math_error)?;
    let green = checked_sub(y, temporary >> 1).map_err(math_error)?;
    let red = checked_add(temporary, green)
        .and_then(|value| checked_sub(value, ceil_half(v)))
        .map_err(math_error)?;
    let blue = checked_add(v, red).map_err(math_error)?;
    Ok([red, green, blue])
}

fn inverse_yuvk(input: &[i32]) -> Result<[i32; 4], OutputFormatError> {
    let [y, u, v, k] = [input[0], input[1], input[2], input[3]];
    let black = checked_add(k, y >> 1).map_err(math_error)?;
    let magenta = checked_sub(black, y)
        .and_then(|value| checked_sub(value, u >> 1))
        .map_err(math_error)?;
    let cyan = checked_add(u, magenta)
        .and_then(|value| checked_add(value, v >> 1))
        .map_err(math_error)?;
    let yellow = checked_sub(cyan, v).map_err(math_error)?;
    Ok([cyan, magenta, yellow, black])
}

const fn ceil_half(value: i32) -> i32 {
    (value >> 1) + (value & 1)
}

fn math_error(_: jxr_math::arithmetic::MathError) -> OutputFormatError {
    OutputFormatError::arithmetic("converting internal colour")
}

#[cfg(test)]
mod tests {
    use jxr_core::{ChromaSampling, ColorFormat};

    use super::convert;

    #[test]
    fn yuv_inverse_uses_normative_odd_rounding() {
        let converted = convert(
            ColorFormat::Yuv(ChromaSampling::Cs444),
            ColorFormat::Rgb,
            &[10, 3, 5],
            false,
            false,
        )
        .unwrap();
        assert_eq!(converted, [6, 12, 11, 0]);
    }

    #[test]
    fn packed_legacy_order_swaps_red_and_blue() {
        let converted = convert(
            ColorFormat::Yuv(ChromaSampling::Cs444),
            ColorFormat::Rgb,
            &[10, 3, 5],
            true,
            false,
        )
        .unwrap();
        assert_eq!(converted, [11, 12, 6, 0]);
    }

    #[test]
    fn yuvk_converts_to_cmyk() {
        let converted = convert(
            ColorFormat::YuvK,
            ColorFormat::Cmyk,
            &[20, 4, 6, 8],
            false,
            false,
        )
        .unwrap();
        assert_eq!(converted, [3, -4, -3, 18]);
    }
}
