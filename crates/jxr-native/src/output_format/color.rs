//! Reversible colour conversions of T.832 clause 9.10.4, in both directions.

use jxr_core::ColorFormat;
use jxr_math::arithmetic::{checked_add, checked_sub};

use super::OutputFormatError;
use crate::reconstruct::Lane;

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

/// Inverse of [`convert`]: output-colour components to internal components.
///
/// `output` holds the semantic channels of the output colour format, red
/// first for RGB; storage order and packing are the caller's concern. Only
/// the pairs `convert` implements are accepted.
pub(crate) fn convert_forward(
    internal: ColorFormat,
    output: ColorFormat,
    values: [i32; 4],
) -> Result<[i32; 4], OutputFormatError> {
    match (internal, output) {
        (ColorFormat::Luma, ColorFormat::Luma) => Ok([values[0], 0, 0, 0]),
        (
            ColorFormat::Yuv(jxr_core::ChromaSampling::Cs444),
            ColorFormat::Rgb | ColorFormat::Rgbe,
        ) => forward_yuv(values[0], values[1], values[2]),
        (ColorFormat::YuvK, ColorFormat::Cmyk) => forward_yuvk(values),
        (ColorFormat::YuvK, ColorFormat::CmykDirect) => {
            Ok([values[3], values[0], values[1], values[2]])
        }
        _ => Err(OutputFormatError::UnsupportedCombination {
            combination: "output-to-internal colour conversion",
        }),
    }
}

/// Exact inverse of [`inverse_yuv`].
fn forward_yuv(red: i32, green: i32, blue: i32) -> Result<[i32; 4], OutputFormatError> {
    let v = checked_sub(blue, red).map_err(math_error)?;
    let temporary = checked_sub(red, green)
        .and_then(|value| checked_add(value, ceil_half(v)))
        .map_err(math_error)?;
    let y = checked_add(green, temporary >> 1).map_err(math_error)?;
    let u = checked_sub(0, temporary).map_err(math_error)?;
    Ok([y, u, v, 0])
}

/// Exact inverse of [`inverse_yuvk`].
fn forward_yuvk(input: [i32; 4]) -> Result<[i32; 4], OutputFormatError> {
    let [cyan, magenta, yellow, black] = input;
    let v = checked_sub(cyan, yellow).map_err(math_error)?;
    let u = checked_sub(cyan, magenta)
        .and_then(|value| checked_sub(value, v >> 1))
        .map_err(math_error)?;
    let y = checked_sub(black, magenta)
        .and_then(|value| checked_sub(value, u >> 1))
        .map_err(math_error)?;
    let k = checked_sub(black, y >> 1).map_err(math_error)?;
    Ok([y, u, v, k])
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

/// Exclusive magnitude bound on the inputs of [`forward_yuv_lanes`] below
/// which no intermediate leaves `i32`: from inputs below `m`, `blue - red`
/// and `red - green` are below `2m`, its half rounded up below `m`, their
/// sum below `3m`, and `3 * 2^29 < 2^31`.
pub(crate) const FORWARD_YUV_BOUND: u32 = 1 << 29;

/// [`forward_yuv`] on lanes, with their wrapping arithmetic: the same values
/// for inputs below [`FORWARD_YUV_BOUND`] in magnitude. `value - (value >>
/// 1)` is the half of `value` rounded up.
#[inline]
pub(crate) fn forward_yuv_lanes<L: Lane>([red, green, blue]: [L; 3]) -> [L; 3] {
    let v = blue.sub(red);
    let temporary = red.sub(green).add(v.sub(v.shr(1)));
    let y = green.add(temporary.shr(1));
    [y, temporary.neg(), v]
}

/// [`inverse_yuv`] on lanes, with their wrapping arithmetic: the same values
/// wherever no intermediate leaves `i32`.
#[inline]
pub(crate) fn inverse_yuv_lanes<L: Lane>([y, u, v]: [L; 3]) -> [L; 3] {
    let temporary = u.neg();
    let green = y.sub(temporary.shr(1));
    let red = temporary.add(green).sub(v.sub(v.shr(1)));
    [red, green, v.add(red)]
}

fn math_error(_: jxr_math::arithmetic::MathError) -> OutputFormatError {
    OutputFormatError::arithmetic("converting internal colour")
}

#[cfg(test)]
mod tests {
    use jxr_core::{ChromaSampling, ColorFormat};

    use super::{convert, convert_forward};

    #[test]
    fn forward_conversions_invert_the_decoder_conversions() {
        // Every parity combination and both extremes, unscaled and scaled.
        let values = [-1024, -1017, -128, -3, -2, -1, 0, 1, 2, 3, 127, 1016];
        let pairs = [
            (ColorFormat::Yuv(ChromaSampling::Cs444), ColorFormat::Rgb),
            (ColorFormat::YuvK, ColorFormat::Cmyk),
            (ColorFormat::YuvK, ColorFormat::CmykDirect),
        ];
        for (internal, output) in pairs {
            for a in values {
                for b in values {
                    for c in values {
                        let source = [a, b, c, a ^ c];
                        let coded = convert_forward(internal, output, source).unwrap();
                        let decoded = convert(internal, output, &coded, false, false).unwrap();
                        let channels = if internal == ColorFormat::YuvK { 4 } else { 3 };
                        assert_eq!(decoded[..channels], source[..channels], "{output:?}");
                    }
                }
            }
        }
        assert_eq!(
            convert_forward(
                ColorFormat::Yuv(ChromaSampling::Cs444),
                ColorFormat::Rgb,
                [40, 40, 40, 0]
            )
            .unwrap(),
            [40, 0, 0, 0]
        );
    }

    #[test]
    fn lane_forms_match_the_checked_conversions_within_their_bound() {
        let limit = i32::try_from(super::FORWARD_YUV_BOUND).unwrap() - 1;
        let values = [
            -limit,
            -limit + 1,
            -1_000_001,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            999_999,
            limit - 1,
            limit,
        ];
        for red in values {
            for green in values {
                for blue in values {
                    let coded = super::forward_yuv(red, green, blue).unwrap();
                    assert_eq!(
                        super::forward_yuv_lanes([red, green, blue]),
                        [coded[0], coded[1], coded[2]],
                        "{red} {green} {blue}"
                    );
                    // The inverse of what the forward form produced: every
                    // value a decoder of such a source reconstructs.
                    assert_eq!(
                        super::inverse_yuv_lanes([coded[0], coded[1], coded[2]]),
                        [red, green, blue]
                    );
                    if let Ok(rgb) = super::inverse_yuv(red, green, blue) {
                        assert_eq!(super::inverse_yuv_lanes([red, green, blue]), rgb);
                    }
                }
            }
        }
    }

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
