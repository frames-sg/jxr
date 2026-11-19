//! T.832 RGBE postscaling and packing arithmetic.

use crate::color::clamp_sample;

/// Convert reconstructed RGB integers to clipped RGBE bytes.
#[must_use]
pub fn postscale_rgbe(rgb: [i32; 3]) -> [u8; 4] {
    let encoded = rgb.map(encode_component);
    let exponent = encoded
        .iter()
        .map(|component| component.exponent)
        .max()
        .unwrap_or(0);
    let mut output = [0_u8; 4];
    for (index, component) in encoded.into_iter().enumerate() {
        let value = if exponent > component.exponent {
            rounded_shift(component.mantissa, exponent - component.exponent)
        } else {
            component.mantissa
        };
        output[index] = u8::try_from(clamp_sample(value, 0, 255)).expect("RGBE byte is clipped");
    }
    output[3] = u8::try_from(exponent.min(255)).expect("RGBE exponent is clipped");
    output
}

/// Convert RGBE bytes to the RGB integers [`postscale_rgbe`] maps back to them.
///
/// Each channel is normalized on its own: its mantissa is shifted up until
/// the leading bit is set or its exponent reaches one. The postscale then
/// realigns the channels to their largest exponent, which restores the
/// original bytes.
#[must_use]
pub const fn prescale_rgbe(bytes: [u8; 4]) -> [i32; 3] {
    let mut rgb = [0_i32; 3];
    let mut channel = 0;
    while channel < 3 {
        let mut mantissa = bytes[channel] as i32;
        let mut exponent = bytes[3] as i32;
        // A zero exponent only ever accompanies a zero pixel.
        if mantissa != 0 && exponent != 0 {
            while mantissa < 128 && exponent > 1 {
                mantissa <<= 1;
                exponent -= 1;
            }
            rgb[channel] = if exponent > 1 {
                (exponent << 7) | (mantissa & 0x7f)
            } else {
                mantissa
            };
        }
        channel += 1;
    }
    rgb
}

/// Split Annex-A RGBE storage into semantic `[R, G, B, E]` bytes.
#[must_use]
pub const fn unpack_rgbe(word: u32) -> [u8; 4] {
    word.to_le_bytes()
}

/// Pack semantic `[R, G, B, E]` as Annex-A bytes in the same order.
#[must_use]
pub const fn pack_rgbe(bytes: [u8; 4]) -> u32 {
    u32::from_le_bytes(bytes)
}

#[derive(Clone, Copy)]
struct EncodedComponent {
    mantissa: i32,
    exponent: i32,
}

fn encode_component(value: i32) -> EncodedComponent {
    if value <= 0 {
        EncodedComponent {
            mantissa: 0,
            exponent: 0,
        }
    } else if value >> 7 > 1 {
        EncodedComponent {
            mantissa: (value & 0x7f) + 128,
            exponent: value >> 7,
        }
    } else {
        EncodedComponent {
            mantissa: value,
            exponent: 1,
        }
    }
}

fn rounded_shift(value: i32, exponent_difference: i32) -> i32 {
    let shift = u32::try_from(exponent_difference).unwrap_or(u32::MAX);
    if shift >= 31 {
        0
    } else {
        (2 * value + 1) >> (shift + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::{pack_rgbe, postscale_rgbe, prescale_rgbe, unpack_rgbe};

    #[test]
    fn prescale_inverts_postscale_for_every_valid_pixel() {
        // A canonical pixel has a mantissa with its leading bit set, unless
        // its exponent is already the smallest. Black is all zero bytes.
        let mantissas = [0_u8, 1, 2, 63, 64, 127, 128, 129, 200, 255];
        for exponent in [1_u8, 2, 3, 100, 128, 254, 255] {
            for red in mantissas {
                for green in mantissas {
                    for blue in mantissas {
                        let brightest = red.max(green).max(blue);
                        let canonical = brightest >= 128 || (exponent == 1 && brightest != 0);
                        if !canonical {
                            continue;
                        }
                        let bytes = [red, green, blue, exponent];
                        assert_eq!(postscale_rgbe(prescale_rgbe(bytes)), bytes);
                    }
                }
            }
        }
        assert_eq!(postscale_rgbe(prescale_rgbe([0; 4])), [0; 4]);
        assert_eq!(prescale_rgbe([9, 9, 9, 0]), [0; 3]);
        assert_eq!(unpack_rgbe(pack_rgbe([1, 2, 3, 4])), [1, 2, 3, 4]);
    }

    #[test]
    fn chooses_largest_exponent_and_aligns_other_components() {
        assert_eq!(postscale_rgbe([128, 256, 64]), [64, 128, 32, 2]);
    }

    #[test]
    fn clips_nonpositive_and_large_fields() {
        assert_eq!(postscale_rgbe([-1, 0, 127]), [0, 0, 127, 1]);
        assert_eq!(postscale_rgbe([i32::MAX, 0, 0])[3], 255);
    }

    #[test]
    fn packs_channels_in_annex_a_rgbe_byte_order() {
        assert_eq!(pack_rgbe([1, 2, 3, 4]), 0x0403_0201);
    }
}
