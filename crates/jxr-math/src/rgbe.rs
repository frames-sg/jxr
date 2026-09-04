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
    use super::{pack_rgbe, postscale_rgbe};

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
