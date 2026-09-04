//! Exact alpha-premultiplication arithmetic for decoded output samples.

use crate::arithmetic::MathError;

/// Premultiply an unsigned channel with nearest rounding.
pub fn premultiply_unsigned(value: u32, alpha: u32, maximum: u32) -> Result<u32, MathError> {
    if maximum == 0 {
        return Err(MathError::overflow("zero alpha normalization range"));
    }
    let value = value.min(maximum);
    let alpha = alpha.min(maximum);
    let product = u64::from(value) * u64::from(alpha);
    u32::try_from((product + u64::from(maximum / 2)) / u64::from(maximum))
        .map_err(|_| MathError::overflow("unsigned alpha premultiplication"))
}

/// Premultiply a signed channel by a nonnegative signed alpha channel.
pub fn premultiply_signed(value: i32, alpha: i32, alpha_maximum: i32) -> Result<i32, MathError> {
    if alpha_maximum <= 0 {
        return Err(MathError::overflow(
            "invalid signed alpha normalization range",
        ));
    }
    let alpha = alpha.clamp(0, alpha_maximum);
    let magnitude = i64::from(value).unsigned_abs();
    let product = magnitude * u64::try_from(alpha).expect("clamped alpha is nonnegative");
    let maximum = u64::try_from(alpha_maximum).expect("validated alpha maximum is positive");
    let magnitude = (product + maximum / 2) / maximum;
    let signed = i64::try_from(magnitude)
        .map_err(|_| MathError::overflow("signed alpha premultiplication"))?;
    let signed = if value < 0 { -signed } else { signed };
    i32::try_from(signed).map_err(|_| MathError::overflow("signed alpha premultiplication"))
}

/// Premultiply one T.832 BD16F sign-magnitude value by another used as alpha.
pub fn premultiply_sign_magnitude_15(value_bits: u16, alpha_bits: u16) -> Result<u16, MathError> {
    let sign = value_bits & 0x8000;
    let value = u32::from(value_bits & 0x7fff);
    let alpha = if alpha_bits & 0x8000 == 0 {
        u32::from(alpha_bits & 0x7fff)
    } else {
        0
    };
    let magnitude = premultiply_unsigned(value, alpha, 0x7fff)?;
    Ok(sign | u16::try_from(magnitude).expect("15-bit premultiplied magnitude"))
}

/// Premultiply a floating-point channel by alpha clamped to `[0, 1]`.
#[must_use]
pub fn premultiply_f32(value: f32, alpha: f32) -> f32 {
    value * alpha.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::{
        premultiply_f32, premultiply_sign_magnitude_15, premultiply_signed, premultiply_unsigned,
    };

    #[test]
    fn unsigned_uses_nearest_normalized_rounding() {
        assert_eq!(premultiply_unsigned(255, 128, 255).unwrap(), 128);
        assert_eq!(premultiply_unsigned(7, 0, 255).unwrap(), 0);
    }

    #[test]
    fn signed_preserves_color_sign_and_clamps_alpha() {
        assert_eq!(
            premultiply_signed(-20_000, 16_384, 32_767).unwrap(),
            -10_000
        );
        assert_eq!(premultiply_signed(20_000, -1, 32_767).unwrap(), 0);
    }

    #[test]
    fn sign_magnitude_preserves_color_sign() {
        assert_eq!(
            premultiply_sign_magnitude_15(0x8000 | 0x4e20, 16_384).unwrap(),
            0x8000 | 0x2710
        );
    }

    #[test]
    fn float_alpha_is_normalized() {
        assert_eq!(premultiply_f32(-4.0, 0.25).to_bits(), (-1.0_f32).to_bits());
        assert_eq!(premultiply_f32(4.0, 2.0).to_bits(), 4.0_f32.to_bits());
    }
}
