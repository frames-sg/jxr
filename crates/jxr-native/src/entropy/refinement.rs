//! LP coefficient refinement and HP flexbits parsing.

use super::{EntropyError, PacketBitReader};

const TRANSPOSE: [usize; 16] = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];

/// Decode one HP flexbits value as specified by T.832 Table 85.
pub fn decode_flex(
    reader: &mut PacketBitReader<'_>,
    vlc_coefficient: i32,
    flex_bits: u8,
) -> Result<i32, EntropyError> {
    if flex_bits > 29 {
        return Err(EntropyError::InvalidParameter {
            parameter: "flexbits width",
            value: i64::from(flex_bits),
        });
    }
    let value = i32::try_from(reader.read_bits(flex_bits)?)
        .map_err(|_| EntropyError::CoefficientOverflow)?;
    match vlc_coefficient.cmp(&0) {
        core::cmp::Ordering::Greater => Ok(value),
        core::cmp::Ordering::Less => signed(value, true),
        core::cmp::Ordering::Equal if value == 0 => Ok(0),
        core::cmp::Ordering::Equal => apply_sign(reader, value),
    }
}

/// Decode all 15 flexbits positions of an HP block in transpose order.
pub fn decode_flex_block(
    reader: &mut PacketBitReader<'_>,
    vlc_coefficients: &[i32; 16],
    model_bits: u8,
    trim_flexbits: u8,
) -> Result<[i32; 16], EntropyError> {
    if model_bits > 15 {
        return Err(EntropyError::InvalidParameter {
            parameter: "HP model bits",
            value: i64::from(model_bits),
        });
    }
    if trim_flexbits > 15 {
        return Err(EntropyError::InvalidParameter {
            parameter: "trim flexbits",
            value: i64::from(trim_flexbits),
        });
    }
    let bits_left = model_bits.saturating_sub(trim_flexbits);
    let mut output = [0_i32; 16];
    if bits_left == 0 {
        return Ok(output);
    }
    for &position in &TRANSPOSE[1..] {
        let flex = decode_flex(reader, vlc_coefficients[position], bits_left)?;
        output[position] = flex
            .checked_shl(u32::from(trim_flexbits))
            .ok_or(EntropyError::CoefficientOverflow)?;
    }
    Ok(output)
}

/// Refine the fifteen LP positions in the normative 4-by-4 transpose order.
pub fn decode_lp_refinement(
    reader: &mut PacketBitReader<'_>,
    coefficients: &mut [i32; 16],
    model_bits: u8,
) -> Result<(), EntropyError> {
    decode_lp_refinement_at(reader, coefficients, &TRANSPOSE[1..], model_bits)
}

/// Refine selected LP positions in the supplied normative traversal order.
pub fn decode_lp_refinement_at(
    reader: &mut PacketBitReader<'_>,
    coefficients: &mut [i32; 16],
    positions: &[usize],
    model_bits: u8,
) -> Result<(), EntropyError> {
    for &position in positions {
        if position == 0 || position >= coefficients.len() {
            return Err(EntropyError::InvalidParameter {
                parameter: "LP refinement position",
                value: i64::try_from(position).unwrap_or(i64::MAX),
            });
        }
        let refinement = decode_flex(reader, coefficients[position], model_bits)?;
        coefficients[position] = coefficients[position]
            .checked_shl(u32::from(model_bits))
            .and_then(|value| value.checked_add(refinement))
            .ok_or(EntropyError::CoefficientOverflow)?;
    }
    Ok(())
}

fn apply_sign(reader: &mut PacketBitReader<'_>, value: i32) -> Result<i32, EntropyError> {
    if value == 0 {
        return Ok(0);
    }
    signed(value, reader.read_bit()?)
}

fn signed(magnitude: i32, negative: bool) -> Result<i32, EntropyError> {
    if negative {
        magnitude
            .checked_neg()
            .ok_or(EntropyError::CoefficientOverflow)
    } else {
        Ok(magnitude)
    }
}

#[cfg(test)]
mod tests {
    use super::{decode_flex, decode_flex_block};
    use crate::entropy::PacketBitReader;

    #[test]
    fn flex_sign_follows_vlc_part_or_explicit_sign() {
        let mut positive = PacketBitReader::with_bit_length(&[0b1100_0000], 2).unwrap();
        assert_eq!(decode_flex(&mut positive, 3, 2).unwrap(), 3);
        let mut negative = PacketBitReader::with_bit_length(&[0b1100_0000], 2).unwrap();
        assert_eq!(decode_flex(&mut negative, -3, 2).unwrap(), -3);
        let mut zero_vlc = PacketBitReader::with_bit_length(&[0b1110_0000], 3).unwrap();
        assert_eq!(decode_flex(&mut zero_vlc, 0, 2).unwrap(), -3);
    }

    #[test]
    fn zero_bits_flex_block_consumes_nothing() {
        let mut reader = PacketBitReader::with_bit_length(&[], 0).unwrap();
        assert_eq!(
            decode_flex_block(&mut reader, &[0; 16], 3, 3).unwrap(),
            [0; 16]
        );
    }
}
