//! LP coefficient refinement and HP flexbits writing.

use crate::bit_writer::BitWriter;

use super::super::{EntropyError, refinement::TRANSPOSE};

/// The VLC-coded part of `coefficient` for a band with `model_bits`.
///
/// A coefficient is coded as this part followed by `model_bits` refinement
/// bits of its magnitude. Splitting the magnitude keeps the two parts the
/// same sign, which is the only form the refinement syntax can express.
#[must_use]
pub(crate) const fn vlc_part(coefficient: i32, model_bits: u8) -> i32 {
    let part = (coefficient.unsigned_abs() >> model_bits).cast_signed();
    // Wrapping keeps `i32::MIN` with zero model bits, the one magnitude
    // whose signed cast is already negative, from overflowing here;
    // `check_magnitudes` refuses it before anything is written.
    if coefficient < 0 {
        part.wrapping_neg()
    } else {
        part
    }
}

/// Refuses a coefficient whose magnitude a decoder cannot hold.
///
/// Every band is read as a magnitude in a signed 32-bit value and a sign,
/// so two to the power 31 has no coded form, although the VLC escape and
/// the refinement bits together are wide enough to spell it.
pub(crate) fn check_magnitudes(coefficients: &[i32]) -> Result<(), EntropyError> {
    if coefficients.contains(&i32::MIN) {
        return Err(EntropyError::CoefficientOverflow);
    }
    Ok(())
}

/// The refinement of one coefficient without its `trim` lowest bits, as
/// `(bits, length)`.
///
/// The sign is included, as the last bit, only when the VLC part is zero
/// and the retained refinement is not, matching T.832 Table 85. Selecting
/// it without a branch matters: whether a coefficient carries its sign here
/// is as unpredictable as the coefficient.
#[inline]
fn flex_code(coefficient: i32, model_bits: u8, trim: u8) -> (u32, u8) {
    let magnitude = coefficient.unsigned_abs();
    let refinement = (magnitude & ((1_u32 << model_bits) - 1)) >> trim;
    // One when the sign is written, as an integer so that nothing branches.
    let signed = u8::from(magnitude >> model_bits == 0) & u8::from(refinement != 0);
    let negative = signed & u8::from(coefficient < 0);
    (
        (refinement << signed) | u32::from(negative),
        model_bits - trim + signed,
    )
}

/// Writes the refinements of the coefficients at `positions`, in that
/// order, as many to a write as 32 bits are sure to hold. `model_bits` is
/// at most fifteen and above `trim`, and every position below sixteen.
#[inline]
fn encode_flex_positions(
    writer: &mut BitWriter,
    coefficients: &[i32; 16],
    positions: &[usize],
    model_bits: u8,
    trim: u8,
) {
    // A refinement and its sign are at most sixteen bits.
    let per_write = usize::from(32 / (model_bits - trim + 1));
    for group in positions.chunks(per_write) {
        let mut bits = 0_u64;
        let mut length = 0_u8;
        for &position in group {
            let (code, code_length) = flex_code(coefficients[position & 15], model_bits, trim);
            bits = bits << code_length | u64::from(code);
            length += code_length;
        }
        writer.write_bits(bits, length);
    }
}

/// Writes the flexbits of the 15 HP positions of a block in transpose order.
///
/// `coefficients` holds the block's complete coefficients; element zero is
/// the block's DC/LP value and is not written.
#[inline]
pub(crate) fn encode_flex_block(
    writer: &mut BitWriter,
    coefficients: &[i32; 16],
    model_bits: u8,
    trim_flexbits: u8,
) -> Result<(), EntropyError> {
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
    if model_bits <= trim_flexbits {
        return Ok(());
    }
    encode_flex_positions(
        writer,
        coefficients,
        &TRANSPOSE[1..],
        model_bits,
        trim_flexbits,
    );
    Ok(())
}

/// Writes the refinements of the fifteen LP positions in transpose order.
#[inline]
pub(crate) fn encode_lp_refinement(
    writer: &mut BitWriter,
    coefficients: &[i32; 16],
    model_bits: u8,
) -> Result<(), EntropyError> {
    encode_lp_refinement_at(writer, coefficients, &TRANSPOSE[1..], model_bits)
}

/// Writes the refinements of the LP positions `positions`, in that order:
/// the inverse of [`super::super::decode_lp_refinement_at`].
#[inline]
pub(crate) fn encode_lp_refinement_at(
    writer: &mut BitWriter,
    coefficients: &[i32; 16],
    positions: &[usize],
    model_bits: u8,
) -> Result<(), EntropyError> {
    if model_bits > 15 {
        return Err(EntropyError::InvalidParameter {
            parameter: "LP model bits",
            value: i64::from(model_bits),
        });
    }
    if let Some(&position) = positions
        .iter()
        .find(|&&position| position == 0 || position >= coefficients.len())
    {
        return Err(EntropyError::InvalidParameter {
            parameter: "LP refinement position",
            value: i64::try_from(position).unwrap_or(i64::MAX),
        });
    }
    if model_bits == 0 {
        return Ok(());
    }
    encode_flex_positions(writer, coefficients, positions, model_bits, 0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::super::{PacketBitReader, decode_flex_block, decode_lp_refinement};
    use super::super::test_support::Lcg;
    use super::*;

    fn random_block(rng: &mut Lcg) -> [i32; 16] {
        core::array::from_fn(|_| rng.coefficient(20))
    }

    #[test]
    fn vlc_part_truncates_the_magnitude_toward_zero() {
        assert_eq!(vlc_part(13, 2), 3);
        assert_eq!(vlc_part(-13, 2), -3);
        assert_eq!(vlc_part(-3, 2), 0);
        assert_eq!(vlc_part(i32::MIN, 0), i32::MIN);
    }

    #[test]
    fn a_magnitude_no_decoder_can_hold_is_refused() {
        assert_eq!(check_magnitudes(&[0, i32::MAX, -i32::MAX]), Ok(()));
        assert_eq!(
            check_magnitudes(&[0, i32::MIN, 1]),
            Err(EntropyError::CoefficientOverflow)
        );
    }

    #[test]
    fn lp_refinement_round_trips_for_every_model_width() {
        let mut rng = Lcg(4);
        for model_bits in 0..=15_u8 {
            for _ in 0..200 {
                let block = random_block(&mut rng);
                let mut writer = BitWriter::new();
                encode_lp_refinement(&mut writer, &block, model_bits).unwrap();
                let bit_length = writer.bit_position();
                let bytes = writer.into_bytes();
                let mut decoded = block.map(|value| vlc_part(value, model_bits));
                decoded[0] = block[0];
                let mut reader = PacketBitReader::with_bit_length(&bytes, bit_length).unwrap();
                decode_lp_refinement(&mut reader, &mut decoded, model_bits).unwrap();
                assert_eq!(decoded, block);
                assert_eq!(reader.bits_remaining(), 0);
            }
        }
    }

    #[test]
    fn lp_refinement_of_chosen_positions_round_trips_in_the_order_given() {
        use super::super::super::decode_lp_refinement_at;

        let mut rng = Lcg(7);
        let positions = [2_usize, 1, 3, 4, 6, 5, 7];
        for model_bits in 0..=15_u8 {
            for _ in 0..50 {
                let block = random_block(&mut rng);
                let mut writer = BitWriter::new();
                encode_lp_refinement_at(&mut writer, &block, &positions, model_bits).unwrap();
                let bit_length = writer.bit_position();
                let bytes = writer.into_bytes();
                // Positions outside the list are not refined.
                let mut decoded = block;
                for &position in &positions {
                    decoded[position] = vlc_part(block[position], model_bits);
                }
                let mut reader = PacketBitReader::with_bit_length(&bytes, bit_length).unwrap();
                decode_lp_refinement_at(&mut reader, &mut decoded, &positions, model_bits).unwrap();
                assert_eq!(decoded, block);
                assert_eq!(reader.bits_remaining(), 0);
            }
        }
        let mut writer = BitWriter::new();
        for invalid in [0_usize, 16] {
            assert!(encode_lp_refinement_at(&mut writer, &[0; 16], &[invalid], 3).is_err());
        }
        assert_eq!(writer.bit_position(), 0);
    }

    #[test]
    fn flexbits_round_trip_up_to_the_trimmed_bits() {
        let mut rng = Lcg(5);
        for model_bits in 0..=15_u8 {
            for trim in 0..=15_u8 {
                for _ in 0..20 {
                    let block = random_block(&mut rng);
                    let mut writer = BitWriter::new();
                    encode_flex_block(&mut writer, &block, model_bits, trim).unwrap();
                    let bit_length = writer.bit_position();
                    let bytes = writer.into_bytes();
                    let vlc = block.map(|value| vlc_part(value, model_bits));
                    let mut reader = PacketBitReader::with_bit_length(&bytes, bit_length).unwrap();
                    let flex = decode_flex_block(&mut reader, &vlc, model_bits, trim).unwrap();
                    assert_eq!(reader.bits_remaining(), 0);
                    for position in 1..16 {
                        let decoded = (vlc[position] << model_bits) + flex[position];
                        // Trimming clears the low bits of the magnitude's
                        // refinement; nothing above them may change.
                        let kept = !((1_u32 << trim.min(model_bits)) - 1);
                        let expected = block[position].unsigned_abs() & kept;
                        assert_eq!(decoded.unsigned_abs(), expected);
                        assert!(decoded == 0 || (decoded < 0) == (block[position] < 0));
                    }
                }
            }
        }
    }
}
