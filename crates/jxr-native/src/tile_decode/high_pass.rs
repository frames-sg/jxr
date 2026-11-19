//! The block order and block completion of the high-pass band.

use crate::entropy::{PacketBitReader, refine_flex_block};

use super::{ComponentGeometry, TileDecodeError};

pub(crate) const HIERARCHICAL_BLOCK_ORDER: [usize; 16] =
    [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];

/// Raster index of each block of a component macroblock, in the order the
/// high-pass syntax codes them and counts them in a coded-block pattern.
///
/// The sixteen blocks of a full-resolution macroblock are coded quadrant by
/// quadrant; the eight or four of a subsampled chroma one in raster order.
pub(crate) fn coded_block_order(geometry: ComponentGeometry) -> &'static [usize] {
    const RASTER: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
    match geometry.block_count() {
        count @ ..=8 => &RASTER[..count],
        _ => &HIERARCHICAL_BLOCK_ORDER,
    }
}

/// What follows each high-pass block in the packet that codes it.
#[derive(Debug, Clone, Copy)]
pub(super) enum HighpassPayload {
    /// Nothing: the block's flexbits, if any, are in a packet of their own,
    /// as in a frequency-mode tile.
    VlcOnly,
    /// The block's flexbits when the band has them, as in a spatial-mode
    /// tile, with the tile's flexbits trim.
    Combined {
        flexbits_present: bool,
        trim_flexbits: u8,
    },
}

/// Completes one high-pass block whose fifteen high-pass positions hold the
/// VLC-coded parts its syntax decoded, with its flexbits: each part is
/// shifted past `model_bits` refinement bits, and the refinement `flexbits`
/// reads, with `trim_flexbits` of its bits trimmed, added. Position zero
/// belongs to the low-pass band and is left alone.
///
/// Each position is completed as its flexbits are read. An overflow is
/// reported once every position's flexbits are read, so that a packet that
/// ends early reports its end, as reading them all before adding any does.
pub(super) fn refine_block(
    flexbits: &mut PacketBitReader<'_>,
    block: &mut [i32; 16],
    model_bits: u8,
    trim_flexbits: u8,
) -> Result<(), TileDecodeError> {
    // Below 16 once `refine_flex_block` has accepted the model bits.
    let shift = u32::from(model_bits);
    let mut overflowed = false;
    refine_flex_block(
        flexbits,
        block,
        model_bits,
        trim_flexbits,
        |coefficient, refinement| {
            let (value, overflow) = (*coefficient << shift).overflowing_add(refinement);
            *coefficient = value;
            overflowed |= overflow;
        },
    )?;
    if overflowed {
        return Err(REFINEMENT_OVERFLOW);
    }
    Ok(())
}

/// [`refine_block`] for the blocks of one component macroblock,
/// `coefficients`, whose raster indices `order` lists in coded order: the
/// order their flexbits follow one another in.
///
/// Its own function, with references for parameters, so that the compiler
/// knows the reader and the coefficients apart throughout the loop.
pub(super) fn refine_blocks(
    flexbits: &mut PacketBitReader<'_>,
    coefficients: &mut [i32; 256],
    order: &[usize],
    model_bits: u8,
    trim_flexbits: u8,
) -> Result<(), TileDecodeError> {
    let blocks = coefficients.as_chunks_mut::<16>().0;
    for &block in order {
        refine_block(flexbits, &mut blocks[block], model_bits, trim_flexbits)?;
    }
    Ok(())
}

/// [`refine_block`] for a block without flexbits to read: each part is only
/// shifted past `model_bits` refinement bits, which adds nothing that could
/// overflow.
pub(super) fn scale_block(block: &mut [i32; 16], model_bits: u8) -> Result<(), TileDecodeError> {
    let shift = u32::from(model_bits);
    if shift >= i32::BITS {
        return Err(REFINEMENT_OVERFLOW);
    }
    if shift != 0 {
        for coefficient in &mut block[1..] {
            *coefficient <<= shift;
        }
    }
    Ok(())
}

/// What a high-pass position that leaves `i32` once refined fails with.
const REFINEMENT_OVERFLOW: TileDecodeError = TileDecodeError::ArithmeticOverflow("HP refinement");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::{decode_flex_block, encode::test_support::Lcg};

    /// What [`refine_block`] fuses: the block's flexbits read whole, then
    /// each VLC-coded part shifted and its refinement added.
    fn read_then_add(
        reader: &mut PacketBitReader<'_>,
        block: &[i32; 16],
        model_bits: u8,
        trim_flexbits: u8,
    ) -> Result<[i32; 16], TileDecodeError> {
        let flex = decode_flex_block(reader, block, model_bits, trim_flexbits)?;
        let mut refined = *block;
        let mut overflowed = false;
        for position in 1..16 {
            let (value, overflow) = (block[position] << model_bits).overflowing_add(flex[position]);
            refined[position] = value;
            overflowed |= overflow;
        }
        if overflowed {
            return Err(REFINEMENT_OVERFLOW);
        }
        Ok(refined)
    }

    /// Refining a block as its flexbits are read gives the coefficients,
    /// the bits read and the error of reading them all and then adding
    /// them: over packets that end part way through a block, where the
    /// reader's cache runs short and the flexbits are read bit by bit, and
    /// over blocks whose refinement leaves `i32`, where the packet's end,
    /// read first, is the error.
    #[test]
    fn refining_a_block_matches_reading_its_flexbits_then_adding_them() {
        let mut rng = Lcg(11);
        let (mut ended, mut overflowed, mut refined) = (0, 0, 0);
        for round in 0..40_000_u32 {
            let bytes: [u8; 40] = core::array::from_fn(|_| rng.next().to_le_bytes()[0]);
            let model_bits = u8::try_from(round % 16).unwrap();
            let trim_flexbits = u8::try_from(rng.below(16)).unwrap();
            let bit_length = usize::try_from(rng.below(321)).unwrap();
            // Some parts the most negative that shifting keeps in `i32`,
            // which a refinement of their sign takes out of it.
            let block: [i32; 16] = core::array::from_fn(|_| {
                if rng.below(8) == 0 {
                    (i32::MIN >> model_bits) + i32::try_from(rng.below(2)).unwrap()
                } else {
                    rng.coefficient(6)
                }
            });
            let mut reader = PacketBitReader::with_bit_length(&bytes, bit_length).unwrap();
            let mut expected_reader = reader.clone();
            let expected = read_then_add(&mut expected_reader, &block, model_bits, trim_flexbits);
            let mut actual = block;
            let result = refine_block(&mut reader, &mut actual, model_bits, trim_flexbits);
            match expected {
                Ok(expected) => {
                    assert_eq!(result, Ok(()), "round {round}");
                    assert_eq!(actual, expected, "round {round}");
                    assert_eq!(reader.bit_position(), expected_reader.bit_position());
                    refined += 1;
                }
                Err(error) => {
                    if error == REFINEMENT_OVERFLOW {
                        overflowed += 1;
                    } else {
                        ended += 1;
                    }
                    assert_eq!(result, Err(error), "round {round}");
                }
            }
        }
        assert!(
            refined > 1000 && overflowed > 1000 && ended > 1000,
            "{refined} {overflowed} {ended}"
        );
    }
}
