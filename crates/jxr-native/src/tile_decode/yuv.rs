//! The syntax elements only YUV planes have: the joint DC presence pattern,
//! the adaptive low-pass coded-block pattern, and the low-pass block 4:2:0
//! and 4:2:2 chroma components share.

use jxr_core::ChromaSampling;

use crate::{
    bit_writer::BitWriter,
    entropy::{
        CodeTable, ComponentClass, DecodedBlock, FrequencyBand, PacketBitReader, PrefixTable,
        TileEntropyState, decode_ac_block, decode_lp_refinement_at, decode_prefix, encode_prefix,
    },
};

use super::TileDecodeError;

const DC_PATTERN_CODES: [(u16, u8, u8); 8] = [
    (0b10, 2, 0),
    (0b001, 3, 1),
    (0b00001, 5, 2),
    (0b0001, 4, 3),
    (0b11, 2, 4),
    (0b010, 3, 5),
    (0b00000, 5, 6),
    (0b011, 3, 7),
];
const LP_PATTERN_CODES: [(u16, u8, u8); 8] = [
    (0b0, 1, 0),
    (0b100, 3, 1),
    (0b1010, 4, 2),
    (0b1011, 4, 3),
    (0b1100, 4, 4),
    (0b1101, 4, 5),
    (0b1110, 4, 6),
    (0b1111, 4, 7),
];
const LP_PATTERN_SUBSAMPLED_CODES: [(u16, u8, u8); 4] =
    [(0b0, 1, 0), (0b10, 2, 1), (0b110, 3, 2), (0b111, 3, 3)];

static DC_PATTERN: PrefixTable = PrefixTable::new(&DC_PATTERN_CODES);
static LP_PATTERN: PrefixTable = PrefixTable::new(&LP_PATTERN_CODES);
static LP_PATTERN_SUBSAMPLED: PrefixTable = PrefixTable::new(&LP_PATTERN_SUBSAMPLED_CODES);
static DC_PATTERN_WRITE: CodeTable = CodeTable::new(&DC_PATTERN_CODES);
static LP_PATTERN_WRITE: CodeTable = CodeTable::new(&LP_PATTERN_CODES);
static LP_PATTERN_SUBSAMPLED_WRITE: CodeTable = CodeTable::new(&LP_PATTERN_SUBSAMPLED_CODES);

/// Reads `VAL_DC_YUV`: one bit per component, luma in the highest, telling
/// whether its DC coefficient carries a VLC-coded level.
pub(super) fn decode_dc_pattern(reader: &mut PacketBitReader<'_>) -> Result<u8, TileDecodeError> {
    Ok(decode_prefix(reader, "VAL_DC_YUV", &DC_PATTERN)?)
}

/// Writes `VAL_DC_YUV`, the inverse of [`decode_dc_pattern`].
pub(crate) fn encode_dc_pattern(
    writer: &mut BitWriter,
    pattern: u8,
) -> Result<(), TileDecodeError> {
    Ok(encode_prefix(
        writer,
        "VAL_DC_YUV",
        &DC_PATTERN_WRITE,
        pattern,
    )?)
}

pub(crate) struct CbplpState {
    sampling: ChromaSampling,
    zero: i8,
    maximum: i8,
}

impl CbplpState {
    pub(crate) const fn new(sampling: ChromaSampling) -> Self {
        Self {
            sampling,
            zero: 1,
            maximum: 1,
        }
    }

    /// Writes one low-pass coded-block pattern, the inverse of [`Self::decode`].
    pub(crate) fn encode(
        &mut self,
        writer: &mut BitWriter,
        pattern: u8,
    ) -> Result<(), TileDecodeError> {
        let full = self.sampling == ChromaSampling::Cs444;
        let maximum_pattern = if full { 7 } else { 3 };
        if pattern > maximum_pattern {
            return Err(TileDecodeError::InvalidPlan("CBPLP pattern"));
        }
        if self.zero <= 0 || self.maximum < 0 {
            // The code favours whichever extreme has been more frequent.
            let symbol = if self.maximum < self.zero {
                maximum_pattern - pattern
            } else {
                pattern
            };
            let table = if full {
                &LP_PATTERN_WRITE
            } else {
                &LP_PATTERN_SUBSAMPLED_WRITE
            };
            encode_prefix(writer, "CBPLP_YUV1", table, symbol)?;
        } else {
            writer.write_bits(u64::from(pattern), if full { 3 } else { 2 });
        }
        self.observe(pattern, maximum_pattern);
        Ok(())
    }

    fn observe(&mut self, pattern: u8, maximum_pattern: u8) {
        let zero_delta = if pattern == 0 { 4 } else { 0 };
        let maximum_delta = if pattern == maximum_pattern { 4 } else { 0 };
        self.zero = (self.zero + 1 - zero_delta).clamp(-8, 7);
        self.maximum = (self.maximum + 1 - maximum_delta).clamp(-8, 7);
    }

    /// Reads one low-pass coded-block pattern: bit zero for luma, then one
    /// bit for each chroma component, or one for both when they share a
    /// block.
    pub(super) fn decode(
        &mut self,
        reader: &mut PacketBitReader<'_>,
    ) -> Result<u8, TileDecodeError> {
        let maximum_pattern = if self.sampling == ChromaSampling::Cs444 {
            7
        } else {
            3
        };
        let pattern = if self.zero <= 0 || self.maximum < 0 {
            let mut pattern = if self.sampling == ChromaSampling::Cs444 {
                decode_prefix(reader, "CBPLP_YUV1", &LP_PATTERN)?
            } else {
                decode_prefix(reader, "CBPLP_YUV1", &LP_PATTERN_SUBSAMPLED)?
            };
            if self.maximum < self.zero {
                pattern = maximum_pattern - pattern;
            }
            pattern
        } else {
            let bits = if self.sampling == ChromaSampling::Cs444 {
                3
            } else {
                2
            };
            u8::try_from(reader.read_bits(bits)?)
                .map_err(|_| TileDecodeError::ArithmeticOverflow("CBPLP_YUV2"))?
        };
        self.observe(pattern, maximum_pattern);
        Ok(pattern)
    }
}

/// Scan location the jointly coded chroma low-pass block of a subsampled
/// macroblock starts at: it holds both components' three coefficients for
/// 4:2:0 and seven for 4:2:2, so it ends at the last location of a block.
pub(crate) const fn joint_chroma_start(sampling: ChromaSampling) -> u8 {
    match sampling {
        ChromaSampling::Cs420 => 10,
        _ => 2,
    }
}

/// Coefficients the jointly coded chroma low-pass block holds.
pub(crate) const fn joint_chroma_count(sampling: ChromaSampling) -> usize {
    16 - joint_chroma_start(sampling) as usize
}

/// Low-pass positions of a subsampled chroma component in the order their
/// refinements are coded, each for U and then for V.
pub(crate) const fn chroma_refinement_positions(sampling: ChromaSampling) -> &'static [usize] {
    match sampling {
        ChromaSampling::Cs420 => &[2, 1, 3],
        _ => &[2, 1, 3, 4, 6, 5, 7],
    }
}

/// The chroma component, one or two, and the low-pass position within it
/// that entry `index` of the jointly coded chroma block holds. U and V
/// alternate, position by position.
pub(crate) const fn joint_chroma_slot(index: usize, sampling: ChromaSampling) -> (usize, usize) {
    const REMAP: [usize; 7] = [4, 1, 2, 3, 5, 6, 7];
    const TRANSPOSE_420: [usize; 4] = [0, 2, 1, 3];
    const TRANSPOSE_422: [usize; 8] = [0, 2, 1, 3, 4, 6, 5, 7];
    let position = match sampling {
        ChromaSampling::Cs420 => TRANSPOSE_420[REMAP[(index >> 1) + 1]],
        _ => TRANSPOSE_422[REMAP[index >> 1]],
    };
    ((index & 1) + 1, position)
}

/// The jointly coded chroma low-pass block of a subsampled macroblock in
/// coded order, from the two components' coefficients: the inverse of
/// [`remap_joint_chroma`]. Entries beyond [`joint_chroma_count`] are zero.
pub(crate) fn joint_chroma_block(chroma: [&[i32; 16]; 2], sampling: ChromaSampling) -> [i32; 14] {
    let mut block = [0_i32; 14];
    for (index, value) in block
        .iter_mut()
        .enumerate()
        .take(joint_chroma_count(sampling))
    {
        let (component, position) = joint_chroma_slot(index, sampling);
        *value = chroma[component - 1][position];
    }
    block
}

/// Reads the jointly coded chroma low-pass block of a subsampled macroblock
/// when `coded`, then the refinements of both components, U before V at
/// each position, into `chroma`: U and then V, holding their DC values.
/// Returns the number of VLC-coded coefficients.
pub(super) fn decode_joint_chroma(
    reader: &mut PacketBitReader<'_>,
    entropy: &mut TileEntropyState,
    coded: bool,
    chroma: &mut [[i32; 16]; 2],
    sampling: ChromaSampling,
) -> Result<i32, TileDecodeError> {
    let mut non_zero = 0;
    if coded {
        let block = decode_ac_block(
            reader,
            FrequencyBand::Lowpass,
            ComponentClass::Chroma,
            joint_chroma_start(sampling),
            &mut entropy.lp_vlc,
        )?;
        non_zero = i32::from(block.non_zero_count());
        remap_joint_chroma(&block, chroma, sampling)?;
    }
    decode_subsampled_chroma_refinement(
        reader,
        chroma,
        chroma_refinement_positions(sampling),
        entropy.lp_model.bits(true),
    )?;
    Ok(non_zero)
}

fn decode_subsampled_chroma_refinement(
    reader: &mut PacketBitReader<'_>,
    chroma: &mut [[i32; 16]; 2],
    positions: &[usize],
    model_bits: u8,
) -> Result<(), TileDecodeError> {
    for position in positions {
        for coefficients in chroma.iter_mut() {
            decode_lp_refinement_at(
                reader,
                coefficients,
                core::slice::from_ref(position),
                model_bits,
            )?;
        }
    }
    Ok(())
}

/// Places the entries of the jointly coded chroma block in U and V: the
/// inverse of [`joint_chroma_block`].
fn remap_joint_chroma(
    block: &DecodedBlock,
    chroma: &mut [[i32; 16]; 2],
    sampling: ChromaSampling,
) -> Result<(), TileDecodeError> {
    let count = joint_chroma_count(sampling);
    let mut temporary = [0_i32; 14];
    let mut cursor = 0_usize;
    for entry in block.entries() {
        cursor = cursor
            .checked_add(usize::from(entry.run))
            .filter(|&index| index < count)
            .ok_or(TileDecodeError::InvalidPlan("subsampled chroma LP run"))?;
        temporary[cursor] = entry.level;
        cursor += 1;
    }
    for (index, &value) in temporary[..count].iter().enumerate() {
        let (component, position) = joint_chroma_slot(index, sampling);
        chroma[component - 1][position] = value;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::encode::test_support::Lcg;

    #[test]
    fn yuv_pattern_prefix_tables_match_serial_search() {
        use crate::entropy::assert_matches_serial;
        assert_matches_serial("VAL_DC_YUV", &DC_PATTERN_CODES);
        assert_matches_serial("CBPLP_YUV1", &LP_PATTERN_CODES);
        assert_matches_serial("CBPLP_YUV1", &LP_PATTERN_SUBSAMPLED_CODES);
    }

    #[test]
    fn cbplp_patterns_round_trip_through_both_code_forms() {
        for sampling in [ChromaSampling::Cs444, ChromaSampling::Cs420] {
            let maximum = if sampling == ChromaSampling::Cs444 {
                8
            } else {
                4
            };
            // Runs of zero, of the full pattern, and of mixed values move the
            // state through fixed-length, direct, and inverted code selection.
            let mut state = 99_u64;
            let patterns: Vec<u8> = (0..600)
                .map(|index| {
                    state = Lcg::advance(state);
                    match index / 100 % 3 {
                        0 => 0,
                        1 => maximum - 1,
                        _ => u8::try_from((state >> 40) % u64::from(maximum)).unwrap(),
                    }
                })
                .collect();
            let mut writer = BitWriter::new();
            let mut write_state = CbplpState::new(sampling);
            for &pattern in &patterns {
                write_state.encode(&mut writer, pattern).unwrap();
            }
            let bytes = writer.into_bytes();
            let mut reader = PacketBitReader::new(&bytes);
            let mut read_state = CbplpState::new(sampling);
            for &pattern in &patterns {
                assert_eq!(read_state.decode(&mut reader).unwrap(), pattern);
            }
            assert!(reader.bits_remaining() < 8);
            assert_eq!(
                (read_state.zero, read_state.maximum),
                (write_state.zero, write_state.maximum)
            );
        }
    }

    #[test]
    fn fixed_length_cbplp_does_not_apply_vlc_inversion() {
        let mut reader = PacketBitReader::with_bit_length(&[0b1110_0000], 3).unwrap();
        let mut state = CbplpState {
            sampling: ChromaSampling::Cs444,
            zero: 4,
            maximum: 0,
        };

        assert_eq!(state.decode(&mut reader).unwrap(), 7);
        assert_eq!((state.zero, state.maximum), (5, -3));
        assert_eq!(reader.bits_remaining(), 0);
    }

    #[test]
    fn joint_chroma_blocks_round_trip_through_the_block_reader() {
        use crate::entropy::{AcVlcState, encode::encode_ordered_block};

        for sampling in [ChromaSampling::Cs420, ChromaSampling::Cs422] {
            let count = joint_chroma_count(sampling);
            let positions = chroma_refinement_positions(sampling);
            assert_eq!(count, 2 * positions.len());
            // The block holds every refined position of U and of V once.
            let mut slots: Vec<(usize, usize)> = (0..count)
                .map(|index| joint_chroma_slot(index, sampling))
                .collect();
            slots.sort_unstable();
            let mut expected: Vec<(usize, usize)> = [1, 2]
                .into_iter()
                .flat_map(|component| positions.iter().map(move |&position| (component, position)))
                .collect();
            expected.sort_unstable();
            assert_eq!(slots, expected, "{sampling:?}");

            let mut state = 31_u64;
            let mut write_vlc = AcVlcState::new();
            let mut read_vlc = AcVlcState::new();
            for round in 0..300_usize {
                // Sparse and dense blocks, and now and then a large value.
                let density = u64::try_from(1 + round % 4).unwrap();
                let mut low = [[0_i32; 16]; 2];
                for component in &mut low {
                    for &position in positions {
                        state = Lcg::advance(state);
                        let value = i32::try_from((state >> 40) % 41).unwrap() - 20;
                        if (state >> 20) % 4 < density {
                            component[position] = if round % 7 == 0 { value * 900 } else { value };
                        }
                    }
                }
                low[0][positions[round % positions.len()]] = 3;
                let block = joint_chroma_block([&low[0], &low[1]], sampling);
                assert!(block[count..].iter().all(|&value| value == 0));
                let mut writer = BitWriter::new();
                let written = encode_ordered_block(
                    &mut writer,
                    ComponentClass::Chroma,
                    joint_chroma_start(sampling),
                    &mut write_vlc,
                    &block[..count],
                )
                .unwrap();
                let bytes = writer.into_bytes();
                let decoded = decode_ac_block(
                    &mut PacketBitReader::new(&bytes),
                    FrequencyBand::Lowpass,
                    ComponentClass::Chroma,
                    joint_chroma_start(sampling),
                    &mut read_vlc,
                )
                .unwrap();
                assert_eq!(decoded.non_zero_count(), written, "{sampling:?}");
                let mut remapped = [[0_i32; 16]; 2];
                remap_joint_chroma(&decoded, &mut remapped, sampling).unwrap();
                assert_eq!(remapped, low, "{sampling:?} round {round}");
                if round % 5 == 4 {
                    write_vlc.adapt();
                    read_vlc.adapt();
                }
            }
            assert_eq!(read_vlc, write_vlc, "{sampling:?}");
        }
    }

    #[test]
    fn subsampled_chroma_refinement_interleaves_u_and_v_per_position() {
        let mut reader = PacketBitReader::with_bit_length(&[0b0011_0000], 4).unwrap();
        let mut chroma = [[0_i32; 16]; 2];
        for component in &mut chroma {
            component[1] = 1;
            component[2] = 1;
        }

        decode_subsampled_chroma_refinement(&mut reader, &mut chroma, &[2, 1], 1).unwrap();

        assert_eq!((chroma[0][2], chroma[1][2]), (2, 2));
        assert_eq!((chroma[0][1], chroma[1][1]), (3, 3));
        assert_eq!(reader.bits_remaining(), 0);
    }
}
