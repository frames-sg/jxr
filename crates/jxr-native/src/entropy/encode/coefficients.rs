//! DC, LP, and HP coefficient writing primitives.

use crate::bit_writer::BitWriter;

use super::super::{
    AcVlcState, AdaptiveHpScan, AdaptiveLpScan, ComponentClass, DcVlcState, EntropyError,
    HpScanDirection,
    adaptive::{AdaptiveVlc, observe_abs_level, observe_first_index, observe_index},
    coefficients::{ABS_FIXED_LENGTH, ABS_REMAP, RUN_BIN, RUN_FIXED_LENGTH, RUN_REMAP},
    scan::ScannedBlock,
    vlc,
};
use super::refinement::check_magnitudes;

/// Largest `ABS_LEVEL` escape width a decoder can represent (4 + 15 + 3 + 7).
const MAX_ESCAPE_BITS: u32 = 29;

/// Whether a DC coefficient carries a VLC-coded level above its model bits.
#[must_use]
pub(crate) const fn dc_has_abs_level(value: i32, model_bits: u8) -> bool {
    value.unsigned_abs() >> model_bits != 0
}

/// Writes one DC coefficient as specified by T.832 Table 49.
///
/// The caller writes the flag reported by [`dc_has_abs_level`] beforehand, in
/// whatever form the colour format's syntax gives it.
pub(crate) fn encode_dc_coefficient(
    writer: &mut BitWriter,
    value: i32,
    model_bits: u8,
    class: ComponentClass,
    state: &mut DcVlcState,
) -> Result<(), EntropyError> {
    if model_bits > 15 {
        return Err(EntropyError::InvalidParameter {
            parameter: "DC model bits",
            value: i64::from(model_bits),
        });
    }
    check_magnitudes(&[value])?;
    let magnitude = value.unsigned_abs();
    let level = magnitude >> model_bits;
    if level != 0 {
        let abs_level = level
            .checked_add(1)
            .ok_or(EntropyError::CoefficientOverflow)?;
        encode_abs_level(writer, abs_level, state.abs_level(class.is_chroma()))?;
    }
    if model_bits != 0 {
        let refinement = magnitude & ((1_u32 << model_bits) - 1);
        writer.write_bits(u64::from(refinement), model_bits);
    }
    if value != 0 {
        writer.write_flag(value < 0);
    }
    Ok(())
}

/// Scans and writes the VLC parts of one LP block.
///
/// `levels` holds the block's VLC parts in raster order and must contain a
/// non-zero value at or after `start_location` in the scan; the caller codes
/// an all-zero block through the syntax's coded flag instead. Returns the
/// number of coefficients written.
pub(crate) fn encode_lp_block(
    writer: &mut BitWriter,
    class: ComponentClass,
    start_location: u8,
    state: &mut AcVlcState,
    scan: &mut AdaptiveLpScan,
    levels: &[i32; 16],
) -> Result<u8, EntropyError> {
    let block = scan.scan(levels, start_location)?;
    encode_ac_block(writer, class, start_location, state, &block)
}

/// Scans and writes the VLC parts of one HP block; see [`encode_lp_block`].
pub(crate) fn encode_hp_block(
    writer: &mut BitWriter,
    class: ComponentClass,
    state: &mut AcVlcState,
    scan: &mut AdaptiveHpScan,
    direction: HpScanDirection,
    levels: &[i32; 16],
) -> Result<u8, EntropyError> {
    let block = scan.scan(direction, levels)?;
    encode_ac_block(writer, class, 1, state, &block)
}

/// Writes the VLC parts of a block that no scan reorders: `levels[n]` lies at
/// scan location `start_location + n`, as [`super::super::decode_ac_block`]
/// reports its runs. `levels` must contain a non-zero value, as for
/// [`encode_lp_block`]. Returns the number of coefficients written.
pub(crate) fn encode_ordered_block(
    writer: &mut BitWriter,
    class: ComponentClass,
    start_location: u8,
    state: &mut AcVlcState,
    levels: &[i32],
) -> Result<u8, EntropyError> {
    let block = ScannedBlock::from_ordered(start_location, levels)?;
    encode_ac_block(writer, class, start_location, state, &block)
}

/// What follows the coefficient at scan index `index`, given the occupied
/// indices after it: nothing (0), a coefficient at the very next scan
/// position (1), or one after a zero run (2).
#[inline]
const fn successor(remaining: u16, index: u32) -> u8 {
    if remaining == 0 {
        0
    } else if remaining >> (index + 1) & 1 != 0 {
        1
    } else {
        2
    }
}

/// Writes the run/level representation of one scanned LP or HP block.
///
/// Each index code goes out with the sign bit that follows it, in one
/// write.
fn encode_ac_block(
    writer: &mut BitWriter,
    class: ComponentClass,
    start_location: u8,
    state: &mut AcVlcState,
    block: &ScannedBlock,
) -> Result<u8, EntropyError> {
    let start = u32::from(start_location);
    let mut remaining = block.occupied();
    let first = remaining.trailing_zeros();
    // An empty block is coded through the syntax's coded flag instead, and
    // a coefficient before the start belongs to another band.
    if remaining == 0 || first < start || start == 0 {
        return Err(EntropyError::InvalidParameter {
            parameter: "non-zero coefficients in block",
            value: 0,
        });
    }
    let count = remaining.count_ones();
    remaining &= remaining - 1;
    let chroma = class.is_chroma();

    let level = block.value(first);
    let large = level.unsigned_abs() > 1;
    let run = first - start;
    let following = successor(remaining, first);
    let symbol = u8::from(run == 0) | (u8::from(large) << 1) | (following << 2);
    let table = state.first_index(chroma);
    let (code, length) = vlc::code(
        "FIRST_INDEX",
        &vlc::write::FIRST_INDEX[table.table_index()],
        symbol,
    )?;
    observe_first_index(table, symbol);
    writer.write_bits(u64::from(code) << 1 | u64::from(level < 0), length + 1);
    let mut context = run == 0 && following & 1 != 0;
    if large {
        encode_abs_level(writer, level.unsigned_abs(), state.abs_level(context))?;
    }
    if run != 0 {
        encode_run(writer, run, 15 - start)?;
    }
    let mut location = first + 1;

    while remaining != 0 {
        let index = remaining.trailing_zeros();
        remaining &= remaining - 1;
        let run = index - location;
        if run != 0 {
            encode_run(writer, run, 15 - location)?;
        }
        location = index + 1;
        let level = block.value(index);
        let large = level.unsigned_abs() > 1;
        let following = successor(remaining, index);
        let symbol = u8::from(large) | (following << 1);
        let (code, length) = match location.cmp(&15) {
            core::cmp::Ordering::Less => {
                let table = state.index(chroma, context);
                let code = vlc::code("INDEX_A", &vlc::write::INDEX_A[table.table_index()], symbol)?;
                observe_index(table, symbol);
                code
            }
            core::cmp::Ordering::Equal => vlc::code("INDEX_B", &vlc::write::INDEX_B, symbol)?,
            // Nothing can follow the last position, so one bit tells
            // whether its level is above one.
            core::cmp::Ordering::Greater => (u16::from(symbol != 0), 1),
        };
        writer.write_bits(u64::from(code) << 1 | u64::from(level < 0), length + 1);
        context = context && following & 1 != 0;
        if large {
            encode_abs_level(writer, level.unsigned_abs(), state.abs_level(context))?;
        }
    }
    u8::try_from(count).map_err(|_| EntropyError::CoefficientOverflow)
}

/// Levels the six direct `ABS_LEVEL_INDEX` values cover without gaps: two
/// up to, but not including, this one.
const DIRECT_LEVELS: usize = 18;

/// The direct `ABS_LEVEL_INDEX` of each level below [`DIRECT_LEVELS`]: the
/// last index whose base in [`ABS_REMAP`] is at most the level.
const DIRECT_LEVEL_INDEX: [u8; DIRECT_LEVELS] = {
    let mut table = [0_u8; DIRECT_LEVELS];
    let mut index = 0_u8;
    let mut slot = 2;
    let mut level = 2_i32;
    while slot < DIRECT_LEVELS {
        let next = index as usize + 1;
        if next < ABS_REMAP.len() && ABS_REMAP[next] <= level {
            index += 1;
        }
        table[slot] = index;
        slot += 1;
        level += 1;
    }
    table
};

/// Writes an `ABS_LEVEL_INDEX` code and its refinement for a level of at least two.
#[inline]
fn encode_abs_level(
    writer: &mut BitWriter,
    level: u32,
    state: &mut AdaptiveVlc,
) -> Result<(), EntropyError> {
    if level < 2 {
        return Err(EntropyError::InvalidParameter {
            parameter: "absolute level",
            value: i64::from(level),
        });
    }
    let table = &vlc::write::ABS_LEVEL[state.table_index()];
    if let Some(&index) = DIRECT_LEVEL_INDEX.get(level as usize) {
        let (code, length) = vlc::code("ABS_LEVEL_INDEX", table, index)?;
        observe_abs_level(state, index);
        let base = ABS_REMAP[usize::from(index)].unsigned_abs();
        let fixed = ABS_FIXED_LENGTH[usize::from(index)];
        // The code and the level's offset from the index's base, together.
        writer.write_bits(
            u64::from(code) << fixed | u64::from(level - base),
            length + fixed,
        );
        return Ok(());
    }
    encode_escaped_level(writer, level, table, state)
}

/// Writes the escape `ABS_LEVEL_INDEX` and the level it introduces: one no
/// direct index covers.
#[cold]
fn encode_escaped_level(
    writer: &mut BitWriter,
    level: u32,
    table: &vlc::CodeTable,
    state: &mut AdaptiveVlc,
) -> Result<(), EntropyError> {
    let excess = level - 2;
    let fixed = excess.ilog2();
    if fixed > MAX_ESCAPE_BITS {
        return Err(EntropyError::CoefficientOverflow);
    }
    vlc::encode(writer, "ABS_LEVEL_INDEX", table, 6)?;
    observe_abs_level(state, 6);
    if fixed < 19 {
        writer.write_bits(u64::from(fixed - 4), 4);
    } else {
        writer.write_bits(15, 4);
        if fixed < 22 {
            writer.write_bits(u64::from(fixed - 19), 2);
        } else {
            writer.write_bits(3, 2);
            writer.write_bits(u64::from(fixed - 22), 3);
        }
    }
    let width = u8::try_from(fixed).map_err(|_| EntropyError::CoefficientOverflow)?;
    writer.write_bits(u64::from(excess - (1 << fixed)), width);
    Ok(())
}

/// Longest zero run a block can hold: fourteen positions between its first
/// and last coefficient.
const MAX_RUN: usize = 14;

/// `(bits, length)` of every run under every longest run a position allows,
/// indexed `[longest run][run]`: the run's code and its refinement as one
/// element. The decoder's run lookup is built from the same code lists.
static RUN_CODES: [[(u16, u8); MAX_RUN + 1]; MAX_RUN + 1] = run_codes();

const fn run_codes() -> [[(u16, u8); MAX_RUN + 1]; MAX_RUN + 1] {
    let mut codes = [[(0_u16, 0_u8); MAX_RUN + 1]; MAX_RUN + 1];
    // A longest run of one leaves a single choice, which codes nothing.
    let mut max_run = 2;
    while max_run <= MAX_RUN {
        let mut run = 1;
        while run <= max_run {
            codes[max_run][run] = if max_run <= 4 {
                vlc::write::RUN_VALUE[max_run - 2].code(run)
            } else {
                assert!(RUN_BIN[max_run] >= 0, "run bin of a coded run");
                let bin = RUN_BIN[max_run].unsigned_abs() as usize;
                // The last run index whose base is at most the run.
                let mut run_index = 4;
                while RUN_REMAP[run_index + 5 * bin] as usize > run {
                    run_index -= 1;
                }
                let base = RUN_REMAP[run_index + 5 * bin] as usize;
                let fixed = RUN_FIXED_LENGTH[run_index + 5 * bin];
                let (bits, length) = vlc::write::RUN_INDEX.code(run_index);
                let refinement = run - base;
                assert!(refinement >> fixed == 0, "run refinement width");
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "const evaluation; the refinement is below two to the power `fixed`"
                )]
                ((bits << fixed) | refinement as u16, length + fixed)
            };
            assert!(codes[max_run][run].1 != 0, "every run has a code");
            run += 1;
        }
        max_run += 1;
    }
    codes
}

/// Writes a non-zero run given the longest run the position allows.
#[inline]
fn encode_run(writer: &mut BitWriter, run: u32, max_run: u32) -> Result<(), EntropyError> {
    if max_run == 0 || max_run as usize > MAX_RUN {
        return Err(EntropyError::InvalidParameter {
            parameter: "maximum coefficient run",
            value: i64::from(max_run),
        });
    }
    if run == 0 || run > max_run {
        return Err(EntropyError::InvalidParameter {
            parameter: "coefficient run",
            value: i64::from(run),
        });
    }
    let (bits, length) = RUN_CODES[max_run as usize][run as usize];
    writer.write_bits(u64::from(bits), length);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::super::{
        FrequencyBand, PacketBitReader, decode_ac_block, decode_dc_coefficient, decode_hp_block,
        decode_lp_block,
    };
    use super::super::test_support::Lcg;
    use super::*;

    #[test]
    fn dc_coefficients_round_trip_with_matching_state() {
        let mut rng = Lcg(1);
        for _ in 0..2_000 {
            let model_bits = u8::try_from(rng.below(16)).unwrap();
            let class = if rng.below(2) == 0 {
                ComponentClass::Luma
            } else {
                ComponentClass::Chroma
            };
            let mut write_state = DcVlcState::new();
            let mut read_state = DcVlcState::new();
            let values: Vec<i32> = (0..40).map(|_| rng.coefficient(30)).collect();
            let mut writer = BitWriter::new();
            for (index, &value) in values.iter().enumerate() {
                writer.write_flag(dc_has_abs_level(value, model_bits));
                encode_dc_coefficient(&mut writer, value, model_bits, class, &mut write_state)
                    .unwrap();
                if index % 7 == 6 {
                    write_state.adapt();
                }
            }
            let bit_length = writer.bit_position();
            let bytes = writer.into_bytes();
            let mut reader = PacketBitReader::with_bit_length(&bytes, bit_length).unwrap();
            for (index, &value) in values.iter().enumerate() {
                let has_abs_level = reader.read_bit().unwrap();
                let decoded = decode_dc_coefficient(
                    &mut reader,
                    model_bits,
                    has_abs_level,
                    class,
                    &mut read_state,
                )
                .unwrap();
                assert_eq!(decoded, value);
                if index % 7 == 6 {
                    read_state.adapt();
                }
            }
            assert_eq!(reader.bits_remaining(), 0);
            assert_eq!(read_state, write_state);
        }
    }

    #[test]
    fn dc_level_beyond_the_escape_range_is_rejected() {
        let mut writer = BitWriter::new();
        let result = encode_dc_coefficient(
            &mut writer,
            i32::MAX,
            0,
            ComponentClass::Luma,
            &mut DcVlcState::new(),
        );
        assert_eq!(result, Err(EntropyError::CoefficientOverflow));
    }

    /// A block whose density is itself random so every `FIRST_INDEX`,
    /// `INDEX_A`/`INDEX_B`/one-bit index and run-table path occurs.
    fn random_block(rng: &mut Lcg, start: usize) -> [i32; 16] {
        let mut block = [0_i32; 16];
        let density = rng.below(17);
        for value in &mut block[start..] {
            if rng.below(16) < density {
                let max_bits = if rng.below(8) == 0 { 30 } else { 3 };
                *value = rng.coefficient(max_bits);
            }
        }
        block
    }

    #[test]
    fn lp_blocks_round_trip_with_matching_scan_and_vlc_state() {
        let mut rng = Lcg(2);
        for round in 0..400 {
            let class = if round % 2 == 0 {
                ComponentClass::Luma
            } else {
                ComponentClass::Chroma
            };
            let start = 1 + u8::try_from(rng.below(3)).unwrap();
            let mut write_vlc = AcVlcState::new();
            let mut write_scan = AdaptiveLpScan::new();
            let blocks: Vec<[i32; 16]> = (0..48)
                .map(|_| random_block(&mut rng, usize::from(start)))
                .collect();
            let mut writer = BitWriter::new();
            let mut counts = Vec::new();
            for (index, block) in blocks.iter().enumerate() {
                let coded = write_scan.order()[usize::from(start)..]
                    .iter()
                    .any(|&position| block[usize::from(position)] != 0);
                writer.write_flag(coded);
                counts.push(if coded {
                    encode_lp_block(
                        &mut writer,
                        class,
                        start,
                        &mut write_vlc,
                        &mut write_scan,
                        block,
                    )
                    .unwrap()
                } else {
                    0
                });
                if index % 5 == 4 {
                    write_vlc.adapt();
                }
                if index % 16 == 15 {
                    write_scan.reset_totals();
                }
            }
            let bit_length = writer.bit_position();
            let bytes = writer.into_bytes();
            let mut reader = PacketBitReader::with_bit_length(&bytes, bit_length).unwrap();
            let mut read_vlc = AcVlcState::new();
            let mut read_scan = AdaptiveLpScan::new();
            for (index, block) in blocks.iter().enumerate() {
                let mut decoded = [0_i32; 16];
                // Positions before the start location are not part of the block.
                for &position in &read_scan.order()[..usize::from(start)] {
                    decoded[usize::from(position)] = block[usize::from(position)];
                }
                let count = if reader.read_bit().unwrap() {
                    decode_lp_block(
                        &mut reader,
                        class,
                        start,
                        &mut read_vlc,
                        &mut read_scan,
                        &mut decoded,
                    )
                    .unwrap()
                } else {
                    0
                };
                assert_eq!(&decoded, block);
                assert_eq!(count, counts[index]);
                if index % 5 == 4 {
                    read_vlc.adapt();
                }
                if index % 16 == 15 {
                    read_scan.reset_totals();
                }
            }
            assert_eq!(reader.bits_remaining(), 0);
            assert_eq!(read_vlc, write_vlc);
            assert_eq!(read_scan, write_scan);
        }
    }

    #[test]
    fn hp_blocks_round_trip_in_both_scan_directions() {
        let mut rng = Lcg(3);
        for _ in 0..400 {
            let mut write_vlc = AcVlcState::new();
            let mut write_scan = AdaptiveHpScan::new();
            let blocks: Vec<([i32; 16], HpScanDirection)> = (0..48)
                .map(|_| {
                    let direction = if rng.below(2) == 0 {
                        HpScanDirection::Horizontal
                    } else {
                        HpScanDirection::Vertical
                    };
                    (random_block(&mut rng, 1), direction)
                })
                .collect();
            let mut writer = BitWriter::new();
            for (index, (block, direction)) in blocks.iter().enumerate() {
                let coded = block[1..].iter().any(|&value| value != 0);
                writer.write_flag(coded);
                if coded {
                    encode_hp_block(
                        &mut writer,
                        ComponentClass::Luma,
                        &mut write_vlc,
                        &mut write_scan,
                        *direction,
                        block,
                    )
                    .unwrap();
                }
                if index % 5 == 4 {
                    write_vlc.adapt();
                }
            }
            let bit_length = writer.bit_position();
            let bytes = writer.into_bytes();
            let mut reader = PacketBitReader::with_bit_length(&bytes, bit_length).unwrap();
            let mut read_vlc = AcVlcState::new();
            let mut read_scan = AdaptiveHpScan::new();
            for (index, (block, direction)) in blocks.iter().enumerate() {
                let mut decoded = [0_i32; 16];
                decoded[0] = block[0];
                if reader.read_bit().unwrap() {
                    decode_hp_block(
                        &mut reader,
                        ComponentClass::Luma,
                        &mut read_vlc,
                        &mut read_scan,
                        *direction,
                        &mut decoded,
                    )
                    .unwrap();
                }
                assert_eq!(&decoded, block);
                if index % 5 == 4 {
                    read_vlc.adapt();
                }
            }
            assert_eq!(reader.bits_remaining(), 0);
            assert_eq!(read_vlc, write_vlc);
            assert_eq!(read_scan, write_scan);
        }
    }

    #[test]
    fn written_block_matches_the_public_run_level_decoder() {
        // Scan order zero is the identity for the first two positions, so
        // raster index 4 is scan index 1 and raster index 1 is scan index 2.
        let mut block = [0_i32; 16];
        block[4] = -1;
        block[1] = 5;
        block[15] = 1;
        let mut writer = BitWriter::new();
        encode_lp_block(
            &mut writer,
            ComponentClass::Luma,
            1,
            &mut AcVlcState::new(),
            &mut AdaptiveLpScan::new(),
            &block,
        )
        .unwrap();
        let bytes = writer.into_bytes();
        let decoded = decode_ac_block(
            &mut PacketBitReader::new(&bytes),
            FrequencyBand::Lowpass,
            ComponentClass::Luma,
            1,
            &mut AcVlcState::new(),
        )
        .unwrap();
        let entries: Vec<(u8, i32)> = decoded
            .entries()
            .iter()
            .map(|entry| (entry.run, entry.level))
            .collect();
        assert_eq!(entries, [(0, -1), (0, 5), (12, 1)]);
    }

    #[test]
    fn ordered_blocks_round_trip_from_every_start_location() {
        let mut rng = Lcg(6);
        for round in 0..600 {
            let start = 1 + u8::try_from(rng.below(15)).unwrap();
            let count = 16 - usize::from(start);
            let class = if round % 2 == 0 {
                ComponentClass::Luma
            } else {
                ComponentClass::Chroma
            };
            let mut write_vlc = AcVlcState::new();
            let mut read_vlc = AcVlcState::new();
            let blocks: Vec<Vec<i32>> = (0..24)
                .map(|_| random_block(&mut rng, 0)[..count].to_vec())
                .filter(|block| block.iter().any(|&value| value != 0))
                .collect();
            let mut writer = BitWriter::new();
            let mut counts = Vec::new();
            for (index, block) in blocks.iter().enumerate() {
                counts.push(
                    encode_ordered_block(&mut writer, class, start, &mut write_vlc, block).unwrap(),
                );
                if index % 5 == 4 {
                    write_vlc.adapt();
                }
            }
            let bit_length = writer.bit_position();
            let bytes = writer.into_bytes();
            let mut reader = PacketBitReader::with_bit_length(&bytes, bit_length).unwrap();
            for (index, block) in blocks.iter().enumerate() {
                let decoded = decode_ac_block(
                    &mut reader,
                    FrequencyBand::Lowpass,
                    class,
                    start,
                    &mut read_vlc,
                )
                .unwrap();
                // Each run counts the zeros since the previous coefficient.
                let mut values = vec![0_i32; count];
                let mut cursor = 0_usize;
                for entry in decoded.entries() {
                    cursor += usize::from(entry.run);
                    values[cursor] = entry.level;
                    cursor += 1;
                }
                assert_eq!(&values, block, "start {start}");
                assert_eq!(decoded.non_zero_count(), counts[index]);
                if index % 5 == 4 {
                    read_vlc.adapt();
                }
            }
            assert_eq!(reader.bits_remaining(), 0);
            assert_eq!(read_vlc, write_vlc);
        }
    }

    #[test]
    fn ordered_blocks_must_fit_the_locations_after_their_start() {
        let mut writer = BitWriter::new();
        let mut state = AcVlcState::new();
        for (start, length) in [(0_u8, 3_usize), (10, 7), (16, 1)] {
            let result = encode_ordered_block(
                &mut writer,
                ComponentClass::Chroma,
                start,
                &mut state,
                &vec![1; length],
            );
            assert!(
                matches!(result, Err(EntropyError::InvalidParameter { .. })),
                "start {start} length {length}"
            );
        }
        assert_eq!(writer.bit_position(), 0);
    }

    #[test]
    fn empty_block_is_rejected_instead_of_written() {
        let mut writer = BitWriter::new();
        let result = encode_lp_block(
            &mut writer,
            ComponentClass::Luma,
            1,
            &mut AcVlcState::new(),
            &mut AdaptiveLpScan::new(),
            &[0; 16],
        );
        assert!(matches!(result, Err(EntropyError::InvalidParameter { .. })));
        assert_eq!(writer.bit_position(), 0);
    }
}
