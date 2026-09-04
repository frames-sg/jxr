//! DC, LP, HP, and flexbits coefficient parsing primitives.

use super::adaptive::{
    AcVlcState, AdaptiveVlc, DcVlcState, observe_abs_level, observe_first_index, observe_index,
};
use super::vlc;
use super::{AdaptiveHpScan, AdaptiveLpScan, EntropyError, HpScanDirection, PacketBitReader};

const ABS_REMAP: [i32; 6] = [2, 3, 4, 6, 10, 14];
const ABS_FIXED_LENGTH: [u8; 6] = [0, 0, 1, 2, 2, 2];
const RUN_REMAP: [u8; 15] = [1, 2, 3, 5, 7, 1, 2, 3, 5, 7, 1, 2, 3, 4, 5];
const RUN_BIN: [i8; 15] = [-1, -1, -1, -1, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0];
const RUN_FIXED_LENGTH: [u8; 15] = [0, 0, 1, 1, 3, 0, 0, 1, 1, 2, 0, 0, 0, 0, 1];

/// A JPEG XR frequency band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrequencyBand {
    /// DC coefficients.
    Dc = 0,
    /// Lowpass coefficients.
    Lowpass = 1,
    /// Highpass coefficients.
    Highpass = 2,
}

/// Whether a syntax element uses luma or chroma adaptive state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentClass {
    /// The first component in an image plane.
    Luma,
    /// Any component after the first.
    Chroma,
}

impl ComponentClass {
    const fn is_chroma(self) -> bool {
        matches!(self, Self::Chroma)
    }
}

/// One run/level pair in codestream order.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunLevel {
    /// Number of zero coefficients before this coefficient.
    pub run: u8,
    /// Signed, non-zero VLC-coded coefficient part.
    pub level: i32,
}

/// The non-zero coefficients decoded from one 4-by-4 block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedBlock {
    entries: [RunLevel; 15],
    len: u8,
    start_location: u8,
}

impl DecodedBlock {
    /// Returns run/level pairs in codestream order.
    #[must_use]
    pub fn entries(&self) -> &[RunLevel] {
        &self.entries[..usize::from(self.len)]
    }

    /// Returns the number of non-zero coefficients.
    #[must_use]
    pub const fn non_zero_count(&self) -> u8 {
        self.len
    }

    /// Applies this block to the tile-local LP inverse scan.
    pub fn inverse_scan_lp(
        &self,
        scan: &mut AdaptiveLpScan,
        output: &mut [i32; 16],
    ) -> Result<(), EntropyError> {
        self.scan_entries(|index, level| scan.place(output, index, level))
    }

    /// Applies this block to the selected tile-local HP inverse scan.
    pub fn inverse_scan_hp(
        &self,
        scan: &mut AdaptiveHpScan,
        direction: HpScanDirection,
        output: &mut [i32; 16],
    ) -> Result<(), EntropyError> {
        self.scan_entries(|index, level| scan.place(direction, output, index, level))
    }

    fn scan_entries(
        &self,
        mut place: impl FnMut(u8, i32) -> Result<(), EntropyError>,
    ) -> Result<(), EntropyError> {
        let mut index = self.start_location;
        for entry in self.entries() {
            index = index
                .checked_add(entry.run)
                .ok_or(EntropyError::CoefficientOverflow)?;
            place(index, entry.level)?;
            index = index
                .checked_add(1)
                .ok_or(EntropyError::CoefficientOverflow)?;
        }
        Ok(())
    }
}

/// Decodes one DC coefficient as specified by T.832 Table 49.
pub fn decode_dc_coefficient(
    reader: &mut PacketBitReader<'_>,
    model_bits: u8,
    has_abs_level: bool,
    class: ComponentClass,
    state: &mut DcVlcState,
) -> Result<i32, EntropyError> {
    if model_bits > 15 {
        return Err(EntropyError::InvalidParameter {
            parameter: "DC model bits",
            value: i64::from(model_bits),
        });
    }
    let mut coefficient = if has_abs_level {
        decode_abs_level(reader, state.abs_level(class.is_chroma()))?
            .checked_sub(1)
            .ok_or(EntropyError::CoefficientOverflow)?
    } else {
        0
    };
    if model_bits != 0 {
        let refinement = i32::try_from(reader.read_bits(model_bits)?)
            .map_err(|_| EntropyError::CoefficientOverflow)?;
        coefficient = coefficient
            .checked_shl(u32::from(model_bits))
            .and_then(|value| value.checked_add(refinement))
            .ok_or(EntropyError::CoefficientOverflow)?;
    }
    apply_sign(reader, coefficient)
}

/// Decodes the run/level representation of one LP or HP 4-by-4 block.
pub fn decode_ac_block(
    reader: &mut PacketBitReader<'_>,
    band: FrequencyBand,
    class: ComponentClass,
    start_location: u8,
    state: &mut AcVlcState,
) -> Result<DecodedBlock, EntropyError> {
    if band == FrequencyBand::Dc {
        return Err(EntropyError::InvalidParameter {
            parameter: "AC frequency band",
            value: i64::from(band as u8),
        });
    }
    if !(1..=15).contains(&start_location) {
        return Err(EntropyError::InvalidParameter {
            parameter: "block start location",
            value: i64::from(start_location),
        });
    }

    let chroma = class.is_chroma();
    let first = decode_first_index(reader, state.first_index(chroma))?;
    let sign = reader.read_bit()?;
    let mut zero_run = (first & 1) == 0;
    let mut successors = first >> 2;
    let mut context = (first & 1 != 0) && (successors & 1 != 0);
    let mut location = start_location;
    let mut output = DecodedBlock {
        entries: [RunLevel::default(); 15],
        len: 0,
        start_location,
    };

    let magnitude = if first & 2 != 0 {
        decode_abs_level(reader, state.abs_level(context))?
    } else {
        1
    };
    let run = if zero_run {
        decode_run(reader, 15 - location)?
    } else {
        0
    };
    location = advance_location(location, run)?;
    output.push(run, signed(magnitude, sign)?)?;

    while successors != 0 {
        zero_run = successors & 1 == 0;
        let run = if zero_run {
            decode_run(reader, 15_u8.saturating_sub(location))?
        } else {
            0
        };
        location = advance_location(location, run)?;
        let index = decode_index(reader, location, state.index(chroma, context))?;
        successors = index >> 1;
        context = context && (successors & 1 != 0);
        let sign = reader.read_bit()?;
        let magnitude = if index & 1 != 0 {
            decode_abs_level(reader, state.abs_level(context))?
        } else {
            1
        };
        output.push(run, signed(magnitude, sign)?)?;
    }
    Ok(output)
}

impl DecodedBlock {
    fn push(&mut self, run: u8, level: i32) -> Result<(), EntropyError> {
        let Some(slot) = self.entries.get_mut(usize::from(self.len)) else {
            return Err(EntropyError::InvalidParameter {
                parameter: "non-zero coefficients in block",
                value: i64::from(self.len) + 1,
            });
        };
        *slot = RunLevel { run, level };
        self.len += 1;
        Ok(())
    }
}

fn advance_location(location: u8, run: u8) -> Result<u8, EntropyError> {
    let next = location
        .checked_add(run)
        .and_then(|value| value.checked_add(1))
        .ok_or(EntropyError::InvalidParameter {
            parameter: "coefficient location",
            value: i64::from(location) + i64::from(run) + 1,
        })?;
    if next > 16 {
        return Err(EntropyError::InvalidParameter {
            parameter: "coefficient location",
            value: i64::from(next),
        });
    }
    Ok(next)
}

fn decode_abs_level(
    reader: &mut PacketBitReader<'_>,
    state: &mut AdaptiveVlc,
) -> Result<i32, EntropyError> {
    let index = vlc::decode(
        reader,
        "ABS_LEVEL_INDEX",
        vlc::ABS_LEVEL[state.table_index()],
    )?;
    observe_abs_level(state, index);
    if index < 6 {
        let fixed = ABS_FIXED_LENGTH[usize::from(index)];
        let refinement = i32::try_from(reader.read_bits(fixed)?)
            .map_err(|_| EntropyError::CoefficientOverflow)?;
        return ABS_REMAP[usize::from(index)]
            .checked_add(refinement)
            .ok_or(EntropyError::CoefficientOverflow);
    }

    let mut fixed =
        u8::try_from(reader.read_bits(4)?).map_err(|_| EntropyError::CoefficientOverflow)? + 4;
    if fixed == 19 {
        fixed +=
            u8::try_from(reader.read_bits(2)?).map_err(|_| EntropyError::CoefficientOverflow)?;
        if fixed == 22 {
            fixed += u8::try_from(reader.read_bits(3)?)
                .map_err(|_| EntropyError::CoefficientOverflow)?;
        }
    }
    let refinement =
        i32::try_from(reader.read_bits(fixed)?).map_err(|_| EntropyError::CoefficientOverflow)?;
    (1_i32.checked_shl(u32::from(fixed)))
        .and_then(|base| base.checked_add(2))
        .and_then(|base| base.checked_add(refinement))
        .ok_or(EntropyError::CoefficientOverflow)
}

fn decode_first_index(
    reader: &mut PacketBitReader<'_>,
    state: &mut AdaptiveVlc,
) -> Result<u8, EntropyError> {
    let symbol = vlc::decode(reader, "FIRST_INDEX", vlc::FIRST_INDEX[state.table_index()])?;
    observe_first_index(state, symbol);
    Ok(symbol)
}

fn decode_index(
    reader: &mut PacketBitReader<'_>,
    location: u8,
    state: &mut AdaptiveVlc,
) -> Result<u8, EntropyError> {
    match location.cmp(&15) {
        core::cmp::Ordering::Less => {
            let symbol = vlc::decode(reader, "INDEX_A", vlc::INDEX_A[state.table_index()])?;
            observe_index(state, symbol);
            Ok(symbol)
        }
        core::cmp::Ordering::Equal => vlc::decode(reader, "INDEX_B", vlc::INDEX_B),
        core::cmp::Ordering::Greater => Ok(u8::from(reader.read_bit()?)),
    }
}

fn decode_run(reader: &mut PacketBitReader<'_>, max_run: u8) -> Result<u8, EntropyError> {
    if max_run == 0 || max_run > 14 {
        return Err(EntropyError::InvalidParameter {
            parameter: "maximum coefficient run",
            value: i64::from(max_run),
        });
    }
    if max_run < 5 {
        return match max_run {
            1 => Ok(1),
            2 => vlc::decode(reader, "RUN_VALUE", vlc::RUN_VALUE_2),
            3 => vlc::decode(reader, "RUN_VALUE", vlc::RUN_VALUE_3),
            4 => vlc::decode(reader, "RUN_VALUE", vlc::RUN_VALUE_4),
            _ => unreachable!(),
        };
    }

    let run_index = vlc::decode(reader, "RUN_INDEX", vlc::RUN_INDEX)?;
    let table_index = i16::from(run_index) + 5 * i16::from(RUN_BIN[usize::from(max_run)]);
    let table_index = usize::try_from(table_index).map_err(|_| EntropyError::InvalidParameter {
        parameter: "run table index",
        value: i64::from(table_index),
    })?;
    let fixed = RUN_FIXED_LENGTH[table_index];
    let refinement =
        u8::try_from(reader.read_bits(fixed)?).map_err(|_| EntropyError::CoefficientOverflow)?;
    Ok(RUN_REMAP[table_index] + refinement)
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

fn apply_sign(reader: &mut PacketBitReader<'_>, value: i32) -> Result<i32, EntropyError> {
    if value == 0 {
        return Ok(0);
    }
    signed(value, reader.read_bit()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_without_abs_level_uses_refinement_and_sign() {
        let mut reader = PacketBitReader::with_bit_length(&[0b1011_0000], 4).unwrap();
        let value = decode_dc_coefficient(
            &mut reader,
            3,
            false,
            ComponentClass::Luma,
            &mut DcVlcState::new(),
        )
        .unwrap();
        assert_eq!(value, -5);
    }

    #[test]
    fn dc_abs_level_table_zero_decodes_initial_level() {
        // ABS_LEVEL_INDEX 0 (`01`), no model bits, positive sign.
        let mut reader = PacketBitReader::with_bit_length(&[0b0100_0000], 3).unwrap();
        let value = decode_dc_coefficient(
            &mut reader,
            0,
            true,
            ComponentClass::Luma,
            &mut DcVlcState::new(),
        )
        .unwrap();
        assert_eq!(value, 1);
    }

    #[test]
    fn one_coefficient_block_decodes_first_index_table_one() {
        // FIRST_INDEX value 1 (`00010`) means zero run, magnitude one, last.
        let mut reader = PacketBitReader::with_bit_length(&[0b0001_0100], 6).unwrap();
        let block = decode_ac_block(
            &mut reader,
            FrequencyBand::Lowpass,
            ComponentClass::Luma,
            1,
            &mut AcVlcState::new(),
        )
        .unwrap();
        assert_eq!(block.entries(), &[RunLevel { run: 0, level: -1 }]);
        let mut coefficients = [0; 16];
        block
            .inverse_scan_lp(&mut AdaptiveLpScan::new(), &mut coefficients)
            .unwrap();
        assert_eq!(coefficients[4], -1);
    }

    #[test]
    fn first_coefficient_reads_level_before_zero_run() {
        // FIRST_INDEX 6 (`00011`) carries both an absolute level and a run.
        // The syntax orders sign, absolute level (`01` => 2), then run
        // (`1` => 1), followed here by a final magnitude-one coefficient.
        let mut reader = PacketBitReader::with_bit_length(&[0b0001_1101, 0b1010_0000], 12).unwrap();
        let block = decode_ac_block(
            &mut reader,
            FrequencyBand::Highpass,
            ComponentClass::Luma,
            1,
            &mut AcVlcState::new(),
        )
        .unwrap();
        assert_eq!(
            block.entries(),
            &[
                RunLevel { run: 1, level: -2 },
                RunLevel { run: 0, level: 1 }
            ]
        );
        assert_eq!(reader.bits_remaining(), 0);
    }
}
