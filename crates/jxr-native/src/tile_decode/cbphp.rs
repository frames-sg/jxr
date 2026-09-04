//! Coded-block-pattern parsing and prediction for Y-only and YUV444 tiles.

use jxr_core::ChromaSampling;

use crate::entropy::PacketBitReader;

use super::TileDecodeError;

const NUM_CODES: [[(u16, u8, u8); 5]; 2] = [
    [
        (0b1, 1, 0),
        (0b01, 2, 1),
        (0b001, 3, 2),
        (0b0000, 4, 3),
        (0b0001, 4, 4),
    ],
    [
        (0b1, 1, 0),
        (0b000, 3, 1),
        (0b001, 3, 2),
        (0b010, 3, 3),
        (0b011, 3, 4),
    ],
];
const NUM_CODES_YUV: [[(u16, u8, u8); 9]; 2] = [
    [
        (0b010, 3, 0),
        (0b00000, 5, 1),
        (0b0010, 4, 2),
        (0b00001, 5, 3),
        (0b00010, 5, 4),
        (0b1, 1, 5),
        (0b011, 3, 6),
        (0b00011, 5, 7),
        (0b0011, 4, 8),
    ],
    [
        (0b1, 1, 0),
        (0b001, 3, 1),
        (0b010, 3, 2),
        (0b0001, 4, 3),
        (0b00_0001, 6, 4),
        (0b011, 3, 5),
        (0b00001, 5, 6),
        (0b000_0000, 7, 7),
        (0b000_0001, 7, 8),
    ],
];
const CHROMA_CODES: [(u16, u8, u8); 3] = [(0b1, 1, 0), (0b01, 2, 1), (0b00, 2, 2)];
const CHROMA_BLOCK_CODES: [(u16, u8, u8); 4] =
    [(0b1, 1, 0), (0b01, 2, 1), (0b000, 3, 2), (0b001, 3, 3)];
const REF_TWO: [(u16, u8, u8); 6] = [
    (0b00, 2, 3),
    (0b01, 2, 5),
    (0b100, 3, 6),
    (0b101, 3, 9),
    (0b110, 3, 10),
    (0b111, 3, 12),
];
const DELTA: [i8; 5] = [0, -1, 0, 1, 1];
const DELTA_YUV: [i8; 9] = [2, 2, 1, 1, -1, -2, -2, -2, -3];
const BLOCK_OUTPUT: [u16; 16] = [0, 15, 3, 12, 1, 2, 4, 8, 5, 6, 9, 10, 7, 11, 13, 14];
const BLOCK_OFFSET: [usize; 6] = [0, 4, 2, 8, 12, 1];
const BLOCK_BITS: [u8; 6] = [0, 2, 1, 2, 2, 0];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TwoTableState {
    table: usize,
    discriminant: i32,
}

impl TwoTableState {
    const fn new() -> Self {
        Self {
            table: 0,
            discriminant: 0,
        }
    }

    fn observe(&mut self, symbol: u8, delta: &[i8]) {
        self.discriminant += i32::from(delta[usize::from(symbol)]);
    }

    fn adapt(&mut self) {
        if self.discriminant < -8 && self.table != 0 {
            self.table -= 1;
            self.discriminant = 0;
        } else if self.discriminant > 8 && self.table != 1 {
            self.table += 1;
            self.discriminant = 0;
        } else {
            self.discriminant = self.discriminant.clamp(-64, 64);
        }
    }
}

/// Tile-local CBPHP entropy and spatial-prediction state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CbphpState {
    groups: TwoTableState,
    blocks: TwoTableState,
    prediction: [PredictionState; 2],
    previous_rows: Vec<Vec<u16>>,
}

impl CbphpState {
    pub(super) fn new(tile_width: usize) -> Self {
        Self::new_components(tile_width, 3)
    }

    pub(super) fn new_components(tile_width: usize, components: usize) -> Self {
        Self {
            groups: TwoTableState::new(),
            blocks: TwoTableState::new(),
            prediction: [PredictionState::new(), PredictionState::new()],
            previous_rows: (0..components).map(|_| vec![0; tile_width]).collect(),
        }
    }

    pub(super) fn decode(
        &mut self,
        reader: &mut PacketBitReader<'_>,
        x: usize,
        y: usize,
    ) -> Result<u16, TileDecodeError> {
        self.decode_component(reader, x, y, 0)
    }

    pub(super) fn decode_component(
        &mut self,
        reader: &mut PacketBitReader<'_>,
        x: usize,
        y: usize,
        component: usize,
    ) -> Result<u16, TileDecodeError> {
        if component >= self.previous_rows.len() {
            return Err(TileDecodeError::InvalidPlan("CBPHP component index"));
        }
        let group_count = read_code(reader, &NUM_CODES[self.groups.table], "NUM_CBPHP")?;
        self.groups.observe(group_count, &DELTA);
        let groups = refine(reader, group_count)?;
        let mut residual = 0_u16;
        for group in 0..4 {
            if groups & (1 << group) == 0 {
                continue;
            }
            let block_count = read_code(reader, &NUM_CODES[self.blocks.table], "NUM_BLKCBPHP")?;
            self.blocks.observe(block_count, &DELTA);
            let value = usize::from(block_count) + 1;
            let increment = if BLOCK_BITS[value] == 0 {
                0
            } else {
                usize::try_from(reader.read_bits(BLOCK_BITS[value])?)
                    .map_err(|_| TileDecodeError::ArithmeticOverflow("CBPHP code increment"))?
            };
            let code = BLOCK_OFFSET[value]
                .checked_add(increment)
                .ok_or(TileDecodeError::ArithmeticOverflow("CBPHP code"))?;
            residual |= BLOCK_OUTPUT[code] << (group * 4);
        }
        let class = usize::from(component != 0);
        let top = (y != 0).then(|| self.previous_rows[component][x]);
        let left = (x != 0).then(|| self.previous_rows[component][x - 1]);
        let actual = self.prediction[class].predict_444(residual, left, top);
        self.previous_rows[component][x] = actual;
        Ok(actual)
    }

    pub(super) fn decode_yuv(
        &mut self,
        reader: &mut PacketBitReader<'_>,
        x: usize,
        y: usize,
        sampling: ChromaSampling,
    ) -> Result<[u16; 3], TileDecodeError> {
        let group_count = read_code(reader, &NUM_CODES[self.groups.table], "NUM_CBPHP")?;
        self.groups.observe(group_count, &DELTA);
        let groups = refine(reader, group_count)?;
        let mut residual = [0_u16; 3];
        for group in 0..4 {
            if groups & (1 << group) == 0 {
                continue;
            }
            let block_count = read_code(
                reader,
                &NUM_CODES_YUV[self.blocks.table],
                "NUM_BLKCBPHP_YUV",
            )?;
            self.blocks.observe(block_count, &DELTA_YUV);
            decode_yuv_group(reader, block_count, group, sampling, &mut residual)?;
        }
        for (component, residual_value) in residual.iter_mut().enumerate() {
            let class = usize::from(component != 0);
            let top = (y != 0).then(|| self.previous_rows[component][x]);
            let left = (x != 0).then(|| self.previous_rows[component][x - 1]);
            let actual = match (component, sampling) {
                (0, _) | (_, ChromaSampling::Cs444) => {
                    self.prediction[class].predict_444(*residual_value, left, top)
                }
                (_, ChromaSampling::Cs422) => {
                    self.prediction[class].predict_422(*residual_value, left, top)
                }
                (_, ChromaSampling::Cs420) => {
                    self.prediction[class].predict_420(*residual_value, left, top)
                }
            };
            self.previous_rows[component][x] = actual;
            *residual_value = actual;
        }
        Ok(residual)
    }

    pub(super) fn adapt(&mut self) {
        self.groups.adapt();
        self.blocks.adapt();
    }
}

fn decode_yuv_group(
    reader: &mut PacketBitReader<'_>,
    block_count: u8,
    group: usize,
    sampling: ChromaSampling,
    residual: &mut [u16; 3],
) -> Result<(), TileDecodeError> {
    let mut value = usize::from(block_count) + 1;
    let mut chroma_mask = 0_u8;
    if value >= 6 {
        chroma_mask = read_code(reader, &CHROMA_CODES, "CHR_CBPHP")? + 1;
        if value >= 9 {
            value = value
                .checked_add(usize::from(read_code(reader, &CHROMA_CODES, "VAL_INC")?))
                .ok_or(TileDecodeError::ArithmeticOverflow("YUV CBPHP value"))?;
        }
        value -= 6;
    }
    let increment = if BLOCK_BITS[value] == 0 {
        0
    } else {
        usize::try_from(reader.read_bits(BLOCK_BITS[value])?)
            .map_err(|_| TileDecodeError::ArithmeticOverflow("YUV CBPHP increment"))?
    };
    let code = BLOCK_OFFSET[value]
        .checked_add(increment)
        .ok_or(TileDecodeError::ArithmeticOverflow("YUV CBPHP code"))?;
    residual[0] |= (BLOCK_OUTPUT[code] & 0x0f) << (group * 4);
    for component in 0..2 {
        if chroma_mask & (1 << component) != 0 {
            match sampling {
                ChromaSampling::Cs444 => {
                    let count = read_code(reader, &CHROMA_BLOCK_CODES, "NUM_CH_BLK")? + 1;
                    residual[component + 1] |= refine(reader, count)? << (group * 4);
                }
                ChromaSampling::Cs422 => {
                    const SHIFT: [usize; 4] = [0, 1, 4, 5];
                    let code = usize::from(read_code(reader, &CHROMA_CODES, "CBPHP_CH_BLK")?);
                    residual[component + 1] |= u16::try_from(SHIFT[code + 1] << SHIFT[group])
                        .map_err(|_| TileDecodeError::ArithmeticOverflow("YUV422 CBPHP remap"))?;
                }
                ChromaSampling::Cs420 => {
                    residual[component + 1] |= 1 << group;
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PredictionState {
    state: i8,
    ones: i8,
    zeroes: i8,
}

impl PredictionState {
    const fn new() -> Self {
        Self {
            state: 0,
            ones: -4,
            zeroes: 4,
        }
    }

    fn predict_444(&mut self, residual: u16, left: Option<u16>, top: Option<u16>) -> u16 {
        let mut actual = residual;
        if self.state == 0 {
            let seed = match (left, top) {
                (None, None) => 1,
                (None, Some(top)) => (top >> 10) & 1,
                (Some(left), _) => (left >> 5) & 1,
            };
            actual ^= seed;
            actual ^= 0x0002 & (actual << 1);
            actual ^= 0x0010 & (actual << 3);
            actual ^= 0x0020 & (actual << 1);
            actual ^= (actual & 0x0033) << 2;
            actual ^= (actual & 0x00cc) << 6;
            actual ^= (actual & 0x3300) << 2;
        } else if self.state == 2 {
            actual ^= u16::MAX;
        }
        self.update(i8::try_from(actual.count_ones()).unwrap_or(16));
        actual
    }

    fn predict_422(&mut self, residual: u16, left: Option<u16>, top: Option<u16>) -> u16 {
        let mut actual = residual;
        if self.state == 0 {
            actual ^= match (left, top) {
                (None, None) => 1,
                (None, Some(top)) => (top >> 6) & 1,
                (Some(left), _) => (left >> 1) & 1,
            };
            actual ^= (actual & 0x01) << 1;
            actual ^= (actual & 0x03) << 2;
            actual ^= (actual & 0x0c) << 2;
            actual ^= (actual & 0x30) << 2;
        } else if self.state == 2 {
            actual ^= 0x00ff;
        }
        self.update(i8::try_from(actual.count_ones() * 2).unwrap_or(16));
        actual
    }

    fn predict_420(&mut self, residual: u16, left: Option<u16>, top: Option<u16>) -> u16 {
        let mut actual = residual;
        if self.state == 0 {
            actual ^= match (left, top) {
                (None, None) => 1,
                (None, Some(top)) => (top >> 2) & 1,
                (Some(left), _) => (left >> 1) & 1,
            };
            actual ^= 0x02 & (actual << 1);
            actual ^= (actual & 0x03) << 2;
        } else if self.state == 2 {
            actual ^= 0x000f;
        }
        self.update(i8::try_from(actual.count_ones() * 4).unwrap_or(16));
        actual
    }

    fn update(&mut self, count: i8) {
        self.ones = (self.ones + count - 3).clamp(-16, 15);
        self.zeroes = (self.zeroes + 16 - count - 3).clamp(-16, 15);
        self.state = if self.ones < 0 {
            if self.ones < self.zeroes { 1 } else { 2 }
        } else if self.zeroes < 0 {
            2
        } else {
            0
        };
    }
}

fn refine(reader: &mut PacketBitReader<'_>, count: u8) -> Result<u16, TileDecodeError> {
    match count {
        0 => Ok(0),
        1 => Ok(1 << reader.read_bits(2)?),
        2 => Ok(u16::from(read_code(reader, &REF_TWO, "REF_CBPHP1")?)),
        3 => Ok(0x0f ^ (1 << reader.read_bits(2)?)),
        4 => Ok(0x0f),
        _ => Err(TileDecodeError::InvalidPlan("CBPHP group count")),
    }
}

fn read_code<const N: usize>(
    reader: &mut PacketBitReader<'_>,
    table: &[(u16, u8, u8); N],
    syntax: &'static str,
) -> Result<u8, TileDecodeError> {
    let start = reader.bit_position();
    let max_length = table.iter().map(|entry| entry.1).max().unwrap_or(0);
    let mut bits = 0_u16;
    for length in 1..=max_length {
        bits = (bits << 1) | u16::from(reader.read_bit()?);
        if let Some(entry) = table
            .iter()
            .find(|entry| entry.1 == length && entry.0 == bits)
        {
            return Ok(entry.2);
        }
    }
    Err(crate::entropy::EntropyError::InvalidVlc {
        syntax,
        bit_position: start,
    }
    .into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_zero_residual_predicts_normative_checker_pattern() {
        let mut prediction = PredictionState::new();
        assert_eq!(prediction.predict_444(0, None, None), 0xffff);
    }

    #[test]
    fn refine_two_uses_table_64() {
        let mut reader = PacketBitReader::with_bit_length(&[0b1010_0000], 3).unwrap();
        assert_eq!(refine(&mut reader, 2).unwrap(), 9);
    }
}
