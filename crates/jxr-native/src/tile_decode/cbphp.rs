//! Coded-block-pattern coding and prediction shared by tile decoding and encoding.

use jxr_core::ChromaSampling;

use crate::{
    bit_writer::BitWriter,
    entropy::{AdaptiveVlc, CodeTable, PacketBitReader, PrefixTable, decode_prefix, encode_prefix},
};

use super::TileDecodeError;

const NUM_CODE_LISTS: [[(u16, u8, u8); 5]; 2] = [
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
const NUM_CODE_LISTS_YUV: [[(u16, u8, u8); 9]; 2] = [
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
const CHROMA_CODE_LIST: [(u16, u8, u8); 3] = [(0b1, 1, 0), (0b01, 2, 1), (0b00, 2, 2)];
const CHROMA_BLOCK_CODE_LIST: [(u16, u8, u8); 4] =
    [(0b1, 1, 0), (0b01, 2, 1), (0b000, 3, 2), (0b001, 3, 3)];
const REF_TWO_CODE_LIST: [(u16, u8, u8); 6] = [
    (0b00, 2, 3),
    (0b01, 2, 5),
    (0b100, 3, 6),
    (0b101, 3, 9),
    (0b110, 3, 10),
    (0b111, 3, 12),
];
static NUM_CODES: [PrefixTable; 2] = [
    PrefixTable::new(&NUM_CODE_LISTS[0]),
    PrefixTable::new(&NUM_CODE_LISTS[1]),
];
static NUM_CODES_YUV: [PrefixTable; 2] = [
    PrefixTable::new(&NUM_CODE_LISTS_YUV[0]),
    PrefixTable::new(&NUM_CODE_LISTS_YUV[1]),
];
static CHROMA_CODES: PrefixTable = PrefixTable::new(&CHROMA_CODE_LIST);
static CHROMA_BLOCK_CODES: PrefixTable = PrefixTable::new(&CHROMA_BLOCK_CODE_LIST);
static REF_TWO: PrefixTable = PrefixTable::new(&REF_TWO_CODE_LIST);
static NUM_CODES_WRITE: [CodeTable; 2] = [
    CodeTable::new(&NUM_CODE_LISTS[0]),
    CodeTable::new(&NUM_CODE_LISTS[1]),
];
static NUM_CODES_YUV_WRITE: [CodeTable; 2] = [
    CodeTable::new(&NUM_CODE_LISTS_YUV[0]),
    CodeTable::new(&NUM_CODE_LISTS_YUV[1]),
];
static CHROMA_CODES_WRITE: CodeTable = CodeTable::new(&CHROMA_CODE_LIST);
static CHROMA_BLOCK_CODES_WRITE: CodeTable = CodeTable::new(&CHROMA_BLOCK_CODE_LIST);
static REF_TWO_WRITE: CodeTable = CodeTable::new(&REF_TWO_CODE_LIST);
const DELTA: [i8; 5] = [0, -1, 0, 1, 1];
const DELTA_YUV: [i8; 9] = [2, 2, 1, 1, -1, -2, -2, -2, -3];
const BLOCK_OUTPUT: [u16; 16] = [0, 15, 3, 12, 1, 2, 4, 8, 5, 6, 9, 10, 7, 11, 13, 14];
const BLOCK_OFFSET: [usize; 6] = [0, 4, 2, 8, 12, 1];
const BLOCK_BITS: [u8; 6] = [0, 2, 1, 2, 2, 0];
/// Bit of a 4:2:2 chroma pattern holding the first of the two blocks each
/// group covers. The group's second block, one block row below the first,
/// is two bits above it.
const GROUP_SHIFT_422: [usize; 4] = [0, 1, 4, 5];

/// The pattern bits of one group of a 4:2:2 chroma component, from the two
/// bits saying which of the group's blocks are coded: the first in bit
/// zero, the second in bit one.
const fn spread_422(blocks: u8) -> u16 {
    (blocks & 1) as u16 | ((blocks & 2) as u16) << 1
}

/// Inverse of [`spread_422`]: which of a group's two blocks are coded, from
/// a 4:2:2 chroma pattern shifted down to the group's first block.
const fn gather_422(pattern: u16) -> u16 {
    (pattern & 1) | (pattern >> 1 & 2)
}

/// Inverse of [`BLOCK_OUTPUT`]: the code selecting each non-zero block nibble.
const BLOCK_CODE: [u8; 16] = block_codes();

const fn block_codes() -> [u8; 16] {
    let mut codes = [0_u8; 16];
    let mut code = 0_u8;
    while (code as usize) < BLOCK_OUTPUT.len() {
        codes[BLOCK_OUTPUT[code as usize] as usize] = code;
        code += 1;
    }
    codes
}

/// Tile-local CBPHP entropy and spatial-prediction state.
///
/// The number of groups and of blocks in a group are each coded with one of
/// two code tables, which adapt as coefficient levels do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CbphpState {
    groups: AdaptiveVlc,
    blocks: AdaptiveVlc,
    prediction: [PredictionState; 2],
    previous_rows: Vec<Vec<u16>>,
}

impl CbphpState {
    /// The state a tile `tile_width` macroblocks wide of a plane of
    /// `components` components starts with.
    pub(crate) fn new(tile_width: usize, components: usize) -> Self {
        Self {
            groups: AdaptiveVlc::two_tables(),
            blocks: AdaptiveVlc::two_tables(),
            prediction: [PredictionState::new(), PredictionState::new()],
            previous_rows: (0..components).map(|_| vec![0; tile_width]).collect(),
        }
    }

    /// Reads one component's coded-block pattern: one bit per 4-by-4 block
    /// in hierarchical block order.
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
        let group_count =
            decode_prefix(reader, "NUM_CBPHP", &NUM_CODES[self.groups.table_index()])?;
        self.groups.observe_one(group_count, &DELTA);
        let groups = refine(reader, group_count)?;
        let mut residual = 0_u16;
        for group in 0..4 {
            if groups & (1 << group) == 0 {
                continue;
            }
            let block_count = decode_prefix(
                reader,
                "NUM_BLKCBPHP",
                &NUM_CODES[self.blocks.table_index()],
            )?;
            self.blocks.observe_one(block_count, &DELTA);
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

    /// Writes one component's coded-block pattern, the inverse of
    /// [`Self::decode_component`].
    ///
    /// `actual` has one bit per 4-by-4 block in hierarchical block order. The
    /// residual is taken against the same neighbours and prediction state a
    /// decoder uses, and that state advances exactly as it does when decoding.
    pub(crate) fn encode_component(
        &mut self,
        writer: &mut BitWriter,
        x: usize,
        y: usize,
        component: usize,
        actual: u16,
    ) -> Result<(), TileDecodeError> {
        if component >= self.previous_rows.len() {
            return Err(TileDecodeError::InvalidPlan("CBPHP component index"));
        }
        let class = usize::from(component != 0);
        let top = (y != 0).then(|| self.previous_rows[component][x]);
        let left = (x != 0).then(|| self.previous_rows[component][x - 1]);
        let residual = self.prediction[class].residual_444(actual, left, top);
        self.previous_rows[component][x] = actual;

        let mut groups = 0_u8;
        for group in 0..4 {
            if residual >> (group * 4) & 0x0f != 0 {
                groups |= 1 << group;
            }
        }
        let group_count = u8::try_from(groups.count_ones()).unwrap_or(4);
        write_code(
            writer,
            &NUM_CODES_WRITE[self.groups.table_index()],
            "NUM_CBPHP",
            group_count,
        )?;
        self.groups.observe_one(group_count, &DELTA);
        write_refinement(writer, groups)?;
        for group in 0..4 {
            if groups & (1 << group) == 0 {
                continue;
            }
            let (value, increment) = block_value(residual >> (group * 4) & 0x0f);
            let symbol = u8::try_from(value - 1).unwrap_or(0);
            write_code(
                writer,
                &NUM_CODES_WRITE[self.blocks.table_index()],
                "NUM_BLKCBPHP",
                symbol,
            )?;
            self.blocks.observe_one(symbol, &DELTA);
            writer.write_bits(increment, BLOCK_BITS[value]);
        }
        Ok(())
    }

    /// Writes the jointly coded patterns of the three components of a YUV
    /// macroblock, the inverse of [`Self::decode_yuv`].
    ///
    /// `actual` holds one bit per block of each component: sixteen in
    /// hierarchical block order for luma and 4:4:4 chroma, and eight or four
    /// in raster block order for 4:2:2 or 4:2:0 chroma.
    pub(crate) fn encode_yuv(
        &mut self,
        writer: &mut BitWriter,
        x: usize,
        y: usize,
        sampling: ChromaSampling,
        actual: [u16; 3],
    ) -> Result<(), TileDecodeError> {
        // Chroma components share one prediction state, so residuals are
        // formed in component order exactly as a decoder resolves them.
        let mut residual = [0_u16; 3];
        for component in 0..3 {
            let class = usize::from(component != 0);
            let top = (y != 0).then(|| self.previous_rows[component][x]);
            let left = (x != 0).then(|| self.previous_rows[component][x - 1]);
            let prediction = &mut self.prediction[class];
            residual[component] = match (component, sampling) {
                (0, _) | (_, ChromaSampling::Cs444) => {
                    prediction.residual_444(actual[component], left, top)
                }
                (_, ChromaSampling::Cs422) => prediction.residual_422(actual[component], left, top),
                (_, ChromaSampling::Cs420) => prediction.residual_420(actual[component], left, top),
            };
            self.previous_rows[component][x] = actual[component];
        }

        // The part of each component's residual one group codes.
        let luma = |group: usize| residual[0] >> (group * 4) & 0x0f;
        let chroma = |component: usize, group: usize| match sampling {
            ChromaSampling::Cs444 => residual[component] >> (group * 4) & 0x0f,
            ChromaSampling::Cs422 => gather_422(residual[component] >> GROUP_SHIFT_422[group]),
            ChromaSampling::Cs420 => residual[component] >> group & 1,
        };
        let mut groups = 0_u8;
        for group in 0..4 {
            if luma(group) != 0 || chroma(1, group) != 0 || chroma(2, group) != 0 {
                groups |= 1 << group;
            }
        }
        let group_count = u8::try_from(groups.count_ones()).unwrap_or(4);
        write_code(
            writer,
            &NUM_CODES_WRITE[self.groups.table_index()],
            "NUM_CBPHP",
            group_count,
        )?;
        self.groups.observe_one(group_count, &DELTA);
        write_refinement(writer, groups)?;
        for group in 0..4 {
            if groups & (1 << group) != 0 {
                let chroma = [chroma(1, group), chroma(2, group)];
                self.encode_yuv_group(writer, sampling, luma(group), chroma)?;
            }
        }
        Ok(())
    }

    /// Writes one group of a jointly coded pattern, the inverse of
    /// [`decode_yuv_group`]: the luma nibble with the chroma presence mask,
    /// then each present chroma component's part of the group.
    fn encode_yuv_group(
        &mut self,
        writer: &mut BitWriter,
        sampling: ChromaSampling,
        luma: u16,
        chroma: [u16; 2],
    ) -> Result<(), TileDecodeError> {
        let chroma_mask = u8::from(chroma[0] != 0) | (u8::from(chroma[1] != 0) << 1);
        // A zero luma nibble is value zero, which only a chroma-bearing
        // code can express.
        let (value, increment) = if luma == 0 { (0, 0) } else { block_value(luma) };
        // Symbols 0-4 are luma-only values 1-5. With chroma, values 0-2
        // are symbols 5-7 and values 3-5 share symbol 8 plus VAL_INC.
        let (symbol, value_increment) = match (chroma_mask, value) {
            (0, _) => (value - 1, None),
            (_, 0..=2) => (value + 5, None),
            _ => (8, Some(value - 3)),
        };
        let symbol = u8::try_from(symbol).unwrap_or(0);
        write_code(
            writer,
            &NUM_CODES_YUV_WRITE[self.blocks.table_index()],
            "NUM_BLKCBPHP_YUV",
            symbol,
        )?;
        self.blocks.observe_one(symbol, &DELTA_YUV);
        if chroma_mask != 0 {
            write_code(writer, &CHROMA_CODES_WRITE, "CHR_CBPHP", chroma_mask - 1)?;
        }
        if let Some(value_increment) = value_increment {
            let code = u8::try_from(value_increment).unwrap_or(0);
            write_code(writer, &CHROMA_CODES_WRITE, "VAL_INC", code)?;
        }
        writer.write_bits(increment, BLOCK_BITS[value]);
        for pattern in chroma {
            let pattern = u8::try_from(pattern).unwrap_or(0);
            if pattern == 0 {
                continue;
            }
            match sampling {
                ChromaSampling::Cs444 => {
                    let count = u8::try_from(pattern.count_ones()).unwrap_or(4);
                    write_code(writer, &CHROMA_BLOCK_CODES_WRITE, "NUM_CH_BLK", count - 1)?;
                    write_refinement(writer, pattern)?;
                }
                // Which of the group's two blocks are coded.
                ChromaSampling::Cs422 => {
                    write_code(writer, &CHROMA_CODES_WRITE, "CBPHP_CH_BLK", pattern - 1)?;
                }
                // The group's one block is coded, which the mask already says.
                ChromaSampling::Cs420 => {}
            }
        }
        Ok(())
    }

    /// Reads the jointly coded patterns of the three components of a YUV
    /// macroblock: see [`Self::encode_yuv`] for their shape.
    pub(super) fn decode_yuv(
        &mut self,
        reader: &mut PacketBitReader<'_>,
        x: usize,
        y: usize,
        sampling: ChromaSampling,
    ) -> Result<[u16; 3], TileDecodeError> {
        let group_count =
            decode_prefix(reader, "NUM_CBPHP", &NUM_CODES[self.groups.table_index()])?;
        self.groups.observe_one(group_count, &DELTA);
        let groups = refine(reader, group_count)?;
        let mut residual = [0_u16; 3];
        for group in 0..4 {
            if groups & (1 << group) == 0 {
                continue;
            }
            let block_count = decode_prefix(
                reader,
                "NUM_BLKCBPHP_YUV",
                &NUM_CODES_YUV[self.blocks.table_index()],
            )?;
            self.blocks.observe_one(block_count, &DELTA_YUV);
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

    pub(crate) fn adapt(&mut self) {
        self.groups.adapt_one();
        self.blocks.adapt_one();
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
        chroma_mask = decode_prefix(reader, "CHR_CBPHP", &CHROMA_CODES)? + 1;
        if value >= 9 {
            value = value
                .checked_add(usize::from(decode_prefix(
                    reader,
                    "VAL_INC",
                    &CHROMA_CODES,
                )?))
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
                    let count = decode_prefix(reader, "NUM_CH_BLK", &CHROMA_BLOCK_CODES)? + 1;
                    residual[component + 1] |= refine(reader, count)? << (group * 4);
                }
                ChromaSampling::Cs422 => {
                    let blocks = decode_prefix(reader, "CBPHP_CH_BLK", &CHROMA_CODES)? + 1;
                    residual[component + 1] |= spread_422(blocks) << GROUP_SHIFT_422[group];
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
            actual ^= seed(left, top, 5, 10);
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

    /// Inverse of [`Self::predict_444`]: the residual that predicts to `actual`.
    ///
    /// Each prediction step reads bits that no later step rewrites, so the
    /// steps undo one another when applied in reverse order.
    fn residual_444(&mut self, actual: u16, left: Option<u16>, top: Option<u16>) -> u16 {
        let mut residual = actual;
        if self.state == 0 {
            residual ^= (residual & 0x3300) << 2;
            residual ^= (residual & 0x00cc) << 6;
            residual ^= (residual & 0x0033) << 2;
            residual ^= 0x0020 & (residual << 1);
            residual ^= 0x0010 & (residual << 3);
            residual ^= 0x0002 & (residual << 1);
            residual ^= seed(left, top, 5, 10);
        } else if self.state == 2 {
            residual ^= u16::MAX;
        }
        self.update(i8::try_from(actual.count_ones()).unwrap_or(16));
        residual
    }

    fn predict_422(&mut self, residual: u16, left: Option<u16>, top: Option<u16>) -> u16 {
        let mut actual = residual;
        if self.state == 0 {
            actual ^= seed(left, top, 1, 6);
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

    /// Inverse of [`Self::predict_422`], undoing its steps in reverse order
    /// as [`Self::residual_444`] does.
    fn residual_422(&mut self, actual: u16, left: Option<u16>, top: Option<u16>) -> u16 {
        let mut residual = actual;
        if self.state == 0 {
            residual ^= (residual & 0x30) << 2;
            residual ^= (residual & 0x0c) << 2;
            residual ^= (residual & 0x03) << 2;
            residual ^= (residual & 0x01) << 1;
            residual ^= seed(left, top, 1, 6);
        } else if self.state == 2 {
            residual ^= 0x00ff;
        }
        self.update(i8::try_from(actual.count_ones() * 2).unwrap_or(16));
        residual
    }

    fn predict_420(&mut self, residual: u16, left: Option<u16>, top: Option<u16>) -> u16 {
        let mut actual = residual;
        if self.state == 0 {
            actual ^= seed(left, top, 1, 2);
            actual ^= 0x02 & (actual << 1);
            actual ^= (actual & 0x03) << 2;
        } else if self.state == 2 {
            actual ^= 0x000f;
        }
        self.update(i8::try_from(actual.count_ones() * 4).unwrap_or(16));
        actual
    }

    /// Inverse of [`Self::predict_420`], undoing its steps in reverse order
    /// as [`Self::residual_444`] does.
    fn residual_420(&mut self, actual: u16, left: Option<u16>, top: Option<u16>) -> u16 {
        let mut residual = actual;
        if self.state == 0 {
            residual ^= (residual & 0x03) << 2;
            residual ^= 0x02 & (residual << 1);
            residual ^= seed(left, top, 1, 2);
        } else if self.state == 2 {
            residual ^= 0x000f;
        }
        self.update(i8::try_from(actual.count_ones() * 4).unwrap_or(16));
        residual
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

/// The bit that predicts the first block of a pattern: the neighbouring
/// pattern's bit for the block beside it, `left_bit` of the pattern to the
/// left or else `top_bit` of the pattern above, and one in a tile's first
/// macroblock.
const fn seed(left: Option<u16>, top: Option<u16>, left_bit: u32, top_bit: u32) -> u16 {
    match (left, top) {
        (None, None) => 1,
        (None, Some(top)) => (top >> top_bit) & 1,
        (Some(left), _) => (left >> left_bit) & 1,
    }
}

fn refine(reader: &mut PacketBitReader<'_>, count: u8) -> Result<u16, TileDecodeError> {
    match count {
        0 => Ok(0),
        1 => Ok(1 << reader.read_bits(2)?),
        2 => Ok(u16::from(decode_prefix(reader, "REF_CBPHP1", &REF_TWO)?)),
        3 => Ok(0x0f ^ (1 << reader.read_bits(2)?)),
        4 => Ok(0x0f),
        _ => Err(TileDecodeError::InvalidPlan("CBPHP group count")),
    }
}

/// The `BLOCK_OFFSET` class and in-class increment coding a non-zero nibble.
///
/// The classes partition the codes by how many of the four blocks are set.
fn block_value(nibble: u16) -> (usize, u64) {
    let code = BLOCK_CODE[usize::from(nibble & 0x0f)];
    let value: usize = match code {
        1 => 5,
        2..=3 => 2,
        4..=7 => 1,
        8..=11 => 3,
        _ => 4,
    };
    let increment = usize::from(code).saturating_sub(BLOCK_OFFSET[value]);
    (value, increment as u64)
}

/// Inverse of [`refine`]: identifies which of the four groups are set.
fn write_refinement(writer: &mut BitWriter, groups: u8) -> Result<(), TileDecodeError> {
    match groups.count_ones() {
        0 | 4 => {}
        1 => writer.write_bits(u64::from(groups.trailing_zeros()), 2),
        2 => write_code(writer, &REF_TWO_WRITE, "REF_CBPHP1", groups)?,
        3 => writer.write_bits(u64::from((groups ^ 0x0f).trailing_zeros()), 2),
        _ => return Err(TileDecodeError::InvalidPlan("CBPHP group count")),
    }
    Ok(())
}

fn write_code(
    writer: &mut BitWriter,
    table: &CodeTable,
    syntax: &'static str,
    symbol: u8,
) -> Result<(), TileDecodeError> {
    Ok(encode_prefix(writer, syntax, table, symbol)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::encode::test_support::Lcg;

    #[test]
    fn cbphp_prefix_tables_match_serial_search() {
        use crate::entropy::assert_matches_serial;
        for codes in &NUM_CODE_LISTS {
            assert_matches_serial("NUM_CBPHP", codes);
        }
        for codes in &NUM_CODE_LISTS_YUV {
            assert_matches_serial("NUM_BLKCBPHP", codes);
        }
        assert_matches_serial("CHR_CBPHP", &CHROMA_CODE_LIST);
        assert_matches_serial("NUM_CH_BLK", &CHROMA_BLOCK_CODE_LIST);
        assert_matches_serial("REF_CBPHP1", &REF_TWO_CODE_LIST);
    }

    #[test]
    fn initial_zero_residual_predicts_normative_checker_pattern() {
        let mut prediction = PredictionState::new();
        assert_eq!(prediction.predict_444(0, None, None), 0xffff);
    }

    #[test]
    fn patterns_round_trip_with_matching_prediction_and_table_state() {
        let width = 37;
        let mut state = 0x1234_5678_9abc_def0_u64;
        let mut next = move || {
            state = Lcg::advance(state);
            state >> 33
        };
        for components in [1_usize, 3, 4, 8, 16] {
            // Sparse, dense, and mixed rows drive all three prediction states.
            let mut patterns = Vec::new();
            for y in 0..24 {
                for _ in 0..width * components {
                    let raw = (next() & 0xffff) as u16;
                    patterns.push(match y / 8 {
                        0 => raw & (next() & 0xffff) as u16 & (next() & 0xffff) as u16,
                        1 => raw | (next() & 0xffff) as u16 | (next() & 0xffff) as u16,
                        _ => raw,
                    });
                }
            }
            let mut writer = BitWriter::new();
            let mut write_state = CbphpState::new(width, components);
            let mut values = patterns.iter();
            for y in 0..24 {
                for x in 0..width {
                    for component in 0..components {
                        write_state
                            .encode_component(&mut writer, x, y, component, *values.next().unwrap())
                            .unwrap();
                    }
                    if x % 16 == 0 || x + 1 == width {
                        write_state.adapt();
                    }
                }
            }
            let bit_length = writer.bit_position();
            let bytes = writer.into_bytes();
            let mut reader = PacketBitReader::with_bit_length(&bytes, bit_length).unwrap();
            let mut read_state = CbphpState::new(width, components);
            let mut values = patterns.iter();
            for y in 0..24 {
                for x in 0..width {
                    for component in 0..components {
                        let decoded = read_state
                            .decode_component(&mut reader, x, y, component)
                            .unwrap();
                        assert_eq!(decoded, *values.next().unwrap());
                    }
                    if x % 16 == 0 || x + 1 == width {
                        read_state.adapt();
                    }
                }
            }
            assert_eq!(reader.bits_remaining(), 0);
            assert_eq!(read_state, write_state);
        }
    }

    #[test]
    fn joint_yuv_patterns_round_trip_with_matching_state() {
        let width = 23;
        let mut state = 0x0bad_cafe_dead_beef_u64;
        let mut next = move || {
            state = Lcg::advance(state);
            ((state >> 33) & 0xffff) as u16
        };
        // A chroma component has sixteen, eight, or four blocks.
        for (sampling, chroma_blocks) in [
            (ChromaSampling::Cs444, 0xffff_u16),
            (ChromaSampling::Cs422, 0x00ff),
            (ChromaSampling::Cs420, 0x000f),
        ] {
            // Sparse, dense, and mixed rows; within each, some components are
            // forced to zero so luma-only and chroma-only groups both occur.
            let mut patterns = Vec::new();
            for y in 0..30 {
                for _ in 0..width {
                    let mut actual = [0_u16; 3];
                    for (component, value) in actual.iter_mut().enumerate() {
                        *value = match y / 10 {
                            0 => next() & next() & next(),
                            1 => next() | next() | next(),
                            _ => next(),
                        };
                        if next() % 5 == 0 {
                            *value = 0;
                        }
                        if component != 0 {
                            *value &= chroma_blocks;
                        }
                    }
                    patterns.push(actual);
                }
            }
            let mut writer = BitWriter::new();
            let mut write_state = CbphpState::new(width, 3);
            for (index, &actual) in patterns.iter().enumerate() {
                let (x, y) = (index % width, index / width);
                write_state
                    .encode_yuv(&mut writer, x, y, sampling, actual)
                    .unwrap();
                if x % 16 == 0 || x + 1 == width {
                    write_state.adapt();
                }
            }
            let bit_length = writer.bit_position();
            let bytes = writer.into_bytes();
            let mut reader = PacketBitReader::with_bit_length(&bytes, bit_length).unwrap();
            let mut read_state = CbphpState::new(width, 3);
            for (index, &actual) in patterns.iter().enumerate() {
                let (x, y) = (index % width, index / width);
                let decoded = read_state.decode_yuv(&mut reader, x, y, sampling).unwrap();
                assert_eq!(decoded, actual, "{sampling:?} macroblock {index}");
                if x % 16 == 0 || x + 1 == width {
                    read_state.adapt();
                }
            }
            assert_eq!(reader.bits_remaining(), 0, "{sampling:?}");
            assert_eq!(read_state, write_state, "{sampling:?}");
        }
    }

    #[test]
    fn subsampled_residuals_invert_their_prediction_in_every_state() {
        // Dense, sparse, and mixed runs move the shared state through
        // prediction from neighbours, inversion, and no prediction.
        let mut state = 77_u64;
        let mut next = move || {
            state = Lcg::advance(state);
            ((state >> 33) & 0xffff) as u16
        };
        for blocks in [0x00ff_u16, 0x000f] {
            let mut forward = PredictionState::new();
            let mut inverse = PredictionState::new();
            let mut states = [false; 3];
            for round in 0..600 {
                let actual = match round / 100 % 3 {
                    0 => blocks,
                    1 => 0,
                    _ => next() & blocks,
                };
                let neighbours = [
                    (None, None),
                    (None, Some(next() & blocks)),
                    (Some(next() & blocks), Some(next() & blocks)),
                ][round % 3];
                states[usize::try_from(forward.state).unwrap()] = true;
                let (residual, predicted) = if blocks == 0x00ff {
                    let residual = inverse.residual_422(actual, neighbours.0, neighbours.1);
                    (
                        residual,
                        forward.predict_422(residual, neighbours.0, neighbours.1),
                    )
                } else {
                    let residual = inverse.residual_420(actual, neighbours.0, neighbours.1);
                    (
                        residual,
                        forward.predict_420(residual, neighbours.0, neighbours.1),
                    )
                };
                assert_eq!(predicted, actual, "blocks {blocks:#x} round {round}");
                assert_eq!(residual & !blocks, 0, "blocks {blocks:#x} round {round}");
                assert_eq!(forward, inverse);
            }
            assert_eq!(states, [true; 3], "blocks {blocks:#x}");
        }
    }

    #[test]
    fn a_422_group_covers_one_block_and_the_block_below_it() {
        // Chroma blocks lie two across: a group's blocks are two bits apart.
        for (group, first) in GROUP_SHIFT_422.into_iter().enumerate() {
            for blocks in 1..=3_u8 {
                let pattern = spread_422(blocks) << first;
                assert_eq!(gather_422(pattern >> first), u16::from(blocks));
                let expected =
                    u16::from(blocks & 1) << first | u16::from(blocks >> 1) << (first + 2);
                assert_eq!(pattern, expected, "group {group}");
            }
        }
        // The four groups partition the eight blocks.
        let all = GROUP_SHIFT_422
            .into_iter()
            .fold(0_u16, |all, first| all | spread_422(3) << first);
        assert_eq!(all, 0x00ff);
    }

    #[test]
    fn refine_two_uses_table_64() {
        let mut reader = PacketBitReader::with_bit_length(&[0b1010_0000], 3).unwrap();
        assert_eq!(refine(&mut reader, 2).unwrap(), 9);
    }
}
