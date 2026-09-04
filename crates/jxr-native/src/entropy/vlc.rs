//! Normative prefix-code tables from T.832 Tables 52 and 76 through 82.

use super::{EntropyError, PacketBitReader};

#[derive(Clone, Copy)]
pub(super) struct VlcCode {
    bits: u16,
    length: u8,
    symbol: u8,
}

const fn c(bits: u16, length: u8, symbol: u8) -> VlcCode {
    VlcCode {
        bits,
        length,
        symbol,
    }
}

pub(super) fn decode(
    reader: &mut PacketBitReader<'_>,
    syntax: &'static str,
    codes: &[VlcCode],
) -> Result<u8, EntropyError> {
    let start = reader.bit_position();
    let max_length = codes.iter().map(|code| code.length).max().unwrap_or(0);
    let mut prefix = 0_u16;
    for length in 1..=max_length {
        prefix = (prefix << 1) | u16::from(reader.read_bit()?);
        if let Some(code) = codes
            .iter()
            .find(|code| code.length == length && code.bits == prefix)
        {
            return Ok(code.symbol);
        }
    }
    Err(EntropyError::InvalidVlc {
        syntax,
        bit_position: start,
    })
}

pub(super) const ABS_LEVEL: [&[VlcCode]; 2] = [
    &[
        c(0b01, 2, 0),
        c(0b10, 2, 1),
        c(0b11, 2, 2),
        c(0b001, 3, 3),
        c(0b0001, 4, 4),
        c(0b00000, 5, 5),
        c(0b00001, 5, 6),
    ],
    &[
        c(0b1, 1, 0),
        c(0b01, 2, 1),
        c(0b001, 3, 2),
        c(0b0001, 4, 3),
        c(0b00001, 5, 4),
        c(0b00_0000, 6, 5),
        c(0b00_0001, 6, 6),
    ],
];

pub(super) const RUN_VALUE_2: &[VlcCode] = &[c(0b1, 1, 1), c(0b0, 1, 2)];
pub(super) const RUN_VALUE_3: &[VlcCode] = &[c(0b1, 1, 1), c(0b01, 2, 2), c(0b00, 2, 3)];
pub(super) const RUN_VALUE_4: &[VlcCode] =
    &[c(0b1, 1, 1), c(0b01, 2, 2), c(0b001, 3, 3), c(0b000, 3, 4)];
pub(super) const RUN_INDEX: &[VlcCode] = &[
    c(0b1, 1, 0),
    c(0b01, 2, 1),
    c(0b001, 3, 2),
    c(0b0000, 4, 3),
    c(0b0001, 4, 4),
];

pub(super) const INDEX_A: [&[VlcCode]; 4] = [
    &[
        c(0b1, 1, 0),
        c(0b00000, 5, 1),
        c(0b001, 3, 2),
        c(0b00001, 5, 3),
        c(0b01, 2, 4),
        c(0b0001, 4, 5),
    ],
    &[
        c(0b01, 2, 0),
        c(0b0000, 4, 1),
        c(0b10, 2, 2),
        c(0b0001, 4, 3),
        c(0b11, 2, 4),
        c(0b001, 3, 5),
    ],
    &[
        c(0b0000, 4, 0),
        c(0b0001, 4, 1),
        c(0b01, 2, 2),
        c(0b10, 2, 3),
        c(0b11, 2, 4),
        c(0b001, 3, 5),
    ],
    &[
        c(0b00000, 5, 0),
        c(0b00001, 5, 1),
        c(0b01, 2, 2),
        c(0b1, 1, 3),
        c(0b0001, 4, 4),
        c(0b001, 3, 5),
    ],
];

pub(super) const INDEX_B: &[VlcCode] =
    &[c(0b0, 1, 0), c(0b10, 2, 2), c(0b110, 3, 1), c(0b111, 3, 3)];

pub(super) const FIRST_INDEX: [&[VlcCode]; 5] = [
    &[
        c(0b00001, 5, 0),
        c(0b00_0001, 6, 1),
        c(0b000_0000, 7, 2),
        c(0b000_0001, 7, 3),
        c(0b00100, 5, 4),
        c(0b010, 3, 5),
        c(0b00101, 5, 6),
        c(0b1, 1, 7),
        c(0b00110, 5, 8),
        c(0b0001, 4, 9),
        c(0b00111, 5, 10),
        c(0b011, 3, 11),
    ],
    &[
        c(0b0010, 4, 0),
        c(0b00010, 5, 1),
        c(0b00_0000, 6, 2),
        c(0b00_0001, 6, 3),
        c(0b0011, 4, 4),
        c(0b010, 3, 5),
        c(0b00011, 5, 6),
        c(0b11, 2, 7),
        c(0b011, 3, 8),
        c(0b100, 3, 9),
        c(0b00001, 5, 10),
        c(0b101, 3, 11),
    ],
    &[
        c(0b11, 2, 0),
        c(0b001, 3, 1),
        c(0b000_0000, 7, 2),
        c(0b000_0001, 7, 3),
        c(0b00001, 5, 4),
        c(0b010, 3, 5),
        c(0b000_0010, 7, 6),
        c(0b011, 3, 7),
        c(0b100, 3, 8),
        c(0b101, 3, 9),
        c(0b000_0011, 7, 10),
        c(0b0001, 4, 11),
    ],
    &[
        c(0b001, 3, 0),
        c(0b11, 2, 1),
        c(0b000_0000, 7, 2),
        c(0b00001, 5, 3),
        c(0b00010, 5, 4),
        c(0b010, 3, 5),
        c(0b000_0001, 7, 6),
        c(0b011, 3, 7),
        c(0b00011, 5, 8),
        c(0b100, 3, 9),
        c(0b00_0001, 6, 10),
        c(0b101, 3, 11),
    ],
    &[
        c(0b010, 3, 0),
        c(0b1, 1, 1),
        c(0b000_0001, 7, 2),
        c(0b0001, 4, 3),
        c(0b000_0010, 7, 4),
        c(0b011, 3, 5),
        c(0b0000_0000, 8, 6),
        c(0b0010, 4, 7),
        c(0b000_0011, 7, 8),
        c(0b0011, 4, 9),
        c(0b0000_0001, 8, 10),
        c(0b00001, 5, 11),
    ],
];

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_decodes_table(syntax: &'static str, table: &[VlcCode]) {
        for code in table {
            let shifted = code.bits << (8 - code.length);
            let bytes = [u8::try_from(shifted).unwrap()];
            let mut reader =
                PacketBitReader::with_bit_length(&bytes, usize::from(code.length)).unwrap();
            assert_eq!(decode(&mut reader, syntax, table).unwrap(), code.symbol);
            assert_eq!(reader.bits_remaining(), 0);
        }
    }

    #[test]
    fn decodes_every_normative_vlc_entry() {
        for table in ABS_LEVEL {
            assert_decodes_table("ABS_LEVEL_INDEX", table);
        }
        for table in INDEX_A {
            assert_decodes_table("INDEX_A", table);
        }
        for table in FIRST_INDEX {
            assert_decodes_table("FIRST_INDEX", table);
        }
        assert_decodes_table("RUN_VALUE", RUN_VALUE_2);
        assert_decodes_table("RUN_VALUE", RUN_VALUE_3);
        assert_decodes_table("RUN_VALUE", RUN_VALUE_4);
        assert_decodes_table("RUN_INDEX", RUN_INDEX);
        assert_decodes_table("INDEX_B", INDEX_B);
    }

    #[test]
    fn decodes_first_index_examples_from_table_82() {
        let cases: [(usize, u16, u8, u8); 5] = [
            (0, 0b1, 1, 7),
            (1, 0b0010, 4, 0),
            (2, 0b11, 2, 0),
            (3, 0b11, 2, 1),
            (4, 0b1, 1, 1),
        ];
        for (table, bits, length, expected) in cases {
            let bytes = [u8::try_from(bits << (8 - length)).unwrap()];
            let mut reader = PacketBitReader::with_bit_length(&bytes, usize::from(length)).unwrap();
            assert_eq!(
                decode(&mut reader, "FIRST_INDEX", FIRST_INDEX[table]).unwrap(),
                expected
            );
        }
    }
}
