//! Normative prefix-code tables from T.832 Tables 52 and 76 through 82.

use super::{EntropyError, PacketBitReader};

/// One normative prefix code: `(code bits, code length, symbol)`.
pub(crate) type VlcCode = (u16, u8, u8);

/// Longest code in any T.832 prefix table used by the tile syntax.
const LOOKUP_BITS: u8 = 8;

/// A prefix-code table resolved through one eight-bit lookup.
///
/// Each entry stores `(length << 8) | symbol`; zero marks a prefix that no
/// code matches. Construction is `const`, so an ambiguous or oversized table
/// fails compilation instead of decoding.
pub(crate) struct PrefixTable {
    max_length: u8,
    lookup: [u16; 1 << LOOKUP_BITS],
}

impl PrefixTable {
    pub(crate) const fn new(codes: &[VlcCode]) -> Self {
        let mut lookup = [0_u16; 1 << LOOKUP_BITS];
        let mut max_length = 0;
        let mut index = 0;
        while index < codes.len() {
            let (bits, length, symbol) = codes[index];
            assert!(length > 0 && length <= LOOKUP_BITS, "prefix code length");
            assert!(bits >> length == 0, "prefix code bits exceed length");
            if length > max_length {
                max_length = length;
            }
            let span = 1_usize << (LOOKUP_BITS - length);
            let first = (bits as usize) << (LOOKUP_BITS - length);
            let mut slot = first;
            while slot < first + span {
                assert!(lookup[slot] == 0, "prefix table is not prefix-free");
                lookup[slot] = ((length as u16) << 8) | symbol as u16;
                slot += 1;
            }
            index += 1;
        }
        Self { max_length, lookup }
    }
}

/// Decodes one prefix-coded symbol with bit-serial error semantics.
///
/// A truncated packet reports `UnexpectedEnd` at the bit where a serial
/// decoder would stop, and an unmatched full-length prefix reports
/// `InvalidVlc` at the code start. Neither error consumes input.
pub(crate) fn decode(
    reader: &mut PacketBitReader<'_>,
    syntax: &'static str,
    table: &PrefixTable,
) -> Result<u8, EntropyError> {
    let remaining = reader.bits_remaining();
    let entry = table.lookup[usize::from(reader.peek_byte())];
    let [length, symbol] = entry.to_be_bytes();
    if entry != 0 && usize::from(length) <= remaining {
        reader.skip_bits(length)?;
        return Ok(symbol);
    }
    if remaining < usize::from(table.max_length) {
        return Err(reader.unexpected_end_after(remaining));
    }
    Err(EntropyError::InvalidVlc {
        syntax,
        bit_position: reader.bit_position(),
    })
}

const fn c(bits: u16, length: u8, symbol: u8) -> VlcCode {
    (bits, length, symbol)
}

const ABS_LEVEL_CODES: [&[VlcCode]; 2] = [
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

const RUN_VALUE_2_CODES: &[VlcCode] = &[c(0b1, 1, 1), c(0b0, 1, 2)];
const RUN_VALUE_3_CODES: &[VlcCode] = &[c(0b1, 1, 1), c(0b01, 2, 2), c(0b00, 2, 3)];
const RUN_VALUE_4_CODES: &[VlcCode] =
    &[c(0b1, 1, 1), c(0b01, 2, 2), c(0b001, 3, 3), c(0b000, 3, 4)];
const RUN_INDEX_CODES: &[VlcCode] = &[
    c(0b1, 1, 0),
    c(0b01, 2, 1),
    c(0b001, 3, 2),
    c(0b0000, 4, 3),
    c(0b0001, 4, 4),
];

const INDEX_A_CODES: [&[VlcCode]; 4] = [
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

const INDEX_B_CODES: &[VlcCode] = &[c(0b0, 1, 0), c(0b10, 2, 2), c(0b110, 3, 1), c(0b111, 3, 3)];

const FIRST_INDEX_CODES: [&[VlcCode]; 5] = [
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

pub(super) static ABS_LEVEL: [PrefixTable; 2] = [
    PrefixTable::new(ABS_LEVEL_CODES[0]),
    PrefixTable::new(ABS_LEVEL_CODES[1]),
];
pub(super) static RUN_VALUE_2: PrefixTable = PrefixTable::new(RUN_VALUE_2_CODES);
pub(super) static RUN_VALUE_3: PrefixTable = PrefixTable::new(RUN_VALUE_3_CODES);
pub(super) static RUN_VALUE_4: PrefixTable = PrefixTable::new(RUN_VALUE_4_CODES);
pub(super) static RUN_INDEX: PrefixTable = PrefixTable::new(RUN_INDEX_CODES);
pub(super) static INDEX_A: [PrefixTable; 4] = [
    PrefixTable::new(INDEX_A_CODES[0]),
    PrefixTable::new(INDEX_A_CODES[1]),
    PrefixTable::new(INDEX_A_CODES[2]),
    PrefixTable::new(INDEX_A_CODES[3]),
];
pub(super) static INDEX_B: PrefixTable = PrefixTable::new(INDEX_B_CODES);
pub(super) static FIRST_INDEX: [PrefixTable; 5] = [
    PrefixTable::new(FIRST_INDEX_CODES[0]),
    PrefixTable::new(FIRST_INDEX_CODES[1]),
    PrefixTable::new(FIRST_INDEX_CODES[2]),
    PrefixTable::new(FIRST_INDEX_CODES[3]),
    PrefixTable::new(FIRST_INDEX_CODES[4]),
];

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The normative bit-serial search that the lookup tables replace.
    pub(crate) fn serial_decode(
        reader: &mut PacketBitReader<'_>,
        syntax: &'static str,
        codes: &[VlcCode],
    ) -> Result<u8, EntropyError> {
        let start = reader.bit_position();
        let max_length = codes.iter().map(|code| code.1).max().unwrap_or(0);
        let mut prefix = 0_u16;
        for length in 1..=max_length {
            prefix = (prefix << 1) | u16::from(reader.read_bit()?);
            if let Some(code) = codes
                .iter()
                .find(|code| code.1 == length && code.0 == prefix)
            {
                return Ok(code.2);
            }
        }
        Err(EntropyError::InvalidVlc {
            syntax,
            bit_position: start,
        })
    }

    /// Compares table and serial decoding for every 16-bit input, offset, and truncation.
    pub(crate) fn assert_matches_serial(syntax: &'static str, codes: &[VlcCode]) {
        let table = PrefixTable::new(codes);
        for pattern in 0_u16..=u16::MAX {
            let bytes = pattern.to_be_bytes();
            for offset in [0_usize, 3] {
                for bit_length in offset..=16 {
                    let mut serial = PacketBitReader::with_bit_length(&bytes, bit_length).unwrap();
                    let mut lookup = serial.clone();
                    if offset > 0 {
                        serial.read_bits(3).unwrap();
                        lookup.read_bits(3).unwrap();
                    }
                    let expected = serial_decode(&mut serial, syntax, codes);
                    let actual = decode(&mut lookup, syntax, &table);
                    assert_eq!(
                        actual, expected,
                        "{syntax} {pattern:016b} @{offset}/{bit_length}"
                    );
                    if expected.is_ok() {
                        assert_eq!(lookup.bit_position(), serial.bit_position());
                    }
                }
            }
        }
    }

    #[test]
    fn lookup_tables_match_serial_prefix_search() {
        for codes in ABS_LEVEL_CODES {
            assert_matches_serial("ABS_LEVEL_INDEX", codes);
        }
        for codes in INDEX_A_CODES {
            assert_matches_serial("INDEX_A", codes);
        }
        for codes in FIRST_INDEX_CODES {
            assert_matches_serial("FIRST_INDEX", codes);
        }
        assert_matches_serial("RUN_VALUE", RUN_VALUE_2_CODES);
        assert_matches_serial("RUN_VALUE", RUN_VALUE_3_CODES);
        assert_matches_serial("RUN_VALUE", RUN_VALUE_4_CODES);
        assert_matches_serial("RUN_INDEX", RUN_INDEX_CODES);
        assert_matches_serial("INDEX_B", INDEX_B_CODES);
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
                decode(&mut reader, "FIRST_INDEX", &FIRST_INDEX[table]).unwrap(),
                expected
            );
            assert_eq!(reader.bits_remaining(), 0);
        }
    }
}
