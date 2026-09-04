//! Narrow typed reads for T.832 header syntax fields.

use crate::{NativeError, bit_reader::BitReader};

pub(super) fn read_dimension(
    reader: &mut BitReader<'_>,
    bits: u8,
    operation: &'static str,
) -> Result<u32, NativeError> {
    let minus_one = u32::try_from(reader.read_bits(bits)?)
        .map_err(|_| NativeError::IntegerOverflow { operation })?;
    minus_one
        .checked_add(1)
        .ok_or(NativeError::IntegerOverflow { operation })
}

pub(super) fn require_value(
    value: u64,
    expected: u64,
    field: &'static str,
) -> Result<(), NativeError> {
    if value == expected {
        Ok(())
    } else {
        Err(NativeError::ReservedValue { field, value })
    }
}

pub(super) fn read_u8(reader: &mut BitReader<'_>, bits: u8) -> Result<u8, NativeError> {
    u8::try_from(reader.read_bits(bits)?).map_err(|_| NativeError::IntegerOverflow {
        operation: "converting parsed field to u8",
    })
}

pub(super) fn read_u16(reader: &mut BitReader<'_>, bits: u8) -> Result<u16, NativeError> {
    u16::try_from(reader.read_bits(bits)?).map_err(|_| NativeError::IntegerOverflow {
        operation: "converting parsed field to u16",
    })
}
