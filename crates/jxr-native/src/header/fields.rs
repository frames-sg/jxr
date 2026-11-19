//! Narrow typed reads for T.832 header syntax fields.

use jxr_core::OutputBitDepth;

use crate::{NativeError, bit_reader::BitReader};

/// The bit depth an `OUTPUT_BITDEPTH` code names, or `None` for a reserved
/// code: the plane-header fields that qualify it are left at zero.
pub(crate) const fn output_bit_depth(code: u8) -> Option<OutputBitDepth> {
    OutputBitDepth::from_header_fields(code, 0, 0, 0)
}

/// Whether `OUTPUT_BITDEPTH` code `code` is reserved.
pub(super) const fn reserved_bit_depth(code: u8) -> bool {
    output_bit_depth(code).is_none()
}

/// Whether each plane header of an image of `OUTPUT_BITDEPTH` code `code`
/// carries `SHIFT_BITS`.
pub(super) const fn carries_shift_bits(code: u8) -> bool {
    matches!(
        output_bit_depth(code),
        Some(OutputBitDepth::U16 { .. } | OutputBitDepth::I16 { .. } | OutputBitDepth::I32 { .. })
    )
}

/// Whether each plane header of an image of `OUTPUT_BITDEPTH` code `code`
/// carries the floating-point mantissa length and exponent bias.
pub(super) const fn carries_float_parameters(code: u8) -> bool {
    matches!(output_bit_depth(code), Some(OutputBitDepth::F32 { .. }))
}

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
