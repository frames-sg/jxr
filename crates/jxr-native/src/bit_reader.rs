//! Header bit reading: the tile packets' reader, [`PacketBitReader`],
//! reporting its errors as a header's.

use crate::{
    NativeError,
    entropy::{EntropyError, PacketBitReader},
};

/// Most-significant-bit-first reader used by T.832 syntax structures: the
/// packet reader of the tile decoder, reporting header errors.
pub(crate) struct BitReader<'a>(PacketBitReader<'a>);

impl<'a> BitReader<'a> {
    pub(crate) const fn new(bytes: &'a [u8]) -> Self {
        Self(PacketBitReader::new(bytes))
    }

    pub(crate) const fn bit_position(&self) -> usize {
        self.0.bit_position()
    }

    pub(crate) fn read_flag(&mut self) -> Result<bool, NativeError> {
        self.read_bits(1).map(|value| value != 0)
    }

    pub(crate) fn read_bits(&mut self, count: u8) -> Result<u64, NativeError> {
        self.0
            .read_bits(count)
            .map_err(|error| header_error(&error))
    }

    /// The packet reader underneath, for syntax that a header and a tile
    /// packet share, whose errors [`header_error`] reports as a header's.
    pub(crate) const fn packet(&mut self) -> &mut PacketBitReader<'a> {
        &mut self.0
    }

    pub(crate) fn align_zero(&mut self) -> Result<(), NativeError> {
        while !self.bit_position().is_multiple_of(8) {
            let position = self.bit_position();
            if self.read_flag()? {
                return Err(NativeError::NonZeroAlignmentBit {
                    bit_position: position,
                });
            }
        }
        Ok(())
    }
}

/// A read error of the packet reader, as a header reports it.
pub(crate) fn header_error(error: &EntropyError) -> NativeError {
    match *error {
        EntropyError::UnexpectedEnd {
            bit_position,
            requested_bits,
            ..
        } => NativeError::Truncated {
            bit_position,
            requested_bits,
        },
        // The one other error a read makes: a width above 64 bits.
        EntropyError::InvalidParameter { value, .. } => NativeError::ReservedValue {
            field: "bit read width",
            value: value.unsigned_abs(),
        },
        // Not an error a read makes; reported rather than assumed away.
        _ => NativeError::InvalidSyntax {
            field: "header bit read",
        },
    }
}

#[cfg(test)]
mod tests {
    use super::BitReader;

    #[test]
    fn reads_across_byte_boundaries_msb_first() {
        let mut reader = BitReader::new(&[0b1011_0010, 0b0110_0000]);
        assert_eq!(reader.read_bits(3).unwrap(), 0b101);
        assert_eq!(reader.read_bits(7).unwrap(), 0b100_1001);
        assert_eq!(reader.read_bits(2).unwrap(), 0b10);
    }
}
