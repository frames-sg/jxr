//! Bounds-checked JPEG XR bit reading.

use crate::NativeError;

/// Most-significant-bit-first reader used by T.832 syntax structures.
pub(crate) struct BitReader<'a> {
    bytes: &'a [u8],
    bit_position: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) const fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            bit_position: 0,
        }
    }

    pub(crate) const fn bit_position(&self) -> usize {
        self.bit_position
    }

    pub(crate) fn read_flag(&mut self) -> Result<bool, NativeError> {
        self.read_bits(1).map(|value| value != 0)
    }

    pub(crate) fn read_bits(&mut self, count: u8) -> Result<u64, NativeError> {
        if count > 64 {
            return Err(NativeError::ReservedValue {
                field: "bit read width",
                value: u64::from(count),
            });
        }
        let end = self.bit_position.checked_add(usize::from(count)).ok_or(
            NativeError::IntegerOverflow {
                operation: "advancing bit reader",
            },
        )?;
        if end > self.bytes.len().saturating_mul(8) {
            return Err(NativeError::Truncated {
                bit_position: self.bit_position,
                requested_bits: count,
            });
        }

        let mut value = 0_u64;
        while self.bit_position < end {
            let byte = self.bytes[self.bit_position / 8];
            let shift = 7 - (self.bit_position % 8);
            value = (value << 1) | u64::from((byte >> shift) & 1);
            self.bit_position += 1;
        }
        Ok(value)
    }

    pub(crate) fn align_zero(&mut self) -> Result<(), NativeError> {
        while !self.bit_position.is_multiple_of(8) {
            let position = self.bit_position;
            if self.read_flag()? {
                return Err(NativeError::NonZeroAlignmentBit {
                    bit_position: position,
                });
            }
        }
        Ok(())
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
