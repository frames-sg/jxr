//! Packet-scoped, most-significant-bit-first reading.

use super::EntropyError;

/// A bounds-checked view over exactly one tile packet payload.
#[derive(Debug, Clone)]
pub struct PacketBitReader<'a> {
    bytes: &'a [u8],
    bit_position: usize,
    bit_length: usize,
}

impl<'a> PacketBitReader<'a> {
    /// Creates a reader covering every bit in `bytes`.
    #[must_use]
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            bit_position: 0,
            bit_length: bytes.len().saturating_mul(8),
        }
    }

    /// Creates a reader bounded to `bit_length`, which may end mid-byte.
    pub fn with_bit_length(bytes: &'a [u8], bit_length: usize) -> Result<Self, EntropyError> {
        let available_bits = bytes.len().saturating_mul(8);
        if bit_length > available_bits {
            return Err(EntropyError::InvalidBitLength {
                bit_length,
                available_bits,
            });
        }
        Ok(Self {
            bytes,
            bit_position: 0,
            bit_length,
        })
    }

    /// Returns the current offset from the start of the packet payload.
    #[must_use]
    pub const fn bit_position(&self) -> usize {
        self.bit_position
    }

    /// Returns the number of unread bits in this packet view.
    #[must_use]
    pub const fn bits_remaining(&self) -> usize {
        self.bit_length - self.bit_position
    }

    /// Reads one bit.
    pub fn read_bit(&mut self) -> Result<bool, EntropyError> {
        Ok(self.read_bits(1)? != 0)
    }

    /// Reads up to 64 bits, most-significant bit first.
    pub fn read_bits(&mut self, count: u8) -> Result<u64, EntropyError> {
        if count > 64 {
            return Err(EntropyError::InvalidParameter {
                parameter: "bit count",
                value: i64::from(count),
            });
        }
        let Some(end) = self.bit_position.checked_add(usize::from(count)) else {
            return Err(EntropyError::UnexpectedEnd {
                bit_position: self.bit_position,
                requested_bits: count,
                bit_length: self.bit_length,
            });
        };
        if end > self.bit_length {
            return Err(EntropyError::UnexpectedEnd {
                bit_position: self.bit_position,
                requested_bits: count,
                bit_length: self.bit_length,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_across_bytes_and_honours_mid_byte_limit() {
        let mut reader = PacketBitReader::with_bit_length(&[0b1011_0010, 0b0110_0000], 11).unwrap();
        assert_eq!(reader.read_bits(3).unwrap(), 0b101);
        assert_eq!(reader.read_bits(7).unwrap(), 0b100_1001);
        assert!(reader.read_bit().unwrap());
        assert_eq!(reader.bits_remaining(), 0);
        assert!(matches!(
            reader.read_bit(),
            Err(EntropyError::UnexpectedEnd {
                bit_position: 11,
                ..
            })
        ));
    }

    #[test]
    fn rejects_limit_beyond_backing_slice() {
        assert_eq!(
            PacketBitReader::with_bit_length(&[0], 9).unwrap_err(),
            EntropyError::InvalidBitLength {
                bit_length: 9,
                available_bits: 8,
            }
        );
    }

    #[test]
    fn zero_width_read_does_not_advance() {
        let mut reader = PacketBitReader::new(&[0xff]);
        assert_eq!(reader.read_bits(0).unwrap(), 0);
        assert_eq!(reader.bit_position(), 0);
    }
}
