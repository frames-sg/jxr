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
    #[inline]
    pub fn read_bit(&mut self) -> Result<bool, EntropyError> {
        Ok(self.read_bits(1)? != 0)
    }

    /// Reads up to 64 bits, most-significant bit first.
    #[inline]
    pub fn read_bits(&mut self, count: u8) -> Result<u64, EntropyError> {
        // A byte-aligned 64-bit window always holds 57 bits after the in-byte
        // offset, so every entropy-syntax read needs exactly one load.
        if count > 57 {
            return self.read_wide_bits(count);
        }
        let end = self.checked_end(count)?;
        let value = match count {
            0 => 0,
            _ => self.window() >> (64 - count),
        };
        self.bit_position = end;
        Ok(value)
    }

    /// Returns the next eight bits without consuming them, zero-filled past the packet end.
    #[inline]
    pub(crate) fn peek_byte(&self) -> u8 {
        let remaining = self.bits_remaining();
        let mut window = self.window();
        if remaining < 8 {
            window &= !(u64::MAX >> remaining);
        }
        window.to_be_bytes()[0]
    }

    /// Advances past bits already inspected with [`Self::peek_byte`].
    #[inline]
    pub(crate) fn skip_bits(&mut self, count: u8) -> Result<(), EntropyError> {
        self.bit_position = self.checked_end(count)?;
        Ok(())
    }

    /// Builds the error a bit-serial reader reports after consuming `consumed` more bits.
    #[cold]
    pub(crate) fn unexpected_end_after(&self, consumed: usize) -> EntropyError {
        EntropyError::UnexpectedEnd {
            bit_position: self.bit_position.saturating_add(consumed),
            requested_bits: 1,
            bit_length: self.bit_length,
        }
    }

    #[inline(never)]
    fn read_wide_bits(&mut self, count: u8) -> Result<u64, EntropyError> {
        if count > 64 {
            return Err(EntropyError::InvalidParameter {
                parameter: "bit count",
                value: i64::from(count),
            });
        }
        let end = self.checked_end(count)?;
        let high_bits = count - 32;
        let high = self.window() >> (64 - high_bits);
        self.bit_position += usize::from(high_bits);
        let low = self.window() >> 32;
        self.bit_position = end;
        Ok((high << 32) | low)
    }

    #[inline]
    fn checked_end(&self, count: u8) -> Result<usize, EntropyError> {
        match self.bit_position.checked_add(usize::from(count)) {
            Some(end) if end <= self.bit_length => Ok(end),
            _ => Err(self.end_error(count)),
        }
    }

    #[cold]
    #[inline(never)]
    fn end_error(&self, count: u8) -> EntropyError {
        EntropyError::UnexpectedEnd {
            bit_position: self.bit_position,
            requested_bits: count,
            bit_length: self.bit_length,
        }
    }

    /// Returns 64 bits starting at `bit_position`, most-significant-bit aligned.
    ///
    /// Bytes beyond the backing slice read as zero. Bits between the packet
    /// bit length and the end of its final byte are returned unchanged, so
    /// callers must bound their use by `bits_remaining`.
    #[inline]
    fn window(&self) -> u64 {
        let byte = self.bit_position / 8;
        let word = match self.bytes.get(byte..byte + 8) {
            Some(chunk) => {
                u64::from_be_bytes(chunk.try_into().expect("slice is exactly eight bytes"))
            }
            None => Self::padded_word(self.bytes, byte),
        };
        word << (self.bit_position % 8)
    }

    #[cold]
    #[inline(never)]
    fn padded_word(bytes: &[u8], byte: usize) -> u64 {
        let mut padded = [0_u8; 8];
        let tail = bytes.get(byte..).unwrap_or_default();
        padded[..tail.len()].copy_from_slice(tail);
        u64::from_be_bytes(padded)
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

    /// Bit-serial reference for the word-at-a-time reader.
    fn reference_bits(bytes: &[u8], start: usize, count: usize) -> u64 {
        (start..start + count).fold(0, |value, position| {
            (value << 1) | u64::from((bytes[position / 8] >> (7 - position % 8)) & 1)
        })
    }

    #[test]
    fn every_width_and_offset_matches_bit_serial_reads() {
        let bytes: Vec<u8> = (0_u8..24)
            .map(|index| index.wrapping_mul(157) ^ 0xa5)
            .collect();
        let bit_length = bytes.len() * 8 - 3;
        for start in 0..bit_length {
            for count in 0..=64_u8 {
                let mut reader = PacketBitReader::with_bit_length(&bytes, bit_length).unwrap();
                if start > 0 {
                    reader.read_bits(u8::try_from(start % 64).unwrap()).unwrap();
                    for _ in 0..start / 64 {
                        reader.read_bits(64).unwrap();
                    }
                }
                assert_eq!(reader.bit_position(), start);
                let result = reader.read_bits(count);
                if start + usize::from(count) > bit_length {
                    assert_eq!(
                        result,
                        Err(EntropyError::UnexpectedEnd {
                            bit_position: start,
                            requested_bits: count,
                            bit_length,
                        })
                    );
                    assert_eq!(reader.bit_position(), start);
                } else {
                    assert_eq!(
                        result.unwrap(),
                        reference_bits(&bytes, start, usize::from(count)),
                        "start {start}, count {count}"
                    );
                    assert_eq!(reader.bit_position(), start + usize::from(count));
                }
            }
        }
    }

    #[test]
    fn peek_zero_fills_past_the_bounded_packet() {
        let mut reader = PacketBitReader::with_bit_length(&[0xff, 0xff], 11).unwrap();
        assert_eq!(reader.peek_byte(), 0xff);
        reader.read_bits(5).unwrap();
        assert_eq!(reader.peek_byte(), 0b1111_1100);
        reader.read_bits(6).unwrap();
        assert_eq!(reader.peek_byte(), 0);
        assert_eq!(reader.bit_position(), 11);
    }
}
