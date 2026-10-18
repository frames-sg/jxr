//! Packet-scoped, most-significant-bit-first reading.

use super::EntropyError;

/// A bounds-checked view over exactly one tile packet payload.
///
/// Reads are served from a cached, most-significant-bit-aligned window of the
/// unread bits, so consecutive syntax elements shift a register instead of
/// reloading the packet. The cache never extends past `bit_length` and its
/// bits after `cached` are zero, which makes every end check one comparison.
#[derive(Debug, Clone)]
pub struct PacketBitReader<'a> {
    bytes: &'a [u8],
    bit_position: usize,
    bit_length: usize,
    cache: u64,
    cached: u32,
}

impl<'a> PacketBitReader<'a> {
    /// Creates a reader covering every bit in `bytes`.
    #[must_use]
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            bit_position: 0,
            bit_length: bytes.len().saturating_mul(8),
            cache: 0,
            cached: 0,
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
            cache: 0,
            cached: 0,
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
        // A refilled cache holds at least 57 bits unless the packet ends first.
        if count > 57 {
            return self.read_wide_bits(count);
        }
        if u32::from(count) > self.cached {
            self.refill();
            if u32::from(count) > self.cached {
                return Err(self.end_error(count));
            }
        }
        Ok(self.consume_cached(count))
    }

    /// Returns the next eight bits without consuming them, zero-filled past the packet end.
    ///
    /// Afterwards at least `min(8, bits_remaining)` bits are cached.
    #[inline]
    pub(crate) fn peek_byte(&mut self) -> u8 {
        if self.cached < 8 {
            self.refill();
        }
        self.cache.to_be_bytes()[0]
    }

    /// Advances past `count` cached bits when they are available.
    ///
    /// Returns `false` without consuming input when fewer bits are cached.
    #[inline]
    pub(crate) fn skip_cached(&mut self, count: u8) -> bool {
        if u32::from(count) > self.cached {
            return false;
        }
        self.consume_cached(count);
        true
    }

    /// Reads a `bits`-wide magnitude followed, when `signed` and non-zero, by its sign bit.
    ///
    /// Returns the magnitude and whether the sign bit was set. Returns `None`
    /// without consuming input when fewer than `bits + 1` bits can be cached,
    /// so callers near the packet end can repeat the read bit-serially.
    #[inline]
    pub(crate) fn read_magnitude_and_sign(
        &mut self,
        bits: u8,
        signed: bool,
    ) -> Option<(u32, bool)> {
        debug_assert!(bits <= 29);
        let needed = u32::from(bits) + 1;
        if needed > self.cached {
            self.refill();
            if needed > self.cached {
                return None;
            }
        }
        let magnitude = (self.cache >> 1) >> (63 - bits);
        // The sign follows only a non-zero magnitude; selecting it without a
        // branch avoids mispredicting on data-dependent flexbits.
        let take_sign = signed && magnitude != 0;
        let negative = take_sign && (self.cache << bits) >> 63 != 0;
        let consumed = bits + u8::from(take_sign);
        self.cache <<= consumed;
        self.cached -= u32::from(consumed);
        self.bit_position += usize::from(consumed);
        Some((
            u32::try_from(magnitude).expect("magnitude has at most 29 bits"),
            negative,
        ))
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

    /// Consumes `count <= cached` bits, where `count` is at most 57.
    #[inline]
    fn consume_cached(&mut self, count: u8) -> u64 {
        debug_assert!(u32::from(count) <= self.cached && count <= 57);
        // Splitting the shift keeps a zero-width read defined.
        let value = (self.cache >> 1) >> (63 - count);
        self.cache <<= count;
        self.cached -= u32::from(count);
        self.bit_position += usize::from(count);
        value
    }

    #[inline(never)]
    fn read_wide_bits(&mut self, count: u8) -> Result<u64, EntropyError> {
        if count > 64 {
            return Err(EntropyError::InvalidParameter {
                parameter: "bit count",
                value: i64::from(count),
            });
        }
        if usize::from(count) > self.bits_remaining() {
            return Err(self.end_error(count));
        }
        let high = self.read_bits(count - 32)?;
        let low = self.read_bits(32)?;
        Ok((high << 32) | low)
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

    /// Reloads the cache with the unread bits at `bit_position`.
    ///
    /// A byte-aligned 64-bit window holds at least 57 bits after the in-byte
    /// offset. Bytes beyond the backing slice read as zero, and bits beyond
    /// `bit_length` are cleared.
    #[inline(never)]
    fn refill(&mut self) {
        let byte = self.bit_position / 8;
        let word = match self.bytes.get(byte..byte + 8) {
            Some(chunk) => {
                u64::from_be_bytes(chunk.try_into().expect("slice is exactly eight bytes"))
            }
            None => Self::padded_word(self.bytes, byte),
        };
        let offset = self.bit_position % 8;
        let available = (64 - offset).min(self.bits_remaining());
        // `available` is at least 57 unless the packet ends, and never zero
        // when any bit remains.
        self.cached = u32::try_from(available).expect("window holds at most 64 bits");
        self.cache = match available {
            0 => 0,
            bits => (word << offset) & (u64::MAX << (64 - bits)),
        };
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
    fn cached_reads_cross_refills_and_preserve_end_errors() {
        let bytes: Vec<u8> = (0_u8..40)
            .map(|index| index.wrapping_mul(73) ^ 0x3c)
            .collect();
        let bit_length = bytes.len() * 8 - 5;
        let widths = [1_u8, 7, 3, 13, 57, 2, 64, 0, 31, 9, 58, 5];
        let mut reader = PacketBitReader::with_bit_length(&bytes, bit_length).unwrap();
        let mut position = 0;
        for &count in widths.iter().cycle().take(64) {
            let result = reader.read_bits(count);
            if position + usize::from(count) > bit_length {
                assert_eq!(
                    result,
                    Err(EntropyError::UnexpectedEnd {
                        bit_position: position,
                        requested_bits: count,
                        bit_length,
                    })
                );
            } else {
                assert_eq!(
                    result.unwrap(),
                    reference_bits(&bytes, position, usize::from(count)),
                    "position {position}, count {count}"
                );
                position += usize::from(count);
            }
            assert_eq!(reader.bit_position(), position);
            let expected_peek = (0..8).fold(0_u8, |value, bit| {
                let index = position + bit;
                let set = index < bit_length && (bytes[index / 8] >> (7 - index % 8)) & 1 != 0;
                (value << 1) | u8::from(set)
            });
            assert_eq!(reader.peek_byte(), expected_peek);
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
