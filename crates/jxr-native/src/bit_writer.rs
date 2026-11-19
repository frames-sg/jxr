//! Most-significant-bit-first JPEG XR bit writing.

/// Growable writer for T.832 header syntax and tile packets.
///
/// Bits accumulate in a 64-bit register that is emptied four bytes at a
/// time, so an entropy-coded element costs one shift and, about once in
/// every 32 bits, one four-byte append. The backing vector therefore lags
/// the written bits by up to 31; the final partial byte is zero-padded only
/// when the writer is aligned or finished.
#[derive(Debug, Clone, Default)]
pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    /// Bytes of `bytes` from before the writer's first bit, which it keeps.
    start: usize,
    /// The bits not yet in `bytes`, in the low `pending` bits. Higher bits
    /// are stale and never read.
    accumulator: u64,
    /// Fewer than [`WORD_BITS`] between writes.
    pending: u32,
}

/// Bits moved from the register to the byte vector at a time.
const WORD_BITS: u32 = 32;

impl BitWriter {
    pub(crate) const fn new() -> Self {
        Self {
            bytes: Vec::new(),
            start: 0,
            accumulator: 0,
            pending: 0,
        }
    }

    /// An empty writer that writes into the storage of `bytes`, whose
    /// contents are discarded.
    pub(crate) fn reusing(mut bytes: Vec<u8>) -> Self {
        bytes.clear();
        Self {
            bytes,
            start: 0,
            accumulator: 0,
            pending: 0,
        }
    }

    /// A writer that appends to `bytes`, whose contents it keeps ahead of
    /// what it writes and returns with it.
    pub(crate) const fn appending(bytes: Vec<u8>) -> Self {
        let start = bytes.len();
        Self {
            bytes,
            start,
            accumulator: 0,
            pending: 0,
        }
    }

    /// Number of bits written so far.
    #[cfg(test)]
    pub(crate) const fn bit_position(&self) -> usize {
        (self.bytes.len() - self.start) * 8 + self.pending as usize
    }

    /// Bytes written once finished, were the writer finished now, not
    /// counting those it was given to append to.
    pub(crate) const fn byte_len(&self) -> usize {
        self.bytes.len() - self.start + (self.pending as usize).div_ceil(8)
    }

    /// Make room for `additional` more bytes as [`reserve_for`] does, for
    /// bytes expected to come to `expected` in all.
    pub(crate) fn reserve(&mut self, additional: usize, expected: usize) {
        reserve_for(&mut self.bytes, additional, expected);
    }

    #[inline]
    pub(crate) fn write_flag(&mut self, value: bool) {
        self.write_word(u64::from(value), 1);
    }

    /// Writes the low `count` bits of `value`, most significant first.
    ///
    /// Bits of `value` above `count` are ignored, matching a fixed-width
    /// syntax element whose caller has already range-checked the value.
    #[inline]
    pub(crate) fn write_bits(&mut self, value: u64, count: u8) {
        debug_assert!(count <= 64, "bit write width");
        let count = u32::from(count);
        if count > WORD_BITS {
            self.write_word(value >> WORD_BITS, count - WORD_BITS);
            self.write_word(value, WORD_BITS);
        } else {
            self.write_word(value, count);
        }
    }

    /// Writes the low `count` bits of `value`, at most [`WORD_BITS`] of them.
    #[inline]
    fn write_word(&mut self, value: u64, count: u32) {
        debug_assert!(count <= WORD_BITS, "word write width");
        // Fewer than 32 bits are pending, so 32 more always fit in 64.
        let mask = (1_u64 << count) - 1;
        self.accumulator = (self.accumulator << count) | (value & mask);
        self.pending += count;
        if self.pending >= WORD_BITS {
            self.pending -= WORD_BITS;
            let [.., a, b, c, d] = (self.accumulator >> self.pending).to_be_bytes();
            self.bytes.extend_from_slice(&[a, b, c, d]);
        }
    }

    /// Pads with zero bits to the next byte boundary.
    pub(crate) fn align_zero(&mut self) {
        let padding = self.pending.wrapping_neg() % 8;
        // Zero bits up to the boundary, then the whole bytes left pending.
        self.accumulator <<= padding;
        self.pending += padding;
        while self.pending != 0 {
            self.pending -= 8;
            let [.., byte] = (self.accumulator >> self.pending).to_be_bytes();
            self.bytes.push(byte);
        }
        self.accumulator = 0;
    }

    /// Zero-pads the final byte and returns the written bytes, after those
    /// the writer was given to append to.
    pub(crate) fn into_bytes(mut self) -> Vec<u8> {
        self.align_zero();
        self.bytes
    }
}

/// Make room in `bytes` for `additional` more bytes, of `expected` that it
/// is expected to hold in all once complete.
///
/// When it has to grow, a buffer appended to piece by piece grows at once
/// to what it is expected to hold, and by at least an eighth, so that it is
/// reallocated a bounded number of times, and only once more after the
/// expectation is known where it holds. A reallocation can move the
/// buffer, and in a WebAssembly memory, which never shrinks, a buffer that
/// has outgrown the space freed below it moves to the top of memory every
/// time it grows.
pub(crate) fn reserve_for(bytes: &mut Vec<u8>, additional: usize, expected: usize) {
    let spare = bytes.capacity() - bytes.len();
    if spare < additional {
        let wanted = expected.saturating_sub(bytes.len());
        bytes.reserve_exact(additional.max(wanted).max(bytes.len() / 8));
    }
}

#[cfg(test)]
mod tests {
    use super::{BitWriter, reserve_for};
    use crate::bit_reader::BitReader;
    use crate::entropy::encode::test_support::Lcg;

    #[test]
    fn writes_across_byte_boundaries_msb_first() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b101, 3);
        writer.write_bits(0b1_1111_0000, 9);
        writer.write_flag(true);
        assert_eq!(writer.bit_position(), 13);
        assert_eq!(writer.byte_len(), 2);
        assert_eq!(writer.into_bytes(), [0b1011_1111, 0b0000_1000]);
    }

    #[test]
    fn every_width_round_trips_through_the_header_reader() {
        let mut writer = BitWriter::new();
        let mut expected = Vec::new();
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for count in 0..=64_u8 {
            state = Lcg::advance(state);
            let value = if count == 64 {
                state
            } else {
                state & ((1_u64 << count) - 1)
            };
            // High garbage bits must be ignored by the writer.
            writer.write_bits(state, count);
            expected.push((value, count));
        }
        let bytes = writer.into_bytes();
        let mut reader = BitReader::new(&bytes);
        for (value, count) in expected {
            assert_eq!(reader.read_bits(count).unwrap(), value, "width {count}");
        }
    }

    #[test]
    fn narrow_writes_match_a_bit_at_a_time_reference() {
        // Widths that leave every count of bits pending when the register
        // is emptied, and every count when the writer is finished.
        for total in 0..200_usize {
            let mut writer = BitWriter::new();
            let mut expected = vec![0_u8; total.div_ceil(8)];
            let mut state = 0x9e37_79b9_7f4a_7c15_u64;
            let mut position = 0;
            while position < total {
                state = Lcg::advance(state);
                let count = usize::try_from(state >> 60).unwrap().min(total - position);
                let value = state >> 20;
                for bit in (0..count).rev() {
                    if value >> bit & 1 != 0 {
                        expected[position / 8] |= 0x80 >> (position % 8);
                    }
                    position += 1;
                }
                writer.write_bits(value, u8::try_from(count).unwrap());
                assert_eq!(writer.bit_position(), position);
            }
            assert_eq!(writer.into_bytes(), expected, "{total} bits");
        }
    }

    #[test]
    fn alignment_pads_with_zero_bits_and_is_idempotent() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b11, 2);
        writer.align_zero();
        writer.align_zero();
        writer.write_bits(0xab, 8);
        assert_eq!(writer.into_bytes(), [0b1100_0000, 0xab]);
    }

    #[test]
    fn an_appending_writer_counts_only_what_it_writes() {
        let mut writer = BitWriter::appending(vec![7, 8, 9]);
        assert_eq!((writer.bit_position(), writer.byte_len()), (0, 0));
        writer.write_bits(0b101, 3);
        writer.reserve(100, 0);
        writer.write_bits(0xff, 8);
        assert_eq!((writer.bit_position(), writer.byte_len()), (11, 2));
        assert_eq!(writer.into_bytes(), [7, 8, 9, 0b1011_1111, 0b1110_0000]);
    }

    #[test]
    fn a_buffer_grows_by_an_eighth_or_by_what_is_asked() {
        let mut bytes = vec![0; 800];
        bytes.shrink_to_fit();
        reserve_for(&mut bytes, 10, 0);
        assert_eq!(bytes.capacity(), 900);
        // Room enough already: no change.
        reserve_for(&mut bytes, 100, 0);
        assert_eq!(bytes.capacity(), 900);
        reserve_for(&mut bytes, 500, 0);
        assert_eq!(bytes.capacity(), 1300);
    }

    #[test]
    fn a_buffer_that_must_grow_grows_to_what_it_is_expected_to_hold() {
        let mut bytes = vec![0; 800];
        bytes.shrink_to_fit();
        reserve_for(&mut bytes, 10, 5_000);
        assert_eq!(bytes.capacity(), 5_000);
        // A buffer with room does not grow to the expectation.
        reserve_for(&mut bytes, 10, 9_000);
        assert_eq!(bytes.capacity(), 5_000);
        // Nor to less than what is asked, or an eighth.
        bytes.resize(4_990, 0);
        reserve_for(&mut bytes, 100, 5_010);
        assert_eq!(bytes.capacity(), 4_990 + 4_990 / 8);
        reserve_for(&mut bytes, 2_000, 0);
        assert_eq!(bytes.capacity(), 6_990);
    }
}
