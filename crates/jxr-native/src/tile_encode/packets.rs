//! Where the bits of each band of a tile are written.

use jxr_core::BitstreamMode;

use crate::{bit_writer::BitWriter, tail::PacketBand, tile_decode::spatial::TILE_START_CODE};

use super::{PacketOptions, TileEncodeError};

/// The packets of one tile being written: one that interleaves the bands of
/// every macroblock, or one for each band.
///
/// The macroblock writer hands every syntax element to the writer of the
/// band it belongs to, in the one order both layouts share. In spatial mode
/// every band names the same writer, so the elements land interleaved as
/// they are produced. In frequency mode each band's elements collect in a
/// packet of its own, and the flexbits of a block, which a spatial packet
/// holds right after the block, follow the previous block's flexbits in the
/// flexbits packet instead.
#[derive(Debug)]
pub(super) struct TilePackets {
    /// The one packet of a spatial-mode tile, or the packet of each
    /// [`PacketBand::coded`] band of a frequency-mode tile.
    writers: Vec<BitWriter>,
    mode: BitstreamMode,
    /// Whether the first packet is written onto the end of the output.
    lent: bool,
}

impl TilePackets {
    /// Starts the packets of a tile, each with its start code and arbitrary
    /// byte. `TRIM_FLEXBITS`, when the image header enables it, opens the
    /// packet that carries flexbits: the flexbits packet of a frequency-mode
    /// tile, and otherwise the tile's one packet.
    ///
    /// Quantizers are taken from the image plane headers, so no packet
    /// carries the quantizer header of its bands and no macroblock a
    /// quantizer index.
    ///
    /// Each packet is written in one of `buffers` while there is one, so a
    /// buffer that has held a packet before does not grow again for a
    /// packet no longer than that one. The first packet is written onto
    /// the end of `output` instead when there is one: the output the tile's
    /// packets are next in, lent to the tile and given back by
    /// [`Self::finish_into`], so that its packet is never copied.
    pub(super) fn start(
        options: PacketOptions,
        buffers: &mut Vec<Vec<u8>>,
        output: Option<Vec<u8>>,
    ) -> Result<Self, TileEncodeError> {
        if options.trim_flexbits.is_some_and(|trim| trim > 15) {
            return Err(TileEncodeError::InvalidInput("trim flexbits"));
        }
        let lent = output.is_some();
        let mut output = output;
        let mut start = |trim_flexbits| {
            let writer = match output.take() {
                Some(output) => {
                    // Grown by an eighth, not doubled by the start code.
                    let mut writer = BitWriter::appending(output);
                    writer.reserve(START_BYTES, 0);
                    writer
                }
                None => BitWriter::reusing(buffers.pop().unwrap_or_default()),
            };
            start_packet(writer, trim_flexbits)
        };
        let mut writers = Vec::with_capacity(PacketBand::coded(options.bands).len());
        match options.mode {
            BitstreamMode::Spatial => writers.push(start(options.trim_flexbits)),
            BitstreamMode::Frequency => {
                for &band in PacketBand::coded(options.bands) {
                    let carries_trim = band == PacketBand::Flexbits;
                    writers.push(start(options.trim_flexbits.filter(|_| carries_trim)));
                }
            }
        }
        Ok(Self {
            writers,
            mode: options.mode,
            lent,
        })
    }

    /// Whether the first packet holds the output it is written onto.
    pub(super) const fn holds_output(&self) -> bool {
        self.lent
    }

    /// Make room in the output the first packet holds for `additional`
    /// more bytes, of `expected` that the output is expected to hold in
    /// all, growing it as the output grows when a packet is appended to it.
    pub(super) fn reserve_output(&mut self, additional: usize, expected: usize) {
        if let (true, Some(writer)) = (self.lent, self.writers.first_mut()) {
            writer.reserve(additional, expected);
        }
    }

    /// The writer of the packet that carries `band`.
    pub(super) fn band(&mut self, band: PacketBand) -> Result<&mut BitWriter, TileEncodeError> {
        let packet = match self.mode {
            BitstreamMode::Spatial => 0,
            BitstreamMode::Frequency => band.index(),
        };
        self.writers
            .get_mut(packet)
            .ok_or(TileEncodeError::InvalidInput("band without a packet"))
    }

    /// Bytes the packets hold once finished, were they finished now.
    pub(super) fn written_bytes(&self) -> usize {
        self.writers.iter().map(BitWriter::byte_len).sum()
    }

    /// Pads each packet to a whole byte and appends it to `output`, in the
    /// order the tile index lists them, with its length to `lengths`. A
    /// first packet written onto the output lent to it becomes `output`,
    /// which must be empty, as it is left while the output is lent.
    ///
    /// The buffer a packet of at most [`RETAINED_PACKET_BYTES`] was written
    /// in goes back to `buffers` for the next tile; a longer one is freed,
    /// so no worker keeps a large buffer.
    pub(super) fn finish_into(
        self,
        output: &mut Vec<u8>,
        lengths: &mut Vec<usize>,
        buffers: &mut Vec<Vec<u8>>,
    ) -> Result<(), TileEncodeError> {
        let mut lent = self.lent;
        for writer in self.writers {
            lengths.push(writer.byte_len());
            let written = writer.into_bytes();
            if core::mem::take(&mut lent) {
                if !output.is_empty() {
                    return Err(TileEncodeError::InvalidInput("output written while lent"));
                }
                *output = written;
                continue;
            }
            output.extend_from_slice(&written);
            if written.len() <= RETAINED_PACKET_BYTES {
                buffers.push(written);
            }
        }
        Ok(())
    }

    /// The finished packets, each padded to a whole byte and holding no
    /// more storage than its bytes, in the order the tile index lists them:
    /// [`Self::finish_into`] split back into packets, which leaves the
    /// buffers it keeps in `buffers`.
    #[cfg(test)]
    pub(super) fn finish(self, buffers: &mut Vec<Vec<u8>>) -> Vec<Vec<u8>> {
        let (mut output, mut lengths) = (Vec::new(), Vec::new());
        self.finish_into(&mut output, &mut lengths, buffers)
            .expect("packets that hold no output finish");
        let mut rest = output.as_slice();
        lengths
            .iter()
            .map(|&length| {
                let (packet, after) = rest.split_at(length);
                rest = after;
                packet.to_vec()
            })
            .collect()
    }
}

/// Longest packet whose buffer a worker keeps for its next tile: more than
/// the packets of a default-sized tile of most images, and little enough
/// that the scratch of every worker together stays small beside one image.
const RETAINED_PACKET_BYTES: usize = 1 << 20;

/// Bytes a packet opens with: its start code, its arbitrary byte, and the
/// trimmed flexbits count.
const START_BYTES: usize = 5;

/// A packet up to its first macroblock: start code, arbitrary byte, and
/// `trim_flexbits` when the packet carries it.
fn start_packet(mut writer: BitWriter, trim_flexbits: Option<u8>) -> BitWriter {
    writer.write_bits(u64::from(TILE_START_CODE), 24);
    // ARBITRARY_BYTE carries no meaning; zero keeps the output deterministic.
    writer.write_bits(0, 8);
    if let Some(trim) = trim_flexbits {
        writer.write_bits(u64::from(trim), 4);
    }
    writer
}

#[cfg(test)]
mod tests {
    use jxr_core::{BandPresence, ColorFormat};

    use super::*;

    fn options(mode: BitstreamMode, bands: BandPresence, trim: Option<u8>) -> PacketOptions {
        PacketOptions {
            colour: ColorFormat::Luma,
            bands,
            trim_flexbits: trim,
            mode,
        }
    }

    const BANDS: [PacketBand; 4] = [
        PacketBand::Dc,
        PacketBand::LowPass,
        PacketBand::HighPass,
        PacketBand::Flexbits,
    ];

    #[test]
    fn a_spatial_tile_is_one_packet_whatever_the_band() {
        let mut packets = TilePackets::start(
            options(BitstreamMode::Spatial, BandPresence::All, Some(5)),
            &mut Vec::new(),
            None,
        )
        .unwrap();
        for (value, band) in (1..).zip(BANDS) {
            packets.band(band).unwrap().write_bits(value, 4);
        }
        // Start code, arbitrary byte, the trim, then the bands interleaved.
        assert_eq!(
            packets.finish(&mut Vec::new()),
            [vec![0, 0, 1, 0, 0x51, 0x23, 0x40]]
        );
    }

    #[test]
    fn a_frequency_tile_has_one_packet_for_each_coded_band() {
        for (bands, count) in [
            (BandPresence::DcOnly, 1),
            (BandPresence::NoHighPass, 2),
            (BandPresence::NoFlexbits, 3),
            (BandPresence::All, 4),
        ] {
            let trim = bands.has_flexbits().then_some(5);
            let mut packets = TilePackets::start(
                options(BitstreamMode::Frequency, bands, trim),
                &mut Vec::new(),
                None,
            )
            .unwrap();
            for (value, band) in (1..).zip(BANDS).take(count) {
                packets.band(band).unwrap().write_bits(value, 4);
            }
            // A band the image does not code has no packet to write to.
            for band in BANDS.into_iter().skip(count) {
                assert_eq!(
                    packets.band(band).unwrap_err(),
                    TileEncodeError::InvalidInput("band without a packet")
                );
            }
            let packets = packets.finish(&mut Vec::new());
            assert_eq!(packets.len(), count, "{bands:?}");
            for (value, packet) in (1_u8..).zip(&packets) {
                // Only the flexbits packet carries the trim.
                let expected = if value == 4 {
                    vec![0, 0, 1, 0, 0x54]
                } else {
                    vec![0, 0, 1, 0, value << 4]
                };
                assert_eq!(packet, &expected, "{bands:?} packet {value}");
            }
        }
    }

    #[test]
    fn only_packets_within_the_retained_size_leave_their_buffer_behind() {
        for (payload, retained) in [
            (0, true),
            (RETAINED_PACKET_BYTES - 4, true),
            (RETAINED_PACKET_BYTES - 3, false),
            (3 * RETAINED_PACKET_BYTES, false),
        ] {
            let mut buffers = Vec::new();
            let mut packets = TilePackets::start(
                options(BitstreamMode::Spatial, BandPresence::All, None),
                &mut buffers,
                None,
            )
            .unwrap();
            for index in 0..payload {
                let byte = u64::try_from(index % 251).unwrap();
                packets.band(PacketBand::Dc).unwrap().write_bits(byte, 8);
            }
            let [packet] = <[Vec<u8>; 1]>::try_from(packets.finish(&mut buffers)).unwrap();
            // Start code and arbitrary byte, then the payload.
            assert_eq!(packet.len(), payload + 4);
            assert_eq!(packet[..4], [0, 0, 1, 0]);
            assert!(
                packet[4..]
                    .iter()
                    .zip((0..251_u8).cycle())
                    .all(|(byte, expected)| *byte == expected)
            );
            // Either way the packet holds no spare storage.
            assert_eq!(packet.capacity(), packet.len());
            assert_eq!(buffers.len(), usize::from(retained), "{payload} bytes");
            assert!(
                buffers
                    .iter()
                    .all(|buffer| buffer.capacity() >= packet.len())
            );
        }
    }

    #[test]
    fn an_out_of_range_trim_is_rejected() {
        for mode in [BitstreamMode::Spatial, BitstreamMode::Frequency] {
            assert_eq!(
                TilePackets::start(
                    options(mode, BandPresence::All, Some(16)),
                    &mut Vec::new(),
                    None
                )
                .unwrap_err(),
                TileEncodeError::InvalidInput("trim flexbits")
            );
        }
    }
}
