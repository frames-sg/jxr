//! Codestreams the integration tests build bit by bit, from the syntax
//! rather than through the encoder, so that what the parser reads is checked
//! against the standard and not against the writer. The tests of `jxr-native`
//! hold the same module: a crate both could depend on, such as
//! `jxr-test-support`, depends on both, and would make a dependency cycle.

#![allow(
    dead_code,
    reason = "each test binary compiles this module and uses part of it"
)]

/// A 16-by-16 short-header codestream: the image header, whose last byte,
/// `output`, holds the output colour format and bit depth (`0x01` 8-bit
/// gray, `0x71` 8-bit RGB), then the plane header fields `fields` as
/// `(value, bits)` padded to a byte, then `tail`, the index and the packets.
pub(crate) fn codestream(output: u8, fields: &[(u64, u8)], tail: &[u8]) -> Vec<u8> {
    let mut bytes = b"WMPHOTO\0".to_vec();
    bytes.extend_from_slice(&[0x11, 0x00, 0x80, output]);
    bytes.extend_from_slice(&15_u16.to_be_bytes());
    bytes.extend_from_slice(&15_u16.to_be_bytes());
    let mut bit_position = bytes.len() * 8;
    for &(value, count) in fields {
        for shift in (0..count).rev() {
            if bit_position / 8 == bytes.len() {
                bytes.push(0);
            }
            let bit = u8::from(((value >> shift) & 1) != 0);
            bytes[bit_position / 8] |= bit << (7 - (bit_position % 8));
            bit_position += 1;
        }
    }
    bytes.extend_from_slice(tail);
    bytes
}

/// A 16-by-16 DC-only luma codestream of one tile. `scaled` sets the plane
/// header's scaled-arithmetic bit; `macroblock` is the tile's entropy-coded
/// payload.
pub(crate) fn dc_only_luma_codestream(scaled: bool, macroblock: [u8; 3]) -> Vec<u8> {
    // Y-only, DC only, uniform DC quantizer of one.
    let fields = [(0, 3), (u64::from(scaled), 1), (3, 4), (1, 1), (1, 8)];
    // The index in VLW escape form, then the tile packet.
    let mut tail = vec![0xFD, 0, 0, 1, 0x5a];
    tail.extend_from_slice(&macroblock);
    codestream(0x01, &fields, &tail)
}
