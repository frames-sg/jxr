//! The synthetic source images of the memory tests.

/// Interleaved 8-bit pixels: gradients, texture, and noise in every
/// channel, which code to about half their size losslessly.
pub(crate) fn pixels(channels: usize, width: u32, height: u32) -> Vec<u8> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let size = usize::try_from(width * height).unwrap() * channels;
    let mut samples = Vec::with_capacity(size);
    for y in 0..height {
        for x in 0..width {
            for channel in 0..u32::try_from(channels).unwrap() {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let ramp = x * (2 + channel) + y * 3;
                let texture = if (x / 5 + y / 7 + channel).is_multiple_of(2) {
                    24
                } else {
                    0
                };
                let noise = u32::try_from((state >> 40) % 16).unwrap();
                samples.push(u8::try_from((ramp / 8 + texture + noise) % 256).unwrap());
            }
        }
    }
    samples
}
