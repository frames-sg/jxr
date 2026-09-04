// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::Error;

/// Fixed RGB weights used by the reference graph and CPU oracle.
pub const RGB8_REFERENCE_CHANNEL_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// CPU oracle for the RGB8/NHWC reference graph.
#[expect(clippy::cast_precision_loss)]
pub fn rgb8_nhwc_reference_cpu(
    pixels: &[u8],
    batch: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, Error> {
    let image_samples = height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or(Error::TensorShapeOverflow)?;
    let expected = batch
        .checked_mul(image_samples)
        .ok_or(Error::TensorShapeOverflow)?;
    if pixels.len() != expected || batch == 0 || height == 0 || width == 0 {
        return Err(Error::InvalidTensorContract {
            reason: "RGB8/NHWC CPU oracle input does not match a nonempty shape",
        });
    }
    let spatial = height
        .checked_mul(width)
        .ok_or(Error::TensorShapeOverflow)?;
    let mut output = Vec::with_capacity(batch);
    for image in pixels.chunks_exact(image_samples) {
        let sum = image.chunks_exact(3).fold(0.0_f32, |sum, rgb| {
            sum + f32::from(rgb[0]) * RGB8_REFERENCE_CHANNEL_WEIGHTS[0]
                + f32::from(rgb[1]) * RGB8_REFERENCE_CHANNEL_WEIGHTS[1]
                + f32::from(rgb[2]) * RGB8_REFERENCE_CHANNEL_WEIGHTS[2]
        });
        output.push(sum / (255.0 * spatial as f32));
    }
    Ok(output)
}
