//! T.832 clause 9.10.8 crop, clipping, and luma packing.

use jxr_math::color::clamp_sample;

use super::{CropWindow, PlanarSamples, ReconstructionError};

/// Crop and clip already biased/post-scaled luma samples to unsigned 8-bit values.
///
/// This function deliberately does not apply JPEG XR bias or post-scaling: its
/// input contract begins after those output-formatting stages.
pub fn pack_luma_u8(
    plane: &PlanarSamples,
    crop: CropWindow,
) -> Result<Vec<u8>, ReconstructionError> {
    let range = checked_crop(plane, crop)?;
    let mut output = Vec::with_capacity(range.elements);
    for row in 0..range.height {
        let start = (range.y + row) * range.stride + range.x;
        output.extend(
            plane.samples[start..start + range.width]
                .iter()
                .map(|&sample| {
                    u8::try_from(clamp_sample(sample, 0, 255)).expect("clamped sample fits u8")
                }),
        );
    }
    Ok(output)
}

/// Crop and clip already biased/post-scaled luma samples to unsigned 16-bit values.
///
/// This function deliberately does not apply JPEG XR bias or post-scaling: its
/// input contract begins after those output-formatting stages.
pub fn pack_luma_u16(
    plane: &PlanarSamples,
    crop: CropWindow,
) -> Result<Vec<u16>, ReconstructionError> {
    let range = checked_crop(plane, crop)?;
    let mut output = Vec::with_capacity(range.elements);
    for row in 0..range.height {
        let start = (range.y + row) * range.stride + range.x;
        output.extend(
            plane.samples[start..start + range.width]
                .iter()
                .map(|&sample| {
                    u16::try_from(clamp_sample(sample, 0, 65_535)).expect("clamped sample fits u16")
                }),
        );
    }
    Ok(output)
}

struct CheckedCrop {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    stride: usize,
    elements: usize,
}

fn checked_crop(
    plane: &PlanarSamples,
    crop: CropWindow,
) -> Result<CheckedCrop, ReconstructionError> {
    let right = crop
        .x
        .checked_add(crop.width)
        .ok_or(ReconstructionError::CropOutsidePlane)?;
    let bottom = crop
        .y
        .checked_add(crop.height)
        .ok_or(ReconstructionError::CropOutsidePlane)?;
    if right > plane.width || bottom > plane.height {
        return Err(ReconstructionError::CropOutsidePlane);
    }
    let stride = usize::try_from(plane.width)
        .map_err(|_| ReconstructionError::ArithmeticOverflow("plane stride conversion"))?;
    let plane_height = usize::try_from(plane.height)
        .map_err(|_| ReconstructionError::ArithmeticOverflow("plane height conversion"))?;
    let required =
        stride
            .checked_mul(plane_height)
            .ok_or(ReconstructionError::ArithmeticOverflow(
                "plane sample count",
            ))?;
    if plane.samples.len() < required {
        return Err(ReconstructionError::BufferTooSmall {
            required,
            available: plane.samples.len(),
        });
    }
    let width = usize::try_from(crop.width)
        .map_err(|_| ReconstructionError::ArithmeticOverflow("crop width conversion"))?;
    let height = usize::try_from(crop.height)
        .map_err(|_| ReconstructionError::ArithmeticOverflow("crop height conversion"))?;
    let elements = width
        .checked_mul(height)
        .ok_or(ReconstructionError::ArithmeticOverflow("crop sample count"))?;
    Ok(CheckedCrop {
        x: usize::try_from(crop.x)
            .map_err(|_| ReconstructionError::ArithmeticOverflow("crop x conversion"))?,
        y: usize::try_from(crop.y)
            .map_err(|_| ReconstructionError::ArithmeticOverflow("crop y conversion"))?,
        width,
        height,
        stride,
        elements,
    })
}

#[cfg(test)]
mod tests {
    use super::{CropWindow, PlanarSamples, pack_luma_u8, pack_luma_u16};

    #[test]
    fn luma_u8_crop_clips_both_bounds() {
        let plane = PlanarSamples {
            origin_x: 0,
            origin_y: 0,
            width: 4,
            height: 2,
            samples: vec![-1, 1, 2, 3, 4, 255, 256, 7],
        };
        let crop = CropWindow {
            x: 1,
            y: 0,
            width: 2,
            height: 2,
        };
        assert_eq!(pack_luma_u8(&plane, crop).unwrap(), [1, 2, 255, 255]);
    }

    #[test]
    fn luma_u16_crop_clips_and_preserves_row_order() {
        let plane = PlanarSamples {
            origin_x: 0,
            origin_y: 0,
            width: 3,
            height: 2,
            samples: vec![9, -2, 70_000, 8, 42, 43],
        };
        let crop = CropWindow {
            x: 1,
            y: 0,
            width: 2,
            height: 2,
        };
        assert_eq!(pack_luma_u16(&plane, crop).unwrap(), [0, 65_535, 42, 43]);
    }

    #[test]
    fn crop_outside_plane_is_rejected() {
        let plane = PlanarSamples {
            origin_x: 0,
            origin_y: 0,
            width: 1,
            height: 1,
            samples: vec![0],
        };
        let crop = CropWindow {
            x: 1,
            y: 0,
            width: 1,
            height: 1,
        };
        assert!(pack_luma_u8(&plane, crop).is_err());
    }
}
