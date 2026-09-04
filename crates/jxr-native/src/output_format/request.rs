//! Input plane and output policy contracts.

pub use jxr_core::{AlphaFormatRequest, OutputBitDepth, OutputFormatRequest};

/// Borrowed row-major reconstructed component samples.
#[derive(Clone, Copy, Debug)]
pub struct ComponentPlane<'a> {
    /// Horizontal origin in the extended image-plane coordinate system.
    pub origin_x: u32,
    /// Vertical origin in the extended image-plane coordinate system.
    pub origin_y: u32,
    /// Number of addressable samples in each row.
    pub width: u32,
    /// Number of addressable sample rows.
    pub height: u32,
    /// Distance between consecutive rows, in samples.
    pub stride: usize,
    /// Signed reconstructed samples before output colour conversion and bias.
    pub samples: &'a [i32],
}

impl<'a> ComponentPlane<'a> {
    /// Construct a tightly packed component plane.
    #[must_use]
    pub fn tightly_packed(width: u32, height: u32, samples: &'a [i32]) -> Self {
        Self {
            origin_x: 0,
            origin_y: 0,
            width,
            height,
            stride: usize::try_from(width).unwrap_or(usize::MAX),
            samples,
        }
    }

    /// Construct a tightly packed component window at an extended-plane origin.
    #[must_use]
    pub fn positioned(
        origin_x: u32,
        origin_y: u32,
        width: u32,
        height: u32,
        samples: &'a [i32],
    ) -> Self {
        Self {
            origin_x,
            origin_y,
            width,
            height,
            stride: usize::try_from(width).unwrap_or(usize::MAX),
            samples,
        }
    }

    pub(crate) fn sample(self, x: usize, y: usize) -> i32 {
        let x = x - self.origin_x as usize;
        let y = y - self.origin_y as usize;
        self.samples[y * self.stride + x]
    }
}

#[cfg(test)]
mod tests {
    use super::OutputBitDepth;

    #[test]
    fn output_depth_maps_normative_header_codes() {
        assert_eq!(
            OutputBitDepth::from_header_fields(2, 7, 0, 0).unwrap(),
            OutputBitDepth::U16 { shift_bits: 7 }
        );
        assert_eq!(
            OutputBitDepth::from_header_fields(7, 0, 12, -3).unwrap(),
            OutputBitDepth::F32 {
                mantissa_length: 12,
                exponent_bias: -3,
            }
        );
        assert_eq!(
            OutputBitDepth::from_header_fields(9, 0, 0, 0).unwrap(),
            OutputBitDepth::U10
        );
        assert!(OutputBitDepth::from_header_fields(5, 0, 0, 0).is_none());
    }
}
