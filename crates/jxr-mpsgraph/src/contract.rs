// SPDX-License-Identifier: MIT OR Apache-2.0

use jxr_core::{ChannelLayout, PixelFormat, SurfaceLayout};

use crate::Error;

/// Native integer element type exposed to `MPSGraph`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MpsGraphElementType {
    U8,
    U16,
    I16,
}

/// Validated static rank-four `[N, H, W, C]` tensor contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MpsGraphTensorSpec {
    shape: [usize; 4],
    element_type: MpsGraphElementType,
}

impl MpsGraphTensorSpec {
    /// Construct an explicit nonempty static NHWC tensor contract.
    pub fn new(shape: [usize; 4], element_type: MpsGraphElementType) -> Result<Self, Error> {
        if shape.contains(&0) {
            return Err(Error::InvalidTensorContract {
                reason: "MPSGraph static image dimensions must be nonzero",
            });
        }
        shape
            .into_iter()
            .try_fold(1_usize, usize::checked_mul)
            .ok_or(Error::TensorShapeOverflow)?;
        Ok(Self {
            shape,
            element_type,
        })
    }

    /// Derive the dense NHWC contract for a homogeneous JPEG XR batch.
    pub fn from_image_layout(layout: &SurfaceLayout, image_count: usize) -> Result<Self, Error> {
        if image_count == 0 {
            return Err(Error::InvalidTensorContract {
                reason: "MPSGraph image count must be nonzero",
            });
        }
        let element_type = element_type(layout.format)?;
        let channels = channels(layout.format)?;
        let height = usize::try_from(layout.height).map_err(|_| Error::TensorShapeOverflow)?;
        let width = usize::try_from(layout.width).map_err(|_| Error::TensorShapeOverflow)?;
        Self::new([image_count, height, width, channels], element_type)
    }

    #[must_use]
    pub const fn shape(self) -> [usize; 4] {
        self.shape
    }

    #[must_use]
    pub const fn element_type(self) -> MpsGraphElementType {
        self.element_type
    }

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    pub(crate) fn byte_len(self) -> Result<usize, Error> {
        let element_size = match self.element_type {
            MpsGraphElementType::U8 => 1,
            MpsGraphElementType::U16 | MpsGraphElementType::I16 => 2,
        };
        self.shape
            .into_iter()
            .try_fold(element_size, usize::checked_mul)
            .ok_or(Error::TensorShapeOverflow)
    }
}

pub(crate) fn element_type(format: PixelFormat) -> Result<MpsGraphElementType, Error> {
    match format {
        PixelFormat::U8(_) => Ok(MpsGraphElementType::U8),
        PixelFormat::U16(_) => Ok(MpsGraphElementType::U16),
        PixelFormat::I16(_) => Ok(MpsGraphElementType::I16),
        _ => Err(Error::InvalidTensorContract {
            reason: "direct MPSGraph tensors require U8, U16, or I16 samples",
        }),
    }
}

pub(crate) fn channels(format: PixelFormat) -> Result<usize, Error> {
    let (PixelFormat::U8(layout) | PixelFormat::U16(layout) | PixelFormat::I16(layout)) = format
    else {
        return Err(Error::InvalidTensorContract {
            reason: "direct MPSGraph tensors require U8, U16, or I16 samples",
        });
    };
    match layout {
        ChannelLayout::Luma => Ok(1),
        ChannelLayout::Rgb => Ok(3),
        ChannelLayout::Rgba => Ok(4),
        _ => Err(Error::InvalidTensorContract {
            reason: "direct MPSGraph tensors require Gray, RGB, or RGBA channel order",
        }),
    }
}
