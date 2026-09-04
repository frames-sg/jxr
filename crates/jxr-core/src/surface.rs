//! Device-neutral resident and external surface layouts.

use alloc::{vec, vec::Vec};

use crate::{
    ChannelLayout, ChromaSampling, ColorFormat, JxrError, JxrErrorKind, OutputFormatRequest,
    PixelFormat,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SurfacePlaneLayout {
    pub byte_offset: usize,
    pub row_stride_bytes: usize,
    pub width: u32,
    pub height: u32,
    pub channels: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurfaceLayout {
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub planes: Vec<SurfacePlaneLayout>,
    pub byte_len: usize,
    pub required_alignment: usize,
}

impl SurfaceLayout {
    /// Construct the default tightly packed layout for a validated output policy.
    pub fn for_output(
        policy: OutputFormatRequest,
        required_alignment: usize,
    ) -> Result<Self, JxrError> {
        if !required_alignment.is_power_of_two() {
            return Err(JxrError::new(
                JxrErrorKind::InvalidRequest,
                "surface alignment",
            ));
        }
        let width = policy.crop.width;
        let height = policy.crop.height;
        let sampling = match policy.output_color {
            ColorFormat::Yuv(ChromaSampling::Cs420) => Some(ChromaSampling::Cs420),
            ColorFormat::Yuv(ChromaSampling::Cs422) => Some(ChromaSampling::Cs422),
            _ => None,
        };
        let Some(sampling) = sampling else {
            return Self::tightly_packed(width, height, policy.pixel_format, required_alignment);
        };
        if width == 0 || height == 0 {
            return Err(JxrError::new(
                JxrErrorKind::InvalidRequest,
                "planar surface dimensions",
            ));
        }
        let include_alpha = matches!(
            channel_layout(policy.pixel_format),
            Some(ChannelLayout::Yuva(_))
        );
        let chroma_height = if sampling == ChromaSampling::Cs420 {
            height / 2
        } else {
            height
        };
        let geometries = [
            Some((width, height)),
            Some((width / 2, chroma_height)),
            Some((width / 2, chroma_height)),
            include_alpha.then_some((width, height)),
        ];
        let mut planes = Vec::with_capacity(3 + usize::from(include_alpha));
        let mut byte_len = 0_usize;
        for (plane_width, plane_height) in geometries.into_iter().flatten() {
            let byte_offset = align_up(byte_len, required_alignment)?;
            let row_stride_bytes = policy.pixel_format.row_bytes_for_channels(plane_width, 1)?;
            let plane_bytes = row_stride_bytes
                .checked_mul(
                    usize::try_from(plane_height)
                        .map_err(|_| JxrError::arithmetic("planar surface height conversion"))?,
                )
                .ok_or_else(|| JxrError::arithmetic("planar surface byte length"))?;
            byte_len = byte_offset
                .checked_add(plane_bytes)
                .ok_or_else(|| JxrError::arithmetic("planar surface extent"))?;
            planes.push(SurfacePlaneLayout {
                byte_offset,
                row_stride_bytes,
                width: plane_width,
                height: plane_height,
                channels: 1,
            });
        }
        let layout = Self {
            width,
            height,
            format: policy.pixel_format,
            planes,
            byte_len,
            required_alignment,
        };
        layout.validate()?;
        Ok(layout)
    }

    /// Construct one tightly packed interleaved plane with checked byte sizing.
    pub fn tightly_packed(
        width: u32,
        height: u32,
        format: PixelFormat,
        required_alignment: usize,
    ) -> Result<Self, JxrError> {
        if width == 0 || height == 0 {
            return Err(JxrError::new(
                JxrErrorKind::InvalidRequest,
                "surface dimensions",
            ));
        }
        let row_stride_bytes = format.row_bytes(width)?;
        let height_usize = usize::try_from(height)
            .map_err(|_| JxrError::arithmetic("surface height conversion"))?;
        let byte_len = row_stride_bytes
            .checked_mul(height_usize)
            .ok_or_else(|| JxrError::arithmetic("surface byte length"))?;
        let layout = Self {
            width,
            height,
            format,
            planes: vec![SurfacePlaneLayout {
                byte_offset: 0,
                row_stride_bytes,
                width,
                height,
                channels: format.channel_count(),
            }],
            byte_len,
            required_alignment,
        };
        layout.validate()?;
        Ok(layout)
    }

    pub fn validate(&self) -> Result<(), JxrError> {
        if self.width == 0 || self.height == 0 || self.planes.is_empty() {
            return Err(JxrError::new(
                JxrErrorKind::InvalidRequest,
                "surface dimensions",
            ));
        }
        if !self.required_alignment.is_power_of_two() {
            return Err(JxrError::new(
                JxrErrorKind::InvalidRequest,
                "surface alignment",
            ));
        }
        for plane in &self.planes {
            self.validate_plane(*plane)?;
        }
        for (index, plane) in self.planes.iter().enumerate() {
            let first = self.plane_extent(*plane)?;
            for other in &self.planes[index + 1..] {
                let second = self.plane_extent(*other)?;
                if first.0 < second.1 && second.0 < first.1 {
                    return Err(JxrError::new(
                        JxrErrorKind::InvalidRequest,
                        "overlapping surface planes",
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_plane(&self, plane: SurfacePlaneLayout) -> Result<(), JxrError> {
        if plane.width > self.width || plane.height > self.height {
            return Err(JxrError::new(
                JxrErrorKind::InvalidRequest,
                "surface plane dimensions",
            ));
        }
        if !plane.byte_offset.is_multiple_of(self.required_alignment) {
            return Err(JxrError::new(
                JxrErrorKind::InvalidRequest,
                "surface plane alignment",
            ));
        }
        let row_bytes = self
            .format
            .row_bytes_for_channels(plane.width, plane.channels)?;
        if plane.row_stride_bytes < row_bytes {
            return Err(JxrError::new(
                JxrErrorKind::InvalidRequest,
                "surface plane stride",
            ));
        }
        let (_, required) = self.plane_extent(plane)?;
        if required > self.byte_len {
            return Err(JxrError::new(
                JxrErrorKind::BufferTooSmall {
                    required,
                    available: self.byte_len,
                },
                "surface plane extent",
            ));
        }
        Ok(())
    }

    fn plane_extent(&self, plane: SurfacePlaneLayout) -> Result<(usize, usize), JxrError> {
        let row_bytes = self
            .format
            .row_bytes_for_channels(plane.width, plane.channels)?;
        let row_count = usize::try_from(plane.height.saturating_sub(1))
            .map_err(|_| JxrError::arithmetic("surface plane height"))?;
        let preceding_rows = plane
            .row_stride_bytes
            .checked_mul(row_count)
            .ok_or_else(|| JxrError::arithmetic("surface plane rows"))?;
        let required = plane
            .byte_offset
            .checked_add(preceding_rows)
            .and_then(|end| end.checked_add(row_bytes))
            .ok_or_else(|| JxrError::arithmetic("surface plane extent"))?;
        Ok((plane.byte_offset, required))
    }
}

const fn channel_layout(format: PixelFormat) -> Option<ChannelLayout> {
    match format {
        PixelFormat::BitPacked(layout)
        | PixelFormat::U8(layout)
        | PixelFormat::U16(layout)
        | PixelFormat::I16(layout)
        | PixelFormat::I32(layout)
        | PixelFormat::F16(layout)
        | PixelFormat::F32(layout) => Some(layout),
        PixelFormat::Rgb555 | PixelFormat::Rgb565 | PixelFormat::Rgb101010 | PixelFormat::Rgbe => {
            None
        }
    }
}

fn align_up(value: usize, alignment: usize) -> Result<usize, JxrError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| JxrError::arithmetic("surface alignment"))
}
