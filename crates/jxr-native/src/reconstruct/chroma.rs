//! Full-resolution chroma reconstruction for T.832 4:2:0 and 4:2:2 planes.

use jxr_core::ChromaSampling;
use jxr_math::sampling::{ChromaCentering, upsample_chroma_pair};

use crate::ImagePlaneHeader;

use super::{PlanarSamples, ReconstructionError};

/// Geometry and parsed chroma-grid positioning for one reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChromaReconstructionConfig {
    /// Coded chroma subsampling mode.
    pub sampling: ChromaSampling,
    /// Full-resolution luma width in samples, including coded margins.
    pub full_width: u32,
    /// Full-resolution luma height in samples, including coded margins.
    pub full_height: u32,
    /// Horizontal quarter-sample chroma-grid offset.
    pub centering_x: u8,
    /// Vertical quarter-sample chroma-grid offset.
    pub centering_y: u8,
}

impl ChromaReconstructionConfig {
    /// Build reconstruction geometry directly from a parsed image-plane header.
    pub fn from_header(
        header: &ImagePlaneHeader,
        full_width: u32,
        full_height: u32,
    ) -> Result<Self, ReconstructionError> {
        let sampling = match header.internal_color_format {
            1 => ChromaSampling::Cs420,
            2 => ChromaSampling::Cs422,
            _ => {
                return Err(ReconstructionError::Unsupported(
                    "chroma reconstruction requires YUV420 or YUV422",
                ));
            }
        };
        Ok(Self {
            sampling,
            full_width,
            full_height,
            centering_x: header.chroma_centering_x,
            centering_y: header.chroma_centering_y,
        })
    }
}

/// Reconstruct U and V to full resolution using the T.832 example filter.
///
/// The returned planes are YUV444 geometry and can be passed directly to the
/// output formatter together with a full-resolution luma plane.
pub fn reconstruct_chroma_444(
    u: &PlanarSamples,
    v: &PlanarSamples,
    config: ChromaReconstructionConfig,
) -> Result<[PlanarSamples; 2], ReconstructionError> {
    validate_matching_chroma(u, v)?;
    Ok([reconstruct_plane(u, config)?, reconstruct_plane(v, config)?])
}

fn reconstruct_plane(
    input: &PlanarSamples,
    config: ChromaReconstructionConfig,
) -> Result<PlanarSamples, ReconstructionError> {
    validate_plane(input)?;
    let centering_x = centering(config.centering_x, "horizontal chroma centering")?;
    match config.sampling {
        ChromaSampling::Cs422 => {
            validate_geometry(input, config.full_width, config.full_height, true, false)?;
            upsample_horizontal(input, centering_x)
        }
        ChromaSampling::Cs420 => {
            validate_geometry(input, config.full_width, config.full_height, true, true)?;
            let centering_y = centering(config.centering_y, "vertical chroma centering")?;
            let vertical = upsample_vertical(input, centering_y)?;
            upsample_horizontal(&vertical, centering_x)
        }
        ChromaSampling::Cs444 => Err(ReconstructionError::Unsupported(
            "YUV444 chroma does not require upsampling",
        )),
    }
}

fn upsample_horizontal(
    input: &PlanarSamples,
    centering: ChromaCentering,
) -> Result<PlanarSamples, ReconstructionError> {
    let input_width = usize::try_from(input.width)
        .map_err(|_| ReconstructionError::ArithmeticOverflow("chroma width conversion"))?;
    let height = usize::try_from(input.height)
        .map_err(|_| ReconstructionError::ArithmeticOverflow("chroma height conversion"))?;
    let output_width =
        input_width
            .checked_mul(2)
            .ok_or(ReconstructionError::ArithmeticOverflow(
                "upsampled chroma width",
            ))?;
    let capacity =
        output_width
            .checked_mul(height)
            .ok_or(ReconstructionError::ArithmeticOverflow(
                "horizontal chroma allocation",
            ))?;
    let mut samples = Vec::with_capacity(capacity);
    for row in input.samples.chunks_exact(input_width) {
        for index in 0..input_width {
            let previous = row[index.saturating_sub(1)];
            let current = row[index];
            let next = row[(index + 1).min(input_width - 1)];
            samples.extend_from_slice(
                &upsample_chroma_pair(previous, current, next, centering).map_err(|_| {
                    ReconstructionError::ArithmeticOverflow("horizontal chroma interpolation")
                })?,
            );
        }
    }
    Ok(PlanarSamples {
        origin_x: input
            .origin_x
            .checked_mul(2)
            .ok_or(ReconstructionError::ArithmeticOverflow(
                "upsampled x origin",
            ))?,
        origin_y: input.origin_y,
        width: u32::try_from(output_width)
            .map_err(|_| ReconstructionError::ArithmeticOverflow("upsampled width conversion"))?,
        height: input.height,
        samples,
    })
}

fn upsample_vertical(
    input: &PlanarSamples,
    centering: ChromaCentering,
) -> Result<PlanarSamples, ReconstructionError> {
    let width = usize::try_from(input.width)
        .map_err(|_| ReconstructionError::ArithmeticOverflow("chroma width conversion"))?;
    let input_height = usize::try_from(input.height)
        .map_err(|_| ReconstructionError::ArithmeticOverflow("chroma height conversion"))?;
    let output_height =
        input_height
            .checked_mul(2)
            .ok_or(ReconstructionError::ArithmeticOverflow(
                "upsampled chroma height",
            ))?;
    let length =
        width
            .checked_mul(output_height)
            .ok_or(ReconstructionError::ArithmeticOverflow(
                "vertical chroma allocation",
            ))?;
    let mut samples = vec![0; length];
    for row in 0..input_height {
        let previous_row = row.saturating_sub(1);
        let next_row = (row + 1).min(input_height - 1);
        for column in 0..width {
            let output = upsample_chroma_pair(
                input.samples[previous_row * width + column],
                input.samples[row * width + column],
                input.samples[next_row * width + column],
                centering,
            )
            .map_err(|_| {
                ReconstructionError::ArithmeticOverflow("vertical chroma interpolation")
            })?;
            samples[(row * 2) * width + column] = output[0];
            samples[(row * 2 + 1) * width + column] = output[1];
        }
    }
    Ok(PlanarSamples {
        origin_x: input.origin_x,
        origin_y: input
            .origin_y
            .checked_mul(2)
            .ok_or(ReconstructionError::ArithmeticOverflow(
                "upsampled y origin",
            ))?,
        width: input.width,
        height: u32::try_from(output_height)
            .map_err(|_| ReconstructionError::ArithmeticOverflow("upsampled height conversion"))?,
        samples,
    })
}

fn validate_matching_chroma(
    u: &PlanarSamples,
    v: &PlanarSamples,
) -> Result<(), ReconstructionError> {
    if u.width != v.width
        || u.height != v.height
        || u.origin_x != v.origin_x
        || u.origin_y != v.origin_y
    {
        return Err(ReconstructionError::InvalidPlaneGeometry(
            "U and V dimensions differ",
        ));
    }
    Ok(())
}

fn validate_plane(plane: &PlanarSamples) -> Result<(), ReconstructionError> {
    if plane.width == 0 || plane.height == 0 {
        return Err(ReconstructionError::InvalidPlaneGeometry(
            "chroma plane is empty",
        ));
    }
    let expected = usize::try_from(plane.width)
        .ok()
        .and_then(|width| {
            usize::try_from(plane.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or(ReconstructionError::ArithmeticOverflow(
            "chroma plane extent",
        ))?;
    if plane.samples.len() < expected {
        return Err(ReconstructionError::BufferTooSmall {
            required: expected,
            available: plane.samples.len(),
        });
    }
    if plane.samples.len() > expected {
        return Err(ReconstructionError::InvalidPlaneGeometry(
            "chroma plane has trailing samples",
        ));
    }
    Ok(())
}

fn validate_geometry(
    input: &PlanarSamples,
    full_width: u32,
    full_height: u32,
    double_width: bool,
    double_height: bool,
) -> Result<(), ReconstructionError> {
    let width_factor = if double_width { 2 } else { 1 };
    let height_factor = if double_height { 2 } else { 1 };
    let expected_width = input
        .width
        .checked_mul(width_factor)
        .ok_or(ReconstructionError::ArithmeticOverflow("full chroma width"))?;
    let expected_height =
        input
            .height
            .checked_mul(height_factor)
            .ok_or(ReconstructionError::ArithmeticOverflow(
                "full chroma height",
            ))?;
    if expected_width != full_width || expected_height != full_height {
        return Err(ReconstructionError::InvalidPlaneGeometry(
            "subsampled chroma dimensions do not match luma geometry",
        ));
    }
    Ok(())
}

fn centering(value: u8, operation: &'static str) -> Result<ChromaCentering, ReconstructionError> {
    ChromaCentering::new(value).ok_or(ReconstructionError::Unsupported(operation))
}

#[cfg(test)]
mod tests {
    use jxr_core::ChromaSampling;

    use super::{ChromaReconstructionConfig, PlanarSamples, reconstruct_chroma_444};

    fn plane(width: u32, height: u32, samples: &[i32]) -> PlanarSamples {
        PlanarSamples {
            origin_x: 0,
            origin_y: 0,
            width,
            height,
            samples: samples.to_vec(),
        }
    }

    #[test]
    fn reconstructs_422_horizontally_with_clamped_edges() {
        let input = plane(2, 1, &[8, 16]);
        let [u, v] = reconstruct_chroma_444(
            &input,
            &input,
            ChromaReconstructionConfig {
                sampling: ChromaSampling::Cs422,
                full_width: 4,
                full_height: 1,
                centering_x: 0,
                centering_y: 0,
            },
        )
        .unwrap();
        assert_eq!(u.samples, [8, 12, 16, 16]);
        assert_eq!(v, u);
    }

    #[test]
    fn reconstructs_420_vertically_then_horizontally() {
        let input = plane(1, 2, &[0, 16]);
        let [u, _] = reconstruct_chroma_444(
            &input,
            &input,
            ChromaReconstructionConfig {
                sampling: ChromaSampling::Cs420,
                full_width: 2,
                full_height: 4,
                centering_x: 0,
                centering_y: 0,
            },
        )
        .unwrap();
        assert_eq!(u.samples, [0, 0, 8, 8, 16, 16, 16, 16]);
    }

    #[test]
    fn rejects_unknown_centering_and_mismatched_geometry() {
        let input = plane(1, 1, &[0]);
        let mut config = ChromaReconstructionConfig {
            sampling: ChromaSampling::Cs422,
            full_width: 2,
            full_height: 1,
            centering_x: 7,
            centering_y: 0,
        };
        assert!(reconstruct_chroma_444(&input, &input, config).is_err());
        config.centering_x = 0;
        config.full_width = 3;
        assert!(reconstruct_chroma_444(&input, &input, config).is_err());
    }
}
