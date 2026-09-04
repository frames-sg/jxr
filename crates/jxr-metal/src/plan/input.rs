// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use jxr_core::{AlphaMode, CoefficientArena, ColorFormat, PreparedPlan};

use super::{
    MetalArenaInput, MetalCoefficientSource, MetalError, MetalPlaneInput, build_plane, invalid,
    validate_arena,
};

pub(super) fn build_arenas(
    primary: Arc<CoefficientArena>,
    separate_alpha: Option<(Arc<CoefficientArena>, PreparedPlan)>,
    prepared: &PreparedPlan,
    expected_primary_planes: usize,
) -> Result<Vec<MetalArenaInput>, MetalError> {
    validate_arena(&primary)?;
    if primary.planes.len() != expected_primary_planes {
        return Err(invalid(
            "primary coefficient arena component count does not match the image",
        ));
    }
    let mut arenas = vec![MetalArenaInput {
        source: MetalCoefficientSource::Cpu(primary),
    }];
    if let Some((coefficients, alpha_plan)) = separate_alpha {
        validate_arena(&coefficients)?;
        if coefficients.planes.len() != 1 || alpha_plan.info.alpha_mode != AlphaMode::None {
            return Err(invalid(
                "separate alpha arena is not one independent luma plane",
            ));
        }
        if alpha_plan.output_region != prepared.output_region {
            return Err(invalid("separate alpha output region differs from primary"));
        }
        arenas.push(MetalArenaInput {
            source: MetalCoefficientSource::Cpu(coefficients),
        });
    }
    Ok(arenas)
}

pub(super) fn build_planes(
    arenas: &[MetalArenaInput],
    prepared: &PreparedPlan,
    primary_components: usize,
    expected_primary_planes: usize,
    integrated_alpha: bool,
) -> Result<(Vec<MetalPlaneInput>, usize, usize), MetalError> {
    let mut planes = Vec::with_capacity(expected_primary_planes + usize::from(arenas.len() > 1));
    let mut low_len = 0;
    let mut sample_len = 0;
    for plane_index in 0..expected_primary_planes {
        let alpha = integrated_alpha && plane_index == primary_components;
        let scale_chroma = !alpha
            && plane_index != 0
            && prepared.info.primary.scaled
            && matches!(prepared.info.primary.color_format, ColorFormat::Yuv(_));
        planes.push(build_plane(
            &arenas[0],
            plane_index,
            0,
            scale_chroma,
            alpha,
            &mut low_len,
            &mut sample_len,
        )?);
    }
    if arenas.len() > 1 {
        planes.push(build_plane(
            &arenas[1],
            0,
            1,
            false,
            true,
            &mut low_len,
            &mut sample_len,
        )?);
    }
    Ok((planes, low_len, sample_len))
}
