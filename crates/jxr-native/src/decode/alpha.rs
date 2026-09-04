//! Separate-alpha planning and policy for the scalar decode route.

use jxr_core::{
    AlphaHandling, AlphaMode, BackendRequest, ChannelLayout, DecodeRequest, JxrError, JxrErrorKind,
    PixelFormat, PreparedPlan,
};

use crate::CpuCapabilities;
use crate::{ParsedCodestream, prepare_plan, reconstruct::PlanarSamples};

use super::reconstruction::{
    ReconstructionWorkspace, reconstruct_components, reconstruct_components_from_arena,
};

pub(super) fn reconstruct_separate_alpha(
    source: &[u8],
    parsed: &ParsedCodestream,
    primary_plan: &PreparedPlan,
    request: &DecodeRequest,
    cpu: CpuCapabilities,
) -> Result<PlanarSamples, JxrError> {
    let (alpha_parsed, alpha_plan) = prepare_separate_alpha(source, parsed, primary_plan, request)?;
    take_alpha_component(reconstruct_components(
        source,
        &alpha_parsed,
        &alpha_plan,
        cpu,
    )?)
}

pub(super) fn reconstruct_separate_alpha_with_workspace(
    source: &[u8],
    parsed: &ParsedCodestream,
    primary_plan: &PreparedPlan,
    request: &DecodeRequest,
    cpu: CpuCapabilities,
    arena: &mut jxr_core::CoefficientArena,
    workspace: &mut ReconstructionWorkspace,
) -> Result<(PlanarSamples, bool), JxrError> {
    let (alpha_parsed, alpha_plan) = prepare_separate_alpha(source, parsed, primary_plan, request)?;
    let reused =
        crate::coefficient::decode_coefficients_reusing(source, &alpha_parsed, &alpha_plan, arena)?;
    let components =
        reconstruct_components_from_arena(arena, &alpha_parsed, &alpha_plan, cpu, workspace)?;
    Ok((take_alpha_component(components)?, reused))
}

pub(crate) fn prepare_separate_alpha_coefficients(
    source: &[u8],
    parsed: &ParsedCodestream,
    primary_plan: &jxr_core::PreparedPlan,
    request: &DecodeRequest,
) -> Result<Option<(jxr_core::PreparedPlan, jxr_core::CoefficientArena)>, JxrError> {
    if primary_plan.info.alpha_mode != AlphaMode::Separate || request.alpha == AlphaHandling::Drop {
        return Ok(None);
    }
    let (alpha_parsed, alpha_plan) = prepare_separate_alpha(source, parsed, primary_plan, request)?;
    let coefficients = crate::decode_coefficients(source, &alpha_parsed, &alpha_plan)?;
    Ok(Some((alpha_plan, coefficients)))
}

fn prepare_separate_alpha(
    source: &[u8],
    parsed: &ParsedCodestream,
    primary_plan: &PreparedPlan,
    request: &DecodeRequest,
) -> Result<(ParsedCodestream, PreparedPlan), JxrError> {
    let alpha_parsed = separate_alpha_codestream(parsed)?;
    validate_separate_alpha_compatibility(parsed, &alpha_parsed)?;
    let mut alpha_request = request.clone();
    alpha_request.alpha = AlphaHandling::Drop;
    alpha_request.backend = BackendRequest::Cpu;
    alpha_request.output = luma_output(request.output)?;
    let alpha_plan = prepare_plan(source.len(), &alpha_parsed, &alpha_request)?;
    if alpha_plan.output_region != primary_plan.output_region {
        return Err(JxrError::new(
            JxrErrorKind::InvalidSyntax,
            "separate alpha output geometry",
        ));
    }
    Ok((alpha_parsed, alpha_plan))
}

fn take_alpha_component(mut components: Vec<PlanarSamples>) -> Result<PlanarSamples, JxrError> {
    if components.len() != 1 {
        return Err(JxrError::new(
            JxrErrorKind::InternalInvariant,
            "separate alpha component count",
        ));
    }
    Ok(components.remove(0))
}

pub(super) fn validate_alpha_policy(
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    handling: AlphaHandling,
) -> Result<bool, JxrError> {
    let decode = alpha_action(plan.info.alpha_mode, handling);
    if decode {
        let prepared = match plan.info.alpha_mode {
            AlphaMode::Integrated => parsed.headers.alpha.is_some() && plan.alpha.is_some(),
            AlphaMode::Separate => {
                parsed.separate_alpha_headers.is_some()
                    && parsed.separate_alpha_directory.is_some()
                    && plan.alpha.is_some()
            }
            AlphaMode::None => true,
        };
        if !prepared {
            return Err(JxrError::new(
                JxrErrorKind::InternalInvariant,
                "prepared alpha plane",
            ));
        }
    }
    Ok(decode)
}

fn alpha_action(mode: AlphaMode, handling: AlphaHandling) -> bool {
    mode != AlphaMode::None && handling != AlphaHandling::Drop
}

fn separate_alpha_codestream(parsed: &ParsedCodestream) -> Result<ParsedCodestream, JxrError> {
    let range = parsed
        .annex_a
        .as_ref()
        .and_then(|annex| annex.alpha_range.clone())
        .ok_or_else(|| JxrError::new(JxrErrorKind::InternalInvariant, "separate alpha range"))?;
    let headers = parsed
        .separate_alpha_headers
        .clone()
        .ok_or_else(|| JxrError::new(JxrErrorKind::InternalInvariant, "separate alpha headers"))?;
    let directory = parsed.separate_alpha_directory.clone().ok_or_else(|| {
        JxrError::new(JxrErrorKind::InternalInvariant, "separate alpha directory")
    })?;
    Ok(ParsedCodestream {
        codestream_range: range,
        headers,
        directory,
        separate_alpha_headers: None,
        separate_alpha_directory: None,
        annex_a: None,
    })
}

fn validate_separate_alpha_compatibility(
    primary: &ParsedCodestream,
    alpha: &ParsedCodestream,
) -> Result<(), JxrError> {
    let primary_image = &primary.headers.image;
    let alpha_image = &alpha.headers.image;
    let alpha_plane = &alpha.headers.primary;
    if alpha_image.width != primary_image.width
        || alpha_image.height != primary_image.height
        || alpha_image.margins != primary_image.margins
    {
        return Err(JxrError::new(
            JxrErrorKind::InvalidSyntax,
            "separate alpha coded geometry",
        ));
    }
    if alpha_image.output_bit_depth != primary_image.output_bit_depth
        || alpha_plane.shift_bits != primary.headers.primary.shift_bits
        || alpha_plane.mantissa_length != primary.headers.primary.mantissa_length
        || alpha_plane.exponent_bias != primary.headers.primary.exponent_bias
        || alpha_plane.scaled != primary.headers.primary.scaled
    {
        return Err(JxrError::new(
            JxrErrorKind::InvalidSyntax,
            "separate alpha output depth",
        ));
    }
    if alpha_image.output_color_format != 0
        || alpha_plane.internal_color_format != 0
        || alpha_plane.components != 1
        || alpha.headers.alpha.is_some()
    {
        return Err(JxrError::new(
            JxrErrorKind::Unsupported,
            "separate alpha color format",
        ));
    }
    Ok(())
}

fn luma_output(output: PixelFormat) -> Result<PixelFormat, JxrError> {
    match output {
        PixelFormat::BitPacked(_) => Ok(PixelFormat::BitPacked(ChannelLayout::Luma)),
        PixelFormat::U8(_) => Ok(PixelFormat::U8(ChannelLayout::Luma)),
        PixelFormat::U16(_) => Ok(PixelFormat::U16(ChannelLayout::Luma)),
        PixelFormat::I16(_) => Ok(PixelFormat::I16(ChannelLayout::Luma)),
        PixelFormat::I32(_) => Ok(PixelFormat::I32(ChannelLayout::Luma)),
        PixelFormat::F16(_) => Ok(PixelFormat::F16(ChannelLayout::Luma)),
        PixelFormat::F32(_) => Ok(PixelFormat::F32(ChannelLayout::Luma)),
        PixelFormat::Rgb555 | PixelFormat::Rgb565 | PixelFormat::Rgb101010 | PixelFormat::Rgbe => {
            Err(JxrError::new(
                JxrErrorKind::Unsupported,
                "alpha with packed pixel output",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use jxr_core::{AlphaHandling, AlphaMode, ChannelLayout, PixelFormat};

    use super::{alpha_action, luma_output};

    #[test]
    fn separate_alpha_policy_drops_preserves_and_premultiplies() {
        assert!(!alpha_action(AlphaMode::Separate, AlphaHandling::Drop));
        assert!(alpha_action(AlphaMode::Separate, AlphaHandling::Preserve));
        assert!(alpha_action(
            AlphaMode::Separate,
            AlphaHandling::Premultiply
        ));
    }

    #[test]
    fn integrated_alpha_policy_matches_separate_alpha() {
        assert!(!alpha_action(AlphaMode::Integrated, AlphaHandling::Drop));
        assert!(alpha_action(AlphaMode::Integrated, AlphaHandling::Preserve));
        assert!(alpha_action(
            AlphaMode::Integrated,
            AlphaHandling::Premultiply
        ));
    }

    #[test]
    fn alpha_planning_preserves_storage_but_uses_luma_layout() {
        assert_eq!(
            luma_output(PixelFormat::U16(ChannelLayout::Rgba)).unwrap(),
            PixelFormat::U16(ChannelLayout::Luma)
        );
        assert!(luma_output(PixelFormat::Rgb565).is_err());
    }
}
