//! Coefficient unpacking, raster placement, and scalar reconstruction geometry.

use jxr_core::{
    BandPresence, CoefficientArena, CoefficientPlane, DecodeScale, JxrError, JxrErrorKind,
    PredictionMode, PreparedPlan,
};
use jxr_math::prediction::{
    HighpassPrediction, predict_high_pass, predict_high_pass_420, predict_high_pass_422,
};
use rayon::prelude::*;

use crate::{
    CpuCapabilities, ParsedCodestream, decode_coefficients,
    reconstruct::{
        ChromaReconstructionConfig, PlanarSamples, QuantizedMacroblock, ReconstructionConfig,
        ReconstructionError, ReconstructionPipelineWorkspace, TilePartition,
        reconstruct_chroma_444, reconstruct_luma_scaled_with_cpu,
    },
};

#[derive(Debug, Default)]
pub(super) struct ReconstructionWorkspace {
    components: Vec<ComponentWorkspace>,
}

#[derive(Debug, Default)]
struct ComponentWorkspace {
    macroblocks: Vec<QuantizedMacroblock>,
    metadata_by_position: Vec<usize>,
    raster_reuses: u64,
    pipeline: ReconstructionPipelineWorkspace,
}

impl ReconstructionWorkspace {
    pub(super) fn reuses(&self) -> u64 {
        self.components
            .iter()
            .map(|component| {
                component
                    .raster_reuses
                    .saturating_add(component.pipeline.reuses())
            })
            .fold(0_u64, u64::saturating_add)
    }

    pub(super) fn retained_bytes(&self) -> usize {
        self.components.iter().fold(0_usize, |total, component| {
            total
                .saturating_add(
                    component
                        .macroblocks
                        .capacity()
                        .saturating_mul(core::mem::size_of::<QuantizedMacroblock>()),
                )
                .saturating_add(
                    component
                        .metadata_by_position
                        .capacity()
                        .saturating_mul(core::mem::size_of::<usize>()),
                )
                .saturating_add(component.pipeline.retained_bytes())
        })
    }

    pub(super) fn recycle_components(&mut self, components: Vec<PlanarSamples>) {
        debug_assert!(components.len() <= self.components.len());
        for (component, workspace) in components.into_iter().zip(&mut self.components) {
            workspace.pipeline.recycle_samples(component.samples);
        }
    }

    pub(super) fn recycle_component(&mut self, index: usize, component: PlanarSamples) {
        debug_assert!(index < self.components.len());
        if let Some(workspace) = self.components.get_mut(index) {
            workspace.pipeline.recycle_samples(component.samples);
        }
    }
}

pub(super) fn reconstruct_components_from_arena(
    arena: &CoefficientArena,
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    cpu: CpuCapabilities,
    workspace: &mut ReconstructionWorkspace,
) -> Result<Vec<PlanarSamples>, JxrError> {
    let (components, _) = reconstruct_arena(arena, parsed, plan, false, cpu, workspace)?;
    Ok(components)
}

pub(super) fn reconstruct_components_and_integrated_alpha_from_arena(
    arena: &CoefficientArena,
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    cpu: CpuCapabilities,
    workspace: &mut ReconstructionWorkspace,
) -> Result<(Vec<PlanarSamples>, PlanarSamples), JxrError> {
    let (components, alpha) = reconstruct_arena(arena, parsed, plan, true, cpu, workspace)?;
    let alpha = alpha.ok_or_else(|| {
        JxrError::new(
            JxrErrorKind::InternalInvariant,
            "integrated alpha reconstruction plane",
        )
    })?;
    Ok((components, alpha))
}

pub(super) fn reconstruct_components(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    cpu: CpuCapabilities,
) -> Result<Vec<PlanarSamples>, JxrError> {
    let arena = decode_coefficients(source, parsed, plan)?;
    let mut workspace = ReconstructionWorkspace::default();
    let (components, _) = reconstruct_arena(&arena, parsed, plan, false, cpu, &mut workspace)?;
    Ok(components)
}

pub(super) fn reconstruct_components_and_integrated_alpha(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    cpu: CpuCapabilities,
) -> Result<(Vec<PlanarSamples>, PlanarSamples), JxrError> {
    let arena = decode_coefficients(source, parsed, plan)?;
    let mut workspace = ReconstructionWorkspace::default();
    let (components, alpha) = reconstruct_arena(&arena, parsed, plan, true, cpu, &mut workspace)?;
    let alpha = alpha.ok_or_else(|| {
        JxrError::new(
            JxrErrorKind::InternalInvariant,
            "integrated alpha reconstruction plane",
        )
    })?;
    Ok((components, alpha))
}

fn reconstruct_arena(
    arena: &CoefficientArena,
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    integrated_alpha: bool,
    cpu: CpuCapabilities,
    workspace: &mut ReconstructionWorkspace,
) -> Result<(Vec<PlanarSamples>, Option<PlanarSamples>), JxrError> {
    arena.validate()?;
    let expected_components = usize::from(
        plan.info
            .primary
            .color_format
            .component_count()
            .ok_or_else(|| JxrError::new(JxrErrorKind::InvalidSyntax, "primary component count"))?,
    );
    let expected_planes = expected_components + usize::from(integrated_alpha);
    if arena.planes.len() != expected_planes {
        return Err(JxrError::new(
            JxrErrorKind::InternalInvariant,
            "coefficient plane count",
        ));
    }
    if workspace.components.len() < expected_planes {
        workspace
            .components
            .resize_with(expected_planes, ComponentWorkspace::default);
    }
    let window = reconstruction_window(arena, &arena.planes[0], plan)?;
    let (component_workspaces, alpha_workspaces) =
        workspace.components.split_at_mut(expected_components);
    let mut components = if expected_components == 1 {
        vec![reconstruct_component(
            arena,
            &arena.planes[0],
            false,
            plan,
            &window,
            cpu,
            &mut component_workspaces[0],
        )?]
    } else {
        arena.planes[..expected_components]
            .par_iter()
            .zip(component_workspaces.par_iter_mut())
            .enumerate()
            .map(|(component, (plane, component_workspace))| {
                reconstruct_component(
                    arena,
                    plane,
                    component != 0
                        && plan.info.primary.scaled
                        && matches!(
                            plan.info.primary.color_format,
                            jxr_core::ColorFormat::Yuv(_)
                        ),
                    plan,
                    &window,
                    cpu,
                    component_workspace,
                )
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    components = upsample_chroma(parsed, components, plan.scale, component_workspaces)?;
    let alpha = if integrated_alpha {
        Some(reconstruct_component(
            arena,
            &arena.planes[expected_components],
            false,
            plan,
            &window,
            cpu,
            &mut alpha_workspaces[0],
        )?)
    } else {
        None
    };
    Ok((components, alpha))
}

fn upsample_chroma(
    parsed: &ParsedCodestream,
    mut components: Vec<PlanarSamples>,
    scale: DecodeScale,
    workspaces: &mut [ComponentWorkspace],
) -> Result<Vec<PlanarSamples>, JxrError> {
    if !matches!(parsed.headers.primary.internal_color_format, 1 | 2) {
        return Ok(components);
    }
    if components.len() != 3 {
        return Err(JxrError::new(
            JxrErrorKind::InternalInvariant,
            "subsampled YUV component count",
        ));
    }
    if matches!(parsed.headers.image.output_color_format, 1 | 2) || scale == DecodeScale::Sixteenth
    {
        return Ok(components);
    }
    let config = ChromaReconstructionConfig::from_header(
        &parsed.headers.primary,
        components[0].width,
        components[0].height,
    )
    .map_err(|error| map_reconstruction_error(&error))?;
    let [u, v] = reconstruct_chroma_444(&components[1], &components[2], config)
        .map_err(|error| map_reconstruction_error(&error))?;
    let old_u = core::mem::replace(&mut components[1], u);
    let old_v = core::mem::replace(&mut components[2], v);
    workspaces[1].pipeline.recycle_samples(old_u.samples);
    workspaces[2].pipeline.recycle_samples(old_v.samples);
    Ok(components)
}

fn reconstruct_component(
    arena: &CoefficientArena,
    plane: &CoefficientPlane,
    scale_after_first_transform: bool,
    plan: &PreparedPlan,
    window: &ReconstructionWindow,
    cpu: CpuCapabilities,
    workspace: &mut ComponentWorkspace,
) -> Result<PlanarSamples, JxrError> {
    let ComponentWorkspace {
        macroblocks,
        metadata_by_position,
        raster_reuses,
        pipeline,
    } = workspace;
    let macroblocks = raster_macroblocks(
        arena,
        plane,
        window,
        macroblocks,
        metadata_by_position,
        raster_reuses,
    )?;
    let config = ReconstructionConfig {
        macroblock_origin_x: window.origin_x,
        macroblock_origin_y: window.origin_y,
        macroblocks_x: window.width,
        macroblocks_y: window.height,
        block_columns: plane.block_columns,
        block_rows: plane.block_rows,
        scale_after_first_transform,
        overlap: plan.primary.overlap,
        tiles: window.tiles.clone(),
    };
    reconstruct_luma_scaled_with_cpu(macroblocks, &config, plan.scale, cpu, pipeline)
        .map_err(|error| map_reconstruction_error(&error))
}

fn raster_macroblocks<'workspace>(
    arena: &CoefficientArena,
    plane: &CoefficientPlane,
    window: &ReconstructionWindow,
    macroblocks: &'workspace mut Vec<QuantizedMacroblock>,
    metadata_by_position: &mut Vec<usize>,
    reuse_count: &mut u64,
) -> Result<&'workspace [QuantizedMacroblock], JxrError> {
    let width =
        usize::try_from(window.width).map_err(|_| JxrError::arithmetic("macroblock row width"))?;
    let height = usize::try_from(window.height)
        .map_err(|_| JxrError::arithmetic("macroblock row height"))?;
    let count = width
        .checked_mul(height)
        .ok_or_else(|| JxrError::arithmetic("macroblock raster size"))?;
    if plane.macroblock_count != count {
        return Err(JxrError::new(
            JxrErrorKind::InternalInvariant,
            "decoded component macroblock count",
        ));
    }
    if window.positions.len() != count {
        return Err(JxrError::new(
            JxrErrorKind::InternalInvariant,
            "planned macroblock traversal",
        ));
    }
    let capacities_reused =
        macroblocks.capacity() >= count && metadata_by_position.capacity() >= count;
    macroblocks.clear();
    metadata_by_position.clear();
    metadata_by_position.resize(count, usize::MAX);
    let metadata_end = plane
        .macroblock_offset
        .checked_add(plane.macroblock_count)
        .ok_or_else(|| JxrError::arithmetic("component metadata range"))?;
    for (local_index, index) in (plane.macroblock_offset..metadata_end).enumerate() {
        let (x, y) = window.positions[local_index];
        let position = y
            .checked_mul(width)
            .and_then(|row| row.checked_add(x))
            .filter(|&position| position < count)
            .ok_or_else(|| JxrError::new(JxrErrorKind::InternalInvariant, "macroblock position"))?;
        if metadata_by_position[position] != usize::MAX {
            return Err(JxrError::new(
                JxrErrorKind::InternalInvariant,
                "duplicate macroblock position",
            ));
        }
        metadata_by_position[position] = index;
    }
    for &index in metadata_by_position.iter() {
        if index == usize::MAX {
            return Err(JxrError::new(
                JxrErrorKind::InternalInvariant,
                "missing macroblock position",
            ));
        }
        macroblocks.push(unpack_macroblock(
            arena,
            index,
            usize::from(plane.block_columns),
            usize::from(plane.block_rows),
        )?);
    }
    *reuse_count = reuse_count.saturating_add(u64::from(capacities_reused));
    Ok(macroblocks)
}

fn unpack_macroblock(
    arena: &CoefficientArena,
    index: usize,
    block_columns: usize,
    block_rows: usize,
) -> Result<QuantizedMacroblock, JxrError> {
    let block_count = block_columns
        .checked_mul(block_rows)
        .ok_or_else(|| JxrError::arithmetic("component block count"))?;
    let offset = usize::try_from(arena.macroblocks.coefficient_offsets[index])
        .map_err(|_| JxrError::arithmetic("coefficient offset"))?;
    let bands = arena.macroblocks.bands[index];
    let end = offset
        .checked_add(packed_coefficient_count(bands, block_count))
        .ok_or_else(|| JxrError::arithmetic("macroblock coefficient range"))?;
    let coefficients = arena
        .coefficients
        .get(offset..end)
        .ok_or_else(|| JxrError::new(JxrErrorKind::InvalidSyntax, "macroblock coefficients"))?;
    let mut dc_low_pass = [0; 16];
    let mut high_pass = [0; 256];
    unpack_coefficients(
        coefficients,
        bands,
        block_count,
        &mut dc_low_pass,
        &mut high_pass,
    );
    apply_high_pass_prediction(
        &mut high_pass,
        arena.macroblocks.hp_predictions[index],
        block_columns,
        block_rows,
    )?;
    Ok(QuantizedMacroblock {
        dc_low_pass,
        high_pass,
        quantizers: arena.macroblocks.quantizers[index],
        bands,
    })
}

fn apply_high_pass_prediction(
    coefficients: &mut [i32; 256],
    mode: PredictionMode,
    block_columns: usize,
    block_rows: usize,
) -> Result<(), JxrError> {
    let direction = match mode {
        PredictionMode::None => HighpassPrediction::None,
        PredictionMode::FromLeft => HighpassPrediction::FromLeft,
        PredictionMode::FromTop => HighpassPrediction::FromTop,
        PredictionMode::FromTopLeft => {
            return Err(JxrError::new(
                JxrErrorKind::InternalInvariant,
                "high-pass prediction direction",
            ));
        }
    };
    let result = match (block_columns, block_rows) {
        (4, 4) => predict_high_pass(coefficients, direction),
        (2, 4) => predict_high_pass_422(coefficients, direction),
        (2, 2) => predict_high_pass_420(coefficients, direction),
        _ => {
            return Err(JxrError::new(
                JxrErrorKind::InternalInvariant,
                "high-pass prediction block geometry",
            ));
        }
    };
    result.map_err(|_| JxrError::arithmetic("high-pass coefficient prediction"))
}

struct ReconstructionWindow {
    origin_x: u32,
    origin_y: u32,
    width: u32,
    height: u32,
    positions: Vec<(usize, usize)>,
    tiles: TilePartition,
}

fn reconstruction_window(
    arena: &CoefficientArena,
    plane: &CoefficientPlane,
    plan: &PreparedPlan,
) -> Result<ReconstructionWindow, JxrError> {
    if plane.macroblock_count == 0 {
        return Err(JxrError::new(
            JxrErrorKind::InternalInvariant,
            "empty reconstruction window",
        ));
    }
    let end = plane
        .macroblock_offset
        .checked_add(plane.macroblock_count)
        .ok_or_else(|| JxrError::arithmetic("reconstruction metadata range"))?;
    let xs = arena
        .macroblocks
        .coded_x
        .get(plane.macroblock_offset..end)
        .ok_or_else(|| JxrError::new(JxrErrorKind::InternalInvariant, "coded x range"))?;
    let ys = arena
        .macroblocks
        .coded_y
        .get(plane.macroblock_offset..end)
        .ok_or_else(|| JxrError::new(JxrErrorKind::InternalInvariant, "coded y range"))?;
    let origin_x = *xs
        .iter()
        .min()
        .ok_or_else(|| JxrError::new(JxrErrorKind::InternalInvariant, "coded x minimum"))?;
    let origin_y = *ys
        .iter()
        .min()
        .ok_or_else(|| JxrError::new(JxrErrorKind::InternalInvariant, "coded y minimum"))?;
    let right = xs
        .iter()
        .max()
        .copied()
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| JxrError::arithmetic("coded x extent"))?;
    let bottom = ys
        .iter()
        .max()
        .copied()
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| JxrError::arithmetic("coded y extent"))?;
    let width = right - origin_x;
    let height = bottom - origin_y;
    let expected = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| JxrError::arithmetic("reconstruction window size"))?;
    if expected != plane.macroblock_count {
        return Err(JxrError::new(
            JxrErrorKind::InternalInvariant,
            "non-rectangular reconstruction window",
        ));
    }
    let positions = xs
        .iter()
        .zip(ys)
        .map(|(&x, &y)| {
            Ok((
                usize::try_from(x - origin_x)
                    .map_err(|_| JxrError::arithmetic("local macroblock x"))?,
                usize::try_from(y - origin_y)
                    .map_err(|_| JxrError::arithmetic("local macroblock y"))?,
            ))
        })
        .collect::<Result<Vec<_>, JxrError>>()?;
    Ok(ReconstructionWindow {
        origin_x,
        origin_y,
        width,
        height,
        positions,
        tiles: tile_partition(plan, origin_x, origin_y, right, bottom)?,
    })
}

const fn packed_coefficient_count(bands: BandPresence, block_count: usize) -> usize {
    match bands {
        BandPresence::DcOnly => 1,
        BandPresence::NoHighPass => block_count,
        BandPresence::NoFlexbits | BandPresence::All => block_count * 16,
    }
}

fn unpack_coefficients(
    packed: &[i32],
    bands: BandPresence,
    block_count: usize,
    dc_low_pass: &mut [i32; 16],
    high_pass: &mut [i32; 256],
) {
    match bands {
        BandPresence::DcOnly => dc_low_pass[0] = packed[0],
        BandPresence::NoHighPass => dc_low_pass[..block_count].copy_from_slice(packed),
        BandPresence::NoFlexbits | BandPresence::All => {
            for block in 0..block_count {
                let source = &packed[block * 16..(block + 1) * 16];
                dc_low_pass[block] = source[0];
                high_pass[block * 16 + 1..(block + 1) * 16].copy_from_slice(&source[1..]);
            }
        }
    }
}

fn tile_partition(
    plan: &PreparedPlan,
    origin_x: u32,
    origin_y: u32,
    right: u32,
    bottom: u32,
) -> Result<TilePartition, JxrError> {
    Ok(TilePartition {
        column_boundaries: clipped_boundaries(
            &plan.info.tiles.column_widths,
            plan.primary.macroblocks_x,
            origin_x,
            right,
            "tile column boundaries",
        )?,
        row_boundaries: clipped_boundaries(
            &plan.info.tiles.row_heights,
            plan.primary.macroblocks_y,
            origin_y,
            bottom,
            "tile row boundaries",
        )?,
        hard_boundaries: plan.info.tiles.hard_tiles,
    })
}

fn clipped_boundaries(
    sizes: &[u32],
    expected: u32,
    start: u32,
    end: u32,
    operation: &'static str,
) -> Result<Vec<u32>, JxrError> {
    let mut source = Vec::with_capacity(sizes.len() + 1);
    source.push(0);
    let mut current = 0_u32;
    for &size in sizes {
        current = current
            .checked_add(size)
            .ok_or_else(|| JxrError::arithmetic(operation))?;
        source.push(current);
    }
    if current != expected || start >= end || end > expected {
        return Err(JxrError::new(JxrErrorKind::InternalInvariant, operation));
    }
    let mut boundaries = Vec::with_capacity(source.len());
    boundaries.push(0);
    for boundary in source {
        if boundary > start && boundary < end {
            boundaries.push(boundary - start);
        }
    }
    boundaries.push(end - start);
    Ok(boundaries)
}

fn map_reconstruction_error(error: &ReconstructionError) -> JxrError {
    let kind = match error {
        ReconstructionError::ArithmeticOverflow(_) => JxrErrorKind::ArithmeticOverflow,
        ReconstructionError::Unsupported(_) => JxrErrorKind::Unsupported,
        ReconstructionError::BufferTooSmall {
            required,
            available,
        } => JxrErrorKind::BufferTooSmall {
            required: *required,
            available: *available,
        },
        ReconstructionError::MacroblockCount { .. }
        | ReconstructionError::InvalidPlaneGeometry(_)
        | ReconstructionError::ZeroQuantizer(_)
        | ReconstructionError::InvalidTilePartition(_)
        | ReconstructionError::CropOutsidePlane => JxrErrorKind::InvalidSyntax,
    };
    JxrError::new(kind, "reconstruct samples")
}

#[cfg(test)]
mod tests {
    use super::clipped_boundaries;

    #[test]
    fn regional_tile_boundaries_are_rebased_and_clipped() {
        assert_eq!(
            clipped_boundaries(&[2, 3, 1], 6, 1, 5, "test boundaries").unwrap(),
            [0, 1, 4]
        );
    }
}
