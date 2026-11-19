//! Coefficient unpacking, raster placement, and scalar reconstruction geometry.

use core::ops::Range;

use jxr_core::{
    BandPresence, CoefficientArena, CoefficientPlane, DecodeScale, JxrError, JxrErrorKind,
    PredictionMode, PreparedPlan, scaled_after_first_transform,
};
use jxr_math::prediction::{
    HighpassPrediction, predict_high_pass, predict_high_pass_420, predict_high_pass_422,
};

use crate::{
    CpuCapabilities, ParsedCodestream,
    metadata::interpolates_chroma,
    reconstruct::{
        ChromaReconstructionConfig, ChromaStrip, ComponentStrip, MacroblockSource, PlanarSamples,
        QuantizedMacroblock, ReconstructionConfig, ReconstructionError,
        ReconstructionPipelineWorkspace, ReconstructionSimd, TilePartition, reconstruct_chroma_444,
        reconstruct_luma_scaled_with_cpu,
    },
    tile_decode::coefficients_per_macroblock,
};

/// Retained reconstruction storage of one codestream's planes: its primary
/// components, then an integrated alpha plane.
#[derive(Debug, Default)]
pub(super) struct ReconstructionWorkspace {
    pub(super) components: Vec<ComponentWorkspace>,
    active_components: usize,
    /// Full-resolution interpolation of subsampled U and V.
    pub(super) chroma: [ChromaStrip; 2],
    /// Where each macroblock position of a band is in its coefficient plane.
    pub(super) positions: Vec<u32>,
}

#[derive(Debug, Default)]
pub(super) struct ComponentWorkspace {
    /// The strip a full-resolution decode streams the component through.
    pub(super) strip: ComponentStrip,
    strip_reuses: u64,
    /// Raster macroblocks and transform storage of a reduced decode.
    macroblocks: Vec<QuantizedMacroblock>,
    raster_reuses: u64,
    pipeline: ReconstructionPipelineWorkspace,
}

impl ComponentWorkspace {
    /// Begins the component's strip, counting storage kept from an earlier
    /// decode as a reuse.
    pub(super) fn begin_strip(
        &mut self,
        config: &ReconstructionConfig,
        cpu: CpuCapabilities,
        first_row: usize,
        held_rows: usize,
    ) -> Result<(), JxrError> {
        let reused = self.strip.retained_bytes() > 0;
        self.strip
            .begin(config, Some(cpu), first_row, held_rows)
            .map_err(|error| map_reconstruction_error(&error))?;
        self.strip_reuses = self.strip_reuses.saturating_add(u64::from(reused));
        Ok(())
    }
}

impl ReconstructionWorkspace {
    pub(super) fn reuses(&self) -> u64 {
        self.components[..self.active_components]
            .iter()
            .map(|component| {
                component
                    .raster_reuses
                    .saturating_add(component.strip_reuses)
                    .saturating_add(component.pipeline.reuses())
            })
            .fold(0_u64, u64::saturating_add)
    }

    pub(super) fn retained_bytes(&self) -> usize {
        let chroma = self.chroma.iter().fold(0_usize, |total, chroma| {
            total.saturating_add(chroma.retained_bytes())
        });
        self.components
            .iter()
            .fold(chroma, |total, component| {
                total
                    .saturating_add(
                        component
                            .macroblocks
                            .capacity()
                            .saturating_mul(core::mem::size_of::<QuantizedMacroblock>()),
                    )
                    .saturating_add(component.strip.retained_bytes())
                    .saturating_add(component.pipeline.retained_bytes())
            })
            .saturating_add(
                self.positions
                    .capacity()
                    .saturating_mul(core::mem::size_of::<u32>()),
            )
    }

    pub(super) fn reduced_simd(&self) -> ReconstructionSimd {
        self.components[..self.active_components].iter().fold(
            ReconstructionSimd::default(),
            |mut simd, component| {
                simd.include(component.pipeline.simd());
                simd
            },
        )
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

    /// Makes room for `planes` components.
    pub(super) fn prepare(&mut self, planes: usize) {
        if self.components.len() < planes {
            self.components
                .resize_with(planes, ComponentWorkspace::default);
        }
        self.active_components = planes;
    }
}

/// Primary components of the planes a codestream's coefficients describe,
/// after checking that an arena has one plane for each of them.
pub(super) fn primary_component_count(plan: &PreparedPlan) -> Result<usize, JxrError> {
    plan.info
        .primary
        .color_format
        .component_count()
        .map(usize::from)
        .ok_or_else(|| JxrError::new(JxrErrorKind::InvalidSyntax, "primary component count"))
}

/// Reduced-resolution reconstruction of every plane of a coefficient arena
/// that holds the whole reconstruction window.
pub(super) fn reconstruct_reduced(
    arena: &CoefficientArena,
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    integrated_alpha: bool,
    cpu: CpuCapabilities,
    workspace: &mut ReconstructionWorkspace,
) -> Result<(Vec<PlanarSamples>, Option<PlanarSamples>), JxrError> {
    arena.validate()?;
    let expected_components = primary_component_count(plan)?;
    let expected_planes = expected_components + usize::from(integrated_alpha);
    if arena.planes.len() != expected_planes {
        return Err(JxrError::new(
            JxrErrorKind::InternalInvariant,
            "coefficient plane count",
        ));
    }
    let window = Window::of_plan(plan)?;
    workspace.prepare(expected_planes);
    let ReconstructionWorkspace {
        components: workspaces,
        positions,
        ..
    } = workspace;
    let band = Band::new(arena, &window, 0..window.rows(), positions)?;
    let (component_workspaces, alpha_workspaces) = workspaces.split_at_mut(expected_components);
    let mut components = if expected_components == 1 {
        vec![reconstruct_component(
            &band,
            0,
            false,
            plan,
            &window,
            cpu,
            &mut component_workspaces[0],
        )?]
    } else {
        let components = component_workspaces.iter_mut().enumerate().collect();
        cpu.executor()
            .try_map(components, |(component, component_workspace)| {
                reconstruct_component(
                    &band,
                    component,
                    scaled_after_first_transform(plan.info.primary.scaled, component),
                    plan,
                    &window,
                    cpu,
                    component_workspace,
                )
            })?
    };
    components = upsample_chroma(parsed, components, plan.scale, component_workspaces)?;
    let alpha = if integrated_alpha {
        Some(reconstruct_component(
            &band,
            expected_components,
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
    let headers = &parsed.headers;
    let interpolated = interpolates_chroma(
        headers.primary.internal_color_format,
        headers.image.output_color_format,
        scale,
    );
    if !interpolated {
        return Ok(components);
    }
    if components.len() != 3 {
        return Err(JxrError::new(
            JxrErrorKind::InternalInvariant,
            "subsampled YUV component count",
        ));
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
    band: &Band<'_>,
    plane: usize,
    scale_after_first_transform: bool,
    plan: &PreparedPlan,
    window: &Window,
    cpu: CpuCapabilities,
    workspace: &mut ComponentWorkspace,
) -> Result<PlanarSamples, JxrError> {
    let source = band.plane(plane)?;
    let blocks = [source.plane.block_columns, source.plane.block_rows];
    let config = window.config(blocks, scale_after_first_transform, plan);
    let count = window.columns() * window.rows();
    let reused = workspace.macroblocks.capacity() >= count;
    workspace.macroblocks.clear();
    for y in 0..window.rows() {
        for x in 0..window.columns() {
            let mut macroblock = QuantizedMacroblock::ZERO;
            source.unpack(x, y, &mut macroblock)?;
            workspace.macroblocks.push(macroblock);
        }
    }
    workspace.raster_reuses = workspace.raster_reuses.saturating_add(u64::from(reused));
    reconstruct_luma_scaled_with_cpu(
        &workspace.macroblocks,
        &config,
        plan.scale,
        cpu,
        &mut workspace.pipeline,
    )
    .map_err(|error| map_reconstruction_error(&error))
}

/// Marks a position of a band that no macroblock has been found for yet.
const UNFILLED: u32 = u32::MAX;

/// The macroblocks of a coefficient arena that holds some macroblock rows
/// of a reconstruction window, addressed by position in the window.
///
/// An arena stores each plane tile by tile. Every plane of an arena has the
/// same layout, so one table of where each position is within a plane
/// serves them all.
pub(super) struct Band<'a> {
    arena: &'a CoefficientArena,
    /// Macroblock rows of the window the arena holds.
    rows: Range<usize>,
    /// Macroblocks across the window.
    columns: usize,
    positions: &'a [u32],
}

impl<'a> Band<'a> {
    /// Locates the macroblocks of the rows `rows` of `window` in `arena`,
    /// which must hold exactly those rows. `positions` is scratch storage.
    pub(super) fn new(
        arena: &'a CoefficientArena,
        window: &Window,
        rows: Range<usize>,
        positions: &'a mut Vec<u32>,
    ) -> Result<Self, JxrError> {
        let columns = window.columns();
        let count = columns
            .checked_mul(rows.len())
            .ok_or_else(|| JxrError::arithmetic("macroblock raster size"))?;
        let plane = arena.planes.first().ok_or_else(|| {
            JxrError::new(
                JxrErrorKind::InternalInvariant,
                "empty reconstruction window",
            )
        })?;
        if plane.macroblock_count != count
            || arena
                .planes
                .iter()
                .any(|other| other.macroblock_count != count)
        {
            return Err(JxrError::new(
                JxrErrorKind::InternalInvariant,
                "decoded component macroblock count",
            ));
        }
        let end = plane
            .macroblock_offset
            .checked_add(count)
            .ok_or_else(|| JxrError::arithmetic("component metadata range"))?;
        let coded = |values: &'a [u32], operation| {
            values
                .get(plane.macroblock_offset..end)
                .ok_or_else(|| JxrError::new(JxrErrorKind::InternalInvariant, operation))
        };
        let xs = coded(&arena.macroblocks.coded_x, "coded x range")?;
        let ys = coded(&arena.macroblocks.coded_y, "coded y range")?;
        positions.clear();
        positions.resize(count, UNFILLED);
        let first_row = window.origin_y as usize + rows.start;
        for (local, (&x, &y)) in xs.iter().zip(ys).enumerate() {
            let position = (x as usize)
                .checked_sub(window.origin_x as usize)
                .filter(|&x| x < columns)
                .zip((y as usize).checked_sub(first_row))
                .map(|(x, y)| y * columns + x)
                .filter(|&position| position < count)
                .ok_or_else(|| {
                    JxrError::new(JxrErrorKind::InternalInvariant, "macroblock position")
                })?;
            if positions[position] != UNFILLED {
                return Err(JxrError::new(
                    JxrErrorKind::InternalInvariant,
                    "duplicate macroblock position",
                ));
            }
            positions[position] = u32::try_from(local)
                .ok()
                .filter(|&local| local != UNFILLED)
                .ok_or_else(|| JxrError::arithmetic("macroblock position"))?;
        }
        // As many distinct positions as there are places: every one filled.
        Ok(Self {
            arena,
            rows,
            columns,
            positions,
        })
    }

    /// The macroblocks of coefficient plane `plane`.
    pub(super) fn plane(&self, plane: usize) -> Result<BandPlane<'_>, JxrError> {
        let plane = self.arena.planes.get(plane).ok_or_else(|| {
            JxrError::new(JxrErrorKind::InternalInvariant, "coefficient plane count")
        })?;
        Ok(BandPlane { band: self, plane })
    }
}

/// One component's macroblocks in a [`Band`].
pub(super) struct BandPlane<'a> {
    band: &'a Band<'a>,
    plane: &'a CoefficientPlane,
}

impl BandPlane<'_> {
    /// Unpacks the macroblock in column `x` of macroblock row `y` of the
    /// window into `macroblock` and resolves its high-pass prediction, as
    /// [`MacroblockSource::macroblock`] describes.
    fn unpack(
        &self,
        x: usize,
        y: usize,
        macroblock: &mut QuantizedMacroblock,
    ) -> Result<(), JxrError> {
        let band = self.band;
        let local = band.positions[(y - band.rows.start) * band.columns + x];
        unpack_macroblock(
            band.arena,
            self.plane.macroblock_offset + local as usize,
            usize::from(self.plane.block_columns),
            usize::from(self.plane.block_rows),
            macroblock,
        )
    }
}

impl MacroblockSource for BandPlane<'_> {
    #[inline]
    fn macroblock<'a>(
        &'a self,
        x: usize,
        y: usize,
        scratch: &'a mut QuantizedMacroblock,
    ) -> Result<&'a QuantizedMacroblock, ReconstructionError> {
        // Unpacking fails only on an arena its decoder did not fill.
        self.unpack(x, y, scratch)
            .map_err(|_| ReconstructionError::InvalidPlaneGeometry("arena macroblock"))?;
        Ok(scratch)
    }
}

fn unpack_macroblock(
    arena: &CoefficientArena,
    index: usize,
    block_columns: usize,
    block_rows: usize,
    macroblock: &mut QuantizedMacroblock,
) -> Result<(), JxrError> {
    let block_count = block_columns
        .checked_mul(block_rows)
        .ok_or_else(|| JxrError::arithmetic("component block count"))?;
    let offset = usize::try_from(arena.macroblocks.coefficient_offsets[index])
        .map_err(|_| JxrError::arithmetic("coefficient offset"))?;
    let bands = arena.macroblocks.bands[index];
    let end = offset
        .checked_add(coefficients_per_macroblock(bands, block_count))
        .ok_or_else(|| JxrError::arithmetic("macroblock coefficient range"))?;
    let coefficients = arena
        .coefficients
        .get(offset..end)
        .ok_or_else(|| JxrError::new(JxrErrorKind::InvalidSyntax, "macroblock coefficients"))?;
    unpack_coefficients(coefficients, bands, block_count, macroblock);
    apply_high_pass_prediction(
        &mut macroblock.high_pass,
        arena.macroblocks.hp_predictions[index],
        block_columns,
        block_rows,
    )?;
    macroblock.quantizers = arena.macroblocks.quantizers[index];
    macroblock.bands = bands;
    Ok(())
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

/// The macroblocks a decode reconstructs: the tiles a request needs, which
/// are a rectangle of whole tiles.
#[derive(Debug, Clone)]
pub(super) struct Window {
    /// First macroblock column and row, in the coded image.
    pub(super) origin_x: u32,
    pub(super) origin_y: u32,
    /// Macroblocks across and down.
    pub(super) width: u32,
    pub(super) height: u32,
    /// Tile boundaries within the window.
    pub(super) tiles: TilePartition,
}

impl Window {
    /// The window of the tiles `plan` marks as needed.
    pub(super) fn of_plan(plan: &PreparedPlan) -> Result<Self, JxrError> {
        let grid = &plan.info.tiles;
        let columns = grid.column_widths.len();
        let mut bounds: Option<[u32; 4]> = None;
        let mut top = 0_u32;
        for (row, &height) in grid.row_heights.iter().enumerate() {
            let mut left = 0_u32;
            for (column, &width) in grid.column_widths.iter().enumerate() {
                let needed = plan
                    .tiles
                    .get(row * columns + column)
                    .is_some_and(|tile| tile.required_for_reconstruction);
                if needed {
                    let [x0, y0, x1, y1] = bounds.unwrap_or([u32::MAX, u32::MAX, 0, 0]);
                    bounds = Some([
                        x0.min(left),
                        y0.min(top),
                        x1.max(left.saturating_add(width)),
                        y1.max(top.saturating_add(height)),
                    ]);
                }
                left = left.saturating_add(width);
            }
            top = top.saturating_add(height);
        }
        let [origin_x, origin_y, right, bottom] = bounds.ok_or_else(|| {
            JxrError::new(
                JxrErrorKind::InternalInvariant,
                "empty reconstruction window",
            )
        })?;
        Ok(Self {
            origin_x,
            origin_y,
            width: right - origin_x,
            height: bottom - origin_y,
            tiles: tile_partition(plan, origin_x, origin_y, right, bottom)?,
        })
    }

    /// Macroblocks across the window.
    pub(super) const fn columns(&self) -> usize {
        self.width as usize
    }

    /// Macroblock rows of the window.
    pub(super) const fn rows(&self) -> usize {
        self.height as usize
    }

    /// Reconstruction geometry of a component whose macroblocks have
    /// `blocks` blocks across and down.
    pub(super) fn config(
        &self,
        [block_columns, block_rows]: [u8; 2],
        scale_after_first_transform: bool,
        plan: &PreparedPlan,
    ) -> ReconstructionConfig {
        ReconstructionConfig {
            macroblock_origin_x: self.origin_x,
            macroblock_origin_y: self.origin_y,
            macroblocks_x: self.width,
            macroblocks_y: self.height,
            block_columns,
            block_rows,
            scale_after_first_transform,
            overlap: plan.primary.overlap,
            tiles: self.tiles.clone(),
        }
    }
}

/// Places the `packed` coefficients of a macroblock of `block_count` blocks
/// that codes `bands` in `macroblock`: every coefficient of its first
/// `block_count` blocks, zero where the bands code none, and element zero of
/// each high-pass block, which the low-pass value supplies, zero too.
fn unpack_coefficients(
    packed: &[i32],
    bands: BandPresence,
    block_count: usize,
    macroblock: &mut QuantizedMacroblock,
) {
    let low = &mut macroblock.dc_low_pass[..block_count];
    let high = &mut macroblock.high_pass[..block_count * 16];
    match bands {
        BandPresence::DcOnly => {
            low.fill(0);
            low[0] = packed[0];
        }
        BandPresence::NoHighPass => low.copy_from_slice(packed),
        BandPresence::NoFlexbits | BandPresence::All => {
            let blocks = high.as_chunks_mut::<16>().0;
            for ((low, high), source) in low.iter_mut().zip(blocks).zip(packed.as_chunks::<16>().0)
            {
                *low = source[0];
                *high = *source;
                high[0] = 0;
            }
            return;
        }
    }
    high.fill(0);
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
            plan.info.tiles.column_widths.iter().copied(),
            plan.primary.macroblocks_x,
            origin_x,
            right,
            "tile column boundaries",
        )?,
        row_boundaries: clipped_boundaries(
            plan.info.tiles.row_heights.iter().copied(),
            plan.primary.macroblocks_y,
            origin_y,
            bottom,
            "tile row boundaries",
        )?,
        hard_boundaries: plan.info.tiles.hard_tiles,
    })
}

/// The boundaries of the tiles of one axis whose sizes, `sizes`, add up to
/// `expected`, clipped to `start..end` and counted from `start`: zero, each
/// boundary inside, and `end - start`. Over the whole axis, `0..expected`,
/// they are every boundary, when no tile is empty.
pub(crate) fn clipped_boundaries(
    sizes: impl ExactSizeIterator<Item = u32>,
    expected: u32,
    start: u32,
    end: u32,
    operation: &'static str,
) -> Result<Vec<u32>, JxrError> {
    let mut source = Vec::with_capacity(sizes.len() + 1);
    source.push(0);
    let mut current = 0_u32;
    for size in sizes {
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

pub(super) fn map_reconstruction_error(error: &ReconstructionError) -> JxrError {
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
    use jxr_core::BandPresence;

    use super::{QuantizedMacroblock, clipped_boundaries, unpack_coefficients};
    use crate::tile_decode::coefficients_per_macroblock;

    /// Macroblocks unpacked one after another into the same storage, as a
    /// strip unpacks a component's, come out as each does unpacked into
    /// zeroed storage, whatever bands the one before coded.
    #[test]
    fn unpacking_into_reused_storage_matches_unpacking_into_zeroed_storage() {
        let bands = [
            BandPresence::All,
            BandPresence::DcOnly,
            BandPresence::NoFlexbits,
            BandPresence::NoHighPass,
            BandPresence::All,
        ];
        for block_count in [16, 8, 4] {
            let mut reused = QuantizedMacroblock::ZERO;
            for (step, &bands) in bands.iter().enumerate() {
                let packed: Vec<i32> = (0..coefficients_per_macroblock(bands, block_count))
                    .map(|index| i32::try_from(index * 7 + step * 3).unwrap() - 50)
                    .collect();
                let mut fresh = QuantizedMacroblock::ZERO;
                unpack_coefficients(&packed, bands, block_count, &mut fresh);
                unpack_coefficients(&packed, bands, block_count, &mut reused);
                assert_eq!(
                    reused.dc_low_pass[..block_count],
                    fresh.dc_low_pass[..block_count]
                );
                assert_eq!(
                    reused.high_pass[..block_count * 16],
                    fresh.high_pass[..block_count * 16]
                );
                // Element zero of each high-pass block is left zero.
                assert!(fresh.high_pass.iter().step_by(16).all(|&value| value == 0));
            }
        }
    }

    #[test]
    fn regional_tile_boundaries_are_rebased_and_clipped() {
        assert_eq!(
            clipped_boundaries([2, 3, 1].into_iter(), 6, 1, 5, "test boundaries").unwrap(),
            [0, 1, 4]
        );
    }
}
