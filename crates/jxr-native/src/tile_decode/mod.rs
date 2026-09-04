//! Bounds-checked tile packets decoded into macroblock-major coefficient storage.

mod cbphp;
mod dispatch;
mod error;
mod frequency;
mod high_pass;
mod integrated_alpha;
mod integrated_alpha_frequency;
mod multicomponent;
mod multicomponent_frequency;
mod quantizer;
mod spatial;
mod yuv;

use jxr_core::{
    BandPresence, ChromaSampling, CoefficientArena, CoefficientArenaDescriptor, CoefficientPlane,
    ColorFormat, MacroblockMetadata, PreparedPlan, TileEdgeFlags,
};

use crate::ParsedCodestream;

pub use error::TileDecodeError;

/// Decode tile packets into the device-neutral coefficient arena.
///
/// Spatial and frequency packet layouts are accepted for the implemented
/// primary formats. A supported integrated alpha plane is appended after the
/// primary component planes and retains independent entropy state.
pub fn decode_tiles(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
) -> Result<CoefficientArena, TileDecodeError> {
    validate_decode_scope(parsed)?;
    let grid = &plan.info.tiles;
    let expected_tiles = grid
        .column_widths
        .len()
        .checked_mul(grid.row_heights.len())
        .ok_or(TileDecodeError::ArithmeticOverflow("tile count"))?;
    if plan.tiles.len() != expected_tiles {
        return Err(TileDecodeError::InvalidPlan("tile count"));
    }
    let primary_components = usize::from(parsed.headers.primary.components);
    let component_count = primary_components + usize::from(parsed.headers.alpha.is_some());
    let mut arena = allocate_arena(plan, component_count, primary_components)?;
    let locations = tile_locations(parsed, plan, expected_tiles)?;
    let decoded = dispatch::decode_packets(source, parsed, plan, &locations)?;
    append_planes(
        &mut arena.coefficients,
        &mut arena.macroblocks,
        &mut arena.planes,
        decoded,
        component_count,
        primary_components,
        plan.info.primary.color_format,
    )?;
    arena
        .validate()
        .map_err(|_| TileDecodeError::InvalidPlan("coefficient arena"))?;
    Ok(arena)
}

/// Decode tile packets while retaining caller-owned coefficient capacities.
pub fn decode_tiles_reusing(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    arena: &mut CoefficientArena,
) -> Result<bool, TileDecodeError> {
    validate_decode_scope(parsed)?;
    let grid = &plan.info.tiles;
    let expected_tiles = grid
        .column_widths
        .len()
        .checked_mul(grid.row_heights.len())
        .ok_or(TileDecodeError::ArithmeticOverflow("tile count"))?;
    if plan.tiles.len() != expected_tiles {
        return Err(TileDecodeError::InvalidPlan("tile count"));
    }
    let primary_components = usize::from(parsed.headers.primary.components);
    let component_count = primary_components + usize::from(parsed.headers.alpha.is_some());
    let (coefficient_count, macroblock_count) =
        arena_capacity(plan, component_count, primary_components)?;
    let reused = arena.coefficients.capacity() >= coefficient_count
        && arena.planes.capacity() >= component_count
        && macroblock_capacity(&arena.macroblocks) >= macroblock_count;
    prepare_reusable_arena(arena, coefficient_count, macroblock_count, component_count);
    let locations = tile_locations(parsed, plan, expected_tiles)?;
    let decoded = dispatch::decode_packets(source, parsed, plan, &locations)?;
    append_planes(
        &mut arena.coefficients,
        &mut arena.macroblocks,
        &mut arena.planes,
        decoded,
        component_count,
        primary_components,
        plan.info.primary.color_format,
    )?;
    arena
        .validate()
        .map_err(|_| TileDecodeError::InvalidPlan("coefficient arena"))?;
    Ok(reused)
}

/// Decode coefficients directly into caller-owned exact-size storage.
pub fn decode_tiles_into(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    destination: &mut [i32],
) -> Result<CoefficientArenaDescriptor, TileDecodeError> {
    validate_decode_scope(parsed)?;
    let grid = &plan.info.tiles;
    let expected_tiles = grid
        .column_widths
        .len()
        .checked_mul(grid.row_heights.len())
        .ok_or(TileDecodeError::ArithmeticOverflow("tile count"))?;
    if plan.tiles.len() != expected_tiles {
        return Err(TileDecodeError::InvalidPlan("tile count"));
    }
    let primary_components = usize::from(parsed.headers.primary.components);
    let component_count = primary_components + usize::from(parsed.headers.alpha.is_some());
    let (coefficient_count, macroblock_count) =
        arena_capacity(plan, component_count, primary_components)?;
    if destination.len() != coefficient_count {
        return Err(TileDecodeError::InvalidPlan(
            "external coefficient storage length",
        ));
    }
    let locations = tile_locations(parsed, plan, expected_tiles)?;
    let decoded = dispatch::decode_packets(source, parsed, plan, &locations)?;
    let mut target = SliceTarget::new(destination);
    let mut macroblocks = macroblock_metadata(macroblock_count);
    let mut planes = Vec::with_capacity(component_count);
    append_planes(
        &mut target,
        &mut macroblocks,
        &mut planes,
        decoded,
        component_count,
        primary_components,
        plan.info.primary.color_format,
    )?;
    if target.len() != coefficient_count {
        return Err(TileDecodeError::InvalidPlan(
            "external coefficient storage fill length",
        ));
    }
    let descriptor = CoefficientArenaDescriptor {
        coefficient_count,
        macroblocks,
        planes,
    };
    descriptor
        .validate()
        .map_err(|_| TileDecodeError::InvalidPlan("coefficient arena descriptor"))?;
    Ok(descriptor)
}

/// Return the exact coefficient count for this prepared tile selection.
pub fn coefficient_count(
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
) -> Result<usize, TileDecodeError> {
    let primary_components = usize::from(parsed.headers.primary.components);
    let component_count = primary_components + usize::from(parsed.headers.alpha.is_some());
    arena_capacity(plan, component_count, primary_components).map(|(coefficients, _)| coefficients)
}

fn allocate_arena(
    plan: &PreparedPlan,
    component_count: usize,
    primary_components: usize,
) -> Result<CoefficientArena, TileDecodeError> {
    let (coefficient_count, total_macroblocks) =
        arena_capacity(plan, component_count, primary_components)?;
    Ok(CoefficientArena {
        coefficients: Vec::with_capacity(coefficient_count),
        macroblocks: macroblock_metadata(total_macroblocks),
        planes: Vec::with_capacity(component_count),
    })
}

fn arena_capacity(
    plan: &PreparedPlan,
    component_count: usize,
    primary_components: usize,
) -> Result<(usize, usize), TileDecodeError> {
    let plane_macroblocks = plan
        .tiles
        .iter()
        .filter(|tile| tile.required_for_reconstruction)
        .try_fold(0_usize, |total, tile| {
            usize::try_from(tile.macroblock_count)
                .ok()
                .and_then(|count| total.checked_add(count))
        })
        .ok_or(TileDecodeError::ArithmeticOverflow(
            "plane macroblock count",
        ))?;
    let total_macroblocks = plane_macroblocks.checked_mul(component_count).ok_or(
        TileDecodeError::ArithmeticOverflow("component macroblock count"),
    )?;
    let coefficients_per_macroblock =
        (0..component_count).try_fold(0_usize, |total, component| {
            let geometry = if component < primary_components {
                component_geometry(plan.info.primary.color_format, component)
            } else {
                ComponentGeometry {
                    columns: 4,
                    rows: 4,
                }
            };
            let available_bands = if component < primary_components {
                plan.info.primary.bands
            } else {
                plan.info
                    .alpha
                    .as_ref()
                    .map_or(plan.info.primary.bands, |alpha| alpha.bands)
            };
            total
                .checked_add(coefficients_per_macroblock(
                    plan.scale.retained_bands(available_bands),
                    geometry.block_count(),
                ))
                .ok_or(TileDecodeError::ArithmeticOverflow(
                    "coefficient arena size",
                ))
        })?;
    let coefficient_count = plane_macroblocks
        .checked_mul(coefficients_per_macroblock)
        .ok_or(TileDecodeError::ArithmeticOverflow(
            "coefficient arena size",
        ))?;
    Ok((coefficient_count, total_macroblocks))
}

fn macroblock_metadata(capacity: usize) -> MacroblockMetadata {
    MacroblockMetadata {
        coefficient_offsets: Vec::with_capacity(capacity),
        quantizers: Vec::with_capacity(capacity),
        bands: Vec::with_capacity(capacity),
        predictions: Vec::with_capacity(capacity),
        hp_predictions: Vec::with_capacity(capacity),
        tile_edges: Vec::with_capacity(capacity),
        coded_x: Vec::with_capacity(capacity),
        coded_y: Vec::with_capacity(capacity),
        output_x: Vec::with_capacity(capacity),
        output_y: Vec::with_capacity(capacity),
    }
}

fn macroblock_capacity(metadata: &MacroblockMetadata) -> usize {
    [
        metadata.coefficient_offsets.capacity(),
        metadata.quantizers.capacity(),
        metadata.bands.capacity(),
        metadata.predictions.capacity(),
        metadata.hp_predictions.capacity(),
        metadata.tile_edges.capacity(),
        metadata.coded_x.capacity(),
        metadata.coded_y.capacity(),
        metadata.output_x.capacity(),
        metadata.output_y.capacity(),
    ]
    .into_iter()
    .min()
    .unwrap_or(0)
}

fn prepare_reusable_arena(
    arena: &mut CoefficientArena,
    coefficient_count: usize,
    macroblock_count: usize,
    component_count: usize,
) {
    arena.coefficients.clear();
    if arena.coefficients.capacity() < coefficient_count {
        arena.coefficients.reserve(coefficient_count);
    }
    arena.planes.clear();
    if arena.planes.capacity() < component_count {
        arena.planes.reserve(component_count);
    }
    let metadata = &mut arena.macroblocks;
    metadata.coefficient_offsets.clear();
    metadata.quantizers.clear();
    metadata.bands.clear();
    metadata.predictions.clear();
    metadata.hp_predictions.clear();
    metadata.tile_edges.clear();
    metadata.coded_x.clear();
    metadata.coded_y.clear();
    metadata.output_x.clear();
    metadata.output_y.clear();
    macro_rules! reserve_metadata {
        ($field:ident) => {
            if metadata.$field.capacity() < macroblock_count {
                metadata.$field.reserve(macroblock_count);
            }
        };
    }
    reserve_metadata!(coefficient_offsets);
    reserve_metadata!(quantizers);
    reserve_metadata!(bands);
    reserve_metadata!(predictions);
    reserve_metadata!(hp_predictions);
    reserve_metadata!(tile_edges);
    reserve_metadata!(coded_x);
    reserve_metadata!(coded_y);
    reserve_metadata!(output_x);
    reserve_metadata!(output_y);
}

fn tile_locations(
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    expected_tiles: usize,
) -> Result<Vec<TileLocation>, TileDecodeError> {
    let grid = &plan.info.tiles;
    let mut locations = Vec::with_capacity(expected_tiles);
    let mut global_y = 0_u32;
    for &tile_height in &grid.row_heights {
        let mut global_x = 0_u32;
        for &tile_width in &grid.column_widths {
            locations.push(TileLocation {
                global_x,
                global_y,
                width: tile_width,
                height: tile_height,
                hard: grid.hard_tiles,
                margin_left: u32::from(parsed.headers.image.margins[1]),
                margin_top: u32::from(parsed.headers.image.margins[0]),
            });
            global_x = global_x
                .checked_add(tile_width)
                .ok_or(TileDecodeError::ArithmeticOverflow("tile x origin"))?;
        }
        global_y = global_y
            .checked_add(tile_height)
            .ok_or(TileDecodeError::ArithmeticOverflow("tile y origin"))?;
    }
    Ok(locations)
}

fn validate_decode_scope(parsed: &ParsedCodestream) -> Result<(), TileDecodeError> {
    if let Some(alpha) = &parsed.headers.alpha
        && (alpha.internal_color_format != 0 || alpha.components != 1)
    {
        return Err(TileDecodeError::Unsupported(
            "integrated alpha plane component layout",
        ));
    }
    let plane = &parsed.headers.primary;
    let supported = matches!(
        (plane.internal_color_format, plane.components),
        (0, 1) | (1..=3, 3) | (4, 4) | (6, 2..=16)
    );
    if !supported {
        return Err(TileDecodeError::Unsupported(
            "unsupported primary component layout",
        ));
    }
    Ok(())
}

fn packet_slice(source: &[u8], range: jxr_core::ByteRange) -> Result<&[u8], TileDecodeError> {
    let end = range
        .offset
        .checked_add(range.length)
        .ok_or(TileDecodeError::ArithmeticOverflow("tile packet range"))?;
    source
        .get(range.offset..end)
        .ok_or(TileDecodeError::PacketRangeOutsideInput {
            offset: range.offset,
            length: range.length,
            input_length: source.len(),
        })
}

#[derive(Debug, Clone, Copy)]
struct TileLocation {
    global_x: u32,
    global_y: u32,
    width: u32,
    height: u32,
    hard: bool,
    margin_left: u32,
    margin_top: u32,
}

pub(in crate::tile_decode) struct DecodedTile {
    pub(in crate::tile_decode) components: Vec<Vec<spatial::SpatialMacroblock>>,
}

trait CoefficientTarget {
    fn len(&self) -> usize;
    fn push(&mut self, value: i32) -> Result<(), TileDecodeError>;
    fn extend_from_slice(&mut self, values: &[i32]) -> Result<(), TileDecodeError>;
}

impl CoefficientTarget for Vec<i32> {
    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn push(&mut self, value: i32) -> Result<(), TileDecodeError> {
        Vec::push(self, value);
        Ok(())
    }

    fn extend_from_slice(&mut self, values: &[i32]) -> Result<(), TileDecodeError> {
        Vec::extend_from_slice(self, values);
        Ok(())
    }
}

struct SliceTarget<'a> {
    destination: &'a mut [i32],
    written: usize,
}

impl<'a> SliceTarget<'a> {
    const fn new(destination: &'a mut [i32]) -> Self {
        Self {
            destination,
            written: 0,
        }
    }
}

impl CoefficientTarget for SliceTarget<'_> {
    fn len(&self) -> usize {
        self.written
    }

    fn push(&mut self, value: i32) -> Result<(), TileDecodeError> {
        let slot = self
            .destination
            .get_mut(self.written)
            .ok_or(TileDecodeError::InvalidPlan(
                "external coefficient storage overflow",
            ))?;
        *slot = value;
        self.written += 1;
        Ok(())
    }

    fn extend_from_slice(&mut self, values: &[i32]) -> Result<(), TileDecodeError> {
        let end =
            self.written
                .checked_add(values.len())
                .ok_or(TileDecodeError::ArithmeticOverflow(
                    "external coefficient write",
                ))?;
        let target =
            self.destination
                .get_mut(self.written..end)
                .ok_or(TileDecodeError::InvalidPlan(
                    "external coefficient storage overflow",
                ))?;
        target.copy_from_slice(values);
        self.written = end;
        Ok(())
    }
}

impl DecodedTile {
    fn single(macroblocks: Vec<spatial::SpatialMacroblock>) -> Self {
        Self {
            components: vec![macroblocks],
        }
    }
}

fn append_planes<T: CoefficientTarget>(
    coefficients: &mut T,
    macroblocks: &mut MacroblockMetadata,
    planes: &mut Vec<CoefficientPlane>,
    mut decoded: Vec<(DecodedTile, TileLocation)>,
    component_count: usize,
    primary_components: usize,
    color_format: ColorFormat,
) -> Result<(), TileDecodeError> {
    if decoded
        .iter()
        .any(|(tile, _)| tile.components.len() != component_count)
    {
        return Err(TileDecodeError::InvalidPlan("decoded component count"));
    }
    for component in 0..component_count {
        let geometry = if component < primary_components {
            component_geometry(color_format, component)
        } else {
            ComponentGeometry {
                columns: 4,
                rows: 4,
            }
        };
        let coefficient_offset = coefficients.len();
        let macroblock_offset = macroblocks.len();
        for (tile, location) in &mut decoded {
            append_tile(
                coefficients,
                macroblocks,
                core::mem::take(&mut tile.components[component]),
                *location,
                geometry,
            )?;
        }
        planes.push(CoefficientPlane {
            coefficient_offset,
            coefficient_count: coefficients.len() - coefficient_offset,
            macroblock_offset,
            macroblock_count: macroblocks.len() - macroblock_offset,
            block_columns: geometry.columns,
            block_rows: geometry.rows,
        });
    }
    Ok(())
}

fn append_tile<T: CoefficientTarget>(
    coefficients: &mut T,
    metadata: &mut MacroblockMetadata,
    macroblocks: Vec<spatial::SpatialMacroblock>,
    location: TileLocation,
    geometry: ComponentGeometry,
) -> Result<(), TileDecodeError> {
    let expected = usize::try_from(location.width)
        .ok()
        .and_then(|width| {
            usize::try_from(location.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or(TileDecodeError::ArithmeticOverflow("tile traversal"))?;
    if macroblocks.len() != expected {
        return Err(TileDecodeError::InvalidPlan(
            "decoded tile macroblock count",
        ));
    }
    let width = usize::try_from(location.width)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("tile width"))?;
    for (index, macroblock) in macroblocks.into_iter().enumerate() {
        let x = index % width;
        let y = index / width;
        let coefficient_offset = u32::try_from(coefficients.len())
            .map_err(|_| TileDecodeError::ArithmeticOverflow("coefficient offset"))?;
        metadata.coefficient_offsets.push(coefficient_offset);
        metadata.quantizers.push(macroblock.coefficients.quantizers);
        metadata.bands.push(macroblock.coefficients.bands);
        metadata.predictions.push(macroblock.prediction);
        metadata.hp_predictions.push(macroblock.hp_prediction);
        metadata.tile_edges.push(tile_edges(
            x,
            y,
            width,
            usize::try_from(location.height)
                .map_err(|_| TileDecodeError::ArithmeticOverflow("tile height"))?,
            location.hard,
        ));
        let x_mb = location
            .global_x
            .checked_add(
                u32::try_from(x)
                    .map_err(|_| TileDecodeError::ArithmeticOverflow("macroblock x conversion"))?,
            )
            .ok_or(TileDecodeError::ArithmeticOverflow("macroblock x"))?;
        let y_mb = location
            .global_y
            .checked_add(
                u32::try_from(y)
                    .map_err(|_| TileDecodeError::ArithmeticOverflow("macroblock y conversion"))?,
            )
            .ok_or(TileDecodeError::ArithmeticOverflow("macroblock y"))?;
        metadata.coded_x.push(x_mb);
        metadata.coded_y.push(y_mb);
        metadata
            .output_x
            .push(x_mb.saturating_mul(16).saturating_sub(location.margin_left));
        metadata
            .output_y
            .push(y_mb.saturating_mul(16).saturating_sub(location.margin_top));
        append_coefficients(
            coefficients,
            &macroblock.coefficients,
            geometry.block_count(),
        )?;
    }
    Ok(())
}

fn coefficients_per_macroblock(bands: BandPresence, block_count: usize) -> usize {
    match bands {
        BandPresence::DcOnly => 1,
        BandPresence::NoHighPass => block_count,
        BandPresence::NoFlexbits | BandPresence::All => block_count * 16,
    }
}

fn append_coefficients<T: CoefficientTarget>(
    arena: &mut T,
    macroblock: &crate::reconstruct::QuantizedMacroblock,
    block_count: usize,
) -> Result<(), TileDecodeError> {
    match macroblock.bands {
        BandPresence::DcOnly => arena.push(macroblock.dc_low_pass[0])?,
        BandPresence::NoHighPass => {
            arena.extend_from_slice(&macroblock.dc_low_pass[..block_count])?;
        }
        BandPresence::NoFlexbits | BandPresence::All => {
            for block in 0..block_count {
                arena.push(macroblock.dc_low_pass[block])?;
                arena.extend_from_slice(&macroblock.high_pass[block * 16 + 1..(block + 1) * 16])?;
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct ComponentGeometry {
    columns: u8,
    rows: u8,
}

impl ComponentGeometry {
    const fn block_count(self) -> usize {
        self.columns as usize * self.rows as usize
    }
}

const fn component_geometry(color: ColorFormat, component: usize) -> ComponentGeometry {
    match (color, component) {
        (ColorFormat::Yuv(ChromaSampling::Cs420), 1..) => ComponentGeometry {
            columns: 2,
            rows: 2,
        },
        (ColorFormat::Yuv(ChromaSampling::Cs422), 1..) => ComponentGeometry {
            columns: 2,
            rows: 4,
        },
        _ => ComponentGeometry {
            columns: 4,
            rows: 4,
        },
    }
}

fn tile_edges(x: usize, y: usize, width: usize, height: usize, hard: bool) -> TileEdgeFlags {
    let mut edges = TileEdgeFlags::default();
    if x == 0 {
        edges = edges.union(TileEdgeFlags::LEFT);
    }
    if y == 0 {
        edges = edges.union(TileEdgeFlags::TOP);
    }
    if x + 1 == width {
        edges = edges.union(TileEdgeFlags::RIGHT);
    }
    if y + 1 == height {
        edges = edges.union(TileEdgeFlags::BOTTOM);
    }
    if hard {
        edges = edges.union(TileEdgeFlags::HARD_TILE);
    }
    edges
}

#[cfg(test)]
mod tests;
