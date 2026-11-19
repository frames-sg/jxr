//! Bounds-checked tile packets decoded into macroblock-major coefficient storage.

pub(crate) mod cbphp;
mod dispatch;
mod error;
mod frequency;
pub(crate) mod high_pass;
mod output;
mod plane;
pub(crate) mod quantizer;
pub(crate) mod spatial;
mod spatial_tile;
pub(crate) mod syntax;
pub(crate) mod yuv;

use jxr_core::{
    BandPresence, ChromaSampling, CoefficientArena, CoefficientArenaDescriptor, CoefficientPlane,
    ColorFormat, MacroblockMetadata, PreparedPlan, TileEdgeFlags,
};

use crate::{ParsedCodestream, executor::Executor, metadata::internal_color};

pub(crate) use dispatch::BandDecoder;
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
    let executor = Executor::DEFAULT;
    let (layouts, coefficient_count, macroblock_count) = planned(parsed, plan)?;
    let mut arena = CoefficientArena {
        coefficients: vec![0; coefficient_count],
        macroblocks: macroblock_metadata(macroblock_count),
        planes: Vec::with_capacity(layouts.len()),
    };
    decode_planes(
        source,
        parsed,
        plan,
        &layouts,
        PlaneStorage {
            coefficients: &mut arena.coefficients,
            macroblocks: &mut arena.macroblocks,
            planes: &mut arena.planes,
        },
        executor,
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
    decode_tiles_reusing_on(source, parsed, plan, arena, Executor::DEFAULT)
}

/// [`decode_tiles_reusing`], reading several tiles side by side through
/// `executor`.
pub(crate) fn decode_tiles_reusing_on(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    arena: &mut CoefficientArena,
    executor: Executor,
) -> Result<bool, TileDecodeError> {
    let (layouts, coefficient_count, macroblock_count) = planned(parsed, plan)?;
    let reused = prepare_reusable_arena(arena, coefficient_count, macroblock_count, layouts.len());
    decode_planes(
        source,
        parsed,
        plan,
        &layouts,
        PlaneStorage {
            coefficients: &mut arena.coefficients,
            macroblocks: &mut arena.macroblocks,
            planes: &mut arena.planes,
        },
        executor,
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
    let (layouts, coefficient_count, macroblock_count) = planned(parsed, plan)?;
    if destination.len() != coefficient_count {
        return Err(TileDecodeError::InvalidPlan(
            "external coefficient storage length",
        ));
    }
    let mut macroblocks = macroblock_metadata(macroblock_count);
    let mut planes = Vec::with_capacity(layouts.len());
    decode_planes(
        source,
        parsed,
        plan,
        &layouts,
        PlaneStorage {
            coefficients: destination,
            macroblocks: &mut macroblocks,
            planes: &mut planes,
        },
        Executor::DEFAULT,
    )?;
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
    arena_capacity(plan, &plane_layouts(parsed, plan)).map(|(coefficients, _)| coefficients)
}

/// Where [`decode_planes`] puts what it decodes: the plane-major
/// coefficients, and the macroblock metadata and planes it appends.
struct PlaneStorage<'a> {
    coefficients: &'a mut [i32],
    macroblocks: &'a mut MacroblockMetadata,
    planes: &'a mut Vec<CoefficientPlane>,
}

/// Decode every selected tile directly into its plane-major coefficient
/// windows, side by side through `executor`, then append the macroblock
/// metadata in arena order.
fn decode_planes(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    layouts: &[output::PlaneLayout],
    storage: PlaneStorage<'_>,
    executor: Executor,
) -> Result<(), TileDecodeError> {
    let PlaneStorage {
        coefficients,
        macroblocks,
        planes,
    } = storage;
    let locations = tile_locations(parsed, plan, plan.tiles.len())?;
    let coefficient_count = coefficients.len();
    let writers = output::split_planes(coefficients, layouts, &plan.tiles)?;
    let decoded = dispatch::decode_packets(source, parsed, plan, &locations, writers, executor)?;
    let written = output::append_metadata(macroblocks, planes, &decoded, layouts)?;
    if written != coefficient_count {
        return Err(TileDecodeError::InvalidPlan(
            "external coefficient storage fill length",
        ));
    }
    Ok(())
}

fn validate_tile_count(plan: &PreparedPlan) -> Result<(), TileDecodeError> {
    let grid = &plan.info.tiles;
    let expected_tiles = grid
        .column_widths
        .len()
        .checked_mul(grid.row_heights.len())
        .ok_or(TileDecodeError::ArithmeticOverflow("tile count"))?;
    if plan.tiles.len() != expected_tiles {
        return Err(TileDecodeError::InvalidPlan("tile count"));
    }
    Ok(())
}

/// Per-component coefficient geometry: primary components, then an
/// integrated alpha plane when present.
fn plane_layouts(parsed: &ParsedCodestream, plan: &PreparedPlan) -> Vec<output::PlaneLayout> {
    let primary_components = usize::from(parsed.headers.primary.components);
    let component_count = primary_components + usize::from(parsed.headers.alpha.is_some());
    (0..component_count)
        .map(|component| {
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
            output::PlaneLayout {
                geometry,
                coefficients_per_macroblock: coefficients_per_macroblock(
                    plan.scale.retained_bands(available_bands),
                    geometry.block_count(),
                ),
            }
        })
        .collect()
}

/// The planes a decode of `plan` stores, with the coefficients and the
/// macroblocks they take, once the codestream is checked to be one the
/// tile readers read.
fn planned(
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
) -> Result<(Vec<output::PlaneLayout>, usize, usize), TileDecodeError> {
    validate_decode_scope(parsed)?;
    validate_tile_count(plan)?;
    let layouts = plane_layouts(parsed, plan);
    let (coefficients, macroblocks) = arena_capacity(plan, &layouts)?;
    Ok((layouts, coefficients, macroblocks))
}

fn arena_capacity(
    plan: &PreparedPlan,
    layouts: &[output::PlaneLayout],
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
    let total_macroblocks =
        plane_macroblocks
            .checked_mul(layouts.len())
            .ok_or(TileDecodeError::ArithmeticOverflow(
                "component macroblock count",
            ))?;
    let coefficients_per_macroblock = layouts.iter().try_fold(0_usize, |total, layout| {
        total.checked_add(layout.coefficients_per_macroblock).ok_or(
            TileDecodeError::ArithmeticOverflow("coefficient arena size"),
        )
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
) -> bool {
    let reused = arena.coefficients.capacity() >= coefficient_count
        && arena.planes.capacity() >= component_count
        && macroblock_capacity(&arena.macroblocks) >= macroblock_count;
    // Tile decoders overwrite every element, so retained values are never
    // read and reused storage is resized without clearing it first.
    if arena.coefficients.capacity() < coefficient_count {
        arena.coefficients = vec![0; coefficient_count];
    } else {
        arena.coefficients.resize(coefficient_count, 0);
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
    reused
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
    // A plane whose colour format has a macroblock syntax, with as many
    // components as that format has.
    let plane = &parsed.headers.primary;
    let supported =
        internal_color(plane.internal_color_format, plane.components).is_ok_and(|colour| {
            colour.component_count() == Some(plane.components)
                && syntax::ComponentSyntax::new(colour).is_ok()
        });
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

/// Coefficients a macroblock of a component of `block_count` blocks stores
/// with the bands `bands`: its DC, the DC or low-pass coefficient of each
/// block, or every coefficient of every block.
pub(crate) const fn coefficients_per_macroblock(bands: BandPresence, block_count: usize) -> usize {
    match bands {
        BandPresence::DcOnly => 1,
        BandPresence::NoHighPass => block_count,
        BandPresence::NoFlexbits | BandPresence::All => block_count * 16,
    }
}

/// Transform blocks across and down one macroblock of a component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ComponentGeometry {
    pub(crate) columns: u8,
    pub(crate) rows: u8,
}

impl ComponentGeometry {
    pub(crate) const fn block_count(self) -> usize {
        self.columns as usize * self.rows as usize
    }
}

/// Macroblock shape of component `component` of a plane coded in `color`:
/// four blocks square, except for the chroma components of YUV 4:2:2, two
/// across and four down, and of YUV 4:2:0, two square.
pub(crate) const fn component_geometry(color: ColorFormat, component: usize) -> ComponentGeometry {
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
