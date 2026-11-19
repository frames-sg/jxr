//! Plane-major coefficient windows written directly by tile decoders.

use jxr_core::{
    BandPresence, CoefficientPlane, MacroblockMetadata, PredictionMode, QuantizerSet, TilePlan,
};

use super::{
    ComponentGeometry, TileDecodeError, coefficients_per_macroblock, dispatch::DecodedRows,
    tile_edges,
};

/// Metadata retained for one decoded macroblock after its coefficients are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::tile_decode) struct MacroblockRecord {
    pub(in crate::tile_decode) quantizers: QuantizerSet,
    pub(in crate::tile_decode) bands: BandPresence,
    pub(in crate::tile_decode) prediction: PredictionMode,
    pub(in crate::tile_decode) hp_prediction: PredictionMode,
}

/// Coefficient geometry shared by every macroblock of one component plane.
#[derive(Debug, Clone, Copy)]
pub(in crate::tile_decode) struct PlaneLayout {
    pub(in crate::tile_decode) geometry: ComponentGeometry,
    pub(in crate::tile_decode) coefficients_per_macroblock: usize,
}

/// One tile's exact window into one component plane of the coefficient arena.
///
/// Macroblocks are written in raster order, so a window whose cursor reaches
/// its end has had every element overwritten.
pub(in crate::tile_decode) struct PlaneWriter<'a> {
    coefficients: &'a mut [i32],
    written: usize,
    block_count: usize,
    macroblocks: usize,
    records: Vec<MacroblockRecord>,
}

impl<'a> PlaneWriter<'a> {
    pub(in crate::tile_decode) fn new(
        coefficients: &'a mut [i32],
        block_count: usize,
        macroblocks: usize,
    ) -> Self {
        Self {
            coefficients,
            written: 0,
            block_count,
            macroblocks,
            records: Vec::with_capacity(macroblocks),
        }
    }

    /// Store one macroblock in arena order: per block, its DC/LP value then
    /// its 15 HP values. `high` uses the 16-entry block stride of the HP
    /// decoders; entry 0 of each block is replaced by the block's DC/LP value.
    pub(in crate::tile_decode) fn push(
        &mut self,
        low: &[i32; 16],
        high: &[i32; 256],
        record: MacroblockRecord,
    ) -> Result<(), TileDecodeError> {
        let count = coefficients_per_macroblock(record.bands, self.block_count);
        let end = self
            .written
            .checked_add(count)
            .ok_or(TileDecodeError::ArithmeticOverflow(
                "external coefficient write",
            ))?;
        let target =
            self.coefficients
                .get_mut(self.written..end)
                .ok_or(TileDecodeError::InvalidPlan(
                    "external coefficient storage overflow",
                ))?;
        match record.bands {
            BandPresence::DcOnly => target[0] = low[0],
            BandPresence::NoHighPass => target.copy_from_slice(&low[..count]),
            BandPresence::NoFlexbits | BandPresence::All => {
                // A block at a time: a copy of a fixed size is a few loads
                // and stores, where WebAssembly without bulk memory (the
                // MVP target) calls `memcpy` for one whose size is known
                // only when it runs.
                let blocks = target.as_chunks_mut::<16>().0;
                for ((block, source), &value) in
                    blocks.iter_mut().zip(high.as_chunks::<16>().0).zip(low)
                {
                    *block = *source;
                    block[0] = value;
                }
            }
        }
        self.written = end;
        self.records.push(record);
        Ok(())
    }

    /// Require a complete window and return its macroblock records.
    pub(in crate::tile_decode) fn finish(self) -> Result<Vec<MacroblockRecord>, TileDecodeError> {
        if self.records.len() != self.macroblocks {
            return Err(TileDecodeError::InvalidPlan(
                "decoded tile macroblock count",
            ));
        }
        if self.written != self.coefficients.len() {
            return Err(TileDecodeError::InvalidPlan(
                "external coefficient storage fill length",
            ));
        }
        Ok(self.records)
    }
}

/// A selected tile's coefficient windows, one for each component plane, in
/// component order.
pub(in crate::tile_decode) struct TileWriters<'a, T> {
    /// What names the tile: its index, or the reader that fills it.
    pub(in crate::tile_decode) tile: T,
    /// Macroblocks each window stores.
    macroblocks: usize,
    pub(in crate::tile_decode) writers: Vec<PlaneWriter<'a>>,
}

/// Split exact-size storage into plane-major windows for each selected tile.
///
/// Planes are stored in component order; within a plane, tiles follow plan
/// order and macroblocks follow tile raster order.
pub(in crate::tile_decode) fn split_planes<'a>(
    destination: &'a mut [i32],
    layouts: &[PlaneLayout],
    tiles: &[TilePlan],
) -> Result<Vec<TileWriters<'a, usize>>, TileDecodeError> {
    let mut selected = Vec::with_capacity(tiles.len());
    for (index, tile) in tiles.iter().enumerate() {
        if tile.required_for_reconstruction {
            let macroblocks = usize::try_from(tile.macroblock_count)
                .map_err(|_| TileDecodeError::ArithmeticOverflow("plane macroblock count"))?;
            selected.push((index, macroblocks));
        }
    }
    split_windows(destination, layouts, selected)
}

/// Split exact-size storage into plane-major windows for the tiles of
/// `tiles`, each given with the macroblocks it stores: all of a tile's, or
/// those of some of its macroblock rows.
pub(in crate::tile_decode) fn split_windows<'a, T>(
    destination: &'a mut [i32],
    layouts: &[PlaneLayout],
    tiles: impl IntoIterator<Item = (T, usize)>,
) -> Result<Vec<TileWriters<'a, T>>, TileDecodeError> {
    let mut selected: Vec<TileWriters<'a, T>> = tiles
        .into_iter()
        .map(|(tile, macroblocks)| TileWriters {
            tile,
            macroblocks,
            writers: Vec::with_capacity(layouts.len()),
        })
        .collect();
    let mut rest = destination;
    for layout in layouts {
        for selected in &mut selected {
            let macroblocks = selected.macroblocks;
            let length = macroblocks
                .checked_mul(layout.coefficients_per_macroblock)
                .ok_or(TileDecodeError::ArithmeticOverflow(
                    "coefficient arena size",
                ))?;
            let (window, tail) = core::mem::take(&mut rest)
                .split_at_mut_checked(length)
                .ok_or(TileDecodeError::InvalidPlan(
                    "external coefficient storage length",
                ))?;
            selected.writers.push(PlaneWriter::new(
                window,
                layout.geometry.block_count(),
                macroblocks,
            ));
            rest = tail;
        }
    }
    if !rest.is_empty() {
        return Err(TileDecodeError::InvalidPlan(
            "external coefficient storage length",
        ));
    }
    Ok(selected)
}

/// Append plane descriptors and macroblock metadata for decoded tile records.
///
/// Returns the number of coefficients the records describe.
pub(in crate::tile_decode) fn append_metadata(
    metadata: &mut MacroblockMetadata,
    planes: &mut Vec<CoefficientPlane>,
    decoded: &[DecodedRows],
    layouts: &[PlaneLayout],
) -> Result<usize, TileDecodeError> {
    if decoded
        .iter()
        .any(|tile| tile.records.len() != layouts.len())
    {
        return Err(TileDecodeError::InvalidPlan("decoded component count"));
    }
    let mut coefficients = 0_usize;
    for (component, layout) in layouts.iter().enumerate() {
        let coefficient_offset = coefficients;
        let macroblock_offset = metadata.len();
        for tile in decoded {
            coefficients = append_tile(
                metadata,
                &tile.records[component],
                tile,
                layout.geometry.block_count(),
                coefficients,
            )?;
        }
        planes.push(CoefficientPlane {
            coefficient_offset,
            coefficient_count: coefficients - coefficient_offset,
            macroblock_offset,
            macroblock_count: metadata.len() - macroblock_offset,
            block_columns: layout.geometry.columns,
            block_rows: layout.geometry.rows,
        });
    }
    Ok(coefficients)
}

/// Appends the metadata of one component's records of the rows `tile`
/// holds of its tile.
fn append_tile(
    metadata: &mut MacroblockMetadata,
    records: &[MacroblockRecord],
    tile: &DecodedRows,
    block_count: usize,
    mut coefficients: usize,
) -> Result<usize, TileDecodeError> {
    let location = tile.location;
    let width = usize::try_from(location.width)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("tile width"))?;
    let height = usize::try_from(location.height)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("tile height"))?;
    let first_row = usize::try_from(tile.rows.start)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("tile row"))?;
    let rows = usize::try_from(tile.rows.end - tile.rows.start)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("tile rows"))?;
    let expected = width
        .checked_mul(rows)
        .ok_or(TileDecodeError::ArithmeticOverflow("tile traversal"))?;
    if records.len() != expected {
        return Err(TileDecodeError::InvalidPlan(
            "decoded tile macroblock count",
        ));
    }
    for (index, record) in records.iter().enumerate() {
        let x = index % width;
        let y = first_row + index / width;
        let coefficient_offset = u32::try_from(coefficients)
            .map_err(|_| TileDecodeError::ArithmeticOverflow("coefficient offset"))?;
        metadata.coefficient_offsets.push(coefficient_offset);
        metadata.quantizers.push(record.quantizers);
        metadata.bands.push(record.bands);
        metadata.predictions.push(record.prediction);
        metadata.hp_predictions.push(record.hp_prediction);
        metadata
            .tile_edges
            .push(tile_edges(x, y, width, height, location.hard));
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
        coefficients = coefficients
            .checked_add(coefficients_per_macroblock(record.bands, block_count))
            .ok_or(TileDecodeError::ArithmeticOverflow(
                "coefficient arena size",
            ))?;
    }
    Ok(coefficients)
}
