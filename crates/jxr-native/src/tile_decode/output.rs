//! Plane-major coefficient windows written directly by tile decoders.

use jxr_core::{
    BandPresence, CoefficientPlane, MacroblockMetadata, PredictionMode, QuantizerSet, TilePlan,
};

use super::{
    ComponentGeometry, TileDecodeError, TileLocation, coefficients_per_macroblock,
    spatial::SpatialMacroblock, tile_edges,
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
                target.copy_from_slice(&high[..count]);
                for (slot, &value) in target.iter_mut().step_by(16).zip(low) {
                    *slot = value;
                }
            }
        }
        self.written = end;
        self.records.push(record);
        Ok(())
    }

    /// Store a macroblock produced by a frequency-mode traversal.
    pub(in crate::tile_decode) fn push_macroblock(
        &mut self,
        macroblock: &SpatialMacroblock,
    ) -> Result<(), TileDecodeError> {
        self.push(
            &macroblock.coefficients.dc_low_pass,
            &macroblock.coefficients.high_pass,
            MacroblockRecord {
                quantizers: macroblock.coefficients.quantizers,
                bands: macroblock.coefficients.bands,
                prediction: macroblock.prediction,
                hp_prediction: macroblock.hp_prediction,
            },
        )
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

/// Selected tile index paired with its per-component coefficient windows.
pub(in crate::tile_decode) type TileWriters<'a> = (usize, Vec<PlaneWriter<'a>>);

/// Split exact-size storage into plane-major windows for each selected tile.
///
/// Planes are stored in component order; within a plane, tiles follow plan
/// order and macroblocks follow tile raster order.
pub(in crate::tile_decode) fn split_planes<'a>(
    destination: &'a mut [i32],
    layouts: &[PlaneLayout],
    tiles: &[TilePlan],
) -> Result<Vec<TileWriters<'a>>, TileDecodeError> {
    let mut selected = Vec::with_capacity(tiles.len());
    for (index, tile) in tiles.iter().enumerate() {
        if tile.required_for_reconstruction {
            let macroblocks = usize::try_from(tile.macroblock_count)
                .map_err(|_| TileDecodeError::ArithmeticOverflow("plane macroblock count"))?;
            selected.push((index, macroblocks, Vec::with_capacity(layouts.len())));
        }
    }
    let mut rest = destination;
    for layout in layouts {
        for (_, macroblocks, writers) in &mut selected {
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
            writers.push(PlaneWriter::new(
                window,
                layout.geometry.block_count(),
                *macroblocks,
            ));
            rest = tail;
        }
    }
    if !rest.is_empty() {
        return Err(TileDecodeError::InvalidPlan(
            "external coefficient storage length",
        ));
    }
    Ok(selected
        .into_iter()
        .map(|(index, _, writers)| (index, writers))
        .collect())
}

/// Append plane descriptors and macroblock metadata for decoded tile records.
///
/// Returns the number of coefficients the records describe.
pub(in crate::tile_decode) fn append_metadata(
    metadata: &mut MacroblockMetadata,
    planes: &mut Vec<CoefficientPlane>,
    decoded: &[(Vec<Vec<MacroblockRecord>>, TileLocation)],
    layouts: &[PlaneLayout],
) -> Result<usize, TileDecodeError> {
    if decoded.iter().any(|(tile, _)| tile.len() != layouts.len()) {
        return Err(TileDecodeError::InvalidPlan("decoded component count"));
    }
    let mut coefficients = 0_usize;
    for (component, layout) in layouts.iter().enumerate() {
        let coefficient_offset = coefficients;
        let macroblock_offset = metadata.len();
        for (tile, location) in decoded {
            coefficients = append_tile(
                metadata,
                &tile[component],
                *location,
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

fn append_tile(
    metadata: &mut MacroblockMetadata,
    records: &[MacroblockRecord],
    location: TileLocation,
    block_count: usize,
    mut coefficients: usize,
) -> Result<usize, TileDecodeError> {
    let width = usize::try_from(location.width)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("tile width"))?;
    let height = usize::try_from(location.height)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("tile height"))?;
    let expected = width
        .checked_mul(height)
        .ok_or(TileDecodeError::ArithmeticOverflow("tile traversal"))?;
    if records.len() != expected {
        return Err(TileDecodeError::InvalidPlan(
            "decoded tile macroblock count",
        ));
    }
    for (index, record) in records.iter().enumerate() {
        let x = index % width;
        let y = index / width;
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
