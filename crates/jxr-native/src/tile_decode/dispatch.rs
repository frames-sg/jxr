//! Tile packet routing: one reader for each tile, read a macroblock row at
//! a time into coefficient windows.
//!
//! [`TileRows`] is the one reader of a tile. [`decode_packets`] runs every
//! tile to its end, which fills a coefficient arena that holds the whole
//! image; [`BandDecoder`] takes the tiles of each tile row down the image
//! together, a band of macroblock rows at a time, for a decode that
//! reconstructs each band before it reads the next.

use core::ops::Range;

use jxr_core::{BitstreamMode, CoefficientArena, PreparedPlan, TilePlan};

use crate::{ParsedCodestream, executor::Executor};

use super::{
    TileDecodeError, TileLocation,
    frequency::{self, FrequencyPacketRanges, FrequencyRows},
    output::{self, MacroblockRecord, PlaneLayout, PlaneWriter, TileWriters},
    packet_slice, plane_layouts, prepare_reusable_arena,
    spatial_tile::{SpatialRows, TilePlanes},
    tile_locations, validate_decode_scope, validate_tile_count,
};

/// Macroblock records of some rows of one decoded tile, per component.
pub(super) struct DecodedRows {
    pub(super) records: Vec<Vec<MacroblockRecord>>,
    pub(super) location: TileLocation,
    /// The macroblock rows of the tile the records are of.
    pub(super) rows: Range<u32>,
}

/// One tile's packets and the state of reading them, in either packet
/// layout.
pub(super) enum TileRows<'a> {
    Spatial(Box<SpatialRows<'a>>),
    Frequency(Box<FrequencyRows<'a>>),
}

impl<'a> TileRows<'a> {
    /// Reads the packet headers of the tile at `location`, which `tile`
    /// plans, and whose band packets `frequency_ranges` locates when the
    /// image is in frequency mode.
    fn open(
        source: &'a [u8],
        parsed: &'a ParsedCodestream,
        plan: &PreparedPlan,
        frequency_ranges: Option<FrequencyPacketRanges>,
        location: TileLocation,
        tile: &TilePlan,
    ) -> Result<Self, TileDecodeError> {
        let expected_macroblocks = location
            .width
            .checked_mul(location.height)
            .ok_or(TileDecodeError::ArithmeticOverflow("tile macroblock count"))?;
        if tile.macroblock_count != expected_macroblocks {
            return Err(TileDecodeError::InvalidPlan("tile macroblock count"));
        }
        let alpha = match &parsed.headers.alpha {
            Some(alpha) => {
                let bands = plan
                    .info
                    .alpha
                    .as_ref()
                    .ok_or(TileDecodeError::InvalidPlan("integrated alpha plane"))?
                    .bands;
                Some((alpha, plan.scale.retained_bands(bands)))
            }
            None => None,
        };
        let planes = TilePlanes {
            primary: &parsed.headers.primary,
            primary_colour: plan.info.primary.color_format,
            primary_bands: plan.scale.retained_bands(plan.info.primary.bands),
            alpha,
            trim_present: parsed.headers.image.flags.trim_flexbits(),
        };
        let width = usize::try_from(location.width)
            .map_err(|_| TileDecodeError::ArithmeticOverflow("tile width conversion"))?;
        let height = usize::try_from(location.height)
            .map_err(|_| TileDecodeError::ArithmeticOverflow("tile height conversion"))?;
        Ok(match frequency_ranges {
            Some(ranges) => Self::Frequency(Box::new(FrequencyRows::open(
                source, planes, ranges, width, height,
            )?)),
            None => Self::Spatial(Box::new(SpatialRows::open(
                packet_slice(source, tile.packet_range)?,
                planes,
                width,
            )?)),
        })
    }

    /// Reads macroblock row `y` of the tile, which follows the rows already
    /// read, into the writers of its components.
    fn decode_row(
        &mut self,
        y: usize,
        writers: &mut [PlaneWriter<'_>],
    ) -> Result<(), TileDecodeError> {
        match self {
            Self::Spatial(rows) => rows.decode_row(y, writers),
            Self::Frequency(rows) => rows.decode_row(y, writers),
        }
    }

    /// Reads what ends the tile's packets after its last row.
    fn finish(&mut self) -> Result<(), TileDecodeError> {
        match self {
            Self::Spatial(rows) => rows.finish(),
            Self::Frequency(rows) => rows.finish(),
        }
    }
}

/// Where the band packets of each tile are, for an image in frequency mode.
fn frequency_ranges(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    tiles: usize,
) -> Result<Option<Vec<FrequencyPacketRanges>>, TileDecodeError> {
    if plan.info.primary.bitstream_mode == BitstreamMode::Frequency {
        frequency::packet_ranges(source.len(), parsed, plan.info.primary.bands, tiles).map(Some)
    } else {
        Ok(None)
    }
}

/// Reads every selected tile to its end, each into its own windows.
pub(super) fn decode_packets(
    source: &[u8],
    parsed: &ParsedCodestream,
    plan: &PreparedPlan,
    locations: &[TileLocation],
    mut tiles: Vec<TileWriters<'_, usize>>,
    executor: Executor,
) -> Result<Vec<DecodedRows>, TileDecodeError> {
    let frequency_ranges = frequency_ranges(source, parsed, plan, locations.len())?;
    let decode = |selected: TileWriters<'_, usize>| {
        let (index, mut writers) = (selected.tile, selected.writers);
        let location = locations[index];
        let mut rows = TileRows::open(
            source,
            parsed,
            plan,
            frequency_ranges.as_deref().map(|ranges| ranges[index]),
            location,
            &plan.tiles[index],
        )?;
        for y in 0..location.height as usize {
            rows.decode_row(y, &mut writers)?;
        }
        rows.finish()?;
        let records = writers
            .into_iter()
            .map(PlaneWriter::finish)
            .collect::<Result<_, _>>()?;
        Ok(DecodedRows {
            records,
            location,
            rows: 0..location.height,
        })
    };
    match tiles.len() {
        0 => Err(TileDecodeError::InvalidPlan("empty tile grid")),
        1 => Ok(vec![decode(tiles.remove(0))?]),
        _ => executor.try_map(tiles, decode),
    }
}

/// The selected tiles of a prepared plan, read a band of macroblock rows at
/// a time from the top of the image to the bottom.
///
/// The tiles of one tile row are read together, each a row at a time, so
/// that a band is the same macroblock rows of every one of them. Each band
/// fills a coefficient arena laid out as [`super::decode_tiles`] lays out
/// the whole image, holding only the band's macroblocks.
pub(crate) struct BandDecoder<'a> {
    source: &'a [u8],
    parsed: &'a ParsedCodestream,
    plan: &'a PreparedPlan,
    layouts: Vec<PlaneLayout>,
    locations: Vec<TileLocation>,
    frequency_ranges: Option<Vec<FrequencyPacketRanges>>,
    /// The selected tiles of each tile row that has any, top to bottom.
    tile_rows: Vec<Vec<usize>>,
    /// Tile rows read to their end.
    finished_tile_rows: usize,
    /// The tiles of the tile row being read, with their readers.
    open: Vec<(usize, TileRows<'a>)>,
    /// Macroblock rows of the tile row being read that have been read.
    row: u32,
    /// Where the tiles of a band are read.
    executor: Executor,
}

impl<'a> BandDecoder<'a> {
    /// Prepares to read the tiles `plan` selects, the tiles of a band side
    /// by side through `executor`.
    pub(crate) fn new(
        source: &'a [u8],
        parsed: &'a ParsedCodestream,
        plan: &'a PreparedPlan,
        executor: Executor,
    ) -> Result<Self, TileDecodeError> {
        validate_decode_scope(parsed)?;
        validate_tile_count(plan)?;
        let locations = tile_locations(parsed, plan, plan.tiles.len())?;
        let columns = plan.info.tiles.column_widths.len().max(1);
        let tile_rows: Vec<Vec<usize>> = (0..plan.tiles.len())
            .collect::<Vec<_>>()
            .chunks(columns)
            .map(|row| {
                row.iter()
                    .copied()
                    .filter(|&index| plan.tiles[index].required_for_reconstruction)
                    .collect::<Vec<_>>()
            })
            .filter(|row| !row.is_empty())
            .collect();
        if tile_rows.is_empty() {
            return Err(TileDecodeError::InvalidPlan("empty tile grid"));
        }
        Ok(Self {
            source,
            parsed,
            plan,
            layouts: plane_layouts(parsed, plan),
            frequency_ranges: frequency_ranges(source, parsed, plan, locations.len())?,
            locations,
            tile_rows,
            finished_tile_rows: 0,
            open: Vec::new(),
            row: 0,
            executor,
        })
    }

    /// Limits the reader to the tile rows `tile_rows`, counted among those
    /// that hold a selected tile, which must be done before any band is
    /// read: several readers then take a share of the image each.
    pub(crate) fn restrict(&mut self, tile_rows: Range<usize>) {
        self.tile_rows.truncate(tile_rows.end);
        self.finished_tile_rows = tile_rows.start.min(self.tile_rows.len());
    }

    /// Reads the next macroblock rows, at most `rows` of them and never
    /// past the end of a tile row, into `arena`, whose storage is reused.
    /// Returns how many rows it read, which is zero after the last, and
    /// whether every arena allocation needed for the band already had
    /// sufficient capacity.
    pub(crate) fn decode_band(
        &mut self,
        rows: u32,
        arena: &mut CoefficientArena,
    ) -> Result<(u32, bool), TileDecodeError> {
        if self.open.is_empty() {
            let Some(tiles) = self.tile_rows.get(self.finished_tile_rows) else {
                return Ok((0, false));
            };
            for &index in tiles {
                let ranges = self.frequency_ranges.as_deref().map(|ranges| ranges[index]);
                let reader = TileRows::open(
                    self.source,
                    self.parsed,
                    self.plan,
                    ranges,
                    self.locations[index],
                    &self.plan.tiles[index],
                )?;
                self.open.push((index, reader));
            }
            self.row = 0;
        }
        let height = self.locations[self.open[0].0].height;
        let band = self.row..self.row.saturating_add(rows.max(1)).min(height);
        let last = band.end == height;
        let locations = &self.locations;
        let macroblocks = |index: usize| locations[index].width as usize * band.len();
        let plane_macroblocks: usize = self.open.iter().map(|(index, _)| macroblocks(*index)).sum();
        let coefficient_count = self
            .layouts
            .iter()
            .map(|layout| layout.coefficients_per_macroblock * plane_macroblocks)
            .sum();
        let reused = prepare_reusable_arena(
            arena,
            coefficient_count,
            plane_macroblocks * self.layouts.len(),
            self.layouts.len(),
        );
        let open = self.open.len();
        let tiles = self.open.iter_mut().map(|tile| {
            let count = macroblocks(tile.0);
            (tile, count)
        });
        let writers = output::split_windows(&mut arena.coefficients, &self.layouts, tiles)?;
        let read = |selected: TileWriters<'_, &mut (usize, TileRows<'a>)>| {
            let ((index, reader), mut writers) = (selected.tile, selected.writers);
            for y in band.clone() {
                reader.decode_row(y as usize, &mut writers)?;
            }
            if last {
                reader.finish()?;
            }
            let records = writers
                .into_iter()
                .map(PlaneWriter::finish)
                .collect::<Result<_, _>>()?;
            Ok::<_, TileDecodeError>(DecodedRows {
                records,
                location: locations[*index],
                rows: band.clone(),
            })
        };
        let executor = if open > 1 {
            self.executor
        } else {
            Executor::Sequential
        };
        let decoded = executor.try_map(writers, read)?;
        output::append_metadata(
            &mut arena.macroblocks,
            &mut arena.planes,
            &decoded,
            &self.layouts,
        )?;
        self.row = band.end;
        if last {
            self.open.clear();
            self.finished_tile_rows += 1;
        }
        Ok((band.end - band.start, reused))
    }
}
