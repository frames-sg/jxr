//! Tile packets written from quantized macroblocks.
//!
//! This is the inverse of [`crate::tile_decode`]: it forms the same
//! predictions from the same neighbours, drives the same adaptive entropy
//! state in the same order, and writes the syntax that state selects.

mod error;
mod high_pass;
mod low_pass;
mod packets;
mod tile;

use jxr_core::{BandPresence, BitstreamMode, ColorFormat};

use crate::reconstruct::QuantizedMacroblock;

pub(crate) use error::TileEncodeError;
#[cfg(test)]
pub(crate) use tile::{EncodedTile, PlaneState, encode_tile_with_state};
pub(crate) use tile::{TileScratch, TileShape, TileWriter};

pub(crate) use crate::tile_decode::syntax::MAX_COMPONENTS;

/// A rectangle of a tile within raster-ordered macroblocks: a whole tile
/// within an image's, or the rows of a tile within a band of rows.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TileRegion {
    /// Width of the macroblock rows in macroblocks.
    pub(crate) stride: usize,
    /// Origin of the rectangle among the rows, in macroblocks.
    pub(crate) x: usize,
    pub(crate) y: usize,
    /// Size of the rectangle in macroblocks.
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl TileRegion {
    /// The macroblock at `(x, y)` of the rectangle, in one component.
    fn macroblock<'a>(
        &self,
        macroblocks: &'a [QuantizedMacroblock],
        x: usize,
        y: usize,
    ) -> Result<&'a QuantizedMacroblock, TileEncodeError> {
        (self.y + y)
            .checked_mul(self.stride)
            .and_then(|row| row.checked_add(self.x + x))
            .and_then(|index| macroblocks.get(index))
            .ok_or(TileEncodeError::InvalidInput("tile macroblock range"))
    }
}

/// Syntax choices shared by every macroblock of a tile.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PacketOptions {
    /// Internal colour format the primary plane's components are coded in.
    pub(crate) colour: ColorFormat,
    /// Bands the image plane headers declare; an integrated alpha plane
    /// codes the same ones.
    pub(crate) bands: BandPresence,
    /// `TRIM_FLEXBITS`, written only when the image header enables it.
    pub(crate) trim_flexbits: Option<u8>,
    /// Whether the tile is one packet or one packet for each band.
    pub(crate) mode: BitstreamMode,
}
