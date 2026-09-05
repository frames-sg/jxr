// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{MetalError, plan::MetalPlaneInput};
pub(crate) use jxr_core::device_plan::OverlapSchedule;

pub(crate) fn first_overlap_schedule(
    plane: MetalPlaneInput,
    hard_tiles: bool,
    tile_columns: &[u32],
    tile_rows: &[u32],
) -> Result<OverlapSchedule, MetalError> {
    jxr_core::device_plan::first_overlap_schedule(plane, hard_tiles, tile_columns, tile_rows)
        .map_err(Into::into)
}

pub(crate) fn second_overlap_schedule(
    plane: MetalPlaneInput,
    hard_tiles: bool,
    tile_columns: &[u32],
    tile_rows: &[u32],
) -> Result<OverlapSchedule, MetalError> {
    jxr_core::device_plan::second_overlap_schedule(plane, hard_tiles, tile_columns, tile_rows)
        .map_err(Into::into)
}
