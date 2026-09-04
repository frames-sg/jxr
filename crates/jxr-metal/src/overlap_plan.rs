// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{MetalError, abi::JxrOverlapWorkAbi, plan::MetalPlaneInput};

#[derive(Debug, Default)]
pub(crate) struct OverlapSchedule {
    pub(crate) prefix: Vec<JxrOverlapWorkAbi>,
    pub(crate) filters: Vec<JxrOverlapWorkAbi>,
    pub(crate) suffix: Vec<JxrOverlapWorkAbi>,
}

pub(crate) fn first_overlap_schedule(
    plane: MetalPlaneInput,
    hard_tiles: bool,
    tile_columns: &[u32],
    tile_rows: &[u32],
) -> Result<OverlapSchedule, MetalError> {
    let regions = regions(
        plane,
        hard_tiles,
        tile_columns,
        tile_rows,
        u32::from(plane.block_columns),
        u32::from(plane.block_rows),
    )?;
    if plane.block_columns == 4 {
        full_schedule(plane.low_offset, plane.macroblocks_x * 4, &regions)
    } else {
        subsampled_schedule(
            plane.low_offset,
            plane.macroblocks_x * u32::from(plane.block_columns),
            &regions,
        )
    }
}

pub(crate) fn second_overlap_schedule(
    plane: MetalPlaneInput,
    hard_tiles: bool,
    tile_columns: &[u32],
    tile_rows: &[u32],
) -> Result<OverlapSchedule, MetalError> {
    let horizontal = u32::from(plane.block_columns) * 4;
    let vertical = u32::from(plane.block_rows) * 4;
    let regions = regions(
        plane,
        hard_tiles,
        tile_columns,
        tile_rows,
        horizontal,
        vertical,
    )?;
    full_schedule(plane.sample_offset, plane.sample_width, &regions)
}

#[derive(Clone, Copy)]
struct Region {
    left: u32,
    right: u32,
    top: u32,
    bottom: u32,
}

fn regions(
    plane: MetalPlaneInput,
    hard_tiles: bool,
    tile_columns: &[u32],
    tile_rows: &[u32],
    horizontal_scale: u32,
    vertical_scale: u32,
) -> Result<Vec<Region>, MetalError> {
    if !hard_tiles {
        return Ok(vec![Region {
            left: 0,
            right: plane.macroblocks_x * horizontal_scale,
            top: 0,
            bottom: plane.macroblocks_y * vertical_scale,
        }]);
    }
    let columns = clipped_axis(tile_columns, plane.macroblock_origin_x, plane.macroblocks_x)?;
    let rows = clipped_axis(tile_rows, plane.macroblock_origin_y, plane.macroblocks_y)?;
    let mut output = Vec::with_capacity(
        columns
            .len()
            .saturating_sub(1)
            .saturating_mul(rows.len().saturating_sub(1)),
    );
    for vertical in rows.windows(2) {
        for horizontal in columns.windows(2) {
            output.push(Region {
                left: scaled(horizontal[0], horizontal_scale)?,
                right: scaled(horizontal[1], horizontal_scale)?,
                top: scaled(vertical[0], vertical_scale)?,
                bottom: scaled(vertical[1], vertical_scale)?,
            });
        }
    }
    Ok(output)
}

fn clipped_axis(sizes: &[u32], origin: u32, extent: u32) -> Result<Vec<u32>, MetalError> {
    let end = origin
        .checked_add(extent)
        .ok_or_else(|| invalid("tile axis extent overflows u32"))?;
    let mut boundaries = vec![0];
    let mut position = 0_u32;
    for &size in sizes {
        position = position
            .checked_add(size)
            .ok_or_else(|| invalid("tile boundary overflows u32"))?;
        if position > origin && position < end {
            boundaries.push(position - origin);
        }
    }
    boundaries.push(extent);
    if position < end {
        return Err(invalid(
            "tile partition does not cover reconstruction window",
        ));
    }
    Ok(boundaries)
}

fn full_schedule(
    plane_offset: usize,
    stride: u32,
    regions: &[Region],
) -> Result<OverlapSchedule, MetalError> {
    let mut filters = Vec::new();
    for &region in regions {
        let mut y = region.top + 2;
        while y + 2 < region.bottom {
            let mut x = region.left + 2;
            while x + 2 < region.right {
                filters.push(work(absolute(plane_offset, x, y, stride)?, stride, 0));
                x += 4;
            }
            y += 4;
        }
        let mut y = region.top + 2;
        while y + 2 < region.bottom {
            for x in [
                region.left,
                region.left + 1,
                region.right - 2,
                region.right - 1,
            ] {
                filters.push(work(absolute(plane_offset, x, y, stride)?, stride, 1));
            }
            y += 4;
        }
        let mut x = region.left + 2;
        while x + 2 < region.right {
            for y in [
                region.top,
                region.top + 1,
                region.bottom - 2,
                region.bottom - 1,
            ] {
                filters.push(work(absolute(plane_offset, x, y, stride)?, 1, 1));
            }
            x += 4;
        }
        for (x, y) in [
            (region.left, region.top),
            (region.right - 2, region.top),
            (region.left, region.bottom - 2),
            (region.right - 2, region.bottom - 2),
        ] {
            filters.push(work(absolute(plane_offset, x, y, stride)?, stride, 2));
        }
    }
    Ok(OverlapSchedule {
        filters,
        ..OverlapSchedule::default()
    })
}

fn subsampled_schedule(
    plane_offset: usize,
    stride: u32,
    regions: &[Region],
) -> Result<OverlapSchedule, MetalError> {
    let mut schedule = OverlapSchedule::default();
    for &region in regions {
        for (destination, source) in corner_residuals(plane_offset, stride, region)? {
            schedule.prefix.push(work(destination, source, 5));
        }
        for y in (region.top + 2..region.bottom).step_by(2) {
            for x in (region.left + 2..region.right).step_by(2) {
                schedule.filters.push(work(
                    absolute(plane_offset, x - 1, y - 1, stride)?,
                    stride,
                    3,
                ));
            }
        }
        for y in (region.top + 2..region.bottom).step_by(2) {
            schedule.filters.push(work(
                absolute(plane_offset, region.left, y - 1, stride)?,
                absolute(plane_offset, region.left, y, stride)?,
                4,
            ));
            schedule.filters.push(work(
                absolute(plane_offset, region.right - 1, y - 1, stride)?,
                absolute(plane_offset, region.right - 1, y, stride)?,
                4,
            ));
        }
        for x in (region.left + 2..region.right).step_by(2) {
            schedule.filters.push(work(
                absolute(plane_offset, x - 1, region.top, stride)?,
                absolute(plane_offset, x, region.top, stride)?,
                4,
            ));
            schedule.filters.push(work(
                absolute(plane_offset, x - 1, region.bottom - 1, stride)?,
                absolute(plane_offset, x, region.bottom - 1, stride)?,
                4,
            ));
        }
        for (destination, source) in corner_residuals(plane_offset, stride, region)? {
            schedule.suffix.push(work(destination, source, 6));
        }
    }
    Ok(schedule)
}

fn corner_residuals(
    plane_offset: usize,
    stride: u32,
    region: Region,
) -> Result<[(u32, u32); 4], MetalError> {
    Ok([
        (
            absolute(plane_offset, region.left, region.top, stride)?,
            absolute(plane_offset, region.left + 1, region.top, stride)?,
        ),
        (
            absolute(plane_offset, region.right - 1, region.top, stride)?,
            absolute(plane_offset, region.right - 2, region.top, stride)?,
        ),
        (
            absolute(plane_offset, region.left, region.bottom - 1, stride)?,
            absolute(plane_offset, region.left + 1, region.bottom - 1, stride)?,
        ),
        (
            absolute(plane_offset, region.right - 1, region.bottom - 1, stride)?,
            absolute(plane_offset, region.right - 2, region.bottom - 1, stride)?,
        ),
    ])
}

fn absolute(offset: usize, x: u32, y: u32, stride: u32) -> Result<u32, MetalError> {
    usize::try_from(y)
        .ok()
        .and_then(|row| row.checked_mul(usize::try_from(stride).ok()?))
        .and_then(|index| index.checked_add(usize::try_from(x).ok()?))
        .and_then(|index| index.checked_add(offset))
        .and_then(|index| u32::try_from(index).ok())
        .ok_or_else(|| invalid("overlap sample index exceeds the Metal ABI"))
}

const fn work(first: u32, second: u32, kind: u32) -> JxrOverlapWorkAbi {
    JxrOverlapWorkAbi {
        first,
        second,
        kind,
        reserved: 0,
    }
}

fn scaled(value: u32, scale: u32) -> Result<u32, MetalError> {
    value
        .checked_mul(scale)
        .ok_or_else(|| invalid("scaled tile boundary overflows u32"))
}

const fn invalid(reason: &'static str) -> MetalError {
    MetalError::InvalidPlan { reason }
}

#[cfg(test)]
mod tests {
    use super::{Region, full_schedule};

    #[test]
    fn full_schedule_centres_interior_work_inside_two_sample_edges() {
        let schedule = full_schedule(
            0,
            16,
            &[Region {
                left: 0,
                right: 16,
                top: 0,
                bottom: 16,
            }],
        )
        .unwrap();
        assert_eq!(schedule.filters[0].first, 2 + 2 * 16);
        assert_eq!(schedule.filters[1].first, 6 + 2 * 16);
        assert_eq!(schedule.filters[3].first, 2 + 6 * 16);
    }
}
