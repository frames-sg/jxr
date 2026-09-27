// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{cell::RefCell, collections::VecDeque, rc::Rc};

use j2k_metal_support::checked_shared_buffer_with_slice;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLDevice};

use crate::{MetalError, abi::JxrOverlapWorkAbi, plan::MetalPlaneInput};

// Distinct plane geometries whose uploaded schedules stay resident. Batches of
// equally sized tiles reuse one entry per plane shape.
const RETAINED_SCHEDULES: usize = 64;

/// Overlap filtering pass: low-pass (mode two only) or full-resolution samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OverlapPass {
    First,
    Second,
}

/// One uploaded overlap work list.
pub(crate) struct ResidentWork {
    pub(crate) buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) count: u32,
}

/// Plane-relative overlap work uploaded once per plane geometry.
///
/// Indices start at zero for the plane's first sample; each dispatch passes
/// the plane's base offset so one upload serves every image of that shape.
pub(crate) struct ResidentOverlapSchedule {
    pub(crate) prefix: Option<ResidentWork>,
    pub(crate) filters: Option<ResidentWork>,
    pub(crate) suffix: Option<ResidentWork>,
    /// Largest relative sample index any work item reads or writes.
    max_index: u32,
}

impl ResidentOverlapSchedule {
    /// Validates that rebasing every work item stays inside the device ABI.
    pub(crate) fn checked_base(&self, base: usize) -> Result<u32, MetalError> {
        u32::try_from(base)
            .ok()
            .filter(|&base| base.checked_add(self.max_index).is_some())
            .ok_or(MetalError::InvalidPlan {
                reason: "overlap sample index exceeds the device ABI",
            })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ScheduleKey {
    pass: OverlapPass,
    block_columns: u8,
    block_rows: u8,
    macroblocks_x: u32,
    macroblocks_y: u32,
    sample_width: u32,
    hard_tiles: Option<HardTileKey>,
}

/// Hard-tile partitions depend on the window origin and the complete tile grid.
#[derive(Clone, Debug, PartialEq, Eq)]
struct HardTileKey {
    macroblock_origin_x: u32,
    macroblock_origin_y: u32,
    tile_columns: Vec<u32>,
    tile_rows: Vec<u32>,
}

/// Least-recently-used cache of uploaded plane-relative overlap schedules.
pub(crate) struct OverlapScheduleCache {
    entries: RefCell<VecDeque<(ScheduleKey, Rc<ResidentOverlapSchedule>)>>,
}

impl OverlapScheduleCache {
    pub(crate) fn new() -> Self {
        Self {
            entries: RefCell::new(VecDeque::new()),
        }
    }

    pub(crate) fn get_or_upload(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        pass: OverlapPass,
        plane: MetalPlaneInput,
        hard_tiles: bool,
        tile_columns: &[u32],
        tile_rows: &[u32],
    ) -> Result<Rc<ResidentOverlapSchedule>, MetalError> {
        let key = ScheduleKey {
            pass,
            block_columns: plane.block_columns,
            block_rows: plane.block_rows,
            macroblocks_x: plane.macroblocks_x,
            macroblocks_y: plane.macroblocks_y,
            sample_width: plane.sample_width,
            hard_tiles: hard_tiles.then(|| HardTileKey {
                macroblock_origin_x: plane.macroblock_origin_x,
                macroblock_origin_y: plane.macroblock_origin_y,
                tile_columns: tile_columns.to_vec(),
                tile_rows: tile_rows.to_vec(),
            }),
        };
        let mut entries = self.entries.borrow_mut();
        if let Some(position) = entries.iter().position(|(cached, _)| *cached == key) {
            let entry = entries.remove(position).ok_or(MetalError::StateInvariant {
                state: "Metal overlap schedule cache",
                reason: "cached schedule position is out of range",
            })?;
            let schedule = entry.1.clone();
            entries.push_back(entry);
            return Ok(schedule);
        }
        let relative = MetalPlaneInput {
            low_offset: 0,
            sample_offset: 0,
            ..plane
        };
        let schedule = match pass {
            OverlapPass::First => jxr_core::device_plan::first_overlap_schedule(
                relative,
                hard_tiles,
                tile_columns,
                tile_rows,
            ),
            OverlapPass::Second => jxr_core::device_plan::second_overlap_schedule(
                relative,
                hard_tiles,
                tile_columns,
                tile_rows,
            ),
        }?;
        let max_index = [&schedule.prefix, &schedule.filters, &schedule.suffix]
            .into_iter()
            .flatten()
            .map(max_accessed_index)
            .try_fold(0_u32, |maximum, index| Some(maximum.max(index?)))
            .ok_or(MetalError::InvalidPlan {
                reason: "overlap sample index exceeds the device ABI",
            })?;
        let resident = Rc::new(ResidentOverlapSchedule {
            prefix: upload(device, &schedule.prefix)?,
            filters: upload(device, &schedule.filters)?,
            suffix: upload(device, &schedule.suffix)?,
            max_index,
        });
        if entries.len() == RETAINED_SCHEDULES {
            entries.pop_front();
        }
        entries.push_back((key, resident.clone()));
        Ok(resident)
    }
}

/// Largest sample index touched by one work item, mirroring the kernel's addressing.
fn max_accessed_index(work: &JxrOverlapWorkAbi) -> Option<u32> {
    let [first, second, kind, _] = *work;
    match kind {
        0 => second.checked_mul(3)?.checked_add(first)?.checked_add(3),
        1 => second.checked_mul(3)?.checked_add(first),
        2 | 3 => second.checked_add(first)?.checked_add(1),
        _ => Some(first.max(second)),
    }
}

fn upload(
    device: &ProtocolObject<dyn MTLDevice>,
    work: &[JxrOverlapWorkAbi],
) -> Result<Option<ResidentWork>, MetalError> {
    if work.is_empty() {
        return Ok(None);
    }
    Ok(Some(ResidentWork {
        buffer: checked_shared_buffer_with_slice(device, work)?,
        count: u32::try_from(work.len()).map_err(|_| MetalError::InvalidPlan {
            reason: "overlap work count exceeds the Metal ABI",
        })?,
    }))
}

#[cfg(test)]
mod tests {
    use super::{MetalPlaneInput, max_accessed_index};

    fn plane(block_columns: u8, low_offset: usize, sample_offset: usize) -> MetalPlaneInput {
        let macroblocks = 3_u32;
        MetalPlaneInput {
            arena_index: 0,
            macroblock_offset: 0,
            macroblock_count: 9,
            block_columns,
            block_rows: block_columns,
            macroblock_origin_x: 1,
            macroblock_origin_y: 1,
            macroblocks_x: macroblocks,
            macroblocks_y: macroblocks,
            sample_origin_x: 0,
            sample_origin_y: 0,
            sample_width: macroblocks * u32::from(block_columns) * 4,
            sample_height: macroblocks * u32::from(block_columns) * 4,
            low_offset,
            sample_offset,
            scale_after_first_transform: false,
            alpha: false,
        }
    }

    #[test]
    fn rebased_relative_schedules_equal_absolute_schedules() {
        for block_columns in [4, 2] {
            for hard_tiles in [false, true] {
                let absolute_plane = plane(block_columns, 4_096, 65_536);
                let relative_plane = plane(block_columns, 0, 0);
                let passes = [
                    (
                        jxr_core::device_plan::first_overlap_schedule as fn(_, _, _, _) -> _,
                        absolute_plane.low_offset,
                    ),
                    (
                        jxr_core::device_plan::second_overlap_schedule,
                        absolute_plane.sample_offset,
                    ),
                ];
                for (build, base) in passes {
                    let base = u32::try_from(base).unwrap();
                    let absolute = build(absolute_plane, hard_tiles, &[2, 2], &[1, 3]).unwrap();
                    let relative = build(relative_plane, hard_tiles, &[2, 2], &[1, 3]).unwrap();
                    for (absolute, relative) in [
                        (&absolute.prefix, &relative.prefix),
                        (&absolute.filters, &relative.filters),
                        (&absolute.suffix, &relative.suffix),
                    ] {
                        let rebased: Vec<_> = relative
                            .iter()
                            .map(|&[first, second, kind, reserved]| {
                                let second = if kind >= 4 { second + base } else { second };
                                [first + base, second, kind, reserved]
                            })
                            .collect();
                        assert_eq!(&rebased, absolute);
                        for work in relative {
                            assert!(max_accessed_index(work).is_some());
                        }
                    }
                }
            }
        }
    }
}
