// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use jxr_math::{overlap::*, transform::*};

pub fn first_transform(
    packed: *const i32,
    macroblocks: *const JxrMacroblockAbi,
    low_plane: *mut i32,
    plane: JxrPlaneAbi,
    gid: u32,
) -> Result<(), MathError> {
    let metadata = device_load(macroblocks, plane.macroblock_offset + gid);
    let count = plane.block_columns * plane.block_rows;
    let mut low = [0; 16];
    low[0] = math_try!(dequantize(
        device_load(packed, metadata.coefficient_offset),
        metadata.quantizer_dc,
    ));
    if metadata.bands != 0 {
        for block in 1..count {
            let source = metadata.coefficient_offset
                + if metadata.bands >= 2 {
                    block * 16
                } else {
                    block
                };
            low[block as usize] = math_try!(dequantize(
                device_load(packed, source),
                metadata.quantizer_low_pass
            ));
        }
    }
    if plane.block_columns == 4 {
        math_try!(inverse_core_transform(&mut low));
    } else if plane.block_rows == 2 {
        let mut values = [low[0], low[1], low[2], low[3]];
        math_try!(inverse_chroma_420(&mut values));
        low[..4].copy_from_slice(&values);
    } else {
        let mut values = [
            low[0], low[1], low[2], low[3], low[4], low[5], low[6], low[7],
        ];
        math_try!(inverse_chroma_422(&mut values));
        low[..8].copy_from_slice(&values);
    }
    if plane.scale_after_first_transform != 0 {
        for block in 0..count {
            low[block as usize] = math_try!(checked_mul(low[block as usize], 2));
        }
    }
    let base_x = (metadata.coded_x - plane.macroblock_origin_x) * plane.block_columns;
    let base_y = (metadata.coded_y - plane.macroblock_origin_y) * plane.block_rows;
    for row in 0..plane.block_rows {
        for column in 0..plane.block_columns {
            store(
                low_plane,
                plane.low_offset + (base_y + row) * plane.low_width + base_x + column,
                low[(row * plane.block_columns + column) as usize],
            );
        }
    }
    Ok(())
}

pub fn predict_high(
    high: *mut i32,
    metadata: JxrMacroblockAbi,
    plane: JxrPlaneAbi,
) -> Result<(), MathError> {
    let count = plane.block_columns * plane.block_rows;
    if metadata.bands >= 2 {
        if metadata.hp_prediction == 1 {
            for block in 1..count {
                if block % plane.block_columns != 0 {
                    for step in 1..4 {
                        let coefficient = step * 4;
                        let destination = block * 16 + coefficient;
                        store(
                            high,
                            destination,
                            math_try!(checked_add(
                                device_load(high, destination),
                                device_load(high, destination - 16),
                            )),
                        );
                    }
                }
            }
        } else if metadata.hp_prediction == 2 {
            for block in plane.block_columns..count {
                for coefficient in 1..4 {
                    let destination = block * 16 + coefficient;
                    let source = destination - plane.block_columns * 16;
                    store(
                        high,
                        destination,
                        math_try!(checked_add(
                            device_load(high, destination),
                            device_load(high, source)
                        )),
                    );
                }
            }
        }
    }
    Ok(())
}

pub fn second_transform(
    high: *const i32,
    low_plane: *const i32,
    samples: *mut i32,
    metadata: JxrMacroblockAbi,
    plane: JxrPlaneAbi,
    block: u32,
) -> Result<(), MathError> {
    let mut coefficients = [0; 16];
    for coefficient in 1..16 {
        coefficients[coefficient as usize] = math_try!(dequantize(
            device_load(high, block * 16 + coefficient),
            metadata.quantizer_high_pass,
        ));
    }
    let local_x = metadata.coded_x - plane.macroblock_origin_x;
    let local_y = metadata.coded_y - plane.macroblock_origin_y;
    let low_x = local_x * plane.block_columns + block % plane.block_columns;
    let low_y = local_y * plane.block_rows + block / plane.block_columns;
    coefficients[0] = device_load(
        low_plane,
        plane.low_offset + low_y * plane.low_width + low_x,
    );
    math_try!(inverse_core_transform(&mut coefficients));
    let output_x = low_x * 4;
    let output_y = low_y * 4;
    for row in 0..4 {
        for column in 0..4 {
            store(
                samples,
                plane.sample_offset + (output_y + row) * plane.sample_width + output_x + column,
                coefficients[(row * 4 + column) as usize],
            );
        }
    }
    Ok(())
}

pub fn overlap(samples: *mut i32, work: JxrOverlapWorkAbi) -> Result<(), MathError> {
    if work.kind == 0 {
        let mut values = [0; 16];
        for y in 0..4 {
            for x in 0..4 {
                values[(y * 4 + x) as usize] =
                    device_load(samples, work.first + y * work.second + x);
            }
        }
        math_try!(inverse_overlap_filter_4x4(&mut values));
        for y in 0..4 {
            for x in 0..4 {
                store(
                    samples,
                    work.first + y * work.second + x,
                    values[(y * 4 + x) as usize],
                );
            }
        }
    } else if work.kind <= 3 {
        let indices = if work.kind == 1 {
            [
                work.first,
                work.first + work.second,
                work.first + work.second * 2,
                work.first + work.second * 3,
            ]
        } else {
            [
                work.first,
                work.first + 1,
                work.first + work.second,
                work.first + work.second + 1,
            ]
        };
        let mut values = [
            device_load(samples, indices[0]),
            device_load(samples, indices[1]),
            device_load(samples, indices[2]),
            device_load(samples, indices[3]),
        ];
        if work.kind == 3 {
            math_try!(inverse_overlap_filter_2x2(&mut values));
        } else {
            math_try!(inverse_overlap_filter_4(&mut values));
        }
        for i in 0..4 {
            store(samples, indices[i], values[i]);
        }
    } else if work.kind == 4 {
        let mut values = [
            device_load(samples, work.first),
            device_load(samples, work.second),
        ];
        math_try!(inverse_overlap_filter_2(&mut values));
        store(samples, work.first, values[0]);
        store(samples, work.second, values[1]);
    } else if work.kind == 5 {
        store(
            samples,
            work.first,
            math_try!(checked_sub(
                device_load(samples, work.first),
                device_load(samples, work.second),
            )),
        );
    } else if work.kind == 6 {
        store(
            samples,
            work.first,
            math_try!(checked_add(
                device_load(samples, work.first),
                device_load(samples, work.second),
            )),
        );
    }
    Ok(())
}
