inline void jxr_highpass_second_transform_one(
    device const int *packed,
    device const JxrMacroblockAbi *macroblocks,
    device const int *low_plane,
    device int *samples,
    device atomic_uint *status,
    JxrPlaneAbi plane,
    threadgroup int *high,
    uint group_id,
    uint tid,
    uint threads) {
    if (group_id >= plane.macroblock_count || jxr_failed(status)) return;
    const uint metadata_index = plane.macroblock_offset + group_id;
    const JxrMacroblockAbi metadata = macroblocks[metadata_index];
    const uint block_count = plane.block_columns * plane.block_rows;
    const uint coefficient_count = block_count * 16u;
    for (uint index = tid; index < coefficient_count; index += threads) {
        if (metadata.bands < 2u || (index & 15u) == 0u) {
            high[index] = 0;
        } else {
            high[index] = packed[metadata.coefficient_offset + index];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u && metadata.bands >= 2u) {
        if (metadata.hp_prediction == 1u) {
            for (uint block = 1; block < block_count; ++block) {
                if ((block % plane.block_columns) != 0u) {
                    for (uint coefficient = 4; coefficient <= 12; coefficient += 4) {
                        const uint destination = block * 16u + coefficient;
                        int predicted;
                        if (!jxr_add(high[destination], high[destination - 16u],
                                     predicted, status, 3u)) break;
                        high[destination] = predicted;
                    }
                }
            }
        } else if (metadata.hp_prediction == 2u) {
            for (uint block = plane.block_columns; block < block_count; ++block) {
                    for (uint coefficient = 1; coefficient <= 3; ++coefficient) {
                        const uint destination = block * 16u + coefficient;
                        const uint source = destination - plane.block_columns * 16u;
                        int predicted;
                        if (!jxr_add(high[destination], high[source], predicted, status, 3u)) break;
                        high[destination] = predicted;
                    }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (jxr_failed(status)) return;
    for (uint block = tid; block < block_count; block += threads) {
        int coefficients[16];
        coefficients[0] = 0;
        for (uint coefficient = 1; coefficient < 16; ++coefficient) {
            if (!jxr_mul(high[block * 16u + coefficient], metadata.quantizer_high_pass,
                         coefficients[coefficient], status, 4u)) return;
        }
        const uint local_x = metadata.coded_x - plane.macroblock_origin_x;
        const uint local_y = metadata.coded_y - plane.macroblock_origin_y;
        const uint low_x = local_x * plane.block_columns + block % plane.block_columns;
        const uint low_y = local_y * plane.block_rows + block / plane.block_columns;
        coefficients[0] = low_plane[plane.low_offset + low_y * plane.low_width + low_x];
        if (!jxr_inverse_transform(coefficients, status, 4u)) return;
        const uint output_x = local_x * plane.block_columns * 4u + (block % plane.block_columns) * 4u;
        const uint output_y = local_y * plane.block_rows * 4u + (block / plane.block_columns) * 4u;
        for (uint row = 0; row < 4; ++row)
            for (uint column = 0; column < 4; ++column)
                samples[plane.sample_offset + (output_y + row) * plane.sample_width + output_x + column] =
                    coefficients[row * 4u + column];
    }
}

kernel void jxr_highpass_second_transform(
    device const int *packed [[buffer(0)]],
    device const JxrMacroblockAbi *macroblocks [[buffer(1)]],
    device const int *low_plane [[buffer(2)]],
    device int *samples [[buffer(3)]],
    device atomic_uint *status [[buffer(4)]],
    constant JxrPlaneAbi &plane [[buffer(5)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup int high[256];
    jxr_highpass_second_transform_one(
        packed, macroblocks, low_plane, samples, status, plane, high, group_id, tid, threads);
}

kernel void jxr_highpass_second_transform_batch(
    device const int *packed [[buffer(0)]],
    device const JxrMacroblockAbi *macroblocks [[buffer(1)]],
    device const int *low_plane [[buffer(2)]],
    device int *samples [[buffer(3)]],
    device atomic_uint *statuses [[buffer(4)]],
    device const JxrPlaneAbi *planes [[buffer(5)]],
    constant JxrBatchDispatchAbi &batch [[buffer(6)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    if (group_id.y >= batch.image_count || group_id.z >= batch.plane_count) return;
    threadgroup int high[256];
    const JxrPlaneAbi plane = planes[group_id.y * batch.plane_count + group_id.z];
    jxr_highpass_second_transform_one(
        packed, macroblocks, low_plane, samples, statuses + group_id.y,
        plane, high, group_id.x, tid.x, threads.x);
}
