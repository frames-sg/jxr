// Accumulates one HP prediction chain in normative order: each block adds its
// raw coefficient to the already-predicted value of its predecessor, so every
// partial sum (and its overflow check) matches a serial macroblock traversal.
inline int jxr_predicted_hp(device const int *coefficients, uint first_block, uint last_block,
                            uint block_step, uint coefficient, thread bool &overflow) {
    int value = coefficients[first_block * 16u + coefficient];
    for (uint block = first_block + block_step; block <= last_block; block += block_step)
        value = jxr_add(coefficients[block * 16u + coefficient], value, overflow);
    return value;
}

// One thread reconstructs one 4x4 block. Consecutive threads cover the blocks
// of consecutive macroblocks, so SIMD groups stay full for every chroma
// layout and no threadgroup memory or barrier is needed.
inline void jxr_highpass_second_transform_one(
    device const int *packed,
    device const JxrMacroblockAbi *macroblocks,
    device const int *low_plane,
    device int *samples,
    device atomic_uint *status,
    JxrPlaneAbi plane,
    uint thread_index) {
    const uint block_count = plane.block_columns * plane.block_rows;
    const uint macroblock = thread_index / block_count;
    const uint block = thread_index - macroblock * block_count;
    if (macroblock >= plane.macroblock_count || jxr_failed(status)) return;
    const JxrMacroblockAbi metadata = macroblocks[plane.macroblock_offset + macroblock];
    const uint column = block % plane.block_columns;
    const uint row = block / plane.block_columns;
    int coefficients[16];
    for (uint coefficient = 0; coefficient < 16; ++coefficient) coefficients[coefficient] = 0;
    if (metadata.bands >= 2u) {
        device const int *high = packed + metadata.coefficient_offset;
        for (uint coefficient = 1; coefficient < 16; ++coefficient)
            coefficients[coefficient] = high[block * 16u + coefficient];
        bool prediction_overflow = false;
        if (metadata.hp_prediction == 1u && column != 0u) {
            for (uint coefficient = 4; coefficient <= 12; coefficient += 4)
                coefficients[coefficient] = jxr_predicted_hp(
                    high, block - column, block, 1u, coefficient, prediction_overflow);
        } else if (metadata.hp_prediction == 2u && row != 0u) {
            for (uint coefficient = 1; coefficient <= 3; ++coefficient)
                coefficients[coefficient] = jxr_predicted_hp(
                    high, column, block, plane.block_columns, coefficient, prediction_overflow);
        }
        if (prediction_overflow) {
            jxr_fail(status, 3u);
            return;
        }
    }
    bool overflow = false;
    for (uint coefficient = 1; coefficient < 16; ++coefficient)
        coefficients[coefficient] =
            jxr_mul(coefficients[coefficient], metadata.quantizer_high_pass, overflow);
    const uint local_x = metadata.coded_x - plane.macroblock_origin_x;
    const uint local_y = metadata.coded_y - plane.macroblock_origin_y;
    const uint low_x = local_x * plane.block_columns + column;
    const uint low_y = local_y * plane.block_rows + row;
    coefficients[0] = low_plane[plane.low_offset + low_y * plane.low_width + low_x];
    jxr_inverse_transform(coefficients, overflow);
    if (overflow) {
        jxr_fail(status, 4u);
        return;
    }
    const uint output_x = low_x * 4u;
    const uint output_y = low_y * 4u;
    for (uint y = 0; y < 4; ++y) {
        device int4 *destination = reinterpret_cast<device int4 *>(
            samples + plane.sample_offset + (output_y + y) * plane.sample_width + output_x);
        *destination = int4(coefficients[y * 4u], coefficients[y * 4u + 1u],
                            coefficients[y * 4u + 2u], coefficients[y * 4u + 3u]);
    }
}

kernel void jxr_highpass_second_transform(
    device const int *packed [[buffer(0)]],
    device const JxrMacroblockAbi *macroblocks [[buffer(1)]],
    device const int *low_plane [[buffer(2)]],
    device int *samples [[buffer(3)]],
    device atomic_uint *status [[buffer(4)]],
    constant JxrPlaneAbi &plane [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    jxr_highpass_second_transform_one(
        packed, macroblocks, low_plane, samples, status, plane, gid);
}

kernel void jxr_highpass_second_transform_batch(
    device const int *packed [[buffer(0)]],
    device const JxrMacroblockAbi *macroblocks [[buffer(1)]],
    device const int *low_plane [[buffer(2)]],
    device int *samples [[buffer(3)]],
    device atomic_uint *statuses [[buffer(4)]],
    device const JxrPlaneAbi *planes [[buffer(5)]],
    constant JxrBatchDispatchAbi &batch [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]) {
    if (gid.y >= batch.image_count || gid.z >= batch.plane_count) return;
    const JxrPlaneAbi plane = planes[gid.y * batch.plane_count + gid.z];
    jxr_highpass_second_transform_one(
        packed, macroblocks, low_plane, samples, statuses + gid.y, plane, gid.x);
}
