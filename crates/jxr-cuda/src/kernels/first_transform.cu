// SPDX-License-Identifier: MIT OR Apache-2.0

__device__ __forceinline__ void jxr_dequantize_first_transform_one(
    const int *packed,
    const JxrMacroblockAbi *macroblocks,
    int *low_plane,
    uint *status,
    JxrPlaneAbi plane,
    uint gid) {
    const bool failed = jxr_block_failed(status);
    if (gid >= plane.macroblock_count || failed) return;
    const uint metadata_index = plane.macroblock_offset + gid;
    const JxrMacroblockAbi metadata = macroblocks[metadata_index];
    const uint block_count = plane.block_columns * plane.block_rows;
    int low[16];
    for (uint i = 0; i < 16; ++i) low[i] = 0;
    if (!jxr_mul(
            packed[metadata.coefficient_offset], metadata.quantizer_dc, low[0], status, 1u)) {
        return;
    }
    if (metadata.bands != 0u) {
        for (uint block = 1; block < block_count; ++block) {
            const uint source = metadata.bands >= 2u
                ? metadata.coefficient_offset + block * 16u
                : metadata.coefficient_offset + block;
            if (!jxr_mul(
                    packed[source], metadata.quantizer_low_pass, low[block], status, 1u)) {
                return;
            }
        }
    }
    if (plane.block_columns == 4u) {
        if (!jxr_inverse_transform(low, status, 1u)) return;
    } else if (plane.block_rows == 2u) {
        int values[4] = {low[0], low[1], low[2], low[3]};
        if (!jxr_t2x2h(values, 0, status, 1u)) return;
        low[0] = values[0];
        low[1] = values[2];
        low[2] = values[1];
        low[3] = values[3];
    } else {
        int pair0;
        int pair1;
        int temporary;
        if (!jxr_add(low[4], 1, temporary, status, 1u)
            || !jxr_sub(low[0], temporary >> 1, pair0, status, 1u)
            || !jxr_add(low[4], pair0, pair1, status, 1u)) {
            return;
        }
        low[0] = pair0;
        low[4] = pair1;
        int first[4] = {low[0], low[1], low[2], low[3]};
        int second[4] = {low[4], low[6], low[5], low[7]};
        if (!jxr_t2x2h(first, 0, status, 1u)
            || !jxr_t2x2h(second, 0, status, 1u)) {
            return;
        }
        low[0] = first[0];
        low[1] = first[2];
        low[2] = first[1];
        low[3] = first[3];
        low[4] = second[0];
        low[5] = second[1];
        low[6] = second[2];
        low[7] = second[3];
    }
    if (plane.scale_after_first_transform != 0u) {
        for (uint block = 0; block < block_count; ++block) {
            if (!jxr_narrow((jxr_long)low[block] * 2ll, low[block], status, 1u)) return;
        }
    }
    const uint local_x = metadata.coded_x - plane.macroblock_origin_x;
    const uint local_y = metadata.coded_y - plane.macroblock_origin_y;
    const uint base_x = local_x * plane.block_columns;
    const uint base_y = local_y * plane.block_rows;
    for (uint row = 0; row < plane.block_rows; ++row) {
        for (uint column = 0; column < plane.block_columns; ++column) {
            low_plane[
                plane.low_offset + (base_y + row) * plane.low_width + base_x + column] =
                low[row * plane.block_columns + column];
        }
    }
}

extern "C" __global__ void jxr_dequantize_first_transform(
    const int *packed,
    const JxrMacroblockAbi *macroblocks,
    int *low_plane,
    uint *status,
    JxrPlaneAbi plane) {
    const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    jxr_dequantize_first_transform_one(
        packed, macroblocks, low_plane, status, plane, gid);
}
