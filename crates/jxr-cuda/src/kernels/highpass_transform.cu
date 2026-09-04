// SPDX-License-Identifier: MIT OR Apache-2.0

extern "C" __global__ void jxr_highpass_second_transform(
    const int *packed,
    const JxrMacroblockAbi *macroblocks,
    const int *low_plane,
    int *samples,
    uint *status,
    JxrPlaneAbi plane) {
    const uint group_id = blockIdx.x;
    const uint tid = threadIdx.x;
    const uint threads = blockDim.x;
    const bool failed = jxr_block_failed(status);
    if (group_id >= plane.macroblock_count || failed) return;
    const uint metadata_index = plane.macroblock_offset + group_id;
    const JxrMacroblockAbi metadata = macroblocks[metadata_index];
    const uint block_count = plane.block_columns * plane.block_rows;
    const uint coefficient_count = block_count * 16u;
    extern __shared__ int high[];
    for (uint index = tid; index < coefficient_count; index += threads) {
        if (metadata.bands < 2u || (index & 15u) == 0u) {
            high[index] = 0;
        } else {
            high[index] = packed[metadata.coefficient_offset + index];
        }
    }
    __syncthreads();
    if (tid == 0u && metadata.bands >= 2u) {
        if (metadata.hp_prediction == 1u) {
            for (uint block = 1; block < block_count; ++block) {
                if ((block % plane.block_columns) != 0u) {
                    for (uint coefficient = 4; coefficient <= 12; coefficient += 4) {
                        const uint destination = block * 16u + coefficient;
                        int predicted;
                        if (!jxr_add(
                                high[destination], high[destination - 16u],
                                predicted, status, 3u)) {
                            break;
                        }
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
    __syncthreads();
    if (jxr_block_failed(status)) return;
    for (uint block = tid; block < block_count; block += threads) {
        int coefficients[16];
        coefficients[0] = 0;
        for (uint coefficient = 1; coefficient < 16; ++coefficient) {
            if (!jxr_mul(
                    high[block * 16u + coefficient], metadata.quantizer_high_pass,
                    coefficients[coefficient], status, 4u)) {
                return;
            }
        }
        const uint local_x = metadata.coded_x - plane.macroblock_origin_x;
        const uint local_y = metadata.coded_y - plane.macroblock_origin_y;
        const uint low_x = local_x * plane.block_columns + block % plane.block_columns;
        const uint low_y = local_y * plane.block_rows + block / plane.block_columns;
        coefficients[0] =
            low_plane[plane.low_offset + low_y * plane.low_width + low_x];
        if (!jxr_inverse_transform(coefficients, status, 4u)) return;
        const uint output_x =
            local_x * plane.block_columns * 4u + (block % plane.block_columns) * 4u;
        const uint output_y =
            local_y * plane.block_rows * 4u + (block / plane.block_columns) * 4u;
        for (uint row = 0; row < 4; ++row) {
            for (uint column = 0; column < 4; ++column) {
                samples[
                    plane.sample_offset + (output_y + row) * plane.sample_width
                        + output_x + column] = coefficients[row * 4u + column];
            }
        }
    }
}
