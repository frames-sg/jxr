// SPDX-License-Identifier: MIT OR Apache-2.0

__device__ __forceinline__ bool jxr_inverse_rotate(
    int v[2], uint *status, uint code) {
    int temporary;
    return jxr_add(v[1], 1, temporary, status, code)
        && jxr_sub(v[0], temporary >> 1, v[0], status, code)
        && jxr_add(v[0], 1, temporary, status, code)
        && jxr_add(v[1], temporary >> 1, v[1], status, code);
}

__device__ __forceinline__ bool jxr_inverse_scale(
    int v[2], uint *status, uint code) {
    int temporary;
    return jxr_add(v[0], v[1], v[0], status, code)
        && jxr_sub(v[0] >> 1, v[1], v[1], status, code)
        && jxr_narrow((jxr_long)v[1] * 3ll, temporary, status, code)
        && jxr_add(v[0], temporary >> 3, v[0], status, code)
        && jxr_narrow((jxr_long)v[0] * 3ll, temporary, status, code)
        && jxr_add(v[1], temporary >> 4, v[1], status, code)
        && jxr_add(v[1], v[0] >> 7, v[1], status, code)
        && jxr_sub(v[1], v[0] >> 10, v[1], status, code);
}

__device__ __forceinline__ bool jxr_inverse_hadamard_post(
    int v[4], uint *status, uint code) {
    int temporary;
    return jxr_sub(v[1], v[2], v[1], status, code)
        && jxr_mul3_round(v[3], 4, 3, temporary, status, code)
        && jxr_add(v[0], temporary, v[0], status, code)
        && jxr_sub(v[3], v[1] >> 1, v[3], status, code)
        && jxr_sub(v[0], v[1], temporary, status, code)
        && jxr_sub(temporary >> 1, v[2], v[2], status, code)
        && (temporary = v[2], v[2] = v[3], v[3] = temporary, true)
        && jxr_sub(v[0], v[3], v[0], status, code)
        && jxr_add(v[1], v[2], v[1], status, code);
}

__device__ __forceinline__ bool jxr_inverse_todd_odd_post(
    int v[4], uint *status, uint code) {
    int first;
    int second;
    int temporary;
    return jxr_add(v[3], v[0], v[3], status, code)
        && jxr_sub(v[2], v[1], v[2], status, code)
        && (first = v[3] >> 1, second = v[2] >> 1, true)
        && jxr_sub(v[0], first, v[0], status, code)
        && jxr_add(v[1], second, v[1], status, code)
        && jxr_mul3_round(v[1], 6, 3, temporary, status, code)
        && jxr_sub(v[0], temporary, v[0], status, code)
        && jxr_mul3_round(v[0], 2, 2, temporary, status, code)
        && jxr_add(v[1], temporary, v[1], status, code)
        && jxr_mul3_round(v[1], 4, 3, temporary, status, code)
        && jxr_sub(v[0], temporary, v[0], status, code)
        && jxr_sub(v[1], second, v[1], status, code)
        && jxr_add(v[0], first, v[0], status, code)
        && jxr_add(v[2], v[1], v[2], status, code)
        && jxr_sub(v[3], v[0], v[3], status, code);
}

__device__ __forceinline__ bool jxr_overlap4(
    int v[4], uint *status, uint code) {
    int temporary;
    int pair[2];
    return jxr_add(v[0], v[3], v[0], status, code)
        && jxr_add(v[1], v[2], v[1], status, code)
        && jxr_add(v[0], 1, temporary, status, code)
        && jxr_sub(v[3], temporary >> 1, v[3], status, code)
        && jxr_add(v[1], 1, temporary, status, code)
        && jxr_sub(v[2], temporary >> 1, v[2], status, code)
        && (pair[0] = v[0], pair[1] = v[3], true)
        && jxr_inverse_scale(pair, status, code)
        && (v[0] = pair[0], v[3] = pair[1], pair[0] = v[1], pair[1] = v[2], true)
        && jxr_inverse_scale(pair, status, code)
        && (v[1] = pair[0], v[2] = pair[1], true)
        && jxr_mul3_round(v[3], 4, 3, temporary, status, code)
        && jxr_add(v[0], temporary, v[0], status, code)
        && jxr_mul3_round(v[2], 4, 3, temporary, status, code)
        && jxr_add(v[1], temporary, v[1], status, code)
        && jxr_sub(v[3], v[0] >> 1, v[3], status, code)
        && jxr_sub(v[2], v[1] >> 1, v[2], status, code)
        && jxr_add(v[0], v[3], v[0], status, code)
        && jxr_add(v[1], v[2], v[1], status, code)
        && jxr_narrow(-(jxr_long)v[3], v[3], status, code)
        && jxr_narrow(-(jxr_long)v[2], v[2], status, code)
        && (pair[0] = v[2], pair[1] = v[3], true)
        && jxr_inverse_rotate(pair, status, code)
        && (v[2] = pair[0], v[3] = pair[1], true)
        && jxr_add(v[0], 1, temporary, status, code)
        && jxr_add(v[3], temporary >> 1, v[3], status, code)
        && jxr_add(v[1], 1, temporary, status, code)
        && jxr_add(v[2], temporary >> 1, v[2], status, code)
        && jxr_sub(v[0], v[3], v[0], status, code)
        && jxr_sub(v[1], v[2], v[1], status, code);
}

__device__ __forceinline__ bool jxr_overlap2(
    int v[2], uint *status, uint code) {
    int temporary;
    return jxr_add(v[0], 2, temporary, status, code)
        && jxr_add(v[1], temporary >> 2, v[1], status, code)
        && jxr_add(v[1], 1, temporary, status, code)
        && jxr_add(v[0], temporary >> 1, v[0], status, code)
        && jxr_add(v[0], v[1] >> 5, v[0], status, code)
        && jxr_add(v[0], v[1] >> 9, v[0], status, code)
        && jxr_add(v[0], v[1] >> 13, v[0], status, code)
        && jxr_add(v[0], 2, temporary, status, code)
        && jxr_add(v[1], temporary >> 2, v[1], status, code);
}

__device__ __forceinline__ bool jxr_overlap2x2(
    int v[4], uint *status, uint code) {
    int temporary;
    return jxr_add(v[0], v[3], v[0], status, code)
        && jxr_add(v[1], v[2], v[1], status, code)
        && jxr_add(v[0], 1, temporary, status, code)
        && jxr_sub(v[3], temporary >> 1, v[3], status, code)
        && jxr_add(v[1], 1, temporary, status, code)
        && jxr_sub(v[2], temporary >> 1, v[2], status, code)
        && jxr_add(v[0], 2, temporary, status, code)
        && jxr_add(v[1], temporary >> 2, v[1], status, code)
        && jxr_add(v[1], 1, temporary, status, code)
        && jxr_add(v[0], temporary >> 1, v[0], status, code)
        && jxr_add(v[0], v[1] >> 5, v[0], status, code)
        && jxr_add(v[0], v[1] >> 9, v[0], status, code)
        && jxr_add(v[0], v[1] >> 13, v[0], status, code)
        && jxr_add(v[0], 2, temporary, status, code)
        && jxr_add(v[1], temporary >> 2, v[1], status, code)
        && jxr_add(v[0], 1, temporary, status, code)
        && jxr_add(v[3], temporary >> 1, v[3], status, code)
        && jxr_add(v[1], 1, temporary, status, code)
        && jxr_add(v[2], temporary >> 1, v[2], status, code)
        && jxr_sub(v[0], v[3], v[0], status, code)
        && jxr_sub(v[1], v[2], v[1], status, code);
}

__device__ __forceinline__ bool jxr_group4x4(
    int values[16], JxrUint4 indices, uint kind, uint *status, uint code) {
    int group[4] = {
        values[indices.x], values[indices.y], values[indices.z], values[indices.w]};
    bool ok = kind == 0u ? jxr_t2x2h(group, 0, status, code)
        : (kind == 1u ? jxr_inverse_todd_odd_post(group, status, code)
                      : jxr_inverse_hadamard_post(group, status, code));
    if (!ok) return false;
    values[indices.x] = group[0];
    values[indices.y] = group[1];
    values[indices.z] = group[2];
    values[indices.w] = group[3];
    return true;
}

__device__ __forceinline__ bool jxr_overlap4x4(
    int values[16], uint *status, uint code) {
    const JxrUint4 groups[4] = {
        {0, 3, 12, 15}, {1, 2, 13, 14}, {4, 7, 8, 11}, {5, 6, 9, 10}};
    for (uint i = 0; i < 4; ++i) {
        if (!jxr_group4x4(values, groups[i], 0, status, code)) return false;
    }
    const uint rotations[8] = {13, 12, 9, 8, 7, 3, 6, 2};
    for (uint i = 0; i < 4; ++i) {
        int pair[2] = {values[rotations[i * 2]], values[rotations[i * 2 + 1]]};
        if (!jxr_inverse_rotate(pair, status, code)) return false;
        values[rotations[i * 2]] = pair[0];
        values[rotations[i * 2 + 1]] = pair[1];
    }
    if (!jxr_group4x4(values, jxr_uint4(10, 11, 14, 15), 1, status, code)) return false;
    const uint scales[8] = {0, 15, 1, 14, 4, 11, 5, 10};
    for (uint i = 0; i < 4; ++i) {
        int pair[2] = {values[scales[i * 2]], values[scales[i * 2 + 1]]};
        if (!jxr_inverse_scale(pair, status, code)) return false;
        values[scales[i * 2]] = pair[0];
        values[scales[i * 2 + 1]] = pair[1];
    }
    for (uint i = 0; i < 4; ++i) {
        if (!jxr_group4x4(values, groups[i], 2, status, code)) return false;
    }
    return true;
}

__device__ __forceinline__ void jxr_apply_overlap(
    int *samples, JxrOverlapWorkAbi work, uint *status) {
    if (work.kind == 0u) {
        int values[16];
        for (uint y = 0; y < 4; ++y) {
            for (uint x = 0; x < 4; ++x) {
                values[y * 4 + x] = samples[work.first + y * work.second + x];
            }
        }
        if (!jxr_overlap4x4(values, status, 2u)) return;
        for (uint y = 0; y < 4; ++y) {
            for (uint x = 0; x < 4; ++x) {
                samples[work.first + y * work.second + x] = values[y * 4 + x];
            }
        }
    } else if (work.kind == 1u) {
        int values[4] = {
            samples[work.first], samples[work.first + work.second],
            samples[work.first + work.second * 2u], samples[work.first + work.second * 3u]};
        if (!jxr_overlap4(values, status, 2u)) return;
        for (uint i = 0; i < 4; ++i) samples[work.first + work.second * i] = values[i];
    } else if (work.kind == 2u || work.kind == 3u) {
        int values[4] = {
            samples[work.first], samples[work.first + 1u],
            samples[work.first + work.second], samples[work.first + work.second + 1u]};
        bool ok = work.kind == 2u ? jxr_overlap4(values, status, 2u)
                                  : jxr_overlap2x2(values, status, 2u);
        if (!ok) return;
        samples[work.first] = values[0];
        samples[work.first + 1u] = values[1];
        samples[work.first + work.second] = values[2];
        samples[work.first + work.second + 1u] = values[3];
    } else if (work.kind == 4u) {
        int values[2] = {samples[work.first], samples[work.second]};
        if (!jxr_overlap2(values, status, 2u)) return;
        samples[work.first] = values[0];
        samples[work.second] = values[1];
    } else if (work.kind == 5u) {
        int result;
        if (jxr_sub(samples[work.first], samples[work.second], result, status, 2u)) {
            samples[work.first] = result;
        }
    } else if (work.kind == 6u) {
        int result;
        if (jxr_add(samples[work.first], samples[work.second], result, status, 2u)) {
            samples[work.first] = result;
        }
    }
}

extern "C" __global__ void jxr_first_overlap(
    int *samples, const JxrOverlapWorkAbi *work, uint *status, uint work_count) {
    const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    const bool failed = jxr_block_failed(status);
    if (gid < work_count && !failed) jxr_apply_overlap(samples, work[gid], status);
}

extern "C" __global__ void jxr_second_overlap(
    int *samples, const JxrOverlapWorkAbi *work, uint *status, uint work_count) {
    const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    const bool failed = jxr_block_failed(status);
    if (gid < work_count && !failed) jxr_apply_overlap(samples, work[gid], status);
}
