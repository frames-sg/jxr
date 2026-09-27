inline void jxr_inverse_rotate(thread int v[2], thread bool &overflow) {
    v[0] = jxr_sub(v[0], jxr_add(v[1], 1, overflow) >> 1, overflow);
    v[1] = jxr_add(v[1], jxr_add(v[0], 1, overflow) >> 1, overflow);
}

inline void jxr_inverse_scale(thread int v[2], thread bool &overflow) {
    v[0] = jxr_add(v[0], v[1], overflow);
    v[1] = jxr_sub(v[0] >> 1, v[1], overflow);
    v[0] = jxr_add(v[0], jxr_mul3(v[1], overflow) >> 3, overflow);
    v[1] = jxr_add(v[1], jxr_mul3(v[0], overflow) >> 4, overflow);
    v[1] = jxr_add(v[1], v[0] >> 7, overflow);
    v[1] = jxr_sub(v[1], v[0] >> 10, overflow);
}

inline void jxr_inverse_hadamard_post(thread int v[4], thread bool &overflow) {
    v[1] = jxr_sub(v[1], v[2], overflow);
    v[0] = jxr_add(v[0], jxr_mul3_round(v[3], 4, 3, overflow), overflow);
    v[3] = jxr_sub(v[3], v[1] >> 1, overflow);
    v[2] = jxr_sub(jxr_sub(v[0], v[1], overflow) >> 1, v[2], overflow);
    const int swapped = v[2];
    v[2] = v[3];
    v[3] = swapped;
    v[0] = jxr_sub(v[0], v[3], overflow);
    v[1] = jxr_add(v[1], v[2], overflow);
}

inline void jxr_inverse_todd_odd_post(thread int v[4], thread bool &overflow) {
    v[3] = jxr_add(v[3], v[0], overflow);
    v[2] = jxr_sub(v[2], v[1], overflow);
    const int first = v[3] >> 1;
    const int second = v[2] >> 1;
    v[0] = jxr_sub(v[0], first, overflow);
    v[1] = jxr_add(v[1], second, overflow);
    v[0] = jxr_sub(v[0], jxr_mul3_round(v[1], 6, 3, overflow), overflow);
    v[1] = jxr_add(v[1], jxr_mul3_round(v[0], 2, 2, overflow), overflow);
    v[0] = jxr_sub(v[0], jxr_mul3_round(v[1], 4, 3, overflow), overflow);
    v[1] = jxr_sub(v[1], second, overflow);
    v[0] = jxr_add(v[0], first, overflow);
    v[2] = jxr_add(v[2], v[1], overflow);
    v[3] = jxr_sub(v[3], v[0], overflow);
}

inline void jxr_overlap4(thread int v[4], thread bool &overflow) {
    v[0] = jxr_add(v[0], v[3], overflow);
    v[1] = jxr_add(v[1], v[2], overflow);
    v[3] = jxr_sub(v[3], jxr_add(v[0], 1, overflow) >> 1, overflow);
    v[2] = jxr_sub(v[2], jxr_add(v[1], 1, overflow) >> 1, overflow);
    int pair[2] = { v[0], v[3] };
    jxr_inverse_scale(pair, overflow);
    v[0] = pair[0]; v[3] = pair[1];
    pair[0] = v[1]; pair[1] = v[2];
    jxr_inverse_scale(pair, overflow);
    v[1] = pair[0]; v[2] = pair[1];
    v[0] = jxr_add(v[0], jxr_mul3_round(v[3], 4, 3, overflow), overflow);
    v[1] = jxr_add(v[1], jxr_mul3_round(v[2], 4, 3, overflow), overflow);
    v[3] = jxr_sub(v[3], v[0] >> 1, overflow);
    v[2] = jxr_sub(v[2], v[1] >> 1, overflow);
    v[0] = jxr_add(v[0], v[3], overflow);
    v[1] = jxr_add(v[1], v[2], overflow);
    v[3] = jxr_neg(v[3], overflow);
    v[2] = jxr_neg(v[2], overflow);
    pair[0] = v[2]; pair[1] = v[3];
    jxr_inverse_rotate(pair, overflow);
    v[2] = pair[0]; v[3] = pair[1];
    v[3] = jxr_add(v[3], jxr_add(v[0], 1, overflow) >> 1, overflow);
    v[2] = jxr_add(v[2], jxr_add(v[1], 1, overflow) >> 1, overflow);
    v[0] = jxr_sub(v[0], v[3], overflow);
    v[1] = jxr_sub(v[1], v[2], overflow);
}

inline void jxr_overlap2(thread int v[2], thread bool &overflow) {
    v[1] = jxr_add(v[1], jxr_add(v[0], 2, overflow) >> 2, overflow);
    v[0] = jxr_add(v[0], jxr_add(v[1], 1, overflow) >> 1, overflow);
    v[0] = jxr_add(v[0], v[1] >> 5, overflow);
    v[0] = jxr_add(v[0], v[1] >> 9, overflow);
    v[0] = jxr_add(v[0], v[1] >> 13, overflow);
    v[1] = jxr_add(v[1], jxr_add(v[0], 2, overflow) >> 2, overflow);
}

inline void jxr_overlap2x2(thread int v[4], thread bool &overflow) {
    v[0] = jxr_add(v[0], v[3], overflow);
    v[1] = jxr_add(v[1], v[2], overflow);
    v[3] = jxr_sub(v[3], jxr_add(v[0], 1, overflow) >> 1, overflow);
    v[2] = jxr_sub(v[2], jxr_add(v[1], 1, overflow) >> 1, overflow);
    v[1] = jxr_add(v[1], jxr_add(v[0], 2, overflow) >> 2, overflow);
    v[0] = jxr_add(v[0], jxr_add(v[1], 1, overflow) >> 1, overflow);
    v[0] = jxr_add(v[0], v[1] >> 5, overflow);
    v[0] = jxr_add(v[0], v[1] >> 9, overflow);
    v[0] = jxr_add(v[0], v[1] >> 13, overflow);
    v[1] = jxr_add(v[1], jxr_add(v[0], 2, overflow) >> 2, overflow);
    v[3] = jxr_add(v[3], jxr_add(v[0], 1, overflow) >> 1, overflow);
    v[2] = jxr_add(v[2], jxr_add(v[1], 1, overflow) >> 1, overflow);
    v[0] = jxr_sub(v[0], v[3], overflow);
    v[1] = jxr_sub(v[1], v[2], overflow);
}

inline void jxr_group4x4(thread int values[16], uint4 indices, uint kind, thread bool &overflow) {
    int group[4] = { values[indices.x], values[indices.y], values[indices.z], values[indices.w] };
    if (kind == 0u) jxr_t2x2h(group, 0, overflow);
    else if (kind == 1u) jxr_inverse_todd_odd_post(group, overflow);
    else jxr_inverse_hadamard_post(group, overflow);
    values[indices.x] = group[0]; values[indices.y] = group[1];
    values[indices.z] = group[2]; values[indices.w] = group[3];
}

inline void jxr_overlap4x4(thread int values[16], thread bool &overflow) {
    const uint4 groups[4] = { uint4(0,3,12,15), uint4(1,2,13,14),
                              uint4(4,7,8,11), uint4(5,6,9,10) };
    for (uint i = 0; i < 4; ++i) jxr_group4x4(values, groups[i], 0, overflow);
    const uint2 rotations[4] = { uint2(13,12), uint2(9,8), uint2(7,3), uint2(6,2) };
    for (uint i = 0; i < 4; ++i) {
        int pair[2] = { values[rotations[i].x], values[rotations[i].y] };
        jxr_inverse_rotate(pair, overflow);
        values[rotations[i].x] = pair[0]; values[rotations[i].y] = pair[1];
    }
    jxr_group4x4(values, uint4(10,11,14,15), 1, overflow);
    const uint2 scales[4] = { uint2(0,15), uint2(1,14), uint2(4,11), uint2(5,10) };
    for (uint i = 0; i < 4; ++i) {
        int pair[2] = { values[scales[i].x], values[scales[i].y] };
        jxr_inverse_scale(pair, overflow);
        values[scales[i].x] = pair[0]; values[scales[i].y] = pair[1];
    }
    for (uint i = 0; i < 4; ++i) jxr_group4x4(values, groups[i], 2, overflow);
}

inline void jxr_apply_overlap(device int *samples, JxrOverlapWorkAbi work,
                              device atomic_uint *status) {
    bool overflow = false;
    if (work.kind == 0u) {
        int values[16];
        for (uint y = 0; y < 4; ++y)
            for (uint x = 0; x < 4; ++x) values[y * 4 + x] = samples[work.first + y * work.second + x];
        jxr_overlap4x4(values, overflow);
        if (overflow) { jxr_fail(status, 2u); return; }
        for (uint y = 0; y < 4; ++y)
            for (uint x = 0; x < 4; ++x) samples[work.first + y * work.second + x] = values[y * 4 + x];
    } else if (work.kind == 1u) {
        int values[4] = { samples[work.first], samples[work.first + work.second],
                          samples[work.first + work.second * 2u], samples[work.first + work.second * 3u] };
        jxr_overlap4(values, overflow);
        if (overflow) { jxr_fail(status, 2u); return; }
        for (uint i = 0; i < 4; ++i) samples[work.first + work.second * i] = values[i];
    } else if (work.kind == 2u || work.kind == 3u) {
        int values[4] = { samples[work.first], samples[work.first + 1u],
                          samples[work.first + work.second], samples[work.first + work.second + 1u] };
        if (work.kind == 2u) jxr_overlap4(values, overflow);
        else jxr_overlap2x2(values, overflow);
        if (overflow) { jxr_fail(status, 2u); return; }
        samples[work.first] = values[0]; samples[work.first + 1u] = values[1];
        samples[work.first + work.second] = values[2]; samples[work.first + work.second + 1u] = values[3];
    } else if (work.kind == 4u) {
        int values[2] = { samples[work.first], samples[work.second] };
        jxr_overlap2(values, overflow);
        if (overflow) { jxr_fail(status, 2u); return; }
        samples[work.first] = values[0]; samples[work.second] = values[1];
    } else if (work.kind == 5u) {
        const int result = jxr_sub(samples[work.first], samples[work.second], overflow);
        if (overflow) { jxr_fail(status, 2u); return; }
        samples[work.first] = result;
    } else if (work.kind == 6u) {
        const int result = jxr_add(samples[work.first], samples[work.second], overflow);
        if (overflow) { jxr_fail(status, 2u); return; }
        samples[work.first] = result;
    }
}

// Work items are relative to one plane; `base` selects that plane's first
// sample so cached schedules serve every image with the same geometry.
kernel void jxr_first_overlap(
    device int *samples [[buffer(0)]],
    device const JxrOverlapWorkAbi *work [[buffer(1)]],
    device atomic_uint *status [[buffer(2)]],
    constant uint &work_count [[buffer(3)]],
    constant uint &base [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < work_count && !jxr_failed(status)) jxr_apply_overlap(samples + base, work[gid], status);
}

kernel void jxr_second_overlap(
    device int *samples [[buffer(0)]],
    device const JxrOverlapWorkAbi *work [[buffer(1)]],
    device atomic_uint *status [[buffer(2)]],
    constant uint &work_count [[buffer(3)]],
    constant uint &base [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < work_count && !jxr_failed(status)) jxr_apply_overlap(samples + base, work[gid], status);
}
