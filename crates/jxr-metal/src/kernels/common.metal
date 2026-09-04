#include <metal_stdlib>
using namespace metal;

inline bool jxr_failed(device atomic_uint *status) {
    return atomic_load_explicit(status, memory_order_relaxed) != 0u;
}

inline void jxr_fail(device atomic_uint *status, uint code) {
    uint expected = 0u;
    while (expected == 0u
           && !atomic_compare_exchange_weak_explicit(
               status, &expected, code, memory_order_relaxed, memory_order_relaxed)) {}
}

inline bool jxr_narrow(long value, thread int &result, device atomic_uint *status, uint code) {
    if (value < long(INT_MIN) || value > long(INT_MAX)) {
        jxr_fail(status, code);
        return false;
    }
    result = int(value);
    return true;
}

inline bool jxr_add(int a, int b, thread int &result, device atomic_uint *status, uint code) {
    return jxr_narrow(long(a) + long(b), result, status, code);
}

inline bool jxr_sub(int a, int b, thread int &result, device atomic_uint *status, uint code) {
    return jxr_narrow(long(a) - long(b), result, status, code);
}

inline bool jxr_mul(int a, uint b, thread int &result, device atomic_uint *status, uint code) {
    return jxr_narrow(long(a) * long(b), result, status, code);
}

inline bool jxr_mul3_round(int value, int rounding, uint shift, thread int &result,
                           device atomic_uint *status, uint code) {
    int product;
    int rounded;
    return jxr_narrow(long(value) * 3l, product, status, code)
        && jxr_add(product, rounding, rounded, status, code)
        && (result = rounded >> shift, true);
}

inline bool jxr_t2x2h(thread int v[4], int rounding, device atomic_uint *status, uint code) {
    int difference;
    int midpoint;
    int old2 = v[2];
    return jxr_add(v[0], v[3], v[0], status, code)
        && jxr_sub(v[1], v[2], v[1], status, code)
        && jxr_sub(v[0], v[1], difference, status, code)
        && jxr_add(difference, rounding, midpoint, status, code)
        && (midpoint >>= 1, true)
        && jxr_sub(midpoint, v[3], v[2], status, code)
        && jxr_sub(midpoint, old2, v[3], status, code)
        && jxr_sub(v[0], v[3], v[0], status, code)
        && jxr_add(v[1], v[2], v[1], status, code);
}

inline bool jxr_todd(thread int v[4], device atomic_uint *status, uint code) {
    int temporary;
    return jxr_add(v[1], v[3], v[1], status, code)
        && jxr_sub(v[0], v[2], v[0], status, code)
        && jxr_sub(v[3], v[1] >> 1, v[3], status, code)
        && jxr_add(v[0], 1, temporary, status, code)
        && jxr_add(v[2], temporary >> 1, v[2], status, code)
        && jxr_mul3_round(v[1], 4, 3, temporary, status, code)
        && jxr_sub(v[0], temporary, v[0], status, code)
        && jxr_mul3_round(v[0], 4, 3, temporary, status, code)
        && jxr_add(v[1], temporary, v[1], status, code)
        && jxr_mul3_round(v[3], 4, 3, temporary, status, code)
        && jxr_sub(v[2], temporary, v[2], status, code)
        && jxr_mul3_round(v[2], 4, 3, temporary, status, code)
        && jxr_add(v[3], temporary, v[3], status, code)
        && jxr_add(v[1], 1, temporary, status, code)
        && jxr_sub(v[2], temporary >> 1, v[2], status, code)
        && jxr_add(v[0], 1, temporary, status, code)
        && jxr_sub(temporary >> 1, v[3], v[3], status, code)
        && jxr_add(v[1], v[2], v[1], status, code)
        && jxr_sub(v[0], v[3], v[0], status, code);
}

inline bool jxr_todd_odd(thread int v[4], device atomic_uint *status, uint code) {
    int temporary;
    int first;
    int second;
    return jxr_add(v[3], v[0], v[3], status, code)
        && jxr_sub(v[2], v[1], v[2], status, code)
        && (first = v[3] >> 1, second = v[2] >> 1, true)
        && jxr_sub(v[0], first, v[0], status, code)
        && jxr_add(v[1], second, v[1], status, code)
        && jxr_mul3_round(v[1], 3, 3, temporary, status, code)
        && jxr_sub(v[0], temporary, v[0], status, code)
        && jxr_mul3_round(v[0], 3, 2, temporary, status, code)
        && jxr_add(v[1], temporary, v[1], status, code)
        && jxr_mul3_round(v[1], 4, 3, temporary, status, code)
        && jxr_sub(v[0], temporary, v[0], status, code)
        && jxr_sub(v[1], second, v[1], status, code)
        && jxr_add(v[0], first, v[0], status, code)
        && jxr_add(v[2], v[1], v[2], status, code)
        && jxr_sub(v[3], v[0], v[3], status, code)
        && jxr_narrow(-long(v[1]), v[1], status, code)
        && jxr_narrow(-long(v[2]), v[2], status, code);
}

inline bool jxr_transform_group(thread int c[16], uint4 indices, uint kind, int rounding,
                                device atomic_uint *status, uint code) {
    int v[4] = { c[indices.x], c[indices.y], c[indices.z], c[indices.w] };
    bool ok = kind == 0u ? jxr_t2x2h(v, rounding, status, code)
        : (kind == 1u ? jxr_todd(v, status, code) : jxr_todd_odd(v, status, code));
    if (!ok) return false;
    c[indices.x] = v[0]; c[indices.y] = v[1]; c[indices.z] = v[2]; c[indices.w] = v[3];
    return true;
}

inline bool jxr_inverse_transform(thread int c[16], device atomic_uint *status, uint code) {
    int input[16];
    for (uint i = 0; i < 16; ++i) input[i] = c[i];
    for (uint i = 0; i < 16; ++i) c[JXR_INVERSE_PERMUTATION[i]] = input[i];
    return jxr_transform_group(c, uint4(0,1,4,5), 0, 1, status, code)
        && jxr_transform_group(c, uint4(2,3,6,7), 1, 0, status, code)
        && jxr_transform_group(c, uint4(8,12,9,13), 1, 0, status, code)
        && jxr_transform_group(c, uint4(10,11,14,15), 2, 0, status, code)
        && jxr_transform_group(c, uint4(0,3,12,15), 0, 0, status, code)
        && jxr_transform_group(c, uint4(5,6,9,10), 0, 0, status, code)
        && jxr_transform_group(c, uint4(1,2,13,14), 0, 0, status, code)
        && jxr_transform_group(c, uint4(4,7,8,11), 0, 0, status, code);
}
