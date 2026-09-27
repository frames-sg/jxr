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

// Checked 32-bit arithmetic. Each helper returns the wrapped two's-complement
// result and ORs an exact overflow predicate into a sticky flag. Kernels test
// the flag once before storing, which keeps the transforms branch-free and
// avoids emulated 64-bit integer math on Apple GPUs. A result derived from an
// overflowed intermediate is never stored because the flag stays set.
inline int jxr_add(int a, int b, thread bool &overflow) {
    const int result = as_type<int>(as_type<uint>(a) + as_type<uint>(b));
    overflow |= ((a ^ result) & (b ^ result)) < 0;
    return result;
}

inline int jxr_sub(int a, int b, thread bool &overflow) {
    const int result = as_type<int>(as_type<uint>(a) - as_type<uint>(b));
    overflow |= ((a ^ b) & (a ^ result)) < 0;
    return result;
}

inline int jxr_neg(int value, thread bool &overflow) {
    overflow |= value == INT_MIN;
    return as_type<int>(0u - as_type<uint>(value));
}

inline int jxr_mul(int value, uint factor, thread bool &overflow) {
    const bool negative = value < 0;
    const uint magnitude = negative ? 0u - as_type<uint>(value) : as_type<uint>(value);
    const uint product = magnitude * factor;
    overflow |= mulhi(magnitude, factor) != 0u || product > (negative ? 0x80000000u : 0x7fffffffu);
    return as_type<int>(negative ? 0u - product : product);
}

inline int jxr_mul3(int value, thread bool &overflow) {
    overflow |= value > INT_MAX / 3 || value < INT_MIN / 3;
    return as_type<int>(as_type<uint>(value) * 3u);
}

inline int jxr_mul3_round(int value, int rounding, uint shift, thread bool &overflow) {
    return jxr_add(jxr_mul3(value, overflow), rounding, overflow) >> shift;
}

inline void jxr_t2x2h(thread int v[4], int rounding, thread bool &overflow) {
    const int old2 = v[2];
    v[0] = jxr_add(v[0], v[3], overflow);
    v[1] = jxr_sub(v[1], v[2], overflow);
    const int midpoint = jxr_add(jxr_sub(v[0], v[1], overflow), rounding, overflow) >> 1;
    v[2] = jxr_sub(midpoint, v[3], overflow);
    v[3] = jxr_sub(midpoint, old2, overflow);
    v[0] = jxr_sub(v[0], v[3], overflow);
    v[1] = jxr_add(v[1], v[2], overflow);
}

inline void jxr_todd(thread int v[4], thread bool &overflow) {
    v[1] = jxr_add(v[1], v[3], overflow);
    v[0] = jxr_sub(v[0], v[2], overflow);
    v[3] = jxr_sub(v[3], v[1] >> 1, overflow);
    v[2] = jxr_add(v[2], jxr_add(v[0], 1, overflow) >> 1, overflow);
    v[0] = jxr_sub(v[0], jxr_mul3_round(v[1], 4, 3, overflow), overflow);
    v[1] = jxr_add(v[1], jxr_mul3_round(v[0], 4, 3, overflow), overflow);
    v[2] = jxr_sub(v[2], jxr_mul3_round(v[3], 4, 3, overflow), overflow);
    v[3] = jxr_add(v[3], jxr_mul3_round(v[2], 4, 3, overflow), overflow);
    v[2] = jxr_sub(v[2], jxr_add(v[1], 1, overflow) >> 1, overflow);
    v[3] = jxr_sub(jxr_add(v[0], 1, overflow) >> 1, v[3], overflow);
    v[1] = jxr_add(v[1], v[2], overflow);
    v[0] = jxr_sub(v[0], v[3], overflow);
}

inline void jxr_todd_odd(thread int v[4], thread bool &overflow) {
    v[3] = jxr_add(v[3], v[0], overflow);
    v[2] = jxr_sub(v[2], v[1], overflow);
    const int first = v[3] >> 1;
    const int second = v[2] >> 1;
    v[0] = jxr_sub(v[0], first, overflow);
    v[1] = jxr_add(v[1], second, overflow);
    v[0] = jxr_sub(v[0], jxr_mul3_round(v[1], 3, 3, overflow), overflow);
    v[1] = jxr_add(v[1], jxr_mul3_round(v[0], 3, 2, overflow), overflow);
    v[0] = jxr_sub(v[0], jxr_mul3_round(v[1], 4, 3, overflow), overflow);
    v[1] = jxr_sub(v[1], second, overflow);
    v[0] = jxr_add(v[0], first, overflow);
    v[2] = jxr_add(v[2], v[1], overflow);
    v[3] = jxr_sub(v[3], v[0], overflow);
    v[1] = jxr_neg(v[1], overflow);
    v[2] = jxr_neg(v[2], overflow);
}

inline void jxr_transform_group(thread int c[16], uint4 indices, uint kind, int rounding,
                                thread bool &overflow) {
    int v[4] = { c[indices.x], c[indices.y], c[indices.z], c[indices.w] };
    if (kind == 0u) jxr_t2x2h(v, rounding, overflow);
    else if (kind == 1u) jxr_todd(v, overflow);
    else jxr_todd_odd(v, overflow);
    c[indices.x] = v[0]; c[indices.y] = v[1]; c[indices.z] = v[2]; c[indices.w] = v[3];
}

inline void jxr_inverse_transform(thread int c[16], thread bool &overflow) {
    int input[16];
    for (uint i = 0; i < 16; ++i) input[i] = c[i];
    for (uint i = 0; i < 16; ++i) c[JXR_INVERSE_PERMUTATION[i]] = input[i];
    jxr_transform_group(c, uint4(0,1,4,5), 0, 1, overflow);
    jxr_transform_group(c, uint4(2,3,6,7), 1, 0, overflow);
    jxr_transform_group(c, uint4(8,12,9,13), 1, 0, overflow);
    jxr_transform_group(c, uint4(10,11,14,15), 2, 0, overflow);
    jxr_transform_group(c, uint4(0,3,12,15), 0, 0, overflow);
    jxr_transform_group(c, uint4(5,6,9,10), 0, 0, overflow);
    jxr_transform_group(c, uint4(1,2,13,14), 0, 0, overflow);
    jxr_transform_group(c, uint4(4,7,8,11), 0, 0, overflow);
}
