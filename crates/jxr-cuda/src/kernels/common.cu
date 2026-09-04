// SPDX-License-Identifier: MIT OR Apache-2.0

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
typedef unsigned long long ulong;
typedef long long jxr_long;

#define JXR_INT_MIN (-2147483647 - 1)
#define JXR_INT_MAX 2147483647
#define JXR_UINT_MAX 0xffffffffu

struct JxrUint4 { uint x; uint y; uint z; uint w; };

__device__ __forceinline__ JxrUint4 jxr_uint4(uint x, uint y, uint z, uint w) {
    JxrUint4 result = {x, y, z, w};
    return result;
}

__device__ __forceinline__ int jxr_min_i32(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ int jxr_max_i32(int a, int b) { return a > b ? a : b; }
__device__ __forceinline__ uint jxr_min_u32(uint a, uint b) { return a < b ? a : b; }
__device__ __forceinline__ uint jxr_max_u32(uint a, uint b) { return a > b ? a : b; }
__device__ __forceinline__ int jxr_clamp_i32(int value, int low, int high) {
    return jxr_min_i32(jxr_max_i32(value, low), high);
}
__device__ __forceinline__ float jxr_clamp_f32(float value, float low, float high) {
    return value < low ? low : (value > high ? high : value);
}

__device__ __forceinline__ bool jxr_failed(uint *status) {
    return atomicAdd(status, 0u) != 0u;
}

__device__ __forceinline__ bool jxr_block_failed(uint *status) {
    __shared__ uint jxr_block_status;
    const uint linear_thread = threadIdx.x
        + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    if (linear_thread == 0u) jxr_block_status = (uint)jxr_failed(status);
    __syncthreads();
    return jxr_block_status != 0u;
}

__device__ __forceinline__ void jxr_fail(uint *status, uint code) {
    atomicCAS(status, 0u, code);
}

__device__ __forceinline__ bool jxr_narrow(
    jxr_long value, int &result, uint *status, uint code) {
    if (value < (jxr_long)JXR_INT_MIN || value > (jxr_long)JXR_INT_MAX) {
        jxr_fail(status, code);
        return false;
    }
    result = (int)value;
    return true;
}

__device__ __forceinline__ bool jxr_add(
    int a, int b, int &result, uint *status, uint code) {
    return jxr_narrow((jxr_long)a + (jxr_long)b, result, status, code);
}

__device__ __forceinline__ bool jxr_sub(
    int a, int b, int &result, uint *status, uint code) {
    return jxr_narrow((jxr_long)a - (jxr_long)b, result, status, code);
}

__device__ __forceinline__ bool jxr_mul(
    int a, uint b, int &result, uint *status, uint code) {
    return jxr_narrow((jxr_long)a * (jxr_long)b, result, status, code);
}

__device__ __forceinline__ bool jxr_mul3_round(
    int value, int rounding, uint shift, int &result, uint *status, uint code) {
    int product;
    int rounded;
    return jxr_narrow((jxr_long)value * 3ll, product, status, code)
        && jxr_add(product, rounding, rounded, status, code)
        && (result = rounded >> shift, true);
}

__device__ __forceinline__ bool jxr_t2x2h(
    int v[4], int rounding, uint *status, uint code) {
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

__device__ __forceinline__ bool jxr_todd(int v[4], uint *status, uint code) {
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

__device__ __forceinline__ bool jxr_todd_odd(int v[4], uint *status, uint code) {
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
        && jxr_narrow(-(jxr_long)v[1], v[1], status, code)
        && jxr_narrow(-(jxr_long)v[2], v[2], status, code);
}

__device__ __forceinline__ bool jxr_transform_group(
    int c[16], JxrUint4 indices, uint kind, int rounding, uint *status, uint code) {
    int v[4] = {c[indices.x], c[indices.y], c[indices.z], c[indices.w]};
    bool ok = kind == 0u ? jxr_t2x2h(v, rounding, status, code)
        : (kind == 1u ? jxr_todd(v, status, code) : jxr_todd_odd(v, status, code));
    if (!ok) return false;
    c[indices.x] = v[0];
    c[indices.y] = v[1];
    c[indices.z] = v[2];
    c[indices.w] = v[3];
    return true;
}

__device__ __forceinline__ bool jxr_inverse_transform(
    int c[16], uint *status, uint code) {
    int input[16];
    for (uint i = 0; i < 16; ++i) input[i] = c[i];
    for (uint i = 0; i < 16; ++i) c[JXR_INVERSE_PERMUTATION[i]] = input[i];
    return jxr_transform_group(c, jxr_uint4(0, 1, 4, 5), 0, 1, status, code)
        && jxr_transform_group(c, jxr_uint4(2, 3, 6, 7), 1, 0, status, code)
        && jxr_transform_group(c, jxr_uint4(8, 12, 9, 13), 1, 0, status, code)
        && jxr_transform_group(c, jxr_uint4(10, 11, 14, 15), 2, 0, status, code)
        && jxr_transform_group(c, jxr_uint4(0, 3, 12, 15), 0, 0, status, code)
        && jxr_transform_group(c, jxr_uint4(5, 6, 9, 10), 0, 0, status, code)
        && jxr_transform_group(c, jxr_uint4(1, 2, 13, 14), 0, 0, status, code)
        && jxr_transform_group(c, jxr_uint4(4, 7, 8, 11), 0, 0, status, code);
}
