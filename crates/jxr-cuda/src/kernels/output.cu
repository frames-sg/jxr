// SPDX-License-Identifier: MIT OR Apache-2.0

struct JxrInt4 { int x; int y; int z; int w; };

__device__ __forceinline__ int jxr_read_plane(
    const int *samples, JxrSamplePlaneAbi plane, uint x, uint y) {
    return samples[plane.sample_offset + (y - plane.origin_y) * plane.width + x - plane.origin_x];
}

__device__ __forceinline__ JxrInt4 jxr_centering(uint code) {
    switch (code) {
        case 0u: return {4, 4, 0, 8};
        case 1u: return {5, 3, 1, 7};
        case 2u: return {6, 2, 2, 6};
        case 3u: return {7, 1, 3, 5};
        default: return {8, 0, 4, 4};
    }
}

__device__ __forceinline__ bool jxr_weighted(
    int first, int first_weight, int second, int second_weight,
    int &result, uint *status) {
    jxr_long value = (jxr_long)first * first_weight
        + (jxr_long)second * second_weight + 4ll;
    return jxr_narrow(value >> 3, result, status, 16u);
}

__device__ __forceinline__ bool jxr_upsample_pair(
    int previous, int current, int next, uint centering, int pair[2], uint *status) {
    JxrInt4 weights = jxr_centering(centering);
    return jxr_weighted(previous, weights.z, current, weights.w, pair[0], status)
        && jxr_weighted(current, weights.x, next, weights.y, pair[1], status);
}

__device__ __forceinline__ int jxr_clamped_plane(
    const int *samples, JxrSamplePlaneAbi plane, int x, int y) {
    int local_x = jxr_clamp_i32(x - (int)plane.origin_x, 0, (int)plane.width - 1);
    int local_y = jxr_clamp_i32(y - (int)plane.origin_y, 0, (int)plane.height - 1);
    return samples[plane.sample_offset + (uint)local_y * plane.width + (uint)local_x];
}

__device__ __forceinline__ bool jxr_chroma_sample(
    const int *samples, JxrSamplePlaneAbi plane, uint full_x, uint full_y,
    const JxrOutputAbi &params, int &result, uint *status) {
    if (params.chroma_sampling == 3u) {
        result = jxr_read_plane(samples, plane, full_x, full_y);
        return true;
    }
    int chroma_x = (int)(full_x >> 1u);
    if (params.chroma_sampling == 2u) {
        int pair[2];
        if (!jxr_upsample_pair(
                jxr_clamped_plane(samples, plane, chroma_x - 1, (int)full_y),
                jxr_clamped_plane(samples, plane, chroma_x, (int)full_y),
                jxr_clamped_plane(samples, plane, chroma_x + 1, (int)full_y),
                params.chroma_centering_x, pair, status)) {
            return false;
        }
        result = pair[full_x & 1u];
        return true;
    }
    int chroma_y = (int)(full_y >> 1u);
    int vertical[3];
    for (int column = -1; column <= 1; ++column) {
        int pair[2];
        if (!jxr_upsample_pair(
                jxr_clamped_plane(samples, plane, chroma_x + column, chroma_y - 1),
                jxr_clamped_plane(samples, plane, chroma_x + column, chroma_y),
                jxr_clamped_plane(samples, plane, chroma_x + column, chroma_y + 1),
                params.chroma_centering_y, pair, status)) {
            return false;
        }
        vertical[column + 1] = pair[full_y & 1u];
    }
    int horizontal[2];
    if (!jxr_upsample_pair(
            vertical[0], vertical[1], vertical[2], params.chroma_centering_x,
            horizontal, status)) {
        return false;
    }
    result = horizontal[full_x & 1u];
    return true;
}

__device__ __forceinline__ bool jxr_primary_values(
    const int *samples, const JxrSamplePlaneAbi *planes, uint x, uint y,
    const JxrOutputAbi &params, int values[4], uint *status) {
    for (uint i = 0; i < 4; ++i) values[i] = 0;
    values[0] = jxr_read_plane(samples, planes[0], x, y);
    if (params.component_count == 1u) return true;
    if (params.internal_color == 1u) {
        if (!jxr_chroma_sample(samples, planes[1], x, y, params, values[1], status)
            || !jxr_chroma_sample(samples, planes[2], x, y, params, values[2], status)) {
            return false;
        }
    } else {
        for (uint i = 1; i < jxr_min_u32(params.component_count, 4u); ++i) {
            values[i] = jxr_read_plane(samples, planes[i], x, y);
        }
    }
    return true;
}

__device__ __forceinline__ bool jxr_converted(
    const int *samples, const JxrSamplePlaneAbi *planes, uint x, uint y,
    const JxrOutputAbi &params, int values[4], uint *status) {
    int input[4];
    if (!jxr_primary_values(samples, planes, x, y, params, input, status)) return false;
    for (uint i = 0; i < 4; ++i) values[i] = input[i];
    if (params.internal_color == 0u && params.output_color == 2u) {
        values[1] = input[0];
        values[2] = input[0];
    } else if (params.internal_color == 1u
               && (params.output_color == 2u || params.output_color == 6u)) {
        int temporary;
        int green;
        int red;
        if (!jxr_sub(0, input[1], temporary, status, 16u)
            || !jxr_sub(input[0], temporary >> 1, green, status, 16u)
            || !jxr_add(temporary, green, red, status, 16u)
            || !jxr_sub(red, (input[2] >> 1) + (input[2] & 1), red, status, 16u)
            || !jxr_add(input[2], red, values[2], status, 16u)) {
            return false;
        }
        values[0] = red;
        values[1] = green;
        if (params.bit_depth >= 8u && params.bit_depth <= 10u
            && params.red_blue_not_swapped == 0u) {
            int swap = values[0];
            values[0] = values[2];
            values[2] = swap;
        }
    } else if (params.internal_color == 5u && params.output_color == 3u) {
        int black;
        int magenta;
        int cyan;
        if (!jxr_add(input[3], input[0] >> 1, black, status, 16u)
            || !jxr_sub(black, input[0], magenta, status, 16u)
            || !jxr_sub(magenta, input[1] >> 1, magenta, status, 16u)
            || !jxr_add(input[1], magenta, cyan, status, 16u)
            || !jxr_add(cyan, input[2] >> 1, cyan, status, 16u)
            || !jxr_sub(cyan, input[2], values[2], status, 16u)) {
            return false;
        }
        values[0] = cyan;
        values[1] = magenta;
        values[3] = black;
    } else if (params.internal_color == 5u && params.output_color == 4u) {
        values[0] = input[1];
        values[1] = input[2];
        values[2] = input[3];
        values[3] = input[0];
    }
    return true;
}

__device__ __forceinline__ int jxr_base_bias(uint depth, uint shift_bits) {
    switch (depth) {
        case 1u: return 128;
        case 2u:
        case 9u: return 512;
        case 3u: return 32768 >> shift_bits;
        case 8u: return 16;
        case 10u: return 32;
        default: return 0;
    }
}

__device__ __forceinline__ bool jxr_scale(
    int sample, uint component, bool alpha, const JxrOutputAbi &params,
    int &result, uint *status) {
    uint depth = params.bit_depth;
    uint scaled = alpha ? params.alpha_scaled : params.scaled;
    uint shift_bits = alpha ? params.alpha_shift_bits : params.shift_bits;
    int bias = jxr_base_bias(depth, shift_bits);
    if (!alpha && params.output_color == 3u) {
        bias = component < 3u ? bias >> 1 : -(bias >> 1);
    }
    if (!alpha && params.output_color == 6u) bias = 0;
    uint scale = scaled * 3u;
    int shifted_bias;
    if (!jxr_narrow((jxr_long)bias * (1ll << scale), shifted_bias, status, 16u)) return false;
    int biased;
    if (!jxr_add(sample, shifted_bias, biased, status, 16u)) return false;
    int rounding = scaled == 0u ? 0 : ((depth == 0u || depth == 3u) ? 4 : 3);
    int rounded;
    if (!jxr_add(biased, rounding, rounded, status, 16u)) return false;
    uint downshift = scale + ((depth == 10u && component != 1u) ? 1u : 0u);
    int downscaled = rounded >> downshift;
    if (depth == 3u || depth == 4u || depth == 6u) {
        return jxr_narrow(
            (jxr_long)downscaled * (1ll << shift_bits), result, status, 16u);
    }
    result = downscaled;
    return true;
}

__device__ __forceinline__ ulong jxr_abs_long(int value) {
    return value < 0 ? (ulong)(-(jxr_long)value) : (ulong)value;
}

__device__ __forceinline__ uint jxr_f16_bits(
    int sample, bool alpha, const JxrOutputAbi &params, uint *status) {
    int scaled;
    if (!jxr_scale(sample, 0u, alpha, params, scaled, status)) return 0u;
    uint sign = scaled < 0 ? 0x8000u : 0u;
    uint magnitude = (uint)jxr_min_u32((uint)jxr_abs_long(scaled), 32767u);
    return sign | magnitude;
}

__device__ __forceinline__ uint jxr_f32_bits(
    int sample, bool alpha, const JxrOutputAbi &params, uint *status) {
    int scaled;
    if (!jxr_scale(sample, 0u, alpha, params, scaled, status)) return 0u;
    uint length = alpha ? params.alpha_mantissa_length : params.mantissa_length;
    int exponent_bias = (int)(alpha ? params.alpha_exponent_bias_bits : params.exponent_bias_bits);
    uint sign = scaled < 0 ? 0x80000000u : 0u;
    ulong magnitude = jxr_abs_long(scaled);
    ulong implicit = 1ull << length;
    jxr_long exponent = (jxr_long)(magnitude >> length);
    ulong mantissa = (magnitude & (implicit - 1ull)) | implicit;
    if (exponent == 0ll) {
        mantissa ^= implicit;
        exponent = 1ll;
    }
    exponent = exponent - (jxr_long)exponent_bias + 127ll;
    while (mantissa < implicit && exponent > 1ll && mantissa > 0ull) {
        --exponent;
        mantissa <<= 1u;
    }
    if (mantissa < implicit) {
        exponent = 0ll;
    } else {
        mantissa ^= implicit;
    }
    mantissa <<= 23u - length;
    if (exponent < 0ll || exponent > 255ll || mantissa > 0x7fffffull) {
        jxr_fail(status, 16u);
        return 0u;
    }
    return sign | ((uint)exponent << 23u) | (uint)mantissa;
}

__device__ __forceinline__ uint jxr_unsigned_premultiply(
    uint value, uint alpha, uint maximum) {
    return (uint)(((ulong)jxr_min_u32(value, maximum) * (ulong)jxr_min_u32(alpha, maximum)
                   + (ulong)(maximum / 2u)) / (ulong)maximum);
}

__device__ __forceinline__ int jxr_signed_premultiply(
    int value, int alpha, int maximum) {
    jxr_long magnitude = (jxr_long)jxr_abs_long(value);
    jxr_long result = (magnitude * (jxr_long)jxr_clamp_i32(alpha, 0, maximum)
                       + (jxr_long)(maximum / 2)) / (jxr_long)maximum;
    return (int)(value < 0 ? -result : result);
}

__device__ __forceinline__ bool jxr_component(
    const int *samples, const JxrSamplePlaneAbi *planes, uint channel, uint x, uint y,
    const JxrOutputAbi &params, int &sample, bool &alpha, uint *status) {
    uint primary_count = params.alpha_plane == JXR_UINT_MAX
        ? params.channels : params.channels - 1u;
    alpha = params.alpha_plane != JXR_UINT_MAX && channel == primary_count;
    if (alpha) {
        sample = jxr_read_plane(samples, planes[params.alpha_plane], x, y);
        return true;
    }
    if (params.output_color == 7u) {
        sample = jxr_read_plane(samples, planes[channel], x, y);
        return true;
    }
    if ((params.channel_layout == 12u || params.channel_layout == 13u) && channel == 3u) {
        sample = 0;
        alpha = false;
        return true;
    }
    int values[4];
    if (!jxr_converted(samples, planes, x, y, params, values, status)) return false;
    uint source_channel = channel;
    if ((params.channel_layout == 6u || params.channel_layout == 7u
         || params.channel_layout == 13u) && channel < 3u) {
        source_channel = 2u - channel;
    }
    sample = values[source_channel];
    return true;
}

__device__ __forceinline__ bool jxr_formatted_integer(
    const int *samples, const JxrSamplePlaneAbi *planes, uint channel, uint x, uint y,
    const JxrOutputAbi &params, int &value, uint *status) {
    if ((params.channel_layout == 12u || params.channel_layout == 13u) && channel == 3u) {
        value = 0;
        return true;
    }
    int sample;
    bool alpha;
    if (!jxr_component(samples, planes, channel, x, y, params, sample, alpha, status)
        || !jxr_scale(sample, channel, alpha, params, value, status)) {
        return false;
    }
    if (params.premultiply_alpha != 0u && !alpha && params.alpha_plane != JXR_UINT_MAX) {
        int alpha_sample = jxr_read_plane(samples, planes[params.alpha_plane], x, y);
        int alpha_value;
        if (!jxr_scale(alpha_sample, 0u, true, params, alpha_value, status)) return false;
        if (params.bit_depth == 1u) {
            value = (int)jxr_unsigned_premultiply(
                (uint)jxr_clamp_i32(value, 0, 255),
                (uint)jxr_clamp_i32(alpha_value, 0, 255), 255u);
        } else if (params.bit_depth == 2u || params.bit_depth == 3u) {
            value = (int)jxr_unsigned_premultiply(
                (uint)jxr_clamp_i32(value, 0, 65535),
                (uint)jxr_clamp_i32(alpha_value, 0, 65535), 65535u);
        } else if (params.bit_depth == 4u) {
            value = jxr_signed_premultiply(value, alpha_value, 32767);
        } else if (params.bit_depth == 6u) {
            value = jxr_signed_premultiply(value, alpha_value, JXR_INT_MAX);
        }
    }
    return true;
}

__device__ __forceinline__ uint jxr_output_index(
    JxrSurfacePlaneAbi surface, uint x, uint y, uint channel, uint bytes) {
    return surface.byte_offset + y * surface.row_stride_bytes
        + (x * surface.channels + channel) * bytes;
}

__device__ __forceinline__ void jxr_store_u16(uchar *output, uint index, uint value) {
    output[index] = (uchar)value;
    output[index + 1u] = (uchar)(value >> 8u);
}

__device__ __forceinline__ void jxr_store_u32(uchar *output, uint index, uint value) {
    output[index] = (uchar)value;
    output[index + 1u] = (uchar)(value >> 8u);
    output[index + 2u] = (uchar)(value >> 16u);
    output[index + 3u] = (uchar)(value >> 24u);
}

extern "C" __global__ void jxr_output_u8(
    const int *samples, const JxrSamplePlaneAbi *planes,
    const JxrSurfacePlaneAbi *surfaces, uchar *output, uint *status,
    JxrOutputAbi params, uint output_base) {
    output += output_base;
    uint x_out = blockIdx.x * blockDim.x + threadIdx.x;
    uint y_out = blockIdx.y * blockDim.y + threadIdx.y;
    JxrSurfacePlaneAbi surface = surfaces[params.output_plane];
    const bool failed = jxr_block_failed(status);
    if (x_out >= surface.width || y_out >= surface.height || failed) return;
    if (params.output_plane_count > 1u) {
        uint source_plane = params.output_plane < 3u ? params.output_plane : params.alpha_plane;
        uint x = params.crop_x
            / (params.output_plane == 1u || params.output_plane == 2u ? 2u : 1u) + x_out;
        uint y_divisor = params.chroma_sampling == 1u
                && (params.output_plane == 1u || params.output_plane == 2u) ? 2u : 1u;
        uint y = params.crop_y / y_divisor + y_out;
        int scaled;
        bool alpha = params.output_plane >= 3u;
        if (!jxr_scale(
                jxr_read_plane(samples, planes[source_plane], x, y),
                params.output_plane, alpha, params, scaled, status)) return;
        output[jxr_output_index(surface, x_out, y_out, 0u, 1u)] =
            (uchar)jxr_clamp_i32(scaled, 0, 255);
        return;
    }
    uint x = params.crop_x + x_out;
    uint y = params.crop_y + y_out;
    for (uint channel = 0; channel < params.channels; ++channel) {
        int value;
        if (!jxr_formatted_integer(
                samples, planes, channel, x, y, params, value, status)) return;
        output[jxr_output_index(surface, x_out, y_out, channel, 1u)] =
            (uchar)jxr_clamp_i32(value, 0, 255);
    }
}

__device__ __forceinline__ bool jxr_integer_store_value(
    const int *samples, const JxrSamplePlaneAbi *planes,
    JxrSurfacePlaneAbi surface, uint x_out, uint y_out, uint channel,
    const JxrOutputAbi &params, int &value, uint *status) {
    uint x = params.crop_x + x_out;
    uint y = params.crop_y + y_out;
    if (params.output_plane_count > 1u) {
        uint source_plane = params.output_plane < 3u ? params.output_plane : params.alpha_plane;
        x = params.crop_x
            / (params.output_plane == 1u || params.output_plane == 2u ? 2u : 1u) + x_out;
        uint divisor = params.chroma_sampling == 1u
                && (params.output_plane == 1u || params.output_plane == 2u) ? 2u : 1u;
        y = params.crop_y / divisor + y_out;
        bool alpha = params.output_plane >= 3u;
        return jxr_scale(
            jxr_read_plane(samples, planes[source_plane], x, y),
            params.output_plane, alpha, params, value, status);
    }
    return jxr_formatted_integer(samples, planes, channel, x, y, params, value, status);
}

#define JXR_INTEGER_KERNEL(NAME, BYTES, MINIMUM, MAXIMUM, STORE) \
extern "C" __global__ void NAME( \
    const int *samples, const JxrSamplePlaneAbi *planes, \
    const JxrSurfacePlaneAbi *surfaces, uchar *output, uint *status, \
    JxrOutputAbi params, uint output_base) { \
    output += output_base; \
    uint x_out = blockIdx.x * blockDim.x + threadIdx.x; \
    uint y_out = blockIdx.y * blockDim.y + threadIdx.y; \
    JxrSurfacePlaneAbi surface = surfaces[params.output_plane]; \
    const bool failed = jxr_block_failed(status); \
    if (x_out >= surface.width || y_out >= surface.height || failed) return; \
    uint channels = params.output_plane_count > 1u ? 1u : surface.channels; \
    for (uint channel = 0; channel < channels; ++channel) { \
        int value; \
        if (!jxr_integer_store_value(samples, planes, surface, x_out, y_out, channel, params, value, status)) return; \
        uint index = jxr_output_index(surface, x_out, y_out, channel, BYTES); \
        STORE(output, index, (uint)jxr_clamp_i32(value, MINIMUM, MAXIMUM)); \
    } \
}

#define JXR_STORE16(output, index, value) jxr_store_u16(output, index, value)
#define JXR_STORE32(output, index, value) jxr_store_u32(output, index, value)
JXR_INTEGER_KERNEL(jxr_output_u16, 2u, 0,
    (params.bit_depth == 2u ? 1023 : 65535), JXR_STORE16)
JXR_INTEGER_KERNEL(jxr_output_i16, 2u, -32768, 32767, JXR_STORE16)
JXR_INTEGER_KERNEL(jxr_output_i32, 4u, JXR_INT_MIN, JXR_INT_MAX, JXR_STORE32)

extern "C" __global__ void jxr_output_f16(
    const int *samples, const JxrSamplePlaneAbi *planes,
    const JxrSurfacePlaneAbi *surfaces, uchar *output, uint *status,
    JxrOutputAbi params, uint output_base) {
    output += output_base;
    uint x_out = blockIdx.x * blockDim.x + threadIdx.x;
    uint y_out = blockIdx.y * blockDim.y + threadIdx.y;
    JxrSurfacePlaneAbi surface = surfaces[0];
    const bool failed = jxr_block_failed(status);
    if (x_out >= surface.width || y_out >= surface.height || failed) return;
    uint x = params.crop_x + x_out;
    uint y = params.crop_y + y_out;
    uint alpha_bits = params.alpha_plane == JXR_UINT_MAX ? 0u
        : jxr_f16_bits(
            jxr_read_plane(samples, planes[params.alpha_plane], x, y), true, params, status);
    for (uint channel = 0; channel < params.channels; ++channel) {
        uint index = jxr_output_index(surface, x_out, y_out, channel, 2u);
        if ((params.channel_layout == 12u || params.channel_layout == 13u) && channel == 3u) {
            jxr_store_u16(output, index, 0u);
            continue;
        }
        int sample;
        bool alpha;
        if (!jxr_component(
                samples, planes, channel, x, y, params, sample, alpha, status)) return;
        uint bits = jxr_f16_bits(sample, alpha, params, status);
        if (params.premultiply_alpha != 0u && !alpha) {
            uint sign = bits & 0x8000u;
            bits = sign | jxr_unsigned_premultiply(
                bits & 0x7fffu,
                (alpha_bits & 0x8000u) == 0u ? alpha_bits & 0x7fffu : 0u,
                0x7fffu);
        }
        jxr_store_u16(output, index, bits);
    }
}

extern "C" __global__ void jxr_output_f32(
    const int *samples, const JxrSamplePlaneAbi *planes,
    const JxrSurfacePlaneAbi *surfaces, uchar *output, uint *status,
    JxrOutputAbi params, uint output_base) {
    output += output_base;
    uint x_out = blockIdx.x * blockDim.x + threadIdx.x;
    uint y_out = blockIdx.y * blockDim.y + threadIdx.y;
    JxrSurfacePlaneAbi surface = surfaces[0];
    const bool failed = jxr_block_failed(status);
    if (x_out >= surface.width || y_out >= surface.height || failed) return;
    uint x = params.crop_x + x_out;
    uint y = params.crop_y + y_out;
    float alpha_value = 1.0f;
    if (params.alpha_plane != JXR_UINT_MAX) {
        alpha_value = jxr_clamp_f32(__uint_as_float(jxr_f32_bits(
            jxr_read_plane(samples, planes[params.alpha_plane], x, y), true, params, status)),
            0.0f, 1.0f);
    }
    for (uint channel = 0; channel < params.channels; ++channel) {
        uint index = jxr_output_index(surface, x_out, y_out, channel, 4u);
        if ((params.channel_layout == 12u || params.channel_layout == 13u) && channel == 3u) {
            jxr_store_u32(output, index, 0u);
            continue;
        }
        int sample;
        bool alpha;
        if (!jxr_component(
                samples, planes, channel, x, y, params, sample, alpha, status)) return;
        float value = __uint_as_float(jxr_f32_bits(sample, alpha, params, status));
        if (params.premultiply_alpha != 0u && !alpha) value *= alpha_value;
        jxr_store_u32(output, index, __float_as_uint(value));
    }
}

extern "C" __global__ void jxr_output_bits(
    const int *samples, const JxrSamplePlaneAbi *planes,
    const JxrSurfacePlaneAbi *surfaces, uchar *output, uint *status,
    JxrOutputAbi params, uint output_base) {
    output += output_base;
    uint byte_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y_out = blockIdx.y * blockDim.y + threadIdx.y;
    JxrSurfacePlaneAbi surface = surfaces[0];
    uint row_bytes = (surface.width + 7u) / 8u;
    const bool failed = jxr_block_failed(status);
    if (byte_x >= row_bytes || y_out >= surface.height || failed) return;
    uchar byte = 0;
    for (uint bit = 0; bit < 8u; ++bit) {
        uint pixel_x = byte_x * 8u + bit;
        if (pixel_x >= surface.width) break;
        int value;
        if (!jxr_scale(jxr_read_plane(
                samples, planes[0], params.crop_x + pixel_x, params.crop_y + y_out),
                0u, false, params, value, status)) return;
        uint packed = (uint)jxr_clamp_i32(value, 0, 1);
        if (params.bit_black != 0u) packed = 1u - packed;
        byte |= (uchar)(packed << (7u - bit));
    }
    output[surface.byte_offset + y_out * surface.row_stride_bytes + byte_x] = byte;
}

extern "C" __global__ void jxr_output_packed16(
    const int *samples, const JxrSamplePlaneAbi *planes,
    const JxrSurfacePlaneAbi *surfaces, uchar *output, uint *status,
    JxrOutputAbi params, uint output_base) {
    output += output_base;
    uint x_out = blockIdx.x * blockDim.x + threadIdx.x;
    uint y_out = blockIdx.y * blockDim.y + threadIdx.y;
    JxrSurfacePlaneAbi surface = surfaces[0];
    const bool failed = jxr_block_failed(status);
    if (x_out >= surface.width || y_out >= surface.height || failed) return;
    int values[4];
    if (!jxr_converted(
            samples, planes, params.crop_x + x_out, params.crop_y + y_out,
            params, values, status)) return;
    uint packed = 0u;
    for (uint channel = 0; channel < 3u; ++channel) {
        int value;
        if (!jxr_scale(values[channel], channel, false, params, value, status)) return;
        uint maximum = params.bit_depth == 10u && channel == 1u ? 63u : 31u;
        uint shift = params.bit_depth == 10u
            ? (channel == 0u ? 11u : (channel == 1u ? 5u : 0u))
            : (2u - channel) * 5u;
        packed |= (uint)jxr_clamp_i32(value, 0, (int)maximum) << shift;
    }
    jxr_store_u16(
        output, surface.byte_offset + y_out * surface.row_stride_bytes + x_out * 2u, packed);
}

extern "C" __global__ void jxr_output_packed32(
    const int *samples, const JxrSamplePlaneAbi *planes,
    const JxrSurfacePlaneAbi *surfaces, uchar *output, uint *status,
    JxrOutputAbi params, uint output_base) {
    output += output_base;
    uint x_out = blockIdx.x * blockDim.x + threadIdx.x;
    uint y_out = blockIdx.y * blockDim.y + threadIdx.y;
    JxrSurfacePlaneAbi surface = surfaces[0];
    const bool failed = jxr_block_failed(status);
    if (x_out >= surface.width || y_out >= surface.height || failed) return;
    int values[4];
    if (!jxr_converted(
            samples, planes, params.crop_x + x_out, params.crop_y + y_out,
            params, values, status)) return;
    uint packed = 0u;
    if (params.output_color == 6u) {
        int scaled[3];
        uint exponent = 0u;
        uint mantissa[3];
        uint local_exponent[3];
        for (uint channel = 0; channel < 3u; ++channel) {
            if (!jxr_scale(
                    values[channel], channel, false, params, scaled[channel], status)) return;
            if (scaled[channel] <= 0) {
                mantissa[channel] = 0;
                local_exponent[channel] = 0;
            } else if ((scaled[channel] >> 7) > 1) {
                mantissa[channel] = (uint)((scaled[channel] & 127) + 128);
                local_exponent[channel] = (uint)(scaled[channel] >> 7);
            } else {
                mantissa[channel] = (uint)scaled[channel];
                local_exponent[channel] = 1u;
            }
            exponent = jxr_max_u32(exponent, local_exponent[channel]);
        }
        for (uint channel = 0; channel < 3u; ++channel) {
            if (exponent > local_exponent[channel]) {
                uint difference = exponent - local_exponent[channel];
                mantissa[channel] = difference >= 31u ? 0u
                    : (uint)((2u * mantissa[channel] + 1u) >> (difference + 1u));
            }
        }
        packed = jxr_min_u32(mantissa[0], 255u)
            | (jxr_min_u32(mantissa[1], 255u) << 8u)
            | (jxr_min_u32(mantissa[2], 255u) << 16u)
            | (jxr_min_u32(exponent, 255u) << 24u);
    } else {
        for (uint channel = 0; channel < 3u; ++channel) {
            int value;
            if (!jxr_scale(values[channel], channel, false, params, value, status)) return;
            packed |= (uint)jxr_clamp_i32(value, 0, 1023) << ((2u - channel) * 10u);
        }
    }
    jxr_store_u32(
        output, surface.byte_offset + y_out * surface.row_stride_bytes + x_out * 4u, packed);
}
