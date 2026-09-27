inline int jxr_read_plane(device const int *samples, JxrSamplePlaneAbi plane, uint x, uint y) {
    return samples[plane.sample_offset + (y - plane.origin_y) * plane.width + x - plane.origin_x];
}

inline int4 jxr_centering(uint code) {
    switch (code) {
        case 0u: return int4(4,4,0,8);
        case 1u: return int4(5,3,1,7);
        case 2u: return int4(6,2,2,6);
        case 3u: return int4(7,1,3,5);
        default: return int4(8,0,4,4);
    }
}

// Exact `(first * fw + second * sw + 4) >> 3` for non-negative weights that
// sum to eight. Splitting each operand into `8q + r` keeps every intermediate
// inside `int`, and the weighted average of two `int` values always fits.
inline int jxr_weighted(int first, int fw, int second, int sw) {
    return fw * (first >> 3) + sw * (second >> 3)
        + ((fw * (first & 7) + sw * (second & 7) + 4) >> 3);
}

inline void jxr_upsample_pair(int previous, int current, int next, uint centering,
                              thread int pair[2]) {
    int4 h = jxr_centering(centering);
    pair[0] = jxr_weighted(previous, h.z, current, h.w);
    pair[1] = jxr_weighted(current, h.x, next, h.y);
}

inline int jxr_clamped_plane(device const int *samples, JxrSamplePlaneAbi plane, int x, int y) {
    int local_x = clamp(x - int(plane.origin_x), 0, int(plane.width) - 1);
    int local_y = clamp(y - int(plane.origin_y), 0, int(plane.height) - 1);
    return samples[plane.sample_offset + uint(local_y) * plane.width + uint(local_x)];
}

inline int jxr_chroma_sample(device const int *samples, JxrSamplePlaneAbi plane,
                             uint full_x, uint full_y, constant JxrOutputAbi &params) {
    if (params.chroma_sampling == 3u) return jxr_read_plane(samples, plane, full_x, full_y);
    int chroma_x = int(full_x >> 1u);
    int pair[2];
    if (params.chroma_sampling == 2u) {
        jxr_upsample_pair(
            jxr_clamped_plane(samples, plane, chroma_x - 1, int(full_y)),
            jxr_clamped_plane(samples, plane, chroma_x, int(full_y)),
            jxr_clamped_plane(samples, plane, chroma_x + 1, int(full_y)),
            params.chroma_centering_x, pair);
        return pair[full_x & 1u];
    }
    int chroma_y = int(full_y >> 1u);
    int vertical[3];
    for (int column = -1; column <= 1; ++column) {
        jxr_upsample_pair(
            jxr_clamped_plane(samples, plane, chroma_x + column, chroma_y - 1),
            jxr_clamped_plane(samples, plane, chroma_x + column, chroma_y),
            jxr_clamped_plane(samples, plane, chroma_x + column, chroma_y + 1),
            params.chroma_centering_y, pair);
        vertical[column + 1] = pair[full_y & 1u];
    }
    jxr_upsample_pair(vertical[0], vertical[1], vertical[2], params.chroma_centering_x, pair);
    return pair[full_x & 1u];
}

inline void jxr_primary_values(device const int *samples, device const JxrSamplePlaneAbi *planes,
                               uint x, uint y, constant JxrOutputAbi &params, thread int values[4]) {
    for (uint i = 0; i < 4; ++i) values[i] = 0;
    values[0] = jxr_read_plane(samples, planes[0], x, y);
    if (params.component_count == 1u) return;
    if (params.internal_color == 1u) {
        values[1] = jxr_chroma_sample(samples, planes[1], x, y, params);
        values[2] = jxr_chroma_sample(samples, planes[2], x, y, params);
    } else {
        for (uint i = 1; i < min(params.component_count, 4u); ++i)
            values[i] = jxr_read_plane(samples, planes[i], x, y);
    }
}

inline void jxr_converted(device const int *samples, device const JxrSamplePlaneAbi *planes,
                          uint x, uint y, constant JxrOutputAbi &params,
                          thread int values[4], thread bool &overflow) {
    int input[4];
    jxr_primary_values(samples, planes, x, y, params, input);
    for (uint i = 0; i < 4; ++i) values[i] = input[i];
    if (params.internal_color == 0u && params.output_color == 2u) {
        values[1] = input[0]; values[2] = input[0];
    } else if (params.internal_color == 1u && (params.output_color == 2u || params.output_color == 6u)) {
        const int temporary = jxr_sub(0, input[1], overflow);
        const int green = jxr_sub(input[0], temporary >> 1, overflow);
        int red = jxr_add(temporary, green, overflow);
        red = jxr_sub(red, (input[2] >> 1) + (input[2] & 1), overflow);
        values[2] = jxr_add(input[2], red, overflow);
        values[0] = red; values[1] = green;
        if (params.bit_depth >= 8u && params.bit_depth <= 10u && params.red_blue_not_swapped == 0u) {
            int swap = values[0]; values[0] = values[2]; values[2] = swap;
        }
    } else if (params.internal_color == 5u && params.output_color == 3u) {
        const int black = jxr_add(input[3], input[0] >> 1, overflow);
        int magenta = jxr_sub(black, input[0], overflow);
        magenta = jxr_sub(magenta, input[1] >> 1, overflow);
        int cyan = jxr_add(input[1], magenta, overflow);
        cyan = jxr_add(cyan, input[2] >> 1, overflow);
        values[2] = jxr_sub(cyan, input[2], overflow);
        values[0] = cyan; values[1] = magenta; values[3] = black;
    } else if (params.internal_color == 5u && params.output_color == 4u) {
        values[0] = input[1]; values[1] = input[2]; values[2] = input[3]; values[3] = input[0];
    }
}

// Loads and color-converts one pixel. Every channel store selects from this
// result instead of repeating chroma upsampling and conversion per channel.
inline void jxr_load_pixel(device const int *samples, device const JxrSamplePlaneAbi *planes,
                           uint x, uint y, constant JxrOutputAbi &params,
                           thread int converted[4], thread bool &overflow) {
    for (uint i = 0; i < 4; ++i) converted[i] = 0;
    if (params.output_color != 7u) jxr_converted(samples, planes, x, y, params, converted, overflow);
}

inline int jxr_base_bias(uint depth, uint shift_bits) {
    switch (depth) {
        case 1u: return 128;
        case 2u: case 9u: return 512;
        case 3u: return 32768 >> shift_bits;
        case 8u: return 16;
        case 10u: return 32;
        default: return 0;
    }
}

// Matches the CPU `checked_shl` plus round-trip test.
inline int jxr_shl(int value, uint bits, thread bool &overflow) {
    if (bits >= 32u) {
        overflow = true;
        return 0;
    }
    const int shifted = as_type<int>(as_type<uint>(value) << bits);
    overflow |= (shifted >> bits) != value;
    return shifted;
}

inline int jxr_scale(int sample, uint component, bool alpha, constant JxrOutputAbi &params,
                     thread bool &overflow) {
    uint depth = params.bit_depth;
    uint scaled = alpha ? params.alpha_scaled : params.scaled;
    uint shift_bits = alpha ? params.alpha_shift_bits : params.shift_bits;
    int bias = jxr_base_bias(depth, shift_bits);
    if (!alpha && params.output_color == 3u) bias = component < 3u ? bias >> 1 : -(bias >> 1);
    if (!alpha && params.output_color == 6u) bias = 0;
    uint scale = scaled * 3u;
    // |bias| <= 32768 and scale <= 3, so the scaled bias cannot overflow.
    int biased = jxr_add(sample, bias * int(1u << scale), overflow);
    int rounding = scaled == 0u ? 0 : ((depth == 0u || depth == 3u) ? 4 : 3);
    int rounded = jxr_add(biased, rounding, overflow);
    uint downshift = scale + ((depth == 10u && component != 1u) ? 1u : 0u);
    int downscaled = rounded >> downshift;
    if (depth == 3u || depth == 4u || depth == 6u) return jxr_shl(downscaled, shift_bits, overflow);
    return downscaled;
}

inline uint jxr_f16_bits(int sample, bool alpha, constant JxrOutputAbi &params,
                         thread bool &overflow) {
    int scaled = jxr_scale(sample, 0u, alpha, params, overflow);
    uint sign = scaled < 0 ? 0x8000u : 0u;
    uint magnitude = scaled < 0 ? 0u - as_type<uint>(scaled) : as_type<uint>(scaled);
    return sign | min(magnitude, 32767u);
}

inline uint jxr_f32_bits(int sample, bool alpha, constant JxrOutputAbi &params,
                         thread bool &overflow) {
    int scaled = jxr_scale(sample, 0u, alpha, params, overflow);
    uint length = alpha ? params.alpha_mantissa_length : params.mantissa_length;
    int exponent_bias = as_type<int>(alpha ? params.alpha_exponent_bias_bits : params.exponent_bias_bits);
    uint sign = scaled < 0 ? 0x80000000u : 0u;
    ulong magnitude = ulong(abs(long(scaled)));
    ulong implicit = 1ul << length;
    long exponent = long(magnitude >> length);
    ulong mantissa = (magnitude & (implicit - 1ul)) | implicit;
    if (exponent == 0l) { mantissa ^= implicit; exponent = 1l; }
    exponent = exponent - long(exponent_bias) + 127l;
    while (mantissa < implicit && exponent > 1l && mantissa > 0ul) { --exponent; mantissa <<= 1u; }
    if (mantissa < implicit) exponent = 0l; else mantissa ^= implicit;
    mantissa <<= 23u - length;
    if (exponent < 0l || exponent > 255l || mantissa > 0x7ffffful) {
        overflow = true;
        return 0u;
    }
    return sign | (uint(exponent) << 23u) | uint(mantissa);
}

// Callers pass `maximum <= 65535`, so the rounded product fits in 32 bits.
inline uint jxr_unsigned_premultiply(uint value, uint alpha, uint maximum) {
    return (min(value, maximum) * min(alpha, maximum) + maximum / 2u) / maximum;
}

inline int jxr_signed_premultiply(int value, int alpha, int maximum) {
    long magnitude = abs(long(value));
    long result = (magnitude * long(clamp(alpha, 0, maximum)) + long(maximum / 2)) / long(maximum);
    return int(value < 0 ? -result : result);
}

inline bool jxr_is_padding(uint channel, constant JxrOutputAbi &params) {
    return (params.channel_layout == 12u || params.channel_layout == 13u) && channel == 3u;
}

inline int jxr_channel_sample(device const int *samples, device const JxrSamplePlaneAbi *planes,
                              uint channel, uint x, uint y, constant JxrOutputAbi &params,
                              thread const int converted[4], thread bool &alpha) {
    uint primary_count = params.alpha_plane == UINT_MAX ? params.channels : params.channels - 1u;
    alpha = params.alpha_plane != UINT_MAX && channel == primary_count;
    if (alpha) return jxr_read_plane(samples, planes[params.alpha_plane], x, y);
    if (params.output_color == 7u) return jxr_read_plane(samples, planes[channel], x, y);
    if (jxr_is_padding(channel, params)) return 0;
    uint source_channel = channel;
    if ((params.channel_layout == 6u || params.channel_layout == 7u || params.channel_layout == 13u) && channel < 3u)
        source_channel = 2u - channel;
    return converted[source_channel];
}

// Scaled alpha used to premultiply integer color channels, or zero when unused.
inline int jxr_premultiply_alpha(device const int *samples, device const JxrSamplePlaneAbi *planes,
                                 uint x, uint y, constant JxrOutputAbi &params,
                                 thread bool &overflow) {
    if (params.premultiply_alpha == 0u || params.alpha_plane == UINT_MAX) return 0;
    return jxr_scale(jxr_read_plane(samples, planes[params.alpha_plane], x, y), 0u, true, params, overflow);
}

inline int jxr_formatted_integer(device const int *samples, device const JxrSamplePlaneAbi *planes,
                                 uint channel, uint x, uint y, constant JxrOutputAbi &params,
                                 thread const int converted[4], int alpha_value,
                                 thread bool &overflow) {
    if (jxr_is_padding(channel, params)) return 0;
    bool alpha;
    int sample = jxr_channel_sample(samples, planes, channel, x, y, params, converted, alpha);
    int value = jxr_scale(sample, channel, alpha, params, overflow);
    if (params.premultiply_alpha != 0u && !alpha && params.alpha_plane != UINT_MAX) {
        if (params.bit_depth == 1u) value = int(jxr_unsigned_premultiply(uint(clamp(value,0,255)), uint(clamp(alpha_value,0,255)), 255u));
        else if (params.bit_depth == 2u || params.bit_depth == 3u) value = int(jxr_unsigned_premultiply(uint(clamp(value,0,65535)), uint(clamp(alpha_value,0,65535)), 65535u));
        else if (params.bit_depth == 4u) value = jxr_signed_premultiply(value, alpha_value, 32767);
        else if (params.bit_depth == 6u) value = jxr_signed_premultiply(value, alpha_value, INT_MAX);
    }
    return value;
}

inline uint jxr_output_index(JxrSurfacePlaneAbi surface, uint x, uint y, uint channel, uint bytes) {
    return surface.byte_offset + y * surface.row_stride_bytes + (x * surface.channels + channel) * bytes;
}

// Planar surfaces store one scaled sample per thread from their own plane.
inline int jxr_planar_value(device const int *samples, device const JxrSamplePlaneAbi *planes,
                            uint2 gid, constant JxrOutputAbi &params, thread bool &overflow) {
    uint source_plane = params.output_plane < 3u ? params.output_plane : params.alpha_plane;
    bool chroma = params.output_plane == 1u || params.output_plane == 2u;
    uint x = params.crop_x / (chroma ? 2u : 1u) + gid.x;
    uint y = params.crop_y / ((params.chroma_sampling == 1u && chroma) ? 2u : 1u) + gid.y;
    return jxr_scale(jxr_read_plane(samples, planes[source_plane], x, y), params.output_plane,
                     params.output_plane >= 3u, params, overflow);
}

kernel void jxr_output_u8(device const int *samples [[buffer(0)]],
                          device const JxrSamplePlaneAbi *planes [[buffer(1)]],
                          device const JxrSurfacePlaneAbi *surfaces [[buffer(2)]],
                          device uchar *output [[buffer(3)]], device atomic_uint *status [[buffer(4)]],
                          constant JxrOutputAbi &params [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {
    JxrSurfacePlaneAbi surface = surfaces[params.output_plane];
    if (gid.x >= surface.width || gid.y >= surface.height || jxr_failed(status)) return;
    bool overflow = false;
    if (params.output_plane_count > 1u) {
        int value = jxr_planar_value(samples, planes, gid, params, overflow);
        if (overflow) { jxr_fail(status, 16u); return; }
        output[jxr_output_index(surface, gid.x, gid.y, 0u, 1u)] = uchar(clamp(value, 0, 255));
        return;
    }
    uint x = params.crop_x + gid.x, y = params.crop_y + gid.y;
    int converted[4];
    jxr_load_pixel(samples, planes, x, y, params, converted, overflow);
    int alpha_value = jxr_premultiply_alpha(samples, planes, x, y, params, overflow);
    for (uint channel = 0; channel < params.channels; ++channel) {
        int value = jxr_formatted_integer(samples, planes, channel, x, y, params, converted,
                                          alpha_value, overflow);
        output[jxr_output_index(surface, gid.x, gid.y, channel, 1u)] = uchar(clamp(value, 0, 255));
    }
    if (overflow) jxr_fail(status, 16u);
}

#define JXR_INTEGER_STORE(NAME, TYPE, BYTES, MINIMUM, MAXIMUM) \
kernel void NAME(device const int *samples [[buffer(0)]], device const JxrSamplePlaneAbi *planes [[buffer(1)]], \
                 device const JxrSurfacePlaneAbi *surfaces [[buffer(2)]], device uchar *output [[buffer(3)]], \
                 device atomic_uint *status [[buffer(4)]], constant JxrOutputAbi &params [[buffer(5)]], \
                 uint2 gid [[thread_position_in_grid]]) { \
    JxrSurfacePlaneAbi surface = surfaces[params.output_plane]; \
    if (gid.x >= surface.width || gid.y >= surface.height || jxr_failed(status)) return; \
    bool overflow = false; \
    if (params.output_plane_count > 1u) { \
        int value = jxr_planar_value(samples, planes, gid, params, overflow); \
        if (overflow) { jxr_fail(status, 16u); return; } \
        *reinterpret_cast<device TYPE *>(output + jxr_output_index(surface, gid.x, gid.y, 0u, BYTES)) = \
            TYPE(clamp(value, MINIMUM, MAXIMUM)); \
        return; \
    } \
    uint x = params.crop_x + gid.x, y = params.crop_y + gid.y; \
    int converted[4]; \
    jxr_load_pixel(samples, planes, x, y, params, converted, overflow); \
    int alpha_value = jxr_premultiply_alpha(samples, planes, x, y, params, overflow); \
    for (uint channel = 0; channel < surface.channels; ++channel) { \
        int value = jxr_formatted_integer(samples, planes, channel, x, y, params, converted, \
                                          alpha_value, overflow); \
        *reinterpret_cast<device TYPE *>(output + jxr_output_index(surface, gid.x, gid.y, channel, BYTES)) = \
            TYPE(clamp(value, MINIMUM, MAXIMUM)); \
    } \
    if (overflow) jxr_fail(status, 16u); \
}

JXR_INTEGER_STORE(jxr_output_u16, ushort, 2u, 0, (params.bit_depth == 2u ? 1023 : 65535))
JXR_INTEGER_STORE(jxr_output_i16, short, 2u, -32768, 32767)
JXR_INTEGER_STORE(jxr_output_i32, int, 4u, INT_MIN, INT_MAX)

kernel void jxr_output_f16(device const int *samples [[buffer(0)]], device const JxrSamplePlaneAbi *planes [[buffer(1)]],
                           device const JxrSurfacePlaneAbi *surfaces [[buffer(2)]], device uchar *output [[buffer(3)]],
                           device atomic_uint *status [[buffer(4)]], constant JxrOutputAbi &params [[buffer(5)]],
                           uint2 gid [[thread_position_in_grid]]) {
    JxrSurfacePlaneAbi surface = surfaces[0];
    if (gid.x >= surface.width || gid.y >= surface.height || jxr_failed(status)) return;
    uint x = params.crop_x + gid.x, y = params.crop_y + gid.y;
    bool overflow = false;
    uint alpha_bits = params.alpha_plane == UINT_MAX
        ? 0u : jxr_f16_bits(jxr_read_plane(samples, planes[params.alpha_plane], x, y), true, params, overflow);
    int converted[4];
    jxr_load_pixel(samples, planes, x, y, params, converted, overflow);
    for (uint c = 0; c < params.channels; ++c) {
        device ushort *destination = reinterpret_cast<device ushort *>(output + jxr_output_index(surface, gid.x, gid.y, c, 2u));
        if (jxr_is_padding(c, params)) { *destination = 0; continue; }
        bool alpha;
        uint bits = jxr_f16_bits(jxr_channel_sample(samples, planes, c, x, y, params, converted, alpha), alpha, params, overflow);
        if (params.premultiply_alpha != 0u && !alpha) {
            uint sign = bits & 0x8000u;
            bits = sign | jxr_unsigned_premultiply(bits & 0x7fffu, (alpha_bits & 0x8000u) == 0u ? alpha_bits & 0x7fffu : 0u, 0x7fffu);
        }
        *destination = ushort(bits);
    }
    if (overflow) jxr_fail(status, 16u);
}

kernel void jxr_output_f32(device const int *samples [[buffer(0)]], device const JxrSamplePlaneAbi *planes [[buffer(1)]],
                           device const JxrSurfacePlaneAbi *surfaces [[buffer(2)]], device uchar *output [[buffer(3)]],
                           device atomic_uint *status [[buffer(4)]], constant JxrOutputAbi &params [[buffer(5)]],
                           uint2 gid [[thread_position_in_grid]]) {
    JxrSurfacePlaneAbi surface = surfaces[0];
    if (gid.x >= surface.width || gid.y >= surface.height || jxr_failed(status)) return;
    uint x = params.crop_x + gid.x, y = params.crop_y + gid.y;
    bool overflow = false;
    float alpha_value = 1.0f;
    if (params.alpha_plane != UINT_MAX)
        alpha_value = clamp(as_type<float>(jxr_f32_bits(jxr_read_plane(samples, planes[params.alpha_plane], x, y), true, params, overflow)), 0.0f, 1.0f);
    int converted[4];
    jxr_load_pixel(samples, planes, x, y, params, converted, overflow);
    for (uint c = 0; c < params.channels; ++c) {
        device float *destination = reinterpret_cast<device float *>(output + jxr_output_index(surface, gid.x, gid.y, c, 4u));
        if (jxr_is_padding(c, params)) { *destination = 0.0f; continue; }
        bool alpha;
        float value = as_type<float>(jxr_f32_bits(jxr_channel_sample(samples, planes, c, x, y, params, converted, alpha), alpha, params, overflow));
        if (params.premultiply_alpha != 0u && !alpha) value *= alpha_value;
        *destination = value;
    }
    if (overflow) jxr_fail(status, 16u);
}

kernel void jxr_output_bits(device const int *samples [[buffer(0)]], device const JxrSamplePlaneAbi *planes [[buffer(1)]],
                            device const JxrSurfacePlaneAbi *surfaces [[buffer(2)]], device uchar *output [[buffer(3)]],
                            device atomic_uint *status [[buffer(4)]], constant JxrOutputAbi &params [[buffer(5)]],
                            uint2 gid [[thread_position_in_grid]]) {
    JxrSurfacePlaneAbi surface = surfaces[0];
    uint row_bytes = (surface.width + 7u) / 8u;
    if (gid.x >= row_bytes || gid.y >= surface.height || jxr_failed(status)) return;
    bool overflow = false;
    uchar byte = 0;
    for (uint bit = 0; bit < 8u; ++bit) {
        uint px = gid.x * 8u + bit;
        if (px >= surface.width) break;
        int value = jxr_scale(jxr_read_plane(samples, planes[0], params.crop_x + px, params.crop_y + gid.y), 0u, false, params, overflow);
        uint packed = uint(clamp(value, 0, 1));
        if (params.bit_black != 0u) packed = 1u - packed;
        byte |= uchar(packed << (7u - bit));
    }
    if (overflow) { jxr_fail(status, 16u); return; }
    output[surface.byte_offset + gid.y * surface.row_stride_bytes + gid.x] = byte;
}

kernel void jxr_output_packed16(device const int *samples [[buffer(0)]], device const JxrSamplePlaneAbi *planes [[buffer(1)]],
                                device const JxrSurfacePlaneAbi *surfaces [[buffer(2)]], device uchar *output [[buffer(3)]],
                                device atomic_uint *status [[buffer(4)]], constant JxrOutputAbi &params [[buffer(5)]],
                                uint2 gid [[thread_position_in_grid]]) {
    JxrSurfacePlaneAbi surface = surfaces[0];
    if (gid.x >= surface.width || gid.y >= surface.height || jxr_failed(status)) return;
    bool overflow = false;
    int values[4];
    jxr_converted(samples, planes, params.crop_x + gid.x, params.crop_y + gid.y, params, values, overflow);
    uint packed = 0u;
    for (uint c = 0; c < 3u; ++c) {
        int value = jxr_scale(values[c], c, false, params, overflow);
        uint maximum = params.bit_depth == 10u && c == 1u ? 63u : 31u;
        uint shift = params.bit_depth == 10u ? (c == 0u ? 11u : (c == 1u ? 5u : 0u)) : (2u - c) * 5u;
        packed |= uint(clamp(value, 0, int(maximum))) << shift;
    }
    if (overflow) { jxr_fail(status, 16u); return; }
    *reinterpret_cast<device ushort *>(output + surface.byte_offset + gid.y * surface.row_stride_bytes + gid.x * 2u) = ushort(packed);
}

kernel void jxr_output_packed32(device const int *samples [[buffer(0)]], device const JxrSamplePlaneAbi *planes [[buffer(1)]],
                                device const JxrSurfacePlaneAbi *surfaces [[buffer(2)]], device uchar *output [[buffer(3)]],
                                device atomic_uint *status [[buffer(4)]], constant JxrOutputAbi &params [[buffer(5)]],
                                uint2 gid [[thread_position_in_grid]]) {
    JxrSurfacePlaneAbi surface = surfaces[0];
    if (gid.x >= surface.width || gid.y >= surface.height || jxr_failed(status)) return;
    bool overflow = false;
    int values[4];
    jxr_converted(samples, planes, params.crop_x + gid.x, params.crop_y + gid.y, params, values, overflow);
    uint packed = 0u;
    if (params.output_color == 6u) {
        int scaled[3]; uint exponent = 0u; uint mantissa[3]; uint local_exp[3];
        for (uint c = 0; c < 3u; ++c) {
            scaled[c] = jxr_scale(values[c], c, false, params, overflow);
            if (scaled[c] <= 0) { mantissa[c] = 0; local_exp[c] = 0; }
            else if ((scaled[c] >> 7) > 1) { mantissa[c] = uint((scaled[c] & 127) + 128); local_exp[c] = uint(scaled[c] >> 7); }
            else { mantissa[c] = uint(scaled[c]); local_exp[c] = 1u; }
            exponent = max(exponent, local_exp[c]);
        }
        for (uint c = 0; c < 3u; ++c)
            if (exponent > local_exp[c]) {
                uint d = exponent - local_exp[c];
                mantissa[c] = d >= 31u ? 0u : uint((2u * mantissa[c] + 1u) >> (d + 1u));
            }
        packed = (min(mantissa[0], 255u)) | (min(mantissa[1], 255u) << 8u) | (min(mantissa[2], 255u) << 16u) | (min(exponent, 255u) << 24u);
    } else {
        for (uint c = 0; c < 3u; ++c)
            packed |= uint(clamp(jxr_scale(values[c], c, false, params, overflow), 0, 1023)) << ((2u - c) * 10u);
    }
    if (overflow) { jxr_fail(status, 16u); return; }
    *reinterpret_cast<device uint *>(output + surface.byte_offset + gid.y * surface.row_stride_bytes + gid.x * 4u) = packed;
}
