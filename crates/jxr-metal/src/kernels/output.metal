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

inline bool jxr_weighted(int first, int fw, int second, int sw, thread int &result,
                         device atomic_uint *status) {
    long value = long(first) * long(fw) + long(second) * long(sw) + 4l;
    return jxr_narrow(value >> 3, result, status, 16u);
}

inline bool jxr_upsample_pair(int previous, int current, int next, uint centering,
                              thread int pair[2], device atomic_uint *status) {
    int4 h = jxr_centering(centering);
    return jxr_weighted(previous, h.z, current, h.w, pair[0], status)
        && jxr_weighted(current, h.x, next, h.y, pair[1], status);
}

inline int jxr_clamped_plane(device const int *samples, JxrSamplePlaneAbi plane, int x, int y) {
    int local_x = clamp(x - int(plane.origin_x), 0, int(plane.width) - 1);
    int local_y = clamp(y - int(plane.origin_y), 0, int(plane.height) - 1);
    return samples[plane.sample_offset + uint(local_y) * plane.width + uint(local_x)];
}

inline bool jxr_chroma_sample(device const int *samples, JxrSamplePlaneAbi plane,
                              uint full_x, uint full_y, constant JxrOutputAbi &params,
                              thread int &result, device atomic_uint *status) {
    if (params.chroma_sampling == 3u) {
        result = jxr_read_plane(samples, plane, full_x, full_y);
        return true;
    }
    int chroma_x = int(full_x >> 1u);
    if (params.chroma_sampling == 2u) {
        int pair[2];
        if (!jxr_upsample_pair(
            jxr_clamped_plane(samples, plane, chroma_x - 1, int(full_y)),
            jxr_clamped_plane(samples, plane, chroma_x, int(full_y)),
            jxr_clamped_plane(samples, plane, chroma_x + 1, int(full_y)),
            params.chroma_centering_x, pair, status)) return false;
        result = pair[full_x & 1u];
        return true;
    }
    int chroma_y = int(full_y >> 1u);
    int vertical[3];
    for (int column = -1; column <= 1; ++column) {
        int pair[2];
        if (!jxr_upsample_pair(
            jxr_clamped_plane(samples, plane, chroma_x + column, chroma_y - 1),
            jxr_clamped_plane(samples, plane, chroma_x + column, chroma_y),
            jxr_clamped_plane(samples, plane, chroma_x + column, chroma_y + 1),
            params.chroma_centering_y, pair, status)) return false;
        vertical[column + 1] = pair[full_y & 1u];
    }
    int horizontal[2];
    if (!jxr_upsample_pair(vertical[0], vertical[1], vertical[2],
                           params.chroma_centering_x, horizontal, status)) return false;
    result = horizontal[full_x & 1u];
    return true;
}

inline bool jxr_primary_values(device const int *samples, device const JxrSamplePlaneAbi *planes,
                               uint x, uint y, constant JxrOutputAbi &params,
                               thread int values[4], device atomic_uint *status) {
    for (uint i = 0; i < 4; ++i) values[i] = 0;
    values[0] = jxr_read_plane(samples, planes[0], x, y);
    if (params.component_count == 1u) return true;
    if (params.internal_color == 1u) {
        if (!jxr_chroma_sample(samples, planes[1], x, y, params, values[1], status)
            || !jxr_chroma_sample(samples, planes[2], x, y, params, values[2], status)) return false;
    } else {
        for (uint i = 1; i < min(params.component_count, 4u); ++i)
            values[i] = jxr_read_plane(samples, planes[i], x, y);
    }
    return true;
}

inline bool jxr_converted(device const int *samples, device const JxrSamplePlaneAbi *planes,
                          uint x, uint y, constant JxrOutputAbi &params,
                          thread int values[4], device atomic_uint *status) {
    int input[4];
    if (!jxr_primary_values(samples, planes, x, y, params, input, status)) return false;
    for (uint i = 0; i < 4; ++i) values[i] = input[i];
    if (params.internal_color == 0u && params.output_color == 2u) {
        values[1] = input[0]; values[2] = input[0];
    } else if (params.internal_color == 1u && (params.output_color == 2u || params.output_color == 6u)) {
        int temporary;
        int green;
        int red;
        if (!jxr_sub(0, input[1], temporary, status, 16u)
            || !jxr_sub(input[0], temporary >> 1, green, status, 16u)
            || !jxr_add(temporary, green, red, status, 16u)
            || !jxr_sub(red, (input[2] >> 1) + (input[2] & 1), red, status, 16u)
            || !jxr_add(input[2], red, values[2], status, 16u)) return false;
        values[0] = red; values[1] = green;
        if (params.bit_depth >= 8u && params.bit_depth <= 10u && params.red_blue_not_swapped == 0u) {
            int swap = values[0]; values[0] = values[2]; values[2] = swap;
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
            || !jxr_sub(cyan, input[2], values[2], status, 16u)) return false;
        values[0] = cyan; values[1] = magenta; values[3] = black;
    } else if (params.internal_color == 5u && params.output_color == 4u) {
        values[0] = input[1]; values[1] = input[2]; values[2] = input[3]; values[3] = input[0];
    }
    return true;
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

inline bool jxr_scale(int sample, uint component, bool alpha, constant JxrOutputAbi &params,
                      thread int &result, device atomic_uint *status) {
    uint depth = params.bit_depth;
    uint scaled = alpha ? params.alpha_scaled : params.scaled;
    uint shift_bits = alpha ? params.alpha_shift_bits : params.shift_bits;
    int bias = jxr_base_bias(depth, shift_bits);
    if (!alpha && params.output_color == 3u) bias = component < 3u ? bias >> 1 : -(bias >> 1);
    if (!alpha && params.output_color == 6u) bias = 0;
    uint scale = scaled * 3u;
    int shifted_bias;
    if (!jxr_narrow(long(bias) << scale, shifted_bias, status, 16u)) return false;
    int biased;
    if (!jxr_add(sample, shifted_bias, biased, status, 16u)) return false;
    int rounding = scaled == 0u ? 0 : ((depth == 0u || depth == 3u) ? 4 : 3);
    int rounded;
    if (!jxr_add(biased, rounding, rounded, status, 16u)) return false;
    uint downshift = scale + ((depth == 10u && component != 1u) ? 1u : 0u);
    int downscaled = rounded >> downshift;
    if (depth == 3u || depth == 4u || depth == 6u)
        return jxr_narrow(long(downscaled) << shift_bits, result, status, 16u);
    result = downscaled;
    return true;
}

inline uint jxr_f16_bits(int sample, bool alpha, constant JxrOutputAbi &params,
                         device atomic_uint *status) {
    int scaled;
    if (!jxr_scale(sample, 0u, alpha, params, scaled, status)) return 0u;
    uint sign = scaled < 0 ? 0x8000u : 0u;
    uint magnitude = uint(min(abs(long(scaled)), 32767l));
    return sign | magnitude;
}

inline uint jxr_f32_bits(int sample, bool alpha, constant JxrOutputAbi &params,
                         device atomic_uint *status) {
    int scaled;
    if (!jxr_scale(sample, 0u, alpha, params, scaled, status)) return 0u;
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
        jxr_fail(status, 16u); return 0u;
    }
    return sign | (uint(exponent) << 23u) | uint(mantissa);
}

inline uint jxr_unsigned_premultiply(uint value, uint alpha, uint maximum) {
    return uint((ulong(min(value, maximum)) * ulong(min(alpha, maximum)) + ulong(maximum / 2u)) / ulong(maximum));
}

inline int jxr_signed_premultiply(int value, int alpha, int maximum) {
    long magnitude = abs(long(value));
    long result = (magnitude * long(clamp(alpha, 0, maximum)) + long(maximum / 2)) / long(maximum);
    return int(value < 0 ? -result : result);
}

inline bool jxr_component(device const int *samples, device const JxrSamplePlaneAbi *planes,
                          uint channel, uint x, uint y, constant JxrOutputAbi &params,
                          thread int &sample, thread bool &alpha, device atomic_uint *status) {
    uint primary_count = params.alpha_plane == UINT_MAX ? params.channels : params.channels - 1u;
    alpha = params.alpha_plane != UINT_MAX && channel == primary_count;
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
    if ((params.channel_layout == 6u || params.channel_layout == 7u || params.channel_layout == 13u) && channel < 3u)
        source_channel = 2u - channel;
    sample = values[source_channel];
    return true;
}

inline bool jxr_formatted_integer(device const int *samples, device const JxrSamplePlaneAbi *planes,
                                  uint channel, uint x, uint y, constant JxrOutputAbi &params,
                                  thread int &value, device atomic_uint *status) {
    if ((params.channel_layout == 12u || params.channel_layout == 13u) && channel == 3u) {
        value = 0;
        return true;
    }
    int sample; bool alpha;
    if (!jxr_component(samples, planes, channel, x, y, params, sample, alpha, status)
        || !jxr_scale(sample, channel, alpha, params, value, status)) return false;
    if (params.premultiply_alpha != 0u && !alpha && params.alpha_plane != UINT_MAX) {
        int alpha_sample = jxr_read_plane(samples, planes[params.alpha_plane], x, y);
        int alpha_value;
        if (!jxr_scale(alpha_sample, 0u, true, params, alpha_value, status)) return false;
        if (params.bit_depth == 1u) value = int(jxr_unsigned_premultiply(uint(clamp(value,0,255)), uint(clamp(alpha_value,0,255)), 255u));
        else if (params.bit_depth == 2u || params.bit_depth == 3u) value = int(jxr_unsigned_premultiply(uint(clamp(value,0,65535)), uint(clamp(alpha_value,0,65535)), 65535u));
        else if (params.bit_depth == 4u) value = jxr_signed_premultiply(value, alpha_value, 32767);
        else if (params.bit_depth == 6u) value = jxr_signed_premultiply(value, alpha_value, INT_MAX);
    }
    return true;
}

inline uint jxr_output_index(JxrSurfacePlaneAbi surface, uint x, uint y, uint channel, uint bytes) {
    return surface.byte_offset + y * surface.row_stride_bytes + (x * surface.channels + channel) * bytes;
}

kernel void jxr_output_u8(device const int *samples [[buffer(0)]],
                          device const JxrSamplePlaneAbi *planes [[buffer(1)]],
                          device const JxrSurfacePlaneAbi *surfaces [[buffer(2)]],
                          device uchar *output [[buffer(3)]], device atomic_uint *status [[buffer(4)]],
                          constant JxrOutputAbi &params [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {
    JxrSurfacePlaneAbi surface = surfaces[params.output_plane];
    if (gid.x >= surface.width || gid.y >= surface.height || jxr_failed(status)) return;
    if (params.output_plane_count > 1u) {
        uint source_plane = params.output_plane < 3u ? params.output_plane : params.alpha_plane;
        uint x = params.crop_x / (params.output_plane == 1u || params.output_plane == 2u ? 2u : 1u) + gid.x;
        uint y_divisor = (params.chroma_sampling == 1u && (params.output_plane == 1u || params.output_plane == 2u)) ? 2u : 1u;
        uint y = params.crop_y / y_divisor + gid.y;
        int scaled;
        bool alpha = params.output_plane >= 3u;
        if (!jxr_scale(jxr_read_plane(samples, planes[source_plane], x, y), params.output_plane, alpha, params, scaled, status)) return;
        output[jxr_output_index(surface, gid.x, gid.y, 0u, 1u)] = uchar(clamp(scaled, 0, 255));
        return;
    }
    uint x = params.crop_x + gid.x, y = params.crop_y + gid.y;
    for (uint channel = 0; channel < params.channels; ++channel) {
        int value; if (!jxr_formatted_integer(samples, planes, channel, x, y, params, value, status)) return;
        output[jxr_output_index(surface, gid.x, gid.y, channel, 1u)] = uchar(clamp(value,0,255));
    }
}

#define JXR_INTEGER_STORE(NAME, TYPE, BYTES, MINIMUM, MAXIMUM) \
kernel void NAME(device const int *samples [[buffer(0)]], device const JxrSamplePlaneAbi *planes [[buffer(1)]], \
                 device const JxrSurfacePlaneAbi *surfaces [[buffer(2)]], device uchar *output [[buffer(3)]], \
                 device atomic_uint *status [[buffer(4)]], constant JxrOutputAbi &params [[buffer(5)]], \
                 uint2 gid [[thread_position_in_grid]]) { \
    JxrSurfacePlaneAbi surface = surfaces[params.output_plane]; \
    if (gid.x >= surface.width || gid.y >= surface.height || jxr_failed(status)) return; \
    uint x = params.crop_x + gid.x, y = params.crop_y + gid.y; \
    uint channels = surface.channels; \
    uint source_plane = params.output_plane < 3u ? params.output_plane : params.alpha_plane; \
    if (params.output_plane_count > 1u) { \
        x = params.crop_x / (params.output_plane == 1u || params.output_plane == 2u ? 2u : 1u) + gid.x; \
        uint yd = (params.chroma_sampling == 1u && (params.output_plane == 1u || params.output_plane == 2u)) ? 2u : 1u; \
        y = params.crop_y / yd + gid.y; channels = 1u; \
    } \
    for (uint channel = 0; channel < channels; ++channel) { \
        int value; \
        if (params.output_plane_count > 1u) { \
            bool alpha = params.output_plane >= 3u; \
            if (!jxr_scale(jxr_read_plane(samples, planes[source_plane], x, y), params.output_plane, alpha, params, value, status)) return; \
        } else if (!jxr_formatted_integer(samples, planes, channel, x, y, params, value, status)) return; \
        device TYPE *destination = reinterpret_cast<device TYPE *>(output + jxr_output_index(surface, gid.x, gid.y, channel, BYTES)); \
        *destination = TYPE(clamp(value, MINIMUM, MAXIMUM)); \
    } \
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
    uint x=params.crop_x+gid.x,y=params.crop_y+gid.y;
    uint alpha_bits = params.alpha_plane == UINT_MAX ? 0u : jxr_f16_bits(jxr_read_plane(samples,planes[params.alpha_plane],x,y),true,params,status);
    for(uint c=0;c<params.channels;++c){ if((params.channel_layout==12u||params.channel_layout==13u)&&c==3u){*reinterpret_cast<device ushort *>(output+jxr_output_index(surface,gid.x,gid.y,c,2u))=0;continue;} int sample; bool alpha; if(!jxr_component(samples,planes,c,x,y,params,sample,alpha,status))return;
        uint bits=jxr_f16_bits(sample,alpha,params,status);
        if(params.premultiply_alpha!=0u&&!alpha){ uint sign=bits&0x8000u; bits=sign|jxr_unsigned_premultiply(bits&0x7fffu,(alpha_bits&0x8000u)==0u?alpha_bits&0x7fffu:0u,0x7fffu); }
        *reinterpret_cast<device ushort *>(output+jxr_output_index(surface,gid.x,gid.y,c,2u))=ushort(bits); }
}

kernel void jxr_output_f32(device const int *samples [[buffer(0)]], device const JxrSamplePlaneAbi *planes [[buffer(1)]],
                           device const JxrSurfacePlaneAbi *surfaces [[buffer(2)]], device uchar *output [[buffer(3)]],
                           device atomic_uint *status [[buffer(4)]], constant JxrOutputAbi &params [[buffer(5)]],
                           uint2 gid [[thread_position_in_grid]]) {
    JxrSurfacePlaneAbi surface=surfaces[0]; if(gid.x>=surface.width||gid.y>=surface.height||jxr_failed(status))return;
    uint x=params.crop_x+gid.x,y=params.crop_y+gid.y; float alpha_value=1.0f;
    if(params.alpha_plane!=UINT_MAX) alpha_value=clamp(as_type<float>(jxr_f32_bits(jxr_read_plane(samples,planes[params.alpha_plane],x,y),true,params,status)),0.0f,1.0f);
    for(uint c=0;c<params.channels;++c){if((params.channel_layout==12u||params.channel_layout==13u)&&c==3u){*reinterpret_cast<device float *>(output+jxr_output_index(surface,gid.x,gid.y,c,4u))=0.0f;continue;}int sample;bool alpha;if(!jxr_component(samples,planes,c,x,y,params,sample,alpha,status))return;
        uint bits=jxr_f32_bits(sample,alpha,params,status);float value=as_type<float>(bits);if(params.premultiply_alpha!=0u&&!alpha)value*=alpha_value;
        *reinterpret_cast<device float *>(output+jxr_output_index(surface,gid.x,gid.y,c,4u))=value;}
}

kernel void jxr_output_bits(device const int *samples [[buffer(0)]], device const JxrSamplePlaneAbi *planes [[buffer(1)]],
                            device const JxrSurfacePlaneAbi *surfaces [[buffer(2)]], device uchar *output [[buffer(3)]],
                            device atomic_uint *status [[buffer(4)]], constant JxrOutputAbi &params [[buffer(5)]],
                            uint2 gid [[thread_position_in_grid]]) {
    JxrSurfacePlaneAbi surface=surfaces[0]; uint row_bytes=(surface.width+7u)/8u;
    if(gid.x>=row_bytes||gid.y>=surface.height||jxr_failed(status))return; uchar byte=0;
    for(uint bit=0;bit<8u;++bit){uint px=gid.x*8u+bit;if(px>=surface.width)break;int value;
        if(!jxr_scale(jxr_read_plane(samples,planes[0],params.crop_x+px,params.crop_y+gid.y),0u,false,params,value,status))return;
        uint packed=uint(clamp(value,0,1));if(params.bit_black!=0u)packed=1u-packed;byte|=uchar(packed<<(7u-bit));}
    output[surface.byte_offset+gid.y*surface.row_stride_bytes+gid.x]=byte;
}

kernel void jxr_output_packed16(device const int *samples [[buffer(0)]], device const JxrSamplePlaneAbi *planes [[buffer(1)]],
                                device const JxrSurfacePlaneAbi *surfaces [[buffer(2)]], device uchar *output [[buffer(3)]],
                                device atomic_uint *status [[buffer(4)]], constant JxrOutputAbi &params [[buffer(5)]],
                                uint2 gid [[thread_position_in_grid]]) {
    JxrSurfacePlaneAbi surface=surfaces[0];if(gid.x>=surface.width||gid.y>=surface.height||jxr_failed(status))return;
    int values[4];if(!jxr_converted(samples,planes,params.crop_x+gid.x,params.crop_y+gid.y,params,values,status))return;uint packed=0u;
    for(uint c=0;c<3u;++c){int value;if(!jxr_scale(values[c],c,false,params,value,status))return;uint maximum=params.bit_depth==10u&&c==1u?63u:31u;
        uint shift=params.bit_depth==10u?(c==0u?11u:(c==1u?5u:0u)):(2u-c)*5u;packed|=uint(clamp(value,0,int(maximum)))<<shift;}
    *reinterpret_cast<device ushort *>(output+surface.byte_offset+gid.y*surface.row_stride_bytes+gid.x*2u)=ushort(packed);
}

kernel void jxr_output_packed32(device const int *samples [[buffer(0)]], device const JxrSamplePlaneAbi *planes [[buffer(1)]],
                                device const JxrSurfacePlaneAbi *surfaces [[buffer(2)]], device uchar *output [[buffer(3)]],
                                device atomic_uint *status [[buffer(4)]], constant JxrOutputAbi &params [[buffer(5)]],
                                uint2 gid [[thread_position_in_grid]]) {
    JxrSurfacePlaneAbi surface=surfaces[0];if(gid.x>=surface.width||gid.y>=surface.height||jxr_failed(status))return;int values[4];
    if(!jxr_converted(samples,planes,params.crop_x+gid.x,params.crop_y+gid.y,params,values,status))return;uint packed=0u;
    if(params.output_color==6u){int scaled[3];uint exponent=0u;uint mantissa[3];uint local_exp[3];
        for(uint c=0;c<3u;++c){if(!jxr_scale(values[c],c,false,params,scaled[c],status))return;if(scaled[c]<=0){mantissa[c]=0;local_exp[c]=0;}
            else if((scaled[c]>>7)>1){mantissa[c]=uint((scaled[c]&127)+128);local_exp[c]=uint(scaled[c]>>7);}else{mantissa[c]=uint(scaled[c]);local_exp[c]=1u;}exponent=max(exponent,local_exp[c]);}
        for(uint c=0;c<3u;++c)if(exponent>local_exp[c]){uint d=exponent-local_exp[c];mantissa[c]=d>=31u?0u:uint((2u*mantissa[c]+1u)>>(d+1u));}
        packed=(min(mantissa[0],255u))|(min(mantissa[1],255u)<<8u)|(min(mantissa[2],255u)<<16u)|(min(exponent,255u)<<24u);
    }else{for(uint c=0;c<3u;++c){int value;if(!jxr_scale(values[c],c,false,params,value,status))return;packed|=uint(clamp(value,0,1023))<<((2u-c)*10u);}}
    *reinterpret_cast<device uint *>(output+surface.byte_offset+gid.y*surface.row_stride_bytes+gid.x*4u)=packed;
}
