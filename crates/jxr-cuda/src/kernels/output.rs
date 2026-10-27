// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn read_plane(samples: *const i32, plane: JxrSamplePlaneAbi, x: u32, y: u32) -> i32 {
    device_load(
        samples,
        plane.sample_offset + (y - plane.origin_y) * plane.width + x - plane.origin_x,
    )
}

fn clamped_plane(samples: *const i32, plane: JxrSamplePlaneAbi, x: i32, y: i32) -> i32 {
    let local_x = (x - plane.origin_x as i32).clamp(0, plane.width as i32 - 1);
    let local_y = (y - plane.origin_y as i32).clamp(0, plane.height as i32 - 1);
    device_load(
        samples,
        plane.sample_offset + local_y as u32 * plane.width + local_x as u32,
    )
}

fn upsample(previous: i32, current: i32, next: i32, centering: u32) -> Result<[i32; 2], MathError> {
    let weights = match centering {
        0 => [4, 4, 0, 8],
        1 => [5, 3, 1, 7],
        2 => [6, 2, 2, 6],
        3 => [7, 1, 3, 5],
        _ => [8, 0, 4, 4],
    };
    Ok([
        math_try!(checked_i64_to_i32(
            (previous as i64 * weights[2] + current as i64 * weights[3] + 4) >> 3
        )),
        math_try!(checked_i64_to_i32(
            (current as i64 * weights[0] + next as i64 * weights[1] + 4) >> 3
        )),
    ])
}

fn chroma(
    samples: *const i32,
    plane: JxrSamplePlaneAbi,
    x: u32,
    y: u32,
    p: JxrOutputAbi,
) -> Result<i32, MathError> {
    if p.chroma_sampling == 3 {
        return Ok(read_plane(samples, plane, x, y));
    }
    let cx = (x >> 1) as i32;
    if p.chroma_sampling == 2 {
        let pair = math_try!(upsample(
            clamped_plane(samples, plane, cx - 1, y as i32),
            clamped_plane(samples, plane, cx, y as i32),
            clamped_plane(samples, plane, cx + 1, y as i32),
            p.chroma_centering_x,
        ));
        return Ok(pair[(x & 1) as usize]);
    }
    let cy = (y >> 1) as i32;
    let mut vertical = [0; 3];
    for column in -1..=1 {
        let pair = math_try!(upsample(
            clamped_plane(samples, plane, cx + column, cy - 1),
            clamped_plane(samples, plane, cx + column, cy),
            clamped_plane(samples, plane, cx + column, cy + 1),
            p.chroma_centering_y,
        ));
        vertical[(column + 1) as usize] = pair[(y & 1) as usize];
    }
    let horizontal = math_try!(upsample(
        vertical[0],
        vertical[1],
        vertical[2],
        p.chroma_centering_x
    ));
    Ok(horizontal[(x & 1) as usize])
}

fn converted(
    samples: *const i32,
    planes: *const JxrSamplePlaneAbi,
    x: u32,
    y: u32,
    p: JxrOutputAbi,
) -> Result<[i32; 4], MathError> {
    let mut input = [0; 4];
    input[0] = read_plane(samples, device_load(planes, 0), x, y);
    if p.component_count > 1 {
        if p.internal_color == 1 {
            input[1] = math_try!(chroma(samples, device_load(planes, 1), x, y, p));
            input[2] = math_try!(chroma(samples, device_load(planes, 2), x, y, p));
        } else {
            for i in 1..p.component_count.min(4) {
                input[i as usize] = read_plane(samples, device_load(planes, i), x, y);
            }
        }
    }
    let mut values = input;
    if p.internal_color == 0 && p.output_color == 2 {
        values[1] = input[0];
        values[2] = input[0];
    } else if p.internal_color == 1 && (p.output_color == 2 || p.output_color == 6) {
        let temporary = math_try!(checked_sub(0, input[1]));
        let green = math_try!(checked_sub(input[0], temporary >> 1));
        let red = math_try!(checked_sub(
            math_try!(checked_add(temporary, green)),
            (input[2] >> 1) + (input[2] & 1),
        ));
        values[0] = red;
        values[1] = green;
        values[2] = math_try!(checked_add(input[2], red));
        if p.bit_depth >= 8 && p.bit_depth <= 10 && p.red_blue_not_swapped == 0 {
            let red = values[0];
            values[0] = values[2];
            values[2] = red;
        }
    } else if p.internal_color == 5 && p.output_color == 3 {
        let black = math_try!(checked_add(input[3], input[0] >> 1));
        let magenta = math_try!(checked_sub(
            math_try!(checked_sub(black, input[0])),
            input[1] >> 1
        ));
        let cyan = math_try!(checked_add(
            math_try!(checked_add(input[1], magenta)),
            input[2] >> 1
        ));
        values = [cyan, magenta, math_try!(checked_sub(cyan, input[2])), black];
    } else if p.internal_color == 5 && p.output_color == 4 {
        values = [input[1], input[2], input[3], input[0]];
    }
    Ok(values)
}

fn scale(sample: i32, component: u32, alpha: bool, p: JxrOutputAbi) -> Result<i32, MathError> {
    let depth = p.bit_depth;
    let scaled = if alpha { p.alpha_scaled } else { p.scaled };
    let shift_bits = if alpha {
        p.alpha_shift_bits
    } else {
        p.shift_bits
    };
    let mut bias = match depth {
        1 => 128,
        2 | 9 => 512,
        3 => 32768 >> shift_bits,
        8 => 16,
        10 => 32,
        _ => 0,
    };
    if !alpha && p.output_color == 3 {
        bias = if component < 3 {
            bias >> 1
        } else {
            -(bias >> 1)
        };
    }
    if !alpha && p.output_color == 6 {
        bias = 0;
    }
    let scale = scaled * 3;
    let shifted_bias = math_try!(checked_i64_to_i32(bias as i64 * (1i64 << scale)));
    let biased = math_try!(checked_add(sample, shifted_bias));
    let rounding = if scaled == 0 {
        0
    } else if depth == 0 || depth == 3 {
        4
    } else {
        3
    };
    let rounded = math_try!(checked_add(biased, rounding));
    let downshift = scale + if depth == 10 && component != 1 { 1 } else { 0 };
    let downscaled = rounded >> downshift;
    if depth == 3 || depth == 4 || depth == 6 {
        checked_i64_to_i32(downscaled as i64 * (1i64 << shift_bits))
    } else {
        Ok(downscaled)
    }
}

fn f16_bits(sample: i32, alpha: bool, p: JxrOutputAbi) -> Result<u16, MathError> {
    let scaled = math_try!(scale(sample, 0, alpha, p));
    let sign = if scaled < 0 { 0x8000 } else { 0 };
    Ok(sign | (scaled as i64).unsigned_abs().min(32767) as u16)
}

fn f32_bits(sample: i32, alpha: bool, p: JxrOutputAbi) -> Result<u32, MathError> {
    let scaled = math_try!(scale(sample, 0, alpha, p));
    let length = if alpha {
        p.alpha_mantissa_length
    } else {
        p.mantissa_length
    };
    let bias = if alpha {
        p.alpha_exponent_bias_bits
    } else {
        p.exponent_bias_bits
    } as i32;
    let sign = if scaled < 0 { 0x80000000 } else { 0 };
    let magnitude = (scaled as i64).unsigned_abs();
    let implicit = 1u64 << length;
    let mut exponent = (magnitude >> length) as i64;
    let mut mantissa = (magnitude & (implicit - 1)) | implicit;
    if exponent == 0 {
        mantissa ^= implicit;
        exponent = 1;
    }
    exponent = exponent - bias as i64 + 127;
    while mantissa < implicit && exponent > 1 && mantissa > 0 {
        exponent -= 1;
        mantissa <<= 1;
    }
    if mantissa < implicit {
        exponent = 0;
    } else {
        mantissa ^= implicit;
    }
    mantissa <<= 23 - length;
    if exponent < 0 || exponent > 255 || mantissa > 0x7fffff {
        return Err(1);
    }
    Ok(sign | ((exponent as u32) << 23) | mantissa as u32)
}

fn is_alpha(channel: u32, p: JxrOutputAbi) -> bool {
    p.alpha_plane != u32::MAX && channel == p.channels - 1
}

fn component(
    samples: *const i32,
    planes: *const JxrSamplePlaneAbi,
    channel: u32,
    x: u32,
    y: u32,
    p: JxrOutputAbi,
) -> Result<i32, MathError> {
    if is_alpha(channel, p) {
        return Ok(read_plane(
            samples,
            device_load(planes, p.alpha_plane),
            x,
            y,
        ));
    }
    if p.output_color == 7 {
        return Ok(read_plane(samples, device_load(planes, channel), x, y));
    }
    if (p.channel_layout == 12 || p.channel_layout == 13) && channel == 3 {
        return Ok(0);
    }
    let values = math_try!(converted(samples, planes, x, y, p));
    let source = if (p.channel_layout == 6 || p.channel_layout == 7 || p.channel_layout == 13)
        && channel < 3
    {
        2 - channel
    } else {
        channel
    };
    Ok(values[source as usize])
}

fn formatted_integer(
    samples: *const i32,
    planes: *const JxrSamplePlaneAbi,
    channel: u32,
    x: u32,
    y: u32,
    p: JxrOutputAbi,
) -> Result<i32, MathError> {
    if (p.channel_layout == 12 || p.channel_layout == 13) && channel == 3 {
        return Ok(0);
    }
    let alpha = is_alpha(channel, p);
    let sample = math_try!(component(samples, planes, channel, x, y, p));
    let mut value = math_try!(scale(sample, channel, alpha, p));
    if p.premultiply_alpha != 0 && !alpha && p.alpha_plane != u32::MAX {
        let alpha_sample = read_plane(samples, device_load(planes, p.alpha_plane), x, y);
        let alpha_value = math_try!(scale(alpha_sample, 0, true, p));
        if p.bit_depth == 1 {
            value = math_try!(premultiply_unsigned(
                value.clamp(0, 255) as u32,
                alpha_value.clamp(0, 255) as u32,
                255,
            )) as i32;
        } else if p.bit_depth == 2 || p.bit_depth == 3 {
            value = math_try!(premultiply_unsigned(
                value.clamp(0, 65535) as u32,
                alpha_value.clamp(0, 65535) as u32,
                65535,
            )) as i32;
        } else if p.bit_depth == 4 {
            value = math_try!(premultiply_signed(value, alpha_value, 32767));
        } else if p.bit_depth == 6 {
            value = math_try!(premultiply_signed(value, alpha_value, i32::MAX));
        }
    }
    Ok(value)
}

fn integer_value(
    samples: *const i32,
    planes: *const JxrSamplePlaneAbi,
    channel: u32,
    x_out: u32,
    y_out: u32,
    p: JxrOutputAbi,
) -> Result<i32, MathError> {
    if p.output_plane_count > 1 {
        let chroma = p.output_plane == 1 || p.output_plane == 2;
        let source = if p.output_plane < 3 {
            p.output_plane
        } else {
            p.alpha_plane
        };
        let x = p.crop_x / if chroma { 2 } else { 1 } + x_out;
        let y = p.crop_y
            / if p.chroma_sampling == 1 && chroma {
                2
            } else {
                1
            }
            + y_out;
        scale(
            read_plane(samples, device_load(planes, source), x, y),
            p.output_plane,
            p.output_plane >= 3,
            p,
        )
    } else {
        formatted_integer(
            samples,
            planes,
            channel,
            p.crop_x + x_out,
            p.crop_y + y_out,
            p,
        )
    }
}

fn output_index(surface: JxrSurfacePlaneAbi, x: u32, y: u32, channel: u32, bytes: u32) -> u32 {
    surface.byte_offset + y * surface.row_stride_bytes + (x * surface.channels + channel) * bytes
}

fn store_word(output: *mut u8, index: u32, value: u32, bytes: u32) {
    for byte in 0..bytes {
        store(output, index + byte, (value >> (byte * 8)) as u8);
    }
}

pub fn write(
    samples: *const i32,
    planes: *const JxrSamplePlaneAbi,
    surface: JxrSurfacePlaneAbi,
    output: *mut u8,
    p: JxrOutputAbi,
    x_out: u32,
    y_out: u32,
    kind: u32,
) -> Result<(), MathError> {
    let x = p.crop_x + x_out;
    let y = p.crop_y + y_out;
    if kind == 0 {
        let mut byte = 0u8;
        for bit in 0..8 {
            let pixel_x = x_out * 8 + bit;
            if pixel_x >= surface.width {
                break;
            }
            let sample = read_plane(samples, device_load(planes, 0), p.crop_x + pixel_x, y);
            let mut packed = math_try!(scale(sample, 0, false, p)).clamp(0, 1) as u32;
            if p.bit_black != 0 {
                packed = 1 - packed;
            }
            byte |= (packed << (7 - bit)) as u8;
        }
        store(
            output,
            surface.byte_offset + y_out * surface.row_stride_bytes + x_out,
            byte,
        );
    } else if kind <= 4 {
        let bytes = if kind == 1 {
            1
        } else if kind == 4 {
            4
        } else {
            2
        };
        let channels = if p.output_plane_count > 1 {
            1
        } else {
            surface.channels
        };
        for channel in 0..channels {
            let value = math_try!(integer_value(samples, planes, channel, x_out, y_out, p));
            let (minimum, maximum) = match kind {
                1 => (0, 255),
                2 => (0, if p.bit_depth == 2 { 1023 } else { 65535 }),
                3 => (-32768, 32767),
                _ => (i32::MIN, i32::MAX),
            };
            store_word(
                output,
                output_index(surface, x_out, y_out, channel, bytes),
                value.clamp(minimum, maximum) as u32,
                bytes,
            );
        }
    } else if kind == 5 || kind == 6 {
        let alpha_sample = if p.alpha_plane == u32::MAX {
            0
        } else {
            read_plane(samples, device_load(planes, p.alpha_plane), x, y)
        };
        let alpha_bits = if kind == 5 && p.alpha_plane != u32::MAX {
            math_try!(f16_bits(alpha_sample, true, p))
        } else {
            0
        };
        let alpha_value = if kind == 6 && p.alpha_plane != u32::MAX {
            f32::from_bits(math_try!(f32_bits(alpha_sample, true, p))).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let bytes = if kind == 5 { 2 } else { 4 };
        for channel in 0..p.channels {
            let index = output_index(surface, x_out, y_out, channel, bytes);
            if (p.channel_layout == 12 || p.channel_layout == 13) && channel == 3 {
                store_word(output, index, 0, bytes);
                continue;
            }
            let alpha = is_alpha(channel, p);
            let sample = math_try!(component(samples, planes, channel, x, y, p));
            let bits = if kind == 5 {
                let bits = math_try!(f16_bits(sample, alpha, p));
                if p.premultiply_alpha != 0 && !alpha {
                    math_try!(premultiply_sign_magnitude_15(bits, alpha_bits)) as u32
                } else {
                    bits as u32
                }
            } else {
                let mut value = f32::from_bits(math_try!(f32_bits(sample, alpha, p)));
                if p.premultiply_alpha != 0 && !alpha {
                    value *= alpha_value;
                }
                value.to_bits()
            };
            store_word(output, index, bits, bytes);
        }
    } else {
        let values = math_try!(converted(samples, planes, x, y, p));
        let mut packed = 0;
        if kind == 8 && p.output_color == 6 {
            let mut scaled = [0; 3];
            for channel in 0..3 {
                scaled[channel as usize] =
                    math_try!(scale(values[channel as usize], channel, false, p));
            }
            packed = pack_rgbe(scaled);
        } else {
            for channel in 0..3 {
                let value = math_try!(scale(values[channel as usize], channel, false, p));
                let maximum = if kind == 8 {
                    1023
                } else if p.bit_depth == 10 && channel == 1 {
                    63
                } else {
                    31
                };
                let shift = if kind == 8 {
                    (2 - channel) * 10
                } else if p.bit_depth == 10 {
                    if channel == 0 {
                        11
                    } else if channel == 1 {
                        5
                    } else {
                        0
                    }
                } else {
                    (2 - channel) * 5
                };
                packed |= (value.clamp(0, maximum) as u32) << shift;
            }
        }
        let bytes = if kind == 7 { 2 } else { 4 };
        store_word(
            output,
            surface.byte_offset + y_out * surface.row_stride_bytes + x_out * bytes,
            packed,
            bytes,
        );
    }
    Ok(())
}

fn premultiply_unsigned(value: u32, alpha: u32, maximum: u32) -> Result<u32, MathError> {
    if maximum == 0 {
        return Err(1);
    }
    Ok(
        ((value.min(maximum) as u64 * alpha.min(maximum) as u64 + (maximum / 2) as u64)
            / maximum as u64) as u32,
    )
}

fn premultiply_signed(value: i32, alpha: i32, maximum: i32) -> Result<i32, MathError> {
    if maximum <= 0 {
        return Err(1);
    }
    let magnitude = (value as i64).unsigned_abs();
    let result = ((magnitude * alpha.clamp(0, maximum) as u64 + (maximum / 2) as u64)
        / maximum as u64) as i64;
    checked_i64_to_i32(if value < 0 { -result } else { result })
}

fn premultiply_sign_magnitude_15(value: u16, alpha: u16) -> Result<u16, MathError> {
    let magnitude = math_try!(premultiply_unsigned(
        (value & 0x7fff) as u32,
        if alpha & 0x8000 == 0 {
            (alpha & 0x7fff) as u32
        } else {
            0
        },
        0x7fff
    ));
    Ok((value & 0x8000) | magnitude as u16)
}

fn pack_rgbe(rgb: [i32; 3]) -> u32 {
    let mut exponent = 0;
    let mut mantissa = [0u32; 3];
    let mut local_exponent = [0u32; 3];
    for channel in 0..3 {
        let value = rgb[channel];
        if value <= 0 {
            mantissa[channel] = 0;
            local_exponent[channel] = 0;
        } else if value >> 7 > 1 {
            mantissa[channel] = ((value & 127) + 128) as u32;
            local_exponent[channel] = (value >> 7) as u32;
        } else {
            mantissa[channel] = value as u32;
            local_exponent[channel] = 1;
        }
        exponent = exponent.max(local_exponent[channel]);
    }
    for channel in 0..3 {
        if exponent > local_exponent[channel] {
            let difference = exponent - local_exponent[channel];
            mantissa[channel] = if difference >= 31 {
                0
            } else {
                (2 * mantissa[channel] + 1) >> (difference + 1)
            };
        }
    }
    mantissa[0].min(255)
        | (mantissa[1].min(255) << 8)
        | (mantissa[2].min(255) << 16)
        | (exponent.min(255) << 24)
}
