//! Sample patterns, and the storage layouts an image's samples are given in.

use jxr::{DecodedSamples, EncodeRequest, PixelFormat, SourceImage, SourcePlane, SourceSamples};

use super::{Placement, formats::sample_count};

pub(super) fn size(value: u32) -> usize {
    usize::try_from(value).expect("a 32-bit count fits the address space")
}

/// The low half of `value`.
pub(super) fn low32(value: u64) -> u32 {
    let [first, second, third, fourth, ..] = value.to_le_bytes();
    u32::from_le_bytes([first, second, third, fourth])
}

/// The top byte of `value`, which carries a pattern's picture.
pub(super) fn top8(value: u32) -> u8 {
    value.to_be_bytes()[0]
}

/// The top half of `value`.
pub(super) fn top16(value: u32) -> u16 {
    let [first, second, ..] = value.to_be_bytes();
    u16::from_be_bytes([first, second])
}

/// A stream of 32-bit values whose high bits carry the pattern, so that
/// every sample depth sees the same picture.
pub(super) struct Pattern<'a> {
    pub(super) kind: u8,
    pub(super) state: u64,
    pub(super) step: u32,
    /// Fuzz input left over, which the raw pattern replays as samples.
    pub(super) raw: &'a [u8],
    pub(super) position: usize,
}

impl Pattern<'_> {
    pub(super) fn noise(&mut self) -> u32 {
        low32(jxr_test_support::split_mix64(&mut self.state) >> 32)
    }

    pub(super) fn value(&mut self) -> u32 {
        match self.kind {
            // One value everywhere.
            1 => self.step,
            // A slow ramp, which prediction follows.
            2 => {
                self.state = self.state.wrapping_add(u64::from(self.step >> 6));
                low32(self.state)
            }
            // The extremes of the range.
            3 => {
                if self.noise() & 1 == 0 {
                    0
                } else {
                    u32::MAX
                }
            }
            // The input's own bytes, four to a value.
            4 if !self.raw.is_empty() => {
                let bytes = std::array::from_fn(|_| {
                    let byte = self.raw[self.position % self.raw.len()];
                    self.position += 1;
                    byte
                });
                u32::from_be_bytes(bytes)
            }
            // A power of two or its neighbour, where arithmetic changes
            // width: one everywhere,
            5 => boundary(self.step),
            // or another at every sample.
            6 => {
                let selector = self.noise();
                boundary(selector)
            }
            _ => self.noise(),
        }
    }
}

/// The power of two `selector` picks, or the value on either side of it, of
/// either sign, as a two's-complement bit pattern: 32 powers, three offsets,
/// and two signs, so selectors below 192 name each such value once.
pub(super) fn boundary(selector: u32) -> u32 {
    let power = 1_u32 << (selector % 32);
    let near = match (selector / 32) % 3 {
        0 => power,
        1 => power - 1,
        _ => power.wrapping_add(1),
    };
    if (selector / 96).is_multiple_of(2) {
        near
    } else {
        near.wrapping_neg()
    }
}

/// Samples of an image in tight layout drawn from `pattern`. With
/// `representable`, every sample is one the coded form reproduces under
/// `request`; otherwise any bit pattern the storage holds.
pub(super) fn pattern_samples(
    (format, sample_bits): (PixelFormat, Option<u8>),
    (width, height): (u32, u32),
    request: &EncodeRequest,
    representable: bool,
    pattern: &mut Pattern<'_>,
) -> DecodedSamples {
    let count = sample_count(format, width, height);
    let values = (0..count).map(|_| pattern.value());
    // Bits a 16-bit sample keeps: its declared depth, or all but those the
    // request shifts out.
    let shift = u32::from(request.shift_bits.unwrap_or(0));
    let kept16 = match sample_bits {
        _ if !representable => 0xffff,
        Some(bits) => 1_u16
            .checked_shl(u32::from(bits))
            .map_or(0xffff, |limit| limit - 1),
        None => 0xffff_u16.checked_shl(shift).unwrap_or(0),
    };
    match format {
        PixelFormat::BitPacked(_) => {
            // Bits past the last pixel of a row are not part of the image.
            let unused = 0xff_u8 << ((8 - width % 8) % 8);
            let mut bytes: Vec<u8> = values.map(top8).collect();
            if representable {
                for row in bytes.chunks_exact_mut(size(width).div_ceil(8)) {
                    if let Some(last) = row.last_mut() {
                        *last &= unused;
                    }
                }
            }
            DecodedSamples::BitPacked(bytes)
        }
        PixelFormat::U8(_) => DecodedSamples::U8(values.map(top8).collect()),
        PixelFormat::U16(_) => {
            // A ten-bit sample is the top ten bits of the pattern.
            let down = match sample_bits {
                Some(bits @ 1..=16) if representable => 16 - u32::from(bits),
                _ => 0,
            };
            DecodedSamples::U16(
                values
                    .map(|value| (top16(value) >> down) & kept16)
                    .collect(),
            )
        }
        PixelFormat::I16(_) => DecodedSamples::I16(
            values
                .map(|value| (top16(value) & kept16).cast_signed())
                .collect(),
        ),
        PixelFormat::I32(_) => {
            let shift = u32::from(
                request
                    .shift_bits
                    .unwrap_or(EncodeRequest::DEFAULT_I32_SHIFT_BITS),
            );
            DecodedSamples::I32(
                values
                    .map(|value| fixed_point(value, shift, representable))
                    .collect(),
            )
        }
        PixelFormat::F16(_) => DecodedSamples::F16(
            values
                .map(top16)
                // Negative zero has no coded sample of its own.
                .map(|bits| {
                    if representable && bits == 0x8000 {
                        0
                    } else {
                        bits
                    }
                })
                .collect(),
        ),
        PixelFormat::F32(_) => DecodedSamples::F32(
            values
                .map(|value| {
                    if representable {
                        moderate_float(value)
                    } else {
                        f32::from_bits(value)
                    }
                })
                .collect(),
        ),
        // The top bit of a 5:5:5 word belongs to no channel.
        PixelFormat::Rgb555 => DecodedSamples::Rgb555(
            values
                .map(|value| top16(value) & if representable { 0x7fff } else { 0xffff })
                .collect(),
        ),
        PixelFormat::Rgb565 => DecodedSamples::Rgb565(values.map(top16).collect()),
        // As do the top two of a 10:10:10 word.
        PixelFormat::Rgb101010 => DecodedSamples::Rgb101010(
            values
                .map(|value| if representable { value >> 2 } else { value })
                .collect(),
        ),
        PixelFormat::Rgbe => DecodedSamples::Rgbe(
            values
                .map(|value| {
                    if representable {
                        canonical_rgbe(value)
                    } else {
                        value
                    }
                })
                .collect(),
        ),
    }
}

/// A 32-bit integer sample: `value` itself, or when `representable` the
/// nearest below it whose bits under `shift`, which are not coded, are zero.
fn fixed_point(value: u32, shift: u32, representable: bool) -> i32 {
    let value = value.cast_signed();
    if !representable {
        return value;
    }
    value
        .checked_shr(shift)
        .and_then(|coded| coded.checked_shl(shift))
        .unwrap_or(0)
}

/// A finite float of either sign from `2^-10` up to `2^10`, or zero: the
/// range the default float precision codes with a full mantissa.
fn moderate_float(value: u32) -> f32 {
    let exponent = (value >> 23) & 0xff;
    if exponent % 23 == 22 {
        return 0.0;
    }
    f32::from_bits((value & 0x807f_ffff) | ((117 + exponent % 20) << 23))
}

/// An RGBE pixel a decoder can return: black is all zero, and any other
/// pixel has the leading bit of a mantissa set.
fn canonical_rgbe(value: u32) -> u32 {
    let mut bytes = value.to_le_bytes();
    if bytes[3] == 0 {
        return 0;
    }
    bytes[usize::from(bytes[0] ^ bytes[1] ^ bytes[2]) % 3] |= 0x80;
    u32::from_le_bytes(bytes)
}

/// Apply `$body` to the vector inside any sample storage and rewrap it.
macro_rules! map_storage {
    ($samples:expr, $values:ident => $body:expr) => {
        match $samples {
            DecodedSamples::BitPacked($values) => DecodedSamples::BitPacked($body),
            DecodedSamples::U8($values) => DecodedSamples::U8($body),
            DecodedSamples::U16($values) => DecodedSamples::U16($body),
            DecodedSamples::I16($values) => DecodedSamples::I16($body),
            DecodedSamples::I32($values) => DecodedSamples::I32($body),
            DecodedSamples::F16($values) => DecodedSamples::F16($body),
            DecodedSamples::F32($values) => DecodedSamples::F32($body),
            DecodedSamples::Rgb555($values) => DecodedSamples::Rgb555($body),
            DecodedSamples::Rgb565($values) => DecodedSamples::Rgb565($body),
            DecodedSamples::Rgb101010($values) => DecodedSamples::Rgb101010($body),
            DecodedSamples::Rgbe($values) => DecodedSamples::Rgbe($body),
        }
    };
}

/// One run of rows moved from tight storage: `(first tight element, row
/// length, rows)` and where the rows go.
type Run = ((usize, usize, usize), SourcePlane);

/// Storage of `total` elements holding the rows of `tight` at `runs`. The
/// elements between them repeat image samples out of place, so an encoder
/// that read one would not reproduce the image.
fn spread<T: Copy>(tight: &[T], runs: &[Run], total: usize) -> Vec<T> {
    let mut loose: Vec<T> = (0..total)
        .map(|index| tight[(index * 7 + 3) % tight.len()])
        .collect();
    for &((start, row, rows), plane) in runs {
        for y in 0..rows {
            let source = &tight[start + y * row..][..row];
            loose[plane.offset + y * plane.row_stride..][..row].copy_from_slice(source);
        }
    }
    loose
}

/// Lay the rows of a tight image out with `padding` unused elements after
/// each, and the planes of a planar image last to first with a gap before
/// each. `None` when the tight image is not a valid one to begin with.
pub(super) fn loosen(
    tight: &DecodedSamples,
    image: Option<SourceImage<'_>>,
    padding: usize,
) -> Option<(DecodedSamples, Placement)> {
    let image = image?;
    if SourceSamples::from(tight).is_empty() {
        return None;
    }
    let mut runs = Vec::new();
    let mut total = 0;
    let placement = if let Some(count) = image.plane_count() {
        let tight_planes = image.planes?;
        let mut planes = vec![SourcePlane::default(); count];
        for index in (0..count).rev() {
            let (width, height) = image.plane_dimensions(index);
            let (width, height) = (size(width), size(height));
            let plane = SourcePlane {
                offset: total + padding,
                row_stride: width + padding,
            };
            runs.push(((tight_planes[index].offset, width, height), plane));
            planes[index] = plane;
            total = plane.offset + plane.row_stride * height;
        }
        Placement::Planes(planes)
    } else {
        let row = image.row_stride;
        let plane = SourcePlane {
            offset: 0,
            row_stride: row + padding,
        };
        let height = size(image.height);
        runs.push(((0, row, height), plane));
        total = plane.row_stride * height;
        Placement::Rows(plane.row_stride)
    };
    let stored = map_storage!(tight, values => spread(values, &runs, total));
    Some((stored, placement))
}
