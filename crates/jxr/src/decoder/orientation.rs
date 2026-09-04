use jxr_core::{
    DecodedImage, DecodedSamples, JxrError, JxrErrorKind, Orientation, PlaneDescriptor, Rect,
};

pub(super) fn apply_orientation(
    mut image: DecodedImage,
    orientation: Orientation,
    is_region: bool,
) -> Result<DecodedImage, JxrError> {
    if orientation == Orientation::Identity {
        return Ok(image);
    }
    if is_region {
        return Err(JxrError::new(
            JxrErrorKind::Unsupported,
            "oriented decode of a source-space region",
        ));
    }
    let [plane] = image.planes.as_slice() else {
        return Err(JxrError::new(
            JxrErrorKind::Unsupported,
            "oriented decode of subsampled planar output",
        ));
    };
    let plane = *plane;
    if let DecodedSamples::BitPacked(values) = &image.samples {
        let (samples, output_plane) = orient_bitpacked(values, plane, orientation)?;
        image.samples = DecodedSamples::BitPacked(samples);
        image.decoded_region = Rect {
            x: 0,
            y: 0,
            w: output_plane.width,
            h: output_plane.height,
        };
        image.planes = vec![output_plane];
        image.validate_layout()?;
        return Ok(image);
    }
    let width = usize::try_from(plane.width).map_err(|_| JxrError::arithmetic("oriented width"))?;
    let height =
        usize::try_from(plane.height).map_err(|_| JxrError::arithmetic("oriented height"))?;
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| JxrError::arithmetic("oriented pixel count"))?;
    let sample_count = image.samples.sample_count();
    let elements_per_pixel = sample_count
        .checked_div(pixels)
        .filter(|&elements| elements != 0 && elements.saturating_mul(pixels) == sample_count)
        .ok_or_else(|| JxrError::new(JxrErrorKind::InternalInvariant, "oriented sample extent"))?;
    macro_rules! orient {
        ($values:expr) => {
            *$values = orient_values($values, width, height, elements_per_pixel, orientation)?
        };
    }
    match &mut image.samples {
        DecodedSamples::BitPacked(_) => unreachable!("bit-packed output was rejected"),
        DecodedSamples::U8(values) => orient!(values),
        DecodedSamples::U16(values)
        | DecodedSamples::F16(values)
        | DecodedSamples::Rgb555(values)
        | DecodedSamples::Rgb565(values) => orient!(values),
        DecodedSamples::I16(values) => orient!(values),
        DecodedSamples::I32(values) => orient!(values),
        DecodedSamples::F32(values) => orient!(values),
        DecodedSamples::Rgb101010(values) | DecodedSamples::Rgbe(values) => orient!(values),
    }
    let (output_width, output_height) = if swaps_axes(orientation) {
        (plane.height, plane.width)
    } else {
        (plane.width, plane.height)
    };
    image.decoded_region = Rect {
        x: 0,
        y: 0,
        w: output_width,
        h: output_height,
    };
    image.planes = vec![PlaneDescriptor {
        byte_offset: 0,
        row_stride_bytes: image.format.row_bytes(output_width)?,
        width: output_width,
        height: output_height,
        channels: plane.channels,
    }];
    image.validate_layout()?;
    Ok(image)
}

fn orient_bitpacked(
    source: &[u8],
    plane: PlaneDescriptor,
    orientation: Orientation,
) -> Result<(Vec<u8>, PlaneDescriptor), JxrError> {
    if plane.channels != 1 {
        return Err(JxrError::new(
            JxrErrorKind::Unsupported,
            "oriented multi-channel bit-packed output",
        ));
    }
    let width = usize::try_from(plane.width).map_err(|_| JxrError::arithmetic("oriented width"))?;
    let height =
        usize::try_from(plane.height).map_err(|_| JxrError::arithmetic("oriented height"))?;
    let (output_width, output_height) = if swaps_axes(orientation) {
        (height, width)
    } else {
        (width, height)
    };
    let output_stride = output_width
        .checked_add(7)
        .map(|bits| bits / 8)
        .ok_or_else(|| JxrError::arithmetic("oriented bit-packed row"))?;
    let output_length = output_stride
        .checked_mul(output_height)
        .ok_or_else(|| JxrError::arithmetic("oriented bit-packed output"))?;
    let mut output = vec![0_u8; output_length];
    for destination_y in 0..output_height {
        for destination_x in 0..output_width {
            let (source_x, source_y) =
                source_coordinates(orientation, width, height, destination_x, destination_y);
            let source_byte = plane
                .byte_offset
                .checked_add(
                    source_y
                        .checked_mul(plane.row_stride_bytes)
                        .ok_or_else(|| JxrError::arithmetic("oriented bit source row"))?,
                )
                .and_then(|row| row.checked_add(source_x / 8))
                .ok_or_else(|| JxrError::arithmetic("oriented bit source"))?;
            let value = source.get(source_byte).ok_or_else(|| {
                JxrError::new(JxrErrorKind::InternalInvariant, "oriented bit source")
            })? & (0x80 >> (source_x % 8));
            if value != 0 {
                let destination_byte = destination_y * output_stride + destination_x / 8;
                output[destination_byte] |= 0x80 >> (destination_x % 8);
            }
        }
    }
    let output_width =
        u32::try_from(output_width).map_err(|_| JxrError::arithmetic("oriented output width"))?;
    let output_height =
        u32::try_from(output_height).map_err(|_| JxrError::arithmetic("oriented output height"))?;
    Ok((
        output,
        PlaneDescriptor {
            byte_offset: 0,
            row_stride_bytes: output_stride,
            width: output_width,
            height: output_height,
            channels: 1,
        },
    ))
}

fn orient_values<T: Copy>(
    source: &[T],
    width: usize,
    height: usize,
    elements_per_pixel: usize,
    orientation: Orientation,
) -> Result<Vec<T>, JxrError> {
    let &first = source
        .first()
        .ok_or_else(|| JxrError::new(JxrErrorKind::InternalInvariant, "empty oriented output"))?;
    let (output_width, output_height) = if swaps_axes(orientation) {
        (height, width)
    } else {
        (width, height)
    };
    let mut output = vec![first; source.len()];
    for destination_y in 0..output_height {
        for destination_x in 0..output_width {
            let (source_x, source_y) =
                source_coordinates(orientation, width, height, destination_x, destination_y);
            let source_start = (source_y * width + source_x) * elements_per_pixel;
            let destination_start =
                (destination_y * output_width + destination_x) * elements_per_pixel;
            output[destination_start..destination_start + elements_per_pixel]
                .copy_from_slice(&source[source_start..source_start + elements_per_pixel]);
        }
    }
    Ok(output)
}

const fn swaps_axes(orientation: Orientation) -> bool {
    matches!(
        orientation,
        Orientation::Transpose
            | Orientation::Rotate90
            | Orientation::Transverse
            | Orientation::Rotate270
    )
}

const fn source_coordinates(
    orientation: Orientation,
    width: usize,
    height: usize,
    destination_x: usize,
    destination_y: usize,
) -> (usize, usize) {
    match orientation {
        Orientation::Identity => (destination_x, destination_y),
        Orientation::MirrorHorizontal => (width - 1 - destination_x, destination_y),
        Orientation::Rotate180 => (width - 1 - destination_x, height - 1 - destination_y),
        Orientation::MirrorVertical => (destination_x, height - 1 - destination_y),
        Orientation::Transpose => (destination_y, destination_x),
        Orientation::Rotate90 => (destination_y, height - 1 - destination_x),
        Orientation::Transverse => (width - 1 - destination_y, height - 1 - destination_x),
        Orientation::Rotate270 => (width - 1 - destination_y, destination_x),
    }
}

#[cfg(test)]
mod tests {
    use jxr_core::{
        AlphaMode, BackendRequest, BandPresence, BitstreamMode, ColorFormat, DecodeReport,
        DecodedImage, DecodedSamples, ImageInfo, ImageMetadata, Orientation, OverlapMode,
        PixelFormat, PlaneDescriptor, PlaneInfo, Rect, SampleFormat, TileGrid,
    };

    use super::{apply_orientation, orient_values};

    fn bitpacked_image() -> DecodedImage {
        DecodedImage {
            info: ImageInfo {
                width: 3,
                height: 2,
                profile: None,
                level: None,
                primary: PlaneInfo {
                    color_format: ColorFormat::Luma,
                    sample_format: SampleFormat::Bit1,
                    bands: BandPresence::DcOnly,
                    bitstream_mode: BitstreamMode::Spatial,
                    overlap: OverlapMode::None,
                    short_header: false,
                    long_word: false,
                    scaled: false,
                    chroma_centering: [0, 0],
                    shift_bits: 0,
                    mantissa_length: 0,
                    exponent_bias: 0,
                    width: 3,
                    height: 2,
                },
                alpha_mode: AlphaMode::None,
                premultiplied_alpha: false,
                alpha: None,
                tiles: TileGrid {
                    column_widths: vec![1],
                    row_heights: vec![1],
                    hard_tiles: false,
                },
                metadata: ImageMetadata::default(),
            },
            decoded_region: Rect {
                x: 0,
                y: 0,
                w: 3,
                h: 2,
            },
            format: PixelFormat::BitPacked(jxr_core::ChannelLayout::Luma),
            planes: vec![PlaneDescriptor {
                byte_offset: 0,
                row_stride_bytes: 1,
                width: 3,
                height: 2,
                channels: 1,
            }],
            samples: DecodedSamples::BitPacked(vec![0b1010_0000, 0b0100_0000]),
            report: DecodeReport::cpu(BackendRequest::Cpu),
        }
    }

    fn unpack_bits(image: &DecodedImage) -> Vec<u8> {
        let DecodedSamples::BitPacked(bytes) = &image.samples else {
            panic!("expected bit-packed samples");
        };
        let plane = image.planes[0];
        let mut bits = Vec::new();
        for y in 0..usize::try_from(plane.height).unwrap() {
            for x in 0..usize::try_from(plane.width).unwrap() {
                let byte = bytes[plane.byte_offset + y * plane.row_stride_bytes + x / 8];
                bits.push(u8::from(byte & (0x80 >> (x % 8)) != 0));
            }
        }
        bits
    }

    #[test]
    fn maps_rectangular_pixels_without_channel_reordering() {
        let rgb = [1_u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        assert_eq!(
            orient_values(&rgb, 2, 2, 3, Orientation::Rotate90).unwrap(),
            vec![7, 8, 9, 1, 2, 3, 10, 11, 12, 4, 5, 6]
        );
        assert_eq!(
            orient_values(&[1_u8, 2, 3, 4, 5, 6], 3, 2, 1, Orientation::Rotate270).unwrap(),
            vec![3, 6, 2, 5, 1, 4]
        );
    }

    #[test]
    fn rotates_bitpacked_rows_and_rebuilds_the_plane_layout() {
        let image = apply_orientation(bitpacked_image(), Orientation::Rotate90, false).unwrap();

        assert_eq!(
            image.decoded_region,
            Rect {
                x: 0,
                y: 0,
                w: 2,
                h: 3,
            }
        );
        assert_eq!(
            image.planes,
            vec![PlaneDescriptor {
                byte_offset: 0,
                row_stride_bytes: 1,
                width: 2,
                height: 3,
                channels: 1,
            }]
        );
        assert_eq!(
            image.samples,
            DecodedSamples::BitPacked(vec![0b0100_0000, 0b1000_0000, 0b0100_0000])
        );
        image.validate_layout().unwrap();
    }

    #[test]
    fn every_bitpacked_orientation_matches_the_scalar_pixel_mapping() {
        let source = [1_u8, 0, 1, 0, 1, 0];
        for orientation in [
            Orientation::MirrorHorizontal,
            Orientation::Rotate180,
            Orientation::MirrorVertical,
            Orientation::Transpose,
            Orientation::Rotate90,
            Orientation::Transverse,
            Orientation::Rotate270,
        ] {
            let expected = orient_values(&source, 3, 2, 1, orientation).unwrap();
            let image = apply_orientation(bitpacked_image(), orientation, false).unwrap();
            assert_eq!(unpack_bits(&image), expected, "{orientation:?}");
        }
    }
}
