//! T.832 image and image-plane header parsing.

mod fields;
mod image;
mod plane;
mod types;

use crate::{NativeError, bit_reader::BitReader};

use image::parse_image_header;
use plane::parse_plane_header;
pub use types::{CodestreamHeader, HeaderFlags, ImagePlaneHeader, ParsedHeaders, QuantizerSet};

/// Parse the T.832 image and image-plane headers.
pub fn parse_codestream_headers(bytes: &[u8]) -> Result<ParsedHeaders, NativeError> {
    let mut reader = BitReader::new(bytes);
    let image = parse_image_header(&mut reader)?;
    let primary = parse_plane_header(&mut reader, image.output_bit_depth)?;
    let alpha = image
        .flags
        .alpha_plane()
        .then(|| parse_plane_header(&mut reader, image.output_bit_depth))
        .transpose()?;
    let bytes_consumed = reader.bit_position().div_ceil(8);
    Ok(ParsedHeaders {
        image,
        primary,
        alpha,
        bytes_consumed,
    })
}
