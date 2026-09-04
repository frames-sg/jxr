//! Container-neutral codestream discovery and header preparation.

use core::ops::Range;

use crate::{
    AnnexAImage, CodestreamDirectory, NativeError, ParsedHeaders, parse_annex_a,
    parse_codestream_headers,
};

/// Parsed JPEG XR codestream and optional Annex-A metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedCodestream {
    /// Byte range containing the primary T.832 codestream.
    pub codestream_range: Range<usize>,
    /// Parsed T.832 image and plane headers.
    pub headers: ParsedHeaders,
    /// Tile index, profile declarations, and first tile byte.
    pub directory: CodestreamDirectory,
    /// Parsed separate alpha codestream headers, when Annex A stores one.
    pub separate_alpha_headers: Option<ParsedHeaders>,
    /// Separate alpha tile directory, when present.
    pub separate_alpha_directory: Option<CodestreamDirectory>,
    /// Annex-A metadata when the source is an Annex-A file.
    pub annex_a: Option<AnnexAImage>,
}

/// Detect an Annex-A file or raw T.832 codestream and parse its headers.
pub fn parse_codestream(bytes: &[u8]) -> Result<ParsedCodestream, NativeError> {
    if bytes.starts_with(b"WMPHOTO\0") {
        let headers = parse_codestream_headers(bytes)?;
        let directory = CodestreamDirectory::parse(bytes, &headers)?;
        return Ok(ParsedCodestream {
            codestream_range: 0..bytes.len(),
            headers,
            directory,
            separate_alpha_headers: None,
            separate_alpha_directory: None,
            annex_a: None,
        });
    }
    let annex_a = parse_annex_a(bytes)?;
    let range = annex_a.codestream_range.clone();
    let codestream = &bytes[range.clone()];
    let headers = parse_codestream_headers(codestream)?;
    let directory = CodestreamDirectory::parse(codestream, &headers)?;
    if headers.image.width != annex_a.width || headers.image.height != annex_a.height {
        return Err(NativeError::ReservedValue {
            field: "Annex-A/codestream dimension mismatch",
            value: 1,
        });
    }
    let (separate_alpha_headers, separate_alpha_directory) = parse_separate_alpha(bytes, &annex_a)?;
    let parsed = ParsedCodestream {
        codestream_range: range,
        headers,
        directory,
        separate_alpha_headers,
        separate_alpha_directory,
        annex_a: Some(annex_a),
    };
    crate::pixel_format::validate_annex_a_pixel_format(&parsed)?;
    Ok(parsed)
}

fn parse_separate_alpha(
    bytes: &[u8],
    annex_a: &AnnexAImage,
) -> Result<(Option<ParsedHeaders>, Option<CodestreamDirectory>), NativeError> {
    let Some(range) = annex_a.alpha_range.clone() else {
        return Ok((None, None));
    };
    let codestream = &bytes[range];
    let headers = parse_codestream_headers(codestream)?;
    if headers.image.width != annex_a.width || headers.image.height != annex_a.height {
        return Err(NativeError::InvalidSyntax {
            field: "separate alpha dimensions",
        });
    }
    let directory = CodestreamDirectory::parse(codestream, &headers)?;
    Ok((Some(headers), Some(directory)))
}
