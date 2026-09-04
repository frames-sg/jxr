//! Bounds-checked Annex-A tag container parsing.

use core::ops::Range;

use crate::NativeError;

mod writer;

pub use writer::{AnnexAWriteOptions, write_annex_a};

const TAG_PIXEL_FORMAT: u16 = 0xBC01;
const TAG_TRANSFORMATION: u16 = 0xBC02;
const TAG_IMAGE_WIDTH: u16 = 0xBC80;
const TAG_IMAGE_HEIGHT: u16 = 0xBC81;
const TAG_HORIZONTAL_RESOLUTION: u16 = 0xBC82;
const TAG_VERTICAL_RESOLUTION: u16 = 0xBC83;
const TAG_IMAGE_OFFSET: u16 = 0xBCC0;
const TAG_IMAGE_BYTE_COUNT: u16 = 0xBCC1;
const TAG_ALPHA_OFFSET: u16 = 0xBCC2;
const TAG_ALPHA_BYTE_COUNT: u16 = 0xBCC3;
const TAG_ICC_PROFILE: u16 = 0x8773;

/// Metadata retained from the first Annex-A image file directory.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AnnexAMetadata {
    /// Optional spatial transformation value.
    pub transformation: Option<u32>,
    /// Optional horizontal and vertical DPI as exact IEEE-754 bit patterns.
    pub resolution_dpi_bits: Option<[u32; 2]>,
    /// Optional ICC profile byte range.
    pub icc_profile_range: Option<Range<usize>>,
}

/// Parsed Annex-A still image location and metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnnexAImage {
    /// Image width declared by the container.
    pub width: u32,
    /// Image height declared by the container.
    pub height: u32,
    /// Sixteen-byte pixel-format identifier.
    pub pixel_format_guid: [u8; 16],
    /// Primary JPEG XR codestream range.
    pub codestream_range: Range<usize>,
    /// Separate alpha codestream range, when present.
    pub alpha_range: Option<Range<usize>>,
    /// Additional metadata exposed without interpretation.
    pub metadata: AnnexAMetadata,
}

#[derive(Clone, Copy)]
struct DirectoryEntry {
    tag: u16,
    element_type: u16,
    count: u32,
    value: u32,
}

/// Parse the first image directory of an Annex-A JPEG XR file.
pub fn parse_annex_a(bytes: &[u8]) -> Result<AnnexAImage, NativeError> {
    validate_file_header(bytes)?;
    let ifd_offset = read_u32(bytes, 4, "FIRST_IFD_OFFSET")? as usize;
    if !ifd_offset.is_multiple_of(2) {
        return Err(NativeError::ReservedValue {
            field: "FIRST_IFD_OFFSET alignment",
            value: ifd_offset as u64,
        });
    }
    let entries = parse_directory(bytes, ifd_offset)?;
    build_image(bytes, &entries)
}

fn validate_file_header(bytes: &[u8]) -> Result<(), NativeError> {
    if bytes.len() < 8 {
        return Err(NativeError::Truncated {
            bit_position: bytes.len().saturating_mul(8),
            requested_bits: 64,
        });
    }
    if bytes[..4] != [b'I', b'I', 0xBC, 1] {
        return Err(NativeError::InvalidSignature);
    }
    Ok(())
}

fn parse_directory(bytes: &[u8], offset: usize) -> Result<Vec<DirectoryEntry>, NativeError> {
    let count = usize::from(read_u16(bytes, offset, "NUM_ENTRIES")?);
    if count == 0 {
        return Err(NativeError::ReservedValue {
            field: "NUM_ENTRIES",
            value: 0,
        });
    }
    let table_start = offset.checked_add(2).ok_or(NativeError::IntegerOverflow {
        operation: "locating Annex-A directory entries",
    })?;
    checked_range(
        bytes,
        "IMAGE_FILE_DIRECTORY",
        table_start,
        count.saturating_mul(12),
    )?;
    let mut entries = Vec::with_capacity(count);
    let mut previous = None;
    for index in 0..count {
        let entry_offset = table_start + index * 12;
        let entry = DirectoryEntry {
            tag: read_u16(bytes, entry_offset, "FIELD_TAG")?,
            element_type: read_u16(bytes, entry_offset + 2, "ELEMENT_TYPE")?,
            count: read_u32(bytes, entry_offset + 4, "NUM_ELEMENTS")?,
            value: read_u32(bytes, entry_offset + 8, "VALUES_OR_OFFSET")?,
        };
        if let Some(previous) = previous
            && entry.tag <= previous
        {
            return Err(NativeError::UnsortedAnnexATags {
                previous,
                current: entry.tag,
            });
        }
        previous = Some(entry.tag);
        entries.push(entry);
    }
    Ok(entries)
}

fn build_image(bytes: &[u8], entries: &[DirectoryEntry]) -> Result<AnnexAImage, NativeError> {
    let pixel_entry = required_entry(entries, TAG_PIXEL_FORMAT)?;
    validate_entry(pixel_entry, &[1], 16)?;
    let guid_range = checked_range(bytes, "PIXEL_FORMAT", pixel_entry.value as usize, 16)?;
    let mut pixel_format_guid = [0_u8; 16];
    pixel_format_guid.copy_from_slice(&bytes[guid_range]);

    let width = scalar_value(required_entry(entries, TAG_IMAGE_WIDTH)?)?;
    let height = scalar_value(required_entry(entries, TAG_IMAGE_HEIGHT)?)?;
    let image_offset = scalar_value(required_entry(entries, TAG_IMAGE_OFFSET)?)? as usize;
    let image_count = scalar_value(required_entry(entries, TAG_IMAGE_BYTE_COUNT)?)? as usize;
    let image_count = if image_count == 0 {
        bytes
            .len()
            .checked_sub(image_offset)
            .ok_or(NativeError::RangeOutsideInput {
                field: "IMAGE_OFFSET",
                offset: image_offset,
                length: 0,
                input_length: bytes.len(),
            })?
    } else {
        image_count
    };
    let codestream_range =
        checked_codestream_range(bytes, "IMAGE_BYTE_COUNT", image_offset, image_count)?;
    let alpha_range = parse_alpha_range(bytes, entries)?;
    if alpha_range.as_ref().is_some_and(|alpha| {
        alpha.start < codestream_range.end && codestream_range.start < alpha.end
    }) {
        return Err(NativeError::InvalidSyntax {
            field: "overlapping primary and alpha codestream ranges",
        });
    }
    let metadata = AnnexAMetadata {
        transformation: optional_scalar(entries, TAG_TRANSFORMATION)?,
        resolution_dpi_bits: optional_resolution(entries)?,
        icc_profile_range: optional_blob(bytes, entries, TAG_ICC_PROFILE)?,
    };
    Ok(AnnexAImage {
        width,
        height,
        pixel_format_guid,
        codestream_range,
        alpha_range,
        metadata,
    })
}

fn optional_resolution(entries: &[DirectoryEntry]) -> Result<Option<[u32; 2]>, NativeError> {
    let horizontal = entries
        .iter()
        .find(|entry| entry.tag == TAG_HORIZONTAL_RESOLUTION);
    let vertical = entries
        .iter()
        .find(|entry| entry.tag == TAG_VERTICAL_RESOLUTION);
    match (horizontal, vertical) {
        (None, None) => Ok(None),
        (Some(_), None) => Err(NativeError::MissingAnnexAField {
            tag: TAG_VERTICAL_RESOLUTION,
        }),
        (None, Some(_)) => Err(NativeError::MissingAnnexAField {
            tag: TAG_HORIZONTAL_RESOLUTION,
        }),
        (Some(horizontal), Some(vertical)) => {
            validate_entry(horizontal, &[11], 1)?;
            validate_entry(vertical, &[11], 1)?;
            Ok(Some([horizontal.value, vertical.value]))
        }
    }
}

fn parse_alpha_range(
    bytes: &[u8],
    entries: &[DirectoryEntry],
) -> Result<Option<Range<usize>>, NativeError> {
    let offset = optional_scalar(entries, TAG_ALPHA_OFFSET)?;
    let count = optional_scalar(entries, TAG_ALPHA_BYTE_COUNT)?;
    match (offset, count) {
        (None, None) | (Some(0), Some(0)) => Ok(None),
        (Some(_), None) => Err(NativeError::MissingAnnexAField {
            tag: TAG_ALPHA_BYTE_COUNT,
        }),
        (None, Some(_)) => Err(NativeError::MissingAnnexAField {
            tag: TAG_ALPHA_OFFSET,
        }),
        (Some(offset), Some(count)) => {
            checked_codestream_range(bytes, "ALPHA_BYTE_COUNT", offset as usize, count as usize)
                .map(Some)
        }
    }
}

fn checked_codestream_range(
    bytes: &[u8],
    field: &'static str,
    offset: usize,
    length: usize,
) -> Result<Range<usize>, NativeError> {
    match checked_range(bytes, field, offset, length) {
        Ok(range) => Ok(range),
        Err(NativeError::RangeOutsideInput { .. })
            if length.is_multiple_of(2)
                && bytes
                    .len()
                    .checked_sub(offset)
                    .and_then(|available| available.checked_add(1))
                    == Some(length) =>
        {
            // T.834 includes Annex-A writers that round an odd codestream byte
            // count to an even value without materializing the final pad byte.
            // The returned range remains strictly bounded by the input.
            Ok(offset..bytes.len())
        }
        Err(error) => Err(error),
    }
}

fn optional_blob(
    bytes: &[u8],
    entries: &[DirectoryEntry],
    tag: u16,
) -> Result<Option<Range<usize>>, NativeError> {
    let Some(entry) = entries.iter().find(|entry| entry.tag == tag) else {
        return Ok(None);
    };
    let size = element_size(entry.element_type).ok_or(NativeError::InvalidAnnexAEntry {
        tag,
        element_type: entry.element_type,
        count: entry.count,
    })?;
    let length = size
        .checked_mul(entry.count as usize)
        .ok_or(NativeError::IntegerOverflow {
            operation: "sizing Annex-A metadata",
        })?;
    checked_range(bytes, "Annex-A metadata", entry.value as usize, length).map(Some)
}

fn optional_scalar(entries: &[DirectoryEntry], tag: u16) -> Result<Option<u32>, NativeError> {
    entries
        .iter()
        .find(|entry| entry.tag == tag)
        .map(scalar_value)
        .transpose()
}

fn scalar_value(entry: &DirectoryEntry) -> Result<u32, NativeError> {
    validate_entry(entry, &[1, 3, 4], 1)?;
    Ok(match entry.element_type {
        1 => entry.value & 0xFF,
        3 => entry.value & 0xFFFF,
        4 => entry.value,
        _ => unreachable!(),
    })
}

fn validate_entry(
    entry: &DirectoryEntry,
    allowed_types: &[u16],
    expected_count: u32,
) -> Result<(), NativeError> {
    if entry.count == expected_count && allowed_types.contains(&entry.element_type) {
        Ok(())
    } else {
        Err(NativeError::InvalidAnnexAEntry {
            tag: entry.tag,
            element_type: entry.element_type,
            count: entry.count,
        })
    }
}

fn required_entry(entries: &[DirectoryEntry], tag: u16) -> Result<&DirectoryEntry, NativeError> {
    entries
        .iter()
        .find(|entry| entry.tag == tag)
        .ok_or(NativeError::MissingAnnexAField { tag })
}

fn checked_range(
    bytes: &[u8],
    field: &'static str,
    offset: usize,
    length: usize,
) -> Result<Range<usize>, NativeError> {
    let end = offset
        .checked_add(length)
        .ok_or(NativeError::IntegerOverflow {
            operation: "computing input byte range",
        })?;
    if end > bytes.len() {
        return Err(NativeError::RangeOutsideInput {
            field,
            offset,
            length,
            input_length: bytes.len(),
        });
    }
    Ok(offset..end)
}

fn read_u16(bytes: &[u8], offset: usize, field: &'static str) -> Result<u16, NativeError> {
    let range = checked_range(bytes, field, offset, 2)?;
    Ok(u16::from_le_bytes(bytes[range].try_into().unwrap()))
}

fn read_u32(bytes: &[u8], offset: usize, field: &'static str) -> Result<u32, NativeError> {
    let range = checked_range(bytes, field, offset, 4)?;
    Ok(u32::from_le_bytes(bytes[range].try_into().unwrap()))
}

const fn element_size(element_type: u16) -> Option<usize> {
    match element_type {
        1 | 2 | 6 | 7 => Some(1),
        3 | 8 => Some(2),
        4 | 9 | 11 => Some(4),
        5 | 10 | 12 => Some(8),
        _ => None,
    }
}
