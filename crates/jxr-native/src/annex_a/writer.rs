//! Validated Annex-A still-image serialization.

use jxr_core::Orientation;

use crate::{NativeError, parse_codestream};

const TAG_ICC_PROFILE: u16 = 0x8773;
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

const TYPE_BYTE: u16 = 1;
const TYPE_LONG: u16 = 4;
const TYPE_UNDEFINED: u16 = 7;
const TYPE_FLOAT: u16 = 11;

/// Metadata and optional payloads for an Annex-A still-image container.
#[derive(Clone, Copy, Debug)]
pub struct AnnexAWriteOptions<'a> {
    width: u32,
    height: u32,
    pixel_format_guid: [u8; 16],
    orientation: Orientation,
    horizontal_resolution_dpi: f32,
    vertical_resolution_dpi: f32,
    icc_profile: Option<&'a [u8]>,
    separate_alpha: Option<&'a [u8]>,
}

impl<'a> AnnexAWriteOptions<'a> {
    /// Create options for one raw codestream using identity orientation and 96 DPI.
    #[must_use]
    pub const fn new(width: u32, height: u32, pixel_format_guid: [u8; 16]) -> Self {
        Self {
            width,
            height,
            pixel_format_guid,
            orientation: Orientation::Identity,
            horizontal_resolution_dpi: 96.0,
            vertical_resolution_dpi: 96.0,
            icc_profile: None,
            separate_alpha: None,
        }
    }

    /// Set the lossless Annex-A presentation transformation.
    #[must_use]
    pub const fn with_orientation(mut self, orientation: Orientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Set horizontal and vertical display resolution in dots per inch.
    ///
    /// Both values must be finite and greater than zero.
    #[must_use]
    pub const fn with_resolution_dpi(mut self, horizontal: f32, vertical: f32) -> Self {
        self.horizontal_resolution_dpi = horizontal;
        self.vertical_resolution_dpi = vertical;
        self
    }

    /// Attach an ICC profile, copied exactly into the output container.
    #[must_use]
    pub const fn with_icc_profile(mut self, profile: &'a [u8]) -> Self {
        self.icc_profile = Some(profile);
        self
    }

    /// Attach a separately encoded raw alpha codestream.
    #[must_use]
    pub const fn with_separate_alpha(mut self, codestream: &'a [u8]) -> Self {
        self.separate_alpha = Some(codestream);
        self
    }
}

/// Serialize a raw T.832 codestream into a deterministic Annex-A still-image file.
///
/// The primary and optional separate-alpha codestreams are fully parsed before
/// serialization. Their dimensions must match the declared container dimensions.
/// The finished file is parsed again, including known pixel-format consistency
/// checks, before it is returned.
pub fn write_annex_a(
    primary: &[u8],
    options: &AnnexAWriteOptions<'_>,
) -> Result<Vec<u8>, NativeError> {
    validate_options(primary, options)?;
    let layout = ContainerLayout::new(primary, options)?;
    let mut output = vec![0_u8; layout.output_len];
    write_directory(&mut output, primary, options, &layout)?;
    write_payloads(&mut output, primary, options, &layout);
    parse_codestream(&output)?;
    Ok(output)
}

#[derive(Clone, Copy, Debug)]
struct ContainerLayout {
    entry_count: usize,
    pixel_format_offset: usize,
    icc_offset: Option<usize>,
    primary_offset: usize,
    alpha_offset: Option<usize>,
    output_len: usize,
}

impl ContainerLayout {
    fn new(primary: &[u8], options: &AnnexAWriteOptions<'_>) -> Result<Self, NativeError> {
        let entry_count = 8_usize
            + usize::from(options.icc_profile.is_some())
            + 2 * usize::from(options.separate_alpha.is_some());
        let directory_size = 2_usize
            .checked_add(
                entry_count
                    .checked_mul(12)
                    .ok_or_else(|| overflow("sizing Annex-A directory"))?,
            )
            .and_then(|value| value.checked_add(4))
            .ok_or_else(|| overflow("sizing Annex-A directory"))?;
        let pixel_format_offset = align_four(
            8_usize
                .checked_add(directory_size)
                .ok_or_else(|| overflow("locating Annex-A payloads"))?,
        )?;
        let mut cursor = checked_advance(pixel_format_offset, options.pixel_format_guid.len())?;
        let icc_offset = if let Some(profile) = options.icc_profile {
            cursor = align_four(cursor)?;
            let offset = cursor;
            cursor = checked_advance(cursor, profile.len())?;
            Some(offset)
        } else {
            None
        };
        let primary_offset = align_four(cursor)?;
        cursor = checked_advance(primary_offset, primary.len())?;
        let alpha_offset = if let Some(alpha) = options.separate_alpha {
            cursor = align_four(cursor)?;
            let offset = cursor;
            cursor = checked_advance(cursor, alpha.len())?;
            Some(offset)
        } else {
            None
        };
        checked_u32(primary.len(), "representing primary codestream length")?;
        if let Some(profile) = options.icc_profile {
            checked_u32(profile.len(), "representing ICC profile length")?;
        }
        if let Some(alpha) = options.separate_alpha {
            checked_u32(alpha.len(), "representing alpha codestream length")?;
        }
        checked_u32(cursor, "representing Annex-A file length")?;
        Ok(Self {
            entry_count,
            pixel_format_offset,
            icc_offset,
            primary_offset,
            alpha_offset,
            output_len: cursor,
        })
    }
}

fn write_directory(
    output: &mut [u8],
    primary: &[u8],
    options: &AnnexAWriteOptions<'_>,
    layout: &ContainerLayout,
) -> Result<(), NativeError> {
    output[..4].copy_from_slice(&[b'I', b'I', 0xBC, 1]);
    output[4..8].copy_from_slice(&8_u32.to_le_bytes());
    output[8..10].copy_from_slice(
        &u16::try_from(layout.entry_count)
            .map_err(|_| overflow("representing Annex-A directory count"))?
            .to_le_bytes(),
    );

    let mut entries = DirectoryEntries::new(output);
    if let (Some(profile), Some(offset)) = (options.icc_profile, layout.icc_offset) {
        entries.push(
            TAG_ICC_PROFILE,
            TYPE_UNDEFINED,
            checked_u32(profile.len(), "representing ICC profile length")?,
            checked_u32(offset, "representing ICC profile offset")?,
        );
    }
    entries.push(
        TAG_PIXEL_FORMAT,
        TYPE_BYTE,
        16,
        checked_u32(
            layout.pixel_format_offset,
            "representing pixel format offset",
        )?,
    );
    entries.push(
        TAG_TRANSFORMATION,
        TYPE_LONG,
        1,
        orientation_code(options.orientation),
    );
    entries.push(TAG_IMAGE_WIDTH, TYPE_LONG, 1, options.width);
    entries.push(TAG_IMAGE_HEIGHT, TYPE_LONG, 1, options.height);
    entries.push(
        TAG_HORIZONTAL_RESOLUTION,
        TYPE_FLOAT,
        1,
        options.horizontal_resolution_dpi.to_bits(),
    );
    entries.push(
        TAG_VERTICAL_RESOLUTION,
        TYPE_FLOAT,
        1,
        options.vertical_resolution_dpi.to_bits(),
    );
    entries.push(
        TAG_IMAGE_OFFSET,
        TYPE_LONG,
        1,
        checked_u32(
            layout.primary_offset,
            "representing primary codestream offset",
        )?,
    );
    entries.push(
        TAG_IMAGE_BYTE_COUNT,
        TYPE_LONG,
        1,
        checked_u32(primary.len(), "representing primary codestream length")?,
    );
    if let (Some(alpha), Some(offset)) = (options.separate_alpha, layout.alpha_offset) {
        entries.push(
            TAG_ALPHA_OFFSET,
            TYPE_LONG,
            1,
            checked_u32(offset, "representing alpha codestream offset")?,
        );
        entries.push(
            TAG_ALPHA_BYTE_COUNT,
            TYPE_LONG,
            1,
            checked_u32(alpha.len(), "representing alpha codestream length")?,
        );
    }
    Ok(())
}

struct DirectoryEntries<'a> {
    output: &'a mut [u8],
    offset: usize,
}

impl<'a> DirectoryEntries<'a> {
    fn new(output: &'a mut [u8]) -> Self {
        Self { output, offset: 10 }
    }

    fn push(&mut self, tag: u16, element_type: u16, count: u32, value: u32) {
        self.output[self.offset..self.offset + 2].copy_from_slice(&tag.to_le_bytes());
        self.output[self.offset + 2..self.offset + 4].copy_from_slice(&element_type.to_le_bytes());
        self.output[self.offset + 4..self.offset + 8].copy_from_slice(&count.to_le_bytes());
        self.output[self.offset + 8..self.offset + 12].copy_from_slice(&value.to_le_bytes());
        self.offset += 12;
    }
}

fn write_payloads(
    output: &mut [u8],
    primary: &[u8],
    options: &AnnexAWriteOptions<'_>,
    layout: &ContainerLayout,
) {
    output[layout.pixel_format_offset..layout.pixel_format_offset + 16]
        .copy_from_slice(&options.pixel_format_guid);
    if let (Some(profile), Some(offset)) = (options.icc_profile, layout.icc_offset) {
        output[offset..offset + profile.len()].copy_from_slice(profile);
    }
    output[layout.primary_offset..layout.primary_offset + primary.len()].copy_from_slice(primary);
    if let (Some(alpha), Some(offset)) = (options.separate_alpha, layout.alpha_offset) {
        output[offset..offset + alpha.len()].copy_from_slice(alpha);
    }
}

fn validate_options(primary: &[u8], options: &AnnexAWriteOptions<'_>) -> Result<(), NativeError> {
    validate_raw_codestream(primary, options.width, options.height, false)?;
    if !options.horizontal_resolution_dpi.is_finite()
        || options.horizontal_resolution_dpi <= 0.0
        || !options.vertical_resolution_dpi.is_finite()
        || options.vertical_resolution_dpi <= 0.0
    {
        return Err(NativeError::InvalidSyntax {
            field: "Annex-A display resolution",
        });
    }
    if options.icc_profile.is_some_and(<[u8]>::is_empty) {
        return Err(NativeError::InvalidSyntax {
            field: "empty Annex-A ICC profile",
        });
    }
    if let Some(alpha) = options.separate_alpha {
        validate_raw_codestream(alpha, options.width, options.height, true)?;
    }
    Ok(())
}

fn validate_raw_codestream(
    bytes: &[u8],
    width: u32,
    height: u32,
    alpha: bool,
) -> Result<(), NativeError> {
    if !bytes.starts_with(b"WMPHOTO\0") {
        return Err(NativeError::InvalidSyntax {
            field: "Annex-A writer requires a raw codestream",
        });
    }
    let parsed = parse_codestream(bytes)?;
    if parsed.headers.image.width != width || parsed.headers.image.height != height {
        return Err(NativeError::InvalidSyntax {
            field: if alpha {
                "Annex-A separate alpha dimensions"
            } else {
                "Annex-A/codestream dimension mismatch"
            },
        });
    }
    Ok(())
}

const fn orientation_code(orientation: Orientation) -> u32 {
    match orientation {
        Orientation::Identity => 0,
        Orientation::MirrorVertical => 1,
        Orientation::MirrorHorizontal => 2,
        Orientation::Rotate180 => 3,
        Orientation::Rotate90 => 4,
        Orientation::Transverse => 5,
        Orientation::Transpose => 6,
        Orientation::Rotate270 => 7,
    }
}

fn align_four(value: usize) -> Result<usize, NativeError> {
    value
        .checked_add(3)
        .map(|value| value & !3)
        .ok_or_else(|| overflow("aligning Annex-A payload"))
}

fn checked_advance(offset: usize, length: usize) -> Result<usize, NativeError> {
    offset
        .checked_add(length)
        .ok_or_else(|| overflow("sizing Annex-A output"))
}

fn checked_u32(value: usize, operation: &'static str) -> Result<u32, NativeError> {
    u32::try_from(value).map_err(|_| overflow(operation))
}

const fn overflow(operation: &'static str) -> NativeError {
    NativeError::IntegerOverflow { operation }
}
