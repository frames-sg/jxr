//! Borrowed JPEG XR inspection view.

use jxr_core::{ImageInfo, JxrError};
use jxr_native::{ParsedCodestream, image_info, parse_codestream};

use crate::{JxrDecoder, map_native_error};

/// Parsed, borrowed JPEG XR image view.
#[derive(Debug)]
pub struct JxrView<'a> {
    bytes: &'a [u8],
    parsed: ParsedCodestream,
    info: ImageInfo,
}

impl<'a> JxrView<'a> {
    /// Parse raw T.832 codestream or Annex-A file metadata without copying input bytes.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, JxrError> {
        let parsed = parse_codestream(bytes).map_err(map_native_error)?;
        let info = image_info(&parsed).map_err(map_native_error)?;
        Ok(Self {
            bytes,
            parsed,
            info,
        })
    }

    /// Parsed image information.
    #[must_use]
    pub const fn info(&self) -> &ImageInfo {
        &self.info
    }

    /// Original compressed bytes.
    #[must_use]
    pub const fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Borrow the raw primary T.832 codestream without its Annex-A wrapper.
    #[must_use]
    pub fn codestream(&self) -> &'a [u8] {
        &self.bytes[self.parsed.codestream_range.clone()]
    }

    /// Borrow the separately encoded raw alpha codestream, when present.
    #[must_use]
    pub fn separate_alpha_codestream(&self) -> Option<&'a [u8]> {
        self.parsed
            .annex_a
            .as_ref()?
            .alpha_range
            .clone()
            .map(|range| &self.bytes[range])
    }

    /// Borrow the embedded Annex-A ICC profile without copying it.
    #[must_use]
    pub fn icc_profile(&self) -> Option<&'a [u8]> {
        let range = self.info.metadata.icc_profile?;
        range
            .end()
            .and_then(|end| self.bytes.get(range.offset..end))
    }

    /// Construct a reusable decoder borrowing this view.
    #[must_use]
    pub fn decoder(&self) -> JxrDecoder<'_> {
        JxrDecoder::new(self.bytes, &self.parsed, &self.info)
    }
}
