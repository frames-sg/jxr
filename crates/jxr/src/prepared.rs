//! Owned, parse-once JPEG XR input.

use std::sync::Arc;

use jxr_core::{ImageInfo, JxrError};
use jxr_native::{ParsedCodestream, image_info, parse_codestream};

use crate::{JxrDecoder, map_native_error};

/// Immutable owned JPEG XR input and parsed decode metadata.
#[derive(Clone, Debug)]
pub struct PreparedJxr {
    inner: Arc<PreparedJxrInner>,
}

#[derive(Debug)]
struct PreparedJxrInner {
    bytes: Arc<[u8]>,
    parsed: ParsedCodestream,
    info: ImageInfo,
}

impl PreparedJxr {
    /// Parse and retain an owned compressed input without duplicating it.
    pub fn from_arc(bytes: Arc<[u8]>) -> Result<Self, JxrError> {
        let parsed = parse_codestream(&bytes).map_err(map_native_error)?;
        let info = image_info(&parsed).map_err(map_native_error)?;
        Ok(Self {
            inner: Arc::new(PreparedJxrInner {
                bytes,
                parsed,
                info,
            }),
        })
    }

    /// Parsed image information.
    #[must_use]
    pub fn info(&self) -> &ImageInfo {
        &self.inner.info
    }

    /// Shared compressed storage retained by this prepared image.
    #[must_use]
    pub fn bytes(&self) -> &Arc<[u8]> {
        &self.inner.bytes
    }

    /// Borrow the raw primary T.832 codestream without its Annex-A wrapper.
    #[must_use]
    pub fn codestream(&self) -> &[u8] {
        &self.inner.bytes[self.inner.parsed.codestream_range.clone()]
    }

    /// Borrow the separately encoded raw alpha codestream, when present.
    #[must_use]
    pub fn separate_alpha_codestream(&self) -> Option<&[u8]> {
        self.inner
            .parsed
            .annex_a
            .as_ref()?
            .alpha_range
            .clone()
            .map(|range| &self.inner.bytes[range])
    }

    /// Borrow the embedded Annex-A ICC profile without copying it.
    #[must_use]
    pub fn icc_profile(&self) -> Option<&[u8]> {
        let range = self.inner.info.metadata.icc_profile?;
        range
            .end()
            .and_then(|end| self.inner.bytes.get(range.offset..end))
    }

    /// Construct a reusable decoder borrowing this prepared image.
    #[must_use]
    pub fn decoder(&self) -> JxrDecoder<'_> {
        JxrDecoder::new(&self.inner.bytes, &self.inner.parsed, &self.inner.info)
    }
}
