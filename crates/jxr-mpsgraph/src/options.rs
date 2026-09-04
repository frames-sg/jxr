// SPDX-License-Identifier: MIT OR Apache-2.0

use jxr::{AlphaHandling, DecodeLimits, PixelFormat, PreparedJxr, Rect};

/// Per-image decode policy accepted by the `MPSGraph` adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MpsGraphDecodeOptions {
    pub region: Option<Rect>,
    pub alpha: AlphaHandling,
    pub limits: DecodeLimits,
    pub format: PixelFormat,
}

impl MpsGraphDecodeOptions {
    #[must_use]
    pub fn new(format: PixelFormat) -> Self {
        Self {
            region: None,
            alpha: AlphaHandling::Preserve,
            limits: DecodeLimits::default(),
            format,
        }
    }

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    pub(crate) fn strict_request(&self) -> jxr::DecodeRequest {
        let mut request = jxr::DecodeRequest::new(self.format)
            .with_alpha(self.alpha)
            .with_limits(self.limits)
            .with_backend(jxr::BackendRequest::Metal);
        request.region = self.region;
        request
    }
}

/// One already parsed JPEG XR input and its tensor output policy.
#[derive(Debug, Clone)]
pub struct MpsGraphDecodeInput {
    pub image: PreparedJxr,
    pub options: MpsGraphDecodeOptions,
}
