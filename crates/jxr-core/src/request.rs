//! Decode request policy and allocation limits.

use crate::{BackendRequest, BandPresence, JxrError, LimitKind, PixelFormat, Rect};

/// Native JPEG XR transform resolution selected during decode.
///
/// JPEG XR exposes only the full DC+LP+HP resolution, the DC+LP resolution,
/// and the DC-only resolution without spatial resampling.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum DecodeScale {
    /// Full-resolution DC, LP, and HP reconstruction.
    #[default]
    Full,
    /// DC plus LP reconstruction at one quarter width and height.
    Quarter,
    /// DC-only reconstruction at one sixteenth width and height.
    Sixteenth,
}

impl DecodeScale {
    /// Integer denominator applied independently to width and height.
    #[must_use]
    pub const fn denominator(self) -> u32 {
        match self {
            Self::Full => 1,
            Self::Quarter => 4,
            Self::Sixteenth => 16,
        }
    }

    /// Frequency bands needed at this native resolution, capped by availability.
    #[must_use]
    pub const fn retained_bands(self, available: BandPresence) -> BandPresence {
        match self {
            Self::Full => available,
            Self::Quarter if available.has_low_pass() => BandPresence::NoHighPass,
            Self::Quarter | Self::Sixteenth => BandPresence::DcOnly,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AlphaHandling {
    #[default]
    Preserve,
    Drop,
    Premultiply,
}

/// Caller-controlled resource ceilings checked before allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecodeLimits {
    pub max_width: u32,
    pub max_height: u32,
    pub max_pixels: u64,
    pub max_components: u16,
    pub max_tiles: u32,
    pub max_compressed_bytes: u64,
    pub max_coefficient_bytes: u64,
    pub max_host_allocation_bytes: u64,
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            max_width: 1 << 20,
            max_height: 1 << 20,
            max_pixels: 1 << 30,
            max_components: 64,
            max_tiles: 1 << 20,
            max_compressed_bytes: 1 << 30,
            max_coefficient_bytes: 1 << 32,
            max_host_allocation_bytes: 1 << 32,
        }
    }
}

impl DecodeLimits {
    pub fn check_dimensions(self, width: u32, height: u32) -> Result<(), JxrError> {
        check_limit(
            "image width",
            LimitKind::Width,
            u64::from(width),
            u64::from(self.max_width),
        )?;
        check_limit(
            "image height",
            LimitKind::Height,
            u64::from(height),
            u64::from(self.max_height),
        )?;
        let pixels = u64::from(width)
            .checked_mul(u64::from(height))
            .ok_or_else(|| JxrError::arithmetic("image pixel count"))?;
        check_limit(
            "image pixel count",
            LimitKind::Pixels,
            pixels,
            self.max_pixels,
        )
    }

    pub fn check_components(self, components: u16) -> Result<(), JxrError> {
        check_limit(
            "component count",
            LimitKind::Components,
            u64::from(components),
            u64::from(self.max_components),
        )
    }

    pub fn check_tiles(self, tiles: u32) -> Result<(), JxrError> {
        check_limit(
            "tile count",
            LimitKind::Tiles,
            u64::from(tiles),
            u64::from(self.max_tiles),
        )
    }

    pub fn check_compressed_bytes(self, bytes: usize) -> Result<(), JxrError> {
        let requested =
            u64::try_from(bytes).map_err(|_| JxrError::arithmetic("compressed input size"))?;
        check_limit(
            "compressed input size",
            LimitKind::CompressedBytes,
            requested,
            self.max_compressed_bytes,
        )
    }

    /// Reject a coefficient arena before attempting its allocation.
    pub fn check_coefficient_bytes(self, bytes: usize) -> Result<(), JxrError> {
        let requested =
            u64::try_from(bytes).map_err(|_| JxrError::arithmetic("coefficient arena size"))?;
        check_limit(
            "coefficient arena size",
            LimitKind::CoefficientBytes,
            requested,
            self.max_coefficient_bytes,
        )
    }

    /// Reject host output before returning an allocation beyond the caller's limit.
    pub fn check_host_allocation_bytes(self, bytes: usize) -> Result<(), JxrError> {
        let requested =
            u64::try_from(bytes).map_err(|_| JxrError::arithmetic("host output size"))?;
        check_limit(
            "host output size",
            LimitKind::HostAllocationBytes,
            requested,
            self.max_host_allocation_bytes,
        )
    }
}

fn check_limit(
    operation: &'static str,
    limit: LimitKind,
    requested: u64,
    maximum: u64,
) -> Result<(), JxrError> {
    if requested > maximum {
        Err(JxrError::limit(operation, limit, requested, maximum))
    } else {
        Ok(())
    }
}

/// Host decode request. `None` region means the full image.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DecodeRequest {
    pub region: Option<Rect>,
    /// Native transform resolution. Regions remain in full-resolution source coordinates.
    pub scale: DecodeScale,
    pub output: PixelFormat,
    pub alpha: AlphaHandling,
    pub backend: BackendRequest,
    pub limits: DecodeLimits,
}

impl DecodeRequest {
    #[must_use]
    pub fn new(output: PixelFormat) -> Self {
        Self {
            region: None,
            scale: DecodeScale::Full,
            output,
            alpha: AlphaHandling::Preserve,
            backend: BackendRequest::Auto,
            limits: DecodeLimits::default(),
        }
    }

    /// Set a display-space output region.
    #[must_use]
    pub const fn with_region(mut self, region: Rect) -> Self {
        self.region = Some(region);
        self
    }

    /// Select a native JPEG XR transform resolution.
    #[must_use]
    pub const fn with_scale(mut self, scale: DecodeScale) -> Self {
        self.scale = scale;
        self
    }

    /// Set alpha preservation or premultiplication policy.
    #[must_use]
    pub const fn with_alpha(mut self, alpha: AlphaHandling) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set automatic, CPU, or strict accelerator routing.
    #[must_use]
    pub const fn with_backend(mut self, backend: BackendRequest) -> Self {
        self.backend = backend;
        self
    }

    /// Replace the caller-controlled resource ceilings.
    #[must_use]
    pub const fn with_limits(mut self, limits: DecodeLimits) -> Self {
        self.limits = limits;
        self
    }
}
