//! Annex-A still-image container writing.

use crate::{JxrError, map_native_error};

pub use jxr_native::AnnexAWriteOptions;

/// Wrap a validated raw T.832 codestream in an Annex-A still-image container.
///
/// The returned container owns exact copies of the primary codestream and the
/// optional ICC and separate-alpha payloads supplied through `options`.
///
/// ```no_run
/// use jxr::{AnnexAWriteOptions, JxrError, Orientation, write_annex_a};
///
/// # fn wrap(raw_codestream: &[u8], pixel_format_guid: [u8; 16]) -> Result<Vec<u8>, JxrError> {
/// let options = AnnexAWriteOptions::new(640, 480, pixel_format_guid)
///     .with_orientation(Orientation::Identity)
///     .with_resolution_dpi(300.0, 300.0)
///     .with_icc_profile(b"embedded ICC profile");
/// write_annex_a(raw_codestream, &options)
/// # }
/// ```
pub fn write_annex_a(
    primary: &[u8],
    options: &AnnexAWriteOptions<'_>,
) -> Result<Vec<u8>, JxrError> {
    jxr_native::write_annex_a(primary, options).map_err(map_native_error)
}
