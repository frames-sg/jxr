use std::{
    num::NonZeroUsize,
    sync::{Arc, OnceLock},
};

use jxr_core::{
    DecodeReport, DecodeRequest, ImageInfo, PixelFormat, PreparedPlan, Rect, StorageKind,
    SurfaceLayout,
};

use super::{BatchInfrastructureError, IndexedBatchError};
use crate::PreparedJxr;

/// Policy retained by an owned native batch decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchDecodeOptions {
    /// Dense tensor channel ordering. Native preserves the codec surface layout.
    pub layout: BatchLayout,
    /// Image-level worker count. `None` uses available parallelism.
    pub workers: Option<NonZeroUsize>,
    /// Maximum number of inputs accepted by one batch call.
    pub max_inputs: usize,
    /// Maximum aggregate bytes retained by successful dense CPU outputs.
    pub max_host_allocation_bytes: u64,
    /// Maximum retained parse/plan entries keyed by `Arc` identity and request.
    pub preparation_cache_entries: usize,
}

impl Default for BatchDecodeOptions {
    fn default() -> Self {
        Self {
            layout: BatchLayout::Native,
            workers: None,
            max_inputs: 1 << 20,
            max_host_allocation_bytes: 1 << 32,
            preparation_cache_entries: 256,
        }
    }
}

/// Dense channel ordering for owned CPU batch samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum BatchLayout {
    /// Preserve the exact interleaved or planar codec surface layout.
    #[default]
    Native,
    /// Batch, channel, height, width for single-plane unpacked formats.
    Nchw,
}

/// One owned compressed JPEG XR image and its output request.
#[derive(Debug, Clone)]
pub struct EncodedImage {
    /// Raw T.832 codestream or Annex-A bytes retained without copying.
    pub bytes: Arc<[u8]>,
    /// Per-image region, format, alpha, backend, and resource policy.
    pub request: DecodeRequest,
}

impl EncodedImage {
    /// Construct one owned image request.
    #[must_use]
    pub fn new(bytes: Arc<[u8]>, request: DecodeRequest) -> Self {
        Self { bytes, request }
    }

    /// Construct a full-image request using automatic backend policy.
    #[must_use]
    pub fn full(bytes: Arc<[u8]>, format: PixelFormat) -> Self {
        Self::new(bytes, DecodeRequest::new(format))
    }
}

/// Dense native output contract shared by one homogeneous batch group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchGroupInfo {
    layout: SurfaceLayout,
    batch_layout: BatchLayout,
}

impl BatchGroupInfo {
    pub(crate) const fn new(layout: SurfaceLayout, batch_layout: BatchLayout) -> Self {
        Self {
            layout,
            batch_layout,
        }
    }

    /// Output width and height for every image in the group.
    #[must_use]
    pub const fn dimensions(&self) -> (u32, u32) {
        (self.layout.width, self.layout.height)
    }

    /// Exact requested pixel representation.
    #[must_use]
    pub const fn format(&self) -> PixelFormat {
        self.layout.format
    }

    /// Native Rust storage type used by the dense sample owner.
    #[must_use]
    pub const fn storage_kind(&self) -> StorageKind {
        self.layout.format.storage_kind()
    }

    /// Exact byte distance between consecutive dense images.
    #[must_use]
    pub const fn image_stride_bytes(&self) -> usize {
        self.layout.byte_len
    }

    /// Number of native Rust storage elements occupied by one image.
    ///
    /// For bit-packed output, one storage element is one byte.
    #[must_use]
    pub const fn image_stride_elements(&self) -> usize {
        let element_bytes = match self.storage_kind() {
            StorageKind::BitPacked | StorageKind::U8 => 1,
            StorageKind::U16 | StorageKind::I16 | StorageKind::F16Bits | StorageKind::PackedU16 => {
                2
            }
            StorageKind::I32 | StorageKind::F32 | StorageKind::PackedU32 => 4,
        };
        self.layout.byte_len / element_bytes
    }

    /// Per-image planar or interleaved surface layout.
    #[must_use]
    pub const fn image_layout(&self) -> &SurfaceLayout {
        &self.layout
    }

    /// Dense tensor channel ordering applied by the CPU batch owner.
    #[must_use]
    pub const fn batch_layout(&self) -> BatchLayout {
        self.batch_layout
    }
}

/// Cheaply cloneable parsed request and retained execution plan for one input.
#[derive(Clone, Debug)]
pub struct PreparedImage {
    pub(crate) inner: Arc<PreparedImageInner>,
}

#[derive(Debug)]
pub(crate) struct PreparedImageInner {
    pub(crate) contract: Arc<PreparedImageContract>,
    pub(crate) original_source_index: usize,
}

#[derive(Debug)]
pub(crate) struct PreparedImageContract {
    pub(crate) image: PreparedJxr,
    pub(crate) request: DecodeRequest,
    pub(crate) plan: PreparedPlan,
    pub(crate) info: BatchGroupInfo,
    pub(crate) reconstruction: OnceLock<Result<crate::PreparedReconstruction, jxr_core::JxrError>>,
}

impl PreparedImage {
    /// Parsed, owned compressed image.
    #[must_use]
    pub fn image(&self) -> &PreparedJxr {
        &self.inner.contract.image
    }

    /// Request captured during preparation.
    #[must_use]
    pub fn request(&self) -> &DecodeRequest {
        &self.inner.contract.request
    }

    /// Validated plan reused without reparsing or replanning.
    #[must_use]
    pub fn plan(&self) -> &PreparedPlan {
        &self.inner.contract.plan
    }

    /// Dense output contract derived during preparation.
    #[must_use]
    pub fn info(&self) -> &BatchGroupInfo {
        &self.inner.contract.info
    }

    /// Input position from the preparation call that created this image.
    #[must_use]
    pub fn original_source_index(&self) -> usize {
        self.inner.original_source_index
    }

    /// Decode and cache coefficient-ready reconstruction state on first use.
    ///
    /// Images deduplicated by the same preparation session share this cache.
    pub fn prepare_reconstruction(
        &self,
    ) -> Result<crate::PreparedReconstruction, jxr_core::JxrError> {
        self.inner
            .contract
            .reconstruction
            .get_or_init(|| {
                self.image()
                    .decoder()
                    .prepare_reconstruction_from_plan(self.request(), self.plan().clone())
            })
            .clone()
    }

    /// Whether coefficient-ready reconstruction has already been attempted.
    #[must_use]
    pub fn reconstruction_is_cached(&self) -> bool {
        self.inner.contract.reconstruction.get().is_some()
    }
}

/// One homogeneous set of prepared images in stable caller order.
#[derive(Debug)]
pub struct PreparedBatchGroup {
    pub(crate) info: BatchGroupInfo,
    pub(crate) images: Vec<PreparedImage>,
    pub(crate) source_indices: Vec<usize>,
}

impl PreparedBatchGroup {
    /// Shared dense output contract.
    #[must_use]
    pub const fn info(&self) -> &BatchGroupInfo {
        &self.info
    }

    /// Prepared images in stable group order.
    #[must_use]
    pub fn images(&self) -> &[PreparedImage] {
        &self.images
    }

    /// Positions in the current batch input collection.
    #[must_use]
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }
}

/// Parsed, grouped batch reusable across CPU and accelerator sessions.
#[derive(Clone, Debug)]
pub struct PreparedBatch {
    pub(crate) groups: Arc<[PreparedBatchGroup]>,
    pub(crate) errors: Arc<[IndexedBatchError]>,
    pub(crate) options: BatchDecodeOptions,
}

impl PreparedBatch {
    /// Total successful and failed inputs represented by this batch.
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.groups.iter().fold(self.errors.len(), |count, group| {
            count.saturating_add(group.images.len())
        })
    }

    /// Homogeneous groups in first-occurrence order.
    #[must_use]
    pub fn groups(&self) -> &[PreparedBatchGroup] {
        &self.groups
    }

    /// Indexed input-local preparation failures.
    #[must_use]
    pub fn errors(&self) -> &[IndexedBatchError] {
        &self.errors
    }

    /// Batch policy captured during preparation.
    #[must_use]
    pub const fn options(&self) -> BatchDecodeOptions {
        self.options
    }
}

/// Native-width contiguous samples for one homogeneous CPU batch group.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum CpuBatchSamples {
    /// One-bit rows retained in byte-packed form per image.
    BitPacked(Vec<u8>),
    /// Unsigned eight-bit samples.
    U8(Vec<u8>),
    /// Unsigned sixteen-bit samples.
    U16(Vec<u16>),
    /// Signed sixteen-bit samples.
    I16(Vec<i16>),
    /// Signed thirty-two-bit samples.
    I32(Vec<i32>),
    /// IEEE binary16 bit patterns.
    F16(Vec<u16>),
    /// IEEE binary32 samples.
    F32(Vec<f32>),
    /// Packed RGB 5:5:5 words.
    Rgb555(Vec<u16>),
    /// Packed RGB 5:6:5 words.
    Rgb565(Vec<u16>),
    /// Packed RGB 10:10:10 words.
    Rgb101010(Vec<u32>),
    /// Packed shared-exponent RGB words.
    Rgbe(Vec<u32>),
}

/// Caller-owned storage for one prepared homogeneous CPU batch group.
pub type CpuBatchDestination<'a> = jxr_core::DecodedSamplesMut<'a>;

impl CpuBatchSamples {
    /// Native storage type for this owner.
    #[must_use]
    pub const fn storage_kind(&self) -> StorageKind {
        match self {
            Self::BitPacked(_) => StorageKind::BitPacked,
            Self::U8(_) => StorageKind::U8,
            Self::U16(_) => StorageKind::U16,
            Self::I16(_) => StorageKind::I16,
            Self::I32(_) => StorageKind::I32,
            Self::F16(_) => StorageKind::F16Bits,
            Self::F32(_) => StorageKind::F32,
            Self::Rgb555(_) | Self::Rgb565(_) => StorageKind::PackedU16,
            Self::Rgb101010(_) | Self::Rgbe(_) => StorageKind::PackedU32,
        }
    }

    /// Number of native storage elements across every image.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::BitPacked(values) | Self::U8(values) => values.len(),
            Self::U16(values) | Self::F16(values) | Self::Rgb555(values) | Self::Rgb565(values) => {
                values.len()
            }
            Self::I16(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::Rgb101010(values) | Self::Rgbe(values) => values.len(),
        }
    }

    /// Whether this owner contains no native storage elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total bytes occupied by all native elements.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        match self {
            Self::BitPacked(values) | Self::U8(values) => values.len(),
            Self::U16(values) | Self::F16(values) | Self::Rgb555(values) | Self::Rgb565(values) => {
                values.len().saturating_mul(2)
            }
            Self::I16(values) => values.len().saturating_mul(2),
            Self::I32(values) => values.len().saturating_mul(4),
            Self::F32(values) => values.len().saturating_mul(4),
            Self::Rgb101010(values) | Self::Rgbe(values) => values.len().saturating_mul(4),
        }
    }
}

/// One successful homogeneous native CPU output group.
#[derive(Debug, PartialEq)]
pub struct CpuBatchGroup {
    pub(crate) info: BatchGroupInfo,
    pub(crate) source_indices: Vec<usize>,
    pub(crate) image_infos: Vec<ImageInfo>,
    pub(crate) decoded_regions: Vec<Rect>,
    pub(crate) reports: Vec<DecodeReport>,
    pub(crate) samples: CpuBatchSamples,
}

impl CpuBatchGroup {
    /// Shared dense output contract.
    #[must_use]
    pub const fn info(&self) -> &BatchGroupInfo {
        &self.info
    }

    /// Original input positions represented by the dense batch dimension.
    #[must_use]
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }

    /// Parsed source metadata for every successful image.
    #[must_use]
    pub fn image_infos(&self) -> &[ImageInfo] {
        &self.image_infos
    }

    /// Actual decoded region for every successful image.
    #[must_use]
    pub fn decoded_regions(&self) -> &[Rect] {
        &self.decoded_regions
    }

    /// Route and stage reports in dense batch order.
    #[must_use]
    pub fn reports(&self) -> &[DecodeReport] {
        &self.reports
    }

    /// Exact byte distance between consecutive images.
    #[must_use]
    pub const fn image_stride_bytes(&self) -> usize {
        self.info.image_stride_bytes()
    }

    /// Number of native storage elements between consecutive images.
    #[must_use]
    pub const fn image_stride_elements(&self) -> usize {
        self.info.image_stride_elements()
    }

    /// Contiguous native samples for every successful image.
    #[must_use]
    pub const fn samples(&self) -> &CpuBatchSamples {
        &self.samples
    }
}

/// Successful native groups plus indexed input-local failures.
#[derive(Debug, PartialEq)]
pub struct CpuBatchDecodeResult {
    pub(crate) groups: Vec<CpuBatchGroup>,
    pub(crate) errors: Vec<IndexedBatchError>,
}

/// Metadata and indexed failures from decoding into caller-owned group storage.
#[derive(Debug, PartialEq)]
pub struct CpuBatchIntoResult {
    pub(crate) source_indices: Vec<usize>,
    pub(crate) image_infos: Vec<ImageInfo>,
    pub(crate) decoded_regions: Vec<Rect>,
    pub(crate) reports: Vec<DecodeReport>,
    pub(crate) errors: Vec<IndexedBatchError>,
}

impl CpuBatchIntoResult {
    /// Successful source indices. Their fixed destination slots are unchanged by failures.
    #[must_use]
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }

    /// Parsed metadata for successful images.
    #[must_use]
    pub fn image_infos(&self) -> &[ImageInfo] {
        &self.image_infos
    }

    /// Decoded regions for successful images.
    #[must_use]
    pub fn decoded_regions(&self) -> &[Rect] {
        &self.decoded_regions
    }

    /// Route reports for successful images.
    #[must_use]
    pub fn reports(&self) -> &[DecodeReport] {
        &self.reports
    }

    /// Input-local decode failures.
    #[must_use]
    pub fn errors(&self) -> &[IndexedBatchError] {
        &self.errors
    }
}

/// Monotonic preparation, execution, and retained-workspace counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuBatchDiagnostics {
    /// Calls to batch preparation.
    pub preparation_calls: u64,
    /// Inputs represented by preparation calls.
    pub prepared_inputs: u64,
    /// Identity-and-request preparation cache hits, including same-call duplicates.
    pub preparation_cache_hits: u64,
    /// Unique inputs parsed and planned.
    pub preparation_cache_misses: u64,
    /// Calls to owned CPU batch execution.
    pub decode_calls: u64,
    /// Images written directly into their final dense typed allocation.
    pub direct_dense_images: u64,
    /// Images materialized through the general typed fallback.
    pub fallback_materialized_images: u64,
    /// Reuses of a retained coefficient arena with sufficient capacity.
    pub coefficient_workspace_reuses: u64,
    /// Aggregate bytes retained by coefficient arenas across workers.
    pub retained_coefficient_bytes: u64,
    /// Reuses of retained component raster and transform scratch.
    pub reconstruction_workspace_reuses: u64,
    /// Aggregate bytes retained by reconstruction scratch across workers.
    pub retained_reconstruction_bytes: u64,
    /// Reuses of retained NHWC-to-NCHW typed layout scratch.
    pub layout_workspace_reuses: u64,
    /// Aggregate bytes retained by typed layout scratch across workers.
    pub retained_layout_bytes: u64,
    /// Samples copied while compacting groups after an input-local failure.
    pub output_compaction_copied_samples: u64,
}

impl CpuBatchDecodeResult {
    /// Successful groups in prepared group order.
    #[must_use]
    pub fn groups(&self) -> &[CpuBatchGroup] {
        &self.groups
    }

    /// Preparation and decode failures in original input order.
    #[must_use]
    pub fn errors(&self) -> &[IndexedBatchError] {
        &self.errors
    }
}

/// Common synchronous boundary implemented by persistent batch sessions.
pub trait BatchDecoder {
    /// Backend-specific successful output.
    type Output;
    /// Backend-specific infrastructure or execution error.
    type Error: From<BatchInfrastructureError>;

    /// Retained preparation and aggregate resource policy.
    fn options(&self) -> BatchDecodeOptions;

    /// Parse and group owned compressed inputs.
    fn prepare_batch(&self, inputs: Vec<EncodedImage>) -> Result<PreparedBatch, Self::Error>;

    /// Regroup previously prepared images without reparsing them.
    fn prepare_prepared_images(
        &self,
        images: Vec<PreparedImage>,
    ) -> Result<PreparedBatch, Self::Error>;

    /// Decode a reusable prepared batch.
    fn decode_prepared(&mut self, prepared: &PreparedBatch) -> Result<Self::Output, Self::Error>;

    /// Prepare and decode one owned batch.
    fn decode_batch(&mut self, inputs: Vec<EncodedImage>) -> Result<Self::Output, Self::Error> {
        let prepared = self.prepare_batch(inputs)?;
        self.decode_prepared(&prepared)
    }
}
