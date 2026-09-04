//! Device-neutral value types and contracts for JPEG XR decoders.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

mod container;
mod error;
mod format;
mod image;
mod output;
mod output_policy;
mod plan;
mod report;
mod request;
mod surface;

pub use container::{
    AnnexABitDepth, AnnexAChannelOrder, AnnexANumericKind, AnnexAPixelFamily, AnnexAPixelFormat,
    AnnexAPixelFormatDescriptor,
};
pub use error::{JxrError, JxrErrorKind, LimitKind};
pub use format::{
    AlphaMode, BandPresence, BitstreamMode, ChannelLayout, ChromaSampling, ColorFormat, Level,
    Orientation, OverlapMode, PixelFormat, Profile, SampleFormat, StorageKind,
};
pub use image::{ByteRange, ImageInfo, ImageMetadata, PlaneInfo, TileGrid};
pub use j2k_core::{BackendKind, BackendRequest, Rect};
pub use output::{DecodedImage, DecodedSamples, DecodedSamplesMut, PlaneDescriptor};
pub use output_policy::{AlphaFormatRequest, CropWindow, OutputBitDepth, OutputFormatRequest};
pub use plan::{
    CoefficientArena, CoefficientArenaDescriptor, CoefficientPlane, MacroblockMetadata, PlanePlan,
    PredictionMode, PreparedPlan, QuantizerSet, TileEdgeFlags, TilePlan,
};
pub use report::{DecodeReport, DecodeStage, FallbackReason, StageExecutor, StageReport};
pub use request::{AlphaHandling, DecodeLimits, DecodeRequest, DecodeScale};
pub use surface::{SurfaceLayout, SurfacePlaneLayout};
