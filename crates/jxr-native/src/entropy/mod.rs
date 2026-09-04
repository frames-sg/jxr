//! Bounds-checked JPEG XR entropy decoding primitives.

mod adaptive;
mod bit_reader;
mod coefficients;
mod error;
mod model;
mod refinement;
mod scan;
mod vlc;

pub use adaptive::{AcVlcState, DcVlcState, TileEntropyState};
pub use bit_reader::PacketBitReader;
pub use coefficients::{
    ComponentClass, DecodedBlock, FrequencyBand, RunLevel, decode_ac_block, decode_dc_coefficient,
};
pub use error::EntropyError;
pub use model::{CoefficientModel, ColourModel};
pub use refinement::{
    decode_flex, decode_flex_block, decode_lp_refinement, decode_lp_refinement_at,
};
pub use scan::{AdaptiveHpScan, AdaptiveLpScan, HpScanDirection};
