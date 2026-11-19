//! Harnesses shared by fuzz targets and the deterministic drivers that run
//! them under `cargo test`, where no fuzzing engine is needed.

mod encode_roundtrip;

pub use encode_roundtrip::encode_roundtrip;
