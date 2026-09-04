// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    Error, MpsGraphDecodeInput, MpsGraphPreparedBatch, MpsGraphPreparedGroup, MpsGraphTensorSpec,
};

/// Unavailable decoder on non-Apple-Silicon targets.
#[derive(Debug)]
pub struct MpsGraphBatchDecoder;

/// Unavailable input group on non-Apple-Silicon targets.
#[derive(Debug)]
pub struct MpsGraphInputGroup;

/// Unavailable batch result on non-Apple-Silicon targets.
#[derive(Debug)]
pub struct MpsGraphBatchDecode;

/// Unavailable graph program on non-Apple-Silicon targets.
#[derive(Debug)]
pub struct MpsGraphProgram;

/// Unavailable submitted graph run on non-Apple-Silicon targets.
#[derive(Debug)]
pub struct SubmittedMpsGraphRun;

/// Unavailable graph output on non-Apple-Silicon targets.
#[derive(Debug)]
pub struct MpsGraphRunOutput;

impl MpsGraphBatchDecoder {
    pub fn system_default() -> Result<Self, Error> {
        Err(Error::UnsupportedPlatform)
    }

    pub fn system_default_with_options(_options: jxr::BatchDecodeOptions) -> Result<Self, Error> {
        Err(Error::UnsupportedPlatform)
    }

    pub fn prepare(
        &self,
        _inputs: Vec<MpsGraphDecodeInput>,
    ) -> Result<MpsGraphPreparedBatch, Error> {
        Err(Error::UnsupportedPlatform)
    }

    pub fn prepare_batch(
        &self,
        _inputs: Vec<jxr::EncodedImage>,
    ) -> Result<jxr::PreparedBatch, Error> {
        Err(Error::UnsupportedPlatform)
    }

    pub fn prepare_prepared_images(
        &self,
        _images: Vec<jxr::PreparedImage>,
    ) -> Result<jxr::PreparedBatch, Error> {
        Err(Error::UnsupportedPlatform)
    }

    pub fn decode_batch(
        &mut self,
        _prepared: &jxr::PreparedBatch,
    ) -> Result<MpsGraphBatchDecode, Error> {
        Err(Error::UnsupportedPlatform)
    }

    pub fn decode_prepared_images(
        &mut self,
        _images: Vec<jxr::PreparedImage>,
    ) -> Result<MpsGraphBatchDecode, Error> {
        Err(Error::UnsupportedPlatform)
    }

    pub fn decode_prepared(
        &mut self,
        _prepared: &MpsGraphPreparedBatch,
    ) -> Result<MpsGraphBatchDecode, Error> {
        Err(Error::UnsupportedPlatform)
    }

    pub fn submit_prepared_group(
        &mut self,
        _program: &MpsGraphProgram,
        _group: &MpsGraphPreparedGroup,
    ) -> Result<SubmittedMpsGraphRun, Error> {
        Err(Error::UnsupportedPlatform)
    }

    pub fn run_prepared_group(
        &mut self,
        _program: &MpsGraphProgram,
        _group: &MpsGraphPreparedGroup,
    ) -> Result<MpsGraphRunOutput, Error> {
        Err(Error::UnsupportedPlatform)
    }
}

impl MpsGraphProgram {
    pub fn identity(_input_spec: MpsGraphTensorSpec) -> Result<Self, Error> {
        Err(Error::UnsupportedPlatform)
    }

    pub fn rgb8_nhwc_reference(
        _batch: usize,
        _height: usize,
        _width: usize,
    ) -> Result<Self, Error> {
        Err(Error::UnsupportedPlatform)
    }
}

impl SubmittedMpsGraphRun {
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        false
    }

    pub fn wait(self) -> Result<MpsGraphRunOutput, Error> {
        Err(Error::UnsupportedPlatform)
    }
}
