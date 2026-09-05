// SPDX-License-Identifier: MIT OR Apache-2.0

use core::mem::{offset_of, size_of};

use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use jxr_core::{BandPresence, PredictionMode};

use crate::{CudaError, plan::CudaPlaneInput};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct JxrMacroblockAbi {
    pub(crate) coefficient_offset: u32,
    pub(crate) quantizer_dc: u32,
    pub(crate) quantizer_low_pass: u32,
    pub(crate) quantizer_high_pass: u32,
    pub(crate) bands: u32,
    pub(crate) hp_prediction: u32,
    pub(crate) coded_x: u32,
    pub(crate) coded_y: u32,
}

// SAFETY: Eight consecutive u32 fields have the asserted offsets and size,
// contain no padding, and accept every bit pattern on both host and CUDA.
unsafe impl DeviceRepr for JxrMacroblockAbi {}
// SAFETY: Every field is a u32, so an all-zero value is valid.
unsafe impl ValidAsZeroBits for JxrMacroblockAbi {}

const _: () = {
    assert!(
        offset_of!(JxrMacroblockAbi, coefficient_offset)
            == jxr_math::tables::ABI_JXRMACROBLOCKABI_COEFFICIENT_OFFSET_OFFSET
    );
    assert!(
        offset_of!(JxrMacroblockAbi, coded_y)
            == jxr_math::tables::ABI_JXRMACROBLOCKABI_CODED_Y_OFFSET
    );
    assert!(size_of::<JxrMacroblockAbi>() == jxr_math::tables::ABI_JXRMACROBLOCKABI_SIZE);
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct JxrPlaneAbi {
    pub(crate) macroblock_offset: u32,
    pub(crate) macroblock_count: u32,
    pub(crate) block_columns: u32,
    pub(crate) block_rows: u32,
    pub(crate) macroblock_origin_x: u32,
    pub(crate) macroblock_origin_y: u32,
    pub(crate) low_offset: u32,
    pub(crate) low_width: u32,
    pub(crate) sample_offset: u32,
    pub(crate) sample_width: u32,
    pub(crate) sample_height: u32,
    pub(crate) scale_after_first_transform: u32,
}

// SAFETY: Twelve consecutive u32 fields have the asserted layout and every
// bit pattern is valid for the matching CUDA C structure.
unsafe impl DeviceRepr for JxrPlaneAbi {}
// SAFETY: Every field is a u32, so an all-zero value is valid.
unsafe impl ValidAsZeroBits for JxrPlaneAbi {}

const _: () = {
    assert!(
        offset_of!(JxrPlaneAbi, scale_after_first_transform)
            == jxr_math::tables::ABI_JXRPLANEABI_SCALE_AFTER_FIRST_TRANSFORM_OFFSET
    );
    assert!(size_of::<JxrPlaneAbi>() == jxr_math::tables::ABI_JXRPLANEABI_SIZE);
};

pub(crate) use jxr_core::device_plan::{
    OutputParameterWords as JxrOutputAbi, OverlapWork as JxrOverlapWorkAbi,
    SamplePlaneWords as JxrSamplePlaneAbi, SurfacePlaneWords as JxrSurfacePlaneAbi,
};

// Arrays contain consecutive initialized u32 words; upload traits are supplied
// by the backend's existing array implementation. Check the shader ABI here.
const _: () = {
    use jxr_core::device_plan::{
        OUTPUT_PLANE, SAMPLE_ALPHA, SAMPLE_OFFSET, SURFACE_HEIGHT, SURFACE_OFFSET, SURFACE_WIDTH,
    };
    use jxr_math::tables::{
        ABI_JXROUTPUTABI_OUTPUT_PLANE_OFFSET, ABI_JXROUTPUTABI_RESERVED0_OFFSET,
        ABI_JXROUTPUTABI_SIZE, ABI_JXROVERLAPWORKABI_SIZE, ABI_JXRSAMPLEPLANEABI_ALPHA_OFFSET,
        ABI_JXRSAMPLEPLANEABI_SAMPLE_OFFSET_OFFSET, ABI_JXRSAMPLEPLANEABI_SIZE,
        ABI_JXRSURFACEPLANEABI_BYTE_OFFSET_OFFSET, ABI_JXRSURFACEPLANEABI_HEIGHT_OFFSET,
        ABI_JXRSURFACEPLANEABI_SIZE, ABI_JXRSURFACEPLANEABI_WIDTH_OFFSET,
    };
    assert!(size_of::<JxrOverlapWorkAbi>() == ABI_JXROVERLAPWORKABI_SIZE);
    assert!(size_of::<JxrSamplePlaneAbi>() == ABI_JXRSAMPLEPLANEABI_SIZE);
    assert!(size_of::<JxrSurfacePlaneAbi>() == ABI_JXRSURFACEPLANEABI_SIZE);
    assert!(size_of::<JxrOutputAbi>() == ABI_JXROUTPUTABI_SIZE);
    assert!(SAMPLE_OFFSET * 4 == ABI_JXRSAMPLEPLANEABI_SAMPLE_OFFSET_OFFSET);
    assert!(SAMPLE_ALPHA * 4 == ABI_JXRSAMPLEPLANEABI_ALPHA_OFFSET);
    assert!(SURFACE_OFFSET * 4 == ABI_JXRSURFACEPLANEABI_BYTE_OFFSET_OFFSET);
    assert!(SURFACE_WIDTH * 4 == ABI_JXRSURFACEPLANEABI_WIDTH_OFFSET);
    assert!(SURFACE_HEIGHT * 4 == ABI_JXRSURFACEPLANEABI_HEIGHT_OFFSET);
    assert!(OUTPUT_PLANE * 4 == ABI_JXROUTPUTABI_OUTPUT_PLANE_OFFSET);
    assert!(27 * 4 == ABI_JXROUTPUTABI_RESERVED0_OFFSET);
};

impl JxrPlaneAbi {
    pub(crate) fn from_plan(plane: CudaPlaneInput) -> Result<Self, CudaError> {
        let u32_value = |value: usize, reason| {
            u32::try_from(value).map_err(|_| CudaError::InvalidPlan { reason })
        };
        Ok(Self {
            macroblock_offset: u32_value(
                plane.macroblock_offset,
                "macroblock offset exceeds the CUDA ABI",
            )?,
            macroblock_count: u32_value(
                plane.macroblock_count,
                "macroblock count exceeds the CUDA ABI",
            )?,
            block_columns: u32::from(plane.block_columns),
            block_rows: u32::from(plane.block_rows),
            macroblock_origin_x: plane.macroblock_origin_x,
            macroblock_origin_y: plane.macroblock_origin_y,
            low_offset: u32_value(plane.low_offset, "low-pass offset exceeds the CUDA ABI")?,
            low_width: plane
                .macroblocks_x
                .checked_mul(u32::from(plane.block_columns))
                .ok_or(CudaError::InvalidPlan {
                    reason: "low-pass row width exceeds the CUDA ABI",
                })?,
            sample_offset: u32_value(plane.sample_offset, "sample offset exceeds the CUDA ABI")?,
            sample_width: plane.sample_width,
            sample_height: plane.sample_height,
            scale_after_first_transform: u32::from(plane.scale_after_first_transform),
        })
    }
}

pub(crate) fn macroblock_abi(
    macroblocks: &jxr_core::MacroblockMetadata,
) -> Result<Vec<JxrMacroblockAbi>, CudaError> {
    macroblocks
        .coefficient_offsets
        .iter()
        .enumerate()
        .map(|(index, &coefficient_offset)| {
            let quantizers = macroblocks.quantizers[index];
            Ok(JxrMacroblockAbi {
                coefficient_offset,
                quantizer_dc: quantizers.dc,
                quantizer_low_pass: quantizers.low_pass,
                quantizer_high_pass: quantizers.high_pass,
                bands: match macroblocks.bands[index] {
                    BandPresence::DcOnly => 0,
                    BandPresence::NoHighPass => 1,
                    BandPresence::NoFlexbits => 2,
                    BandPresence::All => 3,
                },
                hp_prediction: match macroblocks.hp_predictions[index] {
                    PredictionMode::None => 0,
                    PredictionMode::FromLeft => 1,
                    PredictionMode::FromTop => 2,
                    PredictionMode::FromTopLeft => {
                        return Err(CudaError::InvalidPlan {
                            reason: "top-left high-pass prediction is invalid",
                        });
                    }
                },
                coded_x: macroblocks.coded_x[index],
                coded_y: macroblocks.coded_y[index],
            })
        })
        .collect()
}
