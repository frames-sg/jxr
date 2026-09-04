// SPDX-License-Identifier: MIT OR Apache-2.0

use core::mem::{offset_of, size_of};

use j2k_core::accelerator::GpuAbi;
use jxr_core::{BandPresence, PredictionMode};

use crate::{MetalError, plan::MetalPlaneInput};

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

// SAFETY: Eight consecutive u32 fields have the asserted offsets and size, no
// padding, and every bit pattern is valid for the shader-side uint fields.
unsafe impl GpuAbi for JxrMacroblockAbi {
    const NAME: &'static str = "JxrMacroblockAbi";
}

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

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct JxrBatchDispatchAbi {
    pub(crate) image_count: u32,
    pub(crate) plane_count: u32,
    pub(crate) plane_index: u32,
    pub(crate) reserved: u32,
}

// SAFETY: Four consecutive u32 fields are padding-free and accept every bit pattern.
unsafe impl GpuAbi for JxrBatchDispatchAbi {
    const NAME: &'static str = "JxrBatchDispatchAbi";
}

const _: () =
    assert!(size_of::<JxrBatchDispatchAbi>() == jxr_math::tables::ABI_JXRBATCHDISPATCHABI_SIZE);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct JxrOverlapWorkAbi {
    pub(crate) first: u32,
    pub(crate) second: u32,
    pub(crate) kind: u32,
    pub(crate) reserved: u32,
}

// SAFETY: Four consecutive u32 fields are padding-free and accept every bit pattern.
unsafe impl GpuAbi for JxrOverlapWorkAbi {
    const NAME: &'static str = "JxrOverlapWorkAbi";
}

const _: () =
    assert!(size_of::<JxrOverlapWorkAbi>() == jxr_math::tables::ABI_JXROVERLAPWORKABI_SIZE);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct JxrSamplePlaneAbi {
    pub(crate) sample_offset: u32,
    pub(crate) origin_x: u32,
    pub(crate) origin_y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) alpha: u32,
}

// SAFETY: Six consecutive u32 fields are padding-free and accept every bit pattern.
unsafe impl GpuAbi for JxrSamplePlaneAbi {
    const NAME: &'static str = "JxrSamplePlaneAbi";
}

const _: () =
    assert!(size_of::<JxrSamplePlaneAbi>() == jxr_math::tables::ABI_JXRSAMPLEPLANEABI_SIZE);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct JxrSurfacePlaneAbi {
    pub(crate) byte_offset: u32,
    pub(crate) row_stride_bytes: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) channels: u32,
    pub(crate) reserved: u32,
}

// SAFETY: Six consecutive u32 fields are padding-free and accept every bit pattern.
unsafe impl GpuAbi for JxrSurfacePlaneAbi {
    const NAME: &'static str = "JxrSurfacePlaneAbi";
}

const _: () =
    assert!(size_of::<JxrSurfacePlaneAbi>() == jxr_math::tables::ABI_JXRSURFACEPLANEABI_SIZE);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct JxrOutputAbi {
    pub(crate) output_width: u32,
    pub(crate) output_height: u32,
    pub(crate) crop_x: u32,
    pub(crate) crop_y: u32,
    pub(crate) component_count: u32,
    pub(crate) alpha_plane: u32,
    pub(crate) internal_color: u32,
    pub(crate) output_color: u32,
    pub(crate) chroma_sampling: u32,
    pub(crate) chroma_centering_x: u32,
    pub(crate) chroma_centering_y: u32,
    pub(crate) bit_depth: u32,
    pub(crate) channel_layout: u32,
    pub(crate) channels: u32,
    pub(crate) scaled: u32,
    pub(crate) alpha_scaled: u32,
    pub(crate) premultiply_alpha: u32,
    pub(crate) red_blue_not_swapped: u32,
    pub(crate) shift_bits: u32,
    pub(crate) alpha_shift_bits: u32,
    pub(crate) mantissa_length: u32,
    pub(crate) alpha_mantissa_length: u32,
    pub(crate) exponent_bias_bits: u32,
    pub(crate) alpha_exponent_bias_bits: u32,
    pub(crate) output_plane: u32,
    pub(crate) output_plane_count: u32,
    pub(crate) bit_black: u32,
    pub(crate) reserved0: u32,
}

// SAFETY: Twenty-eight consecutive u32 fields are padding-free, with terminal
// offset and total size asserted, and all bit patterns are valid.
unsafe impl GpuAbi for JxrOutputAbi {
    const NAME: &'static str = "JxrOutputAbi";
}

const _: () = {
    assert!(
        offset_of!(JxrOutputAbi, reserved0) == jxr_math::tables::ABI_JXROUTPUTABI_RESERVED0_OFFSET
    );
    assert!(size_of::<JxrOutputAbi>() == jxr_math::tables::ABI_JXROUTPUTABI_SIZE);
};

// SAFETY: Twelve consecutive u32 fields have no padding, as established by
// the size and terminal offset assertions, and all bit patterns are valid.
unsafe impl GpuAbi for JxrPlaneAbi {
    const NAME: &'static str = "JxrPlaneAbi";
}

const _: () = {
    assert!(
        offset_of!(JxrPlaneAbi, scale_after_first_transform)
            == jxr_math::tables::ABI_JXRPLANEABI_SCALE_AFTER_FIRST_TRANSFORM_OFFSET
    );
    assert!(size_of::<JxrPlaneAbi>() == jxr_math::tables::ABI_JXRPLANEABI_SIZE);
};

impl JxrPlaneAbi {
    pub(crate) fn from_plan(plane: MetalPlaneInput) -> Result<Self, MetalError> {
        let u32_value = |value: usize, reason| {
            u32::try_from(value).map_err(|_| MetalError::InvalidPlan { reason })
        };
        Ok(Self {
            macroblock_offset: u32_value(
                plane.macroblock_offset,
                "macroblock offset exceeds the Metal ABI",
            )?,
            macroblock_count: u32_value(
                plane.macroblock_count,
                "macroblock count exceeds the Metal ABI",
            )?,
            block_columns: u32::from(plane.block_columns),
            block_rows: u32::from(plane.block_rows),
            macroblock_origin_x: plane.macroblock_origin_x,
            macroblock_origin_y: plane.macroblock_origin_y,
            low_offset: u32_value(plane.low_offset, "low-pass offset exceeds the Metal ABI")?,
            low_width: plane
                .macroblocks_x
                .checked_mul(u32::from(plane.block_columns))
                .ok_or(MetalError::InvalidPlan {
                    reason: "low-pass row width exceeds the Metal ABI",
                })?,
            sample_offset: u32_value(plane.sample_offset, "sample offset exceeds the Metal ABI")?,
            sample_width: plane.sample_width,
            sample_height: plane.sample_height,
            scale_after_first_transform: u32::from(plane.scale_after_first_transform),
        })
    }
}

pub(crate) fn macroblock_abi(
    arena: &jxr_core::CoefficientArena,
) -> Result<Vec<JxrMacroblockAbi>, MetalError> {
    macroblock_abi_metadata(&arena.macroblocks)
}

pub(crate) fn macroblock_abi_metadata(
    macroblocks: &jxr_core::MacroblockMetadata,
) -> Result<Vec<JxrMacroblockAbi>, MetalError> {
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
                        return Err(MetalError::InvalidPlan {
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
