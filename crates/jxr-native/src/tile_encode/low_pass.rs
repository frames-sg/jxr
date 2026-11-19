//! Low-pass residual formation and block writing shared by colour formats.

use jxr_core::{ChromaSampling, PredictionMode};

use crate::{
    bit_writer::BitWriter,
    entropy::{
        AcVlcState, AdaptiveLpScan, ComponentClass,
        encode::{
            check_magnitudes, encode_lp_block, encode_lp_refinement, encode_lp_refinement_at,
            encode_ordered_block, vlc_part,
        },
    },
    tile_decode::{
        ComponentGeometry,
        spatial::{MacroblockPosition, PredictionDirection, PredictionRecord, resolve_low_pass},
        yuv::{
            chroma_refinement_positions, joint_chroma_block, joint_chroma_count, joint_chroma_start,
        },
    },
};

use super::TileEncodeError;

/// Low-pass residuals of one component and their VLC-coded parts.
#[derive(Debug, Clone, Copy)]
pub(super) struct LowPass {
    /// Prediction residuals; element zero is the DC position and is zero.
    residuals: [i32; 16],
    /// The residuals with their model bits removed.
    levels: [i32; 16],
    model_bits: u8,
}

impl LowPass {
    /// Subtracts the prediction a decoder adds for the macroblock at
    /// `position` of a component whose blocks are `geometry`, given the DC
    /// prediction mode already selected for it.
    pub(super) fn new(
        low: &[i32; 16],
        coded: &[PredictionRecord],
        (position, geometry): (MacroblockPosition, ComponentGeometry),
        dc_mode: PredictionMode,
        model_bits: u8,
    ) -> Result<Self, TileEncodeError> {
        let mut residuals = *low;
        residuals[0] = 0;
        // Every macroblock a tile writes is quantized with the first
        // low-pass quantizer.
        resolve_low_pass(
            &mut residuals,
            coded,
            position,
            (dc_mode, 0),
            geometry,
            PredictionDirection::Remove,
        )?;
        check_magnitudes(&residuals)?;
        Ok(Self {
            residuals,
            levels: residuals.map(|residual| vlc_part(residual, model_bits)),
            model_bits,
        })
    }

    /// Whether any coefficient has a VLC-coded part, so the block is coded.
    pub(super) fn is_coded(&self) -> bool {
        self.levels[1..].iter().any(|&level| level != 0)
    }

    /// Writes the block when coded, then every coefficient's refinement.
    /// Returns the number of VLC-coded coefficients.
    pub(super) fn write(
        &self,
        writer: &mut BitWriter,
        class: ComponentClass,
        vlc: &mut AcVlcState,
        scan: &mut AdaptiveLpScan,
    ) -> Result<i32, TileEncodeError> {
        let non_zero = if self.is_coded() {
            encode_lp_block(writer, class, 1, vlc, scan, &self.levels)?
        } else {
            0
        };
        encode_lp_refinement(writer, &self.residuals, self.model_bits)?;
        Ok(i32::from(non_zero))
    }
}

/// Low-pass residuals of the two chroma components of a YUV 4:2:0 or 4:2:2
/// macroblock, whose VLC-coded parts share one block.
#[derive(Debug, Clone, Copy)]
pub(super) struct JointChromaLowPass {
    /// Prediction residuals of U and of V; element zero is the DC position
    /// and is zero.
    residuals: [[i32; 16]; 2],
    /// The VLC-coded parts of both components in the joint block's order.
    levels: [i32; 14],
    sampling: ChromaSampling,
    model_bits: u8,
}

impl JointChromaLowPass {
    /// Subtracts the prediction a decoder adds to each component of the
    /// macroblock at `position`, whose blocks are `geometry`, given the DC
    /// prediction mode already selected for it. `low` and `coded` hold U and
    /// then V.
    pub(super) fn new(
        low: [&[i32; 16]; 2],
        coded: [&[PredictionRecord]; 2],
        (position, geometry): (MacroblockPosition, ComponentGeometry),
        dc_mode: PredictionMode,
        model_bits: u8,
        sampling: ChromaSampling,
    ) -> Result<Self, TileEncodeError> {
        let mut residuals = [*low[0], *low[1]];
        for (residuals, coded) in residuals.iter_mut().zip(coded) {
            residuals[0] = 0;
            resolve_low_pass(
                residuals,
                coded,
                position,
                (dc_mode, 0),
                geometry,
                PredictionDirection::Remove,
            )?;
            check_magnitudes(residuals)?;
        }
        let parts = residuals.map(|component| component.map(|value| vlc_part(value, model_bits)));
        Ok(Self {
            residuals,
            levels: joint_chroma_block([&parts[0], &parts[1]], sampling),
            sampling,
            model_bits,
        })
    }

    /// Whether either component has a VLC-coded part, so the joint block is
    /// coded.
    pub(super) fn is_coded(&self) -> bool {
        self.levels.iter().any(|&level| level != 0)
    }

    /// Writes the joint block when coded, then every coefficient's
    /// refinement, U before V at each position. Returns the number of
    /// VLC-coded coefficients.
    pub(super) fn write(
        &self,
        writer: &mut BitWriter,
        vlc: &mut AcVlcState,
    ) -> Result<i32, TileEncodeError> {
        let non_zero = if self.is_coded() {
            encode_ordered_block(
                writer,
                ComponentClass::Chroma,
                joint_chroma_start(self.sampling),
                vlc,
                &self.levels[..joint_chroma_count(self.sampling)],
            )?
        } else {
            0
        };
        for position in chroma_refinement_positions(self.sampling) {
            for residuals in &self.residuals {
                encode_lp_refinement_at(
                    writer,
                    residuals,
                    core::slice::from_ref(position),
                    self.model_bits,
                )?;
            }
        }
        Ok(i32::from(non_zero))
    }
}
