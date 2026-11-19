//! What sets one internal colour format's macroblock syntax apart.
//!
//! Every internal colour format codes its macroblocks with the same syntax
//! elements, so one reader ([`super::plane::PlaneReader`]) and one writer
//! (`tile_encode`'s tile writer) serve them all. They differ only in the
//! decisions [`ComponentSyntax`] makes: how many components a macroblock
//! has, whether one code carries the presence of every component in a band
//! or each component has a flag of its own, which components decide the
//! prediction modes, how the coefficient models weigh the components, and
//! how many blocks each component has. YUV 4:2:0 and 4:2:2 also code their
//! two chroma components' low-pass coefficients as one block.
//!
//! An integrated alpha plane is a Y-only plane of its own.

use jxr_core::{ChromaSampling, ColorFormat, PredictionMode};
use jxr_math::prediction::{PREDICTED_FROM_LEFT, PREDICTED_FROM_TOP};

use crate::entropy::{ColourModel, ComponentClass};

use super::{
    ComponentGeometry, TileDecodeError, component_geometry,
    spatial::{MacroblockPosition, PredictionRecord, spatial_macroblock},
};

/// Most components the macroblock syntax of one plane can carry.
pub(crate) const MAX_COMPONENTS: usize = 16;

/// The macroblock syntax family of an internal colour format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ComponentSyntax {
    /// One Y component.
    Luma,
    /// Y, U, and V with jointly coded presence patterns, the chroma
    /// components at the given sampling.
    Yuv(ChromaSampling),
    /// Y, U, V, and K, each with its own presence flags.
    YuvK,
    /// Independent components, each with its own presence flags.
    NComponent(u8),
}

impl ComponentSyntax {
    /// The syntax of a plane coded in `colour`.
    pub(crate) fn new(colour: ColorFormat) -> Result<Self, TileDecodeError> {
        match colour {
            ColorFormat::Luma => Ok(Self::Luma),
            ColorFormat::Yuv(sampling) => Ok(Self::Yuv(sampling)),
            ColorFormat::YuvK => Ok(Self::YuvK),
            ColorFormat::NComponent(count) => u8::try_from(count)
                .ok()
                .filter(|&count| (2..=MAX_COMPONENTS).contains(&usize::from(count)))
                .map(Self::NComponent)
                .ok_or(TileDecodeError::Unsupported("N-component count")),
            _ => Err(TileDecodeError::Unsupported("internal colour format")),
        }
    }

    pub(crate) const fn components(self) -> usize {
        match self {
            Self::Luma => 1,
            Self::Yuv(_) => 3,
            Self::YuvK => 4,
            Self::NComponent(count) => count as usize,
        }
    }

    /// Whether one code carries the presence of all components in each band,
    /// instead of one flag or pattern per component.
    pub(crate) const fn joint_patterns(self) -> bool {
        matches!(self, Self::Yuv(_))
    }

    /// The sampling of chroma components whose low-pass coefficients share
    /// one jointly coded block: those of YUV 4:2:0 and 4:2:2.
    pub(crate) const fn joint_chroma(self) -> Option<ChromaSampling> {
        match self {
            Self::Yuv(sampling @ (ChromaSampling::Cs420 | ChromaSampling::Cs422)) => Some(sampling),
            _ => None,
        }
    }

    /// The chroma sampling of the plane: that of a YUV plane, and 4:4:4 for
    /// the other formats, whose components all have full resolution.
    pub(crate) const fn sampling(self) -> ChromaSampling {
        match self {
            Self::Yuv(sampling) => sampling,
            _ => ChromaSampling::Cs444,
        }
    }

    /// How the coefficient models weigh the components.
    pub(crate) const fn colour_model(self) -> ColourModel {
        match self {
            Self::Luma => ColourModel::LumaOnly,
            Self::Yuv(ChromaSampling::Cs420) => ColourModel::Yuv420,
            Self::Yuv(ChromaSampling::Cs422) => ColourModel::Yuv422,
            Self::Yuv(ChromaSampling::Cs444) => ColourModel::Other { components: 3 },
            Self::YuvK => ColourModel::Other { components: 4 },
            Self::NComponent(components) => ColourModel::Other { components },
        }
    }

    /// Code tables used by the DC coefficient of `component`. Formats with
    /// per-component DC flags code every component with the luma tables.
    pub(crate) const fn dc_class(self, component: usize) -> ComponentClass {
        if self.joint_patterns() {
            ComponentClass::of(component)
        } else {
            ComponentClass::Luma
        }
    }

    /// The blocks of one macroblock of `component`.
    pub(crate) fn geometry(self, component: usize) -> ComponentGeometry {
        let colour = match self {
            Self::Luma => ColorFormat::Luma,
            Self::Yuv(sampling) => ColorFormat::Yuv(sampling),
            Self::YuvK => ColorFormat::YuvK,
            Self::NComponent(count) => ColorFormat::NComponent(u16::from(count)),
        };
        component_geometry(colour, component)
    }

    /// The components whose DC values select the DC prediction mode, and
    /// the weight of the first of them; each other one weighs one.
    const fn dc_mode_components(self) -> (usize, u32) {
        match self {
            Self::Luma | Self::NComponent(_) => (1, 1),
            Self::Yuv(ChromaSampling::Cs420) => (3, 8),
            Self::Yuv(ChromaSampling::Cs422) => (3, 4),
            Self::Yuv(ChromaSampling::Cs444) | Self::YuvK => (3, 2),
        }
    }

    /// The DC prediction mode the components of the macroblock at
    /// `position` share, and each component's predictor, from the DC values
    /// of the macroblocks before it in `history`, one per component.
    ///
    /// Both depend only on macroblocks already coded in the tile, so a
    /// decoder adds the predictors to parsed residuals and an encoder
    /// subtracts them from the values it is about to write.
    pub(crate) fn dc_prediction(
        self,
        history: &[Vec<PredictionRecord>],
        position: MacroblockPosition,
    ) -> Result<(PredictionMode, [i32; MAX_COMPONENTS]), TileDecodeError> {
        let MacroblockPosition { width, x, y } = position;
        let dc = |component: usize, x: usize, y: usize| {
            spatial_macroblock(&history[component], width, x, y).map(|record| record.dc_low_pass[0])
        };
        let mode = match (x == 0, y == 0) {
            (true, true) => PredictionMode::None,
            (true, false) => PredictionMode::FromTop,
            (false, true) => PredictionMode::FromLeft,
            (false, false) => {
                let (deciding, first_weight) = self.dc_mode_components();
                let mut horizontal = 0_u32;
                let mut vertical = 0_u32;
                for component in 0..deciding {
                    let weight = if component == 0 { first_weight } else { 1 };
                    let top_left = dc(component, x - 1, y - 1)?;
                    let left = top_left.abs_diff(dc(component, x - 1, y)?);
                    let top = top_left.abs_diff(dc(component, x, y - 1)?);
                    horizontal = horizontal.saturating_add(left.saturating_mul(weight));
                    vertical = vertical.saturating_add(top.saturating_mul(weight));
                }
                if horizontal.saturating_mul(4) < vertical {
                    PredictionMode::FromTop
                } else if vertical.saturating_mul(4) < horizontal {
                    PredictionMode::FromLeft
                } else {
                    PredictionMode::FromTopLeft
                }
            }
        };
        let mut predictors = [0_i32; MAX_COMPONENTS];
        for (component, predictor) in predictors.iter_mut().enumerate().take(self.components()) {
            *predictor = match mode {
                PredictionMode::None => 0,
                PredictionMode::FromLeft => dc(component, x - 1, y)?,
                PredictionMode::FromTop => dc(component, x, y - 1)?,
                PredictionMode::FromTopLeft => {
                    let sum =
                        i64::from(dc(component, x - 1, y)?) + i64::from(dc(component, x, y - 1)?);
                    // Subsampled chroma rounds its neighbours' mean up.
                    let rounding = i64::from(component != 0 && self.joint_chroma().is_some());
                    i32::try_from((sum + rounding) >> 1)
                        .map_err(|_| TileDecodeError::ArithmeticOverflow("DC prediction average"))?
                }
            };
        }
        Ok((mode, predictors))
    }

    /// The high-pass prediction direction the components of a macroblock
    /// share, selected from its own DC and low-pass coefficients: `low`
    /// gives those of each component.
    ///
    /// Luma, and the first component of an N-component plane, weigh the
    /// coefficients each direction would predict. The chroma components of
    /// YUV and YUVK add their first horizontal and vertical frequencies, at
    /// the low-pass positions of their sampling.
    pub(crate) fn high_pass_mode<'r>(self, low: impl Fn(usize) -> &'r [i32; 16]) -> PredictionMode {
        let energy = |low: &[i32; 16], positions: &[usize]| {
            positions.iter().fold(0_u32, |sum, &position| {
                sum.saturating_add(low[position].unsigned_abs())
            })
        };
        let luma = low(0);
        let mut horizontal = energy(luma, &PREDICTED_FROM_TOP);
        let mut vertical = energy(luma, &PREDICTED_FROM_LEFT);
        let chroma: Option<(&[usize], &[usize])> = match self {
            Self::Luma | Self::NComponent(_) => None,
            Self::Yuv(ChromaSampling::Cs420) => Some((&[1], &[2])),
            Self::Yuv(ChromaSampling::Cs422) => Some((&[1, 5], &[2, 6])),
            Self::Yuv(ChromaSampling::Cs444) | Self::YuvK => Some((&[1], &[4])),
        };
        if let Some((across, down)) = chroma {
            for component in 1..3 {
                horizontal = horizontal.saturating_add(energy(low(component), across));
                vertical = vertical.saturating_add(energy(low(component), down));
            }
        }
        if horizontal.saturating_mul(4) < vertical {
            PredictionMode::FromLeft
        } else if vertical.saturating_mul(4) < horizontal {
            PredictionMode::FromTop
        } else {
            PredictionMode::None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_yuv_codes_its_components_jointly() {
        for (sampling, model, joint_chroma) in [
            (
                ChromaSampling::Cs444,
                ColourModel::Other { components: 3 },
                None,
            ),
            (
                ChromaSampling::Cs422,
                ColourModel::Yuv422,
                Some(ChromaSampling::Cs422),
            ),
            (
                ChromaSampling::Cs420,
                ColourModel::Yuv420,
                Some(ChromaSampling::Cs420),
            ),
        ] {
            let syntax = ComponentSyntax::Yuv(sampling);
            assert!(syntax.joint_patterns());
            assert_eq!(syntax.dc_class(2), ComponentClass::Chroma);
            assert_eq!(syntax.colour_model(), model);
            // Only subsampled chroma shares one low-pass block.
            assert_eq!(syntax.joint_chroma(), joint_chroma);
            assert_eq!(syntax.sampling(), sampling);
        }
        for syntax in [
            ComponentSyntax::Luma,
            ComponentSyntax::YuvK,
            ComponentSyntax::NComponent(6),
        ] {
            assert!(!syntax.joint_patterns());
            assert_eq!(syntax.joint_chroma(), None);
            assert_eq!(syntax.sampling(), ChromaSampling::Cs444);
            // Per-component DC flags come with luma DC tables throughout.
            assert_eq!(syntax.dc_class(0), ComponentClass::Luma);
            assert_eq!(syntax.dc_class(3), ComponentClass::Luma);
        }
        assert_eq!(ComponentSyntax::Luma.colour_model(), ColourModel::LumaOnly);
        assert_eq!(
            ComponentSyntax::YuvK.colour_model(),
            ColourModel::Other { components: 4 }
        );
        assert_eq!(
            ComponentSyntax::NComponent(6).colour_model(),
            ColourModel::Other { components: 6 }
        );
    }

    #[test]
    fn every_component_count_the_syntax_carries_has_a_syntax() {
        assert_eq!(
            ComponentSyntax::new(ColorFormat::NComponent(16)),
            Ok(ComponentSyntax::NComponent(16))
        );
        assert_eq!(ComponentSyntax::NComponent(16).components(), MAX_COMPONENTS);
        for colour in [
            ColorFormat::NComponent(1),
            ColorFormat::NComponent(17),
            ColorFormat::Rgb,
            ColorFormat::Cmyk,
        ] {
            assert!(matches!(
                ComponentSyntax::new(colour),
                Err(TileDecodeError::Unsupported(_))
            ));
        }
    }

    /// Prediction history of a tile two macroblocks wide holding, for each
    /// component, the DC values `dc` in raster order.
    fn history(dc: &[&[i32]]) -> Vec<Vec<PredictionRecord>> {
        dc.iter()
            .map(|values| {
                values
                    .iter()
                    .map(|&value| {
                        let mut dc_low_pass = [0_i32; 16];
                        dc_low_pass[0] = value;
                        PredictionRecord {
                            dc_low_pass,
                            lp_qp_index: 0,
                        }
                    })
                    .collect()
            })
            .collect()
    }

    const fn at(x: usize, y: usize) -> MacroblockPosition {
        MacroblockPosition { width: 2, x, y }
    }

    #[test]
    fn tile_edges_fix_the_dc_prediction_mode() {
        let decoded = history(&[&[10, 500, -90][..]; 4]);
        for syntax in [
            ComponentSyntax::Luma,
            ComponentSyntax::YuvK,
            ComponentSyntax::NComponent(4),
        ] {
            for (position, mode) in [
                (at(0, 0), PredictionMode::None),
                (at(1, 0), PredictionMode::FromLeft),
                (at(0, 1), PredictionMode::FromTop),
            ] {
                assert_eq!(syntax.dc_prediction(&decoded, position).unwrap().0, mode);
            }
        }
    }

    #[test]
    fn the_formats_weigh_different_components_to_select_the_dc_mode() {
        // Raster order: top-left, top, left. The first component changes
        // going down the tile and the next two change going across it, more
        // strongly.
        let decoded = history(&[&[0, 0, 40], &[0, 300, 0], &[0, 300, 0], &[7, 7, 7]]);
        // The first component alone changes only downwards, so the left
        // neighbour predicts it and the one above does not.
        let mode = |syntax: ComponentSyntax| syntax.dc_prediction(&decoded, at(1, 1)).unwrap().0;
        assert_eq!(
            mode(ComponentSyntax::NComponent(4)),
            PredictionMode::FromLeft
        );
        // Y counts twice and U and V once each: 80 downwards against 600
        // across, so the neighbour above predicts better.
        assert_eq!(mode(ComponentSyntax::YuvK), PredictionMode::FromTop);
        // 4:2:0 counts Y eight times: 320 against 600 is still not four to
        // one, so both neighbours predict.
        let yuv = history(&[&[0, 0, 40], &[0, 300, 0], &[0, 300, 0]]);
        let mode = |syntax: ComponentSyntax| syntax.dc_prediction(&yuv, at(1, 1)).unwrap().0;
        assert_eq!(
            mode(ComponentSyntax::Yuv(ChromaSampling::Cs420)),
            PredictionMode::FromTopLeft
        );
        assert_eq!(
            mode(ComponentSyntax::Yuv(ChromaSampling::Cs444)),
            PredictionMode::FromTop
        );
        // Comparable gradients select the average of both neighbours.
        let balanced = history(&[&[0, 30, 40][..]; 4]);
        for syntax in [ComponentSyntax::YuvK, ComponentSyntax::NComponent(4)] {
            assert_eq!(
                syntax.dc_prediction(&balanced, at(1, 1)).unwrap().0,
                PredictionMode::FromTopLeft
            );
        }
    }

    #[test]
    fn dc_predictors_average_both_neighbours_and_round_subsampled_chroma_up() {
        // Top-left, top, left: the top and left neighbours sum to an odd value.
        let decoded = history(&[&[0, 21, -40], &[0, 21, -40], &[0, 21, -40]]);
        let predictors = |syntax: ComponentSyntax| {
            let (mode, predictors) = syntax.dc_prediction(&decoded, at(1, 1)).unwrap();
            assert_eq!(mode, PredictionMode::FromTopLeft);
            [predictors[0], predictors[1], predictors[2]]
        };
        // Halving rounds toward negative infinity, but subsampled chroma
        // adds one first.
        assert_eq!(predictors(ComponentSyntax::NComponent(3)), [-10; 3]);
        assert_eq!(
            predictors(ComponentSyntax::Yuv(ChromaSampling::Cs444)),
            [-10; 3]
        );
        for sampling in [ChromaSampling::Cs420, ChromaSampling::Cs422] {
            assert_eq!(
                predictors(ComponentSyntax::Yuv(sampling)),
                [-10, -9, -9],
                "{sampling:?}"
            );
        }
        // A single neighbour predicts every component with its own value.
        let (mode, predictors) = ComponentSyntax::NComponent(3)
            .dc_prediction(&decoded, at(1, 0))
            .unwrap();
        assert_eq!(
            (mode, &predictors[..3]),
            (PredictionMode::FromLeft, &[0; 3][..])
        );
    }

    #[test]
    fn the_high_pass_direction_weighs_the_chroma_of_yuv_and_yuvk_only() {
        // Luma has energy across and down in equal measure; the chroma
        // components have it only down.
        let mut luma = [0_i32; 16];
        luma[1] = 10;
        luma[4] = 10;
        let mut chroma = [0_i32; 16];
        for position in [2, 4, 6] {
            chroma[position] = 100;
        }
        let low = [luma, chroma, chroma];
        let mode = |syntax: ComponentSyntax| syntax.high_pass_mode(|component| &low[component]);
        assert_eq!(mode(ComponentSyntax::Luma), PredictionMode::None);
        assert_eq!(mode(ComponentSyntax::NComponent(3)), PredictionMode::None);
        for syntax in [
            ComponentSyntax::YuvK,
            ComponentSyntax::Yuv(ChromaSampling::Cs444),
            ComponentSyntax::Yuv(ChromaSampling::Cs422),
            ComponentSyntax::Yuv(ChromaSampling::Cs420),
        ] {
            assert_eq!(mode(syntax), PredictionMode::FromLeft, "{syntax:?}");
        }
        // Energy across alone predicts from the top.
        let mut across = [0_i32; 16];
        across[3] = -9;
        assert_eq!(
            ComponentSyntax::Luma.high_pass_mode(|_| &across),
            PredictionMode::FromTop
        );
    }
}
