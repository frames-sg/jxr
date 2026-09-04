//! Adaptive coefficient normalization from T.832 Tables 115 and 116.

use super::{EntropyError, FrequencyBand};

const LUMA_WEIGHT: [i32; 3] = [240, 12, 1];
const OTHER_WEIGHT: [[i32; 16]; 3] = [
    [
        0, 240, 120, 80, 60, 48, 40, 34, 30, 27, 24, 22, 20, 18, 17, 16,
    ],
    [0, 12, 6, 4, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1],
    [0, 16, 8, 5, 4, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1],
];
const SUBSAMPLED_WEIGHT: [[i32; 3]; 2] = [[120, 37, 2], [120, 18, 1]];

/// Colour information required by the coefficient-normalization model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColourModel {
    /// A single luma component with no chroma model.
    LumaOnly,
    /// YUV 4:2:0 internal colour format.
    Yuv420,
    /// YUV 4:2:2 internal colour format.
    Yuv422,
    /// Any other format, carrying its total component count.
    Other {
        /// Number of components, in the normative range 2 through 16 for this model.
        components: u8,
    },
}

/// Model bits and integrator state for one frequency band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoefficientModel {
    band: FrequencyBand,
    state: [i32; 2],
    bits: [u8; 2],
}

impl CoefficientModel {
    /// Creates a model with the values specified in T.832 Table 115.
    #[must_use]
    pub const fn new(band: FrequencyBand) -> Self {
        let bits = match band {
            FrequencyBand::Dc => 8,
            FrequencyBand::Lowpass => 4,
            FrequencyBand::Highpass => 0,
        };
        Self {
            band,
            state: [0, 0],
            bits: [bits, bits],
        }
    }

    /// Restores the first-macroblock-of-tile state.
    pub fn reset_tile(&mut self) {
        *self = Self::new(self.band);
    }

    /// Returns model bits for luma (`false`) or chroma (`true`).
    #[must_use]
    pub const fn bits(&self, chroma: bool) -> u8 {
        self.bits[chroma as usize]
    }

    /// Updates the model from the macroblock luma/chroma Laplacian means.
    pub fn update(
        &mut self,
        mut lap_mean: [i32; 2],
        colour: ColourModel,
    ) -> Result<(), EntropyError> {
        let band = self.band as usize;
        lap_mean[0] = lap_mean[0]
            .checked_mul(LUMA_WEIGHT[band])
            .ok_or(EntropyError::CoefficientOverflow)?;
        let model_count = match colour {
            ColourModel::LumaOnly => 1,
            ColourModel::Yuv420 => {
                lap_mean[1] = lap_mean[1]
                    .checked_mul(SUBSAMPLED_WEIGHT[0][band])
                    .ok_or(EntropyError::CoefficientOverflow)?;
                2
            }
            ColourModel::Yuv422 => {
                lap_mean[1] = lap_mean[1]
                    .checked_mul(SUBSAMPLED_WEIGHT[1][band])
                    .ok_or(EntropyError::CoefficientOverflow)?;
                2
            }
            ColourModel::Other { components } => {
                if !(2..=16).contains(&components) {
                    return Err(EntropyError::InvalidParameter {
                        parameter: "coefficient-model component count",
                        value: i64::from(components),
                    });
                }
                let chroma_count = usize::from(components - 1);
                lap_mean[1] = lap_mean[1]
                    .checked_mul(OTHER_WEIGHT[band][chroma_count])
                    .ok_or(EntropyError::CoefficientOverflow)?;
                if self.band == FrequencyBand::Highpass {
                    lap_mean[1] >>= 4;
                }
                2
            }
        };

        for (index, mean) in lap_mean.into_iter().enumerate().take(model_count) {
            update_one(&mut self.state[index], &mut self.bits[index], mean);
        }
        Ok(())
    }
}

fn update_one(state: &mut i32, bits: &mut u8, lap_mean: i32) {
    let mut delta = (lap_mean - 70) >> 2;
    if delta <= -8 {
        delta = (delta + 4).max(-16);
        *state += delta;
        if *state < -8 {
            if *bits == 0 {
                *state = -8;
            } else {
                *state = 0;
                *bits -= 1;
            }
        }
    } else if delta >= 8 {
        delta = (delta - 4).min(15);
        *state += delta;
        if *state > 8 {
            if *bits >= 15 {
                *bits = 15;
                *state = 8;
            } else {
                *state = 0;
                *bits += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_model_bits_match_table_115() {
        assert_eq!(CoefficientModel::new(FrequencyBand::Dc).bits(false), 8);
        assert_eq!(CoefficientModel::new(FrequencyBand::Lowpass).bits(false), 4);
        assert_eq!(
            CoefficientModel::new(FrequencyBand::Highpass).bits(false),
            0
        );
    }

    #[test]
    fn low_activity_reduces_model_bits() {
        let mut model = CoefficientModel::new(FrequencyBand::Lowpass);
        model.update([0, 0], ColourModel::LumaOnly).unwrap();
        assert_eq!(model.bits(false), 3);
        assert_eq!(model.state[0], 0);
    }

    #[test]
    fn high_activity_increases_model_bits() {
        let mut model = CoefficientModel::new(FrequencyBand::Highpass);
        model.update([122, 0], ColourModel::LumaOnly).unwrap();
        assert_eq!(model.bits(false), 1);
        assert_eq!(model.state[0], 0);
    }

    #[test]
    fn luma_only_does_not_modify_chroma_model() {
        let mut model = CoefficientModel::new(FrequencyBand::Dc);
        model.update([0, i32::MAX], ColourModel::LumaOnly).unwrap();
        assert_eq!(model.bits(true), 8);
        assert_eq!(model.state[1], 0);
    }

    #[test]
    fn invalid_component_count_is_not_silently_clamped() {
        let mut model = CoefficientModel::new(FrequencyBand::Lowpass);
        assert!(matches!(
            model.update([0, 0], ColourModel::Other { components: 1 }),
            Err(EntropyError::InvalidParameter {
                parameter: "coefficient-model component count",
                value: 1,
            })
        ));
    }
}
