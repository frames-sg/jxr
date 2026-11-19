//! High-pass residual formation and block writing shared by colour formats.

use jxr_core::{BandPresence, PredictionMode};
use jxr_math::prediction::{PREDICTED_FROM_LEFT, PREDICTED_FROM_TOP};

use crate::{
    entropy::{
        AcVlcState, AdaptiveHpScan, ComponentClass, HpScanDirection,
        encode::{check_magnitudes, encode_flex_block, encode_hp_block, vlc_part},
    },
    tail::PacketBand,
    tile_decode::{ComponentGeometry, high_pass::coded_block_order},
};

use super::{TileEncodeError, packets::TilePackets};

/// How the high-pass syntax of a macroblock represents its coefficients.
#[derive(Debug, Clone, Copy)]
pub(super) struct HighPassSyntax {
    /// Prediction direction selected from the macroblock's low-pass values.
    pub(super) mode: PredictionMode,
    pub(super) bands: BandPresence,
    pub(super) trim_flexbits: u8,
}

/// High-pass residuals of one component macroblock, split into the parts
/// the syntax codes separately.
///
/// A plane keeps one of these for each component and forms every
/// macroblock's residuals in it, so no macroblock moves two kilobytes of
/// coefficients to where the last one's were.
#[derive(Debug, Clone)]
pub(super) struct HighPass {
    /// Prediction residuals of each block, in raster block order.
    residuals: [[i32; 16]; 16],
    /// VLC-coded parts of each block, in coded block order.
    levels: [[i32; 16]; 16],
    /// One bit per block in coded order: set when the block is coded.
    pattern: u16,
    /// Raster index of each block the component has, in coded order.
    order: &'static [usize],
    model_bits: u8,
}

impl HighPass {
    /// Storage for the residuals of one component, holding no macroblock
    /// yet.
    pub(super) const fn empty() -> Self {
        Self {
            residuals: [[0; 16]; 16],
            levels: [[0; 16]; 16],
            pattern: 0,
            order: &[],
            model_bits: 0,
        }
    }

    /// The residuals of `actual`, split at `model_bits`; see [`Self::form`].
    #[cfg(test)]
    pub(super) fn new(
        actual: &[i32; 256],
        syntax: HighPassSyntax,
        model_bits: u8,
        geometry: ComponentGeometry,
    ) -> Result<Self, TileEncodeError> {
        let mut band = Self::empty();
        band.form(actual, syntax, model_bits, geometry)?;
        Ok(band)
    }

    /// Forms the residuals of `actual` and splits them at `model_bits`,
    /// replacing those of the macroblock before.
    ///
    /// `actual` holds the blocks of a component macroblock shaped as
    /// `geometry` says, in raster block order; element zero of each block
    /// belongs to the low-pass band and is ignored, as are blocks beyond the
    /// component's count.
    pub(super) fn form(
        &mut self,
        actual: &[i32; 256],
        syntax: HighPassSyntax,
        model_bits: u8,
        geometry: ComponentGeometry,
    ) -> Result<(), TileEncodeError> {
        // Bits of each residual magnitude the syntax drops: trimmed flexbits,
        // or every refinement bit when the flexbits band is absent.
        let dropped_bits = if syntax.bands.has_flexbits() {
            model_bits.min(syntax.trim_flexbits)
        } else {
            model_bits
        };
        self.residuals.as_flattened_mut().copy_from_slice(actual);
        subtract_prediction(&mut self.residuals, syntax.mode, dropped_bits, geometry)?;
        check_magnitudes(self.residuals.as_flattened())?;
        self.order = coded_block_order(geometry);
        self.model_bits = model_bits;
        self.pattern = 0;
        for (bit, (levels, &block)) in self.levels.iter_mut().zip(self.order).enumerate() {
            let residuals = &self.residuals[block & 15];
            for (level, &residual) in levels.iter_mut().zip(residuals) {
                *level = vlc_part(residual, model_bits);
            }
            // Element zero is the block's low-pass coefficient.
            levels[0] = 0;
            let coded = levels.iter().fold(0, |any, &level| any | level) != 0;
            self.pattern |= u16::from(coded) << bit;
        }
        Ok(())
    }

    /// The coded-block pattern the macroblock's pattern syntax must carry.
    pub(super) const fn pattern(&self) -> u16 {
        self.pattern
    }

    /// Writes each coded block and then its flexbits, in coded block order,
    /// each to the packet that carries it: a block's flexbits follow the
    /// block itself where one packet carries both, and the flexbits of the
    /// block before where they have a packet of their own. Returns the
    /// number of VLC-coded coefficients.
    pub(super) fn write(
        &self,
        packets: &mut TilePackets,
        class: ComponentClass,
        syntax: HighPassSyntax,
        vlc: &mut AcVlcState,
        scan: &mut AdaptiveHpScan,
    ) -> Result<i32, TileEncodeError> {
        let mut non_zero = 0_i32;
        let flexbits = syntax.bands.has_flexbits();
        for (bit, &block) in self.order.iter().enumerate() {
            if self.pattern >> bit & 1 != 0 {
                non_zero += i32::from(encode_hp_block(
                    packets.band(PacketBand::HighPass)?,
                    class,
                    vlc,
                    scan,
                    HpScanDirection::for_mode(syntax.mode),
                    &self.levels[bit & 15],
                )?);
            }
            if flexbits {
                encode_flex_block(
                    packets.band(PacketBand::Flexbits)?,
                    &self.residuals[block & 15],
                    self.model_bits,
                    syntax.trim_flexbits,
                )?;
            }
        }
        Ok(non_zero)
    }
}

/// Turns the coefficients in `blocks` into the residuals a decoder turns
/// back into them by in-macroblock prediction over the blocks of `geometry`,
/// which `blocks` holds in raster block order.
///
/// When the syntax drops low residual bits, each block is predicted from its
/// neighbour as the decoder will reconstruct it, not from the neighbour's
/// original value, so the loss of one block does not leak into the next.
fn subtract_prediction(
    blocks: &mut [[i32; 16]; 16],
    mode: PredictionMode,
    dropped_bits: u8,
    geometry: ComponentGeometry,
) -> Result<(), TileEncodeError> {
    let columns = usize::from(geometry.columns);
    // Predicted positions, and how many blocks back the predicting block
    // lies: the one to the left, or the one a block row above.
    let (positions, distance) = match mode {
        PredictionMode::FromLeft => (PREDICTED_FROM_LEFT, 1),
        PredictionMode::FromTop => (PREDICTED_FROM_TOP, columns),
        _ => return Ok(()),
    };
    // Decoder-side values of the predicted positions, block by block.
    let mut reconstructed = [[0_i32; 3]; 16];
    for (block, coefficients) in blocks.iter_mut().enumerate().take(geometry.block_count()) {
        // The first block column has no left neighbour within the
        // macroblock, and the first block row none above.
        let source = match mode {
            PredictionMode::FromLeft if block % columns == 0 => None,
            _ => block.checked_sub(distance),
        };
        for (slot, position) in positions.into_iter().enumerate() {
            let predictor = source.map_or(0, |source| reconstructed[source][slot]);
            let residual = coefficients[position].checked_sub(predictor).ok_or(
                TileEncodeError::ArithmeticOverflow("HP prediction residual"),
            )?;
            coefficients[position] = residual;
            reconstructed[block][slot] = retained(residual, dropped_bits)
                .checked_add(predictor)
                .ok_or(TileEncodeError::ArithmeticOverflow("HP prediction"))?;
        }
    }
    Ok(())
}

/// The value a decoder recovers once `dropped_bits` magnitude bits are lost.
const fn retained(residual: i32, dropped_bits: u8) -> i32 {
    let magnitude = (residual.unsigned_abs() >> dropped_bits) << dropped_bits;
    if residual < 0 {
        magnitude.cast_signed().wrapping_neg()
    } else {
        magnitude.cast_signed()
    }
}

#[cfg(test)]
mod tests {
    use jxr_math::prediction::{
        HighpassPrediction, predict_high_pass, predict_high_pass_420, predict_high_pass_422,
    };

    use super::*;
    use crate::entropy::encode::test_support::Lcg;

    const FULL: ComponentGeometry = ComponentGeometry {
        columns: 4,
        rows: 4,
    };
    const CHROMA_422: ComponentGeometry = ComponentGeometry {
        columns: 2,
        rows: 4,
    };
    const CHROMA_420: ComponentGeometry = ComponentGeometry {
        columns: 2,
        rows: 2,
    };
    const GEOMETRIES: [ComponentGeometry; 3] = [FULL, CHROMA_422, CHROMA_420];

    /// The residuals [`subtract_prediction`] leaves in place of `actual`.
    fn residuals(
        actual: &[i32; 256],
        mode: PredictionMode,
        dropped_bits: u8,
        geometry: ComponentGeometry,
    ) -> Result<[i32; 256], TileEncodeError> {
        let mut blocks = [[0_i32; 16]; 16];
        blocks.as_flattened_mut().copy_from_slice(actual);
        subtract_prediction(&mut blocks, mode, dropped_bits, geometry)?;
        let mut residuals = [0_i32; 256];
        residuals.copy_from_slice(blocks.as_flattened());
        Ok(residuals)
    }

    /// Coefficients in the blocks a component of `geometry` has.
    fn block_values(seed: u64, geometry: ComponentGeometry) -> [i32; 256] {
        let mut state = seed;
        core::array::from_fn(|index| {
            state = Lcg::advance(state);
            if index % 16 == 0 || index >= geometry.block_count() * 16 {
                0
            } else {
                i32::try_from((state >> 40) % 2001).unwrap() - 1000
            }
        })
    }

    /// The decoder's in-macroblock prediction for a component of `geometry`.
    fn predict(
        coefficients: &mut [i32; 256],
        direction: HighpassPrediction,
        geometry: ComponentGeometry,
    ) {
        match (geometry.columns, geometry.rows) {
            (4, 4) => predict_high_pass(coefficients, direction),
            (2, 4) => predict_high_pass_422(coefficients, direction),
            _ => predict_high_pass_420(coefficients, direction),
        }
        .unwrap();
    }

    #[test]
    fn lossless_residuals_invert_the_decoder_prediction() {
        for geometry in GEOMETRIES {
            let actual = block_values(21, geometry);
            for (mode, direction) in [
                (PredictionMode::None, HighpassPrediction::None),
                (PredictionMode::FromLeft, HighpassPrediction::FromLeft),
                (PredictionMode::FromTop, HighpassPrediction::FromTop),
            ] {
                let mut decoded = residuals(&actual, mode, 0, geometry).unwrap();
                // A predicted direction changes the residuals it predicts.
                assert_eq!(decoded != actual, mode != PredictionMode::None);
                predict(&mut decoded, direction, geometry);
                assert_eq!(decoded, actual, "{geometry:?} {mode:?}");
            }
        }
    }

    #[test]
    fn dropped_bits_do_not_accumulate_along_the_prediction_chain() {
        let dropped = 3;
        for geometry in GEOMETRIES {
            let actual = block_values(22, geometry);
            for (mode, direction) in [
                (PredictionMode::FromLeft, HighpassPrediction::FromLeft),
                (PredictionMode::FromTop, HighpassPrediction::FromTop),
            ] {
                let mut decoded = residuals(&actual, mode, dropped, geometry)
                    .unwrap()
                    .map(|residual| retained(residual, dropped));
                predict(&mut decoded, direction, geometry);
                let worst = decoded
                    .iter()
                    .zip(&actual)
                    .map(|(a, b)| a.abs_diff(*b))
                    .max()
                    .unwrap();
                // Every coefficient loses fewer than 2^dropped, however long its chain.
                assert!(
                    worst < 1 << dropped,
                    "{geometry:?} {mode:?} worst error {worst}"
                );
            }
        }
    }

    #[test]
    fn pattern_marks_exactly_the_blocks_with_vlc_coded_parts() {
        let mut actual = [0_i32; 256];
        // Raster block 2 is hierarchical position 4; block 5 is position 3.
        actual[2 * 16 + 7] = 9;
        actual[5 * 16 + 1] = -1;
        let syntax = HighPassSyntax {
            mode: PredictionMode::None,
            bands: BandPresence::All,
            trim_flexbits: 0,
        };
        assert_eq!(
            HighPass::new(&actual, syntax, 0, FULL).unwrap().pattern(),
            (1 << 4) | (1 << 3)
        );
        // With one model bit, the coefficient of magnitude one moves entirely
        // into the flexbits and its block is no longer VLC coded.
        assert_eq!(
            HighPass::new(&actual, syntax, 1, FULL).unwrap().pattern(),
            1 << 4
        );
    }

    #[test]
    fn subsampled_chroma_patterns_count_blocks_in_raster_order() {
        let mut actual = [0_i32; 256];
        actual[2 * 16 + 7] = 9;
        actual[5 * 16 + 1] = -1;
        // A block beyond the four of 4:2:0 chroma is not part of it.
        actual[9 * 16 + 3] = 40;
        let syntax = HighPassSyntax {
            mode: PredictionMode::None,
            bands: BandPresence::All,
            trim_flexbits: 0,
        };
        let pattern = |geometry| {
            HighPass::new(&actual, syntax, 0, geometry)
                .unwrap()
                .pattern()
        };
        assert_eq!(pattern(CHROMA_422), (1 << 2) | (1 << 5));
        assert_eq!(pattern(CHROMA_420), 1 << 2);
    }

    #[test]
    fn subsampled_chroma_blocks_are_written_in_raster_order() {
        use jxr_core::{BitstreamMode, ChromaSampling, ColorFormat};

        use crate::{
            entropy::{PacketBitReader, decode_hp_block},
            tile_encode::PacketOptions,
        };

        for geometry in [CHROMA_422, CHROMA_420] {
            let actual = block_values(23, geometry);
            let syntax = HighPassSyntax {
                mode: PredictionMode::FromTop,
                bands: BandPresence::NoFlexbits,
                trim_flexbits: 0,
            };
            let band = HighPass::new(&actual, syntax, 0, geometry).unwrap();
            let mut packets = TilePackets::start(
                PacketOptions {
                    colour: ColorFormat::Yuv(ChromaSampling::Cs420),
                    bands: syntax.bands,
                    trim_flexbits: None,
                    mode: BitstreamMode::Spatial,
                },
                &mut Vec::new(),
                None,
            )
            .unwrap();
            let mut write_scan = AdaptiveHpScan::new();
            band.write(
                &mut packets,
                ComponentClass::Chroma,
                syntax,
                &mut AcVlcState::new(),
                &mut write_scan,
            )
            .unwrap();
            let [bytes] = <[Vec<u8>; 1]>::try_from(packets.finish(&mut Vec::new())).unwrap();
            let mut reader = PacketBitReader::new(&bytes);
            // Start code and arbitrary byte.
            assert_eq!(reader.read_bits(32).unwrap(), 0x100);
            let mut vlc = AcVlcState::new();
            let mut scan = AdaptiveHpScan::new();
            let mut decoded = [0_i32; 256];
            for block in 0..geometry.block_count() {
                assert_ne!(band.pattern() >> block & 1, 0);
                let mut values = [0_i32; 16];
                decode_hp_block(
                    &mut reader,
                    ComponentClass::Chroma,
                    &mut vlc,
                    &mut scan,
                    HpScanDirection::Vertical,
                    &mut values,
                )
                .unwrap();
                decoded[block * 16..block * 16 + 16].copy_from_slice(&values);
            }
            predict(&mut decoded, HighpassPrediction::FromTop, geometry);
            assert_eq!(decoded, actual, "{geometry:?}");
            assert_eq!(scan, write_scan);
        }
    }
}
