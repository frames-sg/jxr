//! What both packet layouts share around the macroblock syntax: packet
//! framing, the history of macroblocks DC and low-pass prediction read, the
//! low-pass prediction itself, and when adaptive state is reset and adapted.

use jxr_core::PredictionMode;
use jxr_math::prediction::{PREDICTED_FROM_LEFT, PREDICTED_FROM_TOP};

use crate::entropy::{PacketBitReader, TileEntropyState};

use super::{ComponentGeometry, TileDecodeError, cbphp::CbphpState};

pub(crate) const TILE_START_CODE: u32 = 1;

/// Prediction-resolved DC/LP values and LP quantizer index of one macroblock
/// of a component, as DC and low-pass prediction read them.
///
/// A tile reader keeps a record for each macroblock of the row it is
/// reading and of the row above, because coefficients are stored as each
/// macroblock row completes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PredictionRecord {
    pub(crate) dc_low_pass: [i32; 16],
    pub(crate) lp_qp_index: u8,
}

pub(super) fn parse_packet_prefix(reader: &mut PacketBitReader<'_>) -> Result<(), TileDecodeError> {
    let start = u32::try_from(reader.read_bits(24)?)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("tile start code"))?;
    if start != TILE_START_CODE {
        return Err(TileDecodeError::InvalidStartCode { value: start });
    }
    let _arbitrary_byte = reader.read_bits(8)?;
    Ok(())
}

/// The flexbits trim a tile packet carries when the image has one.
pub(super) fn read_trim(
    reader: &mut PacketBitReader<'_>,
    trim_flexbits_present: bool,
) -> Result<u8, TileDecodeError> {
    if trim_flexbits_present {
        read_u8(reader, 4)
    } else {
        Ok(0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MacroblockPosition {
    pub(crate) width: usize,
    pub(crate) x: usize,
    pub(crate) y: usize,
}

impl MacroblockPosition {
    /// This position as a history of two macroblock rows has it.
    ///
    /// Prediction reads the macroblock row being coded and the one above
    /// it, and asks of the row only whether it is the first of its tile.
    /// A tile coded a row at a time therefore keeps a history of those two
    /// rows, which [`end_history_row`] slides down, and names positions in
    /// it: row zero for the first row of the tile and row one after it,
    /// with the row above as row zero.
    pub(crate) const fn in_two_rows(self) -> Self {
        Self {
            y: if self.y == 0 { 0 } else { 1 },
            ..self
        }
    }

    /// Where the macroblock at this position of a two-row history is held.
    pub(crate) const fn history_index(self) -> usize {
        self.y * self.width + self.x
    }
}

/// Ends a macroblock row of a two-row prediction history: the row above
/// the one just coded is dropped, and the one just coded becomes the row
/// above the next. See [`MacroblockPosition::in_two_rows`].
pub(crate) fn end_history_row<M>(history: &mut Vec<M>, width: usize) {
    let above = history.len().saturating_sub(width);
    history.drain(..above);
}

/// Whether low-pass prediction is being applied to parsed residuals or
/// removed from coefficients about to be written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PredictionDirection {
    Apply,
    Remove,
}

/// The low-pass prediction mode of a macroblock whose DC prediction mode
/// is `dc_mode`: a neighbour predicts the low-pass band only when it alone
/// predicted the DC value and was quantized with the same low-pass
/// quantizer.
pub(crate) fn low_pass_prediction_mode(
    history: &[PredictionRecord],
    position: MacroblockPosition,
    dc_mode: PredictionMode,
    lp_qp_index: u8,
) -> Result<PredictionMode, TileDecodeError> {
    let MacroblockPosition { width, x, y } = position;
    Ok(match dc_mode {
        PredictionMode::FromLeft
            if spatial_macroblock(history, width, x - 1, y)?.lp_qp_index == lp_qp_index =>
        {
            PredictionMode::FromLeft
        }
        PredictionMode::FromTop
            if spatial_macroblock(history, width, x, y - 1)?.lp_qp_index == lp_qp_index =>
        {
            PredictionMode::FromTop
        }
        _ => PredictionMode::None,
    })
}

/// Where the predictor of one low-pass coefficient lies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LowPassPredictor {
    /// The neighbouring macroblock's coefficient at this position.
    Neighbour(usize),
    /// The macroblock's own coefficient at this position, itself already
    /// predicted.
    Own(usize),
}

/// The steps of low-pass prediction from a neighbour at the positions of
/// one 4-by-4 block: each position from the neighbour's same one.
const fn from_neighbour(positions: [usize; 3]) -> [(usize, LowPassPredictor); 3] {
    [
        (positions[0], LowPassPredictor::Neighbour(positions[0])),
        (positions[1], LowPassPredictor::Neighbour(positions[1])),
        (positions[2], LowPassPredictor::Neighbour(positions[2])),
    ]
}

const FULL_FROM_LEFT: [(usize, LowPassPredictor); 3] = from_neighbour(PREDICTED_FROM_LEFT);
const FULL_FROM_TOP: [(usize, LowPassPredictor); 3] = from_neighbour(PREDICTED_FROM_TOP);

/// The coefficients low-pass prediction changes in a component whose
/// macroblocks have the blocks of `geometry`, each with its predictor, in
/// the order a decoder resolves them.
///
/// `mode` is the low-pass prediction mode and `dc_mode` the DC mode it was
/// selected from. A full-resolution component's low-pass band is a 4-by-4
/// block of its own. In a chroma component two blocks wide, positions two
/// and four are the first vertical frequencies and one and five the
/// horizontal ones.
fn low_pass_prediction_steps(
    geometry: ComponentGeometry,
    mode: PredictionMode,
    dc_mode: PredictionMode,
) -> &'static [(usize, LowPassPredictor)] {
    use LowPassPredictor::{Neighbour, Own};
    match ((geometry.columns, geometry.rows), mode) {
        ((2, 2), PredictionMode::FromLeft) => &[(2, Neighbour(2))],
        ((2, 2), PredictionMode::FromTop) => &[(1, Neighbour(1))],
        ((2, 4), PredictionMode::FromLeft) => {
            &[(4, Neighbour(4)), (2, Neighbour(2)), (6, Neighbour(6))]
        }
        // The lower half of a 4:2:2 macroblock is predicted from its upper
        // half whenever DC is predicted from above.
        ((2, 4), PredictionMode::FromTop) => &[(4, Neighbour(4)), (1, Neighbour(5)), (5, Own(1))],
        ((2, 4), PredictionMode::None) if dc_mode == PredictionMode::FromTop => &[(5, Own(1))],
        ((2, _), _) => &[],
        (_, PredictionMode::FromLeft) => &FULL_FROM_LEFT,
        (_, PredictionMode::FromTop) => &FULL_FROM_TOP,
        _ => &[],
    }
}

/// Adds the low-pass prediction of one component of the macroblock at
/// `position` to the residuals a decoder parsed, or removes it from the
/// coefficients an encoder is about to write.
///
/// `history` holds the component's earlier macroblocks, whose blocks have
/// the shape `geometry`, and `dc_mode` and `lp_qp_index` are the
/// macroblock's DC prediction mode and low-pass quantizer index. Every
/// predictor of [`low_pass_prediction_steps`] is added in order, or
/// subtracted in reverse order: a step that predicts from the macroblock's
/// own coefficient reads its predicted value, which the reverse order has
/// not yet touched when removing.
pub(crate) fn resolve_low_pass(
    current: &mut [i32; 16],
    history: &[PredictionRecord],
    position: MacroblockPosition,
    (dc_mode, lp_qp_index): (PredictionMode, u8),
    geometry: ComponentGeometry,
    direction: PredictionDirection,
) -> Result<(), TileDecodeError> {
    let mode = low_pass_prediction_mode(history, position, dc_mode, lp_qp_index)?;
    let steps = low_pass_prediction_steps(geometry, mode, dc_mode);
    if steps.is_empty() {
        return Ok(());
    }
    let MacroblockPosition { width, x, y } = position;
    let neighbour = match mode {
        PredictionMode::FromLeft => Some(spatial_macroblock(history, width, x - 1, y)?),
        PredictionMode::FromTop => Some(spatial_macroblock(history, width, x, y - 1)?),
        _ => None,
    }
    .map(|record| &record.dc_low_pass);
    let resolve = |current: &mut [i32; 16], (target, predictor): (usize, LowPassPredictor)| {
        let value = match predictor {
            LowPassPredictor::Neighbour(source) => {
                neighbour.ok_or(TileDecodeError::InvalidPlan("LP prediction neighbour"))?[source]
            }
            LowPassPredictor::Own(source) => current[source],
        };
        current[target] = match direction {
            PredictionDirection::Apply => current[target].checked_add(value),
            PredictionDirection::Remove => current[target].checked_sub(value),
        }
        .ok_or(TileDecodeError::ArithmeticOverflow("LP prediction"))?;
        Ok::<(), TileDecodeError>(())
    };
    match direction {
        PredictionDirection::Apply => steps.iter().try_for_each(|&step| resolve(current, step)),
        PredictionDirection::Remove => steps
            .iter()
            .rev()
            .try_for_each(|&step| resolve(current, step)),
    }
}

/// Whether the scan totals of a tile are reset before the macroblock at
/// column `x`: at the first of every sixteen columns.
///
/// The rule belongs to the low-pass and high-pass bands alike, so the one
/// packet of a spatial-mode tile and each band packet of a frequency-mode
/// tile reset at the same macroblocks.
pub(crate) const fn resets_scan_totals(x: usize) -> bool {
    x.is_multiple_of(16)
}

/// Whether the adaptive code tables of a tile are selected anew after the
/// macroblock at column `x` of a tile `width` macroblocks wide: after the
/// first of every sixteen columns and after the last one of a row.
///
/// Every band adapts its own tables by this rule after its part of a
/// macroblock, in either packet layout; [`adapt_at_boundary`] adapts all of
/// them together after a whole macroblock.
pub(crate) const fn adapts_after(x: usize, width: usize) -> bool {
    x + 1 == width || x.is_multiple_of(16)
}

/// Selects the code tables of every band, and the coded-block-pattern
/// tables, where [`adapts_after`] says a tile adapts.
pub(crate) fn adapt_at_boundary(
    entropy: &mut TileEntropyState,
    cbphp: &mut CbphpState,
    x: usize,
    width: usize,
) {
    if adapts_after(x, width) {
        entropy.dc_vlc.adapt();
        entropy.lp_vlc.adapt();
        entropy.hp_vlc.adapt();
        cbphp.adapt();
    }
}

pub(super) fn consume_byte_alignment(
    reader: &mut PacketBitReader<'_>,
) -> Result<(), TileDecodeError> {
    while !reader.bit_position().is_multiple_of(8) {
        let _ = reader.read_bit()?;
    }
    Ok(())
}

/// The record of the macroblock at `(x, y)` of a two-row history `width`
/// macroblocks wide.
pub(crate) fn spatial_macroblock(
    history: &[PredictionRecord],
    width: usize,
    x: usize,
    y: usize,
) -> Result<&PredictionRecord, TileDecodeError> {
    history
        .get(y.saturating_mul(width).saturating_add(x))
        .ok_or(TileDecodeError::InvalidPlan("prediction neighbour"))
}

pub(in crate::tile_decode) fn checked_add(
    left: i32,
    right: i32,
    operation: &'static str,
) -> Result<i32, TileDecodeError> {
    left.checked_add(right)
        .ok_or(TileDecodeError::ArithmeticOverflow(operation))
}

pub(super) fn read_u8(reader: &mut PacketBitReader<'_>, bits: u8) -> Result<u8, TileDecodeError> {
    u8::try_from(reader.read_bits(bits)?)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("tile field conversion"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_alignment_discards_unspecified_trailing_bits() {
        let mut reader = PacketBitReader::new(&[0xff]);
        assert!(reader.read_bit().unwrap());
        consume_byte_alignment(&mut reader).unwrap();
        assert_eq!(reader.bit_position(), 8);
    }

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

    /// Prediction history of a tile two macroblocks wide whose records hold
    /// `fill(index, position)`.
    fn history(count: usize, fill: impl Fn(usize, usize) -> i32) -> Vec<PredictionRecord> {
        (0..count)
            .map(|index| PredictionRecord {
                dc_low_pass: core::array::from_fn(|position| fill(index, position)),
                lp_qp_index: 0,
            })
            .collect()
    }

    #[test]
    fn left_low_pass_prediction_uses_the_left_macroblock_column() {
        let decoded = history(1, |_, position| match position {
            4 | 5 | 6 | 8 | 12 => i32::try_from(position).unwrap(),
            _ => 0,
        });
        let mut current = [0; 16];
        let position = MacroblockPosition {
            width: 2,
            x: 1,
            y: 0,
        };
        let context = (PredictionMode::FromLeft, 0);
        resolve_low_pass(
            &mut current,
            &decoded,
            position,
            context,
            FULL,
            PredictionDirection::Apply,
        )
        .unwrap();
        assert_eq!((current[4], current[8], current[12]), (4, 8, 12));
        assert_eq!(current.iter().filter(|&&value| value != 0).count(), 3);
    }

    #[test]
    fn removing_low_pass_prediction_inverts_applying_it() {
        let decoded = history(3, |index, position| {
            i32::try_from(index * 100 + position * 7).unwrap() - 150
        });
        let actual: [i32; 16] =
            core::array::from_fn(|position| i32::try_from(position * position).unwrap() - 21);
        for geometry in [FULL, CHROMA_422, CHROMA_420] {
            // `(position, DC mode)`: no neighbour, the left one, the one
            // above, and both.
            for (x, y, dc_mode) in [
                (0, 0, PredictionMode::None),
                (1, 0, PredictionMode::FromLeft),
                (0, 1, PredictionMode::FromTop),
                (1, 1, PredictionMode::FromLeft),
                (1, 1, PredictionMode::FromTop),
                (1, 1, PredictionMode::FromTopLeft),
            ] {
                let position = MacroblockPosition { width: 2, x, y };
                let case = format!("{geometry:?} at {x},{y} {dc_mode:?}");
                let mut residual = actual;
                let context = (dc_mode, 0);
                resolve_low_pass(
                    &mut residual,
                    &decoded,
                    position,
                    context,
                    geometry,
                    PredictionDirection::Remove,
                )
                .unwrap();
                // Each mode predicts something but for the two with no
                // predicted coefficient: none at all, and both neighbours.
                let predicted =
                    !matches!(dc_mode, PredictionMode::None | PredictionMode::FromTopLeft);
                assert_eq!(residual != actual, predicted, "{case}");
                let mut restored = residual;
                resolve_low_pass(
                    &mut restored,
                    &decoded,
                    position,
                    context,
                    geometry,
                    PredictionDirection::Apply,
                )
                .unwrap();
                assert_eq!(restored, actual, "{case}");
            }
        }
    }

    #[test]
    fn the_lower_half_of_a_422_macroblock_is_predicted_from_its_upper_half() {
        // The macroblock above holds 40 at position four and 9 at five.
        let decoded = history(2, |_, position| match position {
            4 => 40,
            5 => 9,
            _ => 1000,
        });
        let position = MacroblockPosition {
            width: 2,
            x: 0,
            y: 1,
        };
        let mut current = [0_i32; 16];
        current[1] = 5;
        current[4] = 2;
        current[5] = 100;
        let remove = |current: &mut [i32; 16], lp_qp_index, geometry| {
            resolve_low_pass(
                current,
                &decoded,
                position,
                (PredictionMode::FromTop, lp_qp_index),
                geometry,
                PredictionDirection::Remove,
            )
            .unwrap();
        };
        let mut residual = current;
        remove(&mut residual, 0, CHROMA_422);
        // Position one takes the neighbour's five; position five takes the
        // macroblock's own position one, as coded, not its residual.
        assert_eq!(
            (residual[1], residual[4], residual[5]),
            (5 - 9, 2 - 40, 100 - 5)
        );
        // A quantizer change stops the neighbour's prediction but not the
        // macroblock's own.
        let mut other_quantizer = current;
        remove(&mut other_quantizer, 1, CHROMA_422);
        assert_eq!(
            (other_quantizer[1], other_quantizer[4], other_quantizer[5]),
            (5, 2, 95)
        );
        // 4:2:0 has no lower half, and a full-resolution component predicts
        // only from its neighbour: nothing is predicted without one.
        for geometry in [CHROMA_420, FULL] {
            let mut unchanged = current;
            remove(&mut unchanged, 1, geometry);
            assert_eq!(unchanged, current, "{geometry:?}");
        }
    }
}
