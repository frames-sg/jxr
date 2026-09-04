//! Adaptive VLC selection and tile-owned entropy state.

use super::{AdaptiveHpScan, AdaptiveLpScan, CoefficientModel, FrequencyBand};

const ABS_LEVEL_DELTA: [i8; 7] = [1, 0, -1, -1, -1, -1, -1];
const FIRST_INDEX_DELTA: [[i8; 12]; 4] = [
    [1, 1, 1, 1, 1, 0, 0, -1, 2, 1, 0, 0],
    [2, 2, -1, -1, -1, 0, -2, -1, 0, 0, -2, -1],
    [-1, 1, 0, 2, 0, 0, 0, 0, -2, 0, 1, 1],
    [0, 1, 0, 1, -2, 0, -1, -1, -2, -1, -2, -2],
];
const INDEX_DELTA: [[i8; 6]; 3] = [
    [-1, 1, 1, 1, 0, 1],
    [-2, 0, 0, 2, 0, 0],
    [-1, -1, 0, 1, -2, 0],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AdaptiveVlc {
    discrim: [i32; 2],
    table_index: u8,
    delta_index: [u8; 2],
}

impl AdaptiveVlc {
    const fn two_tables() -> Self {
        Self {
            discrim: [0, 0],
            table_index: 0,
            delta_index: [0, 0],
        }
    }

    const fn multiple_tables() -> Self {
        Self {
            discrim: [0, 0],
            table_index: 1,
            delta_index: [0, 1],
        }
    }

    pub(super) const fn table_index(self) -> usize {
        self.table_index as usize
    }

    fn observe_one(&mut self, symbol: u8, delta: &[i8]) {
        self.discrim[0] += i32::from(delta[usize::from(symbol)]);
    }

    fn observe_two(&mut self, symbol: u8, delta: &[[i8; 12]; 4]) {
        let symbol = usize::from(symbol);
        self.discrim[0] += i32::from(delta[usize::from(self.delta_index[0])][symbol]);
        self.discrim[1] += i32::from(delta[usize::from(self.delta_index[1])][symbol]);
    }

    fn observe_two_index(&mut self, symbol: u8) {
        let symbol = usize::from(symbol);
        self.discrim[0] += i32::from(INDEX_DELTA[usize::from(self.delta_index[0])][symbol]);
        self.discrim[1] += i32::from(INDEX_DELTA[usize::from(self.delta_index[1])][symbol]);
    }

    fn adapt_one(&mut self) {
        if self.discrim[0] < -8 && self.table_index != 0 {
            self.table_index -= 1;
            self.discrim[0] = 0;
        } else if self.discrim[0] > 8 && self.table_index != 1 {
            self.table_index += 1;
            self.discrim[0] = 0;
        } else {
            self.discrim[0] = self.discrim[0].clamp(-64, 64);
        }
    }

    fn adapt_two(&mut self, max_table_index: u8) {
        let changed = if self.discrim[0] < -8 && self.table_index != 0 {
            self.table_index -= 1;
            true
        } else if self.discrim[1] > 8 && self.table_index != max_table_index {
            self.table_index += 1;
            true
        } else {
            false
        };
        if changed {
            self.discrim = [0, 0];
            if self.table_index == max_table_index {
                self.delta_index = [self.table_index - 1; 2];
            } else if self.table_index == 0 {
                self.delta_index = [0, 0];
            } else {
                self.delta_index = [self.table_index - 1, self.table_index];
            }
        } else {
            self.discrim[0] = self.discrim[0].clamp(-64, 64);
            self.discrim[1] = self.discrim[1].clamp(-64, 64);
        }
    }
}

/// Adaptive `ABS_LEVEL_INDEX` state for DC luma and chroma symbols.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DcVlcState {
    abs_level: [AdaptiveVlc; 2],
}

impl Default for DcVlcState {
    fn default() -> Self {
        Self::new()
    }
}

impl DcVlcState {
    /// Returns the normative tile-initial state from T.832 Table 92.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            abs_level: [AdaptiveVlc::two_tables(); 2],
        }
    }

    /// Reinitializes this state at the first macroblock of a tile.
    pub fn reset_tile(&mut self) {
        *self = Self::new();
    }

    /// Adapts both DC code table selectors at a normative context boundary.
    pub fn adapt(&mut self) {
        for state in &mut self.abs_level {
            state.adapt_one();
        }
    }

    pub(super) fn abs_level(&mut self, chroma: bool) -> &mut AdaptiveVlc {
        &mut self.abs_level[usize::from(chroma)]
    }
}

/// Adaptive `FIRST_INDEX`, `INDEX_A`, and `ABS_LEVEL_INDEX` state for one AC band.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcVlcState {
    first_index: [AdaptiveVlc; 2],
    index: [[AdaptiveVlc; 2]; 2],
    abs_level: [AdaptiveVlc; 2],
}

impl Default for AcVlcState {
    fn default() -> Self {
        Self::new()
    }
}

impl AcVlcState {
    /// Returns the normative LP/HP tile-initial state from T.832 Tables 93 and 94.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            first_index: [AdaptiveVlc::multiple_tables(); 2],
            index: [[AdaptiveVlc::multiple_tables(); 2]; 2],
            abs_level: [AdaptiveVlc::two_tables(); 2],
        }
    }

    /// Reinitializes this state at the first macroblock of a tile.
    pub fn reset_tile(&mut self) {
        *self = Self::new();
    }

    /// Adapts every AC selector at a normative context boundary.
    pub fn adapt(&mut self) {
        for state in &mut self.first_index {
            state.adapt_two(4);
        }
        for class in &mut self.index {
            for state in class {
                state.adapt_two(3);
            }
        }
        for state in &mut self.abs_level {
            state.adapt_one();
        }
    }

    pub(super) fn first_index(&mut self, chroma: bool) -> &mut AdaptiveVlc {
        &mut self.first_index[usize::from(chroma)]
    }

    pub(super) fn index(&mut self, chroma: bool, context: bool) -> &mut AdaptiveVlc {
        &mut self.index[usize::from(chroma)][usize::from(context)]
    }

    pub(super) fn abs_level(&mut self, context: bool) -> &mut AdaptiveVlc {
        &mut self.abs_level[usize::from(context)]
    }
}

pub(super) fn observe_abs_level(state: &mut AdaptiveVlc, symbol: u8) {
    state.observe_one(symbol, &ABS_LEVEL_DELTA);
}

pub(super) fn observe_first_index(state: &mut AdaptiveVlc, symbol: u8) {
    state.observe_two(symbol, &FIRST_INDEX_DELTA);
}

pub(super) fn observe_index(state: &mut AdaptiveVlc, symbol: u8) {
    state.observe_two_index(symbol);
}

/// Entropy state whose lifetime is exactly one coded tile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileEntropyState {
    /// DC adaptive VLC selectors.
    pub dc_vlc: DcVlcState,
    /// LP adaptive VLC selectors.
    pub lp_vlc: AcVlcState,
    /// HP adaptive VLC selectors.
    pub hp_vlc: AcVlcState,
    /// Adaptive LP inverse scan.
    pub lp_scan: AdaptiveLpScan,
    /// Adaptive horizontal and vertical HP inverse scans.
    pub hp_scan: AdaptiveHpScan,
    /// DC coefficient-normalization model.
    pub dc_model: CoefficientModel,
    /// LP coefficient-normalization model.
    pub lp_model: CoefficientModel,
    /// HP coefficient-normalization model.
    pub hp_model: CoefficientModel,
}

impl Default for TileEntropyState {
    fn default() -> Self {
        Self::new()
    }
}

impl TileEntropyState {
    /// Creates all state with the first-macroblock-of-tile values.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            dc_vlc: DcVlcState::new(),
            lp_vlc: AcVlcState::new(),
            hp_vlc: AcVlcState::new(),
            lp_scan: AdaptiveLpScan::new(),
            hp_scan: AdaptiveHpScan::new(),
            dc_model: CoefficientModel::new(FrequencyBand::Dc),
            lp_model: CoefficientModel::new(FrequencyBand::Lowpass),
            hp_model: CoefficientModel::new(FrequencyBand::Highpass),
        }
    }

    /// Applies the normative initialization required at a tile's upper-left macroblock.
    pub fn reset_tile(&mut self) {
        *self = Self::new();
    }

    /// Resets scan totals at each 16-macroblock column boundary without changing scan order.
    pub fn reset_scan_totals(&mut self) {
        self.lp_scan.reset_totals();
        self.hp_scan.reset_totals();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_table_state_switches_only_past_normative_threshold() {
        let mut state = AdaptiveVlc::two_tables();
        for _ in 0..8 {
            state.observe_one(0, &ABS_LEVEL_DELTA);
        }
        state.adapt_one();
        assert_eq!(state.table_index(), 0);
        state.observe_one(0, &ABS_LEVEL_DELTA);
        state.adapt_one();
        assert_eq!(state.table_index(), 1);
        assert_eq!(state.discrim, [0, 0]);
    }

    #[test]
    fn multi_table_transition_updates_both_delta_indices() {
        let mut state = AdaptiveVlc::multiple_tables();
        state.discrim = [-9, 0];
        state.adapt_two(4);
        assert_eq!(state.table_index, 0);
        assert_eq!(state.delta_index, [0, 0]);
        state.discrim = [0, 9];
        state.adapt_two(4);
        assert_eq!(state.table_index, 1);
        assert_eq!(state.delta_index, [0, 1]);
    }

    #[test]
    fn tile_reset_restores_every_owned_state() {
        let mut state = TileEntropyState::new();
        state.dc_vlc.abs_level[0].discrim[0] = 40;
        state.lp_scan.place(&mut [0; 16], 2, 7).unwrap();
        state
            .hp_model
            .update([500, 500], crate::entropy::ColourModel::Yuv420)
            .unwrap();
        state.reset_tile();
        assert_eq!(state, TileEntropyState::new());
    }
}
