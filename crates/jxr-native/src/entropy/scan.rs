//! Adaptive inverse scans from T.832 Tables 107 through 114.

use jxr_core::PredictionMode;

use super::EntropyError;

const SCAN_ORDER_0: [u8; 16] = [0, 4, 1, 5, 8, 2, 9, 6, 12, 3, 10, 13, 7, 14, 11, 15];
const SCAN_ORDER_1: [u8; 16] = [0, 1, 2, 5, 4, 3, 6, 9, 8, 7, 12, 15, 13, 10, 11, 14];
const SCAN_TOTALS: [i32; 16] = [0, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4];

/// One adaptive inverse scan: the raster position of each scan index, and
/// how many coefficients each scan index has held.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Scan {
    order: [u8; 16],
    totals: [i32; 16],
}

impl Scan {
    const fn new(order: [u8; 16]) -> Self {
        Self {
            order,
            totals: SCAN_TOTALS,
        }
    }

    /// Counts a coefficient at scan index `index` and moves that index's
    /// raster position one place earlier once it has held more coefficients
    /// than its predecessor.
    ///
    /// This is the whole of scan adaptation; placing a parsed coefficient
    /// and scanning a block to write both end in it.
    #[inline]
    fn adapt(&mut self, index: usize) {
        let index = index & 15;
        self.totals[index] += 1;
        if index > 1 && self.totals[index] > self.totals[index - 1] {
            self.totals.swap(index, index - 1);
            self.order.swap(index, index - 1);
        }
    }

    /// Places one parsed coefficient and adapts to it.
    #[inline]
    fn place(
        &mut self,
        output: &mut [i32; 16],
        scan_index: u8,
        value: i32,
    ) -> Result<(), EntropyError> {
        if !(1..=15).contains(&scan_index) {
            return Err(EntropyError::InvalidParameter {
                parameter: "adaptive scan index",
                value: i64::from(scan_index),
            });
        }
        let index = usize::from(scan_index);
        output[usize::from(self.order[index] & 15)] = value;
        self.adapt(index);
        Ok(())
    }

    /// Forward counterpart of [`Self::place`] for a whole block.
    ///
    /// Collects the non-zero coefficients of `input` at scan indices from
    /// `start` on and applies, for each, exactly the update a decoder
    /// performs when it places that coefficient. Adapting index `n` only
    /// exchanges indices `n` and `n - 1`, so the indices after `n` are read
    /// through an order the decoder has reached by the time it places them:
    /// the whole block can be read first, through the order it starts with
    /// and without a branch on any coefficient, and the updates applied
    /// afterwards.
    #[inline]
    fn scan(&mut self, input: &[i32; 16], start: u8) -> Result<ScannedBlock, EntropyError> {
        if !(1..=15).contains(&start) {
            return Err(EntropyError::InvalidParameter {
                parameter: "adaptive scan start",
                value: i64::from(start),
            });
        }
        let mut values = [0; 16];
        let mut occupied = 0_u16;
        for (index, (value, &position)) in values.iter_mut().zip(&self.order).enumerate() {
            *value = input[usize::from(position & 15)];
            occupied |= u16::from(*value != 0) << index;
        }
        // Indices before the start belong to another band.
        occupied &= u16::MAX << start;
        let mut remaining = occupied;
        while remaining != 0 {
            self.adapt(remaining.trailing_zeros() as usize);
            remaining &= remaining - 1;
        }
        Ok(ScannedBlock { values, occupied })
    }
}

/// The non-zero coefficients of one block in coded order.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScannedBlock {
    /// The coefficient at each scan index that `occupied` marks; the other
    /// elements are not part of the block.
    values: [i32; 16],
    /// Bit `n` is set when the block has a non-zero coefficient at scan
    /// index `n`.
    occupied: u16,
}

impl ScannedBlock {
    /// The scan indices that hold a non-zero coefficient, one bit each.
    pub(crate) const fn occupied(&self) -> u16 {
        self.occupied
    }

    /// The coefficient at `scan_index`, one [`Self::occupied`] marks.
    #[inline]
    pub(crate) const fn value(&self, scan_index: u32) -> i32 {
        self.values[(scan_index & 15) as usize]
    }

    /// The block whose coefficients are already in coded order: `values[n]`
    /// lies at scan index `start + n`. No scan adapts to such a block.
    pub(crate) fn from_ordered(start: u8, values: &[i32]) -> Result<Self, EntropyError> {
        let end = usize::from(start) + values.len();
        if !(1..=15).contains(&start) || end > 16 {
            return Err(EntropyError::InvalidParameter {
                parameter: "ordered block extent",
                value: i64::try_from(end).unwrap_or(i64::MAX),
            });
        }
        let mut block = Self {
            values: [0; 16],
            occupied: 0,
        };
        block.values[usize::from(start)..end].copy_from_slice(values);
        for (scan_index, &value) in (start..).zip(values) {
            block.occupied |= u16::from(value != 0) << scan_index;
        }
        Ok(block)
    }
}

/// Tile-local adaptive inverse scan for LP coefficients.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveLpScan(Scan);

impl Default for AdaptiveLpScan {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveLpScan {
    /// Creates the scan defined by T.832 Tables 107 through 109.
    #[must_use]
    pub const fn new() -> Self {
        Self(Scan::new(SCAN_ORDER_0))
    }

    /// Restores scan order and totals at a tile boundary.
    pub fn reset_tile(&mut self) {
        *self = Self::new();
    }

    /// Restores only the totals at a 16-macroblock column boundary.
    pub fn reset_totals(&mut self) {
        self.0.totals = SCAN_TOTALS;
    }

    /// Places one parsed coefficient and performs the adjacent scan update.
    #[inline]
    pub fn place(
        &mut self,
        output: &mut [i32; 16],
        scan_index: u8,
        value: i32,
    ) -> Result<(), EntropyError> {
        self.0.place(output, scan_index, value)
    }

    /// Collects the non-zero coefficients of `input` from scan position
    /// `start` and performs the scan updates a decoder makes placing them.
    #[inline]
    pub(crate) fn scan(
        &mut self,
        input: &[i32; 16],
        start: u8,
    ) -> Result<ScannedBlock, EntropyError> {
        self.0.scan(input, start)
    }

    /// Returns the current inverse-scan permutation.
    #[must_use]
    pub const fn order(&self) -> &[u8; 16] {
        &self.0.order
    }
}

/// HP prediction direction selecting one of the two adaptive scans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HpScanDirection {
    /// The scan of a macroblock predicted from the left or not at all.
    Horizontal,
    /// The scan of a macroblock predicted from the top.
    Vertical,
}

impl HpScanDirection {
    /// The scan the high-pass blocks of a macroblock whose high-pass
    /// prediction direction is `mode` are coded in.
    pub(crate) const fn for_mode(mode: PredictionMode) -> Self {
        match mode {
            PredictionMode::FromTop => Self::Vertical,
            _ => Self::Horizontal,
        }
    }
}

/// Tile-local horizontal and vertical inverse scans for HP coefficients.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveHpScan {
    horizontal: Scan,
    vertical: Scan,
}

impl Default for AdaptiveHpScan {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveHpScan {
    /// Creates both scans defined by T.832 Tables 107, 108, and 110.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            horizontal: Scan::new(SCAN_ORDER_0),
            vertical: Scan::new(SCAN_ORDER_1),
        }
    }

    /// Restores both scan orders and totals at a tile boundary.
    pub fn reset_tile(&mut self) {
        *self = Self::new();
    }

    /// Restores only both totals lists at a 16-macroblock column boundary.
    pub fn reset_totals(&mut self) {
        self.horizontal.totals = SCAN_TOTALS;
        self.vertical.totals = SCAN_TOTALS;
    }

    const fn selected(&mut self, direction: HpScanDirection) -> &mut Scan {
        match direction {
            HpScanDirection::Horizontal => &mut self.horizontal,
            HpScanDirection::Vertical => &mut self.vertical,
        }
    }

    /// Places one parsed coefficient into the selected HP scan.
    #[inline]
    pub fn place(
        &mut self,
        direction: HpScanDirection,
        output: &mut [i32; 16],
        scan_index: u8,
        value: i32,
    ) -> Result<(), EntropyError> {
        self.selected(direction).place(output, scan_index, value)
    }

    /// Collects the non-zero HP coefficients of `input` through the selected
    /// scan and performs the scan updates a decoder makes placing them.
    #[inline]
    pub(crate) fn scan(
        &mut self,
        direction: HpScanDirection,
        input: &[i32; 16],
    ) -> Result<ScannedBlock, EntropyError> {
        self.selected(direction).scan(input, 1)
    }

    /// Returns the current inverse-scan permutation for `direction`.
    #[must_use]
    pub const fn order(&self, direction: HpScanDirection) -> &[u8; 16] {
        match direction {
            HpScanDirection::Horizontal => &self.horizontal.order,
            HpScanDirection::Vertical => &self.vertical.order,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::encode::test_support::Lcg;

    #[test]
    fn initial_orders_match_table_107() {
        assert_eq!(AdaptiveLpScan::new().order(), &SCAN_ORDER_0);
        let hp = AdaptiveHpScan::new();
        assert_eq!(hp.order(HpScanDirection::Horizontal), &SCAN_ORDER_0);
        assert_eq!(hp.order(HpScanDirection::Vertical), &SCAN_ORDER_1);
    }

    #[test]
    fn scan_swaps_only_when_total_overtakes_predecessor() {
        let mut scan = AdaptiveLpScan::new();
        let mut output = [0; 16];
        scan.place(&mut output, 2, 9).unwrap();
        scan.place(&mut output, 2, 9).unwrap();
        scan.place(&mut output, 2, 9).unwrap();
        assert_eq!(output[1], 9);
        assert_eq!(&scan.order()[1..=2], &[1, 4]);
    }

    #[test]
    fn scanning_a_block_is_reading_it_through_the_order_and_placing_each_coefficient() {
        let mut state = 0x1234_5678_9abc_def1_u64;
        let mut written = Scan::new(SCAN_ORDER_0);
        let mut read = Scan::new(SCAN_ORDER_0);
        for round in 0..2_000 {
            let start = 1 + u8::try_from(round % 3).unwrap();
            let input: [i32; 16] = core::array::from_fn(|_| {
                state = Lcg::advance(state);
                if (state >> 40).is_multiple_of(3) {
                    i32::try_from((state >> 50) % 9).unwrap() - 4
                } else {
                    0
                }
            });
            // What a reader holding the same scan is told, coefficient by
            // coefficient in coded order.
            let order = read.order;
            let block = written.scan(&input, start).unwrap();
            let mut expected = 0_u16;
            let mut output = [0_i32; 16];
            for index in start..16 {
                let value = input[usize::from(order[usize::from(index)])];
                if value != 0 {
                    expected |= 1 << index;
                    assert_eq!(block.value(u32::from(index)), value);
                    read.place(&mut output, index, value).unwrap();
                }
            }
            assert_eq!(block.occupied(), expected);
            // The reader put every coefficient back where the writer read it.
            for index in start..16 {
                let position = usize::from(order[usize::from(index)]);
                assert_eq!(output[position], input[position]);
            }
            assert_eq!(read, written);
        }
    }

    #[test]
    fn totals_reset_preserves_adapted_order() {
        let mut scan = AdaptiveLpScan::new();
        // The third coefficient at scan index two overtakes index one.
        for _ in 0..3 {
            scan.place(&mut [0; 16], 2, 1).unwrap();
        }
        let adapted = *scan.order();
        assert_ne!(adapted, SCAN_ORDER_0);
        scan.reset_totals();
        assert_eq!(scan.order(), &adapted);
    }
}
