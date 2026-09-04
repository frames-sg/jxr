//! Adaptive inverse scans from T.832 Tables 107 through 114.

use super::EntropyError;

const SCAN_ORDER_0: [u8; 16] = [0, 4, 1, 5, 8, 2, 9, 6, 12, 3, 10, 13, 7, 14, 11, 15];
const SCAN_ORDER_1: [u8; 16] = [0, 1, 2, 5, 4, 3, 6, 9, 8, 7, 12, 15, 13, 10, 11, 14];
const SCAN_TOTALS: [i32; 16] = [0, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4];

fn place_and_adapt(
    order: &mut [u8; 16],
    totals: &mut [i32; 16],
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
    output[usize::from(order[index])] = value;
    totals[index] += 1;
    if index > 1 && totals[index] > totals[index - 1] {
        totals.swap(index, index - 1);
        order.swap(index, index - 1);
    }
    Ok(())
}

/// Tile-local adaptive inverse scan for LP coefficients.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveLpScan {
    order: [u8; 16],
    totals: [i32; 16],
}

impl Default for AdaptiveLpScan {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveLpScan {
    /// Creates the scan defined by T.832 Tables 107 through 109.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            order: SCAN_ORDER_0,
            totals: SCAN_TOTALS,
        }
    }

    /// Restores scan order and totals at a tile boundary.
    pub fn reset_tile(&mut self) {
        *self = Self::new();
    }

    /// Restores only the totals at a 16-macroblock column boundary.
    pub fn reset_totals(&mut self) {
        self.totals = SCAN_TOTALS;
    }

    /// Places one parsed coefficient and performs the adjacent scan update.
    pub fn place(
        &mut self,
        output: &mut [i32; 16],
        scan_index: u8,
        value: i32,
    ) -> Result<(), EntropyError> {
        place_and_adapt(&mut self.order, &mut self.totals, output, scan_index, value)
    }

    /// Returns the current inverse-scan permutation.
    #[must_use]
    pub const fn order(&self) -> &[u8; 16] {
        &self.order
    }
}

/// HP prediction direction selecting one of the two adaptive scans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HpScanDirection {
    /// Prediction from the left, using the horizontal scan.
    Horizontal,
    /// Prediction from the top, using the vertical scan.
    Vertical,
}

/// Tile-local horizontal and vertical inverse scans for HP coefficients.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveHpScan {
    horizontal_order: [u8; 16],
    vertical_order: [u8; 16],
    horizontal_totals: [i32; 16],
    vertical_totals: [i32; 16],
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
            horizontal_order: SCAN_ORDER_0,
            vertical_order: SCAN_ORDER_1,
            horizontal_totals: SCAN_TOTALS,
            vertical_totals: SCAN_TOTALS,
        }
    }

    /// Restores both scan orders and totals at a tile boundary.
    pub fn reset_tile(&mut self) {
        *self = Self::new();
    }

    /// Restores only both totals lists at a 16-macroblock column boundary.
    pub fn reset_totals(&mut self) {
        self.horizontal_totals = SCAN_TOTALS;
        self.vertical_totals = SCAN_TOTALS;
    }

    /// Places one parsed coefficient into the selected HP scan.
    pub fn place(
        &mut self,
        direction: HpScanDirection,
        output: &mut [i32; 16],
        scan_index: u8,
        value: i32,
    ) -> Result<(), EntropyError> {
        match direction {
            HpScanDirection::Horizontal => place_and_adapt(
                &mut self.horizontal_order,
                &mut self.horizontal_totals,
                output,
                scan_index,
                value,
            ),
            HpScanDirection::Vertical => place_and_adapt(
                &mut self.vertical_order,
                &mut self.vertical_totals,
                output,
                scan_index,
                value,
            ),
        }
    }

    /// Returns the current inverse-scan permutation for `direction`.
    #[must_use]
    pub const fn order(&self, direction: HpScanDirection) -> &[u8; 16] {
        match direction {
            HpScanDirection::Horizontal => &self.horizontal_order,
            HpScanDirection::Vertical => &self.vertical_order,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn totals_reset_preserves_adapted_order() {
        let mut scan = AdaptiveLpScan::new();
        scan.place(&mut [0; 16], 2, 1).unwrap();
        let adapted = *scan.order();
        scan.reset_totals();
        assert_eq!(scan.order(), &adapted);
        scan.reset_tile();
        assert_eq!(scan.order(), &SCAN_ORDER_0);
    }
}
