//! Exact high-pass coefficient prediction shared by reconstruction backends.

use crate::arithmetic::{MathError, checked_add};

/// Direction selected from reconstructed low-pass coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HighpassPrediction {
    /// Do not predict high-pass coefficients.
    None,
    /// Add coefficients from the block immediately to the left.
    FromLeft,
    /// Add coefficients from the block immediately above.
    FromTop,
}

/// Apply in-macroblock JPEG XR high-pass prediction in place.
pub fn predict_high_pass(
    coefficients: &mut [i32; 256],
    direction: HighpassPrediction,
) -> Result<(), MathError> {
    predict_geometry(coefficients, direction, 4, 4)
}

/// Apply in-macroblock prediction to one YUV 4:2:0 chroma macroblock.
pub fn predict_high_pass_420(
    coefficients: &mut [i32; 256],
    direction: HighpassPrediction,
) -> Result<(), MathError> {
    predict_geometry(coefficients, direction, 2, 2)
}

/// Apply in-macroblock prediction to one YUV 4:2:2 chroma macroblock.
pub fn predict_high_pass_422(
    coefficients: &mut [i32; 256],
    direction: HighpassPrediction,
) -> Result<(), MathError> {
    predict_geometry(coefficients, direction, 2, 4)
}

fn predict_geometry(
    coefficients: &mut [i32; 256],
    direction: HighpassPrediction,
    columns: usize,
    rows: usize,
) -> Result<(), MathError> {
    match direction {
        HighpassPrediction::FromLeft => {
            for row in 0..rows {
                for column in 1..columns {
                    predict_block(coefficients, row * columns + column, 1, [4, 8, 12])?;
                }
            }
        }
        HighpassPrediction::FromTop => {
            for row in 1..rows {
                for column in 0..columns {
                    predict_block(coefficients, row * columns + column, columns, [1, 2, 3])?;
                }
            }
        }
        HighpassPrediction::None => {}
    }
    Ok(())
}

fn predict_block(
    coefficients: &mut [i32; 256],
    block: usize,
    block_stride: usize,
    positions: [usize; 3],
) -> Result<(), MathError> {
    for position in positions {
        let destination = block * 16 + position;
        let source = (block - block_stride) * 16 + position;
        coefficients[destination] = checked_add(coefficients[destination], coefficients[source])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subsampled_420_prediction_uses_two_by_two_block_geometry() {
        let mut coefficients = [0_i32; 256];
        coefficients[12] = 5;
        coefficients[16 + 12] = 7;
        coefficients[2 * 16 + 12] = 11;
        coefficients[3 * 16 + 12] = 13;

        predict_high_pass_420(&mut coefficients, HighpassPrediction::FromLeft).unwrap();

        assert_eq!(coefficients[16 + 12], 12);
        assert_eq!(coefficients[3 * 16 + 12], 24);
    }
}
