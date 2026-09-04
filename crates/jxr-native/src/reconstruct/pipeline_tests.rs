use jxr_core::{BandPresence, DecodeScale, OverlapMode, QuantizerSet};

use super::{
    QuantizedMacroblock, ReconstructionConfig, ReconstructionPipelineWorkspace, TilePartition,
    reconstruct_luma, reconstruct_luma_scaled, reconstruct_luma_scaled_with_cpu,
};
use crate::CpuCapabilities;

fn dc_macroblock(value: i32, step: u32) -> QuantizedMacroblock {
    let mut dc_low_pass = [0; 16];
    dc_low_pass[0] = value;
    QuantizedMacroblock {
        dc_low_pass,
        high_pass: [0; 256],
        quantizers: QuantizerSet {
            dc: step,
            low_pass: 1,
            high_pass: 1,
        },
        bands: BandPresence::DcOnly,
    }
}

#[test]
fn dc_only_macroblock_reconstructs_constant_plane() {
    let config = ReconstructionConfig {
        macroblock_origin_x: 2,
        macroblock_origin_y: 3,
        macroblocks_x: 1,
        macroblocks_y: 1,
        block_columns: 4,
        block_rows: 4,
        scale_after_first_transform: false,
        overlap: OverlapMode::None,
        tiles: TilePartition::single(1, 1),
    };
    let plane = reconstruct_luma(&[dc_macroblock(256, 1)], &config).unwrap();
    assert_eq!((plane.origin_x, plane.origin_y), (32, 48));
    assert_eq!((plane.width, plane.height), (16, 16));
    assert_eq!(plane.samples, vec![16; 256]);
}

#[test]
fn native_quarter_reconstruction_stops_after_the_first_inverse_transform() {
    let config = ReconstructionConfig {
        macroblock_origin_x: 2,
        macroblock_origin_y: 3,
        macroblocks_x: 1,
        macroblocks_y: 1,
        block_columns: 4,
        block_rows: 4,
        scale_after_first_transform: false,
        overlap: OverlapMode::Two,
        tiles: TilePartition::single(1, 1),
    };
    let mut macroblock = dc_macroblock(16, 1);
    macroblock.bands = BandPresence::NoHighPass;
    let plane = reconstruct_luma_scaled(&[macroblock], &config, DecodeScale::Quarter).unwrap();
    assert_eq!((plane.origin_x, plane.origin_y), (8, 12));
    assert_eq!((plane.width, plane.height), (4, 4));
    assert_eq!(plane.samples, vec![4; 16]);
}

#[test]
fn native_sixteenth_reconstruction_applies_both_dc_gain_steps_without_expansion() {
    let config = ReconstructionConfig {
        macroblock_origin_x: 2,
        macroblock_origin_y: 3,
        macroblocks_x: 1,
        macroblocks_y: 1,
        block_columns: 4,
        block_rows: 4,
        scale_after_first_transform: false,
        overlap: OverlapMode::Two,
        tiles: TilePartition::single(1, 1),
    };
    let plane =
        reconstruct_luma_scaled(&[dc_macroblock(256, 1)], &config, DecodeScale::Sixteenth).unwrap();
    assert_eq!((plane.origin_x, plane.origin_y), (2, 3));
    assert_eq!((plane.width, plane.height), (1, 1));
    assert_eq!(plane.samples, vec![16]);
}

#[test]
fn macroblock_count_is_checked_before_allocation() {
    let config = ReconstructionConfig {
        macroblock_origin_x: 0,
        macroblock_origin_y: 0,
        macroblocks_x: 2,
        macroblocks_y: 1,
        block_columns: 4,
        block_rows: 4,
        scale_after_first_transform: false,
        overlap: OverlapMode::None,
        tiles: TilePartition::single(2, 1),
    };
    assert!(reconstruct_luma(&[dc_macroblock(1, 1)], &config).is_err());
}

#[test]
fn repeated_full_reconstruction_reuses_transform_and_output_scratch() {
    let config = ReconstructionConfig {
        macroblock_origin_x: 0,
        macroblock_origin_y: 0,
        macroblocks_x: 1,
        macroblocks_y: 1,
        block_columns: 4,
        block_rows: 4,
        scale_after_first_transform: false,
        overlap: OverlapMode::Two,
        tiles: TilePartition::single(1, 1),
    };
    let macroblocks = [dc_macroblock(256, 1)];
    let mut workspace = ReconstructionPipelineWorkspace::default();

    let first = reconstruct_luma_scaled_with_cpu(
        &macroblocks,
        &config,
        DecodeScale::Full,
        CpuCapabilities::detect(),
        &mut workspace,
    )
    .unwrap();
    let first_reuses = workspace.reuses();
    let expected_samples = first.samples.clone();
    workspace.recycle_samples(first.samples);
    let retained_bytes = workspace.retained_bytes();
    let retained_output_bytes = workspace.retained_output_bytes();
    let second = reconstruct_luma_scaled_with_cpu(
        &macroblocks,
        &config,
        DecodeScale::Full,
        CpuCapabilities::detect(),
        &mut workspace,
    )
    .unwrap();

    assert_eq!(second.samples, expected_samples);
    assert_eq!(first_reuses, 0);
    assert!(workspace.reuses() > first_reuses);
    assert!(retained_bytes > 0);
    assert!(retained_output_bytes > 0);
    assert_eq!(workspace.output_reuses(), 1);
    workspace.recycle_samples(second.samples);
    assert_eq!(workspace.retained_output_bytes(), retained_output_bytes);
    assert_eq!(workspace.retained_bytes(), retained_bytes);
}
