use jxr_core::{BandPresence, DecodeScale, OverlapMode, QuantizerSet};

use jxr_math::quantization::Quantizer;

use super::{
    QuantizedMacroblock, ReconstructionConfig, ReconstructionError,
    ReconstructionPipelineWorkspace, TilePartition, pipeline::dequantize_high_pass_into,
    reconstruct_luma, reconstruct_luma_scaled, reconstruct_luma_scaled_with_cpu,
};
use crate::CpuCapabilities;
use crate::entropy::encode::test_support::Lcg;

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
    // The picture stops at one sample for each block, and the overlap
    // filter of the samples does not run. Its samples are still those of
    // full reconstruction, which the filters of both levels raise from
    // sixteen to between 32 and 36, and to 33 on average over every block.
    let mut macroblock = dc_macroblock(256, 1);
    macroblock.bands = BandPresence::NoHighPass;
    let full = reconstruct_luma(&[macroblock.clone()], &config).unwrap();
    assert_eq!(full.samples.iter().min(), Some(&32));
    assert_eq!(full.samples.iter().max(), Some(&36));
    assert_eq!(unit_means(&full, (4, 4)), vec![33; 16]);
    let plane = reconstruct_luma_scaled(&[macroblock], &config, DecodeScale::Quarter).unwrap();
    assert_eq!((plane.origin_x, plane.origin_y), (8, 12));
    assert_eq!((plane.width, plane.height), (4, 4));
    // What the block transform would round away is all that is missing.
    assert_eq!(
        plane.samples,
        [
            33, 33, 33, 33, 34, 34, 34, 34, 33, 33, 33, 33, 33, 34, 34, 34
        ]
    );
}

/// A macroblock of subsampled chroma has fewer blocks, and its transform a
/// smaller DC gain, than a full-resolution one. Every reduced resolution
/// must still return the value full reconstruction gives its samples.
#[test]
fn reduced_reconstruction_of_subsampled_chroma_keeps_the_sample_scale() {
    for (block_columns, block_rows) in [(4, 4), (2, 4), (2, 2)] {
        let config = ReconstructionConfig {
            macroblock_origin_x: 0,
            macroblock_origin_y: 0,
            macroblocks_x: 1,
            macroblocks_y: 1,
            block_columns,
            block_rows,
            scale_after_first_transform: false,
            overlap: OverlapMode::None,
            tiles: TilePartition::single(1, 1),
        };
        // The DC of a macroblock whose samples are all sixteen.
        let dc = if block_columns == 4 { 256 } else { 128 };
        let blocks = usize::from(block_columns) * usize::from(block_rows);
        let mut macroblock = dc_macroblock(dc, 1);
        macroblock.bands = BandPresence::NoHighPass;
        let full = reconstruct_luma(&[macroblock.clone()], &config).unwrap();
        assert_eq!(
            full.samples,
            vec![16; blocks * 16],
            "{block_columns}x{block_rows}"
        );
        let quarter =
            reconstruct_luma_scaled(&[macroblock.clone()], &config, DecodeScale::Quarter).unwrap();
        assert_eq!(
            quarter.samples,
            vec![16; blocks],
            "{block_columns}x{block_rows}"
        );
        let sixteenth =
            reconstruct_luma_scaled(&[macroblock], &config, DecodeScale::Sixteenth).unwrap();
        assert_eq!(sixteenth.samples, vec![16], "{block_columns}x{block_rows}");
    }
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
    // Both transform stages spread a DC of 256 to samples of sixteen, and
    // the overlap filters of both levels raise those to 33 on average,
    // which is the sample the macroblock stands for.
    let full = reconstruct_luma(&[dc_macroblock(256, 1)], &config).unwrap();
    assert_eq!(unit_means(&full, (16, 16)), [33]);
    let plane =
        reconstruct_luma_scaled(&[dc_macroblock(256, 1)], &config, DecodeScale::Sixteenth).unwrap();
    assert_eq!((plane.origin_x, plane.origin_y), (2, 3));
    assert_eq!((plane.width, plane.height), (1, 1));
    assert_eq!(plane.samples, vec![33]);
    // Without overlap filtering the transform gain is all there is.
    let unfiltered = ReconstructionConfig {
        overlap: OverlapMode::None,
        ..config
    };
    let plane = reconstruct_luma_scaled(
        &[dc_macroblock(256, 1)],
        &unfiltered,
        DecodeScale::Sixteenth,
    )
    .unwrap();
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

#[test]
fn a_reused_reduced_workspace_tracks_geometry_and_resolution_changes() {
    let base = reduced_config(
        (4, 4),
        OverlapMode::Two,
        (5, 5, vec![0, 2, 5], vec![0, 1, 5], false),
    );
    let mut configurations = vec![base.clone()];
    let mut next = base.clone();
    next.tiles.hard_boundaries = true;
    configurations.push(next.clone());
    next.tiles.column_boundaries = vec![0, 1, 5];
    configurations.push(next.clone());
    next.tiles.row_boundaries = vec![0, 2, 5];
    configurations.push(next.clone());
    next.block_columns = 2;
    configurations.push(next.clone());
    next.block_rows = 2;
    configurations.push(next.clone());
    next.scale_after_first_transform = true;
    configurations.push(next);
    configurations.push(reduced_config(
        (4, 4),
        OverlapMode::Two,
        (6, 4, vec![0, 2, 6], vec![0, 1, 4], false),
    ));
    for overlap in [OverlapMode::One, OverlapMode::None, OverlapMode::Two] {
        configurations.push(ReconstructionConfig {
            overlap,
            ..base.clone()
        });
    }
    let cases = configurations
        .into_iter()
        .map(|config| (config, DecodeScale::Quarter))
        // These keep the same geometry while changing only the resolution.
        .chain([
            (base.clone(), DecodeScale::Sixteenth),
            (base, DecodeScale::Quarter),
        ]);
    let cpu = CpuCapabilities::detect().on_calling_thread();
    let mut workspace = ReconstructionPipelineWorkspace::default();
    for (config, scale) in cases {
        for dc in [777, -1_113] {
            let macroblocks =
                vec![dc_macroblock(dc, 1); (config.macroblocks_x * config.macroblocks_y) as usize];
            let expected = reconstruct_luma_scaled(&macroblocks, &config, scale).unwrap();
            let actual =
                reconstruct_luma_scaled_with_cpu(&macroblocks, &config, scale, cpu, &mut workspace)
                    .unwrap();
            assert_eq!(actual, expected, "{config:?} {scale:?} DC {dc}");
            workspace.recycle_samples(actual.samples);
        }
    }
}

/// Reduced reconstruction leaves out the overlap filtering of the stages
/// it does not run, which is no power of two away from a flat area's level.
/// Whatever the overlap mode, the block geometry, and the weight of the
/// component, the reduced samples of a flat area must be the samples full
/// reconstruction gives it, but for rounding.
#[test]
fn reduced_reconstruction_of_a_flat_area_matches_full_reconstruction_under_overlap() {
    const MACROBLOCKS: u32 = 6;
    for overlap in [OverlapMode::None, OverlapMode::One, OverlapMode::Two] {
        for (block_columns, block_rows) in [(4, 4), (2, 4), (2, 2)] {
            for scale_after_first_transform in [false, true] {
                for dc in [3_000, -1_777, 40_001] {
                    let config = ReconstructionConfig {
                        macroblock_origin_x: 0,
                        macroblock_origin_y: 0,
                        macroblocks_x: MACROBLOCKS,
                        macroblocks_y: MACROBLOCKS,
                        block_columns,
                        block_rows,
                        scale_after_first_transform,
                        overlap,
                        tiles: TilePartition::single(MACROBLOCKS, MACROBLOCKS),
                    };
                    let case = format!(
                        "{overlap:?} {block_columns}x{block_rows} doubled \
                         {scale_after_first_transform} DC {dc}"
                    );
                    let mut macroblock = dc_macroblock(dc, 1);
                    macroblock.bands = BandPresence::NoHighPass;
                    let macroblocks = vec![macroblock; (MACROBLOCKS * MACROBLOCKS) as usize];
                    let centre = |plane: &super::PlanarSamples| {
                        let index = (plane.height / 2 * plane.width + plane.width / 2) as usize;
                        plane.samples[index]
                    };
                    let full = centre(&reconstruct_luma(&macroblocks, &config).unwrap());
                    for scale in [DecodeScale::Quarter, DecodeScale::Sixteenth] {
                        let reduced =
                            reconstruct_luma_scaled(&macroblocks, &config, scale).unwrap();
                        let reduced = centre(&reduced);
                        // Each filter stage rounds a sample by up to a half.
                        assert!(
                            reduced.abs_diff(full) <= 2,
                            "{case} {scale:?}: {reduced} for {full}"
                        );
                        // The area is flat, so its level is not that of
                        // the coefficient unless nothing was filtered.
                        if overlap != OverlapMode::None {
                            let unfiltered = if block_columns == 4 { dc >> 4 } else { dc >> 3 };
                            let doubled = i32::from(scale_after_first_transform) + 1;
                            assert!(full.abs_diff(unfiltered * doubled) > 16, "{case}");
                        }
                    }
                }
            }
        }
    }
}

const UNIT_STEPS: QuantizerSet = QuantizerSet {
    dc: 1,
    low_pass: 1,
    high_pass: 1,
};

/// A macroblock grid, its tile column and row boundaries, and whether
/// those are hard.
type Layout = (u32, u32, Vec<u32>, Vec<u32>, bool);

/// The layouts a reduced decode must be right at every position of: single
/// macroblocks, images one macroblock wide or high, tiles of one
/// macroblock, and regions long enough for their middle to be folded onto
/// the probe of the sample scale.
fn reduced_layouts() -> Vec<Layout> {
    let tiled = |hard| {
        [
            (7, 6, vec![0, 1, 3, 7], vec![0, 2, 3, 6], hard),
            (9, 8, vec![0, 1, 2, 9], vec![0, 6, 8], hard),
        ]
    };
    let mut layouts = vec![
        (1, 1, vec![0, 1], vec![0, 1], false),
        (2, 1, vec![0, 2], vec![0, 1], false),
        (1, 3, vec![0, 1], vec![0, 3], false),
    ];
    layouts.extend(tiled(false));
    layouts.extend(tiled(true));
    layouts
}

fn reduced_config(
    (block_columns, block_rows): (u8, u8),
    overlap: OverlapMode,
    (macroblocks_x, macroblocks_y, columns, rows, hard_boundaries): Layout,
) -> ReconstructionConfig {
    ReconstructionConfig {
        macroblock_origin_x: 0,
        macroblock_origin_y: 0,
        macroblocks_x,
        macroblocks_y,
        block_columns,
        block_rows,
        scale_after_first_transform: false,
        overlap,
        tiles: TilePartition {
            column_boundaries: columns,
            row_boundaries: rows,
            hard_boundaries,
        },
    }
}

fn sample_count(config: &ReconstructionConfig) -> usize {
    (config.macroblocks_x * config.macroblocks_y) as usize
        * usize::from(config.block_columns)
        * usize::from(config.block_rows)
        * 16
}

/// Samples across and down what one sample of a reduced picture of a
/// component stands for.
fn reduced_unit(config: &ReconstructionConfig, scale: DecodeScale) -> (usize, usize) {
    match scale {
        DecodeScale::Full => (1, 1),
        DecodeScale::Quarter => (4, 4),
        DecodeScale::Sixteenth => (
            usize::from(config.block_columns) * 4,
            usize::from(config.block_rows) * 4,
        ),
    }
}

/// The mean of every `unit` of a plane, rounded to nearest.
fn unit_means(plane: &super::PlanarSamples, (across, down): (usize, usize)) -> Vec<i32> {
    let width = plane.width as usize;
    let mut means = Vec::new();
    for y in (0..plane.height as usize).step_by(down) {
        for x in (0..width).step_by(across) {
            let sum: i64 = (0..down)
                .flat_map(|row| &plane.samples[(y + row) * width + x..][..across])
                .map(|&sample| i64::from(sample))
                .sum();
            let count = i64::try_from(across * down).unwrap();
            means.push(i32::try_from((2 * sum + count).div_euclid(2 * count)).unwrap());
        }
    }
    means
}

/// The first and last macroblock of the region of `config` that macroblock
/// `position` along an axis with the tile `boundaries` lies in: its tile
/// where tile boundaries are hard, the whole axis otherwise.
fn region_ends(config: &ReconstructionConfig, boundaries: &[u32], position: usize) -> [usize; 2] {
    let position = u32::try_from(position).unwrap();
    let (first, last) = if config.tiles.hard_boundaries {
        let tile = boundaries.iter().rposition(|&start| start <= position);
        let tile = tile.unwrap();
        (boundaries[tile], boundaries[tile + 1] - 1)
    } else {
        (0, boundaries[boundaries.len() - 1] - 1)
    };
    [first as usize, last as usize]
}

/// The coefficients of a flat image are those of no flat field of
/// coefficients: each overlap filter stops at the image edges and at hard
/// tile edges, a region of subsampled chroma one macroblock wide is
/// filtered unlike any other, and the forward filters round upwards.
/// Whatever the position, the overlap mode, the block geometry, and the
/// weight of the component, both reduced pictures of a flat image must be
/// that image.
///
/// They are, but for what the forward filters round away, which differs
/// from level to level and which only the bands a reduced decode leaves
/// unread could return: one level, and two at the corners of a region,
/// where a filter of its own runs. A sixteenth of a region of subsampled
/// chroma one macroblock wide under two levels of overlap, which the
/// encoder refuses to write, is one level further off. So is a component
/// whose low bands are doubled, which no stream codes exactly.
#[test]
fn reduced_reconstruction_of_a_flat_image_is_flat_at_every_position() {
    for overlap in [OverlapMode::None, OverlapMode::One, OverlapMode::Two] {
        for geometry in [(4, 4), (2, 4), (2, 2)] {
            for (layout, doubled) in reduced_layouts()
                .into_iter()
                .map(|layout| (layout, false))
                .chain([(reduced_layouts().remove(5), true)])
            {
                let config = ReconstructionConfig {
                    scale_after_first_transform: doubled,
                    ..reduced_config(geometry, overlap, layout)
                };
                for level in [-128, -78, -1, 0, 62, 127] {
                    let flat = vec![level; sample_count(&config)];
                    let macroblocks = crate::analyze::analyze_component(
                        flat,
                        &config,
                        UNIT_STEPS,
                        BandPresence::All,
                        CpuCapabilities::detect(),
                    )
                    .unwrap();
                    let full = reconstruct_luma(&macroblocks, &config).unwrap();
                    if !doubled {
                        assert!(full.samples.iter().all(|&sample| sample == level));
                    }
                    for scale in [DecodeScale::Quarter, DecodeScale::Sixteenth] {
                        let reduced =
                            reconstruct_luma_scaled(&macroblocks, &config, scale).unwrap();
                        let unit = reduced_unit(&config, scale);
                        let expected = unit_means(&full, unit);
                        assert_eq!(reduced.samples.len(), expected.len());
                        // Reduced samples across and down one macroblock.
                        let across = usize::from(geometry.0) * 4 / unit.0;
                        let down = usize::from(geometry.1) * 4 / unit.1;
                        let width = reduced.width as usize;
                        for (index, (&sample, &expected)) in
                            reduced.samples.iter().zip(&expected).enumerate()
                        {
                            let (x, y) = (index % width, index / width);
                            let columns =
                                region_ends(&config, &config.tiles.column_boundaries, x / across);
                            let rows = region_ends(&config, &config.tiles.row_boundaries, y / down);
                            let corner = (x == columns[0] * across
                                || x + 1 == (columns[1] + 1) * across)
                                && (y == rows[0] * down || y + 1 == (rows[1] + 1) * down);
                            let narrow = geometry.0 == 2
                                && overlap == OverlapMode::Two
                                && columns[0] == columns[1]
                                && scale == DecodeScale::Sixteenth;
                            let tolerance = if overlap == OverlapMode::None || level == 0 {
                                0
                            } else {
                                1 + u32::from(corner) + u32::from(narrow) + u32::from(doubled)
                            };
                            assert!(
                                sample.abs_diff(expected) <= tolerance,
                                "{overlap:?} {geometry:?} {:?} doubled {doubled} {scale:?} \
                                 ({x}, {y}): {sample} for {expected}",
                                config.tiles
                            );
                        }
                    }
                }
            }
        }
    }
}

/// No overlap filter crosses a hard tile boundary, so a hard tile has the
/// reduced pictures of an image of its own. The whole picture here is large
/// enough to be scaled a row to a thread, and each half is not.
#[test]
fn hard_tiles_have_the_reduced_pictures_of_images_of_their_own() {
    let whole = reduced_config(
        (4, 4),
        OverlapMode::Two,
        (33, 32, vec![0, 33], vec![0, 16, 32], true),
    );
    let half = reduced_config(
        (4, 4),
        OverlapMode::Two,
        (33, 16, vec![0, 33], vec![0, 16], true),
    );
    let mut macroblocks = busy_dc_macroblocks(&whole, 77);
    for (index, macroblock) in macroblocks.iter_mut().enumerate() {
        macroblock.bands = BandPresence::NoHighPass;
        macroblock.dc_low_pass[1 + index % 15] = 300 - i32::try_from(index % 601).unwrap();
    }
    let (top, bottom) = macroblocks.split_at(33 * 16);
    for scale in [DecodeScale::Quarter, DecodeScale::Sixteenth] {
        let picture = reconstruct_luma_scaled(&macroblocks, &whole, scale).unwrap();
        if scale == DecodeScale::Quarter {
            assert!(picture.samples.len() >= super::lanes::PARALLEL_SAMPLES);
        }
        let mut halves = reconstruct_luma_scaled(top, &half, scale).unwrap().samples;
        assert!(halves.len() < super::lanes::PARALLEL_SAMPLES);
        halves.extend(
            reconstruct_luma_scaled(bottom, &half, scale)
                .unwrap()
                .samples,
        );
        assert_eq!(picture.samples, halves, "{scale:?}");
    }
}

/// Deterministic DC coefficients that differ widely between neighbouring
/// macroblocks, as those of a busy image do.
fn busy_dc_macroblocks(config: &ReconstructionConfig, seed: u64) -> Vec<QuantizedMacroblock> {
    let mut state = seed;
    (0..config.macroblocks_x * config.macroblocks_y)
        .map(|_| {
            state = Lcg::advance(state);
            dc_macroblock(i32::try_from((state >> 40) % 8_001).unwrap() - 4_000, 1)
        })
        .collect()
}

/// The low-pass-level overlap filter of [`OverlapMode::Two`] reaches half
/// a macroblock into each neighbour: where neighbouring macroblocks
/// differ, it moves a large part of the difference across, and a DC
/// coefficient scaled on its own stands for no part of the picture. A
/// sixteenth-size picture is therefore the quarter-size picture of the DC
/// band, which runs that filter, averaged over each macroblock.
#[test]
fn sixteenth_reconstruction_is_the_quarter_picture_of_the_dc_band_by_macroblock() {
    for geometry in [(4, 4), (2, 4), (2, 2)] {
        for (index, layout) in reduced_layouts().into_iter().enumerate() {
            // Chroma regions one macroblock wide are filtered unlike any
            // other, and their flat fields are not those of the DC band.
            if geometry.0 == 2 && (layout.0 == 1 || (layout.4 && layout.2.contains(&1))) {
                continue;
            }
            let config = reduced_config(geometry, OverlapMode::Two, layout);
            let macroblocks = busy_dc_macroblocks(&config, 50 + index as u64);
            let quarter =
                reconstruct_luma_scaled(&macroblocks, &config, DecodeScale::Quarter).unwrap();
            let blocks = (usize::from(geometry.0), usize::from(geometry.1));
            let expected = unit_means(&quarter, blocks);
            let sixteenth =
                reconstruct_luma_scaled(&macroblocks, &config, DecodeScale::Sixteenth).unwrap();
            let case = format!("{geometry:?} {:?}", config.tiles);
            assert_eq!(sixteenth.samples.len(), expected.len(), "{case}");
            for (sample, expected) in sixteenth.samples.iter().zip(&expected) {
                assert!(
                    sample.abs_diff(*expected) <= 1,
                    "{case}: {sample} for {expected}"
                );
            }
            // The filter is what makes the difference: the coefficients
            // scaled on their own are far from this picture.
            let unfiltered = ReconstructionConfig {
                overlap: OverlapMode::One,
                ..config.clone()
            };
            let unfiltered =
                reconstruct_luma_scaled(&macroblocks, &unfiltered, DecodeScale::Sixteenth).unwrap();
            if expected.len() > 4 {
                let furthest = unfiltered
                    .samples
                    .iter()
                    .zip(&expected)
                    .map(|(sample, expected)| sample.abs_diff(*expected))
                    .max();
                assert!(furthest > Some(64), "{case}: {furthest:?}");
            }
        }
    }
}

/// Soft tile boundaries stop no overlap filter, so they are no edges to
/// the scale either: a softly tiled component has the reduced pictures of
/// the same component in one tile.
#[test]
fn soft_tile_boundaries_do_not_change_a_reduced_picture() {
    for overlap in [OverlapMode::One, OverlapMode::Two] {
        for geometry in [(4, 4), (2, 2)] {
            let tiled = reduced_config(
                geometry,
                overlap,
                (7, 6, vec![0, 1, 3, 7], vec![0, 2, 3, 6], false),
            );
            let single = ReconstructionConfig {
                tiles: TilePartition::single(7, 6),
                ..tiled.clone()
            };
            let macroblocks = busy_dc_macroblocks(&tiled, 9);
            for scale in [DecodeScale::Quarter, DecodeScale::Sixteenth] {
                assert_eq!(
                    reconstruct_luma_scaled(&macroblocks, &tiled, scale).unwrap(),
                    reconstruct_luma_scaled(&macroblocks, &single, scale).unwrap(),
                    "{overlap:?} {geometry:?} {scale:?}"
                );
            }
        }
    }
}

/// Dequantizes the high-pass band of `macroblock`, of `block_columns` by
/// `block_rows` blocks, into the second of two such macroblocks of a
/// macroblock row whose samples start as `-7`, and returns the samples.
fn place_high_pass(
    macroblock: &QuantizedMacroblock,
    block_columns: usize,
    block_rows: usize,
) -> (Vec<i32>, Result<(), ReconstructionError>) {
    let width = 2 * block_columns * 4;
    let mut rows = vec![-7; width * block_rows * 4];
    let result = dequantize_high_pass_into(
        macroblock,
        block_columns,
        block_columns * block_rows,
        &mut rows,
        width,
        block_columns,
    );
    (rows, result)
}

/// The checked product of each high-pass coefficient of `macroblock` at
/// its place in the samples [`place_high_pass`] returns, zero for element
/// zero of each block, and the first macroblock's samples untouched.
fn assert_placed(samples: &[i32], macroblock: &QuantizedMacroblock, block_columns: usize) {
    let quantizer = Quantizer::new(macroblock.quantizers.high_pass).unwrap();
    let width = 2 * block_columns * 4;
    for (index, &sample) in samples.iter().enumerate() {
        let (x, y) = (index % width, index / width);
        let expected = if x < block_columns * 4 {
            -7
        } else {
            let block = y / 4 * block_columns + (x - block_columns * 4) / 4;
            let position = y % 4 * 4 + x % 4;
            if position == 0 {
                0
            } else {
                quantizer
                    .dequantize(macroblock.high_pass[block * 16 + position])
                    .unwrap()
            }
        };
        assert_eq!(sample, expected, "sample ({x}, {y})");
    }
}

fn high_pass_macroblock(high_pass: &[i32; 256], step: u32) -> QuantizedMacroblock {
    QuantizedMacroblock {
        dc_low_pass: [0; 16],
        high_pass: *high_pass,
        quantizers: QuantizerSet {
            dc: 1,
            low_pass: 1,
            high_pass: step,
        },
        bands: BandPresence::All,
    }
}

#[test]
fn high_pass_blocks_are_placed_as_their_checked_products() {
    for (block_columns, block_rows) in [(4, 4), (2, 4), (2, 2)] {
        for step in [1, 2, 3, 37, 113, 65_536] {
            // Negative and positive values, with element zero of each
            // block, which the block DC coefficient replaces, non-zero.
            let high_pass =
                core::array::from_fn(|index| i32::try_from(index * 37 % 255).unwrap() - 127);
            let macroblock = high_pass_macroblock(&high_pass, step);
            let (samples, result) = place_high_pass(&macroblock, block_columns, block_rows);
            result.unwrap();
            assert_placed(&samples, &macroblock, block_columns);
        }
    }
}

#[test]
fn high_pass_dequantization_fails_exactly_where_the_checked_product_does() {
    for step in [
        1_u32,
        2,
        3,
        37,
        113,
        65_536,
        i32::MAX.unsigned_abs(),
        1 << 31,
        u32::MAX,
    ] {
        let quantizer = Quantizer::new(step).unwrap();
        // The inclusive range whose products fit `i32`.
        let [lower, upper] = [i64::from(i32::MIN), i64::from(i32::MAX)]
            .map(|bound| i32::try_from(bound / i64::from(step)).unwrap());
        let candidates = [
            i32::MIN,
            lower.saturating_sub(1),
            lower,
            -1,
            0,
            1,
            upper,
            upper.saturating_add(1),
            i32::MAX,
        ];
        for candidate in candidates {
            // In the first block, in the last, and as element zero, which
            // is never scaled.
            for position in [5, 250, 32] {
                let mut high_pass = [0; 256];
                high_pass[position] = candidate;
                let macroblock = high_pass_macroblock(&high_pass, step);
                let (samples, result) = place_high_pass(&macroblock, 4, 4);
                let expected = if position % 16 == 0 {
                    Ok(0)
                } else {
                    quantizer.dequantize(candidate)
                };
                match expected {
                    Ok(_) => {
                        result.unwrap();
                        assert_placed(&samples, &macroblock, 4);
                    }
                    Err(_) => assert_eq!(
                        result,
                        Err(ReconstructionError::ArithmeticOverflow(
                            "coefficient dequantization"
                        )),
                        "step {step} value {candidate} at {position}"
                    ),
                }
            }
        }
    }
}

#[test]
fn a_macroblock_without_a_high_pass_band_places_zero_blocks() {
    for bands in [BandPresence::DcOnly, BandPresence::NoHighPass] {
        let mut macroblock = high_pass_macroblock(&[i32::MAX; 256], 3);
        macroblock.bands = bands;
        let (samples, result) = place_high_pass(&macroblock, 2, 4);
        result.unwrap();
        assert_placed(&samples, &high_pass_macroblock(&[0; 256], 3), 2);
    }
    // A zero step fails whether or not the band is coded.
    let mut macroblock = high_pass_macroblock(&[0; 256], 0);
    macroblock.bands = BandPresence::DcOnly;
    assert_eq!(
        place_high_pass(&macroblock, 4, 4).1,
        Err(ReconstructionError::ZeroQuantizer("high-pass"))
    );
}
