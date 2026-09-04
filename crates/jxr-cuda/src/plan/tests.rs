// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use jxr_core::{
    AlphaMode, BackendRequest, BandPresence, BitstreamMode, ByteRange, ChannelLayout,
    CoefficientArena, CoefficientPlane, ColorFormat, CropWindow, DecodeScale, ImageInfo,
    ImageMetadata, MacroblockMetadata, OutputBitDepth, OutputFormatRequest, OverlapMode,
    PixelFormat, PlaneInfo, PlanePlan, PredictionMode, PreparedPlan, QuantizerSet, Rect,
    SampleFormat, SurfaceLayout, TileEdgeFlags, TileGrid, TilePlan,
};

use super::{CudaDecodePlan, build_arenas};
use crate::CudaError;

#[test]
fn rejects_invalid_prediction_before_any_cuda_resource_can_be_created() {
    let mut arena = luma_arena(&[(0, 0)]);
    Arc::get_mut(&mut arena).unwrap().macroblocks.hp_predictions[0] = PredictionMode::FromTopLeft;
    let prepared = prepared_plan(1, 1);
    let error = cuda_plan(arena, &prepared).unwrap_err();
    assert!(matches!(
        error,
        CudaError::InvalidPlan {
            reason: "top-left high-pass prediction is invalid"
        }
    ));
}

#[test]
fn rejects_duplicate_macroblock_coordinates_that_would_race_device_writes() {
    let arena = luma_arena(&[(0, 0), (0, 0), (0, 1), (1, 1)]);
    let prepared = prepared_plan(2, 2);
    let error = cuda_plan(arena, &prepared).unwrap_err();
    assert!(matches!(
        error,
        CudaError::InvalidPlan {
            reason: "coefficient plane repeats a macroblock coordinate"
        }
    ));
}

#[test]
fn rejects_a_crop_outside_the_uploaded_sample_window() {
    let arena = luma_arena(&[(1, 0)]);
    let prepared = prepared_plan(1, 1);
    let error = cuda_plan(arena, &prepared).unwrap_err();
    assert!(matches!(
        error,
        CudaError::InvalidPlan {
            reason: "output crop is outside a CUDA sample plane"
        }
    ));
}

#[test]
fn rejects_a_macroblock_coefficient_span_outside_the_upload() {
    let mut arena = luma_arena(&[(0, 0)]);
    Arc::get_mut(&mut arena)
        .unwrap()
        .macroblocks
        .coefficient_offsets[0] = 1;
    let prepared = prepared_plan(1, 1);
    let error = cuda_plan(arena, &prepared).unwrap_err();
    assert!(matches!(
        error,
        CudaError::InvalidPlan {
            reason: "macroblock coefficient span exceeds the uploaded arena"
        }
    ));
}

#[test]
fn separate_alpha_retains_its_own_overlap_and_tile_policy() {
    let mut primary_plan = prepared_plan(1, 1);
    primary_plan.info.alpha_mode = AlphaMode::Separate;
    primary_plan.info.alpha = Some(primary_plan.info.primary.clone());
    let mut alpha_plan = prepared_plan(1, 1);
    alpha_plan.primary.overlap = OverlapMode::Two;
    alpha_plan.info.primary.overlap = OverlapMode::Two;
    alpha_plan.info.tiles.hard_tiles = true;
    let arenas = build_arenas(
        luma_arena(&[(0, 0)]),
        Some((luma_arena(&[(0, 0)]), alpha_plan)),
        &primary_plan,
        1,
    )
    .unwrap();
    assert_eq!(arenas[0].overlap, OverlapMode::None);
    assert_eq!(arenas[1].overlap, OverlapMode::Two);
    assert!(arenas[1].hard_tiles);
}

fn cuda_plan(
    arena: Arc<CoefficientArena>,
    prepared: &PreparedPlan,
) -> Result<CudaDecodePlan, CudaError> {
    let policy = OutputFormatRequest {
        internal_color: ColorFormat::Luma,
        output_color: ColorFormat::Luma,
        bit_depth: OutputBitDepth::U8,
        pixel_format: PixelFormat::U8(ChannelLayout::Luma),
        scaled: false,
        alpha_format: None,
        red_blue_not_swapped: true,
        premultiply_alpha: false,
        crop: CropWindow {
            x: 0,
            y: 0,
            width: prepared.output_region.w,
            height: prepared.output_region.h,
        },
    };
    CudaDecodePlan::from_prepared(
        arena,
        None,
        prepared,
        policy,
        SurfaceLayout::for_output(policy, 1).unwrap(),
        [0, 0],
        BackendRequest::Cuda,
    )
}

fn luma_arena(coordinates: &[(u32, u32)]) -> Arc<CoefficientArena> {
    let count = coordinates.len();
    Arc::new(CoefficientArena {
        coefficients: vec![0; count],
        macroblocks: MacroblockMetadata {
            coefficient_offsets: (0..u32::try_from(count).unwrap()).collect(),
            quantizers: vec![
                QuantizerSet {
                    dc: 1,
                    low_pass: 1,
                    high_pass: 1,
                };
                count
            ],
            bands: vec![BandPresence::DcOnly; count],
            predictions: vec![PredictionMode::None; count],
            hp_predictions: vec![PredictionMode::None; count],
            tile_edges: vec![TileEdgeFlags::default(); count],
            coded_x: coordinates.iter().map(|&(x, _)| x).collect(),
            coded_y: coordinates.iter().map(|&(_, y)| y).collect(),
            output_x: coordinates.iter().map(|&(x, _)| x * 16).collect(),
            output_y: coordinates.iter().map(|&(_, y)| y * 16).collect(),
        },
        planes: vec![CoefficientPlane {
            coefficient_offset: 0,
            coefficient_count: count,
            macroblock_offset: 0,
            macroblock_count: count,
            block_columns: 4,
            block_rows: 4,
        }],
    })
}

fn prepared_plan(macroblocks_x: u32, macroblocks_y: u32) -> PreparedPlan {
    let width = macroblocks_x * 16;
    let height = macroblocks_y * 16;
    let macroblock_count_u32 = macroblocks_x * macroblocks_y;
    let macroblock_count = usize::try_from(macroblock_count_u32).unwrap();
    let primary = PlaneInfo {
        color_format: ColorFormat::Luma,
        sample_format: SampleFormat::Unsigned { bits: 8 },
        bands: BandPresence::DcOnly,
        bitstream_mode: BitstreamMode::Spatial,
        overlap: OverlapMode::None,
        short_header: false,
        long_word: false,
        scaled: false,
        chroma_centering: [0, 0],
        shift_bits: 0,
        mantissa_length: 0,
        exponent_bias: 0,
        width,
        height,
    };
    let region = Rect::full((width, height));
    PreparedPlan {
        info: ImageInfo {
            width,
            height,
            profile: None,
            level: None,
            primary: primary.clone(),
            alpha_mode: AlphaMode::None,
            premultiplied_alpha: false,
            alpha: None,
            tiles: TileGrid {
                column_widths: vec![macroblocks_x],
                row_heights: vec![macroblocks_y],
                hard_tiles: false,
            },
            metadata: ImageMetadata::default(),
        },
        codestream_range: ByteRange {
            offset: 0,
            length: 1,
        },
        primary: PlanePlan {
            width,
            height,
            macroblocks_x,
            macroblocks_y,
            overlap: OverlapMode::None,
            coefficient_plane: 0,
        },
        alpha: None,
        tiles: vec![TilePlan {
            packet_range: ByteRange {
                offset: 0,
                length: 1,
            },
            output_region: region,
            macroblock_start: 0,
            macroblock_count: macroblock_count_u32,
            hard_boundaries: false,
            required_for_reconstruction: true,
        }],
        reconstruction_region: region,
        output_region: region,
        decoded_region: region,
        scale: DecodeScale::Full,
        coefficient_bytes: macroblock_count * core::mem::size_of::<i32>(),
    }
}
