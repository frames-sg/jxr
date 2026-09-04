use jxr_core::{
    AlphaHandling, AlphaMode, BandPresence, ChannelLayout, ChromaSampling, ColorFormat,
    DecodeScale, OverlapMode, PixelFormat, Rect,
};

use super::{
    coefficients_per_macroblock, host_output_bytes, indexed_tile_packet_ranges,
    native_output_region, reconstruction_bounds, reconstruction_region, should_prepare_alpha,
};

#[test]
fn coded_region_bounds_include_overlap_and_chroma_halos() {
    let region = Rect {
        x: 40,
        y: 16,
        w: 8,
        h: 16,
    };
    assert_eq!(
        reconstruction_bounds([8, 0], [8, 8], OverlapMode::One, 0, 0, region).unwrap(),
        [2, 0, 5, 3]
    );
    assert_eq!(
        reconstruction_bounds([8, 0], [8, 8], OverlapMode::None, 1, 7, region).unwrap(),
        [2, 0, 5, 3]
    );
    assert_eq!(
        reconstruction_bounds([8, 0], [8, 8], OverlapMode::None, 1, 1, region).unwrap(),
        [3, 1, 4, 2]
    );
}

#[test]
fn integrated_alpha_is_prepared_even_when_output_drops_it() {
    assert!(should_prepare_alpha(
        AlphaMode::Integrated,
        AlphaHandling::Drop
    ));
    assert!(!should_prepare_alpha(
        AlphaMode::Separate,
        AlphaHandling::Drop
    ));
}

#[test]
fn planar_output_allocation_matches_normative_plane_extents() {
    let region = Rect {
        x: 0,
        y: 0,
        w: 5,
        h: 3,
    };
    let format = PixelFormat::U8(ChannelLayout::Yuva(ChromaSampling::Cs420));
    assert_eq!(host_output_bytes(format, 1, true, region).unwrap(), 34);
    assert_eq!(host_output_bytes(format, 1, false, region).unwrap(), 19);
    assert_eq!(host_output_bytes(format, 2, false, region).unwrap(), 27);
}

#[test]
fn coefficient_accounting_matches_normative_macroblock_shapes() {
    assert_eq!(
        coefficients_per_macroblock(ColorFormat::Yuv(ChromaSampling::Cs420), BandPresence::All,)
            .unwrap(),
        384
    );
    assert_eq!(
        coefficients_per_macroblock(
            ColorFormat::Yuv(ChromaSampling::Cs422),
            BandPresence::NoHighPass,
        )
        .unwrap(),
        32
    );
    assert_eq!(
        coefficients_per_macroblock(ColorFormat::Luma, BandPresence::DcOnly).unwrap(),
        1
    );
}

#[test]
fn overlap_region_includes_one_macroblock_halo() {
    let region = reconstruction_region(
        Rect {
            x: 32,
            y: 32,
            w: 16,
            h: 16,
        },
        (128, 128),
        OverlapMode::One,
    );
    assert_eq!(
        region,
        Rect {
            x: 16,
            y: 16,
            w: 48,
            h: 48,
        }
    );
}

#[test]
fn native_output_region_covers_source_coordinates_at_transform_scale() {
    let source = Rect {
        x: 3,
        y: 5,
        w: 17,
        h: 33,
    };
    assert_eq!(
        native_output_region(source, DecodeScale::Full).unwrap(),
        source
    );
    assert_eq!(
        native_output_region(source, DecodeScale::Quarter).unwrap(),
        Rect {
            x: 0,
            y: 1,
            w: 5,
            h: 9,
        }
    );
}

#[test]
fn frequency_index_order_is_not_forced_into_spatial_tile_extents() {
    let ranges = indexed_tile_packet_ranges(100, 400, &[0, 200, 50, 250], 2, true).unwrap();
    assert_eq!(ranges.len(), 2);
    assert_eq!(ranges[0].offset, 100);
    assert_eq!(ranges[0].length, 300);
    assert_eq!(ranges[1], ranges[0]);
}

#[test]
fn spatial_index_supports_physical_reordering_and_shared_packets() {
    let ranges = indexed_tile_packet_ranges(100, 400, &[200, 0, 50, 0], 4, false).unwrap();
    assert_eq!(ranges[0], jxr_core::ByteRange::new(300, 100, 400).unwrap());
    assert_eq!(ranges[1], jxr_core::ByteRange::new(100, 50, 400).unwrap());
    assert_eq!(ranges[2], jxr_core::ByteRange::new(150, 150, 400).unwrap());
    assert_eq!(ranges[3], ranges[1]);
}
