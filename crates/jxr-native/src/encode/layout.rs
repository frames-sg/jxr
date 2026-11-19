//! Macroblock grid, tile partition, and profile/level declaration.

use jxr_core::{ColorFormat, EncodeRequest, JxrError, JxrErrorKind, OutputBitDepth, Profile};

use crate::{
    CodestreamHeader, ImagePlaneHeader, ProfileLevel, decode::clipped_boundaries,
    header::output_bit_depth, metadata::output_color,
};

/// Samples across and down one macroblock.
pub(super) const MACROBLOCK_SAMPLES: u32 = 16;

/// One tile's rectangle in macroblocks.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TileRect {
    pub(super) x: usize,
    pub(super) y: usize,
    pub(super) width: usize,
    pub(super) height: usize,
}

/// Extended image geometry and its tile partition, in macroblocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ImageLayout {
    pub(super) macroblocks_x: u32,
    pub(super) macroblocks_y: u32,
    /// Width of each tile column, left to right.
    pub(super) tile_columns: Vec<u16>,
    /// Height of each tile row, top to bottom.
    pub(super) tile_rows: Vec<u16>,
    /// Cumulative column boundaries, from zero to `macroblocks_x`.
    pub(super) column_boundaries: Vec<u32>,
    /// Cumulative row boundaries, from zero to `macroblocks_y`.
    pub(super) row_boundaries: Vec<u32>,
}

impl ImageLayout {
    pub(super) fn new(width: u32, height: u32, request: &EncodeRequest) -> Result<Self, JxrError> {
        let macroblocks_x = width.div_ceil(MACROBLOCK_SAMPLES);
        let macroblocks_y = height.div_ceil(MACROBLOCK_SAMPLES);
        let tile_columns = split_axis(macroblocks_x, request.max_tile_macroblocks)?;
        let tile_rows = split_axis(macroblocks_y, request.max_tile_macroblocks)?;
        Ok(Self {
            macroblocks_x,
            macroblocks_y,
            column_boundaries: boundaries(&tile_columns, macroblocks_x)?,
            row_boundaries: boundaries(&tile_rows, macroblocks_y)?,
            tile_columns,
            tile_rows,
        })
    }

    /// Extended plane size in samples.
    pub(super) fn plane_dimensions(&self) -> Result<(usize, usize), JxrError> {
        let samples = |macroblocks: u32| {
            usize::try_from(macroblocks)
                .ok()
                .and_then(|count| count.checked_mul(MACROBLOCK_SAMPLES as usize))
                .ok_or_else(|| JxrError::arithmetic("encode plane dimensions"))
        };
        Ok((samples(self.macroblocks_x)?, samples(self.macroblocks_y)?))
    }

    /// Tiles in the raster order the tile index lists them.
    #[cfg(test)]
    pub(super) fn tiles(&self) -> Vec<TileRect> {
        let mut tiles = Vec::with_capacity(self.tile_columns.len() * self.tile_rows.len());
        for (row, &height) in self.tile_rows.iter().enumerate() {
            for (column, &width) in self.tile_columns.iter().enumerate() {
                tiles.push(TileRect {
                    x: self.column_boundaries[column] as usize,
                    y: self.row_boundaries[row] as usize,
                    width: usize::from(width),
                    height: usize::from(height),
                });
            }
        }
        tiles
    }

    /// Whether 16-bit dimensions and 8-bit tile sizes can describe the image.
    pub(super) fn short_header(&self, width: u32, height: u32) -> bool {
        width <= 1 << 16
            && height <= 1 << 16
            && self
                .tile_columns
                .iter()
                .chain(&self.tile_rows)
                .all(|&size| size <= 0xff)
    }
}

/// Split `macroblocks` into the fewest near-equal tiles of at most `maximum`.
fn split_axis(macroblocks: u32, maximum: u16) -> Result<Vec<u16>, JxrError> {
    let count = macroblocks.div_ceil(u32::from(maximum));
    // NUM_*_TILES_MINUS1 is twelve bits.
    if count > 1 << 12 {
        return Err(JxrError::new(
            JxrErrorKind::InvalidRequest,
            "encode tile count",
        ));
    }
    let base = macroblocks / count;
    let larger = macroblocks % count;
    (0..count)
        .map(|index| {
            u16::try_from(base + u32::from(index < larger))
                .map_err(|_| JxrError::arithmetic("encode tile size"))
        })
        .collect()
}

/// Every boundary of the tiles `sizes`, none of them empty, that partition
/// `macroblocks`.
fn boundaries(sizes: &[u16], macroblocks: u32) -> Result<Vec<u32>, JxrError> {
    let sizes = sizes.iter().map(|&size| u32::from(size));
    clipped_boundaries(sizes, macroblocks, 0, macroblocks, "encode tile boundaries")
}

/// The lowest profile and level whose limits the codestream meets.
///
/// The conditions are those of T.832 Annex B, in the form the T.835 reference
/// software checks them.
pub(super) fn profile_level(
    image: &CodestreamHeader,
    plane: &ImagePlaneHeader,
    layout: &ImageLayout,
) -> ProfileLevel {
    ProfileLevel {
        profile_idc: profile(image, plane),
        level_idc: level(image, plane, layout),
    }
}

fn profile(image: &CodestreamHeader, plane: &ImagePlaneHeader) -> u8 {
    let depth = output_bit_depth(image.output_bit_depth);
    let color = output_color(image.output_color_format, 0).ok();
    let gray_or_rgb = matches!(color, Some(ColorFormat::Luma | ColorFormat::Rgb));
    let one_or_three = matches!(plane.components, 1 | 3);
    let alpha = image.flags.alpha_plane();
    let wide = matches!(
        depth,
        Some(OutputBitDepth::F16 | OutputBitDepth::I32 { .. } | OutputBitDepth::F32 { .. })
    );
    let baseline = one_or_three && !alpha && gray_or_rgb && !wide;
    let profile = if baseline
        && image.overlap_mode < 2
        && !image.flags.long_word()
        && !matches!(
            depth,
            Some(OutputBitDepth::U16 { .. } | OutputBitDepth::I16 { .. })
        ) {
        Profile::SubBaseline
    } else if baseline {
        Profile::Baseline
    } else if !matches!(color, Some(ColorFormat::Yuv(_) | ColorFormat::CmykDirect)) {
        Profile::Main
    } else {
        Profile::Advanced
    };
    profile.idc()
}

/// `(level, log2 of: image edge, tiles per axis, tile edge, buffer bytes)`.
const LEVEL_LIMITS: [(u8, u32, u32, u32, u32); 6] = [
    (4, 10, 4, 10, 22),
    (8, 11, 5, 11, 24),
    (16, 12, 6, 12, 26),
    (32, 13, 7, 12, 28),
    (64, 14, 8, 12, 30),
    (128, 16, 10, 12, 32),
];

fn level(image: &CodestreamHeader, plane: &ImagePlaneHeader, layout: &ImageLayout) -> u8 {
    let width = u64::from(layout.macroblocks_x) * u64::from(MACROBLOCK_SAMPLES);
    let height = u64::from(layout.macroblocks_y) * u64::from(MACROBLOCK_SAMPLES);
    let channels = u64::from(plane.components) + u64::from(image.flags.alpha_plane());
    let pixels = width * height;
    let buffer = match output_bit_depth(image.output_bit_depth) {
        Some(OutputBitDepth::Bit1White | OutputBitDepth::Bit1Black) => {
            width.div_ceil(8) * height.div_ceil(8) * 8
        }
        Some(OutputBitDepth::I32 { .. } | OutputBitDepth::F32 { .. }) => 4 * channels * pixels,
        Some(OutputBitDepth::Rgb555 | OutputBitDepth::Rgb565) => 2 * pixels,
        // Ten-bit RGB is packed, a word to a pixel.
        Some(OutputBitDepth::U10)
            if output_color(image.output_color_format, 0) == Ok(ColorFormat::Rgb) =>
        {
            4 * pixels
        }
        Some(
            OutputBitDepth::U16 { .. }
            | OutputBitDepth::I16 { .. }
            | OutputBitDepth::F16
            | OutputBitDepth::U10,
        ) => 2 * channels * pixels,
        _ => channels * pixels,
    };
    let largest = |sizes: &[u16]| {
        u64::from(sizes.iter().copied().max().unwrap_or(0)) * u64::from(MACROBLOCK_SAMPLES)
    };
    let tile_width = largest(&layout.tile_columns);
    let tile_height = largest(&layout.tile_rows);
    let tile_columns = layout.tile_columns.len() as u64;
    let tile_rows = layout.tile_rows.len() as u64;
    LEVEL_LIMITS
        .into_iter()
        .find(|&(_, edge, tiles, tile_edge, bytes)| {
            width.max(height) <= 1 << edge
                && tile_columns.max(tile_rows) <= 1 << tiles
                // The reference software requires tiles strictly smaller
                // than the edge limit, unlike the image itself.
                && tile_width.max(tile_height) < 1 << tile_edge
                && buffer < 1 << bytes
        })
        .map_or(255, |(level, ..)| level)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axes_split_into_the_fewest_near_equal_tiles() {
        assert_eq!(split_axis(5, 256).unwrap(), [5]);
        assert_eq!(split_axis(255, 255).unwrap(), [255]);
        assert_eq!(split_axis(256, 255).unwrap(), [128, 128]);
        assert_eq!(split_axis(10, 3).unwrap(), [3, 3, 2, 2]);
        assert!(split_axis(5000, 1).is_err());
        assert_eq!(boundaries(&[3, 3, 2, 2], 10).unwrap(), [0, 3, 6, 8, 10]);
    }

    fn declared_level(width: u32, height: u32, max_tile: u16) -> u8 {
        use crate::{HeaderFlags, QuantizerSet};

        let request = EncodeRequest::lossless().with_max_tile_macroblocks(max_tile);
        let layout = ImageLayout::new(width, height, &request).unwrap();
        let image = CodestreamHeader {
            width,
            height,
            flags: HeaderFlags::from_parsed([false; 9]),
            orientation: 0,
            overlap_mode: 0,
            output_color_format: 0,
            output_bit_depth: 1,
            tile_widths_mb: Vec::new(),
            tile_heights_mb: Vec::new(),
            margins: [0; 4],
        };
        let plane = ImagePlaneHeader {
            internal_color_format: 0,
            scaled: false,
            bands_present: 0,
            components: 1,
            chroma_centering_x: 0,
            chroma_centering_y: 0,
            shift_bits: 0,
            mantissa_length: 0,
            exponent_bias: 0,
            dc_quantizers: Some(QuantizerSet {
                components: vec![1],
            }),
            lp_quantizers: None,
            hp_quantizers: None,
        };
        level(&image, &plane, &layout)
    }

    #[test]
    fn profile_follows_the_output_colour_and_component_count() {
        use crate::{HeaderFlags, QuantizerSet};

        // `(OUTPUT_CLR_FMT, components, profile)` of eight-bit images.
        for (output_color_format, components, expected) in [
            (0, 1, 44),  // gray: Sub-Baseline
            (7, 3, 44),  // RGB: Sub-Baseline
            (4, 4, 66),  // CMYK: Main
            (6, 3, 66),  // three independent components are not RGB: Main
            (6, 8, 66),  // N-component: Main
            (5, 4, 111), // CMYKDIRECT: Advanced
            (3, 3, 111), // YUV 4:4:4 output: Advanced
        ] {
            let image = CodestreamHeader {
                width: 16,
                height: 16,
                flags: HeaderFlags::from_parsed([false; 9]),
                orientation: 0,
                overlap_mode: 0,
                output_color_format,
                output_bit_depth: 1,
                tile_widths_mb: Vec::new(),
                tile_heights_mb: Vec::new(),
                margins: [0; 4],
            };
            let plane = ImagePlaneHeader {
                internal_color_format: 0,
                scaled: false,
                bands_present: 0,
                components,
                chroma_centering_x: 0,
                chroma_centering_y: 0,
                shift_bits: 0,
                mantissa_length: 0,
                exponent_bias: 0,
                dc_quantizers: Some(QuantizerSet {
                    components: vec![1; usize::from(components)],
                }),
                lp_quantizers: None,
                hp_quantizers: None,
            };
            assert_eq!(
                profile(&image, &plane),
                expected,
                "colour {output_color_format} with {components} components"
            );
        }
    }

    /// Headers of an eight-bit image in the given output colour, with or
    /// without an integrated alpha plane.
    fn headers(
        output_color_format: u8,
        components: u16,
        alpha_plane: bool,
    ) -> (CodestreamHeader, ImagePlaneHeader) {
        use crate::{HeaderFlags, QuantizerSet};

        let mut flags = [false; 9];
        flags[8] = alpha_plane;
        let image = CodestreamHeader {
            width: 1024,
            height: 1024,
            flags: HeaderFlags::from_parsed(flags),
            orientation: 0,
            overlap_mode: 0,
            output_color_format,
            output_bit_depth: 1,
            tile_widths_mb: Vec::new(),
            tile_heights_mb: Vec::new(),
            margins: [0; 4],
        };
        let plane = ImagePlaneHeader {
            internal_color_format: 0,
            scaled: false,
            bands_present: 0,
            components,
            chroma_centering_x: 0,
            chroma_centering_y: 0,
            shift_bits: 0,
            mantissa_length: 0,
            exponent_bias: 0,
            dc_quantizers: Some(QuantizerSet {
                components: vec![1; usize::from(components)],
            }),
            lp_quantizers: None,
            hp_quantizers: None,
        };
        (image, plane)
    }

    #[test]
    fn an_integrated_alpha_plane_raises_the_profile_and_counts_towards_the_level() {
        // `(OUTPUT_CLR_FMT, components, profile with alpha)`: alpha takes an
        // image out of the Baseline profiles, and no further.
        for (output_color_format, components, expected) in [
            (0, 1, 66),  // gray
            (7, 3, 66),  // RGB
            (4, 4, 66),  // CMYK
            (6, 5, 66),  // N-component
            (5, 4, 111), // CMYKDIRECT stays Advanced
        ] {
            let (image, plane) = headers(output_color_format, components, true);
            assert_eq!(
                profile(&image, &plane),
                expected,
                "colour {output_color_format} with alpha"
            );
        }
        // Three eight-bit channels of 1024 x 1024 samples fit the level 4
        // sample buffer; the alpha channel is a fourth and does not.
        let request = EncodeRequest::lossless().with_max_tile_macroblocks(32);
        let layout = ImageLayout::new(1024, 1024, &request).unwrap();
        let (image, plane) = headers(7, 3, false);
        assert_eq!(profile(&image, &plane), 44);
        assert_eq!(level(&image, &plane, &layout), 4);
        let (image, plane) = headers(7, 3, true);
        assert_eq!(level(&image, &plane, &layout), 8);
    }

    #[test]
    fn level_limits_include_the_image_edge_but_exclude_the_tile_edge() {
        // A 1024-sample image fits level 4 only when no tile spans all of it.
        assert_eq!(declared_level(1024, 16, 32), 4);
        assert_eq!(declared_level(1024, 16, 64), 8);
        assert_eq!(declared_level(1025, 16, 32), 8);
        // Sixteen tiles per axis is the level 4 limit; seventeen is not.
        assert_eq!(declared_level(1024, 16, 4), 4);
        assert_eq!(declared_level(1024, 16, 3), 8);
        // The default tile size keeps tiles under 4096 samples, so only the
        // image size and sample buffer decide the level of a large image.
        let default = EncodeRequest::DEFAULT_MAX_TILE_MACROBLOCKS;
        assert_eq!(declared_level(8192, 16, default), 32);
        assert_eq!(declared_level(65536, 16, default), 128);
        assert_eq!(declared_level(65537, 16, default), 255);
        // 1024 x 1024 gray fits level 4; four times the samples needs level 8
        // for both its edge and its sample buffer.
        assert_eq!(declared_level(1024, 1024, 32), 4);
        assert_eq!(declared_level(2048, 2048, 32), 8);
    }

    #[test]
    fn the_default_tile_size_splits_large_images_without_raising_their_level() {
        let default = EncodeRequest::DEFAULT_MAX_TILE_MACROBLOCKS;
        // The largest tiles every level from 16 up allows.
        let largest = 255;
        let tiles = |width, height, max_tile| {
            let request = EncodeRequest::lossless().with_max_tile_macroblocks(max_tile);
            ImageLayout::new(width, height, &request)
                .unwrap()
                .tiles()
                .len()
        };
        // An image no larger than a default tile is one tile.
        let edge = u32::from(default) * MACROBLOCK_SAMPLES;
        assert_eq!(tiles(edge, edge, default), 1);
        assert_eq!(tiles(edge + 1, edge, default), 2);
        assert_eq!(tiles(4000, 3000, default), 48);
        assert_eq!(tiles(4000, 3000, largest), 1);
        // More, smaller tiles never need a higher level than the fewest.
        for width in [
            1, 16, 512, 513, 1024, 1025, 2048, 4096, 8192, 16_384, 65_536,
        ] {
            for height in [1, 16, 512, 1024, 4096, 65_536] {
                assert!(
                    declared_level(width, height, default)
                        <= declared_level(width, height, largest),
                    "{width}x{height}"
                );
            }
        }
    }

    #[test]
    fn layout_reports_margins_and_raster_tiles() {
        let request = EncodeRequest::lossless().with_max_tile_macroblocks(2);
        let layout = ImageLayout::new(37, 21, &request).unwrap();
        assert_eq!((layout.macroblocks_x, layout.macroblocks_y), (3, 2));
        assert_eq!(layout.plane_dimensions().unwrap(), (48, 32));
        // The header infers the padding to the planes' edges.
        assert_eq!(crate::header::inferred_margin(37), 48 - 37);
        assert_eq!(crate::header::inferred_margin(21), 32 - 21);
        assert_eq!(
            layout.tiles(),
            [
                TileRect {
                    x: 0,
                    y: 0,
                    width: 2,
                    height: 2
                },
                TileRect {
                    x: 2,
                    y: 0,
                    width: 1,
                    height: 2
                },
            ]
        );
        assert!(layout.short_header(37, 21));
        assert!(!layout.short_header(1 << 17, 21));
    }
}
