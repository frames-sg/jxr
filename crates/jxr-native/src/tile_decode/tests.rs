use jxr_core::{BandPresence, BitstreamMode, ColorFormat, PredictionMode};
use jxr_math::prediction::{
    HighpassPrediction, PREDICTED_FROM_LEFT, PREDICTED_FROM_TOP, predict_high_pass,
    predict_high_pass_420, predict_high_pass_422,
};

use crate::{
    CodestreamDirectory, CodestreamHeader, HeaderFlags, ImagePlaneHeader, ParsedCodestream,
    ParsedHeaders, QuantizerSet,
    bit_writer::BitWriter,
    reconstruct::QuantizedMacroblock,
    tail::PacketBand,
    tile_encode::{PacketOptions, PlaneState, TileRegion, encode_tile_with_state},
};

use super::{
    ComponentGeometry, TileDecodeError, coefficients_per_macroblock, component_geometry,
    frequency::{FrequencyPacketRanges, packet_ranges, trace},
    output::{MacroblockRecord, PlaneWriter},
    spatial_tile::{SpatialRows, TilePlanes},
};
use crate::entropy::encode::test_support::Lcg;

/// Tile size crossing a 16-macroblock adaptation boundary mid-row.
pub(super) const TILE_WIDTH: usize = 19;
pub(super) const TILE_HEIGHT: usize = 3;

/// Macroblocks with coefficients in the bands `bands` codes: sparse,
/// dense, and large-valued rows move the adaptive state through its
/// table, scan, and model-bit choices.
pub(super) fn macroblocks(seed: u64, bands: BandPresence) -> Vec<QuantizedMacroblock> {
    let mut state = seed;
    let mut next = move |range: i32| {
        state = Lcg::advance(state);
        let value = i32::try_from((state >> 40) % 100_003).unwrap();
        value % (2 * range + 1) - range
    };
    (0..TILE_WIDTH * TILE_HEIGHT)
        .map(|index| {
            let row = index / TILE_WIDTH;
            let mut dc_low_pass = [0_i32; 16];
            dc_low_pass[0] = next(20_000);
            if bands.has_low_pass() {
                for value in &mut dc_low_pass[1..] {
                    *value = next(if row == 1 { 900 } else { 12 });
                }
            }
            let mut high_pass = [0_i32; 256];
            if bands.has_high_pass() {
                for (position, value) in high_pass.iter_mut().enumerate() {
                    let coded = match row {
                        0 => next(8) == 0,
                        _ => true,
                    };
                    if position % 16 != 0 && coded {
                        *value = next(if row == 2 { 300 } else { 3 });
                    }
                }
            }
            QuantizedMacroblock {
                dc_low_pass,
                high_pass,
                quantizers: jxr_core::QuantizerSet {
                    dc: 1,
                    low_pass: 1,
                    high_pass: 1,
                },
                bands,
            }
        })
        .collect()
}

/// The header of a plane whose every band carries unit quantizers.
pub(super) fn unit_quantizer_plane(
    internal_color_format: u8,
    components: u16,
    bands: BandPresence,
) -> ImagePlaneHeader {
    let unit = || {
        Some(QuantizerSet {
            components: vec![1; usize::from(components)],
        })
    };
    ImagePlaneHeader {
        internal_color_format,
        scaled: false,
        bands_present: match bands {
            BandPresence::All => 0,
            BandPresence::NoFlexbits => 1,
            BandPresence::NoHighPass => 2,
            BandPresence::DcOnly => 3,
        },
        components,
        chroma_centering_x: 0,
        chroma_centering_y: 0,
        shift_bits: 0,
        mantissa_length: 0,
        exponent_bias: 0,
        dc_quantizers: unit(),
        lp_quantizers: bands.has_low_pass().then(unit).flatten(),
        hp_quantizers: bands.has_high_pass().then(unit).flatten(),
    }
}

/// [`macroblocks`] for a component whose macroblocks have the blocks of
/// `geometry`: entries of blocks the component lacks are zero.
pub(super) fn component_macroblocks(
    seed: u64,
    bands: BandPresence,
    geometry: ComponentGeometry,
) -> Vec<QuantizedMacroblock> {
    let blocks = geometry.block_count();
    let mut macroblocks = macroblocks(seed, bands);
    for macroblock in &mut macroblocks {
        macroblock.dc_low_pass[blocks..].fill(0);
        macroblock.high_pass[blocks * 16..].fill(0);
    }
    macroblocks
}

/// Macroblocks of every component of a plane coded in `colour`, each with
/// the blocks of its own geometry and coefficients unrelated to the other
/// components'.
///
/// Unrelated low-pass values seldom favour one direction four to one,
/// which is what selects a high-pass prediction direction, and the more
/// values a format weighs the more seldom. Every third macroblock therefore
/// has no low-pass energy across it and the one after none down it, in any
/// component, so each direction predicts some macroblock of every format.
pub(super) fn plane_macroblocks(
    colour: ColorFormat,
    bands: BandPresence,
) -> Vec<Vec<QuantizedMacroblock>> {
    let components = usize::from(colour.component_count().unwrap());
    (0_u64..)
        .take(components)
        .zip(0..)
        .map(|(seed, component)| {
            let geometry = component_geometry(colour, component);
            let mut macroblocks = component_macroblocks(seed * 977 + 5, bands, geometry);
            // The low-pass positions the direction is selected from: those
            // weighed as energy across a macroblock, and those down it.
            let (across, down): (&[usize], &[usize]) = match (geometry.columns, geometry.rows) {
                (4, 4) => (&PREDICTED_FROM_TOP, &PREDICTED_FROM_LEFT),
                (2, 4) => (&[1, 5], &[2, 6]),
                _ => (&[1], &[2]),
            };
            for (index, macroblock) in macroblocks.iter_mut().enumerate() {
                let cleared = match index % 3 {
                    1 => across,
                    2 => down,
                    _ => &[],
                };
                for &position in cleared {
                    macroblock.dc_low_pass[position] = 0;
                }
            }
            macroblocks
        })
        .collect()
}

/// The values the arena holds for one macroblock of a component whose
/// macroblocks have the blocks of `geometry`, with its high-pass prediction
/// resolved as reconstruction resolves it.
pub(super) fn stored_component_macroblock(
    stored: &[i32],
    record: MacroblockRecord,
    geometry: ComponentGeometry,
) -> ([i32; 16], [i32; 256]) {
    let mut low = [0_i32; 16];
    let mut high = [0_i32; 256];
    if record.bands.has_high_pass() {
        high[..stored.len()].copy_from_slice(stored);
        for block in 0..geometry.block_count() {
            low[block] = core::mem::take(&mut high[block * 16]);
        }
        resolve_high_pass(&mut high, record.hp_prediction, geometry);
    } else {
        low[..stored.len()].copy_from_slice(stored);
    }
    (low, high)
}

/// Adds the in-macroblock high-pass prediction of `mode` to the residuals
/// of a component whose macroblocks have the blocks of `geometry`, as
/// reconstruction does.
fn resolve_high_pass(high: &mut [i32; 256], mode: PredictionMode, geometry: ComponentGeometry) {
    let direction = match mode {
        PredictionMode::FromLeft => HighpassPrediction::FromLeft,
        PredictionMode::FromTop => HighpassPrediction::FromTop,
        _ => HighpassPrediction::None,
    };
    match (geometry.columns, geometry.rows) {
        (2, 2) => predict_high_pass_420(high, direction),
        (2, 4) => predict_high_pass_422(high, direction),
        _ => predict_high_pass(high, direction),
    }
    .unwrap();
}

/// One component's window of a decoded tile, with the records of its
/// macroblocks.
#[derive(Debug)]
pub(super) struct DecodedWindow {
    pub(super) stored: Vec<i32>,
    pub(super) records: Vec<MacroblockRecord>,
    pub(super) geometry: ComponentGeometry,
}

impl DecodedWindow {
    /// Each macroblock's DC and low-pass coefficients, its high-pass
    /// coefficients with their prediction resolved, and its record.
    pub(super) fn macroblocks(&self) -> Vec<([i32; 16], [i32; 256], MacroblockRecord)> {
        let per_macroblock = self.stored.len() / self.records.len().max(1);
        self.stored
            .chunks_exact(per_macroblock.max(1))
            .zip(&self.records)
            .map(|(stored, &record)| {
                let (low, high) = stored_component_macroblock(stored, record, self.geometry);
                (low, high, record)
            })
            .collect()
    }
}

/// Reads a tile of `macroblocks` macroblocks through `read`, which is given
/// a writer into a window for each component of `planes`, the alpha
/// plane's last, and returns the windows.
pub(super) fn read_windows(
    planes: TilePlanes<'_>,
    macroblocks: usize,
    read: impl FnOnce(&mut [PlaneWriter<'_>]) -> Result<(), TileDecodeError>,
) -> Result<Vec<DecodedWindow>, TileDecodeError> {
    let primary = (0..usize::from(planes.primary.components)).map(|component| {
        (
            component_geometry(planes.primary_colour, component),
            planes.primary_bands,
        )
    });
    let alpha = planes
        .alpha
        .map(|(_, bands)| (component_geometry(ColorFormat::Luma, 0), bands));
    let shapes: Vec<(ComponentGeometry, BandPresence)> = primary.chain(alpha).collect();
    let mut storage: Vec<Vec<i32>> = shapes
        .iter()
        .map(|(geometry, bands)| {
            vec![0; coefficients_per_macroblock(*bands, geometry.block_count()) * macroblocks]
        })
        .collect();
    let mut writers: Vec<PlaneWriter<'_>> = storage
        .iter_mut()
        .zip(&shapes)
        .map(|(storage, (geometry, _))| {
            PlaneWriter::new(storage, geometry.block_count(), macroblocks)
        })
        .collect();
    read(&mut writers)?;
    let records: Vec<Vec<MacroblockRecord>> = writers
        .into_iter()
        .map(PlaneWriter::finish)
        .collect::<Result<_, _>>()?;
    Ok(storage
        .into_iter()
        .zip(records)
        .zip(shapes)
        .map(|((stored, records), (geometry, _))| DecodedWindow {
            stored,
            records,
            geometry,
        })
        .collect())
}

/// Band sets a tile is coded in, with the flexbits trim of each.
pub(super) const BAND_SETS: [(BandPresence, Option<u8>); 5] = [
    (BandPresence::All, None),
    (BandPresence::All, Some(2)),
    (BandPresence::NoFlexbits, None),
    (BandPresence::NoHighPass, None),
    (BandPresence::DcOnly, None),
];

/// One frequency-mode tile as a decoder holds it: the band packets end to
/// end, behind the headers and the tile index that locate them.
pub(super) struct FrequencyTile {
    pub(super) source: Vec<u8>,
    /// The internal colour format of the primary plane.
    pub(super) colour: ColorFormat,
    pub(super) parsed: ParsedCodestream,
    pub(super) ranges: FrequencyPacketRanges,
}

impl FrequencyTile {
    /// A single-tile codestream of `packets`, which the index lists in the
    /// order given, with the planes and flexbits trim its headers declare,
    /// the primary plane coded in `colour`.
    fn new(
        packets: &[Vec<u8>],
        colour: ColorFormat,
        primary: ImagePlaneHeader,
        alpha: Option<ImagePlaneHeader>,
        bands: BandPresence,
        trim_flexbits: bool,
    ) -> Self {
        let mut source = Vec::new();
        let mut tile_offsets = Vec::new();
        for packet in packets {
            tile_offsets.push(source.len() as u64);
            source.extend_from_slice(packet);
        }
        let flags = [
            false,
            true,
            true,
            true,
            true,
            trim_flexbits,
            false,
            false,
            alpha.is_some(),
        ];
        let parsed = ParsedCodestream {
            codestream_range: 0..source.len(),
            headers: ParsedHeaders {
                image: CodestreamHeader {
                    width: u32::try_from(TILE_WIDTH * 16).unwrap(),
                    height: u32::try_from(TILE_HEIGHT * 16).unwrap(),
                    flags: HeaderFlags::from_parsed(flags),
                    orientation: 0,
                    overlap_mode: 0,
                    output_color_format: 0,
                    output_bit_depth: 1,
                    tile_widths_mb: Vec::new(),
                    tile_heights_mb: Vec::new(),
                    margins: [0; 4],
                },
                primary,
                alpha,
                bytes_consumed: 0,
            },
            directory: CodestreamDirectory {
                tile_offsets,
                profiles: Vec::new(),
                tile_data_offset: 0,
            },
            separate_alpha_headers: None,
            separate_alpha_directory: None,
            annex_a: None,
        };
        let ranges = packet_ranges(source.len(), &parsed, bands, 1).unwrap()[0];
        Self {
            source,
            colour,
            parsed,
            ranges,
        }
    }
}

/// Writes one tile of a plane coded in `colour`, with an alpha plane beside
/// it when `with_alpha`, as frequency-mode packets in every band set, reads
/// each back through `decode`, and checks what the reader made of it.
///
/// `decode` is the frequency reader of the format: it returns the
/// macroblocks of every component, the alpha plane's last. The reader must
/// find one packet for each coded band where the index says, consume each to
/// its last bit, end every band in the writer's adaptive state, and recover
/// the coefficients of both planes in their own block geometry.
pub(super) fn check_frequency_tiles(
    colour: ColorFormat,
    internal_color_format: u8,
    with_alpha: bool,
    decode: impl Fn(&FrequencyTile, BandPresence) -> Vec<DecodedWindow>,
) {
    let component_count = colour.component_count().unwrap();
    let tile = TileRegion {
        stride: TILE_WIDTH,
        x: 0,
        y: 0,
        width: TILE_WIDTH,
        height: TILE_HEIGHT,
    };
    for (bands, trim_flexbits) in BAND_SETS {
        let case = format!("{colour:?} alpha {with_alpha} {bands:?} trim {trim_flexbits:?}");
        let components = plane_macroblocks(colour, bands);
        // Alpha coefficients unrelated to the primary plane's.
        let alpha = with_alpha.then(|| macroblocks(4_242, bands));
        let options = PacketOptions {
            colour,
            bands,
            trim_flexbits,
            mode: BitstreamMode::Frequency,
        };
        let written = encode_tile_with_state(&components, alpha.as_deref(), tile, options).unwrap();
        let packets = &written.packets;
        assert_eq!(packets.len(), PacketBand::coded(bands).len(), "{case}");

        let coded = FrequencyTile::new(
            packets,
            colour,
            unit_quantizer_plane(internal_color_format, component_count, bands),
            with_alpha.then(|| unit_quantizer_plane(0, 1, bands)),
            bands,
            trim_flexbits.is_some(),
        );
        // Leave nothing behind from an earlier tile on this thread.
        trace::take();
        let decoded: Vec<_> = decode(&coded, bands)
            .iter()
            .map(DecodedWindow::macroblocks)
            .collect();
        let writers: Vec<&PlaneState> = core::iter::once(&written.primary)
            .chain(&written.alpha)
            .collect();
        assert_packet_ends(bands, &writers, &case);

        let sources: Vec<&Vec<QuantizedMacroblock>> = components.iter().chain(&alpha).collect();
        assert_eq!(decoded.len(), sources.len(), "{case}");
        let mut directions = Vec::new();
        for (plane, (read, source)) in decoded.iter().zip(&sources).enumerate() {
            assert_eq!(read.len(), source.len(), "{case} plane {plane}");
            // Every component of an image plane takes the prediction
            // direction of the plane's macroblock, subsampled chroma
            // included; the alpha plane, the last one, has its own.
            let leader = if with_alpha && plane + 1 == decoded.len() {
                plane
            } else {
                0
            };
            for (index, ((low, high, record), source)) in read.iter().zip(source.iter()).enumerate()
            {
                let at = format!("{case} plane {plane} macroblock {index}");
                assert_eq!(low, &source.dc_low_pass, "{at}");
                let mode = decoded[leader][index].2.hp_prediction;
                assert_eq!(record.hp_prediction, mode, "{at}");
                if !directions.contains(&mode) {
                    directions.push(mode);
                }
                // Trimmed or absent flexbits drop refinement bits.
                if bands == BandPresence::All && trim_flexbits.is_none() {
                    assert_eq!(high, &source.high_pass, "{at}");
                }
            }
        }
        // The tile is only a test of high-pass prediction if macroblocks
        // are predicted in both directions and in neither; a macroblock
        // without a high-pass band records no direction.
        let expected: &[PredictionMode] = if bands.has_high_pass() {
            &[
                PredictionMode::None,
                PredictionMode::FromLeft,
                PredictionMode::FromTop,
            ]
        } else {
            &[PredictionMode::None]
        };
        assert_eq!(directions.len(), expected.len(), "{case} {directions:?}");
        assert!(
            expected.iter().all(|mode| directions.contains(mode)),
            "{case} {directions:?}"
        );
    }
}

/// Checks what the frequency readers of one tile reported to [`trace`]
/// against the state `writers` ended in, the primary plane's first: one
/// packet for each band `bands` codes, in index order, each read to its
/// last bit, and in each the adaptive state of its band as the plane's
/// writer left it.
fn assert_packet_ends(bands: BandPresence, writers: &[&PlaneState], case: &str) {
    let ends = trace::take();
    let read: Vec<PacketBand> = ends.iter().map(|end| end.band).collect();
    assert_eq!(read, PacketBand::coded(bands), "{case}");
    for end in &ends {
        let band = end.band;
        assert_eq!(end.bits_remaining, 0, "{case} {band:?}");
        if band == PacketBand::Flexbits {
            // Flexbits are raw bits: the packet has no adaptive state.
            assert!(end.planes.is_empty(), "{case}");
            continue;
        }
        assert_eq!(end.planes.len(), writers.len(), "{case} {band:?}");
        for (plane, ((read, read_pattern), (written, written_pattern))) in
            end.planes.iter().zip(writers).enumerate()
        {
            let at = format!("{case} {band:?} plane {plane}");
            match band {
                PacketBand::Dc => assert_eq!(
                    (&read.dc_vlc, &read.dc_model),
                    (&written.dc_vlc, &written.dc_model),
                    "{at}"
                ),
                PacketBand::LowPass => assert_eq!(
                    (&read.lp_vlc, &read.lp_scan, &read.lp_model),
                    (&written.lp_vlc, &written.lp_scan, &written.lp_model),
                    "{at}"
                ),
                _ => {
                    assert_eq!(
                        (&read.hp_vlc, &read.hp_scan, &read.hp_model),
                        (&written.hp_vlc, &written.hp_scan, &written.hp_model),
                        "{at}"
                    );
                    assert_eq!(read_pattern.as_ref(), Some(written_pattern), "{at}");
                }
            }
        }
    }
}

fn record(bands: BandPresence) -> MacroblockRecord {
    MacroblockRecord {
        quantizers: jxr_core::QuantizerSet {
            dc: 1,
            low_pass: 1,
            high_pass: 1,
        },
        bands,
        prediction: PredictionMode::None,
        hp_prediction: PredictionMode::None,
    }
}

#[test]
fn plane_writer_fills_exact_window_and_rejects_overflow() {
    let mut storage = [0_i32; 8];
    {
        let mut writer = PlaneWriter::new(&mut storage, 4, 2);
        let high = [0; 256];
        writer
            .push(
                &[3, -2, 7, 11, 99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                &high,
                record(BandPresence::NoHighPass),
            )
            .unwrap();
        writer
            .push(
                &[5, 6, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                &high,
                record(BandPresence::NoHighPass),
            )
            .unwrap();
        let error = writer
            .push(&[0; 16], &high, record(BandPresence::DcOnly))
            .unwrap_err();
        assert_eq!(
            error,
            TileDecodeError::InvalidPlan("external coefficient storage overflow")
        );
        assert_eq!(writer.finish().unwrap().len(), 2);
    }
    assert_eq!(storage, [3, -2, 7, 11, 5, 6, 8, 9]);
}

#[test]
fn plane_writer_rejects_incomplete_windows() {
    let mut storage = [0_i32; 8];
    let mut writer = PlaneWriter::new(&mut storage, 4, 1);
    writer
        .push(&[0; 16], &[0; 256], record(BandPresence::NoHighPass))
        .unwrap();
    assert_eq!(
        writer.finish().unwrap_err(),
        TileDecodeError::InvalidPlan("external coefficient storage fill length")
    );
    let mut writer = PlaneWriter::new(&mut storage, 4, 2);
    writer
        .push(&[0; 16], &[0; 256], record(BandPresence::NoHighPass))
        .unwrap();
    assert_eq!(
        writer.finish().unwrap_err(),
        TileDecodeError::InvalidPlan("decoded tile macroblock count")
    );
}

#[test]
fn plane_writer_interleaves_low_pass_with_each_high_pass_block() {
    let mut low = [0; 16];
    low[..4].copy_from_slice(&[-10, -20, -30, -40]);
    let high: [i32; 256] = core::array::from_fn(|index| i32::try_from(index).unwrap() + 1);
    let mut storage = [0_i32; 64];
    let mut writer = PlaneWriter::new(&mut storage, 4, 1);
    writer.push(&low, &high, record(BandPresence::All)).unwrap();
    writer.finish().unwrap();
    for block in 0..4 {
        assert_eq!(storage[block * 16], low[block]);
        assert_eq!(
            storage[block * 16 + 1..(block + 1) * 16],
            high[block * 16 + 1..(block + 1) * 16]
        );
    }
}

/// Decode a one-macroblock Y-only tile and return its stored coefficients.
fn decode_one(
    packet: &[u8],
    plane: &ImagePlaneHeader,
    bands: BandPresence,
) -> Result<(Vec<i32>, MacroblockRecord), TileDecodeError> {
    let mut coefficients = vec![-1; coefficients_per_macroblock(bands, 16)];
    let mut writer = PlaneWriter::new(&mut coefficients, 16, 1);
    let planes = TilePlanes {
        primary: plane,
        primary_colour: ColorFormat::Luma,
        primary_bands: bands,
        alpha: None,
        trim_present: false,
    };
    let mut rows = SpatialRows::open(packet, planes, 1)?;
    rows.decode_row(0, core::slice::from_mut(&mut writer))?;
    rows.finish()?;
    let records = writer.finish()?;
    Ok((coefficients, records[0]))
}

fn packet_through_low_pass() -> BitWriter {
    let mut bits = BitWriter::new();
    bits.write_bits(1, 24);
    bits.write_bits(0x5a, 8);
    bits.write_bits(0, 1); // IS_DC_CH_FLAG
    bits.write_bits(0, 8); // Initial DC model refinement.
    bits.write_bits(0, 1); // CBPLP_CH_BIT
    for _ in 1..16 {
        bits.write_bits(0, 4); // Initial LP model refinement.
    }
    bits
}

fn plane(bands_present: u8) -> ImagePlaneHeader {
    ImagePlaneHeader {
        internal_color_format: 0,
        scaled: false,
        bands_present,
        components: 1,
        chroma_centering_x: 0,
        chroma_centering_y: 0,
        shift_bits: 0,
        mantissa_length: 0,
        exponent_bias: 0,
        dc_quantizers: Some(QuantizerSet {
            components: vec![0],
        }),
        lp_quantizers: Some(QuantizerSet {
            components: vec![0],
        }),
        hp_quantizers: Some(QuantizerSet {
            components: vec![0],
        }),
    }
}

#[test]
fn rejects_wrong_tile_start_code_before_reading_headers() {
    let error = decode_one(&[0, 0, 2, 0], &plane(3), BandPresence::DcOnly).unwrap_err();
    assert_eq!(error, TileDecodeError::InvalidStartCode { value: 2 });
}

#[test]
fn decodes_one_zero_dc_only_macroblock() {
    // Start code, arbitrary byte, IS_DC_CH_FLAG=0, eight DC refinement bits,
    // then seven zero alignment bits.
    let packet = [0, 0, 1, 0x5a, 0, 0, 0];
    let (coefficients, record) = decode_one(&packet, &plane(3), BandPresence::DcOnly).unwrap();
    assert_eq!(coefficients, [0]);
    assert_eq!(record.quantizers.dc, 1);
}

#[test]
fn decodes_zero_low_pass_refinements_when_high_pass_is_absent() {
    let packet = packet_through_low_pass().into_bytes();
    let (coefficients, _) = decode_one(&packet, &plane(2), BandPresence::NoHighPass).unwrap();
    assert_eq!(coefficients, [0; 16]);
}

fn packet_with_high_pass() -> Vec<u8> {
    let mut packet = packet_through_low_pass();
    packet.write_bits(0b1, 1); // NUM_CBPHP=0; initial prediction expands this to all blocks.
    for _ in 0..16 {
        packet.write_bits(0b00010, 5); // FIRST_INDEX=1 in initial table 1.
        packet.write_bits(0, 1); // Positive sign.
    }
    packet.into_bytes()
}

#[test]
fn decodes_y_only_high_pass_vlc_for_no_flexbits_mode() {
    let (coefficients, _) = decode_one(
        &packet_with_high_pass(),
        &plane(1),
        BandPresence::NoFlexbits,
    )
    .unwrap();
    assert_eq!(coefficients.len(), 256);
    assert_eq!(coefficients.iter().filter(|&&value| value != 0).count(), 16);
}

#[test]
fn all_bands_with_zero_model_bits_consumes_no_flex_payload() {
    let (_, record) = decode_one(&packet_with_high_pass(), &plane(0), BandPresence::All).unwrap();
    assert_eq!(record.bands, BandPresence::All);
}
