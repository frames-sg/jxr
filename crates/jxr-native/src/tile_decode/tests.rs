use jxr_core::{BandPresence, PredictionMode};

use crate::{ImagePlaneHeader, QuantizerSet, reconstruct::QuantizedMacroblock};

use super::{
    CoefficientTarget, SliceTarget, TileDecodeError,
    spatial::{MacroblockPosition, SpatialMacroblock, decode_spatial_packet, predict_low_pass},
};

#[test]
fn direct_coefficient_target_fills_exact_slice_and_rejects_overflow() {
    let mut storage = [0_i32; 4];
    {
        let mut target = SliceTarget::new(&mut storage);
        target.push(3).unwrap();
        target.extend_from_slice(&[-2, 7, 11]).unwrap();
        assert_eq!(target.len(), 4);
        let error = target.push(13).unwrap_err();
        assert_eq!(
            error,
            TileDecodeError::InvalidPlan("external coefficient storage overflow")
        );
    }
    assert_eq!(storage, [3, -2, 7, 11]);
}

#[derive(Default)]
struct Bits {
    bytes: Vec<u8>,
    length: usize,
}

impl Bits {
    fn push(&mut self, value: u64, count: u8) {
        for shift in (0..count).rev() {
            if self.length.is_multiple_of(8) {
                self.bytes.push(0);
            }
            let bit = u8::try_from((value >> shift) & 1).unwrap();
            let byte = self.length / 8;
            self.bytes[byte] |= bit << (7 - self.length % 8);
            self.length += 1;
        }
    }

    fn align_zero(&mut self) {
        while !self.length.is_multiple_of(8) {
            self.push(0, 1);
        }
    }
}

fn packet_through_low_pass() -> Bits {
    let mut bits = Bits::default();
    bits.push(1, 24);
    bits.push(0x5a, 8);
    bits.push(0, 1); // IS_DC_CH_FLAG
    bits.push(0, 8); // Initial DC model refinement.
    bits.push(0, 1); // CBPLP_CH_BIT
    for _ in 1..16 {
        bits.push(0, 4); // Initial LP model refinement.
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
    let error = decode_spatial_packet(&[0, 0, 2, 0], &plane(3), BandPresence::DcOnly, 1, 1, false)
        .unwrap_err();
    assert_eq!(error, TileDecodeError::InvalidStartCode { value: 2 });
}

#[test]
fn decodes_one_zero_dc_only_macroblock() {
    // Start code, arbitrary byte, IS_DC_CH_FLAG=0, eight DC refinement bits,
    // then seven zero alignment bits.
    let packet = [0, 0, 1, 0x5a, 0, 0, 0];
    let decoded =
        decode_spatial_packet(&packet, &plane(3), BandPresence::DcOnly, 1, 1, false).unwrap();
    assert_eq!(decoded.len(), 1);
    assert_eq!(decoded[0].coefficients.dc_low_pass, [0; 16]);
    assert_eq!(decoded[0].coefficients.high_pass, [0; 256]);
    assert_eq!(decoded[0].coefficients.quantizers.dc, 1);
}

#[test]
fn decodes_zero_low_pass_refinements_when_high_pass_is_absent() {
    let mut packet = packet_through_low_pass();
    packet.align_zero();
    let decoded = decode_spatial_packet(
        &packet.bytes,
        &plane(2),
        BandPresence::NoHighPass,
        1,
        1,
        false,
    )
    .unwrap();
    assert_eq!(decoded[0].coefficients.dc_low_pass, [0; 16]);
    assert_eq!(decoded[0].coefficients.high_pass, [0; 256]);
}

fn packet_with_high_pass() -> Vec<u8> {
    let mut packet = packet_through_low_pass();
    packet.push(0b1, 1); // NUM_CBPHP=0; initial prediction expands this to all blocks.
    for _ in 0..16 {
        packet.push(0b00010, 5); // FIRST_INDEX=1 in initial table 1.
        packet.push(0, 1); // Positive sign.
    }
    packet.align_zero();
    packet.bytes
}

#[test]
fn decodes_y_only_high_pass_vlc_for_no_flexbits_mode() {
    let decoded = decode_spatial_packet(
        &packet_with_high_pass(),
        &plane(1),
        BandPresence::NoFlexbits,
        1,
        1,
        false,
    )
    .unwrap();
    assert_eq!(
        decoded[0]
            .coefficients
            .high_pass
            .iter()
            .filter(|&&value| value != 0)
            .count(),
        16
    );
}

#[test]
fn all_bands_with_zero_model_bits_consumes_no_flex_payload() {
    let decoded = decode_spatial_packet(
        &packet_with_high_pass(),
        &plane(0),
        BandPresence::All,
        1,
        1,
        false,
    )
    .unwrap();
    assert_eq!(decoded[0].coefficients.bands, BandPresence::All);
}

#[test]
fn left_low_pass_prediction_uses_the_left_macroblock_column() {
    let mut left_low = [0; 16];
    left_low[4] = 4;
    left_low[5] = 5;
    left_low[6] = 6;
    left_low[8] = 8;
    left_low[12] = 12;
    let decoded = [SpatialMacroblock {
        coefficients: QuantizedMacroblock {
            dc_low_pass: left_low,
            high_pass: [0; 256],
            quantizers: jxr_core::QuantizerSet {
                dc: 1,
                low_pass: 1,
                high_pass: 1,
            },
            bands: BandPresence::All,
        },
        prediction: PredictionMode::None,
        hp_prediction: PredictionMode::None,
        lp_qp_index: 0,
    }];
    let mut current = [0; 16];
    predict_low_pass(
        &mut current,
        &decoded,
        MacroblockPosition {
            width: 2,
            x: 1,
            y: 0,
        },
        PredictionMode::FromLeft,
        0,
    )
    .unwrap();
    assert_eq!((current[4], current[8], current[12]), (4, 8, 12));
}
