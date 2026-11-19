use jxr_math::quantization::Quantizer;

#[test]
fn quantizer_rejects_zero_and_dequantize_overflow() {
    let q = Quantizer::new(4).unwrap();
    assert_eq!(q.dequantize(-7).unwrap(), -28);
    assert!(q.dequantize(i32::MAX).is_err());
    assert!(Quantizer::new(0).is_none());
}

#[test]
fn quantize_rounds_magnitudes_and_inverts_dequantize_at_step_one() {
    let q = Quantizer::new(8).unwrap();
    assert_eq!(q.quantize(11, 4), 1);
    assert_eq!(q.quantize(12, 4), 2);
    assert_eq!(q.quantize(-12, 4), -2);
    assert_eq!(q.quantize(-5, 3), -1);
    assert_eq!(q.quantize(4, 3), 0);
    // An oversized offset cannot push the result past the input magnitude.
    assert_eq!(q.quantize(1, u32::MAX), 1);
    let one = Quantizer::new(1).unwrap();
    for value in [i32::MIN, -7, 0, 7, i32::MAX] {
        assert_eq!(one.quantize(value, u32::MAX), value);
        assert_eq!(one.quantize(value, 0), value);
    }
}

#[test]
fn reconstruction_descriptors_match_the_canonical_word_layout() {
    use core::mem::{offset_of, size_of};
    use jxr_math::tables::{self, reconstruction_abi::*};
    assert_eq!(
        size_of::<JxrMacroblockAbi>(),
        tables::ABI_JXRMACROBLOCKABI_SIZE
    );
    assert_eq!(size_of::<JxrPlaneAbi>(), tables::ABI_JXRPLANEABI_SIZE);
    assert_eq!(size_of::<JxrOutputAbi>(), tables::ABI_JXROUTPUTABI_SIZE);
    assert_eq!(
        offset_of!(JxrOutputAbi, reserved0),
        tables::ABI_JXROUTPUTABI_RESERVED0_OFFSET
    );
}
