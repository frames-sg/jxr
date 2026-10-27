use jxr_math::quantization::Quantizer;

#[test]
fn quantizer_rejects_zero_and_dequantize_overflow() {
    let q = Quantizer::new(4).unwrap();
    assert_eq!(q.dequantize(-7).unwrap(), -28);
    assert!(q.dequantize(i32::MAX).is_err());
    assert!(Quantizer::new(0).is_none());
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
