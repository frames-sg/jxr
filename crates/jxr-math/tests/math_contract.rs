use jxr_math::quantization::Quantizer;

#[test]
fn quantizer_rejects_zero_and_dequantize_overflow() {
    let q = Quantizer::new(4).unwrap();
    assert_eq!(q.dequantize(-7).unwrap(), -28);
    assert!(q.dequantize(i32::MAX).is_err());
    assert!(Quantizer::new(0).is_none());
}

#[test]
fn cuda_reconstruction_abi_is_generated_from_the_canonical_schema() {
    let source = jxr_math::tables::CUDA_RECONSTRUCTION_CONSTANTS;
    assert!(source.contains("JXR_INVERSE_PERMUTATION[16]"));
    assert!(source.contains("struct JxrMacroblockAbi"));
    assert!(source.contains("struct JxrOutputAbi"));
}
