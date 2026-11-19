use jxr_native::entropy::{
    AcVlcState, AdaptiveHpScan, AdaptiveLpScan, CoefficientModel, ColourModel, ComponentClass,
    DcVlcState, FrequencyBand, HpScanDirection, PacketBitReader, TileEntropyState, decode_ac_block,
};

#[test]
fn bounded_reader_and_tile_reset_apis_remain_available() {
    let mut reader = PacketBitReader::with_bit_length(&[0b1010_0000], 4).unwrap();
    assert_eq!(reader.read_bits(4), Ok(0b1010));
    assert_eq!(reader.bits_remaining(), 0);

    let mut dc = DcVlcState::new();
    dc.reset_tile();
    assert_eq!(dc, DcVlcState::new());

    let mut ac = AcVlcState::new();
    ac.reset_tile();
    assert_eq!(ac, AcVlcState::new());

    let mut lp = AdaptiveLpScan::new();
    for _ in 0..3 {
        lp.place(&mut [0; 16], 2, 1).unwrap();
    }
    assert_ne!(lp, AdaptiveLpScan::new());
    lp.reset_tile();
    assert_eq!(lp, AdaptiveLpScan::new());

    let mut hp = AdaptiveHpScan::new();
    for _ in 0..3 {
        hp.place(HpScanDirection::Horizontal, &mut [0; 16], 2, 1)
            .unwrap();
    }
    assert_ne!(hp, AdaptiveHpScan::new());
    hp.reset_tile();
    assert_eq!(hp, AdaptiveHpScan::new());

    let mut model = CoefficientModel::new(FrequencyBand::Lowpass);
    model.update([500, 500], ColourModel::Yuv420).unwrap();
    assert_ne!(model, CoefficientModel::new(FrequencyBand::Lowpass));
    model.reset_tile();
    assert_eq!(model, CoefficientModel::new(FrequencyBand::Lowpass));

    let mut tile = TileEntropyState::new();
    for _ in 0..3 {
        tile.lp_scan.place(&mut [0; 16], 2, 1).unwrap();
    }
    assert_ne!(tile, TileEntropyState::new());
    tile.reset_tile();
    assert_eq!(tile, TileEntropyState::new());
}

#[test]
fn decoded_blocks_retain_the_public_inverse_scan_methods() {
    // FIRST_INDEX value 1 (`00010`) means zero run, magnitude one, last.
    let mut reader = PacketBitReader::with_bit_length(&[0b0001_0100], 6).unwrap();
    let block = decode_ac_block(
        &mut reader,
        FrequencyBand::Lowpass,
        ComponentClass::Luma,
        1,
        &mut AcVlcState::new(),
    )
    .unwrap();

    let mut low = [0; 16];
    block
        .inverse_scan_lp(&mut AdaptiveLpScan::new(), &mut low)
        .unwrap();
    assert_eq!(low[4], -1);

    let mut high = [0; 16];
    block
        .inverse_scan_hp(
            &mut AdaptiveHpScan::new(),
            HpScanDirection::Horizontal,
            &mut high,
        )
        .unwrap();
    assert_eq!(high[4], -1);
}
