//! Tile quantizer headers, inheritance, and macroblock indices.

use jxr_core::QuantizerSet as ReconstructionQuantizers;

use crate::{ImagePlaneHeader, entropy::PacketBitReader};

use super::TileDecodeError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TileQuantizers {
    components: usize,
    dc: Vec<u8>,
    low_pass: Vec<u8>,
    high_pass: Vec<u8>,
    lp_inherits_dc: bool,
    hp_inherits_lp: bool,
}

impl TileQuantizers {
    pub(super) fn parse(
        reader: &mut PacketBitReader<'_>,
        plane: &ImagePlaneHeader,
    ) -> Result<Self, TileDecodeError> {
        let dc = match &plane.dc_quantizers {
            Some(values) => values.components.clone(),
            None => parse_set(reader, plane.components)?,
        };
        let (low_pass, lp_inherits_dc) = parse_low_pass(reader, plane, &dc)?;
        let (high_pass, hp_inherits_lp) = parse_high_pass(reader, plane, &low_pass)?;
        Ok(Self {
            components: usize::from(plane.components),
            dc,
            low_pass,
            high_pass,
            lp_inherits_dc,
            hp_inherits_lp,
        })
    }

    /// Parse only the DC tile header used by frequency-mode codestreams.
    pub(super) fn parse_dc_packet(
        reader: &mut PacketBitReader<'_>,
        plane: &ImagePlaneHeader,
    ) -> Result<Self, TileDecodeError> {
        let dc = match &plane.dc_quantizers {
            Some(values) => values.components.clone(),
            None => parse_set(reader, plane.components)?,
        };
        Ok(Self {
            components: usize::from(plane.components),
            low_pass: dc.clone(),
            high_pass: dc.clone(),
            dc,
            lp_inherits_dc: true,
            hp_inherits_lp: true,
        })
    }

    /// Parse the LP tile header and retain its inheritance/index policy.
    pub(super) fn parse_low_pass_packet(
        &mut self,
        reader: &mut PacketBitReader<'_>,
        plane: &ImagePlaneHeader,
    ) -> Result<(), TileDecodeError> {
        let (low_pass, inherits) = parse_low_pass(reader, plane, &self.dc)?;
        self.low_pass = low_pass;
        self.lp_inherits_dc = inherits;
        Ok(())
    }

    /// Parse the HP tile header after the LP header has established its QP set.
    pub(super) fn parse_high_pass_packet(
        &mut self,
        reader: &mut PacketBitReader<'_>,
        plane: &ImagePlaneHeader,
    ) -> Result<(), TileDecodeError> {
        let (high_pass, inherits) = parse_high_pass(reader, plane, &self.low_pass)?;
        self.high_pass = high_pass;
        self.hp_inherits_lp = inherits;
        Ok(())
    }

    pub(super) fn low_pass_index(
        &self,
        reader: &mut PacketBitReader<'_>,
    ) -> Result<u8, TileDecodeError> {
        let sets = self.low_pass.len() / self.components;
        if self.lp_inherits_dc || sets == 1 {
            Ok(0)
        } else {
            decode_index(reader, sets)
        }
    }

    pub(super) fn high_pass_index(
        &self,
        reader: &mut PacketBitReader<'_>,
        low_pass_index: u8,
    ) -> Result<u8, TileDecodeError> {
        let sets = self.high_pass.len() / self.components;
        if self.hp_inherits_lp {
            Ok(low_pass_index)
        } else if sets == 1 {
            Ok(0)
        } else {
            decode_index(reader, sets)
        }
    }

    pub(super) fn indices(
        &self,
        reader: &mut PacketBitReader<'_>,
    ) -> Result<QuantizerIndices, TileDecodeError> {
        let lp = self.low_pass_index(reader)?;
        let hp = self.high_pass_index(reader, lp)?;
        Ok(QuantizerIndices { lp, hp })
    }

    pub(super) fn reconstruction_steps(
        &self,
        indices: QuantizerIndices,
        scaled: bool,
    ) -> Result<ReconstructionQuantizers, TileDecodeError> {
        self.reconstruction_steps_for(0, indices, scaled)
    }

    pub(super) fn reconstruction_steps_for(
        &self,
        component: usize,
        indices: QuantizerIndices,
        scaled: bool,
    ) -> Result<ReconstructionQuantizers, TileDecodeError> {
        if component >= self.components {
            return Err(TileDecodeError::InvalidPlan("quantizer component index"));
        }
        let dc = table_component(&self.dc, 0, self.components, component)?;
        let lp = table_component(
            &self.low_pass,
            usize::from(indices.lp),
            self.components,
            component,
        )?;
        let hp = table_component(
            &self.high_pass,
            usize::from(indices.hp),
            self.components,
            component,
        )?;
        let low_band_shift = u32::from(component == 0);
        Ok(ReconstructionQuantizers {
            dc: quant_map(dc, scaled, low_band_shift)?,
            low_pass: quant_map(lp, scaled, low_band_shift)?,
            high_pass: quant_map(hp, scaled, 1)?,
        })
    }
}

fn table_component(
    table: &[u8],
    set_index: usize,
    components: usize,
    component: usize,
) -> Result<u8, TileDecodeError> {
    set_index
        .checked_mul(components)
        .and_then(|offset| offset.checked_add(component))
        .and_then(|index| table.get(index).copied())
        .ok_or(TileDecodeError::InvalidPlan(
            "component quantizer table index",
        ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct QuantizerIndices {
    pub(super) lp: u8,
    pub(super) hp: u8,
}

fn parse_low_pass(
    reader: &mut PacketBitReader<'_>,
    plane: &ImagePlaneHeader,
    dc: &[u8],
) -> Result<(Vec<u8>, bool), TileDecodeError> {
    if plane.bands_present == 3 {
        return Ok((dc.to_vec(), true));
    }
    if let Some(values) = &plane.lp_quantizers {
        return Ok((values.components.clone(), false));
    }
    let inherits = reader.read_bit()?;
    if inherits {
        Ok((dc.to_vec(), true))
    } else {
        let count = usize::try_from(reader.read_bits(4)? + 1)
            .map_err(|_| TileDecodeError::ArithmeticOverflow("LP QP count"))?;
        Ok((parse_sets(reader, plane.components, count)?, false))
    }
}

fn parse_high_pass(
    reader: &mut PacketBitReader<'_>,
    plane: &ImagePlaneHeader,
    low_pass: &[u8],
) -> Result<(Vec<u8>, bool), TileDecodeError> {
    if plane.bands_present >= 2 {
        return Ok((low_pass.to_vec(), true));
    }
    if let Some(values) = &plane.hp_quantizers {
        return Ok((values.components.clone(), false));
    }
    let inherits = reader.read_bit()?;
    if inherits {
        Ok((low_pass.to_vec(), true))
    } else {
        let count = usize::try_from(reader.read_bits(4)? + 1)
            .map_err(|_| TileDecodeError::ArithmeticOverflow("HP QP count"))?;
        Ok((parse_sets(reader, plane.components, count)?, false))
    }
}

fn parse_sets(
    reader: &mut PacketBitReader<'_>,
    components: u16,
    count: usize,
) -> Result<Vec<u8>, TileDecodeError> {
    let capacity = count
        .checked_mul(usize::from(components))
        .ok_or(TileDecodeError::ArithmeticOverflow("QP table capacity"))?;
    let mut values = Vec::with_capacity(capacity);
    for _ in 0..count {
        values.extend(parse_set(reader, components)?);
    }
    Ok(values)
}

fn parse_set(
    reader: &mut PacketBitReader<'_>,
    components: u16,
) -> Result<Vec<u8>, TileDecodeError> {
    let mode = if components == 1 {
        0
    } else {
        u8::try_from(reader.read_bits(2)?)
            .map_err(|_| TileDecodeError::ArithmeticOverflow("QP component mode"))?
    };
    match mode {
        0 => {
            let value = read_u8(reader)?;
            Ok(vec![value; usize::from(components)])
        }
        1 => {
            let luma = read_u8(reader)?;
            let chroma = read_u8(reader)?;
            let mut values = vec![chroma; usize::from(components)];
            values[0] = luma;
            Ok(values)
        }
        2 => (0..components).map(|_| read_u8(reader)).collect(),
        _ => Err(TileDecodeError::Unsupported("reserved QP component mode")),
    }
}

fn decode_index(
    reader: &mut PacketBitReader<'_>,
    table_length: usize,
) -> Result<u8, TileDecodeError> {
    let table_length_u8 = u8::try_from(table_length)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("QP table length"))?;
    if table_length == 0 || table_length > 16 {
        return Err(TileDecodeError::InvalidPlan("QP table length"));
    }
    if !reader.read_bit()? {
        return Ok(0);
    }
    let bits = match table_length {
        2..=3 => 1,
        4..=5 => 2,
        6..=9 => 3,
        10..=16 => 4,
        _ => 0,
    };
    let index = u8::try_from(reader.read_bits(bits)? + 1)
        .map_err(|_| TileDecodeError::ArithmeticOverflow("QP index"))?;
    if usize::from(index) >= table_length {
        return Err(TileDecodeError::InvalidQuantizerIndex {
            table_length: table_length_u8,
            index,
        });
    }
    Ok(index)
}

fn quant_map(qp: u8, scaled: bool, scaled_shift: u32) -> Result<u32, TileDecodeError> {
    if qp == 0 {
        return Ok(1);
    }
    let qp = u32::from(qp);
    let (mantissa, exponent) = if scaled {
        if qp < 16 {
            (qp, scaled_shift)
        } else {
            (16 + qp % 16, (qp >> 4) - 1 + scaled_shift)
        }
    } else if qp < 32 {
        ((qp + 3) >> 2, 0)
    } else if qp < 48 {
        ((17 + qp % 16) >> 1, (qp >> 4) - 2)
    } else {
        (16 + qp % 16, (qp >> 4) - 3)
    };
    mantissa
        .checked_shl(exponent)
        .ok_or(TileDecodeError::ArithmeticOverflow("quantizer scaling"))
}

fn read_u8(reader: &mut PacketBitReader<'_>) -> Result<u8, TileDecodeError> {
    u8::try_from(reader.read_bits(8)?).map_err(|_| TileDecodeError::ArithmeticOverflow("QP value"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::QuantizerSet;

    #[test]
    fn qp_index_rejects_reserved_tail_values() {
        let mut reader = PacketBitReader::with_bit_length(&[0b1100_0000], 2).unwrap();
        assert_eq!(
            decode_index(&mut reader, 2).unwrap_err(),
            TileDecodeError::InvalidQuantizerIndex {
                table_length: 2,
                index: 2,
            }
        );
    }

    #[test]
    fn ten_entry_qp_index_consumes_four_payload_bits() {
        let mut reader = PacketBitReader::with_bit_length(&[0b1100_0000], 5).unwrap();
        assert_eq!(decode_index(&mut reader, 10).unwrap(), 9);
        assert_eq!(reader.bits_remaining(), 0);
    }

    #[test]
    fn quant_map_covers_scaled_and_unscaled_boundaries() {
        assert_eq!(quant_map(0, false, 0).unwrap(), 1);
        assert_eq!(quant_map(31, false, 1).unwrap(), 8);
        assert_eq!(quant_map(32, false, 1).unwrap(), 8);
        assert_eq!(quant_map(48, false, 1).unwrap(), 16);
        assert_eq!(quant_map(15, true, 1).unwrap(), 30);
        assert_eq!(quant_map(16, true, 1).unwrap(), 32);
    }

    #[test]
    fn single_component_set_is_exactly_one_byte() {
        let mut reader = PacketBitReader::new(&[37]);
        assert_eq!(
            parse_set(&mut reader, 1).unwrap(),
            QuantizerSet {
                components: vec![37]
            }
            .components
        );
        assert_eq!(reader.bits_remaining(), 0);
    }

    #[test]
    fn scaled_quantizer_shifts_follow_component_and_band_semantics() {
        let quantizers = TileQuantizers {
            components: 2,
            dc: vec![5, 5],
            low_pass: vec![5, 5],
            high_pass: vec![5, 5],
            lp_inherits_dc: false,
            hp_inherits_lp: false,
        };
        let indices = QuantizerIndices { lp: 0, hp: 0 };
        assert_eq!(
            quantizers
                .reconstruction_steps_for(0, indices, true)
                .unwrap(),
            ReconstructionQuantizers {
                dc: 10,
                low_pass: 10,
                high_pass: 10,
            }
        );
        assert_eq!(
            quantizers
                .reconstruction_steps_for(1, indices, true)
                .unwrap(),
            ReconstructionQuantizers {
                dc: 5,
                low_pass: 5,
                high_pass: 10,
            }
        );
    }
}
