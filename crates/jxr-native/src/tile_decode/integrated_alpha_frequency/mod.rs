//! Integrated-alpha frequency packet orchestration.

mod dc;
mod high_pass;
mod low_pass;

use jxr_core::BandPresence;

use crate::ImagePlaneHeader;

use super::{
    DecodedTile, TileDecodeError,
    frequency::FrequencyPacketRanges,
    quantizer::{QuantizerIndices, TileQuantizers},
    spatial::SpatialMacroblock,
};

pub(super) fn decode(
    source: &[u8],
    primary: PlaneDescriptor<'_>,
    alpha: PlaneDescriptor<'_>,
    ranges: FrequencyPacketRanges,
    width: usize,
    height: usize,
    trim_present: bool,
) -> Result<DecodedTile, TileDecodeError> {
    validate_band_subset(primary.bands, alpha.bands)?;
    let capacity = width
        .checked_mul(height)
        .ok_or(TileDecodeError::ArithmeticOverflow(
            "integrated alpha frequency macroblock count",
        ))?;
    let mut primary = PlaneState::new(primary, capacity);
    let mut alpha = PlaneState::new(alpha, capacity);
    dc::decode_packet(source, ranges.dc, &mut primary, &mut alpha, width, height)?;
    if primary.bands.has_low_pass() {
        let range = ranges.low_pass.ok_or(TileDecodeError::InvalidPlan(
            "missing integrated alpha LP packet",
        ))?;
        low_pass::decode_packet(source, range, &mut primary, &mut alpha, width, height)?;
    }
    if primary.bands.has_high_pass() {
        let range = ranges.high_pass.ok_or(TileDecodeError::InvalidPlan(
            "missing integrated alpha HP packet",
        ))?;
        high_pass::decode_packet(source, range, &mut primary, &mut alpha, width, height)?;
    }
    let flexbits_escaped = primary.bands.has_flexbits() && ranges.flexbits.is_none();
    if let Some(range) = ranges.flexbits {
        high_pass::decode_flex_packet(source, range, &mut primary, &mut alpha, trim_present)?;
    }
    high_pass::finish_without_flexbits(&mut primary, flexbits_escaped)?;
    high_pass::finish_without_flexbits(&mut alpha, flexbits_escaped)?;
    primary.assign_quantizers()?;
    alpha.assign_quantizers()?;
    let mut components = primary.components;
    let alpha_component = alpha
        .components
        .pop()
        .ok_or(TileDecodeError::InvalidPlan("integrated alpha component"))?;
    components.push(alpha_component);
    Ok(DecodedTile { components })
}

#[derive(Clone, Copy)]
pub(super) struct PlaneDescriptor<'a> {
    pub(super) header: &'a ImagePlaneHeader,
    pub(super) bands: BandPresence,
}

struct PlaneState<'a> {
    header: &'a ImagePlaneHeader,
    bands: BandPresence,
    components: Vec<Vec<SpatialMacroblock>>,
    quantizers: Option<TileQuantizers>,
    low_pass_indices: Vec<u8>,
    high_pass_indices: Vec<u8>,
    high_pass_model_bits: Vec<[u8; 2]>,
}

impl<'a> PlaneState<'a> {
    fn new(descriptor: PlaneDescriptor<'a>, capacity: usize) -> Self {
        Self {
            header: descriptor.header,
            bands: descriptor.bands,
            components: (0..usize::from(descriptor.header.components))
                .map(|_| Vec::with_capacity(capacity))
                .collect(),
            quantizers: None,
            low_pass_indices: Vec::with_capacity(capacity),
            high_pass_indices: Vec::with_capacity(capacity),
            high_pass_model_bits: Vec::with_capacity(capacity),
        }
    }

    fn quantizers(&self) -> Result<&TileQuantizers, TileDecodeError> {
        self.quantizers
            .as_ref()
            .ok_or(TileDecodeError::InvalidPlan("frequency quantizer state"))
    }

    fn quantizers_mut(&mut self) -> Result<&mut TileQuantizers, TileDecodeError> {
        self.quantizers
            .as_mut()
            .ok_or(TileDecodeError::InvalidPlan("frequency quantizer state"))
    }

    fn assign_quantizers(&mut self) -> Result<(), TileDecodeError> {
        let quantizers = self
            .quantizers
            .as_ref()
            .ok_or(TileDecodeError::InvalidPlan("frequency quantizer state"))?;
        let dc_only = self.bands == BandPresence::DcOnly;
        for (component, plane) in self.components.iter_mut().enumerate() {
            for (index, macroblock) in plane.iter_mut().enumerate() {
                let indices = if dc_only {
                    QuantizerIndices { lp: 0, hp: 0 }
                } else {
                    QuantizerIndices {
                        lp: self.low_pass_indices[index],
                        hp: if self.bands.has_high_pass() {
                            self.high_pass_indices[index]
                        } else {
                            0
                        },
                    }
                };
                macroblock.coefficients.quantizers =
                    quantizers.reconstruction_steps_for(component, indices, self.header.scaled)?;
            }
        }
        Ok(())
    }
}

fn validate_band_subset(primary: BandPresence, alpha: BandPresence) -> Result<(), TileDecodeError> {
    if (alpha.has_low_pass() && !primary.has_low_pass())
        || (alpha.has_high_pass() && !primary.has_high_pass())
        || (alpha.has_flexbits() && !primary.has_flexbits())
    {
        Err(TileDecodeError::Unsupported(
            "integrated alpha bands exceed primary bands",
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use jxr_core::{BandPresence, ByteRange};

    use super::{PlaneDescriptor, decode};
    use crate::{ImagePlaneHeader, QuantizerSet, tile_decode::frequency::FrequencyPacketRanges};

    #[test]
    fn dc_packet_interleaves_primary_then_alpha() {
        let packet = [0, 0, 1, 0x5a, 0, 0, 0];
        let range = ByteRange::new(0, packet.len(), packet.len()).unwrap();
        let header = luma_header();
        let decoded = decode(
            &packet,
            PlaneDescriptor {
                header: &header,
                bands: BandPresence::DcOnly,
            },
            PlaneDescriptor {
                header: &header,
                bands: BandPresence::DcOnly,
            },
            FrequencyPacketRanges {
                dc: range,
                low_pass: None,
                high_pass: None,
                flexbits: None,
            },
            1,
            1,
            false,
        )
        .unwrap();
        assert_eq!(decoded.components.len(), 2);
        assert_eq!(decoded.components[0][0].coefficients.dc_low_pass, [0; 16]);
        assert_eq!(decoded.components[1][0].coefficients.dc_low_pass, [0; 16]);
    }

    #[test]
    fn hp_packet_keeps_primary_and_alpha_entropy_independent() {
        let mut source = Vec::new();
        let dc = append(&mut source, &dc_packet());
        let low_pass = append(&mut source, &low_pass_packet());
        let high_pass = append(&mut source, &high_pass_packet());
        let header = luma_header_with_bands(1);
        let decoded = decode(
            &source,
            PlaneDescriptor {
                header: &header,
                bands: BandPresence::NoFlexbits,
            },
            PlaneDescriptor {
                header: &header,
                bands: BandPresence::NoFlexbits,
            },
            FrequencyPacketRanges {
                dc,
                low_pass: Some(low_pass),
                high_pass: Some(high_pass),
                flexbits: None,
            },
            1,
            1,
            false,
        )
        .unwrap();
        for plane in &decoded.components {
            assert_eq!(
                plane[0]
                    .coefficients
                    .high_pass
                    .iter()
                    .filter(|&&value| value != 0)
                    .count(),
                16
            );
        }
    }

    fn luma_header() -> ImagePlaneHeader {
        luma_header_with_bands(3)
    }

    fn luma_header_with_bands(bands_present: u8) -> ImagePlaneHeader {
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
            lp_quantizers: (bands_present < 3).then(|| QuantizerSet {
                components: vec![0],
            }),
            hp_quantizers: (bands_present < 2).then(|| QuantizerSet {
                components: vec![0],
            }),
        }
    }

    fn append(source: &mut Vec<u8>, packet: &[u8]) -> ByteRange {
        let offset = source.len();
        source.extend_from_slice(packet);
        ByteRange::new(offset, packet.len(), source.len()).unwrap()
    }

    fn dc_packet() -> Vec<u8> {
        let mut bits = Bits::packet();
        for _ in 0..2 {
            bits.push(0, 1);
            bits.push(0, 8);
        }
        bits.finish()
    }

    fn low_pass_packet() -> Vec<u8> {
        let mut bits = Bits::packet();
        for _ in 0..2 {
            bits.push(0, 1);
            for _ in 1..16 {
                bits.push(0, 4);
            }
        }
        bits.finish()
    }

    fn high_pass_packet() -> Vec<u8> {
        let mut bits = Bits::packet();
        for _ in 0..2 {
            bits.push(1, 1);
            for _ in 0..16 {
                bits.push(0b00010, 5);
                bits.push(0, 1);
            }
        }
        bits.finish()
    }

    struct Bits {
        bytes: Vec<u8>,
        length: usize,
    }

    impl Bits {
        fn packet() -> Self {
            Self {
                bytes: vec![0, 0, 1, 0x5a],
                length: 32,
            }
        }

        fn push(&mut self, value: u64, count: u8) {
            for shift in (0..count).rev() {
                if self.length.is_multiple_of(8) {
                    self.bytes.push(0);
                }
                let bit = u8::try_from((value >> shift) & 1).unwrap();
                self.bytes[self.length / 8] |= bit << (7 - self.length % 8);
                self.length += 1;
            }
        }

        fn finish(mut self) -> Vec<u8> {
            while !self.length.is_multiple_of(8) {
                self.push(0, 1);
            }
            self.bytes
        }
    }
}
