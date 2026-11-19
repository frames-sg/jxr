//! The reader of one image plane's macroblocks, in either packet layout.
//!
//! A spatial-mode tile codes every band of a macroblock together, and a
//! frequency-mode tile codes each band of a macroblock row in a packet of
//! its own. No band's syntax reads another band's adaptive state, so both
//! layouts read a band of a macroblock the same way, with the same state:
//! [`PlaneReader`] has one reader for each band, which the spatial reader
//! calls in turn for each macroblock and the frequency reader calls along
//! each band packet's row. Each band resets and adapts its own state as it
//! goes, which in a spatial packet is the same as resetting and adapting
//! every band's state around a whole macroblock.
//!
//! The reader keeps what a macroblock's bands say of it in a slot until the
//! macroblock is stored: one slot reused by every macroblock of a spatial
//! tile, and one slot for each macroblock of the row of a frequency tile.

use jxr_core::{BandPresence, ColorFormat, PredictionMode};

use crate::{
    ImagePlaneHeader,
    entropy::{
        ComponentClass, HpScanDirection, PacketBitReader, TileEntropyState, decode_dc_coefficient,
        decode_hp_block, decode_lp_block, decode_lp_refinement,
    },
};

use super::{
    TileDecodeError,
    cbphp::CbphpState,
    high_pass::{HighpassPayload, coded_block_order, refine_block, refine_blocks, scale_block},
    output::{MacroblockRecord, PlaneWriter},
    quantizer::{QuantizerIndices, TileQuantizers},
    spatial::{
        MacroblockPosition, PredictionDirection, PredictionRecord, adapts_after, checked_add,
        end_history_row, resets_scan_totals, resolve_low_pass,
    },
    syntax::{ComponentSyntax, MAX_COMPONENTS},
    yuv::{CbplpState, decode_dc_pattern, decode_joint_chroma},
};

/// What the bands of one macroblock say of it until it is stored.
#[derive(Debug, Clone, Copy)]
struct MacroblockSlot {
    /// DC prediction mode, which every component shares.
    dc_mode: PredictionMode,
    lp_index: u8,
    hp_index: u8,
    /// High-pass model bits of luma and of chroma, as the high-pass band
    /// was read with them.
    hp_model_bits: [u8; 2],
    /// High-pass prediction direction, which every component shares. A
    /// macroblock without a high-pass band has none.
    hp_direction: PredictionMode,
}

impl MacroblockSlot {
    const EMPTY: Self = Self {
        dc_mode: PredictionMode::None,
        lp_index: 0,
        hp_index: 0,
        hp_model_bits: [0; 2],
        hp_direction: PredictionMode::None,
    };
}

/// High-pass coefficients of a component macroblock that has none.
const NO_HIGH_PASS: [i32; 256] = [0; 256];

/// The reader of one image plane of a tile: the primary plane, or an
/// integrated alpha plane, with its own adaptive state.
pub(super) struct PlaneReader<'a> {
    syntax: ComponentSyntax,
    header: &'a ImagePlaneHeader,
    bands: BandPresence,
    quantizers: TileQuantizers,
    entropy: TileEntropyState,
    /// Joint low-pass pattern state; only YUV planes advance it.
    low_pass_pattern: CbplpState,
    high_pass_pattern: CbphpState,
    /// For each component, the DC and low-pass coefficients of the row
    /// being read and of the row above it.
    history: Vec<Vec<PredictionRecord>>,
    slots: Vec<MacroblockSlot>,
    /// The high-pass coefficients of each component of each slot's
    /// macroblock; empty when the plane has no high-pass band.
    high: Vec<[i32; 256]>,
}

impl<'a> PlaneReader<'a> {
    /// The reader of a plane coded in `colour` by `header`, of which the
    /// tile `width` macroblocks wide codes the bands `bands` with the
    /// quantizers `quantizers`, holding `slots` macroblocks at a time.
    pub(super) fn new(
        header: &'a ImagePlaneHeader,
        colour: ColorFormat,
        bands: BandPresence,
        quantizers: TileQuantizers,
        width: usize,
        slots: usize,
    ) -> Result<Self, TileDecodeError> {
        let syntax = ComponentSyntax::new(colour)?;
        let components = syntax.components();
        let high = if bands.has_high_pass() {
            let count = slots
                .checked_mul(components)
                .ok_or(TileDecodeError::ArithmeticOverflow("high-pass row size"))?;
            vec![[0; 256]; count]
        } else {
            Vec::new()
        };
        Ok(Self {
            syntax,
            header,
            bands,
            quantizers,
            entropy: TileEntropyState::new(),
            low_pass_pattern: CbplpState::new(syntax.sampling()),
            high_pass_pattern: CbphpState::new(width, components),
            history: (0..components)
                .map(|_| Vec::with_capacity(2 * width))
                .collect(),
            slots: vec![MacroblockSlot::EMPTY; slots],
            high,
        })
    }

    pub(super) const fn bands(&self) -> BandPresence {
        self.bands
    }

    pub(super) const fn components(&self) -> usize {
        self.syntax.components()
    }

    /// Reads the plane's part of the header of a frequency-mode low-pass
    /// packet: its low-pass quantizers, when the plane codes the band.
    pub(super) fn read_low_pass_header(
        &mut self,
        reader: &mut PacketBitReader<'_>,
    ) -> Result<(), TileDecodeError> {
        if self.bands.has_low_pass() {
            self.quantizers.parse_low_pass_packet(reader, self.header)?;
        }
        Ok(())
    }

    /// Reads the plane's part of the header of a frequency-mode high-pass
    /// packet: its high-pass quantizers, when the plane codes the band.
    pub(super) fn read_high_pass_header(
        &mut self,
        reader: &mut PacketBitReader<'_>,
    ) -> Result<(), TileDecodeError> {
        if self.bands.has_high_pass() {
            self.quantizers
                .parse_high_pass_packet(reader, self.header)?;
        }
        Ok(())
    }

    /// The entropy and coded-block-pattern state the reader holds.
    #[cfg(test)]
    pub(super) const fn state(&self) -> (&TileEntropyState, &CbphpState) {
        (&self.entropy, &self.high_pass_pattern)
    }

    /// Reads every band of the macroblock at `position` of a spatial packet,
    /// in which each high-pass block's flexbits follow it, and stores it in
    /// `output`, one writer for each component.
    pub(super) fn read_macroblock(
        &mut self,
        reader: &mut PacketBitReader<'_>,
        position: MacroblockPosition,
        trim_flexbits: u8,
        output: &mut [PlaneWriter<'_>],
    ) -> Result<(), TileDecodeError> {
        self.read_low_pass_index(reader, 0)?;
        self.read_high_pass_index(reader, 0)?;
        self.read_dc(reader, position, 0)?;
        if self.bands.has_low_pass() {
            self.read_low_pass(reader, position, 0)?;
        }
        if self.bands.has_high_pass() {
            let payload = HighpassPayload::Combined {
                flexbits_present: self.bands.has_flexbits(),
                trim_flexbits,
            };
            self.read_high_pass(reader, position, 0, payload)?;
        }
        self.store(position, 0, output)
    }

    /// Reads the low-pass quantizer index of the macroblock in `slot`.
    pub(super) fn read_low_pass_index(
        &mut self,
        reader: &mut PacketBitReader<'_>,
        slot: usize,
    ) -> Result<(), TileDecodeError> {
        self.slots[slot].lp_index = self.quantizers.low_pass_index(reader)?;
        Ok(())
    }

    /// Reads the high-pass quantizer index of the macroblock in `slot`,
    /// whose low-pass index has been read.
    pub(super) fn read_high_pass_index(
        &mut self,
        reader: &mut PacketBitReader<'_>,
        slot: usize,
    ) -> Result<(), TileDecodeError> {
        let slot = &mut self.slots[slot];
        slot.hp_index = self.quantizers.high_pass_index(reader, slot.lp_index)?;
        Ok(())
    }

    /// Reads the DC band of the macroblock at `position` of a two-row
    /// history: each component's presence and residual, which prediction
    /// turns into the DC value its history then holds.
    pub(super) fn read_dc(
        &mut self,
        reader: &mut PacketBitReader<'_>,
        position: MacroblockPosition,
        slot: usize,
    ) -> Result<(), TileDecodeError> {
        let syntax = self.syntax;
        let components = syntax.components();
        let pattern = if syntax.joint_patterns() {
            Some(decode_dc_pattern(reader)?)
        } else {
            None
        };
        let mut residuals = [0_i32; MAX_COMPONENTS];
        let mut lap_mean = [0_i32; 2];
        for (component, residual) in residuals.iter_mut().enumerate().take(components) {
            // A joint pattern carries luma in its highest bit.
            let present = match pattern {
                Some(pattern) => pattern & (4 >> component) != 0,
                None => reader.read_bit()?,
            };
            lap_mean[usize::from(component != 0)] += i32::from(present);
            *residual = decode_dc_coefficient(
                reader,
                self.entropy.dc_model.bits(component != 0),
                present,
                syntax.dc_class(component),
                &mut self.entropy.dc_vlc,
            )?;
        }
        self.entropy
            .dc_model
            .update(lap_mean, syntax.colour_model())?;
        let (mode, predictors) = syntax.dc_prediction(&self.history, position)?;
        let operation = match mode {
            PredictionMode::FromLeft => "left DC prediction",
            PredictionMode::FromTop => "top DC prediction",
            _ => "two-neighbour DC prediction",
        };
        for (component, history) in self.history.iter_mut().enumerate() {
            let mut dc_low_pass = [0_i32; 16];
            dc_low_pass[0] = checked_add(residuals[component], predictors[component], operation)?;
            history.push(PredictionRecord {
                dc_low_pass,
                lp_qp_index: 0,
            });
        }
        self.slots[slot].dc_mode = mode;
        if adapts_after(position.x, position.width) {
            self.entropy.dc_vlc.adapt();
        }
        Ok(())
    }

    /// Reads the low-pass band of the macroblock at `position` of a two-row
    /// history, whose DC band and low-pass quantizer index have been read:
    /// which components code a block, then each component's block and
    /// refinements in component order, the chroma components of 4:2:0 and
    /// 4:2:2 in one block. Prediction then completes each component's
    /// coefficients in its history.
    pub(super) fn read_low_pass(
        &mut self,
        reader: &mut PacketBitReader<'_>,
        position: MacroblockPosition,
        slot: usize,
    ) -> Result<(), TileDecodeError> {
        if resets_scan_totals(position.x) {
            self.entropy.lp_scan.reset_totals();
        }
        let syntax = self.syntax;
        let index = position.history_index();
        let MacroblockSlot {
            dc_mode, lp_index, ..
        } = self.slots[slot];
        let prediction = (dc_mode, lp_index);
        // Components coded as blocks of their own: all of them, or luma
        // alone beside jointly coded chroma.
        let joint_chroma = syntax.joint_chroma();
        let separate = if joint_chroma.is_some() {
            1
        } else {
            syntax.components()
        };
        // Bit `n` tells whether component `n` codes a block, but for the
        // joint chroma block, whose bit is bit one.
        let pattern = if syntax.joint_patterns() {
            u32::from(self.low_pass_pattern.decode(reader)?)
        } else {
            let mut pattern = 0;
            for component in 0..separate {
                pattern |= u32::from(reader.read_bit()?) << component;
            }
            pattern
        };
        let mut lap_mean = [0_i32; 2];
        for component in 0..separate {
            let chroma = component != 0;
            let mut low = self.history[component][index].dc_low_pass;
            if pattern & (1 << component) != 0 {
                lap_mean[usize::from(chroma)] += i32::from(decode_lp_block(
                    reader,
                    ComponentClass::of(component),
                    1,
                    &mut self.entropy.lp_vlc,
                    &mut self.entropy.lp_scan,
                    &mut low,
                )?);
            }
            decode_lp_refinement(reader, &mut low, self.entropy.lp_model.bits(chroma))?;
            // Luma beside jointly coded chroma is predicted once the chroma
            // block has been read.
            if joint_chroma.is_none() {
                self.predict_low_pass(component, &mut low, position, prediction)?;
            }
            self.history[component][index] = PredictionRecord {
                dc_low_pass: low,
                lp_qp_index: lp_index,
            };
        }
        if let Some(sampling) = joint_chroma {
            let mut chroma = [1, 2].map(|component| self.history[component][index].dc_low_pass);
            lap_mean[1] = decode_joint_chroma(
                reader,
                &mut self.entropy,
                pattern & 2 != 0,
                &mut chroma,
                sampling,
            )?;
            let luma = self.history[0][index].dc_low_pass;
            for (component, mut low) in [luma, chroma[0], chroma[1]].into_iter().enumerate() {
                self.predict_low_pass(component, &mut low, position, prediction)?;
                self.history[component][index] = PredictionRecord {
                    dc_low_pass: low,
                    lp_qp_index: lp_index,
                };
            }
        }
        self.entropy
            .lp_model
            .update(lap_mean, syntax.colour_model())?;
        if adapts_after(position.x, position.width) {
            self.entropy.lp_vlc.adapt();
        }
        Ok(())
    }

    /// Adds the low-pass prediction of `component` to its residuals `low`.
    fn predict_low_pass(
        &self,
        component: usize,
        low: &mut [i32; 16],
        position: MacroblockPosition,
        prediction: (PredictionMode, u8),
    ) -> Result<(), TileDecodeError> {
        resolve_low_pass(
            low,
            &self.history[component],
            position,
            prediction,
            self.syntax.geometry(component),
            PredictionDirection::Apply,
        )
    }

    /// Reads the high-pass band of the macroblock at `position` of a
    /// two-row history, whose low-pass band and high-pass quantizer index
    /// have been read, into `slot`: every component's coded-block pattern,
    /// then each component's blocks in coded order, each followed by what
    /// `payload` says follows it.
    pub(super) fn read_high_pass(
        &mut self,
        reader: &mut PacketBitReader<'_>,
        position: MacroblockPosition,
        slot: usize,
        payload: HighpassPayload,
    ) -> Result<(), TileDecodeError> {
        let MacroblockPosition { x, y, width } = position;
        if resets_scan_totals(x) {
            self.entropy.hp_scan.reset_totals();
        }
        let syntax = self.syntax;
        let components = syntax.components();
        let index = position.history_index();
        let history = &self.history;
        let direction = syntax.high_pass_mode(|component| &history[component][index].dc_low_pass);
        let mut patterns = [0_u16; MAX_COMPONENTS];
        if let ComponentSyntax::Yuv(sampling) = syntax {
            let yuv = self.high_pass_pattern.decode_yuv(reader, x, y, sampling)?;
            patterns[..3].copy_from_slice(&yuv);
        } else {
            for (component, pattern) in patterns.iter_mut().enumerate().take(components) {
                *pattern = self
                    .high_pass_pattern
                    .decode_component(reader, x, y, component)?;
            }
        }
        let model_bits = [
            self.entropy.hp_model.bits(false),
            self.entropy.hp_model.bits(true),
        ];
        let scan = HpScanDirection::for_mode(direction);
        let start = slot * components;
        let high = self
            .high
            .get_mut(start..start + components)
            .ok_or(TileDecodeError::InvalidPlan("high-pass slot"))?;
        let mut lap_mean = [0_i32; 2];
        for (component, coefficients) in high.iter_mut().enumerate() {
            let class = usize::from(component != 0);
            let mut coded = patterns[component];
            let blocks = coefficients.as_chunks_mut::<16>().0;
            for &block in coded_block_order(syntax.geometry(component)) {
                let block = &mut blocks[block];
                *block = [0; 16];
                if coded & 1 != 0 {
                    lap_mean[class] += i32::from(decode_hp_block(
                        reader,
                        ComponentClass::of(component),
                        &mut self.entropy.hp_vlc,
                        &mut self.entropy.hp_scan,
                        scan,
                        block,
                    )?);
                }
                coded >>= 1;
                match payload {
                    HighpassPayload::VlcOnly => {}
                    HighpassPayload::Combined {
                        flexbits_present: true,
                        trim_flexbits,
                    } => refine_block(reader, block, model_bits[class], trim_flexbits)?,
                    HighpassPayload::Combined { .. } => scale_block(block, model_bits[class])?,
                }
            }
        }
        self.entropy
            .hp_model
            .update(lap_mean, syntax.colour_model())?;
        let slot = &mut self.slots[slot];
        slot.hp_direction = direction;
        slot.hp_model_bits = model_bits;
        if adapts_after(x, width) {
            self.entropy.hp_vlc.adapt();
            self.high_pass_pattern.adapt();
        }
        Ok(())
    }

    /// Completes the high-pass blocks of the macroblock in `slot`, read
    /// without their flexbits, with the flexbits `flexbits` reads, trimmed
    /// of `trim_flexbits` bits.
    pub(super) fn refine_high_pass(
        &mut self,
        slot: usize,
        flexbits: &mut PacketBitReader<'_>,
        trim_flexbits: u8,
    ) -> Result<(), TileDecodeError> {
        for (coefficients, order, model_bits) in self.slot_high_pass(slot)? {
            refine_blocks(flexbits, coefficients, order, model_bits, trim_flexbits)?;
        }
        Ok(())
    }

    /// Completes the high-pass blocks of the macroblock in `slot`, read
    /// without their flexbits, with none.
    pub(super) fn scale_high_pass(&mut self, slot: usize) -> Result<(), TileDecodeError> {
        for (coefficients, order, model_bits) in self.slot_high_pass(slot)? {
            let blocks = coefficients.as_chunks_mut::<16>().0;
            for &block in order {
                scale_block(&mut blocks[block], model_bits)?;
            }
        }
        Ok(())
    }

    /// The high-pass coefficients of each component of the macroblock in
    /// `slot`, with the raster index of each of the component's blocks in
    /// coded order and the model bits its high-pass band was read with.
    fn slot_high_pass(
        &mut self,
        slot: usize,
    ) -> Result<impl Iterator<Item = (&mut [i32; 256], &'static [usize], u8)>, TileDecodeError>
    {
        let syntax = self.syntax;
        let components = syntax.components();
        let model_bits = self.slots[slot].hp_model_bits;
        let start = slot * components;
        let high = self
            .high
            .get_mut(start..start + components)
            .ok_or(TileDecodeError::InvalidPlan("high-pass slot"))?;
        Ok(high
            .iter_mut()
            .enumerate()
            .map(move |(component, coefficients)| {
                (
                    coefficients,
                    coded_block_order(syntax.geometry(component)),
                    model_bits[usize::from(component != 0)],
                )
            }))
    }

    /// Stores the macroblock at `position` of a two-row history, whose
    /// bands are in `slot`, in `output`: one writer for each component.
    pub(super) fn store(
        &self,
        position: MacroblockPosition,
        slot: usize,
        output: &mut [PlaneWriter<'_>],
    ) -> Result<(), TileDecodeError> {
        let components = self.syntax.components();
        if output.len() != components {
            return Err(TileDecodeError::InvalidPlan("decoded component count"));
        }
        let index = position.history_index();
        let state = self.slots[slot];
        let indices = QuantizerIndices {
            lp: state.lp_index,
            hp: state.hp_index,
        };
        let high = self.high.get(slot * components..);
        for (component, writer) in output.iter_mut().enumerate() {
            let high = high
                .and_then(|high| high.get(component))
                .unwrap_or(&NO_HIGH_PASS);
            writer.push(
                &self.history[component][index].dc_low_pass,
                high,
                MacroblockRecord {
                    quantizers: self.quantizers.reconstruction_steps_for(
                        component,
                        indices,
                        self.header.scaled,
                    )?,
                    bands: self.bands,
                    prediction: state.dc_mode,
                    hp_prediction: state.hp_direction,
                },
            )?;
        }
        Ok(())
    }

    /// Makes the row just read, `width` macroblocks wide, the row above the
    /// next.
    pub(super) fn end_row(&mut self, width: usize) {
        for history in &mut self.history {
            end_history_row(history, width);
        }
    }
}
