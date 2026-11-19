//! Tiles of every internal colour format, in either packet layout.
//!
//! One macroblock loop serves Y-only, YUV, YUVK, and N-component planes.
//! The formats differ only in the decisions the decoder's
//! [`ComponentSyntax`] makes, which this writer and the decoder's plane
//! reader share: how per-component presence is signalled in each band,
//! which components decide the prediction modes, how the coefficient models
//! weigh the components, and how many blocks a chroma component has. YUV
//! 4:2:0 and 4:2:2 code their two chroma components' low-pass coefficients
//! as one block and their high-pass blocks in the component's own block
//! geometry.
//!
//! An integrated alpha plane is a second Y-only plane through the same loop:
//! a tile holds one [`Plane`] of adaptive state per image plane and writes
//! the alpha macroblock after the primary one at every position, as
//! [`crate::tile_decode`]'s integrated-alpha readers consume them.
//!
//! The loop is also the same for both packet layouts. It forms every
//! element of a macroblock once and hands it to [`TilePackets`] under the
//! band it belongs to; that alone decides whether the bands of a tile end up
//! interleaved in one packet or apart in a packet each. No band's syntax
//! reads another band's adaptive state, so the state of a [`Plane`] moves
//! through the same values either way, and a frequency-mode reader, which
//! restarts adaptation in every band packet, arrives at the same ones.

use jxr_core::{ColorFormat, PredictionMode};

use crate::{
    bit_writer::BitWriter,
    entropy::{
        ComponentClass, TileEntropyState,
        encode::{dc_has_abs_level, encode_dc_coefficient},
    },
    reconstruct::QuantizedMacroblock,
    tail::PacketBand,
    tile_decode::{
        cbphp::CbphpState,
        spatial::{
            MacroblockPosition, PredictionRecord, adapt_at_boundary, end_history_row,
            resets_scan_totals,
        },
        syntax::ComponentSyntax,
        yuv::{CbplpState, encode_dc_pattern},
    },
};

use super::{
    MAX_COMPONENTS, PacketOptions, TileEncodeError, TileRegion,
    high_pass::{HighPass, HighPassSyntax},
    low_pass::{JointChromaLowPass, LowPass},
    packets::TilePackets,
};

/// Storage a tile writer fills while it writes one tile, kept for the next.
///
/// A worker that codes tile after tile hands the same scratch to each, so
/// in its steady state a tile allocates nothing: the packets are written in
/// buffers kept here, which stop growing once they have held the worker's
/// largest packet, and the prediction history and residuals of each plane
/// reuse the storage of the tile before. What is kept is bounded by the
/// largest tile the worker has coded and by the longest packet a buffer is
/// kept for, never by the image.
#[derive(Debug, Default)]
pub(crate) struct TileScratch {
    /// The buffers the packets of the last tile were written in.
    buffers: Vec<Vec<u8>>,
    primary: PlaneScratch,
    alpha: PlaneScratch,
}

/// The per-tile storage of one [`Plane`].
#[derive(Debug, Default)]
struct PlaneScratch {
    coded: Vec<Vec<PredictionRecord>>,
    low_pass: Vec<LowPass>,
    high_pass: Vec<HighPass>,
}

/// Writes the packets of one tile of a plane coded in `options.colour`: one
/// in spatial mode, and in frequency mode one for each band the image
/// codes, in the order the tile index lists them.
///
/// `components` holds every macroblock of each component in image raster
/// order; `tile` selects the ones this tile codes. A chroma component of
/// YUV 4:2:0 or 4:2:2 holds the fewer blocks of its macroblock shape, in
/// raster block order. `alpha` is the integrated alpha plane of an image
/// that has one: a Y-only plane with its own adaptive state, whose
/// macroblock follows the primary plane's at every position. `scratch` is
/// the working storage, in any state an earlier tile left it in.
///
/// This is [`TileWriter`] given the whole tile at once.
#[cfg(test)]
pub(crate) fn encode_tile(
    components: &[Vec<QuantizedMacroblock>],
    alpha: Option<&[QuantizedMacroblock]>,
    tile: TileRegion,
    options: PacketOptions,
    scratch: &mut TileScratch,
) -> Result<Vec<Vec<u8>>, TileEncodeError> {
    let writer = TileWriter::whole(components, alpha, tile, options, core::mem::take(scratch))?;
    let (packets, storage) = writer.into_parts();
    let mut storage = storage.into_scratch();
    let packets = packets.finish(&mut storage.buffers);
    *scratch = storage;
    Ok(packets)
}

/// The entropy and coded-block-pattern state a plane's writer ended in: the
/// state a decoder of the plane must hold once it has read the tile.
#[cfg(test)]
pub(crate) type PlaneState = (TileEntropyState, CbphpState);

/// The packets [`encode_tile`] writes, with the state each plane's writer
/// ended in.
#[cfg(test)]
pub(crate) struct EncodedTile {
    pub(crate) packets: Vec<Vec<u8>>,
    pub(crate) primary: PlaneState,
    /// The alpha plane's state, when the tile has one.
    pub(crate) alpha: Option<PlaneState>,
}

/// [`encode_tile`], keeping the final state of every plane.
#[cfg(test)]
pub(crate) fn encode_tile_with_state(
    components: &[Vec<QuantizedMacroblock>],
    alpha: Option<&[QuantizedMacroblock]>,
    tile: TileRegion,
    options: PacketOptions,
) -> Result<EncodedTile, TileEncodeError> {
    let writer = TileWriter::whole(components, alpha, tile, options, TileScratch::default())?;
    let (packets, mut storage) = writer.into_parts();
    let state = |plane: Plane| (plane.entropy, plane.high_pass_pattern);
    Ok(EncodedTile {
        packets: packets.finish(&mut storage.buffers),
        primary: state(storage.primary),
        alpha: storage.alpha.map(state),
    })
}

/// How many macroblocks a tile spans, and which planes it codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TileShape {
    /// Tile size in macroblocks.
    pub(crate) width: usize,
    pub(crate) height: usize,
    /// Components of the primary plane.
    pub(crate) components: usize,
    /// Whether an integrated alpha plane follows the primary plane.
    pub(crate) alpha: bool,
}

/// One tile being written, a band of macroblock rows at a time.
///
/// [`Self::begin`] opens the packets and the adaptive state of every plane;
/// [`Self::push_rows`] codes the tile's next macroblock rows from the band
/// of coefficients that holds them, which the caller is free to drop
/// afterwards; [`Self::finish`] closes the packets once every row has been
/// coded. Pushing a tile's rows in any number of bands writes the same
/// packets as pushing them at once: nothing in the syntax looks ahead of the
/// macroblock being coded.
///
/// Between bands a writer keeps only what the syntax carries from one
/// macroblock row to the next: each plane's adaptive entropy and
/// coded-block-pattern state, the DC and low-pass values of the macroblocks
/// it has coded, which DC and low-pass prediction read, and the packets
/// written so far.
pub(crate) struct TileWriter {
    packets: TilePackets,
    primary: Plane,
    alpha: Option<Plane>,
    shape: TileShape,
    /// Tile-local row of the next macroblock row to code.
    next_row: usize,
    /// Packet buffers of the scratch the tile began with that its packets
    /// did not take.
    buffers: Vec<Vec<u8>>,
    /// Alpha-plane storage of the scratch, kept for a later tile when this
    /// one has no alpha plane.
    spare_alpha: PlaneScratch,
}

impl TileWriter {
    /// Starts a tile of `shape` whose packets carry the syntax `options`
    /// selects. The storage of `scratch` moves into the tile and comes back
    /// from [`Self::finish`].
    ///
    /// `output`, when given, is the output the tile's packets come next in:
    /// the tile's first packet is written onto its end, and
    /// [`Self::finish`] gives it back.
    pub(crate) fn begin(
        shape: TileShape,
        options: PacketOptions,
        scratch: TileScratch,
        output: Option<Vec<u8>>,
    ) -> Result<Self, TileEncodeError> {
        let TileScratch {
            mut buffers,
            primary,
            alpha,
        } = scratch;
        let packets = TilePackets::start(options, &mut buffers, output)?;
        let primary = Plane::new(shape.components, shape, options, primary)?;
        let (alpha, spare_alpha) = if shape.alpha {
            let options = PacketOptions {
                colour: ColorFormat::Luma,
                ..options
            };
            let plane = Plane::new(1, shape, options, alpha)?;
            (Some(plane), PlaneScratch::default())
        } else {
            (None, alpha)
        };
        Ok(Self {
            packets,
            primary,
            alpha,
            shape,
            next_row: 0,
            buffers,
            spare_alpha,
        })
    }

    /// A tile written from the macroblocks of `components` and `alpha` that
    /// `tile` selects, all of its rows at once.
    #[cfg(test)]
    fn whole(
        components: &[Vec<QuantizedMacroblock>],
        alpha: Option<&[QuantizedMacroblock]>,
        tile: TileRegion,
        options: PacketOptions,
        scratch: TileScratch,
    ) -> Result<Self, TileEncodeError> {
        let shape = TileShape {
            width: tile.width,
            height: tile.height,
            components: components.len(),
            alpha: alpha.is_some(),
        };
        let mut writer = Self::begin(shape, options, scratch, None)?;
        writer.push_rows(components, alpha, tile)?;
        Ok(writer)
    }

    /// Codes the tile's next `rows.height` macroblock rows, the alpha
    /// plane's macroblock after the primary plane's at each position.
    ///
    /// `components` holds a band of macroblock rows of each component in
    /// raster order and `rows` selects the tile's part of it: `rows.width`
    /// must be the tile's width, and the rows must not run past the tile's
    /// last. `alpha` holds the same band of the alpha plane when the tile
    /// has one.
    pub(crate) fn push_rows(
        &mut self,
        components: &[Vec<QuantizedMacroblock>],
        alpha: Option<&[QuantizedMacroblock]>,
        rows: TileRegion,
    ) -> Result<(), TileEncodeError> {
        let remaining = self.shape.height - self.next_row;
        if rows.width != self.shape.width || rows.height > remaining {
            return Err(TileEncodeError::InvalidInput("tile rows"));
        }
        if alpha.is_some() != self.alpha.is_some() || components.len() != self.shape.components {
            return Err(TileEncodeError::InvalidInput("component count"));
        }
        let Some(first) = components.first() else {
            return Ok(());
        };
        if rows.height == 0 {
            return Ok(());
        }
        // The co-located macroblock of every component; a plane has at
        // least one, and entries beyond its count are never read.
        let mut macroblocks = [rows.macroblock(first, 0, 0)?; MAX_COMPONENTS];
        let macroblocks = &mut macroblocks[..components.len()];
        for y in 0..rows.height {
            for x in 0..rows.width {
                for (macroblock, component) in macroblocks.iter_mut().zip(components) {
                    *macroblock = rows.macroblock(component, x, y)?;
                }
                let position = MacroblockPosition {
                    width: self.shape.width,
                    x,
                    y: self.next_row + y,
                };
                self.primary
                    .encode_macroblock(&mut self.packets, position, macroblocks)?;
                if let (Some(plane), Some(alpha)) = (&mut self.alpha, alpha) {
                    let macroblock = rows.macroblock(alpha, x, y)?;
                    plane.encode_macroblock(&mut self.packets, position, &[macroblock])?;
                }
            }
        }
        self.next_row += rows.height;
        Ok(())
    }

    /// Make room in the output the tile writes its first packet onto, when
    /// it holds one, for `additional` more bytes, of `expected` that the
    /// output is expected to hold in all; see
    /// [`reserve_for`](crate::bit_writer::reserve_for).
    pub(crate) fn reserve_output(&mut self, additional: usize, expected: usize) {
        self.packets.reserve_output(additional, expected);
    }

    /// Bytes the tile's packets hold once finished, were they finished now,
    /// not counting the output its first packet may be written onto.
    pub(crate) fn written_bytes(&self) -> usize {
        self.packets.written_bytes()
    }

    /// Whether the tile was given the output to write its first packet
    /// onto, which [`Self::finish`] gives back.
    pub(crate) const fn holds_output(&self) -> bool {
        self.packets.holds_output()
    }

    /// Closes the packets of a tile whose every row has been coded and
    /// appends them to `output` in the order the tile index lists them, the
    /// length of each to `lengths`. A tile that holds the output puts it
    /// back in `output`, which must be empty. Returns the storage the tile
    /// used, for the next tile.
    pub(crate) fn finish(
        self,
        output: &mut Vec<u8>,
        lengths: &mut Vec<usize>,
    ) -> Result<TileScratch, TileEncodeError> {
        if self.next_row != self.shape.height {
            return Err(TileEncodeError::InvalidInput("tile rows"));
        }
        let (packets, storage) = self.into_parts();
        let mut storage = storage.into_scratch();
        packets.finish_into(output, lengths, &mut storage.buffers)?;
        Ok(storage)
    }

    /// The tile's packets apart from the planes and buffers it holds.
    fn into_parts(self) -> (TilePackets, TileStorage) {
        (
            self.packets,
            TileStorage {
                buffers: self.buffers,
                primary: self.primary,
                alpha: self.alpha,
                spare_alpha: self.spare_alpha,
            },
        )
    }
}

/// What a tile holds besides its packets.
struct TileStorage {
    buffers: Vec<Vec<u8>>,
    primary: Plane,
    alpha: Option<Plane>,
    spare_alpha: PlaneScratch,
}

impl TileStorage {
    /// The storage, for the next tile.
    fn into_scratch(self) -> TileScratch {
        TileScratch {
            buffers: self.buffers,
            primary: self.primary.into_scratch(),
            alpha: self.alpha.map_or(self.spare_alpha, Plane::into_scratch),
        }
    }
}

/// One image plane of a tile, with the adaptive state a decoder of the
/// plane rebuilds as it reads. The primary and alpha planes of a tile each
/// hold their own and share nothing but the packets they are written to.
struct Plane {
    entropy: TileEntropyState,
    /// Joint low-pass pattern state; only YUV formats advance it.
    low_pass_pattern: CbplpState,
    high_pass_pattern: CbphpState,
    /// DC and low-pass values of the macroblocks coded so far, per component.
    coded: Vec<Vec<PredictionRecord>>,
    /// Per-component residuals of the current macroblock, kept to reuse
    /// their storage. Jointly coded chroma components have no entry in
    /// `low_pass`; `high_pass` always holds one for every component.
    low_pass: Vec<LowPass>,
    high_pass: Vec<HighPass>,
    syntax: ComponentSyntax,
    options: PacketOptions,
}

impl Plane {
    /// The state a plane of `components` components coded in
    /// `options.colour` starts a tile with.
    fn new(
        components: usize,
        tile: TileShape,
        options: PacketOptions,
        scratch: PlaneScratch,
    ) -> Result<Self, TileEncodeError> {
        let syntax = ComponentSyntax::new(options.colour)?;
        if components != syntax.components() {
            return Err(TileEncodeError::InvalidInput("component count"));
        }
        let count = tile
            .width
            .checked_mul(tile.height)
            .filter(|&count| count != 0)
            .ok_or(TileEncodeError::InvalidInput("tile macroblock count"))?;
        let PlaneScratch {
            mut coded,
            mut low_pass,
            mut high_pass,
        } = scratch;
        coded.resize_with(components, Vec::new);
        for history in &mut coded {
            history.clear();
            // Prediction reads the row being coded and the one above it, so
            // the history is two rows of the widest tile a worker has coded
            // and not the tile, nor the doubling of one.
            history.reserve_exact(tile.width.saturating_mul(2).min(count));
        }
        low_pass.clear();
        low_pass.reserve_exact(components);
        high_pass.resize(components, HighPass::empty());
        Ok(Self {
            entropy: TileEntropyState::new(),
            low_pass_pattern: CbplpState::new(syntax.sampling()),
            high_pass_pattern: CbphpState::new(tile.width, components),
            coded,
            low_pass,
            high_pass,
            syntax,
            options,
        })
    }

    /// The plane's storage, for the next tile.
    fn into_scratch(self) -> PlaneScratch {
        PlaneScratch {
            coded: self.coded,
            low_pass: self.low_pass,
            high_pass: self.high_pass,
        }
    }

    /// Writes every band of the co-located macroblocks of all components,
    /// each to the packet that carries it.
    fn encode_macroblock(
        &mut self,
        packets: &mut TilePackets,
        position: MacroblockPosition,
        macroblocks: &[&QuantizedMacroblock],
    ) -> Result<(), TileEncodeError> {
        if resets_scan_totals(position.x) {
            self.entropy.reset_scan_totals();
        }
        // The history holds two macroblock rows, as the readers' does.
        let in_history = position.in_two_rows();
        let dc_mode = self.encode_dc(packets.band(PacketBand::Dc)?, in_history, macroblocks)?;
        if self.options.bands.has_low_pass() {
            let writer = packets.band(PacketBand::LowPass)?;
            self.encode_low_pass(writer, in_history, dc_mode, macroblocks)?;
        }
        if self.options.bands.has_high_pass() {
            self.encode_high_pass(packets, position, macroblocks)?;
        }
        for (history, macroblock) in self.coded.iter_mut().zip(macroblocks) {
            history.push(PredictionRecord {
                dc_low_pass: macroblock.dc_low_pass,
                lp_qp_index: 0,
            });
            if position.x + 1 == position.width {
                end_history_row(history, position.width);
            }
        }
        adapt_at_boundary(
            &mut self.entropy,
            &mut self.high_pass_pattern,
            position.x,
            position.width,
        );
        Ok(())
    }

    /// Writes each component's DC residual with its presence, and returns
    /// the prediction mode the components share.
    fn encode_dc(
        &mut self,
        writer: &mut BitWriter,
        position: MacroblockPosition,
        macroblocks: &[&QuantizedMacroblock],
    ) -> Result<PredictionMode, TileEncodeError> {
        let (mode, predictors) = self.syntax.dc_prediction(&self.coded, position)?;
        let mut residuals = [0_i32; MAX_COMPONENTS];
        let mut present = [false; MAX_COMPONENTS];
        let mut lap_mean = [0_i32; 2];
        for (component, macroblock) in macroblocks.iter().enumerate() {
            let chroma = component != 0;
            residuals[component] = macroblock.dc_low_pass[0]
                .checked_sub(predictors[component])
                .ok_or(TileEncodeError::ArithmeticOverflow(
                    "DC prediction residual",
                ))?;
            present[component] =
                dc_has_abs_level(residuals[component], self.entropy.dc_model.bits(chroma));
            lap_mean[usize::from(chroma)] += i32::from(present[component]);
        }
        let joint = self.syntax.joint_patterns();
        if joint {
            // VAL_DC_YUV carries luma in its highest bit.
            let pattern = (0..3).fold(0_u8, |pattern, component| {
                pattern | u8::from(present[component]) << (2 - component)
            });
            encode_dc_pattern(writer, pattern)?;
        }
        for component in 0..macroblocks.len() {
            if !joint {
                writer.write_flag(present[component]);
            }
            encode_dc_coefficient(
                writer,
                residuals[component],
                self.entropy.dc_model.bits(component != 0),
                self.syntax.dc_class(component),
                &mut self.entropy.dc_vlc,
            )?;
        }
        self.entropy
            .dc_model
            .update(lap_mean, self.syntax.colour_model())?;
        Ok(mode)
    }

    /// Writes which components code a low-pass block, then each component's
    /// block and refinements in component order.
    fn encode_low_pass(
        &mut self,
        writer: &mut BitWriter,
        position: MacroblockPosition,
        dc_mode: PredictionMode,
        macroblocks: &[&QuantizedMacroblock],
    ) -> Result<(), TileEncodeError> {
        // Components coded as blocks of their own: all of them, or luma
        // alone beside jointly coded chroma.
        let joint_chroma = self.syntax.joint_chroma();
        let separate = if joint_chroma.is_some() {
            1
        } else {
            macroblocks.len()
        };
        self.low_pass.clear();
        for (component, macroblock) in macroblocks.iter().enumerate().take(separate) {
            self.low_pass.push(LowPass::new(
                &macroblock.dc_low_pass,
                &self.coded[component],
                (position, self.syntax.geometry(component)),
                dc_mode,
                self.entropy.lp_model.bits(component != 0),
            )?);
        }
        let chroma = joint_chroma
            .map(|sampling| {
                JointChromaLowPass::new(
                    [&macroblocks[1].dc_low_pass, &macroblocks[2].dc_low_pass],
                    [&self.coded[1], &self.coded[2]],
                    (position, self.syntax.geometry(1)),
                    dc_mode,
                    self.entropy.lp_model.bits(true),
                    sampling,
                )
            })
            .transpose()?;
        if self.syntax.joint_patterns() {
            // The joint pattern carries luma in its lowest bit, then each
            // chroma component, or both of them in one bit when they share
            // a block.
            let pattern = self
                .low_pass
                .iter()
                .map(LowPass::is_coded)
                .chain(chroma.iter().map(JointChromaLowPass::is_coded))
                .enumerate()
                .fold(0_u8, |pattern, (bit, coded)| {
                    pattern | u8::from(coded) << bit
                });
            self.low_pass_pattern.encode(writer, pattern)?;
        } else {
            for block in &self.low_pass {
                writer.write_flag(block.is_coded());
            }
        }
        let mut lap_mean = [0_i32; 2];
        for (component, block) in self.low_pass.iter().enumerate() {
            lap_mean[usize::from(component != 0)] += block.write(
                writer,
                ComponentClass::of(component),
                &mut self.entropy.lp_vlc,
                &mut self.entropy.lp_scan,
            )?;
        }
        if let Some(chroma) = chroma {
            lap_mean[1] += chroma.write(writer, &mut self.entropy.lp_vlc)?;
        }
        self.entropy
            .lp_model
            .update(lap_mean, self.syntax.colour_model())?;
        Ok(())
    }

    /// Writes every component's coded-block pattern, then each component's
    /// blocks in component order, the flexbits of each block to the packet
    /// that carries flexbits.
    fn encode_high_pass(
        &mut self,
        packets: &mut TilePackets,
        position: MacroblockPosition,
        macroblocks: &[&QuantizedMacroblock],
    ) -> Result<(), TileEncodeError> {
        let syntax = HighPassSyntax {
            mode: self
                .syntax
                .high_pass_mode(|component| &macroblocks[component].dc_low_pass),
            bands: self.options.bands,
            trim_flexbits: self.options.trim_flexbits.unwrap_or(0),
        };
        for (component, (band, macroblock)) in
            self.high_pass.iter_mut().zip(macroblocks).enumerate()
        {
            band.form(
                &macroblock.high_pass,
                syntax,
                self.entropy.hp_model.bits(component != 0),
                self.syntax.geometry(component),
            )?;
        }
        let MacroblockPosition { x, y, .. } = position;
        let writer = packets.band(PacketBand::HighPass)?;
        if let ComponentSyntax::Yuv(sampling) = self.syntax {
            let patterns = core::array::from_fn(|component| self.high_pass[component].pattern());
            self.high_pass_pattern
                .encode_yuv(writer, x, y, sampling, patterns)?;
        } else {
            for (component, band) in self.high_pass.iter().enumerate() {
                self.high_pass_pattern
                    .encode_component(writer, x, y, component, band.pattern())?;
            }
        }
        let mut lap_mean = [0_i32; 2];
        for (component, band) in self.high_pass.iter().enumerate() {
            lap_mean[usize::from(component != 0)] += band.write(
                packets,
                ComponentClass::of(component),
                syntax,
                &mut self.entropy.hp_vlc,
                &mut self.entropy.hp_scan,
            )?;
        }
        self.entropy
            .hp_model
            .update(lap_mean, self.syntax.colour_model())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use jxr_core::{BandPresence, BitstreamMode, ChromaSampling, QuantizerSet};

    use super::*;
    use crate::tile_decode::component_geometry;

    fn options(colour: ColorFormat) -> PacketOptions {
        PacketOptions {
            colour,
            bands: BandPresence::All,
            trim_flexbits: None,
            mode: BitstreamMode::Spatial,
        }
    }

    fn components(count: usize) -> Vec<Vec<QuantizedMacroblock>> {
        let macroblock = QuantizedMacroblock {
            dc_low_pass: [0; 16],
            high_pass: [0; 256],
            quantizers: QuantizerSet {
                dc: 1,
                low_pass: 1,
                high_pass: 1,
            },
            bands: BandPresence::All,
        };
        vec![vec![macroblock]; count]
    }

    const TILE: TileRegion = TileRegion {
        stride: 1,
        x: 0,
        y: 0,
        width: 1,
        height: 1,
    };

    /// The packets of the one-macroblock tile of `components`.
    fn encode(
        components: &[Vec<QuantizedMacroblock>],
        colour: ColorFormat,
    ) -> Result<Vec<Vec<u8>>, TileEncodeError> {
        encode_tile(
            components,
            None,
            TILE,
            options(colour),
            &mut TileScratch::default(),
        )
    }

    #[test]
    fn every_internal_colour_format_has_a_syntax() {
        for (colour, syntax, count) in [
            (ColorFormat::Luma, ComponentSyntax::Luma, 1),
            (
                ColorFormat::Yuv(ChromaSampling::Cs444),
                ComponentSyntax::Yuv(ChromaSampling::Cs444),
                3,
            ),
            (
                ColorFormat::Yuv(ChromaSampling::Cs422),
                ComponentSyntax::Yuv(ChromaSampling::Cs422),
                3,
            ),
            (
                ColorFormat::Yuv(ChromaSampling::Cs420),
                ComponentSyntax::Yuv(ChromaSampling::Cs420),
                3,
            ),
            (ColorFormat::YuvK, ComponentSyntax::YuvK, 4),
            (
                ColorFormat::NComponent(2),
                ComponentSyntax::NComponent(2),
                2,
            ),
            (
                ColorFormat::NComponent(16),
                ComponentSyntax::NComponent(16),
                16,
            ),
        ] {
            assert_eq!(ComponentSyntax::new(colour), Ok(syntax));
            assert_eq!(syntax.components(), count);
            assert!(encode(&components(count), colour).is_ok());
            // A plane must supply exactly the components its format codes.
            assert_eq!(
                encode(&components(count + 1), colour),
                Err(TileEncodeError::InvalidInput("component count"))
            );
        }
    }

    /// Macroblocks with coefficients in every band, enough for a tile of
    /// `count` positions.
    fn busy_components(
        components: usize,
        count: usize,
        seed: u64,
    ) -> Vec<Vec<QuantizedMacroblock>> {
        use crate::entropy::encode::test_support::Lcg;

        let mut rng = Lcg(seed);
        (0..components)
            .map(|_| {
                (0..count)
                    .map(|_| QuantizedMacroblock {
                        dc_low_pass: core::array::from_fn(|_| rng.coefficient(9)),
                        high_pass: core::array::from_fn(|index| {
                            if index % 16 == 0 {
                                0
                            } else {
                                rng.coefficient(7)
                            }
                        }),
                        quantizers: QuantizerSet {
                            dc: 1,
                            low_pass: 1,
                            high_pass: 1,
                        },
                        bands: BandPresence::All,
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn scratch_left_by_one_tile_does_not_change_the_next() {
        // Tiles of different shapes, formats, and packet layouts through
        // one scratch, as a worker codes them: larger after smaller, fewer
        // components after more, with and without an alpha plane.
        let mut scratch = TileScratch::default();
        for (round, (colour, count, width, height, alpha, mode)) in [
            (
                ColorFormat::Yuv(ChromaSampling::Cs444),
                3,
                3,
                2,
                false,
                BitstreamMode::Spatial,
            ),
            (ColorFormat::Luma, 1, 5, 3, true, BitstreamMode::Frequency),
            (
                ColorFormat::NComponent(6),
                6,
                2,
                2,
                false,
                BitstreamMode::Spatial,
            ),
            (
                ColorFormat::Yuv(ChromaSampling::Cs444),
                3,
                4,
                4,
                true,
                BitstreamMode::Spatial,
            ),
            (ColorFormat::Luma, 1, 1, 1, false, BitstreamMode::Frequency),
            (ColorFormat::YuvK, 4, 3, 3, false, BitstreamMode::Frequency),
        ]
        .into_iter()
        .enumerate()
        {
            let seed = 7 + u64::try_from(round).unwrap();
            let components = busy_components(count, width * height, seed);
            let alpha = alpha.then(|| busy_components(1, width * height, seed + 100).remove(0));
            let tile = TileRegion {
                stride: width,
                x: 0,
                y: 0,
                width,
                height,
            };
            let options = PacketOptions {
                mode,
                ..options(colour)
            };
            let fresh = encode_tile(
                &components,
                alpha.as_deref(),
                tile,
                options,
                &mut TileScratch::default(),
            )
            .unwrap();
            let reused =
                encode_tile(&components, alpha.as_deref(), tile, options, &mut scratch).unwrap();
            assert_eq!(reused, fresh, "tile {round}");
            // A packet takes exactly the storage it needs; the buffers it
            // was written in stay with the scratch.
            assert!(
                reused
                    .iter()
                    .all(|packet| packet.capacity() == packet.len())
            );
            assert_eq!(
                scratch.buffers.len(),
                fresh.len().max(scratch.buffers.len())
            );
        }
        // What the scratch keeps is two rows of its widest tile, which is
        // five macroblocks wide, and not a tile.
        assert!(
            scratch
                .primary
                .coded
                .iter()
                .all(|history| history.capacity() <= 10)
        );
    }

    /// [`busy_components`] in the block geometry of `colour`: entries
    /// beyond a component's block count are zero, as analysis leaves them.
    fn shaped_components(
        colour: ColorFormat,
        count: usize,
        seed: u64,
    ) -> Vec<Vec<QuantizedMacroblock>> {
        let mut components =
            busy_components(usize::from(colour.component_count().unwrap()), count, seed);
        for (component, macroblocks) in components.iter_mut().enumerate() {
            let blocks = component_geometry(colour, component).block_count();
            for macroblock in macroblocks {
                macroblock.dc_low_pass[blocks..].fill(0);
                macroblock.high_pass[blocks * 16..].fill(0);
            }
        }
        components
    }

    #[test]
    fn a_tile_coded_band_by_band_writes_the_packets_of_the_whole_tile() {
        // A tile three macroblocks wide, two in from the left of an image
        // five wide, coded from bands that hold only their own rows.
        const IMAGE_WIDTH: usize = 5;
        let (x, width, height) = (2, 3, 7);
        let bands: [&[usize]; 4] = [&[1; 7], &[2, 2, 2, 1], &[3, 4], &[7]];
        for (round, colour) in [
            ColorFormat::Luma,
            ColorFormat::Yuv(ChromaSampling::Cs444),
            ColorFormat::Yuv(ChromaSampling::Cs422),
            ColorFormat::Yuv(ChromaSampling::Cs420),
            ColorFormat::YuvK,
            ColorFormat::NComponent(5),
        ]
        .into_iter()
        .enumerate()
        {
            let seed = 31 + 2 * u64::try_from(round).unwrap();
            let components = shaped_components(colour, IMAGE_WIDTH * height, seed);
            for (alpha, mode, trim_flexbits) in [
                (false, BitstreamMode::Spatial, None),
                (false, BitstreamMode::Frequency, Some(2)),
                (true, BitstreamMode::Spatial, None),
                (true, BitstreamMode::Frequency, None),
            ] {
                let alpha =
                    alpha.then(|| busy_components(1, IMAGE_WIDTH * height, seed + 1).remove(0));
                let options = PacketOptions {
                    trim_flexbits,
                    mode,
                    ..options(colour)
                };
                let tile = TileRegion {
                    stride: IMAGE_WIDTH,
                    x,
                    y: 0,
                    width,
                    height,
                };
                let whole = encode_tile(
                    &components,
                    alpha.as_deref(),
                    tile,
                    options,
                    &mut TileScratch::default(),
                )
                .unwrap();
                let shape = TileShape {
                    width,
                    height,
                    components: components.len(),
                    alpha: alpha.is_some(),
                };
                for (rows, lend) in bands.iter().zip([false, true].into_iter().cycle()) {
                    let case = format!(
                        "{colour:?} {mode:?} alpha {} bands {rows:?} lent output {lend}",
                        alpha.is_some()
                    );
                    // Packets follow whatever the output already holds,
                    // whether it is lent to the first packet or not.
                    let mut output = vec![0xa5];
                    let lent = lend.then(|| core::mem::take(&mut output));
                    let mut writer =
                        TileWriter::begin(shape, options, TileScratch::default(), lent).unwrap();
                    assert_eq!(writer.holds_output(), lend, "{case}");
                    let mut first = 0;
                    for &count in *rows {
                        let band = |plane: &[QuantizedMacroblock]| {
                            plane[first * IMAGE_WIDTH..(first + count) * IMAGE_WIDTH].to_vec()
                        };
                        let band_components: Vec<_> =
                            components.iter().map(|plane| band(plane)).collect();
                        let band_alpha = alpha.as_deref().map(band);
                        let rows = TileRegion {
                            height: count,
                            ..tile
                        };
                        writer
                            .push_rows(&band_components, band_alpha.as_deref(), rows)
                            .unwrap();
                        first += count;
                    }
                    let mut lengths = Vec::new();
                    writer.finish(&mut output, &mut lengths).unwrap();
                    assert_eq!(output[0], 0xa5, "{case}");
                    assert_eq!(output[1..], whole.concat(), "{case}");
                    assert_eq!(
                        lengths,
                        whole.iter().map(Vec::len).collect::<Vec<_>>(),
                        "{case}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_tile_takes_exactly_its_own_rows() {
        let components = busy_components(1, 6, 3);
        let shape = TileShape {
            width: 2,
            height: 3,
            components: 1,
            alpha: false,
        };
        let rows = |height| TileRegion {
            stride: 2,
            x: 0,
            y: 0,
            width: 2,
            height,
        };
        let begin = || {
            TileWriter::begin(
                shape,
                options(ColorFormat::Luma),
                TileScratch::default(),
                None,
            )
        };
        let invalid = Err(TileEncodeError::InvalidInput("tile rows"));
        // More rows than the tile has left, or a band of another width.
        let mut writer = begin().unwrap();
        writer.push_rows(&components, None, rows(2)).unwrap();
        assert_eq!(writer.push_rows(&components, None, rows(2)), invalid);
        let narrow = TileRegion {
            width: 1,
            ..rows(1)
        };
        assert_eq!(writer.push_rows(&components, None, narrow), invalid);
        // A plane the tile does not code.
        assert_eq!(
            writer.push_rows(&components, Some(&components[0]), rows(1)),
            Err(TileEncodeError::InvalidInput("component count"))
        );
        // A tile is finished only once all of its rows are coded.
        let mut output = Vec::new();
        assert!(matches!(
            writer.finish(&mut output, &mut Vec::new()),
            Err(TileEncodeError::InvalidInput("tile rows"))
        ));
        let mut writer = begin().unwrap();
        writer.push_rows(&components, None, rows(3)).unwrap();
        let mut lengths = Vec::new();
        writer.finish(&mut output, &mut lengths).unwrap();
        assert_eq!(lengths, [output.len()]);
        // A tile lent the output gives it back only to an empty one.
        let lent = Some(core::mem::take(&mut output));
        let mut writer = TileWriter::begin(
            shape,
            options(ColorFormat::Luma),
            TileScratch::default(),
            lent,
        )
        .unwrap();
        writer.push_rows(&components, None, rows(3)).unwrap();
        assert!(matches!(
            writer.finish(&mut vec![0], &mut Vec::new()),
            Err(TileEncodeError::InvalidInput("output written while lent"))
        ));
    }

    #[test]
    fn formats_without_a_writer_are_unsupported() {
        for colour in [
            ColorFormat::Rgb,
            ColorFormat::Cmyk,
            ColorFormat::NComponent(1),
            ColorFormat::NComponent(17),
        ] {
            assert!(matches!(
                encode(&components(3), colour),
                Err(TileEncodeError::Unsupported(_))
            ));
        }
    }
}
