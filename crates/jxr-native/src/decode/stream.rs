//! Full-resolution CPU decode as a stream of macroblock rows.
//!
//! A decode moves down the reconstruction window a band of macroblock rows
//! at a time. Each band is reconstructed by the strips of its components,
//! the sample rows that this finishes are converted and stored straight into
//! the caller's output, and they are then let go. What is held at any moment is
//! a few rows of every component, whatever the height of the image.

use core::ops::Range;

use jxr_core::{
    CoefficientArena, JxrError, JxrErrorKind, PreparedPlan, scaled_after_first_transform,
};

use crate::{
    CpuCapabilities, ParsedCodestream,
    coefficient::map_tile_error,
    executor::Executor,
    metadata::interpolates_chroma,
    output_format::ComponentPlane,
    reconstruct::{ChromaReconstructionConfig, ChromaStrip, ReconstructionSimd},
    tile_decode::{BandDecoder, component_geometry},
};

use super::{
    output::OutputRows,
    reconstruction::{
        Band, ComponentWorkspace, ReconstructionWorkspace, Window, map_reconstruction_error,
        primary_component_count,
    },
};

/// Luma samples a band of a decode on one thread is sized to hold in each
/// sample row's worth of macroblock rows: narrow images take several
/// macroblock rows at a step, so that the cost of a step is spread over a
/// useful amount of work.
const BAND_SAMPLES: usize = 16 * 2048;
/// Macroblocks a band of a decode through the thread pool is sized to
/// hold, so that handing it to another thread costs little beside it.
const POOL_BAND_MACROBLOCKS: usize = 2048;
/// Most macroblock rows of one band on one thread, and through the pool.
const MAX_BAND_ROWS: usize = 16;
const MAX_POOL_BAND_ROWS: usize = 8;

/// The stage of a decode an error came from, in the order a decode that
/// finished each stage over the whole image before starting the next would
/// meet them: the primary image's packets, its reconstruction, the same of
/// a separate alpha image, and the output.
///
/// A stream meets them a band at a time instead, so an error in a late
/// stage of an early band is held while the earlier stages run on to the
/// end of the image: if one of them fails, that is the error a decode
/// reports, as it always has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Stage {
    Entropy,
    Reconstruction,
    AlphaEntropy,
    AlphaReconstruction,
    Output,
}

/// A band of macroblock rows whose coefficients are kept after the strip
/// that read them has passed, for the strip that joins it to its neighbour.
#[derive(Debug)]
pub(super) struct KeptBand {
    /// Macroblock rows of the window the band holds.
    pub(super) rows: Range<usize>,
    pub(super) arena: CoefficientArena,
}

/// The macroblock rows of the window one strip runs over.
#[derive(Debug, Clone)]
pub(super) struct StripRows {
    /// The tile rows the strip reads, of those the plan selects.
    pub(super) tile_rows: Range<usize>,
    /// The rows it takes in: the rows it stores and, where a neighbour's
    /// rows reach into them, one more on that side.
    pub(super) taken: Range<usize>,
    /// The rows whose samples it stores.
    pub(super) stored: Range<usize>,
    /// Whether the bands of its first two rows, and of its last two, are
    /// kept for the strip that joins it to the neighbour on that side.
    pub(super) keep: [bool; 2],
}

impl StripRows {
    /// The one strip of a window of `rows` macroblock rows in `tile_rows`
    /// tile rows.
    pub(super) const fn whole(tile_rows: usize, rows: usize) -> Self {
        Self {
            tile_rows: 0..tile_rows,
            taken: 0..rows,
            stored: 0..rows,
            keep: [false; 2],
        }
    }

    /// Whether a band that begins at `row` is one of those kept.
    const fn keeps(&self, row: usize) -> bool {
        (self.keep[0] && row < self.taken.start + 2) || (self.keep[1] && row + 2 >= self.taken.end)
    }

    /// Macroblock rows to ask for in the band that begins at `row`, of
    /// which `step` are taken where nothing is kept.
    fn band_rows(&self, row: usize, step: usize) -> usize {
        if self.keep[0] && row < self.taken.start + 2 {
            self.taken.start + 2 - row
        } else if self.keep[1] && row + 2 >= self.taken.end {
            self.taken.end - row
        } else if self.keep[1] {
            step.min(self.taken.end - 2 - row)
        } else {
            step
        }
    }
}

/// Where the bands of a stream come from.
enum BandSource<'a> {
    /// The packets of the tiles, read as the stream goes.
    Packets(BandDecoder<'a>),
    /// Bands other strips read and kept.
    Kept(core::slice::Iter<'a, KeptBand>),
}

/// The storage one stream runs in, retained between decodes.
#[derive(Debug, Default)]
pub(super) struct StreamWorkspace {
    /// Coefficients of the band being reconstructed.
    pub(super) arena: CoefficientArena,
    /// Coefficients of the band read while that one is reconstructed.
    pub(super) ahead: CoefficientArena,
    pub(super) reconstruction: ReconstructionWorkspace,
}

/// How a stream uses the threads it is given.
#[derive(Debug, Clone, Copy)]
pub(super) struct Threads {
    /// Whether the tiles of a band are read through the thread pool.
    pub(super) tiles: bool,
    /// Whether the next band is read while the last one is reconstructed.
    pub(super) read_ahead: bool,
    /// Macroblock rows of a band.
    pub(super) step: usize,
    /// Where the tiles of a band, the next band and the components of a
    /// band run, where the stream reads them side by side.
    pub(super) executor: Executor,
}

impl Threads {
    /// A decode on the calling thread alone, in the smallest bands that
    /// spread the cost of a step over a useful amount of work.
    pub(super) fn caller(columns: usize) -> Self {
        Self {
            tiles: false,
            read_ahead: false,
            step: (BAND_SAMPLES / (columns * 256).max(1)).clamp(1, MAX_BAND_ROWS),
            executor: Executor::Sequential,
        }
    }

    /// A decode through the thread pool, in bands large enough to be worth
    /// handing to another thread: the tiles of a band are read on several
    /// threads, and the next band is read while the last one is
    /// reconstructed and stored.
    pub(super) fn pool(executor: Executor, columns: usize, tile_columns: usize) -> Self {
        Self {
            tiles: tile_columns > 1,
            read_ahead: true,
            step: (POOL_BAND_MACROBLOCKS / columns.max(1)).clamp(1, MAX_POOL_BAND_ROWS),
            executor,
        }
    }

    /// The same for an image shared out among several strips, which read a
    /// band ahead only if `spare` says there are threads that have no strip
    /// to run.
    pub(super) const fn strips(self, spare: bool) -> Self {
        Self {
            read_ahead: spare,
            ..self
        }
    }
}

/// The strips of the planes of one codestream: its primary components, then
/// an integrated alpha plane, and the interpolation of subsampled chroma.
pub(super) struct PlaneStrips<'a> {
    window: Window,
    positions: &'a mut Vec<u32>,
    components: &'a mut [ComponentWorkspace],
    chroma: &'a mut [ChromaStrip; 2],
    /// Sample columns and rows of one macroblock of each plane.
    macroblock: Vec<[u32; 2]>,
    /// Primary components: the planes before any integrated alpha plane.
    primary: usize,
    /// Whether U and V are interpolated to full resolution.
    interpolated: bool,
}

/// The planes of one codestream, streamed down a strip of its
/// reconstruction window: the bands of coefficients it reads and the
/// strips that reconstruct them.
pub(super) struct PlaneStream<'a> {
    strips: PlaneStrips<'a>,
    source: BandSource<'a>,
    /// The coefficients of the band being reconstructed.
    arena: &'a mut CoefficientArena,
    /// The coefficients of a band read ahead, and its rows.
    ahead_arena: &'a mut CoefficientArena,
    ahead: Option<Range<usize>>,
    rows: StripRows,
    /// Macroblock rows of the window read.
    read: usize,
    /// Macroblock rows of the window taken in.
    taken: usize,
    threads: Threads,
    /// Whether the bands read are still reconstructed: not once
    /// reconstruction has failed and only the packets are read on.
    reconstructing: bool,
    coefficient_reuse: ArenaReuse,
    /// The bands kept for the strips that join this one to its neighbours.
    kept: Vec<KeptBand>,
}

/// Whether retained arena storage was used, and whether any arena still had
/// to grow during this stream.
#[derive(Debug, Default, Clone, Copy)]
struct ArenaReuse {
    read: bool,
    grew: bool,
}

impl ArenaReuse {
    fn record(&mut self, reused: bool) {
        self.read = true;
        self.grew |= !reused;
    }

    const fn complete(self) -> bool {
        self.read && !self.grew
    }
}

/// One codestream and the plan of decoding it.
#[derive(Clone, Copy)]
pub(super) struct Codestream<'a> {
    pub(super) source: &'a [u8],
    pub(super) parsed: &'a ParsedCodestream,
    pub(super) plan: &'a PreparedPlan,
    /// Whether the codestream has an alpha plane beside its primary one.
    pub(super) integrated_alpha: bool,
}

impl<'a> PlaneStream<'a> {
    /// Begins a stream over the rows `rows` of the tiles the plan selects,
    /// read from the codestream's packets.
    pub(super) fn new(
        codestream: Codestream<'a>,
        rows: StripRows,
        cpu: CpuCapabilities,
        threads: Threads,
        workspace: &'a mut StreamWorkspace,
    ) -> Result<Self, (Stage, JxrError)> {
        let Codestream {
            source,
            parsed,
            plan,
            ..
        } = codestream;
        let mut decoder = BandDecoder::new(
            source,
            parsed,
            plan,
            threads.executor.if_worth(threads.tiles),
        )
        .map_err(|error| (Stage::Entropy, map_tile_error(&error)))?;
        decoder.restrict(rows.tile_rows.clone());
        Self::begin(
            BandSource::Packets(decoder),
            codestream,
            rows,
            cpu,
            threads,
            workspace,
        )
        .map_err(|error| (Stage::Reconstruction, error))
    }

    /// Begins a stream over the rows `rows` from `bands`, which other
    /// strips read and kept and which hold exactly the rows it takes in.
    pub(super) fn joining(
        codestream: Codestream<'a>,
        rows: StripRows,
        bands: &'a [KeptBand],
        cpu: CpuCapabilities,
        workspace: &'a mut StreamWorkspace,
    ) -> Result<Self, (Stage, JxrError)> {
        let threads = Threads {
            tiles: false,
            read_ahead: false,
            step: 2,
            executor: Executor::Sequential,
        };
        Self::begin(
            BandSource::Kept(bands.iter()),
            codestream,
            rows,
            cpu,
            threads,
            workspace,
        )
        .map_err(|error| (Stage::Reconstruction, error))
    }

    fn begin(
        source: BandSource<'a>,
        codestream: Codestream<'_>,
        rows: StripRows,
        cpu: CpuCapabilities,
        threads: Threads,
        workspace: &'a mut StreamWorkspace,
    ) -> Result<Self, JxrError> {
        let Codestream {
            parsed,
            plan,
            integrated_alpha,
            ..
        } = codestream;
        let primary = primary_component_count(plan)?;
        let planes = primary + usize::from(integrated_alpha);
        let window = Window::of_plan(plan)?;
        let StreamWorkspace {
            arena,
            ahead,
            reconstruction,
        } = workspace;
        reconstruction.prepare(planes);
        let ReconstructionWorkspace {
            components,
            chroma,
            positions,
            ..
        } = reconstruction;
        let mut macroblock = Vec::with_capacity(planes);
        for (index, component) in components[..planes].iter_mut().enumerate() {
            let geometry = if index < primary {
                component_geometry(plan.info.primary.color_format, index)
            } else {
                component_geometry(jxr_core::ColorFormat::Luma, 0)
            };
            let scaled =
                index < primary && scaled_after_first_transform(plan.info.primary.scaled, index);
            let config = window.config([geometry.columns, geometry.rows], scaled, plan);
            component.begin_strip(&config, cpu, rows.taken.start, threads.step)?;
            macroblock.push([
                u32::from(geometry.columns) * 4,
                u32::from(geometry.rows) * 4,
            ]);
        }
        let interpolated = interpolates_chroma(
            parsed.headers.primary.internal_color_format,
            parsed.headers.image.output_color_format,
            plan.scale,
        );
        if interpolated {
            if primary != 3 {
                return Err(JxrError::new(
                    JxrErrorKind::InternalInvariant,
                    "subsampled YUV component count",
                ));
            }
            let config = ChromaReconstructionConfig::from_header(
                &parsed.headers.primary,
                window.width * 16,
                window.height * 16,
            )
            .map_err(|error| map_reconstruction_error(&error))?;
            let [width, height] = macroblock[1].map(|samples| samples as usize);
            let size = [window.columns() * width, window.rows() * height];
            for strip in chroma.iter_mut() {
                // The chroma row above the first one stored is the last of
                // the row the strip takes in above it.
                strip
                    .begin(
                        config,
                        size,
                        rows.stored.start * height,
                        (threads.step + 1) * 16,
                    )
                    .map_err(|error| map_reconstruction_error(&error))?;
            }
        }
        Ok(Self {
            strips: PlaneStrips {
                window,
                positions,
                components: &mut components[..planes],
                chroma,
                macroblock,
                primary,
                interpolated,
            },
            source,
            arena,
            ahead_arena: ahead,
            ahead: None,
            read: rows.taken.start,
            taken: rows.taken.start,
            rows,
            threads,
            reconstructing: true,
            coefficient_reuse: ArenaReuse::default(),
            kept: Vec::new(),
        })
    }

    /// Whether every macroblock row of the strip has been taken in.
    pub(super) fn exhausted(&self) -> bool {
        self.taken == self.rows.taken.end
    }

    /// Stops reconstructing the bands read: the packets alone are read on.
    pub(super) fn stop_reconstructing(&mut self) {
        self.reconstructing = false;
    }

    /// Whether this stream decoded a band into an arena that already had
    /// sufficient capacity before the stream began.
    pub(super) const fn coefficient_reused(&self) -> bool {
        self.coefficient_reuse.complete()
    }

    /// The bands kept for the strips that join this one to its neighbours,
    /// in row order, once the stream is exhausted.
    pub(super) fn take_kept(&mut self) -> Vec<KeptBand> {
        core::mem::take(&mut self.kept)
    }

    /// The strips of the stream's planes, for their finished rows.
    pub(super) fn strips(&mut self) -> &mut PlaneStrips<'a> {
        &mut self.strips
    }

    /// Reconstruction stages that used native vector groups in this stream.
    pub(super) fn simd(&self) -> ReconstructionSimd {
        self.strips.simd()
    }

    /// Takes in the next band of macroblock rows, reconstructs what it
    /// finishes, and then calls `finished` with the strips and whether the
    /// band was the last. With a thread to spare, the band after it is read
    /// while all that happens. The stage that failed comes with an error:
    /// `Entropy` or `Reconstruction`, whichever image the stream is of.
    /// `finished` is not called for a band that fails to reconstruct.
    pub(super) fn step(
        &mut self,
        finished: impl FnOnce(&mut PlaneStrips<'a>, bool) + Send,
    ) -> Result<(), (Stage, JxrError)> {
        let Self {
            strips,
            source,
            arena,
            ahead_arena,
            ahead,
            rows,
            read,
            taken,
            threads,
            reconstructing,
            coefficient_reuse,
            kept,
        } = self;
        let last_row = rows.taken.end;
        let spread = threads.executor.if_worth(threads.read_ahead);
        let reconstruct = *reconstructing;
        // Reconstructs a band and hands its strips on.
        let take_in =
            |strips: &mut PlaneStrips<'a>, arena: &CoefficientArena, band: Range<usize>| {
                let last = band.end == last_row;
                if reconstruct {
                    strips.take_in(arena, band, last, spread)?;
                }
                finished(strips, last);
                Ok(())
            };
        let decoder = match source {
            BandSource::Kept(bands) => {
                let band = bands.next().ok_or_else(|| {
                    let error = JxrError::new(JxrErrorKind::InternalInvariant, "kept band");
                    (Stage::Entropy, error)
                })?;
                *taken = band.rows.end;
                *read = band.rows.end;
                return take_in(strips, &band.arena, band.rows.clone())
                    .map_err(|error| (Stage::Reconstruction, error));
            }
            BandSource::Packets(decoder) => decoder,
        };
        let mut read_band =
            |arena: &mut CoefficientArena| -> Result<(Range<usize>, bool), JxrError> {
                let wanted = u32::try_from(rows.band_rows(*read, threads.step)).unwrap_or(1);
                let (count, reused) = decoder
                    .decode_band(wanted, arena)
                    .map_err(|error| map_tile_error(&error))?;
                let band = *read..*read + count as usize;
                if band.is_empty() || band.end > last_row {
                    return Err(JxrError::new(
                        JxrErrorKind::InternalInvariant,
                        "decoded macroblock rows",
                    ));
                }
                *read = band.end;
                Ok((band, reused))
            };
        let band = if let Some(band) = ahead.take() {
            band
        } else {
            let (band, reused) = read_band(arena).map_err(|error| (Stage::Entropy, error))?;
            coefficient_reuse.record(reused);
            band
        };
        *taken = band.end;
        let reconstructed = if threads.read_ahead && band.end != last_row {
            let (reconstructed, next) = threads.executor.join(
                || take_in(strips, arena, band.clone()),
                || read_band(ahead_arena),
            );
            let (next, reused) = next.map_err(|error| (Stage::Entropy, error))?;
            coefficient_reuse.record(reused);
            *ahead = Some(next);
            reconstructed
        } else {
            take_in(strips, arena, band.clone())
        };
        if rows.keeps(band.start) {
            kept.push(KeptBand {
                rows: band,
                arena: core::mem::take(&mut **arena),
            });
        }
        if ahead.is_some() {
            core::mem::swap(&mut **arena, &mut **ahead_arena);
        }
        reconstructed.map_err(|error| (Stage::Reconstruction, error))
    }
}

impl PlaneStrips<'_> {
    fn simd(&self) -> ReconstructionSimd {
        self.components
            .iter()
            .fold(ReconstructionSimd::default(), |mut simd, component| {
                simd.include(component.strip.simd());
                simd
            })
    }

    /// Reconstructs the macroblock rows `rows` of the window from `arena`,
    /// which holds exactly them: every component's strip takes them in,
    /// side by side through `spread`, and subsampled chroma is interpolated
    /// as far as they finish it. `last` says that no rows follow, so the
    /// strips finish what they hold.
    fn take_in(
        &mut self,
        arena: &CoefficientArena,
        rows: Range<usize>,
        last: bool,
        spread: Executor,
    ) -> Result<(), JxrError> {
        let band = Band::new(arena, &self.window, rows.clone(), self.positions)?;
        let take_in = |plane: usize, component: &mut ComponentWorkspace| {
            let source = band.plane(plane)?;
            component
                .strip
                .push_rows(&source, rows.clone())
                .and_then(|()| {
                    if last {
                        component.strip.flush()
                    } else {
                        Ok(())
                    }
                })
                .map_err(|error| map_reconstruction_error(&error))
        };
        spread
            .if_worth(self.components.len() > 1)
            .try_for_each(self.components, take_in)?;
        if self.interpolated {
            for (chroma, component) in self.chroma.iter_mut().zip(&mut self.components[1..3]) {
                chroma
                    .advance(&mut component.strip)
                    .map_err(|error| map_reconstruction_error(&error))?;
            }
        }
        Ok(())
    }

    /// Whether plane `plane` is read from the interpolation of its strip.
    fn is_interpolated(&self, plane: usize) -> bool {
        self.interpolated && matches!(plane, 1 | 2)
    }

    /// Sample origin of plane `plane` in its own extended plane.
    fn origin(&self, plane: usize) -> [u32; 2] {
        let [width, height] = if self.is_interpolated(plane) {
            [16, 16]
        } else {
            self.macroblock[plane]
        };
        [self.window.origin_x * width, self.window.origin_y * height]
    }

    /// First row of plane `plane`, in its own extended plane, that is not
    /// finished.
    fn finished(&self, plane: usize) -> u32 {
        let rows = if self.is_interpolated(plane) {
            self.chroma[plane - 1].finished()
        } else {
            self.components[plane].strip.finished()
        };
        self.origin(plane)[1] + u32::try_from(rows).unwrap_or(u32::MAX)
    }

    /// The finished rows `rows` of plane `plane`, by row of its extended
    /// plane.
    fn view(&self, plane: usize, rows: Range<u32>) -> ComponentPlane<'_> {
        let [origin_x, origin_y] = self.origin(plane);
        let local = (rows.start - origin_y) as usize..(rows.end - origin_y) as usize;
        let (width, samples) = if self.is_interpolated(plane) {
            let chroma = &self.chroma[plane - 1];
            (chroma.width(), chroma.rows(local))
        } else {
            let strip = &self.components[plane].strip;
            (strip.width(), strip.rows(local))
        };
        ComponentPlane::positioned(
            origin_x,
            rows.start,
            u32::try_from(width).unwrap_or(u32::MAX),
            rows.end - rows.start,
            samples,
        )
    }

    /// Lets go of the rows of plane `plane` above `row` of its extended
    /// plane.
    fn release(&mut self, plane: usize, row: u32) {
        let local = row.saturating_sub(self.origin(plane)[1]) as usize;
        if self.is_interpolated(plane) {
            self.chroma[plane - 1].release(local);
        } else {
            self.components[plane].strip.release(local);
        }
    }

    /// Lets go of every row held: nothing will read them.
    fn release_all(&mut self) {
        for plane in 0..self.components.len() {
            self.release(plane, u32::MAX);
        }
    }

    /// Number of planes: the primary components and any integrated alpha.
    fn planes(&self) -> usize {
        self.components.len()
    }
}

/// Reads every band of the tile rows `tile_rows` without reconstructing
/// them, for the errors their packets hold.
pub(super) fn read_packets(
    codestream: Codestream<'_>,
    tile_rows: Range<usize>,
) -> Result<(), JxrError> {
    let Codestream {
        source,
        parsed,
        plan,
        ..
    } = codestream;
    let mut arena = CoefficientArena::default();
    let mut decoder = BandDecoder::new(source, parsed, plan, Executor::Sequential)
        .map_err(|error| map_tile_error(&error))?;
    decoder.restrict(tile_rows);
    loop {
        match decoder.decode_band(1, &mut arena) {
            Ok((0, _)) => return Ok(()),
            Ok(_) => {}
            Err(error) => return Err(map_tile_error(&error)),
        }
    }
}

/// Where the alpha rows of an output come from.
pub(super) enum AlphaRows<'s, 'a> {
    /// The image has none, or the request drops it.
    None,
    /// The plane after the primary components of the primary stream.
    Integrated,
    /// The one plane of a stream of its own.
    Separate(&'s mut PlaneStream<'a>),
}

/// Runs `primary`, and a separate alpha stream beside it, to the end of the
/// image, storing every finished row in `output`.
///
/// When a stage fails, the stages that a whole-image decode runs before it
/// are carried to the end of the image without it, and the first of them to
/// fail is the failure returned: see [`Stage`]. `failure` is a stage that
/// failed before the streams began.
pub(super) fn run(
    primary: &mut PlaneStream<'_>,
    mut alpha: AlphaRows<'_, '_>,
    mut output: Option<&mut OutputRows<'_>>,
    mut failure: Option<(Stage, JxrError)>,
) -> Option<(Stage, JxrError)> {
    loop {
        // What follows a band of the primary image: the alpha image is
        // brought level with it and the rows both have finished are stored,
        // while the primary image's next band is read.
        let follow = |strips: &mut PlaneStrips<'_>, last: bool| {
            follow_band(
                strips,
                last,
                &mut alpha,
                output.as_deref_mut(),
                &mut failure,
            );
        };
        let stepped = if primary.exhausted() {
            let mut follow = follow;
            follow(primary.strips(), true);
            Ok(())
        } else {
            primary.step(follow)
        };
        match stepped {
            Ok(()) => {}
            Err((Stage::Entropy, error)) => return Some((Stage::Entropy, error)),
            Err((_, error)) => {
                fail(&mut failure, Stage::Reconstruction, error);
                primary.stop_reconstructing();
            }
        }
        let alpha_done = match &alpha {
            AlphaRows::Separate(stream) => {
                stream.exhausted()
                    || failure
                        .as_ref()
                        .is_some_and(|(stage, _)| *stage <= Stage::AlphaEntropy)
            }
            _ => true,
        };
        if primary.exhausted() && alpha_done {
            break;
        }
    }
    let complete = output.as_deref().is_none_or(|output| {
        (0..output.plane_count()).all(|plane| output.pending(plane).is_empty())
    });
    if failure.is_none() && !complete {
        let error = JxrError::new(
            JxrErrorKind::InternalInvariant,
            "decoded rows do not cover the output",
        );
        return Some((Stage::Output, error));
    }
    failure
}

/// Holds a failure of `stage` unless a stage that comes before it has
/// already failed.
fn fail(failure: &mut Option<(Stage, JxrError)>, stage: Stage, error: JxrError) {
    if failure.as_ref().is_none_or(|(held, _)| stage < *held) {
        *failure = Some((stage, error));
    }
}

/// Follows a band of the primary image, whose strips are `primary`: a
/// separate alpha stream is stepped until it is level with them, or once
/// if its rows are no longer stored, and the rows every plane of the
/// output can now take are stored. `last` says the primary image has no
/// more bands.
fn follow_band(
    primary: &mut PlaneStrips<'_>,
    last: bool,
    alpha: &mut AlphaRows<'_, '_>,
    output: Option<&mut OutputRows<'_>>,
    failure: &mut Option<(Stage, JxrError)>,
) {
    let held = failure.as_ref().map(|(stage, _)| *stage);
    if let AlphaRows::Separate(stream) = alpha
        && held.is_none_or(|stage| stage > Stage::AlphaEntropy)
    {
        // The alpha stream keeps level with the primary one; once its
        // rows are no longer stored it only has to reach the end.
        let level = held.is_none();
        while !stream.exhausted()
            && (last || !level || stream.strips().finished(0) < primary.finished(0))
        {
            match stream.step(|_, _| ()) {
                Ok(()) => {}
                Err((Stage::Entropy, error)) => {
                    fail(failure, Stage::AlphaEntropy, error);
                    break;
                }
                Err((_, error)) => {
                    fail(failure, Stage::AlphaReconstruction, error);
                    stream.stop_reconstructing();
                }
            }
            if !level {
                break;
            }
        }
    }
    if failure.is_none()
        && let Some(output) = output
        && let Err(error) = store_finished(primary, alpha, output)
    {
        fail(failure, Stage::Output, error);
    }
    if failure.is_some() {
        // Nothing reads the rows any more.
        primary.release_all();
        if let AlphaRows::Separate(stream) = alpha {
            stream.strips().release_all();
        }
    }
}

/// Stores the rows every plane of the output can now take, and lets go of
/// the component rows nothing will read again.
fn store_finished(
    primary: &mut PlaneStrips<'_>,
    alpha: &mut AlphaRows<'_, '_>,
    output: &mut OutputRows<'_>,
) -> Result<(), JxrError> {
    let colour_planes = primary.primary;
    for plane in 0..output.plane_count() {
        let pending = output.pending(plane);
        let stored = output.stored(plane);
        // The rows of a planar plane are those of one component; the rows
        // of an interleaved output need every component and alpha.
        let alpha_finished = match alpha {
            AlphaRows::None => None,
            AlphaRows::Integrated => Some(primary.finished(colour_planes)),
            AlphaRows::Separate(stream) => Some(stream.strips().finished(0)),
        };
        let finished = match stored {
            Some(stored) if stored.alpha => alpha_finished.unwrap_or(pending.start),
            Some(stored) => primary.finished(stored.component),
            None => (0..colour_planes)
                .map(|component| primary.finished(component))
                .chain(alpha_finished)
                .min()
                .unwrap_or(pending.start),
        };
        let rows = pending.start..finished.min(pending.end);
        if !rows.is_empty() {
            let wants_alpha = stored.is_none_or(|stored| stored.alpha);
            let wants_colour = stored.is_none_or(|stored| !stored.alpha);
            let components: Vec<ComponentPlane<'_>> = (0..colour_planes)
                .map(|component| {
                    // A planar plane reads one component: the others stand
                    // in with the rows they happen to hold.
                    let rows = if wants_colour && stored.is_none_or(|s| s.component == component) {
                        rows.clone()
                    } else {
                        let held = primary.finished(component);
                        held..held
                    };
                    primary.view(component, rows)
                })
                .collect();
            let alpha_plane = match alpha {
                AlphaRows::None => None,
                _ if !wants_alpha => None,
                AlphaRows::Integrated => Some(primary.view(colour_planes, rows.clone())),
                AlphaRows::Separate(stream) => Some(stream.strips.view(0, rows.clone())),
            };
            output.store(plane, &components, alpha_plane, rows)?;
        }
    }
    release_stored(primary, alpha, output);
    Ok(())
}

/// Lets go of every component row that each plane reading it has stored.
fn release_stored(
    primary: &mut PlaneStrips<'_>,
    alpha: &mut AlphaRows<'_, '_>,
    output: &OutputRows<'_>,
) {
    let colour_planes = primary.primary;
    // Rows below the last one a plane stores are never read.
    let next = |plane: usize| {
        let pending = output.pending(plane);
        if pending.is_empty() {
            u32::MAX
        } else {
            pending.start
        }
    };
    let mut colour = vec![u32::MAX; colour_planes];
    let mut alpha_row = u32::MAX;
    for plane in 0..output.plane_count() {
        match output.stored(plane) {
            Some(stored) if stored.alpha => alpha_row = alpha_row.min(next(plane)),
            Some(stored) => colour[stored.component] = colour[stored.component].min(next(plane)),
            None => {
                for row in &mut colour {
                    *row = (*row).min(next(plane));
                }
                alpha_row = alpha_row.min(next(plane));
            }
        }
    }
    for (component, row) in colour.into_iter().enumerate() {
        primary.release(component, row);
    }
    match alpha {
        AlphaRows::None => {}
        AlphaRows::Integrated => primary.release(colour_planes, alpha_row),
        AlphaRows::Separate(stream) => stream.strips().release(0, alpha_row),
    }
    // An integrated alpha plane the request drops is reconstructed for its
    // errors alone.
    for plane in colour_planes..primary.planes() {
        if matches!(alpha, AlphaRows::None) {
            primary.release(plane, u32::MAX);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ArenaReuse;

    #[test]
    fn retained_arena_counts_only_when_no_storage_grows_during_the_stream() {
        let mut complete = ArenaReuse::default();
        complete.record(true);
        complete.record(true);
        assert!(complete.complete());

        let mut later_growth = ArenaReuse::default();
        later_growth.record(true);
        later_growth.record(false);
        assert!(!later_growth.complete());

        let mut allocated_then_reused = ArenaReuse::default();
        allocated_then_reused.record(false);
        allocated_then_reused.record(true);
        assert!(!allocated_then_reused.complete());
    }
}
