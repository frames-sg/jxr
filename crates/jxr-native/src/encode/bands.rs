//! Source rows to tile packets, a band of macroblock rows at a time.
//!
//! Each coded component is analyzed as a stream down its plane, an
//! [`AnalysisStrip`]: the source is read a few macroblock rows at a time,
//! straight into the rows each strip holds, chroma subsampled here is
//! filtered as its rows arrive ([`ChromaDecimation`]), and a macroblock row
//! is quantized once the filters that reach across macroblock rows have
//! finished it. The rows a filter waits on stay in the strip until the rows
//! below them arrive, so every source row is read and converted once and
//! every sample filtered and transformed once, and no stage holds a
//! whole-image plane of samples.
//!
//! Tiles are coded as their rows arrive: a tile row's writers begin with
//! the first band that reaches it and finish, appending their packets in
//! index order, with the band that reaches its end. On one thread, where it
//! bounds the memory of an encode, the source is read [`READ_ROWS`]
//! macroblock rows at a time, and each macroblock row is quantized and coded
//! on its own, so that an encode holds the samples of a few rows and the
//! coefficients of one. With threads, every tile a band reaches is coded in
//! parallel while the samples of the next band are read, filtered and
//! transformed; that band is quantized once the coding is done, into the
//! coefficients it leaves. The bands are admitted against the pool's
//! working bound ([`executor::admitted`]) by the samples and coefficients
//! they hold: an image that fits is one band, which is then the whole-image
//! encoder; a larger one is cut at tile rows, as many to a band as the
//! threads have one of its tiles each for. The coefficients, and therefore
//! the packets, depend on neither the band height nor the thread count.

use core::ops::Range;
use std::collections::VecDeque;

use jxr_core::{
    BandPresence, BitstreamMode, ChromaSampling, JxrError, JxrErrorKind, OverlapMode, QuantizerSet,
};

use crate::{
    CpuCapabilities,
    analyze::{AnalysisStrip, waiting_rows},
    bit_writer::reserve_for,
    executor::{self, Executor},
    reconstruct::{ChromaDecimation, QuantizedMacroblock, ReconstructionConfig},
    tile_encode::{PacketOptions, TileEncodeError, TileRegion, TileScratch, TileShape, TileWriter},
};

use super::{
    input::{PlaneSet, SourcePlanes, SourceReader},
    layout::{ImageLayout, MACROBLOCK_SAMPLES},
    reconstruction_error, tile_error,
};

/// Macroblock rows read from the source at a time on one thread: enough
/// that a read is a useful amount of work, and few enough that the samples
/// an encode holds go with the image width alone, as what a decode holds
/// does. With the rows the filters wait on, two macroblock rows at most, a
/// strip holds no more rows than a band of eight with the rows its filters
/// read around it. A macroblock row of a 4096-sample-wide image is 1 MiB of
/// samples as four planes.
const READ_ROWS: usize = 7;

/// Macroblock rows below a band that its last rows wait on: a macroblock
/// row is final once the first rows of the next but one have been read,
/// under second-level overlap, and so is chroma subsampled from them.
const LOOKAHEAD_ROWS: usize = waiting_rows(OverlapMode::Two);

/// Bytes of one macroblock of one component in a band: its coefficients,
/// and the samples of a full-resolution plane it is analyzed from.
const MACROBLOCK_BYTES: usize = size_of::<QuantizedMacroblock>()
    + (MACROBLOCK_SAMPLES * MACROBLOCK_SAMPLES) as usize * size_of::<i32>();

/// How the components of one image plane are transformed and quantized.
pub(super) struct PlaneAnalysis {
    /// The reconstruction configuration of each component, which the
    /// analysis inverts.
    pub(super) configs: Vec<ReconstructionConfig>,
    /// The decoder's quantizer steps of each component.
    pub(super) steps: Vec<QuantizerSet>,
    pub(super) bands: BandPresence,
}

impl PlaneAnalysis {
    fn components(&self) -> usize {
        self.configs.len()
    }
}

/// The source of a codestream's planes: what is read, and how its chroma is
/// subsampled.
pub(super) struct BandSource<'a> {
    pub(super) reader: SourceReader<'a>,
    /// Which planes the codestream codes: its colour planes, the alpha
    /// plane as a codestream of its own, or both.
    pub(super) set: PlaneSet,
    /// The sampling full-resolution chroma is reduced to, if it is.
    pub(super) decimation: Option<ChromaSampling>,
    /// Width and height of the padded full-resolution plane.
    pub(super) padded: (usize, usize),
    /// Macroblock rows of each band, and read at a time, when a test sets
    /// them rather than the working bound and [`READ_ROWS`].
    pub(super) band_rows: Option<usize>,
}

/// The planes of one codestream: the primary plane and an integrated alpha
/// plane.
pub(super) struct CodestreamPlanes<'a> {
    pub(super) primary: &'a PlaneAnalysis,
    pub(super) alpha: Option<&'a PlaneAnalysis>,
    pub(super) options: PacketOptions,
}

/// Code every tile of the codestream `planes` describes from `source`,
/// appending its packets to `output` in index order and their lengths to
/// `lengths`, on the vectors and executor of `cpu`.
///
/// Through a pool the bands are coded on one of its workers. Every stage of
/// a band runs its own parallel loops, and a loop started from outside the
/// pool hands its work to the pool and waits to be woken, once for each
/// loop of each band; from a worker it starts in place.
pub(super) fn code_bands(
    source: &BandSource<'_>,
    planes: &CodestreamPlanes<'_>,
    layout: &ImageLayout,
    cpu: CpuCapabilities,
    output: &mut Vec<u8>,
    lengths: &mut Vec<usize>,
) -> Result<(), JxrError> {
    cpu.executor()
        .enter(|| code_bands_entered(source, planes, layout, cpu, output, lengths))
}

/// [`code_bands`] where its executor runs it.
fn code_bands_entered(
    source: &BandSource<'_>,
    planes: &CodestreamPlanes<'_>,
    layout: &ImageLayout,
    cpu: CpuCapabilities,
    output: &mut Vec<u8>,
    lengths: &mut Vec<usize>,
) -> Result<(), JxrError> {
    let rows = usize::try_from(layout.macroblocks_y)
        .map_err(|_| JxrError::arithmetic("encode macroblock rows"))?;
    let executor = cpu.executor();
    // On one thread a band is quantized and coded a macroblock row at a
    // time, so that it holds the coefficients of a single row.
    let serial = executor.threads() == 1;
    let bands = if serial {
        Vec::new()
    } else {
        pool_bands(source, planes, layout, executor)?
    };
    // The most macroblock rows read at once: a band and the rows below it
    // that its last rows wait on, through the pool.
    let tallest = bands.iter().map(ExactSizeIterator::len).max().unwrap_or(0);
    let read = source.band_rows.unwrap_or(if serial {
        READ_ROWS
    } else {
        tallest + LOOKAHEAD_ROWS
    });
    // The stages of a band run one after another on one thread: the strips
    // run each stage on the calling thread.
    let analysis_cpu = if serial { cpu.on_calling_thread() } else { cpu };
    let mut analysis = Analysis::begin(source, planes, (rows, read), analysis_cpu)?;
    let mut coding = Coding {
        tiles: TileRows::new(layout, planes, output.len(), executor)?,
        output,
        lengths,
    };
    let pipeline = Pipeline {
        planes,
        executor: analysis_cpu.executor(),
    };
    if serial {
        let rows = (0..rows).map(|row| row..row + 1);
        pipeline.run(rows, &mut analysis, &mut coding)?;
    } else {
        pipeline.run(bands.iter().cloned(), &mut analysis, &mut coding)?;
    }
    let tiles = coding.tiles;
    if tiles.open.is_empty() && tiles.next_row == layout.tile_rows.len() {
        Ok(())
    } else {
        Err(JxrError::new(
            JxrErrorKind::InternalInvariant,
            "encode tiles left unfinished",
        ))
    }
}

/// The bands of macroblock rows of an image coded through the pool.
///
/// While one band is coded the samples of the next are analyzed, which are
/// quantized into the coefficients the band coded leaves, so the samples of
/// one band and the coefficients of one are held at once, and
/// [`executor::admitted`] admits the macroblock rows of a band against the
/// working bound at what both of them take. An image whose every row is
/// admitted is one band, coded as the whole-image encoder coded it. A
/// larger one is cut at tile rows, so that no tile is coded in two bands:
/// each band as many tile rows as are admitted and as the threads have one
/// of their tiles each for, so that those coded together can keep every
/// thread busy while the rest analyze the next band, and two at least,
/// as a band ends with its slowest tile and the next band's quantization
/// waits on it. Measured on an 8000-by-6000 RGB image, whose 16 tile columns
/// leave one tile row to 18 threads, a lossless encode takes some 85 ms in
/// bands of two tile rows and 92 to 102 ms in bands of one. Where not one
/// tile row is admitted, the image is cut into bands of the rows that are,
/// all of a height.
fn pool_bands(
    source: &BandSource<'_>,
    planes: &CodestreamPlanes<'_>,
    layout: &ImageLayout,
    executor: Executor,
) -> Result<Vec<Range<usize>>, JxrError> {
    let rows = usize::try_from(layout.macroblocks_y)
        .map_err(|_| JxrError::arithmetic("encode macroblock rows"))?;
    if let Some(height) = source.band_rows {
        return Ok(cut_rows(rows, height.max(1)));
    }
    let macroblocks_x = usize::try_from(layout.macroblocks_x)
        .map_err(|_| JxrError::arithmetic("encode macroblock columns"))?;
    let components =
        planes.primary.components() + planes.alpha.map_or(0, PlaneAnalysis::components);
    let row_bytes = components
        .saturating_mul(macroblocks_x)
        .saturating_mul(MACROBLOCK_BYTES);
    let unit = u64::try_from(row_bytes).unwrap_or(u64::MAX);
    let admitted = executor::admitted(unit, rows, u64::MAX);
    if admitted >= rows {
        return Ok(cut_rows(rows, rows));
    }
    let tallest = layout
        .tile_rows
        .iter()
        .copied()
        .max()
        .map_or(1, usize::from);
    if admitted < tallest {
        // Every band of a height, as near the admitted one as divides the
        // rows evenly.
        let bands = rows.div_ceil(admitted);
        return Ok(cut_rows(rows, rows.div_ceil(bands)));
    }
    let threads = executor.threads();
    let tile_rows = (admitted / tallest).min((threads / layout.tile_columns.len().max(1)).max(2));
    let boundaries: Vec<usize> = layout
        .row_boundaries
        .iter()
        .step_by(tile_rows)
        .map(|&boundary| boundary as usize)
        .chain(core::iter::once(rows))
        .collect();
    Ok(boundaries
        .windows(2)
        .filter(|pair| pair[0] < pair[1])
        .map(|pair| pair[0]..pair[1])
        .collect())
}

/// `rows` macroblock rows cut into bands of `height` rows and a shorter
/// last one.
fn cut_rows(rows: usize, height: usize) -> Vec<Range<usize>> {
    (0..rows)
        .step_by(height)
        .map(|first| first..(first + height).min(rows))
        .collect()
}

/// The tile writers of a codestream and the output they finish tiles into.
struct Coding<'t, 'o> {
    tiles: TileRows<'t>,
    output: &'o mut Vec<u8>,
    lengths: &'o mut Vec<usize>,
}

impl Coding<'_, '_> {
    /// Code the macroblock rows `rows` of every tile, whose coefficients
    /// `coefficients` holds, primary components then alpha.
    fn code(
        &mut self,
        planes: &CodestreamPlanes<'_>,
        rows: Range<usize>,
        coefficients: &[Vec<QuantizedMacroblock>],
    ) -> Result<(), JxrError> {
        let (primary, alpha) = coefficients.split_at(planes.primary.components());
        let alpha = alpha.first().map(Vec::as_slice);
        self.tiles
            .code(rows, primary, alpha, self.output, self.lengths)
            .map_err(|error| tile_error(&error))
    }
}

/// The bands of a codestream, analyzed and coded in turn: through a pool,
/// the samples of each band are read, filtered and transformed while the
/// band above it is entropy coded, so that the threads a band's tiles leave
/// idle work on the next one; on one thread, where the bands are single
/// macroblock rows, the same steps run one after another.
struct Pipeline<'a, 'p> {
    planes: &'a CodestreamPlanes<'p>,
    /// Where a band is coded beside the analysis of the next.
    executor: Executor,
}

impl Pipeline<'_, '_> {
    /// Analyze and code every band of `bands`, each the macroblock rows
    /// that follow the one before: the samples of the next band are read,
    /// filtered and transformed while the band above it is coded, and then
    /// quantized into the coefficients that band leaves.
    fn run(
        &self,
        bands: impl Iterator<Item = Range<usize>>,
        analysis: &mut Analysis<'_>,
        coding: &mut Coding<'_, '_>,
    ) -> Result<(), JxrError> {
        let mut bands = bands.peekable();
        let Some(first) = bands.peek().cloned() else {
            return Ok(());
        };
        let mut coefficients = Vec::new();
        analysis.analyze(first, &mut coefficients)?;
        while let Some(band) = bands.next() {
            let next = bands.peek().cloned();
            let (coded, prepared) = self.executor.join(
                || coding.code(self.planes, band, &coefficients),
                || {
                    next.clone()
                        .map_or(Ok(()), |next| analysis.prepare(next.end))
                },
            );
            coded?;
            prepared?;
            if let Some(next) = next {
                analysis.quantize(next, &mut coefficients)?;
            }
        }
        Ok(())
    }
}

/// The coded components of one codestream, each analyzed down its plane
/// from the source rows read so far.
struct Analysis<'a> {
    source: &'a BandSource<'a>,
    /// The components, primary then alpha.
    components: Vec<ComponentStream>,
    /// Macroblock rows read from the source, read when the components were
    /// last advanced, and in all.
    read: usize,
    advanced: usize,
    rows: usize,
    /// Macroblock rows read at a time, or, where `None`, the rows a band
    /// needs read at once.
    read_rows: Option<usize>,
    /// Where the components are analyzed side by side.
    executor: Executor,
}

/// One coded component as it is analyzed.
struct ComponentStream {
    strip: AnalysisStrip,
    /// The plane it is read from, indexed as the reader indexes them.
    plane: usize,
    /// Sample rows of the plane read in each macroblock row.
    read_height: usize,
    /// The subsampling of chroma that is subsampled here.
    decimation: Option<Decimation>,
    /// Macroblock rows ready to quantize, and quantized.
    ready: usize,
    quantized: usize,
}

/// Chroma subsampled here from the full-resolution rows read for it.
struct Decimation {
    stream: ChromaDecimation,
    /// The rows read, after `offset` rows that align them with the planes
    /// read beside them and that are never written.
    read: Vec<i32>,
    offset: usize,
    /// Rows read and not yet subsampled.
    fresh: usize,
}

impl<'a> Analysis<'a> {
    /// Begins the analysis of every coded component of the codestream
    /// `planes` describes, `rows` macroblock rows tall, of which `read` at
    /// most are read at once, on the vectors and executor of `cpu`: on one
    /// thread, a few rows at a time, where it runs on the calling thread.
    fn begin(
        source: &'a BandSource<'a>,
        planes: &CodestreamPlanes<'_>,
        (rows, read): (usize, usize),
        cpu: CpuCapabilities,
    ) -> Result<Self, JxrError> {
        let read = read.min(rows);
        let executor = cpu.executor();
        let serial = executor == Executor::Sequential;
        let read_rows = source
            .band_rows
            .or(serial.then_some(READ_ROWS))
            .map(|count| count.clamp(1, rows.max(1)));
        let analyses = core::iter::once(planes.primary).chain(planes.alpha);
        let coded = analyses.flat_map(|analysis| {
            (0..analysis.components()).map(move |component| (analysis, component))
        });
        let mut components = Vec::with_capacity(planes.primary.components() + 1);
        for (index, (analysis, component)) in coded.enumerate() {
            let config = &analysis.configs[component];
            let mut strip = AnalysisStrip::default();
            strip
                .begin(config, analysis.steps[component], analysis.bands, read, cpu)
                .map_err(|error| reconstruction_error(&error))?;
            let plane = source.plane_of(index, planes.primary.components());
            let decimation = source
                .subsampled(plane)
                .map(|sampling| {
                    let (width, height) = source.padded;
                    let mut stream = ChromaDecimation::default();
                    stream
                        .begin([width, height], sampling)
                        .map_err(|error| reconstruction_error(&error))?;
                    // The rows read, below as many as a strip waits on.
                    let samples = (read + 2) * MACROBLOCK_SAMPLES as usize * width;
                    Ok::<_, JxrError>(Decimation {
                        stream,
                        read: Vec::with_capacity(samples),
                        offset: 0,
                        fresh: 0,
                    })
                })
                .transpose()?;
            // As the source stores the plane: subsampled where the source
            // is, and full resolution where chroma is subsampled here.
            let (_, plane_height) = source.reader.plane_dimensions(plane)?;
            let read_height = plane_height / rows.max(1);
            components.push(ComponentStream {
                strip,
                plane,
                read_height,
                decimation,
                ready: 0,
                quantized: 0,
            });
        }
        Ok(Self {
            source,
            components,
            read: 0,
            advanced: 0,
            rows,
            read_rows,
            executor,
        })
    }

    /// Quantize the macroblock rows `rows` of every component, the rows
    /// that follow those quantized before, into `coefficients`, one list of
    /// macroblocks for each component, whose earlier contents are
    /// discarded. Source rows are read as the rows need them.
    fn analyze(
        &mut self,
        rows: Range<usize>,
        coefficients: &mut Vec<Vec<QuantizedMacroblock>>,
    ) -> Result<(), JxrError> {
        self.prepare(rows.end)?;
        self.quantize(rows, coefficients)
    }

    /// Read the source and run every component's filters and transforms
    /// until its first `end` macroblock rows are ready to quantize.
    fn prepare(&mut self, end: usize) -> Result<(), JxrError> {
        let width = self.source.padded.0;
        loop {
            // The strips have nothing to run until more rows are read.
            if self.advanced < self.read {
                self.executor
                    .try_consume(self.components.iter_mut().collect(), |component| {
                        component.advance(width)
                    })?;
                self.advanced = self.read;
            }
            if self
                .components
                .iter()
                .all(|component| component.ready >= end)
            {
                return Ok(());
            }
            if self.read == self.rows {
                return Err(JxrError::new(
                    JxrErrorKind::InternalInvariant,
                    "encode rows left unfinished",
                ));
            }
            let next = match self.read_rows {
                Some(count) => self.read + count,
                None => end + LOOKAHEAD_ROWS,
            };
            self.read_rows(next.clamp(self.read + 1, self.rows))?;
        }
    }

    /// Quantize the macroblock rows `rows`, which [`Self::prepare`] has made
    /// ready, as [`Self::analyze`] does, and let them go.
    fn quantize(
        &mut self,
        rows: Range<usize>,
        coefficients: &mut Vec<Vec<QuantizedMacroblock>>,
    ) -> Result<(), JxrError> {
        coefficients.resize_with(self.components.len(), Vec::new);
        for (component, output) in self.components.iter().zip(coefficients.iter_mut()) {
            if component.quantized != rows.start {
                return Err(JxrError::new(
                    JxrErrorKind::InternalInvariant,
                    "encode rows analyzed out of order",
                ));
            }
            output.clear();
            output.reserve_exact(rows.len() * component.strip.macroblocks_x());
        }
        let tasks = self
            .components
            .iter_mut()
            .zip(coefficients.iter_mut())
            .collect();
        self.executor.try_consume(tasks, |(component, output)| {
            component.quantize(rows.end, output)
        })
    }

    /// Read the next macroblock rows of every plane from the source, each
    /// into the storage it is analyzed from, below the rows its strip
    /// holds.
    ///
    /// The planes of an interleaved source are read together, over one
    /// window: the rows the strips that take them in hold, which hold the
    /// same rows as they move down the image together, and the rows read.
    /// Chroma subsampled here is read beside them, at full resolution,
    /// below as many rows as the strips hold, which only align it.
    fn read_rows(&mut self, end: usize) -> Result<(), JxrError> {
        let reader = &self.source.reader;
        let colour = reader.colour_components();
        let mut windows = vec![0..0; colour + 1];
        let mut held = vec![0; colour + 1];
        let mut storage = SourcePlanes {
            colour: if self.source.set.colour() {
                vec![Vec::new(); colour]
            } else {
                Vec::new()
            },
            alpha: None,
        };
        let mut shared = None;
        for component in &mut self.components {
            if component.decimation.is_some() {
                continue;
            }
            let rows = component.strip.held();
            let plane = component.plane;
            windows[plane] = rows.start..end * component.read_height;
            held[plane] = rows.len();
            shared.get_or_insert((windows[plane].clone(), held[plane]));
            *slot(&mut storage, plane, colour) = component.strip.take_storage();
        }
        let first = self.read * MACROBLOCK_SAMPLES as usize;
        for component in &mut self.components {
            let Some(decimation) = &mut component.decimation else {
                continue;
            };
            let (window, offset) = shared
                .clone()
                .unwrap_or((first..end * MACROBLOCK_SAMPLES as usize, 0));
            let plane = component.plane;
            windows[plane] = window;
            held[plane] = offset;
            decimation.offset = offset;
            decimation.fresh = (end - self.read) * MACROBLOCK_SAMPLES as usize;
            *slot(&mut storage, plane, colour) = core::mem::take(&mut decimation.read);
        }
        let read = reader.read(&windows, &held, &mut storage);
        // Storage goes back to every strip whether or not the read failed.
        for component in &mut self.components {
            let samples = core::mem::take(slot(&mut storage, component.plane, colour));
            match &mut component.decimation {
                Some(decimation) => decimation.read = samples,
                None => component.strip.put_back(samples),
            }
        }
        read?;
        self.read = end;
        Ok(())
    }
}

/// The storage of plane `plane` among planes read into `storage`, of which
/// `colour` are colour planes and the one after them alpha.
fn slot(storage: &mut SourcePlanes, plane: usize, colour: usize) -> &mut Vec<i32> {
    if plane < colour {
        if storage.colour.len() <= plane {
            storage.colour.resize_with(plane + 1, Vec::new);
        }
        &mut storage.colour[plane]
    } else {
        storage.alpha.get_or_insert_with(Vec::new)
    }
}

impl ComponentStream {
    /// Subsample the chroma rows read since the last call and run the
    /// strip as far as they allow. `width` is the width of the
    /// full-resolution plane.
    fn advance(&mut self, width: usize) -> Result<(), JxrError> {
        let error = |error| reconstruction_error(&error);
        if let Some(decimation) = &mut self.decimation {
            let rows = decimation.offset..decimation.offset + decimation.fresh;
            decimation.fresh = 0;
            let read = decimation
                .read
                .get(rows.start * width..rows.end * width)
                .ok_or_else(|| {
                    JxrError::new(JxrErrorKind::InternalInvariant, "encode chroma rows read")
                })?;
            decimation.stream.push_rows(read).map_err(error)?;
            let count = decimation.stream.finished() - decimation.stream.written();
            let rows = self.strip.append(count);
            decimation.stream.write_rows(rows).map_err(error)?;
        }
        self.ready = self.strip.advance().map_err(error)?;
        Ok(())
    }

    /// Quantize into `output` the rows from those quantized before to
    /// `end`, and let them go.
    fn quantize(
        &mut self,
        end: usize,
        output: &mut Vec<QuantizedMacroblock>,
    ) -> Result<(), JxrError> {
        self.strip
            .quantize_rows(self.quantized..end, output)
            .map_err(|error| reconstruction_error(&error))?;
        self.quantized = end;
        self.strip.release(end);
        Ok(())
    }
}

impl BandSource<'_> {
    /// The plane, indexed as the reader indexes them, that coded component
    /// `index` of a codestream whose primary plane has `primary` components
    /// is read into: a colour plane, or the alpha plane for an alpha
    /// codestream and for an integrated alpha plane.
    fn plane_of(&self, index: usize, primary: usize) -> usize {
        if self.set.colour() && index < primary {
            index
        } else {
            self.reader.colour_components()
        }
    }

    /// The sampling a plane's full-resolution chroma is reduced to, for the
    /// two chroma planes of a source whose chroma is subsampled here.
    fn subsampled(&self, plane: usize) -> Option<ChromaSampling> {
        self.decimation
            .filter(|_| self.set.colour() && (1..3).contains(&plane))
    }
}

/// The tile rows a codestream's bands have reached and not yet finished.
struct TileRows<'a> {
    layout: &'a ImageLayout,
    /// Where the tiles a band reaches are coded side by side.
    executor: Executor,
    shape: (usize, bool),
    options: PacketOptions,
    macroblocks_x: usize,
    macroblocks_y: usize,
    /// Tile rows begun and not yet finished, top first.
    open: VecDeque<OpenRow>,
    /// The next tile row to begin.
    next_row: usize,
    /// Storage of finished tiles, for tiles yet to begin.
    scratch: Vec<TileScratch>,
    /// Length of the output where the codestream's packets begin, and the
    /// bytes of the packets of the tiles finished since.
    start: usize,
    finished: usize,
}

/// The writers of one tile row, left to right.
struct OpenRow {
    /// First macroblock row of the tile row, and its height.
    first: usize,
    height: usize,
    writers: Vec<TileWriter>,
}

impl<'a> TileRows<'a> {
    /// The tile rows of a codestream whose packets follow the `start`
    /// bytes the output holds.
    fn new(
        layout: &'a ImageLayout,
        planes: &CodestreamPlanes<'_>,
        start: usize,
        executor: Executor,
    ) -> Result<Self, JxrError> {
        Ok(Self {
            layout,
            executor,
            shape: (planes.primary.components(), planes.alpha.is_some()),
            options: planes.options,
            macroblocks_x: usize::try_from(layout.macroblocks_x)
                .map_err(|_| JxrError::arithmetic("encode macroblock stride"))?,
            macroblocks_y: usize::try_from(layout.macroblocks_y)
                .map_err(|_| JxrError::arithmetic("encode macroblock rows"))?,
            open: VecDeque::new(),
            next_row: 0,
            scratch: Vec::new(),
            start,
            finished: 0,
        })
    }

    /// The length the output is expected to have once every packet of the
    /// codestream is in it, after the first `coded` of its macroblock rows
    /// have been coded: what their packets took, for each of its rows, and
    /// an eighth more for the share of the rows still to code, so that an
    /// estimate a little short does not move the output once more when it
    /// is all but complete, and an exact one once every row is coded. It is
    /// estimated once a tile row is finished or an eighth of the rows are
    /// coded, and is 0 before.
    ///
    /// The output grows to it, rather than by a fraction of what it holds, so
    /// that it moves about once, early, and is not held twice over at the
    /// end: in WebAssembly every move of a growing output leaves its old
    /// storage below the new, and the linear memory never shrinks. Measured
    /// on a 4096-square RGBA image, a lossless encode grows the linear memory
    /// by 30.6 MiB where growing by eighths grew it by 62.9 MiB.
    fn expected(&self, coded: usize) -> usize {
        let rows = self.macroblocks_y;
        if self.finished == 0 && coded.saturating_mul(8) < rows {
            return 0;
        }
        let open = self.open.iter().flat_map(|row| &row.writers);
        let written = self.finished + open.map(TileWriter::written_bytes).sum::<usize>();
        let packets = written.saturating_mul(rows) / coded.max(1);
        let margin = (packets / 8).saturating_mul(rows.saturating_sub(coded)) / rows.max(1);
        self.start.saturating_add(packets).saturating_add(margin)
    }

    /// The tile the output is lent to, if it is: the first of the first
    /// tile row begun.
    fn lent(&self) -> Option<&TileWriter> {
        self.open
            .front()
            .and_then(|row| row.writers.first())
            .filter(|writer| writer.holds_output())
    }

    /// Code the macroblock rows `rows` of every tile, whose coefficients
    /// `components` and `alpha` hold, and finish the tile rows that end
    /// with them.
    fn code(
        &mut self,
        rows: Range<usize>,
        components: &[Vec<QuantizedMacroblock>],
        alpha: Option<&[QuantizedMacroblock]>,
        output: &mut Vec<u8>,
        lengths: &mut Vec<usize>,
    ) -> Result<(), TileEncodeError> {
        let Range { start: first, end } = rows;
        self.begin_rows(end, output)?;
        let lent_before = self.lent().map_or(0, TileWriter::written_bytes);
        let columns = &self.layout.column_boundaries;
        let mut tasks = Vec::new();
        for row in &mut self.open {
            let top = row.first.max(first);
            let bottom = (row.first + row.height).min(end);
            for (writer, column) in row.writers.iter_mut().zip(columns.windows(2)) {
                let region = TileRegion {
                    stride: self.macroblocks_x,
                    x: column[0] as usize,
                    y: top - first,
                    width: (column[1] - column[0]) as usize,
                    height: bottom.saturating_sub(top),
                };
                tasks.push((writer, region));
            }
        }
        self.executor
            .if_worth(tasks.len() > 1)
            .try_consume(tasks, |(writer, region)| {
                writer.push_rows(components, alpha, region)
            })?;
        let expected = self.expected(end);
        // An output a tile writes onto keeps room for as many bytes again as
        // these rows took.
        let added = self.lent().map_or(0, TileWriter::written_bytes) - lent_before;
        if let Some(writer) = self
            .open
            .front_mut()
            .and_then(|row| row.writers.first_mut())
        {
            writer.reserve_output(added, expected);
        }
        let finishing = self
            .open
            .iter()
            .take_while(|row| row.first + row.height <= end)
            .count();
        // The output grows once for the packets of every row that finishes,
        // after a tile that holds it has put it back.
        let appended = self
            .open
            .iter()
            .take(finishing)
            .flat_map(|row| &row.writers);
        let appended = appended
            .filter(|writer| !writer.holds_output())
            .map(TileWriter::written_bytes)
            .sum();
        if self.lent().is_none() {
            reserve_for(output, appended, expected);
        }
        for row in self.open.drain(..finishing) {
            for writer in row.writers {
                let held = writer.holds_output();
                let listed = lengths.len();
                self.scratch.push(writer.finish(output, lengths)?);
                self.finished += lengths[listed..].iter().sum::<usize>();
                if held {
                    reserve_for(output, appended, expected);
                }
            }
        }
        Ok(())
    }

    /// Begin every tile row that starts above macroblock row `end`.
    ///
    /// A spatial-mode tile that begins when every earlier tile is finished
    /// is the next in `output`, and is lent it to write its packet onto;
    /// a frequency-mode tile's first packet, its DC packet, is too small to
    /// be worth it.
    fn begin_rows(&mut self, end: usize, output: &mut Vec<u8>) -> Result<(), TileEncodeError> {
        let layout = self.layout;
        while self.next_row < layout.tile_rows.len()
            && (layout.row_boundaries[self.next_row] as usize) < end
        {
            let height = usize::from(layout.tile_rows[self.next_row]);
            // Only the row's first tile, and only while no earlier row is open.
            let mut next_in_output =
                self.open.is_empty() && self.options.mode == BitstreamMode::Spatial;
            let writers = layout
                .tile_columns
                .iter()
                .map(|&width| {
                    let shape = TileShape {
                        width: usize::from(width),
                        height,
                        components: self.shape.0,
                        alpha: self.shape.1,
                    };
                    let scratch = self.scratch.pop().unwrap_or_default();
                    let lent =
                        core::mem::take(&mut next_in_output).then(|| core::mem::take(output));
                    TileWriter::begin(shape, self.options, scratch, lent)
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.open.push_back(OpenRow {
                first: layout.row_boundaries[self.next_row] as usize,
                height,
                writers,
            });
            self.next_row += 1;
        }
        Ok(())
    }
}
