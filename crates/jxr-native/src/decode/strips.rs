//! The strips a full-resolution CPU decode is divided into, and the threads
//! and memory they are given.
//!
//! One thread decodes the image as one strip, top to bottom. With a thread
//! pool the tile rows of the image are shared out among several strips, each
//! of which is the same stream over its own rows and stores into its own
//! part of the output. Where rows on either side of a boundary between two
//! strips depend on each other, through an overlap filter or the vertical
//! interpolation of chroma, neither strip stores the macroblock row next to
//! the boundary: each keeps the coefficients of its two rows nearest it,
//! and once both are done a third strip of those four rows stores the two
//! in the middle. Every strip therefore takes in one macroblock row beyond
//! the rows it stores wherever a neighbour's rows reach into them, which is
//! all that the rows it stores can depend on.

use core::ops::Range;

use jxr_core::{
    ChromaSampling, DecodeRequest, DecodeScale, DecodedSamplesMut, JxrError, OutputFormatRequest,
    PreparedPlan,
};

use crate::{
    CpuCapabilities, ParsedCodestream,
    executor::{self, Executor},
    metadata::{internal_subsampling, interpolates_chroma},
    reconstruct::ReconstructionSimd,
};

use super::{
    output::OutputRows,
    reconstruction::Window,
    stream::{
        self, AlphaRows, Codestream, KeptBand, PlaneStream, Stage, StreamWorkspace, StripRows,
        Threads,
    },
};

/// The storage one strip runs in, retained between decodes.
#[derive(Debug, Default)]
pub(super) struct StripWorkspace {
    pub(super) primary: StreamWorkspace,
    pub(super) alpha: StreamWorkspace,
}

/// What a decode is of: its primary codestream and a separate alpha one.
#[derive(Clone, Copy)]
pub(super) struct Images<'a> {
    pub(super) primary: Codestream<'a>,
    /// A separate alpha image whose rows the output stores.
    pub(super) alpha: Option<Codestream<'a>>,
    /// Whether the output stores the alpha plane of the primary codestream.
    pub(super) integrated_alpha: bool,
}

/// A failure of one stage of a decode, with the stage for its precedence.
pub(super) type Failure = Option<(Stage, JxrError)>;

/// What the strips of a full-resolution decode completed and reused.
pub(super) struct DecodeOutcome {
    pub(super) failure: Failure,
    pub(super) coefficient_reuses: u64,
    pub(super) simd: ReconstructionSimd,
}

/// The earlier of two failures in the order of [`Stage`].
fn earlier(held: Failure, other: Failure) -> Failure {
    match (held, other) {
        (Some(held), Some(other)) => Some(if other.0 < held.0 { other } else { held }),
        (held, other) => held.or(other),
    }
}

/// How the rows of the image are shared out.
struct Layout {
    /// The strips that read packets, top to bottom.
    strips: Vec<StripRows>,
    /// The strip that joins each of them to the next, where rows across
    /// the boundary depend on each other.
    joins: Vec<StripRows>,
    threads: Threads,
}

/// The tile rows that hold a tile `plan` selects: the macroblock row of
/// the image the first begins at, and the macroblock rows of each.
fn selected_tile_rows(plan: &PreparedPlan) -> (u32, Vec<u32>) {
    let columns = plan.info.tiles.column_widths.len().max(1);
    let mut first = None;
    let mut top = 0_u32;
    let mut heights = Vec::new();
    for (tiles, &height) in plan.tiles.chunks(columns).zip(&plan.info.tiles.row_heights) {
        if tiles.iter().any(|tile| tile.required_for_reconstruction) {
            first.get_or_insert(top);
            heights.push(height);
        }
        top = top.saturating_add(height);
    }
    (first.unwrap_or(0), heights)
}

/// Whether no row of an image depends on a row of another tile row.
fn tile_rows_are_independent(parsed: &ParsedCodestream, plan: &PreparedPlan) -> bool {
    let filtered_across =
        plan.primary.overlap != jxr_core::OverlapMode::None && !plan.info.tiles.hard_tiles;
    // Chroma at half height is interpolated from the rows above and below.
    let internal = parsed.headers.primary.internal_color_format;
    let interpolated_across = internal_subsampling(internal) == Some(ChromaSampling::Cs420)
        && interpolates_chroma(
            internal,
            parsed.headers.image.output_color_format,
            DecodeScale::Full,
        );
    !filtered_across && !interpolated_across
}

impl Layout {
    /// Shares the tile rows of `images` out among as many strips as the
    /// threads of `executor` and `limits` admit.
    fn plan(images: Images<'_>, request: &DecodeRequest, executor: Executor) -> Self {
        let plan = images.primary.plan;
        let selection = selected_tile_rows(plan);
        let tile_rows = &selection.1;
        let rows: usize = tile_rows.iter().map(|&height| height as usize).sum();
        // The tiles selected are the same columns of every selected row.
        let selected: Vec<usize> = plan
            .tiles
            .chunks(plan.info.tiles.column_widths.len().max(1))
            .find(|tiles| tiles.iter().any(|tile| tile.required_for_reconstruction))
            .map(|tiles| {
                tiles
                    .iter()
                    .zip(&plan.info.tiles.column_widths)
                    .filter(|(tile, _)| tile.required_for_reconstruction)
                    .map(|(_, &width)| width as usize)
                    .collect()
            })
            .unwrap_or_default();
        let columns: usize = selected.iter().sum();
        let whole = |threads| Self {
            strips: vec![StripRows::whole(tile_rows.len(), rows)],
            joins: Vec::new(),
            threads,
        };
        let pool = executor.threads();
        if pool <= 1 {
            return whole(Threads::caller(columns));
        }
        let tile_columns = selected.len();
        let threads = Threads::pool(executor, columns, tile_columns);
        // A separate alpha image is divided with the primary one, so it
        // must be tiled down the image as the primary one is.
        let alpha_matches = images
            .alpha
            .is_none_or(|alpha| selected_tile_rows(alpha.plan) == selection);
        if tile_rows.len() < 2 || !alpha_matches {
            return whole(threads);
        }
        let independent = tile_rows_are_independent(images.primary.parsed, plan)
            && images
                .alpha
                .is_none_or(|alpha| tile_rows_are_independent(alpha.parsed, alpha.plan));
        let admitted = admitted_strips(images, request, rows, threads.step, pool);
        let (strips, joins) = share_tile_rows(tile_rows, admitted, independent);
        if strips.len() < 2 {
            return whole(threads);
        }
        Self {
            threads: threads.strips(strips.len() * 2 <= pool),
            strips,
            joins,
        }
    }
}

/// How many strips may run at once: as many as the thread pool has threads
/// for, and as [`executor::admitted`] admits against the working bound and
/// the request's coefficient limit.
///
/// A strip holds the coefficients of the band it reconstructs, of the band
/// read ahead of it and of the four rows it keeps for its neighbours, and
/// the samples of a band and of the few rows that wait on the next. The
/// samples of a macroblock row take about what its coefficients do.
fn admitted_strips(
    images: Images<'_>,
    request: &DecodeRequest,
    rows: usize,
    step: usize,
    pool: usize,
) -> usize {
    let coefficient_bytes = images.primary.plan.coefficient_bytes
        + images.alpha.map_or(0, |alpha| alpha.plan.coefficient_bytes);
    let row_bytes = (coefficient_bytes / rows.max(1)).max(1) as u64;
    let strip_bytes = row_bytes.saturating_mul(3 * step as u64 + 7);
    executor::admitted(strip_bytes, pool, request.limits.max_coefficient_bytes)
}

/// Shares tile rows of the heights `tile_rows` out among at most `strips`
/// strips of about as many macroblock rows each.
///
/// Unless tile rows are `independent`, a strip stores no row next to a
/// boundary with another, keeps the two rows nearest each boundary, and is
/// at least four rows tall so that it keeps different rows for each; and
/// a joining strip stores the two rows either side of each boundary.
fn share_tile_rows(
    tile_rows: &[u32],
    strips: usize,
    independent: bool,
) -> (Vec<StripRows>, Vec<StripRows>) {
    let rows: usize = tile_rows.iter().map(|&height| height as usize).sum();
    let minimum = if independent { 1 } else { 4 };
    let target = rows.div_ceil(strips.max(1)).max(minimum);
    // The tile rows and macroblock rows of each strip.
    let mut groups: Vec<(Range<usize>, Range<usize>)> = Vec::new();
    let (mut first_tile_row, mut first_row, mut row) = (0, 0, 0);
    for (tile_row, &height) in tile_rows.iter().enumerate() {
        row += height as usize;
        let last = tile_row + 1 == tile_rows.len();
        if last || (row - first_row >= target && rows - row >= minimum) {
            groups.push((first_tile_row..tile_row + 1, first_row..row));
            (first_tile_row, first_row) = (tile_row + 1, row);
        }
    }
    let count = groups.len();
    let strips = groups
        .iter()
        .enumerate()
        .map(|(index, (tile_rows, taken))| {
            let keep = [index > 0 && !independent, index + 1 < count && !independent];
            StripRows {
                tile_rows: tile_rows.clone(),
                taken: taken.clone(),
                stored: taken.start + usize::from(keep[0])..taken.end - usize::from(keep[1]),
                keep,
            }
        })
        .collect();
    let joins = if independent {
        Vec::new()
    } else {
        groups[..count - 1]
            .iter()
            .map(|(_, taken)| StripRows {
                tile_rows: 0..0,
                taken: taken.end - 2..taken.end + 2,
                stored: taken.end - 1..taken.end + 1,
                keep: [false; 2],
            })
            .collect()
    };
    (strips, joins)
}

/// What one strip leaves behind.
struct StripOutcome {
    failure: Failure,
    coefficient_reuses: u64,
    simd: ReconstructionSimd,
    /// The bands the primary stream kept, then those the alpha stream did.
    kept: [Vec<KeptBand>; 2],
}

/// The bands of four rows a joining strip runs over: the last two rows of
/// the strip above the boundary, then the first two of the strip below it.
fn joined_bands(
    above: &mut Vec<KeptBand>,
    below: &mut Vec<KeptBand>,
    boundary: usize,
) -> Vec<KeptBand> {
    let (bottom, rest): (Vec<_>, Vec<_>) = core::mem::take(above)
        .into_iter()
        .partition(|band| band.rows.start + 2 >= boundary);
    *above = rest;
    let (top, rest): (Vec<_>, Vec<_>) = core::mem::take(below)
        .into_iter()
        .partition(|band| band.rows.start < boundary + 2);
    *below = rest;
    bottom.into_iter().chain(top).collect()
}

/// Decodes `images` into `destination`, which holds the whole output
/// `output` describes, or reads and reconstructs them for their errors
/// alone when the output could not be prepared.
///
/// `held` is a stage that failed before any strip began. Returns the
/// failure a decode that ran each stage over the whole image in turn
/// would have reported: the earliest stage that failed anywhere.
pub(super) fn decode(
    images: Images<'_>,
    request: &DecodeRequest,
    cpu: CpuCapabilities,
    output: Option<(OutputFormatRequest, DecodedSamplesMut<'_>)>,
    workspaces: &mut Vec<StripWorkspace>,
    mut held: Failure,
) -> DecodeOutcome {
    let layout = Layout::plan(images, request, cpu.executor());
    let mut outputs: Vec<Option<OutputRows<'_>>> = match output.filter(|_| held.is_none()) {
        Some((request, destination)) => {
            match layout.split_output(images, request, cpu, destination) {
                Ok(outputs) => outputs.into_iter().map(Some).collect(),
                Err(error) => {
                    held = Some((Stage::Output, error));
                    Vec::new()
                }
            }
        }
        None => Vec::new(),
    };
    outputs.resize_with(layout.strips.len() + layout.joins.len(), || None);
    if workspaces.len() < layout.strips.len() {
        workspaces.resize_with(layout.strips.len(), StripWorkspace::default);
    }
    let (strip_outputs, join_outputs) = outputs.split_at_mut(layout.strips.len());
    let run = |((rows, workspace), output): (
        (&StripRows, &mut StripWorkspace),
        &mut Option<OutputRows<'_>>,
    )| {
        let threads = layout.threads;
        let held = held.clone();
        run_strip(
            images,
            rows,
            None,
            cpu,
            threads,
            workspace,
            output.as_mut(),
            held,
        )
    };
    let mut strips = layout
        .strips
        .iter()
        .zip(workspaces.iter_mut())
        .zip(strip_outputs);
    let executor = layout.threads.executor;
    if executor == Executor::Sequential {
        let outcome = run(strips.next().expect("a layout has one strip"));
        return DecodeOutcome {
            failure: outcome.failure,
            coefficient_reuses: outcome.coefficient_reuses,
            simd: outcome.simd,
        };
    }
    let mut outcomes: Vec<StripOutcome> = if layout.strips.len() == 1 {
        if layout.threads.read_ahead {
            // On a thread of the pool, which hands the band it reads ahead
            // to another without leaving the pool for each one.
            executor.enter(|| strips.map(run).collect())
        } else {
            strips.map(run).collect()
        }
    } else {
        executor.map(strips.collect(), run)
    };
    let failure = outcomes.iter_mut().fold(held, |failure, outcome| {
        earlier(failure, outcome.failure.take())
    });
    let coefficient_reuses = outcomes
        .iter()
        .map(|outcome| outcome.coefficient_reuses)
        .fold(0_u64, u64::saturating_add);
    let simd = outcomes
        .iter()
        .fold(ReconstructionSimd::default(), |mut simd, outcome| {
            simd.include(outcome.simd);
            simd
        });
    // A join reconstructs and stores: nothing it can report comes before a
    // failure of the packets or of reconstruction.
    let settled = failure
        .as_ref()
        .is_some_and(|(stage, _)| *stage <= Stage::Reconstruction);
    if layout.joins.is_empty() || settled {
        return DecodeOutcome {
            failure,
            coefficient_reuses,
            simd,
        };
    }
    let (joined, joined_simd) = run_joins(
        &layout,
        images,
        cpu,
        workspaces,
        join_outputs,
        &mut outcomes,
        &failure,
    );
    let mut simd = simd;
    simd.include(joined_simd);
    DecodeOutcome {
        failure: earlier(failure, joined),
        coefficient_reuses,
        simd,
    }
}

/// Runs the strips that join each strip to the next, from the bands those
/// kept, and returns the earliest failure among them.
fn run_joins(
    layout: &Layout,
    images: Images<'_>,
    cpu: CpuCapabilities,
    workspaces: &mut [StripWorkspace],
    outputs: &mut [Option<OutputRows<'_>>],
    outcomes: &mut [StripOutcome],
    held: &Failure,
) -> (Failure, ReconstructionSimd) {
    let bands: Vec<[Vec<KeptBand>; 2]> = layout
        .joins
        .iter()
        .enumerate()
        .map(|(index, rows)| {
            let (above, below) = outcomes.split_at_mut(index + 1);
            let boundary = rows.taken.start + 2;
            [0, 1].map(|image| {
                joined_bands(
                    &mut above[index].kept[image],
                    &mut below[0].kept[image],
                    boundary,
                )
            })
        })
        .collect();
    // Each join runs in the storage of the strip above it, which is done.
    let joins = layout
        .joins
        .iter()
        .zip(&bands)
        .zip(workspaces.iter_mut())
        .zip(outputs.iter_mut())
        .collect();
    let joined = layout
        .threads
        .executor
        .map(joins, |(((rows, bands), workspace), output)| {
            let threads = layout.threads;
            let held = held.clone();
            let bands = Some(bands);
            let outcome = run_strip(
                images,
                rows,
                bands,
                cpu,
                threads,
                workspace,
                output.as_mut(),
                held,
            );
            (outcome.failure, outcome.simd)
        });
    joined.into_iter().fold(
        (None, ReconstructionSimd::default()),
        |(failure, mut simd), (joined_failure, joined_simd)| {
            simd.include(joined_simd);
            (earlier(failure, joined_failure), simd)
        },
    )
}

impl Layout {
    /// Divides `destination`, which holds the whole output `request`
    /// describes, among the strips and joins: an output for each strip, in
    /// order, then one for each join.
    fn split_output<'d>(
        &self,
        images: Images<'_>,
        request: OutputFormatRequest,
        cpu: CpuCapabilities,
        destination: DecodedSamplesMut<'d>,
    ) -> Result<Vec<OutputRows<'d>>, JxrError> {
        let has_alpha = images.alpha.is_some() || images.integrated_alpha;
        let origin = Window::of_plan(images.primary.plan)?.origin_y;
        // Strips and the joins between them alternate down the image.
        let mut order: Vec<&StripRows> = self.strips.iter().chain(&self.joins).collect();
        order.sort_by_key(|rows| rows.stored.start);
        let cuts: Vec<u32> = order[..order.len() - 1]
            .iter()
            .map(|rows| (origin + u32::try_from(rows.stored.end).unwrap_or(u32::MAX)) * 16)
            .collect();
        let split = OutputRows::split(request, has_alpha, cpu, destination, &cuts)?;
        if self.joins.is_empty() {
            return Ok(split);
        }
        let (mut strips, mut joins) = (Vec::new(), Vec::new());
        for (index, output) in split.into_iter().enumerate() {
            if index % 2 == 0 {
                strips.push(output);
            } else {
                joins.push(output);
            }
        }
        strips.append(&mut joins);
        Ok(strips)
    }
}

/// Runs one strip to its end: the primary stream over `rows`, a separate
/// alpha stream beside it, and the storing of the rows they finish.
///
/// `kept` gives a joining strip the bands it runs over in place of packets,
/// the primary image's and then the alpha image's.
#[expect(
    clippy::too_many_arguments,
    reason = "one strip's share of everything a decode is given"
)]
fn run_strip(
    images: Images<'_>,
    rows: &StripRows,
    kept: Option<&[Vec<KeptBand>; 2]>,
    cpu: CpuCapabilities,
    threads: Threads,
    workspace: &mut StripWorkspace,
    output: Option<&mut OutputRows<'_>>,
    held: Failure,
) -> StripOutcome {
    let mut outcome = StripOutcome {
        failure: held,
        coefficient_reuses: 0,
        simd: ReconstructionSimd::default(),
        kept: [Vec::new(), Vec::new()],
    };
    let StripWorkspace { primary, alpha } = workspace;
    let stream = match kept {
        Some(bands) => PlaneStream::joining(images.primary, rows.clone(), &bands[0], cpu, primary),
        None => PlaneStream::new(images.primary, rows.clone(), cpu, threads, primary),
    };
    let mut primary = match stream {
        Ok(stream) => stream,
        Err((Stage::Entropy, error)) => {
            outcome.failure = Some((Stage::Entropy, error));
            return outcome;
        }
        Err((_, error)) => {
            // Reconstruction that cannot begin still comes after every packet.
            let read = stream::read_packets(images.primary, rows.tile_rows.clone());
            outcome.failure = Some(match read {
                Ok(()) => (Stage::Reconstruction, error),
                Err(error) => (Stage::Entropy, error),
            });
            return outcome;
        }
    };
    let mut alpha_stream = None;
    let alpha_wanted = outcome
        .failure
        .as_ref()
        .is_none_or(|(stage, _)| *stage > Stage::AlphaEntropy);
    if let Some(codestream) = images.alpha.filter(|_| alpha_wanted) {
        // A separate alpha image tiled down the image as the primary one is
        // shares its strips; any other is one strip of its own tiles.
        let selection = selected_tile_rows(codestream.plan);
        let alpha_rows = if selected_tile_rows(images.primary.plan) == selection {
            rows.clone()
        } else {
            let total = selection.1.iter().map(|&height| height as usize).sum();
            StripRows::whole(selection.1.len(), total)
        };
        let alpha_tile_rows = alpha_rows.tile_rows.clone();
        let stream = match kept {
            Some(bands) => PlaneStream::joining(codestream, alpha_rows, &bands[1], cpu, alpha),
            None => PlaneStream::new(codestream, alpha_rows.clone(), cpu, threads, alpha),
        };
        match stream {
            Ok(stream) => alpha_stream = Some(stream),
            Err((Stage::Entropy, error)) => {
                outcome.failure = earlier(outcome.failure, Some((Stage::AlphaEntropy, error)));
            }
            Err((_, error)) => {
                let read = stream::read_packets(codestream, alpha_tile_rows);
                let failure = match read {
                    Ok(()) => (Stage::AlphaReconstruction, error),
                    Err(error) => (Stage::AlphaEntropy, error),
                };
                outcome.failure = earlier(outcome.failure, Some(failure));
            }
        }
    }
    let alpha_rows = match (&mut alpha_stream, images.integrated_alpha) {
        (Some(stream), _) => AlphaRows::Separate(stream),
        (None, true) => AlphaRows::Integrated,
        (None, false) => AlphaRows::None,
    };
    outcome.failure = stream::run(&mut primary, alpha_rows, output, outcome.failure.take());
    outcome.simd.include(primary.simd());
    if let Some(stream) = alpha_stream.as_ref() {
        outcome.simd.include(stream.simd());
    }
    outcome.coefficient_reuses = u64::from(primary.coefficient_reused()).saturating_add(u64::from(
        alpha_stream
            .as_ref()
            .is_some_and(PlaneStream::coefficient_reused),
    ));
    outcome.kept = [
        primary.take_kept(),
        alpha_stream.map_or_else(Vec::new, |mut stream| stream.take_kept()),
    ];
    outcome
}

#[cfg(test)]
mod tests {
    use super::share_tile_rows;

    /// The rows every strip and join stores, in order down the image.
    fn stored(tile_rows: &[u32], strips: usize, independent: bool) -> Vec<(usize, usize)> {
        let (strips, joins) = share_tile_rows(tile_rows, strips, independent);
        let mut stored: Vec<_> = strips
            .iter()
            .chain(&joins)
            .map(|rows| (rows.stored.start, rows.stored.end))
            .collect();
        stored.sort_unstable();
        stored
    }

    #[test]
    fn strips_and_joins_store_every_row_once() {
        let layouts: [&[u32]; 5] = [
            &[8, 8, 8, 8, 5],
            &[1; 23],
            &[3, 1, 9, 2, 2, 6],
            &[40],
            &[2, 2],
        ];
        for tile_rows in layouts {
            let rows: usize = tile_rows.iter().map(|&height| height as usize).sum();
            for strips in 1..=9 {
                for independent in [false, true] {
                    let stored = stored(tile_rows, strips, independent);
                    let case = format!("{tile_rows:?} {strips} strips independent {independent}");
                    assert_eq!(stored.first().unwrap().0, 0, "{case}");
                    assert_eq!(stored.last().unwrap().1, rows, "{case}");
                    assert!(
                        stored.windows(2).all(|pair| pair[0].1 == pair[1].0),
                        "{case}: {stored:?}"
                    );
                    assert!(stored.iter().all(|rows| rows.0 < rows.1), "{case}");
                }
            }
        }
    }

    #[test]
    fn a_strip_keeps_two_rows_for_each_join_and_stores_none_beside_one() {
        let (strips, joins) = share_tile_rows(&[8, 8, 8], 3, false);
        let taken: Vec<_> = strips.iter().map(|rows| rows.taken.clone()).collect();
        assert_eq!(taken, [0..8, 8..16, 16..24]);
        let stored: Vec<_> = strips.iter().map(|rows| rows.stored.clone()).collect();
        assert_eq!(stored, [0..7, 9..15, 17..24]);
        let keep: Vec<_> = strips.iter().map(|rows| rows.keep).collect();
        assert_eq!(keep, [[false, true], [true, true], [true, false]]);
        // A join takes in the two rows either side of the boundary and
        // stores the one next to it on each side.
        let joined: Vec<_> = joins
            .iter()
            .map(|rows| (rows.taken.clone(), rows.stored.clone()))
            .collect();
        assert_eq!(joined, [(6..10, 7..9), (14..18, 15..17)]);

        // Independent tile rows need neither.
        let (strips, joins) = share_tile_rows(&[8, 8, 8], 3, true);
        assert!(joins.is_empty());
        assert!(strips.iter().all(|rows| rows.stored == rows.taken));
    }

    #[test]
    fn strips_are_no_more_than_asked_for_and_at_least_four_rows_tall() {
        let tile_rows = [1_u32; 23];
        for strips in 1..=12 {
            let (shared, _) = share_tile_rows(&tile_rows, strips, false);
            assert!(shared.len() <= strips, "{strips}");
            assert!(shared.iter().all(|rows| rows.taken.len() >= 4), "{strips}");
        }
        // Tile rows that depend on nothing may be shared one to a strip.
        assert_eq!(share_tile_rows(&tile_rows, 30, true).0.len(), 23);
    }
}
