//! Where a decode runs the work it shares out.
//!
//! A stage that can split its work into independent tasks decides once how
//! to split it, and hands the tasks to an [`Executor`], which decides only
//! where they run: spread over the rayon thread pool, or one after another
//! on the calling thread. Both run the same tasks to the same result, so a
//! stage is written once for both, and its own size thresholds pick the
//! calling thread for work too small to be worth sharing out.
//!
//! A target that cannot have threads, WebAssembly without the `atomics`
//! feature, has no pool: [`Pool`] cannot be made there, the sequential
//! executor is the only one, and nothing of the pool is compiled into what
//! is built for it. Every other target has both, decodes through the pool,
//! and runs the sequential executor when a caller asks for it
//! ([`crate::CpuCapabilities::on_calling_thread`]): that is how the tests
//! run, on the targets tests run on, the code a thread-less target runs.

use rayon::prelude::*;

/// Where a stage runs its tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Executor {
    /// Every task on the calling thread, in order.
    Sequential,
    /// The tasks spread over the rayon thread pool.
    Pool(Pool),
}

/// The rayon thread pool, on a target that has threads.
#[cfg(not(all(target_arch = "wasm32", not(target_feature = "atomics"))))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Pool(());

/// No pool: WebAssembly without the `atomics` feature cannot start a thread,
/// so this has no value and the pool's code is never reached.
#[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pool {}

impl Pool {
    /// The pool, where the target has one.
    #[cfg(not(all(target_arch = "wasm32", not(target_feature = "atomics"))))]
    const AVAILABLE: Option<Self> = Some(Self(()));
    #[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
    const AVAILABLE: Option<Self> = None;

    /// Runs `operation`, which hands tasks to the pool; in tests, counts
    /// that the calling thread did so (`pool_entries`).
    #[inline]
    #[expect(
        clippy::unused_self,
        reason = "taking the pool shows the target has one: where it has none, none can be made"
    )]
    fn run<R>(self, operation: impl FnOnce() -> R) -> R {
        #[cfg(test)]
        POOL_ENTRIES.with(|entries| entries.set(entries.get() + 1));
        operation()
    }
}

#[cfg(test)]
thread_local! {
    static POOL_ENTRIES: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

/// How many times the calling thread has handed tasks to the pool, or asked
/// it how many threads it has: never, for a decode or encode on the calling
/// thread ([`crate::CpuCapabilities::on_calling_thread`]), whose every task
/// stays on this thread.
#[cfg(test)]
pub(crate) fn pool_entries() -> usize {
    POOL_ENTRIES.with(core::cell::Cell::get)
}

/// Bytes of working storage that the work of one decode or encode running
/// through the thread pool may hold at once: the strips a decode reads side
/// by side, or the samples and coefficients of the bands an encode holds
/// while it codes one and analyzes the next. A request's own limit lowers
/// it.
const POOL_WORKING_BYTES: u64 = 256 << 20;

/// How many units of work, each holding `unit_bytes` of working storage,
/// may be held at once in a pool: as many as hold no more than the working
/// bound between them, or `limit` if that is lower, and no more than
/// `most`, nor fewer than one.
pub(crate) fn admitted(unit_bytes: u64, most: usize, limit: u64) -> usize {
    let bound = POOL_WORKING_BYTES.min(limit);
    usize::try_from(bound / unit_bytes.max(1))
        .unwrap_or(usize::MAX)
        .clamp(1, most.max(1))
}

impl Executor {
    /// The pool where the target has threads, else the calling thread.
    pub(crate) const DEFAULT: Self = match Pool::AVAILABLE {
        Some(pool) => Self::Pool(pool),
        None => Self::Sequential,
    };

    /// This executor for work that is `worth_sharing` out, and the calling
    /// thread for work too small for a share of it to cost little beside
    /// handing it to another thread. Each stage sizes that by its own
    /// measure of its work.
    pub(crate) const fn if_worth(self, worth_sharing: bool) -> Self {
        if worth_sharing {
            self
        } else {
            Self::Sequential
        }
    }

    /// Threads the tasks may run on at once.
    pub(crate) fn threads(self) -> usize {
        match self {
            Self::Sequential => 1,
            Self::Pool(pool) => pool.run(rayon::current_num_threads),
        }
    }

    /// Runs `first` and `second`, on the calling thread in that order or
    /// side by side in the pool.
    pub(crate) fn join<A, B, RA, RB>(self, first: A, second: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        match self {
            Self::Sequential => (first(), second()),
            Self::Pool(pool) => pool.run(|| rayon::join(first, second)),
        }
    }

    /// Runs `operation` on a thread of the pool, where the joins it makes
    /// hand work to other threads without leaving the pool for each one;
    /// sequentially, on the calling thread.
    pub(crate) fn enter<R: Send>(self, operation: impl FnOnce() -> R + Send) -> R {
        match self {
            Self::Sequential => operation(),
            Self::Pool(pool) => pool.run(|| rayon::scope(|_| operation())),
        }
    }

    /// The results of `operation` on each of `items`, in order.
    pub(crate) fn map<I, R>(self, items: Vec<I>, operation: impl Fn(I) -> R + Sync + Send) -> Vec<R>
    where
        I: Send,
        R: Send,
    {
        match self {
            Self::Sequential => items.into_iter().map(operation).collect(),
            Self::Pool(pool) => pool.run(|| items.into_par_iter().map(operation).collect()),
        }
    }

    /// The results of `operation` on each of `items`, in order, or an error
    /// one of them returned; sequentially, the first, and no item after it
    /// is run.
    pub(crate) fn try_map<I, R, E>(
        self,
        items: Vec<I>,
        operation: impl Fn(I) -> Result<R, E> + Sync + Send,
    ) -> Result<Vec<R>, E>
    where
        I: Send,
        R: Send,
        E: Send,
    {
        match self {
            Self::Sequential => items.into_iter().map(operation).collect(),
            Self::Pool(pool) => pool.run(|| items.into_par_iter().map(operation).collect()),
        }
    }

    /// Runs `operation` on each of `items`, taking them, and stops at an
    /// error as [`Self::try_map`] does; it collects nothing, so the pool
    /// allocates nothing for it.
    pub(crate) fn try_consume<I, E>(
        self,
        items: Vec<I>,
        operation: impl Fn(I) -> Result<(), E> + Sync + Send,
    ) -> Result<(), E>
    where
        I: Send,
        E: Send,
    {
        match self {
            Self::Sequential => items.into_iter().try_for_each(operation),
            Self::Pool(pool) => pool.run(|| items.into_par_iter().try_for_each(operation)),
        }
    }

    /// Runs `operation` on each of `items` with its index, stopping at an
    /// error as [`Self::try_map`] does.
    pub(crate) fn try_for_each<T, E>(
        self,
        items: &mut [T],
        operation: impl Fn(usize, &mut T) -> Result<(), E> + Sync + Send,
    ) -> Result<(), E>
    where
        T: Send,
        E: Send,
    {
        match self {
            Self::Sequential => items
                .iter_mut()
                .enumerate()
                .try_for_each(|(index, item)| operation(index, item)),
            Self::Pool(pool) => pool.run(|| {
                items
                    .par_iter_mut()
                    .enumerate()
                    .try_for_each(|(index, item)| operation(index, item))
            }),
        }
    }

    /// Runs `operation` on each consecutive chunk of `size` of `items`, the
    /// last of which may be shorter, with its index, stopping at an error
    /// as [`Self::try_map`] does.
    pub(crate) fn try_for_each_chunk<T, E>(
        self,
        items: &mut [T],
        size: usize,
        operation: impl Fn(usize, &mut [T]) -> Result<(), E> + Sync + Send,
    ) -> Result<(), E>
    where
        T: Send,
        E: Send,
    {
        match self {
            Self::Sequential => items
                .chunks_mut(size)
                .enumerate()
                .try_for_each(|(index, chunk)| operation(index, chunk)),
            Self::Pool(pool) => pool.run(|| {
                items
                    .par_chunks_mut(size)
                    .enumerate()
                    .try_for_each(|(index, chunk)| operation(index, chunk))
            }),
        }
    }

    /// Combines with `combine`, starting from `identity`, the results of
    /// `operation` on each consecutive chunk of `size` of `items` with its
    /// index. `combine` must be associative, with `identity` its identity:
    /// the pool combines neighbouring results in any grouping.
    pub(crate) fn fold_chunks<T, R>(
        self,
        items: &mut [T],
        size: usize,
        identity: R,
        operation: impl Fn(usize, &mut [T]) -> R + Sync + Send,
        combine: impl Fn(R, R) -> R + Sync + Send,
    ) -> R
    where
        T: Send,
        R: Copy + Send + Sync,
    {
        match self {
            Self::Sequential => items
                .chunks_mut(size)
                .enumerate()
                .fold(identity, |total, (index, chunk)| {
                    combine(total, operation(index, chunk))
                }),
            Self::Pool(pool) => pool.run(|| {
                items
                    .par_chunks_mut(size)
                    .enumerate()
                    .map(|(index, chunk)| operation(index, chunk))
                    .reduce(|| identity, combine)
            }),
        }
    }

    /// The first, in index order, of the results `operation` gives each of
    /// `items` with its index; the pool hands out no fewer than `min_len`
    /// items to a task.
    pub(crate) fn find_map_first<T, R>(
        self,
        items: &mut [T],
        min_len: usize,
        operation: impl Fn(usize, &mut T) -> Option<R> + Sync + Send,
    ) -> Option<R>
    where
        T: Send,
        R: Send,
    {
        match self {
            Self::Sequential => items
                .iter_mut()
                .enumerate()
                .find_map(|(index, item)| operation(index, item)),
            Self::Pool(pool) => pool.run(|| {
                items
                    .par_iter_mut()
                    .with_min_len(min_len)
                    .enumerate()
                    .find_map_first(|(index, item)| operation(index, item))
            }),
        }
    }

    /// Appends to `output` the results of `operation` on `0..count`, in
    /// order; the pool hands out no fewer than `min_len` of them to a task.
    pub(crate) fn extend_with<R: Send>(
        self,
        output: &mut Vec<R>,
        count: usize,
        min_len: usize,
        operation: impl Fn(usize) -> R + Sync + Send,
    ) {
        match self {
            Self::Sequential => output.extend((0..count).map(operation)),
            Self::Pool(pool) => pool.run(|| {
                output.par_extend(
                    (0..count)
                        .into_par_iter()
                        .with_min_len(min_len)
                        .map(operation),
                );
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Executor;

    /// Both executors, where the target has both.
    fn executors() -> Vec<Executor> {
        let mut executors = vec![Executor::Sequential];
        if Executor::DEFAULT != Executor::Sequential {
            executors.push(Executor::DEFAULT);
        }
        executors
    }

    #[test]
    fn the_pool_is_the_default_where_threads_exist() {
        let threads = cfg!(not(all(
            target_arch = "wasm32",
            not(target_feature = "atomics")
        )));
        assert_eq!(Executor::DEFAULT != Executor::Sequential, threads);
        assert_eq!(Executor::Sequential.threads(), 1);
    }

    #[test]
    fn every_executor_runs_every_task_in_order() {
        for executor in executors() {
            let items: Vec<u32> = (0..1000).collect();
            assert_eq!(
                executor.map(items.clone(), |item| item * 3),
                items.iter().map(|item| item * 3).collect::<Vec<_>>(),
            );
            assert_eq!(
                executor.try_map(items.clone(), |item| Ok::<_, ()>(item + 1)),
                Ok(items.iter().map(|item| item + 1).collect::<Vec<_>>()),
            );
            let mut values = items.clone();
            executor
                .try_for_each(&mut values, |index, value| {
                    *value += u32::try_from(index).unwrap();
                    Ok::<_, ()>(())
                })
                .unwrap();
            assert_eq!(
                values,
                items.iter().map(|item| item * 2).collect::<Vec<_>>()
            );
            let mut chunks = items.clone();
            executor
                .try_for_each_chunk(&mut chunks, 7, |index, chunk| {
                    chunk.fill(u32::try_from(index).unwrap());
                    Ok::<_, ()>(())
                })
                .unwrap();
            assert_eq!(
                chunks,
                items.iter().map(|item| item / 7).collect::<Vec<_>>()
            );
            let mut bits: Vec<u32> = (0..1000).map(|item| 1 << (item % 32)).collect();
            let combined = executor.fold_chunks(
                &mut bits,
                9,
                0,
                |_, chunk| chunk.iter().fold(0, |total, bit| total | bit),
                |left, right| left | right,
            );
            assert_eq!(combined, u32::MAX);
            let sum = core::sync::atomic::AtomicU32::new(0);
            executor
                .try_consume(items.clone(), |item| {
                    sum.fetch_add(item, core::sync::atomic::Ordering::Relaxed);
                    Ok::<_, ()>(())
                })
                .unwrap();
            assert_eq!(sum.into_inner(), items.iter().sum::<u32>());
            assert_eq!(executor.join(|| 1, || 2), (1, 2));
            assert_eq!(executor.enter(|| 5), 5);
            let mut found = items.clone();
            assert_eq!(
                executor.find_map_first(&mut found, 16, |index, item| {
                    (*item % 97 == 96).then_some(index)
                }),
                Some(96),
            );
            let mut extended = vec![7];
            executor.extend_with(&mut extended, 1000, 16, |index| index * 2);
            assert_eq!(
                extended,
                core::iter::once(7)
                    .chain((0..1000).map(|index| index * 2))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn the_sequential_executor_stops_at_the_first_error() {
        let ran = core::sync::atomic::AtomicUsize::new(0);
        let result = Executor::Sequential.try_map((0..10).collect(), |item: usize| {
            ran.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            if item >= 3 { Err(item) } else { Ok(item) }
        });
        assert_eq!(result, Err(3));
        assert_eq!(ran.into_inner(), 4);
        for executor in executors() {
            let mut items = [0_u8; 64];
            let error = executor.try_for_each_chunk(&mut items, 8, |index, _| {
                if index == 5 { Err(index) } else { Ok(()) }
            });
            assert_eq!(error, Err(5));
        }
    }
}
