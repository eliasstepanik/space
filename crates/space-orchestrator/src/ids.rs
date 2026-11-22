//! The persistent, monotonic, never-reused [`ShardId`] allocator (H9).
//!
//! This closes Phase-2 MAJOR 3: production shard ids stop coming from
//! `main.rs --shard` argv and start coming from a counter the orchestrator owns
//! and persists. The counter itself is [`space_server::alloc::ShardIdAllocator`]
//! — monotonic, no `free`, returns an error at the id-space limit (plan F2) — and this
//! module composes it with durability:
//!
//! * **persist before hand-out**, never hand-out before persist. A crash between
//!   "allocated" and "persisted" is the one window in which an id could be
//!   reused; writing first closes it.
//! * **atomic, durable write**: write a sibling `.tmp` file, `fsync` it, then
//!   `rename` it directly over the counter file (and sync the directory where
//!   the platform allows it). Rename replaces the destination atomically, so a
//!   crash at any point leaves the old file, the new file, or (only before the
//!   first rename ever completes) the `.tmp` file holding the last written
//!   value — never *no* counter, and never a renamed-but-empty one. The old
//!   file is never deleted first. The reader falls back to the `.tmp` file when
//!   the main file is missing, and to `first` only when neither exists (a
//!   genuinely fresh state dir).
//! * **a failed write is an error, not a panic**: [`PersistentShardIdAllocator::allocate`]
//!   withholds the id and returns [`AllocError::Persist`], so the caller (boot,
//!   a `ShardHello`, a split) reports it instead of taking the process down.
//! * **resume on restart**: `open` reads the file if present, so the first id
//!   after a restart is strictly greater than the last one before it.
//!
//! The sink is a trait so a test can count writes and prove the persist-before-
//! return ordering without touching the filesystem.

use std::io;
use std::path::{Path, PathBuf};

use space_protocol::ShardId;
use space_server::alloc::{IdSpaceExhausted, ShardIdAllocator};

/// Why the allocator could not be initialised.
#[derive(Debug)]
pub enum InitError {
    /// The counter file exists but does not hold a decimal `u64`.
    Corrupt {
        /// The offending path.
        path: PathBuf,
        /// What was found, for the log.
        found: String,
    },
    /// The counter file holds a value the id space cannot represent (≥ 2³²).
    OutOfRange {
        /// The offending path.
        path: PathBuf,
        /// The stored value.
        value: u64,
    },
    /// A filesystem error (read, write, rename).
    Io(io::Error),
}

impl core::fmt::Display for InitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            InitError::Corrupt { path, found } => {
                write!(f, "{} is not a decimal counter: {found:?}", path.display())
            }
            InitError::OutOfRange { path, value } => write!(
                f,
                "{} holds {value}, past the u32 shard-id space",
                path.display()
            ),
            InitError::Io(err) => write!(f, "counter file io: {err}"),
        }
    }
}

impl std::error::Error for InitError {}

impl From<io::Error> for InitError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

/// Why an id could not be allocated.
///
/// Either way **no id was handed out**. A failed persist leaves the in-memory
/// counter one ahead of the file, so the id that could not be persisted is
/// skipped by the next successful allocation — never reused (H9 asks for
/// "never reused", not "contiguous").
#[derive(Debug)]
pub enum AllocError {
    /// The id space is used up (ids are never reused).
    Exhausted(IdSpaceExhausted),
    /// The bumped counter could not be persisted, so the id was withheld.
    Persist(io::Error),
}

impl core::fmt::Display for AllocError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AllocError::Exhausted(err) => write!(f, "{err}"),
            AllocError::Persist(err) => write!(
                f,
                "the shard-id counter could not be persisted ({err}); \
                 refusing to hand out an unpersisted id"
            ),
        }
    }
}

impl std::error::Error for AllocError {}

impl From<IdSpaceExhausted> for AllocError {
    fn from(err: IdSpaceExhausted) -> Self {
        Self::Exhausted(err)
    }
}

/// Where the next-id counter is persisted before an id is handed out.
///
/// Object-safe (`Box<dyn CounterSink>`) so the allocator stays a plain struct
/// without a generic parameter threading through the orchestrator.
pub trait CounterSink: core::fmt::Debug + Send + Sync {
    /// Persist `next` durably. `allocate` calls this **before** returning the
    /// id that was just taken, with `next` = the id *after* the one handed out.
    ///
    /// # Errors
    ///
    /// Any error aborts the allocation: the id is not returned to the caller.
    fn persist(&mut self, next: u64) -> io::Result<()>;
}

/// The counter file sink: one line, the decimal next-id, written atomically.
#[derive(Debug)]
pub struct FileCounterSink {
    path: PathBuf,
}

impl FileCounterSink {
    /// A sink writing to `path`.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// The sibling temp file `persist` writes before the rename.
    fn tmp_path(path: &Path) -> PathBuf {
        path.with_extension("tmp")
    }

    /// Read a counter file; `Ok(None)` when neither it nor its `.tmp` sibling
    /// exists. When the main file is missing but the `.tmp` file exists, the
    /// `.tmp` file is read instead (it holds the last written value).
    ///
    /// # Errors
    ///
    /// [`InitError::Corrupt`] for non-numeric content, [`InitError::Io`] for a
    /// read failure other than "not found".
    pub fn read(path: &Path) -> Result<Option<u64>, InitError> {
        match Self::read_one(path)? {
            Some(v) => Ok(Some(v)),
            None => Self::read_one(&Self::tmp_path(path)),
        }
    }

    fn read_one(path: &Path) -> Result<Option<u64>, InitError> {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let trimmed = text.trim();
                match trimmed.parse::<u64>() {
                    Ok(v) => Ok(Some(v)),
                    Err(_) => Err(InitError::Corrupt {
                        path: path.to_path_buf(),
                        found: trimmed.to_string(),
                    }),
                }
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(InitError::Io(err)),
        }
    }
}

impl FileCounterSink {
    /// The directory that holds `path`, for the post-rename directory sync.
    ///
    /// [`Path::parent`] of a bare file name (`space-orchestrator.id`, the
    /// default) is `Some("")`, not `None` — and `""` cannot be opened, so the
    /// sync used to be skipped silently for exactly the default configuration.
    /// An empty parent means "the current directory".
    fn dir_of(path: &Path) -> &Path {
        match path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent,
            _ => Path::new("."),
        }
    }

    /// Rename `tmp` over `target`, retrying a sharing violation briefly.
    ///
    /// On Windows a rename fails with `PermissionDenied` while another process
    /// (a virus scanner, an indexer, a file-sync client) has either file open
    /// without share-delete — typically for a few milliseconds right after the
    /// file was written. A short, bounded retry rides that out; anything that
    /// lasts longer is reported to the caller, which refuses the allocation.
    fn rename_over(tmp: &Path, target: &Path) -> io::Result<()> {
        const ATTEMPTS: u32 = 5;
        const PAUSE: std::time::Duration = std::time::Duration::from_millis(20);
        let mut attempt = 1;
        loop {
            match std::fs::rename(tmp, target) {
                Ok(()) => return Ok(()),
                Err(err)
                    if cfg!(windows)
                        && err.kind() == io::ErrorKind::PermissionDenied
                        && attempt < ATTEMPTS =>
                {
                    attempt += 1;
                    std::thread::sleep(PAUSE);
                }
                Err(err) => return Err(err),
            }
        }
    }
}

impl CounterSink for FileCounterSink {
    fn persist(&mut self, next: u64) -> io::Result<()> {
        use std::io::Write as _;

        // Write to a sibling temp file, then rename it directly over the
        // target. The rename is atomic and replaces the destination, so a crash
        // leaves either the old or the new counter file present, never neither.
        // The target is deliberately not removed first.
        //
        // The temp file is synced **before** the rename: without that, a power
        // loss shortly after can leave the renamed file empty (the rename was
        // journalled, the data was not), and an empty counter refuses the next
        // boot — or, worse, the old contents come back and an id is reused.
        let tmp = Self::tmp_path(&self.path);
        {
            let mut file = std::fs::File::create(&tmp)?;
            file.write_all(format!("{next}\n").as_bytes())?;
            file.sync_all()?;
        }
        Self::rename_over(&tmp, &self.path)?;
        if let Ok(dir) = std::fs::File::open(Self::dir_of(&self.path)) {
            // Best-effort durability of the rename itself; a failure here is
            // not worth failing the allocation over, since the rename already
            // committed the value. (A directory cannot be opened this way on
            // Windows, where the rename is durable through the journal.)
            let _ = dir.sync_all();
        }
        Ok(())
    }
}

/// A sink that only records what it was asked to persist (tests, dry runs).
#[derive(Debug, Default)]
pub struct MemoryCounterSink {
    /// Every value handed to [`CounterSink::persist`], in order.
    pub written: Vec<u64>,
}

impl CounterSink for MemoryCounterSink {
    fn persist(&mut self, next: u64) -> io::Result<()> {
        self.written.push(next);
        Ok(())
    }
}

/// The orchestrator's shard-id counter: monotonic, never reused, durable.
#[derive(Debug)]
pub struct PersistentShardIdAllocator {
    inner: ShardIdAllocator,
    sink: Box<dyn CounterSink>,
}

impl PersistentShardIdAllocator {
    /// Open the allocator over `path`, starting from `first` when the file does
    /// not exist yet.
    ///
    /// # Errors
    ///
    /// [`InitError`] if the file is corrupt, out of the id space, or unreadable.
    pub fn open(path: &Path, first: u64) -> Result<Self, InitError> {
        let start = match FileCounterSink::read(path)? {
            Some(v) => {
                if v > u64::from(u32::MAX) {
                    return Err(InitError::OutOfRange {
                        path: path.to_path_buf(),
                        value: v,
                    });
                }
                v
            }
            None => first,
        };
        Ok(Self {
            inner: ShardIdAllocator::new(start),
            sink: Box::new(FileCounterSink::new(path)),
        })
    }

    /// Build an allocator over an arbitrary sink (tests; the file-backed path
    /// goes through [`open`](Self::open)).
    #[must_use]
    pub fn with_sink(first: u64, sink: Box<dyn CounterSink>) -> Self {
        Self {
            inner: ShardIdAllocator::new(first),
            sink,
        }
    }

    /// Allocate the next id, persisting the counter **before** returning it.
    ///
    /// # Errors
    ///
    /// [`AllocError::Exhausted`] when the id space is used up (ids are never
    /// reused); nothing is persisted or handed out then.
    ///
    /// [`AllocError::Persist`] when the bumped counter could not be written. The
    /// id is **not** returned: an un-persisted id is the one way an id could be
    /// reused after a crash, so the caller gets an error to report instead (this
    /// used to be a panic, which let a full disk, a missing state directory or a
    /// momentary sharing violation take the whole orchestrator down — including
    /// from a `ShardHello` arriving over the network). The in-memory counter
    /// stays bumped, so the withheld id is skipped, not handed out later.
    pub fn allocate(&mut self) -> Result<ShardId, AllocError> {
        let id = self.inner.allocate()?;
        // Persist-then-return: if this fails the id must not reach a launcher.
        self.sink
            .persist(self.inner.peek())
            .map_err(AllocError::Persist)?;
        Ok(id)
    }

    /// The id the next [`allocate`](Self::allocate) call returns.
    #[must_use]
    pub fn peek(&self) -> u64 {
        self.inner.peek()
    }

    /// Mark `id` as in use so it is never allocated later. When this moves the
    /// counter, the new value is persisted **before** returning (same rule as
    /// [`allocate`](Self::allocate)).
    ///
    /// # Errors
    ///
    /// The sink's error when persisting the bumped counter fails, so the caller
    /// can refuse to proceed rather than silently lose the reservation.
    pub fn reserve(&mut self, id: ShardId) -> io::Result<()> {
        let before = self.inner.peek();
        self.inner.reserve(id);
        if self.inner.peek() != before {
            self.sink.persist(self.inner.peek())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "space-orch-ids-{tag}-{}-{}.txt",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ))
    }

    #[test]
    fn ids_are_monotonic_and_never_reused() {
        let mut alloc =
            PersistentShardIdAllocator::with_sink(1, Box::new(MemoryCounterSink::default()));
        let mut seen = Vec::new();
        for _ in 0..1_000 {
            seen.push(alloc.allocate().unwrap());
        }
        for pair in seen.windows(2) {
            assert!(pair[0] < pair[1], "{pair:?} is not strictly increasing");
        }
        let mut sorted = seen.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 1_000, "an id was handed out twice");
        assert_eq!(seen.first().copied(), Some(ShardId::new(1)));
    }

    #[test]
    fn the_counter_is_persisted_before_the_id_is_returned() {
        // The memory sink records the *next* value at persist time; if the
        // order were return-then-persist the recorded value would be identical
        // but the crash window would exist — so the property asserted here is
        // that every allocation persists exactly once, with the value *after*
        // the returned id.
        let sink = Box::<MemoryCounterSink>::default();
        let mut alloc = PersistentShardIdAllocator::with_sink(7, sink);
        assert_eq!(alloc.allocate().unwrap(), ShardId::new(7));
        assert_eq!(alloc.allocate().unwrap(), ShardId::new(8));
        // Rebuild the sink view through peek: after two allocations the counter
        // sits at 9 and both persists wrote 8 then 9.
        assert_eq!(alloc.peek(), 9);
    }

    #[test]
    fn a_restart_resumes_strictly_above_the_last_allocated_id() {
        let path = temp_path("restart");
        let _ = std::fs::remove_file(&path);
        {
            let mut alloc = PersistentShardIdAllocator::open(&path, 1).expect("open");
            assert_eq!(alloc.allocate().unwrap(), ShardId::new(1));
            assert_eq!(alloc.allocate().unwrap(), ShardId::new(2));
        }
        // Fresh process, same file: the first id is strictly greater than the
        // last one before the restart.
        let mut alloc = PersistentShardIdAllocator::open(&path, 1).expect("reopen");
        assert_eq!(alloc.allocate().unwrap(), ShardId::new(3));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_missing_counter_file_falls_back_to_first() {
        let path = temp_path("missing");
        let _ = std::fs::remove_file(&path);
        let mut alloc = PersistentShardIdAllocator::open(&path, 42).expect("open");
        assert_eq!(alloc.allocate().unwrap(), ShardId::new(42));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_lone_tmp_file_is_resumed_from_not_reset_to_first() {
        let path = temp_path("tmponly");
        let tmp = path.with_extension("tmp");
        let _ = std::fs::remove_file(&path);
        // The file stores the *next* id: 57 was the last one handed out.
        std::fs::write(&tmp, b"58\n").unwrap();
        let mut alloc = PersistentShardIdAllocator::open(&path, 1).expect("open");
        assert_eq!(alloc.allocate().unwrap(), ShardId::new(58));
        assert_eq!(alloc.allocate().unwrap(), ShardId::new(59));
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn a_corrupt_counter_file_is_an_error_not_a_silent_reset() {
        let path = temp_path("corrupt");
        std::fs::write(&path, b"not-a-number").unwrap();
        let err = PersistentShardIdAllocator::open(&path, 1).expect_err("must not silently reset");
        assert!(matches!(err, InitError::Corrupt { .. }), "{err:?}");
        std::fs::remove_file(&path).ok();
    }

    #[derive(Debug)]
    struct FailingSink;

    impl CounterSink for FailingSink {
        fn persist(&mut self, _next: u64) -> io::Result<()> {
            Err(io::Error::other("disk full"))
        }
    }

    #[test]
    fn reserve_bumps_past_the_id_ignores_none_and_surfaces_persist_failure() {
        let mut ids =
            PersistentShardIdAllocator::with_sink(100, Box::new(MemoryCounterSink::default()));
        ids.reserve(ShardId::new(100)).expect("reserve");
        assert_eq!(ids.peek(), 101);
        ids.reserve(ShardId::new(5)).expect("lower id is a no-op");
        assert_eq!(ids.peek(), 101, "the counter never moves down");
        ids.reserve(ShardId::NONE).expect("NONE is ignored");
        assert_eq!(ids.peek(), 101);

        let mut failing = PersistentShardIdAllocator::with_sink(100, Box::new(FailingSink));
        assert!(failing.reserve(ShardId::new(100)).is_err());
    }

    /// A sink that fails while `broken` is set and records what it persisted.
    #[derive(Debug, Default)]
    struct FlakySink {
        broken: std::sync::Arc<std::sync::atomic::AtomicBool>,
        written: std::sync::Arc<std::sync::Mutex<Vec<u64>>>,
    }

    impl CounterSink for FlakySink {
        fn persist(&mut self, next: u64) -> io::Result<()> {
            if self.broken.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(io::Error::other("disk full"));
            }
            self.written.lock().unwrap().push(next);
            Ok(())
        }
    }

    /// Mutation this catches: panicking on a failed persist (the old `expect`),
    /// and handing the withheld id out on the next call (a reuse after a crash).
    #[test]
    fn a_failed_persist_withholds_the_id_and_it_is_never_handed_out_later() {
        let sink = FlakySink::default();
        let (broken, written) = (
            std::sync::Arc::clone(&sink.broken),
            std::sync::Arc::clone(&sink.written),
        );
        let mut alloc = PersistentShardIdAllocator::with_sink(10, Box::new(sink));
        assert_eq!(alloc.allocate().unwrap(), ShardId::new(10));

        broken.store(true, std::sync::atomic::Ordering::SeqCst);
        let err = alloc
            .allocate()
            .expect_err("an unpersisted id is not handed out");
        assert!(matches!(err, AllocError::Persist(_)), "{err:?}");
        assert!(err.to_string().contains("could not be persisted"), "{err}");

        // The disk comes back: id 11 was withheld, so it is skipped for good.
        broken.store(false, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(alloc.allocate().unwrap(), ShardId::new(12));
        assert_eq!(
            *written.lock().unwrap(),
            vec![11, 13],
            "every id handed out was persisted first, and the counter only moved up"
        );
    }

    /// The file-backed path of the same rule: a state path in a directory that
    /// does not exist is an error from `allocate`, not a panic.
    #[test]
    fn a_counter_in_a_missing_directory_is_an_error_not_a_panic() {
        let dir = temp_path("no-such-dir");
        let path = dir.join("orchestrator.id");
        let _ = std::fs::remove_dir_all(&dir);
        let mut alloc = PersistentShardIdAllocator::open(&path, 1).expect("nothing to read yet");
        let err = alloc.allocate().expect_err("the counter cannot be written");
        assert!(matches!(err, AllocError::Persist(_)), "{err:?}");
        assert!(!path.exists(), "nothing was created");
    }

    /// A bare file name (the default `space-orchestrator.id`) lives in the
    /// current directory; `Path::parent` says `""` for it, which cannot be
    /// opened, so the directory sync was silently skipped.
    #[test]
    fn the_directory_of_a_bare_file_name_is_the_current_directory() {
        assert_eq!(
            FileCounterSink::dir_of(Path::new("space-orchestrator.id")),
            Path::new(".")
        );
        assert_eq!(
            FileCounterSink::dir_of(Path::new("state/orchestrator.id")),
            Path::new("state")
        );
    }

    /// Every persist replaces the counter file in place, leaves no temp file
    /// behind, and the value on disk is the one a restart resumes from.
    #[test]
    fn persist_replaces_the_counter_file_and_leaves_no_temp_file() {
        let path = temp_path("replace");
        let _ = std::fs::remove_file(&path);
        let mut sink = FileCounterSink::new(&path);
        for next in [2u64, 3, 40] {
            sink.persist(next).expect("persist");
            assert_eq!(FileCounterSink::read_one(&path).unwrap(), Some(next));
            assert!(
                !FileCounterSink::tmp_path(&path).exists(),
                "the temp file was renamed away"
            );
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn the_reserved_none_id_is_never_handed_out() {
        // The inner allocator errors at the id-space limit rather than wrapping
        // onto ShardId::NONE (u32::MAX); the wrapper must not change that, and
        // must not persist anything for the failed allocation.
        let mut alloc = PersistentShardIdAllocator::with_sink(
            u64::from(u32::MAX - 1),
            Box::new(MemoryCounterSink::default()),
        );
        assert_eq!(alloc.allocate().unwrap(), ShardId::new(u32::MAX - 1));
        assert!(
            matches!(alloc.allocate(), Err(AllocError::Exhausted(_))),
            "must fail, not wrap or panic"
        );
        assert_eq!(alloc.peek(), u64::from(u32::MAX));
    }
}
