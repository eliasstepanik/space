//! The cluster's scene: what the orchestrator holds, and where it keeps it.
//!
//! The scene is a piece of cluster state the orchestrator **owns** (see
//! [`space_protocol::scene`]): it tells every shard which scene it is, every
//! shard tells its clients, and an operator can change it while the cluster
//! runs. This module is the pure half — the current scene, its revision and its
//! persistence. Sending it to the shards and keeping track of who confirmed
//! which revision is [`crate::orchestrator`]'s.
//!
//! # Not an epoch
//!
//! Changing the scene does not change the shard map, so it has nothing to do
//! with the epoch protocol: it is not proposed, prepared, acked into a commit
//! or aborted, it does not wait for an epoch in flight and no epoch waits for
//! it. H9–H15 are statements about ids and the map; none of them reads or is
//! affected by anything here.
//!
//! # Revisions
//!
//! Every change raises the revision by one, and a shard (and through it a
//! client) only ever adopts a higher revision. The revision is persisted with
//! the scene, so it keeps rising across an orchestrator restart.
//!
//! # Persistence
//!
//! The scene an operator chose survives a restart: it is written to a small
//! file next to the id counter ([`scene_path`]: `<state file>.scene`), one line
//! `"<revision> <name>"`, with the same care as the counter — written to a
//! temp file, synced, then renamed over the old one ([`crate::ids`]) —
//! **before** the change is told to anyone. A change that cannot be saved is
//! refused, so what the shards run never gets ahead of what a restart would
//! bring back.
//!
//! Reading is forgiving where the counter's is strict. A reused shard id
//! aliases two entities, so a corrupt counter refuses the boot; a lost scene
//! only costs the operator one click. A scene file that is missing, unreadable
//! or does not make sense therefore **never stops the orchestrator from
//! starting**: the configured initial scene is used and the problem is reported
//! ([`SceneStore::warning`]).
//!
//! The initial scene (`--scene`) is exactly that: what a cluster shows until an
//! operator chooses. Once a scene has been saved, the saved one wins.
//!
//! A file naming a **retired** scene ([`RETIRED_SCENE_NAMES`]: `solar`, now
//! part of the demo area) is not a problem either: it loads as
//! [`Scene::DEFAULT`] one revision above the saved one (so it supersedes what
//! the shards held), and [`SceneStore::note`] says the retired scene was
//! replaced.

use std::io;
use std::path::{Path, PathBuf};

use space_protocol::{Scene, SceneState};

/// The revision a scene has when nothing was persisted before it.
pub const FIRST_REVISION: u64 = 1;

/// The largest gap between this orchestrator's revision and one a shard
/// reports that is still taken for the truth ("my scene file was lost") rather
/// than for nonsense.
///
/// The orchestrator catches up with a revision a shard reports (see
/// [`SceneStore::raise_above`]). Without a bound, a single message naming a
/// revision near `u64::MAX` — the control port checks a cluster name and
/// nothing else — would leave no revision above it, and the scene could never
/// change again. A real gap is the number of scene changes ever made; four
/// billion is far above that and far below the ceiling.
pub const MAX_REVISION_CATCH_UP: u64 = 1 << 32;

/// How often the run loop offers the scene again to a connected shard that has
/// not confirmed the current revision, in ticks (once a second).
///
/// The scene rides the same per-shard queue as everything else, and a message
/// offered to a full queue is dropped rather than waited for. This is what
/// makes a dropped `SceneSet` (or a dropped ack) heal itself.
pub const SCENE_RESEND_INTERVAL_TICKS: u64 = crate::config::TICK_HZ;

/// Names of scenes that were retired from the catalogue
/// ([`Scene::RETIRED_IDS`]). A scene file naming one loads as
/// [`Scene::DEFAULT`] (see the module docs, "Persistence").
pub const RETIRED_SCENE_NAMES: [&str; 1] = ["solar"];

/// Where the scene is persisted for the state file at `state_path`: the same
/// path with `.scene` **appended** (`space-orchestrator.id.scene`).
///
/// Appended rather than swapped for the extension so the name can never
/// collide with the counter file or with the `.tmp` sibling the counter is
/// written through, whatever the operator called the state file.
#[must_use]
pub fn scene_path(state_path: &Path) -> PathBuf {
    with_suffix(state_path, ".scene")
}

/// `path` with `suffix` appended to its file name.
fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// Where a scene is persisted before it is published.
///
/// Object-safe, like [`crate::ids::CounterSink`], so a test can make the write
/// fail or record what was written.
pub trait SceneSink: core::fmt::Debug + Send + Sync {
    /// Persist `state` durably. [`SceneStore::set`] calls this **before** the
    /// new scene becomes the current one.
    ///
    /// # Errors
    ///
    /// Any error refuses the change: the scene stays what it was.
    fn persist(&mut self, state: SceneState) -> io::Result<()>;
}

/// The scene file: one line, `"<revision> <name>"`, replaced atomically.
#[derive(Debug)]
pub struct FileSceneSink {
    path: PathBuf,
}

impl FileSceneSink {
    /// A sink writing to `path`.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// The sibling temp file a write goes through.
    fn tmp_path(path: &Path) -> PathBuf {
        with_suffix(path, ".tmp")
    }
}

impl SceneSink for FileSceneSink {
    fn persist(&mut self, state: SceneState) -> io::Result<()> {
        // Only a catalogue scene is ever set (see `SceneStore::set`), so the
        // id always has a name; the fallback keeps this total.
        let line = format!("{} {}\n", state.revision, state.scene.or_default().name());
        crate::ids::replace_file(&self.path, &Self::tmp_path(&self.path), line.as_bytes())
    }
}

/// A sink that only records what it was asked to persist (tests, and an
/// orchestrator assembled without a state directory).
#[derive(Debug, Default)]
pub struct MemorySceneSink {
    /// Every state handed to [`SceneSink::persist`], in order.
    pub written: Vec<SceneState>,
}

impl SceneSink for MemorySceneSink {
    fn persist(&mut self, state: SceneState) -> io::Result<()> {
        self.written.push(state);
        Ok(())
    }
}

/// What a scene file turned out to hold.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Stored {
    /// Neither the file nor its temp sibling exists: a fresh state directory.
    Nothing,
    /// A scene this build knows, at its revision.
    Scene(Scene, u64),
    /// A revision, but a scene name this build does not know (the file was
    /// written by a build with a larger catalogue).
    UnknownScene(String, u64),
    /// A scene that was retired from the catalogue, at its revision.
    Retired(String, u64),
    /// Something else. The text says what.
    Unusable(String),
}

/// Read the scene file at `path` (falling back to its temp sibling when the
/// file itself is missing, like the counter does).
fn read_stored(path: &Path) -> Stored {
    for candidate in [path.to_path_buf(), FileSceneSink::tmp_path(path)] {
        match std::fs::read_to_string(&candidate) {
            Ok(text) => return parse_stored(&text),
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => return Stored::Unusable(format!("it cannot be read ({err})")),
        }
    }
    Stored::Nothing
}

/// Parse the one line of a scene file.
fn parse_stored(text: &str) -> Stored {
    let mut fields = text.split_whitespace();
    let (Some(revision), Some(name), None) = (fields.next(), fields.next(), fields.next()) else {
        return Stored::Unusable(format!(
            "it does not hold \"<revision> <scene>\" (found {:?})",
            shown(text)
        ));
    };
    let Some(revision) = revision
        .parse::<u64>()
        .ok()
        .filter(|r| *r >= FIRST_REVISION)
    else {
        return Stored::Unusable(format!(
            "its revision {:?} is not a number from {FIRST_REVISION} up",
            shown(revision)
        ));
    };
    match Scene::from_name(name) {
        Some(scene) => Stored::Scene(scene, revision),
        None if RETIRED_SCENE_NAMES.contains(&name) => Stored::Retired(name.to_owned(), revision),
        None => Stored::UnknownScene(shown(name), revision),
    }
}

/// At most 60 characters of `text`, for a message (a corrupt file can hold
/// anything, and any amount of it).
fn shown(text: &str) -> String {
    let trimmed = text.trim();
    let mut out: String = trimmed.chars().take(60).collect();
    if out.len() < trimmed.len() {
        out.push('…');
    }
    out
}

/// The cluster's scene and its revision, with where it is persisted.
#[derive(Debug)]
pub struct SceneStore {
    /// The current scene (always one of the catalogue).
    scene: Scene,
    /// Its revision (never below [`FIRST_REVISION`]).
    revision: u64,
    sink: Box<dyn SceneSink>,
    /// The file behind `sink`, when there is one.
    path: Option<PathBuf>,
    /// What went wrong reading the file at startup, if anything did.
    warning: Option<String>,
    /// Whether the current scene was read from the file (rather than being
    /// the configured initial scene).
    restored: bool,
    /// A remark about the file read at startup that is not a problem (a
    /// retired scene replaced by the default), if any.
    note: Option<String>,
}

impl SceneStore {
    /// A store that keeps `initial` at [`FIRST_REVISION`] and persists nowhere
    /// (tests; an orchestrator assembled without [`crate::boot`]).
    #[must_use]
    pub fn in_memory(initial: Scene) -> Self {
        Self::with_sink(initial, Box::new(MemorySceneSink::default()))
    }

    /// A store over an arbitrary sink, starting at `initial`,
    /// [`FIRST_REVISION`].
    #[must_use]
    pub fn with_sink(initial: Scene, sink: Box<dyn SceneSink>) -> Self {
        Self {
            scene: initial,
            revision: FIRST_REVISION,
            sink,
            path: None,
            warning: None,
            restored: false,
            note: None,
        }
    }

    /// Open the store over the scene file at `path`.
    ///
    /// The scene saved there is restored, with its revision. When there is no
    /// file the store starts at `initial`, [`FIRST_REVISION`]. When the file
    /// cannot be used the store starts at `initial` too and
    /// [`Self::warning`] says why — this never fails (see the module docs).
    /// When the file names a retired scene the store starts at
    /// [`Scene::DEFAULT`], one revision above the saved one, and
    /// [`Self::note`] says so.
    ///
    /// Nothing is written until the scene is changed.
    #[must_use]
    pub fn open(path: &Path, initial: Scene) -> Self {
        let mut note = None;
        let (scene, revision, restored, warning) = match read_stored(path) {
            Stored::Nothing => (initial, FIRST_REVISION, false, None),
            Stored::Scene(scene, revision) => (scene, revision, true, None),
            // Not a problem: the retired scene's content lives on in the
            // default. One revision up, so every shard and client that still
            // holds the old revision adopts the replacement.
            Stored::Retired(name, revision) => {
                let revision = revision.saturating_add(1);
                note = Some(format!(
                    "the scene file {} names the retired scene {name:?}; it was replaced by \
                     the default scene {} at revision {revision}",
                    path.display(),
                    Scene::DEFAULT
                ));
                (Scene::DEFAULT, revision, false, None)
            }
            // The revision is still good: keep counting from it, so the scene
            // that replaces the unknown one is a newer revision, not a repeat.
            Stored::UnknownScene(name, revision) => (
                initial,
                revision.saturating_add(1),
                false,
                Some(format!(
                    "the scene file {} names the scene {name:?}, which this build does not know \
                     (the scenes are {}); using {initial} at revision {}",
                    path.display(),
                    Scene::names(),
                    revision.saturating_add(1)
                )),
            ),
            Stored::Unusable(why) => (
                initial,
                FIRST_REVISION,
                false,
                Some(format!(
                    "the scene file {} is not usable: {why}; using {initial} at revision \
                     {FIRST_REVISION}",
                    path.display()
                )),
            ),
        };
        Self {
            scene,
            revision,
            sink: Box::new(FileSceneSink::new(path)),
            path: Some(path.to_path_buf()),
            warning,
            restored,
            note,
        }
    }

    /// The current scene.
    #[must_use]
    pub fn scene(&self) -> Scene {
        self.scene
    }

    /// The current scene's revision.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// The current scene and revision, as the wire carries them.
    #[must_use]
    pub fn current(&self) -> SceneState {
        SceneState::new(self.scene.id(), self.revision)
    }

    /// The file the scene is persisted in (`None`: nowhere).
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// What went wrong reading the scene file at startup (`None`: nothing).
    #[must_use]
    pub fn warning(&self) -> Option<&str> {
        self.warning.as_deref()
    }

    /// A remark about the scene file read at startup that is not a problem
    /// (a retired scene replaced by the default); `None`: nothing to say.
    #[must_use]
    pub fn note(&self) -> Option<&str> {
        self.note.as_deref()
    }

    /// Whether the current scene was read back from the scene file.
    #[must_use]
    pub fn restored(&self) -> bool {
        self.restored
    }

    /// Make `scene` the current scene at the next revision, persisting it
    /// **before** it becomes current, and return the new state.
    ///
    /// Setting the scene that is already current is still a new revision; a
    /// caller that wants "no change" checks [`Self::scene`] first.
    ///
    /// # Errors
    ///
    /// The sink's error when the new state could not be persisted, or an error
    /// when the revision cannot rise any further. Either way nothing changed.
    pub fn set(&mut self, scene: Scene) -> io::Result<SceneState> {
        let revision = self
            .revision
            .checked_add(1)
            .ok_or_else(|| io::Error::other("the scene revision cannot rise any further"))?;
        let next = SceneState::new(scene.id(), revision);
        self.sink.persist(next)?;
        self.scene = scene;
        self.revision = revision;
        self.restored = false;
        Ok(next)
    }

    /// Re-issue the **current** scene at a revision above `seen` (and above
    /// the current one), persisting it first, and return the new state.
    ///
    /// This is how the orchestrator stays the authority when it finds a shard
    /// running a revision it does not have — its scene file was lost or
    /// replaced, so it restarted counting at 1 while shards and clients still
    /// hold a higher number. Receivers only ever adopt a higher revision, so
    /// from below nothing it says would be heard, by anyone, for ever. The
    /// scene stays the orchestrator's; only the number moves.
    ///
    /// # Errors
    ///
    /// The sink's error, or an error when no higher revision exists. Nothing
    /// changed then.
    pub fn raise_above(&mut self, seen: u64) -> io::Result<SceneState> {
        let revision = seen
            .max(self.revision)
            .checked_add(1)
            .ok_or_else(|| io::Error::other("the scene revision cannot rise any further"))?;
        let next = SceneState::new(self.scene.id(), revision);
        self.sink.persist(next)?;
        self.revision = revision;
        self.restored = false;
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(tag: &str) -> PathBuf {
        space_test_support::temp_path("space-orch-scene", tag).with_extension("id")
    }

    fn remove(state_path: &Path) {
        let scene = scene_path(state_path);
        let _ = std::fs::remove_file(&scene);
        let _ = std::fs::remove_file(FileSceneSink::tmp_path(&scene));
    }

    /// Mutation this catches: deriving the scene file with `with_extension`.
    /// A state file called `x.scene` would then be overwritten by the scene,
    /// and one called `x.tmp`'s sibling would be read back as the id counter.
    #[test]
    fn the_scene_file_sits_next_to_the_state_file_and_collides_with_nothing() {
        assert_eq!(
            scene_path(Path::new("space-orchestrator.id")),
            Path::new("space-orchestrator.id.scene")
        );
        assert_eq!(
            scene_path(Path::new("/data/orchestrator.id")),
            Path::new("/data/orchestrator.id.scene")
        );
        for state in ["a.id", "a", "a.scene", "a.tmp", "dir/a.b.c"] {
            let state = Path::new(state);
            let scene = scene_path(state);
            let scene_tmp = FileSceneSink::tmp_path(&scene);
            // The counter's own files: the state file and its `.tmp` sibling.
            let counter_tmp = state.with_extension("tmp");
            for ours in [&scene, &scene_tmp] {
                assert_ne!(ours.as_path(), state, "{state:?}");
                assert_ne!(ours.as_path(), counter_tmp.as_path(), "{state:?}");
            }
            assert_ne!(scene, scene_tmp);
        }
    }

    #[test]
    fn a_fresh_state_directory_starts_at_the_initial_scene_and_writes_nothing() {
        let state = temp_path("fresh");
        remove(&state);
        let path = scene_path(&state);
        let store = SceneStore::open(&path, Scene::Galaxy);
        assert_eq!(store.current(), SceneState::new(Scene::Galaxy.id(), 1));
        assert_eq!(store.scene(), Scene::Galaxy);
        assert!(!store.restored());
        assert_eq!(store.warning(), None, "a missing file is the normal start");
        assert_eq!(store.path(), Some(path.as_path()));
        assert!(!path.exists(), "nothing is written until the scene changes");
    }

    /// The point of the file: the scene an operator chose, and its revision,
    /// are what a restarted orchestrator comes back with — whatever its
    /// `--scene` says.
    ///
    /// Mutation this catches: letting the initial scene win over a saved one,
    /// or restarting the revision at 1 (a shard holding revision 3 would then
    /// ignore the next two changes).
    #[test]
    fn a_chosen_scene_and_its_revision_survive_a_restart() {
        let state = temp_path("restart");
        remove(&state);
        let path = scene_path(&state);
        {
            let mut store = SceneStore::open(&path, Scene::DemoArea);
            assert_eq!(
                store.set(Scene::Materials).expect("saved"),
                SceneState::new(Scene::Materials.id(), 2)
            );
            assert_eq!(
                store.set(Scene::Galaxy).expect("saved"),
                SceneState::new(Scene::Galaxy.id(), 3)
            );
            assert_eq!(store.current(), SceneState::new(Scene::Galaxy.id(), 3));
        }
        assert_eq!(
            std::fs::read_to_string(&path).expect("written"),
            "3 galaxy\n"
        );
        assert!(
            !FileSceneSink::tmp_path(&path).exists(),
            "the temp file was renamed away"
        );

        // A new process, started with another initial scene.
        let mut store = SceneStore::open(&path, Scene::Materials);
        assert_eq!(store.current(), SceneState::new(Scene::Galaxy.id(), 3));
        assert!(store.restored());
        assert_eq!(store.warning(), None);
        // And the revision keeps rising from there.
        assert_eq!(
            store.set(Scene::Materials).expect("saved"),
            SceneState::new(Scene::Materials.id(), 4)
        );
        remove(&state);
    }

    /// Mutation this catches: treating the scene file like the id counter (a
    /// corrupt one refuses the boot). A lost scene costs one click; an
    /// orchestrator that does not start costs the cluster.
    #[test]
    fn a_corrupt_scene_file_falls_back_to_the_initial_scene_with_a_warning() {
        let state = temp_path("corrupt");
        let path = scene_path(&state);
        let huge = "x".repeat(10_000);
        for junk in [
            "",
            "\n",
            "galaxy",
            "galaxy 3",
            "three galaxy",
            "0 galaxy",
            "-1 galaxy",
            "3 galaxy extra",
            "18446744073709551616 galaxy",
            "\u{0}\u{1}\u{2}",
            huge.as_str(),
        ] {
            std::fs::write(&path, junk).expect("write junk");
            let mut store = SceneStore::open(&path, Scene::Materials);
            assert_eq!(
                store.current(),
                SceneState::new(Scene::Materials.id(), FIRST_REVISION),
                "{junk:?}"
            );
            assert!(!store.restored());
            let warning = store.warning().expect("the problem is reported").to_owned();
            assert!(
                warning.contains(&path.display().to_string()) && warning.contains("materials"),
                "{warning}"
            );
            assert!(warning.len() < 600, "a huge file does not make a huge line");
            // The store still works, and the next change repairs the file.
            assert!(store.set(Scene::Galaxy).is_ok(), "{junk:?}");
            assert_eq!(
                std::fs::read_to_string(&path).expect("repaired"),
                "2 galaxy\n"
            );
        }
        // Invalid UTF-8 is "cannot be read", not a panic.
        std::fs::write(&path, [0xff, 0xfe, 0x00]).expect("write bytes");
        let store = SceneStore::open(&path, Scene::Galaxy);
        assert_eq!(store.scene(), Scene::Galaxy);
        assert!(store.warning().is_some());
        remove(&state);
    }

    /// A file written by a build with a larger catalogue: the scene is not
    /// known here, but the revision is, and it keeps counting.
    #[test]
    fn an_unknown_scene_name_keeps_the_revision_rising() {
        let state = temp_path("unknown");
        let path = scene_path(&state);
        std::fs::write(&path, "7 nebula\n").expect("write");
        let store = SceneStore::open(&path, Scene::DemoArea);
        assert_eq!(store.current(), SceneState::new(Scene::DemoArea.id(), 8));
        let warning = store.warning().expect("reported");
        assert!(
            warning.contains("nebula") && warning.contains("revision 8"),
            "{warning}"
        );
        remove(&state);
    }

    /// `solar` was merged into the demo area. A cluster whose operator had
    /// chosen it comes back up on the default scene, without a warning, one
    /// revision above the saved one (so shards and clients still holding the
    /// old revision adopt it), and says what it replaced.
    ///
    /// Mutation this catches: treating the retired name like an unknown one
    /// (a warning on the operator's page for a planned change, and `--scene`
    /// instead of the default), or restarting the revision at 1.
    #[test]
    fn a_retired_scene_loads_as_the_default_and_keeps_the_revision_rising() {
        let state = temp_path("retired");
        remove(&state);
        let path = scene_path(&state);
        std::fs::write(&path, "6 solar\n").expect("write");
        let mut store = SceneStore::open(&path, Scene::Materials);
        assert_eq!(store.current(), SceneState::new(Scene::DEFAULT.id(), 7));
        assert_eq!(store.scene(), Scene::DemoArea);
        assert!(
            store
                .current()
                .supersedes(&SceneState::new(space_protocol::SceneId(2), 6)),
            "the replacement supersedes the retired scene's revision"
        );
        assert_eq!(store.warning(), None, "a retired scene is not a problem");
        assert!(!store.restored());
        let note = store.note().expect("the replacement is reported");
        assert!(
            note.contains("retired scene \"solar\"")
                && note.contains("demo-area")
                && note.contains("revision 7"),
            "{note}"
        );
        // Nothing is written until the scene changes; then the file is clean.
        assert_eq!(
            std::fs::read_to_string(&path).expect("untouched"),
            "6 solar\n"
        );
        assert_eq!(store.set(Scene::Galaxy).expect("saved").revision, 8);
        assert_eq!(std::fs::read_to_string(&path).expect("saved"), "8 galaxy\n");
        remove(&state);
    }

    /// The same crash window the counter has: only the temp file was written.
    #[test]
    fn a_lone_temp_file_is_read_like_the_scene_file() {
        let state = temp_path("tmponly");
        remove(&state);
        let path = scene_path(&state);
        std::fs::write(FileSceneSink::tmp_path(&path), "5 galaxy\n").expect("write");
        let store = SceneStore::open(&path, Scene::DemoArea);
        assert_eq!(store.current(), SceneState::new(Scene::Galaxy.id(), 5));
        remove(&state);
    }

    #[derive(Debug)]
    struct BrokenSink;

    impl SceneSink for BrokenSink {
        fn persist(&mut self, _state: SceneState) -> io::Result<()> {
            Err(io::Error::other("disk full"))
        }
    }

    /// Mutation this catches: publishing the scene first and saving it after
    /// (or ignoring the write error). The shards would run a scene a restart
    /// silently takes away again.
    #[test]
    fn a_scene_that_cannot_be_saved_is_not_adopted() {
        let mut store = SceneStore::with_sink(Scene::DemoArea, Box::new(BrokenSink));
        let before = store.current();
        let err = store.set(Scene::Galaxy).expect_err("the write fails");
        assert!(err.to_string().contains("disk full"));
        assert_eq!(store.current(), before, "nothing changed");
        assert_eq!(store.scene(), Scene::DemoArea);

        // The file-backed path of the same rule: a directory that is not there.
        let dir = temp_path("no-such-dir");
        let _ = std::fs::remove_dir_all(&dir);
        let mut store = SceneStore::open(&dir.join("orchestrator.id.scene"), Scene::DemoArea);
        assert!(
            store.warning().is_none(),
            "nothing to read is not a problem"
        );
        assert!(store.set(Scene::Galaxy).is_err());
        assert_eq!(store.scene(), Scene::DemoArea);
    }

    /// Mutation this catches: re-issuing at `seen` itself (the shard that
    /// holds `seen` would still not adopt it), changing the scene along the
    /// way, or raising without saving (a restart would fall behind again).
    #[test]
    fn raising_above_a_seen_revision_keeps_the_scene_and_saves_the_number() {
        let state = temp_path("raise");
        remove(&state);
        let path = scene_path(&state);
        let mut store = SceneStore::open(&path, Scene::Galaxy);
        assert_eq!(store.current(), SceneState::new(Scene::Galaxy.id(), 1));
        assert_eq!(
            store.raise_above(9).expect("saved"),
            SceneState::new(Scene::Galaxy.id(), 10)
        );
        assert_eq!(
            std::fs::read_to_string(&path).expect("saved"),
            "10 galaxy\n"
        );
        // Something at or below the current revision still moves one up.
        assert_eq!(store.raise_above(3).expect("saved").revision, 11);
        assert_eq!(store.raise_above(11).expect("saved").revision, 12);
        assert_eq!(store.scene(), Scene::Galaxy);
        // The ceiling is an error, not a wrap-around.
        assert!(store.raise_above(u64::MAX).is_err());
        assert_eq!(store.revision(), 12);
        // A restart resumes from the raised number.
        assert_eq!(SceneStore::open(&path, Scene::DemoArea).revision(), 12);
        remove(&state);

        let mut broken = SceneStore::with_sink(Scene::DemoArea, Box::new(BrokenSink));
        assert!(broken.raise_above(9).is_err());
        assert_eq!(broken.revision(), 1, "nothing changed");
    }

    #[test]
    fn the_memory_store_counts_revisions_from_one() {
        let mut store = SceneStore::in_memory(Scene::DEFAULT);
        assert_eq!(store.current(), SceneState::new(Scene::DemoArea.id(), 1));
        assert_eq!(store.path(), None);
        assert_eq!(store.set(Scene::Galaxy).expect("ok").revision, 2);
        // The same scene again is still a new revision at this level.
        assert_eq!(store.set(Scene::Galaxy).expect("ok").revision, 3);
        assert_eq!(store.revision(), 3);
    }
}
