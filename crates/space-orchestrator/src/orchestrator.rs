//! [`Orchestrator`]: the tie-in point (plan §3.1, §3.6).
//!
//! The orchestrator is a **synchronous state machine** driven by a tick, exactly
//! like [`space_server::shard::Shard`]: every decision — id allocation, topology,
//! epoch, autoscale — is a pure call into [`crate::control`] / [`crate::topology`]
//! / [`crate::epoch`] / [`crate::autoscale`], and the async layer only moves
//! messages. That split is what makes the whole control loop testable without a
//! wall clock: a test calls [`Orchestrator::tick`] and injects reports/acks
//! directly, so dwell timers advance by ticks, not by sleeping (plan §3.6, H14).
//!
//! # The loop (plan §3.6)
//!
//! ```text
//! loop { select! {
//!     report = load_reports()      => orch.handle_load_report(report),
//!     ack    = control.acks()      => orch.handle_epoch_ack(shard, epoch),
//!     hello  = control.hellos()    => orch.handle_shard_hello(hello),
//!     _      = clock.tick()        => { orch.tick(); drain outbox; launcher.poll(); }
//! }}
//! ```
//!
//! [`Orchestrator::tick`] is the synchronous heart of that loop: it advances the
//! orchestrator's own clock, runs the epoch deadline, feeds the autoscaler the
//! latest report per shard, and on a `Split`/`Merge` computes a new topology,
//! bumps the epoch through [`crate::epoch::EpochPublisher`] and stages the
//! launcher work ([`Orchestrator::take_pending_spawn`] /
//! [`Orchestrator::take_pending_kill`]) for the async layer to execute. A
//! split's child is staged for launch when the split is proposed (it has to
//! register and ack before the epoch can commit); a
//! merged shard is staged for killing only when the merge **commits** — until
//! then it still owns its region — and a child whose split aborts is staged for
//! killing at the abort. The async [`Orchestrator::run`] wires those pieces
//! together over real QUIC; the tests drive the synchronous half directly.
//!
//! # What the orchestrator believes about load
//!
//! Only a [`LoadReport`] from a shard that owns a region in the published map is
//! kept, and only while it is fresh
//! ([`crate::config::REPORT_EXPIRY_TICKS`]): a shard that was merged away,
//! killed, or simply stopped reporting is forgotten — by the autoscaler too —
//! instead of being re-split or re-merged for ever on its last words.
//!
//! # What an operator can see and do
//!
//! [`Orchestrator::snapshot`] copies everything above into a
//! [`Snapshot`], and [`Orchestrator::handle_operator`] carries out an
//! [`OperatorCommand`]. Both are synchronous like the rest of this half; the
//! run loop offers them to a front end through [`Orchestrator::attach_ops`]
//! (see [`crate::ops`]). An operator's split or merge goes through
//! `begin_split` / `begin_merge` — the very functions the autoscaler calls — so
//! it is planned, validated, proposed as one epoch and launched exactly like an
//! autoscaler decision, and it is refused, with the reason, whenever one of
//! those steps says no. There is no second path to the map.
//!
//! The orchestrator also keeps a bounded [`EventLog`] of what happened
//! (registrations, lost links, proposals, commits, aborts, operator requests).
//! It is written for people; no decision ever reads it.
//!
//! # The scene
//!
//! The cluster's scene ([`crate::scene`]) is the one piece of state here that
//! is **not** the map. An operator's [`OperatorCommand::SetScene`] persists
//! the new scene, raises its revision and marks it for sending; every shard
//! that has said hello is then sent a `SceneSet` (once per beat, however many
//! changes were asked for in between), and a shard that registers later is
//! handed the current one right behind its welcome. Each shard answers with a
//! `SceneAck` ([`Orchestrator::handle_scene_ack`]); what it confirmed is kept
//! per shard and shown, and a connected shard that has not confirmed the
//! current revision is offered it again once a second
//! ([`Orchestrator::remind_scene_laggards`]).
//!
//! None of this goes near the epoch publisher: a scene change is accepted
//! while a prepare is outstanding, and a split or merge is accepted while a
//! scene change is unconfirmed.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use space_protocol::control::{AdminCommand, AdminReply, LoadReport};
use space_protocol::shard::{EpochAck, ShardId, ShardMap, ShardRegion};
use space_protocol::{Message, Scene, SceneAck, SceneState, ShardHello, ShardWelcome};

use crate::autoscale::ScaleAction;
use crate::config::{OrchestratorConfig, MIN_EPOCH_INTERVAL_TICKS, REPORT_EXPIRY_TICKS};
use crate::control::ControlPlane;
use crate::epoch::{Outbound, Phase};
use crate::events::{EventKind, EventLog, Severity};
use crate::launcher::{Launcher, Poll, SpawnSpec};
use crate::ops::{OperatorCommand, OperatorReply, OperatorRequest, OpsHandle, OpsPort};
use crate::scene::SCENE_RESEND_INTERVAL_TICKS;
use crate::snapshot::{
    node_path, octant_of, octant_signs, AutoscalerView, Decision, EpochInFlight, Heard, HeardKind,
    RegionView, ReportView, SceneView, ShardRole, ShardView, Snapshot,
};
use crate::topology;

/// How often the run loop publishes a fresh [`Snapshot`] to an attached front
/// end when nothing was logged in between, in ticks (5 times a second). A new
/// event-log entry or an operator command publishes at once.
pub const SNAPSHOT_INTERVAL_TICKS: u64 = 6;

/// The length of the window over which the run loop limits how many
/// peer-caused event-log entries (refused hellos) it writes to standard
/// output, in ticks (10 s).
pub const PEER_PRINT_WINDOW_TICKS: u64 = 10 * crate::config::TICK_HZ;

/// How many peer-caused entries are written to standard output per
/// [`PEER_PRINT_WINDOW_TICKS`]; the rest are counted and summed up in one line.
pub const PEER_PRINTS_PER_WINDOW: u32 = 3;

/// Which octant of a region a split carves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Carve {
    /// The first free octant counting round from the one holding the region's
    /// centre (the autoscaler's choice).
    FromCentre,
    /// The first free octant counting round from this one (the admin verb).
    From(usize),
    /// Exactly this octant, or nothing (an operator named it).
    Exactly(usize),
}

/// The orchestrator: control plane + the launcher seam.
///
/// `L` is generic rather than `dyn` because [`Launcher`] uses associated-return-
/// position `impl Future` (not dyn-compatible) — which is also what lets the
/// acceptance tests inject a `MockLauncher` as a plain struct.
#[derive(Debug)]
pub struct Orchestrator<L: Launcher> {
    /// The pure control state (ids, epoch publisher, autoscaler).
    pub control: ControlPlane,
    /// The process layer. `Arc` so a task can hold one while the loop keeps it.
    pub launcher: Arc<L>,
    /// The orchestrator's own clock, in ticks. Advanced only by [`Self::tick`].
    tick: u64,
    /// The most recent [`LoadReport`] per shard, fed to the autoscaler each tick.
    latest: BTreeMap<ShardId, LoadReport>,
    /// The most recent tick each shard reported at (drives `activation_tick`).
    shard_ticks: BTreeMap<ShardId, u64>,
    /// A child id whose shard the loop must spawn (staged by a split).
    pending_spawn: Vec<SpawnSpec>,
    /// A shard the loop must kill (staged by a committed merge, an aborted
    /// split's child, or the admin verb).
    pending_kill: Vec<ShardId>,
    /// The shard the merge **in flight** removes. It is killed when that epoch
    /// commits and left alone when it aborts: until the commit it still owns its
    /// region, so killing it any earlier leaves the map naming a dead owner.
    pending_merge: Option<ShardId>,
    /// The child the split **in flight** launched. If that epoch aborts the
    /// child owns nothing and is killed, instead of running on unowned.
    pending_split: Option<ShardId>,
    /// Messages the async layer must deliver, keyed by destination shard.
    outbox: Vec<(ShardId, Message)>,
    /// The endpoints learned from each shard's `ShardHello` (for spawn specs).
    endpoints: BTreeMap<ShardId, std::net::SocketAddr>,

    // ---- what follows is for the operator view; no decision reads it, with
    // the one exception of `autoscaler_paused`, which an operator sets.
    /// What happened recently.
    events: EventLog,
    /// When each known shard was last heard from, and through what.
    last_heard: BTreeMap<ShardId, Heard>,
    /// The shards whose control link the async layer last saw open.
    connected: BTreeSet<ShardId>,
    /// Whether an operator has paused the autoscaler.
    autoscaler_paused: bool,
    /// The autoscaler's most recent decision and what became of it.
    last_decision: Option<Decision>,
    /// What the epoch in flight is, in words.
    in_flight: Option<String>,
    /// Launches that failed, reported by their tasks and logged on the next
    /// beat (a launch runs outside the loop, so it cannot write the log).
    launch_failures: Arc<std::sync::Mutex<Vec<(ShardId, String)>>>,
    /// The operator port, once a front end is attached.
    ops: Option<OpsPort>,
    /// The event log's revision when the last snapshot was published.
    published_revision: u64,
    /// The number of the newest event-log entry the run loop has written to
    /// standard output.
    printed_event: u64,

    // ---- the scene (the state itself is `control.scene`).
    /// What each shard last confirmed, since it (last) said hello. Only ever
    /// moves to a higher revision, and only for a shard the orchestrator
    /// knows, so nothing a peer sends can grow or churn it without bound.
    scene_confirmed: BTreeMap<ShardId, SceneState>,
    /// The current scene has to be sent to every known shard; done once, when
    /// the outbox is next drained ([`Self::drain_outbox`]).
    scene_push_pending: bool,
    /// The tick of the last scene change since startup.
    scene_changed_tick: Option<u64>,
    /// An operator asked for the committed map to be sent again; done once,
    /// when the outbox is next drained, however often it was asked for.
    map_push_pending: bool,
    /// The standard-output budget for peer-caused entries.
    peer_prints: PeerPrintBudget,
}

/// How many peer-caused event-log entries (refused hellos) may still be
/// written to standard output, and how many were not.
#[derive(Debug, Default, Clone, Copy)]
struct PeerPrintBudget {
    /// The tick the current window started at.
    window_start: u64,
    /// Lines written in the current window.
    printed: u32,
    /// [`EventLog::peer_caused_total`] when the current window started.
    total_at_window_start: u64,
    /// [`EventLog::peer_caused_total`] when the lines were last collected.
    total_at_last_look: u64,
}

impl<L: Launcher> Orchestrator<L> {
    /// Assemble an orchestrator from its parts.
    #[must_use]
    pub fn new(control: ControlPlane, launcher: Arc<L>) -> Self {
        let mut events = EventLog::default();
        if let Some(warning) = control.scene.warning() {
            // The scene file could not be used at startup: say so where the
            // operator looks, not only on standard error.
            events.push(0, EventKind::Scene, Severity::Warn, warning);
        }
        Self {
            control,
            launcher,
            tick: 0,
            latest: BTreeMap::new(),
            shard_ticks: BTreeMap::new(),
            pending_spawn: Vec::new(),
            pending_kill: Vec::new(),
            pending_merge: None,
            pending_split: None,
            outbox: Vec::new(),
            endpoints: BTreeMap::new(),
            events,
            last_heard: BTreeMap::new(),
            connected: BTreeSet::new(),
            autoscaler_paused: false,
            last_decision: None,
            in_flight: None,
            launch_failures: Arc::new(std::sync::Mutex::new(Vec::new())),
            ops: None,
            published_revision: 0,
            printed_event: 0,
            scene_confirmed: BTreeMap::new(),
            scene_push_pending: false,
            scene_changed_tick: None,
            map_push_pending: false,
            peer_prints: PeerPrintBudget::default(),
        }
    }

    /// The config the orchestrator was built with.
    #[must_use]
    pub fn config(&self) -> &OrchestratorConfig {
        &self.control.config
    }

    /// The orchestrator's current tick.
    #[must_use]
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// The published map (the only map shards may observe).
    #[must_use]
    pub fn map(&self) -> &ShardMap {
        self.control.map()
    }

    /// Drain the messages the async layer must send, in order.
    ///
    /// What an operator asked to have sent since the last drain is turned into
    /// messages here, once: the committed map (one `DirectoryReply` per owner)
    /// and the scene (one `SceneSet` with the *current* scene per known shard),
    /// however many requests there were. They are placed **behind** everything
    /// else, so a burst of operator requests can neither get ahead of epoch
    /// traffic in a shard's queue nor fill that queue: per beat a shard is
    /// offered at most one of each.
    pub fn drain_outbox(&mut self) -> Vec<(ShardId, Message)> {
        if std::mem::take(&mut self.map_push_pending) {
            self.queue_map_push();
        }
        if std::mem::take(&mut self.scene_push_pending) {
            let scene = self.scene_message();
            for to in self.scene_recipients() {
                self.outbox.push((to, scene.clone()));
            }
        }
        std::mem::take(&mut self.outbox)
    }

    /// Drain the spawn specs the loop staged this tick (a split's child).
    pub fn take_pending_spawn(&mut self) -> Vec<SpawnSpec> {
        std::mem::take(&mut self.pending_spawn)
    }

    /// Drain the shard ids the loop staged for killing (a committed merge's
    /// child, the child of a split that aborted, an admin kill).
    pub fn take_pending_kill(&mut self) -> Vec<ShardId> {
        std::mem::take(&mut self.pending_kill)
    }

    /// The shard the merge in flight will remove once its epoch commits.
    #[must_use]
    pub fn pending_merge(&self) -> Option<ShardId> {
        self.pending_merge
    }

    /// The latest load report held for `shard`, if it is still fresh.
    #[must_use]
    pub fn load_report(&self, shard: ShardId) -> Option<&LoadReport> {
        self.latest.get(&shard)
    }

    // ------------------------------------------------------------- ingestion

    /// Handle a `ShardHello`: allocate/confirm the id and region, remember the
    /// endpoint, and return the `ShardWelcome` to send back.
    ///
    /// This is the orchestrator-owns-allocation path (closes Phase-2 MAJOR 3):
    /// the id comes from [`crate::ids`], never from argv.
    pub fn handle_shard_hello(&mut self, hello: &ShardHello) -> Result<ShardWelcome, String> {
        match self.control.handle_shard_hello(hello) {
            Ok(welcome) => {
                self.endpoints.insert(welcome.shard, hello.endpoint);
                self.heard(welcome.shard, HeardKind::Hello);
                // A shard that (re-)registers starts without a scene: whatever
                // it confirmed before is void until it confirms again.
                self.scene_confirmed.remove(&welcome.shard);
                self.log(
                    EventKind::Shard,
                    Severity::Good,
                    format!(
                        "{} registered from {} for region {}",
                        welcome.shard,
                        hello.endpoint,
                        node_path(welcome.region.node)
                    ),
                );
                Ok(welcome)
            }
            Err(reason) => {
                // Anybody who can reach the control port can cause this entry,
                // with any text: it goes into the ring kept for such entries so
                // it can never push out an operator's or an epoch's.
                self.events.push_peer_caused(
                    self.tick,
                    EventKind::Shard,
                    Severity::Warn,
                    format!("hello from {} refused: {reason}", hello.endpoint),
                );
                Err(reason)
            }
        }
    }

    /// Append an entry to the event log, stamped with the current tick.
    fn log(&mut self, kind: EventKind, severity: Severity, text: String) -> u64 {
        self.events.push(self.tick, kind, severity, text)
    }

    /// Remember that `shard` was just heard from.
    fn heard(&mut self, shard: ShardId, what: HeardKind) {
        self.last_heard.insert(
            shard,
            Heard {
                tick: self.tick,
                what,
            },
        );
    }

    /// Ingest a `LoadReport`: remember it as the shard's latest and record the
    /// orchestrator tick it arrived at (its freshness, and the observer tick an
    /// epoch's `activation_tick` is derived from).
    ///
    /// A report naming a shard that owns no region in the published map is
    /// dropped: it is from a shard that was merged away or killed (its last
    /// reports can still be in flight), or from nobody the cluster knows. Kept,
    /// it would be replayed to the autoscaler every tick and acted on.
    pub fn handle_load_report(&mut self, report: &LoadReport) {
        if !self.owns_a_region(report.shard) {
            return;
        }
        self.latest.insert(report.shard, *report);
        self.shard_ticks.insert(report.shard, self.tick);
        self.heard(report.shard, HeardKind::Report);
    }

    /// Whether `shard` owns a region in the published map.
    fn owns_a_region(&self, shard: ShardId) -> bool {
        self.control.map().regions.iter().any(|r| r.shard == shard)
    }

    /// Stop believing anything about `shard`'s load: drop its report and its
    /// autoscaler state. Its endpoint is kept — it may still own a region.
    fn forget_reports(&mut self, shard: ShardId) {
        self.latest.remove(&shard);
        self.shard_ticks.remove(&shard);
        self.control.autoscaler.forget(shard);
    }

    /// `shard` has left the cluster (merged away, or a child whose split
    /// aborted): forget its load and where it was reachable.
    fn forget_shard(&mut self, shard: ShardId) {
        self.forget_reports(shard);
        self.endpoints.remove(&shard);
        self.last_heard.remove(&shard);
        self.scene_confirmed.remove(&shard);
    }

    /// Drop every report that has not been refreshed for
    /// [`REPORT_EXPIRY_TICKS`]. A shard reports every
    /// [`crate::config::LOAD_REPORT_INTERVAL_TICKS`]; one that has been silent
    /// for several beats is not "still as loaded as it last said" — it is gone
    /// or stuck, and neither is a reason to split or merge it.
    fn expire_stale_reports(&mut self) {
        let now = self.tick;
        let stale: Vec<ShardId> = self
            .shard_ticks
            .iter()
            .filter(|(_, seen)| now.saturating_sub(**seen) > REPORT_EXPIRY_TICKS)
            .map(|(shard, _)| *shard)
            .collect();
        for shard in stale {
            self.forget_reports(shard);
        }
    }

    /// Record an `EpochAck` from `shard`. When the last required ack lands the
    /// publisher commits; the commit message is queued for every shard and the
    /// autoscaler is told an action completed (cooldown).
    pub fn handle_epoch_ack(&mut self, shard: ShardId, epoch: u64) {
        // An ack names its sender itself, so only an id the orchestrator
        // already knows is remembered as "heard from": a stray ack must not be
        // able to grow the table.
        if self.endpoints.contains_key(&shard) || self.owns_a_region(shard) {
            self.heard(shard, HeardKind::Ack);
        }
        if let Some(commit) = self.control.publisher.ack(shard, epoch, self.tick) {
            let what = self.in_flight.take().unwrap_or_else(|| "map change".into());
            self.log(
                EventKind::Epoch,
                Severity::Good,
                format!("epoch {epoch} committed: {what}"),
            );
            // The epoch in flight settled as a commit: a split's child is now a
            // region owner like any other, and a merge's child owns nothing.
            self.pending_split = None;
            let merged = self.pending_merge.take();
            self.queue_outbound(&commit);
            // Then push the committed map with every known shard's real endpoint
            // (plan §3.3, F9). The commit names ownership only; this is what lets
            // an existing shard dial a child it has never heard of, and what makes
            // the `Redirect` it hands a client point at the child's real address.
            // It rides the same per-shard FIFO as the commit, so a shard always
            // sees the commit first.
            self.control.publisher.set_endpoints(&self.endpoints);
            let push = Outbound {
                to: commit.to.clone(),
                message: Message::DirectoryReply(space_protocol::DirectoryReply {
                    shard_map: self.control.publisher.published().clone(),
                    tick: self.tick,
                }),
            };
            self.queue_outbound(&push);
            if let Some(gone) = merged {
                // The merged shard is not in the new map, so the fan-out above
                // skipped it. Tell it too (it had the prepare staged; now it
                // learns it owns nothing), and only *then* stage its kill: the
                // loop delivers the outbox before it runs the launcher work.
                self.outbox.push((gone, commit.message.clone()));
                self.outbox.push((gone, push.message.clone()));
                self.pending_kill.push(gone);
                self.forget_shard(gone);
            }
            // The epoch settled: the publisher returns to Idle so the next
            // change can be proposed after the minimum interval.
            self.control.publisher.reset();
        }
    }

    /// Record an `EpochAck` carried on the wire (the shard fills `shard`).
    pub fn handle_epoch_ack_message(&mut self, ack: &EpochAck) {
        self.handle_epoch_ack(ack.shard, ack.epoch);
    }

    // ----------------------------------------------------------------- scene

    /// The cluster's current scene and its revision.
    #[must_use]
    pub fn scene(&self) -> SceneState {
        self.control.scene.current()
    }

    /// The scene and revision `shard` last confirmed, if it has confirmed one
    /// since it (last) registered.
    #[must_use]
    pub fn scene_confirmed_by(&self, shard: ShardId) -> Option<SceneState> {
        self.scene_confirmed.get(&shard).copied()
    }

    /// The `SceneSet` that tells a shard the current scene.
    #[must_use]
    pub fn scene_message(&self) -> Message {
        Message::SceneSet(self.scene())
    }

    /// Everyone the scene is sent to: every shard that has said hello or holds
    /// a control link. (A shard the map names but that never registered has no
    /// link to receive it on; it is handed the scene when it registers.)
    fn scene_recipients(&self) -> BTreeSet<ShardId> {
        self.endpoints
            .keys()
            .chain(self.connected.iter())
            .copied()
            .collect()
    }

    /// Record a `SceneAck`: `ack.shard` runs `ack.applied`.
    ///
    /// An ack names its own sender (like an `EpochAck`), so it is only
    /// believed about a shard the orchestrator knows, and it only ever moves
    /// what that shard confirmed to a **higher** revision. Together these keep
    /// the table — and the event log, which gets one "confirmed" entry per
    /// shard per revision — bounded whatever a peer sends. The answer is for
    /// the operator view only; no decision reads it except
    /// [`Self::remind_scene_laggards`], which stops reminding.
    pub fn handle_scene_ack(&mut self, ack: &SceneAck) {
        let shard = ack.shard;
        if !(self.endpoints.contains_key(&shard) || self.connected.contains(&shard)) {
            return;
        }
        self.heard(shard, HeardKind::SceneAck);
        let before = self.scene_confirmed.get(&shard).copied();
        if before.is_some_and(|held| ack.applied.revision <= held.revision) {
            return; // a repeat, or an older answer arriving late
        }
        self.scene_confirmed.insert(shard, ack.applied);
        let current = self.scene();
        if ack.applied == current {
            self.log(
                EventKind::Shard,
                Severity::Good,
                format!("{shard} confirmed scene {}", ack.applied),
            );
        } else if ack.applied.revision >= current.revision
            && before.is_none_or(|held| held.revision <= current.revision)
        {
            // A revision this orchestrator never issued (or its own revision
            // with another scene). Said once; the page keeps showing it.
            self.log(
                EventKind::Shard,
                Severity::Warn,
                format!(
                    "{shard} reports scene {}, which is not the cluster's ({current})",
                    ack.applied
                ),
            );
        }
    }

    /// Offer the current scene again to every **connected** shard that has not
    /// confirmed the current revision — on every
    /// [`SCENE_RESEND_INTERVAL_TICKS`]th tick, nothing otherwise.
    ///
    /// The run loop calls this on each beat. A `SceneSet` (or its ack) can be
    /// lost — a full queue drops rather than waits — and this is what repairs
    /// it; a shard that is merely slow to answer is asked again, which is
    /// harmless (the shard acks without re-applying). A shard that reports a
    /// *higher* revision is left alone: asking again would change nothing.
    pub fn remind_scene_laggards(&mut self) {
        if !self.tick.is_multiple_of(SCENE_RESEND_INTERVAL_TICKS) {
            return;
        }
        let current = self.scene();
        let behind: Vec<ShardId> = self
            .connected
            .iter()
            .copied()
            .filter(|shard| {
                self.scene_confirmed
                    .get(shard)
                    .is_none_or(|held| held.revision < current.revision)
            })
            .collect();
        for shard in behind {
            self.outbox.push((shard, Message::SceneSet(current)));
        }
    }

    /// An operator's scene change: persist, raise the revision, mark for
    /// sending. Returns what happened, in words.
    ///
    /// Deliberately **no** look at the epoch publisher: the map does not
    /// change, so there is nothing to wait for and nothing to hold up.
    ///
    /// # Errors
    ///
    /// The new scene could not be persisted. Nothing changed then: the shards
    /// keep the scene they have, and so does a restart.
    fn set_scene(&mut self, scene: Scene) -> Result<String, String> {
        let before = self.control.scene.current();
        let told = list_shards(&self.scene_recipients().into_iter().collect::<Vec<_>>());
        if self.control.scene.scene() == scene {
            // Nothing to change. Sending it again is free and is what an
            // operator pressing the button twice is asking for.
            self.scene_push_pending = true;
            return Ok(format!(
                "the scene is already {before}; it is sent again to {told}"
            ));
        }
        let now = self.control.scene.set(scene).map_err(|err| {
            format!(
                "the scene could not be saved ({err}), so it was not changed: it is still {before}"
            )
        })?;
        self.scene_changed_tick = Some(self.tick);
        self.scene_push_pending = true;
        Ok(format!(
            "scene {} -> {} (revision {}), sent to {told}",
            before.scene, now.scene, now.revision
        ))
    }

    /// Handle an `AdminCommand`, returning the reply to send.
    ///
    /// `ForceSplit`/`ForceMerge`/`KillShard` are driven through the same code
    /// paths the autoscaler uses, so an admin verb and an autoscale decision go
    /// through the same plan, epoch and launcher steps. A verb that cannot be
    /// carried out (an epoch is in flight, the region cannot split, the shard
    /// still has players) is answered `Refused` with the reason — it used to be
    /// `Accepted` and silently dropped.
    ///
    /// `ForceSplit` names no octant, so it carves the first free one counting
    /// from octant 0 (the autoscaler starts at the region's centre octant
    /// instead, because that is where joining ships spawn).
    pub fn handle_admin(&mut self, command: &AdminCommand) -> AdminReply {
        let outcome = match command {
            AdminCommand::ForceSplit(shard) => self.begin_split(*shard, Carve::From(0)),
            AdminCommand::ForceMerge(shard) => self.begin_merge(*shard),
            AdminCommand::KillShard(shard) => {
                self.pending_kill.push(*shard);
                // A killed shard reports nothing more; its last report must not
                // go on driving the autoscaler.
                self.forget_reports(*shard);
                Ok(format!("{shard} is staged for killing"))
            }
            other => return self.control.handle_admin(other),
        };
        match outcome {
            Ok(started) => {
                self.log(
                    EventKind::Operator,
                    Severity::Info,
                    format!("admin {command:?}: accepted: {started}"),
                );
                AdminReply::Accepted {
                    command: command.clone(),
                }
            }
            Err(reason) => {
                self.log(
                    EventKind::Operator,
                    Severity::Warn,
                    format!("admin {command:?}: refused: {reason}"),
                );
                AdminReply::Refused {
                    command: command.clone(),
                    reason,
                }
            }
        }
    }

    // ----------------------------------------------------------------- tick

    /// Advance the orchestrator's clock by one tick and run the control beat:
    /// the epoch deadline, then the autoscaler over the latest report per shard.
    ///
    /// This is the synchronous heart of the loop (plan §3.6). It never touches a
    /// socket or a wall clock, so a test drives it with a `for` loop.
    pub fn tick(&mut self) {
        self.tick += 1;
        // 1. Epoch deadline: an un-acked prepare aborts and the map is untouched.
        // (Who still owed an ack is read first, for the log only.)
        let missing = self.missing_acks();
        if let Some(abort) = self.control.publisher.timeout(self.tick) {
            self.queue_outbound(&abort);
            self.control.publisher.reset();
            let what = self.in_flight.take().unwrap_or_else(|| "map change".into());
            let epoch = self.control.map().epoch + 1;
            self.log(
                EventKind::Epoch,
                Severity::Warn,
                format!(
                    "epoch {epoch} aborted at its deadline, the map is unchanged: {what}; \
                     no ack from {}",
                    list_shards(&missing)
                ),
            );
            // Nothing the aborted epoch planned happened. A merge's shard still
            // owns its region, so it is *not* killed; a split's child owns
            // nothing, so it is — after the abort above has been delivered.
            self.pending_merge = None;
            if let Some(child) = self.pending_split.take() {
                self.pending_kill.push(child);
                self.forget_shard(child);
            }
        }
        // 2. Autoscaler: feed the latest *fresh* report per shard, get at most
        // one action.
        self.expire_stale_reports();
        if self.autoscaler_paused {
            // An operator paused it: reports are still kept (and expired) and
            // shown, but none is fed, so no dwell accrues and nothing fires.
            return;
        }
        let reports: Vec<LoadReport> = self.latest.values().copied().collect();
        if reports.is_empty() {
            return;
        }
        if let Some(action) = self.control.autoscaler.on_reports(&reports, self.tick) {
            let (decided, outcome) = match action {
                // The autoscaler's octant is a placeholder (it sees load, not
                // positions); the split starts at the region's centre octant.
                ScaleAction::Split { shard, .. } => (
                    format!("split {shard}"),
                    self.begin_split(shard, Carve::FromCentre),
                ),
                ScaleAction::Merge { shard } => (format!("merge {shard}"), self.begin_merge(shard)),
            };
            match &outcome {
                Ok(started) => {
                    self.log(
                        EventKind::Autoscaler,
                        Severity::Info,
                        format!("{decided}: {started}"),
                    );
                }
                Err(why) => {
                    eprintln!("space-orchestrator: {action} was not carried out: {why}");
                    self.log(
                        EventKind::Autoscaler,
                        Severity::Warn,
                        format!("{decided} was not carried out: {why}"),
                    );
                }
            }
            self.last_decision = Some(Decision {
                tick: self.tick,
                action: decided,
                outcome,
            });
        }
    }

    /// The shards that still owe an ack for the prepare in flight (empty when
    /// nothing is in flight).
    fn missing_acks(&self) -> Vec<ShardId> {
        match self.control.publisher.phase() {
            Phase::Preparing { awaiting, .. } => awaiting.iter().copied().collect(),
            _ => Vec::new(),
        }
    }

    // ------------------------------------------------- autoscale -> topology

    /// Turn a split decision into a topology plan + epoch proposal + a staged
    /// child spawn.
    ///
    /// `carve` says where the search for an octant to carve starts:
    /// [`Carve::FromCentre`] means the region's centre octant (where a shard
    /// spawns joining ships, so the first split of a fresh region moves the
    /// players it was meant to relieve), [`Carve::From`] an explicit
    /// preference. Either way the octant carved is the first one, counting
    /// round from there, whose plan validates — an octant already carved out
    /// for another shard is skipped, so a region can be split again and again
    /// until none is left ([`topology::pick_split_octant`]).
    /// [`Carve::Exactly`] is an operator naming one octant: that one is carved
    /// or the split is refused, never a different octant than the one asked for.
    ///
    /// The child's id is allocated only once a valid plan exists: a split that
    /// cannot happen (region too small, nothing left to carve) burns no id and
    /// writes nothing to the counter file.
    ///
    /// Returns what was started, in words.
    ///
    /// # Errors
    ///
    /// Why the split was not started. Apart from "an epoch is in flight" the
    /// shard is put on cooldown first (a shard that cannot be split is not
    /// helped by retrying every tick); the caller logs or reports the reason.
    fn begin_split(&mut self, shard: ShardId, carve: Carve) -> Result<String, String> {
        if !self
            .control
            .publisher
            .can_propose(self.tick, MIN_EPOCH_INTERVAL_TICKS)
        {
            return Err("an epoch change is in flight or was committed too recently".into());
        }
        // The map is full: no split, whoever asks (the autoscaler included).
        if let Some(full) = self.region_limit_refusal() {
            self.control.autoscaler.note_action(shard, self.tick);
            return Err(full);
        }
        let picked = match carve {
            Carve::FromCentre => {
                let first = topology::region_of(self.control.map(), shard)
                    .map_or(0, |region| topology::centre_octant(region.node));
                topology::pick_split_octant(self.control.map(), shard, first)
            }
            Carve::From(first) => topology::pick_split_octant(self.control.map(), shard, first),
            // The placeholder owner stands in for the id allocated below: the
            // owner is irrelevant to whether a plan validates.
            Carve::Exactly(octant) => {
                topology::split_plan(self.control.map(), shard, octant, ShardId::NONE)
                    .map(|_| octant)
            }
        };
        let octant = match picked {
            Ok(octant) => octant,
            Err(err) => {
                self.control.autoscaler.note_action(shard, self.tick);
                return Err(err.to_string());
            }
        };
        let new_id = match self.control.ids.allocate() {
            Ok(id) => id,
            Err(err) => {
                // The id space is exhausted or the counter cannot be written:
                // nothing can be spawned. Cool down so we do not retry every tick.
                self.control.autoscaler.note_action(shard, self.tick);
                return Err(err.to_string());
            }
        };
        let plan = match topology::split_plan(self.control.map(), shard, octant, new_id) {
            Ok(plan) => plan,
            Err(err) => {
                self.control.autoscaler.note_action(shard, self.tick);
                return Err(err.to_string());
            }
        };
        let observer_ticks: Vec<(ShardId, u64)> =
            self.shard_ticks.iter().map(|(s, t)| (*s, *t)).collect();
        let Some(prepare) = self.control.publisher.propose(
            plan.map.clone(),
            vec![plan.change],
            &observer_ticks,
            self.tick,
        ) else {
            // Lost the race to an in-flight epoch (can_propose said Idle but the
            // publisher disagreed); cool down so we do not spin.
            self.control.autoscaler.note_action(shard, self.tick);
            return Err("the publisher refused the proposal".into());
        };
        self.queue_outbound(&prepare);
        self.control.autoscaler.note_action(shard, self.tick);
        // Stage the child spawn: the async layer runs the launcher and the child
        // then registers with its own ShardHello (and is handed the prepare it
        // still owes an ack for).
        let parent_endpoint = self
            .endpoints
            .get(&shard)
            .copied()
            .unwrap_or(plan.map.regions[0].endpoint);
        self.pending_spawn.push(SpawnSpec {
            shard: new_id,
            region: space_protocol::shard::ShardRegion::new(plan.child, new_id, parent_endpoint),
            cluster: self.control.config.cluster.clone(),
            control_addr: self.control.config.control_addr(),
            host_port: 0,
        });
        self.pending_split = Some(new_id);
        let child = node_path(plan.child);
        self.in_flight = Some(format!(
            "split of {shard}, region {child} (octant {octant}, {}) goes to new {new_id}",
            octant_signs(u8::try_from(octant % 8).unwrap_or(0)),
        ));
        Ok(format!(
            "epoch {} proposed, region {child} goes to new {new_id}",
            plan.map.epoch
        ))
    }

    /// Turn a merge decision into a topology plan + epoch proposal. The merged
    /// shard is killed when (and only if) that epoch commits.
    ///
    /// **Only a shard with no players is merged away.** A merge removes the
    /// shard's region from the map and then the shard itself; there is no step
    /// in which its players are handed to the parent (once the merge commits
    /// the parent no longer even knows the shard, so it would refuse the
    /// offers). "Cold" to the autoscaler means a load at or below the merge
    /// threshold — up to a third of a full shard — so acting on that alone
    /// dropped every one of those players. The shard's own fresh report has to
    /// say `players == 0`; a shard that has not reported is not known to be
    /// empty and is left alone.
    ///
    /// Returns what was started, in words.
    ///
    /// # Errors
    ///
    /// Why the merge was not started (see [`Self::begin_split`] for the
    /// cooldown rule).
    fn begin_merge(&mut self, shard: ShardId) -> Result<String, String> {
        if !self
            .control
            .publisher
            .can_propose(self.tick, MIN_EPOCH_INTERVAL_TICKS)
        {
            return Err("an epoch change is in flight or was committed too recently".into());
        }
        let players = self.latest.get(&shard).map(|report| report.players);
        if players != Some(0) {
            self.control.autoscaler.note_action(shard, self.tick);
            return Err(match players {
                Some(n) => {
                    format!("{shard} still has {n} player(s); only an empty shard is merged")
                }
                None => {
                    format!("no fresh load report from {shard}, so it is not known to be empty")
                }
            });
        }
        let plan = match topology::merge_plan(self.control.map(), shard) {
            Ok(plan) => plan,
            Err(err) => {
                self.control.autoscaler.note_action(shard, self.tick);
                return Err(err.to_string());
            }
        };
        let observer_ticks: Vec<(ShardId, u64)> =
            self.shard_ticks.iter().map(|(s, t)| (*s, *t)).collect();
        let Some(prepare) = self.control.publisher.propose(
            plan.map.clone(),
            vec![plan.change],
            &observer_ticks,
            self.tick,
        ) else {
            self.control.autoscaler.note_action(shard, self.tick);
            return Err("the publisher refused the proposal".into());
        };
        self.queue_outbound(&prepare);
        self.control.autoscaler.note_action(shard, self.tick);
        // The merged child's region folds back into the parent — when the epoch
        // commits. Until then the shard owns its region and keeps running; it
        // is staged for killing by the commit (`handle_epoch_ack`), and left
        // alone if the epoch aborts (`tick`).
        self.pending_merge = Some(shard);
        let (child, into) = (node_path(plan.child), plan.into);
        self.in_flight = Some(format!(
            "merge of {shard}, region {child} goes back to {into}"
        ));
        Ok(format!(
            "epoch {} proposed, region {child} goes back to {into}",
            plan.map.epoch
        ))
    }

    /// Fan an [`Outbound`] out to its recipients, tagging each with the shard it
    /// goes to so the async layer can route it.
    fn queue_outbound(&mut self, outbound: &Outbound) {
        for to in &outbound.to {
            self.outbox.push((*to, outbound.message.clone()));
        }
    }

    // ------------------------------------------------------------- operator

    /// The recent-events log.
    #[must_use]
    pub fn events(&self) -> &EventLog {
        &self.events
    }

    /// Whether an operator has paused the autoscaler.
    #[must_use]
    pub fn autoscaler_paused(&self) -> bool {
        self.autoscaler_paused
    }

    /// Tell the orchestrator which shards' control links are open right now.
    ///
    /// The run loop calls this on every beat with what its front end sees
    /// ([`crate::control::ControlServer::connected`]). A shard that was
    /// connected and no longer is gets a "lost" entry in the event log; that is
    /// all — losing a link changes no ownership and proposes nothing (H15's
    /// reassignment is a later slice). The set is also what
    /// [`Self::split_refusal`] reads to turn down a split whose shard could
    /// not acknowledge it.
    pub fn note_connected(&mut self, live: impl IntoIterator<Item = ShardId>) {
        let live: BTreeSet<ShardId> = live.into_iter().collect();
        let lost: Vec<ShardId> = self.connected.difference(&live).copied().collect();
        for shard in lost {
            self.log(
                EventKind::Shard,
                Severity::Warn,
                format!("{shard} lost: its control link closed"),
            );
        }
        self.connected = live;
    }

    /// Why an operator's split of `shard` would be refused right now, or
    /// `None` when it would be accepted. `octant` is the one to carve, `None`
    /// for "the first free one from the centre".
    ///
    /// This is the check [`Self::handle_operator`] runs before it starts a
    /// split, and it changes nothing — unlike a refusal from `begin_split`
    /// itself, which puts the shard on the autoscaler's cooldown. An operator's
    /// mistake must not hold the autoscaler back.
    #[must_use]
    pub fn split_refusal(&self, shard: ShardId, octant: Option<u8>) -> Option<String> {
        let map = self.control.map();
        let Some(region) = topology::region_of(map, shard) else {
            return Some(topology::TopologyError::UnknownShard(shard).to_string());
        };
        // Which octants are free is *looked up*, not planned: this runs for
        // every region of every snapshot, on the control loop, and planning a
        // split clones and re-validates the whole map (see
        // `topology::free_octants`). `begin_split` plans, and is the authority.
        let free = match topology::free_octants(map, shard) {
            Ok(free) => free,
            Err(err) => return Some(err.to_string()),
        };
        match octant {
            Some(octant) if octant > 7 => {
                return Some(format!(
                    "octant {octant} does not exist (octants are 0 to 7)"
                ));
            }
            Some(octant) => {
                if !free.contains(&octant) {
                    return Some(format!(
                        "octant {octant} ({}) of region {} is already carved out",
                        octant_signs(octant),
                        node_path(region.node)
                    ));
                }
            }
            None => {
                if free.is_empty() {
                    return Some(topology::TopologyError::NoFreeOctant(region.node).to_string());
                }
            }
        }
        if let Some(full) = self.region_limit_refusal() {
            return Some(full);
        }
        if let Some(busy) = self.proposal_refusal() {
            return Some(busy);
        }
        if !self.connected.contains(&shard) {
            return Some(format!(
                "{shard} is not connected to the orchestrator, so it could not acknowledge the change"
            ));
        }
        None
    }

    /// Why an operator's merge of `shard` would be refused right now, or
    /// `None` when it would be accepted. Changes nothing (see
    /// [`Self::split_refusal`]).
    ///
    /// The rules are `begin_merge`'s: the region must have a parent region
    /// owned by another shard, and the shard's own fresh load report must say
    /// it has no players — a shard that has not reported is not known to be
    /// empty (H13).
    #[must_use]
    pub fn merge_refusal(&self, shard: ShardId) -> Option<String> {
        // Looked up, not planned (see `split_refusal`).
        if let Err(err) = topology::merge_target(self.control.map(), shard) {
            return Some(err.to_string());
        }
        match self.latest.get(&shard).map(|report| report.players) {
            Some(0) => {}
            Some(players) => {
                return Some(format!(
                    "{shard} still has {players} player(s); only an empty shard is merged"
                ));
            }
            None => {
                return Some(format!(
                    "no fresh load report from {shard}, so it is not known to be empty"
                ));
            }
        }
        self.proposal_refusal()
    }

    /// Why the map cannot take one more region, if it cannot: it already holds
    /// [`OrchestratorConfig::max_regions`].
    fn region_limit_refusal(&self) -> Option<String> {
        let (regions, limit) = (
            self.control.map().regions.len(),
            self.control.config.max_regions,
        );
        (regions >= limit).then(|| {
            format!(
                "the map already has {regions} region(s) and the limit is {limit} \
                 (--max-regions): merge a region away first, or restart the orchestrator with \
                 a higher limit"
            )
        })
    }

    /// Why no epoch change can be proposed at this tick, if one cannot.
    fn proposal_refusal(&self) -> Option<String> {
        if let Phase::Preparing { epoch, .. } = self.control.publisher.phase() {
            return Some(format!(
                "epoch {epoch} is still in flight; one change at a time"
            ));
        }
        if self
            .control
            .publisher
            .can_propose(self.tick, MIN_EPOCH_INTERVAL_TICKS)
        {
            return None;
        }
        let wait = self.control.publisher.last_commit_tick().map_or(0, |at| {
            (at + MIN_EPOCH_INTERVAL_TICKS).saturating_sub(self.tick)
        });
        Some(format!(
            "the last epoch was committed too recently; the next change can be proposed in \
             {wait} tick(s), about {:.1} s",
            wait as f64 / crate::config::TICK_HZ as f64
        ))
    }

    /// Carry out an operator's command and say what became of it.
    ///
    /// Synchronous and immediate: the answer is "accepted" (for a split or a
    /// merge: its epoch was proposed, through the same `begin_split` /
    /// `begin_merge` the autoscaler uses) or "refused" with the reason.
    /// Nothing is queued for later. Either way the request is recorded in the
    /// event log, and [`OperatorReply::event`] is that entry's number.
    pub fn handle_operator(&mut self, command: &OperatorCommand) -> OperatorReply {
        let outcome = match command {
            OperatorCommand::Split {
                shard,
                octant,
                expected_epoch,
            } => self.operator_split(*shard, *octant, *expected_epoch),
            OperatorCommand::Merge {
                shard,
                expected_epoch,
            } => self.operator_merge(*shard, *expected_epoch),
            OperatorCommand::PauseAutoscaler => Ok(self.set_autoscaler_paused(true)),
            OperatorCommand::ResumeAutoscaler => Ok(self.set_autoscaler_paused(false)),
            OperatorCommand::PushMap => Ok(self.push_map()),
            OperatorCommand::SetScene { scene } => self.set_scene(*scene),
        };
        let (accepted, severity, verdict, message) = match outcome {
            Ok(message) => (true, Severity::Info, "accepted", message),
            Err(message) => (false, Severity::Warn, "refused", message),
        };
        let event = self.log(
            EventKind::Operator,
            severity,
            format!("{command}: {verdict}: {message}"),
        );
        OperatorReply {
            accepted,
            message,
            event,
        }
    }

    /// An operator's split: checked without side effects first, then started
    /// through `begin_split` like an autoscaler split.
    fn operator_split(
        &mut self,
        shard: ShardId,
        octant: Option<u8>,
        expected_epoch: Option<u64>,
    ) -> Result<String, String> {
        self.epoch_is_still(expected_epoch)?;
        if let Some(why) = self.split_refusal(shard, octant) {
            return Err(why);
        }
        let carve = octant.map_or(Carve::FromCentre, |o| Carve::Exactly(usize::from(o)));
        self.begin_split(shard, carve)
    }

    /// An operator's merge: checked without side effects first, then started
    /// through `begin_merge` like an autoscaler merge.
    fn operator_merge(
        &mut self,
        shard: ShardId,
        expected_epoch: Option<u64>,
    ) -> Result<String, String> {
        self.epoch_is_still(expected_epoch)?;
        if let Some(why) = self.merge_refusal(shard) {
            return Err(why);
        }
        self.begin_merge(shard)
    }

    /// Refuse a command aimed at a map the operator was not looking at.
    fn epoch_is_still(&self, expected: Option<u64>) -> Result<(), String> {
        let now = self.control.map().epoch;
        match expected {
            Some(seen) if seen != now => Err(format!(
                "the map changed since the page was loaded (epoch {seen} then, {now} now); \
                 look again and repeat the request"
            )),
            _ => Ok(()),
        }
    }

    /// Pause or resume the autoscaler; returns what happened, in words.
    fn set_autoscaler_paused(&mut self, paused: bool) -> String {
        if self.autoscaler_paused == paused {
            return format!(
                "the autoscaler was already {}",
                if paused { "paused" } else { "active" }
            );
        }
        self.autoscaler_paused = paused;
        // A dwell timer must not span a pause (see `Autoscaler::reset_dwell`).
        self.control.autoscaler.reset_dwell();
        if paused {
            "the autoscaler is paused: it takes no action until it is resumed".into()
        } else {
            "the autoscaler is active again; every shard needs a fresh dwell".into()
        }
    }

    /// Mark the committed map for sending to every shard in it — what follows
    /// each commit (plan §3.3, F9), on request. Ownership and the epoch are
    /// untouched; a shard that already has this epoch only refreshes its
    /// peers' addresses.
    ///
    /// The messages are built when the outbox is next drained
    /// ([`Self::drain_outbox`]), once however often this is asked for: each
    /// carries the whole map, and a request per message would let a burst of
    /// requests fill a shard's queue ahead of an epoch prepare or commit.
    fn push_map(&mut self) -> String {
        self.map_push_pending = true;
        let map = self.control.map();
        let recipients: BTreeSet<ShardId> = map.regions.iter().map(|r| r.shard).collect();
        format!(
            "the committed map (epoch {}) was queued for {}",
            map.epoch,
            list_shards(&recipients.into_iter().collect::<Vec<_>>())
        )
    }

    /// Put the committed map, with every known shard's current address, into
    /// the outbox for every shard in it.
    fn queue_map_push(&mut self) {
        self.control.publisher.set_endpoints(&self.endpoints);
        let map = self.control.publisher.published().clone();
        let recipients: BTreeSet<ShardId> = map.regions.iter().map(|r| r.shard).collect();
        for to in recipients {
            self.outbox.push((
                to,
                Message::DirectoryReply(space_protocol::DirectoryReply {
                    shard_map: map.clone(),
                    tick: self.tick,
                }),
            ));
        }
    }

    /// Everything the orchestrator knows, as plain data (see [`Snapshot`]).
    #[must_use]
    pub fn snapshot(&self) -> Snapshot {
        let config = &self.control.config;
        let map = self.control.map();
        Snapshot {
            tick: self.tick,
            cluster: config.cluster.clone(),
            bind: config.bind,
            control_addr: config.control_addr(),
            launcher: config.launcher,
            state_path: config.state_path.clone(),
            root: config.root_node(),
            first_shard_id: config.first_shard_id,
            host_port_range: config.host_port_range,
            next_shard_id: self.control.ids.peek(),
            unclaimed_root_owner: self.control.unclaimed_root_owner(),
            epoch: map.epoch,
            last_commit_tick: self.control.publisher.last_commit_tick(),
            regions: self.region_views(map),
            max_regions: config.max_regions,
            shards: self.shard_views(map),
            in_flight: self.in_flight_view(),
            autoscaler: AutoscalerView {
                paused: self.autoscaler_paused,
                busy_until: self.control.autoscaler.busy_until(),
                last_decision: self.last_decision.clone(),
            },
            scene: SceneView {
                scene: self.control.scene.scene(),
                revision: self.control.scene.revision(),
                changed_tick: self.scene_changed_tick,
                path: self.control.scene.path().map(std::path::Path::to_path_buf),
            },
            events: self.events.newest_first().cloned().collect(),
        }
    }

    /// The map's regions, each enclosing region before the ones carved out of it.
    fn region_views(&self, map: &ShardMap) -> Vec<RegionView> {
        let is_region = |node| map.regions.iter().any(|r| r.node == node);
        let mut keyed: Vec<(Vec<u8>, RegionView)> = map
            .regions
            .iter()
            .map(|region| {
                // The node's address as octants from the octree root down.
                // Sorting by it puts a region right before everything carved
                // out of it, and siblings in octant order.
                let mut path = Vec::with_capacity(usize::from(region.node.depth));
                let mut nesting = 0u8;
                let mut node = region.node;
                while let Some(parent) = space_world::octree::parent(node) {
                    path.push(octant_of(node).unwrap_or(0));
                    if is_region(parent) {
                        nesting = nesting.saturating_add(1);
                    }
                    node = parent;
                }
                path.reverse();
                (path, self.region_view(map, region, nesting))
            })
            .collect();
        keyed.sort_by(|a, b| a.0.cmp(&b.0));
        keyed.into_iter().map(|(_, view)| view).collect()
    }

    fn region_view(&self, map: &ShardMap, region: &ShardRegion, nesting: u8) -> RegionView {
        let shard = region.shard;
        // A split or merge acts on a shard's deepest region; any other region
        // the same shard owns cannot be acted on directly.
        let acted_on = topology::region_of(map, shard).is_some_and(|r| r.node == region.node);
        let (free_octants, split_refusal, merge_refusal) = if acted_on {
            // All three are lookups in the map as it is; none plans a change
            // (a snapshot is built on the control loop, several times a second).
            (
                topology::free_octants(map, shard).unwrap_or_default(),
                self.split_refusal(shard, None),
                self.merge_refusal(shard),
            )
        } else {
            let why = format!("a change to {shard} acts on its innermost region, not this one");
            (Vec::new(), Some(why.clone()), Some(why))
        };
        // Two endpoints in the map are placeholders, not addresses: the
        // sentinel a split gives its child until the child says hello, and the
        // control port `boot` puts on the root until a shard claims it.
        let placeholder = region.endpoint == ShardRegion::UNRESOLVED_ENDPOINT
            || self.control.unclaimed_root_owner() == Some(shard);
        let parent = space_world::octree::parent(region.node)
            .filter(|parent| map.regions.iter().any(|r| r.node == *parent));
        RegionView {
            node: region.node,
            shard,
            endpoint: (!placeholder).then_some(region.endpoint),
            nesting,
            parent,
            octant: parent.and_then(|_| octant_of(region.node)),
            free_octants,
            split_refusal,
            merge_refusal,
        }
    }

    /// Every shard the orchestrator knows of: the map's owners, the child of a
    /// split in flight, and anyone who said hello or holds a control link.
    fn shard_views(&self, map: &ShardMap) -> Vec<ShardView> {
        let mut ids: BTreeSet<ShardId> = map.regions.iter().map(|r| r.shard).collect();
        ids.extend(self.endpoints.keys().copied());
        ids.extend(self.connected.iter().copied());
        ids.extend(self.pending_split);
        ids.into_iter()
            .map(|shard| {
                let regions: Vec<_> = map
                    .regions
                    .iter()
                    .filter(|r| r.shard == shard)
                    .map(|r| r.node)
                    .collect();
                let role = if self.pending_split == Some(shard) {
                    ShardRole::SplitChild
                } else if self.pending_merge == Some(shard) {
                    ShardRole::Merging
                } else if self.control.unclaimed_root_owner() == Some(shard) {
                    ShardRole::ReservedRoot
                } else if regions.is_empty() {
                    ShardRole::Unowned
                } else {
                    ShardRole::Owner
                };
                let report = self.latest.get(&shard).map(|report| ReportView {
                    tick: self.shard_ticks.get(&shard).copied().unwrap_or(self.tick),
                    players: report.players,
                    tick_p95_ms: report.tick_p95_ms,
                    reported_load: report.load,
                    load_metric: crate::autoscale::load_metric(report),
                });
                ShardView {
                    shard,
                    role,
                    connected: self.connected.contains(&shard),
                    registered: self.endpoints.contains_key(&shard),
                    advertised: self.endpoints.get(&shard).copied(),
                    regions,
                    last_heard: self.last_heard.get(&shard).copied(),
                    report,
                    scale: self.control.autoscaler.state_of(shard),
                    scene: self.scene_confirmed.get(&shard).copied(),
                }
            })
            .collect()
    }

    fn in_flight_view(&self) -> Option<EpochInFlight> {
        let Phase::Preparing {
            epoch,
            activation_tick,
            deadline,
            awaiting,
            recipients,
        } = self.control.publisher.phase()
        else {
            return None;
        };
        Some(EpochInFlight {
            epoch: *epoch,
            what: self
                .in_flight
                .clone()
                .unwrap_or_else(|| "map change".into()),
            activation_tick: *activation_tick,
            deadline_tick: *deadline,
            recipients: recipients
                .iter()
                .map(|shard| (*shard, !awaiting.contains(shard)))
                .collect(),
            changes: self.control.publisher.outstanding_changes().to_vec(),
        })
    }

    /// Attach a front end: returns the [`OpsHandle`] through which it reads
    /// snapshots and offers [`OperatorCommand`]s to the run loop.
    ///
    /// Until this is called the loop publishes nothing and accepts no operator
    /// command. Calling it again replaces the previous attachment (its handles
    /// then see a stopped loop).
    pub fn attach_ops(&mut self) -> OpsHandle {
        let (handle, port) = crate::ops::channel(self.snapshot());
        self.published_revision = self.events.revision();
        self.ops = Some(port);
        handle
    }

    /// Publish a fresh snapshot to the attached front end, if there is one and
    /// there is a reason to: `force`, a new event-log entry, or the regular
    /// [`SNAPSHOT_INTERVAL_TICKS`] beat.
    fn publish(&mut self, force: bool) {
        let Some(port) = &self.ops else {
            return;
        };
        let revision = self.events.revision();
        if force
            || revision != self.published_revision
            || self.tick.is_multiple_of(SNAPSHOT_INTERVAL_TICKS)
        {
            port.publish(self.snapshot());
            self.published_revision = revision;
        }
    }

    /// Answer one operator request from the queue.
    fn answer_operator(&mut self, request: OperatorRequest) {
        let reply = self.handle_operator(&request.command);
        // Publish before answering: the page the front end renders next
        // already shows what the command did.
        self.publish(true);
        // The asker may have given up waiting; that is its business.
        let _ = request.reply.send(reply);
    }

    // ------------------------------------------------------------- async loop

    /// Bind the control endpoint on the configured address. Split from
    /// [`Self::run_with`] so a caller (a test binding port 0) can read the real
    /// address from [`crate::control::ControlServer::local_addr`] before the loop
    /// starts.
    ///
    /// # Errors
    ///
    /// [`std::io::Error`] if the control endpoint cannot be bound.
    pub fn bind(&self) -> Result<Arc<crate::control::ControlServer>, std::io::Error> {
        Ok(Arc::new(crate::control::ControlServer::bind(
            self.control.config.bind,
        )?))
    }

    /// A liveness probe over the launcher (H15's launcher half): report whether
    /// `shard` is still running. Exposed so the loop (and tests) can reconcile.
    ///
    /// # Errors
    ///
    /// A launcher-specific failure.
    pub async fn poll(&self, shard: ShardId) -> Result<Poll, crate::launcher::LaunchError> {
        self.launcher.poll(shard).await
    }
}

/// The async half. `L: 'static` because a launch runs in its own task (see
/// [`Orchestrator::run_with`]).
impl<L: Launcher + 'static> Orchestrator<L> {
    /// The event loop (plan §3.6): bind the control endpoint, accept shard
    /// connections, and on each tick beat run the synchronous [`Self::tick`],
    /// deliver the outbox, and execute the staged launcher work.
    ///
    /// The loop only moves messages and drives the launcher; every decision is
    /// made by the synchronous methods above. The clock is a wall-clock ticker
    /// here (the production beat); tests drive [`Self::tick`] directly instead.
    ///
    /// # Errors
    ///
    /// [`std::io::Error`] if the control endpoint cannot be bound.
    pub async fn run(&mut self) -> Result<(), std::io::Error> {
        let server = self.bind()?;
        self.run_with(server).await
    }

    /// Run the event loop over an already-bound `server` until it closes.
    ///
    /// Nothing in the loop waits on a peer: messages are *offered* to a shard's
    /// queue ([`crate::control::ControlServer::send_to`]), and a launch runs in
    /// its own task. The second matters for more than latency — a launched
    /// shard registers by sending a `ShardHello` that only **this loop** can
    /// answer, so awaiting the launch here would wait on ourselves.
    ///
    /// # Errors
    ///
    /// Currently infallible; the `Result` keeps [`Self::run`]'s signature.
    pub async fn run_with(
        &mut self,
        server: Arc<crate::control::ControlServer>,
    ) -> Result<(), std::io::Error> {
        use crate::control::Inbound;
        use std::time::Duration;

        let accept = Arc::clone(&server).spawn_accept();
        let mut clock = tokio::time::interval(Duration::from_secs_f64(
            1.0 / f64::from(crate::config::TICK_HZ as u32),
        ));
        clock.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                inbound = server.next_inbound() => {
                    match inbound {
                        Some(Inbound::Hello { hello, reply }) => {
                            self.answer_hello(&server, &hello, &reply).await;
                        }
                        Some(Inbound::Report(report)) => self.handle_load_report(&report),
                        Some(Inbound::Ack(ack)) => self.handle_epoch_ack_message(&ack),
                        Some(Inbound::SceneAck(ack)) => self.handle_scene_ack(&ack),
                        None => break, // every sender gone
                    }
                }
                // An operator's command, taken between two beats like any
                // other input and answered at once (never while a beat runs,
                // so it sees and leaves a consistent state).
                request = next_operator_request(&mut self.ops) => {
                    match request {
                        Some(request) => self.answer_operator(request),
                        // Every handle is gone: nobody is attached any more.
                        None => self.ops = None,
                    }
                }
                _ = clock.tick() => self.beat(&server).await,
            }
        }
        accept.abort();
        server.close();
        Ok(())
    }

    /// Answer one `ShardHello` on the connection it arrived on.
    ///
    /// `reply` is that connection's own outbound queue, created empty a moment
    /// ago, so offering the answer to it never waits and never fails for lack
    /// of room.
    async fn answer_hello(
        &mut self,
        server: &crate::control::ControlServer,
        hello: &ShardHello,
        reply: &tokio::sync::mpsc::Sender<Message>,
    ) {
        match self.handle_shard_hello(hello) {
            Ok(welcome) => {
                let shard = welcome.shard;
                server.register(shard, reply.clone()).await;
                self.connected.insert(shard);
                let _ = reply.try_send(Message::ShardWelcome(welcome));
                // The cluster's scene, right behind the welcome: a shard that
                // registers (or re-registers) is never left to guess, and never
                // has to wait for the next change to learn the current one.
                let _ = reply.try_send(self.scene_message());
                // A split names its child as a prepare recipient before the
                // child exists, so the prepare published at propose time never
                // reached it. Now that it has registered, hand it the prepare it
                // still owes an ack for (behind the welcome, on the same FIFO) —
                // without this no split could ever collect its last ack.
                if let Some(prepare) = self.control.publisher.outstanding_prepare_for(shard) {
                    let _ = reply.try_send(prepare);
                }
            }
            Err(reason) => {
                // Say why, on the hello connection, with the existing `Reject`
                // message (no protocol change): the shard fails fast with the
                // reason instead of timing out. (The refusal is in the event
                // log, which is what reaches standard output — at a bounded
                // rate, because anybody can cause it.)
                let _ = reply.try_send(Message::Reject(space_protocol::Reject {
                    reason: space_protocol::RejectReason::Other(reason),
                }));
            }
        }
    }

    /// One tick beat: the synchronous [`Self::tick`], then its consequences in
    /// order — epoch messages out, launches started, kills carried out.
    async fn beat(&mut self, server: &crate::control::ControlServer) {
        self.tick();
        // Once a second: the scene again, for whoever has not confirmed it.
        self.remind_scene_laggards();
        // Deliver the epoch messages the tick (and the inbound handlers since
        // the last beat) produced. Before the kills below: a shard that is about
        // to be killed is told the outcome of its epoch first.
        for (to, message) in self.drain_outbox() {
            server.send_to(to, message).await;
        }
        // Launch the children a split staged, each in its own task (see
        // `run_with`). The child boots, registers with its own ShardHello, is
        // handed the outstanding prepare and acks it; a launch that fails
        // leaves the prepare un-acked, so the epoch aborts at its deadline.
        for mut spec in self.take_pending_spawn() {
            if spec.control_addr.port() == 0 {
                // The control endpoint was bound on port 0 ("any"): the child
                // has to dial the port that was actually bound.
                if let Ok(bound) = server.local_addr() {
                    spec.control_addr.set_port(bound.port());
                }
            }
            let launcher = Arc::clone(&self.launcher);
            let failures = Arc::clone(&self.launch_failures);
            tokio::spawn(async move {
                let shard = spec.shard;
                if let Err(err) = launcher.spawn(spec).await {
                    eprintln!("space-orchestrator: spawn {shard} failed: {err}");
                    // For the event log; the loop picks it up on its next beat.
                    if let Ok(mut failures) = failures.lock() {
                        failures.push((shard, err.to_string()));
                    }
                }
            });
        }
        // Kill the shards a committed merge, an aborted split or the admin verb
        // staged.
        for shard in self.take_pending_kill() {
            if let Err(err) = self.launcher.kill(shard).await {
                eprintln!("space-orchestrator: kill {shard} failed: {err}");
                self.log(
                    EventKind::Launcher,
                    Severity::Warn,
                    format!("stopping {shard} failed: {err}"),
                );
            }
            server.unregister(shard).await;
            // Stopped on purpose: not a link that was "lost".
            self.connected.remove(&shard);
            self.scene_confirmed.remove(&shard);
        }
        // What follows is for the operator view only.
        let failed = self
            .launch_failures
            .lock()
            .map(|mut failures| std::mem::take(&mut *failures))
            .unwrap_or_default();
        for (shard, why) in failed {
            self.log(
                EventKind::Launcher,
                Severity::Warn,
                format!("launching {shard} failed: {why}"),
            );
        }
        self.note_connected(server.connected().await);
        self.print_new_events();
        self.publish(false);
    }

    /// Write the event-log entries not yet written to standard output, oldest
    /// first, one line each.
    ///
    /// This is what puts a registration, a commit or an abort into the
    /// process's log (a container's `docker logs`), where until now only
    /// failures appeared. It runs on the beat, so only the real loop prints —
    /// a test driving [`Self::tick`] by hand stays quiet — and an entry that
    /// merely repeats is not printed again, so a peer retrying for ever cannot
    /// flood the log.
    fn print_new_events(&mut self) {
        for line in self.new_event_lines() {
            println!("{line}");
        }
    }

    /// The lines [`Self::print_new_events`] writes, marking the entries as
    /// written.
    ///
    /// Entries that anybody who can reach the control port can cause (refused
    /// hellos, with a text of the sender's choosing each time) are held to
    /// [`PEER_PRINTS_PER_WINDOW`] lines per [`PEER_PRINT_WINDOW_TICKS`]; what
    /// is held back is counted and reported in one line when the next window
    /// opens. Everything else is written as it happens.
    fn new_event_lines(&mut self) -> Vec<String> {
        let mut lines = Vec::new();
        let budget = self.peer_prints;
        if self.tick.saturating_sub(budget.window_start) >= PEER_PRINT_WINDOW_TICKS {
            // What happened in the window that just closed and was not
            // written: counted from the log's own total, so entries the small
            // ring had already dropped again are included.
            let happened = budget
                .total_at_last_look
                .saturating_sub(budget.total_at_window_start);
            let held_back = happened.saturating_sub(u64::from(budget.printed));
            if held_back > 0 {
                lines.push(format!(
                    "space-orchestrator: [tick {}] shard: {held_back} more refused hello(s) were \
                     not written here (the newest are in the event log)",
                    self.tick
                ));
            }
            self.peer_prints = PeerPrintBudget {
                window_start: self.tick,
                printed: 0,
                total_at_window_start: budget.total_at_last_look,
                total_at_last_look: budget.total_at_last_look,
            };
        }
        self.peer_prints.total_at_last_look = self.events.peer_caused_total();
        let mut fresh: Vec<_> = self
            .events
            .newest_first()
            .take_while(|event| event.seq > self.printed_event)
            .collect();
        fresh.reverse();
        let Some(newest) = fresh.last().map(|event| event.seq) else {
            return lines;
        };
        for event in fresh {
            if event.peer_caused {
                if self.peer_prints.printed >= PEER_PRINTS_PER_WINDOW {
                    continue;
                }
                self.peer_prints.printed += 1;
            }
            lines.push(format!(
                "space-orchestrator: [tick {}] {}: {}",
                event.first_tick,
                event.kind.label(),
                event.text
            ));
        }
        self.printed_event = newest;
        lines
    }
}

/// The next operator request, or never when no front end is attached.
async fn next_operator_request(ops: &mut Option<OpsPort>) -> Option<OperatorRequest> {
    match ops {
        Some(port) => port.next_request().await,
        None => std::future::pending().await,
    }
}

/// `shards` as "shard 1, shard 2" ("nobody" when empty).
fn list_shards(shards: &[ShardId]) -> String {
    if shards.is_empty() {
        return "nobody".into();
    }
    shards
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::ControlPlane;
    use crate::epoch::EpochPublisher;
    use crate::ids::{MemoryCounterSink, PersistentShardIdAllocator};
    use crate::launcher::{LaunchError, Poll, ShardHandle, SpawnSpec};
    use space_protocol::shard::{ShardId, ShardMap, ShardRegion};
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    fn addr(n: u16) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), n)
    }

    /// A launcher that records calls and never touches a process.
    #[derive(Debug, Default)]
    struct MockLauncher {
        spawned: std::sync::Mutex<Vec<ShardId>>,
    }

    impl Launcher for MockLauncher {
        // `async fn` implementing an AFIT method: rustc checks the generated
        // future against the trait's `+ Send` bound.
        async fn spawn(&self, spec: SpawnSpec) -> Result<ShardHandle, LaunchError> {
            // The lock is taken and released with no await point in between, so
            // the future stays `Send`.
            self.spawned.lock().unwrap().push(spec.shard);
            Ok(ShardHandle {
                shard: spec.shard,
                endpoint: spec.region.endpoint,
                process: "mock".into(),
            })
        }

        async fn kill(&self, _shard: ShardId) -> Result<(), LaunchError> {
            Ok(())
        }

        async fn poll(&self, _shard: ShardId) -> Result<Poll, LaunchError> {
            Ok(Poll::Alive)
        }
    }

    /// Drive a future that is ready without I/O to completion. The mock
    /// launcher's futures never yield, so a no-op waker suffices — this keeps
    /// the test dependency-free (no tokio/futures-lite in the offline cache) and
    /// safe (an `Arc`-backed `Wake`, so no `unsafe` in a `forbid(unsafe_code)`
    /// crate).
    fn block_on<F: std::future::Future>(fut: F) -> F::Output {
        use std::task::{Context, Poll, Waker};
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        let mut fut = std::pin::pin!(fut);
        loop {
            match std::future::Future::poll(fut.as_mut(), &mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => std::hint::spin_loop(),
            }
        }
    }

    fn orchestrator() -> Orchestrator<MockLauncher> {
        orchestrator_with_first_id(1)
    }

    fn orchestrator_with_first_id(first_id: u64) -> Orchestrator<MockLauncher> {
        let config = OrchestratorConfig::default();
        let map = ShardMap::new(
            1,
            vec![ShardRegion::new(
                config.root_node(),
                ShardId::new(1),
                addr(7_001),
            )],
        );
        let ids =
            PersistentShardIdAllocator::with_sink(first_id, Box::new(MemoryCounterSink::default()));
        let control = ControlPlane::new(config, ids, EpochPublisher::new(map), "secret".into());
        Orchestrator::new(control, std::sync::Arc::new(MockLauncher::default()))
    }

    /// An orchestrator over an explicit map and id sink.
    fn orchestrator_over(
        config: OrchestratorConfig,
        regions: Vec<ShardRegion>,
        ids: PersistentShardIdAllocator,
    ) -> Orchestrator<MockLauncher> {
        let map = ShardMap::new(1, regions);
        let control = ControlPlane::new(config, ids, EpochPublisher::new(map), "secret".into());
        Orchestrator::new(control, std::sync::Arc::new(MockLauncher::default()))
    }

    /// Root (depth 40) owned by shard 1 with octant 0 carved out for shard 2.
    fn orchestrator_with_a_child() -> Orchestrator<MockLauncher> {
        let config = OrchestratorConfig::default();
        let root = config.root_node();
        let child = space_world::octree::children(root).expect("root splits")[0];
        let ids =
            PersistentShardIdAllocator::with_sink(100, Box::new(MemoryCounterSink::default()));
        orchestrator_over(
            config,
            vec![
                ShardRegion::new(root, ShardId::new(1), addr(7_001)),
                ShardRegion::new(child, ShardId::new(2), addr(7_002)),
            ],
            ids,
        )
    }

    fn report(shard: u32, players: u32) -> LoadReport {
        LoadReport {
            shard: ShardId::new(shard),
            players,
            tick_p95_ms: 1.0,
            load: 0.0,
        }
    }

    /// The `(recipient, epoch)` of every `EpochPrepare` in `outbox`.
    fn prepares(outbox: &[(ShardId, Message)]) -> Vec<(ShardId, u64)> {
        outbox
            .iter()
            .filter_map(|(to, m)| match m {
                Message::EpochPrepare(p) => Some((*to, p.epoch)),
                _ => None,
            })
            .collect()
    }

    /// Drain the outbox and ack the prepare in it on behalf of every recipient.
    /// Returns how many prepares were acked.
    fn ack_the_prepare(orch: &mut Orchestrator<MockLauncher>) -> usize {
        let recipients = prepares(&orch.drain_outbox());
        for (shard, epoch) in &recipients {
            orch.handle_epoch_ack(*shard, *epoch);
        }
        recipients.len()
    }

    fn accepted(reply: &AdminReply) -> bool {
        matches!(reply, AdminReply::Accepted { .. })
    }

    fn refusal(reply: AdminReply) -> String {
        match reply {
            AdminReply::Refused { reason, .. } => reason,
            other => panic!("expected Refused, got {other:?}"),
        }
    }

    // ------------------------------------------------------------- finding 7

    /// Mutation this catches: always carving the same octant. The second split
    /// of a region then proposes a node the map already has, fails validation,
    /// and the hot shard can never be relieved again.
    #[test]
    fn a_region_can_be_split_again_and_each_split_carves_a_new_octant() {
        let mut orch = orchestrator_with_first_id(100);
        let root = orch.config().root_node();
        let kids = space_world::octree::children(root).expect("root splits");
        let mut carved = Vec::new();
        for round in 0..3u64 {
            let reply = orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1)));
            assert!(accepted(&reply), "split {round} is accepted: {reply:?}");
            let spec = orch.take_pending_spawn().pop().expect("a child is staged");
            assert!(ack_the_prepare(&mut orch) > 0, "a prepare went out");
            assert_eq!(orch.map().epoch, 2 + round, "split {round} committed");
            carved.push(spec.region.node);
            for _ in 0..MIN_EPOCH_INTERVAL_TICKS {
                orch.tick();
            }
            let _ = orch.drain_outbox();
        }
        // The admin verb counts from octant 0: 0, then 1, then 2.
        assert_eq!(carved, vec![kids[0], kids[1], kids[2]]);
        assert_eq!(orch.map().regions.len(), 4);
        crate::topology::validate(orch.map()).expect("the map stays valid");
    }

    /// The autoscaler's own split starts at the region's centre octant — where
    /// joining ships spawn — not at octant 0, the far corner.
    #[test]
    fn an_autoscaler_split_carves_the_centre_octant_first() {
        let mut orch = orchestrator_with_first_id(100);
        let root = orch.config().root_node();
        let mut staged = None;
        for _ in 0..=(crate::autoscale::SPLIT_DWELL_TICKS + 1) {
            orch.handle_load_report(&report(1, 90));
            orch.tick();
            if let Some(spec) = orch.take_pending_spawn().pop() {
                staged = Some(spec);
                break;
            }
        }
        let spec = staged.expect("a shard hot for the dwell is split");
        let kids = space_world::octree::children(root).expect("root splits");
        assert_eq!(
            spec.region.node, kids[7],
            "the carved octant is the one holding the region centre"
        );
        assert!(space_world::octree::contains(
            spec.region.node,
            root.center()
        ));
    }

    /// Mutation this catches: allocating the child's id before the plan is
    /// known to work. Every refused split then burned an id (and a counter
    /// write) per cooldown, for as long as the shard stayed hot.
    #[test]
    fn a_split_that_cannot_happen_burns_no_id_and_says_why() {
        // A depth-55 root is below the split floor.
        let config = OrchestratorConfig {
            root_depth: 55,
            ..OrchestratorConfig::default()
        };
        let root = config.root_node();
        let ids =
            PersistentShardIdAllocator::with_sink(100, Box::new(MemoryCounterSink::default()));
        let mut orch = orchestrator_over(
            config,
            vec![ShardRegion::new(root, ShardId::new(1), addr(7_001))],
            ids,
        );
        let why = refusal(orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1))));
        assert!(why.contains("split floor"), "{why}");
        assert_eq!(orch.control.ids.peek(), 100, "no id was allocated");
        assert!(orch.take_pending_spawn().is_empty());
        assert!(orch.drain_outbox().is_empty(), "no prepare was published");

        // The autoscaler path refuses the same way: hot for several dwells, and
        // still not a single id gone.
        for _ in 0..(4 * crate::autoscale::SCALE_COOLDOWN_TICKS) {
            orch.handle_load_report(&report(1, 95));
            orch.tick();
        }
        assert_eq!(orch.control.ids.peek(), 100, "a refused split burns no id");
        assert_eq!(orch.map().epoch, 1);

        // A shard the map does not know is refused too, again without an id.
        let why = refusal(orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(9))));
        assert!(why.contains("no region owned"), "{why}");
        assert_eq!(orch.control.ids.peek(), 100);
    }

    // ------------------------------------------------------------ finding 11

    #[derive(Debug)]
    struct BrokenSink;

    impl crate::ids::CounterSink for BrokenSink {
        fn persist(&mut self, _next: u64) -> std::io::Result<()> {
            Err(std::io::Error::other("disk full"))
        }
    }

    /// Mutation this catches: the `expect` on the counter write. A split with a
    /// full disk took the whole orchestrator down; it must be a refused split.
    #[test]
    fn a_split_with_an_unwritable_counter_is_refused_not_a_panic() {
        let config = OrchestratorConfig::default();
        let root = config.root_node();
        let ids = PersistentShardIdAllocator::with_sink(100, Box::new(BrokenSink));
        let mut orch = orchestrator_over(
            config,
            vec![ShardRegion::new(root, ShardId::new(1), addr(7_001))],
            ids,
        );
        let before = orch.map().clone();
        let why = refusal(orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1))));
        assert!(why.contains("could not be persisted"), "{why}");
        assert!(orch.take_pending_spawn().is_empty(), "nothing is launched");
        assert!(orch.drain_outbox().is_empty(), "no prepare was published");
        assert_eq!(orch.map(), &before);
        // The same through the autoscaler: it logs and carries on ticking.
        for _ in 0..=(crate::autoscale::SPLIT_DWELL_TICKS + 1) {
            orch.handle_load_report(&report(1, 90));
            orch.tick();
        }
        assert!(orch.take_pending_spawn().is_empty());
        assert_eq!(orch.map(), &before);
    }

    // ------------------------------------------------------------- finding 6

    /// Mutation this catches: keeping a report from a shard the published map
    /// does not name. It was replayed to the autoscaler every tick and acted on.
    #[test]
    fn a_report_from_a_shard_outside_the_map_is_ignored() {
        let mut orch = orchestrator_with_first_id(100);
        for _ in 0..(2 * crate::autoscale::SPLIT_DWELL_TICKS) {
            orch.handle_load_report(&report(77, 95));
            orch.tick();
        }
        assert!(orch.load_report(ShardId::new(77)).is_none());
        assert!(!orch.control.autoscaler.tracks(ShardId::new(77)));
        assert!(
            orch.drain_outbox().is_empty(),
            "nothing was proposed for it"
        );
        assert_eq!(orch.control.ids.peek(), 100, "and no id was spent on it");
    }

    /// Mutation this catches: replaying a shard's last report for ever. A shard
    /// that stopped reporting two thirds of the way through its dwell was split
    /// anyway, a hundred ticks after it went silent.
    #[test]
    fn a_stale_report_expires_and_never_completes_a_dwell() {
        let mut orch = orchestrator_with_first_id(100);
        let dwell = crate::autoscale::SPLIT_DWELL_TICKS;
        for _ in 0..(dwell * 2 / 3) {
            orch.handle_load_report(&report(1, 95));
            orch.tick();
        }
        assert!(orch.load_report(ShardId::new(1)).is_some());
        assert!(orch.control.autoscaler.tracks(ShardId::new(1)));
        // The shard goes silent.
        for _ in 0..(2 * dwell) {
            orch.tick();
        }
        assert!(
            orch.load_report(ShardId::new(1)).is_none(),
            "a report older than REPORT_EXPIRY_TICKS is no longer believed"
        );
        assert!(
            !orch.control.autoscaler.tracks(ShardId::new(1)),
            "and the autoscaler has forgotten the shard"
        );
        assert!(orch.take_pending_spawn().is_empty(), "nothing was split");
        assert!(orch.drain_outbox().is_empty());
        assert_eq!(orch.map().epoch, 1);
        // A report that keeps arriving within the expiry is still believed: one
        // every LOAD_REPORT_INTERVAL_TICKS is enough to complete a dwell.
        let mut split = false;
        for t in 0..=(dwell + crate::config::LOAD_REPORT_INTERVAL_TICKS) {
            if t % crate::config::LOAD_REPORT_INTERVAL_TICKS == 0 {
                orch.handle_load_report(&report(1, 95));
            }
            orch.tick();
            split |= !orch.take_pending_spawn().is_empty();
        }
        assert!(
            split,
            "a shard reporting once a beat is split after its dwell"
        );
    }

    // ------------------------------------------------------------- finding 3

    /// Mutation this catches: staging the kill when the merge is *proposed*.
    /// The loop then killed the shard in the same tick it sent the prepare: the
    /// shard never acked, the epoch aborted, and the map kept naming a dead
    /// owner for its region.
    #[test]
    fn a_merge_kills_the_child_only_once_the_epoch_commits() {
        let mut orch = orchestrator_with_a_child();
        let child = ShardId::new(2);
        orch.handle_load_report(&report(2, 0));
        let reply = orch.handle_admin(&AdminCommand::ForceMerge(child));
        assert!(accepted(&reply), "{reply:?}");
        assert_eq!(orch.pending_merge(), Some(child));
        assert!(
            orch.take_pending_kill().is_empty(),
            "nothing is killed while the prepare is outstanding"
        );
        // A tick passes with the prepare still un-acked: still nobody to kill.
        let outbox = orch.drain_outbox();
        orch.tick();
        assert!(orch.take_pending_kill().is_empty());
        assert_eq!(
            orch.map().regions.len(),
            2,
            "the child still owns its region"
        );

        // Every recipient acks: the merge commits, and only now is the kill staged.
        let recipients = prepares(&outbox);
        assert!(recipients.iter().any(|(to, _)| *to == child));
        for (shard, epoch) in recipients {
            orch.handle_epoch_ack(shard, epoch);
        }
        assert_eq!(orch.map().epoch, 2);
        assert_eq!(orch.map().regions.len(), 1, "the child's region is gone");
        assert_eq!(orch.take_pending_kill(), vec![child]);
        assert_eq!(orch.pending_merge(), None);

        // The departing shard is told the outcome (commit, then the map) before
        // it is killed: the loop delivers the outbox ahead of the kills.
        let to_child: Vec<Message> = orch
            .drain_outbox()
            .into_iter()
            .filter(|(to, _)| *to == child)
            .map(|(_, m)| m)
            .collect();
        assert!(matches!(to_child.first(), Some(Message::EpochCommit(_))));
        assert!(matches!(to_child.get(1), Some(Message::DirectoryReply(_))));
        // And nothing about it is remembered: no report, no autoscaler state.
        assert!(orch.load_report(child).is_none());
        assert!(!orch.control.autoscaler.tracks(child));
        // Its last reports may still arrive; they are dropped (it is off the map).
        orch.handle_load_report(&report(2, 0));
        assert!(orch.load_report(child).is_none());
    }

    /// Mutation this catches: not clearing the pending merge when its epoch
    /// times out — the next commit of anything would kill a shard that still
    /// owns its region.
    #[test]
    fn a_merge_that_aborts_kills_nobody_and_is_forgotten() {
        let mut orch = orchestrator_with_a_child();
        let child = ShardId::new(2);
        orch.handle_load_report(&report(2, 0));
        assert!(accepted(
            &orch.handle_admin(&AdminCommand::ForceMerge(child))
        ));
        let _ = orch.drain_outbox();
        // Nobody acks; the deadline passes.
        for _ in 0..=crate::config::EPOCH_ACK_TIMEOUT_TICKS {
            orch.tick();
        }
        assert_eq!(orch.map().epoch, 1, "the merge aborted");
        assert_eq!(
            orch.map().regions.len(),
            2,
            "the child still owns its region"
        );
        assert_eq!(orch.pending_merge(), None, "the aborted merge is forgotten");
        assert!(orch.take_pending_kill().is_empty(), "an abort kills nobody");
        let _ = orch.drain_outbox();

        // A later, unrelated epoch commits: the child must survive it.
        assert!(accepted(
            &orch.handle_admin(&AdminCommand::ForceSplit(child))
        ));
        let grandchild = orch.take_pending_spawn().pop().expect("staged").shard;
        assert!(ack_the_prepare(&mut orch) > 0);
        assert_eq!(orch.map().epoch, 2, "the split committed");
        assert!(orch.map().regions.iter().any(|r| r.shard == grandchild));
        assert!(
            orch.take_pending_kill().is_empty(),
            "committing another epoch must not kill the shard whose merge aborted"
        );
    }

    /// Mutation this catches: merging on "cold" alone. Cold is a load at or
    /// below a third of a full shard; a merge removes the shard, and there is no
    /// step that hands its players to anyone.
    #[test]
    fn only_a_shard_known_to_be_empty_is_merged() {
        let mut orch = orchestrator_with_a_child();
        let child = ShardId::new(2);
        // No report at all: not known to be empty.
        let why = refusal(orch.handle_admin(&AdminCommand::ForceMerge(child)));
        assert!(why.contains("not known to be empty"), "{why}");
        // Three players: cold by the load metric, but not empty.
        orch.handle_load_report(&report(2, 3));
        let why = refusal(orch.handle_admin(&AdminCommand::ForceMerge(child)));
        assert!(why.contains("3 player"), "{why}");
        assert!(orch.drain_outbox().is_empty(), "no prepare was published");
        assert_eq!(orch.pending_merge(), None);

        // The autoscaler path: cold-but-populated for well past the merge dwell
        // (and its cooldowns) never merges...
        for _ in 0..(3 * crate::autoscale::MERGE_DWELL_TICKS) {
            orch.handle_load_report(&report(2, 3));
            orch.tick();
        }
        assert_eq!(orch.map().epoch, 1, "a populated shard is never merged");
        assert!(orch.take_pending_kill().is_empty());
        assert!(orch.drain_outbox().is_empty());
        // ...and once it reports empty for the dwell, it is.
        let mut proposed = false;
        for _ in 0..(2 * crate::autoscale::MERGE_DWELL_TICKS) {
            orch.handle_load_report(&report(2, 0));
            orch.tick();
            if orch.pending_merge() == Some(child) {
                proposed = true;
                break;
            }
        }
        assert!(proposed, "an empty shard, cold for the dwell, is merged");
    }

    // ------------------------------------------------- finding 2 (the abort)

    /// A split whose epoch aborts leaves a launched child that owns nothing: it
    /// is staged for killing (it used to run on, unowned, for ever), and the
    /// abort reaches every recipient of the prepare.
    #[test]
    fn a_split_that_aborts_stages_the_kill_of_its_child() {
        let mut orch = orchestrator_with_first_id(100);
        assert!(accepted(
            &orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1)))
        ));
        let child = orch.take_pending_spawn().pop().expect("staged").shard;
        let told: Vec<ShardId> = prepares(&orch.drain_outbox())
            .into_iter()
            .map(|(to, _)| to)
            .collect();
        assert!(orch.take_pending_kill().is_empty());
        for _ in 0..=crate::config::EPOCH_ACK_TIMEOUT_TICKS {
            orch.tick();
        }
        assert_eq!(orch.map().epoch, 1, "the split aborted");
        assert_eq!(orch.take_pending_kill(), vec![child]);
        let aborted: Vec<ShardId> = orch
            .drain_outbox()
            .into_iter()
            .filter(|(_, m)| matches!(m, Message::EpochAbort(_)))
            .map(|(to, _)| to)
            .collect();
        assert_eq!(aborted, told, "everyone who got the prepare gets the abort");
        // A committed split, by contrast, kills nobody.
        assert!(accepted(
            &orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1)))
        ));
        let _ = orch.take_pending_spawn();
        assert!(ack_the_prepare(&mut orch) > 0);
        assert_eq!(orch.map().epoch, 2);
        assert!(orch.take_pending_kill().is_empty());
    }

    #[test]
    fn the_orchestrator_is_constructible_without_a_runtime() {
        // The whole point of this slice: the type builds in a plain #[test].
        let orch = orchestrator();
        assert_eq!(orch.config().cluster, OrchestratorConfig::default().cluster);
        assert_eq!(orch.control.map().epoch, 1);
    }

    #[test]
    fn a_commit_is_followed_by_a_map_push_carrying_every_real_endpoint() {
        // Ids start above the root's so the child is a distinct shard.
        let mut orch = orchestrator_with_first_id(100);
        let hello = |shard: ShardId, port: u16, region| ShardHello {
            cluster: OrchestratorConfig::default().cluster,
            shard,
            endpoint: addr(port),
            region,
        };
        orch.handle_shard_hello(&hello(ShardId::new(1), 9_001, None))
            .expect("root shard registers");
        orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1)));
        let spec = orch.take_pending_spawn().pop().expect("a child is staged");
        orch.handle_shard_hello(&hello(spec.shard, 9_002, Some(spec.region.clone())))
            .expect("child registers");
        let outbox = orch.drain_outbox();
        let epoch = outbox
            .iter()
            .find_map(|(_, m)| match m {
                Message::EpochPrepare(p) => Some(p.epoch),
                _ => None,
            })
            .expect("the prepare went out");
        let recipients: Vec<ShardId> = outbox.into_iter().map(|(to, _)| to).collect();
        assert!(!recipients.is_empty(), "the prepare went out");
        for shard in recipients {
            orch.handle_epoch_ack(shard, epoch);
        }

        let out = orch.drain_outbox();
        for shard in [ShardId::new(1), spec.shard] {
            let kinds: Vec<&Message> = out
                .iter()
                .filter(|(to, _)| *to == shard)
                .map(|(_, m)| m)
                .collect();
            assert!(
                matches!(kinds.first(), Some(Message::EpochCommit(_))),
                "the commit reaches {shard} first"
            );
            let Some(Message::DirectoryReply(push)) = kinds.get(1) else {
                panic!("the commit is followed by a map push for {shard}");
            };
            assert_eq!(push.shard_map.epoch, 2);
            let endpoint_of = |s: ShardId| {
                push.shard_map
                    .regions
                    .iter()
                    .find(|r| r.shard == s)
                    .map(|r| r.endpoint)
            };
            assert_eq!(endpoint_of(ShardId::new(1)), Some(addr(9_001)));
            assert_eq!(endpoint_of(spec.shard), Some(addr(9_002)));
        }
        // The orchestrator's own published map carries the same addresses.
        assert!(orch
            .map()
            .regions
            .iter()
            .any(|r| r.shard == spec.shard && r.endpoint == addr(9_002)));
    }

    #[test]
    fn a_child_that_never_registered_stays_at_the_sentinel_in_the_pushed_map() {
        let mut orch = orchestrator_with_first_id(100);
        orch.handle_shard_hello(&ShardHello {
            cluster: OrchestratorConfig::default().cluster,
            shard: ShardId::new(1),
            endpoint: addr(9_001),
            region: None,
        })
        .expect("root shard registers");
        orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1)));
        let spec = orch.take_pending_spawn().pop().expect("a child is staged");
        // The child is never spawned/registered: nobody tells the orchestrator
        // its address. Ack the prepare from whoever it went to.
        let outbox = orch.drain_outbox();
        let epoch = outbox
            .iter()
            .find_map(|(_, m)| match m {
                Message::EpochPrepare(p) => Some(p.epoch),
                _ => None,
            })
            .expect("the prepare went out");
        let recipients: Vec<ShardId> = outbox.into_iter().map(|(to, _)| to).collect();
        assert!(!recipients.is_empty(), "the prepare went out");
        for shard in recipients {
            orch.handle_epoch_ack(shard, epoch);
        }

        let out = orch.drain_outbox();
        let pushes: Vec<&space_protocol::DirectoryReply> = out
            .iter()
            .filter_map(|(_, m)| match m {
                Message::DirectoryReply(push) => Some(push),
                _ => None,
            })
            .collect();
        assert!(!pushes.is_empty(), "the commit is followed by a map push");
        for push in pushes {
            let child = push
                .shard_map
                .regions
                .iter()
                .find(|r| r.shard == spec.shard)
                .expect("the child's region is in the pushed map");
            assert_eq!(
                child.endpoint,
                ShardRegion::UNRESOLVED_ENDPOINT,
                "an unregistered child must not inherit the parent's address"
            );
            let parent = push
                .shard_map
                .regions
                .iter()
                .find(|r| r.shard == ShardId::new(1))
                .expect("parent region");
            assert_eq!(parent.endpoint, addr(9_001));
        }
    }

    #[test]
    fn the_mock_launcher_satisfies_the_trait() {
        // Proves the Launcher trait is implementable as a plain struct (the
        // acceptance tests rely on this).
        let launcher = MockLauncher::default();
        let spec = SpawnSpec {
            shard: ShardId::new(1),
            region: ShardRegion::new(
                OrchestratorConfig::default().root_node(),
                ShardId::new(1),
                addr(7_001),
            ),
            cluster: "space-local".into(),
            control_addr: addr(7_900),
            host_port: 7_870,
        };
        let handle = block_on(launcher.spawn(spec)).expect("spawn");
        assert_eq!(handle.shard, ShardId::new(1));
        assert_eq!(
            block_on(launcher.poll(ShardId::new(1))).unwrap(),
            Poll::Alive
        );
    }

    // ------------------------------------------------------ the operator path

    fn split(shard: u32, octant: Option<u8>) -> OperatorCommand {
        OperatorCommand::Split {
            shard: ShardId::new(shard),
            octant,
            expected_epoch: None,
        }
    }

    fn merge(shard: u32) -> OperatorCommand {
        OperatorCommand::Merge {
            shard: ShardId::new(shard),
            expected_epoch: None,
        }
    }

    /// Root owned by shard 1, which holds a control link; ids start at 100.
    fn connected_orchestrator() -> Orchestrator<MockLauncher> {
        let mut orch = orchestrator_with_first_id(100);
        orch.note_connected([ShardId::new(1)]);
        orch
    }

    fn event_texts(orch: &Orchestrator<MockLauncher>) -> Vec<String> {
        orch.events()
            .newest_first()
            .map(|event| event.text.clone())
            .collect()
    }

    /// An operator's split is an ordinary epoch change: proposed, acked by
    /// every recipient, committed — and it carves exactly the octant named.
    ///
    /// Mutation this catches: an operator path that writes the map itself
    /// (the map would change before the acks), or that treats the octant as a
    /// starting point like the admin verb does.
    #[test]
    fn an_operator_split_carves_the_octant_asked_for_through_the_epoch_protocol() {
        let mut orch = connected_orchestrator();
        let kids = space_world::octree::children(orch.config().root_node()).expect("splits");

        let reply = orch.handle_operator(&split(1, Some(5)));
        assert!(reply.accepted, "{reply:?}");
        assert!(reply.message.contains("epoch 2 proposed"), "{reply:?}");
        // Proposed is not published: no shard may see the map before the commit (H10).
        assert_eq!(orch.map().epoch, 1);
        assert_eq!(orch.map().regions.len(), 1);
        let spec = orch
            .take_pending_spawn()
            .pop()
            .expect("the child is staged");
        assert_eq!(spec.region.node, kids[5], "exactly the octant asked for");
        assert_eq!(spec.shard, ShardId::new(100), "a freshly allocated id (H9)");

        let snapshot = orch.snapshot();
        let flight = snapshot.in_flight.expect("the change is in flight");
        assert_eq!(flight.epoch, 2);
        assert_eq!(flight.changes, vec![(kids[5], ShardId::new(100))]);
        assert_eq!(
            flight.recipients,
            vec![(ShardId::new(1), false), (ShardId::new(100), false)],
            "everyone who must ack, nobody has yet"
        );
        assert!(snapshot
            .shards
            .iter()
            .any(|s| s.shard == ShardId::new(100) && s.role == ShardRole::SplitChild));
        assert_eq!(
            snapshot.regions.len(),
            1,
            "the snapshot shows the committed map"
        );

        assert!(ack_the_prepare(&mut orch) > 0);
        assert_eq!(orch.map().epoch, 2);
        assert!(orch
            .map()
            .regions
            .iter()
            .any(|r| r.node == kids[5] && r.shard == ShardId::new(100)));
        crate::topology::validate(orch.map()).expect("the map stays valid (H11)");
        assert!(orch.take_pending_kill().is_empty());

        let snapshot = orch.snapshot();
        assert!(snapshot.in_flight.is_none());
        assert_eq!(snapshot.regions.len(), 2);
        assert_eq!(snapshot.last_commit_tick, Some(orch.current_tick()));
        let texts = event_texts(&orch);
        assert!(
            texts[0].starts_with("epoch 2 committed: split of shard 1"),
            "{texts:?}"
        );
        assert!(
            texts
                .iter()
                .any(|t| t.starts_with("split shard 1, octant 5 (+x -y +z): accepted")),
            "{texts:?}"
        );
        assert_eq!(
            orch.events().get(reply.event).map(|e| e.kind),
            Some(EventKind::Operator),
            "the reply names its log entry"
        );
    }

    #[test]
    fn an_operator_split_with_no_octant_starts_at_the_centre_like_the_autoscaler() {
        let mut orch = connected_orchestrator();
        let kids = space_world::octree::children(orch.config().root_node()).expect("splits");
        assert!(orch.handle_operator(&split(1, None)).accepted);
        let spec = orch.take_pending_spawn().pop().expect("staged");
        assert_eq!(spec.region.node, kids[7]);
    }

    /// Mutation this catches: letting an operator's mistake go through
    /// `begin_split`'s refusal path, which cools the shard down and makes the
    /// autoscaler "busy" — or worse, allocating the child's id first.
    #[test]
    fn a_refused_operator_request_changes_nothing_at_all() {
        let mut orch = connected_orchestrator();
        // Carve octant 3 for shard 100 and let the minimum interval pass.
        assert!(orch.handle_operator(&split(1, Some(3))).accepted);
        let _ = orch.take_pending_spawn();
        assert!(ack_the_prepare(&mut orch) > 0);
        for _ in 0..MIN_EPOCH_INTERVAL_TICKS {
            orch.tick();
        }
        let _ = orch.drain_outbox();
        assert_eq!(orch.map().epoch, 2);

        let state = |orch: &Orchestrator<MockLauncher>| {
            (
                orch.map().clone(),
                orch.control.ids.peek(),
                orch.control.autoscaler.busy_until(),
                orch.control.autoscaler.tracks(ShardId::new(1)),
            )
        };
        let before = state(&orch);
        let stale = OperatorCommand::Split {
            shard: ShardId::new(1),
            octant: Some(0),
            expected_epoch: Some(1),
        };
        for (command, why) in [
            (split(1, Some(3)), "already carved out"),
            (split(1, Some(9)), "does not exist"),
            (split(42, Some(0)), "no region owned by shard 42"),
            (split(100, Some(0)), "shard 100 is not connected"),
            (stale, "the map changed"),
            (merge(1), "no parent to merge into"),
            (merge(42), "no region owned by shard 42"),
            (merge(100), "not known to be empty"),
        ] {
            let reply = orch.handle_operator(&command);
            assert!(!reply.accepted, "{command}: {reply:?}");
            assert!(reply.message.contains(why), "{command}: {reply:?}");
            let entry = orch
                .events()
                .get(reply.event)
                .expect("the refusal is logged");
            assert_eq!(entry.severity, Severity::Warn);
            assert!(entry.text.contains("refused"), "{entry:?}");
        }
        assert_eq!(state(&orch), before, "a refusal has no side effect");
        assert!(orch.take_pending_spawn().is_empty());
        assert!(orch.take_pending_kill().is_empty());
        assert!(orch.drain_outbox().is_empty(), "nothing was published");
        assert_eq!(orch.pending_merge(), None);
        assert!(orch.snapshot().in_flight.is_none());
    }

    /// One change at a time (H12): a second request while a prepare is
    /// outstanding is refused, not queued, and burns no id. When the first one
    /// aborts, the log says who did not ack and the next request is accepted.
    #[test]
    fn an_operator_request_while_an_epoch_is_in_flight_is_refused_not_queued() {
        let mut orch = connected_orchestrator();
        assert!(orch.handle_operator(&split(1, Some(0))).accepted);
        let ids = orch.control.ids.peek();
        let reply = orch.handle_operator(&split(1, Some(1)));
        assert!(!reply.accepted);
        assert!(
            reply.message.contains("epoch 2 is still in flight"),
            "{reply:?}"
        );
        assert_eq!(orch.control.ids.peek(), ids, "no id was burned");
        assert_eq!(
            orch.take_pending_spawn().len(),
            1,
            "only the first split's child"
        );
        let prepared = prepares(&orch.drain_outbox());
        assert!(prepared.iter().all(|(_, epoch)| *epoch == 2));

        // Nobody acks: the first change aborts at its deadline.
        for _ in 0..=crate::config::EPOCH_ACK_TIMEOUT_TICKS {
            orch.tick();
        }
        assert_eq!(orch.map().epoch, 1, "the map is unchanged");
        let texts = event_texts(&orch);
        let aborted = texts
            .iter()
            .find(|t| t.starts_with("epoch 2 aborted"))
            .unwrap_or_else(|| panic!("the abort is logged: {texts:?}"));
        assert!(
            aborted.contains("no ack from shard 1, shard 100"),
            "{aborted}"
        );
        assert!(orch.snapshot().in_flight.is_none());
        let _ = orch.drain_outbox();
        let _ = orch.take_pending_kill();

        // Nothing was queued: the refused split does not happen on its own...
        for _ in 0..(2 * MIN_EPOCH_INTERVAL_TICKS) {
            orch.tick();
        }
        assert!(orch.drain_outbox().is_empty());
        assert!(orch.take_pending_spawn().is_empty());
        // ...and asking again now is accepted.
        let reply = orch.handle_operator(&split(1, Some(1)));
        assert!(reply.accepted, "{reply:?}");
    }

    #[test]
    fn an_operator_merge_obeys_the_empty_shard_rule_and_kills_only_on_commit() {
        let mut orch = orchestrator_with_a_child();
        let child = ShardId::new(2);
        orch.note_connected([ShardId::new(1), child]);

        let reply = orch.handle_operator(&merge(2));
        assert!(reply.message.contains("not known to be empty"), "{reply:?}");
        orch.handle_load_report(&report(2, 3));
        let reply = orch.handle_operator(&merge(2));
        assert!(
            !reply.accepted && reply.message.contains("3 player"),
            "{reply:?}"
        );
        assert!(orch.drain_outbox().is_empty(), "no prepare was published");

        orch.handle_load_report(&report(2, 0));
        let reply = orch.handle_operator(&merge(2));
        assert!(reply.accepted, "{reply:?}");
        assert_eq!(orch.pending_merge(), Some(child));
        assert!(
            orch.take_pending_kill().is_empty(),
            "the shard keeps running until the commit (H13)"
        );
        let role = orch
            .snapshot()
            .shards
            .iter()
            .find(|s| s.shard == child)
            .map(|s| s.role);
        assert_eq!(role, Some(ShardRole::Merging));

        assert!(ack_the_prepare(&mut orch) > 0);
        assert_eq!(orch.map().epoch, 2);
        assert_eq!(orch.map().regions.len(), 1);
        assert_eq!(orch.take_pending_kill(), vec![child]);
        assert!(event_texts(&orch)[0].starts_with("epoch 2 committed: merge of shard 2"));
    }

    /// Mutation this catches: a pause that only hides the autoscaler's
    /// decisions (it would still split), and a resume that keeps the dwell
    /// timer from before the pause (it would split on the first tick).
    #[test]
    fn a_paused_autoscaler_takes_no_action_and_needs_a_fresh_dwell_afterwards() {
        let mut orch = connected_orchestrator();
        let dwell = crate::autoscale::SPLIT_DWELL_TICKS;
        let hot_tick = |orch: &mut Orchestrator<MockLauncher>| {
            orch.handle_load_report(&report(1, 95));
            orch.tick();
            !orch.take_pending_spawn().is_empty()
        };
        // Hot for all but a few ticks of the dwell, then paused.
        for _ in 0..(dwell - 5) {
            assert!(!hot_tick(&mut orch));
        }
        let reply = orch.handle_operator(&OperatorCommand::PauseAutoscaler);
        assert!(reply.accepted && orch.autoscaler_paused(), "{reply:?}");
        assert!(orch.snapshot().autoscaler.paused);
        // Still hot for three more dwells: nothing happens while paused.
        for _ in 0..(3 * dwell) {
            assert!(!hot_tick(&mut orch), "a paused autoscaler does not split");
        }
        assert_eq!(orch.map().epoch, 1);
        assert!(orch.drain_outbox().is_empty());
        assert!(orch.snapshot().autoscaler.last_decision.is_none());
        // Reports are still received and shown meanwhile.
        assert!(orch.load_report(ShardId::new(1)).is_some());
        // Pausing twice is not an error.
        assert!(
            orch.handle_operator(&OperatorCommand::PauseAutoscaler)
                .accepted
        );

        // Resumed: not at once, only after a fresh, full dwell.
        assert!(
            orch.handle_operator(&OperatorCommand::ResumeAutoscaler)
                .accepted
        );
        assert!(!orch.autoscaler_paused());
        for tick in 0..(dwell - 1) {
            assert!(!hot_tick(&mut orch), "no split {tick} ticks after resuming");
        }
        let mut did_split = false;
        for _ in 0..3 {
            did_split |= hot_tick(&mut orch);
        }
        assert!(did_split, "after a full fresh dwell the hot shard is split");
        let decision = orch
            .snapshot()
            .autoscaler
            .last_decision
            .expect("the decision is remembered");
        assert_eq!(decision.action, "split shard 1");
        assert!(decision.outcome.is_ok(), "{decision:?}");
    }

    #[test]
    fn an_operator_can_still_split_while_the_autoscaler_is_paused() {
        let mut orch = connected_orchestrator();
        assert!(
            orch.handle_operator(&OperatorCommand::PauseAutoscaler)
                .accepted
        );
        assert!(orch.handle_operator(&split(1, Some(2))).accepted);
        assert!(ack_the_prepare(&mut orch) > 0);
        assert_eq!(orch.map().epoch, 2);
    }

    /// The spacing between two commits (H12) binds an operator exactly as it
    /// binds the autoscaler: a request right after a commit is refused, with
    /// how long to wait, and accepted once the interval has passed.
    #[test]
    fn an_operator_change_right_after_a_commit_waits_for_the_minimum_interval() {
        let mut orch = connected_orchestrator();
        assert!(orch.handle_operator(&split(1, Some(0))).accepted);
        let _ = orch.take_pending_spawn();
        assert!(ack_the_prepare(&mut orch) > 0);
        assert_eq!(orch.map().epoch, 2);

        let reply = orch.handle_operator(&split(1, Some(1)));
        assert!(!reply.accepted, "{reply:?}");
        assert!(
            reply.message.contains("committed too recently")
                && reply
                    .message
                    .contains(&format!("{MIN_EPOCH_INTERVAL_TICKS} tick(s), about 3.0 s")),
            "{reply:?}"
        );
        assert!(orch.take_pending_spawn().is_empty());
        for _ in 0..(MIN_EPOCH_INTERVAL_TICKS - 1) {
            orch.tick();
        }
        assert!(!orch.handle_operator(&split(1, Some(1))).accepted);
        orch.tick();
        let reply = orch.handle_operator(&split(1, Some(1)));
        assert!(reply.accepted, "{reply:?}");
    }

    /// The snapshot is what the orchestrator knows — in particular a shard
    /// that never sent a load report has **no** report, not a report of zero.
    #[test]
    fn the_snapshot_says_what_is_known_and_nothing_that_is_not() {
        let mut orch = orchestrator_with_a_child();
        // Shard 1 registers and reports; shard 2 (octant 0) never does.
        orch.handle_shard_hello(&ShardHello {
            cluster: OrchestratorConfig::default().cluster,
            shard: ShardId::new(1),
            endpoint: addr(9_001),
            region: None,
        })
        .expect("the root owner registers");
        orch.note_connected([ShardId::new(1)]);
        orch.handle_load_report(&report(1, 7));
        orch.tick();

        let snapshot = orch.snapshot();
        assert_eq!(snapshot.tick, 1);
        assert_eq!(snapshot.epoch, 1);
        assert_eq!(snapshot.next_shard_id, 100);
        assert_eq!(snapshot.cluster, OrchestratorConfig::default().cluster);
        assert_eq!(snapshot.last_commit_tick, None);
        assert!(snapshot.in_flight.is_none());
        assert!(!snapshot.autoscaler.paused);

        // Regions: the enclosing one first, then the one carved out of it.
        let root = orch.config().root_node();
        let paths: Vec<String> = snapshot.regions.iter().map(|r| node_path(r.node)).collect();
        assert_eq!(paths, vec!["40:0:0:0", "41:0:0:0"]);
        let (outer, inner) = (&snapshot.regions[0], &snapshot.regions[1]);
        assert_eq!((outer.nesting, outer.parent, outer.octant), (0, None, None));
        assert_eq!(
            outer.endpoint,
            Some(addr(7_001)),
            "the address the map holds for a claimed region"
        );
        assert_eq!(outer.free_octants, vec![1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(outer.split_refusal, None, "connected and idle: possible");
        assert!(outer
            .merge_refusal
            .as_deref()
            .is_some_and(|why| why.contains("no parent")));
        assert_eq!(
            (inner.nesting, inner.parent, inner.octant),
            (1, Some(root), Some(0))
        );
        assert_eq!(inner.free_octants, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert!(inner
            .split_refusal
            .as_deref()
            .is_some_and(|why| why.contains("not connected")));
        assert!(inner
            .merge_refusal
            .as_deref()
            .is_some_and(|why| why.contains("not known to be empty")));

        // Shards.
        assert_eq!(snapshot.shards.len(), 2);
        let (one, two) = (&snapshot.shards[0], &snapshot.shards[1]);
        assert_eq!((one.shard, one.role), (ShardId::new(1), ShardRole::Owner));
        assert!(one.connected && one.registered);
        assert_eq!(one.advertised, Some(addr(9_001)));
        assert_eq!(one.regions, vec![root]);
        let report = one.report.expect("shard 1 reported");
        assert_eq!((report.players, report.tick), (7, 0));
        assert!((report.load_metric - 0.07).abs() < 1e-9);
        assert_eq!(
            one.last_heard,
            Some(Heard {
                tick: 0,
                what: HeardKind::Report
            })
        );
        assert!(
            one.scale.is_some(),
            "the autoscaler tracks a reporting shard"
        );

        assert_eq!((two.shard, two.role), (ShardId::new(2), ShardRole::Owner));
        assert!(!two.connected && !two.registered);
        assert_eq!(two.advertised, None);
        assert_eq!(two.report, None, "no report is not a report of zero");
        assert_eq!(two.last_heard, None);
        assert_eq!(two.scale, None);

        // The report expires like it does for the autoscaler.
        for _ in 0..=REPORT_EXPIRY_TICKS {
            orch.tick();
        }
        assert_eq!(orch.snapshot().shards[0].report, None);
    }

    #[test]
    fn a_lost_control_link_is_logged_once_and_changes_no_ownership() {
        let mut orch = orchestrator_with_first_id(100);
        orch.note_connected([ShardId::new(1)]);
        let before = orch.events().len();
        orch.note_connected([ShardId::new(1)]);
        assert_eq!(
            orch.events().len(),
            before,
            "still connected: nothing to say"
        );

        orch.note_connected([]);
        let texts = event_texts(&orch);
        assert!(texts[0].starts_with("shard 1 lost"), "{texts:?}");
        orch.note_connected([]);
        assert_eq!(
            orch.events().len(),
            before + 1,
            "it is lost once, not every beat"
        );

        assert_eq!(orch.map().epoch, 1);
        assert_eq!(orch.map().regions.len(), 1, "losing a link moves no region");
        assert!(orch.drain_outbox().is_empty());
        assert!(!orch.snapshot().shards[0].connected);
        let reply = orch.handle_operator(&split(1, Some(0)));
        assert!(reply.message.contains("not connected"), "{reply:?}");
    }

    /// Mutation this catches: remembering whoever an ack *claims* to be from.
    /// An ack names its own sender, so a stray one could otherwise grow the
    /// "last heard" table (and the shard list on the page) without bound.
    #[test]
    fn an_ack_naming_an_unknown_shard_is_not_remembered() {
        let mut orch = orchestrator_with_first_id(100);
        for id in 500..600 {
            orch.handle_epoch_ack(ShardId::new(id), 2);
        }
        assert!(orch.last_heard.is_empty());
        assert_eq!(orch.snapshot().shards.len(), 1);
        // The map's owner is remembered.
        orch.handle_epoch_ack(ShardId::new(1), 2);
        assert_eq!(
            orch.snapshot().shards[0].last_heard.map(|heard| heard.what),
            Some(HeardKind::Ack)
        );
    }

    #[test]
    fn a_registration_and_a_refused_hello_are_logged() {
        let mut orch = orchestrator_with_first_id(100);
        let hello = |cluster: &str| ShardHello {
            cluster: cluster.to_owned(),
            shard: ShardId::new(1),
            endpoint: addr(9_001),
            region: None,
        };
        orch.handle_shard_hello(&hello(&OrchestratorConfig::default().cluster))
            .expect("registers");
        assert!(orch.handle_shard_hello(&hello("another-cluster")).is_err());
        let texts = event_texts(&orch);
        assert!(
            texts[0].starts_with("hello from 127.0.0.1:9001 refused:"),
            "{texts:?}"
        );
        assert_eq!(
            texts[1],
            "shard 1 registered from 127.0.0.1:9001 for region 40:0:0:0"
        );
    }

    /// Re-sending the map is the push that follows every commit, on request:
    /// current addresses, same ownership, same epoch.
    #[test]
    fn re_sending_the_map_queues_the_committed_map_for_every_owner() {
        let mut orch = orchestrator_with_a_child();
        let root = orch.config().root_node();
        let child = space_world::octree::children(root).expect("splits")[0];
        // Shard 2 has said hello from another address than the map was built with.
        orch.handle_shard_hello(&ShardHello {
            cluster: OrchestratorConfig::default().cluster,
            shard: ShardId::new(2),
            endpoint: addr(9_002),
            region: Some(ShardRegion::new(child, ShardId::new(2), addr(9_002))),
        })
        .expect("a known owner registers");
        let owners =
            |map: &ShardMap| -> Vec<_> { map.regions.iter().map(|r| (r.node, r.shard)).collect() };
        let before = owners(orch.map());

        let reply = orch.handle_operator(&OperatorCommand::PushMap);
        assert!(reply.accepted, "{reply:?}");
        let out = orch.drain_outbox();
        let to: Vec<ShardId> = out.iter().map(|(to, _)| *to).collect();
        assert_eq!(to, vec![ShardId::new(1), ShardId::new(2)]);
        for (_, message) in &out {
            let Message::DirectoryReply(push) = message else {
                panic!("expected a map push, got {message:?}");
            };
            assert_eq!(push.shard_map.epoch, 1, "the epoch is untouched");
            assert_eq!(owners(&push.shard_map), before, "ownership is untouched");
            let endpoint = push
                .shard_map
                .regions
                .iter()
                .find(|r| r.shard == ShardId::new(2))
                .map(|r| r.endpoint);
            assert_eq!(endpoint, Some(addr(9_002)), "the current address goes out");
        }
        assert!(orch.take_pending_spawn().is_empty());
        assert!(orch.take_pending_kill().is_empty());
        assert!(orch.snapshot().in_flight.is_none());
    }

    /// The handle a front end holds reads snapshots and nothing else: taking
    /// one does not need the run loop, and it never sees the peer token.
    #[test]
    fn an_attached_front_end_reads_the_snapshot_that_was_published() {
        let mut orch = connected_orchestrator();
        let handle = orch.attach_ops();
        assert_eq!(handle.snapshot().tick, 0);
        assert_eq!(*handle.snapshot(), orch.snapshot());
        // Ticks that are not on the publishing beat publish nothing new...
        orch.tick();
        orch.publish(false);
        assert_eq!(handle.snapshot().tick, 0);
        // ...the beat does, and so does anything new in the log.
        for _ in 1..SNAPSHOT_INTERVAL_TICKS {
            orch.tick();
        }
        orch.publish(false);
        assert_eq!(handle.snapshot().tick, SNAPSHOT_INTERVAL_TICKS);
        orch.tick();
        orch.note_connected([]);
        orch.publish(false);
        assert_eq!(handle.snapshot().tick, SNAPSHOT_INTERVAL_TICKS + 1);
        assert!(!format!("{:?}", handle.snapshot()).contains("secret"));
    }

    // ------------------------------------------------- the security review

    /// The map cannot grow past the configured limit, whoever asks: the
    /// operator (told why, with no side effect), the admin verb and the
    /// autoscaler. A merge makes room again.
    ///
    /// Mutation this catches: a limit that only the operator's pre-check knows
    /// (the autoscaler would keep splitting a hot shard for ever), or one that
    /// burns an id per refused attempt.
    #[test]
    fn a_split_at_the_region_limit_is_refused_whoever_asks() {
        let config = OrchestratorConfig {
            max_regions: 2,
            ..OrchestratorConfig::default()
        };
        let root = config.root_node();
        let ids =
            PersistentShardIdAllocator::with_sink(100, Box::new(MemoryCounterSink::default()));
        let mut orch = orchestrator_over(
            config,
            vec![ShardRegion::new(root, ShardId::new(1), addr(7_001))],
            ids,
        );
        orch.note_connected([ShardId::new(1)]);
        assert_eq!(orch.snapshot().max_regions, 2);

        // Below the limit a split is an ordinary epoch change.
        assert!(orch.handle_operator(&split(1, Some(0))).accepted);
        let _ = orch.take_pending_spawn();
        assert!(ack_the_prepare(&mut orch) > 0);
        assert_eq!(orch.map().regions.len(), 2);
        for _ in 0..MIN_EPOCH_INTERVAL_TICKS {
            orch.tick();
        }
        let _ = orch.drain_outbox();

        // At the limit: the operator is told, and nothing at all moves.
        let state = |orch: &Orchestrator<MockLauncher>| {
            (
                orch.map().clone(),
                orch.control.ids.peek(),
                orch.control.autoscaler.busy_until(),
            )
        };
        let before = state(&orch);
        for command in [split(1, Some(1)), split(1, None)] {
            let reply = orch.handle_operator(&command);
            assert!(!reply.accepted, "{reply:?}");
            assert!(
                reply
                    .message
                    .contains("already has 2 region(s) and the limit is 2")
                    && reply.message.contains("--max-regions"),
                "{reply:?}"
            );
        }
        assert_eq!(
            state(&orch),
            before,
            "an operator's refusal has no side effect"
        );
        // The page says so before anybody presses the button.
        let snapshot = orch.snapshot();
        assert!(snapshot.regions.iter().all(|region| region
            .split_refusal
            .as_deref()
            .is_some_and(|why| why.contains("limit is 2"))));

        // The admin verb and the autoscaler are refused too, and burn no id.
        let why = refusal(orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1))));
        assert!(why.contains("limit is 2"), "{why}");
        for _ in 0..(4 * crate::autoscale::SCALE_COOLDOWN_TICKS) {
            orch.handle_load_report(&report(1, 95));
            orch.tick();
        }
        assert_eq!(
            orch.map().regions.len(),
            2,
            "a hot shard is not split past the limit"
        );
        assert_eq!(
            orch.control.ids.peek(),
            before.1,
            "and no id was spent trying"
        );
        assert!(orch.take_pending_spawn().is_empty());
        let decision = orch
            .snapshot()
            .autoscaler
            .last_decision
            .expect("the autoscaler did decide to split");
        assert!(
            decision
                .outcome
                .as_ref()
                .is_err_and(|why| why.contains("limit is 2")),
            "{decision:?}"
        );
        assert!(orch.drain_outbox().is_empty(), "nothing was proposed");

        // A merge makes room, and then a split is accepted again.
        orch.handle_load_report(&report(100, 0));
        let reply = orch.handle_operator(&merge(100));
        assert!(reply.accepted, "{reply:?}");
        assert!(ack_the_prepare(&mut orch) > 0);
        assert_eq!(orch.map().regions.len(), 1);
        for _ in 0..MIN_EPOCH_INTERVAL_TICKS {
            orch.tick();
        }
        let reply = orch.handle_operator(&split(1, Some(1)));
        assert!(reply.accepted, "{reply:?}");
    }

    /// A map holding as many regions as the limit allows, each owned by a
    /// shard of its own that is connected and reports no players.
    fn orchestrator_with_a_full_map() -> Orchestrator<MockLauncher> {
        let config = OrchestratorConfig::default();
        let limit = config.max_regions;
        let mut regions = vec![ShardRegion::new(
            config.root_node(),
            ShardId::new(1),
            addr(7_001),
        )];
        // Carve children breadth-first until the map is full.
        let mut parent = 0;
        while regions.len() < limit {
            let children =
                space_world::octree::children(regions[parent].node).expect("the node splits");
            parent += 1;
            for child in children {
                if regions.len() == limit {
                    break;
                }
                let id = u32::try_from(regions.len()).expect("a small number") + 1;
                regions.push(ShardRegion::new(child, ShardId::new(id), addr(7_001)));
            }
        }
        let owners: Vec<ShardId> = regions.iter().map(|region| region.shard).collect();
        let ids =
            PersistentShardIdAllocator::with_sink(1_000, Box::new(MemoryCounterSink::default()));
        let mut orch = orchestrator_over(config, regions, ids);
        crate::topology::validate(orch.map()).expect("a valid map to start from");
        orch.note_connected(owners.iter().copied());
        for owner in &owners {
            orch.handle_load_report(&report(owner.0, 0));
        }
        orch.tick();
        orch
    }

    /// Building a snapshot plans nothing: for a map at the region limit it
    /// does not validate a single candidate map, and it is quick. The run loop
    /// builds one several times a second, on every logged event and on every
    /// operator request.
    ///
    /// Mutation this catches: working out the free octants (or whether a merge
    /// is possible) by planning each one — eight map clones and eight pairwise
    /// validations per region, which made a snapshot cubic in the map's size.
    #[test]
    fn a_snapshot_of_a_full_map_plans_nothing_and_is_quick() {
        let orch = orchestrator_with_a_full_map();
        let limit = orch.config().max_regions;
        let validations = || crate::topology::VALIDATE_CALLS.with(std::cell::Cell::get);

        let before = validations();
        let started = std::time::Instant::now();
        let mut snapshot = orch.snapshot();
        for _ in 1..50 {
            snapshot = orch.snapshot();
        }
        let took = started.elapsed();
        assert_eq!(
            validations(),
            before,
            "building a snapshot must not validate (or clone) a candidate map"
        );
        assert!(
            took < std::time::Duration::from_secs(2),
            "50 snapshots of a {limit}-region map took {took:?}"
        );

        // And it still says everything it said before.
        assert_eq!(snapshot.regions.len(), limit);
        assert_eq!(snapshot.shards.len(), limit);
        assert!(
            snapshot.regions[0].free_octants.is_empty(),
            "all eight octants of the root are carved out"
        );
        assert!(snapshot
            .regions
            .iter()
            .all(|region| region.split_refusal.is_some()));
        let mergeable = snapshot
            .regions
            .iter()
            .filter(|region| region.merge_refusal.is_none())
            .count();
        assert_eq!(
            mergeable,
            limit - 1,
            "every region but the root can be merged"
        );
        // What the page offers is what the planner would do.
        for region in &snapshot.regions {
            let planned: Vec<u8> = (0..8u8)
                .filter(|octant| {
                    crate::topology::split_plan(
                        orch.map(),
                        region.shard,
                        usize::from(*octant),
                        ShardId::NONE,
                    )
                    .is_ok()
                })
                .collect();
            assert_eq!(region.free_octants, planned, "{:?}", region.node);
        }
    }

    /// Mutation this catches: one full-map message per shard **per request**.
    /// A burst of "re-send map" requests would then fill every shard's queue,
    /// and an epoch prepare or commit offered behind them would be dropped.
    #[test]
    fn re_sending_the_map_many_times_between_two_beats_sends_it_once() {
        let mut orch = orchestrator_with_a_child();
        orch.note_connected([ShardId::new(1), ShardId::new(2)]);
        for _ in 0..50 {
            assert!(orch.handle_operator(&OperatorCommand::PushMap).accepted);
        }
        // An epoch change is proposed in the same beat, after the requests.
        assert!(orch.handle_operator(&split(1, Some(5))).accepted);
        let out = orch.drain_outbox();
        let pushes = out
            .iter()
            .filter(|(_, m)| matches!(m, Message::DirectoryReply(_)))
            .count();
        assert_eq!(pushes, 2, "one map per owner, not one per request");
        let first_push = out
            .iter()
            .position(|(_, m)| matches!(m, Message::DirectoryReply(_)))
            .expect("a map push");
        assert!(
            out[..first_push]
                .iter()
                .all(|(_, m)| matches!(m, Message::EpochPrepare(_)))
                && !prepares(&out[..first_push]).is_empty(),
            "the epoch traffic is in front of the pushes: {out:?}"
        );
        assert!(orch.drain_outbox().is_empty(), "and it is sent once");
    }

    /// Whoever can reach the control port can have hellos refused, with texts
    /// of their choosing, as fast as they like. That must cost neither the
    /// event log's history nor more than a few lines of standard output.
    ///
    /// Mutation this catches: logging a refused hello like any other event
    /// (256 of them, taking turns between two cluster names, erase everything
    /// an operator did), or writing each one to standard output.
    #[test]
    fn refused_hellos_cannot_flush_the_event_log_or_flood_standard_output() {
        let mut orch = connected_orchestrator();
        let kept = orch
            .handle_operator(&OperatorCommand::PauseAutoscaler)
            .event;
        let _ = orch.new_event_lines(); // what was logged so far is written
        let refused_hello = |orch: &mut Orchestrator<MockLauncher>, n: u32| {
            let cluster = if n.is_multiple_of(2) {
                "cluster-a"
            } else {
                "cluster-b"
            };
            let hello = ShardHello {
                cluster: cluster.into(),
                shard: ShardId::NONE,
                endpoint: addr(9_000),
                region: None,
            };
            assert!(orch.handle_shard_hello(&hello).is_err());
        };

        for n in 0..2_000 {
            refused_hello(&mut orch, n);
        }
        assert!(
            orch.events().get(kept).is_some(),
            "the operator's entry is still in the log"
        );
        assert!(
            orch.events().len() <= 2 + crate::events::PEER_EVENT_CAPACITY,
            "{} entries",
            orch.events().len()
        );
        assert_eq!(orch.events().peer_caused_total(), 2_000);
        let newest = orch.events().newest_first().next().expect("an entry");
        assert!(newest.peer_caused && newest.text.contains("refused"));

        // Standard output: a few lines, then silence for the rest of the window.
        let lines = orch.new_event_lines();
        assert_eq!(lines.len(), PEER_PRINTS_PER_WINDOW as usize, "{lines:?}");
        assert!(lines.iter().all(|line| line.contains("refused")));
        for n in 0..500 {
            refused_hello(&mut orch, n);
            assert!(orch.new_event_lines().is_empty());
        }
        // Something an operator did is written at once, as always.
        let _ = orch.handle_operator(&OperatorCommand::ResumeAutoscaler);
        let lines = orch.new_event_lines();
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(lines[0].contains("operator: resume the autoscaler"));

        // The next window opens with one line for everything that was held
        // back — all of it, also what the ring had long dropped — and a fresh
        // budget.
        for _ in 0..PEER_PRINT_WINDOW_TICKS {
            orch.tick();
        }
        refused_hello(&mut orch, 0);
        let lines = orch.new_event_lines();
        assert_eq!(lines.len(), 2, "{lines:?}");
        let held_back = 2_500 - u64::from(PEER_PRINTS_PER_WINDOW);
        assert!(
            lines[0].contains(&format!(
                "{held_back} more refused hello(s) were not written"
            )),
            "{lines:?}"
        );
        assert!(lines[1].contains("refused"));

        // A quiet window says nothing.
        for _ in 0..PEER_PRINT_WINDOW_TICKS {
            orch.tick();
        }
        assert!(orch.new_event_lines().is_empty());
    }

    // -------------------------------------------------------------- the scene

    use crate::snapshot::SceneSync;

    fn set_scene(scene: Scene) -> OperatorCommand {
        OperatorCommand::SetScene { scene }
    }

    fn at(scene: Scene, revision: u64) -> SceneState {
        SceneState::new(scene.id(), revision)
    }

    fn scene_ack(shard: u32, applied: SceneState) -> SceneAck {
        SceneAck {
            shard: ShardId::new(shard),
            applied,
        }
    }

    /// The `(recipient, state)` of every `SceneSet` in `outbox`.
    fn scene_sets(outbox: &[(ShardId, Message)]) -> Vec<(ShardId, SceneState)> {
        outbox
            .iter()
            .filter_map(|(to, m)| match m {
                Message::SceneSet(state) => Some((*to, *state)),
                _ => None,
            })
            .collect()
    }

    /// Root (shard 1) with octant 0 carved out for shard 2; both have said
    /// hello and hold a control link.
    fn two_registered_shards() -> Orchestrator<MockLauncher> {
        let mut orch = orchestrator_with_a_child();
        let root = orch.config().root_node();
        let child = space_world::octree::children(root).expect("splits")[0];
        for (shard, port, region) in [
            (1, 9_001, None),
            (
                2,
                9_002,
                Some(ShardRegion::new(child, ShardId::new(2), addr(9_002))),
            ),
        ] {
            orch.handle_shard_hello(&ShardHello {
                cluster: OrchestratorConfig::default().cluster,
                shard: ShardId::new(shard),
                endpoint: addr(port),
                region,
            })
            .expect("a known owner registers");
        }
        orch.note_connected([ShardId::new(1), ShardId::new(2)]);
        orch
    }

    fn sync_of(orch: &Orchestrator<MockLauncher>, shard: u32) -> SceneSync {
        let snapshot = orch.snapshot();
        let view = snapshot
            .shards
            .iter()
            .find(|view| view.shard == ShardId::new(shard))
            .unwrap_or_else(|| panic!("shard {shard} is in the snapshot"));
        snapshot.scene.sync_of(view.scene)
    }

    /// The whole operator path: the command is accepted, every shard that has
    /// said hello is sent the new scene exactly once, each confirmation shows
    /// in the snapshot and in the log — and the map is not touched.
    ///
    /// Mutation this catches: sending the scene only to the map's owners that
    /// happen to be the command's target, keeping the old revision (a shard
    /// would ignore the change), or counting a shard as confirmed when the
    /// message went out rather than when its ack came back.
    #[test]
    fn an_operator_scene_change_reaches_every_shard_and_each_confirmation_is_tracked() {
        let mut orch = two_registered_shards();
        let (map, next_id) = (orch.map().clone(), orch.control.ids.peek());
        assert_eq!(orch.scene(), at(Scene::DemoArea, 1), "the default scene");
        let snapshot = orch.snapshot();
        assert_eq!(
            (snapshot.scene.scene, snapshot.scene.revision),
            (Scene::DemoArea, 1)
        );
        assert_eq!(snapshot.scene.changed_tick, None);
        assert_eq!(sync_of(&orch, 1), SceneSync::Unconfirmed);

        // The shards confirm the scene they were handed with their welcome.
        orch.handle_scene_ack(&scene_ack(1, at(Scene::DemoArea, 1)));
        orch.handle_scene_ack(&scene_ack(2, at(Scene::DemoArea, 1)));
        assert_eq!(sync_of(&orch, 1), SceneSync::InSync);
        assert_eq!(
            event_texts(&orch)[0],
            "shard 2 confirmed scene demo-area (revision 1)"
        );
        assert!(orch.drain_outbox().is_empty(), "nothing was asked for yet");

        orch.tick();
        let reply = orch.handle_operator(&set_scene(Scene::Solar));
        assert!(reply.accepted, "{reply:?}");
        assert_eq!(
            reply.message,
            "scene demo-area -> solar (revision 2), sent to shard 1, shard 2"
        );
        assert_eq!(
            orch.events().get(reply.event).map(|e| e.text.clone()),
            Some(
                "switch every shard to scene solar: accepted: scene demo-area -> solar \
                 (revision 2), sent to shard 1, shard 2"
                    .to_owned()
            )
        );
        assert_eq!(orch.scene(), at(Scene::Solar, 2));
        let snapshot = orch.snapshot();
        assert_eq!(
            (snapshot.scene.scene, snapshot.scene.revision),
            (Scene::Solar, 2)
        );
        assert_eq!(snapshot.scene.changed_tick, Some(1));
        assert_eq!(
            (sync_of(&orch, 1), sync_of(&orch, 2)),
            (SceneSync::Behind, SceneSync::Behind),
            "told is not confirmed"
        );

        let out = orch.drain_outbox();
        assert_eq!(
            scene_sets(&out),
            vec![
                (ShardId::new(1), at(Scene::Solar, 2)),
                (ShardId::new(2), at(Scene::Solar, 2))
            ]
        );
        assert_eq!(out.len(), 2, "and nothing else: {out:?}");
        assert!(orch.drain_outbox().is_empty(), "it is sent once");

        orch.handle_scene_ack(&scene_ack(1, at(Scene::Solar, 2)));
        assert_eq!(
            (sync_of(&orch, 1), sync_of(&orch, 2)),
            (SceneSync::InSync, SceneSync::Behind)
        );
        assert_eq!(
            orch.scene_confirmed_by(ShardId::new(1)),
            Some(at(Scene::Solar, 2))
        );
        assert_eq!(
            event_texts(&orch)[0],
            "shard 1 confirmed scene solar (revision 2)"
        );
        let heard = orch
            .snapshot()
            .shards
            .iter()
            .find(|view| view.shard == ShardId::new(1))
            .and_then(|view| view.last_heard)
            .map(|heard| heard.what);
        assert_eq!(heard, Some(HeardKind::SceneAck));

        // A scene is not an epoch: no id, no region, no proposal.
        assert_eq!(orch.map(), &map);
        assert_eq!(orch.control.ids.peek(), next_id);
        assert!(orch.snapshot().in_flight.is_none());
        assert!(orch.take_pending_spawn().is_empty() && orch.take_pending_kill().is_empty());
    }

    /// A scene change is accepted while an epoch change is in flight and does
    /// not disturb it; an epoch change is accepted while a scene change is
    /// still unconfirmed. Neither waits for the other.
    ///
    /// Mutation this catches: routing the scene through `proposal_refusal` (it
    /// would be refused for "epoch 2 is still in flight" and for three seconds
    /// after every commit), or making a split wait for scene acks.
    #[test]
    fn a_scene_change_and_an_epoch_change_do_not_wait_for_each_other() {
        let mut orch = two_registered_shards();
        let _ = orch.drain_outbox();

        // An epoch change is prepared and nobody has acked it yet.
        assert!(orch.handle_operator(&split(1, Some(5))).accepted);
        let in_flight = orch.snapshot().in_flight.expect("epoch 2 is in flight");
        assert_eq!(in_flight.epoch, 2);

        // The scene changes anyway, at once.
        let reply = orch.handle_operator(&set_scene(Scene::Galaxy));
        assert!(reply.accepted, "{reply:?}");
        assert_eq!(orch.scene(), at(Scene::Galaxy, 2));
        let out = orch.drain_outbox();
        let prepared = prepares(&out);
        assert!(!prepared.is_empty(), "the prepare is still going out");
        assert_eq!(
            scene_sets(&out),
            vec![
                (ShardId::new(1), at(Scene::Galaxy, 2)),
                (ShardId::new(2), at(Scene::Galaxy, 2))
            ]
        );
        // The scene messages are behind the epoch traffic in the same batch.
        let first_scene = out
            .iter()
            .position(|(_, m)| matches!(m, Message::SceneSet(_)))
            .expect("a scene message");
        assert!(
            out[..first_scene]
                .iter()
                .all(|(_, m)| matches!(m, Message::EpochPrepare(_))),
            "{out:?}"
        );
        // The epoch in flight is exactly what it was.
        assert_eq!(
            orch.snapshot()
                .in_flight
                .as_ref()
                .map(|flight| flight.epoch),
            Some(2)
        );
        assert_eq!(orch.map().epoch, 1);

        // It commits as if nothing had happened, with the scene unconfirmed.
        for (shard, epoch) in prepared {
            orch.handle_epoch_ack(shard, epoch);
        }
        assert_eq!(orch.map().epoch, 2, "the split committed");
        crate::topology::validate(orch.map()).expect("the map stays valid");
        assert_eq!(
            sync_of(&orch, 1),
            SceneSync::Unconfirmed,
            "no scene ack came"
        );

        // Right after a commit a map change has to wait (H12) — a scene does not.
        assert!(!orch.handle_operator(&split(1, Some(6))).accepted);
        let reply = orch.handle_operator(&set_scene(Scene::Materials));
        assert!(reply.accepted, "{reply:?}");
        assert_eq!(orch.scene(), at(Scene::Materials, 3));

        // And with that scene still unconfirmed, the next map change starts.
        for _ in 0..MIN_EPOCH_INTERVAL_TICKS {
            orch.tick();
        }
        let _ = orch.drain_outbox();
        assert!(orch.handle_operator(&split(1, Some(6))).accepted);
        assert!(ack_the_prepare(&mut orch) > 0);
        assert_eq!(orch.map().epoch, 3);
        assert_eq!(
            orch.scene(),
            at(Scene::Materials, 3),
            "untouched by the commit"
        );
    }

    /// However many changes are asked for between two beats, a shard is sent
    /// one message, with the scene that is current when the beat comes. Asking
    /// for the scene that is already current changes nothing and sends it
    /// again.
    ///
    /// Mutation this catches: queueing one message per shard per request — a
    /// burst of requests would fill a shard's queue ahead of an epoch prepare.
    #[test]
    fn scene_changes_between_two_beats_cost_one_message_per_shard() {
        let mut orch = two_registered_shards();
        for scene in [Scene::Solar, Scene::Galaxy, Scene::Materials, Scene::Solar] {
            assert!(orch.handle_operator(&set_scene(scene)).accepted);
        }
        assert_eq!(
            orch.scene(),
            at(Scene::Solar, 5),
            "every change is a revision"
        );
        let out = orch.drain_outbox();
        assert_eq!(
            scene_sets(&out),
            vec![
                (ShardId::new(1), at(Scene::Solar, 5)),
                (ShardId::new(2), at(Scene::Solar, 5))
            ],
            "one message each, with the latest scene"
        );

        // The same scene again: no new revision, but it is sent again.
        let reply = orch.handle_operator(&set_scene(Scene::Solar));
        assert!(reply.accepted, "{reply:?}");
        assert!(
            reply.message.contains("already solar (revision 5)"),
            "{reply:?}"
        );
        assert_eq!(orch.scene(), at(Scene::Solar, 5));
        assert_eq!(orch.snapshot().scene.changed_tick, Some(0));
        assert_eq!(scene_sets(&orch.drain_outbox()).len(), 2);
    }

    /// A connected shard that has not confirmed the current revision is told
    /// again once a second, until it confirms; nobody else is.
    ///
    /// Mutation this catches: no reminder (a `SceneSet` lost to a full queue
    /// leaves the shard on the old scene for good), a reminder every tick (30
    /// messages a second per shard), or reminding a shard that is not
    /// connected or that already confirmed.
    #[test]
    fn a_shard_that_has_not_confirmed_is_reminded_once_a_second_until_it_does() {
        let mut orch = two_registered_shards();
        // Shard 2's link is gone; shard 1 is connected and has confirmed nothing.
        orch.note_connected([ShardId::new(1)]);
        let _ = orch.drain_outbox();

        let beat = |orch: &mut Orchestrator<MockLauncher>| {
            orch.tick();
            orch.remind_scene_laggards();
            scene_sets(&orch.drain_outbox())
        };
        let mut reminders = Vec::new();
        for _ in 0..(3 * SCENE_RESEND_INTERVAL_TICKS) {
            reminders.extend(beat(&mut orch));
        }
        assert_eq!(
            reminders,
            vec![(ShardId::new(1), at(Scene::DemoArea, 1)); 3],
            "once a second, to the connected shard only"
        );

        // It confirms: no more reminders.
        orch.handle_scene_ack(&scene_ack(1, at(Scene::DemoArea, 1)));
        for _ in 0..(3 * SCENE_RESEND_INTERVAL_TICKS) {
            assert!(beat(&mut orch).is_empty());
        }

        // The scene changes and the first message is "lost" (drained, never
        // acked): the reminder carries the current scene.
        assert!(orch.handle_operator(&set_scene(Scene::Solar)).accepted);
        let _ = orch.drain_outbox();
        let mut reminders = Vec::new();
        for _ in 0..SCENE_RESEND_INTERVAL_TICKS {
            reminders.extend(beat(&mut orch));
        }
        assert_eq!(reminders, vec![(ShardId::new(1), at(Scene::Solar, 2))]);

        // A shard reporting a revision the orchestrator never issued is not
        // asked again: it would change nothing.
        orch.handle_scene_ack(&scene_ack(1, at(Scene::Galaxy, 40)));
        for _ in 0..(2 * SCENE_RESEND_INTERVAL_TICKS) {
            assert!(beat(&mut orch).is_empty());
        }
        assert_eq!(sync_of(&orch, 1), SceneSync::Differs);
    }

    /// What a shard confirmed only moves forward, only for a shard the
    /// orchestrator knows, and is logged once — whatever arrives.
    ///
    /// Mutation this catches: believing an ack about any shard id (the table
    /// and the page grow with whatever a peer sends), letting a late ack
    /// replace a newer one (a shard would flip back to "behind"), or logging
    /// every ack (a peer could flush the event log).
    #[test]
    fn a_scene_ack_is_believed_only_about_a_known_shard_and_only_forwards() {
        let mut orch = two_registered_shards();
        assert!(orch.handle_operator(&set_scene(Scene::Solar)).accepted);
        let _ = orch.drain_outbox();
        let shards_before = orch.snapshot().shards.len();

        for id in 500..600 {
            orch.handle_scene_ack(&scene_ack(id, at(Scene::Solar, 2)));
        }
        assert_eq!(orch.snapshot().shards.len(), shards_before);
        assert!(orch.scene_confirmed_by(ShardId::new(500)).is_none());
        assert!(
            !event_texts(&orch).iter().any(|t| t.contains("shard 500")),
            "nothing is logged about a stranger"
        );

        orch.handle_scene_ack(&scene_ack(1, at(Scene::Solar, 2)));
        let logged = orch.events().len();
        // The same ack again, and the late answer to the revision before it.
        for _ in 0..50 {
            orch.handle_scene_ack(&scene_ack(1, at(Scene::Solar, 2)));
            orch.handle_scene_ack(&scene_ack(1, at(Scene::DemoArea, 1)));
        }
        assert_eq!(
            orch.scene_confirmed_by(ShardId::new(1)),
            Some(at(Scene::Solar, 2)),
            "a late ack does not move the shard back"
        );
        assert_eq!(orch.events().len(), logged, "and nothing more is logged");

        // A revision that was never issued is recorded and said once.
        orch.handle_scene_ack(&scene_ack(2, at(Scene::Galaxy, 9)));
        assert_eq!(sync_of(&orch, 2), SceneSync::Differs);
        assert!(
            event_texts(&orch)[0].starts_with("shard 2 reports scene galaxy (revision 9)"),
            "{:?}",
            event_texts(&orch)[0]
        );
        let logged = orch.events().len();
        orch.handle_scene_ack(&scene_ack(2, at(Scene::Galaxy, 10)));
        orch.handle_scene_ack(&scene_ack(2, at(Scene::Galaxy, 11)));
        assert_eq!(orch.events().len(), logged, "once, not for every repeat");
        assert_eq!(orch.scene(), at(Scene::Solar, 2), "and it changes no scene");
    }

    /// A shard that registers again (it restarted) has confirmed nothing yet,
    /// whatever its previous incarnation said; a shard that left the cluster is
    /// forgotten.
    #[test]
    fn a_shard_that_registers_again_has_to_confirm_again() {
        let mut orch = two_registered_shards();
        orch.handle_scene_ack(&scene_ack(1, at(Scene::DemoArea, 1)));
        assert_eq!(sync_of(&orch, 1), SceneSync::InSync);
        orch.handle_shard_hello(&ShardHello {
            cluster: OrchestratorConfig::default().cluster,
            shard: ShardId::new(1),
            endpoint: addr(9_001),
            region: None,
        })
        .expect("the owner registers again");
        assert_eq!(sync_of(&orch, 1), SceneSync::Unconfirmed);
        assert!(orch.scene_confirmed_by(ShardId::new(1)).is_none());

        // Shard 2 is merged away: nothing about it is kept.
        orch.handle_scene_ack(&scene_ack(2, at(Scene::DemoArea, 1)));
        orch.handle_load_report(&report(2, 0));
        assert!(orch.handle_operator(&merge(2)).accepted);
        assert!(ack_the_prepare(&mut orch) > 0);
        assert!(orch.scene_confirmed_by(ShardId::new(2)).is_none());
    }

    #[derive(Debug)]
    struct BrokenSceneSink;

    impl crate::scene::SceneSink for BrokenSceneSink {
        fn persist(&mut self, _state: SceneState) -> std::io::Result<()> {
            Err(std::io::Error::other("disk full"))
        }
    }

    /// Mutation this catches: telling the shards first and saving afterwards.
    /// A scene that cannot be saved is refused, with the reason, and nothing —
    /// not the revision, not a message — moves.
    #[test]
    fn a_scene_that_cannot_be_saved_is_refused_and_nothing_is_sent() {
        let mut orch = two_registered_shards();
        orch.control.scene =
            crate::scene::SceneStore::with_sink(Scene::DemoArea, Box::new(BrokenSceneSink));
        let _ = orch.drain_outbox();
        let reply = orch.handle_operator(&set_scene(Scene::Solar));
        assert!(!reply.accepted, "{reply:?}");
        assert!(
            reply.message.contains("could not be saved")
                && reply.message.contains("disk full")
                && reply.message.contains("still demo-area (revision 1)"),
            "{reply:?}"
        );
        assert_eq!(
            orch.events().get(reply.event).map(|e| e.severity),
            Some(Severity::Warn)
        );
        assert_eq!(orch.scene(), at(Scene::DemoArea, 1));
        assert_eq!(orch.snapshot().scene.changed_tick, None);
        assert!(orch.drain_outbox().is_empty(), "no shard was told anything");
    }

    /// The scene an operator chose, and its revision, are what a restarted
    /// orchestrator comes back with — and hands to the first shard that asks.
    /// A damaged scene file costs the scene, never the boot, and says so in
    /// the event log.
    #[test]
    fn an_operators_scene_survives_an_orchestrator_restart() {
        let state_path = std::env::temp_dir().join(format!(
            "space-orch-scene-restart-{}-{}.id",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        let scene_file = crate::scene::scene_path(&state_path);
        let _ = std::fs::remove_file(&state_path);
        let _ = std::fs::remove_file(&scene_file);
        let config = OrchestratorConfig {
            state_path: state_path.clone(),
            ..OrchestratorConfig::default()
        };
        let start = |config: &OrchestratorConfig| {
            let control = crate::boot::boot(config.clone(), "secret".into()).expect("boot");
            Orchestrator::new(control, std::sync::Arc::new(MockLauncher::default()))
        };

        let mut orch = start(&config);
        assert_eq!(orch.scene(), at(Scene::DemoArea, 1));
        assert!(orch.handle_operator(&set_scene(Scene::Solar)).accepted);
        assert!(orch.handle_operator(&set_scene(Scene::Galaxy)).accepted);
        assert_eq!(orch.scene(), at(Scene::Galaxy, 3));
        drop(orch);

        // The process is gone; a new one starts over the same state directory,
        // even with another initial scene on its command line.
        let restarted_with = OrchestratorConfig {
            initial_scene: Scene::Materials,
            ..config.clone()
        };
        let orch = start(&restarted_with);
        assert_eq!(orch.scene(), at(Scene::Galaxy, 3));
        assert_eq!(orch.snapshot().scene.path, Some(scene_file.clone()));
        assert_eq!(
            orch.scene_message(),
            Message::SceneSet(at(Scene::Galaxy, 3))
        );
        assert!(orch.events().is_empty(), "nothing to report");
        drop(orch);

        // The file is damaged: the orchestrator starts with the initial scene
        // and the operator is told.
        std::fs::write(&scene_file, "garbage").expect("damage the file");
        let orch = start(&restarted_with);
        assert_eq!(orch.scene(), at(Scene::Materials, 1));
        let texts = event_texts(&orch);
        assert!(
            texts
                .first()
                .is_some_and(|t| t.contains("is not usable") && t.contains("materials")),
            "{texts:?}"
        );
        let _ = std::fs::remove_file(&state_path);
        let _ = std::fs::remove_file(&scene_file);
    }
}
