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
//!     _      = clock.tick()        => { orch.tick(); drain outbox; launch; stop; poll; }
//! }}
//! ```
//!
//! [`Orchestrator::tick`] is the synchronous heart of that loop: it advances the
//! orchestrator's own clock, runs the epoch deadline, feeds the autoscaler the
//! latest report per shard, and stages the launcher work — the launch of every
//! shard that is missing ([`Orchestrator::take_pending_spawn`]) and the
//! stopping of every shard that has to go ([`Orchestrator::take_pending_kill`],
//! [`Orchestrator::take_pending_launch_kill`]) — for the async layer to
//! execute. The async [`Orchestrator::run`] wires those pieces together over
//! real QUIC; the tests drive the synchronous half directly.
//!
//! # The life of a shard (`docs/plan-sharding.md` §2)
//!
//! There is **one path for every shard**, started by a launcher or by hand: it
//! starts without a region and without an id and is given both when it says
//! hello ([`Orchestrator::handle_shard_hello`]) — the root while nobody owns
//! it, else the regions of a shard that died, else the node a split asked
//! for, else the next layout cell around the scene
//! ([`topology::layout_cells`]). For everything but the root that is an epoch
//! change, proposed **with the shard already connected**: the prepare goes to
//! every shard of the map and to the new one, all of them must ack before the
//! map changes ([`crate::epoch`]), and the 3 s ack deadline never has to cover
//! the start of a process. (It used to: a split planned the change, launched
//! a shard with the id and the node and waited for its hello inside the
//! deadline.) A shard owns nothing before its epoch commits, and is stopped
//! if it aborts.
//!
//! * **One change at a time.** A hello that arrives while a change is in
//!   flight is told to wait ([`crate::control::HelloRefusal::Wait`]) and the
//!   shard asks again. So is one that arrives while a shard of the map holds
//!   no control link: every one of them has to acknowledge a change, so none
//!   is proposed — and none asked for — while it is known that one cannot.
//! * **Asking for a region.** A split ([`Orchestrator::begin_carve`] is its
//!   general form) no longer proposes anything: it notes the node as *wanted*
//!   and a shard is launched; whichever shard registers next takes it.
//! * **The fleet.** [`Orchestrator::reconcile_launches`] stages a launch while
//!   fewer shards are live or starting than there should be: `--min-shards`,
//!   plus one for a wanted node and for every dead shard it replaces by a
//!   launch, never more than `--max-shards`. With `--min-shards 0` (the
//!   default) nothing is launched but a split's shard. A launch that is no
//!   longer needed — a shard started by hand took what it was started for —
//!   is stopped before it registers, and refused for good if its hello gets
//!   there first: it is not given a region nobody asked for. A launch that
//!   fails is followed by a pause that doubles (1 s to 30 s) until a launch's
//!   shard owns a region again; a join that runs into its deadline is such a
//!   failure.
//! * **Supervision.** The run loop asks the launcher twice a second whether
//!   what it started is alive, treats a registered shard whose control link
//!   closed as dead too, and one that holds its link but lets
//!   [`ACK_STRIKES`] map changes in a row run into their deadline
//!   ([`Orchestrator::declare_dead`]). A dead shard's
//!   regions go to the next shard that registers — a replacement launched for
//!   it, or one started by hand — under a **new id** (H15: ids are never
//!   reused; a process's guid serials start over, so the old id would mint
//!   the same guids twice), through an epoch change the dead shard is not
//!   asked to acknowledge, before any other change. When it commits the
//!   scene's revision is raised, once, so every shard drops the scene's bodies
//!   and the spawner builds them afresh — instead of half a scene, or some of
//!   it twice.
//!
//! A merged shard is stopped only when the merge **commits** — until then it
//! still owns its region. Stopping a shard is closing its control link (a
//! shard does not run without its orchestrator) and, for one the orchestrator
//! started, telling the launcher to stop its process.
//!
//! # Who is heard
//!
//! The cluster secret gates a hello, and the hello gates the connection
//! ([`crate::control::Inbound`]): a connection whose hello was refused is
//! closed and never read again, and one that was welcomed is taken to speak
//! for the shard it was welcomed as — a load report or an ack reaches
//! [`Orchestrator::handle_load_report_from`] and its siblings with that id,
//! and is dropped if it names another. The plain handlers
//! ([`Orchestrator::handle_load_report`], [`Orchestrator::handle_epoch_ack`],
//! [`Orchestrator::handle_scene_ack`]) believe the name in the message: they
//! are for callers that vouch for it, which the tests do and the run loop
//! does not.
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
//!
//! ## The orchestrator's revision is never below a shard's
//!
//! Shards and clients adopt a scene only at a **higher** revision than the
//! one they hold. That makes late and repeated messages harmless — and it
//! would make an orchestrator that has fallen *behind* inaudible for ever: one
//! whose scene file was lost or deleted starts counting at 1 again, and
//! nothing it says below the revision a shard (and that shard's clients)
//! already holds is ever adopted. Every switch would be "accepted" and change
//! nothing.
//!
//! So the rule is: **when a shard reports a revision at or above the
//! orchestrator's that is not the orchestrator's own state, the orchestrator
//! re-issues its current scene one above it** ([`Orchestrator::reconcile_scene`],
//! through [`crate::scene::SceneStore::raise_above`]: saved first, at most
//! once a second, logged). The orchestrator stays the authority on *which*
//! scene; only the number catches up, and from then on everybody hears it
//! again — the shards, and through them the clients, with no change to the
//! wire or to either of them. A reported revision absurdly far ahead
//! ([`crate::scene::MAX_REVISION_CATCH_UP`]) is not followed: it is shown and
//! logged as "not the cluster's", so one forged message cannot push the
//! revision to its ceiling.

use std::collections::{BTreeMap, BTreeSet};
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use space_protocol::control::{AdminCommand, AdminReply, LoadReport};
use space_protocol::shard::{EpochAck, ShardId, ShardMap, ShardRegion};
use space_protocol::{
    JoinRedirect, Message, Scene, SceneAck, SceneState, ShardHello, ShardWelcome,
};
use space_world::NodeRef;

use crate::autoscale::ScaleAction;
use crate::config::{
    LauncherKind, OrchestratorConfig, LAUNCH_BACKOFF_MAX_TICKS, LAUNCH_BACKOFF_TICKS,
    LAUNCH_REGISTER_TIMEOUT_TICKS, MIN_EPOCH_INTERVAL_TICKS, POLL_INTERVAL_TICKS,
    REPORT_EXPIRY_TICKS, WANTED_TIMEOUT_TICKS,
};
use crate::control::{as_seen_from, client_endpoint_of, ControlPlane, HelloRefusal};
use crate::epoch::{Outbound, Phase};
use crate::events::{EventKind, EventLog, Severity};
use crate::launcher::{LaunchId, Launcher, Poll, ShardHandle, SpawnSpec};
use crate::ops::{OperatorCommand, OperatorReply, OperatorRequest, OpsHandle, OpsPort};
use crate::player::{PlayerAnswer, PlayerServer};
use crate::scene::{MAX_REVISION_CATCH_UP, SCENE_RESEND_INTERVAL_TICKS};
use crate::snapshot::{
    node_path, octant_of, octant_signs, AutoscalerView, Decision, EpochInFlight, Heard, HeardKind,
    PlayerPortView, RegionView, ReportView, SceneView, ShardRole, ShardView, Snapshot,
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
enum Octant {
    /// The first free octant counting round from the one holding the region's
    /// centre (the autoscaler's choice).
    FromCentre,
    /// The first free octant counting round from this one (the admin verb).
    From(usize),
    /// Exactly this octant, or nothing (an operator named it).
    Exactly(usize),
}

/// A carve that was asked for ([`Orchestrator::begin_carve`]): the node is
/// **wanted** for a new shard and one is being started. Nothing is proposed
/// yet — the node's epoch is proposed when a shard has registered to take it
/// (it is welcomed with the id allocated for it then), and the map changes
/// when every recipient has acked, or not at all.
///
/// Displayed, it is the sentence the event log and an operator's answer use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CarveWanted {
    /// The node to be carved out.
    pub node: NodeRef,
    /// The owner of the enclosing region: who loses the node, and who gets it
    /// back if the carve is merged away.
    pub from: ShardId,
}

impl core::fmt::Display for CarveWanted {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "region {} is wanted for a new shard: one is being started, and the region is \
             carved out of {}'s for it when it has registered",
            node_path(self.node),
            self.from
        )
    }
}

/// A node that waits for a shard to own it: what a split or a carve asked
/// for. At most one at a time (one change at a time).
#[derive(Debug, Clone, PartialEq, Eq)]
struct Wanted {
    /// The node to carve.
    node: NodeRef,
    /// The owner of the region it is carved out of.
    from: ShardId,
    /// What the change is, up to the new owner's name: `"split of shard 1,
    /// region 41:… (octant 5, +x -y +z) goes to new"`.
    what: String,
    /// The tick it was asked for (it is given up after
    /// [`WANTED_TIMEOUT_TICKS`]).
    since: u64,
}

/// One launch the orchestrator asked its launcher for.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Launch {
    /// The host port handed to it (0: none, the in-process launcher).
    host_port: u16,
    /// The tick it was staged at.
    since: u64,
    /// Where its shard listens, once the launcher has said (its `spawn`
    /// returned). Until then the launch is not polled.
    endpoint: Option<SocketAddr>,
    /// The shard it became, once known: **starting** while `None`.
    shard: Option<ShardId>,
}

impl Launch {
    /// Whether this launch's shard listens at `at`: the address its launcher
    /// reported, or — before the launcher has said — the host port it was
    /// handed (each launch has its own; the launcher binds it on the
    /// orchestrator's host).
    fn listens_at(&self, at: SocketAddr) -> bool {
        match self.endpoint {
            Some(endpoint) => endpoint == at,
            None => self.host_port != 0 && self.host_port == at.port(),
        }
    }

    /// Whether nothing is known yet about where this launch's shard listens:
    /// no host port was handed to it (the in-process launcher binds one of
    /// its own choosing) and its launcher has not reported.
    fn address_unknown(&self) -> bool {
        self.endpoint.is_none() && self.host_port == 0
    }
}

/// How long after a shard registered it may still turn out to be a launch's
/// (its launcher has not reported yet), in ticks (5 s). While such a shard
/// exists no launch is stopped as surplus: it might be that very shard's
/// process. A launcher reports within a beat or two, so this only has to be
/// long enough never to be wrong.
pub const LAUNCH_LIMBO_TICKS: u64 = 5 * crate::config::TICK_HZ;

/// The shard the epoch in flight gives its first region(s).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Join {
    /// The shard that registered and was welcomed for the change.
    shard: ShardId,
    /// The dead shard whose regions it takes over, if this is a replacement.
    replaces: Option<ShardId>,
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
    /// Launches the loop must hand to the launcher (staged by
    /// [`Self::reconcile_launches`]: a shard is missing).
    pending_spawn: Vec<SpawnSpec>,
    /// Shards the loop must stop: close their control link (staged by a
    /// committed merge, a change that aborted under its new shard, a death).
    pending_kill: Vec<ShardId>,
    /// Launches the loop must tell the launcher to stop: the processes of the
    /// shards above that it started, and launches that never registered.
    pending_launch_kill: Vec<LaunchId>,
    /// The shard the merge **in flight** removes. It is killed when that epoch
    /// commits and left alone when it aborts: until the commit it still owns its
    /// region, so killing it any earlier leaves the map naming a dead owner.
    pending_merge: Option<ShardId>,
    /// The shard the carve or replacement **in flight** gives its region(s).
    /// It is registered and connected, and owns nothing until that epoch
    /// commits; if it aborts, the shard is stopped instead of running on
    /// unowned.
    pending_join: Option<Join>,
    /// The node a split or a carve asked for, waiting for a shard to register
    /// and take it.
    wanted: Option<Wanted>,
    /// Shards known to be dead that still own regions of the published map,
    /// each with whether the orchestrator had launched it. Their regions go to
    /// the next shards that register, before anything else changes.
    dead: BTreeMap<ShardId, bool>,
    /// Everything the launcher was asked to start and still runs (or is
    /// starting).
    launches: BTreeMap<LaunchId, Launch>,
    /// The last launch id handed out.
    last_launch: u64,
    /// Where the search for a free host port starts next (an offset into the
    /// range): ports are handed out round robin, so the port of a shard that
    /// just died is not the next one used.
    next_port: u32,
    /// Launches that failed in a row (drives the pause before the next one).
    /// Reset when **a launch's shard comes to own a region** — not merely
    /// because some other launched shard is registered.
    launch_failures_in_a_row: u32,
    /// No launch is staged before this tick (the pause after a failure).
    launch_backoff_until: u64,
    /// Messages the async layer must deliver, keyed by destination shard.
    outbox: Vec<(ShardId, Message)>,
    /// The endpoints learned from each shard's `ShardHello`: the **internal**
    /// address other shards dial it at, with an unspecified IP filled in from
    /// where its control connection comes from.
    endpoints: BTreeMap<ShardId, std::net::SocketAddr>,
    /// Where each registered shard's **players** are sent: the client
    /// endpoint its hello asked for, else its endpoint above. It goes into
    /// the map beside the endpoint ([`ShardRegion::client_endpoint`]).
    client_endpoints: BTreeMap<ShardId, std::net::SocketAddr>,
    /// Where the player port listens, once the run loop serves one.
    player_addr: Option<SocketAddr>,
    /// The player port's counters as of the last beat (for the operator
    /// view): hellos redirected, told "starting", and refused for want of
    /// room.
    player_counts: (u64, u64, u64),
    /// The tick each registered shard said hello at. A launch is only ever
    /// matched to a shard that registered **after** the launch was staged
    /// ([`Self::attribute_launches`]).
    registered_at: BTreeMap<ShardId, u64>,
    /// How many map changes in a row each shard has let run into their
    /// deadline without acknowledging them ([`ACK_STRIKES`] make it dead).
    ack_strikes: BTreeMap<ShardId, u32>,
    /// The reason of the last "wait" that will not pass by itself and was
    /// logged ([`Self::stuck`]): the same reason is not logged again until a
    /// hello has been served in between.
    logged_wait: Option<String>,
    /// Set by [`Self::stuck`] while a hello is being decided: the "wait" it
    /// is answered with is worth a line in the log, and whether its text is
    /// the peer's to choose.
    noted_wait: Option<(String, bool)>,
    /// Reports and acks the run loop dropped because they did not name the
    /// shard their connection was admitted as.
    misattributed: u64,

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
    /// What became of the launches the loop started, reported by their tasks
    /// and taken in on the next beat (a launch runs outside the loop, so it
    /// cannot touch the orchestrator).
    launch_results: Arc<std::sync::Mutex<Vec<LaunchOutcome>>>,
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
    /// The highest revision a shard reported that this orchestrator has to
    /// get above (see the module docs); carried out by
    /// [`Self::reconcile_scene`].
    scene_raise_wanted: Option<u64>,
    /// The tick the revision was last raised to get above a shard's.
    scene_raised_tick: Option<u64>,
    /// A replacement committed and the scene's revision could not be raised
    /// for it ([`Self::respawn_scene`]): tried again once a second by
    /// [`Self::reconcile_scene`] until it — or any other raise — succeeds.
    scene_respawn_wanted: Option<Respawn>,
    /// An operator asked for the committed map to be sent again; done once,
    /// when the outbox is next drained, however often it was asked for.
    map_push_pending: bool,
    /// The standard-output budget for peer-caused entries.
    peer_prints: PeerPrintBudget,
}

/// What a launch task reports back: the launch, and the launcher's handle or
/// why it could not start the shard.
type LaunchOutcome = (LaunchId, Result<ShardHandle, String>);

/// A scene re-issue that is still owed after a replacement
/// ([`Orchestrator::respawn_scene`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Respawn {
    /// The shard that was replaced.
    dead: ShardId,
    /// Its replacement.
    new: ShardId,
    /// The revision the scene had when the raise failed: once it is above
    /// that, by whatever cause, the scene's bodies have been rebuilt.
    at_revision: u64,
}

/// Why an epoch in flight was aborted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AbortCause {
    /// Its deadline passed with acks missing.
    Deadline,
    /// A shard that had to acknowledge it is gone.
    Gone(ShardId),
}

/// How many map changes in a row a connected shard may let run into their
/// deadline without acknowledging them before it is taken for dead.
///
/// A shard whose link is up but which does not answer holds up every change
/// of the map (every shard of it must ack), and with `--min-shards` every
/// join it spoils is launched again. One missed deadline can be a hiccup;
/// three in a row — nine seconds of not answering — is a shard that is stuck,
/// and it is replaced like any other dead one.
pub const ACK_STRIKES: u32 = 3;

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
            pending_launch_kill: Vec::new(),
            pending_merge: None,
            pending_join: None,
            wanted: None,
            dead: BTreeMap::new(),
            launches: BTreeMap::new(),
            last_launch: 0,
            next_port: 0,
            launch_failures_in_a_row: 0,
            launch_backoff_until: 0,
            outbox: Vec::new(),
            endpoints: BTreeMap::new(),
            client_endpoints: BTreeMap::new(),
            player_addr: None,
            player_counts: (0, 0, 0),
            registered_at: BTreeMap::new(),
            ack_strikes: BTreeMap::new(),
            logged_wait: None,
            noted_wait: None,
            misattributed: 0,
            events,
            last_heard: BTreeMap::new(),
            connected: BTreeSet::new(),
            autoscaler_paused: false,
            last_decision: None,
            in_flight: None,
            launch_results: Arc::new(std::sync::Mutex::new(Vec::new())),
            ops: None,
            published_revision: 0,
            printed_event: 0,
            scene_confirmed: BTreeMap::new(),
            scene_push_pending: false,
            scene_changed_tick: None,
            scene_raise_wanted: None,
            scene_raised_tick: None,
            scene_respawn_wanted: None,
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

    /// Drain the launches staged since the last drain: the shards the
    /// launcher has to start ([`Self::reconcile_launches`]).
    pub fn take_pending_spawn(&mut self) -> Vec<SpawnSpec> {
        std::mem::take(&mut self.pending_spawn)
    }

    /// Drain the shards staged for stopping — the loop closes their control
    /// links — each once: a committed merge's shard, the shard of a change
    /// that aborted, a shard found dead.
    pub fn take_pending_kill(&mut self) -> Vec<ShardId> {
        let mut seen = BTreeSet::new();
        std::mem::take(&mut self.pending_kill)
            .into_iter()
            .filter(|shard| seen.insert(*shard))
            .collect()
    }

    /// Drain the launches the launcher has to stop: the processes behind the
    /// shards of [`Self::take_pending_kill`] that the orchestrator had
    /// started, and launches that never registered.
    pub fn take_pending_launch_kill(&mut self) -> Vec<LaunchId> {
        std::mem::take(&mut self.pending_launch_kill)
    }

    /// The node that is waiting for a shard to register and take it, if a
    /// split or a carve asked for one.
    #[must_use]
    pub fn wanted_node(&self) -> Option<NodeRef> {
        self.wanted.as_ref().map(|wanted| wanted.node)
    }

    /// The shards known to be dead whose regions have not been given to a
    /// replacement yet.
    #[must_use]
    pub fn dead_shards(&self) -> Vec<ShardId> {
        self.dead.keys().copied().collect()
    }

    /// The launch that became `shard`, if the orchestrator started it.
    #[must_use]
    pub fn launch_of(&self, shard: ShardId) -> Option<LaunchId> {
        self.launches
            .iter()
            .find(|(_, launch)| launch.shard == Some(shard))
            .map(|(id, _)| *id)
    }

    /// How many launches have not become a registered shard yet.
    #[must_use]
    pub fn starting(&self) -> usize {
        self.launches
            .values()
            .filter(|launch| launch.shard.is_none())
            .count()
    }

    /// The shards that are alive as far as the orchestrator knows: they hold a
    /// control link and are not known to be dead.
    #[must_use]
    pub fn live_shards(&self) -> BTreeSet<ShardId> {
        self.connected
            .iter()
            .copied()
            .filter(|shard| !self.dead.contains_key(shard))
            .collect()
    }

    /// The address `shard` advertised in its hello — where other shards dial
    /// it; an unspecified IP filled in with the one its control connection
    /// comes from — if it has registered.
    #[must_use]
    pub fn endpoint_of(&self, shard: ShardId) -> Option<SocketAddr> {
        self.endpoints.get(&shard).copied()
    }

    /// Where `shard`'s players are sent — the client endpoint its hello
    /// asked for, else its endpoint — if it has registered. An unspecified
    /// IP there means "the host the player dialed".
    #[must_use]
    pub fn client_endpoint_of(&self, shard: ShardId) -> Option<SocketAddr> {
        self.client_endpoints.get(&shard).copied()
    }

    /// The shard a new player starts on: the owner of the scene's spawn
    /// point ([`space_scene::scene_anchor`] of the published map).
    #[must_use]
    pub fn spawn_owner(&self) -> Option<ShardId> {
        let map = self.control.map();
        let spawn = space_scene::scene_anchor(map)?;
        map.lookup(&spawn)
    }

    /// What the player port answers a client's hello with right now
    /// (`docs/plan-sharding.md` §4): a [`JoinRedirect`] naming the shard that
    /// owns the scene's spawn point, with the published map — or
    /// [`PlayerAnswer::Starting`] while there is no shard to send a player
    /// to:
    ///
    /// * nobody has claimed the root yet (the cluster is starting);
    /// * the spawn point's owner holds no control link or is known to be dead
    ///   (it is being replaced: a redirect would send the player to a shard
    ///   that is gone);
    /// * the map does not say where players reach it.
    ///
    /// A pure function of the orchestrator's state. The run loop publishes it
    /// to the player port on every beat ([`crate::player::PlayerServer`]),
    /// which is all that port ever learns.
    #[must_use]
    pub fn player_answer(&self) -> PlayerAnswer {
        match self.player_target() {
            Some(shard) => PlayerAnswer::Redirect(JoinRedirect {
                shard,
                shard_map: self.control.map().clone(),
            }),
            None => PlayerAnswer::Starting,
        }
    }

    /// The shard a new player is sent to right now, if there is one (see
    /// [`Self::player_answer`] for when there is not).
    #[must_use]
    pub fn player_target(&self) -> Option<ShardId> {
        self.spawn_owner().filter(|owner| {
            self.control.unclaimed_root_owner() != Some(*owner)
                && self.connected.contains(owner)
                && !self.dead.contains_key(owner)
                && self.control.map().client_endpoint_of(*owner).is_some()
        })
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

    /// Handle a `ShardHello`: decide what the shard gets, remember where it
    /// is, and return the `ShardWelcome` to send back.
    ///
    /// This is the one way a shard comes into the cluster, launched or started
    /// by hand (`docs/plan-sharding.md` §2): it says hello **without an id and
    /// without a region** and is given, in this order,
    ///
    /// 1. **the root**, while nobody has claimed it — under the id reserved at
    ///    startup, with no map change (the map already names that id);
    /// 2. **the regions of a dead shard**, if one is waiting to be replaced —
    ///    under a new id, through an epoch change the dead shard is not asked
    ///    to acknowledge;
    /// 3. **the node a split or a carve asked for** ([`Self::begin_carve`]);
    /// 4. otherwise **the next free layout cell** around the scene
    ///    ([`topology::layout_cells`]).
    ///
    /// For 2 to 4 the id is allocated here and the change is proposed here,
    /// **with the shard already connected**: the prepare is in the outbox for
    /// every recipient, the new shard included, and the 3 s ack deadline never
    /// has to cover a process start. The welcome carries the published map,
    /// in which the shard does not own its region yet — it owns it when the
    /// epoch commits, and is stopped if it aborts.
    ///
    /// # Errors
    ///
    /// [`HelloRefusal::Wait`] — the one refusal that is not final — when the
    /// hello cannot be served *yet*: an epoch change is in flight (one at a
    /// time), or a shard that would have to acknowledge the change is not
    /// connected. The shard asks again. Final: another cluster, a missing or
    /// wrong cluster secret, `--max-shards` (or `--max-regions`) reached, no
    /// region left to give, a claim the published map does not support
    /// ([`ControlPlane::handle_shard_hello`]).
    ///
    /// This form takes the hello as coming **from the address it advertises**
    /// (a test playing the shard; an in-process caller). The run loop, which
    /// knows where a hello's connection really comes from, uses
    /// [`Self::handle_shard_hello_from`].
    pub fn handle_shard_hello(&mut self, hello: &ShardHello) -> Result<ShardWelcome, HelloRefusal> {
        self.handle_shard_hello_from(hello, hello.endpoint.ip())
    }

    /// [`Self::handle_shard_hello`] for a hello whose control connection
    /// comes from `seen_from`.
    ///
    /// The address matters for two things:
    ///
    /// * a hello that advertises an endpoint **without an IP**
    ///   (`0.0.0.0:<port>`: a shard in a container that does not know its own
    ///   address) is recorded with `seen_from` as its IP
    ///   ([`crate::control::as_seen_from`]) — that is the address other
    ///   shards are told to dial, and the one every rule below works with;
    /// * a hello that advertises the address of a **live** registered shard
    ///   takes that shard's place only when it comes from that address's host
    ///   (`take_over_in_place`) — which a hello without an IP of its own
    ///   does by construction, and no other host's can fake.
    ///
    /// Where the shard's **players** are sent is recorded here too: the
    /// client endpoint the hello asks for, else its (filled-in) endpoint.
    ///
    /// # Errors
    ///
    /// As [`Self::handle_shard_hello`].
    pub fn handle_shard_hello_from(
        &mut self,
        hello: &ShardHello,
        seen_from: IpAddr,
    ) -> Result<ShardWelcome, HelloRefusal> {
        let hello = &as_seen_from(hello, seen_from);
        self.noted_wait = None;
        match self.admit(hello, seen_from) {
            Ok(welcome) => {
                let shard = welcome.shard;
                self.endpoints.insert(shard, hello.endpoint);
                self.client_endpoints
                    .insert(shard, client_endpoint_of(hello));
                self.registered_at.insert(shard, self.tick);
                // A hello arrives over a live control link by definition; the
                // run loop corrects this on every beat from what it sees.
                self.connected.insert(shard);
                self.heard(shard, HeardKind::Hello);
                // A shard that registers starts without a scene: nothing is
                // confirmed until it confirms.
                self.scene_confirmed.remove(&shard);
                // Whatever a hello was waiting for has passed.
                self.logged_wait = None;
                self.attribute_launches();
                self.log(
                    EventKind::Shard,
                    Severity::Good,
                    format!(
                        "{} registered from {} for region {}{}",
                        welcome.shard,
                        hello.endpoint,
                        node_path(welcome.region.node),
                        players_at(hello)
                    ),
                );
                Ok(welcome)
            }
            Err(refusal) => {
                // A final refusal is worth a line. Anybody who can reach the
                // control port can cause one, with any text, so it goes into
                // the ring kept for such entries and can never push out an
                // operator's or an epoch's.
                if refusal.is_final() {
                    self.events.push_peer_caused(
                        self.tick,
                        EventKind::Shard,
                        Severity::Warn,
                        format!("hello from {} refused: {refusal}", hello.endpoint),
                    );
                } else if let Some((why, peers_text)) = self.noted_wait.take() {
                    // "Not yet" is not worth a line while it passes by itself
                    // (a change in flight, a shard that is not connected yet):
                    // the shard asks again a few times a second until it is
                    // served. One that does **not** pass by itself is — once
                    // per reason, or the operator sees nothing but a launch
                    // that "did not register".
                    if self.logged_wait.as_deref() != Some(why.as_str()) {
                        let text = format!(
                            "the shard at {} is told to wait, and that will not pass by itself: \
                             {why}",
                            hello.endpoint
                        );
                        if peers_text {
                            self.events.push_peer_caused(
                                self.tick,
                                EventKind::Shard,
                                Severity::Warn,
                                text,
                            );
                        } else {
                            self.log(EventKind::Shard, Severity::Warn, text);
                        }
                        self.logged_wait = Some(why);
                    }
                }
                Err(refusal)
            }
        }
    }

    /// A "wait" that does not pass by itself: the answer for the shard, noted
    /// so that [`Self::handle_shard_hello_from`] writes it to the event log
    /// (once per reason). `peers_text` says the reason contains text of the
    /// peer's choosing, which goes to the bounded ring for such entries.
    fn stuck(&mut self, why: String, peers_text: bool) -> HelloRefusal {
        self.noted_wait = Some((why.clone(), peers_text));
        HelloRefusal::Wait(why)
    }

    /// Decide what `hello` gets (see [`Self::handle_shard_hello`]).
    fn admit(
        &mut self,
        hello: &ShardHello,
        seen_from: IpAddr,
    ) -> Result<ShardWelcome, HelloRefusal> {
        self.control.check_credentials(hello)?;
        if hello.shard != ShardId::NONE || hello.region.is_some() {
            // A claim on what the map already says. An id is confirmed once:
            // a shard that comes back is a new process, whose guid counter
            // starts over, so it is given a new id like any other shard.
            if hello.shard != ShardId::NONE && self.endpoints.contains_key(&hello.shard) {
                return Err(HelloRefusal::Refused(format!(
                    "{} has already registered: an id is used by one process, once. Start \
                     the shard without an id and it is given a new one (and the regions of \
                     the one it replaces)",
                    hello.shard
                )));
            }
            return self
                .control
                .handle_shard_hello(hello)
                .map_err(HelloRefusal::Refused);
        }
        self.assign(hello, seen_from)
    }

    /// Give a shard that registered with neither id nor region its place.
    fn assign(
        &mut self,
        hello: &ShardHello,
        seen_from: IpAddr,
    ) -> Result<ShardWelcome, HelloRefusal> {
        // 1. The root, while nobody has claimed it: the id reserved at startup
        //    and no map change.
        if self.control.unclaimed_root_owner().is_some() {
            return self
                .control
                .handle_shard_hello(hello)
                .map_err(HelloRefusal::Refused);
        }
        // One change at a time. This comes before everything below: a hello
        // that has to wait changes nothing — in particular it makes nobody
        // dead.
        if let Phase::Preparing { epoch, .. } = self.control.publisher.phase() {
            return Err(HelloRefusal::Wait(format!(
                "epoch {epoch} is still in flight; one change at a time"
            )));
        }
        // A shard at the very address a registered, live shard advertises.
        let at_its_address = self
            .endpoints
            .iter()
            .find(|(shard, endpoint)| {
                **endpoint == hello.endpoint && !self.dead.contains_key(shard)
            })
            .map(|(shard, _)| *shard);
        if let Some(old) = at_its_address {
            return self.take_over_in_place(old, hello, seen_from);
        }
        // 2. A dead shard's regions go first: until they have an owner again
        //    nothing else can change (the dead shard cannot acknowledge it).
        if let Some(dead) = self.dead.keys().next().copied() {
            return self.replace(dead, hello, seen_from);
        }
        // A shard the orchestrator started, for something that is no longer
        // missing (see `unneeded_launch`): it is not given a region nobody
        // asked for.
        if let Some(refusal) = self.unneeded_launch(hello) {
            return Err(refusal);
        }
        // Room for one more shard?
        let owners = self.owner_count();
        let limit = self.control.config.max_shards;
        if owners >= limit {
            return Err(HelloRefusal::Full(format!(
                "the cluster already has {owners} shard(s) and the limit is {limit} \
                 (--max-shards)"
            )));
        }
        if let Some(full) = self.region_limit_refusal() {
            return Err(HelloRefusal::Full(full));
        }
        // Everybody who has to acknowledge the change must be there to do it.
        if let Some(absent) = self.absent_recipient(&BTreeSet::new()) {
            return Err(HelloRefusal::Wait(not_connected(absent)));
        }
        // 3. The node somebody asked for, 4. else the next layout cell.
        let (node, what) = match &self.wanted {
            Some(wanted) => (wanted.node, wanted.what.clone()),
            None => {
                let root = self.control.config.root_node();
                let Some(cell) = topology::next_layout_cell(self.control.map(), root) else {
                    return Err(HelloRefusal::Refused(
                        "there is no region left to give a new shard: every layout cell \
                         around the scene has an owner"
                            .into(),
                    ));
                };
                (cell, format!("layout cell {} goes to new", node_path(cell)))
            }
        };
        // Whatever becomes of it, the request is served by this hello or not
        // at all: a node that can no longer be carved is not asked for again.
        let asked_for = self.wanted.take().is_some();
        match self.carve(node, &what, hello) {
            Ok(welcome) => Ok(welcome),
            Err(why) if asked_for => {
                self.log(
                    EventKind::Epoch,
                    Severity::Warn,
                    format!("not carried out: {what} shard ({why})"),
                );
                Err(HelloRefusal::Wait(why))
            }
            // Nothing the next hello would not run into as well (the id
            // counter cannot be written, the plan does not validate): said in
            // the log, once, not only to the shard.
            Err(why) => Err(self.stuck(why, false)),
        }
    }

    /// A hello from the address the **live** registered shard `old`
    /// advertises: `old`'s process is gone (two cannot listen on one address)
    /// and this is the process that came back in its place — or `old` itself,
    /// saying hello a second time because it never saw its welcome. Either
    /// way it is not handed the old id: `old` is taken for dead and the hello
    /// is given its regions under a new one, like any replacement.
    ///
    /// Three things keep that from being a way to knock a healthy shard out:
    ///
    /// * **Only when the hello is going to be served.** The caller has made
    ///   sure no change is in flight; here everybody who has to acknowledge
    ///   the replacement must be connected. A hello that is told to wait has
    ///   made nobody dead.
    /// * **Only from that address's own host.** A hello whose connection
    ///   comes from another IP than the one it advertises is told to wait:
    ///   if `old` really is gone its own link closes (at the latest when the
    ///   transport's idle timeout runs out) and the hello is then served as
    ///   an ordinary replacement; if it is not, nothing happens to it.
    /// * **The process is kept.** If the orchestrator launched `old`, the
    ///   launch is the process now saying hello: it is taken off the old id
    ///   (and matched to the new one by its address) instead of being killed.
    ///
    /// No launch is staged for `old` either (see [`Self::mark_dead`]): its
    /// regions are being taken by this very hello.
    fn take_over_in_place(
        &mut self,
        old: ShardId,
        hello: &ShardHello,
        seen_from: IpAddr,
    ) -> Result<ShardWelcome, HelloRefusal> {
        if seen_from.to_canonical() != hello.endpoint.ip().to_canonical() {
            return Err(self.stuck(
                format!(
                    "{old} is registered at {} and holds a control link; this hello names that \
                     address but comes from {seen_from}. It is served when {old}'s own link has \
                     closed",
                    hello.endpoint
                ),
                true,
            ));
        }
        let mut absent: BTreeSet<ShardId> = self.dead.keys().copied().collect();
        absent.insert(old);
        if let Some(missing) = self.absent_recipient(&absent) {
            return Err(HelloRefusal::Wait(not_connected(missing)));
        }
        let launch = self.launch_of(old);
        if let Some(entry) = launch.and_then(|launch| self.launches.get_mut(&launch)) {
            // Starting again, from now: the shard it became is gone, the
            // process is the one saying hello.
            entry.shard = None;
            entry.since = self.tick;
        }
        self.mark_dead(
            old,
            "a new shard registered from its address",
            launch.is_some(),
            false,
        );
        if self.dead.contains_key(&old) {
            self.replace(old, hello, seen_from)
        } else {
            // It owned nothing, so there is nothing to take over.
            self.assign(hello, seen_from)
        }
    }

    /// The refusal for a hello from **a shard the orchestrator started** that
    /// finds nothing it was started for: the root is taken, no shard is dead,
    /// no node is wanted, and the cluster has the shards it is to keep
    /// running. `None` for every other hello.
    ///
    /// A launch is staged for something that is missing — a dead shard, a
    /// wanted node, a shard short of `--min-shards`. If a shard started by
    /// hand takes that first, the launch is one too many; given the next
    /// layout cell it would be a shard (and a region) nobody asked for, and
    /// one more on every repeat. So it is refused for good and stopped.
    /// [`Self::reconcile_launches`] stops such a launch before it registers
    /// when it can; this is for the hello that gets there first.
    ///
    /// A shard started by hand still gets the next layout cell — that is the
    /// rule "a shard that registers is assigned a region". The two are told
    /// apart by the address the hello advertises, which the orchestrator
    /// knows for a launch that was handed a host port (`process`). The
    /// in-process launcher binds a port of its own choosing, so its hello
    /// cannot be told from a hand-started shard's: while such a launch is
    /// still starting and nothing is missing, the hello is told to wait —
    /// within a beat or two the launch is either matched to the shard it
    /// became or stopped as surplus, and whoever asks again is served.
    fn unneeded_launch(&mut self, hello: &ShardHello) -> Option<HelloRefusal> {
        let (owners, min) = (self.owner_count(), self.control.config.min_shards);
        if self.wanted.is_some() || owners < min {
            return None;
        }
        if let Some(launch) = self.launch_saying(hello) {
            let port = self
                .launches
                .remove(&launch)
                .map_or(0, |entry| entry.host_port);
            self.pending_launch_kill.push(launch);
            self.log(
                EventKind::Launcher,
                Severity::Info,
                format!(
                    "the shard of {launch}{} registered when nothing was left for it to take \
                     (the cluster has the {owners} shard(s) it is to have): it is stopped",
                    on_port(port)
                ),
            );
            return Some(HelloRefusal::Refused(
                "this shard was started for a region that another shard has taken since; the \
                 cluster has the shards it is to keep running, so it is not needed"
                    .into(),
            ));
        }
        let unplaced = self
            .launches
            .values()
            .any(|launch| launch.shard.is_none() && launch.address_unknown());
        unplaced.then(|| {
            HelloRefusal::Wait(
                "a shard the orchestrator started is still registering, and nothing else is \
                 missing"
                    .into(),
            )
        })
    }

    /// The launch whose shard is saying `hello`, if it is one the orchestrator
    /// started and has not matched to a shard yet: the one that listens where
    /// the hello says its shard does — the address its launcher reported, or,
    /// before that, the host port it was handed.
    fn launch_saying(&self, hello: &ShardHello) -> Option<LaunchId> {
        self.launches
            .iter()
            .find(|(_, launch)| launch.shard.is_none() && launch.listens_at(hello.endpoint))
            .map(|(id, _)| *id)
    }

    /// Plan the carve of `node` for the shard that just said `hello`:
    /// allocate its id, propose the epoch to every shard of the map and to
    /// it, and build its welcome. `what` words the change up to the new
    /// owner's name.
    ///
    /// The id is allocated only once a valid plan exists: a carve that cannot
    /// happen burns no id and writes nothing to the counter file.
    fn carve(
        &mut self,
        node: NodeRef,
        what: &str,
        hello: &ShardHello,
    ) -> Result<ShardWelcome, String> {
        // Planned once with a placeholder owner (the owner is irrelevant to
        // whether a plan validates), so a carve that cannot happen burns no id.
        topology::carve_plan(self.control.map(), node, ShardId::NONE)
            .map_err(|err| err.to_string())?;
        // The id space may be exhausted, or the counter cannot be written:
        // then nobody can be given a region.
        let new_id = self.control.ids.allocate().map_err(|err| err.to_string())?;
        let plan = topology::carve_plan(self.control.map(), node, new_id)
            .map_err(|err| err.to_string())?;
        let observer_ticks: Vec<(ShardId, u64)> =
            self.shard_ticks.iter().map(|(s, t)| (*s, *t)).collect();
        // (`None`: lost the race to an in-flight epoch — the caller's check
        // said Idle but the publisher disagreed.)
        let prepare = self
            .control
            .publisher
            .propose(
                plan.map.clone(),
                vec![plan.change],
                &observer_ticks,
                self.tick,
            )
            .ok_or_else(|| String::from("the publisher refused the proposal"))?;
        self.queue_outbound(&prepare);
        self.pending_join = Some(Join {
            shard: new_id,
            replaces: None,
        });
        self.in_flight = Some(format!("{what} {new_id}"));
        // The shard losing the node is on the autoscaler's cooldown, like the
        // parent of a split.
        self.control.autoscaler.note_action(plan.from, self.tick);
        Ok(ShardWelcome {
            shard: new_id,
            region: ShardRegion::new(plan.node, new_id, hello.endpoint)
                .with_client_endpoint(client_endpoint_of(hello)),
            // The published map: the node is not the new shard's yet.
            shard_map: self.control.map().clone(),
            peer_token: self.control.peer_token.expose().to_owned(),
        })
    }

    /// Give the regions of the dead shard `dead` to the shard that just said
    /// hello: a new id, and one epoch change that moves every region `dead`
    /// owns to it. No shard known to be dead is asked to acknowledge it.
    fn replace(
        &mut self,
        dead: ShardId,
        hello: &ShardHello,
        seen_from: IpAddr,
    ) -> Result<ShardWelcome, HelloRefusal> {
        let absent: BTreeSet<ShardId> = self.dead.keys().copied().collect();
        if let Some(missing) = self.absent_recipient(&absent) {
            return Err(HelloRefusal::Wait(not_connected(missing)));
        }
        let mut regions: Vec<NodeRef> = self
            .control
            .map()
            .regions
            .iter()
            .filter(|region| region.shard == dead)
            .map(|region| region.node)
            .collect();
        // The shallowest first: it is the region the welcome names.
        regions.sort_by_key(|node| node.depth);
        let Some(first) = regions.first().copied() else {
            // It owns nothing (any more): nothing to take over.
            self.dead.remove(&dead);
            self.forget_shard(dead);
            return self.assign(hello, seen_from);
        };
        // (Each of the three failures below would meet the next hello too:
        // they are said in the log, not only to the shard.)
        let new_id = match self.control.ids.allocate() {
            Ok(id) => id,
            Err(err) => return Err(self.stuck(err.to_string(), false)),
        };
        let mut map = self.control.map().clone();
        map.epoch += 1;
        for region in &mut map.regions {
            if region.shard == dead {
                region.shard = new_id;
                region.endpoint = hello.endpoint;
                region.client_endpoint = client_endpoint_of(hello);
            }
        }
        if let Err(why) = topology::validate(&map) {
            return Err(self.stuck(why, false));
        }
        let changes: Vec<_> = regions.iter().map(|node| (*node, new_id)).collect();
        let observer_ticks: Vec<(ShardId, u64)> =
            self.shard_ticks.iter().map(|(s, t)| (*s, *t)).collect();
        let Some(prepare) = self.control.publisher.propose_without(
            map,
            changes,
            &observer_ticks,
            self.tick,
            &absent,
        ) else {
            return Err(self.stuck("the publisher refused the proposal".into(), false));
        };
        self.queue_outbound(&prepare);
        self.pending_join = Some(Join {
            shard: new_id,
            replaces: Some(dead),
        });
        let paths: Vec<String> = regions.iter().map(|node| node_path(*node)).collect();
        self.in_flight = Some(format!(
            "replacement of {dead}: region {} goes to new {new_id}",
            paths.join(", ")
        ));
        Ok(ShardWelcome {
            shard: new_id,
            region: ShardRegion::new(first, new_id, hello.endpoint)
                .with_client_endpoint(client_endpoint_of(hello)),
            shard_map: self.control.map().clone(),
            peer_token: self.control.peer_token.expose().to_owned(),
        })
    }

    /// How many shards own a region of the published map.
    fn owner_count(&self) -> usize {
        self.control
            .map()
            .regions
            .iter()
            .map(|region| region.shard)
            .collect::<BTreeSet<_>>()
            .len()
    }

    /// A shard of the published map that could not acknowledge a change right
    /// now: it holds no control link. `except` are shards left out of the
    /// question — the dead ones a replacement is not sent to. `None` when
    /// every recipient is there.
    ///
    /// Every shard of the map must ack a change ([`crate::epoch`]), so one
    /// that is not there would hold it up until the deadline and abort it.
    /// Nothing is proposed while that is known in advance.
    fn absent_recipient(&self, except: &BTreeSet<ShardId>) -> Option<ShardId> {
        self.control
            .map()
            .regions
            .iter()
            .map(|region| region.shard)
            .find(|shard| !except.contains(shard) && !self.connected.contains(shard))
    }

    // ------------------------------------------------------------- supervision

    /// `shard` is gone: its process exited, its control link closed, a new
    /// shard came up at its address, or an operator said so. `why` words it
    /// for the log.
    ///
    /// Nothing about the map changes here. The shard stops counting as a
    /// recipient of any change, an epoch in flight that waits for it is
    /// aborted at once instead of at its deadline, what is left of the shard is
    /// staged for stopping, and — if it owns regions — it is remembered as
    /// **dead**: the next shard that registers is given its regions under a
    /// new id ([`Self::handle_shard_hello`]), before any other change, and one
    /// is launched for that if the orchestrator runs a launcher for the
    /// cluster ([`Self::reconcile_launches`]).
    pub fn declare_dead(&mut self, shard: ShardId, why: &str) {
        let managed = self.launch_of(shard).is_some();
        self.mark_dead(shard, why, managed, true);
    }

    /// [`Self::declare_dead`] with its two decisions made by the caller:
    /// whether the orchestrator had launched `shard` (`managed`: it is then
    /// replaced by a launch even with `--min-shards 0`), and whether to look
    /// at once for what is missing now (`reconcile`).
    ///
    /// `reconcile` is false in exactly one case — the hello that makes the
    /// shard dead is about to take its regions ([`Self::take_over_in_place`]).
    /// Staging a launch for it there would start a shard for regions that
    /// have an owner again a few lines later; the next tick looks instead.
    fn mark_dead(&mut self, shard: ShardId, why: &str, managed: bool, reconcile: bool) {
        if self.dead.contains_key(&shard) || !self.knows(shard) {
            return;
        }
        self.connected.remove(&shard);
        // A dead shard reports nothing more; its last report must not go on
        // driving the autoscaler.
        self.forget_reports(shard);
        self.ack_strikes.remove(&shard);
        self.stage_kill(shard);
        if self.owns_a_region(shard) {
            self.dead.insert(shard, managed);
            self.log(
                EventKind::Shard,
                Severity::Warn,
                format!(
                    "{shard} lost: {why}. Its region(s) go to a replacement under a new id; \
                     no other map change until then"
                ),
            );
        } else {
            self.log(
                EventKind::Shard,
                Severity::Warn,
                format!("{shard} lost: {why}. It owned no region"),
            );
            self.forget_shard(shard);
        }
        // An epoch it would have to acknowledge can never commit.
        let waits_for_it = matches!(
            self.control.publisher.phase(),
            Phase::Preparing { recipients, .. } if recipients.contains(&shard)
        );
        if waits_for_it {
            let missing = self.missing_acks();
            if let Some(abort) = self.control.publisher.abort() {
                self.settle_abort(&abort, AbortCause::Gone(shard), &missing);
            }
        }
        if reconcile {
            self.reconcile_launches();
        }
    }

    /// Whether `shard` is anybody to this orchestrator: it owns a region, has
    /// registered, or holds a control link.
    fn knows(&self, shard: ShardId) -> bool {
        self.connected.contains(&shard)
            || self.endpoints.contains_key(&shard)
            || self.owns_a_region(shard)
    }

    /// Stage `shard` for stopping: the loop closes its control link, and
    /// tells the launcher to stop its process if the orchestrator started it.
    /// From here on it does not count as a shard of the cluster: stopped on
    /// purpose is not a link that was "lost".
    fn stage_kill(&mut self, shard: ShardId) {
        self.connected.remove(&shard);
        self.pending_kill.push(shard);
        if let Some(launch) = self.launch_of(shard) {
            self.launches.remove(&launch);
            self.pending_launch_kill.push(launch);
        }
    }

    /// Take in what a launch task reported: the launcher's handle for the
    /// shard it started, or why it could not start one.
    ///
    /// The handle is how a launch is matched to the shard it became — by the
    /// id when the launcher knows it, otherwise by the address the shard
    /// listens on, which is the address its hello advertises
    /// ([`crate::launcher`]). The run loop calls this for every launch it
    /// started; a test that plays the launcher calls it too.
    pub fn note_launch_result(&mut self, launch: LaunchId, result: Result<ShardHandle, String>) {
        match result {
            Ok(handle) => {
                let Some(entry) = self.launches.get_mut(&launch) else {
                    // Given up on meanwhile (it took too long, or its shard was
                    // stopped): whatever came up must not stay up.
                    self.pending_launch_kill.push(launch);
                    return;
                };
                entry.endpoint = Some(handle.endpoint);
                if handle.shard.is_some() {
                    // The launcher knows which shard it became.
                    entry.shard = handle.shard;
                } else if entry
                    .shard
                    .is_some_and(|guess| self.endpoints.get(&guess) != Some(&handle.endpoint))
                {
                    // Matched by the host port alone, before the launcher
                    // had said where the shard listens — and that shard
                    // turns out to listen elsewhere.
                    entry.shard = None;
                }
                let became = entry.shard;
                self.attribute_launches();
                if became.is_some_and(|shard| self.owns_a_region(shard)) {
                    // This launch's shard owns a region: launching works.
                    self.launch_failures_in_a_row = 0;
                }
            }
            Err(why) => {
                if self.launches.remove(&launch).is_some() {
                    self.log(
                        EventKind::Launcher,
                        Severity::Warn,
                        format!("launching a shard failed ({launch}): {why}"),
                    );
                    self.launch_failed();
                }
            }
        }
    }

    /// The launcher says what `launch` started is no longer alive.
    ///
    /// A registered shard whose process is gone is dead
    /// ([`Self::declare_dead`]); a launch that died before it ever registered
    /// failed, and the next one waits a little. Either way **the launch is
    /// gone from the table afterwards**, and its host port free again: a
    /// launch whose shard is already known to be dead, or is no longer known
    /// at all (it was stopped when its join aborted), is simply removed and
    /// reaped.
    pub fn note_launch_dead(&mut self, launch: LaunchId) {
        let Some(entry) = self.launches.get(&launch) else {
            return;
        };
        match entry.shard {
            Some(shard) if self.dead.contains_key(&shard) || !self.knows(shard) => {
                self.launches.remove(&launch);
                self.pending_launch_kill.push(launch);
            }
            Some(shard) => self.declare_dead(shard, "its process exited"),
            None => {
                let port = entry.host_port;
                self.launches.remove(&launch);
                // Reap it.
                self.pending_launch_kill.push(launch);
                self.log(
                    EventKind::Launcher,
                    Severity::Warn,
                    format!(
                        "the shard of {launch}{} exited before it registered (its log says why)",
                        on_port(port)
                    ),
                );
                self.launch_failed();
            }
        }
    }

    /// The launches whose shard the launcher can be asked about: those it has
    /// reported a handle for.
    #[must_use]
    pub fn launches_to_poll(&self) -> Vec<LaunchId> {
        self.launches
            .iter()
            .filter(|(_, launch)| launch.endpoint.is_some())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Match launches to the shards they became: a launch whose shard is not
    /// known yet, and a registered shard that advertises the address that
    /// launch listens on ([`Launch::listens_at`]).
    ///
    /// Two shards at that address are **never** the launch's:
    ///
    /// * one that is **dead** — its endpoint is kept while it still owns
    ///   regions, and a replacement launched for it may be handed the very
    ///   port it had;
    /// * one that **registered before the launch was staged** — whoever it
    ///   is, it is not the process this launch started.
    ///
    /// Matched to either, a launch would be tied to a shard that is not its
    /// process: the death of that process would be put down to the wrong
    /// shard (or to none, with the launch and its port kept for ever), and a
    /// launch that fails would take a healthy shard with it.
    fn attribute_launches(&mut self) {
        let mut taken: BTreeSet<ShardId> = self
            .launches
            .values()
            .filter_map(|launch| launch.shard)
            .collect();
        let mut matched = Vec::new();
        for launch in self.launches.values_mut() {
            if launch.shard.is_some() {
                continue;
            }
            let found = self
                .endpoints
                .iter()
                .find(|(shard, at)| {
                    launch.listens_at(**at)
                        && !taken.contains(shard)
                        && !self.dead.contains_key(shard)
                        && self
                            .registered_at
                            .get(shard)
                            .is_some_and(|at| *at >= launch.since)
                })
                .map(|(shard, _)| *shard);
            if let Some(shard) = found {
                launch.shard = Some(shard);
                taken.insert(shard);
                matched.push(shard);
            }
        }
        if matched.iter().any(|shard| self.owns_a_region(*shard)) {
            // A launch's shard owns a region: launching works. (Not "some
            // launched shard is registered": with three healthy shards and a
            // fourth that keeps dying, the pause would never grow.)
            self.launch_failures_in_a_row = 0;
        }
    }

    /// A launch failed: pause before the next one, longer every time in a row.
    fn launch_failed(&mut self) {
        self.launch_failures_in_a_row = self.launch_failures_in_a_row.saturating_add(1);
        let doublings = self.launch_failures_in_a_row.saturating_sub(1).min(16);
        let pause = LAUNCH_BACKOFF_TICKS
            .saturating_mul(1 << doublings)
            .min(LAUNCH_BACKOFF_MAX_TICKS);
        self.launch_backoff_until = self.tick + pause;
    }

    /// Give up on what has waited too long: a launch that has not registered
    /// within [`LAUNCH_REGISTER_TIMEOUT_TICKS`] is stopped, and a node that
    /// has been wanted for [`WANTED_TIMEOUT_TICKS`] without a shard taking it
    /// is no longer wanted.
    fn expire_lifecycle(&mut self) {
        let now = self.tick;
        let overdue: Vec<(LaunchId, u16)> = self
            .launches
            .iter()
            .filter(|(_, launch)| {
                launch.shard.is_none()
                    && now.saturating_sub(launch.since) > LAUNCH_REGISTER_TIMEOUT_TICKS
            })
            .map(|(id, launch)| (*id, launch.host_port))
            .collect();
        for (launch, port) in overdue {
            self.launches.remove(&launch);
            self.pending_launch_kill.push(launch);
            self.log(
                EventKind::Launcher,
                Severity::Warn,
                format!(
                    "the shard of {launch}{} did not register within {} s and is stopped",
                    on_port(port),
                    LAUNCH_REGISTER_TIMEOUT_TICKS / crate::config::TICK_HZ
                ),
            );
            self.launch_failed();
        }
        let given_up = self
            .wanted
            .as_ref()
            .filter(|wanted| now.saturating_sub(wanted.since) > WANTED_TIMEOUT_TICKS)
            .map(|wanted| wanted.what.clone());
        if let Some(what) = given_up {
            self.wanted = None;
            self.log(
                EventKind::Epoch,
                Severity::Warn,
                format!(
                    "not carried out: {what} shard (no shard registered to take the region \
                     within {} s)",
                    WANTED_TIMEOUT_TICKS / crate::config::TICK_HZ
                ),
            );
        }
    }

    /// Stage a launch for every shard that is missing.
    ///
    /// The cluster should have `--min-shards` shards, one for every dead shard
    /// it is replacing by a launch, and one for the node a split asked for;
    /// it has the live ones and the launches that are still starting. While
    /// it has fewer than it should — and fewer than `--max-shards` — a launch
    /// is staged ([`Self::take_pending_spawn`]). With `--min-shards 0`, the
    /// default, that only ever happens for a split and for the replacement of
    /// a shard the orchestrator had launched itself: nothing else is started.
    ///
    /// Until the root is claimed only one launch is in flight: the shard that
    /// registers first gets the root, and with it the scene's spawn point —
    /// that is the first one launched, on the first port of the range.
    ///
    /// **And it stops what is no longer missing.** A launch is started for
    /// something — a dead shard, a wanted node, a shard short of
    /// `--min-shards` — and that can be gone before the launch has
    /// registered: a shard started by hand took the dead shard's regions, the
    /// wanted node was given up or taken. Such a launch would register, find
    /// nothing, and be given the next layout cell: one shard more than asked
    /// for, and one more on every repeat. So when there are more live and
    /// starting shards than there should be, the launches that have not
    /// registered yet are stopped, newest first (`stop_surplus_launches`).
    ///
    /// Called on every tick and whenever something changed what is missing;
    /// it stages nothing twice.
    pub fn reconcile_launches(&mut self) {
        let config = &self.control.config;
        let (min, max) = (config.min_shards, config.max_shards);
        let live = self.live_shards();
        let owners: BTreeSet<ShardId> = self
            .control
            .map()
            .regions
            .iter()
            .map(|region| region.shard)
            .filter(|shard| live.contains(shard))
            .collect();
        // A dead shard is replaced by a launch when the orchestrator launched
        // it, or runs the cluster's shards anyway.
        let replacing = self
            .dead
            .values()
            .filter(|managed| **managed || min > 0)
            .count();
        let want = min
            .max(owners.len() + replacing + usize::from(self.wanted.is_some()))
            .min(max);
        self.stop_surplus_launches(live.len(), want);
        if self.tick < self.launch_backoff_until {
            return;
        }
        let root_unclaimed = self.control.unclaimed_root_owner().is_some();
        loop {
            let starting = self.starting();
            let have = live.len() + starting;
            if have >= want || (root_unclaimed && starting >= 1) {
                return;
            }
            if !self.stage_launch() {
                return;
            }
        }
    }

    /// Stop the launches that are one too many: with `live` shards connected
    /// and `want` wanted, every launch beyond that which has **not registered
    /// yet** — newest first.
    ///
    /// A launch whose shard has registered is never stopped here, whatever
    /// the count says. That is easy to tell for a launch that was handed a
    /// host port: its shard is matched to it the moment it says hello. The
    /// in-process launcher's shard is only matched when the launcher reports
    /// back, a beat or two after the hello — until then the shard counts
    /// twice, as live and as starting. So while a shard that registered a
    /// moment ago is matched to no launch and such a launch is waiting to be
    /// matched ([`Self::launch_in_limbo`]), nothing is stopped: the next tick
    /// looks again.
    fn stop_surplus_launches(&mut self, live: usize, want: usize) {
        let surplus = (live + self.starting()).saturating_sub(want);
        if surplus == 0 || self.starting() == 0 || self.launch_in_limbo() {
            return;
        }
        let stopped: Vec<(LaunchId, u16)> = self
            .launches
            .iter()
            .rev()
            .filter(|(_, launch)| launch.shard.is_none())
            .take(surplus)
            .map(|(id, launch)| (*id, launch.host_port))
            .collect();
        for (launch, port) in stopped {
            self.launches.remove(&launch);
            self.pending_launch_kill.push(launch);
            self.log(
                EventKind::Launcher,
                Severity::Info,
                format!(
                    "the shard of {launch}{} is no longer needed (what it was started for has \
                     an owner again, or is no longer wanted) and is stopped before it registers",
                    on_port(port)
                ),
            );
        }
    }

    /// Whether a shard that registered within the last
    /// [`LAUNCH_LIMBO_TICKS`] is matched to no launch while a launch whose
    /// address is not known is still waiting to be matched: that shard may be
    /// that launch's (see [`Self::stop_surplus_launches`]).
    fn launch_in_limbo(&self) -> bool {
        let waiting = self
            .launches
            .values()
            .any(|launch| launch.shard.is_none() && launch.address_unknown());
        if !waiting {
            return false;
        }
        let matched: BTreeSet<ShardId> = self
            .launches
            .values()
            .filter_map(|launch| launch.shard)
            .collect();
        self.registered_at.iter().any(|(shard, at)| {
            self.tick.saturating_sub(*at) <= LAUNCH_LIMBO_TICKS
                && self.connected.contains(shard)
                && !matched.contains(shard)
        })
    }

    /// Stage one launch. `false` when no host port is free.
    fn stage_launch(&mut self) -> bool {
        let config = &self.control.config;
        // The in-process launcher binds an ephemeral port by itself.
        let host_port = if config.launcher == LauncherKind::Local {
            0
        } else {
            let Some(port) = self.free_host_port() else {
                let (lo, hi) = self.control.config.host_port_range;
                self.log(
                    EventKind::Launcher,
                    Severity::Warn,
                    format!("no shard can be started: every host port from {lo} to {hi} is in use"),
                );
                self.launch_failed();
                return false;
            };
            port
        };
        self.last_launch += 1;
        let launch = LaunchId(self.last_launch);
        self.launches.insert(
            launch,
            Launch {
                host_port,
                since: self.tick,
                endpoint: None,
                shard: None,
            },
        );
        let config = &self.control.config;
        self.pending_spawn.push(SpawnSpec {
            launch,
            cluster: config.cluster.clone(),
            control_addr: config.control_addr(),
            host_port,
            secret: self.control.cluster_secret().expose().to_owned(),
        });
        true
    }

    /// The next host port of the configured range that no launch holds,
    /// counting round from where the last search ended.
    fn free_host_port(&mut self) -> Option<u16> {
        let (lo, hi) = self.control.config.host_port_range;
        let span = u32::from(hi.checked_sub(lo)?) + 1;
        let used: BTreeSet<u16> = self.launches.values().map(|l| l.host_port).collect();
        (0..span)
            .map(|step| (self.next_port + step) % span)
            .find_map(|offset| {
                let port = lo + u16::try_from(offset).ok()?;
                (!used.contains(&port)).then_some((offset, port))
            })
            .map(|(offset, port)| {
                self.next_port = (offset + 1) % span;
                port
            })
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

    // ------------------------------------------------- who a message is from
    //
    // A report and both kinds of ack name their own sender. The three
    // handlers below this block believe that name: they are what a test (or
    // any caller that vouches for the sender) uses. The run loop does not
    // vouch — it knows which shard each control connection was admitted as,
    // and goes through the `_from` forms, which drop what does not name that
    // shard.

    /// A `LoadReport` that arrived on the control connection admitted as
    /// `from`: taken in ([`Self::handle_load_report`]) only if it is about
    /// `from` itself.
    pub fn handle_load_report_from(&mut self, from: ShardId, report: &LoadReport) {
        if report.shard == from {
            self.handle_load_report(report);
        } else {
            self.misattributed += 1;
        }
    }

    /// An `EpochAck` that arrived on the control connection admitted as
    /// `from`: counted ([`Self::handle_epoch_ack`]) only as `from`'s own.
    pub fn handle_epoch_ack_from(&mut self, from: ShardId, ack: &EpochAck) {
        if ack.shard == from {
            self.handle_epoch_ack(from, ack.epoch);
        } else {
            self.misattributed += 1;
        }
    }

    /// A `SceneAck` that arrived on the control connection admitted as
    /// `from`: believed ([`Self::handle_scene_ack`]) only about `from`.
    pub fn handle_scene_ack_from(&mut self, from: ShardId, ack: &SceneAck) {
        if ack.shard == from {
            self.handle_scene_ack(ack);
        } else {
            self.misattributed += 1;
        }
    }

    /// How many reports and acks were dropped by the three `_from` handlers
    /// because they named another shard than their connection's.
    #[must_use]
    pub fn misattributed(&self) -> u64 {
        self.misattributed
    }

    /// Ingest a `LoadReport`: remember it as the shard's latest and record the
    /// orchestrator tick it arrived at (its freshness, and the observer tick an
    /// epoch's `activation_tick` is derived from).
    ///
    /// A report naming a shard that owns no region in the published map is
    /// dropped: it is from a shard that was merged away or killed (its last
    /// reports can still be in flight), or from nobody the cluster knows. Kept,
    /// it would be replayed to the autoscaler every tick and acted on.
    ///
    /// The report is believed to be from the shard it names: see
    /// [`Self::handle_load_report_from`] for a report off the wire.
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

    /// `shard` has left the cluster (merged away, replaced, or stopped when
    /// the change that would have given it a region aborted): forget its load
    /// and where it was reachable.
    fn forget_shard(&mut self, shard: ShardId) {
        self.forget_reports(shard);
        self.endpoints.remove(&shard);
        self.client_endpoints.remove(&shard);
        self.registered_at.remove(&shard);
        self.ack_strikes.remove(&shard);
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
        // It answers: whatever deadlines it let pass before are forgotten.
        self.ack_strikes.remove(&shard);
        if let Some(commit) = self.control.publisher.ack(shard, epoch, self.tick) {
            let what = self.in_flight.take().unwrap_or_else(|| "map change".into());
            self.log(
                EventKind::Epoch,
                Severity::Good,
                format!("epoch {epoch} committed: {what}"),
            );
            // The epoch in flight settled as a commit: the shard it gave a
            // region is now an owner like any other, and a merge's shard owns
            // nothing.
            let joined = self.pending_join.take();
            let merged = self.pending_merge.take();
            if joined.is_some_and(|join| self.launch_of(join.shard).is_some()) {
                // A shard the orchestrator launched owns a region now:
                // launching works.
                self.launch_failures_in_a_row = 0;
            }
            self.queue_outbound(&commit);
            // Then push the committed map with every known shard's real endpoint
            // (plan §3.3, F9). The commit names ownership only; this is what lets
            // an existing shard dial a child it has never heard of, and what makes
            // the `Redirect` it hands a client point at the child's real address.
            // It rides the same per-shard FIFO as the commit, so a shard always
            // sees the commit first.
            self.control
                .publisher
                .set_endpoints(&self.endpoints, &self.client_endpoints);
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
                // skipped it. Queue the commit for it too (it had the prepare
                // staged; with this it learns it owns nothing), and only
                // *then* stage its kill: the loop offers the outbox to the
                // shards' queues before it closes any link. That is an order,
                // not a guarantee — the link is closed in the same beat, and a
                // close does not wait for what was queued to be written. A
                // shard whose link closes stops either way.
                self.outbox.push((gone, commit.message.clone()));
                self.outbox.push((gone, push.message.clone()));
                self.stage_kill(gone);
                self.forget_shard(gone);
            }
            if let Some(Join {
                shard: new,
                replaces: Some(dead),
            }) = joined
            {
                // The dead shard is out of the map: nothing waits for it any
                // more, and nothing is remembered about it.
                self.dead.remove(&dead);
                self.forget_shard(dead);
                self.respawn_scene(dead, new);
            }
            // The epoch settled: the publisher returns to Idle so the next
            // change can be proposed after the minimum interval.
            self.control.publisher.reset();
        }
    }

    /// Raise the scene's revision — the scene itself stays — after `new` took
    /// over the regions of the dead `dead`, so that every shard builds the
    /// scene's bodies **once, afresh**.
    ///
    /// The bodies the dead shard simulated went with it, the ones it had
    /// handed to its neighbours are still there, and if it owned the
    /// shallowest region its replacement is the new spawner of the scene's
    /// bodies, with new guids: without a new revision the world would keep
    /// half a scene, with one it would have some of it twice. A new revision
    /// makes every shard drop the scene's bodies and the spawner build them
    /// all again (`space-server`'s `ensure_scene`). It goes out behind the
    /// commit and the map push, so the replacement already owns its regions
    /// when it hears it.
    ///
    /// The raise is a write to the scene file and can fail. Then it is
    /// **owed** ([`Self::scene_respawn_wanted`]) and tried again once a
    /// second ([`Self::retry_scene_respawn`]) until it succeeds — or until
    /// the revision rises for another reason, which rebuilds the bodies just
    /// as well.
    fn respawn_scene(&mut self, dead: ShardId, new: ShardId) {
        let before = self.scene();
        match self.control.scene.raise_above(before.revision) {
            Ok(now) => {
                self.scene_respawn_wanted = None;
                self.scene_changed_tick = Some(self.tick);
                self.scene_push_pending = true;
                self.log(
                    EventKind::Scene,
                    Severity::Info,
                    format!(
                        "{new} replaced {dead}: the scene {} is re-issued as revision {} \
                         (it was {}) so that its bodies are built once, afresh",
                        now.scene, now.revision, before.revision
                    ),
                );
            }
            Err(err) => {
                let owed = Respawn {
                    dead,
                    new,
                    at_revision: before.revision,
                };
                // Said once, not once a second.
                if self.scene_respawn_wanted.replace(owed) != Some(owed) {
                    self.log(
                        EventKind::Scene,
                        Severity::Warn,
                        format!(
                            "{new} replaced {dead}, but the scene's revision could not be \
                             raised ({err}): the scene's bodies are not rebuilt until it can \
                             (tried again every second) or the scene is switched"
                        ),
                    );
                }
            }
        }
    }

    /// Try again to raise the scene's revision for a replacement that is
    /// still owed one ([`Self::respawn_scene`]), once per
    /// [`SCENE_RESEND_INTERVAL_TICKS`]. Nothing is owed any more once the
    /// revision is above what it was when the raise failed.
    fn retry_scene_respawn(&mut self) {
        let Some(owed) = self.scene_respawn_wanted else {
            return;
        };
        if self.scene().revision > owed.at_revision {
            self.scene_respawn_wanted = None;
            return;
        }
        if self.tick.is_multiple_of(SCENE_RESEND_INTERVAL_TICKS) {
            self.respawn_scene(owed.dead, owed.new);
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
    /// shard per revision — bounded whatever a peer sends.
    ///
    /// Two things read what a shard confirmed: [`Self::remind_scene_laggards`],
    /// which stops reminding it, and — when the shard turns out to run a
    /// revision at or above this orchestrator's that is not this
    /// orchestrator's — [`Self::reconcile_scene`], which gets above it (see
    /// the module docs). Nothing else: no ack changes *which* scene it is.
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
        } else if ack.applied.revision >= current.revision {
            // A revision this orchestrator never issued (or its own revision
            // with another scene): the shard will not adopt anything said at
            // the current revision, and neither will its clients.
            if ack.applied.revision - current.revision <= MAX_REVISION_CATCH_UP {
                // The orchestrator is the one that is behind (its scene file
                // was lost): get above the shard, on the next beat.
                self.scene_raise_wanted = Some(
                    self.scene_raise_wanted
                        .map_or(ack.applied.revision, |seen| seen.max(ack.applied.revision)),
                );
            } else if before.is_none_or(|held| held.revision <= current.revision) {
                // Too far ahead to be anything but nonsense. Said once; the
                // page keeps showing it.
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
    }

    /// The once-a-beat upkeep of the scene: first the raise a replacement is
    /// still owed ([`Self::retry_scene_respawn`]), then get above a revision
    /// a shard reported (the rule of the module docs, "The orchestrator's
    /// revision is never below a shard's"), then remind whoever has not
    /// confirmed ([`Self::remind_scene_laggards`]). The run loop calls this
    /// on every beat.
    pub fn reconcile_scene(&mut self) {
        self.retry_scene_respawn();
        self.catch_up_scene_revision();
        self.remind_scene_laggards();
    }

    /// Re-issue the current scene above the highest revision a shard reported,
    /// if one reported a revision this orchestrator is not above (see the
    /// module docs). At most once per [`SCENE_RESEND_INTERVAL_TICKS`]: each
    /// raise is a write to the scene file on the control loop, and whoever can
    /// send acks must not be able to turn that into a stream of them.
    fn catch_up_scene_revision(&mut self) {
        let Some(seen) = self.scene_raise_wanted else {
            return;
        };
        let before = self.scene();
        if seen < before.revision {
            // Already above it (an operator's change got there first).
            self.scene_raise_wanted = None;
            return;
        }
        if self
            .scene_raised_tick
            .is_some_and(|at| self.tick.saturating_sub(at) < SCENE_RESEND_INTERVAL_TICKS)
        {
            return; // not yet: it stays wanted
        }
        self.scene_raised_tick = Some(self.tick);
        match self.control.scene.raise_above(seen) {
            Ok(now) => {
                self.scene_raise_wanted = None;
                self.scene_changed_tick = Some(self.tick);
                self.scene_push_pending = true;
                self.log(
                    EventKind::Scene,
                    Severity::Warn,
                    format!(
                        "a shard runs scene revision {seen}, which this orchestrator is not \
                         above (it is at revision {}: was its scene file lost?). The scene {} \
                         is re-issued as revision {} so that every shard and client follows \
                         the orchestrator again",
                        before.revision, now.scene, now.revision
                    ),
                );
            }
            Err(err) => {
                // It stays wanted and is tried again in a second.
                self.log(
                    EventKind::Scene,
                    Severity::Warn,
                    format!(
                        "a shard runs scene revision {seen}, above this orchestrator's {}, and \
                         the revision could not be raised above it ({err}): that shard will \
                         not follow a scene change until it can",
                        before.revision
                    ),
                );
            }
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
    /// The second value is the new state when the scene changed, `None` when
    /// the scene asked for was already running.
    ///
    /// # Errors
    ///
    /// The new scene could not be persisted. Nothing changed then: the shards
    /// keep the scene they have, and so does a restart.
    fn set_scene(&mut self, scene: Scene) -> Result<(String, Option<SceneState>), String> {
        let before = self.control.scene.current();
        if self.control.scene.scene() == scene {
            // Nothing to change, and the answer must not read as if something
            // had been: no new revision, nothing sent. (A shard that has not
            // confirmed is reminded every second anyway.)
            let waiting: Vec<ShardId> = self
                .connected
                .iter()
                .copied()
                .filter(|shard| self.scene_confirmed.get(shard) != Some(&before))
                .collect();
            let message = if waiting.is_empty() {
                format!("already running {before}; nothing to do")
            } else {
                format!(
                    "already running {before}; nothing to change. Still waiting for {} to \
                     confirm it (told again every second)",
                    list_shards(&waiting)
                )
            };
            return Ok((message, None));
        }
        let told = list_shards(&self.scene_recipients().into_iter().collect::<Vec<_>>());
        let now = self.control.scene.set(scene).map_err(|err| {
            format!(
                "the scene could not be saved ({err}), so it was not changed: it is still {before}"
            )
        })?;
        self.scene_changed_tick = Some(self.tick);
        self.scene_push_pending = true;
        Ok((
            format!(
                "scene {} -> {} (revision {}), sent to {told}",
                before.scene, now.scene, now.revision
            ),
            Some(now),
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
            AdminCommand::ForceSplit(shard) => self.begin_split(*shard, Octant::From(0)),
            AdminCommand::ForceMerge(shard) => self.begin_merge(*shard),
            AdminCommand::KillShard(shard) => {
                // The same path a death takes: the shard is stopped, and its
                // regions (if it owns any) go to a replacement under a new id.
                if self.dead.contains_key(shard) {
                    Ok(format!("{shard} is already known to be dead"))
                } else if self.knows(*shard) {
                    self.declare_dead(*shard, "killed by the admin verb");
                    Ok(format!("{shard} is staged for killing"))
                } else {
                    Err(format!("{shard} is not a shard of this cluster"))
                }
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
    /// the epoch deadline, the autoscaler over the latest report per shard,
    /// then the fleet — what has waited too long is given up, and a launch is
    /// staged for every shard that is missing.
    ///
    /// This is the synchronous heart of the loop (plan §3.6). It never touches a
    /// socket or a wall clock, so a test drives it with a `for` loop.
    pub fn tick(&mut self) {
        self.tick += 1;
        // 1. Epoch deadline: an un-acked prepare aborts and the map is untouched.
        // (Who still owed an ack is read first, for the log only.)
        let missing = self.missing_acks();
        if let Some(abort) = self.control.publisher.timeout(self.tick) {
            self.settle_abort(&abort, AbortCause::Deadline, &missing);
        }
        // 2. Autoscaler: feed the latest *fresh* report per shard, get at most
        // one action.
        self.expire_stale_reports();
        self.run_autoscaler();
        // 3. The fleet.
        self.expire_lifecycle();
        self.reconcile_launches();
    }

    /// Carry out an abort the publisher produced: tell every recipient, free
    /// the publisher, log it, and undo what the aborted change had staged.
    /// `cause` is why it aborted, `missing` who still owed an ack.
    ///
    /// # What an abort at the deadline costs
    ///
    /// The shard the change had welcomed is stopped (below), and with
    /// `--min-shards` the next one is launched. Left at that, a join that
    /// cannot commit is a loop without an end and without a pause — launch,
    /// hello, new id, prepare, abort, kill — burning an id every three
    /// seconds. Two things end it:
    ///
    /// * **the joiner was the orchestrator's launch**: the launch failed, and
    ///   the next one waits like after any failed launch — longer every time
    ///   in a row ([`Self::launch_failed`]);
    /// * **a shard of the map did not acknowledge**: that is counted per
    ///   shard, and [`ACK_STRIKES`] deadlines missed in a row make it dead —
    ///   its link is closed and it is replaced like any dead shard. (An ack
    ///   for any change clears the count.)
    fn settle_abort(&mut self, abort: &Outbound, cause: AbortCause, missing: &[ShardId]) {
        self.queue_outbound(abort);
        self.control.publisher.reset();
        let what = self.in_flight.take().unwrap_or_else(|| "map change".into());
        let epoch = self.control.map().epoch + 1;
        let when = match cause {
            AbortCause::Deadline => "at its deadline".to_owned(),
            AbortCause::Gone(shard) => format!("because {shard} is gone"),
        };
        self.log(
            EventKind::Epoch,
            Severity::Warn,
            format!(
                "epoch {epoch} aborted {when}, the map is unchanged: {what}; no ack from {}",
                list_shards(missing)
            ),
        );
        // Nothing the aborted epoch planned happened. A merge's shard still
        // owns its region, so it is *not* stopped. The shard a carve or a
        // replacement had welcomed owns nothing, so it is: the abort above is
        // queued for it first, and its link is closed in the same beat —
        // whether it still reads the abort is not guaranteed, and does not
        // matter (a shard stops when its link closes). A shard started by
        // hand therefore exits and has to be started again. (A dead shard it
        // was to replace is still dead and still waiting: the next shard that
        // registers is given its regions.)
        self.pending_merge = None;
        let joiner = self.pending_join.take().map(|join| join.shard);
        if let Some(joiner) = joiner {
            if cause == AbortCause::Deadline && self.launch_of(joiner).is_some() {
                self.launch_failed();
            }
            self.stage_kill(joiner);
            self.forget_shard(joiner);
        }
        if cause != AbortCause::Deadline {
            return;
        }
        // Who let the deadline pass, and who did not.
        let mut stuck = Vec::new();
        for shard in &abort.to {
            if Some(*shard) == joiner {
                continue; // stopped above
            }
            if missing.contains(shard) {
                let strikes = self.ack_strikes.entry(*shard).or_insert(0);
                *strikes += 1;
                if *strikes >= ACK_STRIKES {
                    stuck.push(*shard);
                }
            } else {
                self.ack_strikes.remove(shard);
            }
        }
        for shard in stuck {
            self.declare_dead(
                shard,
                &format!(
                    "it holds a control link but did not acknowledge {ACK_STRIKES} map changes \
                     in a row"
                ),
            );
        }
    }

    /// Feed the autoscaler the latest fresh report per shard and carry out
    /// the one action it may decide on.
    fn run_autoscaler(&mut self) {
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
                    self.begin_split(shard, Octant::FromCentre),
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

    /// Turn a split decision into a **wanted node** and a staged launch: the
    /// carve ([`Self::begin_carve`]) of one child octant of `shard`'s region.
    ///
    /// `which` says where the search for an octant to carve starts:
    /// [`Octant::FromCentre`] means the region's centre octant (where a shard
    /// spawns joining ships, so the first split of a fresh region moves the
    /// players it was meant to relieve), [`Octant::From`] an explicit
    /// preference. Either way the octant carved is the first one, counting
    /// round from there, whose plan validates — an octant already carved out
    /// for another shard is skipped, so a region can be split again and again
    /// until none is left ([`topology::pick_split_octant`]).
    /// [`Octant::Exactly`] is an operator naming one octant: that one is carved
    /// or the split is refused, never a different octant than the one asked for.
    ///
    /// Nothing is proposed and no id is allocated here: the octant is noted
    /// as wanted, a shard is launched, and the epoch is proposed — with a
    /// freshly allocated id — when a shard registers to take it
    /// ([`Self::handle_shard_hello`]). A split that cannot happen (region too
    /// small, nothing left to carve) launches nothing.
    ///
    /// Returns what was started, in words.
    ///
    /// # Errors
    ///
    /// Why the split was not started. Apart from "an epoch is in flight" the
    /// shard is put on cooldown first (a shard that cannot be split is not
    /// helped by retrying every tick); the caller logs or reports the reason.
    fn begin_split(&mut self, shard: ShardId, which: Octant) -> Result<String, String> {
        if !self
            .control
            .publisher
            .can_propose(self.tick, MIN_EPOCH_INTERVAL_TICKS)
        {
            return Err("an epoch change is in flight or was committed too recently".into());
        }
        if let Some(waiting) = self.wanted_refusal() {
            return Err(waiting);
        }
        let outcome = self.split_octant(shard, which);
        // Started or not, the shard is on cooldown from here: a split that was
        // asked for must settle first, and one that cannot happen (the map is
        // full, the region too small, nothing left to carve) is not helped by
        // retrying every tick.
        self.control.autoscaler.note_action(shard, self.tick);
        outcome
    }

    /// The part of [`Self::begin_split`] behind its first checks: pick the
    /// octant, then ask for its node like for any other ([`Self::want`]).
    fn split_octant(&mut self, shard: ShardId, which: Octant) -> Result<String, String> {
        // The map or the cluster is full: no split, whoever asks (the
        // autoscaler included). Said before an octant is looked for, so that
        // is the reason given.
        if let Some(full) = self.region_limit_refusal() {
            return Err(full);
        }
        if let Some(full) = self.shard_limit_refusal() {
            return Err(full);
        }
        let map = self.control.map();
        let octant = match which {
            Octant::FromCentre => {
                let first = topology::region_of(map, shard)
                    .map_or(0, |region| topology::centre_octant(region.node));
                topology::pick_split_octant(map, shard, first)
            }
            Octant::From(first) => topology::pick_split_octant(map, shard, first),
            // Not picked: named. Whether it is still free is the carve's call.
            Octant::Exactly(octant) => Ok(octant),
        }
        .map_err(|err| err.to_string())?;
        let child = topology::octant_node(map, shard, octant).map_err(|err| err.to_string())?;
        let signs = octant_signs(u8::try_from(octant % 8).unwrap_or(0));
        let wanted = self.want(
            child,
            format!(
                "split of {shard}, region {} (octant {octant}, {signs}) goes to new",
                node_path(child)
            ),
        )?;
        Ok(wanted.to_string())
    }

    /// Ask for `node` to be carved out of the region that encloses it, for a
    /// **new shard**: the general form of a split, and the one entry point
    /// through which somebody asks for a region to come into being.
    ///
    /// `node` may be any octree node inside an existing region of the
    /// published map that is not a region itself — a child octant, or a single
    /// cell (depth 64) under the depth-40 root; see [`topology::carve_plan`].
    /// What happens is what happens for a split, step for step: the plan is
    /// checked now, the node is noted as **wanted**, and a shard is launched
    /// for it ([`Self::take_pending_spawn`]). When a shard has registered —
    /// that one, or any other that says hello without a region first — a
    /// fresh id is allocated for it and the change is proposed as **one
    /// epoch** to every shard of the map plus the new one
    /// ([`Self::handle_shard_hello`]); it commits when everybody has acked, or
    /// aborts at the deadline (and the new shard is stopped). If no shard
    /// registers within [`WANTED_TIMEOUT_TICKS`] the request is given up.
    ///
    /// A refusal changes nothing at all.
    ///
    /// Returns what was asked for ([`CarveWanted`]; it reads as a sentence).
    ///
    /// # Errors
    ///
    /// Why the carve was not started: an epoch change is in flight or was
    /// committed less than [`MIN_EPOCH_INTERVAL_TICKS`] ago, another node is
    /// still waiting for its shard, the map already holds
    /// [`OrchestratorConfig::max_regions`] or the cluster
    /// [`OrchestratorConfig::max_shards`], a shard of the map is not
    /// connected (it could not acknowledge the change), or the node cannot be
    /// carved ([`topology::TopologyError`]).
    pub fn begin_carve(&mut self, node: NodeRef) -> Result<CarveWanted, String> {
        if let Some(busy) = self.proposal_refusal() {
            return Err(busy);
        }
        if let Some(full) = self.region_limit_refusal() {
            return Err(full);
        }
        if let Some(full) = self.shard_limit_refusal() {
            return Err(full);
        }
        let wanted = self.want(node, format!("carve of region {} for new", node_path(node)))?;
        self.control.autoscaler.note_action(wanted.from, self.tick);
        Ok(wanted)
    }

    /// Note `node` as wanted for a new shard and stage the launch of one.
    /// `what` words the change up to the new owner's name. The callers have
    /// checked that a change may be asked for and that there is room.
    fn want(&mut self, node: NodeRef, what: String) -> Result<CarveWanted, String> {
        // Planned with a placeholder owner (the owner is irrelevant to whether
        // a plan validates): a carve that cannot happen is refused here.
        let plan = topology::carve_plan(self.control.map(), node, ShardId::NONE)
            .map_err(|err| err.to_string())?;
        // And one that nobody could acknowledge is not asked for either.
        if let Some(absent) = self.absent_recipient(&BTreeSet::new()) {
            return Err(not_connected(absent));
        }
        self.wanted = Some(Wanted {
            node,
            from: plan.from,
            what,
            since: self.tick,
        });
        self.reconcile_launches();
        Ok(CarveWanted {
            node,
            from: plan.from,
        })
    }

    /// Turn a merge decision into a topology plan + epoch proposal. The merged
    /// shard is killed when (and only if) that epoch commits.
    ///
    /// The shard's region goes back to the owner of the region around it
    /// ([`topology::merge_target`]): its parent node's after an octant split,
    /// the nearest enclosing region's — however many levels up — for a carved
    /// cell.
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
        if let Some(waiting) = self.wanted_refusal() {
            return Err(waiting);
        }
        // A merge takes a shard away; `--min-shards` says how many stay.
        if let Some(needed) = self.min_shards_refusal(shard) {
            self.control.autoscaler.note_action(shard, self.tick);
            return Err(needed);
        }
        // Every shard of the map has to acknowledge a merge too.
        if let Some(absent) = self.absent_recipient(&BTreeSet::new()) {
            return Err(not_connected(absent));
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
        // The merged region folds back into the region around it — when the epoch
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
    /// ([`crate::control::ControlServer::connected`]). **A shard that was
    /// connected and no longer is, is dead** ([`Self::declare_dead`]): a shard
    /// registers once and stops itself when it loses the orchestrator, so a
    /// closed link is not something that comes back. Nothing about the map
    /// changes here — the dead shard's regions go to the next shard that
    /// registers, through an epoch change. The set is also what every "could
    /// this shard acknowledge a change?" question reads.
    pub fn note_connected(&mut self, live: impl IntoIterator<Item = ShardId>) {
        let live: BTreeSet<ShardId> = live.into_iter().collect();
        let lost: Vec<ShardId> = self.connected.difference(&live).copied().collect();
        self.connected = live
            .into_iter()
            .filter(|shard| !self.dead.contains_key(shard))
            .collect();
        for shard in lost {
            // (Still in `connected` as far as `declare_dead` can tell would
            // be wrong: put it back for the one call that takes it out.)
            self.connected.insert(shard);
            self.declare_dead(shard, "its control link closed");
        }
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
        if let Some(full) = self.shard_limit_refusal() {
            return Some(full);
        }
        if let Some(busy) = self.proposal_refusal() {
            return Some(busy);
        }
        // The shard being split first (it is the one the operator named), then
        // everybody else: every shard of the map has to acknowledge a change.
        if !self.connected.contains(&shard) {
            return Some(not_connected(shard));
        }
        self.absent_recipient(&BTreeSet::new()).map(not_connected)
    }

    /// Why an operator's merge of `shard` would be refused right now, or
    /// `None` when it would be accepted. Changes nothing (see
    /// [`Self::split_refusal`]).
    ///
    /// The rules are `begin_merge`'s: the region must have an enclosing region
    /// owned by another shard (its parent node, or — for a carved cell — the
    /// nearest region around it, however many levels up), and the shard's own
    /// fresh load report must say it has no players — a shard that has not
    /// reported is not known to be empty (H13).
    #[must_use]
    pub fn merge_refusal(&self, shard: ShardId) -> Option<String> {
        // Looked up, not planned (see `split_refusal`).
        if let Err(err) = topology::merge_target(self.control.map(), shard) {
            return Some(err.to_string());
        }
        if let Some(needed) = self.min_shards_refusal(shard) {
            return Some(needed);
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
            .or_else(|| self.absent_recipient(&BTreeSet::new()).map(not_connected))
    }

    /// Why the cluster cannot take one more shard, if it cannot: it already
    /// has [`OrchestratorConfig::max_shards`].
    fn shard_limit_refusal(&self) -> Option<String> {
        let (shards, limit) = (self.owner_count(), self.control.config.max_shards);
        (shards >= limit).then(|| {
            format!(
                "the cluster already has {shards} shard(s) and the limit is {limit} \
                 (--max-shards): merge a region away first, or restart the orchestrator with \
                 a higher limit"
            )
        })
    }

    /// Why `shard` may not be merged away, if it may not: the cluster has no
    /// more shards than the orchestrator is to keep running (`--min-shards`).
    /// The merge would commit, the shard be stopped — and the reconciler
    /// start one in its place at once, which is given the next layout cell.
    fn min_shards_refusal(&self, shard: ShardId) -> Option<String> {
        let (shards, min) = (self.owner_count(), self.control.config.min_shards);
        (shards <= min).then(|| {
            format!(
                "the cluster has {shards} shard(s) and is to keep {min} running (--min-shards): \
                 with {shard} merged away the orchestrator would only start another one"
            )
        })
    }

    /// Why no further region can be asked for right now, if none can: one is
    /// still waiting for its shard (one change at a time).
    fn wanted_refusal(&self) -> Option<String> {
        self.wanted.as_ref().map(|wanted| {
            format!(
                "region {} is still waiting for its new shard to register; one change at a time",
                node_path(wanted.node)
            )
        })
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

    /// Why no map change can be asked for at this tick, if none can.
    fn proposal_refusal(&self) -> Option<String> {
        if let Phase::Preparing { epoch, .. } = self.control.publisher.phase() {
            return Some(format!(
                "epoch {epoch} is still in flight; one change at a time"
            ));
        }
        if let Some(waiting) = self.wanted_refusal() {
            return Some(waiting);
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
        let mut scene_changed = None;
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
            OperatorCommand::SetScene { scene } => {
                self.set_scene(*scene).map(|(message, changed)| {
                    scene_changed = changed;
                    message
                })
            }
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
            scene_changed,
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
        let which = octant.map_or(Octant::FromCentre, |o| Octant::Exactly(usize::from(o)));
        self.begin_split(shard, which)
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
        self.control
            .publisher
            .set_endpoints(&self.endpoints, &self.client_endpoints);
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
            player_port: self
                .player_addr
                .or(config.player_bind)
                .map(|addr| PlayerPortView {
                    addr,
                    sends_to: self.player_target(),
                    redirected: self.player_counts.0,
                    turned_away: self.player_counts.1,
                    refused: self.player_counts.2,
                }),
            spawn_owner: self.spawn_owner(),
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
            min_shards: config.min_shards,
            max_shards: config.max_shards,
            hello_auth_required: self.control.secret_required(),
            starting: self.starting(),
            wanted: self.wanted.as_ref().map(|wanted| wanted.node),
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
        // The region around this one: its direct parent after an octant
        // split, any number of levels up for a carved cell. A lookup.
        let parent = topology::enclosing_region(map, region.node).map(|around| around.node);
        let octant = parent
            .filter(|parent| space_world::octree::parent(region.node) == Some(*parent))
            .and_then(|_| octant_of(region.node));
        RegionView {
            node: region.node,
            shard,
            endpoint: (!placeholder).then_some(region.endpoint),
            client_endpoint: (!placeholder).then_some(region.client_endpoint),
            nesting,
            parent,
            octant,
            free_octants,
            split_refusal,
            merge_refusal,
        }
    }

    /// Every shard the orchestrator knows of: the map's owners, the shard the
    /// change in flight gives a region, and anyone who said hello or holds a
    /// control link.
    fn shard_views(&self, map: &ShardMap) -> Vec<ShardView> {
        let joining = self.pending_join.map(|join| join.shard);
        let mut ids: BTreeSet<ShardId> = map.regions.iter().map(|r| r.shard).collect();
        ids.extend(self.endpoints.keys().copied());
        ids.extend(self.connected.iter().copied());
        ids.extend(joining);
        ids.into_iter()
            .map(|shard| {
                let regions: Vec<_> = map
                    .regions
                    .iter()
                    .filter(|r| r.shard == shard)
                    .map(|r| r.node)
                    .collect();
                let role = if joining == Some(shard) {
                    ShardRole::Joining
                } else if self.dead.contains_key(&shard) {
                    ShardRole::Dead
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
                    client_endpoint: self.client_endpoints.get(&shard).copied(),
                    regions,
                    last_heard: self.last_heard.get(&shard).copied(),
                    report,
                    scale: self.control.autoscaler.state_of(shard),
                    scene: self.scene_confirmed.get(&shard).copied(),
                    launched: self.launch_of(shard).is_some(),
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

    /// Bind the player port on the configured address
    /// ([`OrchestratorConfig::player_bind`]); `None` when none is configured.
    /// Split from the run loop for the same reason as [`Self::bind`]: a
    /// caller that binds port 0 reads the real address before the loop
    /// starts ([`Self::run_with_player`]).
    ///
    /// # Errors
    ///
    /// [`std::io::Error`] if the player port cannot be bound.
    pub fn bind_player(&self) -> Result<Option<Arc<PlayerServer>>, std::io::Error> {
        self.control
            .config
            .player_bind
            .map(|bind| PlayerServer::bind(bind).map(Arc::new))
            .transpose()
    }

    /// A liveness probe over the launcher (H15's launcher half): report whether
    /// the process behind `shard` is still running. A shard the orchestrator
    /// did not launch has no process it could ask about: `Dead`.
    ///
    /// # Errors
    ///
    /// A launcher-specific failure.
    pub async fn poll(&self, shard: ShardId) -> Result<Poll, crate::launcher::LaunchError> {
        match self.launch_of(shard) {
            Some(launch) => self.launcher.poll(launch).await,
            None => Ok(Poll::Dead),
        }
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
    /// [`std::io::Error`] if the control endpoint — or the player port, when
    /// one is configured — cannot be bound.
    pub async fn run(&mut self) -> Result<(), std::io::Error> {
        let server = self.bind()?;
        let player = self.bind_player()?;
        self.run_with_player(server, player).await
    }

    /// Run the event loop over an already-bound `server` until it closes,
    /// with no player port ([`Self::run_with_player`] serves one).
    ///
    /// # Errors
    ///
    /// Currently infallible; the `Result` keeps [`Self::run`]'s signature.
    pub async fn run_with(
        &mut self,
        server: Arc<crate::control::ControlServer>,
    ) -> Result<(), std::io::Error> {
        self.run_with_player(server, None).await
    }

    /// Run the event loop over an already-bound control `server` — and, if
    /// given, an already-bound `player` port — until the control endpoint
    /// closes.
    ///
    /// Nothing in the loop waits on a peer: messages are *offered* to a shard's
    /// queue ([`crate::control::ControlServer::send_to`]), and a launch runs in
    /// its own task. The second matters for more than latency — a launched
    /// shard registers by sending a `ShardHello` that only **this loop** can
    /// answer, so awaiting the launch here would wait on ourselves.
    ///
    /// The player port is served by its own tasks ([`PlayerServer`]). All the
    /// loop does for it is publish what it is to answer
    /// ([`Self::player_answer`]) whenever something was decided — after every
    /// inbound message and on every beat. Nothing comes back from it.
    ///
    /// # Errors
    ///
    /// Currently infallible; the `Result` keeps [`Self::run`]'s signature.
    pub async fn run_with_player(
        &mut self,
        server: Arc<crate::control::ControlServer>,
        player: Option<Arc<PlayerServer>>,
    ) -> Result<(), std::io::Error> {
        use crate::control::Inbound;
        use std::time::Duration;

        let accept = Arc::clone(&server).spawn_accept();
        let player_accept = player.as_ref().map(|player| {
            self.player_addr = player.local_addr().ok();
            player.publish(self.player_answer());
            Arc::clone(player).spawn_accept()
        });
        if player.is_some() {
            // The operator view shows where the port really listens (a bind
            // on port 0 has only now become an address).
            self.publish(true);
        }
        let mut clock = tokio::time::interval(Duration::from_secs_f64(
            1.0 / f64::from(crate::config::TICK_HZ as u32),
        ));
        clock.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                inbound = server.next_inbound() => {
                    match inbound {
                        Some(Inbound::Hello { hello, reply, conn, verdict }) => {
                            self.answer_hello(&server, &hello, reply, conn, verdict).await;
                        }
                        // What a connection says after its hello is taken as
                        // said by the shard it was admitted as — never by
                        // whoever the message names.
                        Some(Inbound::Report { from, report }) => {
                            self.handle_load_report_from(from, &report);
                        }
                        Some(Inbound::Ack { from, ack }) => {
                            self.handle_epoch_ack_from(from, &ack);
                        }
                        Some(Inbound::SceneAck { from, ack }) => {
                            self.handle_scene_ack_from(from, &ack);
                        }
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
            // What players are told follows what was just decided: a root
            // that was claimed, a shard found dead, a replacement that
            // committed.
            if let Some(player) = &player {
                player.publish(self.player_answer());
                self.player_counts = (player.redirected(), player.turned_away(), player.refused());
            }
        }
        accept.abort();
        server.close();
        if let Some(accept) = player_accept {
            accept.abort();
        }
        if let Some(player) = &player {
            player.close();
        }
        Ok(())
    }

    /// Answer one `ShardHello` on the connection it arrived on, and tell that
    /// connection's task the verdict.
    ///
    /// `reply` is that connection's own outbound queue, created empty a moment
    /// ago, so offering the answer to it never waits and never fails for lack
    /// of room. A welcomed shard's queue and connection are registered under
    /// its id **before** this returns: the prepare of the change that gives it
    /// its region is already in the outbox, and the next beat delivers it —
    /// behind the welcome, on the same queue.
    ///
    /// The verdict is what opens the connection or ends it
    /// ([`crate::control::Inbound`]): welcomed, it is admitted as the id of
    /// the welcome and may from then on report and acknowledge **as that
    /// shard**; refused, its task delivers the reason and closes it — nothing
    /// else it sent is ever read.
    async fn answer_hello(
        &mut self,
        server: &crate::control::ControlServer,
        hello: &ShardHello,
        reply: tokio::sync::mpsc::Sender<Message>,
        conn: space_net::Conn,
        verdict: tokio::sync::oneshot::Sender<crate::control::Admission>,
    ) {
        let seen_from = conn.remote_address().ip();
        match self.handle_shard_hello_from(hello, seen_from) {
            Ok(welcome) => {
                let shard = welcome.shard;
                let _ = reply.try_send(Message::ShardWelcome(welcome));
                // The cluster's scene, right behind the welcome: a shard that
                // registers is never left to guess, and never has to wait for
                // the next change to learn the current one.
                let _ = reply.try_send(self.scene_message());
                server.register_link(shard, reply, conn).await;
                // (If the connection's task is gone, so is the connection:
                // the next beat finds the link closed.)
                let _ = verdict.send(Ok(shard));
            }
            Err(refusal) => {
                // The connection's task says why, with the existing `Reject`
                // message — "not yet" as `Draining` (the shard asks again, on
                // a new connection), everything else as what it is (the shard
                // stops, with the reason) — and closes the connection. A
                // final refusal is in the event log, which is what reaches
                // standard output — at a bounded rate, because anybody can
                // cause it.
                let _ = verdict.send(Err(refusal.reject_reason()));
            }
        }
    }

    /// One tick beat: the synchronous [`Self::tick`], then its consequences in
    /// order — epoch messages out, launches started, stops carried out, and
    /// what the launcher and the front end know about who is alive taken in.
    async fn beat(&mut self, server: &crate::control::ControlServer) {
        self.tick();
        // The scene's upkeep: get above a revision a shard reported, and (once
        // a second) say the scene again to whoever has not confirmed it.
        self.reconcile_scene();
        // Offer the epoch messages the tick (and the inbound handlers since
        // the last beat) produced to the shards' queues. Before the stops
        // below, so that the outcome of its epoch is queued for a shard that
        // is about to be stopped — queued, not delivered: its link is closed
        // a few lines down, and a close does not wait for the queue to be
        // written. (The shard stops when its link closes, whichever of the
        // two it sees.)
        for (to, message) in self.drain_outbox() {
            server.send_to(to, message).await;
        }
        // Start the shards that are missing, each in its own task (see
        // `run_with`). A launched shard registers with its own ShardHello and
        // is given its region then; a launch that fails is reported back and
        // the next one waits a little.
        for mut spec in self.take_pending_spawn() {
            if spec.control_addr.port() == 0 {
                // The control endpoint was bound on port 0 ("any"): the shard
                // has to dial the port that was actually bound.
                if let Ok(bound) = server.local_addr() {
                    spec.control_addr.set_port(bound.port());
                }
            }
            let launcher = Arc::clone(&self.launcher);
            let results = Arc::clone(&self.launch_results);
            tokio::spawn(async move {
                let launch = spec.launch;
                let result = launcher.spawn(spec).await.map_err(|err| err.to_string());
                // For the orchestrator; the loop takes it in on its next beat.
                if let Ok(mut results) = results.lock() {
                    results.push((launch, result));
                }
            });
        }
        // Stop the shards a committed merge, an aborted change or a death
        // staged: close their links (a shard does not run without one), and
        // have the launcher stop the processes it started.
        for shard in self.take_pending_kill() {
            server.close_link(shard).await;
            // Stopped on purpose: not a link that was "lost".
            self.connected.remove(&shard);
            self.scene_confirmed.remove(&shard);
        }
        for launch in self.take_pending_launch_kill() {
            if let Err(err) = self.launcher.kill(launch).await {
                eprintln!("space-orchestrator: stopping the shard of {launch} failed: {err}");
                self.log(
                    EventKind::Launcher,
                    Severity::Warn,
                    format!("stopping the shard of {launch} failed: {err}"),
                );
            }
        }
        // What the launches did, and whether what was launched is still alive.
        let reported = self
            .launch_results
            .lock()
            .map(|mut results| std::mem::take(&mut *results))
            .unwrap_or_default();
        for (launch, result) in reported {
            self.note_launch_result(launch, result);
        }
        if self.tick.is_multiple_of(POLL_INTERVAL_TICKS) {
            for launch in self.launches_to_poll() {
                if let Ok(Poll::Dead) = self.launcher.poll(launch).await {
                    self.note_launch_dead(launch);
                }
            }
        }
        // Whose control link is open: a registered shard whose link closed is
        // dead.
        self.note_connected(server.connected().await);
        // What follows is for the operator view only.
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

/// Why a change is not made while `shard` holds no control link.
fn not_connected(shard: ShardId) -> String {
    format!("{shard} is not connected to the orchestrator, so it could not acknowledge the change")
}

/// `" (players dial 0.0.0.0:7870: the host they dialed)"` for a hello that
/// asks for a client endpoint of its own, or nothing.
fn players_at(hello: &ShardHello) -> String {
    match hello.client_endpoint {
        Some(at) if at != hello.endpoint && at.ip().is_unspecified() => {
            format!(" (players dial {at}: the host they dialed)")
        }
        Some(at) if at != hello.endpoint => format!(" (players dial {at})"),
        _ => String::new(),
    }
}

/// `" on port 7870"`, or nothing for a launch without a host port.
fn on_port(port: u16) -> String {
    if port == 0 {
        String::new()
    } else {
        format!(" on port {port}")
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
        spawned: std::sync::Mutex<Vec<LaunchId>>,
    }

    impl Launcher for MockLauncher {
        // `async fn` implementing an AFIT method: rustc checks the generated
        // future against the trait's `+ Send` bound.
        async fn spawn(&self, spec: SpawnSpec) -> Result<ShardHandle, LaunchError> {
            // The lock is taken and released with no await point in between, so
            // the future stays `Send`.
            self.spawned.lock().unwrap().push(spec.launch);
            Ok(ShardHandle {
                launch: spec.launch,
                shard: None,
                endpoint: launched_at(spec.launch),
                process: "mock".into(),
            })
        }

        async fn kill(&self, _launch: LaunchId) -> Result<(), LaunchError> {
            Ok(())
        }

        async fn poll(&self, _launch: LaunchId) -> Result<Poll, LaunchError> {
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

    /// The root owned by shard 1, which has never said hello and holds no
    /// control link (see [`live`]).
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

    /// `orch` with every owner of its map holding a control link: a cluster
    /// in which a map change can be made (every shard of the map has to
    /// acknowledge one, so none is asked for while one of them is not there).
    fn live(mut orch: Orchestrator<MockLauncher>) -> Orchestrator<MockLauncher> {
        let owners: Vec<ShardId> = orch.map().regions.iter().map(|r| r.shard).collect();
        orch.note_connected(owners);
        orch
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

    /// The hello of a shard that just started, listening on `endpoint`: no
    /// id, no region — what every shard says, launched or started by hand.
    fn fresh_hello(endpoint: SocketAddr) -> ShardHello {
        ShardHello {
            cluster: OrchestratorConfig::default().cluster,
            shard: ShardId::NONE,
            endpoint,
            region: None,
            client_endpoint: None,
            secret: String::new(),
        }
    }

    /// Where the shard of `launch` listens in these tests.
    fn launched_at(launch: LaunchId) -> SocketAddr {
        addr(9_500 + u16::try_from(launch.0).expect("a small launch number"))
    }

    /// Play the launcher for `spec`: its shard says hello and the launcher
    /// reports the handle. Returns the hello's outcome.
    fn hello_of(
        orch: &mut Orchestrator<MockLauncher>,
        spec: &SpawnSpec,
    ) -> Result<ShardWelcome, HelloRefusal> {
        let endpoint = launched_at(spec.launch);
        let outcome = orch.handle_shard_hello(&fresh_hello(endpoint));
        orch.note_launch_result(
            spec.launch,
            Ok(ShardHandle {
                launch: spec.launch,
                shard: None,
                endpoint,
                process: "mock".into(),
            }),
        );
        outcome
    }

    /// Play the **`process`** launcher for `spec`: its shard listens on the
    /// host port the launch was handed, says hello from there, and the
    /// launcher reports where it listens (it does not know the shard's id).
    fn process_hello_of(
        orch: &mut Orchestrator<MockLauncher>,
        spec: &SpawnSpec,
    ) -> Result<ShardWelcome, HelloRefusal> {
        let endpoint = addr(spec.host_port);
        let outcome = orch.handle_shard_hello(&fresh_hello(endpoint));
        orch.note_launch_result(spec.launch, Ok(process_handle(spec)));
        outcome
    }

    /// What the `process` launcher reports for `spec`.
    fn process_handle(spec: &SpawnSpec) -> ShardHandle {
        ShardHandle {
            launch: spec.launch,
            shard: None,
            endpoint: addr(spec.host_port),
            process: "pid of a mock".into(),
        }
    }

    /// An orchestrator as `boot` leaves it that starts its shards as
    /// processes on the host ports `ports` and keeps `min_shards` running.
    fn process_cluster(min_shards: usize, ports: (u16, u16)) -> Orchestrator<MockLauncher> {
        fresh_cluster(OrchestratorConfig {
            min_shards,
            launcher: LauncherKind::Process,
            host_port_range: ports,
            ..OrchestratorConfig::default()
        })
    }

    /// Bring `process_cluster`'s shards up, one after the other: each launch
    /// registers and its epoch (none for the root) commits. Returns the
    /// `(launch, shard)` pairs in the order they came up.
    fn bring_up(orch: &mut Orchestrator<MockLauncher>, count: usize) -> Vec<(SpawnSpec, ShardId)> {
        let mut up = Vec::new();
        while up.len() < count {
            orch.tick();
            for spec in orch.take_pending_spawn() {
                let welcome = process_hello_of(orch, &spec).expect("a launched shard is welcomed");
                ack_the_prepare(orch);
                up.push((spec, welcome.shard));
            }
            assert!(orch.current_tick() < 1_000, "the shards never came up");
        }
        let _ = (orch.drain_outbox(), orch.take_pending_kill());
        up
    }

    /// Play the launcher for the launch the orchestrator staged (exactly one):
    /// its shard registers and is welcomed.
    fn join(orch: &mut Orchestrator<MockLauncher>) -> ShardWelcome {
        let mut staged = orch.take_pending_spawn();
        assert_eq!(staged.len(), 1, "exactly one launch is staged: {staged:?}");
        let spec = staged.pop().expect("one");
        hello_of(orch, &spec).expect("the launched shard is welcomed")
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
        let mut orch = live(orchestrator_with_first_id(100));
        let root = orch.config().root_node();
        let kids = space_world::octree::children(root).expect("root splits");
        let mut carved = Vec::new();
        for round in 0..3u64 {
            let reply = orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1)));
            assert!(accepted(&reply), "split {round} is accepted: {reply:?}");
            // A shard is launched for it, registers, and is given the octant.
            let welcome = join(&mut orch);
            assert!(ack_the_prepare(&mut orch) > 0, "a prepare went out");
            assert_eq!(orch.map().epoch, 2 + round, "split {round} committed");
            carved.push(welcome.region.node);
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
        let mut orch = live(orchestrator_with_first_id(100));
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
        let spec = staged.expect("a shard hot for the dwell is split: one is launched");
        let kids = space_world::octree::children(root).expect("root splits");
        assert_eq!(
            orch.wanted_node(),
            Some(kids[7]),
            "the octant asked for is the one holding the region centre"
        );
        let welcome = hello_of(&mut orch, &spec).expect("the launched shard is welcomed");
        assert_eq!(welcome.region.node, kids[7]);
        assert!(space_world::octree::contains(
            welcome.region.node,
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
        let mut orch = live(orchestrator_over(
            config,
            vec![ShardRegion::new(root, ShardId::new(1), addr(7_001))],
            ids,
        ));
        let why = refusal(orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1))));
        assert!(why.contains("split floor"), "{why}");
        assert_eq!(orch.control.ids.peek(), 100, "no id was allocated");
        assert!(orch.take_pending_spawn().is_empty(), "nothing is launched");
        assert_eq!(orch.wanted_node(), None, "nothing is wanted");
        assert!(orch.drain_outbox().is_empty(), "no prepare was published");

        // The autoscaler path refuses the same way: hot for several dwells, and
        // still not a single id gone.
        for _ in 0..(4 * crate::autoscale::SCALE_COOLDOWN_TICKS) {
            orch.handle_load_report(&report(1, 95));
            orch.tick();
        }
        assert_eq!(orch.control.ids.peek(), 100, "a refused split burns no id");
        assert_eq!(orch.map().epoch, 1);
        assert!(orch.take_pending_spawn().is_empty());

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

    /// Mutation this catches: the `expect` on the counter write. With a full
    /// disk no id can be given to the shard that registers for a split; that
    /// took the whole orchestrator down, and must be a split that is not
    /// carried out. (The id is allocated when the shard registers, so that is
    /// where it shows: the split itself is asked for without one.)
    #[test]
    fn a_split_with_an_unwritable_counter_is_not_carried_out_not_a_panic() {
        let config = OrchestratorConfig::default();
        let root = config.root_node();
        let ids = PersistentShardIdAllocator::with_sink(100, Box::new(BrokenSink));
        let mut orch = live(orchestrator_over(
            config,
            vec![ShardRegion::new(root, ShardId::new(1), addr(7_001))],
            ids,
        ));
        let before = orch.map().clone();
        assert!(accepted(
            &orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1)))
        ));
        let spec = orch
            .take_pending_spawn()
            .pop()
            .expect("a shard is launched");
        let refused = hello_of(&mut orch, &spec).expect_err("there is no id to give it");
        assert!(
            matches!(&refused, HelloRefusal::Wait(why) if why.contains("could not be persisted")),
            "{refused:?}"
        );
        assert!(orch.drain_outbox().is_empty(), "no prepare was published");
        assert_eq!(orch.map(), &before);
        assert_eq!(orch.wanted_node(), None, "the split is given up");
        assert!(orch.snapshot().in_flight.is_none());
        let texts = event_texts(&orch);
        assert!(
            texts
                .iter()
                .any(|t| t.starts_with("not carried out: split of shard 1")
                    && t.contains("could not be persisted")),
            "{texts:?}"
        );
        // The same through the autoscaler: it logs and carries on ticking.
        for _ in 0..=(crate::autoscale::SPLIT_DWELL_TICKS + 1) {
            orch.handle_load_report(&report(1, 90));
            orch.tick();
            for spec in orch.take_pending_spawn() {
                let _ = hello_of(&mut orch, &spec);
            }
        }
        assert!(orch.drain_outbox().is_empty());
        assert_eq!(orch.map(), &before);
    }

    // ------------------------------------------------------------- finding 6

    /// Mutation this catches: keeping a report from a shard the published map
    /// does not name. It was replayed to the autoscaler every tick and acted on.
    #[test]
    fn a_report_from_a_shard_outside_the_map_is_ignored() {
        let mut orch = live(orchestrator_with_first_id(100));
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
        assert!(orch.take_pending_spawn().is_empty(), "nothing was launched");
        assert_eq!(orch.control.ids.peek(), 100, "and no id was spent on it");
    }

    /// Mutation this catches: replaying a shard's last report for ever. A shard
    /// that stopped reporting two thirds of the way through its dwell was split
    /// anyway, a hundred ticks after it went silent.
    #[test]
    fn a_stale_report_expires_and_never_completes_a_dwell() {
        let mut orch = live(orchestrator_with_first_id(100));
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
        assert_eq!(orch.wanted_node(), None);
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
        let mut orch = live(orchestrator_with_a_child());
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
        let mut orch = live(orchestrator_with_a_child());
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
        let grandchild = join(&mut orch).shard;
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
        let mut orch = live(orchestrator_with_a_child());
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

    /// A split whose epoch aborts leaves a registered shard that owns nothing:
    /// it is staged for stopping (it used to run on, unowned, for ever) — its
    /// control link, and its process — and the abort reaches every recipient
    /// of the prepare.
    #[test]
    fn a_split_that_aborts_stages_the_kill_of_its_child() {
        let mut orch = live(orchestrator_with_first_id(100));
        assert!(accepted(
            &orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1)))
        ));
        let child = join(&mut orch).shard;
        let launch = orch.launch_of(child).expect("the launch that became it");
        let told: Vec<ShardId> = prepares(&orch.drain_outbox())
            .into_iter()
            .map(|(to, _)| to)
            .collect();
        assert_eq!(told, vec![ShardId::new(1), child]);
        assert!(orch.take_pending_kill().is_empty());
        for _ in 0..=crate::config::EPOCH_ACK_TIMEOUT_TICKS {
            orch.tick();
        }
        assert_eq!(orch.map().epoch, 1, "the split aborted");
        assert_eq!(orch.take_pending_kill(), vec![child]);
        assert_eq!(orch.take_pending_launch_kill(), vec![launch]);
        assert_eq!(
            orch.launch_of(child),
            None,
            "and it is no longer supervised"
        );
        let aborted: Vec<ShardId> = orch
            .drain_outbox()
            .into_iter()
            .filter(|(_, m)| matches!(m, Message::EpochAbort(_)))
            .map(|(to, _)| to)
            .collect();
        assert_eq!(aborted, told, "everyone who got the prepare gets the abort");
        // A committed split, by contrast, kills nobody. (The shard of the
        // aborted one was a launch that came to nothing: the next launch
        // waits the pause a failed launch earns.)
        assert!(accepted(
            &orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1)))
        ));
        assert!(
            orch.take_pending_spawn().is_empty(),
            "not before the pause is over"
        );
        for _ in 0..LAUNCH_BACKOFF_TICKS {
            orch.tick();
        }
        let _ = join(&mut orch);
        assert!(ack_the_prepare(&mut orch) > 0);
        assert_eq!(orch.map().epoch, 2);
        assert!(orch.take_pending_kill().is_empty());
        assert!(orch.take_pending_launch_kill().is_empty());
    }

    /// A split asks for a node and launches a shard; nothing is proposed until
    /// a shard has registered to take it — and if none ever does, the request
    /// is given up and the launch is stopped. No id is spent, no prepare is
    /// sent, the map never moves.
    ///
    /// Mutation this catches: a request that waits for ever (no other change
    /// could be asked for again), or one that proposes its epoch at once — the
    /// order this replaced, in which the 3 s ack deadline had to cover the
    /// start of a process.
    #[test]
    fn a_split_whose_shard_never_registers_is_given_up() {
        let mut orch = live(orchestrator_with_first_id(100));
        let before = orch.map().clone();
        assert!(accepted(
            &orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1)))
        ));
        let staged = orch.take_pending_spawn();
        assert_eq!(staged.len(), 1, "one shard is launched for it");
        assert!(orch.wanted_node().is_some());
        assert_eq!(orch.starting(), 1);
        assert!(orch.snapshot().in_flight.is_none(), "nothing is proposed");
        assert!(orch.drain_outbox().is_empty(), "no prepare without a shard");
        assert_eq!(orch.control.ids.peek(), 100, "and no id either");

        // While it waits no other change is asked for.
        let why = refusal(orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1))));
        assert!(why.contains("still waiting for its new shard"), "{why}");

        // The launch never registers: stopped after its patience...
        for _ in 0..=LAUNCH_REGISTER_TIMEOUT_TICKS {
            orch.tick();
        }
        assert_eq!(orch.take_pending_launch_kill(), vec![staged[0].launch]);
        assert_eq!(orch.starting(), 0);
        // ...and another one is tried while the node is still wanted, after
        // the pause a failed launch earns.
        for _ in 0..=LAUNCH_BACKOFF_TICKS {
            orch.tick();
        }
        let second = orch.take_pending_spawn();
        assert_eq!(second.len(), 1, "a second launch");
        // Then the request itself is given up — and with it, in the same
        // beat, the launch that was started for it: left running, it would
        // register a little later, find nothing wanted and be given a layout
        // cell nobody asked for.
        let asked_at = orch.current_tick();
        while orch.wanted_node().is_some() {
            orch.tick();
            assert!(
                orch.current_tick() - asked_at <= WANTED_TIMEOUT_TICKS,
                "the request is given up"
            );
        }
        assert_eq!(
            orch.take_pending_launch_kill(),
            vec![second[0].launch],
            "stopped with the request it was started for"
        );
        assert_eq!(orch.starting(), 0);
        assert!(
            event_texts(&orch)
                .iter()
                .any(|t| t.contains("is no longer needed") && t.contains("stopped")),
            "{:?}",
            event_texts(&orch)
        );
        assert_eq!(orch.wanted_node(), None);
        let texts = event_texts(&orch);
        assert!(
            texts
                .iter()
                .any(|t| t.starts_with("not carried out: split of shard 1")
                    && t.contains("no shard registered")),
            "{texts:?}"
        );
        assert_eq!(orch.map(), &before);
        assert!(orch.drain_outbox().is_empty(), "nothing was ever proposed");
        assert_eq!(orch.control.ids.peek(), 100);
        // And a new request is accepted again.
        let _ = orch.take_pending_spawn();
        let _ = orch.take_pending_launch_kill();
        for _ in 0..=(2 * LAUNCH_REGISTER_TIMEOUT_TICKS) {
            orch.tick();
        }
        assert!(accepted(
            &orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1)))
        ));
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
        orch.handle_shard_hello(&ShardHello {
            cluster: OrchestratorConfig::default().cluster,
            shard: ShardId::new(1),
            endpoint: addr(9_001),
            region: None,
            client_endpoint: None,
            secret: String::new(),
        })
        .expect("root shard registers");
        orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1)));
        let _ = orch.take_pending_spawn();
        // The shard for the split registers from its own address, with no id.
        let child = orch
            .handle_shard_hello(&fresh_hello(addr(9_002)))
            .expect("the new shard registers and is given the octant")
            .shard;
        let outbox = orch.drain_outbox();
        let epoch = outbox
            .iter()
            .find_map(|(_, m)| match m {
                Message::EpochPrepare(p) => Some(p.epoch),
                _ => None,
            })
            .expect("the prepare went out");
        let recipients: Vec<ShardId> = outbox.into_iter().map(|(to, _)| to).collect();
        assert_eq!(recipients, vec![ShardId::new(1), child]);
        for shard in recipients {
            orch.handle_epoch_ack(shard, epoch);
        }

        let out = orch.drain_outbox();
        for shard in [ShardId::new(1), child] {
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
            assert_eq!(endpoint_of(child), Some(addr(9_002)));
        }
        // The orchestrator's own published map carries the same addresses.
        assert!(orch
            .map()
            .regions
            .iter()
            .any(|r| r.shard == child && r.endpoint == addr(9_002)));
        assert_eq!(orch.endpoint_of(child), Some(addr(9_002)));
    }

    #[test]
    fn the_mock_launcher_satisfies_the_trait() {
        // Proves the Launcher trait is implementable as a plain struct (the
        // acceptance tests rely on this).
        let launcher = MockLauncher::default();
        let spec = SpawnSpec {
            launch: LaunchId(1),
            cluster: "space-local".into(),
            control_addr: addr(7_900),
            host_port: 7_870,
            secret: String::new(),
        };
        let handle = block_on(launcher.spawn(spec)).expect("spawn");
        assert_eq!(handle.launch, LaunchId(1));
        assert_eq!(block_on(launcher.poll(LaunchId(1))).unwrap(), Poll::Alive);
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
        live(orchestrator_with_first_id(100))
    }

    fn event_texts(orch: &Orchestrator<MockLauncher>) -> Vec<String> {
        orch.events()
            .newest_first()
            .map(|event| event.text.clone())
            .collect()
    }

    /// An operator's split is an ordinary epoch change: asked for, proposed
    /// when its shard has registered, acked by every recipient, committed —
    /// and it carves exactly the octant named.
    ///
    /// Mutation this catches: an operator path that writes the map itself
    /// (the map would change before the acks), or that treats the octant as a
    /// starting point like the admin verb does; and proposing (or allocating
    /// the id) before a shard is there to take the octant.
    #[test]
    fn an_operator_split_carves_the_octant_asked_for_through_the_epoch_protocol() {
        let mut orch = connected_orchestrator();
        let kids = space_world::octree::children(orch.config().root_node()).expect("splits");

        let reply = orch.handle_operator(&split(1, Some(5)));
        assert!(reply.accepted, "{reply:?}");
        assert!(
            reply.message.contains(&format!(
                "region {} is wanted for a new shard",
                node_path(kids[5])
            )),
            "{reply:?}"
        );
        // Asked for is not proposed, let alone published: the octant waits for
        // a shard, no id is allocated, nobody is told anything.
        assert_eq!(orch.map().epoch, 1);
        assert_eq!(orch.map().regions.len(), 1);
        assert_eq!(
            orch.wanted_node(),
            Some(kids[5]),
            "exactly the octant asked for"
        );
        let snapshot = orch.snapshot();
        assert!(snapshot.in_flight.is_none());
        assert_eq!(snapshot.wanted, Some(kids[5]));
        assert_eq!(snapshot.starting, 1, "a shard is being started for it");
        assert_eq!(orch.control.ids.peek(), 100);
        assert!(orch.drain_outbox().is_empty());

        // The launched shard registers — without an id, without a region — and
        // is welcomed with a freshly allocated id (H9) for exactly that octant.
        // The map it is handed is the published one: not its octant yet (H10).
        let welcome = join(&mut orch);
        assert_eq!(welcome.shard, ShardId::new(100));
        assert_eq!(welcome.region.node, kids[5]);
        assert_eq!(&welcome.shard_map, orch.map());
        assert_eq!(welcome.shard_map.epoch, 1);
        assert_eq!(orch.wanted_node(), None, "served");

        let snapshot = orch.snapshot();
        let flight = snapshot.in_flight.expect("the change is in flight");
        assert_eq!(flight.epoch, 2);
        assert_eq!(flight.changes, vec![(kids[5], ShardId::new(100))]);
        assert_eq!(
            flight.recipients,
            vec![(ShardId::new(1), false), (ShardId::new(100), false)],
            "everyone who must ack, nobody has yet"
        );
        assert!(snapshot.shards.iter().any(|s| s.shard == ShardId::new(100)
            && s.role == ShardRole::Joining
            && s.connected
            && s.launched));
        assert_eq!(
            snapshot.regions.len(),
            1,
            "the snapshot shows the committed map"
        );
        assert_eq!(snapshot.starting, 0);

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
        assert_eq!(orch.wanted_node(), Some(kids[7]));
        assert_eq!(join(&mut orch).region.node, kids[7]);
    }

    /// Mutation this catches: letting an operator's mistake go through
    /// `begin_split`'s refusal path, which cools the shard down and makes the
    /// autoscaler "busy" — or worse, launching a shard for it.
    #[test]
    fn a_refused_operator_request_changes_nothing_at_all() {
        let mut orch = connected_orchestrator();
        // Carve octant 3 for shard 100 and let the minimum interval pass.
        assert!(orch.handle_operator(&split(1, Some(3))).accepted);
        let _ = join(&mut orch);
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
                orch.wanted_node(),
                orch.starting(),
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
            (stale, "the map changed"),
            (merge(1), "no enclosing region to merge into"),
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

    /// One change at a time (H12): a second request while a region is waiting
    /// for its shard, or while its prepare is outstanding, is refused, not
    /// queued, and burns no id. When the first one aborts, the log says who did
    /// not ack and the next request is accepted.
    #[test]
    fn an_operator_request_while_a_change_is_under_way_is_refused_not_queued() {
        let mut orch = connected_orchestrator();
        assert!(orch.handle_operator(&split(1, Some(0))).accepted);
        // Waiting for its shard: nothing else is asked for.
        let reply = orch.handle_operator(&split(1, Some(1)));
        assert!(!reply.accepted);
        assert!(
            reply.message.contains("still waiting for its new shard"),
            "{reply:?}"
        );
        // The shard registers: the prepare is outstanding.
        let _ = join(&mut orch);
        let ids = orch.control.ids.peek();
        let reply = orch.handle_operator(&split(1, Some(1)));
        assert!(!reply.accepted);
        assert!(
            reply.message.contains("epoch 2 is still in flight"),
            "{reply:?}"
        );
        assert_eq!(orch.control.ids.peek(), ids, "no id was burned");
        assert!(
            orch.take_pending_spawn().is_empty(),
            "only the first split launched a shard"
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
        assert_eq!(orch.wanted_node(), None);
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

    // ------------------------------------------------------------- carving

    /// The cell `d` cells from the root's centre corner on each axis: `[0,0,0]`
    /// is the one the scene's spawn point lies in (`docs/plan-sharding.md` §1).
    fn cell_at_the_centre(orch: &Orchestrator<MockLauncher>, d: [i64; 3]) -> space_world::NodeRef {
        let k = orch.config().root_node().center().cell;
        space_world::NodeRef::new(64, [k[0] + d[0], k[1] + d[1], k[2] + d[2]])
    }

    /// `begin_carve` is a split whose node is named: one cell, twenty-four
    /// levels below the root, goes to a new shard through the same steps —
    /// asked for, a shard launched, its hello, the id, one epoch to **every**
    /// shard of the map and the new one, acks, commit — and the merge that
    /// undoes it gives the cell back to the root's owner.
    ///
    /// Mutations this catches: a carve path that writes the map itself; a
    /// recipient rule that misses the shards of a deep carve; a merge that
    /// needs the direct parent node to be a region; a `validate` that refuses
    /// depth 64.
    #[test]
    fn a_cell_is_carved_for_a_new_shard_through_the_epoch_protocol_and_merged_back() {
        let mut orch = connected_orchestrator();
        let (one, first, second) = (ShardId::new(1), ShardId::new(100), ShardId::new(101));
        let cell = cell_at_the_centre(&orch, [0, 0, -1]);
        let inside = cell.center();
        assert_eq!(orch.map().lookup(&inside), Some(one));

        let wanted = orch
            .begin_carve(cell)
            .expect("the cell lies inside the root");
        assert_eq!(
            wanted,
            CarveWanted {
                node: cell,
                from: one
            }
        );
        assert!(
            wanted.to_string().starts_with(&format!(
                "region {} is wanted for a new shard",
                node_path(cell)
            )),
            "{wanted}"
        );
        // Asked for is not proposed (H10): nothing is in flight, nobody is
        // told, no id is allocated — a shard is launched and the cell waits.
        assert_eq!(orch.map().epoch, 1);
        assert_eq!(orch.map().lookup(&inside), Some(one));
        assert_eq!(orch.wanted_node(), Some(cell));
        assert!(orch.snapshot().in_flight.is_none());
        assert!(orch.drain_outbox().is_empty());
        assert_eq!(orch.control.ids.peek(), 100);
        // The shard losing the cell is on the autoscaler's cooldown.
        assert!(orch.control.autoscaler.busy_until() > orch.current_tick());

        // The launched shard says hello with neither id nor region and is
        // given both; the map it is handed still gives the cell to the root.
        let welcome = join(&mut orch);
        assert_eq!((welcome.shard, welcome.region.node), (first, cell));
        assert_eq!(welcome.shard_map.lookup(&inside), Some(one));
        let snapshot = orch.snapshot();
        let in_flight = snapshot.in_flight.expect("the carve is in flight");
        assert_eq!(in_flight.changes, vec![(cell, first)]);
        assert_eq!(
            in_flight.what,
            format!("carve of region {} for new shard 100", node_path(cell))
        );
        assert_eq!(in_flight.recipients, vec![(one, false), (first, false)]);
        let role = |orch: &Orchestrator<MockLauncher>, shard| {
            orch.snapshot()
                .shards
                .iter()
                .find(|s| s.shard == shard)
                .map(|s| s.role)
        };
        assert_eq!(role(&orch, first), Some(ShardRole::Joining));

        // The prepare went to the root's owner and to the new shard, which is
        // connected already.
        assert_eq!(prepares(&orch.drain_outbox()), vec![(one, 2), (first, 2)]);
        orch.handle_epoch_ack(one, 2);
        assert_eq!(orch.map().epoch, 1, "the new shard has not acked");
        assert_eq!(
            orch.map().lookup(&inside),
            Some(one),
            "not before the commit"
        );
        orch.handle_epoch_ack(first, 2);

        // Committed: the cell is the new shard's, at its real address, and
        // everything around it is still the root owner's.
        assert_eq!(orch.map().epoch, 2);
        crate::topology::validate(orch.map()).expect("a depth-64 region is valid (H11)");
        assert_eq!(orch.map().lookup(&inside), Some(first));
        for d in [[0, 0, 0], [0, 0, -2], [-1, 0, -1], [0, 1, -1], [0, -1, -1]] {
            let neighbour = cell_at_the_centre(&orch, d).center();
            assert_eq!(orch.map().lookup(&neighbour), Some(one), "{d:?}");
        }
        let published = orch
            .map()
            .regions
            .iter()
            .find(|r| r.shard == first)
            .expect("published");
        assert_eq!(
            (published.node, published.endpoint),
            (cell, welcome.region.endpoint)
        );
        let sent = orch.drain_outbox();
        for to in [one, first] {
            assert!(sent
                .iter()
                .any(|(s, m)| *s == to && matches!(m, Message::EpochCommit(c) if c.epoch == 2)));
            assert!(sent.iter().any(|(s, m)| *s == to
                && matches!(m, Message::DirectoryReply(r) if r.shard_map == *orch.map())));
        }
        assert!(orch.take_pending_kill().is_empty(), "a carve kills nobody");
        assert_eq!(role(&orch, first), Some(ShardRole::Owner));
        assert!(event_texts(&orch)[0].starts_with("epoch 2 committed: carve of region 64:"));

        // The operator view: the cell is listed under the root, as carved out
        // of it (it is nobody's octant), too small to split, and mergeable
        // once it is known to be empty.
        let snapshot = orch.snapshot();
        let paths: Vec<String> = snapshot.regions.iter().map(|r| node_path(r.node)).collect();
        assert_eq!(paths, vec!["40:0:0:0".to_string(), node_path(cell)]);
        let view = &snapshot.regions[1];
        assert_eq!(
            (view.nesting, view.parent, view.octant),
            (1, Some(orch.config().root_node()), None)
        );
        assert!(view.free_octants.is_empty());
        assert!(view
            .split_refusal
            .as_deref()
            .is_some_and(|why| why.contains("below the split floor")));
        assert!(view
            .merge_refusal
            .as_deref()
            .is_some_and(|why| why.contains("not known to be empty")));

        // A second cell, for a third shard: now everybody in the map hears
        // about it — the first cell's owner too, though the cells only share
        // an edge and no parent.
        for _ in 0..MIN_EPOCH_INTERVAL_TICKS {
            orch.tick();
        }
        let other = cell_at_the_centre(&orch, [-1, 0, 0]);
        let wanted = orch.begin_carve(other).expect("the second cell");
        assert_eq!((wanted.node, wanted.from), (other, one));
        let welcome = join(&mut orch);
        assert_eq!((welcome.shard, welcome.region.node), (second, other));
        assert_eq!(
            prepares(&orch.drain_outbox()),
            vec![(one, 3), (first, 3), (second, 3)]
        );
        for shard in [one, first, second] {
            assert_eq!(orch.map().epoch, 2);
            orch.handle_epoch_ack(shard, 3);
        }
        assert_eq!(orch.map().epoch, 3);
        assert_eq!(orch.map().regions.len(), 3);
        let _ = orch.drain_outbox();

        // Merge the first cell back. It is refused until its shard is known
        // to be empty; then it is one epoch to everyone, the shard itself
        // included, and the shard is stopped only by the commit.
        for _ in 0..MIN_EPOCH_INTERVAL_TICKS {
            orch.tick();
        }
        let reply = orch.handle_operator(&merge(100));
        assert!(
            !reply.accepted && reply.message.contains("not known to be empty"),
            "{reply:?}"
        );
        orch.handle_load_report(&report(100, 0));
        assert_eq!(orch.merge_refusal(first), None);
        let reply = orch.handle_operator(&merge(100));
        assert!(reply.accepted, "{reply:?}");
        assert!(
            reply.message.contains(&format!(
                "epoch 4 proposed, region {} goes back to shard 1",
                node_path(cell)
            )),
            "{reply:?}"
        );
        assert_eq!(orch.pending_merge(), Some(first));
        assert_eq!(
            orch.map().lookup(&inside),
            Some(first),
            "not before the commit"
        );
        assert_eq!(
            prepares(&orch.drain_outbox()),
            vec![(one, 4), (first, 4), (second, 4)]
        );
        for shard in [one, first, second] {
            assert!(orch.take_pending_kill().is_empty());
            orch.handle_epoch_ack(shard, 4);
        }
        assert_eq!(orch.map().epoch, 4);
        crate::topology::validate(orch.map()).expect("valid");
        assert_eq!(orch.map().regions.len(), 2);
        assert_eq!(
            orch.map().lookup(&inside),
            Some(one),
            "the cell fell back to the root's owner"
        );
        assert_eq!(orch.map().lookup(&other.center()), Some(second));
        assert_eq!(orch.take_pending_kill(), vec![first]);
        // The merged shard is told the outcome before it is stopped.
        let sent = orch.drain_outbox();
        assert!(sent
            .iter()
            .any(|(s, m)| *s == first && matches!(m, Message::EpochCommit(c) if c.epoch == 4)));
        assert!(event_texts(&orch)[0].starts_with("epoch 4 committed: merge of shard 100"));
    }

    /// What `begin_carve` refuses, and that a refusal has no side effect: no
    /// id is allocated, nothing is published, staged or launched, nobody is
    /// put on cooldown.
    #[test]
    fn a_carve_that_cannot_happen_changes_nothing_and_says_why() {
        let config = OrchestratorConfig {
            max_regions: 2,
            ..OrchestratorConfig::default()
        };
        let root = config.root_node();
        let ids =
            PersistentShardIdAllocator::with_sink(100, Box::new(MemoryCounterSink::default()));
        let mut orch = live(orchestrator_over(
            config,
            vec![ShardRegion::new(root, ShardId::new(1), addr(7_001))],
            ids,
        ));
        let state = |orch: &Orchestrator<MockLauncher>| {
            (
                orch.map().clone(),
                orch.control.ids.peek(),
                orch.control.autoscaler.busy_until(),
                orch.snapshot().in_flight,
                orch.wanted_node(),
                orch.starting(),
            )
        };
        let before = state(&orch);
        let cell = cell_at_the_centre(&orch, [0, 0, -1]);
        let k = root.center().cell[0];
        for (node, why) in [
            (root, "already a region"),
            (
                space_world::NodeRef::new(64, [-k, k, k]),
                "inside no region",
            ),
            (space_world::NodeRef::new(39, [0, 0, 0]), "inside no region"),
            (
                space_world::NodeRef::new(65, [k, k, k]),
                "exceeds MAX_DEPTH",
            ),
        ] {
            let err = orch.begin_carve(node).expect_err("refused");
            assert!(err.contains(why), "{node:?}: {err}");
            assert_eq!(
                state(&orch),
                before,
                "{node:?}: a refusal has no side effect"
            );
            assert!(orch.take_pending_spawn().is_empty());
            assert!(orch.drain_outbox().is_empty());
        }

        // One at a time (H12): while the first carve waits for its shard a
        // second is refused, while its epoch is in flight too, and so is one
        // right after its commit.
        orch.begin_carve(cell)
            .expect("the first carve is asked for");
        let waiting = state(&orch);
        let other = cell_at_the_centre(&orch, [-1, 0, 0]);
        let err = orch.begin_carve(other).expect_err("one change at a time");
        assert!(err.contains("still waiting for its new shard"), "{err}");
        assert_eq!(state(&orch), waiting);
        let _ = join(&mut orch);
        let in_flight = state(&orch);
        let err = orch.begin_carve(other).expect_err("one change at a time");
        assert!(err.contains("epoch 2 is still in flight"), "{err}");
        assert_eq!(state(&orch), in_flight);
        assert!(orch.take_pending_spawn().is_empty(), "only the first one's");
        assert!(ack_the_prepare(&mut orch) > 0);
        assert_eq!(orch.map().epoch, 2);
        let err = orch
            .begin_carve(other)
            .expect_err("too soon after a commit");
        assert!(err.contains("committed too recently"), "{err}");
        for _ in 0..MIN_EPOCH_INTERVAL_TICKS {
            orch.tick();
        }
        // The map is full (two regions, the limit of this cluster).
        let err = orch.begin_carve(other).expect_err("the map is full");
        assert!(err.contains("--max-regions"), "{err}");
        assert_eq!(orch.control.ids.peek(), 101, "one id for the one carve");
        assert!(orch.take_pending_spawn().is_empty());
    }

    /// A carve whose prepare is not acked aborts at the deadline like a
    /// split: the map is untouched and the shard that had registered for it
    /// is stopped.
    #[test]
    fn a_carve_nobody_acks_aborts_and_its_shard_is_stopped() {
        let mut orch = connected_orchestrator();
        let cell = cell_at_the_centre(&orch, [-1, 0, -1]);
        let before = orch.map().clone();
        orch.begin_carve(cell).expect("asked for");
        let new = join(&mut orch).shard;
        let _ = orch.drain_outbox();
        orch.handle_epoch_ack(ShardId::new(1), 2);
        for _ in 0..=crate::config::EPOCH_ACK_TIMEOUT_TICKS {
            orch.tick();
        }
        assert_eq!(orch.map(), &before, "byte-identical after the abort (H10)");
        assert_eq!(orch.take_pending_kill(), vec![new]);
        assert!(
            !orch.live_shards().contains(&new),
            "it is no longer counted as a shard of the cluster"
        );
        let texts = event_texts(&orch);
        let aborted = texts
            .iter()
            .find(|t| t.starts_with("epoch 2 aborted"))
            .unwrap_or_else(|| panic!("the abort is logged: {texts:?}"));
        assert!(
            aborted.contains("carve of region 64:") && aborted.contains("no ack from shard 100"),
            "{aborted}"
        );
        // The abort went to both recipients of the prepare.
        let aborts: Vec<ShardId> = orch
            .drain_outbox()
            .into_iter()
            .filter(|(_, m)| matches!(m, Message::EpochAbort(_)))
            .map(|(to, _)| to)
            .collect();
        assert_eq!(aborts, vec![ShardId::new(1), new]);
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
        let _ = join(&mut orch);
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
        let _ = join(&mut orch);
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
            client_endpoint: None,
            secret: String::new(),
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
        // The fleet: nothing is kept running by the orchestrator by default.
        assert_eq!((snapshot.min_shards, snapshot.max_shards), (0, 8));
        assert_eq!((snapshot.starting, snapshot.wanted), (0, None));
        assert!(!snapshot.hello_auth_required, "a loopback control port");

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
        assert!(
            outer
                .split_refusal
                .as_deref()
                .is_some_and(|why| why.contains("shard 2 is not connected")),
            "every shard of the map has to acknowledge a change, and shard 2 is not there: {:?}",
            outer.split_refusal
        );
        assert!(outer
            .merge_refusal
            .as_deref()
            .is_some_and(|why| why.contains("no enclosing region")));
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
        assert!(!one.launched, "it was started by hand");
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

    /// A registered shard whose control link closed is **dead** — a shard
    /// registers once and stops itself without its orchestrator, so a closed
    /// link does not come back. That is said once, the shard is stopped and
    /// waits to be replaced; the map itself only changes when a replacement
    /// has registered and its epoch committed.
    ///
    /// (This used to be "is logged once and changes no ownership", full stop:
    /// the dead owner stayed in the map for good.)
    #[test]
    fn a_lost_control_link_is_logged_once_and_makes_the_shard_dead() {
        let mut orch = orchestrator_with_first_id(100);
        orch.note_connected([ShardId::new(1)]);
        let before = orch.events().len();
        orch.note_connected([ShardId::new(1)]);
        assert_eq!(
            orch.events().len(),
            before,
            "still connected: nothing to say"
        );
        assert!(orch.dead_shards().is_empty());

        orch.note_connected([]);
        let texts = event_texts(&orch);
        assert!(
            texts[0].starts_with("shard 1 lost: its control link closed"),
            "{texts:?}"
        );
        assert_eq!(orch.dead_shards(), vec![ShardId::new(1)]);
        assert_eq!(orch.take_pending_kill(), vec![ShardId::new(1)]);
        orch.note_connected([]);
        assert_eq!(
            orch.events().len(),
            before + 1,
            "it is lost once, not every beat"
        );
        assert!(orch.take_pending_kill().is_empty(), "and stopped once");

        assert_eq!(orch.map().epoch, 1);
        assert_eq!(orch.map().regions.len(), 1, "losing a link moves no region");
        assert_eq!(orch.map().regions[0].shard, ShardId::new(1));
        assert!(orch.drain_outbox().is_empty());
        let view = orch.snapshot().shards[0].clone();
        assert!(!view.connected);
        assert_eq!(view.role, ShardRole::Dead);
        assert!(orch.live_shards().is_empty());
        let reply = orch.handle_operator(&split(1, Some(0)));
        assert!(reply.message.contains("not connected"), "{reply:?}");
        // It was started by hand and the orchestrator runs no shards itself
        // (`--min-shards 0`): nothing is launched for it. The shard that
        // registers next is given its region.
        assert!(orch.take_pending_spawn().is_empty());
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
            client_endpoint: None,
            secret: String::new(),
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
            client_endpoint: None,
            secret: String::new(),
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
        // Neither the peer token (it is "secret" in this fixture) nor anything
        // called a secret is part of a snapshot.
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
        let _ = join(&mut orch);
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
        // An epoch change is proposed in the same beat, after the requests
        // (a split, as soon as its shard has registered).
        assert!(orch.handle_operator(&split(1, Some(5))).accepted);
        let _ = join(&mut orch);
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
                client_endpoint: None,
                secret: String::new(),
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

    // ------------------------------------------------------------ the lifecycle

    /// An orchestrator as `boot` leaves it: the root reserved for shard 1 and
    /// claimed by nobody yet; ids from 2.
    fn fresh_cluster(config: OrchestratorConfig) -> Orchestrator<MockLauncher> {
        let map = ShardMap::new(
            1,
            vec![ShardRegion::new(
                config.root_node(),
                ShardId::new(1),
                config.bind,
            )],
        );
        let ids = PersistentShardIdAllocator::with_sink(2, Box::new(MemoryCounterSink::default()));
        let mut control = ControlPlane::new(config, ids, EpochPublisher::new(map), "secret".into());
        control.set_unclaimed_root_owner(ShardId::new(1));
        Orchestrator::new(control, std::sync::Arc::new(MockLauncher::default()))
    }

    /// The owner of the point `(x, z)` metres from the scene's spawn point.
    fn owner_at(orch: &Orchestrator<MockLauncher>, x: f64, z: f64) -> Option<ShardId> {
        let anchor = space_scene::scene_anchor(orch.map()).expect("a map has an anchor");
        orch.map()
            .lookup(&space_world::WorldPos::from_f64_metres_around(
                anchor,
                [x, 0.0, z],
            ))
    }

    /// `--min-shards 4`: the orchestrator launches shards until four are
    /// live. The first is alone until it has the root (the shard that owns
    /// the spawn point is the first one launched); the other three are
    /// launched together and given the first three layout cells, one epoch
    /// each, one at a time, each proposed with its shard already connected.
    /// The scene then lies in four quadrants with four owners, meeting at
    /// scene x = -6 / z = -4, and nothing more is launched.
    ///
    /// Mutations this catches: launching nothing (the default before); a
    /// launch per tick for as long as one is starting; the old order (plan,
    /// launch with an id, wait for the hello inside the 3 s deadline); giving
    /// away the spawn cell or the cells in another order.
    #[test]
    fn min_shards_4_ends_with_four_regions_owned_by_four_live_shards() {
        let mut orch = fresh_cluster(OrchestratorConfig {
            min_shards: 4,
            ..OrchestratorConfig::default()
        });
        let root = orch.config().root_node();
        let cells = topology::layout_cells(root);
        assert!(
            orch.take_pending_spawn().is_empty(),
            "nothing before a tick"
        );

        // Until the root has an owner: one launch, however long it takes.
        orch.tick();
        let first = orch.take_pending_spawn();
        assert_eq!(first.len(), 1);
        for _ in 0..30 {
            orch.tick();
        }
        assert!(
            orch.take_pending_spawn().is_empty(),
            "one at a time until the root is claimed"
        );
        assert_eq!(orch.starting(), 1);
        let welcome = hello_of(&mut orch, &first[0]).expect("the first shard is welcomed");
        assert_eq!(
            (welcome.shard, welcome.region.node),
            (ShardId::new(1), root),
            "the reserved id, the root"
        );
        assert_eq!(
            welcome.shard_map.lookup(&root.center()),
            Some(ShardId::new(1))
        );
        assert!(
            orch.drain_outbox().is_empty(),
            "the root needs no epoch change"
        );
        assert_eq!(orch.map().epoch, 1);
        assert_eq!(owner_at(&orch, 0.0, 0.0), Some(ShardId::new(1)));

        // Now the other three, together...
        orch.tick();
        let rest = orch.take_pending_spawn();
        assert_eq!(rest.len(), 3);
        assert_eq!(orch.starting(), 3);
        // ...served one at a time: while one's epoch is in flight the others
        // are told to wait, and the next is served as soon as it has committed
        // (no pause between these changes).
        let mut given = Vec::new();
        for (n, spec) in rest.iter().enumerate() {
            let welcome = hello_of(&mut orch, spec).expect("served in turn");
            assert_eq!(
                &welcome.shard_map,
                orch.map(),
                "welcomed with the published map: not the owner yet"
            );
            assert_eq!(
                orch.map().lookup(&welcome.region.node.center()),
                Some(ShardId::new(1))
            );
            for later in &rest[n + 1..] {
                let refused = hello_of(&mut orch, later).expect_err("one change at a time");
                assert!(matches!(refused, HelloRefusal::Wait(_)), "{refused:?}");
            }
            let recipients = prepares(&orch.drain_outbox());
            assert!(recipients.contains(&(welcome.shard, orch.map().epoch + 1)));
            assert_eq!(
                recipients.len(),
                n + 2,
                "every shard of the map and the new one"
            );
            for (shard, epoch) in recipients {
                orch.handle_epoch_ack(shard, epoch);
            }
            given.push((welcome.shard, welcome.region.node));
        }
        assert_eq!(
            given,
            vec![
                (ShardId::new(2), cells[0]),
                (ShardId::new(3), cells[1]),
                (ShardId::new(4), cells[2]),
            ]
        );
        assert_eq!(orch.map().epoch, 4, "one epoch per cell");
        assert_eq!(orch.map().regions.len(), 4);
        crate::topology::validate(orch.map()).expect("valid (H11)");

        // Four quadrants, four owners, meeting at scene x = -6 / z = -4.
        assert_eq!(owner_at(&orch, 0.0, 0.0), Some(ShardId::new(1)));
        assert_eq!(owner_at(&orch, -5.0, -3.0), Some(ShardId::new(1)));
        assert_eq!(owner_at(&orch, -5.0, -5.0), Some(ShardId::new(2)));
        assert_eq!(owner_at(&orch, -7.0, -3.0), Some(ShardId::new(3)));
        assert_eq!(owner_at(&orch, -7.0, -5.0), Some(ShardId::new(4)));
        assert_eq!(owner_at(&orch, 300.0, -300.0), Some(ShardId::new(2)));
        assert_eq!(owner_at(&orch, -300.0, 300.0), Some(ShardId::new(3)));

        // Four live shards, each launched and none still starting; nothing
        // more is launched, however long it runs.
        let snapshot = orch.snapshot();
        assert_eq!(orch.live_shards().len(), 4);
        assert_eq!(orch.starting(), 0);
        assert_eq!(snapshot.shards.len(), 4);
        assert!(snapshot
            .shards
            .iter()
            .all(|s| s.role == ShardRole::Owner && s.connected && s.launched));
        for _ in 0..(2 * LAUNCH_REGISTER_TIMEOUT_TICKS) {
            orch.tick();
        }
        assert!(orch.take_pending_spawn().is_empty());
        assert!(orch.take_pending_launch_kill().is_empty());
        assert!(orch.take_pending_kill().is_empty());
        assert_eq!(orch.map().epoch, 4);
    }

    /// Shards started by hand, three of them saying hello at once: one is
    /// served, the other two are told to wait — not refused for good, not
    /// logged, no id spent — and each is served when it asks again after the
    /// change before it has committed.
    ///
    /// Mutation this catches: refusing a hello for good while a change is in
    /// flight (the shard exited: every shard after the second would have to
    /// be started again by hand), or serving two at once (two prepares
    /// outstanding).
    #[test]
    fn hellos_while_a_change_is_in_flight_are_told_to_wait_and_served_in_turn() {
        let mut orch = fresh_cluster(OrchestratorConfig::default());
        let cells = topology::layout_cells(orch.config().root_node());
        let hello = |port: u16| fresh_hello(addr(port));
        let root = orch.handle_shard_hello(&hello(9_001)).expect("the root");
        assert_eq!(root.shard, ShardId::new(1));

        let a = orch.handle_shard_hello(&hello(9_002)).expect("served");
        assert_eq!((a.shard, a.region.node), (ShardId::new(2), cells[0]));
        let (ids, events) = (orch.control.ids.peek(), orch.events().len());
        for port in [9_003, 9_004, 9_003] {
            let refused = orch
                .handle_shard_hello(&hello(port))
                .expect_err("a change is in flight");
            assert_eq!(
                refused,
                HelloRefusal::Wait("epoch 2 is still in flight; one change at a time".into())
            );
            assert!(!refused.is_final());
            assert_eq!(
                refused.reject_reason(),
                space_protocol::RejectReason::Draining,
                "on the wire: ask again"
            );
        }
        assert_eq!(
            orch.control.ids.peek(),
            ids,
            "a hello that waits burns no id"
        );
        assert_eq!(orch.events().len(), events, "and is not worth a log line");
        assert_eq!(
            prepares(&orch.drain_outbox()).len(),
            2,
            "one prepare, two recipients"
        );
        assert!(orch.snapshot().in_flight.is_some());

        // The change commits; whoever asks next is served, at once.
        orch.handle_epoch_ack(ShardId::new(1), 2);
        orch.handle_epoch_ack(ShardId::new(2), 2);
        assert_eq!(orch.map().epoch, 2);
        let c = orch.handle_shard_hello(&hello(9_004)).expect("served next");
        assert_eq!((c.shard, c.region.node), (ShardId::new(3), cells[1]));
        let refused = orch
            .handle_shard_hello(&hello(9_003))
            .expect_err("its turn is next");
        assert!(
            matches!(&refused, HelloRefusal::Wait(why) if why.contains("epoch 3")),
            "{refused:?}"
        );
        assert!(ack_the_prepare(&mut orch) > 0);
        let b = orch.handle_shard_hello(&hello(9_003)).expect("served last");
        assert_eq!((b.shard, b.region.node), (ShardId::new(4), cells[2]));
        assert!(ack_the_prepare(&mut orch) > 0);

        assert_eq!(orch.map().epoch, 4);
        assert_eq!(orch.map().regions.len(), 4);
        assert_eq!(
            orch.control.ids.peek(),
            5,
            "ids 2, 3 and 4, in the order served"
        );
        assert_eq!(owner_at(&orch, -7.0, -3.0), Some(ShardId::new(3)));
        // Started by hand, and `--min-shards` is 0: nothing was launched.
        assert!(orch.take_pending_spawn().is_empty());
        assert!(orch.snapshot().shards.iter().all(|s| !s.launched));
    }

    /// `--max-shards`: a shard that registers when the cluster has as many as
    /// it may is refused **for good** (`Full` on the wire), the orchestrator
    /// launches none beyond it, and a split is refused too.
    ///
    /// Mutation this catches: telling the extra shard to wait (it would ask
    /// for ever), or a limit the reconciler does not know (it would keep
    /// launching shards that are then refused).
    #[test]
    fn max_shards_refuses_a_further_shard_for_good() {
        let mut orch = fresh_cluster(OrchestratorConfig {
            min_shards: 2,
            max_shards: 2,
            ..OrchestratorConfig::default()
        });
        orch.tick();
        let root = join(&mut orch);
        assert_eq!(root.shard, ShardId::new(1));
        orch.tick();
        let second = join(&mut orch);
        assert_eq!(second.shard, ShardId::new(2));
        assert!(ack_the_prepare(&mut orch) > 0);
        assert_eq!(orch.map().regions.len(), 2);
        let _ = orch.drain_outbox();

        let ids = orch.control.ids.peek();
        let refused = orch
            .handle_shard_hello(&fresh_hello(addr(9_900)))
            .expect_err("the cluster is full");
        assert!(
            matches!(&refused, HelloRefusal::Full(why)
                if why.contains("already has 2 shard(s)") && why.contains("--max-shards")),
            "{refused:?}"
        );
        assert!(refused.is_final());
        assert_eq!(refused.reject_reason(), space_protocol::RejectReason::Full);
        assert_eq!(orch.control.ids.peek(), ids, "no id was allocated");
        assert!(orch.drain_outbox().is_empty(), "nothing was proposed");
        assert!(orch.snapshot().in_flight.is_none());
        assert!(
            event_texts(&orch)[0].starts_with("hello from 127.0.0.1:9900 refused:"),
            "a final refusal is logged"
        );

        // Nothing is launched beyond the limit, and a split is refused.
        for _ in 0..(2 * MIN_EPOCH_INTERVAL_TICKS) {
            orch.tick();
        }
        assert!(orch.take_pending_spawn().is_empty());
        let reply = orch.handle_operator(&split(1, Some(0)));
        assert!(
            !reply.accepted && reply.message.contains("--max-shards"),
            "{reply:?}"
        );
        let why = refusal(orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1))));
        assert!(why.contains("--max-shards"), "{why}");
        assert!(orch.take_pending_spawn().is_empty());
        assert_eq!(orch.snapshot().max_shards, 2);
    }

    /// Supervision: a launched shard whose process exits is dead. Nothing
    /// else changes while it is; a replacement is launched, registers, and is
    /// given the dead shard's region under a **new id**, through an epoch the
    /// dead shard is not asked to acknowledge. When that commits the scene's
    /// revision is raised — once — so that its bodies are built afresh.
    ///
    /// Mutations this catches: reusing the dead shard's id (its guid serials
    /// start over with the process, so guids would be minted twice); counting
    /// the dead shard as a recipient (the change that replaces it could never
    /// commit — the state before this change, for every change); leaving the
    /// revision alone (half a scene) or raising it on every tick.
    #[test]
    fn a_dead_shard_is_replaced_under_a_new_id_and_the_scene_revision_rises_once() {
        let mut orch = fresh_cluster(OrchestratorConfig {
            min_shards: 2,
            ..OrchestratorConfig::default()
        });
        let cell = topology::layout_cells(orch.config().root_node())[0];
        orch.tick();
        assert_eq!(join(&mut orch).shard, ShardId::new(1));
        orch.tick();
        let dead = join(&mut orch).shard;
        assert_eq!(dead, ShardId::new(2));
        assert!(ack_the_prepare(&mut orch) > 0);
        for _ in 0..MIN_EPOCH_INTERVAL_TICKS {
            orch.tick();
        }
        let _ = orch.drain_outbox();
        let before = orch.scene();
        let map_before = orch.map().clone();
        for shard in [1, 2] {
            orch.handle_scene_ack(&scene_ack(shard, before));
        }

        // Its process exits: the launcher says so on the next poll.
        let launch = orch.launch_of(dead).expect("the orchestrator launched it");
        assert_eq!(orch.launches_to_poll().len(), 2);
        orch.note_launch_dead(launch);
        assert_eq!(orch.dead_shards(), vec![dead]);
        assert!(!orch.live_shards().contains(&dead));
        assert_eq!(orch.map(), &map_before, "nothing about the map changes yet");
        assert_eq!(orch.take_pending_kill(), vec![dead]);
        assert_eq!(orch.take_pending_launch_kill(), vec![launch]);
        let texts = event_texts(&orch);
        assert!(
            texts[0].starts_with("shard 2 lost: its process exited"),
            "{texts:?}"
        );
        let role = |orch: &Orchestrator<MockLauncher>, shard| {
            orch.snapshot()
                .shards
                .iter()
                .find(|s| s.shard == shard)
                .map(|s| s.role)
        };
        assert_eq!(role(&orch, dead), Some(ShardRole::Dead));
        // Saying it again changes nothing.
        orch.note_launch_dead(launch);
        orch.declare_dead(dead, "again");
        assert!(orch.take_pending_kill().is_empty());

        // No other change while a shard of the map cannot acknowledge one.
        let reply = orch.handle_operator(&split(1, Some(0)));
        assert!(
            !reply.accepted && reply.message.contains("shard 2 is not connected"),
            "{reply:?}"
        );

        // The replacement is launched at once (no pause: nothing failed)...
        let welcome = join(&mut orch);
        let new = welcome.shard;
        assert_eq!(new, ShardId::new(3), "a new id, never the dead one's");
        assert_eq!(welcome.region.node, cell, "the dead shard's region");
        assert_eq!(
            welcome.shard_map.lookup(&cell.center()),
            Some(dead),
            "not the owner before the commit"
        );
        // ...and the change goes to the live shards only.
        let flight = orch.snapshot().in_flight.expect("in flight");
        assert_eq!(
            flight.what,
            format!(
                "replacement of shard 2: region {} goes to new shard 3",
                node_path(cell)
            )
        );
        assert_eq!(flight.changes, vec![(cell, new)]);
        assert_eq!(
            flight.recipients,
            vec![(ShardId::new(1), false), (new, false)],
            "the dead shard is not asked"
        );
        assert_eq!(role(&orch, new), Some(ShardRole::Joining));
        assert_eq!(
            prepares(&orch.drain_outbox()),
            vec![(ShardId::new(1), 3), (new, 3)]
        );
        assert_eq!(orch.scene(), before, "not before the commit");
        orch.handle_epoch_ack(ShardId::new(1), 3);
        orch.handle_epoch_ack(new, 3);

        // Committed.
        assert_eq!(orch.map().epoch, 3);
        assert_eq!(orch.map().regions.len(), 2);
        assert_eq!(orch.map().lookup(&cell.center()), Some(new));
        assert_eq!(owner_at(&orch, 0.0, 0.0), Some(ShardId::new(1)));
        crate::topology::validate(orch.map()).expect("valid");
        assert!(orch.dead_shards().is_empty());
        assert_eq!(role(&orch, dead), None, "nothing is kept of the dead shard");
        assert_eq!(role(&orch, new), Some(ShardRole::Owner));
        assert_eq!(orch.endpoint_of(dead), None);

        // The scene: the same scene, one revision up, told to every live
        // shard behind the commit and the map.
        let raised = SceneState::new(before.scene, before.revision + 1);
        assert_eq!(orch.scene(), raised);
        let out = orch.drain_outbox();
        assert_eq!(
            scene_sets(&out),
            vec![(ShardId::new(1), raised), (new, raised)]
        );
        let first_scene = out
            .iter()
            .position(|(_, m)| matches!(m, Message::SceneSet(_)))
            .expect("a scene message");
        assert!(
            out[..first_scene].iter().any(|(to, m)| *to == new
                && matches!(m, Message::DirectoryReply(r) if r.shard_map == *orch.map())),
            "the replacement has the committed map before it hears the new revision"
        );
        assert_eq!(
            sync_of(&orch, 1),
            SceneSync::Behind,
            "it has to confirm again"
        );
        let texts = event_texts(&orch);
        assert!(
            texts
                .iter()
                .any(|t| t.starts_with("shard 3 replaced shard 2: the scene")
                    && t.contains("revision 2")),
            "{texts:?}"
        );
        // Once: the revision does not move again by itself.
        for _ in 0..(2 * MIN_EPOCH_INTERVAL_TICKS) {
            orch.tick();
            orch.reconcile_scene();
        }
        assert_eq!(orch.scene(), raised);
        assert!(orch.take_pending_spawn().is_empty(), "two shards, as asked");
        // The next id is above every id ever issued (H9).
        assert_eq!(orch.control.ids.peek(), 4);
    }

    /// The same when the dead shard was started by hand and its link closes,
    /// with nobody else in the cluster: the shard that registers next — from
    /// any address — is given the root under a new id; the change has one
    /// recipient, the new shard itself.
    #[test]
    fn the_only_shard_of_a_cluster_is_replaced_by_the_next_one_that_registers() {
        let mut orch = fresh_cluster(OrchestratorConfig::default());
        let root = orch.config().root_node();
        orch.handle_shard_hello(&fresh_hello(addr(9_001)))
            .expect("the root");
        let before = orch.scene();
        orch.note_connected([]); // its control link closed
        assert_eq!(orch.dead_shards(), vec![ShardId::new(1)]);
        assert!(
            orch.take_pending_spawn().is_empty(),
            "started by hand, and no shard is run by the orchestrator: none is launched"
        );

        let welcome = orch
            .handle_shard_hello(&fresh_hello(addr(9_007)))
            .expect("welcomed in its place");
        assert_eq!(
            (welcome.shard, welcome.region.node),
            (ShardId::new(2), root)
        );
        assert_eq!(prepares(&orch.drain_outbox()), vec![(ShardId::new(2), 2)]);
        orch.handle_epoch_ack(ShardId::new(2), 2);
        assert_eq!(orch.map().epoch, 2);
        assert_eq!(orch.map().regions.len(), 1);
        assert_eq!(orch.map().regions[0].shard, ShardId::new(2));
        assert_eq!(orch.map().regions[0].endpoint, addr(9_007));
        assert_eq!(orch.scene().revision, before.revision + 1);
        assert!(orch.dead_shards().is_empty());
    }

    /// A shard that comes back at the address a registered shard advertises
    /// **is** that shard's replacement: the old process is gone (two cannot
    /// listen on one address) even if its control link has not closed yet. It
    /// is not handed the old id. But only a hello that is going to be served
    /// makes anybody dead: while a change is in flight the hello waits like
    /// any other, and the old shard is left alone.
    ///
    /// Mutation this catches: taking the old shard for dead *before* the
    /// checks that can still say "wait" (the state before: a hello that was
    /// then told to wait had already knocked a shard out — and with it the
    /// change in flight and the shard that change was for).
    #[test]
    fn a_shard_that_comes_back_at_its_address_replaces_itself_once_it_can_be_served() {
        let mut orch = fresh_cluster(OrchestratorConfig::default());
        let root = orch.config().root_node();
        let cells = topology::layout_cells(root);
        orch.handle_shard_hello(&fresh_hello(addr(9_001)))
            .expect("the root");
        let second = orch
            .handle_shard_hello(&fresh_hello(addr(9_002)))
            .expect("a cell")
            .shard;
        assert!(ack_the_prepare(&mut orch) > 0);
        // A third is being given its cell — the root's shard has not acked —
        // when the root's process restarts and says hello again.
        let third = orch
            .handle_shard_hello(&fresh_hello(addr(9_003)))
            .expect("another cell")
            .shard;
        assert_eq!(prepares(&orch.drain_outbox()).len(), 3);
        orch.handle_epoch_ack(second, 3);
        orch.handle_epoch_ack(third, 3);
        assert_eq!(orch.map().epoch, 2, "shard 1 has not acked");

        // A change is in flight: the hello waits, and nothing happens to
        // anybody — not to shard 1, not to the change, not to its shard.
        let refused = orch
            .handle_shard_hello(&fresh_hello(addr(9_001)))
            .expect_err("one change at a time");
        assert_eq!(
            refused,
            HelloRefusal::Wait("epoch 3 is still in flight; one change at a time".into())
        );
        assert!(orch.dead_shards().is_empty(), "nobody was made dead");
        assert!(orch.take_pending_kill().is_empty());
        assert!(orch.snapshot().in_flight.is_some(), "the change goes on");
        assert!(orch.live_shards().contains(&ShardId::new(1)));

        // The change runs into its deadline (shard 1's process is gone and
        // never acks); its shard is stopped, as after any abort.
        for _ in 0..=crate::config::EPOCH_ACK_TIMEOUT_TICKS {
            orch.tick();
        }
        assert_eq!(orch.map().epoch, 2);
        assert_eq!(orch.take_pending_kill(), vec![third]);
        assert!(
            orch.dead_shards().is_empty(),
            "one missed deadline is not death"
        );
        let _ = orch.drain_outbox();

        // Now the hello can be served, and it is: as shard 1's replacement.
        let welcome = orch
            .handle_shard_hello(&fresh_hello(addr(9_001)))
            .expect("the restarted shard is served");
        let new = welcome.shard;
        assert_eq!(new, ShardId::new(4), "not the old id");
        assert_eq!(welcome.region.node, root);
        let texts = event_texts(&orch);
        assert!(
            texts
                .iter()
                .any(|t| t.starts_with("shard 1 lost: a new shard registered from its address")),
            "{texts:?}"
        );
        assert_eq!(orch.take_pending_kill(), vec![ShardId::new(1)]);
        // The replacement is the change in flight now, to the live shards.
        assert_eq!(
            orch.snapshot().in_flight.expect("in flight").recipients,
            vec![(second, false), (new, false)]
        );
        orch.handle_epoch_ack(second, 3);
        orch.handle_epoch_ack(new, 3);
        assert_eq!(orch.map().epoch, 3);
        assert_eq!(orch.map().lookup(&root.center()), Some(new));
        assert_eq!(orch.map().lookup(&cells[0].center()), Some(second));
        assert_eq!(
            orch.map().regions.len(),
            2,
            "the aborted cell was not carved"
        );
        assert_eq!(orch.endpoint_of(new), Some(addr(9_001)));
        // Started by hand, `--min-shards` 0: nothing was launched for any of it.
        for _ in 0..(2 * MIN_EPOCH_INTERVAL_TICKS) {
            orch.tick();
        }
        assert!(orch.take_pending_spawn().is_empty());
    }

    /// The other half of the same rule: a replacement that somebody would
    /// have to acknowledge who is not connected is not started either — the
    /// hello waits, and the shard at whose address it arrived is not dead.
    #[test]
    fn a_hello_at_a_live_shards_address_waits_while_another_shard_could_not_ack() {
        let mut orch = fresh_cluster(OrchestratorConfig::default());
        orch.handle_shard_hello(&fresh_hello(addr(9_001)))
            .expect("the root");
        let second = orch
            .handle_shard_hello(&fresh_hello(addr(9_002)))
            .expect("a cell")
            .shard;
        assert!(ack_the_prepare(&mut orch) > 0);
        let third = orch
            .handle_shard_hello(&fresh_hello(addr(9_003)))
            .expect("a cell")
            .shard;
        assert!(ack_the_prepare(&mut orch) > 0);
        let _ = orch.drain_outbox();
        // The third shard holds no control link at this moment (the front end
        // has not seen it, or it is about to be found dead).
        orch.connected.remove(&third);

        let refused = orch
            .handle_shard_hello(&fresh_hello(addr(9_002)))
            .expect_err("the replacement could not be acknowledged");
        assert!(
            matches!(&refused, HelloRefusal::Wait(why) if why.contains("shard 3 is not connected")),
            "{refused:?}"
        );
        assert!(orch.dead_shards().is_empty(), "shard 2 is not dead for it");
        assert!(orch.take_pending_kill().is_empty());
        assert!(orch.live_shards().contains(&second));
        assert!(orch.snapshot().in_flight.is_none());
    }

    /// A hello may **name** any address. It takes a live shard's place only
    /// when its connection really comes from that address's host: from
    /// anywhere else it waits, the shard it named is untouched, and the
    /// operator is told (once). If that shard's own link then closes, the
    /// same hello is served as an ordinary replacement.
    ///
    /// Mutation this catches: believing the advertised address (anybody who
    /// may register could take any shard's regions, and have its process
    /// killed, by naming its address).
    #[test]
    fn a_hello_naming_a_live_shards_address_from_elsewhere_harms_nobody() {
        let mut orch = fresh_cluster(OrchestratorConfig::default());
        let root = orch.config().root_node();
        orch.handle_shard_hello(&fresh_hello(addr(9_001)))
            .expect("the root");
        let elsewhere: IpAddr = "10.0.0.9".parse().expect("an address");
        let impostor = fresh_hello(addr(9_001));
        let events = orch.events().len();
        for _ in 0..3 {
            let refused = orch
                .handle_shard_hello_from(&impostor, elsewhere)
                .expect_err("not from that address");
            assert!(
                matches!(&refused, HelloRefusal::Wait(why)
                    if why.contains("shard 1 is registered at 127.0.0.1:9001")
                        && why.contains("comes from 10.0.0.9")),
                "{refused:?}"
            );
            assert!(!refused.is_final(), "it may ask again");
        }
        assert!(orch.dead_shards().is_empty(), "shard 1 is not dead for it");
        assert!(orch.take_pending_kill().is_empty());
        assert!(orch.take_pending_launch_kill().is_empty());
        assert!(orch.drain_outbox().is_empty(), "nothing was proposed");
        assert_eq!(orch.control.ids.peek(), 2, "and no id was spent");
        assert_eq!(
            orch.events().len(),
            events + 1,
            "said once, however often it asks"
        );
        let said = &event_texts(&orch)[0];
        assert!(
            said.contains("is told to wait, and that will not pass by itself")
                && said.contains("comes from 10.0.0.9"),
            "{said}"
        );
        assert!(
            orch.events().newest_first().next().expect("it").peer_caused,
            "the text is partly the peer's: it goes to the bounded ring"
        );

        // Shard 1's own link closes: it is dead by the usual rule, and the
        // hello — from wherever — replaces it like any shard that registers.
        orch.note_connected([]);
        assert_eq!(orch.dead_shards(), vec![ShardId::new(1)]);
        let welcome = orch
            .handle_shard_hello_from(&impostor, elsewhere)
            .expect("an ordinary replacement now");
        assert_eq!(
            (welcome.shard, welcome.region.node),
            (ShardId::new(2), root)
        );
        orch.handle_epoch_ack(ShardId::new(2), 2);
        assert_eq!(orch.map().regions[0].shard, ShardId::new(2));
        // ...and that one, restarting in place, is recognised when its
        // address is seen in its IPv4-mapped IPv6 form too: the same host.
        let mapped: IpAddr = "::ffff:127.0.0.1".parse().expect("an address");
        let again = orch
            .handle_shard_hello_from(&impostor, mapped)
            .expect("the same host");
        assert_eq!(again.shard, ShardId::new(3));
    }

    /// A shard the orchestrator **launched** that says hello a second time —
    /// it did not see its welcome within its handshake timeout — is the very
    /// process of the launch: it gets a new id and the old one's regions, and
    /// **the process is kept**. No second shard is launched for the regions
    /// it is taking itself.
    ///
    /// Mutations this catches: staging the kill of the old shard's launch
    /// (the orchestrator would kill the process that is saying hello), and
    /// reconciling the fleet in the middle of the hello (a launch would be
    /// staged for regions that have an owner again a moment later — one shard
    /// too many, every time).
    #[test]
    fn a_launched_shard_that_says_hello_twice_keeps_its_process() {
        let mut orch = fresh_cluster(OrchestratorConfig {
            min_shards: 1,
            launcher: LauncherKind::Process,
            host_port_range: (7_000, 7_001),
            ..OrchestratorConfig::default()
        });
        let root = orch.config().root_node();
        orch.tick();
        let spec = orch.take_pending_spawn().pop().expect("one launch");
        let first = process_hello_of(&mut orch, &spec).expect("the root");
        assert_eq!(first.shard, ShardId::new(1));
        assert_eq!(orch.launch_of(first.shard), Some(spec.launch));

        // The same process, the same address, a second hello.
        let again = orch
            .handle_shard_hello(&fresh_hello(addr(spec.host_port)))
            .expect("served: it replaces itself");
        let new = again.shard;
        assert_eq!(new, ShardId::new(2), "a new id");
        assert_eq!(again.region.node, root);
        assert_eq!(
            orch.take_pending_kill(),
            vec![first.shard],
            "the old id's link is closed"
        );
        assert!(
            orch.take_pending_launch_kill().is_empty(),
            "the process is the one saying hello: it is not killed"
        );
        assert_eq!(
            orch.launch_of(new),
            Some(spec.launch),
            "the launch is the new shard's now"
        );
        assert_eq!(orch.launch_of(first.shard), None);
        assert!(
            orch.take_pending_spawn().is_empty(),
            "nothing is launched for regions this hello is taking"
        );
        assert_eq!(prepares(&orch.drain_outbox()), vec![(new, 2)]);
        // While the replacement is in flight, and after it committed: one
        // shard, as asked, and its process supervised.
        for _ in 0..10 {
            orch.tick();
            assert!(orch.take_pending_spawn().is_empty());
        }
        orch.handle_epoch_ack(new, 2);
        assert_eq!(orch.map().regions[0].shard, new);
        for _ in 0..(2 * MIN_EPOCH_INTERVAL_TICKS) {
            orch.tick();
        }
        assert!(orch.take_pending_spawn().is_empty());
        assert!(orch.take_pending_launch_kill().is_empty());
        assert_eq!(orch.launches_to_poll(), vec![spec.launch]);
        let view = orch.snapshot();
        assert_eq!(view.shards.len(), 1);
        assert!(view.shards[0].launched && view.shards[0].role == ShardRole::Owner);
    }

    /// The same for a shard **started by hand** that restarts in place while
    /// the orchestrator keeps `--min-shards` running: it takes its old
    /// regions itself, so nothing is launched for them.
    ///
    /// Mutation this catches: `declare_dead` reconciling the fleet from
    /// inside the hello — the launch staged there registered a moment later,
    /// found nothing dead and was given the next layout cell: N+1 shards, and
    /// one more on every restart.
    #[test]
    fn a_hand_started_shard_restarting_in_place_does_not_cause_a_launch() {
        let mut orch = fresh_cluster(OrchestratorConfig {
            min_shards: 2,
            ..OrchestratorConfig::default()
        });
        // The root's shard is started by hand, before the orchestrator's
        // first beat; the orchestrator launches the one that is missing.
        let by_hand = orch
            .handle_shard_hello(&fresh_hello(addr(9_001)))
            .expect("the root");
        orch.tick();
        let launched = join(&mut orch).shard;
        assert!(ack_the_prepare(&mut orch) > 0);
        for _ in 0..MIN_EPOCH_INTERVAL_TICKS {
            orch.tick();
        }
        assert!(orch.take_pending_spawn().is_empty(), "two shards, as asked");
        let _ = (orch.drain_outbox(), orch.take_pending_kill());

        for round in 0..3u32 {
            let old = orch.map().regions[0].shard;
            let welcome = orch
                .handle_shard_hello(&fresh_hello(addr(9_001)))
                .expect("it replaces itself");
            assert_ne!(welcome.shard, old, "round {round}");
            assert!(
                orch.take_pending_spawn().is_empty(),
                "round {round}: nothing is launched for regions the hello takes itself"
            );
            assert!(ack_the_prepare(&mut orch) > 0);
            for _ in 0..(2 * MIN_EPOCH_INTERVAL_TICKS) {
                orch.tick();
                assert!(orch.take_pending_spawn().is_empty(), "round {round}");
            }
            assert_eq!(orch.map().regions.len(), 2, "round {round}: still two");
            assert_eq!(orch.live_shards().len(), 2, "round {round}");
            assert_eq!(orch.starting(), 0);
        }
        assert!(orch.live_shards().contains(&launched));
        assert_ne!(orch.map().regions[0].shard, by_hand.shard);
    }

    /// A replacement is handed **the very port of the shard it replaces**
    /// (the range is exactly as large as the fleet) — while that dead shard
    /// is still known at that address, because it still owns its regions.
    /// The launch must not be put down to the dead shard: it becomes the
    /// replacement, and when *that* dies the launcher's word is enough to
    /// notice, and the port comes free again.
    ///
    /// Mutation this catches: matching a launch to any registered shard at
    /// its address. The replacement launch was tied to the dead shard, the
    /// replacement itself was supervised by nobody, its death was only
    /// noticed when its link timed out, and the launch — and its port — were
    /// kept for ever: with a range as large as the fleet no shard could be
    /// launched again.
    #[test]
    fn a_replacement_on_the_port_of_the_shard_it_replaces_is_supervised_like_any_other() {
        let mut orch = process_cluster(2, (7_000, 7_001));
        let up = bring_up(&mut orch, 2);
        let (root_launch, root) = (up[0].0.launch, up[0].1);
        let mut current = (up[1].0.clone(), up[1].1);
        assert_eq!(current.0.host_port, 7_001);
        let cell = topology::layout_cells(orch.config().root_node())[0];

        // The second shard is replaced three times over, each time on the
        // only port there is, and each death is seen through the launcher.
        for round in 0..3u32 {
            let (spec, shard) = current.clone();
            assert_eq!(orch.launch_of(shard), Some(spec.launch), "round {round}");
            orch.note_launch_dead(spec.launch);
            assert_eq!(
                orch.dead_shards(),
                vec![shard],
                "round {round}: the launcher's word is enough"
            );
            assert_eq!(orch.take_pending_launch_kill(), vec![spec.launch]);
            assert_eq!(orch.take_pending_kill(), vec![shard]);

            // Its replacement is launched at once — on the same port.
            let next = orch.take_pending_spawn().pop().expect("a replacement");
            assert_eq!(next.host_port, 7_001, "round {round}: the only free port");
            // The launcher reports before the shard has said hello: the dead
            // shard still advertises this address, and is not the launch's.
            orch.note_launch_result(next.launch, Ok(process_handle(&next)));
            assert_eq!(
                orch.launch_of(shard),
                None,
                "round {round}: not the dead one"
            );
            assert_eq!(orch.starting(), 1, "round {round}: still starting");
            let welcome = orch
                .handle_shard_hello(&fresh_hello(addr(next.host_port)))
                .expect("the replacement is welcomed");
            assert!(welcome.shard > shard, "round {round}: a new id");
            assert_eq!(welcome.region.node, cell);
            assert_eq!(
                orch.launch_of(welcome.shard),
                Some(next.launch),
                "round {round}: the launch became the replacement"
            );
            assert_eq!(orch.starting(), 0);
            assert!(ack_the_prepare(&mut orch) > 0);
            assert_eq!(orch.map().lookup(&cell.center()), Some(welcome.shard));
            assert!(orch.dead_shards().is_empty());
            for _ in 0..MIN_EPOCH_INTERVAL_TICKS {
                orch.tick();
            }
            assert!(orch.take_pending_spawn().is_empty(), "round {round}");
            assert_eq!(
                orch.launches_to_poll(),
                vec![root_launch, next.launch],
                "round {round}: two launches, no leftover"
            );
            let _ = orch.drain_outbox();
            current = (next, welcome.shard);
        }
        assert_eq!(
            orch.map().lookup(&orch.config().root_node().center()),
            Some(root)
        );
        assert_eq!(orch.live_shards().len(), 2);
        assert_eq!(orch.map().regions.len(), 2);
    }

    /// A launch whose shard is **no longer anybody** — it was stopped when
    /// its join aborted, before its launcher had reported — or is **already
    /// known to be dead**: when its process is gone the launch is removed and
    /// reaped, nothing else. It is never left in the table (holding its host
    /// port), and nobody is declared dead a second time.
    #[test]
    fn a_dead_launch_whose_shard_is_gone_or_already_dead_is_just_removed() {
        let mut orch = fresh_cluster(OrchestratorConfig {
            min_shards: 2,
            ..OrchestratorConfig::default()
        });
        orch.tick();
        let root = join(&mut orch).shard;
        orch.tick();
        // The second shard registers, and its join runs into the deadline
        // before the in-process launcher has reported which shard it became.
        let spec = orch.take_pending_spawn().pop().expect("the second launch");
        let joiner = orch
            .handle_shard_hello(&fresh_hello(launched_at(spec.launch)))
            .expect("welcomed")
            .shard;
        while orch.snapshot().in_flight.is_some() {
            orch.tick();
        }
        assert!(orch.take_pending_kill().contains(&joiner), "stopped");
        assert!(orch.endpoint_of(joiner).is_none(), "and forgotten");
        // Now the launcher reports — naming a shard nobody knows any more —
        // and then that the process is gone.
        let handle = |shard| ShardHandle {
            launch: spec.launch,
            shard: Some(shard),
            endpoint: launched_at(spec.launch),
            process: "mock".into(),
        };
        orch.note_launch_result(spec.launch, Ok(handle(joiner)));
        assert_eq!(orch.launch_of(joiner), Some(spec.launch));
        let _ = orch.take_pending_launch_kill();
        orch.note_launch_dead(spec.launch);
        assert_eq!(orch.take_pending_launch_kill(), vec![spec.launch], "reaped");
        assert_eq!(orch.launch_of(joiner), None, "and gone from the table");
        assert!(orch.dead_shards().is_empty());
        assert!(orch.live_shards().contains(&root));

        // The same for a launch tied to a shard that is already dead.
        let root_launch = orch.launch_of(root).expect("launched");
        orch.note_launch_dead(root_launch);
        assert_eq!(orch.dead_shards(), vec![root]);
        let _ = (orch.take_pending_kill(), orch.take_pending_launch_kill());
        let events = orch.events().len();
        let next = orch.take_pending_spawn().pop().expect("a replacement");
        orch.note_launch_result(
            next.launch,
            Ok(ShardHandle {
                launch: next.launch,
                shard: Some(root),
                endpoint: launched_at(next.launch),
                process: "mock".into(),
            }),
        );
        orch.note_launch_dead(next.launch);
        assert_eq!(orch.take_pending_launch_kill(), vec![next.launch]);
        assert_eq!(orch.launch_of(root), None);
        assert_eq!(orch.dead_shards(), vec![root], "still dead, once");
        assert_eq!(orch.events().len(), events, "nothing new to say about it");
    }

    /// A shard that registered **before** a launch was staged is never that
    /// launch's, whatever address it advertises: a shard started by hand on a
    /// port of the host port range is not tied to the launch that is then
    /// handed that port — and that launch failing (it cannot bind) does not
    /// take the healthy shard with it.
    #[test]
    fn a_shard_started_by_hand_on_a_port_of_the_range_is_not_taken_for_a_launch() {
        let mut orch = process_cluster(2, (7_000, 7_001));
        // Started by hand, on the first port of the range, before any launch.
        let by_hand = orch
            .handle_shard_hello(&fresh_hello(addr(7_000)))
            .expect("the root")
            .shard;
        orch.tick();
        let spec = orch.take_pending_spawn().pop().expect("the second shard");
        assert_eq!(spec.host_port, 7_000, "no launch holds it");
        orch.note_launch_result(spec.launch, Ok(process_handle(&spec)));
        assert_eq!(
            orch.launch_of(by_hand),
            None,
            "it registered before the launch was staged"
        );
        assert_eq!(orch.starting(), 1);
        // The launch's process cannot bind the port and exits.
        orch.note_launch_dead(spec.launch);
        assert!(
            orch.dead_shards().is_empty(),
            "the healthy shard is untouched"
        );
        assert!(orch.live_shards().contains(&by_hand));
        assert_eq!(orch.take_pending_launch_kill(), vec![spec.launch], "reaped");
        assert!(orch.take_pending_kill().is_empty());
        assert!(event_texts(&orch)[0].contains("on port 7000 exited before it registered"));
    }

    /// The pause after a failed launch doubles, up to 30 s, **also while
    /// other launched shards are registered and healthy**: three shards up,
    /// and the fourth exits every time before it registers.
    ///
    /// Mutation this catches: forgetting the failures whenever *any* launch
    /// has a shard — with healthy launched shards in the cluster the pause
    /// stayed at one second, and a shard that could not start was started
    /// again every second or two for ever.
    #[test]
    fn the_pause_between_failing_launches_grows_whoever_else_is_healthy() {
        let mut orch = process_cluster(4, (7_000, 7_009));
        // Three come up; the fourth launch is staged with them, and fails.
        let mut up = Vec::new();
        let mut failing = None;
        while up.len() < 3 {
            orch.tick();
            for spec in orch.take_pending_spawn() {
                if up.len() == 3 {
                    failing = Some(spec);
                    continue;
                }
                process_hello_of(&mut orch, &spec).expect("welcomed");
                ack_the_prepare(&mut orch);
                up.push(spec);
            }
        }
        assert_eq!(orch.map().regions.len(), 3);
        assert_eq!(orch.live_shards().len(), 3);
        let mut failing = failing.expect("a fourth launch was staged");

        let mut pauses = Vec::new();
        for _ in 0..8 {
            // Its process starts (the launcher reports it) and exits before
            // it registers.
            orch.note_launch_result(failing.launch, Ok(process_handle(&failing)));
            orch.note_launch_dead(failing.launch);
            assert_eq!(orch.take_pending_launch_kill(), vec![failing.launch]);
            let failed_at = orch.current_tick();
            let next = loop {
                orch.tick();
                if let Some(spec) = orch.take_pending_spawn().pop() {
                    break spec;
                }
                assert!(
                    orch.current_tick() - failed_at <= LAUNCH_BACKOFF_MAX_TICKS,
                    "no launch within the longest pause"
                );
            };
            pauses.push(orch.current_tick() - failed_at);
            failing = next;
        }
        let second = LAUNCH_BACKOFF_TICKS;
        assert_eq!(
            pauses,
            vec![
                second,
                2 * second,
                4 * second,
                8 * second,
                16 * second,
                LAUNCH_BACKOFF_MAX_TICKS,
                LAUNCH_BACKOFF_MAX_TICKS,
                LAUNCH_BACKOFF_MAX_TICKS,
            ],
            "1 s, doubling, 30 s at most"
        );
        assert_eq!(
            orch.live_shards().len(),
            3,
            "the healthy ones are untouched"
        );

        // When a launch does get its shard a region, the count starts over:
        // the next failure is followed by the short pause again.
        process_hello_of(&mut orch, &failing).expect("welcomed at last");
        assert!(ack_the_prepare(&mut orch) > 0);
        assert_eq!(orch.map().regions.len(), 4);
        let fourth = orch.map().regions[3].shard;
        orch.note_launch_dead(orch.launch_of(fourth).expect("launched"));
        let again = orch.take_pending_spawn().pop().expect("replaced at once");
        orch.note_launch_result(again.launch, Ok(process_handle(&again)));
        orch.note_launch_dead(again.launch);
        let failed_at = orch.current_tick();
        while orch.take_pending_spawn().is_empty() {
            orch.tick();
        }
        assert_eq!(
            orch.current_tick() - failed_at,
            LAUNCH_BACKOFF_TICKS,
            "one second again, not thirty"
        );
    }

    /// A launch is started for something that is missing. When a shard
    /// **started by hand** takes that first, the launch is one too many: it
    /// is stopped before it registers — and if its hello gets there first, it
    /// is refused for good and stopped, not given a layout cell. A shard
    /// started by hand that registers with nothing missing still gets the
    /// next cell.
    ///
    /// Mutation this catches: leaving the launch alone. It registered, found
    /// nothing dead, and was given the next layout cell: a shard and a region
    /// nobody asked for, and one more every time it happened.
    #[test]
    fn a_launch_whose_purpose_a_hand_started_shard_took_is_stopped_not_given_a_cell() {
        for hello_gets_there_first in [false, true] {
            let mut orch = process_cluster(2, (7_000, 7_003));
            let up = bring_up(&mut orch, 2);
            let (victim_launch, victim) = (up[1].0.launch, up[1].1);
            let cells = topology::layout_cells(orch.config().root_node());

            // The second shard dies; a replacement is launched...
            orch.note_launch_dead(victim_launch);
            assert_eq!(orch.dead_shards(), vec![victim]);
            let _ = (orch.take_pending_kill(), orch.take_pending_launch_kill());
            let replacement = orch.take_pending_spawn().pop().expect("launched");
            orch.note_launch_result(replacement.launch, Ok(process_handle(&replacement)));
            // ...but a shard started by hand registers first and takes the
            // dead shard's region.
            let by_hand = orch
                .handle_shard_hello(&fresh_hello(addr(9_900)))
                .expect("it replaces the dead shard");
            assert_eq!(by_hand.region.node, cells[0]);
            let ids = orch.control.ids.peek();
            // The launch's shard asks meanwhile: it waits its turn.
            let waiting = orch
                .handle_shard_hello(&fresh_hello(addr(replacement.host_port)))
                .expect_err("a change is in flight");
            assert!(matches!(waiting, HelloRefusal::Wait(_)), "{waiting:?}");

            if hello_gets_there_first {
                // The replacement commits and the launch's shard asks again
                // before the orchestrator's next beat.
                assert!(ack_the_prepare(&mut orch) > 0);
                let refused = orch
                    .handle_shard_hello(&fresh_hello(addr(replacement.host_port)))
                    .expect_err("nothing is left for it");
                assert!(
                    matches!(&refused, HelloRefusal::Refused(why) if why.contains("not needed")),
                    "{refused:?}"
                );
                assert!(refused.is_final(), "it does not ask again");
            } else {
                // The orchestrator's next beat sees one shard too many.
                orch.tick();
                assert!(ack_the_prepare(&mut orch) > 0);
            }
            assert_eq!(
                orch.take_pending_launch_kill(),
                vec![replacement.launch],
                "hello first: {hello_gets_there_first}: the launch is stopped"
            );
            assert_eq!(orch.starting(), 0);
            assert_eq!(orch.control.ids.peek(), ids, "and it was given no id");
            assert_eq!(orch.map().regions.len(), 2, "no region nobody asked for");
            assert_eq!(orch.map().lookup(&cells[0].center()), Some(by_hand.shard));
            let texts = event_texts(&orch);
            assert!(
                texts.iter().any(|t| t.starts_with(&format!(
                    "the shard of {} on port {}",
                    replacement.launch, replacement.host_port
                )) && t.contains("stopped")),
                "{texts:?}"
            );
            // It stays that way.
            for _ in 0..(2 * LAUNCH_REGISTER_TIMEOUT_TICKS) {
                orch.tick();
            }
            assert!(orch.take_pending_spawn().is_empty());
            assert_eq!(orch.map().regions.len(), 2);
            assert_eq!(orch.live_shards().len(), 2);

            // A shard started by hand, with nothing missing, is still given
            // the next layout cell.
            let extra = orch
                .handle_shard_hello(&fresh_hello(addr(9_901)))
                .expect("a hand-started shard is assigned a region");
            assert_eq!(extra.region.node, cells[1]);
        }
    }

    /// The in-process launcher's shards cannot be told from hand-started
    /// ones by their hello (the launcher binds a port of its own choosing).
    /// So a launch that is one too many is found by counting — and never
    /// while a shard that registered a moment ago might be that very launch,
    /// not yet matched to it.
    ///
    /// Mutation this catches: stopping the newest unmatched launch whenever
    /// the count is over (the launch of a shard that had just registered was
    /// killed under it: with `--min-shards 4` the fourth shard, every time).
    #[test]
    fn a_surplus_in_process_launch_is_stopped_but_never_one_whose_shard_just_registered() {
        let mut orch = fresh_cluster(OrchestratorConfig {
            min_shards: 2,
            ..OrchestratorConfig::default()
        });
        orch.tick();
        assert_eq!(join(&mut orch).shard, ShardId::new(1));
        orch.tick();
        // The second shard registers; its launcher has not reported yet.
        let spec = orch.take_pending_spawn().pop().expect("the second launch");
        let welcome = orch
            .handle_shard_hello(&fresh_hello(launched_at(spec.launch)))
            .expect("welcomed");
        assert!(ack_the_prepare(&mut orch) > 0);
        assert_eq!(orch.launch_of(welcome.shard), None, "not matched yet");
        assert_eq!((orch.live_shards().len(), orch.starting()), (2, 1));
        // Counted, that is three where two are wanted. Nothing is stopped: the
        // shard that just registered may be the launch — and it is.
        for _ in 0..10 {
            orch.tick();
            assert!(orch.take_pending_launch_kill().is_empty());
        }
        // A shard started by hand that asks meanwhile is told to wait, not
        // given a cell and not sent away.
        let refused = orch
            .handle_shard_hello(&fresh_hello(addr(9_900)))
            .expect_err("a launch is still being matched");
        assert!(matches!(refused, HelloRefusal::Wait(_)), "{refused:?}");
        orch.note_launch_result(
            spec.launch,
            Ok(ShardHandle {
                launch: spec.launch,
                shard: Some(welcome.shard),
                endpoint: launched_at(spec.launch),
                process: "mock".into(),
            }),
        );
        assert_eq!(orch.launch_of(welcome.shard), Some(spec.launch));
        assert_eq!(orch.starting(), 0);
        let cells = topology::layout_cells(orch.config().root_node());
        let extra = orch
            .handle_shard_hello(&fresh_hello(addr(9_900)))
            .expect("served once the launch is matched");
        assert_eq!(extra.region.node, cells[1]);
        assert!(ack_the_prepare(&mut orch) > 0);

        // Now a real surplus: the second shard dies, a launch is staged, and
        // the shard started by hand... is already there: three live owners
        // would be more than enough, but a dead one is replaced all the same
        // — unless a shard started by hand takes its region first.
        let launch = orch.launch_of(welcome.shard).expect("launched");
        orch.note_launch_dead(launch);
        let _ = (orch.take_pending_kill(), orch.take_pending_launch_kill());
        let staged = orch.take_pending_spawn().pop().expect("a replacement");
        let by_hand = orch
            .handle_shard_hello(&fresh_hello(addr(9_901)))
            .expect("it takes the dead shard's region");
        assert_eq!(by_hand.region.node, cells[0]);
        assert!(ack_the_prepare(&mut orch) > 0);
        // The shard started by hand registered a moment ago and is matched
        // to no launch: it might be the launch. Not yet, then.
        orch.tick();
        assert!(orch.take_pending_launch_kill().is_empty());
        assert_eq!(orch.starting(), 1);
        // Once that moment has passed it cannot be: the launch is surplus.
        for _ in 0..=LAUNCH_LIMBO_TICKS {
            orch.tick();
        }
        assert_eq!(orch.take_pending_launch_kill(), vec![staged.launch]);
        assert_eq!(orch.starting(), 0);
        assert_eq!(orch.map().regions.len(), 3);
        assert!(orch.take_pending_spawn().is_empty());
    }

    /// A join that cannot commit must not turn into a loop: launch, hello,
    /// new id, prepare, abort at the deadline, kill, launch again — an id
    /// every three seconds, for ever. The launch that came to nothing is
    /// followed by a pause that grows, and the shard that keeps letting the
    /// deadline pass — connected, but not answering — is taken for dead after
    /// three in a row and replaced.
    ///
    /// Mutation this catches: an abort that only stops the joiner (the loop
    /// above), or counting the joiner's own missed ack against a healthy
    /// shard.
    #[test]
    fn a_join_that_keeps_missing_its_deadline_backs_off_and_the_silent_shard_is_replaced() {
        let mut orch = fresh_cluster(OrchestratorConfig {
            min_shards: 2,
            ..OrchestratorConfig::default()
        });
        let root = orch.config().root_node();
        orch.tick();
        let silent = join(&mut orch).shard;
        assert_eq!(silent, ShardId::new(1));
        let silent_launch = orch.launch_of(silent).expect("launched");
        orch.tick();

        // Three joiners in a row: each acks, the root's shard never does.
        let mut pauses = Vec::new();
        for round in 1..=ACK_STRIKES {
            let joiner = join(&mut orch).shard;
            let launch = orch.launch_of(joiner).expect("launched");
            let epoch = orch.map().epoch + 1;
            assert_eq!(
                prepares(&orch.drain_outbox()),
                vec![(silent, epoch), (joiner, epoch)]
            );
            orch.handle_epoch_ack(joiner, epoch);
            let proposed_at = orch.current_tick();
            while orch.snapshot().in_flight.is_some() {
                orch.tick();
            }
            let aborted_at = orch.current_tick();
            assert_eq!(
                aborted_at - proposed_at,
                crate::config::EPOCH_ACK_TIMEOUT_TICKS,
                "round {round}: at the deadline"
            );
            assert_eq!(orch.map().epoch, 1, "round {round}: the map is unchanged");
            let stopped = orch.take_pending_kill();
            let reaped = orch.take_pending_launch_kill();
            assert!(stopped.contains(&joiner), "round {round}: {stopped:?}");
            assert!(reaped.contains(&launch), "round {round}: {reaped:?}");
            if round < ACK_STRIKES {
                assert!(
                    orch.dead_shards().is_empty(),
                    "round {round}: not dead for {round} missed deadline(s)"
                );
                assert_eq!(stopped, vec![joiner]);
                assert_eq!(reaped, vec![launch]);
            } else {
                assert_eq!(orch.dead_shards(), vec![silent], "three in a row");
                assert!(stopped.contains(&silent), "its link is closed");
                assert!(reaped.contains(&silent_launch), "and its process stopped");
            }
            // The next launch waits: the pause a failed launch earns.
            loop {
                orch.tick();
                if !orch.pending_spawn.is_empty() {
                    break;
                }
            }
            pauses.push(orch.current_tick() - aborted_at);
        }
        assert_eq!(
            pauses,
            vec![
                LAUNCH_BACKOFF_TICKS,
                2 * LAUNCH_BACKOFF_TICKS,
                4 * LAUNCH_BACKOFF_TICKS
            ],
            "the pause grows with every join that came to nothing"
        );
        let texts = event_texts(&orch);
        assert!(
            texts.iter().any(|t| t.starts_with(
                "shard 1 lost: it holds a control link but did not acknowledge 3 map changes in \
                 a row"
            )),
            "{texts:?}"
        );

        // Both shards are missing now, and both are launched. The first that
        // registers replaces the silent one; nobody else has to ack.
        let staged = orch.take_pending_spawn();
        assert_eq!(staged.len(), 2, "{staged:?}");
        let _ = orch.drain_outbox();
        let heir = hello_of(&mut orch, &staged[0]).expect("welcomed");
        assert_eq!(heir.region.node, root, "the silent shard's region");
        assert_eq!(
            prepares(&orch.drain_outbox()),
            vec![(heir.shard, 2)],
            "the dead shard is not asked"
        );
        orch.handle_epoch_ack(heir.shard, 2);
        assert_eq!(orch.map().epoch, 2);
        assert_eq!(orch.map().regions[0].shard, heir.shard);
        // ...and the second is given its cell, as if nothing had happened.
        let second = hello_of(&mut orch, &staged[1]).expect("welcomed");
        assert!(ack_the_prepare(&mut orch) > 0);
        assert_eq!(orch.map().regions.len(), 2);
        assert_eq!(orch.live_shards().len(), 2);
        assert!(orch.live_shards().contains(&second.shard));
        for _ in 0..(2 * MIN_EPOCH_INTERVAL_TICKS) {
            orch.tick();
        }
        assert!(orch.take_pending_spawn().is_empty(), "two shards, as asked");
        assert!(orch.dead_shards().is_empty());
    }

    /// An ack — for any change — clears the count: three missed deadlines
    /// have to be **in a row**.
    #[test]
    fn a_shard_that_answers_again_is_not_taken_for_dead() {
        let mut orch = fresh_cluster(OrchestratorConfig::default());
        let slow = orch
            .handle_shard_hello(&fresh_hello(addr(9_001)))
            .expect("the root")
            .shard;
        let miss_a_deadline = |orch: &mut Orchestrator<MockLauncher>, port: u16| {
            let joiner = orch
                .handle_shard_hello(&fresh_hello(addr(port)))
                .expect("a cell")
                .shard;
            orch.handle_epoch_ack(joiner, orch.map().epoch + 1);
            while orch.snapshot().in_flight.is_some() {
                orch.tick();
            }
        };
        miss_a_deadline(&mut orch, 9_010);
        miss_a_deadline(&mut orch, 9_011);
        assert!(orch.dead_shards().is_empty(), "two are not three");
        // It answers this time: the change commits.
        orch.handle_shard_hello(&fresh_hello(addr(9_012)))
            .expect("a cell");
        assert!(ack_the_prepare(&mut orch) > 0);
        assert_eq!(orch.map().regions.len(), 2);
        // Two more missed deadlines are again only two.
        miss_a_deadline(&mut orch, 9_013);
        miss_a_deadline(&mut orch, 9_014);
        assert!(orch.dead_shards().is_empty(), "the count started over");
        assert!(orch.live_shards().contains(&slow));
        // The third in a row is the one.
        miss_a_deadline(&mut orch, 9_015);
        assert!(orch.dead_shards().contains(&slow));
    }

    /// A report and both kinds of ack name their own sender. Off the wire
    /// they are taken only from the control connection that was admitted as
    /// that shard — the run loop hands them over with that id, and what names
    /// another shard is dropped.
    ///
    /// Mutation this catches: believing the message. A registered shard — or,
    /// before the connection was gated at all, anybody who could reach the
    /// control port — could report load for another shard (and have it split
    /// or merged), acknowledge an epoch on its behalf (and commit a change it
    /// never saw), or confirm a scene for it.
    #[test]
    fn a_report_or_an_ack_is_only_taken_from_the_shard_it_names() {
        let mut orch = fresh_cluster(OrchestratorConfig::default());
        let one = orch
            .handle_shard_hello(&fresh_hello(addr(9_001)))
            .expect("the root")
            .shard;
        let two = orch
            .handle_shard_hello(&fresh_hello(addr(9_002)))
            .expect("a cell")
            .shard;
        let epoch = orch.map().epoch + 1;
        let _ = orch.drain_outbox();
        let ack = |shard: ShardId| EpochAck { shard, epoch };

        // Shard 2's connection acknowledging for shard 1: dropped. The change
        // does not commit on shard 2's word alone.
        orch.handle_epoch_ack_from(two, &ack(one));
        orch.handle_epoch_ack_from(two, &ack(two));
        assert_eq!(orch.map().epoch, epoch - 1, "shard 1 has not acked");
        assert_eq!(
            orch.snapshot().in_flight.expect("in flight").recipients,
            vec![(one, false), (two, true)]
        );
        // Shard 2's connection reporting for shard 1: dropped.
        orch.handle_load_report_from(two, &report(one.0, 99));
        assert!(orch.load_report(one).is_none());
        // Shard 2's connection confirming a scene for shard 1: dropped (it
        // would otherwise re-issue the scene above revision 9).
        let ahead = SceneState::new(orch.scene().scene, 9);
        orch.handle_scene_ack_from(
            two,
            &SceneAck {
                shard: one,
                applied: ahead,
            },
        );
        assert_eq!(orch.scene_confirmed_by(one), None);
        orch.tick();
        orch.reconcile_scene();
        assert_eq!(orch.scene().revision, 1, "nothing was raised");
        assert_eq!(orch.misattributed(), 3, "each one is counted");

        // Each shard speaking for itself is heard.
        orch.handle_epoch_ack_from(one, &ack(one));
        assert_eq!(
            orch.map().epoch,
            epoch,
            "committed on both shards' own acks"
        );
        orch.handle_load_report_from(one, &report(one.0, 7));
        assert_eq!(orch.load_report(one).map(|r| r.players), Some(7));
        let current = orch.scene();
        orch.handle_scene_ack_from(
            two,
            &SceneAck {
                shard: two,
                applied: current,
            },
        );
        assert_eq!(orch.scene_confirmed_by(two), Some(current));
        assert_eq!(orch.misattributed(), 3);
    }

    /// The orchestrator holds the cluster's two secrets and derives `Debug`:
    /// neither shows in a print of it.
    #[test]
    fn the_orchestrator_never_prints_a_secret() {
        let config = OrchestratorConfig {
            cluster_secret: Some(crate::config::Secret::new(
                "do-not-print-the-cluster-secret",
            )),
            ..OrchestratorConfig::default()
        };
        let map = ShardMap::new(
            1,
            vec![ShardRegion::new(
                config.root_node(),
                ShardId::new(1),
                addr(7_001),
            )],
        );
        let ids = PersistentShardIdAllocator::with_sink(2, Box::new(MemoryCounterSink::default()));
        let control = ControlPlane::new(
            config,
            ids,
            EpochPublisher::new(map),
            "do-not-print-the-peer-token".into(),
        );
        let orch = Orchestrator::new(control, std::sync::Arc::new(MockLauncher::default()));
        let printed = format!("{orch:?}");
        assert!(!printed.contains("do-not-print"), "{printed}");
        assert!(printed.contains("<redacted>"), "{printed}");
    }

    /// A merge takes a shard away, and `--min-shards` says how many the
    /// orchestrator keeps running: at or below that number a merge is
    /// refused, whoever asks — it would commit, the shard be stopped, and one
    /// be started in its place at once.
    #[test]
    fn a_merge_that_would_go_below_min_shards_is_refused() {
        let with_min = |min_shards: usize| {
            let config = OrchestratorConfig {
                min_shards,
                ..OrchestratorConfig::default()
            };
            let root = config.root_node();
            let child = space_world::octree::children(root).expect("root splits")[0];
            let ids =
                PersistentShardIdAllocator::with_sink(100, Box::new(MemoryCounterSink::default()));
            let mut orch = live(orchestrator_over(
                config,
                vec![
                    ShardRegion::new(root, ShardId::new(1), addr(7_001)),
                    ShardRegion::new(child, ShardId::new(2), addr(7_002)),
                ],
                ids,
            ));
            orch.handle_load_report(&report(2, 0));
            orch
        };
        // Two shards, two to keep: no merge.
        let mut orch = with_min(2);
        let reply = orch.handle_operator(&merge(2));
        assert!(
            !reply.accepted
                && reply.message.contains("--min-shards")
                && reply
                    .message
                    .contains("has 2 shard(s) and is to keep 2 running"),
            "{reply:?}"
        );
        let why = refusal(orch.handle_admin(&AdminCommand::ForceMerge(ShardId::new(2))));
        assert!(why.contains("--min-shards"), "{why}");
        assert!(orch
            .snapshot()
            .regions
            .iter()
            .filter(|region| region.shard == ShardId::new(2))
            .all(|region| region
                .merge_refusal
                .as_deref()
                .is_some_and(|why| why.contains("--min-shards"))));
        // The autoscaler: cold and empty for well past the dwell, and nothing.
        for _ in 0..(2 * crate::autoscale::MERGE_DWELL_TICKS) {
            orch.handle_load_report(&report(2, 0));
            orch.handle_load_report(&report(1, 0));
            orch.tick();
        }
        assert_eq!(orch.pending_merge(), None);
        assert!(orch.drain_outbox().is_empty());
        assert_eq!(orch.map().regions.len(), 2);

        // Two shards, one to keep: the merge goes ahead as ever.
        let mut orch = with_min(1);
        let reply = orch.handle_operator(&merge(2));
        assert!(reply.accepted, "{reply:?}");
        assert_eq!(orch.pending_merge(), Some(ShardId::new(2)));
    }

    /// A counter that cannot be written.
    #[derive(Debug)]
    struct BrokenCounterSink;

    impl crate::ids::CounterSink for BrokenCounterSink {
        fn persist(&mut self, _next: u64) -> std::io::Result<()> {
            Err(std::io::Error::other("disk full"))
        }
    }

    /// "Wait" is not worth a log line while it passes by itself — a change in
    /// flight, a shard that is not connected yet. One that does not pass by
    /// itself is: the id counter cannot be written, so no shard can ever be
    /// given a region. It is said once, not on every hello.
    ///
    /// Mutation this catches: saying nothing (the operator saw a launched
    /// shard that "did not register within 30 s", again and again, and no
    /// reason anywhere).
    #[test]
    fn a_wait_that_will_not_pass_by_itself_is_logged_once() {
        let config = OrchestratorConfig::default();
        let root = config.root_node();
        let ids = PersistentShardIdAllocator::with_sink(2, Box::new(BrokenCounterSink));
        let mut orch = live(orchestrator_over(
            config,
            vec![ShardRegion::new(root, ShardId::new(1), addr(7_001))],
            ids,
        ));
        let events = orch.events().len();
        for _ in 0..5 {
            let refused = orch
                .handle_shard_hello(&fresh_hello(addr(9_100)))
                .expect_err("no id can be allocated");
            assert!(
                matches!(&refused, HelloRefusal::Wait(why) if why.contains("disk full")),
                "{refused:?}"
            );
        }
        assert_eq!(orch.events().len(), events + 1, "said once");
        let said = &event_texts(&orch)[0];
        assert!(
            said.contains("the shard at 127.0.0.1:9100 is told to wait")
                && said.contains("will not pass by itself")
                && said.contains("disk full"),
            "{said}"
        );
        assert!(
            !orch.events().newest_first().next().expect("it").peer_caused,
            "the orchestrator's own trouble: an ordinary entry"
        );
        // The same for a replacement: a dead shard, and no id for its heir.
        orch.note_connected([]);
        assert_eq!(orch.dead_shards(), vec![ShardId::new(1)]);
        let before = orch.events().len();
        for _ in 0..3 {
            assert!(orch.handle_shard_hello(&fresh_hello(addr(9_100))).is_err());
        }
        assert_eq!(
            orch.events().len(),
            before,
            "the same reason: not said again"
        );
    }

    /// The scene's revision is raised when a replacement commits — a write
    /// to the scene file, which can fail. Then it is owed: tried again once a
    /// second until it succeeds, so that the scene's bodies are rebuilt
    /// without an operator having to switch the scene.
    ///
    /// Mutation this catches: logging the failure and leaving it at that
    /// (half a scene until somebody noticed).
    #[test]
    fn a_scene_raise_that_fails_after_a_replacement_is_tried_again_until_it_succeeds() {
        let mut orch = fresh_cluster(OrchestratorConfig::default());
        orch.handle_shard_hello(&fresh_hello(addr(9_001)))
            .expect("the root");
        let before = orch.scene();
        orch.note_connected([]);
        orch.control.scene =
            crate::scene::SceneStore::with_sink(Scene::DemoArea, Box::new(BrokenSceneSink));
        let heir = orch
            .handle_shard_hello(&fresh_hello(addr(9_002)))
            .expect("the replacement")
            .shard;
        orch.handle_epoch_ack(heir, 2);
        assert_eq!(
            orch.map().regions[0].shard,
            heir,
            "the replacement committed"
        );
        assert_eq!(orch.scene(), before, "the raise failed");
        let warned = |orch: &Orchestrator<MockLauncher>| {
            event_texts(orch)
                .iter()
                .filter(|t| t.contains("the scene's revision could not be raised"))
                .count()
        };
        assert_eq!(warned(&orch), 1);
        let _ = orch.drain_outbox();

        // It keeps failing: tried once a second, said once.
        for _ in 0..(3 * SCENE_RESEND_INTERVAL_TICKS) {
            orch.tick();
            orch.reconcile_scene();
        }
        assert_eq!(orch.scene(), before);
        assert_eq!(warned(&orch), 1, "not once a second");
        assert!(scene_sets(&orch.drain_outbox())
            .iter()
            .all(|(_, state)| *state == before));

        // The file can be written again: within a second the revision rises,
        // once, and every shard is told.
        orch.control.scene = crate::scene::SceneStore::in_memory(Scene::DemoArea);
        for _ in 0..SCENE_RESEND_INTERVAL_TICKS {
            orch.tick();
            orch.reconcile_scene();
        }
        let raised = SceneState::new(before.scene, before.revision + 1);
        assert_eq!(orch.scene(), raised);
        assert!(scene_sets(&orch.drain_outbox()).contains(&(heir, raised)));
        let texts = event_texts(&orch);
        assert!(
            texts
                .iter()
                .any(|t| t.contains("replaced shard 1: the scene") && t.contains("revision 2")),
            "{texts:?}"
        );
        for _ in 0..(3 * SCENE_RESEND_INTERVAL_TICKS) {
            orch.tick();
            orch.reconcile_scene();
        }
        assert_eq!(orch.scene(), raised, "once");
    }

    /// Every shard of the map has to acknowledge a change, so none is asked
    /// for — by an operator, the admin verb or the autoscaler — and none is
    /// proposed for a shard that registers, while one of them holds no control
    /// link. A shard that registers meanwhile is told to wait, not sent away.
    ///
    /// Mutation this catches: checking only the shard being split (the state
    /// before: with a third shard disconnected a split was accepted, its shard
    /// launched, and the epoch rolled back at the deadline).
    #[test]
    fn no_change_is_proposed_while_a_recipient_is_not_connected() {
        let mut orch = orchestrator_with_a_child();
        // Shard 1 is there; shard 2 owns its octant and has never registered.
        orch.note_connected([ShardId::new(1)]);
        assert!(orch.dead_shards().is_empty(), "not connected is not dead");
        let state = |orch: &Orchestrator<MockLauncher>| {
            (
                orch.map().clone(),
                orch.control.ids.peek(),
                orch.wanted_node(),
                orch.starting(),
                orch.snapshot().in_flight,
            )
        };
        let before = state(&orch);

        let reply = orch.handle_operator(&split(1, Some(3)));
        assert!(
            !reply.accepted && reply.message.contains("shard 2 is not connected"),
            "{reply:?}"
        );
        let why = refusal(orch.handle_admin(&AdminCommand::ForceSplit(ShardId::new(1))));
        assert!(why.contains("shard 2 is not connected"), "{why}");
        orch.handle_load_report(&report(2, 0));
        let reply = orch.handle_operator(&merge(2));
        assert!(
            !reply.accepted && reply.message.contains("shard 2 is not connected"),
            "{reply:?}"
        );
        let why = refusal(orch.handle_admin(&AdminCommand::ForceMerge(ShardId::new(2))));
        assert!(why.contains("shard 2 is not connected"), "{why}");
        let cell = cell_at_the_centre(&orch, [0, 0, -1]);
        let err = orch
            .begin_carve(cell)
            .expect_err("nobody could acknowledge it");
        assert!(err.contains("shard 2 is not connected"), "{err}");

        // A shard that registers is told to wait.
        let refused = orch
            .handle_shard_hello(&fresh_hello(addr(9_700)))
            .expect_err("not yet");
        assert!(
            matches!(&refused, HelloRefusal::Wait(why) if why.contains("shard 2 is not connected")),
            "{refused:?}"
        );
        assert!(!refused.is_final());
        assert_eq!(
            state(&orch),
            before,
            "nothing was asked for, nothing proposed"
        );
        assert!(orch.drain_outbox().is_empty());
        assert!(orch.take_pending_spawn().is_empty());

        // The autoscaler: hot for well past the dwell, and still nothing.
        for _ in 0..(2 * crate::autoscale::SPLIT_DWELL_TICKS) {
            orch.handle_load_report(&report(1, 95));
            orch.tick();
        }
        assert!(orch.drain_outbox().is_empty());
        assert!(orch.take_pending_spawn().is_empty());
        assert_eq!(orch.map().epoch, 1);

        // Once shard 2 is there, the same hello is served: to all three.
        orch.note_connected([ShardId::new(1), ShardId::new(2)]);
        let welcome = orch
            .handle_shard_hello(&fresh_hello(addr(9_700)))
            .expect("served now");
        assert_eq!(welcome.shard, ShardId::new(100));
        assert_eq!(
            prepares(&orch.drain_outbox()),
            vec![
                (ShardId::new(1), 2),
                (ShardId::new(2), 2),
                (ShardId::new(100), 2)
            ]
        );
    }

    /// What the launcher is asked for: one launch per missing shard, each with
    /// a host port of the range (round robin: the port of a shard that just
    /// went is not the next one used) and the cluster secret; a launch that
    /// fails is followed by a pause that doubles; when no port is free nothing
    /// is launched, and that is said.
    #[test]
    fn launches_get_a_host_port_each_and_a_failed_launch_is_followed_by_a_pause() {
        let mut orch = fresh_cluster(OrchestratorConfig {
            min_shards: 4,
            launcher: LauncherKind::Process,
            host_port_range: (7_000, 7_002),
            ..OrchestratorConfig::default()
        });
        orch.control
            .set_generated_secret(crate::config::Secret::new("generated-for-the-shards"));
        orch.tick();
        let first = orch.take_pending_spawn().pop().expect("one launch");
        assert_eq!(first.host_port, 7_000);
        assert_eq!(first.launch, LaunchId(1));
        assert_eq!(first.cluster, orch.config().cluster);
        assert_eq!(first.control_addr, orch.config().control_addr());
        assert_eq!(first.secret, "generated-for-the-shards");

        // The launcher could not start it.
        orch.note_launch_result(first.launch, Err("spawn failed: no such file".into()));
        assert_eq!(orch.starting(), 0);
        assert!(event_texts(&orch)[0].starts_with("launching a shard failed (launch 1)"));
        for _ in 0..(LAUNCH_BACKOFF_TICKS - 1) {
            orch.tick();
            assert!(
                orch.take_pending_spawn().is_empty(),
                "not before the pause is over"
            );
        }
        orch.tick();
        let second = orch.take_pending_spawn().pop().expect("tried again");
        assert_eq!(second.host_port, 7_001, "the next port, not the same one");
        // It starts, and exits before it ever registers: twice as long a pause.
        orch.note_launch_result(
            second.launch,
            Ok(ShardHandle {
                launch: second.launch,
                shard: None,
                endpoint: addr(7_001),
                process: "pid 1".into(),
            }),
        );
        assert_eq!(orch.launches_to_poll(), vec![second.launch]);
        orch.note_launch_dead(second.launch);
        assert_eq!(
            orch.take_pending_launch_kill(),
            vec![second.launch],
            "reaped"
        );
        assert!(event_texts(&orch)[0].contains("on port 7001 exited before it registered"));
        for _ in 0..(2 * LAUNCH_BACKOFF_TICKS - 1) {
            orch.tick();
            assert!(orch.take_pending_spawn().is_empty());
        }
        orch.tick();
        let third = orch.take_pending_spawn().pop().expect("tried again");
        assert_eq!(third.host_port, 7_002);

        // This one registers: the root. It is found by the address it
        // advertises — at once, by the host port the launch was handed, even
        // before the launcher has reported where its shard listens.
        let endpoint = addr(third.host_port);
        let root = orch
            .handle_shard_hello(&fresh_hello(endpoint))
            .expect("the root");
        assert_eq!(
            orch.launch_of(root.shard),
            Some(third.launch),
            "matched by its host port, the launcher's report or not"
        );
        assert_eq!(orch.starting(), 0);
        orch.note_launch_result(
            third.launch,
            Ok(ShardHandle {
                launch: third.launch,
                shard: None,
                endpoint,
                process: "pid 2".into(),
            }),
        );
        assert_eq!(orch.launch_of(root.shard), Some(third.launch));
        assert_eq!(orch.starting(), 0);

        // Three more are wanted and two ports are free: two launches, and the
        // third is said to be impossible.
        orch.tick();
        let more = orch.take_pending_spawn();
        let ports: Vec<u16> = more.iter().map(|spec| spec.host_port).collect();
        assert_eq!(ports, vec![7_000, 7_001]);
        assert!(
            event_texts(&orch)
                .iter()
                .any(|t| t.contains("every host port from 7000 to 7002 is in use")),
            "{:?}",
            event_texts(&orch)
        );
        // The in-process launcher needs no port.
        let mut local = fresh_cluster(OrchestratorConfig {
            min_shards: 1,
            host_port_range: (7_000, 7_000),
            ..OrchestratorConfig::default()
        });
        local.tick();
        assert_eq!(local.take_pending_spawn()[0].host_port, 0);
    }

    /// The admin verb's kill takes the path a death takes: the shard is
    /// stopped and, if it owns a region, replaced.
    #[test]
    fn a_shard_killed_by_the_admin_verb_is_replaced() {
        let mut orch = fresh_cluster(OrchestratorConfig {
            min_shards: 1,
            ..OrchestratorConfig::default()
        });
        orch.tick();
        let first = join(&mut orch).shard;
        let launch = orch.launch_of(first).expect("launched");
        assert!(accepted(
            &orch.handle_admin(&AdminCommand::KillShard(first))
        ));
        assert_eq!(orch.take_pending_kill(), vec![first]);
        assert_eq!(orch.take_pending_launch_kill(), vec![launch]);
        assert_eq!(orch.dead_shards(), vec![first]);
        let replacement = join(&mut orch);
        assert_ne!(replacement.shard, first);
        assert_eq!(replacement.region.node, orch.config().root_node());
        assert!(ack_the_prepare(&mut orch) > 0);
        assert_eq!(orch.map().regions[0].shard, replacement.shard);
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
                client_endpoint: None,
                secret: String::new(),
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
        let new = join(&mut orch).shard;
        let in_flight = orch.snapshot().in_flight.expect("epoch 2 is in flight");
        assert_eq!(in_flight.epoch, 2);

        // The scene changes anyway, at once.
        let reply = orch.handle_operator(&set_scene(Scene::Galaxy));
        assert!(reply.accepted, "{reply:?}");
        assert_eq!(orch.scene(), at(Scene::Galaxy, 2));
        let out = orch.drain_outbox();
        let prepared = prepares(&out);
        assert!(!prepared.is_empty(), "the prepare is still going out");
        // Everybody who has registered is told, the shard of the split
        // included (it is connected; it only owns nothing yet).
        assert_eq!(
            scene_sets(&out),
            vec![
                (ShardId::new(1), at(Scene::Galaxy, 2)),
                (ShardId::new(2), at(Scene::Galaxy, 2)),
                (new, at(Scene::Galaxy, 2))
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
        let _ = join(&mut orch);
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

        // The same scene again: no new revision, nothing sent, and an answer
        // that cannot be taken for a switch. Nobody has confirmed yet, so it
        // also says who is still being waited for.
        let reply = orch.handle_operator(&set_scene(Scene::Solar));
        assert!(reply.accepted, "{reply:?}");
        assert_eq!(reply.scene_changed, None, "nothing changed");
        assert_eq!(
            reply.message,
            "already running solar (revision 5); nothing to change. Still waiting for shard 1, \
             shard 2 to confirm it (told again every second)"
        );
        assert_eq!(orch.scene(), at(Scene::Solar, 5));
        assert_eq!(orch.snapshot().scene.changed_tick, Some(0));
        assert!(orch.drain_outbox().is_empty(), "nothing is sent for it");
        // Once everybody has confirmed, there is simply nothing to do.
        orch.handle_scene_ack(&scene_ack(1, at(Scene::Solar, 5)));
        orch.handle_scene_ack(&scene_ack(2, at(Scene::Solar, 5)));
        let reply = orch.handle_operator(&set_scene(Scene::Solar));
        assert_eq!(
            reply.message,
            "already running solar (revision 5); nothing to do"
        );
        assert!(reply.accepted && reply.scene_changed.is_none());
        // A real change, by contrast, names the state it made current.
        let reply = orch.handle_operator(&set_scene(Scene::Galaxy));
        assert_eq!(reply.scene_changed, Some(at(Scene::Galaxy, 6)));
        // And nothing else an operator can ask for claims to have changed it.
        let reply = orch.handle_operator(&OperatorCommand::PauseAutoscaler);
        assert_eq!(reply.scene_changed, None);
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

        // A shard reporting a revision too far ahead to be anything but
        // nonsense is not asked again: it would change nothing.
        orch.handle_scene_ack(&scene_ack(1, at(Scene::Galaxy, NONSENSE_REVISION)));
        for _ in 0..(2 * SCENE_RESEND_INTERVAL_TICKS) {
            assert!(beat(&mut orch).is_empty());
        }
        assert_eq!(sync_of(&orch, 1), SceneSync::Differs);
    }

    /// A revision no real cluster reaches: past what the orchestrator will
    /// catch up with ([`MAX_REVISION_CATCH_UP`]).
    const NONSENSE_REVISION: u64 = MAX_REVISION_CATCH_UP * 4;

    /// The convergence rule, with a real shard state machine on the other
    /// end: an orchestrator that restarted **without its scene file** counts
    /// from 1 again while a shard (and that shard's clients) still holds a
    /// higher revision. Left alone, nothing the orchestrator says is ever
    /// adopted. It finds out from the shard's answer, re-issues its scene
    /// above the shard's revision, and from then on is heard again — by the
    /// shard and by its client.
    ///
    /// Mutation this catches: treating a shard that is ahead as merely "not
    /// the cluster's" (every later switch is accepted, saved, sent — and
    /// ignored by that shard for ever), or adopting the *shard's* scene (the
    /// orchestrator would stop being the authority).
    #[test]
    fn an_orchestrator_that_lost_its_scene_file_gets_back_above_a_shard_that_kept_running() {
        use space_server::shard::{Destination, Shard};
        use space_server::ShardConfig;

        // The shard has been running since before the restart: solar, rev 9,
        // with one client connected that was told so.
        let mut shard = Shard::new(ShardConfig::new(
            ShardId::new(1),
            addr(9_001),
            OrchestratorConfig::default().root_node(),
        ));
        assert!(shard.set_scene(at(Scene::Solar, 9)));
        let player = shard
            .join(&space_protocol::Hello {
                protocol: space_protocol::PROTOCOL_VERSION,
                player_token: "player".into(),
            })
            .expect("joins")
            .session;
        let _ = shard.take_outbox();

        // The orchestrator starts over: the default scene at revision 1.
        let mut orch = orchestrator_with_first_id(100);
        orch.handle_shard_hello(&ShardHello {
            cluster: OrchestratorConfig::default().cluster,
            shard: ShardId::new(1),
            endpoint: addr(9_001),
            region: None,
            client_endpoint: None,
            secret: String::new(),
        })
        .expect("the shard registers");
        orch.note_connected([ShardId::new(1)]);
        assert_eq!(orch.scene(), at(Scene::DemoArea, 1));

        // Everything the two say to each other, until nothing is left to say.
        // Returns what the shard told its client on the way.
        let exchange = |orch: &mut Orchestrator<MockLauncher>, shard: &mut Shard| {
            let mut told_the_client = Vec::new();
            let mut to_shard = vec![orch.scene_message()];
            for _ in 0..10 {
                for message in to_shard.drain(..) {
                    shard.orchestrator_message(message);
                }
                for out in shard.take_outbox() {
                    match (out.dest, out.message) {
                        (Destination::Orchestrator, Message::SceneAck(ack)) => {
                            orch.handle_scene_ack(&ack);
                        }
                        (Destination::Client(session), Message::SceneUpdate(state)) => {
                            told_the_client.push((session, state));
                        }
                        other => panic!("unexpected {other:?}"),
                    }
                }
                // One beat of the run loop (well over a second apart, so the
                // once-a-second limit on raises is not what is being tested).
                for _ in 0..=SCENE_RESEND_INTERVAL_TICKS {
                    orch.tick();
                }
                orch.reconcile_scene();
                to_shard = orch
                    .drain_outbox()
                    .into_iter()
                    .filter(|(to, _)| *to == ShardId::new(1))
                    .map(|(_, message)| message)
                    .collect();
                if to_shard.is_empty() {
                    break;
                }
            }
            told_the_client
        };

        // The welcome's scene (demo-area, 1) is not adopted by the shard; its
        // answer makes the orchestrator re-issue demo-area above revision 9.
        let told = exchange(&mut orch, &mut shard);
        assert_eq!(
            orch.scene(),
            at(Scene::DemoArea, 10),
            "the orchestrator's scene, one above the shard's revision"
        );
        assert_eq!(
            shard.scene(),
            at(Scene::DemoArea, 10),
            "and the shard follows"
        );
        assert_eq!(told, vec![(player, at(Scene::DemoArea, 10))]);
        assert_eq!(sync_of(&orch, 1), SceneSync::InSync);
        assert_eq!(orch.snapshot().scene.revision, 10);
        let texts = event_texts(&orch);
        assert!(
            texts
                .iter()
                .any(|t| t.contains("a shard runs scene revision 9")
                    && t.contains("demo-area is re-issued as revision 10")),
            "{texts:?}"
        );

        // From here on a switch works like any other: accepted, adopted, told.
        let reply = orch.handle_operator(&set_scene(Scene::Galaxy));
        assert_eq!(reply.scene_changed, Some(at(Scene::Galaxy, 11)));
        let told = exchange(&mut orch, &mut shard);
        assert_eq!(shard.scene(), at(Scene::Galaxy, 11));
        assert_eq!(told, vec![(player, at(Scene::Galaxy, 11))]);
        assert_eq!(sync_of(&orch, 1), SceneSync::InSync);
    }

    /// The same rule in the three shapes it can take, and its two limits.
    #[test]
    fn the_orchestrator_gets_above_a_reported_revision_once_a_second_and_within_reason() {
        let mut orch = two_registered_shards();
        let _ = orch.drain_outbox();
        let log_before = orch.events().len();

        // The same revision with another scene is "not above" too.
        orch.handle_scene_ack(&scene_ack(1, at(Scene::Solar, 1)));
        assert_eq!(
            orch.scene(),
            at(Scene::DemoArea, 1),
            "nothing until the beat"
        );
        orch.tick();
        orch.reconcile_scene();
        assert_eq!(orch.scene(), at(Scene::DemoArea, 2));
        assert_eq!(
            scene_sets(&orch.drain_outbox()),
            vec![
                (ShardId::new(1), at(Scene::DemoArea, 2)),
                (ShardId::new(2), at(Scene::DemoArea, 2))
            ],
            "re-issued to every shard"
        );
        assert_eq!(orch.snapshot().scene.changed_tick, Some(1));

        // Several shards ahead, several answers: one raise, above the highest,
        // and not before a second has passed since the last one.
        orch.handle_scene_ack(&scene_ack(1, at(Scene::Solar, 7)));
        orch.handle_scene_ack(&scene_ack(2, at(Scene::Galaxy, 30)));
        orch.handle_scene_ack(&scene_ack(1, at(Scene::Solar, 12)));
        for _ in 1..SCENE_RESEND_INTERVAL_TICKS {
            orch.tick();
            orch.reconcile_scene();
            assert_eq!(
                orch.scene(),
                at(Scene::DemoArea, 2),
                "at most one raise (one write of the scene file) a second"
            );
        }
        orch.tick();
        orch.reconcile_scene();
        assert_eq!(orch.scene(), at(Scene::DemoArea, 31));
        assert_eq!(orch.events().len(), log_before + 2, "one entry per raise");

        // Nonsense is not followed (it would leave no revision above it): it
        // is shown and said once, and the scene can still be changed.
        orch.handle_scene_ack(&scene_ack(2, at(Scene::Galaxy, u64::MAX - 1)));
        for _ in 0..(2 * SCENE_RESEND_INTERVAL_TICKS) {
            orch.tick();
            orch.reconcile_scene();
        }
        assert_eq!(orch.scene(), at(Scene::DemoArea, 31));
        assert_eq!(sync_of(&orch, 2), SceneSync::Differs);
        assert!(event_texts(&orch)[0].starts_with("shard 2 reports scene galaxy"));
        assert!(orch.handle_operator(&set_scene(Scene::Solar)).accepted);
        assert_eq!(orch.scene(), at(Scene::Solar, 32));

        // An operator's change that already got above a reported revision
        // leaves nothing to catch up with.
        let mut orch = two_registered_shards();
        orch.handle_scene_ack(&scene_ack(1, at(Scene::Solar, 1)));
        assert!(orch.handle_operator(&set_scene(Scene::Galaxy)).accepted);
        orch.tick();
        orch.reconcile_scene();
        assert_eq!(orch.scene(), at(Scene::Galaxy, 2), "no second bump");

        // A raise that cannot be saved is not made, says so, and is tried
        // again; the shard stays visibly out of step meanwhile.
        let mut orch = two_registered_shards();
        orch.control.scene =
            crate::scene::SceneStore::with_sink(Scene::DemoArea, Box::new(BrokenSceneSink));
        orch.handle_scene_ack(&scene_ack(1, at(Scene::Solar, 9)));
        orch.tick();
        orch.reconcile_scene();
        assert_eq!(orch.scene(), at(Scene::DemoArea, 1));
        assert!(event_texts(&orch)[0].contains("could not be raised above it"));
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

        // A revision far past anything this cluster could have issued is
        // recorded and said once.
        orch.handle_scene_ack(&scene_ack(2, at(Scene::Galaxy, NONSENSE_REVISION)));
        assert_eq!(sync_of(&orch, 2), SceneSync::Differs);
        assert!(
            event_texts(&orch)[0].starts_with("shard 2 reports scene galaxy (revision "),
            "{:?}",
            event_texts(&orch)[0]
        );
        let logged = orch.events().len();
        orch.handle_scene_ack(&scene_ack(2, at(Scene::Galaxy, NONSENSE_REVISION + 1)));
        orch.handle_scene_ack(&scene_ack(2, at(Scene::Galaxy, NONSENSE_REVISION + 2)));
        for _ in 0..(2 * SCENE_RESEND_INTERVAL_TICKS) {
            orch.tick();
            orch.reconcile_scene();
        }
        assert_eq!(orch.events().len(), logged, "once, not for every repeat");
        assert_eq!(orch.scene(), at(Scene::Solar, 2), "and it changes no scene");
    }

    /// A shard that comes back (it restarted) is a new shard: it has confirmed
    /// nothing yet, whatever its previous incarnation said, and nothing is kept
    /// about the old one; a shard that left the cluster is forgotten too.
    ///
    /// (A restarted shard used to register again under its old id. It is now
    /// its own replacement, under a new one.)
    #[test]
    fn a_shard_that_comes_back_has_to_confirm_again() {
        let mut orch = two_registered_shards();
        orch.handle_scene_ack(&scene_ack(1, at(Scene::DemoArea, 1)));
        assert_eq!(sync_of(&orch, 1), SceneSync::InSync);
        // Its old id is not something it can register under a second time.
        let refused = orch
            .handle_shard_hello(&ShardHello {
                cluster: OrchestratorConfig::default().cluster,
                shard: ShardId::new(1),
                endpoint: addr(9_001),
                region: None,
                client_endpoint: None,
                secret: String::new(),
            })
            .expect_err("an id is used by one process, once");
        assert!(refused.is_final(), "{refused:?}");
        assert!(
            refused.to_string().contains("has already registered"),
            "{refused}"
        );
        assert_eq!(sync_of(&orch, 1), SceneSync::InSync, "nothing changed");

        // Without an id, from its old address: it replaces itself.
        let welcome = orch
            .handle_shard_hello(&fresh_hello(addr(9_001)))
            .expect("the restarted shard is welcomed");
        let new = welcome.shard;
        assert_eq!(new, ShardId::new(100), "a new id");
        assert_eq!(welcome.region.node, orch.config().root_node());
        assert!(ack_the_prepare(&mut orch) > 0);
        assert_eq!(orch.map().regions[0].shard, new);
        assert_eq!(sync_of(&orch, 100), SceneSync::Unconfirmed);
        assert!(orch.scene_confirmed_by(ShardId::new(1)).is_none());
        assert!(
            !orch
                .snapshot()
                .shards
                .iter()
                .any(|s| s.shard == ShardId::new(1)),
            "the old id is gone from the cluster"
        );

        // Shard 2 is merged away: nothing about it is kept.
        for _ in 0..MIN_EPOCH_INTERVAL_TICKS {
            orch.tick();
        }
        let _ = orch.drain_outbox();
        orch.handle_scene_ack(&scene_ack(2, orch.scene()));
        assert!(orch.scene_confirmed_by(ShardId::new(2)).is_some());
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

    // ------------------------------------------------------------ one address

    /// `0.0.0.0:port`: "the host the player dialed, this port".
    fn any_host(port: u16) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), port)
    }

    /// The hello of a shard the `process` launcher started on `port`: it
    /// listens on loopback and sends its players to the host they dialed.
    fn process_hello(port: u16) -> ShardHello {
        ShardHello {
            client_endpoint: Some(any_host(port)),
            ..fresh_hello(addr(port))
        }
    }

    /// What the player port answers follows the owner of the scene's spawn
    /// point through its life: "starting" while nobody has claimed the root;
    /// a redirect to the root's shard, with the published map, once one has;
    /// "starting" again from the moment that shard is known to be dead — a
    /// redirect would send the player to a shard that is gone — and a
    /// redirect to the **replacement**, under its new id, as soon as its
    /// epoch has committed. Shards that own other ground never change it.
    ///
    /// Mutation this catches: answering with the map's owner whether it is
    /// there or not (players are sent to a dead address and wait out a
    /// handshake timeout), or with the reserved root id before any shard has
    /// claimed it (the map names the control port as its address).
    #[test]
    fn the_player_port_follows_the_owner_of_the_spawn_point() {
        let mut orch = process_cluster(2, (7_870, 7_879));
        assert_eq!(orch.spawn_owner(), Some(ShardId::new(1)));
        assert_eq!(
            orch.player_answer(),
            PlayerAnswer::Starting,
            "the root is reserved, nobody has claimed it"
        );
        assert_eq!(orch.player_target(), None);

        // The root's shard registers: players are sent to it, with the map.
        orch.tick();
        let spec = orch.take_pending_spawn().pop().expect("the root's launch");
        let root = orch
            .handle_shard_hello(&process_hello(spec.host_port))
            .expect("the root");
        orch.note_launch_result(spec.launch, Ok(process_handle(&spec)));
        assert_eq!(root.shard, ShardId::new(1));
        let PlayerAnswer::Redirect(redirect) = orch.player_answer() else {
            panic!("a shard owns the spawn point: {:?}", orch.player_answer());
        };
        assert_eq!(redirect.shard, ShardId::new(1));
        assert_eq!(&redirect.shard_map, orch.map());
        assert_eq!(
            redirect.shard_map.client_endpoint_of(ShardId::new(1)),
            Some(any_host(spec.host_port)),
            "the host the player dialed, the shard's port"
        );
        assert_eq!(redirect.shard_map.regions[0].endpoint, addr(spec.host_port));

        // A second shard takes a layout cell: the spawn point is not in it.
        orch.tick();
        let second_spec = orch.take_pending_spawn().pop().expect("the second launch");
        let second = orch
            .handle_shard_hello(&process_hello(second_spec.host_port))
            .expect("a layout cell");
        orch.note_launch_result(second_spec.launch, Ok(process_handle(&second_spec)));
        assert!(ack_the_prepare(&mut orch) > 0);
        assert_eq!(orch.map().regions.len(), 2);
        assert_eq!(orch.player_target(), Some(ShardId::new(1)));
        assert_eq!(
            orch.map().client_endpoint_of(second.shard),
            Some(any_host(second_spec.host_port)),
            "a carved shard's client endpoint is in the committed map"
        );
        let _ = (orch.drain_outbox(), orch.take_pending_kill());

        // The root's shard dies: nobody to send a player to.
        orch.declare_dead(ShardId::new(1), "its process exited");
        assert_eq!(
            orch.spawn_owner(),
            Some(ShardId::new(1)),
            "the map is unchanged"
        );
        assert_eq!(orch.player_answer(), PlayerAnswer::Starting);
        let port = orch.snapshot().player_port;
        assert_eq!(
            port, None,
            "no player port configured: none in the snapshot"
        );

        // Its replacement registers; until the epoch commits it owns nothing.
        orch.tick();
        let replacement_spec = orch
            .take_pending_spawn()
            .pop()
            .expect("a replacement is launched");
        let replacement = orch
            .handle_shard_hello(&process_hello(replacement_spec.host_port))
            .expect("the dead shard's region");
        orch.note_launch_result(
            replacement_spec.launch,
            Ok(process_handle(&replacement_spec)),
        );
        assert_eq!(replacement.shard, ShardId::new(3), "a new id");
        assert_eq!(orch.player_answer(), PlayerAnswer::Starting);
        assert!(ack_the_prepare(&mut orch) > 0);
        // Committed: the replacement is the owner, at its own address.
        let PlayerAnswer::Redirect(redirect) = orch.player_answer() else {
            panic!("the replacement owns the spawn point");
        };
        assert_eq!(redirect.shard, ShardId::new(3));
        assert_eq!(
            redirect.shard_map.client_endpoint_of(ShardId::new(3)),
            Some(any_host(replacement_spec.host_port))
        );
        assert_ne!(replacement_spec.host_port, spec.host_port, "another port");
        assert_eq!(orch.snapshot().spawn_owner, Some(ShardId::new(3)));
    }

    /// The player port and each shard's client endpoint are in the snapshot:
    /// where the port listens and whom it sends players to, per region and
    /// per shard where players dial — beside the internal address.
    #[test]
    fn the_snapshot_shows_the_player_port_and_every_client_endpoint() {
        let player_bind: SocketAddr = "127.0.0.1:7777".parse().expect("an address");
        let mut orch = fresh_cluster(OrchestratorConfig {
            player_bind: Some(player_bind),
            ..OrchestratorConfig::default()
        });
        let before = orch.snapshot();
        let port = before.player_port.expect("a player port is configured");
        assert_eq!((port.addr, port.sends_to), (player_bind, None));
        assert_eq!(before.regions[0].client_endpoint, None, "a placeholder");

        orch.handle_shard_hello(&process_hello(7_870))
            .expect("the root");
        let after = orch.snapshot();
        let port = after.player_port.expect("a player port");
        assert_eq!(port.sends_to, Some(ShardId::new(1)));
        assert_eq!(after.spawn_owner, Some(ShardId::new(1)));
        assert_eq!(after.regions[0].endpoint, Some(addr(7_870)));
        assert_eq!(after.regions[0].client_endpoint, Some(any_host(7_870)));
        let shard = &after.shards[0];
        assert_eq!(shard.advertised, Some(addr(7_870)));
        assert_eq!(shard.client_endpoint, Some(any_host(7_870)));
        assert_eq!(
            orch.client_endpoint_of(ShardId::new(1)),
            Some(any_host(7_870))
        );

        // A shard that asks for none: players dial the address it advertises.
        let mut plain = fresh_cluster(OrchestratorConfig::default());
        plain
            .handle_shard_hello(&fresh_hello(addr(9_001)))
            .expect("the root");
        let snapshot = plain.snapshot();
        assert_eq!(snapshot.player_port, None);
        assert_eq!(snapshot.regions[0].client_endpoint, Some(addr(9_001)));
        assert_eq!(snapshot.shards[0].client_endpoint, Some(addr(9_001)));
    }

    /// A shard that advertises an address **without an IP** (`0.0.0.0:port`:
    /// a container that does not know its own) is recorded with the IP its
    /// control connection comes from: that is what other shards are told to
    /// dial. The client endpoint keeps its own meaning — there the
    /// unspecified IP is "the host the player dialed" and is left alone.
    ///
    /// The rule of phase 1b holds with the filled-in address: a hello from
    /// that same host and port takes the live shard's place (it *is* from
    /// that address), one from another host is simply another address — it
    /// cannot name the first one's by leaving the IP out.
    ///
    /// Mutation this catches: publishing `0.0.0.0` as a shard's internal
    /// endpoint (no neighbour could dial it), or treating two shards on
    /// different hosts that both say `0.0.0.0:7870` as one address (the
    /// second would knock the first out).
    #[test]
    fn a_shard_without_an_ip_of_its_own_is_listed_at_the_address_it_is_seen_from() {
        let mut orch = fresh_cluster(OrchestratorConfig::default());
        let root = orch.config().root_node();
        let seen_from: IpAddr = "10.1.2.3".parse().expect("an address");
        let hello = ShardHello {
            endpoint: any_host(7_870),
            client_endpoint: Some(any_host(7_870)),
            ..fresh_hello(addr(0))
        };
        let welcome = orch
            .handle_shard_hello_from(&hello, seen_from)
            .expect("the root");
        let observed: SocketAddr = "10.1.2.3:7870".parse().expect("an address");
        assert_eq!(welcome.region.endpoint, observed);
        assert_eq!(welcome.region.client_endpoint, any_host(7_870));
        assert_eq!(orch.endpoint_of(welcome.shard), Some(observed));
        assert_eq!(orch.map().regions[0].endpoint, observed);
        assert_eq!(orch.map().regions[0].client_endpoint, any_host(7_870));
        let snapshot = orch.snapshot();
        assert_eq!(snapshot.shards[0].advertised, Some(observed));
        assert_eq!(snapshot.regions[0].endpoint, Some(observed));
        assert!(
            event_texts(&orch)[0].contains("registered from 10.1.2.3:7870"),
            "{:?}",
            event_texts(&orch)
        );

        // Without a client endpoint of its own, players dial that address.
        let mut plain = fresh_cluster(OrchestratorConfig::default());
        let bare = ShardHello {
            endpoint: any_host(7_870),
            ..fresh_hello(addr(0))
        };
        plain
            .handle_shard_hello_from(&bare, seen_from)
            .expect("the root");
        assert_eq!(plain.map().regions[0].client_endpoint, observed);

        // Another host that says the same `0.0.0.0:7870` is another address:
        // it is given a layout cell, and shard 1 is not touched.
        let other_host: IpAddr = "10.1.2.4".parse().expect("an address");
        let second = orch
            .handle_shard_hello_from(&hello, other_host)
            .expect("a shard of its own");
        assert_ne!(second.region.node, root);
        assert_eq!(
            second.region.endpoint,
            "10.1.2.4:7870".parse::<SocketAddr>().expect("an address")
        );
        assert!(orch.dead_shards().is_empty(), "nobody was knocked out");
        assert!(ack_the_prepare(&mut orch) > 0);
        assert_eq!(orch.map().regions.len(), 2);
        let _ = (orch.drain_outbox(), orch.take_pending_kill());
        for _ in 0..MIN_EPOCH_INTERVAL_TICKS {
            orch.tick();
        }

        // The first host's shard comes back (same host, same port, still
        // without an IP of its own): it replaces itself in place, under a
        // new id — the rule for a hello at a live shard's address.
        let again = orch
            .handle_shard_hello_from(&hello, seen_from)
            .expect("it takes its own place");
        assert_eq!(again.region.node, root);
        assert_eq!(again.region.endpoint, observed);
        assert_eq!(again.shard, ShardId::new(3));
        assert_eq!(orch.dead_shards(), vec![ShardId::new(1)]);
    }

    /// A replacement's client endpoint goes into the staged map with its
    /// endpoint, and the map push behind every commit carries both.
    #[test]
    fn a_map_push_carries_every_client_endpoint() {
        let mut orch = fresh_cluster(OrchestratorConfig::default());
        orch.handle_shard_hello(&process_hello(7_870))
            .expect("the root");
        orch.handle_shard_hello(&process_hello(7_871))
            .expect("a layout cell");
        let outbox = orch.drain_outbox();
        for (shard, epoch) in prepares(&outbox) {
            orch.handle_epoch_ack(shard, epoch);
        }
        let pushed: Vec<ShardMap> = orch
            .drain_outbox()
            .into_iter()
            .filter_map(|(_, message)| match message {
                Message::DirectoryReply(reply) => Some(reply.shard_map),
                _ => None,
            })
            .collect();
        assert_eq!(pushed.len(), 2, "one push per owner");
        for map in &pushed {
            assert_eq!(map, orch.map());
            for (shard, port) in [(1, 7_870), (2, 7_871)] {
                let region = map
                    .regions
                    .iter()
                    .find(|r| r.shard == ShardId::new(shard))
                    .expect("in the map");
                assert_eq!(
                    (region.endpoint, region.client_endpoint),
                    (addr(port), any_host(port))
                );
            }
        }
    }
}
