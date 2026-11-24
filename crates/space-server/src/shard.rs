//! [`Shard`]: the state one tick of the shard loop drives.
//!
//! Everything here is synchronous and deterministic: given the same sequence of
//! [`Shard::join`] / [`Shard::apply_input`] / [`Shard::peer_message`] calls and
//! the same number of [`Shard::tick`] calls, two shards produce the same
//! [`SimWorld::guid_state_hash`]. The quinn front end ([`crate::net`]) is the
//! only async part, and it only moves messages in and out. That split is what
//! lets the acceptance tests run two shards inside one process and hand their
//! outboxes to each other.

use std::collections::{BTreeMap, HashMap, HashSet};

use space_protocol::{
    EntityGuid, HandoffReason, Hello, InputTick, JoinRedirect, Message, NeighbourStatus,
    PlayerInput, Redirect, RejectReason, SceneAck, SceneClock, SceneState, ShardId, ShardMap,
    ShardRegion, StaleEpoch, Welcome,
};
use space_sim::{ShipControls, ShipState, SimTick, SimWorld};
use space_world::{NodeRef, WorldPos};

use crate::config::{
    ShardConfig, ADOPT_TIMEOUT_TICKS, MAX_PENDING_INPUTS, REDIRECT_TIMEOUT_TICKS,
    REOFFER_BACKOFF_TICKS, SCENE_REANNOUNCE_TICKS,
};
use crate::epoch::EpochParticipant;
use crate::handoff::{
    Ledger, TransferKind, HANDOFF_LEAD, HANDOFF_TIMEOUT_TICKS, MAX_CATCH_UP, MAX_STAGED_INPUTS,
};
use crate::replicate::{
    self, ClientReplica, ComponentLog, Components, Frame, GuidIndex, Interest, PassCache, Visible,
};

// The physics half of the shard: the scene's bodies, spawn points, ghosts at
// the borders, island consolidation and the body handoff path. A child
// module, so it works on the same private state as the rest of this file.
#[path = "shard_physics.rs"]
mod physics;
pub use physics::BodyInfo;

// The far half: what the other shards simulate in view of this shard's
// players, and what this shard tells them of its own.
#[path = "far.rs"]
mod far;
pub use far::{
    FarInfo, FAR_COMPONENTS, FAR_INTERESTS, FAR_INTEREST_STALE_TICKS, FAR_INTEREST_TICKS,
    FAR_MAX_LAG, FAR_ORPHAN_TICKS, FAR_RECEIVE_BYTES, FAR_REFRESH_TICKS, FAR_SILENCE_TICKS,
    MAX_FAR_ENTITY_BYTES, MAX_FAR_PER_PEER,
};

/// Hard cap on concurrent client sessions per shard (docs/plan.md U9).
pub const MAX_PLAYERS_PER_SHARD: usize = 100;

/// Most undelivered messages the outbox holds before old client deltas are
/// discarded.
///
/// The front end drains the outbox after every tick, so a running shard never
/// holds more than one tick's worth (at most one frame per client plus a few
/// handoff messages). The bound is for a caller that steps the shard *without*
/// draining it — [`Shard::tick_n`] from a tool or a test. Every client is sent a
/// frame every tick, so an undrained outbox would otherwise grow by one message
/// per client per tick for as long as nobody collects it. Only what a later
/// message makes redundant is discarded, oldest first: a client's
/// [`Message::Delta`] frames (a newer delta supersedes an older one, and what a
/// discarded one carried is re-sent once the client's acknowledgements show it
/// never arrived — see [`crate::replicate`]) and its [`Message::SceneUpdate`]s
/// (the scene is announced again every
/// [`SCENE_REANNOUNCE_TICKS`](crate::config::SCENE_REANNOUNCE_TICKS)).
/// Snapshots, redirects and everything addressed to a shard or the orchestrator
/// are kept.
pub const MAX_OUTBOX: usize = 4096;

/// Why a hello was not answered with a [`Welcome`] ([`Shard::join`]).
#[derive(Debug, Clone, PartialEq)]
pub enum JoinRefusal {
    /// Refused, for the reason given (the shard is full, or it would send the
    /// player on but does not know where to yet:
    /// [`RejectReason::Starting`]).
    Rejected(RejectReason),
    /// Not here: a new player starts on the scene's spawn point, and another
    /// shard owns it. The client is to dial that shard instead.
    Elsewhere(JoinRedirect),
}

impl JoinRefusal {
    /// The message that answers the hello in place of a welcome.
    #[must_use]
    pub fn into_message(self) -> Message {
        match self {
            JoinRefusal::Rejected(reason) => Message::Reject(space_protocol::Reject { reason }),
            JoinRefusal::Elsewhere(redirect) => Message::JoinRedirect(redirect),
        }
    }
}

impl From<RejectReason> for JoinRefusal {
    fn from(reason: RejectReason) -> Self {
        JoinRefusal::Rejected(reason)
    }
}

/// One message routed to a peer: a client session or another shard.
#[derive(Debug, Clone)]
pub struct OutMessage {
    /// Who should receive it.
    pub dest: Destination,
    /// The payload.
    pub message: Message,
}

/// Where an [`OutMessage`] goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Destination {
    /// A connected client, by the session id the [`Welcome`] gave it.
    Client(u64),
    /// Another shard, by id.
    Shard(ShardId),
    /// The orchestrator's control connection (epoch messages, map pushes).
    Orchestrator,
}

/// A connected client.
#[derive(Debug)]
pub struct Client {
    /// The session id this shard issued.
    pub session: u64,
    /// The player token from the [`Hello`] (opaque; not authenticated today).
    pub player_token: String,
    /// The ship this client drives, or `None` while it is **awaiting adopt**.
    ///
    /// A client that connects to us while we hold a staged transfer owned by its
    /// token is a redirected player: it is coming for that entity, so we do not
    /// spawn a placeholder for it (that placeholder was a phantom ship at the
    /// region centre whenever such a client crashed before reporting in). It
    /// gets its ship by adopting the staged entity ([`Shard::client_status`]).
    pub ship: Option<EntityGuid>,
    /// What we have told this client and what we still owe it.
    pub replica: ClientReplica,
    /// Inputs received since the last tick, drained by [`Shard::tick`].
    pub pending_inputs: Vec<PlayerInput>,
    /// A redirect we sent that has not been confirmed yet.
    pub redirect_pending: Option<ShardId>,
    /// The tick by which a redirect confirmation must arrive (see
    /// [`crate::config::REDIRECT_TIMEOUT_TICKS`]); `None` once confirmed.
    pub redirect_deadline: Option<SimTick>,
    /// The tick by which an awaiting-adopt client must adopt its staged entity
    /// (see [`crate::config::ADOPT_TIMEOUT_TICKS`]); `None` once it has a ship.
    pub adopt_deadline: Option<SimTick>,
}

/// Counters worth exposing (and asserting on in tests).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ShardStats {
    /// Simulation tick.
    pub tick: SimTick,
    /// Live ships.
    pub ships: usize,
    /// Connected clients.
    pub clients: usize,
    /// Handoffs completed as the offering shard.
    pub handed_off: usize,
    /// Handoffs completed as the accepting shard.
    pub received: usize,
    /// Handoffs abandoned.
    pub aborted: usize,
    /// Physics bodies simulated here that are not ships.
    pub bodies: usize,
    /// Ghosts of neighbours' bodies held here.
    pub ghosts: usize,
    /// Ticks this shard jumped over because it could not keep up with the
    /// cluster clock (see [`crate::clock`]).
    pub skipped: SimTick,
    /// Far replicas of other shards' entities held here (`far.rs`).
    pub far: usize,
    /// Far messages of peers that were dropped: over a bound, or naming a
    /// shard the connection is not.
    pub far_refused: u64,
    /// Bytes of far messages sent to other shards, in all.
    pub far_bytes_out: u64,
    /// Bytes of far updates taken from other shards, in all.
    pub far_bytes_in: u64,
    /// Bytes of frames built for clients, in all.
    pub client_bytes: u64,
    /// Ticks handed-over entities were brought forward at their install
    /// because the commit arrived late (see [`crate::handoff`], guarantee 4).
    pub caught_up: SimTick,
    /// Wall clock minus this shard's tick clock as last measured, µs:
    /// positive when the shard is behind (see [`crate::clock`], "Keeping to
    /// the wall clock"). 0 for a shard that is not run by [`crate::clock`].
    pub clock_offset_us: i64,
}

/// One transfer that was called off, as [`Shard::recent_aborts`] lists it
/// and the shard's log says it.
#[derive(Debug, Clone, PartialEq)]
pub struct AbortNote {
    /// This shard's tick when it happened.
    pub tick: SimTick,
    /// The entity whose transfer it was.
    pub entity: EntityGuid,
    /// The other shard of the transfer.
    pub peer: ShardId,
    /// Whether this shard was the one offering the entity (`true`) or the
    /// one that had staged it.
    pub offering: bool,
    /// Whether the entity is a ship a client of this shard flies.
    pub player: bool,
    /// Whether it was called off here (`true`) or by the peer's abort.
    pub by_us: bool,
    /// Why.
    pub reason: HandoffReason,
}

/// How many [`AbortNote`]s a shard keeps.
const ABORT_NOTES: usize = 64;
/// Abort log lines per window of [`ABORT_LINE_WINDOW`] ticks.
const ABORT_LINES: u32 = 6;
/// The window of [`ABORT_LINES`], ticks (ten seconds).
const ABORT_LINE_WINDOW: SimTick = 10 * space_sim::TICK_HZ as SimTick;

/// A commit that has arrived but not been installed yet.
///
/// The gap between "the commit is delivered" and "our step reaches
/// `install_at`" is where the entity must **not** be simulated: its state is
/// the state as of the offering shard's activation tick, so stepping it earlier
/// would double-step.
///
/// **When** it is installed is decided by the tick the state describes
/// (`state_tick`, see [`crate::handoff`], "The tick of a state"): the first
/// tick pass that finds our clock at or past it installs the entity and, if
/// it is past it, brings it forward by the ticks in between. Only when the
/// two shards' clocks do not agree (`state_tick` is `None`) are the old two
/// numbers used: `activation_tick`, the *offering* shard's clock, as the
/// input-ack floor, and `install_at`, **our** clock, derived when we accepted
/// the offer.
#[derive(Debug)]
struct PendingCommit {
    guid: EntityGuid,
    session: u64,
    state: ShipState,
    /// The offering shard's activation tick: the input-ack floor we hand the
    /// client when the clocks do not agree (its inputs are then tagged in
    /// that clock).
    activation_tick: SimTick,
    /// Our own tick at which to install: the state's tick, or — clocks that
    /// do not agree — the accept tick plus the lead (see
    /// [`crate::handoff::Incoming`]).
    install_at: SimTick,
    /// The tick of **our** clock the state describes, when the sender's
    /// scene time of it lies within [`MAX_CATCH_UP`] of our present.
    state_tick: Option<SimTick>,
    /// Whether the ship was in fast travel (its `ShipControls`).
    fast: bool,
    /// What arrives: a player's ship (spawned from `state`) or a body
    /// (imported from `components`).
    kind: TransferKind,
    /// A body's full component list, private components stripped.
    components: Components,
    /// The shard that committed it (and now holds a ghost of it).
    from: ShardId,
    /// The sender's scene time of the state
    /// ([`crate::handoff::SCENE_TIME_COMPONENT`]).
    scene_time: Option<i64>,
}

impl PendingCommit {
    /// The ticks to bring the entity forward by when it is installed at
    /// `tick`: the ticks since the one its state describes — or **none** if
    /// that is more than [`MAX_CATCH_UP`] (or the clocks do not agree). The
    /// catch-up is flight without contacts; it is good for the few ticks a
    /// commit takes to arrive, never for the seconds of a shard that stood
    /// still ([`Shard::install_pending_commits`]).
    pub(crate) fn catch_up_at(&self, tick: SimTick) -> SimTick {
        self.state_tick
            .map(|at| tick.saturating_sub(at))
            .filter(|late| *late <= MAX_CATCH_UP)
            .unwrap_or(0)
    }
}

/// One shard: its region, its simulation, its clients and its handoff ledger.
#[derive(Debug)]
pub struct Shard {
    config: ShardConfig,
    sim: SimWorld,
    guids: GuidIndex,
    interest: Interest,
    clients: HashMap<u64, Client>,
    /// The next session id to hand out.
    next_session: u64,
    /// The next guid serial for this shard.
    next_serial: u64,
    ledger: Ledger,
    /// Commits waiting for their activation tick (see [`PendingCommit`]).
    pending_commits: Vec<PendingCommit>,
    /// Inputs for ships that are on their way here (staged, or committed and
    /// not yet installed), by ship and tick: their players send them to both
    /// shards around the switch, and the install replays the ones for the
    /// ticks it catches up on and files the rest (guarantee 5).
    staged_inputs: HashMap<EntityGuid, BTreeMap<SimTick, PlayerInput>>,
    /// Entities a peer refused, with the peer and the tick before which they are
    /// not offered to it again (see
    /// [`REOFFER_BACKOFF_TICKS`](crate::config::REOFFER_BACKOFF_TICKS)).
    refused: HashMap<EntityGuid, (ShardId, SimTick)>,
    outbox: Vec<OutMessage>,
    /// Entities whose commit went out this tick, with the shard they went to;
    /// turned into ghosts of that shard's copy once the tick's frames are
    /// built (see [`Shard::tick`] step 7).
    departing: Vec<(EntityGuid, ShardId)>,
    /// Ghosts of neighbours' bodies held here (see [`crate::border`]).
    ghosts: HashMap<EntityGuid, physics::GhostEntry>,
    /// The tick a [`Message::GhostUpdate`] last arrived from each shard.
    ghost_heard: HashMap<ShardId, SimTick>,
    /// Per neighbour: the ghosts it holds of ours, as last sent.
    ghost_sent: HashMap<ShardId, HashMap<EntityGuid, physics::SentGhost>>,
    /// Per entity and component: the bytes last replicated and when they
    /// changed (deltas carry changed components only).
    log: ComponentLog,
    /// The scene the physics world was built for, and its anchor.
    installed: Option<(SceneState, WorldPos)>,
    /// The islands of the last border pass.
    island_view: physics::IslandView,
    /// Per body this shard simulates: the clocks of the time hysteresis
    /// (see [`crate::border`], "Hysteresis").
    clocks: HashMap<EntityGuid, physics::BodyClock>,
    /// Bodies this shard sent home after an interaction: how often in a row
    /// they came back soon after, and when they last went (the hold's
    /// backoff, see [`crate::border`]).
    sent_home: HashMap<EntityGuid, (u32, SimTick)>,
    /// The last group id handed to an island transfer.
    next_group: u64,
    /// What the last census line said (see `Shard::report_census`).
    census: Option<ShardStats>,
    /// The byte counters when the census last looked (far out, far in,
    /// clients): what its rates are taken against.
    census_traffic: (u64, u64, u64),
    /// Far replicas: held, and sent (`far.rs`).
    far: far::FarState,
    /// The transfers called off lately, newest last (see [`AbortNote`]).
    abort_notes: std::collections::VecDeque<AbortNote>,
    /// Abort lines written in the current window, and the tick it began at
    /// (a few lines per ten seconds; the rest are counted).
    abort_lines: (SimTick, u32, u64),
    /// The shard map this shard publishes and follows.
    shard_map: ShardMap,
    /// The shard's epoch-protocol participation (Phase 3): the staged prepare,
    /// if one is outstanding.
    epoch_participant: EpochParticipant,
    /// The scene clients are told to draw, and the revision it was set at (see
    /// [`Shard::set_scene`]).
    scene: SceneState,
    /// Whether the scene clock was set ([`Shard::set_scene_clock`]): this
    /// shard counts scene time on a shared clock, not from its own zero.
    scene_clock_set: bool,
    /// Shards a map change took out of the map a moment ago, with where they
    /// were reached and the tick until which they are still heard (see
    /// [`FORMER_SHARD_GRACE_TICKS`]).
    former: HashMap<ShardId, (std::net::SocketAddr, SimTick)>,
    /// The tick times since the last load report ([`crate::load`]).
    load: crate::load::LoadWindow,
    /// Whether this shard reports its load to an orchestrator
    /// ([`Shard::report_load`]), and the tick it last did.
    reports_load: Option<SimTick>,
    stats: ShardStats,
}

/// How long a shard that a map change removed from the map is still heard by
/// the others, in ticks (60 s).
///
/// A region that is merged away leaves its shard owning nothing, with ships
/// and bodies still in its world: it **drains** — everything it simulates
/// lies on another shard's ground now and is offered there by the ordinary
/// handoff — and is stopped by the orchestrator when it is empty
/// (`docs/plan-sharding.md` §3). For that the shards of the new map must
/// still take its offers, commits and ghosts, and know where to answer it,
/// although their map no longer names it. Long enough for any drain (the
/// orchestrator gives one up after at most 45 s); a shard that died and was
/// replaced is "heard" for this long too, and says nothing.
pub const FORMER_SHARD_GRACE_TICKS: SimTick = 60 * space_sim::TICK_HZ as SimTick;

impl Shard {
    /// An empty shard owning `config.region`.
    #[must_use]
    pub fn new(config: ShardConfig) -> Self {
        let shard = config.shard;
        let map = ShardMap::new(
            1,
            vec![ShardRegion::new(config.region, shard, config.endpoint)],
        );
        Self {
            config,
            sim: SimWorld::new(shard),
            guids: GuidIndex::new(),
            interest: Interest::new(),
            clients: HashMap::new(),
            next_session: 1,
            next_serial: 1,
            ledger: Ledger::new(),
            pending_commits: Vec::new(),
            staged_inputs: HashMap::new(),
            refused: HashMap::new(),
            outbox: Vec::new(),
            departing: Vec::new(),
            ghosts: HashMap::new(),
            ghost_heard: HashMap::new(),
            ghost_sent: HashMap::new(),
            log: ComponentLog::new(),
            installed: None,
            island_view: physics::IslandView::default(),
            clocks: HashMap::new(),
            sent_home: HashMap::new(),
            next_group: 0,
            census: None,
            census_traffic: (0, 0, 0),
            far: far::FarState::default(),
            abort_notes: std::collections::VecDeque::new(),
            abort_lines: (0, 0, 0),
            shard_map: map,
            epoch_participant: EpochParticipant::new(),
            scene: SceneState::UNSET,
            scene_clock_set: false,
            former: HashMap::new(),
            load: crate::load::LoadWindow::default(),
            reports_load: None,
            stats: ShardStats::default(),
        }
    }

    /// Switch the load reports on: from now on this shard queues a
    /// [`LoadReport`](space_protocol::LoadReport) for its orchestrator every
    /// [`crate::load::REPORT_INTERVAL_TICKS`] (an orchestrated shard; a
    /// standalone one has nobody to tell and never calls this).
    pub fn report_load(&mut self) {
        self.reports_load = Some(self.sim.tick());
    }

    /// What this shard would report about its load right now (and the tick
    /// times collected since the last report are taken: the next window
    /// starts here).
    pub fn load_report(&mut self) -> space_protocol::LoadReport {
        let cells: Vec<[i64; 3]> = self
            .clients
            .values()
            .filter_map(|client| client.ship)
            .filter_map(|ship| self.sim.ship_state(ship))
            .map(|state| state.pos.cell)
            .collect();
        let entities = self
            .sim
            .bodies()
            .iter()
            .filter(|(guid, _, ghost)| !ghost && self.sim.ship_state(*guid).is_some())
            .count();
        crate::load::report(
            self.config.shard,
            u32::try_from(self.clients.len()).unwrap_or(u32::MAX),
            &cells,
            u32::try_from(entities).unwrap_or(u32::MAX),
            self.stats.skipped,
            self.load.take(),
        )
    }

    /// Remember the shards `new` no longer names, so that they are still
    /// heard for a while ([`FORMER_SHARD_GRACE_TICKS`]); called with the map
    /// that is about to be replaced still in place.
    fn note_departing(&mut self, new: &ShardMap) {
        let until = self.sim.tick() + FORMER_SHARD_GRACE_TICKS;
        let own = self.config.shard;
        for region in &self.shard_map.regions {
            let shard = region.shard;
            let stays = new.regions.iter().any(|r| r.shard == shard);
            if shard != own && shard != ShardId::NONE && !stays && region.endpoint.port() != 0 {
                self.former.insert(shard, (region.endpoint, until));
            }
        }
        // Whoever is (back) in the map is no former shard.
        self.former
            .retain(|shard, _| !new.regions.iter().any(|r| r.shard == *shard));
    }

    /// The mapping from this shard's ticks to scene time (see
    /// [`space_protocol::SceneClock`]).
    #[must_use]
    pub fn scene_clock(&self) -> SceneClock {
        self.sim.scene_clock()
    }

    /// Use `clock` from now on and tell every client
    /// ([`Shard::start_clock`], tests).
    pub fn set_scene_clock(&mut self, clock: SceneClock) {
        self.sim.set_scene_clock(clock);
        self.scene_clock_set = true;
        self.send_to_all(Message::SceneClock(clock));
    }

    /// Start counting at `tick` ([`Shard::start_clock`]).
    pub(crate) fn restart_ticks(&mut self, tick: SimTick) {
        self.sim.reset_clock(tick);
    }

    /// Jump forward to `tick` without simulating the ticks in between: what
    /// a shard does that is too far behind the cluster clock to make them up
    /// ([`crate::clock`]). Nothing moves by itself across the jump — no
    /// body flies on, falls or collides for the ticks that are skipped —
    /// with one exception: scene time does not stop, so the next step moves
    /// the orbiting large bodies by the whole of it **and carries what is
    /// near them along** ([`space_sim::SimWorld::skip_to`]): a ship standing
    /// on the Moon is still standing on it. Inputs filed for the skipped
    /// ticks are dropped, the ones for later ticks are kept; every timeout
    /// simply finds that much time gone. A `tick` at or before the present
    /// does nothing.
    pub fn skip_to(&mut self, tick: SimTick) {
        let at = self.sim.tick();
        if tick <= at {
            return;
        }
        self.stats.skipped += tick - at;
        self.sim.skip_to(tick);
    }

    /// Record what the tick clock last measured against the wall clock
    /// ([`ShardStats::clock_offset_us`]; the loop in [`crate::clock`] calls
    /// this every turn).
    pub fn set_clock_offset(&mut self, offset_us: i64) {
        self.stats.clock_offset_us = offset_us;
    }

    /// This shard's id.
    #[must_use]
    pub const fn shard(&self) -> ShardId {
        self.config.shard
    }

    /// Record the address the front end actually bound to: where other shards
    /// and players alike reach this shard (a standalone shard has one address).
    ///
    /// A shard started on port 0 does not know its port until the socket exists,
    /// and the shard map it hands out in a [`Welcome`] must carry a reachable one.
    pub fn set_endpoint(&mut self, addr: std::net::SocketAddr) {
        self.set_endpoints(addr, addr);
    }

    /// Record this shard's two addresses in its own regions of the map: the
    /// internal `endpoint` other shards dial, and the `client_endpoint`
    /// players are sent to ([`ShardRegion::client_endpoint`]). An orchestrated
    /// shard takes both from the region its `ShardWelcome` names — they are
    /// the orchestrator's to decide.
    pub fn set_endpoints(
        &mut self,
        endpoint: std::net::SocketAddr,
        client_endpoint: std::net::SocketAddr,
    ) {
        self.config.endpoint = endpoint;
        for region in &mut self.shard_map.regions {
            if region.shard == self.config.shard {
                region.endpoint = endpoint;
                region.client_endpoint = client_endpoint;
            }
        }
    }

    /// Where the shard map says `peer` can be reached, if it says so usefully.
    ///
    /// An unresolved endpoint ([`ShardRegion::UNRESOLVED_ENDPOINT`], port 0) is
    /// skipped: a region carved by an epoch change names its new owner before the
    /// orchestrator's map push has supplied the owner's real address, and until
    /// then the carved region carries that un-dialable sentinel. An endpoint
    /// equal to this shard's own is skipped too, since dialling it would loop
    /// back to this shard.
    #[must_use]
    pub fn peer_endpoint(&self, peer: ShardId) -> Option<std::net::SocketAddr> {
        self.shard_map
            .regions
            .iter()
            .filter(|region| region.shard == peer)
            .map(|region| region.endpoint)
            .find(|endpoint| *endpoint != self.config.endpoint && endpoint.port() != 0)
            // A shard the map named until a moment ago is still answered
            // where it was (it is draining: [`FORMER_SHARD_GRACE_TICKS`]).
            .or_else(|| {
                self.former
                    .get(&peer)
                    .filter(|(_, until)| self.sim.tick() <= *until)
                    .map(|(endpoint, _)| *endpoint)
            })
    }

    /// The region this shard is authoritative for.
    #[must_use]
    pub const fn region(&self) -> NodeRef {
        self.config.region
    }

    /// The simulation (read-only).
    #[must_use]
    pub const fn sim(&self) -> &SimWorld {
        &self.sim
    }

    /// The guid index (read-only).
    #[must_use]
    pub const fn guids(&self) -> &GuidIndex {
        &self.guids
    }

    /// The shard map this shard publishes.
    #[must_use]
    pub fn shard_map(&self) -> &ShardMap {
        &self.shard_map
    }

    /// Replace the shard map (an orchestrator, or a test wiring two shards).
    ///
    /// A new map also ends every re-offer pause: a refusal was a statement about
    /// the map it was made under.
    pub fn set_shard_map(&mut self, map: ShardMap) {
        self.note_departing(&map);
        self.shard_map = map;
        self.refused.clear();
        self.forget_departed_neighbours();
        self.ensure_scene();
    }

    /// The current epoch.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.shard_map.epoch
    }

    /// The shard's epoch-protocol participant (read-only).
    #[must_use]
    pub fn epoch_participant(&self) -> &EpochParticipant {
        &self.epoch_participant
    }

    /// The scene this shard tells its clients to draw, and its revision.
    ///
    /// [`SceneState::UNSET`] (the default scene, revision 0) until somebody
    /// sets one: the command line of a standalone shard, or the orchestrator.
    #[must_use]
    pub const fn scene(&self) -> SceneState {
        self.scene
    }

    /// Make `state` this shard's scene and tell every connected client — if it
    /// [supersedes](SceneState::supersedes) the scene held. Returns whether it
    /// did.
    ///
    /// The scene is what clients draw around the ships **and** what this
    /// shard simulates: adopting one builds the physics world from it
    /// (statics, gravity and the scene's bodies whose initial position this
    /// shard owns; see [`crate::border`]), replacing the previous scene's
    /// bodies. No tick, handoff or epoch is affected. Only a strictly newer
    /// revision is adopted, which is what makes a late or repeated message
    /// harmless: it changes nothing and is relayed to nobody.
    pub fn set_scene(&mut self, state: SceneState) -> bool {
        if !state.supersedes(&self.scene) {
            return false;
        }
        self.scene = state;
        self.send_to_all(Message::SceneUpdate(state));
        self.ensure_scene();
        true
    }

    /// Install a committed shard map and immediately abort any in-flight
    /// transfer whose epoch no longer matches.
    ///
    /// This is the "shards get the new map" seam (plan F9). It is a *call-order*
    /// guarantee rather than new logic: [`Shard::handoff_pass`] already aborts a
    /// stale-epoch transfer the moment the map epoch moves, so calling this from
    /// the commit path simply makes that abort happen on the commit tick instead
    /// of on the next tick's sweep. Guarantee 7 (a handoff cannot straddle a map
    /// change) is unchanged.
    pub fn apply_epoch(&mut self, map: ShardMap) {
        // Installing epoch N supersedes any prepare staged for N or earlier: a
        // leftover staged prepare would otherwise wedge every later prepare
        // (the participant allows one outstanding prepare at a time).
        let _ = self.epoch_participant.discard_through(map.epoch);
        self.note_departing(&map);
        self.shard_map = map;
        // A refusal under the old map says nothing about the new one.
        self.refused.clear();
        self.forget_departed_neighbours();
        self.ensure_scene();
        // Abort every in-flight outgoing transfer that is now stale, so the guid
        // is released immediately (guarantee 8) rather than after the next tick.
        let stale: Vec<EntityGuid> = self
            .ledger
            .outgoing()
            .iter()
            .filter(|(_, e)| e.epoch != self.epoch())
            .map(|(g, _)| *g)
            .collect();
        for guid in stale {
            self.abort_outgoing(guid, HandoffReason::EpochChanged);
        }
    }

    /// Handle an epoch-protocol message from the orchestrator.
    ///
    /// The orchestrator is not a shard in this shard's map, so these messages do
    /// not arrive through [`Shard::peer_message`] (which drops anything from an
    /// off-map id); the front end routes them here. The handled messages are
    /// the shard's half of the two-phase epoch publication (plan H10):
    ///
    /// * [`Message::EpochPrepare`] — stage the change and answer with
    ///   [`Message::EpochAck`] to the orchestrator;
    /// * [`Message::EpochCommit`] — install the new map (via [`Shard::apply_epoch`],
    ///   which also aborts stale in-flight transfers);
    /// * [`Message::EpochAbort`] — discard the staged change; the map is untouched;
    /// * [`Message::DirectoryReply`] — the orchestrator's authoritative map push,
    ///   which fills in the endpoints a commit cannot carry.
    ///
    /// One more message arrives on the same link and is **not** part of that
    /// protocol: [`Message::SceneSet`], the cluster's scene. It is adopted
    /// through [`Shard::set_scene`] and always answered with a
    /// [`Message::SceneAck`] naming the scene the shard runs afterwards. It
    /// neither reads nor touches the map, the staged prepare or any transfer.
    ///
    /// A prepare for an epoch that is not exactly one above the current one, or a
    /// second prepare while one is staged, is ignored (the participant rejects
    /// it), matching the publisher's "one change at a time" rule.
    pub fn orchestrator_message(&mut self, message: Message) {
        match message {
            Message::EpochPrepare(prepare) => {
                if let Some(ack) =
                    self.epoch_participant
                        .handle_prepare(&prepare, self.epoch(), self.config.shard)
                {
                    self.send_to(Destination::Orchestrator, Message::EpochAck(ack));
                }
            }
            Message::EpochCommit(commit) => {
                if let Some(new_map) = self
                    .epoch_participant
                    .handle_commit(&commit, &self.shard_map)
                {
                    self.apply_epoch(new_map);
                }
            }
            Message::EpochAbort(abort) => {
                // Discard the staged change; the map is untouched. Whether a
                // matching prepare was staged is not acted on here — an abort for
                // an epoch we never staged is simply a no-op.
                let _ = self.epoch_participant.handle_abort(&abort);
            }
            Message::DirectoryReply(reply) => {
                // The orchestrator's authoritative map push (plan F9): the same
                // ownership as the epoch just committed, with every region's real
                // endpoint. A map older than ours is ignored; the same epoch only
                // refreshes endpoints; a newer one (a commit we missed) is
                // installed like a commit, so stale transfers abort.
                let map = reply.shard_map;
                if map.epoch > self.epoch() {
                    self.apply_epoch(map);
                } else if map.epoch == self.epoch() {
                    self.set_shard_map(map);
                }
            }
            Message::SceneSet(state) => {
                // Adopted only if newer (a repeat or a late message is a no-op),
                // and confirmed either way with what this shard runs *now*: the
                // orchestrator learns the truth, not an echo of what it sent.
                let _ = self.set_scene(state);
                self.send_to(
                    Destination::Orchestrator,
                    Message::SceneAck(SceneAck {
                        shard: self.config.shard,
                        applied: self.scene,
                    }),
                );
            }
            other => {
                debug_assert!(false, "unexpected orchestrator message: {other:?}");
            }
        }
    }

    /// Counters.
    #[must_use]
    pub fn stats(&self) -> ShardStats {
        let mut s = self.stats;
        s.tick = self.sim.tick();
        s.ships = self.sim.ship_count();
        s.clients = self.clients.len();
        s.ghosts = self.ghosts.len();
        s.far = self.far.len();
        s.bodies = self
            .sim
            .bodies()
            .iter()
            .filter(|(guid, _, ghost)| !ghost && self.sim.ship_state(*guid).is_some())
            .count()
            .saturating_sub(s.ships);
        s
    }

    /// Messages the front end must deliver, draining the queue.
    pub fn take_outbox(&mut self) -> Vec<OutMessage> {
        std::mem::take(&mut self.outbox)
    }

    /// Queue `message` for `dest`.
    pub fn send_to(&mut self, dest: Destination, message: Message) {
        self.outbox.push(OutMessage { dest, message });
    }

    /// Queue `message` for every connected client.
    pub fn send_to_all(&mut self, message: Message) {
        for session in self.clients.keys().copied() {
            self.outbox.push(OutMessage {
                dest: Destination::Client(session),
                message: message.clone(),
            });
        }
    }

    // ---------------------------------------------------------------- handshake

    /// Run the hello exchange for a new connection.
    ///
    /// The version gate itself lives in [`space_net::accept_connection`] (it
    /// answers the mismatch before this is ever called); what is checked here is
    /// the shard's own policy. There are two kinds of hello:
    ///
    /// **A fresh join.** A new player always starts on the scene's **spawn
    /// point** ([`space_scene::scene_anchor`] of this shard's map), so it is
    /// welcomed — and its ship spawned there — only by the shard that owns
    /// that point. Every other shard answers with
    /// [`JoinRefusal::Elsewhere`], a [`JoinRedirect`] naming the owner as its
    /// own map has it, and the client dials that shard instead
    /// (`docs/plan-sharding.md` §4). A shard used to put such a player on the
    /// centre of its own region, a place nobody chose. If the map does not say
    /// yet where the owner is reached (it was carved or replaced a moment ago
    /// and the orchestrator's map push is still on its way), the answer is
    /// [`RejectReason::Starting`]: try again shortly.
    ///
    /// **A redirected client.** A client that connects to us while we hold a
    /// staged handoff whose owner token matches the client's hello token is
    /// coming *for that entity*: it is welcomed wherever the spawn point is,
    /// and gets no ship of its own (a placeholder used to sit at the region
    /// centre until the client reported in, and a client that crashed before
    /// reporting left a phantom ship there forever). Such a client is
    /// *awaiting adopt* ([`Client::ship`] is `None`) and gets its ship by
    /// adopting the staged entity in [`Shard::client_status`]; if adoption
    /// never comes within [`crate::config::ADOPT_TIMEOUT_TICKS`] it is treated
    /// as a fresh join after all (`client_timeout_pass`).
    ///
    /// Every client that is admitted — a newcomer and a redirected player's
    /// second connection alike — is queued a [`Message::SceneUpdate`] with the
    /// shard's current scene, so it is the first thing behind the [`Welcome`].
    ///
    /// # Errors
    ///
    /// [`JoinRefusal::Rejected`] with [`RejectReason::Full`] at capacity;
    /// for a fresh join on a shard that does not own the spawn point,
    /// [`JoinRefusal::Elsewhere`] — or `Rejected(Starting)` while the owner's
    /// address is not known.
    pub fn join(&mut self, hello: &Hello) -> Result<Welcome, JoinRefusal> {
        // A player's token is its identity towards the shards for as long as
        // its link lives: it is what a staged ship is matched to its owner
        // by. An empty one is nobody's — and it is also what a transfer that
        // *has* no owner (a body) carries, so it must never be taken for a
        // match. Refused here, before anything is decided on it.
        if hello.player_token.is_empty() {
            return Err(RejectReason::Other("a hello must carry a player token".into()).into());
        }
        if self.clients.len() >= MAX_PLAYERS_PER_SHARD {
            return Err(RejectReason::Full.into());
        }
        // Does this hello match a staged transfer's owner? If so the client is
        // the redirected player and gets no placeholder (see the method docs).
        // Only a **ship** has an owner to come for it.
        let staged_owner = self
            .ledger
            .incoming()
            .values()
            .any(|staged| staged.is_owned_by(&hello.player_token));
        // A fresh join is this shard's only if the spawn point is. (Decided
        // before a session id is spent on it.)
        let spawn = if staged_owner {
            None
        } else {
            match self.free_spawn_point() {
                Some(pos) => Some(pos),
                None => return Err(self.send_elsewhere()),
            }
        };
        let session = self.next_session;
        self.next_session += 1;
        let ship = spawn.map(|pos| self.spawn_ship_for(session, pos));
        let adopt_deadline = if ship.is_none() {
            Some(self.sim.tick().saturating_add(ADOPT_TIMEOUT_TICKS))
        } else {
            None
        };
        self.clients.insert(
            session,
            Client {
                session,
                player_token: hello.player_token.clone(),
                ship,
                // An awaiting-adopt client has no anchor yet, so it is anchored on
                // a sentinel guid this shard can never mint (serials start at 1)
                // and [`Shard::replicate_pass`] skips it until it has a ship.
                replica: ClientReplica::new(ship.unwrap_or(EntityGuid::new(self.config.shard, 0))),
                pending_inputs: Vec::new(),
                redirect_pending: None,
                redirect_deadline: None,
                adopt_deadline,
            },
        );
        // The scene to draw, right behind the welcome. Queued here rather than
        // sent by the front end so it is in the outbox before this session's
        // first frame, whatever tick the join lands on.
        self.send_to(
            Destination::Client(session),
            Message::SceneUpdate(self.scene),
        );
        // And the clock the scene's planets move by.
        self.send_to(
            Destination::Client(session),
            Message::SceneClock(self.sim.scene_clock()),
        );
        Ok(Welcome {
            session,
            shard: self.config.shard,
            // What a player needs of the map: the regions and where to dial
            // each shard — not where the shards reach each other.
            shard_map: self.shard_map.for_players(),
        })
    }

    /// Forget a client and despawn its ship.
    ///
    /// The ship goes with it: an unowned ship would keep drifting forever,
    /// consuming a guid and replication bandwidth for nobody. A transfer that
    /// ship was in goes with it — the player is gone, so the handoff is off, and
    /// telling the peer so is what stops it waiting for a commit that can never
    /// come (guarantee 3's gate is a *connected* client).
    ///
    /// The same holds on the **accepting** side. A client that adopted a staged
    /// entity and then left has a ship that is not in the world yet: it is
    /// either still staged (the commit has not arrived) or committed and waiting
    /// for its install tick. Neither is touched by `despawn_ship`, so both are
    /// dropped here — the staged one with an abort to the offering shard, which
    /// then simply keeps the ship. Left alone, the commit would later install a
    /// ship owned by a session that no longer exists, and nothing would ever
    /// despawn it.
    pub fn leave(&mut self, session: u64) {
        let Some(client) = self.clients.remove(&session) else {
            return;
        };
        let Some(ship) = client.ship else {
            return; // awaiting adopt: nothing was spawned, nothing to clean up
        };
        if self.ledger.activation_of(ship).is_some() {
            self.abort_outgoing(ship, HandoffReason::ClientNotConnected);
        }
        if self
            .ledger
            .incoming()
            .get(&ship)
            .is_some_and(|staged| staged.session == Some(session))
        {
            self.abort_incoming(ship, HandoffReason::ClientNotConnected);
        }
        self.pending_commits
            .retain(|pending| !(pending.guid == ship && pending.session == session));
        // A ship already handed to another shard is a ghost of that shard's
        // ship here: it is that shard's to keep or remove.
        self.despawn_ship(ship);
    }

    /// The session issued so far.
    #[must_use]
    pub fn next_session(&self) -> u64 {
        self.next_session
    }

    /// Spawn a ship owned by `session` at `pos`, minting a guid from this shard's
    /// serial counter.
    ///
    /// The guid is `(this shard, serial)`: minting is the only place a guid is
    /// created here, and the serial never decreases, so a guid this shard mints
    /// can never collide with another shard's (see [`crate::alloc`]).
    pub fn spawn_ship_for(&mut self, session: u64, pos: WorldPos) -> EntityGuid {
        let guid = self.mint_guid();
        let entity = self.sim.spawn_ship(guid, session, ShipState::at_rest(pos));
        self.guids.insert(guid, entity);
        self.interest.upsert(guid, entity, pos);
        self.mark_player(guid, session);
        self.far.rescan = true;
        guid
    }

    /// A ship a client flies (`session` not 0) carries the player model as
    /// its [`space_sim::Visual`], so every client — this shard's, and a
    /// neighbour's that sees it as a ghost with no owner it knows — draws a
    /// character for it. A ship nobody flies has no visual (a ship model).
    fn mark_player(&mut self, guid: EntityGuid, session: u64) {
        if session != 0 {
            self.sim
                .set_visual(guid, space_sim::Visual(space_scene::visual::PLAYER));
            self.guids.insert(
                guid,
                self.sim.entity_of(guid).expect("the ship was just spawned"),
            );
        }
    }

    /// A fresh guid from this shard's serial counter (see
    /// [`Shard::spawn_ship_for`]).
    fn mint_guid(&mut self) -> EntityGuid {
        let guid = EntityGuid::new(self.config.shard, self.next_serial);
        self.next_serial += 1;
        guid
    }

    /// The shard a new player starts on, as this shard's map has it: the
    /// owner of the scene's spawn point ([`space_scene::scene_anchor`]).
    /// `None` for a map with no region at all.
    #[must_use]
    pub fn spawn_owner(&self) -> Option<ShardId> {
        let anchor = space_scene::scene_anchor(&self.shard_map)?;
        self.shard_map.lookup(&anchor)
    }

    /// The answer to a fresh join this shard does not take: the redirect to
    /// the spawn point's owner, or — while this shard's map does not say
    /// where players reach that owner — "starting, try again"
    /// ([`Shard::join`]).
    fn send_elsewhere(&self) -> JoinRefusal {
        let known = self.spawn_owner().filter(|owner| {
            *owner != self.config.shard && self.shard_map.client_endpoint_of(*owner).is_some()
        });
        match known {
            Some(shard) => JoinRefusal::Elsewhere(JoinRedirect {
                shard,
                shard_map: self.shard_map.for_players(),
            }),
            None => RejectReason::Starting.into(),
        }
    }

    /// Despawn `guid` — a ship or a body this shard simulates — and drop it
    /// from every local index. A ghost is left alone: it is the copy of
    /// another shard's entity, removed only by that shard's word.
    pub fn despawn_ship(&mut self, guid: EntityGuid) {
        if self.sim.is_ghost(guid) {
            return;
        }
        if self.guids.entity_of(guid).is_some() || self.sim.entity_of(guid).is_some() {
            self.guids.remove(guid);
            self.interest.remove(guid);
            self.sim.despawn(guid);
            self.refused.remove(&guid);
            self.log.forget(guid);
        }
    }

    // ------------------------------------------------------------------ inputs

    /// File a client's input batch.
    ///
    /// The batch is *not* applied here — [`SimWorld::apply_inputs`] buffers by
    /// each input's own tick and [`Shard::tick`] drains the buffer when it
    /// reaches that tick. What this function does is the trust check: every input
    /// is rewritten to name the guid this session owns, so a client cannot drive
    /// an entity it does not own.
    pub fn apply_input(&mut self, session: u64, batch: &InputTick) {
        let Some(client) = self.clients.get_mut(&session) else {
            return;
        };
        // The client's view of what it has applied settles the frames we are
        // still holding for it: the acknowledged one arrived, an older one that
        // was never acknowledged is presumed lost and owed again (see
        // [`ClientReplica::acknowledge`]). **Clamped to our own tick**: a client
        // cannot have applied a frame that has not been built, and an
        // acknowledgement from the future would write off every frame still in
        // flight.
        let ack = batch.ack_tick.min(self.sim.tick());
        client.replica.acknowledge(ack);
        // Rewrite every input to this session's ship. This is the `entity_of`
        // trust boundary: the guid that reaches the simulation is ours, whatever
        // the client sent.
        //
        // An awaiting-adopt client has no ship yet — the old shard is still
        // simulating its entity. If a ship is **staged for it** (the transfer
        // whose owner token is this client's: the test `client_status` makes
        // too), its inputs are kept for that ship like those of a client that
        // has reported in (guarantee 5): a client says its inputs for the
        // ticks to come the moment it has connected, and its datagrams are
        // not ordered with the status it sends on the stream. Anybody else's
        // are dropped.
        let Some(ship) = client.ship else {
            let now = self.sim.tick();
            let staged = self
                .ledger
                .incoming()
                .iter()
                .find(|(_, staged)| staged.is_owned_by(&client.player_token))
                .map(|(guid, _)| *guid);
            if let Some(ship) = staged {
                let kept = self.staged_inputs.entry(ship).or_default();
                keep_staged(kept, batch.inputs.iter().copied(), ship, now);
            }
            return;
        };
        // Bounded: only one input per tick can ever be applied to the one ship
        // this session drives, so anything past a few windows is noise (see
        // [`MAX_PENDING_INPUTS`]).
        let room = MAX_PENDING_INPUTS.saturating_sub(client.pending_inputs.len());
        for input in batch.inputs.iter().take(room) {
            let mut input = *input;
            input.entity = ship;
            client.pending_inputs.push(input);
        }
    }

    /// A client reported its connection status towards `shard`.
    pub fn client_status(&mut self, session: u64, status: NeighbourStatus) {
        // The client telling us it reached the accepting shard is the gate for the
        // commit (guarantee 3). We mirror it to the offering shard, per
        // `NeighbourStatus`'s contract.
        //
        // Either way the status is a claim about a staged entity, and a claim is
        // only honoured from the player that entity belongs to — **the client
        // whose hello token is the token the offer carried the owner under**. A
        // guid is `(shard, serial)` with serials starting at 1, so it is trivially
        // guessable; the token is what makes the claim non-forgeable. Without the
        // match, a second client could name somebody else's staged ship and be
        // handed its `Owner` stamp and its inputs (`connected = true`), or cancel
        // somebody else's handoff (`connected = false`).
        let claim_is_authorized =
            self.ledger
                .incoming()
                .get(&status.entity)
                .is_some_and(|staged| {
                    // (An ownerless transfer — a body, a stray ship — carries
                    // an empty owner token: nobody's, whatever a client's
                    // own token is. See [`Incoming::is_owned_by`].)
                    self.clients
                        .get(&session)
                        .is_some_and(|client| staged.is_owned_by(&client.player_token))
                });
        if status.connected {
            if claim_is_authorized {
                let placeholder = self
                    .clients
                    .get_mut(&session)
                    .filter(|client| client.ship != Some(status.entity))
                    .map(|client| client.ship);
                if let Some(placeholder) = placeholder {
                    if let Some(placeholder) = placeholder {
                        self.despawn_ship(placeholder);
                    }
                    if let Some(client) = self.clients.get_mut(&session) {
                        client.ship = Some(status.entity);
                        client.replica = ClientReplica::new(status.entity);
                        client.redirect_deadline = None;
                        client.adopt_deadline = None;
                    }
                }
                // Binding the staged session is what opens the offering shard's
                // commit gate, so it is part of the authorized claim: a client
                // whose hello token is not the one the offer carried must neither
                // adopt the entity nor open the gate for it.
                if let Some(staged) = self.ledger.mark_session(status.entity, session) {
                    // Mirror to the *offering* shard (the one waiting on this gate),
                    // not to the shard the status names — which is us.
                    let from = staged.from;
                    self.send_to(Destination::Shard(from), Message::NeighbourStatus(status));
                }
            }
            if let Some(client) = self.clients.get_mut(&session) {
                if client.redirect_pending == Some(status.shard) {
                    client.redirect_pending = None;
                    client.redirect_deadline = None;
                }
            }
        } else if claim_is_authorized {
            // The owner could not reach us: give the staged entity back. The
            // abort names *this* shard as the one that would have taken it — not
            // whatever shard id the client put in its status.
            self.abort_incoming(status.entity, HandoffReason::ClientNotConnected);
        }
    }

    // ------------------------------------------------------------- shard-to-shard

    /// A message arrived from another shard.
    ///
    /// # Who a peer is
    ///
    /// A peer connection announces itself with the hello token
    /// <code>"shard:&lt;id&gt;"</code> ([`crate::net::peer_shard_id`]), which is a
    /// string any client could type. Two checks stand behind it:
    ///
    /// * **The cluster's peer token**, at the front end, before a single peer
    ///   message is read: an orchestrated shard holds the token of its
    ///   `ShardWelcome`, presents it in every hello to another shard and
    ///   refuses a peer hello without it ([`crate::net::ShardNet::set_peer_token`]).
    ///   A shard that holds none — standalone shards wired by hand — skips
    ///   this one.
    /// * **Map membership**, here: the claimed [`ShardId`] must be one of the
    ///   ids in **this shard's own shard map**; a peer outside it is dropped. A
    ///   commit is additionally required to come from the shard that actually
    ///   staged the transfer ([`Ledger::take_incoming_from`]).
    ///
    /// The token says "a shard of this cluster", not *which* one: it is one
    /// secret for all of them, the transport is QUIC with a self-signed
    /// certificate and no pinning, and a shard that holds the token can still
    /// claim a neighbour's id. See `docs/security-notes.md`.
    ///
    /// Every handoff message is then epoch-checked (guarantee 7): a message under a
    /// stale epoch is answered with [`Message::StaleEpoch`] or abandons the
    /// transfer with [`HandoffReason::EpochChanged`], rather than being applied to
    /// a map that no longer describes the world.
    pub fn peer_message(&mut self, from: ShardId, message: Message) {
        if !self.knows_shard(from) {
            // Not a shard in our map: not a neighbour, whatever its token claimed.
            return;
        }
        match message {
            Message::HandoffOffer(offer) => self.on_offer(offer),
            Message::HandoffAccept(accept) => self.on_accept(accept),
            Message::HandoffCommit(commit) => self.on_commit(commit),
            Message::HandoffAbort(abort) => self.on_abort(abort),
            Message::NeighbourStatus(status) => self.on_neighbour_status(from, status),
            Message::GhostUpdate(update) => self.on_ghost_update(from, update),
            Message::FarInterest(interest) => self.on_far_interest(from, interest),
            Message::FarUpdate(update) => self.on_far_update(from, update),
            Message::StaleEpoch(stale) => {
                // Our map is behind: any transfer we have in flight is invalid.
                let stale_transfers: Vec<(EntityGuid, ShardId)> = self
                    .ledger
                    .outgoing()
                    .iter()
                    .filter(|_| stale.current > self.epoch())
                    .map(|(guid, entry)| (*guid, entry.to))
                    .collect();
                for (entity, to) in stale_transfers {
                    self.abort_outgoing(entity, HandoffReason::EpochChanged);
                    // The peer will keep refusing until our map catches up, and
                    // a new map clears this pause: there is nothing to gain from
                    // offering again before then.
                    self.pause_reoffer(entity, to);
                }
            }
            other => {
                debug_assert!(false, "unexpected shard message from {from}: {other:?}");
            }
        }
    }

    /// Whether `shard` is one of the ids this shard's map names — or named
    /// until a moment ago: a shard whose region was merged away is still
    /// heard while it hands over what it simulates
    /// ([`FORMER_SHARD_GRACE_TICKS`]).
    #[must_use]
    pub fn knows_shard(&self, shard: ShardId) -> bool {
        shard != ShardId::NONE
            && (self
                .shard_map
                .regions
                .iter()
                .any(|region| region.shard == shard)
                || self
                    .former
                    .get(&shard)
                    .is_some_and(|(_, until)| self.sim.tick() <= *until))
    }

    fn on_neighbour_status(&mut self, from: ShardId, status: NeighbourStatus) {
        if status.connected {
            self.ledger.mark_client_connected(status.entity, from);
        } else if self
            .ledger
            .outgoing()
            .get(&status.entity)
            .is_some_and(|entry| entry.to == from)
        {
            self.abort_outgoing(status.entity, HandoffReason::ClientNotConnected);
        }
    }

    fn on_offer(&mut self, offer: space_protocol::HandoffOffer) {
        if offer.epoch != self.epoch() {
            self.send_to(
                Destination::Shard(offer.from),
                Message::StaleEpoch(StaleEpoch {
                    seen: offer.epoch,
                    current: self.epoch(),
                }),
            );
            return;
        }
        let refuse = |shard: &mut Self, reason: HandoffReason| {
            let abort = space_protocol::HandoffAbort {
                entity: offer.entity,
                from: shard.config.shard,
                to: offer.to,
                reason,
            };
            shard.send_to(Destination::Shard(offer.from), Message::HandoffAbort(abort));
        };
        let Some((kind, lead)) = TransferKind::of(&offer.state) else {
            refuse(self, HandoffReason::Other("unknown transfer kind".into()));
            return;
        };
        // Capacity first: a shard must never be pushed past the player cap by
        // incoming handoffs. Answering `TargetFull` lets the offering shard keep
        // the ship rather than stage it into a shard that cannot take it. (A
        // body takes no player slot.)
        if kind == TransferKind::Ship && self.clients.len() >= MAX_PLAYERS_PER_SHARD {
            let abort = space_protocol::HandoffAbort {
                entity: offer.entity,
                from: self.config.shard,
                to: offer.to,
                reason: HandoffReason::TargetFull,
            };
            self.send_to(Destination::Shard(offer.from), Message::HandoffAbort(abort));
            return;
        }
        // Three guarantee-6 failures: we cannot take an entity we are ourselves
        // moving, we cannot take one we already hold (a duplicate offer after a
        // lost accept), and we cannot take one whose commit has already arrived
        // and is waiting for its install tick — it is in neither the ledger nor
        // the world for those few ticks, and staging it a second time would end
        // in two installs of one guid.
        if self.ledger.is_busy(offer.entity)
            || self.sim.ship_state(offer.entity).is_some()
            || self
                .pending_commits
                .iter()
                .any(|pending| pending.guid == offer.entity)
        {
            let abort = space_protocol::HandoffAbort {
                entity: offer.entity,
                from: self.config.shard,
                to: offer.to,
                reason: HandoffReason::Other("entity is not available here".into()),
            };
            self.send_to(Destination::Shard(offer.from), Message::HandoffAbort(abort));
            return;
        }
        // The state must be one we can install. Refusing here, while the offering
        // shard still simulates the entity, is the only point at which an
        // undecodable state costs nothing: once the commit is out the sender has
        // let go, and there is no good answer left.
        if decode_ship_state(&offer.state).is_none() {
            let abort = space_protocol::HandoffAbort {
                entity: offer.entity,
                from: self.config.shard,
                to: offer.to,
                reason: HandoffReason::Other("the offered state is not a decodable ship".into()),
            };
            self.send_to(Destination::Shard(offer.from), Message::HandoffAbort(abort));
            return;
        }
        // A body must import here as it is (shapes included), and belong to
        // the scene this shard runs; and two shards offering each other
        // members of one island at once settle it by shard id.
        if kind == TransferKind::Body {
            if let Err(why) = self.check_body_offer(&offer) {
                refuse(self, HandoffReason::Other(why));
                return;
            }
        }
        // (Bodies only: a player's ship goes where it is, whatever crosses
        // it — see [`crate::border`], "Players' ships".)
        if kind == TransferKind::Body && self.is_crossing_offer(&offer) {
            refuse(
                self,
                HandoffReason::Other(
                    "crossing island transfer; the lower shard id takes it".into(),
                ),
            );
            return;
        }
        let accept = Ledger::accept_for(&offer);
        // The owner's token rides inside the offer's state as a private
        // component (see [`OWNER_TOKEN_COMPONENT`]): Phase 2 has no signed handoff
        // ticket and `HandoffOffer` has no token field (space-protocol is frozen
        // for this phase), so the token is smuggled in the state payload the
        // offering shard already controls. It is the best identity available and
        // is not authenticated: a signed handoff ticket is deferred.
        let owner_token = offer_token_of(&offer.state);
        self.ledger.accept_sent_as(
            &accept,
            *offer.state.clone(),
            owner_token,
            self.sim.tick(),
            kind,
            lead,
        );
        self.send_to(
            Destination::Shard(offer.from),
            Message::HandoffAccept(accept),
        );
    }

    fn on_accept(&mut self, accept: space_protocol::HandoffAccept) {
        if accept.epoch != self.epoch() {
            self.abort_outgoing(accept.entity, HandoffReason::EpochChanged);
            return;
        }
        // The first accept of a ship's transfer is what redirects its
        // player (see [`Shard::redirect_owner`]); a repeated one is not.
        let first_ship_accept = self.ledger.outgoing().get(&accept.entity).is_some_and(|e| {
            !e.accepted && e.kind == TransferKind::Ship && self.session_of(accept.entity).is_some()
        });
        if self.ledger.mark_accepted(&accept) && first_ship_accept {
            self.redirect_owner(accept.entity, accept.to, accept.activation_tick);
        }
    }

    fn on_commit(&mut self, commit: space_protocol::HandoffCommit) {
        if commit.epoch != self.epoch() {
            self.abort_incoming(commit.entity, HandoffReason::EpochChanged);
            return;
        }
        // The commit must come from the shard that actually staged the transfer.
        // Without this, a peer that guessed a valid neighbour id (see
        // [`Shard::peer_message`]) could commit an entity another shard staged and
        // spawn an attacker-chosen state under `Owner(0)`.
        let Some(staged) = self.ledger.take_incoming_from(commit.entity, commit.from) else {
            return; // aborted, never staged, or staged by a different shard
        };
        // The commit's final state replaces the staged one (guarantee 4). A
        // commit whose state does not decode falls back to the state the offer
        // staged — a few ticks old, but the entity's own — and says so. It is
        // never replaced by a default: a ship silently re-created at
        // `ShipState::IDENTITY` is a ship teleported to the world origin. The
        // staged state was checked when the offer was accepted, so the refusal
        // below is unreachable through `on_offer`; it is here so that no path can
        // install a state nobody sent.
        let state = match decode_ship_state(&commit.state) {
            Some(state) => state,
            None => match decode_ship_state(&staged.state) {
                Some(state) => {
                    eprintln!(
                        "space-server: the commit for {} from {} carried an undecodable \
                         state; installing the state its offer staged",
                        commit.entity, staged.from
                    );
                    state
                }
                None => {
                    eprintln!(
                        "space-server: neither the commit nor the offer for {} from {} \
                         carried a decodable state; the transfer is refused",
                        commit.entity, staged.from
                    );
                    self.stats.aborted += 1;
                    let abort = space_protocol::HandoffAbort {
                        entity: commit.entity,
                        from: staged.from,
                        to: self.config.shard,
                        reason: HandoffReason::Other(
                            "the committed state is not a decodable ship".into(),
                        ),
                    };
                    self.send_to(
                        Destination::Shard(staged.from),
                        Message::HandoffAbort(abort),
                    );
                    return;
                }
            },
        };
        let session = staged.session.unwrap_or(0);
        // A body arrives as its whole component list: the commit's, or the
        // offer's if the commit's does not import.
        let components = if staged.kind == TransferKind::Body {
            match self.body_components_of(&commit.state, &staged.state) {
                Some(list) => list,
                None => {
                    self.stats.aborted += 1;
                    let abort = space_protocol::HandoffAbort {
                        entity: commit.entity,
                        from: staged.from,
                        to: self.config.shard,
                        reason: HandoffReason::Other("the committed body does not import".into()),
                    };
                    self.send_to(
                        Destination::Shard(staged.from),
                        Message::HandoffAbort(abort),
                    );
                    return;
                }
            }
        } else {
            Vec::new()
        };
        // The entity is **not** spawned here. The commit's state is the state as
        // of the offering shard's `activation_tick`, and a commit can be delivered
        // at any point around that tick; spawning on arrival would step the entity
        // during the tick the commit already describes (a double step, breaking
        // guarantee 2 and guarantee 4 at once). It goes on the pending list and
        // [`Shard::tick`] installs it *before* its own step, so the first tick it
        // is simulated on is strictly the first tick after the state it arrived
        // with.
        //
        // The install gate is the tick the state describes, on **our** clock
        // (`state_tick`): the sender's scene time of it, read off our own
        // scene clock. On one cluster clock that is the sender's commit tick
        // itself; between shards that count differently but both follow the
        // wall clock it is the tick of ours nearest to that instant. Before
        // it we wait (stepping the state earlier would double-step it), past
        // it the install brings the entity forward. That needs both clocks
        // to be shared ones. Where one of them counts from its own zero
        // (shards stepped by hand), or the instant lies further from our
        // present than any delivery explains, the gate is
        // `local_activation_tick`, derived when we accepted — never the
        // commit's `activation_tick`, which is the offering shard's clock.
        let scene_time = crate::handoff::scene_time_of(&commit.state);
        let state_tick = crate::handoff::shared_scene_time_of(&commit.state)
            .filter(|_| self.scene_clock_set)
            .and_then(|sent| self.local_tick_of(sent));
        self.pending_commits.push(PendingCommit {
            guid: commit.entity,
            session,
            state,
            activation_tick: commit.activation_tick,
            install_at: state_tick.unwrap_or(staged.local_activation_tick),
            state_tick,
            fast: decode_ship_controls(&commit.state).is_some_and(|c| c.fast),
            kind: staged.kind,
            components,
            from: staged.from,
            scene_time: scene_time.or_else(|| crate::handoff::scene_time_of(&staged.state)),
        });
    }

    /// The tick of this shard's clock that is nearest to scene time `sent_us`
    /// — if it lies within [`MAX_CATCH_UP`] ticks of the present, either way.
    /// `None` means the instant is nowhere near our present: the shard that
    /// named it does not count scene time as we do (see `on_commit`).
    fn local_tick_of(&self, sent_us: i64) -> Option<SimTick> {
        let half_tick = 500_000 / SceneClock::TICKS_PER_SECOND;
        let tick = self
            .sim
            .scene_clock()
            .tick_at(sent_us.saturating_add(half_tick));
        let now = self.sim.tick();
        (tick.saturating_add(MAX_CATCH_UP) >= now && tick <= now.saturating_add(MAX_CATCH_UP))
            .then_some(tick)
    }

    fn on_abort(&mut self, abort: space_protocol::HandoffAbort) {
        // Either way an abort means "nothing happened": drop our half and keep
        // simulating whatever we have (guarantee 6).
        if let Some(outgoing) = self.ledger.take_outgoing(abort.entity) {
            self.stats.aborted += 1;
            self.note_abort(abort.entity, outgoing.to, true, false, abort.reason.clone());
            // The peer turned our offer down. The ship is still on its side of
            // the boundary, so the handoff pass would offer it again on the very
            // next tick, be refused again, and so on at tick rate.
            self.pause_reoffer(abort.entity, outgoing.to);
            // An island moves together or not at all (guarantee 6) — but a
            // player's ship is not held back by the bodies travelling with
            // it: when one of those is refused, the bodies stay and the ship
            // goes on.
            for member in self.group_to_abort_with(outgoing.kind, outgoing.group) {
                self.abort_one_outgoing(
                    member,
                    HandoffReason::Other("a group member was refused".into()),
                );
                self.pause_reoffer(member, outgoing.to);
            }
        } else if let Some(staged) = self.ledger.take_incoming(abort.entity) {
            self.stats.aborted += 1;
            self.note_abort(abort.entity, staged.from, false, false, abort.reason);
        }
    }

    /// Remember that a transfer was called off, and say so in the log — a
    /// few lines per ten seconds, the rest counted (a body that keeps being
    /// refused must not fill the log).
    fn note_abort(
        &mut self,
        entity: EntityGuid,
        peer: ShardId,
        offering: bool,
        by_us: bool,
        reason: HandoffReason,
    ) {
        let tick = self.sim.tick();
        let player = self.session_of(entity).is_some();
        if tick.saturating_sub(self.abort_lines.0) >= ABORT_LINE_WINDOW {
            self.abort_lines.0 = tick;
            self.abort_lines.1 = 0;
        }
        if self.abort_lines.1 < ABORT_LINES {
            self.abort_lines.1 += 1;
            let held_back = std::mem::take(&mut self.abort_lines.2);
            let more = if held_back == 0 {
                String::new()
            } else {
                format!(" ({held_back} more like it were not written)")
            };
            eprintln!(
                "space-server: {} at tick {tick}: the handoff of {}{entity} {} {peer} was called off {}: {reason:?}{more}",
                self.config.shard,
                if player { "the player's ship " } else { "" },
                if offering { "to" } else { "from" },
                if by_us { "here" } else { "by it" },
            );
        } else {
            self.abort_lines.2 += 1;
        }
        self.abort_notes.push_back(AbortNote {
            tick,
            entity,
            peer,
            offering,
            player,
            by_us,
            reason,
        });
        while self.abort_notes.len() > ABORT_NOTES {
            self.abort_notes.pop_front();
        }
    }

    /// The transfers called off lately (the last 64), oldest first: which
    /// entity, with which shard, by whom and why.
    #[must_use]
    pub fn recent_aborts(&self) -> Vec<AbortNote> {
        self.abort_notes.iter().cloned().collect()
    }

    /// Do not offer `entity` to `to` again for
    /// [`REOFFER_BACKOFF_TICKS`](crate::config::REOFFER_BACKOFF_TICKS).
    fn pause_reoffer(&mut self, entity: EntityGuid, to: ShardId) {
        let until = self.sim.tick().saturating_add(REOFFER_BACKOFF_TICKS);
        self.refused.insert(entity, (to, until));
    }

    fn abort_outgoing(&mut self, entity: EntityGuid, reason: HandoffReason) {
        let (kind, group) = self
            .ledger
            .outgoing()
            .get(&entity)
            .map_or((TransferKind::Body, 0), |e| (e.kind, e.group));
        self.abort_one_outgoing(entity, reason.clone());
        // The rest of its island goes back too (guarantee 6).
        for member in self.group_to_abort_with(kind, group) {
            self.abort_one_outgoing(member, reason.clone());
        }
    }

    /// Call the outgoing transfer of `entity` off, and nothing else.
    fn abort_one_outgoing(&mut self, entity: EntityGuid, reason: HandoffReason) {
        if let Some(abort) = self
            .ledger
            .abort_outgoing(entity, self.config.shard, reason.clone())
        {
            self.stats.aborted += 1;
            self.note_abort(entity, abort.to, true, true, reason);
            self.send_to(Destination::Shard(abort.to), Message::HandoffAbort(abort));
        }
    }

    /// The members of outgoing group `group` that are called off with a
    /// member of `kind`: all of them when a player's ship is (the bodies
    /// were travelling with it), the bodies only when a body is — a
    /// player's ship is handed over for where it is, whatever becomes of
    /// what travels with it (see [`crate::border`], "Players' ships").
    fn group_to_abort_with(&self, kind: TransferKind, group: u64) -> Vec<EntityGuid> {
        self.ledger
            .group_members(group)
            .into_iter()
            .filter(|member| {
                kind == TransferKind::Ship
                    || self
                        .ledger
                        .outgoing()
                        .get(member)
                        .is_some_and(|e| e.kind == TransferKind::Body)
            })
            .collect()
    }

    fn abort_incoming(&mut self, entity: EntityGuid, reason: HandoffReason) {
        if let Some(abort) = self
            .ledger
            .abort_incoming(entity, self.config.shard, reason.clone())
        {
            self.stats.aborted += 1;
            self.note_abort(entity, abort.from, false, true, reason);
            self.send_to(Destination::Shard(abort.from), Message::HandoffAbort(abort));
        }
    }
    // ----------------------------------------------------------------- the loop

    /// One simulation tick: apply inputs, step, refresh interest, run the handoff
    /// state machine, then build one frame per client.
    ///
    /// The order is the one the module docs promise and each step depends on the
    /// one before it: the interest grid must hold the *new* positions before a
    /// boundary test or a visibility query runs, the handoff pass must run before
    /// replication so a commit's entity is in the world when frames are built, and
    /// replication must run last so a frame describes the finished tick.
    pub fn tick(&mut self) -> SimTick {
        let began = std::time::Instant::now();
        let tick = self.tick_inner();
        // What the tick cost, for the load report; and the report itself,
        // once a second, for a shard that has an orchestrator to tell.
        self.load
            .note(u64::try_from(began.elapsed().as_micros()).unwrap_or(u64::MAX));
        if let Some(last) = self.reports_load {
            if tick.saturating_sub(last) >= crate::load::REPORT_INTERVAL_TICKS {
                self.reports_load = Some(tick);
                let report = self.load_report();
                self.send_to(Destination::Orchestrator, Message::LoadReport(report));
            }
        }
        tick
    }

    /// The tick itself ([`Shard::tick`] times it).
    fn tick_inner(&mut self) -> SimTick {
        // 1. Inputs: filed for the ship this shard simulates, or — for a
        // ship on its way here (its player sends to both shards around the
        // switch) — kept until it is installed. Anything else is dropped: a
        // ship that left is the other shard's to drive, and its ghost here
        // takes no input.
        let now = self.sim.tick();
        for client in self.clients.values_mut() {
            if client.pending_inputs.is_empty() {
                continue;
            }
            let inputs = std::mem::take(&mut client.pending_inputs);
            let Some(ship) = client.ship else {
                continue;
            };
            if self.sim.ship_state(ship).is_some() {
                self.sim.apply_inputs(now.saturating_add(1), &inputs);
            } else if self.ledger.has_incoming(ship)
                || self.pending_commits.iter().any(|p| p.guid == ship)
            {
                let kept = self.staged_inputs.entry(ship).or_default();
                keep_staged(kept, inputs, ship, now);
            }
        }
        // (A transfer that was called off leaves nothing behind.)
        if !self.staged_inputs.is_empty() {
            let (ledger, pending) = (&self.ledger, &self.pending_commits);
            self.staged_inputs.retain(|guid, _| {
                ledger.has_incoming(*guid) || pending.iter().any(|p| p.guid == *guid)
            });
        }
        // 1.5 The scene's physics, if the map moved its anchor since it was
        // built (an idempotent check otherwise).
        self.ensure_scene();
        // 2. Install any commit whose **local** activation tick we have reached,
        // then step.
        //
        // The install goes *before* the step because the commit's state is the
        // state *as of* the offering shard's activation tick: installing it while
        // our tick is at its local install tick means the step below produces the
        // next state the entity should have. Installing after the step would leave
        // the entity unstepped for one tick (a visible freeze), and installing
        // before it would integrate a state that is already newer than the tick it
        // describes. The gate is `install_at` (our clock), never the commit's
        // `activation_tick` (the offering shard's clock — see [`PendingCommit`]).
        self.install_pending_commits(self.sim.tick());
        // 3. Step.
        let tick = self.sim.step();
        // 3.5 Ghosts move on one tick along their velocity: the step did not
        // integrate them, and what they show now is their owner's state of
        // a tick ago (see [`crate::border`]).
        self.reckon_ghosts();

        // 4. Refresh the guid index and the interest grid from the new
        // positions: ships, bodies and ghosts alike.
        self.refresh_index();

        // 4.5 Client-side timeouts: a redirect that was never confirmed, and an
        // awaiting-adopt client whose staged entity never materialised.
        self.client_timeout_pass(tick);

        // 5. Islands (who belongs on which shard), then handoffs: start,
        // commit, time out; then the border ghosts for the neighbours.
        self.border_pass();
        self.handoff_pass(tick);
        self.ghost_pass(tick);
        // ... and, beyond the ghosts' reach, what the other shards' players
        // can see of ours (`far.rs`).
        self.far_pass(tick);

        // 5.5 The scene, again, for whoever missed it (see
        // [`SCENE_REANNOUNCE_TICKS`]). Before the frames, so a tick's last
        // message to a client is still its frame.
        if tick.is_multiple_of(SCENE_REANNOUNCE_TICKS) {
            self.send_to_all(Message::SceneUpdate(self.scene));
            self.send_to_all(Message::SceneClock(self.sim.scene_clock()));
        }

        // 6. Replication.
        self.replicate_pass();

        // 7. Ships handed off this tick leave the world. Deferring this past the
        // frames (rather than despawning inside the handoff pass) is what keeps the
        // client's frame clock continuous across a handoff: the accepting shard
        // cannot install the ship before the commit has been delivered, i.e. not
        // before the next tick, so if the offering shard sent nothing for the
        // commit tick the client would see a tick with no frame at all.
        // An entity handed away stays visible here as a ghost of the
        // receiver's copy (the receiver updates or removes it from then on).
        for (guid, to) in std::mem::take(&mut self.departing) {
            self.depart(guid, to);
        }
        self.report_census(tick);

        tick
    }

    /// Run `steps` ticks.
    pub fn tick_n(&mut self, steps: SimTick) -> SimTick {
        for _ in 0..steps {
            self.tick();
        }
        self.sim.tick()
    }

    /// Install every commit whose install tick has been reached.
    ///
    /// Called from [`Shard::tick`] immediately before the step, so the entity
    /// joins the world with the state of the tick the world is at and the
    /// step that follows produces the next one — never stepped twice for one
    /// tick (guarantee 2) and never rewound to the offer's older state
    /// (guarantee 4).
    ///
    /// The gate is `install_at`: the tick the committed state describes
    /// (`state_tick`), so a commit that arrives before this shard has got
    /// there waits, and one that arrives after it is installed at once and
    /// **brought forward** by the ticks in between — a ship with the flight
    /// model and the inputs its player sent here for those ticks
    /// ([`catch_up_ship`]), a body along its velocity. The client's
    /// input-ack floor is then this tick: every input up to it has been
    /// applied, by the old shard or by the catch-up (guarantee 5). Between
    /// shards whose clocks do not agree the gate is the accept tick plus the
    /// lead and the floor the commit's `activation_tick`, as it always was.
    ///
    /// **The catch-up is bounded here, not only when the commit arrives**
    /// ([`PendingCommit::catch_up_at`]): a commit that was within
    /// [`MAX_CATCH_UP`] ticks of the present when it was read may be
    /// installed much later — this shard stalled in between and then jumped
    /// ([`Shard::skip_to`]) — and bringing an entity forward by seconds of
    /// contact-free flight would put it through walls. Past the bound the
    /// state is installed as it is: the entity stood still for the stall,
    /// like everything else on this shard.
    fn install_pending_commits(&mut self, tick: SimTick) {
        let due: Vec<PendingCommit> = self
            .pending_commits
            .extract_if(.., |p| p.install_at <= tick)
            .collect();
        for pending in due {
            if pending.kind == TransferKind::Body {
                self.install_body(pending);
                continue;
            }
            // Never spawn a guid that is already alive here. `on_offer` refuses
            // an entity whose commit is pending, so two installs of one guid
            // should not be reachable — but `SimWorld::spawn_ship` *panics* on a
            // duplicate, a panic here kills the tick loop, and a shard whose tick
            // loop is dead keeps accepting players into a frozen world. A guard
            // is cheaper than that failure. (A ghost of it is replaced.)
            if self.sim.ship_state(pending.guid).is_some() {
                eprintln!(
                    "space-server: dropped a second install of {}: it is already alive here",
                    pending.guid
                );
                continue;
            }
            // The player this ship was committed for has left in the meantime
            // (see [`Shard::leave`]): installing it would leave a ship nobody
            // owns and nothing ever despawns.
            if pending.session != 0 && !self.clients.contains_key(&pending.session) {
                continue;
            }
            self.remove_ghost(pending.guid);
            let inputs = self.staged_inputs.remove(&pending.guid).unwrap_or_default();
            // What the ship has done since its state was taken, if the commit
            // came late: flown on, under the inputs its player sent us.
            let late = pending.catch_up_at(tick);
            let (mut state, fast) = catch_up_ship(
                pending.state,
                pending.fast,
                &self.sim.params(),
                &inputs,
                tick - late,
                late,
            );
            self.stats.caught_up += late;
            // (The record of what was applied: the catch-up's ticks too.)
            if late > 0 {
                for at in inputs.range(tick - late + 1..=tick).map(|(at, _)| *at) {
                    self.sim.note_applied(at, pending.guid);
                }
            }
            // Where it was relative to the planet it is near, by our clock.
            if let Some(sent) = pending.scene_time {
                let c = self.sim.carry(state.pos, sent, self.sim.scene_time());
                if c != [0.0; 3] {
                    state.pos = space_world::WorldPos::from_f64_metres_around(state.pos, c);
                }
            }
            let pending = PendingCommit { state, ..pending };
            let entity = self
                .sim
                .spawn_ship(pending.guid, pending.session, pending.state);
            if fast {
                if let Some(controls) = self.sim.world_mut().get_mut::<ShipControls>(entity) {
                    controls.fast = true;
                }
            }
            self.guids.insert(pending.guid, entity);
            self.interest
                .upsert(pending.guid, entity, pending.state.pos);
            self.mark_player(pending.guid, pending.session);
            self.far.claimed_by_the_world(pending.guid);
            self.far.rescan = true;
            self.stats.received += 1;
            self.note_sender_holds_ghost(pending.from, pending.guid);
            // The inputs for the ticks to come are the simulation's now.
            let ahead: Vec<PlayerInput> = inputs.range(tick + 1..).map(|(_, i)| *i).collect();
            if !ahead.is_empty() {
                self.sim.apply_inputs(tick, &ahead);
            }
            // A client driving this ship must not be sent a delta against a tick
            // it never saw, and its inputs up to this tick were already applied:
            // by the old shard up to the state's tick, by the catch-up since
            // (guarantee 5). (Clocks that do not agree: up to the old shard's
            // activation tick, in the old shard's clock.)
            let floor = if pending.state_tick.is_some() {
                tick
            } else {
                pending.activation_tick
            };
            for client in self.clients.values_mut() {
                if client.ship == Some(pending.guid) {
                    client.replica.input_ack_floor = floor;
                    client.replica.ack_tick = tick;
                }
            }
        }
    }

    /// Clear a redirect that was never confirmed, and settle an awaiting-adopt
    /// client whose adopt budget has run out: a ship on the spawn point if
    /// this shard owns it, otherwise a [`JoinRedirect`] to the shard that
    /// does — as for any fresh join ([`Shard::join`]).
    ///
    /// Both are bounded waits: a redirect that never gets a `NeighbourStatus` must
    /// not hold the client half-redirected forever, and a client we withheld a
    /// placeholder for (because it matched a staged transfer's owner) must not be
    /// left with no ship and no word at all if adoption never comes.
    fn client_timeout_pass(&mut self, tick: SimTick) {
        let sessions: Vec<u64> = self.clients.keys().copied().collect();
        for session in sessions {
            let Some(client) = self.clients.get_mut(&session) else {
                continue;
            };
            if client
                .redirect_deadline
                .is_some_and(|deadline| tick >= deadline)
            {
                client.redirect_pending = None;
                client.redirect_deadline = None;
            }
        }
        // An awaiting-adopt client whose deadline passed becomes an ordinary
        // newcomer, and is treated as `join` treats one: on the shard that
        // owns the spawn point it gets a ship there; on any other it is sent
        // to that shard, and is no client of this one any more. (It used to
        // get a ship at this shard's region centre.)
        let overdue: Vec<u64> = self
            .clients
            .values()
            .filter(|client| {
                client.ship.is_none() && client.adopt_deadline.is_some_and(|d| tick >= d)
            })
            .map(|client| client.session)
            .collect();
        for session in overdue {
            match self.free_spawn_point() {
                Some(pos) => {
                    let guid = self.spawn_ship_for(session, pos);
                    if let Some(client) = self.clients.get_mut(&session) {
                        client.ship = Some(guid);
                        client.replica = ClientReplica::new(guid);
                        client.adopt_deadline = None;
                    }
                }
                None => {
                    // The last word on this connection (the front end closes
                    // it once it is written): where a new player starts, or
                    // "try again" while that is not known.
                    let answer = self.send_elsewhere().into_message();
                    self.send_to(Destination::Client(session), answer);
                    self.clients.remove(&session);
                }
            }
        }
    }

    /// The handoff pass: boundary crossings, commits, timeouts.
    fn handoff_pass(&mut self, tick: SimTick) {
        // A re-offer pause that has run out is forgotten.
        self.refused.retain(|_, (_, until)| tick < *until);

        // --- start a transfer for every island (ship, body, or several
        // bodies that may touch) whose home is another shard: a ship or body
        // that left our region, or an island consolidated elsewhere (see
        // [`crate::border`]). All members of an island go together.
        for (members, to, going_home) in self.islands_to_hand_off() {
            if going_home {
                self.note_sent_home(&members, tick);
            }
            let owned = members.iter().any(|guid| self.session_of(*guid).is_some());
            let lead = if owned {
                HANDOFF_LEAD
            } else {
                crate::border::BODY_HANDOFF_LEAD
            };
            let group = if members.len() > 1 {
                self.next_group += 1;
                self.next_group
            } else {
                0
            };
            for guid in members {
                if self.session_of(guid).is_some() {
                    self.offer_ship(guid, to, tick, lead, group);
                } else {
                    self.offer_body(guid, to, tick, lead, group);
                }
            }
        }

        self.commit_pass(tick);
        self.timeout_pass(tick);
    }

    /// Offer the ship `guid` a client drives to `to`. Its player is redirected
    /// there once `to` has accepted ([`Shard::redirect_owner`]).
    fn offer_ship(
        &mut self,
        guid: EntityGuid,
        to: ShardId,
        tick: SimTick,
        lead: SimTick,
        group: u64,
    ) {
        let Some(entity) = self.sim.entity_of(guid) else {
            return;
        };
        let Some(state) = replicate::snapshot_of(&self.sim, &self.guids, entity) else {
            return;
        };
        // Stamp the owner's token into the offer's state so the accepting
        // shard can require a claiming client's hello token to match it
        // (see [`OWNER_TOKEN_COMPONENT`]).
        let mut state = with_owner_token(state, &self.owner_token_of(guid));
        state.components.push(TransferKind::Ship.component(lead));
        state.components.push(crate::handoff::scene_time_component(
            self.sim.scene_time(),
            self.scene_clock_set,
        ));
        let offer =
            Ledger::offer_with_lead(guid, self.config.shard, to, self.epoch(), tick, state, lead);
        self.ledger
            .offer_sent_as(&offer, tick, TransferKind::Ship, group);
        self.send_to(Destination::Shard(to), Message::HandoffOffer(offer));
    }

    /// `to` accepted the ship `guid`: tell the player who flies it, with the
    /// activation tick, so its client opens the new connection while the old
    /// path is still live (guarantee 3).
    ///
    /// Not before the accept: until then the accepting shard has not staged
    /// the transfer, so the owner's hello there would not match a staged
    /// owner token and would be answered like a fresh join — and its
    /// `NeighbourStatus` would be dropped as a claim for nothing. The accept
    /// is sent after the staging, so a client redirected here can only
    /// arrive after it.
    ///
    /// **Only the owner.** Everyone who had the ship in sight used to be sent
    /// the redirect too ("follow it rather than lose it"). A bystander has
    /// nothing to follow it *with*: a client plays on the shard that
    /// simulates its own ship, and what it sees of the ship that left is
    /// this shard's ghost of it (`crate::border`), fed by the new owner. All
    /// the redirect did for a bystander was make it say hello to the
    /// accepting shard with its own token — where nothing is staged for it,
    /// so it was sent away again, or, on the shard that owns the spawn
    /// point, welcomed as a newcomer and given a **second ship** there for
    /// as long as its client kept the useless connection (ten seconds).
    fn redirect_owner(&mut self, guid: EntityGuid, to: ShardId, activation: SimTick) {
        let Some(session) = self.session_of(guid) else {
            return;
        };
        let tick = self.sim.tick();
        let redirect = Redirect {
            entity: guid,
            new_shard: to,
            // The map as a player is to see it (client endpoints only).
            shard_map: self.shard_map.for_players(),
            activation_tick: activation,
        };
        if let Some(client) = self.clients.get_mut(&session) {
            client.redirect_pending = Some(to);
            client.redirect_deadline = Some(tick.saturating_add(REDIRECT_TIMEOUT_TICKS));
        }
        self.send_to(Destination::Client(session), Message::Redirect(redirect));
    }

    /// Commit every transfer that is ready (and, for a group, every member
    /// of it is), and abort the ones the epoch has invalidated.
    fn commit_pass(&mut self, tick: SimTick) {
        // --- commit everything that is ready at or after the activation tick.
        //
        // An in-flight transfer whose epoch no longer matches our map can never be
        // committed (guarantee 7): abort it here rather than let it sit past the
        // commit gate forever. This is the *after-the-gate* case — the timeout sweep
        // below no longer needs to special-case accepted+connected transfers because
        // this one fires the moment the epoch moves.
        let stale: Vec<EntityGuid> = self
            .ledger
            .outgoing()
            .iter()
            .filter(|(_, e)| e.epoch != self.epoch())
            .map(|(g, _)| *g)
            .collect();
        for guid in stale {
            self.abort_outgoing(guid, HandoffReason::EpochChanged);
        }

        let is_ready = |e: &crate::handoff::Outgoing| {
            tick >= e.activation_tick && e.accepted && e.client_connected
        };
        // The player's ship of a group, if it has one: the group goes when
        // the ship does.
        let ship_of = |group: u64| {
            self.ledger.group_members(group).into_iter().find(|m| {
                self.ledger
                    .outgoing()
                    .get(m)
                    .is_some_and(|e| e.kind == TransferKind::Ship)
            })
        };
        let mut ready: Vec<EntityGuid> = self
            .ledger
            .outgoing()
            .iter()
            .filter(|(_, e)| is_ready(e))
            .filter(|(_, e)| match (e.kind, ship_of(e.group)) {
                // A player's ship waits for nothing but its own gate.
                (TransferKind::Ship, _) => true,
                // A body travelling with a ship goes on the tick the ship
                // goes.
                (TransferKind::Body, Some(ship)) => {
                    self.ledger.outgoing().get(&ship).is_some_and(is_ready)
                }
                // A member of a group of bodies commits only together with
                // all the others.
                (TransferKind::Body, None) => self
                    .ledger
                    .group_members(e.group)
                    .iter()
                    .all(|m| self.ledger.outgoing().get(m).is_some_and(is_ready)),
            })
            .map(|(g, _)| *g)
            .collect();
        ready.sort_unstable();
        // The bodies of a group whose ship goes now and which are not ready
        // themselves (not accepted yet) stay behind: they follow as an
        // island that touches the ship's ghost.
        let left_behind: Vec<EntityGuid> = ready
            .iter()
            .filter_map(|guid| self.ledger.outgoing().get(guid))
            .filter(|e| e.kind == TransferKind::Ship && e.group != 0)
            .flat_map(|e| self.ledger.group_members(e.group))
            .filter(|member| !ready.contains(member))
            .collect();
        for guid in left_behind {
            self.abort_one_outgoing(
                guid,
                HandoffReason::Other("the ship it was travelling with went ahead".into()),
            );
        }
        for guid in ready {
            let kind = self
                .ledger
                .outgoing()
                .get(&guid)
                .map_or(TransferKind::Ship, |e| e.kind);
            let Some(state) = self.commit_state(guid, kind) else {
                continue;
            };
            // The commit carries the state *as of this tick* (guarantee 4), and we
            // stop simulating the entity the moment it goes out (guarantee 2): it
            // is never stepped again. It leaves the world at the end of the tick
            // (see `departing`), after this tick's frames are built, so the
            // client still receives a frame for the tick it was handed over on.
            if let Some(commit) =
                self.ledger
                    .ready_commit(guid, self.config.shard, self.epoch(), state)
            {
                let to = commit.to;
                self.send_to(
                    Destination::Shard(commit.to),
                    Message::HandoffCommit(commit),
                );
                self.departing.push((guid, to));
                self.stats.handed_off += 1;
            }
        }
    }

    /// The timeout sweeps of both sides (guarantee 8).
    fn timeout_pass(&mut self, tick: SimTick) {
        // --- timeouts.
        //
        // Any outgoing transfer older than the budget terminates, whatever it is
        // waiting on. The old filter skipped transfers that were already
        // accepted *and* client-connected, so a transfer that met the commit gate
        // but could no longer commit (e.g. the epoch moved) leaked forever — the
        // offering shard never committed, never aborted, and `is_busy` blocked a
        // fresh offer. Guarantee 8 says every waiting state is bounded; this is
        // where that is enforced.
        let expired: Vec<(EntityGuid, HandoffReason)> = self
            .ledger
            .outgoing()
            .iter()
            .filter(|(_, e)| tick.saturating_sub(e.started_at) > HANDOFF_TIMEOUT_TICKS)
            .map(|(g, e)| {
                (
                    *g,
                    if e.accepted {
                        HandoffReason::CommitTimeout
                    } else {
                        HandoffReason::OfferTimeout
                    },
                )
            })
            .collect();
        for (guid, reason) in expired {
            self.abort_outgoing(guid, reason);
        }

        // --- the accepting side's timeout.
        //
        // A staged transfer whose commit never arrived (the offering shard died, or
        // its commit was lost) must not hold the guid forever: `on_offer` refuses
        // anything `is_busy`, so a leaked staged entry poisons the guid for the
        // shard's whole life. Drop any staged entry with no confirmed session that
        // is older than the budget.
        //
        // A body's commit needs no session, so the sender may commit it right up
        // to the end of *its* budget (a group waits for its slowest member): a
        // staged body waits twice as long here, so it is never dropped while a
        // commit for it can still be on its way — that commit would be the only
        // copy left (the sender has let go).
        let stale_incoming: Vec<EntityGuid> = self
            .ledger
            .incoming()
            .iter()
            .filter(|(_, e)| {
                let budget = match e.kind {
                    TransferKind::Ship => HANDOFF_TIMEOUT_TICKS,
                    TransferKind::Body => 2 * HANDOFF_TIMEOUT_TICKS,
                };
                e.session.is_none() && tick.saturating_sub(e.started_at) > budget
            })
            .map(|(g, _)| *g)
            .collect();
        for guid in stale_incoming {
            self.abort_incoming(guid, HandoffReason::CommitTimeout);
        }
    }

    /// Keep an outbox nobody is draining from growing without bound (see
    /// [`MAX_OUTBOX`]): once it is over the limit, the oldest client deltas and
    /// scene announcements are discarded until it is half full again. A no-op
    /// in a running shard.
    fn trim_outbox(&mut self) {
        if self.outbox.len() <= MAX_OUTBOX {
            return;
        }
        let mut to_drop = self.outbox.len() - MAX_OUTBOX / 2;
        self.outbox.retain(|out| {
            let stale = to_drop > 0
                && matches!(out.dest, Destination::Client(_))
                && matches!(out.message, Message::Delta(_) | Message::SceneUpdate(_));
            if stale {
                to_drop -= 1;
            }
            !stale
        });
    }

    /// Build and queue one frame per client.
    ///
    /// Every client that has a ship in this world gets a frame every tick (see
    /// [`crate::replicate`], "A frame every tick"). The interest sphere is
    /// queried once per client, and whatever is the same for every client this
    /// tick — the encoded entities, the changed set — is computed once in a
    /// [`PassCache`].
    fn replicate_pass(&mut self) {
        self.trim_outbox();
        // What every entity's components are now, and which changed.
        {
            let ghosts = &self.ghosts;
            self.log.refresh(&mut self.sim, &self.guids, |sim, guid| {
                physics::replicated_of(sim, ghosts, guid)
            });
        }
        // ... and the far replicas whose sample is new.
        self.record_far(self.sim.tick());
        let radius = self.config.interest_radius;
        let cx = replicate::FrameContext {
            shard: self.config.shard,
            epoch: self.shard_map.epoch,
            budget: self.config.frame_budget,
            guids: &self.guids,
            log: &self.log,
        };
        let mut cache = PassCache::new();
        let mut visible_set: HashSet<EntityGuid> = HashSet::new();
        for client in self.clients.values_mut() {
            let Some(ship) = client.ship else {
                continue; // awaiting adopt: nothing to anchor a view on yet
            };
            // The anchor is the client's own ship; if it is gone (handed off and
            // despawned) the client's view is about to be replaced by a redirect
            // anyway, so nothing is sent.
            let Some(anchor) = self.sim.ship_state(ship).map(|s| s.pos) else {
                continue;
            };
            let (visible, distances) = self.interest.visible_at(anchor, radius);
            visible_set.clear();
            visible_set.extend(visible.iter().copied());
            let frame = replicate::build_frame(
                &mut self.sim,
                cx,
                &mut cache,
                &mut client.replica,
                Visible {
                    ordered: &visible,
                    set: &visible_set,
                    distances: &distances,
                },
            );
            self.stats.client_bytes += frame.wire_len() as u64;
            let message = match frame {
                Frame::Snapshot(snapshot) => Message::Snapshot(snapshot),
                Frame::Delta(delta) => Message::Delta(delta),
            };
            self.outbox.push(OutMessage {
                dest: Destination::Client(client.session),
                message,
            });
        }
    }

    /// The session driving `guid`, if one of our clients drives it.
    #[must_use]
    pub fn session_of(&self, guid: EntityGuid) -> Option<u64> {
        self.clients
            .values()
            .find(|c| c.ship == Some(guid))
            .map(|c| c.session)
    }

    /// The player token of the client that owns `entity` here, if one does.
    ///
    /// Stamped into the offer's state so the accepting shard can require a
    /// claiming client's hello token to match it. An entity with no owning client
    /// (a stray ship) yields the empty string, which no real hello token equals.
    fn owner_token_of(&self, entity: EntityGuid) -> String {
        self.clients
            .values()
            .find(|client| client.ship == Some(entity))
            .map(|client| client.player_token.clone())
            .unwrap_or_default()
    }

    /// The client's ship guid, if the session is connected.
    #[must_use]
    pub fn ship_of(&self, session: u64) -> Option<EntityGuid> {
        self.clients.get(&session).and_then(|c| c.ship)
    }
}

/// Decode the [`ShipState`] out of an [`EntitySnapshot`]'s `ShipState` component.
fn decode_ship_state(snapshot: &space_protocol::EntitySnapshot) -> Option<ShipState> {
    let id = <space_sim::ShipState as space_ecs::ComponentDef>::ID;
    let payload = snapshot
        .components
        .iter()
        .find(|(c, _)| *c == id)
        .map(|(_, p)| p.clone())?;
    space_sim::wire::decode_ship_state(&payload).ok()
}

/// Decode the [`ShipControls`] out of an [`EntitySnapshot`], if it carries them.
fn decode_ship_controls(snapshot: &space_protocol::EntitySnapshot) -> Option<ShipControls> {
    let id = <ShipControls as space_ecs::ComponentDef>::ID;
    let payload = snapshot.components.iter().find(|(c, _)| *c == id)?;
    space_protocol::codec::decode_raw::<ShipControls>(&payload.1)
        .ok()
        .map(|(controls, _)| controls)
}

/// Keep `inputs` for the ship `ship` that is on its way here, by tick: the
/// ones for ticks the install may still have to replay or file (not older
/// than the catch-up reaches, not further ahead than the simulation takes
/// one), one per tick, [`MAX_STAGED_INPUTS`] at most (the oldest go first).
/// Every input is rewritten to `ship`, whatever the client named.
fn keep_staged(
    kept: &mut BTreeMap<SimTick, PlayerInput>,
    inputs: impl IntoIterator<Item = PlayerInput>,
    ship: EntityGuid,
    now: SimTick,
) {
    for mut input in inputs {
        let wanted = input.tick.saturating_add(MAX_CATCH_UP) > now
            && input.tick <= now.saturating_add(space_sim::sim_world::MAX_INPUT_LEAD);
        if wanted {
            input.entity = ship;
            kept.insert(input.tick, input);
        }
    }
    while kept.len() > MAX_STAGED_INPUTS {
        kept.pop_first();
    }
}

/// Bring a handed-over ship forward by the `late` ticks after `from`: the
/// ticks between the one its committed state describes and the one it is
/// installed on (see [`Shard::install_pending_commits`]). Returns the state
/// and whether the ship is in fast travel after them.
///
/// Each tick is the flight model's free-flight step
/// ([`space_sim::physics::ship_velocity_step`], then the pose) under the
/// input `inputs` holds for it — a tick without one coasts, as in the
/// simulation. It is **not** the full physics: no gravity, no contact, no
/// spring acts on the ship for those ticks (the step that follows the install
/// resolves what it then touches), and a ship in fast travel simply flies on
/// along its velocity (its speed limit depends on the scene around it).
/// That is the price of a late commit; one that arrives within its tick
/// costs nothing (`late == 0` returns `state` untouched).
fn catch_up_ship(
    mut state: ShipState,
    mut fast: bool,
    params: &space_sim::ShipParams,
    inputs: &BTreeMap<SimTick, PlayerInput>,
    from: SimTick,
    late: SimTick,
) -> (ShipState, bool) {
    use space_sim::physics;
    for tick in (from + 1)..=(from + late) {
        let input = inputs.get(&tick);
        if let Some(input) = input {
            if fast && !input.fast {
                // Leaving fast travel drops to the normal top speed at once.
                physics::limit_speed(&mut state, f64::from(physics::top_speed(params)));
            }
            fast = input.fast;
        }
        if !fast {
            let (thrust, torque) = input.map_or(([0.0; 3], [0.0; 3]), |i| (i.thrust, i.torque));
            physics::ship_velocity_step(&mut state, thrust, torque, params, [0.0; 3]);
        }
        physics::ship_pose_step(&mut state);
    }
    (state, fast)
}

/// The private component id a [`HandoffOffer`]'s state carries the owner's player
/// token under.
///
/// The accepting shard must only let a client adopt a staged entity when the
/// client's hello token matches the owner's — a guid is `(shard, serial)` with
/// serials starting at 1, so it is trivially guessable, and adopting on a bare
/// guid claim lets any client drive somebody else's ship. `space-protocol` is
/// frozen for this phase (no new field on `HandoffOffer`), so the token rides as
/// an extra component in the state payload the offering shard already controls.
///
/// The id is deliberately outside the registry's id space (space-sim registers
/// 0..=3) so it can never collide with a real component, and it is **never**
/// emitted in a client snapshot or delta — only in the offer's state, which goes
/// shard-to-shard. Nothing signs or authenticates it today: a signed handoff
/// ticket is deferred (see `docs/security-notes.md`).
const OWNER_TOKEN_COMPONENT: u16 = u16::MAX;

/// Return a copy of `snapshot` carrying the owner's player token as a private
/// component (see [`OWNER_TOKEN_COMPONENT`]).
fn with_owner_token(
    snapshot: space_protocol::EntitySnapshot,
    token: &str,
) -> space_protocol::EntitySnapshot {
    let mut components = snapshot.components;
    components.retain(|(id, _)| *id != OWNER_TOKEN_COMPONENT);
    components.push((OWNER_TOKEN_COMPONENT, token.as_bytes().to_vec()));
    space_protocol::EntitySnapshot::new(snapshot.guid, components)
}

/// The owner token a handoff state carries, or the empty string when it carries
/// none (an entity with no owning client).
fn offer_token_of(snapshot: &space_protocol::EntitySnapshot) -> String {
    snapshot
        .components
        .iter()
        .find(|(id, _)| *id == OWNER_TOKEN_COMPONENT)
        .and_then(|(_, payload)| String::from_utf8(payload.clone()).ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A client cannot make the shard hold (and feed to the simulation, under
    /// the shard lock) more inputs than it could ever use: one frame can carry
    /// tens of thousands, and at most one per tick is ever applied to the one
    /// ship a session drives.
    ///
    /// Mutation this catches: pushing every input of a batch without the
    /// `MAX_PENDING_INPUTS` bound in `Shard::apply_input`.
    #[test]
    fn a_flood_of_inputs_is_capped_between_ticks() {
        let mut shard = Shard::new(ShardConfig::new(
            ShardId::new(1),
            std::net::SocketAddr::from(([127, 0, 0, 1], 0)),
            NodeRef::new(64, [0, 0, 0]),
        ));
        let session = shard
            .join(&Hello {
                protocol: space_protocol::PROTOCOL_VERSION,
                player_token: "flooder".into(),
            })
            .expect("join")
            .session;
        let ship = shard.ship_of(session).expect("a ship");
        let flood = InputTick {
            tick: 1,
            seq: 1,
            inputs: vec![
                PlayerInput {
                    entity: ship,
                    thrust: [1.0, 0.0, 0.0],
                    torque: [0.0; 3],
                    tick: 1,
                    fast: false,
                };
                23_000
            ],
            ack_tick: 0,
        };
        shard.apply_input(session, &flood);
        shard.apply_input(session, &flood);
        assert_eq!(
            shard.clients[&session].pending_inputs.len(),
            MAX_PENDING_INPUTS,
            "the backlog is bounded however much is sent"
        );

        // The cap costs a well-behaved input nothing: the tick drains the
        // backlog and the ship answers.
        shard.tick();
        assert!(shard.clients[&session].pending_inputs.is_empty());
        let state = shard.sim().ship_state(ship).expect("alive");
        assert!(state.vel[0] > 0.0, "the input was applied: {state:?}");
    }

    /// An outbox nobody drains stops growing, and what it sheds is old client
    /// deltas only.
    ///
    /// Every client gets a frame every tick, so a caller that steps the shard
    /// with `tick_n` and never collects the outbox would otherwise leak one
    /// message per client per tick.
    ///
    /// Mutation this catches: removing `trim_outbox` from the replication pass
    /// (the outbox then holds one frame per tick ever stepped), or letting it
    /// discard anything but a client delta (the snapshot or the peer message
    /// would be gone).
    #[test]
    fn an_undrained_outbox_is_bounded_and_sheds_only_old_deltas() {
        let mut config = ShardConfig::new(
            ShardId::new(1),
            std::net::SocketAddr::from(([127, 0, 0, 1], 0)),
            NodeRef::new(64, [0, 0, 0]),
        );
        // A small sphere keeps ten thousand ticks cheap; it changes nothing else.
        config.interest_radius = 100.0;
        let mut shard = Shard::new(config);
        let session = shard
            .join(&Hello {
                protocol: space_protocol::PROTOCOL_VERSION,
                player_token: "nobody-listens".into(),
            })
            .expect("join")
            .session;
        shard.send_to(
            Destination::Shard(ShardId::new(9)),
            Message::StaleEpoch(StaleEpoch {
                seen: 1,
                current: 2,
            }),
        );

        let last = shard.tick_n(MAX_OUTBOX as SimTick * 2 + 500);
        assert!(
            shard.outbox.len() <= MAX_OUTBOX + 1,
            "the outbox must stop growing, got {} messages",
            shard.outbox.len()
        );
        assert!(
            shard
                .outbox
                .iter()
                .any(|o| matches!(o.message, Message::Snapshot(_))),
            "the snapshot is never shed"
        );
        assert!(
            shard
                .outbox
                .iter()
                .any(|o| matches!(o.dest, Destination::Shard(_))),
            "a message for another shard is never shed"
        );
        let newest = shard.outbox.last().expect("the outbox is not empty");
        assert_eq!(newest.dest, Destination::Client(session));
        assert!(
            matches!(&newest.message, Message::Delta(d) if d.tick == last),
            "the newest frame is kept: {newest:?}"
        );
        // The periodic scene announcements are shed with the deltas (a later
        // one says the same): they do not pile up either. Over this many ticks
        // there were dozens; what is left is what fits the kept window.
        let announced = (last / SCENE_REANNOUNCE_TICKS) as usize;
        let kept = shard
            .outbox
            .iter()
            .filter(|o| matches!(o.message, Message::SceneUpdate(_)))
            .count();
        assert!(announced > 50, "precondition: {announced} announcements");
        assert!(
            kept <= MAX_OUTBOX / SCENE_REANNOUNCE_TICKS as usize + 2,
            "{kept} of {announced} scene announcements are still queued"
        );
    }

    // ------------------------------------------------------------- the scene

    use space_protocol::Scene;

    fn shard() -> Shard {
        Shard::new(ShardConfig::new(
            ShardId::new(1),
            std::net::SocketAddr::from(([127, 0, 0, 1], 0)),
            NodeRef::new(64, [0, 0, 0]),
        ))
    }

    fn join(shard: &mut Shard, token: &str) -> u64 {
        shard
            .join(&Hello {
                protocol: space_protocol::PROTOCOL_VERSION,
                player_token: token.into(),
            })
            .expect("join")
            .session
    }

    /// The `(session, state)` of every scene update in the outbox, draining it.
    fn scene_updates(shard: &mut Shard) -> Vec<(u64, SceneState)> {
        shard
            .take_outbox()
            .into_iter()
            .filter_map(|out| match (out.dest, out.message) {
                (Destination::Client(session), Message::SceneUpdate(state)) => {
                    Some((session, state))
                }
                _ => None,
            })
            .collect()
    }

    fn state(scene: Scene, revision: u64) -> SceneState {
        SceneState::new(scene.id(), revision)
    }

    /// A client is told the scene as the first thing behind its welcome: the
    /// update is queued by the join itself, ahead of the session's first frame.
    ///
    /// Mutation this catches: announcing the scene only on a change — a client
    /// that joins a shard whose scene was set earlier would never learn it.
    #[test]
    fn a_joining_client_is_told_the_scene_before_its_first_frame() {
        let mut shard = shard();
        assert_eq!(shard.scene(), SceneState::UNSET, "nobody has set one yet");
        assert!(shard.set_scene(state(Scene::Galaxy, 4)));
        assert!(shard.take_outbox().is_empty(), "nobody to tell yet");

        let session = join(&mut shard, "player");
        shard.tick();
        let outbox = shard.take_outbox();
        let mine: Vec<&Message> = outbox
            .iter()
            .filter(|out| out.dest == Destination::Client(session))
            .map(|out| &out.message)
            .collect();
        assert_eq!(
            mine.first(),
            Some(&&Message::SceneUpdate(state(Scene::Galaxy, 4))),
            "the scene comes first"
        );
        assert_eq!(
            mine.get(1),
            Some(&&Message::SceneClock(shard.scene_clock())),
            "then the scene clock"
        );
        assert!(
            matches!(mine.get(2), Some(Message::Snapshot(_))),
            "then the first frame: {mine:?}"
        );
        assert_eq!(
            mine.iter()
                .filter(|m| matches!(m, Message::SceneUpdate(_)))
                .count(),
            1,
            "it is said once, not every tick"
        );
        // A shard nobody configured still says what applies: the default.
        let mut bare = self::shard();
        let session = join(&mut bare, "player");
        assert_eq!(scene_updates(&mut bare), vec![(session, SceneState::UNSET)]);
        assert_eq!(SceneState::UNSET.scene.known(), Some(Scene::DEFAULT));
    }

    /// A client whose scene announcement never arrived — the front end drops
    /// what it offers to a full write queue — is told again within
    /// `SCENE_REANNOUNCE_TICKS`, without any change of scene, and the repeat
    /// costs the other ticks nothing: every tick still carries exactly one
    /// frame per client and nothing else.
    ///
    /// Mutation this catches: announcing the scene on join and on change only
    /// (the client that lost its announcement draws the old scene until the
    /// next change), or re-announcing on every tick.
    #[test]
    fn a_client_that_missed_the_scene_is_told_again_without_a_change() {
        let mut shard = shard();
        assert!(shard.set_scene(state(Scene::Galaxy, 4)));
        let (a, b) = (join(&mut shard, "a"), join(&mut shard, "b"));
        // Both announcements are lost on the way to the clients.
        let lost = scene_updates(&mut shard);
        assert_eq!(lost.len(), 2, "precondition: the join announced the scene");

        let mut told: Vec<(SimTick, u64, SceneState)> = Vec::new();
        let mut clocks = 0;
        for _ in 0..(2 * SCENE_REANNOUNCE_TICKS) {
            let tick = shard.tick();
            let mut frames = 0;
            for out in shard.take_outbox() {
                match (out.dest, out.message) {
                    (Destination::Client(session), Message::SceneUpdate(scene)) => {
                        assert_eq!(frames, 0, "the announcement comes before the frames");
                        told.push((tick, session, scene));
                    }
                    (Destination::Client(_), Message::SceneClock(clock)) => {
                        assert_eq!(frames, 0, "the clock comes before the frames");
                        assert_eq!(tick % SCENE_REANNOUNCE_TICKS, 0, "with the scene");
                        assert_eq!(clock, shard.scene_clock());
                        clocks += 1;
                    }
                    (Destination::Client(_), Message::Snapshot(_) | Message::Delta(_)) => {
                        frames += 1;
                    }
                    other => panic!("tick {tick}: unexpected {other:?}"),
                }
            }
            assert_eq!(frames, 2, "tick {tick}: one frame per client, as ever");
        }
        told.sort_unstable_by_key(|(tick, session, _)| (*tick, *session));
        let solar = state(Scene::Galaxy, 4);
        assert_eq!(
            told,
            vec![
                (SCENE_REANNOUNCE_TICKS, a, solar),
                (SCENE_REANNOUNCE_TICKS, b, solar),
                (2 * SCENE_REANNOUNCE_TICKS, a, solar),
                (2 * SCENE_REANNOUNCE_TICKS, b, solar),
            ],
            "the current scene, to every client, once per interval and on no other tick"
        );
        assert_eq!(shard.scene(), solar, "no change of scene was needed");
        assert_eq!(clocks, 4, "the clock is said again with the scene");

        // A shard with nobody connected says nothing to nobody.
        let mut empty = self::shard();
        empty.tick_n(2 * SCENE_REANNOUNCE_TICKS);
        assert!(empty.take_outbox().is_empty());
    }

    /// A change reaches every connected client, once; a stale or repeated
    /// revision reaches nobody and changes nothing.
    ///
    /// Mutation this catches: `>=` instead of `>` in the revision test (a
    /// repeat would be re-broadcast), or adopting whatever arrives last (a
    /// late message rolls the scene back).
    #[test]
    fn a_scene_change_reaches_every_client_and_a_stale_one_is_ignored() {
        let mut shard = shard();
        let (a, b) = (join(&mut shard, "a"), join(&mut shard, "b"));
        let _ = shard.take_outbox();

        assert!(shard.set_scene(state(Scene::Galaxy, 2)));
        let mut told = scene_updates(&mut shard);
        told.sort_unstable_by_key(|(session, _)| *session);
        assert_eq!(
            told,
            vec![(a, state(Scene::Galaxy, 2)), (b, state(Scene::Galaxy, 2))]
        );

        for stale in [
            state(Scene::Galaxy, 2),
            state(Scene::Galaxy, 1),
            state(Scene::Galaxy, 2),
            SceneState::UNSET,
        ] {
            assert!(!shard.set_scene(stale), "{stale} is not newer");
            assert_eq!(shard.scene(), state(Scene::Galaxy, 2));
            assert!(
                shard.take_outbox().is_empty(),
                "{stale} is relayed to nobody"
            );
        }

        // A scene this build has never heard of is still relayed: the shard
        // only passes the id on, and the client decides what to draw.
        let unknown = SceneState::new(space_protocol::SceneId(4_000), 3);
        assert!(shard.set_scene(unknown));
        assert_eq!(scene_updates(&mut shard).len(), 2);
        assert_eq!(shard.scene(), unknown);
    }

    /// The orchestrator's scene is adopted, relayed to the clients and
    /// confirmed; a stale one is confirmed with what the shard really runs.
    /// Nothing about the map or the epoch protocol moves.
    ///
    /// Mutation this catches: acking the revision that was *sent* (the
    /// orchestrator would believe a shard confirmed a scene it refused), or no
    /// ack for a no-op (a shard that already had the scene would show as
    /// behind for ever).
    #[test]
    fn the_orchestrators_scene_is_adopted_relayed_and_confirmed() {
        let mut shard = shard();
        let session = join(&mut shard, "player");
        let _ = shard.take_outbox();
        let (epoch, map) = (shard.epoch(), shard.shard_map().clone());

        let drain = |shard: &mut Shard| -> (Vec<SceneAck>, Vec<(u64, SceneState)>) {
            let mut acks = Vec::new();
            let mut updates = Vec::new();
            for out in shard.take_outbox() {
                match (out.dest, out.message) {
                    (Destination::Orchestrator, Message::SceneAck(ack)) => acks.push(ack),
                    (Destination::Client(s), Message::SceneUpdate(state)) => {
                        updates.push((s, state));
                    }
                    other => panic!("unexpected {other:?}"),
                }
            }
            (acks, updates)
        };

        shard.orchestrator_message(Message::SceneSet(state(Scene::Galaxy, 4)));
        let (confirmed, relayed) = drain(&mut shard);
        assert_eq!(
            confirmed,
            vec![SceneAck {
                shard: ShardId::new(1),
                applied: state(Scene::Galaxy, 4)
            }]
        );
        assert_eq!(relayed, vec![(session, state(Scene::Galaxy, 4))]);

        // The same message again (the orchestrator re-sends until it hears an
        // ack): confirmed again, relayed to nobody.
        shard.orchestrator_message(Message::SceneSet(state(Scene::Galaxy, 4)));
        let (confirmed, relayed) = drain(&mut shard);
        assert_eq!(confirmed.len(), 1);
        assert_eq!(confirmed[0].applied, state(Scene::Galaxy, 4));
        assert!(relayed.is_empty());

        // A late, older one: the answer names the newer scene the shard runs.
        shard.orchestrator_message(Message::SceneSet(state(Scene::Galaxy, 3)));
        let (confirmed, relayed) = drain(&mut shard);
        assert_eq!(confirmed[0].applied, state(Scene::Galaxy, 4));
        assert!(relayed.is_empty());
        assert_eq!(shard.scene(), state(Scene::Galaxy, 4));

        assert_eq!(shard.epoch(), epoch, "a scene is not an epoch");
        assert_eq!(shard.shard_map(), &map);
        assert!(!shard.epoch_participant().has_staged());
    }
}
