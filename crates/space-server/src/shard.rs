//! [`Shard`]: the state one tick of the shard loop drives.
//!
//! Everything here is synchronous and deterministic: given the same sequence of
//! [`Shard::join`] / [`Shard::apply_input`] / [`Shard::peer_message`] calls and
//! the same number of [`Shard::tick`] calls, two shards produce the same
//! [`SimWorld::guid_state_hash`]. The quinn front end ([`crate::net`]) is the
//! only async part, and it only moves messages in and out. That split is what
//! lets the acceptance tests run two shards inside one process and hand their
//! outboxes to each other.

use std::collections::{HashMap, HashSet};

use space_protocol::{
    EntityGuid, HandoffReason, Hello, InputTick, Message, NeighbourStatus, PlayerInput, Redirect,
    RejectReason, SceneAck, SceneState, ShardId, ShardMap, ShardRegion, StaleEpoch, Welcome,
};
use space_sim::{ShipState, SimTick, SimWorld};
use space_world::{NodeRef, WorldPos};

use crate::config::{
    ShardConfig, ADOPT_TIMEOUT_TICKS, MAX_PENDING_INPUTS, REDIRECT_TIMEOUT_TICKS,
    REOFFER_BACKOFF_TICKS, SCENE_REANNOUNCE_TICKS,
};
use crate::epoch::EpochParticipant;
use crate::handoff::{Ledger, TransferKind, HANDOFF_LEAD, HANDOFF_TIMEOUT_TICKS};
use crate::replicate::{
    self, ClientReplica, ComponentLog, Components, Frame, GuidIndex, Interest, PassCache, Visible,
};

// The physics half of the shard: the scene's bodies, spawn points, ghosts at
// the borders, island consolidation and the body handoff path. A child
// module, so it works on the same private state as the rest of this file.
#[path = "shard_physics.rs"]
mod physics;
pub use physics::BodyInfo;

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
}

/// A commit that has arrived but not been installed yet.
///
/// The gap between "the commit is delivered" and "our step reaches
/// `install_at`" is where the entity must **not** be simulated: its state is
/// the state as of the offering shard's activation tick, so stepping it earlier
/// would double-step.
///
/// The two ticks are deliberately separate. `activation_tick` is the *offering*
/// shard's clock and is only meaningful for input reconciliation (the client's
/// pending inputs are tagged with the old shard's ticks); `install_at` is
/// **our** clock, derived when we accepted the offer. Two shards run independent
/// loops and drift, so one number cannot serve both.
#[derive(Debug)]
struct PendingCommit {
    guid: EntityGuid,
    session: u64,
    state: ShipState,
    /// The offering shard's activation tick: the input-ack floor we hand the
    /// client, whose inputs are tagged in that clock.
    activation_tick: SimTick,
    /// Our own tick at which to install (see [`crate::handoff::Incoming`]).
    install_at: SimTick,
    /// What arrives: a player's ship (spawned from `state`) or a body
    /// (imported from `components`).
    kind: TransferKind,
    /// A body's full component list, private components stripped.
    components: Components,
    /// The shard that committed it (and now holds a ghost of it).
    from: ShardId,
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
    /// The shard map this shard publishes and follows.
    shard_map: ShardMap,
    /// The shard's epoch-protocol participation (Phase 3): the staged prepare,
    /// if one is outstanding.
    epoch_participant: EpochParticipant,
    /// The scene clients are told to draw, and the revision it was set at (see
    /// [`Shard::set_scene`]).
    scene: SceneState,
    stats: ShardStats,
}

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
            shard_map: map,
            epoch_participant: EpochParticipant::new(),
            scene: SceneState::UNSET,
            stats: ShardStats::default(),
        }
    }

    /// This shard's id.
    #[must_use]
    pub const fn shard(&self) -> ShardId {
        self.config.shard
    }

    /// Record the address the front end actually bound to.
    ///
    /// A shard started on port 0 does not know its port until the socket exists,
    /// and the shard map it hands out in a [`Welcome`] must carry a reachable one.
    pub fn set_endpoint(&mut self, addr: std::net::SocketAddr) {
        self.config.endpoint = addr;
        for region in &mut self.shard_map.regions {
            if region.shard == self.config.shard {
                region.endpoint = addr;
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
    /// the shard's own policy: a token is turned away only for capacity or
    /// draining, and a successful join spawns the player's ship **inside this
    /// shard's region** so the client starts on authoritative ground.
    ///
    /// **The one exception is a redirected client.** A client that connects to us
    /// while we hold a staged handoff whose owner token matches the client's hello
    /// token is coming *for that entity*, so we do not spawn a placeholder for it:
    /// the placeholder used to sit at the region centre until the client reported
    /// in, and a client that crashed before reporting left a phantom ship there
    /// forever. Such a client is *awaiting adopt* ([`Client::ship`] is `None`) and
    /// gets its ship by adopting the staged entity in [`Shard::client_status`]; if
    /// adoption never comes it is spawned a ship like any newcomer after
    /// [`crate::config::ADOPT_TIMEOUT_TICKS`].
    ///
    /// Every client that is admitted — a newcomer and a redirected player's
    /// second connection alike — is queued a [`Message::SceneUpdate`] with the
    /// shard's current scene, so it is the first thing behind the [`Welcome`].
    pub fn join(&mut self, hello: &Hello) -> Result<Welcome, RejectReason> {
        if self.clients.len() >= MAX_PLAYERS_PER_SHARD {
            return Err(RejectReason::Full);
        }
        let session = self.next_session;
        self.next_session += 1;
        // Does this hello match a staged transfer's owner? If so the client is
        // the redirected player and gets no placeholder (see the method docs).
        let staged_owner = self
            .ledger
            .incoming()
            .values()
            .any(|staged| staged.owner_token == hello.player_token);
        let ship = if staged_owner {
            None
        } else {
            Some(self.spawn_player_ship(session))
        };
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
        Ok(Welcome {
            session,
            shard: self.config.shard,
            shard_map: self.shard_map.clone(),
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

    /// Spawn a joining player's ship at a free spawn point (see
    /// [`Shard::free_spawn_point`]), owned by `session`.
    fn spawn_player_ship(&mut self, session: u64) -> EntityGuid {
        let pos = self.free_spawn_point();
        self.spawn_ship_for(session, pos)
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
        // the client sent. An awaiting-adopt client has no ship yet, so its
        // inputs are dropped — the old shard is still simulating its entity.
        let Some(ship) = client.ship else {
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
                    self.clients
                        .get(&session)
                        .is_some_and(|client| client.player_token == staged.owner_token)
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

    /// Whether `shard` is one of the ids this shard's map names.
    #[must_use]
    pub fn knows_shard(&self, shard: ShardId) -> bool {
        shard != ShardId::NONE
            && self
                .shard_map
                .regions
                .iter()
                .any(|region| region.shard == shard)
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
        if self.is_crossing_offer(&offer) {
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
        self.ledger.mark_accepted(&accept);
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
        // The install gate is `local_activation_tick` — **our** clock, derived when
        // we accepted — never the commit's `activation_tick`, which is the offering
        // shard's clock and drifts against ours (the two shards run independent
        // tick loops). Comparing the two directly would double-step the ship (if we
        // lag) or freeze it (if we lead).
        self.pending_commits.push(PendingCommit {
            guid: commit.entity,
            session,
            state,
            activation_tick: commit.activation_tick,
            install_at: staged.local_activation_tick,
            kind: staged.kind,
            components,
            from: staged.from,
        });
    }

    fn on_abort(&mut self, abort: space_protocol::HandoffAbort) {
        // Either way an abort means "nothing happened": drop our half and keep
        // simulating whatever we have (guarantee 6).
        if let Some(outgoing) = self.ledger.take_outgoing(abort.entity) {
            self.stats.aborted += 1;
            // The peer turned our offer down. The ship is still on its side of
            // the boundary, so the handoff pass would offer it again on the very
            // next tick, be refused again, and so on at tick rate.
            self.pause_reoffer(abort.entity, outgoing.to);
            // An island moves together or not at all (guarantee 6).
            for member in self.ledger.group_members(outgoing.group) {
                self.abort_outgoing(
                    member,
                    HandoffReason::Other("a group member was refused".into()),
                );
                self.pause_reoffer(member, outgoing.to);
            }
        } else if self.ledger.take_incoming(abort.entity).is_some() {
            self.stats.aborted += 1;
        }
    }

    /// Do not offer `entity` to `to` again for
    /// [`REOFFER_BACKOFF_TICKS`](crate::config::REOFFER_BACKOFF_TICKS).
    fn pause_reoffer(&mut self, entity: EntityGuid, to: ShardId) {
        let until = self.sim.tick().saturating_add(REOFFER_BACKOFF_TICKS);
        self.refused.insert(entity, (to, until));
    }

    fn abort_outgoing(&mut self, entity: EntityGuid, reason: HandoffReason) {
        let group = self.ledger.outgoing().get(&entity).map_or(0, |e| e.group);
        if let Some(abort) = self
            .ledger
            .abort_outgoing(entity, self.config.shard, reason.clone())
        {
            self.stats.aborted += 1;
            self.send_to(Destination::Shard(abort.to), Message::HandoffAbort(abort));
        }
        // The rest of its island goes back too (guarantee 6).
        for member in self.ledger.group_members(group) {
            if let Some(abort) =
                self.ledger
                    .abort_outgoing(member, self.config.shard, reason.clone())
            {
                self.stats.aborted += 1;
                self.send_to(Destination::Shard(abort.to), Message::HandoffAbort(abort));
            }
        }
    }

    fn abort_incoming(&mut self, entity: EntityGuid, reason: HandoffReason) {
        if let Some(abort) = self
            .ledger
            .abort_incoming(entity, self.config.shard, reason)
        {
            self.stats.aborted += 1;
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
        // 1. Inputs.
        for client in self.clients.values_mut() {
            if client.pending_inputs.is_empty() {
                continue;
            }
            let inputs = std::mem::take(&mut client.pending_inputs);
            let next = self.sim.tick().saturating_add(1);
            self.sim.apply_inputs(next, &inputs);
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

        // 5.5 The scene, again, for whoever missed it (see
        // [`SCENE_REANNOUNCE_TICKS`]). Before the frames, so a tick's last
        // message to a client is still its frame.
        if tick.is_multiple_of(SCENE_REANNOUNCE_TICKS) {
            self.send_to_all(Message::SceneUpdate(self.scene));
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

    /// Install every commit whose **local** install tick has been reached.
    ///
    /// Called from [`Shard::tick`] immediately before the step, so the entity
    /// joins the world with the state it arrived with and the step that follows
    /// produces the next state — never stepped twice for one tick (guarantee 2)
    /// and never rewound to the offer's older state (guarantee 4).
    ///
    /// The gate is `install_at`, this shard's own clock, derived at accept time;
    /// the commit's `activation_tick` (the offering shard's clock) is used only as
    /// the client's input-ack floor, because the client's pending inputs are tagged
    /// in the old shard's clock.
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
            let entity = self
                .sim
                .spawn_ship(pending.guid, pending.session, pending.state);
            self.guids.insert(pending.guid, entity);
            self.interest
                .upsert(pending.guid, entity, pending.state.pos);
            self.mark_player(pending.guid, pending.session);
            self.stats.received += 1;
            self.note_sender_holds_ghost(pending.from, pending.guid);
            // A client driving this ship must not be sent a delta against a tick
            // it never saw, and its inputs up to activation were already applied
            // by the old shard (guarantee 5).
            for client in self.clients.values_mut() {
                if client.ship == Some(pending.guid) {
                    client.replica.input_ack_floor = pending.activation_tick;
                    client.replica.ack_tick = tick;
                }
            }
        }
    }

    /// Clear a redirect that was never confirmed, and give an awaiting-adopt
    /// client its fallback ship once its adopt budget has run out.
    ///
    /// Both are bounded waits: a redirect that never gets a `NeighbourStatus` must
    /// not hold the client half-redirected forever, and a client we withheld a
    /// placeholder for (because it matched a staged transfer's owner) must not be
    /// left with no ship at all if adoption never comes.
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
        // newcomer: give it a ship at the region centre.
        let to_spawn: Vec<u64> = self
            .clients
            .values()
            .filter(|client| {
                client.ship.is_none() && client.adopt_deadline.is_some_and(|d| tick >= d)
            })
            .map(|client| client.session)
            .collect();
        for session in to_spawn {
            let guid = self.spawn_player_ship(session);
            if let Some(client) = self.clients.get_mut(&session) {
                client.ship = Some(guid);
                client.replica = ClientReplica::new(guid);
                client.adopt_deadline = None;
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

    /// Offer the ship `guid` a client drives to `to`, and redirect everyone
    /// watching it there (see the module docs of [`crate::handoff`]).
    fn offer_ship(
        &mut self,
        guid: EntityGuid,
        to: ShardId,
        tick: SimTick,
        lead: SimTick,
        group: u64,
    ) {
        {
            let Some(entity) = self.sim.entity_of(guid) else {
                return;
            };
            let Some(ship_pos) = self.sim.ship_state(guid).map(|s| s.pos) else {
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
            let offer = Ledger::offer_with_lead(
                guid,
                self.config.shard,
                to,
                self.epoch(),
                tick,
                state,
                lead,
            );
            let activation = offer.activation_tick;
            self.ledger
                .offer_sent_as(&offer, tick, TransferKind::Ship, group);
            self.send_to(Destination::Shard(to), Message::HandoffOffer(offer));
            // Tell the client *now*, with the activation tick, so it can open the
            // new connection while the old path is still live (guarantee 3).
            let redirect = Redirect {
                entity: guid,
                new_shard: to,
                shard_map: self.shard_map.clone(),
                activation_tick: activation,
            };
            // Everyone whose interest set holds the entity is redirected, not just
            // the player driving it: `Redirect::entity` is "the entity the client
            // is now interested in", so an observer that has been watching the ship
            // must be told to follow it to `to` rather than simply lose it.
            //
            // "Holds the entity" is a distance test against the one ship being
            // offered — the same `distance² <= radius²` rule the interest grid
            // applies — not a full sphere query per client, which walked ~9 000
            // grid cells for every client for every offer.
            let radius_sq =
                f64::from(self.config.interest_radius) * f64::from(self.config.interest_radius);
            let watchers: Vec<u64> = self
                .clients
                .values()
                .filter(|client| {
                    let anchor = client
                        .ship
                        .and_then(|g| self.sim.ship_state(g))
                        .map_or(WorldPos::ZERO, |s| s.pos);
                    WorldPos::distance_squared(anchor, ship_pos) <= radius_sq
                })
                .map(|client| client.session)
                .collect();
            for session in watchers {
                if let Some(client) = self.clients.get_mut(&session) {
                    client.redirect_pending = Some(to);
                    client.redirect_deadline = Some(tick.saturating_add(REDIRECT_TIMEOUT_TICKS));
                }
                self.send_to(
                    Destination::Client(session),
                    Message::Redirect(redirect.clone()),
                );
            }
        }
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
        let mut ready: Vec<EntityGuid> = self
            .ledger
            .outgoing()
            .iter()
            .filter(|(_, e)| is_ready(e))
            // A member of a group commits only together with all the others.
            .filter(|(_, e)| {
                self.ledger
                    .group_members(e.group)
                    .iter()
                    .all(|m| self.ledger.outgoing().get(m).is_some_and(is_ready))
            })
            .map(|(g, _)| *g)
            .collect();
        ready.sort_unstable();
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
            let visible = self.interest.visible(anchor, radius);
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
                },
            );
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
        assert!(shard.set_scene(state(Scene::Solar, 4)));
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
            Some(&&Message::SceneUpdate(state(Scene::Solar, 4))),
            "the scene comes first"
        );
        assert!(
            matches!(mine.get(1), Some(Message::Snapshot(_))),
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
        assert!(shard.set_scene(state(Scene::Solar, 4)));
        let (a, b) = (join(&mut shard, "a"), join(&mut shard, "b"));
        // Both announcements are lost on the way to the clients.
        let lost = scene_updates(&mut shard);
        assert_eq!(lost.len(), 2, "precondition: the join announced the scene");

        let mut told: Vec<(SimTick, u64, SceneState)> = Vec::new();
        for _ in 0..(2 * SCENE_REANNOUNCE_TICKS) {
            let tick = shard.tick();
            let mut frames = 0;
            for out in shard.take_outbox() {
                match (out.dest, out.message) {
                    (Destination::Client(session), Message::SceneUpdate(scene)) => {
                        assert_eq!(frames, 0, "the announcement comes before the frames");
                        told.push((tick, session, scene));
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
        let solar = state(Scene::Solar, 4);
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
            state(Scene::Solar, 2),
            state(Scene::Solar, 1),
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

        shard.orchestrator_message(Message::SceneSet(state(Scene::Solar, 4)));
        let (confirmed, relayed) = drain(&mut shard);
        assert_eq!(
            confirmed,
            vec![SceneAck {
                shard: ShardId::new(1),
                applied: state(Scene::Solar, 4)
            }]
        );
        assert_eq!(relayed, vec![(session, state(Scene::Solar, 4))]);

        // The same message again (the orchestrator re-sends until it hears an
        // ack): confirmed again, relayed to nobody.
        shard.orchestrator_message(Message::SceneSet(state(Scene::Solar, 4)));
        let (confirmed, relayed) = drain(&mut shard);
        assert_eq!(confirmed.len(), 1);
        assert_eq!(confirmed[0].applied, state(Scene::Solar, 4));
        assert!(relayed.is_empty());

        // A late, older one: the answer names the newer scene the shard runs.
        shard.orchestrator_message(Message::SceneSet(state(Scene::Galaxy, 3)));
        let (confirmed, relayed) = drain(&mut shard);
        assert_eq!(confirmed[0].applied, state(Scene::Solar, 4));
        assert!(relayed.is_empty());
        assert_eq!(shard.scene(), state(Scene::Solar, 4));

        assert_eq!(shard.epoch(), epoch, "a scene is not an epoch");
        assert_eq!(shard.shard_map(), &map);
        assert!(!shard.epoch_participant().has_staged());
    }
}
