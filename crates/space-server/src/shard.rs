//! [`Shard`]: the state one tick of the shard loop drives.
//!
//! Everything here is synchronous and deterministic: given the same sequence of
//! [`Shard::join`] / [`Shard::apply_input`] / [`Shard::peer_message`] calls and
//! the same number of [`Shard::tick`] calls, two shards produce the same
//! [`SimWorld::guid_state_hash`]. The quinn front end ([`crate::net`]) is the
//! only async part, and it only moves messages in and out. That split is what
//! lets the acceptance tests run two shards inside one process and hand their
//! outboxes to each other.

use std::collections::HashMap;

use space_protocol::{
    EntityGuid, HandoffReason, Hello, InputTick, Message, NeighbourStatus, PlayerInput, Redirect,
    RejectReason, ShardId, ShardMap, ShardRegion, StaleEpoch, Welcome,
};
use space_sim::{ShipState, SimTick, SimWorld};
use space_world::{NodeRef, WorldPos};

use crate::config::{ShardConfig, ADOPT_TIMEOUT_TICKS, REDIRECT_TIMEOUT_TICKS};
use crate::epoch::EpochParticipant;
use crate::handoff::{Ledger, HANDOFF_TIMEOUT_TICKS};
use crate::replicate::{self, ClientReplica, Frame, GuidIndex, Interest};

/// Hard cap on concurrent client sessions per shard (docs/plan.md U9).
pub const MAX_PLAYERS_PER_SHARD: usize = 100;

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
    outbox: Vec<OutMessage>,
    /// Ships whose commit went out this tick; despawned once the tick's frames
    /// are built (see [`Shard::tick`] step 7).
    departing: Vec<EntityGuid>,
    /// The shard map this shard publishes and follows.
    shard_map: ShardMap,
    /// The shard's epoch-protocol participation (Phase 3): the staged prepare,
    /// if one is outstanding.
    epoch_participant: EpochParticipant,
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
            outbox: Vec::new(),
            departing: Vec::new(),
            shard_map: map,
            epoch_participant: EpochParticipant::new(),
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
    pub fn set_shard_map(&mut self, map: ShardMap) {
        self.shard_map = map;
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
            Some(self.spawn_ship_at_region_center(session))
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
        let guid = EntityGuid::new(self.config.shard, self.next_serial);
        self.next_serial += 1;
        let entity = self.sim.spawn_ship(guid, session, ShipState::at_rest(pos));
        self.guids.insert(guid, entity);
        self.interest.upsert(guid, entity, pos);
        guid
    }

    /// Spawn a ship at the centre of this shard's region, owned by `session`.
    fn spawn_ship_at_region_center(&mut self, session: u64) -> EntityGuid {
        self.spawn_ship_for(session, self.config.region.center())
    }

    /// Despawn `guid` and drop it from every local index.
    pub fn despawn_ship(&mut self, guid: EntityGuid) {
        if let Some(entity) = self.guids.entity_of(guid) {
            self.guids.remove(guid);
            self.interest.remove(guid);
            self.sim.despawn_ship(guid);
            let _ = entity;
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
        // The client's view of what it has applied: never let the replica fall
        // behind it, or a delta would be built against a tick the client has
        // already moved past. **Clamped to our own tick**: a client claiming a
        // far-future ack would otherwise push `ack_tick` past every future change
        // tick and silently stop its own replication (nothing is ever
        // "changed since" a tick that has not happened).
        let ack = batch.ack_tick.min(self.sim.tick());
        if client.replica.ack_tick < ack {
            client.replica.ack_tick = ack;
        }
        // Rewrite every input to this session's ship. This is the `entity_of`
        // trust boundary: the guid that reaches the simulation is ours, whatever
        // the client sent. An awaiting-adopt client has no ship yet, so its
        // inputs are dropped — the old shard is still simulating its entity.
        let Some(ship) = client.ship else {
            return;
        };
        for input in &batch.inputs {
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
        if status.connected {
            // Adopt: the client reporting that it reached us *for this entity* is
            // the player that entity belongs to — **but only if its hello token is
            // the token the offer carried the owner under**. A guid is
            // `(shard, serial)` with serials starting at 1, so it is trivially
            // guessable; the token is what makes the claim non-forgeable. Without
            // the match, a second client could name somebody else's staged ship
            // and be handed its `Owner` stamp and its inputs.
            let claim_is_authorized =
                self.ledger
                    .incoming()
                    .get(&status.entity)
                    .is_some_and(|staged| {
                        self.clients
                            .get(&session)
                            .is_some_and(|client| client.player_token == staged.owner_token)
                    });
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
        } else if let Some(abort) = self.ledger.abort_incoming(
            status.entity,
            status.shard,
            HandoffReason::ClientNotConnected,
        ) {
            self.stats.aborted += 1;
            self.send_to(Destination::Shard(abort.from), Message::HandoffAbort(abort));
        }
    }

    // ------------------------------------------------------------- shard-to-shard

    /// A message arrived from another shard.
    ///
    /// # Peer identity is only a map-membership check
    ///
    /// A peer connection announces itself with the hello token
    /// <code>"shard:&lt;id&gt;"</code> ([`crate::net::peer_shard_id`]), which is a
    /// string any client could type. There is **no** shard-to-shard
    /// authentication today: the transport is QUIC with a self-signed
    /// certificate (no pinning), the hello check covers the protocol version
    /// only, and the orchestrator-issued `ShardWelcome::peer_token` is never
    /// presented by shards nor verified here. The only check is that the claimed
    /// [`ShardId`] is one of the ids in **this shard's own shard map**; a peer
    /// outside it is dropped. A commit is additionally required to come from the
    /// shard that actually staged the transfer ([`Ledger::take_incoming_from`]).
    ///
    /// So a party that can reach this shard's port and guesses a *valid*
    /// neighbour's id can speak the handoff protocol as that neighbour. The map
    /// check closes "anyone can be a peer"; it does not close "anyone can
    /// impersonate a known peer". Enforcing `peer_token` is deferred (see
    /// `docs/security-notes.md`).
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
            Message::StaleEpoch(stale) => {
                // Our map is behind: any transfer we have in flight is invalid.
                let entities: Vec<EntityGuid> = self
                    .ledger
                    .outgoing()
                    .keys()
                    .copied()
                    .filter(|_| stale.current > self.epoch())
                    .collect();
                for entity in entities {
                    self.abort_outgoing(entity, HandoffReason::EpochChanged);
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
        } else if let Some(abort) = self.ledger.abort_outgoing(
            status.entity,
            self.config.shard,
            HandoffReason::ClientNotConnected,
        ) {
            self.stats.aborted += 1;
            self.send_to(Destination::Shard(from), Message::HandoffAbort(abort));
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
        // Capacity first: a shard must never be pushed past the player cap by
        // incoming handoffs. Answering `TargetFull` lets the offering shard keep
        // the ship rather than stage it into a shard that cannot take it.
        if self.clients.len() >= MAX_PLAYERS_PER_SHARD {
            let abort = space_protocol::HandoffAbort {
                entity: offer.entity,
                from: self.config.shard,
                to: offer.to,
                reason: HandoffReason::TargetFull,
            };
            self.send_to(Destination::Shard(offer.from), Message::HandoffAbort(abort));
            return;
        }
        // Two guarantee-6 failures: we cannot take an entity we are ourselves
        // moving, and we cannot take one we already hold (a duplicate offer after
        // a lost accept).
        if self.ledger.is_busy(offer.entity) || self.sim.entity_of(offer.entity).is_some() {
            let abort = space_protocol::HandoffAbort {
                entity: offer.entity,
                from: self.config.shard,
                to: offer.to,
                reason: HandoffReason::Other("entity is not available here".into()),
            };
            self.send_to(Destination::Shard(offer.from), Message::HandoffAbort(abort));
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
        self.ledger
            .accept_sent(&accept, *offer.state.clone(), owner_token, self.sim.tick());
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
        // The commit's final state replaces the staged one (guarantee 4).
        let state = decode_ship_state(&commit.state)
            .or_else(|| decode_ship_state(&staged.state))
            .unwrap_or(ShipState::IDENTITY);
        let session = staged.session.unwrap_or(0);
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
        });
    }

    fn on_abort(&mut self, abort: space_protocol::HandoffAbort) {
        // Either way an abort means "nothing happened": drop our half and keep
        // simulating whatever we have (guarantee 6).
        if self.ledger.take_outgoing(abort.entity).is_some()
            || self.ledger.take_incoming(abort.entity).is_some()
        {
            self.stats.aborted += 1;
        }
    }

    fn abort_outgoing(&mut self, entity: EntityGuid, reason: HandoffReason) {
        if let Some(abort) = self
            .ledger
            .abort_outgoing(entity, self.config.shard, reason)
        {
            self.stats.aborted += 1;
            self.send_to(Destination::Shard(abort.to), Message::HandoffAbort(abort));
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

        // 4. Refresh the interest grid from the new positions.
        for entity in self.sim.ships() {
            let Some(guid) = self.guids.guid_of(entity) else {
                continue;
            };
            if let Some(state) = self.sim.ship_state(guid) {
                self.interest.upsert(guid, entity, state.pos);
            }
        }

        // 4.5 Client-side timeouts: a redirect that was never confirmed, and an
        // awaiting-adopt client whose staged entity never materialised.
        self.client_timeout_pass(tick);

        // 5. Handoffs: start, commit, time out.
        self.handoff_pass(tick);

        // 6. Replication.
        self.replicate_pass();

        // 7. Ships handed off this tick leave the world. Deferring this past the
        // frames (rather than despawning inside the handoff pass) is what keeps the
        // client's frame clock continuous across a handoff: the accepting shard
        // cannot install the ship before the commit has been delivered, i.e. not
        // before the next tick, so if the offering shard sent nothing for the
        // commit tick the client would see a tick with no frame at all.
        for guid in std::mem::take(&mut self.departing) {
            self.despawn_ship(guid);
        }

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
            let entity = self
                .sim
                .spawn_ship(pending.guid, pending.session, pending.state);
            self.guids.insert(pending.guid, entity);
            self.interest
                .upsert(pending.guid, entity, pending.state.pos);
            self.stats.received += 1;
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
            let guid = self.spawn_ship_at_region_center(session);
            if let Some(client) = self.clients.get_mut(&session) {
                client.ship = Some(guid);
                client.replica = ClientReplica::new(guid);
                client.adopt_deadline = None;
            }
        }
    }

    /// The handoff pass: boundary crossings, commits, timeouts.
    fn handoff_pass(&mut self, tick: SimTick) {
        // --- start a transfer for every client ship that left our region.
        let mut to_offer: Vec<(EntityGuid, ShardId)> = Vec::new();
        for client in self.clients.values() {
            let Some(guid) = client.ship else {
                continue; // awaiting adopt: no ship of ours to offer
            };
            if self.ledger.is_busy(guid) {
                continue;
            }
            let Some(state) = self.sim.ship_state(guid) else {
                continue;
            };
            let owner = self.shard_map.lookup(&state.pos);
            if let Some(owner) = owner {
                if owner != self.config.shard && owner != ShardId::NONE {
                    to_offer.push((guid, owner));
                }
            }
        }
        for (guid, to) in to_offer {
            let Some(entity) = self.sim.entity_of(guid) else {
                continue;
            };
            let Some(state) = replicate::snapshot_of(&self.sim, &self.guids, entity) else {
                continue;
            };
            // Stamp the owner's token into the offer's state so the accepting
            // shard can require a claiming client's hello token to match it
            // (see [`OWNER_TOKEN_COMPONENT`]).
            let state = with_owner_token(state, &self.owner_token_of(guid));
            let offer = Ledger::offer(guid, self.config.shard, to, self.epoch(), tick, state);
            let activation = offer.activation_tick;
            self.ledger.offer_sent(&offer, tick);
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
            let watchers: Vec<u64> = self
                .clients
                .values()
                .filter(|client| {
                    self.interest
                        .visible(
                            client
                                .ship
                                .and_then(|g| self.sim.ship_state(g))
                                .map_or(WorldPos::ZERO, |s| s.pos),
                            self.config.interest_radius,
                        )
                        .contains(&guid)
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

        let ready: Vec<EntityGuid> = self
            .ledger
            .outgoing()
            .iter()
            .filter(|(_, e)| tick >= e.activation_tick && e.accepted && e.client_connected)
            .map(|(g, _)| *g)
            .collect();
        for guid in ready {
            let Some(entity) = self.sim.entity_of(guid) else {
                continue;
            };
            let Some(state) = replicate::snapshot_of(&self.sim, &self.guids, entity) else {
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
                self.send_to(
                    Destination::Shard(commit.to),
                    Message::HandoffCommit(commit),
                );
                self.departing.push(guid);
                self.stats.handed_off += 1;
            }
        }

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
        let stale_incoming: Vec<EntityGuid> = self
            .ledger
            .incoming()
            .iter()
            .filter(|(_, e)| {
                e.session.is_none() && tick.saturating_sub(e.started_at) > HANDOFF_TIMEOUT_TICKS
            })
            .map(|(g, _)| *g)
            .collect();
        for guid in stale_incoming {
            self.abort_incoming(guid, HandoffReason::CommitTimeout);
        }
    }

    /// Build and queue one frame per client.
    fn replicate_pass(&mut self) {
        let radius = self.config.interest_radius;
        let budget = self.config.frame_budget;
        let shard = self.config.shard;
        let epoch = self.epoch();
        let mut frames: Vec<(u64, Frame)> = Vec::with_capacity(self.clients.len());
        let sessions: Vec<u64> = self.clients.keys().copied().collect();
        for session in sessions {
            let Some(ship) = self.clients.get(&session).and_then(|c| c.ship) else {
                continue; // awaiting adopt: nothing to anchor a view on yet
            };
            // The anchor is the client's own ship; if it is gone (handed off and
            // despawned) the client's view is about to be replaced by a redirect
            // anyway, so nothing is sent.
            let Some(anchor) = self.sim.ship_state(ship).map(|s| s.pos) else {
                continue;
            };
            let visible = self.interest.visible(anchor, radius);
            let ack = self.clients.get(&session).map_or(0, |c| c.replica.ack_tick);
            let changed = replicate::changed_in_interest(
                &mut self.sim,
                &self.guids,
                &self.interest,
                anchor,
                radius,
                ack,
            );
            let replica = &mut self
                .clients
                .get_mut(&session)
                .expect("session just listed")
                .replica;
            frames.push((
                session,
                replicate::build_frame(
                    &mut self.sim,
                    replicate::FrameContext {
                        shard,
                        epoch,
                        budget,
                        guids: &self.guids,
                    },
                    replica,
                    &visible,
                    &changed,
                ),
            ));
        }
        for (session, frame) in frames {
            let message = match frame {
                Frame::Snapshot(snapshot) => Message::Snapshot(snapshot),
                Frame::Delta(delta) => Message::Delta(delta),
                Frame::Idle => continue,
            };
            self.send_to(Destination::Client(session), message);
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
