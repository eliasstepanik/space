//! Seamless handoff: moving an entity between shards without the client seeing it.
//!
//! A player ship that flies out of this shard's octree region must end up
//! simulated by the shard that owns the region it flew into, and the player must
//! not notice: no teleport, no dropped inputs, no duplicate ship, no window where
//! nobody is simulating them. The three-phase exchange in [`space_protocol`]
//! ([`HandoffOffer`] → [`HandoffAccept`] → [`HandoffCommit`], with
//! [`HandoffAbort`] as the escape hatch) is what buys that, and this module is the
//! state machine that drives it on one shard.
//!
//! The same exchange moves **every physics body**, not only ships: a crate,
//! a ball, a door or a whole chain whose home ([`crate::border`]) is another
//! shard — because it left the region, or because it is in an island of
//! bodies consolidated on that shard — is offered, accepted and committed
//! the same way. What differs for an **unowned body** (no client drives it)
//! is only what a player needs:
//!
//! * **No client gate.** Nobody has to connect anywhere, so nobody is
//!   redirected, nobody adopts it, and the commit waits for the accept only
//!   (an outgoing body transfer is born `client_connected`). Its activation
//!   lead is [`crate::border::BODY_HANDOFF_LEAD`] (the offer/accept round
//!   trip), not [`HANDOFF_LEAD`].
//! * **The whole entity moves.** The offer and the commit carry its full
//!   replicated component list ([`SimWorld::export_entity`]: `Body`,
//!   `Visual`, `Springs`, `Articulation`, `PartStates`, `PosResidual`), and the
//!   receiver restores it with [`SimWorld::import_entity`] — an articulated
//!   object (a chain, a door, a seesaw) is one entity with all its parts and
//!   joints, so it moves atomically, and its position residual makes the
//!   motion continue exactly. The receiver checks the list
//!   ([`SimWorld::check_import`]) before it accepts, while a refusal still
//!   costs nothing.
//! * **Clients keep seeing it.** At the commit the offering shard turns its
//!   copy into a **ghost** of the receiver's body ([`crate::border`]), so its
//!   clients see no gap; the receiver replaces the ghost it already showed
//!   with the real body. Neither side's clients change connection.
//! * **Groups.** The members of one island are offered together, with one
//!   activation tick, and committed together only when every member is ready
//!   (accepted, and connected for a player's ship); an abort of any member
//!   aborts the group. The receiver installs what one sender committed for
//!   one activation tick on one local tick.
//!
//! A private component of the offer's state ([`TRANSFER_COMPONENT`]) tells
//! the receiver which kind of transfer it is and the lead in use, the same way
//! the owner's token already rides there; the message layouts are unchanged.
//!
//! # The guarantees, and the mechanism behind each
//!
//! | # | guarantee | how it holds |
//! |---|-----------|--------------|
//! | 1 | **The guid never changes.** The client's own ship object survives the move; nothing it holds has to be re-keyed. | [`EntityGuid`] is `(minting shard, serial)` and is *never* rewritten on handoff ([`space_protocol::EntityGuid`]). The receiving shard adopts the existing guid instead of minting a new one — a body's exactly as a ship's. |
//! | 2 | **Exactly one shard simulates the entity at every tick.** No double-step (which would double the physics) and no gap (which would freeze the ship). | The offering shard keeps stepping the entity until the activation tick; the accepting shard **stages** the state outside its [`SimWorld`] and only spawns it at commit. An entity is therefore in at most one world at any tick. A **ghost** of it — the receiver's before the install, the sender's after the commit — is a kinematic copy that is never integrated ([`SimWorld::insert_ghost`]), so it is not a second simulator; the install replaces the receiver's ghost. |
//! | 3 | **The client keeps receiving from the old shard until the new one is proven up** (make-before-break). | The [`Redirect`] tells the ship's player — and nobody else: a bystander needs no second connection, it keeps seeing the ship as this shard's ghost of it — the activation tick *in advance*, once the accepting shard has accepted (so the transfer is staged there before the client can arrive: its hello matches the staged owner ([`Incoming::is_owned_by`]) and it is given no ship of its own); the old shard keeps sending deltas until it commits. The commit waits for [`NeighbourStatus`] with `connected = true`, so a client that cannot reach the new shard never loses its feed. An unowned body has no client to move: the clients watching it stay where they are and keep seeing it, as a ghost on the side it left and as a body on the side it entered. |
//! | 4 | **The state the new shard activates with is the state at the activation tick.** | The commit carries the *final* state as of `activation_tick` ([`HandoffCommit::state`]), replacing what the offer staged. Staging a stale snapshot and then only telling the peer "go" would leave the entity jumping backwards by the offer→activation distance. For a body it is the full component list, parts, joints, springs and position residual included. |
//! | 5 | **Inputs are never lost or applied twice.** | Inputs are tick-keyed ([`SimWorld::apply_inputs`]), so everything the client sent up to activation is applied by the old shard, and the new shard's `input_ack` starts at the activation tick — the client's `input_ack`-based reconciliation drops exactly what the old shard already applied. (An unowned body takes no inputs.) |
//! | 6 | **Any failure leaves the world exactly as it was.** | Every timeout and refusal path is an [`HandoffAbort`]: the offering shard keeps simulating and the accepting shard drops its staged copy. An abort of one member of a group aborts the whole group, so an island is never left half moved by a failure. **What the client sees of a failure** depends on when it happened. Before the accept (a refused or unanswered offer) it is told nothing: no redirect was sent. After it, the client has been redirected and may hold a second connection, on which it has given nothing up: the accepting shard ends it when the adoption has not come within `ADOPT_TIMEOUT_TICKS` — with a `JoinRedirect` or a `Reject(Starting)` as its last word on a shard that does not own the spawn point, which the client takes as the end of that attempt; on the spawn point's owner with a ship of the client's own on the spawn point, which the client recognises as not the ship it flies and leaves at once (the ship goes with the session). A hello that arrives after the staged copy was dropped is answered the same two ways. Either way the client keeps playing on the old shard, which offers the ship again after its pause. So the *world* is as it was; a session, and on the spawn point's owner a ship, may have existed on the accepting shard for a moment. |
//! | 7 | **A handoff cannot straddle a shard-map change.** | Offer, accept and commit all carry the [`epoch`](HandoffOffer::epoch); a message under a stale epoch is answered with [`StaleEpoch`], an in-flight transfer whose epoch no longer matches the shard's own map aborts with [`HandoffReason::EpochChanged`] the moment the tick pass sees it (before *and* after the commit gate), and a commit under a stale epoch is refused. |
//! | 8 | **It finishes in bounded time.** | [`HANDOFF_TIMEOUT_TICKS`] bounds **every** waiting state on **both** sides: the offering shard's sweep terminates an outgoing transfer whatever it is waiting on (offer unanswered, commit waiting on the client, a commit that can no longer be built), and the accepting shard's sweep drops a staged transfer whose commit never arrived, so a dead peer degrades to "nothing happened" rather than a frozen entity or a permanently poisoned guid. A body's lead ([`crate::border::BODY_HANDOFF_LEAD`]) is well inside that budget; a staged body (no session to wait for) is dropped only after twice the budget, so never while the sender may still commit it — its commit would be the only copy left. |
//!
//! Where the table is checked: `tests/acceptance.rs` (ships, H1–H8, the
//! mapping is in its module docs) and `tests/physics.rs` (unowned bodies and
//! groups: `an_unowned_body_handoff_keeps_h1_to_h8`,
//! `an_unanswered_body_offer_times_out_and_nothing_moved`,
//! `a_refused_body_offer_leaves_the_body_where_it_was`, the island tests).
//! # Roles
//!
//! The same ledger plays both sides: a shard can be *offering* one entity while
//! *accepting* another. The two halves never interact — an outgoing transfer is
//! keyed by the entity it is giving up, an incoming one by the entity it is
//! taking on, and an entity can only be in one of the two at a time.
//!
//! # Transport
//!
//! The state machine is synchronous and produces messages into the shard's
//! outbox ([`crate::shard::Shard::send_to_shard`]); the quinn front end
//! ([`crate::net`]) delivers them. That is what lets the acceptance tests run two
//! shards in one process and exchange the outboxes by hand.

use std::collections::HashMap;

use space_protocol::{
    EntitySnapshot, HandoffAbort, HandoffAccept, HandoffCommit, HandoffOffer, HandoffReason,
    ShardId,
};
use space_sim::SimTick;

/// The private component id a handoff state carries its transfer kind under:
/// `[version = 1, kind, lead]`, `kind` 0 = a player's ship, 1 = an unowned
/// body; `lead` the activation lead in ticks the sender used. Outside the
/// registry's id space (like the owner token at `u16::MAX`), never sent to a
/// client, stripped before the state is imported.
pub const TRANSFER_COMPONENT: u16 = u16::MAX - 1;

/// The private component id a handoff state carries the **scene time** it
/// describes under (`i64` µs, big endian): the sending shard's
/// `SimWorld::scene_time` when it took the state. The receiving shard's own
/// scene clock may differ by a few milliseconds (each shard keeps its own
/// mapping onto the wall clock), which near a moon moving thousands of
/// kilometres a second is kilometres: the receiver moves the state by what
/// the orbiting bodies carry it from that time to its own
/// (`SimWorld::carry`), so it arrives where it was relative to the planet it
/// is near. Never sent to a client, stripped before the state is imported.
/// Private ids are [`PRIVATE_COMPONENTS`] and up.
pub const SCENE_TIME_COMPONENT: u16 = u16::MAX - 2;

/// The lowest private component id of a handoff state.
pub const PRIVATE_COMPONENTS: u16 = SCENE_TIME_COMPONENT;

/// The [`SCENE_TIME_COMPONENT`] for scene time `us`.
#[must_use]
pub fn scene_time_component(us: i64) -> (u16, Vec<u8>) {
    (SCENE_TIME_COMPONENT, us.to_be_bytes().to_vec())
}

/// The scene time a handoff state carries, if it carries one.
#[must_use]
pub fn scene_time_of(state: &EntitySnapshot) -> Option<i64> {
    state
        .components
        .iter()
        .find(|(id, _)| *id == SCENE_TIME_COMPONENT)
        .and_then(|(_, b)| <[u8; 8]>::try_from(b.as_slice()).ok())
        .map(i64::from_be_bytes)
}

/// What a transfer moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferKind {
    /// A ship a client drives: redirected, adopted, client-gated.
    Ship,
    /// Anything else (a body, or a ship nobody drives): no client gate,
    /// installed with `SimWorld::import_entity`.
    Body,
}

impl TransferKind {
    /// The private component describing this kind with activation `lead`.
    #[must_use]
    pub fn component(self, lead: SimTick) -> (u16, Vec<u8>) {
        let kind = match self {
            Self::Ship => 0,
            Self::Body => 1,
        };
        (
            TRANSFER_COMPONENT,
            vec![1, kind, u8::try_from(lead).unwrap_or(u8::MAX)],
        )
    }

    /// The kind and lead a handoff state names: a state without the
    /// component is a ship with [`HANDOFF_LEAD`] (what every handoff was
    /// before bodies moved); a malformed one is `None`.
    #[must_use]
    pub fn of(state: &EntitySnapshot) -> Option<(Self, SimTick)> {
        let Some((_, bytes)) = state
            .components
            .iter()
            .find(|(id, _)| *id == TRANSFER_COMPONENT)
        else {
            return Some((Self::Ship, HANDOFF_LEAD));
        };
        match bytes.as_slice() {
            [1, 0, lead] => Some((Self::Ship, SimTick::from(*lead).min(HANDOFF_LEAD))),
            [1, 1, lead] => Some((Self::Body, SimTick::from(*lead).min(HANDOFF_LEAD))),
            _ => None,
        }
    }
}

/// Ticks between an offer and the activation tick it proposes.
///
/// This is the window the client gets to open its connection to the accepting
/// shard before the transfer must complete. One second at 30 Hz is generous for a
/// QUIC handshake on the same host and for a WAN round trip, and short enough that
/// a handoff does not sit in flight across a shard's whole tick budget.
pub const HANDOFF_LEAD: SimTick = 30;

/// Ticks a handoff may wait in any state before it is abandoned.
///
/// [`HANDOFF_LEAD`] already covers the happy path, so this only ever fires for a
/// peer that stopped answering — which is exactly when the offering shard must
/// stop waiting and keep simulating (guarantee 6).
pub const HANDOFF_TIMEOUT_TICKS: SimTick = 2 * space_sim::TICK_HZ as SimTick;

/// An offer this shard sent and is waiting on.
#[derive(Debug)]
pub struct Outgoing {
    /// Shard we offered to.
    pub to: ShardId,
    /// Epoch the offer was made under.
    pub epoch: u64,
    /// Tick the transfer takes effect.
    pub activation_tick: SimTick,
    /// Tick we sent the offer, for the timeout.
    pub started_at: SimTick,
    /// The accepting shard answered [`HandoffAccept`].
    pub accepted: bool,
    /// The client confirmed its connection to `to` ([`space_protocol::NeighbourStatus`]).
    /// Born `true` for a transfer with no client ([`TransferKind::Body`]).
    pub client_connected: bool,
    /// What moves.
    pub kind: TransferKind,
    /// The group (island) the transfer belongs to; 0 = on its own.
    pub group: u64,
}

/// A state another shard offered us and we have staged.
#[derive(Debug)]
pub struct Incoming {
    /// Shard that offered it.
    pub from: ShardId,
    /// Epoch the accept was sent under.
    pub epoch: u64,
    /// Tick the transfer takes effect **on the offering shard's clock**.
    ///
    /// Carried for the wire only: the two shards run independent tick loops and
    /// drift, so this shard never compares it against its own [`SimWorld::tick`].
    /// Activation is decided by [`Incoming::local_activation_tick`].
    pub activation_tick: SimTick,
    /// The tick **this** shard was at when it accepted, plus [`HANDOFF_LEAD`]:
    /// the activation tick translated onto this shard's own clock (see
    /// [`Ledger::accept_sent`]). This — never [`Incoming::activation_tick`] — is
    /// what the install gate compares against [`SimWorld::tick`].
    pub local_activation_tick: SimTick,
    /// Tick we accepted, for the timeout.
    pub started_at: SimTick,
    /// The staged state: the offer's payload, replaced by the commit's final one.
    ///
    /// It deliberately lives **outside** [`SimWorld`] until commit — that is what
    /// makes guarantee 2 (one authority per tick) structural rather than a matter
    /// of remembering not to step it.
    pub state: EntitySnapshot,
    /// The owner's player token, taken from the offer's state.
    ///
    /// A client may only take over a staged entity when its own hello token
    /// matches this, which is what stops a player that guessed the guid from
    /// claiming somebody else's ship.
    pub owner_token: String,
    /// The client session that confirmed its connection here, once it has.
    pub session: Option<u64>,
    /// What moves.
    pub kind: TransferKind,
}

impl Incoming {
    /// Whether the staged entity is the ship of the player whose hello token
    /// is `player_token`: the one client that may be welcomed for it on a
    /// shard that does not own the spawn point, adopt it, and open or cancel
    /// its commit gate.
    ///
    /// Only a **ship** transfer with a **non-empty** owner token has such a
    /// player. A body — and a ship nobody drives — is staged with the empty
    /// token ("no owner"), and the empty string must not be read as a token
    /// that an empty `player_token` equals: that made a hello without a token
    /// the "owner" of every body in flight. The compare itself is constant
    /// time, as for every secret ([`space_net::constant_time_eq`]).
    #[must_use]
    pub fn is_owned_by(&self, player_token: &str) -> bool {
        self.kind == TransferKind::Ship
            && !self.owner_token.is_empty()
            && space_net::constant_time_eq(player_token.as_bytes(), self.owner_token.as_bytes())
    }
}

/// The ledger of in-flight transfers on one shard.
#[derive(Debug, Default)]
pub struct Ledger {
    /// Transfers we offered, keyed by entity.
    outgoing: HashMap<space_protocol::EntityGuid, Outgoing>,
    /// Transfers we accepted, keyed by entity.
    incoming: HashMap<space_protocol::EntityGuid, Incoming>,
}

impl Ledger {
    /// An empty ledger.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Transfers we are offering, for the timeout sweep.
    #[must_use]
    pub fn outgoing(&self) -> &HashMap<space_protocol::EntityGuid, Outgoing> {
        &self.outgoing
    }

    /// Transfers we have staged, for the commit pass.
    #[must_use]
    pub fn incoming(&self) -> &HashMap<space_protocol::EntityGuid, Incoming> {
        &self.incoming
    }

    /// Whether `entity` is staged here, waiting for its commit.
    #[must_use]
    pub fn has_incoming(&self, entity: space_protocol::EntityGuid) -> bool {
        self.incoming.contains_key(&entity)
    }

    /// Whether `entity` is in any transfer at all.
    #[must_use]
    pub fn is_busy(&self, entity: space_protocol::EntityGuid) -> bool {
        self.outgoing.contains_key(&entity) || self.incoming.contains_key(&entity)
    }

    /// Record that we sent `offer` (a player's ship, on its own) at `now`.
    pub fn offer_sent(&mut self, offer: &HandoffOffer, now: SimTick) {
        self.offer_sent_as(offer, now, TransferKind::Ship, 0);
    }

    /// Record that we sent `offer` of `kind`, in `group` (0: none), at `now`.
    pub fn offer_sent_as(
        &mut self,
        offer: &HandoffOffer,
        now: SimTick,
        kind: TransferKind,
        group: u64,
    ) {
        self.outgoing.insert(
            offer.entity,
            Outgoing {
                to: offer.to,
                epoch: offer.epoch,
                activation_tick: offer.activation_tick,
                started_at: now,
                accepted: false,
                client_connected: kind == TransferKind::Body,
                kind,
                group,
            },
        );
    }

    /// The members of outgoing group `group` (empty for 0), ascending.
    #[must_use]
    pub fn group_members(&self, group: u64) -> Vec<space_protocol::EntityGuid> {
        if group == 0 {
            return Vec::new();
        }
        let mut out: Vec<_> = self
            .outgoing
            .iter()
            .filter(|(_, e)| e.group == group)
            .map(|(g, _)| *g)
            .collect();
        out.sort_unstable();
        out
    }

    /// Record that we staged `accept`'s matching offer at `now`.
    ///
    /// `now` is **this** shard's tick at the moment of the accept, and it is what
    /// the timeout is measured from. The activation tick is translated onto this
    /// shard's clock the same way: the two shards run independent tick loops and
    /// drift, so `accept.activation_tick` (the *offering* shard's clock) is kept
    /// only for the wire, and [`Incoming::local_activation_tick`] — `now` plus
    /// [`HANDOFF_LEAD`] — is what this shard installs at.
    pub fn accept_sent(
        &mut self,
        accept: &HandoffAccept,
        state: EntitySnapshot,
        owner_token: String,
        now: SimTick,
    ) {
        self.accept_sent_as(
            accept,
            state,
            owner_token,
            now,
            TransferKind::Ship,
            HANDOFF_LEAD,
        );
    }

    /// [`Ledger::accept_sent`] for a transfer of `kind` whose sender used
    /// activation `lead`. The local activation tick is `now + lead` — or, if
    /// this sender already has a transfer staged here for the same activation
    /// tick (one island, offered together), that transfer's local tick, so a
    /// group is installed on one tick.
    pub fn accept_sent_as(
        &mut self,
        accept: &HandoffAccept,
        state: EntitySnapshot,
        owner_token: String,
        now: SimTick,
        kind: TransferKind,
        lead: SimTick,
    ) {
        let local_activation_tick = self
            .incoming
            .values()
            .find(|e| e.from == accept.from && e.activation_tick == accept.activation_tick)
            .map_or(now.saturating_add(lead), |e| e.local_activation_tick);
        self.incoming.insert(
            accept.entity,
            Incoming {
                from: accept.from,
                epoch: accept.epoch,
                activation_tick: accept.activation_tick,
                local_activation_tick,
                started_at: now,
                state,
                owner_token,
                session: None,
                kind,
            },
        );
    }

    /// Mark an outgoing transfer accepted. Returns `false` when no such transfer
    /// is in flight (a stale or forged accept).
    pub fn mark_accepted(&mut self, accept: &HandoffAccept) -> bool {
        match self.outgoing.get_mut(&accept.entity) {
            Some(entry)
                if entry.to == accept.to && entry.activation_tick == accept.activation_tick =>
            {
                entry.accepted = true;
                true
            }
            _ => false,
        }
    }

    /// Mark that the client confirmed its connection to the accepting shard.
    /// Returns the transfer's shard on success.
    pub fn mark_client_connected(
        &mut self,
        entity: space_protocol::EntityGuid,
        shard: ShardId,
    ) -> Option<ShardId> {
        let entry = self.outgoing.get_mut(&entity)?;
        if entry.to != shard {
            return None;
        }
        entry.client_connected = true;
        Some(shard)
    }

    /// Record the session that confirmed its connection for a staged entity.
    pub fn mark_session(
        &mut self,
        entity: space_protocol::EntityGuid,
        session: u64,
    ) -> Option<&Incoming> {
        self.incoming.get_mut(&entity)?.session = Some(session);
        self.incoming.get(&entity)
    }

    /// Take a staged transfer, removing it from the ledger.
    pub fn take_incoming(&mut self, entity: space_protocol::EntityGuid) -> Option<Incoming> {
        self.incoming.remove(&entity)
    }

    /// Take a staged transfer only if it was staged by `from`.
    ///
    /// A commit must come from the shard that actually staged the transfer; a peer
    /// that merely *claims* a valid neighbour id must not be able to install an
    /// entity another shard staged. `None` when nothing is staged or it was staged
    /// by a different shard (the entry is left in place in that case).
    pub fn take_incoming_from(
        &mut self,
        entity: space_protocol::EntityGuid,
        from: ShardId,
    ) -> Option<Incoming> {
        if self.incoming.get(&entity)?.from == from {
            self.incoming.remove(&entity)
        } else {
            None
        }
    }

    /// Take an outgoing transfer, removing it from the ledger.
    pub fn take_outgoing(&mut self, entity: space_protocol::EntityGuid) -> Option<Outgoing> {
        self.outgoing.remove(&entity)
    }

    /// The activation tick of an outgoing transfer, if one is in flight.
    #[must_use]
    pub fn activation_of(&self, entity: space_protocol::EntityGuid) -> Option<SimTick> {
        self.outgoing.get(&entity).map(|e| e.activation_tick)
    }

    /// Build the abort message for an outgoing transfer and drop it.
    pub fn abort_outgoing(
        &mut self,
        entity: space_protocol::EntityGuid,
        from: ShardId,
        reason: HandoffReason,
    ) -> Option<HandoffAbort> {
        let entry = self.outgoing.remove(&entity)?;
        Some(HandoffAbort {
            entity,
            from,
            to: entry.to,
            reason,
        })
    }

    /// Build the abort message an incoming transfer deserves and drop it.
    pub fn abort_incoming(
        &mut self,
        entity: space_protocol::EntityGuid,
        to: ShardId,
        reason: HandoffReason,
    ) -> Option<HandoffAbort> {
        let entry = self.incoming.remove(&entity)?;
        Some(HandoffAbort {
            entity,
            from: entry.from,
            to,
            reason,
        })
    }

    /// Build a commit for a transfer that is ready: accepted **and** the client's
    /// new connection is proven. `state` is the entity's state *as of the
    /// activation tick*, which the caller reads at the moment it calls this.
    pub fn ready_commit(
        &mut self,
        entity: space_protocol::EntityGuid,
        from: ShardId,
        epoch: u64,
        state: EntitySnapshot,
    ) -> Option<HandoffCommit> {
        let entry = self.outgoing.get(&entity)?;
        if !entry.accepted || !entry.client_connected {
            return None;
        }
        if entry.epoch != epoch {
            return None;
        }
        let activation_tick = entry.activation_tick;
        let to = entry.to;
        self.outgoing.remove(&entity);
        Some(HandoffCommit {
            entity,
            from,
            to,
            epoch,
            activation_tick,
            state: Box::new(state),
        })
    }

    /// The offer message for `entity`, as the offering shard would send it.
    #[must_use]
    pub fn offer(
        entity: space_protocol::EntityGuid,
        from: ShardId,
        to: ShardId,
        epoch: u64,
        now: SimTick,
        state: EntitySnapshot,
    ) -> HandoffOffer {
        Self::offer_with_lead(entity, from, to, epoch, now, state, HANDOFF_LEAD)
    }

    /// [`Ledger::offer`] with an activation lead of `lead` ticks.
    #[must_use]
    pub fn offer_with_lead(
        entity: space_protocol::EntityGuid,
        from: ShardId,
        to: ShardId,
        epoch: u64,
        now: SimTick,
        state: EntitySnapshot,
        lead: SimTick,
    ) -> HandoffOffer {
        HandoffOffer {
            entity,
            from,
            to,
            epoch,
            state: Box::new(state),
            activation_tick: now.saturating_add(lead),
        }
    }

    /// The accept message matching `offer`.
    #[must_use]
    pub fn accept_for(offer: &HandoffOffer) -> HandoffAccept {
        HandoffAccept {
            entity: offer.entity,
            from: offer.from,
            to: offer.to,
            epoch: offer.epoch,
            activation_tick: offer.activation_tick,
        }
    }
}

#[cfg(test)]
mod owner_tests {
    use super::{Incoming, TransferKind};
    use space_protocol::{EntityGuid, EntitySnapshot, ShardId};

    fn staged(kind: TransferKind, owner_token: &str) -> Incoming {
        Incoming {
            from: ShardId(1),
            epoch: 1,
            activation_tick: 30,
            local_activation_tick: 30,
            started_at: 0,
            state: EntitySnapshot::new(EntityGuid::new(ShardId(1), 1), Vec::new()),
            owner_token: owner_token.into(),
            session: None,
            kind,
        }
    }

    /// A staged transfer belongs to a player only if it is a ship **and**
    /// carries an owner token, and then only to the player with that token.
    ///
    /// Mutation this catches: `staged.owner_token == player_token` (the
    /// empty token of an ownerless transfer equals an empty hello token), or
    /// dropping the kind (a body staged under some token is taken for a
    /// player's ship, and its "owner" welcomed off the spawn shard).
    #[test]
    fn only_a_ship_with_an_owner_token_has_an_owner() {
        let ship = staged(TransferKind::Ship, "the-players-token");
        assert!(ship.is_owned_by("the-players-token"));
        assert!(!ship.is_owned_by("another-token"));
        assert!(!ship.is_owned_by(""));
        assert!(!ship.is_owned_by("the-players-token-and-more"));
        // Nobody flies it: nobody's, and above all not the empty token's.
        let stray = staged(TransferKind::Ship, "");
        assert!(!stray.is_owned_by(""));
        assert!(!stray.is_owned_by("the-players-token"));
        // A body has no player, whatever its state claims.
        let body = staged(TransferKind::Body, "");
        assert!(!body.is_owned_by(""));
        let labelled = staged(TransferKind::Body, "the-players-token");
        assert!(!labelled.is_owned_by("the-players-token"));
    }
}

#[cfg(test)]
mod scene_time_tests {
    use super::{scene_time_component, scene_time_of, PRIVATE_COMPONENTS, TRANSFER_COMPONENT};
    use space_protocol::{EntityGuid, EntitySnapshot, ShardId};

    /// The scene time a handoff state carries comes back exactly, a state
    /// without one has none, and the id is private (stripped before import).
    #[test]
    fn a_handoff_state_carries_its_scene_time() {
        let guid = EntityGuid::new(ShardId(1), 1);
        let mut state = EntitySnapshot::new(guid, vec![(0, vec![1, 2, 3])]);
        assert_eq!(scene_time_of(&state), None);
        state.components.push(scene_time_component(-42));
        state
            .components
            .push(scene_time_component(1_763_900_000_123_456));
        assert_eq!(scene_time_of(&state), Some(-42), "the first one counts");
        let (id, _) = scene_time_component(0);
        assert!((PRIVATE_COMPONENTS..TRANSFER_COMPONENT).contains(&id));
        let bad = EntitySnapshot::new(guid, vec![(id, vec![1, 2])]);
        assert_eq!(scene_time_of(&bad), None, "a malformed one is ignored");
    }
}
