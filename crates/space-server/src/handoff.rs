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
//! # The guarantees, and the mechanism behind each
//!
//! | # | guarantee | how it holds |
//! |---|-----------|--------------|
//! | 1 | **The guid never changes.** The client's own ship object survives the move; nothing it holds has to be re-keyed. | [`EntityGuid`] is `(minting shard, serial)` and is *never* rewritten on handoff ([`space_protocol::EntityGuid`]). The receiving shard adopts the existing guid instead of minting a new one. |
//! | 2 | **Exactly one shard simulates the entity at every tick.** No double-step (which would double the physics) and no gap (which would freeze the ship). | The offering shard keeps stepping the entity until the activation tick; the accepting shard **stages** the state outside its [`SimWorld`] and only spawns it at commit. An entity is therefore in at most one world at any tick. |
//! | 3 | **The client keeps receiving from the old shard until the new one is proven up** (make-before-break). | The [`Redirect`] tells the client the activation tick *in advance*; the old shard keeps sending deltas until it commits. The commit waits for [`NeighbourStatus`] with `connected = true`, so a client that cannot reach the new shard never loses its feed. |
//! | 4 | **The state the new shard activates with is the state at the activation tick.** | The commit carries the *final* state as of `activation_tick` ([`HandoffCommit::state`]), replacing what the offer staged. Staging a stale snapshot and then only telling the peer "go" would leave the entity jumping backwards by the offer→activation distance. |
//! | 5 | **Inputs are never lost or applied twice.** | Inputs are tick-keyed ([`SimWorld::apply_inputs`]), so everything the client sent up to activation is applied by the old shard, and the new shard's `input_ack` starts at the activation tick — the client's `input_ack`-based reconciliation drops exactly what the old shard already applied. |
//! | 6 | **Any failure leaves the world exactly as it was.** | Every timeout and refusal path is an [`HandoffAbort`]: the offering shard keeps simulating, the accepting shard drops its staged copy, and the client is never told anything happened. |
//! | 7 | **A handoff cannot straddle a shard-map change.** | Offer, accept and commit all carry the [`epoch`](HandoffOffer::epoch); a message under a stale epoch is answered with [`StaleEpoch`], an in-flight transfer whose epoch no longer matches the shard's own map aborts with [`HandoffReason::EpochChanged`] the moment the tick pass sees it (before *and* after the commit gate), and a commit under a stale epoch is refused. |
//! | 8 | **It finishes in bounded time.** | [`HANDOFF_TIMEOUT_TICKS`] bounds **every** waiting state on **both** sides: the offering shard's sweep terminates an outgoing transfer whatever it is waiting on (offer unanswered, commit waiting on the client, a commit that can no longer be built), and the accepting shard's sweep drops a staged transfer whose commit never arrived, so a dead peer degrades to "nothing happened" rather than a frozen entity or a permanently poisoned guid. |
//!
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
    pub client_connected: bool,
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

    /// Record that we sent `offer` at `now`.
    pub fn offer_sent(&mut self, offer: &HandoffOffer, now: SimTick) {
        self.outgoing.insert(
            offer.entity,
            Outgoing {
                to: offer.to,
                epoch: offer.epoch,
                activation_tick: offer.activation_tick,
                started_at: now,
                accepted: false,
                client_connected: false,
            },
        );
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
        self.incoming.insert(
            accept.entity,
            Incoming {
                from: accept.from,
                epoch: accept.epoch,
                activation_tick: accept.activation_tick,
                local_activation_tick: now.saturating_add(HANDOFF_LEAD),
                started_at: now,
                state,
                owner_token,
                session: None,
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
        HandoffOffer {
            entity,
            from,
            to,
            epoch,
            state: Box::new(state),
            activation_tick: now.saturating_add(HANDOFF_LEAD),
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
