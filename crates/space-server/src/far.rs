//! Far replicas: what the other shards simulate in view of this shard's
//! players, and what this shard simulates in view of theirs
//! (`docs/plan-sharding.md` §6; the wire and the distance steps:
//! [`space_protocol::far`]).
//!
//! A shard shows its clients three kinds of entity, and a guid is exactly one
//! of them here at any time:
//!
//! | | where it is | what it is for |
//! |---|---|---|
//! | **own** | in the simulation, stepped | everything |
//! | **ghost** | in the simulation, kinematic (`border`) | exact contact across a border, and seeing |
//! | **far replica** | in no world: the interest grid and the component log only | seeing |
//!
//! The precedence is that order — own before ghost before far: a
//! [`FarUpdate`] never touches a guid that is in this shard's world, and a
//! guid that enters the world (a handoff arrives, a ghost update names it)
//! stops being a far replica in the same moment, **in place**: its entry in
//! the component log goes on, so a client is sent the difference and keeps
//! the entity's history. The other way — a ghost its owner withdraws because
//! it left the border margin — becomes a far replica where it stands
//! ([`Shard::ghost_to_far`]) and is confirmed by its owner's next
//! [`FarUpdate`] or dropped [`FAR_ORPHAN_TICKS`] later.
//!
//! # Who sends what
//!
//! **Shard to shard, relayed to the client by its own shard** — a client keeps
//! one play connection. Every [`FAR_INTEREST_TICKS`] a shard tells every
//! other shard of its map where its players are ([`FarInterest`]; the players
//! whose ships are on their way to it included, so the view is there before
//! the player is). A shard answers each with [`FarUpdate`]s: every ship and
//! body it simulates that it does not already send that shard as a ghost and
//! that lies within [`far::FAR_RANGE`] of one of that shard's players — at
//! the rate of the distance step of the **nearest** of them, and only when it
//! changed. The receiving shard thins that again for each of its clients by
//! that client's own distance (`replicate`). All of it rides the peer
//! connections the handoff uses, behind the cluster's peer token; there is no
//! new listener and no new credential.
//!
//! (The alternative — a client that connects to every shard in view — was not
//! built: up to eight connections a client, a second authentication path for
//! shards that players only watch, and the same entity arriving on two
//! connections with nothing but the client to settle which one counts.)
//!
//! # What a far replica carries
//!
//! What it takes to draw it and no more ([`FAR_COMPONENTS`]): the motion
//! (`ShipState`, with its velocities — the client interpolates sparse samples
//! along them), the shape and the model (`Body`, `Visual`: a player's ship
//! carries the player model), `Springs`, `Articulation` and `PartStates`. Not
//! the `Owner` (a session of another shard), not the `ShipControls`, not the
//! sub-millimetre `PosResidual` — and nothing private ever: the owner's token
//! exists only in a handoff's state. The receiver applies the same list to
//! what it is sent.
//!
//! Each sample keeps **the tick it is the state of**: the sender's tick, which
//! is the cluster's. The receiver hands it to its clients with the entity
//! ([`space_protocol::Origin`]) and does not move the replica between
//! samples; the client interpolates on that timeline.
//!
//! # Soft state, and what it costs
//!
//! * An entity the receiver does not hold comes in full, once; afterwards
//!   only what changed, at most every step's spacing. **An entity at rest is
//!   not sent again.**
//! * Every [`FAR_REFRESH_TICKS`] the sender names what the receiver should
//!   hold — twelve bytes an entity, the keep-alive. What the receiver holds
//!   beyond that goes; what it lacks it asks for in its next [`FarInterest`]
//!   and is sent in full (a message to a peer can be dropped: the queue was
//!   full, the peer was being dialled).
//! * A shard that has sent nothing for [`FAR_SILENCE_TICKS`], or that the map
//!   no longer names — once the minute is over in which a shard that was
//!   merged away is still heard, see [`Shard::knows_shard`] — loses its far
//!   replicas here; a scene body of another
//!   scene revision is never kept.
//!
//! So a peer link costs, per second, at most
//! `sum over the moving far entities of (one sample x the rate of the step of
//! the receiver's nearest player)` plus `12 B x held / 2 s`, in messages of
//! at most [`far::MAX_FAR_UPDATE_BYTES`]; what does not fit a message waits a
//! tick. `tests/far.rs` holds both the client's and the peer's bound to
//! numbers.
//!
//! # What a peer may do
//!
//! A far message is taken only from a shard of the map, on its authenticated
//! peer connection, and only as that shard (`from` is checked against the
//! connection). It is bounded on receipt: a [`FarUpdate`] of more than
//! [`far::MAX_FAR_UPDATE_BYTES`], more than [`FAR_RECEIVE_BYTES`] or
//! [`FAR_UPDATES`] of them a second, more than [`FAR_INTERESTS`] interests a
//! second, an interest naming
//! more than [`far::MAX_FAR_PLAYERS`] players, an entity that would be the
//! [`MAX_FAR_PER_PEER`]`+1`th of its sender — each is dropped, counted
//! ([`ShardStats::far_refused`](super::ShardStats::far_refused)) and said in
//! a few log lines a window. A guid this shard simulates, or holds a ghost
//! of, cannot be written through a far message at all. Among far replicas a
//! shard may take over one another shard sent only with a newer sample (an
//! entity handed from one neighbour to another), so a peer of the cluster
//! could show this shard's clients a wrong picture of a third shard's distant
//! entity — and nothing else: a far replica is in no solver and no handoff.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::time::Duration;

use space_ecs::{ComponentDef, ComponentId};
use space_net::LogBudget;
use space_protocol::far::{self, FarInterest, FarPlayer, FarUpdate, Origin, ORIGIN_COMPONENT};
use space_protocol::{EntityGuid, EntitySnapshot, Message, ShardId};
use space_sim::{ShipState, SimTick};
use space_world::WorldPos;

use super::physics::{component, diff, merge, replicated_of};
use super::{Destination, Shard};
use crate::border;
use crate::handoff::TransferKind;
use crate::replicate::{self, Components};

/// A shard says where its players are this often, ticks.
pub const FAR_INTEREST_TICKS: SimTick = 10;
/// A shard whose last word on its players is older than this is sent
/// nothing, ticks.
pub const FAR_INTEREST_STALE_TICKS: SimTick = 3 * space_sim::TICK_HZ as SimTick;
/// The names of everything a peer should hold go out this often, ticks: the
/// keep-alive of a far replica at rest.
pub const FAR_REFRESH_TICKS: SimTick = 2 * space_sim::TICK_HZ as SimTick;
/// A shard silent for this many ticks loses its far replicas here.
pub const FAR_SILENCE_TICKS: SimTick = 3 * space_sim::TICK_HZ as SimTick;
/// A far replica its sender has withdrawn (or a ghost that was) is kept this
/// long for whoever simulates the entity now to claim it, ticks.
pub const FAR_ORPHAN_TICKS: SimTick = 6;
/// A sample's tick is taken as at most this far behind the receiver's own,
/// ticks (and never ahead of its next): what a peer says cannot put a sample
/// anywhere else on a client's timeline.
pub const FAR_MAX_LAG: SimTick = 8;
/// The most far replicas kept from one shard.
pub const MAX_FAR_PER_PEER: usize = 4096;
/// The most bytes of [`FarUpdate`]s taken from one shard a second.
pub const FAR_RECEIVE_BYTES: usize = 512 * 1024;
/// The most [`FarInterest`]s taken from one shard a second.
pub const FAR_INTERESTS: u32 = 9;
/// The most [`FarUpdate`]s taken from one shard a second, whatever their
/// size: twice what a shard sends at its busiest (an update and the two
/// chunks of a refresh of [`MAX_FAR_PER_PEER`] names a tick). The byte bound
/// alone let a peer make this shard decode and walk half a megabyte of empty
/// updates a second — tens of thousands of them.
pub const FAR_UPDATES: u32 = 6 * space_sim::TICK_HZ;
/// The most bytes of components one far replica may have.
pub const MAX_FAR_ENTITY_BYTES: usize = 16 * 1024;
/// What a sender fills one [`FarUpdate`] to, bytes (the first entity always
/// goes; the receiver's bound is [`far::MAX_FAR_UPDATE_BYTES`]).
const FAR_UPDATE_BUDGET: usize = far::MAX_FAR_UPDATE_BYTES * 3 / 4;
/// What one removal costs in a [`FarUpdate`], bytes.
const REMOVAL_BYTES: usize = 12;
/// The names one chunk of a refresh carries.
const FAR_REFRESH_CHUNK: usize = 2048;
/// Log lines about refused far messages per [`FAR_REFUSAL_WINDOW`].
const FAR_REFUSAL_LINES: u32 = 3;
const FAR_REFUSAL_WINDOW: Duration = Duration::from_secs(10);

const SHIP_STATE: ComponentId = <ShipState as ComponentDef>::ID;

/// The components a far replica has: what a client draws it from (see the
/// module docs, "What a far replica carries").
pub const FAR_COMPONENTS: [ComponentId; 6] = [
    SHIP_STATE,
    <space_sim::Body as ComponentDef>::ID,
    <space_sim::Visual as ComponentDef>::ID,
    <space_sim::Springs as ComponentDef>::ID,
    <space_sim::Articulation as ComponentDef>::ID,
    <space_sim::PartStates as ComponentDef>::ID,
];

/// `list` cut down to [`FAR_COMPONENTS`] (and to components that are there:
/// no empty payloads).
fn far_components(list: &[(ComponentId, Vec<u8>)]) -> Components {
    list.iter()
        .filter(|(id, bytes)| FAR_COMPONENTS.contains(id) && !bytes.is_empty())
        .cloned()
        .collect()
}

/// A far replica held here.
#[derive(Debug)]
struct FarEntry {
    /// The shard that simulates the entity (and sent this).
    from: ShardId,
    /// Its components ([`FAR_COMPONENTS`]) as last received.
    components: Components,
    /// The tick `components` are the state of.
    sample_tick: SimTick,
    /// Its sender withdrew it at this tick: kept for [`FAR_ORPHAN_TICKS`].
    orphaned: Option<SimTick>,
}

/// What a peer was last sent of one of our entities.
#[derive(Debug)]
struct FarSent {
    components: Components,
    /// The tick it was last sent at.
    at: SimTick,
    /// Its distance step for that peer (its nearest player).
    step: u8,
}

/// What a peer last said about its players.
#[derive(Debug, Default)]
struct PeerInterest {
    players: Vec<WorldPos>,
    heard: SimTick,
    /// What it asked for again.
    missing: Vec<EntityGuid>,
    /// Look at every entity for this peer on the next pass (its players
    /// moved; an entity came, or stopped being its ghost).
    rescan: bool,
}

/// What was taken from a peer in the current second.
#[derive(Debug, Default, Clone, Copy)]
struct Received {
    since: SimTick,
    bytes: usize,
    interests: u32,
    updates: u32,
}

/// The far half of a shard's state.
#[derive(Debug)]
pub(super) struct FarState {
    /// The far replicas held here.
    held: HashMap<EntityGuid, FarEntry>,
    /// Per peer: where its players are.
    interest: HashMap<ShardId, PeerInterest>,
    /// Per peer: the far replicas it holds of ours.
    sent: HashMap<ShardId, HashMap<EntityGuid, FarSent>>,
    /// Per peer: what it told us to hold and we do not.
    missing: HashMap<ShardId, BTreeSet<EntityGuid>>,
    /// Per peer: the tick a [`FarUpdate`] last arrived.
    heard: HashMap<ShardId, SimTick>,
    /// Per peer: the names of a refresh that is arriving in chunks.
    refreshing: HashMap<ShardId, Vec<EntityGuid>>,
    /// Per peer: what it has sent this second.
    received: HashMap<ShardId, Received>,
    /// Replicas whose new sample the component log has not seen yet.
    dirty: Vec<EntityGuid>,
    /// Look at every entity for every peer on the next pass.
    pub(super) rescan: bool,
    lines: LogBudget,
}

impl Default for FarState {
    fn default() -> Self {
        Self {
            held: HashMap::new(),
            interest: HashMap::new(),
            sent: HashMap::new(),
            missing: HashMap::new(),
            heard: HashMap::new(),
            refreshing: HashMap::new(),
            received: HashMap::new(),
            dirty: Vec::new(),
            rescan: false,
            lines: LogBudget::new(FAR_REFUSAL_LINES, FAR_REFUSAL_WINDOW),
        }
    }
}

impl FarState {
    /// How many far replicas are held.
    pub(super) fn len(&self) -> usize {
        self.held.len()
    }

    /// `guid` has entered this shard's world (a ghost of it, or the entity
    /// itself): it is no far replica any more. The interest grid and the
    /// component log are the world's from here on and are left alone.
    pub(super) fn claimed_by_the_world(&mut self, guid: EntityGuid) {
        self.held.remove(&guid);
    }
}

/// One far replica as [`Shard::far_list`] reports it.
#[derive(Debug, Clone, PartialEq)]
pub struct FarInfo {
    /// Its guid.
    pub guid: EntityGuid,
    /// The shard that simulates it.
    pub from: ShardId,
    /// Its motion as last received.
    pub state: ShipState,
    /// The tick that is the state of.
    pub sample_tick: SimTick,
}

impl Shard {
    // ----------------------------------------------------------- tools, tests

    /// The far replicas held here, by guid.
    #[must_use]
    pub fn far_list(&self) -> Vec<FarInfo> {
        let mut out: Vec<FarInfo> = self
            .far
            .held
            .iter()
            .filter_map(|(guid, entry)| {
                let state = component(&entry.components, SHIP_STATE)
                    .and_then(|b| space_sim::wire::decode_ship_state(b).ok())?;
                Some(FarInfo {
                    guid: *guid,
                    from: entry.from,
                    state,
                    sample_tick: entry.sample_tick,
                })
            })
            .collect();
        out.sort_by_key(|info| info.guid);
        out
    }

    /// How many of this shard's entities `peer` holds as far replicas.
    #[must_use]
    pub fn far_sent_to(&self, peer: ShardId) -> usize {
        self.far.sent.get(&peer).map_or(0, HashMap::len)
    }

    // ------------------------------------------------------------ the log

    /// Hand the component log the far replicas whose sample is new (called
    /// with the tick's frames about to be built, so the change carries that
    /// tick like any other).
    pub(super) fn record_far(&mut self, tick: SimTick) {
        for guid in std::mem::take(&mut self.far.dirty) {
            let Some(entry) = self.far.held.get(&guid) else {
                continue;
            };
            if self.sim.entity_of(guid).is_some() {
                continue;
            }
            let mut list = entry.components.clone();
            list.push((
                ORIGIN_COMPONENT,
                Origin::far(entry.from, entry.sample_tick).encode(),
            ));
            list.sort_by_key(|(id, _)| *id);
            self.log.record_sampled(guid, list, tick, entry.sample_tick);
        }
    }

    /// Stop holding the far replica `guid`.
    fn drop_far(&mut self, guid: EntityGuid) {
        if self.far.held.remove(&guid).is_some() && self.sim.entity_of(guid).is_none() {
            self.interest.remove(guid);
            self.log.forget(guid);
        }
    }

    /// The ghost `guid` was withdrawn by its owner (it left the border
    /// margin, or is no longer that shard's): it becomes a far replica where
    /// it is, for [`FAR_ORPHAN_TICKS`] — the shard that simulates it now
    /// confirms it with its next [`FarUpdate`], if a player of ours can see
    /// it. Its history on our clients goes on either way.
    pub(super) fn ghost_to_far(&mut self, guid: EntityGuid) {
        let from = self.ghosts.get(&guid).map(|entry| entry.from);
        let list = replicated_of(&self.sim, &self.ghosts, guid);
        let now = self.sim.tick();
        match (from, list) {
            (Some(from), Some(list)) if self.sim.is_ghost(guid) => {
                let components = far_components(&list);
                let pos = component(&components, SHIP_STATE)
                    .and_then(|b| space_sim::wire::decode_ship_state(b).ok())
                    .map(|s| s.pos);
                self.drop_ghost(guid, false);
                let Some(pos) = pos else {
                    self.log.forget(guid);
                    return;
                };
                self.interest.upsert_far(guid, pos);
                // (Its owner has just spoken — the ghost frame that withdrew
                // it — whether or not a far update of it was ever heard.)
                self.far.heard.insert(from, now);
                self.far.held.insert(
                    guid,
                    FarEntry {
                        from,
                        components,
                        sample_tick: now,
                        orphaned: Some(now),
                    },
                );
                self.far.dirty.push(guid);
                self.far.rescan = true;
            }
            _ => self.remove_ghost(guid),
        }
    }

    // ------------------------------------------------------------- receive

    /// Count a far message of `from` that was not taken, and say so (a few
    /// lines a window).
    fn far_refuse(&mut self, from: ShardId, why: &str) {
        self.stats.far_refused += 1;
        if let Some(held_back) = self.far.lines.admit() {
            eprintln!(
                "space-server: {} dropped a far message of shard {from}: {why}{}",
                self.config.shard,
                space_net::held_back_note(held_back)
            );
        }
    }

    /// The account of what `from` has sent this second.
    fn far_received(&mut self, from: ShardId) -> &mut Received {
        let now = self.sim.tick();
        let entry = self.far.received.entry(from).or_default();
        if now.saturating_sub(entry.since) >= space_sim::TICK_HZ as SimTick {
            *entry = Received {
                since: now,
                ..Received::default()
            };
        }
        entry
    }

    /// A peer said where its players are.
    pub(super) fn on_far_interest(&mut self, from: ShardId, interest: FarInterest) {
        if interest.from != from {
            return self.far_refuse(from, "it names another shard");
        }
        if interest.players.len() > far::MAX_FAR_PLAYERS
            || interest.missing.len() > far::MAX_FAR_MISSING
        {
            return self.far_refuse(from, "an interest that names too much");
        }
        let account = self.far_received(from);
        account.interests += 1;
        if account.interests > FAR_INTERESTS {
            return self.far_refuse(from, "more interests than a second takes");
        }
        let now = self.sim.tick();
        let entry = self.far.interest.entry(from).or_default();
        entry.players = interest
            .players
            .iter()
            .filter_map(FarPlayer::position)
            .collect();
        entry.heard = now;
        entry.rescan = true;
        for guid in interest.missing {
            if entry.missing.len() < far::MAX_FAR_MISSING {
                entry.missing.push(guid);
            }
        }
    }

    /// A peer's far replicas arrived.
    pub(super) fn on_far_update(&mut self, from: ShardId, update: FarUpdate) {
        if update.from != from {
            return self.far_refuse(from, "it names another shard");
        }
        let bytes = update.wire_len();
        if bytes > far::MAX_FAR_UPDATE_BYTES {
            return self.far_refuse(from, "an update over the size bound");
        }
        let account = self.far_received(from);
        account.updates += 1;
        if account.updates > FAR_UPDATES {
            return self.far_refuse(from, "more updates than a second takes");
        }
        account.bytes += bytes;
        if account.bytes > FAR_RECEIVE_BYTES {
            return self.far_refuse(from, "more bytes than a second takes");
        }
        self.stats.far_bytes_in += bytes as u64;
        let now = self.sim.tick();
        self.far.heard.insert(from, now);
        // The tick the states are of, held to where a sample of a shard on
        // our clock can be.
        let sample_tick = update.tick.clamp(now.saturating_sub(FAR_MAX_LAG), now + 1);
        let mut count = self.far.held.values().filter(|e| e.from == from).count();
        for entity in update.new {
            self.far_accept(from, entity, true, sample_tick, &mut count);
        }
        for entity in update.changed {
            self.far_accept(from, entity, false, sample_tick, &mut count);
        }
        for guid in update.removed {
            if let Some(entry) = self.far.held.get_mut(&guid) {
                if entry.from == from && entry.orphaned.is_none() {
                    entry.orphaned = Some(now);
                }
            }
        }
        if update.refresh {
            let names = self.far.refreshing.entry(from).or_default();
            let room = MAX_FAR_PER_PEER.saturating_sub(names.len());
            names.extend(update.held.into_iter().take(room));
            if !update.more {
                let names: HashSet<EntityGuid> = self
                    .far
                    .refreshing
                    .remove(&from)
                    .unwrap_or_default()
                    .into_iter()
                    .collect();
                self.far_refreshed(from, &names);
            }
        }
    }

    /// Take one entity of a [`FarUpdate`] (see the module docs for what is
    /// refused).
    fn far_accept(
        &mut self,
        from: ShardId,
        entity: EntitySnapshot,
        full: bool,
        sample_tick: SimTick,
        count: &mut usize,
    ) {
        let guid = entity.guid;
        if border::is_stale_scene_body(guid, self.scene.revision) {
            return;
        }
        // In our world — ours, or a ghost: the nearer copy counts, and no
        // peer writes it through this door.
        if self.sim.entity_of(guid).is_some() {
            return;
        }
        let now = self.sim.tick();
        let held = self.far.held.get(&guid);
        let new_here = held.is_none_or(|h| h.from != from);
        if let Some(other) = held.filter(|h| h.from != from) {
            // Another shard's replica: taken over once that shard has let
            // go of it, or with a newer sample (the entity changed hands).
            let alive = self
                .far
                .heard
                .get(&other.from)
                .is_some_and(|at| now.saturating_sub(*at) <= FAR_SILENCE_TICKS);
            if alive && other.orphaned.is_none() && sample_tick <= other.sample_tick {
                return;
            }
        }
        let components = match (full, held) {
            (true, _) => far_components(&entity.components),
            (false, Some(held)) => far_components(&merge(&held.components, &entity.components)),
            (false, None) => {
                // A change of something we do not hold: ask for all of it.
                let missing = self.far.missing.entry(from).or_default();
                if missing.len() < far::MAX_FAR_MISSING {
                    missing.insert(guid);
                }
                return;
            }
        };
        let size: usize = components.iter().map(|(_, bytes)| bytes.len()).sum();
        if size > MAX_FAR_ENTITY_BYTES {
            return self.far_refuse(from, "an entity over the size bound");
        }
        let Some(state) = component(&components, SHIP_STATE)
            .and_then(|b| space_sim::wire::decode_ship_state(b).ok())
        else {
            return; // nothing to place it by
        };
        if new_here {
            if *count >= MAX_FAR_PER_PEER {
                return self.far_refuse(from, "more far replicas than one shard may have here");
            }
            *count += 1;
        }
        self.interest.upsert_far(guid, state.pos);
        self.far.held.insert(
            guid,
            FarEntry {
                from,
                components,
                sample_tick,
                orphaned: None,
            },
        );
        self.far.dirty.push(guid);
    }

    /// `from` named everything we should hold of it: drop the rest, and note
    /// what is not here.
    fn far_refreshed(&mut self, from: ShardId, names: &HashSet<EntityGuid>) {
        let gone: Vec<EntityGuid> = self
            .far
            .held
            .iter()
            .filter(|(guid, entry)| entry.from == from && !names.contains(guid))
            .map(|(guid, _)| *guid)
            .collect();
        for guid in gone {
            self.drop_far(guid);
        }
        let revision = self.scene.revision;
        for guid in names {
            if self.far.held.contains_key(guid)
                || self.sim.entity_of(*guid).is_some()
                || border::is_stale_scene_body(*guid, revision)
            {
                continue;
            }
            let missing = self.far.missing.entry(from).or_default();
            if missing.len() < far::MAX_FAR_MISSING {
                missing.insert(*guid);
            }
        }
    }

    // ---------------------------------------------------------------- send

    /// Where this shard's players are: the ships its clients fly, and the
    /// ships that are on their way here for a client (staged, or committed
    /// and not yet installed) where their ghosts are — a player's view is
    /// here before the player is.
    fn far_players(&self) -> Vec<WorldPos> {
        let mut out: Vec<WorldPos> = self
            .clients
            .values()
            .filter_map(|client| client.ship)
            .filter_map(|ship| self.sim.state_of(ship))
            .map(|state| state.pos)
            .collect();
        let arriving = self
            .ledger
            .incoming()
            .iter()
            .filter(|(_, staged)| staged.kind == TransferKind::Ship)
            .map(|(guid, _)| *guid)
            .chain(
                self.pending_commits
                    .iter()
                    .filter(|p| p.kind == TransferKind::Ship)
                    .map(|p| p.guid),
            );
        for guid in arriving {
            if let Some(state) = self.sim.state_of(guid) {
                out.push(state.pos);
            }
        }
        out.truncate(far::MAX_FAR_PLAYERS);
        out
    }

    /// The shards far replicas are exchanged with: every other shard of the
    /// map, and every shard the map named until a moment ago
    /// ([`Shard::knows_shard`]) — a shard whose region was merged away goes
    /// on simulating what it has not handed over yet, with players on it,
    /// for up to a minute. Left out, what it still simulates vanished from
    /// every other shard's players the moment the map changed (their far
    /// replicas of it were dropped as "a shard the map no longer names"),
    /// and its own players lost sight of everything beyond the ghosts.
    fn far_neighbours(&self) -> Vec<ShardId> {
        let mut out = border::neighbours(&self.shard_map, self.config.shard);
        let own = self.config.shard;
        out.extend(self.former_shards().filter(|shard| *shard != own));
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The far pass of a tick: forget what has gone stale, say where our
    /// players are, and send every peer what its players should see of ours.
    pub(super) fn far_pass(&mut self, tick: SimTick) {
        let neighbours = self.far_neighbours();
        self.far_expire(tick, &neighbours);
        if neighbours.is_empty() {
            return;
        }
        if tick.is_multiple_of(FAR_INTEREST_TICKS) {
            let players: Vec<FarPlayer> =
                self.far_players().into_iter().map(FarPlayer::at).collect();
            for n in &neighbours {
                let missing: Vec<EntityGuid> = self
                    .far
                    .missing
                    .remove(n)
                    .unwrap_or_default()
                    .into_iter()
                    .collect();
                let interest = FarInterest {
                    from: self.config.shard,
                    tick,
                    players: players.clone(),
                    missing,
                };
                self.stats.far_bytes_out += interest.wire_len() as u64;
                self.send_to(Destination::Shard(*n), Message::FarInterest(interest));
            }
        }
        let rescan = std::mem::take(&mut self.far.rescan);
        let wanted = neighbours.iter().any(|n| {
            self.far
                .interest
                .get(n)
                .is_some_and(|peer| !peer.players.is_empty())
                || self.far.sent.get(n).is_some_and(|sent| !sent.is_empty())
        });
        if !wanted {
            return;
        }
        let locals: Vec<(EntityGuid, ShipState)> = self
            .sim
            .bodies()
            .into_iter()
            .filter(|(_, _, ghost)| !ghost)
            .filter_map(|(guid, _, _)| Some((guid, self.sim.ship_state(guid)?)))
            .collect();
        let alive: HashSet<EntityGuid> = locals.iter().map(|(guid, _)| *guid).collect();
        for n in neighbours {
            self.far_send(n, tick, rescan, &locals, &alive);
        }
    }

    /// Drop the far replicas that are nobody's any more (see the module
    /// docs, "Soft state"), and what is kept about shards the map no longer
    /// names.
    fn far_expire(&mut self, tick: SimTick, neighbours: &[ShardId]) {
        let revision = self.scene.revision;
        let gone: Vec<EntityGuid> = self
            .far
            .held
            .iter()
            .filter(|(guid, entry)| {
                self.sim.entity_of(**guid).is_some()
                    || !neighbours.contains(&entry.from)
                    || border::is_stale_scene_body(**guid, revision)
                    || entry
                        .orphaned
                        .is_some_and(|at| tick.saturating_sub(at) > FAR_ORPHAN_TICKS)
                    || self
                        .far
                        .heard
                        .get(&entry.from)
                        .is_none_or(|heard| tick.saturating_sub(*heard) > FAR_SILENCE_TICKS)
            })
            .map(|(guid, _)| *guid)
            .collect();
        for guid in gone {
            self.drop_far(guid);
        }
        let far = &mut self.far;
        far.interest.retain(|shard, _| neighbours.contains(shard));
        far.sent.retain(|shard, _| neighbours.contains(shard));
        far.missing.retain(|shard, _| neighbours.contains(shard));
        far.heard.retain(|shard, _| neighbours.contains(shard));
        far.refreshing.retain(|shard, _| neighbours.contains(shard));
        far.received.retain(|shard, _| neighbours.contains(shard));
    }

    /// Send `n` what its players should see of ours (see the module docs,
    /// "Who sends what").
    fn far_send(
        &mut self,
        n: ShardId,
        tick: SimTick,
        rescan: bool,
        locals: &[(EntityGuid, ShipState)],
        alive: &HashSet<EntityGuid>,
    ) {
        let mut sent = self.far.sent.remove(&n).unwrap_or_default();
        let Some(peer) = self
            .far
            .interest
            .get_mut(&n)
            .filter(|peer| tick.saturating_sub(peer.heard) <= FAR_INTEREST_STALE_TICKS)
        else {
            // Nobody there to tell (its own silence rule drops what it holds).
            return;
        };
        let scan = rescan || std::mem::take(&mut peer.rescan);
        for guid in std::mem::take(&mut peer.missing) {
            sent.remove(&guid);
        }
        let players = peer.players.clone();
        let nearest = |pos: WorldPos| {
            players
                .iter()
                .map(|player| WorldPos::distance(*player, pos))
                .fold(f64::INFINITY, f64::min)
        };
        let mut update = FarUpdate {
            from: self.config.shard,
            tick,
            new: Vec::new(),
            changed: Vec::new(),
            removed: Vec::new(),
            refresh: false,
            held: Vec::new(),
            more: false,
        };
        // What stopped being ours (handed on, despawned) is no longer ours
        // to show. Removals are budgeted like everything else in the
        // message — twelve bytes each: what does not fit stays in `sent` and
        // is said on the next tick. (Unbudgeted, 2 700 removals at once — a
        // peer's player leaving the range of a crowded shard — made one
        // message over the receiver's bound, which drops it whole.)
        let mut gone: Vec<EntityGuid> = sent
            .keys()
            .filter(|guid| !alive.contains(guid))
            .copied()
            .collect();
        gone.sort_unstable();
        gone.truncate(FAR_UPDATE_BUDGET / REMOVAL_BYTES);
        let mut bytes = gone.len() * REMOVAL_BYTES;
        update.removed = gone;
        let ghosted = self.ghost_sent.get(&n);
        for (guid, state) in locals {
            let member = sent.get(guid).map(|held| (held.at, held.step));
            match member {
                None if !scan => continue,
                Some((at, step)) if !scan && tick.saturating_sub(at) < far::interval(step) => {
                    continue;
                }
                _ => {}
            }
            // A ghost on `n` shows it already (and exactly).
            if ghosted.is_some_and(|ghosts| ghosts.contains_key(guid)) {
                sent.remove(guid);
                continue;
            }
            let distance = nearest(state.pos);
            let range = if member.is_some() {
                far::FAR_RANGE * far::STEP_HYSTERESIS
            } else {
                far::FAR_RANGE
            };
            if distance > range {
                if member.is_some() && bytes + REMOVAL_BYTES <= FAR_UPDATE_BUDGET {
                    bytes += REMOVAL_BYTES;
                    sent.remove(guid);
                    update.removed.push(*guid);
                }
                continue;
            }
            let step = match member {
                Some((_, step)) => far::step_after(distance, step),
                None => far::step_for(distance),
            };
            if let Some(held) = sent.get_mut(guid) {
                held.step = step;
                if tick.saturating_sub(held.at) < far::interval(step) {
                    continue;
                }
            }
            let Some(list) = replicate::replicated_components(&self.sim, *guid) else {
                continue;
            };
            let list = far_components(&list);
            let entry = match sent.get(guid) {
                Some(held) => {
                    let mut changed = diff(&held.components, &list);
                    if changed.is_empty() {
                        continue; // at rest: nothing to say
                    }
                    if component(&changed, SHIP_STATE).is_none() {
                        if let Some(state) = component(&list, SHIP_STATE) {
                            changed.push((SHIP_STATE, state.to_vec()));
                            changed.sort_by_key(|(id, _)| *id);
                        }
                    }
                    EntitySnapshot::new(*guid, changed)
                }
                None => EntitySnapshot::new(*guid, list.clone()),
            };
            let size = entry.encoded_len();
            if bytes > 0 && bytes + size > FAR_UPDATE_BUDGET {
                continue; // the next tick's message
            }
            bytes += size;
            if member.is_some() {
                update.changed.push(entry);
            } else {
                update.new.push(entry);
            }
            sent.insert(
                *guid,
                FarSent {
                    components: list,
                    at: tick,
                    step,
                },
            );
        }
        for guid in &update.removed {
            sent.remove(guid);
        }
        if !(update.new.is_empty() && update.changed.is_empty() && update.removed.is_empty()) {
            self.stats.far_bytes_out += update.wire_len() as u64;
            self.send_to(Destination::Shard(n), Message::FarUpdate(update));
        }
        // The keep-alive: the names of what `n` should hold (staggered by
        // shard, so not every link refreshes on one tick).
        if (tick + SimTick::from(n.0)).is_multiple_of(FAR_REFRESH_TICKS) && !sent.is_empty() {
            let mut names: Vec<EntityGuid> = sent.keys().copied().collect();
            names.sort_unstable();
            let chunks: Vec<&[EntityGuid]> = names.chunks(FAR_REFRESH_CHUNK).collect();
            let last = chunks.len().saturating_sub(1);
            for (i, chunk) in chunks.into_iter().enumerate() {
                let refresh = FarUpdate {
                    from: self.config.shard,
                    tick,
                    new: Vec::new(),
                    changed: Vec::new(),
                    removed: Vec::new(),
                    refresh: true,
                    held: chunk.to_vec(),
                    more: i < last,
                };
                self.stats.far_bytes_out += refresh.wire_len() as u64;
                self.send_to(Destination::Shard(n), Message::FarUpdate(refresh));
            }
        }
        self.far.sent.insert(n, sent);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A far replica is cut down to what draws it, on the way out and on the
    /// way in.
    ///
    /// Mutation this catches: letting `Owner` (a session id of another
    /// shard), `ShipControls`, `PosResidual` or a private handoff component
    /// through.
    #[test]
    fn a_far_replica_carries_only_what_draws_it() {
        let list: Components = vec![
            (0, vec![1; 76]),
            (1, vec![2; 25]),
            (2, vec![3; 8]),
            (4, vec![4; 30]),
            (5, vec![5; 5]),
            (8, Vec::new()),
            (10, vec![6; 14]),
            (ORIGIN_COMPONENT, vec![7; 14]),
            (u16::MAX, b"token".to_vec()),
            (u16::MAX - 2, vec![9; 9]),
        ];
        let kept: Vec<ComponentId> = far_components(&list).iter().map(|(id, _)| *id).collect();
        assert_eq!(kept, vec![0, 4, 5]);
    }
}
