//! The ticked simulation world and its canonical state hash.
//!
//! [`SimWorld`] owns a [`space_ecs::World`], the [`SimTick`] counter and the
//! [`ShipParams`] every ship is stepped with. It is the unit both sides run: the
//! server steps it once per network tick with the inputs it received, the client
//! steps a mirror of it to predict (see [`crate::prediction`]).
//!
//! # Determinism rules enforced here
//!
//! * **Inputs are sorted before they are applied.** [`SimWorld::apply_inputs`]
//!   resolves every [`space_protocol::PlayerInput`] to its entity and sorts the
//!   resolved list by [`space_ecs::Entity`] (and, for two inputs naming the same
//!   entity in one tick, by the input's bit pattern — the *last* one in that
//!   order wins). The result is therefore independent of the order the inputs
//!   arrived in, which is what the acceptance test
//!   `determinism_10k_ticks` third run checks.
//! * **The hash is order-insensitive.** [`SimWorld::state_hash`] walks the
//!   entity list **sorted by** [`space_ecs::Entity`] and serialises each
//!   component in a fixed field order (see [`SimWorld::serialize_state`]), so
//!   the hash is a function of the state alone. No [`std::collections::HashMap`]
//!   iteration order can leak into it — the guid→entity map is only ever used
//!   for point lookups.
//! * **No clock.** The only time source is [`SimWorld::tick`], advanced by
//!   [`SimWorld::step`].

use std::collections::{BTreeMap, HashMap};

use space_ecs::{Component, ComponentDef, Entity, WithEntity, World};
use space_protocol::{EntityGuid, PlayerInput, ShardId};

use crate::components::{Owner, ShardAuthority, ShipControls, ShipState, COMPONENT_REGISTRY};
use crate::params::ShipParams;
use crate::physics::step_ship;
use crate::tick::SimTick;

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// How far ahead of the current tick [`SimWorld::apply_inputs`] will buffer an
/// input, in ticks.
///
/// An input tagged more than this many ticks in the future is dropped. Without a
/// bound, a client that sends `tick = u64::MAX` (or floods future-tick inputs)
/// grows `input_buf` without limit — a trivial memory DoS on an authoritative
/// shard. 30 ticks is one second at [`crate::tick::TICK_HZ`], comfortably more
/// than any realistic client/server clock skew or one-way latency, so a genuine
/// input is never dropped by this rule; only a malformed or hostile one is.
pub const MAX_INPUT_LEAD: SimTick = 30;

/// FNV-1a over a byte slice, starting from `hash`.
fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Append the big-endian bytes of the integer `v` to `out`.
///
/// Big endian is chosen so the byte order of the hash input matches the pinned
/// wire config in [`space_protocol::codec`] (big endian, fixed-width integers):
/// one byte order to reason about across the whole project.
fn put_int(out: &mut Vec<u8>, v: i128, width: usize) {
    let bytes = v.to_be_bytes();
    out.extend_from_slice(&bytes[16 - width..]);
}

/// Append the big-endian bits of `f` to `out`.
///
/// `to_bits` is used rather than the value so that `-0.0` and `0.0` (and two
/// NaN payloads) hash distinctly: the hash must be a hash of the *state as
/// stored*, not of a lossy reading of it.
fn put_f32(out: &mut Vec<u8>, f: f32) {
    out.extend_from_slice(&f.to_bits().to_be_bytes());
}

/// The ticked simulation world.
#[derive(Debug)]
pub struct SimWorld {
    world: World,
    params: ShipParams,
    /// This shard's id, stamped onto every ship this world spawns.
    shard: ShardId,
    tick: SimTick,
    /// guid → entity, for input routing only (never iterated).
    by_guid: HashMap<EntityGuid, Entity>,
    /// Every ship ever spawned, in spawn order; `state_hash` sorts this copy.
    ships: Vec<Entity>,
    /// Inputs waiting for the tick they were generated for, keyed by
    /// [`PlayerInput::tick`].
    ///
    /// A `BTreeMap` (not a `HashMap`) so draining and the stale-drop sweep are
    /// deterministic in tick order.
    input_buf: BTreeMap<SimTick, Vec<PlayerInput>>,
    /// Highest input tick applied per [`Owner`] session — the `input_ack` a
    /// snapshot/delta carries (see [`SimWorld::input_ack`]).
    input_ack: HashMap<u64, SimTick>,
}

impl SimWorld {
    /// An empty world owned by `shard`, using [`ShipParams::PLACEHOLDER`].
    #[must_use]
    pub fn new(shard: ShardId) -> Self {
        Self::with_params(shard, ShipParams::PLACEHOLDER)
    }

    /// An empty world with explicit ship parameters.
    ///
    /// Equivalent to [`SimWorld::with_tick`] at tick 0.
    #[must_use]
    pub fn with_params(shard: ShardId, params: ShipParams) -> Self {
        Self::with_tick(shard, params, 0)
    }

    /// An empty world whose simulation clock starts at `start_tick`.
    ///
    /// This is the **one place** the two tick counters are tied together: the
    /// [`space_ecs`] change tick is set to `start_tick + 1`, so an ECS change
    /// tick and a [`SimTick`] are the same number and a replication pass can hand
    /// [`space_ecs::Query::changed_since`] a tick it read from
    /// [`SimWorld::tick`] directly. (The ECS tick is one ahead because it names
    /// the tick the *next* step will stamp its writes with — see
    /// [`SimWorld::step`]'s tick-ordering note and
    /// [`space_ecs::World::tick`].)
    ///
    /// Both counters are `u64`, so there is no truncation and no wrap policy to
    /// document: `space_ecs::World::tick` was widened to `u64` precisely so this
    /// coupling is exact. A world that starts at a non-zero tick is a world that
    /// joined mid-session (a handoff receiver, a client mirror); its first step
    /// produces `start_tick + 1`.
    #[must_use]
    pub fn with_tick(shard: ShardId, params: ShipParams, start_tick: SimTick) -> Self {
        let mut world = World::with_registry(&COMPONENT_REGISTRY);
        // The one place the two tick counters are joined. The ECS change tick is
        // the *open* tick — the tick the next step will produce — so it is
        // `start_tick + 1` (see [`SimWorld::step`] for the invariant). Both
        // counters are `u64`, so nothing is truncated.
        world.set_tick(start_tick.wrapping_add(1));
        Self {
            world,
            params,
            shard,
            tick: start_tick,
            by_guid: HashMap::new(),
            ships: Vec::new(),
            input_buf: BTreeMap::new(),
            input_ack: HashMap::new(),
        }
    }

    /// The current tick.
    #[must_use]
    pub const fn tick(&self) -> SimTick {
        self.tick
    }

    /// The shard this world is authoritative for.
    #[must_use]
    pub const fn shard(&self) -> ShardId {
        self.shard
    }

    /// The ship parameters in force.
    #[must_use]
    pub const fn params(&self) -> ShipParams {
        self.params
    }

    /// The underlying ECS world (read-only).
    #[must_use]
    pub const fn world(&self) -> &World {
        &self.world
    }

    /// The underlying ECS world, for systems this crate does not provide.
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    /// Number of live ships.
    #[must_use]
    pub fn ship_count(&self) -> usize {
        self.ships
            .iter()
            .filter(|&&e| self.world.entity_is_alive(e))
            .count()
    }

    /// The live ships, sorted by [`Entity`] — the same order
    /// [`SimWorld::state_hash`] hashes them in.
    ///
    /// Sorted rather than spawn-ordered so a caller that walks it (a replication
    /// pass, a test) sees the same sequence on every run regardless of how the
    /// entities happened to be created.
    #[must_use]
    pub fn ships(&self) -> Vec<Entity> {
        let mut out: Vec<Entity> = self
            .ships
            .iter()
            .copied()
            .filter(|&e| self.world.entity_is_alive(e))
            .collect();
        out.sort_unstable_by_key(|&e| (e.index(), e.generation()));
        out
    }

    /// The entity `guid` maps to, if it is still alive here.
    #[must_use]
    pub fn entity_of(&self, guid: EntityGuid) -> Option<Entity> {
        self.by_guid
            .get(&guid)
            .copied()
            .filter(|&e| self.world.entity_is_alive(e))
    }

    /// Spawn a ship: [`ShipState`] + [`ShipControls`] + [`Owner`] +
    /// [`ShardAuthority`] (this shard).
    ///
    /// The guid is registered for input routing. Spawning the same guid twice is
    /// a logic error and panics — a duplicate guid would make input routing
    /// ambiguous.
    pub fn spawn_ship(&mut self, guid: EntityGuid, owner: u64, state: ShipState) -> Entity {
        assert!(
            self.entity_of(guid).is_none(),
            "spawn_ship: guid {guid} is already alive in this world"
        );
        let controls = ShipControls::default();
        let authority = ShardAuthority(self.shard.0);
        let entity = self.world.spawn(&[
            Component::new(&state),
            Component::new(&controls),
            Component::new(&Owner(owner)),
            Component::new(&authority),
        ]);
        self.by_guid.insert(guid, entity);
        self.ships.push(entity);
        entity
    }

    /// Despawn the ship `guid` and forget its input routing.
    ///
    /// Returns `false` if the guid is unknown here.
    pub fn despawn_ship(&mut self, guid: EntityGuid) -> bool {
        match self.by_guid.remove(&guid) {
            Some(entity) => {
                let removed = self.world.despawn(entity);
                self.ships.retain(|&e| e != entity);
                removed
            }
            None => false,
        }
    }

    /// The state of the ship `guid`, if it is alive here.
    #[must_use]
    pub fn ship_state(&self, guid: EntityGuid) -> Option<ShipState> {
        self.entity_of(guid)
            .and_then(|e| self.world.get::<ShipState>(e).copied())
    }

    /// Buffer `inputs` for the tick each one names, to be applied when [`SimWorld::step`]
    /// reaches that tick.
    ///
    /// # Inputs are buffered by tick, not applied on arrival
    ///
    /// Every [`PlayerInput`] carries the [`SimTick`] it was generated for
    /// (`PlayerInput::tick`). `apply_inputs` files each input in a per-tick
    /// buffer keyed by that tick; [`SimWorld::step`] drains **only** the buffer
    /// for the tick it is about to produce and drops anything already stale
    /// (`tick <= current`). A batch that arrives late therefore lands on the tick
    /// it was generated for, and a batch that arrives after that tick has passed
    /// is dropped rather than applied to the wrong tick — which is what a
    /// jittery, lossy, out-of-order network requires and what the client's
    /// `input_ack`-based reconciliation assumes.
    ///
    /// `tick` is the batch's nominal tick (the caller's `InputTick::tick`); it is
    /// only checked for sanity — the per-input `PlayerInput::tick` is what
    /// decides when each input is applied.
    ///
    /// # Order independence
    ///
    /// The buffered list is resolved to entities and **sorted by entity** before
    /// it is applied (inside [`SimWorld::step`]), so applying the same set in any
    /// order produces the same world state. An input naming an unknown (or dead)
    /// entity is dropped — an input for an entity this shard does not simulate is
    /// normal traffic, not an error. If several inputs name the same entity in one
    /// tick, the winner is the one with the greatest bit pattern in (`thrust`,
    /// `torque`) order, which is deterministic and independent of arrival order.
    ///
    /// Applying a batch also records, per [`Owner`] session, the highest input
    /// tick applied (see [`SimWorld::input_ack`]) — the `input_ack` a
    /// snapshot/delta carries so the client can drop acknowledged inputs by tick.
    pub fn apply_inputs(&mut self, tick: SimTick, inputs: &[PlayerInput]) {
        debug_assert!(
            tick >= self.tick,
            "apply_inputs: batch tick {tick} is behind the world tick {}",
            self.tick
        );
        // File each input under the tick it names. Stale inputs (a tick already
        // simulated) are dropped here; the rest wait for step() to reach them.
        for input in inputs {
            if input.tick <= self.tick {
                continue;
            }
            // Drop an input naming an entity this shard does not simulate *before*
            // filing it: otherwise a hostile client could invent a fresh guid on
            // every input, nothing would ever merge (the merge below is per entity
            // per tick) and the buffer would grow without bound.
            if self.entity_of(input.entity).is_none() {
                continue;
            }
            // Bound the buffer against a hostile or buggy client: an input tagged
            // further ahead than `MAX_INPUT_LEAD` ticks is dropped (it would sit in
            // the buffer for that many ticks otherwise, and `u64::MAX` would sit
            // forever), and at most one input is kept per entity per tick — the one
            // with the greatest bit pattern, which is exactly the winner
            // `drain_inputs` would pick anyway. Together these cap the buffer at
            // `ships alive when the input was filed × MAX_INPUT_LEAD` entries no
            // matter how much is thrown at it, so a flood cannot grow it without
            // bound (see [`MAX_INPUT_LEAD`]). The bound counts the ships alive at
            // filing time, not the ones alive now: a despawn leaves its already
            // filed inputs in the buffer until their tick arrives, so the live
            // count can briefly exceed the current live-ship count.
            if input.tick > self.tick.saturating_add(MAX_INPUT_LEAD) {
                continue;
            }
            let bucket = self.input_buf.entry(input.tick).or_default();
            match bucket.iter().position(|e| e.entity == input.entity) {
                Some(i) => {
                    if input_bits(input) > input_bits(&bucket[i]) {
                        bucket[i] = *input;
                    }
                }
                None => bucket.push(*input),
            }
        }
    }

    /// The highest input tick applied so far for the session `owner`, or `0` if
    /// none. This is the value a [`space_protocol::Snapshot`] / [`space_protocol::Delta`]
    /// puts in `input_ack`: the client drops every pending input with
    /// `tick <= input_ack`.
    #[must_use]
    pub fn input_ack(&self, owner: u64) -> SimTick {
        self.input_ack.get(&owner).copied().unwrap_or(0)
    }

    /// How many inputs are currently buffered waiting for their tick.
    ///
    /// Exposed so a caller (and the buffer-bound tests) can observe that
    /// [`SimWorld::apply_inputs`] keeps this small no matter how much is thrown at
    /// it (see [`MAX_INPUT_LEAD`]).
    #[must_use]
    pub fn buffered_input_count(&self) -> usize {
        self.input_buf.values().map(Vec::len).sum()
    }

    /// Drain the buffered batch for `next_tick`, resolve it to entities and apply
    /// it, recording the per-owner input ack. Returns nothing; a missing batch is
    /// a coasting tick (every ship with no input coasts).
    fn drain_inputs(&mut self, next_tick: SimTick) {
        let Some(inputs) = self.input_buf.remove(&next_tick) else {
            return;
        };
        // Resolve, then sort so application order is canonical (independent of the
        // order the inputs were buffered in).
        let mut resolved: Vec<(u32, u32, PlayerInput)> = Vec::with_capacity(inputs.len());
        for input in &inputs {
            if let Some(entity) = self.entity_of(input.entity) {
                resolved.push((entity.index(), entity.generation(), *input));
            }
        }
        resolved.sort_unstable_by(|a, b| {
            (a.0, a.1, input_bits(&a.2)).cmp(&(b.0, b.1, input_bits(&b.2)))
        });
        // Apply the last input per entity (the winner under that order).
        let mut i = 0;
        while i < resolved.len() {
            let (index, generation, _) = resolved[i];
            let mut last = i;
            while last + 1 < resolved.len()
                && (resolved[last + 1].0, resolved[last + 1].1) == (index, generation)
            {
                last += 1;
            }
            let input = resolved[last].2;
            let entity = Entity::new(index, generation);
            if let Some(controls) = self.world.get_mut::<ShipControls>(entity) {
                controls.thrust = input.thrust;
                controls.torque = input.torque;
            }
            // Record the ack for this session: the server has now stepped this
            // owner's input for `next_tick` into the state.
            if let Some(owner) = self.world.get::<Owner>(entity).copied() {
                let slot = self.input_ack.entry(owner.0).or_insert(0);
                *slot = (*slot).max(next_tick);
            }
            i = last + 1;
        }
    }

    /// Advance every ship by one fixed step and return the new tick.
    ///
    /// # Inputs
    ///
    /// First the buffered inputs are pruned: any batch for a tick at or before the
    /// current tick is stale (that tick is already simulated) and is dropped. Then
    /// the batch for the tick this step produces (`tick + 1`) is drained and
    /// applied (see [`SimWorld::apply_inputs`]). A tick with no batch is a
    /// coasting tick.
    ///
    /// # Which ships are stepped
    ///
    /// A read-only pass first collects the ships that actually have work: a
    /// non-zero [`ShipControls`] **or** a non-zero velocity (a coasting ship
    /// still moves, and its position must change). Only those rows are then
    /// stepped. A ship that is at rest with no input is therefore never written,
    /// so its change tick stays where it was and a `changed_since` replication
    /// pass does not see it — which is the whole point: an idle fleet costs
    /// nothing on the wire.
    ///
    /// # Only a real change stamps the row
    ///
    /// Linear damping is asymptotic: a coasting ship's velocity decays toward but
    /// never reaches exactly `0.0` (it stalls in the subnormals around `1e-43`),
    /// so the "moving" filter above keeps such a ship in the active set forever
    /// and a naive `get_mut` would stamp it every tick — replicating a ship that
    /// has, for every practical purpose, stopped. To avoid that, each active ship
    /// is stepped into a **copy** and the row is written through `get_mut` (which
    /// stamps the change tick) **only if the stepped copy's bits differ from the
    /// pre-step bits**. A ship whose physics has settled to a bit-stable state is
    /// therefore not stamped and not replicated, even though its velocity is a
    /// non-zero subnormal.
    ///
    /// [`ShipControls`] are reset to zero only on the ships that had them, and
    /// only when they were non-zero, for the same reason.
    ///
    /// # Tick ordering
    ///
    /// The ECS change tick is advanced at the **end** of the step, together with
    /// the [`SimTick`]. The invariant is *ECS tick = last produced SimTick + 1*:
    /// every write inside this step — the physics pass and the controls reset —
    /// is stamped with the tick the step produces (`next_tick`), and the tick is
    /// only closed once the step is over. A replication pass therefore runs
    /// **after** `step()` and filters with `changed_since(last_acked)`: it sees
    /// exactly the rows the finished step wrote, and a write that lands between
    /// two steps is stamped with the *next* step's tick — a tick no delta has
    /// been sent for yet — so it can never be missed. (Advancing at the start,
    /// as earlier revisions did, stamps such a write with the tick a delta was
    /// already sent for, which loses it forever.)
    ///
    /// A ship without controls (a derelict, or a ghost replica the shard only
    /// mirrors) keeps its state untouched. [`ShardAuthority`] is the marker the
    /// *caller* uses to decide which entities belong to this shard's world at
    /// all; everything inside a `SimWorld` is simulated.
    pub fn step(&mut self) -> SimTick {
        let next_tick = self.tick.wrapping_add(1);
        let params = self.params;
        // 0. Drop stale buffered batches (a tick already simulated), then apply
        //    the batch for the tick this step produces.
        self.input_buf.retain(|&t, _| t > self.tick);
        self.drain_inputs(next_tick);
        // 1. Read-only pass: which ships have work this tick? The entity term
        //    (`WithEntity`) is what lets the pass name the rows it selects, so
        //    the write phase below touches only those.
        let mut active: Vec<Entity> = Vec::new();
        {
            let mut query = self
                .world
                .query::<WithEntity<(&ShipState, &ShipControls)>>();
            query.for_each(|((state, controls), entity)| {
                let moving =
                    state.vel.iter().any(|v| *v != 0.0) || state.ang_vel.iter().any(|v| *v != 0.0);
                let driven = controls.thrust.iter().any(|v| *v != 0.0)
                    || controls.torque.iter().any(|v| *v != 0.0);
                if moving || driven {
                    active.push(entity);
                }
            });
        }
        // Canonical order: the write order below must not depend on archetype
        // layout, so sort by entity (the same order `state_hash` uses).
        active.sort_unstable_by_key(|&e| (e.index(), e.generation()));
        // 2. Step exactly those ships — into a copy. The row is only written
        //    (stamped) when the copy's bits actually changed (see the module
        //    note "Only a real change stamps the row").
        for entity in &active {
            let Some(controls) = self.world.get::<ShipControls>(*entity).copied() else {
                continue;
            };
            let Some(before) = self.world.get::<ShipState>(*entity).copied() else {
                continue;
            };
            let mut next_state = before;
            step_ship(
                &mut next_state,
                &PlayerInput {
                    // Routing metadata; step_ship ignores it.
                    entity: EntityGuid::new(ShardId::NONE, 0),
                    thrust: controls.thrust,
                    torque: controls.torque,
                    tick: next_tick,
                },
                &params,
            );
            if state_bits_changed(&before, &next_state) {
                if let Some(state) = self.world.get_mut::<ShipState>(*entity) {
                    *state = next_state;
                }
            }
        }
        // 3. Consume the controls of the ships that had them: a ship with no
        //    input next tick coasts. Only non-zero controls are reset, so the
        //    write does not stamp an already-idle row.
        for entity in &active {
            let zero = self.world.get::<ShipControls>(*entity).is_some_and(|c| {
                c.thrust.iter().all(|v| *v == 0.0) && c.torque.iter().all(|v| *v == 0.0)
            });
            if !zero {
                if let Some(controls) = self.world.get_mut::<ShipControls>(*entity) {
                    *controls = ShipControls::default();
                }
            }
        }
        // 4. Close the tick: both counters move together (see the tick-ordering
        //    note above).
        self.world.tick();
        self.tick = next_tick;
        self.tick
    }

    /// Run `steps` ticks (see [`SimWorld::step`]).
    pub fn step_n(&mut self, steps: SimTick) -> SimTick {
        for _ in 0..steps {
            self.step();
        }
        self.tick
    }

    /// FNV-1a over the canonical serialisation of the whole world
    /// ([`SimWorld::serialize_state`]).
    ///
    /// Two worlds with the same ships in the same states hash equal, whatever
    /// order their entities were spawned or their inputs arrived in.
    #[must_use]
    pub fn state_hash(&self) -> u64 {
        fnv1a(FNV_OFFSET, &self.serialize_state())
    }

    /// FNV-1a over the **guid-keyed** replicated state — the cross-shard /
    /// cross-process state check.
    ///
    /// Unlike [`SimWorld::state_hash`], which is keyed by the local
    /// [`space_ecs::Entity`] index (so two worlds that spawned the same logical
    /// ships in a different order hash differently), this hash is keyed by
    /// [`EntityGuid`]: it walks the live ships **sorted by guid** and hashes, for
    /// each, only the components the replication registry marks
    /// `replicated == true` — [`ShipState`] and [`Owner`] — using the lossless
    /// wire bytes (the same bytes a snapshot/delta carries).
    ///
    /// Two components are deliberately excluded:
    ///
    /// * [`ShardAuthority`] is not replicated (its registry flag is `false`): it
    ///   names *this* process's ownership, which differs between two shards
    ///   holding the same entity.
    /// * [`ShipControls`] is replicated but **consumed** by [`SimWorld::step`]
    ///   (reset to zero after each tick), so it is not part of the durable state
    ///   a peer must agree on; a shard that is not authoritative for the entity
    ///   never has the same controls anyway.
    ///
    /// Because the guid is hashed and the local ECS index is not, two worlds that
    /// hold the same logical state — same guids, same component values — hash
    /// equal even when their entities live at different ECS indices (different
    /// spawn order) or the same index was recycled by a despawn+respawn. That is
    /// exactly the property a handoff / ghost-replica check needs.
    #[must_use]
    pub fn guid_state_hash(&self) -> u64 {
        // Live (guid, entity) pairs, sorted by guid. `by_guid` is only ever used
        // for point lookups elsewhere; here it is the guid-keyed source of truth,
        // and sorting by guid makes the hash independent of ECS index.
        let mut ships: Vec<(EntityGuid, Entity)> = self
            .by_guid
            .iter()
            .filter(|(_, &e)| self.world.entity_is_alive(e))
            .map(|(&g, &e)| (g, e))
            .collect();
        ships.sort_unstable_by_key(|&(g, _)| g);
        let mut out = Vec::new();
        put_int(&mut out, i128::from(self.tick), 8);
        put_int(
            &mut out,
            i128::try_from(ships.len()).unwrap_or(i128::MAX),
            8,
        );
        for (guid, entity) in ships {
            // The guid keys the entry: shard u32 then serial u64.
            put_int(&mut out, i128::from(guid.shard.0), 4);
            put_int(&mut out, i128::from(guid.serial), 8);
            // ShipState (id 0, replicated): the lossless wire bytes.
            if let Some(state) = self.world.get::<ShipState>(entity) {
                out.push(ShipState::ID as u8);
                out.extend_from_slice(&crate::wire::encode_ship_state(state));
            }
            // ShipControls (id 1, replicated but consumed) — skipped by design.
            // Owner (id 2, replicated): the session id as 8 big-endian bytes.
            if let Some(owner) = self.world.get::<Owner>(entity) {
                out.push(Owner::ID as u8);
                put_int(&mut out, i128::from(owner.0), 8);
            }
            // ShardAuthority (id 3, not replicated) — skipped by design.
        }
        fnv1a(FNV_OFFSET, &out)
    }

    /// The canonical serialisation of every entity's state, sorted by
    /// [`space_ecs::Entity`].
    ///
    /// Layout (all integers big endian, all floats as their IEEE-754 bits, big
    /// endian — the same byte order as [`space_protocol::codec`]):
    ///
    /// ```text
    /// tick:u64 | entity_count:u64
    /// entity: index:u32 | generation:u32 | present:u8 | components in id order
    ///   ShipState      : cell 3×i64 | local 3×f32 | vel 3×f32 | quat 4×f32 | ang_vel 3×f32
    ///   ShipControls   : thrust 3×f32 | torque 3×f32
    ///   Owner          : u64
    ///   ShardAuthority : u32
    /// ```
    ///
    /// `present` is a bitmask over [`crate::components::IDS`] (bit `i` = id `i`),
    /// so a component an entity does not carry is *absent*, not defaulted.
    #[must_use]
    pub fn serialize_state(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_int(&mut out, i128::from(self.tick), 8);
        let mut entities: Vec<Entity> = self
            .ships
            .iter()
            .copied()
            .filter(|&e| self.world.entity_is_alive(e))
            .collect();
        entities.sort_unstable_by_key(|&e| (e.index(), e.generation()));
        put_int(
            &mut out,
            i128::try_from(entities.len()).unwrap_or(i128::MAX),
            8,
        );
        for entity in entities {
            put_int(&mut out, i128::from(entity.index()), 4);
            put_int(&mut out, i128::from(entity.generation()), 4);
            let mut present = 0u8;
            if self.world.get::<ShipState>(entity).is_some() {
                present |= 1;
            }
            if self.world.get::<ShipControls>(entity).is_some() {
                present |= 2;
            }
            if self.world.get::<Owner>(entity).is_some() {
                present |= 4;
            }
            if self.world.get::<ShardAuthority>(entity).is_some() {
                present |= 8;
            }
            out.push(present);
            if present & 1 != 0 {
                let s = self.world.get::<ShipState>(entity).unwrap();
                for &v in &s.pos.cell {
                    put_int(&mut out, i128::from(v), 8);
                }
                for &v in &s.pos.local {
                    put_f32(&mut out, v);
                }
                for &v in &s.vel {
                    put_f32(&mut out, v);
                }
                for &v in &s.quat {
                    put_f32(&mut out, v);
                }
                for &v in &s.ang_vel {
                    put_f32(&mut out, v);
                }
            }
            if present & 2 != 0 {
                let c = self.world.get::<ShipControls>(entity).unwrap();
                for &v in &c.thrust {
                    put_f32(&mut out, v);
                }
                for &v in &c.torque {
                    put_f32(&mut out, v);
                }
            }
            if present & 4 != 0 {
                put_int(
                    &mut out,
                    i128::from(self.world.get::<Owner>(entity).unwrap().0),
                    8,
                );
            }
            if present & 8 != 0 {
                put_int(
                    &mut out,
                    i128::from(self.world.get::<ShardAuthority>(entity).unwrap().0),
                    4,
                );
            }
        }
        out
    }
}

/// The comparison key of an input: its raw bits, so the ordering is total and
/// independent of arrival order.
fn input_bits(input: &PlayerInput) -> ([u32; 3], [u32; 3]) {
    (
        [
            input.thrust[0].to_bits(),
            input.thrust[1].to_bits(),
            input.thrust[2].to_bits(),
        ],
        [
            input.torque[0].to_bits(),
            input.torque[1].to_bits(),
            input.torque[2].to_bits(),
        ],
    )
}

/// `true` if any field of `after` has different bits than `before`.
///
/// Compares the IEEE-754 **bits** (not `==`), so `-0.0` vs `0.0` and two NaN
/// payloads count as a change — the same lossless reading [`SimWorld`] hashes
/// with. This is the "did the physics actually move this ship this tick" test
/// that decides whether to stamp the row (see [`SimWorld::step`]).
fn state_bits_changed(before: &ShipState, after: &ShipState) -> bool {
    if before.pos.cell != after.pos.cell {
        return true;
    }
    let f32s_changed =
        |a: &[f32], b: &[f32]| a.iter().zip(b).any(|(x, y)| x.to_bits() != y.to_bits());
    f32s_changed(&before.pos.local, &after.pos.local)
        || f32s_changed(&before.vel, &after.vel)
        || f32s_changed(&before.quat, &after.quat)
        || f32s_changed(&before.ang_vel, &after.ang_vel)
}

#[cfg(test)]
mod tests {
    use super::{fnv1a, FNV_OFFSET};
    use crate::sim_world::SimWorld;
    use crate::ShipState;
    use space_protocol::{EntityGuid, PlayerInput, ShardId};

    fn guid(serial: u64) -> EntityGuid {
        EntityGuid::new(ShardId(1), serial)
    }

    fn input(serial: u64, thrust: [f32; 3]) -> PlayerInput {
        PlayerInput {
            entity: guid(serial),
            thrust,
            torque: [0.0; 3],
            // Inputs are buffered by tick; these tests apply them for the tick
            // `step()` is about to produce (tick 1).
            tick: 1,
        }
    }

    #[test]
    fn fnv1a_matches_the_reference_vector() {
        // Reference: FNV-1a 64 of "hello world" (Fowler/Noll/Vo, 64-bit).
        assert_eq!(fnv1a(FNV_OFFSET, b"hello world"), 0x779a_65e7_023c_d2e7);
        assert_eq!(fnv1a(FNV_OFFSET, b""), FNV_OFFSET);
    }

    #[test]
    fn spawn_and_step_moves_the_ship() {
        let mut sim = SimWorld::new(ShardId(1));
        sim.spawn_ship(guid(1), 42, ShipState::IDENTITY);
        assert_eq!(sim.ship_count(), 1);
        sim.apply_inputs(1, &[input(1, [1.0, 0.0, 0.0])]);
        sim.step();
        let s = sim.ship_state(guid(1)).unwrap();
        assert!(s.vel[0] > 0.0, "thrust must accelerate: {:?}", s.vel);
        assert_eq!(sim.tick(), 1);
    }

    #[test]
    fn unknown_guids_are_dropped() {
        let mut sim = SimWorld::new(ShardId(1));
        sim.apply_inputs(1, &[input(99, [1.0, 0.0, 0.0])]);
        sim.step();
        assert_eq!(sim.ship_count(), 0);
    }

    // S2: the input buffer is bounded. An input tagged far in the future is
    // dropped, and a flood of future-tick inputs cannot grow the buffer without
    // bound (at most one input per entity per tick, within MAX_INPUT_LEAD ticks).
    #[test]
    fn far_future_inputs_are_dropped() {
        let mut sim = SimWorld::new(ShardId(1));
        sim.spawn_ship(guid(1), 1, ShipState::IDENTITY);
        // tick 0 now; an input for tick 0 + MAX_INPUT_LEAD + 1 is beyond the lead.
        let too_far = crate::sim_world::MAX_INPUT_LEAD + 1;
        let bad = PlayerInput {
            entity: guid(1),
            thrust: [1.0, 0.0, 0.0],
            torque: [0.0; 3],
            tick: too_far,
        };
        sim.apply_inputs(1, &[bad]);
        assert_eq!(
            sim.buffered_input_count(),
            0,
            "a far-future input must be dropped"
        );

        // The extreme case: tick == u64::MAX must not be buffered (or panic).
        let max = PlayerInput {
            tick: u64::MAX,
            ..bad
        };
        sim.apply_inputs(1, &[max]);
        assert_eq!(
            sim.buffered_input_count(),
            0,
            "a u64::MAX-tick input must be dropped"
        );

        // An input exactly at the lead boundary is kept.
        let edge = PlayerInput {
            tick: sim.tick() + crate::sim_world::MAX_INPUT_LEAD,
            ..bad
        };
        sim.apply_inputs(1, &[edge]);
        assert_eq!(
            sim.buffered_input_count(),
            1,
            "an input at exactly MAX_INPUT_LEAD is kept"
        );
    }

    #[test]
    fn input_buffer_stays_bounded_under_a_flood() {
        let mut sim = SimWorld::new(ShardId(1));
        sim.spawn_ship(guid(1), 1, ShipState::IDENTITY);
        // Throw a huge flood of future-tick inputs at the world: many per tick,
        // many ticks, all within the lead window and beyond.
        for tick in 1..=1_000_u64 {
            let batch: Vec<PlayerInput> = (0..50)
                .map(|k| PlayerInput {
                    entity: guid(1),
                    thrust: [f32::from(u8::try_from(k).unwrap()), 0.0, 0.0],
                    torque: [0.0; 3],
                    tick,
                })
                .collect();
            sim.apply_inputs(tick, &batch);
        }
        // One ship, at most MAX_INPUT_LEAD live ticks, one input per entity per
        // tick — so the buffer is capped regardless of the 50 000 inputs sent.
        let cap = crate::sim_world::MAX_INPUT_LEAD as usize;
        assert!(
            sim.buffered_input_count() <= cap,
            "buffer grew to {} (cap {cap})",
            sim.buffered_input_count()
        );
    }

    /// The bound must hold against a client that invents a fresh guid per input:
    /// inputs for entities this shard does not simulate are dropped on arrival,
    /// so they never enter the buffer at all.
    #[test]
    fn spoofed_guids_never_enter_the_buffer() {
        let mut sim = SimWorld::new(ShardId(1));
        // No ship is spawned: every guid below is unknown.
        for tick in 1..=1_000_u64 {
            let batch: Vec<PlayerInput> = (0..50)
                .map(|k| PlayerInput {
                    // A different unknown guid for every input, so nothing can
                    // merge with anything else.
                    entity: EntityGuid::new(ShardId(1), tick * 100 + k),
                    thrust: [0.5, 0.0, 0.0],
                    torque: [0.0; 3],
                    tick,
                })
                .collect();
            sim.apply_inputs(tick, &batch);
        }
        assert_eq!(
            sim.buffered_input_count(),
            0,
            "inputs for unknown entities must be dropped before filing"
        );
    }

    /// One real ship plus a flood of spoofed guids: the buffer stays within
    /// `live_ships × MAX_INPUT_LEAD`.
    #[test]
    fn buffer_bound_holds_with_real_and_spoofed_guids() {
        let mut sim = SimWorld::new(ShardId(1));
        sim.spawn_ship(guid(1), 1, ShipState::IDENTITY);
        for tick in 1..=1_000_u64 {
            let mut batch: Vec<PlayerInput> = (0..50)
                .map(|k| PlayerInput {
                    entity: EntityGuid::new(ShardId(1), 100_000 + tick * 100 + k),
                    thrust: [0.5, 0.0, 0.0],
                    torque: [0.0; 3],
                    tick,
                })
                .collect();
            batch.push(PlayerInput {
                entity: guid(1),
                thrust: [0.25, 0.0, 0.0],
                torque: [0.0; 3],
                tick,
            });
            sim.apply_inputs(tick, &batch);
        }
        let cap = sim.ship_count() * crate::sim_world::MAX_INPUT_LEAD as usize;
        assert!(
            sim.buffered_input_count() <= cap,
            "buffer grew to {} (cap {cap} for {} live ship(s))",
            sim.buffered_input_count(),
            sim.ship_count()
        );
    }

    #[test]
    fn despawn_removes_the_ship_from_the_hash() {
        let mut sim = SimWorld::new(ShardId(1));
        sim.spawn_ship(guid(1), 1, ShipState::IDENTITY);
        let with = sim.state_hash();
        assert!(sim.despawn_ship(guid(1)));
        assert_ne!(with, sim.state_hash());
        assert!(!sim.despawn_ship(guid(1)));
    }

    #[test]
    fn duplicate_inputs_are_resolved_deterministically() {
        let mut a = SimWorld::new(ShardId(1));
        let mut b = SimWorld::new(ShardId(1));
        for sim in [&mut a, &mut b] {
            sim.spawn_ship(guid(1), 1, ShipState::IDENTITY);
        }
        let low = input(1, [0.25, 0.0, 0.0]);
        let high = input(1, [1.0, 0.0, 0.0]);
        a.apply_inputs(1, &[low, high]);
        b.apply_inputs(1, &[high, low]);
        assert_eq!(a.state_hash(), b.state_hash());
        // The greater-magnitude input wins in both.
        assert_eq!(
            a.ship_state(guid(1)).unwrap().vel,
            b.ship_state(guid(1)).unwrap().vel
        );
    }

    // B2: the guid-keyed hash is independent of ECS index / spawn order, while
    // the entity-keyed `state_hash` is not — that is the whole point of having
    // a second hash for cross-shard checks.
    #[test]
    fn guid_state_hash_is_spawn_order_independent() {
        let state_a = ShipState::at_rest(space_world::WorldPos::from_metres([10, 0, 0]));
        let state_b = ShipState::at_rest(space_world::WorldPos::from_metres([0, 20, 0]));

        // World 1 spawns guid(1) then guid(2); world 2 spawns them the other way
        // round, so the same logical ship lives at a different ECS index.
        let mut w1 = SimWorld::new(ShardId(1));
        w1.spawn_ship(guid(1), 100, state_a);
        w1.spawn_ship(guid(2), 200, state_b);

        let mut w2 = SimWorld::new(ShardId(1));
        w2.spawn_ship(guid(2), 200, state_b);
        w2.spawn_ship(guid(1), 100, state_a);

        // The entity-keyed hash sees the different indices and differs.
        assert_ne!(
            w1.state_hash(),
            w2.state_hash(),
            "state_hash is keyed by ECS index, so spawn order must change it"
        );
        // The guid-keyed hash does not: same guids, same replicated state.
        assert_eq!(
            w1.guid_state_hash(),
            w2.guid_state_hash(),
            "guid_state_hash must be independent of spawn order / ECS index"
        );

        // It is still sensitive to a real state change on a ship.
        let mut w3 = SimWorld::new(ShardId(1));
        w3.spawn_ship(guid(1), 100, state_a);
        w3.spawn_ship(guid(2), 999, state_b); // different owner
        assert_ne!(
            w1.guid_state_hash(),
            w3.guid_state_hash(),
            "guid_state_hash must change when a replicated component differs"
        );
    }

    // S3: `ShardAuthority` is excluded from `guid_state_hash` (it is not
    // replicated — it names *this* process's ownership). Two worlds holding the
    // same ships but owned by different shards must therefore hash equal, even
    // though their `ShardAuthority` components differ.
    #[test]
    fn guid_state_hash_ignores_shard_authority() {
        let state = ShipState::at_rest(space_world::WorldPos::from_metres([10, 0, 0]));

        // Same logical ship (same guid, owner and state) in two worlds owned by
        // different shards, so the ShardAuthority payload differs (1 vs 2).
        let mut w1 = SimWorld::new(ShardId(1));
        w1.spawn_ship(guid(1), 100, state);
        let mut w2 = SimWorld::new(ShardId(2));
        w2.spawn_ship(guid(1), 100, state);

        // The ShardAuthority really does differ between the two worlds.
        let e1 = w1.ships()[0];
        let e2 = w2.ships()[0];
        assert_eq!(w1.world().get::<crate::ShardAuthority>(e1).unwrap().0, 1);
        assert_eq!(w2.world().get::<crate::ShardAuthority>(e2).unwrap().0, 2);

        // The entity-keyed hash includes ShardAuthority, so it differs.
        assert_ne!(
            w1.state_hash(),
            w2.state_hash(),
            "state_hash hashes ShardAuthority, so a different shard must change it"
        );
        // The guid-keyed hash excludes it: same ships, same replicated state.
        assert_eq!(
            w1.guid_state_hash(),
            w2.guid_state_hash(),
            "guid_state_hash must exclude ShardAuthority (it is not replicated)"
        );
    }

    // B6: a ship whose physics has settled to a bit-stable state is not stamped,
    // so a `changed_since` replication pass does not keep replicating it.
    //
    // Linear damping is asymptotic: a coasting ship's velocity decays toward but
    // never reaches exactly `0.0`. Once it stalls in the subnormals the stepped
    // copy's bits equal the pre-step bits, and `step()` must not write (stamp)
    // the row — otherwise an effectively-stopped ship is replicated forever.
    //
    // The realistic case is a ship that *moved*: it has a non-identity
    // quaternion, and `step_ship` re-normalises the quaternion every tick. f32
    // normalisation is not guaranteed to be a fixed point, so the quaternion bits
    // could keep flipping (and the row keep being stamped) forever. `step_ship`
    // therefore snaps `|v|`, `|ω|` and the quaternion to a canonical settled form
    // once they are below the settling epsilon (see `crate::physics`); this test
    // drives that path: thrust + torque for a few ticks, then coast 10 000 ticks.
    #[test]
    fn settled_ship_is_not_stamped() {
        use space_ecs::WithEntity;

        let mut sim = SimWorld::new(ShardId(1));
        let e = sim.spawn_ship(guid(1), 1, ShipState::IDENTITY);

        // Thrust + torque for a few ticks: the ship gains velocity, spin and a
        // NON-identity quaternion (the case the old test skipped).
        for tick in 1..=5_u64 {
            sim.apply_inputs(
                tick,
                &[PlayerInput {
                    entity: guid(1),
                    thrust: [1.0, 0.3, -0.2],
                    torque: [0.1, 0.9, 0.0],
                    tick,
                }],
            );
            sim.step();
        }
        let driven = sim.ship_state(guid(1)).unwrap();
        assert_ne!(
            driven.quat,
            [1.0, 0.0, 0.0, 0.0],
            "torque must turn the ship"
        );
        assert!(
            driven.vel.iter().any(|v| *v != 0.0) && driven.ang_vel.iter().any(|v| *v != 0.0),
            "the ship must be moving and spinning: {driven:?}"
        );

        // Coast 10 000 ticks (no input): linear and angular damping bleed both to
        // the settling epsilon, at which point the state becomes bit-stable.
        for _ in 0..10_000 {
            sim.step();
        }
        let settled = sim.ship_state(guid(1)).unwrap();
        assert!(
            settled.vel.iter().all(|v| *v == 0.0),
            "linear velocity must snap to exactly 0: {:?}",
            settled.vel
        );
        assert!(
            settled.ang_vel.iter().all(|v| *v == 0.0),
            "angular velocity must snap to exactly 0: {:?}",
            settled.ang_vel
        );

        // The row must not be stamped by a further step: `changed_since(t - 1)`
        // (the tick the last step wrote) sees only rows the *next* step writes.
        let t = sim.tick();
        sim.step();
        let mut changed: Vec<space_ecs::Entity> = Vec::new();
        {
            let mut q = sim
                .world_mut()
                .query::<WithEntity<&ShipState>>()
                .changed_since(t);
            q.for_each(|(_state, entity)| changed.push(entity));
        }
        assert!(
            !changed.contains(&e),
            "a settled ship (zero velocity, canonical quat) must not be stamped: {changed:?}"
        );
    }

    // B6 (idle case): a ship that never moved is never stamped by step().
    #[test]
    fn idle_ship_is_never_stamped() {
        let mut sim = SimWorld::new(ShardId(1));
        let e = sim.spawn_ship(guid(1), 1, ShipState::IDENTITY);
        let spawn_stamp = sim.world().row_tick(e).unwrap();
        for _ in 0..10 {
            sim.step();
            assert_eq!(
                sim.world().row_tick(e).unwrap(),
                spawn_stamp,
                "an at-rest ship with no input must never be written"
            );
        }
    }
}
