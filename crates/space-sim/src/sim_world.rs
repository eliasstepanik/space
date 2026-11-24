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
//!
//! # Physics
//!
//! Every simulated entity — a ship, a crate, a door, a chain — is a rigid body
//! of the world's physics ([`crate::rigid`]): its motion is its [`ShipState`]
//! (plus [`PartStates`] for an articulated object), what it is is its
//! [`Body`]. A ship is a body with [`ShipControls`]: the flight model drives it
//! ([`crate::physics`]), and in free flight it moves exactly as
//! [`crate::step_ship`] says. The scene's static colliders
//! ([`SimWorld::set_statics`]), its gravity ([`SimWorld::set_gravity`]) and the
//! catalogue of big shapes ([`SimWorld::set_catalog`]) are per world.
//!
//! The phase-2 hooks are here too: [`SimWorld::export_entity`] /
//! [`SimWorld::import_entity`] turn an entity into its replicated component
//! list and back (a handoff moves a body, articulation and springs included,
//! exactly), [`SimWorld::insert_ghost`] / [`SimWorld::drive_ghost`] keep a
//! kinematic copy of a body simulated elsewhere, [`SimWorld::islands`] groups
//! the bodies that may interact soon, [`SimWorld::bubbles`] shows the physics
//! bubbles and [`SimWorld::contacts`] the contacts of the last tick.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use space_ecs::{Component, ComponentDef, ComponentId, Entity, World};
use space_protocol::{EntityGuid, EntitySnapshot, PlayerInput, ShardId};

use crate::articulation::{
    decode_articulation, decode_part_states, encode_articulation, encode_part_states, Articulation,
    PartStates,
};
use crate::body::{
    decode_body, decode_residual, decode_visual, encode_body, encode_residual, encode_visual, Body,
    PosResidual, Visual,
};
use crate::components::{
    Ghost, Owner, ShardAuthority, ShipControls, ShipState, COMPONENT_REGISTRY,
};
use crate::gravity::GravityField;
use crate::params::ShipParams;
use crate::rigid::{
    BubbleId, BubbleInfo, ContactEvent, Island, IslandQuery, Physics, PhysicsError, StaticCollider,
};
use crate::shape::ShapeCatalog;
use crate::spring::{decode_springs, encode_springs, Springs};
use crate::tick::SimTick;

/// A body to spawn with [`SimWorld::spawn_body`].
#[derive(Clone, Debug, PartialEq)]
pub struct BodySpawn {
    /// Initial motion (of part 0).
    pub state: ShipState,
    /// What the body is.
    pub body: Body,
    /// What a client draws.
    pub visual: Option<Visual>,
    /// Springs it owns.
    pub springs: Option<Springs>,
    /// Extra parts and joints, with the initial motion of the extra parts.
    pub articulation: Option<(Articulation, PartStates)>,
}

impl BodySpawn {
    /// A plain body.
    #[must_use]
    pub fn new(state: ShipState, body: Body) -> Self {
        Self {
            state,
            body,
            visual: None,
            springs: None,
            articulation: None,
        }
    }

    /// With a visual.
    #[must_use]
    pub fn with_visual(mut self, visual: Visual) -> Self {
        self.visual = Some(visual);
        self
    }

    /// With springs.
    #[must_use]
    pub fn with_springs(mut self, springs: Springs) -> Self {
        self.springs = Some(springs);
        self
    }

    /// With extra parts and joints.
    #[must_use]
    pub fn with_articulation(mut self, articulation: Articulation, parts: PartStates) -> Self {
        self.articulation = Some((articulation, parts));
        self
    }
}

/// Most components [`SimWorld::import_entity`] accepts for one entity (one of
/// each replicated id).
pub const MAX_IMPORT_COMPONENTS: usize = 9;

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

/// How many applied inputs a world remembers ([`SimWorld::applied_inputs`]):
/// a couple of minutes of one ship.
pub const APPLIED_LOG: usize = 4096;

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
    /// The ships spawned through [`SimWorld::spawn_ship`] and not yet despawned
    /// through [`SimWorld::despawn_ship`], as `(index, generation)`.
    ///
    /// An ordered set, so it is *always* in the canonical order
    /// ([`SimWorld::ships`], [`SimWorld::state_hash`]) — nothing has to be
    /// sorted per call — and a despawn is a lookup, not a scan. An entity
    /// despawned behind this world's back (through [`SimWorld::world_mut`]) can
    /// linger here, which is why every reader still filters on liveness.
    ships: BTreeSet<(u32, u32)>,
    /// Inputs waiting for the tick they were generated for, keyed by
    /// [`PlayerInput::tick`], then by the entity they drive (at most one input
    /// per entity per tick, see [`SimWorld::apply_inputs`]).
    ///
    /// `BTreeMap`s (not `HashMap`s) so draining and the stale-drop sweep are
    /// deterministic in tick order, and so filing an input is a keyed lookup
    /// rather than a scan of everything already filed for that tick.
    input_buf: BTreeMap<SimTick, BTreeMap<EntityGuid, PlayerInput>>,
    /// Highest input tick applied per [`Owner`] session — the `input_ack` a
    /// snapshot/delta carries (see [`SimWorld::input_ack`]).
    input_ack: HashMap<u64, SimTick>,
    /// The last [`APPLIED_LOG`] inputs applied: the tick and the ship
    /// ([`SimWorld::applied_inputs`]).
    applied: std::collections::VecDeque<(SimTick, EntityGuid)>,
    /// The rigid-body world (bubbles, statics, gravity).
    physics: Physics,
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
            ships: BTreeSet::new(),
            input_buf: BTreeMap::new(),
            input_ack: HashMap::new(),
            applied: std::collections::VecDeque::new(),
            physics: Physics::new(Arc::new(ShapeCatalog::new())),
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
        self.live_ships().count()
    }

    /// The live ships, sorted by [`Entity`] — the same order
    /// [`SimWorld::state_hash`] hashes them in.
    ///
    /// Sorted rather than spawn-ordered so a caller that walks it (a replication
    /// pass, a test) sees the same sequence on every run regardless of how the
    /// entities happened to be created.
    #[must_use]
    pub fn ships(&self) -> Vec<Entity> {
        self.live_ships().collect()
    }

    /// The live ships in canonical order: ascending `(index, generation)`,
    /// which is the order the set is kept in.
    fn live_ships(&self) -> impl Iterator<Item = Entity> + '_ {
        self.ships
            .iter()
            .map(|&(index, generation)| Entity::new(index, generation))
            .filter(|&e| self.world.entity_is_alive(e))
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
    /// [`ShardAuthority`] (this shard) + [`Body::ship`] — a dynamic body that
    /// collides with the world and pushes what it hits.
    ///
    /// The guid is registered for input routing. Spawning the same guid twice is
    /// a logic error and panics — a duplicate guid would make input routing
    /// ambiguous.
    pub fn spawn_ship(&mut self, guid: EntityGuid, owner: u64, state: ShipState) -> Entity {
        let body = Body::ship(&self.params);
        self.spawn_ship_with_body(guid, owner, state, body)
            .expect("the ship body is valid")
    }

    /// [`SimWorld::spawn_ship`] with a body of the caller's choice.
    ///
    /// # Errors
    /// [`PhysicsError`] if the body cannot be built (nothing is spawned).
    ///
    /// # Panics
    /// If `guid` is already alive here.
    pub fn spawn_ship_with_body(
        &mut self,
        guid: EntityGuid,
        owner: u64,
        state: ShipState,
        body: Body,
    ) -> Result<Entity, PhysicsError> {
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
            Component::new(&body),
        ]);
        if let Err(e) = self
            .physics
            .register(guid, entity, &self.world, false, &self.params)
        {
            self.world.despawn(entity);
            return Err(e);
        }
        self.by_guid.insert(guid, entity);
        self.ships.insert((entity.index(), entity.generation()));
        Ok(entity)
    }

    /// Despawn the ship `guid` and forget its input routing.
    ///
    /// Returns `false` if the guid is unknown here.
    pub fn despawn_ship(&mut self, guid: EntityGuid) -> bool {
        self.despawn(guid)
    }

    /// Despawn any entity of this world (ship, body or ghost) by guid.
    ///
    /// Returns `false` if the guid is unknown here.
    pub fn despawn(&mut self, guid: EntityGuid) -> bool {
        self.physics.unregister(guid);
        match self.by_guid.remove(&guid) {
            Some(entity) => {
                let removed = self.world.despawn(entity);
                self.ships.remove(&(entity.index(), entity.generation()));
                removed
            }
            None => false,
        }
    }

    /// Use `catalog` for [`crate::ShapeRef::Asset`] shapes. Set it before
    /// spawning a body or static that names an asset.
    pub fn set_catalog(&mut self, catalog: Arc<ShapeCatalog>) {
        self.physics.catalog = catalog;
    }

    /// The shape catalogue in use.
    #[must_use]
    pub fn catalog(&self) -> &Arc<ShapeCatalog> {
        &self.physics.catalog
    }

    /// Replace the scene's static colliders (instantiated into the bubbles
    /// that come near them from the next step on).
    ///
    /// # Errors
    /// [`PhysicsError::Shape`] if a shape cannot be built; nothing changes.
    pub fn set_statics(&mut self, statics: Vec<StaticCollider>) -> Result<(), PhysicsError> {
        self.physics.set_statics(statics)
    }

    /// The scene's static colliders.
    pub fn statics(&self) -> impl Iterator<Item = &StaticCollider> {
        self.physics.statics()
    }

    /// Replace the gravity field (zones and large bodies). The world's scene
    /// clock ([`SimWorld::set_scene_clock`]) stays: `field.clock` is
    /// overwritten with it.
    pub fn set_gravity(&mut self, mut field: GravityField) {
        field.clock = self.physics.gravity.clock;
        self.physics.gravity = field;
    }

    /// Map ticks to scene time by `clock` from now on (what orbiting bodies
    /// and moving statics move by; see [`crate::gravity`] and
    /// [`crate::rigid`]). The next step runs from the scene time the last one
    /// ended at to the new mapping's time of the tick it produces, so a
    /// re-anchored clock moves the planets once, and everything they carry
    /// with them.
    pub fn set_scene_clock(&mut self, clock: space_protocol::SceneClock) {
        self.physics.gravity.clock = clock;
    }

    /// The mapping from ticks to scene time in use.
    #[must_use]
    pub fn scene_clock(&self) -> space_protocol::SceneClock {
        self.physics.gravity.clock
    }

    /// The scene time of the current tick, µs (where the last step ended, or
    /// the clock's time of the tick after a reset).
    #[must_use]
    pub fn scene_time(&self) -> i64 {
        self.physics
            .last_time
            .unwrap_or_else(|| self.physics.gravity.clock.time_us_at(self.tick))
    }

    /// How far the orbiting bodies carry the point `pos` from scene time
    /// `from_us` to `to_us`, metres ([`GravityField::carry`]). A shard moves
    /// a body handed over from another shard (whose clock may differ by a few
    /// milliseconds) by this, so it arrives where it was relative to the
    /// planet it is near.
    #[must_use]
    pub fn carry(&self, pos: space_world::WorldPos, from_us: i64, to_us: i64) -> [f64; 3] {
        self.physics
            .gravity
            .carry(pos, [0.0; 3], self.tick, from_us, to_us)
    }

    /// Distance from `pos` to the nearest surface of the scene now (a large
    /// body, a gravity zone or a static), metres — what fast travel scales
    /// its speed limit by ([`crate::physics::fast_travel_limit`]).
    #[must_use]
    pub fn surface_distance(&self, pos: space_world::WorldPos) -> f64 {
        self.physics
            .surface_distance(pos, [0.0; 3], self.tick, self.scene_time())
    }

    /// The gravity field in force.
    #[must_use]
    pub fn gravity(&self) -> &GravityField {
        &self.physics.gravity
    }

    /// Restart the clock at `tick` and drop every buffered input — for a
    /// prediction mirror that replays from an older authoritative state (see
    /// [`crate::prediction::PredictedPhysics`]). The ECS change tick moves
    /// with it (see [`SimWorld::with_tick`]).
    pub fn reset_clock(&mut self, tick: SimTick) {
        self.tick = tick;
        // Scene time restarts at the clock's time of `tick`.
        self.physics.last_time = None;
        self.world.set_tick(tick.wrapping_add(1));
        self.input_buf.clear();
    }

    /// Jump forward to `tick` without simulating the ticks in between — for
    /// a shard that has fallen too far behind its clock to make them up. A
    /// `tick` at or before the present does nothing.
    ///
    /// Unlike [`SimWorld::reset_clock`] this is a jump **on one timeline**:
    ///
    /// * scene time is not restarted. The next step runs from where the last
    ///   one ended to the scene time of the tick it produces, so the orbiting
    ///   bodies move by the whole of the jump and **carry** what is near
    ///   them by as much ([`crate::rigid`], step 0) — a ship standing on a
    ///   moon is still standing on it afterwards. (Restarting scene time
    ///   moved the moon and left the ship where the moon had been.)
    /// * only the inputs for the ticks jumped over are dropped; an input
    ///   filed for a later tick is still applied on it.
    ///
    /// Nothing else moves across the jump: velocities, contacts and springs
    /// see none of the skipped ticks.
    pub fn skip_to(&mut self, tick: SimTick) {
        if tick <= self.tick {
            return;
        }
        // (Before the first step there is no "where the last one ended":
        // the time of the tick the jump starts from.)
        let from = self.physics.gravity.clock.time_us_at(self.tick);
        self.physics.last_time.get_or_insert(from);
        self.tick = tick;
        self.world.set_tick(tick.wrapping_add(1));
        self.input_buf.retain(|at, _| *at > tick);
    }

    /// Spawn a physics body (not a ship): see [`BodySpawn`]. It is
    /// authoritative here ([`ShardAuthority`] = this shard).
    ///
    /// # Errors
    /// [`PhysicsError::Duplicate`] for a guid alive here, [`PhysicsError`] for
    /// a body, spring set or articulation outside the wire bounds (a spawned
    /// body can always be exported and imported again).
    pub fn spawn_body(
        &mut self,
        guid: EntityGuid,
        spawn: BodySpawn,
    ) -> Result<Entity, PhysicsError> {
        if self.entity_of(guid).is_some() {
            return Err(PhysicsError::Duplicate(guid));
        }
        let invalid = |e: space_protocol::CodecError| PhysicsError::Invalid(e.to_string());
        spawn.body.validate().map_err(invalid)?;
        let authority = ShardAuthority(self.shard.0);
        let parts = 1 + spawn
            .articulation
            .as_ref()
            .map_or(0, |(a, _)| a.parts.len());
        let residual = PosResidual(vec![[0.0; 3]; parts]);
        let mut components = vec![
            Component::new(&spawn.state),
            Component::new(&spawn.body),
            Component::new(&authority),
            Component::new(&residual),
        ];
        if let Some(v) = &spawn.visual {
            components.push(Component::new(v));
        }
        if let Some(s) = &spawn.springs {
            decode_springs(&encode_springs(s)).map_err(invalid)?;
            components.push(Component::new(s));
        }
        if let Some((a, p)) = &spawn.articulation {
            decode_articulation(&encode_articulation(a)).map_err(invalid)?;
            decode_part_states(&encode_part_states(p)).map_err(invalid)?;
            components.push(Component::new(a));
            components.push(Component::new(p));
        }
        let entity = self.world.spawn(&components);
        if let Err(e) = self
            .physics
            .register(guid, entity, &self.world, false, &self.params)
        {
            self.world.despawn(entity);
            return Err(e);
        }
        self.by_guid.insert(guid, entity);
        Ok(entity)
    }

    /// Every physics entity of this world (ships, bodies, ghosts) by guid,
    /// ascending: `(guid, entity, is_ghost)`.
    #[must_use]
    pub fn bodies(&self) -> Vec<(EntityGuid, Entity, bool)> {
        self.physics.guids().collect()
    }

    /// The state of any entity `guid` (ship, body or ghost).
    #[must_use]
    pub fn state_of(&self, guid: EntityGuid) -> Option<ShipState> {
        self.entity_of(guid)
            .and_then(|e| self.world.get::<ShipState>(e).copied())
    }

    /// The motion of every part of `guid`: part 0 is its [`ShipState`], the
    /// rest its [`PartStates`].
    #[must_use]
    pub fn part_states_of(&self, guid: EntityGuid) -> Option<Vec<ShipState>> {
        let e = self.entity_of(guid)?;
        let mut out = vec![*self.world.get::<ShipState>(e)?];
        if let Some(p) = self.world.get::<PartStates>(e) {
            out.extend_from_slice(&p.0);
        }
        Some(out)
    }

    /// Whether the physics has put `guid` to sleep (at rest, not integrated
    /// until something wakes it).
    #[must_use]
    pub fn is_sleeping(&self, guid: EntityGuid) -> Option<bool> {
        self.physics.is_sleeping(guid)
    }

    /// Wake `guid` (a sleeping body is integrated again from the next step).
    pub fn wake(&mut self, guid: EntityGuid) {
        self.physics.wake(guid);
    }

    /// The physics bubble `guid` is in (assigned by the first step after it
    /// joined).
    #[must_use]
    pub fn bubble_of(&self, guid: EntityGuid) -> Option<BubbleId> {
        self.physics.bubble_of(guid)
    }

    /// The physics bubbles, by id (see [`crate::rigid`]).
    #[must_use]
    pub fn bubbles(&self) -> Vec<BubbleInfo> {
        self.physics.bubbles()
    }

    /// The contacts the solver resolved in the last step, sorted.
    #[must_use]
    pub fn contacts(&self) -> &[ContactEvent] {
        self.physics.contacts()
    }

    /// The groups of dynamic bodies that may interact within `query`'s reach
    /// (contact proximity, springs), for co-locating each group on one shard.
    #[must_use]
    pub fn islands(&self, query: IslandQuery) -> Vec<Island> {
        self.physics.islands(&self.world, query)
    }

    /// The replicated component list of `guid` — what an
    /// [`EntitySnapshot`] (and a handoff) carries — in id order: [`ShipState`]
    /// (`wire`), [`ShipControls`] and [`Owner`] (bincode, as `space-server`'s
    /// `snapshot_of` sends them) and the physics components it has ([`Body`],
    /// [`Visual`], [`Springs`], [`Articulation`], [`PartStates`],
    /// [`PosResidual`], each in its versioned wire form). `None` if `guid` is not alive here.
    #[must_use]
    pub fn export_entity(&self, guid: EntityGuid) -> Option<Vec<(ComponentId, Vec<u8>)>> {
        let e = self.entity_of(guid)?;
        let w = &self.world;
        let mut out = vec![(
            ShipState::ID,
            crate::wire::encode_ship_state(w.get::<ShipState>(e)?),
        )];
        if let Some(c) = w.get::<ShipControls>(e) {
            out.push((ShipControls::ID, space_protocol::codec::encode_raw(c)));
        }
        if let Some(o) = w.get::<Owner>(e) {
            out.push((Owner::ID, space_protocol::codec::encode_raw(&o.0)));
        }
        if let Some(b) = w.get::<Body>(e) {
            out.push((Body::ID, encode_body(b)));
        }
        if let Some(v) = w.get::<Visual>(e) {
            out.push((Visual::ID, encode_visual(*v)));
        }
        if let Some(s) = w.get::<Springs>(e) {
            out.push((Springs::ID, encode_springs(s)));
        }
        if let Some(a) = w.get::<Articulation>(e) {
            out.push((Articulation::ID, encode_articulation(a)));
        }
        if let Some(p) = w.get::<PartStates>(e) {
            out.push((PartStates::ID, encode_part_states(p)));
        }
        if let Some(r) = w.get::<PosResidual>(e) {
            out.push((PosResidual::ID, encode_residual(r)));
        }
        Some(out)
    }

    /// [`SimWorld::export_entity`] as an [`EntitySnapshot`].
    #[must_use]
    pub fn entity_snapshot(&self, guid: EntityGuid) -> Option<EntitySnapshot> {
        Some(EntitySnapshot::new(guid, self.export_entity(guid)?))
    }

    /// Spawn `guid` from its replicated component list (the inverse of
    /// [`SimWorld::export_entity`]), authoritative here. An entity with
    /// [`ShipControls`] becomes a ship (its [`Owner`] routes inputs; a ship
    /// without a [`Body`] gets [`Body::ship`]); anything else needs a [`Body`].
    ///
    /// Decoding is strict and total: an unknown or repeated id, more than
    /// [`MAX_IMPORT_COMPONENTS`] entries, a payload that does not decode or a
    /// [`PartStates`] that does not match the [`Articulation`] is an error and
    /// nothing is spawned.
    ///
    /// # Errors
    /// [`PhysicsError::Duplicate`] for a guid alive here, [`PhysicsError`]
    /// for invalid components.
    pub fn import_entity(
        &mut self,
        guid: EntityGuid,
        components: &[(ComponentId, Vec<u8>)],
    ) -> Result<Entity, PhysicsError> {
        self.import(guid, components, false)
    }

    /// Insert (or replace) a **ghost**: a kinematic copy of `guid`, which
    /// another shard simulates, built from its replicated component list.
    /// Bodies of this world collide with it; it is never integrated here —
    /// move it with [`SimWorld::drive_ghost`] every tick. A spring a ghost
    /// owns pulls on this world's end of it.
    ///
    /// # Errors
    /// As [`SimWorld::import_entity`]; replacing a non-ghost is
    /// [`PhysicsError::Duplicate`].
    pub fn insert_ghost(
        &mut self,
        guid: EntityGuid,
        components: &[(ComponentId, Vec<u8>)],
    ) -> Result<Entity, PhysicsError> {
        if self.entity_of(guid).is_some() {
            if !self.physics.is_ghost(guid) {
                return Err(PhysicsError::Duplicate(guid));
            }
            self.despawn(guid);
        }
        self.import(guid, components, true)
    }

    /// Set a ghost's motion (and its parts'): what its owner says it is now.
    /// Returns `false` if `guid` is not a ghost here.
    pub fn drive_ghost(
        &mut self,
        guid: EntityGuid,
        state: ShipState,
        parts: Option<&[ShipState]>,
    ) -> bool {
        if !self.physics.is_ghost(guid) {
            return false;
        }
        let Some(e) = self.entity_of(guid) else {
            return false;
        };
        // Written only when the bits change: a write stamps the row, and a
        // stamped row is replicated again (a ghost at rest costs nothing).
        let same = |a: &ShipState, b: &ShipState| {
            crate::wire::encode_ship_state(a) == crate::wire::encode_ship_state(b)
        };
        if self
            .world
            .get::<ShipState>(e)
            .is_some_and(|s| !same(s, &state))
        {
            if let Some(s) = self.world.get_mut::<ShipState>(e) {
                *s = state;
            }
        }
        if let Some(parts) = parts {
            let differs = self.world.get::<PartStates>(e).is_some_and(|p| {
                p.0.len() == parts.len() && p.0.iter().zip(parts).any(|(a, b)| !same(a, b))
            });
            if differs {
                if let Some(p) = self.world.get_mut::<PartStates>(e) {
                    p.0.copy_from_slice(parts);
                }
            }
        }
        true
    }

    /// Whether `guid` is a ghost here.
    #[must_use]
    pub fn is_ghost(&self, guid: EntityGuid) -> bool {
        self.physics.is_ghost(guid)
    }

    /// Give the entity `guid` the [`Visual`] `visual` (replacing one it has),
    /// e.g. the player model on a ship a client flies. Only what a client
    /// draws changes; the physics does not read it. Returns `false` if
    /// `guid` is not alive here.
    pub fn set_visual(&mut self, guid: EntityGuid, visual: Visual) -> bool {
        let Some(entity) = self.entity_of(guid) else {
            return false;
        };
        if self.world.get::<Visual>(entity) == Some(&visual) {
            return true;
        }
        self.world.insert(entity, Component::new(&visual))
    }

    /// Every check [`SimWorld::import_entity`] makes on `components` —
    /// decoding, consistency and building the shapes with this world's
    /// catalogue — without spawning anything here. A handoff's receiver runs
    /// it on an offer, while the sender still simulates the entity and a
    /// refusal costs nothing.
    ///
    /// # Errors
    /// What the import would return.
    pub fn check_import(&self, components: &[(ComponentId, Vec<u8>)]) -> Result<(), PhysicsError> {
        let mut scratch = Self::with_params(self.shard, self.params);
        scratch.set_catalog(Arc::clone(&self.physics.catalog));
        scratch
            .import(EntityGuid::new(self.shard, 0), components, false)
            .map(|_| ())
    }

    /// Whether a body of `body`'s shape placed at `state` would come closer
    /// than `clearance` metres to a static of the scene or to any entity of
    /// this world (ghosts included), as far as the collision groups let the
    /// two touch. For finding a free spawn point: a shard looks for a place
    /// a new ship does not start inside anything.
    ///
    /// A shape pair the contact query does not support counts as
    /// overlapping when the bounding spheres do.
    ///
    /// # Errors
    /// [`PhysicsError::Shape`] if `body`'s shape cannot be built.
    pub fn overlaps(
        &self,
        body: &Body,
        state: &ShipState,
        clearance: f64,
    ) -> Result<bool, PhysicsError> {
        self.physics.overlaps(&self.world, body, state, clearance)
    }

    fn import(
        &mut self,
        guid: EntityGuid,
        components: &[(ComponentId, Vec<u8>)],
        ghost: bool,
    ) -> Result<Entity, PhysicsError> {
        if self.entity_of(guid).is_some() {
            return Err(PhysicsError::Duplicate(guid));
        }
        if components.len() > MAX_IMPORT_COMPONENTS {
            return Err(PhysicsError::Invalid(format!(
                "{} components, at most {MAX_IMPORT_COMPONENTS}",
                components.len()
            )));
        }
        let bad = |e: space_protocol::CodecError| PhysicsError::Invalid(e.to_string());
        let mut seen = BTreeSet::new();
        let mut state = None;
        let mut controls = None;
        let mut owner = None;
        let mut body = None;
        let mut visual = None;
        let mut springs = None;
        let mut articulation = None;
        let mut parts = None;
        let mut residual = None;
        for (id, bytes) in components {
            if !seen.insert(*id) {
                return Err(PhysicsError::Invalid(format!("component {id} twice")));
            }
            match *id {
                ShipState::ID => {
                    state = Some(crate::wire::decode_ship_state(bytes).map_err(bad)?);
                }
                ShipControls::ID => {
                    let (c, n) =
                        space_protocol::codec::decode_raw::<ShipControls>(bytes).map_err(bad)?;
                    if n != bytes.len() {
                        return Err(PhysicsError::Invalid("ShipControls: trailing bytes".into()));
                    }
                    controls = Some(c);
                }
                Owner::ID => {
                    let (o, n) = space_protocol::codec::decode_raw::<u64>(bytes).map_err(bad)?;
                    if n != bytes.len() {
                        return Err(PhysicsError::Invalid("Owner: trailing bytes".into()));
                    }
                    owner = Some(Owner(o));
                }
                Body::ID => body = Some(decode_body(bytes).map_err(bad)?),
                Visual::ID => visual = Some(decode_visual(bytes).map_err(bad)?),
                Springs::ID => springs = Some(decode_springs(bytes).map_err(bad)?),
                Articulation::ID => {
                    articulation = Some(decode_articulation(bytes).map_err(bad)?);
                }
                PartStates::ID => parts = Some(decode_part_states(bytes).map_err(bad)?),
                PosResidual::ID => residual = Some(decode_residual(bytes).map_err(bad)?),
                other => {
                    return Err(PhysicsError::Invalid(format!(
                        "unknown component id {other}"
                    )))
                }
            }
        }
        let state = state.ok_or_else(|| PhysicsError::Invalid("no ShipState".into()))?;
        let is_ship = controls.is_some();
        let ship = is_ship && !ghost;
        let body = match body {
            Some(b) => b,
            None if controls.is_some() => Body::ship(&self.params),
            None => return Err(PhysicsError::Invalid("no Body".into())),
        };
        if articulation.as_ref().map_or(0, |a| a.parts.len())
            != parts.as_ref().map_or(0, |p| p.0.len())
        {
            return Err(PhysicsError::Invalid(
                "PartStates do not match the Articulation".into(),
            ));
        }
        let part_count = 1 + articulation.as_ref().map_or(0, |a| a.parts.len());
        if residual.as_ref().is_some_and(|r| r.0.len() != part_count) {
            return Err(PhysicsError::Invalid(
                "PosResidual does not match the parts".into(),
            ));
        }
        // A body (not a ship) always carries its residual.
        let residual = if ship {
            None
        } else {
            Some(residual.unwrap_or_else(|| PosResidual(vec![[0.0; 3]; part_count])))
        };
        let authority = ShardAuthority(self.shard.0);
        let controls = controls.unwrap_or_default();
        let owner = owner.unwrap_or_default();
        let mut list = vec![Component::new(&state), Component::new(&body)];
        if ship {
            list.push(Component::new(&controls));
            list.push(Component::new(&owner));
        }
        if ghost {
            list.push(Component::new(&Ghost));
        } else {
            list.push(Component::new(&authority));
        }
        if let Some(v) = &visual {
            list.push(Component::new(v));
        }
        if let Some(s) = &springs {
            list.push(Component::new(s));
        }
        if let (Some(a), Some(p)) = (&articulation, &parts) {
            list.push(Component::new(a));
            list.push(Component::new(p));
        }
        if let Some(r) = &residual {
            list.push(Component::new(r));
        }
        let entity = self.world.spawn(&list);
        if let Err(e) = self
            .physics
            .register(guid, entity, &self.world, ghost, &self.params)
            .map(|()| self.physics.mark_ship_ghost(guid, ghost && is_ship))
        {
            self.world.despawn(entity);
            return Err(e);
        }
        self.by_guid.insert(guid, entity);
        if ship {
            self.ships.insert((entity.index(), entity.generation()));
        }
        Ok(entity)
    }

    /// The state of the ship (or body) `guid`, if this world simulates it —
    /// `None` for a ghost, which another world simulates (use
    /// [`SimWorld::state_of`] for any entity).
    #[must_use]
    pub fn ship_state(&self, guid: EntityGuid) -> Option<ShipState> {
        if self.physics.is_ghost(guid) {
            return None;
        }
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
            // Keyed by entity, so filing is a lookup: with one input per ship
            // per tick a linear scan of the bucket made a tick's worth of
            // inputs quadratic in the number of ships.
            match self
                .input_buf
                .entry(input.tick)
                .or_default()
                .entry(input.entity)
            {
                Entry::Occupied(mut filed) => {
                    if input_bits(input) > input_bits(filed.get()) {
                        *filed.get_mut() = *input;
                    }
                }
                Entry::Vacant(slot) => {
                    slot.insert(*input);
                }
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

    /// The inputs applied lately, oldest first: `(tick, ship)` for every
    /// tick a ship's controls were set from a player's input — the last
    /// [`APPLIED_LOG`] of them. A record for whoever has to show that every
    /// input was applied **exactly once** (across a handoff, say: the union
    /// of two shards' records names every tick once); nothing in the
    /// simulation reads it.
    pub fn applied_inputs(&self) -> impl Iterator<Item = (SimTick, EntityGuid)> + '_ {
        self.applied.iter().copied()
    }

    /// Record that the input of `ship` for `tick` was applied outside
    /// [`SimWorld::step`] (a handed-over ship brought forward by the shard
    /// that installs it): see [`SimWorld::applied_inputs`].
    pub fn note_applied(&mut self, tick: SimTick, ship: EntityGuid) {
        self.applied.push_back((tick, ship));
        while self.applied.len() > APPLIED_LOG {
            self.applied.pop_front();
        }
    }

    /// How many inputs are currently buffered waiting for their tick.
    ///
    /// Exposed so a caller (and the buffer-bound tests) can observe that
    /// [`SimWorld::apply_inputs`] keeps this small no matter how much is thrown at
    /// it (see [`MAX_INPUT_LEAD`]).
    #[must_use]
    pub fn buffered_input_count(&self) -> usize {
        self.input_buf.values().map(BTreeMap::len).sum()
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
        for input in inputs.values() {
            // (An input filed for a ship that has since been handed away
            // names its ghost here. A ghost has neither `ShipControls` nor
            // an `Owner` — `import` gives them to ships only — so nothing
            // below writes to it: a ghost takes no input.)
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
            let leaving_fast = self
                .world
                .get::<ShipControls>(entity)
                .is_some_and(|c| c.fast && !input.fast);
            if let Some(controls) = self.world.get_mut::<ShipControls>(entity) {
                controls.thrust = input.thrust;
                controls.torque = input.torque;
                controls.fast = input.fast;
                self.note_applied(next_tick, input.entity);
            }
            // Leaving fast travel drops the ship to its normal top speed at
            // once (rather than coasting on at fast-travel speed).
            if leaving_fast {
                let top = f64::from(crate::physics::top_speed(&self.params));
                let too_fast = self.world.get::<ShipState>(entity).is_some_and(|s| {
                    let v = s.vel.map(f64::from);
                    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() > top
                });
                if too_fast {
                    if let Some(s) = self.world.get_mut::<ShipState>(entity) {
                        crate::physics::limit_speed(s, top);
                    }
                }
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

    /// Advance the world by one fixed step and return the new tick.
    ///
    /// # Inputs
    ///
    /// First the buffered inputs are pruned: any batch for a tick at or before the
    /// current tick is stale (that tick is already simulated) and is dropped. Then
    /// the batch for the tick this step produces (`tick + 1`) is drained and
    /// applied (see [`SimWorld::apply_inputs`]). A tick with no batch is a
    /// coasting tick.
    ///
    /// # Physics
    ///
    /// Every entity of the physics — ships, bodies, articulated objects;
    /// ghosts only as obstacles — is advanced by [`crate::rigid`]: forces,
    /// gravity and springs, then the contact and joint solver in the physics
    /// bubbles that can have a contact, then free flight for everything the
    /// solver did not touch. A ship's velocity change is its flight model's
    /// ([`crate::physics::ship_velocity_step`]); in free flight its new state
    /// is exactly what [`crate::step_ship`] gives.
    ///
    /// # Only a real change stamps the row
    ///
    /// Each body is stepped into a **copy** and its row is written through
    /// `get_mut` (which stamps the change tick) **only if the copy's bits
    /// differ from the pre-step bits**. A body at rest, a settled ship (see
    /// [`crate::physics::SETTLE_EPS`]) and a body the physics has put to sleep
    /// are therefore not stamped and not replicated.
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
    /// Only entities spawned through this world ([`SimWorld::spawn_ship`],
    /// [`SimWorld::spawn_body`], [`SimWorld::import_entity`],
    /// [`SimWorld::insert_ghost`]) are simulated; [`ShardAuthority`] is the
    /// marker the *caller* uses to decide which entities belong to this
    /// shard's world at all.
    pub fn step(&mut self) -> SimTick {
        let next_tick = self.tick.wrapping_add(1);
        let params = self.params;
        // 0. Drop stale buffered batches (a tick already simulated), then apply
        //    the batch for the tick this step produces.
        self.input_buf.retain(|&t, _| t > self.tick);
        self.drain_inputs(next_tick);
        // 1. Physics, with gravity sampled at the start of the step.
        self.physics.step(&mut self.world, self.tick, &params);
        // 2. Consume the controls of the ships that had them: a ship with no
        //    input next tick coasts. Only non-zero controls are reset, so the
        //    write does not stamp an already-idle row.
        let ships: Vec<Entity> = self.live_ships().collect();
        for entity in ships {
            let zero = self.world.get::<ShipControls>(entity).is_none_or(|c| {
                c.thrust.iter().all(|v| *v == 0.0) && c.torque.iter().all(|v| *v == 0.0)
            });
            if !zero {
                if let Some(controls) = self.world.get_mut::<ShipControls>(entity) {
                    // Fast travel is a mode, not an input: it stays.
                    *controls = ShipControls {
                        fast: controls.fast,
                        ..ShipControls::default()
                    };
                }
            }
        }
        // 3. Close the tick: both counters move together (see the tick-ordering
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
    /// entities in a different order hash differently), this hash is keyed by
    /// [`EntityGuid`]: it walks the live entities (ghosts excluded: they are
    /// another world's state) **sorted by guid** and hashes, for each, the
    /// components the replication registry marks `replicated == true` —
    /// [`ShipState`], [`Owner`] and the physics components [`Body`],
    /// [`Visual`], [`Springs`], [`Articulation`], [`PartStates`],
    /// [`PosResidual`] — using the
    /// lossless wire bytes (the same bytes a snapshot/delta carries).
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
        let mut ships: Vec<(EntityGuid, Entity)> = self
            .by_guid
            .iter()
            .filter(|(g, &e)| self.world.entity_is_alive(e) && !self.physics.is_ghost(**g))
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
            // The physics components (ids 4..=8): id, length, wire bytes.
            self.put_physics_components(&mut out, entity);
        }
        fnv1a(FNV_OFFSET, &out)
    }

    /// Append the wire bytes of `entity`'s physics components (ids 4..=8, in
    /// id order) as `id u8, length u32, bytes`.
    fn put_physics_components(&self, out: &mut Vec<u8>, entity: Entity) {
        let w = &self.world;
        let mut put = |id: ComponentId, bytes: Vec<u8>| {
            out.push(id as u8);
            put_int(out, i128::try_from(bytes.len()).unwrap_or(0), 4);
            out.extend_from_slice(&bytes);
        };
        if let Some(b) = w.get::<Body>(entity) {
            put(Body::ID, encode_body(b));
        }
        if let Some(v) = w.get::<Visual>(entity) {
            put(Visual::ID, encode_visual(*v));
        }
        if let Some(s) = w.get::<Springs>(entity) {
            put(Springs::ID, encode_springs(s));
        }
        if let Some(a) = w.get::<Articulation>(entity) {
            put(Articulation::ID, encode_articulation(a));
        }
        if let Some(p) = w.get::<PartStates>(entity) {
            put(PartStates::ID, encode_part_states(p));
        }
        if let Some(r) = w.get::<PosResidual>(entity) {
            put(PosResidual::ID, encode_residual(r));
        }
    }

    /// The canonical serialisation of every entity's state, sorted by
    /// [`space_ecs::Entity`], followed by the physics' derived state.
    ///
    /// Layout (all integers big endian, all floats as their IEEE-754 bits, big
    /// endian — the same byte order as [`space_protocol::codec`]):
    ///
    /// ```text
    /// tick:u64 | entity_count:u64
    /// entity: index:u32 | generation:u32 | present:u16 | components in id order
    ///   ShipState      : cell 3×i64 | local 3×f32 | vel 3×f32 | quat 4×f32 | ang_vel 3×f32
    ///   ShipControls   : thrust 3×f32 | torque 3×f32 | fast:u8
    ///   Owner          : u64
    ///   ShardAuthority : u32
    ///   Body .. PartStates (ids 4..=8), PosResidual (10): id:u8 | len:u32 | wire bytes
    ///   Ghost          : (marker, no payload)
    /// bubble_count:u64, per bubble: id:u64 | anchor cell 3×i64 | member_count:u64 | members (shard:u32 serial:u64)…
    /// sleeping_count:u64, sleeping guids (shard:u32 serial:u64)…
    /// ```
    ///
    /// `present` is a bitmask over [`crate::components::IDS`] (bit `i` = id `i`),
    /// so a component an entity does not carry is *absent*, not defaulted.
    #[must_use]
    pub fn serialize_state(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_int(&mut out, i128::from(self.tick), 8);
        // Ships and every physics entity, in the canonical order (ascending
        // index, then generation).
        let mut entities: Vec<Entity> = self.ships();
        entities.extend(
            self.physics
                .guids()
                .map(|(_, e, _)| e)
                .filter(|&e| self.world.entity_is_alive(e)),
        );
        entities.sort_unstable_by_key(|e| (e.index(), e.generation()));
        entities.dedup();
        put_int(
            &mut out,
            i128::try_from(entities.len()).unwrap_or(i128::MAX),
            8,
        );
        for entity in entities {
            put_int(&mut out, i128::from(entity.index()), 4);
            put_int(&mut out, i128::from(entity.generation()), 4);
            let w = &self.world;
            let mut present = 0u16;
            for (bit, has) in [
                w.get::<ShipState>(entity).is_some(),
                w.get::<ShipControls>(entity).is_some(),
                w.get::<Owner>(entity).is_some(),
                w.get::<ShardAuthority>(entity).is_some(),
                w.get::<Body>(entity).is_some(),
                w.get::<Visual>(entity).is_some(),
                w.get::<Springs>(entity).is_some(),
                w.get::<Articulation>(entity).is_some(),
                w.get::<PartStates>(entity).is_some(),
                w.get::<Ghost>(entity).is_some(),
                w.get::<PosResidual>(entity).is_some(),
            ]
            .into_iter()
            .enumerate()
            {
                if has {
                    present |= 1 << bit;
                }
            }
            put_int(&mut out, i128::from(present), 2);
            if let Some(s) = w.get::<ShipState>(entity) {
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
            if let Some(c) = w.get::<ShipControls>(entity) {
                for &v in &c.thrust {
                    put_f32(&mut out, v);
                }
                for &v in &c.torque {
                    put_f32(&mut out, v);
                }
                put_int(&mut out, i128::from(u8::from(c.fast)), 1);
            }
            if let Some(o) = w.get::<Owner>(entity) {
                put_int(&mut out, i128::from(o.0), 8);
            }
            if let Some(a) = w.get::<ShardAuthority>(entity) {
                put_int(&mut out, i128::from(a.0), 4);
            }
            self.put_physics_components(&mut out, entity);
        }
        self.physics.hash_bytes(&mut out);
        out
    }
}

/// The comparison key of an input: its raw bits, so the ordering is total and
/// independent of arrival order.
fn input_bits(input: &PlayerInput) -> ([u32; 3], [u32; 3], bool) {
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
        input.fast,
    )
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
            fast: false,
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
            fast: false,
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
                    fast: false,
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
                    fast: false,
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
                    fast: false,
                })
                .collect();
            batch.push(PlayerInput {
                entity: guid(1),
                thrust: [0.25, 0.0, 0.0],
                torque: [0.0; 3],
                tick,
                fast: false,
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

    /// Filing keeps exactly one input per entity per tick — the one with the
    /// greatest bit pattern — whatever order a tick's inputs arrive in, and the
    /// step then behaves exactly as if only the winners had been sent. (The
    /// bucket is a map keyed by entity now; this pins the merge rule the old
    /// linear scan implemented.)
    #[test]
    fn one_input_per_entity_per_tick_whatever_the_arrival_order() {
        const SHIPS: u64 = 64;
        let world = || {
            let mut sim = SimWorld::new(ShardId(1));
            for serial in 1..=SHIPS {
                sim.spawn_ship(guid(serial), serial, ShipState::IDENTITY);
            }
            sim
        };
        // Three candidates per ship; the largest positive thrust has the
        // greatest bits. The winner differs per ship so a mix-up would show.
        let candidates = |serial: u64| -> [PlayerInput; 3] {
            let top = 0.25 + (serial % 3) as f32 * 0.25;
            [
                input(serial, [top * 0.5, 0.0, 0.0]),
                input(serial, [top, 0.0, 0.0]),
                input(serial, [0.0, 0.0, 0.0]),
            ]
        };
        let all: Vec<PlayerInput> = (1..=SHIPS).flat_map(&candidates).collect();
        let winners: Vec<PlayerInput> = (1..=SHIPS).map(|s| candidates(s)[1]).collect();

        let mut in_order = world();
        in_order.apply_inputs(1, &all);
        let mut reversed = world();
        let backwards: Vec<PlayerInput> = all.iter().rev().copied().collect();
        // Split over two calls, as two clients' batches would be.
        reversed.apply_inputs(1, &backwards[..100]);
        reversed.apply_inputs(1, &backwards[100..]);
        let mut only_winners = world();
        only_winners.apply_inputs(1, &winners);

        for sim in [&in_order, &reversed, &only_winners] {
            assert_eq!(sim.buffered_input_count() as u64, SHIPS);
        }
        for sim in [&mut in_order, &mut reversed, &mut only_winners] {
            sim.step();
            assert_eq!(sim.buffered_input_count(), 0);
        }
        assert_eq!(in_order.state_hash(), only_winners.state_hash());
        assert_eq!(reversed.state_hash(), only_winners.state_hash());
        // And the winner really was applied: every ship moved.
        for serial in 1..=SHIPS {
            let s = in_order.ship_state(guid(serial)).unwrap();
            assert!(s.vel[0] > 0.0, "ship {serial} did not thrust: {:?}", s.vel);
        }
    }

    /// The live-ship list is always in canonical `(index, generation)` order —
    /// through despawns, recycled indices and an entity despawned behind the
    /// world's back — and agrees with `ship_count`.
    #[test]
    fn ships_are_listed_in_entity_order_through_despawns_and_reuse() {
        let key = |e: space_ecs::Entity| (e.index(), e.generation());
        let mut sim = SimWorld::new(ShardId(1));
        let mut live: Vec<(u64, space_ecs::Entity)> = Vec::new();
        for serial in 1..=20u64 {
            live.push((
                serial,
                sim.spawn_ship(guid(serial), serial, ShipState::IDENTITY),
            ));
        }
        // Despawn in a scrambled order, so the free list is scrambled too.
        for serial in [7u64, 19, 2, 11, 14, 3] {
            assert!(sim.despawn_ship(guid(serial)));
            live.retain(|(s, _)| *s != serial);
        }
        // Respawn: the new ships reuse those indices with a new generation.
        for serial in 21..=30u64 {
            live.push((
                serial,
                sim.spawn_ship(guid(serial), serial, ShipState::IDENTITY),
            ));
        }
        assert!(
            live.iter().any(|(_, e)| e.generation() > 0),
            "the test must recycle at least one index"
        );

        let check = |sim: &SimWorld, live: &[(u64, space_ecs::Entity)]| {
            let listed = sim.ships();
            assert!(
                listed.windows(2).all(|w| key(w[0]) < key(w[1])),
                "ships() must be strictly ascending: {listed:?}"
            );
            let mut expected: Vec<space_ecs::Entity> = live.iter().map(|(_, e)| *e).collect();
            expected.sort_unstable_by_key(|&e| key(e));
            assert_eq!(listed, expected);
            assert_eq!(sim.ship_count(), expected.len());
        };
        check(&sim, &live);

        // An entity despawned through the raw ECS world is filtered out.
        let (gone_serial, gone) = live[4];
        assert!(sim.world_mut().despawn(gone));
        live.retain(|(s, _)| *s != gone_serial);
        check(&sim, &live);
        // ...and despawning it properly afterwards is a no-op that still tidies up.
        assert!(!sim.despawn_ship(guid(gone_serial)));
        check(&sim, &live);
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
                    fast: false,
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
