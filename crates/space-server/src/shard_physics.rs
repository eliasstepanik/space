//! The physics half of [`Shard`]: the scene's bodies, free spawn points,
//! ghosts at the borders, island consolidation and the body handoff path.
//! The rules themselves are [`crate::border`]'s; this is where the shard
//! applies them to its world.

use std::collections::{BTreeSet, HashMap, HashSet};

use space_ecs::ComponentDef;
use space_protocol::{
    EntityGuid, EntitySnapshot, GhostUpdate, HandoffOffer, HandoffReason, IslandKey, Message,
    Scene, SceneState, ShardId,
};
use space_sim::{
    Body, BodySpawn, IslandQuery, Owner, PartStates, PhysicsError, PosResidual, ShipControls,
    ShipState, SimTick, SimWorld, SpringTarget, Springs,
};
use space_world::WorldPos;

use super::{Destination, PendingCommit, Shard};
use crate::border;
use crate::handoff::{Ledger, TransferKind, TRANSFER_COMPONENT};
use crate::replicate::{self, Components};

/// Spacing of the spawn grid, metres: a ship is 0.6 m wide, so two players
/// on neighbouring points stand 0.9 m apart.
pub const SPAWN_SPACING: f64 = 1.5;
/// Rings of the spawn grid around the spawn point (21 × 21 points, 15 m out).
pub const SPAWN_RINGS: i32 = 10;
/// How far a new ship must be from anything, metres. Less than the 2 cm the
/// ship's capsule floats above the demo area's floor at the spawn point.
pub const SPAWN_CLEARANCE: f64 = 0.01;

/// A ghost held here: the copy of a body another shard simulates.
#[derive(Debug)]
pub(super) struct GhostEntry {
    /// The shard that simulates it.
    pub(super) from: ShardId,
    /// A ship (no hysteresis; replicated as a ship nobody here flies).
    ship: bool,
    /// Its replicated components as last received (`Owner` set to 0: a
    /// session of another shard means nothing here, and must never be
    /// mistaken for one of ours).
    components: Components,
    /// Ticks it has been dead-reckoned since the last update.
    reckoned: u32,
    /// The island key its owner announced: `(lowest guid, home)`.
    island: Option<(EntityGuid, ShardId)>,
    /// The tick this shard handed the entity to `from` and kept this ghost
    /// of it, if it did: `from`'s full refresh may have been sent before the
    /// commit reached it, so for a while only an explicit removal counts.
    departed_at: Option<SimTick>,
}

/// The clocks of one body's time hysteresis (see [`crate::border`],
/// "Hysteresis").
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct BodyClock {
    /// The shard whose territory the map has put the body in, continuously,
    /// and since when (`None` while it is in ours).
    outside: Option<(ShardId, SimTick)>,
    /// The last tick it was in an island of two or more (ghosts counted).
    grouped_at: Option<SimTick>,
    /// The other shard its island has wanted to go to, continuously, and
    /// since when.
    want: Option<(ShardId, SimTick)>,
    /// Its hold, ticks (0: [`border::HOLD_TICKS`]): longer for a body that
    /// keeps coming back here soon after it was sent home.
    hold: SimTick,
}

/// What a neighbour was last told about one of our bodies.
#[derive(Debug, Default)]
pub(super) struct SentGhost {
    components: Components,
    island: Option<(EntityGuid, ShardId)>,
}

/// The islands of the last border pass.
#[derive(Debug, Default)]
pub(super) struct IslandView {
    /// Local entity → its island.
    of: HashMap<EntityGuid, usize>,
    islands: Vec<IslandInfo>,
}

#[derive(Debug)]
struct IslandInfo {
    /// The entities of this shard in it, ascending.
    members: Vec<EntityGuid>,
    /// The ghosts touching them, ascending.
    ghosts: Vec<EntityGuid>,
    /// The one-hop key this shard announces for its members.
    announce: (EntityGuid, ShardId),
    /// The shard the island belongs on (the two-hop key's home).
    target: ShardId,
    /// Whether going there is a body going home after an interaction (the
    /// island touches no ghost), not a consolidation.
    going_home: bool,
}

impl IslandView {
    fn announce_of(&self, guid: EntityGuid) -> Option<(EntityGuid, ShardId)> {
        self.of.get(&guid).map(|&i| self.islands[i].announce)
    }

    /// Whether the island of our entity `member` touches the ghost `ghost`.
    fn touches(&self, member: EntityGuid, ghost: EntityGuid) -> bool {
        self.of
            .get(&member)
            .is_some_and(|&i| self.islands[i].ghosts.binary_search(&ghost).is_ok())
    }
}

/// One physics entity as [`Shard::body_list`] reports it.
#[derive(Debug, Clone, PartialEq)]
pub struct BodyInfo {
    /// Its guid.
    pub guid: EntityGuid,
    /// Its motion (of part 0).
    pub state: ShipState,
    /// A ghost of another shard's body (`simulator` names that shard).
    pub ghost: bool,
    /// The shard simulating it.
    pub simulator: ShardId,
    /// A ship (a player's, or one nobody flies).
    pub ship: bool,
    /// The model a client draws for it, if it says.
    pub visual: Option<u32>,
    /// Whether the physics has put it to sleep.
    pub sleeping: bool,
    /// The shard its island belongs on (last border pass).
    pub island_home: Option<ShardId>,
}

const SHIP_STATE: u16 = <ShipState as ComponentDef>::ID;
const CONTROLS: u16 = <ShipControls as ComponentDef>::ID;
const OWNER: u16 = <Owner as ComponentDef>::ID;
const PART_STATES: u16 = <PartStates as ComponentDef>::ID;
const RESIDUAL: u16 = <PosResidual as ComponentDef>::ID;

/// The replicated components of `guid`, a local entity or a ghost (a ghost's
/// as its owner sent them, with the motion it is driven with here).
pub(super) fn replicated_of(
    sim: &SimWorld,
    ghosts: &HashMap<EntityGuid, GhostEntry>,
    guid: EntityGuid,
) -> Option<Components> {
    match ghosts.get(&guid) {
        Some(entry) if sim.is_ghost(guid) => {
            let live = sim.export_entity(guid)?;
            let mut out = entry.components.clone();
            for id in [SHIP_STATE, PART_STATES] {
                if let Some((_, bytes)) = live.iter().find(|(i, _)| *i == id) {
                    if let Some(slot) = out.iter_mut().find(|(i, _)| *i == id) {
                        slot.1.clone_from(bytes);
                    }
                }
            }
            Some(out)
        }
        _ => replicate::replicated_components(sim, guid),
    }
}

/// `list` without the handoff's private components.
fn strip_private(list: &[(u16, Vec<u8>)]) -> Components {
    list.iter()
        .filter(|(id, _)| *id < TRANSFER_COMPONENT)
        .cloned()
        .collect()
}

/// `list` with its `Owner` (if any) set to 0.
fn zero_owner(mut list: Components) -> Components {
    for (id, bytes) in &mut list {
        if *id == OWNER {
            *bytes = space_protocol::codec::encode_raw(&0u64);
        }
    }
    list
}

/// `old` updated by the partial list `update` (an empty payload removes).
fn merge(old: &[(u16, Vec<u8>)], update: &[(u16, Vec<u8>)]) -> Components {
    let mut out: Components = old.to_vec();
    for (id, bytes) in update {
        out.retain(|(i, _)| i != id);
        if !bytes.is_empty() {
            out.push((*id, bytes.clone()));
        }
    }
    out.sort_by_key(|(id, _)| *id);
    out
}

/// What changed from `old` to `new`: new or different components, and an
/// empty payload for each one that went away.
fn diff(old: &[(u16, Vec<u8>)], new: &[(u16, Vec<u8>)]) -> Components {
    let mut out: Components = new
        .iter()
        .filter(|(id, bytes)| !old.iter().any(|(i, b)| i == id && b == bytes))
        .cloned()
        .collect();
    for (id, _) in old {
        if !new.iter().any(|(i, _)| i == id) {
            out.push((*id, Vec::new()));
        }
    }
    out.sort_by_key(|(id, _)| *id);
    out
}

/// Whether two component lists differ in anything but motion and identity
/// (a ghost then has to be rebuilt rather than driven).
fn structure_differs(old: &[(u16, Vec<u8>)], new: &[(u16, Vec<u8>)]) -> bool {
    let shape = |list: &[(u16, Vec<u8>)]| -> Components {
        list.iter()
            .filter(|(id, _)| ![SHIP_STATE, CONTROLS, OWNER, PART_STATES, RESIDUAL].contains(id))
            .cloned()
            .collect()
    };
    shape(old) != shape(new)
}

fn component(list: &[(u16, Vec<u8>)], id: u16) -> Option<&[u8]> {
    list.iter()
        .find(|(i, _)| *i == id)
        .map(|(_, b)| b.as_slice())
}

/// The spawn grid around the spawn point: `(i, j)` steps of
/// [`SPAWN_SPACING`] along x and z, nearest first, then by bearing.
fn spawn_grid() -> Vec<(i32, i32)> {
    let mut out: Vec<(i32, i32)> = (-SPAWN_RINGS..=SPAWN_RINGS)
        .flat_map(|i| (-SPAWN_RINGS..=SPAWN_RINGS).map(move |j| (i, j)))
        .collect();
    out.sort_by(|a, b| {
        let r = |p: &(i32, i32)| p.0 * p.0 + p.1 * p.1;
        let angle = |p: &(i32, i32)| f64::from(p.1).atan2(f64::from(p.0));
        r(a).cmp(&r(b)).then(angle(a).total_cmp(&angle(b)))
    });
    out
}

impl Shard {
    // ------------------------------------------------------------- the scene

    /// Build the physics world for the scene this shard runs, if it has not
    /// been built for this scene at this anchor yet: the catalogue, statics
    /// and gravity everywhere, and — on the scene's spawner only (see
    /// [`border::scene_spawner`]) — the scene's bodies, replacing every body
    /// of a previous scene. A shard nobody gave a scene (revision 0)
    /// simulates an empty world.
    pub(super) fn ensure_scene(&mut self) {
        if !self.scene.is_set() {
            return;
        }
        let Some(anchor) = space_scene::scene_anchor(&self.shard_map) else {
            return;
        };
        if self.installed == Some((self.scene, anchor)) {
            return;
        }
        let scene = self.scene.scene.or_default();
        if let Err(e) = space_scene::install_environment(&mut self.sim, scene, anchor) {
            eprintln!(
                "space-server: {} cannot build the physics of {}: {e}",
                self.config.shard, self.scene
            );
            return;
        }
        self.installed = Some((self.scene, anchor));
        self.replace_scene_bodies(scene, anchor);
    }

    /// Despawn every scene body here (this shard's, and the ghosts of its
    /// neighbours'), then spawn the current scene's share of them.
    fn replace_scene_bodies(&mut self, scene: Scene, anchor: WorldPos) {
        let revision = self.scene.revision;
        let old: Vec<EntityGuid> = self
            .sim
            .bodies()
            .into_iter()
            .map(|(guid, _, _)| guid)
            .filter(|guid| border::scene_revision_of(*guid).is_some())
            .collect();
        for guid in old {
            if self.sim.is_ghost(guid) {
                self.remove_ghost(guid);
            } else {
                if self.ledger.activation_of(guid).is_some() {
                    self.abort_outgoing(guid, HandoffReason::Despawned);
                }
                self.despawn_ship(guid);
            }
        }
        let staged: Vec<EntityGuid> = self
            .ledger
            .incoming()
            .keys()
            .copied()
            .filter(|guid| border::scene_revision_of(*guid).is_some())
            .collect();
        for guid in staged {
            self.abort_incoming(guid, HandoffReason::Despawned);
        }
        self.pending_commits
            .retain(|p| border::scene_revision_of(p.guid).is_none());

        // One shard spawns the whole scene, with guids every shard can name;
        // a body outside its region is handed to the region's owner by the
        // ordinary handoff on the next ticks.
        let Some(spawner) = border::scene_spawner(&self.shard_map) else {
            return;
        };
        if spawner != self.config.shard {
            return;
        }
        let mut index = 0usize;
        let placed = space_scene::bodies(scene, anchor, &mut || {
            let guid = border::scene_guid(spawner, revision, index);
            index += 1;
            guid
        });
        let total = placed.len();
        let mut spawned = 0usize;
        for body in placed {
            let pos = body.spawn.state.pos;
            match self.sim.spawn_body(body.guid, body.spawn) {
                Ok(entity) => {
                    self.guids.insert(body.guid, entity);
                    self.interest.upsert(body.guid, entity, pos);
                    spawned += 1;
                }
                Err(e) => eprintln!(
                    "space-server: {} cannot spawn the scene body {}: {e}",
                    self.config.shard, body.name
                ),
            }
        }
        if total > 0 {
            println!(
                "space-server: {} spawned {spawned} of the {total} bodies of {}",
                self.config.shard, self.scene
            );
        }
    }

    /// Every five seconds, one log line with what this shard simulates and
    /// what it handed over — when it changed since the last one.
    pub(super) fn report_census(&mut self, tick: SimTick) {
        if !tick.is_multiple_of(5 * SimTick::from(space_sim::TICK_HZ)) {
            return;
        }
        let mut now = self.stats();
        now.tick = 0;
        now.clients = 0;
        if self.census == Some(now) {
            return;
        }
        self.census = Some(now);
        println!(
            "space-server: {} at tick {tick}: {} ships, {} bodies, {} ghosts; handoffs {} out, {} in, {} aborted",
            self.config.shard, now.ships, now.bodies, now.ghosts, now.handed_off, now.received, now.aborted
        );
    }

    /// The scene the physics world was built for and its anchor, if any.
    #[must_use]
    pub fn installed_scene(&self) -> Option<(SceneState, WorldPos)> {
        self.installed
    }

    // ---------------------------------------------------------- spawn points

    /// Where a joining player's ship starts: the scene's spawn point — the
    /// map's anchor, [`space_scene::scene_anchor`] — if this shard owns it,
    /// else the centre of this shard's region; or, where a ship, body or
    /// static is in the way, the nearest free point of a grid around it
    /// ([`SPAWN_SPACING`] apart, [`SPAWN_RINGS`] rings out, nearest first,
    /// then by bearing: deterministic). The base point itself if none is
    /// free.
    ///
    /// **Around the spawn point, only points of this shard's own territory
    /// are handed out.** The grid reaches 15 m and the borders of the cell
    /// the spawn point lies in pass 6 m and 4 m from it
    /// ([`space_scene::ANCHOR_OFFSET`]): once the cells beside it belong to
    /// other shards, a busy spawn would otherwise put a new ship on a
    /// neighbour's ground — a player whose first second is a handoff. A point
    /// across a border is skipped like an occupied one. (A shard that does not
    /// own the spawn point and falls back to its region's centre is not held
    /// to this: that centre may itself be another shard's, in which case the
    /// ship is that shard's from its first tick and the ordinary handoff takes
    /// it there.)
    ///
    /// The spawn point is a function of the map, not of the scene: a shard
    /// nobody gave a scene puts its players on the same point its clients
    /// measure from.
    #[must_use]
    pub fn free_spawn_point(&self) -> WorldPos {
        let own = |pos: &WorldPos| self.shard_map.lookup(pos) == Some(self.config.shard);
        let spawn = space_scene::scene_anchor(&self.shard_map).filter(|anchor| own(anchor));
        let base = spawn.unwrap_or_else(|| self.config.region.center());
        let body = Body::ship(&self.sim.params());
        for (i, j) in spawn_grid() {
            let pos = WorldPos::from_f64_metres_around(
                base,
                [
                    f64::from(i) * SPAWN_SPACING,
                    0.0,
                    f64::from(j) * SPAWN_SPACING,
                ],
            );
            if spawn.is_some() && !own(&pos) {
                continue; // across a border: a neighbour's ground
            }
            let blocked = self
                .sim
                .overlaps(&body, &ShipState::at_rest(pos), SPAWN_CLEARANCE)
                .unwrap_or(true);
            if !blocked {
                return pos;
            }
        }
        base
    }

    // ------------------------------------------------------------ the index

    /// Keep the guid index and the interest grid in step with the world:
    /// every physics entity, ships, bodies and ghosts alike.
    pub(super) fn refresh_index(&mut self) {
        for (guid, entity, _) in self.sim.bodies() {
            if self.guids.entity_of(guid) != Some(entity) {
                self.guids.insert(guid, entity);
            }
            if let Some(state) = self.sim.state_of(guid) {
                self.interest.upsert(guid, entity, state.pos);
            }
        }
    }

    /// Index `guid` (just spawned or rebuilt) right away.
    fn index_now(&mut self, guid: EntityGuid) {
        if let (Some(entity), Some(state)) = (self.sim.entity_of(guid), self.sim.state_of(guid)) {
            self.guids.insert(guid, entity);
            self.interest.upsert(guid, entity, state.pos);
        }
    }

    // ------------------------------------------------------- tools and tests

    /// Spawn a body (not a ship) simulated here, minting its guid from this
    /// shard's counter.
    ///
    /// # Errors
    /// What [`SimWorld::spawn_body`] refuses.
    pub fn spawn_body(&mut self, spawn: BodySpawn) -> Result<EntityGuid, PhysicsError> {
        let guid = self.mint_guid();
        self.sim.spawn_body(guid, spawn)?;
        self.index_now(guid);
        Ok(guid)
    }

    /// Put the ship or body `guid` this shard simulates at `state` (an admin
    /// teleport, a test). The write is behind the physics' back, so the body
    /// wakes and its position residual starts again from zero. `false` if
    /// this shard does not simulate `guid`.
    pub fn set_state(&mut self, guid: EntityGuid, state: ShipState) -> bool {
        if self.sim.ship_state(guid).is_none() {
            return false;
        }
        let Some(entity) = self.sim.entity_of(guid) else {
            return false;
        };
        if let Some(s) = self.sim.world_mut().get_mut::<ShipState>(entity) {
            *s = state;
        }
        self.index_now(guid);
        true
    }

    /// Every physics entity here — ships, bodies and ghosts — by guid.
    #[must_use]
    pub fn body_list(&self) -> Vec<BodyInfo> {
        self.sim
            .bodies()
            .into_iter()
            .filter_map(|(guid, entity, ghost)| {
                let state = self.sim.state_of(guid)?;
                let w = self.sim.world();
                Some(BodyInfo {
                    guid,
                    state,
                    ghost,
                    simulator: if ghost {
                        self.ghosts.get(&guid).map_or(ShardId::NONE, |e| e.from)
                    } else {
                        self.config.shard
                    },
                    ship: if ghost {
                        self.ghosts.get(&guid).is_some_and(|e| e.ship)
                    } else {
                        w.get::<ShipControls>(entity).is_some()
                    },
                    visual: w.get::<space_sim::Visual>(entity).map(|v| v.0),
                    sleeping: self.sim.is_sleeping(guid).unwrap_or(false),
                    island_home: self
                        .island_view
                        .of
                        .get(&guid)
                        .map(|&i| self.island_view.islands[i].target),
                })
            })
            .collect()
    }

    /// The shard simulating the ghost `guid`, if it is one here.
    #[must_use]
    pub fn ghost_owner(&self, guid: EntityGuid) -> Option<ShardId> {
        self.ghosts
            .get(&guid)
            .filter(|_| self.sim.is_ghost(guid))
            .map(|e| e.from)
    }

    /// Bytes per component a client is sent for `guid` in full (tests,
    /// tools): what a snapshot of it costs.
    #[must_use]
    pub fn replicated_entity(&self, guid: EntityGuid) -> Option<EntitySnapshot> {
        replicated_of(&self.sim, &self.ghosts, guid).map(|list| EntitySnapshot::new(guid, list))
    }

    // ------------------------------------------------------------- islands

    /// The shard a body of ours belongs on by its own position: the
    /// region's owner once it is [`border::BODY_HYSTERESIS`] inside it, or
    /// once it has been inside it for [`border::SETTLE_TICKS`] (a body
    /// resting just past a border is settled there too).
    fn home_local(&self, guid: EntityGuid) -> ShardId {
        let Some(state) = self.sim.ship_state(guid) else {
            return self.config.shard;
        };
        let ship = self
            .sim
            .entity_of(guid)
            .is_some_and(|e| self.sim.world().get::<ShipControls>(e).is_some());
        let hysteresis = if ship { 0.0 } else { border::BODY_HYSTERESIS };
        let home = border::home(&self.shard_map, state.pos, self.config.shard, hysteresis);
        if home != self.config.shard {
            return home;
        }
        let now = self.sim.tick();
        match self.clocks.get(&guid).and_then(|c| c.outside) {
            Some((owner, since))
                if now.saturating_sub(since) >= border::SETTLE_TICKS
                    && self.shard_map.lookup(&state.pos) == Some(owner) =>
            {
                owner
            }
            _ => home,
        }
    }

    /// Wind the clocks of every body this shard simulates: whose territory
    /// it is in, and since when. (Bodies that left are forgotten.)
    fn wind_clocks(&mut self) {
        let now = self.sim.tick();
        let own = self.config.shard;
        let mut clocks = std::mem::take(&mut self.clocks);
        let mut kept: HashMap<EntityGuid, BodyClock> = HashMap::new();
        for (guid, _, ghost) in self.sim.bodies() {
            if ghost {
                continue;
            }
            let Some(state) = self.sim.ship_state(guid) else {
                continue;
            };
            let mut clock = clocks.remove(&guid).unwrap_or_default();
            clock.outside = match self.shard_map.lookup(&state.pos) {
                Some(owner) if owner != own && owner != ShardId::NONE => match clock.outside {
                    Some((was, since)) if was == owner => Some((owner, since)),
                    _ => Some((owner, now)),
                },
                _ => None,
            };
            kept.insert(guid, clock);
        }
        self.clocks = kept;
    }

    /// Whether an island of ours with no ghost touching it may go to
    /// `target` now: unless one of its members was in a group within the
    /// last [`border::HOLD_TICKS`], at once; otherwise only once every member
    /// has wanted to go there for that long, continuously.
    fn ready_to_go(&self, members: &[EntityGuid], target: ShardId) -> bool {
        let now = self.sim.tick();
        members.iter().all(|guid| {
            let clock = self.clocks.get(guid).copied().unwrap_or_default();
            let hold = clock.hold.max(border::HOLD_TICKS);
            let quiet = clock
                .grouped_at
                .is_none_or(|at| now.saturating_sub(at) > hold);
            let wanted = clock
                .want
                .is_some_and(|(to, since)| to == target && now.saturating_sub(since) >= hold);
            quiet || wanted
        })
    }

    /// `members` are being sent home after an interaction: remember it, so
    /// that if they are back here soon their next hold is longer.
    pub(super) fn note_sent_home(&mut self, members: &[EntityGuid], now: SimTick) {
        self.sent_home
            .retain(|_, (_, at)| now.saturating_sub(*at) <= border::RETURN_WINDOW_TICKS);
        for guid in members {
            let entry = self.sent_home.entry(*guid).or_insert((0, now));
            entry.0 = entry.0.saturating_add(1);
            entry.1 = now;
        }
    }

    /// A body arrived: if this shard sent it home a moment ago, it is back
    /// because it keeps meeting something of ours — hold it longer this time.
    fn back_again(&mut self, guid: EntityGuid) {
        let now = self.sim.tick();
        if let Some(&(times, at)) = self.sent_home.get(&guid) {
            if now.saturating_sub(at) <= border::RETURN_WINDOW_TICKS {
                let doublings = times.min(border::MAX_HOLD_DOUBLINGS);
                self.clocks.entry(guid).or_default().hold = border::HOLD_TICKS << doublings;
            } else {
                self.sent_home.remove(&guid);
            }
        }
    }

    /// Record, after deciding, that `members` were in a group of two or more
    /// (`grouped`) and where their island wanted to go.
    fn note_island(&mut self, members: &[EntityGuid], grouped: bool, target: ShardId) {
        let now = self.sim.tick();
        let own = self.config.shard;
        for guid in members {
            let clock = self.clocks.entry(*guid).or_default();
            if grouped {
                clock.grouped_at = Some(now);
            }
            clock.want = if target == own || target == ShardId::NONE {
                None
            } else {
                match clock.want {
                    Some((to, since)) if to == target => Some((to, since)),
                    _ => Some((target, now)),
                }
            };
        }
    }

    /// The shard a ghost belongs on by its own position, as its owner
    /// computes it.
    fn home_ghost(&self, guid: EntityGuid) -> ShardId {
        let Some(entry) = self.ghosts.get(&guid) else {
            return ShardId::NONE;
        };
        let Some(state) = self.sim.state_of(guid) else {
            return entry.from;
        };
        let hysteresis = if entry.ship {
            0.0
        } else {
            border::BODY_HYSTERESIS
        };
        border::home(&self.shard_map, state.pos, entry.from, hysteresis)
    }

    /// The border pass: group this shard's entities into islands (with the
    /// ghosts touching them), give every island its key and its home (see
    /// [`crate::border`], "Islands and consolidation").
    pub(super) fn border_pass(&mut self) {
        self.wind_clocks();
        let islands = self.sim.islands(IslandQuery {
            margin: border::ISLAND_MARGIN,
            lead_time: border::ISLAND_LEAD,
            springs: false,
        });
        let mut view = IslandView::default();
        for island in islands {
            let ghosts: Vec<EntityGuid> = island
                .touching
                .iter()
                .copied()
                .filter(|g| self.ghosts.contains_key(g) && self.sim.is_ghost(*g))
                .collect();
            let mut key1: Option<(EntityGuid, ShardId)> = None;
            let mut consider = |candidate: (EntityGuid, ShardId)| {
                if key1.is_none_or(|k| candidate.0 < k.0) {
                    key1 = Some(candidate);
                }
            };
            for guid in &island.bodies {
                consider((*guid, self.home_local(*guid)));
            }
            for guid in &ghosts {
                consider((*guid, self.home_ghost(*guid)));
            }
            let Some(key1) = key1 else {
                continue;
            };
            // Two hops: what the owners of the touching ghosts announced.
            let mut key2 = key1;
            for guid in &ghosts {
                let Some(entry) = self.ghosts.get(guid) else {
                    continue;
                };
                let Some(announced) = entry.island else {
                    continue;
                };
                if announced.0 < key2.0 {
                    key2 = announced;
                } else if announced.0 == key2.0
                    && self
                        .ghosts
                        .get(&announced.0)
                        .is_some_and(|min| min.from == entry.from)
                {
                    // The shard simulating the lowest member knows its home
                    // better than our view of its ghost does.
                    key2.1 = announced.1;
                }
            }
            // An island touching another shard's bodies goes where its key
            // says, at once: that is what keeps contacts on one solver. One
            // that is only ours is going home after an interaction, and is
            // held until that has been its wish for a while (see `border`,
            // "Hysteresis").
            let wish = key2.1;
            // An island with a ship in it follows the ship (a ship has no
            // hysteresis: a player is on the shard the map names).
            let with_ship = island.bodies.iter().any(|guid| {
                self.sim
                    .entity_of(*guid)
                    .is_some_and(|e| self.sim.world().get::<ShipControls>(e).is_some())
            });
            let target = if !ghosts.is_empty()
                || wish == self.config.shard
                || with_ship
                || self.ready_to_go(&island.bodies, wish)
            {
                wish
            } else {
                self.config.shard
            };
            self.note_island(&island.bodies, island.bodies.len() + ghosts.len() > 1, wish);
            let going_home = ghosts.is_empty() && !with_ship && target != self.config.shard;
            let index = view.islands.len();
            for guid in &island.bodies {
                view.of.insert(*guid, index);
            }
            view.islands.push(IslandInfo {
                members: island.bodies,
                ghosts,
                announce: key1,
                target,
                going_home,
            });
        }
        // Kinematic bodies (the island query only groups dynamic ones) are
        // islands of one.
        for (guid, _, ghost) in self.sim.bodies() {
            if ghost || view.of.contains_key(&guid) {
                continue;
            }
            let home = self.home_local(guid);
            let target = if home == self.config.shard || self.ready_to_go(&[guid], home) {
                home
            } else {
                self.config.shard
            };
            self.note_island(&[guid], false, home);
            view.of.insert(guid, view.islands.len());
            view.islands.push(IslandInfo {
                members: vec![guid],
                ghosts: Vec::new(),
                announce: (guid, home),
                target,
                going_home: target != self.config.shard,
            });
        }
        self.island_view = view;
    }

    /// The islands to hand off this tick, with where to: every island whose
    /// home is another known shard and none of whose members is already in a
    /// transfer or paused for that shard.
    pub(super) fn islands_to_hand_off(&self) -> Vec<(Vec<EntityGuid>, ShardId, bool)> {
        let mut out = Vec::new();
        for island in &self.island_view.islands {
            let to = island.target;
            if to == self.config.shard || to == ShardId::NONE || !self.knows_shard(to) {
                continue;
            }
            let free = island.members.iter().all(|guid| {
                !self.ledger.is_busy(*guid)
                    && !self.pending_commits.iter().any(|p| p.guid == *guid)
                    && !self
                        .refused
                        .get(guid)
                        .is_some_and(|(refused_by, _)| *refused_by == to)
                    && self.sim.ship_state(*guid).is_some()
            });
            if free {
                out.push((island.members.clone(), to, island.going_home));
            }
        }
        out
    }

    // ------------------------------------------------------ the body handoff

    /// Offer the body `guid` (no client drives it) to `to`: its whole
    /// component list, no redirect, no client gate.
    pub(super) fn offer_body(
        &mut self,
        guid: EntityGuid,
        to: ShardId,
        tick: SimTick,
        lead: SimTick,
        group: u64,
    ) {
        let Some(list) = self.sim.export_entity(guid) else {
            return;
        };
        let mut state = EntitySnapshot::new(guid, list);
        state.components.push(TransferKind::Body.component(lead));
        let offer =
            Ledger::offer_with_lead(guid, self.config.shard, to, self.epoch(), tick, state, lead);
        self.ledger
            .offer_sent_as(&offer, tick, TransferKind::Body, group);
        self.send_to(Destination::Shard(to), Message::HandoffOffer(offer));
    }

    /// The state a commit of `guid` carries (as of now: the activation tick).
    pub(super) fn commit_state(
        &self,
        guid: EntityGuid,
        kind: TransferKind,
    ) -> Option<EntitySnapshot> {
        match kind {
            TransferKind::Ship => {
                let entity = self.sim.entity_of(guid)?;
                replicate::snapshot_of(&self.sim, &self.guids, entity)
            }
            TransferKind::Body => {
                let mut state = EntitySnapshot::new(guid, self.sim.export_entity(guid)?);
                state.components.push(TransferKind::Body.component(0));
                Some(state)
            }
        }
    }

    /// Whether a body offer can be taken: it imports here as it is, and it
    /// is not a body of another scene revision.
    pub(super) fn check_body_offer(&self, offer: &HandoffOffer) -> Result<(), String> {
        if border::is_stale_scene_body(offer.entity, self.scene.revision) {
            return Err("a body of another scene revision".into());
        }
        self.sim
            .check_import(&strip_private(&offer.state.components))
            .map_err(|e| format!("the offered body does not import: {e}"))
    }

    /// Two shards offering each other members of one island at the same
    /// time: the transfer towards the lower shard id goes ahead, the other
    /// is refused (both sides apply the same rule).
    pub(super) fn is_crossing_offer(&self, offer: &HandoffOffer) -> bool {
        self.config.shard > offer.from
            && self.ledger.outgoing().iter().any(|(guid, entry)| {
                entry.to == offer.from && self.island_view.touches(*guid, offer.entity)
            })
    }

    /// The component list a body's commit installs: the commit's, or the
    /// offer's if the commit's does not import (never a default).
    pub(super) fn body_components_of(
        &self,
        commit: &EntitySnapshot,
        staged: &EntitySnapshot,
    ) -> Option<Components> {
        [commit, staged].into_iter().find_map(|state| {
            let list = strip_private(&state.components);
            self.sim.check_import(&list).is_ok().then_some(list)
        })
    }

    /// Install a committed body: it replaces our ghost of it, and the shard
    /// that sent it holds a ghost of it from now on.
    pub(super) fn install_body(&mut self, pending: PendingCommit) {
        let guid = pending.guid;
        if self.sim.ship_state(guid).is_some() {
            eprintln!("space-server: dropped a second install of {guid}: it is already alive here");
            return;
        }
        if border::is_stale_scene_body(guid, self.scene.revision) {
            return;
        }
        self.drop_ghost(guid, false);
        match self.sim.import_entity(guid, &pending.components) {
            Ok(_) => {
                self.index_now(guid);
                self.stats.received += 1;
                self.note_sender_holds_ghost(pending.from, guid);
                self.back_again(guid);
            }
            Err(e) => eprintln!("space-server: the committed body {guid} does not import: {e}"),
        }
    }

    /// `from` handed us `guid` and keeps a ghost of it: from now on we tell
    /// it what changes (or that the ghost is to go).
    pub(super) fn note_sender_holds_ghost(&mut self, from: ShardId, guid: EntityGuid) {
        if let Some(components) = replicate::replicated_components(&self.sim, guid) {
            self.ghost_sent.entry(from).or_default().insert(
                guid,
                SentGhost {
                    components,
                    island: None,
                },
            );
        }
    }

    /// `guid` was committed to `to`: it stops being ours and stays visible
    /// here as a ghost of `to`'s copy.
    pub(super) fn depart(&mut self, guid: EntityGuid, to: ShardId) {
        let components = replicated_of(&self.sim, &self.ghosts, guid);
        if let Some(sent) = self.ghost_sent.get_mut(&to) {
            sent.remove(&guid);
        }
        self.refused.remove(&guid);
        self.sim.despawn(guid);
        let ghost = components.and_then(|list| {
            self.sim.insert_ghost(guid, &list).ok()?;
            Some(list)
        });
        match ghost {
            Some(list) => {
                self.index_now(guid);
                let ship = component(&list, CONTROLS).is_some();
                self.ghosts.insert(
                    guid,
                    GhostEntry {
                        from: to,
                        ship,
                        components: zero_owner(list),
                        reckoned: 0,
                        island: None,
                        departed_at: Some(self.sim.tick()),
                    },
                );
                self.ghost_heard.insert(to, self.sim.tick());
            }
            None => {
                self.guids.remove(guid);
                self.interest.remove(guid);
                self.log.forget(guid);
            }
        }
    }

    // ---------------------------------------------------------- ghosts in

    /// Drop the ghost `guid` (and, with `forget`, what clients were told).
    fn drop_ghost(&mut self, guid: EntityGuid, forget: bool) {
        let held = self.ghosts.remove(&guid).is_some();
        if self.sim.is_ghost(guid) {
            self.sim.despawn(guid);
        } else if !held {
            return;
        }
        self.guids.remove(guid);
        self.interest.remove(guid);
        if forget {
            self.log.forget(guid);
        }
    }

    /// Remove the ghost `guid`.
    pub(super) fn remove_ghost(&mut self, guid: EntityGuid) {
        self.drop_ghost(guid, true);
    }

    /// Forget ghosts and ghost bookkeeping of shards the map no longer names.
    pub(super) fn forget_departed_neighbours(&mut self) {
        let known: HashSet<ShardId> = self.shard_map.regions.iter().map(|r| r.shard).collect();
        self.ghost_sent.retain(|shard, _| known.contains(shard));
        let gone: Vec<EntityGuid> = self
            .ghosts
            .iter()
            .filter(|(_, e)| !known.contains(&e.from))
            .map(|(g, _)| *g)
            .collect();
        for guid in gone {
            self.remove_ghost(guid);
        }
    }

    /// A neighbour's ghosts arrived.
    pub(super) fn on_ghost_update(&mut self, from: ShardId, update: GhostUpdate) {
        if update.epoch != self.epoch() || update.from != from {
            return; // another map: wait for the next one
        }
        self.ghost_heard.insert(from, self.sim.tick());
        let revision = self.scene.revision;
        let mut listed: HashSet<EntityGuid> = HashSet::new();
        for entity in update.entities {
            let guid = entity.guid;
            listed.insert(guid);
            if border::is_stale_scene_body(guid, revision) || self.sim.ship_state(guid).is_some() {
                continue; // another scene's, or ours (we simulate it)
            }
            let held = self.ghosts.get(&guid).filter(|_| self.sim.is_ghost(guid));
            let merged = match held {
                Some(entry) if !update.full => merge(&entry.components, &entity.components),
                _ => strip_private(&entity.components),
            };
            let merged = zero_owner(merged);
            if component(&merged, SHIP_STATE).is_none() {
                continue; // a partial update of a ghost we do not hold
            }
            let rebuild = held.is_none_or(|entry| structure_differs(&entry.components, &merged));
            let island = self.ghosts.get(&guid).and_then(|e| e.island);
            if rebuild {
                if let Err(e) = self.sim.insert_ghost(guid, &merged) {
                    // Said once a second (with the full refresh), not every
                    // tick: a body whose shape this shard cannot build stays
                    // invisible here until the scenes agree again.
                    if update.full {
                        eprintln!(
                            "space-server: the ghost of {guid} from {from} does not import: {e}"
                        );
                    }
                    continue;
                }
                self.index_now(guid);
            } else {
                let state = component(&merged, SHIP_STATE)
                    .and_then(|b| space_sim::wire::decode_ship_state(b).ok());
                let parts = component(&merged, PART_STATES)
                    .and_then(|b| space_sim::articulation::decode_part_states(b).ok());
                if let Some(state) = state {
                    self.sim
                        .drive_ghost(guid, state, parts.as_ref().map(|p| p.0.as_slice()));
                }
            }
            let ship = component(&merged, CONTROLS).is_some();
            self.ghosts.insert(
                guid,
                GhostEntry {
                    from,
                    ship,
                    components: merged,
                    reckoned: 0,
                    island,
                    departed_at: None,
                },
            );
        }
        for guid in update.removed {
            if self.ghosts.get(&guid).is_some_and(|e| e.from == from) {
                self.remove_ghost(guid);
            }
        }
        if update.full {
            // A refresh lists what its sender held when it was sent. A ghost
            // of an entity on its way here (staged, or committed and not yet
            // installed), or one we just handed to the sender, may not be in
            // it yet: those go only when told explicitly.
            let now = self.sim.tick();
            let in_flight = |shard: &Self, guid: EntityGuid, entry: &GhostEntry| {
                shard.ledger.has_incoming(guid)
                    || shard.pending_commits.iter().any(|p| p.guid == guid)
                    || entry
                        .departed_at
                        .is_some_and(|at| now.saturating_sub(at) <= 2 * border::GHOST_REFRESH_TICKS)
            };
            let stale: Vec<EntityGuid> = self
                .ghosts
                .iter()
                .filter(|(g, e)| e.from == from && !listed.contains(g) && !in_flight(self, **g, e))
                .map(|(g, _)| *g)
                .collect();
            for guid in stale {
                self.remove_ghost(guid);
            }
        }
        for key in update.islands {
            if let Some(entry) = self.ghosts.get_mut(&key.body) {
                if entry.from == from {
                    entry.island = Some((key.min, key.home));
                }
            }
        }
    }

    /// After the step: every ghost moves on one tick along its velocity, for
    /// up to [`border::GHOST_DEAD_RECKON_TICKS`] ticks without an update.
    ///
    /// A ghost's update is its owner's state at the end of the owner's last
    /// tick, which is the start of the tick this shard steps next: the step
    /// collides against it there (it is not integrated). Moving it on
    /// afterwards makes what this shard's clients see, and what the next step
    /// starts from if no update arrives in between, the owner's state *now* —
    /// so a body crossing a border is not a tick behind on the side it enters,
    /// and does not jump by that tick when it arrives.
    pub(super) fn reckon_ghosts(&mut self) {
        let dt = f64::from(space_sim::dt());
        let step = |s: &ShipState| ShipState {
            pos: WorldPos::from_f64_metres_around(s.pos, s.vel.map(|v| f64::from(v) * dt)),
            ..*s
        };
        let guids: Vec<EntityGuid> = self.ghosts.keys().copied().collect();
        for guid in guids {
            let Some(entry) = self.ghosts.get_mut(&guid) else {
                continue;
            };
            if entry.reckoned >= border::GHOST_DEAD_RECKON_TICKS {
                continue;
            }
            entry.reckoned += 1;
            let Some(parts) = self.sim.part_states_of(guid) else {
                continue;
            };
            if parts.iter().all(|s| s.vel == [0.0; 3]) {
                continue;
            }
            let moved: Vec<ShipState> = parts.iter().map(step).collect();
            let rest = (moved.len() > 1).then(|| &moved[1..]);
            self.sim.drive_ghost(guid, moved[0], rest);
        }
    }

    // --------------------------------------------------------- ghosts out

    /// The entities the springs of `guid` pull on.
    fn spring_targets(&self, guid: EntityGuid) -> Vec<EntityGuid> {
        self.sim
            .entity_of(guid)
            .and_then(|e| self.sim.world().get::<Springs>(e))
            .map(|springs| {
                springs
                    .0
                    .iter()
                    .filter_map(|s| match s.target {
                        SpringTarget::Entity { guid, .. } => Some(guid),
                        SpringTarget::World(_) => None,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Tell every neighbour about our bodies near its territory and the ends
    /// of springs that cross to it (see [`crate::border`], "Ghosts"); drop
    /// the ghosts of neighbours that have gone silent.
    pub(super) fn ghost_pass(&mut self, tick: SimTick) {
        let silent: Vec<EntityGuid> = self
            .ghosts
            .iter()
            .filter(|(_, e)| {
                self.ghost_heard
                    .get(&e.from)
                    .is_none_or(|heard| tick.saturating_sub(*heard) > border::GHOST_SILENCE_TICKS)
            })
            .map(|(g, _)| *g)
            .collect();
        for guid in silent {
            self.remove_ghost(guid);
        }
        let neighbours = border::neighbours(&self.shard_map, self.config.shard);
        if neighbours.is_empty() {
            return;
        }
        let full = tick.is_multiple_of(border::GHOST_REFRESH_TICKS);
        let locals: Vec<(EntityGuid, ShipState)> = self
            .sim
            .bodies()
            .into_iter()
            .filter(|(_, _, ghost)| !ghost)
            .filter_map(|(guid, _, _)| Some((guid, self.sim.ship_state(guid)?)))
            .collect();
        // The two ends of a spring are ghosted to each other's shard.
        let mut linked: HashMap<ShardId, BTreeSet<EntityGuid>> = HashMap::new();
        for (guid, _) in &locals {
            for target in self.spring_targets(*guid) {
                if self.sim.ship_state(target).is_some() {
                    continue; // both ends are here
                }
                match self.ghosts.get(&target) {
                    Some(entry) => {
                        linked.entry(entry.from).or_default().insert(*guid);
                    }
                    // Not seen anywhere yet: every neighbour, until its far
                    // end shows up as somebody's ghost.
                    None => {
                        for n in &neighbours {
                            linked.entry(*n).or_default().insert(*guid);
                        }
                    }
                }
            }
        }
        for (ghost, entry) in &self.ghosts {
            for target in self.spring_targets(*ghost) {
                if self.sim.ship_state(target).is_some() {
                    linked.entry(entry.from).or_default().insert(target);
                }
            }
        }
        for n in neighbours {
            let mut want: BTreeSet<EntityGuid> = linked.remove(&n).unwrap_or_default();
            for (guid, state) in &locals {
                let v = state.vel;
                let speed = f64::from((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
                if border::territory_distance(&self.shard_map, n, state.pos)
                    <= border::ghost_reach(speed)
                {
                    want.insert(*guid);
                }
            }
            let mut sent = self.ghost_sent.remove(&n).unwrap_or_default();
            let mut entities = Vec::new();
            let mut islands = Vec::new();
            for guid in &want {
                let Some(components) = replicate::replicated_components(&self.sim, *guid) else {
                    continue;
                };
                let island = self.island_view.announce_of(*guid);
                let key = |island: Option<(EntityGuid, ShardId)>| {
                    island.map(|(min, home)| IslandKey {
                        body: *guid,
                        min,
                        home,
                    })
                };
                match sent.get(guid) {
                    Some(prev) if !full => {
                        let mut changed = diff(&prev.components, &components);
                        if !changed.is_empty() {
                            if component(&changed, SHIP_STATE).is_none() {
                                if let Some(state) = component(&components, SHIP_STATE) {
                                    changed.push((SHIP_STATE, state.to_vec()));
                                    changed.sort_by_key(|(id, _)| *id);
                                }
                            }
                            entities.push(EntitySnapshot::new(*guid, changed));
                        }
                        if prev.island != island {
                            islands.extend(key(island));
                        }
                    }
                    _ => {
                        entities.push(EntitySnapshot::new(*guid, components.clone()));
                        islands.extend(key(island));
                    }
                }
                sent.insert(*guid, SentGhost { components, island });
            }
            let mut removed: Vec<EntityGuid> =
                sent.keys().filter(|g| !want.contains(g)).copied().collect();
            removed.sort_unstable();
            for guid in &removed {
                sent.remove(guid);
            }
            self.ghost_sent.insert(n, sent);
            if full || !entities.is_empty() || !removed.is_empty() || !islands.is_empty() {
                let update = GhostUpdate {
                    from: self.config.shard,
                    epoch: self.epoch(),
                    tick,
                    full,
                    entities,
                    removed,
                    islands,
                };
                self.send_to(Destination::Shard(n), Message::GhostUpdate(update));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_spawn_grid_starts_at_the_spawn_point_and_is_nearest_first() {
        let grid = spawn_grid();
        assert_eq!(grid[0], (0, 0));
        assert_eq!(grid.len(), (2 * SPAWN_RINGS as usize + 1).pow(2));
        let r = |p: &(i32, i32)| p.0 * p.0 + p.1 * p.1;
        assert!(grid.windows(2).all(|w| r(&w[0]) <= r(&w[1])));
    }

    #[test]
    fn partial_component_lists_merge_and_diff_back() {
        let old: Components = vec![(0, vec![1]), (4, vec![2]), (5, vec![3])];
        let new: Components = vec![(0, vec![9]), (4, vec![2]), (6, vec![7])];
        let d = diff(&old, &new);
        assert_eq!(d, vec![(0, vec![9]), (5, Vec::new()), (6, vec![7])]);
        assert_eq!(merge(&old, &d), new);
        assert!(structure_differs(&old, &new));
        assert!(!structure_differs(
            &old,
            &[(0, vec![5]), (4, vec![2]), (5, vec![3])]
        ));
    }
}
