//! Physics across shard borders: the rules, as pure functions of the map.
//!
//! A shard simulates the bodies of its region exactly ([`space_sim`]'s rigid
//! bodies, springs, joints and gravity). What happens at a border is decided
//! here, and every rule is a deterministic function of what *both* shards on
//! either side can see — the shard map, the positions and the guids — so the
//! two sides agree without asking each other.
//!
//! # Ghosts
//!
//! Every body (ships included) a shard simulates within [`GHOST_MARGIN`] plus
//! [`GHOST_LEAD`] seconds of its own motion of a neighbour's territory is sent
//! to that neighbour every tick it changes ([`space_protocol::GhostUpdate`]),
//! and in full every [`GHOST_REFRESH_TICKS`]. The neighbour keeps it as a
//! **ghost** ([`space_sim::SimWorld::insert_ghost`]): a kinematic copy its own
//! bodies collide with, its springs pull on, and its clients see. An update
//! is the owner's state at the end of its tick — the start of the
//! receiver's next step, which collides against it there; after the step the
//! ghost is moved on one tick along its velocity, so it shows the owner's
//! state *now*, and it keeps being dead-reckoned that way for up to
//! [`GHOST_DEAD_RECKON_TICKS`] without an update. A shard that sends nothing
//! for [`GHOST_SILENCE_TICKS`] loses its ghosts. So a body near a border is seen
//! from both sides, and when it crosses, the side it enters already shows it.
//!
//! The two ends of a spring are ghosted to each other's shard whatever the
//! distance, so each side applies the spring's force to its own end from the
//! other end's ghost (a delay of one or two ticks; see the spring rule below).
//!
//! # Islands and consolidation
//!
//! Two bodies simulated by two shards would collide only approximately (each
//! sees the other as an immovable ghost). So bodies that may touch soon —
//! an **island**: [`space_sim::SimWorld::islands`] with [`ISLAND_MARGIN`] and
//! [`ISLAND_LEAD`] — are consolidated onto **one** shard before they meet.
//!
//! "May touch soon" is a statement about two bodies' **motion**: their
//! bounding spheres are within twice the margin of each other now, or come
//! that close within the lead time if each moves on in a straight line as it
//! moves now (the closest approach of their relative motion). A body that
//! flies *past* another, however fast, is not in its island; one that flies
//! *at* it is, a second before they meet. (An island used to be every body
//! within `|v| · lead` of a body in **any** direction. A ship crossing a
//! hundred metres above the demo area at 120 m/s was then one island with
//! everything below it, behind it and ahead of it, and since bodies follow a
//! player — below — the whole showcase changed shard with every pass and
//! came back afterwards: 73 body handoffs a pass in
//! `tests/four_cells.rs`' fly-by, where there are none now; on a live stack
//! 659 handoffs and 197 transfers called off in half an hour of play.)
//!
//! What it costs: the test is on bounding spheres, so a tall thin thing — a
//! ship is a capsule 1.8 m tall and 0.6 m wide — reaches about a metre and a
//! half sideways from its hull at rest. A player who walks past a stack that
//! close has it brought to the player's shard and sent back three seconds
//! after the player has gone.
//!
//! * Every island is given a **key**: its member with the lowest guid and that
//!   member's [`home`]. A shard computes it over its own bodies in the island
//!   and the ghosts touching them (the one-hop key, which it announces with
//!   each ghost in [`space_protocol::IslandKey`]), then folds in the keys the
//!   owners of those ghosts announced (two hops). Both shards on either side
//!   of a border therefore arrive at the same key within a tick.
//! * The island belongs on the key's home. A shard holding bodies of an
//!   island whose home is another shard hands them to it, all in one group
//!   (the handoff commits a group together or not at all). So contacts and
//!   joints across a border are always solved by one solver. (A ship a
//!   player flies is the exception on both counts: see "Players' ships".)
//! * While a body is in such an island it may be simulated by a shard whose
//!   region does not contain it — the one exception to "the shard a body is
//!   on is the one the map names". When the island dissolves, each body's own
//!   home applies again and it returns to its region's owner.
//! * **Two shards never offer each other members of one island.** What each
//!   side knows of the other's key is a tick old, so on the tick an island
//!   forms, or its key's home changes, the two can disagree for that tick.
//!   Two rules keep that from becoming two offers that cross: an island that
//!   touches another shard's bodies is offered only once its home has been
//!   the same for [`CONFIRM_TICKS`] passes (the other side's key of the first
//!   of them is in by then); and an island one of whose bodies is **on its
//!   way here** (staged, or committed and not yet installed) waits for it and
//!   then goes on with it, rather than being offered against it. (The second
//!   was seen with late commits: a crate sliding across a border towards
//!   another that had just been sent back to its side — its home changed
//!   while that transfer was under way, it was offered the other way, and
//!   refused.) Should two offers cross all the same, the accepting shard
//!   with the higher id refuses its side ("crossing island transfer") and
//!   the refused bodies are offered again after the pause; that costs
//!   nothing a client sees, and no test here makes it happen any more.
//!
//! # Hysteresis
//!
//! Three rules keep bodies from changing shard back and forth, and none of
//! them can keep a settled body away from its region for good:
//!
//! * **Distance.** A body is taken over by the region it is in once it is
//!   [`BODY_HYSTERESIS`] inside it ([`home`]): a pendulum swinging over a
//!   border, a crate pushed a little across one and back, do not change shard
//!   at every swing. A ship has [`SHIP_HYSTERESIS`], half of a body's (see
//!   "Players' ships" below).
//! * **Settling.** A body that has been inside another shard's territory
//!   for [`SETTLE_TICKS`] in a row is taken over too, however close to the
//!   border it is: on distance alone a body resting within the hysteresis of
//!   a border would never move. (Found when a scene's spawn point still lay
//!   on the corner where cells meet: a carved octant then had a border plane
//!   at spawn height, and the demo area's zero-g corner floats a few
//!   decimetres above that. The anchor now lies half a cell above the corner
//!   and a few metres beside it — `space_scene::ANCHOR_OFFSET` — so the only
//!   border planes through a scene are the vertical ones at scene `x = −6`
//!   and `z = −4`; the demo area has a crate floating exactly on the
//!   latter.)
//! * **Holding.** Moving an island that touches another shard's ghosts is a
//!   necessity (its contacts must be solved by one solver) and happens at
//!   once. Moving one that touches none is a body going home after an
//!   interaction: if any of its members was in a group within the last
//!   [`HOLD_TICKS`], it waits until the island has wanted to go to the same
//!   shard for that long, continuously. (Not an island with a ship in it: it
//!   goes where its key says at once.) A swinging pendulum whose reach
//!   flickers in and out of a group of crates therefore does not send the
//!   crates to and fro at every swing (seen in a live run of the demo area
//!   split by a border) — and a body that simply rests near others of its
//!   shard goes home once the wish has held for [`HOLD_TICKS`]. (An earlier
//!   rule waited until no other body of the shard was loosely near; a body
//!   resting a few metres from one that never moves then never went home.)
//!   The hold backs off: a body a shard sent home and gets back within
//!   [`RETURN_WINDOW_TICKS`] — something of that shard keeps reaching it,
//!   slower than the hold (a spring pair drifting in and out of a parked
//!   player's reach, seen live) — waits twice as long the next time, up to
//!   [`MAX_HOLD_DOUBLINGS`] doublings.
//!
//! # Players' ships
//!
//! A ship a player flies is handed over **for where it is, and for nothing
//! else**:
//!
//! * **It is never moved by an island.** An island's home is where its
//!   *bodies* go; a player's ship in it stays on the shard that simulates it
//!   until its own position says otherwise. (It used to travel with its
//!   island: a player standing still was handed to the neighbour shard and
//!   back because somebody else flew past within a second's reach — an
//!   island of two ships whose key was the other one.)
//! * **Bodies follow the player, not the other way round.** In an island's
//!   key a player's ship counts with the shard that **simulates** it as its
//!   home (both sides know that shard: the one it is local to, and the one
//!   that holds its ghost). So the bodies a player touches are consolidated
//!   on the player's shard, as before. When the player is handed over and
//!   its ship is its island's key, the island's bodies **travel with it**:
//!   offered in one group with the ship's lead, committed on the tick the
//!   ship is committed, installed with it — a crate being pushed across a
//!   border is on the ship's shard at every tick.
//! * **Nothing follows a player back.** While a player's ship is on its way
//!   to a shard — its transfer is staged there, or it is far enough into
//!   that shard's ground ([`SHIP_HYSTERESIS`]) for its own shard to be
//!   offering it now — the bodies of that shard which come within its reach
//!   **wait where they are**: the ship will be theirs within the second.
//!   (They used to be offered to the shard the ship was leaving, against the
//!   ship's own transfer: refused as a crossing island transfer when the
//!   ship's shard had the higher id, sent there and back when it had the
//!   lower. Bodies the ship reached *before* that are already with it and
//!   travel with it, above.) For that second, at most, the ship meets such a
//!   body as body against ghost.
//! * **It waits for nothing but its own gate.** The ship commits when the
//!   accepting shard has accepted it and its player has connected there. A
//!   body of its group that is refused, or not accepted by then, stays
//!   behind — with the group's other bodies — and follows a few ticks after
//!   the ship has arrived, as an island that touches the ghost the ship
//!   left; in between the ship and those bodies collide as body against
//!   ghost. And the rule that settles two shards offering each other members
//!   of one island (the lower shard id takes it) is never applied to a
//!   player's ship. (Both used to call the player's handoff off: crossing
//!   into the demo area's zero-g corner the accepting shard was sending a
//!   crate the other way at that moment, refused the whole group as a
//!   "crossing island transfer", and the player waited out the two seconds
//!   of the re-offer pause — three seconds for a crossing instead of one.)
//! * **Hysteresis.** The ship is offered to the region it has entered once
//!   it is [`SHIP_HYSTERESIS`] inside it — measured to the nearest ground of
//!   the shard that simulates it ([`near_territory_distance`], exact around a
//!   corner where three or four shards meet) — or once it has been inside it
//!   for [`SETTLE_TICKS`], like a body. A player standing on a border, or
//!   walking along one and straying less than that to either side, is not
//!   handed back and forth; one who stops just across it is taken over three
//!   seconds later, once.
//!
//! A ship nobody flies is a body like any other here, with the ship's
//! hysteresis.
//!
//! Exactness has a speed limit: an island is only noticed once both sides see
//! each other's bodies as ghosts, i.e. within [`GHOST_MARGIN`] (plus the
//! velocity lead), and the handoff then needs its lead time (a second for a
//! player, whose client must connect first; [`BODY_HANDOFF_LEAD`] for an
//! unowned body). Bodies closing faster than about 50 m/s across a border can
//! meet before they are consolidated; they then collide as body against ghost
//! until the handoff lands.
//!
//! # Springs across a border
//!
//! A spring does **not** make its two ends an island on its own (the
//! `springs: false` island query): only contact proximity does. A spring or
//! rope can be long, and pulling its ends onto one shard would drag bodies
//! arbitrarily far from their regions. Its force is explicit — evaluated from
//! the two end states — so each shard applies it to the end it simulates,
//! reading the other end's ghost; the only difference to one shard is the one
//! or two ticks the ghost lags. Ends close enough to touch are an island like
//! any other bodies and are solved together.
//!
//! # Scene bodies
//!
//! A scene's bodies are spawned **once per cluster**, by the scene's
//! **spawner** ([`scene_spawner`]): the shard of the map's shallowest region,
//! the root the scene's anchor is placed in (`space_scene::scene_anchor`).
//! Carves only add deeper regions and merges only remove them, so the spawner
//! of a running cluster never changes — a shard born from a split or a carve,
//! told the scene when it registers, spawns nothing (it would duplicate what
//! the root's shard already simulates and is about to hand it). The spawner
//! need not own the spawn point, or any of the scene: every body it holds
//! outside its own territory — all of them, when the cells the scene lies in
//! were carved out for other shards — goes to the shard that owns it by the
//! ordinary handoff, a few ticks after the spawn: **at once and all on one
//! tick**, also a body that lies within the distance hysteresis of the
//! spawner's own ground (the hysteresis is for a body that has crossed a
//! border; one that was put down on a neighbour's ground has not, and held
//! back it would meet the bodies that went ahead as body against ghost — the
//! demo area's third bowl ball, 1.9 m from the border, did, and the three
//! ended 18 cm from where one shard has them). On a scene change every shard despawns the
//! bodies of the old scene it simulates (their ghosts go with the next ghost
//! frame) and the spawner spawns the new scene's.
//!
//! The guids are a function of the scene revision and the body's index,
//! under the spawner's id ([`scene_guid`]), so every shard can name every
//! body — a spring from one shard's body to another's resolves — and none can
//! collide with a guid a shard's own counter mints. (A spawner that restarts
//! spawns the scene again; bodies another shard still holds then exist twice
//! until the next scene change. The world is not persisted.)

use space_protocol::{EntityGuid, ShardId, ShardMap};
use space_sim::SimTick;
use space_world::{octree, NodeRef, WorldPos};

/// How close (metres) a body must come to a neighbour's territory to be
/// ghosted to it, before the velocity lead.
pub const GHOST_MARGIN: f64 = 64.0;
/// Velocity lead of the ghost margin, seconds: a body's margin grows by
/// `|v| · GHOST_LEAD`.
pub const GHOST_LEAD: f64 = 3.0;
/// Contact-proximity margin of an island, metres (two bodies whose bounding
/// spheres are within twice this are one island; see
/// [`space_sim::IslandQuery`]). It has to cover what the velocity lead cannot
/// see: what a body gains from rest in the time a body handoff takes — a
/// third of a second: 0.3 m under a ship's thrust, 0.5 m in free fall.
/// Small beyond that: the velocity lead is what gives a moving body its
/// reach; at rest two bodies a metre apart do not
/// touch, and a larger margin chained whole resting groups across a border
/// into one island (the demo area's zero-g corner, back when a horizontal
/// border ran through a scene at spawn height: a hinged door below an
/// octant's floor and a spring pair just above it).
pub const ISLAND_MARGIN: f64 = 0.5;
/// Velocity lead of an island, seconds: two bodies that come within reach of
/// each other in this time, moving on as they move now, are one island. Long
/// enough for a player's handoff (one second) to land before two bodies
/// closing at moderate speed meet.
pub const ISLAND_LEAD: f64 = 1.0;
/// How far inside another shard's territory an unowned body must be before
/// that shard takes it over, metres.
pub const BODY_HYSTERESIS: f64 = 2.0;
/// How far inside another shard's territory a ship must be before that shard
/// takes it over, metres (see the module docs, "Players' ships"). A dead
/// band of two metres around a border: wide enough for a player who stands
/// on it or walks along it, crossed in a fifteenth of a second at flying
/// speed.
pub const SHIP_HYSTERESIS: f64 = 1.0;
/// How many border passes in a row an island that touches another shard's
/// bodies must have belonged on the same other shard before its bodies are
/// offered there (see the module docs, "Islands and consolidation"): one
/// for the other side's key of this tick to arrive, one to act on it.
pub const CONFIRM_TICKS: SimTick = 2;
/// A body inside another shard's territory this long in a row is settled
/// there and taken over, however close to the border (see "Hysteresis").
pub const SETTLE_TICKS: SimTick = 3 * space_sim::TICK_HZ as SimTick;
/// How long an island of one shard that was in a group recently must want
/// to go to another shard before it goes (see "Hysteresis").
pub const HOLD_TICKS: SimTick = 3 * space_sim::TICK_HZ as SimTick;
/// A body back on the shard that sent it home within this many ticks has
/// its hold doubled (see "Hysteresis").
pub const RETURN_WINDOW_TICKS: SimTick = 30 * space_sim::TICK_HZ as SimTick;
/// The most a hold doubles: 3 s · 2⁴ = 48 s.
pub const MAX_HOLD_DOUBLINGS: u32 = 4;
/// Activation lead of an unowned body's handoff, ticks. No client has to
/// connect, so the lead only has to cover the offer/accept round trip.
pub const BODY_HANDOFF_LEAD: SimTick = 8;
/// A full ghost refresh to every neighbour this often, ticks.
pub const GHOST_REFRESH_TICKS: SimTick = space_sim::TICK_HZ as SimTick;
/// A ghost with no update is dead-reckoned for at most this many ticks.
pub const GHOST_DEAD_RECKON_TICKS: u32 = 10;
/// A neighbour silent for this many ticks loses its ghosts here.
pub const GHOST_SILENCE_TICKS: SimTick = 3 * space_sim::TICK_HZ as SimTick;

/// The serial bit marking a scene body's guid (a shard's own counter starts
/// at 1 and never gets near it).
pub const SCENE_SERIAL_BIT: u64 = 1 << 63;
/// Bits of a scene guid's serial holding the body's index in its scene.
const SCENE_INDEX_BITS: u32 = 20;
/// Mask of the revision bits.
const SCENE_REVISION_MASK: u64 = (1 << (63 - SCENE_INDEX_BITS)) - 1;

/// The shard that spawns a scene's bodies: the owner of the map's
/// shallowest region (ties: the lowest shard id, then the lowest node index —
/// the rule `space_scene::scene_anchor` picks the anchor by).
#[must_use]
pub fn scene_spawner(map: &ShardMap) -> Option<ShardId> {
    map.regions
        .iter()
        .min_by_key(|region| (region.node.depth, region.shard.0, region.node.index))
        .map(|region| region.shard)
}

/// The guid of body `index` of the scene at `revision`, spawned by `owner`
/// (the scene's spawner).
#[must_use]
pub fn scene_guid(owner: ShardId, revision: u64, index: usize) -> EntityGuid {
    let index = (index as u64) & ((1 << SCENE_INDEX_BITS) - 1);
    EntityGuid::new(
        owner,
        SCENE_SERIAL_BIT | ((revision & SCENE_REVISION_MASK) << SCENE_INDEX_BITS) | index,
    )
}

/// The scene revision a scene body's guid was minted for (`None` for any
/// other guid). Compared as [`scene_guid`] stores it, i.e. modulo 2^43.
#[must_use]
pub fn scene_revision_of(guid: EntityGuid) -> Option<u64> {
    (guid.serial & SCENE_SERIAL_BIT != 0)
        .then_some((guid.serial >> SCENE_INDEX_BITS) & SCENE_REVISION_MASK)
}

/// Whether `guid` is a scene body of another revision than `revision`.
#[must_use]
pub fn is_stale_scene_body(guid: EntityGuid, revision: u64) -> bool {
    scene_revision_of(guid).is_some_and(|r| r != revision & SCENE_REVISION_MASK)
}

/// Distance from `pos` to the box of `node`, metres (0 inside).
fn box_distance(node: NodeRef, pos: WorldPos) -> f64 {
    let (lo, hi) = octree::node_bounds(node);
    let below = WorldPos::delta(pos, lo); // lo - pos: > 0 when pos is below
    let above = WorldPos::delta(hi, pos); // pos - hi: > 0 when pos is above
    (0..3)
        .map(|k| below[k].max(above[k]).max(0.0))
        .map(|d| d * d)
        .sum::<f64>()
        .sqrt()
}

/// Distance from `pos`, inside `node`, to the outside of its box.
fn depth_inside(node: NodeRef, pos: WorldPos) -> f64 {
    let (lo, hi) = octree::node_bounds(node);
    let from_lo = WorldPos::delta(lo, pos);
    let to_hi = WorldPos::delta(pos, hi);
    (0..3)
        .map(|k| from_lo[k].min(to_hi[k]))
        .fold(f64::INFINITY, f64::min)
        .max(0.0)
}

/// How far `pos` is from the part of the world `map` gives `shard`, metres:
/// 0 where `shard` owns `pos`. Never more than the true distance (it may be
/// less around carved-out regions), so a margin test built on it errs on the
/// side of sending a ghost.
#[must_use]
pub fn territory_distance(map: &ShardMap, shard: ShardId, pos: WorldPos) -> f64 {
    if map.lookup(&pos) == Some(shard) {
        return 0.0;
    }
    // The deepest region covering `pos`: what carves `pos` out of a shallower
    // region of `shard` that also covers it.
    let deepest = map
        .regions
        .iter()
        .filter(|r| r.contains(&pos))
        .max_by_key(|r| r.node.depth)
        .map(|r| r.node);
    map.regions
        .iter()
        .filter(|r| r.shard == shard)
        .map(|r| {
            if r.contains(&pos) {
                deepest.map_or(0.0, |carve| depth_inside(carve, pos))
            } else {
                box_distance(r.node, pos)
            }
        })
        .fold(f64::INFINITY, f64::min)
}

/// How far `pos` is from the nearest ground `map` gives `shard`, metres —
/// **exactly**, where that ground lies in the cell of `pos` or in one of the
/// 26 cells around it; [`f64::INFINITY`] where it does not (it is then more
/// than a cell away). 0 where `shard` owns `pos`.
///
/// A region is a whole number of cells, so who owns a point depends on its
/// cell alone, and the nearest point of a neighbouring cell is found by
/// crossing the cell faces between: the distance is the root of the squared
/// distances to the faces crossed. Unlike [`territory_distance`] this does
/// not take the nearest face of the *region* around `pos` for the way out:
/// beside the corner where four shards' cells meet, the ground behind that
/// face is somebody else's, and the simulator's own ground lies diagonally
/// across the corner.
#[must_use]
pub fn near_territory_distance(map: &ShardMap, shard: ShardId, pos: WorldPos) -> f64 {
    let half = f64::from(space_world::CELL_HALF);
    let mut best = f64::INFINITY;
    for dx in -1i64..=1 {
        for dy in -1i64..=1 {
            for dz in -1i64..=1 {
                let step = [dx, dy, dz];
                let cell = [0, 1, 2].map(|k| pos.cell[k].wrapping_add(step[k]));
                let probe = WorldPos {
                    cell,
                    local: [0.0; 3],
                };
                if map.lookup(&probe) != Some(shard) {
                    continue;
                }
                let d2: f64 = (0..3)
                    .map(|k| match step[k] {
                        -1 => f64::from(pos.local[k]) + half,
                        1 => half - f64::from(pos.local[k]),
                        _ => 0.0,
                    })
                    .map(|d| d * d)
                    .sum();
                best = best.min(d2.sqrt());
            }
        }
    }
    best
}

/// The shard a body belongs on by its own position: the shard simulating it
/// (`simulator`) while it is in that shard's territory or within
/// `hysteresis` metres of it, otherwise the shard the map names (or the
/// simulator again, where the map names nobody).
#[must_use]
pub fn home(map: &ShardMap, pos: WorldPos, simulator: ShardId, hysteresis: f64) -> ShardId {
    match map.lookup(&pos) {
        Some(owner) if owner == simulator => simulator,
        Some(owner) if owner != ShardId::NONE => {
            if hysteresis > 0.0 && near_territory_distance(map, simulator, pos) <= hysteresis {
                simulator
            } else {
                owner
            }
        }
        _ => simulator,
    }
}

/// The shards of `map` other than `own`, ascending, each once.
#[must_use]
pub fn neighbours(map: &ShardMap, own: ShardId) -> Vec<ShardId> {
    let mut out: Vec<ShardId> = map
        .regions
        .iter()
        .map(|r| r.shard)
        .filter(|s| *s != own && *s != ShardId::NONE)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// The margin a body moving at `speed` m/s is ghosted within.
#[must_use]
pub fn ghost_reach(speed: f64) -> f64 {
    GHOST_MARGIN + speed * GHOST_LEAD
}

#[cfg(test)]
mod tests {
    use super::*;
    use space_protocol::ShardRegion;
    use std::net::SocketAddr;

    fn region(depth: u8, index: [i64; 3], shard: u32) -> ShardRegion {
        ShardRegion::new(
            NodeRef::new(depth, index),
            ShardId::new(shard),
            SocketAddr::from(([127, 0, 0, 1], 0)),
        )
    }

    /// Distances to a neighbour's region, to a carved-out octant and back
    /// out of it, and hysteresis.
    #[test]
    fn territory_distance_and_home_follow_the_map() {
        let a = ShardId::new(1);
        let b = ShardId::new(2);
        // Two cells side by side: a owns cell 0, b cell 1 (border at x = 512).
        let side = ShardMap::new(1, vec![region(64, [0, 0, 0], 1), region(64, [1, 0, 0], 2)]);
        let at = |x: f32| WorldPos::new([0, 0, 0], [0.0, 0.0, 0.0]).offset([x, 0.0, 0.0]);
        assert_eq!(territory_distance(&side, a, at(100.0)), 0.0);
        assert!((territory_distance(&side, b, at(500.0)) - 12.0).abs() < 1e-3);
        assert!((territory_distance(&side, a, at(515.0)) - 3.0).abs() < 1e-3);
        // Hysteresis: 1 m into b's cell stays with a, 3 m in goes to b.
        assert_eq!(home(&side, at(513.0), a, BODY_HYSTERESIS), a);
        assert_eq!(home(&side, at(515.0), a, BODY_HYSTERESIS), b);
        assert_eq!(home(&side, at(513.0), a, 0.0), b, "ships have none");
        assert_eq!(home(&side, at(100.0), b, BODY_HYSTERESIS), a);
        // A carved-out octant: b owns the +x+y+z child of a's two-cell node.
        let carved = ShardMap::new(1, vec![region(63, [0, 0, 0], 1), region(64, [1, 1, 1], 2)]);
        let inside = WorldPos::new([1, 1, 1], [0.0; 3]); // centre of b's cell
        assert_eq!(carved.lookup(&inside), Some(b));
        assert!(
            (territory_distance(&carved, a, inside) - 512.0).abs() < 1e-3,
            "a's territory starts where b's octant ends"
        );
        assert_eq!(territory_distance(&carved, b, inside), 0.0);
        assert_eq!(neighbours(&carved, a), vec![b]);
    }

    /// The same distances hold far from the origin (a border at ~1e13 m).
    #[test]
    fn border_geometry_is_exact_far_from_the_origin() {
        let far = 9_765_625_000i64; // 1e13 m / 1024 m
        let side = ShardMap::new(
            1,
            vec![region(64, [far, 0, 0], 1), region(64, [far + 1, 0, 0], 2)],
        );
        let p = WorldPos::new([far, 0, 0], [511.5, 0.0, 0.0]);
        assert!((territory_distance(&side, ShardId::new(2), p) - 0.5).abs() < 1e-3);
        assert_eq!(home(&side, p, ShardId::new(2), 1.0), ShardId::new(2));
        assert_eq!(home(&side, p, ShardId::new(2), 0.0), ShardId::new(1));
    }

    /// Where four cells meet, the distance to a shard's ground is measured
    /// to that ground — round the corner if need be — and the hysteresis
    /// holds a body or a ship on its simulator only while that is near.
    ///
    /// Mutation this catches: measuring to the nearest face of the cell the
    /// point is in (`territory_distance`): half a metre from the face to a
    /// *third* shard's cell then counts as half a metre from the simulator's
    /// ground, thirty metres away, and the hysteresis never lets go.
    #[test]
    fn near_a_corner_of_four_cells_the_distance_is_to_the_shards_own_ground() {
        let (root, west, north, diagonal) = (
            ShardId::new(1),
            ShardId::new(2),
            ShardId::new(3),
            ShardId::new(4),
        );
        // The root owns cell [1,1,1] (and the layer below); the three cells
        // that meet it along the vertical edge x = z = 512 are carved out.
        let map = ShardMap::new(
            1,
            vec![
                region(63, [0, 0, 0], 1),
                region(64, [0, 1, 1], 2),
                region(64, [1, 1, 0], 3),
                region(64, [0, 1, 0], 4),
            ],
        );
        // Metres from the corner edge, at mid height of the upper layer.
        let at = |x: f64, z: f64| {
            WorldPos::from_f64_metres_around(
                WorldPos::new([1, 1, 1], [-512.0, 0.0, -512.0]),
                [x, 0.0, z],
            )
        };
        let near = |shard, x, z| near_territory_distance(&map, shard, at(x, z));
        assert_eq!(map.lookup(&at(3.0, 4.0)), Some(root));
        assert_eq!(map.lookup(&at(-3.0, 4.0)), Some(west));
        assert_eq!(map.lookup(&at(3.0, -4.0)), Some(north));
        assert_eq!(map.lookup(&at(-3.0, -4.0)), Some(diagonal));
        assert_eq!(near(root, 3.0, 4.0), 0.0);
        assert!(
            (near(root, -3.0, 4.0) - 3.0).abs() < 1e-3,
            "straight across"
        );
        assert!(
            (near(root, -3.0, -4.0) - 5.0).abs() < 1e-3,
            "round the corner"
        );
        // In the diagonal cell, half a metre from the west cell and thirty
        // from the corner: the root's ground is thirty metres away.
        let p = at(-30.0, -0.5);
        assert_eq!(map.lookup(&p), Some(diagonal));
        assert!((near_territory_distance(&map, west, p) - 0.5).abs() < 1e-3);
        assert!((near_territory_distance(&map, root, p) - 30.004).abs() < 1e-2);
        assert!(
            territory_distance(&map, root, p) < 1.0,
            "what the lower bound says (and why `home` does not use it)"
        );
        assert_eq!(home(&map, p, west, SHIP_HYSTERESIS), west, "held: 0.5 m");
        assert_eq!(home(&map, p, root, SHIP_HYSTERESIS), diagonal, "let go");
        assert_eq!(home(&map, p, root, BODY_HYSTERESIS), diagonal);
        // More than a cell from anything of `north`'s: not near at all.
        let far = WorldPos::new([-1, 1, 3], [0.0; 3]);
        assert_eq!(near_territory_distance(&map, north, far), f64::INFINITY);
    }

    #[test]
    fn scene_guids_carry_their_revision_and_never_meet_counter_guids() {
        let g = scene_guid(ShardId::new(3), 7, 12);
        assert_eq!(scene_revision_of(g), Some(7));
        assert!(is_stale_scene_body(g, 8));
        assert!(!is_stale_scene_body(g, 7));
        assert_eq!(
            scene_revision_of(EntityGuid::new(ShardId::new(3), 12)),
            None
        );
        assert_ne!(
            scene_guid(ShardId::new(3), 7, 12),
            scene_guid(ShardId::new(3), 7, 13)
        );
        assert_ne!(
            scene_guid(ShardId::new(3), 7, 12),
            scene_guid(ShardId::new(4), 7, 12)
        );
    }
}
