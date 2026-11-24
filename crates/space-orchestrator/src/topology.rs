//! The authoritative carve/merge geometry over `space_world::octree` (H11).
//!
//! Pure and synchronous: given a [`ShardMap`] and a node or a shard, a plan
//! function returns the new map, the epoch it belongs to and the
//! [`EpochChange`]s the orchestrator publishes. Every H-worth of geometry
//! reasoning is testable without a single socket.
//!
//! # The geometry
//!
//! `NodeRef::size_m(d) = CELL_SIZE × 2^(64−d)` with `CELL_SIZE = 1024 m`, so
//! `size_m(64) = 1024 m` (one cell), `size_m(54) = 1 048 576 m = 1024 km`, and
//! a child is exactly half its parent's edge (plan §3.2).
//!
//! * A **carve** ([`carve_plan`]) is *additive* (plan F5): one node lying
//!   inside an existing region is appended with a new owner and every other
//!   region keeps its owner, so `ShardMap::lookup` (deepest match) changes
//!   ownership only inside that node — and only where no deeper region already
//!   says otherwise. The node may be **any** descendant of the region it is
//!   carved out of, down to a single cell ([`MAX_REGION_DEPTH`], depth 64):
//!   that is how the four cells that meet at a scene become four shard areas
//!   under a depth-40 root (`docs/plan-sharding.md` §1).
//! * A **split** ([`split_plan`]) is the special case the autoscaler and the
//!   operator use: carve one **direct child octant** of a shard's region.
//!   Splits are **one octant at a time**, not 8-wide: carving all eight
//!   children of a hot region creates seven empty shards and leaves the eighth
//!   still hot (plan R5).
//! * A **merge** ([`merge_plan`]) is the exact inverse of the carve that made
//!   the region: the region is removed, so `lookup` falls back to the
//!   **enclosing region** ([`enclosing_region`]: the nearest ancestor node that
//!   is a region of the map, however many levels up) and its owner.
//! * A region may only be *split* when its edge is at least
//!   [`SPLIT_MIN_NODE_SIZE_M`] (1024 km, i.e. **depth ≤ 54**): below that a
//!   region is still ~100× the 10 km interest radius, so halving it on load
//!   alone changes routing without relieving anything. The floor is a rule of
//!   the octant split only: a carve names its node and has no size floor.
//! * No region is smaller than one cell: depth 64 is where the octree ends
//!   (`octree::children` is `None` there), so a crowd inside one cell cannot
//!   be split further.
//! * The root must be declared with headroom (see
//!   [`OrchestratorConfig::root_depth`](crate::config::OrchestratorConfig::root_depth)):
//!   nothing can be carved out of a depth-64 root.
//!
//! Every plan is a pure function of the map it is given (H14: the same history
//! yields the same maps), and every produced map is checked by [`validate`]
//! before it is handed back (H11's structural half): a plan returns either a
//! map in which every position has exactly the one owner `lookup` names, or
//! an error and nothing.

use space_protocol::shard::{validate_node, EpochChange, ShardId, ShardMap, ShardRegion};
use space_world::octree;
use space_world::NodeRef;

/// Only a region whose edge is at least this (1024 km, depth ≤ 54) may split.
pub const SPLIT_MIN_NODE_SIZE_M: f64 = 1_048_576.0;

/// The deepest a region may be: one cell (1024 m), where the octree ends.
///
/// It used to be 60 (`SPLIT_MAX_DEPTH`), a defensive bound four levels above
/// the addressing limit. The node rules ([`validate_node`]) are exact at every
/// depth and `octree::children` refuses to go below a cell, so there is
/// nothing to defend against, and a shard area of one cell is what the demo
/// area's four quadrants are made of.
pub const MAX_REGION_DEPTH: u8 = space_world::MAX_DEPTH;

/// A carve plan: the node carved, its new owner, and the map that results.
#[derive(Debug, Clone, PartialEq)]
pub struct Carve {
    /// The node carved out of the region that encloses it.
    pub node: NodeRef,
    /// The id that owns the node from now on.
    pub new_shard: ShardId,
    /// The owner of the enclosing region: whose territory the node is carved
    /// out of, and who gets it back when the carve is merged away.
    pub from: ShardId,
    /// The map with the node's region appended (nobody else changes).
    pub map: ShardMap,
    /// The change to publish through the epoch protocol.
    pub change: EpochChange,
}

/// A split plan: the carved child, its new owner, and the map that results.
#[derive(Debug, Clone, PartialEq)]
pub struct Split {
    /// The octant carved out of the parent (one child, not eight).
    pub child: NodeRef,
    /// The freshly allocated id that owns the child.
    pub new_shard: ShardId,
    /// The map with the child region appended (the parent keeps its owner).
    pub map: ShardMap,
    /// The change to publish through the epoch protocol.
    pub change: EpochChange,
}

/// A merge plan: the carved region removed, and the map that results.
#[derive(Debug, Clone, PartialEq)]
pub struct Merge {
    /// The region being removed (its owner's area folds back into the
    /// enclosing region).
    pub child: NodeRef,
    /// The shard that owns the enclosing region and therefore the merged
    /// region now.
    pub into: ShardId,
    /// The map with the region removed.
    pub map: ShardMap,
    /// The change to publish: the node reverts to the enclosing region's
    /// owner.
    pub change: EpochChange,
}

/// Why a plan could not be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyError {
    /// No region in the map is owned by the shard.
    UnknownShard(ShardId),
    /// The shard's region is too small to split (edge < [`SPLIT_MIN_NODE_SIZE_M`],
    /// i.e. depth > 54).
    TooSmallToSplit(NodeRef),
    /// The shard's region has no enclosing region owned by another shard to
    /// merge into.
    NoMergeTarget(NodeRef),
    /// Every octant of the shard's region is already carved out for another
    /// shard: there is nothing left of it to split off.
    NoFreeOctant(NodeRef),
    /// The node to carve is already a region of the map.
    AlreadyARegion(NodeRef),
    /// The node to carve lies inside no region of the map: there is nobody to
    /// carve it out of.
    NotInsideARegion(NodeRef),
    /// The node to carve is not a node of the octree: deeper than a cell, or
    /// an index its depth does not have ([`validate_node`]'s reason).
    NotANode(String),
    /// A produced map violated an invariant (never expected; a bug guard).
    Invalid(String),
}

impl core::fmt::Display for TopologyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let path =
            |n: &NodeRef| format!("{}:{}:{}:{}", n.depth, n.index[0], n.index[1], n.index[2]);
        match self {
            TopologyError::UnknownShard(s) => write!(f, "no region owned by {s}"),
            TopologyError::TooSmallToSplit(n) => write!(
                f,
                "region at depth {} (edge {:.0} m) is below the split floor",
                n.depth,
                NodeRef::size_m(n.depth)
            ),
            TopologyError::NoMergeTarget(n) => write!(
                f,
                "region at depth {} has no enclosing region to merge into",
                n.depth
            ),
            TopologyError::NoFreeOctant(n) => write!(
                f,
                "every octant of the region at depth {} is already carved out",
                n.depth
            ),
            TopologyError::AlreadyARegion(n) => {
                write!(f, "the node {} is already a region of the map", path(n))
            }
            TopologyError::NotInsideARegion(n) => {
                write!(f, "the node {} lies inside no region of the map", path(n))
            }
            TopologyError::NotANode(reason) => write!(f, "not a node of the octree: {reason}"),
            TopologyError::Invalid(reason) => write!(f, "produced map is invalid: {reason}"),
        }
    }
}

impl std::error::Error for TopologyError {}

/// Edge length of a node at `depth`, in metres (`1024 × 2^(64−depth)`).
#[must_use]
pub fn size_m(depth: u8) -> f64 {
    NodeRef::size_m(depth)
}

/// Whether a node may be split into octants: at least
/// [`SPLIT_MIN_NODE_SIZE_M`] on a side (depth ≤ 54).
///
/// This is the gate of the octant split ([`split_plan`]), which the
/// autoscaler and the operator use. [`carve_plan`] does not ask it.
#[must_use]
pub fn can_split(node: NodeRef) -> bool {
    size_m(node.depth) >= SPLIT_MIN_NODE_SIZE_M
}

/// The region of `map` owned by `shard`, preferring the deepest (most specific).
#[must_use]
pub fn region_of(map: &ShardMap, shard: ShardId) -> Option<&ShardRegion> {
    map.regions
        .iter()
        .filter(|r| r.shard == shard)
        .max_by_key(|r| r.node.depth)
}

/// Whether `inner` lies strictly inside `outer`: `outer` is an ancestor of
/// `inner` in the octree (never the same node).
#[must_use]
pub fn is_inside(inner: NodeRef, outer: NodeRef) -> bool {
    // A node of depth >= 1 is addressed by its lowest cell; the ancestor at
    // `outer.depth` is the node of that depth containing the cell.
    outer.depth < inner.depth && NodeRef::containing_cell(outer.depth, inner.lower_cell()) == outer
}

/// The region of `map` that encloses `node`: the **nearest ancestor** of
/// `node` that is a region, however many levels up. `None` when no region
/// lies around it (a top-level region, or a node outside the map).
///
/// This is whose territory a carve of `node` comes out of, and who a merge of
/// the region at `node` gives it back to. A lookup, not a plan: nothing is
/// cloned or validated, so the operator view may ask it for every region of
/// every snapshot.
#[must_use]
pub fn enclosing_region(map: &ShardMap, node: NodeRef) -> Option<&ShardRegion> {
    map.regions
        .iter()
        .filter(|r| is_inside(node, r.node))
        .max_by_key(|r| r.node.depth)
}

/// The octant of `node` whose child contains the node's centre.
///
/// The centre of a root is the corner the scene's spawn point lies a few
/// metres inside of, on the +x+y+z side (`space_scene::ANCHOR_OFFSET`), and
/// the centre of any other region is where its shard puts a joining ship; so
/// in a region nobody has flown far in, this is the octant that holds the
/// players. `0` for a node with no children (it cannot be split anyway).
#[must_use]
pub fn centre_octant(node: NodeRef) -> usize {
    let centre = node.center();
    octree::children(node)
        .and_then(|children| {
            children
                .iter()
                .position(|child| octree::contains(*child, centre))
        })
        .unwrap_or(0)
}

/// The octants of `shard`'s region that can still be carved out for a new
/// shard (0..=7; bit 0 = +x, bit 1 = +y, bit 2 = +z), in order — **without
/// planning anything**.
///
/// An octant is free exactly when its node is not already a region of the
/// map: for a region that passes [`can_split`] in a valid map, a duplicate node
/// is the only thing [`split_plan`]'s validation can find. So this answers
/// "which octants could be split off?" by looking, where trying each
/// [`split_plan`] clones and re-validates the whole map eight times. It is what
/// the operator view and the pre-check of an operator's request use (they run
/// on the control loop for every region, several times a second);
/// [`split_plan`] remains the authority when a split is actually planned.
///
/// # Errors
///
/// [`TopologyError::UnknownShard`] or [`TopologyError::TooSmallToSplit`].
pub fn free_octants(map: &ShardMap, shard: ShardId) -> Result<Vec<u8>, TopologyError> {
    let region = region_of(map, shard).ok_or(TopologyError::UnknownShard(shard))?;
    let children = octree::children(region.node)
        .filter(|_| can_split(region.node))
        .ok_or(TopologyError::TooSmallToSplit(region.node))?;
    Ok((0..8u8)
        .filter(|octant| {
            let child = children[usize::from(*octant)];
            !map.regions.iter().any(|r| r.node == child)
        })
        .collect())
}

/// The shard `shard`'s region would be merged into — the owner of the region
/// around it ([`enclosing_region`]) — **without planning the merge**.
///
/// The same rule as [`merge_plan`] (which uses this), minus building and
/// validating the resulting map.
///
/// # Errors
///
/// [`TopologyError::UnknownShard`], or [`TopologyError::NoMergeTarget`] if no
/// region of the map encloses the shard's region (it is a top-level region)
/// or the enclosing region is owned by the shard itself.
pub fn merge_target(map: &ShardMap, shard: ShardId) -> Result<ShardId, TopologyError> {
    let region = region_of(map, shard).ok_or(TopologyError::UnknownShard(shard))?;
    enclosing_region(map, region.node)
        .map(|around| around.shard)
        .filter(|owner| *owner != shard)
        .ok_or(TopologyError::NoMergeTarget(region.node))
}

/// Which octant of `shard`'s region to carve: the first one, starting at
/// `first` and going round (`first`, `first + 1`, … modulo 8), whose
/// [`split_plan`] validates.
///
/// A [`LoadReport`](space_protocol::control::LoadReport) carries a player count,
/// not positions, so the octant holding the most players is not knowable here.
/// What *is* knowable is which octants are still free: an octant already carved
/// out for another shard can never be carved again, and always proposing the
/// same one made every split after a region's first fail validation. Callers
/// start at [`centre_octant`] (where ships spawn) so the first split of a fresh
/// region actually moves someone. The choice is a pure function of the map, so
/// the same history yields the same splits (H14).
///
/// # Errors
///
/// [`TopologyError::UnknownShard`], [`TopologyError::TooSmallToSplit`], or
/// [`TopologyError::NoFreeOctant`] when all eight are taken.
pub fn pick_split_octant(
    map: &ShardMap,
    shard: ShardId,
    first: usize,
) -> Result<usize, TopologyError> {
    let region = region_of(map, shard).ok_or(TopologyError::UnknownShard(shard))?;
    if !can_split(region.node) {
        return Err(TopologyError::TooSmallToSplit(region.node));
    }
    // The owner is irrelevant to whether a plan validates, so a placeholder
    // stands in for the id that is only allocated once a plan is known to work.
    (0..8)
        .map(|step| (first + step) % 8)
        .find(|octant| split_plan(map, shard, *octant, ShardId::NONE).is_ok())
        .ok_or(TopologyError::NoFreeOctant(region.node))
}

/// Plan the carve of `node` for `new_owner`: the primitive every new shard
/// area is made with.
///
/// `node` may be any octree node that lies **inside an existing region** of
/// the map ([`enclosing_region`]) — a direct child octant or a single cell
/// twenty-four levels down alike — and is not a region already. Its region is
/// appended with `new_owner`; nothing else in the map changes, so `lookup`
/// answers `new_owner` inside `node` (except inside a still deeper region,
/// which keeps its owner) and exactly what it answered before everywhere else.
/// There is no size floor: the node is named by the caller, not picked by
/// load. [`Carve::from`] is the owner of the enclosing region.
///
/// The new region's endpoint is the un-dialable
/// [`ShardRegion::UNRESOLVED_ENDPOINT`] sentinel (never the enclosing
/// region's address): the orchestrator patches in the real address when the
/// new shard registers (plan §3.4). The returned map is validated before it
/// is handed back.
///
/// # Errors
///
/// [`TopologyError::NotANode`] if `node` is not a node of the octree
/// ([`validate_node`]), [`TopologyError::AlreadyARegion`] if the map already
/// has a region at `node`, [`TopologyError::NotInsideARegion`] if no region
/// encloses it, [`TopologyError::Invalid`] if the produced map fails
/// [`validate`].
pub fn carve_plan(
    map: &ShardMap,
    node: NodeRef,
    new_owner: ShardId,
) -> Result<Carve, TopologyError> {
    validate_node(node.depth, node.index).map_err(TopologyError::NotANode)?;
    if map.regions.iter().any(|r| r.node == node) {
        return Err(TopologyError::AlreadyARegion(node));
    }
    let from = enclosing_region(map, node)
        .ok_or(TopologyError::NotInsideARegion(node))?
        .shard;
    let mut new_map = map.clone();
    new_map.epoch += 1;
    new_map.regions.push(ShardRegion::new(
        node,
        new_owner,
        ShardRegion::UNRESOLVED_ENDPOINT,
    ));
    validate(&new_map).map_err(TopologyError::Invalid)?;
    Ok(Carve {
        node,
        new_shard: new_owner,
        from,
        change: (node, new_owner),
        map: new_map,
    })
}

/// The **layout cells** of a root: the cells handed, in this order, to the
/// shards that register after the root's own (`docs/plan-sharding.md` §2).
///
/// With `k` the cell index of the root's centre corner, the scene's spawn
/// point lies in the cell `[k, k, k]`, which stays with the root's shard. The
/// three other cells of that layer that meet at the corner come first —
/// `[k, k, k-1]`, `[k-1, k, k]`, `[k-1, k, k-1]`: with the root's shard these
/// are the four quadrants of the scene, meeting at scene x = -6 / z = -4
/// (`space_scene::ANCHOR_OFFSET`) — then the four cells of the layer below
/// (`y = k-1`). Seven cells: with the root that is the default `--max-shards`.
///
/// A pure function of the root; whether a cell can still be given away is
/// [`next_layout_cell`]'s question.
#[must_use]
pub fn layout_cells(root: NodeRef) -> [NodeRef; 7] {
    let k = root.center().cell;
    let cell = |dx: i64, dy: i64, dz: i64| {
        NodeRef::new(space_world::MAX_DEPTH, [k[0] + dx, k[1] + dy, k[2] + dz])
    };
    [
        cell(0, 0, -1),
        cell(-1, 0, 0),
        cell(-1, 0, -1),
        cell(0, -1, 0),
        cell(0, -1, -1),
        cell(-1, -1, 0),
        cell(-1, -1, -1),
    ]
}

/// The next layout cell of `root` that can be carved out of `map` for a new
/// shard: the first of [`layout_cells`] that is not a region already and lies
/// inside one ([`carve_plan`] accepts it). `None` when every one is taken —
/// or when the root is a single cell, which has no cells to give.
#[must_use]
pub fn next_layout_cell(map: &ShardMap, root: NodeRef) -> Option<NodeRef> {
    layout_cells(root)
        .into_iter()
        .find(|cell| carve_plan(map, *cell, ShardId::NONE).is_ok())
}

/// Plan a one-octant split of `shard`'s region: the [`carve_plan`] of one of
/// its eight child octants, behind the split's own gate ([`can_split`]).
///
/// `octant` selects which of the eight children to carve (bit 0 = +x, bit 1 = +y,
/// bit 2 = +z, matching `octree::children` order); the caller picks it with
/// [`pick_split_octant`]. `new_shard` is the id the caller allocated for the
/// child.
///
/// # Errors
///
/// [`TopologyError::UnknownShard`] if no region is owned by `shard`,
/// [`TopologyError::TooSmallToSplit`] if the region fails [`can_split`], and
/// what [`carve_plan`] refuses: [`TopologyError::AlreadyARegion`] for an
/// octant that is already carved out, [`TopologyError::Invalid`] if the
/// produced map fails [`validate`].
pub fn split_plan(
    map: &ShardMap,
    shard: ShardId,
    octant: usize,
    new_shard: ShardId,
) -> Result<Split, TopologyError> {
    let carve = carve_plan(map, octant_node(map, shard, octant)?, new_shard)?;
    Ok(Split {
        child: carve.node,
        new_shard,
        change: carve.change,
        map: carve.map,
    })
}

/// The node a split of `shard`'s region carves for `octant` (0..=7, taken
/// modulo 8): that child octant of the region — whether or not it is still
/// free.
///
/// # Errors
///
/// [`TopologyError::UnknownShard`], or [`TopologyError::TooSmallToSplit`] if
/// the region fails [`can_split`].
pub fn octant_node(
    map: &ShardMap,
    shard: ShardId,
    octant: usize,
) -> Result<NodeRef, TopologyError> {
    let region = region_of(map, shard).ok_or(TopologyError::UnknownShard(shard))?;
    octree::children(region.node)
        .filter(|_| can_split(region.node))
        .map(|children| children[octant % 8])
        .ok_or(TopologyError::TooSmallToSplit(region.node))
}

/// Plan the merge of `shard`'s region back into the region around it.
///
/// The merge target is the owner of the **enclosing region**
/// ([`enclosing_region`]: the nearest ancestor node that is a region — the
/// direct parent after an octant split, the root twenty-four levels up for a
/// cell carved out of it). The shard's region is removed and the change
/// reverts its node to that owner, so `lookup` falls back exactly as it did
/// before the carve — the inverse. Regions carved out of the removed one stay
/// where they are, with their owners.
///
/// # Errors
///
/// [`TopologyError::UnknownShard`], [`TopologyError::NoMergeTarget`] if no
/// region encloses the shard's region or the one that does is owned by the
/// shard itself, or [`TopologyError::Invalid`].
pub fn merge_plan(map: &ShardMap, shard: ShardId) -> Result<Merge, TopologyError> {
    let region = region_of(map, shard).ok_or(TopologyError::UnknownShard(shard))?;
    let into = merge_target(map, shard)?;
    let mut new_map = map.clone();
    new_map.epoch += 1;
    new_map
        .regions
        .retain(|r| !(r.node == region.node && r.shard == shard));
    validate(&new_map).map_err(TopologyError::Invalid)?;
    Ok(Merge {
        child: region.node,
        into,
        change: (region.node, into),
        map: new_map,
    })
}

/// Validate a produced map (H11's structural half).
///
/// Checks, in order:
/// * every node passes [`validate_node`] (the same rules decode enforces),
///   which is also the depth limit: no region is deeper than
///   [`MAX_REGION_DEPTH`], one cell;
/// * no two regions of **equal depth** overlap (ownership would be ambiguous).
///
/// Regions of different depths may nest freely — that is what a carve is —
/// and `lookup`'s deepest match gives every position one owner.
///
/// `Ok(())` otherwise `Err(reason)`.
///
/// The pairwise check makes this quadratic in the number of regions, which is
/// fine for the one or two calls a carve or merge makes and is why nothing
/// that runs per region per snapshot may call it (see [`free_octants`]).
pub fn validate(map: &ShardMap) -> Result<(), String> {
    #[cfg(test)]
    VALIDATE_CALLS.with(|calls| calls.set(calls.get() + 1));
    for region in &map.regions {
        validate_node(region.node.depth, region.node.index)?;
    }
    // Equal-depth regions must not overlap: two regions of the same depth
    // overlap iff their nodes are equal (distinct same-depth octrees are
    // disjoint by construction).
    for (i, a) in map.regions.iter().enumerate() {
        for b in map.regions.iter().skip(i + 1) {
            if a.node.depth == b.node.depth && a.node == b.node {
                return Err(format!(
                    "two regions share the depth-{} node {:?}: owners {} and {}",
                    a.node.depth, a.node.index, a.shard.0, b.shard.0
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
thread_local! {
    /// How often [`validate`] ran on this thread: lets a test prove that a
    /// code path (building a snapshot) does not validate at all.
    pub(crate) static VALIDATE_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use space_world::WorldPos;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    /// The cheap answers are the planners' answers: an octant is listed as
    /// free exactly when its split plan validates, and a merge has a target
    /// exactly when its plan does — the same target.
    ///
    /// Mutation this catches: a `free_octants` that forgets a rule the planner
    /// has (the page would offer a button the orchestrator then refuses) or
    /// adds one it does not have.
    #[test]
    fn the_cheap_view_checks_agree_with_the_planners() {
        let root = NodeRef::new(40, [0, 0, 0]);
        let kids = octree::children(root).expect("splits");
        let grandkids = octree::children(kids[3]).expect("splits");
        let floor = NodeRef::new(55, [0, 0, 0]); // below the split floor
        let k = root.center().cell[0];
        let maps = [
            root_map(),
            // Cells carved straight out of the root, twenty-four levels down,
            // and one carved out of a carved octant.
            ShardMap::new(
                6,
                vec![
                    ShardRegion::new(root, shard(1), addr(1)),
                    ShardRegion::new(NodeRef::new(64, [k, k, k - 1]), shard(2), addr(2)),
                    ShardRegion::new(NodeRef::new(64, [k - 1, k, k]), shard(3), addr(3)),
                    ShardRegion::new(kids[0], shard(4), addr(4)),
                    ShardRegion::new(NodeRef::new(64, [0, 0, 0]), shard(5), addr(5)),
                ],
            ),
            ShardMap::new(
                4,
                vec![
                    ShardRegion::new(root, shard(1), addr(1)),
                    ShardRegion::new(kids[0], shard(2), addr(2)),
                    ShardRegion::new(kids[3], shard(3), addr(3)),
                    ShardRegion::new(kids[7], shard(4), addr(4)),
                    ShardRegion::new(grandkids[5], shard(5), addr(5)),
                ],
            ),
            // Every octant of the root is taken.
            ShardMap::new(
                9,
                std::iter::once(ShardRegion::new(root, shard(1), addr(1)))
                    .chain(
                        kids.iter()
                            .enumerate()
                            .map(|(i, kid)| ShardRegion::new(*kid, shard(10 + i as u32), addr(9))),
                    )
                    .collect(),
            ),
            ShardMap::new(1, vec![ShardRegion::new(floor, shard(1), addr(1))]),
        ];
        for map in &maps {
            validate(map).expect("a valid map to start from");
            let mut owners: Vec<ShardId> = map.regions.iter().map(|r| r.shard).collect();
            owners.push(shard(99)); // and somebody the map does not know
            for owner in owners {
                let planned: Vec<u8> = (0..8u8)
                    .filter(|o| split_plan(map, owner, usize::from(*o), ShardId::NONE).is_ok())
                    .collect();
                assert_eq!(
                    free_octants(map, owner).unwrap_or_default(),
                    planned,
                    "{owner} in a map of {} region(s)",
                    map.regions.len()
                );
                // (An octant that is merely taken is not a refusal of the
                // region: `free_octants` leaves it out of its list.)
                assert_eq!(
                    free_octants(map, owner).err(),
                    split_plan(map, owner, 0, ShardId::NONE)
                        .err()
                        .filter(|err| !matches!(err, TopologyError::AlreadyARegion(_))),
                    "{owner}: the same refusal"
                );
                assert_eq!(
                    merge_target(map, owner),
                    merge_plan(map, owner).map(|plan| plan.into),
                    "{owner}"
                );
            }
        }
    }

    fn addr(n: u16) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), n)
    }

    fn shard(n: u32) -> ShardId {
        ShardId::new(n)
    }

    /// The order in which new shards are given their cells is fixed: the three
    /// cells that share the spawn cell's layer and meet it at the root's centre
    /// corner, then the four below. Each is carved in turn; the spawn point's
    /// own cell is never given away, and when all seven are taken there is
    /// nothing left.
    ///
    /// Mutation this catches: handing out the spawn cell (the root's shard
    /// would own no ground at the scene), a different order (which shard owns
    /// which quadrant would depend on something else), or an eighth cell.
    #[test]
    fn layout_cells_are_handed_out_in_a_fixed_order_around_the_spawn_cell() {
        let root = NodeRef::new(40, [0, 0, 0]);
        let k = root.center().cell;
        let cells = layout_cells(root);
        let offsets: Vec<[i64; 3]> = cells
            .iter()
            .map(|c| [c.index[0] - k[0], c.index[1] - k[1], c.index[2] - k[2]])
            .collect();
        assert_eq!(
            offsets,
            vec![
                [0, 0, -1],
                [-1, 0, 0],
                [-1, 0, -1],
                [0, -1, 0],
                [0, -1, -1],
                [-1, -1, 0],
                [-1, -1, -1],
            ]
        );
        assert!(cells.iter().all(|c| c.depth == 64), "single cells");
        let spawn_cell = NodeRef::new(64, k);
        assert!(!cells.contains(&spawn_cell), "the spawn cell stays put");
        // The scene's spawn point is in the spawn cell, a few metres from the
        // corner the first three cells meet it at.
        let map = root_map();
        let anchor = space_scene::scene_anchor(&map).expect("a map has an anchor");
        assert!(octree::contains(spawn_cell, anchor));

        // Carved in turn, each for the next shard, until none is left.
        let mut map = map;
        for (i, expected) in cells.iter().enumerate() {
            let next = next_layout_cell(&map, root).expect("a cell is left");
            assert_eq!(next, *expected, "cell {i}");
            map = carve_plan(&map, next, shard(10 + i as u32))
                .expect("a layout cell lies inside the root")
                .map;
            assert_eq!(map.lookup(&next.center()), Some(shard(10 + i as u32)));
            assert_eq!(
                map.lookup(&anchor),
                Some(shard(1)),
                "the root keeps the spawn"
            );
        }
        assert_eq!(next_layout_cell(&map, root), None, "all seven are taken");
        validate(&map).expect("eight regions, one owner everywhere");

        // After the first three, the four quadrants of the scene have four
        // owners: a point a few metres either side of scene x = -6 / z = -4.
        let at =
            |x: f64, z: f64| map.lookup(&WorldPos::from_f64_metres_around(anchor, [x, 0.0, z]));
        assert_eq!(at(0.0, 0.0), Some(shard(1)));
        assert_eq!(at(0.0, -10.0), Some(shard(10)));
        assert_eq!(at(-12.0, 0.0), Some(shard(11)));
        assert_eq!(at(-12.0, -10.0), Some(shard(12)));

        // A cell that was merged away is the next one handed out again.
        let mut freed = map.clone();
        freed.regions.retain(|r| r.node != cells[1]);
        assert_eq!(next_layout_cell(&freed, root), Some(cells[1]));

        // A root of one cell has nothing to give.
        let one_cell = NodeRef::new(64, [3, 3, 3]);
        let tiny = ShardMap::new(1, vec![ShardRegion::new(one_cell, shard(1), addr(1))]);
        assert_eq!(next_layout_cell(&tiny, one_cell), None);
    }

    fn region(depth: u8, index: [i64; 3], owner: u32) -> ShardRegion {
        ShardRegion::new(
            NodeRef::new(depth, index),
            shard(owner),
            addr(7_000 + owner as u16),
        )
    }

    /// A map with one depth-40 root owned by shard 1 (the config default).
    fn root_map() -> ShardMap {
        ShardMap::new(1, vec![region(40, [0, 0, 0], 1)])
    }

    #[test]
    fn the_depth_size_math_matches_the_plan() {
        // size_m(d) = 1024 × 2^(64−d): the three anchors plan §3.2 names.
        assert_eq!(size_m(64), 1_024.0);
        assert_eq!(size_m(54), 1_048_576.0); // exactly 1024 km
        assert!((size_m(44) - 1.073_741_824e9).abs() < 1e3);
        // A child is exactly half its parent's edge.
        for d in 1..=64u8 {
            assert!(
                (size_m(d) - size_m(d - 1) / 2.0).abs() <= 1e-9 * size_m(d),
                "depth {d} is not half its parent"
            );
        }
    }

    /// The octant split keeps its size floor; the deepest region a map may
    /// hold is one cell, and only a carve gets there.
    #[test]
    fn the_split_gate_is_depth_54_and_a_region_may_be_one_cell() {
        assert!(can_split(NodeRef::new(54, [0, 0, 0])), "1024 km splits");
        assert!(!can_split(NodeRef::new(55, [0, 0, 0])), "512 km does not");
        assert!(!can_split(NodeRef::new(64, [0, 0, 0])), "one cell does not");
        for d in 55..=64u8 {
            assert!(
                !can_split(NodeRef::new(d, [0, 0, 0])),
                "depth {d} is below the floor"
            );
        }
        // The deepest region is one cell: where the octree ends.
        assert_eq!(MAX_REGION_DEPTH, 64);
        assert_eq!(size_m(MAX_REGION_DEPTH), 1_024.0);
        assert!(octree::children(NodeRef::new(MAX_REGION_DEPTH, [0, 0, 0])).is_none());
        // A map with regions at every depth from the root down to a cell is
        // valid (it used to be refused below depth 60)...
        let mut regions = vec![region(40, [0, 0, 0], 1)];
        for depth in 41..=MAX_REGION_DEPTH {
            regions.push(region(depth, [0, 0, 0], u32::from(depth)));
        }
        validate(&ShardMap::new(1, regions)).expect("nested down to one cell");
        // ...and one with a node below a cell, or outside the index range of
        // its depth, is not.
        let below = ShardMap::new(1, vec![region(65, [0, 0, 0], 1)]);
        assert!(validate(&below).is_err(), "depth 65 does not exist");
        let outside = ShardMap::new(1, vec![region(40, [1 << 39, 0, 0], 1)]);
        assert!(validate(&outside).is_err(), "no such node at depth 40");
    }

    /// The cell index of the root's centre corner: the eight cells that meet
    /// there are `k - 1` and `k` on each axis.
    fn centre_cell(root: NodeRef) -> i64 {
        let centre = root.center();
        assert_eq!(centre.local, [-512.0; 3], "a corner, not a cell centre");
        assert_eq!(centre.cell, [centre.cell[0]; 3]);
        centre.cell[0]
    }

    /// A point `d` metres from the root's centre corner.
    fn from_corner(root: NodeRef, d: [f64; 3]) -> WorldPos {
        WorldPos::from_f64_metres_around(root.center(), d)
    }

    /// `docs/plan-sharding.md` §1: three cells carved out of a depth-40 root
    /// — `[k,k,k-1]`, `[k-1,k,k]`, `[k-1,k,k-1]`, with `k` the cell index of
    /// the root's centre corner — leave four shard areas that meet along the
    /// vertical line through that corner. `lookup` is exact on both sides of
    /// every border and at the corner, and merging one cell away gives it back
    /// to the root's owner, twenty-four levels up.
    ///
    /// Mutations this catches: a carve that only takes a direct child octant;
    /// a `validate` that refuses a region below depth 60; a merge that needs
    /// the direct parent to be a region; a merge that removes more than the
    /// one region.
    #[test]
    fn three_cells_carved_out_of_a_depth_40_root_and_one_merged_back() {
        let root = NodeRef::new(40, [0, 0, 0]);
        let k = centre_cell(root);
        assert_eq!(k, 1 << 23, "the depth-40 root is 2^24 cells wide");
        let cells = [
            (NodeRef::new(64, [k, k, k - 1]), shard(2)),
            (NodeRef::new(64, [k - 1, k, k]), shard(3)),
            (NodeRef::new(64, [k - 1, k, k - 1]), shard(4)),
        ];
        let mut map = root_map();
        for (cell, owner) in cells {
            let before = map.clone();
            let plan = carve_plan(&map, cell, owner).expect("a cell inside the root");
            assert_eq!(plan.node, cell);
            assert_eq!(plan.new_shard, owner);
            assert_eq!(plan.from, shard(1), "carved out of the root's territory");
            assert_eq!(plan.change, (cell, owner));
            assert_eq!(plan.map.epoch, before.epoch + 1, "one epoch per carve");
            assert_eq!(plan.map.regions.len(), before.regions.len() + 1);
            assert_eq!(
                &plan.map.regions[..before.regions.len()],
                &before.regions[..],
                "nobody else's region changes"
            );
            let added = plan.map.regions.last().expect("the new region");
            assert_eq!(
                (added.node, added.shard, added.endpoint),
                (cell, owner, ShardRegion::UNRESOLVED_ENDPOINT),
                "named before its shard has an address"
            );
            validate(&plan.map).expect("a depth-64 region validates");
            map = plan.map;
        }
        assert_eq!(map.epoch, 4);

        // The four areas, seen from the vertical line where they meet, at the
        // height of the middle of the cell layer (512 m above the corner):
        // +x+z stays the root's, the other three quadrants are the cells'.
        let owner_of = |map: &ShardMap, x: f64, y: f64, z: f64| {
            map.lookup(&from_corner(root, [x, y, z]))
                .expect("the root covers everything near its centre")
        };
        let quadrants = |map: &ShardMap, y: f64, d: f64| {
            [
                owner_of(map, d, y, d),
                owner_of(map, d, y, -d),
                owner_of(map, -d, y, d),
                owner_of(map, -d, y, -d),
            ]
        };
        let four = [shard(1), shard(2), shard(3), shard(4)];
        for y in [0.001, 512.0, 1023.999] {
            for d in [0.001, 4.0, 6.0, 300.0, 1023.999] {
                assert_eq!(quadrants(&map, y, d), four, "y {y}, {d} m from the line");
            }
        }
        // A border belongs to the cell above it on its axis (a cell spans
        // [lower, upper)): exactly on x = 0 is the +x side, exactly on z = 0
        // the +z side, and the line itself is the root's quadrant.
        assert_eq!(owner_of(&map, 0.0, 512.0, 5.0), shard(1));
        assert_eq!(owner_of(&map, -0.001, 512.0, 5.0), shard(3));
        assert_eq!(owner_of(&map, 5.0, 512.0, 0.0), shard(1));
        assert_eq!(owner_of(&map, 5.0, 512.0, -0.001), shard(2));
        assert_eq!(owner_of(&map, 0.0, 512.0, -5.0), shard(2));
        assert_eq!(owner_of(&map, -0.001, 512.0, -5.0), shard(4));
        assert_eq!(owner_of(&map, -5.0, 512.0, 0.0), shard(3));
        assert_eq!(owner_of(&map, -5.0, 512.0, -0.001), shard(4));
        assert_eq!(owner_of(&map, 0.0, 512.0, 0.0), shard(1));
        // The corner itself — where eight cells meet — is the lowest point of
        // cell [k,k,k], which nobody carved: the root's.
        assert_eq!(map.lookup(&root.center()), Some(shard(1)));
        assert_eq!(owner_of(&map, 0.0, 0.0, 0.0), shard(1));
        // The horizontal borders: the cells are one layer, 1024 m high. Just
        // under it and just over it everything is the root's again, in all
        // four quadrants.
        for y in [-0.001, -300.0, 1024.0, 1300.0] {
            assert_eq!(quadrants(&map, y, 5.0), [shard(1); 4], "y {y}");
        }
        // And the cells end where the next cell begins.
        for (x, z) in [(5.0, -1024.001), (-1024.001, 5.0), (-1024.001, -5.0)] {
            assert_eq!(owner_of(&map, x, 512.0, z), shard(1), "({x}, {z})");
        }
        assert_eq!(owner_of(&map, -5.0, 512.0, -1023.999), shard(4));

        // Merge one cell away: it goes back to the root's owner although the
        // root is not its parent node, and nothing else moves.
        assert_eq!(merge_target(&map, shard(3)), Ok(shard(1)));
        let merge = merge_plan(&map, shard(3)).expect("a carved cell merges");
        assert_eq!(merge.child, NodeRef::new(64, [k - 1, k, k]));
        assert_eq!(merge.into, shard(1), "the nearest enclosing region's owner");
        assert_eq!(merge.change, (merge.child, shard(1)));
        assert_eq!(merge.map.epoch, map.epoch + 1);
        assert_eq!(merge.map.regions.len(), 3);
        assert!(region_of(&merge.map, shard(3)).is_none());
        validate(&merge.map).expect("the merged map validates");
        for y in [0.001, 512.0, 1023.999] {
            for d in [0.001, 6.0, 300.0, 1023.999] {
                assert_eq!(
                    quadrants(&merge.map, y, d),
                    [shard(1), shard(2), shard(1), shard(4)],
                    "y {y}, {d} m from the line: -x+z fell back to the root's owner"
                );
            }
        }
        // The merge is the exact inverse of the carve that made the region.
        let mut without = map.clone();
        without.regions.retain(|r| r.shard != shard(3));
        assert_eq!(merge.map.regions, without.regions);
    }

    /// What a carve refuses, and that a refusal is an error and nothing else.
    #[test]
    fn a_carve_needs_a_node_inside_a_region_that_is_not_one_already() {
        let root = NodeRef::new(40, [0, 0, 0]);
        let k = centre_cell(root);
        let cell = NodeRef::new(64, [k, k, k - 1]);
        let map = carve_plan(&root_map(), cell, shard(2)).expect("carve").map;
        // Already a region: the same cell again, and the root itself.
        assert_eq!(
            carve_plan(&map, cell, shard(3)).unwrap_err(),
            TopologyError::AlreadyARegion(cell)
        );
        assert_eq!(
            carve_plan(&map, root, shard(3)).unwrap_err(),
            TopologyError::AlreadyARegion(root)
        );
        // Inside no region: a neighbour of the root, the root's own parent
        // (which contains the root, and is contained by nothing), a cell far
        // outside it.
        for outside in [
            NodeRef::new(40, [1, 0, 0]),
            NodeRef::new(39, [0, 0, 0]),
            NodeRef::new(64, [-5, k, k]),
        ] {
            assert_eq!(
                carve_plan(&map, outside, shard(3)).unwrap_err(),
                TopologyError::NotInsideARegion(outside),
                "{outside:?}"
            );
        }
        // Not a node of the octree at all: below a cell, or an index its
        // depth does not have.
        for bogus in [
            NodeRef::new(65, [k, k, k]),
            NodeRef::new(41, [1 << 40, 0, 0]),
        ] {
            assert!(
                matches!(
                    carve_plan(&map, bogus, shard(3)).unwrap_err(),
                    TopologyError::NotANode(_)
                ),
                "{bogus:?}"
            );
        }
        // Every refusal left the map alone (a plan borrows it).
        assert_eq!(map.regions.len(), 2);
        assert_eq!(map.epoch, 2);
        // A carve is a pure function of the map: the same call, the same plan.
        let other = NodeRef::new(64, [k - 1, k, k]);
        assert_eq!(
            carve_plan(&map, other, shard(3)),
            carve_plan(&map, other, shard(3))
        );
    }

    /// A node may be carved out of a carved region, and a region may be carved
    /// *around* deeper ones: the deeper regions keep their owners, a merge
    /// gives a region back to whatever encloses it now, and the enclosing
    /// region is always the nearest one.
    #[test]
    fn carves_nest_and_a_merge_falls_back_to_the_nearest_enclosing_region() {
        let root = NodeRef::new(40, [0, 0, 0]);
        let k = centre_cell(root);
        let cell = NodeRef::new(64, [k, k, k]);
        let octant = octree::children(root).expect("children")[7];
        assert!(is_inside(cell, octant) && is_inside(octant, root));
        assert!(is_inside(cell, root), "any number of levels up");
        assert!(!is_inside(root, root) && !is_inside(root, cell));
        assert!(!is_inside(NodeRef::new(64, [k - 1, k, k]), octant));

        // The cell first, then the octant around it.
        let map = carve_plan(&root_map(), cell, shard(2)).expect("cell").map;
        assert_eq!(
            enclosing_region(&map, cell).map(|r| r.shard),
            Some(shard(1))
        );
        let around = carve_plan(&map, octant, shard(3)).expect("the octant around the cell");
        assert_eq!(around.from, shard(1));
        let map = around.map;
        let inside_cell = WorldPos::new([k, k, k], [0.0; 3]);
        let inside_octant = WorldPos::new([k + 5, k, k], [0.0; 3]);
        let outside = WorldPos::new([k - 1, k, k], [0.0; 3]);
        assert_eq!(
            map.lookup(&inside_cell),
            Some(shard(2)),
            "the cell kept its owner"
        );
        assert_eq!(map.lookup(&inside_octant), Some(shard(3)));
        assert_eq!(map.lookup(&outside), Some(shard(1)));
        // The cell is now enclosed by the octant, the octant by the root.
        assert_eq!(
            enclosing_region(&map, cell).map(|r| r.shard),
            Some(shard(3))
        );
        assert_eq!(
            enclosing_region(&map, octant).map(|r| r.shard),
            Some(shard(1))
        );
        assert!(enclosing_region(&map, root).is_none(), "a top-level region");
        assert_eq!(merge_target(&map, shard(2)), Ok(shard(3)));
        assert_eq!(merge_target(&map, shard(3)), Ok(shard(1)));
        // Merging the cell gives it to the octant's owner...
        let merged = merge_plan(&map, shard(2)).expect("merge the cell");
        assert_eq!(merged.into, shard(3));
        assert_eq!(merged.map.lookup(&inside_cell), Some(shard(3)));
        // ...and merging the octant instead leaves the cell where it is, now
        // enclosed by the root.
        let merged = merge_plan(&map, shard(3)).expect("merge the octant");
        assert_eq!(merged.into, shard(1));
        assert_eq!(merged.map.lookup(&inside_cell), Some(shard(2)));
        assert_eq!(merged.map.lookup(&inside_octant), Some(shard(1)));
        assert_eq!(merge_target(&merged.map, shard(2)), Ok(shard(1)));
        validate(&merged.map).expect("valid");
    }

    /// The octant split is the carve of that octant, nothing more.
    #[test]
    fn a_split_is_the_carve_of_one_child_octant() {
        let map = root_map();
        let kids = octree::children(NodeRef::new(40, [0, 0, 0])).expect("children");
        for (octant, kid) in kids.iter().enumerate() {
            let split = split_plan(&map, shard(1), octant, shard(2)).expect("split");
            let carve = carve_plan(&map, *kid, shard(2)).expect("carve");
            assert_eq!(
                (split.child, split.new_shard, split.change, &split.map),
                (carve.node, carve.new_shard, carve.change, &carve.map),
                "octant {octant}"
            );
            assert_eq!(carve.from, shard(1));
        }
        // A taken octant is refused as what it is.
        let taken = split_plan(&map, shard(1), 3, shard(2)).expect("split").map;
        assert_eq!(
            split_plan(&taken, shard(1), 3, shard(5)).unwrap_err(),
            TopologyError::AlreadyARegion(kids[3])
        );
        // The split's size floor is the split's, not the carve's: a 512 km
        // region cannot be split, and a cell can still be carved out of it.
        let small = ShardMap::new(1, vec![region(55, [0, 0, 0], 1)]);
        assert!(matches!(
            split_plan(&small, shard(1), 0, shard(2)).unwrap_err(),
            TopologyError::TooSmallToSplit(_)
        ));
        let cell = NodeRef::new(64, [3, 4, 5]);
        assert!(is_inside(cell, NodeRef::new(55, [0, 0, 0])));
        carve_plan(&small, cell, shard(2)).expect("no size floor on a carve");
    }

    #[test]
    fn a_split_carves_exactly_one_octant() {
        let map = root_map();
        let before = map.regions.len();
        let plan = split_plan(&map, shard(1), 0, shard(2)).expect("split");
        assert_eq!(plan.map.regions.len(), before + 1, "one region appended");
        assert_eq!(plan.new_shard, shard(2));
        assert_eq!(plan.change, (plan.child, shard(2)));
        assert_eq!(plan.map.epoch, map.epoch + 1, "the map version bumps");
        // The child is the octant-0 child of the root, and the parent keeps shard 1.
        let kids = octree::children(NodeRef::new(40, [0, 0, 0])).unwrap();
        assert_eq!(plan.child, kids[0]);
        assert_eq!(
            region_of(&plan.map, shard(1)).unwrap().node,
            NodeRef::new(40, [0, 0, 0])
        );
        assert_eq!(region_of(&plan.map, shard(2)).unwrap().node, kids[0]);
        // The child is not yet reachable: its endpoint is the un-dialable
        // sentinel, never the parent's address.
        assert_eq!(
            region_of(&plan.map, shard(2)).unwrap().endpoint,
            ShardRegion::UNRESOLVED_ENDPOINT
        );
        validate(&plan.map).expect("produced map validates");
    }

    #[test]
    fn a_split_changes_ownership_only_inside_the_child() {
        // H11: lookup agrees with "old owner outside, new owner inside" at every
        // octant boundary of the parent.
        let map = root_map();
        let plan = split_plan(&map, shard(1), 3, shard(2)).expect("split");
        let parent = NodeRef::new(40, [0, 0, 0]);
        for kid in octree::children(parent).unwrap() {
            let centre = kid.center();
            let owner = plan
                .map
                .lookup(&centre)
                .expect("parent covers its children");
            if kid == plan.child {
                assert_eq!(owner, shard(2), "inside the child: new owner");
            } else {
                assert_eq!(owner, shard(1), "outside the child: old owner");
            }
        }
        // And the parent's owner is unchanged for a point outside every child
        // boundary check above — the root still resolves to shard 1 elsewhere.
        assert_eq!(plan.map.lookup(&WorldPos::ZERO).unwrap(), shard(1));
    }

    #[test]
    fn a_merge_is_the_exact_inverse_of_the_split() {
        let map = root_map();
        let split = split_plan(&map, shard(1), 5, shard(2)).expect("split");
        let merge = merge_plan(&split.map, shard(2)).expect("merge");
        assert_eq!(merge.into, shard(1), "the parent's owner absorbs the child");
        assert_eq!(merge.change, (split.child, shard(1)));
        // The merged map has the same regions as the original (order aside).
        assert_eq!(merge.map.regions.len(), map.regions.len());
        let mut a = map.regions.clone();
        let mut b = merge.map.regions.clone();
        a.sort_by_key(|r| (r.node.depth, r.node.index, r.shard.0));
        b.sort_by_key(|r| (r.node.depth, r.node.index, r.shard.0));
        assert_eq!(a, b, "merge undoes split exactly");
        // Epochs keep moving: split then merge is two bumps.
        assert_eq!(merge.map.epoch, map.epoch + 2);
    }

    #[test]
    fn splitting_a_too_small_region_is_refused() {
        let small = ShardMap::new(1, vec![region(55, [0, 0, 0], 1)]);
        let err = split_plan(&small, shard(1), 0, shard(2)).unwrap_err();
        assert!(matches!(err, TopologyError::TooSmallToSplit(_)), "{err:?}");
        // A depth-64 region (Phase 2's test convention) can never split.
        let cell = ShardMap::new(1, vec![region(64, [0, 0, 0], 1)]);
        assert!(split_plan(&cell, shard(1), 0, shard(2)).is_err());
        // An unknown shard is a different error.
        assert_eq!(
            split_plan(&root_map(), shard(99), 0, shard(2)).unwrap_err(),
            TopologyError::UnknownShard(shard(99))
        );
    }

    #[test]
    fn the_centre_of_a_region_lies_in_its_last_octant() {
        // A node spans [lower, lower + size) per axis and its centre is the
        // lower corner of the +x+y+z child, so that child (octant 7) is the one
        // a ship spawned at the centre is in.
        let root = NodeRef::new(40, [0, 0, 0]);
        assert_eq!(centre_octant(root), 7);
        let kids = octree::children(root).unwrap();
        assert!(octree::contains(kids[7], root.center()));
        assert_eq!(root_map().lookup(&root.center()), Some(shard(1)));
        // A single cell has no children; the answer is the harmless 0.
        assert_eq!(centre_octant(NodeRef::new(64, [0, 0, 0])), 0);
    }

    /// Mutation this catches: always proposing the same octant. The second split
    /// of a region then names a node the map already has and fails validation —
    /// for ever, so a hot shard could be relieved exactly once.
    #[test]
    fn the_picked_octant_skips_the_ones_already_carved_and_starts_where_told() {
        let mut map = root_map();
        // Fresh region: the octant asked for first is the one picked.
        assert_eq!(pick_split_octant(&map, shard(1), 7), Ok(7));
        assert_eq!(pick_split_octant(&map, shard(1), 0), Ok(0));
        // Carve all eight, one at a time, always asking for 7 first: every pick
        // is a new octant, in the documented order 7, 0, 1, … 6.
        let mut picked = Vec::new();
        for round in 0..8u32 {
            let octant = pick_split_octant(&map, shard(1), 7).expect("a free octant");
            picked.push(octant);
            map = split_plan(&map, shard(1), octant, shard(round + 2))
                .expect("the picked octant's plan validates")
                .map;
        }
        assert_eq!(picked, vec![7, 0, 1, 2, 3, 4, 5, 6]);
        // Nothing is left of the root to split off.
        assert_eq!(
            pick_split_octant(&map, shard(1), 7),
            Err(TopologyError::NoFreeOctant(NodeRef::new(40, [0, 0, 0])))
        );
        // The other refusals are reported as themselves.
        assert_eq!(
            pick_split_octant(&map, shard(99), 0),
            Err(TopologyError::UnknownShard(shard(99)))
        );
        let small = ShardMap::new(1, vec![region(55, [0, 0, 0], 1)]);
        assert!(matches!(
            pick_split_octant(&small, shard(1), 0),
            Err(TopologyError::TooSmallToSplit(_))
        ));
    }

    #[test]
    fn merging_a_root_region_is_refused() {
        // The root has no parent, so it cannot merge.
        let err = merge_plan(&root_map(), shard(1)).unwrap_err();
        assert!(matches!(err, TopologyError::NoMergeTarget(_)), "{err:?}");
    }

    #[test]
    fn no_two_regions_at_equal_depth_overlap_in_any_produced_map() {
        // Fan the root out one octant at a time, five rounds, and validate after
        // each: the equal-depth-overlap rule must hold throughout.
        let mut map = root_map();
        for (round, octant) in (0..5usize).enumerate() {
            let new_id = round as u32 + 2;
            let hot = region_of(&map, shard(new_id - 1))
                .map(|r| r.shard)
                .unwrap_or(shard(1));
            let plan = split_plan(&map, hot, octant, shard(new_id)).expect("split");
            map = plan.map;
            validate(&map).expect("map after fan-out round");
        }
        // Explicit duplicate detection: hand-build a map that violates the rule.
        let bad = ShardMap::new(1, vec![region(40, [0, 0, 0], 1), region(40, [0, 0, 0], 2)]);
        assert!(
            validate(&bad).is_err(),
            "equal-depth overlap must be rejected"
        );
    }

    #[test]
    fn children_composed_from_the_root_stay_valid_down_to_one_cell() {
        // Plan §3.2's property: children() composed repeatedly from the root
        // always yields nodes validate_node accepts, all the way down to a
        // single cell. Walking one octant per level (not the whole frontier)
        // keeps the test linear: the property is about index arithmetic, not
        // about breadth. Every node on the way can be carved out of the root.
        let root = NodeRef::new(40, [0, 0, 0]);
        let mut node = root;
        let mut steps = 0;
        while node.depth < MAX_REGION_DEPTH {
            let kids = octree::children(node).expect("above a cell, children exist");
            node = kids[steps % 8];
            validate_node(node.depth, node.index).expect("a child of a valid node is a valid node");
            assert!(is_inside(node, root), "depth {}", node.depth);
            let plan = carve_plan(&root_map(), node, shard(2)).expect("any descendant carves");
            assert_eq!(plan.map.lookup(&node.center()), Some(shard(2)));
            steps += 1;
        }
        assert_eq!(node.depth, MAX_REGION_DEPTH);
        assert_eq!(steps, usize::from(MAX_REGION_DEPTH - 40));
        assert!(
            octree::children(node).is_none(),
            "the octree ends at a cell"
        );
        // The same at the far corners of the cell space, where the indices
        // are at the ends of the i64 range.
        for cell in [[i64::MAX; 3], [i64::MIN; 3], [i64::MIN, 0, i64::MAX]] {
            let leaf = NodeRef::containing_cell(MAX_REGION_DEPTH, cell);
            let top = NodeRef::containing_cell(40, cell);
            validate_node(leaf.depth, leaf.index).expect("a valid cell");
            assert!(is_inside(leaf, top), "{cell:?}");
            let map = ShardMap::new(1, vec![ShardRegion::new(top, shard(1), addr(1))]);
            let plan = carve_plan(&map, leaf, shard(2)).expect("carves at the edge of space");
            assert_eq!(plan.map.lookup(&leaf.center()), Some(shard(2)));
        }
    }
}
