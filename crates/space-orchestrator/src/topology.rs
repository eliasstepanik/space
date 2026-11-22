//! The authoritative split/merge geometry over `space_world::octree` (H11).
//!
//! Pure and synchronous: given a [`ShardMap`] and a shard, a plan function
//! returns the new map, the epoch it belongs to and the [`EpochChange`]s the
//! orchestrator publishes. Every H-worth of geometry reasoning is testable
//! without a single socket.
//!
//! # The geometry
//!
//! `NodeRef::size_m(d) = CELL_SIZE × 2^(64−d)` with `CELL_SIZE = 1024 m`, so
//! `size_m(64) = 1024 m` (one cell), `size_m(54) = 1 048 576 m = 1024 km`, and
//! a child is exactly half its parent's edge (plan §3.2).
//!
//! * A **split** is *additive* (plan F5): one child octant is appended with the
//!   new owner and the parent keeps the old one, so `ShardMap::lookup` (deepest
//!   match) changes ownership only inside the child. Splits are **one octant at
//!   a time**, not 8-wide: carving all eight children of a hot region creates
//!   seven empty shards and leaves the eighth still hot (plan R5).
//! * A **merge** is the exact inverse: the child region is removed, so `lookup`
//!   falls back to the parent's owner.
//! * A region may only split when its edge is at least [`SPLIT_MIN_NODE_SIZE_M`]
//!   (1024 km, i.e. **depth ≤ 54**): below that a region is still ~100× the 10 km
//!   interest radius, so splitting it changes routing without relieving load.
//! * Splitting is capped at [`SPLIT_MAX_DEPTH`] (60) — four levels below the
//!   addressing limit where `i64` indices saturate (plan §3.2 "ceiling").
//! * The root must be declared with split headroom (see
//!   [`OrchestratorConfig::root_depth`](crate::config::OrchestratorConfig::root_depth)):
//!   a depth-64 root can never split.

use space_protocol::shard::{validate_node, EpochChange, ShardId, ShardMap, ShardRegion};
use space_world::octree;
use space_world::NodeRef;

/// Only a region whose edge is at least this (1024 km, depth ≤ 54) may split.
pub const SPLIT_MIN_NODE_SIZE_M: f64 = 1_048_576.0;

/// Deepest a region may be split to. See the module docs' "ceiling" note.
pub const SPLIT_MAX_DEPTH: u8 = 60;

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

/// A merge plan: the child region removed, and the map that results.
#[derive(Debug, Clone, PartialEq)]
pub struct Merge {
    /// The child region being removed (its owner folds back into the parent).
    pub child: NodeRef,
    /// The shard that owns the parent and therefore the merged region now.
    pub into: ShardId,
    /// The map with the child region removed.
    pub map: ShardMap,
    /// The change to publish: the child node reverts to the parent's owner.
    pub change: EpochChange,
}

/// Why a plan could not be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyError {
    /// No region in the map is owned by the shard.
    UnknownShard(ShardId),
    /// The shard's region is too small to split (edge < [`SPLIT_MIN_NODE_SIZE_M`],
    /// i.e. depth > 54) or deeper than [`SPLIT_MAX_DEPTH`].
    TooSmallToSplit(NodeRef),
    /// The shard's region has no parent in the map to merge into.
    NoMergeTarget(NodeRef),
    /// Every octant of the shard's region is already carved out for another
    /// shard: there is nothing left of it to split off.
    NoFreeOctant(NodeRef),
    /// A produced map violated an invariant (never expected; a bug guard).
    Invalid(String),
}

impl core::fmt::Display for TopologyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TopologyError::UnknownShard(s) => write!(f, "no region owned by {s}"),
            TopologyError::TooSmallToSplit(n) => write!(
                f,
                "region at depth {} (edge {:.0} m) is below the split floor",
                n.depth,
                NodeRef::size_m(n.depth)
            ),
            TopologyError::NoMergeTarget(n) => {
                write!(f, "region at depth {} has no parent to merge into", n.depth)
            }
            TopologyError::NoFreeOctant(n) => write!(
                f,
                "every octant of the region at depth {} is already carved out",
                n.depth
            ),
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

/// Whether a node may be split: at least [`SPLIT_MIN_NODE_SIZE_M`] on a side and
/// no deeper than [`SPLIT_MAX_DEPTH`].
#[must_use]
pub fn can_split(node: NodeRef) -> bool {
    // The size gate is the binding one (depth <= 54); the depth cap is enforced
    // on every produced map by `validate`, so it never binds here in practice.
    size_m(node.depth) >= SPLIT_MIN_NODE_SIZE_M && node.depth < SPLIT_MAX_DEPTH
}

/// The region of `map` owned by `shard`, preferring the deepest (most specific).
#[must_use]
pub fn region_of(map: &ShardMap, shard: ShardId) -> Option<&ShardRegion> {
    map.regions
        .iter()
        .filter(|r| r.shard == shard)
        .max_by_key(|r| r.node.depth)
}

/// The octant of `node` whose child contains the node's centre.
///
/// The centre is where a shard spawns a joining ship, so in a region nobody has
/// flown far in, this is the octant that holds the players. `0` for a node with
/// no children (it cannot be split anyway).
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

/// Plan a one-octant split of `shard`'s region.
///
/// `octant` selects which of the eight children to carve (bit 0 = +x, bit 1 = +y,
/// bit 2 = +z, matching `octree::children` order); the caller picks it with
/// [`pick_split_octant`]. `new_shard` is the id the caller allocated for the
/// child.
///
/// The child's endpoint is the un-dialable [`ShardRegion::UNRESOLVED_ENDPOINT`]
/// sentinel (never the parent's address): the orchestrator patches in the real
/// address when the new shard registers (plan §3.4). The returned map is
/// validated before it is handed back.
///
/// # Errors
///
/// [`TopologyError::UnknownShard`] if no region is owned by `shard`,
/// [`TopologyError::TooSmallToSplit`] if the region fails [`can_split`],
/// [`TopologyError::Invalid`] if the produced map fails [`validate`].
pub fn split_plan(
    map: &ShardMap,
    shard: ShardId,
    octant: usize,
    new_shard: ShardId,
) -> Result<Split, TopologyError> {
    let region = region_of(map, shard).ok_or(TopologyError::UnknownShard(shard))?;
    if !can_split(region.node) {
        return Err(TopologyError::TooSmallToSplit(region.node));
    }
    let children = octree::children(region.node).expect("can_split implies children exist");
    let child = children[octant % 8];
    let mut new_map = map.clone();
    new_map.epoch += 1;
    new_map.regions.push(ShardRegion::new(
        child,
        new_shard,
        ShardRegion::UNRESOLVED_ENDPOINT,
    ));
    validate(&new_map).map_err(TopologyError::Invalid)?;
    Ok(Split {
        child,
        new_shard,
        change: (child, new_shard),
        map: new_map,
    })
}

/// Plan the merge of `shard`'s region back into its parent's owner.
///
/// The merge target is the **parent** node's owner (the common case after a
/// split, plan §3.2). The child region is removed and the change reverts the
/// child node to the parent's owner, so `lookup` falls back exactly as it did
/// before the split — the inverse.
///
/// # Errors
///
/// [`TopologyError::UnknownShard`], [`TopologyError::NoMergeTarget`] if the
/// parent is not in the map or is owned by the shard itself, or
/// [`TopologyError::Invalid`].
pub fn merge_plan(map: &ShardMap, shard: ShardId) -> Result<Merge, TopologyError> {
    let region = region_of(map, shard).ok_or(TopologyError::UnknownShard(shard))?;
    let parent = octree::parent(region.node).ok_or(TopologyError::NoMergeTarget(region.node))?;
    let parent_owner = map
        .regions
        .iter()
        .find(|r| r.node == parent)
        .map(|r| r.shard)
        .filter(|owner| *owner != shard)
        .ok_or(TopologyError::NoMergeTarget(region.node))?;
    let mut new_map = map.clone();
    new_map.epoch += 1;
    new_map
        .regions
        .retain(|r| !(r.node == region.node && r.shard == shard));
    validate(&new_map).map_err(TopologyError::Invalid)?;
    Ok(Merge {
        child: region.node,
        into: parent_owner,
        change: (region.node, parent_owner),
        map: new_map,
    })
}

/// Validate a produced map (H11's structural half).
///
/// Checks, in order:
/// * every node passes [`validate_node`] (the same rules decode enforces);
/// * no two regions of **equal depth** overlap (ownership would be ambiguous);
/// * no region is deeper than [`SPLIT_MAX_DEPTH`] (the split ceiling).
///
/// `Ok(())` otherwise `Err(reason)`.
pub fn validate(map: &ShardMap) -> Result<(), String> {
    for region in &map.regions {
        validate_node(region.node.depth, region.node.index)?;
        if region.node.depth > SPLIT_MAX_DEPTH {
            return Err(format!(
                "region at depth {} is deeper than SPLIT_MAX_DEPTH {SPLIT_MAX_DEPTH}",
                region.node.depth
            ));
        }
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
mod tests {
    use super::*;
    use space_world::WorldPos;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    fn addr(n: u16) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), n)
    }

    fn shard(n: u32) -> ShardId {
        ShardId::new(n)
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

    #[test]
    fn the_split_gate_is_depth_54_and_the_cap_is_60() {
        assert!(can_split(NodeRef::new(54, [0, 0, 0])), "1024 km splits");
        assert!(!can_split(NodeRef::new(55, [0, 0, 0])), "512 km does not");
        assert!(!can_split(NodeRef::new(64, [0, 0, 0])), "one cell does not");
        // The cap: a node at SPLIT_MAX_DEPTH cannot be split further.
        assert!(!can_split(NodeRef::new(SPLIT_MAX_DEPTH, [0, 0, 0])));
        // The size gate binds first: 54 < SPLIT_MAX_DEPTH, so no node can ever
        // reach the cap through can_split alone. The cap is a belt-and-braces
        // ceiling on the *produced map*, not the usual gate.
        const { assert!(54 < SPLIT_MAX_DEPTH) };
        for d in 55..=64u8 {
            assert!(
                !can_split(NodeRef::new(d, [0, 0, 0])),
                "depth {d} is below the floor"
            );
        }
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
    fn children_composed_from_the_root_stay_valid_to_the_cap() {
        // Plan §3.2's property: children() composed repeatedly from the root
        // always yields nodes validate_node accepts, down to the cap. Walking
        // one octant per level (not the whole frontier) keeps the test linear:
        // the property is about index arithmetic, not about breadth.
        let mut node = NodeRef::new(40, [0, 0, 0]);
        let mut steps = 0;
        while node.depth < SPLIT_MAX_DEPTH {
            let kids = octree::children(node).expect("above the cap, children exist");
            node = kids[steps % 8];
            validate_node(node.depth, node.index).expect("a child of a valid node is a valid node");
            steps += 1;
        }
        assert_eq!(node.depth, SPLIT_MAX_DEPTH);
        assert_eq!(steps, usize::from(SPLIT_MAX_DEPTH - 40));
    }
}
