//! Shard identity, the epoch-versioned shard map and the epoch-change handshake.
//!
//! A [`ShardMap`] is the one piece of routing state every participant caches:
//! it says which octree region of the world which shard owns, over which
//! endpoint, and at which [`epoch`](ShardMap::epoch). Because regions are
//! [`space_world::NodeRef`]s of the implicit octree, a region is described by
//! two integers and containment is O(1) — no polygon, no tree to ship.
//!
//! Ownership changes are versioned by an epoch. The orchestrator publishes an
//! [`EpochPrepare`] naming the new owner of each affected node, collects
//! [`EpochAck`]s, then publishes [`EpochCommit`] (or [`EpochAbort`] to roll the
//! attempt back). Every other message carries or is checked against an epoch,
//! so a shard that is still running an older map answers with
//! [`crate::StaleEpoch`] instead of silently processing a message for a region
//! it no longer owns.

use std::net::SocketAddr;

use serde::{Deserialize, Serialize};
use space_world::{octree, NodeRef, WorldPos};

/// Wire form of [`NodeRef`].
///
/// `space-world` deliberately has no dependencies, so its types carry no serde
/// impls; the protocol converts them here instead of pushing `serde` into the
/// world model. The layout is the natural one — depth plus the per-axis index —
/// and it is stable because `NodeRef`'s addressing is pure (see
/// `space_world::octree`).
#[derive(Serialize, Deserialize)]
struct NodeWire {
    depth: u8,
    index: [i64; 3],
}

mod node_wire {
    use super::NodeWire;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use space_world::NodeRef;

    pub fn serialize<S: Serializer>(node: &NodeRef, s: S) -> Result<S::Ok, S::Error> {
        let wire = NodeWire {
            depth: node.depth,
            index: node.index,
        };
        wire.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<NodeRef, D::Error> {
        let wire = NodeWire::deserialize(d)?;
        super::validate_node(wire.depth, wire.index).map_err(serde::de::Error::custom)?;
        Ok(NodeRef::new(wire.depth, wire.index))
    }
}

/// Validate an octree node that arrived on the wire.
///
/// A [`NodeRef`](space_world::NodeRef) is a bare `(depth, index)` and
/// `NodeRef::new` accepts anything, so a malformed shard map could carry a node
/// no real position can ever address. Rejecting on decode keeps every consumer
/// of a decoded map (lookup, path, handoff) inside the addressing rules:
///
/// * `depth <= `[`space_world::MAX_DEPTH`] (64);
/// * at depth 0 the index is `[0, 0, 0]` — there is exactly one root;
/// * at depth `d >= 1`, each axis index is in `[-2^(d-1), 2^(d-1))`, the range
///   `cell >> (64 - d)` can produce.
///
/// Public because `space_orchestrator::topology::validate` enforces exactly
/// these rules on every map it produces (plan §3.2) — one rule, one
/// implementation.
pub fn validate_node(depth: u8, index: [i64; 3]) -> Result<(), String> {
    if depth > space_world::MAX_DEPTH {
        return Err(format!(
            "octree node depth {depth} exceeds MAX_DEPTH {}",
            space_world::MAX_DEPTH
        ));
    }
    if depth == 0 {
        if index != [0, 0, 0] {
            return Err(format!(
                "root node must have index [0, 0, 0], got {index:?}"
            ));
        }
        return Ok(());
    }
    // half = 2^(d-1); the valid range is [-half, half).
    let half: i128 = 1i128 << (u32::from(depth) - 1);
    for (axis, &i) in index.iter().enumerate() {
        let v = i128::from(i);
        if v < -half || v >= half {
            return Err(format!(
                "octree node index axis {axis} value {i} is outside [-{half}, {half}) at depth {depth}"
            ));
        }
    }
    Ok(())
}

/// Identifier of a shard process.
///
/// # Allocation rule: monotonic and never reused
///
/// Shard ids are allocated from a counter that only ever increases; an id is
/// **never** handed back to a shard that stopped (see `docs/plan.md` FIX 9).
/// This is a correctness requirement, not a policy preference: an
/// [`EntityGuid`](crate::EntityGuid) is unique only as the pair
/// `(shard, serial)`, and a shard mints `serial`s from its own counter. If a
/// shard id were recycled, the new shard's serials would collide with the old
/// shard's still-live guids and two distinct entities would share one guid. The
/// orchestrator owns the counter and persists it across restarts. If
/// monotonicity proves impractical, the alternative is an incarnation field on
/// the guid (a Phase 3 decision) — until then, never reuse an id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ShardId(pub u32);

impl ShardId {
    /// The reserved id meaning "no shard" (unassigned region).
    pub const NONE: Self = Self(u32::MAX);

    /// Build a shard id.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }
}

impl core::fmt::Display for ShardId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "shard {}", self.0)
    }
}

/// One region of the shard map: an octree node, its owner and how to reach it.
///
/// `node` is the *exclusive* authority region of `shard`: nodes may nest (a
/// depth-20 node inside a depth-16 node) and the deepest match wins, which is
/// how a split is expressed — the child node is added with the new owner and
/// the parent keeps the old one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShardRegion {
    /// Octree node covered by this region.
    #[serde(with = "node_wire")]
    pub node: NodeRef,
    /// Shard that owns everything inside `node`.
    pub shard: ShardId,
    /// Where to reach that shard (UDP for quinn, per `docs/plan.md`).
    pub endpoint: SocketAddr,
}

impl ShardRegion {
    /// Sentinel endpoint (`0.0.0.0:0`) for a region whose owner has not yet
    /// registered a real address.
    ///
    /// A carved child is named in the map before its shard has bound a socket.
    /// Borrowing the parent's address instead would make a peer dial the wrong
    /// shard; this port-0 address is un-dialable, and consumers
    /// (`Shard::peer_endpoint`) skip it until a map push supplies the real one.
    pub const UNRESOLVED_ENDPOINT: SocketAddr =
        SocketAddr::new(std::net::IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED), 0);

    /// Build a region.
    #[must_use]
    pub fn new(node: NodeRef, shard: ShardId, endpoint: SocketAddr) -> Self {
        Self {
            node,
            shard,
            endpoint,
        }
    }

    /// Whether this region covers `pos`.
    #[must_use]
    pub fn contains(&self, pos: &WorldPos) -> bool {
        octree::contains(self.node, *pos)
    }
}

/// The routing table: which octree region which shard owns, at which epoch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShardMap {
    /// Epoch this map describes. Monotonic across the world.
    pub epoch: u64,
    /// Regions, in any order. Overlapping regions are resolved by depth.
    pub regions: Vec<ShardRegion>,
}

impl ShardMap {
    /// Build a map.
    #[must_use]
    pub fn new(epoch: u64, regions: Vec<ShardRegion>) -> Self {
        Self { epoch, regions }
    }

    /// The shard that owns `pos`, or `None` when no region covers it.
    ///
    /// The **deepest** matching node wins, because a nested node is a more
    /// specific statement about ownership than its ancestor (that is how a
    /// split overrides the region it carved out). Regions of equal depth never
    /// overlap in a well-formed map; if a malformed map says otherwise, the
    /// first match in `regions` order is returned, deterministically.
    #[must_use]
    pub fn lookup(&self, pos: &WorldPos) -> Option<ShardId> {
        let mut best: Option<(u8, ShardId)> = None;
        for region in &self.regions {
            if !region.contains(pos) {
                continue;
            }
            let depth = region.node.depth;
            if best.is_none_or(|(d, _)| depth > d) {
                best = Some((depth, region.shard));
            }
        }
        best.map(|(_, shard)| shard)
    }

    /// Every region covering `pos`, shallowest first — useful for logging why a
    /// lookup picked what it picked.
    #[must_use]
    pub fn path(&self, pos: &WorldPos) -> Vec<&ShardRegion> {
        let mut hits: Vec<&ShardRegion> = self
            .regions
            .iter()
            .filter(|region| region.contains(pos))
            .collect();
        hits.sort_by_key(|region| region.node.depth);
        hits
    }
}

/// A shard's new ownership of one octree node, part of an [`EpochPrepare`]:
/// `(node, new_owner)`.
pub type EpochChange = (NodeRef, ShardId);

mod changes_wire {
    //! `Vec<(NodeRef, ShardId)>` needs the same `NodeRef` conversion as
    //! [`super::ShardRegion`], applied element-wise.

    use super::EpochChange;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use space_world::NodeRef;

    #[derive(Serialize, Deserialize)]
    struct ChangeWire {
        depth: u8,
        index: [i64; 3],
        new_owner: u32,
    }

    pub fn serialize<S: Serializer>(changes: &[EpochChange], s: S) -> Result<S::Ok, S::Error> {
        let wires: Vec<ChangeWire> = changes
            .iter()
            .map(|(node, owner)| ChangeWire {
                depth: node.depth,
                index: node.index,
                new_owner: owner.0,
            })
            .collect();
        wires.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<EpochChange>, D::Error> {
        let wires = Vec::<ChangeWire>::deserialize(d)?;
        wires
            .into_iter()
            .map(|w| {
                super::validate_node(w.depth, w.index).map_err(serde::de::Error::custom)?;
                Ok((NodeRef::new(w.depth, w.index), super::ShardId(w.new_owner)))
            })
            .collect()
    }
}

/// Announce a new epoch: which nodes change owner, and at which tick the change
/// takes effect.
///
/// Sent by the orchestrator to every shard that participates. Shards stage the
/// change, answer with [`EpochAck`], and switch atomically at
/// `activation_tick` once [`EpochCommit`] arrives.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpochPrepare {
    /// Epoch being prepared.
    pub epoch: u64,
    /// Tick at which the new map becomes active in every shard.
    pub activation_tick: u64,
    /// Nodes that change owner, with their new owner.
    #[serde(with = "changes_wire")]
    pub changes: Vec<EpochChange>,
}

/// A shard has staged [`EpochPrepare`] and will switch at its activation tick.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpochAck {
    /// Epoch being acknowledged.
    pub epoch: u64,
    /// The shard acknowledging.
    pub shard: ShardId,
}

/// Activate the prepared epoch everywhere.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpochCommit {
    /// Epoch now in effect from the activation tick.
    pub epoch: u64,
}

/// Roll a prepared epoch back; every shard keeps its previous map.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpochAbort {
    /// Epoch that was abandoned.
    pub epoch: u64,
    /// Why.
    pub reason: AbortReason,
}

/// Why a handoff or an epoch change was abandoned.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AbortReason {
    /// The receiving shard is at capacity.
    TargetFull,
    /// A phase of the two-phase exchange did not answer in time.
    Timeout,
    /// The shard map moved on underneath the transfer.
    EpochChanged,
    /// A shard is shutting down and is giving its region away.
    ShuttingDown,
    /// Anything else; the string is for logs, never for logic.
    Other(String),
}
