//! An **implicit** octree over the infinite cell space.
//!
//! No tree is stored: a [`NodeRef`] is just `(depth, index)` and every query
//! ([`node_bounds`], [`children`], [`parent`], [`node_at`], [`contains`]) is
//! computed from the address in O(1). This is what makes the structure usable
//! for streaming: nodes are generated on demand and are stable across
//! processes because the addressing is pure.
//!
//! Addressing (per axis, all three axes independent):
//!
//! * depth `0` (the root) spans the whole `i64` cell space, `2^64` cells;
//! * a node at depth `d` spans `2^(64 - d)` cells per axis, so depth
//!   [`MAX_DEPTH`] = 64 addresses a single cell;
//! * `index[axis] = cell[axis] >> (64 - depth)` (arithmetic shift, i.e. floor
//!   division, so negative cells address correctly).
//!
//! The edge length of a node at depth `d` is [`CELL_SIZE`] × 2^(64 − d) metres.

use crate::pos::{WorldPos, CELL_HALF, CELL_SIZE};

/// Maximum octree depth: a node at this depth is exactly one world cell.
pub const MAX_DEPTH: u8 = 64;

/// A node of the implicit octree.
///
/// `depth` is `0..=MAX_DEPTH`; `index` is the node's integer coordinate at that
/// depth (see the [module docs](self)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct NodeRef {
    /// Distance from the root, `0..=MAX_DEPTH`.
    pub depth: u8,
    /// Node coordinate at `depth`, per axis.
    pub index: [i64; 3],
}

impl NodeRef {
    /// The root node (the whole cell space).
    pub const ROOT: Self = Self {
        depth: 0,
        index: [0; 3],
    };

    /// Build a node reference from a depth and an index without validation.
    #[must_use]
    pub const fn new(depth: u8, index: [i64; 3]) -> Self {
        Self { depth, index }
    }

    /// The node of the given depth containing `cell`.
    #[must_use]
    pub const fn containing_cell(depth: u8, cell: [i64; 3]) -> Self {
        let depth = if depth > MAX_DEPTH { MAX_DEPTH } else { depth };
        if depth == 0 {
            return Self {
                depth: 0,
                index: [0; 3],
            };
        }
        let shift = 64u32 - depth as u32;
        Self {
            depth,
            index: [cell[0] >> shift, cell[1] >> shift, cell[2] >> shift],
        }
    }

    /// Number of cells spanned per axis by nodes of this depth.
    ///
    /// Exact in `f64` for every depth (`2^64` at the root).
    #[must_use]
    pub const fn size_cells(depth: u8) -> f64 {
        let depth = if depth > MAX_DEPTH { MAX_DEPTH } else { depth };
        (1u128 << (64 - depth as u32)) as f64
    }

    /// Edge length of a node at `depth`, in metres.
    #[must_use]
    pub const fn size_m(depth: u8) -> f64 {
        CELL_SIZE * Self::size_cells(depth)
    }

    /// The lowest cell belonging to this node, per axis.
    #[must_use]
    pub const fn lower_cell(&self) -> [i64; 3] {
        if self.depth == 0 {
            return [i64::MIN; 3];
        }
        let shift = 64u32 - self.depth as u32;
        [
            self.index[0].wrapping_shl(shift),
            self.index[1].wrapping_shl(shift),
            self.index[2].wrapping_shl(shift),
        ]
    }

    /// The centre of this node, as a world position.
    #[must_use]
    pub fn center(&self) -> WorldPos {
        if self.depth == 0 {
            return WorldPos::ZERO;
        }
        let lower = self.lower_cell();
        let shift = 64u32 - u32::from(self.depth);
        // Half the node width in cells; 0 for a single-cell node.
        let half = if shift == 0 { 0 } else { 1i64 << (shift - 1) };
        WorldPos {
            cell: [
                lower[0].wrapping_add(half),
                lower[1].wrapping_add(half),
                lower[2].wrapping_add(half),
            ],
            local: [0.0; 3],
        }
    }
}

/// The geometric bounds of a node: `(min, max)`.
///
/// `min` is the inclusive lower corner (lowest cell, local `-CELL_HALF`).
/// `max` is the **exclusive** upper corner, expressed as the next cell with
/// local `-CELL_HALF`, so `min <= p < max` (cell-wise) describes containment.
///
/// At the root the upper corner wraps onto the lower corner: the root spans
/// exactly the whole representable space (`2^64` cells), so no distinct corner
/// exists. Use [`contains`] for exact membership.
#[must_use]
pub fn node_bounds(node: NodeRef) -> (WorldPos, WorldPos) {
    let lower = node.lower_cell();
    let min = WorldPos {
        cell: lower,
        local: [-CELL_HALF; 3],
    };
    if node.depth == 0 {
        // The root spans exactly the representable space, so its upper corner
        // wraps onto its lower corner.
        return (min, min);
    }
    let shift = 64u32 - u32::from(node.depth);
    let mut upper_cell = lower;
    for axis in 0..3 {
        // Last cell inside the node (wrapping, so a node whose upper edge falls
        // outside the representable cell space stays well-defined).
        upper_cell[axis] = lower[axis]
            .wrapping_add(1i64.wrapping_shl(shift))
            .wrapping_sub(1);
    }
    // Normalising `+CELL_HALF` carries into the next cell: the exclusive edge.
    let max = WorldPos::new(upper_cell, [CELL_HALF; 3]);
    (min, max)
}

/// Exact membership test: does `node` contain `pos`?
#[must_use]
pub fn contains(node: NodeRef, pos: WorldPos) -> bool {
    NodeRef::containing_cell(node.depth, pos.cell) == node
}

/// The node of the given depth containing `pos`.
#[must_use]
pub fn node_at(pos: WorldPos, depth: u8) -> NodeRef {
    NodeRef::containing_cell(depth, pos.cell)
}

/// The eight children of `node`, ordered by the usual octant bits
/// (`bit 0 = +x`, `bit 1 = +y`, `bit 2 = +z`, relative to the node centre).
///
/// Returns `None` at [`MAX_DEPTH`], where nodes are single cells.
#[must_use]
pub fn children(node: NodeRef) -> Option<[NodeRef; 8]> {
    if node.depth >= MAX_DEPTH {
        return None;
    }
    let depth = node.depth + 1;
    // Children of index `i` are `2i` and `2i + 1` per axis, which inverts the
    // `index = cell >> shift` addressing. The root is the one exception: its
    // index is pinned to `0` (there is exactly one root), while its children
    // are the depth-1 nodes with indices `-1` and `0`.
    let base = if node.depth == 0 {
        [-1i64, -1, -1]
    } else {
        [
            node.index[0].wrapping_mul(2),
            node.index[1].wrapping_mul(2),
            node.index[2].wrapping_mul(2),
        ]
    };
    let mut out = [NodeRef::default(); 8];
    for (octant, slot) in out.iter_mut().enumerate() {
        let index = [
            base[0] + i64::from((octant & 1) as u8),
            base[1] + i64::from(((octant >> 1) & 1) as u8),
            base[2] + i64::from(((octant >> 2) & 1) as u8),
        ];
        *slot = NodeRef { depth, index };
    }
    Some(out)
}

/// The parent of `node`, or `None` for the root.
#[must_use]
pub fn parent(node: NodeRef) -> Option<NodeRef> {
    if node.depth == 0 {
        return None;
    }
    if node.depth == 1 {
        return Some(NodeRef::ROOT);
    }
    Some(NodeRef {
        depth: node.depth - 1,
        index: [node.index[0] >> 1, node.index[1] >> 1, node.index[2] >> 1],
    })
}

/// The implicit octree: a stateless namespace over the free functions of this
/// module.
///
/// Nothing is stored — every operation is computed from a [`NodeRef`] address,
/// which is what makes the tree usable for deterministic streaming. The
/// associated functions simply forward to [`node_bounds`], [`children`],
/// [`parent`] and [`node_at`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Octree;

impl Octree {
    /// The root node (the whole cell space).
    pub const ROOT: NodeRef = NodeRef::ROOT;

    /// Forward to [`node_bounds`].
    #[must_use]
    pub fn node_bounds(node: NodeRef) -> (WorldPos, WorldPos) {
        node_bounds(node)
    }

    /// Forward to [`children`].
    #[must_use]
    pub fn children(node: NodeRef) -> Option<[NodeRef; 8]> {
        children(node)
    }

    /// Forward to [`parent`].
    #[must_use]
    pub fn parent(node: NodeRef) -> Option<NodeRef> {
        parent(node)
    }

    /// Forward to [`node_at`].
    #[must_use]
    pub fn node_at(pos: WorldPos, depth: u8) -> NodeRef {
        node_at(pos, depth)
    }

    /// Forward to [`contains`].
    #[must_use]
    pub fn contains(node: NodeRef, pos: WorldPos) -> bool {
        contains(node, pos)
    }
}

#[cfg(test)]
mod tests {
    use super::{children, contains, node_at, node_bounds, parent, NodeRef, MAX_DEPTH};
    use crate::WorldPos;

    #[test]
    fn bounds_span_the_node_size() {
        // Only depths whose node fits inside the representable cell space near
        // the origin (coarser nodes wrap the i64 cell space at their upper
        // edge, which `node_bounds` documents).
        for depth in 20..=63u8 {
            // Build the node from a real cell so the index is valid at depth.
            let node = NodeRef::containing_cell(depth, [3, -2, 0]);
            let (min, max) = node_bounds(node);
            let span = WorldPos::delta(min, max);
            let expected = NodeRef::size_m(depth);
            for (axis, &s) in span.iter().enumerate() {
                assert!(
                    (s - expected).abs() <= 1e-6 * expected,
                    "depth {depth} axis {axis}: {s} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn parent_child_round_trip() {
        // Built from a real position so every index is valid at its depth:
        // at depth `d` the per-axis index range is -(2^(d-1)) ..= 2^(d-1) - 1.
        let pos = WorldPos::from_metres([-9_876_543_210_987, 12, 4_000_000_000]);
        for depth in 1..=12u8 {
            let node = node_at(pos, depth);
            let p = parent(node).expect("non-root has a parent");
            assert_eq!(p.depth, depth - 1);
            assert!(children(p).unwrap().contains(&node), "depth {depth}");
        }
        assert!(parent(NodeRef::ROOT).is_none());
        assert!(children(NodeRef::new(MAX_DEPTH, [0, 0, 0])).is_none());
    }

    #[test]
    fn children_partition_parent() {
        let node = NodeRef::new(6, [1, -1, 2]);
        let (pmin, _pmax) = node_bounds(node);
        for kid in children(node).unwrap() {
            assert_eq!(parent(kid), Some(node));
            let (kmin, kmax) = node_bounds(kid);
            assert!(contains(node, kmin));
            assert!(contains(
                node,
                WorldPos::from_f64_metres_around(kmax, [-1.0, -1.0, -1.0])
            ));
            assert!(WorldPos::delta(pmin, kmin)[0] >= 0.0);
        }
    }

    #[test]
    fn node_at_matches_containing_cell() {
        let pos = WorldPos::from_metres([-9_876_543_210_987, 12, 4_000_000_000]);
        for depth in 0..=MAX_DEPTH {
            let n = node_at(pos, depth);
            assert_eq!(n.depth, depth);
            assert!(contains(n, pos), "depth {depth}");
            assert_eq!(n, NodeRef::containing_cell(depth, pos.cell));
        }
    }
}
