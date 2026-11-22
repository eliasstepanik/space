//! Cell coordinates: the integer half of a [`crate::WorldPos`].
//!
//! [`CellCoord`] is a thin newtype over `[i64; 3]` whose only real job is to
//! map a cell into the implicit octree of [`crate::octree`].

use crate::octree::NodeRef;

/// A world cell coordinate (the `cell` half of [`crate::WorldPos`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct CellCoord(pub [i64; 3]);

impl CellCoord {
    /// The cell containing a world position.
    #[must_use]
    pub fn from_pos(pos: crate::WorldPos) -> Self {
        Self(pos.cell)
    }

    /// The cell's integer coordinates.
    #[must_use]
    pub fn as_array(self) -> [i64; 3] {
        self.0
    }

    /// The octree node of the given depth that contains this cell.
    ///
    /// See [`crate::octree`]: a node at depth `d` spans `2^(MAX_DEPTH - d)`
    /// cells per axis, so the root (depth 0) spans the whole `i64` cell space.
    #[must_use]
    pub fn to_octree_node(self, depth: u8) -> NodeRef {
        NodeRef::containing_cell(depth, self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::CellCoord;
    use crate::octree::MAX_DEPTH;
    use crate::WorldPos;

    #[test]
    fn to_octree_node_matches_node_at() {
        let pos = WorldPos::from_metres([12_345_678_901, -987_654_321, 5]);
        let cell = CellCoord::from_pos(pos);
        for depth in [0u8, 1, 7, 20, 40, MAX_DEPTH] {
            assert_eq!(
                cell.to_octree_node(depth),
                crate::octree::node_at(pos, depth),
                "depth {depth}"
            );
        }
    }

    #[test]
    fn root_node_is_shared_by_the_whole_space() {
        for c in [CellCoord([0, 0, 0]), CellCoord([i64::MIN, i64::MAX, -1])] {
            assert_eq!(c.to_octree_node(0).index, [0, 0, 0]);
            assert_eq!(c.to_octree_node(0).depth, 0);
        }
    }
}
