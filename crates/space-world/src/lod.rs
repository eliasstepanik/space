//! 3D-Tiles-style LOD selection over the implicit octree.
//!
//! [`LodSelector`] turns a camera (position, view direction, field of view) and
//! a *candidate tree* into the set of octree nodes to render, using screen-space
//! error (SSE). Every node has a `geometric_error` derived from its size
//! ([`LodSelector::geometric_error`]: `node_size / 2^split_levels`) and its
//! projected error in pixels is
//!
//! ```text
//! sse = geometric_error * screen_height_px / (2 * dist * tan(fov_y / 2))
//! ```
//!
//! where `dist` is the distance from the camera to the closest point of the
//! node's bounding box. Selection uses **REPLACE** refinement: the *deepest*
//! node whose projected error fits the budget (`sse <= max_error_px`) is
//! selected and its whole subtree is dropped.
//!
//! The candidate tree comes from [`LodSelector::candidates`], which walks the
//! octree from the active depth band around the camera, subdividing nodes whose
//! SSE exceeds the budget and culling nodes outside the frustum.
//! [`LodSelector::select`] then walks that candidate list on its own, so the two
//! halves can run on different data (server-generated band, client selection).
//!
//! # Guarantees
//!
//! * **Coverage** — every candidate node is either selected itself or has a
//!   selected descendant, so the selected set tiles the traversed region.
//! * **Bounded** — `select(...).len() <= [MAX_SELECTED]` for any camera and any
//!   candidate list. [`LodSelector::candidates`] never produces more than
//!   `MAX_SELECTED` candidate leaves (its traversal frontier is budgeted, so
//!   the work it does is bounded too), and a REPLACE selection is an antichain
//!   of that tree, which cannot be larger than its leaf set.
//! * **Monotonicity** — for a fixed candidate tree, moving the camera closer can
//!   only keep or deepen the node covering a region: SSE is monotonically
//!   non-increasing in distance, so a node that was refined stays refined.

use std::collections::BinaryHeap;
use std::collections::HashSet;

use crate::octree::{self, NodeRef, MAX_DEPTH};
use crate::pos::{WorldPos, CELL_HALF};

/// Hard upper bound on the number of nodes [`LodSelector::select`] returns.
///
/// [`LodSelector::candidates`] bounds its leaf count to this value, and a
/// REPLACE selection is an antichain of the candidate tree, so the bound holds
/// for every camera position.
pub const MAX_SELECTED: usize = 4096;

/// Tunable parameters of the selector.
///
/// The vertical field of view is *not* stored here: it is passed to
/// [`LodSelector::candidates`] / [`LodSelector::select`] so one selector can be
/// shared by several cameras.
#[derive(Clone, Copy, Debug)]
pub struct LodSelector {
    /// Coarsest depth considered (bottom of the active band).
    pub min_depth: u8,
    /// Finest depth considered (top of the active band, clamped to
    /// [`MAX_DEPTH`]).
    pub max_depth: u8,
    /// Screen-space error budget, in pixels.
    pub max_error_px: f32,
    /// Viewport height in pixels, used by the SSE projection.
    pub screen_height_px: f32,
    /// Viewport aspect ratio (width / height).
    pub aspect: f32,
    /// `geometric_error(depth) = node_size(depth) / 2^split_levels`.
    pub split_levels: u32,
}

impl Default for LodSelector {
    fn default() -> Self {
        Self {
            min_depth: 50,
            max_depth: 60,
            max_error_px: 16.0,
            screen_height_px: 1080.0,
            aspect: 16.0 / 9.0,
            split_levels: 2,
        }
    }
}

impl LodSelector {
    /// A selector for the depth band `min_depth..=max_depth` and the given
    /// pixel-error budget; the remaining parameters are
    /// [`Default::default`].
    #[must_use]
    pub fn new(min_depth: u8, max_depth: u8, max_error_px: f32) -> Self {
        let min_depth = min_depth.min(MAX_DEPTH);
        let max_depth = max_depth.clamp(min_depth, MAX_DEPTH);
        Self {
            min_depth,
            max_depth,
            max_error_px,
            ..Self::default()
        }
    }

    /// Geometric (world-space) error of a node at `depth`, in metres.
    #[must_use]
    pub fn geometric_error(&self, depth: u8) -> f64 {
        // `split_levels` is a public field: clamp it so the shift cannot
        // overflow (2^62 already makes the error vanish).
        NodeRef::size_m(depth) / (1u64 << self.split_levels.min(62)) as f64
    }

    /// Projected screen-space error of `node`, in pixels.
    ///
    /// `fov_y` is the full vertical field of view in radians.
    #[must_use]
    pub fn screen_space_error(&self, node: NodeRef, camera: WorldPos, fov_y: f32) -> f32 {
        self.sse(node, camera, fov_y) as f32
    }

    fn sse(&self, node: NodeRef, camera: WorldPos, fov_y: f32) -> f64 {
        self.sse_in(&self.view(camera, [0.0; 3], fov_y), node)
    }

    /// The per-camera constants of a traversal.
    fn view(&self, camera: WorldPos, dir: [f32; 3], fov_y: f32) -> View {
        let tan_y = (f64::from(fov_y) * 0.5).tan().max(1e-9);
        let dir = dir.map(f64::from);
        let dir_len = norm(dir);
        // A degenerate direction never culls.
        let basis = (dir_len != 0.0).then(|| {
            let f = [dir[0] / dir_len, dir[1] / dir_len, dir[2] / dir_len];
            // Orthonormal camera basis, built from the axis least aligned with f.
            let helper = if f[0].abs() <= f[1].abs() && f[0].abs() <= f[2].abs() {
                [1.0, 0.0, 0.0]
            } else if f[1].abs() <= f[2].abs() {
                [0.0, 1.0, 0.0]
            } else {
                [0.0, 0.0, 1.0]
            };
            let r = cross(helper, f);
            let r_len = norm(r);
            let r = [r[0] / r_len, r[1] / r_len, r[2] / r_len];
            (f, r, cross(f, r))
        });
        View {
            camera,
            basis,
            tan_y,
            tan_x: tan_y * f64::from(self.aspect),
            screen_height_px: f64::from(self.screen_height_px),
        }
    }

    /// Projected screen-space error of `node` for `view`, in pixels.
    fn sse_in(&self, view: &View, node: NodeRef) -> f64 {
        let dist = self.distance_to_node(node, view.camera);
        let denom = 2.0 * dist * view.tan_y;
        self.geometric_error(node.depth) * view.screen_height_px / denom.max(1e-6)
    }

    /// Distance from `camera` to the closest point of `node`'s bounding box,
    /// floored at 1 mm so a camera inside a node still refines.
    #[must_use]
    pub fn distance_to(&self, node: NodeRef, camera: WorldPos) -> f64 {
        self.distance_to_node(node, camera)
    }

    fn distance_to_node(&self, node: NodeRef, camera: WorldPos) -> f64 {
        // The node's lower corner (what `octree::node_bounds` returns as `min`).
        let min = WorldPos {
            cell: node.lower_cell(),
            local: [-CELL_HALF; 3],
        };
        let size = NodeRef::size_m(node.depth);
        let d = WorldPos::delta(min, camera);
        let mut sum = 0.0f64;
        for &axis_delta in &d {
            let clamped = axis_delta.clamp(0.0, size);
            let outside = axis_delta - clamped;
            sum += outside * outside;
        }
        sum.sqrt().max(1e-3)
    }

    /// Conservative frustum test for `node`'s bounding sphere against the
    /// infinite-far-plane view frustum.
    ///
    /// Errs towards keeping nodes, so coverage is never lost to rounding.
    fn in_frustum(view: &View, node: NodeRef) -> bool {
        let Some((f, r, u)) = view.basis else {
            return true;
        };
        let c = WorldPos::delta(view.camera, node.center());
        let cz = dot(c, f);
        let radius = NodeRef::size_m(node.depth) * (3.0f64).sqrt() * 0.5;
        if cz + radius < 0.0 {
            return false; // entirely behind the camera
        }
        let reach = cz + radius;
        let cx = dot(c, r).abs();
        let cy = dot(c, u).abs();
        cx <= reach * view.tan_x + radius && cy <= reach * view.tan_y + radius
    }
}

/// The per-camera constants of a traversal, computed once instead of per node.
struct View {
    camera: WorldPos,
    /// Orthonormal (forward, right, up); `None` for a zero view direction,
    /// which never culls.
    basis: Option<([f64; 3], [f64; 3], [f64; 3])>,
    /// `tan(fov_y / 2)`, floored away from zero.
    tan_y: f64,
    tan_x: f64,
    screen_height_px: f64,
}

/// Cross product.
const fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product.
const fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Euclidean norm.
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// One entry of the traversal priority queue, ordered by descending SSE.
///
/// `sse_bits` is the IEEE-754 bit pattern of a non-negative `f64`; for
/// non-negative floats the bit order equals the value order, which makes the
/// comparison total (no `NaN` can enter: [`LodSelector::sse`] floors its
/// denominator).
#[derive(Clone, Copy, Debug)]
struct QueueItem {
    sse_bits: u64,
    node: NodeRef,
}

impl PartialEq for QueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.sse_bits == other.sse_bits && self.node == other.node
    }
}

impl Eq for QueueItem {}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sse_bits
            .cmp(&other.sse_bits)
            .then_with(|| self.node.index.cmp(&other.node.index))
            .then_with(|| other.node.depth.cmp(&self.node.depth))
    }
}

impl LodSelector {
    /// Generate the candidate tree for a camera: every node the traversal
    /// reaches, internal nodes *and* leaves, as a flat list.
    ///
    /// The traversal is best-first on screen-space error (nearest / coarsest
    /// first), so the node budget is spent where the error is worst. A node is
    /// a leaf (not subdivided) when its SSE fits the budget, it is at
    /// `max_depth`, it is frustum-culled, or subdividing it would push the
    /// frontier past the leaf budget [`MAX_SELECTED`]. The list is a *proper*
    /// octree fragment — an internal node is present together with the
    /// children the traversal kept — so [`LodSelector::select`] can walk it.
    ///
    /// The budget counts the whole frontier — the leaves already emitted plus
    /// every node still queued, each of which ends up as at least one leaf —
    /// so the number of leaves never exceeds [`MAX_SELECTED`] *and* the
    /// traversal stops subdividing as soon as that many are in sight. (Counting
    /// only emitted leaves let it subdivide every over-budget node of a deep
    /// band first: millions of nodes for a camera looking into open space.)
    #[must_use]
    pub fn candidates(&self, camera: WorldPos, dir: [f32; 3], fov_y: f32) -> Vec<NodeRef> {
        self.candidates_from(octree::node_at(camera, self.min_depth), camera, dir, fov_y)
    }

    /// [`LodSelector::candidates`] rooted at an explicit node (used by servers
    /// that generate a band for a region rather than around the camera).
    #[must_use]
    pub fn candidates_from(
        &self,
        root: NodeRef,
        camera: WorldPos,
        dir: [f32; 3],
        fov_y: f32,
    ) -> Vec<NodeRef> {
        let mut out = Vec::new();
        self.traverse(root, camera, dir, fov_y, |node, _selected| out.push(node));
        out
    }

    /// The selection for the camera the traversal runs for, in one pass:
    /// exactly `select(&candidates_from(root, camera, dir, fov_y), camera,
    /// fov_y)`, without materialising the candidate tree and walking it a
    /// second time. This is what a renderer that builds and selects for the
    /// same camera every frame should call.
    #[must_use]
    pub fn select_from(
        &self,
        root: NodeRef,
        camera: WorldPos,
        dir: [f32; 3],
        fov_y: f32,
    ) -> Vec<NodeRef> {
        let mut selected = Vec::new();
        self.traverse(root, camera, dir, fov_y, |node, is_selected| {
            if is_selected {
                selected.push(node);
            }
        });
        selected.sort_unstable_by_key(|n| (n.depth, n.index));
        selected
    }

    /// The budgeted best-first traversal behind [`LodSelector::candidates_from`]
    /// and [`LodSelector::select_from`]. `visit(node, selected)` is called once
    /// per traversed node; `selected` is true for the nodes a REPLACE selection
    /// with the same camera picks: the leaves, and any subdivided node all of
    /// whose children were frustum-culled.
    fn traverse(
        &self,
        root: NodeRef,
        camera: WorldPos,
        dir: [f32; 3],
        fov_y: f32,
        mut visit: impl FnMut(NodeRef, bool),
    ) {
        let view = self.view(camera, dir, fov_y);
        let budget = f64::from(self.max_error_px);
        let mut leaves = 0usize;
        let mut heap: BinaryHeap<QueueItem> = BinaryHeap::new();
        if Self::in_frustum(&view, root) {
            heap.push(QueueItem {
                sse_bits: self.sse_in(&view, root).to_bits(),
                node: root,
            });
        }
        while let Some(item) = heap.pop() {
            // Refine only while the frontier (emitted leaves + queued nodes +
            // this node's up-to-8 children) fits the leaf budget; otherwise
            // this node becomes a (coarse) leaf. Invariant: `leaves +
            // heap.len() <= MAX_SELECTED`.
            let refine = item.node.depth < self.max_depth
                && item.sse_bits > budget.to_bits()
                && leaves + heap.len() + 8 <= MAX_SELECTED;
            let kids = if refine {
                octree::children(item.node)
            } else {
                None
            };
            match kids {
                Some(kids) => {
                    let mut kept = 0;
                    for kid in kids {
                        if !Self::in_frustum(&view, kid) {
                            continue;
                        }
                        kept += 1;
                        heap.push(QueueItem {
                            sse_bits: self.sse_in(&view, kid).to_bits(),
                            node: kid,
                        });
                    }
                    // With every child culled nothing below covers this node:
                    // it is a leaf of the candidate tree after all (and counts
                    // against the budget like one), and a selection takes it.
                    if kept == 0 {
                        leaves += 1;
                    }
                    visit(item.node, kept == 0);
                }
                None => {
                    visit(item.node, true);
                    leaves += 1;
                }
            }
        }
    }

    /// REPLACE refinement over a candidate tree: return the deepest nodes whose
    /// projected error fits the budget, one per region (an antichain: no
    /// selected node contains another).
    ///
    /// No frustum test is applied here — the candidate list is already culled —
    /// which keeps the selection a pure function of the candidate tree and the
    /// camera distance, and therefore monotone.
    ///
    /// Guarantees: `result.len() <= MAX_SELECTED`, and every candidate leaf is
    /// either selected itself or has a selected ancestor (coverage). A node
    /// whose children would not fit the budget any more is selected coarsely
    /// instead of being refined, so coverage is never lost. (A candidate tree
    /// from [`LodSelector::candidates`] has at most `MAX_SELECTED` leaves, so
    /// for it that cutoff never triggers.)
    #[must_use]
    pub fn select(&self, candidates: &[NodeRef], camera: WorldPos, fov_y: f32) -> Vec<NodeRef> {
        let set: HashSet<NodeRef> = candidates.iter().copied().collect();
        if set.is_empty() {
            return Vec::new();
        }
        let root = candidates_root(candidates);
        let budget = f64::from(self.max_error_px);
        let view = self.view(camera, [0.0; 3], fov_y);
        let mut selected = Vec::new();
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            // The children present in the candidate tree, without allocating.
            let mut kids = [node; 8];
            let mut kid_count = 0;
            for kid in octree::children(node).into_iter().flatten() {
                if set.contains(&kid) {
                    kids[kid_count] = kid;
                    kid_count += 1;
                }
            }
            // Everything selected or still on the stack ends up as at least
            // one selected node, so refining is only allowed while that total
            // stays within the bound. Invariant: `selected.len() + stack.len()
            // <= MAX_SELECTED`.
            let fits = selected.len() + stack.len() + kid_count <= MAX_SELECTED;
            let must_select = kid_count == 0
                || !fits
                || node.depth >= self.max_depth
                || self.sse_in(&view, node) <= budget;
            if must_select {
                selected.push(node);
            } else {
                stack.extend_from_slice(&kids[..kid_count]);
            }
        }
        selected.sort_unstable_by_key(|n| (n.depth, n.index));
        selected
    }
}

/// The shallowest node containing every candidate (the traversal root).
fn candidates_root(candidates: &[NodeRef]) -> NodeRef {
    let mut root = candidates[0];
    for &c in candidates {
        root = common_ancestor(root, c);
    }
    root
}

/// The shallowest node containing both `a` and `b`.
fn common_ancestor(a: NodeRef, b: NodeRef) -> NodeRef {
    let (mut x, mut y) = (a, b);
    while x.depth > y.depth {
        x = octree::parent(x).unwrap_or(x);
    }
    while y.depth > x.depth {
        y = octree::parent(y).unwrap_or(y);
    }
    while x != y {
        x = octree::parent(x).unwrap_or(x);
        y = octree::parent(y).unwrap_or(y);
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    const FOV: f32 = std::f32::consts::FRAC_PI_3;

    /// Cameras around the origin of a deep band, looking various ways: the
    /// configuration the client renders the galaxy with.
    fn cameras() -> Vec<(WorldPos, [f32; 3])> {
        let at = |m: [f64; 3]| WorldPos::from_f64_metres_around(WorldPos::ZERO, m);
        vec![
            (at([0.0, 0.0, -5.0e4]), [0.0, 0.0, 1.0]),
            (at([1.0e3, 2.0e2, -4.0e4]), [0.2, 0.0, 0.98]),
            (at([3.0e3, 1.0e2, 2.0e3]), [0.0, 0.0, 1.0]),
            (at([-7.0e5, 3.0e4, 9.0e5]), [0.6, -0.1, -0.8]),
            (at([12.0, -40.0, 7.0]), [0.0, 1.0, 0.0]),
        ]
    }

    /// The eight depth-`depth` nodes meeting at the world origin.
    fn roots(depth: u8) -> impl Iterator<Item = NodeRef> {
        (0..8i64).map(move |o| NodeRef::new(depth, [-(o & 1), -((o >> 1) & 1), -((o >> 2) & 1)]))
    }

    /// Looking into open space through a deep band must not walk the whole
    /// band: the leaf budget bounds the frontier, so the candidate list stays
    /// within a small multiple of `MAX_SELECTED`. (It used to be ~800 000
    /// nodes per root for the first camera here.)
    #[test]
    fn deep_band_traversal_is_bounded_by_the_leaf_budget() {
        let sel = LodSelector::new(30, 63, 8.0);
        for (camera, dir) in cameras() {
            for root in roots(sel.min_depth) {
                let candidates = sel.candidates_from(root, camera, dir, FOV);
                let set: HashSet<NodeRef> = candidates.iter().copied().collect();
                assert_eq!(set.len(), candidates.len(), "no node is visited twice");
                let leaves = candidates
                    .iter()
                    .filter(|n| {
                        octree::children(**n)
                            .is_none_or(|kids| kids.iter().all(|k| !set.contains(k)))
                    })
                    .count();
                assert!(leaves <= MAX_SELECTED, "{leaves} leaves");
                // Every refinement that keeps a child adds at least one leaf
                // per level of the band at worst; in practice ~8/7 nodes per
                // leaf. Two per leaf is a comfortable, still tight ceiling.
                assert!(
                    candidates.len() <= 2 * MAX_SELECTED,
                    "{} candidates for {root:?}",
                    candidates.len()
                );
            }
        }
    }

    /// `select_from` is `select` over `candidates_from`, node for node.
    #[test]
    fn select_from_matches_select_over_the_candidates() {
        for sel in [
            LodSelector::new(30, 63, 8.0),
            LodSelector::new(54, 58, 32.0),
        ] {
            for (camera, dir) in cameras() {
                for root in roots(sel.min_depth) {
                    let candidates = sel.candidates_from(root, camera, dir, FOV);
                    let two_pass = sel.select(&candidates, camera, FOV);
                    let one_pass = sel.select_from(root, camera, dir, FOV);
                    assert_eq!(one_pass, two_pass, "{root:?} from {camera:?}");
                    assert!(one_pass.len() <= MAX_SELECTED);
                }
            }
        }
    }

    /// `select` stays within `MAX_SELECTED` for *any* candidate list, not only
    /// one produced by `candidates`: here a complete 5-level tree (32 768
    /// leaves) with a camera close enough that every node wants refining.
    #[test]
    fn select_is_bounded_for_an_oversized_candidate_tree() {
        let sel = LodSelector::new(50, 55, 0.001);
        let root = NodeRef::new(50, [0, 0, 0]);
        let mut tree = vec![root];
        let mut level = vec![root];
        for _ in 0..5 {
            level = level
                .iter()
                .flat_map(|n| octree::children(*n).into_iter().flatten())
                .collect();
            tree.extend_from_slice(&level);
        }
        assert_eq!(level.len(), 32_768);
        let camera = root.center();
        let selected = sel.select(&tree, camera, FOV);
        assert!(selected.len() <= MAX_SELECTED, "{}", selected.len());
        // Coverage: every leaf is selected or has a selected ancestor.
        let chosen: HashSet<NodeRef> = selected.iter().copied().collect();
        for leaf in &level {
            let mut node = *leaf;
            let mut covered = chosen.contains(&node);
            while !covered {
                match octree::parent(node) {
                    Some(parent) if parent.depth >= root.depth => {
                        node = parent;
                        covered = chosen.contains(&node);
                    }
                    _ => break,
                }
            }
            assert!(covered, "{leaf:?} lost its cover");
        }
    }

    /// A huge public `split_levels` must not overflow the error shift.
    #[test]
    fn geometric_error_tolerates_any_split_level() {
        let mut sel = LodSelector::new(50, 60, 16.0);
        sel.split_levels = 40;
        assert!(sel.geometric_error(50) > 0.0);
        sel.split_levels = u32::MAX;
        assert!(sel.geometric_error(50) > 0.0);
    }
}
