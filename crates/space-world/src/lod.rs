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
//! * **Bounded** — `select(...).len() <= [MAX_SELECTED]` for any camera.
//!   [`LodSelector::candidates`] never produces more than `MAX_SELECTED`
//!   candidate leaves, and a REPLACE selection is an antichain of that tree,
//!   which cannot be larger than its leaf set.
//! * **Monotonicity** — for a fixed candidate tree, moving the camera closer can
//!   only keep or deepen the node covering a region: SSE is monotonically
//!   non-increasing in distance, so a node that was refined stays refined.

use std::collections::BinaryHeap;
use std::collections::HashSet;

use crate::octree::{self, NodeRef, MAX_DEPTH};
use crate::pos::WorldPos;

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
        NodeRef::size_m(depth) / f64::from(1u32 << self.split_levels.min(62))
    }

    /// Projected screen-space error of `node`, in pixels.
    ///
    /// `fov_y` is the full vertical field of view in radians.
    #[must_use]
    pub fn screen_space_error(&self, node: NodeRef, camera: WorldPos, fov_y: f32) -> f32 {
        self.sse(node, camera, fov_y) as f32
    }

    fn sse(&self, node: NodeRef, camera: WorldPos, fov_y: f32) -> f64 {
        let dist = self.distance_to_node(node, camera);
        let half_fov = f64::from(fov_y) * 0.5;
        let denom = 2.0 * dist * half_fov.tan().max(1e-9);
        self.geometric_error(node.depth) * f64::from(self.screen_height_px) / denom.max(1e-6)
    }

    /// Distance from `camera` to the closest point of `node`'s bounding box,
    /// floored at 1 mm so a camera inside a node still refines.
    #[must_use]
    pub fn distance_to(&self, node: NodeRef, camera: WorldPos) -> f64 {
        self.distance_to_node(node, camera)
    }

    fn distance_to_node(&self, node: NodeRef, camera: WorldPos) -> f64 {
        let (min, _max) = octree::node_bounds(node);
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
    fn frustum_test(&self, node: NodeRef, camera: WorldPos, dir: [f32; 3], fov_y: f32) -> bool {
        let dir_len = (f64::from(dir[0]) * f64::from(dir[0])
            + f64::from(dir[1]) * f64::from(dir[1])
            + f64::from(dir[2]) * f64::from(dir[2]))
        .sqrt();
        if dir_len == 0.0 {
            return true; // degenerate direction: never cull
        }
        let f = [
            f64::from(dir[0]) / dir_len,
            f64::from(dir[1]) / dir_len,
            f64::from(dir[2]) / dir_len,
        ];
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
        let u = cross(f, r);
        let c = WorldPos::delta(camera, node.center());
        let cz = dot(c, f);
        let radius = NodeRef::size_m(node.depth) * (3.0f64).sqrt() * 0.5;
        if cz + radius < 0.0 {
            return false; // entirely behind the camera
        }
        let tan_y = (f64::from(fov_y) * 0.5).tan().max(1e-9);
        let tan_x = tan_y * f64::from(self.aspect);
        let reach = cz + radius;
        let cx = dot(c, r).abs();
        let cy = dot(c, u).abs();
        cx <= reach * tan_x + radius && cy <= reach * tan_y + radius
    }
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
    /// `max_depth`, it is frustum-culled, or the leaf budget [`MAX_SELECTED`] is
    /// reached. The list is a *proper* octree fragment — an internal node is
    /// present together with the children the traversal kept — so
    /// [`LodSelector::select`] can walk it.
    ///
    /// The number of leaves never exceeds [`MAX_SELECTED`] (plus the at-most-7
    /// overshoot of the final subdivision), which is what lets
    /// [`LodSelector::select`] stay within its bound without dropping coverage.
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
        let budget = f64::from(self.max_error_px);
        let mut out: Vec<NodeRef> = Vec::new();
        let mut leaves = 0usize;
        let mut heap: BinaryHeap<QueueItem> = BinaryHeap::new();
        if self.frustum_test(root, camera, dir, fov_y) {
            heap.push(QueueItem {
                sse_bits: self.sse(root, camera, fov_y).to_bits(),
                node: root,
            });
        }
        while let Some(item) = heap.pop() {
            // Refine only while there is leaf budget left; otherwise this node
            // becomes a (coarse) leaf, which keeps the leaf count bounded.
            let refine = item.node.depth < self.max_depth
                && item.sse_bits > budget.to_bits()
                && leaves < MAX_SELECTED;
            let kids = if refine {
                octree::children(item.node)
            } else {
                None
            };
            match kids {
                Some(kids) => {
                    out.push(item.node);
                    for kid in kids {
                        if !self.frustum_test(kid, camera, dir, fov_y) {
                            continue;
                        }
                        heap.push(QueueItem {
                            sse_bits: self.sse(kid, camera, fov_y).to_bits(),
                            node: kid,
                        });
                    }
                }
                None => {
                    out.push(item.node);
                    leaves += 1;
                }
            }
        }
        out
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
    /// either selected itself or has a selected ancestor (coverage). When the
    /// budget is reached the traversal stops refining and selects the current
    /// node coarsely instead of dropping it, so coverage is never lost.
    #[must_use]
    pub fn select(&self, candidates: &[NodeRef], camera: WorldPos, fov_y: f32) -> Vec<NodeRef> {
        let set: HashSet<NodeRef> = candidates.iter().copied().collect();
        if set.is_empty() {
            return Vec::new();
        }
        let root = candidates_root(candidates);
        let budget = f64::from(self.max_error_px);
        let mut selected = Vec::new();
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            let kids: Vec<NodeRef> = if selected.len() >= MAX_SELECTED {
                // Out of budget: select this node coarsely (coverage preserved).
                Vec::new()
            } else {
                match octree::children(node) {
                    Some(k) => k.into_iter().filter(|k| set.contains(k)).collect(),
                    None => Vec::new(),
                }
            };
            let must_select = kids.is_empty()
                || node.depth >= self.max_depth
                || self.sse(node, camera, fov_y) <= budget;
            if must_select {
                selected.push(node);
            } else {
                stack.extend(kids);
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
