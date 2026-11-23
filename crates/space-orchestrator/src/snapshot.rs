//! [`Snapshot`]: everything the orchestrator knows, as plain data.
//!
//! The run loop builds one of these ([`crate::orchestrator::Orchestrator::snapshot`])
//! and hands it to whoever wants to *look* at the cluster — the web UI renders
//! it as a page and as JSON. A snapshot is a copy: holding one never borrows
//! from the orchestrator, so a slow reader cannot delay the control loop, and
//! nothing in it can be used to change anything.
//!
//! Every time in a snapshot is an orchestrator **tick** (the loop runs at
//! [`crate::config::TICK_HZ`]); `tick` is "now" on the same clock.
//!
//! The shared peer token is deliberately **not** part of a snapshot.

use std::net::SocketAddr;
use std::path::PathBuf;

use space_protocol::shard::ShardId;
use space_protocol::{Scene, SceneState};
use space_world::octree;
use space_world::NodeRef;

use crate::autoscale::ShardScaleState;
use crate::config::LauncherKind;
use crate::events::Event;

/// The orchestrator's state at one tick.
#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot {
    /// The orchestrator's clock when the snapshot was taken.
    pub tick: u64,
    /// The cluster id shards must name in their hello.
    pub cluster: String,
    /// Where the control endpoint was asked to listen.
    pub bind: SocketAddr,
    /// The control address launched shards are told to dial.
    pub control_addr: SocketAddr,
    /// Which launcher starts the shards a split creates.
    pub launcher: LauncherKind,
    /// The file the shard-id counter is persisted in.
    pub state_path: PathBuf,
    /// The declared root region.
    pub root: NodeRef,
    /// The first id handed out when no counter file existed.
    pub first_shard_id: u64,
    /// The host UDP port range reserved for launched shards.
    pub host_port_range: (u16, u16),
    /// The id the next allocation returns.
    pub next_shard_id: u64,
    /// The root owner reserved at startup that no shard has claimed yet.
    pub unclaimed_root_owner: Option<ShardId>,
    /// The epoch of the committed map.
    pub epoch: u64,
    /// The tick the last commit landed at (`None`: none since startup).
    pub last_commit_tick: Option<u64>,
    /// The committed map's regions, parents before their children.
    pub regions: Vec<RegionView>,
    /// The most regions the map may hold; a split at the limit is refused.
    pub max_regions: usize,
    /// Every shard the orchestrator knows of, by id.
    pub shards: Vec<ShardView>,
    /// The epoch change in flight, if any.
    pub in_flight: Option<EpochInFlight>,
    /// The autoscaler's state.
    pub autoscaler: AutoscalerView,
    /// The cluster's scene.
    pub scene: SceneView,
    /// The recent-events log, newest first.
    pub events: Vec<Event>,
}

/// The cluster's scene: what every shard is told to show.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SceneView {
    /// The current scene.
    pub scene: Scene,
    /// Its revision. Rises by one with every change, also across restarts.
    pub revision: u64,
    /// The tick of the last change (`None`: none since startup).
    pub changed_tick: Option<u64>,
    /// The file the scene is persisted in (`None`: it is not persisted).
    pub path: Option<PathBuf>,
}

impl SceneView {
    /// The current scene and revision, as shards are told it.
    #[must_use]
    pub fn current(&self) -> SceneState {
        SceneState::new(self.scene.id(), self.revision)
    }

    /// How what a shard last confirmed (`None`: nothing yet) compares with the
    /// current scene.
    #[must_use]
    pub fn sync_of(&self, confirmed: Option<SceneState>) -> SceneSync {
        let current = self.current();
        match confirmed {
            None => SceneSync::Unconfirmed,
            Some(state) if state == current => SceneSync::InSync,
            Some(state) if state.revision < current.revision => SceneSync::Behind,
            Some(_) => SceneSync::Differs,
        }
    }
}

/// How a shard's confirmed scene compares with the cluster's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneSync {
    /// It confirmed the current scene at the current revision.
    InSync,
    /// It has not confirmed anything since it (last) registered.
    Unconfirmed,
    /// It confirmed an older revision: the current one has not reached it, or
    /// its answer has not come back yet.
    Behind,
    /// It reports a revision the orchestrator never issued (higher than the
    /// current one, or the current one with another scene).
    Differs,
}

impl SceneSync {
    /// A short lower-case name (`"in-sync"`, `"unconfirmed"`, …).
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            SceneSync::InSync => "in-sync",
            SceneSync::Unconfirmed => "unconfirmed",
            SceneSync::Behind => "behind",
            SceneSync::Differs => "differs",
        }
    }
}

/// One region of the committed map.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionView {
    /// The octree node the region covers.
    pub node: NodeRef,
    /// The shard that owns it.
    pub shard: ShardId,
    /// Where the map says that shard is reachable. `None` while the map only
    /// holds a placeholder: a new child that has not said hello yet, or the
    /// root before any shard has claimed it.
    pub endpoint: Option<SocketAddr>,
    /// How many regions of the map contain this one (0 for a top-level region).
    pub nesting: u8,
    /// The enclosing region this one was carved out of, when its octree parent
    /// is itself a region of the map.
    pub parent: Option<NodeRef>,
    /// Which octant of `parent` this region is (`None` when it has no such
    /// parent). Bit 0 = +x, bit 1 = +y, bit 2 = +z.
    pub octant: Option<u8>,
    /// The octants of this region that can still be carved out for a new shard.
    pub free_octants: Vec<u8>,
    /// Why a split of this region would be refused right now (`None`: it
    /// would be accepted).
    pub split_refusal: Option<String>,
    /// Why a merge of this region into its parent would be refused right now
    /// (`None`: it would be accepted).
    pub merge_refusal: Option<String>,
}

/// One shard the orchestrator knows of.
#[derive(Debug, Clone, PartialEq)]
pub struct ShardView {
    /// The shard's id.
    pub shard: ShardId,
    /// What the shard is to the cluster right now.
    pub role: ShardRole,
    /// Whether its control connection is open.
    pub connected: bool,
    /// Whether it has said hello since this orchestrator started.
    pub registered: bool,
    /// The address it advertised in its hello.
    pub advertised: Option<SocketAddr>,
    /// The regions of the committed map it owns.
    pub regions: Vec<NodeRef>,
    /// When it was last heard from, and how.
    pub last_heard: Option<Heard>,
    /// Its latest load report, while fresh. `None` means **no report**, not a
    /// report of zero.
    pub report: Option<ReportView>,
    /// What the autoscaler remembers about it.
    pub scale: Option<ShardScaleState>,
    /// The scene and revision it last confirmed (`None`: nothing since it
    /// registered). Compare with [`Snapshot::scene`] through
    /// [`SceneView::sync_of`].
    pub scene: Option<SceneState>,
}

/// What a shard is to the cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardRole {
    /// It owns at least one region of the committed map.
    Owner,
    /// The root owner reserved at startup; no live shard has claimed it yet.
    ReservedRoot,
    /// The child of the split in flight: launched, owns nothing until the
    /// epoch commits.
    SplitChild,
    /// The shard the merge in flight removes once its epoch commits.
    Merging,
    /// Known (it said hello) but it owns no region of the committed map.
    Unowned,
}

impl ShardRole {
    /// A short lower-case name (`"owner"`, `"split-child"`, …).
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            ShardRole::Owner => "owner",
            ShardRole::ReservedRoot => "reserved-root",
            ShardRole::SplitChild => "split-child",
            ShardRole::Merging => "merging",
            ShardRole::Unowned => "unowned",
        }
    }
}

/// The last sign of life from a shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Heard {
    /// The tick it arrived at.
    pub tick: u64,
    /// What it was.
    pub what: HeardKind,
}

/// The kind of message a shard was last heard through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeardKind {
    /// Its `ShardHello`.
    Hello,
    /// A `LoadReport`.
    Report,
    /// An `EpochAck`.
    Ack,
    /// A `SceneAck`.
    SceneAck,
}

impl HeardKind {
    /// A short lower-case name (`"hello"`, `"load report"`, `"epoch ack"`,
    /// `"scene ack"`).
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            HeardKind::Hello => "hello",
            HeardKind::Report => "load report",
            HeardKind::Ack => "epoch ack",
            HeardKind::SceneAck => "scene ack",
        }
    }
}

/// A shard's latest load report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReportView {
    /// The tick it arrived at.
    pub tick: u64,
    /// Players connected to the shard.
    pub players: u32,
    /// Its tick time at the 95th percentile, in milliseconds.
    pub tick_p95_ms: f32,
    /// The load figure the shard itself put in the report.
    pub reported_load: f32,
    /// The metric the autoscaler computes from the report
    /// ([`crate::autoscale::load_metric`]).
    pub load_metric: f64,
}

/// An epoch change that has been prepared and not yet committed or aborted.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochInFlight {
    /// The epoch being prepared.
    pub epoch: u64,
    /// What the change is, in words.
    pub what: String,
    /// The tick the new map activates at, if it commits.
    pub activation_tick: u64,
    /// The tick the prepare aborts at if an ack is still missing.
    pub deadline_tick: u64,
    /// Everyone the prepare went to, and whether each has acked.
    pub recipients: Vec<(ShardId, bool)>,
    /// The ownership changes: each node and its owner-to-be.
    pub changes: Vec<(NodeRef, ShardId)>,
}

/// The autoscaler's state.
#[derive(Debug, Clone, PartialEq)]
pub struct AutoscalerView {
    /// Whether an operator has paused it.
    pub paused: bool,
    /// The tick until which it takes no action at all (0: never busy yet).
    pub busy_until: u64,
    /// Its most recent decision and what became of it.
    pub last_decision: Option<Decision>,
}

/// One autoscaler decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    /// The tick it was taken at.
    pub tick: u64,
    /// What it decided (`"split shard 1"`).
    pub action: String,
    /// `Ok(what was started)` or `Err(why it was not carried out)`.
    pub outcome: Result<String, String>,
}

/// A node as `depth:x:y:z` — the form `space-server --region` takes.
#[must_use]
pub fn node_path(node: NodeRef) -> String {
    format!(
        "{}:{}:{}:{}",
        node.depth, node.index[0], node.index[1], node.index[2]
    )
}

/// Which octant of its octree parent `node` is (`None` for the octree root).
#[must_use]
pub fn octant_of(node: NodeRef) -> Option<u8> {
    let parent = octree::parent(node)?;
    let children = octree::children(parent)?;
    let position = children.iter().position(|child| *child == node)?;
    u8::try_from(position).ok()
}

/// The half-space signs of an octant, e.g. `"+x +y -z"` for octant 3
/// (bit 0 = +x, bit 1 = +y, bit 2 = +z). Only the low three bits are read.
#[must_use]
pub const fn octant_signs(octant: u8) -> &'static str {
    match octant & 7 {
        0 => "-x -y -z",
        1 => "+x -y -z",
        2 => "-x +y -z",
        3 => "+x +y -z",
        4 => "-x -y +z",
        5 => "+x -y +z",
        6 => "-x +y +z",
        _ => "+x +y +z",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_node_reads_as_depth_and_index() {
        assert_eq!(node_path(NodeRef::new(40, [0, 0, 0])), "40:0:0:0");
        assert_eq!(node_path(NodeRef::new(41, [1, -2, 3])), "41:1:-2:3");
    }

    #[test]
    fn the_octant_of_a_child_is_its_position_among_its_siblings() {
        let root = NodeRef::new(40, [0, 0, 0]);
        let children = octree::children(root).expect("the root splits");
        for (octant, child) in children.iter().enumerate() {
            assert_eq!(octant_of(*child), Some(octant as u8), "{child:?}");
        }
        assert_eq!(octant_of(NodeRef::ROOT), None, "the octree root has none");
        // The depth-1 nodes use the special -1/0 indices; the answer still holds.
        let top = octree::children(NodeRef::ROOT).expect("children");
        assert_eq!(octant_of(top[5]), Some(5));
    }

    #[test]
    fn a_confirmed_scene_is_compared_with_the_current_one() {
        let view = SceneView {
            scene: Scene::Solar,
            revision: 4,
            changed_tick: None,
            path: None,
        };
        assert_eq!(view.current(), SceneState::new(Scene::Solar.id(), 4));
        let at = |scene: Scene, revision| Some(SceneState::new(scene.id(), revision));
        assert_eq!(view.sync_of(None), SceneSync::Unconfirmed);
        assert_eq!(view.sync_of(at(Scene::Solar, 4)), SceneSync::InSync);
        assert_eq!(view.sync_of(at(Scene::DemoArea, 3)), SceneSync::Behind);
        assert_eq!(view.sync_of(at(Scene::Solar, 1)), SceneSync::Behind);
        assert_eq!(
            view.sync_of(at(Scene::Galaxy, 4)),
            SceneSync::Differs,
            "the right revision with the wrong scene is not a confirmation"
        );
        assert_eq!(view.sync_of(at(Scene::Solar, 9)), SceneSync::Differs);
        assert_eq!(SceneSync::Behind.label(), "behind");
    }

    #[test]
    fn the_signs_follow_the_octant_bits() {
        assert_eq!(octant_signs(0), "-x -y -z");
        assert_eq!(octant_signs(1), "+x -y -z");
        assert_eq!(octant_signs(2), "-x +y -z");
        assert_eq!(octant_signs(4), "-x -y +z");
        assert_eq!(octant_signs(7), "+x +y +z");
        assert_eq!(
            octant_signs(8 + 3),
            octant_signs(3),
            "only three bits count"
        );
    }
}
