//! The multiplayer seam: a normal `space-net` player connection.
//!
//! [`ServerLink::join`] is the same client path a native client uses:
//! [`space_net::client_endpoint_for`] + [`space_net::join`] with a
//! [`space_net::hello`] (so the `PROTOCOL_VERSION` gate is the transport's, not
//! reimplemented here). No server protocol is added: the link sends
//! [`Message::InputTick`] batches (datagrams) and reads the replicated
//! [`Message::Snapshot`] / [`Message::Delta`] frames the shard sends every
//! client.
//!
//! # One address
//!
//! The address a player types — the **entry** address — need not be a shard.
//! Every address is treated alike: a `Welcome` means it is one; a
//! `JoinRedirect` (the orchestrator's player port answers with one, and so
//! does a shard that does not own the scene's spawn point) names the shard a
//! new player starts on, and the link dials that shard's **client endpoint**
//! in the redirect's map — [`MAX_JOIN_REDIRECTS`] times at most. A client
//! endpoint without an IP (`0.0.0.0:port`) means "the host of the entry
//! address, this port": the shards run where the player port does, so no
//! host name is configured anywhere. The same rule resolves the address of a
//! handoff's [`Message::Redirect`]. The link keeps the entry address for
//! that ([`ServerLink::entry`]) and says where it is really playing
//! ([`ServerLink::shard_address`]); whoever keeps a connection up joins again
//! through the **entry** address, never through a shard's — the shard may be
//! gone, the entry knows who owns the spawn point now.
//!
//! Three kinds of task run on the caller's tokio runtime: an input sender (one
//! batch per simulation tick) and, per connection, a reader for the
//! reliable-stream frames (snapshots, redirects) and one for datagram frames
//! (deltas). They share one [`Replica`] — the client's view of its *own* ship —
//! which the render loop reads each frame. The replica also keeps the other
//! ships the shard reports ([`Replica::others`]), so players can see each
//! other.
//!
//! The server is authoritative: the link only sends thrust intent; the ship's
//! position comes back through replication ([`Replica::offset`]).
//!
//! # Shard transitions
//!
//! When the ship crosses into a region another shard owns, its shard starts a
//! handoff and tells the client with a [`Message::Redirect`]. The link then
//! does its half of the make-before-break exchange:
//!
//! 1. it opens a second connection, to the accepting shard named in the
//!    redirect's shard map, presenting the **same player token** (that token
//!    is how the accepting shard recognises the redirected player);
//! 2. it reports [`NeighbourStatus`]`{ connected: true }` there, which is the
//!    gate the shards' commit waits for;
//! 3. it keeps playing on the old shard — inputs and frames — until a frame on
//!    the new connection shows its ship, now owned by the new session. That
//!    frame is the switch: inputs go to the new shard from then on and the old
//!    connection is closed. "Its ship" is the ship it has been flying, by
//!    guid. The shard sends the redirect only once the accepting shard has
//!    staged the ship, so the hello there is recognised (by the token) and
//!    the client is given no ship of its own. Should the accepting shard have
//!    dropped the transfer again by then (an abort that crossed the hello),
//!    it answers like for a fresh join: it sends the client on (the attempt
//!    is over, see below) or — the shard that owns the spawn point — lets it
//!    in with a **new** ship. A frame showing that ship is not the switch;
//!    it is the sign that this connection is of no use, and the link closes
//!    it at once, which removes the ship.
//!
//! Only a redirect for the **own** ship is followed. A shard sends one to
//! the ship's player and to nobody else; a redirect naming any other entity
//! is ignored (there is nothing a second connection could do for a ship this
//! client does not fly — it is seen through the own shard's ghost of it).
//!
//! The ship keeps its guid and its world position across the switch, so
//! [`Replica::offset`] — and with it the camera — is continuous: the
//! transition is invisible apart from [`Replica::handoffs`] ticking up. If the
//! new shard cannot be reached, or the handoff is aborted, nothing changes:
//! the old connection was never given up.
//!
//! ## One timeline across the switch
//!
//! The shards of a cluster count the same ticks (the cluster's clock,
//! `space_server::clock`): a tick number names the same instant on the shard
//! the ship leaves and on the one that takes it. The link relies on that:
//!
//! * **Inputs go to both shards** from the moment the second connection is
//!   up until the switch — the same batch, tagged with the same tick. The old
//!   shard applies the ones up to its commit; the new one keeps what it is
//!   sent for a ship that is on its way and applies the rest from the tick it
//!   installs the ship on. Nothing a player does around the switch is lost,
//!   and nothing is applied twice (a shard only ever steps the ship it
//!   simulates).
//! * **The view continues.** At the switch the new shard's frames take over,
//!   and nothing else changes: the tick the link counts by, the inputs it has
//!   sent ([`ServerLink::take_sent`] — the app's prediction keeps replaying
//!   them), the `input_ack`, and the motion histories of every entity the
//!   new shard reports too, merged by guid ([`Tracked::history`]): a body
//!   or a player seen before and after is one object with one history, not a
//!   new one that starts where the old one stopped.
//!
//! Whether the new shard's ticks do continue the old one's is checked at the
//! switch, against the link's own estimate of the present tick
//! ([`SWITCH_TOLERANCE`]). Between shards that count differently — standalone
//! shards given different tick origins, shards driven by loops of their own —
//! the link starts over instead, as it always did: new ticks, new histories,
//! and [`Replica::tick_epoch`] goes up, which tells the app to restart its
//! clock and its prediction.
//!
//! # The anchor
//!
//! Every client must agree on *where* the scene is, so it is tied to the map
//! and not to the client: the scene's spawn point lies on the **anchor**, a
//! point of the map's root region ([`map_anchor`]: the centre of a one-cell
//! root, else a few metres beside and half a cell above the corner where the
//! root's cells meet, clear of every shard border). Every shard hands its
//! clients the same map in the `Welcome`, so every client derives the same
//! point, whichever shard it joined through. The link takes it from the first
//! `Welcome` and keeps it ([`Replica::anchor`]): a handoff, a split or a merge
//! never moves it.
//!
//! # The scene
//!
//! Which world is drawn around the ships is the server's decision, not the
//! client's: every shard sends a [`Message::SceneUpdate`] behind its `Welcome`
//! and again whenever its scene changes. The link keeps the newest one in
//! [`Replica::scene`] ([`ServerLink::scene`]): a [`SceneState`], i.e. a scene
//! id and the revision it was chosen at. The app reads it each frame and
//! compares it with the one it last drew — a different value is a scene
//! change:
//!
//! ```ignore
//! let scene = link.scene();
//! if scene != self.drawn {
//!     self.drawn = scene;
//!     self.load(scene.scene.or_default()); // a `space_protocol::Scene`
//! }
//! ```
//!
//! Revision 0 ([`SceneState::UNSET`]) means the server has not said yet; the
//! default scene applies until it does. An id this build does not know
//! resolves to the default scene too ([`space_protocol::SceneId::or_default`]),
//! and the link says so in the log.
//!
//! # Physics bodies
//!
//! Besides ships, a shard replicates the bodies of its scene — crates, balls,
//! a door, a chain — with their physics components (`Body`, `Visual`,
//! `Springs`, `Articulation`, `PartStates`, `PosResidual`; see `space-sim`).
//! The replica keeps every entity it is told about as a [`Tracked`]: the
//! newest state, the decoded physics components ([`Physical`]) and the last
//! few ticks of motion ([`Sample`]s), which is what lets the app draw a 30 Hz
//! simulation smoothly at any frame rate. [`Replica::others`] lists the
//! ships, [`Replica::bodies`] the bodies ([`Physical::is_body`] tells them
//! apart). A body that a neighbouring shard simulates arrives as its
//! ghost — the same guid, the same components — and is drawn like any other.
//!
//! # The scene clock
//!
//! Every shard maps its ticks to **scene time** (the time the demo area's
//! planets move by) and says how in a [`Message::SceneClock`] behind its
//! `Welcome`, again whenever it re-anchors and every few seconds. The link
//! keeps the mapping of the shard being played on ([`Replica::scene_clock`]);
//! during a handoff the accepting shard's is held back and taken at the
//! switch, together with that shard's ticks — a mapping is only meaningful
//! with the ticks of the shard that made it. (The shards of one cluster all
//! tell the same mapping: it is the cluster's clock, see below.)
//!
//! # Prediction
//!
//! The link remembers every input batch it sends ([`ServerLink::take_sent`])
//! and every frame's `input_ack` ([`Replica::input_ack`]), so the app can run
//! its own ship ahead of the server (`crate::predict`).
//!
//! The rule for which update counts is "the highest revision wins", across
//! **both** connections of a handoff: the scene belongs to the cluster, so a
//! newer revision heard from the accepting shard is taken at once, and an older
//! one — the accepting shard has not caught up yet — never rolls the scene
//! back, before or after the switch. Between two shards that name *different*
//! scenes at the *same* revision (standalone shards started with different
//! `--scene` flags) the one being played on decides.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use space_ecs::{ComponentDef, ComponentId};
use space_net::{client_endpoint_for, hello, join, Conn, Joined, NetError};
use space_protocol::{
    resolve_client_endpoint, EntityGuid, EntitySnapshot, InputTick, Message, NeighbourStatus,
    PlayerInput, Redirect, SceneClock, SceneState, ShardId, ShardMap, Welcome,
};
use space_sim::articulation::{decode_articulation, decode_part_states};
use space_sim::body::{decode_body, decode_residual, decode_visual};
use space_sim::spring::decode_springs;
use space_sim::{
    Articulation, Body, CollisionGroups, Owner, PartStates, PosResidual, ShipControls, ShipState,
    Springs, Visual,
};
use space_world::WorldPos;
use tokio::task::JoinHandle;

/// The TLS server name the shard's self-signed certificate is issued for
/// (`space_server::net::SERVER_NAME`; a test asserts they agree).
pub const SERVER_NAME: &str = "space.local";

/// Prefix of the `player_token` a link presents in its hello. The rest is
/// unique per link ([`unique_player_token`]): the token is how a shard tells
/// players apart during a handoff, so two clients must never share one.
pub const PLAYER_TOKEN_PREFIX: &str = "space-client-";

/// How many ticks ahead of the estimated server tick an input batch is tagged,
/// to land after the one-way delay (the server drops stale inputs and inputs
/// more than `MAX_INPUT_LEAD` ticks ahead).
pub const INPUT_LEAD: u64 = 3;

/// How far the link runs its estimate of the server's tick ahead of the last
/// frame it received, in ticks. Frames normally arrive every tick; this only
/// bridges gaps (a lost datagram, a stalled frame), so inputs keep landing on
/// ticks the server has not simulated yet instead of all being stale.
const MAX_TICK_EXTRAPOLATION: u64 = 30;

/// The most ticks one input batch covers: the ticks since the last batch,
/// each with its own input (see [`inputs_for`]). After a longer stall the
/// ticks before these are in the past and get none.
const MAX_INPUT_FILL: u64 = 4;

/// The ticks an input batch sent now carries inputs for: `first..=last`.
///
/// `estimate` is the tick the link's clock says an input sent now belongs to
/// ([`input_tick`]); `sent` the last tick an input was sent for. A batch goes
/// out every thirtieth of a second, but the estimate comes from frames that
/// arrive with jitter: read naively it names the same tick twice now and
/// then, and skips one as often. A tick without an input **coasts** on the
/// shard — in fast travel that is a fifteenth of the speed gone, for
/// nothing. So the link gives every tick exactly one input: it continues
/// where it stopped (`sent + 1`), runs at most one tick ahead of its clock
/// (`None`: nothing to send this time), and when the clock has moved on by
/// more than one it fills the ticks in between, [`MAX_INPUT_FILL`] at most.
fn inputs_for(estimate: u64, sent: Option<u64>) -> Option<(u64, u64)> {
    let Some(sent) = sent else {
        return Some((estimate, estimate));
    };
    let first = sent + 1;
    if first > estimate + 1 {
        return None;
    }
    let last = estimate.max(first);
    Some((first.max(last.saturating_sub(MAX_INPUT_FILL - 1)), last))
}

/// The most sent inputs the link keeps for an app that does not take them
/// ([`ServerLink::take_sent`]): ten seconds' worth.
const SENT_LOG: usize = 300;

/// How many of the inputs sent last the link keeps to say again to a shard
/// it has just connected to for a handoff ([`resend_tail`]): the ticks a
/// commit can be ahead of the connection — the input lead, a batch's fill
/// and what the estimate of the present tick may be off by.
const RESEND_LOG: usize = (INPUT_LEAD + MAX_INPUT_FILL + SWITCH_TOLERANCE) as usize;

/// The inputs the shard that is taking the ship over has to be sent the
/// moment its connection is up: of the `recent` ones, those for ticks after
/// `newest_frame_tick` — the ticks the old shard may not have stepped yet.
///
/// Inputs are tagged [`INPUT_LEAD`] ticks ahead and each goes out once.
/// Those sent just **before** the second connection existed went to the old
/// shard only; if the commit follows the connect within the lead, some of
/// them are for ticks the *new* shard simulates, and it never heard them:
/// the ship coasted for up to three ticks at every crossing with a quick
/// commit. So the link says them again, once, to the new shard. A tick the
/// old shard did step is not applied twice: a shard only applies an input
/// for a tick it simulates the ship on (the new one keeps what it is sent
/// for a staged ship and files what is for ticks after its install), and
/// one input per tick is kept.
fn resend_tail(recent: &VecDeque<PlayerInput>, newest_frame_tick: u64) -> Vec<PlayerInput> {
    recent
        .iter()
        .filter(|input| input.tick > newest_frame_tick)
        .copied()
        .collect()
}

/// How long a connection opened for a redirect may wait for the handoff to
/// complete before it is abandoned (the shards time a handoff out long before
/// this; a later redirect then starts afresh).
const REDIRECT_PATIENCE: Duration = Duration::from_secs(10);

/// How far the first tick heard from the shard that took the ship over may be
/// from the tick the link expects at that moment, in ticks, for the two
/// shards to count as being on one timeline (see the module docs, "One
/// timeline across the switch"). On the cluster's clock the difference is the
/// delivery of a frame or two; anything like this far apart is two clocks.
pub const SWITCH_TOLERANCE: u64 = 6;

/// The most [`JoinRedirect`](space_protocol::JoinRedirect)s one join follows:
/// the entry address may send the player on (the orchestrator's player port
/// does), and the shard it names may once more (its map was a moment behind).
/// A third is not a path to a shard any more — it is a loop or a
/// misconfigured cluster — and ends the attempt with a readable reason.
pub const MAX_JOIN_REDIRECTS: usize = 2;

/// Why a join did not end on a shard, and at which address.
#[derive(Debug, Clone, PartialEq)]
pub struct JoinError {
    /// The address that was being dialed when it failed: the entry address,
    /// or the shard a redirect named.
    pub at: SocketAddr,
    /// What went wrong there.
    pub failure: JoinFailure,
}

/// What went wrong at one address of a join ([`JoinError`]).
#[derive(Debug, Clone, PartialEq)]
pub enum JoinFailure {
    /// The connection or the hello failed, or was refused.
    Net(NetError),
    /// The address sent the player on once more than
    /// [`MAX_JOIN_REDIRECTS`] allow; these are the shards named, in order.
    TooManyRedirects(Vec<ShardId>),
    /// The redirect named a shard its map gives no address for.
    NoAddress(ShardId),
}

impl std::fmt::Display for JoinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.failure {
            JoinFailure::Net(error) => write!(f, "{}: {error}", self.at),
            JoinFailure::TooManyRedirects(chain) => {
                let chain: Vec<String> = chain.iter().map(ToString::to_string).collect();
                write!(
                    f,
                    "{} sent this client on a {}. time (to {}); a join follows at most {}",
                    self.at,
                    chain.len(),
                    chain.join(", then "),
                    MAX_JOIN_REDIRECTS
                )
            }
            JoinFailure::NoAddress(shard) => write!(
                f,
                "{} sent this client to {shard}, and its map names no address for it",
                self.at
            ),
        }
    }
}

impl std::error::Error for JoinError {}

/// A fresh player token: the prefix plus 64 process-unique random bits.
#[must_use]
pub fn unique_player_token() -> String {
    use std::hash::{BuildHasher, Hasher};
    // `RandomState` is seeded from the OS per instance: no extra dependency.
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u32(std::process::id());
    format!("{PLAYER_TOKEN_PREFIX}{:016x}", hasher.finish())
}

/// Where a scene's spawn point lies in the world of `map`: the **anchor** of
/// the map's **root region**.
///
/// The root region is the shallowest region of the map — the one every split
/// and carve took its node out of. Carves only *add* a deeper region and
/// merges only remove one (`space-orchestrator`'s topology), so the root, and
/// with it this point, stays where it is for as long as the cluster lives,
/// and it is where the shard that owns it spawns the ships of the players who
/// join it. A map without a single root (standalone shards side by side, all
/// of one depth) takes the region of the lowest shard id; the node index
/// breaks what ties are left. `None` for a map with no region at all.
///
/// Within the root the anchor is the centre of a root of one cell, and for a
/// larger one the corner where its cells meet moved by
/// [`space_scene::ANCHOR_OFFSET`] — 6 m, half a cell up, 4 m — so that no
/// horizontal shard border comes within 512 m of the scene and the vertical
/// ones pass beside the spawn point (at `pos` x = −6 and z = −4 on the HUD),
/// not through it ([`space_scene::anchor_in`]).
///
/// A pure function of the map: two clients handed the same map — by whichever
/// shards — get the same point. It is [`space_scene::scene_anchor`], the rule
/// the shards place their scene's statics and bodies by, so what a client
/// draws and what a shard simulates are in the same place.
#[must_use]
pub fn map_anchor(map: &ShardMap) -> Option<WorldPos> {
    space_scene::scene_anchor(map)
}

/// How many ticks of motion a [`Tracked`] entity keeps for interpolation.
pub const HISTORY_TICKS: usize = 10;

/// A gap between two updates of a *moving* entity longer than this, in
/// ticks, is not interpolated across: the entity left the interest range and
/// came back, or was moved by hand. Its history starts over.
const MAX_INTERPOLATION_GAP: u64 = 15;

/// Below this speed (m/s) and spin (rad/s) an entity counts as resting: a
/// shard sends nothing for a resting entity until it moves again, so the gap
/// before its next update is time it spent where it was.
const RESTING: f32 = 0.05;

/// The motion of one replicated entity at one tick: every part (part 0 is
/// the entity's `ShipState`, then its `PartStates`), and each part's
/// [`PosResidual`] (zero for a ship).
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    /// The tick this is the state of.
    pub tick: u64,
    /// The parts' states.
    pub parts: Vec<ShipState>,
    /// The parts' residuals, metres (empty: none).
    pub residual: Vec<[f32; 3]>,
}

impl Sample {
    fn resting(&self) -> bool {
        let slow = |v: [f32; 3]| v.iter().map(|c| c * c).sum::<f32>() < RESTING * RESTING;
        self.parts.iter().all(|s| slow(s.vel) && slow(s.ang_vel))
    }
}

/// The physics components of a replicated entity, decoded. Shared between the
/// replica's copies (they change rarely; the motion changes every tick).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Physical {
    /// Its rigid body, if it carries one.
    pub body: Option<Body>,
    /// Which model to draw (a `space_scene::visual` id).
    pub visual: Option<u32>,
    /// Its springs.
    pub springs: Springs,
    /// Its extra parts and joints.
    pub articulation: Option<Articulation>,
    /// The encoded components this was decoded from (to notice a change).
    raw: Vec<(ComponentId, Vec<u8>)>,
}

impl Physical {
    /// Whether an entity with these components, driven by session `owner`
    /// (0: nobody), is a physics body (a crate, a ball, a door) rather than
    /// a ship: it has a model of its own to draw — a visual other than
    /// "none" and "player" — or a rigid body that is not a ship's.
    #[must_use]
    pub fn is_body(&self, owner: u64) -> bool {
        if owner != 0 {
            return false;
        }
        match self.visual {
            Some(space_scene::visual::PLAYER) => false,
            Some(v) if v != space_scene::visual::NONE => true,
            _ => self
                .body
                .as_ref()
                .is_some_and(|b| b.groups != CollisionGroups::SHIP),
        }
    }

    /// The number of rigid parts (1 + the articulation's extra parts).
    #[must_use]
    pub fn part_count(&self) -> usize {
        1 + self.articulation.as_ref().map_or(0, |a| a.parts.len())
    }

    /// Decode the physics components of `entity`; `None` when it has none.
    /// A component that does not decode is left out (and logged once).
    fn of(entity: &EntitySnapshot, previous: Option<&Arc<Self>>) -> Option<Arc<Self>> {
        let raw: Vec<(ComponentId, Vec<u8>)> = entity
            .components
            .iter()
            .filter(|(id, _)| [Body::ID, Visual::ID, Springs::ID, Articulation::ID].contains(id))
            .cloned()
            .collect();
        if raw.is_empty() {
            return None;
        }
        if let Some(previous) = previous.filter(|p| p.raw == raw) {
            return Some(Arc::clone(previous));
        }
        let mut out = Self::default();
        for (id, bytes) in &raw {
            let result = match *id {
                Body::ID => decode_body(bytes).map(|b| out.body = Some(b)),
                Visual::ID => decode_visual(bytes).map(|v| out.visual = Some(v.0)),
                Springs::ID => decode_springs(bytes).map(|s| out.springs = s),
                Articulation::ID => decode_articulation(bytes).map(|a| out.articulation = Some(a)),
                _ => Ok(()),
            };
            if let Err(e) = result {
                log::warn!("space-client: entity {}: component {id}: {e}", entity.guid);
            }
        }
        out.raw = raw;
        Some(Arc::new(out))
    }
}

/// The client's replicated view of its own ship.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Replica {
    /// Session id from the current shard's `Welcome`; matches the ship's `Owner`.
    pub session: u64,
    /// The shard the client is currently playing on.
    pub shard: Option<ShardId>,
    /// Highest snapshot/delta tick seen (0 = none yet).
    pub tick: u64,
    /// Our ship's guid, once a frame carrying it has arrived.
    pub ship: Option<EntityGuid>,
    /// Our ship's latest replicated state.
    pub state: Option<ShipState>,
    /// The tick `state` is from.
    state_tick: u64,
    /// The `input_ack` of the frame `state` came in: the highest input tick
    /// the shard had applied from this client.
    input_ack: u64,
    /// The `input_ack` of the newest frame (by tick), whatever it carried.
    frame_ack: u64,
    /// When `tick` last advanced (the base of the app's estimate of the
    /// shard's clock).
    pub tick_seen: Option<Instant>,
    /// Where the ship was in the first frame that showed it.
    origin: Option<WorldPos>,
    /// The anchor of the map the first shard sent ([`map_anchor`]); kept for
    /// the life of the link.
    anchor: Option<WorldPos>,
    /// Frames (snapshots + deltas) applied.
    pub frames: u64,
    /// Completed shard transitions.
    pub handoffs: u64,
    /// How often the ticks started over: it goes up at a shard transition
    /// onto a shard that does not continue the previous one's ticks (see the
    /// module docs, "One timeline across the switch"). While it stays, every
    /// tick of this view is on one timeline.
    pub tick_epoch: u64,
    /// The scene the server says to draw, and the revision it was chosen at
    /// ([`SceneState::UNSET`] until the first [`Message::SceneUpdate`]). It
    /// survives a shard transition; compare it with the value last drawn to
    /// notice a change (see the module docs, "The scene").
    pub scene: SceneState,
    /// How the shard being played on maps its ticks to scene time (see the
    /// module docs, "The scene clock"); `None` until it has said.
    pub scene_clock: Option<SceneClock>,
    /// Every ship the shard being played on has reported and not removed
    /// again, our own included (see [`Replica::others`]).
    ships: ShipView,
}

/// Another ship the shard reports: somebody else's, or nobody's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OtherShip {
    /// The ship's guid (stable across handoffs).
    pub guid: EntityGuid,
    /// The session driving it on the shard being played on; 0 for a ship no
    /// player owns (sessions start at 1).
    pub owner: u64,
    /// Its latest replicated state.
    pub state: ShipState,
}

/// The entities one connection has been told about: what its snapshots and
/// deltas add up to.
///
/// A shard sends a snapshot (everything in the interest sphere) on join and
/// deltas (changed entities plus explicit removals) afterwards, on two paths
/// that may overtake each other — so every entity remembers the tick its state
/// is from, and an older frame never rewinds or resurrects one.
#[derive(Clone, Debug, Default, PartialEq)]
struct ShipView {
    ships: Vec<Tracked>,
    /// A snapshot that is arriving in several chunks
    /// ([`space_protocol::Snapshot::more`]): the tick of its first chunk and
    /// every entity its chunks have named so far. `None` between snapshots.
    arriving: Option<(u64, Vec<EntityGuid>)>,
}

/// One entity a shard replicates — a ship or a physics body — as the client
/// keeps it.
#[derive(Clone, Debug, PartialEq)]
pub struct Tracked {
    /// Guid, owner and newest state (of part 0).
    pub ship: OtherShip,
    /// The tick `ship.state` is from.
    pub tick: u64,
    /// Its physics components, if it has any.
    pub physics: Option<Arc<Physical>>,
    /// Its motion over the last ticks, oldest first (at most
    /// [`HISTORY_TICKS`]): one sample per tick it was sent, plus a copy of a
    /// resting state just before it moved again.
    pub history: VecDeque<Sample>,
    /// Its replicated component list as last received (what a prediction
    /// world builds its copy of the entity from).
    pub components: Arc<Vec<(ComponentId, Vec<u8>)>>,
}

impl Tracked {
    /// Whether this is a physics body rather than a ship
    /// ([`Physical::is_body`]).
    #[must_use]
    pub fn is_body(&self) -> bool {
        self.physics
            .as_ref()
            .is_some_and(|p| p.is_body(self.ship.owner))
    }

    /// Whether this is a player's ship: one a session of this shard drives,
    /// or one the shard marks with the player visual
    /// (`space_scene::visual::PLAYER`) — the ghost of a player another shard
    /// simulates arrives with owner 0 and that visual. A ship that is
    /// neither is a ship nobody flies.
    #[must_use]
    pub fn is_player(&self) -> bool {
        self.ship.owner != 0
            || self
                .physics
                .as_ref()
                .is_some_and(|p| p.visual == Some(space_scene::visual::PLAYER))
    }

    /// The newest sample.
    #[must_use]
    pub fn latest(&self) -> Option<&Sample> {
        self.history.back()
    }

    /// Add the motion of `tick` to the history (see [`Tracked::history`]).
    fn record(&mut self, sample: Sample) {
        if let Some(last) = self.history.back() {
            if sample.tick <= last.tick {
                // A repeat of the newest tick replaces it; an older one has
                // nothing to add.
                if sample.tick == last.tick {
                    self.history.pop_back();
                    self.history.push_back(sample);
                }
                return;
            }
            let gap = sample.tick - last.tick;
            if gap > 1 && last.resting() {
                // Nothing was sent in between because nothing changed: it
                // stood where it was until the tick before this one.
                let hold = Sample {
                    tick: sample.tick - 1,
                    ..last.clone()
                };
                self.history.push_back(hold);
            } else if gap > MAX_INTERPOLATION_GAP {
                self.history.clear();
            }
        }
        self.history.push_back(sample);
        while self.history.len() > HISTORY_TICKS {
            self.history.pop_front();
        }
    }
}

/// The motion of `entity` at `tick`: its `ShipState`, its `PartStates` (when
/// they match its articulation) and its `PosResidual`.
fn sample_of(entity: &EntitySnapshot, state: ShipState, tick: u64, parts: usize) -> Sample {
    let mut states = vec![state];
    if parts > 1 {
        let extra = component(entity, PartStates::ID).and_then(|b| decode_part_states(b).ok());
        match extra {
            Some(p) if p.0.len() + 1 == parts => states.extend(p.0),
            // A mismatch is the server's mistake: draw the extra parts where
            // the root is rather than not at all.
            _ => states.resize(parts, state),
        }
    }
    let residual = component(entity, PosResidual::ID)
        .and_then(|b| decode_residual(b).ok())
        .map(|r| r.0)
        .filter(|r| r.len() == states.len())
        .unwrap_or_default();
    Sample {
        tick,
        parts: states,
        residual,
    }
}

impl ShipView {
    /// Apply one snapshot or delta; anything else is ignored.
    fn apply(&mut self, message: &Message) {
        let (tick, entities) = match message {
            Message::Snapshot(s) => (s.tick, &s.entities),
            Message::Delta(d) => (d.tick, &d.entities),
            _ => return,
        };
        let named = |guid: EntityGuid| entities.iter().any(|e| e.guid == guid);
        // Removals first: a guid a frame both removes and names (an entity
        // that left one shard's simulation and is shown as its ghost from the
        // same tick on) stays, with its history.
        match message {
            // A snapshot is the whole interest set: what it does not name is
            // gone — unless a newer frame has already said otherwise. The
            // **whole** snapshot: one that does not fit a frame arrives in
            // chunks, a tick apart, and only the last of them (`more` is
            // false) ends it. Until then the names are collected and nothing
            // is dropped; taking each chunk for the whole kept only the last
            // chunk's entities, for good — the shard counts every entity it
            // sent as known and does not send one at rest again.
            Message::Snapshot(snapshot) => {
                let (first_tick, mut names) =
                    self.arriving.take().unwrap_or_else(|| (tick, Vec::new()));
                names.extend(entities.iter().map(|e| e.guid));
                if snapshot.more {
                    self.arriving = Some((first_tick, names));
                } else {
                    self.ships
                        .retain(|seen| seen.tick > first_tick || names.contains(&seen.ship.guid));
                }
            }
            Message::Delta(d) => self.ships.retain(|seen| {
                seen.tick > tick || !d.removed.contains(&seen.ship.guid) || named(seen.ship.guid)
            }),
            _ => {}
        }
        let full = matches!(message, Message::Snapshot(_));
        for entry in entities {
            let held = self.ships.iter().position(|s| s.ship.guid == entry.guid);
            if held.is_some_and(|i| tick < self.ships[i].tick) {
                // Older than what is held: it can only rewind.
                continue;
            }
            // A snapshot, and a delta entry for an entity not held yet, carry
            // the full component set; any other delta entry only what changed
            // (protocol 5): merged into what is held.
            let entity = match held {
                Some(i) if !full => EntitySnapshot::new(
                    entry.guid,
                    merge_components(&self.ships[i].components, &entry.components),
                ),
                _ => EntitySnapshot::new(
                    entry.guid,
                    entry
                        .components
                        .iter()
                        .filter(|(_, bytes)| !bytes.is_empty())
                        .cloned()
                        .collect(),
                ),
            };
            let Some(state) = ship_state_of(&entity) else {
                continue;
            };
            let ship = OtherShip {
                guid: entity.guid,
                owner: owner_of(&entity).unwrap_or(0),
                state,
            };
            match held {
                Some(i) => {
                    let seen = &mut self.ships[i];
                    seen.physics = Physical::of(&entity, seen.physics.as_ref());
                    let parts = seen.physics.as_ref().map_or(1, |p| p.part_count());
                    seen.ship = ship;
                    seen.tick = tick;
                    seen.record(sample_of(&entity, state, tick, parts));
                    seen.components = Arc::new(entity.components);
                }
                None => {
                    let physics = Physical::of(&entity, None);
                    let parts = physics.as_ref().map_or(1, |p| p.part_count());
                    let mut seen = Tracked {
                        ship,
                        tick,
                        physics,
                        history: VecDeque::new(),
                        components: Arc::new(Vec::new()),
                    };
                    seen.record(sample_of(&entity, state, tick, parts));
                    seen.components = Arc::new(entity.components);
                    self.ships.push(seen);
                }
            }
        }
    }
}

impl ShipView {
    /// Make this view — what the shard that took the ship over has reported
    /// — the continuation of `previous`, the view of the shard played on
    /// until now, on the same tick timeline: every entity both report keeps
    /// the motion `previous` recorded before this view's first word on it.
    /// What only `previous` knew is not this shard's to report and is gone.
    fn continue_from(&mut self, previous: &ShipView) {
        for seen in &mut self.ships {
            let Some(before) = previous
                .ships
                .iter()
                .find(|old| old.ship.guid == seen.ship.guid)
            else {
                continue;
            };
            let first = seen.history.front().map_or(u64::MAX, |sample| sample.tick);
            let news = std::mem::take(&mut seen.history);
            seen.history = before
                .history
                .iter()
                .filter(|sample| sample.tick < first)
                .cloned()
                .collect();
            // Through `record`, so a gap is treated as on one connection.
            for sample in news {
                seen.record(sample);
            }
        }
    }
}

/// `held` updated by a delta entry's `changes`: a listed component replaces
/// the held one, an empty payload removes it, one not listed stays. In id
/// order.
fn merge_components(
    held: &[(ComponentId, Vec<u8>)],
    changes: &[(ComponentId, Vec<u8>)],
) -> Vec<(ComponentId, Vec<u8>)> {
    let mut out: Vec<(ComponentId, Vec<u8>)> = held
        .iter()
        .filter(|(id, _)| !changes.iter().any(|(c, _)| c == id))
        .cloned()
        .collect();
    out.extend(
        changes
            .iter()
            .filter(|(_, bytes)| !bytes.is_empty())
            .cloned(),
    );
    out.sort_by_key(|(id, _)| *id);
    out
}

impl Replica {
    fn new(session: u64, shard: ShardId) -> Self {
        Self {
            session,
            shard: Some(shard),
            ..Self::default()
        }
    }

    /// A view on the shard that sent `map` in its welcome: anchored to it.
    fn joined(session: u64, shard: ShardId, map: &ShardMap) -> Self {
        Self {
            anchor: map_anchor(map),
            ..Self::new(session, shard)
        }
    }

    /// The point of the world a scene's spawn point lies on (see the module
    /// docs, "The anchor"): the anchor of the map this link was welcomed
    /// with. It never changes while the link lives. A map without regions
    /// (no shard sends one) falls back to where the ship first appeared;
    /// `None` only until then.
    #[must_use]
    pub fn anchor(&self) -> Option<WorldPos> {
        self.anchor.or(self.origin)
    }

    /// Take `update` as the scene if it counts (see the module docs, "The
    /// scene"): a newer revision always does; the same revision naming another
    /// scene does only when it comes from the shard being played on
    /// (`from_active`). Returns whether the scene changed.
    fn adopt_scene(&mut self, update: SceneState, from_active: bool) -> bool {
        let newer = update.supersedes(&self.scene);
        let same_revision_other_scene = from_active
            && update.revision == self.scene.revision
            && update.scene != self.scene.scene;
        if !(newer || same_revision_other_scene) {
            return false;
        }
        self.scene = update;
        true
    }

    /// Apply one server message; anything that is not a snapshot/delta is
    /// ignored. Returns whether the frame carried this session's ship.
    pub fn apply(&mut self, message: &Message) -> bool {
        let (tick, entities, input_ack) = match message {
            Message::Snapshot(s) => (s.tick, &s.entities, s.input_ack),
            Message::Delta(d) => (d.tick, &d.entities, d.input_ack),
            _ => return false,
        };
        self.frames += 1;
        self.ships.apply(message);
        if tick >= self.tick {
            self.frame_ack = self.frame_ack.max(input_ack);
        }
        // Frames arrive on two paths (reliable snapshots, datagram deltas) and
        // may overtake each other, so the clock only moves forwards — but an
        // older frame is not thrown away wholesale: it may be the only one
        // that names our ship (the join snapshot arriving after a first delta).
        self.tick = self.tick.max(tick);
        let mut seen = false;
        for entity in entities {
            if let Some(state) = self.own_state(entity) {
                seen = true;
                self.ship = Some(entity.guid);
                self.origin.get_or_insert(state.pos);
                // A late frame must not rewind the ship.
                if tick >= self.state_tick || self.state.is_none() {
                    self.state = Some(state);
                    self.state_tick = tick;
                    self.input_ack = input_ack;
                }
            }
        }
        seen
    }

    /// Whether `message` is a frame carrying the ship `guid`, owned by
    /// `session`.
    ///
    /// The guid matters: a shard gives a client that joins it a ship of its
    /// own unless it already holds the handoff's offer for that player, so a
    /// ship owned by `session` is not necessarily the one being handed over.
    fn frame_has_ship_of(message: &Message, session: u64, guid: EntityGuid) -> bool {
        let entities = match message {
            Message::Snapshot(s) => &s.entities,
            Message::Delta(d) => &d.entities,
            _ => return false,
        };
        entities.iter().any(|entity| {
            entity.guid == guid
                && owner_of(entity) == Some(session)
                && ship_state_of(entity).is_some()
        })
    }

    /// Whether `message` is a frame carrying a ship owned by `session` that
    /// is **not** `guid`: a ship the shard made for this session itself.
    fn frame_has_other_ship_of(message: &Message, session: u64, guid: EntityGuid) -> bool {
        let entities = match message {
            Message::Snapshot(s) => &s.entities,
            Message::Delta(d) => &d.entities,
            _ => return false,
        };
        entities.iter().any(|entity| {
            entity.guid != guid
                && owner_of(entity) == Some(session)
                && ship_state_of(entity).is_some()
        })
    }

    /// Whether the own ship is in fast travel, as the shard last said (its
    /// replicated `ShipControls`).
    #[must_use]
    pub fn own_fast(&self) -> bool {
        let Some(guid) = self.ship else {
            return false;
        };
        self.ships
            .ships
            .iter()
            .find(|t| t.ship.guid == guid)
            .and_then(|t| {
                t.components
                    .iter()
                    .find(|(id, _)| *id == ShipControls::ID)
                    .and_then(|(_, b)| space_protocol::codec::decode_raw::<ShipControls>(b).ok())
            })
            .is_some_and(|(c, _)| c.fast)
    }

    /// The ship state in `entity`, if it is this session's ship.
    fn own_state(&self, entity: &EntitySnapshot) -> Option<ShipState> {
        if owner_of(entity)? != self.session {
            return None;
        }
        ship_state_of(entity)
    }

    /// Continue this view on another shard after a handoff: a new session,
    /// the same ship, the same origin and the same scene. `ships` is what the
    /// new shard has reported on its connection so far.
    ///
    /// `continuous`: the new shard's ticks continue the old one's (the
    /// cluster's clock). Then nothing but the session changes — the tick, the
    /// ship's last state and its `input_ack` stand until the new shard's next
    /// word, and the entities both shards report keep their histories
    /// ([`ShipView::continue_from`]). Otherwise the ticks start over
    /// ([`Replica::tick_epoch`] goes up) and the view is the new shard's
    /// alone.
    fn switch_to(&mut self, session: u64, shard: ShardId, mut ships: ShipView, continuous: bool) {
        self.session = session;
        self.shard = Some(shard);
        self.handoffs += 1;
        if continuous {
            ships.continue_from(&self.ships);
        } else {
            self.tick = 0;
            self.state_tick = 0;
            self.input_ack = 0;
            self.frame_ack = 0;
            self.tick_epoch += 1;
        }
        self.ships = ships;
    }

    /// The `input_ack` of the newest frame heard.
    #[must_use]
    pub fn frame_ack(&self) -> u64 {
        self.frame_ack
    }

    /// The tick of [`Replica::state`] (0 before the first).
    #[must_use]
    pub fn state_tick(&self) -> u64 {
        self.state_tick
    }

    /// The `input_ack` of the frame [`Replica::state`] came in.
    #[must_use]
    pub fn input_ack(&self) -> u64 {
        self.input_ack
    }

    /// Every entity the shard reports — ships, bodies and our own ship —
    /// with its history.
    pub fn tracked(&self) -> impl Iterator<Item = &Tracked> {
        self.ships.ships.iter()
    }

    /// The physics bodies the shard reports (see [`Physical::is_body`]).
    pub fn bodies(&self) -> impl Iterator<Item = &Tracked> {
        self.ships.ships.iter().filter(|t| t.is_body())
    }

    /// The entity `guid`, if the shard reports it.
    #[must_use]
    pub fn entity(&self, guid: EntityGuid) -> Option<&Tracked> {
        self.ships.ships.iter().find(|t| t.ship.guid == guid)
    }

    /// Where the ship was in the first frame that showed it: the point
    /// [`Replica::offset`] is measured from. `None` until that frame.
    #[must_use]
    pub fn origin(&self) -> Option<WorldPos> {
        self.origin
    }

    /// The other ships the shard reports: every ship in the interest sphere
    /// except our own. Ships owned by a session are other players'; owner 0
    /// is a ship nobody flies. Physics bodies are not ships
    /// ([`Replica::bodies`]).
    pub fn others(&self) -> impl Iterator<Item = &OtherShip> {
        self.other_ships().map(|seen| &seen.ship)
    }

    /// [`Replica::others`] with their histories.
    pub fn other_ships(&self) -> impl Iterator<Item = &Tracked> {
        let own = self.ship;
        let session = self.session;
        self.ships.ships.iter().filter(move |seen| {
            Some(seen.ship.guid) != own && seen.ship.owner != session && !seen.is_body()
        })
    }

    /// The ship's displacement from where it first appeared, metres.
    #[must_use]
    pub fn offset(&self) -> [f64; 3] {
        match (self.origin, self.state) {
            (Some(origin), Some(s)) => WorldPos::delta(origin, s.pos),
            _ => [0.0; 3],
        }
    }
}

fn component(entity: &EntitySnapshot, id: space_ecs::ComponentId) -> Option<&[u8]> {
    entity
        .components
        .iter()
        .find(|(c, _)| *c == id)
        .map(|(_, p)| p.as_slice())
}

/// The session that owns `entity`, if it carries an `Owner`.
fn owner_of(entity: &EntitySnapshot) -> Option<u64> {
    let owner = component(entity, <Owner as ComponentDef>::ID)?;
    let (owner, _) = space_protocol::codec::decode_raw::<u64>(owner).ok()?;
    Some(owner)
}

fn ship_state_of(entity: &EntitySnapshot) -> Option<ShipState> {
    space_sim::wire::decode_ship_state(component(entity, <ShipState as ComponentDef>::ID)?).ok()
}

/// A connection opened for a redirect, waiting for the handoff to complete.
#[derive(Debug)]
struct Pending {
    conn: Conn,
    /// Identifies the connection's reader tasks.
    generation: u64,
    session: u64,
    shard: ShardId,
    /// The entity the redirect was about: the ship being handed over.
    entity: EntityGuid,
    opened: Instant,
    /// The highest tick of a frame heard on this connection (0: none yet):
    /// what the inputs sent to this shard acknowledge.
    tick: u64,
    /// The newest scene this shard has announced on this connection, kept for
    /// the moment it becomes the shard being played on.
    scene: Option<SceneState>,
    /// Its scene clock, kept for the same moment.
    scene_clock: Option<SceneClock>,
    /// The ships this shard has reported on this connection, kept for the
    /// same moment: its join snapshot arrives long before the switch, and the
    /// deltas after it only name ships that change.
    ships: ShipView,
}

#[derive(Debug)]
struct Shared {
    replica: Replica,
    /// When `replica.tick` last advanced: the base of the local tick estimate.
    tick_seen: Instant,
    /// Thrust intent, each axis in `-1..=1` (world frame; the ship is never
    /// torqued, so its body frame stays the world frame).
    thrust: [f32; 3],
    /// Fast travel on, sent with every input.
    fast: bool,
    /// Input batches sent.
    seq: u64,
    /// The last tick an input was sent for ([`inputs_for`]); `None` before
    /// the first, and after a switch onto a shard that counts other ticks.
    input_sent: Option<u64>,
    /// The inputs sent since the app last took them ([`ServerLink::take_sent`]),
    /// in the clock of the shard being played on (cleared at a switch onto a
    /// shard that counts other ticks).
    sent: VecDeque<PlayerInput>,
    /// The last [`RESEND_LOG`] inputs sent, whether or not the app took
    /// them: what a shard connected to for a handoff is told again
    /// ([`resend_tail`]).
    recent: VecDeque<PlayerInput>,
    /// The address the player dialed first — what was typed, or `--server`.
    /// A client endpoint without an IP (`0.0.0.0:port`) means "this host,
    /// that port", for a join redirect and a handoff redirect alike.
    entry: SocketAddr,
    /// The connection being played on: inputs go here, its frames are the view.
    active: Conn,
    /// Which reader generation belongs to `active`.
    active_generation: u64,
    next_generation: u64,
    /// The accepting shard's connection while a handoff is in flight.
    pending: Option<Pending>,
    /// A redirect is being followed right now (the connect is in flight).
    following: bool,
    /// Every task of the link, so `close` can stop them.
    tasks: Vec<JoinHandle<()>>,
}

impl Shared {
    /// Remember `task` so `close` can stop it, forgetting the ones that have
    /// already ended (a long session sees many short-lived tasks).
    fn track(&mut self, task: JoinHandle<()>) {
        self.tasks.retain(|t| !t.is_finished());
        self.tasks.push(task);
    }
}

type SharedRef = Arc<Mutex<Shared>>;

fn lock(shared: &SharedRef) -> MutexGuard<'_, Shared> {
    shared
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A live player connection to the authoritative world.
#[derive(Debug)]
pub struct ServerLink {
    shared: SharedRef,
    player_token: Arc<str>,
}

impl ServerLink {
    /// Join the world through `entry` as an ordinary player and start the
    /// sender/reader tasks. Must be called inside a tokio runtime.
    ///
    /// Every address is treated alike (`docs/plan-sharding.md` §4). Whatever
    /// listens at `entry` answers the hello with one of three things:
    ///
    /// * a `Welcome` — it is a shard, and this is where the player plays;
    /// * a `JoinRedirect` — it is the orchestrator's player port, or a shard
    ///   that does not own the scene's spawn point: the link dials the shard
    ///   it names, at the **client endpoint** its map gives for that shard
    ///   (an endpoint without an IP, `0.0.0.0:port`, meaning `entry`'s host
    ///   and that port), and says hello again there — at most
    ///   [`MAX_JOIN_REDIRECTS`] times in one join;
    /// * a `Reject` — the reason is the error.
    ///
    /// The link remembers `entry` for as long as it lives ([`Self::entry`]):
    /// it is what a later handoff's client endpoints are resolved against.
    ///
    /// # Errors
    ///
    /// [`JoinError`]: the address that was being dialed and what went wrong
    /// there — the transport or the hello failed or was refused (including
    /// [`NetError::VersionMismatch`]), a redirect too many, or a redirect to a
    /// shard the map names no address for.
    pub async fn join(entry: SocketAddr) -> Result<Self, JoinError> {
        let player_token: Arc<str> = unique_player_token().into();
        let (conn, welcome) = join_through(entry, &player_token).await?;
        Ok(Self::start(entry, player_token, conn, &welcome))
    }

    /// [`Self::join`], with a failure reduced to its [`NetError`] (a redirect
    /// too many, or to nowhere, is a [`NetError::Protocol`] that says so).
    ///
    /// # Errors
    ///
    /// As [`Self::join`].
    pub async fn connect(addr: SocketAddr) -> Result<Self, NetError> {
        Self::join(addr).await.map_err(|err| match err.failure {
            JoinFailure::Net(error) => error,
            JoinFailure::TooManyRedirects(_) | JoinFailure::NoAddress(_) => {
                NetError::Protocol(err.to_string())
            }
        })
    }

    /// Start the link's tasks on a connection a shard welcomed.
    fn start(entry: SocketAddr, player_token: Arc<str>, conn: Conn, welcome: &Welcome) -> Self {
        // (The client endpoint may drop: the connection keeps its driver alive.)
        let shared = Arc::new(Mutex::new(Shared {
            replica: Replica::joined(welcome.session, welcome.shard, &welcome.shard_map),
            tick_seen: Instant::now(),
            thrust: [0.0; 3],
            fast: false,
            seq: 0,
            input_sent: None,
            sent: VecDeque::new(),
            recent: VecDeque::new(),
            entry,
            active: conn.clone(),
            active_generation: 0,
            next_generation: 1,
            pending: None,
            following: false,
            tasks: Vec::new(),
        }));
        let tasks = vec![
            tokio::spawn(send_inputs(Arc::clone(&shared))),
            spawn_reader(&shared, &player_token, conn.clone(), 0, false),
            spawn_reader(&shared, &player_token, conn, 0, true),
        ];
        lock(&shared).tasks = tasks;
        Self {
            shared,
            player_token,
        }
    }

    /// The player token this link presents to every shard.
    #[must_use]
    pub fn player_token(&self) -> &str {
        &self.player_token
    }

    /// The address this link was joined through: what the player dialed
    /// first. It never changes — not with a join redirect, not with a
    /// handoff.
    #[must_use]
    pub fn entry(&self) -> SocketAddr {
        lock(&self.shared).entry
    }

    /// The address of the shard being played on right now: the entry address
    /// itself when that is a shard, otherwise where a redirect led — and
    /// after a handoff, the shard that took the ship.
    #[must_use]
    pub fn shard_address(&self) -> SocketAddr {
        lock(&self.shared).active.remote_address()
    }

    /// Set the thrust intent sent with every following input batch.
    pub fn set_thrust(&self, thrust: [f32; 3]) {
        lock(&self.shared).thrust = thrust;
    }

    /// Set fast travel on or off for every following input batch.
    pub fn set_fast(&self, fast: bool) {
        lock(&self.shared).fast = fast;
    }

    /// The inputs sent since the last call, oldest first, each tagged with
    /// the tick the shard is to apply it at. A prediction replays them on top
    /// of the shard's last word — across a shard transition too: the shards
    /// of a cluster count the same ticks, and around the switch every input
    /// went to both. (Only a transition onto a shard that counts other ticks
    /// drops what was sent to the old one: [`Replica::tick_epoch`].)
    pub fn take_sent(&self) -> Vec<PlayerInput> {
        lock(&self.shared).sent.drain(..).collect()
    }

    /// The ship's replicated displacement from its first-seen position, metres.
    #[must_use]
    pub fn offset(&self) -> [f64; 3] {
        lock(&self.shared).replica.offset()
    }

    /// A copy of the current replicated view.
    #[must_use]
    pub fn replica(&self) -> Replica {
        lock(&self.shared).replica.clone()
    }

    /// The scene the server says to draw, and its revision.
    ///
    /// Cheap enough to call every frame. The value changes exactly when the
    /// scene does, so an app keeps the one it last drew and compares; the
    /// scene itself is `scene().scene.or_default()`. Revision 0 means the
    /// server has not said yet (see the module docs, "The scene").
    #[must_use]
    pub fn scene(&self) -> SceneState {
        lock(&self.shared).replica.scene
    }

    /// Whether the connection to the world is gone.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        lock(&self.shared).active.is_closed()
    }

    /// Close every connection and stop the tasks.
    pub fn close(&self) {
        let mut shared = lock(&self.shared);
        shared.active.close(0, "client closing");
        if let Some(pending) = shared.pending.take() {
            pending.conn.close(0, "client closing");
        }
        for task in shared.tasks.drain(..) {
            task.abort();
        }
    }
}

impl Drop for ServerLink {
    fn drop(&mut self) {
        self.close();
    }
}

/// Say hello at `addr` as the player `player_token`.
async fn hello_at(addr: SocketAddr, player_token: &str) -> Result<Joined, NetError> {
    let endpoint = client_endpoint_for(addr).map_err(|e| NetError::Handshake(e.to_string()))?;
    join(&endpoint, addr, SERVER_NAME, &hello(player_token)).await
}

/// Where a player dials `shard` according to `map`, when it entered through
/// `entry`: the shard's client endpoint, an endpoint without an IP meaning
/// `entry`'s host ([`resolve_client_endpoint`]). `None` when the map names no
/// address for the shard.
fn client_address(map: &ShardMap, shard: ShardId, entry: SocketAddr) -> Option<SocketAddr> {
    resolve_client_endpoint(map.client_endpoint_of(shard)?, entry.ip())
}

/// Say hello at `entry` and follow what it answers until a shard welcomes
/// the player (see [`ServerLink::join`]).
async fn join_through(entry: SocketAddr, player_token: &str) -> Result<(Conn, Welcome), JoinError> {
    let mut at = entry;
    let mut sent_on: Vec<ShardId> = Vec::new();
    loop {
        let fail = |failure| JoinError { at, failure };
        match hello_at(at, player_token).await {
            Ok(Joined::Shard(conn, welcome)) => {
                if at != entry {
                    log::info!(
                        "space-client: {entry} sent this client to shard {} at {at}",
                        welcome.shard.0
                    );
                }
                return Ok((conn, welcome));
            }
            Ok(Joined::Elsewhere(redirect)) => {
                sent_on.push(redirect.shard);
                if sent_on.len() > MAX_JOIN_REDIRECTS {
                    return Err(fail(JoinFailure::TooManyRedirects(sent_on)));
                }
                at = client_address(&redirect.shard_map, redirect.shard, entry)
                    .ok_or_else(|| fail(JoinFailure::NoAddress(redirect.shard)))?;
            }
            Err(error) => return Err(fail(JoinFailure::Net(error))),
        }
    }
}

/// Ticks of simulation time in `elapsed`.
fn ticks_in(elapsed: Duration) -> u64 {
    (elapsed.as_secs_f64() / f64::from(space_sim::dt())) as u64
}

/// The tick an input batch sent now should be tagged with: the last tick the
/// server reported, advanced by the time since (bounded), plus the lead.
fn input_tick(last_frame_tick: u64, since_frame: Duration) -> u64 {
    last_frame_tick + ticks_in(since_frame).min(MAX_TICK_EXTRAPOLATION) + INPUT_LEAD
}

/// One [`InputTick`] per simulation tick, carrying the current thrust intent,
/// to whichever connection is active.
async fn send_inputs(shared: SharedRef) {
    let mut clock = tokio::time::interval(Duration::from_secs_f64(f64::from(space_sim::dt())));
    clock.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        clock.tick().await;
        let (conn, batch, also) = {
            let mut s = lock(&shared);
            if s.active.is_closed() {
                return;
            }
            // Give up on a redirect whose handoff never completed.
            if s.pending
                .as_ref()
                .is_some_and(|p| p.opened.elapsed() > REDIRECT_PATIENCE)
            {
                if let Some(stale) = s.pending.take() {
                    stale.conn.close(0, "handoff did not complete");
                }
            }
            // Nothing to target until the first frame names our ship and tick.
            let Some(entity) = s.replica.ship else {
                continue;
            };
            // One input for every tick, and none twice.
            let estimate = input_tick(s.replica.tick, s.tick_seen.elapsed());
            let Some((first, tick)) = inputs_for(estimate, s.input_sent) else {
                continue;
            };
            s.input_sent = Some(tick);
            s.seq += 1;
            let inputs: Vec<PlayerInput> = (first..=tick)
                .map(|tick| PlayerInput {
                    entity,
                    thrust: s.thrust,
                    torque: [0.0; 3],
                    tick,
                    fast: s.fast,
                })
                .collect();
            s.sent.extend(inputs.iter().copied());
            while s.sent.len() > SENT_LOG {
                s.sent.pop_front();
            }
            s.recent.extend(inputs.iter().copied());
            while s.recent.len() > RESEND_LOG {
                s.recent.pop_front();
            }
            let batch = InputTick {
                tick,
                seq: s.seq,
                inputs,
                ack_tick: s.replica.tick,
            };
            // A handoff in flight: the shard that is taking the ship over
            // gets the same input (see the module docs, "One timeline across
            // the switch"), acknowledging what *it* has sent so far.
            let also = s.pending.as_ref().filter(|p| !p.conn.is_closed()).map(|p| {
                let batch = InputTick {
                    ack_tick: p.tick,
                    ..batch.clone()
                };
                (p.conn.clone(), batch)
            });
            (s.active.clone(), batch, also)
        };
        // A failed send on a live connection is a lost input, like any other.
        let _ = conn.send(&Message::InputTick(batch)).await;
        if let Some((conn, batch)) = also {
            let _ = conn.send(&Message::InputTick(batch)).await;
        }
    }
}

fn spawn_reader(
    shared: &SharedRef,
    player_token: &Arc<str>,
    conn: Conn,
    generation: u64,
    datagrams: bool,
) -> JoinHandle<()> {
    tokio::spawn(read_frames(
        Arc::clone(shared),
        Arc::clone(player_token),
        conn,
        generation,
        datagrams,
    ))
}

/// Read frames from one connection's reliable stream (snapshots, redirects) or
/// datagrams (deltas) until it closes.
async fn read_frames(
    shared: SharedRef,
    player_token: Arc<str>,
    conn: Conn,
    generation: u64,
    datagrams: bool,
) {
    loop {
        let message = if datagrams {
            conn.recv_datagram().await
        } else {
            conn.recv_stream().await
        };
        match message {
            Ok(Some(Message::Redirect(redirect))) => {
                follow_redirect(&shared, &player_token, generation, redirect);
            }
            // The shard's last word to a client it has let go of: this
            // connection is over (the shard closes it too). It names where a
            // new player starts — or, while it does not know that, says
            // "starting" (a `Reject`): the same last word without an address.
            Ok(Some(Message::JoinRedirect(redirect))) => {
                let_go(&shared, generation, Some(redirect.shard));
                return;
            }
            Ok(Some(Message::Reject(_))) => {
                let_go(&shared, generation, None);
                return;
            }
            Ok(Some(Message::SceneUpdate(update))) => on_scene(&shared, generation, update),
            Ok(Some(Message::SceneClock(clock))) => on_scene_clock(&shared, generation, clock),
            Ok(Some(message)) => on_frame(&shared, generation, &message),
            // Peer gone. (A handoff's connection that ended is no longer a
            // way to its shard: forget it, so the next redirect dials anew.)
            Ok(None) => {
                forget_closed_pending(&shared, generation);
                return;
            }
            // A bad frame ends nothing; the connection state is the authority
            // on whether to carry on.
            Err(e) => {
                log::warn!("space-client: server frame: {e}");
                if conn.is_closed() {
                    forget_closed_pending(&shared, generation);
                    return;
                }
            }
        }
    }
}

/// The shard on the connection with reader `generation` has let go of this
/// client — a `JoinRedirect` on an established connection, naming `owner` as
/// where a new player starts, or a `Reject` (`owner` unknown: "starting"):
/// it had let the client in for a handoff that never completed.
///
/// For the connection of a handoff in flight that is the end of that attempt
/// and nothing more — the old shard was never given up, and offers the ship
/// again if it still should go. For the connection being played on it is the
/// end of the session: the link closes, and whoever keeps it up (the app,
/// for a client started with a server address) joins again through the
/// entry address.
fn let_go(shared: &SharedRef, generation: u64, owner: Option<ShardId>) {
    let mut s = lock(shared);
    if generation == s.active_generation {
        let starts = owner.map_or_else(
            || "the cluster is starting".to_owned(),
            |owner| format!("a new player starts on shard {}", owner.0),
        );
        log::warn!(
            "space-client: the shard let go of this client ({starts}); the connection is closed"
        );
        s.active.close(0, "let go");
        return;
    }
    if s.pending
        .as_ref()
        .is_some_and(|p| p.generation == generation)
    {
        if let Some(pending) = s.pending.take() {
            log::info!(
                "space-client: shard {} did not take the ship over; staying where it is",
                pending.shard.0
            );
            pending.conn.close(0, "handoff did not complete");
        }
    }
}

/// The connection with reader `generation` has ended. If it was the
/// connection of a handoff in flight, that attempt is over: the entry is
/// dropped, so a later redirect to the same shard opens a new connection
/// instead of "reporting in again" on a dead one (which would keep every
/// further handoff to that shard from completing until the entry's patience
/// ran out, ten seconds later).
fn forget_closed_pending(shared: &SharedRef, generation: u64) {
    let mut s = lock(shared);
    if s.pending
        .as_ref()
        .is_some_and(|p| p.generation == generation && p.conn.is_closed())
    {
        s.pending = None;
    }
}

/// Take a scene announcement from the connection with reader `generation`: the
/// shard being played on, or the accepting shard of a handoff in flight. One
/// from any other connection (already retired, or superseded) is not ours.
fn on_scene(shared: &SharedRef, generation: u64, update: SceneState) {
    let mut s = lock(shared);
    let from_active = generation == s.active_generation;
    if !from_active {
        let Some(pending) = s.pending.as_mut().filter(|p| p.generation == generation) else {
            return;
        };
        // Remembered for the switch; meanwhile only a newer revision counts.
        if pending.scene.is_none_or(|held| update.supersedes(&held)) {
            pending.scene = Some(update);
        }
    }
    if s.replica.adopt_scene(update, from_active) {
        log_scene(update);
    }
}

/// Take a scene clock from the connection with reader `generation`: the
/// shard being played on's at once, the accepting shard's of a handoff kept
/// for the switch.
fn on_scene_clock(shared: &SharedRef, generation: u64, clock: SceneClock) {
    let mut s = lock(shared);
    if generation == s.active_generation {
        s.replica.scene_clock = Some(clock);
    } else if let Some(pending) = s.pending.as_mut().filter(|p| p.generation == generation) {
        pending.scene_clock = Some(clock);
    }
}

/// Say which scene the server chose — and, for one this build does not know,
/// what is drawn instead.
fn log_scene(scene: SceneState) {
    match scene.scene.known() {
        Some(known) => log::info!(
            "space-client: the server's scene is {known} (revision {})",
            scene.revision
        ),
        None => log::warn!(
            "space-client: the server's scene (id {}, revision {}) is not known to this build; \
             showing {} instead",
            scene.scene.0,
            scene.revision,
            scene.scene.or_default()
        ),
    }
}

/// Apply a frame from the connection with reader `generation`.
fn on_frame(shared: &SharedRef, generation: u64, message: &Message) {
    let mut s = lock(shared);
    if generation == s.active_generation {
        let before = s.replica.tick;
        s.replica.apply(message);
        if s.replica.tick != before {
            let now = Instant::now();
            s.tick_seen = now;
            s.replica.tick_seen = Some(now);
        }
        return;
    }
    // A frame from the accepting shard of a handoff in flight. Until it shows
    // our ship under the new session the old shard is still the authority.
    let flying = s.replica.ship;
    let Some(pending) = s.pending.as_mut().filter(|p| p.generation == generation) else {
        return;
    };
    pending.ships.apply(message);
    let heard = match message {
        Message::Snapshot(frame) => Some(frame.tick),
        Message::Delta(frame) => Some(frame.tick),
        _ => None,
    };
    pending.tick = pending.tick.max(heard.unwrap_or(0));
    // The ship we have been flying — or, before any frame has named it, the
    // one the redirect was about.
    let ours = flying.unwrap_or(pending.entity);
    if !Replica::frame_has_ship_of(message, pending.session, ours) {
        // A *different* ship under our new session: this shard did not take
        // us for the player of the ship being handed over (it had dropped
        // the transfer by the time we said hello) and let us in as a
        // newcomer, with a ship on the spawn point. That connection can
        // never become the switch, and while it lives there are two ships
        // of this player. Close it now; the shard removes the ship with it.
        // The old shard was never given up, and offers the ship again.
        if Replica::frame_has_other_ship_of(message, pending.session, ours) {
            if let Some(pending) = s.pending.take() {
                log::info!(
                    "space-client: shard {} let this client in as a new player instead of \
                     taking the ship over; leaving it again",
                    pending.shard.0
                );
                pending.conn.close(0, "not a handoff");
            }
        }
        return;
    }
    // The switch: the new shard is simulating our ship.
    let Some(pending) = s.pending.take() else {
        return;
    };
    let retired = std::mem::replace(&mut s.active, pending.conn);
    s.active_generation = pending.generation;
    // Does this shard count the ticks the old one counted? Its frame's tick
    // against the tick the link expects now: on one cluster clock they are a
    // frame's delivery apart.
    let expected = s.replica.tick + ticks_in(s.tick_seen.elapsed()).min(MAX_TICK_EXTRAPOLATION);
    let continuous =
        s.replica.tick > 0 && heard.is_some_and(|tick| tick.abs_diff(expected) <= SWITCH_TOLERANCE);
    let before = s.replica.tick;
    s.replica
        .switch_to(pending.session, pending.shard, pending.ships, continuous);
    s.replica.apply(message);
    if !continuous || s.replica.tick != before {
        let now = Instant::now();
        s.tick_seen = now;
        s.replica.tick_seen = Some(now);
    }
    if !continuous {
        // What was sent to the old shard is in its clock, not this one's.
        s.sent.clear();
        s.recent.clear();
        s.input_sent = None;
        log::info!(
            "space-client: shard {} counts other ticks than the shard before it (tick {} where \
             {expected} was expected); starting over",
            pending.shard.0,
            heard.unwrap_or(0)
        );
    }
    // What the new shard announced while the handoff was in flight now comes
    // from the shard being played on. (A newer revision was taken when it
    // arrived; an older one still changes nothing.)
    if let Some(theirs) = pending.scene {
        if s.replica.adopt_scene(theirs, true) {
            log_scene(theirs);
        }
    }
    // The new shard's ticks come with its own mapping onto scene time.
    if pending.scene_clock.is_some() {
        s.replica.scene_clock = pending.scene_clock;
    }
    log::info!(
        "space-client: now on shard {} (session {}), transition {}",
        pending.shard.0,
        pending.session,
        s.replica.handoffs
    );
    drop(s);
    retired.close(0, "handed off");
}

/// Start following `redirect` (from the active connection only): connect to
/// the accepting shard and report in. Never blocks the reader.
fn follow_redirect(
    shared: &SharedRef,
    player_token: &Arc<str>,
    generation: u64,
    redirect: Redirect,
) {
    let generation_for_new = {
        let mut s = lock(shared);
        if generation != s.active_generation || s.following {
            return;
        }
        if s.replica.shard == Some(redirect.new_shard) {
            return;
        }
        // Only the ship this client flies is followed anywhere (see the
        // module docs). Before the first frame has named it, the redirect is
        // taken at its word: a shard redirects a ship's own player only.
        if s.replica.ship.is_some_and(|ours| ours != redirect.entity) {
            log::debug!(
                "space-client: ignoring a redirect for {:?}, which is not this client's ship",
                redirect.entity
            );
            return;
        }
        // A connection to that shard that has ended is not a way there.
        if s.pending.as_ref().is_some_and(|p| p.conn.is_closed()) {
            s.pending = None;
        }
        // Already connected there: the shards re-offered (the first attempt
        // timed out or was refused), so just report in again.
        if let Some(pending) = s.pending.as_ref().filter(|p| p.shard == redirect.new_shard) {
            let conn = pending.conn.clone();
            let status = neighbour_status(&redirect);
            let task = tokio::spawn(async move {
                let _ = conn.send_reliable(&status).await;
            });
            s.track(task);
            return;
        }
        s.following = true;
        let generation = s.next_generation;
        s.next_generation += 1;
        generation
    };
    let (shared_task, token) = (Arc::clone(shared), Arc::clone(player_token));
    let task = tokio::spawn(async move {
        let outcome = open_redirect(&shared_task, &token, generation_for_new, &redirect).await;
        lock(&shared_task).following = false;
        if let Err(reason) = outcome {
            // Make-before-break: the old connection was never given up, so a
            // failed redirect costs nothing; the shards abort the handoff.
            log::warn!(
                "space-client: could not follow the redirect to shard {:?}: {reason}",
                redirect.new_shard
            );
        }
    });
    lock(shared).track(task);
}

fn neighbour_status(redirect: &Redirect) -> Message {
    Message::NeighbourStatus(NeighbourStatus {
        entity: redirect.entity,
        shard: redirect.new_shard,
        connected: true,
    })
}

/// Connect to the shard `redirect` names, register the connection as pending
/// and tell that shard we are there.
async fn open_redirect(
    shared: &SharedRef,
    player_token: &Arc<str>,
    generation: u64,
    redirect: &Redirect,
) -> Result<(), String> {
    // Where players reach the accepting shard: its client endpoint in the
    // redirect's map, one without an IP meaning the host this client entered
    // through — the rule a join redirect follows.
    let entry = lock(shared).entry;
    let addr = client_address(&redirect.shard_map, redirect.new_shard, entry)
        .ok_or("the shard map has no reachable endpoint for it")?;
    // One hello. The offering shard sends the redirect only after the
    // accepting shard has staged the ship and said so (`HandoffAccept`), so
    // that shard knows whose hello this is. (While the redirect went out
    // together with the offer the hello could get there first, and the link
    // asked up to five times; that order no longer exists.) What is left is
    // a transfer the accepting shard has dropped again in between — an abort
    // or a map change that crossed our hello. Then it answers as it answers
    // a fresh join, and that is the end of this attempt, not something to
    // ask again about: nothing was given up, and the old shard offers the
    // ship anew (with a new redirect) if it still should go.
    let (conn, welcome) = match hello_at(addr, player_token).await {
        Ok(Joined::Shard(conn, welcome)) => (conn, welcome),
        Ok(Joined::Elsewhere(sent_on)) => {
            return Err(format!(
                "{addr} does not take this client (a new player starts on shard {})",
                sent_on.shard.0
            ));
        }
        Err(e) => return Err(format!("{addr}: {e}")),
    };
    let tail = {
        let mut s = lock(shared);
        if let Some(previous) = s.pending.replace(Pending {
            conn: conn.clone(),
            generation,
            session: welcome.session,
            shard: redirect.new_shard,
            entity: redirect.entity,
            opened: Instant::now(),
            tick: 0,
            scene: None,
            scene_clock: None,
            ships: ShipView::default(),
        }) {
            previous.conn.close(0, "superseded");
        }
        for datagrams in [false, true] {
            let reader = spawn_reader(shared, player_token, conn.clone(), generation, datagrams);
            s.track(reader);
        }
        // From here on the sender gives this shard every batch too. What
        // went out just before, for ticks still to come, it has not heard.
        let tail = resend_tail(&s.recent, s.replica.tick);
        s.seq += 1;
        tail.last().map(|last| InputTick {
            tick: last.tick,
            seq: s.seq,
            inputs: tail.clone(),
            ack_tick: 0,
        })
    };
    // Before the status that opens the commit's gate: the inputs for the
    // ticks the commit may fall on ([`resend_tail`]).
    if let Some(batch) = tail {
        let _ = conn.send(&Message::InputTick(batch)).await;
    }
    // The gate the shards' commit waits for.
    conn.send_reliable(&neighbour_status(redirect))
        .await
        .map_err(|e| format!("{addr}: reporting in: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use space_protocol::Snapshot;

    /// Every tick gets one input: the sender continues where it stopped,
    /// fills what the clock skipped, and waits when it is a tick ahead.
    ///
    /// Mutation this catches: tagging each batch with the clock's estimate
    /// alone (a tick the estimate skips gets no input and the ship coasts
    /// through it; one it names twice gets two).
    #[test]
    fn every_tick_gets_exactly_one_input() {
        assert_eq!(inputs_for(100, None), Some((100, 100)), "the first");
        assert_eq!(inputs_for(101, Some(100)), Some((101, 101)), "in step");
        assert_eq!(inputs_for(100, Some(100)), Some((101, 101)), "one ahead");
        assert_eq!(inputs_for(100, Some(101)), None, "not two ahead");
        assert_eq!(inputs_for(103, Some(100)), Some((101, 103)), "filled");
        assert_eq!(
            inputs_for(200, Some(100)),
            Some((197, 200)),
            "after a stall: the last few, not a hundred"
        );
        // Whatever the estimates do (a jittery clock: repeats, skips), the
        // ticks sent are consecutive.
        let estimates = [10, 10, 12, 13, 13, 13, 16, 17, 19, 19, 20, 22, 22, 23];
        let (mut sent, mut all) = (None, Vec::new());
        for estimate in estimates {
            if let Some((first, last)) = inputs_for(estimate, sent) {
                all.extend(first..=last);
                sent = Some(last);
            }
        }
        assert_eq!(
            all,
            (10..=24).collect::<Vec<u64>>(),
            "consecutive, one ahead at most"
        );
    }

    /// What a shard that has just been connected to for a handoff is told
    /// again: the inputs already sent for ticks that are still to come — and
    /// nothing older.
    ///
    /// Mutation this catches: saying nothing again (the inputs tagged up to
    /// the lead ahead before the connection existed reach only the old
    /// shard: a commit within the lead leaves the new shard without the
    /// ticks right after it), saying everything again (ticks long stepped).
    #[test]
    fn a_shard_joined_for_a_handoff_is_told_the_inputs_still_to_come() {
        let ship = EntityGuid::new(ShardId::new(1), 1);
        let input = |tick| PlayerInput {
            entity: ship,
            thrust: [1.0, 0.0, 0.0],
            torque: [0.0; 3],
            tick,
            fast: false,
        };
        // The newest frame is of tick 100; inputs were sent up to 103 (the
        // lead), and the log still holds older ones.
        let recent: VecDeque<PlayerInput> = (92..=103).map(input).collect();
        let tail = resend_tail(&recent, 100);
        assert_eq!(
            tail.iter().map(|i| i.tick).collect::<Vec<_>>(),
            vec![101, 102, 103]
        );
        assert!(
            resend_tail(&recent, 103).is_empty(),
            "nothing still to come"
        );
        assert!(resend_tail(&VecDeque::new(), 100).is_empty());
        assert!(
            RESEND_LOG as u64 >= INPUT_LEAD + MAX_INPUT_FILL,
            "the log covers what one batch can be ahead"
        );
    }

    fn chunk(tick: u64, entities: Vec<EntitySnapshot>, more: bool) -> Message {
        let Message::Snapshot(mut snapshot) = frame(tick, entities) else {
            unreachable!("`frame` builds a snapshot");
        };
        snapshot.more = more;
        Message::Snapshot(snapshot)
    }

    /// A snapshot that arrives in three chunks is one snapshot: every entity
    /// of every chunk is held afterwards, and what none of them named is
    /// dropped — when the last chunk arrives, not before.
    ///
    /// Mutation this catches: taking every chunk for the whole interest set
    /// (after the third chunk only its entities are left — and the shard,
    /// which counts all of them as known, never sends the others again while
    /// they rest), never ending a snapshot (the stale entity stays for good).
    #[test]
    fn a_snapshot_in_three_chunks_is_one_snapshot() {
        let guid = |n| EntityGuid::new(ShardId::new(1), n);
        let mut r = Replica::new(7, ShardId::new(1));
        // Held from before: the own ship and something the new snapshot will
        // not name.
        r.apply(&chunk(
            10,
            vec![ship(guid(1), 7, 0.0), ship(guid(99), 0, 5.0)],
            false,
        ));
        assert_eq!(r.tracked().count(), 2);
        // A new snapshot, a tick per chunk.
        r.apply(&chunk(
            20,
            vec![ship(guid(1), 7, 1.0), ship(guid(2), 0, 2.0)],
            true,
        ));
        r.apply(&chunk(
            21,
            vec![ship(guid(3), 0, 3.0), ship(guid(4), 0, 4.0)],
            true,
        ));
        assert!(
            r.entity(guid(99)).is_some(),
            "nothing is dropped before the snapshot is complete"
        );
        r.apply(&chunk(22, vec![ship(guid(5), 0, 5.0)], false));
        let mut held: Vec<u64> = r.tracked().map(|t| t.ship.guid.serial).collect();
        held.sort_unstable();
        assert_eq!(
            held,
            vec![1, 2, 3, 4, 5],
            "every chunk's entities, no other"
        );
        assert_eq!(r.ship, Some(guid(1)), "the own ship among them");
        // The next snapshot starts afresh: one chunk, and it is the whole.
        r.apply(&chunk(30, vec![ship(guid(1), 7, 1.5)], false));
        assert_eq!(r.tracked().count(), 1);
    }

    #[test]
    fn server_name_matches_the_shards_certificate() {
        assert_eq!(SERVER_NAME, space_server::net::SERVER_NAME);
    }

    fn frame(tick: u64, entities: Vec<EntitySnapshot>) -> Message {
        Message::Snapshot(Snapshot {
            tick,
            shard: ShardId::new(1),
            epoch: 1,
            entities,
            input_ack: 0,
            more: false,
        })
    }

    /// A replicated ship owned by `session`, `x` metres along +X.
    fn ship(guid: EntityGuid, session: u64, x: f64) -> EntitySnapshot {
        let state = ShipState {
            pos: WorldPos::from_f64_metres_around(WorldPos::ZERO, [x, 0.0, 0.0]),
            ..ShipState::IDENTITY
        };
        EntitySnapshot {
            guid,
            components: vec![
                (
                    <Owner as ComponentDef>::ID,
                    space_protocol::codec::encode_raw(&session),
                ),
                (
                    <ShipState as ComponentDef>::ID,
                    space_sim::wire::encode_ship_state(&state),
                ),
            ],
        }
    }

    #[test]
    fn a_frame_without_our_ship_is_ignored() {
        let mut r = Replica::new(7, ShardId::new(1));
        assert!(!r.apply(&frame(5, vec![])));
        assert_eq!((r.tick, r.ship, r.offset()), (5, None, [0.0; 3]));
    }

    /// Where a redirect — a join redirect or a handoff's — is dialed: the
    /// named shard's **client** endpoint, never its internal one; without an
    /// IP it is the host of the address the player dialed first, with the
    /// endpoint's port; a port of 0 is no address at all.
    ///
    /// Mutation this catches: dialing the internal endpoint (unreachable
    /// from outside the cluster's host), dialing `0.0.0.0`, or resolving
    /// against anything but the entry address.
    #[test]
    fn a_redirect_is_dialed_at_the_client_endpoint_on_the_entry_host() {
        let entry: SocketAddr = "203.0.113.7:7777".parse().expect("an address");
        let node = space_world::NodeRef::new(40, [0, 0, 0]);
        let region = |shard: u32, internal: &str, players: &str| {
            space_protocol::ShardRegion::new(
                node,
                ShardId::new(shard),
                internal.parse().expect("an address"),
            )
            .with_client_endpoint(players.parse().expect("an address"))
        };
        let map = ShardMap::new(
            1,
            vec![
                region(1, "127.0.0.1:7870", "0.0.0.0:7870"),
                region(2, "127.0.0.1:7871", "198.51.100.9:7871"),
                region(3, "127.0.0.1:7872", "0.0.0.0:0"),
                region(4, "10.0.0.4:7873", "[::]:7873"),
            ],
        );
        let dialed = |shard: u32| client_address(&map, ShardId::new(shard), entry);
        assert_eq!(dialed(1), Some("203.0.113.7:7870".parse().unwrap()));
        assert_eq!(dialed(2), Some("198.51.100.9:7871".parse().unwrap()));
        assert_eq!(dialed(3), None, "not known yet");
        assert_eq!(dialed(4), Some("203.0.113.7:7873".parse().unwrap()));
        assert_eq!(dialed(5), None, "not in the map");
        // Entered over IPv6: the same rule.
        let entry6: SocketAddr = "[2001:db8::7]:7777".parse().expect("an address");
        assert_eq!(
            client_address(&map, ShardId::new(1), entry6),
            Some("[2001:db8::7]:7870".parse().unwrap())
        );
        // A map built with `ShardRegion::new` has one address per shard.
        let plain = ShardMap::new(
            1,
            vec![space_protocol::ShardRegion::new(
                node,
                ShardId::new(1),
                "127.0.0.1:7778".parse().expect("an address"),
            )],
        );
        assert_eq!(
            client_address(&plain, ShardId::new(1), entry),
            Some("127.0.0.1:7778".parse().unwrap())
        );
    }

    #[test]
    fn tokens_are_unique_per_link() {
        let (a, b) = (unique_player_token(), unique_player_token());
        assert!(a.starts_with(PLAYER_TOKEN_PREFIX) && b.starts_with(PLAYER_TOKEN_PREFIX));
        assert_ne!(a, b, "two links must never present the same token");
    }

    /// The join snapshot may arrive after a first (empty) delta with a higher
    /// tick: it is still the frame that names our ship, so it must count — but
    /// an older frame must never move the ship backwards.
    #[test]
    fn an_older_frame_still_names_the_ship_but_never_rewinds_it() {
        let guid = EntityGuid::new(ShardId::new(1), 1);
        let mut r = Replica::new(7, ShardId::new(1));
        r.apply(&frame(12, vec![]));
        assert!(r.apply(&frame(10, vec![ship(guid, 7, 0.0)])));
        assert_eq!((r.ship, r.tick), (Some(guid), 12));
        r.apply(&frame(20, vec![ship(guid, 7, 5.0)]));
        r.apply(&frame(15, vec![ship(guid, 7, 2.0)]));
        assert!((r.offset()[0] - 5.0).abs() < 1e-3, "{:?}", r.offset());
        // Somebody else's ship is not ours.
        assert!(!r.apply(&frame(21, vec![ship(guid, 8, 99.0)])));
        assert!((r.offset()[0] - 5.0).abs() < 1e-3);
    }

    /// After a handoff the view continues on the new shard: new session, new
    /// tick domain (lower ticks are accepted), same ship, same origin — so the
    /// offset is continuous across the switch.
    #[test]
    fn switching_shards_keeps_the_ship_and_its_origin() {
        let guid = EntityGuid::new(ShardId::new(1), 1);
        let mut r = Replica::new(7, ShardId::new(1));
        r.apply(&frame(1000, vec![ship(guid, 7, 0.0)]));
        r.apply(&frame(1001, vec![ship(guid, 7, 500.0)]));
        // The accepting shard's frame: our ship, its session, its own clock.
        let from_new = frame(40, vec![ship(guid, 3, 512.5)]);
        assert!(!r.apply(&from_new), "not ours until the switch");
        assert!(Replica::frame_has_ship_of(&from_new, 3, guid));
        r.switch_to(3, ShardId::new(2), ShipView::default(), false);
        assert!(r.apply(&from_new));
        assert_eq!(
            (r.session, r.shard, r.handoffs),
            (3, Some(ShardId::new(2)), 1)
        );
        assert_eq!((r.ship, r.tick), (Some(guid), 40));
        assert!((r.offset()[0] - 512.5).abs() < 1e-3, "{:?}", r.offset());
    }

    /// A client whose hello the accepting shard did not recognise (the
    /// transfer was dropped before the hello arrived) is, on the shard that
    /// owns the spawn point, let in as a new player with a ship of its own.
    /// A frame showing that ship is not the switch: the ship being handed
    /// over is still simulated by the old shard, and switching would abandon
    /// it there. It is what tells the link to leave that shard again
    /// (`frame_has_other_ship_of`).
    ///
    /// Mutation this catches: switching on any ship the new session owns —
    /// the player jumps to the accepting shard's spawn point.
    #[test]
    fn a_placeholder_ship_on_the_accepting_shard_is_not_the_switch() {
        let ours = EntityGuid::new(ShardId::new(1), 1);
        let placeholder = EntityGuid::new(ShardId::new(2), 1);
        let new_session = 3;
        let with_placeholder = frame(5, vec![ship(placeholder, new_session, 512.0)]);
        assert!(!Replica::frame_has_ship_of(
            &with_placeholder,
            new_session,
            ours
        ));
        // It is a ship of this session that is not the one flown: the sign
        // of a connection to leave. Other players' ships, and our own under
        // whatever session, are not.
        assert!(Replica::frame_has_other_ship_of(
            &with_placeholder,
            new_session,
            ours
        ));
        let others = frame(5, vec![ship(placeholder, 8, 512.0), ship(ours, 9, 0.0)]);
        assert!(!Replica::frame_has_other_ship_of(
            &others,
            new_session,
            ours
        ));
        let adopted = frame(5, vec![ship(ours, new_session, 0.0)]);
        assert!(!Replica::frame_has_other_ship_of(
            &adopted,
            new_session,
            ours
        ));
        // Our ship still under another session (a ghost of it): not yet either.
        let not_yet_ours = frame(6, vec![ship(ours, 9, 0.0)]);
        assert!(!Replica::frame_has_ship_of(
            &not_yet_ours,
            new_session,
            ours
        ));
        // Our ship, owned by the new session: that is the switch.
        let handed_over = frame(
            7,
            vec![
                ship(placeholder, new_session, 512.0),
                ship(ours, new_session, 1.0),
            ],
        );
        assert!(Replica::frame_has_ship_of(&handed_over, new_session, ours));
        // Not a frame at all.
        assert!(!Replica::frame_has_ship_of(
            &Message::SceneUpdate(SceneState::UNSET),
            new_session,
            ours
        ));
    }

    fn scene(scene: space_protocol::Scene, revision: u64) -> SceneState {
        SceneState::new(scene.id(), revision)
    }

    /// The scene a replica holds only ever moves to a newer revision — whatever
    /// connection the announcement arrives on, in whatever order.
    ///
    /// Mutation this catches: taking the last announcement received (a late
    /// message, or an accepting shard that has not caught up, rolls the scene
    /// back), or `>=` for a pending connection (a shard one is not yet on
    /// overrides the one being played on).
    #[test]
    fn the_scene_follows_the_highest_revision_and_never_rolls_back() {
        use space_protocol::Scene;
        let mut r = Replica::new(7, ShardId::new(1));
        assert_eq!(r.scene, SceneState::UNSET, "nothing heard yet");
        assert_eq!(r.scene.scene.or_default(), Scene::DEFAULT);

        // A shard nobody configured announces "unset": that is what is held.
        assert!(!r.adopt_scene(SceneState::UNSET, true));
        assert!(r.adopt_scene(scene(Scene::Galaxy, 2), true));
        assert_eq!(r.scene, scene(Scene::Galaxy, 2));

        // Repeats and late messages change nothing, on either connection.
        for from_active in [true, false] {
            assert!(!r.adopt_scene(scene(Scene::Galaxy, 2), from_active));
            assert!(!r.adopt_scene(scene(Scene::Materials, 1), from_active));
            assert!(!r.adopt_scene(SceneState::UNSET, from_active));
        }
        assert_eq!(r.scene, scene(Scene::Galaxy, 2));

        // A newer revision counts wherever it is heard: the scene is the
        // cluster's, and the accepting shard may learn of a change first.
        assert!(r.adopt_scene(scene(Scene::Materials, 3), false));
        assert_eq!(r.scene, scene(Scene::Materials, 3));

        // The handoff itself keeps the scene.
        r.switch_to(3, ShardId::new(2), ShipView::default(), false);
        assert_eq!(r.scene, scene(Scene::Materials, 3));
        assert_eq!(r.handoffs, 1);
    }

    /// Two shards that name different scenes at the same revision (standalone
    /// shards with different `--scene` flags): the one being played on decides.
    #[test]
    fn at_the_same_revision_the_shard_being_played_on_decides() {
        use space_protocol::Scene;
        let mut r = Replica::new(7, ShardId::new(1));
        assert!(r.adopt_scene(scene(Scene::Galaxy, 1), true));
        assert!(
            !r.adopt_scene(scene(Scene::Materials, 1), false),
            "not while it is only the accepting shard of a handoff in flight"
        );
        assert_eq!(r.scene, scene(Scene::Galaxy, 1));
        assert!(
            r.adopt_scene(scene(Scene::Materials, 1), true),
            "once it is the shard being played on"
        );
        assert_eq!(r.scene, scene(Scene::Materials, 1));
    }

    /// A scene this build does not know is kept as it was sent (the revision
    /// still orders it) and resolves to the default scene for drawing.
    #[test]
    fn an_unknown_scene_is_kept_and_resolves_to_the_default() {
        use space_protocol::{Scene, SceneId};
        let mut r = Replica::new(7, ShardId::new(1));
        let unknown = SceneState::new(SceneId(4_000), 5);
        assert!(r.adopt_scene(unknown, true));
        assert_eq!(r.scene, unknown);
        assert_eq!(r.scene.scene.known(), None);
        assert_eq!(r.scene.scene.or_default(), Scene::DEFAULT);
        log_scene(unknown); // says so; must not panic
        assert!(
            !r.adopt_scene(scene(Scene::Materials, 4), true),
            "still ordered"
        );
        assert!(r.adopt_scene(scene(Scene::Materials, 6), true));
    }

    fn region(depth: u8, index: [i64; 3], shard: u32) -> space_protocol::ShardRegion {
        space_protocol::ShardRegion::new(
            space_world::NodeRef::new(depth, index),
            ShardId::new(shard),
            "127.0.0.1:7777".parse().expect("an address"),
        )
    }

    /// The anchor is a point of the root region — the centre of a one-cell
    /// root, and for a larger one 6 m, 512 m up and 4 m from the corner where
    /// its cells meet: the spawn point of the shard that owns it in each of
    /// the topologies the dev scripts start, the same for every client
    /// whatever the order of the map's entries, and the same before and after
    /// a split or a merge.
    #[test]
    fn the_anchor_is_a_point_of_the_maps_root_region() {
        use space_world::NodeRef;
        // Beside and above the centre corner of a root of several cells.
        let off_the_corner = |depth, index| {
            WorldPos::from_f64_metres_around(NodeRef::new(depth, index).center(), [6.0, 512.0, 4.0])
        };
        // One standalone shard: its cell's centre, the world origin.
        let single = ShardMap::new(1, vec![region(64, [0, 0, 0], 1)]);
        assert_eq!(map_anchor(&single), Some(WorldPos::ZERO));
        // Two standalone neighbours of one depth: the lower shard id's cell,
        // whichever way round the map lists them (each shard lists itself
        // first).
        let two = ShardMap::new(1, vec![region(64, [0, 0, 0], 1), region(64, [1, 0, 0], 2)]);
        let owt = ShardMap::new(1, vec![region(64, [1, 0, 0], 2), region(64, [0, 0, 0], 1)]);
        assert_eq!(map_anchor(&two), Some(WorldPos::ZERO));
        assert_eq!(map_anchor(&owt), map_anchor(&two));
        // An orchestrated cluster: the root, before a split, after it (a
        // child octant is added, whoever owns it and wherever it is listed)
        // and after the merge that takes it away again.
        let root = region(40, [3, -2, 5], 7);
        let before = ShardMap::new(4, vec![root.clone()]);
        let anchor = map_anchor(&before);
        assert_eq!(anchor, Some(off_the_corner(40, [3, -2, 5])));
        let child = NodeRef::new(41, [7, -4, 11]);
        assert!(space_world::octree::children(root.node)
            .expect("children")
            .contains(&child));
        for split in [
            vec![root.clone(), region(41, [7, -4, 11], 8)],
            vec![region(41, [7, -4, 11], 8), root.clone()],
            // The new shard's id may be lower than the root's (a restarted
            // root): depth decides before the id does.
            vec![region(41, [7, -4, 11], 2), root.clone()],
            vec![
                root.clone(),
                region(41, [7, -4, 11], 8),
                region(42, [14, -8, 22], 9),
                region(41, [6, -4, 10], 10),
            ],
            // Single cells carved out of the root, the one the anchor lies in
            // among them: the anchor is the root's, whoever owns its cell.
            vec![
                region(64, [7 << 23, -3 << 23, 11 << 23], 3),
                root.clone(),
                region(64, [(7 << 23) - 1, -3 << 23, 11 << 23], 4),
            ],
        ] {
            assert_eq!(map_anchor(&ShardMap::new(5, split)), anchor);
        }
        assert_eq!(map_anchor(&ShardMap::new(6, vec![root.clone()])), anchor);
        // The anchor lies strictly inside one cell: the lowest cell of the
        // root's +x+y+z octant, 6 m and 4 m from its faces towards -x and -z
        // and in the middle of its height.
        let at = anchor.expect("an anchor");
        assert_eq!(at.cell, [7 << 23, -3 << 23, 11 << 23]);
        assert_eq!(at.cell, root.node.center().cell, "the centre corner's cell");
        assert_eq!(at.local, [-506.0, 0.0, -508.0]);
        assert!(space_world::octree::contains(
            NodeRef::new(41, [7, -3, 11]),
            at
        ));
        // The handoff test's topology: A owns the node, B an octant of it.
        let carved = ShardMap::new(1, vec![region(63, [0, 0, 0], 1), region(64, [1, 1, 1], 2)]);
        assert_eq!(map_anchor(&carved), Some(off_the_corner(63, [0, 0, 0])));
        assert_eq!(
            map_anchor(&carved),
            Some(WorldPos {
                cell: [1, 1, 1],
                local: [-506.0, 0.0, -508.0]
            }),
            "inside B's cell"
        );
        // No region, no anchor.
        assert_eq!(map_anchor(&ShardMap::new(1, Vec::new())), None);
    }

    /// A link keeps the anchor it was welcomed with: a handoff does not move
    /// it, and without a map it is where the ship first appeared.
    #[test]
    fn a_replica_keeps_its_anchor_across_a_handoff() {
        let map = ShardMap::new(1, vec![region(63, [0, 0, 0], 1), region(64, [1, 1, 1], 2)]);
        let guid = EntityGuid::new(ShardId::new(1), 1);
        let mut r = Replica::joined(7, ShardId::new(1), &map);
        let anchor = map_anchor(&map);
        assert!(anchor.is_some());
        assert_eq!(r.anchor(), anchor, "known before any frame");
        r.apply(&frame(10, vec![ship(guid, 7, 300.0)]));
        assert_eq!(r.anchor(), anchor, "not where the ship appeared");
        r.switch_to(3, ShardId::new(2), ShipView::default(), false);
        r.apply(&frame(4, vec![ship(guid, 3, 400.0)]));
        assert_eq!(r.anchor(), anchor);

        let mut bare = Replica::new(7, ShardId::new(1));
        assert_eq!(bare.anchor(), None);
        bare.apply(&frame(10, vec![ship(guid, 7, 300.0)]));
        assert_eq!(bare.anchor(), bare.origin());
        assert!(bare.anchor().is_some());
    }

    fn delta(tick: u64, entities: Vec<EntitySnapshot>, removed: Vec<EntityGuid>) -> Message {
        Message::Delta(space_protocol::Delta {
            tick,
            base_tick: tick.saturating_sub(1),
            shard: ShardId::new(1),
            epoch: 1,
            entities,
            removed,
            input_ack: 0,
        })
    }

    /// The replica keeps the other ships the shard reports — other players'
    /// and unowned ones — follows their moves, and forgets them when the shard
    /// says so; its own ship is never one of them.
    #[test]
    fn other_ships_are_tracked_and_removed() {
        let shard = ShardId::new(1);
        let (mine, theirs, drone) = (
            EntityGuid::new(shard, 1),
            EntityGuid::new(shard, 2),
            EntityGuid::new(shard, 3),
        );
        let mut r = Replica::new(7, shard);
        assert_eq!(r.others().count(), 0);
        r.apply(&frame(
            10,
            vec![
                ship(mine, 7, 0.0),
                ship(theirs, 8, 5.0),
                ship(drone, 0, 9.0),
            ],
        ));
        let seen: Vec<(EntityGuid, u64)> = r.others().map(|s| (s.guid, s.owner)).collect();
        assert_eq!(seen, vec![(theirs, 8), (drone, 0)]);
        assert_eq!(r.origin(), r.state.map(|s| s.pos));

        // A delta moves one of them; a late one does not move it back.
        r.apply(&delta(12, vec![ship(theirs, 8, 6.0)], vec![]));
        r.apply(&delta(11, vec![ship(theirs, 8, 5.5)], vec![]));
        let x = |r: &Replica, guid| {
            let s = r.others().find(|s| s.guid == guid).expect("still there");
            WorldPos::delta(WorldPos::ZERO, s.state.pos)[0]
        };
        assert!((x(&r, theirs) - 6.0).abs() < 1e-3);

        // An explicit removal; a removal older than the ship's state is stale.
        r.apply(&delta(13, vec![ship(drone, 0, 9.5)], vec![]));
        r.apply(&delta(12, vec![], vec![drone]));
        assert_eq!(r.others().count(), 2, "a stale removal removes nothing");
        r.apply(&delta(14, vec![], vec![drone]));
        assert_eq!(r.others().map(|s| s.guid).collect::<Vec<_>>(), [theirs]);

        // A snapshot is the whole interest set: who is not in it is gone.
        r.apply(&frame(20, vec![ship(mine, 7, 1.0)]));
        assert_eq!(r.others().count(), 0);
        assert!(r.state.is_some(), "our own ship is still ours");
    }

    /// A physics body as a shard replicates it (protocol 5): the full set in
    /// a snapshot, then deltas with only what changed. The replica merges
    /// them, keeps the body out of the ships, decodes its physics once and
    /// records its motion tick by tick.
    #[test]
    fn bodies_are_merged_tracked_and_kept_apart_from_ships() {
        use space_sim::{Body, BodySpawn, ShapeRef, SimWorld, Visual};
        let shard = ShardId::new(1);
        let mut sim = SimWorld::new(shard);
        let crate_guid = EntityGuid::new(shard, 40);
        let start = WorldPos::from_f64_metres_around(WorldPos::ZERO, [3.0, 0.0, 0.0]);
        sim.spawn_body(
            crate_guid,
            BodySpawn::new(
                ShipState {
                    vel: [1.0, 0.0, 0.0],
                    ..ShipState::at_rest(start)
                },
                Body::dynamic(ShapeRef::cuboid([0.6; 3])),
            )
            .with_visual(Visual(space_scene::visual::CRATE)),
        )
        .expect("a crate");
        let full = sim.entity_snapshot(crate_guid).expect("exported");
        let me = EntityGuid::new(shard, 1);
        let mut r = Replica::new(7, shard);
        r.apply(&frame(10, vec![ship(me, 7, 0.0), full.clone()]));
        assert_eq!(r.others().count(), 0, "a body is not a ship");
        let body = r.bodies().next().expect("the crate");
        assert!(body.is_body());
        let physics = Arc::clone(body.physics.as_ref().unwrap());
        assert_eq!(physics.visual, Some(space_scene::visual::CRATE));
        assert_eq!(body.history.len(), 1);

        // The next tick: only the state changed (and the Owner rule does not
        // apply: a body has none).
        sim.step();
        let moved = sim.entity_snapshot(crate_guid).unwrap();
        let state_only: Vec<_> = moved
            .components
            .iter()
            .filter(|(id, _)| *id == ShipState::ID)
            .cloned()
            .collect();
        r.apply(&delta(
            11,
            vec![EntitySnapshot::new(crate_guid, state_only)],
            vec![],
        ));
        let body = r.bodies().next().expect("still a body");
        assert!(
            Arc::ptr_eq(body.physics.as_ref().unwrap(), &physics),
            "decoded once"
        );
        assert_eq!(body.history.len(), 2);
        assert!(
            body.components.iter().any(|(id, _)| *id == Body::ID),
            "merged"
        );
        let x = |s: &Sample| WorldPos::delta(WorldPos::ZERO, s.parts[0].pos)[0];
        assert!(x(&body.history[1]) > x(&body.history[0]));
        // An empty payload removes a component: without its visual the crate
        // is still a body (its rigid body is not a ship's).
        r.apply(&delta(
            12,
            vec![EntitySnapshot::new(
                crate_guid,
                vec![(Visual::ID, Vec::new())],
            )],
            vec![],
        ));
        let body = r.bodies().next().expect("still a body");
        assert_eq!(body.physics.as_ref().unwrap().visual, None);
        assert!(!body.components.iter().any(|(id, _)| *id == Visual::ID));
        // A removal and a mention in the same frame: it stays (a body that
        // changed hands and is shown as its ghost from this tick on).
        r.apply(&delta(13, vec![full], vec![crate_guid]));
        assert_eq!(r.bodies().count(), 1);
        assert_eq!(r.bodies().next().unwrap().history.len(), 4);
        r.apply(&delta(14, vec![], vec![crate_guid]));
        assert_eq!(r.bodies().count(), 0);
    }

    /// A ship marked with the player visual is a player whatever its owner
    /// (the ghost of a player on a neighbouring shard arrives with owner 0);
    /// one without it and without an owner is a ship nobody flies.
    #[test]
    fn the_player_visual_makes_a_ghost_a_player() {
        use space_sim::Visual;
        let shard = ShardId::new(1);
        let (me, ghost, drone) = (
            EntityGuid::new(shard, 1),
            EntityGuid::new(ShardId::new(2), 5),
            EntityGuid::new(shard, 9),
        );
        let mut marked = ship(ghost, 0, 30.0);
        marked.components.push((
            Visual::ID,
            space_sim::body::encode_visual(Visual(space_scene::visual::PLAYER)),
        ));
        let mut r = Replica::new(7, shard);
        r.apply(&frame(
            10,
            vec![ship(me, 7, 0.0), marked, ship(drone, 0, 9.0)],
        ));
        let players: Vec<EntityGuid> = r
            .other_ships()
            .filter(|t| t.is_player())
            .map(|t| t.ship.guid)
            .collect();
        assert_eq!(players, [ghost]);
        assert_eq!(r.others().count(), 2, "both are ships, neither a body");
        assert_eq!(r.bodies().count(), 0);
    }

    /// A shard sends nothing for an entity at rest; when it moves again, the
    /// history holds its rest pose up to the tick before, so the move is not
    /// smeared back over the quiet ticks — and a long gap of a moving entity
    /// starts the history over.
    #[test]
    fn a_resting_entity_holds_until_it_moves() {
        let guid = EntityGuid::new(ShardId::new(1), 3);
        let mut r = Replica::new(7, ShardId::new(1));
        r.apply(&frame(10, vec![ship(guid, 8, 0.0)]));
        r.apply(&frame(20, vec![ship(guid, 8, 1.0)]));
        let t = r.entity(guid).unwrap();
        let ticks: Vec<u64> = t.history.iter().map(|s| s.tick).collect();
        assert_eq!(ticks, [10, 19, 20]);
        let mut moving = ship(guid, 8, 2.0);
        let state = ShipState {
            vel: [5.0, 0.0, 0.0],
            pos: WorldPos::from_f64_metres_around(WorldPos::ZERO, [2.0, 0.0, 0.0]),
            ..ShipState::IDENTITY
        };
        moving.components[1].1 = space_sim::wire::encode_ship_state(&state);
        r.apply(&frame(21, vec![moving.clone()]));
        r.apply(&frame(60, vec![moving]));
        let ticks: Vec<u64> = r
            .entity(guid)
            .unwrap()
            .history
            .iter()
            .map(|s| s.tick)
            .collect();
        assert_eq!(ticks, [60], "a moving entity's long gap starts over");
    }

    /// The inputs the link sends are kept for the prediction, and the frame's
    /// acknowledgements with them.
    #[test]
    fn a_replica_keeps_the_acknowledgements() {
        let guid = EntityGuid::new(ShardId::new(1), 1);
        let mut r = Replica::new(7, ShardId::new(1));
        let mut f = frame(10, vec![ship(guid, 7, 0.0)]);
        if let Message::Snapshot(s) = &mut f {
            s.input_ack = 9;
        }
        r.apply(&f);
        assert_eq!((r.state_tick(), r.input_ack(), r.frame_ack()), (10, 9, 9));
        let mut d = delta(11, vec![], vec![]);
        if let Message::Delta(d) = &mut d {
            d.input_ack = 11;
        }
        r.apply(&d);
        assert_eq!((r.state_tick(), r.input_ack(), r.frame_ack()), (10, 9, 11));
    }

    /// What the accepting shard reported while the handoff was in flight is
    /// the view after the switch (its join snapshot is never repeated).
    #[test]
    fn the_new_shards_ships_are_taken_over_at_the_switch() {
        let (ours, other) = (
            EntityGuid::new(ShardId::new(1), 1),
            EntityGuid::new(ShardId::new(2), 5),
        );
        let mut r = Replica::new(7, ShardId::new(1));
        r.apply(&frame(100, vec![ship(ours, 7, 0.0)]));
        let mut pending = ShipView::default();
        pending.apply(&frame(3, vec![ship(other, 4, 20.0)]));
        // The frame that shows our ship there is an ordinary delta.
        let switch = delta(4, vec![ship(ours, 9, 1.0)], vec![]);
        pending.apply(&switch);
        r.switch_to(9, ShardId::new(2), pending, false);
        r.apply(&switch);
        assert_eq!(r.others().map(|s| s.guid).collect::<Vec<_>>(), [other]);
        assert_eq!(r.ship, Some(ours));
    }

    /// A switch onto a shard that counts the same ticks changes the session
    /// and nothing else: the tick, the ship's last state and its `input_ack`
    /// stand, every entity the new shard reports too keeps the motion the
    /// old one showed, and what only the old shard knew is gone.
    ///
    /// Mutation this catches: zeroing the ticks at every switch (the app's
    /// clock snaps and its prediction is rebuilt — `tick_epoch` is how it
    /// would notice), taking the new shard's view as it is (every body and
    /// player starts a new history: drawn two ticks behind, it stands still
    /// for those two ticks and then jumps), keeping what the new shard never
    /// reported (a ghost that nobody will ever remove).
    #[test]
    fn a_switch_on_one_timeline_keeps_the_ticks_and_the_histories() {
        let (ours, both, old_only, new_only) = (
            EntityGuid::new(ShardId::new(1), 1),
            EntityGuid::new(ShardId::new(1), 2),
            EntityGuid::new(ShardId::new(1), 3),
            EntityGuid::new(ShardId::new(2), 5),
        );
        let mut r = Replica::new(7, ShardId::new(1));
        for tick in 96..=100 {
            let x = tick as f64;
            let mut f = delta(
                tick,
                vec![
                    ship(ours, 7, x),
                    ship(both, 0, 2.0 * x),
                    ship(old_only, 0, 5.0),
                ],
                vec![],
            );
            if let Message::Delta(d) = &mut f {
                d.input_ack = tick;
            }
            r.apply(&f);
        }
        let history = |r: &Replica, guid| -> Vec<u64> {
            r.entity(guid)
                .map(|t| t.history.iter().map(|s| s.tick).collect())
                .unwrap_or_default()
        };
        assert_eq!(history(&r, both), [96, 97, 98, 99, 100]);
        // The new shard's first frame: the tick after, our ship under its
        // session, the entity both report as its ghost, one of its own.
        let mut switch = frame(
            101,
            vec![
                ship(ours, 9, 101.0),
                ship(both, 0, 202.0),
                ship(new_only, 4, 0.0),
            ],
        );
        if let Message::Snapshot(s) = &mut switch {
            s.input_ack = 100;
        }
        let mut pending = ShipView::default();
        pending.apply(&switch);
        r.switch_to(9, ShardId::new(2), pending, true);
        assert_eq!((r.tick, r.state_tick(), r.input_ack()), (100, 100, 100));
        assert_eq!((r.tick_epoch, r.handoffs, r.session), (0, 1, 9));
        assert!(
            r.apply(&switch),
            "the frame names our ship under the new session"
        );
        assert_eq!((r.tick, r.state_tick(), r.input_ack()), (101, 101, 100));
        assert_eq!(r.ship, Some(ours));
        assert_eq!(history(&r, both), [96, 97, 98, 99, 100, 101], "one history");
        assert_eq!(history(&r, ours), [96, 97, 98, 99, 100, 101]);
        assert_eq!(history(&r, new_only), [101]);
        assert!(r.entity(old_only).is_none(), "not this shard's to report");
        assert_eq!(
            r.others().map(|s| s.guid).collect::<Vec<_>>(),
            [both, new_only]
        );
        // A sample the old shard had for a tick the new one reports too (or
        // a later one) is not kept beside it.
        let mut later = Replica::new(7, ShardId::new(1));
        for tick in 100..=103 {
            later.apply(&delta(
                tick,
                vec![ship(ours, 7, 0.0), ship(both, 0, tick as f64)],
                vec![],
            ));
        }
        let mut pending = ShipView::default();
        pending.apply(&frame(102, vec![ship(ours, 9, 0.0), ship(both, 0, 102.0)]));
        later.switch_to(9, ShardId::new(2), pending, true);
        assert_eq!(history(&later, both), [100, 101, 102]);
        assert_eq!(later.tick, 103, "the clock does not run backwards");

        // Onto a shard that counts other ticks: everything starts over, and
        // the app is told.
        let mut other = Replica::new(7, ShardId::new(1));
        other.apply(&delta(
            100,
            vec![ship(ours, 7, 0.0), ship(both, 0, 1.0)],
            vec![],
        ));
        let mut pending = ShipView::default();
        let first = frame(5_000, vec![ship(ours, 9, 0.0), ship(both, 0, 1.0)]);
        pending.apply(&first);
        other.switch_to(9, ShardId::new(2), pending, false);
        assert_eq!(
            (other.tick, other.state_tick(), other.tick_epoch),
            (0, 0, 1)
        );
        other.apply(&first);
        assert_eq!(history(&other, both), [5_000]);
    }

    /// Inputs are tagged ahead of the server's clock even when frames stop
    /// arriving for a while, and the extrapolation is bounded.
    #[test]
    fn the_input_clock_keeps_running_between_frames() {
        let dt = f64::from(space_sim::dt());
        assert_eq!(input_tick(100, Duration::ZERO), 100 + INPUT_LEAD);
        assert_eq!(
            input_tick(100, Duration::from_secs_f64(dt * 6.5)),
            100 + 6 + INPUT_LEAD
        );
        assert_eq!(
            input_tick(100, Duration::from_secs(3600)),
            100 + MAX_TICK_EXTRAPOLATION + INPUT_LEAD
        );
    }
}
