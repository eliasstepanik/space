//! The messages themselves, and the [`Message`] enum that wraps them.
//!
//! Every message is a plain data type: `Debug + Clone + PartialEq` so a test can
//! assert `decode(encode(m)) == m`, and serde-derived so the codec stays one
//! line. Variant order in [`Message`] is part of the wire format (bincode
//! encodes the variant index), so **new variants are appended, never inserted**.

use serde::{Deserialize, Serialize};
use space_ecs::ComponentId;

use crate::control::{AdminCommand, AdminReply, DirectoryQuery, DirectoryReply, LoadReport};
use crate::guid::EntityGuid;
use crate::scene::{SceneAck, SceneState};
use crate::shard::{
    EpochAbort, EpochAck, EpochCommit, EpochPrepare, ShardId, ShardMap, ShardRegion,
};

/// One player's input for one tick.
///
/// `tick` is the simulation tick the input applies to (the client predicts it);
/// `seq` is the client's own monotonic sequence number, used to detect gaps and
/// to discard replays after a reconnect.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlayerInput {
    /// Entity the input drives.
    pub entity: EntityGuid,
    /// Thrust in entity-local newtons, `(x, y, z)`.
    pub thrust: [f32; 3],
    /// Torque in entity-local newton-metres, `(x, y, z)`.
    pub torque: [f32; 3],
    /// Tick the input applies to.
    pub tick: u64,
}

/// A batch of inputs for one tick.
///
/// Inputs arrive in batches so a client that sent three packets for tick `n`
/// costs the server one frame, and so a dropped packet is visible as a `seq`
/// gap rather than as a lost tick.
///
/// The server **buffers the batch by `tick`** and applies it when it steps to
/// that tick — never on arrival. That is what makes a late or reordered batch
/// harmless: it lands on the tick it was generated for, or it is stale and gets
/// dropped (the client reconciles). `ack_tick` is the client's side of the same
/// handshake: the last snapshot tick the client has applied, which lets the
/// server stop resending state the client already has.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputTick {
    /// Tick these inputs apply to.
    pub tick: u64,
    /// Client-side sequence number of this batch.
    pub seq: u64,
    /// The inputs, in client order.
    pub inputs: Vec<PlayerInput>,
    /// The tick of the newest snapshot/delta the client has **actually applied**
    /// (0 = none applied yet).
    ///
    /// The server acts on it: the frame for exactly this tick is taken as
    /// delivered, and an older frame that was never acknowledged itself is
    /// presumed lost and its contents are sent again. It must therefore be the
    /// tick of a frame that really arrived — never a locally advanced estimate
    /// of the server's clock, which would vouch for a frame still in flight. A
    /// batch may carry an empty `inputs` list just to deliver this
    /// acknowledgement.
    pub ack_tick: u64,
}

/// One entity's replicated state.
///
/// Components are `(ComponentId, encoded payload)` pairs: the id is
/// `space_ecs::ComponentId` and the payload is that component encoded with the
/// same pinned bincode config as the frame itself (see [`crate::codec`]), so a
/// component can be decoded on demand and an unknown id skipped by a client
/// that does not know the component yet.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntitySnapshot {
    /// Which entity this is. Stable across handoffs.
    pub guid: EntityGuid,
    /// `(ComponentId, encoded payload)` for every replicated component present.
    pub components: Vec<(ComponentId, Vec<u8>)>,
}

impl EntitySnapshot {
    /// Build a snapshot from a guid and its encoded components.
    #[must_use]
    pub fn new(guid: EntityGuid, components: Vec<(ComponentId, Vec<u8>)>) -> Self {
        Self { guid, components }
    }

    /// Encoded size of this snapshot under the pinned config, including its
    /// length prefix inside a `Vec`.
    ///
    /// Computed, not measured: the pinned config is fixed-width (see
    /// [`crate::codec`]), so the size is a plain sum — a guid, the component
    /// count, and per component its id, its payload length and its payload.
    /// The replication pass asks for this once per entity per client per tick,
    /// and encoding the entity into a throwaway buffer just to read its length
    /// was the single largest allocation in that loop. The
    /// `entity_snapshot_encoded_len_is_exact` test pins the sum to the bytes the
    /// codec actually writes.
    #[must_use]
    pub fn encoded_len(&self) -> usize {
        ENTITY_GUID_WIRE_LEN
            + LENGTH_PREFIX_WIRE_LEN
            + self
                .components
                .iter()
                .map(|(_, payload)| COMPONENT_ID_WIRE_LEN + LENGTH_PREFIX_WIRE_LEN + payload.len())
                .sum::<usize>()
    }
}

/// Encoded size of an [`EntityGuid`]: `shard u32 (4) + serial u64 (8)`.
pub(crate) const ENTITY_GUID_WIRE_LEN: usize = 12;
/// Encoded size of a container length prefix (a fixed-width `u64`).
pub(crate) const LENGTH_PREFIX_WIRE_LEN: usize = 8;
/// Encoded size of a [`ComponentId`] (`u16`).
const COMPONENT_ID_WIRE_LEN: usize = 2;

/// A full state dump for a shard at one tick.
///
/// Sent on join, after a handoff and whenever a client's view has drifted too
/// far for deltas to be trustworthy.
///
/// Per `docs/plan.md` P9 a snapshot rides a **reliable, ordered QUIC stream**,
/// not an unreliable datagram: at ~158 B per entity a 1k-entity interest set is
/// ~158 KB, far past what datagrams carry without fragmentation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    /// Simulation tick the snapshot describes.
    pub tick: u64,
    /// Shard that produced it.
    pub shard: ShardId,
    /// Shard-map epoch the snapshot was taken under.
    pub epoch: u64,
    /// Every entity the recipient should now have for this shard.
    pub entities: Vec<EntitySnapshot>,
    /// Highest input tick the server has applied **from this client** (0 = none).
    ///
    /// The client drops every pending input with `tick <= input_ack`: the server
    /// has already stepped those inputs into the state this snapshot describes,
    /// so replaying them on top would double-step the ship. Acknowledging by
    /// *tick* (not by count) is what survives reordering and loss: a client that
    /// sent inputs 10..=14 and gets `input_ack = 12` keeps exactly 13 and 14,
    /// whatever order the server received them in.
    pub input_ack: u64,
}

/// A delta: the entities whose components changed since the recipient's last
/// acknowledged tick.
///
/// Same shape as [`Snapshot`] plus the base tick it applies to: the recipient
/// applies it only if `base_tick` matches what it holds, otherwise it asks for a
/// fresh snapshot. This is the network half of `space-ecs`'s change-tick model
/// (`World::tick` + `Query::changed_since`): the server's replication pass
/// collects exactly the rows whose change tick moved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Delta {
    /// Tick this delta describes.
    pub tick: u64,
    /// Tick the recipient is expected to be at.
    pub base_tick: u64,
    /// Shard that produced it.
    pub shard: ShardId,
    /// Shard-map epoch in effect.
    pub epoch: u64,
    /// Changed entities (protocol 5).
    ///
    /// An entity the recipient does not hold yet (it entered interest, or a
    /// frame that introduced it was lost) arrives with its **full**
    /// replicated component set, exactly as in a [`Snapshot`]. An entity it
    /// already holds lists only what changed since `base_tick`: its
    /// `ShipState` and its `Owner` (if it has one) always, and any other
    /// component whose bytes changed; a component listed with an **empty
    /// payload** was removed (every real component encodes to at least one
    /// byte). A component not listed is unchanged.
    pub entities: Vec<EntitySnapshot>,
    /// Entities the recipient must **remove**: despawned, or left the recipient's
    /// interest range.
    ///
    /// A change-tick query cannot express this — a despawned row is gone, and an
    /// entity that left interest stops being sent rather than being sent as
    /// empty — so removals are listed explicitly. The recipient applies them
    /// after `entities`, and a guid it does not have is ignored (it may never
    /// have received the entity).
    pub removed: Vec<EntityGuid>,
    /// Highest input tick the server has applied from this client (see
    /// [`Snapshot::input_ack`]).
    pub input_ack: u64,
}

/// Phase 1 of a two-phase entity transfer: `from` offers an entity to `to`.
///
/// `state` is boxed so a large handoff payload does not inflate every other
/// `Message` variant (the enum's size is set by the largest variant).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HandoffOffer {
    /// Entity being transferred. Its guid never changes.
    pub entity: EntityGuid,
    /// Shard that currently simulates it.
    pub from: ShardId,
    /// Shard it is being offered to.
    pub to: ShardId,
    /// Epoch the transfer was started under; a later map invalidates it.
    pub epoch: u64,
    /// The entity's state as of `activation_tick`.
    pub state: Box<EntitySnapshot>,
    /// Tick at which `to` starts simulating and `from` stops.
    pub activation_tick: u64,
}

/// Phase 2: `to` accepts and has staged the state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HandoffAccept {
    /// Entity being transferred.
    pub entity: EntityGuid,
    /// Offering shard.
    pub from: ShardId,
    /// Accepting shard.
    pub to: ShardId,
    /// Epoch the transfer was accepted under.
    pub epoch: u64,
    /// Tick the transfer becomes effective.
    pub activation_tick: u64,
}

/// Phase 3: `from` stops simulating; `to` owns the entity from
/// `activation_tick`.
///
/// The offering shard keeps simulating the entity until activation, so the
/// state the accepting shard staged from the [`HandoffOffer`] is already stale
/// by the time the commit arrives. The commit therefore carries the **final**
/// state — the entity's replicated state as of `activation_tick` — and the
/// accepting shard replaces its staged copy with it. (The alternative,
/// forwarding every input between offer and activation, would need a per-entity
/// input journal across shards; shipping the final state is one message and is
/// bit-exact.)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HandoffCommit {
    /// Entity transferred.
    pub entity: EntityGuid,
    /// Shard that gave it up.
    pub from: ShardId,
    /// Shard that owns it now.
    pub to: ShardId,
    /// Epoch the commit was made under.
    pub epoch: u64,
    /// Tick the transfer took effect.
    pub activation_tick: u64,
    /// The entity's final replicated state, as of `activation_tick`.
    pub state: Box<EntitySnapshot>,
}

/// A handoff was abandoned; `from` keeps simulating the entity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HandoffAbort {
    /// Entity that was being transferred.
    pub entity: EntityGuid,
    /// Shard that offered it.
    pub from: ShardId,
    /// Shard that would have taken it.
    pub to: ShardId,
    /// Why the transfer was abandoned.
    pub reason: HandoffReason,
}

/// Why a handoff was abandoned.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HandoffReason {
    /// The target shard is at capacity.
    TargetFull,
    /// `to` did not answer the offer in time.
    OfferTimeout,
    /// `from` did not commit in time.
    CommitTimeout,
    /// The shard map moved to a new epoch mid-transfer.
    EpochChanged,
    /// The entity was despawned while the transfer was in flight.
    Despawned,
    /// The client never confirmed its neighbour connection to the accepting
    /// shard (see [`NeighbourStatus`] and [`Redirect`]).
    ClientNotConnected,
    /// Anything else; for logs only.
    Other(String),
}

/// A client's connection status towards one shard, for make-before-break
/// handoffs.
///
/// Sent client → *accepting* shard (and mirrored to the offering shard by the
/// accepting shard) once the client has opened — or failed to open — the
/// connection the [`Redirect`] asked for. A handoff commit **waits** for
/// `connected = true` for the entity's shard; if the confirmation never arrives
/// the transfer times out and aborts with
/// [`HandoffReason::ClientNotConnected`], so the offering shard keeps
/// simulating and the client keeps its existing view (make-before-break: the
/// old path is never torn down before the new one is proven up).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeighbourStatus {
    /// Entity the status is about (the entity that triggered the redirect).
    pub entity: EntityGuid,
    /// The shard the client connected to (or failed to).
    pub shard: ShardId,
    /// `true` once the connection is open and the client is receiving from it.
    pub connected: bool,
}

/// One tick's border ghosts from one shard to a neighbour (protocol 5).
///
/// A shard keeps, on each neighbour, a **ghost** — a kinematic copy its
/// bodies collide with and its clients see — of every body it simulates near
/// that neighbour's region (and of the ends of springs that cross to it). This
/// message is how: sent after the sender's step whenever something changed,
/// and in full every second (`full`), on the same ordered peer connection as
/// the handoff messages.
///
/// The entity lists follow [`Delta`]'s rule: a ghost the receiver is not
/// known to hold yet — and every entity of a `full` update — comes with its
/// full replicated component set; one it holds lists only its `ShipState`
/// and what else changed (an empty payload removes a component).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GhostUpdate {
    /// Shard that owns (simulates) the entities.
    pub from: ShardId,
    /// Shard-map epoch the sender is on.
    pub epoch: u64,
    /// The sender's tick the states are from (its clock, not the receiver's).
    pub tick: u64,
    /// A full refresh: `entities` is the sender's whole ghost set for the
    /// receiver, with full component lists, and every ghost of the sender's
    /// the receiver holds that is not listed is to be removed.
    pub full: bool,
    /// New or changed ghosts.
    pub entities: Vec<EntitySnapshot>,
    /// Ghosts the receiver must drop (they left the border margin, were
    /// despawned, or were handed to another shard).
    pub removed: Vec<EntityGuid>,
    /// The sender's view of the interaction island each ghosted body is in,
    /// for island consolidation (see `space-server`'s `border` module).
    pub islands: Vec<IslandKey>,
}

/// A shard's view of the interaction island a body is in: the island's
/// member with the lowest guid and the shard the island belongs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IslandKey {
    /// The body.
    pub body: EntityGuid,
    /// The lowest guid of its island, as the sender sees it.
    pub min: EntityGuid,
    /// The shard the sender says that island belongs on.
    pub home: ShardId,
}

/// Tell a client to open a connection to a neighbouring shard for `entity`.
///
/// The new shard map is included so the client never has to ask for it again
/// before following the redirect.
///
/// `activation_tick` is the tick the *new* shard starts being authoritative at.
/// The client must report back with a [`NeighbourStatus`] once its connection to
/// `new_shard` is up; the handoff's commit waits for that (make-before-break),
/// and the client keeps receiving from the old shard until `activation_tick`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Redirect {
    /// Entity the client is now interested in.
    pub entity: EntityGuid,
    /// Shard that simulates it.
    pub new_shard: ShardId,
    /// Map in effect, so the client can find every other region it needs.
    pub shard_map: ShardMap,
    /// Tick at which `new_shard` takes authority for `entity`.
    pub activation_tick: u64,
}

/// The sender is running an older shard-map epoch than the receiver.
///
/// Answered to any message whose `epoch` is behind; the receiver expects the
/// sender to refresh its map (from a [`DirectoryReply`] or a [`Redirect`]) and
/// resend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaleEpoch {
    /// Epoch the sender used.
    pub seen: u64,
    /// Epoch the receiver is on.
    pub current: u64,
}

/// Why a connection was refused.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RejectReason {
    /// The handshake announced a different [`crate::PROTOCOL_VERSION`].
    VersionMismatch {
        /// Version the client announced.
        seen: u32,
        /// Version the server speaks.
        expected: u32,
    },
    /// The player token is not valid.
    Unauthorized,
    /// The shard is at capacity; retry elsewhere.
    Full,
    /// Not now: the shard is draining for a split/merge or shutdown — or, from
    /// the orchestrator to a shard's [`ShardHello`], a map change is in flight
    /// and the hello cannot be served yet. The one refusal that is **not
    /// final**: a shard told this asks again after a short pause.
    Draining,
    /// Anything else; for logs only.
    Other(String),
}

/// A connection was refused, with the reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reject {
    /// Why.
    pub reason: RejectReason,
}

/// First message on every connection.
///
/// `Debug` does not print the token (see the impl): a hello ends up in log
/// lines — "the first message must be a Hello, got ..." — and its token is a
/// credential, a player's or, between shards, the cluster's peer token.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hello {
    /// Protocol version the client speaks.
    pub protocol: u32,
    /// Opaque player token the server authenticates. A shard dialling another
    /// shard puts `shard:<id>` here, followed by `:<peer token>` when its
    /// cluster has one.
    pub player_token: String,
}

impl core::fmt::Debug for Hello {
    /// Everything but the secret part of the token: a player's token is not
    /// printed at all, a peer hello keeps the shard id it names
    /// (`shard:7:<redacted>`).
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Hello")
            .field("protocol", &self.protocol)
            .field("player_token", &RedactedToken(&self.player_token))
            .finish()
    }
}

/// A hello's token as `Debug` prints it (see [`Hello`]'s impl).
struct RedactedToken<'a>(&'a str);

impl core::fmt::Debug for RedactedToken<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // `shard:<digits>[:<peer token>]` is a peer hello; the id is not a
        // secret, what follows it is.
        let peer = self.0.strip_prefix("shard:").and_then(|rest| {
            let (id, secret) = rest
                .split_once(':')
                .map_or((rest, None), |(id, secret)| (id, Some(secret)));
            (!id.is_empty() && id.bytes().all(|b| b.is_ascii_digit())).then_some((id, secret))
        });
        match peer {
            Some((id, None)) => write!(f, "shard:{id}"),
            Some((id, Some(secret))) => write!(f, "shard:{id}:{}", redacted(secret)),
            None => f.write_str(redacted(self.0)),
        }
    }
}

/// The connection is accepted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Welcome {
    /// Session id for this connection; echoed on every later message from the
    /// server so a client can drop frames from a stale session.
    pub session: u64,
    /// Shard the client is talking to.
    pub shard: ShardId,
    /// The map in effect, so the client can open neighbour connections.
    pub shard_map: ShardMap,
}

/// Shard → orchestrator: "give me an id and my map" (Phase 3, P3.1).
///
/// The first message on a shard↔orchestrator control connection, after the
/// ordinary [`Hello`] handshake. The normal hello names **neither an id nor a
/// region** (`shard` is [`ShardId::NONE`], `region` is `None`): the
/// orchestrator allocates the id and assigns the region — the root while
/// nobody owns it, otherwise a node it carves for the shard through an epoch
/// change. A hello that names an id or a region is a claim on what the
/// published map already says (the reserved root owner's first registration);
/// it never creates a region.
///
/// The answer is a [`ShardWelcome`], or a [`Reject`]: [`RejectReason::Draining`]
/// means "not yet, ask again" (one map change at a time), anything else is
/// final.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ShardHello {
    /// Cluster the shard is joining; must match the orchestrator's own id.
    pub cluster: String,
    /// The shard's id, or [`ShardId::NONE`] to request a fresh allocation.
    pub shard: ShardId,
    /// Where the shard is listening for peers and clients.
    pub endpoint: std::net::SocketAddr,
    /// Region the shard claims, or `None` for "assign me one".
    pub region: Option<ShardRegion>,
    /// The cluster secret (protocol 6), empty when the shard was given none.
    /// An orchestrator whose control port is not on loopback — or that was
    /// given a secret explicitly — refuses a hello without the right one.
    pub secret: String,
}

impl core::fmt::Debug for ShardHello {
    /// Everything but the secret itself: a hello may end up in a log line.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ShardHello")
            .field("cluster", &self.cluster)
            .field("shard", &self.shard)
            .field("endpoint", &self.endpoint)
            .field("region", &self.region)
            .field("secret", &redacted(&self.secret))
            .finish()
    }
}

/// What a `Debug` impl prints in place of a secret.
fn redacted(secret: &str) -> &'static str {
    if secret.is_empty() {
        "<none>"
    } else {
        "<redacted>"
    }
}

/// Orchestrator → shard: the id, region, map and peer token the shard runs on.
///
/// The answer to [`ShardHello`]. `region` is the region the shard is given;
/// **it owns it only once the map says so**: for the root that is the map in
/// this welcome, for a carved node the map of the epoch the orchestrator
/// proposes right behind the welcome — until that commits `shard_map` names
/// the node's previous owner and the shard owns nothing.
///
/// `peer_token` is the secret every shard of the cluster presents in its hello
/// to another shard (`shard:<id>:<token>`); a shard that holds one refuses a
/// peer hello without it. Empty = the orchestrator issues none and peers are
/// accepted on map membership alone, as between standalone shards.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ShardWelcome {
    /// The id the orchestrator allocated (or confirmed).
    pub shard: ShardId,
    /// The region the shard is given, with the endpoint the orchestrator
    /// recorded.
    pub region: ShardRegion,
    /// The current shard map, so the shard never runs on a stale routing table.
    pub shard_map: ShardMap,
    /// The secret shards present to each other (see the type docs).
    pub peer_token: String,
}

impl core::fmt::Debug for ShardWelcome {
    /// Everything but the peer token itself.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ShardWelcome")
            .field("shard", &self.shard)
            .field("region", &self.region)
            .field("shard_map", &self.shard_map)
            .field("peer_token", &redacted(&self.peer_token))
            .finish()
    }
}

/// Every message that crosses a socket.
///
/// The variant order **is** the wire tag: append new variants, never reorder or
/// remove one, and bump [`crate::PROTOCOL_VERSION`] when the shape of an
/// existing variant changes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Message {
    /// Client → server: handshake.
    Hello(Hello),
    /// Server → client: handshake accepted.
    Welcome(Welcome),
    /// Either direction: connection refused.
    Reject(Reject),
    /// Client → server: input batch.
    InputTick(InputTick),
    /// Server → client: full state.
    Snapshot(Snapshot),
    /// Server → client: changed state.
    Delta(Delta),
    /// Shard → shard: handoff phase 1.
    HandoffOffer(HandoffOffer),
    /// Shard → shard: handoff phase 2.
    HandoffAccept(HandoffAccept),
    /// Shard → shard: handoff phase 3.
    HandoffCommit(HandoffCommit),
    /// Shard → shard: handoff abandoned.
    HandoffAbort(HandoffAbort),
    /// Shard → shard: border ghost replica.
    GhostUpdate(GhostUpdate),
    /// Server → client: talk to this shard instead.
    Redirect(Redirect),
    /// Either direction: your epoch is behind.
    StaleEpoch(StaleEpoch),
    /// Orchestrator → shards: a new epoch is prepared.
    EpochPrepare(EpochPrepare),
    /// Shard → orchestrator: epoch staged.
    EpochAck(EpochAck),
    /// Orchestrator → shards: activate the prepared epoch.
    EpochCommit(EpochCommit),
    /// Orchestrator → shards: abandon the prepared epoch.
    EpochAbort(EpochAbort),
    /// Shard → orchestrator: periodic load report.
    LoadReport(LoadReport),
    /// Client/shard → directory: give me the map.
    DirectoryQuery(DirectoryQuery),
    /// Directory → client/shard: here is the map.
    DirectoryReply(DirectoryReply),
    /// Operator/MCP → orchestrator: administrative command.
    AdminCommand(AdminCommand),
    /// Orchestrator → operator/MCP: command result.
    AdminReply(AdminReply),
    /// Client → shard: my connection to `shard` is up (or failed).
    ///
    /// The make-before-break gate for a handoff: the offering shard waits for
    /// this with `connected = true` (mirrored by the accepting shard) before it
    /// commits, and aborts with [`HandoffReason::ClientNotConnected`] on timeout.
    NeighbourStatus(NeighbourStatus),
    /// Shard → orchestrator: give me an id and my map.
    ShardHello(ShardHello),
    /// Orchestrator → shard: here are your id, region, map and peer token.
    ShardWelcome(ShardWelcome),
    /// Orchestrator → shard: the cluster's scene, at its revision.
    ///
    /// Sent right behind the [`ShardWelcome`] on every (re-)registration and
    /// again on every change. The shard adopts it when it
    /// [supersedes](SceneState::supersedes) what it holds and answers with a
    /// [`Message::SceneAck`] either way. Not part of the epoch protocol: the
    /// map does not change with the scene (see [`crate::scene`]).
    SceneSet(SceneState),
    /// Shard → orchestrator: the scene and revision this shard runs now.
    SceneAck(SceneAck),
    /// Shard → client: the scene to draw around the ships, at its revision.
    ///
    /// Sent right after the [`Welcome`] on every join — including the second
    /// connection a client opens during a handoff — and again whenever the
    /// shard's scene changes. A client keeps the highest revision it has seen.
    SceneUpdate(SceneState),
}

impl Message {
    /// Number of variants of this enum, i.e. one past the highest valid wire tag.
    ///
    /// [`codec`](crate::codec) rejects a larger tag as
    /// [`CodecError::UnknownTag`](crate::codec::CodecError::UnknownTag) before
    /// decoding. Bump it together with a new variant appended below; the
    /// `message_tags_are_dense_and_covered` test fails if the two drift apart.
    pub const VARIANT_COUNT: u32 = 28;

    /// The wire tag (bincode variant index) of this message.
    ///
    /// Exhaustive on purpose: adding a variant without giving it a tag here is a
    /// compile error, which is what keeps [`Self::VARIANT_COUNT`] honest.
    #[must_use]
    pub fn wire_tag(&self) -> u32 {
        match self {
            Message::Hello(_) => 0,
            Message::Welcome(_) => 1,
            Message::Reject(_) => 2,
            Message::InputTick(_) => 3,
            Message::Snapshot(_) => 4,
            Message::Delta(_) => 5,
            Message::HandoffOffer(_) => 6,
            Message::HandoffAccept(_) => 7,
            Message::HandoffCommit(_) => 8,
            Message::HandoffAbort(_) => 9,
            Message::GhostUpdate(_) => 10,
            Message::Redirect(_) => 11,
            Message::StaleEpoch(_) => 12,
            Message::EpochPrepare(_) => 13,
            Message::EpochAck(_) => 14,
            Message::EpochCommit(_) => 15,
            Message::EpochAbort(_) => 16,
            Message::LoadReport(_) => 17,
            Message::DirectoryQuery(_) => 18,
            Message::DirectoryReply(_) => 19,
            Message::AdminCommand(_) => 20,
            Message::AdminReply(_) => 21,
            Message::NeighbourStatus(_) => 22,
            Message::ShardHello(_) => 23,
            Message::ShardWelcome(_) => 24,
            Message::SceneSet(_) => 25,
            Message::SceneAck(_) => 26,
            Message::SceneUpdate(_) => 27,
        }
    }

    /// The shard-map epoch this message carries, if it carries one.
    ///
    /// Used to answer [`StaleEpoch`] without decoding the payload further.
    #[must_use]
    pub fn epoch(&self) -> Option<u64> {
        match self {
            Message::Snapshot(s) => Some(s.epoch),
            Message::Delta(d) => Some(d.epoch),
            Message::HandoffOffer(o) => Some(o.epoch),
            Message::HandoffAccept(a) => Some(a.epoch),
            Message::HandoffCommit(c) => Some(c.epoch),
            Message::EpochPrepare(p) => Some(p.epoch),
            Message::EpochAck(a) => Some(a.epoch),
            Message::EpochCommit(c) => Some(c.epoch),
            Message::EpochAbort(a) => Some(a.epoch),
            Message::DirectoryReply(r) => Some(r.shard_map.epoch),
            Message::Redirect(r) => Some(r.shard_map.epoch),
            Message::Welcome(w) => Some(w.shard_map.epoch),
            _ => None,
        }
    }
}
