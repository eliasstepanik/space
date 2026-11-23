//! Shard configuration.
//!
//! Everything a shard needs to start is here, so a test can build two shards
//! that differ only in their region and endpoint.

use std::net::SocketAddr;

use space_protocol::ShardId;
use space_sim::SimTick;
use space_world::NodeRef;

/// Default interest radius, metres.
///
/// `docs/plan.md` U9 sizes a shard at ~1k entities in a client's interest
/// range; a 10 km sphere around a ship is the volume that budget is quoted
/// against, and it is also what keeps a delta per client small enough for one
/// datagram most of the time.
pub const DEFAULT_INTEREST_RADIUS: f32 = 10_000.0;

/// Ticks a redirected client has to confirm its connection to the accepting
/// shard before we stop waiting on it.
///
/// A client we [`Redirect`]ed is expected to open the new connection within the
/// handoff lead ([`crate::handoff::HANDOFF_LEAD`]); if it never reports in, the
/// redirect is cleared after this budget so the client is not held in a
/// half-redirected state forever. It is a *confirmation* timeout, not a
/// disconnect: the client keeps its current ship on this shard.
pub const REDIRECT_TIMEOUT_TICKS: SimTick = 4 * space_sim::TICK_HZ as SimTick;

/// Ticks a client that connected to us *for* a staged handoff entity (and so
/// was given no placeholder ship) has to adopt it before we treat it as an
/// ordinary newcomer and spawn it a ship of its own.
///
/// This is the other half of not spawning a phantom ship for a redirected
/// client: we hold off the placeholder until adoption, and if adoption never
/// comes (the client crashed, or its token only coincidentally matched a staged
/// transfer) we fall back to the ordinary join behaviour.
pub const ADOPT_TIMEOUT_TICKS: SimTick = 2 * space_sim::TICK_HZ as SimTick;

/// Ticks a shard waits before offering an entity to the **same** peer again
/// after that peer refused it.
///
/// A refusal is the peer's [`HandoffAbort`](space_protocol::HandoffAbort)
/// (`TargetFull`, "entity is not available here", …) or its
/// [`StaleEpoch`](space_protocol::StaleEpoch). The ship is still across the
/// boundary on the next tick, so without a pause the shard re-offered at once —
/// about fifteen offer/refusal round trips a second (each used to come with a
/// `Redirect` to every client watching the ship; now a redirect waits for the
/// accept, so a refused offer redirects nobody) — for as
/// long as the peer kept refusing. Two seconds matches the handoff timeout: it
/// is the rate a peer that does not answer at all is retried at. A shard-map
/// change clears the pause, because the refusal was about the old map.
pub const REOFFER_BACKOFF_TICKS: SimTick = 2 * space_sim::TICK_HZ as SimTick;

/// How often a shard tells every connected client its scene again, in ticks
/// (5 s).
///
/// A [`SceneUpdate`](space_protocol::Message::SceneUpdate) is offered to a
/// client's write queue like any other message, and the front end drops what
/// it offers to a queue that is full (it never waits: see
/// [`crate::net::ShardNet::flush`]). A frame lost that way is sent again once
/// the client's acknowledgements show it is missing; a scene announcement has
/// no acknowledgement, so a client that lost one would draw the old scene
/// until the next change or a reconnect. Saying it again every few seconds
/// closes that: the message is a dozen bytes, the revision makes a repeat a
/// no-op for a client that already has it, and the cost on the tick is one
/// comparison (the clients are only walked on the tick it is due).
pub const SCENE_REANNOUNCE_TICKS: SimTick = 5 * space_sim::TICK_HZ as SimTick;

/// How far a shard's scene clock may drift from the wall clock, µs, before
/// the shard re-anchors it ([`crate::Shard::sync_scene_clock`]): 30 ms.
///
/// A shard's tick count falls behind the wall clock whenever its loop skips a
/// tick (a stall), so its mapping from ticks to scene time drifts; past this
/// it is moved back onto the wall clock — the planets jump by that much, and
/// what they carry with them. Two shards on hosts with synchronised clocks
/// therefore agree on the scene time to within about twice this.
pub const SCENE_CLOCK_TOLERANCE_US: i64 = 30_000;

/// Microseconds of UTC since the Unix epoch, now (what scene time is).
#[must_use]
pub fn wall_clock_us() -> i64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    i64::try_from(now.as_micros()).unwrap_or(i64::MAX)
}

/// How many inputs one client may have waiting between two ticks.
///
/// Every input is rewritten to the session's one ship and the simulation keeps
/// at most one input per ship per tick for the next
/// [`MAX_INPUT_LEAD`](space_sim::sim_world::MAX_INPUT_LEAD) ticks, so a client
/// has no use for more than a few redundant windows of that size. Without a cap
/// a single 1 MiB frame put ~23 000 inputs through the simulation's buffer under
/// the shard lock.
pub const MAX_PENDING_INPUTS: usize = 4 * (space_sim::sim_world::MAX_INPUT_LEAD as usize + 1);

/// The settings of one shard.
#[derive(Debug, Clone)]
pub struct ShardConfig {
    /// This shard's id. Allocated by [`crate::alloc::ShardIdAllocator`] — never
    /// hand-written in production, because a reused id aliases two entities'
    /// guids (see [`crate::alloc`]).
    pub shard: ShardId,
    /// Where the shard listens for QUIC.
    pub endpoint: SocketAddr,
    /// The octree node this shard is authoritative for.
    pub region: NodeRef,
    /// Radius, in metres, at which an entity is replicated to a client.
    pub interest_radius: f32,
    /// Per-client frame budget for one snapshot/delta frame.
    pub frame_budget: usize,
}

impl ShardConfig {
    /// A config for `shard`, listening on `endpoint`, owning `region`.
    #[must_use]
    pub fn new(shard: ShardId, endpoint: SocketAddr, region: NodeRef) -> Self {
        Self {
            shard,
            endpoint,
            region,
            interest_radius: DEFAULT_INTEREST_RADIUS,
            frame_budget: space_protocol::DEFAULT_BUDGET,
        }
    }
}
