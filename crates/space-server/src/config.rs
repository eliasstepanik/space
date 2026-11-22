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
