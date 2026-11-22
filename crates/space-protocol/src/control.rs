//! Orchestrator-facing messages: load reporting, the shard directory and the
//! administrative surface used by operators and the MCP servers.

use serde::{Deserialize, Serialize};

use crate::shard::{ShardId, ShardMap};

/// A shard's periodic report to the orchestrator.
///
/// `load` is the plan's split/merge metric —
/// `max(players / 100, tick_p95_ms / 33.3)` (`docs/plan.md`, U12). The
/// orchestrator splits at `load >= 0.8` sustained for 10 s and merges at
/// `<= 0.35` sustained for 60 s; the shard computes it because only the shard
/// knows its own tick time.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LoadReport {
    /// Reporting shard.
    pub shard: ShardId,
    /// Players connected to it right now.
    pub players: u32,
    /// Tick time at the 95th percentile, in milliseconds.
    pub tick_p95_ms: f32,
    /// The derived load metric, `0.0`..
    pub load: f32,
}

/// Ask the directory for the current shard map.
///
/// A unit struct today: the query carries no parameters because the map is
/// global and epoch-versioned — a client that already has the newest epoch does
/// not ask. It is a struct so a filter can be added later without a new tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectoryQuery {}

/// The directory's answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectoryReply {
    /// The map in effect.
    pub shard_map: ShardMap,
    /// Tick the directory generated it at, so a client can tell two replies
    /// apart when they carry the same epoch.
    pub tick: u64,
}

/// An administrative command from an operator or an MCP server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdminCommand {
    /// Split the given shard along its longest axis, regardless of the load
    /// heuristic.
    ForceSplit(ShardId),
    /// Merge the given shard into its lowest-load neighbour.
    ForceMerge(ShardId),
    /// Kill a shard immediately (test / break-glass). Its regions are reassigned.
    KillShard(ShardId),
    /// Spawn `count` bot players, spread over the current map.
    SpawnBots {
        /// How many bots to create.
        count: u32,
    },
    /// Report every shard's load right now.
    Status,
}

/// The result of an [`AdminCommand`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdminReply {
    /// The command was accepted and is being applied.
    Accepted {
        /// Command this answers, echoed so a caller can match replies to
        /// commands it sent concurrently.
        command: AdminCommand,
    },
    /// The command was refused, with a human-readable reason.
    Refused {
        /// Command this answers.
        command: AdminCommand,
        /// Why, for logs and the MCP text channel.
        reason: String,
    },
    /// Answer to [`AdminCommand::Status`]: one line per shard, already
    /// formatted, because it is meant for a human or an LLM, not for logic.
    Status {
        /// One line per shard: `shard N players=P p95=Xms load=L`.
        lines: Vec<String>,
    },
}
