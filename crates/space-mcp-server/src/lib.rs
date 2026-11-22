//! `space-mcp-server`: an MCP server that boots an in-process authoritative shard
//! and lets an agent inspect and drive it.
//!
//! The shard is the real thing: [`space_server::Shard`] behind a real
//! [`ShardNet`] QUIC front end on loopback (so the `space-mcp-client` tools, or
//! `space-client --server`, can connect to the address `shard_status` reports),
//! ticked at 30 Hz unless started with `--no-tick`.
//!
//! # Tools
//!
//! | tool            | what it does                                              |
//! |-----------------|-----------------------------------------------------------|
//! | `shard_status`  | id, QUIC endpoint, epoch, tick, ship/client counts, hash  |
//! | `topology`      | the shard map this shard follows (regions and endpoints)  |
//! | `list_ships`    | every live ship: guid, owning session, offset, velocity   |
//! | `spawn_ship`    | spawn an unowned ship inside the region, returns its guid |
//! | `ship_state`    | position / velocity / orientation of one ship by guid     |
//! | `despawn_ship`  | remove a ship by guid                                     |
//! | `advance_ticks` | run N simulation ticks right now (deterministic stepping) |
//!
//! `trigger_split` / `trigger_merge` are deliberately **not** offered: topology
//! changes belong to the orchestrator (`Orchestrator::handle_admin` over its
//! control plane), and this binary serves a single standalone shard with no
//! orchestrator to ask.

#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use space_mcp::tool::{arg_u64_opt, BoxFuture};
use space_mcp::{McpServer, Tool, ToolRegistry};
use space_protocol::{EntityGuid, ShardId};
use space_server::net::ShardNet;
use space_server::{Shard, ShardConfig};
use space_world::{NodeRef, WorldPos};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

/// The shard id the harness runs as.
pub const SHARD_ID: u32 = 1;
/// Largest `|offset|` per axis (metres) `spawn_ship` accepts: the region is one
/// 1024 m cell, so this keeps a spawned ship inside it.
pub const MAX_SPAWN_OFFSET_M: f64 = 500.0;
/// Most ticks one `advance_ticks` call may run.
pub const MAX_ADVANCE_TICKS: u64 = 3600;

/// An in-process shard plus its network front end and (optional) tick loop.
#[derive(Debug)]
pub struct ShardHarness {
    /// The live shard.
    pub shard: Arc<Mutex<Shard>>,
    /// The QUIC front end.
    pub net: Arc<ShardNet>,
    /// Where clients dial.
    pub addr: SocketAddr,
    tasks: Vec<JoinHandle<()>>,
}

impl ShardHarness {
    /// Boot a shard owning the cell at the origin, QUIC on `bind`. With
    /// `ticking` a 30 Hz loop steps it and flushes replication; without, it only
    /// advances when `advance_ticks` is called.
    ///
    /// # Errors
    ///
    /// A message when the endpoint cannot be bound.
    pub async fn boot(bind: SocketAddr, ticking: bool) -> Result<Self, String> {
        let net = Arc::new(ShardNet::bind(bind).map_err(|e| format!("bind {bind}: {e}"))?);
        let addr = net.local_addr().map_err(|e| e.to_string())?;
        let region = NodeRef::new(64, [0, 0, 0]);
        let shard = Arc::new(Mutex::new(Shard::new(ShardConfig::new(
            ShardId::new(SHARD_ID),
            addr,
            region,
        ))));
        let mut tasks = vec![tokio::spawn({
            let (net, shard) = (Arc::clone(&net), Arc::clone(&shard));
            async move { net.serve(shard).await }
        })];
        if ticking {
            tasks.push(tokio::spawn({
                let (net, shard) = (Arc::clone(&net), Arc::clone(&shard));
                async move {
                    let mut clock =
                        tokio::time::interval(Duration::from_secs_f64(f64::from(space_sim::dt())));
                    clock.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                    loop {
                        clock.tick().await;
                        shard.lock().await.tick();
                        net.flush(&shard).await;
                    }
                }
            }));
        }
        Ok(Self {
            shard,
            net,
            addr,
            tasks,
        })
    }
}

impl Drop for ShardHarness {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
        self.net.close();
    }
}

/// `"<shard>:<serial>"`, the textual form of an [`EntityGuid`] these tools use.
#[must_use]
pub fn guid_text(guid: EntityGuid) -> String {
    format!("{}:{}", guid.shard.0, guid.serial)
}

/// Parse [`guid_text`].
///
/// # Errors
///
/// A message when `text` is not `<shard>:<serial>`.
pub fn parse_guid(text: &str) -> Result<EntityGuid, String> {
    let bad = || format!("guid {text:?} must look like \"<shard>:<serial>\", e.g. \"1:3\"");
    let (shard, serial) = text.split_once(':').ok_or_else(bad)?;
    Ok(EntityGuid::new(
        ShardId::new(shard.parse().map_err(|_| bad())?),
        serial.parse().map_err(|_| bad())?,
    ))
}

fn guid_arg(args: &Value) -> Result<EntityGuid, String> {
    parse_guid(space_mcp::tool::arg_str(args, "guid")?)
}

fn pretty(value: &Value) -> Result<String, String> {
    serde_json::to_string_pretty(value).map_err(|e| e.to_string())
}

fn ship_json(guid: EntityGuid, shard: &Shard) -> Option<Value> {
    let state = shard.sim().ship_state(guid)?;
    let centre = shard.region().center();
    Some(json!({
        "guid": guid_text(guid),
        "session": shard.session_of(guid),
        "offset_from_region_centre_m": WorldPos::delta(centre, state.pos),
        "velocity_mps": state.vel,
        "orientation_wxyz": state.quat,
        "angular_velocity_radps": state.ang_vel,
    }))
}

fn tool<F, Fut>(name: &str, description: &str, schema: Value, f: F) -> Tool
where
    F: Fn(Value) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<String, String>> + Send + 'static,
{
    Tool::new(name, description, schema, f)
}

/// The server's tools, backed by `harness`.
#[must_use]
pub fn registry(harness: &Arc<ShardHarness>) -> ToolRegistry {
    let empty = json!({"type": "object", "properties": {}});
    let guid_schema = json!({
        "type": "object",
        "properties": {"guid": {"type": "string", "description": "ship guid, \"<shard>:<serial>\""}},
        "required": ["guid"],
    });
    let h = |harness: &Arc<ShardHarness>| Arc::clone(harness);
    let mut r = ToolRegistry::new();

    let this = h(harness);
    r.register(tool(
        "shard_status",
        "Status of the in-process shard: id, QUIC endpoint to connect clients to, map epoch, simulation tick, live ship and client counts, handoff counters and the canonical state hash.",
        empty.clone(),
        move |_| status(h(&this)),
    ));

    let this = h(harness);
    r.register(tool(
        "topology",
        "The shard map this shard follows: epoch plus every region (octree depth/index, owning shard, QUIC endpoint).",
        empty.clone(),
        move |_| topology(h(&this)),
    ));

    let this = h(harness);
    r.register(tool(
        "list_ships",
        "Every live ship on the shard with its guid, owning client session (null if unowned), offset from the region centre (m) and velocity.",
        empty,
        move |_| list_ships(h(&this)),
    ));

    let this = h(harness);
    r.register(tool(
        "spawn_ship",
        "Spawn an unowned ship at rest inside the region and return its guid. `offset` is metres from the region centre, default [0,0,0], each axis within +-500.",
        json!({
            "type": "object",
            "properties": {"offset": {
                "type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                "description": "[x, y, z] metres from the region centre",
            }},
        }),
        move |args| spawn_ship(h(&this), args),
    ));

    let this = h(harness);
    r.register(tool(
        "ship_state",
        "Position (offset from region centre, m), velocity, orientation and angular velocity of the ship with this guid.",
        guid_schema.clone(),
        move |args| ship_state(h(&this), args),
    ));

    let this = h(harness);
    r.register(tool(
        "despawn_ship",
        "Remove the ship with this guid from the shard.",
        guid_schema,
        move |args| despawn_ship(h(&this), args),
    ));

    let this = h(harness);
    r.register(tool(
        "advance_ticks",
        "Run `ticks` simulation ticks immediately (1..=3600; 30 ticks = 1 s) and return the new tick. Useful with --no-tick for deterministic stepping.",
        json!({
            "type": "object",
            "properties": {"ticks": {"type": "integer", "minimum": 1, "maximum": MAX_ADVANCE_TICKS}},
            "required": ["ticks"],
        }),
        move |args| advance_ticks(h(&this), args),
    ));
    r
}

fn status(h: Arc<ShardHarness>) -> BoxFuture<Result<String, String>> {
    Box::pin(async move {
        let shard = h.shard.lock().await;
        let stats = shard.stats();
        pretty(&json!({
            "shard": shard.shard().0,
            "quic_endpoint": h.addr.to_string(),
            "epoch": shard.epoch(),
            "tick": stats.tick,
            "ships": stats.ships,
            "clients": stats.clients,
            "handed_off": stats.handed_off,
            "received": stats.received,
            "aborted": stats.aborted,
            "state_hash": format!("{:016x}", shard.sim().state_hash()),
        }))
    })
}

fn topology(h: Arc<ShardHarness>) -> BoxFuture<Result<String, String>> {
    Box::pin(async move {
        let shard = h.shard.lock().await;
        let map = shard.shard_map();
        let regions: Vec<Value> = map
            .regions
            .iter()
            .map(|r| {
                json!({
                    "shard": r.shard.0,
                    "depth": r.node.depth,
                    "index": r.node.index,
                    "endpoint": r.endpoint.to_string(),
                    "is_this_shard": r.shard == shard.shard(),
                })
            })
            .collect();
        pretty(&json!({"this_shard": shard.shard().0, "epoch": map.epoch, "regions": regions}))
    })
}

fn list_ships(h: Arc<ShardHarness>) -> BoxFuture<Result<String, String>> {
    Box::pin(async move {
        let shard = h.shard.lock().await;
        let ships: Vec<Value> = shard
            .sim()
            .ships()
            .into_iter()
            .filter_map(|e| shard.guids().guid_of(e))
            .filter_map(|g| ship_json(g, &shard))
            .collect();
        pretty(&json!({"count": ships.len(), "ships": ships}))
    })
}

fn spawn_ship(h: Arc<ShardHarness>, args: Value) -> BoxFuture<Result<String, String>> {
    Box::pin(async move {
        let offset = match args.get("offset") {
            None | Some(Value::Null) => [0.0; 3],
            Some(v) => {
                let a = v
                    .as_array()
                    .filter(|a| a.len() == 3)
                    .ok_or("`offset` must be an array of 3 numbers")?;
                let mut o = [0.0; 3];
                for (slot, n) in o.iter_mut().zip(a) {
                    *slot = n
                        .as_f64()
                        .filter(|n| n.is_finite() && n.abs() <= MAX_SPAWN_OFFSET_M)
                        .ok_or(format!(
                            "`offset` components must be finite numbers within +-{MAX_SPAWN_OFFSET_M} m"
                        ))?;
                }
                o
            }
        };
        let mut shard = h.shard.lock().await;
        let pos = WorldPos::from_f64_metres_around(shard.region().center(), offset);
        // Session 0 is never issued (sessions start at 1): an unowned ship.
        let guid = shard.spawn_ship_for(0, pos);
        pretty(&ship_json(guid, &shard).ok_or("spawned ship vanished")?)
    })
}

fn ship_state(h: Arc<ShardHarness>, args: Value) -> BoxFuture<Result<String, String>> {
    Box::pin(async move {
        let guid = guid_arg(&args)?;
        let shard = h.shard.lock().await;
        let ship = ship_json(guid, &shard).ok_or_else(|| format!("no ship {}", guid_text(guid)))?;
        pretty(&ship)
    })
}

fn despawn_ship(h: Arc<ShardHarness>, args: Value) -> BoxFuture<Result<String, String>> {
    Box::pin(async move {
        let guid = guid_arg(&args)?;
        let mut shard = h.shard.lock().await;
        if shard.sim().ship_state(guid).is_none() {
            return Err(format!("no ship {}", guid_text(guid)));
        }
        shard.despawn_ship(guid);
        Ok(format!("despawned {}", guid_text(guid)))
    })
}

fn advance_ticks(h: Arc<ShardHarness>, args: Value) -> BoxFuture<Result<String, String>> {
    Box::pin(async move {
        let ticks = arg_u64_opt(&args, "ticks")?.ok_or("argument `ticks` (integer) is required")?;
        if !(1..=MAX_ADVANCE_TICKS).contains(&ticks) {
            return Err(format!("`ticks` must be in 1..={MAX_ADVANCE_TICKS}"));
        }
        // One tick, then deliver what it queued — exactly what the 30 Hz loop
        // does. Stepping all the ticks under one lock and never flushing (as
        // this used to) meant that with `--no-tick`, where no loop flushes
        // either, a connected client never received a single frame: it never
        // learnt which ship was its own, so it never sent input, and every tick
        // stepped here piled one more undelivered frame per client into the
        // shard's outbox. The lock is released between ticks, so status tools
        // and joining clients are not held up for the whole run.
        let mut tick = h.shard.lock().await.stats().tick;
        for _ in 0..ticks {
            tick = h.shard.lock().await.tick();
            h.net.flush(&h.shard).await;
        }
        pretty(&json!({"tick": tick}))
    })
}

/// The complete MCP server for `harness`.
#[must_use]
pub fn mcp_server(harness: &Arc<ShardHarness>) -> McpServer {
    McpServer::new(
        "space-mcp-server",
        env!("CARGO_PKG_VERSION"),
        registry(harness),
    )
}
