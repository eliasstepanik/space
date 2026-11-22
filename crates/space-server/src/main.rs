//! `space-server` — one authoritative shard.
//!
//! Binds a QUIC endpoint, owns one octree region, and runs the 30 Hz
//! [`Shard`] loop over it: tick, flush the outbox to clients and neighbour
//! shards, repeat. Everything that decides something lives in
//! [`space_server::shard`]; this binary only turns text into a [`ShardConfig`]
//! and keeps the loop turning.
//!
//! ```text
//!   space-server --shard 1 --bind 127.0.0.1:7777 --region 64:0:0:0 \
//!                --peer 2:127.0.0.1:7778:64:1:0:0
//! ```
//!
//! A region is the octree node a shard is authoritative for, written
//! `depth:x:y:z` ([`NodeRef::new`]). The world map is assembled from the
//! command line — this shard's own region plus one `--peer` per neighbour —
//! because a handoff looks the *destination* up in that map: a shard missing
//! from it is never handed anything, and a client's [`Welcome`] would send it a
//! map with holes in it.
//!
//! There is no argument-parsing dependency: the grammar is four flags, so it is
//! matched by hand against [`std::env::args`]. Ctrl-C terminates the process
//! (the runtime installs no handler), which is the right shutdown for a
//! stateless-per-tick shard: the world is rebuilt from snapshots on restart.
//!
//! [`Welcome`]: space_protocol::Welcome

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use space_protocol::{ShardId, ShardMap, ShardRegion};
use space_world::NodeRef;

use space_server::net::ShardNet;
use space_server::shard::Shard;
use space_server::ShardConfig;

/// What the command line said.
#[derive(Debug)]
struct Options {
    /// This shard's id.
    shard: ShardId,
    /// Where to listen. Defaults to `127.0.0.1:7777` standalone and to
    /// `127.0.0.1:0` when orchestrated; `0.0.0.0:<port>` is for containers.
    bind: SocketAddr,
    /// The address reported in `ShardHello` (`--advertise`), or `None` for
    /// "whatever address the socket actually bound". Only meaningful with
    /// `--orchestrator`: a container binds `0.0.0.0:<port>`, which no peer can
    /// dial, so it advertises the address peers and clients can reach it on.
    advertise: Option<SocketAddr>,
    /// The region this shard owns.
    region: NodeRef,
    /// Every region in the world map, this shard's included.
    regions: Vec<ShardRegion>,
    /// The orchestrator's control address, if this shard registers with one.
    ///
    /// With it the shard binds port 0, asks the orchestrator for its id +
    /// region + map (closing Phase-2 MAJOR 3: the orchestrator owns allocation),
    /// and thereafter follows pushed maps. Without it, the static `--shard` path
    /// is unchanged (regression safety for Phase 2's tests and docs).
    orchestrator: Option<SocketAddr>,
    /// The region to claim from the orchestrator (`--claim-region`), or `None`
    /// for "place me at the root". Only meaningful with `--orchestrator`.
    claim_region: Option<NodeRef>,
    /// The cluster id forwarded in `ShardHello`.
    cluster: String,
}

/// The usage line, printed for `--help` and for any parse error.
const USAGE: &str = "usage: space-server --shard <id> --region <depth:x:y:z> \
                     [--bind <addr:port>] [--peer <shard>:<addr:port>:<depth:x:y:z>] \
                     [--orchestrator <addr:port> [--claim-region <depth:x:y:z>] [--cluster <id>] \
                     [--advertise <addr:port>]]";

impl Options {
    /// Parse `argv` without the program name.
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut shard: Option<ShardId> = None;
        let mut bind: Option<SocketAddr> = None;
        let mut region: Option<NodeRef> = None;
        let mut regions: Vec<ShardRegion> = Vec::new();
        let mut advertise: Option<SocketAddr> = None;
        let mut orchestrator: Option<SocketAddr> = None;
        let mut claim_region: Option<NodeRef> = None;
        let mut cluster = "space-local".to_owned();

        let mut i = 0;
        while i < args.len() {
            // Every flag takes exactly one value. The flag and its value are read
            // together and the index advanced by two, so a trailing flag reports
            // itself rather than panicking on a missing value.
            let flag = args[i].as_str();
            let value = |what: &str| -> Result<&String, String> {
                args.get(i + 1)
                    .ok_or_else(|| format!("{what} needs a value"))
            };
            match flag {
                "-h" | "--help" => return Err(USAGE.to_owned()),
                "--shard" => {
                    let text = value("--shard")?;
                    shard = Some(ShardId::new(
                        text.parse::<u32>().map_err(|e| e.to_string())?,
                    ));
                }
                "--bind" => {
                    let text = value("--bind")?;
                    bind = Some(text.parse::<SocketAddr>().map_err(|e| e.to_string())?);
                }
                "--advertise" => {
                    let text = value("--advertise")?;
                    advertise = Some(text.parse::<SocketAddr>().map_err(|e| e.to_string())?);
                }
                "--region" => {
                    let text = value("--region")?;
                    region = Some(parse_node(text)?);
                }
                "--peer" => {
                    let text = value("--peer")?;
                    regions.push(parse_peer(text)?);
                }
                "--orchestrator" => {
                    let text = value("--orchestrator")?;
                    orchestrator = Some(text.parse::<SocketAddr>().map_err(|e| e.to_string())?);
                }
                "--claim-region" => {
                    let text = value("--claim-region")?;
                    claim_region = Some(parse_node(text)?);
                }
                "--cluster" => {
                    cluster = value("--cluster")?.clone();
                }
                other => return Err(format!("unknown argument {other}\n{USAGE}")),
            }
            i += if flag == "--help" || flag == "-h" {
                1
            } else {
                2
            };
        }

        // With an orchestrator the id and region come from the welcome, so
        // `--shard`/`--region` are not required; standalone they are.
        let orchestrated = orchestrator.is_some();
        let shard = match shard {
            Some(s) => s,
            None if orchestrated => ShardId::NONE,
            None => return Err(format!("missing --shard\n{USAGE}")),
        };
        let region = match region {
            Some(r) => r,
            None if orchestrated => claim_region.unwrap_or(space_world::NodeRef::new(
                space_orchestrator_root_depth(),
                [0, 0, 0],
            )),
            None => return Err(format!("missing --region\n{USAGE}")),
        };
        if advertise.is_some() && !orchestrated {
            return Err(format!("--advertise requires --orchestrator\n{USAGE}"));
        }
        // Standalone listens on the well-known port; orchestrated binds port 0
        // (the orchestrator learns the real endpoint from `ShardHello`).
        let default_port = if orchestrated { 0 } else { DEFAULT_PORT };
        let bind = bind.unwrap_or_else(|| SocketAddr::from(([127, 0, 0, 1], default_port)));
        // Our own region belongs in the map, last: `ShardMap::lookup` walks the
        // regions in order and keeps the deepest match, so a peer entry that
        // overlaps ours at equal depth loses to ours deterministically.
        regions.push(ShardRegion::new(region, shard, bind));
        Ok(Self {
            shard,
            bind,
            advertise,
            region,
            regions,
            orchestrator,
            claim_region,
            cluster,
        })
    }
}

/// The root depth a `--claim-region`-less orchestrated launch falls back to.
///
/// Matches the orchestrator's default root depth so a bare `--orchestrator`
/// claims the root; the orchestrator overrides the placement in its welcome
/// anyway (the value here only seeds the standalone map before the welcome).
fn space_orchestrator_root_depth() -> u8 {
    40
}

impl Options {
    /// The orchestrated-boot options this command line describes.
    fn orchestrated(&self) -> space_server::orchestrated::OrchestratedOptions {
        space_server::orchestrated::OrchestratedOptions {
            shard: self.shard,
            bind: self.bind,
            advertise: self.advertise,
            claim_region: self.claim_region,
            cluster: self.cluster.clone(),
        }
    }
}

/// The default listen port.
const DEFAULT_PORT: u16 = 7777;

/// Parse `depth:x:y:z`.
fn parse_node(spec: &str) -> Result<NodeRef, String> {
    let parts: Vec<&str> = spec.split(':').collect();
    if parts.len() != 4 {
        return Err(format!("region must be depth:x:y:z, got {spec}"));
    }
    let depth = parts[0].parse::<u8>().map_err(|e| e.to_string())?;
    let mut index = [0i64; 3];
    for (axis, part) in index.iter_mut().zip(&parts[1..]) {
        *axis = part.parse::<i64>().map_err(|e| e.to_string())?;
    }
    Ok(NodeRef::new(depth, index))
}

/// Parse a peer as `<shard>:<addr:port>:<depth:x:y:z>`.
///
/// The address and the node both contain colons, so the shape is pinned by
/// counting: `1:127.0.0.1:7778:64:1:0:0` is id, host, port, then four node
/// fields — six colons after the id for an IPv4 address.
fn parse_peer(spec: &str) -> Result<ShardRegion, String> {
    let bad = || format!("peer must be shard:addr:port:depth:x:y:z, got {spec}");
    let (id, rest) = spec.split_once(':').ok_or_else(bad)?;
    // Split the address off the front and the node off the back: the host and
    // port are the first two fields, everything after them is the node.
    let mut fields = rest.split(':');
    let Some(host) = fields.next() else {
        return Err(bad());
    };
    let Some(port) = fields.next() else {
        return Err(bad());
    };
    let node = fields.collect::<Vec<_>>().join(":");
    if node.split(':').count() != 4 {
        return Err(bad());
    }
    let shard = ShardId::new(id.parse::<u32>().map_err(|e| e.to_string())?);
    let endpoint = format!("{host}:{port}")
        .parse::<SocketAddr>()
        .map_err(|e| e.to_string())?;
    Ok(ShardRegion::new(parse_node(&node)?, shard, endpoint))
}

fn main() -> Result<(), String> {
    let options = Options::parse(&std::env::args().skip(1).collect::<Vec<_>>())?;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    runtime.block_on(run(options))
}

/// Serve until the process is interrupted.
async fn run(options: Options) -> Result<(), String> {
    // Orchestrated launch: bind port 0, ask the orchestrator for id + region +
    // map (closing Phase-2 MAJOR 3: the orchestrator owns allocation), then run
    // on what it says. The static `--shard` path below is unchanged.
    if let Some(orch_addr) = options.orchestrator {
        return run_orchestrated(options, orch_addr).await;
    }

    let net =
        Arc::new(ShardNet::bind(options.bind).map_err(|e| format!("bind {}: {e}", options.bind))?);
    let bound = net.local_addr().map_err(|e| e.to_string())?;

    let config = ShardConfig::new(options.shard, bound, options.region);
    let mut shard = Shard::new(config);
    shard.set_shard_map(ShardMap::new(1, options.regions.clone()));
    // The socket bound to `bound` (a `--bind` of port 0 resolves to a real one),
    // so patch our own region's endpoint *after* installing the map: the shard map
    // handed out in a Welcome must carry a reachable address for our own region,
    // not the `:0` the command line may have said.
    shard.set_endpoint(bound);
    let shard = Arc::new(tokio::sync::Mutex::new(shard));

    // A handoff is addressed by shard id, so the map's endpoints have to be
    // registered with the transport before the first crossing.
    for region in &options.regions {
        if region.shard != options.shard {
            net.set_shard_endpoint(region.shard, region.endpoint).await;
        }
    }

    println!(
        "shard {} on {bound}, region depth {} at [{}, {}, {}], {} region(s) in the map",
        options.shard.0,
        options.region.depth,
        options.region.index[0],
        options.region.index[1],
        options.region.index[2],
        options.regions.len()
    );

    let serve = {
        let (net, shard) = (Arc::clone(&net), Arc::clone(&shard));
        tokio::spawn(async move { net.serve(shard).await })
    };

    // The 30 Hz loop: one step, then everything the step queued goes out. The
    // interval's first tick fires immediately, which is what a cold start wants
    // — an empty world stepped once costs nothing and the first client to
    // connect is served on the tick it joins.
    let ticking = tokio::spawn(async move {
        let mut clock = tokio::time::interval(Duration::from_secs_f64(f64::from(space_sim::dt())));
        // A tick that overruns must not be made up in a burst: the simulation
        // would run ahead of the network for free. Skip it instead.
        clock.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            clock.tick().await;
            shard.lock().await.tick();
            net.flush(&shard).await;
        }
    });

    // `serve` returns only when the endpoint is closed, so in practice this
    // future is never ready and the process runs until it is killed.
    serve.await.map_err(|e| e.to_string())?;
    ticking.abort();
    Ok(())
}

/// Serve as an orchestrated shard: register with the orchestrator, adopt the id,
/// region and map it hands back, and follow every epoch it publishes. The boot
/// itself lives in [`space_server::orchestrated`] so tests can run it in-process.
async fn run_orchestrated(options: Options, orch_addr: SocketAddr) -> Result<(), String> {
    space_server::orchestrated::run_orchestrated(options.orchestrated(), orch_addr).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Options, String> {
        Options::parse(&args.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>())
    }

    #[test]
    fn a_minimal_command_line_binds_loopback() {
        let options = parse(&["--shard", "3", "--region", "64:2:0:0"]).expect("parses");
        assert_eq!(options.shard, ShardId::new(3));
        assert_eq!(options.bind.port(), DEFAULT_PORT);
        assert_eq!(options.region, NodeRef::new(64, [2, 0, 0]));
        assert_eq!(options.regions.len(), 1, "our own region is in the map");
        assert_eq!(options.regions[0].shard, ShardId::new(3));
    }

    #[test]
    fn a_peer_lands_in_the_shard_map_with_its_endpoint() {
        let options = parse(&[
            "--shard",
            "1",
            "--bind",
            "127.0.0.1:7777",
            "--region",
            "64:0:0:0",
            "--peer",
            "2:127.0.0.1:7778:64:1:0:0",
        ])
        .expect("parses");
        assert_eq!(options.regions.len(), 2);
        let peer = &options.regions[0];
        assert_eq!(peer.shard, ShardId::new(2));
        assert_eq!(peer.endpoint.port(), 7778);
        assert_eq!(peer.node, NodeRef::new(64, [1, 0, 0]));
    }

    #[test]
    fn a_bad_command_line_is_an_error_not_a_panic() {
        for args in [
            &[] as &[&str],
            &["--shard", "1"],
            &["--shard"],
            &["--shard", "x", "--region", "64:0:0:0"],
            &["--region", "64:0:0"],
            &["--peer", "2:127.0.0.1:7778"],
            &["--nope"],
        ] {
            assert!(parse(args).is_err(), "{args:?} must be rejected");
        }
        assert!(
            parse(&["--help"]).is_err(),
            "--help exits non-zero with usage"
        );
    }

    #[test]
    fn orchestrator_flags_are_parsed_and_do_not_require_shard_or_region() {
        // The orchestrated path: no --shard/--region, the id + region come from
        // the welcome. `--claim-region` names the node to claim.
        let options = parse(&[
            "--orchestrator",
            "127.0.0.1:7900",
            "--claim-region",
            "41:0:0:0",
            "--cluster",
            "test",
        ])
        .expect("parses");
        assert_eq!(
            options.orchestrator,
            Some("127.0.0.1:7900".parse().unwrap())
        );
        assert_eq!(options.claim_region, Some(NodeRef::new(41, [0, 0, 0])));
        assert_eq!(options.cluster, "test");
        assert_eq!(
            options.shard,
            ShardId::NONE,
            "no id yet: the orchestrator allocates"
        );
    }

    #[test]
    fn an_orchestrated_shard_binds_loopback_port_zero_by_default() {
        let options = parse(&["--orchestrator", "127.0.0.1:7900"]).expect("parses");
        assert_eq!(options.bind, "127.0.0.1:0".parse().unwrap());
        assert!(options.advertise.is_none());
    }

    #[test]
    fn bind_accepts_a_container_wildcard_address() {
        let options =
            parse(&["--orchestrator", "127.0.0.1:7900", "--bind", "0.0.0.0:7870"]).expect("parses");
        assert_eq!(options.bind, "0.0.0.0:7870".parse().unwrap());
    }

    #[test]
    fn the_advertised_address_defaults_to_the_bound_address() {
        let options = parse(&["--orchestrator", "127.0.0.1:7900"]).expect("parses");
        let bound: SocketAddr = "127.0.0.1:41234".parse().unwrap();
        assert_eq!(options.advertised(bound), bound);
    }

    #[test]
    fn an_explicit_advertise_overrides_the_bound_address() {
        let options = parse(&[
            "--orchestrator",
            "127.0.0.1:7900",
            "--bind",
            "0.0.0.0:7870",
            "--advertise",
            "10.0.0.5:7870",
        ])
        .expect("parses");
        let bound: SocketAddr = "0.0.0.0:7870".parse().unwrap();
        assert_eq!(options.advertised(bound), "10.0.0.5:7870".parse().unwrap());
    }

    #[test]
    fn advertise_is_rejected_without_an_orchestrator_or_when_malformed() {
        assert!(parse(&[
            "--shard",
            "1",
            "--region",
            "64:0:0:0",
            "--advertise",
            "10.0.0.5:1"
        ])
        .is_err());
        assert!(parse(&["--orchestrator", "127.0.0.1:7900", "--advertise", "nope"]).is_err());
    }

    #[test]
    fn a_bare_orchestrator_flag_is_enough() {
        let options = parse(&["--orchestrator", "127.0.0.1:7900"]).expect("parses");
        assert!(options.orchestrator.is_some());
        assert!(
            options.claim_region.is_none(),
            "no claim: place at the root"
        );
    }

    #[test]
    fn the_static_shard_path_still_requires_shard_and_region() {
        // Regression: without --orchestrator, --shard/--region are still required
        // (Phase 2's tests and docs depend on the unchanged standalone path).
        assert!(parse(&["--region", "64:0:0:0"]).is_err());
        assert!(parse(&["--shard", "1"]).is_err());
    }
}
