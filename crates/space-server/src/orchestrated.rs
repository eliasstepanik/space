//! Booting one **orchestrated** shard: register with the orchestrator, adopt the
//! id, region and map it hands back, and follow every epoch it publishes.
//!
//! This lives in the library (not only in `main.rs`) so a test can start a real
//! orchestrated shard in-process, next to a real orchestrator, over loopback.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use space_protocol::{ShardHello, ShardId, ShardRegion};
use space_world::NodeRef;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use crate::net::{OrchestratorLink, ShardNet};
use crate::shard::Shard;
use crate::ShardConfig;

/// What an orchestrated shard needs to know before it asks for its identity.
#[derive(Debug, Clone)]
pub struct OrchestratedOptions {
    /// The id to claim, or [`ShardId::NONE`] to let the orchestrator allocate.
    pub shard: ShardId,
    /// Where to listen (port 0 for "any").
    pub bind: SocketAddr,
    /// The address reported in `ShardHello`, or `None` for the bound address.
    pub advertise: Option<SocketAddr>,
    /// The region to claim, or `None` for "place me at the root".
    pub claim_region: Option<NodeRef>,
    /// The cluster id forwarded in `ShardHello`.
    pub cluster: String,
}

impl OrchestratedOptions {
    /// The address to put in `ShardHello`: `advertise` if given, else the
    /// address the socket actually bound.
    #[must_use]
    pub fn advertised(&self, bound: SocketAddr) -> SocketAddr {
        self.advertise.unwrap_or(bound)
    }
}

/// An orchestrated shard that is up and serving.
#[derive(Debug)]
pub struct RunningShard {
    /// The id the orchestrator confirmed or allocated.
    pub shard: ShardId,
    /// The region the orchestrator placed it on.
    pub region: NodeRef,
    /// The address clients and peers dial (what the hello advertised).
    pub endpoint: SocketAddr,
    /// The shard's live state (tests read it; the serve and tick tasks own the
    /// write side).
    pub state: Arc<Mutex<Shard>>,
    serve: JoinHandle<()>,
    ticking: JoinHandle<()>,
    link: OrchestratorLink,
}

impl RunningShard {
    /// Wait until the shard's endpoint closes.
    ///
    /// # Errors
    ///
    /// The serve task panicked or was cancelled.
    pub async fn wait(self) -> Result<(), String> {
        let Self {
            serve,
            ticking,
            link,
            ..
        } = self;
        let result = serve.await.map_err(|e| e.to_string());
        ticking.abort();
        link.close();
        result
    }

    /// Stop serving and ticking (tests; a process just exits).
    pub fn abort(self) {
        self.serve.abort();
        self.ticking.abort();
        self.link.close();
    }
}

/// Register with the orchestrator at `orch_addr`, adopt the welcome, and start
/// serving and ticking. Returns once the shard is up.
///
/// # Errors
///
/// The socket cannot be bound or the orchestrator handshake fails (wrong cluster,
/// root already claimed, unreachable).
pub async fn start_orchestrated(
    options: OrchestratedOptions,
    orch_addr: SocketAddr,
) -> Result<RunningShard, String> {
    // Bind first (default `127.0.0.1:0`): without `advertise` the endpoint we
    // report is the one we listen on. With it, we report that address instead,
    // so a peer or client on another host/container can dial us.
    let net = Arc::new(ShardNet::bind(options.bind).map_err(|e| format!("bind: {e}"))?);
    let listening = net.local_addr().map_err(|e| e.to_string())?;
    let advertised = options.advertised(listening);

    // Ask the orchestrator for an id + region + map. `ShardId::NONE` requests a
    // fresh allocation; a claimed region names a node, else the orchestrator
    // places us at its root.
    let claim = options
        .claim_region
        .map(|node| ShardRegion::new(node, ShardId::NONE, advertised));
    let hello = ShardHello {
        cluster: options.cluster.clone(),
        shard: options.shard,
        endpoint: advertised,
        region: claim,
    };
    let (welcome, conn) = OrchestratorLink::handshake(orch_addr, hello)
        .await
        .map_err(|e| format!("orchestrator handshake: {e}"))?;

    let config = ShardConfig::new(welcome.shard, advertised, welcome.region.node);
    let mut shard = Shard::new(config);
    shard.set_shard_map(welcome.shard_map.clone());
    shard.set_endpoint(advertised);
    let shard = Arc::new(Mutex::new(shard));

    // Register the map's peer endpoints and route orchestrator traffic through
    // the link's FIFO sender (epoch acks + load reports go out in order).
    for region in &welcome.shard_map.regions {
        if region.shard != welcome.shard {
            net.set_shard_endpoint(region.shard, region.endpoint).await;
        }
    }
    let link = OrchestratorLink::attach(conn, Arc::clone(&shard));
    net.set_orchestrator_sender(link.tx.clone()).await;

    println!(
        "shard {} on {listening} (advertised {advertised}, orchestrated by {orch_addr}), region depth {}",
        welcome.shard.0, welcome.region.node.depth
    );

    let serve = {
        let (net, shard) = (Arc::clone(&net), Arc::clone(&shard));
        tokio::spawn(async move { net.serve(shard).await })
    };
    let state = Arc::clone(&shard);
    let ticking = tokio::spawn(async move {
        let mut clock = tokio::time::interval(Duration::from_secs_f64(f64::from(space_sim::dt())));
        clock.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            clock.tick().await;
            shard.lock().await.tick();
            net.flush(&shard).await;
        }
    });

    Ok(RunningShard {
        shard: welcome.shard,
        region: welcome.region.node,
        endpoint: advertised,
        serve,
        ticking,
        link,
    })
}

/// Serve as an orchestrated shard until its endpoint closes.
///
/// # Errors
///
/// See [`start_orchestrated`]; also if the serve task fails.
pub async fn run_orchestrated(
    options: OrchestratedOptions,
    orch_addr: SocketAddr,
) -> Result<(), String> {
    start_orchestrated(options, orch_addr).await?.wait().await
}
