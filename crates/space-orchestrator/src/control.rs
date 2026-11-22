//! The control-plane message handlers (plan §3.1).
//!
//! Pure handlers: given a message and the orchestrator's state, produce the
//! reply to send and the state change to apply. The QUIC front end (a later
//! slice) only moves frames; every decision lives here so it is testable without
//! a socket.
//!
//! The handlers implemented in this slice:
//! * [`ControlPlane::handle_shard_hello`] — allocate-or-confirm the id, place
//!   the region, return a [`ShardWelcome`] carrying the current map and an
//!   issued `peer_token`. Only the protocol version and the cluster name are
//!   checked on the way in, and the `peer_token` is **not enforced** by anyone
//!   today (shards never present it; `Shard::peer_message` checks map membership
//!   only). The control port binds to loopback by default, which is the only
//!   mitigation. See `docs/security-notes.md`.
//! * [`ControlPlane::handle_load_report`] — feed the autoscaler.
//! * [`ControlPlane::handle_admin`] — the admin verbs. Verbs whose machinery
//!   arrives with a later slice return an honest [`AdminReply::Refused`] rather
//!   than panicking (D4: no `todo!()` in a CI-run path).

use std::collections::HashMap;
use std::io;
use std::net::SocketAddr;
use std::sync::Arc;

use space_protocol::control::{AdminCommand, AdminReply, LoadReport};
use space_protocol::shard::{EpochAck, ShardId, ShardMap, ShardRegion};
use space_protocol::{Message, ShardHello, ShardWelcome};
use tokio::sync::{mpsc, Mutex};

use crate::autoscale::Autoscaler;
use crate::config::OrchestratorConfig;
use crate::epoch::EpochPublisher;
use crate::ids::PersistentShardIdAllocator;

/// The orchestrator's mutable control state that the handlers touch.
#[derive(Debug)]
pub struct ControlPlane {
    /// Config (cluster id, root placement, first id).
    pub config: OrchestratorConfig,
    /// The id counter (H9).
    pub ids: PersistentShardIdAllocator,
    /// The epoch publisher; owns the published map (H10).
    pub publisher: EpochPublisher,
    /// The autoscaler (H14).
    pub autoscaler: Autoscaler,
    /// The shared peer secret handed to every shard on welcome.
    pub peer_token: String,
    /// The root owner the orchestrator reserved at startup that no live shard
    /// has claimed yet. The first root (or region-less) hello adopts it.
    unclaimed_root_owner: Option<ShardId>,
}

impl ControlPlane {
    /// Build a control plane over an already-opened allocator and initial map.
    #[must_use]
    pub fn new(
        config: OrchestratorConfig,
        ids: PersistentShardIdAllocator,
        publisher: EpochPublisher,
        peer_token: String,
    ) -> Self {
        Self {
            config,
            ids,
            publisher,
            autoscaler: Autoscaler::new(),
            peer_token,
            unclaimed_root_owner: None,
        }
    }

    /// Record `shard` as the pre-allocated root owner no live shard has claimed
    /// yet. It must own a region in the initial map. The first hello for that
    /// region that names no id (or names this id) adopts it, and the marker is
    /// cleared.
    pub fn set_unclaimed_root_owner(&mut self, shard: ShardId) {
        self.unclaimed_root_owner = Some(shard);
    }

    /// The reserved root owner still waiting for a live shard, if any.
    #[must_use]
    pub fn unclaimed_root_owner(&self) -> Option<ShardId> {
        self.unclaimed_root_owner
    }

    /// The published map (every reply carries this, never a stale copy).
    pub fn map(&self) -> &ShardMap {
        self.publisher.published()
    }

    /// Handle a `ShardHello`: allocate or confirm the id, place the region, and
    /// answer with the id, region, current map and peer token.
    ///
    /// A hello naming a different cluster is refused (the caller drops the
    /// connection); a hello claiming a region already owned by another shard is
    /// refused (the first claim wins) — that includes a region-less hello, which
    /// means "the root"; a hello with `ShardId::NONE` gets a fresh allocation for
    /// a free node. The one exception is the pre-allocated root owner
    /// ([`Self::set_unclaimed_root_owner`]): the first anonymous (or by-id) hello
    /// for the root adopts that id, since no shard was ever launched for it.
    pub fn handle_shard_hello(&mut self, hello: &ShardHello) -> Result<ShardWelcome, String> {
        if hello.cluster != self.config.cluster {
            return Err(format!(
                "hello names cluster {:?}, this is {:?}",
                hello.cluster, self.config.cluster
            ));
        }
        // Resolve the node first: a hello with no region means "the root".
        let node = hello
            .region
            .as_ref()
            .map_or_else(|| self.config.root_node(), |claimed| claimed.node);

        // Adopt the pre-allocated root owner: the orchestrator reserved that id
        // at startup, but nothing launches a shard for it, so the first live
        // shard asking for the root (anonymously, or by that id) becomes it.
        let adoptable = self.unclaimed_root_owner.filter(|owner| {
            (hello.shard == ShardId::NONE || hello.shard == *owner)
                && self
                    .map()
                    .regions
                    .iter()
                    .any(|r| r.node == node && r.shard == *owner)
        });
        let shard = if let Some(owner) = adoptable {
            // The owner was allocated at startup, so this is normally a no-op;
            // it keeps the "every confirmed id is reserved" invariant local.
            self.ids
                .reserve(owner)
                .map_err(|e| format!("could not persist the reservation of {}: {e}", owner.0))?;
            self.unclaimed_root_owner = None;
            let mut endpoints = std::collections::BTreeMap::new();
            endpoints.insert(owner, hello.endpoint);
            self.publisher.set_endpoints(&endpoints);
            owner
        } else {
            // The first claim on a node wins; a conflicting claim (including a
            // second region-less hello for the root) is refused, and refused
            // before any id is allocated or reserved.
            if self
                .map()
                .regions
                .iter()
                .any(|r| r.node == node && r.shard != hello.shard)
            {
                return Err(format!("region at depth {} is already owned", node.depth));
            }
            if hello.shard == ShardId::NONE {
                self.ids.allocate()
            } else {
                // A relaunch (or an orchestrated child booting with the id the
                // orchestrator allocated for it): confirm the id and make sure
                // the allocator never hands it out again. The endpoint may
                // legitimately differ from a placeholder the split staged, so
                // the known-id path always confirms and lets the region
                // endpoint be refreshed below.
                self.ids.reserve(hello.shard).map_err(|e| {
                    format!(
                        "could not persist the reservation of {}: {e}",
                        hello.shard.0
                    )
                })?;
                hello.shard
            }
        };
        let region = ShardRegion::new(node, shard, hello.endpoint);
        Ok(ShardWelcome {
            shard,
            region,
            shard_map: self.map().clone(),
            peer_token: self.peer_token.clone(),
        })
    }

    /// Handle a `LoadReport`: feed the autoscaler and return its decision.
    ///
    /// The returned action is *advisory*: the orchestrator turns it into a
    /// topology plan and an epoch proposal, and may drop it (an epoch is already
    /// in flight, the shard is mid-handoff).
    #[must_use]
    pub fn handle_load_report(
        &mut self,
        report: &LoadReport,
        tick: u64,
    ) -> Option<crate::autoscale::ScaleAction> {
        self.autoscaler
            .on_reports(std::slice::from_ref(report), tick)
    }

    /// Handle an `AdminCommand`.
    ///
    /// `Status` answers from the published map. Verbs whose machinery is a
    /// later slice (spawn/kill via the launcher, bots via the Phase-7 harness)
    /// return an honest `Refused` — never a panic, never a fake `Accepted`.
    #[must_use]
    pub fn handle_admin(&self, command: &AdminCommand) -> AdminReply {
        match command {
            AdminCommand::Status => AdminReply::Status {
                lines: self
                    .map()
                    .regions
                    .iter()
                    .map(|r| {
                        format!(
                            "epoch {} depth {} {:?} -> {} @ {}",
                            self.map().epoch,
                            r.node.depth,
                            r.node.index,
                            r.shard,
                            r.endpoint
                        )
                    })
                    .collect(),
            },
            AdminCommand::SpawnBots { .. } => AdminReply::Refused {
                command: command.clone(),
                reason: "bots arrive with the Phase-7 load harness".into(),
            },
            AdminCommand::ForceSplit(_)
            | AdminCommand::ForceMerge(_)
            | AdminCommand::KillShard(_) => AdminReply::Refused {
                command: command.clone(),
                reason: "the launcher front end lands in the next Phase-3 slice".into(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// The async control front end (plan §3.5).
// ---------------------------------------------------------------------------

/// The TLS server name both sides present (matches `space_server::net::SERVER_NAME`).
pub const SERVER_NAME: &str = "space.local";

/// How many frames may sit unsent to one shard before the sender is considered
/// stalled. Epoch traffic is a handful of messages; this only bounds growth
/// against a shard that has stopped reading.
const OUT_QUEUE: usize = 256;

/// An event the front end hands to the run loop.
#[derive(Debug)]
pub enum Inbound {
    /// A shard wants an id + region + map. `reply` is the shard's outbound
    /// channel: the run loop sends the [`ShardWelcome`] on it and registers it
    /// for future epoch traffic.
    Hello {
        /// The hello.
        hello: ShardHello,
        /// The shard's outbound sender (the run loop registers it under the id
        /// the welcome names).
        reply: mpsc::Sender<Message>,
    },
    /// A shard's periodic load report.
    Report(LoadReport),
    /// A shard acknowledged a prepared epoch.
    Ack(EpochAck),
}

/// The orchestrator's QUIC front end.
///
/// Binds one endpoint and speaks the shard↔orchestrator protocol: a shard
/// connects (the ordinary [`space_net`] hello handshake), sends a [`ShardHello`],
/// and receives a [`ShardWelcome`] carrying its id, region, map and peer token.
/// Thereafter the shard sends [`LoadReport`]s and [`EpochAck`]s and receives
/// epoch messages. The front end **only moves frames**: every decision is a call
/// into [`ControlPlane`] on the run loop. It mirrors `space_server::net::ShardNet`
/// (one task per connection, a FIFO sender per shard) so reviewers see one idiom.
#[derive(Debug)]
pub struct ControlServer {
    endpoint: quinn::Endpoint,
    /// One FIFO sender per registered shard (epoch messages go out through it).
    registry: Arc<Mutex<HashMap<ShardId, mpsc::Sender<Message>>>>,
    /// The run loop's inbound receiver.
    inbound: Arc<Mutex<mpsc::Receiver<Inbound>>>,
    /// A clone of the inbound sender, handed to each connection task.
    tx: mpsc::Sender<Inbound>,
}

impl ControlServer {
    /// Bind the control endpoint on `bind`.
    ///
    /// # Errors
    ///
    /// [`io::Error`] when the socket cannot be bound.
    pub fn bind(bind: SocketAddr) -> io::Result<Self> {
        let (config, _cert) = space_net::crypto::make_server_config();
        let endpoint = space_net::server_endpoint(bind, config)?;
        let (tx, rx) = mpsc::channel::<Inbound>(OUT_QUEUE);
        Ok(Self {
            endpoint,
            registry: Arc::new(Mutex::new(HashMap::new())),
            inbound: Arc::new(Mutex::new(rx)),
            tx,
        })
    }

    /// The bound address (the `:0` case resolves to a real port).
    ///
    /// # Errors
    ///
    /// [`io::Error`] if the socket was closed.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.endpoint.local_addr()
    }

    /// Start accepting connections; each is handled in its own task. Returns the
    /// accept task's handle so a caller can abort it on shutdown.
    pub fn spawn_accept(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            loop {
                let Some(incoming) = self.endpoint.accept().await else {
                    return; // endpoint closed
                };
                let server = Arc::clone(&self);
                tokio::spawn(async move {
                    server.handle_connection(incoming).await;
                });
            }
        })
    }

    /// The next inbound event, or `None` when every sender is gone.
    pub async fn next_inbound(&self) -> Option<Inbound> {
        self.inbound.lock().await.recv().await
    }

    /// Register `shard`'s outbound sender (called by the run loop after a hello).
    pub async fn register(&self, shard: ShardId, sender: mpsc::Sender<Message>) {
        self.registry.lock().await.insert(shard, sender);
    }

    /// Send `message` to `shard`, if it is registered. A shard that is not (or no
    /// longer) connected drops the message — the epoch protocol tolerates a lost
    /// push (a shard learns the new epoch from the next `Redirect`/`StaleEpoch`).
    pub async fn send_to(&self, shard: ShardId, message: Message) {
        if let Some(tx) = self.registry.lock().await.get(&shard).cloned() {
            let _ = tx.send(message).await;
        }
    }

    /// Forget a shard's sender (after a kill).
    pub async fn unregister(&self, shard: ShardId) {
        self.registry.lock().await.remove(&shard);
    }

    /// Stop accepting.
    pub fn close(&self) {
        self.endpoint
            .close(0u32.into(), b"orchestrator shutting down");
    }

    /// Drive one shard's control connection.
    async fn handle_connection(self: Arc<Self>, incoming: quinn::Incoming) {
        let Ok((conn, _hello)) = space_net::accept_incoming(incoming).await else {
            return; // refused at the version gate
        };
        // `space_net::connect` blocks until it sees a Welcome; answer the
        // handshake with a placeholder so the shard's connect() completes, then
        // read the real ShardHello on the control stream.
        let placeholder = Message::Welcome(space_protocol::Welcome {
            session: 0,
            shard: ShardId::NONE,
            shard_map: ShardMap::new(1, Vec::new()),
        });
        if conn.send_control(&placeholder).await.is_err() {
            return;
        }
        // The first real message must be a ShardHello.
        let hello = match conn.recv_control().await {
            Ok(Some(Message::ShardHello(h))) => h,
            _ => return,
        };
        // This shard's outbound channel: the run loop registers it and writes
        // the welcome + every epoch message through it, in order.
        let (tx, mut rx) = mpsc::channel::<Message>(OUT_QUEUE);
        let writer = {
            let conn = conn.clone();
            tokio::spawn(async move {
                while let Some(message) = rx.recv().await {
                    if conn.send_reliable(&message).await.is_err() {
                        return;
                    }
                }
            })
        };
        if self
            .tx
            .send(Inbound::Hello {
                hello,
                reply: tx.clone(),
            })
            .await
            .is_err()
        {
            writer.abort();
            return;
        }
        // Read loop: load reports and epoch acks, forwarded to the run loop.
        loop {
            if conn.is_closed() {
                break;
            }
            let next = tokio::select! {
                m = conn.recv_control() => m,
                m = conn.recv_stream() => m,
                m = conn.recv_datagram() => m,
            };
            match next {
                Ok(Some(Message::LoadReport(report))) => {
                    if self.tx.send(Inbound::Report(report)).await.is_err() {
                        break;
                    }
                }
                Ok(Some(Message::EpochAck(ack))) => {
                    if self.tx.send(Inbound::Ack(ack)).await.is_err() {
                        break;
                    }
                }
                Ok(Some(_)) => {}
                _ => break,
            }
        }
        writer.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::MemoryCounterSink;
    use space_protocol::control::AdminCommand;
    use space_protocol::shard::ShardMap;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    fn addr(n: u16) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), n)
    }

    fn plane() -> ControlPlane {
        let config = OrchestratorConfig::default();
        let map = ShardMap::new(
            1,
            vec![ShardRegion::new(
                config.root_node(),
                ShardId::new(1),
                addr(7_001),
            )],
        );
        let ids = PersistentShardIdAllocator::with_sink(1, Box::new(MemoryCounterSink::default()));
        ControlPlane::new(config, ids, EpochPublisher::new(map), "secret".into())
    }

    #[test]
    fn a_hello_with_no_id_gets_a_fresh_allocation_and_the_current_map() {
        // The orchestrator pre-allocated root owner 1 at startup; the first
        // region-less hello adopts it (no fresh allocation burns an id).
        let mut plane = plane();
        plane.set_unclaimed_root_owner(ShardId::new(1));
        let hello = ShardHello {
            cluster: plane.config.cluster.clone(),
            shard: ShardId::NONE,
            endpoint: addr(9_000),
            region: None,
        };
        let welcome = plane.handle_shard_hello(&hello).expect("welcome");
        assert_eq!(welcome.shard, ShardId::new(1), "adopted the root owner");
        assert_eq!(welcome.region.shard, welcome.shard);
        assert_eq!(welcome.region.endpoint, addr(9_000));
        assert_eq!(plane.unclaimed_root_owner(), None, "now claimed");
        assert_eq!(
            welcome.shard_map,
            *plane.map(),
            "the welcome carries the current map"
        );
        let root = welcome
            .shard_map
            .regions
            .iter()
            .find(|r| r.shard == ShardId::new(1))
            .expect("root region");
        assert_eq!(
            root.endpoint,
            addr(9_000),
            "the map names the live shard's endpoint, not the placeholder"
        );
        assert_eq!(welcome.peer_token, "secret");
    }

    #[test]
    fn a_second_region_less_hello_for_the_root_is_refused() {
        let mut plane = plane();
        plane.set_unclaimed_root_owner(ShardId::new(1));
        let hello = |port| ShardHello {
            cluster: plane.config.cluster.clone(),
            shard: ShardId::NONE,
            endpoint: addr(port),
            region: None,
        };
        let (first, second) = (hello(9_000), hello(9_001));
        plane.handle_shard_hello(&first).expect("first adopts");
        assert!(
            plane.handle_shard_hello(&second).is_err(),
            "the root is already claimed by a live shard"
        );
        // The refusal burned no id.
        assert_eq!(plane.ids.peek(), 1);
    }

    #[test]
    fn a_no_region_hello_claiming_a_non_root_id_is_refused_when_the_root_is_owned() {
        let mut plane = plane();
        plane.set_unclaimed_root_owner(ShardId::new(1));
        let hello = ShardHello {
            cluster: plane.config.cluster.clone(),
            shard: ShardId::new(2),
            endpoint: addr(9_000),
            region: None,
        };
        assert!(plane.handle_shard_hello(&hello).is_err());
        assert_eq!(
            plane.unclaimed_root_owner(),
            Some(ShardId::new(1)),
            "a refused hello leaves the reservation intact"
        );
    }

    #[test]
    fn confirming_a_known_id_reserves_it_in_the_allocator() {
        let mut plane = plane();
        let hello = ShardHello {
            cluster: plane.config.cluster.clone(),
            shard: ShardId::new(1),
            endpoint: addr(9_000),
            region: None,
        };
        plane.handle_shard_hello(&hello).expect("owner confirms");
        assert_eq!(plane.ids.peek(), 2, "id 1 can never be allocated again");
    }

    #[test]
    fn a_shard_claiming_the_next_to_issue_id_does_not_cause_a_collision() {
        // Counter at 100; the root is staged for owner 100 (the id the
        // orchestrator is about to hand out elsewhere).
        let config = OrchestratorConfig::default();
        let root = config.root_node();
        let map = ShardMap::new(
            1,
            vec![ShardRegion::new(root, ShardId::new(100), addr(7_001))],
        );
        let ids =
            PersistentShardIdAllocator::with_sink(100, Box::new(MemoryCounterSink::default()));
        let mut plane = ControlPlane::new(config, ids, EpochPublisher::new(map), "secret".into());

        let claim = ShardHello {
            cluster: plane.config.cluster.clone(),
            shard: ShardId::new(100),
            endpoint: addr(9_000),
            region: None,
        };
        let welcome = plane.handle_shard_hello(&claim).expect("claim confirmed");
        assert_eq!(welcome.shard, ShardId::new(100));

        // A later anonymous hello (for a free child node) must not get 100.
        let child = space_world::octree::children(root).expect("root splits")[0];
        let anon = ShardHello {
            cluster: plane.config.cluster.clone(),
            shard: ShardId::NONE,
            endpoint: addr(9_001),
            region: Some(ShardRegion::new(child, ShardId::NONE, addr(9_001))),
        };
        let second = plane.handle_shard_hello(&anon).expect("allocation");
        assert_eq!(second.shard, ShardId::new(101), "100 was already taken");
        assert_ne!(second.shard, welcome.shard, "no id is handed out twice");
    }

    #[test]
    fn a_hello_naming_another_cluster_is_refused() {
        let mut plane = plane();
        let hello = ShardHello {
            cluster: "wrong-cluster".into(),
            shard: ShardId::NONE,
            endpoint: addr(9_000),
            region: None,
        };
        assert!(plane.handle_shard_hello(&hello).is_err());
    }

    #[test]
    fn a_hello_claiming_an_owned_region_is_refused() {
        let mut plane = plane();
        // Shard 2 asks for the root, which shard 1 already owns: the first
        // claim wins, so this is refused.
        let hello = ShardHello {
            cluster: plane.config.cluster.clone(),
            shard: ShardId::new(2),
            endpoint: addr(9_000),
            region: Some(ShardRegion::new(
                plane.config.root_node(),
                ShardId::new(2),
                addr(9_000),
            )),
        };
        assert!(
            plane.handle_shard_hello(&hello).is_err(),
            "the root is owned by shard 1"
        );
    }

    #[test]
    fn admin_status_lists_the_published_regions() {
        let plane = plane();
        match plane.handle_admin(&AdminCommand::Status) {
            AdminReply::Status { lines } => assert_eq!(lines.len(), 1),
            other => panic!("expected Status, got {other:?}"),
        }
    }

    #[test]
    fn spawn_bots_is_refused_not_panicked() {
        // D4: an honest Refused, never a todo!() in a CI-run path.
        let plane = plane();
        match plane.handle_admin(&AdminCommand::SpawnBots { count: 5 }) {
            AdminReply::Refused { reason, .. } => assert!(reason.contains("Phase-7")),
            other => panic!("expected Refused, got {other:?}"),
        }
    }

    #[test]
    fn a_load_report_feeds_the_autoscaler() {
        let mut plane = plane();
        let report = LoadReport {
            shard: ShardId::new(1),
            players: 90,
            tick_p95_ms: 1.0,
            load: 0.0,
        };
        // Below the dwell: no action yet.
        assert_eq!(plane.handle_load_report(&report, 10), None);
    }
}
