//! The control-plane message handlers (plan §3.1).
//!
//! Pure handlers: given a message and the orchestrator's state, produce the
//! reply to send and the state change to apply. The QUIC front end (a later
//! slice) only moves frames; every decision lives here so it is testable without
//! a socket.
//!
//! The handlers implemented in this slice:
//! * [`ControlPlane::check_credentials`] — the way in: the **cluster secret**
//!   when one is required (one was configured, or the control port is not on
//!   loopback), compared in constant time and checked first; then the cluster
//!   name.
//! * [`ControlPlane::handle_shard_hello`] — a hello that *claims* what the
//!   published map already says: the reserved root (by a region-less hello, or
//!   by its id), or an owner's first registration. It confirms the id and
//!   returns a [`ShardWelcome`] carrying the current map and the cluster's
//!   `peer_token` (which shards present to each other and refuse a peer
//!   without). It never allocates an id and never creates a region: a shard
//!   that registers without a region once the root is taken is given one by
//!   [`crate::orchestrator::Orchestrator::handle_shard_hello`], through an
//!   epoch change.
//! * [`ControlPlane::handle_load_report`] — feed the autoscaler.
//! * [`ControlPlane::handle_admin`] — the admin verbs. Verbs whose machinery
//!   arrives with a later slice return an honest [`AdminReply::Refused`] rather
//!   than panicking (D4: no `todo!()` in a CI-run path).

use std::collections::HashMap;
use std::io;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use space_net::Conn;
use space_protocol::control::{AdminCommand, AdminReply, LoadReport};
use space_protocol::shard::{EpochAck, ShardId, ShardMap, ShardRegion};
use space_protocol::{Message, RejectReason, SceneAck, ShardHello, ShardWelcome};
use tokio::sync::{mpsc, oneshot, Mutex};

use crate::autoscale::Autoscaler;
use crate::config::{OrchestratorConfig, Secret};
use crate::epoch::EpochPublisher;
use crate::ids::PersistentShardIdAllocator;
use crate::scene::SceneStore;

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
    /// The shared peer secret handed to every shard on welcome. Shards present
    /// it to each other and refuse a peer hello without it; empty = none is
    /// issued (peers are then accepted on map membership alone). A
    /// [`Secret`]: this struct derives `Debug`, and so does the orchestrator
    /// around it.
    pub peer_token: Secret,
    /// The cluster secret: what a `ShardHello` must carry when
    /// [`Self::secret_required`], and what the orchestrator hands the shards it
    /// starts itself. The configured one, or one [`crate::boot::boot`]
    /// generated; empty when there is neither.
    cluster_secret: Secret,
    /// Whether a hello without the right cluster secret is refused
    /// ([`OrchestratorConfig::secret_required`]).
    secret_required: bool,
    /// The cluster's scene, its revision and where it is persisted. Not part
    /// of the map and not published through the epoch protocol
    /// ([`crate::scene`]). [`Self::new`] starts it in memory at the configured
    /// initial scene; [`crate::boot::boot`] replaces it with the file-backed
    /// store.
    pub scene: SceneStore,
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
            scene: SceneStore::in_memory(config.initial_scene),
            cluster_secret: config.cluster_secret.clone().unwrap_or_default(),
            secret_required: config.secret_required(),
            config,
            ids,
            publisher,
            autoscaler: Autoscaler::new(),
            peer_token: Secret::new(peer_token),
            unclaimed_root_owner: None,
        }
    }

    /// The cluster secret (see the field docs); empty when there is none.
    #[must_use]
    pub fn cluster_secret(&self) -> &Secret {
        &self.cluster_secret
    }

    /// Whether a `ShardHello` must carry the cluster secret.
    #[must_use]
    pub fn secret_required(&self) -> bool {
        self.secret_required
    }

    /// Use `secret` as the cluster secret when none was configured: what
    /// [`crate::boot::boot`] does with the one it generates, so that the
    /// shards the orchestrator starts have one to present. A configured secret
    /// is never replaced.
    pub fn set_generated_secret(&mut self, secret: Secret) {
        if self.config.cluster_secret.is_none() {
            self.cluster_secret = secret;
        }
    }

    /// The way in for every `ShardHello`: when a secret is required it must
    /// carry the cluster secret, and the cluster it names must be this one.
    /// Both refusals are final.
    ///
    /// **The secret is checked first.** Whoever does not hold it is told
    /// [`HelloRefusal::Unauthorized`] and nothing else — in particular not
    /// what this cluster is called, which the refusal for a wrong cluster
    /// names. (Where no secret is required — a loopback control port with
    /// none configured — every local process is trusted with that anyway.)
    ///
    /// The secret is compared in constant time and never appears in the
    /// refusal (or anywhere else).
    ///
    /// # Errors
    ///
    /// [`HelloRefusal::Unauthorized`] for a missing or wrong secret,
    /// [`HelloRefusal::Refused`] for another cluster.
    pub fn check_credentials(&self, hello: &ShardHello) -> Result<(), HelloRefusal> {
        if self.secret_required
            && !space_net::constant_time_eq(
                hello.secret.as_bytes(),
                self.cluster_secret.expose().as_bytes(),
            )
        {
            return Err(HelloRefusal::Unauthorized);
        }
        if hello.cluster != self.config.cluster {
            return Err(HelloRefusal::Refused(format!(
                "hello names cluster {:?}, this is {:?}",
                hello.cluster, self.config.cluster
            )));
        }
        Ok(())
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

    /// Handle a `ShardHello` that **claims** what the published map already
    /// says, and answer with the id, region, current map and peer token.
    ///
    /// Three hellos are welcomed here:
    ///
    /// * the first hello for the root — region-less and id-less, naming the
    ///   root node, or naming the reserved id — adopts the pre-allocated root
    ///   owner ([`Self::set_unclaimed_root_owner`]), since no shard was ever
    ///   launched for it;
    /// * a hello naming an id that **owns the node it names** (the root when
    ///   it names none) is confirmed as that owner.
    ///
    /// Everything else is refused, and every refusal here is final: a hello
    /// naming another cluster; a claim on a node somebody else owns (the first
    /// claim wins) — that includes a region-less hello, which at this level
    /// means "the root"; a claim on a node the published map does not name,
    /// with or without an id (a region comes into being through an epoch
    /// change the orchestrator proposes for a shard that registered without
    /// one — never through a claim, which would get a region no map ever
    /// carried). No hello allocates an id here, and none moves the persistent
    /// counter beyond confirming an id the map already holds.
    ///
    /// What this handler does **not** decide is what a shard that registers
    /// with neither id nor region gets once the root is taken: that is
    /// [`crate::orchestrator::Orchestrator::handle_shard_hello`] (a dead
    /// shard's regions, the node a split asked for, or the next layout cell),
    /// which also turns a shard coming back at an owner's address into that
    /// owner's replacement **under a new id** — it used to get the same id
    /// back here, and with it a guid counter that starts over.
    ///
    /// # Errors
    ///
    /// Why the hello is refused, for the shard to print.
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
        let owner_of_node = self
            .map()
            .regions
            .iter()
            .find(|r| r.node == node)
            .map(|r| r.shard);
        let shard = if let Some(owner) = adoptable {
            owner
        } else {
            match owner_of_node {
                // The node is not in the published map. Welcoming such a claim
                // used to hand back a region that was never published: the
                // shard's own map (the welcome's) did not name it, so it
                // handed every ship it was given straight back to the map's
                // real owner, and two shards could "own" the same node.
                None => {
                    return Err(format!(
                        "the node at depth {} index {:?} is not in the published map; a region \
                         is given to a shard that registers without one, not claimed",
                        node.depth, node.index
                    ));
                }
                // The first claim on a node wins; a conflicting claim
                // (including a second region-less hello for the root) is
                // refused before anything is reserved.
                Some(owner) if owner != hello.shard => {
                    return Err(format!("region at depth {} is already owned", node.depth));
                }
                Some(owner) => owner,
            }
        };
        // The id owns a region of the map, so it was issued; reserving it is
        // normally a no-op and keeps the "every confirmed id is reserved"
        // invariant local (a map built by hand may name an id the counter has
        // not passed).
        self.ids
            .reserve(shard)
            .map_err(|e| format!("could not persist the reservation of {}: {e}", shard.0))?;
        if adoptable.is_some() {
            // Claimed (a refused hello leaves the reservation intact). The map
            // held a placeholder for the root's address until now: the live
            // shard's goes in at once, so the very first welcome and every
            // client's map name it.
            self.unclaimed_root_owner = None;
            let mut endpoints = std::collections::BTreeMap::new();
            endpoints.insert(shard, hello.endpoint);
            self.publisher.set_endpoints(&endpoints);
        }
        let region = ShardRegion::new(node, shard, hello.endpoint);
        Ok(ShardWelcome {
            shard,
            region,
            shard_map: self.map().clone(),
            peer_token: self.peer_token.expose().to_owned(),
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

/// Why a `ShardHello` was not answered with a welcome.
///
/// One refusal is not final: [`HelloRefusal::Wait`]. The orchestrator changes
/// the map one step at a time, so a shard that registers while a change is in
/// flight is told to ask again — as a [`RejectReason::Draining`] on the wire.
/// Everything else ends the shard that asked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HelloRefusal {
    /// Not yet, and why: ask again after a pause.
    Wait(String),
    /// The cluster secret was missing or wrong. Final.
    Unauthorized,
    /// The cluster has as many shards (or the map as many regions) as it may.
    /// Final.
    Full(String),
    /// Refused for the reason given (another cluster, a claim the map does
    /// not support, no region left to give). Final.
    Refused(String),
}

impl HelloRefusal {
    /// Whether the shard should give up (everything but [`Self::Wait`]).
    #[must_use]
    pub const fn is_final(&self) -> bool {
        !matches!(self, HelloRefusal::Wait(_))
    }

    /// The refusal as it goes on the wire.
    #[must_use]
    pub fn reject_reason(&self) -> RejectReason {
        match self {
            HelloRefusal::Wait(_) => RejectReason::Draining,
            HelloRefusal::Unauthorized => RejectReason::Unauthorized,
            HelloRefusal::Full(_) => RejectReason::Full,
            HelloRefusal::Refused(why) => RejectReason::Other(why.clone()),
        }
    }
}

impl core::fmt::Display for HelloRefusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            HelloRefusal::Wait(why) => write!(f, "not yet ({why}); ask again"),
            HelloRefusal::Unauthorized => f.write_str("the cluster secret is missing or wrong"),
            HelloRefusal::Full(why) | HelloRefusal::Refused(why) => f.write_str(why),
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

/// How long a freshly accepted control connection has to send its
/// [`ShardHello`] before it is closed.
///
/// The transport keeps a quiet connection alive, so without a bound a peer that
/// completes the transport handshake and then never says who it is (a game
/// client pointed at the control port, a stuck shard) would hold a task and a
/// connection for as long as it liked.
pub const HELLO_TIMEOUT: Duration = Duration::from_secs(5);

/// Application close code for a control connection that never sent its hello.
const NO_HELLO_CLOSE_CODE: u32 = 0x21;

/// Application close code for a control connection the orchestrator closes
/// because it has stopped the shard (merged away, replaced, its epoch aborted).
/// The shard sees its link lost and stops itself.
const SHARD_STOPPED_CLOSE_CODE: u32 = 0x22;

/// Application close code for a control connection whose `ShardHello` the run
/// loop never answered (it is stopping).
const NO_VERDICT_CLOSE_CODE: u32 = 0x23;

/// What the run loop decided about a connection's `ShardHello`: the id the
/// connection is admitted as — for its whole life — or the refusal to deliver
/// before the connection is closed.
pub type Admission = Result<ShardId, RejectReason>;

/// An event the front end hands to the run loop.
///
/// # Who may say what
///
/// The cluster secret gates the hello, and **the hello gates everything
/// else**. A connection's task hands its `ShardHello` to the run loop and then
/// waits for the verdict ([`Admission`]). Refused, the connection is told why
/// and closed; nothing else it sent is ever read. Admitted, it is admitted as
/// **one shard id**, and from then on a report or an ack is passed on only if
/// it names that id — `from` below is that id, taken from the connection and
/// never from the message. (These messages name their own sender; believing
/// that let anybody who could reach the control port report load for a shard,
/// acknowledge an epoch on its behalf, or confirm a scene for it.)
#[derive(Debug)]
pub enum Inbound {
    /// A shard wants an id + region + map. `reply` is the shard's outbound
    /// channel: the run loop sends the [`ShardWelcome`] on it and registers it
    /// for future epoch traffic.
    Hello {
        /// The hello.
        hello: ShardHello,
        /// The shard's outbound sender (the run loop registers it under the id
        /// the welcome names). Nothing put into it is written before the
        /// connection is admitted.
        reply: mpsc::Sender<Message>,
        /// The connection the hello arrived on: its remote address is where
        /// the shard is seen from, and the run loop keeps it with the sender so
        /// it can close the link of a shard it stops
        /// ([`ControlServer::close_link`]).
        conn: space_net::Conn,
        /// Where the run loop says what became of the hello. Dropping it
        /// without an answer closes the connection.
        verdict: oneshot::Sender<Admission>,
    },
    /// A shard's periodic load report.
    Report {
        /// The shard the connection it arrived on was admitted as.
        from: ShardId,
        /// The report; it names `from`.
        report: LoadReport,
    },
    /// A shard acknowledged a prepared epoch.
    Ack {
        /// The shard the connection it arrived on was admitted as.
        from: ShardId,
        /// The ack; it names `from`.
        ack: EpochAck,
    },
    /// A shard said which scene it runs (its answer to a `SceneSet`).
    SceneAck {
        /// The shard the connection it arrived on was admitted as.
        from: ShardId,
        /// The ack; it names `from`.
        ack: SceneAck,
    },
}

/// A registered shard's way out: its FIFO sender and, when it registered over
/// a real connection, that connection.
#[derive(Debug)]
struct Registered {
    sender: mpsc::Sender<Message>,
    conn: Option<space_net::Conn>,
}

/// The orchestrator's QUIC front end.
///
/// Binds one endpoint and speaks the shard↔orchestrator protocol: a shard
/// connects (the ordinary [`space_net`] hello handshake), sends a [`ShardHello`],
/// and receives a [`ShardWelcome`] carrying its id, region, map and peer token.
/// Thereafter the shard sends [`LoadReport`]s, [`EpochAck`]s and [`SceneAck`]s
/// and receives epoch messages and the cluster's scene. A connection whose
/// hello is refused is told why and closed; one that is welcomed speaks for
/// the id it was welcomed as and for no other ([`Inbound`]). The front end
/// **only moves frames**: every decision is a call
/// into [`ControlPlane`] on the run loop. It mirrors `space_server::net::ShardNet`
/// (one task per connection, a FIFO sender per shard) so reviewers see one idiom.
#[derive(Debug)]
pub struct ControlServer {
    endpoint: quinn::Endpoint,
    /// One FIFO sender per registered shard (epoch messages go out through
    /// it), and the connection it writes to when the shard registered over one.
    registry: Arc<Mutex<HashMap<ShardId, Registered>>>,
    /// The run loop's inbound receiver.
    inbound: Arc<Mutex<mpsc::Receiver<Inbound>>>,
    /// A clone of the inbound sender, handed to each connection task.
    tx: mpsc::Sender<Inbound>,
    /// How long a connection may take to send its `ShardHello`.
    hello_timeout: Duration,
    /// Reports and acks that named another shard than the one their
    /// connection was admitted as, and were dropped for it
    /// ([`Self::misattributed`]).
    misattributed: AtomicU64,
    /// How many of those are written to standard error (a registered shard
    /// can send as many as it likes).
    misattributed_lines: space_net::LogBudget,
}

/// How many dropped reports and acks ([`ControlServer::misattributed`]) are
/// written to standard error per [`MISATTRIBUTED_WINDOW`]; the rest are
/// counted and summed up in the next line that is.
pub const MISATTRIBUTED_LINES: u32 = 3;

/// The window [`MISATTRIBUTED_LINES`] applies to.
pub const MISATTRIBUTED_WINDOW: Duration = Duration::from_secs(10);

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
            hello_timeout: HELLO_TIMEOUT,
            misattributed: AtomicU64::new(0),
            misattributed_lines: space_net::LogBudget::new(
                MISATTRIBUTED_LINES,
                MISATTRIBUTED_WINDOW,
            ),
        })
    }

    /// How many reports and acks were dropped because they named another
    /// shard than the one the connection they arrived on was admitted as.
    /// Zero in a cluster of honest shards.
    #[must_use]
    pub fn misattributed(&self) -> u64 {
        self.misattributed.load(Ordering::Relaxed)
    }

    /// Use `limit` instead of [`HELLO_TIMEOUT`] as the bound on a connection's
    /// `ShardHello` (a test that wants the bound to run out quickly).
    #[must_use]
    pub fn with_hello_timeout(mut self, limit: Duration) -> Self {
        self.hello_timeout = limit;
        self
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

    /// Register `shard`'s outbound sender, with no connection behind it (a
    /// test wiring a queue by hand).
    pub async fn register(&self, shard: ShardId, sender: mpsc::Sender<Message>) {
        self.registry
            .lock()
            .await
            .insert(shard, Registered { sender, conn: None });
    }

    /// Register `shard`'s outbound sender and the connection it registered
    /// over (called by the run loop after a hello it welcomed).
    pub async fn register_link(
        &self,
        shard: ShardId,
        sender: mpsc::Sender<Message>,
        conn: space_net::Conn,
    ) {
        self.registry.lock().await.insert(
            shard,
            Registered {
                sender,
                conn: Some(conn),
            },
        );
    }

    /// Forget `shard` and **close its control connection**: the orchestrator
    /// has stopped it (its region was merged away, it was replaced, or the
    /// change that would have given it a region aborted). A shard does not run
    /// without its orchestrator, so closing the link is what stops a shard no
    /// launcher of ours can reach — one started by hand, or by a container
    /// runtime that will start a fresh one.
    pub async fn close_link(&self, shard: ShardId) {
        let registered = self.registry.lock().await.remove(&shard);
        if let Some(conn) = registered.and_then(|r| r.conn) {
            conn.close(SHARD_STOPPED_CLOSE_CODE, "stopped by the orchestrator");
        }
    }

    /// Where `shard`'s control connection comes from (the address the
    /// orchestrator sees the shard at), if it registered over one.
    pub async fn remote_addr(&self, shard: ShardId) -> Option<SocketAddr> {
        self.registry
            .lock()
            .await
            .get(&shard)
            .and_then(|r| r.conn.as_ref().map(space_net::Conn::remote_address))
    }

    /// Offer `message` to `shard`'s FIFO sender, if it is registered. Returns
    /// whether the message was queued.
    ///
    /// A shard that is not (or no longer) connected drops the message — the
    /// epoch protocol tolerates a lost push (a prepare that is never acked
    /// aborts at its deadline; a shard learns a missed commit from the next map
    /// push or `Redirect`/`StaleEpoch`).
    ///
    /// **This never waits.** It is called from the run loop, so the sender is
    /// cloned out of the registry (the lock is released before anything else
    /// happens) and the message is *offered* with `try_send`: a shard whose
    /// queue is full — it stopped reading — loses the message, logged, instead
    /// of stopping the orchestrator's clock and every other shard's epoch
    /// traffic behind it.
    pub async fn send_to(&self, shard: ShardId, message: Message) -> bool {
        let sender = self
            .registry
            .lock()
            .await
            .get(&shard)
            .map(|r| r.sender.clone());
        let Some(tx) = sender else {
            return false;
        };
        match tx.try_send(message) {
            Ok(()) => true,
            Err(mpsc::error::TrySendError::Full(_)) => {
                eprintln!(
                    "space-orchestrator: the queue for {shard} is full (has it stopped reading?); \
                     message dropped"
                );
                false
            }
            // The connection is gone; its writer went with it.
            Err(mpsc::error::TrySendError::Closed(_)) => false,
        }
    }

    /// Forget a shard's sender without closing its connection (see
    /// [`Self::close_link`] for the usual end of a registration).
    pub async fn unregister(&self, shard: ShardId) {
        self.registry.lock().await.remove(&shard);
    }

    /// The registered shards whose control connection is still open.
    ///
    /// A shard's sender closes when its connection task ends (the peer closed,
    /// the transport timed out, or the writer failed), so "registered and not
    /// closed" is what the front end knows about liveness. The run loop hands
    /// it to [`crate::orchestrator::Orchestrator::note_connected`], which takes
    /// a registered shard whose link has closed for dead (a shard does not
    /// outlive its link). Like [`Self::send_to`] this never waits on a peer:
    /// the registry lock is only ever held for a map operation.
    pub async fn connected(&self) -> Vec<ShardId> {
        self.registry
            .lock()
            .await
            .iter()
            .filter(|(_, registered)| !registered.sender.is_closed())
            .map(|(shard, _)| *shard)
            .collect()
    }

    /// Stop accepting.
    pub fn close(&self) {
        self.endpoint
            .close(0u32.into(), b"orchestrator shutting down");
    }

    /// Count — and, a few times a window, say — that a report or an ack
    /// naming `named` arrived on the connection admitted as `admitted` and was
    /// dropped.
    fn drop_misattributed(&self, what: &str, named: ShardId, admitted: ShardId, conn: &Conn) {
        self.misattributed.fetch_add(1, Ordering::Relaxed);
        if let Some(held_back) = self.misattributed_lines.admit() {
            eprintln!(
                "space-orchestrator: dropped {what} naming {named} that arrived on the control \
                 connection of {admitted} ({}): a shard speaks for itself only{}",
                conn.remote_address(),
                space_net::held_back_note(held_back)
            );
        }
    }

    /// Drive one shard's control connection (see [`Inbound`], "Who may say
    /// what"): hello, verdict, and only then — and only for the id it was
    /// admitted as — reports and acks.
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
        // The first real message must be a ShardHello, and it must arrive
        // promptly (see [`HELLO_TIMEOUT`]).
        let hello = match tokio::time::timeout(self.hello_timeout, conn.recv_control()).await {
            Ok(Ok(Some(Message::ShardHello(h)))) => h,
            Ok(_) => return,
            Err(_) => {
                conn.close(NO_HELLO_CLOSE_CODE, "no ShardHello");
                return;
            }
        };
        // This shard's outbound channel. The run loop gets its only sender: it
        // registers it under the id it admits the connection as and writes
        // the welcome and every epoch message through it, in order.
        let (tx, mut rx) = mpsc::channel::<Message>(OUT_QUEUE);
        let (verdict, admission) = oneshot::channel::<Admission>();
        let hello = Inbound::Hello {
            hello,
            reply: tx,
            conn: conn.clone(),
            verdict,
        };
        if self.tx.send(hello).await.is_err() {
            return; // the run loop is gone
        }
        // Nothing else this connection says is read before the verdict is in:
        // the cluster secret gates the hello, and the hello gates the rest.
        let admitted = match admission.await {
            Ok(Ok(shard)) => shard,
            Ok(Err(reason)) => {
                // Refused. The reason has to *arrive* — "not yet" is what
                // makes a shard ask again instead of giving up — so it is
                // written, finished and given a moment to be read before the
                // connection is closed (`Conn::reject`; the shard closes as
                // soon as it has read it). A shard that asks again dials a
                // new connection, so there is nothing more to read on this
                // one: whatever else it sent is never looked at.
                conn.reject(reason).await;
                return;
            }
            Err(_) => {
                // The run loop let go of the hello without an answer (it is
                // stopping).
                conn.close(NO_VERDICT_CLOSE_CODE, "no answer to the ShardHello");
                return;
            }
        };
        // Admitted, as `admitted` and nobody else, for as long as the
        // connection lives. What the run loop queued — the welcome first — is
        // written from here on.
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
        // Read loop: load reports and acks, forwarded to the run loop under
        // the id of this connection. One that names another shard is dropped.
        loop {
            if conn.is_closed() {
                break;
            }
            let next = tokio::select! {
                m = conn.recv_control() => m,
                m = conn.recv_stream() => m,
                m = conn.recv_datagram() => m,
            };
            let forward = match next {
                Ok(Some(Message::LoadReport(report))) if report.shard == admitted => {
                    Inbound::Report {
                        from: admitted,
                        report,
                    }
                }
                Ok(Some(Message::EpochAck(ack))) if ack.shard == admitted => Inbound::Ack {
                    from: admitted,
                    ack,
                },
                Ok(Some(Message::SceneAck(ack))) if ack.shard == admitted => Inbound::SceneAck {
                    from: admitted,
                    ack,
                },
                Ok(Some(Message::LoadReport(report))) => {
                    self.drop_misattributed("a load report", report.shard, admitted, &conn);
                    continue;
                }
                Ok(Some(Message::EpochAck(ack))) => {
                    self.drop_misattributed("an epoch ack", ack.shard, admitted, &conn);
                    continue;
                }
                Ok(Some(Message::SceneAck(ack))) => {
                    self.drop_misattributed("a scene ack", ack.shard, admitted, &conn);
                    continue;
                }
                // A second hello, or anything else: a connection is admitted
                // once, and a shard's vocabulary here is reports and acks.
                Ok(Some(_)) => continue,
                _ => break,
            };
            if self.tx.send(forward).await.is_err() {
                break;
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
            secret: String::new(),
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
            secret: String::new(),
        };
        let (first, second) = (hello(9_000), hello(9_001));
        plane.handle_shard_hello(&first).expect("first adopts");
        let after_adoption = plane.ids.peek();
        assert!(
            plane.handle_shard_hello(&second).is_err(),
            "the root is already claimed by a live shard"
        );
        // The refusal burned no id.
        assert_eq!(plane.ids.peek(), after_adoption);
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
            secret: String::new(),
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
            secret: String::new(),
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
            secret: String::new(),
        };
        let welcome = plane.handle_shard_hello(&claim).expect("claim confirmed");
        assert_eq!(welcome.shard, ShardId::new(100));

        // The next allocation (a split's child) must not get 100.
        let next = plane.ids.allocate().expect("allocation");
        assert_eq!(next, ShardId::new(101), "100 was already taken");
        assert_ne!(next, welcome.shard, "no id is handed out twice");
    }

    /// An anonymous hello for a node the published map does not name is refused
    /// and burns no id: a region is created by a split, not by claiming it.
    ///
    /// Mutation this catches: allocating an id for such a claim again — the
    /// shard would be welcomed onto a region no map carries (its own map
    /// included), and every restart of it would burn another id.
    #[test]
    fn an_anonymous_claim_for_a_node_outside_the_published_map_is_refused() {
        let mut plane = plane();
        plane.set_unclaimed_root_owner(ShardId::new(1));
        plane
            .handle_shard_hello(&ShardHello {
                cluster: plane.config.cluster.clone(),
                shard: ShardId::NONE,
                endpoint: addr(9_000),
                region: None,
                secret: String::new(),
            })
            .expect("adopts the root");
        let (counter, map) = (plane.ids.peek(), plane.map().clone());
        let child = space_world::octree::children(plane.config.root_node()).expect("splits")[0];
        let claim = ShardHello {
            cluster: plane.config.cluster.clone(),
            shard: ShardId::NONE,
            endpoint: addr(9_001),
            region: Some(ShardRegion::new(child, ShardId::NONE, addr(9_001))),
            secret: String::new(),
        };
        let err = plane
            .handle_shard_hello(&claim)
            .expect_err("no region is created by claiming it");
        assert!(err.contains("not in the published map"), "{err}");
        assert_eq!(plane.ids.peek(), counter, "the refusal burned no id");
        assert_eq!(plane.map(), &map, "and the published map is untouched");
        // Asking again changes nothing (it used to allocate a fresh id each time).
        assert!(plane.handle_shard_hello(&claim).is_err());
        assert_eq!(plane.ids.peek(), counter);
    }

    /// A claim on a node the published map does not name is refused **with an
    /// id too** — even one the allocator has issued. This used to be the way a
    /// split's child came in (launched with the id and the node allocated for
    /// it, welcomed for a region no map carried yet); a region is now given at
    /// the hello, with the id allocated then, so nothing legitimate arrives
    /// like this any more.
    ///
    /// Mutation this catches: welcoming such a claim — the shard would be told
    /// it has a region that the map (its own included) does not name.
    #[test]
    fn a_claim_with_an_issued_id_on_a_node_outside_the_map_is_refused_too() {
        let mut plane = plane();
        let issued = plane
            .ids
            .allocate()
            .expect("an id the allocator handed out");
        let (counter, map) = (plane.ids.peek(), plane.map().clone());
        let child = space_world::octree::children(plane.config.root_node()).expect("splits")[0];
        let err = plane
            .handle_shard_hello(&ShardHello {
                cluster: plane.config.cluster.clone(),
                shard: issued,
                endpoint: addr(9_001),
                region: Some(ShardRegion::new(child, issued, addr(9_001))),
                secret: String::new(),
            })
            .expect_err("no region is created by claiming it");
        assert!(err.contains("not in the published map"), "{err}");
        assert_eq!(plane.ids.peek(), counter, "the counter did not move");
        assert_eq!(plane.map(), &map, "and the published map is untouched");
    }

    /// A shard that comes back with no id at the address the root's owner
    /// holds is **not** handed that id again here (it used to be: "the same
    /// process restarting"). A process that restarts starts its guid counter
    /// over, so under the same id it would mint the same guids twice; the
    /// orchestrator gives it the owner's regions under a new id instead
    /// (`Orchestrator::handle_shard_hello`). At this level the root is simply
    /// owned.
    #[test]
    fn a_none_re_hello_for_an_owned_root_is_not_handed_the_owners_id() {
        let mut plane = plane();
        plane.set_unclaimed_root_owner(ShardId::new(1));
        let hello = ShardHello {
            cluster: plane.config.cluster.clone(),
            shard: ShardId::NONE,
            endpoint: addr(9_000),
            region: None,
            secret: String::new(),
        };
        plane.handle_shard_hello(&hello).expect("adopts");
        let counter = plane.ids.peek();
        // The very same hello again: refused here.
        let err = plane
            .handle_shard_hello(&hello)
            .expect_err("the id is not handed out twice");
        assert!(err.contains("already owned"), "{err}");
        assert_eq!(plane.ids.peek(), counter, "no allocation, no reservation");
        assert_eq!(plane.map().regions.len(), 1);
        assert_eq!(plane.map().regions[0].endpoint, addr(9_000));
    }

    #[test]
    fn a_none_re_hello_with_a_different_endpoint_on_an_owned_node_is_refused() {
        let mut plane = plane();
        plane.set_unclaimed_root_owner(ShardId::new(1));
        let hello = |port| ShardHello {
            cluster: plane.config.cluster.clone(),
            shard: ShardId::NONE,
            endpoint: addr(port),
            region: None,
            secret: String::new(),
        };
        let (first, other) = (hello(9_000), hello(9_001));
        plane.handle_shard_hello(&first).expect("adopts");
        let counter = plane.ids.peek();
        assert!(plane.handle_shard_hello(&other).is_err());
        assert_eq!(plane.ids.peek(), counter);
        let root = plane.map().regions[0].clone();
        assert_eq!(root.endpoint, addr(9_000), "the owner's endpoint is intact");
    }

    #[test]
    fn a_claim_of_an_implausible_id_is_refused_and_does_not_poison_the_counter() {
        let mut plane = plane();
        plane.set_unclaimed_root_owner(ShardId::new(1));
        plane
            .handle_shard_hello(&ShardHello {
                cluster: plane.config.cluster.clone(),
                shard: ShardId::NONE,
                endpoint: addr(9_000),
                region: None,
                secret: String::new(),
            })
            .expect("adopts the root");
        let counter = plane.ids.peek();
        let child = space_world::octree::children(plane.config.root_node()).expect("splits")[0];
        let poison = ShardHello {
            cluster: plane.config.cluster.clone(),
            shard: ShardId::new(u32::MAX - 1),
            endpoint: addr(9_001),
            region: Some(ShardRegion::new(child, ShardId::NONE, addr(9_001))),
            secret: String::new(),
        };
        assert!(plane.handle_shard_hello(&poison).is_err(), "never issued");
        assert_eq!(plane.ids.peek(), counter, "the counter did not move");
        // The next real allocation (a split's child) is exactly the next id.
        let next = plane.ids.allocate().expect("allocates");
        assert_eq!(u64::from(next.0), counter);
    }

    /// The way in: the cluster secret when one is required, then the cluster
    /// the hello names. On a loopback control port with no configured secret
    /// a hello needs none; with a configured secret, or off loopback, a hello
    /// without exactly it is refused, and the refusal never contains it — nor
    /// anything else about this cluster.
    ///
    /// Mutation this catches: comparing only when the hello carries a secret
    /// (an empty one would pass), accepting a prefix, or checking the cluster
    /// before the secret (anybody who can reach the port would be told what
    /// the cluster is called).
    #[test]
    fn a_hello_needs_the_cluster_secret_when_one_is_required() {
        use crate::config::Secret;
        let hello = |cluster: &str, secret: &str| ShardHello {
            cluster: cluster.to_owned(),
            shard: ShardId::NONE,
            endpoint: addr(9_000),
            region: None,
            secret: secret.to_owned(),
        };
        let plane_for = |config: OrchestratorConfig| {
            let map = ShardMap::new(
                1,
                vec![ShardRegion::new(
                    config.root_node(),
                    ShardId::new(1),
                    addr(7_001),
                )],
            );
            let ids =
                PersistentShardIdAllocator::with_sink(1, Box::new(MemoryCounterSink::default()));
            ControlPlane::new(config, ids, EpochPublisher::new(map), "peer".into())
        };
        const SECRET: &str = "the-cluster-secret-0123";

        // Loopback, nothing configured: no secret is needed, and one that is
        // presented anyway is not looked at (a launched shard presents the
        // generated one).
        let mut open = plane_for(OrchestratorConfig::default());
        assert!(!open.secret_required());
        assert_eq!(open.check_credentials(&hello("space-local", "")), Ok(()));
        open.set_generated_secret(Secret::new("generated-secret-456"));
        assert_eq!(open.cluster_secret().expose(), "generated-secret-456");
        assert_eq!(open.check_credentials(&hello("space-local", "")), Ok(()));
        assert_eq!(
            open.check_credentials(&hello("space-local", "anything")),
            Ok(())
        );
        // Another cluster is refused whatever else it says.
        let refused = open
            .check_credentials(&hello("other", ""))
            .expect_err("another cluster");
        assert!(refused.is_final());
        assert!(refused.to_string().contains("names cluster"), "{refused}");

        // A configured secret, and a control port that is not on loopback.
        let configured = OrchestratorConfig {
            cluster_secret: Some(Secret::new(SECRET)),
            ..OrchestratorConfig::default()
        };
        let off_loopback = OrchestratorConfig {
            bind: "0.0.0.0:7900".parse().expect("an address"),
            cluster_secret: Some(Secret::new(SECRET)),
            ..OrchestratorConfig::default()
        };
        for config in [configured, off_loopback] {
            let mut plane = plane_for(config);
            assert!(plane.secret_required());
            // A configured secret is never replaced by a generated one.
            plane.set_generated_secret(Secret::new("generated-secret-456"));
            assert_eq!(plane.cluster_secret().expose(), SECRET);
            assert_eq!(
                plane.check_credentials(&hello("space-local", SECRET)),
                Ok(())
            );
            for wrong in [
                "",
                "the-cluster-secret-012",
                "the-cluster-secret-01234",
                "x",
            ] {
                let refused = plane
                    .check_credentials(&hello("space-local", wrong))
                    .expect_err(wrong);
                assert_eq!(refused, HelloRefusal::Unauthorized, "{wrong:?}");
                assert!(refused.is_final());
                assert_eq!(refused.reject_reason(), RejectReason::Unauthorized);
                assert!(!refused.to_string().contains(SECRET));
            }
            // The secret is checked first: whoever does not hold it learns
            // nothing — not even that it named the wrong cluster, let alone
            // what this one is called.
            for wrong in ["", "the-cluster-secret-012", "x"] {
                let refused = plane
                    .check_credentials(&hello("other", wrong))
                    .expect_err("no secret");
                assert_eq!(refused, HelloRefusal::Unauthorized, "{wrong:?}");
                assert!(!refused.to_string().contains("space-local"), "{refused}");
            }
            // Whoever holds it is told which cluster this is.
            let refused = plane
                .check_credentials(&hello("other", SECRET))
                .expect_err("another cluster");
            assert!(
                matches!(&refused, HelloRefusal::Refused(why) if why.contains("space-local")),
                "{refused:?}"
            );
        }

        // Off loopback with only a generated secret (the orchestrator starts
        // the shards): that one is what a hello must carry.
        let mut managed = plane_for(OrchestratorConfig {
            bind: "0.0.0.0:7900".parse().expect("an address"),
            min_shards: 2,
            ..OrchestratorConfig::default()
        });
        assert!(managed.secret_required());
        assert_eq!(
            managed.check_credentials(&hello("space-local", "")),
            Err(HelloRefusal::Unauthorized),
            "no secret yet: nobody gets in, not even with an empty one"
        );
        managed.set_generated_secret(Secret::new("generated-secret-456"));
        assert_eq!(
            managed.check_credentials(&hello("space-local", "generated-secret-456")),
            Ok(())
        );
        assert_eq!(
            managed.check_credentials(&hello("space-local", "")),
            Err(HelloRefusal::Unauthorized)
        );
    }

    /// The control plane holds both of the cluster's secrets and derives
    /// `Debug` (and the orchestrator around it does): neither may show in a
    /// print of it.
    ///
    /// Mutation this catches: `peer_token` as a plain `String` again.
    #[test]
    fn the_control_plane_never_prints_a_secret() {
        use crate::config::Secret;
        let config = OrchestratorConfig {
            cluster_secret: Some(Secret::new("do-not-print-the-cluster-secret")),
            ..OrchestratorConfig::default()
        };
        let map = ShardMap::new(
            1,
            vec![ShardRegion::new(
                config.root_node(),
                ShardId::new(1),
                addr(7_001),
            )],
        );
        let ids = PersistentShardIdAllocator::with_sink(1, Box::new(MemoryCounterSink::default()));
        let plane = ControlPlane::new(
            config,
            ids,
            EpochPublisher::new(map),
            "do-not-print-the-peer-token".into(),
        );
        assert_eq!(plane.peer_token.expose(), "do-not-print-the-peer-token");
        let printed = format!("{plane:?}");
        assert!(!printed.contains("do-not-print"), "{printed}");
        assert!(printed.contains("<redacted>"), "{printed}");
    }

    /// How each refusal goes on the wire, and which one a shard may ask again
    /// after.
    #[test]
    fn only_a_wait_is_not_final_and_it_travels_as_draining() {
        let wait = HelloRefusal::Wait("epoch 3 is still in flight".into());
        assert!(!wait.is_final());
        assert_eq!(wait.reject_reason(), RejectReason::Draining);
        assert!(wait.to_string().contains("ask again"), "{wait}");
        let full = HelloRefusal::Full("the cluster already has 8 shard(s)".into());
        assert!(full.is_final());
        assert_eq!(full.reject_reason(), RejectReason::Full);
        let refused = HelloRefusal::Refused("region at depth 40 is already owned".into());
        assert!(refused.is_final());
        assert_eq!(
            refused.reject_reason(),
            RejectReason::Other("region at depth 40 is already owned".into())
        );
        assert_eq!(refused.to_string(), "region at depth 40 is already owned");
    }

    /// An id the allocator has issued does not make a node outside the map
    /// claimable either — it used to ("a restart with a known id"), which
    /// welcomed the shard onto a region no map carried.
    #[test]
    fn a_claim_of_an_already_issued_id_on_a_free_node_is_refused() {
        // Counter at 5: ids 1..=4 were issued.
        let config = OrchestratorConfig::default();
        let root = config.root_node();
        let map = ShardMap::new(
            1,
            vec![ShardRegion::new(root, ShardId::new(1), addr(7_001))],
        );
        let ids = PersistentShardIdAllocator::with_sink(5, Box::new(MemoryCounterSink::default()));
        let mut plane = ControlPlane::new(config, ids, EpochPublisher::new(map), "secret".into());
        let child = space_world::octree::children(root).expect("splits")[0];
        let claim = ShardHello {
            cluster: plane.config.cluster.clone(),
            shard: ShardId::new(3),
            endpoint: addr(9_001),
            region: Some(ShardRegion::new(child, ShardId::NONE, addr(9_001))),
            secret: String::new(),
        };
        let err = plane.handle_shard_hello(&claim).expect_err("refused");
        assert!(err.contains("not in the published map"), "{err}");
        assert_eq!(plane.ids.peek(), 5, "the counter did not move");
        assert_eq!(plane.map().regions.len(), 1);
    }

    /// A sink that cannot write: every persist fails.
    #[derive(Debug)]
    struct BrokenSink;

    impl crate::ids::CounterSink for BrokenSink {
        fn persist(&mut self, _next: u64) -> io::Result<()> {
            Err(io::Error::other("disk full"))
        }
    }

    /// The hello path with a counter that cannot be persisted: a hello that
    /// would have to move the counter is refused with the reason, never a panic,
    /// and one that moves nothing still works.
    #[test]
    fn a_hello_is_refused_not_panicked_when_the_counter_cannot_be_persisted() {
        let config = OrchestratorConfig::default();
        let root = config.root_node();
        // The root is staged for id 100, which the counter has not passed yet,
        // so confirming it has to persist 101.
        let map = ShardMap::new(
            1,
            vec![ShardRegion::new(root, ShardId::new(100), addr(7_001))],
        );
        let ids = PersistentShardIdAllocator::with_sink(100, Box::new(BrokenSink));
        let mut plane = ControlPlane::new(config, ids, EpochPublisher::new(map), "secret".into());
        let claim = ShardHello {
            cluster: plane.config.cluster.clone(),
            shard: ShardId::new(100),
            endpoint: addr(9_000),
            region: None,
            secret: String::new(),
        };
        let err = plane
            .handle_shard_hello(&claim)
            .expect_err("the reservation cannot be persisted");
        assert!(err.contains("could not persist"), "{err}");
        // And an anonymous claim allocates nothing, so it cannot even reach the
        // sink: it is refused for its own reason.
        let child = space_world::octree::children(root).expect("splits")[0];
        let anonymous = ShardHello {
            cluster: plane.config.cluster.clone(),
            shard: ShardId::NONE,
            endpoint: addr(9_001),
            region: Some(ShardRegion::new(child, ShardId::NONE, addr(9_001))),
            secret: String::new(),
        };
        let err = plane.handle_shard_hello(&anonymous).expect_err("refused");
        assert!(err.contains("not in the published map"), "{err}");
    }

    #[test]
    fn a_hello_naming_another_cluster_is_refused() {
        let mut plane = plane();
        let hello = ShardHello {
            cluster: "wrong-cluster".into(),
            shard: ShardId::NONE,
            endpoint: addr(9_000),
            region: None,
            secret: String::new(),
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
            secret: String::new(),
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

    fn probe() -> Message {
        Message::EpochCommit(space_protocol::EpochCommit { epoch: 2 })
    }

    /// The run loop calls `send_to`; a shard that stopped reading must cost that
    /// shard its message, never the loop its clock.
    ///
    /// Mutation this catches: `tx.send(message).await` on the bounded per-shard
    /// queue (it waits for room that never comes), and holding the registry lock
    /// while doing so (nothing else could register or unregister meanwhile).
    #[tokio::test]
    async fn send_to_never_waits_on_a_full_queue_or_holds_the_registry() {
        let server = ControlServer::bind(addr(0)).expect("bind control");
        // A registered shard whose queue holds one message and is never read.
        let (stuck_tx, _stuck_rx) = mpsc::channel::<Message>(1);
        server.register(ShardId::new(1), stuck_tx).await;
        assert!(
            server.send_to(ShardId::new(1), probe()).await,
            "there was room for the first message"
        );

        let bounded = Duration::from_secs(2);
        let queued = tokio::time::timeout(bounded, server.send_to(ShardId::new(1), probe()))
            .await
            .expect("a full queue must not be waited on");
        assert!(!queued, "the message for the stuck shard is dropped");

        // The registry is free, and other shards are unaffected.
        let (tx, mut rx) = mpsc::channel::<Message>(4);
        tokio::time::timeout(bounded, server.register(ShardId::new(2), tx))
            .await
            .expect("the registry lock is not held across a send");
        assert!(server.send_to(ShardId::new(2), probe()).await);
        assert_eq!(rx.recv().await, Some(probe()));

        // An unknown or departed shard is a quiet `false`.
        assert!(!server.send_to(ShardId::new(9), probe()).await);
        drop(rx);
        assert!(!server.send_to(ShardId::new(2), probe()).await);
        server.close();
    }

    /// A peer that completes the transport handshake and then never says who it
    /// is must not hold a connection and a task open.
    ///
    /// Mutation this catches: reading the `ShardHello` with no bound — the
    /// transport's keep-alive keeps such a connection up indefinitely.
    #[tokio::test]
    async fn a_connection_that_never_sends_its_hello_is_closed() {
        let server = Arc::new(
            ControlServer::bind(addr(0))
                .expect("bind control")
                .with_hello_timeout(Duration::from_millis(200)),
        );
        let bound = server.local_addr().expect("bound");
        let accept = Arc::clone(&server).spawn_accept();

        let endpoint = space_net::client_endpoint().expect("client endpoint");
        let (conn, _placeholder) =
            space_net::connect(&endpoint, bound, SERVER_NAME, &space_net::hello("shard:1"))
                .await
                .expect("the transport handshake completes");
        // Say nothing. The orchestrator gives up on its own.
        let closed = tokio::time::timeout(Duration::from_secs(5), conn.connection().closed())
            .await
            .expect("the orchestrator must close a connection that never says hello");
        assert!(
            matches!(closed, quinn::ConnectionError::ApplicationClosed(_)),
            "closed by the orchestrator, not timed out: {closed:?}"
        );
        assert!(conn.is_closed());
        accept.abort();
        server.close();
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
