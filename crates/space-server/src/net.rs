//! The quinn front end: frames in, [`Shard`] calls out.
//!
//! [`Shard`](crate::shard::Shard) is a synchronous state machine (its module docs
//! say why). This module is the only async code in the crate: it accepts
//! connections, runs the hello handshake through
//! [`space_net::accept_incoming`], feeds the resulting messages into the shard,
//! and delivers what the shard queued in its outbox. Nothing here decides
//! anything about the simulation.
//!
//! # One task per connection
//!
//! Each client gets a reader task and a writer task. The reader blocks on
//! whichever receive path delivers next (datagram, reliable stream, control
//! stream) and calls into the shard; the writer drains a small per-connection
//! queue. Splitting them means the shard's tick loop never blocks on a slow
//! client: a client that cannot keep up loses throughput, not the shard's tick
//! budget (U9's ~100 players/shard is exactly the case where that matters).
//!
//! # Peer connections
//!
//! A shard announces itself on the same endpoint as its clients, with a
//! [`Hello`](space_protocol::Hello) whose `player_token` is
//! <code>"shard:&lt;id&gt;"</code> ([`PEER_TOKEN_PREFIX`]). Such a connection is
//! routed to [`Shard::peer_message`] instead of [`Shard::join`], so no ship is
//! spawned for a neighbour shard.
//!
//! **This is identification, not authentication.** The transport is QUIC with a
//! self-signed certificate and no pinning (a peer cannot verify who it dialled);
//! the hello handshake checks the protocol version only; the `shard:<id>` token
//! is a plain string any client can type; and [`Shard::peer_message`] only
//! checks that the claimed id is in the shard's map. The orchestrator-issued
//! `ShardWelcome::peer_token` is **not** presented on peer connections and
//! nothing verifies it. See `docs/security-notes.md`.
//!
//! # Outbox ordering to a peer
//!
//! The handoff protocol is order-sensitive (offer → accept → commit, with abort
//! as the escape hatch): an abort that overtakes its commit, or an accept that
//! lands after the commit was built, corrupts the transfer. So each peer shard
//! gets **one persistent connection and one FIFO sender task**
//! ([`ShardNet::peer_sender`]): every message destined for that peer is queued in
//! order and written by that single task, in queue order, each on one reliable
//! stream over the *same* connection. The connection is opened lazily on the
//! first message, reused while it is alive, and reconnected if it closes. A
//! connect or send failure is **logged and counted**
//! ([`ShardNet::peer_send_failures`]), never silently dropped — a transient
//! failure must be visible, not a silent stall. Because the connection is
//! persistent and never closed right after a send, there is no drain-before-close
//! race (a `CONNECTION_CLOSE` beating its stream data — the race
//! [`space_net::accept_incoming`]'s reject path deliberately avoids).
//!
//! # The tick loop never waits on the network
//!
//! [`ShardNet::flush`] runs inside the tick loop, so it only ever *offers* a
//! message to a queue (`try_send`): a client's, a peer's, the orchestrator's. A
//! queue that is full loses the message — counted and logged for a peer — and
//! the tick goes on. The one thing a dead neighbour must never be able to do is
//! stop this shard's simulation; the handoff protocol already treats a message
//! that does not arrive as a transfer that did not happen (its timeouts), so
//! dropping is the correct degradation.

use std::collections::HashMap;
use std::io;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use space_net::{
    accept_incoming, client_endpoint_for, connect, connect_within, hello, server_endpoint, Conn,
};
use space_protocol::{Message, ShardHello, ShardId, ShardWelcome};
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;

use crate::shard::{Destination, Shard};

/// How long a peer sender waits for one dial (QUIC/TLS plus the hello exchange)
/// to a neighbour shard before it gives up on that attempt.
///
/// A neighbour on the same cluster answers in milliseconds, so this only ever
/// runs out for a peer that is down — and while a dial is pending nothing queued
/// for that peer moves, so the bound is what keeps a dead neighbour's queue from
/// backing up for the transport's full ten seconds per attempt.
pub const PEER_DIAL_TIMEOUT: Duration = Duration::from_secs(2);

/// How many frames may sit unsent on one connection before the reader notices.
///
/// Comfortably more than one tick's worth (a delta per tick plus a snapshot chunk)
/// so a momentary stall is absorbed, and bounded so a client that has stopped
/// reading cannot make the shard grow without limit.
const WRITE_QUEUE: usize = 64;

/// How many handoff messages may sit queued for one peer before the sender is
/// considered stalled. A handoff is a handful of messages, so this is generous;
/// it only bounds growth against a peer that has stopped reading.
const PEER_QUEUE: usize = 256;

/// The `player_token` prefix that marks a connection as coming from another shard.
pub const PEER_TOKEN_PREFIX: &str = "shard:";

/// The TLS server name both sides present ([`space_net::crypto`]'s self-signed
/// certificate is issued for it).
pub const SERVER_NAME: &str = "space.local";

/// Parse a peer hello token into the shard id it names.
#[must_use]
pub fn peer_shard_id(token: &str) -> Option<ShardId> {
    token
        .strip_prefix(PEER_TOKEN_PREFIX)
        .and_then(|rest| rest.parse::<u32>().ok())
        .map(ShardId)
}

/// The queue end of a peer's FIFO sender, plus a flag the sender sets when it has
/// given up (so the next flush starts a fresh one).
#[derive(Debug, Clone)]
struct PeerSender {
    tx: mpsc::Sender<Message>,
    dead: Arc<AtomicBool>,
    /// The address this sender dials; a different current address means the
    /// sender is stale and must be replaced (see [`ShardNet::peer_sender`]).
    addr: SocketAddr,
}

impl PeerSender {
    /// Whether this sender is still draining its queue.
    fn is_live(&self) -> bool {
        !self.dead.load(Ordering::SeqCst)
    }
}

/// The front end for one shard.
#[derive(Debug)]
pub struct ShardNet {
    endpoint: quinn::Endpoint,
    /// How to reach each peer shard, from the shard map's endpoints.
    shard_endpoints: Mutex<HashMap<ShardId, SocketAddr>>,
    /// Per-session write channels.
    writers: Mutex<HashMap<u64, mpsc::Sender<Message>>>,
    /// One FIFO sender per peer shard (see the module docs).
    peers: Mutex<HashMap<ShardId, PeerSender>>,
    /// The orchestrator's FIFO sender, if this shard registered with one
    /// (Phase 3: `--orchestrator`). Epoch messages and load reports go out here.
    orchestrator: Mutex<Option<mpsc::Sender<Message>>>,
    /// Count of peer sends that failed to connect or send (logged, not silently
    /// dropped). Exposed for tests and a future metrics surface.
    peer_send_failures: Arc<AtomicU64>,
}

impl ShardNet {
    /// Bind a shard's endpoint on `bind`.
    ///
    /// # Errors
    ///
    /// [`io::Error`] when the socket cannot be bound.
    pub fn bind(bind: SocketAddr) -> io::Result<Self> {
        let (config, _cert) = space_net::crypto::make_server_config();
        let endpoint = server_endpoint(bind, config)?;
        Ok(Self {
            endpoint,
            shard_endpoints: Mutex::new(HashMap::new()),
            writers: Mutex::new(HashMap::new()),
            peers: Mutex::new(HashMap::new()),
            orchestrator: Mutex::new(None),
            peer_send_failures: Arc::new(AtomicU64::new(0)),
        })
    }

    /// The bound address.
    ///
    /// # Errors
    ///
    /// [`io::Error`] if the socket was closed.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.endpoint.local_addr()
    }

    /// How many peer sends have failed to connect or send since the shard started.
    #[must_use]
    pub fn peer_send_failures(&self) -> u64 {
        self.peer_send_failures.load(Ordering::SeqCst)
    }

    /// Remember how to reach `shard`.
    ///
    /// An unresolved address (port 0,
    /// [`space_protocol::ShardRegion::UNRESOLVED_ENDPOINT`]) is
    /// not remembered: it is un-dialable and must not shadow the real address a
    /// later map push supplies.
    pub async fn set_shard_endpoint(&self, shard: ShardId, addr: SocketAddr) {
        if addr.port() == 0 {
            return;
        }
        self.shard_endpoints.lock().await.insert(shard, addr);
    }

    /// Accept connections until the endpoint is closed.
    pub async fn serve(self: Arc<Self>, shard: Arc<Mutex<Shard>>) {
        loop {
            let Some(incoming) = self.endpoint.accept().await else {
                return; // endpoint closed
            };
            let net = Arc::clone(&self);
            let shard = Arc::clone(&shard);
            tokio::spawn(async move {
                let Ok((conn, hello)) = accept_incoming(incoming).await else {
                    return; // refused: the reason was already sent
                };
                if let Some(peer) = peer_shard_id(&hello.player_token) {
                    net.serve_peer(conn, peer, shard).await;
                } else {
                    net.serve_client(conn, hello, shard).await;
                }
            });
        }
    }

    /// Drive one client session: welcome it, then pump frames both ways.
    async fn serve_client(
        self: Arc<Self>,
        conn: Conn,
        hello: space_protocol::Hello,
        shard: Arc<Mutex<Shard>>,
    ) {
        let (tx, mut rx) = mpsc::channel::<Message>(WRITE_QUEUE);
        let joined = {
            let mut s = shard.lock().await;
            let joined = s.join(&hello);
            if let Ok(welcome) = &joined {
                // Registered **before the shard lock is released**. The session
                // exists from the moment `join` returns, so the very next tick
                // builds its first frame — the snapshot that tells the client
                // which ship is its own. `flush` drops a frame for a session it
                // has no queue for, and that snapshot is never sent twice: when
                // the queue was only registered after the welcome had been
                // written, a tick landing in between left the client connected,
                // shipless and silent for good. Holding the shard lock across
                // this insert is safe: nothing waits for the shard while holding
                // `writers`.
                self.writers.lock().await.insert(welcome.session, tx);
            }
            joined
        };
        let welcome = match joined {
            Ok(welcome) => welcome,
            Err(reason) => {
                // The refusal has to *arrive*: writing it and returning would
                // drop the connection and the close would beat the reason there.
                conn.reject(reason).await;
                return;
            }
        };
        if conn
            .send_control(&Message::Welcome(welcome.clone()))
            .await
            .is_err()
        {
            self.writers.lock().await.remove(&welcome.session);
            shard.lock().await.leave(welcome.session);
            return;
        }
        let dead = Arc::new(AtomicBool::new(false));

        let writer = {
            let conn = conn.clone();
            let dead = Arc::clone(&dead);
            tokio::spawn(async move {
                while let Some(message) = rx.recv().await {
                    if conn.send(&message).await.is_err() {
                        dead.store(true, Ordering::SeqCst);
                        return;
                    }
                }
            })
        };

        let reader = conn.clone();
        loop {
            if dead.load(Ordering::SeqCst) || conn.is_closed() {
                break;
            }
            // Whichever path delivers first wins; all three return
            // `Ok(None)` when the peer is gone.
            let next = tokio::select! {
                m = reader.recv_datagram() => m,
                m = reader.recv_stream() => m,
                m = reader.recv_control() => m,
            };
            match next {
                Ok(Some(message)) => {
                    let mut s = shard.lock().await;
                    match message {
                        Message::InputTick(batch) => s.apply_input(welcome.session, &batch),
                        Message::NeighbourStatus(status) => {
                            s.client_status(welcome.session, status)
                        }
                        // A client's whole vocabulary is inputs and status
                        // reports; anything else is ignored, not fatal.
                        _ => {}
                    }
                }
                // `Ok(None)` and any error both mean the peer is gone.
                _ => break,
            }
        }
        dead.store(true, Ordering::SeqCst);
        writer.abort();
        self.writers.lock().await.remove(&welcome.session);
        shard.lock().await.leave(welcome.session);
    }

    /// Drive one peer-shard connection: no session, no ship, no replication.
    async fn serve_peer(self: Arc<Self>, conn: Conn, peer: ShardId, shard: Arc<Mutex<Shard>>) {
        // The peer opened us through `space_net::connect`, which blocks until it
        // sees a Welcome; answer it with our own map so both sides agree.
        let map = {
            let s = shard.lock().await;
            space_protocol::Welcome {
                session: 0,
                shard: s.shard(),
                shard_map: s.shard_map().clone(),
            }
        };
        if conn.send_control(&Message::Welcome(map)).await.is_err() {
            return;
        }
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
                Ok(Some(message)) => shard.lock().await.peer_message(peer, message),
                Ok(None) => {
                    eprintln!("space-server: the connection from peer shard {peer} closed");
                    break;
                }
                Err(e) => {
                    eprintln!("space-server: the connection from peer shard {peer} failed: {e}");
                    break;
                }
            }
        }
    }

    /// Deliver everything the shard queued.
    ///
    /// Client frames go into that client's write queue (dropped if it is full: a
    /// client that cannot keep up loses frames; what a lost frame carried is
    /// sent again once the client's acknowledgements show it never arrived — see
    /// [`crate::replicate`]). Shard-to-shard frames go into that peer's FIFO
    /// sender, which preserves order on one persistent connection (see the
    /// module docs).
    ///
    /// **Nothing in here waits on a queue.** This runs inside the tick loop, so
    /// a full queue — a client that stopped reading, a neighbour that is down,
    /// an orchestrator that is not answering — costs that destination its
    /// message, never the shard its tick.
    pub async fn flush(&self, shard: &Arc<Mutex<Shard>>) {
        let (outbox, local) = {
            let mut s = shard.lock().await;
            (s.take_outbox(), s.shard())
        };
        if outbox.is_empty() {
            return;
        }
        // Client frames are moved into their queues (not cloned: a frame is a
        // few allocations per entity), in outbox order; everything else is kept
        // back, still in outbox order, for the peer senders.
        let mut elsewhere = Vec::new();
        {
            let writers = self.writers.lock().await;
            for out in outbox {
                match out.dest {
                    Destination::Client(session) => {
                        if let Some(tx) = writers.get(&session) {
                            // `try_send` rather than `send`: never let one slow
                            // client hold the tick loop.
                            let _ = tx.try_send(out.message);
                        }
                    }
                    Destination::Shard(_) | Destination::Orchestrator => elsewhere.push(out),
                }
            }
        }
        // Peer messages go out in outbox order through each peer's FIFO sender.
        for out in elsewhere {
            match out.dest {
                Destination::Shard(shard_id) => {
                    // The peer is dialled at the address the shard's own map
                    // names for it, so a shard that appears or migrates through
                    // an epoch change is reachable without any extra wiring; an
                    // endpoint registered by hand is only the fallback.
                    let from_map = shard.lock().await.peer_endpoint(shard_id);
                    self.send_to_peer(local, shard_id, from_map, out.message)
                        .await;
                }
                Destination::Orchestrator => {
                    if let Some(tx) = self.orchestrator.lock().await.as_ref() {
                        if tx.try_send(out.message).is_err() {
                            self.peer_send_failures.fetch_add(1, Ordering::SeqCst);
                            eprintln!(
                                "space-server: the orchestrator link is closed or backed up; \
                                 message dropped"
                            );
                        }
                    }
                }
                Destination::Client(_) => {}
            }
        }
    }

    /// Register the orchestrator's FIFO sender (Phase 3: `--orchestrator`).
    pub async fn set_orchestrator_sender(&self, tx: mpsc::Sender<Message>) {
        *self.orchestrator.lock().await = Some(tx);
    }

    /// Queue `message` for `shard_id`'s FIFO sender, starting one if needed.
    ///
    /// Ordering is preserved because every message for a peer goes through the
    /// same channel and the same sender task. An unknown endpoint, a sender that
    /// has gone away, or a queue that is full (the peer is down and its sender is
    /// still dialling) is logged and counted — never silently dropped, and never
    /// waited for: this is called from the tick loop, and a neighbour that is
    /// down must not be able to stop this shard's simulation. Every handoff
    /// message is covered by a timeout on the side that is waiting for it, so a
    /// dropped one degrades to "the transfer did not happen".
    async fn send_to_peer(
        &self,
        local: ShardId,
        shard_id: ShardId,
        from_map: Option<SocketAddr>,
        message: Message,
    ) {
        // The shard map is current; a hand-registered endpoint is only a fallback
        // for a peer the map does not (yet) name, so it can never shadow a
        // migrated shard's new address.
        let registered = self.shard_endpoints.lock().await.get(&shard_id).copied();
        let Some(addr) = from_map.or(registered) else {
            self.peer_send_failures.fetch_add(1, Ordering::SeqCst);
            eprintln!("space-server: no endpoint known for peer shard {shard_id}; message dropped");
            return;
        };
        let sender = self.peer_sender(local, shard_id, addr).await;
        match sender.tx.try_send(message) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.peer_send_failures.fetch_add(1, Ordering::SeqCst);
                eprintln!(
                    "space-server: the queue for peer shard {shard_id} is full (is it down?); \
                     message dropped"
                );
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                self.peer_send_failures.fetch_add(1, Ordering::SeqCst);
                eprintln!("space-server: peer sender for shard {shard_id} gone; message dropped");
            }
        }
    }

    /// The FIFO sender for `shard_id`, starting one if none is live.
    ///
    /// A cached sender is reused only while it is live **and** was dialled at
    /// `addr`. When a map refresh changes the peer's address (a same-epoch
    /// `DirectoryReply`), the next message re-dials the new address instead of
    /// continuing to feed the old one. The superseded sender drains whatever it
    /// already holds and then stops, because its queue's last handle is dropped.
    async fn peer_sender(&self, local: ShardId, shard_id: ShardId, addr: SocketAddr) -> PeerSender {
        let mut peers = self.peers.lock().await;
        if let Some(sender) = peers
            .get(&shard_id)
            .filter(|s| s.is_live() && s.addr == addr)
        {
            return sender.clone();
        }
        let (tx, rx) = mpsc::channel::<Message>(PEER_QUEUE);
        let dead = Arc::new(AtomicBool::new(false));
        let failures = Arc::clone(&self.peer_send_failures);
        let task_dead = Arc::clone(&dead);
        tokio::spawn(async move {
            run_peer_sender(rx, addr, shard_id, local, task_dead, failures).await;
        });
        let sender = PeerSender {
            tx,
            dead: Arc::clone(&dead),
            addr,
        };
        peers.insert(shard_id, sender.clone());
        sender
    }

    /// Stop accepting.
    pub fn close(&self) {
        self.endpoint.close(0u32.into(), b"shard shutting down");
    }
}

/// Wait for a shard's two long-running tasks and say why the shard stopped.
///
/// * `serve` ending is the ordinary shutdown — the endpoint was closed. The tick
///   loop is stopped and the result is `Ok`.
/// * `ticking` ending is **never** ordinary: the loop has no exit, so it means
///   the task panicked (or was cancelled from outside). The accept loop is
///   stopped and the result is an error.
///
/// The second case is the reason this exists. A caller that only waits on
/// `serve` does not notice the tick loop dying: the endpoint keeps accepting,
/// every newcomer is welcomed, and the world they joined never advances again.
/// A shard that cannot tick must stop, loudly.
///
/// # Errors
///
/// The tick loop ended, or the accept loop panicked.
pub async fn supervise(
    mut serve: JoinHandle<()>,
    mut ticking: JoinHandle<()>,
) -> Result<(), String> {
    tokio::select! {
        served = &mut serve => {
            ticking.abort();
            served.map_err(|e| format!("the accept loop failed: {e}"))
        }
        ticked = &mut ticking => {
            serve.abort();
            Err(match ticked {
                Ok(()) => "the tick loop ended".to_owned(),
                Err(e) => format!("the tick loop died ({e}); the shard cannot simulate any more"),
            })
        }
    }
}

/// Drain `rx` and write each message to `addr` on one persistent connection.
///
/// The connection is opened on the first message and reused; if it closes, the
/// next message reconnects. Every message is written with
/// [`Conn::send_reliable`] on that persistent connection — never a fresh
/// connection per message, and never closed right after a send. A connect or send
/// failure bumps `failures` and logs, and the loop keeps trying on the next
/// message rather than tearing the sender down (a transient failure must not
/// permanently stall the transfer).
///
/// A dial is bounded by [`PEER_DIAL_TIMEOUT`], and a dial that fails takes
/// **everything queued behind it** with it, each counted as a failure. Those
/// messages belong to handoffs that have been waiting for the length of the
/// dial already; delivering them one failed dial apart (which is what "try again
/// on the next message" amounted to while a peer was down) only kept the queue
/// full, and a full queue used to stall the tick loop.
async fn run_peer_sender(
    mut rx: mpsc::Receiver<Message>,
    addr: SocketAddr,
    peer: ShardId,
    local: ShardId,
    dead: Arc<AtomicBool>,
    failures: Arc<AtomicU64>,
) {
    // One client endpoint for the life of the sender: a quinn `Endpoint` drives
    // its connections, so it must outlive them. It is bound in the address
    // family of the peer it dials.
    let endpoint = match client_endpoint_for(addr) {
        Ok(endpoint) => endpoint,
        Err(e) => {
            failures.fetch_add(1, Ordering::SeqCst);
            eprintln!("space-server: cannot build client endpoint for peer {peer}: {e}");
            dead.store(true, Ordering::SeqCst);
            return;
        }
    };
    let mut conn: Option<Conn> = None;
    while let Some(message) = rx.recv().await {
        if conn.as_ref().is_none_or(|c| c.is_closed()) {
            let hello = peer_hello(local);
            let dial = connect_within(&endpoint, addr, SERVER_NAME, &hello, PEER_DIAL_TIMEOUT);
            conn = match dial.await {
                Ok((conn, _welcome)) => Some(conn),
                Err(e) => {
                    // This message is lost, and so is everything that queued up
                    // behind it while the dial was pending (see the docs above).
                    let mut dropped = 1u64;
                    while rx.try_recv().is_ok() {
                        dropped += 1;
                    }
                    failures.fetch_add(dropped, Ordering::SeqCst);
                    eprintln!(
                        "space-server: connect to peer shard {peer} at {addr} failed: {e}; \
                         {dropped} queued message(s) dropped"
                    );
                    continue; // try again on the next message
                }
            };
        }
        if let Some(c) = conn.as_ref() {
            if let Err(e) = c.send_reliable(&message).await {
                failures.fetch_add(1, Ordering::SeqCst);
                eprintln!("space-server: send to peer shard {peer} failed: {e}");
                conn = None; // reconnect on the next message
            }
        }
    }
    // Queue dropped (the `ShardNet` went away): nothing more to send.
    dead.store(true, Ordering::SeqCst);
    drop(endpoint);
}

/// The [`Hello`](space_protocol::Hello) a shard uses to reach a peer.
///
/// The token carries the bare numeric id (`shard:7`), the form
/// [`peer_shard_id`] parses. `ShardId`'s `Display` is the human-readable
/// `shard 7`, which must not be used here: the accepting side would fail to
/// parse it and treat the peer as a client.
#[must_use]
pub fn peer_hello(shard: ShardId) -> space_protocol::Hello {
    hello(format!("{PEER_TOKEN_PREFIX}{}", shard.0))
}

/// The shard's persistent control connection to the orchestrator (Phase 3).
///
/// One connection, one FIFO sender (epoch acks, scene acks and load reports go
/// out in order) and one reader task (epoch prepare/commit/abort, map pushes
/// and the cluster's scene come in and are fed to
/// [`Shard::orchestrator_message`]). It reuses `run_peer_sender`'s shape because
/// order matters for epoch messages for the same reason it matters for handoff
/// messages.
pub struct OrchestratorLink {
    /// The shard's outbound sender: queue a message to send to the orchestrator.
    pub tx: mpsc::Sender<Message>,
    /// The reader task handle (kept so a caller can abort it on shutdown).
    reader: JoinHandle<()>,
    /// The sender task handle.
    sender: JoinHandle<()>,
}

impl std::fmt::Debug for OrchestratorLink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrchestratorLink").finish_non_exhaustive()
    }
}

impl OrchestratorLink {
    /// Run the transport handshake + `ShardHello`/`ShardWelcome` exchange with
    /// the orchestrator at `addr`, returning the welcome and the live connection.
    ///
    /// The caller builds its [`Shard`] from the welcome (id + region + map) and
    /// then calls [`OrchestratorLink::attach`] to start the reader + sender.
    ///
    /// # Errors
    ///
    /// A [`space_net::NetError`] if the connection or handshake fails.
    pub async fn handshake(
        addr: SocketAddr,
        hello: ShardHello,
    ) -> Result<(ShardWelcome, Conn), space_net::NetError> {
        let endpoint =
            client_endpoint_for(addr).map_err(|e| space_net::NetError::Handshake(e.to_string()))?;
        let (conn, _welcome) =
            connect(&endpoint, addr, SERVER_NAME, &peer_hello(hello.shard)).await?;
        // The orchestrator answered the transport handshake with a placeholder;
        // now send the real ShardHello and read the ShardWelcome back.
        conn.send_control(&Message::ShardHello(hello))
            .await
            .map_err(|e| space_net::NetError::Io(e.to_string()))?;
        // The welcome comes back on the reliable stream (the orchestrator's
        // per-shard FIFO sender writes every message, welcome included, with
        // `send_reliable`), so read whichever path delivers it. Bounded like the
        // transport handshake: the connection is kept alive by pings, so an
        // orchestrator that never answers would otherwise be waited on forever.
        let answer = async {
            tokio::select! {
                m = conn.recv_control() => m,
                m = conn.recv_stream() => m,
            }
        };
        let next = tokio::time::timeout(space_net::HANDSHAKE_TIMEOUT, answer)
            .await
            .map_err(|_| {
                space_net::NetError::Handshake(format!(
                    "no ShardWelcome from the orchestrator at {addr} within {:?}",
                    space_net::HANDSHAKE_TIMEOUT
                ))
            })?;
        let welcome = match next.map_err(|e| space_net::NetError::Io(e.to_string()))? {
            Some(Message::ShardWelcome(w)) => w,
            Some(Message::Reject(reject)) => {
                let why = match reject.reason {
                    space_protocol::RejectReason::Other(text) => text,
                    other => format!("{other:?}"),
                };
                return Err(space_net::NetError::Protocol(format!(
                    "the orchestrator rejected this shard's hello: {why}"
                )));
            }
            Some(other) => {
                return Err(space_net::NetError::Protocol(format!(
                    "expected ShardWelcome, got {other:?}"
                )));
            }
            None => {
                return Err(space_net::NetError::Protocol(
                    "the orchestrator closed before welcoming".into(),
                ));
            }
        };
        drop(endpoint);
        Ok((welcome, conn))
    }

    /// Start the reader + sender tasks for `conn`, feeding inbound epoch messages
    /// into `shard`. Returns the link and its outbound sender.
    #[must_use]
    pub fn attach(conn: Conn, shard: Arc<Mutex<Shard>>) -> Self {
        // Outbound: epoch acks + load reports, FIFO on the persistent connection.
        let (tx, mut rx) = mpsc::channel::<Message>(PEER_QUEUE);
        let sender_conn = conn.clone();
        let sender = tokio::spawn(async move {
            while let Some(message) = rx.recv().await {
                if let Err(e) = sender_conn.send_reliable(&message).await {
                    eprintln!(
                        "space-server: the orchestrator link stopped sending ({e}); epoch acks \
                         from this shard no longer reach the orchestrator"
                    );
                    return;
                }
            }
        });
        // Inbound: epoch prepare/commit/abort, fed to the shard.
        //
        // The reader ending is the end of this shard's membership of the
        // cluster: no further epoch or map push will reach it, and nothing
        // re-dials. That must not happen silently, so whatever ends the loop is
        // logged.
        let reader = {
            let conn = conn.clone();
            tokio::spawn(async move {
                let why = loop {
                    if let Some(reason) = conn.close_reason() {
                        break reason.to_string();
                    }
                    let next = tokio::select! {
                        m = conn.recv_control() => m,
                        m = conn.recv_stream() => m,
                        m = conn.recv_datagram() => m,
                    };
                    match next {
                        Ok(Some(message)) => match message {
                            Message::EpochPrepare(_)
                            | Message::EpochCommit(_)
                            | Message::EpochAbort(_)
                            | Message::DirectoryReply(_) => {
                                shard.lock().await.orchestrator_message(message);
                            }
                            Message::SceneSet(_) => {
                                // The cluster's scene. The shard adopts it if it
                                // is newer, relays it to its clients and queues
                                // the ack; the switch is worth a line in the log.
                                let (id, before, after) = {
                                    let mut s = shard.lock().await;
                                    let before = s.scene();
                                    s.orchestrator_message(message);
                                    (s.shard(), before, s.scene())
                                };
                                if after != before {
                                    println!("{}", scene_switch_line(id, before, after));
                                }
                            }
                            _ => {}
                        },
                        Ok(None) => {
                            break conn
                                .close_reason()
                                .map_or_else(|| "the stream ended".to_owned(), |r| r.to_string());
                        }
                        Err(e) => break e.to_string(),
                    }
                };
                eprintln!(
                    "space-server: the orchestrator link closed ({why}); this shard no longer \
                     receives epoch changes or map pushes"
                );
            })
        };
        Self { tx, reader, sender }
    }

    /// Stop the link's tasks.
    pub fn close(self) {
        self.reader.abort();
        self.sender.abort();
    }
}

/// The log line for a shard whose scene the orchestrator just changed.
fn scene_switch_line(
    shard: ShardId,
    before: space_protocol::SceneState,
    after: space_protocol::SceneState,
) -> String {
    if before.is_set() {
        format!("space-server: {shard} switched scene: {before} -> {after}")
    } else {
        format!("space-server: {shard} runs scene {after}, as told by the orchestrator")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The token a shard dials its peers with must parse back to its own id, or
    /// the accepting shard mistakes the peer for a client (and every handoff
    /// message is lost).
    #[test]
    fn a_peer_hello_token_round_trips_to_the_shard_id() {
        for id in [1u32, 7, 100, u32::MAX - 1] {
            let hello = peer_hello(ShardId::new(id));
            assert_eq!(peer_shard_id(&hello.player_token), Some(ShardId::new(id)));
        }
    }

    fn probe() -> Message {
        Message::Reject(space_protocol::Reject {
            reason: space_protocol::RejectReason::Draining,
        })
    }

    fn loopback() -> SocketAddr {
        SocketAddr::from(([127, 0, 0, 1], 0))
    }

    /// A served shard on loopback with **no** tick loop: the tests below decide
    /// exactly when a tick and a flush happen.
    async fn served_shard() -> (Arc<ShardNet>, Arc<Mutex<Shard>>, JoinHandle<()>) {
        let net = Arc::new(ShardNet::bind(loopback()).expect("bind"));
        let shard = Arc::new(Mutex::new(Shard::new(crate::ShardConfig::new(
            ShardId::new(1),
            net.local_addr().expect("addr"),
            space_world::NodeRef::new(64, [0, 0, 0]),
        ))));
        let serve = tokio::spawn({
            let (net, shard) = (Arc::clone(&net), Arc::clone(&shard));
            async move { net.serve(shard).await }
        });
        (net, shard, serve)
    }

    /// A session is never visible to the tick loop without its write queue.
    ///
    /// The test holds the `writers` lock, so a serving task that joins the shard
    /// has to stop wherever it registers its queue. The old order was "join,
    /// release the shard, send the welcome, *then* register": the join was
    /// visible (one client, shard lock free) while the queue was still missing,
    /// and a tick + flush in that window built the client's first snapshot and
    /// threw it away — for good, since a ship at rest is never sent twice. The
    /// fixed order registers under the shard lock, so while registration is
    /// blocked the shard lock is held and no tick can observe the session at all.
    ///
    /// Mutation this catches: moving `writers.insert` back after the shard lock
    /// is released — `try_lock` then succeeds with one client and no queue.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_session_is_never_visible_without_its_write_queue() {
        let (net, shard, serve) = served_shard().await;
        let bound = net.local_addr().expect("addr");

        let writers = net.writers.lock().await; // block registration
        let client = tokio::spawn(async move {
            let endpoint = client_endpoint_for(bound).expect("client endpoint");
            connect(&endpoint, bound, SERVER_NAME, &hello("player")).await
        });

        // Wait until the serving task is parked on `writers` with the shard lock
        // held: the shard stays locked across many consecutive looks. A join that
        // merely passes through the lock releases it within microseconds, so it
        // can never produce a run this long — and then the session shows.
        let mut held_for = 0;
        for _ in 0..2_000 {
            match shard.try_lock() {
                Ok(s) => {
                    held_for = 0;
                    assert_eq!(
                        s.stats().clients,
                        0,
                        "the session is visible to a tick, but it has no write queue \
                         (registered: {}): its first snapshot would be dropped",
                        writers.len()
                    );
                }
                Err(_) => held_for += 1,
            }
            if held_for >= 40 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(
            held_for >= 40,
            "the serving task never reached registration under the shard lock"
        );
        drop(writers);

        // From here on it is an ordinary join: one tick, one flush, and the
        // first snapshot — the frame that names the client's own ship — arrives.
        let (conn, welcome) = client.await.expect("client task").expect("connect");
        assert!(net.writers.lock().await.contains_key(&welcome.session));
        shard.lock().await.tick();
        net.flush(&shard).await;
        // The scene announcement the join queued comes first (it is not a
        // frame of the world); the snapshot is right behind it.
        let announced = tokio::time::timeout(Duration::from_secs(5), conn.recv_stream())
            .await
            .expect("the scene must arrive")
            .expect("recv")
            .expect("a message");
        assert_eq!(
            announced,
            Message::SceneUpdate(space_protocol::SceneState::UNSET),
            "the first thing behind the welcome is the scene"
        );
        let frame = tokio::time::timeout(Duration::from_secs(5), conn.recv_stream())
            .await
            .expect("the first snapshot must arrive")
            .expect("recv")
            .expect("a frame");
        let Message::Snapshot(snapshot) = frame else {
            panic!("the first frame for a new client is a snapshot, got {frame:?}");
        };
        let ship = shard
            .lock()
            .await
            .ship_of(welcome.session)
            .expect("a ship was spawned");
        assert!(
            snapshot.entities.iter().any(|e| e.guid == ship),
            "the snapshot must carry the client's own ship"
        );
        serve.abort();
        net.close();
    }

    /// Over the real transport: a client is told the scene behind its welcome,
    /// a change made while it is connected reaches it, and a stale revision
    /// reaches nobody.
    ///
    /// Mutation this catches: a scene update that never leaves the outbox (a
    /// destination the flush does not deliver), or one sent on a path the
    /// client does not read.
    #[tokio::test]
    async fn a_connected_client_is_told_the_scene_and_follows_a_change() {
        use space_protocol::{Scene, SceneState};
        let (net, shard, serve) = served_shard().await;
        let bound = net.local_addr().expect("addr");
        let solar = SceneState::new(Scene::Solar.id(), 1);
        let galaxy = SceneState::new(Scene::Galaxy.id(), 2);
        assert!(shard.lock().await.set_scene(solar));

        let endpoint = client_endpoint_for(bound).expect("client endpoint");
        let (conn, _welcome) = connect(&endpoint, bound, SERVER_NAME, &hello("player"))
            .await
            .expect("connect");
        // The next scene update on the reliable path (frames of the world are
        // skipped: each message rides its own stream).
        let next_scene = |conn: Conn| async move {
            loop {
                let message = tokio::time::timeout(Duration::from_secs(5), conn.recv_stream())
                    .await
                    .expect("a scene update must arrive")
                    .expect("recv")
                    .expect("the connection is open");
                if let Message::SceneUpdate(state) = message {
                    return state;
                }
            }
        };

        shard.lock().await.tick();
        net.flush(&shard).await;
        assert_eq!(next_scene(conn.clone()).await, solar, "told on join");

        // A late message for an older revision changes nothing and is not
        // relayed; the change behind it is.
        assert!(!shard
            .lock()
            .await
            .set_scene(SceneState::new(Scene::Materials.id(), 1)));
        assert!(shard.lock().await.set_scene(galaxy));
        shard.lock().await.tick();
        net.flush(&shard).await;
        assert_eq!(
            next_scene(conn.clone()).await,
            galaxy,
            "the change, and only the change, reaches the client"
        );
        serve.abort();
        net.close();
    }

    /// Over the real transport: the announcement queued by the join never
    /// reaches the client (it is taken out of the outbox before any flush —
    /// what a full write queue does to it), the scene does not change again,
    /// and the client is told it all the same within the re-announce interval.
    ///
    /// Mutation this catches: no periodic re-announcement — the client then
    /// waits for a scene that is never sent, and the read below times out.
    #[tokio::test]
    async fn a_client_whose_scene_announcement_was_dropped_still_learns_the_scene() {
        use space_protocol::{Scene, SceneState};
        let (net, shard, serve) = served_shard().await;
        let bound = net.local_addr().expect("addr");
        let solar = SceneState::new(Scene::Solar.id(), 3);
        assert!(shard.lock().await.set_scene(solar));

        let endpoint = client_endpoint_for(bound).expect("client endpoint");
        let (conn, _welcome) = connect(&endpoint, bound, SERVER_NAME, &hello("player"))
            .await
            .expect("connect");
        // The join's announcement is dropped before it is ever offered.
        let dropped = shard.lock().await.take_outbox();
        assert!(
            dropped
                .iter()
                .any(|out| out.message == Message::SceneUpdate(solar)),
            "precondition: the join queued the announcement: {dropped:?}"
        );

        // The client reads what arrives on the reliable path while the shard
        // ticks through one re-announce interval; nothing changes the scene.
        let reader = tokio::spawn({
            let conn = conn.clone();
            async move {
                loop {
                    match conn.recv_stream().await {
                        Ok(Some(Message::SceneUpdate(state))) => return Some(state),
                        Ok(Some(_)) => {}
                        _ => return None,
                    }
                }
            }
        });
        for _ in 0..crate::config::SCENE_REANNOUNCE_TICKS {
            shard.lock().await.tick();
            net.flush(&shard).await;
        }
        let learned = tokio::time::timeout(Duration::from_secs(10), reader)
            .await
            .expect("the scene is announced again within the interval")
            .expect("the reader task");
        assert_eq!(learned, Some(solar));
        assert_eq!(
            shard.lock().await.scene(),
            solar,
            "without a further change"
        );
        serve.abort();
        net.close();
    }

    #[test]
    fn a_scene_switch_reads_as_what_happened() {
        use space_protocol::{Scene, SceneState};
        let solar = SceneState::new(Scene::Solar.id(), 2);
        assert_eq!(
            scene_switch_line(ShardId::new(3), SceneState::UNSET, solar),
            "space-server: shard 3 runs scene solar (revision 2), as told by the orchestrator"
        );
        assert_eq!(
            scene_switch_line(
                ShardId::new(3),
                SceneState::new(Scene::DemoArea.id(), 1),
                solar
            ),
            "space-server: shard 3 switched scene: demo-area (revision 1) -> solar (revision 2)"
        );
    }

    /// A shard at the player cap tells the client so.
    ///
    /// Mutation this catches: answering a refused `join` with
    /// `send_control(Reject)` and returning — the connection is dropped, its
    /// close overtakes the reject, and the client sees a transport error.
    #[tokio::test]
    async fn a_full_shard_tells_the_client_it_is_full() {
        let (net, shard, serve) = served_shard().await;
        {
            let mut s = shard.lock().await;
            for i in 0..crate::shard::MAX_PLAYERS_PER_SHARD {
                s.join(&hello(format!("filler-{i}"))).expect("filler joins");
            }
        }
        let bound = net.local_addr().expect("addr");
        let endpoint = client_endpoint_for(bound).expect("client endpoint");
        let err = connect(&endpoint, bound, SERVER_NAME, &hello("one-too-many"))
            .await
            .expect_err("a full shard refuses");
        assert_eq!(
            err,
            space_net::NetError::Rejected(space_protocol::RejectReason::Full),
            "the client must learn that the shard is full"
        );
        serve.abort();
        net.close();
    }

    /// A neighbour that is down cannot stop this shard's tick loop.
    ///
    /// More messages than a peer queue holds are queued for a peer whose port
    /// nothing listens on, and `flush` — the call the tick loop makes — must
    /// still return at once, with the overflow counted. When the dial has timed
    /// out, everything that was queued behind it is counted as dropped too.
    ///
    /// Mutation this catches: `sender.tx.send(message).await` in `send_to_peer`
    /// — the flush then waits for a slot the dialling sender does not free, and
    /// the first assertion times out.
    #[tokio::test]
    async fn a_dead_neighbour_cannot_stall_the_tick_loop() {
        let net = Arc::new(ShardNet::bind(loopback()).expect("bind"));
        // A port that was just free: bind one, note it, let it go.
        let dead = {
            let probe = std::net::UdpSocket::bind(loopback()).expect("probe socket");
            probe.local_addr().expect("probe addr")
        };
        let peer = ShardId::new(2);
        let mut shard = Shard::new(crate::ShardConfig::new(
            ShardId::new(1),
            net.local_addr().expect("addr"),
            space_world::NodeRef::new(64, [0, 0, 0]),
        ));
        shard.set_shard_map(space_protocol::ShardMap::new(
            1,
            vec![
                space_protocol::ShardRegion::new(
                    space_world::NodeRef::new(64, [0, 0, 0]),
                    ShardId::new(1),
                    net.local_addr().expect("addr"),
                ),
                space_protocol::ShardRegion::new(
                    space_world::NodeRef::new(64, [1, 0, 0]),
                    peer,
                    dead,
                ),
            ],
        ));
        let queued = PEER_QUEUE as u64 + 50;
        for _ in 0..queued {
            shard.send_to(Destination::Shard(peer), probe());
        }
        let shard = Arc::new(Mutex::new(shard));

        tokio::time::timeout(Duration::from_secs(1), net.flush(&shard))
            .await
            .expect("flush must never wait on a peer queue");
        assert!(
            net.peer_send_failures() >= 49,
            "the messages that did not fit must be counted, got {}",
            net.peer_send_failures()
        );

        // The dial gives up after `PEER_DIAL_TIMEOUT`; the whole backlog goes
        // with it rather than being delivered one failed dial apart.
        let deadline = tokio::time::Instant::now() + PEER_DIAL_TIMEOUT + Duration::from_secs(5);
        while net.peer_send_failures() < queued && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert_eq!(
            net.peer_send_failures(),
            queued,
            "every message for the dead peer must end up counted as dropped"
        );
        net.close();
    }

    /// The tick loop dying is an error, not something to serve through.
    ///
    /// Mutation this catches: `supervise` (or its callers) waiting on `serve`
    /// only — the first case below then never returns.
    #[tokio::test]
    async fn a_dead_tick_loop_stops_the_shard_with_an_error() {
        // The tick loop panics while the accept loop is still running.
        let serve = tokio::spawn(std::future::pending::<()>());
        let ticking = tokio::spawn(async { panic!("tick loop bug") });
        let stopped = tokio::time::timeout(Duration::from_secs(5), supervise(serve, ticking))
            .await
            .expect("a dead tick loop must end the wait");
        let why = stopped.expect_err("a dead tick loop is an error");
        assert!(
            why.contains("tick loop"),
            "the error names the cause: {why}"
        );

        // The ordinary shutdown: the endpoint closed, so the accept loop ended.
        let serve = tokio::spawn(async {});
        let ticking = tokio::spawn(std::future::pending::<()>());
        assert_eq!(supervise(serve, ticking).await, Ok(()));
    }

    /// A hand-registered endpoint must not shadow the map's current address: the
    /// sender `send_to_peer` selects is dialled at the map-pushed endpoint.
    ///
    /// Mutation this catches: `registered.or(from_map)` in `send_to_peer`.
    #[tokio::test]
    async fn a_map_pushed_endpoint_supersedes_a_hand_registered_one() {
        let net = ShardNet::bind(SocketAddr::from(([127, 0, 0, 1], 0))).expect("bind");
        let peer = ShardId::new(5);
        let stale = SocketAddr::from(([127, 0, 0, 1], 9_001));
        let fresh = SocketAddr::from(([127, 0, 0, 1], 9_002));
        net.set_shard_endpoint(peer, stale).await;

        // No map entry yet: the registered endpoint is the fallback.
        net.send_to_peer(ShardId::new(1), peer, None, probe()).await;
        assert_eq!(
            net.peers.lock().await.get(&peer).map(|s| s.addr),
            Some(stale)
        );

        // The map now names a new address: sends must target it.
        net.send_to_peer(ShardId::new(1), peer, Some(fresh), probe())
            .await;
        assert_eq!(
            net.peers.lock().await.get(&peer).map(|s| s.addr),
            Some(fresh)
        );
        net.close();
    }
}
