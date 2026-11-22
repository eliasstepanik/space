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

use std::collections::HashMap;
use std::io;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use space_net::{accept_incoming, client_endpoint, connect, hello, server_endpoint, Conn};
use space_protocol::{Message, Reject, ShardHello, ShardId, ShardWelcome};
use tokio::sync::{mpsc, Mutex};

use crate::shard::{Destination, Shard};

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
        let welcome = {
            let mut s = shard.lock().await;
            match s.join(&hello) {
                Ok(w) => w,
                Err(reason) => {
                    let _ = conn.send_control(&Message::Reject(Reject { reason })).await;
                    return;
                }
            }
        };
        if conn
            .send_control(&Message::Welcome(welcome.clone()))
            .await
            .is_err()
        {
            shard.lock().await.leave(welcome.session);
            return;
        }
        let (tx, mut rx) = mpsc::channel::<Message>(WRITE_QUEUE);
        self.writers
            .lock()
            .await
            .insert(welcome.session, tx.clone());
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
                _ => break,
            }
        }
    }

    /// Deliver everything the shard queued.
    ///
    /// Client frames go into that client's write queue (dropped if it is full: a
    /// client that cannot keep up loses frames, and the next delta or snapshot
    /// resynchronises it). Shard-to-shard frames go into that peer's FIFO sender,
    /// which preserves order on one persistent connection (see the module docs).
    pub async fn flush(&self, shard: &Arc<Mutex<Shard>>) {
        let outbox = shard.lock().await.take_outbox();
        let local = shard.lock().await.shard();
        {
            let writers = self.writers.lock().await;
            for out in &outbox {
                if let Destination::Client(session) = out.dest {
                    if let Some(tx) = writers.get(&session) {
                        // `try_send` rather than `send`: never let one slow client
                        // hold the tick loop.
                        let _ = tx.try_send(out.message.clone());
                    }
                }
            }
        }
        // Peer messages go out in outbox order through each peer's FIFO sender.
        for out in outbox {
            match out.dest {
                Destination::Shard(shard_id) => {
                    // An endpoint registered by hand wins; otherwise the peer is
                    // dialled at the address the shard's own map names for it, so
                    // a shard that appears through an epoch change is reachable
                    // without any extra wiring.
                    let from_map = shard.lock().await.peer_endpoint(shard_id);
                    self.send_to_peer(local, shard_id, from_map, out.message)
                        .await;
                }
                Destination::Orchestrator => {
                    if let Some(tx) = self.orchestrator.lock().await.as_ref() {
                        let _ = tx.send(out.message).await;
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
    /// same channel and the same sender task. An unknown endpoint, or a sender
    /// that has gone away, is logged and counted — never silently dropped.
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
        if sender.tx.send(message).await.is_err() {
            self.peer_send_failures.fetch_add(1, Ordering::SeqCst);
            eprintln!("space-server: peer sender for shard {shard_id} gone; message dropped");
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

/// Drain `rx` and write each message to `addr` on one persistent connection.
///
/// The connection is opened on the first message and reused; if it closes, the
/// next message reconnects. Every message is written with
/// [`Conn::send_reliable`] on that persistent connection — never a fresh
/// connection per message, and never closed right after a send. A connect or send
/// failure bumps `failures` and logs, and the loop keeps trying on the next
/// message rather than tearing the sender down (a transient failure must not
/// permanently stall the transfer).
async fn run_peer_sender(
    mut rx: mpsc::Receiver<Message>,
    addr: SocketAddr,
    peer: ShardId,
    local: ShardId,
    dead: Arc<AtomicBool>,
    failures: Arc<AtomicU64>,
) {
    // One client endpoint for the life of the sender: a quinn `Endpoint` drives
    // its connections, so it must outlive them.
    let endpoint = match client_endpoint() {
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
            conn = match connect(&endpoint, addr, SERVER_NAME, &peer_hello(local)).await {
                Ok((conn, _welcome)) => Some(conn),
                Err(e) => {
                    failures.fetch_add(1, Ordering::SeqCst);
                    eprintln!("space-server: connect to peer shard {peer} at {addr} failed: {e}");
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
/// One connection, one FIFO sender (epoch acks and load reports go out in order)
/// and one reader task (epoch prepare/commit/abort come in and are fed to
/// [`Shard::orchestrator_message`]). It reuses `run_peer_sender`'s shape because
/// order matters for epoch messages for the same reason it matters for handoff
/// messages.
pub struct OrchestratorLink {
    /// The shard's outbound sender: queue a message to send to the orchestrator.
    pub tx: mpsc::Sender<Message>,
    /// The reader task handle (kept so a caller can abort it on shutdown).
    reader: tokio::task::JoinHandle<()>,
    /// The sender task handle.
    sender: tokio::task::JoinHandle<()>,
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
            client_endpoint().map_err(|e| space_net::NetError::Handshake(e.to_string()))?;
        let (conn, _welcome) =
            connect(&endpoint, addr, SERVER_NAME, &peer_hello(hello.shard)).await?;
        // The orchestrator answered the transport handshake with a placeholder;
        // now send the real ShardHello and read the ShardWelcome back.
        conn.send_control(&Message::ShardHello(hello))
            .await
            .map_err(|e| space_net::NetError::Io(e.to_string()))?;
        // The welcome comes back on the reliable stream (the orchestrator's
        // per-shard FIFO sender writes every message, welcome included, with
        // `send_reliable`), so read whichever path delivers it.
        let next = tokio::select! {
            m = conn.recv_control() => m,
            m = conn.recv_stream() => m,
        };
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
                if sender_conn.send_reliable(&message).await.is_err() {
                    return;
                }
            }
        });
        // Inbound: epoch prepare/commit/abort, fed to the shard.
        let reader = {
            let conn = conn.clone();
            tokio::spawn(async move {
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
                        Ok(Some(message)) => match message {
                            Message::EpochPrepare(_)
                            | Message::EpochCommit(_)
                            | Message::EpochAbort(_)
                            | Message::DirectoryReply(_) => {
                                shard.lock().await.orchestrator_message(message);
                            }
                            _ => {}
                        },
                        _ => break,
                    }
                }
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
}
