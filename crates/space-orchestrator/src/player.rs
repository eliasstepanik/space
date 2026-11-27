//! The **player port**: the one address players type
//! (`docs/plan-sharding.md` §4).
//!
//! A second QUIC listener beside the control port. A client says hello to it
//! exactly as it would to a shard, and is answered — in place of a `Welcome`
//! — with one of two things, after which the connection is closed:
//!
//! * a [`JoinRedirect`] naming **the shard that owns the scene's spawn point**
//!   and carrying the published map, which says where players reach that shard
//!   ([`space_protocol::ShardRegion::client_endpoint`]); the client dials it
//!   and plays there. Game traffic flows between the client and the shard:
//!   the orchestrator is no proxy and sees none of it;
//! * a `Reject` with [`RejectReason::Starting`] while there is no such shard
//!   to send a player to — the cluster is starting, or the shard that owns the
//!   spawn point died and its replacement has not taken over yet. A client
//!   that keeps its connection up by itself tries again.
//!
//! # What it is, and what it cannot do
//!
//! The player port is the cluster's **public entry**, not a control surface:
//! it needs no secret on any bind, because there is nothing behind it to
//! protect.
//!
//! * **It decides nothing and changes nothing.** [`PlayerServer`] holds one
//!   value, [`PlayerAnswer`], which the orchestrator's run loop publishes
//!   ([`PlayerServer::publish`]); a connection's task reads it and writes it.
//!   The server has no way into the orchestrator: no queue to the run loop,
//!   no handle on the control plane. A `ShardHello`, an ack, a load report —
//!   whatever a peer sends here — is never read, let alone acted on: **no
//!   shard can register through this port.**
//! * **It keeps no state per player.** Nothing outlives a connection but two
//!   counters.
//! * **It is cheap.** A first packet from an address that has not been
//!   validated is answered with a QUIC retry — no slot, no handshake; a real
//!   client comes back with the token within the same connect call. A
//!   connection then gets one bounded wait for its hello ([`HELLO_TIMEOUT`]),
//!   the answer — a write that gives up after a second if the peer grants no
//!   flow-control window ([`space_net::LAST_WORD_TIMEOUT`]) — and a close
//!   that waits at most a second for the answer to be read; the whole of it
//!   is bounded once more from outside ([`ANSWER_SLACK`]). At most
//!   [`MAX_PENDING`] connections are in that state at once; one more is
//!   refused before any handshake work is done for it. There is **no limit
//!   per source address**: one host can take all the room for as long as it
//!   keeps dialing (`docs/security-notes.md`).
//! * **It cannot be made to fill a log.** The lines a peer can cause (being
//!   refused for the limit above, a shard that dialed the wrong port) go
//!   through a [`space_net::LogBudget`]: a few per window, the rest counted.
//!
//! The version gate is the transport's, as everywhere
//! ([`space_net::accept_incoming_within`]): a client built for another
//! protocol version is told so with a readable `Reject` and never gets as far
//! as an answer from here.

use std::io;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use space_protocol::{JoinRedirect, Message, Reject, RejectReason};
use tokio::sync::{watch, Semaphore};

/// How long a freshly accepted player connection has to complete the
/// handshake and send its hello before it is dropped.
pub const HELLO_TIMEOUT: Duration = Duration::from_secs(5);

/// The most connections the player port serves at once (each lives for a
/// handshake and an answer). One more is refused unanswered.
pub const MAX_PENDING: usize = 256;

/// What the bound on one whole connection adds to the sum of its parts (the
/// hello's limit and [`space_net::TURN_AWAY_LIMIT`]): room for the scheduler,
/// so the parts' own limits are what normally ends a connection.
pub const ANSWER_SLACK: Duration = Duration::from_millis(500);

/// How many lines a peer can cause the player port to write per
/// [`LOG_WINDOW`]; the rest are counted and summed up in the next one.
pub const LOG_LINES: u32 = 3;

/// The window [`LOG_LINES`] applies to.
pub const LOG_WINDOW: Duration = Duration::from_secs(10);

/// What the player port says to a hello right now. Published by the
/// orchestrator's run loop, read by every connection.
#[derive(Debug, Clone, PartialEq)]
pub enum PlayerAnswer {
    /// Dial that shard: it owns the scene's spawn point, holds a control
    /// link, and the map says where players reach it.
    Redirect(JoinRedirect),
    /// There is no shard to send a player to right now: try again shortly.
    Starting,
}

impl PlayerAnswer {
    /// The message that answers a hello: the redirect, or the refusal.
    #[must_use]
    pub fn into_message(self) -> Message {
        match self {
            PlayerAnswer::Redirect(redirect) => Message::JoinRedirect(redirect),
            PlayerAnswer::Starting => Message::Reject(Reject {
                reason: RejectReason::Starting,
            }),
        }
    }

    /// The shard a player is sent to, if one is.
    #[must_use]
    pub fn shard(&self) -> Option<space_protocol::ShardId> {
        match self {
            PlayerAnswer::Redirect(redirect) => Some(redirect.shard),
            PlayerAnswer::Starting => None,
        }
    }
}

/// The player port's QUIC front end (see the module docs).
#[derive(Debug)]
pub struct PlayerServer {
    endpoint: quinn::Endpoint,
    /// The answer in force. Starts as [`PlayerAnswer::Starting`].
    answer: watch::Sender<PlayerAnswer>,
    /// How long a connection may take to say hello.
    hello_timeout: Duration,
    /// Room for [`MAX_PENDING`] connections.
    room: Arc<Semaphore>,
    /// Hellos answered with a redirect.
    redirected: AtomicU64,
    /// Hellos answered with "starting".
    turned_away: AtomicU64,
    /// Connections refused unanswered for want of room.
    refused: AtomicU64,
    /// First packets answered with a retry (the sender's address was not
    /// validated yet; a real client comes straight back).
    retried: AtomicU64,
    /// Connections dropped because the whole exchange ran out of its bound.
    timed_out: AtomicU64,
    /// The budget for lines a peer can cause.
    lines: space_net::LogBudget,
}

impl PlayerServer {
    /// Bind the player port on `bind`.
    ///
    /// # Errors
    ///
    /// [`io::Error`] when the socket cannot be bound.
    pub fn bind(bind: SocketAddr) -> io::Result<Self> {
        Self::bind_trusted(bind, None)
    }

    /// Bind the player port on `bind`, presenting certificates issued by the
    /// cluster CA of `trust`; `None` presents a throwaway self-signed one
    /// ([`Self::bind`]).
    ///
    /// # Errors
    ///
    /// [`io::Error`] when the socket cannot be bound.
    pub fn bind_trusted(
        bind: SocketAddr,
        trust: Option<&space_net::ClusterTrust>,
    ) -> io::Result<Self> {
        let config = match trust {
            Some(trust) => trust.server_config(),
            None => space_net::crypto::make_server_config().0,
        };
        let endpoint = space_net::server_endpoint(bind, config)?;
        let (answer, _) = watch::channel(PlayerAnswer::Starting);
        Ok(Self {
            endpoint,
            answer,
            hello_timeout: HELLO_TIMEOUT,
            room: Arc::new(Semaphore::new(MAX_PENDING)),
            redirected: AtomicU64::new(0),
            turned_away: AtomicU64::new(0),
            refused: AtomicU64::new(0),
            retried: AtomicU64::new(0),
            timed_out: AtomicU64::new(0),
            lines: space_net::LogBudget::new(LOG_LINES, LOG_WINDOW),
        })
    }

    /// Use `limit` instead of [`HELLO_TIMEOUT`] (a test that wants the bound
    /// to run out quickly).
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

    /// Make `answer` what every hello from now on is answered with. Cheap
    /// when nothing changed: the run loop calls this on every beat.
    pub fn publish(&self, answer: PlayerAnswer) {
        self.answer.send_if_modified(|held| {
            if *held == answer {
                false
            } else {
                *held = answer;
                true
            }
        });
    }

    /// The answer in force.
    #[must_use]
    pub fn answer(&self) -> PlayerAnswer {
        self.answer.borrow().clone()
    }

    /// How many hellos were answered with a redirect.
    #[must_use]
    pub fn redirected(&self) -> u64 {
        self.redirected.load(Ordering::Relaxed)
    }

    /// How many hellos were answered with "starting".
    #[must_use]
    pub fn turned_away(&self) -> u64 {
        self.turned_away.load(Ordering::Relaxed)
    }

    /// How many connections were refused unanswered because
    /// [`MAX_PENDING`] were being served.
    #[must_use]
    pub fn refused(&self) -> u64 {
        self.refused.load(Ordering::Relaxed)
    }

    /// How many first packets were answered with a QUIC retry instead of a
    /// handshake (every honest connection counts once here).
    #[must_use]
    pub fn retried(&self) -> u64 {
        self.retried.load(Ordering::Relaxed)
    }

    /// How many connections were dropped because hello, answer and close
    /// together ran out of their bound.
    #[must_use]
    pub fn timed_out(&self) -> u64 {
        self.timed_out.load(Ordering::Relaxed)
    }

    /// Start accepting; each connection is answered in its own task. Returns
    /// the accept task's handle so a caller can abort it on shutdown.
    pub fn spawn_accept(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            loop {
                let Some(incoming) = self.endpoint.accept().await else {
                    return; // endpoint closed
                };
                // Is the sender where it says it is? A first packet proves
                // nothing about its source address, and this port is public:
                // answered as it is, a forged source would have the port do
                // a TLS handshake's work — and send its certificate — towards
                // somebody else, and hold one of the slots below while it
                // waits. So it is answered with a QUIC retry (a stateless
                // token, no slot, no handshake); a real client comes back
                // with the token at once and is `remote_address_validated`.
                if !incoming.remote_address_validated() {
                    self.retried.fetch_add(1, Ordering::Relaxed);
                    // (Cannot fail: an unvalidated connection may be retried.)
                    let _ = incoming.retry();
                    continue;
                }
                // Room for it? Refused before any handshake work otherwise.
                let Ok(permit) = Arc::clone(&self.room).try_acquire_owned() else {
                    incoming.refuse();
                    self.refused.fetch_add(1, Ordering::Relaxed);
                    if let Some(held_back) = self.lines.admit() {
                        eprintln!(
                            "space-orchestrator: the player port is answering {MAX_PENDING} \
                             connections at once and refused one more{}",
                            space_net::held_back_note(held_back)
                        );
                    }
                    continue;
                };
                let server = Arc::clone(&self);
                tokio::spawn(async move {
                    // The whole of it is bounded, whatever the parts do: the
                    // hello, the answer and the close each have a limit of
                    // their own, and this one stands behind them. Running
                    // out of it drops the connection, which closes it.
                    let limit = server.hello_timeout + space_net::TURN_AWAY_LIMIT + ANSWER_SLACK;
                    if tokio::time::timeout(limit, server.answer_one(incoming))
                        .await
                        .is_err()
                    {
                        server.timed_out.fetch_add(1, Ordering::Relaxed);
                    }
                    drop(permit);
                });
            }
        })
    }

    /// One connection: the hello (bounded), the answer, the close. Nothing
    /// else the peer sends is ever read.
    async fn answer_one(&self, incoming: quinn::Incoming) {
        let from = incoming.remote_address();
        // The version gate and the wait for the hello are the transport's: a
        // peer of another version has been told so, one that never says hello
        // is dropped when the bound runs out.
        let Ok((conn, hello)) =
            space_net::accept_incoming_within(incoming, self.hello_timeout).await
        else {
            return;
        };
        // A shard that was pointed at this port instead of the control port
        // (its transport hello names it `shard:<id>`): told what this is. It
        // is given the same kind of answer as anybody — nothing is registered
        // here, whatever it goes on to send.
        if space_server::net::peer_shard_id(&hello.player_token).is_some() {
            if let Some(held_back) = self.lines.admit() {
                eprintln!(
                    "space-orchestrator: a shard's hello arrived on the player port (from \
                     {from}): shards register at the control port{}",
                    space_net::held_back_note(held_back)
                );
            }
            conn.reject(RejectReason::Other(
                "this is the cluster's player port; a shard registers at the orchestrator's \
                 control port (--bind)"
                    .into(),
            ))
            .await;
            return;
        }
        let answer = self.answer();
        let counter = match answer {
            PlayerAnswer::Redirect(_) => &self.redirected,
            PlayerAnswer::Starting => &self.turned_away,
        };
        counter.fetch_add(1, Ordering::Relaxed);
        // Written, finished, given a moment to be read, closed.
        conn.turn_away(&answer.into_message()).await;
    }

    /// Stop accepting.
    pub fn close(&self) {
        self.endpoint
            .close(0u32.into(), b"orchestrator shutting down");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use space_net::{Joined, NetError};
    use space_protocol::{Hello, ShardId, ShardMap, ShardRegion, PROTOCOL_VERSION};
    use space_test_support::loopback0;
    use std::time::Instant;

    const SERVER_NAME: &str = crate::control::SERVER_NAME;

    /// A player port serving on loopback, and its address.
    fn serving(hello_timeout: Duration) -> (Arc<PlayerServer>, SocketAddr) {
        let server = Arc::new(
            PlayerServer::bind(loopback0())
                .expect("bind")
                .with_hello_timeout(hello_timeout),
        );
        let addr = server.local_addr().expect("bound");
        // (Detached: it ends with the endpoint, when the server is dropped.)
        drop(Arc::clone(&server).spawn_accept());
        (server, addr)
    }

    /// A pinned client trusts a player port bound with the cluster trust and
    /// is told `Untrusted` by a plain one.
    #[tokio::test]
    async fn a_pinned_client_trusts_a_player_port_bound_with_the_cluster_trust() {
        let trust = space_net::ClusterTrust::from_secret("the-cluster-secret-0123").expect("trust");
        let pin = trust.pin();
        let trusted =
            Arc::new(PlayerServer::bind_trusted(loopback0(), Some(&trust)).expect("bind"));
        let plain = Arc::new(PlayerServer::bind(loopback0()).expect("bind"));
        let t_addr = trusted.local_addr().expect("bound");
        let p_addr = plain.local_addr().expect("bound");
        drop(Arc::clone(&trusted).spawn_accept());
        drop(Arc::clone(&plain).spawn_accept());

        let hello = space_net::hello("player");
        let endpoint = space_net::client_endpoint_pinned(t_addr, Some(&pin)).expect("endpoint");
        // The handshake completed: the port answered "starting".
        let told = space_net::join(&endpoint, t_addr, SERVER_NAME, &hello)
            .await
            .expect_err("nothing published yet");
        assert_eq!(told, NetError::Rejected(RejectReason::Starting));
        let endpoint = space_net::client_endpoint_pinned(p_addr, Some(&pin)).expect("endpoint");
        let err = space_net::join(&endpoint, p_addr, SERVER_NAME, &hello)
            .await
            .expect_err("a plain port is not the cluster's");
        assert!(matches!(err, NetError::Untrusted(_)), "{err:?}");
    }

    fn redirect_to(shard: u32) -> JoinRedirect {
        let region = ShardRegion::new(
            space_world::NodeRef::new(40, [0, 0, 0]),
            ShardId::new(shard),
            "127.0.0.1:7870".parse().expect("an address"),
        )
        .with_client_endpoint("0.0.0.0:7870".parse().expect("an address"));
        JoinRedirect {
            shard: ShardId::new(shard),
            shard_map: ShardMap::new(3, vec![region]),
        }
    }

    async fn hello(addr: SocketAddr, hello: &Hello) -> Result<Joined, NetError> {
        let endpoint = space_net::client_endpoint_for(addr).expect("client endpoint");
        space_net::join(&endpoint, addr, SERVER_NAME, hello).await
    }

    /// The port says what was last published — "starting" until something
    /// is — to every hello, and counts what it said.
    ///
    /// Mutation this catches: an answer captured when a connection is
    /// accepted for good (a port that started before its shards would say
    /// "starting" for ever), or one that is not the published one.
    #[tokio::test]
    async fn every_hello_is_answered_with_what_was_last_published() {
        let (server, addr) = serving(HELLO_TIMEOUT);
        assert_eq!(server.answer(), PlayerAnswer::Starting);
        let refused = hello(addr, &space_net::hello("a-player"))
            .await
            .expect_err("nothing published yet");
        assert_eq!(refused, NetError::Rejected(RejectReason::Starting));

        server.publish(PlayerAnswer::Redirect(redirect_to(1)));
        for _ in 0..2 {
            let joined = hello(addr, &space_net::hello("a-player"))
                .await
                .expect("an answer");
            let Joined::Elsewhere(redirect) = joined else {
                panic!("expected the redirect, got {joined:?}");
            };
            assert_eq!(redirect, redirect_to(1));
        }
        // The owner changed, then went away.
        server.publish(PlayerAnswer::Redirect(redirect_to(4)));
        let joined = hello(addr, &space_net::hello("a-player"))
            .await
            .expect("an answer");
        assert!(matches!(&joined, Joined::Elsewhere(r) if r.shard == ShardId::new(4)));
        server.publish(PlayerAnswer::Starting);
        assert!(hello(addr, &space_net::hello("a-player")).await.is_err());

        assert_eq!((server.redirected(), server.turned_away()), (3, 2));
        assert_eq!(server.refused(), 0);
    }

    /// The version gate is the same as on every port: a client built for
    /// another protocol version is told so, with both versions, and gets no
    /// redirect.
    #[tokio::test]
    async fn a_client_of_another_version_is_told_so_by_the_player_port() {
        let (server, addr) = serving(HELLO_TIMEOUT);
        server.publish(PlayerAnswer::Redirect(redirect_to(1)));
        let old = Hello {
            protocol: PROTOCOL_VERSION - 1,
            player_token: "an-old-client".into(),
        };
        let err = hello(addr, &old).await.expect_err("another version");
        assert_eq!(
            err,
            NetError::VersionMismatch {
                seen: PROTOCOL_VERSION - 1,
                expected: PROTOCOL_VERSION,
            }
        );
        assert_eq!(
            (server.redirected(), server.turned_away()),
            (0, 0),
            "it never got as far as an answer"
        );
    }

    /// A peer that completes the transport handshake and never says hello is
    /// dropped when the bound runs out; one that says hello and never reads
    /// the answer is closed after the drain. Nothing is held open.
    ///
    /// Mutation this catches: no bound on the hello (a connection and a task
    /// per silent peer, for as long as it likes), or a close that waits for
    /// the peer to go first.
    #[tokio::test]
    async fn a_silent_peer_is_dropped_and_no_connection_is_held_open() {
        let (server, addr) = serving(Duration::from_millis(300));
        server.publish(PlayerAnswer::Redirect(redirect_to(1)));
        let endpoint = space_net::client_endpoint_for(addr).expect("client endpoint");

        // Never says hello. (The clock is read before the dial: the port's
        // bound runs from when *it* has the connection, which on a loaded
        // machine is well before this side hears that the handshake is done
        // — read afterwards, the wait measured here came out under the
        // bound, once in ten runs on one starved CPU.)
        let started = Instant::now();
        let silent = endpoint
            .connect(addr, SERVER_NAME)
            .expect("dial")
            .await
            .expect("the transport handshake");
        tokio::time::timeout(Duration::from_secs(3), silent.closed())
            .await
            .expect("dropped when the hello bound runs out");
        assert!(started.elapsed() >= Duration::from_millis(250));

        // Says hello, never reads, never closes.
        let deaf = endpoint
            .connect(addr, SERVER_NAME)
            .expect("dial")
            .await
            .expect("the transport handshake");
        let (mut send, _recv) = deaf.open_bi().await.expect("the control stream");
        send.write_all(&space_protocol::encode_frame(&Message::Hello(
            space_net::hello("a-deaf-player"),
        )))
        .await
        .expect("the hello");
        tokio::time::timeout(Duration::from_secs(3), deaf.closed())
            .await
            .expect("closed after the drain, whether the answer was read or not");
        assert_eq!(server.redirected(), 1);
        // All the room is free again: nothing is still being served. (The
        // port's task lets go of its slot a moment after it closed.)
        let deadline = Instant::now() + Duration::from_secs(5);
        while server.room.available_permits() != MAX_PENDING && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(server.room.available_permits(), MAX_PENDING);
    }

    /// A first packet proves nothing about where it came from, so it is
    /// answered with a QUIC retry instead of a handshake and takes no slot;
    /// a real client comes back with the token inside the same connect call
    /// and gets its answer as before. One retry per connection, not more.
    ///
    /// Mutation this catches: accepting an unvalidated connection (the port
    /// does handshake work, and holds a slot, for a source address nobody
    /// has shown to exist), or a retry a real client cannot get past.
    #[tokio::test]
    async fn a_first_packet_is_retried_and_a_real_client_still_gets_its_answer() {
        let (server, addr) = serving(HELLO_TIMEOUT);
        server.publish(PlayerAnswer::Redirect(redirect_to(1)));
        for n in 1..=3u64 {
            let joined = hello(addr, &space_net::hello("a-player"))
                .await
                .expect("an answer, through the retry");
            assert!(matches!(&joined, Joined::Elsewhere(r) if r.shard == ShardId::new(1)));
            assert_eq!(server.retried(), n, "one retry per connection");
            assert_eq!(server.redirected(), n);
        }
        assert_eq!((server.refused(), server.timed_out()), (0, 0));
    }

    /// A peer that says hello and then grants the port **no** flow-control
    /// window cannot be written to: the answer's write would wait for as
    /// long as the peer keeps its connection alive, and with it one of the
    /// 256 slots — 256 such peers and the port answers nobody. The write
    /// gives up after a second and the whole connection is bounded besides,
    /// so the slot is free again in a few seconds.
    ///
    /// Mutation this catches: an unbounded write of the last word together
    /// with no bound on `answer_one` (the slot is then never returned and
    /// the last assertion fails).
    #[tokio::test]
    async fn a_peer_that_grants_no_receive_window_does_not_keep_a_slot() {
        let hello_timeout = Duration::from_millis(500);
        let (server, addr) = serving(hello_timeout);
        server.publish(PlayerAnswer::Redirect(redirect_to(1)));

        let mut transport = quinn::TransportConfig::default();
        transport.stream_receive_window(quinn::VarInt::from_u32(0));
        transport.keep_alive_interval(Some(space_net::KEEP_ALIVE_INTERVAL));
        let mut config = space_net::crypto::make_client_config();
        config.transport_config(Arc::new(transport));
        let mut endpoint = quinn::Endpoint::client(loopback0()).expect("client endpoint");
        endpoint.set_default_client_config(config);

        let mut held = Vec::new();
        for _ in 0..4 {
            let conn = endpoint
                .connect(addr, SERVER_NAME)
                .expect("dial")
                .await
                .expect("the transport handshake");
            let (mut send, recv) = conn.open_bi().await.expect("the control stream");
            send.write_all(&space_protocol::encode_frame(&Message::Hello(
                space_net::hello("a-windowless-player"),
            )))
            .await
            .expect("the hello");
            held.push((conn, send, recv));
        }
        // All four are being answered: each holds a slot now.
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(server.room.available_permits(), MAX_PENDING - 4);

        // And none of them holds it for long, although every one of them
        // keeps its connection up and never reads.
        let limit = hello_timeout + space_net::TURN_AWAY_LIMIT + ANSWER_SLACK;
        let started = Instant::now();
        while server.room.available_permits() != MAX_PENDING {
            assert!(
                started.elapsed() < limit + Duration::from_secs(1),
                "{} slots still held after {:?}",
                MAX_PENDING - server.room.available_permits(),
                started.elapsed()
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        for (conn, _send, _recv) in &held {
            tokio::time::timeout(Duration::from_secs(3), conn.closed())
                .await
                .expect("the port closed the connection");
        }
        // A real client is served as ever.
        let joined = hello(addr, &space_net::hello("a-player"))
            .await
            .expect("an answer");
        assert!(matches!(joined, Joined::Elsewhere(_)));
    }

    /// A shard pointed at the player port (its transport hello names it
    /// `shard:<id>`) is told what it dialed. Nothing is registered.
    #[tokio::test]
    async fn a_shards_hello_on_the_player_port_is_told_what_it_dialed() {
        let (server, addr) = serving(HELLO_TIMEOUT);
        server.publish(PlayerAnswer::Redirect(redirect_to(1)));
        let err = hello(addr, &space_server::net::peer_hello(ShardId::NONE))
            .await
            .expect_err("not a control port");
        let NetError::Rejected(RejectReason::Other(why)) = &err else {
            panic!("a readable refusal, got {err:?}");
        };
        assert!(
            why.contains("player port") && why.contains("control port"),
            "{why}"
        );
        assert_eq!((server.redirected(), server.turned_away()), (0, 0));
    }
}
