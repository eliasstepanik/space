//! Integration tests for `space-net` (Phase 2, task 2.1).
//!
//! Run with `cargo test -p space-net` (or `cargo xtask ci`).
//!
//! Everything here runs over a real quinn endpoint pair on loopback, so the TLS
//! stack, the datagram path and the stream path are all exercised for real.
//!
//! | test | checks |
//! |------|--------|
//! | [`hello_handshake_round_trip`] | client `connect` + server `accept_connection` → `Welcome` with the session id and shard map, `Hello` seen server-side |
//! | [`version_mismatch_is_rejected`] | a `Hello` with a wrong `protocol` → client gets `NetError::VersionMismatch`, server gets `NetError::VersionMismatch` |
//! | [`a_peer_built_for_another_protocol_version_is_refused_at_the_hello`] | a hello whose **frame header** names another version (what a build at another version really sends) → the server answers a `Reject` naming both versions and reports `VersionMismatch` |
//! | [`a_server_speaking_another_version_is_a_version_mismatch_to_the_client`] | a reply whose frame header names another version → `connect` returns `VersionMismatch`, not a transport error |
//! | [`first_frame_must_be_hello`] | a non-`Hello` first frame → `NetError::Protocol` |
//! | [`datagram_message_round_trips`] | an `InputTick` rides a datagram and decodes identically |
//! | [`oversized_delta_promoted_to_stream`] | a `Delta` past the datagram budget is sent on a stream and still arrives |
//! | [`snapshot_stream_round_trips`] | a `Snapshot` (stream class) round-trips on a fresh uni stream |
//! | [`control_stream_round_trips`] | `send_control`/`recv_control` carry a `Redirect` both ways |
//! | [`disconnect_is_observed`] | after the peer closes, receives return `Ok(None)` and `is_closed()` is true |
//! | [`oversized_frame_rejected_before_alloc`] | a stream frame declaring a length over `MAX_FRAME_SIZE` errors without a huge allocation |
//! | [`a_quiet_connection_outlives_the_idle_timeout`] | keep-alive: a link with no application traffic survives past the idle timeout (takes ~12 s) |
//! | [`connect_gives_up_when_the_server_never_answers_the_hello`] | `connect_within` is bounded by its own limit, not by the idle timeout |
//! | [`accept_gives_up_on_a_client_that_never_sends_a_hello`] | `accept_incoming_within` is bounded the same way |
//! | [`a_policy_reject_reaches_the_client_with_its_reason`] | `Conn::reject` drains before closing, so the client sees `Rejected(reason)` |
//! | [`join_returns_the_redirect_a_server_turned_it_away_with`] | a `JoinRedirect` in place of a welcome reaches `join` whole (`Joined::Elsewhere`), and the connection is finished |
//! | [`connect_reports_a_redirect_as_an_error_that_says_what_the_address_is`] | `connect` (a shard's dial) is not sent on: a redirect is a readable error |
//! | [`join_still_takes_a_welcome_and_a_version_mismatch_is_still_readable`] | the two old answers are unchanged through `join` |
//! | [`client_endpoint_for_binds_the_family_it_dials`], [`an_ipv6_server_is_diallable_through_client_endpoint_for`] | the client socket is bound in the address family of its target |

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use space_net::{
    accept_connection, client_endpoint_on, connect, join, server_endpoint, Joined, MessageClass,
    NetError,
};
use space_protocol::{
    encode_frame, Delta, EntityGuid, EntitySnapshot, Hello, InputTick, JoinRedirect, Message,
    PlayerInput, Redirect, RejectReason, ShardId, ShardMap, Snapshot, Welcome, MAX_FRAME_SIZE,
    PROTOCOL_VERSION,
};
use space_world::WorldPos;

/// Loopback address both endpoints use. IPv4 on purpose: quinn binds IPv6
/// sockets dual-stack, which Windows rejects with WSAEINVAL (10022).
fn addr() -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)
}

/// A shard map with one region covering the whole world (depth 0), owned by
/// shard 7 and reachable at `endpoint`.
fn shard_map(endpoint: SocketAddr) -> ShardMap {
    ShardMap::new(
        1,
        vec![space_protocol::ShardRegion::new(
            space_world::NodeRef::new(0, [0, 0, 0]),
            ShardId::new(7),
            endpoint,
        )],
    )
}

/// A server endpoint bound to loopback plus the certificate it presents.
fn server() -> (quinn::Endpoint, SocketAddr) {
    let (config, _cert) = space_net::crypto::make_server_config();
    let endpoint = server_endpoint(addr(), config).expect("bind server");
    let bound = endpoint.local_addr().expect("local addr");
    (endpoint, bound)
}

fn welcome(endpoint: SocketAddr) -> Welcome {
    Welcome {
        session: 42,
        shard: ShardId::new(7),
        shard_map: shard_map(endpoint),
    }
}

fn input_tick(tick: u64) -> Message {
    Message::InputTick(InputTick {
        tick,
        seq: 1,
        inputs: vec![PlayerInput {
            entity: EntityGuid::new(ShardId::new(7), 1),
            thrust: [0.0, 0.0, 1_000.0],
            torque: [0.0; 3],
            tick,
            fast: false,
        }],
        ack_tick: 0,
    })
}

#[tokio::test]
async fn hello_handshake_round_trip() {
    let (server, bound) = server();
    let server_task = tokio::spawn(async move {
        let (conn, hello) = accept_connection(server.accept()).await.expect("accept");
        assert_eq!(hello.protocol, PROTOCOL_VERSION);
        assert_eq!(hello.player_token, "player-1");
        conn.send_control(&Message::Welcome(welcome(
            "127.0.0.1:0".parse().expect("addr"),
        )))
        .await
        .expect("send welcome");
        conn
    });

    let client = client_endpoint_on(addr()).expect("client endpoint");
    let (conn, got) = connect(
        &client,
        bound,
        "space.local",
        &Hello {
            protocol: PROTOCOL_VERSION,
            player_token: "player-1".into(),
        },
    )
    .await
    .expect("connect");

    assert_eq!(got.session, 42);
    assert_eq!(got.shard, ShardId::new(7));
    assert_eq!(got.shard_map.epoch, 1);
    assert_eq!(got.shard_map.regions.len(), 1);
    assert!(!conn.is_closed());
    server_task.await.expect("server task");
    client.close(0u32.into(), b"done");
}

#[tokio::test]
async fn version_mismatch_is_rejected() {
    let (server, bound) = server();
    let server_task = tokio::spawn(async move {
        // The server refuses and reports the mismatch itself.
        let err = accept_connection(server.accept())
            .await
            .expect_err("a wrong protocol must be refused");
        match err {
            NetError::VersionMismatch { seen, expected } => {
                assert_eq!(seen, PROTOCOL_VERSION + 1);
                assert_eq!(expected, PROTOCOL_VERSION);
            }
            other => panic!("expected VersionMismatch, got {other:?}"),
        }
    });

    let client = client_endpoint_on(addr()).expect("client endpoint");
    let err = connect(
        &client,
        bound,
        "space.local",
        &Hello {
            protocol: PROTOCOL_VERSION + 1,
            player_token: "player-1".into(),
        },
    )
    .await
    .expect_err("the server must refuse a wrong protocol");
    match err {
        NetError::VersionMismatch { seen, expected } => {
            assert_eq!(seen, PROTOCOL_VERSION + 1);
            assert_eq!(expected, PROTOCOL_VERSION);
        }
        other => panic!("expected VersionMismatch, got {other:?}"),
    }
    server_task.await.expect("server task");
}

/// `frame` with the protocol version in its header replaced by `version`: what
/// a build at that version puts on the wire.
fn frame_at_version(message: &Message, version: u32) -> Vec<u8> {
    let mut frame = encode_frame(message);
    frame[4..8].copy_from_slice(&version.to_be_bytes());
    frame
}

/// A peer built for another protocol version is turned away **at the hello**,
/// with the reason — not dropped, and not let into a session it would fail in.
///
/// [`version_mismatch_is_rejected`] sends a hello that only *claims* another
/// version inside a current-version frame. A real older build differs in the
/// frame header too, and that is checked first: nothing it sends decodes. This
/// is that case.
///
/// Mutation this catches: treating the undecodable hello like any other
/// transport error (the connection is dropped and the peer sees "connection
/// lost" — indistinguishable from a crashed server).
#[tokio::test]
async fn a_peer_built_for_another_protocol_version_is_refused_at_the_hello() {
    let theirs = PROTOCOL_VERSION - 1;
    let (server, bound) = server();
    let server_task = tokio::spawn(async move {
        let err = accept_connection(server.accept())
            .await
            .expect_err("another protocol version must be refused");
        assert_eq!(
            err,
            NetError::VersionMismatch {
                seen: theirs,
                expected: PROTOCOL_VERSION
            }
        );
    });

    // A raw client: QUIC and TLS as usual, then a hello as that build frames it.
    let client = client_endpoint_on(addr()).expect("client endpoint");
    let conn = client
        .connect(bound, "space.local")
        .expect("connect")
        .await
        .expect("handshake");
    let (mut send, mut recv) = conn.open_bi().await.expect("open bi");
    let hello = Message::Hello(Hello {
        protocol: theirs,
        player_token: "old-build".into(),
    });
    send.write_all(&frame_at_version(&hello, theirs))
        .await
        .expect("write hello");

    // The answer arrives before the connection closes: a reject naming both
    // versions, in a frame whose header says which version the server speaks.
    let mut header = [0u8; 8];
    recv.read_exact(&mut header)
        .await
        .expect("the server answers instead of hanging up");
    let length = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as usize;
    assert_eq!(
        u32::from_be_bytes([header[4], header[5], header[6], header[7]]),
        PROTOCOL_VERSION,
        "even a peer that cannot decode the payload reads the version here"
    );
    let mut frame = header.to_vec();
    frame.resize(8 + length, 0);
    recv.read_exact(&mut frame[8..]).await.expect("the payload");
    assert_eq!(
        space_protocol::decode_frame(&frame),
        Ok(Message::Reject(space_protocol::Reject {
            reason: RejectReason::VersionMismatch {
                seen: theirs,
                expected: PROTOCOL_VERSION
            }
        }))
    );
    conn.close(0u32.into(), b"read it");
    server_task.await.expect("server task");
}

/// The other end of the same situation: this build dials a server that speaks
/// another version. The server's answer cannot be decoded, but its frame
/// header says why — and `connect` reports exactly that.
///
/// Mutation this catches: surfacing the undecodable reply as a transport
/// error, so a client pointed at a not-yet-updated server reports an I/O
/// failure instead of "version mismatch".
#[tokio::test]
async fn a_server_speaking_another_version_is_a_version_mismatch_to_the_client() {
    let theirs = PROTOCOL_VERSION + 1;
    let (server, bound) = server();
    let server_task = tokio::spawn(async move {
        // A raw server at another version: it reads whatever the client sent
        // and answers in its own framing.
        let conn = server
            .accept()
            .await
            .expect("incoming")
            .await
            .expect("handshake");
        let (mut send, mut recv) = conn.accept_bi().await.expect("the control stream");
        let mut header = [0u8; 8];
        recv.read_exact(&mut header)
            .await
            .expect("the hello's header");
        let reject = Message::Reject(space_protocol::Reject {
            reason: RejectReason::VersionMismatch {
                seen: PROTOCOL_VERSION,
                expected: theirs,
            },
        });
        send.write_all(&frame_at_version(&reject, theirs))
            .await
            .expect("write the answer");
        let _ = send.finish();
        // Stay until the client has read it.
        conn.closed().await;
    });

    let client = client_endpoint_on(addr()).expect("client endpoint");
    let err = connect(&client, bound, "space.local", &space_net::hello("player-1"))
        .await
        .expect_err("another protocol version cannot be joined");
    assert_eq!(
        err,
        NetError::VersionMismatch {
            seen: PROTOCOL_VERSION,
            expected: theirs
        }
    );
    server_task.await.expect("server task");
}

#[tokio::test]
async fn first_frame_must_be_hello() {
    let (server, bound) = server();
    let server_task = tokio::spawn(async move {
        let err = accept_connection(server.accept())
            .await
            .expect_err("a non-hello first frame must be refused");
        assert!(
            matches!(err, NetError::Protocol(_)),
            "expected Protocol error, got {err:?}"
        );
    });

    // A raw client that sends an InputTick instead of a Hello.
    let client = client_endpoint_on(addr()).expect("client endpoint");
    let conn = client
        .connect(bound, "space.local")
        .expect("connect")
        .await
        .expect("handshake");
    let (mut send, _recv) = conn.open_bi().await.expect("open bi");
    let frame = encode_frame(&input_tick(1));
    send.write_all(&frame).await.expect("write frame");
    server_task.await.expect("server task");
}

#[tokio::test]
async fn datagram_message_round_trips() {
    let (server, bound) = server();
    let server_task = tokio::spawn(async move {
        let (conn, _hello) = accept_connection(server.accept()).await.expect("accept");
        conn.send_control(&Message::Welcome(welcome(bound)))
            .await
            .expect("welcome");
        let msg = conn
            .recv_datagram()
            .await
            .expect("recv")
            .expect("a message");
        msg
    });

    let client = client_endpoint_on(addr()).expect("client endpoint");
    let (conn, _welcome) = connect(&client, bound, "space.local", &space_net::hello("player-1"))
        .await
        .expect("connect");
    // An input tick is datagram-class and small: it must fit the budget.
    let msg = input_tick(5);
    assert_eq!(MessageClass::of(&msg), MessageClass::Datagram);
    let budget =
        space_net::framed::datagram_frame_budget(conn.connection()).expect("datagrams are enabled");
    assert!(space_protocol::encode_frame(&msg).len() - space_protocol::FRAME_HEADER_LEN <= budget);
    conn.send(&msg).await.expect("send input");

    let got = server_task.await.expect("server task");
    assert_eq!(got, msg);
}

/// A `Delta` with enough entities to blow past one datagram.
fn big_delta(tick: u64) -> Message {
    let entities: Vec<EntitySnapshot> = (0..200)
        .map(|i| {
            EntitySnapshot::new(
                EntityGuid::new(ShardId::new(7), i),
                vec![(0, space_sim_wire_placeholder_payload())],
            )
        })
        .collect();
    Message::Delta(Delta {
        tick,
        base_tick: tick - 1,
        shard: ShardId::new(7),
        epoch: 1,
        entities,
        removed: Vec::new(),
        input_ack: 0,
    })
}

/// ~200 bytes of opaque component payload, so 200 entities exceed the datagram
/// budget by a wide margin.
fn space_sim_wire_placeholder_payload() -> Vec<u8> {
    vec![0xAB; 200]
}

#[tokio::test]
async fn oversized_delta_promoted_to_stream() {
    let (server, bound) = server();
    let server_task = tokio::spawn(async move {
        let (conn, _hello) = accept_connection(server.accept()).await.expect("accept");
        conn.send_control(&Message::Welcome(welcome(bound)))
            .await
            .expect("welcome");
        // The delta was promoted to the stream path, so read it from a stream.
        conn.recv_stream().await.expect("recv").expect("a message")
    });

    let client = client_endpoint_on(addr()).expect("client endpoint");
    let (conn, _welcome) = connect(&client, bound, "space.local", &space_net::hello("player-1"))
        .await
        .expect("connect");
    let msg = big_delta(10);
    let budget =
        space_net::framed::datagram_frame_budget(conn.connection()).expect("datagrams are enabled");
    let frame_len = space_protocol::encode_frame(&msg).len();
    assert!(
        frame_len - space_protocol::FRAME_HEADER_LEN > budget,
        "test setup: the delta must exceed the {budget} B datagram budget (was {} B)",
        frame_len - space_protocol::FRAME_HEADER_LEN
    );
    conn.send(&msg).await.expect("send delta");
    let got = server_task.await.expect("server task");
    assert_eq!(got, msg);
}

#[tokio::test]
async fn snapshot_stream_round_trips() {
    let (server, bound) = server();
    let server_task = tokio::spawn(async move {
        let (conn, _hello) = accept_connection(server.accept()).await.expect("accept");
        conn.send_control(&Message::Welcome(welcome(bound)))
            .await
            .expect("welcome");
        conn.recv_stream().await.expect("recv").expect("a message")
    });

    let client = client_endpoint_on(addr()).expect("client endpoint");
    let (conn, _welcome) = connect(&client, bound, "space.local", &space_net::hello("player-1"))
        .await
        .expect("connect");
    let msg = Message::Snapshot(Snapshot {
        tick: 3,
        shard: ShardId::new(7),
        epoch: 1,
        entities: vec![EntitySnapshot::new(
            EntityGuid::new(ShardId::new(7), 9),
            vec![(0, space_sim_wire_placeholder_payload())],
        )],
        input_ack: 2,
    });
    assert_eq!(MessageClass::of(&msg), MessageClass::Stream);
    conn.send(&msg).await.expect("send snapshot");
    assert_eq!(server_task.await.expect("server task"), msg);
}

#[tokio::test]
async fn control_stream_round_trips() {
    let (server, bound) = server();
    let redirect = Message::Redirect(Redirect {
        entity: EntityGuid::new(ShardId::new(7), 1),
        new_shard: ShardId::new(8),
        shard_map: shard_map(bound),
        activation_tick: 100,
    });
    let redirect_for_server = redirect.clone();
    let server_task = tokio::spawn(async move {
        let (conn, _hello) = accept_connection(server.accept()).await.expect("accept");
        conn.send_control(&Message::Welcome(welcome(bound)))
            .await
            .expect("welcome");
        // Server → client control message.
        conn.send_control(&redirect_for_server)
            .await
            .expect("send redirect");
        // Client → server control message.
        conn.recv_control().await.expect("recv").expect("a message")
    });

    let client = client_endpoint_on(addr()).expect("client endpoint");
    let (conn, _welcome) = connect(&client, bound, "space.local", &space_net::hello("player-1"))
        .await
        .expect("connect");
    let got = conn.recv_control().await.expect("recv").expect("a message");
    assert_eq!(got, redirect);
    let back = input_tick(7);
    conn.send_control(&back).await.expect("send control");
    assert_eq!(server_task.await.expect("server task"), back);
}

#[tokio::test]
async fn disconnect_is_observed() {
    let (server, bound) = server();
    let server_task = tokio::spawn(async move {
        let (conn, _hello) = accept_connection(server.accept()).await.expect("accept");
        conn.send_control(&Message::Welcome(welcome(bound)))
            .await
            .expect("welcome");
        // Both receive paths report the peer going away as `Ok(None)`.
        assert!(
            matches!(conn.recv_datagram().await, Ok(None)),
            "datagram receive must report a clean close"
        );
        conn
    });

    let client = client_endpoint_on(addr()).expect("client endpoint");
    let (conn, _welcome) = connect(&client, bound, "space.local", &space_net::hello("player-1"))
        .await
        .expect("connect");
    conn.close(0x20, "bye");
    let server_conn = server_task.await.expect("server task");
    assert!(server_conn.is_closed());
    assert!(server_conn.close_reason().is_some());
}

#[tokio::test]
async fn oversized_frame_rejected_before_alloc() {
    let (server, bound) = server();
    let server_task = tokio::spawn(async move {
        let (conn, _hello) = accept_connection(server.accept()).await.expect("accept");
        conn.send_control(&Message::Welcome(welcome(bound)))
            .await
            .expect("welcome");
        // A uni stream whose header declares an impossible payload length.
        conn.recv_stream().await
    });

    let client = client_endpoint_on(addr()).expect("client endpoint");
    let (conn, _welcome) = connect(&client, bound, "space.local", &space_net::hello("player-1"))
        .await
        .expect("connect");
    let mut stream = conn.connection().open_uni().await.expect("open uni");
    // Header: len = MAX_FRAME_SIZE + 1, then a plausible version.
    let mut bogus = Vec::new();
    bogus.extend_from_slice(&((MAX_FRAME_SIZE + 1) as u32).to_be_bytes());
    bogus.extend_from_slice(&PROTOCOL_VERSION.to_be_bytes());
    stream.write_all(&bogus).await.expect("write bogus");
    stream.finish().expect("finish");

    let result = server_task.await.expect("server task");
    let err = result.expect_err("an oversized declared length must be rejected");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

/// The `Redirect` shard map must be reachable from the client's view; this
/// asserts the map we hand out actually resolves a position (guards the test
/// fixtures the server tests rely on).
#[test]
fn shard_map_fixture_resolves_the_origin() {
    let map = shard_map("127.0.0.1:1".parse().expect("addr"));
    assert_eq!(
        map.lookup(&WorldPos::ZERO),
        Some(ShardId::new(7)),
        "the depth-0 region must cover the origin"
    );
}

/// A `Reject` reason survives the round trip (the client's `NetError` mapping
/// is checked in `version_mismatch_is_rejected`; this checks the wire value).
#[tokio::test]
async fn reject_reason_survives_the_wire() {
    let reason = RejectReason::Full;
    let frame = encode_frame(&Message::Reject(space_protocol::Reject {
        reason: reason.clone(),
    }));
    let decoded = space_protocol::decode_frame(&frame).expect("decode");
    match decoded {
        Message::Reject(r) => assert_eq!(r.reason, reason),
        other => panic!("expected Reject, got {other:?}"),
    }
}

// ---------------------------------------------------------------- keep-alive,
// handshake bounds, policy rejects and address families

/// The keep-alive has to fit inside the idle timeout with room for lost pings,
/// or a quiet link dies anyway.
#[test]
fn keep_alive_fits_inside_the_idle_timeout() {
    assert!(
        space_net::KEEP_ALIVE_INTERVAL * 3 <= space_net::IDLE_TIMEOUT,
        "three pings must fit in one idle timeout"
    );
}

/// A connection that carries no application traffic at all stays up past the
/// idle timeout, and still works afterwards.
///
/// This is the shard-to-orchestrator control link (epoch messages only) and
/// every shard-to-shard link (handoff messages only): both sat silent for longer
/// than the idle timeout as a matter of course, timed out, and were never
/// re-opened.
///
/// Mutation this catches: `keep_alive_interval(None)` in `transport_config` —
/// the client then sees the connection closed and the server's receive returns
/// `Ok(None)` instead of the datagram.
#[tokio::test]
async fn a_quiet_connection_outlives_the_idle_timeout() {
    let (server, bound) = server();
    let server_task = tokio::spawn(async move {
        let (conn, _hello) = accept_connection(server.accept()).await.expect("accept");
        conn.send_control(&Message::Welcome(welcome(bound)))
            .await
            .expect("welcome");
        // Says nothing and hears nothing until the client speaks again.
        conn.recv_datagram().await.expect("recv")
    });

    let client = client_endpoint_on(addr()).expect("client endpoint");
    let (conn, _welcome) = connect(&client, bound, "space.local", &space_net::hello("quiet"))
        .await
        .expect("connect");
    tokio::time::sleep(space_net::IDLE_TIMEOUT + std::time::Duration::from_millis(1_500)).await;
    assert!(
        !conn.is_closed(),
        "a quiet connection must not idle out: {:?}",
        conn.close_reason()
    );
    let msg = input_tick(9);
    conn.send(&msg)
        .await
        .expect("the link still carries traffic");
    assert_eq!(
        server_task.await.expect("server task"),
        Some(msg),
        "the server must still be listening on the same connection"
    );
}

/// A server whose transport answers but whose application never sends a
/// `Welcome` does not hold `connect` forever.
///
/// Mutation this catches: removing the `tokio::time::timeout` from
/// `connect_within` — with keep-alive on, nothing else ever ends the wait.
#[tokio::test]
async fn connect_gives_up_when_the_server_never_answers_the_hello() {
    let (server, bound) = server();
    let server_task = tokio::spawn(async move {
        // Complete the QUIC handshake, then sit on the connection.
        let conn = server
            .accept()
            .await
            .expect("incoming")
            .await
            .expect("handshake");
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        drop(conn);
    });

    let client = client_endpoint_on(addr()).expect("client endpoint");
    let started = std::time::Instant::now();
    let err = space_net::connect_within(
        &client,
        bound,
        "space.local",
        &space_net::hello("player-1"),
        std::time::Duration::from_millis(300),
    )
    .await
    .expect_err("a server that never welcomes must not be waited on forever");
    assert!(
        matches!(err, NetError::Handshake(_)),
        "expected a handshake timeout, got {err:?}"
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(3),
        "the bound must be the caller-chosen one, not the idle timeout: {:?}",
        started.elapsed()
    );
    server_task.abort();
}

/// A client that completes the QUIC handshake and never sends a hello does not
/// hold the accepting task forever.
///
/// Mutation this catches: removing the `tokio::time::timeout` from
/// `accept_incoming_within`.
#[tokio::test]
async fn accept_gives_up_on_a_client_that_never_sends_a_hello() {
    let (server, bound) = server();
    let server_task = tokio::spawn(async move {
        let incoming = server.accept().await.expect("incoming");
        let started = std::time::Instant::now();
        let result =
            space_net::accept_incoming_within(incoming, std::time::Duration::from_millis(300))
                .await;
        (result.map(|_| ()), started.elapsed())
    });

    // A raw client: transport handshake only, no control stream, no hello.
    let client = client_endpoint_on(addr()).expect("client endpoint");
    let conn = client
        .connect(bound, "space.local")
        .expect("connect")
        .await
        .expect("handshake");

    let (result, elapsed) = server_task.await.expect("server task");
    let err = result.expect_err("a silent client must be given up on");
    assert!(
        matches!(err, NetError::Handshake(_)),
        "expected a handshake timeout, got {err:?}"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(3),
        "the accept must give up on its own bound: {elapsed:?}"
    );
    drop(conn);
}

/// A refusal the server decides *after* the version gate (a full shard) reaches
/// the client with its reason.
///
/// Mutation this catches: replacing `Conn::reject` with "write the `Reject`,
/// drop the connection" — the close then overtakes the stream data and the
/// client sees a transport error instead of `Rejected(Full)`.
#[tokio::test]
async fn a_policy_reject_reaches_the_client_with_its_reason() {
    let (server, bound) = server();
    let server_task = tokio::spawn(async move {
        let (conn, _hello) = accept_connection(server.accept()).await.expect("accept");
        conn.reject(RejectReason::Full).await;
        // Dropped here, exactly as a serving task that returns would drop it.
    });

    let client = client_endpoint_on(addr()).expect("client endpoint");
    let err = connect(
        &client,
        bound,
        "space.local",
        &space_net::hello("player-101"),
    )
    .await
    .expect_err("a full server refuses");
    assert_eq!(
        err,
        NetError::Rejected(RejectReason::Full),
        "the client must learn why it was turned away"
    );
    server_task.await.expect("server task");
}

/// A join redirect naming shard 7, whose map sends players to
/// `client_endpoint`.
fn join_redirect(client_endpoint: SocketAddr) -> JoinRedirect {
    let mut map = shard_map("10.1.2.3:7871".parse().expect("addr"));
    map.regions[0] = map.regions[0].clone().with_client_endpoint(client_endpoint);
    JoinRedirect {
        shard: ShardId::new(7),
        shard_map: map,
    }
}

/// A server that is not the shard to play on answers a hello with a
/// `JoinRedirect` and closes; `join` hands the redirect back — whole, with
/// the map that names the client endpoint — and the server's close does not
/// have to run into its drain timeout, because the client closes as soon as
/// it has read the answer.
///
/// Mutation this catches: `join` treating the redirect as a protocol error
/// (no client could ever enter through the orchestrator), or `turn_away`
/// closing before the answer was read (the client sees a lost connection).
#[tokio::test]
async fn join_returns_the_redirect_a_server_turned_it_away_with() {
    let (server, bound) = server();
    let sent = join_redirect("0.0.0.0:7871".parse().expect("addr"));
    let to_send = sent.clone();
    let server_task = tokio::spawn(async move {
        let (conn, hello) = accept_connection(server.accept()).await.expect("accept");
        assert_eq!(hello.player_token, "player-1");
        let started = std::time::Instant::now();
        conn.turn_away(&Message::JoinRedirect(to_send)).await;
        (started.elapsed(), conn.is_closed())
    });

    let client = client_endpoint_on(addr()).expect("client endpoint");
    let joined = join(&client, bound, "space.local", &space_net::hello("player-1"))
        .await
        .expect("a redirect is an answer, not an error");
    let Joined::Elsewhere(got) = joined else {
        panic!("expected the redirect, got {joined:?}");
    };
    assert_eq!(got, sent);
    assert_eq!(
        got.shard_map.client_endpoint_of(got.shard),
        Some("0.0.0.0:7871".parse().expect("addr"))
    );
    let (took, closed) = server_task.await.expect("server task");
    assert!(closed, "the connection is finished once the answer is read");
    assert!(
        took < std::time::Duration::from_millis(900),
        "the client closes when it has read the redirect; the server does not wait out its \
         drain: {took:?}"
    );
}

/// `connect` is for a caller that wants the peer at the address itself — a
/// shard dialling a neighbour or its orchestrator. Sent on, it reports an
/// error that says what the address is, instead of a welcome it never got.
#[tokio::test]
async fn connect_reports_a_redirect_as_an_error_that_says_what_the_address_is() {
    let (server, bound) = server();
    let server_task = tokio::spawn(async move {
        let (conn, _hello) = accept_connection(server.accept()).await.expect("accept");
        conn.turn_away(&Message::JoinRedirect(join_redirect(
            "0.0.0.0:7871".parse().expect("addr"),
        )))
        .await;
    });
    let client = client_endpoint_on(addr()).expect("client endpoint");
    let err = connect(&client, bound, "space.local", &space_net::hello("shard:3"))
        .await
        .expect_err("not a welcome");
    let NetError::Protocol(why) = &err else {
        panic!("expected a protocol error, got {err:?}");
    };
    assert!(
        why.contains("player entry") && why.contains("shard 7"),
        "{why}"
    );
    server_task.await.expect("server task");
}

/// `join` takes a welcome like `connect` does, and a version mismatch is
/// still the readable refusal it always was — the redirect did not replace
/// either answer.
#[tokio::test]
async fn join_still_takes_a_welcome_and_a_version_mismatch_is_still_readable() {
    // A welcome.
    let (shard, bound) = server();
    let server_task = tokio::spawn(async move {
        let (conn, _hello) = accept_connection(shard.accept()).await.expect("accept");
        conn.send_control(&Message::Welcome(welcome(bound)))
            .await
            .expect("welcome");
        conn
    });
    let client = client_endpoint_on(addr()).expect("client endpoint");
    let joined = join(&client, bound, "space.local", &space_net::hello("player-1"))
        .await
        .expect("join");
    let Joined::Shard(conn, got) = joined else {
        panic!("expected a welcome, got {joined:?}");
    };
    assert_eq!(got.session, 42);
    assert!(!conn.is_closed());
    let _server_conn = server_task.await.expect("server task");

    // Another protocol version: refused at the gate, with both versions.
    let (gate, bound) = self::server();
    let server_task = tokio::spawn(async move {
        accept_connection(gate.accept())
            .await
            .expect_err("the gate refuses")
    });
    let client = client_endpoint_on(addr()).expect("client endpoint");
    let old = Hello {
        protocol: PROTOCOL_VERSION - 1,
        player_token: "player-1".into(),
    };
    let err = join(&client, bound, "space.local", &old)
        .await
        .expect_err("a version mismatch");
    assert_eq!(
        err,
        NetError::VersionMismatch {
            seen: PROTOCOL_VERSION - 1,
            expected: PROTOCOL_VERSION,
        }
    );
    assert!(err.to_string().contains("protocol version"), "{err}");
    assert!(matches!(
        server_task.await.expect("server task"),
        NetError::VersionMismatch { .. }
    ));
}

/// `client_endpoint_for` binds a socket of the family it is asked to dial.
///
/// The IPv6 half is skipped on a host without IPv6 (the bind fails there, which
/// is the right answer for an address that cannot be reached anyway).
#[tokio::test]
async fn client_endpoint_for_binds_the_family_it_dials() {
    let v4: SocketAddr = "127.0.0.1:7777".parse().expect("addr");
    let endpoint = space_net::client_endpoint_for(v4).expect("an IPv4 client endpoint");
    assert!(
        endpoint.local_addr().expect("local addr").is_ipv4(),
        "an IPv4 target needs an IPv4 socket"
    );

    let v6: SocketAddr = "[::1]:7777".parse().expect("addr");
    match space_net::client_endpoint_for(v6) {
        Ok(endpoint) => assert!(
            endpoint.local_addr().expect("local addr").is_ipv6(),
            "an IPv6 target needs an IPv6 socket"
        ),
        Err(e) => eprintln!("skipping the IPv6 half: no IPv6 on this host ({e})"),
    }
}

/// A server reached over IPv6 loopback through `client_endpoint_for`: the case
/// `client_endpoint()` cannot serve on Windows, where it binds IPv4 only.
#[tokio::test]
async fn an_ipv6_server_is_diallable_through_client_endpoint_for() {
    let (config, _cert) = space_net::crypto::make_server_config();
    let Ok(server) = server_endpoint("[::1]:0".parse().expect("addr"), config) else {
        eprintln!("skipping: no IPv6 loopback on this host");
        return;
    };
    let bound = server.local_addr().expect("local addr");
    let server_task = tokio::spawn(async move {
        let (conn, hello) = accept_connection(server.accept()).await.expect("accept");
        conn.send_control(&Message::Welcome(welcome(bound)))
            .await
            .expect("welcome");
        (conn, hello)
    });

    let Ok(client) = space_net::client_endpoint_for(bound) else {
        eprintln!("skipping: cannot bind an IPv6 client socket on this host");
        return;
    };
    let (conn, got) = connect(&client, bound, "space.local", &space_net::hello("v6"))
        .await
        .expect("connect over IPv6");
    assert_eq!(got.session, 42);
    let (_server_conn, hello) = server_task.await.expect("server task");
    assert_eq!(hello.player_token, "v6");
    assert!(!conn.is_closed());
}

/// A client endpoint whose connections grant the peer **no** flow-control
/// credit on a stream: whatever the server writes on the control stream waits
/// for a window that never opens. It keeps its connection alive like any
/// other (keep-alive pings), so the idle timeout does not end the wait.
fn windowless_client() -> quinn::Endpoint {
    let mut transport = quinn::TransportConfig::default();
    transport.stream_receive_window(quinn::VarInt::from_u32(0));
    transport.keep_alive_interval(Some(space_net::KEEP_ALIVE_INTERVAL));
    let mut config = space_net::crypto::make_client_config();
    config.transport_config(std::sync::Arc::new(transport));
    let mut endpoint = quinn::Endpoint::client(addr()).expect("client endpoint");
    endpoint.set_default_client_config(config);
    endpoint
}

/// The last word to a peer — a `Reject`, a `JoinRedirect` — is written with a
/// bound. A peer that says hello and grants no receive window cannot be
/// written to at all; without the bound `turn_away` waits on that write for as
/// long as the peer keeps its connection up, and with it whatever the caller
/// holds (one of the player port's 256 slots, a shard's serving task).
///
/// Mutation this catches: `send_last` awaiting `write_all` with no timeout —
/// the server task below then never finishes and the outer timeout fails the
/// test.
#[tokio::test]
async fn a_peer_that_grants_no_receive_window_cannot_hold_the_last_word() {
    let (server, bound) = server();
    let server_task = tokio::spawn(async move {
        let (conn, _hello) = accept_connection(server.accept()).await.expect("accept");
        let started = std::time::Instant::now();
        conn.turn_away(&Message::JoinRedirect(join_redirect(
            "0.0.0.0:7871".parse().expect("addr"),
        )))
        .await;
        (started.elapsed(), conn.is_closed())
    });

    let client = windowless_client();
    let conn = client
        .connect(bound, "space.local")
        .expect("dial")
        .await
        .expect("the transport handshake");
    let (mut send, _recv) = conn.open_bi().await.expect("the control stream");
    send.write_all(&encode_frame(&Message::Hello(space_net::hello("deaf"))))
        .await
        .expect("the hello");

    let (took, closed) = tokio::time::timeout(std::time::Duration::from_secs(8), server_task)
        .await
        .expect("turn_away must return although the peer never opens a window")
        .expect("server task");
    assert!(closed, "the connection is closed, read or not");
    assert!(
        took >= space_net::LAST_WORD_TIMEOUT,
        "the write really was blocked (else this test proves nothing): {took:?}"
    );
    assert!(
        took <= space_net::TURN_AWAY_LIMIT + std::time::Duration::from_millis(500),
        "bounded by the write's limit and the drain: {took:?}"
    );
    // The peer sees the connection end; it was not left hanging either.
    tokio::time::timeout(std::time::Duration::from_secs(3), conn.closed())
        .await
        .expect("the server closed the connection");
}

/// The first frame of a connection is a hello, and is read with a hello's
/// size limit: a header that declares more is refused at once, from the
/// header alone. Before, a stranger's first eight bytes had the server
/// allocate up to the general frame limit (1 MiB) and wait for the payload
/// until the handshake timeout.
///
/// Mutation this catches: reading the first frame with `read_frame` (the
/// general limit) — the accept below then waits for a payload that never
/// comes and ends as a `Handshake` timeout five seconds later.
#[tokio::test]
async fn a_first_frame_larger_than_a_hello_is_refused_from_its_header() {
    let (server, bound) = server();
    let server_task = tokio::spawn(async move {
        let started = std::time::Instant::now();
        let outcome = accept_connection(server.accept()).await.map(|_| ());
        (outcome, started.elapsed())
    });
    let client = client_endpoint_on(addr()).expect("client endpoint");
    let conn = client
        .connect(bound, "space.local")
        .expect("dial")
        .await
        .expect("the transport handshake");
    let (mut send, _recv) = conn.open_bi().await.expect("the control stream");
    // A header only: one byte more than a hello may be, far below the general
    // frame limit, in this build's protocol version. No payload follows.
    let declared = space_net::MAX_HELLO_PAYLOAD + 1;
    assert!(declared < MAX_FRAME_SIZE);
    let mut header = Vec::new();
    header.extend_from_slice(&(declared as u32).to_be_bytes());
    header.extend_from_slice(&PROTOCOL_VERSION.to_be_bytes());
    send.write_all(&header).await.expect("the header");

    let (outcome, took) = server_task.await.expect("server task");
    let err = outcome.expect_err("not a hello");
    assert!(
        matches!(err, NetError::Io(_)),
        "refused as a bad frame, not timed out as a missing hello: {err:?}"
    );
    assert!(
        took < std::time::Duration::from_secs(2),
        "refused from the header, without waiting for the payload: {took:?}"
    );
}

/// The largest hello anything legitimate sends still fits: a shard's hello to
/// a peer with the longest peer token the orchestrator accepts
/// (`MAX_SECRET_LEN`) behind the largest shard id. And a hello that cannot
/// fit is refused where it is built, with the reason, not sent to a server
/// that would close on it without a word.
#[tokio::test]
async fn the_largest_legitimate_hello_fits_and_a_larger_one_is_refused_by_the_sender() {
    let token = format!(
        "shard:{}:{}",
        u32::MAX,
        "t".repeat(space_net::MAX_SECRET_LEN)
    );
    let frame = encode_frame(&Message::Hello(space_net::hello(token.clone())));
    let payload = frame.len() - space_protocol::FRAME_HEADER_LEN;
    assert!(
        payload * 2 < space_net::MAX_HELLO_PAYLOAD,
        "the largest hello is {payload} bytes: it must fit with room to spare"
    );

    let (server, bound) = server();
    let server_task = tokio::spawn(async move {
        let (conn, hello) = accept_connection(server.accept()).await.expect("accept");
        conn.send_control(&Message::Welcome(welcome(bound)))
            .await
            .expect("welcome");
        (conn, hello)
    });
    let client = client_endpoint_on(addr()).expect("client endpoint");
    let (_conn, _welcome) = connect(
        &client,
        bound,
        "space.local",
        &space_net::hello(token.clone()),
    )
    .await
    .expect("the largest legitimate hello is taken");
    let (_server_conn, hello) = server_task.await.expect("server task");
    assert_eq!(hello.player_token, token);

    let too_long = space_net::hello("x".repeat(space_net::MAX_HELLO_PAYLOAD));
    let err = connect(&client, bound, "space.local", &too_long)
        .await
        .expect_err("a hello no server reads");
    assert!(
        matches!(&err, NetError::Protocol(why) if why.contains("hello")),
        "{err:?}"
    );
}
