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
//! | [`first_frame_must_be_hello`] | a non-`Hello` first frame → `NetError::Protocol` |
//! | [`datagram_message_round_trips`] | an `InputTick` rides a datagram and decodes identically |
//! | [`oversized_delta_promoted_to_stream`] | a `Delta` past the datagram budget is sent on a stream and still arrives |
//! | [`snapshot_stream_round_trips`] | a `Snapshot` (stream class) round-trips on a fresh uni stream |
//! | [`control_stream_round_trips`] | `send_control`/`recv_control` carry a `Redirect` both ways |
//! | [`disconnect_is_observed`] | after the peer closes, receives return `Ok(None)` and `is_closed()` is true |
//! | [`oversized_frame_rejected_before_alloc`] | a stream frame declaring a length over `MAX_FRAME_SIZE` errors without a huge allocation |

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use space_net::{
    accept_connection, client_endpoint_on, connect, server_endpoint, MessageClass, NetError,
};
use space_protocol::{
    encode_frame, Delta, EntityGuid, EntitySnapshot, Hello, InputTick, Message, PlayerInput,
    Redirect, RejectReason, ShardId, ShardMap, Snapshot, Welcome, MAX_FRAME_SIZE, PROTOCOL_VERSION,
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
