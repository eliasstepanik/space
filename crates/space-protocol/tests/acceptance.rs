//! Acceptance tests for `space-protocol` (Phase 1, task 1.3).
//!
//! Named after the brief's requirements: `roundtrip_all`,
//! `version_mismatch_rejected`, `oversized_rejected`, `truncated_rejected`,
//! `unknown_tag_rejected`, `shard_map_lookup`, `guid_stable_across_handoff`,
//! `snapshot_budget`, `property_random_messages` and `delta_size_sanity`.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use space_protocol::{
    check_hello, decode_frame, encode_frame, entity_snapshot, try_encode_frame, AdminCommand,
    AdminReply, CodecError, Delta, DirectoryQuery, DirectoryReply, EntityGuid, EntitySnapshot,
    GhostUpdate, HandoffAbort, HandoffAccept, HandoffCommit, HandoffOffer, HandoffReason, Hello,
    InputTick, IslandKey, JoinRedirect, LoadReport, Message, NeighbourStatus, PlayerInput,
    Redirect, Reject, RejectReason, Scene, SceneAck, SceneClock, SceneId, SceneState, ShardHello,
    ShardId, ShardMap, ShardRegion, ShardWelcome, Snapshot, SnapshotBuilder, StaleEpoch, Welcome,
    FRAME_HEADER_LEN, MAX_FRAME_SIZE, PROTOCOL_VERSION,
};
use space_world::{NodeRef, WorldPos};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn addr(n: u8) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, n)), 7_000 + u16::from(n))
}

fn shard(n: u32) -> ShardId {
    ShardId::new(n)
}

fn guid(shard_no: u32, serial: u64) -> EntityGuid {
    EntityGuid::new(shard(shard_no), serial)
}

/// A small replicated component (3 × f32), the shape the size test budgets for.
#[derive(Debug, PartialEq, Clone, Copy, serde::Serialize, serde::Deserialize)]
struct Position {
    x: f32,
    y: f32,
    z: f32,
}

fn entity(shard_no: u32, serial: u64, components: usize) -> EntitySnapshot {
    let parts: Vec<(u16, Position)> = (0..components as u16)
        .map(|i| {
            (
                i,
                Position {
                    x: f32::from(i),
                    y: 0.0,
                    z: 0.0,
                },
            )
        })
        .collect();
    let refs: Vec<(u16, &Position)> = parts.iter().map(|(id, p)| (*id, p)).collect();
    entity_snapshot(guid(shard_no, serial), &refs)
}

fn region(depth: u8, index: [i64; 3], owner: u32) -> ShardRegion {
    ShardRegion::new(NodeRef::new(depth, index), shard(owner), addr(owner as u8))
}

fn sample_map() -> ShardMap {
    ShardMap::new(
        7,
        vec![
            region(8, [0, 0, 0], 1),
            region(12, [0, 0, 0], 2),
            region(16, [1, 0, 0], 3),
        ],
    )
}

fn offer() -> HandoffOffer {
    HandoffOffer {
        entity: guid(1, 42),
        from: shard(1),
        to: shard(2),
        epoch: 7,
        state: Box::new(entity(1, 42, 3)),
        activation_tick: 1_000,
    }
}

/// Every message variant, in tag order.
fn all_messages() -> Vec<Message> {
    vec![
        Message::Hello(Hello {
            protocol: PROTOCOL_VERSION,
            player_token: "player-7".into(),
        }),
        Message::Welcome(Welcome {
            session: 0xdead_beef,
            shard: shard(3),
            shard_map: sample_map(),
        }),
        Message::Reject(Reject {
            reason: RejectReason::VersionMismatch {
                seen: 0,
                expected: PROTOCOL_VERSION,
            },
        }),
        Message::Reject(Reject {
            reason: RejectReason::Other("nope".into()),
        }),
        Message::Reject(Reject {
            reason: RejectReason::Starting,
        }),
        Message::InputTick(InputTick {
            tick: 12,
            seq: 34,
            inputs: vec![
                PlayerInput {
                    entity: guid(1, 5),
                    thrust: [0.0, 1.5, -2.25],
                    torque: [0.0, 0.0, 0.125],
                    tick: 12,
                    fast: true,
                },
                PlayerInput {
                    entity: guid(2, 6),
                    thrust: [1.0, 0.0, 0.0],
                    torque: [0.0, 0.0, 0.0],
                    tick: 12,
                    fast: false,
                },
            ],
            ack_tick: 11,
        }),
        Message::Snapshot(Snapshot {
            tick: 99,
            shard: shard(1),
            epoch: 7,
            entities: (0..5).map(|i| entity(1, i, 3)).collect(),
            input_ack: 98,
            more: false,
        }),
        Message::Delta(Delta {
            tick: 100,
            base_tick: 99,
            shard: shard(1),
            epoch: 7,
            entities: (0..3).map(|i| entity(1, i, 2)).collect(),
            removed: vec![guid(1, 900), guid(1, 901)],
            input_ack: 99,
        }),
        Message::HandoffOffer(offer()),
        Message::HandoffAccept(HandoffAccept {
            entity: guid(1, 42),
            from: shard(1),
            to: shard(2),
            epoch: 7,
            activation_tick: 1_000,
        }),
        Message::HandoffCommit(HandoffCommit {
            entity: guid(1, 42),
            from: shard(1),
            to: shard(2),
            epoch: 7,
            activation_tick: 1_000,
            state: Box::new(entity(1, 42, 3)),
        }),
        Message::HandoffAbort(HandoffAbort {
            entity: guid(1, 42),
            from: shard(1),
            to: shard(2),
            reason: HandoffReason::TargetFull,
        }),
        Message::HandoffAbort(HandoffAbort {
            entity: guid(1, 42),
            from: shard(1),
            to: shard(2),
            reason: HandoffReason::ClientNotConnected,
        }),
        Message::GhostUpdate(GhostUpdate {
            from: shard(4),
            epoch: 3,
            tick: 1_234,
            full: true,
            entities: vec![entity(4, 77, 4), entity(4, 78, 2)],
            removed: vec![guid(4, 12)],
            islands: vec![IslandKey {
                body: guid(4, 77),
                min: guid(1, 3),
                home: shard(1),
            }],
        }),
        Message::Redirect(Redirect {
            entity: guid(2, 9),
            new_shard: shard(2),
            shard_map: sample_map(),
            activation_tick: 2_000,
        }),
        Message::StaleEpoch(StaleEpoch {
            seen: 3,
            current: 7,
        }),
        Message::EpochPrepare(space_protocol::EpochPrepare {
            epoch: 8,
            activation_tick: 2_000,
            changes: vec![(NodeRef::new(16, [1, 0, 0]), shard(4))],
        }),
        Message::EpochAck(space_protocol::EpochAck {
            epoch: 8,
            shard: shard(4),
        }),
        Message::EpochCommit(space_protocol::EpochCommit { epoch: 8 }),
        Message::EpochAbort(space_protocol::EpochAbort {
            epoch: 8,
            reason: space_protocol::AbortReason::Timeout,
        }),
        Message::LoadReport(LoadReport {
            shard: shard(1),
            players: 87,
            tick_p95_ms: 21.5,
            load: 0.645_7,
        }),
        Message::DirectoryQuery(DirectoryQuery {}),
        Message::DirectoryReply(DirectoryReply {
            shard_map: sample_map(),
            tick: 12_345,
        }),
        Message::AdminCommand(AdminCommand::ForceSplit(shard(1))),
        Message::AdminCommand(AdminCommand::ForceMerge(shard(2))),
        Message::AdminCommand(AdminCommand::KillShard(shard(3))),
        Message::AdminCommand(AdminCommand::SpawnBots { count: 250 }),
        Message::AdminCommand(AdminCommand::Status),
        Message::AdminReply(AdminReply::Accepted {
            command: AdminCommand::Status,
        }),
        Message::AdminReply(AdminReply::Refused {
            command: AdminCommand::ForceSplit(shard(1)),
            reason: "shard is draining".into(),
        }),
        Message::AdminReply(AdminReply::Status {
            lines: vec!["shard 1 players=87 p95=21.5ms load=0.65".into()],
        }),
        Message::NeighbourStatus(NeighbourStatus {
            entity: guid(2, 9),
            shard: shard(2),
            connected: true,
        }),
        Message::ShardHello(ShardHello {
            cluster: "cluster-a".into(),
            shard: ShardId::NONE,
            endpoint: addr(9),
            region: Some(region(40, [0, 0, 0], 1)),
            client_endpoint: Some(any_host(7_009)),
            secret: "cluster-secret".into(),
        }),
        Message::ShardWelcome(ShardWelcome {
            shard: shard(5),
            region: region(40, [0, 0, 0], 5),
            shard_map: sample_map(),
            peer_token: "peer-secret".into(),
            tick_origin_us: 1_763_900_000_000_000,
        }),
        Message::SceneSet(SceneState::new(Scene::Galaxy.id(), 4)),
        Message::SceneAck(SceneAck {
            shard: shard(5),
            applied: SceneState::new(Scene::Galaxy.id(), 4),
        }),
        Message::SceneUpdate(SceneState::new(Scene::Materials.id(), 9)),
        Message::SceneClock(SceneClock::new(1234, 1_763_900_000_123_456)),
        join_redirect_message(),
        Message::FarInterest(space_protocol::FarInterest {
            from: shard(4),
            tick: 77,
            players: vec![space_protocol::FarPlayer {
                cell: [1, -2, 3],
                local: [0.5, -100.0, 511.0],
            }],
            missing: vec![guid(4, 9)],
        }),
        last_message(),
    ]
}

/// The last message in tag order (`FarUpdate`, tag 31).
fn last_message() -> Message {
    Message::FarUpdate(space_protocol::FarUpdate {
        from: shard(2),
        tick: 31,
        new: vec![entity(2, 5, 3)],
        changed: vec![entity(2, 6, 1)],
        removed: vec![guid(2, 7)],
        refresh: true,
        held: vec![guid(2, 5), guid(2, 6)],
        more: false,
    })
}

/// `JoinRedirect` (tag 29): the map names the shard to dial with a client
/// endpoint of its own.
fn join_redirect_message() -> Message {
    let mut map = sample_map();
    map.regions[1] = map.regions[1].clone().with_client_endpoint(any_host(7_002));
    Message::JoinRedirect(JoinRedirect {
        shard: shard(2),
        shard_map: map,
    })
}

/// `0.0.0.0:port`: a client endpoint that means "the host you dialed".
fn any_host(port: u16) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), port)
}

// ---------------------------------------------------------------------------
// roundtrip
// ---------------------------------------------------------------------------

#[test]
fn roundtrip_all() {
    let messages = all_messages();
    // Every declared variant of `Message` is exercised; the count is asserted so
    // a new variant that is not added here fails the test.
    assert_eq!(messages.len(), 41);
    for message in &messages {
        let frame = encode_frame(message);
        assert_eq!(
            frame.len(),
            FRAME_HEADER_LEN + payload_of(&frame).len(),
            "frame layout for {message:?}"
        );
        let decoded = decode_frame(&frame).expect("decode");
        assert_eq!(&decoded, message, "roundtrip mismatch");
    }
}

/// The payload of a frame (everything after the 8-byte header).
fn payload_of(frame: &[u8]) -> &[u8] {
    &frame[FRAME_HEADER_LEN..]
}

#[test]
fn header_is_length_then_version() {
    let frame = encode_frame(&Message::StaleEpoch(StaleEpoch {
        seen: 1,
        current: 2,
    }));
    let len = u32::from_be_bytes([frame[0], frame[1], frame[2], frame[3]]) as usize;
    let version = u32::from_be_bytes([frame[4], frame[5], frame[6], frame[7]]);
    assert_eq!(version, PROTOCOL_VERSION);
    assert_eq!(len, frame.len() - FRAME_HEADER_LEN);
}

// ---------------------------------------------------------------------------
// Codec rejections
// ---------------------------------------------------------------------------

#[test]
fn version_mismatch_rejected() {
    let mut frame = encode_frame(&Message::StaleEpoch(StaleEpoch {
        seen: 1,
        current: 2,
    }));
    // Rewrite the header's version field.
    frame[4..8].copy_from_slice(&(PROTOCOL_VERSION + 1).to_be_bytes());
    assert_eq!(
        decode_frame(&frame),
        Err(CodecError::VersionMismatch {
            seen: PROTOCOL_VERSION + 1,
            expected: PROTOCOL_VERSION,
        })
    );

    // A Hello carrying the wrong protocol version is refused at the handshake
    // level, not just at the frame level.
    let hello = Message::Hello(Hello {
        protocol: PROTOCOL_VERSION - 1,
        player_token: "t".into(),
    });
    let decoded = decode_frame(&encode_frame(&hello)).expect("hello decodes");
    match decoded {
        Message::Hello(h) => assert_eq!(h.protocol, PROTOCOL_VERSION - 1),
        other => panic!("expected Hello, got {other:?}"),
    }
    // The server's answer to that Hello:
    let reject = Message::Reject(Reject {
        reason: RejectReason::VersionMismatch {
            seen: PROTOCOL_VERSION - 1,
            expected: PROTOCOL_VERSION,
        },
    });
    assert_eq!(decode_frame(&encode_frame(&reject)), Ok(reject));
}

/// B7: `check_hello` is the handshake-level version gate. The frame header
/// already carries the version (checked by `decode_frame`), but the version also
/// travels in `Hello::protocol`; a server calls `check_hello` on the decoded
/// `Hello` and answers a mismatch with `RejectReason::VersionMismatch` so the
/// client learns *why* it was turned away.
#[test]
fn check_hello_gates_the_handshake() {
    // A matching version is accepted.
    let ok = Hello {
        protocol: PROTOCOL_VERSION,
        player_token: "token".into(),
    };
    assert_eq!(check_hello(&ok), Ok(()));

    // A version one below and one above are both rejected with the reason that
    // names both sides.
    for seen in [PROTOCOL_VERSION - 1, PROTOCOL_VERSION + 1] {
        let bad = Hello {
            protocol: seen,
            player_token: "token".into(),
        };
        assert_eq!(
            check_hello(&bad),
            Err(RejectReason::VersionMismatch {
                seen,
                expected: PROTOCOL_VERSION,
            })
        );
    }
}

#[test]
fn oversized_rejected() {
    // A header declaring more than MAX_FRAME_SIZE is refused without decoding.
    let mut frame = Vec::new();
    frame.extend_from_slice(&(MAX_FRAME_SIZE as u32 + 1).to_be_bytes());
    frame.extend_from_slice(&PROTOCOL_VERSION.to_be_bytes());
    frame.extend_from_slice(&[0u8; 16]);
    assert_eq!(
        decode_frame(&frame),
        Err(CodecError::Oversized {
            declared: MAX_FRAME_SIZE + 1
        })
    );

    // And the encoder refuses to build an oversized frame in the first place.
    let huge = Message::Snapshot(Snapshot {
        tick: 1,
        shard: shard(1),
        epoch: 1,
        entities: (0..200_000u64).map(|i| entity(1, i, 3)).collect(),
        input_ack: 0,
        more: false,
    });
    let err = try_encode_frame(&huge).expect_err("must not encode an oversized frame");
    assert!(matches!(err, CodecError::Oversized { .. }), "{err:?}");
    assert!(err.to_string().contains("limit"));
}

#[test]
fn truncated_rejected() {
    let frame = encode_frame(&Message::Snapshot(Snapshot {
        tick: 1,
        shard: shard(1),
        epoch: 1,
        entities: (0..4).map(|i| entity(1, i, 3)).collect(),
        input_ack: 0,
        more: false,
    }));

    // Shorter than the header.
    assert_eq!(decode_frame(&frame[..4]), Err(CodecError::Truncated));
    assert_eq!(decode_frame(&[]), Err(CodecError::Truncated));
    // Header intact, payload cut short.
    assert_eq!(decode_frame(&frame[..12]), Err(CodecError::Truncated));
    for cut in [FRAME_HEADER_LEN + 1, frame.len() - 1] {
        assert_eq!(decode_frame(&frame[..cut]), Err(CodecError::Truncated));
    }
    // Truncation inside a component payload must also be caught.
    let one = encode_frame(&Message::GhostUpdate(GhostUpdate {
        from: shard(1),
        epoch: 1,
        tick: 9,
        full: false,
        entities: vec![entity(1, 1, 3)],
        removed: Vec::new(),
        islands: Vec::new(),
    }));
    assert_eq!(
        decode_frame(&one[..one.len() - 5]),
        Err(CodecError::Truncated)
    );
    // Trailing bytes are not silently ignored either.
    let mut extra = frame.clone();
    extra.push(0);
    assert!(matches!(
        decode_frame(&extra),
        Err(CodecError::InvalidPayload(_))
    ));
}

#[test]
fn unknown_tag_rejected() {
    // `Message::VARIANT_COUNT` is 32, so the valid wire tags are `0..=31`.
    // The boundary that matters is tag 32 — the first tag past the last variant.
    // (An earlier revision tested the literals 28/29 while the count was 25:
    // unknown only incidentally, and "passing" for ever as variants were
    // appended — which is why the bound below is the count itself.)
    assert_eq!(Message::VARIANT_COUNT, 32);
    let first_invalid = Message::VARIANT_COUNT; // 32
    let mut payload = first_invalid.to_be_bytes().to_vec();
    payload.extend_from_slice(&[0xAA; 8]);
    let mut frame = Vec::new();
    frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    frame.extend_from_slice(&PROTOCOL_VERSION.to_be_bytes());
    frame.extend_from_slice(&payload);
    assert_eq!(
        decode_frame(&frame),
        Err(CodecError::UnknownTag { tag: first_invalid })
    );

    // The last valid tag (31, `FarUpdate`) must decode, proving 32 is the
    // first rejection and not an off-by-one that also drops the last variant.
    let last_valid = last_message();
    assert_eq!(last_valid.wire_tag(), Message::VARIANT_COUNT - 1);
    assert_eq!(
        decode_frame(&encode_frame(&last_valid)),
        Ok(last_valid),
        "the last valid tag must decode"
    );

    // Tags well past the boundary (and the maximum u32) are also rejected.
    for tag in [first_invalid + 1, first_invalid + 6, u32::MAX] {
        let mut payload = Vec::new();
        payload.extend_from_slice(&tag.to_be_bytes());
        let mut frame = Vec::new();
        frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        frame.extend_from_slice(&PROTOCOL_VERSION.to_be_bytes());
        frame.extend_from_slice(&payload);
        assert_eq!(decode_frame(&frame), Err(CodecError::UnknownTag { tag }));
    }
}

// ---------------------------------------------------------------------------
// Shard map lookup
// ---------------------------------------------------------------------------

#[test]
fn shard_map_lookup() {
    // Nested regions: a depth-16 node inside a depth-12 node inside a depth-8
    // node. The deepest match must win, because the nested node is the more
    // specific ownership statement (that is how a split overrides its parent).
    let map = ShardMap::new(
        7,
        vec![
            region(8, [0, 0, 0], 1),
            region(12, [0, 0, 0], 2),
            region(16, [0, 0, 0], 3),
        ],
    );

    // A position inside the depth-16 node: owner 3, not 2 or 1.
    let deep = WorldPos::new([1, 0, 0], [0.0, 0.0, 0.0]);
    assert_eq!(map.lookup(&deep), Some(shard(3)));

    // A position inside the depth-12 node but outside the depth-16 node.
    // Depth 12 spans 2^52 cells per axis, so index [0,0,0] covers cells
    // [0, 2^52); the depth-16 child covers [0, 2^48). Cell 2^48 is inside the
    // depth-12 node but outside the depth-16 one.
    let mid = WorldPos::new([1i64 << 48, 0, 0], [0.0, 0.0, 0.0]);
    assert_eq!(map.lookup(&mid), Some(shard(2)));

    // A position inside the depth-8 node only.
    let outer = WorldPos::new([1i64 << 52, 0, 0], [0.0, 0.0, 0.0]);
    assert_eq!(map.lookup(&outer), Some(shard(1)));

    // A position outside every region.
    let outside = WorldPos::new([1i64 << 60, 0, 0], [0.0, 0.0, 0.0]);
    assert_eq!(map.lookup(&outside), None);

    // Order in `regions` must not change the answer.
    let mut shuffled = map.clone();
    shuffled.regions.reverse();
    assert_eq!(shuffled.lookup(&deep), Some(shard(3)));
    assert_eq!(shuffled.lookup(&mid), Some(shard(2)));
    assert_eq!(shuffled.lookup(&outer), Some(shard(1)));

    // `path` lists the covering regions shallowest first.
    let path = map.path(&deep);
    assert_eq!(path.len(), 3);
    assert_eq!(path[0].node.depth, 8);
    assert_eq!(path[2].node.depth, 16);

    // An empty map resolves nothing.
    assert_eq!(ShardMap::new(0, Vec::new()).lookup(&deep), None);
}

// ---------------------------------------------------------------------------
// Guid stability
// ---------------------------------------------------------------------------

#[test]
fn guid_stable_across_handoff() {
    let original = guid(1, 0x1234_5678_9abc_def0);

    // Serialise on shard 1 inside a handoff offer...
    let frame = encode_frame(&Message::HandoffOffer(HandoffOffer {
        entity: original,
        from: shard(1),
        to: shard(2),
        epoch: 7,
        state: Box::new(entity(1, original.serial, 3)),
        activation_tick: 1_000,
    }));

    // ...deserialise on shard 2: the guid is byte-identical, not aliased into
    // shard 2's own serial space.
    let decoded = decode_frame(&frame).expect("offer decodes");
    let Message::HandoffOffer(offer) = decoded else {
        panic!("expected HandoffOffer")
    };
    assert_eq!(offer.entity, original);
    assert_eq!(
        offer.entity.shard,
        shard(1),
        "minting shard is never rewritten"
    );
    assert_eq!(offer.entity.serial, original.serial);
    assert_eq!(
        offer.state.guid, original,
        "the payload keeps the same guid"
    );

    // N12: the guid-uniqueness property the monotonic/never-reused ShardId rule
    // protects (see the `ShardId` doc). Two entities with the same serial are
    // distinct only because their minting shard ids differ — so a recycled shard
    // id would alias them. `NONE` is the reserved "no shard" sentinel and is a
    // real u32, which is exactly why ids must never be reused to reach it.
    assert_eq!(
        guid(1, 7),
        guid(1, 7),
        "same (shard, serial) is the same guid"
    );
    assert_ne!(
        guid(1, 7),
        guid(2, 7),
        "the same serial under different shard ids must stay distinct"
    );
    assert_eq!(ShardId::NONE, ShardId(u32::MAX));
    // The same guid survives a snapshot round trip on the receiving shard too.
    let snap = Message::Snapshot(Snapshot {
        tick: 1,
        shard: shard(2),
        epoch: 7,
        entities: vec![entity(1, original.serial, 2)],
        input_ack: 0,
        more: false,
    });
    let Message::Snapshot(back) = decode_frame(&encode_frame(&snap)).expect("snapshot decodes")
    else {
        panic!("expected Snapshot")
    };
    assert_eq!(back.entities[0].guid, original);
    assert_eq!(
        back.shard,
        shard(2),
        "the producing shard is separate metadata"
    );

    // Two shards minting the same serial produce different guids.
    assert_ne!(guid(1, 7), guid(2, 7));
    // And a guid is hashable/copyable as an identity key.
    let mut set = std::collections::HashSet::new();
    set.insert(guid(1, 7));
    set.insert(guid(2, 7));
    set.insert(guid(1, 7));
    assert_eq!(set.len(), 2);
}

// ---------------------------------------------------------------------------
// Byte budget
// ---------------------------------------------------------------------------

/// Payload size of a message as it would go on the wire.
fn payload_len(message: &Message) -> usize {
    try_encode_frame(message)
        .expect("fits")
        .len()
        .saturating_sub(FRAME_HEADER_LEN)
}

#[test]
fn snapshot_budget() {
    const BUDGET: usize = 64 * 1024;
    // 5 000 entities of 3 small components: several times the budget, so the
    // test measures real truncation rather than a corpus that trivially fits.
    let entities: Vec<EntitySnapshot> = (0..5_000u64).map(|i| entity(1, i, 3)).collect();

    let mut builder = SnapshotBuilder::with_budget(BUDGET, 1, shard(1), 7);
    let mut batches: Vec<Snapshot> = Vec::new();
    let mut next = 0usize;
    while next < entities.len() {
        if builder.push(&entities[next]) {
            next += 1;
            continue;
        }
        batches.push(builder.finish_snapshot(0));
    }
    batches.push(builder.finish_snapshot(0));

    // Sanity: the corpus is far bigger than one budget, so the test measures
    // real truncation rather than a corpus that trivially fits.
    let total_payload: usize = batches
        .iter()
        .map(|b| payload_len(&Message::Snapshot(b.clone())))
        .sum();
    assert!(
        total_payload > 4 * BUDGET,
        "corpus is only {total_payload} bytes"
    );
    assert!(batches.len() > 4, "the corpus must be split many times");
    let total_entities: usize = batches.iter().map(|b| b.entities.len()).sum();
    assert_eq!(total_entities, entities.len(), "nothing may be dropped");

    for batch in &batches {
        let size = payload_len(&Message::Snapshot(batch.clone()));
        assert!(
            size <= BUDGET,
            "batch of {} entities encodes to {size} bytes, budget is {BUDGET}",
            batch.entities.len()
        );
        // With enough entities available a batch must fill the budget: never
        // below half of it. (Only the last batch is allowed to be short.)
        if batch.entities.len() < entities.len() {
            assert!(
                size >= BUDGET / 2,
                "batch of {} entities only used {size} of {BUDGET} bytes",
                batch.entities.len()
            );
        }
    }
}

#[test]
fn snapshot_budget_edge_cases() {
    // A single entity larger than the budget is still emitted (otherwise it
    // could never be sent), and `from_entities` reports how many it took.
    let fat = entity(1, 0, 1);
    let mut builder = SnapshotBuilder::with_budget(1, 1, shard(1), 7);
    assert!(builder.push(&fat), "the first entity is always accepted");
    assert!(!builder.push(&entity(1, 1, 1)));
    let snap = builder.finish_snapshot(0);
    assert_eq!(snap.entities.len(), 1);
    assert!(builder.is_empty());

    let (snap, taken) = SnapshotBuilder::from_entities(
        1,
        shard(1),
        7,
        64 * 1024,
        &(0..10_000u64).map(|i| entity(1, i, 3)).collect::<Vec<_>>(),
    );
    assert!(taken > 0 && taken < 10_000);
    assert_eq!(snap.entities.len(), taken);
    assert!(payload_len(&Message::Snapshot(snap)) <= 64 * 1024);

    // A budget above the codec's frame limit is clamped to it.
    let clamped = SnapshotBuilder::with_budget(usize::MAX, 1, shard(1), 7);
    assert_eq!(clamped.budget(), MAX_FRAME_SIZE);
}

/// N10: the `removed` list a `finish_delta` carries is charged against the byte
/// budget. Without reserving room for it, a delta whose entities exactly filled
/// the budget would overrun it once the `removed` guids were appended.
#[test]
fn removed_list_is_charged_to_the_budget() {
    const BUDGET: usize = 4096;
    // A pile of removals the replication pass knows it must report.
    let removed: Vec<EntityGuid> = (0..40u64).map(|i| guid(1, i)).collect();

    // Reserve room for them, then fill the builder to the (reduced) budget.
    let mut builder = SnapshotBuilder::with_budget(BUDGET, 10, shard(1), 7);
    builder.reserve_removed(removed.len());
    let mut pushed = 0usize;
    for i in 0..10_000u64 {
        if builder.push(&entity(1, i, 3)) {
            pushed += 1;
        } else {
            break;
        }
    }
    assert!(pushed > 0, "must fit at least one entity");
    let delta = builder.finish_delta(9, removed.clone(), 0);
    assert_eq!(delta.removed.len(), removed.len());
    let size = payload_len(&Message::Delta(delta));
    assert!(
        size <= BUDGET,
        "delta with {} entities + {} removed encodes to {size}, budget {BUDGET}",
        pushed,
        removed.len()
    );

    // Contrast: without the reserve, the same greedy fill overruns the budget
    // once the removals are added — proving the reserve is what keeps it in
    // bounds (the property the reviewer asked to be tested at the boundary).
    let mut unreserved = SnapshotBuilder::with_budget(BUDGET, 10, shard(1), 7);
    let mut pushed2 = 0usize;
    for i in 0..10_000u64 {
        if unreserved.push(&entity(1, i, 3)) {
            pushed2 += 1;
        } else {
            break;
        }
    }
    let delta2 = unreserved.finish_delta(9, removed.clone(), 0);
    let size2 = payload_len(&Message::Delta(delta2));
    assert!(
        pushed2 >= pushed,
        "without a reserve the builder fits at least as many entities ({pushed2} vs {pushed})"
    );
    assert!(
        size2 > BUDGET,
        "without reserving the removed list the delta overruns the budget \
         (got {size2}, budget {BUDGET}); the reserve is load-bearing"
    );
}

// ---------------------------------------------------------------------------
// Property: 1000 random messages roundtrip
// ---------------------------------------------------------------------------

/// Tiny deterministic RNG (the same mixer `space-world` uses for generation).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        space_world::splitmix64(self.0)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
    fn u32_below(&mut self, n: u32) -> u32 {
        self.below(u64::from(n)) as u32
    }
    /// A finite f32 in `[-1000, 1000]` with 1e-3 resolution (never NaN, so
    /// `PartialEq` after a round trip is meaningful).
    fn f32(&mut self) -> f32 {
        self.below(2_000_001) as f32 / 1_000.0 - 1_000.0
    }
    fn vec3(&mut self) -> [f32; 3] {
        [self.f32(), self.f32(), self.f32()]
    }
    fn string(&mut self) -> String {
        let n = self.below(24);
        (0..n)
            .map(|_| b'a' + self.below(26) as u8)
            .map(char::from)
            .collect()
    }
    fn addr(&mut self) -> SocketAddr {
        SocketAddr::new(
            IpAddr::V4(Ipv4Addr::new(
                self.u32_below(256) as u8,
                self.u32_below(256) as u8,
                self.u32_below(256) as u8,
                self.u32_below(256) as u8,
            )),
            self.u32_below(65_535) as u16,
        )
    }
    fn node(&mut self) -> NodeRef {
        let depth = self.below(u64::from(space_world::MAX_DEPTH) + 1) as u8;
        if depth == 0 {
            return NodeRef::new(0, [0, 0, 0]);
        }
        // A valid node at depth d has each axis index in [-2^(d-1), 2^(d-1)).
        // Keep the magnitude small so the value fits comfortably in i64 for any
        // depth the generator picks.
        let shift = (u32::from(depth) - 1).min(20);
        let half = 1i64 << shift;
        let mut index = [0i64; 3];
        for axis in &mut index {
            let v = (self.next_u64() as i64) % half;
            *axis = if self.next_u64() & 1 == 0 { v } else { -v };
        }
        NodeRef::new(depth, index)
    }
    fn shard(&mut self) -> ShardId {
        ShardId::new(self.u32_below(4_096))
    }
    fn guid(&mut self) -> EntityGuid {
        EntityGuid::new(self.shard(), self.next_u64())
    }
    fn components(&mut self) -> Vec<(u16, Vec<u8>)> {
        let n = self.below(6);
        (0..n)
            .map(|_| {
                let len = self.below(48);
                let payload = (0..len).map(|_| self.next_u64() as u8).collect();
                (self.u32_below(4096) as u16, payload)
            })
            .collect()
    }
    fn entity(&mut self) -> EntitySnapshot {
        EntitySnapshot::new(self.guid(), self.components())
    }
    fn entities(&mut self) -> Vec<EntitySnapshot> {
        (0..self.below(12)).map(|_| self.entity()).collect()
    }
    fn map(&mut self) -> ShardMap {
        let regions = (0..self.below(6)).map(|_| self.region()).collect();
        ShardMap::new(self.next_u64(), regions)
    }
    /// A region: half of them with a client endpoint of their own.
    fn region(&mut self) -> ShardRegion {
        let region = ShardRegion::new(self.node(), self.shard(), self.addr());
        if self.next_u64() & 1 == 0 {
            region
        } else {
            region.with_client_endpoint(self.addr())
        }
    }
    fn message(&mut self) -> Message {
        match self.below(32) {
            0 => Message::Hello(Hello {
                protocol: self.u32_below(4),
                player_token: self.string(),
            }),
            1 => Message::Welcome(Welcome {
                session: self.next_u64(),
                shard: self.shard(),
                shard_map: self.map(),
            }),
            2 => Message::Reject(Reject {
                reason: match self.below(6) {
                    0 => RejectReason::VersionMismatch {
                        seen: self.u32_below(8),
                        expected: PROTOCOL_VERSION,
                    },
                    1 => RejectReason::Unauthorized,
                    2 => RejectReason::Full,
                    3 => RejectReason::Draining,
                    4 => RejectReason::Starting,
                    _ => RejectReason::Other(self.string()),
                },
            }),
            3 => Message::InputTick(InputTick {
                tick: self.next_u64(),
                seq: self.next_u64(),
                inputs: (0..self.below(8))
                    .map(|_| PlayerInput {
                        entity: self.guid(),
                        thrust: self.vec3(),
                        torque: self.vec3(),
                        tick: self.next_u64(),
                        fast: self.below(2) == 1,
                    })
                    .collect(),
                ack_tick: self.next_u64(),
            }),
            4 => Message::Snapshot(Snapshot {
                tick: self.next_u64(),
                shard: self.shard(),
                epoch: self.next_u64(),
                entities: self.entities(),
                input_ack: self.next_u64(),
                more: false,
            }),
            5 => Message::Delta(Delta {
                tick: self.next_u64(),
                base_tick: self.next_u64(),
                shard: self.shard(),
                epoch: self.next_u64(),
                entities: self.entities(),
                removed: (0..self.below(4)).map(|_| self.guid()).collect(),
                input_ack: self.next_u64(),
            }),
            6 => Message::HandoffOffer(HandoffOffer {
                entity: self.guid(),
                from: self.shard(),
                to: self.shard(),
                epoch: self.next_u64(),
                state: Box::new(self.entity()),
                activation_tick: self.next_u64(),
            }),
            7 => Message::HandoffAccept(HandoffAccept {
                entity: self.guid(),
                from: self.shard(),
                to: self.shard(),
                epoch: self.next_u64(),
                activation_tick: self.next_u64(),
            }),
            8 => Message::HandoffCommit(HandoffCommit {
                entity: self.guid(),
                from: self.shard(),
                to: self.shard(),
                epoch: self.next_u64(),
                activation_tick: self.next_u64(),
                state: Box::new(self.entity()),
            }),
            9 => Message::HandoffAbort(HandoffAbort {
                entity: self.guid(),
                from: self.shard(),
                to: self.shard(),
                reason: match self.below(7) {
                    0 => HandoffReason::TargetFull,
                    1 => HandoffReason::OfferTimeout,
                    2 => HandoffReason::CommitTimeout,
                    3 => HandoffReason::EpochChanged,
                    4 => HandoffReason::Despawned,
                    5 => HandoffReason::ClientNotConnected,
                    _ => HandoffReason::Other(self.string()),
                },
            }),
            10 => Message::GhostUpdate(GhostUpdate {
                from: self.shard(),
                epoch: self.next_u64(),
                tick: self.next_u64(),
                full: self.next_u64().is_multiple_of(2),
                entities: self.entities(),
                removed: vec![self.guid()],
                islands: vec![IslandKey {
                    body: self.guid(),
                    min: self.guid(),
                    home: self.shard(),
                }],
            }),
            11 => Message::Redirect(Redirect {
                entity: self.guid(),
                new_shard: self.shard(),
                shard_map: self.map(),
                activation_tick: self.next_u64(),
            }),
            12 => Message::StaleEpoch(StaleEpoch {
                seen: self.next_u64(),
                current: self.next_u64(),
            }),
            13 => Message::EpochPrepare(space_protocol::EpochPrepare {
                epoch: self.next_u64(),
                activation_tick: self.next_u64(),
                changes: (0..self.below(5))
                    .map(|_| (self.node(), self.shard()))
                    .collect(),
            }),
            14 => Message::EpochAck(space_protocol::EpochAck {
                epoch: self.next_u64(),
                shard: self.shard(),
            }),
            15 => Message::EpochCommit(space_protocol::EpochCommit {
                epoch: self.next_u64(),
            }),
            16 => Message::EpochAbort(space_protocol::EpochAbort {
                epoch: self.next_u64(),
                reason: match self.below(5) {
                    0 => space_protocol::AbortReason::TargetFull,
                    1 => space_protocol::AbortReason::Timeout,
                    2 => space_protocol::AbortReason::EpochChanged,
                    3 => space_protocol::AbortReason::ShuttingDown,
                    _ => space_protocol::AbortReason::Other(self.string()),
                },
            }),
            17 => Message::LoadReport(LoadReport {
                shard: self.shard(),
                players: self.u32_below(1_000),
                tick_p95_ms: self.f32(),
                load: self.f32(),
            }),
            18 => Message::DirectoryQuery(DirectoryQuery {}),
            19 => Message::DirectoryReply(DirectoryReply {
                shard_map: self.map(),
                tick: self.next_u64(),
            }),
            20 => Message::AdminCommand(match self.below(5) {
                0 => AdminCommand::ForceSplit(self.shard()),
                1 => AdminCommand::ForceMerge(self.shard()),
                2 => AdminCommand::KillShard(self.shard()),
                3 => AdminCommand::SpawnBots {
                    count: self.u32_below(10_000),
                },
                _ => AdminCommand::Status,
            }),
            21 | 22 => Message::AdminReply(match self.below(3) {
                0 => AdminReply::Accepted {
                    command: AdminCommand::Status,
                },
                1 => AdminReply::Refused {
                    command: AdminCommand::KillShard(self.shard()),
                    reason: self.string(),
                },
                _ => AdminReply::Status {
                    lines: (0..self.below(4)).map(|_| self.string()).collect(),
                },
            }),
            23 => Message::ShardHello(ShardHello {
                cluster: self.string(),
                shard: self.shard(),
                endpoint: self.addr(),
                region: if self.next_u64() & 1 == 0 {
                    None
                } else {
                    Some(self.region())
                },
                client_endpoint: if self.next_u64() & 1 == 0 {
                    None
                } else {
                    Some(self.addr())
                },
                secret: self.string(),
            }),
            24 => Message::ShardWelcome(ShardWelcome {
                shard: self.shard(),
                region: self.region(),
                shard_map: self.map(),
                peer_token: self.string(),
                tick_origin_us: self.next_u64() as i64,
            }),
            25 => Message::SceneSet(self.scene_state()),
            26 => Message::SceneAck(SceneAck {
                shard: self.shard(),
                applied: self.scene_state(),
            }),
            27 => Message::SceneUpdate(self.scene_state()),
            28 => Message::SceneClock(SceneClock::new(self.next_u64(), self.next_u64() as i64)),
            29 => Message::JoinRedirect(JoinRedirect {
                shard: self.shard(),
                shard_map: self.map(),
            }),
            30 => Message::FarInterest(space_protocol::FarInterest {
                from: self.shard(),
                tick: self.next_u64(),
                players: (0..self.below(4))
                    .map(|_| space_protocol::FarPlayer {
                        cell: [self.next_u64() as i64, self.next_u64() as i64, 7],
                        local: [self.below(500) as f32, -1.5, 0.0],
                    })
                    .collect(),
                missing: (0..self.below(4)).map(|_| self.guid()).collect(),
            }),
            _ => Message::FarUpdate(space_protocol::FarUpdate {
                from: self.shard(),
                tick: self.next_u64(),
                new: self.entities(),
                changed: self.entities(),
                removed: (0..self.below(4)).map(|_| self.guid()).collect(),
                refresh: self.below(2) == 0,
                held: (0..self.below(4)).map(|_| self.guid()).collect(),
                more: self.below(2) == 0,
            }),
        }
    }
    /// A scene state over the **whole** id space, not just the catalogue: an
    /// id this build does not know has to cross the wire like any other.
    fn scene_state(&mut self) -> SceneState {
        SceneState::new(SceneId(self.next_u64() as u16), self.next_u64())
    }
}

#[test]
fn property_random_messages_roundtrip() {
    let mut rng = Rng::new(0x5EED_1234_ABCD_0001);
    let mut total_bytes = 0usize;
    let mut max_bytes = 0usize;
    for i in 0..1000 {
        let message = rng.message();
        let frame = encode_frame(&message);
        total_bytes += frame.len();
        max_bytes = max_bytes.max(frame.len());
        let decoded = decode_frame(&frame)
            .unwrap_or_else(|e| panic!("message #{i} failed to decode: {e} ({message:?})"));
        assert_eq!(decoded, message, "message #{i} did not roundtrip");
    }
    println!(
        "1000 random messages: total {} bytes, mean {:.0} B, max {max_bytes} B",
        total_bytes,
        total_bytes as f64 / 1000.0
    );
}

// ---------------------------------------------------------------------------
// Size sanity
// ---------------------------------------------------------------------------

#[test]
fn delta_size_sanity() {
    // A typical delta: 100 entities × 3 small components (3 × f32 each).
    let entities: Vec<EntitySnapshot> = (0..100u64).map(|i| entity(1, i, 3)).collect();
    let delta = Message::Delta(Delta {
        tick: 1_000,
        base_tick: 999,
        shard: shard(1),
        epoch: 7,
        entities,
        removed: Vec::new(),
        input_ack: 0,
    });
    let frame = encode_frame(&delta);
    let payload = frame.len() - FRAME_HEADER_LEN;
    println!(
        "Delta(100 entities x 3 components): frame {} B, payload {} B, {:.1} B/entity",
        frame.len(),
        payload,
        payload as f64 / 100.0
    );
    assert!(
        payload < 20 * 1024,
        "delta payload is {payload} bytes, budget is 20 KiB"
    );
    assert!(frame.len() < 20 * 1024 + FRAME_HEADER_LEN);

    // The same corpus as a Snapshot costs less: a `Delta` additionally carries
    // `base_tick` (8 B) and the `removed` length prefix (8 B when empty); a
    // `Snapshot` carries `more` (1 B: "further chunks follow", protocol 7).
    let snap = Message::Snapshot(Snapshot {
        tick: 1_000,
        shard: shard(1),
        epoch: 7,
        entities: (0..100u64).map(|i| entity(1, i, 3)).collect(),
        input_ack: 0,
        more: false,
    });
    assert_eq!(encode_frame(&snap).len() + 15, frame.len());
}

// ---------------------------------------------------------------------------
// FIX 3: new message fields survive a round trip
// ---------------------------------------------------------------------------

/// FIX 3: the Phase-1 remediation fields are real wire fields, not defaults —
/// each survives encode/decode with a non-zero value.
#[test]
fn remediation_fields_roundtrip() {
    // InputTick::ack_tick
    let m = Message::InputTick(InputTick {
        tick: 5,
        seq: 6,
        inputs: Vec::new(),
        ack_tick: 4,
    });
    let Message::InputTick(back) = decode_frame(&encode_frame(&m)).unwrap() else {
        unreachable!()
    };
    assert_eq!(back.ack_tick, 4);

    // Snapshot::input_ack
    let m = Message::Snapshot(Snapshot {
        tick: 10,
        shard: shard(1),
        epoch: 2,
        entities: Vec::new(),
        input_ack: 9,
        more: false,
    });
    let Message::Snapshot(back) = decode_frame(&encode_frame(&m)).unwrap() else {
        unreachable!()
    };
    assert_eq!(back.input_ack, 9);

    // Delta::input_ack and Delta::removed
    let m = Message::Delta(Delta {
        tick: 11,
        base_tick: 10,
        shard: shard(1),
        epoch: 2,
        entities: Vec::new(),
        removed: vec![guid(1, 100), guid(2, 200)],
        input_ack: 10,
    });
    let Message::Delta(back) = decode_frame(&encode_frame(&m)).unwrap() else {
        unreachable!()
    };
    assert_eq!(back.input_ack, 10);
    assert_eq!(back.removed, vec![guid(1, 100), guid(2, 200)]);

    // HandoffCommit::state
    let m = Message::HandoffCommit(HandoffCommit {
        entity: guid(1, 42),
        from: shard(1),
        to: shard(2),
        epoch: 7,
        activation_tick: 1_000,
        state: Box::new(entity(1, 42, 3)),
    });
    let Message::HandoffCommit(back) = decode_frame(&encode_frame(&m)).unwrap() else {
        unreachable!()
    };
    assert_eq!(back.state.guid, guid(1, 42));
    assert_eq!(back.state.components.len(), 3);

    // Redirect::activation_tick
    let m = Message::Redirect(Redirect {
        entity: guid(2, 9),
        new_shard: shard(2),
        shard_map: sample_map(),
        activation_tick: 2_500,
    });
    let Message::Redirect(back) = decode_frame(&encode_frame(&m)).unwrap() else {
        unreachable!()
    };
    assert_eq!(back.activation_tick, 2_500);

    // NeighbourStatus message + HandoffReason::ClientNotConnected
    let m = Message::NeighbourStatus(NeighbourStatus {
        entity: guid(2, 9),
        shard: shard(2),
        connected: true,
    });
    let Message::NeighbourStatus(back) = decode_frame(&encode_frame(&m)).unwrap() else {
        unreachable!()
    };
    assert!(back.connected);
    assert_eq!(back.shard, shard(2));

    let m = Message::HandoffAbort(HandoffAbort {
        entity: guid(1, 42),
        from: shard(1),
        to: shard(2),
        reason: HandoffReason::ClientNotConnected,
    });
    let Message::HandoffAbort(back) = decode_frame(&encode_frame(&m)).unwrap() else {
        unreachable!()
    };
    assert_eq!(back.reason, HandoffReason::ClientNotConnected);
}

// ---------------------------------------------------------------------------
// FIX 5: octree nodes are validated on decode
// ---------------------------------------------------------------------------

/// FIX 5: a shard map or epoch change carrying an octree node that no real
/// position can address is rejected on decode, so every consumer of a decoded
/// map stays inside the addressing rules.
///
/// `NodeRef::new` accepts any `(depth, index)`, so a bad node can be *encoded*;
/// the rejection happens on decode (see `shard::validate_node`).
#[test]
fn octree_nodes_validated_on_decode() {
    // A well-formed map round-trips (sanity).
    let good = Message::DirectoryReply(DirectoryReply {
        shard_map: sample_map(),
        tick: 1,
    });
    assert!(decode_frame(&encode_frame(&good)).is_ok());

    // A node deeper than MAX_DEPTH.
    let too_deep = Message::DirectoryReply(DirectoryReply {
        shard_map: ShardMap::new(1, vec![region(space_world::MAX_DEPTH + 1, [0, 0, 0], 1)]),
        tick: 1,
    });
    assert!(
        decode_frame(&encode_frame(&too_deep)).is_err(),
        "a node deeper than MAX_DEPTH must be rejected"
    );

    // A root (depth 0) with a non-zero index.
    let bad_root = Message::DirectoryReply(DirectoryReply {
        shard_map: ShardMap::new(1, vec![region(0, [1, 0, 0], 1)]),
        tick: 1,
    });
    assert!(
        decode_frame(&encode_frame(&bad_root)).is_err(),
        "a depth-0 node with a non-zero index must be rejected"
    );

    // An out-of-range axis index at a small depth. At depth 1 the valid range is
    // [-1, 1), so index 1 is out of range.
    let bad_axis = Message::DirectoryReply(DirectoryReply {
        shard_map: ShardMap::new(1, vec![region(1, [1, 0, 0], 1)]),
        tick: 1,
    });
    assert!(
        decode_frame(&encode_frame(&bad_axis)).is_err(),
        "an axis index outside [-2^(d-1), 2^(d-1)) must be rejected"
    );

    // A valid depth-1 node with index [0,0,0] is accepted (range [-1,1)).
    let ok_axis = Message::DirectoryReply(DirectoryReply {
        shard_map: ShardMap::new(1, vec![region(1, [0, 0, 0], 1)]),
        tick: 1,
    });
    assert!(decode_frame(&encode_frame(&ok_axis)).is_ok());

    // An EpochPrepare change with a bad node is rejected too.
    let bad_prepare = Message::EpochPrepare(space_protocol::EpochPrepare {
        epoch: 8,
        activation_tick: 2_000,
        changes: vec![(
            NodeRef::new(space_world::MAX_DEPTH + 1, [0, 0, 0]),
            shard(4),
        )],
    });
    assert!(
        decode_frame(&encode_frame(&bad_prepare)).is_err(),
        "an EpochPrepare node beyond MAX_DEPTH must be rejected"
    );
}

// ---------------------------------------------------------------------------
// FIX 6: SnapshotBuilder overhead uses max(Snapshot, Delta)
// ---------------------------------------------------------------------------

/// FIX 6: the builder reserves the **larger** of the empty `Snapshot` and empty
/// `Delta` overheads, so a `finish_delta` after `set_tick` cannot overrun the
/// budget by the Snapshot/Delta difference.
#[test]
fn builder_overhead_is_max_of_snapshot_and_delta() {
    const BUDGET: usize = 4096;
    let mut builder = SnapshotBuilder::with_budget(BUDGET, 1, shard(1), 7);
    // Fill to the budget with real entities.
    let entities: Vec<EntitySnapshot> = (0..500u64).map(|i| entity(1, i, 3)).collect();
    let mut pushed = 0;
    for e in &entities {
        if builder.push(e) {
            pushed += 1;
        } else {
            break;
        }
    }
    assert!(pushed > 0);
    // The finished Delta must fit the budget (it is the larger message shape).
    let delta = builder.finish_delta(0, Vec::new(), 0);
    let delta_len = payload_len(&Message::Delta(delta));
    assert!(
        delta_len <= BUDGET,
        "delta of {pushed} entities is {delta_len} B, budget {BUDGET}"
    );

    // After set_tick, a delta batch must still respect the budget (the overhead
    // recomputed by set_tick must be the max, not just the Snapshot overhead).
    let mut builder = SnapshotBuilder::with_budget(BUDGET, 1, shard(1), 7);
    builder.set_tick(50, 9);
    let mut pushed = 0;
    for e in &entities {
        if builder.push(e) {
            pushed += 1;
        } else {
            break;
        }
    }
    assert!(pushed > 0, "the builder must accept at least one entity");
    let delta = builder.finish_delta(49, Vec::new(), 0);
    let delta_len = payload_len(&Message::Delta(delta));
    assert!(
        delta_len <= BUDGET,
        "delta after set_tick is {delta_len} B, budget {BUDGET}"
    );
}

// ---------------------------------------------------------------------------
// Scene messages
// ---------------------------------------------------------------------------

/// The three scene messages keep their tags and their byte layout: tag (u32),
/// then for an ack the shard (u32), then the scene id (u16) and the revision
/// (u64), all big endian.
///
/// Mutation this catches: inserting a variant before them, reordering the
/// fields of `SceneState`, or widening the id — each is a silent wire break
/// between an orchestrator, a shard and a client built at different commits.
#[test]
fn scene_messages_have_a_pinned_wire_layout() {
    let state = SceneState::new(SceneId(0x0102), 0x0A0B_0C0D_0E0F_1011);
    let state_bytes = [0x01, 0x02, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11];
    let cases: [(Message, u32, Vec<u8>); 3] = [
        (Message::SceneSet(state), 25, state_bytes.to_vec()),
        (
            Message::SceneAck(SceneAck {
                shard: shard(7),
                applied: state,
            }),
            26,
            [&[0, 0, 0, 7][..], &state_bytes[..]].concat(),
        ),
        (Message::SceneUpdate(state), 27, state_bytes.to_vec()),
    ];
    for (message, tag, body) in cases {
        assert_eq!(message.wire_tag(), tag, "{message:?}");
        assert_eq!(message.epoch(), None, "a scene message carries no epoch");
        let frame = encode_frame(&message);
        let mut expected = tag.to_be_bytes().to_vec();
        expected.extend_from_slice(&body);
        assert_eq!(payload_of(&frame), &expected[..], "{message:?}");
        assert_eq!(decode_frame(&frame), Ok(message));
    }
}

/// Protocol 7: a `ShardHello` carries the cluster secret as its **last**
/// field (a length-prefixed string behind the optional region and the
/// optional client endpoint), and neither it nor the peer token of a
/// `ShardWelcome` shows in a `Debug` print — a message may end up in a log
/// line or a panic message.
///
/// Mutation this catches: adding a field behind the secret (it is what a
/// hello is admitted or refused by, and stays the last thing in it), leaving
/// the version at 6 with the changed layout, or deriving `Debug` again.
#[test]
fn a_shard_hello_ends_with_the_cluster_secret_and_never_prints_it() {
    assert_eq!(
        PROTOCOL_VERSION, 7,
        "the hello's layout changed with protocol 7"
    );
    let hello = |secret: &str| {
        Message::ShardHello(ShardHello {
            cluster: "cluster-a".into(),
            shard: ShardId::NONE,
            endpoint: addr(9),
            region: None,
            client_endpoint: Some(any_host(7_009)),
            secret: secret.into(),
        })
    };
    let secret = "s3cr3t-of-the-cluster";
    let (without, with) = (encode_frame(&hello("")), encode_frame(&hello(secret)));
    let (without, with) = (payload_of(&without), payload_of(&with));
    assert_eq!(hello("").wire_tag(), 23);
    assert_eq!(&with[..4], &23u32.to_be_bytes());
    // Everything before the secret is the same bytes; the secret is its
    // length (u64, big endian) and then its bytes, and nothing follows.
    let head = without.len() - 8;
    assert_eq!(&with[..head], &without[..head]);
    assert_eq!(&without[head..], &0u64.to_be_bytes());
    let tail = [&(secret.len() as u64).to_be_bytes()[..], secret.as_bytes()].concat();
    assert_eq!(&with[head..], &tail[..]);
    assert_eq!(
        decode_frame(&encode_frame(&hello(secret))),
        Ok(hello(secret))
    );

    let printed = format!("{:?}", hello(secret));
    assert!(!printed.contains(secret), "{printed}");
    assert!(printed.contains("<redacted>"), "{printed}");
    assert!(
        format!("{:?}", hello("")).contains("<none>"),
        "an absent secret is said to be absent"
    );
    let welcome = Message::ShardWelcome(ShardWelcome {
        shard: shard(5),
        region: region(40, [0, 0, 0], 5),
        shard_map: sample_map(),
        peer_token: "peer-s3cr3t-token".into(),
        tick_origin_us: 0,
    });
    let printed = format!("{welcome:?}");
    assert!(!printed.contains("peer-s3cr3t-token"), "{printed}");
    assert!(printed.contains("<redacted>"), "{printed}");
}

/// The first message on every connection carries a credential in its token
/// field — a player's token, or between shards `shard:<id>:<peer token>` —
/// and a hello is printed: an endpoint that expected something else logs
/// "got {message:?}". Its `Debug` keeps what is not secret (the version, the
/// shard id a peer names) and nothing else. The wire form is untouched.
///
/// Mutation this catches: deriving `Debug` on `Hello` again (a shard that is
/// sent a stray hello would write the cluster's peer token to its log).
#[test]
fn a_hello_never_prints_its_token() {
    let hello = |token: &str| {
        Message::Hello(Hello {
            protocol: PROTOCOL_VERSION,
            player_token: token.into(),
        })
    };
    for (token, shown, hidden) in [
        (
            "shard:7:the-peer-token-of-the-cluster",
            "shard:7:<redacted>",
            "the-peer-token",
        ),
        (
            "shard:12:with:colons:in:it",
            "shard:12:<redacted>",
            "colons",
        ),
        ("a-players-own-token", "<redacted>", "players-own"),
        // Not a peer hello (no numeric id): nothing of it is shown.
        ("shard:x:looks-like-a-peer", "<redacted>", "looks-like"),
        ("shard::no-id-at-all", "<redacted>", "no-id-at-all"),
    ] {
        let printed = format!("{:?}", hello(token));
        assert!(printed.contains(shown), "{token}: {printed}");
        assert!(!printed.contains(hidden), "{token}: {printed}");
        assert!(
            printed.contains(&format!("protocol: {PROTOCOL_VERSION}")),
            "{printed}"
        );
        // Printing is all that changed: the token still travels whole.
        assert_eq!(decode_frame(&encode_frame(&hello(token))), Ok(hello(token)));
    }
    // A peer hello without a token has nothing to hide, and no token is said
    // to be none.
    let printed = format!("{:?}", hello("shard:3"));
    assert!(
        printed.contains("shard:3") && !printed.contains("redacted"),
        "{printed}"
    );
    assert!(format!("{:?}", hello("")).contains("<none>"));
}

/// A scene id this build does not know is **not** a decode error: the frame
/// decodes, the id is reported as unknown, and the receiver's fallback is the
/// default scene. (The catalogue grows; an older client must survive a newer
/// scene, not drop the frame.)
///
/// Mutation this catches: putting the `Scene` enum itself on the wire — an
/// unknown variant index is then a decode error and the receiver never learns
/// the revision either.
#[test]
fn an_unknown_scene_id_decodes_and_the_receiver_can_fall_back() {
    // 2 is the retired `solar`: unknown now, like an id from the future.
    for raw in [2, 4, 77, u16::MAX] {
        for message in [
            Message::SceneUpdate(SceneState::new(SceneId(raw), 12)),
            Message::SceneSet(SceneState::new(SceneId(raw), 12)),
        ] {
            let decoded = decode_frame(&encode_frame(&message)).expect("an unknown id decodes");
            assert_eq!(decoded, message);
            let (Message::SceneUpdate(state) | Message::SceneSet(state)) = decoded else {
                panic!("a scene message");
            };
            assert_eq!(state.revision, 12, "the revision is still readable");
            assert_eq!(
                state.scene.known(),
                None,
                "id {raw} is not in this catalogue"
            );
            assert_eq!(state.scene.or_default(), Scene::DEFAULT);
        }
    }
    // Every catalogue scene, by contrast, comes back as itself.
    for scene in Scene::ALL {
        let message = Message::SceneUpdate(SceneState::new(scene.id(), 1));
        let Ok(Message::SceneUpdate(state)) = decode_frame(&encode_frame(&message)) else {
            panic!("{scene} must round-trip");
        };
        assert_eq!(state.scene.known(), Some(scene));
    }
}

// ---------------------------------------------------------------------------
// One address: join redirects and client endpoints (protocol 7)
// ---------------------------------------------------------------------------

/// `JoinRedirect` keeps tag 29 (the last message until the far replicas were
/// appended behind it), carries the shard to dial and the sender's map, and
/// reports that map's epoch.
///
/// Mutation this catches: inserting a variant before it (its tag moves), or
/// dropping the map (the client would have no address to dial).
#[test]
fn a_join_redirect_round_trips_at_tag_29() {
    let message = join_redirect_message();
    assert_eq!(message.wire_tag(), 29);
    assert_eq!(message.epoch(), Some(sample_map().epoch));
    let frame = encode_frame(&message);
    assert_eq!(&payload_of(&frame)[..4], &29u32.to_be_bytes());
    // The shard id follows the tag, then the map.
    assert_eq!(&payload_of(&frame)[4..8], &2u32.to_be_bytes());
    let decoded = decode_frame(&frame).expect("decodes");
    assert_eq!(decoded, message);
    let Message::JoinRedirect(redirect) = decoded else {
        panic!("a join redirect");
    };
    assert_eq!(redirect.shard, shard(2));
    assert_eq!(
        redirect.shard_map.client_endpoint_of(shard(2)),
        Some(any_host(7_002)),
        "the map names where a player dials the shard"
    );
}

/// A region carries two addresses: where shards reach its owner and where
/// players do. Built with `ShardRegion::new` they are the same one; a
/// different client endpoint crosses the wire as itself, behind the endpoint.
///
/// Mutation this catches: a client endpoint that is dropped on the wire (a
/// redirected player dials the internal address), or `new` leaving it unset
/// (every static map would name an un-dialable client address).
#[test]
fn a_region_carries_a_client_endpoint_beside_its_endpoint() {
    let plain = region(40, [0, 0, 0], 3);
    assert_eq!(plain.client_endpoint, plain.endpoint);
    assert_eq!(plain.client_endpoint, addr(3));
    let public = plain.clone().with_client_endpoint(any_host(7_171));
    assert_eq!(public.endpoint, addr(3), "the internal endpoint is kept");
    assert_eq!(public.client_endpoint, any_host(7_171));
    assert_ne!(public, plain);

    let welcome = |region: &ShardRegion| {
        Message::Welcome(Welcome {
            session: 1,
            shard: shard(3),
            shard_map: ShardMap::new(4, vec![region.clone()]),
        })
    };
    for region in [&plain, &public] {
        let message = welcome(region);
        assert_eq!(decode_frame(&encode_frame(&message)), Ok(message));
    }
    // The client endpoint is the region's last field: the two frames differ
    // only in their last six bytes (an IPv4 address and a port).
    let (a, b) = (
        encode_frame(&welcome(&plain)),
        encode_frame(&welcome(&public)),
    );
    assert_eq!(a.len(), b.len());
    let same = a.len() - 6;
    assert_eq!(&a[..same], &b[..same]);
    assert_eq!(&b[same..], &[0, 0, 0, 0, 0x1C, 0x03], "0.0.0.0:7171");

    // The map's lookup of it skips a region whose address is not known yet.
    let unresolved = ShardRegion::new(
        NodeRef::new(41, [0, 0, 0]),
        shard(9),
        ShardRegion::UNRESOLVED_ENDPOINT,
    );
    let map = ShardMap::new(1, vec![unresolved, public.clone()]);
    assert_eq!(map.client_endpoint_of(shard(9)), None);
    assert_eq!(map.client_endpoint_of(shard(3)), Some(any_host(7_171)));
    assert_eq!(map.client_endpoint_of(shard(77)), None, "not in the map");
}

/// The rule a client resolves a client endpoint by: an unspecified IP is the
/// host the player dialed first, port 0 is "not known", anything else is
/// taken as it is.
///
/// Mutation this catches: dialling `0.0.0.0` (which reaches nobody, or the
/// client's own machine), or resolving a real address against the entry host
/// (a shard on another host would be dialled on the orchestrator's).
#[test]
fn an_unspecified_client_endpoint_means_the_host_the_player_dialed() {
    use space_protocol::resolve_client_endpoint as resolve;
    let entry: IpAddr = "203.0.113.7".parse().unwrap();
    let entry6: IpAddr = "2001:db8::7".parse().unwrap();
    assert_eq!(
        resolve(any_host(7_871), entry),
        Some("203.0.113.7:7871".parse().unwrap())
    );
    assert_eq!(
        resolve("[::]:7871".parse().unwrap(), entry),
        Some("203.0.113.7:7871".parse().unwrap()),
        "the unspecified IPv6 address means the same"
    );
    assert_eq!(
        resolve(any_host(7_871), entry6),
        Some("[2001:db8::7]:7871".parse().unwrap())
    );
    // A real address is where the shard is, whatever was dialed.
    assert_eq!(resolve(addr(4), entry), Some(addr(4)));
    assert_eq!(
        resolve("127.0.0.1:7777".parse().unwrap(), entry),
        Some("127.0.0.1:7777".parse().unwrap())
    );
    // Port 0: the owner has not registered an address yet.
    assert_eq!(resolve(ShardRegion::UNRESOLVED_ENDPOINT, entry), None);
    assert_eq!(resolve("10.0.0.1:0".parse().unwrap(), entry), None);
}

/// A `ShardHello` says where players are to dial the shard — or does not, and
/// then it is the endpoint. Both forms cross the wire, the field sits before
/// the secret, and the print still hides the secret.
///
/// Mutation this catches: a hello whose client endpoint is lost (the map
/// would send players to the internal address), or the new field moving the
/// secret off the end.
#[test]
fn a_shard_hello_may_ask_for_a_client_endpoint() {
    let hello = |client_endpoint: Option<SocketAddr>| {
        Message::ShardHello(ShardHello {
            cluster: "cluster-a".into(),
            shard: ShardId::NONE,
            endpoint: addr(9),
            region: None,
            client_endpoint,
            secret: "s3cr3t-of-the-cluster".into(),
        })
    };
    let (none, some) = (hello(None), hello(Some(any_host(7_871))));
    for message in [&none, &some] {
        assert_eq!(decode_frame(&encode_frame(message)).as_ref(), Ok(message));
        let printed = format!("{message:?}");
        assert!(!printed.contains("s3cr3t"), "{printed}");
        assert!(printed.contains("<redacted>"), "{printed}");
    }
    assert!(format!("{some:?}").contains("0.0.0.0:7871"));
    // An absent one is one byte (the option's tag); a present IPv4 one adds
    // the address' own tag (u32), its four bytes and the port. The secret
    // closes both.
    let (none, some) = (encode_frame(&none), encode_frame(&some));
    assert_eq!(some.len(), none.len() + 10);
    let tail = "s3cr3t-of-the-cluster".as_bytes();
    assert!(none.ends_with(tail) && some.ends_with(tail));
}

// ---------------------------------------------------------------------------
// Tag bookkeeping
// ---------------------------------------------------------------------------
#[test]
fn message_tags_are_dense_and_covered() {
    // `all_messages` exercises every variant...
    let tags: std::collections::BTreeSet<u32> =
        all_messages().iter().map(Message::wire_tag).collect();
    assert_eq!(
        tags.len(),
        Message::VARIANT_COUNT as usize,
        "all_messages must cover every variant (tags {tags:?})"
    );
    // ...and the tags are exactly `0..VARIANT_COUNT`, with no gaps or duplicates.
    assert_eq!(tags.iter().next(), Some(&0));
    assert_eq!(tags.iter().next_back(), Some(&(Message::VARIANT_COUNT - 1)));
    for (i, tag) in tags.iter().enumerate() {
        assert_eq!(*tag as usize, i, "tags must be dense");
    }
}
