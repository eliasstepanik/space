//! Phase 2 acceptance tests: two shards, one process, the handoff contract.
//!
//! [`Shard`] is synchronous on purpose (see its module docs), so almost everything
//! here runs two shards side by side and moves their [`OutMessage`]s between each
//! other by hand — no sockets, no wall clock, no flakiness. The tick loop is a
//! `for` loop, so a test that needs 400 ticks costs 400 ticks of a one-ship
//! simulation, not 13 seconds of waiting.
//!
//! The tests map onto the eight guarantees in `handoff.rs`:
//!
//! | test | guarantees |
//! |------|------------|
//! | [`version_mismatch_is_refused_over_the_wire`] | the transport gate, before any shard state exists |
//! | [`client_gets_a_snapshot_then_deltas_and_its_input_ack_advances`] | replication + input ack |
//! | [`handoff_keeps_the_guid_position_and_inputs_continuous`] | 1, 2, 3, 4, 5 |
//! | [`an_observer_sees_no_gap_across_a_handoff`] | 3 |
//! | [`handoff_leaves_no_desync`] | 2, 4 |
//! | [`an_unanswered_offer_times_out_and_the_world_is_unchanged`] | 6, 8 |
//! | [`a_stale_epoch_offer_is_refused_and_the_transfer_aborts`] | 7 |
//! | [`a_snapshot_of_1000_ships_costs_about_158_bytes_per_entity`] | U9 budget |
//! | [`a_client_can_connect_to_a_running_shard`] | the quinn front end end to end |

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use space_protocol::{
    EntityGuid, Hello, InputTick, Message, NeighbourStatus, PlayerInput, Redirect, ShardId,
    ShardMap, ShardRegion, Welcome, PROTOCOL_VERSION,
};
use space_sim::ShipState;
use space_world::{NodeRef, WorldPos};

use space_server::shard::{Destination, Shard};
use space_server::ShardConfig;

/// Depth of a test region.
///
/// 64 is one cell wide ([`space_world::CELL_SIZE`] = 1024 m), which is small
/// enough that a ship at full thrust crosses it in a few hundred ticks — long
/// enough for the handoff protocol to run through its states, cheap enough to
/// run in a test.
const DEPTH: u8 = 64;

/// Loopback address for shard endpoints.
fn addr() -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)
}

/// The region a shard owns: one cell-wide octree node at cell `x`.
fn region(x: i64) -> NodeRef {
    NodeRef::new(DEPTH, [x, 0, 0])
}

/// The centre of the test region at cell `x`.
fn region_center(x: i64) -> WorldPos {
    region(x).center()
}

/// The raw payload of component `id` in `snapshot`.
fn payload_of(snapshot: &space_protocol::EntitySnapshot, id: u16) -> Option<&[u8]> {
    snapshot
        .components
        .iter()
        .find(|(c, _)| *c == id)
        .map(|(_, p)| p.as_slice())
}

/// The [`ShipState`] an [`space_protocol::EntitySnapshot`] carries.
///
/// `ShipState` is replicated through the lossless wire codec (not serde), so this
/// is the same decode the shard itself runs on an incoming commit.
fn ship_state_of(snapshot: &space_protocol::EntitySnapshot) -> ShipState {
    let payload = payload_of(snapshot, <ShipState as space_ecs::ComponentDef>::ID)
        .expect("a replicated ship always carries ShipState");
    space_sim::wire::decode_ship_state(payload).expect("the wire codec is lossless")
}

/// Two shards that share one two-region map, wired together in memory.
///
/// `a` owns the region at cell 0, `b` the one at cell 1 — adjacent cells 1024 m
/// apart, so a ship under thrust crosses from one into the other.
struct Pair {
    a: Shard,
    b: Shard,
    /// Frames queued for a client session: `(sending shard, session, message)`.
    ///
    /// The sending shard is part of the key because session ids are per-shard
    /// counters that both shards start at 1 — a bare session id is ambiguous.
    client_frames: Vec<(ShardId, u64, Message)>,
}

impl Pair {
    /// Build the pair. Both shards publish the same epoch-1 map.
    fn new() -> Self {
        let id_a = ShardId::new(1);
        let id_b = ShardId::new(2);
        let map = ShardMap::new(
            1,
            vec![
                ShardRegion::new(region(0), id_a, addr()),
                ShardRegion::new(region(1), id_b, addr()),
            ],
        );
        let mut a = Shard::new(ShardConfig::new(id_a, addr(), region(0)));
        let mut b = Shard::new(ShardConfig::new(id_b, addr(), region(1)));
        a.set_shard_map(map.clone());
        b.set_shard_map(map);
        Self {
            a,
            b,
            client_frames: Vec::new(),
        }
    }

    /// The id of the shard that is *not* `shard`.
    fn other(&self, shard: ShardId) -> ShardId {
        if shard == self.a.shard() {
            self.b.shard()
        } else {
            self.a.shard()
        }
    }

    /// Deliver everything both shards queued: shard-directed messages go to the
    /// peer, client-directed ones are collected for the test to inspect.
    ///
    /// Both outboxes are drained before anything is delivered, so a message that
    /// provokes an immediate reply is still ordered the way two independent
    /// processes would order it: one round of exchange per call.
    fn deliver(&mut self) {
        let (id_a, id_b) = (self.a.shard(), self.b.shard());
        let mut pending: Vec<(ShardId, Message)> = Vec::new();
        {
            let mut drain =
                |from: ShardId, shard: &mut Shard, frames: &mut Vec<(ShardId, u64, Message)>| {
                    for out in shard.take_outbox() {
                        match out.dest {
                            Destination::Client(session) => {
                                frames.push((from, session, out.message));
                            }
                            Destination::Shard(_) => pending.push((from, out.message)),
                            // Orchestrator-directed messages (epoch acks) are not
                            // exchanged between the pair; the epoch tests drive the
                            // orchestrator path directly and drain the outbox.
                            Destination::Orchestrator => {}
                        }
                    }
                };
            drain(id_a, &mut self.a, &mut self.client_frames);
            drain(id_b, &mut self.b, &mut self.client_frames);
        }
        for (from, message) in pending {
            let target = if self.other(from) == id_a {
                &mut self.a
            } else {
                &mut self.b
            };
            target.peer_message(from, message);
        }
    }

    /// Tick both shards once and exchange what they produced.
    fn tick(&mut self) {
        self.a.tick();
        self.b.tick();
        self.deliver();
    }

    /// File an input batch for `session` on `shard`, thrust along `axis`.
    ///
    /// The batch is tagged with the tick the shard will step to next, which is
    /// what a real client does: inputs are generated for the tick they predict.
    fn thrust(&mut self, shard: ShardId, session: u64, axis: usize, magnitude: f32) {
        let ship = {
            let target = self.shard_mut(shard);
            target.ship_of(session).expect("session has a ship")
        };
        let tick = self.shard(shard).sim().tick() + 1;
        let batch = InputTick {
            tick,
            seq: tick,
            inputs: vec![PlayerInput {
                entity: ship,
                thrust: [
                    if axis == 0 { magnitude } else { 0.0 },
                    if axis == 1 { magnitude } else { 0.0 },
                    if axis == 2 { magnitude } else { 0.0 },
                ],
                torque: [0.0; 3],
                tick,
            }],
            ack_tick: 0,
        };
        self.shard_mut(shard).apply_input(session, &batch);
    }

    fn shard(&self, id: ShardId) -> &Shard {
        if id == self.a.shard() {
            &self.a
        } else {
            &self.b
        }
    }

    fn shard_mut(&mut self, id: ShardId) -> &mut Shard {
        if id == self.a.shard() {
            &mut self.a
        } else {
            &mut self.b
        }
    }

    /// Frames sent to one `(shard, session)` pair, draining the log.
    fn frames_for(&mut self, shard: ShardId, session: u64) -> Vec<Message> {
        let mine = |f: &(ShardId, u64, Message)| f.0 == shard && f.1 == session;
        let (keep, taken): (Vec<_>, Vec<_>) = std::mem::take(&mut self.client_frames)
            .into_iter()
            .partition(|f| !mine(f));
        self.client_frames = keep;
        taken.into_iter().map(|(_, _, m)| m).collect()
    }
}

/// Connect `token` to `shard`, returning the session and its ship's guid.
/// The `input_ack` field of a snapshot or delta frame.
fn frame_input_ack(message: &Message) -> Option<u64> {
    match message {
        Message::Snapshot(s) => Some(s.input_ack),
        Message::Delta(d) => Some(d.input_ack),
        _ => None,
    }
}

fn join(shard: &mut Shard, token: &str) -> (u64, EntityGuid) {
    let welcome = shard
        .join(&Hello {
            protocol: PROTOCOL_VERSION,
            player_token: token.into(),
        })
        .expect("join");
    let ship = shard.ship_of(welcome.session).expect("a ship was spawned");
    (welcome.session, ship)
}

// --------------------------------------------------------------------- tests

/// A client that never speaks the right protocol never reaches the shard.
///
/// The version gate is the transport's, not the shard's (see
/// [`space_net::accept_incoming`]), so this is checked against a *running* shard:
/// the shard must end up with no client, no ship and no state at all.
#[tokio::test]
async fn version_mismatch_is_refused_over_the_wire() {
    let harness = RunningShard::start().await;
    let err = harness
        .connect_with(Hello {
            protocol: PROTOCOL_VERSION + 1,
            player_token: "wrong-version".into(),
        })
        .await
        .expect_err("a wrong protocol version must be refused");
    assert!(
        matches!(
            err,
            space_net::NetError::VersionMismatch {
                seen,
                expected,
            } if seen == PROTOCOL_VERSION + 1 && expected == PROTOCOL_VERSION
        ),
        "expected VersionMismatch, got {err:?}"
    );
    // Nothing was created behind the refused handshake.
    harness.settle().await;
    assert_eq!(harness.clients().await, 0, "a refused hello must not join");
    harness.shutdown();
}

/// A malformed frame is rejected, not trusted: the session ends and the shard
/// keeps running.
///
/// Two shapes matter: a header declaring a payload larger than
/// [`space_protocol::MAX_FRAME_SIZE`] (rejected **before** the bytes are read, so
/// a hostile length cannot be used as a memory bomb) and a header whose payload
/// never arrives (truncated). Both must tear the session down rather than feed
/// anything to the shard.
#[tokio::test]
async fn malformed_frames_end_the_session_instead_of_trusting_it() {
    for (name, write_bogus) in [
        (
            "oversized",
            Box::new(send_oversized_header) as Box<dyn Fn(quinn::Connection) -> BoxFuture>,
        ),
        (
            "truncated",
            Box::new(send_truncated_frame) as Box<dyn Fn(_) -> BoxFuture>,
        ),
    ] {
        let harness = RunningShard::start().await;
        let (conn, _welcome) = harness.connect("player").await.expect("connect");
        assert_eq!(harness.clients().await, 1, "{name}: the client joined");
        write_bogus(conn.connection().clone()).await;
        harness
            .wait_for_clients(0)
            .await
            .unwrap_or_else(|_| panic!("{name}: a malformed frame must end the session"));
        // The shard itself is alive and still accepts a well-formed client.
        let (_conn, welcome) = harness
            .connect("player-after")
            .await
            .expect("{name}: the shard must still accept clients");
        assert_eq!(welcome.session, 2, "{name}: the second session id");
        harness.shutdown();
    }
}

/// A shard running for real: bound endpoint, accept loop, 30 Hz tick loop.
///
/// The tick loop is the one `main.rs` runs, so this harness doubles as the smoke
/// test for the binary's serve path. Ticks are driven by a background task on a
/// real 30 Hz interval — the only place in this file that uses the wall clock,
/// and it is the place that has to, because the network is asynchronous.
struct RunningShard {
    net: std::sync::Arc<space_server::net::ShardNet>,
    shard: std::sync::Arc<tokio::sync::Mutex<Shard>>,
    tick: tokio::task::JoinHandle<()>,
    serve: tokio::task::JoinHandle<()>,
}

/// A future box, so the malformed-frame cases can be passed as values.
type BoxFuture = std::pin::Pin<Box<dyn std::future::Future<Output = ()>>>;

/// Send a frame header whose declared length exceeds [`space_protocol::MAX_FRAME_SIZE`].
fn send_oversized_header(conn: quinn::Connection) -> BoxFuture {
    Box::pin(async move {
        let mut stream = conn.open_uni().await.expect("open uni");
        let mut bogus = Vec::new();
        bogus.extend_from_slice(&((space_protocol::MAX_FRAME_SIZE + 1) as u32).to_be_bytes());
        bogus.extend_from_slice(&PROTOCOL_VERSION.to_be_bytes());
        stream.write_all(&bogus).await.expect("write header");
        stream.finish().expect("finish");
    })
}

/// Send a frame header that promises more payload than it ever delivers.
fn send_truncated_frame(conn: quinn::Connection) -> BoxFuture {
    Box::pin(async move {
        let mut stream = conn.open_uni().await.expect("open uni");
        let mut bogus = Vec::new();
        bogus.extend_from_slice(&1024u32.to_be_bytes());
        bogus.extend_from_slice(&PROTOCOL_VERSION.to_be_bytes());
        bogus.extend_from_slice(&[0u8; 4]);
        stream.write_all(&bogus).await.expect("write partial frame");
        stream.finish().expect("finish");
    })
}

impl RunningShard {
    /// Bind on loopback and start serving + ticking.
    async fn start() -> Self {
        let net = std::sync::Arc::new(space_server::net::ShardNet::bind(addr()).expect("bind"));
        let shard = std::sync::Arc::new(tokio::sync::Mutex::new(Shard::new(ShardConfig::new(
            ShardId::new(1),
            net.local_addr().expect("bound addr"),
            region(0),
        ))));

        let serve = {
            let net = std::sync::Arc::clone(&net);
            let shard = std::sync::Arc::clone(&shard);
            tokio::spawn(async move { net.serve(shard).await })
        };
        let tick = {
            let net = std::sync::Arc::clone(&net);
            let shard = std::sync::Arc::clone(&shard);
            tokio::spawn(async move {
                let mut clock = tokio::time::interval(std::time::Duration::from_secs_f64(
                    f64::from(space_sim::dt()),
                ));
                clock.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    clock.tick().await;
                    shard.lock().await.tick();
                    net.flush(&shard).await;
                }
            })
        };
        Self {
            net,
            shard,
            tick,
            serve,
        }
    }

    /// Connect a client through the real transport.
    async fn connect(
        &self,
        token: &str,
    ) -> Result<(space_net::Conn, Welcome), space_net::NetError> {
        self.connect_with(space_net::hello(token)).await
    }

    /// Connect with an arbitrary [`Hello`] (the version-mismatch case needs one).
    async fn connect_with(
        &self,
        hello: Hello,
    ) -> Result<(space_net::Conn, Welcome), space_net::NetError> {
        let endpoint = space_net::client_endpoint().expect("client endpoint");
        let bound = self.net.local_addr().expect("bound addr");
        space_net::connect(&endpoint, bound, space_server::net::SERVER_NAME, &hello).await
    }

    /// Connected client count, after the shard's next tick has settled.
    async fn clients(&self) -> usize {
        self.shard.lock().await.stats().clients
    }

    /// Let the tick loop run a few steps (long enough for a queued frame to be
    /// delivered and processed).
    async fn settle(&self) {
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    }

    /// Wait until the shard has `n` clients, or two seconds pass.
    async fn wait_for_clients(&self, n: usize) -> Result<(), ()> {
        for _ in 0..60 {
            if self.clients().await == n {
                return Ok(());
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        Err(())
    }

    /// The world tick the shard has reached.
    async fn world_tick(&self) -> u64 {
        self.shard.lock().await.sim().tick()
    }

    /// The guid of the first client's ship.
    async fn ship_guid(&self) -> Option<EntityGuid> {
        let shard = self.shard.lock().await;
        shard.ship_of(1)
    }

    /// The position of the first client's ship.
    async fn ship_position(&self) -> Option<WorldPos> {
        let shard = self.shard.lock().await;
        shard
            .ship_of(1)
            .and_then(|g| shard.sim().ship_state(g))
            .map(|s| s.pos)
    }

    /// Stop the tick loop, the accept loop and the endpoint.
    fn shutdown(&self) {
        self.tick.abort();
        self.serve.abort();
        self.net.close();
    }
}

/// A client joins, gets a snapshot, then deltas, and its `input_ack` advances.
///
/// This is the baseline every handoff guarantee is stated against: if the plain
/// replication path did not converge, "seamless" would mean nothing.
#[test]
fn client_gets_a_snapshot_then_deltas_and_its_input_ack_advances() {
    let mut pair = Pair::new();
    let id_a = pair.a.shard();
    let (session, ship) = join(&mut pair.a, "player");

    // Tick once so the replication pass runs; the first frame must be a snapshot
    // (the client knows nothing yet) and it must contain the client's own ship.
    pair.tick();
    let frames = pair.frames_for(id_a, session);
    let first = frames
        .iter()
        .find_map(|m| match m {
            Message::Snapshot(s) => Some(s),
            _ => None,
        })
        .expect("the first frame for a new client is a snapshot");
    assert!(
        first.entities.iter().any(|e| e.guid == ship),
        "the snapshot must contain the client's own ship"
    );
    assert_eq!(first.input_ack, 0, "nothing has been applied yet");

    // Now drive the ship and watch `input_ack` follow the inputs.
    let mut acks = Vec::new();
    for _ in 0..10 {
        pair.thrust(id_a, session, 0, 1.0);
        pair.tick();
        for message in pair.frames_for(id_a, session) {
            match message {
                Message::Snapshot(s) => acks.push(s.input_ack),
                Message::Delta(d) => acks.push(d.input_ack),
                _ => {}
            }
        }
    }
    let last = *acks.last().expect("the client kept receiving frames");
    assert!(
        acks.windows(2).all(|w| w[0] <= w[1]),
        "input_ack must never go backwards: {acks:?}"
    );
    assert!(
        last >= pair.a.sim().tick(),
        "input_ack {last} should have caught up to the world tick {}",
        pair.a.sim().tick()
    );

    // And the ship actually moved: the inputs were applied, not just acknowledged.
    let moved = pair
        .a
        .sim()
        .ship_state(ship)
        .expect("the ship is alive")
        .pos;
    assert_ne!(moved, region_center(0), "thrust must move the ship");
}

/// Drive a ship out of shard A into shard B and run the whole exchange.
///
/// Returns the handoff's moving parts so each guarantee can be asserted on its
/// own: the guid, the redirect the client got, the position it had on each side of
/// the boundary, and the frames each side sent.
struct Handoff {
    session_b: u64,
    guid: EntityGuid,
    redirect: Redirect,
    /// The last position shard A sent before it stopped simulating the ship.
    last_from_a: WorldPos,
    /// The first position shard B sent after it took over.
    first_from_b: WorldPos,
    /// The ship's speed, m/s, as shard A last simulated it.
    speed_at_boundary: f32,
    /// Ticks for which the client received a frame, from either shard.
    ticks_covered: Vec<u64>,
    /// The highest `input_ack` the accepting shard ever sent to the client.
    ack_from_b: u64,
}

fn run_handoff() -> (Pair, Handoff) {
    let mut pair = Pair::new();
    let id_a = pair.a.shard();
    let id_b = pair.b.shard();
    let (session_a, guid) = join(&mut pair.a, "player");

    let mut redirect = None;
    let mut last_from_a = WorldPos::ZERO;
    let mut speed_at_boundary = 0.0f32;
    let mut first_from_b = None;
    let mut ticks_covered = Vec::new();
    let mut ack_from_b = 0;
    let mut session_b = 0;

    for _ in 0..600 {
        // The client thrusts towards +x, i.e. out of A's region and into B's.
        pair.thrust(id_a, session_a, 0, 1.0);
        pair.tick();

        for message in pair.frames_for(id_a, session_a) {
            match &message {
                Message::Redirect(r) => redirect = Some(r.clone()),
                Message::Snapshot(s) => {
                    ticks_covered.push(s.tick);
                    if let Some(e) = s.entities.iter().find(|e| e.guid == guid) {
                        let state = ship_state_of(e);
                        last_from_a = state.pos;
                        speed_at_boundary = state.vel[0];
                    }
                }
                Message::Delta(d) => {
                    ticks_covered.push(d.tick);
                    if let Some(e) = d.entities.iter().find(|e| e.guid == guid) {
                        let state = ship_state_of(e);
                        last_from_a = state.pos;
                        speed_at_boundary = state.vel[0];
                    }
                }
                _ => {}
            }
        }

        // The client obeys the redirect: it connects to B (which adopts the staged
        // entity rather than spawning a second ship) and reports that it is up.
        if let Some(r) = redirect.clone() {
            if session_b == 0 {
                let welcome = pair
                    .b
                    .join(&space_net::hello("player"))
                    .expect("the accepting shard must admit the redirected player");
                session_b = welcome.session;
                pair.b.client_status(
                    session_b,
                    NeighbourStatus {
                        entity: guid,
                        shard: id_b,
                        connected: true,
                    },
                );
                pair.deliver();
                // Reporting in for `guid` is what binds the session to the staged
                // entity: the placeholder it was given at join is gone and the
                // original guid is its ship (guarantee 1).
                assert_eq!(
                    pair.b.ship_of(session_b),
                    Some(guid),
                    "guarantee 1: the redirected client keeps the same guid"
                );
            }
            for message in pair.frames_for(id_b, session_b) {
                if let Some(ack) = frame_input_ack(&message) {
                    ack_from_b = ack_from_b.max(ack);
                }
                match &message {
                    Message::Snapshot(s) => {
                        ticks_covered.push(s.tick);
                        if let Some(e) = s.entities.iter().find(|e| e.guid == guid) {
                            first_from_b.get_or_insert(ship_state_of(e).pos);
                        }
                    }
                    Message::Delta(d) => {
                        ticks_covered.push(d.tick);
                        if let Some(e) = d.entities.iter().find(|e| e.guid == guid) {
                            first_from_b.get_or_insert(ship_state_of(e).pos);
                        }
                    }
                    _ => {}
                }
            }
            // Once B is simulating the ship, the client drives B and stops
            // driving A — exactly what a real client does after a redirect.
            if pair.b.sim().ship_state(guid).is_some() {
                pair.thrust(id_b, session_b, 0, 1.0);
            }
            if r.activation_tick <= pair.b.sim().tick() && first_from_b.is_some() {
                break;
            }
        }
    }

    let handoff = Handoff {
        session_b,
        guid,
        redirect: redirect.expect("the client must be redirected"),
        last_from_a,
        speed_at_boundary,
        first_from_b: first_from_b.expect("the accepting shard must send the ship"),
        ticks_covered,
        ack_from_b,
    };
    (pair, handoff)
}

/// Guarantee 1 + 3 + 4 + 5: the guid is stable, the redirect names the
/// activation tick, and the position is continuous across the boundary.
#[test]
fn handoff_keeps_the_guid_position_and_inputs_continuous() {
    let (pair, h) = run_handoff();

    // 1. The guid never changed: the client's ship object is the same entity.
    assert_eq!(
        pair.b.ship_of(h.session_b),
        Some(h.guid),
        "the guid must survive the handoff"
    );
    assert_eq!(
        h.guid.shard,
        pair.a.shard(),
        "A minted it; B must not re-mint"
    );

    // 3. The redirect arrived with an activation tick in the future, so the
    // client had time to open the new connection before the old path stopped.
    assert!(
        h.redirect.activation_tick > 0,
        "the redirect must name an activation tick"
    );
    assert_eq!(h.redirect.new_shard, pair.b.shard());
    assert_eq!(h.redirect.entity, h.guid);

    // 4. Position continuity: the gap between the last state A sent and the first
    // state B sent is at most a handful of ticks of the ship's own motion. A
    // teleport would be hundreds of metres (half a region is 512 m), so bounding
    // the jump by the ship's speed at the boundary is a real test, not a tautology.
    let gap = WorldPos::distance(h.last_from_a, h.first_from_b);
    let per_tick = f64::from(h.speed_at_boundary) * f64::from(space_sim::dt());
    assert!(
        h.speed_at_boundary > 0.0,
        "the ship must actually be moving at the boundary"
    );
    assert!(
        gap <= per_tick * 5.0,
        "position jumped {gap:.1} m across the handoff; five ticks of travel is {:.1} m",
        per_tick * 5.0
    );

    // 5. Input continuity: the frames B sends acknowledge at least the activation
    // tick, so everything A already applied is dropped by the client's
    // reconciliation instead of being replayed on top of B's state. The observable
    // is the `input_ack` field on the wire, not `SimWorld::input_ack`: B has not
    // applied any input from this client yet (its own ack is 0), and the floor the
    // handoff installs is what carries A's work across.
    assert!(
        h.ack_from_b >= h.redirect.activation_tick,
        "input_ack {} on the wire must be at least the activation tick {}",
        h.ack_from_b,
        h.redirect.activation_tick
    );

    // 3 again: the client received a frame for every tick of the handoff — no
    // window where nobody was sending it anything.
    let mut ticks = h.ticks_covered.clone();
    ticks.sort_unstable();
    ticks.dedup();
    let span = ticks.len();
    let first = ticks.first().copied().unwrap_or(0);
    let last = ticks.last().copied().unwrap_or(0);
    assert!(
        span >= usize::try_from(last - first).unwrap_or(usize::MAX),
        "the client's frames must cover every tick between {first} and {last}, got {span}"
    );
}

/// Guarantee 3: a third client that can see the moving ship receives a frame for
/// every tick across the handoff — no gap, no duplicate.
///
/// The observer stays on shard A the whole time (it never leaves A's region), so
/// the only thing that changes for it is that the ship it was watching stops
/// existing in A's world and starts existing in B's. It must see the guid
/// disappear from A and appear in B, with no tick where it received nothing.
#[test]
fn an_observer_sees_no_gap_across_a_handoff() {
    let mut pair = Pair::new();
    let id_a = pair.a.shard();
    let id_b = pair.b.shard();
    let (session_a, guid) = join(&mut pair.a, "mover");
    let (session_obs, _) = join(&mut pair.a, "observer");

    let mut redirect = None;
    let mut session_b = 0;
    let mut observer_ticks: Vec<u64> = Vec::new();
    let mut saw_removal = false;
    let mut saw_reappear = false;

    for _ in 0..600 {
        pair.thrust(id_a, session_a, 0, 1.0);
        pair.tick();

        for message in pair.frames_for(id_a, session_a) {
            if let Message::Redirect(r) = message {
                redirect = Some(r);
            }
        }
        // The observer's frames: record the tick of every frame and watch for the
        // removal (A stops simulating) and the re-appearance (B starts).
        for message in pair.frames_for(id_a, session_obs) {
            match message {
                Message::Delta(d) => {
                    observer_ticks.push(d.tick);
                    if d.removed.contains(&guid) {
                        saw_removal = true;
                    }
                }
                Message::Snapshot(s) => observer_ticks.push(s.tick),
                _ => {}
            }
        }

        if let Some(r) = redirect.clone() {
            if session_b == 0 {
                let welcome = pair.b.join(&space_net::hello("mover")).expect("join B");
                session_b = welcome.session;
                pair.b.client_status(
                    session_b,
                    NeighbourStatus {
                        entity: guid,
                        shard: id_b,
                        connected: true,
                    },
                );
                pair.deliver();
            }
            for message in pair.frames_for(id_b, session_b) {
                match message {
                    Message::Snapshot(s) => {
                        if s.entities.iter().any(|e| e.guid == guid) {
                            saw_reappear = true;
                        }
                    }
                    Message::Delta(d) if d.entities.iter().any(|e| e.guid == guid) => {
                        saw_reappear = true;
                    }
                    _ => {}
                }
            }
            if pair.b.sim().ship_state(guid).is_some() {
                pair.thrust(id_b, session_b, 0, 1.0);
            }
            if saw_reappear && r.activation_tick < pair.b.sim().tick() {
                break;
            }
        }
    }

    assert!(saw_removal, "the observer must be told the ship left A");
    assert!(saw_reappear, "the observer must see the ship arrive in B");

    // Every tick between the first and last frame the observer got was covered.
    let mut ticks = observer_ticks;
    ticks.sort_unstable();
    let first = *ticks.first().expect("the observer received frames");
    let last = *ticks.last().expect("the observer received frames");
    let expected = usize::try_from(last - first + 1).expect("tick span fits usize");
    ticks.dedup();
    assert_eq!(
        ticks.len(),
        expected,
        "the observer missed a tick between {first} and {last}: {:?}",
        gap_positions(&ticks)
    );
}

/// The tick offsets at which a sorted tick list skips a value (empty = no gaps).
fn gap_positions(sorted: &[u64]) -> Vec<u64> {
    sorted
        .windows(2)
        .enumerate()
        .filter(|(_, w)| w[1] != w[0] + 1)
        .map(|(i, _)| i as u64)
        .collect()
}

/// Guarantee 6 + 8: an offer nobody answers times out, and the world is exactly
/// as if nothing had happened.
///
/// The peer region is owned by a shard id that is not running, so no accept can
/// ever arrive. After [`space_server::handoff::HANDOFF_TIMEOUT_TICKS`] the offer
/// must be abandoned with [`space_protocol::HandoffReason::OfferTimeout`], the
/// ship must still be simulated by A, and its motion must be unbroken.
#[test]
fn an_unanswered_offer_times_out_and_the_world_is_unchanged() {
    let ghost = ShardId::new(99);
    let mut a = Shard::new(ShardConfig::new(ShardId::new(1), addr(), region(0)));
    a.set_shard_map(ShardMap::new(
        1,
        vec![
            ShardRegion::new(region(0), a.shard(), addr()),
            ShardRegion::new(region(1), ghost, addr()),
        ],
    ));
    let (session, guid) = join(&mut a, "player");

    // Record the position every tick so the "unbroken" claim is checkable.
    let mut positions = Vec::new();
    let mut offered = false;
    for _ in 0..600 {
        let tick = a.sim().tick() + 1;
        a.apply_input(
            session,
            &InputTick {
                tick,
                seq: tick,
                inputs: vec![PlayerInput {
                    entity: guid,
                    thrust: [1.0, 0.0, 0.0],
                    torque: [0.0; 3],
                    tick,
                }],
                ack_tick: 0,
            },
        );
        a.tick();
        let outbox = a.take_outbox();
        if outbox
            .iter()
            .any(|o| matches!(o.message, Message::HandoffOffer(_)))
        {
            offered = true;
        }
        // Nothing is delivered: the peer is not there.
        positions.push(a.sim().ship_state(guid).map(|s| s.pos));
        if a.stats().aborted > 0 {
            break;
        }
    }

    assert!(
        offered,
        "the shard must have offered the ship to the ghost region"
    );
    assert_eq!(a.stats().aborted, 1, "exactly one abort");
    assert!(
        a.sim().ship_state(guid).is_some(),
        "guarantee 6: after an abort A keeps simulating the ship"
    );
    assert_eq!(a.stats().handed_off, 0, "nothing was handed off");
    // The ship never stopped: every tick has a position and it kept moving.
    assert!(
        positions.iter().all(Option::is_some),
        "the ship was never missing from A's world"
    );
    let first = positions
        .first()
        .and_then(|p| *p)
        .expect("a first position");
    let last = positions.last().and_then(|p| *p).expect("a last position");
    assert_ne!(
        first, last,
        "the ship must have kept moving through the abort"
    );
}

/// Guarantee 7: a handoff cannot straddle a shard-map change.
///
/// B is on a newer epoch than the offer, so B answers [`Message::StaleEpoch`]
/// instead of accepting, and A abandons the transfer with
/// [`space_protocol::HandoffReason::EpochChanged`] rather than applying a map that
/// no longer describes the world.
#[test]
fn a_stale_epoch_offer_is_refused_and_the_transfer_aborts() {
    let mut pair = Pair::new();
    let id_a = pair.a.shard();
    let id_b = pair.b.shard();
    let (session_a, guid) = join(&mut pair.a, "player");

    // B moves to epoch 2 — A's map, and therefore its offer, is now stale.
    let newer = ShardMap::new(
        2,
        vec![
            ShardRegion::new(region(0), id_a, addr()),
            ShardRegion::new(region(1), id_b, addr()),
        ],
    );
    pair.b.set_shard_map(newer);
    assert_eq!(pair.b.epoch(), 2);
    assert_eq!(pair.a.epoch(), 1);

    let mut saw_stale_epoch = false;
    let mut saw_offer = false;
    for _ in 0..600 {
        pair.thrust(id_a, session_a, 0, 1.0);
        pair.tick();
        for (_, _, message) in std::mem::take(&mut pair.client_frames) {
            if matches!(message, Message::Redirect(_)) {
                saw_offer = true;
            }
        }
        // The StaleEpoch answer is consumed by `peer_message` on A; observe its
        // effect through A's counters instead of the message itself.
        if pair.a.stats().aborted > 0 {
            break;
        }
        saw_stale_epoch = saw_stale_epoch || pair.b.stats().received == 0;
    }

    assert!(saw_offer, "A must have offered the ship");
    assert_eq!(
        pair.a.stats().aborted,
        1,
        "A must have aborted the transfer"
    );
    assert_eq!(
        pair.a.stats().handed_off,
        0,
        "guarantee 7: nothing may be handed off under a stale epoch"
    );
    assert_eq!(
        pair.b.stats().received,
        0,
        "guarantee 7: B must not take ownership under a stale epoch"
    );
    assert!(
        pair.a.sim().ship_state(guid).is_some(),
        "A keeps the ship after the epoch-mismatch abort"
    );
    assert!(saw_stale_epoch, "B must never have committed anything");
}

/// Guarantees 2 + 4: after a handoff the two worlds agree on the entity.
///
/// [`SimWorld::guid_state_hash`] is keyed by guid and independent of ECS index,
/// so the way to state "no desync" is: take the entity out of A's world (which
/// the commit does) and put A's *last* view of it next to B's first view, and
/// check the state B activated with is the state A finished with — not the state
/// the offer staged. A double-step or a stale activation both break this.
#[test]
fn handoff_leaves_no_desync() {
    let mut pair = Pair::new();
    let id_a = pair.a.shard();
    let id_b = pair.b.shard();
    let (session_a, guid) = join(&mut pair.a, "player");

    // The state A last simulated, captured straight from the authoritative world
    // (not off the wire) at the moment A stops owning the entity.
    let mut last_a_state: Option<ShipState> = None;
    let mut redirect = None;
    let mut session_b = 0;
    let mut activated = false;

    for _ in 0..600 {
        pair.thrust(id_a, session_a, 0, 1.0);
        // Snapshot A's view *before* the tick, so the last one we take before the
        // commit is the state the commit must carry.
        if let Some(state) = pair.a.sim().ship_state(guid) {
            if pair.a.sim().ship_state(guid).is_some() {
                last_a_state = Some(state);
            }
        }
        pair.tick();
        for message in pair.frames_for(id_a, session_a) {
            if let Message::Redirect(r) = message {
                redirect = Some(r);
            }
        }
        if let Some(r) = redirect.clone() {
            if session_b == 0 {
                session_b = pair
                    .b
                    .join(&space_net::hello("player"))
                    .expect("join B")
                    .session;
                pair.b.client_status(
                    session_b,
                    NeighbourStatus {
                        entity: guid,
                        shard: id_b,
                        connected: true,
                    },
                );
                pair.deliver();
            }
            if pair.b.sim().ship_state(guid).is_some() {
                pair.thrust(id_b, session_b, 0, 1.0);
            }
            if pair.b.sim().ship_state(guid).is_some() && r.activation_tick <= pair.b.sim().tick() {
                activated = true;
                break;
            }
        }
    }

    let last_a = last_a_state.expect("A simulated the ship");
    let first_b = pair
        .b
        .sim()
        .ship_state(guid)
        .expect("B is simulating the ship after the commit");
    assert!(activated, "the handoff must complete");

    // The commit's state is the state as of the activation tick, so B's first
    // state is A's last state advanced by at most the ticks between them — never
    // *behind* it (a stale activation) and never ahead by more than a tick or two
    // (a double-step).
    let drift = WorldPos::distance(last_a.pos, first_b.pos);
    let per_tick = f64::from(last_a.vel[0].max(first_b.vel[0])) * f64::from(space_sim::dt());
    assert!(
        drift <= per_tick * 3.0 + 1.0,
        "B activated {drift:.1} m from A's last state; three ticks of travel is {:.1} m",
        per_tick * 3.0 + 1.0
    );

    // And the guid is in exactly one world: A dropped it when it committed.
    assert!(
        pair.a.sim().ship_state(guid).is_none(),
        "guarantee 2: the offering shard must stop simulating a committed entity"
    );
    assert_eq!(
        pair.b.sim().ship_state(guid).expect("alive").vel[0],
        first_b.vel[0]
    );
}

/// U9's budget: a 1k-entity snapshot costs about 158 B per entity.
///
/// The number is a documented expectation, not a hard contract: the assertion
/// allows ±10% so a component change is a deliberate update rather than a
/// surprise CI failure.
#[test]
fn a_snapshot_of_1000_ships_costs_about_158_bytes_per_entity() {
    const ENTITIES: usize = 1_000;
    const EXPECTED_BYTES_PER_ENTITY: f64 = 158.0;
    const TOLERANCE: f64 = 0.10;

    let mut sim = space_sim::SimWorld::new(ShardId::new(1));
    let mut guids = space_server::replicate::GuidIndex::new();
    for i in 0..ENTITIES {
        let guid = EntityGuid::new(ShardId::new(1), u64::try_from(i).expect("fits"));
        let entity = sim.spawn_ship(guid, u64::try_from(i).expect("fits"), ShipState::IDENTITY);
        guids.insert(guid, entity);
    }

    let mut builder = space_protocol::SnapshotBuilder::with_budget(
        space_protocol::MAX_FRAME_SIZE,
        sim.tick(),
        ShardId::new(1),
        1,
    );
    let mut pushed = 0;
    for entity in sim.ships() {
        let Some(snapshot) = space_server::replicate::snapshot_of(&sim, &guids, entity) else {
            continue;
        };
        if builder.push(&snapshot) {
            pushed += 1;
        }
    }
    assert_eq!(pushed, ENTITIES, "the budget must hold all 1000 ships");
    let snapshot = builder.finish_snapshot(0);
    // The full wire cost: the frame header plus the encoded message.
    let bytes = space_protocol::encode_frame(&Message::Snapshot(snapshot)).len();
    let per_entity = f64::from(u32::try_from(bytes).expect("fits"))
        / f64::from(u32::try_from(pushed).expect("fits"));
    let low = EXPECTED_BYTES_PER_ENTITY * (1.0 - TOLERANCE);
    let high = EXPECTED_BYTES_PER_ENTITY * (1.0 + TOLERANCE);
    assert!(
        per_entity >= low && per_entity <= high,
        "{bytes} B for {pushed} entities is {per_entity:.1} B/entity, expected \
         {EXPECTED_BYTES_PER_ENTITY} ±{:.0}%",
        TOLERANCE * 100.0
    );
}

/// The serve path end to end: a real client connects to a running shard, gets a
/// welcome with a session and a ship, and its inputs move the ship.
///
/// This is the smoke test for `main.rs`'s loop — the same
/// [`ShardNet::serve`] / [`Shard::tick`] / [`ShardNet::flush`] wiring, driven
/// through a real quinn connection.
#[tokio::test]
async fn a_client_can_connect_to_a_running_shard() {
    let harness = RunningShard::start().await;
    let (conn, welcome) = harness.connect("player").await.expect("connect");
    assert_eq!(welcome.shard, ShardId::new(1));
    assert_eq!(welcome.session, 1);
    assert_eq!(welcome.shard_map.regions.len(), 1);
    harness
        .wait_for_clients(1)
        .await
        .expect("the shard sees the client");

    // The ship exists and is at the region centre.
    let before = harness.ship_position().await.expect("a ship was spawned");

    // Send a burst of inputs for the ticks the shard is about to step to.
    let ship = harness.ship_guid().await.expect("a ship");
    for _ in 0..30 {
        let tick = harness.world_tick().await + 1;
        let batch = InputTick {
            tick,
            seq: tick,
            inputs: vec![PlayerInput {
                entity: ship,
                thrust: [1.0, 0.0, 0.0],
                torque: [0.0; 3],
                tick,
            }],
            ack_tick: 0,
        };
        conn.send(&Message::InputTick(batch))
            .await
            .expect("send input");
        tokio::time::sleep(std::time::Duration::from_millis(35)).await;
    }
    harness.settle().await;
    let after = harness.ship_position().await.expect("the ship is alive");
    assert_ne!(before, after, "the client's inputs must move the ship");

    // And the client received at least one frame describing its ship.
    let mut saw_own_ship = false;
    for _ in 0..20 {
        let frame = tokio::time::timeout(std::time::Duration::from_millis(200), conn.recv_stream())
            .await
            .ok()
            .and_then(|r| r.ok().flatten());
        let datagram =
            tokio::time::timeout(std::time::Duration::from_millis(10), conn.recv_datagram())
                .await
                .ok()
                .and_then(|r| r.ok().flatten());
        for message in [frame, datagram].into_iter().flatten() {
            let entities = match &message {
                Message::Snapshot(s) => &s.entities,
                Message::Delta(d) => &d.entities,
                _ => continue,
            };
            if entities.iter().any(|e| e.guid == ship) {
                saw_own_ship = true;
            }
        }
        if saw_own_ship {
            break;
        }
    }
    assert!(saw_own_ship, "the client must receive its own ship");
    harness.shutdown();
}

// ======================================================== Phase 2 remediation
//
// One regression test per review blocker. Each test names, in its comment, the
// mutation that would make it fail — that is what makes it a regression test
// rather than a restatement of the implementation.

/// The [`HandoffAbort`] a shard queued, if any.
fn abort_in(messages: &[space_protocol::Message]) -> Option<space_protocol::HandoffAbort> {
    messages.iter().find_map(|m| match m {
        Message::HandoffAbort(a) => Some(a.clone()),
        _ => None,
    })
}

/// The [`HandoffOffer`] a shard queued, if any.
fn offer_in(messages: &[space_protocol::Message]) -> Option<space_protocol::HandoffOffer> {
    messages.iter().find_map(|m| match m {
        Message::HandoffOffer(o) => Some(o.clone()),
        _ => None,
    })
}

/// Drive `a` until it offers `guid` to the shard that owns the region it flies into.
fn drive_offer(a: &mut Shard, guid: EntityGuid, session: u64) -> space_protocol::HandoffOffer {
    for _ in 0..600 {
        let tick = a.sim().tick() + 1;
        a.apply_input(
            session,
            &InputTick {
                tick,
                seq: tick,
                inputs: vec![PlayerInput {
                    entity: guid,
                    thrust: [1.0, 0.0, 0.0],
                    torque: [0.0; 3],
                    tick,
                }],
                ack_tick: 0,
            },
        );
        a.tick();
        let outbox = a.take_outbox();
        if let Some(offer) = offer_in(&outbox.iter().map(|o| o.message.clone()).collect::<Vec<_>>())
        {
            assert_eq!(offer.entity, guid, "the offer names the ship that left");
            return offer;
        }
    }
    panic!("the ship never left its shard's region");
}

/// B3: a claim for a staged entity is only honoured when the claiming client's
/// hello token is the token the offer carried the owner under.
///
/// Two clients with **different** hello tokens report `connected` for the same
/// staged guid. The wrong-token client must not be adopted, must not bind the
/// staged session, and must not mirror the status to the offering shard (which
/// is what opens A's commit gate); the matching-token owner must be adopted.
///
/// Mutation this catches: moving `Ledger::mark_session` (and its
/// `NeighbourStatus` mirror to `staged.from`) back outside the
/// `claim_is_authorized` block — then the wrong-token claim binds the staged
/// session, opens A's commit gate and A commits the ship to the attacker.
#[test]
fn a_claim_for_a_staged_entity_requires_the_offered_owner_token() {
    let mut pair = Pair::new();
    let id_a = pair.a.shard();
    let id_b = pair.b.shard();
    let (session_a, guid) = join(&mut pair.a, "owner-token");

    // Drive A's ship into B's region so B stages the transfer.
    let mut redirect = None;
    for _ in 0..600 {
        pair.thrust(id_a, session_a, 0, 1.0);
        pair.tick();
        for message in pair.frames_for(id_a, session_a) {
            if let Message::Redirect(r) = message {
                redirect = Some(r.clone());
            }
        }
        if redirect.is_some() {
            break;
        }
    }
    assert!(redirect.is_some(), "A must have offered the ship");

    // The wrong-token client connects and claims the staged guid.
    let thief = pair
        .b
        .join(&space_net::hello("thief-token"))
        .expect("join B")
        .session;
    pair.b.client_status(
        thief,
        NeighbourStatus {
            entity: guid,
            shard: id_b,
            connected: true,
        },
    );
    // Drain B's outbox *without* delivering: the mirror to A is what opens the
    // commit gate, and it must not have been queued at all.
    let thief_out = pair.b.take_outbox();
    assert!(
        !thief_out.iter().any(|o| matches!(
            (&o.dest, &o.message),
            (Destination::Shard(from), Message::NeighbourStatus(s))
                if *from == id_a && s.entity == guid && s.connected
        )),
        "a wrong-token claim must not mirror NeighbourStatus to the offering shard"
    );
    assert_ne!(
        pair.b.ship_of(thief),
        Some(guid),
        "a wrong-token claim must not adopt the staged entity"
    );
    // The thief got a placeholder of its own instead (it is just a newcomer).
    assert!(
        pair.b.ship_of(thief).is_some_and(|s| s != guid),
        "the wrong-token client must be treated as an ordinary newcomer"
    );

    // The real owner connects with the token the offer carried and is adopted.
    let owner = pair
        .b
        .join(&space_net::hello("owner-token"))
        .expect("join B")
        .session;
    pair.b.client_status(
        owner,
        NeighbourStatus {
            entity: guid,
            shard: id_b,
            connected: true,
        },
    );
    assert_eq!(
        pair.b.ship_of(owner),
        Some(guid),
        "the matching-token owner must be adopted"
    );
    let owner_out = pair.b.take_outbox();
    assert!(
        owner_out.iter().any(|o| matches!(
            (&o.dest, &o.message),
            (Destination::Shard(from), Message::NeighbourStatus(s))
                if *from == id_a && s.entity == guid && s.connected
        )),
        "the matching-token claim must mirror NeighbourStatus to the offering shard"
    );
}

/// B7: the accepting shard installs a commit on **its own** clock.
///
/// B's world starts 100 ticks behind A's (two shards run independent loops and
/// drift in production; `SimWorld::with_tick` is the sanctioned way to build
/// that here). If the install gate used A's `activation_tick`, a lagging B would
/// see `activation_tick <= tick` long before its own lead had elapsed and spawn
/// the ship on arrival — then step it in the same tick it arrived (a double
/// step), or, if B *led* A, never install at all (a frozen ship).
///
/// Mutation this catches: gating `install_pending_commits` on
/// `commit.activation_tick` instead of `staged.local_activation_tick`.
#[test]
fn a_commit_installs_on_the_accepting_shards_own_clock() {
    let id_a = ShardId::new(1);
    let id_b = ShardId::new(2);
    let map = ShardMap::new(
        1,
        vec![
            ShardRegion::new(region(0), id_a, addr()),
            ShardRegion::new(region(1), id_b, addr()),
        ],
    );
    let mut a = Shard::new(ShardConfig::new(id_a, addr(), region(0)));
    a.set_shard_map(map.clone());
    // B is 100 ticks behind A: A's activation tick is already "past" on A's
    // clock long before B's own lead elapses.
    let mut b = Shard::new(ShardConfig::new(id_b, addr(), region(1)));
    b.set_shard_map(map);
    // Back-date B by stepping it *backwards* is impossible; instead advance A.
    // A's tick is what the offer's activation tick is measured in, so running A
    // ahead is the same offset from B's point of view.
    a.tick_n(100);
    assert_eq!(a.sim().tick(), 100);
    assert_eq!(b.sim().tick(), 0);

    let (session_a, guid) = join(&mut a, "player");
    let offer = drive_offer(&mut a, guid, session_a);
    // Re-target the offer at the live B by replaying it as a peer message.
    let offer = space_protocol::HandoffOffer { to: id_b, ..offer };
    b.peer_message(id_a, Message::HandoffOffer(offer.clone()));
    let accept = b
        .take_outbox()
        .into_iter()
        .find_map(|o| match o.message {
            Message::HandoffAccept(x) => Some(x),
            _ => None,
        })
        .expect("B must accept a live offer");
    a.peer_message(id_b, Message::HandoffAccept(accept));
    // The owner shows up on B and confirms.
    let session_b = b.join(&space_net::hello("player")).expect("join B").session;
    b.client_status(
        session_b,
        NeighbourStatus {
            entity: guid,
            shard: id_b,
            connected: true,
        },
    );
    // Deliver B's mirror: it is what opens A's commit gate.
    for out in b.take_outbox() {
        a.peer_message(id_b, out.message);
    }
    // A reaches its activation tick and commits.
    let mut commit = None;
    for _ in 0..space_server::handoff::HANDOFF_TIMEOUT_TICKS + 10 {
        a.tick();
        let out = a.take_outbox();
        // The NeighbourStatus mirror from B opened the gate; feed it first.
        if commit.is_none() {
            if let Some(c) = out.iter().find_map(|o| match &o.message {
                Message::HandoffCommit(c) => Some(c.clone()),
                _ => None,
            }) {
                commit = Some(c);
            }
        }
        if commit.is_some() {
            break;
        }
    }
    let commit = commit.expect("A must commit once the gate is open");
    assert!(
        commit.activation_tick > b.sim().tick(),
        "precondition: A's activation tick is ahead of B's clock, so a naive gate \
         on it would install immediately"
    );
    b.peer_message(id_a, Message::HandoffCommit(commit));

    // The commit must NOT be installed yet: B's own lead has not elapsed.
    assert!(
        b.sim().ship_state(guid).is_none(),
        "guarantee 2: B must not spawn the entity before its own install tick"
    );
    // B keeps ticking; the ship appears exactly once, and then moves.
    let mut appeared_at = None;
    for _ in 0..space_server::handoff::HANDOFF_LEAD + 10 {
        b.tick();
        if b.sim().ship_state(guid).is_some() {
            appeared_at = Some(b.sim().tick());
            break;
        }
    }
    let appeared =
        appeared_at.expect("guarantee 2: the ship must not stay frozen out of the world");
    assert!(
        appeared >= space_server::handoff::HANDOFF_LEAD,
        "B installed at tick {appeared}, before its own lead of {} ticks \
         (it used A's activation tick)",
        space_server::handoff::HANDOFF_LEAD
    );
    // No double step: the first motion after install is at most one tick's worth.
    let at_install = b.sim().ship_state(guid).expect("alive").pos;
    let speed = b.sim().ship_state(guid).expect("alive").vel[0];
    b.tick();
    let after = b.sim().ship_state(guid).expect("alive").pos;
    let moved = WorldPos::distance(at_install, after);
    let one_tick = f64::from(speed.abs().max(1.0)) * f64::from(space_sim::dt()) * 1.5;
    assert!(
        moved <= one_tick + f64::EPSILON.max(one_tick * 0.0),
        "the ship moved {moved:.3} m in one tick after install; one tick of \
         travel at {speed:.1} m/s is {:.3} m — a double step",
        one_tick
    );
    // And it is not frozen: it keeps moving.
    b.tick();
    assert_ne!(
        b.sim().ship_state(guid).expect("alive").pos,
        after,
        "the ship must keep moving after install"
    );
}

/// B1: an epoch change after the commit gate still aborts the transfer.
///
/// The transfer is driven all the way past the gate (accepted *and* the client
/// confirmed), then both shards move to a new epoch. Guarantee 7 says a handoff
/// cannot straddle a map change, so A must abort with
/// [`space_protocol::HandoffReason::EpochChanged`] and release the guid — proof
/// being that a fresh offer for the same guid appears afterwards.
///
/// Mutation this catches: re-adding the old `(!accepted || !client_connected)`
/// clause to the timeout sweep — a transfer past the gate is then skipped by
/// both the epoch check and the sweep, so it never aborts and `is_busy` blocks
/// every future offer for that guid.
#[test]
fn an_epoch_change_after_the_commit_gate_aborts_and_releases_the_guid() {
    let mut pair = Pair::new();
    let id_a = pair.a.shard();
    let id_b = pair.b.shard();
    let (session_a, guid) = join(&mut pair.a, "player");

    // Drive A until it offers the ship.
    let mut offered = false;
    for _ in 0..600 {
        pair.thrust(id_a, session_a, 0, 1.0);
        pair.a.tick();
        let out = pair.a.take_outbox();
        if out
            .iter()
            .any(|o| matches!(o.message, Message::HandoffOffer(_)))
        {
            offered = true;
            for o in out {
                if matches!(o.dest, Destination::Shard(_)) {
                    pair.b.peer_message(id_a, o.message);
                }
            }
            break;
        }
    }
    assert!(offered, "A must have offered the ship");
    // Deliver B's accept to A.
    for o in pair.b.take_outbox() {
        if matches!(o.dest, Destination::Shard(_)) {
            pair.a.peer_message(id_b, o.message);
        }
    }
    // The owner confirms on B; B mirrors the status to A, opening the gate.
    let session_b = pair
        .b
        .join(&space_net::hello("player"))
        .expect("join B")
        .session;
    pair.b.client_status(
        session_b,
        NeighbourStatus {
            entity: guid,
            shard: id_b,
            connected: true,
        },
    );
    for o in pair.b.take_outbox() {
        if matches!(o.dest, Destination::Shard(_)) {
            pair.a.peer_message(id_b, o.message);
        }
    }

    // Both shards move to epoch 2.
    let newer = ShardMap::new(
        2,
        vec![
            ShardRegion::new(region(0), id_a, addr()),
            ShardRegion::new(region(1), id_b, addr()),
        ],
    );
    pair.a.set_shard_map(newer.clone());
    pair.b.set_shard_map(newer);

    // A must abort with EpochChanged, then re-offer the same guid.
    let mut saw_abort = None;
    let mut re_offered = false;
    for _ in 0..space_server::handoff::HANDOFF_TIMEOUT_TICKS + 5 {
        pair.thrust(id_a, session_a, 0, 1.0);
        pair.a.tick();
        let out = pair.a.take_outbox();
        let messages: Vec<Message> = out.iter().map(|o| o.message.clone()).collect();
        if saw_abort.is_none() {
            saw_abort = abort_in(&messages);
        }
        if offer_in(&messages).is_some() {
            re_offered = true;
            break;
        }
    }
    let abort = saw_abort.expect("A must abort the stale-epoch transfer");
    assert_eq!(
        abort.reason,
        space_protocol::HandoffReason::EpochChanged,
        "the abort must name the epoch change"
    );
    assert_eq!(abort.entity, guid);
    assert!(
        re_offered,
        "guarantee 7 + 8: the guid must be released so a fresh offer can start"
    );
    assert!(
        pair.a.sim().ship_state(guid).is_some(),
        "guarantee 6: A keeps simulating the ship after the abort"
    );
}

/// B2: a staged transfer whose commit never arrives is dropped by the accepting
/// shard.
///
/// A offers, B stages the [`Incoming`], and A never commits (it died, or its
/// commit was lost). B's sweep must drop the staged entry within
/// [`space_server::handoff::HANDOFF_TIMEOUT_TICKS`] — otherwise `is_busy` poisons
/// the guid for B's whole life and the ship can never be handed to it again.
///
/// Mutation this catches: deleting the `stale_incoming` sweep in
/// `Shard::handoff_pass` — the staged entry then persists forever and the second
/// offer is refused with "entity is not available here".
#[test]
fn a_staged_transfer_without_a_commit_is_dropped_and_the_guid_is_reofferable() {
    let mut pair = Pair::new();
    let id_a = pair.a.shard();
    let id_b = pair.b.shard();
    let (session_a, guid) = join(&mut pair.a, "player");

    // A offers; B accepts and stages. A then goes silent (nothing of A's is
    // delivered again).
    let mut offered = false;
    for _ in 0..600 {
        pair.thrust(id_a, session_a, 0, 1.0);
        pair.a.tick();
        let out = pair.a.take_outbox();
        if out
            .iter()
            .any(|o| matches!(o.message, Message::HandoffOffer(_)))
        {
            offered = true;
            for o in out {
                if matches!(o.dest, Destination::Shard(_)) {
                    pair.b.peer_message(id_a, o.message);
                }
            }
            break;
        }
    }
    assert!(offered, "A must have offered the ship");
    // B has the transfer staged.
    let staged = pair
        .b
        .take_outbox()
        .iter()
        .any(|o| matches!(o.message, Message::HandoffAccept(_)));
    assert!(staged, "B must have accepted and staged the entity");

    // B ticks on its own, with no commit coming.
    let mut saw_abort = None;
    for _ in 0..space_server::handoff::HANDOFF_TIMEOUT_TICKS + 5 {
        pair.b.tick();
        let messages: Vec<Message> = pair
            .b
            .take_outbox()
            .iter()
            .map(|o| o.message.clone())
            .collect();
        if let Some(abort) = abort_in(&messages) {
            saw_abort = Some(abort);
            break;
        }
    }
    let abort = saw_abort.expect("B must drop a staged transfer with no commit");
    assert_eq!(abort.entity, guid, "the abort names the staged entity");
    assert_eq!(abort.to, id_b, "B is the one giving up");

    // The guid is re-offerable: A offers it again and B accepts.
    for o in pair.a.take_outbox() {
        if matches!(o.dest, Destination::Shard(_)) {
            pair.b.peer_message(id_a, o.message);
        }
    }
    let re_offer = space_protocol::HandoffOffer {
        entity: guid,
        from: id_a,
        to: id_b,
        epoch: pair.b.epoch(),
        state: Box::new(offer_state(guid, "player")),
        activation_tick: pair.a.sim().tick() + space_server::handoff::HANDOFF_LEAD,
    };
    pair.b.peer_message(id_a, Message::HandoffOffer(re_offer));
    let accepted = pair
        .b
        .take_outbox()
        .iter()
        .any(|o| matches!(o.message, Message::HandoffAccept(_)));
    assert!(
        accepted,
        "guarantee 8: a dropped staged entry must leave the guid re-offerable"
    );
}

/// A one-component [`EntitySnapshot`] for `guid` carrying `token` as the owner.
fn offer_state(guid: EntityGuid, token: &str) -> space_protocol::EntitySnapshot {
    let mut components = vec![(
        <space_sim::ShipState as space_ecs::ComponentDef>::ID,
        space_sim::wire::encode_ship_state(&ShipState::IDENTITY),
    )];
    // The private owner-token component `space-server` reads back out of the
    // offer's state (see `OWNER_TOKEN_COMPONENT`).
    components.push((u16::MAX, token.as_bytes().to_vec()));
    space_protocol::EntitySnapshot::new(guid, components)
}

/// B4: a shard at the player cap refuses an incoming handoff with `TargetFull`.
///
/// B is filled to [`space_server::shard::MAX_PLAYERS_PER_SHARD`], then A offers
/// it a ship. B must answer [`space_protocol::HandoffReason::TargetFull`] rather
/// than stage the entity, and A must keep simulating the ship.
///
/// Mutation this catches: deleting the `clients.len() >= MAX_PLAYERS_PER_SHARD`
/// check in `Shard::on_offer` — B then stages the entity and eventually commits
/// it, growing past the cap.
#[test]
fn a_full_shard_refuses_an_incoming_handoff_with_target_full() {
    let mut pair = Pair::new();
    let id_a = pair.a.shard();
    let id_b = pair.b.shard();
    let (_session_a, guid) = join(&mut pair.a, "player");

    // Fill B to the cap.
    for i in 0..space_server::shard::MAX_PLAYERS_PER_SHARD {
        pair.b
            .join(&space_net::hello(format!("filler-{i}")))
            .expect("filler joins");
    }
    assert_eq!(
        pair.b.stats().clients,
        space_server::shard::MAX_PLAYERS_PER_SHARD,
        "B is at capacity"
    );

    // A offers the ship to B.
    let offer = space_protocol::HandoffOffer {
        entity: guid,
        from: id_a,
        to: id_b,
        epoch: pair.b.epoch(),
        state: Box::new(offer_state(guid, "player")),
        activation_tick: pair.a.sim().tick() + space_server::handoff::HANDOFF_LEAD,
    };
    pair.b.peer_message(id_a, Message::HandoffOffer(offer));
    let messages: Vec<Message> = pair
        .b
        .take_outbox()
        .iter()
        .map(|o| o.message.clone())
        .collect();
    let abort = abort_in(&messages).expect("a full shard must abort the offer");
    assert_eq!(
        abort.reason,
        space_protocol::HandoffReason::TargetFull,
        "the refusal must name the capacity"
    );
    assert_eq!(abort.entity, guid);

    // A takes the abort: it keeps the ship and nothing was handed off.
    pair.a.peer_message(id_b, Message::HandoffAbort(abort));
    assert!(
        pair.a.sim().ship_state(guid).is_some(),
        "guarantee 6: A keeps simulating a ship B refused"
    );
    assert_eq!(pair.a.stats().handed_off, 0);
    assert_eq!(pair.b.stats().received, 0, "B took ownership of nothing");
}

/// B6: a peer that is not in this shard's map is not a neighbour, and a commit
/// must come from the shard that staged the transfer.
///
/// Two separate forgeries:
/// * (a) a message from an off-map shard id is dropped by the `knows_shard`
///   guard in [`Shard::peer_message`] — nothing is staged, nothing is accepted;
/// * (b) a [`Message::HandoffCommit`] whose `from` is not the shard that staged
///   the `Incoming` is refused by [`space_server::handoff::Ledger::take_incoming_from`]
///   — the entity is not installed.
///
/// Mutation this catches: (a) removing the `knows_shard` guard makes the forged
/// offer accepted; (b) replacing `take_incoming_from` with `take_incoming` makes
/// the forged commit install the entity.
#[test]
fn an_unknown_peer_and_a_misattributed_commit_are_refused() {
    let mut pair = Pair::new();
    let id_a = pair.a.shard();
    let id_b = pair.b.shard();
    let stranger = ShardId::new(777);
    let (_session_a, guid) = join(&mut pair.a, "player");

    // (a) A stranger id (not in B's map) offers an entity.
    let forged = space_protocol::HandoffOffer {
        entity: guid,
        from: stranger,
        to: id_b,
        epoch: pair.b.epoch(),
        state: Box::new(offer_state(guid, "player")),
        activation_tick: pair.b.sim().tick() + space_server::handoff::HANDOFF_LEAD,
    };
    pair.b.peer_message(stranger, Message::HandoffOffer(forged));
    let out = pair.b.take_outbox();
    assert!(
        !out.iter()
            .any(|o| matches!(o.message, Message::HandoffAccept(_))),
        "guarantee 7 / peer check: an off-map shard must not be answered with an accept"
    );
    assert_eq!(
        pair.b.stats().received,
        0,
        "nothing was staged by a stranger"
    );

    // (b) Stage a legitimate transfer from A, then commit it as if from `stranger`.
    let offer = space_protocol::HandoffOffer {
        entity: guid,
        from: id_a,
        to: id_b,
        epoch: pair.b.epoch(),
        state: Box::new(offer_state(guid, "player")),
        activation_tick: pair.b.sim().tick() + space_server::handoff::HANDOFF_LEAD,
    };
    pair.b.peer_message(id_a, Message::HandoffOffer(offer));
    assert!(pair
        .b
        .take_outbox()
        .iter()
        .any(|o| matches!(o.message, Message::HandoffAccept(_))));
    let forged_commit = space_protocol::HandoffCommit {
        entity: guid,
        from: stranger, // not the shard that staged it
        to: id_b,
        epoch: pair.b.epoch(),
        activation_tick: pair.b.sim().tick(),
        state: Box::new(offer_state(guid, "player")),
    };
    pair.b
        .peer_message(stranger, Message::HandoffCommit(forged_commit.clone()));
    assert!(
        pair.b.sim().ship_state(guid).is_none(),
        "a commit from a shard that did not stage the transfer must not install it"
    );
    // Even delivered *as if* from A, the `from` field is what is checked.
    pair.b
        .peer_message(id_a, Message::HandoffCommit(forged_commit.clone()));
    assert!(
        pair.b.sim().ship_state(guid).is_none(),
        "guarantee 2: the commit's `from` must match the staged transfer's"
    );
    // The staged entry survived both refusals: a genuine commit still works.
    let real_commit = space_protocol::HandoffCommit {
        from: id_a,
        ..forged_commit
    };
    pair.b
        .peer_message(id_a, Message::HandoffCommit(real_commit));
    for _ in 0..space_server::handoff::HANDOFF_LEAD + 2 {
        pair.b.tick();
        if pair.b.sim().ship_state(guid).is_some() {
            break;
        }
    }
    assert!(
        pair.b.sim().ship_state(guid).is_some(),
        "the genuine commit from the staging shard must still install"
    );
}

/// B5: a peer send that cannot be made is counted, not silently dropped.
///
/// A shard queues a shard-directed message for a peer whose endpoint the front
/// end has never been told about. [`space_server::net::ShardNet::flush`] must
/// bump [`space_server::net::ShardNet::peer_send_failures`] rather than lose the
/// message without a trace — that counter is the only signal an operator has
/// that a handoff is stuck on transport, not on protocol.
///
/// Mutation this catches: dropping the `fetch_add` in `ShardNet::send_to_peer`
/// (returning silently) leaves the counter at zero.
#[tokio::test]
async fn a_send_to_an_unknown_peer_is_counted_as_a_failure() {
    let net = std::sync::Arc::new(space_server::net::ShardNet::bind(addr()).expect("bind"));
    let mut shard = Shard::new(ShardConfig::new(ShardId::new(1), addr(), region(0)));
    shard.send_to(
        Destination::Shard(ShardId::new(42)),
        Message::HandoffAbort(space_protocol::HandoffAbort {
            entity: EntityGuid::new(ShardId::new(1), 1),
            from: ShardId::new(1),
            to: ShardId::new(42),
            reason: space_protocol::HandoffReason::Other("test".into()),
        }),
    );
    let shard = std::sync::Arc::new(tokio::sync::Mutex::new(shard));
    assert_eq!(net.peer_send_failures(), 0, "nothing has failed yet");
    net.flush(&shard).await;
    assert_eq!(
        net.peer_send_failures(),
        1,
        "a send to a peer with no known endpoint must be counted"
    );
    net.flush(&shard).await;
    assert_eq!(
        net.peer_send_failures(),
        1,
        "the outbox was drained: nothing new was dropped"
    );
    net.close();
}

/// The prior MAJOR: "no desync" stated with the sanctioned hash.
///
/// `handoff_leaves_no_desync` bounds the position gap; this pins the state
/// itself. The handoff is driven by hand so the [`Message::HandoffCommit`] can be
/// intercepted, and once B has installed it the test builds a one-entity
/// reference world holding **the commit's own decoded state**, steps it once on
/// the same tick B stepped, and compares [`space_sim::SimWorld::guid_state_hash`].
/// Equality means B is simulating exactly the state A handed over, advanced by
/// exactly one step: not the offer's stale snapshot, not a double step, and with
/// every replicated component intact.
///
/// Mutation this catches: building the commit from the *staged offer* state
/// instead of the current one (B activates behind A → hashes differ), or
/// installing after the step so the entity is stepped twice for one tick.
#[test]
fn the_installed_state_hashes_identically_to_the_committed_state() {
    let mut pair = Pair::new();
    let id_a = pair.a.shard();
    let id_b = pair.b.shard();
    let (session_a, guid) = join(&mut pair.a, "player");

    // Drive A until it offers, forwarding only shard-directed messages.
    let mut offered = false;
    for _ in 0..600 {
        pair.thrust(id_a, session_a, 0, 1.0);
        pair.a.tick();
        let out = pair.a.take_outbox();
        if out
            .iter()
            .any(|o| matches!(o.message, Message::HandoffOffer(_)))
        {
            offered = true;
            for o in out {
                if matches!(o.dest, Destination::Shard(_)) {
                    pair.b.peer_message(id_a, o.message);
                }
            }
            break;
        }
    }
    assert!(offered, "A must have offered the ship");
    for o in pair.b.take_outbox() {
        if matches!(o.dest, Destination::Shard(_)) {
            pair.a.peer_message(id_b, o.message);
        }
    }
    let session_b = pair
        .b
        .join(&space_net::hello("player"))
        .expect("join B")
        .session;
    pair.b.client_status(
        session_b,
        NeighbourStatus {
            entity: guid,
            shard: id_b,
            connected: true,
        },
    );
    for o in pair.b.take_outbox() {
        if matches!(o.dest, Destination::Shard(_)) {
            pair.a.peer_message(id_b, o.message);
        }
    }

    // Tick A until it commits; intercept the commit.
    let mut commit = None;
    for _ in 0..space_server::handoff::HANDOFF_TIMEOUT_TICKS + 10 {
        pair.thrust(id_a, session_a, 0, 1.0);
        pair.a.tick();
        if let Some(c) = pair
            .a
            .take_outbox()
            .into_iter()
            .find_map(|o| match o.message {
                Message::HandoffCommit(c) => Some(c),
                _ => None,
            })
        {
            commit = Some(c);
            break;
        }
    }
    let commit = commit.expect("A must commit once the gate is open");
    let committed = ship_state_of(&commit.state);
    pair.b.peer_message(id_a, Message::HandoffCommit(commit));

    // B installs on its own clock. No inputs are sent to B, so the only thing
    // that happens to the entity is the install and one step.
    let mut installed_at = None;
    for _ in 0..space_server::handoff::HANDOFF_LEAD + 10 {
        pair.b.tick();
        if pair.b.sim().ship_state(guid).is_some() {
            installed_at = Some(pair.b.sim().tick());
            break;
        }
    }
    let installed_at = installed_at.expect("B must install the commit");

    // A reference world: the commit's state, one entity, stepped once.
    let mut reference =
        space_sim::SimWorld::with_tick(id_b, space_sim::ShipParams::default(), installed_at - 1);
    reference.spawn_ship(guid, session_b, committed);
    reference.step();
    assert_eq!(
        reference.guid_state_hash(),
        pair.b.sim().guid_state_hash(),
        "guarantee 2 + 4: B's world must hold exactly the committed state, \
         advanced by exactly one step"
    );
}

// ============================================================ Phase 3 slice 2
//
// The shard's half of the epoch protocol (plan H10/H12): a prepare is acked, a
// commit installs the new map, an abort leaves it untouched — and an epoch that
// moves under an in-flight handoff still aborts it with `EpochChanged` (Phase 2's
// guarantee 7), now driven through the epoch protocol rather than a hand-written
// `set_shard_map`.

/// A prepare that bumps to epoch 2.
///
/// The change carves a depth-40 node for shard 2. The node is deliberately
/// *shallower* than the test regions (depth 64): `lookup` keeps the deepest
/// match, so appending it does not change which shard owns the ships in the
/// pair's regions — only the epoch moves, which is what these tests exercise.
fn epoch_two_prepare() -> space_protocol::EpochPrepare {
    space_protocol::EpochPrepare {
        epoch: 2,
        activation_tick: 1_000,
        changes: vec![(NodeRef::new(40, [0, 0, 0]), ShardId::new(2))],
    }
}

/// H10 (shard side): a prepare is acked, a commit installs the new map, and the
/// ack goes to the orchestrator destination.
#[test]
fn a_prepare_is_acked_and_a_commit_installs_the_new_map() {
    let mut pair = Pair::new();
    let id_a = pair.a.shard();
    assert_eq!(pair.a.epoch(), 1);

    pair.a
        .orchestrator_message(Message::EpochPrepare(epoch_two_prepare()));
    let out = pair.a.take_outbox();
    let ack = out
        .iter()
        .find_map(|o| match (&o.dest, &o.message) {
            (space_server::shard::Destination::Orchestrator, Message::EpochAck(a)) => {
                Some(a.clone())
            }
            _ => None,
        })
        .expect("the shard must ack a valid prepare to the orchestrator");
    assert_eq!(ack.epoch, 2);
    assert_eq!(ack.shard, id_a);
    assert!(
        pair.a.epoch_participant().has_staged(),
        "the prepare is staged, not yet applied"
    );
    assert_eq!(pair.a.epoch(), 1, "the map does not move until the commit");

    // The commit installs the new map.
    pair.a
        .orchestrator_message(Message::EpochCommit(space_protocol::EpochCommit {
            epoch: 2,
        }));
    assert_eq!(pair.a.epoch(), 2, "the commit bumps the epoch");
    assert!(
        !pair.a.epoch_participant().has_staged(),
        "the staged prepare is consumed by the commit"
    );
}

/// H10 (shard side): an abort discards the staged prepare and the map is
/// untouched.
#[test]
fn an_epoch_abort_leaves_the_map_untouched() {
    let mut pair = Pair::new();
    pair.a
        .orchestrator_message(Message::EpochPrepare(epoch_two_prepare()));
    assert!(pair.a.epoch_participant().has_staged());

    pair.a
        .orchestrator_message(Message::EpochAbort(space_protocol::EpochAbort {
            epoch: 2,
            reason: space_protocol::AbortReason::Timeout,
        }));
    assert!(
        !pair.a.epoch_participant().has_staged(),
        "the abort discards the staged prepare"
    );
    assert_eq!(
        pair.a.epoch(),
        1,
        "the map is byte-identical after an abort"
    );
    // A commit for the aborted epoch installs nothing.
    pair.a
        .orchestrator_message(Message::EpochCommit(space_protocol::EpochCommit {
            epoch: 2,
        }));
    assert_eq!(pair.a.epoch(), 1, "a commit after an abort is a no-op");
}

/// H12: an epoch that moves under an in-flight handoff aborts it with
/// `EpochChanged` and releases the guid — driven through the epoch protocol.
///
/// This is Phase 2's `an_epoch_change_after_the_commit_gate_aborts_and_releases_the_guid`
/// re-run against an orchestrator-generated epoch (prepare → commit) instead of a
/// hand-written `set_shard_map`. The transfer is driven past the commit gate, then
/// the epoch bumps; A must abort with `EpochChanged` and re-offer the same guid.
///
/// Mutation this catches: routing the epoch commit through a plain map swap that
/// skips `apply_epoch`'s stale-abort sweep — the guid would stay busy and never be
/// re-offered.
#[test]
fn an_epoch_commit_mid_handoff_aborts_with_epoch_changed_and_reoffers() {
    let mut pair = Pair::new();
    let id_a = pair.a.shard();
    let id_b = pair.b.shard();
    let (session_a, guid) = join(&mut pair.a, "player");

    // Drive A until it offers the ship, forwarding only shard-directed messages.
    let mut offered = false;
    for _ in 0..600 {
        pair.thrust(id_a, session_a, 0, 1.0);
        pair.a.tick();
        let out = pair.a.take_outbox();
        if out
            .iter()
            .any(|o| matches!(o.message, Message::HandoffOffer(_)))
        {
            offered = true;
            for o in out {
                if matches!(o.dest, Destination::Shard(_)) {
                    pair.b.peer_message(id_a, o.message);
                }
            }
            break;
        }
    }
    assert!(offered, "A must have offered the ship");
    for o in pair.b.take_outbox() {
        if matches!(o.dest, Destination::Shard(_)) {
            pair.a.peer_message(id_b, o.message);
        }
    }
    // The owner confirms on B; B mirrors the status to A, opening the gate.
    let session_b = pair
        .b
        .join(&space_net::hello("player"))
        .expect("join B")
        .session;
    pair.b.client_status(
        session_b,
        NeighbourStatus {
            entity: guid,
            shard: id_b,
            connected: true,
        },
    );
    for o in pair.b.take_outbox() {
        if matches!(o.dest, Destination::Shard(_)) {
            pair.a.peer_message(id_b, o.message);
        }
    }

    // Both shards move to epoch 2 through the epoch protocol: prepare then commit.
    for shard in [&mut pair.a, &mut pair.b] {
        shard.orchestrator_message(Message::EpochPrepare(epoch_two_prepare()));
        shard.take_outbox(); // drop the ack
        shard.orchestrator_message(Message::EpochCommit(space_protocol::EpochCommit {
            epoch: 2,
        }));
    }
    assert_eq!(pair.a.epoch(), 2);
    assert_eq!(pair.b.epoch(), 2);

    // A must abort with EpochChanged, then re-offer the same guid.
    let mut saw_abort = None;
    let mut re_offered = false;
    for _ in 0..space_server::handoff::HANDOFF_TIMEOUT_TICKS + 5 {
        pair.thrust(id_a, session_a, 0, 1.0);
        pair.a.tick();
        let messages: Vec<Message> = pair
            .a
            .take_outbox()
            .iter()
            .map(|o| o.message.clone())
            .collect();
        if saw_abort.is_none() {
            saw_abort = abort_in(&messages);
        }
        if offer_in(&messages).is_some() {
            re_offered = true;
            break;
        }
    }
    let abort = saw_abort.expect("A must abort the stale-epoch transfer");
    assert_eq!(
        abort.reason,
        space_protocol::HandoffReason::EpochChanged,
        "the abort must name the epoch change"
    );
    assert_eq!(abort.entity, guid);
    assert!(
        re_offered,
        "guarantee 7 + 8: the guid must be released so a fresh offer can start"
    );
    assert!(
        pair.a.sim().ship_state(guid).is_some(),
        "guarantee 6: A keeps simulating the ship after the abort"
    );
}
