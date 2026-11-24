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
//! | [`a_fresh_join_is_taken_only_by_the_shard_that_owns_the_spawn_point`], [`a_player_being_handed_over_is_welcomed_by_a_shard_that_does_not_own_the_spawn_point`], [`an_adoption_that_never_comes_ends_like_a_fresh_join`] | one address: who takes a fresh join, and who is sent on |
//! | [`a_fresh_join_over_the_wire_is_sent_to_the_spawn_points_owner`], [`a_client_let_go_of_is_sent_the_redirect_and_its_connection_is_closed`] | the same over real QUIC |

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use space_protocol::{
    EntityGuid, Hello, InputTick, Message, NeighbourStatus, PlayerInput, Redirect, RejectReason,
    Scene, SceneState, ShardId, ShardMap, ShardRegion, Welcome, PROTOCOL_VERSION,
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
    /// How many handoff offers each shard has sent the other.
    offers_sent: Vec<ShardId>,
}

impl Pair {
    /// Build the pair. Both shards publish the same epoch-1 map.
    fn new() -> Self {
        let id_a = ShardId::new(1);
        let id_b = ShardId::new(2);
        // (Addresses a shard could be dialed at — nothing listens there, the
        // pair is wired in memory. A map naming port 0 for a shard says "not
        // known yet", and nobody is sent to it.)
        let listening = |n: u16| SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 7_100 + n);
        let map = ShardMap::new(
            1,
            vec![
                ShardRegion::new(region(0), id_a, listening(1)),
                ShardRegion::new(region(1), id_b, listening(2)),
            ],
        );
        let mut a = Shard::new(ShardConfig::new(id_a, listening(1), region(0)));
        let mut b = Shard::new(ShardConfig::new(id_b, listening(2), region(1)));
        a.set_shard_map(map.clone());
        b.set_shard_map(map);
        Self {
            a,
            b,
            client_frames: Vec::new(),
            offers_sent: Vec::new(),
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
            if matches!(message, Message::HandoffOffer(_)) {
                self.offers_sent.push(from);
            }
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
                fast: false,
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

/// Put `tokens` on the pair's shard B as players who are **already there**,
/// and return their sessions.
///
/// Under the pair's map a new player starts on shard A — it owns the scene's
/// spawn point — and B answers a fresh join with a `JoinRedirect`
/// ([`a_fresh_join_is_taken_only_by_the_shard_that_owns_the_spawn_point`]). In
/// a running cluster a player gets onto such a shard by flying there. The
/// tests that need bystanders on B — a full shard, a client that does not own
/// the ship being handed over — do not fly a hundred ships across first: the
/// bystanders are let in while B runs the map of its own region alone, in
/// which the spawn point is B's own (the centre of its cell), and B is given
/// the pair's map back. To be called before any ship is on its way to B.
fn residents_of_b(pair: &mut Pair, tokens: &[String]) -> Vec<u64> {
    let pair_map = pair.b.shard_map().clone();
    let own: Vec<ShardRegion> = pair_map
        .regions
        .iter()
        .filter(|r| r.shard == pair.b.shard())
        .cloned()
        .collect();
    pair.b.set_shard_map(ShardMap::new(pair_map.epoch, own));
    let sessions = tokens
        .iter()
        .map(|token| {
            pair.b
                .join(&space_net::hello(token.as_str()))
                .expect("a resident of B joins")
                .session
        })
        .collect();
    pair.b.set_shard_map(pair_map);
    sessions
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
    /// Bind on loopback and start serving + ticking: shard 1, alone in the
    /// world on the region at cell 0.
    async fn start() -> Self {
        Self::start_as(ShardId::new(1), region(0)).await
    }

    /// [`Self::start`] as shard `id` on `region`.
    async fn start_as(id: ShardId, region: NodeRef) -> Self {
        let net = std::sync::Arc::new(space_server::net::ShardNet::bind(addr()).expect("bind"));
        let shard = std::sync::Arc::new(tokio::sync::Mutex::new(Shard::new(ShardConfig::new(
            id,
            net.local_addr().expect("bound addr"),
            region,
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

/// The second connection a client opens during a handoff — to the accepting
/// shard, while the offering shard still simulates its ship — is told the scene
/// like any other join. (The offering shard goes on saying the same scene at
/// its re-announce interval, which changes nothing for the client.)
///
/// Mutation this catches: announcing the scene only to a client that is given
/// a ship at join. A redirected player is *awaiting adopt* (no ship yet), so it
/// would arrive on the new shard without ever being told what to draw there.
#[test]
fn a_redirected_clients_second_connection_is_told_the_scene() {
    let mut pair = Pair::new();
    let (id_a, id_b) = (pair.a.shard(), pair.b.shard());
    let scene = SceneState::new(Scene::Galaxy.id(), 3);
    assert!(pair.a.set_scene(scene));
    assert!(pair.b.set_scene(scene));
    let (session_a, _guid) = join(&mut pair.a, "player");

    // Fly out of A until it offers the ship to B and redirects the client.
    let mut redirected = false;
    let mut told_by_a = Vec::new();
    for _ in 0..600 {
        pair.thrust(id_a, session_a, 0, 1.0);
        pair.tick();
        for message in pair.frames_for(id_a, session_a) {
            match message {
                Message::Redirect(_) => redirected = true,
                Message::SceneUpdate(state) => told_by_a.push(state),
                _ => {}
            }
        }
        if redirected {
            break;
        }
    }
    assert!(redirected, "A never redirected the client");
    // On join, and again at each re-announce interval the flight lasted: the
    // same scene every time, never anything else.
    assert!(!told_by_a.is_empty(), "A announced the scene on join");
    assert!(told_by_a.iter().all(|told| *told == scene), "{told_by_a:?}");

    // The client follows the redirect. Its hello matches the transfer B has
    // staged, so it gets no ship of its own — and still the scene.
    let session_b = pair
        .b
        .join(&space_net::hello("player"))
        .expect("the accepting shard admits the redirected player")
        .session;
    assert_eq!(
        pair.b.ship_of(session_b),
        None,
        "precondition: this is the handoff connection, awaiting adopt"
    );
    pair.deliver();
    let told_by_b: Vec<SceneState> = pair
        .frames_for(id_b, session_b)
        .into_iter()
        .filter_map(|message| match message {
            Message::SceneUpdate(state) => Some(state),
            _ => None,
        })
        .collect();
    assert_eq!(told_by_b, vec![scene]);
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
/// being simulated by A and starts being simulated by B. Since physics crossed
/// shards (protocol 5) A keeps showing it — as a ghost of B's ship, updated by
/// B's ghost frames — so the observer is never told it left, the ship never
/// jumps, and no tick goes without a frame.
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
    // What the observer was told about the ship, by A's tick.
    let mut observed: Vec<(u64, WorldPos)> = Vec::new();

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
                    if let Some(e) = d.entities.iter().find(|e| e.guid == guid) {
                        observed.push((d.tick, ship_state_of(e).pos));
                    }
                }
                Message::Snapshot(s) => {
                    observer_ticks.push(s.tick);
                    if let Some(e) = s.entities.iter().find(|e| e.guid == guid) {
                        observed.push((s.tick, ship_state_of(e).pos));
                    }
                }
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
            if saw_reappear && r.activation_tick + 30 < pair.b.sim().tick() {
                break;
            }
        }
    }

    assert!(
        !saw_removal,
        "the ship is still in sight: A shows it as a ghost of B's, it is never removed"
    );
    assert!(saw_reappear, "the mover must see its ship arrive in B");
    assert!(
        pair.a.ghost_owner(guid) == Some(id_b),
        "A holds a ghost of the ship B simulates now"
    );
    let activation = redirect.as_ref().expect("redirected").activation_tick;
    assert!(
        observed.iter().any(|(tick, _)| *tick > activation + 10),
        "the observer keeps being told where the ship is after the handoff"
    );
    // No jump: every step of what the observer saw is a few ticks of travel.
    for pair_of in observed.windows(2) {
        let ((t0, p0), (t1, p1)) = (pair_of[0], pair_of[1]);
        let step = WorldPos::distance(p0, p1) / (t1 - t0).max(1) as f64;
        assert!(
            step < 8.0,
            "the observed ship jumped {step:.2} m per tick between ticks {t0} and {t1}"
        );
    }

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
                    fast: false,
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
        // A redirect waits for an accept that never comes here, so the
        // offer itself is what shows A tried.
        saw_offer = saw_offer || pair.offers_sent.contains(&id_a);
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
                fast: false,
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

// ================================================================ one address
//
// A new player starts on the scene's spawn point, so a fresh join is taken by
// the shard that owns it and by no other (`docs/plan-sharding.md` §4).

/// The join rule, on the state machine: the owner of the spawn point welcomes
/// a newcomer and puts its ship there; a shard that does not own it answers
/// with a `JoinRedirect` naming the owner as its own map has it — and spends
/// nothing on the hello: no session, no client, no ship. While its map does
/// not say where players reach the owner, the answer is "starting". A shard
/// alone in the world owns the spawn point and behaves as it always did.
///
/// Mutation this catches: the old fallback (a ship on the centre of the
/// region of whatever shard was dialed), or a redirect that allocates a
/// session per hello (anybody could grow a shard's counters).
#[test]
fn a_fresh_join_is_taken_only_by_the_shard_that_owns_the_spawn_point() {
    use space_server::shard::JoinRefusal;
    let mut pair = Pair::new();
    let (id_a, id_b) = (pair.a.shard(), pair.b.shard());
    let map = pair.a.shard_map().clone();
    let spawn = space_scene::scene_anchor(&map).expect("an anchor");
    assert_eq!(map.lookup(&spawn), Some(id_a), "A owns the spawn point");
    assert_eq!(
        (pair.a.spawn_owner(), pair.b.spawn_owner()),
        (Some(id_a), Some(id_a))
    );

    // The owner: a welcome, and a ship on the spawn point.
    let (_session, ship) = join(&mut pair.a, "newcomer");
    assert_eq!(pair.a.sim().ship_state(ship).expect("a ship").pos, spawn);

    // Everybody else: a redirect to the owner, and nothing spent.
    let sessions_before = pair.b.next_session();
    let refused = pair
        .b
        .join(&space_net::hello("newcomer-2"))
        .expect_err("B does not own the spawn point");
    let JoinRefusal::Elsewhere(redirect) = refused else {
        panic!("expected a redirect, got {refused:?}");
    };
    assert_eq!(redirect.shard, id_a);
    assert_eq!(redirect.shard_map, map, "with B's own map");
    assert_eq!(pair.b.next_session(), sessions_before, "no session spent");
    assert_eq!(
        (pair.b.stats().clients, pair.b.stats().ships),
        (0, 0),
        "nobody was let in and nothing was spawned"
    );
    assert!(pair.b.take_outbox().is_empty(), "and nothing was queued");

    // The owner's address is not known yet (its region was carved or taken
    // over a moment ago and the map push has not arrived): try again shortly.
    let mut unresolved = map.clone();
    for region in &mut unresolved.regions {
        if region.shard == id_a {
            region.endpoint = ShardRegion::UNRESOLVED_ENDPOINT;
            region.client_endpoint = ShardRegion::UNRESOLVED_ENDPOINT;
        }
    }
    pair.b.set_shard_map(unresolved);
    assert_eq!(
        pair.b.join(&space_net::hello("newcomer-3")),
        Err(JoinRefusal::Rejected(RejectReason::Starting))
    );

    // A shard alone in the world owns the spawn point: nothing changed there.
    let mut alone = lone_shard();
    let (_session, ship) = join(&mut alone, "alone");
    assert_eq!(
        alone.sim().ship_state(ship).expect("a ship").pos,
        region_center(0)
    );
    let _ = id_b;
}

/// A player whose ship is being handed over is **not** a fresh join: the
/// shard taking the ship welcomes it wherever the spawn point is, and it
/// adopts the staged ship as before.
///
/// Mutation this catches: applying the spawn-point rule to every hello — no
/// handoff into a shard that does not own the spawn point could ever commit
/// (the client's second connection would be sent away).
#[test]
fn a_player_being_handed_over_is_welcomed_by_a_shard_that_does_not_own_the_spawn_point() {
    let mut pair = Pair::new();
    let id_b = pair.b.shard();
    let (_session_a, guid) = offer_and_accept(&mut pair, "owner-token");
    assert_ne!(pair.b.spawn_owner(), Some(id_b), "B is not the spawn owner");

    let welcome = pair
        .b
        .join(&space_net::hello("owner-token"))
        .expect("the redirected player is let in");
    assert_eq!(welcome.shard, id_b);
    assert_eq!(
        pair.b.ship_of(welcome.session),
        None,
        "no placeholder: it is waiting to adopt its ship"
    );
    pair.b.client_status(
        welcome.session,
        NeighbourStatus {
            entity: guid,
            shard: id_b,
            connected: true,
        },
    );
    assert_eq!(pair.b.ship_of(welcome.session), Some(guid), "adopted");
    // Anybody else is still a fresh join, staged ship or not.
    assert!(matches!(
        pair.b.join(&space_net::hello("somebody-else")),
        Err(space_server::shard::JoinRefusal::Elsewhere(_))
    ));
}

/// A client that came for a staged ship and never adopted it (the handoff
/// was called off, or its token merely matched) is treated as a fresh join
/// once its adopt budget has run out: on a shard that does not own the spawn
/// point it is told where a new player starts and let go — no ship is put on
/// the centre of that shard's region — and on the shard that owns the spawn
/// point it gets a ship there, as before.
///
/// Mutation this catches: the old fallback in `client_timeout_pass` (the last
/// way a player could end up in the middle of a cell nobody chose).
#[test]
fn an_adoption_that_never_comes_ends_like_a_fresh_join() {
    use space_server::config::ADOPT_TIMEOUT_TICKS;
    // On B, which does not own the spawn point.
    let mut pair = Pair::new();
    let (id_a, id_b) = (pair.a.shard(), pair.b.shard());
    let (_session_a, _guid) = offer_and_accept(&mut pair, "owner-token");
    let session = pair
        .b
        .join(&space_net::hello("owner-token"))
        .expect("let in for the staged ship")
        .session;
    pair.b.take_outbox();
    let mut last_words = Vec::new();
    for _ in 0..ADOPT_TIMEOUT_TICKS + 2 {
        pair.b.tick();
        for out in pair.b.take_outbox() {
            if out.dest == Destination::Client(session) {
                last_words.push(out.message);
            }
        }
    }
    let redirect = last_words
        .iter()
        .find_map(|m| match m {
            Message::JoinRedirect(r) => Some(r.clone()),
            _ => None,
        })
        .expect("the client is told where a new player starts");
    assert_eq!(redirect.shard, id_a);
    assert!(
        matches!(last_words.last(), Some(Message::JoinRedirect(_))),
        "and nothing is said to it after that: {last_words:?}"
    );
    assert_eq!(pair.b.stats().clients, 0, "it is no client of B any more");
    assert_eq!(pair.b.ship_of(session), None);
    assert_eq!(pair.b.stats().ships, 0, "no ship at B's region centre");

    // On A, which owns it: a ship on the spawn point.
    let mut pair = Pair::new();
    let guid = EntityGuid::new(id_b, 77);
    pair.a.peer_message(
        id_b,
        Message::HandoffOffer(space_protocol::HandoffOffer {
            entity: guid,
            from: id_b,
            to: id_a,
            epoch: pair.a.epoch(),
            state: Box::new(offer_state(guid, "owner-token")),
            activation_tick: space_server::handoff::HANDOFF_LEAD,
        }),
    );
    let session = pair
        .a
        .join(&space_net::hello("owner-token"))
        .expect("let in for the staged ship")
        .session;
    assert_eq!(pair.a.ship_of(session), None);
    for _ in 0..ADOPT_TIMEOUT_TICKS + 2 {
        pair.a.tick();
        pair.a.take_outbox();
    }
    let ship = pair.a.ship_of(session).expect("a ship of its own");
    assert_ne!(ship, guid);
    assert_eq!(
        pair.a.sim().ship_state(ship).expect("alive").pos,
        region_center(0),
        "on the spawn point"
    );
}

/// The same rule over the wire: a client that says hello to a running shard
/// which does not own the spawn point is answered with a `JoinRedirect` — in
/// place of the welcome, with the map that names where players reach the
/// owner — and the connection is finished.
///
/// Mutation this catches: the front end rejecting instead of redirecting (a
/// client would show an error instead of ending up on the right shard), or
/// answering and keeping the connection (one held open per misdirected hello).
#[tokio::test]
async fn a_fresh_join_over_the_wire_is_sent_to_the_spawn_points_owner() {
    let (id_a, id_b) = (ShardId::new(1), ShardId::new(2));
    let harness = RunningShard::start_as(id_b, region(1)).await;
    let own = harness.net.local_addr().expect("bound addr");
    // Shard 1 owns the spawn point; players reach it on the host they dialed.
    let players_at: SocketAddr = "0.0.0.0:7871".parse().expect("addr");
    let map = ShardMap::new(
        1,
        vec![
            ShardRegion::new(region(0), id_a, "10.9.8.7:7871".parse().expect("addr"))
                .with_client_endpoint(players_at),
            ShardRegion::new(region(1), id_b, own),
        ],
    );
    harness.shard.lock().await.set_shard_map(map.clone());

    let endpoint = space_net::client_endpoint_for(own).expect("client endpoint");
    let joined = space_net::join(
        &endpoint,
        own,
        space_server::net::SERVER_NAME,
        &space_net::hello("newcomer"),
    )
    .await
    .expect("an answer");
    let space_net::Joined::Elsewhere(redirect) = joined else {
        panic!("expected a redirect, got {joined:?}");
    };
    assert_eq!(redirect.shard, id_a);
    // The map as a player gets it: the client endpoints, and nothing of
    // where the shards reach each other (10.9.8.7 never leaves the cluster).
    assert_eq!(redirect.shard_map, map.for_players());
    assert!(!format!("{:?}", redirect.shard_map).contains("10.9.8.7"));
    assert_eq!(
        redirect.shard_map.client_endpoint_of(id_a),
        Some(players_at)
    );
    harness.settle().await;
    assert_eq!(harness.clients().await, 0, "nobody was let in");
    assert_eq!(harness.shard.lock().await.stats().ships, 0);

    // `connect` — a shard's own dial — reports it as what it is.
    let err = harness.connect("not-a-player").await.expect_err("sent on");
    assert!(
        matches!(&err, space_net::NetError::Protocol(why) if why.contains("player entry")),
        "{err:?}"
    );
    harness.shutdown();
}

/// And the late form of it over the wire: a client let in for a staged ship
/// that never adopts it is sent the redirect on its open connection, which
/// the shard then closes.
#[tokio::test]
async fn a_client_let_go_of_is_sent_the_redirect_and_its_connection_is_closed() {
    let (id_a, id_b) = (ShardId::new(1), ShardId::new(2));
    let harness = RunningShard::start_as(id_b, region(1)).await;
    let own = harness.net.local_addr().expect("bound addr");
    let map = ShardMap::new(
        1,
        vec![
            ShardRegion::new(region(0), id_a, "127.0.0.1:7871".parse().expect("addr")),
            ShardRegion::new(region(1), id_b, own),
        ],
    );
    let guid = EntityGuid::new(id_a, 77);
    {
        let mut shard = harness.shard.lock().await;
        shard.set_shard_map(map);
        let epoch = shard.epoch();
        let activation_tick = shard.sim().tick() + space_server::handoff::HANDOFF_LEAD;
        shard.peer_message(
            id_a,
            Message::HandoffOffer(space_protocol::HandoffOffer {
                entity: guid,
                from: id_a,
                to: id_b,
                epoch,
                state: Box::new(offer_state(guid, "owner-token")),
                activation_tick,
            }),
        );
    }
    let (conn, welcome) = harness
        .connect("owner-token")
        .await
        .expect("let in for the staged ship");
    assert_eq!(welcome.shard, id_b);

    // It never reports in. Within the adopt budget (2 s) and a little, the
    // redirect arrives on a reliable stream...
    let redirect = tokio::time::timeout(std::time::Duration::from_secs(6), async {
        loop {
            match conn.recv_stream().await {
                Ok(Some(Message::JoinRedirect(redirect))) => break Some(redirect),
                Ok(Some(_)) => {}
                Ok(None) | Err(_) => break None,
            }
        }
    })
    .await
    .expect("the shard says something within the adopt budget")
    .expect("and it is the redirect, before the connection ends");
    assert_eq!(redirect.shard, id_a);
    // ...and the shard closes the connection (at the latest after its drain).
    tokio::time::timeout(
        std::time::Duration::from_secs(3),
        conn.connection().closed(),
    )
    .await
    .expect("the connection is closed from the shard's side");
    harness.wait_for_clients(0).await.expect("no client left");
    assert_eq!(harness.shard.lock().await.stats().ships, 0);
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
                    fast: false,
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
    // The wrong-token client: somebody who is on B already, with a ship of
    // its own. (It used to join B after the transfer was staged; a fresh join
    // on B is sent to A now, so a client of B is either a resident like this
    // one or a player whose own ship is being handed to B.)
    let thief = residents_of_b(&mut pair, &["thief-token".to_owned()])[0];
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

    // The wrong-token client claims the staged guid.
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
    // The thief keeps the ship it had and nothing else.
    assert!(
        pair.b.ship_of(thief).is_some_and(|s| s != guid),
        "the wrong-token client keeps its own ship"
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
    // Deliver B's mirror: it is what opens A's commit gate. (Only what B
    // addressed to a shard: its outbox also holds the scene announcement for
    // the client that just joined.)
    for out in b.take_outbox() {
        if matches!(out.dest, Destination::Shard(_)) {
            a.peer_message(id_b, out.message);
        }
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

    // Fill B to the cap (with players who are there already: a fresh join on
    // B is sent to A, which owns the spawn point).
    let fillers: Vec<String> = (0..space_server::shard::MAX_PLAYERS_PER_SHARD)
        .map(|i| format!("filler-{i}"))
        .collect();
    residents_of_b(&mut pair, &fillers);
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

    // A reference world: the commit's state, one entity, stepped once. A
    // player's ship carries the player model (what every client draws it
    // as), so the reference ship does too.
    let mut reference =
        space_sim::SimWorld::with_tick(id_b, space_sim::ShipParams::default(), installed_at - 1);
    reference.spawn_ship(guid, session_b, committed);
    reference.set_visual(guid, space_sim::Visual(space_scene::visual::PLAYER));
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

// ================================================================ audit fixes
//
// One regression test per fix from the shard/transport audit. As above, each
// names the mutation that makes it fail.

/// The frames one shard queued for `session`, draining its whole outbox.
///
/// The scene announcement a join queues behind the welcome is not a frame of
/// the world (it has no tick and names no entity), so it is left out here; the
/// scene tests look at it on its own.
fn frames_of(shard: &mut Shard, session: u64) -> Vec<Message> {
    shard
        .take_outbox()
        .into_iter()
        .filter(|o| o.dest == Destination::Client(session))
        .map(|o| o.message)
        .filter(|message| !matches!(message, Message::SceneUpdate(_) | Message::SceneClock(_)))
        .collect()
}

/// The tick a replication frame describes.
fn frame_tick(message: &Message) -> Option<u64> {
    match message {
        Message::Snapshot(s) => Some(s.tick),
        Message::Delta(d) => Some(d.tick),
        _ => None,
    }
}

/// The guids a replication frame carries.
fn frame_entities(message: &Message) -> Vec<EntityGuid> {
    match message {
        Message::Snapshot(s) => s.entities.iter().map(|e| e.guid).collect(),
        Message::Delta(d) => d.entities.iter().map(|e| e.guid).collect(),
        _ => Vec::new(),
    }
}

/// An input batch for `ship`, tagged `tick`, acknowledging `ack_tick`.
fn input(ship: EntityGuid, tick: u64, thrust: [f32; 3], ack_tick: u64) -> InputTick {
    InputTick {
        tick,
        seq: tick,
        inputs: vec![PlayerInput {
            entity: ship,
            thrust,
            torque: [0.0; 3],
            tick,
            fast: false,
        }],
        ack_tick,
    }
}

/// A one-shard world: a single region, nobody to hand anything to.
fn lone_shard() -> Shard {
    Shard::new(ShardConfig::new(ShardId::new(1), addr(), region(0)))
}

/// A client whose ship is at rest is still sent a frame on every tick, and the
/// frame names the tick.
///
/// Mutation this catches: returning "nothing to say" from
/// `replicate::build_frame` when no entity changed — the resting client then
/// receives nothing at all after its snapshot.
#[test]
fn a_client_at_rest_is_sent_a_frame_every_tick() {
    let mut shard = lone_shard();
    let (session, ship) = join(&mut shard, "player");

    let mut last_tick = 0;
    for step in 0..30 {
        // The client acknowledges what it applied (and sends no input: the ship
        // stays at rest and nothing lands that could stamp it).
        if last_tick > 0 {
            shard.apply_input(
                session,
                &InputTick {
                    tick: last_tick + 3,
                    seq: last_tick,
                    inputs: Vec::new(),
                    ack_tick: last_tick,
                },
            );
        }
        let tick = shard.tick();
        let frames = frames_of(&mut shard, session);
        assert_eq!(frames.len(), 1, "tick {tick}: exactly one frame per tick");
        assert_eq!(
            frame_tick(&frames[0]),
            Some(tick),
            "the frame names its tick"
        );
        match &frames[0] {
            Message::Snapshot(s) => {
                assert_eq!(step, 0, "only the first frame is a snapshot");
                assert!(s.entities.iter().any(|e| e.guid == ship));
            }
            Message::Delta(d) => {
                assert!(step > 0, "the first frame is the snapshot");
                assert_eq!(d.base_tick, tick - 1, "it follows the previous frame");
                assert!(
                    d.entities.is_empty() && d.removed.is_empty(),
                    "nothing changed, so the delta is empty: {d:?}"
                );
            }
            other => panic!("unexpected frame {other:?}"),
        }
        last_tick = tick;
    }
}

/// Until a client has acknowledged a frame, every frame names its own ship —
/// for a bounded time.
///
/// The snapshot that tells a client which ship is its own rides a reliable
/// stream; the next tick's delta rides a datagram and can arrive first, and a
/// client that applies the newer frame discards the older one. A client that
/// does not know its ship sends no input and therefore no acknowledgement, so
/// nothing would ever reveal the loss. Whichever frame arrives first has to be
/// enough.
///
/// Mutation this catches: dropping the "own ship while unacknowledged"
/// candidate from `replicate::build_frame` (the deltas before the first
/// acknowledgement are then empty), or dropping its bound (the last loop then
/// still finds the ship).
#[test]
fn until_a_client_acknowledges_every_frame_names_its_ship() {
    let mut shard = lone_shard();
    let (session, ship) = join(&mut shard, "player");

    // Before any acknowledgement: the snapshot, then deltas that repeat the ship.
    let mut last_tick = 0;
    for step in 0..5 {
        last_tick = shard.tick();
        let frames = frames_of(&mut shard, session);
        assert_eq!(frames.len(), 1);
        assert!(
            frame_entities(&frames[0]).contains(&ship),
            "frame {step} must name the client's ship: {:?}",
            frames[0]
        );
    }

    // The first acknowledgement ends it.
    shard.apply_input(
        session,
        &InputTick {
            tick: last_tick + 3,
            seq: 1,
            inputs: Vec::new(),
            ack_tick: last_tick,
        },
    );
    shard.tick();
    let frames = frames_of(&mut shard, session);
    assert!(
        frame_entities(&frames[0]).is_empty(),
        "an acknowledged client is not sent its parked ship again: {:?}",
        frames[0]
    );

    // A client that never acknowledges is not sent its parked ship forever.
    let mut shard = lone_shard();
    let (session, ship) = join(&mut shard, "silent");
    shard.tick_n(space_server::replicate::UNACKNOWLEDGED_SHIP_TICKS + 2);
    shard.take_outbox();
    for _ in 0..5 {
        shard.tick();
        let frames = frames_of(&mut shard, session);
        assert_eq!(frames.len(), 1, "it still gets its frame every tick");
        assert!(
            !frame_entities(&frames[0]).contains(&ship),
            "the repetition is bounded: {:?}",
            frames[0]
        );
    }
}

/// A client that sat still — and lost a frame while it did — can still start
/// moving.
///
/// The client here does exactly what the stock client does: it has no clock of
/// its own, so it tags every input `last frame tick + 3`. While its ship was at
/// rest the shard used to send it a frame only when one of those inputs landed,
/// which made the two a ping-pong: lose one frame and the client keeps tagging
/// the same tick, the shard moves past it, every later input is dropped as
/// stale, nothing changes, and the shard never speaks again. Three consecutive
/// ticks of frames are dropped below, then the player pushes the stick.
///
/// Mutation this catches: the same as
/// [`a_client_at_rest_is_sent_a_frame_every_tick`] — without a frame every tick
/// the clock never recovers and the ship never moves.
#[test]
fn a_client_that_sat_still_can_start_moving() {
    const INPUT_LEAD: u64 = 3;
    let mut shard = lone_shard();
    let (session, ship) = join(&mut shard, "player");
    let start = shard.sim().ship_state(ship).expect("alive").pos;

    let mut last_frame_tick = 0u64;
    for step in 0..200u64 {
        // The client only speaks once a frame has told it which ship is its own.
        if last_frame_tick > 0 {
            let thrust = if step >= 100 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0; 3]
            };
            shard.apply_input(
                session,
                &input(ship, last_frame_tick + INPUT_LEAD, thrust, last_frame_tick),
            );
        }
        shard.tick();
        let frames = frames_of(&mut shard, session);
        if (40..43).contains(&step) {
            continue; // these frames never arrive
        }
        for frame in &frames {
            if let Some(tick) = frame_tick(frame) {
                last_frame_tick = last_frame_tick.max(tick);
            }
        }
    }

    let end = shard.sim().ship_state(ship).expect("alive");
    assert!(
        end.vel[0] > 0.0 && WorldPos::distance(start, end.pos) > 1.0,
        "the ship must answer the stick after sitting still: {end:?}"
    );
}

/// A frame the client never acknowledged is sent again — an entity and a
/// removal both.
///
/// Deltas ride unreliable datagrams and the front end drops a frame when a
/// client's queue is full, so "built" is not "delivered". The client reports the
/// highest tick it applied; a frame older than that which was never itself
/// acknowledged is presumed lost.
///
/// Mutation this catches: `ClientReplica::acknowledge` settling every frame at
/// or below the acknowledged tick as delivered (or `apply_input` not calling it
/// at all) — the lost snapshot's ship and the lost removal are then never sent
/// again.
#[test]
fn a_frame_the_client_never_acknowledged_is_sent_again() {
    let mut shard = lone_shard();
    let (watcher, watcher_ship) = join(&mut shard, "watcher");
    let (leaver, leaver_ship) = join(&mut shard, "leaver");

    // The snapshot goes out and is lost; two more frames follow, and the second
    // of them arrives.
    let snapshot_tick = shard.tick();
    let lost = frames_of(&mut shard, watcher);
    assert!(matches!(lost.first(), Some(Message::Snapshot(_))));
    shard.tick();
    shard.take_outbox();
    let heard = shard.tick();
    let mut received: Vec<EntityGuid> = frames_of(&mut shard, watcher)
        .iter()
        .flat_map(frame_entities)
        .collect();
    assert!(
        !received.contains(&leaver_ship),
        "precondition: only the lost snapshot carried the other ship"
    );

    // The client applied the frame for `heard` and never the snapshot.
    assert!(heard > snapshot_tick);
    shard.apply_input(
        watcher,
        &InputTick {
            tick: heard + 3,
            seq: 1,
            inputs: Vec::new(),
            ack_tick: heard,
        },
    );
    let resend_tick = shard.tick();
    let resent = frames_of(&mut shard, watcher);
    let carried: Vec<EntityGuid> = resent.iter().flat_map(frame_entities).collect();
    assert!(
        carried.contains(&leaver_ship),
        "what only the lost snapshot carried must be sent again, got {carried:?}"
    );
    received.extend(carried);
    assert!(
        received.contains(&watcher_ship) && received.contains(&leaver_ship),
        "between them, the frames that arrived cover the lost snapshot: {received:?}"
    );

    // That frame *is* acknowledged, exactly: nothing more is owed.
    shard.apply_input(
        watcher,
        &InputTick {
            tick: resend_tick + 3,
            seq: 2,
            inputs: Vec::new(),
            ack_tick: resend_tick,
        },
    );
    shard.tick();
    let quiet = frames_of(&mut shard, watcher);
    assert!(
        quiet.iter().flat_map(frame_entities).next().is_none(),
        "an acknowledged frame is not repeated: {quiet:?}"
    );

    // The other ship leaves; the delta that says so is lost.
    shard.leave(leaver);
    let removal_tick = shard.tick();
    let removal = frames_of(&mut shard, watcher);
    assert!(
        matches!(removal.first(), Some(Message::Delta(d)) if d.removed.contains(&leaver_ship)),
        "the watcher is told the ship left: {removal:?}"
    );
    let later = shard.tick();
    shard.take_outbox();
    assert!(later > removal_tick);
    shard.apply_input(
        watcher,
        &InputTick {
            tick: later + 3,
            seq: 3,
            inputs: Vec::new(),
            ack_tick: later,
        },
    );
    shard.tick();
    let again = frames_of(&mut shard, watcher);
    assert!(
        matches!(again.first(), Some(Message::Delta(d)) if d.removed.contains(&leaver_ship)),
        "a removal that was never acknowledged must be said again: {again:?}"
    );
}

/// An entity that is at rest when it comes into a client's interest is sent.
///
/// The change scan only ever finds rows written since the client was last
/// examined, and a parked ship is never written. Coming into range has to be a
/// reason to send it in its own right.
///
/// Mutation this catches: dropping the "entered interest" candidates from
/// `replicate::build_frame` — the parked ship is then invisible to the player
/// flying up to it, however close they get.
#[test]
fn a_parked_ship_is_sent_when_a_client_flies_into_range() {
    let mut config = ShardConfig::new(ShardId::new(1), addr(), region(0));
    config.interest_radius = 150.0;
    let mut shard = Shard::new(config);

    // Parked 400 m along +x of where players spawn: well outside interest.
    let parked_at = region_center(0).offset([400.0, 0.0, 0.0]);
    let parked = shard.spawn_ship_for(0, parked_at);
    shard.tick_n(5);
    shard.take_outbox();

    let (session, ship) = join(&mut shard, "player");
    let mut saw_parked = false;
    let mut first_frame = true;
    for _ in 0..900 {
        let tick = shard.sim().tick() + 1;
        shard.apply_input(session, &input(ship, tick, [1.0, 0.0, 0.0], 0));
        shard.tick();
        for frame in frames_of(&mut shard, session) {
            let carried = frame_entities(&frame);
            if first_frame {
                assert!(
                    !carried.contains(&parked),
                    "precondition: the parked ship starts outside interest"
                );
                first_frame = false;
            }
            saw_parked = saw_parked || carried.contains(&parked);
        }
        if saw_parked {
            break;
        }
    }
    let gap = WorldPos::distance(shard.sim().ship_state(ship).expect("alive").pos, parked_at);
    assert!(
        saw_parked,
        "the parked ship never reached the client, which is now {gap:.0} m from it"
    );
    assert!(
        gap > 100.0,
        "it must be sent on entering the 150 m sphere, not on arrival ({gap:.0} m)"
    );
    assert_eq!(
        shard.sim().ship_state(parked).expect("alive").pos,
        parked_at,
        "the parked ship itself never moved"
    );
}

/// Drive A's player ship out of A's region until A offers it, hand the offer to
/// B and B's accept back to A. Returns A's session and the ship.
fn offer_and_accept(pair: &mut Pair, token: &str) -> (u64, EntityGuid) {
    let id_a = pair.a.shard();
    let id_b = pair.b.shard();
    let (session_a, guid) = join(&mut pair.a, token);
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
    let mut accepted = false;
    for o in pair.b.take_outbox() {
        if matches!(o.dest, Destination::Shard(_)) {
            accepted = accepted || matches!(o.message, Message::HandoffAccept(_));
            pair.a.peer_message(id_b, o.message);
        }
    }
    assert!(accepted, "B must have accepted and staged the ship");
    (session_a, guid)
}

/// A client is redirected only once the accepting shard has staged the ship.
///
/// The redirect used to go out with the offer. Over real connections the
/// client's hello could then reach B before the offer did: B, holding no
/// staged transfer for the client's token, spawned it a placeholder ship at
/// its spawn point — on top of the ship on its way in, the two shoving each
/// other apart through their ghosts — and dropped its `NeighbourStatus` as a
/// claim for nothing. Waiting for the accept makes that order impossible.
///
/// Mutation this catches: sending the `Redirect` from the offer instead of
/// from the accept (the outbox of the offer tick then holds one).
#[test]
fn a_client_is_redirected_only_after_the_peer_accepted() {
    let mut pair = Pair::new();
    let id_a = pair.a.shard();
    let id_b = pair.b.shard();
    let (session_a, guid) = join(&mut pair.a, "player");

    // Fly out of A, holding everything A sends until it offers the ship.
    let mut offer = None;
    for _ in 0..600 {
        pair.thrust(id_a, session_a, 0, 1.0);
        pair.a.tick();
        let mut redirected = false;
        for out in pair.a.take_outbox() {
            match out.message {
                Message::HandoffOffer(o) => offer = Some(o),
                Message::Redirect(_) => redirected = true,
                _ => {}
            }
        }
        assert!(!redirected, "a redirect went out before B accepted");
        if offer.is_some() {
            break;
        }
    }
    let offer = offer.expect("A offered the ship");

    // B stages it and accepts; only the accept makes A redirect the client.
    pair.b.peer_message(id_a, Message::HandoffOffer(offer));
    let accept = pair
        .b
        .take_outbox()
        .into_iter()
        .find_map(|out| match out.message {
            Message::HandoffAccept(a) => Some(a),
            _ => None,
        })
        .expect("B accepted");
    pair.a
        .peer_message(id_b, Message::HandoffAccept(accept.clone()));
    let redirects: Vec<(Destination, Redirect)> = pair
        .a
        .take_outbox()
        .into_iter()
        .filter_map(|out| match out.message {
            Message::Redirect(r) => Some((out.dest, r)),
            _ => None,
        })
        .collect();
    assert_eq!(redirects.len(), 1, "one redirect, to the one client");
    let (dest, redirect) = &redirects[0];
    assert!(matches!(dest, Destination::Client(s) if *s == session_a));
    assert_eq!(redirect.entity, guid);
    assert_eq!(redirect.new_shard, id_b);
    assert_eq!(redirect.activation_tick, accept.activation_tick);

    // A repeated accept redirects nobody again.
    pair.a.peer_message(id_b, Message::HandoffAccept(accept));
    assert!(!pair
        .a
        .take_outbox()
        .iter()
        .any(|out| matches!(out.message, Message::Redirect(_))));

    // So the client arriving at B is the staged ship's owner: no placeholder.
    let welcome = pair.b.join(&space_net::hello("player")).expect("join B");
    assert_eq!(pair.b.ship_of(welcome.session), None, "no placeholder ship");
    assert_eq!(pair.b.stats().ships, 0);
}

/// A peer that refuses an offer is not asked again on the very next tick.
///
/// B is full, so every offer is answered with `TargetFull`. The ship is still
/// across the boundary a tick later, and A used to offer it again at once: an
/// offer, a refusal and a `Redirect` to every watcher, at tick rate, for as long
/// as B stayed full. A new shard map ends the pause, because the refusal was
/// about the old one.
///
/// Mutation this catches: `Shard::on_abort` not recording the refusal (or
/// `handoff_pass` not consulting it) — A then offers on every tick.
#[test]
fn a_refused_offer_is_not_repeated_every_tick() {
    use space_server::config::REOFFER_BACKOFF_TICKS;

    let mut pair = Pair::new();
    let id_a = pair.a.shard();
    let id_b = pair.b.shard();
    let (session_a, guid) = join(&mut pair.a, "player");
    // (Players who are on B already: a fresh join on B is sent to A.)
    let fillers: Vec<String> = (0..space_server::shard::MAX_PLAYERS_PER_SHARD)
        .map(|i| format!("filler-{i}"))
        .collect();
    residents_of_b(&mut pair, &fillers);
    pair.b.take_outbox();

    let mut offers = 0u64;
    let mut redirects = 0u64;
    let mut first_offer_at = None;
    for _ in 0..1_200 {
        pair.thrust(id_a, session_a, 0, 1.0);
        let tick = pair.a.tick();
        for o in pair.a.take_outbox() {
            match o.dest {
                Destination::Shard(_) => {
                    if matches!(o.message, Message::HandoffOffer(_)) {
                        offers += 1;
                        first_offer_at.get_or_insert(tick);
                    }
                    pair.b.peer_message(id_a, o.message);
                }
                Destination::Client(_) => {
                    if matches!(o.message, Message::Redirect(_)) {
                        redirects += 1;
                    }
                }
                Destination::Orchestrator => {}
            }
        }
        // B answers on arrival; it does not need to tick to refuse.
        for o in pair.b.take_outbox() {
            if matches!(o.dest, Destination::Shard(_)) {
                pair.a.peer_message(id_b, o.message);
            }
        }
        if first_offer_at.is_some_and(|first| tick >= first + 4 * REOFFER_BACKOFF_TICKS) {
            break;
        }
    }
    assert!(first_offer_at.is_some(), "A must have offered the ship");
    assert!(
        (2..=6).contains(&offers),
        "over four back-off periods a refused ship is offered about once per period, \
         not once per tick: {offers} offers"
    );
    assert_eq!(
        redirects, 0,
        "a client is redirected only once the peer has accepted: a refused offer          sends it nowhere"
    );
    assert!(pair.a.sim().ship_state(guid).is_some(), "A keeps the ship");

    // A new map is a new question: the pause ends with it.
    let newer = ShardMap::new(
        2,
        vec![
            ShardRegion::new(region(0), id_a, addr()),
            ShardRegion::new(region(1), id_b, addr()),
        ],
    );
    pair.a.set_shard_map(newer);
    pair.a.take_outbox();
    pair.thrust(id_a, session_a, 0, 1.0);
    pair.a.tick();
    let messages: Vec<Message> = pair
        .a
        .take_outbox()
        .into_iter()
        .map(|o| o.message)
        .collect();
    assert!(
        offer_in(&messages).is_some(),
        "after a map change the ship is offered again without waiting"
    );
}

/// An entity whose commit has arrived and is waiting for its install tick
/// cannot be offered (and so committed) a second time.
///
/// Between the commit and the install the entity is in neither the ledger nor
/// the world, so the old busy check let a second offer through. Two installs of
/// one guid is a panic inside `SimWorld::spawn_ship`, inside `Shard::tick` —
/// which kills the tick loop while the shard goes on accepting players.
///
/// Mutation this catches: dropping the `pending_commits` clause from
/// `Shard::on_offer`'s busy check *and* the "already alive" guard in
/// `install_pending_commits` — the loop below then panics.
#[test]
fn an_entity_awaiting_install_cannot_be_offered_again() {
    let mut pair = Pair::new();
    let id_a = pair.a.shard();
    let id_b = pair.b.shard();
    let guid = EntityGuid::new(id_a, 77);
    let offer = space_protocol::HandoffOffer {
        entity: guid,
        from: id_a,
        to: id_b,
        epoch: pair.b.epoch(),
        state: Box::new(offer_state(guid, "player")),
        activation_tick: space_server::handoff::HANDOFF_LEAD,
    };
    let commit = space_protocol::HandoffCommit {
        entity: guid,
        from: id_a,
        to: id_b,
        epoch: pair.b.epoch(),
        activation_tick: space_server::handoff::HANDOFF_LEAD,
        state: Box::new(offer_state(guid, "player")),
    };

    pair.b
        .peer_message(id_a, Message::HandoffOffer(offer.clone()));
    assert!(pair
        .b
        .take_outbox()
        .iter()
        .any(|o| matches!(o.message, Message::HandoffAccept(_))));
    pair.b
        .peer_message(id_a, Message::HandoffCommit(commit.clone()));
    assert!(
        pair.b.sim().ship_state(guid).is_none(),
        "precondition: the commit waits for B's own install tick"
    );

    // The same exchange again, while the first commit is still waiting.
    pair.b.peer_message(id_a, Message::HandoffOffer(offer));
    let messages: Vec<Message> = pair
        .b
        .take_outbox()
        .into_iter()
        .map(|o| o.message)
        .collect();
    assert!(
        !messages
            .iter()
            .any(|m| matches!(m, Message::HandoffAccept(_))),
        "an entity awaiting install must not be staged a second time"
    );
    assert!(
        abort_in(&messages).is_some_and(|a| a.entity == guid),
        "the second offer is refused"
    );
    pair.b.peer_message(id_a, Message::HandoffCommit(commit));

    for _ in 0..space_server::handoff::HANDOFF_LEAD + 5 {
        pair.b.tick();
    }
    assert!(
        pair.b.sim().ship_state(guid).is_some(),
        "the first commit installs"
    );
    assert_eq!(pair.b.stats().received, 1, "exactly once");
    assert_eq!(pair.b.stats().ships, 1);
}

/// Only the player a staged entity belongs to can call its handoff off.
///
/// `NeighbourStatus { connected: true }` was already gated on the hello token
/// the offer carried; `connected: false` was not, so any client could cancel any
/// other player's inbound handoff by naming its (guessable) guid — and the
/// abort it produced carried a shard id straight out of the client's message.
///
/// Mutation this catches: aborting the staged transfer in
/// `Shard::client_status` without the `claim_is_authorized` test.
#[test]
fn only_the_owner_can_call_off_a_staged_handoff() {
    let mut pair = Pair::new();
    let id_a = pair.a.shard();
    let id_b = pair.b.shard();
    // Somebody who is on B already, with a ship of its own. (A fresh join on
    // B is sent to A, which owns the spawn point.)
    let stranger = residents_of_b(&mut pair, &["someone-else".to_owned()])[0];
    let (_session_a, guid) = offer_and_accept(&mut pair, "owner-token");
    pair.b.take_outbox();
    pair.b.client_status(
        stranger,
        NeighbourStatus {
            entity: guid,
            shard: id_b,
            connected: false,
        },
    );
    let messages: Vec<Message> = pair
        .b
        .take_outbox()
        .into_iter()
        .map(|o| o.message)
        .collect();
    assert!(
        abort_in(&messages).is_none(),
        "a client that does not own the ship must not be able to cancel its handoff"
    );
    assert_eq!(pair.b.stats().aborted, 0);

    // The owner really cannot reach B: the transfer is called off, and the
    // abort names B — not the shard id the client wrote into its status.
    let owner = pair
        .b
        .join(&space_net::hello("owner-token"))
        .expect("join B")
        .session;
    pair.b.take_outbox();
    pair.b.client_status(
        owner,
        NeighbourStatus {
            entity: guid,
            shard: ShardId::new(999),
            connected: false,
        },
    );
    let messages: Vec<Message> = pair
        .b
        .take_outbox()
        .into_iter()
        .map(|o| o.message)
        .collect();
    let abort = abort_in(&messages).expect("the owner can call the handoff off");
    assert_eq!(abort.entity, guid);
    assert_eq!(abort.from, id_a);
    assert_eq!(abort.to, id_b, "the abort names the shard that staged it");
    assert_eq!(
        abort.reason,
        space_protocol::HandoffReason::ClientNotConnected
    );
    assert_eq!(pair.b.stats().aborted, 1);
}

/// A player who adopts a staged ship and then leaves does not leave the ship
/// behind — neither while it is still staged nor once its commit is waiting to
/// be installed.
///
/// In both states the ship is not in B's world yet, so the despawn `leave` does
/// found nothing, the transfer stayed where it was, and the commit later
/// installed a ship owned by a session that no longer existed. Nothing ever
/// removed it.
///
/// Mutation this catches: `Shard::leave` not aborting the staged transfer bound
/// to the session (first half) and not dropping the session's pending commit
/// (second half).
#[test]
fn a_player_who_leaves_before_the_install_leaves_no_ship_behind() {
    // --- left while the ship is still staged
    let mut pair = Pair::new();
    let id_a = pair.a.shard();
    let id_b = pair.b.shard();
    let (_session_a, guid) = offer_and_accept(&mut pair, "player");
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
    assert_eq!(pair.b.ship_of(session_b), Some(guid), "adopted");
    pair.b.take_outbox();

    pair.b.leave(session_b);
    let messages: Vec<Message> = pair
        .b
        .take_outbox()
        .into_iter()
        .map(|o| o.message)
        .collect();
    let abort = abort_in(&messages).expect("B hands the staged ship back to A");
    assert_eq!(abort.entity, guid);
    assert_eq!(abort.from, id_a);
    assert_eq!(
        abort.reason,
        space_protocol::HandoffReason::ClientNotConnected
    );
    // A commit that crossed the abort finds nothing to install.
    let commit = space_protocol::HandoffCommit {
        entity: guid,
        from: id_a,
        to: id_b,
        epoch: pair.b.epoch(),
        activation_tick: pair.a.sim().tick(),
        state: Box::new(offer_state(guid, "player")),
    };
    pair.b
        .peer_message(id_a, Message::HandoffCommit(commit.clone()));
    pair.b.tick_n(space_server::handoff::HANDOFF_LEAD + 5);
    assert!(pair.b.sim().ship_state(guid).is_none());
    assert_eq!(pair.b.stats().ships, 0, "no ship without a player");

    // --- left after the commit arrived, before its install tick
    let mut pair = Pair::new();
    let id_a = pair.a.shard();
    let id_b = pair.b.shard();
    let (_session_a, guid) = offer_and_accept(&mut pair, "player");
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
    let commit = space_protocol::HandoffCommit {
        entity: guid,
        ..commit
    };
    pair.b.peer_message(id_a, Message::HandoffCommit(commit));
    assert!(
        pair.b.sim().ship_state(guid).is_none(),
        "precondition: the commit waits for B's own install tick"
    );
    pair.b.leave(session_b);
    pair.b.tick_n(space_server::handoff::HANDOFF_LEAD + 5);
    assert!(
        pair.b.sim().ship_state(guid).is_none(),
        "a commit for a player who has left must not be installed"
    );
    assert_eq!(pair.b.stats().ships, 0);
    assert_eq!(pair.b.stats().received, 0);
}

/// A handoff state that does not decode is refused when it is offered, and a
/// commit whose state does not decode installs the state its offer staged —
/// never a default.
///
/// The old fallback for an undecodable commit state was `ShipState::IDENTITY`:
/// the ship silently reappeared at the world origin. Refusing at the offer is
/// the one point where a bad state costs nothing, because the offering shard is
/// still simulating the entity.
///
/// Mutation this catches: accepting an offer without decoding its state (first
/// half), and falling back to `ShipState::IDENTITY` on an undecodable commit
/// (second half — the ship would be at the origin, not where it was offered).
#[test]
fn an_undecodable_handoff_state_is_never_installed_as_a_default() {
    let mut pair = Pair::new();
    let id_a = pair.a.shard();
    let id_b = pair.b.shard();
    let guid = EntityGuid::new(id_a, 5);
    let garbage = space_protocol::EntitySnapshot::new(
        guid,
        vec![(
            <ShipState as space_ecs::ComponentDef>::ID,
            vec![0xFF; 7], // not a ShipState
        )],
    );

    // (a) An offer nobody could install is refused, and nothing is staged.
    let bad_offer = space_protocol::HandoffOffer {
        entity: guid,
        from: id_a,
        to: id_b,
        epoch: pair.b.epoch(),
        state: Box::new(garbage.clone()),
        activation_tick: space_server::handoff::HANDOFF_LEAD,
    };
    pair.b.peer_message(id_a, Message::HandoffOffer(bad_offer));
    let messages: Vec<Message> = pair
        .b
        .take_outbox()
        .into_iter()
        .map(|o| o.message)
        .collect();
    assert!(
        !messages
            .iter()
            .any(|m| matches!(m, Message::HandoffAccept(_))),
        "an undecodable state must not be accepted"
    );
    assert!(abort_in(&messages).is_some_and(|a| a.entity == guid));

    // (b) A good offer, then a commit whose state is garbage: B installs the
    // state the offer staged.
    let offered_at = region_center(1).offset([10.0, 20.0, 30.0]);
    let staged = space_protocol::EntitySnapshot::new(
        guid,
        vec![(
            <ShipState as space_ecs::ComponentDef>::ID,
            space_sim::wire::encode_ship_state(&ShipState::at_rest(offered_at)),
        )],
    );
    let offer = space_protocol::HandoffOffer {
        entity: guid,
        from: id_a,
        to: id_b,
        epoch: pair.b.epoch(),
        state: Box::new(staged),
        activation_tick: space_server::handoff::HANDOFF_LEAD,
    };
    pair.b.peer_message(id_a, Message::HandoffOffer(offer));
    assert!(pair
        .b
        .take_outbox()
        .iter()
        .any(|o| matches!(o.message, Message::HandoffAccept(_))));
    let commit = space_protocol::HandoffCommit {
        entity: guid,
        from: id_a,
        to: id_b,
        epoch: pair.b.epoch(),
        activation_tick: space_server::handoff::HANDOFF_LEAD,
        state: Box::new(garbage),
    };
    pair.b.peer_message(id_a, Message::HandoffCommit(commit));
    pair.b.tick_n(space_server::handoff::HANDOFF_LEAD + 5);
    let installed = pair
        .b
        .sim()
        .ship_state(guid)
        .expect("the transfer still completes, with the staged state");
    assert_eq!(
        installed.pos, offered_at,
        "the ship is where its offer put it, not at the world origin"
    );
    assert_ne!(installed.pos, ShipState::IDENTITY.pos);
}

/// Only the player who flies a ship is redirected when it is handed over.
///
/// Everybody who had the ship in sight used to be sent the `Redirect` too. A
/// bystander's client then said hello to the accepting shard with its own
/// token: nothing is staged for it there, so it was sent away again — or, on
/// the shard that owns the spawn point, welcomed as a new player and given a
/// second ship for as long as its client kept that connection. A bystander
/// needs no redirect: it stays on the shard that simulates its own ship and
/// keeps seeing the ship that left as that shard's ghost of it
/// ([`an_observer_sees_no_gap_across_a_handoff`]).
///
/// Mutation this catches: redirecting every client within the interest
/// radius of the ship (`redirect_watchers`).
#[test]
fn only_the_ships_own_player_is_redirected() {
    let mut pair = Pair::new();
    let id_a = pair.a.shard();
    let (mover, guid) = join(&mut pair.a, "mover");
    let (bystander, bystander_ship) = join(&mut pair.a, "bystander");
    assert_ne!(guid, bystander_ship);

    let mut redirects: Vec<(u64, Redirect)> = Vec::new();
    for _ in 0..600 {
        pair.thrust(id_a, mover, 0, 1.0);
        pair.tick();
        for (from, session, message) in std::mem::take(&mut pair.client_frames) {
            if let Message::Redirect(redirect) = message {
                assert_eq!(from, id_a);
                redirects.push((session, redirect));
            }
        }
        if !redirects.is_empty() {
            break;
        }
    }
    // A few more ticks: a redirect to the bystander must not follow either.
    for _ in 0..10 {
        pair.thrust(id_a, mover, 0, 1.0);
        pair.tick();
        for (_, session, message) in std::mem::take(&mut pair.client_frames) {
            if let Message::Redirect(redirect) = message {
                redirects.push((session, redirect));
            }
        }
    }
    assert_eq!(redirects.len(), 1, "one redirect: {redirects:?}");
    let (session, redirect) = &redirects[0];
    assert_eq!(*session, mover, "to the ship's own player");
    assert_eq!(redirect.entity, guid);
    assert!(
        pair.a.ship_of(bystander) == Some(bystander_ship),
        "the bystander is where it was, with the ship it had"
    );
}

/// A hello without a player token is refused, by every shard, before
/// anything is decided on it.
///
/// The token is what a staged ship is matched to its player by, and a
/// transfer that has **no** player (a body; a ship nobody flies) is staged
/// with the empty token. A hello whose token is empty used to *equal* that:
/// it was welcomed by a shard that does not own the spawn point as "the
/// player being handed over", and could then claim the staged entity.
///
/// Mutation this catches: dropping the empty-token refusal in `Shard::join`,
/// or comparing a hello's token to a staged owner token with `==` again.
#[test]
fn a_hello_without_a_token_is_refused_and_owns_nothing_that_is_staged() {
    use space_server::shard::JoinRefusal;
    let refused = |answer: Result<Welcome, JoinRefusal>| {
        let Err(JoinRefusal::Rejected(RejectReason::Other(why))) = answer else {
            panic!("expected a refusal in words, got {answer:?}");
        };
        assert!(why.contains("token"), "{why}");
    };
    // On the shard that owns the spawn point: no session, no ship.
    let mut alone = lone_shard();
    let before = alone.next_session();
    refused(alone.join(&space_net::hello("")));
    assert_eq!(alone.next_session(), before, "no session spent");
    assert_eq!((alone.stats().clients, alone.stats().ships), (0, 0));

    // On a shard that does not own it, with an **ownerless** transfer staged
    // (the offer's state carries the empty owner token): still nobody's.
    let mut pair = Pair::new();
    let (id_a, id_b) = (pair.a.shard(), pair.b.shard());
    let guid = EntityGuid::new(id_a, 77);
    let activation_tick = pair.b.sim().tick() + space_server::handoff::HANDOFF_LEAD;
    pair.b.peer_message(
        id_a,
        Message::HandoffOffer(space_protocol::HandoffOffer {
            entity: guid,
            from: id_a,
            to: id_b,
            epoch: pair.b.epoch(),
            state: Box::new(offer_state(guid, "")),
            activation_tick,
        }),
    );
    assert!(
        pair.b
            .take_outbox()
            .iter()
            .any(|o| matches!(o.message, Message::HandoffAccept(_))),
        "B staged the ownerless transfer"
    );
    refused(pair.b.join(&space_net::hello("")));
    assert_eq!(pair.b.stats().clients, 0);
    // A resident of B with a token of its own cannot claim it either — and
    // neither could one whose token were empty: the staged entity has no
    // owner at all.
    let mut pair = Pair::new();
    let resident = residents_of_b(&mut pair, &["resident".to_owned()])[0];
    let own_ship = pair.b.ship_of(resident).expect("a ship of its own");
    pair.b.peer_message(
        id_a,
        Message::HandoffOffer(space_protocol::HandoffOffer {
            entity: guid,
            from: id_a,
            to: id_b,
            epoch: pair.b.epoch(),
            state: Box::new(offer_state(guid, "")),
            activation_tick,
        }),
    );
    let _ = pair.b.take_outbox();
    pair.b.client_status(
        resident,
        NeighbourStatus {
            entity: guid,
            shard: id_b,
            connected: true,
        },
    );
    assert_eq!(pair.b.ship_of(resident), Some(own_ship), "nothing adopted");
    assert!(
        !pair
            .b
            .take_outbox()
            .iter()
            .any(|o| matches!(o.message, Message::NeighbourStatus(_))),
        "and the commit gate of the offering shard was not opened"
    );
}

/// The map a player is sent — in a `Welcome`, a `JoinRedirect`, a handoff's
/// `Redirect` — names where **players** dial each shard and nothing else: the
/// internal endpoints (where the shards reach each other) are replaced by
/// the client endpoints. The shard's own map keeps the real ones.
///
/// Mutation this catches: sending `self.shard_map.clone()` to a client — every
/// player who says hello learns the cluster's internal addresses.
#[test]
fn a_player_is_sent_client_endpoints_and_no_internal_ones() {
    use space_server::shard::JoinRefusal;
    let mut pair = Pair::new();
    let (id_a, id_b) = (pair.a.shard(), pair.b.shard());
    let internal =
        |n: u8| SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 9, 8, n)), 7_900 + u16::from(n));
    let players = |n: u8| SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 7_870 + u16::from(n));
    let map = ShardMap::new(
        1,
        vec![
            ShardRegion::new(region(0), id_a, internal(1)).with_client_endpoint(players(1)),
            ShardRegion::new(region(1), id_b, internal(2)).with_client_endpoint(players(2)),
        ],
    );
    pair.a.set_shard_map(map.clone());
    pair.b.set_shard_map(map.clone());
    let for_players = map.for_players();
    assert_eq!(for_players.epoch, map.epoch);
    let leaks = |sent: &ShardMap| {
        sent.regions
            .iter()
            .any(|r| r.endpoint == internal(1) || r.endpoint == internal(2))
    };
    assert!(leaks(&map) && !leaks(&for_players));
    for (sent, real) in for_players.regions.iter().zip(&map.regions) {
        assert_eq!((sent.node, sent.shard), (real.node, real.shard));
        assert_eq!(sent.client_endpoint, real.client_endpoint);
        assert_eq!(sent.endpoint, real.client_endpoint);
    }
    // Everything a client derives from the map is unchanged by it.
    assert_eq!(
        space_scene::scene_anchor(&for_players),
        space_scene::scene_anchor(&map)
    );
    assert_eq!(for_players.client_endpoint_of(id_b), Some(players(2)));

    // The welcome.
    let welcome = pair.a.join(&space_net::hello("mover")).expect("join A");
    assert_eq!(welcome.shard_map, for_players);
    // The join redirect of a shard that does not own the spawn point.
    let Err(JoinRefusal::Elsewhere(sent_on)) = pair.b.join(&space_net::hello("other")) else {
        panic!("B sends a newcomer to A");
    };
    assert_eq!(sent_on.shard_map, for_players);
    // The handoff's redirect.
    let mover = welcome.session;
    let mut redirect = None;
    for _ in 0..600 {
        pair.thrust(id_a, mover, 0, 1.0);
        pair.tick();
        for message in pair.frames_for(id_a, mover) {
            if let Message::Redirect(r) = message {
                redirect = Some(r);
            }
        }
        if redirect.is_some() {
            break;
        }
    }
    assert_eq!(redirect.expect("a redirect").shard_map, for_players);
    // The shards themselves still hold, and use, the real map.
    assert_eq!(pair.a.shard_map(), &map);
    assert_eq!(pair.b.shard_map(), &map);
}
