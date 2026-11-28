//! Seamless shard transition, end to end over real QUIC on loopback: two
//! in-process shards (the same boot wiring as the `space-server` binary) and a
//! client link that starts on one and is handed off to the other.
//!
//! The topology is what a split leaves behind: shard A owns a region, and
//! shard B owns one octant carved out of it — the cell the scene's spawn
//! point lies in. A new player starts on the spawn point, so it starts on
//! **B**, whichever of the two it dials: A, which does not own that point,
//! answers a hello with a `JoinRedirect` to B (it used to put the player on
//! the centre of its own region — the corner where its cells meet, which is
//! B's — and hand the ship over at once; that is how these tests got their
//! handoff). The handoff is now flown: the spawn point lies 6 m inside B's
//! cell (`space_scene::ANCHOR_OFFSET`), and a ship thrusting −x crosses into
//! A's ground there. The tests assert the client's half of the
//! make-before-break exchange:
//!
//! * the link follows the `Redirect` to A and reports in, and the handoff
//!   commits (`handoffs == 1`, now on shard A under A's session);
//! * the ship keeps its guid and its place: nothing jumps at the switch;
//! * the player stays in control: thrust sent after the switch moves the ship
//!   on A, and on A only — B no longer simulates it.
//!
//! The map sends players to `0.0.0.0:<port>` for both shards — "the host you
//! dialed, this port" — so both redirects (the join redirect and the
//! handoff's) are followed through the client-endpoint rule.
//!
//! The scene rides along: the link learns it from the shard it joins, takes a
//! newer one from the accepting shard while the handoff is still in flight,
//! never falls back to an older one, and follows a change on whichever shard
//! it is playing on.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use space_client::link::ServerLink;
use space_protocol::{Message, Scene, SceneId, SceneState, ShardId, ShardMap, ShardRegion};
use space_server::net::ShardNet;
use space_server::{Shard, ShardConfig};
use space_world::{NodeRef, WorldPos};
use tokio::sync::Mutex as AsyncMutex;
use tokio::task::JoinHandle;

type SharedShard = Arc<AsyncMutex<Shard>>;

struct Running {
    net: Arc<ShardNet>,
    shard: SharedShard,
    tasks: Vec<JoinHandle<()>>,
}

impl Running {
    fn addr(&self) -> SocketAddr {
        self.net.local_addr().expect("bound address")
    }

    fn stop(self) {
        for task in &self.tasks {
            task.abort();
        }
        self.net.close();
    }
}

/// Bind a shard on loopback and start serving and ticking it at 30 Hz, on
/// the tick clock a standalone `space-server` runs on (`clock::run` with the
/// default tick origin): both shards count the same ticks.
fn start(id: u32, region: NodeRef) -> Running {
    start_with(id, region, |_| {})
}

/// [`start`], with the shard's configuration changed by `tune` first.
fn start_with(id: u32, region: NodeRef, tune: impl FnOnce(&mut ShardConfig)) -> Running {
    let net = Arc::new(ShardNet::bind(SocketAddr::from(([127, 0, 0, 1], 0))).expect("bind"));
    let mut config = ShardConfig::new(ShardId::new(id), net.local_addr().expect("addr"), region);
    tune(&mut config);
    let shard = Arc::new(AsyncMutex::new(Shard::new(config)));
    let serve = tokio::spawn({
        let (net, shard) = (Arc::clone(&net), Arc::clone(&shard));
        async move {
            let _ = net.serve(shard).await;
        }
    });
    let ticker = tokio::spawn(space_server::clock::run(
        Arc::clone(&shard),
        Arc::clone(&net),
        space_server::clock::TickClock::new(space_protocol::SceneClock::DEFAULT_ORIGIN_US)
            .expect("the default origin is in the past"),
    ));
    Running {
        net,
        shard,
        tasks: vec![serve, ticker],
    }
}

/// Poll `condition` every 50 ms for up to `seconds`.
async fn eventually(seconds: u64, mut condition: impl FnMut() -> bool) -> bool {
    for _ in 0..seconds * 20 {
        if condition() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    condition()
}

fn scene(scene: Scene, revision: u64) -> SceneState {
    SceneState::new(scene.id(), revision)
}

/// `0.0.0.0:<port>`: where the map sends a shard's players — the host they
/// dialed, the shard's port.
fn players_at(shard: &Running) -> SocketAddr {
    SocketAddr::from(([0, 0, 0, 0], shard.addr().port()))
}

/// Two neighbouring shards as a split leaves them: A owns a two-cell-wide
/// node; B owns the far octant carved out of it, the cell the scene's spawn
/// point lies in — 6 m from the plane beyond which the ground is A's.
async fn neighbours() -> (Running, Running) {
    let region_a = NodeRef::new(63, [0, 0, 0]);
    let region_b = NodeRef::new(64, [1, 1, 1]);
    let a = start(1, region_a);
    let b = start(2, region_b);
    // Shards reach each other at the addresses they bound; players are sent
    // to "the host you dialed" and the shard's port.
    let map = ShardMap::new(
        1,
        vec![
            ShardRegion::new(region_a, ShardId::new(1), a.addr())
                .with_client_endpoint(players_at(&a)),
            ShardRegion::new(region_b, ShardId::new(2), b.addr())
                .with_client_endpoint(players_at(&b)),
        ],
    );
    // The fixture's premise, under the anchor rule: B owns the scene's spawn
    // point, and the ground 6 m further along −x is A's.
    let spawn = space_scene::scene_anchor(&map).expect("an anchor");
    assert_eq!(map.lookup(&spawn), Some(ShardId::new(2)));
    let beyond = WorldPos::from_f64_metres_around(spawn, [-6.5, 0.0, 0.0]);
    assert_eq!(map.lookup(&beyond), Some(ShardId::new(1)));
    a.shard.lock().await.set_shard_map(map.clone());
    b.shard.lock().await.set_shard_map(map);
    (a, b)
}

/// Join through shard A's address. A does not own the spawn point: it sends
/// the client to B, where the player starts — on the spawn point, with no
/// handoff. Returns the link, once B's first frame has shown the ship.
async fn join_through_a(a: &Running, b: &Running) -> ServerLink {
    let link = ServerLink::join(a.addr())
        .await
        .expect("join through shard A");
    assert!(
        eventually(5, || link.replica().state.is_some()).await,
        "the ship never appeared: {:?}",
        link.replica()
    );
    let joined = link.replica();
    assert_eq!(joined.shard, Some(ShardId::new(2)), "sent on to B");
    assert_eq!(joined.handoffs, 0, "a join redirect is not a handoff");
    assert_eq!(link.entry(), a.addr(), "the address that was dialed");
    assert_eq!(
        link.shard_address(),
        b.addr(),
        "0.0.0.0:<B's port> is B's port on the host that was dialed"
    );
    // On the spawn point, which is where the scene is drawn around.
    let spawn = space_scene::scene_anchor(a.shard.lock().await.shard_map()).expect("an anchor");
    assert_eq!(joined.state.expect("a ship").pos, spawn);
    assert_eq!(a.shard.lock().await.stats().clients, 0, "A let nobody in");
    link
}

/// The plane between the two shards of [`neighbours`], in metres from the
/// spawn point along x: beyond it (`x < BORDER_X`) the ground is A's.
const BORDER_X: f64 = -6.0;

/// Seconds per tick.
fn dt() -> f64 {
    f64::from(space_sim::dt())
}

/// How long a handoff may take from the state that first shows the ship
/// across the border until the link plays on the other shard, in seconds
/// **of the shards' ticks** (see [`Flight`], "How the bounds are taken"): the
/// handoff's lead (the second the shards give the client to connect) and one
/// more. **Catches:** a handoff that is refused and offered again (the
/// re-offer pause is two seconds), a redirect the client follows late.
const HANDOFF_BUDGET: f64 = 2.0;

/// How much further the ship may be, between two states the link showed,
/// than its speed and the **ticks** between those states explain, metres.
/// Ordinary flight shows none at all, on any machine (a state `n` ticks
/// newer is at most `n` ticks at the greater of the two speeds further, and
/// nothing in the number is a time this process measured), so this is a
/// floor for rounding and no more. **Catches:** a tick flown twice at the
/// switch (0.45 m at the 14 m/s of the first crossing, over a metre on the
/// way back), a ship put back on a spawn point (metres), a state the
/// accepting shard carried further than the ticks its commit was late by.
const JUMP_SLACK: f64 = 0.05;

/// How far the ship may be seen to move **against** the way it flies.
const BACKWARD_SLACK: f64 = 0.1;

/// The most a ship's speed changes in a second, m/s²: full thrust from rest
/// (`ShipParams::PLACEHOLDER`: thrust / mass). Between two states `t` apart
/// the speed may have been up to `ACCEL · t / 2` above both ends — when the
/// pilot turned round in between — which is `ACCEL · t² / 4` of flight; twice
/// that is allowed (3 mm over one tick).
const ACCEL: f64 = 6.0;

fn norm(v: [f64; 3]) -> f64 {
    v.iter().map(|c| c * c).sum::<f64>().sqrt()
}

/// One state of the link's ship, as a look found it: where the shard put it,
/// how fast it was, and the tick that state is from.
#[derive(Clone, Copy)]
struct Look {
    pos: [f64; 3],
    vel: [f64; 3],
    /// The tick of the state ([`space_client::link::Replica::state_tick`]).
    tick: u64,
    /// The timeline that tick is on (`Replica::tick_epoch`).
    epoch: u64,
    /// The handoffs the link had completed.
    handoffs: u64,
    /// When the link last heard a newer tick. **Not part of any bound**: it
    /// is in the log a failure prints, to tell a late frame from a jump.
    heard: std::time::Instant,
}

/// What two consecutive [`Look`]s say about the flight between them.
struct Step {
    /// Ticks from the first state to the second.
    ticks: i64,
    /// How far the ship moved, metres.
    moved: f64,
    /// How much further that is than its speed and the ticks explain.
    jump: f64,
    /// The same against the time between the two states' *arrivals* — what
    /// this test used to bound. Printed, never asserted.
    jump_by_arrival: f64,
    /// How far the ship moved against its velocity; `None` while it turns
    /// round or rests.
    back: Option<f64>,
}

/// A flight under watch. Every [`Flight::look`] notes the state the link
/// shows when it is a new one; [`Flight::assert_seamless`] then goes through
/// them in pairs:
///
/// * **a jump**: between two states the ship was further than its speed and
///   the ticks between them explain ([`JUMP_SLACK`]);
/// * **a step back**: the ship moved against its own velocity
///   ([`BACKWARD_SLACK`]) — a state from before the handoff's commit shown
///   after it;
/// * **a slow handoff**: the ticks from the state that first showed the ship
///   across [`BORDER_X`] to the state the link showed when it had changed
///   shard ([`HANDOFF_BUDGET`]);
/// * **a second session**: the most clients the spawn point's shard had at
///   once. A player has one session there at a time — and a ship of its own
///   only with a session — so more than there are players is a second ship
///   somebody was given.
///
/// # How the bounds are taken
///
/// These tests run real shard loops in real time on whatever machine runs
/// them, so **nothing is measured against this process's clock** (the rule
/// of `tests/seamless.rs`, whose module docs say why). A state's time is the
/// tick the shard stamped it with: both shards count the same ticks
/// (`clock::run` on the default origin), the flight asserts that the link's
/// ticks never started over, and a state that is `n` ticks newer is `n/30 s`
/// of flight further — however late its frame arrived, however many frames
/// arrived at once and whenever this task got to look.
///
/// (The jump used to be taken against the time between the frames'
/// *arrivals*, with half a metre of slack — "a tick at 15 m/s" — as a maximum
/// over the whole flight. The ship is at 35 m/s on its way back, a tick is
/// then 1.17 m, and one frame that is heard 17 ms late followed by one that
/// is on time is 0.57 m "more than flying explains": on a busy CI runner the
/// test failed with exactly that, in ordinary flight 16 ticks from any
/// change of shard and with every state on its tick to the millimetre. A
/// failure prints the states around it with their ticks and arrival times,
/// so that it says which of the two it is; `SPACE_HANDOFF_TRACE=1` prints
/// them for the worst arrival of a flight that passed.)
struct Flight<'a> {
    link: &'a ServerLink,
    spawn_shard: &'a Running,
    /// Every state the looks found, in order.
    log: Vec<Look>,
    beyond: bool,
    /// The tick of the state that first showed the ship across the border,
    /// until the handoff that follows it.
    crossed: Option<u64>,
    handoffs: u64,
    /// How long each handoff took, seconds of ticks.
    handoff_times: Vec<f64>,
    most_spawn_clients: usize,
}

impl<'a> Flight<'a> {
    fn new(link: &'a ServerLink, spawn_shard: &'a Running) -> Self {
        let handoffs = link.replica().handoffs;
        let beyond = link.offset()[0] < BORDER_X;
        Self {
            link,
            spawn_shard,
            log: Vec::new(),
            beyond,
            crossed: None,
            handoffs,
            handoff_times: Vec::new(),
            most_spawn_clients: 0,
        }
    }

    fn look(&mut self) {
        let replica = self.link.replica();
        if let (Some(state), Some(heard)) = (replica.state, replica.tick_seen) {
            let now = Look {
                pos: replica.offset(),
                vel: state.vel.map(f64::from),
                tick: replica.state_tick(),
                epoch: replica.tick_epoch,
                handoffs: replica.handoffs,
                heard,
            };
            let known = self.log.last().is_some_and(|last| {
                (last.epoch, last.tick, last.handoffs) == (now.epoch, now.tick, now.handoffs)
            });
            if !known {
                self.log.push(now);
            }
            let beyond = now.pos[0] < BORDER_X;
            if beyond != self.beyond {
                self.beyond = beyond;
                self.crossed = Some(now.tick);
            }
            if replica.handoffs != self.handoffs {
                self.handoffs = replica.handoffs;
                let crossed = self.crossed.take().expect("a handoff follows a crossing");
                let took = now.tick.saturating_sub(crossed) as f64 * dt();
                self.handoff_times.push(took);
            }
        }
        if let Ok(shard) = self.spawn_shard.shard.try_lock() {
            self.most_spawn_clients = self.most_spawn_clients.max(shard.stats().clients);
        }
    }

    /// Thrust along x (`-1.0` towards A's ground, `1.0` back) until the link
    /// has been handed over `handoffs` times in all, looking every 10 ms.
    async fn fly(&mut self, thrust_x: f32, handoffs: u64) -> space_client::link::Replica {
        self.link.set_thrust([thrust_x, 0.0, 0.0]);
        for _ in 0..3_000 {
            self.look();
            // (What the look saw, not a second reading of the link: a
            // handoff that completes between the two would end the flight
            // without ever having been looked at.)
            if self.handoffs >= handoffs {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        self.link.set_thrust([0.0; 3]);
        let arrived = self.link.replica();
        assert_eq!(
            arrived.handoffs, handoffs,
            "the client never arrived on the other shard: {arrived:?}"
        );
        arrived
    }

    /// What the states `index - 1` and `index` of the log say about the
    /// flight between them (see [`Step`]).
    fn step(&self, index: usize) -> Step {
        let (last, now) = (self.log[index - 1], self.log[index]);
        let moved = [0, 1, 2].map(|i| now.pos[i] - last.pos[i]);
        let speed = norm(now.vel).max(norm(last.vel));
        let ticks = now.tick as i64 - last.tick as i64;
        // A state that is not newer by its tick explains no flight at all.
        let flown = ticks.max(0) as f64 * dt();
        let heard = now.heard.saturating_duration_since(last.heard);
        // Flying one way (not turning round, not at rest): the ship does not
        // move the other way.
        let same_way = (0..3).map(|i| now.vel[i] * last.vel[i]).sum::<f64>() > 0.0;
        let back = (same_way && norm(now.vel) > 1.0 && norm(last.vel) > 1.0)
            .then(|| -(0..3).map(|i| moved[i] * now.vel[i]).sum::<f64>() / norm(now.vel));
        Step {
            ticks,
            moved: norm(moved),
            jump: norm(moved) - speed * flown - ACCEL * flown * flown / 2.0,
            jump_by_arrival: norm(moved) - speed * heard.as_secs_f64(),
            back,
        }
    }

    /// The log around state `index`, a line a state: what a failure prints,
    /// so that it says by itself whether it happened at a change of shard or
    /// in ordinary flight, and how the frames around it arrived.
    fn around(&self, index: usize) -> String {
        use std::fmt::Write as _;
        let switches: Vec<usize> = (1..self.log.len())
            .filter(|&i| self.log[i].handoffs != self.log[i - 1].handoffs)
            .collect();
        let mut text = String::new();
        let first = index.saturating_sub(8).max(1);
        let last = (index + 4).min(self.log.len().saturating_sub(1));
        for i in first..=last {
            let (look, step) = (self.log[i], self.step(i));
            let heard = look
                .heard
                .saturating_duration_since(self.log[i - 1].heard)
                .as_secs_f64();
            // Ticks to the nearest change of shard, before or after.
            let to_switch = switches
                .iter()
                .map(|&s| look.tick as i64 - self.log[s].tick as i64)
                .min_by_key(|ticks| ticks.abs());
            let _ = writeln!(
                text,
                "  {} state {i:4}: tick {} ({:+} ticks, heard {:+7.1} ms after the one before), \
                 x {:+8.3} m, {:6.3} m/s, moved {:.4} m, jump {:+.4} m (by arrival {:+.4} m), \
                 handoffs {}{}{}",
                if i == index { ">" } else { " " },
                look.tick,
                step.ticks,
                heard * 1e3,
                look.pos[0],
                norm(look.vel),
                step.moved,
                step.jump,
                step.jump_by_arrival,
                look.handoffs,
                to_switch.map_or(String::new(), |ticks| format!(
                    ", {ticks:+} ticks from the nearest change of shard"
                )),
                if switches.contains(&i) {
                    " <- the first state from the other shard"
                } else {
                    ""
                },
            );
        }
        text
    }

    /// What the flight must not have shown (see the type docs).
    fn assert_seamless(&self, players: usize) {
        let steps: Vec<(usize, Step)> = (1..self.log.len()).map(|i| (i, self.step(i))).collect();
        let worst = |of: &dyn Fn(&Step) -> f64| {
            steps
                .iter()
                .map(|(i, step)| (*i, of(step)))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap_or((0, f64::MIN))
        };
        let (jump_at, jump) = worst(&|step| step.jump);
        let (back_at, back) = worst(&|step| step.back.unwrap_or(f64::MIN));
        let (arrival_at, by_arrival) = worst(&|step| step.jump_by_arrival);
        // (Shown with `--nocapture`: the numbers behind the bounds.)
        eprintln!(
            "flight: {} states, handoffs {:.3?} s of ticks, worst jump {jump:+.4} m, worst step \
             back {back:+.3} m, most sessions on the spawn shard {}; not a bound: against the \
             arrival times the worst jump would be {by_arrival:+.3} m (state {arrival_at})",
            self.log.len(),
            self.handoff_times,
            self.most_spawn_clients
        );
        if std::env::var_os("SPACE_HANDOFF_TRACE").is_some() && arrival_at > 0 {
            eprintln!("around state {arrival_at}:\n{}", self.around(arrival_at));
        }
        // One timeline: the shards count the same ticks, so every tick of
        // the log can be compared with every other.
        assert!(
            self.log.iter().all(|look| look.epoch == self.log[0].epoch),
            "the link's ticks started over: the states are not on one timeline"
        );
        assert!(
            jump <= JUMP_SLACK,
            "the ship jumped: {jump:.3} m more between two states than its speed and the \
             ticks between them explain\n{}",
            self.around(jump_at)
        );
        assert!(
            back <= BACKWARD_SLACK,
            "the ship stepped back by {back:.2} m\n{}",
            self.around(back_at)
        );
        for (n, took) in self.handoff_times.iter().enumerate() {
            assert!(
                *took <= HANDOFF_BUDGET,
                "handoff {} took {took:.2} s of ticks from the border to the other shard: {:?}",
                n + 1,
                self.handoff_times
            );
        }
        assert!(
            self.most_spawn_clients <= players,
            "the spawn point's shard had {} sessions for {players} player(s)",
            self.most_spawn_clients
        );
    }
}

/// Thrust −x until the ship has been handed to shard A, watching that it
/// never jumps on the way and that the handoff is prompt ([`Flight`]).
/// Returns the replica as it arrived.
async fn fly_to_a(link: &ServerLink, spawn_shard: &Running) -> space_client::link::Replica {
    let mut flight = Flight::new(link, spawn_shard);
    let arrived = flight.fly(-1.0, 1).await;
    assert_eq!(arrived.shard, Some(ShardId::new(1)), "{arrived:?}");
    // Seamless: across the whole flight — the switch included — the ship
    // never moved further between two frames than its speed explains (a
    // ship that was reset to a spawn point would have jumped by metres, or
    // by half a cell), and never backwards.
    flight.assert_seamless(1);
    assert_eq!(flight.handoff_times.len(), 1);
    // And on one timeline: the shards count the same ticks, so the link's
    // ticks never started over.
    assert_eq!(arrived.tick_epoch, 0, "{arrived:?}");
    let at_switch = arrived.offset();
    assert!(
        at_switch[0] < BORDER_X && at_switch[0] > -60.0,
        "it is just across the border at x = -6: {at_switch:?}"
    );
    assert!(
        at_switch[1].abs() < 0.5 && at_switch[2].abs() < 0.5,
        "and went nowhere else: {at_switch:?}"
    );
    arrived
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_client_follows_its_ship_to_the_neighbouring_shard() {
    let (a, b) = neighbours().await;
    // The cluster's scene was just changed and the change has reached A but
    // not B yet: B, where the player starts, still says the galaxy, A already
    // the material chart.
    assert!(b.shard.lock().await.set_scene(scene(Scene::Galaxy, 2)));
    assert!(a.shard.lock().await.set_scene(scene(Scene::Materials, 3)));

    let link = join_through_a(&a, &b).await;
    let first_session = link.replica().session;
    let ship = link.replica().ship.expect("a ship");
    // The scene arrives behind the welcome: B's.
    assert!(
        eventually(5, || link.scene().is_set()).await,
        "the link was never told a scene: {:?}",
        link.replica()
    );
    assert_eq!(link.scene(), scene(Scene::Galaxy, 2));

    // The handoff: B redirects, the link connects to A — at "the host it
    // dialed" and A's port — and reports in, the shards commit, and the first
    // frame from A showing our ship is the switch.
    let arrived = fly_to_a(&link, &b).await;
    assert_eq!(arrived.ship, Some(ship), "the same ship, by guid");
    assert!(!link.is_closed(), "the new connection is the live one");
    assert_eq!(link.shard_address(), a.addr());
    assert_eq!(link.entry(), a.addr(), "the entry address never changes");
    // The newer scene won (it was heard on the handoff's second connection,
    // before the switch or with it), and the replica and the link agree.
    assert_eq!(arrived.scene, scene(Scene::Materials, 3));
    assert_eq!(link.scene(), scene(Scene::Materials, 3));
    assert_eq!(link.scene().scene.or_default(), Scene::Materials);
    let at_switch = arrived.offset();

    // A simulates it now, under the session A gave us; B has let go.
    {
        let a_shard = a.shard.lock().await;
        assert_eq!(a_shard.session_of(ship), Some(arrived.session));
        assert!(
            a_shard.sim().ship_state(ship).is_some(),
            "A simulates the ship"
        );
    }
    assert!(
        eventually(5, || b
            .shard
            .try_lock()
            .is_ok_and(|b_shard| b_shard.sim().ship_state(ship).is_none()))
        .await,
        "B still simulates the ship after the commit"
    );
    let _ = first_session;

    // Still in control: thrust further into A's region and watch the
    // authoritative ship move there.
    link.set_thrust([-1.0, 0.0, 0.0]);
    assert!(
        eventually(10, || link.offset()[0] < at_switch[0] - 2.0).await,
        "thrust after the handoff never moved the ship: {:?}",
        link.replica()
    );
    let after = link.replica();
    assert_eq!(after.handoffs, 1, "no second transition: {after:?}");
    assert_eq!(after.shard, Some(ShardId::new(1)));
    let server_side = a
        .shard
        .lock()
        .await
        .sim()
        .ship_state(ship)
        .expect("still on A");
    assert!(
        server_side.vel[0] < 0.0,
        "A applied the thrust: {server_side:?}"
    );
    link.set_thrust([0.0; 3]);

    // The scene changes again: the shard the client is on now tells it. The
    // old shard catching up afterwards reaches nobody (it has no client), and
    // an older revision is not a change at all.
    assert!(a.shard.lock().await.set_scene(scene(Scene::Materials, 4)));
    assert!(
        eventually(5, || link.scene() == scene(Scene::Materials, 4)).await,
        "the change on the new shard never reached the client: {}",
        link.scene()
    );
    assert!(b.shard.lock().await.set_scene(scene(Scene::Materials, 4)));
    assert!(!a.shard.lock().await.set_scene(scene(Scene::Galaxy, 4)));
    assert!(!a.shard.lock().await.set_scene(scene(Scene::Galaxy, 3)));
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(link.scene(), scene(Scene::Materials, 4));
    assert_eq!(link.replica().handoffs, 1);

    link.close();
    a.stop();
    b.stop();
}

/// The other order: the scene change has reached the shard the client joins
/// but not yet the one it is handed to. Arriving there must not roll the scene
/// back, and the accepting shard catching up is not a change.
///
/// Mutation this catches: taking the accepting shard's announcement at the
/// switch whatever its revision — the client would briefly draw the old scene
/// again after every handoff that races a scene change.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_handoff_to_a_shard_that_is_behind_does_not_roll_the_scene_back() {
    let (a, b) = neighbours().await;
    // (The player starts on B, which owns the spawn point, and is handed to
    // A: B has the change, A not yet.)
    assert!(b.shard.lock().await.set_scene(scene(Scene::Materials, 5)));
    assert!(a.shard.lock().await.set_scene(scene(Scene::Galaxy, 4)));

    let link = join_through_a(&a, &b).await;
    assert!(
        eventually(5, || link.scene() == scene(Scene::Materials, 5)).await,
        "the link never learned the scene of the shard it joined: {}",
        link.scene()
    );
    let arrived = fly_to_a(&link, &b).await;
    assert_eq!(arrived.shard, Some(ShardId::new(1)));
    assert_eq!(
        link.scene(),
        scene(Scene::Materials, 5),
        "the shard that is behind must not win"
    );

    // A catches up: the same revision, so nothing changes for the client...
    assert!(a.shard.lock().await.set_scene(scene(Scene::Materials, 5)));
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(link.scene(), scene(Scene::Materials, 5));
    // ...and the next change is followed.
    assert!(a.shard.lock().await.set_scene(scene(Scene::Galaxy, 6)));
    assert!(
        eventually(5, || link.scene() == scene(Scene::Galaxy, 6)).await,
        "{}",
        link.scene()
    );

    link.close();
    a.stop();
    b.stop();
}

/// Out of the spawn point's shard and back **into** it, twice: every
/// crossing is one prompt handoff, the ship never jumps, and the shard that
/// owns the spawn point never has a second session for the player.
///
/// The way in is the one that could go wrong: B owns the spawn point, so a
/// hello B does not recognise as "the player of the ship being handed over"
/// is a fresh join there — a welcome and a **new ship** on the spawn point —
/// where on any other shard it is merely sent away. B recognises the player
/// because the redirect is sent only after B has staged the ship.
///
/// Mutation this catches: a redirect sent with the offer (the hello can
/// reach B first: a second ship on the spawn point, a handoff that takes
/// seconds or never completes), a state shown after the switch that is older
/// than one shown before it, a ship put back on a spawn point.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_ship_flies_out_of_the_spawn_shard_and_back_in_seamlessly() {
    let (a, b) = neighbours().await;
    let link = join_through_a(&a, &b).await;
    let ship = link.replica().ship.expect("a ship");
    let (id_a, id_b) = (ShardId::new(1), ShardId::new(2));

    let mut flight = Flight::new(&link, &b);
    for round in 0..2u64 {
        // Out, over x = -6 into A's ground.
        let out = flight.fly(-1.0, 2 * round + 1).await;
        assert_eq!(out.shard, Some(id_a), "{out:?}");
        assert_eq!(out.ship, Some(ship), "the same ship, by guid");
        assert!(out.offset()[0] < BORDER_X, "{:?}", out.offset());
        assert_eq!(link.shard_address(), a.addr());
        // Round, and back in over the same border: into the shard that owns
        // the spawn point.
        let back = flight.fly(1.0, 2 * round + 2).await;
        assert_eq!(back.shard, Some(id_b), "{back:?}");
        assert_eq!(back.ship, Some(ship), "the same ship, by guid");
        let at = back.offset();
        assert!(
            at[0] > BORDER_X && at[0] < 60.0,
            "just back across the border, not on the spawn point: {at:?}"
        );
        assert!(at[1].abs() < 0.5 && at[2].abs() < 0.5, "{at:?}");
        assert_eq!(link.shard_address(), b.addr());
        // B simulates it under the session it gave the player for it; no
        // other ship was made for that player.
        let b_shard = b.shard.lock().await;
        assert_eq!(b_shard.session_of(ship), Some(back.session));
        assert_eq!(b_shard.ship_of(back.session), Some(ship));
    }
    flight.assert_seamless(1);
    assert_eq!(flight.handoff_times.len(), 4, "{:?}", flight.handoff_times);
    assert_eq!(link.entry(), a.addr(), "the entry address never changes");
    assert!(!link.is_closed());
    // Each shard has let go of what it handed over: one session in all.
    assert!(
        eventually(5, || {
            let (Ok(a), Ok(b)) = (a.shard.try_lock(), b.shard.try_lock()) else {
                return false;
            };
            (a.stats().clients, b.stats().clients) == (0, 1)
        })
        .await,
        "sessions left over: A {:?}, B {:?}",
        a.shard.lock().await.stats(),
        b.shard.lock().await.stats()
    );

    link.close();
    a.stop();
    b.stop();
}

/// Somebody else's ship crossing a border is nothing a bystander acts on: it
/// keeps its one session on the shard its own ship is on, and is given no
/// second ship anywhere.
///
/// Both players fly out to A. Then one flies back into B — the shard that
/// owns the spawn point, where a hello that is not recognised is welcomed as
/// a new player — while the other stays on A and watches. Every client that
/// had a crossing ship in sight used to be redirected with it; the bystander
/// then said hello to B and was given a second ship on the spawn point for
/// ten seconds.
///
/// Mutation this catches: redirecting the watchers of a ship
/// (`redirect_watchers`), or a client that follows a redirect for a ship that
/// is not its own.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_bystander_keeps_its_one_session_when_somebody_elses_ship_crosses() {
    let (a, b) = neighbours().await;
    let (id_a, id_b) = (ShardId::new(1), ShardId::new(2));
    let mover = join_through_a(&a, &b).await;
    let bystander = join_through_a_beside(&a).await;
    let (mover_ship, bystander_ship) = (
        mover.replica().ship.expect("a ship"),
        bystander.replica().ship.expect("a ship"),
    );
    assert_ne!(mover_ship, bystander_ship);

    // Both out to A (each its own handoff), the bystander first; it then
    // brakes and stays there.
    let mut watch = Flight::new(&bystander, &b);
    let out = watch.fly(-1.0, 1).await;
    assert_eq!(out.shard, Some(id_a));
    bystander.set_thrust([1.0, 0.0, 0.0]);
    assert!(
        eventually(10, || bystander
            .replica()
            .state
            .is_some_and(|s| s.vel[0] > -0.5))
        .await,
        "the bystander never came to rest: {:?}",
        bystander.replica()
    );
    bystander.set_thrust([0.0; 3]);
    let stays_at = bystander.offset();
    assert!(stays_at[0] < BORDER_X - 2.0, "well inside A: {stays_at:?}");
    let session_on_a = bystander.replica().session;

    let mut flight = Flight::new(&mover, &b);
    assert_eq!(flight.fly(-1.0, 1).await.shard, Some(id_a));
    // Everybody is on A: nobody has a session on the spawn point's shard.
    assert!(
        eventually(5, || b
            .shard
            .try_lock()
            .is_ok_and(|b| b.stats().clients == 0))
        .await
    );
    // The mover flies back into B, the bystander looking on from A.
    let mut flight = Flight::new(&mover, &b);
    let back = flight.fly(1.0, 2).await;
    assert_eq!(back.shard, Some(id_b));
    // One session on B at most, the whole time: the mover's. The bystander
    // never said hello there.
    flight.assert_seamless(1);
    // Give a wrongly redirected bystander the time to show.
    tokio::time::sleep(Duration::from_millis(500)).await;
    let watching = bystander.replica();
    assert_eq!(watching.handoffs, 1, "{watching:?}");
    assert_eq!(watching.shard, Some(id_a));
    assert_eq!(watching.session, session_on_a, "the session it had");
    assert_eq!(bystander.shard_address(), a.addr());
    {
        let (a_shard, b_shard) = (a.shard.lock().await, b.shard.lock().await);
        assert_eq!(b_shard.stats().clients, 1, "B: the mover and nobody else");
        assert_eq!(b_shard.session_of(mover_ship), Some(back.session));
        assert_eq!(a_shard.session_of(bystander_ship), Some(session_on_a));
        assert_eq!(a_shard.ship_of(session_on_a), Some(bystander_ship));
    }
    // And it still sees the ship that left, across the border (a ghost).
    assert!(
        watching.others().any(|other| other.guid == mover_ship),
        "the bystander lost sight of the ship that crossed: {watching:?}"
    );

    mover.close();
    bystander.close();
    a.stop();
    b.stop();
}

/// The player disconnects while its handoff is in flight: the link has
/// followed the redirect and reported in on the accepting shard A, and the
/// commit has not happened yet (the shards give it a lead of a second). Closing
/// the link drops **both** connections at once. Neither shard may keep the
/// player: no session, no ship, no staged transfer left over to be installed
/// later for nobody. A player who joins afterwards gets exactly one ship.
///
/// Not covered here: dropping only the second connection while the first stays
/// (the old shard keeping the feed). `ServerLink` has no way to drop just its
/// pending connection from outside, so that needs a hook in the link (or a
/// stand-in shard that ends the connection, which is the server dropping, not
/// the client).
///
/// Mutation this catches: a shard that keeps the ship of a session whose
/// connection is gone (`leave` not aborting the staged transfer or not
/// dropping the pending commit): a ship owned by nobody, stranded on a shard
/// that no longer has a player.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_client_that_disconnects_mid_handoff_leaves_nothing_behind() {
    let (a, b) = neighbours().await;
    let link = join_through_a(&a, &b).await;
    let ship = link.replica().ship.expect("a ship");

    // Fly towards A until A has the player's second connection (it has a
    // session there) while the link still plays on B.
    link.set_thrust([-1.0, 0.0, 0.0]);
    let mut mid_handoff = false;
    for _ in 0..6_000 {
        let on_a = a
            .shard
            .try_lock()
            .is_ok_and(|a_shard| a_shard.stats().clients > 0);
        if on_a {
            mid_handoff = link.replica().handoffs == 0;
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    link.set_thrust([0.0; 3]);
    assert!(
        mid_handoff,
        "never caught the link between its report on A and the commit: {:?}",
        link.replica()
    );

    // The client goes away, both connections with it.
    link.close();
    drop(link);

    assert!(
        eventually(10, || {
            let (Ok(a), Ok(b)) = (a.shard.try_lock(), b.shard.try_lock()) else {
                return false;
            };
            let (sa, sb) = (a.stats(), b.stats());
            (sa.clients, sa.ships, sb.clients, sb.ships) == (0, 0, 0, 0)
        })
        .await,
        "state left behind by the disconnected player: A {:?}, B {:?}",
        a.shard.lock().await.stats(),
        b.shard.lock().await.stats()
    );
    // Not even later, when a commit that crossed the disconnect would be
    // installed.
    tokio::time::sleep(Duration::from_secs(2)).await;
    {
        let (a_shard, b_shard) = (a.shard.lock().await, b.shard.lock().await);
        for shard in [&a_shard, &b_shard] {
            assert_eq!(shard.stats().ships, 0, "a stranded ship");
            assert_eq!(shard.stats().clients, 0);
            assert!(shard.sim().ship_state(ship).is_none());
            assert_eq!(shard.session_of(ship), None);
        }
    }

    // A fresh player is welcomed normally: one session, one ship, in all.
    let again = join_through_a(&a, &b).await;
    assert_eq!(again.replica().handoffs, 0);
    tokio::time::sleep(Duration::from_millis(500)).await;
    {
        let (a_shard, b_shard) = (a.shard.lock().await, b.shard.lock().await);
        assert_eq!(a_shard.stats().clients + b_shard.stats().clients, 1);
        assert_eq!(a_shard.stats().ships + b_shard.stats().ships, 1);
    }

    again.close();
    a.stop();
    b.stop();
}

/// A second player, joined through A like the first: it starts beside the
/// first one, on the spawn point's shard.
async fn join_through_a_beside(a: &Running) -> ServerLink {
    let link = ServerLink::join(a.addr())
        .await
        .expect("join through shard A");
    assert!(
        eventually(5, || link.replica().state.is_some()).await,
        "the ship never appeared: {:?}",
        link.replica()
    );
    assert_eq!(link.replica().shard, Some(ShardId::new(2)), "sent on to B");
    link
}

// ---------------------------------------------------------------------------
// The link against stand-in shards: what it does with a redirect that is not
// for its ship, and with a handoff's connection that the accepting shard ends.
// ---------------------------------------------------------------------------

/// A stand-in shard: it welcomes every player hello with the map it is given
/// and hands the test the connection. It simulates nothing; the test says
/// what it sends.
struct StandIn {
    addr: SocketAddr,
    arrivals: tokio::sync::mpsc::UnboundedReceiver<(space_net::Conn, String)>,
    map: Arc<std::sync::Mutex<ShardMap>>,
    task: JoinHandle<()>,
}

impl StandIn {
    fn start(shard: ShardId, first_session: u64) -> Self {
        let (config, _cert) = space_net::crypto::make_server_config();
        let endpoint = space_net::server_endpoint(SocketAddr::from(([127, 0, 0, 1], 0)), config)
            .expect("bind");
        let addr = endpoint.local_addr().expect("addr");
        let map = Arc::new(std::sync::Mutex::new(ShardMap::new(1, Vec::new())));
        let (tx, arrivals) = tokio::sync::mpsc::unbounded_channel();
        let task = tokio::spawn({
            let map = Arc::clone(&map);
            async move {
                let mut session = first_session;
                loop {
                    let Ok((conn, hello)) = space_net::accept_connection(endpoint.accept()).await
                    else {
                        return;
                    };
                    let shard_map = map.lock().expect("map").clone();
                    let welcome = space_protocol::Welcome {
                        session,
                        shard,
                        shard_map,
                    };
                    session += 1;
                    if conn.send_control(&Message::Welcome(welcome)).await.is_ok() {
                        let _ = tx.send((conn, hello.player_token));
                    }
                }
            }
        });
        Self {
            addr,
            arrivals,
            map,
            task,
        }
    }

    /// The next player connection, if one arrives within `within`.
    async fn arrival(&mut self, within: Duration) -> Option<(space_net::Conn, String)> {
        tokio::time::timeout(within, self.arrivals.recv())
            .await
            .ok()
            .flatten()
    }
}

/// The link follows a redirect only for the ship it flies, and a handoff's
/// connection that the accepting shard has ended — with a last word, or
/// without one — is not kept: the next redirect dials that shard anew.
///
/// The accepting shard's last word to a client whose adoption never came is
/// a `JoinRedirect`, or — while it does not know where a new player starts —
/// a `Reject(Starting)`. The link used to take only the first for what it
/// is. After the second it kept the dead connection as "the connection to
/// that shard" for ten seconds and answered every further redirect there by
/// "reporting in again" on it: no handoff to that shard could complete until
/// the ten seconds had passed.
///
/// Mutation this catches: a `Reject` on an established connection handled as
/// a frame (ignored); a pending connection reused although it is closed; a
/// redirect followed whatever entity it names.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_handoff_connection_the_accepting_shard_ended_is_not_reused() {
    use space_protocol::{EntityGuid, EntitySnapshot, Redirect, Reject, RejectReason, Snapshot};
    let (id_a, id_b) = (ShardId::new(1), ShardId::new(2));
    let mut shard_a = StandIn::start(id_a, 7);
    let mut shard_b = StandIn::start(id_b, 40);
    let map = ShardMap::new(
        1,
        vec![
            ShardRegion::new(NodeRef::new(64, [0, 0, 0]), id_a, shard_a.addr),
            ShardRegion::new(NodeRef::new(64, [1, 0, 0]), id_b, shard_b.addr),
        ],
    );
    *shard_a.map.lock().expect("map") = map.clone();
    *shard_b.map.lock().expect("map") = map.clone();

    let link = ServerLink::join(shard_a.addr).await.expect("join A");
    let (on_a, token) = shard_a
        .arrival(Duration::from_secs(5))
        .await
        .expect("the link's connection");
    assert_eq!(token, link.player_token());
    // A's first frame names the link's ship.
    let ours = EntityGuid::new(id_a, 1);
    let theirs = EntityGuid::new(id_a, 2);
    let spawn = space_scene::scene_anchor(&map).expect("an anchor");
    let ship = |guid: EntityGuid, session: u64| {
        EntitySnapshot::new(
            guid,
            vec![
                (
                    <space_sim::Owner as space_ecs::ComponentDef>::ID,
                    space_protocol::codec::encode_raw(&session),
                ),
                (
                    <space_sim::ShipState as space_ecs::ComponentDef>::ID,
                    space_sim::wire::encode_ship_state(&space_sim::ShipState::at_rest(spawn)),
                ),
            ],
        )
    };
    on_a.send(&Message::Snapshot(Snapshot {
        tick: 1,
        shard: id_a,
        epoch: 1,
        entities: vec![ship(ours, 7), ship(theirs, 8)],
        input_ack: 0,
        more: false,
    }))
    .await
    .expect("the snapshot");
    assert!(eventually(5, || link.replica().ship == Some(ours)).await);
    let redirect = |entity: EntityGuid| {
        Message::Redirect(Redirect {
            entity,
            new_shard: id_b,
            shard_map: map.clone(),
            activation_tick: 100,
        })
    };

    // Somebody else's ship is handed to B: nothing for this client to do.
    on_a.send(&redirect(theirs)).await.expect("a redirect");
    assert!(
        shard_b.arrival(Duration::from_millis(700)).await.is_none(),
        "the link said hello to B for a ship that is not its own"
    );

    // Its own ship: it dials B and reports in.
    on_a.send(&redirect(ours)).await.expect("a redirect");
    let (first, token) = shard_b
        .arrival(Duration::from_secs(5))
        .await
        .expect("the link follows the redirect for its own ship");
    assert_eq!(token, link.player_token(), "the same player token");
    // B gives the handoff up and says so with "starting" — and (unlike a
    // real shard) leaves the connection open, so only the word can tell.
    first
        .send_reliable(&Message::Reject(Reject {
            reason: RejectReason::Starting,
        }))
        .await
        .expect("the last word");
    assert!(
        eventually(5, || first.is_closed()).await,
        "the link kept a connection the shard had ended with a Reject"
    );
    // The shards try again: a new connection, not a status on the dead one.
    on_a.send(&redirect(ours)).await.expect("a redirect");
    let (second, _) = shard_b
        .arrival(Duration::from_secs(3))
        .await
        .expect("the next redirect dials B anew");

    // This time B just goes away, without a word.
    second.close(0, "gone");
    tokio::time::sleep(Duration::from_millis(300)).await;
    on_a.send(&redirect(ours)).await.expect("a redirect");
    let (third, _) = shard_b
        .arrival(Duration::from_secs(3))
        .await
        .expect("a closed connection is not a way to B either");

    // B lets it in as a new player after all (the transfer was gone when the
    // hello came): a ship of its own under the new session. That is not the
    // switch, and the link leaves again at once instead of keeping a second
    // ship in the world.
    let placeholder = EntityGuid::new(id_b, 1);
    third
        .send(&Message::Snapshot(Snapshot {
            tick: 5,
            shard: id_b,
            epoch: 1,
            entities: vec![ship(placeholder, 42)],
            input_ack: 0,
            more: false,
        }))
        .await
        .expect("a frame");
    assert!(
        eventually(5, || third.is_closed()).await,
        "the link kept the connection on which it was given a second ship"
    );

    // Through all of it the link stayed where it was.
    let replica = link.replica();
    assert_eq!((replica.shard, replica.handoffs), (Some(id_a), 0));
    assert_eq!(replica.ship, Some(ours));
    assert!(!link.is_closed());
    assert_eq!(link.shard_address(), shard_a.addr);

    link.close();
    shard_a.task.abort();
    shard_b.task.abort();
}

/// One shard, no handoff: the link learns the scene on join, follows every
/// change, ignores a stale one, and keeps an id it does not know (drawing the
/// default for it).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_client_learns_the_scene_on_join_and_follows_a_change() {
    let shard = start(1, NodeRef::new(64, [0, 0, 0]));
    assert!(shard.shard.lock().await.set_scene(scene(Scene::Galaxy, 1)));

    let link = ServerLink::connect(shard.addr()).await.expect("join");
    assert!(
        eventually(5, || link.scene() == scene(Scene::Galaxy, 1)).await,
        "not told on join: {}",
        link.scene()
    );
    assert_eq!(link.replica().scene, link.scene());

    // What an app does every frame: compare with the scene it last drew.
    let mut drawn = link.scene();
    let mut changes = Vec::new();
    for next in [scene(Scene::Materials, 2), scene(Scene::DemoArea, 3)] {
        assert!(shard.shard.lock().await.set_scene(next));
        assert!(
            eventually(5, || link.scene() != drawn).await,
            "the change to {next} never arrived"
        );
        drawn = link.scene();
        changes.push(drawn);
        // A late message for an older revision is not relayed, so nothing moves.
        assert!(!shard
            .shard
            .lock()
            .await
            .set_scene(scene(Scene::Materials, 1)));
    }
    assert_eq!(
        changes,
        vec![scene(Scene::Materials, 2), scene(Scene::DemoArea, 3)]
    );

    // A scene a newer server knows and this client does not: kept as sent,
    // drawn as the default.
    let unknown = SceneState::new(SceneId(4_000), 4);
    assert!(shard.shard.lock().await.set_scene(unknown));
    assert!(eventually(5, || link.scene() == unknown).await);
    assert_eq!(link.scene().scene.known(), None);
    assert_eq!(link.scene().scene.or_default(), Scene::DEFAULT);
    assert_eq!(link.replica().handoffs, 0);
    assert!(!link.is_closed());

    link.close();
    shard.stop();
}

/// A join snapshot that needs several frames arrives whole: a shard whose
/// frames carry two or three entities each has nine crates floating around
/// the spawn point, so the snapshot a new player is sent goes out in at
/// least three chunks, a tick apart — and the client holds all ten entities
/// afterwards, and still a second later.
///
/// Mutation this catches (client): taking each `Snapshot` message for the
/// whole interest set — only the last chunk's entities are left, and the
/// shard never sends the others again (they rest, and it counts them as
/// known). Mutation (shard): chunks that do not say that more follow.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_join_snapshot_in_several_chunks_arrives_whole() {
    use space_sim::body::Body;
    use space_sim::shape::ShapeRef;
    use space_sim::{BodySpawn, ShipState};

    let region = NodeRef::new(64, [0, 0, 0]);
    // 600 B a frame: two or three entities (a ship is about 160 B).
    let a = start_with(1, region, |config| config.frame_budget = 600);
    let spawn = space_client::link::map_anchor(&ShardMap::new(
        1,
        vec![ShardRegion::new(region, ShardId::new(1), a.addr())],
    ))
    .expect("an anchor");
    let mut crates = Vec::new();
    {
        let mut shard = a.shard.lock().await;
        for i in 0..9 {
            let at =
                WorldPos::from_f64_metres_around(spawn, [20.0 + 3.0 * f64::from(i), 2.0, 10.0]);
            let body = Body::dynamic(ShapeRef::cuboid([0.3; 3]));
            crates.push(
                shard
                    .spawn_body(BodySpawn::new(ShipState::at_rest(at), body))
                    .expect("a crate"),
            );
        }
    }
    let link = ServerLink::connect(a.addr()).await.expect("joined");
    let whole = eventually(10, || {
        let replica = link.replica();
        replica.ship.is_some() && crates.iter().all(|g| replica.entity(*g).is_some())
    })
    .await;
    let replica = link.replica();
    assert!(
        whole,
        "the client holds {} of the 9 crates",
        crates
            .iter()
            .filter(|g| replica.entity(**g).is_some())
            .count()
    );
    // And keeps them: nothing a later chunk or delta says takes one away.
    tokio::time::sleep(Duration::from_secs(1)).await;
    let replica = link.replica();
    assert_eq!(
        replica.tracked().count(),
        10,
        "nine crates and the own ship"
    );
    assert!(crates.iter().all(|g| replica.entity(*g).is_some()));
    link.close();
    a.stop();
}

/// The ticks of the inputs `conn` receives as datagrams within `within`.
async fn input_ticks(conn: &space_net::Conn, within: Duration) -> Vec<u64> {
    let mut ticks = Vec::new();
    let end = tokio::time::Instant::now() + within;
    while let Ok(Ok(Some(message))) = tokio::time::timeout_at(end, conn.recv_datagram()).await {
        if let Message::InputTick(batch) = message {
            ticks.extend(batch.inputs.iter().map(|input| input.tick));
        }
    }
    ticks
}

/// The link tags its inputs a few ticks ahead and sends each once. When it
/// follows a redirect, the inputs it sent **just before** the second
/// connection existed are for ticks the accepting shard may be the one to
/// simulate (a commit that follows the connection at once): the link says
/// them again on the new connection, through the path every input takes.
///
/// Mutation this catches: opening the connection and only sending what comes
/// after (the accepting shard never hears the last few ticks sent before:
/// the ship coasts through them).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_inputs_sent_just_before_a_handoff_connection_are_said_again_on_it() {
    use space_protocol::{EntityGuid, EntitySnapshot, Redirect, Snapshot};
    let (id_a, id_b) = (ShardId::new(1), ShardId::new(2));
    let mut shard_a = StandIn::start(id_a, 7);
    let mut shard_b = StandIn::start(id_b, 40);
    let map = ShardMap::new(
        1,
        vec![
            ShardRegion::new(NodeRef::new(64, [0, 0, 0]), id_a, shard_a.addr),
            ShardRegion::new(NodeRef::new(64, [1, 0, 0]), id_b, shard_b.addr),
        ],
    );
    *shard_a.map.lock().expect("map") = map.clone();
    *shard_b.map.lock().expect("map") = map.clone();
    let link = ServerLink::join(shard_a.addr).await.expect("join A");
    let (on_a, _) = shard_a
        .arrival(Duration::from_secs(5))
        .await
        .expect("the link's connection");
    let ours = EntityGuid::new(id_a, 1);
    let spawn = space_scene::scene_anchor(&map).expect("an anchor");
    let frame = |tick: u64| {
        Message::Snapshot(Snapshot {
            tick,
            shard: id_a,
            epoch: 1,
            entities: vec![EntitySnapshot::new(
                ours,
                vec![
                    (
                        <space_sim::Owner as space_ecs::ComponentDef>::ID,
                        space_protocol::codec::encode_raw(&7u64),
                    ),
                    (
                        <space_sim::ShipState as space_ecs::ComponentDef>::ID,
                        space_sim::wire::encode_ship_state(&space_sim::ShipState::at_rest(spawn)),
                    ),
                ],
            )],
            input_ack: 0,
            more: false,
        })
    };
    // A's frames, a tick apart, for a third of a second: the link flies.
    for tick in 200..210u64 {
        on_a.send(&frame(tick)).await.expect("a frame");
        tokio::time::sleep(Duration::from_millis(33)).await;
    }
    assert!(eventually(5, || link.replica().ship == Some(ours)).await);
    // What A has been sent so far: inputs up to a few ticks past its frames.
    let sent_to_a = input_ticks(&on_a, Duration::from_millis(60)).await;
    let newest = *sent_to_a.iter().max().expect("the link sends inputs");
    assert!(newest > 209, "tagged ahead of the newest frame: {newest}");
    // The handoff: the link dials B.
    on_a.send(&Message::Redirect(Redirect {
        entity: ours,
        new_shard: id_b,
        shard_map: map.clone(),
        activation_tick: 300,
    }))
    .await
    .expect("a redirect");
    let (on_b, _) = shard_b
        .arrival(Duration::from_secs(5))
        .await
        .expect("the link follows the redirect");
    let said_to_b = input_ticks(&on_b, Duration::from_millis(400)).await;
    // Every tick after A's newest frame that the link had already sent an
    // input for: B hears them too.
    let missing: Vec<u64> = (210..=newest).filter(|t| !said_to_b.contains(t)).collect();
    assert!(
        missing.is_empty(),
        "B was never told the inputs for ticks {missing:?} (sent to A up to {newest}; B heard \
         {said_to_b:?})"
    );
    link.close();
    shard_a.task.abort();
    shard_b.task.abort();
}
