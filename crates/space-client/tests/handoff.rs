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
use space_protocol::{Scene, SceneId, SceneState, ShardId, ShardMap, ShardRegion};
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

/// Bind a shard on loopback and start serving and ticking it at 30 Hz.
fn start(id: u32, region: NodeRef) -> Running {
    let net = Arc::new(ShardNet::bind(SocketAddr::from(([127, 0, 0, 1], 0))).expect("bind"));
    let shard = Arc::new(AsyncMutex::new(Shard::new(ShardConfig::new(
        ShardId::new(id),
        net.local_addr().expect("addr"),
        region,
    ))));
    let serve = tokio::spawn({
        let (net, shard) = (Arc::clone(&net), Arc::clone(&shard));
        async move {
            let _ = net.serve(shard).await;
        }
    });
    let ticker = tokio::spawn({
        let (net, shard) = (Arc::clone(&net), Arc::clone(&shard));
        async move {
            let mut clock =
                tokio::time::interval(Duration::from_secs_f64(f64::from(space_sim::dt())));
            clock.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                clock.tick().await;
                shard.lock().await.tick();
                net.flush(&shard).await;
            }
        }
    });
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

/// Thrust −x until the ship has been handed to shard A, watching that it
/// never jumps on the way. Returns the replica as it arrived.
async fn fly_to_a(link: &ServerLink) -> space_client::link::Replica {
    link.set_thrust([-1.0, 0.0, 0.0]);
    let mut last = (link.offset(), std::time::Instant::now());
    // The worst look: how much further the ship was than 30 m/s (it reaches
    // half of that here) and a metre of slack would explain.
    let mut worst_excess = f64::MIN;
    for _ in 0..800 {
        let now = (link.offset(), std::time::Instant::now());
        let step = (0..3)
            .map(|i| (now.0[i] - last.0[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        let explained = 30.0 * now.1.duration_since(last.1).as_secs_f64() + 1.0;
        worst_excess = worst_excess.max(step - explained);
        last = now;
        if link.replica().handoffs == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    link.set_thrust([0.0; 3]);
    let arrived = link.replica();
    assert_eq!(
        arrived.handoffs, 1,
        "the client never arrived on shard A: {arrived:?}"
    );
    assert_eq!(arrived.shard, Some(ShardId::new(1)), "{arrived:?}");
    // Seamless: across the whole flight — the switch included — the ship
    // never moved further between two looks than its speed explains (a
    // ship that was reset to a spawn point would have jumped by metres, or
    // by half a cell).
    assert!(
        worst_excess <= 0.0,
        "the ship jumped: {worst_excess:.2} m more between two looks than flying explains"
    );
    let at_switch = arrived.offset();
    assert!(
        at_switch[0] < -6.0 && at_switch[0] > -60.0,
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
    // the solar system.
    assert!(b.shard.lock().await.set_scene(scene(Scene::Galaxy, 2)));
    assert!(a.shard.lock().await.set_scene(scene(Scene::Solar, 3)));

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
    let arrived = fly_to_a(&link).await;
    assert_eq!(arrived.ship, Some(ship), "the same ship, by guid");
    assert!(!link.is_closed(), "the new connection is the live one");
    assert_eq!(link.shard_address(), a.addr());
    assert_eq!(link.entry(), a.addr(), "the entry address never changes");
    // The newer scene won (it was heard on the handoff's second connection,
    // before the switch or with it), and the replica and the link agree.
    assert_eq!(arrived.scene, scene(Scene::Solar, 3));
    assert_eq!(link.scene(), scene(Scene::Solar, 3));
    assert_eq!(link.scene().scene.or_default(), Scene::Solar);
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
    assert!(a.shard.lock().await.set_scene(scene(Scene::Solar, 4)));

    let link = join_through_a(&a, &b).await;
    assert!(
        eventually(5, || link.scene() == scene(Scene::Materials, 5)).await,
        "the link never learned the scene of the shard it joined: {}",
        link.scene()
    );
    let arrived = fly_to_a(&link).await;
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
    for next in [scene(Scene::Solar, 2), scene(Scene::DemoArea, 3)] {
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
        vec![scene(Scene::Solar, 2), scene(Scene::DemoArea, 3)]
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
