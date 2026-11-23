//! Seamless shard transition, end to end over real QUIC on loopback: two
//! in-process shards (the same boot wiring as the `space-server` binary) and a
//! client link that starts on one and is handed off to the other.
//!
//! The topology is what a split leaves behind: shard A owns a region, and
//! shard B owns one octant carved out of it. A's spawn point — the centre of
//! its region — lies in the carved-out octant, so the player's ship belongs to
//! B from its first tick and A starts a handoff straight away. The test asserts
//! the client's half of the make-before-break exchange:
//!
//! * the link follows the `Redirect` to B and reports in, and the handoff
//!   commits (`handoffs == 1`, now on shard B under B's session);
//! * the ship keeps its guid and its place: nothing jumps at the switch;
//! * the player stays in control: thrust sent after the switch moves the ship
//!   on B, and on B only — A no longer simulates it.
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
use space_world::NodeRef;
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

/// Two neighbouring shards as a split leaves them: A owns a two-cell-wide
/// node; B owns the far octant carved out of it, which is where A's region
/// centre (the spawn point) lies — so a player joining A is handed to B.
async fn neighbours() -> (Running, Running) {
    let region_a = NodeRef::new(63, [0, 0, 0]);
    let region_b = NodeRef::new(64, [1, 1, 1]);
    let a = start(1, region_a);
    let b = start(2, region_b);
    let map = ShardMap::new(
        1,
        vec![
            ShardRegion {
                node: region_a,
                shard: ShardId::new(1),
                endpoint: a.addr(),
            },
            ShardRegion {
                node: region_b,
                shard: ShardId::new(2),
                endpoint: b.addr(),
            },
        ],
    );
    a.shard.lock().await.set_shard_map(map.clone());
    b.shard.lock().await.set_shard_map(map);
    (a, b)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_client_follows_its_ship_to_the_neighbouring_shard() {
    let (a, b) = neighbours().await;
    // The cluster's scene was just changed and the change has reached B but
    // not A yet: A still says the galaxy, B already the solar system.
    assert!(a.shard.lock().await.set_scene(scene(Scene::Galaxy, 2)));
    assert!(b.shard.lock().await.set_scene(scene(Scene::Solar, 3)));

    let link = ServerLink::connect(a.addr()).await.expect("join shard A");
    let joined = link.replica();
    assert_eq!(joined.shard, Some(ShardId::new(1)));
    let first_session = joined.session;
    // The scene arrives behind the welcome. It is A's, or already B's newer
    // one (heard on the handoff's second connection) — never nothing, and
    // never B's followed by A's.
    assert!(
        eventually(5, || link.scene().is_set()).await,
        "the link was never told a scene: {:?}",
        link.replica()
    );
    let first_scene = link.scene();
    assert!(
        first_scene == scene(Scene::Galaxy, 2) || first_scene == scene(Scene::Solar, 3),
        "{first_scene}"
    );

    // The handoff: A redirects, the link connects to B and reports in, the
    // shards commit, and the first frame from B showing our ship is the switch.
    assert!(
        eventually(20, || link.replica().handoffs == 1).await,
        "the client never arrived on shard B: {:?}",
        link.replica()
    );
    let arrived = link.replica();
    assert_eq!(arrived.shard, Some(ShardId::new(2)), "{arrived:?}");
    let ship = arrived.ship.expect("the ship came along");
    assert!(!link.is_closed(), "the new connection is the live one");
    // The newer scene won, and the replica and the link agree on it.
    assert_eq!(arrived.scene, scene(Scene::Solar, 3));
    assert_eq!(link.scene(), scene(Scene::Solar, 3));
    assert_eq!(link.scene().scene.or_default(), Scene::Solar);

    // Seamless: the ship is where it was (it has not been flown yet), not
    // reset to some other spawn point.
    let at_switch = arrived.offset();
    let moved: f64 = at_switch.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        moved < 1.0,
        "the ship jumped {moved} m in the handoff: {at_switch:?}"
    );

    // B simulates it now, under the session B gave us; A has let go.
    {
        let b_shard = b.shard.lock().await;
        assert_eq!(b_shard.session_of(ship), Some(arrived.session));
        assert!(
            b_shard.sim().ship_state(ship).is_some(),
            "B simulates the ship"
        );
    }
    assert!(
        eventually(5, || a
            .shard
            .try_lock()
            .is_ok_and(|a_shard| a_shard.sim().ship_state(ship).is_none()))
        .await,
        "A still simulates the ship after the commit"
    );
    let _ = first_session;

    // Still in control: thrust further into B's region and watch the
    // authoritative ship move there.
    link.set_thrust([1.0, 0.0, 0.0]);
    assert!(
        eventually(10, || link.offset()[0] > at_switch[0] + 2.0).await,
        "thrust after the handoff never moved the ship: {:?}",
        link.replica()
    );
    let after = link.replica();
    assert_eq!(after.handoffs, 1, "no second transition: {after:?}");
    assert_eq!(after.shard, Some(ShardId::new(2)));
    let server_side = b
        .shard
        .lock()
        .await
        .sim()
        .ship_state(ship)
        .expect("still on B");
    assert!(
        server_side.vel[0] > 0.0,
        "B applied the thrust: {server_side:?}"
    );

    // The scene changes again: the shard the client is on now tells it. The
    // old shard catching up afterwards reaches nobody (it has no client), and
    // an older revision is not a change at all.
    assert!(b.shard.lock().await.set_scene(scene(Scene::Materials, 4)));
    assert!(
        eventually(5, || link.scene() == scene(Scene::Materials, 4)).await,
        "the change on the new shard never reached the client: {}",
        link.scene()
    );
    assert!(a.shard.lock().await.set_scene(scene(Scene::Materials, 4)));
    assert!(!b.shard.lock().await.set_scene(scene(Scene::Galaxy, 4)));
    assert!(!b.shard.lock().await.set_scene(scene(Scene::Galaxy, 3)));
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
    assert!(a.shard.lock().await.set_scene(scene(Scene::Materials, 5)));
    assert!(b.shard.lock().await.set_scene(scene(Scene::Solar, 4)));

    let link = ServerLink::connect(a.addr()).await.expect("join shard A");
    assert!(
        eventually(5, || link.scene() == scene(Scene::Materials, 5)).await,
        "the link never learned the scene of the shard it joined: {}",
        link.scene()
    );
    assert!(
        eventually(20, || link.replica().handoffs == 1).await,
        "the client never arrived on shard B: {:?}",
        link.replica()
    );
    assert_eq!(link.replica().shard, Some(ShardId::new(2)));
    assert_eq!(
        link.scene(),
        scene(Scene::Materials, 5),
        "the shard that is behind must not win"
    );

    // B catches up: the same revision, so nothing changes for the client...
    assert!(b.shard.lock().await.set_scene(scene(Scene::Materials, 5)));
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(link.scene(), scene(Scene::Materials, 5));
    // ...and the next change is followed.
    assert!(b.shard.lock().await.set_scene(scene(Scene::Galaxy, 6)));
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
