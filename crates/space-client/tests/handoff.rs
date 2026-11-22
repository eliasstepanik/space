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

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use space_client::link::ServerLink;
use space_protocol::{ShardId, ShardMap, ShardRegion};
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_client_follows_its_ship_to_the_neighbouring_shard() {
    // A owns a two-cell-wide node; B owns the far octant carved out of it,
    // which is where A's region centre (the spawn point) lies.
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

    let link = ServerLink::connect(a.addr()).await.expect("join shard A");
    let joined = link.replica();
    assert_eq!(joined.shard, Some(ShardId::new(1)));
    let first_session = joined.session;

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

    link.close();
    a.stop();
    b.stop();
}
