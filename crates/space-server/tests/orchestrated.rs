//! An orchestrated shard's side of its lifecycle (`docs/plan-sharding.md` §2),
//! against an orchestrator that is scripted by the test: what the shard does
//! when it is told to wait, when it is refused for good, when the
//! orchestrator is not there, and when the orchestrator goes away.
//!
//! The orchestrator here is a few lines over the real transport
//! ([`space_net`]): it runs the same handshake the real one does (transport
//! hello, placeholder welcome, `ShardHello`, then the answer on a reliable
//! stream) and answers from a script. The real orchestrator's half of all this
//! is tested in `space-orchestrator`.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use space_net::Conn;
use space_protocol::{
    Message, Reject, RejectReason, ShardHello, ShardId, ShardMap, ShardRegion, ShardWelcome,
    Welcome,
};
use space_server::net::{peer_hello, peer_hello_with, SERVER_NAME};
use space_server::orchestrated::{
    register_within, start_orchestrated, OrchestratedOptions, REGISTER_BACKOFF_MIN,
};
use space_world::NodeRef;

const CLUSTER: &str = "test-cluster";
const PEER_TOKEN: &str = "the-peer-token-of-this-cluster";

fn loopback0() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], 0))
}

fn root() -> NodeRef {
    NodeRef::new(40, [0, 0, 0])
}

fn options() -> OrchestratedOptions {
    OrchestratedOptions {
        shard: ShardId::NONE,
        bind: loopback0(),
        advertise: None,
        client_endpoint: None,
        claim_region: None,
        cluster: CLUSTER.to_owned(),
        secret: "the-cluster-secret".to_owned(),
    }
}

/// One scripted answer to a `ShardHello`.
#[derive(Debug, Clone)]
enum Answer {
    Refuse(RejectReason),
    /// Welcome the shard as this id, owning the root.
    Welcome(u32),
}

/// What the scripted orchestrator saw and holds.
#[derive(Debug, Default)]
struct Seen {
    /// Every `ShardHello`, in order, with when it arrived.
    hellos: Vec<(Instant, ShardHello)>,
    /// The control connections of the shards it welcomed.
    welcomed: Vec<Conn>,
}

/// An orchestrator that answers the hellos it gets from `script`, in order
/// (the last answer is repeated). Returns its address and what it has seen.
fn scripted_orchestrator(script: Vec<Answer>) -> (SocketAddr, Arc<StdMutex<Seen>>) {
    let (config, _cert) = space_net::crypto::make_server_config();
    let endpoint = space_net::server_endpoint(loopback0(), config).expect("bind the control port");
    let addr = endpoint.local_addr().expect("bound");
    let seen = Arc::new(StdMutex::new(Seen::default()));
    let shared = Arc::clone(&seen);
    tokio::spawn(async move {
        while let Some(incoming) = endpoint.accept().await {
            let Ok((conn, _hello)) = space_net::accept_incoming(incoming).await else {
                continue;
            };
            let placeholder = Message::Welcome(Welcome {
                session: 0,
                shard: ShardId::NONE,
                shard_map: ShardMap::new(1, Vec::new()),
            });
            if conn.send_control(&placeholder).await.is_err() {
                continue;
            }
            let Ok(Some(Message::ShardHello(hello))) = conn.recv_control().await else {
                continue;
            };
            let answer = {
                let mut seen = shared.lock().expect("seen");
                let nth = seen.hellos.len();
                seen.hellos.push((Instant::now(), hello.clone()));
                script[nth.min(script.len() - 1)].clone()
            };
            match answer {
                Answer::Refuse(reason) => {
                    let _ = conn
                        .send_reliable(&Message::Reject(Reject { reason }))
                        .await;
                    // The shard closes once it has read the answer.
                    let _ =
                        tokio::time::timeout(Duration::from_secs(2), conn.connection().closed())
                            .await;
                }
                Answer::Welcome(id) => {
                    let shard = ShardId::new(id);
                    // As the real orchestrator records a shard: an advertised
                    // address without an IP gets the one the control
                    // connection comes from, and players are sent where the
                    // hello asks (else to that same address).
                    let mut endpoint = hello.endpoint;
                    if endpoint.ip().is_unspecified() {
                        endpoint.set_ip(conn.remote_address().ip());
                    }
                    let region = ShardRegion::new(root(), shard, endpoint)
                        .with_client_endpoint(hello.client_endpoint.unwrap_or(endpoint));
                    let welcome = ShardWelcome {
                        shard,
                        region: region.clone(),
                        shard_map: ShardMap::new(1, vec![region]),
                        peer_token: PEER_TOKEN.to_owned(),
                    };
                    let _ = conn.send_reliable(&Message::ShardWelcome(welcome)).await;
                    shared.lock().expect("seen").welcomed.push(conn);
                }
            }
        }
    });
    (addr, seen)
}

/// A shard that is told "not yet" asks again — with a pause that grows — and
/// is up as soon as it is welcomed. Every hello names neither an id nor a
/// region, and carries the cluster secret.
///
/// Mutation this catches: treating `Draining` as a refusal like any other
/// (the shard would exit on its first "not yet", which is what every refused
/// hello used to do), or asking again without a pause.
#[tokio::test]
async fn a_shard_told_to_wait_asks_again_until_it_is_welcomed() {
    let (orch, seen) = scripted_orchestrator(vec![
        Answer::Refuse(RejectReason::Draining),
        Answer::Refuse(RejectReason::Draining),
        Answer::Welcome(7),
    ]);
    let shard = tokio::time::timeout(Duration::from_secs(20), start_orchestrated(options(), orch))
        .await
        .expect("the shard is welcomed on its third hello")
        .expect("being told to wait is not a failure");
    assert_eq!(shard.shard, ShardId::new(7));
    assert_eq!(shard.region, root());
    assert!(shard.is_running());

    {
        let seen = seen.lock().expect("seen");
        assert_eq!(seen.hellos.len(), 3, "asked three times");
        for (_, hello) in &seen.hellos {
            assert_eq!(hello.shard, ShardId::NONE, "no id of its own choosing");
            assert_eq!(hello.region, None, "and no region: it is assigned one");
            assert_eq!(hello.cluster, CLUSTER);
            assert_eq!(hello.secret, "the-cluster-secret");
            assert_eq!(hello.endpoint, shard.endpoint);
        }
        let gap = |n: usize| seen.hellos[n].0.duration_since(seen.hellos[n - 1].0);
        assert!(
            gap(1) >= REGISTER_BACKOFF_MIN,
            "the second hello waited: {:?}",
            gap(1)
        );
        assert!(
            gap(2) >= 2 * REGISTER_BACKOFF_MIN,
            "and the pause grew: {:?}",
            gap(2)
        );
    }
    // The map it runs is the welcome's, in which it owns the root.
    assert_eq!(
        shard
            .state
            .lock()
            .await
            .shard_map()
            .lookup(&root().center()),
        Some(ShardId::new(7))
    );
    shard.abort();
}

/// Every refusal but "not yet" is final: the boot fails at once, with the
/// orchestrator's reason, after exactly one hello.
///
/// Mutation this catches: asking again on a final refusal — a shard of
/// another cluster (or with the wrong secret) would knock on the control port
/// for ever.
#[tokio::test]
async fn a_final_refusal_ends_the_boot_with_the_reason() {
    for (reason, said) in [
        (
            RejectReason::Other("hello names cluster \"a\", this is \"b\"".into()),
            "names cluster",
        ),
        (RejectReason::Unauthorized, "cluster secret"),
        (RejectReason::Full, "--max-shards"),
    ] {
        let (orch, seen) = scripted_orchestrator(vec![Answer::Refuse(reason.clone())]);
        let started = Instant::now();
        let err =
            tokio::time::timeout(Duration::from_secs(10), start_orchestrated(options(), orch))
                .await
                .expect("a final refusal is not waited out")
                .expect_err("refused for good");
        assert!(
            err.contains("rejected") && err.contains(said),
            "{reason:?}: {err}"
        );
        assert!(started.elapsed() < Duration::from_secs(5), "{reason:?}");
        assert_eq!(
            seen.lock().expect("seen").hellos.len(),
            1,
            "{reason:?}: asked exactly once"
        );
    }
}

/// An orchestrator that is not there is tried again for the patience given,
/// then the boot fails and says so. (One attempt is bounded by the transport's
/// handshake timeout, so this test takes a few seconds.)
#[tokio::test]
async fn an_orchestrator_that_never_answers_is_given_up_on() {
    // A bound UDP socket nobody reads: the handshake gets no answer.
    let silent = std::net::UdpSocket::bind(loopback0()).expect("bind a silent port");
    let orch = silent.local_addr().expect("silent addr");
    let err = tokio::time::timeout(
        Duration::from_secs(30),
        register_within(&options(), loopback0(), orch, Duration::from_millis(100)),
    )
    .await
    .expect("the boot gives up")
    .expect_err("nobody answered");
    assert!(err.contains("no orchestrator answered"), "{err}");
}

/// A shard does not run without its orchestrator: when the control connection
/// is lost the shard stops serving, and whoever waits on it gets an error —
/// which is what makes the `space-server` process exit non-zero.
///
/// Mutation this catches: only logging the lost link (the old behaviour): the
/// shard would go on serving clients into a world no orchestrator knows of,
/// and an orchestrator restart would leave it running for ever.
#[tokio::test]
async fn a_shard_stops_when_it_loses_its_orchestrator() {
    let (orch, seen) = scripted_orchestrator(vec![Answer::Welcome(3)]);
    let shard = start_orchestrated(options(), orch).await.expect("welcomed");
    let endpoint = shard.endpoint;
    assert!(shard.is_running());

    // A client is served while the orchestrator is there.
    let client = space_net::client_endpoint_for(endpoint).expect("client endpoint");
    space_net::connect(&client, endpoint, SERVER_NAME, &space_net::hello("player"))
        .await
        .expect("the shard serves");

    // The orchestrator goes away (it closes the control connection, as a
    // stopping orchestrator or one that replaced this shard does).
    let link = seen.lock().expect("seen").welcomed.pop().expect("its link");
    link.close(0, "the orchestrator stopped");

    let stopped = tokio::time::timeout(Duration::from_secs(10), shard.wait())
        .await
        .expect("the shard stops by itself");
    let why = stopped.expect_err("losing the orchestrator is an error");
    assert!(
        why.contains("shard 3 lost its orchestrator") && why.contains("stops"),
        "{why}"
    );
    // And it no longer answers anybody.
    let refused = tokio::time::timeout(
        Duration::from_secs(10),
        space_net::connect_within(
            &client,
            endpoint,
            SERVER_NAME,
            &space_net::hello("late"),
            Duration::from_secs(2),
        ),
    )
    .await
    .expect("bounded");
    assert!(refused.is_err(), "a stopped shard welcomes nobody");
}

/// A shard's two addresses are the orchestrator's word. Its hello carries
/// what it was started with — an advertised address that may lack an IP, and
/// the client endpoint it asks for — and it then runs on the region of its
/// welcome: the internal endpoint as the orchestrator recorded it (IP filled
/// in), and the client endpoint the orchestrator put into the map. The map a
/// player is handed names both.
///
/// Mutation this catches: the shard writing its *advertised* address back
/// over the orchestrator's (the map it hands out would name `0.0.0.0` as its
/// internal endpoint, or lose the client endpoint), or a hello that does not
/// carry `--client-endpoint`.
#[tokio::test]
async fn an_orchestrated_shard_takes_its_two_addresses_from_its_welcome() {
    let (orch, seen) = scripted_orchestrator(vec![Answer::Welcome(5)]);
    // The port it will advertise has to be the one it binds: take a free one.
    let port = {
        let probe = std::net::UdpSocket::bind(loopback0()).expect("a free port");
        probe.local_addr().expect("bound").port()
    };
    let players_at: SocketAddr = SocketAddr::from(([0, 0, 0, 0], port));
    let mut options = options();
    options.bind = SocketAddr::from(([127, 0, 0, 1], port));
    options.advertise = Some(SocketAddr::from(([0, 0, 0, 0], port)));
    options.client_endpoint = Some(players_at);
    let shard = start_orchestrated(options, orch).await.expect("welcomed");

    {
        let seen = seen.lock().expect("seen");
        let hello = &seen.hellos[0].1;
        assert_eq!(hello.endpoint, SocketAddr::from(([0, 0, 0, 0], port)));
        assert_eq!(hello.client_endpoint, Some(players_at));
    }
    let internal = SocketAddr::from(([127, 0, 0, 1], port));
    assert_eq!(shard.endpoint, internal, "the IP the orchestrator saw");
    assert_eq!(shard.client_endpoint, players_at);
    {
        let state = shard.state.lock().await;
        let own = &state.shard_map().regions[0];
        assert_eq!((own.endpoint, own.client_endpoint), (internal, players_at));
    }
    // A player who joins is handed that map (the shard owns the root, and
    // with it the spawn point).
    let client = space_net::client_endpoint_for(internal).expect("client endpoint");
    let (_conn, welcome) =
        space_net::connect(&client, internal, SERVER_NAME, &space_net::hello("player"))
            .await
            .expect("the shard serves");
    assert_eq!(
        welcome.shard_map.client_endpoint_of(ShardId::new(5)),
        Some(players_at)
    );
    assert_eq!(welcome.shard_map.regions[0].endpoint, internal);
    shard.abort();
}

/// The peer token of the welcome is in force from the first connection: a
/// hello claiming to be a peer shard is refused without it and let in with
/// it. (The rule itself and the dialling half are unit-tested in `net.rs`.)
#[tokio::test]
async fn an_orchestrated_shard_requires_the_peer_token_of_its_welcome() {
    let (orch, _seen) = scripted_orchestrator(vec![Answer::Welcome(4)]);
    let shard = start_orchestrated(options(), orch).await.expect("welcomed");
    let client = space_net::client_endpoint_for(shard.endpoint).expect("client endpoint");
    let peer = ShardId::new(9);

    let err = space_net::connect(&client, shard.endpoint, SERVER_NAME, &peer_hello(peer))
        .await
        .expect_err("no token, no entry");
    assert_eq!(
        err,
        space_net::NetError::Rejected(RejectReason::Unauthorized)
    );
    space_net::connect(
        &client,
        shard.endpoint,
        SERVER_NAME,
        &peer_hello_with(peer, PEER_TOKEN),
    )
    .await
    .expect("the cluster's token opens the door");
    assert_eq!(
        shard.state.lock().await.stats().clients,
        0,
        "neither was taken for a player"
    );
    // The running shard holds the token (its front end does) and derives
    // `Debug`: a print of it must not contain the token.
    let printed = format!("{shard:?}");
    assert!(
        !printed.contains(PEER_TOKEN),
        "a running shard prints its peer token: {printed}"
    );
    assert!(printed.contains("<redacted>"), "{printed}");
    shard.abort();
}
