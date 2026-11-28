//! Booting one **orchestrated** shard: register with the orchestrator, adopt the
//! id, region and map it hands back, and follow every epoch it publishes.
//!
//! This lives in the library (not only in `main.rs`) so a test can start a real
//! orchestrated shard in-process, next to a real orchestrator, over loopback —
//! and so the orchestrator's in-process launcher starts its shards through the
//! very same code.
//!
//! # One path for every shard
//!
//! A shard starts **without a region** and is given one at its hello
//! ([`register`]): the root while nobody owns it, otherwise a node the
//! orchestrator carves for it through an epoch change, with the shard already
//! connected. Until that epoch commits the map the shard runs names somebody
//! else as the node's owner, and the shard owns — and simulates — nothing.
//!
//! * The orchestrator changes the map one step at a time, so a hello may be
//!   answered "not yet" ([`space_protocol::RejectReason::Draining`]). That is
//!   not a failure: the shard asks again, [`REGISTER_BACKOFF_MIN`] apart at
//!   first, [`REGISTER_BACKOFF_MAX`] at most, for as long as it is told so.
//! * Any other refusal is **final** (another cluster, a wrong or missing
//!   cluster secret, the cluster is full, another protocol version): the boot
//!   fails with the reason and the process exits.
//! * An orchestrator that cannot be reached at all is tried again for
//!   [`ORCHESTRATOR_PATIENCE`], then the boot fails.
//! * An orchestrator that does not prove it holds the shard's cluster secret
//!   (its certificate is not issued from the CA the secret makes) is **final**
//!   too, never retried: the shard has not sent anything, and in particular not
//!   its secret. A shard with a secret verifies the orchestrator before the
//!   `ShardHello`, and verifies every peer shard before its hello.
//!
//! # A shard does not run without its orchestrator
//!
//! It registers once. When its control connection is lost — the orchestrator
//! stopped or restarted, closed the link (the shard was replaced or its
//! region merged away), or was not heard from for the transport's idle timeout
//! ([`space_net::IDLE_TIMEOUT`]) — the shard stops serving and ticking, and
//! [`RunningShard::wait`] returns an error, so the process exits non-zero.
//! Nothing re-dials: the orchestrator gives the regions of a shard whose link
//! closed to a replacement under a new id, so the old process has nothing left
//! to be, and an orchestrator that comes back has a new map in which it owns
//! nothing. That is what keeps an orchestrator restart from leaving orphans.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use space_net::{ClusterPin, ClusterTrust, NetError};
use space_protocol::{RejectReason, ShardHello, ShardId, ShardRegion, ShardWelcome};
use space_world::NodeRef;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use crate::net::{OrchestratorLink, ShardNet};
use crate::shard::Shard;
use crate::ShardConfig;

/// The first pause before a hello that was answered "not yet" is said again.
pub const REGISTER_BACKOFF_MIN: Duration = Duration::from_millis(250);

/// The longest pause between two hellos (the pause doubles up to this).
pub const REGISTER_BACKOFF_MAX: Duration = Duration::from_secs(2);

/// How long an orchestrator that does not answer at all is tried again before
/// the boot gives up. (One attempt is itself bounded by
/// [`space_net::HANDSHAKE_TIMEOUT`].)
pub const ORCHESTRATOR_PATIENCE: Duration = Duration::from_secs(10);

/// The environment variable a shard reads its cluster secret from, and the
/// one the orchestrator hands it to a launched shard in (never argv, which
/// shows in the process list).
pub const ENV_CLUSTER_SECRET: &str = "SPACE_CLUSTER_SECRET";

/// The environment variable naming a file that holds the cluster secret (a
/// container secret); used when [`ENV_CLUSTER_SECRET`] is not set.
pub const ENV_CLUSTER_SECRET_FILE: &str = "SPACE_CLUSTER_SECRET_FILE";

/// What an orchestrated shard needs to know before it asks for its identity.
#[derive(Clone)]
pub struct OrchestratedOptions {
    /// The id to claim, or [`ShardId::NONE`] to let the orchestrator allocate
    /// (the normal case: an id is only ever claimed for the reserved root).
    pub shard: ShardId,
    /// Where to listen (port 0 for "any").
    pub bind: SocketAddr,
    /// The address reported in `ShardHello`, or `None` for the bound address.
    /// It is the **internal** endpoint, the one other shards dial; with an
    /// unspecified IP (`0.0.0.0:<port>`) the orchestrator fills in the IP it
    /// sees this shard's control connection come from.
    pub advertise: Option<SocketAddr>,
    /// Where **players** are to dial this shard (`--client-endpoint`), if not
    /// at the advertised address: what the orchestrator puts into the map as
    /// this shard's client endpoint. An unspecified IP (`0.0.0.0:<port>`)
    /// means "the host the player dialed, this port". `None` = the advertised
    /// address.
    pub client_endpoint: Option<SocketAddr>,
    /// The region to claim, or `None` for "assign me one" (the normal case).
    pub claim_region: Option<NodeRef>,
    /// The cluster id forwarded in `ShardHello`.
    pub cluster: String,
    /// The cluster secret presented in `ShardHello` (empty: none). Needed when
    /// the orchestrator's control port is not on loopback, or when it was
    /// given a secret explicitly. With a secret the shard presents a cluster
    /// certificate itself and refuses to send anything (the secret included)
    /// to an orchestrator or peer shard whose certificate was not issued from
    /// the CA this secret makes; without one it behaves as it always did.
    pub secret: String,
}

impl std::fmt::Debug for OrchestratedOptions {
    /// Everything but the secret.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrchestratedOptions")
            .field("shard", &self.shard)
            .field("bind", &self.bind)
            .field("advertise", &self.advertise)
            .field("client_endpoint", &self.client_endpoint)
            .field("claim_region", &self.claim_region)
            .field("cluster", &self.cluster)
            .field("secret", &!self.secret.is_empty())
            .finish()
    }
}

impl OrchestratedOptions {
    /// The address to put in `ShardHello`: `advertise` if given, else the
    /// address the socket actually bound.
    #[must_use]
    pub fn advertised(&self, bound: SocketAddr) -> SocketAddr {
        self.advertise.unwrap_or(bound)
    }
}

/// What became of one hello, seen from the shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HelloFailure {
    /// "Not yet": the orchestrator is in the middle of a map change. Ask
    /// again after a pause; this is never a reason to stop.
    AskAgain,
    /// The orchestrator did not answer at all (it is not up, or not there).
    /// Tried again for [`ORCHESTRATOR_PATIENCE`].
    Unreachable(String),
    /// The orchestrator said no for good, or speaks another protocol: the
    /// shard must stop, with this reason.
    Final(String),
}

/// Sort the error of an [`OrchestratorLink::handshake`] into what the shard
/// does about it.
#[must_use]
pub fn classify(err: &NetError) -> HelloFailure {
    let refused = |why: &str| {
        HelloFailure::Final(format!(
            "the orchestrator rejected this shard's hello: {why}"
        ))
    };
    match err {
        NetError::Rejected(RejectReason::Draining) => HelloFailure::AskAgain,
        NetError::Rejected(RejectReason::Unauthorized) => refused(&format!(
            "the cluster secret is missing or wrong (set {ENV_CLUSTER_SECRET} or \
             {ENV_CLUSTER_SECRET_FILE} to the orchestrator's)"
        )),
        NetError::Rejected(RejectReason::Full) => {
            refused("the cluster already runs as many shards as it may (--max-shards)")
        }
        NetError::Rejected(RejectReason::Other(why)) => refused(why),
        // What a *player* is told while a cluster has no shard for it yet.
        // Nothing a control port says to a shard: the address is a player
        // entry (the orchestrator's `--player-bind`, or a shard).
        NetError::Rejected(RejectReason::Starting) => HelloFailure::Final(
            "that address answers players, not shards: --orchestrator takes the orchestrator's \
             control port (--bind, default 7900), not its player port"
                .to_owned(),
        ),
        NetError::Rejected(RejectReason::VersionMismatch { .. })
        | NetError::VersionMismatch { .. } => HelloFailure::Final(format!(
            "{err}: shard and orchestrator must be built from the same commit"
        )),
        NetError::Protocol(_) => HelloFailure::Final(err.to_string()),
        // Never retried, and the secret was not sent: the orchestrator's
        // certificate does not chain to the CA this shard's secret makes.
        NetError::Untrusted(_) => HelloFailure::Final(format!(
            "the orchestrator at that address did not prove it holds this shard's cluster \
             secret (its certificate was not issued from it): the secret differs from the \
             orchestrator's (check {ENV_CLUSTER_SECRET} / {ENV_CLUSTER_SECRET_FILE} on both), \
             the orchestrator runs without one, or something else answers there"
        )),
        NetError::Handshake(_) | NetError::Io(_) => HelloFailure::Unreachable(err.to_string()),
    }
}

/// Say hello to the orchestrator at `orch_addr` until it welcomes this shard
/// — asking again while it says "not yet" — and return the welcome and the
/// live control connection.
///
/// `advertised` is the address put in the hello. See the module docs for what
/// is tried again and what is final.
///
/// # Errors
///
/// A final refusal, with the orchestrator's reason; or no orchestrator at
/// `orch_addr` for [`ORCHESTRATOR_PATIENCE`].
pub async fn register(
    options: &OrchestratedOptions,
    advertised: SocketAddr,
    orch_addr: SocketAddr,
) -> Result<(ShardWelcome, space_net::Conn), String> {
    register_within(options, advertised, orch_addr, ORCHESTRATOR_PATIENCE).await
}

/// [`register`] with a caller-chosen patience for an orchestrator that does
/// not answer (a test that wants the give-up quickly).
///
/// # Errors
///
/// As [`register`].
pub async fn register_within(
    options: &OrchestratedOptions,
    advertised: SocketAddr,
    orch_addr: SocketAddr,
    patience: Duration,
) -> Result<(ShardWelcome, space_net::Conn), String> {
    let pin = ClusterTrust::from_secret(&options.secret).map(|t| t.pin());
    register_pinned(options, advertised, orch_addr, patience, pin).await
}

/// The loop of [`register_within`], with the orchestrator verified against
/// `pin` (derived from the secret by the callers; `None` only without one).
async fn register_pinned(
    options: &OrchestratedOptions,
    advertised: SocketAddr,
    orch_addr: SocketAddr,
    patience: Duration,
    pin: Option<ClusterPin>,
) -> Result<(ShardWelcome, space_net::Conn), String> {
    // `ShardId::NONE` and no region is the normal hello: the orchestrator
    // allocates the id and assigns the region.
    let claim = options
        .claim_region
        .map(|node| ShardRegion::new(node, ShardId::NONE, advertised));
    let mut backoff = REGISTER_BACKOFF_MIN;
    let mut unreachable_since: Option<Instant> = None;
    let mut asked = 0u32;
    loop {
        let hello = ShardHello {
            cluster: options.cluster.clone(),
            shard: options.shard,
            endpoint: advertised,
            region: claim.clone(),
            client_endpoint: options.client_endpoint,
            secret: options.secret.clone(),
        };
        let err = match OrchestratorLink::handshake_pinned(orch_addr, hello, pin.as_ref()).await {
            Ok(answer) => return Ok(answer),
            Err(err) => err,
        };
        match classify(&err) {
            HelloFailure::Final(why) => return Err(why),
            HelloFailure::AskAgain => {
                // The orchestrator is there and busy: no end to this but its
                // answer. Said once, not on every round.
                unreachable_since = None;
                if asked == 0 {
                    println!(
                        "space-server: the orchestrator at {orch_addr} is in the middle of a map \
                         change and has no region for this shard yet; asking again"
                    );
                }
                asked += 1;
            }
            HelloFailure::Unreachable(why) => {
                let since = *unreachable_since.get_or_insert_with(Instant::now);
                if since.elapsed() >= patience {
                    return Err(format!(
                        "no orchestrator answered at {orch_addr} for {:.0?}: {why}",
                        since.elapsed()
                    ));
                }
                eprintln!(
                    "space-server: the orchestrator at {orch_addr} does not answer ({why}); \
                     trying again"
                );
            }
        }
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(REGISTER_BACKOFF_MAX);
    }
}

/// An orchestrated shard that is up and serving.
#[derive(Debug)]
pub struct RunningShard {
    /// The id the orchestrator confirmed or allocated.
    pub shard: ShardId,
    /// The region the orchestrator gave it (owned once the map says so).
    pub region: NodeRef,
    /// The address peers dial: what the hello advertised, as the orchestrator
    /// recorded it (an unspecified IP filled in with the one it sees the
    /// shard at).
    pub endpoint: SocketAddr,
    /// The address players are sent to: the client endpoint the orchestrator
    /// put into the map for this shard (it may name the unspecified IP, "the
    /// host the player dialed").
    pub client_endpoint: SocketAddr,
    /// The shard's live state (tests read it; the serve and tick tasks own the
    /// write side).
    pub state: Arc<Mutex<Shard>>,
    net: Arc<ShardNet>,
    serve: JoinHandle<()>,
    ticking: JoinHandle<()>,
    /// Stops the shard when the orchestrator is lost.
    watchdog: JoinHandle<()>,
    link: OrchestratorLink,
}

impl RunningShard {
    /// Wait until the shard stops, and say why.
    ///
    /// # Errors
    ///
    /// The shard **lost its orchestrator** (see the module docs: a shard does
    /// not run without one); or the serve task panicked or was cancelled; or
    /// the tick loop ended (see [`crate::net::supervise`]: a shard that cannot
    /// tick must not keep serving).
    pub async fn wait(self) -> Result<(), String> {
        let Self {
            shard,
            serve,
            ticking,
            watchdog,
            link,
            ..
        } = self;
        let stopped = crate::net::supervise(serve, ticking).await;
        watchdog.abort();
        let (retired, lost) = (link.retired_reason(), link.lost_reason());
        link.close();
        match (retired, lost) {
            // Told to stop: the one orderly end (the link the orchestrator
            // closes right behind its word is not a loss).
            (Some(_), _) => Ok(()),
            (None, Some(why)) => Err(lost_line(shard, &why)),
            (None, None) => stopped,
        }
    }

    /// Whether the orchestrator has told this shard to stop
    /// ([`space_protocol::ShardRetire`]).
    #[must_use]
    pub fn is_retired(&self) -> bool {
        self.link.retired_reason().is_some()
    }

    /// Whether the shard is still serving, ticking and linked to its
    /// orchestrator.
    #[must_use]
    pub fn is_running(&self) -> bool {
        !(self.serve.is_finished() || self.ticking.is_finished())
            && self.link.lost_reason().is_none()
    }

    /// Stop serving and ticking and close the endpoint and the control link
    /// (tests and the in-process launcher; a process just exits).
    pub fn abort(self) {
        self.watchdog.abort();
        self.serve.abort();
        self.ticking.abort();
        self.link.close();
        self.net.close();
    }
}

/// What a shard says when it stops because its orchestrator is gone.
fn lost_line(shard: ShardId, why: &str) -> String {
    format!(
        "{shard} lost its orchestrator ({why}); a shard does not run without one, so it stops \
         (the orchestrator replaces it under a new id)"
    )
}

/// What a shard says when it stops because its orchestrator told it to.
fn retired_line(shard: ShardId, why: &str) -> String {
    format!("{shard} stops: the orchestrator retired it ({why})")
}

/// Register with the orchestrator at `orch_addr`, adopt the welcome, and start
/// serving and ticking. Returns once the shard is up.
///
/// # Errors
///
/// The socket cannot be bound, or the registration fails for good (see
/// [`register`]).
pub async fn start_orchestrated(
    options: OrchestratedOptions,
    orch_addr: SocketAddr,
) -> Result<RunningShard, String> {
    let time = Arc::new(crate::clock::SystemTime::new());
    start_orchestrated_on(options, orch_addr, time).await
}

/// [`start_orchestrated`] on the clocks `time` instead of the host's: for a
/// test that decides what time it is ([`crate::clock::ManualTime`]).
///
/// # Errors
///
/// As [`start_orchestrated`]; and the welcome names a tick origin that lies
/// in the future of `time`'s wall clock.
pub async fn start_orchestrated_on(
    options: OrchestratedOptions,
    orch_addr: SocketAddr,
    time: Arc<dyn crate::clock::TimeSource>,
) -> Result<RunningShard, String> {
    // Bind first (whatever `options.bind` says; the binary defaults it to the
    // fixed `127.0.0.1:7777`): without `advertise` the endpoint we
    // report is the one we listen on. With it, we report that address instead,
    // so a peer or client on another host/container can dial us.
    let trust = ClusterTrust::from_secret(&options.secret);
    let net = Arc::new(
        ShardNet::bind_trusted(options.bind, trust.clone()).map_err(|e| format!("bind: {e}"))?,
    );
    let listening = net.local_addr().map_err(|e| e.to_string())?;
    let advertised = options.advertised(listening);

    // Ask the orchestrator for an id + region + map, for as long as it says
    // "not yet". Nothing is served and nothing ticks before the welcome.
    let pin = trust.as_ref().map(ClusterTrust::pin);
    let (welcome, conn) =
        match register_pinned(&options, advertised, orch_addr, ORCHESTRATOR_PATIENCE, pin).await {
            Ok(answer) => answer,
            Err(why) => {
                net.close();
                return Err(format!("orchestrator handshake: {why}"));
            }
        };

    // The cluster's clock: this shard counts the ticks every other shard of
    // the cluster counts, from the origin the welcome names — whenever it
    // was started, a replacement included (`crate::clock`). An origin this
    // host's clock has not reached yet cannot be run on: refuse, before
    // anything is served (the orchestrator sees the link close and says so).
    let clock = match crate::clock::TickClock::on(welcome.tick_origin_us, time) {
        Ok(clock) => clock,
        Err(why) => {
            net.close();
            return Err(format!("the orchestrator's welcome: {why}"));
        }
    };

    // The shard runs the map of the welcome — never one it made up. If the
    // region it was given is not in that map yet (it is being carved for it),
    // the map names its previous owner and this shard owns nothing until the
    // epoch commits: a shard never acts as the owner of a region before that.
    //
    // Its own two addresses are the orchestrator's word too (the welcome's
    // region): the internal one may be the advertised address with the IP
    // filled in that the orchestrator sees this shard at, and the one players
    // dial is what the orchestrator put into the map for it.
    let (endpoint, client_endpoint) = (welcome.region.endpoint, welcome.region.client_endpoint);
    let config = ShardConfig::new(welcome.shard, endpoint, welcome.region.node);
    let mut shard = Shard::new(config);
    shard.set_shard_map(welcome.shard_map.clone());
    shard.set_endpoints(endpoint, client_endpoint);
    // The orchestrator hears once a second what this shard holds and how
    // long its ticks take (`crate::load`): what it scales the cluster on.
    shard.report_load();
    let shard = Arc::new(Mutex::new(shard));

    // The cluster's peer token: presented to every neighbour, required of
    // every neighbour, from the first connection on.
    net.set_peer_token(&welcome.peer_token);

    // Register the map's peer endpoints and route orchestrator traffic through
    // the link's FIFO sender (epoch acks + load reports go out in order).
    for region in &welcome.shard_map.regions {
        if region.shard != welcome.shard {
            net.set_shard_endpoint(region.shard, region.endpoint).await;
        }
    }
    let link = OrchestratorLink::attach(conn, Arc::clone(&shard));
    net.set_orchestrator_sender(link.tx.clone()).await;

    println!(
        "shard {} on {listening} (reached at {endpoint} by shards, {client_endpoint} by players; \
         orchestrated by {orch_addr}), region depth {}",
        welcome.shard.0, welcome.region.node.depth
    );

    let serve = {
        let (net, shard) = (Arc::clone(&net), Arc::clone(&shard));
        tokio::spawn(async move { net.serve(shard).await })
    };
    let state = Arc::clone(&shard);
    let ticking = tokio::spawn(crate::clock::run(shard, Arc::clone(&net), clock));
    // No orchestrator, no shard: stop serving and ticking the moment the
    // link is lost, whoever is (or is not) waiting on `RunningShard::wait`.
    let watchdog = {
        let (lost, retired) = (link.lost_signal(), link.retired_signal());
        let (serve, ticking) = (serve.abort_handle(), ticking.abort_handle());
        let (net, id) = (Arc::clone(&net), welcome.shard);
        tokio::spawn(async move {
            // (Biased: a shard that was told to stop and then sees its link
            // closed was told to stop.)
            tokio::select! {
                biased;
                why = retired.wait() => println!("space-server: {}", retired_line(id, &why)),
                why = lost.wait() => eprintln!("space-server: {}", lost_line(id, &why)),
            }
            ticking.abort();
            serve.abort();
            net.close();
        })
    };

    Ok(RunningShard {
        shard: welcome.shard,
        region: welcome.region.node,
        endpoint,
        client_endpoint,
        state,
        net,
        serve,
        ticking,
        watchdog,
        link,
    })
}

/// Serve as an orchestrated shard until it stops.
///
/// # Errors
///
/// See [`start_orchestrated`] and [`RunningShard::wait`]: in particular the
/// loss of the orchestrator, which ends the shard.
pub async fn run_orchestrated(
    options: OrchestratedOptions,
    orch_addr: SocketAddr,
) -> Result<(), String> {
    start_orchestrated(options, orch_addr).await?.wait().await
}

/// The cluster secret a shard was given: the variable [`ENV_CLUSTER_SECRET`],
/// else the content of the file `file` names (a flag) or
/// [`ENV_CLUSTER_SECRET_FILE`] names, without surrounding whitespace. Empty
/// when none of them is set.
///
/// `env` looks a variable up (the process environment in production, a table
/// in tests).
///
/// # Errors
///
/// A secret file that cannot be read or is empty: a shard that was told where
/// its secret is and cannot get it must not start without one.
pub fn cluster_secret(
    file: Option<&std::path::Path>,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<String, String> {
    let set = |name: &str| env(name).filter(|value| !value.trim().is_empty());
    if let Some(secret) = set(ENV_CLUSTER_SECRET) {
        return Ok(secret.trim().to_owned());
    }
    let from_env = set(ENV_CLUSTER_SECRET_FILE).map(std::path::PathBuf::from);
    let Some(path) = file.map(std::path::Path::to_path_buf).or(from_env) else {
        return Ok(String::new());
    };
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("cluster secret file {}: {e}", path.display()))?;
    let secret = text.trim();
    if secret.is_empty() {
        return Err(format!("cluster secret file {} is empty", path.display()));
    }
    Ok(secret.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What a shard does about each answer to its hello: only "not yet" and
    /// "nobody there" are tried again; everything the orchestrator *said*
    /// otherwise ends the shard, with the reason.
    ///
    /// Mutation this catches: treating every refusal as final (the old
    /// behaviour: a shard told to wait its turn exited), or none (a shard of
    /// another cluster hammering the control port for ever).
    #[test]
    fn only_not_yet_and_nobody_there_are_tried_again() {
        assert_eq!(
            classify(&NetError::Rejected(RejectReason::Draining)),
            HelloFailure::AskAgain
        );
        for unreachable in [
            NetError::Handshake("timed out".into()),
            NetError::Io("connection lost".into()),
        ] {
            assert!(
                matches!(classify(&unreachable), HelloFailure::Unreachable(_)),
                "{unreachable:?}"
            );
        }
        let last_word = |err: NetError| match classify(&err) {
            HelloFailure::Final(why) => why,
            other => panic!("{err:?} must be final, got {other:?}"),
        };
        let why = last_word(NetError::Rejected(RejectReason::Other(
            "hello names cluster \"a\", this is \"b\"".into(),
        )));
        assert!(
            why.contains("rejected") && why.contains("names cluster"),
            "{why}"
        );
        let why = last_word(NetError::Rejected(RejectReason::Unauthorized));
        assert!(why.contains("cluster secret"), "{why}");
        let why = last_word(NetError::Rejected(RejectReason::Full));
        assert!(why.contains("--max-shards"), "{why}");
        let why = last_word(NetError::VersionMismatch {
            seen: 5,
            expected: 6,
        });
        assert!(why.contains("same commit"), "{why}");
        last_word(NetError::Protocol("expected ShardWelcome".into()));
        let why = last_word(NetError::Untrusted("bad certificate".into()));
        assert!(
            why.contains(ENV_CLUSTER_SECRET) && why.contains(ENV_CLUSTER_SECRET_FILE),
            "{why}"
        );
    }

    /// The secret comes from the variable, else from a file (the flag's, else
    /// the variable's); an unreadable or empty file is an error, not "no
    /// secret".
    #[test]
    fn the_cluster_secret_comes_from_the_environment_or_a_file() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |name: &str| {
                pairs
                    .iter()
                    .find(|(n, _)| *n == name)
                    .map(|(_, v)| (*v).to_owned())
            }
        };
        assert_eq!(cluster_secret(None, &env(&[])), Ok(String::new()));
        assert_eq!(
            cluster_secret(None, &env(&[(ENV_CLUSTER_SECRET, " s3cret\n")])),
            Ok("s3cret".to_owned())
        );
        assert_eq!(
            cluster_secret(None, &env(&[(ENV_CLUSTER_SECRET, "  ")])),
            Ok(String::new()),
            "an empty variable is unset"
        );

        let path = std::env::temp_dir().join(format!(
            "space-server-secret-{}-{}.txt",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&path, "from-the-file\n").expect("write the secret file");
        assert_eq!(
            cluster_secret(Some(&path), &env(&[])),
            Ok("from-the-file".to_owned())
        );
        // The variable wins over a file.
        assert_eq!(
            cluster_secret(Some(&path), &env(&[(ENV_CLUSTER_SECRET, "from-the-env")])),
            Ok("from-the-env".to_owned())
        );
        std::fs::write(&path, "\n").expect("empty the secret file");
        let err = cluster_secret(Some(&path), &env(&[])).expect_err("an empty file");
        assert!(err.contains("is empty"), "{err}");
        std::fs::remove_file(&path).ok();
        assert!(cluster_secret(Some(&path), &env(&[])).is_err());
    }

    #[test]
    fn the_options_never_print_the_secret() {
        let options = OrchestratedOptions {
            shard: ShardId::NONE,
            bind: SocketAddr::from(([127, 0, 0, 1], 0)),
            advertise: None,
            client_endpoint: None,
            claim_region: None,
            cluster: "c".into(),
            secret: "do-not-print-me".into(),
        };
        let printed = format!("{options:?}");
        assert!(!printed.contains("do-not-print-me"), "{printed}");
    }
}
