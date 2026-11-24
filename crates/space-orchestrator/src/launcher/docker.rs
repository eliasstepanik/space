//! The Docker launcher: one container per shard, through the Docker Engine API.
//!
//! [`DockerLauncher`] talks to the Docker daemon over its **unix socket** with
//! hand-rolled HTTP/1.1 (no HTTP crate, no `bollard`: the workspace has
//! neither; `webui/http.rs` is the server-side sibling of the client in here).
//! For every [`SpawnSpec`] it creates and starts one container of the shard
//! image:
//!
//! ```text
//!   space-server --orchestrator <control address> --cluster <id> \
//!                --bind 0.0.0.0:<P> --advertise 0.0.0.0:<P> --client-endpoint 0.0.0.0:<P>
//! ```
//!
//! named `<prefix><P>` (`space-shard-7870`), where `P` is the spec's host
//! port. No `--shard`, no region: the shard registers without either, like
//! every other shard.
//!
//! * **Network.** The container joins one Docker network — the configured one
//!   (`--docker-network`), else the one the orchestrator's own container is
//!   on, found by inspecting itself. Its address there and `P` are the
//!   [`ShardHandle::endpoint`]: what its hello is seen to come from, and what
//!   the other shards dial.
//! * **Itself.** The orchestrator's own container is asked for by **name**
//!   (`--docker-self`; the compose files give it) and, failing that, by
//!   `$HOSTNAME`. The Engine looks a container up by id, by name or by the
//!   start of its id — never by host name: `$HOSTNAME` works only as long as
//!   it is the short id Docker gives a container by default, which it no
//!   longer is in a container that was recreated from the old one's
//!   configuration (Watchtower). Whoever is not found is skipped; with a
//!   network **and** a control address configured nobody is asked at all.
//! * **Players.** UDP port `P` of the container is published as port `P` of
//!   the host (on `--docker-publish-ip`, default all), and the shard sends its
//!   players to `0.0.0.0:P` — "the host you dialed, this port".
//! * **Control address.** What the shards dial: `--docker-control-addr`, else
//!   the orchestrator's own address on that network and the control port's
//!   port. (The default is an address because it needs no name service; a
//!   configured `name:port` works too: `space-server --orchestrator` looks a
//!   name up once, when it starts.)
//! * **Secret.** The cluster secret is handed over in the container's
//!   environment (`SPACE_CLUSTER_SECRET`), never in its command, and is taken
//!   out of every error text this module returns.
//! * **Labels.** `space.cluster=<cluster>`, `space.role=shard`,
//!   `space.instance=<instance>`, `space.launch=<n>`, `space.port=<P>`. The
//!   first three are how leftovers are told from strangers **and from the
//!   shards of another stack on the same host**: [`DockerLauncher::prepare`]
//!   removes every container that carries all three — shards of *this*
//!   orchestrator's earlier run that was killed — and a name that is taken is
//!   only ever freed when its container carries them. **A container without
//!   them is never removed.** The cluster's name alone is not enough: two
//!   stacks that were both left at the default name would delete each
//!   other's running shards at every start. The *instance* is what tells
//!   them apart: an id that is the same at every start of one orchestrator
//!   and different between two ([`DockerOptions::instance`]; it is not a
//!   secret and is derived from none). `space.launch` and `space.port` are
//!   for the operator (`docker ps --filter label=space.port=7870`); the
//!   launcher writes them and never reads them.
//! * **Image.** [`DockerLauncher::prepare`] makes sure the image is on the
//!   host; so does a launch whose creation the daemon refuses because the
//!   image is gone (pruned while the orchestrator ran): it is pulled again,
//!   under [`PULL_TIMEOUT`], and the creation tried once more. The launcher
//!   never removes an image.
//! * **Liveness.** [`Launcher::poll`] inspects the container; one that has
//!   stopped is `Dead` and its exit code is kept for [`Launcher::exit_code`].
//!   A daemon that does not answer is an error, not a dead shard.
//! * **End.** [`Launcher::kill`] and [`Launcher::shutdown`] force-remove
//!   (`DELETE …?force=true&v=true`); nothing restarts by itself
//!   (`RestartPolicy: no`), so what the orchestrator did not start does not
//!   run.
//!
//! # What a shard container is given
//!
//! Nothing: **no mount, never the Docker socket, not privileged**, every
//! capability dropped, `no-new-privileges`, a **read-only root filesystem**
//! (a shard writes no file: its log is its standard output), at most
//! [`PIDS_LIMIT`] processes and threads, at most [`DockerOptions::memory_mib`]
//! of memory (and no swap beyond it), and the user [`SHARD_USER`] whatever the
//! image says — the uid the shard image itself runs as, given by number so
//! that an image built without a `USER` line still does not run a shard as
//! root. A shard faces players.
//!
//! # Who can make this module do what
//!
//! **Whoever holds the Docker socket is root on the host.** The orchestrator
//! holds it when it runs this launcher, so what goes into an API call matters:
//! every parameter of every call comes from [`DockerOptions`] and
//! [`SpawnSpec`] — the operator's configuration — and from the daemon's own
//! answers (container ids). **Nothing a shard or a player sends ever reaches
//! this module.**
//!
//! # The HTTP client
//!
//! One connection per request (`Connection: close`). The answer is read under
//! hard limits — head at most [`MAX_HEAD`], a JSON body at most [`MAX_BODY`],
//! the progress stream of a pull read line by line and thrown away — with a
//! body framed by `Content-Length`, by `Transfer-Encoding: chunked` (what the
//! daemon uses for lists, inspections and streams) or by the end of the
//! connection. **Every call has a time limit** ([`CALL_TIMEOUT`],
//! [`REMOVE_TIMEOUT`], [`PULL_TIMEOUT`]). No registry credentials are handled:
//! an `X-Registry-Auth` header is never sent, so a private image has to be on
//! the host before the orchestrator starts.
//!
//! # Platform
//!
//! The daemon's unix socket exists on Linux (the orchestrator's image); on any
//! other platform [`SocketTransport`] refuses with [`UNSUPPORTED_PLATFORM`] and
//! the launcher is [`LaunchError::Unavailable`]. The client itself is generic
//! over a [`Transport`], which is how its tests run everywhere: against a
//! daemon of their own, in memory.

use std::collections::HashMap;
use std::future::Future;
use std::io;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use serde_json::{json, Value};
use space_server::orchestrated::ENV_CLUSTER_SECRET;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use super::{LaunchError, LaunchId, Launcher, Poll, ShardHandle, SpawnSpec};

/// Where the Docker daemon listens unless told otherwise.
pub const DEFAULT_SOCKET: &str = "/var/run/docker.sock";

/// What [`SocketTransport`] says where there is no unix socket to dial.
pub const UNSUPPORTED_PLATFORM: &str = "the Docker launcher needs the Docker Engine's unix socket \
     and runs on Linux only (the orchestrator image is Linux); on this platform use --launcher \
     process";

/// Largest accepted status line + headers of an answer, in bytes.
pub const MAX_HEAD: usize = 16 * 1024;
/// Largest accepted JSON body of an answer, in bytes.
pub const MAX_BODY: usize = 4 * 1024 * 1024;
/// Longest accepted chunk-size line (with its extensions) or trailer line.
const MAX_LINE: usize = 1024;
/// Longest line of a pull's progress stream that is looked at; a longer one
/// is skipped.
const MAX_PROGRESS_LINE: usize = 64 * 1024;
/// How much of the daemon's own words an error text carries, in characters.
const MAX_ERROR_TEXT: usize = 300;
/// How much of the daemon's own words is kept until a secret has been taken
/// out of them: what is shown, and the longest secret there is on top, so
/// that one that starts inside what is shown is there whole.
const MAX_RAW_TEXT: usize = MAX_ERROR_TEXT + space_net::MAX_SECRET_LEN;

/// The time limit of an ordinary call (ping, create, start, inspect, list).
pub const CALL_TIMEOUT: Duration = Duration::from_secs(10);
/// The time limit of a removal (the daemon stops the container first).
pub const REMOVE_TIMEOUT: Duration = Duration::from_secs(15);
/// The time limit of an image pull.
pub const PULL_TIMEOUT: Duration = Duration::from_secs(300);
/// The one time limit around everything [`Launcher::shutdown`] does.
pub const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(20);

/// The label naming the cluster a container is a shard of.
pub const LABEL_CLUSTER: &str = "space.cluster";
/// The label naming what a container is; a shard's is [`ROLE_SHARD`].
pub const LABEL_ROLE: &str = "space.role";
/// The value of [`LABEL_ROLE`] on every container this launcher creates.
pub const ROLE_SHARD: &str = "shard";
/// The label naming the orchestrator a container is a shard of
/// ([`DockerOptions::instance`]): what tells two stacks on one host apart.
pub const LABEL_INSTANCE: &str = "space.instance";
/// The label carrying the launch's number. For the operator: never read here.
pub const LABEL_LAUNCH: &str = "space.launch";
/// The label carrying the shard's port. For the operator: never read here.
pub const LABEL_PORT: &str = "space.port";

/// The user a shard container runs as, `uid:gid`: the one
/// `deploy/Dockerfile.shard` creates, by number (no entry in the image's
/// user table is needed).
pub const SHARD_USER: &str = "10001:10001";
/// The most processes and threads a shard container may have. A shard is one
/// process with a thread per core and a few more; this is far above that and
/// far below what would trouble the host.
pub const PIDS_LIMIT: i64 = 1_024;
/// The memory a shard container may use unless told otherwise
/// (`--shard-memory`), in MiB.
pub const DEFAULT_MEMORY_MIB: u64 = 2_048;

/// How to reach the Docker daemon: one fresh connection per request.
pub trait Transport: Send + Sync {
    /// The connection.
    type Stream: AsyncRead + AsyncWrite + Unpin + Send;

    /// Open a connection to the daemon.
    ///
    /// # Errors
    ///
    /// The daemon cannot be reached; [`io::ErrorKind::Unsupported`] when this
    /// transport cannot exist on this platform at all.
    fn connect(&self) -> impl Future<Output = io::Result<Self::Stream>> + Send;
}

/// The daemon's unix socket (Linux; see [`UNSUPPORTED_PLATFORM`]).
#[derive(Debug, Clone)]
pub struct SocketTransport {
    socket: PathBuf,
}

impl SocketTransport {
    /// The daemon listening on `socket`.
    #[must_use]
    pub fn new(socket: impl Into<PathBuf>) -> Self {
        Self {
            socket: socket.into(),
        }
    }
}

#[cfg(unix)]
impl Transport for SocketTransport {
    type Stream = tokio::net::UnixStream;

    async fn connect(&self) -> io::Result<Self::Stream> {
        tokio::net::UnixStream::connect(&self.socket)
            .await
            .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", self.socket.display())))
    }
}

#[cfg(not(unix))]
impl Transport for SocketTransport {
    // Never made: `connect` refuses.
    type Stream = tokio::io::DuplexStream;

    async fn connect(&self) -> io::Result<Self::Stream> {
        let _ = &self.socket;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            UNSUPPORTED_PLATFORM,
        ))
    }
}

/// What the launcher is configured with. It holds no secret.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DockerOptions {
    /// The daemon's socket (`--docker-socket`).
    pub socket: PathBuf,
    /// The shard image (`--shard-image`), e.g.
    /// `ghcr.io/eliasstepanik/space-shard:sha-abc1234`.
    pub image: String,
    /// The network the shards join (`--docker-network`). `None`: the one the
    /// orchestrator's own container is on.
    pub network: Option<String>,
    /// The `ip:port` the shards dial (`--docker-control-addr`). `None`: the
    /// orchestrator's own address on the network, and `control_port`.
    pub control: Option<String>,
    /// The control port's port, for the default of `control`.
    pub control_port: u16,
    /// The host IP a shard's UDP port is published on
    /// (`--docker-publish-ip`); empty or `0.0.0.0` = all.
    pub publish_ip: String,
    /// What a container's name starts with (`--docker-name-prefix`); the
    /// shard's port follows.
    pub name_prefix: String,
    /// The cluster: the value of [`LABEL_CLUSTER`].
    pub cluster: String,
    /// This orchestrator, as opposed to any other on the host: the value of
    /// [`LABEL_INSTANCE`]. The same at every start of one orchestrator (the
    /// binary keeps it next to its state file, or is told it:
    /// `--docker-instance`), different between two stacks. Not a secret.
    pub instance: String,
    /// The orchestrator's own container by **name** or id
    /// (`--docker-self`). `None`: not told.
    pub self_container: Option<String>,
    /// `$HOSTNAME`: the orchestrator's own container by the start of its id,
    /// as long as nobody gave the container another host name. Asked when
    /// `self_container` is not given or not known to the daemon.
    pub self_hostname: Option<String>,
    /// The memory limit of a shard container in MiB (`--shard-memory`);
    /// 0 = none.
    pub memory_mib: u64,
}

impl Default for DockerOptions {
    fn default() -> Self {
        Self {
            socket: PathBuf::from(DEFAULT_SOCKET),
            image: "space/shard:dev".into(),
            network: None,
            control: None,
            control_port: 7_900,
            publish_ip: "0.0.0.0".into(),
            name_prefix: "space-shard-".into(),
            cluster: "space-local".into(),
            instance: "default".into(),
            self_container: None,
            self_hostname: None,
            memory_mib: DEFAULT_MEMORY_MIB,
        }
    }
}

/// The time limits of the calls; the constants, except in tests.
#[derive(Debug, Clone, Copy)]
struct Timeouts {
    call: Duration,
    remove: Duration,
    pull: Duration,
    shutdown: Duration,
}

impl Default for Timeouts {
    fn default() -> Self {
        Self {
            call: CALL_TIMEOUT,
            remove: REMOVE_TIMEOUT,
            pull: PULL_TIMEOUT,
            shutdown: SHUTDOWN_TIMEOUT,
        }
    }
}

/// Where the shards go: found once ([`DockerLauncher::prepare`], or the first
/// launch) and kept.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Resolved {
    network: String,
    control: String,
}

/// One container of this launcher.
#[derive(Debug, Clone)]
struct Entry {
    /// The container's id; empty while its creation is in flight.
    id: String,
    name: String,
    port: u16,
    /// It was started: only then does "not running" mean "exited".
    started: bool,
    /// It was seen not to run (or to be gone).
    exited: bool,
    /// [`Launcher::kill`] was asked to remove it and could not: it is kept
    /// for [`Launcher::shutdown`] to try again, and is dead to everybody else.
    killed: bool,
    exit_code: Option<i32>,
}

/// Why a call to the daemon came to nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Failure {
    /// The transport cannot exist here ([`UNSUPPORTED_PLATFORM`]).
    Unsupported(String),
    /// No answer: not reachable, timed out, or not HTTP.
    Transport(String),
    /// The daemon answered, and not with a 2xx. `message` is what it said,
    /// **not clipped yet** (at most [`MAX_RAW_TEXT`] characters): a secret is
    /// taken out of the whole of it ([`Failure::without`]) before it is cut
    /// to [`MAX_ERROR_TEXT`] for showing — cut first, the front half of a
    /// secret that lies across the cut would be left standing.
    Status {
        what: String,
        status: u16,
        message: String,
    },
}

impl core::fmt::Display for Failure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Failure::Unsupported(why) | Failure::Transport(why) => f.write_str(why),
            Failure::Status {
                what,
                status,
                message,
            } => write!(
                f,
                "docker: {what}: the daemon answered {status}: {}",
                clip(message)
            ),
        }
    }
}

impl Failure {
    /// This failure with every occurrence of `secret` taken out of its text —
    /// before the text is clipped.
    fn without(self, secret: &str) -> Failure {
        if secret.is_empty() {
            return self;
        }
        let clean = |text: String| text.replace(secret, "<redacted>");
        match self {
            Failure::Unsupported(why) => Failure::Unsupported(clean(why)),
            Failure::Transport(why) => Failure::Transport(clean(why)),
            Failure::Status {
                what,
                status,
                message,
            } => Failure::Status {
                what,
                status,
                message: clean(message),
            },
        }
    }

    fn launch(self) -> LaunchError {
        match self {
            Failure::Unsupported(why) => LaunchError::Unavailable(why),
            other => LaunchError::Spawn(other.to_string()),
        }
    }

    fn is_status(&self, code: u16) -> bool {
        matches!(self, Failure::Status { status, .. } if *status == code)
    }
}

/// Starts one container per shard (see the module docs).
pub struct DockerLauncher<T: Transport = SocketTransport> {
    options: DockerOptions,
    transport: T,
    timeouts: Timeouts,
    /// Never held across an await.
    resolved: Mutex<Option<Resolved>>,
    /// What this launcher has created. Never held across an await.
    containers: Mutex<HashMap<LaunchId, Entry>>,
}

impl<T: Transport> core::fmt::Debug for DockerLauncher<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DockerLauncher")
            .field("options", &self.options)
            .field("containers", &self.table().len())
            .finish_non_exhaustive()
    }
}

impl DockerLauncher {
    /// A launcher that talks to the daemon on `options.socket`.
    #[must_use]
    pub fn new(options: DockerOptions) -> Self {
        let transport = SocketTransport::new(options.socket.clone());
        Self::with_transport(options, transport)
    }

    /// Whether this launcher can exist on this platform at all — the cheap,
    /// synchronous half of [`DockerLauncher::prepare`], which asks the daemon.
    ///
    /// # Errors
    ///
    /// [`LaunchError::Unavailable`] with [`UNSUPPORTED_PLATFORM`] where there
    /// is no unix socket.
    pub fn ensure_available(&self) -> Result<(), LaunchError> {
        if cfg!(unix) {
            Ok(())
        } else {
            Err(LaunchError::Unavailable(UNSUPPORTED_PLATFORM.to_owned()))
        }
    }
}

impl<T: Transport> DockerLauncher<T> {
    /// A launcher that reaches its daemon through `transport`.
    #[must_use]
    pub fn with_transport(options: DockerOptions, transport: T) -> Self {
        Self {
            options,
            transport,
            timeouts: Timeouts::default(),
            resolved: Mutex::new(None),
            containers: Mutex::new(HashMap::new()),
        }
    }

    /// Make the launcher ready, once, before the run loop: ask the daemon
    /// whether it is there (`GET /_ping`), find the network and the control
    /// address, **remove the shard containers that an earlier run of this
    /// orchestrator left behind** (those labelled [`LABEL_CLUSTER`] = this
    /// cluster, [`LABEL_INSTANCE`] = this instance and [`LABEL_ROLE`] =
    /// [`ROLE_SHARD`], and no other — not another stack's, whatever its
    /// cluster is called), and make sure the image is on the host — pulling
    /// it when it is not, without credentials.
    ///
    /// Returns a sentence saying what it found and did.
    ///
    /// A launcher that was not prepared still works: the first launch finds
    /// the network and the control address itself. It does not clean up and
    /// does not pull.
    ///
    /// # Errors
    ///
    /// [`LaunchError::Unavailable`] when the daemon cannot be reached, the
    /// network or control address cannot be decided, or the image is neither
    /// there nor pullable; [`LaunchError::Spawn`] when a call failed.
    pub async fn prepare(&self) -> Result<String, LaunchError> {
        let socket = self.options.socket.display().to_string();
        let mut pong = Sink::collect(MAX_BODY);
        self.call(
            "asking the daemon whether it is there",
            "GET",
            "/_ping",
            None,
            self.timeouts.call,
            &mut pong,
        )
        .await
        .map_err(|failure| match failure {
            Failure::Unsupported(why) => LaunchError::Unavailable(why),
            other => LaunchError::Unavailable(format!(
                "the Docker daemon at {socket} does not answer ({other}); the docker launcher \
                 needs its socket mounted into the orchestrator's container and readable by it"
            )),
        })?;

        let resolved = self.resolved().await?;
        let removed = self.remove_leftovers().await?;
        let pulled = self.ensure_image().await?;

        let image = &self.options.image;
        Ok(format!(
            "docker launcher: the daemon at {socket} answers; shards join the network {:?} and \
             dial {}; the image {image} {}; {removed} leftover shard container{} of cluster {:?} \
             (instance {:?}) removed",
            resolved.network,
            resolved.control,
            if pulled {
                "was pulled"
            } else {
                "is on the host"
            },
            if removed == 1 { "" } else { "s" },
            self.options.cluster,
            self.options.instance,
        ))
    }

    /// The table of containers. A poisoned lock is taken all the same:
    /// nothing done under it leaves the table half-changed, and what it lists
    /// still has to be removed.
    fn table(&self) -> MutexGuard<'_, HashMap<LaunchId, Entry>> {
        self.containers
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// The network and the control address, found on first use.
    async fn resolved(&self) -> Result<Resolved, LaunchError> {
        let known = self
            .resolved
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        if let Some(known) = known {
            return Ok(known);
        }
        let found = self.resolve().await?;
        *self.resolved.lock().unwrap_or_else(PoisonError::into_inner) = Some(found.clone());
        Ok(found)
    }

    async fn resolve(&self) -> Result<Resolved, LaunchError> {
        let unavailable = |why: String| Err(LaunchError::Unavailable(why));
        let configured = |option: &Option<String>| option.clone().filter(|text| !text.is_empty());
        let network = configured(&self.options.network);
        let control = configured(&self.options.control);
        let resolved = if let (Some(network), Some(control)) = (network.clone(), control.clone()) {
            Resolved { network, control }
        } else {
            // The orchestrator's own container: by the name it was told,
            // then by `$HOSTNAME`. The Engine finds a container by id, name
            // or the start of its id - a host name is found only while it is
            // the default one (the short id), so a name that is not known is
            // no verdict: the next one is asked.
            let mut asked: Vec<String> = Vec::new();
            for candidate in [&self.options.self_container, &self.options.self_hostname] {
                if let Some(candidate) = configured(candidate) {
                    if !asked.contains(&candidate) {
                        asked.push(candidate);
                    }
                }
            }
            let mut found = None;
            for own in &asked {
                let target = format!("/containers/{}/json", segment(own));
                let what = format!("inspecting the orchestrator's own container {own}");
                match self.json(&what, "GET", &target, None).await {
                    Ok(me) => {
                        found = Some((own.clone(), me));
                        break;
                    }
                    Err(failure) if failure.is_status(404) => {}
                    Err(failure) => return Err(failure.launch()),
                }
            }
            let Some((own, me)) = found else {
                let missing = match (&network, &control) {
                    (None, None) => "--docker-network and --docker-control-addr",
                    (Some(_), None) => "--docker-control-addr",
                    _ => "--docker-network",
                };
                return unavailable(if asked.is_empty() {
                    format!(
                        "the orchestrator does not seem to run in a container (it was given no \
                         container name of its own, --docker-self, and has no HOSTNAME), so the \
                         shards' network and the address they dial have to be given: pass \
                         {missing}"
                    )
                } else {
                    format!(
                        "the daemon knows no container {} (the orchestrator looks for its own \
                         container by --docker-self, then by HOSTNAME, and a container that was \
                         recreated keeps the old one's host name): set \
                         SPACE_ORCHESTRATOR_DOCKER_SELF (--docker-self) to this container's \
                         name, or pass {missing}",
                        asked
                            .iter()
                            .map(|name| format!("{name:?}"))
                            .collect::<Vec<_>>()
                            .join(" or ")
                    )
                });
            };
            let networks = &me["NetworkSettings"]["Networks"];
            let network = match network {
                Some(network) => network,
                None => {
                    let names: Vec<&String> = networks
                        .as_object()
                        .map(|map| map.keys().collect())
                        .unwrap_or_default();
                    match names.as_slice() {
                        [only] => (*only).clone(),
                        [] => {
                            return unavailable(format!(
                                "the orchestrator's container {own} is on no network; pass \
                                 --docker-network"
                            ));
                        }
                        several => {
                            let list: Vec<&str> = several.iter().map(|n| n.as_str()).collect();
                            return unavailable(format!(
                                "the orchestrator's container {own} is on several networks ({}); \
                                 say which one the shards join: pass --docker-network",
                                list.join(", ")
                            ));
                        }
                    }
                }
            };
            let control = match control {
                Some(control) => control,
                None => {
                    let ip = networks[&network]["IPAddress"].as_str().unwrap_or("");
                    match ip.parse::<IpAddr>() {
                        Ok(ip) => SocketAddr::new(ip, self.options.control_port).to_string(),
                        Err(_) => {
                            return unavailable(format!(
                                "the orchestrator's container {own} has no address on the \
                                 network {network:?}, so there is nothing for the shards to \
                                 dial; pass --docker-control-addr"
                            ));
                        }
                    }
                }
            };
            Resolved { network, control }
        };
        if matches!(resolved.network.as_str(), "host" | "none")
            || resolved.network.starts_with("container:")
        {
            return unavailable(format!(
                "the docker launcher needs a bridge network for its shards (each gets an address \
                 of its own), not {:?}; pass --docker-network",
                resolved.network
            ));
        }
        Ok(resolved)
    }

    /// Remove every container labelled as a shard of this orchestrator (its
    /// cluster **and** its instance); how many.
    async fn remove_leftovers(&self) -> Result<usize, LaunchError> {
        let cluster = &self.options.cluster;
        let instance = &self.options.instance;
        let filters = json!({
            "label": [
                format!("{LABEL_CLUSTER}={cluster}"),
                format!("{LABEL_ROLE}={ROLE_SHARD}"),
                format!("{LABEL_INSTANCE}={instance}"),
            ]
        })
        .to_string();
        let target = format!(
            "/containers/json?all=true&filters={}",
            query_value(&filters)
        );
        let listed = self
            .json("listing leftover shard containers", "GET", &target, None)
            .await
            .map_err(Failure::launch)?;
        let mut removed = 0;
        for container in listed.as_array().map_or(&[][..], Vec::as_slice) {
            // The daemon was asked for ours only. What is removed is still
            // decided here, by the labels it says the container carries.
            if !is_shard_of(&container["Labels"], cluster, instance) {
                continue;
            }
            let Some(id) = container["Id"].as_str().filter(|id| !id.is_empty()) else {
                continue;
            };
            self.remove(id).await.map_err(Failure::launch)?;
            removed += 1;
        }
        Ok(removed)
    }

    /// Make sure the image is on the host; whether it had to be pulled.
    async fn ensure_image(&self) -> Result<bool, LaunchError> {
        let image = &self.options.image;
        let inspect = format!("/images/{}/json", image_path(image));
        let what = format!("looking for the image {image}");
        match self.json(&what, "GET", &inspect, None).await {
            Ok(_) => return Ok(false),
            Err(failure) if failure.is_status(404) => {}
            Err(failure) => return Err(failure.launch()),
        }
        let operator = format!(
            "No registry credentials are sent: a private image has to be pulled on the host by \
             the operator (docker login, then docker pull {image}) before the orchestrator starts"
        );
        let (repository, tag) = split_image(image);
        let target = format!(
            "/images/create?fromImage={}&tag={}",
            query_value(repository),
            query_value(tag)
        );
        let mut progress = Sink::Lines(Progress::default());
        let pulled = self
            .call(
                &format!("pulling the image {image}"),
                "POST",
                &target,
                None,
                self.timeouts.pull,
                &mut progress,
            )
            .await;
        let failed = |why: String| {
            LaunchError::Unavailable(format!(
                "the image {image} is not on this host and could not be pulled: {why}. {operator}"
            ))
        };
        match pulled {
            Err(Failure::Unsupported(why)) => return Err(LaunchError::Unavailable(why)),
            Err(failure) => return Err(failed(failure.to_string())),
            Ok(_) => {}
        }
        if let Sink::Lines(Progress {
            error: Some(error), ..
        }) = &progress
        {
            return Err(failed(clip(error)));
        }
        // The stream ended without an error; the image is what counts.
        match self.json(&what, "GET", &inspect, None).await {
            Ok(_) => Ok(true),
            Err(failure) if failure.is_status(404) => Err(failed(
                "the pull ended without an error and the image is still not there".into(),
            )),
            Err(failure) => Err(failure.launch()),
        }
    }

    /// One call to the daemon: connect, send, read the answer into `sink`,
    /// all within `limit`. `Ok` is a 2xx status.
    async fn call(
        &self,
        what: &str,
        method: &str,
        target: &str,
        body: Option<&Value>,
        limit: Duration,
        sink: &mut Sink,
    ) -> Result<u16, Failure> {
        let body = body.map(Value::to_string);
        let request = request_bytes(method, target, body.as_deref());
        let exchanged = tokio::time::timeout(limit, async {
            let stream = self.transport.connect().await.map_err(|e| {
                if e.kind() == io::ErrorKind::Unsupported {
                    Failure::Unsupported(e.to_string())
                } else {
                    Failure::Transport(format!("docker: {what}: cannot reach the daemon: {e}"))
                }
            })?;
            exchange(stream, &request, sink)
                .await
                .map_err(|why| Failure::Transport(format!("docker: {what}: {why}")))
        })
        .await;
        let status = match exchanged {
            Ok(status) => status?,
            Err(_elapsed) => {
                return Err(Failure::Transport(format!(
                    "docker: {what} timed out after {limit:?}"
                )));
            }
        };
        if (200..300).contains(&status) {
            Ok(status)
        } else {
            Err(Failure::Status {
                what: what.to_owned(),
                status,
                message: sink.message(),
            })
        }
    }

    /// An ordinary call: [`CALL_TIMEOUT`], a JSON answer (`Null` for none).
    async fn json(
        &self,
        what: &str,
        method: &str,
        target: &str,
        body: Option<&Value>,
    ) -> Result<Value, Failure> {
        let mut sink = Sink::collect(MAX_BODY);
        self.call(what, method, target, body, self.timeouts.call, &mut sink)
            .await?;
        let Sink::Collect { body, .. } = sink else {
            return Ok(Value::Null);
        };
        if body.iter().all(u8::is_ascii_whitespace) {
            return Ok(Value::Null);
        }
        serde_json::from_slice(&body).map_err(|e| {
            Failure::Transport(format!("docker: {what}: the answer is not JSON ({e})"))
        })
    }

    /// Force-remove a container with its anonymous volumes. One that is not
    /// there is removed.
    async fn remove(&self, id: &str) -> Result<(), Failure> {
        let target = format!("/containers/{}?force=true&v=true", segment(id));
        let what = format!("removing container {}", short(id));
        let mut sink = Sink::collect(MAX_BODY);
        match self
            .call(
                &what,
                "DELETE",
                &target,
                None,
                self.timeouts.remove,
                &mut sink,
            )
            .await
        {
            Ok(_) => Ok(()),
            Err(failure) if failure.is_status(404) => Ok(()),
            Err(failure) => Err(failure),
        }
    }

    /// One `POST /containers/create`. The body holds the cluster secret, and
    /// a daemon may quote a request back in its refusal: `secret` is taken
    /// out of what it said here, before any of it is clipped.
    async fn post_create(
        &self,
        what: &str,
        target: &str,
        body: &Value,
        secret: &str,
    ) -> Result<Value, Failure> {
        self.json(what, "POST", target, Some(body))
            .await
            .map_err(|failure| failure.without(secret))
    }

    /// Create the container `name` from `body`; its id. An image that is not
    /// on the host any more (the daemon's 404: somebody pruned it while the
    /// orchestrator ran) is pulled again, once, under the pull's time limit.
    /// A name that is taken by a leftover shard of this orchestrator is
    /// freed, once; one taken by anything else is an error and that container
    /// is left alone.
    async fn create(&self, name: &str, body: &Value, secret: &str) -> Result<String, LaunchError> {
        let target = format!("/containers/create?name={}", query_value(name));
        let what = format!("creating container {name}");
        let mut first = self.post_create(&what, &target, body, secret).await;
        if matches!(&first, Err(failure) if failure.is_status(404)) {
            self.ensure_image().await?;
            first = self.post_create(&what, &target, body, secret).await;
        }
        let created = match first {
            Err(failure) if failure.is_status(409) => {
                let inspect = format!("/containers/{}/json", segment(name));
                let looking = format!("inspecting the container that holds the name {name}");
                match self.json(&looking, "GET", &inspect, None).await {
                    Ok(holder) => {
                        if !is_shard_of(
                            &holder["Config"]["Labels"],
                            &self.options.cluster,
                            &self.options.instance,
                        ) {
                            return Err(LaunchError::Spawn(format!(
                                "a container named {name} exists that is not a shard of this \
                                 orchestrator (it does not carry the labels of cluster {:?}, \
                                 instance {:?}), and it is left alone: remove it, or choose \
                                 another --docker-name-prefix",
                                self.options.cluster, self.options.instance
                            )));
                        }
                        let id = holder["Id"].as_str().unwrap_or(name);
                        self.remove(id).await.map_err(Failure::launch)?;
                    }
                    // Gone in the meantime: the name is free.
                    Err(failure) if failure.is_status(404) => {}
                    Err(failure) => return Err(failure.launch()),
                }
                self.post_create(&what, &target, body, secret).await
            }
            other => other,
        }
        .map_err(Failure::launch)?;
        match created["Id"].as_str() {
            Some(id) if !id.is_empty() => Ok(id.to_owned()),
            _ => Err(LaunchError::Spawn(format!(
                "docker: {what}: the daemon's answer names no container id"
            ))),
        }
    }

    /// The container's address on `network`.
    async fn address(&self, id: &str, name: &str, network: &str) -> Result<IpAddr, LaunchError> {
        let target = format!("/containers/{}/json", segment(id));
        let what = format!("inspecting container {name}");
        let found = self
            .json(&what, "GET", &target, None)
            .await
            .map_err(Failure::launch)?;
        let ip = found["NetworkSettings"]["Networks"][network]["IPAddress"]
            .as_str()
            .unwrap_or("");
        ip.parse().map_err(|_| {
            let state = found["State"]["Status"].as_str().unwrap_or("unknown");
            let code = found["State"]["ExitCode"].as_i64().unwrap_or(0);
            LaunchError::Spawn(format!(
                "container {name} has no address on the network {network:?} right after its \
                 start (state {state}, exit code {code}): the shard did not stay up"
            ))
        })
    }

    /// Remove a container of a launch that failed, and forget the launch.
    async fn discard(&self, launch: LaunchId, id: &str) {
        if let Err(failure) = self.remove(id).await {
            eprintln!("space-orchestrator: {failure}");
        }
        let mut table = self.table();
        if table.get(&launch).is_some_and(|entry| entry.id == id) {
            table.remove(&launch);
        }
    }

    /// Whether `launch` is still wanted as the container `id`: not stopped
    /// while it was starting.
    fn still_wanted(&self, launch: LaunchId, id: &str) -> bool {
        self.table()
            .get(&launch)
            .is_some_and(|entry| entry.id == id && !entry.killed)
    }

    /// [`Launcher::spawn`], with error texts that may still name the secret.
    async fn start(&self, spec: &SpawnSpec) -> Result<ShardHandle, LaunchError> {
        let launch = spec.launch;
        let port = spec.host_port;
        if port == 0 {
            return Err(LaunchError::Spawn(
                "the docker launcher needs a host port for the shard (none of the configured \
                 range is free)"
                    .into(),
            ));
        }
        let resolved = self.resolved().await?;
        let name = format!("{}{port}", self.options.name_prefix);
        let stopped = || {
            LaunchError::Spawn(format!(
                "{launch} was stopped while its container was starting"
            ))
        };

        // The launch is in the table from here on: `kill` takes it out, and
        // that is how a launch that is stopped while it starts is noticed.
        let before = self.table().insert(
            launch,
            Entry {
                id: String::new(),
                name: name.clone(),
                port,
                started: false,
                exited: false,
                killed: false,
                exit_code: None,
            },
        );
        if let Some(before) = before.filter(|entry| !entry.id.is_empty()) {
            // A launch id comes once; should one come twice, the first
            // container is not left behind.
            if let Err(failure) = self.remove(&before.id).await {
                eprintln!("space-orchestrator: {failure}");
            }
        }

        let body = create_body(&self.options, &resolved, spec);
        let id = match self.create(&name, &body, &spec.secret).await {
            Ok(id) => id,
            Err(err) => {
                let mut table = self.table();
                if table.get(&launch).is_some_and(|entry| entry.id.is_empty()) {
                    table.remove(&launch);
                }
                return Err(err);
            }
        };
        let wanted = {
            let mut table = self.table();
            match table.get_mut(&launch) {
                Some(entry) if entry.id.is_empty() && !entry.killed => {
                    entry.id.clone_from(&id);
                    true
                }
                _ => false,
            }
        };
        if !wanted {
            if let Err(failure) = self.remove(&id).await {
                eprintln!("space-orchestrator: {failure}");
            }
            return Err(stopped());
        }

        let target = format!("/containers/{}/start", segment(&id));
        let what = format!("starting container {name}");
        match self.json(&what, "POST", &target, None).await {
            Ok(_) => {}
            // 304: it runs already.
            Err(failure) if failure.is_status(304) => {}
            Err(failure) => {
                self.discard(launch, &id).await;
                return Err(failure.launch());
            }
        }
        if let Some(entry) = self.table().get_mut(&launch).filter(|e| e.id == id) {
            entry.started = true;
        }

        let ip = match self.address(&id, &name, &resolved.network).await {
            Ok(ip) => ip,
            Err(err) => {
                self.discard(launch, &id).await;
                return Err(err);
            }
        };
        if !self.still_wanted(launch, &id) {
            // `kill` has removed the container.
            return Err(stopped());
        }
        Ok(ShardHandle {
            launch,
            // Not known here: the shard registers by itself.
            shard: None,
            endpoint: SocketAddr::new(ip, port),
            process: format!("container {} ({name})", short(&id)),
        })
    }
}

impl<T: Transport> Launcher for DockerLauncher<T> {
    async fn spawn(&self, spec: SpawnSpec) -> Result<ShardHandle, LaunchError> {
        self.start(&spec)
            .await
            .map_err(|err| without_secret(err, &spec.secret))
    }

    async fn kill(&self, launch: LaunchId) -> Result<(), LaunchError> {
        // Unknown, or stopped already: a no-op, not an error.
        let Some(entry) = self.table().remove(&launch) else {
            return Ok(());
        };
        if entry.id.is_empty() {
            // Its creation is in flight: `spawn` finds the launch gone and
            // removes what it created.
            return Ok(());
        }
        match self.remove(&entry.id).await {
            Ok(()) => Ok(()),
            Err(failure) => {
                // Not removed: kept, so that `shutdown` tries again.
                self.table().entry(launch).or_insert(Entry {
                    killed: true,
                    ..entry
                });
                Err(failure.launch())
            }
        }
    }

    async fn poll(&self, launch: LaunchId) -> Result<Poll, LaunchError> {
        let (id, name) = {
            let table = self.table();
            let Some(entry) = table.get(&launch) else {
                return Ok(Poll::Dead);
            };
            if entry.killed || entry.exited {
                return Ok(Poll::Dead);
            }
            if !entry.started {
                // Created and not started yet: "not running" would be true
                // and would not mean that it died.
                return Ok(Poll::Alive);
            }
            (entry.id.clone(), entry.name.clone())
        };
        let target = format!("/containers/{}/json", segment(&id));
        let what = format!("asking after container {name}");
        let seen = |exit_code: Option<i32>| {
            if let Some(entry) = self.table().get_mut(&launch).filter(|e| e.id == id) {
                entry.exited = true;
                entry.exit_code = exit_code;
            }
            Ok(Poll::Dead)
        };
        let found = match self.json(&what, "GET", &target, None).await {
            Ok(found) => found,
            // Removed behind the launcher's back.
            Err(failure) if failure.is_status(404) => return seen(None),
            // No answer is no verdict: the orchestrator asks again.
            Err(failure) => return Err(failure.launch()),
        };
        match found["State"]["Running"].as_bool() {
            Some(true) => Ok(Poll::Alive),
            Some(false) => seen(
                found["State"]["ExitCode"]
                    .as_i64()
                    .and_then(|code| i32::try_from(code).ok()),
            ),
            None => Err(LaunchError::Spawn(format!(
                "docker: {what}: the answer does not say whether it runs"
            ))),
        }
    }

    async fn exit_code(&self, launch: LaunchId) -> Option<i32> {
        self.table().get(&launch).and_then(|entry| entry.exit_code)
    }

    async fn shutdown(&self) {
        // Emptied first: a launch whose creation is in flight finds itself
        // gone and removes its own container.
        let entries: Vec<Entry> = self.table().drain().map(|(_, entry)| entry).collect();
        let removals: Vec<Pin<Box<dyn Future<Output = ()> + Send + '_>>> = entries
            .iter()
            .filter(|entry| !entry.id.is_empty())
            .map(|entry| {
                let removal: Pin<Box<dyn Future<Output = ()> + Send + '_>> = Box::pin(async move {
                    if let Err(failure) = self.remove(&entry.id).await {
                        eprintln!(
                            "space-orchestrator: the shard container {} (port {}) was not \
                             removed: {failure}",
                            entry.name, entry.port
                        );
                    }
                });
                removal
            })
            .collect();
        // At once, and under one bound: a daemon that hangs on one container
        // holds up neither the others nor the orchestrator's exit.
        let limit = self.timeouts.shutdown;
        if tokio::time::timeout(limit, all(removals)).await.is_err() {
            eprintln!(
                "space-orchestrator: docker: removing the shard containers did not finish within \
                 {limit:?}; what is left carries the labels {LABEL_CLUSTER}={} and \
                 {LABEL_INSTANCE}={} and is removed when the orchestrator starts again",
                self.options.cluster, self.options.instance
            );
        }
    }
}

/// Run every future to its end, all at once.
async fn all(mut futures: Vec<Pin<Box<dyn Future<Output = ()> + Send + '_>>>) {
    std::future::poll_fn(|cx| {
        futures.retain_mut(|future| future.as_mut().poll(cx).is_pending());
        if futures.is_empty() {
            std::task::Poll::Ready(())
        } else {
            std::task::Poll::Pending
        }
    })
    .await;
}

/// What a shard container is created from (`POST /containers/create`).
///
/// It holds the cluster secret (in `Env`): it is sent to the daemon and goes
/// nowhere else — never into a log line or an error.
fn create_body(options: &DockerOptions, resolved: &Resolved, spec: &SpawnSpec) -> Value {
    let port = spec.host_port;
    let listen = format!("0.0.0.0:{port}");
    let udp = format!("{port}/udp");
    let network = resolved.network.as_str();
    let mut body = json!({
        "Image": options.image,
        "Entrypoint": ["space-server"],
        "Cmd": [
            "--orchestrator", resolved.control,
            "--cluster", spec.cluster,
            "--bind", listen,
            "--advertise", listen,
            "--client-endpoint", listen,
        ],
        // By number: also an image without a `USER` line runs no shard as root.
        "User": SHARD_USER,
        "Labels": {
            LABEL_CLUSTER: options.cluster,
            LABEL_ROLE: ROLE_SHARD,
            LABEL_INSTANCE: options.instance,
            LABEL_LAUNCH: spec.launch.0.to_string(),
            LABEL_PORT: port.to_string(),
        },
        "ExposedPorts": { udp.as_str(): {} },
        "HostConfig": {
            "NetworkMode": network,
            "PortBindings": {
                udp.as_str(): [{ "HostIp": options.publish_ip, "HostPort": port.to_string() }]
            },
            "RestartPolicy": { "Name": "no" },
            "Privileged": false,
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges"],
            // A shard writes no file (its log is its standard output).
            "ReadonlyRootfs": true,
            "PidsLimit": PIDS_LIMIT,
            "AutoRemove": false,
        },
        "NetworkingConfig": { "EndpointsConfig": { network: {} } },
    });
    if options.memory_mib > 0 {
        // The same for both: no swap on top of the memory.
        let bytes = options.memory_mib.saturating_mul(1024 * 1024);
        body["HostConfig"]["Memory"] = json!(bytes);
        body["HostConfig"]["MemorySwap"] = json!(bytes);
    }
    if !spec.secret.is_empty() {
        // In the environment, never in the command (which `docker ps` shows).
        body["Env"] = json!([format!("{ENV_CLUSTER_SECRET}={}", spec.secret)]);
    }
    body
}

/// Whether `labels` (a container's) say "a shard of the orchestrator
/// `instance` of `cluster`". All three labels: a shard of another stack on
/// this host is a stranger, also when both stacks use one cluster name.
fn is_shard_of(labels: &Value, cluster: &str, instance: &str) -> bool {
    labels[LABEL_CLUSTER].as_str() == Some(cluster)
        && labels[LABEL_ROLE].as_str() == Some(ROLE_SHARD)
        && labels[LABEL_INSTANCE].as_str() == Some(instance)
}

/// `err` with every occurrence of `secret` taken out of its text.
fn without_secret(err: LaunchError, secret: &str) -> LaunchError {
    if secret.is_empty() {
        return err;
    }
    let clean = |text: String| text.replace(secret, "<redacted>");
    match err {
        LaunchError::Spawn(text) => LaunchError::Spawn(clean(text)),
        LaunchError::Unsupported(text) => LaunchError::Unsupported(clean(text)),
        LaunchError::Unavailable(text) => LaunchError::Unavailable(clean(text)),
        LaunchError::Unreachable(launch) => LaunchError::Unreachable(launch),
    }
}

/// The first twelve characters of a container id, as `docker ps` prints it.
fn short(id: &str) -> &str {
    id.char_indices().nth(12).map_or(id, |(at, _)| &id[..at])
}

/// `text` on one line and at most [`MAX_ERROR_TEXT`] characters.
fn clip(text: &str) -> String {
    let mut clipped: String = text
        .trim()
        .chars()
        .take(MAX_ERROR_TEXT)
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    if text.trim().chars().count() > MAX_ERROR_TEXT {
        clipped.push_str("...");
    }
    clipped
}

/// `text` percent-encoded: everything but the unreserved characters and
/// those of `keep`.
fn encode(text: &str, keep: &[u8]) -> String {
    let mut encoded = String::with_capacity(text.len());
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || b"-._~".contains(&byte) || keep.contains(&byte) {
            encoded.push(char::from(byte));
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

/// A query value: everything reserved is encoded.
fn query_value(text: &str) -> String {
    encode(text, b"")
}

/// One path segment (`:` and `@` may stand in one as they are).
fn segment(text: &str) -> String {
    encode(text, b":@")
}

/// An image reference as the path below `/images/`: its own slashes stay —
/// the daemon's route takes the rest of the path as the name — and each part
/// between them is encoded.
fn image_path(image: &str) -> String {
    image.split('/').map(segment).collect::<Vec<_>>().join("/")
}

/// An image reference as the `fromImage` and `tag` of a pull: the tag behind
/// the last `:` of the last path part (a `:` before that is a registry's
/// port), a digest (`@sha256:…`) in the tag's place, `latest` for neither.
fn split_image(image: &str) -> (&str, &str) {
    if let Some((repository, digest)) = image.split_once('@') {
        return (repository, digest);
    }
    let name_at = image.rfind('/').map_or(0, |slash| slash + 1);
    match image[name_at..].rfind(':') {
        Some(colon) => (&image[..name_at + colon], &image[name_at + colon + 1..]),
        None => (image, "latest"),
    }
}

/// The bytes of one request.
fn request_bytes(method: &str, target: &str, body: Option<&str>) -> Vec<u8> {
    let mut request = format!(
        "{method} {target} HTTP/1.1\r\nHost: docker\r\nConnection: close\r\nAccept: \
         application/json\r\n"
    );
    match body {
        Some(body) => {
            request.push_str(&format!(
                "Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                body.len()
            ));
            request.push_str(body);
        }
        None if method == "GET" => request.push_str("\r\n"),
        None => request.push_str("Content-Length: 0\r\n\r\n"),
    }
    request.into_bytes()
}

/// What is kept of a pull's progress stream: the last word of each kind.
#[derive(Debug, Default)]
struct Progress {
    /// The line being read, up to [`MAX_PROGRESS_LINE`].
    line: Vec<u8>,
    /// The line being read is longer than that and is skipped.
    overlong: bool,
    /// The last `error` (or, of a refusal, `message`).
    error: Option<String>,
    /// The last `status`.
    status: Option<String>,
}

impl Progress {
    fn push(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            if byte == b'\n' {
                self.end_line();
            } else if self.line.len() < MAX_PROGRESS_LINE {
                self.line.push(byte);
            } else {
                self.overlong = true;
            }
        }
    }

    fn end_line(&mut self) {
        let line = std::mem::take(&mut self.line);
        if std::mem::take(&mut self.overlong) {
            return;
        }
        let Ok(said) = serde_json::from_slice::<Value>(&line) else {
            return;
        };
        if let Some(error) = said["error"].as_str().or_else(|| said["message"].as_str()) {
            self.error = Some(error.to_owned());
        }
        if let Some(status) = said["status"].as_str() {
            self.status = Some(status.to_owned());
        }
    }
}

/// Where the body of an answer goes.
#[derive(Debug)]
enum Sink {
    /// Kept whole, up to `limit` bytes.
    Collect { body: Vec<u8>, limit: usize },
    /// Read line by line and thrown away (a pull's progress).
    Lines(Progress),
}

impl Sink {
    fn collect(limit: usize) -> Self {
        Sink::Collect {
            body: Vec::new(),
            limit,
        }
    }

    /// Take the next bytes of the body.
    fn push(&mut self, bytes: &[u8]) -> Result<(), String> {
        match self {
            Sink::Collect { body, limit } => {
                if body.len() + bytes.len() > *limit {
                    return Err(format!("the answer's body is over {limit} bytes"));
                }
                body.extend_from_slice(bytes);
            }
            Sink::Lines(progress) => progress.push(bytes),
        }
        Ok(())
    }

    /// Whether a body of `length` bytes is refused before any of it is read.
    fn refuses(&self, length: u64) -> Option<String> {
        match self {
            Sink::Collect { limit, .. } if length > *limit as u64 => {
                Some(format!("the answer's body is over {limit} bytes"))
            }
            _ => None,
        }
    }

    /// The body has ended.
    fn finish(&mut self) {
        if let Sink::Lines(progress) = self {
            progress.end_line();
        }
    }

    /// What the daemon said went wrong: the `message` of its error body, else
    /// the body as text — the first [`MAX_RAW_TEXT`] characters, as they are
    /// (see [`Failure::Status`]: clipped when shown).
    fn message(&self) -> String {
        let raw = |text: &str| text.trim().chars().take(MAX_RAW_TEXT).collect::<String>();
        match self {
            Sink::Collect { body, .. } => match serde_json::from_slice::<Value>(body) {
                Ok(said) if said["message"].is_string() => {
                    raw(said["message"].as_str().unwrap_or(""))
                }
                _ => raw(&String::from_utf8_lossy(body)),
            },
            Sink::Lines(progress) => raw(progress
                .error
                .as_deref()
                .or(progress.status.as_deref())
                .unwrap_or("")),
        }
    }
}

/// How the body of an answer is framed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Framing {
    /// There is none (`204`, `304`, `1xx`).
    Nothing,
    /// `Content-Length` bytes.
    Length(u64),
    /// `Transfer-Encoding: chunked`.
    Chunked,
    /// Until the daemon closes the connection.
    ToEnd,
}

/// The status and the framing an answer's head announces.
fn parse_head(head: &str) -> Result<(u16, Framing), String> {
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap_or("");
    let mut parts = status_line.split(' ');
    let version = parts.next().unwrap_or("");
    let status = parts
        .next()
        .filter(|code| code.len() == 3)
        .and_then(|code| code.parse::<u16>().ok());
    let (true, Some(status)) = (version.starts_with("HTTP/1."), status) else {
        return Err(format!(
            "the answer does not start like HTTP ({:?})",
            clip(status_line)
        ));
    };
    let mut length = None;
    let mut chunked = false;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            return Err("the answer has a header line without a colon".into());
        };
        let value = value.trim();
        if name.eq_ignore_ascii_case("transfer-encoding") {
            chunked |= value
                .split(',')
                .any(|coding| coding.trim().eq_ignore_ascii_case("chunked"));
        } else if name.eq_ignore_ascii_case("content-length") {
            if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
                return Err("the answer has an invalid Content-Length".into());
            }
            let value: u64 = value
                .parse()
                .map_err(|_| "the answer has an invalid Content-Length".to_owned())?;
            if length.is_some_and(|earlier| earlier != value) {
                return Err("the answer has conflicting Content-Length headers".into());
            }
            length = Some(value);
        }
    }
    let framing = if status < 200 || status == 204 || status == 304 {
        Framing::Nothing
    } else if chunked {
        Framing::Chunked
    } else if let Some(length) = length {
        Framing::Length(length)
    } else {
        Framing::ToEnd
    };
    Ok((status, framing))
}

/// The reading side of one connection: what has arrived and is not used yet.
struct Wire<S> {
    stream: S,
    buffer: Vec<u8>,
    /// Where the unused bytes of `buffer` start.
    at: usize,
}

impl<S: AsyncRead + Unpin> Wire<S> {
    fn unused(&self) -> &[u8] {
        &self.buffer[self.at..]
    }

    /// Read more; how many bytes came (0: the daemon closed).
    async fn more(&mut self) -> Result<usize, String> {
        if self.at > 0 {
            self.buffer.drain(..self.at);
            self.at = 0;
        }
        let mut piece = [0u8; 4096];
        let got = self
            .stream
            .read(&mut piece)
            .await
            .map_err(|e| format!("reading the answer: {e}"))?;
        self.buffer.extend_from_slice(&piece[..got]);
        Ok(got)
    }

    /// The bytes up to `end` (which is consumed and not returned), at most
    /// `limit` of them.
    async fn until(&mut self, end: &[u8], limit: usize, what: &str) -> Result<Vec<u8>, String> {
        loop {
            let unused = self.unused();
            let found = unused.windows(end.len()).position(|window| window == end);
            match found {
                Some(at) if at <= limit => {
                    let taken = unused[..at].to_vec();
                    self.at += at + end.len();
                    return Ok(taken);
                }
                Some(_) => return Err(format!("the answer's {what} is over {limit} bytes")),
                // (Not `> limit`: the end may have arrived in part.)
                None if unused.len() > limit + end.len() => {
                    return Err(format!("the answer's {what} is over {limit} bytes"));
                }
                None => {}
            }
            if self.more().await? == 0 {
                return Err(format!(
                    "the daemon closed the connection in the middle of the answer's {what}"
                ));
            }
        }
    }

    /// Pass exactly `length` bytes on to `sink`.
    async fn pass(&mut self, mut length: u64, sink: &mut Sink) -> Result<(), String> {
        while length > 0 {
            if self.unused().is_empty() && self.more().await? == 0 {
                return Err(
                    "the daemon closed the connection in the middle of the answer's body".into(),
                );
            }
            let unused = self.unused();
            let take = usize::try_from(length).map_or(unused.len(), |l| l.min(unused.len()));
            sink.push(&unused[..take])?;
            self.at += take;
            length -= take as u64;
        }
        Ok(())
    }

    /// Pass a chunked body on to `sink`: each chunk's size line (hexadecimal,
    /// extensions behind a `;` ignored), its bytes, its line end; after the
    /// chunk of size 0 the trailer, up to the empty line.
    async fn pass_chunks(&mut self, sink: &mut Sink) -> Result<(), String> {
        loop {
            let line = self.until(b"\r\n", MAX_LINE, "chunk size line").await?;
            let line = String::from_utf8_lossy(&line);
            let size = line.split(';').next().unwrap_or("").trim();
            if size.is_empty() || !size.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("the answer has an invalid chunk size".into());
            }
            let size = u64::from_str_radix(size, 16)
                .map_err(|_| "the answer has an invalid chunk size".to_owned())?;
            if size == 0 {
                let mut trailer = 0;
                loop {
                    let line = self.until(b"\r\n", MAX_LINE, "trailer line").await?;
                    if line.is_empty() {
                        return Ok(());
                    }
                    trailer += line.len();
                    if trailer > MAX_HEAD {
                        return Err(format!("the answer's trailer is over {MAX_HEAD} bytes"));
                    }
                }
            }
            self.pass(size, sink).await?;
            while self.unused().len() < 2 {
                if self.more().await? == 0 {
                    return Err(
                        "the daemon closed the connection in the middle of the answer's body"
                            .into(),
                    );
                }
            }
            if !self.unused().starts_with(b"\r\n") {
                return Err("the answer has a chunk that does not end where it says".into());
            }
            self.at += 2;
        }
    }

    /// Pass everything up to the end of the connection on to `sink`.
    async fn pass_rest(&mut self, sink: &mut Sink) -> Result<(), String> {
        loop {
            sink.push(self.unused())?;
            self.at = self.buffer.len();
            if self.more().await? == 0 {
                return Ok(());
            }
        }
    }
}

/// One request and its answer on `stream`: the status, the body into `sink`.
///
/// The request's side of the connection is **not** closed behind the request:
/// the daemon takes that for a client that went away and drops the call.
async fn exchange<S>(stream: S, request: &[u8], sink: &mut Sink) -> Result<u16, String>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    let mut wire = Wire {
        stream,
        buffer: Vec::new(),
        at: 0,
    };
    wire.stream
        .write_all(request)
        .await
        .map_err(|e| format!("sending the request: {e}"))?;
    wire.stream
        .flush()
        .await
        .map_err(|e| format!("sending the request: {e}"))?;
    // The head ends with an empty line; its last header line's end is the
    // first half of that.
    let head = wire.until(b"\r\n\r\n", MAX_HEAD, "head").await?;
    let (status, framing) = parse_head(&String::from_utf8_lossy(&head))?;
    match framing {
        Framing::Nothing => {}
        Framing::Length(length) => {
            if let Some(why) = sink.refuses(length) {
                return Err(why);
            }
            wire.pass(length, sink).await?;
        }
        Framing::Chunked => wire.pass_chunks(sink).await?,
        Framing::ToEnd => wire.pass_rest(sink).await?,
    }
    sink.finish();
    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::io::DuplexStream;
    use tokio::sync::Notify;

    const SECRET: &str = "the-cluster-secret-nobody-may-print";
    const ID: &str = "0123456789abcdef0123456789abcdef";

    /// What the fake daemon answers one request with.
    enum Reply {
        /// A status and a JSON body, framed by `Content-Length`.
        Json(u16, Value),
        /// A status and no body.
        Status(u16),
        /// These bytes, as they are.
        Raw(Vec<u8>),
        /// Nothing, ever: the connection stays open.
        Hang,
        /// Nothing: the connection is closed.
        Close,
        /// The inner answer, once the test says so.
        Gated(Arc<Notify>, Box<Reply>),
    }

    type Handler = Box<dyn Fn(&str, &str, &str) -> Reply + Send + Sync>;

    struct Daemon {
        /// `(method, path and query, body)` of every request, in order.
        calls: Mutex<Vec<(String, String, String)>>,
        /// The head of every request.
        heads: Mutex<Vec<String>>,
        handler: Handler,
        /// The size of the pipe between client and daemon: a small one makes
        /// every read end in the middle of something.
        pipe: usize,
    }

    /// A Docker daemon in memory: every connection is one request, answered
    /// by the handler and recorded.
    #[derive(Clone)]
    struct Fake(Arc<Daemon>);

    impl Fake {
        fn new(handler: impl Fn(&str, &str, &str) -> Reply + Send + Sync + 'static) -> Self {
            Self::piped(4096, handler)
        }

        fn piped(
            pipe: usize,
            handler: impl Fn(&str, &str, &str) -> Reply + Send + Sync + 'static,
        ) -> Self {
            Self(Arc::new(Daemon {
                calls: Mutex::new(Vec::new()),
                heads: Mutex::new(Vec::new()),
                handler: Box::new(handler),
                pipe,
            }))
        }

        fn calls(&self) -> Vec<(String, String, String)> {
            self.0.calls.lock().unwrap().clone()
        }

        /// `METHOD path?query` of every request, in order.
        fn lines(&self) -> Vec<String> {
            self.calls()
                .into_iter()
                .map(|(method, target, _)| format!("{method} {target}"))
                .collect()
        }
    }

    impl Transport for Fake {
        type Stream = DuplexStream;

        async fn connect(&self) -> io::Result<DuplexStream> {
            let (client, server) = tokio::io::duplex(self.0.pipe);
            tokio::spawn(serve(Arc::clone(&self.0), server));
            Ok(client)
        }
    }

    async fn serve(daemon: Arc<Daemon>, mut stream: DuplexStream) {
        let mut request = Vec::new();
        let mut piece = [0u8; 1024];
        let head_end = loop {
            if let Some(at) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                break at;
            }
            match stream.read(&mut piece).await {
                Ok(0) | Err(_) => return,
                Ok(got) => request.extend_from_slice(&piece[..got]),
            }
        };
        let head = String::from_utf8_lossy(&request[..head_end]).into_owned();
        let length = head
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().expect("a length"))
            })
            .unwrap_or(0);
        while request.len() < head_end + 4 + length {
            match stream.read(&mut piece).await {
                Ok(0) | Err(_) => return,
                Ok(got) => request.extend_from_slice(&piece[..got]),
            }
        }
        let body = String::from_utf8_lossy(&request[head_end + 4..]).into_owned();
        let mut first = head.lines().next().unwrap_or("").split(' ');
        let method = first.next().unwrap_or("").to_owned();
        let target = first.next().unwrap_or("").to_owned();
        daemon.heads.lock().unwrap().push(head.clone());
        daemon
            .calls
            .lock()
            .unwrap()
            .push((method.clone(), target.clone(), body.clone()));
        let mut reply = (daemon.handler)(&method, &target, &body);
        if let Reply::Gated(gate, inner) = reply {
            gate.notified().await;
            reply = *inner;
        }
        let bytes = match reply {
            Reply::Json(status, value) => {
                let body = value.to_string();
                format!(
                    "HTTP/1.1 {status} Whatever\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\n\r\n{body}",
                    body.len()
                )
                .into_bytes()
            }
            Reply::Status(status) => {
                format!("HTTP/1.1 {status} Whatever\r\nContent-Length: 0\r\n\r\n").into_bytes()
            }
            Reply::Raw(bytes) => bytes,
            Reply::Hang => {
                std::future::pending::<()>().await;
                return;
            }
            Reply::Close | Reply::Gated(..) => return,
        };
        let _ = stream.write_all(&bytes).await;
    }

    fn options() -> DockerOptions {
        DockerOptions {
            image: "ghcr.io/eliasstepanik/space-shard:sha-abc1234".into(),
            network: Some("space_default".into()),
            control: Some("172.20.0.2:7900".into()),
            publish_ip: "0.0.0.0".into(),
            cluster: "test-cluster".into(),
            instance: "inst-a".into(),
            ..DockerOptions::default()
        }
    }

    fn spec(launch: u64, host_port: u16, secret: &str) -> SpawnSpec {
        SpawnSpec {
            launch: LaunchId(launch),
            cluster: "test-cluster".into(),
            control_addr: "0.0.0.0:7900".parse().unwrap(),
            host_port,
            secret: secret.into(),
        }
    }

    /// The inspection of a running shard container with this address.
    fn running(ip: &str) -> Value {
        json!({
            "Id": ID,
            "State": { "Status": "running", "Running": true, "ExitCode": 0 },
            "NetworkSettings": { "Networks": { "space_default": { "IPAddress": ip } } },
        })
    }

    /// A daemon on which everything works: create, start, inspect, delete.
    fn obliging(method: &str, target: &str, _body: &str) -> Reply {
        match (method, target) {
            ("POST", t) if t.starts_with("/containers/create") => {
                Reply::Json(201, json!({ "Id": ID, "Warnings": [] }))
            }
            ("POST", t) if t.ends_with("/start") => Reply::Status(204),
            ("GET", t) if t.starts_with("/containers/") => Reply::Json(200, running("172.20.0.5")),
            ("DELETE", _) => Reply::Status(204),
            _ => Reply::Json(500, json!({ "message": "the test did not expect this" })),
        }
    }

    fn launcher(fake: &Fake) -> DockerLauncher<Fake> {
        DockerLauncher::with_transport(options(), fake.clone())
    }

    const DELETE: &str = "DELETE /containers/0123456789abcdef0123456789abcdef?force=true&v=true";
    const INSPECT: &str = "GET /containers/0123456789abcdef0123456789abcdef/json";
    const START: &str = "POST /containers/0123456789abcdef0123456789abcdef/start";

    /// What a shard container is made of, to the letter: the image, the
    /// command, the user, the labels, the published UDP port, the network, no
    /// restart, a read-only root, the process and memory limits — and nothing
    /// else: no mount, no socket, no privilege. The secret is in its
    /// environment and nowhere in its command; without one there is no
    /// environment at all, and without a memory limit no `Memory`.
    ///
    /// Mutation this catches: adding `Binds` (the Docker socket, say) or
    /// `"Privileged": true` to the body, putting the secret into `Cmd`, a
    /// restart policy other than `no`, a TCP port binding; dropping
    /// `ReadonlyRootfs`, `PidsLimit`, `User` or the memory limit, or setting
    /// `MemorySwap` to anything but the memory (swap on top of the limit).
    #[tokio::test]
    async fn a_shard_container_is_created_from_exactly_this() {
        let fake = Fake::new(obliging);
        let docker = launcher(&fake);
        let handle = docker.spawn(spec(3, 7_871, SECRET)).await.expect("starts");
        let calls = fake.calls();
        assert_eq!(
            fake.lines(),
            [
                "POST /containers/create?name=space-shard-7871",
                START,
                INSPECT
            ]
        );
        let body: Value = serde_json::from_str(&calls[0].2).expect("a JSON body");
        assert_eq!(
            body,
            json!({
                "Image": "ghcr.io/eliasstepanik/space-shard:sha-abc1234",
                "Entrypoint": ["space-server"],
                "Cmd": [
                    "--orchestrator", "172.20.0.2:7900",
                    "--cluster", "test-cluster",
                    "--bind", "0.0.0.0:7871",
                    "--advertise", "0.0.0.0:7871",
                    "--client-endpoint", "0.0.0.0:7871",
                ],
                "Env": [format!("SPACE_CLUSTER_SECRET={SECRET}")],
                "User": "10001:10001",
                "Labels": {
                    "space.cluster": "test-cluster",
                    "space.role": "shard",
                    "space.instance": "inst-a",
                    "space.launch": "3",
                    "space.port": "7871",
                },
                "ExposedPorts": { "7871/udp": {} },
                "HostConfig": {
                    "NetworkMode": "space_default",
                    "PortBindings": { "7871/udp": [{ "HostIp": "0.0.0.0", "HostPort": "7871" }] },
                    "RestartPolicy": { "Name": "no" },
                    "Privileged": false,
                    "CapDrop": ["ALL"],
                    "SecurityOpt": ["no-new-privileges"],
                    "ReadonlyRootfs": true,
                    "PidsLimit": 1024,
                    "Memory": 2_147_483_648u64,
                    "MemorySwap": 2_147_483_648u64,
                    "AutoRemove": false,
                },
                "NetworkingConfig": { "EndpointsConfig": { "space_default": {} } },
            })
        );
        // Said again, on their own: these are the ones that matter most.
        let host = body["HostConfig"].as_object().unwrap();
        for never in [
            "Binds",
            "Mounts",
            "VolumesFrom",
            "Devices",
            "CapAdd",
            "PidMode",
        ] {
            assert!(!host.contains_key(never), "{never} is never set");
        }
        assert!(body.get("Volumes").is_none());
        assert_eq!(host["Privileged"], json!(false));
        assert!(!calls[0].2.contains("docker.sock"));
        assert!(!body["Cmd"].to_string().contains(SECRET));
        assert_eq!(calls[0].2.matches(SECRET).count(), 1, "in Env, once");
        // The start and the inspection carry nothing.
        assert_eq!(calls[1].2, "");
        assert_eq!(calls[2].2, "");
        assert_eq!(handle.launch, LaunchId(3));
        // The request itself: one per connection, never with credentials.
        for head in fake.0.heads.lock().unwrap().iter() {
            assert!(head.contains("\r\nHost: docker"), "{head}");
            assert!(head.contains("\r\nConnection: close"), "{head}");
            assert!(!head.to_ascii_lowercase().contains("x-registry-auth"));
        }

        // No secret: no environment (and not an empty variable).
        let fake = Fake::new(obliging);
        let docker = launcher(&fake);
        docker.spawn(spec(4, 7_872, "")).await.expect("starts");
        let body: Value = serde_json::from_str(&fake.calls()[0].2).unwrap();
        assert!(body.get("Env").is_none(), "{body}");

        // No memory limit asked for: none set (and the others stay).
        let fake = Fake::new(obliging);
        let unlimited = DockerOptions {
            memory_mib: 0,
            ..options()
        };
        let docker = DockerLauncher::with_transport(unlimited, fake.clone());
        docker.spawn(spec(5, 7_873, "")).await.expect("starts");
        let body: Value = serde_json::from_str(&fake.calls()[0].2).unwrap();
        let host = body["HostConfig"].as_object().unwrap();
        assert!(!host.contains_key("Memory") && !host.contains_key("MemorySwap"));
        assert_eq!(host["ReadonlyRootfs"], json!(true));
        assert_eq!(host["PidsLimit"], json!(1024));
    }

    /// The daemon's refusal is cleaned of the secret **before** it is cut to
    /// length: a secret that lies across the cut is gone whole, not left
    /// standing with its front half.
    ///
    /// Mutation this catches: `post_create` returning the failure as it is
    /// (clipping first, redacting the clipped text afterwards): the text ends
    /// with the first characters of the secret, which no longer match it.
    #[tokio::test]
    async fn a_secret_that_lies_across_the_cut_is_taken_out_whole() {
        // The secret starts ten characters before the cut.
        let fake = Fake::new(|_, _, _| {
            Reply::Json(
                400,
                json!({ "message": format!("{}{SECRET} and more", "x".repeat(MAX_ERROR_TEXT - 10)) }),
            )
        });
        let docker = launcher(&fake);
        let err = docker.spawn(spec(1, 7_870, SECRET)).await.expect_err("no");
        let printed = format!("{err:?} {err}");
        assert!(printed.contains("400"), "{printed}");
        assert!(!printed.contains(&SECRET[..8]), "{printed}");
        assert!(printed.contains("<redact"), "{printed}");
        // And one that starts at the very end of what is kept before the
        // cut: there whole, so taken out whole.
        let fake = Fake::new(|_, _, _| {
            Reply::Json(
                400,
                json!({ "message": format!("{}{SECRET}", "y".repeat(MAX_ERROR_TEXT - 1)) }),
            )
        });
        let docker = launcher(&fake);
        let err = docker.spawn(spec(2, 7_871, SECRET)).await.expect_err("no");
        let printed = format!("{err:?} {err}");
        assert!(printed.contains("yy<"), "{printed}");
        assert!(!printed.contains("yt"), "{printed}");
    }

    /// An image that was removed from the host while the orchestrator ran
    /// (the daemon refuses the creation with 404) is pulled again and the
    /// creation tried once more; an image that cannot be pulled fails the
    /// launch, says so, and leaves nothing in the table.
    ///
    /// Mutation this catches: making sure of the image in `prepare` only
    /// (dropping the 404 arm of `create`): after a prune on the host every
    /// later launch - every replacement, every scale-out - fails for good.
    #[tokio::test]
    async fn a_launch_pulls_an_image_that_was_removed_while_the_orchestrator_ran() {
        let there = Arc::new(AtomicUsize::new(0));
        let images = preparing(
            json!({}),
            Arc::clone(&there),
            "{\"status\":\"Downloaded\"}\n",
        );
        let seen = Arc::clone(&there);
        let fake = Fake::new(move |method, target, body| match (method, target) {
            ("POST", t) if t.starts_with("/containers/create") => {
                if seen.load(Ordering::SeqCst) == 0 {
                    Reply::Json(
                        404,
                        json!({ "message": "No such image: space-shard:sha-abc1234" }),
                    )
                } else {
                    Reply::Json(201, json!({ "Id": ID }))
                }
            }
            (_, t) if t.starts_with("/images/") => images(method, target, body),
            _ => obliging(method, target, body),
        });
        let docker = launcher(&fake);
        let handle = docker.spawn(spec(1, 7_870, SECRET)).await.expect("starts");
        assert_eq!(handle.endpoint, "172.20.0.5:7870".parse().unwrap());
        let pulling =
            "POST /images/create?fromImage=ghcr.io%2Feliasstepanik%2Fspace-shard&tag=sha-abc1234";
        assert_eq!(
            fake.lines(),
            [
                "POST /containers/create?name=space-shard-7870",
                IMAGE,
                pulling,
                IMAGE,
                "POST /containers/create?name=space-shard-7870",
                START,
                INSPECT,
            ]
        );
        // The image is there: the next launch asks nothing about it.
        docker.spawn(spec(2, 7_871, "")).await.expect("starts");
        assert_eq!(fake.calls().len(), 10);

        // Not pullable: the launch fails, in words, and is forgotten.
        let refused = "{\"error\":\"pull access denied for space-shard\"}\n";
        let images = preparing(json!({}), Arc::new(AtomicUsize::new(0)), refused);
        let fake = Fake::new(move |method, target, body| match (method, target) {
            ("POST", t) if t.starts_with("/containers/create") => {
                Reply::Json(404, json!({ "message": "No such image" }))
            }
            (_, t) if t.starts_with("/images/") => images(method, target, body),
            _ => obliging(method, target, body),
        });
        let docker = launcher(&fake);
        let err = docker
            .spawn(spec(1, 7_870, ""))
            .await
            .expect_err("no image");
        assert!(
            matches!(&err, LaunchError::Unavailable(why)
                if why.contains("could not be pulled") && why.contains("pull access denied")),
            "{err:?}"
        );
        assert_eq!(
            fake.lines(),
            [
                "POST /containers/create?name=space-shard-7870",
                IMAGE,
                pulling
            ],
            "one try, and no second creation"
        );
        assert!(docker.table().is_empty());
    }

    /// The launch's endpoint is the container's address on the shards'
    /// network, as the daemon reports it, and the shard's port; its process
    /// is the container's short id and name. Without a port there is no
    /// launch and no call.
    ///
    /// Mutation this catches: returning `0.0.0.0:<port>` (what the shard
    /// binds) or the control address as the endpoint — the orchestrator would
    /// never match the launch to the shard that registers from the
    /// container's address.
    #[tokio::test]
    async fn a_launch_is_found_at_the_containers_address_and_port() {
        let fake = Fake::new(obliging);
        let docker = launcher(&fake);
        let handle = docker.spawn(spec(1, 7_870, "")).await.expect("starts");
        assert_eq!(handle.endpoint, "172.20.0.5:7870".parse().unwrap());
        assert_eq!(handle.shard, None);
        assert_eq!(handle.process, "container 0123456789ab (space-shard-7870)");

        let err = docker.spawn(spec(2, 0, "")).await.expect_err("no port");
        assert!(
            matches!(&err, LaunchError::Spawn(why) if why.contains("needs a host port")),
            "{err:?}"
        );
        assert_eq!(fake.calls().len(), 3, "no call for a launch without a port");
    }

    /// A container that is started and has no address (it exited at once) is
    /// a failed launch, and is not left behind; neither is one whose start
    /// the daemon refuses.
    ///
    /// Mutation this catches: returning the start's error without removing
    /// what was created — every failed launch would leave a container that
    /// holds the name and the port.
    #[tokio::test]
    async fn a_container_that_does_not_start_is_removed() {
        let fake = Fake::new(|method, target, body| match (method, target) {
            ("POST", t) if t.ends_with("/start") => Reply::Json(
                500,
                json!({ "message": "driver failed programming external connectivity: port is already allocated" }),
            ),
            _ => obliging(method, target, body),
        });
        let docker = launcher(&fake);
        let err = docker.spawn(spec(1, 7_870, "")).await.expect_err("fails");
        assert!(
            matches!(&err, LaunchError::Spawn(why)
                if why.contains("500") && why.contains("port is already allocated")
                    && why.contains("starting container space-shard-7870")),
            "{err:?}"
        );
        assert_eq!(
            fake.lines(),
            [
                "POST /containers/create?name=space-shard-7870",
                START,
                DELETE
            ]
        );
        assert_eq!(docker.poll(LaunchId(1)).await.unwrap(), Poll::Dead);
        assert!(docker.table().is_empty());

        let fake = Fake::new(|method, target, body| match (method, target) {
            ("GET", _) => Reply::Json(
                200,
                json!({
                    "State": { "Status": "exited", "Running": false, "ExitCode": 2 },
                    "NetworkSettings": { "Networks": { "space_default": { "IPAddress": "" } } },
                }),
            ),
            _ => obliging(method, target, body),
        });
        let docker = launcher(&fake);
        let err = docker.spawn(spec(1, 7_870, "")).await.expect_err("fails");
        assert!(
            matches!(&err, LaunchError::Spawn(why) if why.contains("exit code 2")),
            "{err:?}"
        );
        assert_eq!(fake.lines().last().map(String::as_str), Some(DELETE));
    }

    /// The secret is in the create body and in nothing that is printed: not
    /// in an error the daemon words with the request in it, not in a
    /// timeout's, not in the launcher's `Debug`.
    ///
    /// Mutation this catches: returning `start`'s error as it is from `spawn`
    /// (the daemon's message goes into the text), or putting the body into an
    /// error with `{body:?}`.
    #[tokio::test]
    async fn no_error_and_no_debug_prints_the_secret() {
        // A daemon that quotes the request back in its refusal.
        let fake = Fake::new(|_, _, body| {
            Reply::Json(
                400,
                json!({ "message": format!("invalid request: {body}") }),
            )
        });
        let docker = launcher(&fake);
        let err = docker.spawn(spec(1, 7_870, SECRET)).await.expect_err("no");
        let printed = format!("{err:?} {err}");
        assert!(printed.contains("400"), "{printed}");
        assert!(!printed.contains(SECRET), "{printed}");

        // The start fails the same way.
        let fake = Fake::new(|method, target, body| match (method, target) {
            ("POST", t) if t.ends_with("/start") => Reply::Json(
                500,
                json!({ "message": format!("SPACE_CLUSTER_SECRET={SECRET}") }),
            ),
            _ => obliging(method, target, body),
        });
        let docker = launcher(&fake);
        let err = docker.spawn(spec(2, 7_871, SECRET)).await.expect_err("no");
        let printed = format!("{err:?} {err}");
        assert!(printed.contains("<redacted>"), "{printed}");
        assert!(!printed.contains(SECRET), "{printed}");

        // No answer at all.
        let fake = Fake::new(|_, _, _| Reply::Hang);
        let mut docker = launcher(&fake);
        docker.timeouts.call = Duration::from_millis(30);
        let err = docker.spawn(spec(3, 7_872, SECRET)).await.expect_err("no");
        let printed = format!("{err:?} {err} {docker:?}");
        assert!(printed.contains("timed out"), "{printed}");
        assert!(!printed.contains(SECRET), "{printed}");
    }

    /// The inspection of whatever holds a name, with these labels.
    fn holder(labels: Value) -> Reply {
        Reply::Json(
            200,
            json!({ "Id": "feedfeedfeedfeed", "Config": { "Labels": labels } }),
        )
    }

    /// A name that is taken by a container that is not this orchestrator's
    /// shard — no labels, another cluster's, another role, **another stack's
    /// of the same cluster name**, one without an instance — fails the launch
    /// and **removes nothing**.
    ///
    /// Mutation this catches: removing whatever holds the name (a stranger's
    /// container that happens to be called `space-shard-7870` would be
    /// deleted by an orchestrator that holds the Docker socket), or checking
    /// only one or two of the three labels (`is_shard_of` without the
    /// instance: a second stack left at the default cluster name loses its
    /// running shard).
    #[tokio::test]
    async fn a_name_held_by_a_strangers_container_is_left_alone() {
        for labels in [
            json!({}),
            json!(null),
            json!({ "space.cluster": "another-cluster", "space.role": "shard", "space.instance": "inst-a" }),
            json!({ "space.cluster": "test-cluster", "space.role": "orchestrator", "space.instance": "inst-a" }),
            json!({ "space.role": "shard", "space.instance": "inst-a" }),
            json!({ "space.cluster": "test-cluster", "space.role": "shard", "space.instance": "inst-b" }),
            json!({ "space.cluster": "test-cluster", "space.role": "shard" }),
        ] {
            let fake = Fake::new(move |method, target, _| match (method, target) {
                ("POST", _) => {
                    Reply::Json(409, json!({ "message": "Conflict. The name is in use" }))
                }
                ("GET", "/containers/space-shard-7870/json") => holder(labels.clone()),
                _ => Reply::Status(204),
            });
            let docker = launcher(&fake);
            let err = docker.spawn(spec(1, 7_870, "")).await.expect_err("taken");
            assert!(
                matches!(&err, LaunchError::Spawn(why)
                    if why.contains("space-shard-7870") && why.contains("left alone")),
                "{err:?}"
            );
            assert_eq!(
                fake.lines(),
                [
                    "POST /containers/create?name=space-shard-7870",
                    "GET /containers/space-shard-7870/json"
                ],
                "nothing is removed and nothing is tried again"
            );
            assert!(docker.table().is_empty());
        }
    }

    /// A name that is taken by a leftover shard of this cluster is freed —
    /// that container, by its id, with force — and the creation tried again,
    /// once.
    ///
    /// Mutation this catches: failing on every 409 (a shard that was not
    /// removed in time would block its port for good), or retrying without
    /// end when the daemon keeps answering 409.
    #[tokio::test]
    async fn a_name_held_by_our_own_leftover_is_freed_and_tried_again() {
        let ours = || json!({ "space.cluster": "test-cluster", "space.role": "shard", "space.instance": "inst-a" });
        let creates = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&creates);
        let fake = Fake::new(move |method, target, body| match (method, target) {
            ("POST", t) if t.starts_with("/containers/create") => {
                if seen.fetch_add(1, Ordering::SeqCst) == 0 {
                    Reply::Json(409, json!({ "message": "Conflict. The name is in use" }))
                } else {
                    Reply::Json(201, json!({ "Id": ID }))
                }
            }
            ("GET", "/containers/space-shard-7870/json") => holder(ours()),
            _ => obliging(method, target, body),
        });
        let docker = launcher(&fake);
        let handle = docker.spawn(spec(1, 7_870, "")).await.expect("starts");
        assert_eq!(handle.endpoint, "172.20.0.5:7870".parse().unwrap());
        assert_eq!(
            fake.lines(),
            [
                "POST /containers/create?name=space-shard-7870",
                "GET /containers/space-shard-7870/json",
                "DELETE /containers/feedfeedfeedfeed?force=true&v=true",
                "POST /containers/create?name=space-shard-7870",
                START,
                INSPECT,
            ]
        );
        let calls = fake.calls();
        assert_eq!(calls[0].2, calls[3].2, "the same body both times");

        // A daemon that says 409 again: one more try, not a loop.
        let fake = Fake::new(move |method, target, _| match (method, target) {
            ("POST", _) => Reply::Json(409, json!({ "message": "Conflict" })),
            ("GET", _) => holder(ours()),
            _ => Reply::Status(204),
        });
        let docker = launcher(&fake);
        let err = docker.spawn(spec(1, 7_870, "")).await.expect_err("taken");
        assert!(
            matches!(&err, LaunchError::Spawn(why) if why.contains("409")),
            "{err:?}"
        );
        assert_eq!(fake.calls().len(), 4);
    }

    /// `kill` removes the container with force and its volumes and forgets
    /// the launch; a container that is gone already is removed; a launch
    /// nobody knows costs no call. One the daemon would not remove is kept
    /// for `shutdown`, and is dead to `poll`.
    ///
    /// Mutation this catches: a `DELETE` without `force=true` (the daemon
    /// refuses to remove a running container: every stopped shard would stay),
    /// or taking a 404 for a failure.
    #[tokio::test]
    async fn kill_removes_with_force_and_a_missing_container_is_removed() {
        let fake = Fake::new(obliging);
        let docker = launcher(&fake);
        docker.kill(LaunchId(9)).await.expect("unknown: a no-op");
        assert!(fake.calls().is_empty());
        docker.spawn(spec(1, 7_870, "")).await.expect("starts");
        docker.kill(LaunchId(1)).await.expect("removed");
        assert_eq!(fake.lines().last().map(String::as_str), Some(DELETE));
        assert_eq!(docker.poll(LaunchId(1)).await.unwrap(), Poll::Dead);
        docker.kill(LaunchId(1)).await.expect("again: a no-op");
        assert_eq!(fake.calls().len(), 4, "forgotten: no further call");

        let fake = Fake::new(|method, target, body| match method {
            "DELETE" => Reply::Json(404, json!({ "message": "No such container" })),
            _ => obliging(method, target, body),
        });
        let docker = launcher(&fake);
        docker.spawn(spec(1, 7_870, "")).await.expect("starts");
        docker.kill(LaunchId(1)).await.expect("gone is removed");
        assert!(docker.table().is_empty());

        let fake = Fake::new(|method, target, body| match method {
            "DELETE" => Reply::Json(500, json!({ "message": "cannot remove" })),
            _ => obliging(method, target, body),
        });
        let docker = launcher(&fake);
        docker.spawn(spec(1, 7_870, "")).await.expect("starts");
        let err = docker.kill(LaunchId(1)).await.expect_err("not removed");
        assert!(matches!(&err, LaunchError::Spawn(why) if why.contains("cannot remove")));
        assert_eq!(docker.poll(LaunchId(1)).await.unwrap(), Poll::Dead);
        docker.shutdown().await;
        assert_eq!(
            fake.lines().iter().filter(|line| *line == DELETE).count(),
            2,
            "shutdown tried again"
        );
    }

    /// A launch that is stopped while its container is being created does not
    /// leave that container behind: `kill` has nothing to remove yet, so
    /// `spawn` removes what it then gets, and fails.
    ///
    /// Mutation this catches: entering a launch into the table only when
    /// `spawn` is done — `kill` finds nothing, `spawn` succeeds, and a shard
    /// nobody supervises runs.
    #[tokio::test]
    async fn a_launch_stopped_while_it_is_created_leaves_no_container() {
        let gate = Arc::new(Notify::new());
        let held = Arc::clone(&gate);
        let fake = Fake::new(move |method, target, body| match (method, target) {
            ("POST", t) if t.starts_with("/containers/create") => Reply::Gated(
                Arc::clone(&held),
                Box::new(Reply::Json(201, json!({ "Id": ID }))),
            ),
            _ => obliging(method, target, body),
        });
        let docker = Arc::new(launcher(&fake));
        let spawning = {
            let docker = Arc::clone(&docker);
            tokio::spawn(async move { docker.spawn(spec(1, 7_870, "")).await })
        };
        while fake.calls().is_empty() {
            tokio::task::yield_now().await;
        }
        assert_eq!(docker.poll(LaunchId(1)).await.unwrap(), Poll::Alive);
        docker
            .kill(LaunchId(1))
            .await
            .expect("nothing to remove yet");
        assert_eq!(fake.calls().len(), 1);
        gate.notify_one();
        let err = spawning.await.unwrap().expect_err("it was stopped");
        assert!(
            matches!(&err, LaunchError::Spawn(why) if why.contains("was stopped while")),
            "{err:?}"
        );
        assert_eq!(
            fake.lines(),
            ["POST /containers/create?name=space-shard-7870", DELETE]
        );
        assert!(docker.table().is_empty());
    }

    /// `poll`: a running container is alive; one that has stopped is dead and
    /// its exit code is known from then on, until `kill`; one that is gone is
    /// dead with no code; a launch nobody knows is dead. **A daemon that does
    /// not answer — closed connection, error status, nonsense — is an error,
    /// not a dead shard**, and the next poll asks again.
    ///
    /// Mutation this catches: mapping every failed inspection to `Dead` (one
    /// hiccup of the daemon and the orchestrator replaces every shard), or
    /// reading `ExitCode` without `Running` (a running container's is 0).
    #[tokio::test]
    async fn poll_tells_running_from_exited_from_no_answer() {
        let state = Arc::new(AtomicUsize::new(0));
        let now = Arc::clone(&state);
        let fake = Fake::new(move |method, target, body| {
            let polled = method == "GET" && target == &INSPECT[4..];
            match (polled, now.load(Ordering::SeqCst)) {
                (true, 1) => Reply::Close,
                (true, 2) => Reply::Json(500, json!({ "message": "daemon is busy" })),
                (true, 3) => Reply::Json(200, json!({ "State": {} })),
                (true, 4) => Reply::Json(
                    200,
                    json!({ "State": { "Status": "exited", "Running": false, "ExitCode": 137 } }),
                ),
                _ => obliging(method, target, body),
            }
        });
        let docker = launcher(&fake);
        assert_eq!(docker.poll(LaunchId(1)).await.unwrap(), Poll::Dead);
        assert!(fake.calls().is_empty(), "unknown: no call");
        docker.spawn(spec(1, 7_870, "")).await.expect("starts");
        assert_eq!(docker.poll(LaunchId(1)).await.unwrap(), Poll::Alive);
        assert_eq!(docker.exit_code(LaunchId(1)).await, None);

        for no_answer in 1..=3 {
            state.store(no_answer, Ordering::SeqCst);
            let polled = docker.poll(LaunchId(1)).await;
            assert!(
                matches!(polled, Err(LaunchError::Spawn(_))),
                "{no_answer}: {polled:?}"
            );
            assert_eq!(docker.exit_code(LaunchId(1)).await, None);
        }
        state.store(0, Ordering::SeqCst);
        assert_eq!(
            docker.poll(LaunchId(1)).await.unwrap(),
            Poll::Alive,
            "asked again, and it runs"
        );

        state.store(4, Ordering::SeqCst);
        assert_eq!(docker.poll(LaunchId(1)).await.unwrap(), Poll::Dead);
        assert_eq!(docker.exit_code(LaunchId(1)).await, Some(137));
        let asked = fake.calls().len();
        assert_eq!(docker.poll(LaunchId(1)).await.unwrap(), Poll::Dead);
        assert_eq!(fake.calls().len(), asked, "seen to have exited: not asked");
        docker.kill(LaunchId(1)).await.expect("removed");
        assert_eq!(docker.exit_code(LaunchId(1)).await, None, "kept until kill");

        // Removed behind the launcher's back: dead, and no code.
        let fake = Fake::new(|method, target, body| {
            if method == "GET" && target == &INSPECT[4..] {
                Reply::Json(404, json!({ "message": "No such container" }))
            } else {
                obliging(method, target, body)
            }
        });
        let docker = launcher(&fake);
        docker.table().insert(
            LaunchId(2),
            Entry {
                id: ID.into(),
                name: "space-shard-7871".into(),
                port: 7_871,
                started: true,
                exited: false,
                killed: false,
                exit_code: None,
            },
        );
        assert_eq!(docker.poll(LaunchId(2)).await.unwrap(), Poll::Dead);
        assert_eq!(docker.exit_code(LaunchId(2)).await, None);
    }

    /// The daemon `prepare` talks to: the orchestrator's own container, a
    /// list of leftovers, an image that is there or not.
    fn preparing(
        networks: Value,
        image_there: Arc<AtomicUsize>,
        pull: &'static str,
    ) -> impl Fn(&str, &str, &str) -> Reply + Send + Sync + 'static {
        move |method, target, _| match (method, target) {
            ("GET", "/_ping") => {
                Reply::Raw(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK".to_vec())
            }
            ("GET", "/containers/orch-1/json") => Reply::Json(
                200,
                json!({ "Name": "/space-orchestrator-1", "NetworkSettings": { "Networks": networks } }),
            ),
            ("GET", t) if t.starts_with("/containers/json?") => Reply::Json(
                200,
                json!([
                    { "Id": "aaaa", "Labels": { "space.cluster": "test-cluster", "space.role": "shard", "space.instance": "inst-a" } },
                    { "Id": "bbbb", "Labels": { "space.cluster": "another", "space.role": "shard", "space.instance": "inst-a" } },
                    { "Id": "cccc", "Labels": {} },
                    { "Id": "dddd", "Labels": { "space.cluster": "test-cluster", "space.role": "shard", "space.instance": "inst-a", "space.port": "7871" } },
                    { "Id": "eeee", "Labels": { "space.cluster": "test-cluster", "space.role": "orchestrator", "space.instance": "inst-a" } },
                    // Another stack on this host that uses the same cluster
                    // name, and a shard from before there were instances.
                    { "Id": "ffff", "Labels": { "space.cluster": "test-cluster", "space.role": "shard", "space.instance": "inst-b" } },
                    { "Id": "9999", "Labels": { "space.cluster": "test-cluster", "space.role": "shard" } },
                ]),
            ),
            ("GET", t) if t.starts_with("/images/") => {
                if image_there.load(Ordering::SeqCst) > 0 {
                    Reply::Json(200, json!({ "Id": "sha256:1234" }))
                } else {
                    Reply::Json(404, json!({ "message": "No such image" }))
                }
            }
            ("POST", t) if t.starts_with("/images/create") => {
                if !pull.contains("\"error\"") {
                    image_there.store(1, Ordering::SeqCst);
                }
                // The progress stream as the daemon sends it: chunked, a
                // JSON object a line.
                let mut bytes = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
                for line in pull.split_inclusive('\n') {
                    bytes.extend_from_slice(format!("{:x}\r\n{line}\r\n", line.len()).as_bytes());
                }
                bytes.extend_from_slice(b"0\r\n\r\n");
                Reply::Raw(bytes)
            }
            ("DELETE", _) => Reply::Status(204),
            _ => Reply::Json(500, json!({ "message": "the test did not expect this" })),
        }
    }

    fn unresolved() -> DockerOptions {
        DockerOptions {
            network: None,
            control: None,
            control_port: 7_900,
            self_container: Some("orch-1".into()),
            ..options()
        }
    }

    const IMAGE: &str = "GET /images/ghcr.io/eliasstepanik/space-shard:sha-abc1234/json";

    /// `prepare` with nothing configured: the network is the one the
    /// orchestrator's own container is on, the control address its own
    /// address there; of the containers the daemon lists, exactly those
    /// labelled as **this orchestrator's** shards (cluster, role and
    /// instance) are removed — also when the daemon lists others, among them
    /// the shards of another stack with the same cluster name; an image that
    /// is there is not pulled.
    ///
    /// Mutation this catches: removing everything the list returns (the
    /// filter is the daemon's; what is deleted must be decided here), listing
    /// without the label filter, telling "ours" by the cluster's name alone
    /// (`is_shard_of` without `LABEL_INSTANCE`: `ffff`, a running shard of
    /// the stack next door, is deleted), or pulling an image that is present
    /// (every start would wait for a registry).
    #[tokio::test]
    async fn prepare_removes_only_labelled_leftovers_and_pulls_nothing_that_is_there() {
        let networks = json!({ "space_default": { "IPAddress": "172.20.0.2" } });
        let fake = Fake::new(preparing(networks, Arc::new(AtomicUsize::new(1)), ""));
        let docker = DockerLauncher::with_transport(unresolved(), fake.clone());
        let said = docker.prepare().await.expect("ready");
        let filters = "%7B%22label%22%3A%5B%22space.cluster%3Dtest-cluster%22%2C%22space.role%3Dshard%22%2C%22space.instance%3Dinst-a%22%5D%7D";
        assert_eq!(
            fake.lines(),
            [
                "GET /_ping".to_owned(),
                "GET /containers/orch-1/json".to_owned(),
                format!("GET /containers/json?all=true&filters={filters}"),
                "DELETE /containers/aaaa?force=true&v=true".to_owned(),
                "DELETE /containers/dddd?force=true&v=true".to_owned(),
                IMAGE.to_owned(),
            ]
        );
        assert!(!said.contains('\n'), "{said:?}");
        for part in [
            "\"space_default\"",
            "dial 172.20.0.2:7900",
            "is on the host",
            "2 leftover shard containers",
            "(instance \"inst-a\")",
        ] {
            assert!(said.contains(part), "{part}: {said}");
        }
        // What was found is what a shard is started with.
        assert_eq!(
            docker.resolved().await.unwrap(),
            Resolved {
                network: "space_default".into(),
                control: "172.20.0.2:7900".into()
            }
        );
        assert_eq!(fake.calls().len(), 6, "found once");
    }

    /// An image that is not on the host is pulled — repository and tag apart,
    /// without credentials — its progress read to the end, and looked for
    /// again. A pull whose stream carries an error line fails, and the error
    /// says what the operator has to do about a private image.
    ///
    /// Mutation this catches: taking the pull's `200` for success (the daemon
    /// reports a refused pull **inside** the stream), or reading only the
    /// stream's first line.
    #[tokio::test]
    async fn prepare_pulls_a_missing_image_and_a_failed_pull_is_an_error() {
        let pull = "{\"status\":\"Pulling from eliasstepanik/space-shard\"}\n\
                    {\"status\":\"Downloading\",\"progressDetail\":{\"current\":1,\"total\":2}}\n\
                    {\"status\":\"Status: Downloaded newer image\"}\n";
        let fake = Fake::piped(9, preparing(json!({}), Arc::new(AtomicUsize::new(0)), pull));
        let docker = launcher(&fake);
        let said = docker.prepare().await.expect("pulled");
        assert!(said.contains("was pulled"), "{said}");
        let lines = fake.lines();
        let pulling =
            "POST /images/create?fromImage=ghcr.io%2Feliasstepanik%2Fspace-shard&tag=sha-abc1234";
        assert_eq!(&lines[lines.len() - 3..], [IMAGE, pulling, IMAGE]);
        for head in fake.0.heads.lock().unwrap().iter() {
            assert!(!head.to_ascii_lowercase().contains("x-registry-auth"));
        }

        let refused = "{\"status\":\"Pulling from eliasstepanik/space-shard\"}\n\
                       {\"errorDetail\":{\"message\":\"denied\"},\"error\":\"pull access denied for space-shard\"}\n";
        let fake = Fake::new(preparing(json!({}), Arc::new(AtomicUsize::new(0)), refused));
        let docker = launcher(&fake);
        let err = docker.prepare().await.expect_err("not pullable");
        let LaunchError::Unavailable(why) = &err else {
            panic!("{err:?}");
        };
        for part in ["pull access denied", "docker login", "docker pull ghcr.io/"] {
            assert!(why.contains(part), "{part}: {why}");
        }
        assert_eq!(fake.lines().last().map(String::as_str), Some(pulling));
    }

    /// What `prepare` refuses to guess: which of several networks the shards
    /// join, and where they go when the orchestrator is in no container. A
    /// daemon that is not there makes the launcher unavailable.
    ///
    /// Mutation this catches: taking the first of several networks (map order
    /// would decide whether the shards can reach the orchestrator).
    #[tokio::test]
    async fn prepare_does_not_guess_a_network() {
        let networks = json!({
            "edge": { "IPAddress": "10.1.0.2" },
            "space_default": { "IPAddress": "172.20.0.2" },
        });
        let there = || Arc::new(AtomicUsize::new(1));
        let fake = Fake::new(preparing(networks.clone(), there(), ""));
        let docker = DockerLauncher::with_transport(unresolved(), fake.clone());
        let err = docker.prepare().await.expect_err("several");
        assert!(
            matches!(&err, LaunchError::Unavailable(why)
                if why.contains("several networks (edge, space_default)")
                    && why.contains("--docker-network")),
            "{err:?}"
        );
        assert_eq!(fake.calls().len(), 2, "nothing is removed before that");

        // Told which: the control address is the orchestrator's on that one.
        let chosen = DockerOptions {
            network: Some("edge".into()),
            ..unresolved()
        };
        let fake = Fake::new(preparing(networks, there(), ""));
        let docker = DockerLauncher::with_transport(chosen, fake.clone());
        let said = docker.prepare().await.expect("ready");
        assert!(said.contains("dial 10.1.0.2:7900"), "{said}");

        // Not in a container, and not told.
        let nowhere = DockerOptions {
            self_container: None,
            self_hostname: None,
            ..unresolved()
        };
        let fake = Fake::new(preparing(json!({}), there(), ""));
        let docker = DockerLauncher::with_transport(nowhere, fake.clone());
        let err = docker.prepare().await.expect_err("nowhere");
        assert!(
            matches!(&err, LaunchError::Unavailable(why)
                if why.contains("--docker-network") && why.contains("--docker-control-addr")),
            "{err:?}"
        );

        // No daemon.
        let fake = Fake::new(|_, _, _| Reply::Close);
        let err = launcher(&fake).prepare().await.expect_err("no daemon");
        assert!(
            matches!(&err, LaunchError::Unavailable(why) if why.contains("does not answer")),
            "{err:?}"
        );
    }

    /// The orchestrator finds its own container by the **name it was told**
    /// and only then by `$HOSTNAME`; a name the daemon does not know is
    /// skipped, not fatal. With nobody found it says what to set - and with
    /// network and control address both given it asks nobody.
    ///
    /// Mutation this catches: asking for `$HOSTNAME` alone, or treating its
    /// 404 as the end (a container recreated from an older one's
    /// configuration - Watchtower - carries the *old* container's short id as
    /// its host name: the orchestrator would not start after every update).
    #[tokio::test]
    async fn the_orchestrator_finds_itself_by_name_before_its_host_name() {
        let networks = || json!({ "space_default": { "IPAddress": "172.20.0.2" } });
        let there = || Arc::new(AtomicUsize::new(1));
        let wanted = Resolved {
            network: "space_default".into(),
            control: "172.20.0.2:7900".into(),
        };
        // (`preparing` knows the container `orch-1` and no other.)
        let stale = Some("0ld1d0ld1d00".to_owned());

        // Told its name, with a stale host name: the name is asked, and only it.
        let named = DockerOptions {
            self_container: Some("orch-1".into()),
            self_hostname: stale.clone(),
            ..unresolved()
        };
        let fake = Fake::new(preparing(networks(), there(), ""));
        let docker = DockerLauncher::with_transport(named, fake.clone());
        assert_eq!(docker.resolved().await.unwrap(), wanted);
        assert_eq!(fake.lines(), ["GET /containers/orch-1/json"]);

        // A name the daemon does not know: the host name is asked next.
        let misnamed = DockerOptions {
            self_container: Some("not-my-name".into()),
            self_hostname: Some("orch-1".into()),
            ..unresolved()
        };
        let fake = Fake::new(|method, target, body| match target {
            "/containers/not-my-name/json" => {
                Reply::Json(404, json!({ "message": "No such container: not-my-name" }))
            }
            _ => preparing(
                json!({ "space_default": { "IPAddress": "172.20.0.2" } }),
                Arc::new(AtomicUsize::new(1)),
                "",
            )(method, target, body),
        });
        let docker = DockerLauncher::with_transport(misnamed, fake.clone());
        assert_eq!(docker.resolved().await.unwrap(), wanted);
        assert_eq!(
            fake.lines(),
            [
                "GET /containers/not-my-name/json",
                "GET /containers/orch-1/json"
            ]
        );

        // Only a stale host name, and nothing configured: unavailable, and
        // the message says what to set.
        let not_found = |method: &str, target: &str, _: &str| match (method, target) {
            ("GET", t) if t.starts_with("/containers/") => {
                Reply::Json(404, json!({ "message": "No such container" }))
            }
            _ => Reply::Json(500, json!({ "message": "the test did not expect this" })),
        };
        let lost = DockerOptions {
            self_container: None,
            self_hostname: stale.clone(),
            ..unresolved()
        };
        let fake = Fake::new(not_found);
        let docker = DockerLauncher::with_transport(lost.clone(), fake.clone());
        let err = docker.resolved().await.expect_err("nobody found");
        let LaunchError::Unavailable(why) = &err else {
            panic!("{err:?}");
        };
        for part in [
            "\"0ld1d0ld1d00\"",
            "SPACE_ORCHESTRATOR_DOCKER_SELF",
            "--docker-network and --docker-control-addr",
        ] {
            assert!(why.contains(part), "{part}: {why}");
        }
        // One of the two given: the other one is what is missing.
        let half = DockerOptions {
            network: Some("space_default".into()),
            ..lost.clone()
        };
        let docker = DockerLauncher::with_transport(half, Fake::new(not_found));
        let err = docker.resolved().await.expect_err("nobody found");
        assert!(
            matches!(&err, LaunchError::Unavailable(why)
                if why.contains("or pass --docker-control-addr")),
            "{err:?}"
        );
        // Both given: nobody is asked, whatever the names are.
        let told = DockerOptions {
            network: Some("space_default".into()),
            control: Some("orchestrator:7900".into()),
            ..lost
        };
        let fake = Fake::new(not_found);
        let docker = DockerLauncher::with_transport(told, fake.clone());
        assert_eq!(
            docker.resolved().await.unwrap(),
            Resolved {
                network: "space_default".into(),
                control: "orchestrator:7900".into()
            }
        );
        assert!(fake.calls().is_empty());
    }

    /// One answer through a pipe of `pipe` bytes.
    async fn answer(
        pipe: usize,
        response: Vec<u8>,
        limit: usize,
    ) -> Result<(u16, Vec<u8>), String> {
        let (client, mut server) = tokio::io::duplex(pipe);
        tokio::spawn(async move {
            let mut request = Vec::new();
            let mut piece = [0u8; 64];
            while !request.ends_with(b"\r\n\r\n") {
                match server.read(&mut piece).await {
                    Ok(0) | Err(_) => return,
                    Ok(got) => request.extend_from_slice(&piece[..got]),
                }
            }
            let _ = server.write_all(&response).await;
        });
        let mut sink = Sink::collect(limit);
        let request = request_bytes("GET", "/x", None);
        let status = exchange(client, &request, &mut sink).await?;
        let Sink::Collect { body, .. } = sink else {
            unreachable!()
        };
        Ok((status, body))
    }

    /// A chunked body is put together from its chunks: sizes in hexadecimal
    /// (either case), a chunk extension ignored, line ends inside the data
    /// left alone, the trailer read to its end — whatever the size of the
    /// reads it arrives in. A chunk that does not end where it says is an
    /// error.
    ///
    /// Mutation this catches: parsing the whole size line as the size (an
    /// extension breaks it), or reading sizes as decimal (`1B`). (Not
    /// caught: stopping at the `0` without reading the trailer — on a
    /// connection that carries one answer nothing comes after it.)
    #[tokio::test]
    async fn a_chunked_body_is_put_together_across_chunks_and_reads() {
        let chunked = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
            Transfer-Encoding: chunked\r\n\r\n\
            4;name=value\r\n[{\"I\r\n\
            1B\r\nd\":\"abc\",\"Text\":\"one\\r\\ntwo\r\n\
            2\r\n\r\n\r\n\
            3\r\n\"}]\r\n\
            0\r\nExpires: never\r\nX-More: yes\r\n\r\n";
        let whole = b"[{\"Id\":\"abc\",\"Text\":\"one\\r\\ntwo\r\n\"}]";
        for pipe in [1, 2, 3, 5, 7, 64, 4096] {
            let (status, body) = answer(pipe, chunked.to_vec(), MAX_BODY)
                .await
                .unwrap_or_else(|why| panic!("pipe {pipe}: {why}"));
            assert_eq!(status, 200);
            assert_eq!(
                String::from_utf8_lossy(&body),
                String::from_utf8_lossy(whole),
                "pipe {pipe}"
            );
        }
        for (broken, why) in [
            ("5\r\nabc\r\n0\r\n\r\n", "does not end where it says"),
            ("2\r\nabc\r\n0\r\n\r\n", "does not end where it says"),
            ("zz\r\nabc\r\n0\r\n\r\n", "invalid chunk size"),
            ("\r\nabc\r\n0\r\n\r\n", "invalid chunk size"),
            ("3\r\nabc\r\n0\r\n", "the daemon closed"),
            ("3\r\nabc", "the daemon closed"),
        ] {
            let mut response = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
            response.extend_from_slice(broken.as_bytes());
            let err = answer(16, response, MAX_BODY).await.expect_err("broken");
            assert!(err.contains(why), "{broken:?}: {err}");
        }

        // The other two framings, and the answers that have no body.
        let counted = b"HTTP/1.1 201 Created\r\nContent-Length: 5\r\n\r\nhellothis is not read";
        assert_eq!(
            answer(3, counted.to_vec(), MAX_BODY).await.unwrap(),
            (201, b"hello".to_vec())
        );
        let to_end = b"HTTP/1.0 200 OK\r\n\r\nall of it";
        assert_eq!(
            answer(3, to_end.to_vec(), MAX_BODY).await.unwrap(),
            (200, b"all of it".to_vec())
        );
        let nothing = b"HTTP/1.1 204 No Content\r\n\r\n";
        assert_eq!(
            answer(3, nothing.to_vec(), MAX_BODY).await.unwrap(),
            (204, Vec::new())
        );
        let short = b"HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\nhello";
        assert!(answer(3, short.to_vec(), MAX_BODY).await.is_err());
        let not_http = b"SSH-2.0-OpenSSH_9.6\r\n\r\n";
        let err = answer(64, not_http.to_vec(), MAX_BODY)
            .await
            .expect_err("not HTTP");
        assert!(err.contains("does not start like HTTP"), "{err}");
    }

    /// The limits hold before anything is buffered: a head of more than
    /// `MAX_HEAD` is refused although a complete answer follows it, a counted
    /// body over the limit is refused by its announcement, a chunked or
    /// uncounted one when it gets there.
    ///
    /// Mutation this catches: reading the head until its empty line however
    /// long that takes, or collecting a body without a bound (a daemon — or
    /// whatever answers on that socket — could make the orchestrator hold any
    /// amount of memory).
    #[tokio::test]
    async fn an_answer_over_the_limits_is_refused() {
        let mut long_head = b"HTTP/1.1 200 OK\r\n".to_vec();
        while long_head.len() <= MAX_HEAD {
            long_head.extend_from_slice(b"X-Padding: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\r\n");
        }
        long_head.extend_from_slice(b"Content-Length: 2\r\n\r\n{}");
        let err = answer(4096, long_head, MAX_BODY)
            .await
            .expect_err("too long");
        assert!(err.contains("head is over 16384 bytes"), "{err}");

        // Just under it: fine.
        let mut head = b"HTTP/1.1 200 OK\r\nX: ".to_vec();
        let tail = b"\r\nContent-Length: 2";
        head.resize(MAX_HEAD - tail.len(), b'a');
        head.extend_from_slice(tail);
        assert_eq!(head.len(), MAX_HEAD);
        head.extend_from_slice(b"\r\n\r\n{}");
        assert_eq!(
            answer(4096, head, MAX_BODY).await.unwrap(),
            (200, b"{}".to_vec())
        );

        let counted = b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\n".to_vec();
        let err = answer(64, counted, 10)
            .await
            .expect_err("announced too long");
        assert!(err.contains("body is over 10 bytes"), "{err}");
        let chunked = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n6\r\nabcdef\r\n6\r\nabcdef\r\n0\r\n\r\n";
        let err = answer(64, chunked.to_vec(), 10)
            .await
            .expect_err("too long");
        assert!(err.contains("body is over 10 bytes"), "{err}");
        let to_end = b"HTTP/1.1 200 OK\r\n\r\nabcdefabcdef";
        let err = answer(64, to_end.to_vec(), 10).await.expect_err("too long");
        assert!(err.contains("body is over 10 bytes"), "{err}");
    }

    /// A daemon that takes the request and never answers does not hold the
    /// caller: the call ends at its time limit, with an error that names the
    /// call. (The limits are shortened here: the workspace's tokio has no
    /// paused clock.)
    ///
    /// Mutation this catches: a call without `tokio::time::timeout` — this
    /// test would never end.
    #[tokio::test]
    async fn a_call_that_is_never_answered_times_out() {
        let fake = Fake::new(|_, _, _| Reply::Hang);
        let mut docker = launcher(&fake);
        docker.timeouts.call = Duration::from_millis(40);
        docker.timeouts.remove = Duration::from_millis(40);
        let started = std::time::Instant::now();
        let err = docker
            .spawn(spec(1, 7_870, ""))
            .await
            .expect_err("no answer");
        assert_eq!(
            err,
            LaunchError::Spawn(
                "docker: creating container space-shard-7870 timed out after 40ms".into()
            )
        );
        assert!(docker.table().is_empty());
        let err = docker.prepare().await.expect_err("no answer");
        assert!(
            matches!(&err, LaunchError::Unavailable(why) if why.contains("timed out after 40ms")),
            "{err:?}"
        );
        assert!(started.elapsed() < Duration::from_secs(5));
        // What ships: every limit is set.
        let shipped = launcher(&fake).timeouts;
        assert_eq!(shipped.call, Duration::from_secs(10));
        assert_eq!(shipped.remove, Duration::from_secs(15));
        assert_eq!(shipped.pull, Duration::from_secs(300));
        assert_eq!(shipped.shutdown, Duration::from_secs(20));
    }

    /// `shutdown` removes every container of the table and empties it, and a
    /// daemon that hangs on one removal holds up neither the others nor the
    /// caller beyond the one bound.
    ///
    /// Mutation this catches: removing one after the other (the containers
    /// behind the one that hangs are never asked for), or no bound around the
    /// whole (the orchestrator's exit waits for every call's own limit).
    #[tokio::test]
    async fn shutdown_removes_everything_within_one_bound() {
        let fake = Fake::new(|method, target, _| match method {
            "DELETE" if target.contains("hangs") => Reply::Hang,
            _ => Reply::Status(204),
        });
        let mut docker = launcher(&fake);
        // Each call's own limit is far away: only the one bound ends this.
        docker.timeouts.remove = Duration::from_secs(600);
        docker.timeouts.shutdown = Duration::from_millis(150);
        for (launch, id) in [(1, "aaaa"), (2, "hangs"), (3, "cccc"), (4, "dddd"), (5, "")] {
            docker.table().insert(
                LaunchId(launch),
                Entry {
                    id: id.into(),
                    name: format!("space-shard-{}", 7_869 + launch),
                    port: 7_870,
                    started: true,
                    exited: launch == 3,
                    killed: false,
                    exit_code: None,
                },
            );
        }
        let started = std::time::Instant::now();
        docker.shutdown().await;
        assert!(started.elapsed() < Duration::from_secs(10));
        let mut removed = fake.lines();
        removed.sort();
        assert_eq!(
            removed,
            [
                "DELETE /containers/aaaa?force=true&v=true",
                "DELETE /containers/cccc?force=true&v=true",
                "DELETE /containers/dddd?force=true&v=true",
                "DELETE /containers/hangs?force=true&v=true",
            ]
        );
        assert!(docker.table().is_empty());
        for launch in 1..=5 {
            assert_eq!(docker.poll(LaunchId(launch)).await.unwrap(), Poll::Dead);
        }
        // Nothing left: nothing to do, and no waiting.
        docker.shutdown().await;
        assert_eq!(fake.calls().len(), 4);
    }

    /// What goes into a request line: a query value with everything reserved
    /// encoded, a path segment likewise, an image's name with its own slashes
    /// kept; and how an image reference comes apart for a pull.
    ///
    /// Mutation this catches: putting a name or the JSON filter into the
    /// target as it is (a space, `"` or `&` in it would end the target or
    /// start another parameter), or splitting an image at its first `:` (a
    /// registry's port).
    #[test]
    fn names_are_encoded_and_image_references_split() {
        assert_eq!(
            query_value("a b&c=d/e:f\"{}"),
            "a%20b%26c%3Dd%2Fe%3Af%22%7B%7D"
        );
        assert_eq!(
            query_value("space-shard-7870_x.y~"),
            "space-shard-7870_x.y~"
        );
        assert_eq!(segment("../x?y#z é"), "..%2Fx%3Fy%23z%20%C3%A9");
        assert_eq!(
            image_path("localhost:5000/team/space shard:dev"),
            "localhost:5000/team/space%20shard:dev"
        );
        assert_eq!(split_image("space/shard:dev"), ("space/shard", "dev"));
        assert_eq!(split_image("space/shard"), ("space/shard", "latest"));
        assert_eq!(
            split_image("localhost:5000/space/shard"),
            ("localhost:5000/space/shard", "latest")
        );
        assert_eq!(
            split_image("localhost:5000/space/shard:v1"),
            ("localhost:5000/space/shard", "v1")
        );
        assert_eq!(
            split_image("ghcr.io/x/shard@sha256:abcd"),
            ("ghcr.io/x/shard", "sha256:abcd")
        );
        assert_eq!(short(ID), "0123456789ab");
        assert_eq!(short("abc"), "abc");
        // The daemon's words in an error: one line, and not all of them.
        let long = format!("first\nsecond {}", "x".repeat(400));
        let clipped = clip(&long);
        assert!(clipped.starts_with("first second xxx") && clipped.ends_with("..."));
        assert_eq!(clipped.chars().count(), MAX_ERROR_TEXT + 3);
    }

    /// The stub's defaults, which `main` builds on.
    #[test]
    fn the_default_options() {
        let options = DockerOptions::default();
        assert_eq!(options.socket, PathBuf::from("/var/run/docker.sock"));
        assert_eq!(options.name_prefix, "space-shard-");
        assert_eq!(options.control_port, 7_900);
        assert_eq!(
            (options.network, options.control, options.self_container),
            (None, None, None)
        );
        assert_eq!(options.self_hostname, None);
        assert_eq!(options.memory_mib, 2_048);
        assert!(!options.instance.is_empty());
    }

    /// Where there is no unix socket the launcher says so, in words an
    /// operator can act on, as "unavailable" — from the cheap check and from
    /// `prepare` alike — and no call is something else.
    ///
    /// Mutation this catches: reporting the platform as an ordinary failed
    /// call ("cannot reach the daemon"): the operator would go looking for a
    /// daemon to start.
    #[cfg(not(unix))]
    #[tokio::test]
    async fn without_a_unix_socket_the_launcher_says_what_to_use_instead() {
        let transport = SocketTransport::new(DEFAULT_SOCKET);
        let err = transport.connect().await.expect_err("no unix socket here");
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
        let docker = DockerLauncher::new(DockerOptions::default());
        let want = LaunchError::Unavailable(UNSUPPORTED_PLATFORM.to_owned());
        assert_eq!(docker.ensure_available().unwrap_err(), want);
        assert_eq!(docker.prepare().await.unwrap_err(), want);
        let said = want.to_string();
        assert!(!said.contains('\n'), "{said:?}");
        for part in ["unix socket", "Linux only", "--launcher process"] {
            assert!(said.contains(part), "{part}: {said}");
        }
        // A launch (configured, so nothing has to be found first) fails the
        // same way.
        let docker = DockerLauncher::new(options());
        assert_eq!(docker.spawn(spec(1, 7_870, "")).await.unwrap_err(), want);
        assert!(docker.table().is_empty());
    }
}
