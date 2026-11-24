//! `space-orchestrator` — the cluster control plane.
//!
//! Allocates shard ids, decides when a region is too hot or too cold, publishes
//! every ownership change through the epoch protocol, and launches the shards
//! that carry the regions. The binary only turns text into an
//! [`OrchestratorConfig`] and runs the loop; every decision lives in the crate.
//!
//! ```text
//!   space-orchestrator --bind 127.0.0.1:7900 --cluster space-local \
//!                      --state space-orchestrator.id
//!   # in a container: bind the wildcard, tell shards the dialable address
//!   space-orchestrator --bind 0.0.0.0:7900 --control-addr orchestrator:7900
//!   # with the built-in web UI (off unless asked for)
//!   space-orchestrator --webui 127.0.0.1:7980
//!   # the scene a fresh cluster shows (default: demo-area)
//!   space-orchestrator --scene galaxy
//!   # the one address players type, and the shards kept running behind it
//!   space-orchestrator --player-bind 127.0.0.1:7777 --min-shards 4 --launcher process
//! ```
//!
//! There is no argument-parsing dependency (mirrors `space-server`): the grammar
//! is a handful of flags, matched by hand against [`std::env::args`].
//!
//! # The web UI
//!
//! `--webui <addr:port>` serves the operator dashboard
//! ([`space_orchestrator::webui`]). On a loopback address it needs nothing
//! else. On any other address the orchestrator refuses to start without a
//! token, which a browser then asks for (any user name, the token as the
//! password). The token is best given through the environment, so it does not
//! show in the process list:
//!
//! | flag | environment fallback | |
//! |---|---|---|
//! | `--webui <addr:port>` | `SPACE_ORCHESTRATOR_WEBUI` | where to listen; unset = no UI |
//! | `--webui-token <secret>` | `SPACE_ORCHESTRATOR_WEBUI_TOKEN` | the password |
//! | `--webui-token-file <path>` | `SPACE_ORCHESTRATOR_WEBUI_TOKEN_FILE` | read it from a file (a container secret) |
//! | `--webui-host <name>[:<port>]` (repeatable) | `SPACE_ORCHESTRATOR_WEBUI_HOSTS` (comma-separated) | extra `Host` names to answer to, e.g. a proxy's public name; with a port, only on that port. **Needs a token**, whatever the bind address |
//!
//! A flag wins over its environment variable; an empty variable counts as unset.
//!
//! "No token on loopback" holds only while nothing but this machine can reach
//! the port. A UI without a token therefore refuses a public `--webui-host`
//! at startup, answers only to a loopback `Host`, and refuses requests that
//! carry a proxy's headers — but a proxy that adds none and rewrites `Host` to
//! `127.0.0.1` cannot be told from a local browser: **give the UI a token
//! before putting any proxy in front of it** ([`space_orchestrator::webui`],
//! "Without a token").
//!
//! # The player port
//!
//! | flag | environment fallback | |
//! |---|---|---|
//! | `--player-bind <addr:port>` | `SPACE_ORCHESTRATOR_PLAYER_BIND` | where players connect; unset = no player port |
//!
//! The one address a player types ([`space_orchestrator::player`]). A client
//! that says hello there is answered with a `JoinRedirect` naming the shard
//! that owns the scene's spawn point and the map that says where players
//! reach it — or, while no connected shard owns the spawn point (the cluster
//! is starting, or that shard is being replaced), with "starting", and the
//! client tries again. The connection is then closed: game traffic flows
//! between the client and the shard, never through the orchestrator.
//!
//! It is the cluster's **public entry**, not a control surface, so unlike the
//! control port and the web UI it needs no secret on any bind. In exchange it
//! can do nothing: it keeps no state per player, reads nothing a peer sends
//! behind its hello, cannot register a shard and reaches no decision of the
//! orchestrator — it only ever reads the answer the control loop last
//! published. The control port (`--bind`) is for shards and tells a player
//! who dials it so.
//!
//! Where a shard's **players** dial it is part of the map: the client
//! endpoint its hello asks for (`space-server --client-endpoint`), else the
//! address it advertises. The `process` launcher gives its shards
//! `0.0.0.0:<host port>`, which a client reads as "the host I dialed, that
//! port" — so no public host name is configured anywhere. A shard may also
//! advertise `0.0.0.0:<port>` as its internal address (a container that does
//! not know its own): the orchestrator then records the IP its control
//! connection comes from.
//!
//! # Limits
//!
//! `--max-regions <n>` (default 64) is the most regions the map may hold; a
//! split that would exceed it is refused, for the autoscaler and the operator
//! alike.
//!
//! # Shards
//!
//! Every shard starts without a region and is given one when it registers:
//! the root first, then the layout cells around the scene, or the regions of
//! a shard that died. Who starts them is the question here.
//!
//! | flag | environment fallback | |
//! |---|---|---|
//! | `--min-shards <n>` | `SPACE_ORCHESTRATOR_MIN_SHARDS` | how many shards the orchestrator keeps running **by itself**; default 0: it starts none (they are started by hand, or by a split). 4 on the demo area = its four quadrants. At most 8 (the root's shard and one per layout cell), and not more than `--max-shards` or `--max-regions`: anything else is refused at startup |
//! | `--max-shards <n>` | `SPACE_ORCHESTRATOR_MAX_SHARDS` | the most shards the cluster may have (default 8); one more that registers is refused for good |
//! | `--launcher local\|process\|docker` | `SPACE_ORCHESTRATOR_LAUNCHER` | how a shard is started: `local` inside this process (the default), `process` as a `space-server` process, `docker` not yet |
//! | `--shard-exe <path>` | `SPACE_ORCHESTRATOR_SHARD_EXE` | the executable `process` starts; default: the `space-server` lying next to this executable |
//! | `--shard-log-dir <dir>` | `SPACE_ORCHESTRATOR_SHARD_LOG_DIR` | where `process` writes `shard-<port>.log` per shard and `launched-shards.pid`; default: the working directory |
//! | `--host-port-range <a>-<b>` | `SPACE_ORCHESTRATOR_HOST_PORTS` | the UDP ports `process` hands its shards, one each (default 7870-7899). **Keep it out of the operating system's dynamic port range**; the binary does not check that |
//!
//! A shard that dies — its process exits, or its control connection closes —
//! is replaced: its regions go to a new shard under a **new id**, and the
//! scene's revision is raised so that its bodies are built once, afresh.
//!
//! A shard the `process` launcher starts inherits this process's environment
//! **without** the variables named `SPACE_ORCHESTRATOR_*` (the web UI token
//! and the peer token among them) and `SPACE_HARNESS_TOKEN`.
//!
//! # Secrets
//!
//! | flag | environment fallback | |
//! |---|---|---|
//! | `--cluster-secret <secret>` | `SPACE_CLUSTER_SECRET` | what a shard's hello must carry |
//! | `--cluster-secret-file <path>` | `SPACE_CLUSTER_SECRET_FILE` | read it from a file (a container secret) |
//! | `--peer-token <secret>` | `SPACE_ORCHESTRATOR_PEER_TOKEN` | what shards present to each other (at least 16 characters); default: generated at start |
//! | `--peer-token-file <path>` | `SPACE_ORCHESTRATOR_PEER_TOKEN_FILE` | read it from a file (a container secret) |
//!
//! A secret given as a flag shows in the process list: prefer the variable
//! or the file.
//!
//! The control port follows the web UI's rule. On a loopback bind a hello
//! needs no secret (unless one is configured). On any other bind a hello
//! without the right secret is refused, and the orchestrator does not start
//! without one a shard could present: a configured one of at least 16
//! characters — the same variable on every shard started by hand — or, with
//! `--min-shards` 1 or more, one it generates and hands to the shards it
//! starts (in their environment, never on a command line). A hello that is
//! refused is told why and its connection closed; one that is welcomed may
//! from then on report and acknowledge for the shard it was welcomed as, and
//! for no other.
//!
//! The peer token is never optional for an orchestrated cluster: without
//! `--peer-token` one is generated. A configured one shorter than 16
//! characters (an empty one would switch peer authentication off without a
//! word) is refused at startup. Neither secret is printed, and both are held
//! in a type whose `Debug` does not show them.
//!
//! # The scene
//!
//! | flag | environment fallback | |
//! |---|---|---|
//! | `--scene <name>` | `SPACE_ORCHESTRATOR_SCENE` | the **initial** scene: `demo-area` (the default), `materials` or `galaxy` |
//!
//! The scene is the cluster's: every shard is told it and tells its clients.
//! An operator changes it in the web UI; a scene chosen there is saved in
//! `<state file>.scene` and is what a restarted orchestrator comes back with —
//! `--scene` only says what to show until then (delete the file to make it
//! apply again). An unknown name is an error at startup, and so is `solar`:
//! that scene was retired, the solar system is part of `demo-area` now (a
//! saved `solar` loads as `demo-area`, see [`space_orchestrator::scene`]).

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use space_orchestrator::boot::boot;
use space_orchestrator::config::{LauncherKind, OrchestratorConfig, Secret};
use space_orchestrator::launcher::docker::DockerLauncher;
use space_orchestrator::launcher::local::LocalLauncher;
use space_orchestrator::launcher::process::ProcessLauncher;
use space_orchestrator::launcher::Launcher;
use space_orchestrator::orchestrator::Orchestrator;
use space_orchestrator::webui::{WebUi, WebUiConfig};

/// What the command line (and the environment) said.
#[derive(Debug)]
struct Options {
    config: OrchestratorConfig,
    /// The peer token given with `--peer-token`; `None`: one is generated.
    peer_token: Option<Secret>,
    /// The web UI to serve; `None` (the default) serves none.
    webui: Option<WebUiConfig>,
}

const USAGE: &str =
    "usage: space-orchestrator [--bind <addr:port>] [--control-addr <addr:port>] [--cluster <id>] \
                     [--player-bind <addr:port>] \
                     [--state <path>] [--first-id <n>] [--launcher local|process|docker] \
                     [--min-shards <n>] [--max-shards <n>] [--shard-exe <path>] \
                     [--shard-log-dir <dir>] [--host-port-range <a>-<b>] \
                     [--cluster-secret <secret> | --cluster-secret-file <path>] \
                     [--peer-token <secret> | --peer-token-file <path>] \
                     [--scene demo-area|materials|galaxy] [--max-regions <n>] \
                     [--webui <addr:port>] [--webui-token <secret> | --webui-token-file <path>] \
                     [--webui-host <name>[:<port>]]... \
     (environment fallbacks: SPACE_ORCHESTRATOR_SCENE for --scene; \
     SPACE_ORCHESTRATOR_PLAYER_BIND for --player-bind; \
     SPACE_ORCHESTRATOR_MIN_SHARDS, SPACE_ORCHESTRATOR_MAX_SHARDS, SPACE_ORCHESTRATOR_LAUNCHER, \
     SPACE_ORCHESTRATOR_SHARD_EXE, SPACE_ORCHESTRATOR_SHARD_LOG_DIR and \
     SPACE_ORCHESTRATOR_HOST_PORTS for the shard flags; SPACE_CLUSTER_SECRET and \
     SPACE_CLUSTER_SECRET_FILE for the cluster secret; SPACE_ORCHESTRATOR_PEER_TOKEN and \
     SPACE_ORCHESTRATOR_PEER_TOKEN_FILE for the peer token; for the --webui flags \
     SPACE_ORCHESTRATOR_WEBUI, SPACE_ORCHESTRATOR_WEBUI_TOKEN, \
     SPACE_ORCHESTRATOR_WEBUI_TOKEN_FILE, SPACE_ORCHESTRATOR_WEBUI_HOSTS)";

/// Environment fallback for `--scene`.
const ENV_SCENE: &str = "SPACE_ORCHESTRATOR_SCENE";

/// Environment fallback for `--player-bind`.
const ENV_PLAYER_BIND: &str = "SPACE_ORCHESTRATOR_PLAYER_BIND";

/// Environment fallback for `--min-shards`.
const ENV_MIN_SHARDS: &str = "SPACE_ORCHESTRATOR_MIN_SHARDS";
/// Environment fallback for `--max-shards`.
const ENV_MAX_SHARDS: &str = "SPACE_ORCHESTRATOR_MAX_SHARDS";
/// Environment fallback for `--launcher`.
const ENV_LAUNCHER: &str = "SPACE_ORCHESTRATOR_LAUNCHER";
/// Environment fallback for `--shard-exe`.
const ENV_SHARD_EXE: &str = "SPACE_ORCHESTRATOR_SHARD_EXE";
/// Environment fallback for `--shard-log-dir`.
const ENV_SHARD_LOG_DIR: &str = "SPACE_ORCHESTRATOR_SHARD_LOG_DIR";
/// Environment fallback for `--host-port-range`.
const ENV_HOST_PORTS: &str = "SPACE_ORCHESTRATOR_HOST_PORTS";
/// Environment fallback for `--cluster-secret` (the variable a shard reads its
/// own from, so one setting serves both).
const ENV_CLUSTER_SECRET: &str = space_server::orchestrated::ENV_CLUSTER_SECRET;
/// Environment fallback for `--cluster-secret-file`.
const ENV_CLUSTER_SECRET_FILE: &str = space_server::orchestrated::ENV_CLUSTER_SECRET_FILE;
/// Environment fallback for `--peer-token`. (An orchestrator's variable: the
/// process launcher keeps `SPACE_ORCHESTRATOR_*` out of its children.)
const ENV_PEER_TOKEN: &str = "SPACE_ORCHESTRATOR_PEER_TOKEN";
/// Environment fallback for `--peer-token-file`.
const ENV_PEER_TOKEN_FILE: &str = "SPACE_ORCHESTRATOR_PEER_TOKEN_FILE";

/// Environment fallback for `--webui`.
const ENV_WEBUI: &str = "SPACE_ORCHESTRATOR_WEBUI";
/// Environment fallback for `--webui-token`.
const ENV_WEBUI_TOKEN: &str = "SPACE_ORCHESTRATOR_WEBUI_TOKEN";
/// Environment fallback for `--webui-token-file`.
const ENV_WEBUI_TOKEN_FILE: &str = "SPACE_ORCHESTRATOR_WEBUI_TOKEN_FILE";
/// Environment fallback for `--webui-host` (comma-separated).
const ENV_WEBUI_HOSTS: &str = "SPACE_ORCHESTRATOR_WEBUI_HOSTS";

/// The web UI flags as given, before they are combined with the environment.
#[derive(Debug, Default)]
struct WebUiFlags {
    bind: Option<String>,
    token: Option<String>,
    token_file: Option<PathBuf>,
    hosts: Vec<String>,
}

impl WebUiFlags {
    /// Combine the flags with the environment into a configuration, or `None`
    /// when no address was given anywhere (the UI stays off).
    ///
    /// The "no open bind without a token" rule is [`WebUiConfig::new`]'s; a
    /// configuration that breaks it is an error here, before anything starts.
    fn resolve(self, env: &dyn Fn(&str) -> Option<String>) -> Result<Option<WebUiConfig>, String> {
        let set = |name: &str| env(name).filter(|value| !value.trim().is_empty());
        let Some(bind) = self.bind.or_else(|| set(ENV_WEBUI)) else {
            if self.token.is_some() || self.token_file.is_some() || !self.hosts.is_empty() {
                return Err(
                    "--webui-token, --webui-token-file and --webui-host need --webui <addr:port>"
                        .to_owned(),
                );
            }
            return Ok(None);
        };
        let bind: SocketAddr = bind
            .trim()
            .parse()
            .map_err(|e| format!("web UI address {bind:?}: {e}"))?;
        let token = match (self.token, self.token_file) {
            (Some(_), Some(_)) => {
                return Err("give --webui-token or --webui-token-file, not both".to_owned());
            }
            (Some(token), None) => Some(token),
            (None, Some(path)) => Some(read_token_file(&path)?),
            (None, None) => match (set(ENV_WEBUI_TOKEN), set(ENV_WEBUI_TOKEN_FILE)) {
                (Some(token), _) => Some(token),
                (None, Some(path)) => Some(read_token_file(Path::new(&path))?),
                (None, None) => None,
            },
        };
        let hosts = if self.hosts.is_empty() {
            set(ENV_WEBUI_HOSTS).map_or_else(Vec::new, |names| {
                names
                    .split(',')
                    .map(str::trim)
                    .filter(|name| !name.is_empty())
                    .map(str::to_owned)
                    .collect()
            })
        } else {
            self.hosts
        };
        WebUiConfig::new(bind, token, hosts).map(Some)
    }
}

/// The token in `path`: the file's content without surrounding whitespace.
fn read_token_file(path: &Path) -> Result<String, String> {
    read_secret_file("web UI token file", path)
}

/// The secret in `path` (`what` names the file in an error): its content
/// without surrounding whitespace.
fn read_secret_file(what: &str, path: &Path) -> Result<String, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("{what} {}: {e}", path.display()))?;
    let token = text.trim();
    if token.is_empty() {
        return Err(format!("{what} {} is empty", path.display()));
    }
    Ok(token.to_owned())
}

/// Parse `--launcher`'s value.
fn parse_launcher(name: &str) -> Result<LauncherKind, String> {
    match name.trim() {
        "local" => Ok(LauncherKind::Local),
        "process" => Ok(LauncherKind::Process),
        "docker" => Ok(LauncherKind::Docker),
        other => Err(format!(
            "unknown launcher {other} (local, process or docker)"
        )),
    }
}

/// Parse `--host-port-range`'s value: `<first>-<last>`, both inclusive.
///
/// Only the form is checked. **Whether the range lies outside the operating
/// system's dynamic port range is not** — and it must
/// ([`OrchestratorConfig::host_port_range`] says why: a shard bound to
/// `127.0.0.1:<p>` takes over the datagrams of a socket the system gave port
/// `<p>` on the wildcard address). That rule is the operator's to keep; the
/// default, 7870-7899, keeps it.
fn parse_port_range(text: &str) -> Result<(u16, u16), String> {
    let bad = || format!("the host port range must be <first>-<last>, e.g. 7870-7899 (got {text})");
    let (lo, hi) = text.trim().split_once('-').ok_or_else(bad)?;
    let lo = lo.trim().parse::<u16>().map_err(|_| bad())?;
    let hi = hi.trim().parse::<u16>().map_err(|_| bad())?;
    if lo == 0 || hi < lo {
        return Err(bad());
    }
    Ok((lo, hi))
}

/// Parse a shard count (`--min-shards`, `--max-shards`).
fn parse_count(flag: &str, text: &str, at_least: usize) -> Result<usize, String> {
    text.trim()
        .parse::<usize>()
        .ok()
        .filter(|n| *n >= at_least)
        .ok_or_else(|| format!("{flag} needs a number from {at_least} up (got {text})"))
}

impl Options {
    /// Parse `argv` without the program name. `env` looks a variable up (the
    /// process environment in production, a table in tests).
    fn parse(args: &[String], env: &dyn Fn(&str) -> Option<String>) -> Result<Self, String> {
        let mut config = OrchestratorConfig::default();
        let mut peer_token: Option<String> = None;
        let mut peer_token_file: Option<PathBuf> = None;
        let mut webui = WebUiFlags::default();
        let mut scene: Option<String> = None;
        let mut player_bind: Option<String> = None;
        // The shard flags as given; the environment fills what is missing.
        let mut launcher: Option<String> = None;
        let mut min_shards: Option<String> = None;
        let mut max_shards: Option<String> = None;
        let mut shard_exe: Option<String> = None;
        let mut shard_log_dir: Option<String> = None;
        let mut host_ports: Option<String> = None;
        let mut cluster_secret: Option<String> = None;
        let mut cluster_secret_file: Option<PathBuf> = None;
        let mut i = 0;
        while i < args.len() {
            let flag = args[i].as_str();
            let value = |what: &str| -> Result<&String, String> {
                args.get(i + 1)
                    .ok_or_else(|| format!("{what} needs a value"))
            };
            match flag {
                "-h" | "--help" => return Err(USAGE.to_owned()),
                "--bind" => {
                    config.bind = value("--bind")?
                        .parse::<SocketAddr>()
                        .map_err(|e| e.to_string())?;
                }
                "--control-addr" => {
                    config.control_addr = Some(
                        value("--control-addr")?
                            .parse::<SocketAddr>()
                            .map_err(|e| e.to_string())?,
                    );
                }
                "--cluster" => {
                    config.cluster = value("--cluster")?.clone();
                }
                "--player-bind" => {
                    player_bind = Some(value("--player-bind")?.clone());
                }
                "--state" => {
                    config.state_path = PathBuf::from(value("--state")?);
                }
                "--first-id" => {
                    config.first_shard_id = value("--first-id")?
                        .parse::<u64>()
                        .map_err(|e| e.to_string())?;
                }
                "--launcher" => {
                    launcher = Some(value("--launcher")?.clone());
                }
                "--min-shards" => {
                    min_shards = Some(value("--min-shards")?.clone());
                }
                "--max-shards" => {
                    max_shards = Some(value("--max-shards")?.clone());
                }
                "--shard-exe" => {
                    shard_exe = Some(value("--shard-exe")?.clone());
                }
                "--shard-log-dir" => {
                    shard_log_dir = Some(value("--shard-log-dir")?.clone());
                }
                "--host-port-range" => {
                    host_ports = Some(value("--host-port-range")?.clone());
                }
                "--cluster-secret" => {
                    cluster_secret = Some(value("--cluster-secret")?.clone());
                }
                "--cluster-secret-file" => {
                    cluster_secret_file = Some(PathBuf::from(value("--cluster-secret-file")?));
                }
                "--peer-token" => {
                    peer_token = Some(value("--peer-token")?.clone());
                }
                "--peer-token-file" => {
                    peer_token_file = Some(PathBuf::from(value("--peer-token-file")?));
                }
                "--scene" => {
                    scene = Some(value("--scene")?.clone());
                }
                "--max-regions" => {
                    config.max_regions = value("--max-regions")?
                        .parse::<usize>()
                        .ok()
                        .filter(|limit| *limit >= 1)
                        .ok_or("--max-regions needs a number from 1 up")?;
                }
                "--webui" => {
                    webui.bind = Some(value("--webui")?.clone());
                }
                "--webui-token" => {
                    webui.token = Some(value("--webui-token")?.clone());
                }
                "--webui-token-file" => {
                    webui.token_file = Some(PathBuf::from(value("--webui-token-file")?));
                }
                "--webui-host" => {
                    webui.hosts.push(value("--webui-host")?.clone());
                }
                other => return Err(format!("unknown argument {other}\n{USAGE}")),
            }
            i += if flag == "--help" || flag == "-h" {
                1
            } else {
                2
            };
        }
        // The initial scene: the flag, else the environment, else the default.
        // An unknown name is refused here rather than falling back, so a typo
        // cannot quietly show another world.
        let from_env = || env(ENV_SCENE).filter(|value| !value.trim().is_empty());
        if let Some(name) = scene.or_else(from_env) {
            config.initial_scene = name
                .trim()
                .parse::<space_protocol::Scene>()
                .map_err(|e| format!("--scene ({ENV_SCENE}): {e}"))?;
        }
        // The shards: each flag, else its variable, else the default. An empty
        // variable counts as unset; a value that makes no sense is an error,
        // never a silent default.
        let set = |name: &str| env(name).filter(|value| !value.trim().is_empty());
        // The player port: off unless an address is given.
        if let Some(text) = player_bind.or_else(|| set(ENV_PLAYER_BIND)) {
            let addr = text
                .trim()
                .parse::<SocketAddr>()
                .map_err(|e| format!("--player-bind ({ENV_PLAYER_BIND}) {text:?}: {e}"))?;
            if addr.port() != 0 && addr.port() == config.bind.port() {
                // One UDP port is one listener; said here rather than as a
                // bind error once the control port is already up.
                return Err(format!(
                    "--player-bind {addr} uses the control port's port ({}): players and shards \
                     connect to different ports",
                    config.bind
                ));
            }
            config.player_bind = Some(addr);
        }
        if let Some(name) = launcher.or_else(|| set(ENV_LAUNCHER)) {
            config.launcher = parse_launcher(&name)?;
        }
        if let Some(text) = min_shards.or_else(|| set(ENV_MIN_SHARDS)) {
            config.min_shards = parse_count("--min-shards", &text, 0)?;
        }
        if let Some(text) = max_shards.or_else(|| set(ENV_MAX_SHARDS)) {
            config.max_shards = parse_count("--max-shards", &text, 1)?;
        }
        // A `--min-shards` the orchestrator could not keep — above
        // `--max-shards`, above `--max-regions`, or more shards than it has
        // regions to give — is refused here: started anyway, the shard that
        // is one too many would be launched, refused and launched again.
        config.check_shard_counts()?;
        if let Some(path) = shard_exe.or_else(|| set(ENV_SHARD_EXE)) {
            config.shard_exe = Some(PathBuf::from(path.trim()));
        }
        if let Some(dir) = shard_log_dir.or_else(|| set(ENV_SHARD_LOG_DIR)) {
            config.shard_log_dir = PathBuf::from(dir.trim());
        }
        if let Some(range) = host_ports.or_else(|| set(ENV_HOST_PORTS)) {
            config.host_port_range = parse_port_range(&range)?;
        }
        // The cluster secret: a flag wins over the environment, the secret
        // itself over a file naming it.
        let secret = match (cluster_secret, cluster_secret_file) {
            (Some(_), Some(_)) => {
                return Err("give --cluster-secret or --cluster-secret-file, not both".to_owned());
            }
            (Some(secret), None) => Some(secret),
            (None, Some(path)) => Some(read_secret_file("cluster secret file", &path)?),
            (None, None) => match (set(ENV_CLUSTER_SECRET), set(ENV_CLUSTER_SECRET_FILE)) {
                (Some(secret), _) => Some(secret.trim().to_owned()),
                (None, Some(path)) => Some(read_secret_file(
                    "cluster secret file",
                    Path::new(path.trim()),
                )?),
                (None, None) => None,
            },
        };
        if secret.as_ref().is_some_and(String::is_empty) {
            return Err("the cluster secret is empty".to_owned());
        }
        config.cluster_secret = secret.map(Secret::new);
        // Refused here, before anything starts: a control port that is not on
        // loopback and no secret a shard could present.
        config.check_secret()?;
        // The peer token, the same way: a flag wins over the environment, the
        // token itself over a file naming it. None at all is fine — one is
        // generated at start. One that is given must be a secret: an empty
        // token means "no peer token" to a shard, which then lets every peer
        // in, so a short or empty one is an error and never a quiet "off".
        let peer_token = match (peer_token, peer_token_file) {
            (Some(_), Some(_)) => {
                return Err("give --peer-token or --peer-token-file, not both".to_owned());
            }
            (Some(token), None) => Some(token),
            (None, Some(path)) => Some(read_secret_file("peer token file", &path)?),
            (None, None) => match (set(ENV_PEER_TOKEN), set(ENV_PEER_TOKEN_FILE)) {
                (Some(token), _) => Some(token.trim().to_owned()),
                (None, Some(path)) => {
                    Some(read_secret_file("peer token file", Path::new(path.trim()))?)
                }
                (None, None) => None,
            },
        };
        if let Some(token) = &peer_token {
            if token.len() < OrchestratorConfig::MIN_SECRET_LEN {
                return Err(format!(
                    "the peer token is too short: use at least {} characters, or give none and \
                     one is generated ({ENV_PEER_TOKEN}, --peer-token-file)",
                    OrchestratorConfig::MIN_SECRET_LEN
                ));
            }
            // It travels in every shard-to-shard hello, and a hello is read
            // with a small size limit (`space_net::MAX_HELLO_PAYLOAD`): a
            // token that does not fit would have every peer hello refused,
            // unreadably. Said here instead.
            if token.len() > space_net::MAX_SECRET_LEN {
                return Err(format!(
                    "the peer token is too long: use at most {} characters \
                     ({ENV_PEER_TOKEN}, --peer-token-file)",
                    space_net::MAX_SECRET_LEN
                ));
            }
        }
        Ok(Self {
            config,
            peer_token: peer_token.map(Secret::new),
            webui: webui.resolve(env)?,
        })
    }
}

fn main() -> Result<(), String> {
    let started = Instant::now();
    let options = Options::parse(&std::env::args().skip(1).collect::<Vec<_>>(), &|name| {
        std::env::var(name).ok()
    })?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    runtime.block_on(run(options, started))
}

/// Build the control plane and run the loop until the endpoint is closed.
async fn run(options: Options, started: Instant) -> Result<(), String> {
    let Options {
        config,
        peer_token,
        webui,
    } = options;
    let webui = webui.map(|webui| webui.with_started(started));
    // The peer token shards present to each other: the one given, else one
    // made now (from the web UI form token's source). Never printed.
    let peer_token = peer_token
        .map_or_else(space_orchestrator::webui::auth::random_token, |token| {
            token.expose().to_owned()
        });
    match config.launcher {
        LauncherKind::Local => serve(config, peer_token, webui, LocalLauncher::new()).await,
        LauncherKind::Process => {
            let exe = match &config.shard_exe {
                Some(exe) => exe.clone(),
                None => ProcessLauncher::default_exe().map_err(|e| e.to_string())?,
            };
            let launcher = ProcessLauncher::new(exe, &config.shard_log_dir, config.bind);
            // Fail now, with the path, rather than at the first launch.
            launcher.ensure_available().map_err(|e| e.to_string())?;
            println!(
                "space-orchestrator: shards are started as {} (logs in {})",
                launcher.exe().display(),
                config.shard_log_dir.display()
            );
            serve(config, peer_token, webui, launcher).await
        }
        LauncherKind::Docker => {
            // Deferred (bollard is not in the offline cache): fail fast with the
            // launcher's own error rather than at the first split.
            let launcher = DockerLauncher::new();
            launcher.ensure_available().map_err(|e| e.to_string())?;
            serve(config, peer_token, webui, launcher).await
        }
    }
}

/// Open the id counter, build the control plane and run the loop with `launcher`.
async fn serve<L: Launcher + 'static>(
    config: OrchestratorConfig,
    peer_token: String,
    webui: Option<WebUiConfig>,
    launcher: L,
) -> Result<(), String> {
    // The initial map: one region at the configured root, owned by the first id.
    // The orchestrator allocates that id so a restart resumes above it (H9); the
    // first live shard that asks for the root adopts it (`boot` marks it unclaimed).
    let control = boot(config.clone(), peer_token)?;
    let root_owner = control
        .unclaimed_root_owner()
        .ok_or("boot left no root owner")?;
    let mut orchestrator = Orchestrator::new(control, Arc::new(launcher));
    println!(
        "space-orchestrator: cluster {:?} on {}, root owned by {root_owner}",
        config.cluster, config.bind
    );
    println!("{}", scene_line(&orchestrator.control.scene, &config));
    println!("{}", shards_line(&config));
    println!("{}", players_line(&config));
    // The web UI runs in its own task and holds only a handle: it reads the
    // snapshots the loop publishes and offers commands to the loop's queue, so
    // nothing it does (or a client does to it) can hold the loop up. An address
    // that cannot be bound is a startup error, like the control port's.
    let _webui = match webui {
        Some(webui) => {
            let bind = webui.bind();
            let ui = WebUi::bind(webui, orchestrator.attach_ops())
                .await
                .map_err(|e| format!("web UI: cannot listen on {bind}: {e}"))?;
            println!("space-orchestrator: web UI on {}", ui.describe());
            Some(tokio::spawn(ui.serve()))
        }
        None => None,
    };
    orchestrator.run().await.map_err(|e| e.to_string())
}

/// One line for the startup log: who starts the shards, and whether a shard
/// has to present the cluster secret. (Never the secret.)
fn shards_line(config: &OrchestratorConfig) -> String {
    let who = if config.min_shards == 0 {
        "none is started by the orchestrator (start them by hand; a split starts its own)"
            .to_owned()
    } else {
        format!(
            "{} kept running by the orchestrator ({} launcher)",
            config.min_shards,
            space_orchestrator::webui::launcher_label(config.launcher)
        )
    };
    let secret = if config.secret_required() {
        "a hello must carry the cluster secret"
    } else {
        "a hello needs no secret (loopback control port)"
    };
    format!(
        "space-orchestrator: shards: {who}, at most {}; {secret}",
        config.max_shards
    )
}

/// One line for the startup log: where players connect, or that this
/// orchestrator has no player port.
fn players_line(config: &OrchestratorConfig) -> String {
    match config.player_bind {
        Some(bind) => format!(
            "space-orchestrator: players connect to {bind} (the player port): each is sent to \
             the shard that owns the spawn point"
        ),
        None => "space-orchestrator: no player port (--player-bind): players dial a shard \
                 directly"
            .to_owned(),
    }
}

/// One line for the startup log: the scene the cluster starts with and where
/// it came from.
fn scene_line(
    scene: &space_orchestrator::scene::SceneStore,
    config: &OrchestratorConfig,
) -> String {
    let current = scene.current();
    let file = scene
        .path()
        .map_or_else(|| "nowhere".to_owned(), |path| path.display().to_string());
    if scene.restored() {
        let unused = if scene.scene() == config.initial_scene {
            String::new()
        } else {
            format!(
                " (a saved scene wins over the initial scene {})",
                config.initial_scene
            )
        };
        format!("space-orchestrator: scene {current}, restored from {file}{unused}")
    } else if scene.note().is_some() {
        format!("space-orchestrator: scene {current}, replacing a retired scene saved in {file}")
    } else {
        format!(
            "space-orchestrator: scene {current}, the initial scene; a change is saved in {file}"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse with an empty environment.
    fn parse(args: &[&str]) -> Result<Options, String> {
        parse_with(args, &[])
    }

    /// Parse with the given environment variables set.
    fn parse_with(args: &[&str], env: &[(&str, &str)]) -> Result<Options, String> {
        let args: Vec<String> = args.iter().map(|s| (*s).to_owned()).collect();
        Options::parse(&args, &|name| {
            env.iter()
                .find(|(n, _)| *n == name)
                .map(|(_, v)| (*v).to_owned())
        })
    }

    const TOKEN: &str = "a-long-enough-test-token";

    #[test]
    fn defaults_match_the_config() {
        let options = parse(&[]).expect("parses");
        assert_eq!(
            options.config.cluster,
            OrchestratorConfig::default().cluster
        );
        assert_eq!(options.config.bind, OrchestratorConfig::default().bind);
        assert_eq!(options.config.control_addr, None);
        assert_eq!(options.config.control_addr(), options.config.bind);
        assert!(options.webui.is_none(), "the web UI is off by default");
    }

    /// The player port is **off unless asked for** — from the flag, else its
    /// variable — takes any address without a token (it is the public entry,
    /// not a control surface), and cannot share the control port's port.
    ///
    /// Mutation this catches: a player port that is on by default (a second
    /// public listener nobody asked for), or one that demands a secret on a
    /// reachable bind like the control surfaces do (no player could connect).
    #[test]
    fn the_player_port_is_off_unless_asked_for_and_needs_no_token() {
        assert_eq!(parse(&[]).expect("parses").config.player_bind, None);
        let by_flag = parse(&["--player-bind", "127.0.0.1:7777"]).expect("parses");
        assert_eq!(
            by_flag.config.player_bind,
            Some("127.0.0.1:7777".parse().unwrap())
        );
        let by_env = parse_with(
            &[],
            &[("SPACE_ORCHESTRATOR_PLAYER_BIND", " 0.0.0.0:7777\n")],
        )
        .expect("parses");
        assert_eq!(
            by_env.config.player_bind,
            Some("0.0.0.0:7777".parse().unwrap())
        );
        // A flag wins over its variable; an empty variable is unset.
        let both = parse_with(
            &["--player-bind", "127.0.0.1:7001"],
            &[("SPACE_ORCHESTRATOR_PLAYER_BIND", "127.0.0.1:7002")],
        )
        .expect("parses");
        assert_eq!(both.config.player_bind.map(|a| a.port()), Some(7_001));
        let empty = parse_with(&[], &[("SPACE_ORCHESTRATOR_PLAYER_BIND", " ")]).expect("parses");
        assert_eq!(empty.config.player_bind, None);
        // Open to the world with nothing else configured: the control port
        // stays on loopback and needs no secret, the player port none either.
        for open in ["0.0.0.0:7777", "[::]:7777", "192.168.1.5:7777"] {
            let options = parse(&["--player-bind", open]).expect(open);
            assert!(!options.config.secret_required(), "{open}");
            assert!(options.webui.is_none());
        }
        // Nonsense, and the control port's own port.
        assert!(parse(&["--player-bind", "nope"]).is_err());
        assert!(parse(&["--player-bind"]).is_err());
        let err = parse(&["--player-bind", "127.0.0.1:7900"]).expect_err("the control port");
        assert!(err.contains("control port"), "{err}");
        let err = parse(&["--bind", "127.0.0.1:7777", "--player-bind", "0.0.0.0:7777"])
            .expect_err("the same port");
        assert!(err.contains("different ports"), "{err}");
        let err = parse_with(&[], &[("SPACE_ORCHESTRATOR_PLAYER_BIND", "x")])
            .expect_err("nonsense in the environment is an error too");
        assert!(err.contains("SPACE_ORCHESTRATOR_PLAYER_BIND"), "{err}");
    }

    #[test]
    fn the_region_limit_has_a_default_and_a_flag() {
        assert_eq!(
            parse(&[]).expect("parses").config.max_regions,
            OrchestratorConfig::DEFAULT_MAX_REGIONS
        );
        assert_eq!(
            parse(&["--max-regions", "12"])
                .expect("parses")
                .config
                .max_regions,
            12
        );
    }

    #[test]
    fn control_addr_is_parsed_and_distinct_from_bind() {
        // (A control port that is not on loopback needs a cluster secret to
        // start at all; that rule has a test of its own.)
        let options = parse(&[
            "--bind",
            "0.0.0.0:7900",
            "--control-addr",
            "10.0.0.2:7900",
            "--cluster-secret",
            "a-long-enough-cluster-secret",
        ])
        .expect("parses");
        assert_eq!(options.config.bind, "0.0.0.0:7900".parse().unwrap());
        assert_eq!(
            options.config.control_addr(),
            "10.0.0.2:7900".parse().unwrap()
        );
        assert!(parse(&["--control-addr", "nope"]).is_err());
    }

    #[tokio::test]
    async fn the_docker_launcher_is_an_unavailable_error_not_a_panic() {
        let options = parse(&["--launcher", "docker"]).expect("parses");
        let err = run(options, Instant::now())
            .await
            .expect_err("docker is deferred");
        assert!(err.contains("bollard"), "{err}");
    }

    #[test]
    fn flags_override_the_config() {
        let options = parse(&[
            "--bind",
            "127.0.0.1:9000",
            "--cluster",
            "test",
            "--first-id",
            "10",
            "--launcher",
            "local",
            "--peer-token",
            "a-long-enough-peer-token",
        ])
        .expect("parses");
        assert_eq!(options.config.bind.port(), 9_000);
        assert_eq!(options.config.cluster, "test");
        assert_eq!(options.config.first_shard_id, 10);
        assert_eq!(
            options.peer_token.as_ref().map(Secret::expose),
            Some("a-long-enough-peer-token")
        );
        assert!(
            parse(&[]).expect("parses").peer_token.is_none(),
            "no peer token given: one is generated at start"
        );
    }

    /// The peer token: from the flag, a file or the environment, like the
    /// cluster secret — and a configured one must be a secret. An empty token
    /// is what a shard takes for "this cluster has no peer token", and it
    /// then lets every peer in.
    ///
    /// Mutation this catches: accepting `--peer-token ""` (peer
    /// authentication silently off for the whole cluster), or a token that
    /// can only be given where the process list shows it.
    #[test]
    fn a_configured_peer_token_is_a_secret_or_an_error() {
        const TOKEN: &str = "a-long-enough-peer-token";
        let token_of = |options: &Options| {
            options
                .peer_token
                .as_ref()
                .map(|token| token.expose().to_owned())
        };
        for short in ["", "x", "short"] {
            let err = parse(&["--peer-token", short]).expect_err(short);
            assert!(err.contains("peer token is too short"), "{short:?}: {err}");
        }
        let err = parse_with(&[], &[("SPACE_ORCHESTRATOR_PEER_TOKEN", "short")])
            .expect_err("from the environment too");
        assert!(err.contains("peer token is too short"), "{err}");
        let err = parse(&["--peer-token", "fifteen-chars-a"]).expect_err("15 characters");
        assert!(err.contains("at least 16"), "{err}");
        assert!(
            parse(&["--peer-token", "sixteen-chars-ab"]).is_ok(),
            "16 are enough"
        );
        // And it has to fit a hello, which is read with a small size limit.
        let longest = "t".repeat(space_net::MAX_SECRET_LEN);
        assert!(parse(&["--peer-token", &longest]).is_ok());
        let err = parse(&["--peer-token", &format!("{longest}t")]).expect_err("too long");
        assert!(err.contains("peer token is too long"), "{err}");

        let by_env = parse_with(
            &[],
            &[(
                "SPACE_ORCHESTRATOR_PEER_TOKEN",
                " a-long-enough-peer-token\n",
            )],
        )
        .expect("parses");
        assert_eq!(token_of(&by_env).as_deref(), Some(TOKEN));
        assert!(
            !format!("{by_env:?}").contains(TOKEN),
            "the token is never printed"
        );
        // An empty variable is unset: one is generated, as without it.
        let unset = parse_with(&[], &[("SPACE_ORCHESTRATOR_PEER_TOKEN", " ")]).expect("parses");
        assert_eq!(token_of(&unset), None);

        let path = std::env::temp_dir().join(format!(
            "space-orch-peer-token-{}-{}.txt",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&path, format!("{TOKEN}\n")).expect("write the token file");
        let file = path.to_str().expect("a UTF-8 temp path");
        let by_file = parse(&["--peer-token-file", file]).expect("parses");
        assert_eq!(token_of(&by_file).as_deref(), Some(TOKEN));
        let by_env_file =
            parse_with(&[], &[("SPACE_ORCHESTRATOR_PEER_TOKEN_FILE", file)]).expect("parses");
        assert_eq!(token_of(&by_env_file).as_deref(), Some(TOKEN));
        // A flag wins over the environment; both flags at once are ambiguous.
        let both = parse_with(
            &["--peer-token", "the-flags-own-peer-token"],
            &[("SPACE_ORCHESTRATOR_PEER_TOKEN", TOKEN)],
        )
        .expect("parses");
        assert_eq!(token_of(&both).as_deref(), Some("the-flags-own-peer-token"));
        assert!(parse(&["--peer-token", TOKEN, "--peer-token-file", file]).is_err());
        std::fs::write(&path, "short\n").expect("shorten the token file");
        let err = parse(&["--peer-token-file", file]).expect_err("a short token in a file");
        assert!(err.contains("peer token is too short"), "{err}");
        std::fs::remove_file(&path).ok();
        assert!(parse(&["--peer-token-file", file]).is_err(), "no file");
    }

    /// `--min-shards` is refused at startup when the orchestrator could not
    /// keep it: above `--max-shards`, above `--max-regions`, or more shards
    /// than it has regions to give (the root and seven layout cells).
    ///
    /// Mutation this catches: checking only `min <= max` — `--min-shards 9
    /// --max-shards 9` started, and the ninth shard was launched, refused for
    /// good and launched again, for ever.
    #[test]
    fn a_min_shards_that_cannot_be_kept_is_refused_at_startup() {
        let ok = parse(&["--min-shards", "8"]).expect("eight fit").config;
        assert_eq!((ok.min_shards, ok.max_shards), (8, 8));
        let err = parse(&["--min-shards", "9", "--max-shards", "9"]).expect_err("no ninth cell");
        assert!(
            err.contains("--min-shards 9") && err.contains("layout cell"),
            "{err}"
        );
        let err = parse(&["--min-shards", "9", "--max-shards", "12"]).expect_err("nor so");
        assert!(err.contains("layout cell"), "{err}");
        let err = parse(&["--min-shards", "4", "--max-regions", "3"]).expect_err("no room");
        assert!(err.contains("--max-regions 3"), "{err}");
        let err = parse_with(
            &["--max-regions", "2"],
            &[("SPACE_ORCHESTRATOR_MIN_SHARDS", "3")],
        )
        .expect_err("from the environment too");
        assert!(err.contains("--max-regions 2"), "{err}");
        assert!(parse(&["--min-shards", "4", "--max-regions", "4"]).is_ok());
    }

    /// The shard flags: each from its flag, else its variable, else the
    /// default; a flag wins, an empty variable is unset, nonsense is an error.
    ///
    /// Mutation this catches: a flag that is parsed and dropped, the default
    /// `--min-shards` becoming non-zero (an orchestrator that suddenly starts
    /// shards nobody asked for), or min above max going unnoticed.
    #[test]
    fn the_shard_flags_come_from_the_flag_then_the_environment_then_the_default() {
        let defaults = parse(&[]).expect("parses").config;
        assert_eq!(defaults.min_shards, 0, "nothing is launched unless asked");
        assert_eq!(defaults.max_shards, 8);
        assert_eq!(defaults.launcher, LauncherKind::Local);
        assert_eq!(defaults.host_port_range, (7_870, 7_899));
        assert_eq!(defaults.shard_exe, None);
        assert_eq!(defaults.shard_log_dir, PathBuf::from("."));

        let flags = parse(&[
            "--min-shards",
            "4",
            "--max-shards",
            "6",
            "--launcher",
            "process",
            "--shard-exe",
            "bin/space-server",
            "--shard-log-dir",
            "target/dev",
            "--host-port-range",
            "8170-8199",
        ])
        .expect("parses")
        .config;
        assert_eq!((flags.min_shards, flags.max_shards), (4, 6));
        assert_eq!(flags.launcher, LauncherKind::Process);
        assert_eq!(flags.shard_exe, Some(PathBuf::from("bin/space-server")));
        assert_eq!(flags.shard_log_dir, PathBuf::from("target/dev"));
        assert_eq!(flags.host_port_range, (8_170, 8_199));

        let env = [
            ("SPACE_ORCHESTRATOR_MIN_SHARDS", "2"),
            ("SPACE_ORCHESTRATOR_MAX_SHARDS", "3"),
            ("SPACE_ORCHESTRATOR_LAUNCHER", "process"),
            ("SPACE_ORCHESTRATOR_SHARD_EXE", "/opt/space-server"),
            ("SPACE_ORCHESTRATOR_SHARD_LOG_DIR", "/var/log/space"),
            ("SPACE_ORCHESTRATOR_HOST_PORTS", " 9000-9007 "),
        ];
        let by_env = parse_with(&[], &env).expect("parses").config;
        assert_eq!((by_env.min_shards, by_env.max_shards), (2, 3));
        assert_eq!(by_env.launcher, LauncherKind::Process);
        assert_eq!(by_env.shard_exe, Some(PathBuf::from("/opt/space-server")));
        assert_eq!(by_env.shard_log_dir, PathBuf::from("/var/log/space"));
        assert_eq!(by_env.host_port_range, (9_000, 9_007));
        // A flag wins over its variable; an empty variable is unset.
        let both = parse_with(&["--min-shards", "1"], &env)
            .expect("parses")
            .config;
        assert_eq!(both.min_shards, 1);
        let empty = parse_with(&[], &[("SPACE_ORCHESTRATOR_MIN_SHARDS", " ")]).expect("parses");
        assert_eq!(empty.config.min_shards, 0);

        for bad in [
            &["--min-shards", "x"] as &[&str],
            &["--min-shards", "-1"],
            &["--max-shards", "0"],
            &["--min-shards", "9"], // more than the default maximum of 8
            &["--min-shards", "3", "--max-shards", "2"],
            &["--host-port-range", "7870"],
            &["--host-port-range", "7899-7870"],
            &["--host-port-range", "0-10"],
            &["--launcher", "kubernetes"],
            &["--min-shards"],
        ] {
            assert!(parse(bad).is_err(), "{bad:?} must be rejected");
        }
        let err = parse_with(&[], &[("SPACE_ORCHESTRATOR_MIN_SHARDS", "many")])
            .expect_err("nonsense in the environment is an error too");
        assert!(err.contains("--min-shards"), "{err}");
    }

    /// The cluster secret: from the flag, a file, or the environment — and a
    /// control port that is not on loopback does not start without one a shard
    /// could present.
    ///
    /// Mutation this catches: an open control port starting with no secret
    /// (the state before this change), or a secret showing in a `Debug` print
    /// of the options.
    #[test]
    fn the_cluster_secret_is_configured_and_required_off_loopback() {
        const SECRET: &str = "a-long-enough-cluster-secret";
        let secret_of = |options: &Options| {
            options
                .config
                .cluster_secret
                .as_ref()
                .map(|secret| secret.expose().to_owned())
        };
        let none = parse(&[]).expect("parses");
        assert_eq!(secret_of(&none), None);
        assert!(!none.config.secret_required(), "loopback needs none");

        let by_flag = parse(&["--cluster-secret", SECRET]).expect("parses");
        assert_eq!(secret_of(&by_flag).as_deref(), Some(SECRET));
        assert!(
            by_flag.config.secret_required(),
            "a configured secret is required, on loopback too"
        );
        assert!(
            !format!("{by_flag:?}").contains(SECRET),
            "the secret is never printed"
        );
        let by_env = parse_with(
            &[],
            &[("SPACE_CLUSTER_SECRET", " a-long-enough-cluster-secret\n")],
        )
        .expect("parses");
        assert_eq!(secret_of(&by_env).as_deref(), Some(SECRET));

        let path = std::env::temp_dir().join(format!(
            "space-orch-cluster-secret-{}-{}.txt",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&path, format!("{SECRET}\n")).expect("write the secret file");
        let file = path.to_str().expect("a UTF-8 temp path");
        let by_file = parse(&["--cluster-secret-file", file]).expect("parses");
        assert_eq!(secret_of(&by_file).as_deref(), Some(SECRET));
        let by_env_file = parse_with(&[], &[("SPACE_CLUSTER_SECRET_FILE", file)]).expect("parses");
        assert_eq!(secret_of(&by_env_file).as_deref(), Some(SECRET));
        assert!(parse(&["--cluster-secret", SECRET, "--cluster-secret-file", file]).is_err());
        std::fs::remove_file(&path).ok();
        assert!(parse(&["--cluster-secret-file", file]).is_err(), "no file");

        // Off loopback.
        for open in ["0.0.0.0:7900", "[::]:7900", "192.168.1.5:7900"] {
            let err = parse(&["--bind", open]).expect_err(open);
            assert!(err.contains("without a cluster secret"), "{err}");
            let err = parse(&["--bind", open, "--cluster-secret", "short"]).expect_err(open);
            assert!(err.contains("too short"), "{err}");
            let with = parse(&["--bind", open, "--cluster-secret", SECRET]).expect("a secret");
            assert!(with.config.secret_required());
            let by_env = parse_with(&["--bind", open], &[("SPACE_CLUSTER_SECRET", SECRET)])
                .expect("a secret from the environment");
            assert!(by_env.config.secret_required());
            // Or shards it starts itself, which are handed a generated one.
            let managed = parse(&["--bind", open, "--min-shards", "4"]).expect("managed");
            assert!(managed.config.secret_required());
            assert_eq!(secret_of(&managed), None, "generated at boot, not here");
        }
    }

    /// `--launcher process` without the executable it would start is an error
    /// at startup that names the path, not a failure at the first launch.
    #[tokio::test]
    async fn the_process_launcher_without_its_executable_is_a_startup_error() {
        let missing = std::env::temp_dir()
            .join("no-such-dir-for-space")
            .join("space-server");
        let options = parse(&[
            "--launcher",
            "process",
            "--shard-exe",
            missing.to_str().expect("a UTF-8 temp path"),
        ])
        .expect("parses");
        let err = run(options, Instant::now())
            .await
            .expect_err("nothing to start");
        assert!(
            err.contains("does not exist") && err.contains("no-such-dir-for-space"),
            "{err}"
        );
    }

    #[test]
    fn the_startup_line_says_who_starts_the_shards_and_never_a_secret() {
        let config = parse(&["--cluster-secret", "a-long-enough-cluster-secret"])
            .expect("parses")
            .config;
        let line = shards_line(&config);
        assert!(
            line.contains("none is started by the orchestrator")
                && line.contains("at most 8")
                && line.contains("must carry the cluster secret"),
            "{line}"
        );
        assert!(!line.contains("a-long-enough"), "{line}");
        let config = parse(&["--min-shards", "4", "--launcher", "process"])
            .expect("parses")
            .config;
        let line = shards_line(&config);
        assert!(
            line.contains("4 kept running by the orchestrator (process launcher)")
                && line.contains("needs no secret"),
            "{line}"
        );
    }

    /// Mutation this catches: a `--scene` that is parsed and dropped, the
    /// environment winning over the flag, or an unknown name quietly becoming
    /// the default (a typo would show another world on every shard).
    #[test]
    fn the_initial_scene_comes_from_the_flag_then_the_environment_then_the_default() {
        use space_protocol::Scene;
        assert_eq!(
            parse(&[]).expect("parses").config.initial_scene,
            Scene::DemoArea,
            "the demo area is the default"
        );
        for scene in Scene::ALL {
            let by_flag = parse(&["--scene", scene.name()])
                .unwrap_or_else(|err| panic!("{scene} must parse: {err}"));
            assert_eq!(by_flag.config.initial_scene, scene);
            let by_env =
                parse_with(&[], &[("SPACE_ORCHESTRATOR_SCENE", scene.name())]).expect("parses");
            assert_eq!(by_env.config.initial_scene, scene);
        }
        // A flag wins over the environment; an empty variable is unset.
        let both = parse_with(
            &["--scene", "materials"],
            &[("SPACE_ORCHESTRATOR_SCENE", "galaxy")],
        )
        .expect("parses");
        assert_eq!(both.config.initial_scene, Scene::Materials);
        let empty = parse_with(&[], &[("SPACE_ORCHESTRATOR_SCENE", "  ")]).expect("parses");
        assert_eq!(empty.config.initial_scene, Scene::DemoArea);
        let padded = parse_with(&[], &[("SPACE_ORCHESTRATOR_SCENE", " galaxy\n")]).expect("parses");
        assert_eq!(padded.config.initial_scene, Scene::Galaxy);

        // An unknown name is an error that lists the catalogue, from either
        // source. `solar` was retired (merged into the demo area): it is
        // refused like any name the catalogue does not have.
        for bad in ["solar", "Solar", "demo", "4"] {
            let err = parse(&["--scene", bad]).expect_err(bad);
            assert!(err.contains("demo-area, materials, galaxy"), "{bad}: {err}");
            let err = parse_with(&[], &[("SPACE_ORCHESTRATOR_SCENE", bad)]).expect_err(bad);
            assert!(err.contains("SPACE_ORCHESTRATOR_SCENE"), "{bad}: {err}");
        }
        assert!(parse(&["--scene"]).is_err(), "the flag needs a value");
    }

    #[test]
    fn the_startup_line_says_which_scene_and_where_it_came_from() {
        use space_orchestrator::scene::{scene_path, SceneStore};
        use space_protocol::Scene;
        let state = std::env::temp_dir().join(format!(
            "space-orch-main-scene-{}-{}.id",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        let file = scene_path(&state);
        let _ = std::fs::remove_file(&file);
        let config = OrchestratorConfig {
            state_path: state,
            initial_scene: Scene::Galaxy,
            ..OrchestratorConfig::default()
        };
        let mut store = SceneStore::open(&file, config.initial_scene);
        let line = scene_line(&store, &config);
        assert!(
            line.contains("scene galaxy (revision 1), the initial scene")
                && line.contains(&file.display().to_string()),
            "{line}"
        );
        store.set(Scene::Materials).expect("saved");
        let restored = SceneStore::open(&file, config.initial_scene);
        let line = scene_line(&restored, &config);
        assert!(
            line.contains("scene materials (revision 2), restored from")
                && line.contains("wins over the initial scene galaxy"),
            "{line}"
        );
        // A saved scene that was retired: the default, not "the initial scene".
        std::fs::write(&file, "4 solar\n").expect("write");
        let replaced = SceneStore::open(&file, config.initial_scene);
        let line = scene_line(&replaced, &config);
        assert!(
            line.contains("scene demo-area (revision 5), replacing a retired scene saved in"),
            "{line}"
        );
        std::fs::remove_file(&file).ok();
    }

    #[test]
    fn a_bad_command_line_is_an_error_not_a_panic() {
        for args in [
            &["--bind", "nope"] as &[&str],
            &["--launcher", "kubernetes"],
            &["--nope"],
            &["--first-id", "x"],
            &["--scene", "nowhere"],
            &["--max-regions", "0"],
            &["--max-regions", "many"],
            &["--max-regions"],
            &["--webui", "nope"],
            &["--webui"],
        ] {
            assert!(parse(args).is_err(), "{args:?} must be rejected");
        }
        assert!(parse(&["--help"]).is_err(), "--help prints usage and exits");
    }

    #[test]
    fn the_web_ui_is_on_only_when_given_an_address() {
        let options = parse(&["--webui", "127.0.0.1:7980"]).expect("parses");
        let webui = options.webui.expect("the UI is on");
        assert_eq!(webui.bind(), "127.0.0.1:7980".parse().unwrap());
        assert!(!webui.token_required(), "loopback needs no token");

        // The environment turns it on too; a flag wins over it.
        let options =
            parse_with(&[], &[("SPACE_ORCHESTRATOR_WEBUI", "127.0.0.1:7981")]).expect("parses");
        assert_eq!(options.webui.expect("on").bind().port(), 7_981);
        let options = parse_with(
            &["--webui", "127.0.0.1:7980"],
            &[("SPACE_ORCHESTRATOR_WEBUI", "127.0.0.1:7981")],
        )
        .expect("parses");
        assert_eq!(options.webui.expect("on").bind().port(), 7_980);

        // An empty variable is unset, and a token alone starts nothing.
        let options = parse_with(
            &[],
            &[
                ("SPACE_ORCHESTRATOR_WEBUI", ""),
                ("SPACE_ORCHESTRATOR_WEBUI_TOKEN", TOKEN),
            ],
        )
        .expect("parses");
        assert!(options.webui.is_none());
        // ...but the flags without an address are a mistake worth reporting.
        assert!(parse(&["--webui-token", TOKEN]).is_err());
        assert!(parse(&["--webui-host", "ops.example.org"]).is_err());
    }

    /// Mutation this catches: starting an unauthenticated control surface on a
    /// reachable address because nobody thought to set a token.
    #[test]
    fn a_non_loopback_web_ui_does_not_start_without_a_token() {
        for open in ["0.0.0.0:7980", "[::]:7980", "192.168.1.5:7980"] {
            let err = parse(&["--webui", open]).expect_err(open);
            assert!(err.contains("without a token"), "{err}");
            let err = parse_with(&[], &[("SPACE_ORCHESTRATOR_WEBUI", open)]).expect_err(open);
            assert!(err.contains("without a token"), "{err}");
            let err = parse(&["--webui", open, "--webui-token", "short"]).expect_err(open);
            assert!(err.contains("too short"), "{err}");

            let by_flag = parse(&["--webui", open, "--webui-token", TOKEN]).expect("with a token");
            assert!(by_flag.webui.expect("on").token_required());
            let by_env = parse_with(
                &["--webui", open],
                &[("SPACE_ORCHESTRATOR_WEBUI_TOKEN", TOKEN)],
            )
            .expect("with a token from the environment");
            assert!(by_env.webui.expect("on").token_required());
        }
    }

    /// Mutation this catches: a loopback bind plus a public name starting
    /// without a token ("no token: loopback only" — behind a proxy that serves
    /// it to everyone).
    #[test]
    fn a_public_host_name_does_not_start_without_a_token() {
        let err = parse(&[
            "--webui",
            "127.0.0.1:7980",
            "--webui-host",
            "ops.example.org",
        ])
        .expect_err("a public name needs a token");
        assert!(err.contains("without a token"), "{err}");
        let err = parse_with(
            &["--webui", "127.0.0.1:7980"],
            &[("SPACE_ORCHESTRATOR_WEBUI_HOSTS", "ops.example.org")],
        )
        .expect_err("from the environment too");
        assert!(err.contains("without a token"), "{err}");

        let ok = parse(&[
            "--webui",
            "127.0.0.1:7980",
            "--webui-host",
            "ops.example.org:8443",
            "--webui-token",
            TOKEN,
        ])
        .expect("with a token, and a pinned port");
        let debug = format!("{:?}", ok.webui.expect("on"));
        assert!(
            debug.contains("ops.example.org") && debug.contains("8443"),
            "{debug}"
        );
    }

    #[test]
    fn the_token_can_come_from_a_file_and_hosts_from_a_list() {
        let path = std::env::temp_dir().join(format!(
            "space-orch-webui-token-{}-{}.txt",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&path, format!("{TOKEN}\n")).expect("write the token file");
        let file = path.to_str().expect("a UTF-8 temp path");

        let by_flag = parse(&["--webui", "0.0.0.0:7980", "--webui-token-file", file])
            .expect("the file's token is long enough");
        assert!(by_flag.webui.expect("on").token_required());
        let by_env = parse_with(
            &["--webui", "0.0.0.0:7980"],
            &[
                ("SPACE_ORCHESTRATOR_WEBUI_TOKEN_FILE", file),
                (
                    "SPACE_ORCHESTRATOR_WEBUI_HOSTS",
                    "ops.example.org, , Other.Example.Org",
                ),
            ],
        )
        .expect("parses");
        let debug = format!("{:?}", by_env.webui.expect("on"));
        assert!(
            debug.contains("ops.example.org") && debug.contains("other.example.org"),
            "{debug}"
        );
        assert!(!debug.contains(TOKEN), "the token is never printed");

        // Both at once is ambiguous; an empty or missing file is an error.
        assert!(parse(&[
            "--webui",
            "0.0.0.0:7980",
            "--webui-token",
            TOKEN,
            "--webui-token-file",
            file
        ])
        .is_err());
        std::fs::write(&path, "\n").expect("empty the token file");
        let err = parse(&["--webui", "0.0.0.0:7980", "--webui-token-file", file])
            .expect_err("an empty file is no token");
        assert!(err.contains("is empty"), "{err}");
        std::fs::remove_file(&path).ok();
        assert!(parse(&["--webui", "0.0.0.0:7980", "--webui-token-file", file]).is_err());

        // A host that is not a bare name is refused.
        assert!(parse(&[
            "--webui",
            "127.0.0.1:7980",
            "--webui-host",
            "https://ops.example.org"
        ])
        .is_err());
    }
}
