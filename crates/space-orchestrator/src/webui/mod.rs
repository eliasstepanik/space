//! The orchestrator's built-in web UI.
//!
//! An HTTP/1.1 server inside the orchestrator process, on tokio's
//! `TcpListener`, with pages rendered by Rust code: no script, no wasm, no
//! template files, no HTTP crate. An operator opens it in a browser, sees what
//! the orchestrator knows ([`crate::snapshot::Snapshot`]) and can ask for the
//! things the run loop can safely do ([`crate::ops::OperatorCommand`]).
//!
//! ```text
//!   space-orchestrator --webui 127.0.0.1:7980          # then open http://127.0.0.1:7980/
//! ```
//!
//! # Routes
//!
//! | method | path | what |
//! |---|---|---|
//! | `GET` | `/` | the dashboard; `?refresh=N` sets the auto-reload interval in seconds, `?refresh=0` switches it off |
//! | `GET` | `/api/state` | the same state as JSON ([`json`]) |
//! | `POST` | `/action/split` | `shard`, `octant` (`0`..`7` or `auto`), optional `epoch` |
//! | `POST` | `/action/merge` | `shard`, optional `epoch` |
//! | `POST` | `/action/autoscaler` | `state` = `pause` or `resume` |
//! | `POST` | `/action/push-map` | re-send the committed map to every shard |
//! | `POST` | `/action/scene` | `scene` = a name of the scene catalogue (`demo-area`, `materials`, `galaxy`): switch every shard to it |
//!
//! # A scene switch answers when the shards have it
//!
//! `/action/scene` does not redirect the moment the orchestrator has accepted
//! the change: it first waits — for at most [`Limits::scene_confirm_wait`], by
//! reading the snapshots the run loop publishes anyway, never by asking the
//! loop — until every connected shard has confirmed the new revision. So the
//! page the browser lands on says *All 2 connected shards switched to galaxy
//! (revision 4)*, and the plain-text answer says the same with an
//! `X-Space-Confirmed: 2/2` header. If the wait runs out it says *Switching to
//! galaxy (revision 4): waiting for shard 2 …*, and the dashboard reloads every
//! second — whatever `?refresh=` says, also "off" — until nobody is waited
//! for.
//!
//! A request for the scene that is already running is answered *already
//! running galaxy (revision 4); nothing to do* without going to the run loop at
//! all, so it can be told from a switch at a glance and never runs into the
//! spacing between actions. The page cannot send one in the ordinary way — the
//! button of the running scene is disabled, and no control holds a choice a
//! reload could reset (see [`page`]) — only as the second half of a double
//! click, or from a tab that has not reloaded since somebody else switched.
//! If a shard has not confirmed that scene yet (the first half of the double
//! click was the switch), this answer waits for the confirmations too, so
//! both halves report the same count. And a request that repeats a switch made
//! within the last [`SCENE_TWIN_WINDOW_TICKS`] is redirected to *that switch's*
//! banner: the browser shows the answer to the second click, and "nothing to
//! do" would read as if the button had done nothing.
//!
//! Every `POST` takes an `application/x-www-form-urlencoded` body that must
//! carry the form token (`csrf`), and answers `303 See Other` back to the
//! dashboard, which shows whether the request was accepted or why it was
//! refused. The plain-text body and the `X-Space-Result` header of that `303`
//! say the same, for scripts.
//!
//! Actions are carried out **one at a time, a moment apart**
//! ([`Limits::min_action_spacing`]): a second one that arrives while one is
//! being carried out, or right behind it, is answered `429` and changes
//! nothing. Each action costs the control loop an event-log entry, a line on
//! standard output and a fresh snapshot, so without the limit a looping script
//! (or a held-down key) could churn the log's history and keep the loop busy.
//!
//! # It cannot stall or crash the control loop
//!
//! The server holds an [`OpsHandle`], not the orchestrator. Rendering reads a
//! shared, immutable snapshot the loop published earlier; an action is offered
//! to a bounded queue with `try_send` and its answer awaited with a timeout.
//! So a request that is slow, huge, malformed or hostile is this module's
//! problem alone: each connection runs in its own task with a read deadline, a
//! write deadline and hard size limits ([`http`]), the number of connections
//! is capped (globally and per client address, [`Limits`]), and a panic in a
//! connection task would end that task only.
//!
//! # Who may use it
//!
//! * The UI is **off** unless it is given an address (`--webui`).
//! * On a loopback address it needs no credentials — **as long as nothing but
//!   this machine can reach it** (see "Without a token" below). On any other
//!   address [`WebUiConfig::new`] refuses to build a configuration without a
//!   token of at least [`auth::MIN_TOKEN_LEN`] characters, and every request
//!   must then present it as the password of HTTP Basic credentials (any user
//!   name).
//! * Plain HTTP only — this project has no TLS in-tree. On another host, reach
//!   it through an SSH tunnel or a TLS-terminating proxy; Basic credentials
//!   over plain HTTP on an open network are readable by anyone on the path.
//! * A request whose `Host` is neither an IP literal, `localhost`, nor a name
//!   listed with `--webui-host` is refused (DNS rebinding). A listed name may
//!   pin its port (`--webui-host ops.example.org:8443`); the scheme cannot be
//!   checked, because behind a TLS proxy this server never sees it.
//! * A state-changing request must come from the UI's own page: its `Origin`
//!   (when present) has to name this server, `Sec-Fetch-Site` (when present)
//!   has to be `same-origin`, and it has to carry the per-process form token.
//!   See [`auth`].
//! * Every response forbids scripts, framing and sniffing
//!   ([`http::SECURITY_HEADERS`], [`http::Response::content_security_policy`]),
//!   and every dynamic value in the page is escaped ([`page`]).
//!
//! # Without a token
//!
//! "Loopback, so no token" is a statement about who can reach the socket, and
//! the bind address alone does not prove it: a reverse proxy on the same
//! machine forwards the whole internet to `127.0.0.1`. Three rules keep a
//! tokenless UI from being published by accident:
//!
//! 1. `--webui-host` (a public name) **requires a token**: naming a public name
//!    says the UI is meant to be reached from elsewhere
//!    ([`WebUiConfig::new`] refuses the combination).
//! 2. Without a token the `Host` must itself be loopback — `localhost`,
//!    `127.0.0.0/8` or `[::1]` ([`auth::loopback_host`]) — not just any IP
//!    literal.
//! 3. Without a token a request carrying `Forwarded`, `X-Forwarded-For`,
//!    `X-Forwarded-Host`, `X-Forwarded-Proto`, `X-Real-IP` or `Via` is refused
//!    with a `403` that says to configure a token: those headers are a proxy's
//!    fingerprints ([`auth::proxy_header`]).
//!
//! **What this does not catch:** a proxy that adds none of those headers *and*
//! rewrites `Host` to a loopback address (a bare nginx `proxy_pass
//! http://127.0.0.1:7980;` does exactly that) looks like a local browser, and a
//! tokenless UI behind it is open to whoever can reach the proxy — they can
//! read `/api/state`, which includes the form token, and post actions. The
//! rules above are tripwires for the common mistakes; the protection is a
//! token. **Never put a proxy in front of a web UI that has no token.**

pub mod auth;
pub mod http;
pub mod json;
pub mod page;

use std::collections::HashMap;
use std::io;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use space_protocol::shard::ShardId;
use space_protocol::Scene;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Semaphore;

use self::auth::{AllowedHost, Auth, MIN_TOKEN_LEN};
use self::http::{field, parse_urlencoded, HttpError, Request, Response, MAX_BODY};
use crate::config::{LauncherKind, TICK_HZ};
use crate::events::EventKind;
use crate::ops::{OperatorCommand, OpsHandle};

/// Seconds between automatic reloads of the dashboard unless `?refresh=` says
/// otherwise.
pub const DEFAULT_REFRESH_SECS: u32 = 3;

/// The longest reload interval `?refresh=` accepts, in seconds.
pub const MAX_REFRESH_SECS: u32 = 3_600;

/// How long the outcome of an operator request stays on top of the dashboard,
/// in ticks (one minute). It remains in the event log after that.
pub const NOTICE_TICKS: u64 = 60 * TICK_HZ;

/// How long the accept loop pauses after a failed `accept` before trying again.
const ACCEPT_RETRY_DELAY: Duration = Duration::from_millis(50);

/// The server's resource limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Connections served at once; one more is closed on arrival.
    pub max_connections: usize,
    /// Connections served at once from one client address — an IPv4 address,
    /// or an IPv6 `/64` (one subscriber) — so one host cannot take every slot.
    pub max_connections_per_ip: usize,
    /// How long a client has to deliver the request line and headers. Shorter
    /// than [`Self::read_timeout`]: a connection that has not said what it
    /// wants is the cheapest way to hold a slot.
    pub head_timeout: Duration,
    /// How long a client has to deliver its whole request.
    pub read_timeout: Duration,
    /// How long a client has to take the response off the socket.
    pub write_timeout: Duration,
    /// How long a request with a *wrong* token is held before it is answered.
    /// With the connection caps this bounds how fast tokens can be guessed.
    pub wrong_token_delay: Duration,
    /// The least time between the end of one operator action and the start of
    /// the next (and only one is carried out at a time). An action that comes
    /// sooner is answered `429`. Long enough to stop a loop, short enough that
    /// a person clicking two buttons does not notice.
    pub min_action_spacing: Duration,
    /// How long an accepted scene switch waits for every connected shard to
    /// confirm it before it answers anyway (see the module docs). A shard
    /// confirms within a tick or two, so this only ever runs out when one is
    /// stuck — and then the answer names it.
    pub scene_confirm_wait: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_connections: 64,
            max_connections_per_ip: 8,
            head_timeout: Duration::from_secs(3),
            read_timeout: Duration::from_secs(10),
            write_timeout: Duration::from_secs(10),
            wrong_token_delay: Duration::from_millis(500),
            min_action_spacing: Duration::from_millis(250),
            scene_confirm_wait: Duration::from_millis(1_500),
        }
    }
}

/// How long a scene request that found another request in flight waits to see
/// whether that one was its own twin (a double click) before it is answered
/// `429` like any other hurried request.
const SCENE_TWIN_GRACE: Duration = Duration::from_millis(300);

/// For how long after a scene change a request for that same scene is taken
/// for the second half of a double click on the button that made the change
/// (two seconds): its redirect then shows the switch's banner instead of
/// "already running; nothing to do".
pub const SCENE_TWIN_WINDOW_TICKS: u64 = 2 * TICK_HZ;

/// How the web UI is set up. Built with [`WebUiConfig::new`], which is where
/// the "no open bind without a token" rule lives.
#[derive(Clone)]
pub struct WebUiConfig {
    bind: SocketAddr,
    token: Option<String>,
    allowed_hosts: Vec<AllowedHost>,
    limits: Limits,
    started: Instant,
}

impl core::fmt::Debug for WebUiConfig {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The token never reaches a log through `{:?}`.
        f.debug_struct("WebUiConfig")
            .field("bind", &self.bind)
            .field("token", &self.token.as_ref().map(|_| "<set>"))
            .field("allowed_hosts", &self.allowed_hosts)
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

impl WebUiConfig {
    /// A configuration serving on `bind`.
    ///
    /// `token` (an empty one counts as none) is the password every request
    /// must present. `allowed_hosts` are extra names the server may be
    /// addressed by in `Host` and `Origin`, besides IP literals and
    /// `localhost` — the public name of a proxy in front of it, as `name` or
    /// `name:port` ([`AllowedHost`]).
    ///
    /// # Errors
    ///
    /// * `bind` is not a loopback address and there is no token, or the token
    ///   is shorter than [`MIN_TOKEN_LEN`]: an unauthenticated control surface
    ///   must not be reachable from another machine.
    /// * `allowed_hosts` is not empty and there is no token: a public name
    ///   means the UI is reached through something that listens publicly, so
    ///   "it is bound to loopback" no longer means "only this machine".
    /// * an entry of `allowed_hosts` is not a host name with an optional port.
    pub fn new(
        bind: SocketAddr,
        token: Option<String>,
        allowed_hosts: Vec<String>,
    ) -> Result<Self, String> {
        let token = token.filter(|token| !token.is_empty());
        if !bind.ip().is_loopback() {
            match &token {
                None => {
                    return Err(format!(
                        "the web UI will not listen on {bind}, which is not a loopback address, \
                         without a token: set one (SPACE_ORCHESTRATOR_WEBUI_TOKEN), or bind \
                         127.0.0.1 and reach it through an SSH tunnel"
                    ));
                }
                Some(token) if token.len() < MIN_TOKEN_LEN => {
                    return Err(format!(
                        "the web UI token is too short for the non-loopback address {bind}: \
                         use at least {MIN_TOKEN_LEN} characters"
                    ));
                }
                Some(_) => {}
            }
        }
        let mut hosts = Vec::with_capacity(allowed_hosts.len());
        for host in allowed_hosts {
            hosts.push(AllowedHost::parse(&host)?);
        }
        if token.is_none() {
            if let Some(public) = hosts.first() {
                return Err(format!(
                    "the web UI will not answer to the public name {public} without a token: a \
                     name it is reached by from elsewhere means it is not loopback-only, \
                     whatever address it is bound to. Set a token \
                     (SPACE_ORCHESTRATOR_WEBUI_TOKEN), or drop --webui-host and reach it as \
                     127.0.0.1 through an SSH tunnel"
                ));
            }
        }
        Ok(Self {
            bind,
            token,
            allowed_hosts: hosts,
            limits: Limits::default(),
            started: Instant::now(),
        })
    }

    /// Use `limits` instead of the defaults.
    #[must_use]
    pub fn with_limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    /// Report uptime from `started` (the process's start) instead of from when
    /// this configuration was built.
    #[must_use]
    pub fn with_started(mut self, started: Instant) -> Self {
        self.started = started;
        self
    }

    /// The address to listen on.
    #[must_use]
    pub fn bind(&self) -> SocketAddr {
        self.bind
    }

    /// Whether requests must present a token.
    #[must_use]
    pub fn token_required(&self) -> bool {
        self.token.is_some()
    }
}

/// What the web server knows that the orchestrator's snapshot does not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerInfo {
    /// The crate version (`CARGO_PKG_VERSION`).
    pub version: &'static str,
    /// A build identifier, when the binary was compiled with
    /// `SPACE_BUILD_SHA` set (e.g. the commit an image was built from).
    pub build: Option<&'static str>,
    /// `"debug"` or `"release"`.
    pub profile: &'static str,
    /// The platform the binary runs on (`linux/x86_64`).
    pub platform: String,
    /// The wire protocol version this build speaks.
    pub protocol_version: u32,
    /// How long the orchestrator has been running.
    pub uptime: Duration,
    /// The wall-clock time, in seconds since the Unix epoch.
    pub now_unix: u64,
    /// The address the web UI is bound to.
    pub bind: SocketAddr,
    /// Whether requests must present a token.
    pub token_required: bool,
}

/// `kind` as the word the `--launcher` flag takes.
#[must_use]
pub const fn launcher_label(kind: LauncherKind) -> &'static str {
    match kind {
        LauncherKind::Local => "local",
        LauncherKind::Process => "process",
        LauncherKind::Docker => "docker",
    }
}

/// A request that is answered with an error instead of a page.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Refusal {
    status: u16,
    message: String,
    headers: Vec<(&'static str, String)>,
}

impl Refusal {
    fn new(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            headers: Vec::new(),
        }
    }

    fn with_header(mut self, name: &'static str, value: impl Into<String>) -> Self {
        self.headers.push((name, value.into()));
        self
    }

    /// The response, in the form the caller of `path` can read best: JSON for
    /// the API, a small page (with a way back) for everything a browser asks.
    fn into_response(self, path: &str) -> Response {
        let mut response = if path.starts_with("/api/") {
            let mut body = String::from("{\"status\":");
            body.push_str(&self.status.to_string());
            body.push_str(",\"error\":");
            json::write_string(&mut body, &self.message);
            body.push_str("}\n");
            Response::new(self.status, "application/json; charset=utf-8", body)
        } else {
            let nonce = auth::random_token();
            // Relative, so it also works behind a path prefix.
            let home = if path.starts_with("/action/") {
                "../"
            } else {
                "./"
            };
            let body = page::error_page(self.status, &self.message, &nonce, home);
            let mut response = Response::new(self.status, "text/html; charset=utf-8", body);
            response.style_nonce = Some(nonce);
            response
        };
        response.headers = self.headers;
        response
    }
}

impl From<HttpError> for Refusal {
    fn from(error: HttpError) -> Self {
        Self::new(error.status, error.message)
    }
}

/// Lets one operator action through at a time, a moment apart (see
/// [`Limits::min_action_spacing`]).
#[derive(Debug, Default)]
struct ActionGate {
    state: Mutex<GateState>,
}

#[derive(Debug, Default)]
struct GateState {
    /// An action is with the run loop right now.
    in_flight: bool,
    /// When the last action finished.
    last_finished: Option<Instant>,
}

/// The right to carry out one action; gives the gate back when dropped (also
/// when the request's task is cancelled).
#[derive(Debug)]
struct ActionPermit<'a> {
    gate: &'a ActionGate,
}

impl ActionGate {
    fn lock(&self) -> std::sync::MutexGuard<'_, GateState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Take the gate, unless an action is in flight or the last one finished
    /// less than `spacing` ago.
    fn enter(&self, spacing: Duration) -> Option<ActionPermit<'_>> {
        let mut state = self.lock();
        let too_soon = state
            .last_finished
            .is_some_and(|finished| finished.elapsed() < spacing);
        if state.in_flight || too_soon {
            return None;
        }
        state.in_flight = true;
        Some(ActionPermit { gate: self })
    }
}

impl Drop for ActionPermit<'_> {
    fn drop(&mut self) {
        let mut state = self.gate.lock();
        state.in_flight = false;
        state.last_finished = Some(Instant::now());
    }
}

/// The request handler: everything about a request that does not need a
/// socket. One per server, shared by its connection tasks.
struct Service {
    config: WebUiConfig,
    ops: OpsHandle,
    /// The per-process token every form carries.
    form_token: String,
    /// The address actually bound (`:0` resolved).
    bound: SocketAddr,
    /// One operator action at a time, a moment apart.
    gate: ActionGate,
}

impl Service {
    fn info(&self) -> ServerInfo {
        ServerInfo {
            version: env!("CARGO_PKG_VERSION"),
            build: option_env!("SPACE_BUILD_SHA"),
            profile: if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            },
            platform: format!("{}/{}", std::env::consts::OS, std::env::consts::ARCH),
            protocol_version: space_protocol::PROTOCOL_VERSION,
            uptime: self.config.started.elapsed(),
            now_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |since| since.as_secs()),
            bind: self.bound,
            token_required: self.config.token.is_some(),
        }
    }

    /// Answer one request.
    async fn handle(&self, request: &Request) -> Response {
        match self.respond(request).await {
            Ok(response) => response,
            Err(refusal) => refusal.into_response(request.path()),
        }
    }

    async fn respond(&self, request: &Request) -> Result<Response, Refusal> {
        // 1. Was this request meant for this server? (DNS rebinding.)
        let host = request
            .sole_header("host")?
            .ok_or_else(|| Refusal::new(400, "the request has no Host header"))?;
        if self.config.token.is_some() {
            if !auth::host_allowed(host, &self.config.allowed_hosts) {
                return Err(Refusal::new(
                    403,
                    "this server does not answer to that Host name. Open it by IP address, or \
                     start the orchestrator with --webui-host <name>.",
                ));
            }
        } else {
            // No token: the only thing that makes this safe is that nobody but
            // this machine can reach it (see the module docs, "Without a
            // token"). A Host that is not loopback, or a proxy's headers, say
            // otherwise.
            if !auth::loopback_host(host) {
                return Err(Refusal::new(
                    403,
                    "this web UI has no token, so it only answers on a loopback address \
                     (localhost, 127.0.0.1 or [::1]). To use it by another address or name, \
                     give it a token (SPACE_ORCHESTRATOR_WEBUI_TOKEN).",
                ));
            }
            let names = request.headers.iter().map(|(name, _)| name.as_str());
            if let Some(header) = auth::proxy_header(names) {
                return Err(Refusal::new(
                    403,
                    format!(
                        "this request came through a proxy (it carries a {header} header), and \
                         this web UI has no token: anyone who can reach the proxy could use it. \
                         Give the web UI a token (SPACE_ORCHESTRATOR_WEBUI_TOKEN) before putting \
                         a proxy in front of it."
                    ),
                ));
            }
        }
        // 2. Is the caller an operator?
        if let Some(token) = &self.config.token {
            let challenge = || {
                Refusal::new(
                    401,
                    "this web UI needs its token: any user name, the token as the password",
                )
                .with_header(
                    "WWW-Authenticate",
                    format!("Basic realm=\"{}\", charset=\"UTF-8\"", auth::REALM),
                )
            };
            match auth::check_basic(request.sole_header("authorization")?, token) {
                Auth::Granted => {}
                Auth::Missing => return Err(challenge()),
                Auth::Wrong => {
                    tokio::time::sleep(self.config.limits.wrong_token_delay).await;
                    return Err(challenge());
                }
            }
        }
        // 3. What is asked for?
        match (request.method.as_str(), request.path()) {
            ("GET" | "HEAD", "/") => self.dashboard(request),
            ("GET" | "HEAD", "/api/state") => Ok(self.state()),
            ("POST", path) if is_action(path) => self.action(request, host).await,
            (_, "/" | "/api/state") => Err(Refusal::new(405, "this address only answers GET")
                .with_header("Allow", "GET, HEAD")),
            (_, path) if is_action(path) => {
                Err(Refusal::new(405, "this address only answers POST")
                    .with_header("Allow", "POST"))
            }
            _ => Err(Refusal::new(404, "there is nothing at this address")),
        }
    }

    /// `GET /`: the dashboard.
    fn dashboard(&self, request: &Request) -> Result<Response, Refusal> {
        let query = parse_urlencoded(request.query())?;
        let refresh = refresh_of(&query);
        let snapshot = self.ops.snapshot();
        // The outcome of the operator request the redirect named, while it is
        // recent. Only the entry's *number* travels in the URL, so a link
        // cannot make the page say anything the log does not hold.
        let notice = field(&query, "notice")
            .and_then(|seq| seq.parse::<u64>().ok())
            .and_then(|seq| snapshot.events.iter().find(|event| event.seq == seq))
            .filter(|event| {
                event.kind == EventKind::Operator
                    && snapshot.tick.saturating_sub(event.tick) <= NOTICE_TICKS
            });
        // The redirect of a scene request: two numbers. What the banner says
        // is read from the snapshot (see `page::SceneNotice`).
        let scene_notice = field(&query, "switched")
            .and_then(|revision| revision.parse::<u64>().ok())
            .map(|revision| page::SceneNotice {
                revision,
                unchanged: field(&query, "same") == Some("1"),
            });
        let nonce = auth::random_token();
        let body = page::dashboard(&page::PageInput {
            snapshot: &snapshot,
            info: &self.info(),
            form_token: &self.form_token,
            style_nonce: &nonce,
            refresh,
            notice,
            scene_notice,
        });
        let mut response = Response::new(200, "text/html; charset=utf-8", body);
        response.style_nonce = Some(nonce);
        Ok(response)
    }

    /// `GET /api/state`: the same state as JSON.
    fn state(&self) -> Response {
        let snapshot = self.ops.snapshot();
        let body = json::state_json(&snapshot, &self.info(), &self.form_token);
        Response::new(200, "application/json; charset=utf-8", body)
    }

    /// `POST /action/…`: check that the UI's own page sent it, hand the
    /// command to the run loop, and send the browser back to the dashboard.
    async fn action(&self, request: &Request, host: &str) -> Result<Response, Refusal> {
        const CROSS_SITE: &str = "this request did not come from the web UI's own page (its \
                                  Origin is another site), so it was not carried out";
        if let Some(site) = request.sole_header("sec-fetch-site")? {
            if !auth::fetch_site_allowed(site) {
                return Err(Refusal::new(403, CROSS_SITE));
            }
        }
        if let Some(origin) = request.sole_header("origin")? {
            if !auth::origin_allowed(origin, host, &self.config.allowed_hosts) {
                return Err(Refusal::new(403, CROSS_SITE));
            }
        }
        let media = request
            .header("content-type")
            .and_then(|value| value.split(';').next())
            .map_or("", str::trim);
        if !media.eq_ignore_ascii_case("application/x-www-form-urlencoded") {
            return Err(Refusal::new(
                415,
                "an action is posted as application/x-www-form-urlencoded",
            ));
        }
        let body = std::str::from_utf8(&request.body)
            .map_err(|_| Refusal::new(400, "the form is not UTF-8"))?;
        let fields = parse_urlencoded(body)?;
        // The token a page served by *this process* carries. Checked for every
        // action, whatever the Origin said: a client that sends no Origin at
        // all (a script) has nothing else to show.
        let token = field(&fields, "csrf").unwrap_or("");
        if !auth::constant_time_eq(token.as_bytes(), self.form_token.as_bytes()) {
            return Err(Refusal::new(
                403,
                "the form token is missing or out of date (the orchestrator was probably \
                 restarted since the page was loaded). Reload the dashboard and try again.",
            ));
        }
        let command = command_of(request.path(), &fields)?;
        let refresh = refresh_of(&fields);
        // A request for the scene that is already running asks the control
        // loop for nothing: it is answered from the published snapshot, says
        // so in as many words, and does not count against the spacing below.
        // (This is what the second half of a double click is, and what a tab
        // that has not reloaded since somebody else's switch posts.)
        let asked_scene = match &command {
            OperatorCommand::SetScene { scene } => Some(*scene),
            _ => None,
        };
        if let Some(scene) = asked_scene {
            let snapshot = self.ops.snapshot();
            if snapshot.scene.scene == scene {
                return Ok(self.scene_already_running(snapshot, refresh).await);
            }
        }
        // One action at a time, a moment apart: everything up to here was
        // free for the control loop, from here on it is not. The permit is
        // held until the loop has answered.
        let Some(permit) = self.gate.enter(self.config.limits.min_action_spacing) else {
            // A scene request may have lost the race against its own twin:
            // once that one has been carried out there is nothing left to do,
            // and saying "nothing was changed" would be wrong.
            if let Some(scene) = asked_scene {
                let snapshot = self
                    .ops
                    .wait_until(SCENE_TWIN_GRACE, |s| s.scene.scene == scene)
                    .await;
                if snapshot.scene.scene == scene {
                    return Ok(self.scene_already_running(snapshot, refresh).await);
                }
            }
            return Err(Refusal::new(
                429,
                "another operator request was carried out a moment ago (they are taken \
                 one at a time, a moment apart). Nothing was changed: look at the \
                 dashboard, then send it again.",
            )
            .with_header("Retry-After", "1"));
        };
        let reply = self
            .ops
            .request(command)
            .await
            .map_err(|error| Refusal::new(503, error.to_string()))?;
        // The loop has done its part; what follows only reads snapshots.
        drop(permit);
        let verdict = if reply.accepted {
            "accepted"
        } else {
            "refused"
        };
        // Relative to /action/<name>, so it also works behind a path prefix.
        // Only numbers go into it.
        let mut location = format!("../?refresh={refresh}&notice={}", reply.event);
        let mut text = format!("{verdict}: {}", reply.message);
        let mut confirmed = None;
        if let Some(target) = reply.scene_changed {
            // The scene changed. Do not send the browser to a page that was
            // rendered before any shard could answer: wait (bounded, and
            // without asking the loop for anything) until every connected
            // shard has confirmed, then say how far it got.
            let snapshot = self
                .ops
                .wait_until(self.config.limits.scene_confirm_wait, |s| {
                    s.scene.revision >= target.revision && s.scene_progress().settled()
                })
                .await;
            let progress = snapshot.scene_progress();
            text = format!("{text}. {}", progress.sentence());
            location = format!("{location}&switched={}", target.revision);
            confirmed = Some(confirmed_header(&progress));
        }
        let mut response = Response::text(303, text)
            .with_header("Location", location)
            .with_header("X-Space-Result", verdict);
        if let Some(confirmed) = confirmed {
            response = response.with_header("X-Space-Confirmed", confirmed);
        }
        Ok(response)
    }

    /// Answer a request for the scene `seen` shows as running (see
    /// [`scene_unchanged`]). When a connected shard has not confirmed that
    /// scene yet — the request is the second half of a double click and the
    /// first half was the switch, an instant ago — the answer waits for the
    /// confirmations exactly as the switch itself does: the browser follows
    /// the *last* answer it gets, and that one must not say "0 of 1" about a
    /// switch that is a few milliseconds from done. Reads snapshots only.
    async fn scene_already_running(
        &self,
        seen: Arc<crate::snapshot::Snapshot>,
        refresh: u32,
    ) -> Response {
        let asked = seen.scene.current();
        let latest = self
            .ops
            .wait_until(self.config.limits.scene_confirm_wait, |s| {
                s.scene.current() != asked || s.scene_progress().settled()
            })
            .await;
        // Changed again while this waited (another operator): what was seen
        // was true when asked, and is what the answer is about.
        let shown = if latest.scene.current() == asked {
            latest
        } else {
            seen
        };
        scene_unchanged(&shown, refresh)
    }

    /// Read one request from `stream`, answer it, close.
    async fn serve_connection(&self, mut stream: TcpStream) {
        let limits = self.config.limits;
        let read = tokio::time::timeout(
            limits.read_timeout,
            http::read_request_within(&mut stream, MAX_BODY, Some(limits.head_timeout)),
        )
        .await;
        let (response, head_only) = match read {
            Ok(Ok(Some(request))) => {
                let head_only = request.method == "HEAD";
                (self.handle(&request).await, head_only)
            }
            // Closed without a word: nothing to answer.
            Ok(Ok(None)) => return,
            Ok(Err(error)) => (Response::text(error.status, error.message), false),
            Err(_) => (
                Response::text(408, "the request did not arrive in time"),
                false,
            ),
        };
        // Dropping the stream closes it, whether or not the write finished.
        let _ = http::write_response(
            &mut stream,
            &response.to_bytes(head_only),
            limits.write_timeout,
        )
        .await;
    }
}

/// `confirmed/connected`: how many of the shards that hold a control link have
/// confirmed the current scene (the `X-Space-Confirmed` header).
fn confirmed_header(progress: &crate::snapshot::SceneProgress) -> String {
    format!("{}/{}", progress.confirmed.len(), progress.connected())
}

/// The answer to a request for the scene that is already running: nothing
/// was sent to the run loop, nothing changed, and the answer says exactly
/// that — plus who is still being waited for, if anyone is, or that no shard
/// is connected at all (the same additions the dashboard's banner makes).
fn scene_unchanged(snapshot: &crate::snapshot::Snapshot, refresh: u32) -> Response {
    let progress = snapshot.scene_progress();
    let mut text = format!(
        "accepted: already running {} (revision {}); nothing to do",
        progress.scene, progress.revision
    );
    if !progress.settled() || progress.connected() == 0 {
        text = format!("{text}. {}", progress.sentence());
    }
    // The second half of a double click on the button that switched: the
    // browser shows the answer to *this* request, and what the person did was
    // switch — so they land on the switch's banner ("All 2 connected shards
    // switched to galaxy"), not on "nothing to do".
    let just_switched = snapshot
        .scene
        .changed_tick
        .is_some_and(|at| snapshot.tick.saturating_sub(at) <= SCENE_TWIN_WINDOW_TICKS);
    let same = if just_switched { "" } else { "&same=1" };
    Response::text(303, text)
        .with_header(
            "Location",
            format!("../?refresh={refresh}&switched={}{same}", progress.revision),
        )
        .with_header("X-Space-Result", "accepted")
        .with_header("X-Space-Confirmed", confirmed_header(&progress))
}

/// The addresses a form can be posted to (see the module docs, "Routes").
const ACTIONS: [&str; 5] = [
    "/action/split",
    "/action/merge",
    "/action/autoscaler",
    "/action/push-map",
    "/action/scene",
];

/// Whether `path` is one of the [`ACTIONS`].
fn is_action(path: &str) -> bool {
    ACTIONS.contains(&path)
}

/// The reload interval a query string or form asks for.
fn refresh_of(fields: &[(String, String)]) -> u32 {
    field(fields, "refresh")
        .and_then(|value| value.parse::<u32>().ok())
        .map_or(DEFAULT_REFRESH_SECS, |seconds| {
            seconds.min(MAX_REFRESH_SECS)
        })
}

/// The command a form posted to `path` asks for.
fn command_of(path: &str, fields: &[(String, String)]) -> Result<OperatorCommand, Refusal> {
    let shard = || -> Result<ShardId, Refusal> {
        field(fields, "shard")
            .and_then(|value| value.parse::<u32>().ok())
            .map(ShardId::new)
            .filter(|shard| *shard != ShardId::NONE)
            .ok_or_else(|| Refusal::new(400, "the form names no valid shard"))
    };
    let expected_epoch = match field(fields, "epoch") {
        None | Some("") => None,
        Some(value) => Some(
            value
                .parse::<u64>()
                .map_err(|_| Refusal::new(400, "the form's epoch is not a number"))?,
        ),
    };
    match path {
        "/action/split" => {
            let octant = match field(fields, "octant") {
                None | Some("" | "auto") => None,
                Some(value) => Some(
                    value
                        .parse::<u8>()
                        .ok()
                        .filter(|octant| *octant < 8)
                        .ok_or_else(|| {
                            Refusal::new(400, "the octant must be a number from 0 to 7, or auto")
                        })?,
                ),
            };
            Ok(OperatorCommand::Split {
                shard: shard()?,
                octant,
                expected_epoch,
            })
        }
        "/action/merge" => Ok(OperatorCommand::Merge {
            shard: shard()?,
            expected_epoch,
        }),
        "/action/autoscaler" => match field(fields, "state") {
            Some("pause") => Ok(OperatorCommand::PauseAutoscaler),
            Some("resume") => Ok(OperatorCommand::ResumeAutoscaler),
            _ => Err(Refusal::new(400, "the state must be pause or resume")),
        },
        "/action/push-map" => Ok(OperatorCommand::PushMap),
        // The name has to be one of the catalogue: an unknown one is refused
        // here, before anything reaches the run loop. (The name sent is not
        // echoed; the catalogue is.)
        "/action/scene" => field(fields, "scene")
            .and_then(Scene::from_name)
            .map(|scene| OperatorCommand::SetScene { scene })
            .ok_or_else(|| {
                Refusal::new(
                    400,
                    format!(
                        "the form names no known scene (the scenes are {})",
                        Scene::names()
                    ),
                )
            }),
        _ => Err(Refusal::new(404, "there is no such action")),
    }
}

/// One client address's share of the connection slots; gives its slot back
/// when dropped (also when the connection's task is cancelled or panics).
#[derive(Debug)]
struct AddressSlot {
    table: Arc<Mutex<HashMap<IpAddr, usize>>>,
    address: IpAddr,
}

impl AddressSlot {
    /// Take a slot for `address`, unless it already holds `limit` of them.
    fn claim(
        table: &Arc<Mutex<HashMap<IpAddr, usize>>>,
        address: IpAddr,
        limit: usize,
    ) -> Option<Self> {
        let address = slot_key(address);
        let mut held = table
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let count = held.entry(address).or_insert(0);
        if *count >= limit {
            return None;
        }
        *count += 1;
        Some(Self {
            table: Arc::clone(table),
            address,
        })
    }
}

/// What a client address counts as for [`Limits::max_connections_per_ip`].
///
/// An IPv4 client on a dual-stack socket is the same client as over IPv4. An
/// IPv6 client is its `/64`: that is what one subscriber is handed, so one
/// machine has 2⁶⁴ addresses to connect from and counting each on its own
/// would count nothing.
fn slot_key(address: IpAddr) -> IpAddr {
    match address.to_canonical() {
        IpAddr::V6(v6) => {
            let prefix = u128::from(v6) & !u128::from(u64::MAX);
            IpAddr::V6(std::net::Ipv6Addr::from(prefix))
        }
        v4 @ IpAddr::V4(_) => v4,
    }
}

impl Drop for AddressSlot {
    fn drop(&mut self) {
        let mut held = self
            .table
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(count) = held.get_mut(&self.address) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                held.remove(&self.address);
            }
        }
    }
}

/// The web UI, bound and ready to serve.
pub struct WebUi {
    listener: TcpListener,
    service: Arc<Service>,
}

impl core::fmt::Debug for WebUi {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Neither the token nor the form token is printed.
        f.debug_struct("WebUi")
            .field("bound", &self.service.bound)
            .field("config", &self.service.config)
            .finish_non_exhaustive()
    }
}

impl WebUi {
    /// Bind the configured address. `ops` is the handle
    /// [`crate::orchestrator::Orchestrator::attach_ops`] returned.
    ///
    /// # Errors
    ///
    /// [`io::Error`] when the address cannot be bound.
    pub async fn bind(config: WebUiConfig, ops: OpsHandle) -> io::Result<Self> {
        let listener = TcpListener::bind(config.bind).await?;
        let bound = listener.local_addr()?;
        Ok(Self {
            listener,
            service: Arc::new(Service {
                config,
                ops,
                form_token: auth::random_token(),
                bound,
                gate: ActionGate::default(),
            }),
        })
    }

    /// The address actually bound (`:0` resolves to a real port).
    #[must_use]
    pub fn local_addr(&self) -> SocketAddr {
        self.service.bound
    }

    /// One line for the startup log: where the UI is and who may use it.
    #[must_use]
    pub fn describe(&self) -> String {
        let bound = self.service.bound;
        if self.service.config.token.is_some() {
            format!("http://{bound}/ (token required: any user name, the token as the password)")
        } else {
            format!("http://{bound}/ (no token: loopback only)")
        }
    }

    /// Serve until the task is dropped. Each connection carries one request
    /// and runs in its own task.
    ///
    /// A failed `accept` is logged and the server keeps accepting. A
    /// connection that arrives when every slot (or every slot of its address)
    /// is taken is closed at once — it is not parked, so the accept loop never
    /// waits on a client.
    pub async fn serve(self) {
        let Self { listener, service } = self;
        let limits = service.config.limits;
        let slots = Arc::new(Semaphore::new(limits.max_connections));
        let per_address: Arc<Mutex<HashMap<IpAddr, usize>>> = Arc::default();
        loop {
            let (stream, peer) = match listener.accept().await {
                Ok(accepted) => accepted,
                Err(err) => {
                    eprintln!("space-orchestrator: web UI accept failed ({err}); still listening");
                    tokio::time::sleep(ACCEPT_RETRY_DELAY).await;
                    continue;
                }
            };
            let Ok(slot) = Arc::clone(&slots).try_acquire_owned() else {
                continue;
            };
            let Some(address_slot) =
                AddressSlot::claim(&per_address, peer.ip(), limits.max_connections_per_ip)
            else {
                continue;
            };
            let service = Arc::clone(&service);
            tokio::spawn(async move {
                service.serve_connection(stream).await;
                drop(address_slot);
                drop(slot);
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::json::tests::check_json;
    use super::*;
    use crate::control::ControlPlane;
    use crate::epoch::EpochPublisher;
    use crate::ids::{MemoryCounterSink, PersistentShardIdAllocator};
    use crate::launcher::{LaunchError, LaunchId, Launcher, Poll, ShardHandle, SpawnSpec};
    use crate::ops::OpsPort;
    use crate::orchestrator::Orchestrator;
    use crate::OrchestratorConfig;
    use space_protocol::shard::{ShardMap, ShardRegion};
    use space_protocol::{LoadReport, ShardHello};

    const TOKEN: &str = "a-long-enough-test-token";

    fn loopback() -> SocketAddr {
        SocketAddr::from(([127, 0, 0, 1], 7_980))
    }

    #[derive(Debug, Default)]
    struct NoLauncher;

    impl Launcher for NoLauncher {
        async fn spawn(&self, spec: SpawnSpec) -> Result<ShardHandle, LaunchError> {
            Ok(ShardHandle {
                launch: spec.launch,
                shard: None,
                endpoint: spec.control_addr,
                process: "none".into(),
            })
        }
        async fn kill(&self, _launch: LaunchId) -> Result<(), LaunchError> {
            Ok(())
        }
        async fn poll(&self, _launch: LaunchId) -> Result<Poll, LaunchError> {
            Ok(Poll::Alive)
        }
    }

    /// An orchestrator whose root is owned by shard 1 (registered and
    /// connected), with cluster name `cluster`.
    fn orchestrator(cluster: &str) -> Orchestrator<NoLauncher> {
        let config = OrchestratorConfig {
            cluster: cluster.to_owned(),
            ..OrchestratorConfig::default()
        };
        let map = ShardMap::new(
            1,
            vec![ShardRegion::new(
                config.root_node(),
                ShardId::new(1),
                SocketAddr::from(([127, 0, 0, 1], 7_001)),
            )],
        );
        let ids = PersistentShardIdAllocator::with_sink(2, Box::new(MemoryCounterSink::default()));
        let control = ControlPlane::new(
            config,
            ids,
            EpochPublisher::new(map),
            "peer-token-never-shown".into(),
        );
        let mut orch = Orchestrator::new(control, Arc::new(NoLauncher));
        orch.handle_shard_hello(&ShardHello {
            cluster: cluster.to_owned(),
            shard: ShardId::new(1),
            endpoint: SocketAddr::from(([127, 0, 0, 1], 7_001)),
            region: None,
            client_endpoint: None,
            secret: String::new(),
        })
        .expect("the root owner registers");
        orch.note_connected([ShardId::new(1)]);
        // The shard confirms the scene it was handed with its welcome, as a
        // real one does within a tick: the baseline is a settled cluster.
        let scene = orch.scene();
        orch.handle_scene_ack(&space_protocol::SceneAck {
            shard: ShardId::new(1),
            applied: scene,
        });
        orch
    }

    fn service_with(ops: OpsHandle, token: Option<&str>, hosts: &[&str]) -> Service {
        let config = WebUiConfig::new(
            loopback(),
            token.map(str::to_owned),
            hosts.iter().map(|h| (*h).to_owned()).collect(),
        )
        .expect("a loopback configuration")
        .with_limits(Limits {
            wrong_token_delay: Duration::from_millis(1),
            // These tests post actions back to back; the spacing between
            // actions has a test of its own.
            min_action_spacing: Duration::ZERO,
            // Long enough for the test loop's shard to confirm (it takes
            // 20 ms), short enough that a test of a stuck shard is quick.
            scene_confirm_wait: Duration::from_millis(400),
            ..Limits::default()
        });
        Service {
            config,
            ops,
            form_token: auth::random_token(),
            bound: loopback(),
            gate: ActionGate::default(),
        }
    }

    /// A service over a snapshot of `orchestrator(cluster)`; nothing answers
    /// its commands (the port is returned so the queue stays open).
    fn service(token: Option<&str>) -> (Service, OpsPort, OpsHandle) {
        let (handle, port) = crate::ops::channel(orchestrator("space-test").snapshot());
        (service_with(handle.clone(), token, &[]), port, handle)
    }

    fn get(target: &str) -> Request {
        Request {
            method: "GET".into(),
            target: target.into(),
            headers: vec![("host".into(), "127.0.0.1:7980".into())],
            body: Vec::new(),
        }
    }

    fn post(target: &str, body: &str) -> Request {
        Request {
            method: "POST".into(),
            target: target.into(),
            headers: vec![
                ("host".into(), "127.0.0.1:7980".into()),
                (
                    "content-type".into(),
                    "application/x-www-form-urlencoded".into(),
                ),
                ("origin".into(), "http://127.0.0.1:7980".into()),
            ],
            body: body.as_bytes().to_vec(),
        }
    }

    fn with(mut request: Request, name: &str, value: &str) -> Request {
        request.headers.retain(|(n, _)| n != name);
        request.headers.push((name.into(), value.into()));
        request
    }

    fn without(mut request: Request, name: &str) -> Request {
        request.headers.retain(|(n, _)| n != name);
        request
    }

    fn text(response: &Response) -> String {
        String::from_utf8(response.body.clone()).expect("UTF-8 body")
    }

    fn basic(password: &str) -> String {
        // "ops:" + password, encoded by hand (see auth's tests for the decoder).
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let raw = format!("ops:{password}");
        let mut out = String::from("Basic ");
        for chunk in raw.as_bytes().chunks(3) {
            let n = (u32::from(chunk[0]) << 16)
                | (u32::from(chunk.get(1).copied().unwrap_or(0)) << 8)
                | u32::from(chunk.get(2).copied().unwrap_or(0));
            for (i, shift) in [18u32, 12, 6, 0].into_iter().enumerate() {
                if i <= chunk.len() {
                    out.push(char::from(ALPHABET[((n >> shift) & 63) as usize]));
                } else {
                    out.push('=');
                }
            }
        }
        out
    }

    // ------------------------------------------------------------ the config

    /// Mutation this catches: serving an unauthenticated control surface on a
    /// reachable address.
    #[test]
    fn a_non_loopback_bind_without_a_token_is_refused() {
        for open in [
            "0.0.0.0:7980",
            "192.168.1.10:7980",
            "[::]:7980",
            "[2001:db8::1]:80",
        ] {
            let bind: SocketAddr = open.parse().expect("an address");
            let err = WebUiConfig::new(bind, None, vec![]).expect_err(open);
            assert!(err.contains("without a token"), "{err}");
            let err = WebUiConfig::new(bind, Some(String::new()), vec![]).expect_err(open);
            assert!(
                err.contains("without a token"),
                "an empty token is no token: {err}"
            );
            let err = WebUiConfig::new(bind, Some("short".into()), vec![]).expect_err(open);
            assert!(err.contains("too short"), "{err}");
            let ok = WebUiConfig::new(bind, Some(TOKEN.into()), vec![]).expect("with a token");
            assert!(ok.token_required());
        }
        for local in ["127.0.0.1:7980", "127.0.0.1:0", "[::1]:7980", "127.9.9.9:1"] {
            let bind: SocketAddr = local.parse().expect("an address");
            let config = WebUiConfig::new(bind, None, vec![]).expect(local);
            assert!(!config.token_required());
            // A token is honoured on loopback too, even a short one.
            assert!(WebUiConfig::new(bind, Some("short".into()), vec![])
                .expect(local)
                .token_required());
        }
    }

    #[test]
    fn allowed_hosts_are_bare_names_and_the_token_stays_out_of_debug() {
        let config = WebUiConfig::new(
            loopback(),
            Some(TOKEN.into()),
            vec![" Ops.Example.Org. ".into(), "other.example.org:8443".into()],
        )
        .expect("two names");
        let listed: Vec<String> = config
            .allowed_hosts
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(listed, vec!["ops.example.org", "other.example.org:8443"]);
        assert!(!format!("{config:?}").contains(TOKEN));
        for bad in [
            "",
            "http://ops.example.org",
            "ops.example.org:0",
            "ops.example.org:port",
            "a/b",
            "a b",
        ] {
            let err =
                WebUiConfig::new(loopback(), Some(TOKEN.into()), vec![bad.into()]).expect_err(bad);
            assert!(err.contains("is not a host name"), "{bad:?}: {err}");
        }
    }

    /// Mutation this catches: deciding "no token needed" from the bind address
    /// alone. A loopback bind with a public name is a UI behind a proxy on the
    /// same host — open to everyone who can reach the proxy.
    #[test]
    fn a_public_name_needs_a_token_even_on_a_loopback_bind() {
        for local in ["127.0.0.1:7980", "[::1]:7980"] {
            let bind: SocketAddr = local.parse().expect("an address");
            for name in ["ops.example.org", "ops.example.org:443"] {
                let err = WebUiConfig::new(bind, None, vec![name.into()]).expect_err(name);
                assert!(
                    err.contains("without a token") && err.contains("ops.example.org"),
                    "{err}"
                );
                let err = WebUiConfig::new(bind, Some(String::new()), vec![name.into()])
                    .expect_err("an empty token is no token");
                assert!(err.contains("without a token"), "{err}");
                let ok = WebUiConfig::new(bind, Some(TOKEN.into()), vec![name.into()])
                    .expect("with a token");
                assert!(ok.token_required());
            }
            // Without a public name nothing changes: loopback needs no token.
            assert!(WebUiConfig::new(bind, None, vec![]).is_ok());
        }
    }

    // ------------------------------------------------------------- the pages

    #[tokio::test]
    async fn the_dashboard_shows_the_cluster_the_map_and_the_shard() {
        let (service, _port, _) = service(None);
        let response = service.handle(&get("/")).await;
        assert_eq!(response.status, 200);
        assert_eq!(response.content_type, "text/html; charset=utf-8");
        let page = text(&response);
        for needle in [
            "<!doctype html>",
            "space-test",
            "epoch 1",
            "40:0:0:0",
            "shard 1",
            "127.0.0.1:7001",
            "connected",
            "no report yet",
            "autoscaler active",
            "<meta http-equiv=\"refresh\" content=\"3\">",
            "action=\"action/split\"",
            "name=\"csrf\" value=\"",
            // The Scene card: the current scene, and one button per scene.
            "<h2>Scene</h2>",
            "<span class=\"pill info\">demo-area</span>",
            "action=\"action/scene\"",
            "<button name=\"scene\" value=\"demo-area\" disabled",
            ">demo-area (current)</button>",
            ">Switch all shards to materials</button>",
            ">Switch all shards to galaxy</button>",
            "demo-area, revision 1 <span class=\"pill ok\">current</span>",
            "<span class=\"pill ok\">1 of 1 connected shards</span>",
        ] {
            assert!(page.contains(needle), "missing {needle:?}");
        }
        assert!(
            !page.contains("while a shard is switching scene"),
            "nothing is pending, so the chosen refresh interval applies"
        );
        assert!(page.contains(&service.form_token));
        // No script, no inline style attribute, no event handler: the policy
        // would block them, and the page has none to block.
        let lower = page.to_ascii_lowercase();
        for forbidden in [
            "<script",
            "javascript:",
            " style=",
            " onclick",
            " onload",
            "<iframe",
        ] {
            assert!(!lower.contains(forbidden), "found {forbidden:?}");
        }
        // The stylesheet's nonce is the one the policy allows.
        let nonce = response.style_nonce.clone().expect("a nonce");
        assert!(page.contains(&format!("<style nonce=\"{nonce}\">")));
        assert!(response
            .content_security_policy()
            .contains(&format!("style-src 'nonce-{nonce}'")));
        // Two renders never share a nonce.
        let again = service.handle(&get("/")).await;
        assert_ne!(again.style_nonce, response.style_nonce);
    }

    #[tokio::test]
    async fn the_refresh_can_be_paused_and_is_clamped() {
        let (service, _port, _) = service(None);
        let paused = text(&service.handle(&get("/?refresh=0")).await);
        assert!(!paused.contains("http-equiv=\"refresh\""));
        assert!(
            paused.contains("name=\"refresh\" value=\"0\""),
            "forms keep it"
        );
        assert!(paused.contains("<a href=\"?refresh=3\">"));
        let slow = text(&service.handle(&get("/?refresh=10")).await);
        assert!(slow.contains("<meta http-equiv=\"refresh\" content=\"10\">"));
        let huge = text(&service.handle(&get("/?refresh=99999999")).await);
        assert!(huge.contains("content=\"3600\""));
        // Nonsense falls back to the default instead of failing the page.
        let junk = text(&service.handle(&get("/?refresh=%3Cscript%3E&x=1")).await);
        assert!(junk.contains("content=\"3\""));
        assert!(!junk.contains("<script"));
    }

    /// Mutation this catches: writing any value into the page unescaped. The
    /// cluster name is the operator's own, but a refused hello's reason quotes
    /// what a peer sent over the network.
    #[tokio::test]
    async fn hostile_strings_are_escaped_in_the_page_and_in_the_json() {
        let nasty = "<script>alert(1)</script>\"'&</title>";
        let mut orch = orchestrator(nasty);
        // A hello naming another cluster is refused; its reason lands in the
        // event log with the peer's text in it.
        let _ = orch.handle_shard_hello(&ShardHello {
            cluster: "<img src=x onerror=alert(2)>".into(),
            shard: ShardId::NONE,
            endpoint: loopback(),
            region: None,
            client_endpoint: None,
            secret: String::new(),
        });
        let (handle, _port) = crate::ops::channel(orch.snapshot());
        let service = service_with(handle, None, &[]);

        let page = text(&service.handle(&get("/")).await);
        assert!(!page.contains("<script>"), "no element came from data");
        assert!(!page.contains("<img"), "no element came from data");
        assert!(
            page.contains("&lt;script&gt;alert(1)&lt;/script&gt;&quot;&#39;&amp;&lt;/title&gt;")
        );
        assert!(page.contains("&lt;img src=x onerror=alert(2)&gt;"));
        assert_eq!(page.matches("</title>").count(), 1);

        let json = text(&service.handle(&get("/api/state")).await);
        check_json(&json).expect("well-formed JSON whatever the strings hold");
        assert!(!json.contains("<script>") && !json.contains("<img"));
        assert!(json.contains("\\u003cscript\\u003e"));
    }

    #[tokio::test]
    async fn the_state_is_served_as_well_formed_json() {
        let mut orch = orchestrator("space-test");
        orch.handle_load_report(&LoadReport {
            shard: ShardId::new(1),
            players: 3,
            tick_p95_ms: f32::NAN,
            load: f32::INFINITY,
            ..Default::default()
        });
        orch.tick();
        let (handle, _port) = crate::ops::channel(orch.snapshot());
        let service = service_with(handle, None, &[]);
        let response = service.handle(&get("/api/state")).await;
        assert_eq!(response.status, 200);
        assert_eq!(response.content_type, "application/json; charset=utf-8");
        let json = text(&response);
        check_json(&json).unwrap_or_else(|why| panic!("{why}\n{json}"));
        for needle in [
            "\"cluster\":\"space-test\"",
            "\"epoch\":1",
            "\"path\":\"40:0:0:0\"",
            "\"id\":1",
            "\"connected\":true",
            "\"players\":3",
            "\"tick_p95_ms\":null",
            "\"next_shard_id\":2",
            "\"paused\":false",
            "\"epoch_in_flight\":null",
            "\"free_octants\":[0,1,2,3,4,5,6,7]",
            "\"scene\":{\"name\":\"demo-area\",\"id\":0,\"revision\":1,\"changed_tick\":null",
            "\"confirmed_by\":[1],\"not_confirmed_by\":[],\"pending\":false,\
             \"status\":\"All 1 connected shard switched to demo-area (revision 1)\"",
            "\"catalogue\":[{\"name\":\"demo-area\",\"id\":0,\"current\":true},\
             {\"name\":\"materials\",\"id\":1,\"current\":false},\
             {\"name\":\"galaxy\",\"id\":3,\"current\":false}]",
            "\"scene\":{\"name\":\"demo-area\",\"id\":0,\"revision\":1},\
             \"scene_sync\":\"in-sync\"",
        ] {
            assert!(json.contains(needle), "missing {needle} in {json}");
        }
        assert!(json.contains(&format!("\"form_token\":\"{}\"", service.form_token)));
        assert!(
            !json.contains("peer-token-never-shown"),
            "the peer token is not in the state"
        );
        let page = text(&service.handle(&get("/")).await);
        assert!(!page.contains("peer-token-never-shown"));
    }

    /// The page and the JSON show the one address players type, whom a hello
    /// there is sent to, and — per shard and per region — where players dial
    /// it beside where shards do. Without a player port the page says so.
    ///
    /// Mutation this catches: a client endpoint that is in the map and
    /// nowhere an operator can see it (a player sent to the wrong address is
    /// then debugged from packet captures), or a page that needs a script to
    /// show it.
    #[tokio::test]
    async fn the_page_and_the_json_show_the_player_port_and_each_client_endpoint() {
        let config = OrchestratorConfig {
            cluster: "space-test".into(),
            player_bind: Some(SocketAddr::from(([127, 0, 0, 1], 7_777))),
            ..OrchestratorConfig::default()
        };
        let map = ShardMap::new(
            1,
            vec![ShardRegion::new(
                config.root_node(),
                ShardId::new(1),
                config.bind,
            )],
        );
        let ids = PersistentShardIdAllocator::with_sink(2, Box::new(MemoryCounterSink::default()));
        let mut control =
            ControlPlane::new(config, ids, EpochPublisher::new(map), "peer-token".into());
        control.set_unclaimed_root_owner(ShardId::new(1));
        let mut orch = Orchestrator::new(control, Arc::new(NoLauncher));

        // Nobody has claimed the root: players are told the cluster is
        // starting.
        let (handle, _port) = crate::ops::channel(orch.snapshot());
        let service = service_with(handle, None, &[]);
        let json = text(&service.handle(&get("/api/state")).await);
        check_json(&json).unwrap_or_else(|why| panic!("{why}\n{json}"));
        assert!(
            json.contains(
                "\"player_port\":{\"bind\":\"127.0.0.1:7777\",\"answer\":\"starting\",\
                 \"sends_to_shard\":null,\"redirected\":0,\"told_starting\":0,\"refused_busy\":0}"
            ),
            "{json}"
        );
        let page = text(&service.handle(&get("/")).await);
        assert!(page.contains("player port (QUIC)"), "{page}");
        assert!(page.contains("127.0.0.1:7777"));
        assert!(page.contains("the cluster is starting"), "{page}");

        // A shard the process launcher started registers: it listens on
        // loopback and sends its players to the host they dialed.
        orch.handle_shard_hello(&ShardHello {
            cluster: "space-test".into(),
            shard: ShardId::NONE,
            endpoint: SocketAddr::from(([127, 0, 0, 1], 7_870)),
            region: None,
            client_endpoint: Some(SocketAddr::from(([0, 0, 0, 0], 7_870))),
            secret: String::new(),
        })
        .expect("the root");
        let (handle, _port) = crate::ops::channel(orch.snapshot());
        let service = service_with(handle, None, &[]);
        let json = text(&service.handle(&get("/api/state")).await);
        check_json(&json).unwrap_or_else(|why| panic!("{why}\n{json}"));
        for needle in [
            "\"answer\":\"redirect\",\"sends_to_shard\":1",
            "\"spawn_owner\":1",
            // The region, and the shard.
            "\"endpoint\":\"127.0.0.1:7870\",\"client_endpoint\":\"0.0.0.0:7870\"",
            "\"advertised\":\"127.0.0.1:7870\",\"client_endpoint\":\"0.0.0.0:7870\"",
        ] {
            assert!(json.contains(needle), "missing {needle} in {json}");
        }
        let page = text(&service.handle(&get("/")).await);
        for needle in [
            "players are sent to",
            "shard 1, which owns the spawn point",
            "<th>players dial</th>",
            "0.0.0.0:7870",
            "the host they dialed, port 7870",
            "127.0.0.1:7870, players dial 0.0.0.0:7870",
            "control endpoint (QUIC, shards only)",
        ] {
            assert!(page.contains(needle), "missing {needle:?} in {page}");
        }
        assert!(!page.contains("<script"), "no script on the page");

        // No player port: the page says so, the JSON says null.
        let (plain, _port, _) = super::tests::service(None);
        let json = text(&plain.handle(&get("/api/state")).await);
        assert!(json.contains("\"player_port\":null"), "{json}");
        assert!(
            json.contains("\"client_endpoint\":\"127.0.0.1:7001\""),
            "a shard that asked for none: players dial its address: {json}"
        );
        let page = text(&plain.handle(&get("/")).await);
        assert!(page.contains("players dial a shard directly"), "{page}");
    }

    /// A shard that has sent no report shows as "no report", never as zeros.
    #[tokio::test]
    async fn a_shard_without_a_report_has_no_report_not_a_report_of_zero() {
        let (service, _port, _) = service(None);
        let json = text(&service.handle(&get("/api/state")).await);
        assert!(json.contains("\"load_report\":null"), "{json}");
        assert!(!json.contains("\"players\""));
        let page = text(&service.handle(&get("/")).await);
        assert!(page.contains("no report yet"));
        assert!(!page.contains("0 player(s)"));
        assert!(page.contains("No shard has a fresh load report"));
    }

    #[tokio::test]
    async fn unknown_paths_and_wrong_methods_are_refused() {
        let (service, _port, _) = service(None);
        assert_eq!(service.handle(&get("/nope")).await.status, 404);
        assert_eq!(service.handle(&get("/favicon.ico")).await.status, 404);
        assert_eq!(service.handle(&get("/api/nope")).await.status, 404);
        // A state-changing address never answers GET.
        let response = service.handle(&get("/action/split?shard=1")).await;
        assert_eq!(
            (response.status, response.header("allow")),
            (405, Some("POST"))
        );
        let mut put = get("/");
        put.method = "PUT".into();
        let response = service.handle(&put).await;
        assert_eq!(
            (response.status, response.header("allow")),
            (405, Some("GET, HEAD"))
        );
        let response = service.handle(&post("/api/state", "")).await;
        assert_eq!(response.status, 405);
        // The API's errors are JSON, a browser's are a page with a way back.
        let api = service.handle(&get("/api/nope")).await;
        assert_eq!(api.content_type, "application/json; charset=utf-8");
        check_json(&text(&api)).expect("a JSON error");
        let html = service.handle(&get("/nope")).await;
        assert!(text(&html).contains("<a href=\"./\">Back to the dashboard</a>"));
    }

    // -------------------------------------------------------------- the Host

    /// Mutation this catches: answering whatever `Host` a request carries. A
    /// hostile page re-pointing its own name at this server (DNS rebinding)
    /// could then read the state and the form token as "same origin".
    #[tokio::test]
    async fn a_request_for_another_host_name_is_refused() {
        let (service, _port, handle) = service(None);
        for bad in [
            "evil.example",
            "evil.example:7980",
            "localhost.evil.example",
        ] {
            let response = service.handle(&with(get("/"), "host", bad)).await;
            assert_eq!(response.status, 403, "{bad}");
            assert!(!text(&response).contains(&service.form_token));
            let response = service.handle(&with(get("/api/state"), "host", bad)).await;
            assert_eq!(response.status, 403, "{bad}");
            assert!(!text(&response).contains(&service.form_token));
        }
        assert_eq!(service.handle(&without(get("/"), "host")).await.status, 400);
        let mut twice = get("/");
        twice.headers.push(("host".into(), "evil.example".into()));
        assert_eq!(service.handle(&twice).await.status, 400);
        for ok in [
            "127.0.0.1:7980",
            "localhost:7980",
            "[::1]:7980",
            "127.9.9.9",
            "LOCALHOST",
        ] {
            assert_eq!(
                service.handle(&with(get("/"), "host", ok)).await.status,
                200,
                "{ok}"
            );
        }
        // A listed name is accepted, and only that one — and listing a name
        // takes a token.
        let listed = service_with(handle, Some(TOKEN), &["ops.example.org:8443"]);
        let as_operator =
            |host: &str| with(with(get("/"), "host", host), "authorization", &basic(TOKEN));
        assert_eq!(
            listed
                .handle(&as_operator("ops.example.org:8443"))
                .await
                .status,
            200
        );
        assert_eq!(
            listed
                .handle(&as_operator("ops.example.org:9000"))
                .await
                .status,
            403,
            "the name on another port is another site"
        );
        assert_eq!(
            listed
                .handle(&as_operator("other.example.org"))
                .await
                .status,
            403
        );
        // With a token an IP literal is fine (no DNS, nothing to rebind).
        assert_eq!(
            listed.handle(&as_operator("10.1.2.3:7980")).await.status,
            200
        );
        assert_eq!(
            service
                .handle(&with(get("/"), "host", "ops.example.org:8443"))
                .await
                .status,
            403
        );
    }

    /// Mutation this catches: accepting any IP literal as `Host` without a
    /// token. A request for `10.1.2.3` that reaches a loopback-only server was
    /// sent to something in front of it — and gets neither the page nor the
    /// form token.
    #[tokio::test]
    async fn without_a_token_only_a_loopback_host_is_served() {
        let (service, mut port, _) = service(None);
        for not_local in [
            "10.1.2.3",
            "192.168.1.5:7980",
            "[2001:db8::1]:7980",
            "0.0.0.0:7980",
        ] {
            for target in ["/", "/api/state"] {
                let response = service.handle(&with(get(target), "host", not_local)).await;
                assert_eq!(response.status, 403, "{not_local} {target}");
                let body = text(&response);
                assert!(
                    body.contains("token"),
                    "the refusal says what to do: {body}"
                );
                assert!(!body.contains(&service.form_token), "{not_local} {target}");
                assert!(!body.contains("space-test"), "no state leaks");
            }
            let post_body = format!("csrf={}&state=pause", service.form_token);
            let action = with(
                with(post("/action/autoscaler", &post_body), "host", not_local),
                "origin",
                &format!("http://{not_local}"),
            );
            assert_eq!(service.handle(&action).await.status, 403, "{not_local}");
        }
        assert!(
            tokio::time::timeout(Duration::from_millis(50), port.next_request())
                .await
                .is_err(),
            "nothing reached the run loop"
        );
    }

    /// Mutation this catches: trusting the bind address when the request
    /// itself shows it came through a proxy. Without a token such a request is
    /// refused — reads and actions alike — with a text that names the header
    /// and says to configure a token. With a token a proxy is fine.
    #[tokio::test]
    async fn without_a_token_a_request_that_came_through_a_proxy_is_refused() {
        let (service, mut port, handle) = service(None);
        for (header, value) in [
            ("forwarded", "for=203.0.113.9;proto=https"),
            ("x-forwarded-for", "203.0.113.9"),
            ("x-forwarded-host", "ops.example.org"),
            ("x-forwarded-proto", "https"),
            ("x-real-ip", "203.0.113.9"),
            ("via", "1.1 proxy"),
        ] {
            for target in ["/", "/api/state"] {
                let response = service.handle(&with(get(target), header, value)).await;
                assert_eq!(response.status, 403, "{header} {target}");
                let body = text(&response);
                assert!(
                    body.contains(header) && body.contains("SPACE_ORCHESTRATOR_WEBUI_TOKEN"),
                    "{header}: {body}"
                );
                assert!(!body.contains(&service.form_token), "{header} {target}");
                assert!(!body.contains("space-test"), "no state leaks");
            }
            let body = format!("csrf={}&state=pause", service.form_token);
            let action = with(post("/action/autoscaler", &body), header, value);
            assert_eq!(service.handle(&action).await.status, 403, "{header}");
        }
        assert!(
            tokio::time::timeout(Duration::from_millis(50), port.next_request())
                .await
                .is_err(),
            "nothing reached the run loop"
        );
        // A header that only looks like one changes nothing.
        let lookalike = with(get("/"), "x-forwarded", "yes");
        assert_eq!(service.handle(&lookalike).await.status, 200);

        // With a token the proxy is the supported setup: the same request is
        // served to whoever presents the token.
        let behind_proxy = service_with(handle, Some(TOKEN), &["ops.example.org"]);
        let mut request = with(get("/"), "host", "ops.example.org");
        for (header, value) in [("x-forwarded-for", "203.0.113.9"), ("via", "1.1 proxy")] {
            request = with(request, header, value);
        }
        assert_eq!(
            behind_proxy.handle(&request).await.status,
            401,
            "no token yet"
        );
        let request = with(request, "authorization", &basic(TOKEN));
        assert_eq!(behind_proxy.handle(&request).await.status, 200);
    }

    /// Operator actions are carried out one at a time and a moment apart: the
    /// second of two that come too close is a `429` and never reaches the run
    /// loop. A request refused *before* it would reach the loop does not use
    /// the slot up.
    ///
    /// Mutation this catches: no limit at all (a looping script churns the
    /// event log and forces a snapshot per request), or counting the requests
    /// that were refused for free (anybody could then lock the operator out).
    #[tokio::test]
    async fn operator_actions_are_taken_one_at_a_time_a_moment_apart() {
        let spacing = Duration::from_millis(300);
        let mut service = served();
        service.config.limits.min_action_spacing = spacing;
        let token = service.form_token.clone();
        let pause = format!("csrf={token}&state=pause");
        let resume = format!("csrf={token}&state=resume");

        // Refused for free: these do not count.
        for _ in 0..5 {
            let no_token = service
                .handle(&post("/action/autoscaler", "state=pause"))
                .await;
            assert_eq!(no_token.status, 403);
            let malformed = format!("csrf={token}&state=explode");
            assert_eq!(
                service
                    .handle(&post("/action/autoscaler", &malformed))
                    .await
                    .status,
                400
            );
        }
        let first = service.handle(&post("/action/autoscaler", &pause)).await;
        assert_eq!(first.status, 303, "{}", text(&first));
        assert_eq!(first.header("x-space-result"), Some("accepted"));

        // Right behind it: refused, and nothing happens.
        for _ in 0..20 {
            let hurried = service.handle(&post("/action/autoscaler", &resume)).await;
            assert_eq!(hurried.status, 429, "{}", text(&hurried));
            assert_eq!(hurried.header("retry-after"), Some("1"));
            assert!(text(&hurried).contains("Nothing was changed"));
        }
        let json = text(&service.handle(&get("/api/state")).await);
        assert!(
            json.contains("\"paused\":true"),
            "the refused resumes did nothing"
        );
        let entries = json.matches("\"kind\":\"operator\"").count();
        assert_eq!(entries, 1, "and cost the event log nothing: {json}");

        // A moment later it goes through.
        tokio::time::sleep(spacing + Duration::from_millis(50)).await;
        let patient = service.handle(&post("/action/autoscaler", &resume)).await;
        assert_eq!(patient.status, 303, "{}", text(&patient));
        assert!(text(&service.handle(&get("/api/state")).await).contains("\"paused\":false"));
    }

    /// While the run loop is still carrying one action out, a second one is
    /// refused at once instead of queueing up behind it.
    #[tokio::test]
    async fn an_action_that_arrives_while_one_is_in_flight_is_refused() {
        let orch = orchestrator("space-test");
        let (handle, mut port) = crate::ops::channel(orch.snapshot());
        let service = Arc::new(service_with(handle, None, &[]));
        let body = format!("csrf={}&state=pause", service.form_token);

        // The loop takes the first request and sits on it.
        let first = {
            let (service, body) = (Arc::clone(&service), body.clone());
            tokio::spawn(async move { service.handle(&post("/action/autoscaler", &body)).await })
        };
        let held = tokio::time::timeout(Duration::from_secs(5), port.next_request())
            .await
            .expect("the first action reaches the loop")
            .expect("a request");
        let second = service.handle(&post("/action/autoscaler", &body)).await;
        assert_eq!(second.status, 429, "{}", text(&second));
        assert!(
            tokio::time::timeout(Duration::from_millis(50), port.next_request())
                .await
                .is_err(),
            "the second never reached the loop"
        );
        // The loop answers the first; the next one is welcome again (the
        // spacing is zero in this service).
        let _ = held.reply.send(crate::ops::OperatorReply {
            accepted: true,
            message: "done".into(),
            event: 1,
            scene_changed: None,
        });
        assert_eq!(first.await.expect("the first request").status, 303);
        let third = {
            let (service, body) = (Arc::clone(&service), body.clone());
            tokio::spawn(async move { service.handle(&post("/action/autoscaler", &body)).await })
        };
        assert!(
            tokio::time::timeout(Duration::from_secs(5), port.next_request())
                .await
                .expect("the third action reaches the loop")
                .is_some()
        );
        third.abort();
    }

    // ------------------------------------------------------------- the token

    #[tokio::test]
    async fn a_token_protected_ui_refuses_requests_without_the_token() {
        let (service, _port, _) = service(Some(TOKEN));
        for target in ["/", "/api/state", "/nope"] {
            let response = service.handle(&get(target)).await;
            assert_eq!(response.status, 401, "{target}");
            assert_eq!(
                response.header("www-authenticate"),
                Some("Basic realm=\"space-orchestrator\", charset=\"UTF-8\"")
            );
            let body = text(&response);
            assert!(
                !body.contains(&service.form_token),
                "{target} leaked the form token"
            );
            assert!(!body.contains("space-test"), "{target} leaked state");
        }
        for wrong in [
            basic("wrong"),
            basic(&format!("{TOKEN}x")),
            "Bearer x".to_owned(),
        ] {
            let request = with(get("/"), "authorization", &wrong);
            assert_eq!(service.handle(&request).await.status, 401, "{wrong}");
        }
        let request = with(get("/"), "authorization", &basic(TOKEN));
        let response = service.handle(&request).await;
        assert_eq!(response.status, 200);
        assert!(text(&response).contains("token required"));
        // Two Authorization headers are ambiguous: refused outright.
        let mut twice = with(get("/"), "authorization", &basic(TOKEN));
        twice.headers.push(("authorization".into(), basic("wrong")));
        assert_eq!(service.handle(&twice).await.status, 400);
    }

    #[tokio::test]
    async fn an_action_needs_the_token_too_and_is_not_carried_out_without_it() {
        let (service, mut port, _) = service(Some(TOKEN));
        let body = format!("shard=1&octant=7&csrf={}", service.form_token);
        let response = service.handle(&post("/action/split", &body)).await;
        assert_eq!(response.status, 401);
        assert!(
            tokio::time::timeout(Duration::from_millis(50), port.next_request())
                .await
                .is_err(),
            "nothing reached the run loop"
        );
    }

    // ------------------------------------------------------ cross-site posts

    /// How the test loop's one shard treats a scene change.
    #[derive(Clone)]
    enum ShardAnswers {
        /// It confirms 20 ms later, as a live shard does within a tick or two.
        Promptly,
        /// It confirms when the test says so (and not before).
        WhenTold(Arc<tokio::sync::Notify>),
    }

    /// Answer every request on `port` with the orchestrator's real handler,
    /// and have shard 1 confirm each scene change as `shard` says.
    fn run_loop(mut orch: Orchestrator<NoLauncher>, mut port: OpsPort, shard: ShardAnswers) {
        tokio::spawn(async move {
            while let Some(request) = port.next_request().await {
                let reply = orch.handle_operator(&request.command);
                port.publish(orch.snapshot());
                let changed = reply.scene_changed;
                let _ = request.reply.send(reply);
                if let Some(applied) = changed {
                    match &shard {
                        ShardAnswers::Promptly => {
                            tokio::time::sleep(Duration::from_millis(20)).await;
                        }
                        ShardAnswers::WhenTold(told) => told.notified().await,
                    }
                    orch.handle_scene_ack(&space_protocol::SceneAck {
                        shard: ShardId::new(1),
                        applied,
                    });
                    port.publish(orch.snapshot());
                }
            }
        });
    }

    fn served() -> Service {
        let orch = orchestrator("space-test");
        let (handle, port) = crate::ops::channel(orch.snapshot());
        run_loop(orch, port, ShardAnswers::Promptly);
        service_with(handle, None, &[])
    }

    /// Every form control in `page` that holds a value a reload would put
    /// back: `<select>`, `<textarea>`, `<datalist>`, and every `<input>` that
    /// is not hidden (text, checkbox, radio, …), as the lower-cased tags.
    fn controls_a_reload_resets(page: &str) -> Vec<String> {
        let lower = page.to_ascii_lowercase();
        let mut found = Vec::new();
        for (at, _) in lower.match_indices('<') {
            let rest = &lower[at..];
            let tag = &rest[..rest.find('>').map_or(rest.len(), |end| end + 1)];
            let holds_a_choice = ["<select", "<textarea", "<datalist", "<option"]
                .iter()
                .any(|name| tag.starts_with(name))
                || (tag.starts_with("<input") && !tag.contains("type=\"hidden\""));
            if holds_a_choice {
                found.push(tag.to_owned());
            }
        }
        found
    }

    /// The scene form of the dashboard: from its `action` to its end.
    fn scene_form(page: &str) -> &str {
        let start = page
            .find("action=\"action/scene\"")
            .expect("the page has a scene form");
        let length = page[start..].find("</form>").expect("the form ends");
        &page[start..start + length]
    }

    /// What a browser posts when the scene button with `value` is pressed:
    /// the form's hidden fields and that button's own `name=value` — nothing
    /// a person had to choose beforehand.
    fn pressing(form: &str, value: &str) -> String {
        let mut fields = Vec::new();
        for input in form.split("<input type=\"hidden\" name=\"").skip(1) {
            let (name, rest) = input.split_once("\" value=\"").expect("a value");
            let (held, _) = rest.split_once('"').expect("a closed value");
            fields.push(format!("{name}={held}"));
        }
        let button = format!("<button name=\"scene\" value=\"{value}\"");
        let at = form
            .find(&button)
            .unwrap_or_else(|| panic!("no button for {value}"));
        let tag = &form[at..at + form[at..].find('>').expect("a closed tag")];
        assert!(
            !tag.contains(" disabled"),
            "the button for {value} cannot be pressed"
        );
        fields.push(format!("scene={value}"));
        fields.join("&")
    }

    /// The page a response redirects to, as the browser would fetch it.
    async fn follow(service: &Service, response: &Response) -> String {
        let location = response.header("location").expect("a redirect");
        text(
            &service
                .handle(&get(location.trim_start_matches("..")))
                .await,
        )
    }

    #[tokio::test]
    async fn a_split_posted_by_the_page_is_carried_out_and_redirects_back() {
        let service = served();
        let body = format!(
            "csrf={}&refresh=10&epoch=1&shard=1&octant=7",
            service.form_token
        );
        let response = service.handle(&post("/action/split", &body)).await;
        assert_eq!(response.status, 303, "{}", text(&response));
        assert_eq!(response.header("x-space-result"), Some("accepted"));
        let location = response.header("location").expect("a redirect").to_owned();
        assert!(location.starts_with("../?refresh=10&notice="), "{location}");
        assert!(
            text(&response).starts_with("accepted: region 41:1:1:1 is wanted for a new shard"),
            "{}",
            text(&response)
        );

        // The page the redirect leads to shows the outcome and what is under
        // way: the octant waits for its shard, and one is being started
        // (nobody registers here, so that is where it stays — nothing is
        // proposed before a shard is there to take the octant).
        let page = text(
            &service
                .handle(&get(location.trim_start_matches("..")))
                .await,
        );
        assert!(
            page.contains("class=\"notice ok\""),
            "the outcome is on top"
        );
        assert!(page.contains("<dt>waiting for a shard</dt><dd>region 41:1:1:1</dd>"));
        assert!(page.contains("<dt>starting</dt><dd>1 shard(s)</dd>"));
        assert!(!page.contains("Epoch change in flight"));

        // A second change while one is under way is refused, and says why.
        let response = service.handle(&post("/action/split", &body)).await;
        assert_eq!(response.status, 303);
        assert_eq!(response.header("x-space-result"), Some("refused"));
        assert!(
            text(&response).contains("still waiting for its new shard"),
            "{}",
            text(&response)
        );
        let location = response.header("location").expect("a redirect").to_owned();
        let page = text(
            &service
                .handle(&get(location.trim_start_matches("..")))
                .await,
        );
        assert!(page.contains("class=\"notice bad\""));
        assert!(page.contains("one change at a time"));
    }

    /// A region carved more than one level down — one cell under the
    /// depth-40 root — is listed under the region around it, the page and the
    /// JSON say what it would be merged into, and the page's merge button
    /// carries the merge out.
    ///
    /// Mutation this catches: taking "the region around it" to be the parent
    /// *node* (the cell would be listed as a top-level region with no merge
    /// button, and its merge refused for want of a parent).
    #[tokio::test]
    async fn a_carved_cell_is_listed_under_its_region_and_merged_from_the_page() {
        let mut orch = orchestrator("space-test");
        let root = orch.config().root_node();
        let k = root.center().cell;
        let cell = space_world::NodeRef::new(64, [k[0] - 1, k[1], k[2]]);
        orch.begin_carve(cell).expect("the cell is inside the root");
        // The shard for it registers (no id, no region) and is given the cell.
        let new = orch
            .handle_shard_hello(&ShardHello {
                cluster: "space-test".to_owned(),
                shard: ShardId::NONE,
                endpoint: SocketAddr::from(([127, 0, 0, 1], 7_002)),
                region: None,
                client_endpoint: None,
                secret: String::new(),
            })
            .expect("the new shard registers");
        assert_eq!((new.shard, new.region.node), (ShardId::new(2), cell));
        // While its epoch is in flight the page says so.
        {
            let (handle, _port) = crate::ops::channel(orch.snapshot());
            let page = text(&service_with(handle, None, &[]).handle(&get("/")).await);
            assert!(page.contains("Epoch change in flight"));
            assert!(page.contains("new, epoch pending"));
            assert!(page.contains(&format!(
                "carve of region {} for new shard 2",
                crate::snapshot::node_path(cell)
            )));
        }
        for shard in [ShardId::new(1), new.shard] {
            orch.handle_epoch_ack(shard, 2);
        }
        assert_eq!(orch.map().epoch, 2, "the carve committed");
        for _ in 0..crate::config::MIN_EPOCH_INTERVAL_TICKS {
            orch.tick();
        }
        // The cell's shard says it is empty.
        orch.handle_load_report(&LoadReport {
            shard: new.shard,
            players: 0,
            tick_p95_ms: 1.0,
            load: 0.0,
            ..Default::default()
        });
        let path = crate::snapshot::node_path(cell);
        let (handle, port) = crate::ops::channel(orch.snapshot());
        run_loop(orch, port, ShardAnswers::Promptly);
        let service = service_with(handle, None, &[]);

        let json = text(&service.handle(&get("/api/state")).await);
        check_json(&json).unwrap_or_else(|why| panic!("{why}\n{json}"));
        let entry = json
            .split("{\"node\":")
            .find(|region| region.starts_with(&format!("{{\"path\":\"{path}\"")))
            .unwrap_or_else(|| panic!("the cell is not among the regions of {json}"));
        for needle in [
            "\"depth\":64",
            "\"shard\":2",
            // Nested in the root, whose node is its enclosing region although
            // it is 24 levels up; it is nobody's octant.
            "\"nesting\":1,\"parent\":\"40:0:0:0\",\"octant\":null,\"free_octants\":[]",
            "\"split_refused_because\":\"region at depth 64 (edge 1024 m) is below the split floor\"",
            "\"merge_refused_because\":null",
        ] {
            assert!(entry.contains(needle), "missing {needle} in {entry}");
        }

        let page = text(&service.handle(&get("/")).await);
        assert!(page.contains(&path), "the cell has a row");
        assert!(
            page.contains(": carved out of 40:0:0:0, 24 levels up"),
            "the row says where the cell was carved from"
        );
        // One region can be merged — the cell — and its button is live.
        assert_eq!(page.matches("action=\"action/merge\"").count(), 1);
        assert!(
            page.contains("<button title=\"Give this region back")
                && page.contains(">merge into shard 1</button>"),
            "an enabled merge button naming the root's owner"
        );

        let body = format!("csrf={}&epoch=2&shard=2", service.form_token);
        let response = service.handle(&post("/action/merge", &body)).await;
        assert_eq!(response.status, 303, "{}", text(&response));
        assert_eq!(response.header("x-space-result"), Some("accepted"));
        assert!(
            text(&response).starts_with(&format!(
                "accepted: epoch 3 proposed, region {path} goes back to shard 1"
            )),
            "{}",
            text(&response)
        );
        let page = follow(&service, &response).await;
        assert!(page.contains("Epoch change in flight"));
        assert!(page.contains("merge of shard 2"));
    }

    /// Mutation this catches: trusting the browser's credentials alone. A page
    /// on another site can make the operator's browser post here.
    #[tokio::test]
    async fn a_cross_origin_post_is_refused_and_nothing_happens() {
        let orch = orchestrator("space-test");
        let (handle, mut port) = crate::ops::channel(orch.snapshot());
        let service = service_with(handle, None, &[]);
        let body = format!("csrf={}&shard=1&octant=7", service.form_token);
        for origin in [
            "http://evil.example",
            "https://evil.example:7980",
            "http://127.0.0.1:9999",
            "http://localhost:7980",
            "null",
            "",
        ] {
            let request = with(post("/action/split", &body), "origin", origin);
            let response = service.handle(&request).await;
            assert_eq!(response.status, 403, "{origin:?}");
        }
        for site in ["cross-site", "same-site"] {
            let request = with(post("/action/split", &body), "sec-fetch-site", site);
            assert_eq!(service.handle(&request).await.status, 403, "{site}");
        }
        assert!(
            tokio::time::timeout(Duration::from_millis(50), port.next_request())
                .await
                .is_err(),
            "no refused request reached the run loop"
        );
    }

    /// Mutation this catches: accepting a post on its `Origin` alone (or on no
    /// `Origin` at all). The form token is what a client that is not a browser
    /// has to show, and what a stale page no longer has.
    #[tokio::test]
    async fn a_post_without_the_form_token_is_refused() {
        let orch = orchestrator("space-test");
        let (handle, mut port) = crate::ops::channel(orch.snapshot());
        let service = service_with(handle, None, &[]);
        for body in [
            "shard=1&octant=7".to_owned(),
            "csrf=&shard=1&octant=7".to_owned(),
            "csrf=00000000000000000000000000000000&shard=1&octant=7".to_owned(),
            format!("csrf={}x&shard=1&octant=7", service.form_token),
        ] {
            let response = service.handle(&post("/action/split", &body)).await;
            assert_eq!(response.status, 403, "{body}");
            assert!(text(&response).contains("form token"));
            // The same without an Origin (a script): still refused.
            let bare = without(post("/action/split", &body), "origin");
            assert_eq!(service.handle(&bare).await.status, 403, "{body}");
        }
        assert!(
            tokio::time::timeout(Duration::from_millis(50), port.next_request())
                .await
                .is_err(),
            "no refused request reached the run loop"
        );
    }

    #[tokio::test]
    async fn a_script_with_the_token_and_no_origin_is_served() {
        let service = served();
        let body = format!("csrf={}&state=pause", service.form_token);
        let request = without(post("/action/autoscaler", &body), "origin");
        let response = service.handle(&request).await;
        assert_eq!(response.status, 303, "{}", text(&response));
        assert_eq!(response.header("x-space-result"), Some("accepted"));
        assert!(text(&service.handle(&get("/")).await).contains("autoscaler paused"));
        let body = format!("csrf={}&state=resume", service.form_token);
        let response = service.handle(&post("/action/autoscaler", &body)).await;
        assert_eq!(response.header("x-space-result"), Some("accepted"));
        assert!(text(&service.handle(&get("/")).await).contains("autoscaler active"));
    }

    /// A scene switch posted by the page is carried out through the run loop
    /// and shows on the page it redirects to; a name that is not in the
    /// catalogue is refused before it gets there, and changes nothing.
    ///
    /// Mutation this catches: a scene action that skips the checks every
    /// other action passes (its own route arm without the Origin and form
    /// token tests), or an unknown name reaching the orchestrator as "the
    /// default".
    #[tokio::test]
    async fn a_scene_posted_by_the_page_switches_and_an_unknown_name_is_refused() {
        let service = served();
        let token = service.form_token.clone();
        let state = |json: &str| {
            let start = json.find("\"scene\":{\"name\":").expect("the scene object");
            json[start..start + 60].to_owned()
        };
        let before = state(&text(&service.handle(&get("/api/state")).await));
        assert!(
            before.contains("\"demo-area\",\"id\":0,\"revision\":1"),
            "{before}"
        );

        // Refused: an unknown name, the retired `solar` (merged into the
        // demo area), no name, the wrong case, a number.
        for fields in [
            "scene=nowhere",
            "scene=solar",
            "",
            "scene=",
            "scene=Galaxy",
            "scene=2",
            "name=galaxy",
        ] {
            let body = format!("csrf={token}&{fields}");
            let response = service.handle(&post("/action/scene", &body)).await;
            assert_eq!(response.status, 400, "{fields}");
            let page = text(&response);
            assert!(
                page.contains("demo-area, materials, galaxy"),
                "the refusal lists the catalogue: {page}"
            );
            assert!(!page.contains("nowhere"), "what was sent is not echoed");
        }
        // Refused like any other action: another origin, no form token, a GET.
        let body = format!("csrf={token}&scene=galaxy");
        let cross = with(
            post("/action/scene", &body),
            "origin",
            "http://evil.example",
        );
        assert_eq!(service.handle(&cross).await.status, 403);
        let fetch = with(post("/action/scene", &body), "sec-fetch-site", "cross-site");
        assert_eq!(service.handle(&fetch).await.status, 403);
        let no_token = service.handle(&post("/action/scene", "scene=galaxy")).await;
        assert_eq!(no_token.status, 403);
        let by_get = service.handle(&get("/action/scene?scene=galaxy")).await;
        assert_eq!((by_get.status, by_get.header("allow")), (405, Some("POST")));
        assert_eq!(
            state(&text(&service.handle(&get("/api/state")).await)),
            before,
            "no refusal changed the scene"
        );

        // Accepted: the page's own form. The answer comes once the shard has
        // confirmed, and says so — to a script in the body and a header, to a
        // person on the page the redirect leads to.
        let response = service
            .handle(&post("/action/scene", &format!("{body}&refresh=0&epoch=1")))
            .await;
        assert_eq!(response.status, 303, "{}", text(&response));
        assert_eq!(response.header("x-space-result"), Some("accepted"));
        assert_eq!(response.header("x-space-confirmed"), Some("1/1"));
        assert_eq!(
            text(&response).trim_end(),
            "accepted: scene demo-area -> galaxy (revision 2), sent to shard 1. All 1 connected \
             shard switched to galaxy (revision 2)"
        );
        let location = response.header("location").expect("a redirect").to_owned();
        assert!(
            location.starts_with("../?refresh=0&notice=") && location.ends_with("&switched=2"),
            "{location}"
        );
        // Auto-refresh is off (`refresh=0` came back with the redirect) and
        // the page is right all the same: it was rendered after the shard
        // answered, not before.
        let page = follow(&service, &response).await;
        for needle in [
            "<div class=\"notice ok\"><span>All 1 connected shard switched to galaxy (revision 2).",
            "<span class=\"pill info\">galaxy</span>",
            "<button name=\"scene\" value=\"galaxy\" disabled",
            ">galaxy (current)</button>",
            ">Switch all shards to demo-area</button>",
            "<dt>revision</dt><dd>2</dd>",
            "<span class=\"pill ok\">1 of 1 connected shards</span>",
            "galaxy, revision 2 <span class=\"pill ok\">current</span>",
            "switch every shard to scene galaxy: accepted",
        ] {
            assert!(page.contains(needle), "missing {needle:?}");
        }
        assert!(!page.contains("http-equiv=\"refresh\""), "off stays off");
        assert!(!page.contains("behind") && !page.contains("switching →"));
        let json = text(&service.handle(&get("/api/state")).await);
        check_json(&json).expect("well-formed");
        assert!(
            json.contains(
                "\"scene\":{\"name\":\"galaxy\",\"id\":3,\"revision\":2,\"changed_tick\":0"
            ),
            "{json}"
        );
        assert!(json.contains("\"pending\":false"), "{json}");
    }

    /// The dashboard reloads itself, and a reload puts every form control back
    /// to what the server rendered. So the page must not hold a choice in a
    /// control — in any of its states — and the scene card in particular
    /// posts a fixed `scene=NAME` per button: one click carries the request.
    ///
    /// Mutation this catches: a `<select>` (or any other control with a value)
    /// next to a submit button. That is what the scene card had: the
    /// auto-refresh put the choice back to the running scene within three
    /// seconds, the button then posted exactly that, and the answer was a
    /// green "accepted" with nothing changed.
    #[tokio::test]
    async fn no_control_on_the_page_holds_a_choice_a_reload_could_reset() {
        let orch = orchestrator("space-test");
        let (handle, port) = crate::ops::channel(orch.snapshot());
        let confirm = Arc::new(tokio::sync::Notify::new());
        run_loop(orch, port, ShardAnswers::WhenTold(Arc::clone(&confirm)));
        let service = service_with(handle, None, &[]);
        let nothing = Vec::<String>::new();

        let page = text(&service.handle(&get("/")).await);
        assert_eq!(controls_a_reload_resets(&page), nothing);
        assert!(
            page.contains("<input type=\"hidden\" name=\"csrf\""),
            "hidden fields are what the forms are made of"
        );

        // The scene card: one button per scene of the catalogue, in catalogue
        // order, each with its own fixed value; the running one is disabled
        // and marked, every other one says what it does.
        let form = scene_form(&page);
        assert_eq!(form.matches("<button").count(), Scene::ALL.len());
        let mut last = 0;
        for scene in Scene::ALL {
            let button = format!("<button name=\"scene\" value=\"{}\"", scene.name());
            assert_eq!(form.matches(&button).count(), 1, "{button}");
            let at = form.find(&button).expect("counted");
            assert!(at >= last, "catalogue order");
            last = at;
        }
        assert_eq!(form.matches(" disabled").count(), 1);
        assert!(form.contains("<button name=\"scene\" value=\"demo-area\" disabled"));
        assert!(form.contains(">demo-area (current)</button>"));
        for name in ["materials", "galaxy"] {
            assert!(
                form.contains(&format!(">Switch all shards to {name}</button>")),
                "the label says what the click does: {form}"
            );
        }

        // Pressing a button is the whole request: what it posts — hidden
        // fields plus its own name=value — switches to exactly that scene,
        // however long ago and however often the page was reloaded.
        let reloaded = text(&service.handle(&get("/")).await);
        assert_eq!(
            pressing(scene_form(&reloaded), "galaxy"),
            pressing(form, "galaxy")
        );
        let response = service
            .handle(&post("/action/scene", &pressing(form, "galaxy")))
            .await;
        assert!(
            text(&response).starts_with("accepted: scene demo-area -> galaxy (revision 2)"),
            "{}",
            text(&response)
        );

        // While the shard has not confirmed (the page reloads every second
        // now): still nothing to reset, and the buttons have moved on.
        let pending = follow(&service, &response).await;
        assert!(pending.contains("<meta http-equiv=\"refresh\" content=\"1\">"));
        assert_eq!(controls_a_reload_resets(&pending), nothing);
        let form = scene_form(&pending);
        assert!(form.contains("<button name=\"scene\" value=\"galaxy\" disabled"));
        assert!(form.contains(">Switch all shards to demo-area</button>"));
        assert_eq!(form.matches(" disabled").count(), 1);

        // And once it has, and on a refusal page.
        confirm.notify_one();
        let settled = service
            .ops
            .wait_until(Duration::from_secs(5), |s| s.scene_progress().settled())
            .await;
        assert!(settled.scene_progress().settled());
        let page = follow(&service, &response).await;
        assert_eq!(controls_a_reload_resets(&page), nothing);
        let refused = service
            .handle(&post("/action/scene", "scene=materials"))
            .await;
        assert_eq!(refused.status, 403);
        assert_eq!(controls_a_reload_resets(&text(&refused)), nothing);
    }

    /// Asking for the scene that is already running is not a switch, and
    /// cannot be mistaken for one: the answer says "already running …; nothing
    /// to do", the banner is not the green of a switch, and the control loop
    /// is not even asked — so it also never runs into the pause between
    /// actions. (A double click is this, and so is a click in a tab that
    /// still shows the buttons from before somebody else's switch.)
    ///
    /// Mutation this catches: answering "accepted" with a generic text and a
    /// green banner (reads as "switched" while nothing changed), or sending
    /// the request to the loop (the second half of a double click then gets a
    /// 429 saying "nothing was changed" — right after the first half changed
    /// it).
    #[tokio::test]
    async fn asking_for_the_scene_that_is_running_says_nothing_to_do_and_is_never_refused() {
        // The limits the binary runs with: the pause between actions is on.
        let orch = orchestrator("space-test");
        let (handle, port) = crate::ops::channel(orch.snapshot());
        run_loop(orch, port, ShardAnswers::Promptly);
        let mut service = service_with(handle, None, &[]);
        service.config.limits.min_action_spacing = Limits::default().min_action_spacing;
        assert!(service.config.limits.min_action_spacing >= Duration::from_millis(100));
        let token = service.form_token.clone();
        let operator_entries = |json: &str| json.matches("\"kind\":\"operator\"").count();

        // The scene that is running when the page is first opened.
        let same = format!("csrf={token}&refresh=0&scene=demo-area");
        let response = service.handle(&post("/action/scene", &same)).await;
        assert_eq!(response.status, 303, "{}", text(&response));
        assert_eq!(response.header("x-space-result"), Some("accepted"));
        assert_eq!(response.header("x-space-confirmed"), Some("1/1"));
        assert_eq!(
            text(&response).trim_end(),
            "accepted: already running demo-area (revision 1); nothing to do"
        );
        assert_eq!(
            response.header("location"),
            Some("../?refresh=0&switched=1&same=1")
        );
        let page = follow(&service, &response).await;
        assert!(
            page.contains(
                "<div class=\"notice info\"><span>Already running demo-area (revision 1); \
                 nothing to do."
            ),
            "the banner says it, and not in the colour of a switch"
        );
        assert!(!page.contains("class=\"notice ok\""));
        let json = text(&service.handle(&get("/api/state")).await);
        assert_eq!(operator_entries(&json), 0, "the loop was not asked: {json}");
        assert!(json.contains("\"name\":\"demo-area\",\"id\":0,\"revision\":1"));

        // A double click on a real switch: the first half switches, the
        // second — an instant later, well inside the pause — is told that it
        // is already so. Neither is refused.
        let materials = format!("csrf={token}&refresh=0&scene=materials");
        let first = service.handle(&post("/action/scene", &materials)).await;
        assert!(
            text(&first).starts_with("accepted: scene demo-area -> materials (revision 2)"),
            "{}",
            text(&first)
        );
        let second = service.handle(&post("/action/scene", &materials)).await;
        assert_eq!(second.status, 303, "{}", text(&second));
        assert_eq!(
            text(&second).trim_end(),
            "accepted: already running materials (revision 2); nothing to do"
        );
        // The browser shows the answer to the second click. What the person
        // did was switch, so that is the banner they land on — not "nothing
        // to do", which would read as if the button had done nothing.
        assert_eq!(
            second.header("location"),
            Some("../?refresh=0&switched=2"),
            "the second half of a double click is sent to the switch's banner"
        );
        let landed = follow(&service, &second).await;
        assert!(
            landed.contains(
                "<div class=\"notice ok\"><span>All 1 connected shard switched to materials \
                 (revision 2)."
            ),
            "{landed}"
        );
        assert!(!landed.contains("Already running"));
        let json = text(&service.handle(&get("/api/state")).await);
        assert_eq!(operator_entries(&json), 1, "one switch, one entry: {json}");

        // A *different* scene inside the pause is still refused — loudly, and
        // truthfully: that one really did not happen.
        let galaxy = format!("csrf={token}&scene=galaxy");
        let hurried = service.handle(&post("/action/scene", &galaxy)).await;
        assert_eq!(hurried.status, 429, "{}", text(&hurried));
        assert!(text(&hurried).contains("Nothing was changed"));
        let json = text(&service.handle(&get("/api/state")).await);
        assert!(
            json.contains("\"name\":\"materials\",\"id\":1,\"revision\":2"),
            "{json}"
        );
        // ...and goes through once the pause is over.
        tokio::time::sleep(service.config.limits.min_action_spacing * 2).await;
        let patient = service.handle(&post("/action/scene", &galaxy)).await;
        assert!(
            text(&patient).starts_with("accepted: scene materials -> galaxy (revision 3)"),
            "{}",
            text(&patient)
        );
    }

    /// The two halves of a double click can also arrive while the first is
    /// still with the run loop. The second must not be told "nothing was
    /// changed": it waits a moment for the first, and is then answered like
    /// any request for the scene that is running.
    #[tokio::test]
    async fn a_scene_request_that_races_its_twin_is_answered_once_the_twin_is_done() {
        let mut orch = orchestrator("space-test");
        let (handle, mut port) = crate::ops::channel(orch.snapshot());
        let service = Arc::new(service_with(handle, None, &[]));
        let body = format!("csrf={}&scene=galaxy", service.form_token);
        let click = |service: &Arc<Service>| {
            let (service, body) = (Arc::clone(service), body.clone());
            tokio::spawn(async move { service.handle(&post("/action/scene", &body)).await })
        };

        let first = click(&service);
        let held = tokio::time::timeout(Duration::from_secs(5), port.next_request())
            .await
            .expect("the first half reaches the loop")
            .expect("a request");
        let second = click(&service);
        // The loop is still busy with the first for a moment...
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert!(!second.is_finished(), "the second half waits for its twin");
        // ...then carries it out...
        let reply = orch.handle_operator(&held.command);
        port.publish(orch.snapshot());
        let applied = reply.scene_changed.expect("the scene changed");
        let _ = held.reply.send(reply);
        // ...and the shard confirms a moment later. Neither half answers
        // before that: the browser follows the second half's answer, and it
        // must not say "0 of 1" about a switch that is about to be done.
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert!(!first.is_finished(), "the switch waits for the shard");
        assert!(
            !second.is_finished(),
            "and so does the request for the scene that is now running"
        );
        orch.handle_scene_ack(&space_protocol::SceneAck {
            shard: ShardId::new(1),
            applied,
        });
        port.publish(orch.snapshot());

        let first = first.await.expect("the first half");
        let second = second.await.expect("the second half");
        assert!(
            text(&first).starts_with("accepted: scene demo-area -> galaxy (revision 2)"),
            "{}",
            text(&first)
        );
        assert_eq!(first.header("x-space-confirmed"), Some("1/1"));
        assert_eq!(second.status, 303, "{}", text(&second));
        assert_eq!(
            text(&second).trim_end(),
            "accepted: already running galaxy (revision 2); nothing to do"
        );
        assert_eq!(second.header("x-space-confirmed"), Some("1/1"));
        assert!(
            tokio::time::timeout(Duration::from_millis(50), port.next_request())
                .await
                .is_err(),
            "only one of the two reached the loop"
        );
    }

    /// A shard that does not confirm in time: the answer comes anyway, names
    /// who is being waited for, and the dashboard keeps reloading every second
    /// — whatever `?refresh=` says, "off" included — until the shard has
    /// confirmed. Then it says so and goes back to the chosen interval.
    ///
    /// Mutation this catches: redirecting as soon as the loop has accepted
    /// (the page is rendered before any shard can answer, says "behind", and
    /// with auto-refresh off says it for ever — which reads as "the switch did
    /// not work"), or waiting without a bound for a shard that is stuck.
    #[tokio::test]
    async fn a_switch_a_shard_has_not_confirmed_names_it_and_the_page_reloads_until_it_has() {
        let orch = orchestrator("space-test");
        let (handle, port) = crate::ops::channel(orch.snapshot());
        let confirm = Arc::new(tokio::sync::Notify::new());
        run_loop(orch, port, ShardAnswers::WhenTold(Arc::clone(&confirm)));
        let service = service_with(handle, None, &[]);
        let wait = service.config.limits.scene_confirm_wait;
        let body = format!("csrf={}&refresh=0&scene=galaxy", service.form_token);

        let started = Instant::now();
        let response = tokio::time::timeout(
            wait + Duration::from_secs(5),
            service.handle(&post("/action/scene", &body)),
        )
        .await
        .expect("the wait for a stuck shard is bounded");
        assert!(started.elapsed() >= wait, "it did wait for the shard");
        assert_eq!(response.status, 303, "{}", text(&response));
        assert_eq!(response.header("x-space-result"), Some("accepted"));
        assert_eq!(response.header("x-space-confirmed"), Some("0/1"));
        assert_eq!(
            text(&response).trim_end(),
            "accepted: scene demo-area -> galaxy (revision 2), sent to shard 1. Switching to \
             galaxy (revision 2): waiting for shard 1 … (0 of 1 connected shard confirmed)"
        );

        // The page: pending, said in the banner, in the Scene card and next to
        // the auto-refresh control — and reloading although refresh is off.
        let pending = follow(&service, &response).await;
        for needle in [
            "<meta http-equiv=\"refresh\" content=\"1\">",
            "<strong class=\"warn\">every 1 s while a shard is switching scene</strong>",
            "<div class=\"notice warn\"><span>Switching to galaxy (revision 2): waiting for \
             shard 1 … (0 of 1 connected shard confirmed).",
            "this page reloads every 1 s until every connected shard has confirmed",
            "<span class=\"pill warn\">switching → galaxy, 0 of 1 confirmed</span>",
            "<div class=\"sub\">waiting for shard 1</div>",
            "demo-area, revision 1 <span class=\"pill warn\">behind</span>",
            // The chosen interval is kept for afterwards, and in every form.
            "<strong>off</strong>",
            "name=\"refresh\" value=\"0\"",
        ] {
            assert!(pending.contains(needle), "missing {needle:?}");
        }
        // The same without the redirect's parameters: any view of the
        // dashboard reloads while a shard is behind.
        let plain = text(&service.handle(&get("/?refresh=0")).await);
        assert!(plain.contains("<meta http-equiv=\"refresh\" content=\"1\">"));
        let slow = text(&service.handle(&get("/?refresh=10")).await);
        assert!(slow.contains("content=\"1\"") && !slow.contains("content=\"10\""));
        let json = text(&service.handle(&get("/api/state")).await);
        check_json(&json).expect("well-formed");
        assert!(
            json.contains("\"confirmed_by\":[],\"not_confirmed_by\":[1],\"pending\":true"),
            "{json}"
        );
        // Asking for the same scene again changes nothing and is not refused;
        // it waits as long as the switch did, and names the same shard.
        let started = Instant::now();
        let again = service.handle(&post("/action/scene", &body)).await;
        assert!(started.elapsed() >= wait, "it waited for the shard as well");
        assert_eq!(again.status, 303, "{}", text(&again));
        assert_eq!(again.header("x-space-confirmed"), Some("0/1"));
        assert_eq!(
            text(&again).trim_end(),
            "accepted: already running galaxy (revision 2); nothing to do. Switching to galaxy \
             (revision 2): waiting for shard 1 … (0 of 1 connected shard confirmed)"
        );

        // The shard confirms. The very page the browser is on (same address,
        // reloaded by the meta refresh) now says so, and stops reloading.
        confirm.notify_one();
        let mut settled = String::new();
        for _ in 0..100 {
            settled = follow(&service, &response).await;
            if settled.contains("class=\"notice ok\"") {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        for needle in [
            "<div class=\"notice ok\"><span>All 1 connected shard switched to galaxy (revision 2).",
            "<span class=\"pill ok\">1 of 1 connected shards</span>",
            "galaxy, revision 2 <span class=\"pill ok\">current</span>",
        ] {
            assert!(settled.contains(needle), "missing {needle:?}");
        }
        assert!(!settled.contains("http-equiv=\"refresh\""), "back to off");
        assert!(!settled.contains("while a shard is switching scene"));
        let slow = text(&service.handle(&get("/?refresh=10")).await);
        assert!(
            slow.contains("content=\"10\""),
            "back to the chosen interval"
        );
    }

    /// With no shard connected a switch is saved and nothing more — and the
    /// answer says exactly that instead of a bare "accepted". (This is also
    /// what an operator sees after restarting only the orchestrator: the
    /// shards that were running do not come back by themselves.)
    #[tokio::test]
    async fn a_switch_with_no_shard_connected_says_that_nobody_was_told() {
        let mut orch = orchestrator("space-test");
        orch.note_connected([]);
        let (handle, port) = crate::ops::channel(orch.snapshot());
        run_loop(orch, port, ShardAnswers::Promptly);
        let service = service_with(handle, None, &[]);
        let body = format!("csrf={}&scene=galaxy", service.form_token);
        let started = Instant::now();
        let response = service.handle(&post("/action/scene", &body)).await;
        assert!(
            started.elapsed() < service.config.limits.scene_confirm_wait,
            "nobody to wait for"
        );
        assert_eq!(response.header("x-space-confirmed"), Some("0/0"));
        let answer = text(&response);
        assert!(
            answer.contains("No shard is connected: galaxy (revision 2) is saved"),
            "{answer}"
        );
        let page = follow(&service, &response).await;
        assert!(
            page.contains("<div class=\"notice warn\"><span>No shard is connected: galaxy"),
            "not the green of a switch that reached somebody"
        );
        assert!(page.contains("<span class=\"dim\">no shard is connected</span>"));
        assert!(
            page.contains("<meta http-equiv=\"refresh\" content=\"3\">"),
            "nobody is behind, so the chosen interval applies"
        );

        // Asked again: nothing to do, and still nobody who runs it — said in
        // the answer and on the page alike, at once (nobody to wait for).
        let started = Instant::now();
        let again = service.handle(&post("/action/scene", &body)).await;
        assert!(started.elapsed() < service.config.limits.scene_confirm_wait);
        assert_eq!(again.header("x-space-confirmed"), Some("0/0"));
        assert_eq!(
            text(&again).trim_end(),
            "accepted: already running galaxy (revision 2); nothing to do. No shard is \
             connected: galaxy (revision 2) is saved and is handed to each shard when it registers"
        );
        // (Asked right after the switch, so the page it leads to is the
        // switch's; a later request leads to the "already running" one.)
        let page = follow(&service, &again).await;
        assert!(
            page.contains("<div class=\"notice warn\"><span>No shard is connected: galaxy"),
            "{page}"
        );
        let page = text(&service.handle(&get("/?switched=2&same=1")).await);
        assert!(
            page.contains(
                "<div class=\"notice warn\"><span>Already running galaxy (revision 2); nothing \
                 to do. No shard is connected: galaxy"
            ),
            "{page}"
        );
    }

    /// For two seconds after a scene change, a request for that scene is the
    /// second half of a double click and is sent to the switch's banner; after
    /// that — and for a scene that has not changed since startup — it is a
    /// request for nothing, and is sent to the page that says so.
    #[test]
    fn a_request_that_repeats_a_fresh_switch_is_sent_to_the_switchs_banner() {
        let mut orch = orchestrator("space-test");
        let location = |orch: &Orchestrator<NoLauncher>| {
            scene_unchanged(&orch.snapshot(), 3)
                .header("location")
                .expect("a redirect")
                .to_owned()
        };
        assert_eq!(location(&orch), "../?refresh=3&switched=1&same=1");

        let reply = orch.handle_operator(&OperatorCommand::SetScene {
            scene: Scene::Galaxy,
        });
        orch.handle_scene_ack(&space_protocol::SceneAck {
            shard: ShardId::new(1),
            applied: reply.scene_changed.expect("the scene changed"),
        });
        assert_eq!(location(&orch), "../?refresh=3&switched=2");
        for _ in 0..SCENE_TWIN_WINDOW_TICKS {
            orch.tick();
        }
        assert_eq!(
            location(&orch),
            "../?refresh=3&switched=2",
            "still within two seconds"
        );
        orch.tick();
        assert_eq!(location(&orch), "../?refresh=3&switched=2&same=1");
    }

    /// The banner is read from the state, not from the link: a link naming a
    /// revision the cluster has not reached (a stale tab after a restart, or
    /// a made-up one) shows no banner at all.
    #[tokio::test]
    async fn a_scene_banner_cannot_be_conjured_by_a_link() {
        let (service, _port, _) = service(None);
        for target in [
            "/?switched=7",
            "/?switched=0",
            "/?switched=x&same=1",
            "/?same=1",
        ] {
            let page = text(&service.handle(&get(target)).await);
            assert!(!page.contains("class=\"notice"), "{target}");
        }
        // The revision the cluster is at: the banner says what is true now.
        let page = text(&service.handle(&get("/?switched=1")).await);
        assert!(page.contains(
            "<div class=\"notice ok\"><span>All 1 connected shard switched to demo-area \
             (revision 1)."
        ));
        let page = text(&service.handle(&get("/?switched=1&same=1")).await);
        assert!(page.contains(
            "<div class=\"notice info\"><span>Already running demo-area (revision 1); nothing \
             to do."
        ));
    }

    /// "Already running X" is about one revision. Once the scene has been
    /// changed again, a tab still showing that answer must not turn it into
    /// "already running" the *new* scene, which nobody asked that tab for.
    #[tokio::test]
    async fn an_already_running_banner_does_not_outlive_its_revision() {
        let orch = orchestrator("space-test");
        let (handle, port) = crate::ops::channel(orch.snapshot());
        run_loop(orch, port, ShardAnswers::Promptly);
        let service = service_with(handle, None, &[]);
        let same = format!("csrf={}&scene=demo-area", service.form_token);
        let asked = service.handle(&post("/action/scene", &same)).await;
        assert!(follow(&service, &asked)
            .await
            .contains("Already running demo-area (revision 1)"));

        let galaxy = format!("csrf={}&scene=galaxy", service.form_token);
        let switched = service.handle(&post("/action/scene", &galaxy)).await;
        assert_eq!(switched.header("x-space-confirmed"), Some("1/1"));
        // The old tab reloads.
        let stale = follow(&service, &asked).await;
        assert!(!stale.contains("Already running"), "{stale}");
        assert!(!stale.contains("class=\"notice"));
        // The switch's own banner stays what it was.
        assert!(follow(&service, &switched)
            .await
            .contains("All 1 connected shard switched to galaxy (revision 2)."));
    }

    /// What each shard confirmed is on the page next to it, and a shard that
    /// is behind — or names a scene the cluster never had — is marked.
    #[tokio::test]
    async fn the_page_marks_a_shard_whose_scene_is_behind() {
        use space_protocol::{SceneAck, SceneState};
        let mut orch = orchestrator("space-test");
        let ack = |orch: &mut Orchestrator<NoLauncher>, scene: Scene, revision| {
            orch.handle_scene_ack(&SceneAck {
                shard: ShardId::new(1),
                applied: SceneState::new(scene.id(), revision),
            });
        };
        let render = |orch: &Orchestrator<NoLauncher>| {
            let (handle, _port) = crate::ops::channel(orch.snapshot());
            let service = service_with(handle, None, &[]);
            async move {
                (
                    text(&service.handle(&get("/")).await),
                    text(&service.handle(&get("/api/state")).await),
                )
            }
        };

        ack(&mut orch, Scene::DemoArea, 1);
        let (page, json) = render(&orch).await;
        assert!(
            page.contains("demo-area, revision 1 <span class=\"pill ok\">current</span>"),
            "the confirmed scene is shown as current"
        );
        assert!(page.contains("<span class=\"pill ok\">1 of 1 connected shards</span>"));
        assert!(
            json.contains("\"confirmed_by\":[1],\"not_confirmed_by\":[]"),
            "{json}"
        );
        assert!(json.contains(
            "\"scene\":{\"name\":\"demo-area\",\"id\":0,\"revision\":1},\"scene_sync\":\"in-sync\""
        ));

        // The scene moves on and the shard has not answered yet.
        assert!(
            orch.handle_operator(&OperatorCommand::SetScene {
                scene: Scene::Galaxy
            })
            .accepted
        );
        let (page, json) = render(&orch).await;
        assert!(
            page.contains("demo-area, revision 1 <span class=\"pill warn\">behind</span>"),
            "a shard on the old revision is marked"
        );
        assert!(
            page.contains("<span class=\"pill warn\">switching → galaxy, 0 of 1 confirmed</span>"),
            "the card says a switch is under way, and how far it is"
        );
        assert!(page.contains("<div class=\"sub\">waiting for shard 1</div>"));
        assert!(json.contains("\"scene_sync\":\"behind\""), "{json}");
        assert!(json.contains("\"confirmed_by\":[],\"not_confirmed_by\":[1]"));

        // It answers with something far past anything the cluster ever had.
        let nonsense = crate::scene::MAX_REVISION_CATCH_UP * 4;
        ack(&mut orch, Scene::Materials, nonsense);
        let (page, json) = render(&orch).await;
        assert!(page.contains(&format!(
            "materials, revision {nonsense} <span class=\"pill bad\">not the cluster&#39;s</span>"
        )));
        assert!(json.contains("\"scene_sync\":\"differs\""), "{json}");
        check_json(&json).expect("well-formed");
    }

    #[tokio::test]
    async fn a_malformed_action_is_refused_before_it_reaches_the_run_loop() {
        let orch = orchestrator("space-test");
        let (handle, mut port) = crate::ops::channel(orch.snapshot());
        let service = service_with(handle, None, &[]);
        let token = service.form_token.clone();
        for (path, fields, status) in [
            ("/action/split", "octant=7", 400),
            ("/action/split", "shard=x&octant=7", 400),
            ("/action/split", "shard=4294967295&octant=7", 400),
            ("/action/split", "shard=1&octant=8", 400),
            ("/action/split", "shard=1&octant=-1", 400),
            ("/action/split", "shard=1&octant=7&epoch=soon", 400),
            ("/action/merge", "", 400),
            ("/action/autoscaler", "state=explode", 400),
            ("/action/split", "shard=1&octant=%zz", 400),
            ("/action/scene", "scene=nowhere", 400),
            ("/action/scene", "scene=galaxy&epoch=soon", 400),
            ("/action/scenes", "scene=galaxy", 404),
        ] {
            let body = format!("csrf={token}&{fields}");
            let response = service.handle(&post(path, &body)).await;
            assert_eq!(response.status, status, "{path} {fields}");
        }
        let body = format!("csrf={token}&shard=1&octant=7");
        let json = with(
            post("/action/split", &body),
            "content-type",
            "application/json",
        );
        assert_eq!(service.handle(&json).await.status, 415);
        let plain = with(post("/action/split", &body), "content-type", "text/plain");
        assert_eq!(service.handle(&plain).await.status, 415);
        assert!(
            tokio::time::timeout(Duration::from_millis(50), port.next_request())
                .await
                .is_err(),
            "nothing malformed reached the run loop"
        );
    }

    /// The run loop is not answering (it is stalled, or gone): the request is
    /// refused in bounded time instead of hanging, and the next one too.
    #[tokio::test]
    async fn a_stopped_run_loop_is_a_503_not_a_hang() {
        let orch = orchestrator("space-test");
        let (handle, port) = crate::ops::channel(orch.snapshot());
        let service = service_with(handle, None, &[]);
        drop(port);
        let body = format!("csrf={}&state=pause", service.form_token);
        let response = tokio::time::timeout(
            Duration::from_secs(2),
            service.handle(&post("/action/autoscaler", &body)),
        )
        .await
        .expect("answered at once");
        assert_eq!(response.status, 503);
        // Reading still works: the last snapshot is still there.
        assert_eq!(service.handle(&get("/")).await.status, 200);
    }

    // --------------------------------------------------------- over a socket

    async fn exchange(addr: SocketAddr, request: &[u8]) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut stream = TcpStream::connect(addr).await.expect("connect");
        stream.write_all(request).await.expect("write");
        let mut out = Vec::new();
        let _ = stream.read_to_end(&mut out).await;
        String::from_utf8_lossy(&out).into_owned()
    }

    async fn serving(limits: Limits) -> (SocketAddr, tokio::task::JoinHandle<()>, OpsPort) {
        let (handle, port) = crate::ops::channel(orchestrator("space-test").snapshot());
        let config = WebUiConfig::new(SocketAddr::from(([127, 0, 0, 1], 0)), None, vec![])
            .expect("loopback")
            .with_limits(limits);
        let ui = WebUi::bind(config, handle).await.expect("bind");
        let addr = ui.local_addr();
        assert!(ui.describe().contains(&addr.to_string()));
        (addr, tokio::spawn(ui.serve()), port)
    }

    #[tokio::test]
    async fn the_socket_path_serves_pages_and_refuses_garbage() {
        let (addr, task, _port) = serving(Limits::default()).await;
        let page = exchange(
            addr,
            format!("GET / HTTP/1.1\r\nHost: {addr}\r\n\r\n").as_bytes(),
        )
        .await;
        assert!(page.starts_with("HTTP/1.1 200 OK\r\n"), "{page}");
        for header in [
            "Content-Type: text/html; charset=utf-8\r\n",
            "Cache-Control: no-store\r\n",
            "X-Content-Type-Options: nosniff\r\n",
            "X-Frame-Options: DENY\r\n",
            "Connection: close\r\n",
            "Content-Security-Policy: default-src 'none'; style-src 'nonce-",
            "frame-ancestors 'none'\r\n",
        ] {
            assert!(page.contains(header), "missing {header:?}");
        }
        assert!(page.contains("space-test"));

        // HEAD: the same headers, no body.
        let head = exchange(
            addr,
            format!("HEAD / HTTP/1.1\r\nHost: {addr}\r\n\r\n").as_bytes(),
        )
        .await;
        assert!(head.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(head.ends_with("\r\n\r\n"), "no body follows a HEAD");

        let garbage = exchange(addr, b"GARBAGE\r\n\r\n").await;
        assert!(garbage.starts_with("HTTP/1.1 400 "), "{garbage}");
        assert!(garbage.contains("X-Content-Type-Options: nosniff"));
        let big = format!(
            "POST /action/split HTTP/1.1\r\nHost: {addr}\r\nContent-Length: 999999\r\n\r\n"
        );
        assert!(exchange(addr, big.as_bytes())
            .await
            .starts_with("HTTP/1.1 413 "));
        let chunked = format!(
            "POST /action/split HTTP/1.1\r\nHost: {addr}\r\nTransfer-Encoding: chunked\r\n\r\n"
        );
        assert!(exchange(addr, chunked.as_bytes())
            .await
            .starts_with("HTTP/1.1 501 "));
        let long = format!(
            "GET /{} HTTP/1.1\r\nHost: {addr}\r\n\r\n",
            "a".repeat(4_000)
        );
        assert!(exchange(addr, long.as_bytes())
            .await
            .starts_with("HTTP/1.1 414 "));
        task.abort();
    }

    /// Mutation this catches: reading a request with no deadline. A client
    /// that connects and then says nothing (or trickles) would hold a
    /// connection slot for ever.
    #[tokio::test]
    async fn a_client_that_never_finishes_its_request_is_timed_out() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let (addr, task, _port) = serving(Limits {
            read_timeout: Duration::from_millis(200),
            ..Limits::default()
        })
        .await;
        let mut slow = TcpStream::connect(addr).await.expect("connect");
        slow.write_all(b"GET / HTTP/1.1\r\nHost: x")
            .await
            .expect("write");
        let started = Instant::now();
        let mut answer = Vec::new();
        tokio::time::timeout(Duration::from_secs(5), slow.read_to_end(&mut answer))
            .await
            .expect("the server gives up on its own")
            .expect("read");
        assert!(String::from_utf8_lossy(&answer).starts_with("HTTP/1.1 408 "));
        assert!(started.elapsed() >= Duration::from_millis(150));
        task.abort();
    }

    /// Mutation this catches: no connection cap (or a cap that parks the
    /// accept loop). Idle connections up to the cap must not keep everyone
    /// else out for longer than the read deadline, and the extra connection is
    /// closed at once rather than queued.
    #[tokio::test]
    async fn connections_beyond_the_cap_are_closed_and_slots_come_back() {
        use tokio::io::AsyncReadExt;
        let (addr, task, _port) = serving(Limits {
            max_connections: 2,
            max_connections_per_ip: 2,
            read_timeout: Duration::from_millis(400),
            ..Limits::default()
        })
        .await;
        // Two connections that say nothing take both slots.
        let idle_a = TcpStream::connect(addr).await.expect("connect");
        let idle_b = TcpStream::connect(addr).await.expect("connect");
        tokio::time::sleep(Duration::from_millis(50)).await;
        // A third is closed without an answer, promptly.
        let mut extra = TcpStream::connect(addr).await.expect("connect");
        let mut answer = Vec::new();
        let closed =
            tokio::time::timeout(Duration::from_millis(300), extra.read_to_end(&mut answer))
                .await
                .expect("closed at once, not parked");
        assert!(closed.is_err() || answer.is_empty(), "no slot, no answer");
        // Once the idle ones have timed out, the server serves again.
        tokio::time::sleep(Duration::from_millis(600)).await;
        let page = exchange(
            addr,
            format!("GET / HTTP/1.1\r\nHost: {addr}\r\n\r\n").as_bytes(),
        )
        .await;
        assert!(page.starts_with("HTTP/1.1 200 OK\r\n"), "{page}");
        drop((idle_a, idle_b));
        task.abort();
    }

    /// Mutation this catches: counting every IPv6 address on its own. One
    /// machine has a whole /64 of them, so a per-address limit would limit
    /// nothing.
    #[test]
    fn an_ipv6_client_is_counted_by_its_slash_64() {
        let table: Arc<Mutex<HashMap<IpAddr, usize>>> = Arc::default();
        let address = |text: &str| text.parse::<IpAddr>().expect("an address");
        let first = AddressSlot::claim(&table, address("2001:db8:1:2::1"), 2).expect("first");
        let second = AddressSlot::claim(&table, address("2001:db8:1:2:aaaa:bbbb:cccc:dddd"), 2)
            .expect("another address of the same /64: the second slot");
        assert!(
            AddressSlot::claim(&table, address("2001:db8:1:2::ffff"), 2).is_none(),
            "the /64 is at its limit, whichever of its addresses asks"
        );
        let neighbour = AddressSlot::claim(&table, address("2001:db8:1:3::1"), 2);
        assert!(neighbour.is_some(), "the next /64 is somebody else");
        assert_eq!(table.lock().unwrap().len(), 2, "one entry per /64");
        drop((first, second, neighbour));
        assert!(table.lock().unwrap().is_empty());

        assert_eq!(
            slot_key(address("::ffff:203.0.113.7")),
            address("203.0.113.7")
        );
        assert_eq!(slot_key(address("203.0.113.7")), address("203.0.113.7"));
        assert_eq!(
            slot_key(address("2001:db8:1:2:3:4:5:6")),
            address("2001:db8:1:2::")
        );
        assert_eq!(slot_key(address("::1")), address("::"));
    }

    #[test]
    fn the_default_limits_give_an_idle_connection_less_time_than_a_request() {
        let limits = Limits::default();
        assert!(limits.head_timeout < limits.read_timeout);
        assert!(limits.head_timeout <= Duration::from_secs(3));
        assert!(limits.max_connections_per_ip * 4 <= limits.max_connections);
        assert!(limits.min_action_spacing > Duration::ZERO);
    }

    /// Over a socket: a connection that never finishes its head is answered
    /// `408` when the head's deadline passes, long before the request's would.
    #[tokio::test]
    async fn an_idle_connection_is_given_up_on_at_the_head_deadline() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let (addr, task, _port) = serving(Limits {
            head_timeout: Duration::from_millis(200),
            read_timeout: Duration::from_secs(30),
            ..Limits::default()
        })
        .await;
        let mut idle = TcpStream::connect(addr).await.expect("connect");
        idle.write_all(b"GET / HTTP/1.1\r\nHost: x")
            .await
            .expect("write");
        let started = Instant::now();
        let mut answer = Vec::new();
        tokio::time::timeout(Duration::from_secs(5), idle.read_to_end(&mut answer))
            .await
            .expect("given up on at the head deadline, not the request's")
            .expect("read");
        assert!(String::from_utf8_lossy(&answer).starts_with("HTTP/1.1 408 "));
        assert!(started.elapsed() >= Duration::from_millis(150));
        task.abort();
    }

    #[test]
    fn an_address_gives_its_slot_back_when_its_connection_ends() {
        let table: Arc<Mutex<HashMap<IpAddr, usize>>> = Arc::default();
        let client: IpAddr = "203.0.113.7".parse().expect("an address");
        let mapped: IpAddr = "::ffff:203.0.113.7".parse().expect("an address");
        let other: IpAddr = "203.0.113.8".parse().expect("an address");
        let first = AddressSlot::claim(&table, client, 2).expect("first");
        let second = AddressSlot::claim(&table, mapped, 2).expect("the same client, second slot");
        assert!(
            AddressSlot::claim(&table, client, 2).is_none(),
            "at its limit"
        );
        assert!(
            AddressSlot::claim(&table, other, 2).is_some(),
            "others are unaffected"
        );
        drop(first);
        let third = AddressSlot::claim(&table, client, 2).expect("a slot came back");
        drop((second, third));
        assert!(table.lock().unwrap().is_empty(), "nothing is remembered");
    }
}
