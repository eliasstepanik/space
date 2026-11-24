//! The web UI against a real, running cluster.
//!
//! Everything here is production code talking over real sockets: the
//! orchestrator is booted through `boot` and runs `Orchestrator::run_with` over
//! its QUIC control endpoint, the root shard is started by
//! `start_orchestrated` (what `space-server --orchestrator` runs), a split's
//! child is launched by the real `LocalLauncher`, and the web UI is bound on
//! `127.0.0.1:0` and spoken to with hand-written HTTP over a `TcpStream` —
//! there is no HTTP client in the workspace, and none is needed to say `GET /`.
//!
//! What is proven:
//!
//! * the page and the JSON show the cluster, the map and the shard that
//!   registered;
//! * a split posted to the UI goes through the run loop and the epoch protocol
//!   and **commits**: the orchestrator's map and the root shard's own map both
//!   move to epoch 2 with the carved region in it;
//! * a scene switch posted to the UI is saved, reaches the shard that is
//!   connected, a shard that registers afterwards and the child a split
//!   launches, is confirmed by each of them, and arrives at a player connected
//!   to a shard — while an epoch change is in flight, without either waiting
//!   for the other;
//! * a cross-origin post, a post without the form token, and any request to a
//!   token-protected UI without the token are refused, and change nothing;
//! * a tokenless UI refuses a request that came through a proxy, and operator
//!   actions closer together than the configured spacing get a `429` and
//!   change nothing;
//! * HTTP clients that connect and say nothing do not hold up the control
//!   plane.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use space_orchestrator::boot::boot;
use space_orchestrator::config::OrchestratorConfig;
use space_orchestrator::launcher::local::LocalLauncher;
use space_orchestrator::launcher::Launcher as _;
use space_orchestrator::ops::OpsHandle;
use space_orchestrator::orchestrator::Orchestrator;
use space_orchestrator::webui::{Limits, WebUi, WebUiConfig};
use space_protocol::{Message, Scene, SceneState, ShardId};
use space_server::orchestrated::{start_orchestrated, OrchestratedOptions, RunningShard};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::task::JoinHandle;

const TOKEN: &str = "integration-test-token-0123456789";

fn loopback0() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], 0))
}

/// The default limits without the pause between operator actions: most tests
/// here post several actions back to back (the pause has a test of its own).
fn unhurried() -> Limits {
    Limits {
        min_action_spacing: Duration::ZERO,
        ..Limits::default()
    }
}

/// A running orchestrator with its operator handle.
struct Cluster {
    name: String,
    control_addr: SocketAddr,
    ops: OpsHandle,
    launcher: Arc<LocalLauncher>,
    run_loop: JoinHandle<Result<(), std::io::Error>>,
    state_path: std::path::PathBuf,
}

impl Cluster {
    /// Boot an orchestrator the way the binary does and start its run loop.
    async fn boot(tag: &str) -> Self {
        let state_path = std::env::temp_dir().join(format!(
            "space-orch-webui-{tag}-{}-{}.id",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        let _ = std::fs::remove_file(&state_path);
        let config = OrchestratorConfig {
            bind: loopback0(),
            state_path: state_path.clone(),
            ..OrchestratorConfig::default()
        };
        let name = config.cluster.clone();
        let control = boot(config, "peer-secret-never-shown".into()).expect("boot");
        let launcher = Arc::new(LocalLauncher::new());
        let mut orchestrator = Orchestrator::new(control, Arc::clone(&launcher));
        let ops = orchestrator.attach_ops();
        let server = orchestrator.bind().expect("bind control");
        let control_addr = server.local_addr().expect("control port");
        let run_loop = tokio::spawn(async move { orchestrator.run_with(server).await });
        Self {
            name,
            control_addr,
            ops,
            launcher,
            run_loop,
            state_path,
        }
    }

    /// Register the root shard, as `space-server --orchestrator` does.
    async fn root_shard(&self) -> RunningShard {
        start_orchestrated(
            OrchestratedOptions {
                shard: ShardId::NONE,
                bind: loopback0(),
                advertise: None,
                claim_region: None,
                cluster: self.name.clone(),
                client_endpoint: None,
                secret: String::new(),
            },
            self.control_addr,
        )
        .await
        .expect("the root shard registers with the orchestrator")
    }

    /// Serve the web UI on a free loopback port.
    async fn web_ui(&self, token: Option<&str>, limits: Limits) -> (SocketAddr, JoinHandle<()>) {
        let config = WebUiConfig::new(loopback0(), token.map(str::to_owned), vec![])
            .expect("a loopback configuration")
            .with_limits(limits);
        let ui = WebUi::bind(config, self.ops.clone())
            .await
            .expect("bind the web UI");
        let addr = ui.local_addr();
        (addr, tokio::spawn(ui.serve()))
    }

    fn stop(self) {
        self.run_loop.abort();
        let _ = std::fs::remove_file(&self.state_path);
    }
}

/// One HTTP response, split into its parts.
#[derive(Debug)]
struct Reply {
    status: u16,
    head: String,
    body: String,
}

impl Reply {
    /// The value of header `name` (as written by the server).
    fn header(&self, name: &str) -> Option<&str> {
        self.head
            .lines()
            .filter_map(|line| line.split_once(": "))
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.trim())
    }
}

/// Send `request` to `addr` over a fresh connection and read the answer to
/// the end (the server closes after one response). `None` when the server
/// closed the connection without answering (it does that when it is full).
async fn try_send(addr: SocketAddr, request: &str) -> Option<Reply> {
    let mut stream = TcpStream::connect(addr).await.ok()?;
    stream.write_all(request.as_bytes()).await.ok()?;
    let mut raw = Vec::new();
    // A reset instead of a clean close still leaves whatever was received.
    let _ = tokio::time::timeout(Duration::from_secs(10), stream.read_to_end(&mut raw))
        .await
        .expect("the server answers or closes; it never leaves a client hanging");
    let text = String::from_utf8(raw).expect("a UTF-8 response");
    let (head, body) = text.split_once("\r\n\r\n")?;
    let status = head
        .split(' ')
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or_else(|| panic!("no status in {head:?}"));
    Some(Reply {
        status,
        head: head.to_owned(),
        body: body.to_owned(),
    })
}

/// [`try_send`], for when an answer is owed.
async fn send(addr: SocketAddr, request: &str) -> Reply {
    try_send(addr, request)
        .await
        .unwrap_or_else(|| panic!("no answer to {request:?}"))
}

async fn get(addr: SocketAddr, path: &str) -> Reply {
    send(
        addr,
        &format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\n\r\n"),
    )
    .await
}

async fn get_as(addr: SocketAddr, path: &str, password: &str) -> Reply {
    send(
        addr,
        &format!(
            "GET {path} HTTP/1.1\r\nHost: {addr}\r\nAuthorization: {}\r\n\r\n",
            basic(password)
        ),
    )
    .await
}

/// Post a form the way the page's own form does: same `Origin` as `Host`.
async fn post(addr: SocketAddr, path: &str, form: &str) -> Reply {
    post_with(addr, path, form, &format!("Origin: http://{addr}\r\n")).await
}

/// Post a form with the given extra header lines (each ending in CRLF).
async fn post_with(addr: SocketAddr, path: &str, form: &str, headers: &str) -> Reply {
    send(
        addr,
        &format!(
            "POST {path} HTTP/1.1\r\nHost: {addr}\r\n{headers}\
             Content-Type: application/x-www-form-urlencoded\r\nContent-Length: {}\r\n\r\n{form}",
            form.len()
        ),
    )
    .await
}

/// `Basic` credentials for user `ops` with `password`.
fn basic(password: &str) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
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

/// The form token a page carries (`name="csrf" value="…"`).
fn form_token(page: &str) -> String {
    let marker = "name=\"csrf\" value=\"";
    let start = page.find(marker).expect("the page has a form") + marker.len();
    let end = page[start..].find('"').expect("the value ends") + start;
    page[start..end].to_owned()
}

/// The unsigned number right after `marker` in `json`.
fn number_after(json: &str, marker: &str) -> u64 {
    let start = json
        .find(marker)
        .unwrap_or_else(|| panic!("{marker} not in {json}"))
        + marker.len();
    let digits: String = json[start..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits
        .parse()
        .unwrap_or_else(|_| panic!("no number after {marker} in {json}"))
}

fn map_epoch(json: &str) -> u64 {
    number_after(json, "\"map\":{\"epoch\":")
}

fn region_count(json: &str) -> usize {
    json.matches("\"free_octants\":").count()
}

/// Poll the JSON state until `done` says so; panics after `limit`.
async fn wait_for_state(
    addr: SocketAddr,
    limit: Duration,
    what: &str,
    done: impl Fn(&str) -> bool,
) -> String {
    let deadline = Instant::now() + limit;
    let request = format!("GET /api/state HTTP/1.1\r\nHost: {addr}\r\n\r\n");
    let mut last = String::from("(no answer yet)");
    loop {
        // No answer at all means the server is full right now; ask again.
        if let Some(state) = try_send(addr, &request).await {
            assert_eq!(state.status, 200, "{state:?}");
            if done(&state.body) {
                return state.body;
            }
            last = state.body;
        }
        assert!(
            Instant::now() < deadline,
            "{what} did not happen within {limit:?}; last state: {last}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn the_web_ui_shows_the_cluster_and_a_split_posted_to_it_commits() {
    let cluster = Cluster::boot("split").await;
    let root = cluster.root_shard().await;
    assert_eq!(root.shard, ShardId::new(1));
    let (ui, ui_task) = cluster.web_ui(None, unhurried()).await;

    // ---- the page shows what the orchestrator knows.
    let state = wait_for_state(
        ui,
        Duration::from_secs(5),
        "the shard to show as connected",
        |s| s.contains("\"connected\":true"),
    )
    .await;
    assert_eq!(map_epoch(&state), 1);
    assert_eq!(region_count(&state), 1);
    assert!(
        state.contains("\"load_report\":null"),
        "no report is not zero: {state}"
    );
    assert!(!state.contains("peer-secret-never-shown"));

    let page = get(ui, "/").await;
    assert_eq!(page.status, 200, "{page:?}");
    assert_eq!(
        page.header("Content-Type"),
        Some("text/html; charset=utf-8")
    );
    assert_eq!(page.header("Cache-Control"), Some("no-store"));
    assert_eq!(page.header("X-Content-Type-Options"), Some("nosniff"));
    assert_eq!(page.header("X-Frame-Options"), Some("DENY"));
    let policy = page.header("Content-Security-Policy").expect("a policy");
    assert!(policy.starts_with("default-src 'none'"), "{policy}");
    assert!(
        !policy.contains("script-src") && !policy.contains("unsafe"),
        "{policy}"
    );
    for needle in [
        cluster.name.as_str(),
        "epoch 1",
        "40:0:0:0",
        "shard 1",
        &root.endpoint.to_string(),
        "connected",
        "no report yet",
        "registered from",
    ] {
        assert!(page.body.contains(needle), "the page lacks {needle:?}");
    }
    assert!(!page.body.to_ascii_lowercase().contains("<script"));
    assert!(!page.body.contains("peer-secret-never-shown"));
    let token = form_token(&page.body);

    // ---- a cross-origin post is refused, with or without the form token.
    let split = format!("csrf={token}&refresh=3&epoch=1&shard=1&octant=7");
    for headers in [
        "Origin: http://evil.example\r\n".to_owned(),
        format!("Origin: http://localhost:{}\r\n", ui.port()),
        "Origin: null\r\n".to_owned(),
        format!("Origin: http://{ui}\r\nSec-Fetch-Site: cross-site\r\n"),
    ] {
        let refused = post_with(ui, "/action/split", &split, &headers).await;
        assert_eq!(refused.status, 403, "{headers:?}: {refused:?}");
    }
    // ---- so is a same-origin post that lacks the form token, or has a wrong one.
    for form in [
        "refresh=3&epoch=1&shard=1&octant=7".to_owned(),
        format!("csrf={}&shard=1&octant=7", "0".repeat(32)),
    ] {
        let refused = post(ui, "/action/split", &form).await;
        assert_eq!(refused.status, 403, "{form}: {refused:?}");
    }
    // ---- and a GET never changes anything.
    let refused = get(ui, &format!("/action/split?{split}")).await;
    assert_eq!(refused.status, 405);
    // None of that reached the map.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let state = get(ui, "/api/state").await.body;
    assert_eq!(
        map_epoch(&state),
        1,
        "a refused request must change nothing"
    );
    assert!(state.contains("\"epoch_in_flight\":null"), "{state}");
    assert_eq!(
        number_after(&state, "\"next_shard_id\":"),
        2,
        "and burn no id"
    );

    // ---- the page's own form: the split is accepted...
    let accepted = post(ui, "/action/split", &split).await;
    assert_eq!(accepted.status, 303, "{accepted:?}");
    assert_eq!(accepted.header("X-Space-Result"), Some("accepted"));
    assert!(
        accepted
            .body
            .starts_with("accepted: region 41:1:1:1 is wanted for a new shard"),
        "{accepted:?}"
    );
    let location = accepted.header("Location").expect("a redirect").to_owned();
    assert!(location.starts_with("../?refresh=3&notice="), "{location}");

    // ---- ...and commits through the run loop: launch, the new shard's
    // hello, the id, prepare, acks, commit. (It would abort 3 s after the
    // prepare if somebody never acked.)
    let state = wait_for_state(ui, Duration::from_secs(10), "the split to commit", |s| {
        map_epoch(s) == 2
    })
    .await;
    assert_eq!(region_count(&state), 2, "{state}");
    assert!(
        state.contains("\"path\":\"41:1:1:1\""),
        "octant 7 of the root: {state}"
    );
    assert!(state.contains("\"epoch_in_flight\":null"), "{state}");
    assert_eq!(number_after(&state, "\"next_shard_id\":"), 3);

    // The root shard itself now runs the new map: the change is real, not a
    // number on a page.
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        {
            let shard = root.state.lock().await;
            let map = shard.shard_map();
            if map.epoch == 2 && map.regions.len() == 2 {
                assert!(map.regions.iter().any(|r| r.shard == ShardId::new(2)));
                break;
            }
        }
        assert!(
            Instant::now() < deadline,
            "the root shard never saw epoch 2"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // The page the redirect leads to shows the outcome and the new map.
    let page = get(ui, location.trim_start_matches("..")).await;
    assert_eq!(page.status, 200);
    for needle in [
        "epoch 2",
        "41:1:1:1",
        "shard 2",
        "octant 7 (+x +y +z) of 40:0:0:0",
        "epoch 2 committed",
        "class=\"notice ok\"",
        "merge into shard 1",
    ] {
        assert!(page.body.contains(needle), "the page lacks {needle:?}");
    }

    // ---- a stale page (it still says epoch 1) cannot change the new map.
    let stale = post(ui, "/action/split", &split).await;
    assert_eq!(stale.header("X-Space-Result"), Some("refused"), "{stale:?}");
    assert!(stale.body.contains("the map changed"), "{stale:?}");
    // ---- the octant that was just carved cannot be carved again.
    let again = format!("csrf={token}&epoch=2&shard=1&octant=7");
    let taken = post(ui, "/action/split", &again).await;
    assert_eq!(taken.header("X-Space-Result"), Some("refused"), "{taken:?}");
    assert!(taken.body.contains("already carved out"), "{taken:?}");
    // ---- a merge is refused honestly: no load report says the shard is empty.
    let merge = post(
        ui,
        "/action/merge",
        &format!("csrf={token}&epoch=2&shard=2"),
    )
    .await;
    assert_eq!(merge.status, 303, "{merge:?}");
    assert_eq!(merge.header("X-Space-Result"), Some("refused"));
    assert!(merge.body.contains("not known to be empty"), "{merge:?}");
    let state = get(ui, "/api/state").await.body;
    assert_eq!(map_epoch(&state), 2, "the refusals changed nothing");
    assert_eq!(region_count(&state), 2);
    assert_eq!(number_after(&state, "\"next_shard_id\":"), 3);

    // ---- pause and resume the autoscaler; re-send the map.
    let paused = post(
        ui,
        "/action/autoscaler",
        &format!("csrf={token}&state=pause"),
    )
    .await;
    assert_eq!(
        paused.header("X-Space-Result"),
        Some("accepted"),
        "{paused:?}"
    );
    assert!(get(ui, "/api/state").await.body.contains("\"paused\":true"));
    assert!(get(ui, "/").await.body.contains("autoscaler paused"));
    let resumed = post(
        ui,
        "/action/autoscaler",
        &format!("csrf={token}&state=resume"),
    )
    .await;
    assert_eq!(resumed.header("X-Space-Result"), Some("accepted"));
    assert!(get(ui, "/api/state")
        .await
        .body
        .contains("\"paused\":false"));
    let pushed = post(ui, "/action/push-map", &format!("csrf={token}")).await;
    assert_eq!(
        pushed.header("X-Space-Result"),
        Some("accepted"),
        "{pushed:?}"
    );
    assert!(pushed.body.contains("epoch 2"), "{pushed:?}");

    // The fleet as the page and the JSON show it: the child was started by
    // the orchestrator, the root's shard by hand; nothing is starting.
    let state = get(ui, "/api/state").await.body;
    assert!(
        state.contains(
            "\"fleet\":{\"min_shards\":0,\"max_shards\":8,\"starting\":0,\"wanted\":null,\"hello_auth_required\":false}"
        ),
        "{state}"
    );
    assert_eq!(state.matches("\"launched\":true").count(), 1, "{state}");
    assert_eq!(state.matches("\"launched\":false").count(), 1, "{state}");

    let child = cluster
        .launcher
        .launch_of(ShardId::new(2))
        .await
        .expect("the launcher started the child");
    cluster.launcher.kill(child).await.expect("stop the child");
    root.abort();
    ui_task.abort();
    cluster.stop();
}

/// The scene `shard` runs right now.
async fn scene_of(shard: &RunningShard) -> SceneState {
    shard.state.lock().await.scene()
}

/// Wait until `shard` runs `wanted`; panics after `limit`.
async fn wait_for_scene(shard: &RunningShard, wanted: SceneState, limit: Duration) {
    let deadline = Instant::now() + limit;
    loop {
        let now = scene_of(shard).await;
        if now == wanted {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "{} never switched to {wanted}; it still runs {now}",
            shard.shard
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// The next scene a player connection is told (frames of the world in between
/// are skipped); panics if none arrives within five seconds.
async fn next_scene_told(player: &space_net::Conn) -> SceneState {
    loop {
        let message = tokio::time::timeout(Duration::from_secs(5), player.recv_stream())
            .await
            .expect("the shard tells its player the scene")
            .expect("recv")
            .expect("the connection is open");
        if let Message::SceneUpdate(state) = message {
            return state;
        }
    }
}

/// The `"scene":{…}` object of the cluster in the JSON state (its head).
fn cluster_scene(json: &str) -> String {
    let start = json
        .find("\"scene\":{\"name\":\"")
        .and_then(|_| json.rfind("\"scene\":{\"name\":\""))
        .expect("the cluster's scene is in the state");
    json[start..].chars().take(48).collect()
}

/// The whole path of a scene change, over real sockets: the operator's form
/// post, the run loop, the scene file, `SceneSet` to the shards, each shard's
/// `SceneAck`, and `SceneUpdate` to a connected player.
#[tokio::test(flavor = "multi_thread")]
async fn a_scene_switch_posted_to_the_web_ui_reaches_every_shard_and_its_players() {
    let at = |scene: Scene, revision: u64| SceneState::new(scene.id(), revision);
    let cluster = Cluster::boot("scene").await;
    let (ui, ui_task) = cluster.web_ui(None, unhurried()).await;
    let page = get(ui, "/").await;
    assert_eq!(page.status, 200, "{page:?}");
    for needle in [
        "<h2>Scene</h2>",
        "action=\"action/scene\"",
        "<button name=\"scene\" value=\"demo-area\" disabled",
        ">demo-area (current)</button>",
        "<button name=\"scene\" value=\"solar\"",
        ">Switch all shards to solar</button>",
    ] {
        assert!(page.body.contains(needle), "the page lacks {needle:?}");
    }
    // Nothing on the page holds a choice that its own reloading could put
    // back: every action is a button with its own value.
    for control in [
        "<select",
        "<option",
        "<textarea",
        "type=\"text\"",
        "type=\"radio\"",
    ] {
        assert!(
            !page.body.contains(control),
            "the page has a {control:?} control"
        );
    }
    let token = form_token(&page.body);
    let switch_to = |scene: &str| format!("csrf={token}&refresh=3&scene={scene}");

    // ---- a change before any shard exists is saved and waits for them.
    let accepted = post(ui, "/action/scene", &switch_to("solar")).await;
    assert_eq!(accepted.status, 303, "{accepted:?}");
    assert_eq!(accepted.header("X-Space-Result"), Some("accepted"));
    assert_eq!(accepted.header("X-Space-Confirmed"), Some("0/0"));
    assert_eq!(
        accepted.body.trim_end(),
        "accepted: scene demo-area -> solar (revision 2), sent to nobody. No shard is \
         connected: solar (revision 2) is saved and is handed to each shard when it registers"
    );
    let scene_file = space_orchestrator::scene::scene_path(&cluster.state_path);
    assert_eq!(
        std::fs::read_to_string(&scene_file).expect("the scene was saved"),
        "2 solar\n"
    );

    // ---- a shard that registers afterwards is handed the current scene with
    // its welcome, and confirms it.
    let root = cluster.root_shard().await;
    wait_for_scene(&root, at(Scene::Solar, 2), Duration::from_secs(5)).await;
    let state = wait_for_state(ui, Duration::from_secs(5), "shard 1 to confirm", |s| {
        s.contains("\"confirmed_by\":[1],\"not_confirmed_by\":[]")
    })
    .await;
    assert!(
        cluster_scene(&state).starts_with("\"scene\":{\"name\":\"solar\",\"id\":2,\"revision\":2,"),
        "{state}"
    );
    assert!(state.contains(
        "\"scene\":{\"name\":\"solar\",\"id\":2,\"revision\":2},\"scene_sync\":\"in-sync\""
    ));

    // ---- a player on that shard is told the scene behind its welcome.
    let endpoint = space_net::client_endpoint_for(root.endpoint).expect("client endpoint");
    let (player, _welcome) = space_net::connect(
        &endpoint,
        root.endpoint,
        space_server::net::SERVER_NAME,
        &space_net::hello("player"),
    )
    .await
    .expect("a player joins the root shard");
    assert_eq!(next_scene_told(&player).await, at(Scene::Solar, 2));

    // ---- refused requests change nothing: an unknown name, another origin,
    // no form token, a GET.
    let unknown = post(ui, "/action/scene", &switch_to("nowhere")).await;
    assert_eq!(unknown.status, 400, "{unknown:?}");
    assert!(
        unknown.body.contains("demo-area, materials, solar, galaxy"),
        "{unknown:?}"
    );
    for headers in [
        "Origin: http://evil.example\r\n".to_owned(),
        format!("Origin: http://{ui}\r\nSec-Fetch-Site: cross-site\r\n"),
    ] {
        let refused = post_with(ui, "/action/scene", &switch_to("galaxy"), &headers).await;
        assert_eq!(refused.status, 403, "{headers:?}: {refused:?}");
    }
    let no_token = post(ui, "/action/scene", "scene=galaxy").await;
    assert_eq!(no_token.status, 403, "{no_token:?}");
    let by_get = get(ui, "/action/scene?scene=galaxy").await;
    assert_eq!(by_get.status, 405, "{by_get:?}");
    tokio::time::sleep(Duration::from_millis(200)).await;
    let state = get(ui, "/api/state").await.body;
    assert!(
        cluster_scene(&state).starts_with("\"scene\":{\"name\":\"solar\",\"id\":2,\"revision\":2,"),
        "a refused request must change nothing: {state}"
    );
    assert_eq!(scene_of(&root).await, at(Scene::Solar, 2));
    assert_eq!(
        std::fs::read_to_string(&scene_file).expect("still there"),
        "2 solar\n"
    );

    // ---- the page's own form, with auto-refresh switched off: the answer
    // comes when the shard has confirmed, so the shard has switched by the
    // time the answer is here — no waiting, no polling — and the one page the
    // browser is sent to (which will never reload) already says so.
    let form = format!("csrf={token}&refresh=0&scene=galaxy");
    let accepted = post(ui, "/action/scene", &form).await;
    assert_eq!(accepted.header("X-Space-Result"), Some("accepted"));
    assert_eq!(accepted.header("X-Space-Confirmed"), Some("1/1"));
    assert_eq!(
        accepted.body.trim_end(),
        "accepted: scene solar -> galaxy (revision 3), sent to shard 1. All 1 connected shard \
         switched to galaxy (revision 3)"
    );
    assert_eq!(
        scene_of(&root).await,
        at(Scene::Galaxy, 3),
        "confirmed means the shard runs it"
    );
    let location = accepted.header("Location").expect("a redirect").to_owned();
    assert!(
        location.starts_with("../?refresh=0&notice=") && location.ends_with("&switched=3"),
        "{location}"
    );
    let page = get(ui, location.trim_start_matches("..")).await;
    for needle in [
        "<div class=\"notice ok\"><span>All 1 connected shard switched to galaxy (revision 3).",
        "<span class=\"pill info\">galaxy</span>",
        "<button name=\"scene\" value=\"galaxy\" disabled",
        ">galaxy (current)</button>",
        ">Switch all shards to solar</button>",
        "galaxy, revision 3 <span class=\"pill ok\">current</span>",
        "<span class=\"pill ok\">1 of 1 connected shards</span>",
        "shard 1 confirmed scene galaxy (revision 3)",
    ] {
        assert!(page.body.contains(needle), "the page lacks {needle:?}");
    }
    for stale in ["behind", "switching →", "http-equiv=\"refresh\""] {
        assert!(
            !page.body.contains(stale),
            "the page after the redirect must not show {stale:?}"
        );
    }
    assert_eq!(next_scene_told(&player).await, at(Scene::Galaxy, 3));

    // ---- the same scene again, at once (the second half of a double click):
    // not a switch, and the answer says so — while the browser, which shows
    // this answer and not the first, is sent to the banner of the switch the
    // double click made.
    let again = post(ui, "/action/scene", &form).await;
    assert_eq!(again.status, 303, "{again:?}");
    assert_eq!(
        again.body.trim_end(),
        "accepted: already running galaxy (revision 3); nothing to do"
    );
    assert_eq!(again.header("X-Space-Confirmed"), Some("1/1"));
    // (Within two seconds of the switch; a runner that took longer to get
    // here is sent to the "already running" page, which is checked below.)
    assert!(
        matches!(
            again.header("Location"),
            Some("../?refresh=0&switched=3" | "../?refresh=0&switched=3&same=1")
        ),
        "{again:?}"
    );
    let page = get(ui, "/?refresh=0&switched=3").await;
    assert!(
        page.body.contains(
            "<div class=\"notice ok\"><span>All 1 connected shard switched to galaxy (revision 3)."
        ),
        "{}",
        page.body
    );
    // What a later request for the running scene is sent to (a tab that has
    // not reloaded since somebody else switched).
    let page = get(ui, "/?refresh=0&switched=3&same=1").await;
    assert!(
        page.body.contains(
            "<div class=\"notice info\"><span>Already running galaxy (revision 3); nothing to do."
        ),
        "{}",
        page.body
    );
    assert_eq!(scene_of(&root).await, at(Scene::Galaxy, 3));

    // ---- a scene change while an epoch change is in flight: both are
    // accepted, and both complete.
    let split = post(
        ui,
        "/action/split",
        &format!("csrf={token}&epoch=1&shard=1&octant=7"),
    )
    .await;
    assert_eq!(
        split.header("X-Space-Result"),
        Some("accepted"),
        "{split:?}"
    );
    let accepted = post(ui, "/action/scene", &switch_to("materials")).await;
    assert_eq!(
        accepted.header("X-Space-Result"),
        Some("accepted"),
        "a scene change does not wait for the epoch: {accepted:?}"
    );
    assert!(
        accepted
            .body
            .starts_with("accepted: scene galaxy -> materials (revision 4)"),
        "{accepted:?}"
    );
    // The split commits, and its child — launched by the split, registered
    // after the scene changed — runs the same scene as the root and has
    // confirmed it.
    let state = wait_for_state(
        ui,
        Duration::from_secs(10),
        "the split to commit and both shards to confirm the scene",
        |s| map_epoch(s) == 2 && s.contains("\"confirmed_by\":[1,2],\"not_confirmed_by\":[]"),
    )
    .await;
    assert!(
        cluster_scene(&state)
            .starts_with("\"scene\":{\"name\":\"materials\",\"id\":1,\"revision\":4,"),
        "{state}"
    );
    assert_eq!(region_count(&state), 2);
    assert_eq!(
        state
            .matches("\"scene\":{\"name\":\"materials\",\"id\":1,\"revision\":4},\"scene_sync\":\"in-sync\"")
            .count(),
        2,
        "both shards report the same scene: {state}"
    );
    wait_for_scene(&root, at(Scene::Materials, 4), Duration::from_secs(5)).await;
    assert_eq!(next_scene_told(&player).await, at(Scene::Materials, 4));
    assert_eq!(
        std::fs::read_to_string(&scene_file).expect("saved"),
        "4 materials\n"
    );

    let child = cluster
        .launcher
        .launch_of(ShardId::new(2))
        .await
        .expect("the launcher started the child");
    cluster.launcher.kill(child).await.expect("stop the child");
    root.abort();
    ui_task.abort();
    let _ = std::fs::remove_file(&scene_file);
    cluster.stop();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_token_protected_web_ui_refuses_everything_without_the_token() {
    // The configuration itself: no open bind without a token.
    for open in ["0.0.0.0:0", "[::]:0"] {
        let bind: SocketAddr = open.parse().expect("an address");
        assert!(WebUiConfig::new(bind, None, vec![]).is_err(), "{open}");
        assert!(WebUiConfig::new(bind, Some("short".into()), vec![]).is_err());
        assert!(WebUiConfig::new(bind, Some(TOKEN.into()), vec![]).is_ok());
    }

    let cluster = Cluster::boot("token").await;
    let limits = Limits {
        wrong_token_delay: Duration::from_millis(20),
        ..unhurried()
    };
    let (ui, ui_task) = cluster.web_ui(Some(TOKEN), limits).await;

    // No credentials: every address answers 401 and gives nothing away.
    for path in ["/", "/api/state", "/action/split", "/nowhere"] {
        let reply = get(ui, path).await;
        assert_eq!(reply.status, 401, "{path}: {reply:?}");
        assert_eq!(
            reply.header("WWW-Authenticate"),
            Some("Basic realm=\"space-orchestrator\", charset=\"UTF-8\"")
        );
        assert!(!reply.body.contains(&cluster.name), "{path} leaked state");
        assert!(!reply.body.contains("csrf"), "{path} leaked a form");
    }
    // Wrong credentials: 401 as well.
    for wrong in ["wrong", "", &format!("{TOKEN}x"), &TOKEN[..TOKEN.len() - 1]] {
        assert_eq!(get_as(ui, "/", wrong).await.status, 401, "{wrong:?}");
    }
    // The token: the page, and the JSON.
    let page = get_as(ui, "/", TOKEN).await;
    assert_eq!(page.status, 200, "{page:?}");
    assert!(page.body.contains(&cluster.name));
    assert!(page.body.contains("token required"));
    assert_eq!(get_as(ui, "/api/state", TOKEN).await.status, 200);
    let token = form_token(&page.body);

    // An action without the credentials is refused although it carries the
    // right form token and the right Origin — and nothing happens.
    let pause = format!("csrf={token}&state=pause");
    let refused = post(ui, "/action/autoscaler", &pause).await;
    assert_eq!(refused.status, 401, "{refused:?}");
    let state = get_as(ui, "/api/state", TOKEN).await.body;
    assert!(state.contains("\"paused\":false"), "{state}");
    // With the credentials it is carried out.
    let headers = format!("Origin: http://{ui}\r\nAuthorization: {}\r\n", basic(TOKEN));
    let accepted = post_with(ui, "/action/autoscaler", &pause, &headers).await;
    assert_eq!(accepted.status, 303, "{accepted:?}");
    assert_eq!(accepted.header("X-Space-Result"), Some("accepted"));
    let state = get_as(ui, "/api/state", TOKEN).await.body;
    assert!(state.contains("\"paused\":true"), "{state}");
    // Credentials do not excuse a cross-origin post.
    let headers = format!(
        "Origin: http://evil.example\r\nAuthorization: {}\r\n",
        basic(TOKEN)
    );
    let resume = format!("csrf={token}&state=resume");
    let refused = post_with(ui, "/action/autoscaler", &resume, &headers).await;
    assert_eq!(refused.status, 403, "{refused:?}");
    let state = get_as(ui, "/api/state", TOKEN).await.body;
    assert!(state.contains("\"paused\":true"), "still paused: {state}");

    ui_task.abort();
    cluster.stop();
}

/// A shard that restarts and registers again is handed the scene that is
/// current **now** — also one chosen while it was away — not the one from
/// when the orchestrator started or from the shard's first registration.
///
/// (It comes back as a **new** shard: a restarted process is given the old
/// one's region under a new id, and because a shard was replaced the scene is
/// then re-issued one revision up, so that its bodies are built once.)
#[tokio::test(flavor = "multi_thread")]
async fn a_shard_that_restarts_is_handed_the_scene_that_is_current_now() {
    let at = |scene: Scene, revision: u64| SceneState::new(scene.id(), revision);
    let cluster = Cluster::boot("restart-scene").await;
    let limits = Limits {
        // The shard that went away may still be counted as connected for a
        // moment; do not wait long for it.
        scene_confirm_wait: Duration::from_millis(300),
        ..unhurried()
    };
    let (ui, ui_task) = cluster.web_ui(None, limits).await;
    let token = form_token(&get(ui, "/").await.body);
    let switch_to = |scene: &str| format!("csrf={token}&scene={scene}");

    let first = cluster.root_shard().await;
    let (id, endpoint, first_region) = (first.shard, first.endpoint, first.region);
    let accepted = post(ui, "/action/scene", &switch_to("solar")).await;
    assert_eq!(
        accepted.header("X-Space-Confirmed"),
        Some("1/1"),
        "{accepted:?}"
    );
    assert_eq!(scene_of(&first).await, at(Scene::Solar, 2));

    // The shard goes away, and the scene is changed while it is gone.
    first.abort();
    let accepted = post(ui, "/action/scene", &switch_to("galaxy")).await;
    assert_eq!(
        accepted.header("X-Space-Result"),
        Some("accepted"),
        "{accepted:?}"
    );
    assert!(
        accepted
            .body
            .contains("No shard is connected: galaxy (revision 3)")
            || accepted
                .body
                .contains("Switching to galaxy (revision 3): waiting for shard 1"),
        "nobody confirmed it, and the answer says so: {accepted:?}"
    );
    assert!(!accepted.body.contains("All 1 connected shard switched"));

    // It comes back (same address, as a restarted process or container does)
    // and is handed galaxy at revision 3 with its welcome.
    let second = start_orchestrated(
        OrchestratedOptions {
            shard: ShardId::NONE,
            bind: loopback0(),
            advertise: Some(endpoint),
            claim_region: None,
            cluster: cluster.name.clone(),
            client_endpoint: None,
            secret: String::new(),
        },
        cluster.control_addr,
    )
    .await
    .expect("the restarted shard is welcomed back");
    assert_ne!(
        second.shard, id,
        "a new shard: an id is used by one process"
    );
    assert_eq!(second.shard, ShardId::new(2));
    assert_eq!(second.region, first_region, "with the old one's region");
    // Galaxy — the scene chosen while it was away — and, once it has taken
    // over the region, galaxy one revision up.
    wait_for_scene(&second, at(Scene::Galaxy, 4), Duration::from_secs(5)).await;
    let state = wait_for_state(ui, Duration::from_secs(5), "the shard to confirm", |s| {
        s.contains("\"confirmed_by\":[2],\"not_confirmed_by\":[],\"pending\":false")
    })
    .await;
    assert!(
        cluster_scene(&state)
            .starts_with("\"scene\":{\"name\":\"galaxy\",\"id\":3,\"revision\":4,"),
        "{state}"
    );
    assert_eq!(map_epoch(&state), 2, "the replacement was an epoch change");
    assert_eq!(region_count(&state), 1);
    assert!(
        !state.contains("\"id\":1,\"role\""),
        "nothing is left of the old shard: {state}"
    );

    second.abort();
    ui_task.abort();
    let _ = std::fs::remove_file(space_orchestrator::scene::scene_path(&cluster.state_path));
    cluster.stop();
}

/// The convergence rule over real sockets: a shard that runs a **higher**
/// revision than the orchestrator (the orchestrator lost its scene file and
/// started counting again; the shard and its player kept theirs) does not
/// swallow the operator's next switch. The orchestrator learns the shard's
/// revision from its answer, re-issues its scene above it, and shard and
/// player follow — within the one request.
#[tokio::test(flavor = "multi_thread")]
async fn a_switch_reaches_a_shard_that_holds_a_higher_revision_than_the_orchestrator() {
    let at = |scene: Scene, revision: u64| SceneState::new(scene.id(), revision);
    let cluster = Cluster::boot("ahead").await;
    let (ui, ui_task) = cluster.web_ui(None, unhurried()).await;
    let token = form_token(&get(ui, "/").await.body);
    let root = cluster.root_shard().await;
    wait_for_scene(&root, at(Scene::DemoArea, 1), Duration::from_secs(5)).await;

    // The shard's state as it would be had it outlived that orchestrator:
    // solar at revision 9, and a player who was told so.
    assert!(root.state.lock().await.set_scene(at(Scene::Solar, 9)));
    let endpoint = space_net::client_endpoint_for(root.endpoint).expect("client endpoint");
    let (player, _welcome) = space_net::connect(
        &endpoint,
        root.endpoint,
        space_server::net::SERVER_NAME,
        &space_net::hello("player"),
    )
    .await
    .expect("a player joins");
    assert_eq!(next_scene_told(&player).await, at(Scene::Solar, 9));

    // The operator switches to galaxy. For the orchestrator that is revision
    // 2 — which the shard, at 9, will not adopt.
    let accepted = post(
        ui,
        "/action/scene",
        &format!("csrf={token}&refresh=0&scene=galaxy"),
    )
    .await;
    assert_eq!(
        accepted.header("X-Space-Result"),
        Some("accepted"),
        "{accepted:?}"
    );
    assert_eq!(
        accepted.header("X-Space-Confirmed"),
        Some("1/1"),
        "the answer waits until the shard really runs it: {accepted:?}"
    );
    assert!(
        accepted
            .body
            .contains("All 1 connected shard switched to galaxy (revision 10)"),
        "re-issued one above the shard's revision: {accepted:?}"
    );
    assert_eq!(scene_of(&root).await, at(Scene::Galaxy, 10));
    assert_eq!(next_scene_told(&player).await, at(Scene::Galaxy, 10));
    let scene_file = space_orchestrator::scene::scene_path(&cluster.state_path);
    assert_eq!(
        std::fs::read_to_string(&scene_file).expect("saved"),
        "10 galaxy\n",
        "and the raised revision is what a restart comes back with"
    );
    let page = get(
        ui,
        accepted
            .header("Location")
            .expect("a redirect")
            .trim_start_matches(".."),
    )
    .await;
    for needle in [
        "<div class=\"notice ok\"><span>All 1 connected shard switched to galaxy (revision 10).",
        "galaxy, revision 10 <span class=\"pill ok\">current</span>",
        "a shard runs scene revision 9",
        "galaxy is re-issued as revision 10",
    ] {
        assert!(page.body.contains(needle), "the page lacks {needle:?}");
    }

    // From here on a switch is an ordinary one.
    let accepted = post(
        ui,
        "/action/scene",
        &format!("csrf={token}&scene=materials"),
    )
    .await;
    assert!(
        accepted
            .body
            .contains("All 1 connected shard switched to materials (revision 11)"),
        "{accepted:?}"
    );
    assert_eq!(next_scene_told(&player).await, at(Scene::Materials, 11));

    root.abort();
    ui_task.abort();
    let _ = std::fs::remove_file(&scene_file);
    cluster.stop();
}

/// With the limits the binary runs with: a tokenless UI refuses whatever shows
/// a proxy's fingerprints or is addressed to something that is not loopback,
/// and operator actions that come faster than the spacing are refused with a
/// `429` — none of it changing anything.
#[tokio::test(flavor = "multi_thread")]
async fn a_tokenless_ui_refuses_proxied_requests_and_hurried_actions() {
    let cluster = Cluster::boot("limits").await;
    let limits = Limits::default();
    assert!(limits.min_action_spacing >= Duration::from_millis(100));
    let (ui, ui_task) = cluster.web_ui(None, limits).await;
    let page = get(ui, "/").await;
    assert_eq!(page.status, 200, "{page:?}");
    let token = form_token(&page.body);

    // ---- a proxy in front of a UI without a token.
    for header in [
        "X-Forwarded-For: 203.0.113.9",
        "Forwarded: for=203.0.113.9",
        "X-Real-IP: 203.0.113.9",
        "Via: 1.1 proxy",
    ] {
        for path in ["/", "/api/state"] {
            let reply = send(
                ui,
                &format!("GET {path} HTTP/1.1\r\nHost: {ui}\r\n{header}\r\n\r\n"),
            )
            .await;
            assert_eq!(reply.status, 403, "{header} {path}: {reply:?}");
            assert!(
                reply.body.contains("SPACE_ORCHESTRATOR_WEBUI_TOKEN"),
                "{reply:?}"
            );
            assert!(
                !reply.body.contains(&token),
                "{header} {path} leaked the form token"
            );
            assert!(
                !reply.body.contains(&cluster.name),
                "{header} {path} leaked state"
            );
        }
        let pause = format!("csrf={token}&state=pause");
        let refused = post_with(
            ui,
            "/action/autoscaler",
            &pause,
            &format!("Origin: http://{ui}\r\n{header}\r\n"),
        )
        .await;
        assert_eq!(refused.status, 403, "{header}: {refused:?}");
    }
    // ---- a Host that is not this machine.
    for host in ["10.1.2.3:7980", "ops.example.org", "[2001:db8::1]:7980"] {
        let reply = send(
            ui,
            &format!("GET /api/state HTTP/1.1\r\nHost: {host}\r\n\r\n"),
        )
        .await;
        assert_eq!(reply.status, 403, "{host}: {reply:?}");
        assert!(!reply.body.contains(&token), "{host} leaked the form token");
    }
    let state = get(ui, "/api/state").await.body;
    assert!(
        state.contains("\"paused\":false"),
        "nothing was carried out: {state}"
    );

    // ---- actions faster than the spacing.
    let pause = format!("csrf={token}&state=pause");
    let resume = format!("csrf={token}&state=resume");
    let first = post(ui, "/action/autoscaler", &pause).await;
    assert_eq!(
        first.header("X-Space-Result"),
        Some("accepted"),
        "{first:?}"
    );
    let hurried = post(ui, "/action/autoscaler", &resume).await;
    assert_eq!(hurried.status, 429, "{hurried:?}");
    assert_eq!(hurried.header("Retry-After"), Some("1"));
    assert!(hurried.body.contains("Nothing was changed"), "{hurried:?}");
    let state = get(ui, "/api/state").await.body;
    assert!(
        state.contains("\"paused\":true"),
        "the hurried resume did nothing: {state}"
    );
    tokio::time::sleep(limits.min_action_spacing + Duration::from_millis(100)).await;
    let patient = post(ui, "/action/autoscaler", &resume).await;
    assert_eq!(
        patient.header("X-Space-Result"),
        Some("accepted"),
        "{patient:?}"
    );
    assert!(get(ui, "/api/state")
        .await
        .body
        .contains("\"paused\":false"));

    ui_task.abort();
    cluster.stop();
}

/// Mutation this catches: anything in the HTTP path that the control loop
/// waits on. Clients that connect and never speak fill every connection slot;
/// meanwhile a shard must still be able to register (a `ShardHello` only the
/// run loop can answer), and the UI must serve again once the idlers time out.
#[tokio::test(flavor = "multi_thread")]
async fn idle_http_clients_do_not_hold_up_the_control_plane() {
    let cluster = Cluster::boot("idle").await;
    let limits = Limits {
        max_connections: 4,
        max_connections_per_ip: 4,
        read_timeout: Duration::from_secs(4),
        ..Limits::default()
    };
    let (ui, ui_task) = cluster.web_ui(None, limits).await;

    // Take every slot: connections that say nothing, and one that trickles.
    let mut idlers = Vec::new();
    for _ in 0..4 {
        idlers.push(TcpStream::connect(ui).await.expect("connect"));
    }
    idlers[0]
        .write_all(b"GET / HTTP/1.1\r\nHost: x")
        .await
        .expect("a partial request");
    tokio::time::sleep(Duration::from_millis(100)).await;

    // The UI is full: one more connection is closed without an answer.
    let mut extra = TcpStream::connect(ui).await.expect("connect");
    let _ = extra
        .write_all(format!("GET / HTTP/1.1\r\nHost: {ui}\r\n\r\n").as_bytes())
        .await;
    let mut answer = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(1), extra.read_to_end(&mut answer))
        .await
        .expect("closed at once, not parked");
    assert!(answer.is_empty(), "no slot, no answer: {answer:?}");

    // The control plane does not care: a shard registers while the UI is full.
    // (Well inside the idlers' four seconds: it is not waiting for them.)
    let root = tokio::time::timeout(Duration::from_secs(3), cluster.root_shard())
        .await
        .expect("the run loop answers a hello while HTTP clients idle");
    assert_eq!(root.shard, ShardId::new(1));

    // Once the idlers have been timed out the UI serves again, and shows it.
    let state = wait_for_state(ui, Duration::from_secs(10), "the UI to serve again", |s| {
        s.contains("\"connected\":true")
    })
    .await;
    assert_eq!(map_epoch(&state), 1);
    drop(idlers);

    root.abort();
    ui_task.abort();
    cluster.stop();
}
