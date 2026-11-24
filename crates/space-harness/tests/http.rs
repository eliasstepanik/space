//! The HTTP API over a real loopback socket, against the fake client: every
//! route, the limits, the token, the `Host` / `Origin` refusals and the
//! one-script-at-a-time rule. Requests are written byte by byte with a blocking
//! socket, so what is tested is what a `curl` would get.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use space_harness::fake::{picture, FakeClient, FakeOptions};
use space_harness::http::{serve, Api, HttpConfig, ServeError, MAX_CONNECTIONS, MAX_HEAD};
use space_harness::image::decode_stored_png;
use space_harness::{Key, Screen};

const LONG: Duration = Duration::from_secs(20);

fn scratch(name: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU32, Ordering};
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "http-{name}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A fake client with the API served for it on a loopback port.
struct Harness {
    fake: FakeClient,
    addr: SocketAddr,
    out_dir: PathBuf,
    runtime: Option<tokio::runtime::Runtime>,
}

impl Drop for Harness {
    fn drop(&mut self) {
        // Do not wait for a script a failing test left running.
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

fn start(configure: impl FnOnce(&mut HttpConfig)) -> Harness {
    let fake = FakeClient::start(FakeOptions::default());
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let listener = runtime
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .unwrap();
    let addr = listener.local_addr().unwrap();
    let out_dir = scratch("out");
    let mut config = HttpConfig::default();
    config.script.out_dir = out_dir.clone();
    configure(&mut config);
    let control = fake.control();
    runtime.spawn(async move {
        let refused = serve(listener, control, config).await;
        panic!("the server stopped: {refused:?}");
    });
    Harness {
        fake,
        addr,
        out_dir,
        runtime: Some(runtime),
    }
}

struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Reply {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    fn text(&self) -> String {
        String::from_utf8(self.body.clone()).expect("a text body")
    }

    fn json(&self) -> Value {
        assert_eq!(self.header("content-type"), Some("application/json"));
        serde_json::from_slice(&self.body).unwrap_or_else(|e| panic!("{e}: {}", self.text()))
    }

    /// The `error` of a JSON error body.
    fn error(&self) -> String {
        self.json()["error"].as_str().expect("an error").to_owned()
    }
}

/// Send `raw` on a fresh connection and parse whatever comes back until the
/// server closes.
fn send(addr: SocketAddr, raw: &[u8]) -> Reply {
    let mut stream = TcpStream::connect(addr).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(60)))
        .unwrap();
    stream.write_all(raw).unwrap();
    let mut bytes = Vec::new();
    stream.read_to_end(&mut bytes).unwrap();
    parse_reply(&bytes)
}

fn parse_reply(bytes: &[u8]) -> Reply {
    let split = bytes
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .unwrap_or_else(|| panic!("no response head in {:?}", String::from_utf8_lossy(bytes)));
    let head = std::str::from_utf8(&bytes[..split]).unwrap();
    let body = bytes[split + 4..].to_vec();
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap();
    let status = status_line
        .strip_prefix("HTTP/1.1 ")
        .and_then(|rest| rest.split(' ').next())
        .and_then(|code| code.parse().ok())
        .unwrap_or_else(|| panic!("bad status line {status_line:?}"));
    let headers: Vec<(String, String)> = lines
        .map(|line| {
            let (name, value) = line.split_once(": ").expect("a header line");
            (name.to_owned(), value.to_owned())
        })
        .collect();
    let reply = Reply {
        status,
        headers,
        body,
    };
    // Every response is framed the same way.
    assert_eq!(reply.header("connection"), Some("close"));
    assert_eq!(
        reply.header("content-length"),
        Some(reply.body.len().to_string().as_str())
    );
    assert_eq!(reply.header("access-control-allow-origin"), None);
    reply
}

/// One request with a `Host` naming the server by address (unless `headers`
/// brings its own) and a `Content-Length` for the body.
fn request(
    addr: SocketAddr,
    method: &str,
    target: &str,
    headers: &[(&str, &str)],
    body: &[u8],
) -> Reply {
    let mut raw = format!("{method} {target} HTTP/1.1\r\n");
    if !headers.iter().any(|(n, _)| n.eq_ignore_ascii_case("host")) {
        raw.push_str(&format!("Host: {addr}\r\n"));
    }
    for (name, value) in headers {
        raw.push_str(&format!("{name}: {value}\r\n"));
    }
    raw.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
    let mut raw = raw.into_bytes();
    raw.extend_from_slice(body);
    send(addr, &raw)
}

fn get(addr: SocketAddr, target: &str) -> Reply {
    request(addr, "GET", target, &[], b"")
}

fn post(addr: SocketAddr, target: &str, body: &str) -> Reply {
    request(addr, "POST", target, &[], body.as_bytes())
}

fn state(addr: SocketAddr) -> Value {
    let reply = get(addr, "/state");
    assert_eq!(reply.status, 200);
    reply.json()
}

fn wait_for_state(addr: SocketAddr, what: &str, condition: impl Fn(&Value) -> bool) -> Value {
    let deadline = Instant::now() + LONG;
    loop {
        let now = state(addr);
        if condition(&now) {
            return now;
        }
        assert!(Instant::now() < deadline, "never {what}: {now}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn the_root_lists_the_routes() {
    let h = start(|_| {});
    let reply = get(h.addr, "/");
    assert_eq!(reply.status, 200);
    assert_eq!(
        reply.header("content-type"),
        Some("text/plain; charset=utf-8")
    );
    assert_eq!(reply.header("cache-control"), Some("no-store"));
    assert_eq!(reply.header("x-content-type-options"), Some("nosniff"));
    let text = reply.text();
    for route in [
        "GET  /state",
        "GET  /frame.png",
        "GET  /frame.jpg",
        "POST /input",
        "POST /connect",
        "POST /disconnect",
        "POST /quit",
        "POST /script",
        "POST /script/stop",
        "client.wait_until",
        "Authorization: Bearer",
    ] {
        assert!(text.contains(route), "the help does not mention {route:?}");
    }
}

#[test]
fn state_is_the_clients_snapshot_as_json() {
    let h = start(|_| {});
    let now = state(h.addr);
    assert_eq!(now["screen"], "connect");
    assert_eq!((&now["width"], &now["height"]), (&json!(64), &json!(48)));
    assert_eq!(now["error"], Value::Null);
    assert_eq!(now["position"], Value::Null);
    for field in [
        "address",
        "shard",
        "session",
        "tick",
        "velocity",
        "speed",
        "yaw_deg",
        "pitch_deg",
        "handoffs",
        "scene",
        "scene_position",
        "scene_revision",
        "retry_in_s",
        "frame",
        "fps",
        "gravity",
        "landed",
        "bodies",
        "bodies_in_view",
        "correction_m",
        "correction_max_m",
        "shard_address",
    ] {
        assert!(now.get(field).is_some(), "no {field} in {now}");
    }
    let later = wait_for_state(h.addr, "advanced", |s| {
        s["frame"].as_u64() > now["frame"].as_u64()
    });
    assert_eq!(later["screen"], "connect");
}

/// `GET /frame.png` is the picture, losslessly: it decodes back to exactly the
/// pixels the client rendered for the frame the headers name.
#[test]
fn frames_come_as_png_with_the_exact_pixels() {
    let h = start(|_| {});
    let reply = get(h.addr, "/frame.png");
    assert_eq!(reply.status, 200);
    assert_eq!(reply.header("content-type"), Some("image/png"));
    assert_eq!(reply.header("x-frame-width"), Some("64"));
    assert_eq!(reply.header("x-frame-height"), Some("48"));
    let seq: u64 = reply.header("x-frame-seq").unwrap().parse().unwrap();
    let (width, height, rgba) = decode_stored_png(&reply.body).unwrap();
    assert_eq!((width, height), (64, 48));
    assert!(
        rgba == picture(64, 48, Screen::Connect, seq),
        "not frame {seq}"
    );

    // ?wait=1 is the next frame; ?after=N the first one past N.
    let before = state(h.addr)["frame"].as_u64().unwrap();
    let next = get(h.addr, "/frame.png?wait=1");
    let next_seq: u64 = next.header("x-frame-seq").unwrap().parse().unwrap();
    assert!(next_seq > before, "{next_seq} is not after {before}");
    let after = get(h.addr, &format!("/frame.png?after={}", next_seq + 3));
    let after_seq: u64 = after.header("x-frame-seq").unwrap().parse().unwrap();
    assert!(after_seq > next_seq + 3);
    let (_, _, pixels) = decode_stored_png(&after.body).unwrap();
    assert!(pixels == picture(64, 48, Screen::Connect, after_seq));
    assert_eq!(get(h.addr, "/frame.png?wait=0").status, 200);
    assert_eq!(get(h.addr, "/frame.png?wait").status, 200);

    for bad in [
        "/frame.png?after=soon",
        "/frame.png?after=-1",
        "/frame.png?wait=maybe",
    ] {
        let reply = get(h.addr, bad);
        assert_eq!(reply.status, 400, "{bad}");
        assert!(!reply.error().is_empty());
    }
}

#[test]
fn frames_come_as_jpeg_too() {
    let h = start(|_| {});
    let reply = get(h.addr, "/frame.jpg");
    assert_eq!(reply.status, 200);
    assert_eq!(reply.header("content-type"), Some("image/jpeg"));
    assert!(reply.header("x-frame-seq").is_some());
    let jpeg = &reply.body;
    // A JFIF file: SOI, APP0 "JFIF", ..., EOI; and a baseline frame header
    // (SOF0) carrying the size.
    assert_eq!(jpeg[..4], [0xff, 0xd8, 0xff, 0xe0]);
    assert_eq!(&jpeg[6..11], b"JFIF\0");
    assert_eq!(jpeg[jpeg.len() - 2..], [0xff, 0xd9]);
    let sof = jpeg
        .windows(2)
        .position(|w| w == [0xff, 0xc0])
        .expect("a baseline frame header");
    let (height, width) = (
        u16::from_be_bytes([jpeg[sof + 5], jpeg[sof + 6]]),
        u16::from_be_bytes([jpeg[sof + 7], jpeg[sof + 8]]),
    );
    assert_eq!((width, height), (64, 48));

    let small = get(h.addr, "/frame.jpg?quality=5");
    let large = get(h.addr, "/frame.jpg?quality=100");
    assert_eq!((small.status, large.status), (200, 200));
    assert!(small.body.len() < large.body.len());
    for bad in ["0", "101", "high", "-1"] {
        assert_eq!(
            get(h.addr, &format!("/frame.jpg?quality={bad}")).status,
            400,
            "{bad}"
        );
    }
}

/// A client that renders nothing: the frame request gives up (504) after the
/// configured wait instead of holding the connection, and other requests are
/// served meanwhile.
#[test]
fn a_frame_that_never_comes_is_a_bounded_504() {
    let h = start(|config| config.frame_timeout = Duration::from_millis(1500));
    wait_for_state(h.addr, "rendering", |s| s["frame"].as_u64() > Some(2));
    h.fake.set_paused(true);
    std::thread::sleep(Duration::from_millis(30));

    let addr = h.addr;
    let started = Instant::now();
    let waiting = std::thread::spawn(move || get(addr, "/frame.png?wait=1"));
    // While that one waits, the server still answers.
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(get(h.addr, "/state").status, 200);
    assert!(!waiting.is_finished(), "the frame request did not wait");

    let reply = waiting.join().unwrap();
    assert_eq!(reply.status, 504);
    assert!(reply.error().contains("timed out"), "{}", reply.error());
    let waited = started.elapsed();
    assert!(
        waited >= Duration::from_millis(1500) && waited < Duration::from_secs(15),
        "{waited:?}"
    );
}

#[test]
fn input_types_presses_holds_and_turns() {
    let h = start(|_| {});
    // Type the address and press Enter, as two requests.
    let reply = post(h.addr, "/input", r#"{"text":"127.0.0.1:7777"}"#);
    assert_eq!(
        (reply.status, reply.json()),
        (200, json!({"ok": true, "events": 1}))
    );
    let typed = wait_for_state(h.addr, "typed", |s| s["address"] == "127.0.0.1:7777");
    assert_eq!(typed["screen"], "connect");
    assert_eq!(post(h.addr, "/input", r#"{"press":"enter"}"#).status, 200);
    let online = wait_for_state(h.addr, "online", |s| s["screen"] == "online");
    assert_eq!(online["session"], 1);
    assert_eq!(online["position"], json!([0.0, 0.0, 0.0]));

    // A batch: hold W for a while, turn, move the pointer.
    let started = Instant::now();
    let reply = post(
        h.addr,
        "/input",
        r#"[{"press":"w","ms":120},{"turn":[30,-10]},{"pointer":[0.5,0.25],"buttons":1},
            {"key":"space","down":true},{"key":"space","down":false}]"#,
    );
    assert_eq!(
        (reply.status, reply.json()),
        (200, json!({"ok": true, "events": 5}))
    );
    assert!(
        started.elapsed() >= Duration::from_millis(120),
        "the press was not held"
    );
    let moved = wait_for_state(h.addr, "turned", |s| {
        (s["yaw_deg"].as_f64().unwrap() - 30.0).abs() < 1e-3
    });
    assert!((moved["pitch_deg"].as_f64().unwrap() + 10.0).abs() < 1e-3);
    assert!(moved["speed"].as_f64().unwrap() > 1.0, "{moved}");

    let record = h.fake.record();
    assert_eq!(record.typed(), "127.0.0.1:7777");
    assert_eq!(
        record.keys(),
        [
            (Key::Enter, true),
            (Key::Enter, false),
            (Key::Forward, true),
            (Key::Forward, false),
            (Key::Up, true),
            (Key::Up, false),
        ]
    );
}

/// A bad request is refused whole: the valid events before the bad one are not
/// applied either.
#[test]
fn bad_input_is_a_400_and_nothing_of_it_is_applied() {
    let h = start(|_| {});
    for body in [
        "",
        "{nope",
        "{}",
        r#"{"key":"w"}"#,
        r#"{"press":"hyper"}"#,
        r#"[{"text":"typed"},{"key":"w","down":true},{"turn":[1]}]"#,
        r#"{"text":"esc\u001b"}"#,
        r#"{"turn":[1e999,0]}"#,
    ] {
        let reply = post(h.addr, "/input", body);
        assert_eq!(reply.status, 400, "{body}");
        assert!(!reply.error().is_empty());
    }
    let reply = post(h.addr, "/input", r#"[{"text":"a"},{"press":"nope"}]"#);
    assert!(
        reply.error().starts_with("event 2: unknown key \"nope\""),
        "{}",
        reply.error()
    );

    wait_for_state(h.addr, "a few frames", |s| s["frame"].as_u64() > Some(5));
    assert!(h.fake.record().inputs.is_empty(), "{:?}", h.fake.record());
    assert_eq!(state(h.addr)["address"], "");
}

#[test]
fn connect_disconnect_and_quit() {
    let h = start(|_| {});
    // JSON, then (after a disconnect) the address as plain text, as curl -d sends it.
    let reply = post(h.addr, "/connect", r#"{"address":"127.0.0.1:7777"}"#);
    assert_eq!((reply.status, reply.json()), (200, json!({"ok": true})));
    wait_for_state(h.addr, "online", |s| s["screen"] == "online");
    assert_eq!(post(h.addr, "/disconnect", "").status, 200);
    wait_for_state(h.addr, "back at the prompt", |s| s["screen"] == "connect");
    assert_eq!(post(h.addr, "/connect", "127.0.0.1:7777\n").status, 200);
    let online = wait_for_state(h.addr, "online again", |s| s["screen"] == "online");
    assert_eq!(online["session"], 2);

    // A server that is not there shows up in the state, not in the response.
    assert_eq!(
        post(h.addr, "/connect", r#"{"address":"10.9.9.9:1"}"#).status,
        200
    );
    let failed = wait_for_state(h.addr, "on the error screen", |s| s["screen"] == "error");
    assert_eq!(
        failed["error"],
        "cannot reach 10.9.9.9:1: connection refused"
    );

    for body in [
        "",
        "   ",
        "{}",
        r#"{"address":5}"#,
        "{broken",
        "two words",
        r#"{"address":""}"#,
    ] {
        let reply = post(h.addr, "/connect", body);
        assert_eq!(reply.status, 400, "{body:?}");
        assert!(!reply.error().is_empty());
    }
    assert_eq!(
        h.fake.record().connects,
        ["127.0.0.1:7777", "127.0.0.1:7777", "10.9.9.9:1"]
    );

    // A POST without any body or length, as `curl -X POST` sends it.
    assert!(!h.fake.has_quit());
    let raw = format!("POST /quit HTTP/1.1\r\nHost: {}\r\n\r\n", h.addr);
    assert_eq!(send(h.addr, raw.as_bytes()).status, 200);
    let deadline = Instant::now() + LONG;
    while !h.fake.has_quit() {
        assert!(Instant::now() < deadline, "the client never quit");
        std::thread::sleep(Duration::from_millis(1));
    }
    // The client is gone: requests that need it say so.
    assert_eq!(post(h.addr, "/input", r#"{"text":"x"}"#).status, 503);
    assert_eq!(post(h.addr, "/connect", "127.0.0.1:7777").status, 503);
    assert_eq!(get(h.addr, "/frame.png?wait=1").status, 503);
    assert_eq!(
        get(h.addr, "/state").status,
        200,
        "the last state stays readable"
    );
}

#[test]
fn a_script_runs_and_its_report_comes_back() {
    let h = start(|_| {});
    let reply = post(
        h.addr,
        "/script",
        r#"
client.connect("127.0.0.1:7777", 10)
client.hold("w", 0.1)
local s = client.state()
assert(s.speed > 1, "ship did not move")
print("speed ok", s.screen)
client.screenshot("thrust.png")
client.screenshot("small/thrust.jpg")
"#,
    );
    assert_eq!(reply.status, 200);
    let report = reply.json();
    assert_eq!(report["ok"], true, "{report}");
    assert_eq!(report["error"], Value::Null);
    assert_eq!(report["output"], json!(["speed ok\tonline"]));
    let png = h.out_dir.join("thrust.png");
    let jpg = h.out_dir.join("small").join("thrust.jpg");
    assert_eq!(
        report["screenshots"],
        json!([png.display().to_string(), jpg.display().to_string()])
    );
    assert_eq!(report["last_call_line"], 8);
    assert!(report["elapsed_ms"].as_u64().unwrap() >= 100);
    let (width, height, _) = decode_stored_png(&std::fs::read(&png).unwrap()).unwrap();
    assert_eq!((width, height), (64, 48));
    assert_eq!(std::fs::read(&jpg).unwrap()[..3], [0xff, 0xd8, 0xff]);
    assert_eq!(state(h.addr)["screen"], "online");
}

/// A failing script is a successful request whose report says so, with the
/// Lua message and line.
#[test]
fn a_failing_script_is_reported_with_message_and_line() {
    let h = start(|_| {});
    let reply = post(
        h.addr,
        "/script",
        "print('before')\n\nassert(1 > 2, 'ship did not move')\nprint('after')",
    );
    assert_eq!(reply.status, 200);
    assert_eq!(
        reply.json(),
        json!({
            "ok": false,
            "error": {"kind": "runtime", "message": "script:3: ship did not move", "line": 3},
            "output": ["before"],
            "screenshots": [],
            "last_call_line": 1,
            "elapsed_ms": reply.json()["elapsed_ms"],
        })
    );

    // ?name= is the chunk name; a compile error is a report too.
    let reply = post(h.addr, "/script?name=tests%2Fjoin.lua", "\nerror('boom')");
    assert_eq!(reply.json()["error"]["message"], "tests/join.lua:2: boom");
    let reply = post(h.addr, "/script", "local = 5");
    let error = &reply.json()["error"];
    assert_eq!(
        (reply.status, &error["kind"], &error["line"]),
        (200, &json!("compile"), &json!(1))
    );

    // The sandbox holds over HTTP as well.
    let reply = post(h.addr, "/script", "return os.getenv('PATH')");
    assert_eq!(reply.json()["ok"], false);
    let escape = post(h.addr, "/script", "client.screenshot('../escape.png')");
    assert!(
        escape.json()["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not a usable screenshot name"),
        "{}",
        escape.text()
    );
    assert!(!h.out_dir.parent().unwrap().join("escape.png").exists());

    // Not Lua at all: bytes that are not UTF-8.
    let reply = request(h.addr, "POST", "/script", &[], &[0xff, 0xfe, 0x00]);
    assert_eq!(reply.status, 400);
}

#[test]
fn script_timeouts_come_from_the_request_and_are_capped_by_the_server() {
    let h = start(|config| config.max_script_timeout = Duration::from_millis(600));
    let started = Instant::now();
    let reply = post(h.addr, "/script?timeout=0.2", "while true do end");
    let report = reply.json();
    assert_eq!(report["error"]["kind"], "timeout", "{report}");
    assert_eq!(
        report["error"]["message"],
        "the script did not finish within 0.2 s"
    );
    let took = started.elapsed();
    assert!(
        took >= Duration::from_millis(200) && took < Duration::from_secs(10),
        "{took:?}"
    );

    // Asking for an hour gets the server's maximum.
    let started = Instant::now();
    let reply = post(h.addr, "/script?timeout=3600", "client.wait(3600)");
    assert_eq!(
        reply.json()["error"],
        json!({"kind": "timeout", "message": "the script did not finish within 0.6 s", "line": null})
    );
    let took = started.elapsed();
    assert!(
        took >= Duration::from_millis(600) && took < Duration::from_secs(10),
        "{took:?}"
    );

    for bad in ["0", "-1", "soon", "nan", "inf", ""] {
        let reply = post(h.addr, &format!("/script?timeout={bad}"), "print(1)");
        assert_eq!(reply.status, 400, "timeout={bad:?}");
    }
}

/// One script at a time: a second one is refused with 409 while the first
/// runs, the running one can be stopped, and the slot is free again after.
#[test]
fn a_second_script_is_a_409_and_the_running_one_can_be_stopped() {
    let h = start(|_| {});
    assert_eq!(
        post(h.addr, "/script/stop", "").json(),
        json!({"ok": true, "stopped": false})
    );

    let addr = h.addr;
    let first = std::thread::spawn(move || {
        post(
            addr,
            "/script?timeout=120",
            "client.type('x')\nclient.wait(3600)",
        )
    });
    // The first script is running once its typing shows.
    wait_for_state(h.addr, "typed by the first script", |s| s["address"] == "x");

    for _ in 0..3 {
        let second = post(h.addr, "/script", "client.type('y')");
        assert_eq!(second.status, 409);
        assert!(
            second.error().contains("already running"),
            "{}",
            second.error()
        );
    }
    // Everything else is still served while a script runs.
    assert_eq!(get(h.addr, "/frame.png").status, 200);
    assert_eq!(post(h.addr, "/input", r#"{"text":"z"}"#).status, 200);
    wait_for_state(h.addr, "typed alongside", |s| s["address"] == "xz");

    let asked = Instant::now();
    assert_eq!(
        post(h.addr, "/script/stop", "").json(),
        json!({"ok": true, "stopped": true})
    );
    let report = first.join().unwrap().json();
    assert!(asked.elapsed() < Duration::from_secs(5), "slow to stop");
    assert_eq!(report["ok"], false);
    assert_eq!(report["error"]["kind"], "stopped");
    assert_eq!(report["last_call_line"], 2);

    // The slot is free: the next script runs (and the refused ones never did).
    let next = post(h.addr, "/script", "return client.state().address");
    assert_eq!(next.json()["ok"], true, "{}", next.text());
    assert_eq!(state(h.addr)["address"], "xz");
    assert_eq!(
        post(h.addr, "/script/stop", "").json(),
        json!({"ok": true, "stopped": false})
    );
}

#[test]
fn unknown_routes_and_wrong_methods() {
    let h = start(|_| {});
    for target in [
        "/nope",
        "/state/",
        "/frame.gif",
        "/script/",
        "/STATE",
        "//state",
    ] {
        let reply = get(h.addr, target);
        assert_eq!(reply.status, 404, "{target}");
        assert!(reply.error().contains("GET /"), "{}", reply.error());
    }
    for (method, target, allow) in [
        ("POST", "/state", "GET"),
        ("POST", "/frame.png", "GET"),
        ("DELETE", "/", "GET"),
        ("GET", "/input", "POST"),
        ("GET", "/quit", "POST"),
        ("GET", "/script", "POST"),
        ("PUT", "/script", "POST"),
        ("GET", "/script/stop", "POST"),
        ("OPTIONS", "/input", "POST"),
        ("HEAD", "/state", "GET"),
    ] {
        let reply = request(h.addr, method, target, &[], b"");
        assert_eq!(reply.status, 405, "{method} {target}");
        assert_eq!(reply.header("allow"), Some(allow), "{method} {target}");
    }
    assert!(!h.fake.has_quit(), "GET /quit did not quit");
    assert_eq!(get(h.addr, "/state?ignored=1").status, 200);
}

#[test]
fn oversized_and_malformed_requests_get_their_statuses() {
    let h = start(|config| {
        config.max_body = 4096;
        config.read_timeout = Duration::from_millis(300);
    });
    let host = format!("Host: {}\r\n", h.addr);

    // A body over the limit is refused on its declared length: only the head
    // is sent, so a 413 means the server did not wait for the body.
    let raw = format!("POST /script HTTP/1.1\r\n{host}Content-Length: 4097\r\n\r\n");
    let reply = send(h.addr, raw.as_bytes());
    assert_eq!(reply.status, 413);
    assert!(
        reply.error().contains("4096 byte limit"),
        "{}",
        reply.error()
    );
    let exactly = request(
        h.addr,
        "POST",
        "/script",
        &[],
        format!("--{}", "x".repeat(4094)).as_bytes(),
    );
    assert_eq!(
        exactly.status, 200,
        "a body of exactly the limit is accepted"
    );

    let huge = format!(
        "GET /state HTTP/1.1\r\n{host}X-Padding: {}\r\n\r\n",
        "a".repeat(MAX_HEAD)
    );
    assert_eq!(send(h.addr, huge.as_bytes()).status, 431);

    let chunked =
        format!("POST /script HTTP/1.1\r\n{host}Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n");
    assert_eq!(send(h.addr, chunked.as_bytes()).status, 501);

    for (raw, status) in [
        ("GARBAGE\r\n\r\n".to_owned(), 400),
        (format!("GET /state HTTP/2.0\r\n{host}\r\n"), 505),
        (format!("get /state HTTP/1.1\r\n{host}\r\n"), 400),
        (format!("GET state HTTP/1.1\r\n{host}\r\n"), 400),
        (format!("GET /state?x=%zz HTTP/1.1\r\n{host}\r\n"), 400),
        (
            format!("POST /input HTTP/1.1\r\n{host}Content-Length: nine\r\n\r\n"),
            400,
        ),
        (
            format!(
                "POST /input HTTP/1.1\r\n{host}Content-Length: 1\r\nContent-Length: 2\r\n\r\nxx"
            ),
            400,
        ),
        ("GET /state HTTP/1.1\r\n\r\n".to_owned(), 400), // no Host
        ("GET /state HTTP/1.0\r\n\r\n".to_owned(), 400), // no Host
    ] {
        let reply = send(h.addr, raw.as_bytes());
        assert_eq!(reply.status, status, "{raw:?}");
        assert!(!reply.error().is_empty());
    }

    // A request that never finishes is answered 408 when the read times out;
    // the connection is not held open for it.
    let started = Instant::now();
    let mut slow = TcpStream::connect(h.addr).unwrap();
    slow.write_all(b"POST /script HTTP/1.1\r\nContent-Length: 10\r\n\r\nhalf")
        .unwrap();
    let mut bytes = Vec::new();
    slow.read_to_end(&mut bytes).unwrap();
    assert_eq!(parse_reply(&bytes).status, 408);
    let waited = started.elapsed();
    assert!(
        waited >= Duration::from_millis(300) && waited < Duration::from_secs(10),
        "{waited:?}"
    );

    // A peer that connects and says nothing is simply dropped.
    let mut silent = TcpStream::connect(h.addr).unwrap();
    silent.shutdown(std::net::Shutdown::Write).unwrap();
    let mut bytes = Vec::new();
    silent.read_to_end(&mut bytes).unwrap();
    assert!(bytes.is_empty());

    // curl asks before sending a large body.
    let mut asking = TcpStream::connect(h.addr).unwrap();
    let head =
        format!("POST /script HTTP/1.1\r\n{host}Content-Length: 8\r\nExpect: 100-continue\r\n\r\n");
    asking.write_all(head.as_bytes()).unwrap();
    let mut interim = [0u8; 25];
    asking.read_exact(&mut interim).unwrap();
    assert_eq!(&interim, b"HTTP/1.1 100 Continue\r\n\r\n");
    asking.write_all(b"print(7)").unwrap();
    let mut bytes = Vec::new();
    asking.read_to_end(&mut bytes).unwrap();
    assert_eq!(parse_reply(&bytes).json()["output"], json!(["7"]));
}

/// At most `MAX_CONNECTIONS` connections are served at once; one more is told
/// so and dropped, and capacity returns when connections end.
#[test]
fn connections_are_capped() {
    let h = start(|_| {});
    // Connections that have sent nothing yet each hold a slot.
    let idle: Vec<TcpStream> = (0..MAX_CONNECTIONS)
        .map(|_| TcpStream::connect(h.addr).unwrap())
        .collect();
    // The server has accepted them all once the next one is refused.
    let deadline = Instant::now() + LONG;
    loop {
        let mut extra = TcpStream::connect(h.addr).unwrap();
        extra
            .set_read_timeout(Some(Duration::from_millis(300)))
            .unwrap();
        let mut bytes = Vec::new();
        let _ = extra.read_to_end(&mut bytes);
        if !bytes.is_empty() {
            let reply = parse_reply(&bytes);
            assert_eq!(reply.status, 503);
            assert_eq!(reply.error(), "too many connections");
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the extra connection was never refused"
        );
    }
    drop(idle);
    let deadline = Instant::now() + LONG;
    loop {
        let mut stream = TcpStream::connect(h.addr).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(30)))
            .unwrap();
        let raw = format!("GET /state HTTP/1.1\r\nHost: {}\r\n\r\n", h.addr);
        let mut bytes = Vec::new();
        if stream.write_all(raw.as_bytes()).is_ok()
            && stream.read_to_end(&mut bytes).is_ok()
            && parse_reply(&bytes).status == 200
        {
            break;
        }
        assert!(Instant::now() < deadline, "capacity never came back");
        std::thread::sleep(Duration::from_millis(10));
    }
}

const TOKEN: &str = "correct-horse-battery-staple";

/// With a token configured every request needs it — on loopback too, and for
/// the help page as well.
#[test]
fn a_configured_token_is_required_on_every_request() {
    let h = start(|config| config.token = Some(TOKEN.to_owned()));
    let with = |authorization: &str, method: &str, target: &str| {
        request(
            h.addr,
            method,
            target,
            &[("Authorization", authorization)],
            b"",
        )
    };
    for (method, target) in [
        ("GET", "/"),
        ("GET", "/state"),
        ("GET", "/frame.png"),
        ("POST", "/input"),
        ("POST", "/quit"),
        ("POST", "/script"),
        ("POST", "/script/stop"),
        ("GET", "/no-such-route"),
    ] {
        let bare = request(h.addr, method, target, &[], b"");
        assert_eq!(bare.status, 401, "{method} {target} without a token");
        assert_eq!(bare.header("www-authenticate"), Some("Bearer"));
        assert_eq!(bare.error(), "missing or wrong bearer token");
        for wrong in [
            "Bearer wrong",
            "Bearer correct-horse-battery-stapl",
            "Bearer correct-horse-battery-staple2",
            "Bearer CORRECT-HORSE-BATTERY-STAPLE",
            "Bearer ",
            "Basic correct-horse-battery-staple",
            "correct-horse-battery-staple",
            "",
        ] {
            assert_eq!(
                with(wrong, method, target).status,
                401,
                "{method} {target} with {wrong:?}"
            );
        }
    }
    assert!(!h.fake.has_quit(), "an unauthenticated /quit did nothing");
    assert!(h.fake.record().inputs.is_empty());

    let right = format!("Bearer {TOKEN}");
    assert_eq!(with(&right, "GET", "/state").status, 200);
    assert_eq!(with(&right, "GET", "/").status, 200);
    assert_eq!(
        with(&format!("bearer {TOKEN}"), "GET", "/state").status,
        200
    );
    assert_eq!(with(&right, "GET", "/no-such-route").status, 404);
    // The other refusals are not waived by a valid token.
    let cross = request(
        h.addr,
        "POST",
        "/quit",
        &[("Authorization", &right), ("Origin", "http://evil.example")],
        b"",
    );
    assert_eq!(cross.status, 403);
    assert!(!h.fake.has_quit());
    assert_eq!(with(&right, "POST", "/quit").status, 200);
}

/// The rule that keeps the API off the network unless a token guards it,
/// checked on the address a listener would have.
#[test]
fn a_non_loopback_address_is_refused_without_an_adequate_token() {
    let fake = FakeClient::start(FakeOptions::default());
    let api = |addr: &str, token: Option<&str>| {
        let config = HttpConfig {
            token: token.map(str::to_owned),
            ..HttpConfig::default()
        };
        Api::new(fake.control(), config, addr.parse().unwrap()).map(|_| ())
    };
    for loopback in ["127.0.0.1:8098", "127.9.9.9:1", "[::1]:8098"] {
        assert!(api(loopback, None).is_ok(), "{loopback}");
        assert!(api(loopback, Some("short")).is_ok(), "{loopback}");
        assert!(matches!(
            api(loopback, Some("")),
            Err(ServeError::WeakToken(_))
        ));
    }
    for exposed in [
        "0.0.0.0:8098",
        "192.168.1.20:8098",
        "10.0.0.5:1",
        "[::]:8098",
        "[2001:db8::1]:80",
        "[::ffff:127.0.0.1]:80",
    ] {
        let refused = api(exposed, None).unwrap_err();
        assert!(
            matches!(refused, ServeError::TokenRequired(_)),
            "{exposed}: {refused}"
        );
        assert!(refused.to_string().contains("without a token"), "{refused}");
        for weak in [
            "",
            "short",
            "fifteen-chars-xx",
            "has a space in it, long enough",
        ] {
            let weak_len = weak.len();
            let result = api(exposed, Some(weak));
            if weak_len >= 16 && !weak.contains(' ') {
                assert!(result.is_ok(), "{exposed} {weak:?}");
            } else {
                assert!(
                    matches!(result, Err(ServeError::WeakToken(_))),
                    "{exposed} {weak:?}"
                );
            }
        }
        assert!(api(exposed, Some(TOKEN)).is_ok(), "{exposed}");
    }
}

/// The same rule on a real listener bound to every interface: `serve` returns
/// the refusal instead of accepting.
#[test]
#[cfg_attr(
    windows,
    ignore = "listening on a public address makes the Windows firewall ask the user"
)]
fn serve_refuses_a_listener_on_every_interface_without_a_token() {
    let fake = FakeClient::start(FakeOptions::default());
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let refused = runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind("0.0.0.0:0").await.unwrap();
        tokio::time::timeout(LONG, serve(listener, fake.control(), HttpConfig::default())).await
    });
    assert!(
        matches!(refused, Ok(Err(ServeError::TokenRequired(_)))),
        "{refused:?}"
    );
    // With a token it serves, and asks for it.
    let listener = runtime
        .block_on(tokio::net::TcpListener::bind("0.0.0.0:0"))
        .unwrap();
    let port = listener.local_addr().unwrap().port();
    let config = HttpConfig {
        token: Some(TOKEN.to_owned()),
        ..HttpConfig::default()
    };
    runtime.spawn(serve(listener, fake.control(), config));
    let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    assert_eq!(get(addr, "/state").status, 401);
    let right = format!("Bearer {TOKEN}");
    assert_eq!(
        request(addr, "GET", "/state", &[("Authorization", &right)], b"").status,
        200
    );
    runtime.shutdown_background();
}

/// DNS rebinding: a page on `evil.example` re-points its name at 127.0.0.1 and
/// is then "same-origin" with this server for the browser. Its requests still
/// say `Host: evil.example`, and are refused for it.
#[test]
fn a_host_that_is_not_an_address_is_refused() {
    let h = start(|config| config.allowed_hosts = vec!["client.lan".to_owned()]);
    let port = h.addr.port();
    let with_host = |host: &str, method: &str, target: &str| {
        request(h.addr, method, target, &[("Host", host)], b"")
    };
    for host in [
        "evil.example".to_owned(),
        format!("evil.example:{port}"),
        format!("127.0.0.1.evil.example:{port}"),
        format!("localhost.evil.example:{port}"),
        "2130706433".to_owned(),
        format!("other.lan:{port}"),
        "[localhost]".to_owned(),
    ] {
        for (method, target) in [
            ("GET", "/state"),
            ("GET", "/frame.png"),
            ("POST", "/quit"),
            ("POST", "/script"),
        ] {
            let reply = with_host(&host, method, target);
            assert_eq!(reply.status, 403, "{method} {target} with Host: {host}");
            assert!(reply.error().contains("Host"), "{}", reply.error());
        }
    }
    assert!(!h.fake.has_quit(), "a rebinding page's /quit did nothing");
    for host in [
        format!("127.0.0.1:{port}"),
        "127.0.0.1".to_owned(),
        format!("localhost:{port}"),
        format!("LocalHost:{port}"),
        format!("[::1]:{port}"),
        format!("192.168.1.20:{port}"),
        format!("client.lan:{port}"),
        "CLIENT.LAN".to_owned(),
    ] {
        assert_eq!(
            with_host(&host, "GET", "/state").status,
            200,
            "Host: {host}"
        );
    }
}

/// A web page must not be able to drive the client: any request a browser
/// marks as coming from another origin is refused, whatever it asks for.
#[test]
fn cross_origin_requests_are_refused() {
    let h = start(|_| {});
    let own = format!("http://{}", h.addr);
    let with = |name: &str, value: &str, method: &str, target: &str, body: &str| {
        request(h.addr, method, target, &[(name, value)], body.as_bytes())
    };
    for origin in [
        "http://evil.example",
        "https://evil.example",
        "null",
        "http://localhost:1",
        &format!("http://127.0.0.1:{}", h.addr.port().wrapping_add(1)),
        &format!("https://{}", h.addr),
        "file://",
        "",
    ] {
        for (method, target, body) in [
            ("GET", "/state", ""),
            ("GET", "/frame.png", ""),
            ("POST", "/quit", ""),
            ("POST", "/input", r#"{"text":"pwned"}"#),
            ("POST", "/script", "client.quit()"),
            ("POST", "/connect", "127.0.0.1:7777"),
        ] {
            let reply = with("Origin", origin, method, target, body);
            assert_eq!(reply.status, 403, "{method} {target} from {origin:?}");
            assert_eq!(reply.error(), "cross-origin requests are not allowed");
        }
    }
    for site in ["cross-site", "same-site", "CROSS-SITE", "anything-else"] {
        let reply = with("Sec-Fetch-Site", site, "POST", "/quit", "");
        assert_eq!(reply.status, 403, "Sec-Fetch-Site: {site}");
        // An <img src> on another site: no Origin header, but still marked.
        assert_eq!(
            with("Sec-Fetch-Site", site, "GET", "/frame.png", "").status,
            403
        );
    }
    std::thread::sleep(Duration::from_millis(30));
    assert!(!h.fake.has_quit(), "a cross-origin /quit did nothing");
    let record = h.fake.record();
    assert!(
        record.inputs.is_empty() && record.connects.is_empty(),
        "{record:?}"
    );

    // The server's own origin, and what a browser sends for an address typed
    // into its location bar, are fine.
    assert_eq!(with("Origin", &own, "GET", "/state", "").status, 200);
    assert_eq!(
        with("Sec-Fetch-Site", "same-origin", "GET", "/state", "").status,
        200
    );
    assert_eq!(
        with("Sec-Fetch-Site", "none", "GET", "/frame.png", "").status,
        200
    );
}

/// `serve` can be handed any `Control`, not just the fake's hub.
#[test]
fn the_api_serves_any_control_behind_an_arc() {
    let fake = FakeClient::start(FakeOptions::default());
    let control: Arc<dyn space_harness::Control> = fake.control();
    let api = Api::new(
        control,
        HttpConfig::default(),
        "127.0.0.1:1".parse().unwrap(),
    )
    .unwrap();
    assert!(format!("{api:?}").contains("Api"));
    let debug = format!(
        "{:?}",
        HttpConfig {
            token: Some(TOKEN.to_owned()),
            ..HttpConfig::default()
        }
    );
    assert!(
        !debug.contains(TOKEN),
        "the token must not be printed: {debug}"
    );
}
