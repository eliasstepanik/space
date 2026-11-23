//! The Lua API, end to end, against the fake client: scripts go in as source
//! text, the outcome is checked on the report, on the fake's record of what it
//! was asked to do, and on the files written.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use space_harness::fake::{picture, FakeClient, FakeOptions};
use space_harness::image::decode_stored_png;
use space_harness::script::{run_script, ScriptConfig, ScriptErrorKind, ScriptReport, StopHandle};
use space_harness::{Key, Screen};

const LONG: Duration = Duration::from_secs(20);

/// A fresh, empty directory nothing else uses, inside the build's temp
/// directory.
fn scratch(name: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU32, Ordering};
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "lua-{name}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn config(out_dir: &Path) -> ScriptConfig {
    ScriptConfig {
        timeout: LONG,
        out_dir: out_dir.to_path_buf(),
        ..ScriptConfig::default()
    }
}

fn run_with(fake: &FakeClient, source: &str, config: &ScriptConfig) -> ScriptReport {
    run_script(fake.control(), source, config, &StopHandle::new())
}

/// Run `source` against a fresh fake client (screenshots, if any, go to a
/// directory that is thrown away).
fn run(source: &str) -> (ScriptReport, FakeClient) {
    let fake = FakeClient::start(FakeOptions::default());
    let report = run_with(&fake, source, &config(&scratch("run")));
    (report, fake)
}

#[track_caller]
fn passed(report: &ScriptReport) {
    assert!(report.ok, "the script failed: {report:#?}");
    assert_eq!(report.error, None);
    assert_eq!(report.exit_code(), 0);
}

#[track_caller]
fn failed(report: &ScriptReport, kind: ScriptErrorKind) -> &str {
    assert!(!report.ok, "the script passed: {report:#?}");
    assert_eq!(report.exit_code(), 1);
    let error = report.error.as_ref().expect("a failed script has an error");
    assert_eq!(error.kind, kind, "{error:?}");
    &error.message
}

/// The script from the task description: type an address, connect, thrust,
/// turn, check the ship moved, take a screenshot, quit.
#[test]
fn a_script_plays_a_whole_session() {
    let fake = FakeClient::start(FakeOptions::default());
    let dir = scratch("session");
    let report = run_with(
        &fake,
        r#"
client.type("127.0.0.1:7777"); client.press("enter")
client.wait_until(function() return client.state().screen == "online" end, 10)
client.key_down("w"); client.wait(0.2); client.key_up("w")
client.turn(30, -10)                    -- degrees
local s = client.state(); assert(s.speed > 1, "ship did not move")
local path = client.screenshot("after-thrust.png")
client.log("saved", path)
client.quit()
"#,
        &config(&dir),
    );
    passed(&report);

    // What the user "did", as the client saw it.
    let record = fake.record();
    assert_eq!(record.typed(), "127.0.0.1:7777");
    assert_eq!(record.connects, ["127.0.0.1:7777"]);
    assert_eq!(
        record.keys(),
        [
            (Key::Enter, true),
            (Key::Enter, false),
            (Key::Forward, true),
            (Key::Forward, false),
        ]
    );
    let state = fake.control().state();
    assert_eq!(state.screen, Screen::Online);
    assert!(state.speed > 1.0, "{state:?}");
    assert!((state.yaw_deg - 30.0).abs() < 1e-3 && (state.pitch_deg + 10.0).abs() < 1e-3);

    // The screenshot: where the script was told it is, a PNG of the online
    // screen, and listed in the report.
    let file = dir.join("after-thrust.png");
    assert_eq!(report.screenshots, [file.display().to_string()]);
    assert_eq!(report.output, [format!("saved {}", file.display())]);
    let (width, height, rgba) = decode_stored_png(&std::fs::read(&file).unwrap()).unwrap();
    assert_eq!((width, height), (64, 48));
    let seq = u64::from(rgba[0]); // pixel (0, 0) is [seq mod 256, 0, screen, 255]
    let frames: Vec<u64> = (0..=state.frame).filter(|f| f % 256 == seq).collect();
    assert!(
        frames
            .iter()
            .any(|f| rgba == picture(64, 48, Screen::Online, *f)),
        "the screenshot is not a frame of the online screen"
    );

    // And the client was asked to quit.
    let deadline = Instant::now() + LONG;
    while !fake.has_quit() {
        assert!(Instant::now() < deadline, "the client never quit");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// A failing `assert` ends the script with an error that carries the Lua
/// message and the line it is on.
#[test]
fn a_failing_assert_is_an_error_with_its_message_and_line() {
    let (report, _fake) =
        run("local x = 1\n\nassert(x > 1, \"ship did not move\")\nclient.log(\"unreachable\")");
    assert_eq!(
        failed(&report, ScriptErrorKind::Runtime),
        "script:3: ship did not move"
    );
    assert_eq!(report.error.as_ref().unwrap().line, Some(3));
    assert!(report.output.is_empty(), "nothing runs after the failure");

    let (report, _fake) = run("assert(false)");
    assert_eq!(
        failed(&report, ScriptErrorKind::Runtime),
        "script:1: assertion failed!"
    );

    let (report, _fake) = run("local ok = true\nerror(\"boom\")");
    assert_eq!(failed(&report, ScriptErrorKind::Runtime), "script:2: boom");
    assert_eq!(report.error.as_ref().unwrap().line, Some(2));

    // An error raised in a function is located where it is raised.
    let (report, _fake) = run("local function check(v)\n  if not v then error(\"bad value\") end\nend\ncheck(true)\ncheck(false)");
    assert_eq!(
        failed(&report, ScriptErrorKind::Runtime),
        "script:2: bad value"
    );

    // A non-string error value is still a failure.
    let (report, _fake) = run("error({code = 7})");
    assert_eq!(
        failed(&report, ScriptErrorKind::Runtime),
        "(error object is a table value)"
    );
    assert_eq!(report.error.as_ref().unwrap().line, None);

    // The chunk name is the script's name.
    let fake = FakeClient::start(FakeOptions::default());
    let named = ScriptConfig {
        name: "tests/thrust.lua".into(),
        ..config(&scratch("named"))
    };
    let report = run_with(&fake, "\n\n\n\nassert(1 == 2, \"numbers\")", &named);
    assert_eq!(
        failed(&report, ScriptErrorKind::Runtime),
        "tests/thrust.lua:5: numbers"
    );
    assert_eq!(report.error.as_ref().unwrap().line, Some(5));
}

/// Errors the interpreter raises itself have no position; the report then
/// says how far the script had got.
#[test]
fn an_interpreter_error_has_no_line_but_the_report_says_where_the_script_was() {
    let (report, _fake) =
        run("local t = nil\nclient.log(\"before\")\n\nlocal y = t.x\nclient.log(\"after\")");
    let message = failed(&report, ScriptErrorKind::Runtime);
    assert!(message.contains("index"), "{message}");
    assert_eq!(report.error.as_ref().unwrap().line, None);
    assert_eq!(report.last_call_line, Some(2));
    assert_eq!(report.output, ["before"]);

    let (report, _fake) = run("nosuchfunction(1)");
    let message = failed(&report, ScriptErrorKind::Runtime);
    assert!(message.contains("call"), "{message}");
    assert_eq!(report.last_call_line, None);
}

#[test]
fn pcall_catches_errors_and_the_script_goes_on() {
    let (report, _fake) = run(r#"
local ok, message = pcall(function() error("inner") end)
assert(not ok)
client.log(message)
local ok2, message2 = pcall(client.press, "no-such-key")
assert(not ok2)
client.log(message2)
return 1, "two", {3}
"#);
    passed(&report);
    assert_eq!(report.output[0], "script:2: inner");
    assert!(
        report.output[1]
            .starts_with("script:5: client.press: unknown key \"no-such-key\" (keys: w forward "),
        "{}",
        report.output[1]
    );
}

#[test]
fn a_script_that_does_not_compile_is_a_compile_error_with_its_line() {
    let (report, fake) = run("client.log('one')\nlocal x = = 1");
    let message = failed(&report, ScriptErrorKind::Compile);
    assert!(message.starts_with("script:2: parse error"), "{message}");
    assert_eq!(report.error.as_ref().unwrap().line, Some(2));
    assert!(report.output.is_empty(), "nothing ran");
    assert!(fake.record().inputs.is_empty());

    let (report, _fake) = run("client.log('unterminated");
    failed(&report, ScriptErrorKind::Compile);
    let (report, _fake) = run("");
    passed(&report);
}

/// The time limit stops a script whatever it is doing: spinning, sleeping, or
/// swallowing errors in a loop.
#[test]
fn the_timeout_stops_runaway_scripts() {
    for source in [
        "while true do end",
        "client.wait(3600)",
        "while true do pcall(client.wait, 3600) end",
        "while true do pcall(client.wait_frames, 100000) end",
        "client.wait_until(function() return false end, 3600)",
        "local function spin() return spin() end spin()",
        "while true do pcall(error, 'x') end",
    ] {
        let fake = FakeClient::start(FakeOptions::default());
        let limited = ScriptConfig {
            timeout: Duration::from_millis(300),
            ..config(&scratch("timeout"))
        };
        let started = Instant::now();
        let report = run_with(&fake, source, &limited);
        let elapsed = started.elapsed();
        let message = failed(&report, ScriptErrorKind::Timeout);
        assert!(message.contains("0.3 s"), "{source}: {message}");
        assert!(
            elapsed >= Duration::from_millis(300),
            "{source}: {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_secs(10),
            "{source}: took {elapsed:?}"
        );
        assert!(report.elapsed_ms >= 300 && report.elapsed_ms < 10_000);
    }
}

#[test]
fn a_running_script_can_be_stopped_from_outside() {
    for source in [
        "while true do end",
        "client.wait(3600)",
        "while true do pcall(client.wait, 3600) end",
    ] {
        let fake = FakeClient::start(FakeOptions::default());
        let stop = StopHandle::new();
        assert!(!stop.is_stopped());
        let runner = {
            let (control, stop) = (fake.control(), stop.clone());
            let config = config(&scratch("stop"));
            let source = source.to_owned();
            std::thread::spawn(move || run_script(control, &source, &config, &stop))
        };
        std::thread::sleep(Duration::from_millis(150));
        assert!(!runner.is_finished(), "{source}: ended on its own");
        let asked = Instant::now();
        stop.stop();
        let report = runner.join().unwrap();
        assert!(
            asked.elapsed() < Duration::from_secs(5),
            "{source}: slow to stop"
        );
        assert_eq!(
            failed(&report, ScriptErrorKind::Stopped),
            "the script was stopped"
        );
    }
    // A handle that is already triggered: the script does not start.
    let fake = FakeClient::start(FakeOptions::default());
    let stop = StopHandle::new();
    stop.stop();
    let report = run_script(
        fake.control(),
        "client.log('ran')",
        &config(&scratch("stopped")),
        &stop,
    );
    failed(&report, ScriptErrorKind::Stopped);
    assert!(report.output.is_empty());
}

/// Only the base functions, `math`, `string`, `table` and `client` exist:
/// nothing that reaches the operating system, the file system or other code.
#[test]
fn the_sandbox_has_no_way_out() {
    let (report, _fake) = run(r##"
assert(os == nil, "os")
assert(io == nil, "io")
assert(package == nil, "package")
assert(debug == nil, "debug")
assert(coroutine == nil, "coroutine")
assert(require == nil, "require")
assert(dofile == nil, "dofile")
assert(loadfile == nil, "loadfile")
assert(load == nil, "load")
assert(loadstring == nil, "loadstring")
assert(string.dump == nil, "string.dump")
-- What a test script does need is there, and complete.
assert(string.format("%5.2f|%d|%s", 3.14159, 42, "x") == " 3.14|42|x")
assert(("ab"):rep(3) == "ababab" and ("abc"):upper() == "ABC" and #"abc" == 3)
assert(string.match("127.0.0.1:7777", "(%d+)$") == "7777")
assert(tonumber("42") == 42 and tonumber("0x10") == 16 and tonumber("z") == nil)
local t = {}
table.insert(t, "b"); table.insert(t, 1, "a"); table.sort(t)
assert(table.concat(t, ",") == "a,b" and table.remove(t) == "b" and #t == 1)
assert(math.floor(3.7) == 3 and math.max(1, 5) == 5 and math.abs(-2) == 2)
assert(type(tostring) == "function" and select("#", 1, 2, 3) == 3)
local keys = {}
for k in pairs({a = 1, b = 2}) do keys[#keys + 1] = k end
for i, v in ipairs({10, 20}) do assert(v == i * 10) end
assert(#keys == 2)
assert(setmetatable({}, {__index = function(_, k) return k .. "!" end}).hi == "hi!")
-- The same table shapes that crash the unpatched piccolo 0.3.3.
local u = {}; u.a = 1; u.b = 2; u.c = 3; u.a = nil; u.d = 4
local v = { k0 = 1, k1 = 1, k2 = 1, 1, 2, 3, 4, 5, 6 }; v.zz = 1
assert(u.d == 4 and v.zz == 1)
client.log("sandbox ok")
"##);
    passed(&report);
    assert_eq!(report.output, ["sandbox ok"]);

    // Using what is not there fails the script; it does not reach anything.
    for source in [
        "os.execute('echo pwned')",
        "io.open('/etc/passwd')",
        "require('os')",
        "dofile('x.lua')",
        "load('return 1')()",
        "debug.getinfo(1)",
    ] {
        let (report, _fake) = run(source);
        failed(&report, ScriptErrorKind::Runtime);
    }
}

/// Memory, string length, recursion depth and pattern matching are bounded:
/// each of these ends in an error within the time limit, not in a dead or
/// swollen host.
#[test]
fn resource_exhaustion_ends_the_script_not_the_host() {
    let small = |name: &str| ScriptConfig {
        memory_limit: 8 << 20,
        timeout: LONG,
        ..config(&scratch(name))
    };
    for (source, kind) in [
        (
            "local t = {} while true do t[#t + 1] = ('x'):rep(1000) .. #t end",
            ScriptErrorKind::Limit,
        ),
        (
            "local s = 'x' while true do s = s .. s end",
            ScriptErrorKind::Limit,
        ),
        ("local s = ('x'):rep(1000000000)", ScriptErrorKind::Limit),
        (
            "local n = 40 string.find(('a'):rep(n), ('a?'):rep(n) .. ('a'):rep(n) .. 'b')",
            ScriptErrorKind::Runtime,
        ),
    ] {
        let fake = FakeClient::start(FakeOptions::default());
        let started = Instant::now();
        let report = run_with(&fake, source, &small("limits"));
        let message = failed(&report, kind);
        assert!(
            started.elapsed() < Duration::from_secs(15),
            "{source}: slow"
        );
        if kind == ScriptErrorKind::Limit {
            assert!(message.contains("limit"), "{source}: {message}");
        }
    }
    // Unbounded recursion: a "stack overflow" error, or the memory limit if
    // that comes first; either way an error, and soon.
    let fake = FakeClient::start(FakeOptions::default());
    let report = run_with(
        &fake,
        "local function f(n) return 1 + f(n + 1) end f(1)",
        &small("recursion"),
    );
    assert!(!report.ok);
    let error = report.error.unwrap();
    assert!(
        matches!(
            error.kind,
            ScriptErrorKind::Runtime | ScriptErrorKind::Limit
        ),
        "{error:?}"
    );
    // A limit cannot be caught by the script.
    let fake = FakeClient::start(FakeOptions::default());
    let report = run_with(
        &fake,
        "local ok = pcall(function() local s = 'x' while true do s = s .. s end end)\nclient.log('survived', ok)",
        &small("limits-pcall"),
    );
    failed(&report, ScriptErrorKind::Limit);
    assert!(report.output.is_empty());
}

/// `client.screenshot` writes below the configured directory and nowhere
/// else: absolute paths, `..`, drives and odd names are refused, and nothing
/// is created for them.
#[test]
fn screenshots_cannot_escape_the_output_directory() {
    let fake = FakeClient::start(FakeOptions::default());
    let root = scratch("escape");
    let out = root.join("shots");
    let absolute = root.join("absolute.png");
    let source = format!(
        r#"
local refused = {{
  "../outside.png", "..\\outside.png", "a/../../outside.png", "/outside.png",
  [[{absolute}]], "C:outside.png", "sub/../x.png", "./x.png", "x.png:stream",
  "nul.png", "x", "x.txt", "", "x.png.",
}}
for _, name in ipairs(refused) do
  local ok, message = pcall(client.screenshot, name)
  assert(not ok, "accepted " .. name)
  assert(string.find(message, "not a usable screenshot name", 1, true), message)
end
assert(not pcall(client.screenshot))
assert(not pcall(client.screenshot, 5))
local a = client.screenshot("ok.png")
local b = client.screenshot("nested/dir/ok.jpg")
client.log(a)
client.log(b)
"#,
        absolute = absolute.display()
    );
    let report = run_with(&fake, &source, &config(&out));
    passed(&report);

    let png = out.join("ok.png");
    let jpg = out.join("nested").join("dir").join("ok.jpg");
    assert_eq!(
        report.screenshots,
        [png.display().to_string(), jpg.display().to_string()]
    );
    assert_eq!(report.output, report.screenshots);
    assert!(decode_stored_png(&std::fs::read(&png).unwrap()).is_ok());
    let jpeg = std::fs::read(&jpg).unwrap();
    assert_eq!(jpeg[..3], [0xff, 0xd8, 0xff]);
    assert_eq!(&jpeg[6..10], b"JFIF");

    // Nothing but the two screenshots exists anywhere under the test's root.
    let mut files = Vec::new();
    let mut pending = vec![root.clone()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files.sort();
    let mut expected = vec![jpg, png];
    expected.sort();
    assert_eq!(files, expected);
    assert!(!root.parent().unwrap().join("outside.png").exists());
}

#[test]
fn the_number_of_screenshots_is_capped() {
    let fake = FakeClient::start(FakeOptions::default());
    let capped = ScriptConfig {
        max_screenshots: 2,
        ..config(&scratch("cap"))
    };
    let report = run_with(
        &fake,
        "for i = 1, 5 do client.screenshot('s' .. i .. '.png') end",
        &capped,
    );
    let message = failed(&report, ScriptErrorKind::Runtime);
    assert!(
        message.starts_with("script:1: client.screenshot: more than 2 screenshots"),
        "{message}"
    );
    assert_eq!(report.screenshots.len(), 2);
}

#[test]
fn state_connect_and_disconnect_as_a_script_sees_them() {
    let (report, fake) = run(r#"
local s = client.state()
assert(s.screen == "connect" and s.error == nil and s.position == nil and s.session == nil)
assert(s.width == 64 and s.height == 48 and s.frame >= 0 and s.fps > 0)
assert(s.address == "" and s.speed == 0 and s.handoffs == 0)
assert(s.scene == nil and s.scene_position == nil and s.scene_revision == 0 and s.retry_in_s == nil)

-- A server that is not there: connect() with a timeout raises, at this line.
local ok, message = pcall(client.connect, "10.0.0.1:1", 10)
assert(not ok)
client.log(message)
assert(client.state().screen == "error")

-- The error screen left behind is not mistaken for the next attempt's answer.
local online = client.connect("127.0.0.1:7777", 10)
assert(online.screen == "online" and online.session == 1 and online.shard == 1)
assert(#online.position == 3 and online.position[1] == 0 and #online.velocity == 3)
assert(online.address == "127.0.0.1:7777")
assert(online.scene == "demo-area" and online.scene_revision == 1)
assert(#online.scene_position == 3 and online.scene_position[3] == online.position[3])

client.disconnect()
client.wait_until(function() return client.state().screen == "connect" end)
assert(client.state().session == nil and client.state().scene == nil)

-- Without a timeout connect() only asks.
assert(client.connect("127.0.0.1:7777") == nil)
client.wait_until(function() return client.state().screen == "online" end)
"#);
    passed(&report);
    assert_eq!(
        report.output,
        ["script:9: client.connect: 10.0.0.1:1: cannot reach 10.0.0.1:1: connection refused"]
    );
    assert_eq!(
        fake.record().connects,
        ["10.0.0.1:1", "127.0.0.1:7777", "127.0.0.1:7777"]
    );
    assert_eq!(fake.record().disconnects, 1);
}

#[test]
fn keys_presses_holds_turns_and_the_pointer() {
    let (report, fake) = run(r#"
client.connect("127.0.0.1:7777", 10)
client.press("w")                       -- a movement key is held for two frames
assert(client.state().speed > 0, "a press did not thrust")
client.hold("space", 0.05)
client.press("Enter")                   -- names are case-insensitive
client.key_down("shift"); client.key_up("shift")
client.turn(90, 0); client.turn(-45, 20)
client.pointer(0.25, 0.75, 1)
client.pointer(1, 0)
local s = client.wait_until(function()
  local now = client.state()
  return math.abs(now.yaw_deg - 45) < 0.001 and now
end)
assert(math.abs(s.pitch_deg - 20) < 0.001)
"#);
    passed(&report);
    let record = fake.record();
    assert_eq!(
        record.keys(),
        [
            (Key::Forward, true),
            (Key::Forward, false),
            (Key::Up, true),
            (Key::Up, false),
            (Key::Enter, true),
            (Key::Enter, false),
            (Key::Down, true),
            (Key::Down, false),
        ]
    );
    let pointers: Vec<_> = record
        .inputs
        .iter()
        .filter_map(|input| match input {
            space_harness::ViewerInput::Pointer { x, y, buttons } => Some((*x, *y, *buttons)),
            _ => None,
        })
        .collect();
    assert_eq!(pointers, [(0.25, 0.75, 1), (1.0, 0.0, 0)]);
}

#[test]
fn waiting_frames_time_pixels_and_output() {
    let fake = FakeClient::start(FakeOptions::default());
    let lines: Arc<Mutex<Vec<String>>> = Arc::default();
    let live = {
        let lines = Arc::clone(&lines);
        ScriptConfig {
            on_output: Some(Arc::new(move |line: &str| {
                lines.lock().unwrap().push(line.to_owned());
            })),
            ..config(&scratch("misc"))
        }
    };
    let report = run_with(
        &fake,
        r#"
local t0, f0 = client.time(), client.state().frame
local reached = client.wait_frames(5)
assert(reached >= f0 + 5 and client.state().frame >= reached)
client.wait(0.05)
assert(client.time() - t0 >= 0.05)
assert(client.wait_frames(0) >= reached)

-- wait_until returns what the condition returned.
local n = 0
local got = client.wait_until(function() n = n + 1 return n >= 3 and "third" end)
assert(got == "third" and n == 3)
local ok, message = pcall(client.wait_until, function() return nil end, 0.05)
assert(not ok)
client.log(message)

-- The picture: the fake's connect screen is [(x + frame) % 256, y, 32, 255].
local r, g, b, a = client.pixel(0, 0)
assert(g == 0 and b == 32 and a == 255, "pixel (0,0)")
local r2, g2 = client.pixel(63, 47)
assert(g2 == 47)
assert(not pcall(client.pixel, 64, 0))
assert(not pcall(client.pixel, -1, 0))
assert(not pcall(client.pixel, 0.5, 0))

print("a", 1, true, nil, 2.5)
client.log("x", 2, {} ~= nil)
"#,
        &live,
    );
    passed(&report);
    assert_eq!(
        report.output,
        [
            "script:13: client.wait_until: the condition was not met within 0.05 s",
            "a\t1\ttrue\tnil\t2.5",
            "x 2 true",
        ]
    );
    assert_eq!(
        *lines.lock().unwrap(),
        report.output,
        "the live sink saw every line"
    );
    assert_eq!(report.last_call_line, Some(27));
}

/// Wrong arguments are Lua errors at the caller's line that say what was
/// expected, and nothing is sent to the client for them.
#[test]
fn wrong_arguments_are_located_errors() {
    for (source, expected) in [
        (
            "client.press('hyper')",
            "script:1: client.press: unknown key \"hyper\"",
        ),
        (
            "\nclient.key_down()",
            "script:2: client.key_down: expected a key name (a string), got nil",
        ),
        (
            "client.key_up(5)",
            "script:1: client.key_up: expected a key name (a string), got 5",
        ),
        (
            "client.type(5)",
            "script:1: client.type: expected the text to type (a string), got 5",
        ),
        (
            "client.type('a\\27b')",
            "script:1: client.type: cannot type the control character U+001B",
        ),
        (
            "client.wait('x')",
            "script:1: client.wait: expected the number of seconds (a number), got \"x\"",
        ),
        (
            "client.wait(-1)",
            "script:1: client.wait: cannot wait -1 seconds",
        ),
        (
            "client.wait(0/0)",
            "script:1: client.wait: the number of seconds must be finite",
        ),
        (
            "client.wait_frames(-1)",
            "script:1: client.wait_frames: the number of frames must be a whole number",
        ),
        (
            "client.wait_frames(1.5)",
            "script:1: client.wait_frames: the number of frames must be a whole number",
        ),
        (
            "client.turn(1)",
            "script:1: client.turn: expected the pitch in degrees (a number), got nil",
        ),
        (
            "client.turn(1/0, 0)",
            "script:1: client.turn: the yaw in degrees must be finite",
        ),
        (
            "client.pointer(2, 0)",
            "script:1: client.pointer: x and y must be in 0..1",
        ),
        (
            "client.pointer(0, 0, 256)",
            "script:1: client.pointer: the button mask must be a whole number in 0..=255",
        ),
        (
            "client.connect('')",
            "script:1: client.connect: server address: empty",
        ),
        (
            "client.connect('a b')",
            "script:1: client.connect: server address: contains a space",
        ),
        (
            "client.connect(nil, 1)",
            "script:1: client.connect: expected the server address (a string), got nil",
        ),
        (
            "client.connect('127.0.0.1:7777', 'soon')",
            "script:1: client.connect: expected a number of seconds, got soon",
        ),
        (
            "\n\nclient.hold('w')",
            "script:3: client.hold: expected a number of seconds, got nil",
        ),
        (
            "client.hold('nope', 1)",
            "script:1: client.key_down: unknown key \"nope\"",
        ),
        (
            "client.wait_until(5)",
            "script:1: client.wait_until: expected a function, got number",
        ),
        (
            "client.wait_until(function() end, -1)",
            "script:1: client.wait_until: expected a number of seconds, got -1",
        ),
    ] {
        let (report, fake) = run(source);
        let message = failed(&report, ScriptErrorKind::Runtime);
        assert!(
            message.starts_with(expected),
            "{source}\n  got:      {message}\n  expected: {expected}"
        );
        let line = expected["script:".len()..]
            .split(':')
            .next()
            .unwrap()
            .parse()
            .ok();
        assert_eq!(report.error.as_ref().unwrap().line, line, "{source}");
        let record = fake.record();
        assert!(
            record.inputs.is_empty() && record.connects.is_empty(),
            "{source}: {record:?}"
        );
    }
}

/// A script that dies between a `key_down` and its `key_up` must not leave the
/// ship thrusting: whatever ends the script, its held keys are released.
#[test]
fn keys_a_script_left_down_are_released_when_it_ends() {
    let (report, fake) = run(
        "client.key_down('w')\nclient.key_down('space')\nclient.key_up('space')\nerror('oops')",
    );
    assert_eq!(failed(&report, ScriptErrorKind::Runtime), "script:4: oops");
    assert!(fake.wait_for(LONG, |s| s.frame > 0));
    let deadline = Instant::now() + LONG;
    while fake.record().keys().len() < 4 {
        assert!(Instant::now() < deadline, "the key was never released");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        fake.record().keys(),
        [
            (Key::Forward, true),
            (Key::Up, true),
            (Key::Up, false),
            (Key::Forward, false),
        ]
    );

    // The same when the script runs out of time while holding a key.
    let fake = FakeClient::start(FakeOptions::default());
    let short = ScriptConfig {
        timeout: Duration::from_millis(100),
        ..config(&scratch("held"))
    };
    let report = run_with(&fake, "client.hold('d', 3600)", &short);
    failed(&report, ScriptErrorKind::Timeout);
    let deadline = Instant::now() + LONG;
    while fake.record().keys().len() < 2 {
        assert!(Instant::now() < deadline, "the key was never released");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        fake.record().keys(),
        [(Key::Right, true), (Key::Right, false)]
    );
}

/// A client whose render loop hangs fails the script with a clear message
/// after the frame timeout; it does not hang the script.
#[test]
fn a_frozen_client_fails_the_script_instead_of_hanging_it() {
    for (source, expected) in [
        (
            "client.wait_frames(1)",
            "script:1: client.wait_frames: the client rendered no frame for 0.2 s",
        ),
        (
            "client.screenshot('x.png')",
            "script:1: client.screenshot: the client rendered no frame within 0.2 s",
        ),
        (
            "client.pixel(0, 0)",
            "script:1: client.pixel: the client rendered no frame within 0.2 s",
        ),
        (
            "client.press('w')",
            "script:1: client.press: the client rendered no frame for 0.2 s",
        ),
    ] {
        let fake = FakeClient::start(FakeOptions::default());
        assert!(fake.wait_for(LONG, |s| s.frame > 2));
        fake.set_paused(true);
        std::thread::sleep(Duration::from_millis(20));
        let impatient = ScriptConfig {
            frame_timeout: Duration::from_millis(200),
            ..config(&scratch("frozen"))
        };
        let started = Instant::now();
        let report = run_with(&fake, source, &impatient);
        assert_eq!(failed(&report, ScriptErrorKind::Runtime), expected);
        assert!(started.elapsed() < Duration::from_secs(10));
        assert!(report.screenshots.is_empty());
    }
}

#[test]
fn calls_after_the_client_quit_fail_cleanly() {
    let (report, fake) = run(
        "client.quit()\nclient.wait_until(function() return not pcall(client.key_down, 'w') end, 10)\nclient.type('x')",
    );
    assert_eq!(
        failed(&report, ScriptErrorKind::Runtime),
        "script:3: client.type: the client has shut down"
    );
    assert!(fake.has_quit());
}

#[test]
fn output_is_capped_and_every_run_starts_clean() {
    let fake = FakeClient::start(FakeOptions::default());
    let quiet = ScriptConfig {
        max_output_bytes: 200,
        ..config(&scratch("output"))
    };
    let report = run_with(
        &fake,
        "leftover = 5\nfor i = 1, 1000 do print('line ' .. i) end\nprint(('x'):rep(100000))",
        &quiet,
    );
    passed(&report);
    assert!(report.output.len() < 40, "{}", report.output.len());
    assert_eq!(report.output[0], "line 1");
    assert_eq!(
        report.output.last().unwrap(),
        "... (further output dropped)"
    );

    // A new run is a new interpreter: no globals, no string-library edits.
    let report = run_with(
        &fake,
        "assert(leftover == nil, 'a global survived')\nstring.format = nil",
        &quiet,
    );
    passed(&report);
    let report = run_with(&fake, "assert(string.format('%d', 1) == '1')", &quiet);
    passed(&report);

    // One very long line is cut, not dropped.
    let report = run_with(&fake, "print(('x'):rep(100000))", &config(&scratch("long")));
    passed(&report);
    assert_eq!(report.output.len(), 1);
    assert_eq!(report.output[0].len(), 4096);
}
