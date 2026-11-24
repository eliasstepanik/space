//! Lua scripts that play the part of a user.
//!
//! ```lua
//! client.type("127.0.0.1:7777"); client.press("enter")
//! client.wait_until(function() return client.state().screen == "online" end, 10)
//! client.key_down("w"); client.wait(2.0); client.key_up("w")
//! client.turn(30, -10)                    -- degrees
//! local s = client.state(); assert(s.speed > 1, "ship did not move")
//! client.screenshot("after-thrust.png")
//! client.quit()
//! ```
//!
//! A script is a test: [`run_script`] returns a [`ScriptReport`], and a failing
//! `assert`/`error` makes it a failure carrying the Lua message and line
//! (`script:5: ship did not move`), which the caller turns into a non-zero
//! exit ([`ScriptReport::exit_code`]).
//!
//! # The `client` table
//!
//! Every function raises a Lua error (with the calling line) on a wrong
//! argument or when the client refuses the request.
//!
//! | function | what it does |
//! |---|---|
//! | `client.key_down(key)` / `client.key_up(key)` | press / release a key and leave it that way |
//! | `client.press(key)` | one key press: a text-entry key is a stroke; a movement key is held for two frames |
//! | `client.hold(key, seconds)` | hold a key for that long, then release it |
//! | `client.type(text)` | type the characters (`"\n"` is Enter, `"\t"` is Tab) |
//! | `client.turn(dyaw, dpitch)` | turn the view, degrees (positive yaw left, positive pitch up) |
//! | `client.pointer(x, y [, buttons])` | pointer position, `0..1` across the picture, and the button mask |
//! | `client.connect(address [, timeout])` | ask for a connection; with a timeout (seconds) wait until online, raising on the error screen or on timeout |
//! | `client.disconnect()` | back to the connect prompt |
//! | `client.state()` | a table: `screen` (`"connect"`, `"connecting"`, `"online"`, `"error"`), `error`, `address`, `shard`, `session`, `tick`, `position` (world frame), `scene_position` (metres from the scene's spawn point: what the HUD shows), `velocity`, `speed`, `yaw_deg`, `pitch_deg`, `handoffs`, `scene` (the name of the scene being drawn, `nil` while not on a server), `scene_revision`, `retry_in_s` (seconds until the client reconnects on its own, or `nil`), `frame`, `width`, `height`, `fps`; physics: `gravity` (m/s² at the ship), `landed`, `bodies` (the physics bodies drawn, nearest first: each a table with `guid`, `visual` (`"crate"`, `"ball"`, `"door"`, `"chain"`, `"pendulum"`, ...), `visual_id`, `position` (scene frame), `distance`, `speed`, `parts`, `in_view`), `bodies_in_view`, `correction_m`, `correction_max_m` (the own ship's prediction corrections); `shard_address` (the `ip:port` of the shard being played on, `nil` while not on a server — `address` is the entry address that was dialed, which may have sent the client on); `fast` (fast travel on), `nearest` / `nearest_surface_m` (the solar-system body whose surface is nearest, and how far: `"Moon"`, 12.5), `sky` (the solar system's bodies: `name`, `position` (scene frame), `radius`, `surface_m`) |
//! | `client.wait(seconds)` | do nothing for that long |
//! | `client.wait_frames(n)` | wait until `n` more frames were rendered; returns the frame number |
//! | `client.wait_until(fn [, timeout [, interval]])` | call `fn` until it returns a true value (returned), raising after `timeout` seconds (default 10) |
//! | `client.screenshot(path)` | write the current picture (`.png`, `.jpg`) below the output directory; returns the path written |
//! | `client.pixel(x, y)` | `r, g, b, a` of one pixel of the current picture |
//! | `client.log(...)` / `print(...)` | a line of captured output |
//! | `client.time()` | seconds since the script started |
//! | `client.quit()` | ask the client to exit |
//!
//! Key names: see [`crate::key_from_name`].
//!
//! # Sandbox
//!
//! A script is untrusted-ish input (it can arrive over HTTP). Each run gets a
//! fresh interpreter with **only** the base functions, `math`, `string`,
//! `table` and `client`: no `os`, `io`, `package`, `debug`, `coroutine`,
//! `require`, `dofile`, `loadfile` or `load`. It has a wall-clock limit
//! ([`ScriptConfig::timeout`]), a memory limit and a string-length limit, and
//! it can be stopped from outside ([`StopHandle`]): the interpreter runs in
//! small steps and every blocking call wakes up to look, so neither
//! `while true do end` nor `pcall` in a loop outlives a stop.
//! `client.screenshot` writes only below one directory
//! ([`ScriptConfig::out_dir`]); absolute paths and `..` are refused.
//!
//! # The interpreter
//!
//! `lunita`, a pure-Rust Lua VM (the project allows no C dependency, so not
//! `mlua`). It is a fork of `piccolo` hardened for exactly this use. `piccolo`
//! 0.3.3 itself was evaluated first and is not usable here: it panics on
//! ordinary table code (`local t = {} t.a=1 t.b=2 t.c=3 t.a=nil t.d=4`), has
//! no `tonumber`, `string.format` or `table.insert`, and cannot bound memory
//! or the time of one step. Everything that touches the interpreter is in
//! this file.

use std::cell::{Cell, RefCell};
use std::fmt;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use lunita::{
    Callback, CallbackReturn, Closure, Context, Error as LuaError, Execution, Executor, Fuel,
    Limits, Lua, Stack, Table, Value, Variadic,
};
use serde::{Deserialize, Serialize};

use crate::control::{key_from_name, key_names, Control, ControlError, Key, ViewerInput};
use crate::image::{encode_frame, ImageFormat};

/// Receives each line of script output as it is produced (for a live log).
pub type OutputSink = Arc<dyn Fn(&str) + Send + Sync>;

/// How a script is run.
#[derive(Clone)]
pub struct ScriptConfig {
    /// The chunk name in error positions (`name:line: message`). Letters,
    /// digits and `._-/` only, at most 48 bytes; anything else is replaced by
    /// `"script"`.
    pub name: String,
    /// Wall-clock limit for the whole script.
    pub timeout: Duration,
    /// The one directory `client.screenshot` may write below (created on the
    /// first screenshot).
    pub out_dir: PathBuf,
    /// Screenshots one script may write.
    pub max_screenshots: usize,
    /// How long `client.screenshot`, `client.pixel`, `client.press` and
    /// `client.wait_frames` wait for the client to render a frame before they
    /// fail (a client that stopped rendering fails the script instead of
    /// hanging it).
    pub frame_timeout: Duration,
    /// Memory the interpreter may use, in bytes.
    pub memory_limit: usize,
    /// Captured output kept, in bytes; the rest is dropped (and said so).
    pub max_output_bytes: usize,
    /// JPEG quality of `.jpg` screenshots, `1..=100`.
    pub jpeg_quality: u8,
    /// Called with each output line as it is produced.
    pub on_output: Option<OutputSink>,
}

impl Default for ScriptConfig {
    /// One minute, screenshots under `./harness-out`, 64 MiB.
    fn default() -> Self {
        Self {
            name: DEFAULT_NAME.to_owned(),
            timeout: Duration::from_secs(60),
            out_dir: PathBuf::from("harness-out"),
            max_screenshots: 256,
            frame_timeout: Duration::from_secs(5),
            memory_limit: 64 << 20,
            max_output_bytes: 256 << 10,
            jpeg_quality: 90,
            on_output: None,
        }
    }
}

impl fmt::Debug for ScriptConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScriptConfig")
            .field("name", &self.name)
            .field("timeout", &self.timeout)
            .field("out_dir", &self.out_dir)
            .field("max_screenshots", &self.max_screenshots)
            .field("frame_timeout", &self.frame_timeout)
            .field("memory_limit", &self.memory_limit)
            .field("max_output_bytes", &self.max_output_bytes)
            .field("jpeg_quality", &self.jpeg_quality)
            .field("on_output", &self.on_output.as_ref().map(|_| "..."))
            .finish()
    }
}

/// Stops a running script from another thread. Clones share the flag.
#[derive(Clone, Debug, Default)]
pub struct StopHandle(Arc<AtomicBool>);

impl StopHandle {
    /// A handle that has not been triggered.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Ask the script to stop. It ends with [`ScriptErrorKind::Stopped`] within
    /// a few milliseconds, whatever it is doing.
    pub fn stop(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Whether [`stop`](Self::stop) was called.
    #[must_use]
    pub fn is_stopped(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// What kind of failure ended a script.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScriptErrorKind {
    /// The source did not compile.
    Compile,
    /// A Lua error: a failed `assert`, an `error(...)`, a `client` call that
    /// failed, or a runtime error such as indexing `nil`.
    Runtime,
    /// The script ran past [`ScriptConfig::timeout`].
    Timeout,
    /// The script was stopped ([`StopHandle::stop`]).
    Stopped,
    /// The script exceeded its memory or string-length limit.
    Limit,
    /// The harness itself failed (a bug: please report the message).
    Internal,
}

/// Why a script failed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScriptError {
    /// The kind of failure.
    pub kind: ScriptErrorKind,
    /// The message, with its position when Lua knows one
    /// (`script:5: ship did not move`).
    pub message: String,
    /// The line of the script the error is at: the position in `message`.
    /// `assert`, `error` and failing `client` calls have one; a compile error
    /// has the line the parser stopped at. Runtime errors raised by the
    /// interpreter itself (indexing `nil`, calling `nil`, arithmetic on a
    /// string) carry no position — see [`ScriptReport::last_call_line`].
    pub line: Option<u32>,
}

impl fmt::Display for ScriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ScriptError {}

/// The outcome of one script run. Serialises to the JSON `POST /script`
/// returns.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScriptReport {
    /// Whether the script ran to its end without an error.
    pub ok: bool,
    /// The failure, when `ok` is false.
    pub error: Option<ScriptError>,
    /// The lines the script printed (`print`, `client.log`), in order.
    pub output: Vec<String>,
    /// The screenshots it wrote, in order (the paths `client.screenshot`
    /// returned).
    pub screenshots: Vec<String>,
    /// The line of the last `client.*` / `print` call the script made: where
    /// it had got to. A hint for errors that have no line of their own.
    pub last_call_line: Option<u32>,
    /// How long the script ran, in milliseconds.
    pub elapsed_ms: u64,
}

impl ScriptReport {
    /// The process exit code for this outcome: 0 when the script passed, 1
    /// when it failed.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        i32::from(!self.ok)
    }

    fn internal(message: String) -> Self {
        Self {
            ok: false,
            error: Some(ScriptError {
                kind: ScriptErrorKind::Internal,
                message,
                line: None,
            }),
            output: Vec::new(),
            screenshots: Vec::new(),
            last_call_line: None,
            elapsed_ms: 0,
        }
    }
}

/// Run `source` against `control` and wait for it to end.
///
/// The interpreter runs on a thread of its own; this call blocks the calling
/// thread until the script has finished, failed, timed out
/// ([`ScriptConfig::timeout`]) or been stopped (`stop`). Keys the script left
/// held are released when it ends, however it ends. From async code call it
/// through `spawn_blocking`.
pub fn run_script(
    control: Arc<dyn Control>,
    source: &str,
    config: &ScriptConfig,
    stop: &StopHandle,
) -> ScriptReport {
    let (source, config, stop) = (source.to_owned(), config.clone(), stop.clone());
    let thread = std::thread::Builder::new()
        .name("space-script".into())
        .spawn(move || execute(control, &source, &config, &stop));
    match thread {
        Ok(thread) => thread.join().unwrap_or_else(|panic| {
            let what = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("(no message)");
            ScriptReport::internal(format!("the script interpreter panicked: {what}"))
        }),
        Err(e) => ScriptReport::internal(format!("cannot start the script thread: {e}")),
    }
}

// ---------------------------------------------------------------- the host

const DEFAULT_NAME: &str = "script";
/// The chunk name of the Lua half of the `client` table.
const PRELUDE_NAME: &str = "client.lua";
/// VM instructions (roughly) per interpreter step; between steps the stop flag
/// and the clock are checked and garbage is collected.
const FUEL_PER_STEP: i32 = 8192;
/// Longest Lua string a script may build.
const MAX_STRING_LEN: usize = 4 << 20;
/// Longest single output line kept, in bytes.
const MAX_LINE_LEN: usize = 4096;
/// How often a blocking call wakes up to look at the stop flag and the clock.
const POLL: Duration = Duration::from_millis(5);
/// How far up the call stack a failing `client` call looks for the script's
/// own line (it may have been called through the Lua half of the table).
const MAX_FRAME_LEVELS: usize = 16;

/// The functions of `client` that are written in Lua: the ones that call back
/// into the script (`wait_until`) or are a few lines over the native ones.
/// `fail` is a native function (taken off the table again) that raises its
/// message at the script's own line, however the function was reached.
const PRELUDE: &str = r#"
local client = client
local key_down, key_up, sleep, now = client.key_down, client.key_up, client.wait, client.time
local state, wait_frames, request = client.state, client.wait_frames, client.connect
local type, tostring = type, tostring
local fail = client.fail
client.fail = nil

local function seconds(name, value, default)
  if value == nil then return default end
  if type(value) ~= "number" or value ~= value or value < 0 then
    fail("client." .. name .. ": expected a number of seconds, got " .. tostring(value))
  end
  return value
end

function client.wait_until(condition, timeout, interval)
  if type(condition) ~= "function" then
    fail("client.wait_until: expected a function, got " .. type(condition))
  end
  timeout = seconds("wait_until", timeout, 10)
  interval = seconds("wait_until", interval, 0.01)
  local deadline = now() + timeout
  while true do
    local value = condition()
    if value then return value end
    local left = deadline - now()
    if left <= 0 then
      fail("client.wait_until: the condition was not met within " .. timeout .. " s")
    end
    sleep(interval < left and interval or left)
  end
end

function client.hold(key, duration)
  if type(duration) ~= "number" or duration ~= duration or duration < 0 then
    fail("client.hold: expected a number of seconds, got " .. tostring(duration))
  end
  key_down(key)
  sleep(duration)
  key_up(key)
end

function client.connect(address, timeout)
  timeout = seconds("connect", timeout, nil)
  request(address)
  if timeout == nil then return end
  -- Two frames, so that what is read below is the client's answer to this
  -- request and not the screen an earlier attempt left behind.
  wait_frames(2)
  local deadline = now() + timeout
  while true do
    local s = state()
    if s.screen == "online" then return s end
    if s.screen == "error" then
      fail("client.connect: " .. tostring(address) .. ": " .. tostring(s.error))
    end
    local left = deadline - now()
    if left <= 0 then
      fail("client.connect: " .. tostring(address) .. ": not online after " .. timeout
        .. " s (screen: " .. tostring(s.screen) .. ")")
    end
    sleep(0.01 < left and 0.01 or left)
  end
end
"#;

/// Why the script is being cut short.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Interrupt {
    Stopped,
    TimedOut,
}

/// A native function's failure: the message, to be given the caller's line.
type Fail = String;

#[derive(Default)]
struct Output {
    lines: Vec<String>,
    bytes: usize,
    truncated: bool,
}

/// What the native functions share. Lives on the script thread only.
struct Host {
    control: Arc<dyn Control>,
    stop: StopHandle,
    started: Instant,
    deadline: Instant,
    /// The script's chunk name.
    chunk: String,
    config: ScriptConfig,
    output: RefCell<Output>,
    screenshots: RefCell<Vec<String>>,
    /// The keys the script holds down right now.
    held: RefCell<[bool; Key::COUNT]>,
    last_call_line: Cell<Option<u32>>,
    interrupt: Cell<Option<Interrupt>>,
}

impl Host {
    /// Whether the script must end now, and why. Sticky.
    fn interrupted(&self) -> Option<Interrupt> {
        if self.interrupt.get().is_none() {
            if self.stop.is_stopped() {
                self.interrupt.set(Some(Interrupt::Stopped));
            } else if Instant::now() >= self.deadline {
                self.interrupt.set(Some(Interrupt::TimedOut));
            }
        }
        self.interrupt.get()
    }

    fn check_interrupt(&self) -> Result<(), Fail> {
        match self.interrupted() {
            None => Ok(()),
            Some(Interrupt::Stopped) => Err("the script was stopped".into()),
            Some(Interrupt::TimedOut) => Err("the script ran out of time".into()),
        }
    }

    /// Sleep, waking up to notice a stop or the deadline.
    fn sleep(&self, duration: Duration) -> Result<(), Fail> {
        let until = Instant::now().checked_add(duration);
        loop {
            self.check_interrupt()?;
            let left = match until {
                Some(until) => until.saturating_duration_since(Instant::now()),
                None => POLL,
            };
            if left.is_zero() {
                return Ok(());
            }
            std::thread::sleep(left.min(POLL));
        }
    }

    /// Wait until the client has rendered `frames` more frames. Fails when no
    /// frame at all arrives for `frame_timeout` (the client stopped rendering).
    fn wait_frames(&self, frames: u64) -> Result<u64, Fail> {
        let mut seen = self.control.state().frame;
        let target = seen.saturating_add(frames);
        let mut progress = Instant::now();
        loop {
            self.check_interrupt()?;
            let now = self.control.state().frame;
            if now >= target {
                return Ok(now);
            }
            if now != seen {
                seen = now;
                progress = Instant::now();
            } else if progress.elapsed() >= self.config.frame_timeout {
                return Err(format!(
                    "the client rendered no frame for {} s",
                    self.config.frame_timeout.as_secs_f32()
                ));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// The picture the client shows now. Waits in slices so a stop is noticed.
    fn current_frame(&self) -> Result<crate::Frame, Fail> {
        // (A timeout too large for the clock is bounded by the script's own.)
        let give_up = Instant::now()
            .checked_add(self.config.frame_timeout)
            .unwrap_or(self.deadline);
        loop {
            self.check_interrupt()?;
            let left = give_up.saturating_duration_since(Instant::now());
            match self
                .control
                .frame(None, left.min(Duration::from_millis(50)))
            {
                Ok(frame) => return Ok(frame),
                Err(ControlError::Timeout) if !left.is_zero() => {}
                Err(ControlError::Timeout) => {
                    return Err(format!(
                        "the client rendered no frame within {} s",
                        self.config.frame_timeout.as_secs_f32()
                    ))
                }
                Err(other) => return Err(other.to_string()),
            }
        }
    }

    fn key(&self, key: Key, down: bool) -> Result<(), Fail> {
        self.control.key(key, down).map_err(|e| e.to_string())?;
        self.held.borrow_mut()[key.index()] = down;
        Ok(())
    }

    /// Let go of every key the script still holds. A script that fails (or is
    /// stopped) between a `key_down` and its `key_up` must not leave the ship
    /// thrusting.
    fn release_keys(&self) {
        let held = std::mem::replace(&mut *self.held.borrow_mut(), [false; Key::COUNT]);
        for key in Key::ALL {
            if held[key.index()] {
                let _ = self.control.key(key, false);
            }
        }
    }

    fn print(&self, line: &str) {
        let mut line = line;
        if line.len() > MAX_LINE_LEN {
            let mut cut = MAX_LINE_LEN;
            while !line.is_char_boundary(cut) {
                cut -= 1;
            }
            line = &line[..cut];
        }
        let mut output = self.output.borrow_mut();
        if output.truncated {
            return;
        }
        if output.bytes + line.len() > self.config.max_output_bytes {
            output.truncated = true;
            output.lines.push("... (further output dropped)".to_owned());
            return;
        }
        output.bytes += line.len() + 1;
        output.lines.push(line.to_owned());
        drop(output);
        if let Some(sink) = &self.config.on_output {
            sink(line);
        }
    }

    fn screenshot(&self, requested: &str) -> Result<String, Fail> {
        let (path, format) = resolve_output_path(&self.config.out_dir, requested)?;
        if self.screenshots.borrow().len() >= self.config.max_screenshots {
            return Err(format!(
                "more than {} screenshots in one script",
                self.config.max_screenshots
            ));
        }
        let frame = self.current_frame()?;
        let bytes =
            encode_frame(&frame, format, self.config.jpeg_quality).map_err(|e| e.to_string())?;
        let shown = path.display().to_string();
        write_below(&self.config.out_dir, &path, &bytes).map_err(|e| format!("{shown}: {e}"))?;
        self.screenshots.borrow_mut().push(shown.clone());
        Ok(shown)
    }
}

/// Where `client.screenshot(requested)` may write, and in which format.
///
/// The name is a relative path of plain components below `out_dir`: no
/// absolute path, no drive, no `..`, no component a file system reads as
/// something else (trailing dots or spaces, device names, alternate data
/// streams). Both `/` and `\` separate components, on every platform.
fn resolve_output_path(out_dir: &Path, requested: &str) -> Result<(PathBuf, ImageFormat), Fail> {
    const MAX_LEN: usize = 200;
    const MAX_DEPTH: usize = 8;
    let refuse = |why: &str| {
        Err(format!(
            "{requested:?} is not a usable screenshot name: {why}"
        ))
    };
    if requested.is_empty() || requested.len() > MAX_LEN {
        return refuse("it is empty or too long");
    }
    if requested.starts_with(['/', '\\']) {
        return refuse("it is an absolute path");
    }
    let parts: Vec<&str> = requested.split(['/', '\\']).collect();
    if parts.len() > MAX_DEPTH {
        return refuse("it is nested too deeply");
    }
    for part in &parts {
        if part.is_empty() || part.chars().all(|c| c == '.') {
            return refuse("\".\", \"..\" and empty path components are not allowed");
        }
        if !part
            .chars()
            .all(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.' | ' '))
        {
            return refuse("only letters, digits, '-', '_', '.' and spaces are allowed");
        }
        if part.starts_with(' ') || part.ends_with([' ', '.']) {
            return refuse("a component starts with a space or ends with a space or dot");
        }
        let stem = part.split('.').next().unwrap_or("").to_ascii_uppercase();
        let device = matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
            || (stem.len() == 4
                && (stem.starts_with("COM") || stem.starts_with("LPT"))
                && stem.ends_with(|c: char| c.is_ascii_digit()));
        if device {
            return refuse("it names a device");
        }
    }
    let Some(format) = ImageFormat::from_file_name(parts[parts.len() - 1]) else {
        return refuse("it must end in .png, .jpg or .jpeg");
    };
    let mut path = out_dir.to_path_buf();
    path.extend(parts);
    Ok((path, format))
}

/// Write `bytes` to `path`, which [`resolve_output_path`] placed below
/// `out_dir`, creating directories. Refuses if the directory the file lands in
/// is not really below `out_dir` once links are resolved.
fn write_below(out_dir: &Path, path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path.parent().ok_or("no directory")?;
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let root = std::fs::canonicalize(out_dir).map_err(|e| e.to_string())?;
    let real = std::fs::canonicalize(parent).map_err(|e| e.to_string())?;
    if !real.starts_with(&root) {
        return Err("the directory is a link that leads out of the output directory".into());
    }
    std::fs::write(path, bytes).map_err(|e| e.to_string())
}

// ------------------------------------------------------ Lua <-> Rust values

fn lua_string<'gc>(ctx: Context<'gc>, text: &str) -> Value<'gc> {
    Value::String(ctx.intern(text.as_bytes()))
}

fn describe(value: Value<'_>) -> String {
    match value {
        Value::String(s) => format!("{:?}", String::from_utf8_lossy(s.as_bytes())),
        other => other.display().to_string(),
    }
}

fn arg_str<'gc>(stack: &Stack<'gc, '_>, at: usize, what: &str) -> Result<String, Fail> {
    match stack.get(at) {
        Value::String(s) => match s.to_str() {
            Ok(text) => Ok(text.to_owned()),
            Err(_) => Err(format!("{what} is not valid UTF-8")),
        },
        other => Err(format!(
            "expected {what} (a string), got {}",
            describe(other)
        )),
    }
}

fn arg_number(stack: &Stack<'_, '_>, at: usize, what: &str) -> Result<f64, Fail> {
    let number = match stack.get(at) {
        Value::Integer(i) => i as f64,
        Value::Number(n) => n,
        other => {
            return Err(format!(
                "expected {what} (a number), got {}",
                describe(other)
            ))
        }
    };
    if number.is_finite() {
        Ok(number)
    } else {
        Err(format!("{what} must be finite, got {number}"))
    }
}

fn arg_key(stack: &Stack<'_, '_>, at: usize) -> Result<Key, Fail> {
    let name = arg_str(stack, at, "a key name")?;
    key_from_name(&name).ok_or_else(|| format!("unknown key {name:?} (keys: {})", key_names()))
}

/// A whole number of pixels / frames in `0..=max`.
fn arg_count(stack: &Stack<'_, '_>, at: usize, what: &str, max: u64) -> Result<u64, Fail> {
    let number = arg_number(stack, at, what)?;
    if number < 0.0 || number.fract() != 0.0 || number > max as f64 {
        return Err(format!(
            "{what} must be a whole number in 0..={max}, got {number}"
        ));
    }
    Ok(number as u64)
}

fn json_to_lua<'gc>(
    ctx: Context<'gc>,
    json: &serde_json::Value,
) -> Result<Value<'gc>, LuaError<'gc>> {
    use serde_json::Value as Json;
    Ok(match json {
        Json::Null => Value::Nil,
        Json::Bool(b) => Value::Boolean(*b),
        Json::Number(n) => match n.as_i64() {
            Some(i) => Value::Integer(i),
            None => Value::Number(n.as_f64().unwrap_or(f64::NAN)),
        },
        Json::String(s) => lua_string(ctx, s),
        Json::Array(items) => {
            let table = Table::new(&ctx);
            for (i, item) in items.iter().enumerate() {
                table.set(ctx, i as i64 + 1, json_to_lua(ctx, item)?)?;
            }
            Value::Table(table)
        }
        Json::Object(fields) => {
            let table = Table::new(&ctx);
            for (name, item) in fields {
                table.set(ctx, lua_string(ctx, name), json_to_lua(ctx, item)?)?;
            }
            Value::Table(table)
        }
    })
}

/// The line of the script's own chunk that (directly, or through the Lua half
/// of `client`) made the running native call.
fn calling_line(exec: &Execution<'_, '_>, chunk: &str) -> Option<u32> {
    (1..=MAX_FRAME_LEVELS)
        .filter_map(|level| exec.lua_frame(level))
        .find(|frame| frame.chunk_name.as_bytes() == chunk.as_bytes())
        .and_then(|frame| u32::try_from(frame.current_line.0.saturating_add(1)).ok())
}

/// Wrap a native function for the `client` table: it records where the script
/// has got to, refuses to run once the script must end, and turns the
/// function's failure into a Lua error at the script's line, prefixed with
/// `name` (unless that is empty).
fn native<'gc, F>(
    ctx: Context<'gc>,
    host: &Rc<Host>,
    name: &'static str,
    function: F,
) -> Callback<'gc>
where
    F: for<'a, 'b> Fn(&Host, Context<'gc>, &'b mut Stack<'gc, 'a>) -> Result<(), Fail> + 'static,
{
    let host = Rc::clone(host);
    Callback::from_fn(&ctx, move |ctx, mut exec, mut stack| {
        let line = calling_line(&exec, &host.chunk);
        if line.is_some() {
            host.last_call_line.set(line);
        }
        let result = host
            .check_interrupt()
            .and_then(|()| function(&host, ctx, &mut stack));
        if host.interrupt.get().is_some() {
            // End this interpreter step now: the run loop is waiting to stop.
            exec.fuel().interrupt();
        }
        match result {
            Ok(()) => Ok(CallbackReturn::Return),
            Err(message) => {
                stack.clear();
                let message = if name.is_empty() {
                    message
                } else {
                    format!("{name}: {message}")
                };
                let message = match line {
                    Some(line) => format!("{}:{line}: {message}", host.chunk),
                    None => message,
                };
                Err(lua_string(ctx, &message).into())
            }
        }
    })
}

/// Build the sandbox: the base library minus everything that loads code,
/// `math`, `string`, `table`, `print` and the native half of `client`.
fn install<'gc>(ctx: Context<'gc>, host: &Rc<Host>) -> Result<(), LuaError<'gc>> {
    lunita::stdlib::load_base(ctx)?;
    lunita::stdlib::load_math(ctx)?;
    lunita::stdlib::load_string(ctx)?;
    lunita::stdlib::load_table(ctx)?;
    // The base library can compile source at run time; a test script has no
    // use for that. (None of the file-loading functions exist in this
    // interpreter; they are cleared anyway, in case an upgrade adds them.)
    for name in ["load", "loadstring", "loadfile", "dofile", "require"] {
        ctx.set_global(name, Value::Nil)?;
    }

    let client = Table::new(&ctx);
    let add = |name: &'static str, callback: Callback<'gc>| client.set_field(ctx, name, callback);

    add(
        "key_down",
        native(ctx, host, "client.key_down", |host, _, stack| {
            host.key(arg_key(stack, 0)?, true)?;
            stack.clear();
            Ok(())
        }),
    )?;
    add(
        "key_up",
        native(ctx, host, "client.key_up", |host, _, stack| {
            host.key(arg_key(stack, 0)?, false)?;
            stack.clear();
            Ok(())
        }),
    )?;
    add(
        "press",
        native(ctx, host, "client.press", |host, _, stack| {
            let key = arg_key(stack, 0)?;
            stack.clear();
            host.key(key, true)?;
            // A text-entry key acts when it goes down. A movement key only
            // matters while it is held, so keep it down until the client has
            // certainly run a frame with it.
            let held = if key.is_movement() {
                host.wait_frames(2).map(|_| ())
            } else {
                Ok(())
            };
            host.key(key, false)?;
            held
        }),
    )?;
    add(
        "type",
        native(ctx, host, "client.type", |host, _, stack| {
            let text = arg_str(stack, 0, "the text to type")?;
            stack.clear();
            host.control.type_text(&text).map_err(|e| e.to_string())
        }),
    )?;
    add(
        "turn",
        native(ctx, host, "client.turn", |host, _, stack| {
            let dyaw = arg_number(stack, 0, "the yaw in degrees")?;
            let dpitch = arg_number(stack, 1, "the pitch in degrees")?;
            stack.clear();
            host.control
                .turn_degrees(dyaw, dpitch)
                .map_err(|e| e.to_string())
        }),
    )?;
    add(
        "pointer",
        native(ctx, host, "client.pointer", |host, _, stack| {
            let x = arg_number(stack, 0, "x")?;
            let y = arg_number(stack, 1, "y")?;
            let buttons = match stack.get(2) {
                Value::Nil => 0,
                _ => arg_count(stack, 2, "the button mask", 255)? as u8,
            };
            stack.clear();
            if !(0.0..=1.0).contains(&x) || !(0.0..=1.0).contains(&y) {
                return Err(format!("x and y must be in 0..1, got ({x}, {y})"));
            }
            let (x, y) = (x as f32, y as f32);
            host.control
                .input(ViewerInput::Pointer { x, y, buttons })
                .map_err(|e| e.to_string())
        }),
    )?;
    add(
        // `client.connect` proper (with its optional wait) is in the prelude.
        "connect",
        native(ctx, host, "client.connect", |host, _, stack| {
            let address = arg_str(stack, 0, "the server address")?;
            stack.clear();
            host.control.connect(&address).map_err(|e| e.to_string())
        }),
    )?;
    add(
        // For the prelude only (it removes it): raise the message as it is.
        "fail",
        native(ctx, host, "", |_, _, stack| {
            Err(arg_str(stack, 0, "a message").unwrap_or_else(|e| e))
        }),
    )?;
    add(
        "disconnect",
        native(ctx, host, "client.disconnect", |host, _, stack| {
            stack.clear();
            host.control.disconnect().map_err(|e| e.to_string())
        }),
    )?;
    add(
        "quit",
        native(ctx, host, "client.quit", |host, _, stack| {
            stack.clear();
            host.control.quit();
            Ok(())
        }),
    )?;
    add(
        "state",
        native(ctx, host, "client.state", |host, ctx, stack| {
            let json = serde_json::to_value(host.control.state()).map_err(|e| e.to_string())?;
            let table = json_to_lua(ctx, &json).map_err(|e| e.to_string())?;
            stack.replace(ctx, table);
            Ok(())
        }),
    )?;
    add(
        "wait",
        native(ctx, host, "client.wait", |host, _, stack| {
            let seconds = arg_number(stack, 0, "the number of seconds")?;
            stack.clear();
            let duration = Duration::try_from_secs_f64(seconds)
                .map_err(|_| format!("cannot wait {seconds} seconds"))?;
            host.sleep(duration)
        }),
    )?;
    add(
        "wait_frames",
        native(ctx, host, "client.wait_frames", |host, ctx, stack| {
            let frames = arg_count(stack, 0, "the number of frames", 1_000_000)?;
            let reached = host.wait_frames(frames)?;
            stack.replace(ctx, i64::try_from(reached).unwrap_or(i64::MAX));
            Ok(())
        }),
    )?;
    add(
        "time",
        native(ctx, host, "client.time", |host, ctx, stack| {
            stack.replace(ctx, host.started.elapsed().as_secs_f64());
            Ok(())
        }),
    )?;
    add(
        "screenshot",
        native(ctx, host, "client.screenshot", |host, ctx, stack| {
            let requested = arg_str(stack, 0, "a file name")?;
            let written = host.screenshot(&requested)?;
            stack.replace(ctx, lua_string(ctx, &written));
            Ok(())
        }),
    )?;
    add(
        "pixel",
        native(ctx, host, "client.pixel", |host, _, stack| {
            let max = u64::from(u32::MAX);
            let x = arg_count(stack, 0, "x", max)? as u32;
            let y = arg_count(stack, 1, "y", max)? as u32;
            let frame = host.current_frame()?;
            let Some([r, g, b, a]) = frame.pixel(x, y) else {
                return Err(format!(
                    "({x}, {y}) is outside the {}x{} picture",
                    frame.width, frame.height
                ));
            };
            stack.clear();
            for channel in [r, g, b, a] {
                stack.push_back(Value::Integer(i64::from(channel)));
            }
            Ok(())
        }),
    )?;
    for (name, separator) in [("log", " "), ("print", "\t")] {
        let full_name = if name == "log" { "client.log" } else { "print" };
        let callback = native(ctx, host, full_name, move |host, _, stack| {
            let line: Vec<String> = stack
                .as_slice()
                .iter()
                .map(|value| value.display().to_string())
                .collect();
            stack.clear();
            host.print(&line.join(separator));
            Ok(())
        });
        if name == "log" {
            add(name, callback)?;
        } else {
            ctx.set_global(name, callback)?;
        }
    }
    ctx.set_global("client", client)?;
    Ok(())
}

// ------------------------------------------------------------------ running

/// An error and its causes as one line, innermost last. A cause that merely
/// repeats and extends what the outer error said ("parse error" caused by
/// "parse error at line 3: ...") replaces it.
fn error_chain(error: &(dyn std::error::Error + 'static)) -> String {
    let mut parts = vec![error.to_string()];
    let mut source = error.source();
    while let Some(inner) = source {
        let text = inner.to_string();
        match parts.last_mut() {
            Some(last) if text.starts_with(last.as_str()) => *last = text,
            _ => parts.push(text),
        }
        source = inner.source();
    }
    parts.join(": ")
}

/// The line in a position prefix `chunk:LINE: ...`.
fn line_of(message: &str, chunk: &str) -> Option<u32> {
    let rest = message.strip_prefix(chunk)?.strip_prefix(':')?;
    let (line, _) = rest.split_once(':')?;
    line.parse().ok()
}

/// The line in a parser message `... at line LINE: ...`.
fn parse_error_line(message: &str) -> Option<u32> {
    let (_, rest) = message.split_once("at line ")?;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

fn runtime_error(error: &LuaError<'_>, chunk: &str) -> ScriptError {
    match error {
        LuaError::Lua(lua) => {
            let message = match lua.0 {
                Value::String(s) => String::from_utf8_lossy(s.as_bytes()).into_owned(),
                Value::Nil => "nil".to_owned(),
                other => format!("(error object is a {} value)", other.type_name()),
            };
            ScriptError {
                kind: ScriptErrorKind::Runtime,
                line: line_of(&message, chunk),
                message,
            }
        }
        LuaError::Runtime(runtime) => match runtime.limit_error() {
            Some(limit) => ScriptError {
                kind: ScriptErrorKind::Limit,
                message: format!("the script exceeded its limits: {limit}"),
                line: None,
            },
            None => {
                let message = format!("{runtime:#}");
                ScriptError {
                    kind: ScriptErrorKind::Runtime,
                    line: line_of(&message, chunk),
                    message,
                }
            }
        },
    }
}

/// Compile and run one chunk to its end.
fn run_chunk(lua: &mut Lua, host: &Host, name: &str, source: &str) -> Result<(), ScriptError> {
    let internal = |message: String| ScriptError {
        kind: ScriptErrorKind::Internal,
        message,
        line: None,
    };
    let executor = lua
        .enter(
            |ctx| match Closure::load(ctx, Some(name), source.as_bytes()) {
                Ok(closure) => Ok(ctx.stash(Executor::start(ctx, closure.into(), ()))),
                Err(error) => Err(error_chain(&error)),
            },
        )
        .map_err(|message| {
            let line = parse_error_line(&message);
            ScriptError {
                kind: ScriptErrorKind::Compile,
                message: match line {
                    Some(line) => format!("{name}:{line}: {message}"),
                    None => format!("{name}: {message}"),
                },
                line,
            }
        })?;
    loop {
        if let Some(why) = host.interrupted() {
            return Err(interrupt_error(why, host));
        }
        let mut fuel = Fuel::with(FUEL_PER_STEP);
        match lua.enter(|ctx| ctx.fetch(&executor).step(ctx, &mut fuel)) {
            Ok(true) => break,
            Ok(false) => {}
            Err(mode) => return Err(internal(format!("the interpreter cannot step: {mode}"))),
        }
    }
    let finished = lua.enter(|ctx| {
        match ctx
            .fetch(&executor)
            .take_result::<Variadic<Vec<Value>>>(ctx)
        {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(error)) => Err(runtime_error(&error, &host.chunk)),
            Err(mode) => Err(internal(format!("the interpreter has no result: {mode}"))),
        }
    });
    // An error raised because the script had to end is that, whatever the
    // script made of it.
    match (finished, host.interrupted()) {
        (Err(_), Some(why)) => Err(interrupt_error(why, host)),
        (finished, _) => finished,
    }
}

fn interrupt_error(why: Interrupt, host: &Host) -> ScriptError {
    let (kind, message) = match why {
        Interrupt::Stopped => (
            ScriptErrorKind::Stopped,
            "the script was stopped".to_owned(),
        ),
        Interrupt::TimedOut => (
            ScriptErrorKind::Timeout,
            format!(
                "the script did not finish within {} s",
                host.config.timeout.as_secs_f32()
            ),
        ),
    };
    ScriptError {
        kind,
        message,
        line: None,
    }
}

fn chunk_name(requested: &str) -> String {
    let usable = !requested.is_empty()
        && requested.len() <= 48
        && requested != PRELUDE_NAME
        && requested
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-' | b'/'));
    if usable { requested } else { DEFAULT_NAME }.to_owned()
}

/// A seed scripts cannot know, for the interpreter's table hashing.
fn hash_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish()
}

/// The body of the script thread.
fn execute(
    control: Arc<dyn Control>,
    source: &str,
    config: &ScriptConfig,
    stop: &StopHandle,
) -> ScriptReport {
    let started = Instant::now();
    let host = Rc::new(Host {
        control,
        stop: stop.clone(),
        started,
        deadline: started
            .checked_add(config.timeout)
            .unwrap_or_else(|| started + Duration::from_secs(86_400 * 365)),
        chunk: chunk_name(&config.name),
        config: config.clone(),
        output: RefCell::default(),
        screenshots: RefCell::default(),
        held: RefCell::new([false; Key::COUNT]),
        last_call_line: Cell::new(None),
        interrupt: Cell::new(None),
    });

    let mut lua = Lua::with_hash_seed(hash_seed());
    lua.set_limits(Limits {
        memory: Some(config.memory_limit),
        string_len: Some(MAX_STRING_LEN),
    });
    let installed = lua.enter(|ctx| install(ctx, &host).map_err(|e| e.to_string()));
    let result = match installed {
        Err(message) => Err(ScriptError {
            kind: ScriptErrorKind::Internal,
            message: format!("cannot set up the interpreter: {message}"),
            line: None,
        }),
        Ok(()) => run_chunk(&mut lua, &host, PRELUDE_NAME, PRELUDE)
            .map_err(|mut error| {
                if error.kind == ScriptErrorKind::Compile || error.kind == ScriptErrorKind::Runtime
                {
                    error.kind = ScriptErrorKind::Internal;
                }
                error
            })
            .and_then(|()| run_chunk(&mut lua, &host, &host.chunk, source)),
    };
    host.release_keys();
    drop(lua);

    let output = std::mem::take(&mut host.output.borrow_mut().lines);
    let screenshots = std::mem::take(&mut *host.screenshots.borrow_mut());
    ScriptReport {
        ok: result.is_ok(),
        error: result.err(),
        output,
        screenshots,
        last_call_line: host.last_call_line.get(),
        elapsed_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn screenshot_names_stay_below_the_output_directory() {
        let out = Path::new("out");
        for (name, format) in [
            ("a.png", ImageFormat::Png),
            ("after-thrust.png", ImageFormat::Png),
            ("run 1/shot_02.JPG", ImageFormat::Jpeg),
            ("a/b/c/d.jpeg", ImageFormat::Jpeg),
            ("a\\b.png", ImageFormat::Png),
            ("größe.png", ImageFormat::Png),
            (".hidden.png", ImageFormat::Png),
            ("v1.2/x.png", ImageFormat::Png),
        ] {
            let (path, got) = resolve_output_path(out, name).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(got, format, "{name}");
            assert!(path.starts_with(out), "{name}: {path:?}");
            assert!(
                path.components()
                    .all(|c| matches!(c, std::path::Component::Normal(_))),
                "{name}: {path:?}"
            );
        }
        assert_eq!(
            resolve_output_path(out, "a\\b.png").unwrap().0,
            out.join("a").join("b.png")
        );
    }

    #[test]
    fn escaping_and_odd_screenshot_names_are_refused() {
        let out = Path::new("out");
        let long = format!("{}.png", "a".repeat(200));
        for name in [
            "",
            "/etc/passwd.png",
            "/abs.png",
            "\\abs.png",
            "\\\\server\\share\\x.png",
            "C:\\Windows\\x.png",
            "C:x.png",
            "c:/x.png",
            "../x.png",
            "a/../../x.png",
            "a/..\\x.png",
            "..",
            ".",
            "./x.png",
            "a//x.png",
            "a/./x.png",
            ".../x.png",
            "x.png/",
            "x.png.",
            "x.png ",
            " x.png",
            "a /x.png",
            "a./x.png",
            "x.png:stream",
            "x:y.png",
            "x\0.png",
            "x\n.png",
            "x*.png",
            "x?.png",
            "~/x.png",
            "$HOME/x.png",
            "%TEMP%/x.png",
            "con.png",
            "NUL.png",
            "a/com1.png",
            "LPT9.jpg",
            "aux",
            "x",
            "x.gif",
            "x.png.exe",
            "png",
            "a/b/c/d/e/f/g/h/i.png",
            long.as_str(),
        ] {
            let got = resolve_output_path(out, name);
            assert!(got.is_err(), "{name:?} was accepted: {got:?}");
        }
        // Near misses of the device rule are ordinary names.
        for name in [
            "console.png",
            "com.png",
            "com10.png",
            "nulls.png",
            "lpt.png",
        ] {
            assert!(resolve_output_path(out, name).is_ok(), "{name}");
        }
    }

    #[test]
    fn chunk_names_are_kept_only_when_plain() {
        assert_eq!(chunk_name("thrust.lua"), "thrust.lua");
        assert_eq!(chunk_name("tests/e2e/join-1.lua"), "tests/e2e/join-1.lua");
        let long = "x".repeat(49);
        for bad in ["", "a b", "a:1", "a\n", "ä", PRELUDE_NAME, long.as_str()] {
            assert_eq!(chunk_name(bad), DEFAULT_NAME, "{bad:?}");
        }
    }

    #[test]
    fn positions_are_read_out_of_messages() {
        assert_eq!(line_of("script:12: boom", "script"), Some(12));
        assert_eq!(line_of("a/b.lua:3: x: y", "a/b.lua"), Some(3));
        assert_eq!(line_of("other:12: boom", "script"), None);
        assert_eq!(line_of("script: boom", "script"), None);
        assert_eq!(line_of("scripted:1: boom", "script"), None);
        assert_eq!(line_of("boom", "script"), None);
        assert_eq!(
            parse_error_line("parse error at line 7: found \"Assign\""),
            Some(7)
        );
        assert_eq!(parse_error_line("no position here"), None);
    }

    #[test]
    fn the_report_is_plain_json_and_gives_an_exit_code() {
        let report = ScriptReport {
            ok: false,
            error: Some(ScriptError {
                kind: ScriptErrorKind::Runtime,
                message: "script:5: ship did not move".into(),
                line: Some(5),
            }),
            output: vec!["hello".into()],
            screenshots: vec!["out/a.png".into()],
            last_call_line: Some(5),
            elapsed_ms: 12,
        };
        assert_eq!(report.exit_code(), 1);
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["ok"], false);
        assert_eq!(json["error"]["kind"], "runtime");
        assert_eq!(json["error"]["message"], "script:5: ship did not move");
        assert_eq!(json["error"]["line"], 5);
        assert_eq!(json["output"][0], "hello");
        assert_eq!(json["screenshots"][0], "out/a.png");
        let back: ScriptReport = serde_json::from_value(json).unwrap();
        assert_eq!(back, report);
        let passed = ScriptReport {
            ok: true,
            error: None,
            ..report
        };
        assert_eq!(passed.exit_code(), 0);
        for (kind, name) in [
            (ScriptErrorKind::Compile, "compile"),
            (ScriptErrorKind::Timeout, "timeout"),
            (ScriptErrorKind::Stopped, "stopped"),
            (ScriptErrorKind::Limit, "limit"),
            (ScriptErrorKind::Internal, "internal"),
        ] {
            assert_eq!(serde_json::to_value(kind).unwrap(), name);
        }
    }
}
