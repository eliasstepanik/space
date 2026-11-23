//! `space-client` — the native client binary.
//!
//! The client is always online: it plays on an authoritative shard, and until
//! it does it shows the connect screen, where the server address is typed.
//! Given its server on the command line (`--server ADDR`) it also keeps the
//! connection by itself: a failed or lost one is retried after a short,
//! growing pause until somebody takes the controls.
//!
//! Only the window can be ended from its keyboard (Esc on the connect
//! screen). A headless or streaming client is a server-side process whose
//! keys come from whoever can reach it, so no input ends it: it runs until
//! the harness asks it to quit, `--frames` are rendered, its `--script` has
//! ended, or a signal arrives.
//! There is one application core ([`space_client::app::App`]) and three front
//! ends over it:
//!
//! * **window** (the default): a winit window presenting through the
//!   swapchain, with keyboard and mouse ([`window`]);
//! * **headless** (`--headless`): no window — an offscreen render loop at a
//!   capped rate, to be driven through the harness;
//! * **stream** (`--stream ADDR`): headless, and every frame is also served as
//!   JPEG over `ws://` to browser viewers, whose input drives the client.
//!
//! In every front end the harness can be attached: `--harness ADDR` serves the
//! control API over HTTP (state, frames, input, scripts) and `--script FILE`
//! runs a Lua script against the client and makes its outcome the exit code.

mod window;

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

use space_client::address::{ServerAddress, DEFAULT_PORT};
use space_client::app::{App, AppConfig, TestModel};
use space_client::headless::{run_headless, HeadlessConfig, DEFAULT_FPS};
use space_client::input::KEY_HELP;
use space_harness::http::{self, HttpConfig};
use space_harness::script::{run_script, ScriptConfig, StopHandle};
use space_harness::{Control, Hub, ScriptReport};
use space_render::{ContextOptions, QualityPreset, RenderContext, MAX_FRAME_DIMENSION};
use space_stream::StreamServer;

/// The environment variable the harness API's bearer token is read from (it
/// is never taken from the command line, where other users could read it).
const TOKEN_ENV: &str = "SPACE_HARNESS_TOKEN";

/// How long a script is given to end by itself once the client is closing
/// (it may be on its last statement) before it is stopped.
const SCRIPT_GRACE: Duration = Duration::from_secs(3);

/// How long a script or an API caller waits for a frame before concluding
/// that the client has stopped rendering. Generous on purpose: on software
/// Vulkan (lavapipe) the first frame - shader compilation, mesh upload,
/// acceleration structures - takes many seconds on a slow or busy machine,
/// which is exactly where a headless client is run. The harness's own
/// default (5 s) failed scripts there with "the client rendered no frame".
const FRAME_TIMEOUT: Duration = Duration::from_secs(30);

/// How the frames leave the process.
#[derive(Clone, Debug, PartialEq, Eq)]
enum FrontEnd {
    /// A window (the default).
    Window,
    /// No window: an offscreen loop.
    Headless,
    /// Headless, plus the `ws://` pixel stream on this address.
    Stream(String),
}

impl FrontEnd {
    /// Whether the person at this front end's keys may end the client with
    /// them (Esc on the connect screen): at the window, yes — it is their
    /// client. The input of a client without a window comes from whoever can
    /// reach its stream port or its harness, and must not be able to take
    /// somebody else's process down.
    fn user_may_quit(&self) -> bool {
        matches!(self, Self::Window)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Config {
    front_end: FrontEnd,
    /// `--server ADDR`: connect at start instead of showing the connect screen.
    server: Option<String>,
    /// `--size WxH` (default: [`Config::size`]).
    size: Option<(u32, u32)>,
    quality: QualityPreset,
    /// Windowed presentation waits for the vertical blank (`--no-vsync` clears it).
    vsync: bool,
    /// `--frames N`: stop after N frames.
    frames: Option<u64>,
    /// `--fps N`: the headless/stream frame rate cap.
    fps: u32,
    /// `--glb FILE`: a client-side test model near the spawn point.
    glb: Option<PathBuf>,
    /// `--glb-scale S`: metres per model unit (default: fit to ~3 m across).
    glb_scale: Option<f32>,
    /// `--harness ADDR`: serve the control API over HTTP there.
    harness: Option<String>,
    /// `--script FILE`: run this Lua script; its outcome is the exit code.
    script: Option<PathBuf>,
    /// `--script-timeout SECONDS`.
    script_timeout: Option<Duration>,
    /// `--out-dir DIR`: where scripts write screenshots.
    out_dir: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            front_end: FrontEnd::Window,
            server: None,
            size: None,
            quality: QualityPreset::default(),
            vsync: true,
            frames: None,
            fps: DEFAULT_FPS,
            glb: None,
            glb_scale: None,
            harness: None,
            script: None,
            script_timeout: None,
            out_dir: PathBuf::from("harness-out"),
        }
    }
}

impl Config {
    /// The frame size: what `--size` said, or 1280x720 for a window and
    /// 640x480 without one.
    fn size(&self) -> (u32, u32) {
        self.size.unwrap_or(match self.front_end {
            FrontEnd::Window => (1280, 720),
            FrontEnd::Headless | FrontEnd::Stream(_) => (640, 480),
        })
    }

    /// Whether anything but a browser viewer can be watching: the harness
    /// API or a script.
    fn harnessed(&self) -> bool {
        self.harness.is_some() || self.script.is_some()
    }
}

/// What the command line asked for.
#[derive(Clone, Debug, PartialEq)]
enum Command {
    Run(Box<Config>),
    Help,
}

/// Parse a `WxH` frame size; both sides must be in `1..=MAX_FRAME_DIMENSION`.
fn parse_size(text: &str) -> Option<(u32, u32)> {
    let (w, h) = text.split_once(['x', 'X'])?;
    let valid = 1..=MAX_FRAME_DIMENSION;
    let (w, h) = (w.parse().ok()?, h.parse().ok()?);
    (valid.contains(&w) && valid.contains(&h)).then_some((w, h))
}

/// Parse the arguments (without the program name).
fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Command, String> {
    let mut config = Config::default();
    let mut headless = false;
    let mut stream = None;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let mut value = |what: &str| args.next().ok_or(format!("{arg} needs {what}"));
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--headless" => headless = true,
            "--stream" => stream = Some(value("an address, e.g. 127.0.0.1:8099")?),
            "--server" => {
                let address = value("a server address, e.g. 127.0.0.1:7777")?;
                // Only the shape is checked here; whether it answers shows on
                // the connect screen.
                ServerAddress::parse(&address).map_err(|e| format!("--server: {e}"))?;
                config.server = Some(address);
            }
            "--size" => {
                config.size = Some(parse_size(&value("WxH")?).ok_or(format!(
                    "--size needs WxH with both sides in 1..={MAX_FRAME_DIMENSION}, \
                     e.g. 1280x720"
                ))?);
            }
            "--quality" => {
                config.quality = value("high, medium, low or ci")?
                    .parse()
                    .map_err(|e| format!("--quality: {e}"))?;
            }
            "--no-vsync" => config.vsync = false,
            "--frames" => {
                config.frames = Some(
                    value("a number")?
                        .parse()
                        .map_err(|_| "--frames needs a number".to_owned())?,
                );
            }
            "--fps" => {
                config.fps = value("a number")?
                    .parse()
                    .ok()
                    .filter(|fps| (1..=1000).contains(fps))
                    .ok_or("--fps needs a number from 1 to 1000")?;
            }
            "--glb" => config.glb = Some(PathBuf::from(value("a .glb file")?)),
            "--glb-scale" => {
                config.glb_scale = Some(
                    value("a positive number")?
                        .parse::<f32>()
                        .ok()
                        .filter(|scale| scale.is_finite() && *scale > 0.0)
                        .ok_or("--glb-scale needs a positive number")?,
                );
            }
            "--harness" => config.harness = Some(value("an address, e.g. 127.0.0.1:8098")?),
            "--script" => config.script = Some(PathBuf::from(value("a Lua file")?)),
            "--script-timeout" => {
                let seconds: f64 = value("a number of seconds")?
                    .parse()
                    .map_err(|_| "--script-timeout needs a number of seconds".to_owned())?;
                config.script_timeout = Some(
                    Duration::try_from_secs_f64(seconds)
                        .ok()
                        .filter(|timeout| !timeout.is_zero())
                        .ok_or("--script-timeout needs a positive number of seconds")?,
                );
            }
            "--out-dir" => config.out_dir = PathBuf::from(value("a directory")?),
            // The scene flags of the offline client. Ignored with a warning,
            // not refused: a deployment that still passes one (the proxy
            // compose did, as `CLIENT_ARGS=--galaxy`) must not lose its client
            // to an image update. The scene is the server's choice now.
            "--demo" | "--galaxy" | "--materials" | "--solar" => {
                eprintln!(
                    "space-client: {arg} is ignored: the scene is chosen on the server \
                     (space-server --scene NAME, or the orchestrator), not by the client"
                );
            }
            "--window" | "--out" => {
                return Err(format!(
                    "{arg} is gone: the client always plays on a shard, in a window unless \
                     --headless or --stream is given"
                ));
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    config.front_end = match stream {
        Some(addr) => FrontEnd::Stream(addr),
        None if headless => FrontEnd::Headless,
        None => FrontEnd::Window,
    };
    if config.glb_scale.is_some() && config.glb.is_none() {
        return Err("--glb-scale needs --glb FILE".into());
    }
    if config.script_timeout.is_some() && config.script.is_none() {
        return Err("--script-timeout needs --script FILE".into());
    }
    Ok(Command::Run(Box::new(config)))
}

fn usage() -> String {
    format!(
        "usage: space-client [--server ADDR] [--headless | --stream ADDR]\n\
        \x20                   [--size WxH] [--quality high|medium|low|ci] [--no-vsync]\n\
        \x20                   [--frames N] [--fps N] [--glb FILE [--glb-scale S]]\n\
        \x20                   [--harness ADDR] [--script FILE [--script-timeout S]]\n\
        \x20                   [--out-dir DIR]\n\
        \n\
        The client always plays on a shard. With no arguments a window opens on\n\
        the connect screen: type the server's address (host, host:port or\n\
        [ipv6]:port; the port defaults to {DEFAULT_PORT}) and press Enter. Esc quits\n\
        (the window only: see --headless).\n\
        The scene that is drawn (demo-area, materials, solar, galaxy) is the\n\
        server's choice, not the client's.\n\
        \n\
        While playing: {KEY_HELP}\n\
        (C toggles the chase camera and first person. B toggles the brake\n\
        assist, which stops the ship when no movement key is held. Esc returns\n\
        to the connect screen.)\n\
        \n\
        \x20 --server ADDR  connect to this shard at once instead of showing the\n\
        \x20                connect screen, and stay connected: if the connection\n\
        \x20                fails or is lost the connect screen says why and the\n\
        \x20                client tries again by itself (after 2, 3, 5, then\n\
        \x20                every 8 seconds) until it is back in. Editing the\n\
        \x20                address or pressing Esc stops the retrying\n\
        \x20 --headless     no window: render offscreen at a capped rate, to be\n\
        \x20                driven through --harness or --script. No key ends a\n\
        \x20                client without a window (Esc on its connect screen\n\
        \x20                only stops the retrying or cancels an attempt): it\n\
        \x20                ends through the harness (POST /quit, client.quit()),\n\
        \x20                after --frames, when its --script has ended, or by a\n\
        \x20                signal\n\
        \x20 --stream ADDR  headless, and serve the frames as JPEG over ws://ADDR\n\
        \x20                (plain ws://, no TLS) to the space-viewer page; the\n\
        \x20                viewer's keyboard and mouse drive the client, and\n\
        \x20                cannot end it\n\
        \x20 --size WxH     frame/window size in pixels (default 1280x720 in a\n\
        \x20                window, 640x480 without one)\n\
        \x20 --quality P    quality preset: high (default), medium, low or ci\n\
        \x20                (shadows, fog, screen-space reflections, refraction)\n\
        \x20 --no-vsync     window: present as fast as frames render\n\
        \x20 --frames N     exit after N frames\n\
        \x20 --fps N        headless/stream: frame rate cap (default {DEFAULT_FPS})\n\
        \x20 --glb FILE     also show a .glb model (binary glTF) near the spawn\n\
        \x20                point; only this client sees it\n\
        \x20 --glb-scale S  metres per model unit (default: fit to about 3 m)\n\
        \n\
        The harness (every front end): drive the client without a human.\n\
        \x20 --harness ADDR serve the control API over HTTP on ADDR (GET / lists\n\
        \x20                the routes: /state, /frame.png, /input, /connect,\n\
        \x20                /script, ...). Loopback needs no credentials; any other\n\
        \x20                address needs a bearer token in the environment\n\
        \x20                variable {TOKEN_ENV}\n\
        \x20 --script FILE  run a Lua script against this client (client.type,\n\
        \x20                client.press, client.connect, client.screenshot, ...)\n\
        \x20                and exit with its result: 0 passed, 1 failed (the\n\
        \x20                message goes to stderr)\n\
        \x20 --script-timeout S  stop the script after S seconds (default 60)\n\
        \x20 --out-dir DIR  where scripts write screenshots (default harness-out)"
    )
}

/// A fatal error: the message, and the process exit code it deserves.
struct Fatal {
    message: String,
    code: u8,
}

impl Fatal {
    /// Something went wrong at run time.
    fn runtime(message: impl std::fmt::Display) -> Self {
        Self {
            message: message.to_string(),
            code: 1,
        }
    }

    /// The command line (or the environment) asks for something impossible.
    fn usage(message: impl std::fmt::Display) -> Self {
        Self {
            message: message.to_string(),
            code: 2,
        }
    }
}

/// How scripts run here: screenshots under `--out-dir`, output echoed live.
fn script_config(config: &Config) -> ScriptConfig {
    let mut scripts = ScriptConfig {
        out_dir: config.out_dir.clone(),
        on_output: Some(Arc::new(|line: &str| println!("{line}"))),
        frame_timeout: FRAME_TIMEOUT,
        ..ScriptConfig::default()
    };
    if let Some(timeout) = config.script_timeout {
        scripts.timeout = timeout;
    }
    scripts
}

/// Serve the harness's HTTP API on `addr`. What the library would refuse (a
/// non-loopback address without a token) is refused here, before anything
/// else starts, with the way out.
fn start_harness(
    runtime: &tokio::runtime::Runtime,
    addr: &str,
    control: Arc<dyn Control>,
    scripts: ScriptConfig,
) -> Result<(), Fatal> {
    let listener = runtime
        .block_on(tokio::net::TcpListener::bind(addr))
        .map_err(|e| Fatal::usage(format!("--harness {addr}: cannot listen there: {e}")))?;
    let bound = listener
        .local_addr()
        .map_err(|e| Fatal::runtime(format!("--harness {addr}: {e}")))?;
    let token = std::env::var(TOKEN_ENV)
        .ok()
        .filter(|token| !token.is_empty());
    let config = HttpConfig {
        token,
        script: scripts,
        frame_timeout: FRAME_TIMEOUT,
        ..HttpConfig::default()
    };
    http::Api::new(Arc::clone(&control), config.clone(), bound).map_err(|refused| {
        Fatal::usage(format!(
            "--harness {addr}: {refused}\n\
             Use a loopback address (127.0.0.1:PORT), or put a token of at least {} \
             characters in the environment variable {TOKEN_ENV}; requests then need the \
             header \"Authorization: Bearer <token>\".",
            http::MIN_TOKEN_LEN
        ))
    })?;
    eprintln!(
        "space-client: harness API on http://{bound}/ ({})",
        if config.token.is_some() {
            "bearer token required"
        } else {
            "no token: loopback only"
        }
    );
    runtime.spawn(async move {
        // `serve` only returns to refuse the listener, which was ruled out above.
        if let Err(refused) = http::serve(listener, control, config).await {
            eprintln!("space-client: harness API stopped: {refused}");
        }
    });
    Ok(())
}

/// A `--script` run on its own thread.
struct ScriptRun {
    path: PathBuf,
    stop: StopHandle,
    thread: std::thread::JoinHandle<ScriptReport>,
}

impl ScriptRun {
    /// Read the script and start it against `hub`. When it ends — passed or
    /// failed — the client is asked to quit: the script's end is the
    /// process's end.
    fn start(path: PathBuf, hub: &Arc<Hub>, mut scripts: ScriptConfig) -> Result<Self, Fatal> {
        let source = std::fs::read_to_string(&path)
            .map_err(|e| Fatal::usage(format!("--script {}: {e}", path.display())))?;
        scripts.name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let stop = StopHandle::new();
        let (hub, stopper) = (Arc::clone(hub), stop.clone());
        let thread = std::thread::Builder::new()
            .name("space-script-host".into())
            .spawn(move || {
                let control: Arc<dyn Control> = hub.clone();
                let report = run_script(control, &source, &scripts, &stopper);
                hub.quit();
                report
            })
            .map_err(|e| Fatal::runtime(format!("cannot start the script thread: {e}")))?;
        Ok(Self { path, stop, thread })
    }

    /// The client has stopped: collect the script's outcome, print it and
    /// turn it into the exit code (0 passed, 1 failed).
    fn finish(self) -> u8 {
        // Normally the script ended first and asked for the quit. If the
        // client ended for another reason, the script may be on its last
        // lines; after that it is cut short (and has failed).
        let patience = Instant::now() + SCRIPT_GRACE;
        while !self.thread.is_finished() && Instant::now() < patience {
            std::thread::sleep(Duration::from_millis(10));
        }
        self.stop.stop();
        let name = self.path.display();
        let report = match self.thread.join() {
            Ok(report) => report,
            Err(_) => {
                eprintln!("space-client: script {name}: FAILED: the script host panicked");
                return 1;
            }
        };
        for screenshot in &report.screenshots {
            eprintln!("space-client: screenshot {screenshot}");
        }
        match &report.error {
            None => eprintln!("space-client: script {name}: ok ({} ms)", report.elapsed_ms),
            Some(error) => {
                let near = match (error.line, report.last_call_line) {
                    (None, Some(line)) => format!(" (after the call on line {line})"),
                    _ => String::new(),
                };
                eprintln!(
                    "space-client: script {name}: FAILED ({:?}): {error}{near}",
                    error.kind
                );
            }
        }
        u8::from(!report.ok)
    }
}

fn run(config: &Config) -> Result<u8, Fatal> {
    let model = match &config.glb {
        Some(path) => {
            let model = TestModel::load(path, config.glb_scale)
                .map_err(|e| Fatal::usage(format!("--glb {e}")))?;
            let mesh = model.mesh();
            eprintln!(
                "space-client: model {}: {} triangles, {} parts, {:.2} m across at scale {}",
                path.display(),
                mesh.triangle_count(),
                mesh.parts.len(),
                2.0 * mesh.radius() * model.scale(),
                model.scale()
            );
            Some(model)
        }
        None => None,
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|e| Fatal::runtime(format!("tokio runtime: {e}")))?;
    let hub = Hub::new();
    let scripts = script_config(config);
    if let Some(addr) = &config.harness {
        start_harness(&runtime, addr, hub.clone(), scripts.clone())?;
    }
    // Bind the stream before anything is drawn, so a taken port fails at once.
    let stream = match &config.front_end {
        FrontEnd::Stream(addr) => Some(
            runtime
                .block_on(StreamServer::bind(addr.as_str()))
                .map_err(|e| Fatal::usage(format!("--stream {addr}: cannot listen there: {e}")))?,
        ),
        _ => None,
    };
    let app = App::new(
        AppConfig {
            server: config.server.clone(),
            quality: config.quality,
            model,
            escape_quits: config.front_end.user_may_quit(),
        },
        runtime.handle().clone(),
        Arc::clone(&hub),
    );
    let (width, height) = config.size();
    // The script starts with the client: let it find the real state at once.
    app.announce(width, height);
    let script = match &config.script {
        Some(path) => Some(ScriptRun::start(path.clone(), &hub, scripts)?),
        None => None,
    };
    let outcome = match &config.front_end {
        FrontEnd::Window => window::run_window(
            app,
            &window::WindowConfig {
                width,
                height,
                vsync: config.vsync,
                frames: config.frames,
            },
        ),
        FrontEnd::Headless | FrontEnd::Stream(_) => {
            let headless = HeadlessConfig {
                width,
                height,
                fps: config.fps,
                frames: config.frames,
                idle_unwatched: !config.harnessed(),
            };
            run_offscreen(app, &headless, stream.as_ref())
        }
    };
    // The app is gone by now, and with it the harness's client: a script
    // still running finds the client closed.
    hub.close();
    let code = script.map(ScriptRun::finish);
    drop(stream);
    outcome.map_err(Fatal::runtime)?;
    Ok(code.unwrap_or(0))
}

/// The headless and stream front ends: the application in an offscreen loop.
fn run_offscreen(
    mut app: App,
    config: &HeadlessConfig,
    stream: Option<&(StreamServer, space_stream::StreamClient)>,
) -> Result<(), space_render::RenderError> {
    let ctx = RenderContext::new(&ContextOptions::default())?;
    eprintln!(
        "space-client: device {:?} ray_query={}; headless {}x{} at up to {} fps",
        ctx.device_name(),
        ctx.ray_query_support().complete(),
        config.width,
        config.height,
        config.fps
    );
    if let Some((server, _)) = stream {
        eprintln!(
            "space-client: streaming on ws://{} (plain ws://, no TLS)",
            server.local_addr()
        );
    }
    let stop = AtomicBool::new(false);
    let started = Instant::now();
    let report = run_headless(
        &ctx,
        &mut app,
        config,
        stream.map(|(_, client)| client),
        &stop,
    );
    // The link and the harness go first, then the GPU context.
    drop(app);
    let report = report?;
    eprintln!(
        "space-client: rendered {} frames in {:.1} s, {} validation messages",
        report.frames,
        started.elapsed().as_secs_f64(),
        report.validation_messages
    );
    Ok(())
}

fn main() -> ExitCode {
    // What the client does on its own initiative — connecting, why an attempt
    // failed, when it tries again, a scene change — is said on stderr unless
    // `RUST_LOG` asks for something else: an unattended client (a container)
    // has no other way to tell its operator. Everything else stays quiet.
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("error,space_client=info"),
    )
    .init();
    let config = match parse_args(std::env::args().skip(1)) {
        Ok(Command::Run(config)) => config,
        Ok(Command::Help) => {
            println!("{}", usage());
            return ExitCode::SUCCESS;
        }
        Err(message) => {
            eprintln!("space-client: {message}\n\n{}", usage());
            return ExitCode::from(2);
        }
    };
    match run(&config) {
        Ok(code) => ExitCode::from(code),
        Err(fatal) => {
            eprintln!("space-client: {}", fatal.message);
            ExitCode::from(fatal.code)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Config, String> {
        match parse_args(args.iter().map(|arg| (*arg).to_owned()))? {
            Command::Run(config) => Ok(*config),
            Command::Help => Err("help".into()),
        }
    }

    #[test]
    fn no_arguments_is_a_window_on_the_connect_screen() {
        let config = parse(&[]).expect("parses");
        assert_eq!(config, Config::default());
        assert_eq!(config.front_end, FrontEnd::Window);
        assert_eq!(config.server, None, "no server: the connect screen");
        assert_eq!(config.size(), (1280, 720));
        assert!(config.vsync && !config.harnessed());
        assert_eq!(config.out_dir, PathBuf::from("harness-out"));
    }

    #[test]
    fn the_front_end_is_window_headless_or_stream() {
        assert_eq!(
            parse(&["--headless"]).unwrap().front_end,
            FrontEnd::Headless
        );
        let stream = FrontEnd::Stream("0.0.0.0:8099".into());
        assert_eq!(
            parse(&["--stream", "0.0.0.0:8099"]).unwrap().front_end,
            stream
        );
        // --stream is headless already.
        assert_eq!(
            parse(&["--headless", "--stream", "0.0.0.0:8099"])
                .unwrap()
                .front_end,
            stream
        );
        assert_eq!(parse(&["--headless"]).unwrap().size(), (640, 480));
        assert_eq!(
            parse(&["--stream", "a:1", "--size", "800x600"])
                .unwrap()
                .size(),
            (800, 600)
        );
        // The container entrypoint's form.
        let docker = parse(&[
            "--stream",
            "0.0.0.0:8099",
            "--server",
            "shard:7777",
            "--quality",
            "low",
        ])
        .expect("the entrypoint form parses");
        assert_eq!(docker.server.as_deref(), Some("shard:7777"));
        assert_eq!(docker.quality, QualityPreset::Low);
        // Only the window's own keyboard may end the client: the keys of the
        // other front ends come from whoever can reach them.
        assert!(parse(&[]).unwrap().front_end.user_may_quit());
        assert!(!parse(&["--headless"]).unwrap().front_end.user_may_quit());
        assert!(!stream.user_may_quit() && !docker.front_end.user_may_quit());
        let help = usage();
        assert!(help.contains("No key ends a") && help.contains("cannot end it"));
    }

    #[test]
    fn the_server_address_is_checked_for_shape_only() {
        for good in [
            "127.0.0.1",
            "127.0.0.1:7000",
            "shard.example.org",
            "[::1]:7777",
        ] {
            assert_eq!(
                parse(&["--server", good]).unwrap().server.as_deref(),
                Some(good)
            );
        }
        for bad in ["", "host:port", "a b", "[::1"] {
            let err = parse(&["--server", bad]).expect_err(bad);
            assert!(err.contains("--server"), "{err}");
        }
        assert!(parse(&["--server"]).is_err());
    }

    #[test]
    fn the_harness_flags_work_with_every_front_end() {
        for front in [
            &[][..],
            &["--headless"][..],
            &["--stream", "127.0.0.1:0"][..],
        ] {
            let mut args = front.to_vec();
            args.extend([
                "--harness",
                "127.0.0.1:8098",
                "--script",
                "test.lua",
                "--out-dir",
                "shots",
                "--script-timeout",
                "2.5",
            ]);
            let config = parse(&args).expect("parses");
            assert_eq!(config.harness.as_deref(), Some("127.0.0.1:8098"));
            assert_eq!(config.script, Some(PathBuf::from("test.lua")));
            assert_eq!(config.out_dir, PathBuf::from("shots"));
            assert_eq!(config.script_timeout, Some(Duration::from_millis(2500)));
            assert!(config.harnessed());
            let scripts = script_config(&config);
            assert_eq!(scripts.out_dir, PathBuf::from("shots"));
            assert_eq!(scripts.timeout, Duration::from_millis(2500));
        }
        assert!(parse(&["--harness", "127.0.0.1:1"]).unwrap().harnessed());
        assert!(parse(&["--script", "x.lua"]).unwrap().harnessed());
        assert_eq!(
            script_config(&Config::default()).timeout,
            ScriptConfig::default().timeout
        );
        for bad in [
            &["--script-timeout", "5"][..],
            &["--script", "x.lua", "--script-timeout", "0"],
            &["--script", "x.lua", "--script-timeout", "-1"],
            &["--script", "x.lua", "--script-timeout", "soon"],
            &["--harness"],
            &["--script"],
            &["--out-dir"],
        ] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_remaining_flags_are_validated() {
        let config = parse(&[
            "--size",
            "64X48",
            "--quality",
            "medium",
            "--no-vsync",
            "--frames",
            "600",
            "--fps",
            "60",
            "--glb",
            "ship.glb",
            "--glb-scale",
            "0.5",
        ])
        .expect("parses");
        assert_eq!(config.size(), (64, 48));
        assert_eq!(config.quality, QualityPreset::Medium);
        assert!(!config.vsync);
        assert_eq!((config.frames, config.fps), (Some(600), 60));
        assert_eq!(config.glb, Some(PathBuf::from("ship.glb")));
        assert_eq!(config.glb_scale, Some(0.5));
        assert_eq!(parse_size("1280x720"), Some((1280, 720)));
        for bad in [
            "", "640", "640x", "x480", "0x480", "640x0", "-1x4", "99999x4", "axb",
        ] {
            assert_eq!(parse_size(bad), None, "{bad:?}");
            assert!(parse(&["--size", bad]).is_err(), "{bad:?}");
        }
        for bad in [
            &["--quality", "ultra"][..],
            &["--frames", "many"],
            &["--fps", "0"],
            &["--fps", "100000"],
            &["--glb-scale", "1.0"],
            &["--glb", "a.glb", "--glb-scale", "0"],
            &["--glb", "a.glb", "--glb-scale", "nan"],
            &["--nope"],
        ] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
    }

    /// The scene flags of the offline client change nothing and do not stop
    /// the client (an old deployment's `CLIENT_ARGS=--galaxy` must survive an
    /// image update); the other offline flags are refused with a pointer.
    #[test]
    fn the_old_scene_flags_are_ignored_and_the_other_offline_flags_refused() {
        let deployed = ["--stream", "0.0.0.0:8099", "--server", "10.0.0.2:7870"];
        for scene in ["--demo", "--galaxy", "--materials", "--solar"] {
            assert_eq!(parse(&[scene]), parse(&[]), "{scene}");
            let mut with_flag = deployed.to_vec();
            with_flag.push(scene);
            assert_eq!(parse(&with_flag), parse(&deployed), "{scene}");
            assert!(parse(&with_flag).is_ok(), "{scene}");
        }
        for gone in ["--window", "--out"] {
            let err = parse(&[gone]).expect_err(gone);
            assert!(err.contains("is gone"), "{err}");
        }
    }

    #[test]
    fn help_is_help_and_lists_every_flag_and_the_keys() {
        assert_eq!(parse_args(["--help".to_owned()]), Ok(Command::Help));
        assert_eq!(
            parse_args(["--headless".to_owned(), "-h".to_owned()]),
            Ok(Command::Help)
        );
        let usage = usage();
        for flag in [
            "--server",
            "--headless",
            "--stream",
            "--size",
            "--quality",
            "--no-vsync",
            "--frames",
            "--fps",
            "--glb",
            "--glb-scale",
            "--harness",
            "--script",
            "--script-timeout",
            "--out-dir",
        ] {
            assert!(usage.contains(flag), "{flag} is not in --help");
        }
        assert!(usage.contains(TOKEN_ENV) && usage.contains(KEY_HELP));
        assert!(usage.contains("C toggles") && usage.contains("7777"));
        for gone in ["--demo", "--galaxy", "--window", "--out "] {
            assert!(!usage.contains(gone), "{gone} is still in --help");
        }
    }
}
