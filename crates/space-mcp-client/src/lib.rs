//! `space-mcp-client`: an MCP server that drives a `space` client.
//!
//! It plays the part of a player: it connects to an authoritative shard through
//! the ordinary `space-net` client path ([`space_client::link::ServerLink`]),
//! sends thrust input, and can render the scene the shard shows
//! ([`space_client::scenes::World`]: the demo area unless the server says
//! otherwise) from the replicated ship's point of view offscreen (via
//! `space-render`, so it works on lavapipe) and hand the frame back as a JPEG.
//!
//! # Tools
//!
//! | tool            | what it does                                                  |
//! |-----------------|---------------------------------------------------------------|
//! | `connect`       | join the shard at `addr` (host:port, QUIC) as a player        |
//! | `disconnect`    | leave the shard                                               |
//! | `client_status` | session, tick, ship guid, replicated offset/velocity, scene   |
//! | `send_input`    | set thrust `[x,y,z]` (optionally for `duration_ms`, then 0)   |
//! | `capture_frame` | render one frame offscreen, return it as base64 JPEG / a file |

#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]

use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use serde_json::{json, Value};
use space_client::input::InputState;
use space_client::link::ServerLink;
use space_client::scenes::{SceneKind, World};
use space_mcp::tool::{arg_str, arg_u64_opt};
use space_mcp::{base64, McpServer, Tool, ToolRegistry};
use space_render::{ContextOptions, QualityPreset, RenderContext, RenderOptions, Scene};
use space_stream::JpegEncoder;
use space_world::WorldPos;
use tokio::sync::Mutex;

/// Longest `send_input` hold.
pub const MAX_HOLD_MS: u64 = 10_000;
/// How long `connect` waits for the shard to answer.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Smallest and largest frame edge `capture_frame` renders.
pub const FRAME_EDGE: std::ops::RangeInclusive<u64> = 16..=1920;
/// JPEG quality of captured frames.
pub const JPEG_QUALITY: u8 = 85;

/// A lazily created render context, shared by captures.
pub type SharedRender = Arc<StdMutex<Option<RenderContext>>>;

/// Everything the tools share.
pub struct ClientState {
    link: Mutex<Option<ServerLink>>,
    render: SharedRender,
    out_dir: PathBuf,
}

impl std::fmt::Debug for ClientState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientState")
            .field("link", &self.link)
            .field("out_dir", &self.out_dir)
            .finish_non_exhaustive()
    }
}

impl ClientState {
    /// A fresh, disconnected client that saves named captures under `out_dir`.
    #[must_use]
    pub fn new(out_dir: PathBuf) -> Arc<Self> {
        Arc::new(Self {
            link: Mutex::new(None),
            render: Arc::new(StdMutex::new(None)),
            out_dir,
        })
    }
}

fn pretty(value: &Value) -> Result<String, String> {
    serde_json::to_string_pretty(value).map_err(|e| e.to_string())
}

/// `scene` as a ship sees it that has flown `offset` metres from where it
/// first appeared: a first-person camera that far from the scene's spawn
/// point, looking the way a newcomer looks (the mapping the client itself
/// uses), and the render options that scene is drawn with.
fn scene_at(scene: SceneKind, offset: [f64; 3], width: u32, height: u32) -> (Scene, RenderOptions) {
    let world = World::new(scene);
    let (yaw, pitch) = world.initial_view();
    let camera = world.camera(
        WorldPos::from_f64_metres_around(world.spawn(), offset),
        InputState::looking(yaw, pitch).forward(),
        width as f32 / height as f32,
    );
    (
        world.build(&camera),
        world.options(QualityPreset::default()),
    )
}

/// A captured, JPEG-encoded frame.
#[derive(Debug)]
pub struct Capture {
    /// JPEG bytes.
    pub jpeg: Vec<u8>,
    /// Pixel width.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
    /// The Vulkan device that rendered it.
    pub device: String,
}

/// Render `scene` (camera displaced by `offset` metres from its spawn point)
/// and JPEG-encode it. Blocking (GPU work).
///
/// # Errors
///
/// `Err` when no Vulkan device is usable or rendering/encoding fails.
pub fn render_jpeg(
    render: &StdMutex<Option<RenderContext>>,
    scene: SceneKind,
    offset: [f64; 3],
    width: u32,
    height: u32,
) -> Result<Capture, String> {
    let mut slot = render.lock().unwrap_or_else(|e| e.into_inner());
    if slot.is_none() {
        *slot = Some(
            RenderContext::new(&ContextOptions::default())
                .map_err(|e| format!("no usable Vulkan device: {}", e.message))?,
        );
    }
    let ctx = slot.as_ref().ok_or("render context missing")?;
    let (scene, options) = scene_at(scene, offset, width, height);
    let frame = ctx
        .render_offscreen(&scene, width, height, &options)
        .map_err(|e| format!("render failed: {}", e.message))?;
    let jpeg = JpegEncoder::new(JPEG_QUALITY, 1)
        .encode(&frame.pixels, frame.width, frame.height)
        .map_err(|e| format!("jpeg encode failed: {e}"))?;
    Ok(Capture {
        jpeg: jpeg.bytes,
        width: jpeg.width,
        height: jpeg.height,
        device: ctx.device_name().to_owned(),
    })
}

/// Write a captured frame to `path`, creating its directory. Blocking.
///
/// # Errors
///
/// A message naming the file when the directory or the file cannot be written.
pub fn save_capture(path: &std::path::Path, jpeg: &[u8]) -> Result<(), String> {
    let written = match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => std::fs::create_dir_all(dir),
        _ => Ok(()),
    }
    .and_then(|()| std::fs::write(path, jpeg));
    written.map_err(|e| format!("cannot save {}: {e}", path.display()))
}

/// A file name that is safe to create under the output directory.
fn valid_file_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-')
}

fn tool<F, Fut>(name: &str, description: &str, schema: Value, f: F) -> Tool
where
    F: Fn(Value) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<String, String>> + Send + 'static,
{
    Tool::new(name, description, schema, f)
}

/// The client's tools, backed by `state`.
#[must_use]
pub fn registry(state: &Arc<ClientState>) -> ToolRegistry {
    let empty = json!({"type": "object", "properties": {}});
    let mut r = ToolRegistry::new();

    let s = Arc::clone(state);
    r.register(tool(
        "connect",
        "Connect to a space shard over QUIC as an ordinary player (e.g. the `quic_endpoint` from the server MCP's shard_status). Errors if already connected.",
        json!({
            "type": "object",
            "properties": {"addr": {"type": "string", "description": "host:port, e.g. 127.0.0.1:7777"}},
            "required": ["addr"],
        }),
        move |args| {
            let s = Arc::clone(&s);
            async move { connect(&s, &args).await }
        },
    ));

    let s = Arc::clone(state);
    r.register(tool(
        "disconnect",
        "Leave the shard (the shard despawns this player's ship).",
        empty.clone(),
        move |_| {
            let s = Arc::clone(&s);
            async move { disconnect(&s).await }
        },
    ));

    let s = Arc::clone(state);
    r.register(tool(
        "client_status",
        "Connection state and the client's replicated view: session id, last tick, frames received, own ship guid, offset from where it first appeared (m), velocity and the scene the shard shows.",
        empty,
        move |_| {
            let s = Arc::clone(&s);
            async move { client_status(&s).await }
        },
    ));

    let s = Arc::clone(state);
    r.register(tool(
        "send_input",
        "Set the thrust intent sent every tick: `thrust` [x,y,z], each in -1..1 (world frame; -Z is forward). With `duration_ms` (<=10000) the thrust is held that long, then reset to zero and the call returns; otherwise it stays until changed.",
        json!({
            "type": "object",
            "properties": {
                "thrust": {"type": "array", "items": {"type": "number", "minimum": -1, "maximum": 1}, "minItems": 3, "maxItems": 3},
                "duration_ms": {"type": "integer", "minimum": 1, "maximum": MAX_HOLD_MS},
            },
            "required": ["thrust"],
        }),
        move |args| {
            let s = Arc::clone(&s);
            async move { send_input(&s, &args).await }
        },
    ));

    let s = Arc::clone(state);
    r.register(tool(
        "capture_frame",
        "Render one frame offscreen (software Vulkan is fine) and return it as a JPEG: JSON with width/height/bytes/device/scene and `base64` (unless inline=false). If connected the picture is the scene the shard shows, seen from the replicated ship (first person); otherwise the default scene from its spawn point. `name` (plain file name) also saves it to the server's output directory and returns the path.",
        json!({
            "type": "object",
            "properties": {
                "width": {"type": "integer", "minimum": 16, "maximum": 1920, "default": 320},
                "height": {"type": "integer", "minimum": 16, "maximum": 1920, "default": 240},
                "inline": {"type": "boolean", "default": true, "description": "include the frame as base64 in the reply"},
                "name": {"type": "string", "description": "file name to save under the output directory, e.g. frame.jpg"},
            },
        }),
        move |args| {
            let s = Arc::clone(&s);
            async move { capture_frame(&s, &args).await }
        },
    ));
    r
}

async fn connect(state: &ClientState, args: &Value) -> Result<String, String> {
    const ALREADY: &str = "already connected; call disconnect first";
    let addr = arg_str(args, "addr")?;
    let connected = |slot: &Option<ServerLink>| slot.as_ref().is_some_and(|l| !l.is_closed());
    if connected(&*state.link.lock().await) {
        return Err(ALREADY.into());
    }
    // The link mutex is **not** held while resolving and dialling: that takes
    // up to ten seconds against a shard that does not answer, and every other
    // tool (`client_status`, `send_input`, `capture_frame`, `disconnect`) needs
    // the same mutex — over HTTP they all hung behind one slow `connect`.
    let target = tokio::net::lookup_host(addr)
        .await
        .map_err(|e| format!("cannot resolve {addr:?}: {e}"))?
        .next()
        .ok_or_else(|| format!("{addr:?} resolved to no address"))?;
    let link = tokio::time::timeout(CONNECT_TIMEOUT, ServerLink::connect(target))
        .await
        .map_err(|_| format!("connecting to {target} timed out"))?
        .map_err(|e| format!("connect to {target} failed: {e}"))?;
    let mut slot = state.link.lock().await;
    if connected(&slot) {
        // Another `connect` got there first while this one was dialling; one
        // player, one link. (Dropping `link` closes it.)
        return Err(ALREADY.into());
    }
    let session = link.replica().session;
    *slot = Some(link);
    pretty(&json!({"connected": true, "addr": target.to_string(), "session": session}))
}

async fn disconnect(state: &ClientState) -> Result<String, String> {
    match state.link.lock().await.take() {
        Some(link) => {
            link.close();
            Ok("disconnected".into())
        }
        None => Err("not connected".into()),
    }
}

async fn client_status(state: &ClientState) -> Result<String, String> {
    let slot = state.link.lock().await;
    let Some(link) = slot.as_ref() else {
        return pretty(&json!({"connected": false}));
    };
    let replica = link.replica();
    pretty(&json!({
        "connected": !link.is_closed(),
        "session": replica.session,
        "tick": replica.tick,
        "frames": replica.frames,
        "ship": replica.ship.map(|g| format!("{}:{}", g.shard.0, g.serial)),
        "offset_m": link.offset(),
        "velocity_mps": replica.state.map(|s| s.vel),
        "scene": replica.scene.scene.or_default().name(),
    }))
}

async fn send_input(state: &ClientState, args: &Value) -> Result<String, String> {
    let raw = args
        .get("thrust")
        .and_then(Value::as_array)
        .filter(|a| a.len() == 3)
        .ok_or("`thrust` must be an array of 3 numbers")?;
    let mut thrust = [0.0f32; 3];
    for (slot, n) in thrust.iter_mut().zip(raw) {
        let n = n
            .as_f64()
            .filter(|n| n.is_finite() && n.abs() <= 1.0)
            .ok_or("`thrust` components must be finite numbers in -1..1")?;
        *slot = n as f32;
    }
    let hold = arg_u64_opt(args, "duration_ms")?;
    if hold.is_some_and(|ms| ms == 0 || ms > MAX_HOLD_MS) {
        return Err(format!("`duration_ms` must be in 1..={MAX_HOLD_MS}"));
    }
    {
        let slot = state.link.lock().await;
        let link = slot.as_ref().ok_or("not connected")?;
        link.set_thrust(thrust);
    }
    let Some(ms) = hold else {
        return pretty(&json!({"thrust": thrust, "held": "until changed"}));
    };
    tokio::time::sleep(Duration::from_millis(ms)).await;
    if let Some(link) = state.link.lock().await.as_ref() {
        link.set_thrust([0.0; 3]);
    }
    pretty(&json!({"thrust": thrust, "held_ms": ms, "now": [0.0, 0.0, 0.0]}))
}

async fn capture_frame(state: &ClientState, args: &Value) -> Result<String, String> {
    let edge = |key: &str, default: u64| -> Result<u32, String> {
        let v = arg_u64_opt(args, key)?.unwrap_or(default);
        if !FRAME_EDGE.contains(&v) {
            return Err(format!(
                "`{key}` must be in {}..={}",
                FRAME_EDGE.start(),
                FRAME_EDGE.end()
            ));
        }
        u32::try_from(v).map_err(|e| e.to_string())
    };
    let (width, height) = (edge("width", 320)?, edge("height", 240)?);
    let inline = args.get("inline").and_then(Value::as_bool).unwrap_or(true);
    let name = match args.get("name") {
        None | Some(Value::Null) => None,
        Some(Value::String(n)) if valid_file_name(n) => Some(n.clone()),
        Some(_) => {
            return Err(
                "`name` must be a plain file name ([A-Za-z0-9._-], max 64, no leading dot)".into(),
            )
        }
    };
    let (scene, offset, connected) = match state.link.lock().await.as_ref() {
        Some(link) => (link.scene().scene.or_default(), link.offset(), true),
        None => (SceneKind::DEFAULT, [0.0; 3], false),
    };
    let render = Arc::clone(&state.render);
    // Rendering, encoding **and** writing the file all block, so they all run
    // on the blocking pool; the file write used to happen on an async worker.
    let save_to = name.as_ref().map(|name| state.out_dir.join(name));
    let saving = save_to.clone();
    let capture = tokio::task::spawn_blocking(move || {
        let capture = render_jpeg(&render, scene, offset, width, height)?;
        if let Some(path) = saving {
            save_capture(&path, &capture.jpeg)?;
        }
        Ok::<Capture, String>(capture)
    })
    .await
    .map_err(|e| format!("render task failed: {e}"))??;
    let mut reply = json!({
        "format": "jpeg",
        "width": capture.width,
        "height": capture.height,
        "bytes": capture.jpeg.len(),
        "device": capture.device,
        "connected": connected,
        "scene": scene.name(),
        "camera_offset_m": offset,
    });
    if let Some(path) = save_to {
        reply["path"] = json!(path.display().to_string());
    }
    if inline {
        reply["base64"] = json!(base64::encode(&capture.jpeg));
    }
    pretty(&reply)
}

/// The complete MCP server for `state`.
#[must_use]
pub fn mcp_server(state: &Arc<ClientState>) -> McpServer {
    McpServer::new(
        "space-mcp-client",
        env!("CARGO_PKG_VERSION"),
        registry(state),
    )
}
