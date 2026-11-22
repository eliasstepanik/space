//! `space-client` — the native client binary (Phase 4).
//!
//! Two modes:
//!
//! * **offline demo** (`--demo` / `--galaxy`): render one frame headlessly with
//!   `space-render` and write it to a PPM file, then exit. No window system is
//!   required, so this runs in a headless CI sandbox and exercises the exact
//!   pixels the GPU tests assert on.
//! * **windowed** (default): open a winit window and present the demo scene
//!   through the swapchain path.
//!
//! * **streaming** (`--stream ADDR`, opt-in): render headlessly in a loop, JPEG
//!   each frame and serve it to browser viewers over `ws://` (see [`stream`]).
//!   Without the flag nothing below changes.
//!
//! * **multiplayer streaming** (`--stream ADDR --server ADDR`): the same, but the
//!   process is also a normal `space-net` player on the authoritative shard;
//!   viewer input becomes the player's input and the replicated state drives the
//!   camera (see [`space_client::link`] and [`stream`]).

mod stream;

use std::net::{SocketAddr, ToSocketAddrs};
use std::path::PathBuf;

use space_render::{
    demo_scene, CameraPose, ContextOptions, Quality, QualityPreset, RenderContext, RenderError,
    RenderOptions, Scene, SceneLight, Swapchain, WindowHandles,
};
use space_world::{Galaxy, LodSelector, NodeRef, WorldPos, MAX_SELECTED};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle};
use winit::window::{Window, WindowId};

/// Offline demo scenes the client can render without a window.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    /// The deterministic one-sphere demo scene (shared with the GPU tests).
    Demo,
    /// The deterministic galaxy through the LOD selector.
    Galaxy,
    /// A live winit window presenting the demo scene.
    Window,
}

struct Config {
    mode: Mode,
    width: u32,
    height: u32,
    out: Option<PathBuf>,
    quality: QualityPreset,
    /// `--stream ADDR`: serve rendered frames to viewers on this `ws://` address.
    stream: Option<String>,
    /// `--server ADDR`: with `--stream`, play on this authoritative shard.
    server: Option<SocketAddr>,
    /// `--frames N`: stop streaming after N frames (default: run until killed).
    frames: Option<u64>,
}

fn parse_args() -> Config {
    let mut config = Config {
        mode: Mode::Window,
        width: 640,
        height: 480,
        out: None,
        quality: QualityPreset::default(),
        stream: None,
        server: None,
        frames: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--demo" => {
                config.mode = Mode::Demo;
            }
            "--galaxy" => {
                config.mode = Mode::Galaxy;
            }
            "--size" => {
                if let Some(dim) = args.next() {
                    let mut it = dim.split(['x', 'x', 'X']);
                    if let (Some(w), Some(h)) = (it.next(), it.next()) {
                        if let (Ok(w), Ok(h)) = (w.parse(), h.parse()) {
                            config.width = w;
                            config.height = h;
                        }
                    }
                }
            }
            "--out" => {
                config.out = args.next().map(PathBuf::from);
            }
            "--quality" => {
                let value = args.next().unwrap_or_default();
                match value.parse() {
                    Ok(preset) => config.quality = preset,
                    Err(e) => {
                        eprintln!("space-client: {e}");
                        std::process::exit(2);
                    }
                }
            }
            "--stream" => match args.next() {
                Some(addr) => config.stream = Some(addr),
                None => {
                    eprintln!("space-client: --stream needs an address, e.g. 127.0.0.1:8099");
                    std::process::exit(2);
                }
            },
            "--server" => {
                let addr = args.next().and_then(|a| a.to_socket_addrs().ok()?.next());
                match addr {
                    Some(addr) => config.server = Some(addr),
                    None => {
                        eprintln!("space-client: --server needs a host:port, e.g. 127.0.0.1:7000");
                        std::process::exit(2);
                    }
                }
            }
            "--frames" => match args.next().and_then(|n| n.parse().ok()) {
                Some(n) => config.frames = Some(n),
                None => {
                    eprintln!("space-client: --frames needs a number");
                    std::process::exit(2);
                }
            },
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                eprintln!("space-client: unknown argument {other:?}");
                print_usage();
                std::process::exit(2);
            }
        }
    }
    // `--server` joins a shard from the stream path or from the windowed path;
    // the headless one-frame modes (`--demo` / `--galaxy`) never connect.
    if config.server.is_some() && config.stream.is_none() && config.mode != Mode::Window {
        eprintln!(
            "space-client: --server needs --stream or the windowed client (not --demo/--galaxy)"
        );
        std::process::exit(2);
    }
    config
}

fn print_usage() {
    eprintln!(
        "usage: space-client [--demo | --galaxy] [--size WxH] [--out FILE.ppm]\n\
               [--quality high|medium|low|ci] [--stream ADDR [--server ADDR] [--frames N]]\n\
        \n\
        \x20 --demo     render the deterministic demo scene headlessly and exit\n\
        \x20 --galaxy   render the deterministic galaxy (LOD selector) headlessly and exit\n\
        \x20 --size WxH frame size in pixels (default 640x480)\n\
        \x20 --out FILE write the frame to FILE (PPM); default demo.ppm / galaxy.ppm\n\
        \x20 --quality P quality preset: high (default), medium, low or ci\n\
        \x20              (fog, screen-space reflections, refraction)\n\
        \x20 --stream ADDR  serve frames as JPEG over ws://ADDR (plain ws://, no TLS)\n\
        \x20              to the space-viewer wasm page; viewer input drives the camera\n\
        \x20              (scene: demo, or --galaxy)\n\
        \x20 --server ADDR  join the authoritative shard at ADDR as a normal player,\n\
        \x20              with --stream (viewer input drives the ship) or in the\n\
        \x20              window (WASD/Space/Shift thrust); replication drives the\n\
        \x20              camera; default: offline, no server\n\
        \x20 --frames N   with --stream, exit after N frames\n\
        \n\
        With neither --demo nor --galaxy a winit window is opened."
    );
}

/// The deterministic galaxy scene, built through `space-world`'s LOD selector.
fn galaxy_scene(width: u32, height: u32) -> Scene {
    galaxy_scene_at(&galaxy_camera(width, height))
}

/// Where the player spawns in the galaxy scene (the base camera's position).
const GALAXY_SPAWN_M: [f64; 3] = [0.0, 0.0, -5.0e4];
/// The spawn view direction (the stream starts turned to face the galaxy, +Z).
const GALAXY_SPAWN_FORWARD: [f32; 3] = [0.0, 0.0, 1.0];
/// The spawn cube sits this far in front of the spawn point, metres.
const SPAWN_CUBE_DISTANCE_M: f64 = 12.0;
/// Distance from the spawn cube's centre to each face, metres.
const SPAWN_CUBE_HALF_M: f32 = 2.0;
/// The spawn cube's colour (linear rgb): magenta, absent from the star field.
const SPAWN_CUBE_COLOR: [f32; 3] = [1.0, 0.05, 0.8];

/// The galaxy scene's base camera.
fn galaxy_camera(width: u32, height: u32) -> CameraPose {
    // Same camera as the `lod_galaxy_renders` GPU test: pulled back along -Z and
    // looking back (+Z) at the galaxy's centre at the origin, so the LOD selector
    // refines a dense band of the octree around the camera.
    CameraPose {
        position: WorldPos::from_f64_metres_around(WorldPos::ZERO, GALAXY_SPAWN_M),
        forward: GALAXY_SPAWN_FORWARD,
        up: [0.0, 1.0, 0.0],
        fov_y: std::f32::consts::FRAC_PI_3,
        aspect: width as f32 / height as f32,
        near: 1.0,
        far: 5.0e6,
    }
}

/// The galaxy through the LOD selector, seen from `camera`.
fn galaxy_scene_at(camera: &CameraPose) -> Scene {
    let galaxy = Galaxy::new(0x5eed);
    let lod = LodSelector::new(30, 63, 8.0);
    // `Scene::build` traverses only the one depth-30 node containing the camera.
    // The galaxy is centred on the cell origin and the camera sits on its x=0/y=0
    // boundary, so that node covers a single octant and the galaxy would fill one
    // frame quadrant. Traverse the 8 nodes of that depth meeting at the origin.
    let mut selected = Vec::new();
    for octant in 0..8i64 {
        let index = [-(octant & 1), -((octant >> 1) & 1), -((octant >> 2) & 1)];
        let root = NodeRef::new(lod.min_depth, index);
        let candidates = lod.candidates_from(root, camera.position, camera.forward, camera.fov_y);
        selected.extend(lod.select(&candidates, camera.position, camera.fov_y));
    }
    // Keep only nodes that hold a star, nearest to the camera first, so the
    // budget goes to the stars around the viewer rather than to one side.
    selected.retain(|&n| galaxy.star_at(n).is_some());
    selected.sort_by(|&a, &b| {
        let (da, db) = (
            lod.distance_to(a, camera.position),
            lod.distance_to(b, camera.position),
        );
        da.total_cmp(&db)
    });
    // Leave room for the spawn cube.
    selected.truncate(MAX_SELECTED - 1);
    let mut scene = Scene::from_selection(&galaxy, camera, &selected);
    // A lit cube a short distance in front of the spawn point, in world space so
    // it stays put when the camera moves.
    let spawn = WorldPos::from_f64_metres_around(WorldPos::ZERO, GALAXY_SPAWN_M);
    let cube = WorldPos::from_f64_metres_around(
        spawn,
        GALAXY_SPAWN_FORWARD.map(|c| f64::from(c) * SPAWN_CUBE_DISTANCE_M),
    );
    let d = WorldPos::delta(camera.position, cube);
    scene.push_box(
        [d[0] as f32, d[1] as f32, d[2] as f32],
        SPAWN_CUBE_HALF_M,
        SPAWN_CUBE_COLOR,
    );
    let light = WorldPos::from_f64_metres_around(cube, [4.0, 5.0, -6.0]);
    scene.with_lights(vec![SceneLight {
        position: light,
        color: [1.0, 1.0, 1.0],
        intensity: 60.0,
    }])
}

fn scene_for(mode: Mode, width: u32, height: u32) -> Scene {
    match mode {
        Mode::Galaxy => galaxy_scene(width, height),
        Mode::Demo | Mode::Window => demo_scene(width, height),
    }
}

/// Write an RGBA8 buffer as a binary PPM (P6) file.
fn write_ppm(path: &std::path::Path, width: u32, height: u32, rgba: &[u8]) -> std::io::Result<()> {
    assert_eq!(rgba.len(), (width * height * 4) as usize);
    let mut out = Vec::with_capacity(16 + (width * height * 3) as usize);
    out.extend_from_slice(format!("P6\n{width} {height}\n255\n").as_bytes());
    for px in rgba.as_chunks::<4>().0 {
        out.extend_from_slice(&px[..3]);
    }
    std::fs::write(path, out)
}

/// The CI-cheap, deterministic render options used by both the offline demo and
/// the windowed/swapchain path (`App` renders with these through
/// `render_frame_gpu`, so the same preset gates the same features, including the
/// SSR/refraction post pass, before `Swapchain::present` copies the tone-mapped
/// image); the preset selects the optional render features.
fn demo_options(preset: QualityPreset) -> RenderOptions {
    RenderOptions {
        gi_enabled: false,
        quality: Quality::CI,
        preset,
        ..Default::default()
    }
}

fn run_offscreen(config: &Config) -> Result<(), RenderError> {
    let ctx = RenderContext::new(&ContextOptions::default())?;
    eprintln!(
        "space-client: device {:?} ray_query={}",
        ctx.device_name(),
        ctx.ray_query_support().complete()
    );
    let scene = scene_for(config.mode, config.width, config.height);
    let out = ctx.render_offscreen(
        &scene,
        config.width,
        config.height,
        &demo_options(config.quality),
    )?;
    let default = match config.mode {
        Mode::Galaxy => PathBuf::from("galaxy.ppm"),
        _ => PathBuf::from("demo.ppm"),
    };
    let path = config.out.clone().unwrap_or(default);
    write_ppm(&path, out.width, out.height, &out.pixels)
        .map_err(|e| RenderError::new(format!("write {}: {e}", path.display())))?;
    eprintln!(
        "space-client: wrote {} ({}x{}, {} instances, {} validation messages)",
        path.display(),
        out.width,
        out.height,
        out.instance_count,
        out.validation_messages
    );
    Ok(())
}

/// How long the windowed client waits for the shard's handshake before giving
/// up (startup only, before any window exists).
const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// The windowed client's server session. The tokio runtime owns the link's
/// network tasks on its own worker threads, so the winit thread only ever
/// reads the shared replica (a short mutex lock) and never waits on the network.
struct Live {
    link: space_client::link::ServerLink,
    input: stream::InputState,
    /// Keeps the link's tasks running.
    _rt: tokio::runtime::Runtime,
}

impl Live {
    /// Connect to the shard (bounded by [`CONNECT_TIMEOUT`]).
    fn connect(server: SocketAddr) -> Result<Self, RenderError> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|e| RenderError::new(format!("tokio runtime: {e}")))?;
        let link = rt
            .block_on(async {
                tokio::time::timeout(
                    CONNECT_TIMEOUT,
                    space_client::link::ServerLink::connect(server),
                )
                .await
            })
            .map_err(|_| RenderError::new(format!("server {server}: connect timed out")))?
            .map_err(|e| RenderError::new(format!("server {server}: {e}")))?;
        eprintln!(
            "space-client: connected to {server} as session {}",
            link.replica().session
        );
        Ok(Self {
            link,
            input: stream::InputState::default(),
            _rt: rt,
        })
    }

    /// Track a movement key and forward the resulting thrust to the shard.
    fn key(&mut self, code: winit::keyboard::KeyCode, down: bool) {
        use space_stream::{Key, ViewerInput};
        use winit::keyboard::KeyCode;
        let key = match code {
            KeyCode::KeyW => Key::Forward,
            KeyCode::KeyS => Key::Back,
            KeyCode::KeyA => Key::Left,
            KeyCode::KeyD => Key::Right,
            KeyCode::Space => Key::Up,
            KeyCode::ShiftLeft => Key::Down,
            _ => return,
        };
        self.input.apply(ViewerInput::Key { key, down });
        self.link.set_thrust(self.input.thrust());
    }

    /// The scene for this frame, built from the current replicated state
    /// (non-blocking: one short lock).
    fn scene(&self, width: u32, height: u32) -> Scene {
        stream::live_scene(&self.link.replica(), &self.input, width, height)
    }
}

/// The windowed app: owns the render context, window and swapchain and drives
/// them from winit's `ApplicationHandler` callbacks.
struct App {
    ctx: RenderContext,
    config: Config,
    window: Option<Window>,
    swapchain: Option<Swapchain>,
    scene: Option<Scene>,
    options: RenderOptions,
    failed: bool,
    /// Frames presented so far (logged so a hang is visible).
    frames: u64,
    /// The server session, when `--server` was given; `None` = offline demo.
    live: Option<Live>,
}

impl App {
    fn new(
        config: &Config,
        display: RawDisplayHandle,
        live: Option<Live>,
    ) -> Result<Self, RenderError> {
        let options = ContextOptions {
            window_display: Some(display),
            ..ContextOptions::default()
        };
        Ok(Self {
            ctx: RenderContext::new(&options)?,
            config: Config {
                mode: Mode::Window,
                width: config.width,
                height: config.height,
                out: None,
                quality: config.quality,
                stream: None,
                server: None,
                frames: None,
            },
            window: None,
            swapchain: None,
            scene: None,
            options: demo_options(config.quality),
            failed: false,
            frames: 0,
            live,
        })
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("space")
            .with_inner_size(winit::dpi::PhysicalSize::new(
                self.config.width,
                self.config.height,
            ));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("space-client: window: {e}");
                self.failed = true;
                event_loop.exit();
                return;
            }
        };
        eprintln!("space-client: device {:?}", self.ctx.device_name());
        let handles = match window_handles(&window) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("space-client: {e}");
                self.failed = true;
                event_loop.exit();
                return;
            }
        };
        let size = window.inner_size();
        let swapchain =
            match Swapchain::new(&self.ctx, handles, size.width.max(1), size.height.max(1)) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("space-client: swapchain: {e}");
                    self.failed = true;
                    event_loop.exit();
                    return;
                }
            };
        let (sw, sh) = swapchain.extent();
        // Joined: the scene is rebuilt from replica state every frame; offline
        // it stays the static demo scene.
        self.scene = Some(match &self.live {
            Some(live) => live.scene(sw, sh),
            None => scene_for(Mode::Window, sw, sh),
        });
        self.swapchain = Some(swapchain);
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if let (Some(live), winit::keyboard::PhysicalKey::Code(code)) =
                    (self.live.as_mut(), event.physical_key)
                {
                    live.key(code, event.state.is_pressed());
                }
                if let winit::keyboard::Key::Character(c) = &event.logical_key {
                    if c.as_str() == "q" {
                        event_loop.exit();
                    }
                }
                if event.logical_key
                    == winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape)
                {
                    event_loop.exit();
                }
            }
            WindowEvent::RedrawRequested => {
                let (Some(window), Some(swapchain), Some(scene)) = (
                    self.window.as_ref(),
                    self.swapchain.as_ref(),
                    self.scene.as_ref(),
                ) else {
                    return;
                };
                let (sw, sh) = swapchain.extent();
                match self.ctx.render_frame_gpu(scene, sw, sh, &self.options) {
                    Ok((_, frame)) => {
                        if let Err(e) = swapchain.present(&self.ctx, &frame) {
                            eprintln!("space-client: present failed: {e}");
                            self.failed = true;
                            event_loop.exit();
                        } else {
                            self.frames += 1;
                            if self.frames <= 3 || self.frames.is_multiple_of(300) {
                                eprintln!("space-client: presented frame {}", self.frames);
                            }
                            let _ = window;
                        }
                    }
                    Err(e) => {
                        eprintln!("space-client: render failed: {e}");
                        self.failed = true;
                        event_loop.exit();
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

/// Borrow the raw window/display handles from a live winit window.
fn window_handles(window: &Window) -> Result<WindowHandles, RenderError> {
    let handle = window
        .window_handle()
        .map_err(|e| RenderError::new(format!("window_handle: {e}")))?;
    let display = window
        .display_handle()
        .map_err(|e| RenderError::new(format!("display_handle: {e}")))?;
    Ok(WindowHandles {
        display: display.as_raw(),
        window: handle.as_raw(),
    })
}

fn run_window(config: &Config) -> Result<(), RenderError> {
    let event_loop = EventLoop::new().map_err(|e| RenderError::new(format!("event loop: {e}")))?;
    event_loop.set_control_flow(ControlFlow::Wait);
    // The instance must be created with the surface extensions this display
    // needs, so take the display handle from the event loop before building
    // the render context.
    let display = event_loop
        .display_handle()
        .map_err(|e| RenderError::new(format!("display_handle: {e}")))?
        .as_raw();
    let mut app = App::new(config, display)?;
    event_loop
        .run_app(&mut app)
        .map_err(|e| RenderError::new(format!("event loop: {e}")))?;
    if app.failed {
        std::process::exit(1);
    }
    Ok(())
}

fn main() {
    env_logger::init();
    let config = parse_args();
    let result = match (&config.stream, config.mode) {
        (Some(addr), _) => stream::run_stream(&config, addr),
        (None, Mode::Demo | Mode::Galaxy) => run_offscreen(&config),
        (None, Mode::Window) => run_window(&config),
    };
    if let Err(e) = result {
        eprintln!("space-client: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The offline demo renders a lit sphere headlessly (no window), proving
    /// the client's render path works end to end on a real device. Skips
    /// loudly when no Vulkan device is available.
    #[test]
    fn demo_renders_lit_pixels() {
        let ctx = match RenderContext::new(&ContextOptions::default()) {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!(
                    "SKIP: no usable Vulkan device ({e}); needs VK_KHR_acceleration_structure"
                );
                return;
            }
        };
        // Same guard as the space-render GPU tests: with the var set the layer
        // must really be active, or `validation_messages == 0` below is vacuous.
        assert!(
            std::env::var("SPACE_RENDER_VALIDATION").as_deref() != Ok("1")
                || ctx.validation_active(),
            "SPACE_RENDER_VALIDATION=1 but VK_LAYER_KHRONOS_validation is not active"
        );
        let size = 64;
        let scene = demo_scene(size, size);
        let out = ctx
            .render_offscreen(&scene, size, size, &demo_options(QualityPreset::Ci))
            .expect("demo render");
        assert_eq!(out.validation_messages, 0);
        let lit = out
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|px| px[0] > 8 || px[1] > 8 || px[2] > 8)
            .count();
        assert!(lit > 0, "the demo scene rendered nothing");
    }

    /// The windowed path renders through `render_frame_gpu` with the same
    /// options as the offline demo: every preset (including the SSR/refraction
    /// presets) must render cleanly through it, carry the preset into the frame
    /// with zero validation messages, produce the same pixels as the offscreen
    /// path, and actually *differ* per preset: High/Medium show the SSR
    /// reflection and fog that Low/Ci (identical to each other) do not, and
    /// High's denser fog differs from Medium's.
    #[test]
    fn every_preset_renders_through_the_swapchain_frame_path() {
        let Ok(ctx) = RenderContext::new(&ContextOptions::default()) else {
            eprintln!("SKIP: no Vulkan device");
            return;
        };
        assert!(
            std::env::var("SPACE_RENDER_VALIDATION").as_deref() != Ok("1")
                || ctx.validation_active(),
            "SPACE_RENDER_VALIDATION=1 but VK_LAYER_KHRONOS_validation is not active"
        );
        let size = 64;
        let camera = CameraPose {
            position: WorldPos::ZERO,
            forward: [0.0, 0.0, -1.0],
            up: [0.0, 1.0, 0.0],
            fov_y: std::f32::consts::FRAC_PI_3,
            aspect: 1.0,
            near: 0.05,
            far: 1.0e5,
        };
        let instance = |center: [f32; 4], albedo: [f32; 4], emissive: [f32; 4], shape| {
            space_render::GpuInstance {
                center,
                albedo,
                emissive,
                roughness: 0.1,
                shape,
                transmission: 0.0,
                _pad1: 0.0,
            }
        };
        // SSR scene: a smooth metallic floor under a bright red sphere.
        let mirror = Scene::from_instances(
            camera,
            vec![
                instance(
                    [0.0, -101.0, 0.0, 100.0],
                    [0.8, 0.8, 0.8, 1.0],
                    [0.0; 4],
                    space_render::SHAPE_BOX,
                ),
                instance(
                    [0.0, -0.4, -6.0, 0.5],
                    [0.0; 4],
                    [6.0, 0.0, 0.0, 1.0],
                    space_render::SHAPE_SPHERE,
                ),
            ],
        );
        // Fog scene: a big bright sphere 800 m away.
        let far = Scene::from_instances(
            camera,
            vec![instance(
                [0.0, 0.0, -800.0, 100.0],
                [1.0, 1.0, 1.0, 0.0],
                [4.0, 4.0, 4.0, 1.0],
                space_render::SHAPE_SPHERE,
            )],
        );

        let render = |scene: &Scene, preset: QualityPreset| {
            let options = demo_options(preset);
            assert_eq!(options.preset, preset);
            let (out, frame) = ctx
                .render_frame_gpu(scene, size, size, &options)
                .expect("frame render");
            drop(frame);
            assert_eq!(out.validation_messages, 0, "{preset}");
            assert!(
                out.pixels.as_chunks::<4>().0.iter().any(|px| px[0] > 8),
                "{preset}: nothing rendered"
            );
            // The swapchain-path frame is the offscreen frame.
            let offscreen = ctx
                .render_offscreen(scene, size, size, &options)
                .expect("offscreen render");
            assert!(
                out.pixels == offscreen.pixels,
                "{preset}: frame path != offscreen"
            );
            out.pixels
        };
        for scene in [&mirror, &far] {
            let [high, medium, low, ci] = QualityPreset::ALL.map(|p| render(scene, p));
            assert!(low == ci, "Low and Ci have every optional feature off");
            assert!(high != ci, "High must differ from Ci");
            assert!(medium != ci, "Medium must differ from Ci");
            assert!(high != medium, "High and Medium must differ");
        }
    }

    /// The galaxy scene builds through the LOD selector and stays within the
    /// `MAX_SELECTED` bound.
    #[test]
    fn galaxy_scene_is_bounded() {
        let scene = galaxy_scene(64, 64);
        assert!(scene.instances.len() <= space_world::MAX_SELECTED);
    }

    /// The spawn cube is the last instance: a box straight ahead (+Z) of the
    /// spawn camera at `SPAWN_CUBE_DISTANCE_M`.
    #[test]
    fn galaxy_scene_has_a_cube_in_front_of_the_spawn_camera() {
        let scene = galaxy_scene(64, 64);
        let cube = scene.instances.last().expect("cube instance");
        assert_eq!(cube.shape, space_render::SHAPE_BOX);
        assert_eq!(scene.instances.len(), scene.meshes.len());
        assert!(cube.center[0].abs() < 1e-3 && cube.center[1].abs() < 1e-3);
        assert!((f64::from(cube.center[2]) - SPAWN_CUBE_DISTANCE_M).abs() < 1e-3);
        assert_eq!(cube.center[3], SPAWN_CUBE_HALF_M);
    }

    /// `write_ppm` emits a valid P6 header and one RGB triple per pixel.
    #[test]
    fn write_ppm_writes_a_valid_p6_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("space_client_test.ppm");
        let rgba = [1u8, 2, 3, 255, 4, 5, 6, 255];
        write_ppm(&path, 2, 1, &rgba).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.starts_with(b"P6\n2 1\n255\n"));
        assert_eq!(&bytes[b"P6\n2 1\n255\n".len()..], &[1, 2, 3, 4, 5, 6]);
        let _ = std::fs::remove_file(&path);
    }
}
