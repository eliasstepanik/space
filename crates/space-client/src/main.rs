//! `space-client` — the native client binary.
//!
//! What it does with a frame is one of three things:
//!
//! * **window** (the default; also `--window`, or `--server` without
//!   `--stream`): open a winit window and present through the swapchain, with
//!   keyboard/mouse input (see [`window`]). `--galaxy` picks the galaxy scene
//!   instead of the demo scene; `--server ADDR` joins the authoritative shard
//!   as a player.
//! * **offline render** (`--demo` / `--galaxy` on their own): render one frame
//!   headlessly with `space-render`, write it to a PPM file and exit. No window
//!   system is required, so this runs in a headless CI sandbox and exercises
//!   the exact pixels the GPU tests assert on.
//! * **streaming** (`--stream ADDR`): render headlessly in a loop, JPEG each
//!   frame and serve it to browser viewers over `ws://` (see [`stream`]). With
//!   `--server ADDR` the process is also a normal `space-net` player on the
//!   shard; viewer input becomes the player's input and the replicated state
//!   drives the camera (see [`space_client::link`]).

mod scenes;
mod stream;
mod window;

use std::net::{SocketAddr, ToSocketAddrs};
use std::path::PathBuf;

use std::sync::Arc;

use space_render::{
    demo_scene, rotation_y, CameraPose, ContextOptions, MeshData, ModelInstance, Quality,
    QualityPreset, RenderContext, RenderError, RenderOptions, Scene, SceneLight,
    MAX_FRAME_DIMENSION,
};
use space_world::{Galaxy, LodSelector, NodeRef, WorldPos, MAX_SELECTED};

/// Which scene is rendered, and — for the two explicit scenes — that a bare
/// invocation renders it once, offline.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    /// The deterministic demo scene (shared with the GPU tests).
    Demo,
    /// The deterministic galaxy through the LOD selector.
    Galaxy,
    /// The PBR material chart ([`scenes::materials_scene`]).
    Materials,
    /// The real-size solar system ([`scenes::solar_scene`]).
    Solar,
    /// No scene was named: a live winit window presenting the demo scene.
    Window,
}

impl Mode {
    /// Short name for logs and the title bar.
    const fn name(self) -> &'static str {
        match self {
            Self::Demo | Self::Window => "demo",
            Self::Galaxy => "galaxy",
            Self::Materials => "materials",
            Self::Solar => "solar",
        }
    }
}

struct Config {
    mode: Mode,
    width: u32,
    height: u32,
    out: Option<PathBuf>,
    quality: QualityPreset,
    /// `--stream ADDR`: serve rendered frames to viewers on this `ws://` address.
    stream: Option<String>,
    /// `--server ADDR`: play on this authoritative shard (window or stream).
    server: Option<SocketAddr>,
    /// `--frames N`: stop after N frames (default: run until closed/killed).
    frames: Option<u64>,
    /// `--window`: present in a window even though a scene was named.
    window: bool,
    /// Windowed presentation waits for the vertical blank (`--no-vsync` clears it).
    vsync: bool,
    /// `--glb FILE`: a model to show in front of the spawn point.
    glb: Option<PathBuf>,
    /// `--glb-scale S`: metres per model unit (default: fit to ~3 m across).
    glb_scale: Option<f32>,
    /// The loaded `--glb` model, placed in the world (set by `main`).
    model: Option<PlacedModel>,
}

/// A mesh model at a fixed place in the world.
#[derive(Clone)]
struct PlacedModel {
    mesh: Arc<MeshData>,
    /// Where the centre of the model's bounding box sits.
    position: WorldPos,
    /// Metres per model unit.
    scale: f32,
    rotation: [[f32; 3]; 3],
}

/// The radius a `--glb` model is scaled to when no `--glb-scale` is given, and
/// how many radii in front of the spawn camera it is placed.
const MODEL_FIT_RADIUS_M: f32 = 1.5;
const MODEL_DISTANCE_RADII: f32 = 4.0;

impl PlacedModel {
    /// Load `path` and place it in front of `mode`'s spawn camera, turned a
    /// little so it is not seen dead-on.
    fn load(path: &std::path::Path, scale: Option<f32>, mode: Mode) -> Result<Self, RenderError> {
        let mesh = MeshData::load(path).map_err(|e| RenderError::new(format!("--glb {e}")))?;
        let radius = mesh.radius().max(1e-6);
        let scale = scale.unwrap_or(MODEL_FIT_RADIUS_M / radius);
        let camera = spawn_camera(mode, 1, 1);
        let distance = f64::from(MODEL_DISTANCE_RADII * radius * scale);
        let position = WorldPos::from_f64_metres_around(
            camera.position,
            camera.forward.map(|c| f64::from(c) * distance),
        );
        eprintln!(
            "space-client: model {}: {} triangles, {} parts, {:.2} m across at scale {scale}",
            path.display(),
            mesh.triangle_count(),
            mesh.parts.len(),
            2.0 * radius * scale
        );
        Ok(Self {
            mesh: Arc::new(mesh),
            position,
            scale,
            rotation: rotation_y(0.6),
        })
    }

    /// Add the model to `scene`, relative to the scene's camera.
    fn add_to(&self, scene: &mut Scene) {
        // The instance's position is where the model's *origin* goes: the
        // bounding-box centre, rotated and scaled, is subtracted so the centre
        // lands on `self.position`.
        let centre = self.mesh.center();
        let d = WorldPos::delta(scene.camera.position, self.position);
        let position = [0, 1, 2].map(|r| {
            let rotated: f32 = (0..3).map(|c| self.rotation[r][c] * centre[c]).sum();
            d[r] as f32 - rotated * self.scale
        });
        scene.models.push(ModelInstance {
            mesh: Arc::clone(&self.mesh),
            position,
            rotation: self.rotation,
            scale: self.scale,
        });
    }
}

/// The camera a scene starts with (what an offline render uses).
fn spawn_camera(mode: Mode, width: u32, height: u32) -> CameraPose {
    match mode {
        Mode::Galaxy => galaxy_camera(width, height),
        Mode::Materials => scenes::materials_camera(width, height),
        Mode::Solar => scenes::solar_camera(width, height),
        Mode::Demo | Mode::Window => demo_scene(width, height).camera,
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mode: Mode::Window,
            width: 640,
            height: 480,
            out: None,
            quality: QualityPreset::default(),
            stream: None,
            server: None,
            frames: None,
            window: false,
            vsync: true,
            glb: None,
            glb_scale: None,
            model: None,
        }
    }
}

impl Config {
    /// Whether this invocation opens a window: asked for explicitly, implied
    /// by joining a server without streaming, or simply no scene named.
    fn windowed(&self) -> bool {
        self.stream.is_none() && (self.window || self.server.is_some() || self.mode == Mode::Window)
    }
}

/// Parse a `WxH` frame size; both sides must be in `1..=MAX_FRAME_DIMENSION`.
fn parse_size(text: &str) -> Option<(u32, u32)> {
    let (w, h) = text.split_once(['x', 'X'])?;
    let valid = 1..=MAX_FRAME_DIMENSION;
    let (w, h) = (w.parse().ok()?, h.parse().ok()?);
    (valid.contains(&w) && valid.contains(&h)).then_some((w, h))
}

fn parse_args() -> Config {
    let mut config = Config::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--demo" => {
                config.mode = Mode::Demo;
            }
            "--galaxy" => {
                config.mode = Mode::Galaxy;
            }
            "--materials" => {
                config.mode = Mode::Materials;
            }
            "--solar" => {
                config.mode = Mode::Solar;
            }
            "--window" => {
                config.window = true;
            }
            "--no-vsync" => {
                config.vsync = false;
            }
            "--glb" => match args.next() {
                Some(path) => config.glb = Some(PathBuf::from(path)),
                None => {
                    eprintln!("space-client: --glb needs a .glb file");
                    std::process::exit(2);
                }
            },
            "--glb-scale" => match args.next().and_then(|s| s.parse::<f32>().ok()) {
                Some(scale) if scale.is_finite() && scale > 0.0 => config.glb_scale = Some(scale),
                _ => {
                    eprintln!("space-client: --glb-scale needs a positive number");
                    std::process::exit(2);
                }
            },
            "--size" => match args.next().as_deref().and_then(parse_size) {
                Some((w, h)) => {
                    config.width = w;
                    config.height = h;
                }
                None => {
                    eprintln!(
                        "space-client: --size needs WxH with both sides in \
                         1..={MAX_FRAME_DIMENSION}, e.g. 1280x720"
                    );
                    std::process::exit(2);
                }
            },
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
    if config.window && config.stream.is_some() {
        eprintln!("space-client: --window and --stream are mutually exclusive");
        std::process::exit(2);
    }
    config
}

fn print_usage() {
    eprintln!(
        "usage: space-client [--demo | --galaxy | --materials | --solar] [--window]\n\
        \x20                   [--server ADDR] [--size WxH] [--quality high|medium|low|ci]\n\
        \x20                   [--no-vsync] [--frames N] [--out FILE.ppm] [--stream ADDR]\n\
        \n\
        With no arguments a window opens on the demo scene: W/A/S/D or the arrow\n\
        keys move, Space/Shift go up/down, dragging with the left mouse button\n\
        looks around, the mouse wheel changes the flying speed, Esc quits.\n\
        \n\
        A scene flag on its own renders one frame headlessly to a PPM file and\n\
        exits; add --window to fly around it instead (e.g. --window --solar).\n\
        \x20 --demo      the deterministic demo scene (a sphere and a cube)\n\
        \x20 --galaxy    the deterministic galaxy through the LOD selector\n\
        \x20 --materials a PBR material chart: gold, plastic and steel spheres from\n\
        \x20             mirror-smooth to rough, a mirror floor, glass and a lamp\n\
        \x20 --solar     the solar system at real size: the Sun, the planets and\n\
        \x20             the Moon as coloured spheres; you start just off Earth\n\
        \x20 --window   show the chosen scene in a window instead\n\
        \x20 --glb FILE  also show a .glb model (binary glTF: geometry and material\n\
        \x20             factors, no textures) in front of the spawn point\n\
        \x20 --glb-scale S  metres per model unit (default: fit to about 3 m)\n\
        \x20 --server ADDR  join the authoritative shard at ADDR (host:port) as a\n\
        \x20            player: input drives the ship, replication drives the\n\
        \x20            camera. Opens a window unless --stream is given\n\
        \x20 --size WxH frame/window size in pixels (default 640x480)\n\
        \x20 --quality P quality preset: high (default), medium, low or ci\n\
        \x20            (fog, screen-space reflections, refraction)\n\
        \x20 --no-vsync window: present as fast as frames render (benchmarking)\n\
        \x20 --frames N window/stream: exit after N frames\n\
        \x20 --out FILE offline render: write the frame to FILE (PPM);\n\
        \x20            default demo.ppm / galaxy.ppm\n\
        \x20 --stream ADDR  serve frames as JPEG over ws://ADDR (plain ws://, no\n\
        \x20            TLS) to the space-viewer wasm page; viewer input drives\n\
        \x20            the camera"
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
    let started = std::time::Instant::now();
    let galaxy = Galaxy::new(0x5eed);
    let mut lod = LodSelector::new(30, 63, 8.0);
    // Cull against the frustum that is actually on screen (the selector's
    // default is 16:9, which would clip the sides of a wider window).
    if camera.aspect.is_finite() && camera.aspect > 0.0 {
        lod.aspect = camera.aspect;
    }
    // `Scene::build` traverses only the one depth-30 node containing the camera.
    // The galaxy is centred on the cell origin and the camera sits on its x=0/y=0
    // boundary, so that node covers a single octant and the galaxy would fill one
    // frame quadrant. Traverse the 8 nodes of that depth meeting at the origin.
    //
    // This runs every frame the camera moves, so it selects in one pass per
    // root and looks each node's distance up once: every selected node that
    // holds a star, with its distance from the camera.
    let mut stars: Vec<(f64, NodeRef)> = Vec::new();
    for octant in 0..8i64 {
        let index = [-(octant & 1), -((octant >> 1) & 1), -((octant >> 2) & 1)];
        let root = NodeRef::new(lod.min_depth, index);
        for node in lod.select_from(root, camera.position, camera.forward, camera.fov_y) {
            if galaxy.star_at(node).is_some() {
                stars.push((lod.distance_to(node, camera.position), node));
            }
        }
    }
    // Nearest to the camera first, so the instance budget goes to the stars
    // around the viewer rather than to one side (stable: ties keep root order).
    stars.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Leave room for the spawn cube.
    stars.truncate(MAX_SELECTED - 1);
    let selected: Vec<NodeRef> = stars.into_iter().map(|(_, node)| node).collect();
    let mut scene = Scene::from_selection(&galaxy, camera, &selected);
    log::trace!(
        "galaxy scene: {} stars in {:.2} ms",
        scene.instances.len(),
        started.elapsed().as_secs_f64() * 1e3
    );
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
        Mode::Materials => scenes::materials_scene(&scenes::materials_camera(width, height)),
        Mode::Solar => scenes::solar_scene(&scenes::solar_camera(width, height)),
        Mode::Demo | Mode::Window => demo_scene(width, height),
    }
}

/// Write an RGBA8 buffer as a binary PPM (P6) file.
fn write_ppm(path: &std::path::Path, width: u32, height: u32, rgba: &[u8]) -> std::io::Result<()> {
    let pixels = width as usize * height as usize;
    if rgba.len() != pixels * 4 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{} bytes is not a {width}x{height} RGBA8 frame", rgba.len()),
        ));
    }
    let mut out = Vec::with_capacity(16 + pixels * 3);
    out.extend_from_slice(format!("P6\n{width} {height}\n255\n").as_bytes());
    for px in rgba.as_chunks::<4>().0 {
        out.extend_from_slice(&px[..3]);
    }
    std::fs::write(path, out)
}

/// The cheap, deterministic render options every front end uses (offline
/// render, window, stream), so the same preset gates the same features —
/// including the SSR/refraction post pass — whichever way the frame leaves the
/// GPU; the preset selects the optional render features.
fn demo_options(preset: QualityPreset) -> RenderOptions {
    RenderOptions {
        gi_enabled: false,
        quality: Quality::CI,
        preset,
        ..Default::default()
    }
}

/// [`demo_options`] for a particular scene: the scenes at astronomical scale
/// turn the presets' distance fog off (it is tuned for metres, and would paint
/// every star and planet solid fog colour).
fn scene_options(mode: Mode, preset: QualityPreset) -> RenderOptions {
    let fog_scale = match mode {
        Mode::Galaxy | Mode::Solar => 0.0,
        Mode::Demo | Mode::Materials | Mode::Window => 1.0,
    };
    RenderOptions {
        fog_scale,
        ..demo_options(preset)
    }
}

fn run_offscreen(config: &Config) -> Result<(), RenderError> {
    let ctx = RenderContext::new(&ContextOptions::default())?;
    eprintln!(
        "space-client: device {:?} ray_query={}",
        ctx.device_name(),
        ctx.ray_query_support().complete()
    );
    let mut scene = scene_for(config.mode, config.width, config.height);
    if let Some(model) = &config.model {
        model.add_to(&mut scene);
    }
    let out = ctx.render_offscreen(
        &scene,
        config.width,
        config.height,
        &scene_options(config.mode, config.quality),
    )?;
    let default = PathBuf::from(format!("{}.ppm", config.mode.name()));
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

fn main() {
    env_logger::init();
    let mut config = parse_args();
    if let Some(path) = &config.glb {
        match PlacedModel::load(path, config.glb_scale, config.mode) {
            Ok(model) => config.model = Some(model),
            Err(e) => {
                eprintln!("space-client: {e}");
                std::process::exit(1);
            }
        }
    }
    let result = match &config.stream {
        Some(addr) => stream::run_stream(&config, addr),
        None if config.windowed() => window::run_window(&config),
        None => run_offscreen(&config),
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
                space_render::skip_or_require_gpu(&e);
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

    /// Every preset (including the SSR/refraction presets) renders cleanly with
    /// the client's options, with zero validation messages, and the presets
    /// actually *differ*: High/Medium show the SSR reflection and fog that
    /// Low/Ci (identical to each other) do not, and High's denser fog differs
    /// from Medium's. Rendering each scene at every preset back to back also
    /// makes the context rebuild its cached frame whenever the feature set
    /// changes, and each preset is rendered twice to prove the reused frame
    /// reproduces the first render exactly.
    #[test]
    fn every_preset_renders_and_the_presets_differ() {
        let ctx = match RenderContext::new(&ContextOptions::default()) {
            Ok(ctx) => ctx,
            Err(e) => {
                space_render::skip_or_require_gpu(&e);
                return;
            }
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
            let out = ctx
                .render_offscreen(scene, size, size, &options)
                .expect("frame render");
            assert_eq!(out.validation_messages, 0, "{preset}");
            assert!(
                out.pixels.as_chunks::<4>().0.iter().any(|px| px[0] > 8),
                "{preset}: nothing rendered"
            );
            // The second render reuses the cached frame resources.
            let again = ctx
                .render_offscreen(scene, size, size, &options)
                .expect("second render");
            assert_eq!(again.validation_messages, 0, "{preset} (reused frame)");
            assert!(
                out.pixels == again.pixels,
                "{preset}: the reused frame rendered different pixels"
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
        assert!(cube.center[0].abs() < 1e-3 && cube.center[1].abs() < 1e-3);
        assert!((f64::from(cube.center[2]) - SPAWN_CUBE_DISTANCE_M).abs() < 1e-3);
        assert_eq!(cube.center[3], SPAWN_CUBE_HALF_M);
    }

    #[test]
    fn size_argument_is_validated() {
        assert_eq!(parse_size("1280x720"), Some((1280, 720)));
        assert_eq!(parse_size("64X48"), Some((64, 48)));
        for bad in [
            "", "640", "640x", "x480", "0x480", "640x0", "-1x4", "99999x4", "axb",
        ] {
            assert_eq!(parse_size(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn a_window_opens_unless_a_scene_is_rendered_offline_or_streamed() {
        let base = Config::default();
        assert!(base.windowed(), "no arguments: window");
        let offline = |mode| Config {
            mode,
            ..Config::default()
        };
        assert!(!offline(Mode::Demo).windowed(), "--demo renders offline");
        assert!(!offline(Mode::Galaxy).windowed());
        let shown = Config {
            window: true,
            ..offline(Mode::Galaxy)
        };
        assert!(shown.windowed(), "--window --galaxy");
        let online = Config {
            server: Some(([127, 0, 0, 1], 7777).into()),
            ..offline(Mode::Galaxy)
        };
        assert!(online.windowed(), "--server implies a window");
        let streamed = Config {
            stream: Some("127.0.0.1:8099".into()),
            ..online
        };
        assert!(!streamed.windowed(), "--stream never opens a window");
    }

    /// A buffer that is not `width * height` RGBA pixels is an error, not a
    /// panic (the size comes from the command line).
    #[test]
    fn write_ppm_rejects_a_mismatched_buffer() {
        let path = std::env::temp_dir().join("space_client_bad.ppm");
        assert!(write_ppm(&path, 2, 2, &[0u8; 4]).is_err());
        assert!(!path.exists());
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
