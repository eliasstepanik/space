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
//! Networking is deliberately *not* a dependency: this binary only ever talks to
//! `space-render` and `space-world`.

use std::path::PathBuf;

use space_render::{
<<<<<<< HEAD
    demo_scene, CameraPose, ContextOptions, Quality, QualityPreset, RenderContext, RenderError,
    RenderOptions, Scene, Swapchain, WindowHandles,
=======
    demo_scene, CameraPose, ContextOptions, Quality, RenderContext, RenderError, RenderOptions,
    Scene, Swapchain, WindowHandles,
>>>>>>> refs/remotes/origin/main
};
use space_world::{Galaxy, LodSelector, WorldPos};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
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
<<<<<<< HEAD
    quality: QualityPreset,
=======
>>>>>>> refs/remotes/origin/main
}

fn parse_args() -> Config {
    let mut config = Config {
        mode: Mode::Window,
        width: 640,
        height: 480,
        out: None,
<<<<<<< HEAD
        quality: QualityPreset::default(),
=======
>>>>>>> refs/remotes/origin/main
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
<<<<<<< HEAD
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
=======
>>>>>>> refs/remotes/origin/main
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
    config
}

fn print_usage() {
    eprintln!(
        "usage: space-client [--demo | --galaxy] [--size WxH] [--out FILE.ppm]\n\
<<<<<<< HEAD
               [--quality high|medium|low|ci]\n\
=======
>>>>>>> refs/remotes/origin/main
        \n\
        \x20 --demo     render the deterministic demo scene headlessly and exit\n\
        \x20 --galaxy   render the deterministic galaxy (LOD selector) headlessly and exit\n\
        \x20 --size WxH frame size in pixels (default 640x480)\n\
        \x20 --out FILE write the frame to FILE (PPM); default demo.ppm / galaxy.ppm\n\
<<<<<<< HEAD
        \x20 --quality P quality preset: high (default), medium, low or ci\n\
        \x20              (fog, screen-space reflections, refraction)\n\
=======
>>>>>>> refs/remotes/origin/main
        \n\
        With neither --demo nor --galaxy a winit window is opened."
    );
}

/// The deterministic galaxy scene, built through `space-world`'s LOD selector.
fn galaxy_scene(width: u32, height: u32) -> Scene {
    // Same camera as the `lod_galaxy_renders` GPU test: pulled back along -Z so
    // the LOD selector refines a dense band of the octree around the camera.
    let camera = CameraPose {
        position: WorldPos::from_f64_metres_around(WorldPos::ZERO, [0.0, 0.0, -5.0e4]),
        forward: [0.0, 0.0, -1.0],
        up: [0.0, 1.0, 0.0],
        fov_y: std::f32::consts::FRAC_PI_3,
        aspect: width as f32 / height as f32,
        near: 1.0,
        far: 5.0e6,
    };
    let galaxy = Galaxy::new(0x5eed);
    let lod = LodSelector::new(30, 63, 8.0);
    Scene::build(&galaxy, &lod, &camera)
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

<<<<<<< HEAD
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
=======
/// The CI-cheap, deterministic render options used by the offline demo.
fn demo_options() -> RenderOptions {
    RenderOptions {
        gi_enabled: false,
        quality: Quality::CI,
>>>>>>> refs/remotes/origin/main
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
<<<<<<< HEAD
    let out = ctx.render_offscreen(
        &scene,
        config.width,
        config.height,
        &demo_options(config.quality),
    )?;
=======
    let out = ctx.render_offscreen(&scene, config.width, config.height, &demo_options())?;
>>>>>>> refs/remotes/origin/main
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
}

impl App {
    fn new(config: &Config) -> Result<Self, RenderError> {
        Ok(Self {
            ctx: RenderContext::new(&ContextOptions::default())?,
            config: Config {
                mode: Mode::Window,
                width: config.width,
                height: config.height,
                out: None,
<<<<<<< HEAD
                quality: config.quality,
=======
>>>>>>> refs/remotes/origin/main
            },
            window: None,
            swapchain: None,
            scene: None,
<<<<<<< HEAD
            options: demo_options(config.quality),
=======
            options: demo_options(),
>>>>>>> refs/remotes/origin/main
            failed: false,
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
        self.scene = Some(scene_for(Mode::Window, sw, sh));
        self.swapchain = Some(swapchain);
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if let winit::keyboard::Key::Character(c) = &event.logical_key {
                    if c.as_str() == "Escape" || c.as_str() == "q" {
                        event_loop.exit();
                    }
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
    let mut app = App::new(config)?;
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
    let result = match config.mode {
        Mode::Demo | Mode::Galaxy => run_offscreen(&config),
        Mode::Window => run_window(&config),
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
        let Ok(ctx) = RenderContext::new(&ContextOptions::default()) else {
            eprintln!("SKIP: no Vulkan device");
            return;
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
<<<<<<< HEAD
            .render_offscreen(&scene, size, size, &demo_options(QualityPreset::Ci))
=======
            .render_offscreen(&scene, size, size, &demo_options())
>>>>>>> refs/remotes/origin/main
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

<<<<<<< HEAD
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

=======
>>>>>>> refs/remotes/origin/main
    /// The galaxy scene builds through the LOD selector and stays within the
    /// `MAX_SELECTED` bound.
    #[test]
    fn galaxy_scene_is_bounded() {
        let scene = galaxy_scene(64, 64);
        assert!(scene.instances.len() <= space_world::MAX_SELECTED);
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
