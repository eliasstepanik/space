//! The windowed client: a winit window presenting through the swapchain path,
//! driven by the local keyboard and mouse.
//!
//! Offline, the input flies a free camera through the scene (the demo scene,
//! or the galaxy with `--galaxy`). With `--server ADDR` the process is an
//! ordinary player on the authoritative shard ([`space_client::link`]): the
//! held keys become thrust intent and the replicated ship position drives the
//! camera — the same seam the streaming proxy uses ([`crate::stream`]), with
//! this window in place of a browser viewer.
//!
//! Controls: `W`/`A`/`S`/`D` (or the arrow keys) move, `Space`/`Shift` go
//! up/down, dragging with the left mouse button looks around, the mouse wheel
//! changes the flying speed (offline), `Esc` or `Q` quits.

use std::time::{Duration, Instant};

use space_client::link::ServerLink;
use space_render::{
    validation_message_count, ContextOptions, PresentOutcome, RenderContext, RenderError,
    RenderOptions, Scene, Swapchain, SwapchainConfig, WindowHandles,
};
use space_stream::{Key, ViewerInput};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle};
use winit::window::{Window, WindowId};

use crate::stream::{initial_input, scene_for_input, scene_speed, InputState};
use crate::{scene_options, scenes, Config, Mode, PlacedModel};

/// View rotation per pixel of mouse drag, radians (the browser viewer's feel).
const RADIANS_PER_PIXEL: f32 = 0.0025;

/// How often the title bar's frame-rate readout is refreshed.
const STATS_INTERVAL: Duration = Duration::from_millis(500);

/// One notch of the mouse wheel changes the flying speed by this factor.
const SPEED_STEP: f64 = 1.25;

/// The fastest the free camera flies, metres/second (a few thousand times the
/// speed of light: Neptune's orbit in a handful of seconds).
const MAX_SPEED: f64 = 1.0e12;

/// The movement key a physical key is bound to (the browser viewer's layout:
/// position-based, so it is the same keys on every keyboard layout).
fn movement_key(code: KeyCode) -> Option<Key> {
    Some(match code {
        KeyCode::KeyW | KeyCode::ArrowUp => Key::Forward,
        KeyCode::KeyS | KeyCode::ArrowDown => Key::Back,
        KeyCode::KeyA | KeyCode::ArrowLeft => Key::Left,
        KeyCode::KeyD | KeyCode::ArrowRight => Key::Right,
        KeyCode::Space => Key::Up,
        KeyCode::ShiftLeft | KeyCode::ShiftRight => Key::Down,
        _ => return None,
    })
}

/// What the camera saw when the cached scene was built: the scene is only
/// rebuilt when this changes.
type Pose = ([f64; 3], f32, f32, (u32, u32));

/// The windowed app: owns the render context, window and swapchain and drives
/// them from winit's `ApplicationHandler` callbacks.
struct App {
    // Field order is drop order, and it matters: the swapchain (and its
    // surface) must go before the render context (device, instance), and both
    // before the window they present to.
    swapchain: Option<Swapchain>,
    ctx: RenderContext,
    window: Option<Window>,
    // The link's tasks run on the runtime, so the link goes first.
    link: Option<ServerLink>,
    _runtime: Option<tokio::runtime::Runtime>,

    mode: Mode,
    /// The `--glb` model, if any.
    model: Option<PlacedModel>,
    size: (u32, u32),
    vsync: bool,
    options: RenderOptions,
    /// Free-camera speed when offline, metres/second.
    speed: f64,
    frame_limit: Option<u64>,

    input: InputState,
    /// The left mouse button is held: cursor motion turns the view.
    dragging: bool,
    cursor: Option<(f64, f64)>,
    scene: Option<(Pose, Scene)>,
    /// The swapchain no longer matches the window and must be recreated.
    stale: bool,
    /// The window is minimised (0x0): nothing is rendered until it is resized.
    suspended: bool,

    last_frame: Instant,
    started: Instant,
    presented: u64,
    stats_since: Instant,
    stats_frames: u32,
    /// Validation messages seen before the first frame (loader chatter).
    validation_baseline: u32,
    failure: Option<RenderError>,
}

impl App {
    fn new(config: &Config, display: RawDisplayHandle) -> Result<Self, RenderError> {
        // Join the shard first, so a wrong address fails before a window opens.
        let (runtime, link) = match config.server {
            Some(server) => {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                    .map_err(|e| RenderError::new(format!("tokio runtime: {e}")))?;
                let link = runtime
                    .block_on(ServerLink::connect(server))
                    .map_err(|e| RenderError::new(format!("server {server}: {e}")))?;
                eprintln!(
                    "space-client: connected to {server} as session {}",
                    link.replica().session
                );
                (Some(runtime), Some(link))
            }
            None => (None, None),
        };
        let ctx = RenderContext::new(&ContextOptions {
            window_display: Some(display),
            ..ContextOptions::default()
        })?;
        eprintln!(
            "space-client: device {:?} ray_query={}",
            ctx.device_name(),
            ctx.ray_query_support().complete()
        );
        let now = Instant::now();
        Ok(Self {
            swapchain: None,
            ctx,
            window: None,
            link,
            _runtime: runtime,
            mode: config.mode,
            model: config.model.clone(),
            size: (config.width, config.height),
            vsync: config.vsync,
            options: scene_options(config.mode, config.quality),
            speed: scene_speed(config.mode),
            frame_limit: config.frames,
            input: initial_input(config.mode),
            dragging: false,
            cursor: None,
            scene: None,
            stale: false,
            suspended: false,
            last_frame: now,
            started: now,
            presented: 0,
            stats_since: now,
            stats_frames: 0,
            validation_baseline: validation_message_count(),
            failure: None,
        })
    }

    fn scene_name(&self) -> &'static str {
        self.mode.name()
    }

    /// Change the flying speed by `notches` of the mouse wheel.
    fn change_speed(&mut self, notches: f64) {
        if !notches.is_finite() {
            return;
        }
        let slowest = scene_speed(self.mode) * 1.0e-3;
        self.speed = (self.speed * SPEED_STEP.powf(notches)).clamp(slowest, MAX_SPEED);
    }

    /// Where the camera is, for the title bar: the nearest body of a scene
    /// that has named bodies, and how far away its surface is.
    fn whereabouts(&self) -> Option<String> {
        if self.mode != Mode::Solar {
            return None;
        }
        let (width, height) = self.swapchain.as_ref()?.extent();
        let camera = self.input.camera(&scenes::solar_camera(width, height));
        let bodies = scenes::solar_bodies();
        let (body, surface) = scenes::nearest(&bodies, camera.position)?;
        Some(format!(
            "{} from {}",
            scenes::format_distance(surface),
            body.name
        ))
    }

    /// Create the window and its swapchain.
    fn open(&mut self, event_loop: &ActiveEventLoop) -> Result<(), RenderError> {
        let attrs = Window::default_attributes()
            .with_title("space")
            .with_inner_size(PhysicalSize::new(self.size.0, self.size.1));
        let window = event_loop
            .create_window(attrs)
            .map_err(|e| RenderError::new(format!("window: {e}")))?;
        let handles = window_handles(&window)?;
        let size = window.inner_size();
        let swapchain = Swapchain::with_config(
            &self.ctx,
            handles,
            size.width.max(1),
            size.height.max(1),
            SwapchainConfig { vsync: self.vsync },
        )?;
        let (width, height) = swapchain.extent();
        eprintln!(
            "space-client: window {width}x{height}, {} scene, vsync {}",
            self.scene_name(),
            if swapchain.is_vsynced() { "on" } else { "off" }
        );
        self.swapchain = Some(swapchain);
        self.window = Some(window);
        let now = Instant::now();
        self.last_frame = now;
        self.started = now;
        self.stats_since = now;
        self.validation_baseline = validation_message_count();
        Ok(())
    }

    /// Record a fatal error and leave the event loop.
    fn fail(&mut self, event_loop: &ActiveEventLoop, error: RenderError) {
        self.failure.get_or_insert(error);
        event_loop.exit();
    }

    /// Turn the view by a mouse drag of `(dx, dy)` pixels.
    fn turn(&mut self, dx: f64, dy: f64) {
        // Dragging right turns right (negative yaw), dragging down looks down;
        // `apply` wraps the yaw and clamps the pitch short of straight up/down.
        self.input.apply(ViewerInput::Turn {
            dyaw: -(dx as f32) * RADIANS_PER_PIXEL,
            dpitch: -(dy as f32) * RADIANS_PER_PIXEL,
        });
    }

    /// Advance the camera, render one frame and present it. `Ok(false)` when
    /// nothing was presented (minimised, or the swapchain has to be rebuilt).
    fn frame(&mut self) -> Result<bool, RenderError> {
        let (Some(window), Some(swapchain)) = (self.window.as_ref(), self.swapchain.as_mut())
        else {
            return Ok(false);
        };
        if self.stale {
            let size = window.inner_size();
            let usable = size.width > 0 && size.height > 0;
            if !usable || !swapchain.recreate(size.width, size.height)? {
                // Minimised: wait for the resize that brings the window back.
                self.suspended = true;
                return Ok(false);
            }
            self.stale = false;
        }
        if self.suspended {
            return Ok(false);
        }

        let now = Instant::now();
        // Clamped so a stall (a dragged window, a breakpoint) is not a jump.
        let dt = now.duration_since(self.last_frame).as_secs_f32().min(0.25);
        self.last_frame = now;
        match &self.link {
            Some(link) => {
                if link.is_closed() {
                    return Err(RenderError::new("lost the server connection"));
                }
                // The server is authoritative: send intent, show its answer.
                link.set_thrust(self.input.thrust());
                self.input.offset = link.offset();
            }
            None => self.input.step(dt, self.speed),
        }

        let extent = swapchain.extent();
        let pose: Pose = (self.input.offset, self.input.yaw, self.input.pitch, extent);
        if self.scene.as_ref().is_none_or(|(built, _)| *built != pose) {
            let mut scene = scene_for_input(self.mode, &self.input, extent.0, extent.1);
            if let Some(model) = &self.model {
                model.add_to(&mut scene);
            }
            self.scene = Some((pose, scene));
        }
        let Some((_, scene)) = &self.scene else {
            return Ok(false);
        };
        match self
            .ctx
            .render_to_swapchain(scene, swapchain, &self.options)?
        {
            PresentOutcome::Presented => Ok(true),
            PresentOutcome::Suboptimal => {
                // Rebuild only if the window really has another size now: a
                // surface that reports "suboptimal" for some other reason (a
                // rotated display, say) would otherwise be rebuilt every frame.
                let size = window.inner_size();
                self.stale = (size.width, size.height) != extent;
                Ok(true)
            }
            PresentOutcome::OutOfDate => {
                self.stale = true;
                Ok(false)
            }
        }
    }

    /// Count a presented frame and refresh the title bar's readout.
    fn count_frame(&mut self) {
        self.presented += 1;
        self.stats_frames += 1;
        if self.presented == 1 {
            eprintln!("space-client: presenting");
        }
        let elapsed = self.stats_since.elapsed();
        if elapsed < STATS_INTERVAL {
            return;
        }
        let fps = f64::from(self.stats_frames) / elapsed.as_secs_f64();
        if let Some(window) = &self.window {
            // Online the server moves the ship (and the title says which shard
            // is simulating it); offline the wheel sets the speed.
            let motion = match &self.link {
                Some(link) => {
                    let replica = link.replica();
                    let shard = replica
                        .shard
                        .map_or_else(|| "?".to_owned(), |shard| shard.0.to_string());
                    match replica.handoffs {
                        0 => format!("online, shard {shard}"),
                        n => format!("online, shard {shard} after {n} handoff(s)"),
                    }
                }
                None => format!("{}/s", scenes::format_distance(self.speed)),
            };
            let place = self
                .whereabouts()
                .map_or_else(String::new, |place| format!(" — {place}"));
            window.set_title(&format!(
                "space — {fps:.0} fps ({:.2} ms) — {} — {motion}{place} — {}",
                1000.0 / fps.max(1e-9),
                self.scene_name(),
                self.ctx.device_name()
            ));
        }
        self.stats_since = Instant::now();
        self.stats_frames = 0;
    }

    /// One line on what the session did, printed when the window closes.
    fn report(&self) {
        let seconds = self.started.elapsed().as_secs_f64();
        eprintln!(
            "space-client: presented {} frames in {seconds:.1} s ({:.0} fps average), \
             {} validation messages",
            self.presented,
            self.presented as f64 / seconds.max(1e-9),
            validation_message_count().wrapping_sub(self.validation_baseline)
        );
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        if let Err(e) = self.open(event_loop) {
            self.fail(event_loop, e);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) => {
                self.stale = true;
                self.suspended = false;
            }
            WindowEvent::Focused(false) => {
                // The key-ups will go to whichever window has the focus now.
                self.input.release_keys();
                self.dragging = false;
                self.cursor = None;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(code) = event.physical_key else {
                    return;
                };
                let down = event.state.is_pressed();
                if let Some(key) = movement_key(code) {
                    self.input.apply(ViewerInput::Key { key, down });
                } else if down && matches!(code, KeyCode::Escape | KeyCode::KeyQ) {
                    event_loop.exit();
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => self.dragging = state.is_pressed(),
            WindowEvent::CursorMoved { position, .. } => {
                if let (true, Some((x, y))) = (self.dragging, self.cursor) {
                    self.turn(position.x - x, position.y - y);
                }
                self.cursor = Some((position.x, position.y));
            }
            WindowEvent::CursorLeft { .. } => self.cursor = None,
            WindowEvent::MouseWheel { delta, .. } => {
                let notches = match delta {
                    MouseScrollDelta::LineDelta(_, lines) => f64::from(lines),
                    // Touchpads report pixels; ~40 px to a wheel notch.
                    MouseScrollDelta::PixelDelta(offset) => offset.y / 40.0,
                };
                self.change_speed(notches);
            }
            WindowEvent::RedrawRequested => {
                match self.frame() {
                    Ok(true) => self.count_frame(),
                    Ok(false) => {}
                    Err(e) => self.fail(event_loop, e),
                }
                if self
                    .frame_limit
                    .is_some_and(|limit| self.presented >= limit)
                {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.suspended {
            // Minimised: sleep until an event (the restoring resize) arrives.
            event_loop.set_control_flow(ControlFlow::Wait);
        } else {
            event_loop.set_control_flow(ControlFlow::Poll);
            if let Some(window) = &self.window {
                window.request_redraw();
            }
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

/// Open the window and run until it is closed (or `--frames` is reached).
pub fn run_window(config: &Config) -> Result<(), RenderError> {
    let event_loop = EventLoop::new().map_err(|e| RenderError::new(format!("event loop: {e}")))?;
    // The instance must be created with the surface extensions this display
    // needs, so take the display handle from the event loop before building
    // the render context.
    let display = event_loop
        .display_handle()
        .map_err(|e| RenderError::new(format!("display_handle: {e}")))?
        .as_raw();
    let mut app = App::new(config, display)?;
    eprintln!(
        "space-client: W/A/S/D or arrows move, Space/Shift go up/down, drag with the \
         left mouse button to look, the wheel changes speed, Esc quits"
    );
    let run = event_loop
        .run_app(&mut app)
        .map_err(|e| RenderError::new(format!("event loop: {e}")));
    app.report();
    let failure = app.failure.take();
    // GPU objects first, then the window (the struct's field order).
    drop(app);
    run?;
    failure.map_or(Ok(()), Err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn movement_keys_match_the_browser_viewer() {
        // The viewer maps DOM codes with `Key::from_dom_code`; the window must
        // bind the same physical keys to the same actions.
        for (code, dom) in [
            (KeyCode::KeyW, "KeyW"),
            (KeyCode::KeyA, "KeyA"),
            (KeyCode::KeyS, "KeyS"),
            (KeyCode::KeyD, "KeyD"),
            (KeyCode::ArrowUp, "ArrowUp"),
            (KeyCode::ArrowDown, "ArrowDown"),
            (KeyCode::ArrowLeft, "ArrowLeft"),
            (KeyCode::ArrowRight, "ArrowRight"),
            (KeyCode::Space, "Space"),
            (KeyCode::ShiftLeft, "ShiftLeft"),
            (KeyCode::ShiftRight, "ShiftRight"),
        ] {
            assert_eq!(movement_key(code), Key::from_dom_code(dom), "{dom}");
        }
        assert_eq!(movement_key(KeyCode::Escape), None);
        assert_eq!(movement_key(KeyCode::KeyQ), None);
    }
}
