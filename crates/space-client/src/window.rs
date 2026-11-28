//! The window front end: a winit window presenting through the swapchain,
//! with the local keyboard and mouse as the application's input.
//!
//! Everything the window does goes through the application core
//! ([`space_client::app::App`]): a key or a mouse drag becomes a
//! [`ViewerInput`] — the same vocabulary the browser viewer and the harness
//! speak — and the core decides what it means on the screen being shown. So
//! on the connect screen the keyboard types the server address (the layout's
//! characters, paste with Ctrl+V or Shift+Insert on Windows, Backspace/Delete,
//! the arrow keys, Home/End; Enter connects, Esc quits), and while playing
//! `W`/`A`/`S`/`D` or the arrow keys move, `Space`/`Shift` go up/down, dragging
//! with the left mouse button looks around, `C` switches the camera, `B` the
//! brake assist, `H` the HUD and `Esc` returns to the connect screen.
//!
//! Frames are presented straight from the GPU; the pixels are read back only
//! for a frame the harness has asked for.

use std::time::{Duration, Instant};

use space_client::app::App;
use space_render::{
    validation_message_count, ContextOptions, PresentOutcome, RenderContext, RenderError,
    Swapchain, SwapchainConfig, WindowHandles,
};
use space_stream::protocol::is_typeable;
use space_stream::{Key, ViewerInput};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key as LogicalKey, KeyCode, ModifiersState, NamedKey, PhysicalKey};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle};
use winit::window::{Window, WindowId};

/// View rotation per pixel of mouse drag, radians (the browser viewer's feel).
const RADIANS_PER_PIXEL: f32 = 0.0025;

/// How often the title bar's readout is refreshed.
const TITLE_INTERVAL: Duration = Duration::from_millis(500);

/// How often the application is run while the window is minimised (nothing
/// can be presented, but the network and the harness still need it).
const MINIMISED_TICK: Duration = Duration::from_millis(100);

/// The most characters one paste types (an address is far shorter).
const MAX_PASTE_CHARS: usize = 512;

/// How the window is opened.
#[derive(Clone, Copy, Debug)]
pub struct WindowConfig {
    pub width: u32,
    pub height: u32,
    /// Wait for the vertical blank when presenting.
    pub vsync: bool,
    /// Close after this many presented frames.
    pub frames: Option<u64>,
}

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

/// The text-entry key a key *means* under the current layout (the browser
/// viewer's `Key::from_dom_key`).
fn text_entry_key(key: &LogicalKey) -> Option<Key> {
    let LogicalKey::Named(named) = key else {
        return None;
    };
    Some(match named {
        NamedKey::Enter => Key::Enter,
        NamedKey::Backspace => Key::Backspace,
        NamedKey::Escape => Key::Escape,
        NamedKey::Tab => Key::Tab,
        NamedKey::Delete => Key::Delete,
        NamedKey::Home => Key::Home,
        NamedKey::End => Key::End,
        NamedKey::ArrowLeft => Key::CaretLeft,
        NamedKey::ArrowRight => Key::CaretRight,
        _ => return None,
    })
}

/// Whether a key press is the paste shortcut: Ctrl+V (Cmd+V) or Shift+Insert.
fn is_paste(key: &LogicalKey, modifiers: ModifiersState) -> bool {
    match key {
        LogicalKey::Character(c) => {
            c.eq_ignore_ascii_case("v")
                && (modifiers.control_key() || modifiers.super_key())
                && !modifiers.alt_key()
        }
        LogicalKey::Named(NamedKey::Insert) => modifiers.shift_key(),
        LogicalKey::Named(NamedKey::Paste) => true,
        _ => false,
    }
}

/// The input events one key event amounts to, in order. `pasted` is the
/// clipboard's text when the event is the paste shortcut.
///
/// A key is reported under each meaning it has — its movement binding (by
/// position), its text-entry role and the characters it types — because only
/// the application knows which screen is up and so which meaning counts.
fn key_inputs(
    event: &KeyEvent,
    modifiers: ModifiersState,
    pasted: Option<&str>,
) -> Vec<ViewerInput> {
    let mut inputs = Vec::new();
    let down = event.state.is_pressed();
    if let PhysicalKey::Code(code) = event.physical_key {
        // Held keys: one down, one up; the auto-repeats in between say nothing new.
        if let Some(key) = movement_key(code).filter(|_| !event.repeat) {
            inputs.push(ViewerInput::Key { key, down });
        }
    }
    if !down {
        return inputs;
    }
    if is_paste(&event.logical_key, modifiers) {
        let text = pasted.unwrap_or_default();
        inputs.extend(typed(text).take(MAX_PASTE_CHARS));
        return inputs;
    }
    // Strokes: a press (or an auto-repeat) is a down directly followed by an up.
    if let Some(key) = text_entry_key(&event.logical_key) {
        inputs.push(ViewerInput::Key { key, down: true });
        inputs.push(ViewerInput::Key { key, down: false });
    }
    // The characters the layout produced (AltGr combinations included; a
    // Ctrl combination produces a control character, which is not text).
    if let Some(text) = &event.text {
        inputs.extend(typed(text));
    }
    inputs
}

/// `text` as typed characters (control characters are keys, not text).
fn typed(text: &str) -> impl Iterator<Item = ViewerInput> + '_ {
    text.chars()
        .filter(|c| is_typeable(*c))
        .map(ViewerInput::Text)
}

/// The system clipboard's text. Implemented for Windows only: winit has no
/// clipboard API, and the X11/Wayland protocols are far more than a few
/// system calls.
#[cfg(windows)]
mod clipboard {
    use std::ffi::c_void;

    const CF_UNICODETEXT: u32 = 13;
    /// The most UTF-16 units read from the clipboard.
    const MAX_UNITS: usize = 4096;

    #[link(name = "user32")]
    extern "system" {
        fn IsClipboardFormatAvailable(format: u32) -> i32;
        fn OpenClipboard(owner: *mut c_void) -> i32;
        fn CloseClipboard() -> i32;
        fn GetClipboardData(format: u32) -> *mut c_void;
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalLock(memory: *mut c_void) -> *mut c_void;
        fn GlobalUnlock(memory: *mut c_void) -> i32;
        fn GlobalSize(memory: *mut c_void) -> usize;
    }

    pub fn text() -> Option<String> {
        // SAFETY: plain Win32 clipboard calls. The clipboard is opened and
        // closed on this thread; the handle `GetClipboardData` returns stays
        // valid until `CloseClipboard`, and it is only read while locked, up
        // to its own size (`GlobalSize`) or the first NUL.
        unsafe {
            if IsClipboardFormatAvailable(CF_UNICODETEXT) == 0
                || OpenClipboard(std::ptr::null_mut()) == 0
            {
                return None;
            }
            let mut text = None;
            let handle = GetClipboardData(CF_UNICODETEXT);
            if !handle.is_null() {
                let data = GlobalLock(handle).cast::<u16>();
                if !data.is_null() {
                    let units = (GlobalSize(handle) / 2).min(MAX_UNITS);
                    let mut len = 0;
                    while len < units && *data.add(len) != 0 {
                        len += 1;
                    }
                    let slice = std::slice::from_raw_parts(data, len);
                    text = Some(String::from_utf16_lossy(slice));
                    GlobalUnlock(handle);
                }
            }
            CloseClipboard();
            text
        }
    }
}

#[cfg(not(windows))]
mod clipboard {
    pub fn text() -> Option<String> {
        None
    }
}

/// The windowed front end: owns the render context, window and swapchain and
/// drives the application from winit's `ApplicationHandler` callbacks.
struct Shell {
    // Field order is drop order, and it matters: the swapchain (and its
    // surface) must go before the render context (device, instance), and both
    // before the window they present to.
    swapchain: Option<Swapchain>,
    ctx: RenderContext,
    window: Option<Window>,
    app: App,

    config: WindowConfig,
    modifiers: ModifiersState,
    /// The left mouse button is held: cursor motion turns the view.
    dragging: bool,
    cursor: Option<(f64, f64)>,
    /// The swapchain no longer matches the window and must be recreated.
    stale: bool,
    /// The window is minimised (0x0): nothing is presented until it is resized.
    suspended: bool,
    /// The size of the last presented frame (what is rendered while minimised).
    extent: (u32, u32),
    minimised_tick: Instant,

    started: Instant,
    presented: u64,
    title_since: Instant,
    /// Validation messages seen before the first frame (loader chatter).
    validation_baseline: u32,
    failure: Option<RenderError>,
}

impl Shell {
    fn new(
        mut app: App,
        config: &WindowConfig,
        display: RawDisplayHandle,
    ) -> Result<Self, RenderError> {
        let ctx = RenderContext::new(&ContextOptions {
            window_display: Some(display),
            ..ContextOptions::default()
        })?;
        app.set_render_device(ctx.device_name());
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
            app,
            config: *config,
            modifiers: ModifiersState::empty(),
            dragging: false,
            cursor: None,
            stale: false,
            suspended: false,
            extent: (config.width, config.height),
            minimised_tick: now,
            started: now,
            presented: 0,
            title_since: now,
            validation_baseline: validation_message_count(),
            failure: None,
        })
    }

    /// Create the window and its swapchain.
    fn open(&mut self, event_loop: &ActiveEventLoop) -> Result<(), RenderError> {
        let attrs = Window::default_attributes()
            .with_title("space")
            .with_inner_size(PhysicalSize::new(self.config.width, self.config.height));
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
            SwapchainConfig {
                vsync: self.config.vsync,
            },
        )?;
        let (width, height) = swapchain.extent();
        eprintln!(
            "space-client: window {width}x{height}, vsync {}",
            if swapchain.is_vsynced() { "on" } else { "off" }
        );
        self.extent = (width, height);
        self.swapchain = Some(swapchain);
        self.window = Some(window);
        let now = Instant::now();
        self.started = now;
        self.title_since = now;
        self.validation_baseline = validation_message_count();
        Ok(())
    }

    /// Record a fatal error and leave the event loop.
    fn fail(&mut self, event_loop: &ActiveEventLoop, error: RenderError) {
        self.failure.get_or_insert(error);
        event_loop.exit();
    }

    fn inputs(&mut self, inputs: impl IntoIterator<Item = ViewerInput>) {
        for input in inputs {
            self.app.input(input);
        }
    }

    /// Run the application for one frame, render it and present it.
    /// `Ok(false)` when nothing was presented (minimised, or the swapchain has
    /// to be rebuilt).
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

        self.app.update();
        if self.app.should_quit() {
            return Ok(false);
        }
        let extent = swapchain.extent();
        let options = self.app.render_options();
        // Read the pixels back only when somebody asked for a picture.
        let wanted = self.app.frame_wanted();
        let scene = self.app.scene(extent.0, extent.1);
        let outcome = self.ctx.render_to_swapchain(scene, swapchain, &options)?;
        let presented = match outcome {
            PresentOutcome::Presented => true,
            PresentOutcome::Suboptimal => {
                // Rebuild only if the window really has another size now: a
                // surface that reports "suboptimal" for some other reason (a
                // rotated display, say) would otherwise be rebuilt every frame.
                let size = window.inner_size();
                self.stale = (size.width, size.height) != extent;
                true
            }
            PresentOutcome::OutOfDate => {
                self.stale = true;
                false
            }
        };
        if presented {
            let pixels = if wanted {
                let out = self
                    .ctx
                    .render_offscreen(scene, extent.0, extent.1, &options)?;
                Some(out.pixels)
            } else {
                None
            };
            self.extent = extent;
            self.app.observe_gpu_time(self.ctx.last_gpu_time());
            self.app.finish_frame(extent.0, extent.1, pixels);
        }
        Ok(presented)
    }

    /// While minimised nothing can be presented, but the application must not
    /// stand still: the link wants its input, and a harness waiting for a
    /// frame should get one. Run it offscreen at the last size, slowly.
    fn minimised_frame(&mut self) -> Result<(), RenderError> {
        self.app.update();
        if self.app.should_quit() {
            return Ok(());
        }
        let (width, height) = self.extent;
        let options = self.app.render_options();
        let scene = self.app.scene(width, height);
        let out = self.ctx.render_offscreen(scene, width, height, &options)?;
        self.app.observe_gpu_time(out.gpu_time);
        self.app.finish_frame(width, height, Some(out.pixels));
        Ok(())
    }

    /// Count a presented frame and refresh the title bar's readout.
    fn count_frame(&mut self) {
        self.presented += 1;
        if self.presented == 1 {
            eprintln!("space-client: presenting");
        }
        if self.title_since.elapsed() < TITLE_INTERVAL {
            return;
        }
        if let Some(window) = &self.window {
            window.set_title(&format!(
                "{} - {}",
                self.app.title(),
                self.ctx.device_name()
            ));
        }
        self.title_since = Instant::now();
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

impl ApplicationHandler for Shell {
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
            WindowEvent::CloseRequested => {
                self.app.request_quit();
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                self.stale = true;
                self.suspended = false;
            }
            WindowEvent::Focused(false) => {
                // The key-ups will go to whichever window has the focus now.
                self.app.release_keys();
                self.modifiers = ModifiersState::empty();
                self.dragging = false;
                self.cursor = None;
            }
            WindowEvent::ModifiersChanged(modifiers) => self.modifiers = modifiers.state(),
            WindowEvent::KeyboardInput {
                event,
                is_synthetic,
                ..
            } => {
                // winit invents a press for every key that is already down
                // when the window gains the focus (Alt+Tab's Tab, a Shift
                // held while clicking in). Nobody pressed those here.
                if is_synthetic && event.state.is_pressed() {
                    return;
                }
                let pasted = (event.state.is_pressed()
                    && is_paste(&event.logical_key, self.modifiers))
                .then(clipboard::text)
                .flatten();
                let inputs = key_inputs(&event, self.modifiers, pasted.as_deref());
                self.inputs(inputs);
            }
            WindowEvent::Ime(winit::event::Ime::Commit(text)) => {
                let inputs: Vec<ViewerInput> = typed(&text).collect();
                self.inputs(inputs);
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => self.dragging = state.is_pressed(),
            WindowEvent::CursorMoved { position, .. } => {
                if let (true, Some((x, y))) = (self.dragging, self.cursor) {
                    // Dragging right turns right (negative yaw), dragging
                    // down looks down.
                    self.app.input(ViewerInput::Turn {
                        dyaw: -((position.x - x) as f32) * RADIANS_PER_PIXEL,
                        dpitch: -((position.y - y) as f32) * RADIANS_PER_PIXEL,
                    });
                }
                self.cursor = Some((position.x, position.y));
            }
            WindowEvent::CursorLeft { .. } => self.cursor = None,
            WindowEvent::RedrawRequested => {
                match self.frame() {
                    Ok(true) => self.count_frame(),
                    Ok(false) => {}
                    Err(e) => self.fail(event_loop, e),
                }
                let done = self
                    .config
                    .frames
                    .is_some_and(|limit| self.presented >= limit);
                if done || self.app.should_quit() {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.app.should_quit() {
            event_loop.exit();
            return;
        }
        if self.suspended {
            // Minimised: tick the application now and then until an event
            // (the restoring resize) arrives.
            let now = Instant::now();
            if now.duration_since(self.minimised_tick) >= MINIMISED_TICK {
                self.minimised_tick = now;
                if let Err(e) = self.minimised_frame() {
                    self.fail(event_loop, e);
                    return;
                }
            }
            event_loop
                .set_control_flow(ControlFlow::WaitUntil(self.minimised_tick + MINIMISED_TICK));
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

/// Open the window and run `app` in it until it quits, the window is closed
/// or `config.frames` frames were presented.
pub fn run_window(app: App, config: &WindowConfig) -> Result<(), RenderError> {
    let event_loop = EventLoop::new().map_err(|e| RenderError::new(format!("event loop: {e}")))?;
    // The instance must be created with the surface extensions this display
    // needs, so take the display handle from the event loop before building
    // the render context.
    let display = event_loop
        .display_handle()
        .map_err(|e| RenderError::new(format!("display_handle: {e}")))?
        .as_raw();
    let mut shell = Shell::new(app, config, display)?;
    eprintln!(
        "space-client: type the server address and press Enter; while playing: {}",
        space_client::input::KEY_HELP
    );
    let run = event_loop
        .run_app(&mut shell)
        .map_err(|e| RenderError::new(format!("event loop: {e}")));
    shell.report();
    let failure = shell.failure.take();
    // GPU objects first, then the window (the struct's field order).
    drop(shell);
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
        assert_eq!(movement_key(KeyCode::Enter), None);
    }

    #[test]
    fn text_entry_keys_match_the_browser_viewer() {
        // The viewer maps `KeyboardEvent.key` with `Key::from_dom_key`.
        for (named, dom) in [
            (NamedKey::Enter, "Enter"),
            (NamedKey::Backspace, "Backspace"),
            (NamedKey::Escape, "Escape"),
            (NamedKey::Tab, "Tab"),
            (NamedKey::Delete, "Delete"),
            (NamedKey::Home, "Home"),
            (NamedKey::End, "End"),
            (NamedKey::ArrowLeft, "ArrowLeft"),
            (NamedKey::ArrowRight, "ArrowRight"),
        ] {
            assert_eq!(
                text_entry_key(&LogicalKey::Named(named)),
                Key::from_dom_key(dom),
                "{dom}"
            );
        }
        // Up/Down move nothing in a one-line field; letters are text.
        assert_eq!(text_entry_key(&LogicalKey::Named(NamedKey::ArrowUp)), None);
        assert_eq!(text_entry_key(&LogicalKey::Named(NamedKey::Shift)), None);
        assert_eq!(text_entry_key(&LogicalKey::Character("w".into())), None);
    }

    #[test]
    fn the_paste_shortcuts_are_recognised() {
        let v = LogicalKey::Character("v".into());
        let insert = LogicalKey::Named(NamedKey::Insert);
        assert!(is_paste(&v, ModifiersState::CONTROL));
        assert!(is_paste(
            &LogicalKey::Character("V".into()),
            ModifiersState::CONTROL
        ));
        assert!(is_paste(&v, ModifiersState::SUPER));
        assert!(is_paste(&insert, ModifiersState::SHIFT));
        assert!(!is_paste(&v, ModifiersState::empty()), "v types a v");
        assert!(!is_paste(&insert, ModifiersState::empty()));
        // AltGr is Ctrl+Alt on Windows: AltGr+V must not paste.
        assert!(!is_paste(&v, ModifiersState::CONTROL | ModifiersState::ALT));
        assert!(!is_paste(
            &LogicalKey::Character("c".into()),
            ModifiersState::CONTROL
        ));
    }

    /// Reading the clipboard never fails loudly, whatever is on it (the
    /// content is the user's: it is not looked at here, and not changed).
    #[test]
    fn reading_the_clipboard_is_harmless() {
        let before = clipboard::text();
        let again = clipboard::text();
        assert_eq!(before.is_some(), again.is_some());
        if let Some(text) = again {
            assert!(text.chars().count() <= 4096);
        }
    }

    #[test]
    fn pasted_and_typed_text_drops_control_characters() {
        let inputs: Vec<ViewerInput> = typed("1.2.3.4:7\r\n\u{16}").collect();
        let text: String = inputs
            .iter()
            .map(|input| match input {
                ViewerInput::Text(c) => *c,
                other => panic!("not text: {other:?}"),
            })
            .collect();
        assert_eq!(text, "1.2.3.4:7");
    }
}
