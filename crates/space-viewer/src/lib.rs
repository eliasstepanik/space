//! `space-viewer` — the thin wasm browser viewer for `space-stream`.
//!
//! It owns no game state. [`web::start`] opens a WebSocket to a `space-client
//! --stream` proxy, draws the JPEG frames it receives on a `<canvas>`
//! (`createImageBitmap` on a `Blob`), and sends mouse/keyboard input back as
//! [`space_stream::ViewerInput`] messages. When the connection drops it says so
//! in a status line and reconnects with a backoff.
//!
//! # Multiplayer seam
//!
//! The viewer never talks to the authoritative game server. Its input goes to the
//! headless native proxy client (`space-client --stream`), which *is* the player
//! as far as the server is concerned: it connects through the ordinary
//! `space-net` client path and injects this viewer's input into its own input
//! state, so it reaches the server over the existing netcode. No new server
//! protocol exists. See the `space-stream` crate docs.
//!
//! The proxy also owns the view direction: the viewer only reports how far the
//! mouse was dragged ([`turn_message`]) and never an absolute angle, so a
//! reload or a second viewer cannot make the camera jump.
//!
//! Transport is plain `ws://` (see `space-stream` for why `wss://` is deferred).
//!
//! The pure logic (input serialisation, layout maths, the frame slot, reconnect
//! timing, status text) is at the top of this file and unit-tested on the host;
//! the browser glue is in [`web`].
#![forbid(unsafe_code)]

use space_stream::protocol::{Key, ViewerInput};

/// Radians of view rotation per pixel of mouse drag.
pub const RADIANS_PER_PIXEL: f32 = 0.0025;
/// The port `space-client --stream` is conventionally bound to.
pub const DEFAULT_STREAM_PORT: u16 = 8099;
/// Delay before the first reconnect attempt, in milliseconds.
pub const RECONNECT_BASE_MS: u32 = 500;
/// Longest delay between reconnect attempts, in milliseconds.
pub const RECONNECT_MAX_MS: u32 = 8_000;

/// Wire message for a DOM key event, or `None` if the key is not mapped.
pub fn key_message(dom_code: &str, down: bool) -> Option<Vec<u8>> {
    let key = Key::from_dom_code(dom_code)?;
    Some(ViewerInput::Key { key, down }.encode())
}

/// Key-up wire messages for every held DOM key code (sent on window blur, when
/// the browser will not deliver the real key-ups).
pub fn release_messages<'a>(held: impl IntoIterator<Item = &'a String>) -> Vec<Vec<u8>> {
    held.into_iter()
        .filter_map(|code| key_message(code, false))
        .collect()
}

/// Key-down wire messages for every held DOM key code (sent when a connection
/// opens: the keys the user is still holding after a reconnect, which the new
/// connection has never heard about and the browser will not report again).
pub fn press_messages<'a>(held: impl IntoIterator<Item = &'a String>) -> Vec<Vec<u8>> {
    held.into_iter()
        .filter_map(|code| key_message(code, true))
        .collect()
}

/// Wire message for a pointer at CSS-pixel `(x, y)` inside a `width`×`height`
/// rectangle, normalised to `0..=1` (clamped). `None` for an empty rect.
pub fn pointer_message(x: f64, y: f64, width: f64, height: f64, buttons: u8) -> Option<Vec<u8>> {
    if !(width > 0.0 && height > 0.0) || !x.is_finite() || !y.is_finite() {
        return None;
    }
    Some(
        ViewerInput::Pointer {
            x: (x / width).clamp(0.0, 1.0) as f32,
            y: (y / height).clamp(0.0, 1.0) as f32,
            buttons,
        }
        .encode(),
    )
}

/// Wire message for a mouse drag of `(dx, dy)` pixels: a *relative* rotation
/// (right = turn right, down = look down). `None` when there is nothing to
/// send (no movement, or a non-finite delta).
pub fn turn_message(dx: f64, dy: f64) -> Option<Vec<u8>> {
    let radians = |pixels: f64| {
        let radians = -(pixels as f32) * RADIANS_PER_PIXEL;
        // An unmoved axis is `-0.0` here; put a plain zero on the wire.
        if radians == 0.0 {
            0.0
        } else {
            radians
        }
    };
    let (dyaw, dpitch) = (radians(dx), radians(dy));
    if !dyaw.is_finite() || !dpitch.is_finite() || (dyaw == 0.0 && dpitch == 0.0) {
        return None;
    }
    Some(ViewerInput::Turn { dyaw, dpitch }.encode())
}

/// Where an `image_w`×`image_h` picture ends up inside a `box_w`×`box_h`
/// element that shows it with `object-fit: contain` (scaled to fit, aspect
/// kept, centred): `(left, top, width, height)` relative to the element.
///
/// The canvas element fills the window while the frame keeps its own aspect
/// ratio, so pointer positions must be measured against this rectangle, not
/// against the element. `None` if any size is zero, negative or not finite.
pub fn contained_rect(
    box_w: f64,
    box_h: f64,
    image_w: f64,
    image_h: f64,
) -> Option<(f64, f64, f64, f64)> {
    let usable = |v: f64| v.is_finite() && v > 0.0;
    if !(usable(box_w) && usable(box_h) && usable(image_w) && usable(image_h)) {
        return None;
    }
    let scale = (box_w / image_w).min(box_h / image_h);
    let (width, height) = (image_w * scale, image_h * scale);
    Some(((box_w - width) / 2.0, (box_h - height) / 2.0, width, height))
}

/// The stream address to use when the page was given none: the proxy on the
/// host the page itself was loaded from (`location.hostname`, which keeps the
/// brackets of an IPv6 literal), port [`DEFAULT_STREAM_PORT`]. Falls back to
/// loopback when the page has no host (`file://`).
pub fn default_stream_url(page_hostname: &str) -> String {
    let host = match page_hostname.trim() {
        "" => "127.0.0.1",
        host => host,
    };
    format!("ws://{host}:{DEFAULT_STREAM_PORT}")
}

/// How long to wait before reconnect attempt number `attempt` (0 = the first
/// one after a drop): doubling from [`RECONNECT_BASE_MS`], capped at
/// [`RECONNECT_MAX_MS`].
pub fn reconnect_delay_ms(attempt: u32) -> u32 {
    let doublings = 1u32.checked_shl(attempt).unwrap_or(u32::MAX);
    RECONNECT_BASE_MS
        .saturating_mul(doublings)
        .min(RECONNECT_MAX_MS)
}

/// The connection state shown in the page's status line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Link {
    /// A WebSocket is being opened.
    Connecting,
    /// Connected; no frame has been drawn yet.
    Waiting,
    /// Frames are being drawn.
    Live,
    /// The connection is down; the next attempt is in `delay_ms`.
    Retrying { delay_ms: u32 },
    /// The address cannot be opened at all (not retried).
    Failed,
}

/// The text of the status line for `link` (empty while frames are flowing, so
/// nothing covers the picture).
pub fn status_line(link: Link, url: &str) -> String {
    match link {
        Link::Connecting => format!("connecting to {url} ..."),
        Link::Waiting => format!("connected to {url}, waiting for the first frame"),
        Link::Live => String::new(),
        Link::Retrying { delay_ms } => format!(
            "no connection to {url}; retrying in {:.1} s",
            f64::from(delay_ms) / 1000.0
        ),
        Link::Failed => format!("cannot open {url} (expected a ws://host:port address)"),
    }
}

/// What to do when a decode finishes: see [`FrameSlot::settle`].
#[derive(Debug, PartialEq, Eq)]
pub struct Settled<T> {
    /// Draw the image that was just decoded.
    pub draw: bool,
    /// Start decoding this frame next.
    pub next: Option<T>,
}

/// Latest-frame-wins bookkeeping for image decoding.
///
/// Decoding a JPEG is asynchronous. At most **one** decode is in flight; a
/// frame that arrives meanwhile waits in a single slot and replaces whatever
/// was waiting there, so a viewer that decodes slower than frames arrive skips
/// frames instead of piling up decodes (and their memory) without bound.
///
/// `T` is whatever is needed to start a decode (the JPEG bytes).
#[derive(Debug)]
pub struct FrameSlot<T> {
    /// Sequence number of the newest frame accepted on this connection.
    newest: u64,
    /// A decode is in flight.
    decoding: bool,
    /// The decode in flight belongs to a connection that has since ended.
    stale: bool,
    /// The newest frame that arrived while a decode was in flight.
    waiting: Option<T>,
}

impl<T> Default for FrameSlot<T> {
    fn default() -> Self {
        Self {
            newest: 0,
            decoding: false,
            stale: false,
            waiting: None,
        }
    }
}

impl<T> FrameSlot<T> {
    /// A frame with sequence number `seq` arrived. Returns the frame back if
    /// its decode should start now; otherwise it was kept as the waiting frame
    /// (replacing an older one) or, if it is not newer than what was already
    /// accepted, dropped.
    pub fn offer(&mut self, seq: u64, frame: T) -> Option<T> {
        if seq <= self.newest {
            return None;
        }
        self.newest = seq;
        if self.decoding {
            self.waiting = Some(frame);
            None
        } else {
            self.decoding = true;
            Some(frame)
        }
    }

    /// The decode in flight finished (successfully or not). Says whether its
    /// image should be drawn and which frame to decode next, if one waited.
    pub fn settle(&mut self) -> Settled<T> {
        let draw = self.decoding && !self.stale;
        self.stale = false;
        let next = self.waiting.take();
        self.decoding = next.is_some();
        Settled { draw, next }
    }

    /// The connection ended: forget the waiting frame, start sequence numbers
    /// over (a restarted proxy counts from 1 again) and make sure a decode
    /// still in flight is not drawn over the next connection's frames.
    pub fn reset(&mut self) {
        self.newest = 0;
        self.waiting = None;
        self.stale = self.decoding;
    }

    /// Sequence number of the newest frame accepted since the last reset.
    pub fn newest(&self) -> u64 {
        self.newest
    }
}

pub mod web {
    //! Browser glue (WebSocket, canvas, DOM events). Only meaningful in a browser.

    use std::cell::{Cell, RefCell};
    use std::collections::BTreeSet;
    use std::rc::Rc;

    use space_stream::protocol::{decode_frame, FRAME_HEADER_LEN};
    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;
    use web_sys::{
        BinaryType, Blob, BlobPropertyBag, CanvasRenderingContext2d, Element, HtmlCanvasElement,
        ImageBitmap, KeyboardEvent, MessageEvent, MouseEvent, WebSocket, Window,
    };

    use super::{
        contained_rect, default_stream_url, key_message, pointer_message, press_messages,
        reconnect_delay_ms, release_messages, status_line, turn_message, FrameSlot, Link,
    };

    /// Id of the (optional) element whose text shows the connection status.
    const STATUS_ELEMENT_ID: &str = "status";

    /// Everything the event handlers share. It lives as long as the page: the
    /// handlers hold it and it holds the handlers.
    struct Viewer {
        window: Window,
        canvas: HtmlCanvasElement,
        ctx: CanvasRenderingContext2d,
        status: Option<Element>,
        url: String,
        /// The current connection (`None` between a close and the next attempt).
        socket: RefCell<Option<WebSocket>>,
        /// Frames waiting to be decoded; the payload is the JPEG as a view into
        /// the WebSocket message's `ArrayBuffer`.
        frames: RefCell<FrameSlot<js_sys::Uint8Array>>,
        /// Reconnect attempts since the last successful open.
        attempt: Cell<u32>,
        /// A frame of the current connection has been drawn.
        live: Cell<bool>,
        /// DOM codes of the mapped keys currently held down.
        held: RefCell<BTreeSet<String>>,
        handlers: RefCell<Option<Handlers>>,
    }

    /// The callbacks that are attached again and again (to every new socket,
    /// every decode, every retry timer). They are created once and reused;
    /// nothing is allocated or leaked per frame.
    struct Handlers {
        open: Closure<dyn FnMut()>,
        message: Closure<dyn FnMut(MessageEvent)>,
        close: Closure<dyn FnMut()>,
        decoded: Closure<dyn FnMut(JsValue)>,
        decode_failed: Closure<dyn FnMut(JsValue)>,
        retry: Closure<dyn FnMut()>,
    }

    impl Viewer {
        fn set_status(&self, link: Link) {
            if let Some(status) = &self.status {
                status.set_text_content(Some(&status_line(link, &self.url)));
            }
        }

        /// Send one input message if the connection is open (input while
        /// disconnected is dropped; held keys are re-sent on reconnect).
        fn send(&self, msg: &[u8]) {
            if let Some(socket) = self.socket.borrow().as_ref() {
                if socket.ready_state() == WebSocket::OPEN {
                    let _ = socket.send_with_u8_array(msg);
                }
            }
        }

        /// Open a new connection. An `Err` means the address itself was
        /// refused by the browser (malformed, or blocked), which a retry would
        /// not fix.
        fn connect(&self) -> Result<(), JsValue> {
            self.set_status(Link::Connecting);
            let socket = match WebSocket::new(&self.url) {
                Ok(socket) => socket,
                Err(error) => {
                    self.set_status(Link::Failed);
                    return Err(error);
                }
            };
            socket.set_binary_type(BinaryType::Arraybuffer);
            if let Some(handlers) = self.handlers.borrow().as_ref() {
                socket.set_onopen(Some(handlers.open.as_ref().unchecked_ref()));
                socket.set_onmessage(Some(handlers.message.as_ref().unchecked_ref()));
                // A failed connection attempt also ends in a close event, so
                // this one handler covers "never connected" and "dropped".
                socket.set_onclose(Some(handlers.close.as_ref().unchecked_ref()));
            }
            *self.socket.borrow_mut() = Some(socket);
            Ok(())
        }

        fn on_open(&self) {
            self.attempt.set(0);
            self.set_status(Link::Waiting);
            // The new connection knows nothing about keys that were already
            // down, and the browser reports a held key only once.
            for msg in press_messages(self.held.borrow().iter()) {
                self.send(&msg);
            }
        }

        fn on_close(&self) {
            self.socket.borrow_mut().take();
            self.frames.borrow_mut().reset();
            self.live.set(false);
            let attempt = self.attempt.get();
            self.attempt.set(attempt.saturating_add(1));
            let delay_ms = reconnect_delay_ms(attempt);
            self.set_status(Link::Retrying { delay_ms });
            if let Some(handlers) = self.handlers.borrow().as_ref() {
                let _ = self
                    .window
                    .set_timeout_with_callback_and_timeout_and_arguments_0(
                        handlers.retry.as_ref().unchecked_ref(),
                        i32::try_from(delay_ms).unwrap_or(i32::MAX),
                    );
            }
        }

        /// A frame arrived: read its header and queue the JPEG for decoding.
        fn on_message(&self, event: &MessageEvent) {
            let Ok(buffer) = event.data().dyn_into::<js_sys::ArrayBuffer>() else {
                return;
            };
            let header_len = FRAME_HEADER_LEN as u32;
            if buffer.byte_length() < header_len {
                return;
            }
            // Only the header is copied into wasm memory. The JPEG stays where
            // it is (a view into the message's buffer) until it is handed to
            // the decoder, and is never copied at all if it gets skipped.
            let mut head = [0u8; FRAME_HEADER_LEN];
            js_sys::Uint8Array::new_with_byte_offset_and_length(&buffer, 0, header_len)
                .copy_to(&mut head);
            let Ok((header, _)) = decode_frame(&head) else {
                return;
            };
            let jpeg = js_sys::Uint8Array::new_with_byte_offset(&buffer, header_len);
            let start = self.frames.borrow_mut().offer(header.seq, jpeg);
            if let Some(jpeg) = start {
                self.decode(jpeg);
            }
        }

        /// Start decoding `jpeg` (the frame slot already counts it as in
        /// flight). If that is impossible, move on to the frame that waited.
        fn decode(&self, jpeg: js_sys::Uint8Array) {
            let mut next = Some(jpeg);
            while let Some(jpeg) = next {
                if self.begin_decode(&jpeg) {
                    return;
                }
                next = self.frames.borrow_mut().settle().next;
            }
        }

        /// Hand `jpeg` to the browser's image decoder. True if a decode is now
        /// in flight (one of the `decoded` / `decode_failed` handlers will run).
        fn begin_decode(&self, jpeg: &js_sys::Uint8Array) -> bool {
            let handlers = self.handlers.borrow();
            let Some(handlers) = handlers.as_ref() else {
                return false;
            };
            let options = BlobPropertyBag::new();
            options.set_type("image/jpeg");
            let parts = js_sys::Array::of1(jpeg);
            let Ok(blob) = Blob::new_with_u8_array_sequence_and_options(&parts, &options) else {
                return false;
            };
            let Ok(promise) = self.window.create_image_bitmap_with_blob(&blob) else {
                return false;
            };
            let _ = promise.then2(&handlers.decoded, &handlers.decode_failed);
            true
        }

        /// The decode in flight finished; `bitmap` is `None` if it failed.
        fn on_decoded(&self, bitmap: Option<ImageBitmap>) {
            let settled = self.frames.borrow_mut().settle();
            if let Some(bitmap) = bitmap {
                if settled.draw {
                    self.draw(&bitmap);
                }
                bitmap.close();
            }
            if let Some(jpeg) = settled.next {
                self.decode(jpeg);
            }
        }

        /// Draw a decoded frame. The canvas takes the frame's own pixel size
        /// (CSS scales it to the window with the aspect ratio kept), so a
        /// stream that is not 4:3 is not stretched.
        fn draw(&self, bitmap: &ImageBitmap) {
            let (width, height) = (bitmap.width(), bitmap.height());
            if width == 0 || height == 0 {
                return;
            }
            if self.canvas.width() != width || self.canvas.height() != height {
                self.canvas.set_width(width);
                self.canvas.set_height(height);
            }
            let _ = self.ctx.draw_image_with_image_bitmap(bitmap, 0.0, 0.0);
            if !self.live.replace(true) {
                self.set_status(Link::Live);
            }
        }

        fn on_key(&self, event: &KeyboardEvent, down: bool) {
            let code = event.code();
            let Some(msg) = key_message(&code, down) else {
                return;
            };
            // Also for auto-repeat: a held Space or arrow key must not scroll.
            event.prevent_default();
            if event.repeat() {
                return;
            }
            if down {
                self.held.borrow_mut().insert(code);
            } else {
                self.held.borrow_mut().remove(&code);
            }
            self.send(&msg);
        }

        /// The window lost focus: the browser will not deliver the key-ups.
        fn on_blur(&self) {
            let held = std::mem::take(&mut *self.held.borrow_mut());
            for msg in release_messages(held.iter()) {
                self.send(&msg);
            }
        }

        /// Pointer position always; dragging with the left button also turns
        /// the view.
        fn on_mouse_move(&self, event: &MouseEvent) {
            let buttons = (event.buttons() & 0xff) as u8;
            let element = self.canvas.get_bounding_client_rect();
            let picture = contained_rect(
                element.width(),
                element.height(),
                f64::from(self.canvas.width()),
                f64::from(self.canvas.height()),
            );
            if let Some((left, top, width, height)) = picture {
                let x = f64::from(event.client_x()) - element.left() - left;
                let y = f64::from(event.client_y()) - element.top() - top;
                if let Some(msg) = pointer_message(x, y, width, height, buttons) {
                    self.send(&msg);
                }
            }
            if buttons & 1 != 0 {
                let (dx, dy) = (event.movement_x(), event.movement_y());
                if let Some(msg) = turn_message(f64::from(dx), f64::from(dy)) {
                    self.send(&msg);
                }
            }
        }
    }

    /// The stream address for a page loaded from `page_hostname` that was not
    /// given one explicitly (see [`default_stream_url`]).
    #[wasm_bindgen]
    pub fn default_ws_url(page_hostname: &str) -> String {
        default_stream_url(page_hostname)
    }

    /// Connect the canvas with id `canvas_id` to the stream at `ws_url`
    /// (e.g. `ws://127.0.0.1:8099`) and keep it connected. Call once from JS
    /// after the wasm loads. If the page has an element with id `status`, the
    /// connection state is written into it.
    #[wasm_bindgen]
    pub fn start(canvas_id: &str, ws_url: &str) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or("no window")?;
        let document = window.document().ok_or("no document")?;
        let canvas: HtmlCanvasElement = document
            .get_element_by_id(canvas_id)
            .ok_or("canvas not found")?
            .dyn_into()?;
        let ctx: CanvasRenderingContext2d = canvas
            .get_context("2d")?
            .ok_or("no 2d context")?
            .dyn_into()?;

        let viewer = Rc::new(Viewer {
            window: window.clone(),
            canvas: canvas.clone(),
            ctx,
            status: document.get_element_by_id(STATUS_ELEMENT_ID),
            url: ws_url.to_owned(),
            socket: RefCell::new(None),
            frames: RefCell::new(FrameSlot::default()),
            attempt: Cell::new(0),
            live: Cell::new(false),
            held: RefCell::new(BTreeSet::new()),
            handlers: RefCell::new(None),
        });

        let handlers = {
            let v = &viewer;
            Handlers {
                open: Closure::<dyn FnMut()>::new({
                    let v = Rc::clone(v);
                    move || v.on_open()
                }),
                message: Closure::<dyn FnMut(MessageEvent)>::new({
                    let v = Rc::clone(v);
                    move |event: MessageEvent| v.on_message(&event)
                }),
                close: Closure::<dyn FnMut()>::new({
                    let v = Rc::clone(v);
                    move || v.on_close()
                }),
                decoded: Closure::<dyn FnMut(JsValue)>::new({
                    let v = Rc::clone(v);
                    move |bitmap: JsValue| v.on_decoded(bitmap.dyn_into::<ImageBitmap>().ok())
                }),
                decode_failed: Closure::<dyn FnMut(JsValue)>::new({
                    let v = Rc::clone(v);
                    move |_error: JsValue| v.on_decoded(None)
                }),
                retry: Closure::<dyn FnMut()>::new({
                    let v = Rc::clone(v);
                    move || {
                        // A refused address is shown in the status line.
                        let _ = v.connect();
                    }
                }),
            }
        };
        *viewer.handlers.borrow_mut() = Some(handlers);

        // Page-lifetime DOM listeners: registered once, never removed.
        for (event, down) in [("keydown", true), ("keyup", false)] {
            let v = Rc::clone(&viewer);
            let on_key = Closure::<dyn FnMut(KeyboardEvent)>::new(move |e: KeyboardEvent| {
                v.on_key(&e, down)
            });
            window.add_event_listener_with_callback(event, on_key.as_ref().unchecked_ref())?;
            on_key.forget();
        }
        {
            let v = Rc::clone(&viewer);
            let on_blur = Closure::<dyn FnMut()>::new(move || v.on_blur());
            window.add_event_listener_with_callback("blur", on_blur.as_ref().unchecked_ref())?;
            on_blur.forget();
        }
        {
            let v = Rc::clone(&viewer);
            let on_move =
                Closure::<dyn FnMut(MouseEvent)>::new(move |e: MouseEvent| v.on_mouse_move(&e));
            canvas
                .add_event_listener_with_callback("mousemove", on_move.as_ref().unchecked_ref())?;
            on_move.forget();
        }

        viewer.connect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_messages_use_the_shared_wire_format() {
        // Same vector as space-stream's `viewer_input_wire_bytes_are_stable`.
        assert_eq!(key_message("KeyA", true), Some(vec![2, 3, 1]));
        assert_eq!(key_message("ArrowLeft", false), Some(vec![2, 3, 0]));
        assert_eq!(key_message("F13", true), None);
    }

    #[test]
    fn blur_releases_every_held_key() {
        let held = ["KeyA".to_string(), "Space".to_string()];
        let msgs = release_messages(&held);
        assert_eq!(msgs.len(), 2);
        for m in msgs {
            assert!(matches!(
                ViewerInput::decode(&m),
                Ok(ViewerInput::Key { down: false, .. })
            ));
        }
        assert!(release_messages(&[]).is_empty());
    }

    /// After a reconnect the keys still physically held are pressed again on
    /// the new connection (unmapped codes are skipped).
    #[test]
    fn reconnect_presses_every_held_key_again() {
        let held = ["KeyW".to_string(), "F13".to_string(), "Space".to_string()];
        let msgs: Vec<_> = press_messages(&held)
            .iter()
            .map(|m| ViewerInput::decode(m).unwrap())
            .collect();
        assert_eq!(
            msgs,
            [
                ViewerInput::Key {
                    key: Key::Forward,
                    down: true
                },
                ViewerInput::Key {
                    key: Key::Up,
                    down: true
                },
            ]
        );
        assert!(press_messages(&[]).is_empty());
    }

    #[test]
    fn viewer_messages_decode_with_the_server_decoder() {
        let msg = key_message("Space", true).unwrap();
        assert_eq!(
            ViewerInput::decode(&msg),
            Ok(ViewerInput::Key {
                key: Key::Up,
                down: true
            })
        );
        // A drag of 400 px to the left is +1 rad of yaw: the same bytes as
        // space-stream's `viewer_input_wire_bytes_are_stable` Turn vector.
        let msg = turn_message(-400.0, 0.0).unwrap();
        assert_eq!(msg, [3, 0x3f, 0x80, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn pointer_is_normalised_and_clamped() {
        let msg = pointer_message(160.0, 120.0, 640.0, 480.0, 1).unwrap();
        assert_eq!(
            ViewerInput::decode(&msg),
            Ok(ViewerInput::Pointer {
                x: 0.25,
                y: 0.25,
                buttons: 1
            })
        );
        let msg = pointer_message(-5.0, 9999.0, 640.0, 480.0, 0).unwrap();
        assert_eq!(
            ViewerInput::decode(&msg),
            Ok(ViewerInput::Pointer {
                x: 0.0,
                y: 1.0,
                buttons: 0
            })
        );
        assert_eq!(pointer_message(1.0, 1.0, 0.0, 480.0, 0), None);
        assert_eq!(pointer_message(f64::NAN, 1.0, 10.0, 10.0, 0), None);
    }

    /// The `(dyaw, dpitch)` a drag of `(dx, dy)` pixels puts on the wire.
    fn turn(dx: f64, dy: f64) -> (f32, f32) {
        let msg = turn_message(dx, dy).expect("a turn");
        match ViewerInput::decode(&msg) {
            Ok(ViewerInput::Turn { dyaw, dpitch }) => (dyaw, dpitch),
            other => panic!("not a Turn: {other:?}"),
        }
    }

    fn assert_turn(dx: f64, dy: f64, want: (f32, f32)) {
        let got = turn(dx, dy);
        let close = |g: f32, w: f32| (g - w).abs() <= 1e-6 * w.abs().max(1.0);
        assert!(
            close(got.0, want.0) && close(got.1, want.1),
            "drag ({dx}, {dy}): {got:?} != {want:?}"
        );
    }

    /// A drag is sent as a rotation *delta*: it depends only on how far the
    /// mouse moved, never on where earlier drags left the view. (An absolute
    /// angle made the first drag snap the camera to the viewer's own zero.)
    #[test]
    fn a_drag_is_a_relative_turn() {
        // Right turns right (negative yaw), down looks down (negative pitch).
        assert_turn(100.0, 0.0, (-0.25, 0.0));
        assert_turn(0.0, 40.0, (0.0, -0.1));
        assert_turn(-8.0, -4.0, (0.02, 0.01));
        // The same drag gives the same message every time: no state is kept,
        // and nothing is clamped or wrapped here (the proxy owns the limits).
        assert_eq!(turn_message(100.0, 0.0), turn_message(100.0, 0.0));
        let mut total = 0.0;
        for _ in 0..1000 {
            total += turn(4.0, 0.0).0;
        }
        assert!((total + 10.0).abs() < 1e-3, "1000 drags of 4 px: {total}");
        assert_turn(1.0e6, -1.0e6, (-2500.0, 2500.0));
        // An axis that did not move is a plain zero on the wire, not `-0.0`.
        assert_eq!(turn(3.0, 0.0).1.to_bits(), 0);
        assert_eq!(turn(0.0, 3.0).0.to_bits(), 0);
        // Nothing to say.
        assert_eq!(turn_message(0.0, 0.0), None);
        assert_eq!(turn_message(f64::NAN, 1.0), None);
        assert_eq!(turn_message(f64::INFINITY, 0.0), None);
        assert_eq!(turn_message(1.0e300, 0.0), None, "overflows f32");
    }

    fn assert_rect(got: Option<(f64, f64, f64, f64)>, want: (f64, f64, f64, f64)) {
        let got = got.expect("a rectangle");
        for (g, w) in [got.0, got.1, got.2, got.3]
            .into_iter()
            .zip([want.0, want.1, want.2, want.3])
        {
            assert!((g - w).abs() < 1e-9, "{got:?} != {want:?}");
        }
    }

    /// The picture keeps its aspect ratio inside the window-sized canvas
    /// element, whatever shape the window has.
    #[test]
    fn the_picture_is_fitted_into_the_element_with_its_aspect_kept() {
        // Same shape: fills the element.
        assert_rect(
            contained_rect(1280.0, 960.0, 640.0, 480.0),
            (0.0, 0.0, 1280.0, 960.0),
        );
        // Wide window, 16:9 picture limited by the height: bars left and right.
        assert_rect(
            contained_rect(2000.0, 720.0, 640.0, 360.0),
            (360.0, 0.0, 1280.0, 720.0),
        );
        // Tall window: bars above and below.
        assert_rect(
            contained_rect(400.0, 800.0, 640.0, 480.0),
            (0.0, 250.0, 400.0, 300.0),
        );
        // Smaller than the picture: scaled down, still centred.
        assert_rect(
            contained_rect(320.0, 320.0, 640.0, 480.0),
            (0.0, 40.0, 320.0, 240.0),
        );
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(contained_rect(bad, 100.0, 640.0, 480.0), None);
            assert_eq!(contained_rect(100.0, 100.0, 640.0, bad), None);
        }
    }

    /// Pointer positions are measured against the picture, not the element:
    /// the picture's corners are (0,0) and (1,1) even with letterbox bars.
    #[test]
    fn pointer_positions_are_relative_to_the_fitted_picture() {
        let (left, top, width, height) = contained_rect(2000.0, 720.0, 640.0, 360.0).unwrap();
        let at = |x: f64, y: f64| match ViewerInput::decode(&pointer_message(
            x - left,
            y - top,
            width,
            height,
            0,
        )?) {
            Ok(ViewerInput::Pointer { x, y, .. }) => Some((x, y)),
            other => panic!("not a Pointer: {other:?}"),
        };
        assert_eq!(at(360.0, 0.0), Some((0.0, 0.0)));
        assert_eq!(at(1640.0, 720.0), Some((1.0, 1.0)));
        assert_eq!(at(1000.0, 360.0), Some((0.5, 0.5)));
        // In the bars: clamped to the picture's edge.
        assert_eq!(at(0.0, 180.0), Some((0.0, 0.25)));
        assert_eq!(at(1999.0, 540.0), Some((1.0, 0.75)));
    }

    #[test]
    fn the_default_stream_is_on_the_host_that_served_the_page() {
        assert_eq!(default_stream_url("example.org"), "ws://example.org:8099");
        assert_eq!(default_stream_url("192.168.1.20"), "ws://192.168.1.20:8099");
        // `location.hostname` keeps the brackets of an IPv6 literal.
        assert_eq!(default_stream_url("[::1]"), "ws://[::1]:8099");
        // No host (file://): the old loopback default.
        assert_eq!(default_stream_url(""), "ws://127.0.0.1:8099");
        assert_eq!(default_stream_url("  "), "ws://127.0.0.1:8099");
    }

    #[test]
    fn reconnect_backs_off_and_is_capped() {
        let delays: Vec<u32> = (0..7).map(reconnect_delay_ms).collect();
        assert_eq!(delays, [500, 1000, 2000, 4000, 8000, 8000, 8000]);
        for attempt in [8, 22, 23, 31, 32, 33, 1000, u32::MAX] {
            assert_eq!(reconnect_delay_ms(attempt), RECONNECT_MAX_MS, "{attempt}");
        }
    }

    #[test]
    fn the_status_line_names_the_address_and_clears_when_live() {
        let url = "ws://host:8099";
        assert_eq!(status_line(Link::Live, url), "", "nothing over the picture");
        for link in [
            Link::Connecting,
            Link::Waiting,
            Link::Retrying { delay_ms: 500 },
            Link::Failed,
        ] {
            let text = status_line(link, url);
            assert!(text.contains(url), "{link:?}: {text}");
        }
        assert!(status_line(Link::Retrying { delay_ms: 500 }, url).contains("0.5 s"));
        assert!(status_line(Link::Retrying { delay_ms: 8000 }, url).contains("8.0 s"));
    }

    fn settled<T>(draw: bool, next: Option<T>) -> Settled<T> {
        Settled { draw, next }
    }

    /// One decode at a time; of the frames that arrive meanwhile only the
    /// newest is kept.
    #[test]
    fn frames_arriving_during_a_decode_collapse_to_the_newest() {
        let mut slot = FrameSlot::default();
        assert_eq!(slot.offer(1, "f1"), Some("f1"), "idle: decode at once");
        assert_eq!(slot.offer(2, "f2"), None, "decode in flight: wait");
        assert_eq!(slot.offer(3, "f3"), None);
        assert_eq!(slot.offer(4, "f4"), None, "f2 and f3 are skipped");
        assert_eq!(slot.newest(), 4);

        assert_eq!(
            slot.settle(),
            settled(true, Some("f4")),
            "draw f1, decode f4"
        );
        assert_eq!(slot.offer(5, "f5"), None, "f4 is in flight now");
        assert_eq!(slot.settle(), settled(true, Some("f5")));
        assert_eq!(slot.settle(), settled(true, None));

        // Idle again: the next frame decodes at once.
        assert_eq!(slot.offer(6, "f6"), Some("f6"));
        assert_eq!(slot.settle(), settled(true, None));
        // A settle with nothing in flight draws nothing.
        assert_eq!(slot.settle(), settled(false, None));
    }

    #[test]
    fn frames_that_are_not_newer_are_dropped() {
        let mut slot = FrameSlot::default();
        assert_eq!(slot.offer(0, "zero"), None, "sequence numbers start at 1");
        assert_eq!(slot.offer(7, "f7"), Some("f7"));
        assert_eq!(slot.offer(7, "again"), None);
        assert_eq!(slot.offer(3, "old"), None);
        assert_eq!(slot.settle(), settled(true, None), "nothing was queued");
    }

    /// A reconnect starts sequence numbers over (a restarted proxy counts from
    /// 1 again; without the reset every new frame would look old and the
    /// picture would stay frozen), and a decode still in flight from the old
    /// connection is not drawn over the new one.
    #[test]
    fn a_reconnect_resets_the_sequence_and_discards_the_old_decode() {
        let mut slot = FrameSlot::default();
        assert_eq!(slot.offer(900, "old-900"), Some("old-900"));
        assert_eq!(slot.offer(901, "old-901"), None);

        slot.reset();
        assert_eq!(slot.newest(), 0);
        // The new connection's first frame is accepted although 1 < 901. The
        // old decode is still in flight, so it waits (still one at a time).
        assert_eq!(slot.offer(1, "new-1"), None);
        assert_eq!(slot.offer(2, "new-2"), None);
        // The old decode finishes: not drawn, the old waiting frame is gone,
        // the newest new frame is next.
        assert_eq!(slot.settle(), settled(false, Some("new-2")));
        assert_eq!(slot.settle(), settled(true, None));

        // A reset while idle leaves nothing to discard.
        slot.reset();
        assert_eq!(slot.offer(1, "again-1"), Some("again-1"));
        assert_eq!(slot.settle(), settled(true, None));
    }
}
