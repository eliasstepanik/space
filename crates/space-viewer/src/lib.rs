//! `space-viewer` — the thin wasm browser viewer for `space-stream`.
//!
//! It owns no game state. [`start`] opens a WebSocket to a `space-client
//! --stream` proxy, draws every JPEG frame it receives on a `<canvas>`
//! (`createImageBitmap` on a `Blob`), and sends mouse/keyboard input back as
//! [`space_stream::ViewerInput`] messages.
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
//! Transport is plain `ws://` (see `space-stream` for why `wss://` is deferred).
//!
//! The pure logic (input serialisation, view-angle maths) is at the top of this
//! file and unit-tested on the host; the browser glue is in [`web`].
#![forbid(unsafe_code)]

use space_stream::protocol::{Key, ViewerInput};

/// Radians of view rotation per pixel of mouse drag.
pub const RADIANS_PER_PIXEL: f32 = 0.0025;
/// Pitch is kept strictly inside ±90° so the view never flips.
pub const MAX_PITCH: f32 = 1.553_343; // 89 degrees

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

/// Wire message for a pointer at CSS-pixel `(x, y)` inside a `width`×`height`
/// canvas rectangle, normalised to `0..=1` (clamped). `None` for an empty rect.
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

/// The viewer's accumulated look direction, driven by mouse-drag deltas.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ViewAngles {
    pub yaw: f32,
    pub pitch: f32,
}

impl ViewAngles {
    /// Apply a drag of `(dx, dy)` pixels (right = turn right, down = look down).
    pub fn drag(&mut self, dx: f64, dy: f64) {
        self.yaw = (self.yaw - dx as f32 * RADIANS_PER_PIXEL).rem_euclid(std::f32::consts::TAU);
        self.pitch = (self.pitch - dy as f32 * RADIANS_PER_PIXEL).clamp(-MAX_PITCH, MAX_PITCH);
    }

    /// Wire message for the current angles.
    pub fn message(&self) -> Vec<u8> {
        ViewerInput::View {
            yaw: self.yaw,
            pitch: self.pitch,
        }
        .encode()
    }
}

pub mod web {
    //! Browser glue (WebSocket, canvas, DOM events). Only meaningful in a browser.

    use std::cell::{Cell, RefCell};
    use std::collections::BTreeSet;
    use std::rc::Rc;

    use space_stream::protocol::decode_frame;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;
    use web_sys::{
        BinaryType, Blob, BlobPropertyBag, CanvasRenderingContext2d, HtmlCanvasElement,
        ImageBitmap, KeyboardEvent, MessageEvent, MouseEvent, WebSocket,
    };

    use super::{key_message, pointer_message, release_messages, ViewAngles};

    fn send(ws: &WebSocket, msg: &[u8]) {
        if ws.ready_state() == WebSocket::OPEN {
            let _ = ws.send_with_u8_array(msg);
        }
    }

    /// Connect the canvas with id `canvas_id` to the stream at `ws_url`
    /// (e.g. `ws://127.0.0.1:8099`). Call once from JS after the wasm loads.
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

        let ws = WebSocket::new(ws_url)?;
        ws.set_binary_type(BinaryType::Arraybuffer);

        // Frames: decode the header, build a JPEG Blob, draw it when decoded.
        // Decoding is async, so drop any frame older than one already drawn.
        let drawn_seq = Rc::new(Cell::new(0u64));
        {
            let (window, canvas, ctx, drawn_seq) =
                (window.clone(), canvas.clone(), ctx, Rc::clone(&drawn_seq));
            let on_message = Closure::<dyn FnMut(MessageEvent)>::new(move |ev: MessageEvent| {
                let Ok(buf) = ev.data().dyn_into::<js_sys::ArrayBuffer>() else {
                    return;
                };
                let bytes = js_sys::Uint8Array::new(&buf).to_vec();
                let Ok((header, jpeg)) = decode_frame(&bytes) else {
                    return;
                };
                let seq = header.seq;
                let parts = js_sys::Array::of1(&js_sys::Uint8Array::from(jpeg));
                let opts = BlobPropertyBag::new();
                opts.set_type("image/jpeg");
                let Ok(blob) = Blob::new_with_u8_array_sequence_and_options(&parts, &opts) else {
                    return;
                };
                let Ok(promise) = window.create_image_bitmap_with_blob(&blob) else {
                    return;
                };
                let (canvas, ctx, drawn_seq) = (canvas.clone(), ctx.clone(), Rc::clone(&drawn_seq));
                let on_bitmap = Closure::once(move |bitmap: JsValue| {
                    let Ok(bitmap) = bitmap.dyn_into::<ImageBitmap>() else {
                        return;
                    };
                    if seq > drawn_seq.get() {
                        drawn_seq.set(seq);
                        let _ = ctx.draw_image_with_image_bitmap_and_dw_and_dh(
                            &bitmap,
                            0.0,
                            0.0,
                            f64::from(canvas.width()),
                            f64::from(canvas.height()),
                        );
                    }
                    bitmap.close();
                });
                let _ = promise.then(&on_bitmap);
            });
            ws.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
            on_message.forget();
        }

        // Keyboard: forward mapped keys, ignoring auto-repeat. The set of keys
        // currently held is tracked so a window blur can release them (the
        // browser delivers no keyup once focus is lost).
        let held: Rc<RefCell<BTreeSet<String>>> = Rc::new(RefCell::new(BTreeSet::new()));
        for (event, down) in [("keydown", true), ("keyup", false)] {
            let (ws, held) = (ws.clone(), Rc::clone(&held));
            let on_key = Closure::<dyn FnMut(KeyboardEvent)>::new(move |ev: KeyboardEvent| {
                if ev.repeat() {
                    return;
                }
                let code = ev.code();
                if let Some(msg) = key_message(&code, down) {
                    ev.prevent_default();
                    if down {
                        held.borrow_mut().insert(code);
                    } else {
                        held.borrow_mut().remove(&code);
                    }
                    send(&ws, &msg);
                }
            });
            window.add_event_listener_with_callback(event, on_key.as_ref().unchecked_ref())?;
            on_key.forget();
        }
        // Best effort: on blur, send key-up for every key still held.
        {
            let (ws, held) = (ws.clone(), Rc::clone(&held));
            let on_blur = Closure::<dyn FnMut()>::new(move || {
                for msg in release_messages(&held.borrow()) {
                    send(&ws, &msg);
                }
                held.borrow_mut().clear();
            });
            window.add_event_listener_with_callback("blur", on_blur.as_ref().unchecked_ref())?;
            on_blur.forget();
        }

        // Mouse: pointer position always; dragging with the left button also
        // turns the view.
        let angles = Rc::new(RefCell::new(ViewAngles::default()));
        {
            let (ws, canvas_c) = (ws.clone(), canvas.clone());
            let on_move = Closure::<dyn FnMut(MouseEvent)>::new(move |ev: MouseEvent| {
                let rect = canvas_c.get_bounding_client_rect();
                let (x, y) = (
                    f64::from(ev.client_x()) - rect.left(),
                    f64::from(ev.client_y()) - rect.top(),
                );
                let buttons = (ev.buttons() & 0xff) as u8;
                if let Some(msg) = pointer_message(x, y, rect.width(), rect.height(), buttons) {
                    send(&ws, &msg);
                }
                if buttons & 1 != 0 {
                    let mut a = angles.borrow_mut();
                    a.drag(f64::from(ev.movement_x()), f64::from(ev.movement_y()));
                    send(&ws, &a.message());
                }
            });
            canvas
                .add_event_listener_with_callback("mousemove", on_move.as_ref().unchecked_ref())?;
            on_move.forget();
        }
        Ok(())
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
        let msg = ViewAngles {
            yaw: 1.0,
            pitch: 0.0,
        }
        .message();
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

    #[test]
    fn drag_turns_the_view_and_clamps_pitch() {
        let mut a = ViewAngles::default();
        a.drag(100.0, 0.0);
        assert!((a.yaw - (std::f32::consts::TAU - 0.25)).abs() < 1e-4);
        a.drag(0.0, -1.0e6);
        assert_eq!(a.pitch, MAX_PITCH);
        a.drag(0.0, 1.0e7);
        assert_eq!(a.pitch, -MAX_PITCH);
        assert!(a.yaw >= 0.0 && a.yaw < std::f32::consts::TAU);
    }
}
