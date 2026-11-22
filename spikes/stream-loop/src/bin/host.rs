//! Phase-0 spike 0.4 host: offscreen Vulkan render -> pure-Rust JPEG -> WebSocket.
//!
//! Renders a moving coloured quad offscreen with ash (no window/surface), reads
//! the frame back, JPEG-encodes it with the pure-Rust `jpeg-encoder` crate, and
//! pushes frames over a binary WebSocket on 127.0.0.1:8099.
//!
//! Wire protocol (binary messages):
//!   [8 bytes: frame_id u64 big-endian][JPEG bytes ...]
//!
//! Input messages (JSON text from the browser):
//!   {"type":"key","code":"ArrowUp","down":true,"seq":N}
//! ArrowUp down => quad advances faster; up => slow. `seq` is tracked and logged.
//!
//! Backpressure: latest-frame-wins. At most one frame is queued for the socket
//! (a `tokio::sync::watch` slot that a slow socket simply overwrites); the
//! overwritten frames are dropped and logged.
//!
//! This is a THROWAWAY spike: it proves the transport loop, not render quality.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[path = "../renderer.rs"]
mod renderer;

use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use tokio::sync::watch;
use tokio_tungstenite::tungstenite::Message;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const ADDR: &str = "127.0.0.1:8099";
const TARGET_FRAME_MS: u64 = 33; // ~30 fps cap

// ---------------------------------------------------------------------------
// Shared state between the (blocking) render thread and the async WS server.
// ---------------------------------------------------------------------------

/// A produced frame: header + JPEG, plus bookkeeping for logging/latency.
struct Frame {
    id: u64,
    bytes: Vec<u8>,
    /// Host wall-clock (unix ms) when the frame was produced. Same machine as
    /// the browser, so smoke can subtract this from the browser arrival time.
    produced_ms: u64,
}

/// Latest-frame-wins single-slot queue, backed by `tokio::sync::watch`.
///
/// `watch` keeps the newest value *in the channel*, so a consumer that is busy
/// sending simply finds the newer frame when it next calls `changed()` — no
/// wakeup can be lost. A `Notify` + `Mutex<Option<_>>` pair cannot express
/// this: `notify_one`/`notify_waiters` store no permit, so a notification that
/// arrives before the consumer awaits is lost, and a `take()` that awaits
/// before checking the slot waits for a notification that already happened
/// (a lost wakeup: the socket stalls until the *next* frame, and with a
/// single slow-but-alive client it can stall forever).
struct Latest {
    tx: watch::Sender<Option<Arc<Frame>>>,
}

impl Latest {
    fn new() -> (Self, watch::Receiver<Option<Arc<Frame>>>) {
        let (tx, rx) = watch::channel(None);
        (Self { tx }, rx)
    }

    /// Replace the queued frame with `frame`. A socket that is behind simply
    /// finds the newer frame in the slot; the frames it skipped are logged by
    /// the socket task (see `handle_socket`).
    fn push(&self, frame: Arc<Frame>) {
        self.tx.send_replace(Some(frame));
    }
}

/// Input state fed by the browser, read by the render loop.
#[derive(Default)]
struct Input {
    /// Current advance speed (quad position delta per frame).
    speed: Mutex<f32>,
    /// Highest input sequence number applied so far.
    last_seq: AtomicU64,
    /// Monotonic count of applied input messages (used by smoke to detect change).
    applied: AtomicU64,
}

const SPEED_SLOW: f32 = 0.004;
const SPEED_FAST: f32 = 0.05;

#[derive(Deserialize)]
struct KeyMsg {
    #[serde(rename = "type")]
    kind: String,
    code: String,
    down: bool,
    seq: u64,
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

// ---------------------------------------------------------------------------
// JPEG encode (pure Rust).
// ---------------------------------------------------------------------------
fn encode_jpeg(rgba: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    {
        let enc = jpeg_encoder::Encoder::new(&mut out, 70);
        enc
            .encode(rgba, WIDTH as u16, HEIGHT as u16, jpeg_encoder::ColorType::Rgba)
            .expect("jpeg encode");
    }
    out
}

// ---------------------------------------------------------------------------
// Render thread: renders frames, encodes, pushes to the latest-frame queue.
// ---------------------------------------------------------------------------
fn render_thread(latest: Arc<Latest>, input: Arc<Input>, shutdown: Arc<AtomicU64>) {
    let r = renderer::Renderer::new();
    let mut offset: f32 = 0.0;
    let mut frame_id: u64 = 0;
    loop {
        if shutdown.load(Ordering::Relaxed) != 0 {
            break;
        }
        let t0 = Instant::now();
        let speed = *input.speed.lock().unwrap();
        offset = (offset + speed) % 1.0;
        let rgba = r.render(offset);
        let render_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let jpeg = encode_jpeg(&rgba);
        let encode_ms = t1.elapsed().as_secs_f64() * 1000.0;

        frame_id += 1;
        let seq = input.last_seq.load(Ordering::Relaxed);
        eprintln!(
            "[host] frame_id={frame_id} render_ms={render_ms:.2} encode_ms={encode_ms:.2} last_seq={seq} jpeg_bytes={}",
            jpeg.len()
        );
        latest.push(Arc::new(Frame {
            id: frame_id,
            bytes: jpeg,
            produced_ms: unix_ms(),
        }));

        let dt = t0.elapsed().as_millis() as u64;
        if dt < TARGET_FRAME_MS {
            thread::sleep(Duration::from_millis(TARGET_FRAME_MS - dt));
        }
    }
}

// ---------------------------------------------------------------------------
// WebSocket server: one client at a time; pushes latest frames, reads input.
// ---------------------------------------------------------------------------
async fn handle_socket(
    socket: tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
    mut latest: watch::Receiver<Option<Arc<Frame>>>,
    input: Arc<Input>,
) {
    let (mut write, mut read) = socket.split();
    eprintln!("[host] browser connected");
    // Highest frame id already handed to the socket; used to log the frames a
    // slow socket skipped over (latest-frame-wins drops).
    let mut last_sent: u64 = 0;
    loop {
        // `changed()` returns as soon as a *newer* frame is in the slot, and
        // the slot itself holds it, so no wakeup can be lost.
        let next = latest.changed();
        tokio::select! {
            changed = next => {
                if changed.is_err() {
                    break; // render thread gone
                }
                // `borrow_and_update` marks the value seen so the next
                // `changed()` waits for a frame after this one, and clones it
                // out so the lock is not held across `await`.
                let f = match latest.borrow_and_update().clone() {
                    Some(f) => f,
                    None => continue,
                };
                if f.id > last_sent + 1 {
                    eprintln!(
                        "[host] backpressure: dropped frame_id(s) {}..{} (socket behind)",
                        last_sent + 1,
                        f.id - 1
                    );
                }
                last_sent = f.id;
                // Meta text message carries the host wall-clock timestamp so the
                // browser (same machine) can measure frame-to-browser latency.
                // The binary frame protocol itself is unchanged (8-byte id + JPEG).
                let meta = format!("{{\"type\":\"meta\",\"id\":{},\"t\":{}}}", f.id, f.produced_ms);
                if write.send(Message::Text(meta.into())).await.is_err() {
                    eprintln!("[host] send failed; dropping client");
                    break;
                }
                let mut msg = Vec::with_capacity(8 + f.bytes.len());
                msg.extend_from_slice(&f.id.to_be_bytes());
                msg.extend_from_slice(&f.bytes);
                if write.send(Message::Binary(msg.into())).await.is_err() {
                    eprintln!("[host] send failed; dropping client");
                    break;
                }
            }
            msg = read.next() => {
                match msg {
                    Some(Ok(Message::Text(t))) => apply_input(&input, &t),
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Err(e)) => { eprintln!("[host] ws recv error: {e}"); break; }
                    _ => {}
                }
            }
        }
    }
    eprintln!("[host] browser disconnected");
}

fn apply_input(input: &Input, text: &str) {
    let msg: KeyMsg = match serde_json::from_str(text) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[host] bad input json: {e} ({text})");
            return;
        }
    };
    if msg.kind != "key" {
        return;
    }
    // ArrowUp down => fast; up => slow. Other keys ignored but still counted.
    if msg.code == "ArrowUp" {
        let mut s = input.speed.lock().unwrap();
        *s = if msg.down { SPEED_FAST } else { SPEED_SLOW };
    }
    input.last_seq.fetch_max(msg.seq, Ordering::Relaxed);
    input.applied.fetch_add(1, Ordering::Relaxed);
    eprintln!(
        "[host] input applied: code={} down={} seq={} speed={}",
        msg.code,
        msg.down,
        msg.seq,
        *input.speed.lock().unwrap()
    );
}

#[tokio::main]
async fn main() {
    let (latest, rx) = Latest::new();
    let latest = Arc::new(latest);
    let input = Arc::new(Input {
        speed: Mutex::new(SPEED_SLOW),
        ..Default::default()
    });
    let shutdown = Arc::new(AtomicU64::new(0));

    let (l, i, s) = (latest.clone(), input.clone(), shutdown.clone());
    thread::spawn(move || render_thread(l, i, s));

    let listener = tokio::net::TcpListener::bind(ADDR)
        .await
        .expect("bind 127.0.0.1:8099");
    eprintln!("[host] listening on ws://{ADDR} (device offscreen, {WIDTH}x{HEIGHT})");
    loop {
        let (stream, peer) = match listener.accept().await {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[host] accept error: {e}");
                continue;
            }
        };
        eprintln!("[host] incoming connection from {peer}");
        // Frame streaming is latency-critical: never let the kernel coalesce
        // small frame messages (Nagle) behind an unflushed write.
        if let Err(e) = stream.set_nodelay(true) {
            eprintln!("[host] failed to set TCP_NODELAY on {peer}: {e}");
        }
        let (input, rx) = (input.clone(), rx.clone());
        tokio::spawn(async move {
            match tokio_tungstenite::accept_async(stream).await {
                Ok(ws) => handle_socket(ws, rx, input).await,
                Err(e) => eprintln!("[host] ws handshake failed: {e}"),
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Tests for the backpressure queue (regression: lost wakeup).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn frame(id: u64) -> Arc<Frame> {
        Arc::new(Frame {
            id,
            bytes: vec![0u8; 4],
            produced_ms: 0,
        })
    }

    /// Regression for the lost-wakeup bug: frames pushed *before* the consumer
    /// awaits must still be observed. With the old `Notify` +
    /// `Mutex<Option<_>>` pair the notification was stored nowhere, so this
    /// blocked forever (the socket stalled until the next frame).
    #[tokio::test]
    async fn push_before_consumer_waits_is_not_lost() {
        let (latest, mut rx) = Latest::new();
        latest.push(frame(1));
        latest.push(frame(2));

        // No `push` happens while we await: the pending frames must be found.
        tokio::time::timeout(Duration::from_millis(100), rx.changed())
            .await
            .expect("changed() must not block: frames were already pushed")
            .expect("sender is alive");
        let f = rx.borrow_and_update().clone().expect("a frame is queued");
        assert_eq!(f.id, 2, "latest-frame-wins must observe the newest frame");
    }

    /// After a frame has been taken, `changed()` waits for a *newer* one, and
    /// the newest of a burst is what comes back.
    #[tokio::test]
    async fn takes_are_sequential_and_latest_wins() {
        let (latest, mut rx) = Latest::new();
        latest.push(frame(1));
        rx.changed().await.unwrap();
        assert_eq!(rx.borrow_and_update().clone().unwrap().id, 1);

        latest.push(frame(2));
        latest.push(frame(3));
        rx.changed().await.unwrap();
        assert_eq!(rx.borrow_and_update().clone().unwrap().id, 3);
    }
}
