//! The `ws://` stream server and the synchronous [`StreamClient`] handle a
//! native client uses to push frames into it and pull viewer input out of it.
//!
//! # Frames (native → viewers)
//!
//! Backpressure is *latest-frame-wins* (from the Phase-0 `stream-loop` spike): a
//! `tokio::sync::watch` slot holds the newest frame, so a slow viewer skips
//! frames instead of building latency, and no wake-up can be lost (a `Notify` +
//! `Mutex<Option<_>>` pair can lose one). The wire message (header + JPEG) is
//! built once per frame and shared by every viewer.
//!
//! # Input (viewers → native)
//!
//! Each connection reads its viewer's input *concurrently* with sending frames,
//! so a slow frame send never delays input. Every viewer feeds one queue
//! ([`StreamClient::poll_input`]):
//!
//! * **Keys are held per viewer and reference-counted.** The native side gets a
//!   key-down when the first viewer presses a key and a key-up when the last
//!   viewer holding it releases it *or disconnects* — a viewer that vanishes
//!   mid-press cannot leave the key stuck while other viewers stay connected.
//! * **The queue never loses a key transition.** Up to `INPUT_QUEUE` (1024)
//!   events are kept verbatim. Past that (the native side is not polling) events are
//!   merged instead of dropped: a `Pointer` replaces the queued one, a `Turn`
//!   is added to the queued one, and a `Key` supersedes the queued event for
//!   the same key, so the state after draining is the same as if nothing had
//!   been merged.

use std::collections::VecDeque;
use std::io;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::net::{TcpListener, ToSocketAddrs};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::tungstenite::{Bytes, Message};

use crate::protocol::{encode_frame, FrameHeader, Key, ViewerInput, MAX_INPUT_MESSAGE_LEN};

/// Maximum simultaneous connections (viewers plus peers still in the WebSocket
/// handshake); further connections are dropped.
pub const MAX_VIEWERS: usize = 8;
/// Viewer → native input events kept verbatim; beyond it events are merged
/// (see the module docs), never dropped.
const INPUT_QUEUE: usize = 1024;
/// A peer that does not finish the WebSocket handshake in this long is dropped.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);
/// A viewer whose frame send does not complete in this long (it is not reading)
/// is dropped, freeing its slot.
const VIEWER_WRITE_TIMEOUT: Duration = Duration::from_secs(2);

/// The newest frame as one ready-to-send WebSocket payload (header + JPEG).
/// `Bytes` is reference-counted, so every viewer sends the same allocation.
type FrameSlot = Option<Bytes>;

/// Viewer input waiting for [`StreamClient::poll_input`], plus how many
/// viewers hold each key.
#[derive(Default)]
struct InputQueue {
    events: VecDeque<ViewerInput>,
    /// Number of connected viewers holding each key (index: [`Key::index`]).
    holders: [u32; Key::COUNT],
}

impl InputQueue {
    /// Queue `event`. Below [`INPUT_QUEUE`] it is appended as is; above it the
    /// event is merged into the newest queued event of the same kind, so the
    /// queue stays bounded (at most one extra `Pointer`, one extra `Turn` and
    /// one extra event per key) without losing state.
    fn push(&mut self, event: ViewerInput) {
        if self.events.len() < INPUT_QUEUE {
            self.events.push_back(event);
            return;
        }
        // The newest queued event the new one can be merged into: any pointer,
        // any turn, or the last transition of the same key.
        let newest = self
            .events
            .iter_mut()
            .rev()
            .find(|queued| match (&**queued, &event) {
                (ViewerInput::Pointer { .. }, ViewerInput::Pointer { .. })
                | (ViewerInput::Turn { .. }, ViewerInput::Turn { .. }) => true,
                (ViewerInput::Key { key: queued, .. }, ViewerInput::Key { key, .. }) => {
                    queued == key
                }
                _ => false,
            });
        match (newest, event) {
            // Rotations add up. Each term is finite but a sum could overflow,
            // so saturate rather than hand out an infinity.
            (
                Some(ViewerInput::Turn { dyaw, dpitch }),
                ViewerInput::Turn {
                    dyaw: more_yaw,
                    dpitch: more_pitch,
                },
            ) => {
                *dyaw = (*dyaw + more_yaw).clamp(f32::MIN, f32::MAX);
                *dpitch = (*dpitch + more_pitch).clamp(f32::MIN, f32::MAX);
            }
            // Only the latest pointer position matters, and the newer
            // transition of a key leaves the same final key state as both.
            (Some(queued), event) => *queued = event,
            // Nothing queued to merge into (so this happens at most once per
            // kind and key): a `Key` in particular is never dropped.
            (None, event) => self.events.push_back(event),
        }
    }

    /// One viewer's key changed; `held` is that viewer's own key state. Emits a
    /// `Key` event only when the key's *shared* state changes: down when the
    /// first viewer presses it, up when the last viewer holding it lets go.
    /// Repeats (down while already down for this viewer, up while not held)
    /// change nothing.
    fn key(&mut self, held: &mut [bool; Key::COUNT], key: Key, down: bool) {
        let i = key.index();
        if held[i] == down {
            return;
        }
        held[i] = down;
        if down {
            self.holders[i] += 1;
            if self.holders[i] == 1 {
                self.push(ViewerInput::Key { key, down: true });
            }
        } else {
            self.holders[i] = self.holders[i].saturating_sub(1);
            if self.holders[i] == 0 {
                self.push(ViewerInput::Key { key, down: false });
            }
        }
    }

    /// A viewer is gone: let go of every key it still held.
    fn release(&mut self, held: &mut [bool; Key::COUNT]) {
        for key in Key::ALL {
            self.key(held, key, false);
        }
    }

    fn drain(&mut self) -> Vec<ViewerInput> {
        self.events.drain(..).collect()
    }
}

#[derive(Default)]
struct Shared {
    /// Accepted sockets, peers still in the handshake included (this is what
    /// [`MAX_VIEWERS`] bounds).
    connections: AtomicUsize,
    /// Connections whose WebSocket handshake completed.
    viewers: AtomicUsize,
    rejected_input: AtomicU64,
    input: Mutex<InputQueue>,
}

impl Shared {
    fn input(&self) -> MutexGuard<'_, InputQueue> {
        self.input.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Native-client side of the stream: pushes frames in, pulls viewer input out.
/// All methods are synchronous and non-blocking (safe to call from a render
/// loop that is not async).
pub struct StreamClient {
    frames: watch::Sender<FrameSlot>,
    next_seq: AtomicU64,
    shared: Arc<Shared>,
}

impl StreamClient {
    /// Publish a JPEG (see [`crate::protocol::FLAG_KEYFRAME`] etc. for `flags`),
    /// replacing any frame a slow viewer has not sent yet. Returns the frame's
    /// sequence number (1, 2, 3, …).
    pub fn push_jpeg(&self, jpeg: Vec<u8>, flags: u8) -> u64 {
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        // Framed once here, not once per viewer in each connection task.
        let wire = Bytes::from(encode_frame(FrameHeader { seq, flags }, &jpeg));
        self.frames.send_replace(Some(wire));
        seq
    }

    /// Drain every viewer input event received since the last call, oldest first.
    ///
    /// `Key` events describe the key state shared by all viewers (see the module
    /// docs): a key-up arrives when the last viewer holding the key releases it
    /// or disconnects.
    pub fn poll_input(&self) -> Vec<ViewerInput> {
        self.shared.input().drain()
    }

    /// Number of viewers currently connected (WebSocket handshake complete; a
    /// peer that only opened a TCP connection is not a viewer). A render loop
    /// can skip encoding while this is 0.
    pub fn viewer_count(&self) -> usize {
        self.shared.viewers.load(Ordering::Relaxed)
    }

    /// Malformed viewer messages dropped so far.
    pub fn rejected_input(&self) -> u64 {
        self.shared.rejected_input.load(Ordering::Relaxed)
    }
}

/// The accepting side. Dropping it stops accepting new viewers.
pub struct StreamServer {
    local_addr: SocketAddr,
    accept: JoinHandle<()>,
}

impl StreamServer {
    /// Bind a `ws://` listener and start accepting viewers on the current tokio
    /// runtime. Returns the server and the [`StreamClient`] that feeds it.
    pub async fn bind(addr: impl ToSocketAddrs) -> io::Result<(Self, StreamClient)> {
        let listener = TcpListener::bind(addr).await?;
        let local_addr = listener.local_addr()?;
        let (frames, frames_rx) = watch::channel(None);
        let shared = Arc::new(Shared::default());
        let accept = tokio::spawn(accept_loop(listener, frames_rx, Arc::clone(&shared)));
        Ok((
            Self { local_addr, accept },
            StreamClient {
                frames,
                next_seq: AtomicU64::new(1),
                shared,
            },
        ))
    }

    /// The bound address (useful with port 0).
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }
}

impl Drop for StreamServer {
    fn drop(&mut self) {
        self.accept.abort();
    }
}

/// One accepted socket, from `accept` until its task ends (however it ends:
/// normal return, an error, or the task being dropped). It holds a connection
/// slot from the start and becomes a *viewer* once the handshake completes.
struct Connection {
    shared: Arc<Shared>,
    /// The WebSocket handshake completed: counted in `Shared::viewers`.
    viewer: bool,
    /// The keys this viewer currently holds.
    held: [bool; Key::COUNT],
}

impl Connection {
    fn new(shared: Arc<Shared>) -> Self {
        Self {
            shared,
            viewer: false,
            held: [false; Key::COUNT],
        }
    }

    fn become_viewer(&mut self) {
        if !self.viewer {
            self.viewer = true;
            self.shared.viewers.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Hand one decoded input event from this viewer to the shared queue.
    fn input(&mut self, event: ViewerInput) {
        let mut queue = self.shared.input();
        match event {
            ViewerInput::Key { key, down } => queue.key(&mut self.held, key, down),
            other => queue.push(other),
        }
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        // Nobody is left to send this viewer's key-ups.
        self.shared.input().release(&mut self.held);
        // Free the slot before the viewer count drops, so whoever sees one
        // viewer fewer can connect a replacement straight away.
        self.shared.connections.fetch_sub(1, Ordering::Relaxed);
        if self.viewer {
            self.shared.viewers.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

async fn accept_loop(
    listener: TcpListener,
    frames: watch::Receiver<FrameSlot>,
    shared: Arc<Shared>,
) {
    loop {
        let (stream, peer) = match listener.accept().await {
            Ok(v) => v,
            Err(e) => {
                log::warn!("space-stream: accept: {e}");
                tokio::time::sleep(Duration::from_millis(50)).await;
                continue;
            }
        };
        if shared.connections.fetch_add(1, Ordering::Relaxed) >= MAX_VIEWERS {
            shared.connections.fetch_sub(1, Ordering::Relaxed);
            log::warn!("space-stream: viewer limit reached, dropping {peer}");
            continue;
        }
        let mut connection = Connection::new(Arc::clone(&shared));
        // Never let Nagle coalesce small frame messages behind an unflushed write.
        let _ = stream.set_nodelay(true);
        let frames = frames.clone();
        tokio::spawn(async move {
            let config = WebSocketConfig::default()
                .max_message_size(Some(MAX_INPUT_MESSAGE_LEN * 4))
                .max_frame_size(Some(MAX_INPUT_MESSAGE_LEN * 4));
            let ws = tokio::time::timeout(
                HANDSHAKE_TIMEOUT,
                tokio_tungstenite::accept_async_with_config(stream, Some(config)),
            )
            .await;
            match ws {
                Ok(Ok(ws)) => serve_viewer(ws, frames, &mut connection).await,
                Ok(Err(e)) => log::debug!("space-stream: handshake with {peer} failed: {e}"),
                Err(_) => log::debug!("space-stream: handshake with {peer} timed out"),
            }
            drop(connection);
        });
    }
}

async fn serve_viewer(
    ws: tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
    mut frames: watch::Receiver<FrameSlot>,
    connection: &mut Connection,
) {
    connection.become_viewer();
    let (mut write, mut read) = ws.split();

    // Reading and writing are two futures polled side by side. A frame send can
    // take as long as the viewer's link needs (up to `VIEWER_WRITE_TIMEOUT`);
    // with a single loop that awaited the send, input sat unread for that whole
    // time and, on a link slower than the frame rate, fell further and further
    // behind.
    let reader = async {
        while let Some(message) = read.next().await {
            match message {
                Ok(Message::Binary(bytes)) => match ViewerInput::decode(&bytes) {
                    Ok(event) => connection.input(event),
                    Err(_) => {
                        let shared = &connection.shared;
                        shared.rejected_input.fetch_add(1, Ordering::Relaxed);
                    }
                },
                Ok(Message::Close(_)) | Err(_) => break,
                // Text / ping / pong: tungstenite answers pings itself.
                Ok(_) => {}
            }
        }
    };
    let writer = async {
        // A late joiner gets the newest frame immediately rather than waiting
        // for the next push.
        frames.mark_changed();
        // `changed` fails once the StreamClient is dropped.
        while frames.changed().await.is_ok() {
            let Some(frame) = frames.borrow_and_update().clone() else {
                continue;
            };
            let sent =
                tokio::time::timeout(VIEWER_WRITE_TIMEOUT, write.send(Message::Binary(frame)))
                    .await;
            match sent {
                Ok(Ok(())) => {}
                Ok(Err(_)) => break,
                Err(_) => {
                    // The viewer is not reading: drop it so it cannot hold a
                    // slot forever (the caller closes the connection).
                    log::debug!("space-stream: viewer write timed out, dropping it");
                    break;
                }
            }
        }
    };
    // Whichever side ends first ends the connection.
    tokio::select! {
        () = reader => {}
        () = writer => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::{jpeg_dimensions, JpegEncoder};
    use crate::protocol::{decode_frame, FLAG_DOWNSCALED, FLAG_KEYFRAME};
    use tokio::net::TcpStream;
    use tokio_tungstenite::WebSocketStream;

    type Ws = WebSocketStream<TcpStream>;

    async fn connect(addr: SocketAddr) -> Ws {
        let tcp = TcpStream::connect(addr).await.unwrap();
        let (ws, _) = tokio_tungstenite::client_async(format!("ws://{addr}/"), tcp)
            .await
            .unwrap();
        ws
    }

    async fn next_binary(ws: &mut Ws) -> Vec<u8> {
        loop {
            let msg = tokio::time::timeout(Duration::from_secs(10), ws.next())
                .await
                .expect("timed out waiting for a frame")
                .expect("stream ended")
                .expect("ws error");
            if let Message::Binary(b) = msg {
                return b.to_vec();
            }
        }
    }

    async fn wait_for(mut cond: impl FnMut() -> bool) {
        for _ in 0..200 {
            if cond() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("condition not reached in time");
    }

    fn key(key: Key, down: bool) -> ViewerInput {
        ViewerInput::Key { key, down }
    }

    async fn send_input(ws: &mut Ws, input: ViewerInput) {
        ws.send(Message::Binary(input.encode().into()))
            .await
            .unwrap();
    }

    /// Collect polled input until `expected.len()` events arrived (or 5 s
    /// passed), then compare.
    async fn expect_input(client: &StreamClient, expected: &[ViewerInput]) {
        let mut got = Vec::new();
        for _ in 0..200 {
            got.extend(client.poll_input());
            if got.len() >= expected.len() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        assert_eq!(got, expected);
    }

    /// End to end over loopback, no browser: push a synthetic RGBA frame, receive
    /// it on a WebSocket client, decode the framing and the JPEG header, and check
    /// the dimensions survived (half resolution).
    #[tokio::test]
    async fn loopback_frame_round_trip() {
        let (server, client) = StreamServer::bind("127.0.0.1:0").await.unwrap();
        let mut ws = connect(server.local_addr()).await;
        wait_for(|| client.viewer_count() == 1).await;

        let (w, h) = (80u32, 60u32);
        let rgba: Vec<u8> = (0..w * h)
            .flat_map(|i| [(i % w) as u8 * 3, (i / w) as u8 * 4, 128, 255])
            .collect();
        let enc = JpegEncoder::default().encode(&rgba, w, h).unwrap();
        let seq = client.push_jpeg(enc.bytes.clone(), FLAG_KEYFRAME | FLAG_DOWNSCALED);
        assert_eq!(seq, 1);

        let msg = next_binary(&mut ws).await;
        let (header, jpeg) = decode_frame(&msg).unwrap();
        assert_eq!(header.seq, 1);
        assert_eq!(header.flags, FLAG_KEYFRAME | FLAG_DOWNSCALED);
        assert_eq!(jpeg, enc.bytes);
        assert_eq!(jpeg_dimensions(jpeg), Some((40, 30)));
    }

    /// The wire message is built once per frame and every viewer is handed the
    /// same allocation (no per-viewer copy of the JPEG).
    #[tokio::test]
    async fn the_framed_message_is_built_once_and_shared() {
        let (_server, client) = StreamServer::bind("127.0.0.1:0").await.unwrap();
        let seq = client.push_jpeg(vec![9, 8, 7], FLAG_KEYFRAME);
        // What two connection tasks each take out of the slot.
        let a = client.frames.subscribe().borrow().clone().unwrap();
        let b = client.frames.subscribe().borrow().clone().unwrap();
        let header = FrameHeader {
            seq,
            flags: FLAG_KEYFRAME,
        };
        assert_eq!(a[..], encode_frame(header, &[9, 8, 7])[..]);
        assert_eq!(a.as_ptr(), b.as_ptr(), "viewers share one buffer");
    }

    /// Viewer input reaches the native side, in order; garbage is counted and
    /// dropped without killing the connection.
    #[tokio::test]
    async fn viewer_input_reaches_client() {
        let (server, client) = StreamServer::bind("127.0.0.1:0").await.unwrap();
        let mut ws = connect(server.local_addr()).await;
        let sent = [
            key(Key::Forward, true),
            ViewerInput::Turn {
                dyaw: 0.5,
                dpitch: -0.25,
            },
            // Below the queue limit nothing is merged: a second Turn stays a
            // second event (the receiver clamps pitch after each one).
            ViewerInput::Turn {
                dyaw: 0.25,
                dpitch: 0.125,
            },
        ];
        send_input(&mut ws, sent[0]).await;
        ws.send(Message::Binary(vec![0xee, 1, 2].into()))
            .await
            .unwrap();
        send_input(&mut ws, sent[1]).await;
        send_input(&mut ws, sent[2]).await;

        expect_input(&client, &sent).await;
        assert_eq!(client.rejected_input(), 1);
    }

    /// A viewer that connects after a frame was pushed still gets it, and a burst
    /// of pushes collapses to the newest (latest-frame-wins).
    #[tokio::test]
    async fn late_joiner_gets_latest_frame() {
        let (server, client) = StreamServer::bind("127.0.0.1:0").await.unwrap();
        client.push_jpeg(vec![1], 0);
        client.push_jpeg(vec![2], 0);
        client.push_jpeg(vec![3], 0);
        let mut ws = connect(server.local_addr()).await;
        let msg = next_binary(&mut ws).await;
        let (header, jpeg) = decode_frame(&msg).unwrap();
        assert_eq!((header.seq, jpeg), (3, &[3u8][..]));
    }

    /// Closing the socket frees the viewer slot.
    #[tokio::test]
    async fn viewer_count_tracks_connections() {
        let (server, client) = StreamServer::bind("127.0.0.1:0").await.unwrap();
        assert_eq!(client.viewer_count(), 0);
        let mut ws = connect(server.local_addr()).await;
        wait_for(|| client.viewer_count() == 1).await;
        ws.close(None).await.unwrap();
        drop(ws);
        wait_for(|| client.viewer_count() == 0).await;
    }

    /// Poll `cond` every 25 ms for up to `secs` seconds; true if it held.
    async fn wait_up_to(secs: u64, mut cond: impl FnMut() -> bool) -> bool {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(secs);
        while tokio::time::Instant::now() < deadline {
            if cond() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        cond()
    }

    /// Finding B: a viewer that never reads must be dropped after the write
    /// timeout, freeing its slot so it cannot be used to exhaust `MAX_VIEWERS`.
    #[tokio::test]
    async fn stalled_viewer_is_dropped_after_write_timeout() {
        let (server, client) = StreamServer::bind("127.0.0.1:0").await.unwrap();
        let addr = server.local_addr();
        // Connects, completes the handshake, then never reads a byte.
        let _stalled = connect(addr).await;
        wait_for(|| client.viewer_count() == 1).await;

        // Frames far larger than the socket buffers, so the send really blocks.
        let big = vec![0u8; 8 * 1024 * 1024];
        let start = tokio::time::Instant::now();
        let mut freed = false;
        while start.elapsed() < VIEWER_WRITE_TIMEOUT + Duration::from_secs(10) {
            client.push_jpeg(big.clone(), 0);
            if wait_up_to(0, || client.viewer_count() == 0).await {
                freed = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert!(freed, "stalled viewer still holds its slot");

        // The slot is reusable: a fresh viewer connects and receives a frame.
        let mut ws = connect(addr).await;
        let seq = client.push_jpeg(vec![7], 0);
        let jpeg = loop {
            let msg = next_binary(&mut ws).await;
            let (header, jpeg) = decode_frame(&msg).unwrap();
            if header.seq >= seq {
                break jpeg.to_vec();
            }
        };
        assert_eq!(jpeg, [7u8]);
    }

    /// Input must be read while a frame send is in flight. The viewer here
    /// never reads, so the server's send of a frame far larger than the socket
    /// buffers is stuck (until the 2 s write timeout drops the viewer); a key
    /// press sent meanwhile must still reach the native side promptly. A loop
    /// that awaits the send before reading again never delivers it.
    #[tokio::test]
    async fn input_is_read_while_a_frame_send_is_stalled() {
        let (server, client) = StreamServer::bind("127.0.0.1:0").await.unwrap();
        let mut ws = connect(server.local_addr()).await;
        wait_for(|| client.viewer_count() == 1).await;

        // Several frames, each far larger than any loopback socket buffer: the
        // first may be swallowed by the buffers, after that the connection
        // task is stuck in a send.
        let big = vec![0u8; 8 * 1024 * 1024];
        for _ in 0..4 {
            client.push_jpeg(big.clone(), 0);
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        let press = key(Key::Forward, true);
        send_input(&mut ws, press).await;
        let mut got = Vec::new();
        let arrived = wait_up_to(1, || {
            got.extend(client.poll_input());
            !got.is_empty()
        })
        .await;
        assert!(arrived, "input was not read while the frame send was stuck");
        assert_eq!(got, [press]);

        // The send really was stuck: the write timeout drops the viewer (and
        // with it releases the key it was holding).
        assert!(
            wait_up_to(10, || client.viewer_count() == 0).await,
            "the frame send was never stalled, so this test proved nothing"
        );
        assert_eq!(client.poll_input(), [key(Key::Forward, false)]);
    }

    /// Finding A(a): the viewer cap is enforced; the extra connection is refused
    /// and a slot freed later can be taken again.
    #[tokio::test]
    async fn max_viewers_is_enforced() {
        let (server, client) = StreamServer::bind("127.0.0.1:0").await.unwrap();
        let addr = server.local_addr();
        let mut viewers = Vec::new();
        for _ in 0..MAX_VIEWERS {
            viewers.push(connect(addr).await);
        }
        wait_for(|| client.viewer_count() == MAX_VIEWERS).await;

        let tcp = TcpStream::connect(addr).await.unwrap();
        let extra = tokio::time::timeout(
            Duration::from_secs(5),
            tokio_tungstenite::client_async(format!("ws://{addr}/"), tcp),
        )
        .await
        .expect("the refused peer was left hanging");
        assert!(extra.is_err(), "viewer MAX_VIEWERS+1 was accepted");
        assert_eq!(client.viewer_count(), MAX_VIEWERS);

        drop(viewers.pop());
        wait_for(|| client.viewer_count() == MAX_VIEWERS - 1).await;
        let _again = connect(addr).await;
        wait_for(|| client.viewer_count() == MAX_VIEWERS).await;
    }

    /// Finding A(b): input over the size cap is rejected (never delivered) and
    /// the server keeps serving. A message past the websocket limit drops only
    /// that connection.
    #[tokio::test]
    async fn oversized_input_is_rejected_and_server_keeps_serving() {
        let (server, client) = StreamServer::bind("127.0.0.1:0").await.unwrap();
        let addr = server.local_addr();
        let mut ws = connect(addr).await;

        // A well-formed key event padded just past the cap: rejected, counted.
        let mut padded = key(Key::Forward, true).encode();
        padded.resize(MAX_INPUT_MESSAGE_LEN + 1, 0);
        ws.send(Message::Binary(padded.into())).await.unwrap();
        wait_for(|| client.rejected_input() == 1).await;
        assert!(client.poll_input().is_empty(), "oversized input delivered");

        // The same connection still works.
        let ok = ViewerInput::Turn {
            dyaw: 1.0,
            dpitch: 0.0,
        };
        send_input(&mut ws, ok).await;
        expect_input(&client, &[ok]).await;

        // Far past the cap: the connection is cut, nothing is delivered.
        let _ = ws
            .send(Message::Binary(
                vec![2u8; MAX_INPUT_MESSAGE_LEN * 64].into(),
            ))
            .await;
        let ended = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                match ws.next().await {
                    None | Some(Err(_)) | Some(Ok(Message::Close(_))) => break,
                    Some(Ok(_)) => {}
                }
            }
        })
        .await;
        assert!(ended.is_ok(), "connection survived a huge input message");
        assert!(client.poll_input().is_empty());
        wait_for(|| client.viewer_count() == 0).await;

        // And the server still serves new viewers.
        let mut ws2 = connect(addr).await;
        client.push_jpeg(vec![9], 0);
        let msg = next_binary(&mut ws2).await;
        assert_eq!(decode_frame(&msg).unwrap().1, &[9u8][..]);
    }

    /// Finding A(c): a peer that never completes the WebSocket handshake is
    /// dropped within `HANDSHAKE_TIMEOUT` (plus slack). It holds a connection
    /// slot meanwhile but is never counted as a viewer, so a render loop does
    /// not start encoding for it and held keys are still released when the last
    /// real viewer leaves.
    #[tokio::test]
    async fn handshake_timeout_drops_silent_peer() {
        let (server, client) = StreamServer::bind("127.0.0.1:0").await.unwrap();
        let connections = || client.shared.connections.load(Ordering::Relaxed);
        let _silent = TcpStream::connect(server.local_addr()).await.unwrap();
        wait_for(|| connections() == 1).await;
        assert_eq!(client.viewer_count(), 0, "a handshaking peer is no viewer");
        let start = tokio::time::Instant::now();
        let slack = Duration::from_secs(3);
        let mut counted_as_viewer = false;
        let dropped = wait_up_to((HANDSHAKE_TIMEOUT + slack).as_secs(), || {
            counted_as_viewer |= client.viewer_count() != 0;
            connections() == 0
        })
        .await;
        assert!(dropped, "silent peer still holds a slot");
        assert!(
            !counted_as_viewer,
            "a handshaking peer was counted as a viewer"
        );
        assert!(
            start.elapsed() >= HANDSHAKE_TIMEOUT / 2,
            "dropped suspiciously early: {:?}",
            start.elapsed()
        );
    }

    /// A viewer that vanishes while holding keys must not leave them held for
    /// the viewers that remain — but a key another viewer also holds stays down.
    #[tokio::test]
    async fn a_leaving_viewer_releases_its_keys_but_not_another_viewers() {
        let (server, client) = StreamServer::bind("127.0.0.1:0").await.unwrap();
        let addr = server.local_addr();
        let mut a = connect(addr).await;
        let mut b = connect(addr).await;
        wait_for(|| client.viewer_count() == 2).await;

        send_input(&mut a, key(Key::Forward, true)).await;
        send_input(&mut a, key(Key::Up, true)).await;
        expect_input(&client, &[key(Key::Forward, true), key(Key::Up, true)]).await;

        // B presses Forward too (already down: no second event) and Left.
        send_input(&mut b, key(Key::Forward, true)).await;
        send_input(&mut b, key(Key::Left, true)).await;
        expect_input(&client, &[key(Key::Left, true)]).await;

        // A's tab closes without a single key-up. Only A held Up; Forward is
        // still held by B.
        drop(a);
        expect_input(&client, &[key(Key::Up, false)]).await;
        wait_for(|| client.viewer_count() == 1).await;
        assert!(client.poll_input().is_empty(), "Forward is still held by B");

        // B releases Forward itself, then vanishes holding Left.
        send_input(&mut b, key(Key::Forward, false)).await;
        expect_input(&client, &[key(Key::Forward, false)]).await;
        drop(b);
        expect_input(&client, &[key(Key::Left, false)]).await;
        wait_for(|| client.viewer_count() == 0).await;
        assert!(client.poll_input().is_empty());
    }

    #[test]
    fn keys_are_reference_counted_across_viewers() {
        let mut queue = InputQueue::default();
        let (mut a, mut b) = ([false; Key::COUNT], [false; Key::COUNT]);

        queue.key(&mut a, Key::Left, true);
        queue.key(&mut a, Key::Left, true); // auto-repeat: nothing new
        queue.key(&mut b, Key::Left, true); // second holder: nothing new
        queue.key(&mut b, Key::Right, false); // up for a key nobody holds
        assert_eq!(queue.drain(), [key(Key::Left, true)]);

        queue.key(&mut a, Key::Left, false); // B still holds it
        assert!(queue.drain().is_empty());
        queue.key(&mut b, Key::Left, false);
        assert_eq!(queue.drain(), [key(Key::Left, false)]);

        // Releasing everything a viewer held emits one key-up per key that
        // nobody else holds, and leaves that viewer holding nothing.
        queue.key(&mut a, Key::Forward, true);
        queue.key(&mut a, Key::Down, true);
        queue.key(&mut b, Key::Down, true);
        queue.drain();
        queue.release(&mut a);
        assert_eq!(queue.drain(), [key(Key::Forward, false)]);
        assert_eq!(a, [false; Key::COUNT]);
        queue.release(&mut a); // idempotent
        queue.release(&mut b);
        assert_eq!(queue.drain(), [key(Key::Down, false)]);
        assert_eq!(queue.holders, [0; Key::COUNT]);
    }

    /// What the native side ends up with after applying a run of events.
    #[derive(Debug, Default, PartialEq)]
    struct FinalState {
        keys_down: [bool; Key::COUNT],
        /// Summed `(dyaw, dpitch)`.
        turned: (f64, f64),
        /// The last pointer `(x, y, buttons)`.
        pointer: Option<(f32, f32, u8)>,
    }

    /// Replay `events` the way the native side does.
    fn final_state(events: &[ViewerInput]) -> FinalState {
        let mut state = FinalState::default();
        for event in events {
            match *event {
                ViewerInput::Key { key, down } => state.keys_down[key.index()] = down,
                ViewerInput::Turn { dyaw, dpitch } => {
                    state.turned.0 += f64::from(dyaw);
                    state.turned.1 += f64::from(dpitch);
                }
                ViewerInput::Pointer { x, y, buttons } => state.pointer = Some((x, y, buttons)),
            }
        }
        state
    }

    /// With the native side not polling, the queue stays bounded by merging
    /// state-like events — and the state after draining is exactly what the
    /// unmerged stream would have produced. In particular no key transition is
    /// lost (dropping the newest event here used to lose key-ups).
    #[test]
    fn a_full_queue_merges_state_but_never_loses_a_key() {
        let mut queue = InputQueue::default();
        let mut held = [false; Key::COUNT];
        let mut sent = Vec::new();
        let mut feed = |queue: &mut InputQueue, held: &mut [bool; Key::COUNT], e: ViewerInput| {
            sent.push(e);
            match e {
                ViewerInput::Key { key, down } => queue.key(held, key, down),
                other => queue.push(other),
            }
        };

        // A long drag with every key pressed along the way fills the queue...
        for i in 0..INPUT_QUEUE as u32 * 3 {
            let (x, y) = (i as f32 / 4096.0, 0.5);
            feed(
                &mut queue,
                &mut held,
                ViewerInput::Pointer { x, y, buttons: 1 },
            );
            feed(
                &mut queue,
                &mut held,
                ViewerInput::Turn {
                    dyaw: 0.25,
                    dpitch: -0.125,
                },
            );
            if i % 500 == 0 {
                let k = Key::ALL[(i / 500) as usize % Key::COUNT];
                feed(&mut queue, &mut held, key(k, true));
            }
        }
        // ...then, far past the limit, every key is released and one is
        // pressed again.
        for k in Key::ALL {
            feed(&mut queue, &mut held, key(k, false));
        }
        feed(&mut queue, &mut held, key(Key::Back, true));
        feed(&mut queue, &mut held, key(Key::Back, false));
        feed(&mut queue, &mut held, key(Key::Up, true));

        assert!(
            queue.events.len() <= INPUT_QUEUE + 2 + Key::COUNT,
            "queue grew to {}",
            queue.events.len()
        );
        let drained = queue.drain();
        assert!(sent.len() > drained.len() * 5, "nothing was merged");
        // Same keys down (no transition lost), same newest pointer, same total
        // turn (0.25 and 0.125 steps are exact in f32, so the sums are too).
        let state = final_state(&drained);
        assert_eq!(state, final_state(&sent));
        let only_up_is_down = Key::ALL.map(|k| k == Key::Up);
        assert_eq!(state.keys_down, only_up_is_down);
        assert_eq!(state.turned, (768.0, -384.0));
        assert!(queue.events.is_empty());
    }

    /// Below the limit nothing is merged, and merged turns saturate instead of
    /// overflowing to infinity (every decoded value is finite; a sum of two
    /// need not be).
    #[test]
    fn merged_turns_stay_finite() {
        let mut queue = InputQueue::default();
        let huge = ViewerInput::Turn {
            dyaw: f32::MAX,
            dpitch: f32::MIN,
        };
        for _ in 0..INPUT_QUEUE {
            queue.push(huge);
        }
        assert_eq!(
            queue.events.len(),
            INPUT_QUEUE,
            "kept verbatim up to the limit"
        );
        queue.push(huge);
        queue.push(huge);
        assert_eq!(queue.events.len(), INPUT_QUEUE);
        assert_eq!(queue.events.back(), Some(&huge), "saturated, still finite");
        for event in queue.drain() {
            assert_eq!(ViewerInput::decode(&event.encode()), Ok(event));
        }
    }
}
