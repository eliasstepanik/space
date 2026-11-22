//! The `ws://` stream server and the synchronous [`StreamClient`] handle a
//! native client uses to push frames into it and pull viewer input out of it.
//!
//! Backpressure is *latest-frame-wins* (from the Phase-0 `stream-loop` spike): a
//! `tokio::sync::watch` slot holds the newest frame, so a slow viewer skips
//! frames instead of building latency, and no wake-up can be lost (a `Notify` +
//! `Mutex<Option<_>>` pair can lose one).

use std::io;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::net::{TcpListener, ToSocketAddrs};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::tungstenite::Message;

use crate::protocol::{encode_frame, FrameHeader, ViewerInput, MAX_INPUT_MESSAGE_LEN};

/// Maximum simultaneously connected viewers; further connections are dropped.
pub const MAX_VIEWERS: usize = 8;
/// Capacity of the viewer → native input queue; input beyond it is dropped.
const INPUT_QUEUE: usize = 1024;
/// A peer that does not finish the WebSocket handshake in this long is dropped.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);
/// A viewer whose frame send does not complete in this long (it is not reading)
/// is dropped, freeing its slot.
const VIEWER_WRITE_TIMEOUT: Duration = Duration::from_secs(2);

struct Frame {
    seq: u64,
    flags: u8,
    jpeg: Vec<u8>,
}

#[derive(Default)]
struct Shared {
    viewers: AtomicUsize,
    rejected_input: AtomicU64,
}

/// Native-client side of the stream: pushes frames in, pulls viewer input out.
/// All methods are synchronous and non-blocking (safe to call from a render
/// loop that is not async).
pub struct StreamClient {
    frames: watch::Sender<Option<Arc<Frame>>>,
    input: Mutex<Receiver<ViewerInput>>,
    next_seq: AtomicU64,
    shared: Arc<Shared>,
}

impl StreamClient {
    /// Publish a JPEG (see [`crate::protocol::FLAG_KEYFRAME`] etc. for `flags`),
    /// replacing any frame a slow viewer has not sent yet. Returns the frame's
    /// sequence number (1, 2, 3, …).
    pub fn push_jpeg(&self, jpeg: Vec<u8>, flags: u8) -> u64 {
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        self.frames
            .send_replace(Some(Arc::new(Frame { seq, flags, jpeg })));
        seq
    }

    /// Drain every viewer input event received since the last call, oldest first.
    pub fn poll_input(&self) -> Vec<ViewerInput> {
        let rx = self.input.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::new();
        loop {
            match rx.try_recv() {
                Ok(i) => out.push(i),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => return out,
            }
        }
    }

    /// Number of viewers currently connected. A render loop can skip encoding
    /// while this is 0.
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
        let (input_tx, input_rx) = std::sync::mpsc::sync_channel(INPUT_QUEUE);
        let shared = Arc::new(Shared::default());
        let accept = tokio::spawn(accept_loop(
            listener,
            frames_rx,
            input_tx,
            Arc::clone(&shared),
        ));
        Ok((
            Self { local_addr, accept },
            StreamClient {
                frames,
                input: Mutex::new(input_rx),
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

/// Decrements the viewer count when a connection task ends.
struct ViewerGuard(Arc<Shared>);

impl Drop for ViewerGuard {
    fn drop(&mut self) {
        self.0.viewers.fetch_sub(1, Ordering::Relaxed);
    }
}

async fn accept_loop(
    listener: TcpListener,
    frames: watch::Receiver<Option<Arc<Frame>>>,
    input: SyncSender<ViewerInput>,
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
        if shared.viewers.fetch_add(1, Ordering::Relaxed) >= MAX_VIEWERS {
            shared.viewers.fetch_sub(1, Ordering::Relaxed);
            log::warn!("space-stream: viewer limit reached, dropping {peer}");
            continue;
        }
        let guard = ViewerGuard(Arc::clone(&shared));
        // Never let Nagle coalesce small frame messages behind an unflushed write.
        let _ = stream.set_nodelay(true);
        let (frames, input) = (frames.clone(), input.clone());
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
                Ok(Ok(ws)) => serve_viewer(ws, frames, input, &guard.0).await,
                Ok(Err(e)) => log::debug!("space-stream: handshake with {peer} failed: {e}"),
                Err(_) => log::debug!("space-stream: handshake with {peer} timed out"),
            }
            drop(guard);
        });
    }
}

async fn serve_viewer(
    ws: tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
    mut frames: watch::Receiver<Option<Arc<Frame>>>,
    input: SyncSender<ViewerInput>,
    shared: &Shared,
) {
    let (mut write, mut read) = ws.split();
    // A late joiner gets the newest frame immediately rather than waiting for
    // the next push.
    frames.mark_changed();
    loop {
        tokio::select! {
            changed = frames.changed() => {
                if changed.is_err() {
                    break; // the StreamClient was dropped
                }
                let Some(frame) = frames.borrow_and_update().clone() else {
                    continue;
                };
                let wire = encode_frame(
                    FrameHeader { seq: frame.seq, flags: frame.flags },
                    &frame.jpeg,
                );
                let sent = tokio::time::timeout(
                    VIEWER_WRITE_TIMEOUT,
                    write.send(Message::Binary(wire.into())),
                )
                .await;
                match sent {
                    Ok(Ok(())) => {}
                    Ok(Err(_)) => break,
                    Err(_) => {
                        // The viewer is not reading: drop it so it cannot hold
                        // a slot forever (the caller closes the connection).
                        log::debug!("space-stream: viewer write timed out, dropping it");
                        break;
                    }
                }
            }
            msg = read.next() => match msg {
                Some(Ok(Message::Binary(b))) => match ViewerInput::decode(&b) {
                    // `try_send`: never block the connection on a slow consumer.
                    Ok(i) => { let _ = input.try_send(i); }
                    Err(_) => { shared.rejected_input.fetch_add(1, Ordering::Relaxed); }
                },
                Some(Ok(Message::Close(_))) | Some(Err(_)) | None => break,
                // Text / ping / pong: tungstenite answers pings itself.
                Some(Ok(_)) => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::{jpeg_dimensions, JpegEncoder};
    use crate::protocol::{decode_frame, Key, FLAG_DOWNSCALED, FLAG_KEYFRAME};
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

    /// Viewer input reaches the native side, in order; garbage is counted and
    /// dropped without killing the connection.
    #[tokio::test]
    async fn viewer_input_reaches_client() {
        let (server, client) = StreamServer::bind("127.0.0.1:0").await.unwrap();
        let mut ws = connect(server.local_addr()).await;
        let sent = [
            ViewerInput::Key {
                key: Key::Forward,
                down: true,
            },
            ViewerInput::View {
                yaw: 0.5,
                pitch: -0.25,
            },
        ];
        ws.send(Message::Binary(sent[0].encode().into()))
            .await
            .unwrap();
        ws.send(Message::Binary(vec![0xee, 1, 2].into()))
            .await
            .unwrap();
        ws.send(Message::Binary(sent[1].encode().into()))
            .await
            .unwrap();

        let mut got = Vec::new();
        wait_for(|| {
            got.extend(client.poll_input());
            got.len() == 2
        })
        .await;
        assert_eq!(got, sent);
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
}
