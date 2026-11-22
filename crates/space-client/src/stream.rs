//! `--stream ADDR`: render headlessly, JPEG-encode each frame and broadcast it to
//! browser viewers over `ws://` (see the `space-stream` crate).
//!
//! # Multiplayer seam
//!
//! A browser viewer is not a peer of the authoritative server. This process is
//! the *proxy client*: with `--server ADDR` it connects to the shard as an
//! ordinary player through the `space-net` client path ([`space_client::link`]), and
//! the viewer is only its remote screen and controller. Viewer input arrives
//! through [`StreamClient::poll_input`], is applied to the proxy's
//! [`InputState`] by [`InputState::apply`] and, by [`pump_input`], turned into
//! thrust sent to the server as `InputTick`s. The server's replicated ship
//! position then drives the camera of the scene that is rendered, JPEG-encoded
//! and streamed back. Without `--server` the proxy stays offline and moves the
//! camera locally, exactly as before.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use space_render::{CameraPose, ContextOptions, RenderContext, RenderError, Scene};
use space_stream::protocol::{FLAG_DOWNSCALED, FLAG_KEYFRAME};
use space_stream::{JpegEncoder, Key, StreamClient, StreamServer, ViewerInput};
use space_world::WorldPos;

use crate::{demo_options, Config, Mode};
use space_client::link::ServerLink;

/// Target stream frame rate.
const FRAME_BUDGET: Duration = Duration::from_millis(33);
/// Poll interval while no viewer is connected (nothing is rendered).
const IDLE_POLL: Duration = Duration::from_millis(50);

/// The proxy client's input state: where the (viewer-driven) camera is and
/// which way it looks. Local and viewer input both end up here.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InputState {
    /// Camera offset from the scene's base camera, metres.
    pub offset: [f64; 3],
    /// Yaw about +Y in radians (0 looks down -Z, positive turns left).
    pub yaw: f32,
    /// Pitch in radians (positive looks up).
    pub pitch: f32,
    /// Held movement keys, indexed by `Key as usize - 1`.
    held: [bool; 6],
    /// Last pointer position (normalised) and button mask, if any.
    pub pointer: Option<(f32, f32, u8)>,
}

impl InputState {
    /// Apply one input event.
    pub fn apply(&mut self, input: ViewerInput) {
        match input {
            ViewerInput::Key { key, down } => self.held[key as usize - 1] = down,
            ViewerInput::View { yaw, pitch } => {
                self.yaw = yaw;
                self.pitch = pitch.clamp(-1.553_343, 1.553_343);
            }
            ViewerInput::Pointer { x, y, buttons } => self.pointer = Some((x, y, buttons)),
        }
    }

    /// Release every held movement key (a viewer that vanished can never send
    /// its key-ups, so its keys would otherwise stay held forever).
    pub fn release_keys(&mut self) {
        self.held = [false; 6];
    }

    fn is_held(&self, key: Key) -> bool {
        self.held[key as usize - 1]
    }

    /// Unit view direction.
    pub fn forward(&self) -> [f32; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        [-sy * cp, sp, -cy * cp]
    }

    fn right(&self) -> [f32; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        [cy, 0.0, -sy]
    }

    /// Advance the camera by `dt` seconds at `speed` metres/second.
    pub fn step(&mut self, dt: f32, speed: f64) {
        let axis = |pos: Key, neg: Key| {
            f64::from(i8::from(self.is_held(pos)) - i8::from(self.is_held(neg)))
        };
        let (fwd, right, up) = (
            axis(Key::Forward, Key::Back),
            axis(Key::Right, Key::Left),
            axis(Key::Up, Key::Down),
        );
        let (f, r) = (self.forward(), self.right());
        let step = f64::from(dt) * speed;
        for i in 0..3 {
            self.offset[i] += step * (fwd * f64::from(f[i]) + right * f64::from(r[i]));
        }
        self.offset[1] += step * up;
    }

    /// Thrust intent for the server, each axis in `-1..=1`, in the world frame.
    ///
    /// Same mapping as [`InputState::step`] (held keys along the view's forward
    /// and right axes, plus up/down), normalised so diagonals are not faster.
    /// The proxy never torques the ship, so the ship's body frame stays the
    /// world frame and the view direction only picks the thrust direction.
    pub fn thrust(&self) -> [f32; 3] {
        let axis = |pos: Key, neg: Key| {
            f32::from(i8::from(self.is_held(pos)) - i8::from(self.is_held(neg)))
        };
        let (fwd, right, up) = (
            axis(Key::Forward, Key::Back),
            axis(Key::Right, Key::Left),
            axis(Key::Up, Key::Down),
        );
        let (f, r) = (self.forward(), self.right());
        let mut t = [0.0; 3];
        for i in 0..3 {
            t[i] = fwd * f[i] + right * r[i];
        }
        t[1] += up;
        let len = t.iter().map(|v| v * v).sum::<f32>().sqrt();
        if len > 1.0 {
            t = t.map(|v| v / len);
        }
        t
    }

    /// The camera for this input, relative to the scene's `base` camera.
    pub fn camera(&self, base: &CameraPose) -> CameraPose {
        CameraPose {
            position: WorldPos::from_f64_metres_around(base.position, self.offset),
            forward: self.forward(),
            up: [0.0, 1.0, 0.0],
            ..*base
        }
    }
}

/// Drain viewer input into `input` and, when connected, hand the resulting
/// thrust to the server link. Returns how many viewer events were applied.
///
/// With no viewer connected the held keys are released: nobody is left to send
/// the key-ups, so a ship would otherwise keep thrusting after the viewer left.
pub fn pump_input(
    client: &StreamClient,
    input: &mut InputState,
    link: Option<&ServerLink>,
) -> usize {
    let events = client.poll_input();
    for event in &events {
        input.apply(*event);
    }
    if client.viewer_count() == 0 {
        input.release_keys();
    }
    if let Some(link) = link {
        link.set_thrust(input.thrust());
    }
    events.len()
}

/// Build the scene as seen by `input`.
fn scene_for_input(mode: Mode, input: &InputState, width: u32, height: u32) -> Scene {
    if mode == Mode::Galaxy {
        let camera = input.camera(&crate::galaxy_camera(width, height));
        return crate::galaxy_scene_at(&camera);
    }
    // The demo scene is camera-relative around the origin: re-base it onto the
    // moved camera by shifting every instance and mesh by -offset.
    let mut scene = space_render::demo_scene(width, height);
    scene.camera = input.camera(&scene.camera);
    let off = input.offset.map(|v| v as f32);
    for inst in &mut scene.instances {
        for (c, o) in inst.center.iter_mut().zip(off) {
            *c -= o;
        }
    }
    for v in scene.meshes.iter_mut().flatten() {
        for (c, o) in v.iter_mut().zip(off) {
            *c -= o;
        }
    }
    scene
}

/// Render → JPEG → push, applying viewer input, until `stop` is set or
/// `max_frames` frames were streamed. While no viewer is connected nothing is
/// rendered. Returns the final input state.
///
/// With a `link` the camera offset is the server's replicated ship position;
/// without one the offset is integrated locally from the viewer's keys.
pub fn stream_loop(
    ctx: &RenderContext,
    config: &Config,
    client: &StreamClient,
    link: Option<&ServerLink>,
    stop: &AtomicBool,
    max_frames: Option<u64>,
) -> Result<InputState, RenderError> {
    let encoder = JpegEncoder::default();
    let mut flags = FLAG_KEYFRAME;
    if encoder.downscale() > 1 {
        flags |= FLAG_DOWNSCALED;
    }
    // Scale movement with the scene: the galaxy spans ~1e5 m, the demo ~6 m.
    let speed = if config.mode == Mode::Galaxy {
        5.0e3
    } else {
        3.0
    };
    let options = demo_options(config.quality);
    let mut input = initial_input(config.mode);
    let mut last = Instant::now();
    let mut streamed = 0u64;
    while !stop.load(Ordering::Relaxed) && max_frames.is_none_or(|m| streamed < m) {
        let start = Instant::now();
        if link.is_some_and(ServerLink::is_closed) {
            return Err(RenderError::new("lost the server connection"));
        }
        pump_input(client, &mut input, link);
        match link {
            Some(link) => input.offset = link.offset(),
            None => input.step(start.duration_since(last).as_secs_f32().min(0.25), speed),
        }
        last = start;
        if client.viewer_count() == 0 {
            std::thread::sleep(IDLE_POLL);
            continue;
        }
        let scene = scene_for_input(config.mode, &input, config.width, config.height);
        let out = ctx.render_offscreen(&scene, config.width, config.height, &options)?;
        let jpeg = encoder
            .encode(&out.pixels, out.width, out.height)
            .map_err(|e| RenderError::new(format!("stream: {e}")))?;
        client.push_jpeg(jpeg.bytes, flags);
        streamed += 1;
        if let Some(rest) = FRAME_BUDGET.checked_sub(start.elapsed()) {
            std::thread::sleep(rest);
        }
    }
    Ok(input)
}

/// Run the streaming client until killed: bind `ws://addr`, stream frames.
pub fn run_stream(config: &Config, addr: &str) -> Result<(), RenderError> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|e| RenderError::new(format!("tokio runtime: {e}")))?;
    let (server, client) = rt
        .block_on(StreamServer::bind(addr))
        .map_err(|e| RenderError::new(format!("stream bind {addr}: {e}")))?;
    let link = match config.server {
        Some(server) => {
            let link = rt
                .block_on(ServerLink::connect(server))
                .map_err(|e| RenderError::new(format!("server {server}: {e}")))?;
            eprintln!(
                "space-client: connected to {server} as session {}",
                link.replica().session
            );
            Some(link)
        }
        None => None,
    };
    let ctx = RenderContext::new(&ContextOptions::default())?;
    eprintln!(
        "space-client: device {:?}; streaming {}x{} on ws://{} (plain ws://, no TLS)",
        ctx.device_name(),
        config.width,
        config.height,
        server.local_addr()
    );
    let stop = AtomicBool::new(false);
    stream_loop(&ctx, config, &client, link.as_ref(), &stop, config.frames)?;
    drop(server);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{SinkExt, StreamExt};
    use space_stream::jpeg_dimensions;
    use space_stream::protocol::decode_frame;
    use std::sync::Arc;
    use tokio_tungstenite::tungstenite::Message;

    fn key(key: Key, down: bool) -> ViewerInput {
        ViewerInput::Key { key, down }
    }

    #[test]
    fn viewer_input_moves_and_turns_the_camera() {
        let mut s = InputState::default();
        s.apply(key(Key::Forward, true));
        s.step(1.0, 2.0);
        // Yaw 0 looks down -Z.
        assert!((s.offset[2] + 2.0).abs() < 1e-9 && s.offset[0].abs() < 1e-9);
        s.apply(key(Key::Forward, false));
        s.step(1.0, 2.0);
        assert!(
            (s.offset[2] + 2.0).abs() < 1e-9,
            "released key stops motion"
        );

        s.apply(ViewerInput::View {
            yaw: std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
        });
        let f = s.forward();
        assert!((f[0] + 1.0).abs() < 1e-6 && f[2].abs() < 1e-6, "{f:?}");
        s.apply(key(Key::Right, true));
        s.step(1.0, 1.0);
        // Turned left 90 degrees, so "right" is now -Z.
        assert!((s.offset[2] + 3.0).abs() < 1e-6, "{:?}", s.offset);

        s.apply(key(Key::Up, true));
        s.step(1.0, 1.0);
        assert!((s.offset[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn thrust_follows_held_keys_and_view_and_is_normalised() {
        let mut s = InputState::default();
        assert_eq!(s.thrust(), [0.0; 3]);
        s.apply(key(Key::Forward, true));
        let t = s.thrust();
        assert!((t[2] + 1.0).abs() < 1e-6 && t[0].abs() < 1e-6, "{t:?}");
        s.apply(key(Key::Right, true));
        s.apply(key(Key::Up, true));
        let len = s.thrust().iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((len - 1.0).abs() < 1e-5, "diagonal thrust is unit: {len}");
        s.apply(key(Key::Forward, false));
        s.apply(key(Key::Right, false));
        s.apply(key(Key::Up, false));
        assert_eq!(s.thrust(), [0.0; 3]);
    }

    /// A viewer that disconnects while holding a key must not leave the ship
    /// thrusting: once `viewer_count()` is 0 the pump releases every held key.
    #[test]
    fn held_keys_are_released_when_the_last_viewer_disconnects() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let (server, client) = StreamServer::bind("127.0.0.1:0").await.unwrap();
            let addr = server.local_addr();
            let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
            let (mut ws, _) = tokio_tungstenite::client_async(format!("ws://{addr}/"), tcp)
                .await
                .unwrap();
            ws.send(Message::Binary(key(Key::Forward, true).encode().into()))
                .await
                .unwrap();

            let mut input = InputState::default();
            let mut thrusting = false;
            for _ in 0..200 {
                pump_input(&client, &mut input, None);
                if input.thrust() != [0.0; 3] {
                    thrusting = true;
                    break;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            assert!(thrusting, "the held key never produced thrust");
            assert_eq!(client.viewer_count(), 1);
            pump_input(&client, &mut input, None);
            assert_ne!(input.thrust(), [0.0; 3], "still held while connected");

            // The viewer vanishes without sending a key-up.
            drop(ws);
            for _ in 0..200 {
                if client.viewer_count() == 0 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            assert_eq!(client.viewer_count(), 0, "viewer slot never freed");
            pump_input(&client, &mut input, None);
            assert_eq!(input.thrust(), [0.0; 3], "ship keeps thrusting");
            drop(server);
        });
    }

    #[test]
    fn view_pitch_is_clamped_and_pointer_recorded() {
        let mut s = InputState::default();
        s.apply(ViewerInput::View {
            yaw: 0.0,
            pitch: 9.0,
        });
        assert!(s.pitch < std::f32::consts::FRAC_PI_2);
        s.apply(ViewerInput::Pointer {
            x: 0.5,
            y: 0.25,
            buttons: 1,
        });
        assert_eq!(s.pointer, Some((0.5, 0.25, 1)));
    }

    #[test]
    fn demo_scene_is_rebased_onto_the_moved_camera() {
        let base = scene_for_input(Mode::Demo, &InputState::default(), 32, 32);
        let s = InputState {
            offset: [0.0, 0.0, -1.0],
            ..Default::default()
        };
        let moved = scene_for_input(Mode::Demo, &s, 32, 32);
        // Camera moved 1 m toward the sphere, so it is 1 m closer.
        assert!((moved.instances[0].center[2] - (base.instances[0].center[2] + 1.0)).abs() < 1e-6);
        assert_eq!(moved.meshes.len(), base.meshes.len());
    }

    /// End to end with a real render: the stream loop renders the demo scene,
    /// streams it to a WebSocket viewer, and applies that viewer's input. Skips
    /// loudly when no Vulkan device is available.
    #[test]
    fn stream_loop_delivers_frames_and_applies_viewer_input() {
        if let Err(e) = RenderContext::new(&ContextOptions::default()) {
            eprintln!("SKIP: no usable Vulkan device ({e}); needs VK_KHR_acceleration_structure");
            return;
        }
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let (server, client) = rt.block_on(StreamServer::bind("127.0.0.1:0")).unwrap();
        let client = Arc::new(client);
        let stop = Arc::new(AtomicBool::new(false));
        let config = Config {
            mode: Mode::Demo,
            width: 64,
            height: 48,
            out: None,
            quality: space_render::QualityPreset::Ci,
            stream: None,
            server: None,
            frames: None,
        };
        let worker = {
            let (client, stop) = (Arc::clone(&client), Arc::clone(&stop));
            std::thread::spawn(move || {
                let ctx = RenderContext::new(&ContextOptions::default()).unwrap();
                stream_loop(&ctx, &config, &client, None, &stop, None)
            })
        };

        let addr = server.local_addr();
        let frames = rt.block_on(async {
            let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
            let (mut ws, _) = tokio_tungstenite::client_async(format!("ws://{addr}/"), tcp)
                .await
                .unwrap();
            let view = ViewerInput::View {
                yaw: 0.5,
                pitch: 0.0,
            };
            ws.send(Message::Binary(view.encode().into()))
                .await
                .unwrap();
            let mut frames = Vec::new();
            while frames.len() < 3 {
                let msg = tokio::time::timeout(Duration::from_secs(60), ws.next())
                    .await
                    .expect("timed out waiting for a frame")
                    .unwrap()
                    .unwrap();
                if let Message::Binary(b) = msg {
                    frames.push(b.to_vec());
                }
            }
            frames
        });
        stop.store(true, Ordering::Relaxed);
        let input = worker.join().unwrap().unwrap();
        for f in &frames {
            let (header, jpeg) = decode_frame(f).unwrap();
            assert_eq!(header.flags, FLAG_KEYFRAME | FLAG_DOWNSCALED);
            assert_eq!(jpeg_dimensions(jpeg), Some((32, 24)), "half of 64x48");
        }
        assert!(
            (input.yaw - 0.5).abs() < 1e-6,
            "viewer input reached the proxy"
        );
        drop(server);
    }

    /// Loopback, no GPU: a real in-process shard (same boot wiring as
    /// `space-server`'s acceptance harness), the stream proxy's `ServerLink`
    /// connecting through the ordinary `space-net` client path, and a viewer
    /// that talks WebSocket to a real `space-stream` server. The viewer holds
    /// "forward"; the test asserts the *authoritative* ship moved along -Z on the
    /// server, and that the replicated state came back to drive the proxy's
    /// camera offset.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn viewer_input_reaches_the_server_and_replication_drives_the_scene() {
        use space_protocol::ShardId;
        use space_server::net::ShardNet;
        use space_server::{Shard, ShardConfig};
        use space_world::NodeRef;
        use std::net::SocketAddr;
        use tokio::sync::Mutex as AsyncMutex;
        use tokio_tungstenite::tungstenite::Message as WsMessage;

        // --- the shard
        let any = SocketAddr::from(([127, 0, 0, 1], 0));
        let net = Arc::new(ShardNet::bind(any).expect("bind"));
        let shard = Arc::new(AsyncMutex::new(Shard::new(ShardConfig::new(
            ShardId::new(1),
            net.local_addr().expect("addr"),
            NodeRef::new(64, [0, 0, 0]),
        ))));
        let serve = tokio::spawn({
            let (net, shard) = (Arc::clone(&net), Arc::clone(&shard));
            async move { net.serve(shard).await }
        });
        let ticker = tokio::spawn({
            let (net, shard) = (Arc::clone(&net), Arc::clone(&shard));
            async move {
                let mut clock =
                    tokio::time::interval(Duration::from_secs_f64(f64::from(space_sim::dt())));
                clock.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    clock.tick().await;
                    shard.lock().await.tick();
                    net.flush(&shard).await;
                }
            }
        });

        // --- the proxy client: server link + stream endpoint
        let link = ServerLink::connect(net.local_addr().expect("addr"))
            .await
            .expect("connect through the space-net client path");
        let (stream_server, stream_client) = StreamServer::bind("127.0.0.1:0").await.unwrap();

        // --- the viewer (browser stand-in): a WebSocket client sending a key
        let ws_addr = stream_server.local_addr();
        let tcp = tokio::net::TcpStream::connect(ws_addr).await.unwrap();
        let (mut ws, _) = tokio_tungstenite::client_async(format!("ws://{ws_addr}/"), tcp)
            .await
            .unwrap();
        let forward = ViewerInput::Key {
            key: Key::Forward,
            down: true,
        };
        ws.send(WsMessage::Binary(forward.encode().into()))
            .await
            .unwrap();

        // --- run the proxy's per-frame input pump until the ship has moved
        let mut input = InputState::default();
        let mut moved = false;
        for _ in 0..200 {
            pump_input(&stream_client, &mut input, Some(&link));
            if link.offset()[2] < -0.05 {
                moved = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(30)).await;
        }
        assert!(moved, "replicated state never showed the ship moving");

        // The server's own ship moved forward (-Z) and sideways not at all.
        let replica = link.replica();
        let ship = replica.ship.expect("our ship was replicated");
        let (server_ship, server_state) = {
            let shard = shard.lock().await;
            let session = shard.session_of(ship).expect("the ship has a session");
            assert_eq!(session, replica.session, "the shard's session is ours");
            (ship, shard.sim().ship_state(ship).expect("ship alive"))
        };
        assert_eq!(Some(server_ship), replica.ship);
        let start = replica.offset();
        assert!(start[2] < 0.0, "forward is -Z: {start:?}");
        assert!(start[0].abs() < 1e-3, "no sideways drift: {start:?}");
        assert!(
            server_state.vel[2] < 0.0,
            "server velocity: {server_state:?}"
        );

        link.close();
        ticker.abort();
        serve.abort();
        net.close();
        drop(stream_server);
    }
}
