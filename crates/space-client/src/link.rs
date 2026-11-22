//! The multiplayer seam: a normal `space-net` player connection.
//!
//! [`ServerLink::connect`] is the same client path a native client uses:
//! [`space_net::client_endpoint`] + [`space_net::connect`] with a
//! [`space_net::hello`] (so the `PROTOCOL_VERSION` gate is the transport's, not
//! reimplemented here). No server protocol is added: the link sends
//! [`Message::InputTick`] batches (datagrams) and reads the replicated
//! [`Message::Snapshot`] / [`Message::Delta`] frames the shard sends every
//! client.
//!
//! Three tasks run on the caller's tokio runtime: an input sender (one batch
//! per simulation tick), a reader for the reliable-stream frames (snapshots)
//! and a reader for datagram frames (deltas). They share one [`Replica`] — the
//! client's view of its *own* ship — which the render loop reads each frame.
//!
//! The server is authoritative: the link only sends thrust intent; the ship's
//! position comes back through replication ([`Replica::offset`]).

use std::net::SocketAddr;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use space_ecs::ComponentDef;
use space_net::{client_endpoint, connect, hello, Conn, NetError};
use space_protocol::{EntityGuid, EntitySnapshot, InputTick, Message, PlayerInput};
use space_sim::{Owner, ShipState};
use space_world::WorldPos;
use tokio::task::JoinHandle;

/// The TLS server name the shard's self-signed certificate is issued for
/// (`space_server::net::SERVER_NAME`; a test asserts they agree).
pub const SERVER_NAME: &str = "space.local";

/// The `player_token` the proxy presents in its hello.
pub const PLAYER_TOKEN: &str = "space-client-stream";

/// How many ticks ahead of the last replicated tick an input batch is tagged,
/// to land after the one-way delay (the server drops stale inputs and inputs
/// more than `MAX_INPUT_LEAD` ticks ahead).
pub const INPUT_LEAD: u64 = 3;

/// The client's replicated view of its own ship.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Replica {
    /// Session id from the server's `Welcome`; matches the ship's `Owner`.
    pub session: u64,
    /// Highest snapshot/delta tick applied (0 = none yet).
    pub tick: u64,
    /// Our ship's guid, once a frame carrying it has arrived.
    pub ship: Option<EntityGuid>,
    /// Our ship's latest replicated state.
    pub state: Option<ShipState>,
    /// Where the ship was in the first frame that showed it.
    origin: Option<WorldPos>,
    /// Frames (snapshots + deltas) applied.
    pub frames: u64,
}

impl Replica {
    fn new(session: u64) -> Self {
        Self {
            session,
            ..Self::default()
        }
    }

    /// Apply one server message; anything that is not a snapshot/delta is ignored.
    pub fn apply(&mut self, message: &Message) {
        let (tick, entities) = match message {
            Message::Snapshot(s) => (s.tick, &s.entities),
            Message::Delta(d) => (d.tick, &d.entities),
            _ => return,
        };
        // A late datagram must not rewind the view.
        if tick < self.tick {
            return;
        }
        self.tick = tick;
        self.frames += 1;
        for entity in entities {
            if let Some(state) = self.own_state(entity) {
                self.ship = Some(entity.guid);
                self.origin.get_or_insert(state.pos);
                self.state = Some(state);
            }
        }
    }

    /// The ship state in `entity`, if it is this session's ship.
    fn own_state(&self, entity: &EntitySnapshot) -> Option<ShipState> {
        let payload = |id| {
            entity
                .components
                .iter()
                .find(|(c, _)| *c == id)
                .map(|(_, p)| p.as_slice())
        };
        let owner = payload(<Owner as ComponentDef>::ID)?;
        let (owner, _) = space_protocol::codec::decode_raw::<u64>(owner).ok()?;
        if owner != self.session {
            return None;
        }
        space_sim::wire::decode_ship_state(payload(<ShipState as ComponentDef>::ID)?).ok()
    }

    /// The ship's displacement from where it first appeared, metres.
    #[must_use]
    pub fn offset(&self) -> [f64; 3] {
        match (self.origin, self.state) {
            (Some(origin), Some(s)) => WorldPos::delta(origin, s.pos),
            _ => [0.0; 3],
        }
    }
}

#[derive(Debug, Default)]
struct Shared {
    replica: Replica,
    /// Thrust intent, each axis in `-1..=1` (world frame; the ship is never
    /// torqued, so its body frame stays the world frame).
    thrust: [f32; 3],
    /// Input batches sent.
    seq: u64,
}

/// A live player connection to the authoritative shard.
#[derive(Debug)]
pub struct ServerLink {
    conn: Conn,
    shared: Arc<Mutex<Shared>>,
    tasks: Vec<JoinHandle<()>>,
}

impl ServerLink {
    /// Connect as an ordinary player and start the sender/reader tasks. Must be
    /// called inside a tokio runtime.
    ///
    /// # Errors
    ///
    /// [`NetError`] when the transport or the hello handshake fails (including
    /// [`NetError::VersionMismatch`]).
    pub async fn connect(addr: SocketAddr) -> Result<Self, NetError> {
        let endpoint = client_endpoint().map_err(|e| NetError::Handshake(e.to_string()))?;
        let (conn, welcome) = connect(&endpoint, addr, SERVER_NAME, &hello(PLAYER_TOKEN)).await?;
        let shared = Arc::new(Mutex::new(Shared {
            replica: Replica::new(welcome.session),
            ..Shared::default()
        }));
        let tasks = vec![
            tokio::spawn(send_inputs(conn.clone(), Arc::clone(&shared))),
            tokio::spawn(read_frames(conn.clone(), Arc::clone(&shared), false)),
            tokio::spawn(read_frames(conn.clone(), Arc::clone(&shared), true)),
        ];
        // `endpoint` may drop: the connection keeps the endpoint driver alive.
        Ok(Self {
            conn,
            shared,
            tasks,
        })
    }

    fn lock(&self) -> MutexGuard<'_, Shared> {
        self.shared.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Set the thrust intent sent with every following input batch.
    pub fn set_thrust(&self, thrust: [f32; 3]) {
        self.lock().thrust = thrust;
    }

    /// The ship's replicated displacement from its first-seen position, metres.
    #[must_use]
    pub fn offset(&self) -> [f64; 3] {
        self.lock().replica.offset()
    }

    /// A copy of the current replicated view.
    #[must_use]
    pub fn replica(&self) -> Replica {
        self.lock().replica.clone()
    }

    /// Whether the server connection is gone.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.conn.is_closed()
    }

    /// Close the connection and stop the tasks.
    pub fn close(&self) {
        self.conn.close(0, "client closing");
        for task in &self.tasks {
            task.abort();
        }
    }
}

impl Drop for ServerLink {
    fn drop(&mut self) {
        self.close();
    }
}

/// One [`InputTick`] per simulation tick, carrying the current thrust intent.
async fn send_inputs(conn: Conn, shared: Arc<Mutex<Shared>>) {
    let mut clock = tokio::time::interval(Duration::from_secs_f64(f64::from(space_sim::dt())));
    clock.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        clock.tick().await;
        if conn.is_closed() {
            return;
        }
        let batch = {
            let mut s = shared.lock().unwrap_or_else(|e| e.into_inner());
            // Nothing to target until the first frame names our ship and tick.
            let (Some(entity), tick) = (s.replica.ship, s.replica.tick + INPUT_LEAD) else {
                continue;
            };
            s.seq += 1;
            InputTick {
                tick,
                seq: s.seq,
                inputs: vec![PlayerInput {
                    entity,
                    thrust: s.thrust,
                    torque: [0.0; 3],
                    tick,
                }],
                ack_tick: s.replica.tick,
            }
        };
        if conn.send(&Message::InputTick(batch)).await.is_err() && conn.is_closed() {
            return;
        }
    }
}

/// Read frames from the reliable stream (snapshots) or datagrams (deltas) into
/// the replica until the connection closes.
async fn read_frames(conn: Conn, shared: Arc<Mutex<Shared>>, datagrams: bool) {
    loop {
        let message = if datagrams {
            conn.recv_datagram().await
        } else {
            conn.recv_stream().await
        };
        match message {
            Ok(Some(message)) => shared
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .replica
                .apply(&message),
            // Peer gone.
            Ok(None) => return,
            // A bad frame ends this reader only; the connection state is the
            // authority on whether to carry on.
            Err(e) => {
                log::warn!("space-client: server frame: {e}");
                if conn.is_closed() {
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::{pump_input, InputState};
    use futures_util::SinkExt;
    use space_protocol::ShardId;
    use space_server::net::ShardNet;
    use space_server::{Shard, ShardConfig};
    use space_stream::{Key, StreamServer, ViewerInput};
    use space_world::NodeRef;
    use tokio::sync::Mutex as AsyncMutex;
    use tokio_tungstenite::tungstenite::Message as WsMessage;

    #[test]
    fn server_name_matches_the_shards_certificate() {
        assert_eq!(SERVER_NAME, space_server::net::SERVER_NAME);
    }

    #[test]
    fn a_frame_without_our_ship_is_ignored() {
        let mut r = Replica::new(7);
        r.apply(&Message::Snapshot(space_protocol::Snapshot {
            tick: 5,
            shard: ShardId::new(1),
            epoch: 1,
            entities: vec![],
            input_ack: 0,
        }));
        assert_eq!((r.tick, r.ship, r.offset()), (5, None, [0.0; 3]));
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
        assert!(server_state.vel[2] < 0.0, "server velocity: {server_state:?}");

        link.close();
        ticker.abort();
        serve.abort();
        net.close();
        drop(stream_server);
    }
}
