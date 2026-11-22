//! The multiplayer seam: a normal `space-net` player connection.
//!
//! [`ServerLink::connect`] is the same client path a native client uses:
//! [`space_net::client_endpoint_for`] + [`space_net::connect`] with a
//! [`space_net::hello`] (so the `PROTOCOL_VERSION` gate is the transport's, not
//! reimplemented here). No server protocol is added: the link sends
//! [`Message::InputTick`] batches (datagrams) and reads the replicated
//! [`Message::Snapshot`] / [`Message::Delta`] frames the shard sends every
//! client.
//!
//! Three kinds of task run on the caller's tokio runtime: an input sender (one
//! batch per simulation tick) and, per connection, a reader for the
//! reliable-stream frames (snapshots, redirects) and one for datagram frames
//! (deltas). They share one [`Replica`] — the client's view of its *own* ship —
//! which the render loop reads each frame.
//!
//! The server is authoritative: the link only sends thrust intent; the ship's
//! position comes back through replication ([`Replica::offset`]).
//!
//! # Shard transitions
//!
//! When the ship crosses into a region another shard owns, its shard starts a
//! handoff and tells the client with a [`Message::Redirect`]. The link then
//! does its half of the make-before-break exchange:
//!
//! 1. it opens a second connection, to the accepting shard named in the
//!    redirect's shard map, presenting the **same player token** (that token
//!    is how the accepting shard recognises the redirected player);
//! 2. it reports [`NeighbourStatus`]`{ connected: true }` there, which is the
//!    gate the shards' commit waits for;
//! 3. it keeps playing on the old shard — inputs and frames — until a frame on
//!    the new connection shows its ship, now owned by the new session. That
//!    frame is the switch: inputs go to the new shard from then on and the old
//!    connection is closed.
//!
//! The ship keeps its guid and its world position across the switch, so
//! [`Replica::offset`] — and with it the camera — is continuous: the
//! transition is invisible apart from [`Replica::handoffs`] ticking up. If the
//! new shard cannot be reached, or the handoff is aborted, nothing changes:
//! the old connection was never given up.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use space_ecs::ComponentDef;
use space_net::{client_endpoint_for, connect, hello, Conn, NetError};
use space_protocol::{
    EntityGuid, EntitySnapshot, InputTick, Message, NeighbourStatus, PlayerInput, Redirect, ShardId,
};
use space_sim::{Owner, ShipState};
use space_world::WorldPos;
use tokio::task::JoinHandle;

/// The TLS server name the shard's self-signed certificate is issued for
/// (`space_server::net::SERVER_NAME`; a test asserts they agree).
pub const SERVER_NAME: &str = "space.local";

/// Prefix of the `player_token` a link presents in its hello. The rest is
/// unique per link ([`unique_player_token`]): the token is how a shard tells
/// players apart during a handoff, so two clients must never share one.
pub const PLAYER_TOKEN_PREFIX: &str = "space-client-";

/// How many ticks ahead of the estimated server tick an input batch is tagged,
/// to land after the one-way delay (the server drops stale inputs and inputs
/// more than `MAX_INPUT_LEAD` ticks ahead).
pub const INPUT_LEAD: u64 = 3;

/// How far the link runs its estimate of the server's tick ahead of the last
/// frame it received, in ticks. Frames normally arrive every tick; this only
/// bridges gaps (a lost datagram, a stalled frame), so inputs keep landing on
/// ticks the server has not simulated yet instead of all being stale.
const MAX_TICK_EXTRAPOLATION: u64 = 30;

/// How long a connection opened for a redirect may wait for the handoff to
/// complete before it is abandoned (the shards time a handoff out long before
/// this; a later redirect then starts afresh).
const REDIRECT_PATIENCE: Duration = Duration::from_secs(10);

/// A fresh player token: the prefix plus 64 process-unique random bits.
#[must_use]
pub fn unique_player_token() -> String {
    use std::hash::{BuildHasher, Hasher};
    // `RandomState` is seeded from the OS per instance: no extra dependency.
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u32(std::process::id());
    format!("{PLAYER_TOKEN_PREFIX}{:016x}", hasher.finish())
}

/// The client's replicated view of its own ship.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Replica {
    /// Session id from the current shard's `Welcome`; matches the ship's `Owner`.
    pub session: u64,
    /// The shard the client is currently playing on.
    pub shard: Option<ShardId>,
    /// Highest snapshot/delta tick seen (0 = none yet).
    pub tick: u64,
    /// Our ship's guid, once a frame carrying it has arrived.
    pub ship: Option<EntityGuid>,
    /// Our ship's latest replicated state.
    pub state: Option<ShipState>,
    /// The tick `state` is from.
    state_tick: u64,
    /// Where the ship was in the first frame that showed it.
    origin: Option<WorldPos>,
    /// Frames (snapshots + deltas) applied.
    pub frames: u64,
    /// Completed shard transitions.
    pub handoffs: u64,
}

impl Replica {
    fn new(session: u64, shard: ShardId) -> Self {
        Self {
            session,
            shard: Some(shard),
            ..Self::default()
        }
    }

    /// Apply one server message; anything that is not a snapshot/delta is
    /// ignored. Returns whether the frame carried this session's ship.
    pub fn apply(&mut self, message: &Message) -> bool {
        let (tick, entities) = match message {
            Message::Snapshot(s) => (s.tick, &s.entities),
            Message::Delta(d) => (d.tick, &d.entities),
            _ => return false,
        };
        self.frames += 1;
        // Frames arrive on two paths (reliable snapshots, datagram deltas) and
        // may overtake each other, so the clock only moves forwards — but an
        // older frame is not thrown away wholesale: it may be the only one
        // that names our ship (the join snapshot arriving after a first delta).
        self.tick = self.tick.max(tick);
        let mut seen = false;
        for entity in entities {
            if let Some(state) = self.own_state(entity) {
                seen = true;
                self.ship = Some(entity.guid);
                self.origin.get_or_insert(state.pos);
                // A late frame must not rewind the ship.
                if tick >= self.state_tick || self.state.is_none() {
                    self.state = Some(state);
                    self.state_tick = tick;
                }
            }
        }
        seen
    }

    /// Whether `message` is a frame carrying a ship owned by `session`.
    fn frame_has_ship_of(message: &Message, session: u64) -> bool {
        let entities = match message {
            Message::Snapshot(s) => &s.entities,
            Message::Delta(d) => &d.entities,
            _ => return false,
        };
        entities
            .iter()
            .any(|entity| owner_of(entity) == Some(session) && ship_state_of(entity).is_some())
    }

    /// The ship state in `entity`, if it is this session's ship.
    fn own_state(&self, entity: &EntitySnapshot) -> Option<ShipState> {
        if owner_of(entity)? != self.session {
            return None;
        }
        ship_state_of(entity)
    }

    /// Continue this view on another shard after a handoff: a new session and
    /// a new tick domain (every shard runs its own clock), the same ship and
    /// the same origin.
    fn switch_to(&mut self, session: u64, shard: ShardId) {
        self.session = session;
        self.shard = Some(shard);
        self.tick = 0;
        self.state_tick = 0;
        self.handoffs += 1;
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

fn component(entity: &EntitySnapshot, id: space_ecs::ComponentId) -> Option<&[u8]> {
    entity
        .components
        .iter()
        .find(|(c, _)| *c == id)
        .map(|(_, p)| p.as_slice())
}

/// The session that owns `entity`, if it carries an `Owner`.
fn owner_of(entity: &EntitySnapshot) -> Option<u64> {
    let owner = component(entity, <Owner as ComponentDef>::ID)?;
    let (owner, _) = space_protocol::codec::decode_raw::<u64>(owner).ok()?;
    Some(owner)
}

fn ship_state_of(entity: &EntitySnapshot) -> Option<ShipState> {
    space_sim::wire::decode_ship_state(component(entity, <ShipState as ComponentDef>::ID)?).ok()
}

/// A connection opened for a redirect, waiting for the handoff to complete.
#[derive(Debug)]
struct Pending {
    conn: Conn,
    /// Identifies the connection's reader tasks.
    generation: u64,
    session: u64,
    shard: ShardId,
    opened: Instant,
}

#[derive(Debug)]
struct Shared {
    replica: Replica,
    /// When `replica.tick` last advanced: the base of the local tick estimate.
    tick_seen: Instant,
    /// Thrust intent, each axis in `-1..=1` (world frame; the ship is never
    /// torqued, so its body frame stays the world frame).
    thrust: [f32; 3],
    /// Input batches sent.
    seq: u64,
    /// The connection being played on: inputs go here, its frames are the view.
    active: Conn,
    /// Which reader generation belongs to `active`.
    active_generation: u64,
    next_generation: u64,
    /// The accepting shard's connection while a handoff is in flight.
    pending: Option<Pending>,
    /// A redirect is being followed right now (the connect is in flight).
    following: bool,
    /// Every task of the link, so `close` can stop them.
    tasks: Vec<JoinHandle<()>>,
}

impl Shared {
    /// Remember `task` so `close` can stop it, forgetting the ones that have
    /// already ended (a long session sees many short-lived tasks).
    fn track(&mut self, task: JoinHandle<()>) {
        self.tasks.retain(|t| !t.is_finished());
        self.tasks.push(task);
    }
}

type SharedRef = Arc<Mutex<Shared>>;

fn lock(shared: &SharedRef) -> MutexGuard<'_, Shared> {
    shared
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A live player connection to the authoritative world.
#[derive(Debug)]
pub struct ServerLink {
    shared: SharedRef,
    player_token: Arc<str>,
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
        let player_token: Arc<str> = unique_player_token().into();
        let endpoint = client_endpoint_for(addr).map_err(|e| NetError::Handshake(e.to_string()))?;
        let (conn, welcome) =
            connect(&endpoint, addr, SERVER_NAME, &hello(player_token.as_ref())).await?;
        // `endpoint` may drop: the connection keeps the endpoint driver alive.
        let shared = Arc::new(Mutex::new(Shared {
            replica: Replica::new(welcome.session, welcome.shard),
            tick_seen: Instant::now(),
            thrust: [0.0; 3],
            seq: 0,
            active: conn.clone(),
            active_generation: 0,
            next_generation: 1,
            pending: None,
            following: false,
            tasks: Vec::new(),
        }));
        let tasks = vec![
            tokio::spawn(send_inputs(Arc::clone(&shared))),
            spawn_reader(&shared, &player_token, conn.clone(), 0, false),
            spawn_reader(&shared, &player_token, conn, 0, true),
        ];
        lock(&shared).tasks = tasks;
        Ok(Self {
            shared,
            player_token,
        })
    }

    /// The player token this link presents to every shard.
    #[must_use]
    pub fn player_token(&self) -> &str {
        &self.player_token
    }

    /// Set the thrust intent sent with every following input batch.
    pub fn set_thrust(&self, thrust: [f32; 3]) {
        lock(&self.shared).thrust = thrust;
    }

    /// The ship's replicated displacement from its first-seen position, metres.
    #[must_use]
    pub fn offset(&self) -> [f64; 3] {
        lock(&self.shared).replica.offset()
    }

    /// A copy of the current replicated view.
    #[must_use]
    pub fn replica(&self) -> Replica {
        lock(&self.shared).replica.clone()
    }

    /// Whether the connection to the world is gone.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        lock(&self.shared).active.is_closed()
    }

    /// Close every connection and stop the tasks.
    pub fn close(&self) {
        let mut shared = lock(&self.shared);
        shared.active.close(0, "client closing");
        if let Some(pending) = shared.pending.take() {
            pending.conn.close(0, "client closing");
        }
        for task in shared.tasks.drain(..) {
            task.abort();
        }
    }
}

impl Drop for ServerLink {
    fn drop(&mut self) {
        self.close();
    }
}

/// Ticks of simulation time in `elapsed`.
fn ticks_in(elapsed: Duration) -> u64 {
    (elapsed.as_secs_f64() / f64::from(space_sim::dt())) as u64
}

/// The tick an input batch sent now should be tagged with: the last tick the
/// server reported, advanced by the time since (bounded), plus the lead.
fn input_tick(last_frame_tick: u64, since_frame: Duration) -> u64 {
    last_frame_tick + ticks_in(since_frame).min(MAX_TICK_EXTRAPOLATION) + INPUT_LEAD
}

/// One [`InputTick`] per simulation tick, carrying the current thrust intent,
/// to whichever connection is active.
async fn send_inputs(shared: SharedRef) {
    let mut clock = tokio::time::interval(Duration::from_secs_f64(f64::from(space_sim::dt())));
    clock.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        clock.tick().await;
        let (conn, batch) = {
            let mut s = lock(&shared);
            if s.active.is_closed() {
                return;
            }
            // Give up on a redirect whose handoff never completed.
            if s.pending
                .as_ref()
                .is_some_and(|p| p.opened.elapsed() > REDIRECT_PATIENCE)
            {
                if let Some(stale) = s.pending.take() {
                    stale.conn.close(0, "handoff did not complete");
                }
            }
            // Nothing to target until the first frame names our ship and tick.
            let Some(entity) = s.replica.ship else {
                continue;
            };
            let tick = input_tick(s.replica.tick, s.tick_seen.elapsed());
            s.seq += 1;
            let batch = InputTick {
                tick,
                seq: s.seq,
                inputs: vec![PlayerInput {
                    entity,
                    thrust: s.thrust,
                    torque: [0.0; 3],
                    tick,
                }],
                ack_tick: s.replica.tick,
            };
            (s.active.clone(), batch)
        };
        // A failed send on a live connection is a lost input, like any other.
        let _ = conn.send(&Message::InputTick(batch)).await;
    }
}

fn spawn_reader(
    shared: &SharedRef,
    player_token: &Arc<str>,
    conn: Conn,
    generation: u64,
    datagrams: bool,
) -> JoinHandle<()> {
    tokio::spawn(read_frames(
        Arc::clone(shared),
        Arc::clone(player_token),
        conn,
        generation,
        datagrams,
    ))
}

/// Read frames from one connection's reliable stream (snapshots, redirects) or
/// datagrams (deltas) until it closes.
async fn read_frames(
    shared: SharedRef,
    player_token: Arc<str>,
    conn: Conn,
    generation: u64,
    datagrams: bool,
) {
    loop {
        let message = if datagrams {
            conn.recv_datagram().await
        } else {
            conn.recv_stream().await
        };
        match message {
            Ok(Some(Message::Redirect(redirect))) => {
                follow_redirect(&shared, &player_token, generation, redirect);
            }
            Ok(Some(message)) => on_frame(&shared, generation, &message),
            // Peer gone.
            Ok(None) => return,
            // A bad frame ends nothing; the connection state is the authority
            // on whether to carry on.
            Err(e) => {
                log::warn!("space-client: server frame: {e}");
                if conn.is_closed() {
                    return;
                }
            }
        }
    }
}

/// Apply a frame from the connection with reader `generation`.
fn on_frame(shared: &SharedRef, generation: u64, message: &Message) {
    let mut s = lock(shared);
    if generation == s.active_generation {
        let before = s.replica.tick;
        s.replica.apply(message);
        if s.replica.tick != before {
            s.tick_seen = Instant::now();
        }
        return;
    }
    // A frame from the accepting shard of a handoff in flight. Until it shows
    // our ship under the new session the old shard is still the authority.
    let Some(pending) = s.pending.as_ref().filter(|p| p.generation == generation) else {
        return;
    };
    if !Replica::frame_has_ship_of(message, pending.session) {
        return;
    }
    // The switch: the new shard is simulating our ship.
    let Some(pending) = s.pending.take() else {
        return;
    };
    let retired = std::mem::replace(&mut s.active, pending.conn);
    s.active_generation = pending.generation;
    s.replica.switch_to(pending.session, pending.shard);
    s.replica.apply(message);
    s.tick_seen = Instant::now();
    log::info!(
        "space-client: now on shard {:?} (session {}), transition {}",
        pending.shard,
        pending.session,
        s.replica.handoffs
    );
    drop(s);
    retired.close(0, "handed off");
}

/// Start following `redirect` (from the active connection only): connect to
/// the accepting shard and report in. Never blocks the reader.
fn follow_redirect(
    shared: &SharedRef,
    player_token: &Arc<str>,
    generation: u64,
    redirect: Redirect,
) {
    let generation_for_new = {
        let mut s = lock(shared);
        if generation != s.active_generation || s.following {
            return;
        }
        if s.replica.shard == Some(redirect.new_shard) {
            return;
        }
        // Already connected there: the shards re-offered (the first attempt
        // timed out or was refused), so just report in again.
        if let Some(pending) = s.pending.as_ref().filter(|p| p.shard == redirect.new_shard) {
            let conn = pending.conn.clone();
            let status = neighbour_status(&redirect);
            let task = tokio::spawn(async move {
                let _ = conn.send_reliable(&status).await;
            });
            s.track(task);
            return;
        }
        s.following = true;
        let generation = s.next_generation;
        s.next_generation += 1;
        generation
    };
    let (shared_task, token) = (Arc::clone(shared), Arc::clone(player_token));
    let task = tokio::spawn(async move {
        let outcome = open_redirect(&shared_task, &token, generation_for_new, &redirect).await;
        lock(&shared_task).following = false;
        if let Err(reason) = outcome {
            // Make-before-break: the old connection was never given up, so a
            // failed redirect costs nothing; the shards abort the handoff.
            log::warn!(
                "space-client: could not follow the redirect to shard {:?}: {reason}",
                redirect.new_shard
            );
        }
    });
    lock(shared).track(task);
}

fn neighbour_status(redirect: &Redirect) -> Message {
    Message::NeighbourStatus(NeighbourStatus {
        entity: redirect.entity,
        shard: redirect.new_shard,
        connected: true,
    })
}

/// Connect to the shard `redirect` names, register the connection as pending
/// and tell that shard we are there.
async fn open_redirect(
    shared: &SharedRef,
    player_token: &Arc<str>,
    generation: u64,
    redirect: &Redirect,
) -> Result<(), String> {
    let addr = redirect
        .shard_map
        .regions
        .iter()
        .find(|region| region.shard == redirect.new_shard)
        .map(|region| region.endpoint)
        .filter(|addr| !addr.ip().is_unspecified() && addr.port() != 0)
        .ok_or("the shard map has no reachable endpoint for it")?;
    let endpoint = client_endpoint_for(addr).map_err(|e| e.to_string())?;
    let (conn, welcome) = connect(&endpoint, addr, SERVER_NAME, &hello(player_token.as_ref()))
        .await
        .map_err(|e| format!("{addr}: {e}"))?;
    {
        let mut s = lock(shared);
        if let Some(previous) = s.pending.replace(Pending {
            conn: conn.clone(),
            generation,
            session: welcome.session,
            shard: redirect.new_shard,
            opened: Instant::now(),
        }) {
            previous.conn.close(0, "superseded");
        }
        for datagrams in [false, true] {
            let reader = spawn_reader(shared, player_token, conn.clone(), generation, datagrams);
            s.track(reader);
        }
    }
    // The gate the shards' commit waits for.
    conn.send_reliable(&neighbour_status(redirect))
        .await
        .map_err(|e| format!("{addr}: reporting in: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use space_protocol::Snapshot;

    #[test]
    fn server_name_matches_the_shards_certificate() {
        assert_eq!(SERVER_NAME, space_server::net::SERVER_NAME);
    }

    fn frame(tick: u64, entities: Vec<EntitySnapshot>) -> Message {
        Message::Snapshot(Snapshot {
            tick,
            shard: ShardId::new(1),
            epoch: 1,
            entities,
            input_ack: 0,
        })
    }

    /// A replicated ship owned by `session`, `x` metres along +X.
    fn ship(guid: EntityGuid, session: u64, x: f64) -> EntitySnapshot {
        let state = ShipState {
            pos: WorldPos::from_f64_metres_around(WorldPos::ZERO, [x, 0.0, 0.0]),
            ..ShipState::IDENTITY
        };
        EntitySnapshot {
            guid,
            components: vec![
                (
                    <Owner as ComponentDef>::ID,
                    space_protocol::codec::encode_raw(&session),
                ),
                (
                    <ShipState as ComponentDef>::ID,
                    space_sim::wire::encode_ship_state(&state),
                ),
            ],
        }
    }

    #[test]
    fn a_frame_without_our_ship_is_ignored() {
        let mut r = Replica::new(7, ShardId::new(1));
        assert!(!r.apply(&frame(5, vec![])));
        assert_eq!((r.tick, r.ship, r.offset()), (5, None, [0.0; 3]));
    }

    #[test]
    fn tokens_are_unique_per_link() {
        let (a, b) = (unique_player_token(), unique_player_token());
        assert!(a.starts_with(PLAYER_TOKEN_PREFIX) && b.starts_with(PLAYER_TOKEN_PREFIX));
        assert_ne!(a, b, "two links must never present the same token");
    }

    /// The join snapshot may arrive after a first (empty) delta with a higher
    /// tick: it is still the frame that names our ship, so it must count — but
    /// an older frame must never move the ship backwards.
    #[test]
    fn an_older_frame_still_names_the_ship_but_never_rewinds_it() {
        let guid = EntityGuid::new(ShardId::new(1), 1);
        let mut r = Replica::new(7, ShardId::new(1));
        r.apply(&frame(12, vec![]));
        assert!(r.apply(&frame(10, vec![ship(guid, 7, 0.0)])));
        assert_eq!((r.ship, r.tick), (Some(guid), 12));
        r.apply(&frame(20, vec![ship(guid, 7, 5.0)]));
        r.apply(&frame(15, vec![ship(guid, 7, 2.0)]));
        assert!((r.offset()[0] - 5.0).abs() < 1e-3, "{:?}", r.offset());
        // Somebody else's ship is not ours.
        assert!(!r.apply(&frame(21, vec![ship(guid, 8, 99.0)])));
        assert!((r.offset()[0] - 5.0).abs() < 1e-3);
    }

    /// After a handoff the view continues on the new shard: new session, new
    /// tick domain (lower ticks are accepted), same ship, same origin — so the
    /// offset is continuous across the switch.
    #[test]
    fn switching_shards_keeps_the_ship_and_its_origin() {
        let guid = EntityGuid::new(ShardId::new(1), 1);
        let mut r = Replica::new(7, ShardId::new(1));
        r.apply(&frame(1000, vec![ship(guid, 7, 0.0)]));
        r.apply(&frame(1001, vec![ship(guid, 7, 500.0)]));
        // The accepting shard's frame: our ship, its session, its own clock.
        let from_new = frame(40, vec![ship(guid, 3, 512.5)]);
        assert!(!r.apply(&from_new), "not ours until the switch");
        assert!(Replica::frame_has_ship_of(&from_new, 3));
        r.switch_to(3, ShardId::new(2));
        assert!(r.apply(&from_new));
        assert_eq!(
            (r.session, r.shard, r.handoffs),
            (3, Some(ShardId::new(2)), 1)
        );
        assert_eq!((r.ship, r.tick), (Some(guid), 40));
        assert!((r.offset()[0] - 512.5).abs() < 1e-3, "{:?}", r.offset());
    }

    /// Inputs are tagged ahead of the server's clock even when frames stop
    /// arriving for a while, and the extrapolation is bounded.
    #[test]
    fn the_input_clock_keeps_running_between_frames() {
        let dt = f64::from(space_sim::dt());
        assert_eq!(input_tick(100, Duration::ZERO), 100 + INPUT_LEAD);
        assert_eq!(
            input_tick(100, Duration::from_secs_f64(dt * 6.5)),
            100 + 6 + INPUT_LEAD
        );
        assert_eq!(
            input_tick(100, Duration::from_secs(3600)),
            100 + MAX_TICK_EXTRAPOLATION + INPUT_LEAD
        );
    }
}
