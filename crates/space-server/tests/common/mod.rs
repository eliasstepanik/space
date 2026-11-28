//! The two-shard harness shared by this crate's integration tests (`mod common;`): shards
//! stepped by hand on one cluster clock, and a player as the client link plays it.
#![allow(dead_code)] // each test binary uses a different part

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use space_protocol::{
    EntityGuid, InputTick, Message, NeighbourStatus, PlayerInput, SceneClock, ShardId, ShardMap,
    ShardRegion,
};
use space_server::shard::{Destination, Shard};
use space_server::ShardConfig;
use space_sim::{ShipState, SimTick};
use space_world::{NodeRef, WorldPos};

/// The cluster's tick origin in these tests.
pub const ORIGIN: i64 = 1_763_900_000_000_000;
/// The tick every shard is put on the clock at.
pub const START: SimTick = 1_000;
/// How far ahead of the newest tick a client tags its inputs
/// (`space_client::link::INPUT_LEAD`).
pub const INPUT_LEAD: SimTick = 3;

pub fn listening(n: u16) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 7_200 + n)
}

/// `d` metres from the middle of the border between cell 0 and cell 1
/// (x = 512 m): `d[0] < 0` is shard 1's ground, `d[0] > 0` shard 2's.
pub fn at_border(d: [f64; 3]) -> WorldPos {
    WorldPos::from_f64_metres_around(WorldPos::new([1, 0, 0], [-512.0, 0.0, 0.0]), d)
}

pub fn border_x(p: WorldPos) -> f64 {
    WorldPos::delta(at_border([0.0; 3]), p)[0]
}

/// Shards stepped by hand on one cluster clock.
pub struct Cluster {
    pub shards: Vec<Shard>,
    /// Frames for clients: `(sending shard, session, message)`.
    pub frames: Vec<(ShardId, u64, Message)>,
    /// Shard-to-shard messages held back: `(deliver at round, from, to, message)`.
    pub held: Vec<(u64, ShardId, ShardId, Message)>,
    /// How many rounds a `HandoffCommit` is held back.
    pub commit_delay: u64,
    pub round: u64,
}

impl Cluster {
    /// Shard 1 owns cell 0, shard 2 cell 1; both on the cluster clock.
    pub fn pair() -> Self {
        Self::of(&[
            (1, NodeRef::new(64, [0, 0, 0])),
            (2, NodeRef::new(64, [1, 0, 0])),
        ])
    }

    /// One shard that owns both cells: where nothing is ever handed over.
    pub fn single() -> Self {
        Self::of(&[(1, NodeRef::new(63, [0, 0, 0]))])
    }

    pub fn of(regions: &[(u32, NodeRef)]) -> Self {
        let map = ShardMap::new(
            1,
            regions
                .iter()
                .map(|(id, node)| ShardRegion::new(*node, ShardId::new(*id), listening(*id as u16)))
                .collect(),
        );
        let shards = regions
            .iter()
            .map(|(id, node)| {
                let mut shard = Shard::new(ShardConfig::new(
                    ShardId::new(*id),
                    listening(*id as u16),
                    *node,
                ));
                shard.start_clock(SceneClock::cluster(ORIGIN), START);
                shard.set_shard_map(map.clone());
                shard
            })
            .collect();
        Self {
            shards,
            frames: Vec::new(),
            held: Vec::new(),
            commit_delay: 0,
            round: 0,
        }
    }

    pub fn shard(&self, id: ShardId) -> &Shard {
        self.shards.iter().find(|s| s.shard() == id).expect("shard")
    }

    pub fn shard_mut(&mut self, id: ShardId) -> &mut Shard {
        self.shards
            .iter_mut()
            .find(|s| s.shard() == id)
            .expect("shard")
    }

    /// Collect what every shard queued; deliver what is due between shards.
    pub fn deliver(&mut self) {
        for shard in &mut self.shards {
            let from = shard.shard();
            for out in shard.take_outbox() {
                match out.dest {
                    Destination::Client(session) => self.frames.push((from, session, out.message)),
                    Destination::Shard(to) => {
                        let delay = if matches!(out.message, Message::HandoffCommit(_)) {
                            self.commit_delay
                        } else {
                            0
                        };
                        self.held.push((self.round + delay, from, to, out.message));
                    }
                    Destination::Orchestrator => {}
                }
            }
        }
        let round = self.round;
        let (due, later): (Vec<_>, Vec<_>) = std::mem::take(&mut self.held)
            .into_iter()
            .partition(|(at, ..)| *at <= round);
        self.held = later;
        for (_, from, to, message) in due {
            self.shard_mut(to).peer_message(from, message);
        }
    }

    /// One tick of every shard, in id order, then one round of delivery.
    pub fn tick(&mut self) {
        for shard in &mut self.shards {
            shard.tick();
        }
        self.round += 1;
        self.deliver();
    }

    /// The shards that simulate `guid` now.
    pub fn simulators(&self, guid: EntityGuid) -> Vec<ShardId> {
        self.shards
            .iter()
            .filter(|s| s.sim().ship_state(guid).is_some())
            .map(Shard::shard)
            .collect()
    }

    pub fn state(&self, guid: EntityGuid) -> Option<ShipState> {
        self.shards.iter().find_map(|s| s.sim().ship_state(guid))
    }

    /// The frames sent to `(shard, session)`, draining them.
    pub fn frames_for(&mut self, shard: ShardId, session: u64) -> Vec<Message> {
        let (taken, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut self.frames)
            .into_iter()
            .partition(|f| f.0 == shard && f.1 == session);
        self.frames = keep;
        taken.into_iter().map(|(_, _, m)| m).collect()
    }
}

/// A player as the client link plays it: one session on the shard that
/// simulates its ship, a second on the shard a redirect names, and every
/// input — tagged [`INPUT_LEAD`] ticks ahead — sent to both.
pub struct Pilot {
    pub token: String,
    pub ship: EntityGuid,
    pub on: ShardId,
    pub session: u64,
    /// The shard a redirect named, and the session there.
    pub next: Option<(ShardId, u64)>,
    pub redirects: usize,
    /// The first frame of each shard after the first, by shard: `(tick,
    /// input_ack)`.
    pub first_frames: HashMap<ShardId, (u64, u64)>,
    /// How many turns the pilot takes to get through to the shard a
    /// redirect names (a slow connect); 0 = within the turn.
    pub connect_takes: u64,
    /// A redirect being followed: the shard, and the turns left.
    pub dialing: Option<(ShardId, u64)>,
    /// Whether a shard that has just been connected to is told the inputs
    /// already sent for ticks to come (`space_client::link::resend_tail`).
    pub resends: bool,
    /// The inputs sent last, oldest first.
    pub recent: Vec<PlayerInput>,
}

impl Pilot {
    pub fn join(c: &mut Cluster, token: &str) -> Self {
        let on = ShardId::new(1);
        let welcome = c
            .shard_mut(on)
            .join(&space_net::hello(token))
            .expect("shard 1 owns the spawn point");
        let ship = c.shard(on).ship_of(welcome.session).expect("a ship");
        Self {
            token: token.into(),
            ship,
            on,
            session: welcome.session,
            next: None,
            redirects: 0,
            first_frames: HashMap::new(),
            connect_takes: 0,
            dialing: None,
            resends: true,
            recent: Vec::new(),
        }
    }

    /// One client turn before the shards tick: read the frames, follow a
    /// redirect, notice a switch, and send the input for the tick
    /// [`INPUT_LEAD`] ahead.
    pub fn act(&mut self, c: &mut Cluster, thrust: [f32; 3]) {
        for m in c.frames_for(self.on, self.session) {
            let Message::Redirect(r) = m else { continue };
            if r.entity != self.ship || self.next.is_some() || self.dialing.is_some() {
                continue;
            }
            self.redirects += 1;
            self.dialing = Some((r.new_shard, self.connect_takes));
        }
        // The connection to the shard the redirect names is up (now, or some
        // turns after the redirect): hello, the inputs already sent for
        // ticks to come, then the status that opens the commit's gate.
        if let Some((to, left)) = self.dialing {
            if left > 0 {
                self.dialing = Some((to, left - 1));
            } else {
                self.dialing = None;
                let welcome = c
                    .shard_mut(to)
                    .join(&space_net::hello(&self.token))
                    .expect("the accepting shard admits the ship's player");
                let newest = c.shard(self.on).sim().tick();
                let tail: Vec<PlayerInput> = self
                    .recent
                    .iter()
                    .filter(|input| input.tick > newest)
                    .copied()
                    .collect();
                if self.resends && !tail.is_empty() {
                    let batch = InputTick {
                        tick: tail[tail.len() - 1].tick,
                        seq: 0,
                        inputs: tail,
                        ack_tick: 0,
                    };
                    c.shard_mut(to).apply_input(welcome.session, &batch);
                }
                c.shard_mut(to).client_status(
                    welcome.session,
                    NeighbourStatus {
                        entity: self.ship,
                        shard: to,
                        connected: true,
                    },
                );
                self.next = Some((to, welcome.session));
                // A slow connect is modelled with everything behind it fast:
                // the status is mirrored to the offering shard at once, so
                // its very next tick commits.
                if self.connect_takes > 0 {
                    c.deliver();
                }
            }
        }
        if let Some((to, session)) = self.next {
            // The switch: the first frame from the new shard.
            let frames = c.frames_for(to, session);
            let first = frames.iter().find_map(|m| match m {
                Message::Snapshot(s) => Some((s.tick, s.input_ack)),
                Message::Delta(d) => Some((d.tick, d.input_ack)),
                _ => None,
            });
            if let Some(first) = first {
                self.first_frames.insert(to, first);
                c.shard_mut(self.on).leave(self.session);
                self.on = to;
                self.session = session;
                self.next = None;
            }
        }
        // The cluster's tick is the same on every shard.
        let tick = c.shard(self.on).sim().tick() + INPUT_LEAD;
        let batch = InputTick {
            tick,
            seq: tick,
            inputs: vec![PlayerInput {
                entity: self.ship,
                thrust,
                torque: [0.0; 3],
                tick,
                fast: false,
            }],
            ack_tick: 0,
        };
        self.recent.extend(batch.inputs.iter().copied());
        let keep = self.recent.len().saturating_sub(12);
        self.recent.drain(..keep);
        c.shard_mut(self.on).apply_input(self.session, &batch);
        if let Some((to, session)) = self.next {
            c.shard_mut(to).apply_input(session, &batch);
        }
    }
}

/// The thrust a pilot holds at `tick`: ahead (+x), with a sideways and an
/// up-down part that change every few ticks — so an input applied a tick
/// early or late, twice or not at all changes where the ship ends up.
pub fn thrust_at(tick: SimTick) -> [f32; 3] {
    let phase = (tick / 3) % 5;
    [
        1.0,
        [0.0, 0.5, -0.5, 0.25, -1.0][phase as usize],
        [0.3, -0.3, 0.0, 1.0, -0.6][((tick / 4) % 5) as usize],
    ]
}

/// Fly the pilot's ship from 25 m before the border across it, `ticks`
/// ticks, and return its state after every tick (`None` where no shard
/// simulated it) with the shards that simulated it.
pub fn fly(
    c: &mut Cluster,
    pilot: &mut Pilot,
    ticks: SimTick,
    each: impl FnMut(&mut Cluster, &Pilot),
) -> Vec<(SimTick, Option<ShipState>, Vec<ShardId>)> {
    fly_with(c, pilot, ticks, thrust_at, each)
}

/// [`fly`] with the thrust a function of the tick an input is tagged for.
pub fn fly_with(
    c: &mut Cluster,
    pilot: &mut Pilot,
    ticks: SimTick,
    thrust: impl Fn(SimTick) -> [f32; 3],
    mut each: impl FnMut(&mut Cluster, &Pilot),
) -> Vec<(SimTick, Option<ShipState>, Vec<ShardId>)> {
    let mut start = ShipState::at_rest(at_border([-25.0, 0.0, 0.0]));
    start.vel = [12.0, 0.0, 0.0];
    assert!(c.shard_mut(ShardId::new(1)).set_state(pilot.ship, start));
    let mut out = Vec::new();
    for _ in 0..ticks {
        let tick = c.shards[0].sim().tick();
        pilot.act(c, thrust(tick + INPUT_LEAD));
        each(c, pilot);
        let tick = c.shards[0].sim().tick();
        out.push((tick, c.state(pilot.ship), c.simulators(pilot.ship)));
    }
    out
}

/// The same flight on one shard that owns both cells: where the ship is
/// after every tick when nothing is handed over.
pub fn reference(ticks: SimTick) -> Vec<(SimTick, Option<ShipState>, Vec<ShardId>)> {
    let mut c = Cluster::single();
    let mut pilot = Pilot::join(&mut c, "pilot");
    let flight = fly(&mut c, &mut pilot, ticks, |c, _| c.tick());
    assert_eq!(pilot.redirects, 0, "nothing to cross on one shard");
    flight
}

/// The tick a flight changed shard at: the first tick shard 2 simulated the
/// ship.
pub fn switch_tick(flight: &[(SimTick, Option<ShipState>, Vec<ShardId>)]) -> SimTick {
    flight
        .iter()
        .find(|(_, _, sims)| sims == &vec![ShardId::new(2)])
        .map(|(tick, ..)| *tick)
        .expect("shard 2 never simulated the ship")
}
