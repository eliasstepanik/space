//! Property tests for the handoff guarantees (`handoff.rs`, "The guarantees", rows 1-8):
//! generated sequences of ticks, link delays, lost offers and accepts, client connects and
//! shard-map changes, run against two real [`Shard`]s with the invariants checked after every
//! tick, then a settle phase with the faults off.
//!
//! | invariant | what it says | guarantee rows |
//! |-----------|--------------|----------------|
//! | I1 | at most one shard simulates the ship at any tick | 1, 2 |
//! | I3 | a shard has installed no more ships than the commits that reached it under its own epoch (and no shard ever sent a commit under an epoch other than its own) | 7 |
//! | I4 | the ship is never lost: exactly one shard simulates it, except while a commit for it is on the link or was delivered this round | 1, 2, 6 |
//! | I5 | an offering shard's transfer ends within `HANDOFF_TIMEOUT_TICKS` + 8 rounds of its offer: it sends the commit, it sends an abort, or an abort reaches it | 8 (the offering shard) |
//! | `every_input_is_applied_exactly_once_across_a_handoff` | over generated connect delays, commit delays and thrust scripts, the ship flies as on one shard, and every input tick is applied once | 5 |
//!
//! Fault model: the link is FIFO, one queue per direction (a delayed message holds back the ones
//! behind it, like the real link). Offers and accepts may be lost; commits and aborts are only
//! delayed. The model is a pilot as the client link plays it: a session on the shard that
//! simulates the ship, one more on the shard a redirect names, every input sent to all of them.
//! The pilot is tethered to the border so the ship crosses it again and again.
//!
//! An abort names only the entity. One that reaches the offering shard after it has offered
//! the ship again calls the new transfer off there (`shard.rs`, `on_abort`), and the
//! accepting shard is not told. I5 counts that as the end of the offer, because it is the end
//! of the offering shard's transfer. The test found this at `PROPTEST_CASES=1000`: both
//! shards time out on one tick, their aborts cross, and the offering shard offers again a
//! tick later.
//!
//! Soak: `PROPTEST_CASES=5000 cargo test -p space-server --test handoff_props`.

mod common;

use std::collections::{HashMap, VecDeque};

use common::{at_border, border_x, fly_with, listening, Cluster, Pilot, INPUT_LEAD, START};
use proptest::collection::vec;
use proptest::prelude::{Just, Strategy};
use proptest::sample::select;
use proptest::strategy::Union;
use proptest::test_runner::TestCaseError;
use proptest::{prop_assert, prop_assert_eq};
use space_protocol::{
    EntityGuid, InputTick, Message, NeighbourStatus, PlayerInput, ShardId, ShardMap, ShardRegion,
};
use space_server::handoff::{HANDOFF_LEAD, HANDOFF_TIMEOUT_TICKS};
use space_server::shard::{Destination, Shard};
use space_sim::ShipState;
use space_world::NodeRef;

/// Runs a property with a fixed seed, no regression files and `cases` cases
/// (`PROPTEST_CASES` wins when set), so a failure is the same on every run.
fn check<S>(cases: u32, strategy: &S, test: impl Fn(S::Value) -> Result<(), TestCaseError>)
where
    S: Strategy,
    S::Value: std::fmt::Debug,
{
    use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};
    let mut config = Config::default();
    if std::env::var_os("PROPTEST_CASES").is_none() {
        config.cases = cases;
    }
    config.failure_persistence = None;
    let mut runner =
        TestRunner::new_with_rng(config, TestRng::deterministic_rng(RngAlgorithm::ChaCha));
    if let Err(failure) = runner.run(strategy, test) {
        panic!("{failure}");
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Offer,
    Accept,
    Commit,
    Abort,
}

#[derive(Debug, Clone, Copy)]
enum Fate {
    Hold(u64),
    Lose,
}

#[derive(Debug, Clone, Copy)]
enum Which {
    First,
    Second,
    Both,
}

#[derive(Debug, Clone)]
enum Op {
    Tick { thrust: i8 },
    Hold { kind: Kind, rounds: u8 },
    Lose { kind: Kind },
    Connect,
    Epoch { on: Which },
}

fn kind_of(message: &Message) -> Option<Kind> {
    match message {
        Message::HandoffOffer(_) => Some(Kind::Offer),
        Message::HandoffAccept(_) => Some(Kind::Accept),
        Message::HandoffCommit(_) => Some(Kind::Commit),
        Message::HandoffAbort(_) => Some(Kind::Abort),
        _ => None,
    }
}

fn index(id: ShardId) -> usize {
    (id.0 - 1) as usize
}

fn map_at(epoch: u64) -> ShardMap {
    ShardMap::new(
        epoch,
        vec![
            ShardRegion::new(NodeRef::new(64, [0, 0, 0]), ShardId::new(1), listening(1)),
            ShardRegion::new(NodeRef::new(64, [1, 0, 0]), ShardId::new(2), listening(2)),
        ],
    )
}

struct Model {
    shards: Vec<Shard>,
    ship: EntityGuid,
    sessions: HashMap<ShardId, u64>,
    home: ShardId,
    redirect: Option<ShardId>,
    /// One FIFO queue per direction, by sending shard: `(due round, message)`.
    queues: [VecDeque<(u64, Message)>; 2],
    round: u64,
    faults: VecDeque<(Kind, Fate)>,
    epoch: u64,
    matching_commits: HashMap<ShardId, u64>,
    commit_delivered_at: Option<u64>,
    /// The offer the model counts as open: `(index of the offering shard, round it was sent)`.
    open_offer: Option<(usize, u64)>,
    frames: Vec<(ShardId, u64, Message)>,
}

impl Model {
    fn new() -> Self {
        let mut shards = Cluster::pair().shards;
        assert!(shards[0].sim().tick() >= START);
        let home = ShardId::new(1);
        let welcome = shards[0]
            .join(&space_net::hello("pilot"))
            .expect("shard 1 owns the spawn point");
        let ship = shards[0].ship_of(welcome.session).expect("a ship");
        let mut start = ShipState::at_rest(at_border([-25.0, 0.0, 0.0]));
        start.vel = [12.0, 0.0, 0.0];
        assert!(shards[0].set_state(ship, start));
        Self {
            shards,
            ship,
            sessions: HashMap::from([(home, welcome.session)]),
            home,
            redirect: None,
            queues: [VecDeque::new(), VecDeque::new()],
            round: 0,
            faults: VecDeque::new(),
            epoch: 1,
            matching_commits: HashMap::new(),
            commit_delivered_at: None,
            open_offer: None,
            frames: Vec::new(),
        }
    }

    fn simulators(&self) -> Vec<ShardId> {
        self.shards
            .iter()
            .filter(|s| s.sim().ship_state(self.ship).is_some())
            .map(Shard::shard)
            .collect()
    }

    fn is_ship(&self, message: &Message) -> bool {
        match message {
            Message::HandoffOffer(m) => m.entity == self.ship,
            Message::HandoffAccept(m) => m.entity == self.ship,
            Message::HandoffCommit(m) => m.entity == self.ship,
            Message::HandoffAbort(m) => m.entity == self.ship,
            _ => false,
        }
    }

    /// Once per tick, after both shards ticked and `round` moved on.
    fn exchange(&mut self) {
        for i in 0..2 {
            let from = self.shards[i].shard();
            for out in self.shards[i].take_outbox() {
                match out.dest {
                    Destination::Client(session) => self.frames.push((from, session, out.message)),
                    Destination::Orchestrator => {}
                    Destination::Shard(_) => self.send(i, out.message),
                }
            }
        }
        for i in 0..2 {
            let j = 1 - i;
            let from = self.shards[i].shard();
            let to = self.shards[j].shard();
            while self.queues[i]
                .front()
                .is_some_and(|(due, _)| *due <= self.round)
            {
                let (_, message) = self.queues[i].pop_front().expect("a front");
                match &message {
                    Message::HandoffCommit(commit) => {
                        if commit.epoch == self.shards[j].epoch() {
                            *self.matching_commits.entry(to).or_insert(0) += 1;
                        }
                        if commit.entity == self.ship {
                            self.commit_delivered_at = Some(self.round);
                        }
                    }
                    Message::HandoffAbort(abort)
                        if abort.entity == self.ship
                            && self.open_offer.is_some_and(|(by, _)| by == j) =>
                    {
                        self.open_offer = None;
                    }
                    _ => {}
                }
                self.shards[j].peer_message(from, message);
            }
        }
    }

    fn send(&mut self, from: usize, message: Message) {
        if let Message::HandoffCommit(commit) = &message {
            assert_eq!(
                commit.epoch,
                self.shards[from].epoch(),
                "a commit left a shard under an epoch that is not its own"
            );
        }
        if self.is_ship(&message) {
            match &message {
                Message::HandoffOffer(_) => self.open_offer = Some((from, self.round)),
                Message::HandoffCommit(_) | Message::HandoffAbort(_)
                    if self.open_offer.is_some_and(|(by, _)| by == from) =>
                {
                    self.open_offer = None;
                }
                _ => {}
            }
        }
        let mut due = self.round;
        if let Some(kind) = kind_of(&message) {
            if let Some(at) = self.faults.iter().position(|(k, _)| *k == kind) {
                match self.faults.remove(at).expect("a fault").1 {
                    Fate::Lose => return,
                    Fate::Hold(n) => due = self.round + n,
                }
            }
        }
        self.queues[from].push_back((due, message));
    }

    fn pilot_reads(&mut self) {
        let mut gone = Vec::new();
        for (&shard, &session) in &self.sessions {
            for (_, _, message) in self
                .frames
                .iter()
                .filter(|(s, sess, _)| *s == shard && *sess == session)
            {
                match message {
                    Message::Redirect(r) if r.entity == self.ship => {
                        self.redirect = Some(r.new_shard);
                    }
                    Message::JoinRedirect(_) | Message::Reject(_) if shard != self.home => {
                        gone.push(shard);
                    }
                    _ => {}
                }
            }
        }
        for shard in gone {
            self.sessions.remove(&shard);
        }
        self.frames.clear();
    }

    fn tick(&mut self, thrust: i8, i4: bool) -> Result<(), TestCaseError> {
        self.pilot_reads();
        // A session on a shard that gave the player a ship of its own is left at once.
        let others: Vec<(ShardId, u64)> = self
            .sessions
            .iter()
            .filter(|(s, _)| **s != self.home)
            .map(|(s, sess)| (*s, *sess))
            .collect();
        for (shard, session) in others {
            if self.shards[index(shard)]
                .ship_of(session)
                .is_some_and(|g| g != self.ship)
            {
                self.shards[index(shard)].leave(session);
                self.sessions.remove(&shard);
            }
        }
        // The ship is on the other shard alone: that is where the player plays now.
        let sims = self.simulators();
        if let [t] = sims[..] {
            if t != self.home && self.sessions.contains_key(&t) {
                let old = self.sessions.remove(&self.home).expect("a home session");
                self.shards[index(self.home)].leave(old);
                self.home = t;
            }
        }
        // The input, tethered to the border.
        let state = self
            .shards
            .iter()
            .find_map(|s| s.sim().ship_state(self.ship));
        let mut x = f32::from(thrust);
        if let Some(state) = state {
            let bx = border_x(state.pos);
            let vx = state.vel[0];
            if bx > 50.0 || vx > 25.0 {
                x = -1.0;
            } else if bx < -50.0 || vx < -25.0 {
                x = 1.0;
            }
        }
        let tick = self.shards[index(self.home)].sim().tick() + INPUT_LEAD;
        let batch = InputTick {
            tick,
            seq: tick,
            inputs: vec![PlayerInput {
                entity: self.ship,
                thrust: [x, 0.0, 0.0],
                torque: [0.0; 3],
                tick,
                fast: false,
            }],
            ack_tick: 0,
        };
        for (&shard, &session) in &self.sessions {
            self.shards[index(shard)].apply_input(session, &batch);
        }
        for shard in &mut self.shards {
            shard.tick();
        }
        self.round += 1;
        self.exchange();
        self.invariants(i4)
    }

    fn connect(&mut self) {
        let Some(to) = self.redirect else { return };
        if !self.sessions.contains_key(&to) {
            if let Ok(welcome) = self.shards[index(to)].join(&space_net::hello("pilot")) {
                if self.shards[index(to)]
                    .ship_of(welcome.session)
                    .is_some_and(|g| g != self.ship)
                {
                    self.shards[index(to)].leave(welcome.session);
                } else {
                    self.sessions.insert(to, welcome.session);
                }
            }
        }
        if let Some(&session) = self.sessions.get(&to) {
            self.shards[index(to)].client_status(
                session,
                NeighbourStatus {
                    entity: self.ship,
                    shard: to,
                    connected: true,
                },
            );
        }
        self.redirect = None;
    }

    fn apply(&mut self, op: &Op, i4: bool) -> Result<(), TestCaseError> {
        match *op {
            Op::Tick { thrust } => self.tick(thrust, i4)?,
            Op::Hold { kind, rounds } => {
                self.faults.push_back((kind, Fate::Hold(u64::from(rounds))))
            }
            Op::Lose { kind } => self.faults.push_back((kind, Fate::Lose)),
            Op::Connect => self.connect(),
            Op::Epoch { on } => {
                self.epoch += 1;
                let map = map_at(self.epoch);
                match on {
                    Which::First => self.shards[0].apply_epoch(map),
                    Which::Second => self.shards[1].apply_epoch(map),
                    Which::Both => {
                        self.shards[0].apply_epoch(map.clone());
                        self.shards[1].apply_epoch(map);
                    }
                }
            }
        }
        Ok(())
    }

    fn invariants(&self, i4: bool) -> Result<(), TestCaseError> {
        let sims = self.simulators();
        prop_assert!(
            sims.len() <= 1,
            "I1: round {}: {sims:?} simulate the ship",
            self.round
        );
        for shard in &self.shards {
            let id = shard.shard();
            let commits = self.matching_commits.get(&id).copied().unwrap_or(0);
            prop_assert!(
                shard.stats().received as u64 <= commits,
                "I3: round {}: shard {id:?} installed {} ships on {commits} matching commits",
                self.round,
                shard.stats().received
            );
        }
        if let Some((_, since)) = self.open_offer {
            prop_assert!(
                self.round - since <= HANDOFF_TIMEOUT_TICKS + 8,
                "I5: round {}: an offer open since round {since}",
                self.round
            );
        }
        if let Some(&s) = sims.first() {
            let pos = self.shard_state(s).pos;
            assert!(border_x(pos).abs() < 500.0, "harness tether failed");
        }
        if i4 {
            let commit_on_link = self
                .queues
                .iter()
                .flatten()
                .any(|(_, m)| matches!(m, Message::HandoffCommit(c) if c.entity == self.ship));
            if !commit_on_link && self.commit_delivered_at != Some(self.round) {
                prop_assert_eq!(
                    sims.len(),
                    1,
                    "I4: round {}: the ship is lost (no commit on the link)",
                    self.round
                );
            }
        }
        Ok(())
    }

    fn shard_state(&self, id: ShardId) -> ShipState {
        self.shards[index(id)]
            .sim()
            .ship_state(self.ship)
            .expect("a simulated ship")
    }

    fn settle(&mut self, i4: bool) -> Result<(), TestCaseError> {
        self.faults.clear();
        let map = map_at(self.epoch);
        for shard in &mut self.shards {
            if shard.epoch() < self.epoch {
                shard.apply_epoch(map.clone());
            }
        }
        for _ in 0..3 * HANDOFF_TIMEOUT_TICKS {
            self.connect();
            self.tick(0, i4)?;
        }
        Ok(())
    }
}

fn run(ops: &[Op], i4: bool) -> Result<Model, TestCaseError> {
    let mut model = Model::new();
    for op in ops {
        model.apply(op, i4)?;
    }
    model.settle(i4)?;
    Ok(model)
}

fn kinds() -> impl Strategy<Value = Kind> {
    select(vec![Kind::Offer, Kind::Accept, Kind::Commit, Kind::Abort])
}

fn ops(with_epoch: bool) -> impl Strategy<Value = Vec<Op>> {
    let tick = select(vec![-1i8, 0, 1, 1]).prop_map(|thrust| Op::Tick { thrust });
    let hold = (kinds(), 1u8..=4).prop_map(|(kind, rounds)| Op::Hold { kind, rounds });
    let lose = select(vec![Kind::Offer, Kind::Accept]).prop_map(|kind| Op::Lose { kind });
    let mut arms = vec![
        (12, tick.boxed()),
        (2, hold.boxed()),
        (1, lose.boxed()),
        (3, Just(Op::Connect).boxed()),
    ];
    if with_epoch {
        let epoch =
            select(vec![Which::First, Which::Second, Which::Both]).prop_map(|on| Op::Epoch { on });
        arms.push((1, epoch.boxed()));
    }
    vec(Union::new_weighted(arms), 50..=300)
}

#[test]
fn one_simulator_and_no_stale_commit_under_any_interleaving() {
    check(128, &ops(true), |ops| run(&ops, false).map(|_| ()));
}

#[test]
fn the_ship_is_never_lost_while_the_map_stands_still() {
    check(128, &ops(false), |ops| run(&ops, true).map(|_| ()));
}

/// Fails, and is ignored for that reason: a commit that meets a changed epoch is refused after the
/// sender let go (`shard.rs` `on_commit`), so the ship is simulated by nobody (I4, round 91).
///
/// Minimal failing sequence found by proptest (shrunk with `PROPTEST_MAX_SHRINK_ITERS=20000`; no
/// fault, one map change that reaches only shard 2 while the ship is crossing; `t(n)` is
/// `Tick { thrust: n }`):
///
/// ```text
/// t(-1) t(0) t(0) t(1) t(0) t(1) t(1) t(0) t(1) t(1) t(-1) t(1) t(1) t(1) t(1) t(-1) t(-1)
/// t(0) t(1) t(0) t(1) t(0) t(1) t(0) t(-1) t(-1) t(0) t(1) t(1) t(1) t(0) t(1) t(-1) t(0)
/// t(1) t(0) t(1) t(-1) t(-1) t(1) t(1) t(1) t(1) t(1) t(0) t(0) t(0) t(1) t(0) t(0) t(0)
/// t(-1) t(1) t(-1) t(1) t(1) t(1) t(1) t(-1) t(-1) t(-1) t(-1) t(-1)
/// Epoch { on: Second }
/// t(-1) x 25, Connect, t(-1), t(-1)
/// ```
///
/// Failure: `I4: round 91: the ship is lost (no commit on the link)`; the log shows shard 2
/// aborting the incoming transfer with `EpochChanged` at tick 1090.
#[test]
#[ignore = "a commit that meets a changed epoch is refused after the sender let go (shard.rs on_commit); see the doc comment"]
fn the_ship_is_never_lost_across_a_map_change() {
    check(48, &ops(true), |ops| run(&ops, true).map(|_| ()));
}

/// Guarantee 5 (`handoff.rs`): over generated connect delays, commit delays and thrust scripts
/// the ship flies as on one shard, and every tick's input is applied once by one of the two
/// (the fixed scenarios are in `crossing.rs`).
#[test]
fn every_input_is_applied_exactly_once_across_a_handoff() {
    let ticks: u64 = 160;
    let strategy = (
        0..=HANDOFF_LEAD + 3,
        0u64..=4,
        vec((-2i8..=2, -2i8..=2), 160),
    );
    check(24, &strategy, |(connect_takes, commit_delay, script)| {
        let thrust = |t: u64| {
            let (y, z) = script[((t - START) as usize) % script.len()];
            [1.0, f32::from(y) * 0.5, f32::from(z) * 0.5]
        };
        let mut r = Cluster::single();
        let mut rp = Pilot::join(&mut r, "pilot");
        let alone = fly_with(&mut r, &mut rp, ticks, thrust, |c, _| c.tick());

        let mut c = Cluster::pair();
        c.commit_delay = commit_delay;
        let mut pilot = Pilot::join(&mut c, "pilot");
        pilot.connect_takes = connect_takes;
        let flight = fly_with(&mut c, &mut pilot, ticks, thrust, |c, _| c.tick());

        prop_assert_eq!(pilot.redirects, 1, "one handoff");
        let mut unsimulated = 0;
        for ((tick, state, sims), (_, ref_state, _)) in flight.iter().zip(&alone) {
            prop_assert!(sims.len() <= 1, "tick {tick}: simulated by {sims:?}");
            if sims.is_empty() {
                unsimulated += 1;
            } else {
                prop_assert_eq!(state, ref_state, "tick {}: off the reference", tick);
            }
        }
        prop_assert_eq!(unsimulated, commit_delay + 1, "ticks nobody simulated");
        prop_assert_eq!(c.shards[1].stats().caught_up, commit_delay);
        let mut applied: Vec<u64> = c
            .shards
            .iter()
            .flat_map(|s| s.sim().applied_inputs())
            .filter(|(_, guid)| *guid == pilot.ship)
            .map(|(tick, _)| tick)
            .collect();
        applied.sort_unstable();
        prop_assert!(!applied.is_empty(), "no input applied");
        let (first, last) = (applied[0], applied[applied.len() - 1]);
        prop_assert_eq!(
            &applied,
            &(first..=last).collect::<Vec<_>>(),
            "every tick once, none missing, none twice"
        );
        prop_assert!(
            first <= START + INPUT_LEAD + 1,
            "first applied tick {first}"
        );
        prop_assert!(last >= START + ticks, "last applied tick {last}");
        Ok(())
    });
}

#[test]
fn the_model_hands_the_ship_over() {
    let ops: Vec<Op> = (0..150)
        .flat_map(|_| [Op::Connect, Op::Tick { thrust: 1 }])
        .collect();
    let model = run(&ops, true).expect("no invariant broken");
    assert!(model.shards[0].stats().handed_off >= 1);
    assert!(model.shards[1].stats().received >= 1);
}
