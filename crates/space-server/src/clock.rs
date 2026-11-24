//! The cluster's tick clock: what makes a tick number mean the same instant
//! on every shard.
//!
//! A shard used to count its ticks from its own start, on an interval that
//! skipped a tick it was late for. Two shards therefore never agreed on a
//! tick number, and a handoff had to translate between two clocks that also
//! drifted apart. Now every shard is given the **tick origin** of its cluster
//! — a wall-clock instant: the orchestrator's `ShardWelcome` carries it, a
//! standalone shard takes `--tick-origin` or the fixed default
//! ([`SceneClock::DEFAULT_ORIGIN_US`]) — and runs on the
//! [cluster clock](SceneClock::cluster) with that origin:
//!
//! * tick `n` is **due** at `origin + n / 30 s`;
//! * a shard reads the wall clock when it starts, to find where it is on
//!   that timeline ([`TickClock::new`]; an origin that lies in the future is
//!   refused: there is no tick to be at), and from then on counts with the
//!   monotonic clock — and **keeps comparing the two** (below);
//! * a shard that is **late** makes the missing ticks up at once, up to
//!   [`MAX_TICK_BURST`] of them; one that is further behind than that cannot
//!   keep up, **jumps** to the tick that is due ([`Shard::skip_to`]) and says
//!   so in its log (a few lines a minute) and in its counters
//!   ([`crate::ShardStats::skipped`]).
//!
//! It is the same clock a shard tells its clients as its
//! [`SceneClock`] — the mapping from ticks to the scene time the planets move
//! by — so there is one mechanism, not two: scene time *is* the cluster
//! clock's time of a tick, on every shard, and that mapping is never
//! re-anchored. What is adjusted is only *when* this shard steps a tick.
//!
//! # Keeping to the wall clock
//!
//! The wall clock and the monotonic clock do not stay together: the wall
//! clock is disciplined (NTP slews it, an operator steps it), the monotonic
//! clock stops while a Linux host is suspended, and the two readings drift.
//! A shard that read the wall clock once would keep the phase it had at its
//! start, and a replacement started hours later would have another: every
//! crossing between the two would be a commit that is a few ticks early or
//! late, for as long as both lived.
//!
//! So once a second ([`MEASURE_INTERVAL_US`]) the tick clock **measures**
//! how far it is from the wall clock ([`TickClock::offset_us`]: wall clock
//! minus tick clock, positive when the shard is behind; exported as
//! [`crate::ShardStats::clock_offset_us`] and in the shard's census line) and
//! corrects it:
//!
//! * within [`DEADBAND_US`] (2 ms): nothing — that is what a reading is
//!   worth;
//! * otherwise it **slews**: the tick clock runs at most [`SLEW_PPM`]
//!   (0.5 %) faster or slower until the offset is gone — a tick's worth of
//!   offset takes seven seconds. It never runs backwards and never steps by
//!   that: a tick is 0.17 ms shorter or longer, which no client can see;
//! * **behind by more than [`STEP_US`]** (half a second: the host was
//!   suspended, the wall clock was stepped forwards): it steps onto the wall
//!   clock at once, and the shard then deals with that as with any stall —
//!   a burst, or a jump over the ticks it cannot make up. Logged (a few
//!   lines a minute);
//! * **ahead by more than [`STEP_US`]** (the wall clock was stepped back):
//!   logged, and slewed like any other offset — a tick clock does not go
//!   back. That takes two hundred times the offset; see the limits below.
//!
//! # Limits
//!
//! All shards of a cluster run on one host (`docs/plan-sharding.md`,
//! decision 5), so they read the same wall clock, and with the above they
//! stay within a few milliseconds of it and of each other. What is **not**
//! solved: hosts whose wall clocks differ (the shards then differ by that);
//! the seconds a slew takes (two shards that are a tick apart handle a
//! crossing between them as a commit a tick early or late until it is
//! gone: 7 s per tick of offset); and a wall clock stepped **back** by more
//! than a few seconds, after which shards started before and after the step
//! disagree for a long time (200 s per second of step) — restart the
//! cluster.
//!
//! # The time source
//!
//! Both clocks are read through a [`TimeSource`]: [`SystemTime`] for a real
//! shard, [`ManualTime`] for a test that wants to decide what time it is —
//! several shards on one `ManualTime` are several shards on one host. The
//! loop is split accordingly: [`Pacer`] is everything but the waiting (and
//! is what tests drive), [`run`] is a `Pacer` plus a sleep.

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use space_net::{held_back_note, LogBudget};
use space_protocol::SceneClock;
use space_sim::SimTick;
use tokio::sync::Mutex;

use crate::net::ShardNet;
use crate::Shard;

/// The most ticks a late shard makes up in one go; further behind than this
/// it jumps (see the [module docs](self)). A quarter of a second: a stall
/// that long is a hiccup, anything longer is a shard that cannot keep up,
/// and replaying it in a burst would only make it later.
pub const MAX_TICK_BURST: SimTick = 8;

/// How often the tick clock is compared with the wall clock, µs.
pub const MEASURE_INTERVAL_US: i64 = 1_000_000;
/// An offset from the wall clock no larger than this is left alone, µs.
pub const DEADBAND_US: i64 = 2_000;
/// The most the tick clock runs fast or slow to remove an offset, parts per
/// million: 0.5 %.
pub const SLEW_PPM: i64 = 5_000;
/// A tick clock further **behind** the wall clock than this steps onto it;
/// one further ahead says so, µs. Half a second: what a handoff's catch-up
/// covers ([`crate::handoff::MAX_CATCH_UP`] ticks).
pub const STEP_US: i64 = 500_000;
/// A reading of the wall clock that took longer than this (the thread was
/// kept waiting in the middle of it) is not used, µs.
const MAX_READ_SPREAD_US: i64 = 1_000;

/// Log lines about jumps, and about clock offsets, per [`LINE_WINDOW`].
const LINES: u32 = 3;
/// The window of [`LINES`].
const LINE_WINDOW: Duration = Duration::from_secs(60);

/// The two clocks a [`TickClock`] reads (see the [module docs](self), "The
/// time source").
pub trait TimeSource: std::fmt::Debug + Send + Sync + 'static {
    /// The wall clock: µs of UTC since the Unix epoch. May be stepped.
    fn wall_us(&self) -> i64;
    /// A monotonic clock: µs since some fixed moment. Never goes back.
    fn mono_us(&self) -> i64;
}

/// The host's clocks.
#[derive(Debug)]
pub struct SystemTime {
    started: Instant,
}

impl SystemTime {
    /// The host's clocks.
    #[must_use]
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
        }
    }
}

impl Default for SystemTime {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeSource for SystemTime {
    fn wall_us(&self) -> i64 {
        crate::config::wall_clock_us()
    }

    fn mono_us(&self) -> i64 {
        i64::try_from(self.started.elapsed().as_micros()).unwrap_or(i64::MAX)
    }
}

/// Clocks that show what a test sets: time passes only when it is told to.
#[derive(Debug, Default)]
pub struct ManualTime {
    wall: AtomicI64,
    mono: AtomicI64,
}

impl ManualTime {
    /// Clocks whose wall clock reads `wall_us` now.
    #[must_use]
    pub fn at(wall_us: i64) -> Arc<Self> {
        Arc::new(Self {
            wall: AtomicI64::new(wall_us),
            mono: AtomicI64::new(0),
        })
    }

    /// `us` microseconds pass, on both clocks.
    pub fn advance(&self, us: i64) {
        let us = us.max(0);
        self.wall.fetch_add(us, Ordering::SeqCst);
        self.mono.fetch_add(us, Ordering::SeqCst);
    }

    /// The wall clock is stepped by `us` (either way); no time passes.
    pub fn step_wall(&self, us: i64) {
        self.wall.fetch_add(us, Ordering::SeqCst);
    }
}

impl TimeSource for ManualTime {
    fn wall_us(&self) -> i64 {
        self.wall.load(Ordering::SeqCst)
    }

    fn mono_us(&self) -> i64 {
        self.mono.load(Ordering::SeqCst)
    }
}

/// What one look at the wall clock did to a [`TickClock`]
/// ([`TickClock::observe`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Observed {
    /// Nothing to say: no measurement was due, or the offset is one the slew
    /// takes care of.
    Quiet,
    /// The tick clock was this far behind the wall clock, µs, and stepped
    /// onto it.
    Stepped(i64),
    /// The tick clock is this far **ahead** of the wall clock, µs — more
    /// than [`STEP_US`] — and is being slewed back.
    Ahead(i64),
}

/// Where a shard is on its cluster's tick timeline (see the
/// [module docs](self)).
#[derive(Debug, Clone)]
pub struct TickClock {
    /// Tick 0 at the origin, every tick `1/30 s` later.
    clock: SceneClock,
    time: Arc<dyn TimeSource>,
    /// The cluster's time is the monotonic clock plus this, µs.
    base_us: i64,
    /// Wall clock minus tick clock at the last measurement, µs.
    measured_us: i64,
    /// What of that is still to be slewed away, µs.
    owed_us: i64,
    /// The monotonic time of the last measurement.
    measured_at: i64,
    /// The monotonic time the slew was last applied up to.
    slewed_at: i64,
}

impl TickClock {
    /// The clock of the cluster whose tick origin is `origin_us`, on the
    /// host's clocks, placed on the timeline by the wall clock as it reads
    /// now.
    ///
    /// # Errors
    /// The origin lies in the future ([`SceneClock::origin_refusal`]): there
    /// is no tick to be at, and a shard that waited for one would admit
    /// players into a world that does not move.
    pub fn new(origin_us: i64) -> Result<Self, String> {
        Self::on(origin_us, Arc::new(SystemTime::new()))
    }

    /// [`Self::new`] on the clocks `time`.
    ///
    /// # Errors
    /// As [`Self::new`].
    pub fn on(origin_us: i64, time: Arc<dyn TimeSource>) -> Result<Self, String> {
        let (wall, mono) = (time.wall_us(), time.mono_us());
        if let Some(why) = SceneClock::origin_refusal(origin_us, wall) {
            return Err(why);
        }
        Ok(Self {
            clock: SceneClock::cluster(origin_us),
            time,
            base_us: wall.saturating_sub(mono),
            measured_us: 0,
            owed_us: 0,
            measured_at: mono,
            slewed_at: mono,
        })
    }

    /// The cluster clock itself: what the shard counts by and tells its
    /// clients.
    #[must_use]
    pub const fn scene_clock(&self) -> SceneClock {
        self.clock
    }

    /// The time on the cluster's timeline now, µs.
    #[must_use]
    pub fn time_us(&self) -> i64 {
        self.time.mono_us().saturating_add(self.base_us)
    }

    /// The tick that is due now: the last one whose time has come.
    #[must_use]
    pub fn tick_now(&self) -> SimTick {
        self.clock.tick_at(self.time_us())
    }

    /// How long from now until `tick` is due (zero if it is).
    #[must_use]
    pub fn until_due(&self, tick: SimTick) -> Duration {
        let us = self.clock.time_us_at(tick).saturating_sub(self.time_us());
        Duration::from_micros(u64::try_from(us).unwrap_or(0))
    }

    /// Wall clock minus tick clock as last measured, µs: positive when this
    /// shard is behind the wall clock (see the [module docs](self), "Keeping
    /// to the wall clock").
    #[must_use]
    pub const fn offset_us(&self) -> i64 {
        self.measured_us
    }

    /// Look at the wall clock if a measurement is due, and move the tick
    /// clock towards it by what the slew rate allows for the time since the
    /// last call (see the [module docs](self), "Keeping to the wall clock").
    /// Called every turn of the loop.
    pub fn observe(&mut self) -> Observed {
        let before = self.time.mono_us();
        let wall = self.time.wall_us();
        let mono = self.time.mono_us();
        let mut observed = Observed::Quiet;
        let due = mono.saturating_sub(self.measured_at) >= MEASURE_INTERVAL_US;
        if due && mono.saturating_sub(before) <= MAX_READ_SPREAD_US {
            self.measured_at = mono;
            let offset = wall.saturating_sub(mono.saturating_add(self.base_us));
            self.measured_us = offset;
            if offset > STEP_US {
                self.base_us = self.base_us.saturating_add(offset);
                self.owed_us = 0;
                self.slewed_at = mono;
                return Observed::Stepped(offset);
            }
            self.owed_us = if offset.abs() <= DEADBAND_US {
                0
            } else {
                offset
            };
            if offset < -STEP_US {
                observed = Observed::Ahead(-offset);
            }
        }
        // The slew: at most `SLEW_PPM` of the time that has passed, so the
        // tick clock's time never goes back and never jumps.
        let passed = mono.saturating_sub(self.slewed_at).max(0);
        self.slewed_at = mono;
        let allowed = passed.saturating_mul(SLEW_PPM) / 1_000_000;
        let step = self.owed_us.clamp(-allowed, allowed);
        self.base_us = self.base_us.saturating_add(step);
        self.owed_us -= step;
        observed
    }
}

/// What [`Shard::run_to`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Paced {
    /// Ticks stepped.
    pub stepped: SimTick,
    /// Ticks jumped over (not simulated).
    pub skipped: SimTick,
}

impl Shard {
    /// Put this shard on the cluster clock `clock` at `tick` — the tick that
    /// was due last: the next [`Shard::tick`] produces `tick + 1`. Its clients
    /// are told the clock as its scene clock. Called once, before the shard
    /// serves anybody.
    pub fn start_clock(&mut self, clock: SceneClock, tick: SimTick) {
        self.restart_ticks(tick);
        self.set_scene_clock(clock);
    }

    /// Bring this shard to tick `due`: step the ticks it is behind, or — if
    /// that is more than [`MAX_TICK_BURST`] — jump to the one before `due`
    /// and step that last one (see the [module docs](self)). A shard that is
    /// not behind does nothing.
    pub fn run_to(&mut self, due: SimTick) -> Paced {
        let mut paced = Paced {
            stepped: 0,
            skipped: 0,
        };
        let at = self.sim().tick();
        if due <= at {
            return paced;
        }
        if due - at > MAX_TICK_BURST {
            paced.skipped = due - 1 - at;
            self.skip_to(due - 1);
        }
        while self.sim().tick() < due {
            self.tick();
            paced.stepped += 1;
        }
        paced
    }
}

/// A shard's tick loop without the waiting: everything [`run`] does between
/// two sleeps (see the [module docs](self), "The time source").
#[derive(Debug)]
pub struct Pacer {
    clock: TickClock,
    skip_lines: LogBudget,
    offset_lines: LogBudget,
}

impl Pacer {
    /// A loop on `clock`.
    #[must_use]
    pub fn new(clock: TickClock) -> Self {
        Self {
            clock,
            skip_lines: LogBudget::new(LINES, LINE_WINDOW),
            offset_lines: LogBudget::new(LINES, LINE_WINDOW),
        }
    }

    /// The clock the loop runs on.
    #[must_use]
    pub const fn clock(&self) -> &TickClock {
        &self.clock
    }

    /// Put `shard` on the clock at the tick that is due now
    /// ([`Shard::start_clock`]). Once, before the first [`Pacer::turn`].
    pub fn start(&mut self, shard: &mut Shard) {
        shard.start_clock(self.clock.scene_clock(), self.clock.tick_now());
    }

    /// One turn: compare the clock with the wall clock, then bring `shard`
    /// to the tick that is due (a step, a burst, or a jump), saying in the
    /// log what a reader has to know.
    pub fn turn(&mut self, shard: &mut Shard) -> Paced {
        debug_stall(shard);
        let observed = self.clock.observe();
        shard.set_clock_offset(self.clock.offset_us());
        match observed {
            Observed::Quiet => {}
            Observed::Stepped(us) => {
                if let Some(held) = self.offset_lines.admit() {
                    eprintln!(
                        "space-server: {} was {:.3} s behind the wall clock (the host was \
                         suspended, or its clock was stepped) and moved its tick clock onto it{}",
                        shard.shard(),
                        us as f64 / 1e6,
                        held_back_note(held)
                    );
                }
            }
            Observed::Ahead(us) => {
                if let Some(held) = self.offset_lines.admit() {
                    eprintln!(
                        "space-server: {} is {:.3} s ahead of the wall clock (was it stepped \
                         back?); its tick clock runs {} % slow until that is gone, and until \
                         then its ticks are not those of a shard started now{}",
                        shard.shard(),
                        us as f64 / 1e6,
                        SLEW_PPM as f64 / 1e4,
                        held_back_note(held)
                    );
                }
            }
        }
        let paced = shard.run_to(self.clock.tick_now());
        if paced.skipped > 0 {
            if let Some(held) = self.skip_lines.admit() {
                eprintln!(
                    "space-server: {} could not keep up: it was {} ticks behind the cluster \
                     clock and jumped to tick {}{}",
                    shard.shard(),
                    paced.skipped + paced.stepped,
                    shard.sim().tick(),
                    held_back_note(held)
                );
            }
        }
        paced
    }

    /// How long to wait before the next turn: until `shard`'s next tick is
    /// due, and never longer than a tick (the clock may be moved meanwhile).
    #[must_use]
    pub fn until_next(&self, shard: &Shard) -> Duration {
        let tick = Duration::from_micros(1_000_000 / SceneClock::TICKS_PER_SECOND as u64);
        self.clock.until_due(shard.sim().tick() + 1).min(tick)
    }
}

/// A stall on request, for looking at what a suspended shard does
/// (`.warren/skills/space-visual-test/SKILL.md`): with `SPACE_DEBUG_STALL_DIR` set,
/// a file `stall-<shard id>` in that directory holding a number of
/// milliseconds makes this shard's loop stand still that long, once (the
/// file is removed first). The shard is held while it does, as a suspended
/// process is: nothing it is sent is handled meanwhile. Unset — always,
/// outside such a test — this is one environment lookup per turn.
fn debug_stall(shard: &Shard) {
    let Some(dir) = std::env::var_os("SPACE_DEBUG_STALL_DIR") else {
        return;
    };
    let path = std::path::Path::new(&dir).join(format!("stall-{}", shard.shard().0));
    let Ok(text) = std::fs::read_to_string(&path) else {
        return;
    };
    if std::fs::remove_file(&path).is_err() {
        return;
    }
    let ms = text.trim().parse::<u64>().unwrap_or(0).min(60_000);
    eprintln!(
        "space-server: {} stalls for {ms} ms at tick {} (SPACE_DEBUG_STALL_DIR)",
        shard.shard(),
        shard.sim().tick()
    );
    std::thread::sleep(Duration::from_millis(ms));
}

/// Run `shard` on the cluster clock for as long as the task lives: every tick
/// stepped when it is due, and what it queued sent (see the
/// [module docs](self)). `clock` is where the shard is on the timeline;
/// [`Shard::start_clock`] is called here, before the first tick.
pub async fn run(shard: Arc<Mutex<Shard>>, net: Arc<ShardNet>, clock: TickClock) {
    let mut pacer = Pacer::new(clock);
    pacer.start(&mut *shard.lock().await);
    loop {
        let paced = pacer.turn(&mut *shard.lock().await);
        if paced.stepped > 0 {
            net.flush(&shard).await;
        }
        let wait = pacer.until_next(&*shard.lock().await);
        tokio::time::sleep(wait).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ShardConfig;
    use space_protocol::ShardId;
    use space_world::NodeRef;

    const ORIGIN: i64 = 1_763_900_000_000_000;

    fn shard(id: u32) -> Shard {
        Shard::new(ShardConfig::new(
            ShardId::new(id),
            std::net::SocketAddr::from(([127, 0, 0, 1], 0)),
            NodeRef::new(64, [0, 0, 0]),
        ))
    }

    /// Two shards placed on the timeline seconds apart — by wall clocks read
    /// at different moments — name the same tick for the same instant, and a
    /// tick is due a thirtieth of a second after the one before it.
    ///
    /// Mutation this catches: counting from the shard's own start (the two
    /// would differ by the 4.2 s between their starts: 126 ticks), or placing
    /// a tick at the instant the wall clock was read instead of at the
    /// origin's grid.
    #[test]
    fn shards_started_seconds_apart_agree_on_the_tick_of_an_instant() {
        // A starts 10 s after the origin; B, a replacement, 4.2 s later.
        let time = ManualTime::at(ORIGIN + 10_000_000);
        let a = TickClock::on(ORIGIN, time.clone()).expect("a");
        // 10 s after the origin is tick 300, exactly; the tick after it is
        // due 33.3 ms later.
        assert_eq!(a.tick_now(), 300);
        assert_eq!(a.until_due(301), Duration::from_micros(33_333));
        time.advance(33_332);
        assert_eq!(a.tick_now(), 300);
        time.advance(1);
        assert_eq!(a.tick_now(), 301);
        time.advance(4_200_000 - 33_333);
        let b = TickClock::on(ORIGIN, time.clone()).expect("b");
        for us in [0i64, 17_000, 783_000, 56_234_000, 3_538_766_000] {
            time.advance(us);
            assert_eq!(a.tick_now(), b.tick_now(), "after another {us} µs");
            assert_eq!(a.until_due(a.tick_now() + 9), b.until_due(b.tick_now() + 9));
        }
    }

    /// A tick origin that has not come yet is refused, with a sentence that
    /// says so; one that has just come is tick 0.
    ///
    /// Mutation this catches: placing the clock anyway (`tick_at` saturates
    /// at 0, the first tick is due at the origin, and the shard's loop
    /// sleeps until then — an hour, a year — while the shard admits players).
    #[test]
    fn a_tick_origin_in_the_future_is_refused() {
        let time = ManualTime::at(ORIGIN);
        assert_eq!(
            TickClock::on(ORIGIN, time.clone()).expect("now").tick_now(),
            0
        );
        let why = TickClock::on(ORIGIN + 1, time.clone()).expect_err("a microsecond ahead");
        assert!(why.contains("future"), "{why}");
        let why = TickClock::on(ORIGIN + 3_600_000_000, time).expect_err("an hour ahead");
        assert!(why.contains("3600"), "says by how much: {why}");
    }

    /// The wall clock and the monotonic clock have come apart by 100 ms
    /// (three ticks) since shard A started; a replacement B started now is
    /// on the wall clock. A measures the offset, exports it, and slews: its
    /// time never goes back, never moves by more than the time that passed
    /// plus half a percent, and after the 20 s that takes the two are within
    /// the dead band (2 ms) of each other for good.
    ///
    /// Mutation this catches: reading the wall clock once (the two stay
    /// three ticks apart for good: every crossing between them is a commit
    /// three ticks late), correcting by a step (the bound on a turn's
    /// motion), a slew that overshoots or never ends (the last loop).
    #[test]
    fn a_clock_that_drifted_from_the_wall_clock_is_slewed_onto_it() {
        for drift in [100_000i64, -100_000] {
            let time = ManualTime::at(ORIGIN + 10_000_000);
            let mut a = TickClock::on(ORIGIN, time.clone()).expect("a");
            time.advance(5_000_000);
            assert_eq!(a.observe(), Observed::Quiet);
            assert_eq!(a.offset_us(), 0, "on the wall clock so far");
            time.step_wall(drift);
            let mut b = TickClock::on(ORIGIN, time.clone()).expect("b");
            time.advance(1_000_000);
            assert_eq!(a.observe(), Observed::Quiet, "an offset the slew handles");
            assert_eq!(a.offset_us(), drift, "measured and exported");
            let apart = a.tick_now().abs_diff(b.tick_now());
            assert!(
                (2..=3).contains(&apart),
                "the premise: three ticks apart, {apart}"
            );
            // 30 s in steps of a third of a tick.
            let step = 11_111;
            let mut last = a.time_us();
            let mut agreed_from = None;
            for i in 0..2_700 {
                time.advance(step);
                assert_eq!(a.observe(), Observed::Quiet);
                assert_eq!(b.observe(), Observed::Quiet);
                let now = a.time_us();
                let moved = now - last;
                assert!(
                    moved > 0 && (moved - step).abs() <= step * SLEW_PPM / 1_000_000 + 1,
                    "the tick clock moved {moved} µs in {step} µs"
                );
                last = now;
                if (b.time_us() - now).abs() <= DEADBAND_US {
                    agreed_from.get_or_insert(i);
                } else {
                    assert_eq!(agreed_from, None, "apart again at step {i}");
                }
            }
            // 100 ms at 5 ms a second: 20 s, 1800 steps.
            let from = agreed_from.expect("the two agree in the end");
            assert!((1_600..=1_900).contains(&from), "agreed from step {from}");
            assert!(a.offset_us().abs() <= DEADBAND_US, "{}", a.offset_us());
            assert_eq!(b.offset_us(), 0, "B never had anything to correct");
        }
    }

    /// A host that was suspended for three seconds (the wall clock went on,
    /// the monotonic clock did not): the tick clock steps onto the wall
    /// clock, and the shard jumps over the ticks it cannot make up. A wall
    /// clock stepped **back** three seconds is said and slewed: the tick
    /// clock never goes back.
    ///
    /// Mutation this catches: slewing a large offset forwards (ten minutes
    /// of a shard that is seconds behind every other), stepping backwards
    /// (a tick that is due twice), a turn that does not read the clock.
    #[test]
    fn a_suspended_host_steps_forward_and_a_clock_set_back_is_slewed() {
        let time = ManualTime::at(ORIGIN + 10_000_000);
        let mut pacer = Pacer::new(TickClock::on(ORIGIN, time.clone()).expect("clock"));
        let mut s = shard(1);
        pacer.start(&mut s);
        assert_eq!(s.sim().tick(), 300);
        for _ in 0..60 {
            time.advance(33_334);
            assert_eq!(pacer.turn(&mut s).stepped, 1);
        }
        assert_eq!(s.sim().tick(), 360);
        assert_eq!(s.stats().clock_offset_us, 0);
        // Suspended: three seconds of wall clock, none of monotonic.
        time.step_wall(3_000_000);
        time.advance(1_000_000);
        let paced = pacer.turn(&mut s);
        assert_eq!(
            (paced.stepped, paced.skipped),
            (1, 119),
            "a second of ticks and the three of the suspension, jumped over"
        );
        assert_eq!(s.sim().tick(), 360 + 120);
        assert_eq!(s.sim().tick(), pacer.clock().tick_now());
        assert_eq!(s.stats().clock_offset_us, 3_000_000, "what was measured");
        // Set back by three seconds: nothing jumps, time goes on.
        time.step_wall(-3_000_000);
        let mut clock = pacer.clock().clone();
        let before = clock.time_us();
        time.advance(1_000_000);
        assert_eq!(clock.observe(), Observed::Ahead(3_000_000));
        let after = clock.time_us();
        assert!(
            after > before && after - before <= 1_000_000,
            "{} µs in a second",
            after - before
        );
    }

    /// A shard is put on the clock at the tick that is due, tells its clients
    /// the cluster clock as its scene clock, makes up a few late ticks and
    /// jumps over a long stall — and its scene time is the cluster clock's
    /// time of its tick throughout, without a re-anchor.
    ///
    /// Mutation this catches: skipping late ticks (the shard falls behind the
    /// timeline for good: its tick 310 is no longer the cluster's 310),
    /// stepping every tick of a long stall (a shard that is seconds behind
    /// replays them all), a jump that moves the scene clock.
    #[test]
    fn a_shard_keeps_to_the_cluster_clock_through_late_ticks_and_a_stall() {
        let clock = SceneClock::cluster(ORIGIN);
        let mut s = shard(1);
        s.start_clock(clock, 300);
        assert_eq!(s.sim().tick(), 300);
        assert_eq!(s.scene_clock(), clock);
        assert_eq!(
            s.run_to(300),
            Paced {
                stepped: 0,
                skipped: 0
            },
            "not behind: nothing to do"
        );
        // On time, then three ticks late: made up.
        assert_eq!(s.run_to(301).stepped, 1);
        assert_eq!(
            s.run_to(304),
            Paced {
                stepped: 3,
                skipped: 0
            }
        );
        assert_eq!(s.sim().tick(), 304);
        assert_eq!(s.sim().scene_time(), clock.time_us_at(304));
        // Exactly the burst is still stepped; one more is a jump.
        assert_eq!(s.run_to(304 + MAX_TICK_BURST).skipped, 0);
        let before = s.sim().tick();
        let paced = s.run_to(before + MAX_TICK_BURST + 1);
        assert_eq!(
            paced,
            Paced {
                stepped: 1,
                skipped: MAX_TICK_BURST
            }
        );
        assert_eq!(s.sim().tick(), before + MAX_TICK_BURST + 1);
        assert_eq!(s.stats().skipped, MAX_TICK_BURST);
        // A three-second stall: one jump, one step, still on the timeline.
        let paced = s.run_to(1_000);
        assert_eq!(paced.stepped, 1);
        assert_eq!(s.sim().tick(), 1_000);
        assert_eq!(s.scene_clock(), clock, "never re-anchored");
        assert_eq!(s.sim().scene_time(), clock.time_us_at(1_000));
        // A second shard started later on the same clock is at the same tick
        // for the same instant.
        let mut late = shard(2);
        late.start_clock(clock, 990);
        late.run_to(1_000);
        assert_eq!(late.sim().tick(), s.sim().tick());
        assert_eq!(late.sim().scene_time(), s.sim().scene_time());
    }
}
