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
//! * a shard reads the wall clock **once**, when it starts, to find where it
//!   is on that timeline ([`TickClock::new`]), and from then on counts with
//!   the monotonic clock — a wall clock that is stepped while the shard runs
//!   does not move its ticks;
//! * a shard that is **late** makes the missing ticks up at once, up to
//!   [`MAX_TICK_BURST`] of them; one that is further behind than that cannot
//!   keep up, **jumps** to the tick that is due ([`Shard::skip_to`]) and says
//!   so in its log (a few lines a minute) and in its counters
//!   ([`crate::ShardStats::skipped`]).
//!
//! It is the same clock a shard tells its clients as its
//! [`SceneClock`] — the mapping from ticks to the scene time the planets move
//! by — so there is one mechanism, not two: scene time *is* the cluster
//! clock's time of a tick, on every shard, and nothing is ever re-anchored.
//! ([`Shard::sync_scene_clock`] remains for a shard driven by a loop of its
//! own, which counts from 0: the MCP server, tests that step by hand.)
//!
//! All shards of a cluster run on one host (`docs/plan-sharding.md`,
//! decision 5), so they read the same wall clock. Shards on hosts whose
//! clocks differ would disagree by that difference; that is not solved here.

use std::sync::Arc;
use std::time::{Duration, Instant};

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

/// Log lines about jumps per [`SKIP_LINE_WINDOW`].
const SKIP_LINES: u32 = 3;
/// The window of [`SKIP_LINES`].
const SKIP_LINE_WINDOW: Duration = Duration::from_secs(60);

/// Where a shard is on its cluster's tick timeline (see the
/// [module docs](self)).
#[derive(Debug, Clone, Copy)]
pub struct TickClock {
    /// Tick 0 at the origin, every tick `1/30 s` later.
    clock: SceneClock,
    /// The wall clock (µs of UTC since the Unix epoch) at `at`.
    wall_us: i64,
    /// The instant the wall clock was read.
    at: Instant,
}

impl TickClock {
    /// The clock of the cluster whose tick origin is `origin_us`, placed on
    /// the timeline by the wall clock as it reads now.
    #[must_use]
    pub fn new(origin_us: i64) -> Self {
        Self::placed(origin_us, crate::config::wall_clock_us(), Instant::now())
    }

    /// [`Self::new`] with the wall clock's reading given: at `at` it read
    /// `wall_us`.
    #[must_use]
    pub fn placed(origin_us: i64, wall_us: i64, at: Instant) -> Self {
        Self {
            clock: SceneClock::cluster(origin_us),
            wall_us,
            at,
        }
    }

    /// The cluster clock itself: what the shard counts by and tells its
    /// clients.
    #[must_use]
    pub const fn scene_clock(&self) -> SceneClock {
        self.clock
    }

    /// The time on the cluster's timeline at `now`, µs.
    #[must_use]
    pub fn time_us(&self, now: Instant) -> i64 {
        let since =
            i64::try_from(now.saturating_duration_since(self.at).as_micros()).unwrap_or(i64::MAX);
        self.wall_us.saturating_add(since)
    }

    /// The tick that is due at `now`: the last one whose time has come.
    #[must_use]
    pub fn tick_at(&self, now: Instant) -> SimTick {
        self.clock.tick_at(self.time_us(now))
    }

    /// When `tick` is due.
    #[must_use]
    pub fn due(&self, tick: SimTick) -> Instant {
        let us = self.clock.time_us_at(tick).saturating_sub(self.wall_us);
        self.at + Duration::from_micros(u64::try_from(us).unwrap_or(0))
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

/// Run `shard` on the cluster clock for as long as the task lives: every tick
/// stepped when it is due, and what it queued sent (see the
/// [module docs](self)). `clock` is where the shard is on the timeline;
/// [`Shard::start_clock`] is called here, before the first tick.
pub async fn run(shard: Arc<Mutex<Shard>>, net: Arc<ShardNet>, clock: TickClock) {
    {
        let mut shard = shard.lock().await;
        let now = clock.tick_at(Instant::now());
        shard.start_clock(clock.scene_clock(), now);
    }
    let mut said = (Instant::now(), 0u32, 0u64);
    loop {
        let paced = {
            let mut shard = shard.lock().await;
            let paced = shard.run_to(clock.tick_at(Instant::now()));
            if paced.skipped > 0 {
                if said.0.elapsed() >= SKIP_LINE_WINDOW {
                    said = (Instant::now(), 0, said.2);
                }
                if said.1 < SKIP_LINES {
                    said.1 += 1;
                    let more = if said.2 == 0 {
                        String::new()
                    } else {
                        format!(" ({} more jumps were not written)", said.2)
                    };
                    said.2 = 0;
                    eprintln!(
                        "space-server: {} could not keep up: it was {} ticks behind the cluster \
                         clock and jumped to tick {}{more}",
                        shard.shard(),
                        paced.skipped + paced.stepped,
                        shard.sim().tick()
                    );
                } else {
                    said.2 += 1;
                }
            }
            paced
        };
        if paced.stepped > 0 {
            net.flush(&shard).await;
        }
        let next = shard.lock().await.sim().tick() + 1;
        tokio::time::sleep_until(tokio::time::Instant::from_std(clock.due(next))).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ShardConfig;
    use space_protocol::ShardId;
    use space_world::NodeRef;

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
        let origin = 1_763_900_000_000_000;
        let t0 = Instant::now();
        // A starts 10 s after the origin; B, a replacement, 4.2 s later.
        let a = TickClock::placed(origin, origin + 10_000_000, t0);
        let b = TickClock::placed(
            origin,
            origin + 14_200_000,
            t0 + Duration::from_micros(4_200_000),
        );
        for ms in [4_200u64, 4_217, 5_000, 61_234, 3_600_000] {
            let at = t0 + Duration::from_millis(ms);
            assert_eq!(a.tick_at(at), b.tick_at(at), "{ms} ms after A's start");
        }
        // 10 s after the origin is tick 300, exactly; the tick after it is
        // due 33.3 ms later, on both.
        assert_eq!(a.tick_at(t0), 300);
        assert_eq!(a.tick_at(t0 + Duration::from_micros(33_332)), 300);
        assert_eq!(a.tick_at(t0 + Duration::from_micros(33_333)), 301);
        let due = a.due(301).duration_since(t0);
        assert!(
            (33_333..=33_334).contains(&due.as_micros()),
            "tick 301 is due {due:?} after tick 300"
        );
        let b_due = b.due(450).duration_since(t0);
        assert_eq!(b_due, a.due(450).duration_since(t0), "the same instant");
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
        let origin = 1_763_900_000_000_000;
        let clock = SceneClock::cluster(origin);
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
