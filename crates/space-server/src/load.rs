//! What a shard tells its orchestrator about its load (`docs/plan-sharding.md`
//! §3): once a second a [`LoadReport`] with its players, the ships and bodies
//! it simulates, how long its ticks took over that second against the budget
//! of [`TICK_BUDGET_MS`], how many ticks it had to jump, and the **hot cell**
//! — the cell holding most of its players, which is where the orchestrator
//! carves a region for a new shard when this one is loaded.
//!
//! [`LoadWindow`] is the arithmetic, and pure: tick times in, a report out.
//! The shard feeds it the time each [`crate::shard::Shard::tick`] took and
//! asks for a report every [`REPORT_INTERVAL_TICKS`]
//! ([`crate::shard::Shard::report_load`] switches the reporting on; an
//! orchestrated shard does, a standalone one has nobody to tell).

use space_protocol::{LoadReport, ShardId};
use space_sim::SimTick;

/// How often a shard reports, in ticks (1 s).
pub const REPORT_INTERVAL_TICKS: SimTick = space_sim::TICK_HZ as SimTick;

/// The time a tick may take: 33.3 ms at 30 Hz.
pub const TICK_BUDGET_MS: f32 = 1_000.0 / space_sim::TICK_HZ as f32;

/// The most tick times a window holds. A window is a second of ticks; after a
/// burst (a shard that fell behind makes up to 8 ticks up at once) it holds
/// more, and anything past this many replaces the oldest.
pub const WINDOW: usize = 4 * REPORT_INTERVAL_TICKS as usize;

/// The tick times since the last report.
#[derive(Debug, Clone, Default)]
pub struct LoadWindow {
    /// Tick times of the window, µs, oldest first.
    times_us: Vec<u32>,
}

/// The mean, the 95th percentile and the worst of a window's tick times, ms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TickTimes {
    /// The mean.
    pub mean_ms: f32,
    /// The 95th percentile (nearest rank: of 30 ticks, the 29th in order).
    pub p95_ms: f32,
    /// The worst.
    pub max_ms: f32,
}

impl LoadWindow {
    /// Note that a tick took `us` microseconds.
    pub fn note(&mut self, us: u64) {
        if self.times_us.len() == WINDOW {
            self.times_us.remove(0);
        }
        self.times_us.push(u32::try_from(us).unwrap_or(u32::MAX));
    }

    /// How many ticks the window holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.times_us.len()
    }

    /// Whether the window holds no tick.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.times_us.is_empty()
    }

    /// The window's tick times (all 0 for an empty window), and an empty
    /// window afterwards.
    pub fn take(&mut self) -> TickTimes {
        let mut times = std::mem::take(&mut self.times_us);
        if times.is_empty() {
            return TickTimes {
                mean_ms: 0.0,
                p95_ms: 0.0,
                max_ms: 0.0,
            };
        }
        times.sort_unstable();
        let ms = |us: u32| us as f32 / 1_000.0;
        let sum: u64 = times.iter().map(|us| u64::from(*us)).sum();
        // Nearest rank: the smallest value with at least 95 % at or below it.
        let rank = (times.len() * 95).div_ceil(100).max(1);
        TickTimes {
            mean_ms: (sum as f64 / times.len() as f64 / 1_000.0) as f32,
            p95_ms: ms(times[rank - 1]),
            max_ms: ms(times[times.len() - 1]),
        }
    }
}

/// The cell holding most of `cells` (one entry per player: the cell its ship
/// is in) and how many it holds; ties go to the smallest cell index, so two
/// shards — and two runs — agree. `None` without players.
#[must_use]
pub fn hot_cell(cells: &[[i64; 3]]) -> Option<([i64; 3], u32)> {
    let mut sorted = cells.to_vec();
    sorted.sort_unstable();
    let mut best: Option<([i64; 3], u32)> = None;
    let mut i = 0;
    while i < sorted.len() {
        let run = sorted[i..].iter().take_while(|c| **c == sorted[i]).count();
        let count = u32::try_from(run).unwrap_or(u32::MAX);
        if best.is_none_or(|(_, most)| count > most) {
            best = Some((sorted[i], count));
        }
        i += run;
    }
    best
}

/// Put a report together. `players` is the number of connected clients,
/// `player_cells` the cell of every player's ship, `entities` the ships and
/// bodies simulated here, `skipped` the ticks jumped since the start.
#[must_use]
pub fn report(
    shard: ShardId,
    players: u32,
    player_cells: &[[i64; 3]],
    entities: u32,
    skipped: u64,
    times: TickTimes,
) -> LoadReport {
    let (hot_cell, hot_cell_players) = hot_cell(player_cells).unwrap_or(([0; 3], 0));
    LoadReport {
        shard,
        players,
        tick_p95_ms: times.p95_ms,
        // The plan's metric at the default capacity; the orchestrator works
        // its own out from the parts (its capacity is a setting).
        load: (players as f32 / crate::shard::MAX_PLAYERS_PER_SHARD as f32)
            .max(times.p95_ms / TICK_BUDGET_MS),
        entities,
        tick_mean_ms: times.mean_ms,
        tick_max_ms: times.max_ms,
        skipped,
        hot_cell,
        hot_cell_players,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mean, 95th percentile and worst of a second of ticks, and the window
    /// starts over with every report.
    ///
    /// Mutation this catches: taking the percentile at index `len * 95 / 100`
    /// (the worst tick of 30 instead of the 29th), or not emptying the window
    /// (one slow tick would be reported for ever).
    #[test]
    fn a_window_is_summed_up_and_starts_over() {
        let mut window = LoadWindow::default();
        // 28 ticks of 2 ms, one of 10 ms, one of 40 ms.
        for _ in 0..28 {
            window.note(2_000);
        }
        window.note(40_000);
        window.note(10_000);
        assert_eq!(window.len(), 30);
        let times = window.take();
        assert!(
            (times.mean_ms - (56.0 + 50.0) / 30.0).abs() < 1e-3,
            "{times:?}"
        );
        assert_eq!(times.p95_ms, 10.0, "the 29th of 30 in order");
        assert_eq!(times.max_ms, 40.0);
        assert!(window.is_empty());
        let empty = window.take();
        assert_eq!((empty.mean_ms, empty.p95_ms, empty.max_ms), (0.0, 0.0, 0.0));
        // One tick: all three are that tick.
        window.note(5_000);
        let one = window.take();
        assert_eq!((one.mean_ms, one.p95_ms, one.max_ms), (5.0, 5.0, 5.0));
    }

    /// The window never grows past its bound.
    ///
    /// Mutation this catches: no bound (a shard whose reports are not taken
    /// would collect tick times for ever).
    #[test]
    fn a_window_nobody_reads_is_bounded() {
        let mut window = LoadWindow::default();
        for i in 0..(3 * WINDOW as u64) {
            window.note(i);
        }
        assert_eq!(window.len(), WINDOW);
        assert_eq!(
            window.take().max_ms,
            (3 * WINDOW as u64 - 1) as f32 / 1_000.0
        );
    }

    /// The hot cell is the one with most players; a tie goes to the smallest
    /// index.
    ///
    /// Mutation this catches: `>=` in place of `>` (the last cell of a tie
    /// wins, so the choice depends on nothing a reader can say), or counting
    /// cells instead of players.
    #[test]
    fn the_hot_cell_holds_most_players() {
        assert_eq!(hot_cell(&[]), None);
        let (a, b, c) = ([5, 0, -1], [5, 0, 0], [4, 9, 9]);
        assert_eq!(hot_cell(&[a]), Some((a, 1)));
        assert_eq!(hot_cell(&[b, a, b, c, b, a]), Some((b, 3)));
        assert_eq!(
            hot_cell(&[b, a, c, a, b]),
            Some((a, 2)),
            "a tie: the smaller"
        );
        assert_eq!(hot_cell(&[b, c]), Some((c, 1)), "a tie of ones");
    }

    /// A report carries what it was given, and the metric of the plan.
    ///
    /// Mutation this catches: the hot cell's count reported as the player
    /// count (the orchestrator could not tell a shard whose players are all in
    /// one cell from one whose players are spread).
    #[test]
    fn a_report_says_what_the_shard_holds() {
        let times = TickTimes {
            mean_ms: 4.0,
            p95_ms: 20.0,
            max_ms: 31.0,
        };
        let cells = [[1, 2, 3], [1, 2, 3], [0, 0, 0]];
        let r = report(ShardId::new(7), 4, &cells, 39, 12, times);
        assert_eq!(r.shard, ShardId::new(7));
        assert_eq!((r.players, r.entities, r.skipped), (4, 39, 12));
        assert_eq!((r.hot_cell, r.hot_cell_players), ([1, 2, 3], 2));
        assert_eq!(
            (r.tick_mean_ms, r.tick_p95_ms, r.tick_max_ms),
            (4.0, 20.0, 31.0)
        );
        assert!((r.load - 0.6).abs() < 1e-6, "20 ms of 33.3: {}", r.load);
        let idle = report(ShardId::new(7), 0, &[], 0, 0, LoadWindow::default().take());
        assert_eq!((idle.hot_cell_players, idle.load), (0, 0.0));
    }
}
