//! The load metric and the split/merge decision (H14).
//!
//! A pure function of `(reports, tick, prior state)` — no sockets, no clock, no
//! RNG — so the whole autoscaler is testable with `step(n)` and a table of
//! reports. Every timer is in **ticks** (plan §3.6): a wall-clock timer would
//! make the decision depend on when the test ran.
//!
//! The metric is the *maximum* of two normalised signals, so either one alone
//! can trigger a scale event:
//!
//! ```text
//! load(shard) = max(players / MAX_PLAYERS_PER_SHARD, tick_p95_ms / TICK_BUDGET_MS)
//! ```
//!
//! Players alone would miss a shard that is slow for a reason the player count
//! does not explain (a dense region, a GC stall); tick time alone would miss a
//! shard at the player cap that is still keeping up.

use std::collections::BTreeMap;

use space_protocol::control::LoadReport;
use space_protocol::ShardId;

/// Player count at which a shard is fully loaded (load = 1.0).
pub const MAX_PLAYERS_PER_SHARD: f64 = 100.0;

/// The per-tick budget in ms (30 Hz → 33.3 ms). A shard at this tick time is at
/// load 1.0 even with few players: it cannot tick faster.
pub const TICK_BUDGET_MS: f64 = 1_000.0 / crate::config::TICK_HZ as f64;

/// Load at or above which a split is considered, after [`SPLIT_DWELL_TICKS`].
pub const SPLIT_THRESHOLD: f64 = 0.8;

/// Load at or below which a merge is considered, after [`MERGE_DWELL_TICKS`].
pub const MERGE_THRESHOLD: f64 = 0.35;

/// Ticks the load must stay at/above [`SPLIT_THRESHOLD`] before splitting (10 s).
pub const SPLIT_DWELL_TICKS: u64 = 300;

/// Ticks the load must stay at/below [`MERGE_THRESHOLD`] before merging (60 s).
pub const MERGE_DWELL_TICKS: u64 = 1_800;

/// Ticks a shard is immune after any scale action involving it (30 s).
pub const SCALE_COOLDOWN_TICKS: u64 = 900;

/// The action the autoscaler decided on, if any.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScaleAction {
    /// Split `shard`'s region, carving `octant` (bit 0 = +x, bit 1 = +y, bit 2 = +z).
    Split {
        /// The hot shard.
        shard: ShardId,
        /// A placeholder (always 0): the autoscaler sees load, not positions, so
        /// it cannot know which octant holds the players. The orchestrator picks
        /// the octant when it plans the split
        /// ([`crate::topology::pick_split_octant`]).
        octant: usize,
    },
    /// Merge `shard`'s region back into its parent's owner.
    Merge {
        /// The cold shard.
        shard: ShardId,
    },
}

impl core::fmt::Display for ScaleAction {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ScaleAction::Split { shard, octant } => write!(f, "split {shard} octant {octant}"),
            ScaleAction::Merge { shard } => write!(f, "merge {shard}"),
        }
    }
}

/// The load metric: `max(players / 100, tick_p95_ms / 33.3)`.
#[must_use]
pub fn load_metric(report: &LoadReport) -> f64 {
    (f64::from(report.players) / MAX_PLAYERS_PER_SHARD)
        .max(f64::from(report.tick_p95_ms) / TICK_BUDGET_MS)
}

/// Per-shard dwell/cooldown state.
#[derive(Debug, Clone, Copy)]
struct ShardState {
    /// Tick the load last crossed *up* through [`SPLIT_THRESHOLD`] (`None` = below).
    hot_since: Option<u64>,
    /// Tick the load last crossed *down* through [`MERGE_THRESHOLD`] (`None` = above).
    cold_since: Option<u64>,
    /// Tick until which the shard is immune to any action (cooldown).
    immune_until: u64,
}

/// The autoscaler: reports in, at most one [`ScaleAction`] out per tick.
#[derive(Debug, Default)]
pub struct Autoscaler {
    states: BTreeMap<ShardId, ShardState>,
    /// Tick until which *no* action may fire (one action in flight / global damper).
    busy_until: u64,
}

impl Autoscaler {
    /// A fresh autoscaler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one tick of reports and get at most one action.
    ///
    /// The dwell timers measure *how long the load has stayed on the hot/cold
    /// side of the threshold*; a report that crosses back resets the timer, so a
    /// shard that oscillates around the threshold never accumulates dwell (the
    /// hysteresis band between 0.35 and 0.8 is what makes oscillation the common
    /// case rather than the pathological one).
    #[must_use]
    pub fn on_reports(&mut self, reports: &[LoadReport], tick: u64) -> Option<ScaleAction> {
        for report in reports {
            let load = load_metric(report);
            let state = self.states.entry(report.shard).or_insert(ShardState {
                hot_since: None,
                cold_since: None,
                immune_until: 0,
            });
            if tick < state.immune_until {
                // Dwell does not accumulate during a cooldown: after an action
                // a shard needs a *fresh* full dwell, not one that accrued
                // while it was still settling.
                state.hot_since = None;
                state.cold_since = None;
                continue;
            }
            if load >= SPLIT_THRESHOLD {
                if state.hot_since.is_none() {
                    state.hot_since = Some(tick);
                }
                state.cold_since = None;
            } else if load <= MERGE_THRESHOLD {
                if state.cold_since.is_none() {
                    state.cold_since = Some(tick);
                }
                state.hot_since = None;
            } else {
                // Inside the hysteresis band: neither hot nor cold, dwell resets.
                state.hot_since = None;
                state.cold_since = None;
            }
        }

        if tick < self.busy_until {
            return None;
        }

        // Deterministic choice: hot shards first (a hot shard is the emergency),
        // lowest id first within a class.
        let mut hot: Vec<(ShardId, u64)> = self
            .states
            .iter()
            .filter(|(_, st)| {
                st.hot_since
                    .is_some_and(|since| tick - since >= SPLIT_DWELL_TICKS)
                    && tick >= st.immune_until
            })
            .map(|(s, st)| (*s, st.hot_since.unwrap()))
            .collect();
        hot.sort_by_key(|(s, _)| *s);
        if let Some((shard, _)) = hot.first() {
            let shard = *shard;
            self.commit(tick);
            return Some(ScaleAction::Split { shard, octant: 0 });
        }
        let mut cold: Vec<(ShardId, u64)> = self
            .states
            .iter()
            .filter(|(_, st)| {
                st.cold_since
                    .is_some_and(|since| tick - since >= MERGE_DWELL_TICKS)
                    && tick >= st.immune_until
            })
            .map(|(s, st)| (*s, st.cold_since.unwrap()))
            .collect();
        cold.sort_by_key(|(s, _)| *s);
        if let Some((shard, _)) = cold.first() {
            let shard = *shard;
            self.commit(tick);
            return Some(ScaleAction::Merge { shard });
        }
        None
    }

    /// Mark the cluster busy and cool down `shard` after an action was taken.
    ///
    /// The orchestrator calls this when it *acts* on an action (the epoch
    /// protocol may still abort it; a failed epoch is still an action for
    /// cooldown purposes — a shard that cannot complete an epoch is not helped
    /// by retrying it every tick).
    pub fn note_action(&mut self, shard: ShardId, tick: u64) {
        self.busy_until = tick + SPLIT_DWELL_TICKS; // one action in flight
        if let Some(state) = self.states.get_mut(&shard) {
            state.immune_until = tick + SCALE_COOLDOWN_TICKS;
            state.hot_since = None;
            state.cold_since = None;
        }
    }

    fn commit(&mut self, tick: u64) {
        self.busy_until = tick + SPLIT_DWELL_TICKS;
    }

    /// Drop everything remembered about `shard`: its dwell timers and its
    /// cooldown.
    ///
    /// Called when a shard leaves the cluster (it was merged away or killed) or
    /// goes silent (its reports stopped). Without this its last dwell timer kept
    /// counting on a report nobody was sending any more, and the autoscaler went
    /// on proposing splits and merges for a shard that was no longer there —
    /// each one a refused plan that made the whole cluster "busy" for a dwell.
    /// A shard that comes back starts from a clean state: a full, fresh dwell.
    pub fn forget(&mut self, shard: ShardId) {
        self.states.remove(&shard);
    }

    /// Whether the autoscaler holds any state for `shard`.
    #[must_use]
    pub fn tracks(&self, shard: ShardId) -> bool {
        self.states.contains_key(&shard)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use space_protocol::ShardId;

    fn report(shard: u32, players: u32, tick_p95_ms: f32) -> LoadReport {
        LoadReport {
            shard: ShardId::new(shard),
            players,
            tick_p95_ms,
            load: 0.0,
        }
    }

    #[test]
    fn the_metric_is_the_max_of_both_signals() {
        // Players dominate: 100 players, fast ticks.
        let r = report(1, 100, 1.0);
        assert!((load_metric(&r) - 1.0).abs() < 1e-6);
        // Tick time dominates: few players, slow ticks.
        let r = report(1, 1, TICK_BUDGET_MS as f32);
        assert!((load_metric(&r) - 1.0).abs() < 1e-3);
        // Both low -> low.
        let r = report(1, 10, 1.0);
        assert!(load_metric(&r) < 0.2);
        // The budget itself is 33.3 ms at 30 Hz.
        assert!((TICK_BUDGET_MS - 33.3333).abs() < 0.01);
    }

    #[test]
    fn a_sustained_hot_shard_splits_after_the_dwell() {
        let mut a = Autoscaler::new();
        let hot = report(1, 90, 1.0); // load 0.9
                                      // Feed the hot report from tick 0; just short of the dwell: no action.
        for t in 0..SPLIT_DWELL_TICKS {
            assert_eq!(a.on_reports(&[hot], t), None, "early at {t}");
        }
        // At the dwell: split.
        assert_eq!(
            a.on_reports(&[hot], SPLIT_DWELL_TICKS),
            Some(ScaleAction::Split {
                shard: ShardId::new(1),
                octant: 0
            })
        );
    }

    #[test]
    fn a_sustained_cold_shard_merges_after_the_longer_dwell() {
        let mut a = Autoscaler::new();
        let cold = report(1, 10, 1.0); // load 0.1
        for t in 0..MERGE_DWELL_TICKS {
            assert_eq!(a.on_reports(&[cold], t), None, "early at {t}");
        }
        assert_eq!(
            a.on_reports(&[cold], MERGE_DWELL_TICKS),
            Some(ScaleAction::Merge {
                shard: ShardId::new(1)
            })
        );
    }

    #[test]
    fn oscillation_around_the_threshold_never_accumulates_dwell() {
        // The flapping guard: a shard that crosses back resets its dwell timer,
        // so alternating hot/cold reports never trigger an action.
        let mut a = Autoscaler::new();
        let hot = report(1, 90, 1.0); // 0.9
        let cold = report(1, 10, 1.0); // 0.1
        for t in 0..(SPLIT_DWELL_TICKS * 4) {
            let r = if t % 2 == 0 { hot } else { cold };
            assert_eq!(
                a.on_reports(&[r], t),
                None,
                "flapping must not act at tick {t}"
            );
        }
    }

    #[test]
    fn the_hysteresis_band_resets_both_dwell_timers() {
        // A report inside (0.35, 0.8) is neither hot nor cold: it clears dwell.
        let mut a = Autoscaler::new();
        let hot = report(1, 90, 1.0);
        let mid = report(1, 50, 1.0); // load 0.5, inside the band
        for t in 0..(SPLIT_DWELL_TICKS - 1) {
            assert_eq!(a.on_reports(&[hot], t), None);
        }
        // One mid-band report resets hot_since, so the dwell restarts.
        assert_eq!(a.on_reports(&[mid], SPLIT_DWELL_TICKS - 1), None);
        assert_eq!(
            a.on_reports(&[hot], SPLIT_DWELL_TICKS),
            None,
            "dwell was reset"
        );
        // Two more full dwells of hot reports and it finally acts.
        for t in SPLIT_DWELL_TICKS..(2 * SPLIT_DWELL_TICKS) {
            assert_eq!(a.on_reports(&[hot], t), None);
        }
        assert_eq!(
            a.on_reports(&[hot], 2 * SPLIT_DWELL_TICKS),
            Some(ScaleAction::Split {
                shard: ShardId::new(1),
                octant: 0
            })
        );
    }

    #[test]
    fn after_an_action_the_shard_is_on_cooldown_and_the_cluster_is_busy() {
        let mut a = Autoscaler::new();
        let hot = report(1, 90, 1.0);
        for t in 0..SPLIT_DWELL_TICKS {
            assert_eq!(a.on_reports(&[hot], t), None, "early at {t}");
        }
        let action = a.on_reports(&[hot], SPLIT_DWELL_TICKS).expect("split");
        a.note_action(ShardId::new(1), SPLIT_DWELL_TICKS);
        // The cluster is busy for one dwell window: no second action.
        assert_eq!(a.on_reports(&[hot], SPLIT_DWELL_TICKS + 1), None);
        // Even past the busy window, the shard is immune until the cooldown ends.
        let t = SPLIT_DWELL_TICKS + SPLIT_DWELL_TICKS + 1;
        assert_eq!(a.on_reports(&[hot], t), None, "still cooling down");
        // After the cooldown, a fresh dwell is required before acting again.
        let after_cooldown = SPLIT_DWELL_TICKS + SCALE_COOLDOWN_TICKS;
        assert_eq!(
            a.on_reports(&[hot], after_cooldown),
            None,
            "dwell reset by note_action"
        );
        assert!(matches!(action, ScaleAction::Split { .. }));
    }

    /// Mutation this catches: `forget` not clearing the dwell timer. A shard
    /// that left with a completed dwell would be picked the moment any other
    /// report is fed, although nobody reports for it any more.
    #[test]
    fn a_forgotten_shard_is_never_chosen_on_its_old_dwell() {
        let mut a = Autoscaler::new();
        let (gone, alive) = (report(1, 90, 1.0), report(2, 50, 1.0));
        // Shard 1 is hot for all but the last tick of the dwell...
        for t in 0..SPLIT_DWELL_TICKS {
            assert_eq!(a.on_reports(&[gone, alive], t), None, "early at {t}");
        }
        assert!(a.tracks(ShardId::new(1)));
        // ...then it leaves the cluster. Only shard 2 (mid-band) still reports.
        a.forget(ShardId::new(1));
        assert!(!a.tracks(ShardId::new(1)));
        for t in SPLIT_DWELL_TICKS..(3 * SPLIT_DWELL_TICKS) {
            assert_eq!(
                a.on_reports(&[alive], t),
                None,
                "a departed shard must not be split at {t}"
            );
        }
        // If it comes back it needs a fresh, full dwell.
        let back = 3 * SPLIT_DWELL_TICKS;
        for t in back..(back + SPLIT_DWELL_TICKS) {
            assert_eq!(a.on_reports(&[gone, alive], t), None, "fresh dwell at {t}");
        }
        assert_eq!(
            a.on_reports(&[gone, alive], back + SPLIT_DWELL_TICKS),
            Some(ScaleAction::Split {
                shard: ShardId::new(1),
                octant: 0
            })
        );
    }

    #[test]
    fn the_choice_is_deterministic_lowest_id_first() {
        // Two equally hot shards: the lower id always wins, so a replay of the
        // same report sequence yields the same action (H14).
        let mut a = Autoscaler::new();
        let reports = [report(7, 90, 1.0), report(3, 90, 1.0), report(5, 90, 1.0)];
        for t in 0..SPLIT_DWELL_TICKS {
            assert_eq!(a.on_reports(&reports, t), None, "early at {t}");
        }
        let first = a.on_reports(&reports, SPLIT_DWELL_TICKS);
        assert_eq!(
            first,
            Some(ScaleAction::Split {
                shard: ShardId::new(3),
                octant: 0
            })
        );
    }
}
