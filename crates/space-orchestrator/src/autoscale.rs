//! The load metric and the scale-out / scale-in decision (H14,
//! `docs/plan-sharding.md` §3).
//!
//! A pure function of `(loads, tick, prior state)` — no sockets, no clock, no
//! RNG — so the whole autoscaler is testable with `step(n)` and a table of
//! loads. Every timer is in **ticks** (plan §3.6): a wall-clock timer would
//! make the decision depend on when the test ran.
//!
//! # The metric
//!
//! The *maximum* of two normalised signals, so either one alone can trigger a
//! scale event:
//!
//! ```text
//! load(shard) = max(players / capacity, tick_p95_ms / TICK_BUDGET_MS)
//! ```
//!
//! Players alone would miss a shard that is slow for a reason the player count
//! does not explain (a dense region); tick time alone would miss a shard at
//! the player cap that is still keeping up.
//!
//! # The rules ([`ScaleConfig`]: every number is a setting)
//!
//! * **Out.** A shard whose load has stayed at or above
//!   [`ScaleConfig::out_load`] for [`ScaleConfig::out_dwell_ticks`] is
//!   relieved: [`ScaleAction::Split`]. What the orchestrator then carves for
//!   a new shard is the shard's **hot cell** ([`scale_out_target`]) — with
//!   one exception, the scene's spawn cell: a shard of its own for a cell
//!   that holds every player of the loaded shard relieves nobody (a region
//!   cannot be smaller than a cell), so that is not done, and the log says
//!   "the spawn cell is at capacity".
//! * **In.** A shard that could be merged away, and whose load **together
//!   with that of the shard it would merge into** ([`ScaleConfig::merged_load`]) has
//!   stayed at or below [`ScaleConfig::in_load`] for
//!   [`ScaleConfig::in_dwell_ticks`], is merged: [`ScaleAction::Merge`]. The
//!   pair is what counts — the shard that takes the region back must have
//!   room for what comes with it.
//! * **No flapping.** The two thresholds leave a band in which nothing
//!   happens, and a load that crosses back starts its dwell over; after any
//!   action the shard is immune for [`ScaleConfig::cooldown_ticks`] and needs
//!   a fresh, full dwell; and the whole cluster takes one action per
//!   out-dwell at most.
//! * **Order.** Hot before cold (a hot shard is the emergency). Of several
//!   hot shards the lowest id; of several cold ones the **highest** id — the
//!   newest shard is the first to go, so a cluster that grew and shrinks
//!   again ends with the shards it began with. That holds for a shard that
//!   was given the spawn cell too: it is merged back like any other carved
//!   cell once the pair is cold (it used to be exempt for owning the spawn
//!   point, and so stayed for ever).
//!
//! Who may be merged at all is the caller's to say ([`ShardLoad::merged`] is
//! `None` for a shard that may not): the orchestrator never offers a shard
//! it did not launch itself, one of the layout cells `--min-shards` stands
//! for, anybody while the cluster is at `--min-shards`, nor a shard whose
//! merge target has no fresh report or no room for its players
//! (`Orchestrator::shard_loads`).

use std::collections::BTreeMap;

use space_protocol::control::LoadReport;
use space_protocol::{ShardId, ShardMap};
use space_world::NodeRef;

/// Player count at which a shard is fully loaded (load = 1.0): the default of
/// [`ScaleConfig::capacity`].
pub const MAX_PLAYERS_PER_SHARD: f64 = 100.0;

/// The per-tick budget in ms (30 Hz → 33.3 ms). A shard at this tick time is at
/// load 1.0 even with few players: it cannot tick faster.
pub const TICK_BUDGET_MS: f64 = 1_000.0 / crate::config::TICK_HZ as f64;

/// Default of [`ScaleConfig::out_load`].
pub const SPLIT_THRESHOLD: f64 = 0.8;

/// Default of [`ScaleConfig::in_load`].
pub const MERGE_THRESHOLD: f64 = 0.35;

/// Default of [`ScaleConfig::out_dwell_ticks`] (10 s).
pub const SPLIT_DWELL_TICKS: u64 = 300;

/// Default of [`ScaleConfig::in_dwell_ticks`] (60 s).
pub const MERGE_DWELL_TICKS: u64 = 1_800;

/// Default of [`ScaleConfig::cooldown_ticks`] (30 s).
pub const SCALE_COOLDOWN_TICKS: u64 = 900;

/// Default of [`ScaleConfig::drain_timeout_ticks`] (30 s).
pub const DRAIN_TIMEOUT_TICKS: u64 = 900;

/// The longest drain timeout that may be configured (45 s): the shards of the
/// map go on hearing a shard that left it for
/// [`space_server::shard::FORMER_SHARD_GRACE_TICKS`] (60 s), and a drain must
/// be over well before that.
pub const MAX_DRAIN_TIMEOUT_TICKS: u64 = 45 * crate::config::TICK_HZ;

/// The autoscaler's settings (`space-orchestrator --shard-capacity`,
/// `--scale-out-load`, `--scale-out-after`, `--scale-in-load`,
/// `--scale-in-after`, `--scale-cooldown`, `--drain-timeout`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleConfig {
    /// Players at which a shard counts as fully loaded.
    pub capacity: u32,
    /// Load at or above which a shard is relieved, after the dwell.
    pub out_load: f64,
    /// Ticks the load must stay there before a scale-out.
    pub out_dwell_ticks: u64,
    /// Load of a shard **and** its merge target together at or below which
    /// the shard is merged away, after the dwell.
    pub in_load: f64,
    /// Ticks the pair's load must stay there before a scale-in.
    pub in_dwell_ticks: u64,
    /// Ticks a shard is left alone after any action that involved it.
    pub cooldown_ticks: u64,
    /// Ticks a merged shard is given to hand everything over before it is
    /// stopped as it is.
    pub drain_timeout_ticks: u64,
}

impl Default for ScaleConfig {
    fn default() -> Self {
        Self {
            capacity: MAX_PLAYERS_PER_SHARD as u32,
            out_load: SPLIT_THRESHOLD,
            out_dwell_ticks: SPLIT_DWELL_TICKS,
            in_load: MERGE_THRESHOLD,
            in_dwell_ticks: MERGE_DWELL_TICKS,
            cooldown_ticks: SCALE_COOLDOWN_TICKS,
            drain_timeout_ticks: DRAIN_TIMEOUT_TICKS,
        }
    }
}

impl ScaleConfig {
    /// Why these settings cannot be run with, if they cannot.
    ///
    /// # Errors
    ///
    /// The reason, for the operator: no capacity, thresholds that leave no
    /// band between them (the cluster would flap), a dwell of nothing, or a
    /// drain longer than the shards go on hearing a shard that left the map.
    pub fn check(&self) -> Result<(), String> {
        if self.capacity == 0 {
            return Err("--shard-capacity must be at least 1 player".into());
        }
        let limit = space_server::shard::MAX_PLAYERS_PER_SHARD;
        if usize::try_from(self.capacity).unwrap_or(usize::MAX) > limit {
            return Err(format!(
                "--shard-capacity ({}) must not be above {limit}: a shard refuses a player \
                 beyond that many, so a higher capacity would call a shard \"not loaded\" that \
                 is already turning players away",
                self.capacity
            ));
        }
        let band = self.in_load > 0.0 && self.in_load < self.out_load && self.out_load.is_finite();
        if !band {
            return Err(format!(
                "--scale-in-load ({}) must be above 0 and below --scale-out-load ({}): without a \
                 band between the two the cluster would grow and shrink in turns",
                self.in_load, self.out_load
            ));
        }
        if self.out_dwell_ticks == 0 || self.in_dwell_ticks == 0 {
            return Err("--scale-out-after and --scale-in-after must be at least 1 second".into());
        }
        if self.drain_timeout_ticks == 0 || self.drain_timeout_ticks > MAX_DRAIN_TIMEOUT_TICKS {
            return Err(format!(
                "--drain-timeout must be between 1 and {} seconds (the other shards hear a shard \
                 that left the map for a minute, and a drain must be over before that)",
                MAX_DRAIN_TIMEOUT_TICKS / crate::config::TICK_HZ
            ));
        }
        Ok(())
    }

    /// The load metric of `report` under these settings.
    #[must_use]
    pub fn load_of(&self, report: &LoadReport) -> f64 {
        (f64::from(report.players) / f64::from(self.capacity.max(1)))
            .max(f64::from(report.tick_p95_ms) / TICK_BUDGET_MS)
    }

    /// The load the shard that takes a region back would have afterwards:
    /// both shards' players against one capacity, both tick times against
    /// one budget. `into` is the merge target's report, and there is no
    /// form of this without one: a target that has not reported is **not**
    /// an idle shard, and the orchestrator offers no merge into it
    /// (`Orchestrator::shard_loads`).
    #[must_use]
    pub fn merged_load(&self, shard: &LoadReport, into: &LoadReport) -> f64 {
        (f64::from(shard.players.saturating_add(into.players)) / f64::from(self.capacity.max(1)))
            .max(f64::from(shard.tick_p95_ms + into.tick_p95_ms) / TICK_BUDGET_MS)
    }
}

/// The action the autoscaler decided on, if any.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScaleAction {
    /// Scale **out**: relieve `shard` by giving part of its region to a new
    /// shard.
    Split {
        /// The hot shard.
        shard: ShardId,
        /// A placeholder (always 0): the autoscaler sees load, not positions.
        /// The orchestrator chooses what to carve ([`scale_out_target`]).
        octant: usize,
    },
    /// Scale **in**: merge `shard`'s region back into the owner of the region
    /// around it ([`crate::topology::merge_target`]); the shard then drains
    /// and is stopped.
    Merge {
        /// The cold shard.
        shard: ShardId,
    },
}

impl core::fmt::Display for ScaleAction {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ScaleAction::Split { shard, .. } => write!(f, "scale out {shard}"),
            ScaleAction::Merge { shard } => write!(f, "scale in {shard}"),
        }
    }
}

/// The load metric at the default capacity: `max(players / 100, tick_p95_ms /
/// 33.3)`. ([`ScaleConfig::load_of`] is the one the autoscaler uses.)
#[must_use]
pub fn load_metric(report: &LoadReport) -> f64 {
    ScaleConfig::default().load_of(report)
}

/// What the autoscaler is told about one shard on one tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShardLoad {
    /// The shard.
    pub shard: ShardId,
    /// Its own load ([`ScaleConfig::load_of`]).
    pub load: f64,
    /// The load it and its merge target would have together
    /// ([`ScaleConfig::merged_load`]) — or `None` when this shard may not be
    /// merged away now (it owns the root, the orchestrator did not launch it,
    /// its merge target has not reported or has no room, or the cluster has
    /// no shard to spare).
    pub merged: Option<f64>,
}

/// What to give a new shard when `report`'s shard is to be relieved
/// ([`scale_out_target`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScaleOut {
    /// Carve this cell out of the shard's region: its hot cell.
    Cell(NodeRef),
    /// The report names no hot cell (the shard is slow without players):
    /// halve its region instead — the octant split, which only regions of
    /// 1024 km or more allow.
    Octant,
    /// Nothing can be done, and why.
    Nothing(String),
}

/// Where a new shard would relieve the shard of `report`, by the published
/// `map`: its **hot cell** — the cell holding most of its players — carved
/// out of its region. A pure function.
///
/// The one thing it cannot do is make a region smaller than a cell: a shard
/// whose hot cell **is** its region stays as loaded as it is, and the answer
/// says so.
///
/// # The spawn point's cell
///
/// `spawn_cell` is the cell the scene's spawn point lies in — where every
/// new player starts — if the map has one. It is the one hot cell a new shard
/// does not help with by itself: carved with **all** of the loaded shard's
/// players in it, the new shard is exactly as loaded as the old one was, can
/// never be relieved (a cell is the smallest region), and the old one sits
/// idle beside it. So:
///
/// * the spawn cell is its own region already, or every player of the shard
///   is in it: **nothing**, and the answer says "the spawn cell is at
///   capacity";
/// * the shard has players outside it too: the spawn cell is carved, once —
///   that takes the crowd off the tick of everybody else in the region. The
///   new shard owns the spawn point from then on, and is merged back like any
///   carved cell when it and the shard around it are cold together.
#[must_use]
pub fn scale_out_target(
    map: &ShardMap,
    report: &LoadReport,
    spawn_cell: Option<NodeRef>,
) -> ScaleOut {
    if report.hot_cell_players == 0 {
        return ScaleOut::Octant;
    }
    let cell = NodeRef::new(crate::topology::MAX_REGION_DEPTH, report.hot_cell);
    let shard = report.shard;
    let is_spawn = spawn_cell == Some(cell);
    if let Some(region) = map.regions.iter().find(|region| region.node == cell) {
        return ScaleOut::Nothing(if region.shard == shard && is_spawn {
            format!(
                "the spawn cell is at capacity: a region cannot be smaller than a cell (1024 m). \
                 {} of {shard}'s {} player(s) are in it, it is {shard}'s whole region, and every \
                 new player starts there — no new shard can take part of this load",
                report.hot_cell_players, report.players
            )
        } else if region.shard == shard {
            format!(
                "{} of {shard}'s {} player(s) are in one cell, and that cell is its whole region: \
                 a region cannot be smaller than a cell (1024 m), so no new shard can take part of \
                 this load",
                report.hot_cell_players, report.players
            )
        } else {
            format!(
                "{shard}'s players are on {}'s ground (they are being handed over)",
                region.shard
            )
        });
    }
    match crate::topology::enclosing_region(map, cell) {
        Some(region)
            if region.shard == shard && is_spawn && report.hot_cell_players >= report.players =>
        {
            ScaleOut::Nothing(format!(
                "the spawn cell is at capacity: a region cannot be smaller than a cell (1024 m). \
                 All {} of {shard}'s player(s) are in it and every new player starts there — a \
                 shard of its own would be exactly as loaded and could never be relieved",
                report.players
            ))
        }
        Some(region) if region.shard == shard => ScaleOut::Cell(cell),
        Some(region) => ScaleOut::Nothing(format!(
            "{shard}'s players are on {}'s ground (they are being handed over)",
            region.shard
        )),
        None => ScaleOut::Nothing(format!("{shard}'s hot cell lies outside the map")),
    }
}

/// Per-shard dwell/cooldown state.
#[derive(Debug, Clone, Copy)]
struct ShardState {
    /// Tick the load last crossed *up* through the out threshold (`None` = below).
    hot_since: Option<u64>,
    /// Tick the merged load last crossed *down* through the in threshold.
    cold_since: Option<u64>,
    /// Tick until which the shard is immune to any action (cooldown).
    immune_until: u64,
}

/// The autoscaler: loads in, at most one [`ScaleAction`] out per tick.
#[derive(Debug, Default)]
pub struct Autoscaler {
    config: ScaleConfig,
    states: BTreeMap<ShardId, ShardState>,
    /// Tick until which *no* action may fire (one action in flight / global damper).
    busy_until: u64,
}

impl Autoscaler {
    /// A fresh autoscaler with the default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A fresh autoscaler with `config`.
    #[must_use]
    pub fn with_config(config: ScaleConfig) -> Self {
        Self {
            config,
            ..Self::default()
        }
    }

    /// The settings it decides by.
    #[must_use]
    pub const fn config(&self) -> &ScaleConfig {
        &self.config
    }

    /// [`Self::on_loads`] for reports taken on their own: every shard's own
    /// load, and each counted as mergeable into an idle shard. (The
    /// orchestrator computes the pairs and who may be merged, and calls
    /// [`Self::on_loads`].)
    #[must_use]
    pub fn on_reports(&mut self, reports: &[LoadReport], tick: u64) -> Option<ScaleAction> {
        let loads: Vec<ShardLoad> = reports
            .iter()
            .map(|report| ShardLoad {
                shard: report.shard,
                load: self.config.load_of(report),
                merged: Some(self.config.merged_load(report, &LoadReport::default())),
            })
            .collect();
        self.on_loads(&loads, tick)
    }

    /// Feed one tick of loads and get at most one action.
    ///
    /// The dwell timers measure *how long the load has stayed on the hot/cold
    /// side of its threshold*; a load that crosses back resets the timer, so a
    /// shard that oscillates around a threshold never accumulates dwell (the
    /// band between the two thresholds is what makes oscillation the common
    /// case rather than the pathological one).
    #[must_use]
    pub fn on_loads(&mut self, loads: &[ShardLoad], tick: u64) -> Option<ScaleAction> {
        let config = self.config;
        for load in loads {
            let state = self.states.entry(load.shard).or_insert(ShardState {
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
            if load.load >= config.out_load {
                if state.hot_since.is_none() {
                    state.hot_since = Some(tick);
                }
                state.cold_since = None;
            } else if load.merged.is_some_and(|merged| merged <= config.in_load) {
                if state.cold_since.is_none() {
                    state.cold_since = Some(tick);
                }
                state.hot_since = None;
            } else {
                // Inside the band — or cold, but not to be merged: neither
                // hot nor cold, dwell resets.
                state.hot_since = None;
                state.cold_since = None;
            }
        }

        if tick < self.busy_until {
            return None;
        }

        // Deterministic choice: hot shards first (a hot shard is the
        // emergency), lowest id first; then cold ones, highest id first (the
        // newest shard is the first to go).
        let hot = self
            .states
            .iter()
            .find(|(_, st)| {
                st.hot_since
                    .is_some_and(|since| tick - since >= config.out_dwell_ticks)
                    && tick >= st.immune_until
            })
            .map(|(shard, _)| *shard);
        if let Some(shard) = hot {
            self.commit(tick);
            return Some(ScaleAction::Split { shard, octant: 0 });
        }
        let cold = self
            .states
            .iter()
            .rev()
            .find(|(_, st)| {
                st.cold_since
                    .is_some_and(|since| tick - since >= config.in_dwell_ticks)
                    && tick >= st.immune_until
            })
            .map(|(shard, _)| *shard);
        if let Some(shard) = cold {
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
        self.busy_until = tick + self.config.out_dwell_ticks; // one action in flight
        if let Some(state) = self.states.get_mut(&shard) {
            state.immune_until = tick + self.config.cooldown_ticks;
            state.hot_since = None;
            state.cold_since = None;
        }
    }

    fn commit(&mut self, tick: u64) {
        self.busy_until = tick + self.config.out_dwell_ticks;
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

    /// Tick until which no action fires at all (the one-action-in-flight
    /// damper). Read-only: an operator view shows it, nothing decides on it.
    #[must_use]
    pub fn busy_until(&self) -> u64 {
        self.busy_until
    }

    /// What is remembered about `shard`: its dwell timers and its cooldown.
    /// Read-only, for an operator view.
    #[must_use]
    pub fn state_of(&self, shard: ShardId) -> Option<ShardScaleState> {
        self.states.get(&shard).map(|state| ShardScaleState {
            hot_since: state.hot_since,
            cold_since: state.cold_since,
            immune_until: state.immune_until,
        })
    }

    /// Clear every dwell timer; cooldowns are kept.
    ///
    /// Called when an operator pauses (or resumes) the autoscaler. While it is
    /// paused no report is fed, so a timer left standing would go on "counting"
    /// the pause: a shard that was hot for one tick before a ten-minute pause
    /// would be split the instant the autoscaler resumed. After this every
    /// shard needs a fresh, full dwell of reports before anything fires.
    pub fn reset_dwell(&mut self) {
        for state in self.states.values_mut() {
            state.hot_since = None;
            state.cold_since = None;
        }
    }
}

/// What the autoscaler remembers about one shard, as plain data
/// ([`Autoscaler::state_of`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardScaleState {
    /// Tick since which the load has stayed at/above the out threshold.
    pub hot_since: Option<u64>,
    /// Tick since which the merged load has stayed at/below the in threshold.
    pub cold_since: Option<u64>,
    /// Tick until which the shard is immune to any action (cooldown).
    pub immune_until: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use space_protocol::{ShardId, ShardRegion};

    fn report(shard: u32, players: u32, tick_p95_ms: f32) -> LoadReport {
        LoadReport {
            shard: ShardId::new(shard),
            players,
            tick_p95_ms,
            ..Default::default()
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

    /// The capacity is a setting, and a pair's load is both shards' players
    /// against one capacity and both tick times against one budget.
    ///
    /// Mutation this catches: `merged_load` taking the larger of the two
    /// loads (two shards at 0.3 each would be merged into one at 0.6, above
    /// the low-water mark it was merged under — and split again).
    #[test]
    fn the_metric_follows_the_capacity_and_a_pair_adds_up() {
        let config = ScaleConfig {
            capacity: 4,
            ..ScaleConfig::default()
        };
        let table = [
            // (players, p95 ms, load)
            (0, 0.0, 0.0),
            (1, 0.0, 0.25),
            (4, 0.0, 1.0),
            (6, 0.0, 1.5),
            (1, 20.0, 0.6),
            (3, 10.0, 0.75),
        ];
        for (players, tick, load) in table {
            let got = config.load_of(&report(1, players, tick));
            assert!(
                (got - load).abs() < 1e-6,
                "{players} players, {tick} ms: {got}"
            );
        }
        let pairs = [
            // (a players, a ms, b players, b ms, merged load)
            (1, 0.0, 1, 0.0, 0.5),
            (0, 10.0, 0, 10.0, 0.6),
            (1, 5.0, 0, 5.0, 0.3),
            (3, 1.0, 2, 1.0, 1.25),
        ];
        for (ap, at, bp, bt, load) in pairs {
            let got = config.merged_load(&report(2, ap, at), &report(1, bp, bt));
            assert!((got - load).abs() < 1e-6, "{ap}+{bp} players: {got}");
        }
    }

    /// The decision, as a table: loads held for a number of ticks, and what
    /// comes out at the end.
    ///
    /// Mutations this catches: hot at `>` instead of `>=` the threshold; a
    /// cold shard that may not be merged (`merged: None`) chosen all the
    /// same; cold shards taken lowest id first (the oldest shard would go
    /// before the newest); a hot shard not preferred over a cold one.
    #[test]
    fn the_decision_table() {
        let config = ScaleConfig {
            capacity: 2,
            out_load: 0.8,
            out_dwell_ticks: 10,
            in_load: 0.35,
            in_dwell_ticks: 30,
            cooldown_ticks: 50,
            drain_timeout_ticks: 100,
        };
        let load = |shard: u32, load: f64, merged: Option<f64>| ShardLoad {
            shard: ShardId::new(shard),
            load,
            merged,
        };
        let out = |shard: u32| {
            Some(ScaleAction::Split {
                shard: ShardId::new(shard),
                octant: 0,
            })
        };
        let inn = |shard: u32| {
            Some(ScaleAction::Merge {
                shard: ShardId::new(shard),
            })
        };
        // (name, phases of (loads, ticks they are fed for), what the tick
        // after the last phase's last tick decides). Nothing is decided on
        // any tick before that one.
        type Row = (
            &'static str,
            Vec<(Vec<ShardLoad>, u64)>,
            Option<ScaleAction>,
        );
        let table: Vec<Row> = vec![
            (
                "exactly at the out threshold is hot",
                vec![(vec![load(1, 0.8, None)], 10)],
                out(1),
            ),
            (
                "one tick short of the dwell",
                vec![(vec![load(1, 1.0, None)], 9)],
                None,
            ),
            (
                "just under the threshold",
                vec![(vec![load(1, 0.79, None)], 40)],
                None,
            ),
            (
                "in the band",
                vec![(vec![load(2, 0.5, Some(0.5))], 100)],
                None,
            ),
            (
                "a cold pair after the long dwell",
                vec![(vec![load(2, 0.0, Some(0.35))], 30)],
                inn(2),
            ),
            (
                "cold, one tick short",
                vec![(vec![load(2, 0.0, Some(0.0))], 29)],
                None,
            ),
            (
                "cold itself, the pair is not",
                vec![(vec![load(2, 0.0, Some(0.5))], 100)],
                None,
            ),
            (
                "cold but not to be merged",
                vec![(vec![load(1, 0.0, None)], 100)],
                None,
            ),
            (
                "of two cold shards the newest goes",
                vec![(
                    vec![
                        load(2, 0.0, Some(0.0)),
                        load(5, 0.0, Some(0.0)),
                        load(1, 0.0, None),
                    ],
                    30,
                )],
                inn(5),
            ),
            (
                "of two hot shards the lowest id is relieved first",
                vec![(vec![load(7, 1.0, None), load(3, 1.0, None)], 10)],
                out(3),
            ),
            (
                "hot before cold, when both dwells end on one tick",
                vec![
                    (vec![load(2, 0.0, Some(0.0)), load(4, 0.5, None)], 20),
                    (vec![load(2, 0.0, Some(0.0)), load(4, 1.0, None)], 10),
                ],
                out(4),
            ),
            (
                "a load that crosses back starts its dwell over",
                vec![
                    (vec![load(1, 1.0, None)], 9),
                    (vec![load(1, 0.5, None)], 1),
                    (vec![load(1, 1.0, None)], 9),
                ],
                None,
            ),
        ];
        for (name, phases, want) in table {
            let mut scaler = Autoscaler::with_config(config);
            let mut tick = 0;
            let mut last = Vec::new();
            for (loads, ticks) in phases {
                for _ in 0..ticks {
                    assert_eq!(
                        scaler.on_loads(&loads, tick),
                        None,
                        "{name}: early at {tick}"
                    );
                    tick += 1;
                }
                last = loads;
            }
            assert_eq!(scaler.on_loads(&last, tick), want, "{name}: at {tick}");
        }
    }

    /// After a scale-out the cluster does not scale back in at once, and the
    /// shard needs a fresh dwell: no flapping.
    ///
    /// Mutation this catches: `note_action` not setting the cooldown (the
    /// shard that was just relieved — now cold — would be merged back after
    /// one in-dwell, and relieved again).
    #[test]
    fn an_action_is_followed_by_a_cooldown_whatever_the_load_does() {
        let config = ScaleConfig {
            capacity: 2,
            out_dwell_ticks: 10,
            in_dwell_ticks: 20,
            cooldown_ticks: 100,
            ..ScaleConfig::default()
        };
        let mut scaler = Autoscaler::with_config(config);
        let hot = [ShardLoad {
            shard: ShardId::new(1),
            load: 1.0,
            merged: None,
        }];
        let mut fired = None;
        for tick in 0..=10 {
            fired = scaler.on_loads(&hot, tick);
        }
        assert!(matches!(fired, Some(ScaleAction::Split { .. })));
        scaler.note_action(ShardId::new(1), 10);
        // Its load falls at once (the new shard took the players): cold.
        let cold = [ShardLoad {
            shard: ShardId::new(1),
            load: 0.0,
            merged: Some(0.0),
        }];
        for tick in 11..110 {
            assert_eq!(scaler.on_loads(&cold, tick), None, "cooling down at {tick}");
        }
        // The cooldown is over at 110; a full in-dwell later it may go.
        for tick in 110..130 {
            assert_eq!(scaler.on_loads(&cold, tick), None, "fresh dwell at {tick}");
        }
        assert_eq!(
            scaler.on_loads(&cold, 130),
            Some(ScaleAction::Merge {
                shard: ShardId::new(1)
            })
        );
    }

    /// Where a new shard relieves a loaded one: its hot cell, unless that is
    /// all it has.
    ///
    /// Mutations this catches: carving the hot cell of a one-cell shard (the
    /// carve can never validate: the autoscaler would ask for it every
    /// cooldown, for ever); carving a cell on another shard's ground.
    #[test]
    fn a_loaded_shard_is_relieved_at_its_hot_cell() {
        let addr: std::net::SocketAddr = "127.0.0.1:1".parse().unwrap();
        let root = NodeRef::new(40, [0, 0, 0]);
        let k = root.center().cell;
        let spawn_cell = NodeRef::new(64, k);
        let west = NodeRef::new(64, [k[0] - 1, k[1], k[2]]);
        let map = ShardMap::new(
            3,
            vec![
                ShardRegion::new(root, ShardId::new(1), addr),
                ShardRegion::new(west, ShardId::new(2), addr),
            ],
        );
        let hot = |shard: u32, cell: [i64; 3], players: u32| LoadReport {
            shard: ShardId::new(shard),
            players,
            hot_cell: cell,
            hot_cell_players: players,
            ..Default::default()
        };
        let spawn = Some(spawn_cell);
        // Any cell of its ground that is not the spawn point's: carved.
        let far = [k[0] + 5, k[1], k[2] - 2];
        assert_eq!(
            scale_out_target(&map, &hot(1, far, 9), spawn),
            ScaleOut::Cell(NodeRef::new(64, far))
        );
        // A one-cell shard cannot be relieved.
        let ScaleOut::Nothing(why) = scale_out_target(&map, &hot(2, west.index, 9), spawn) else {
            panic!("a cell cannot be cut");
        };
        assert!(why.contains("cannot be smaller than a cell"), "{why}");
        assert!(!why.contains("spawn cell"), "not the spawn cell: {why}");
        // Players it still holds on a neighbour's ground are that
        // neighbour's in a moment.
        let ScaleOut::Nothing(why) = scale_out_target(&map, &hot(1, west.index, 9), spawn) else {
            panic!("not its ground");
        };
        assert!(why.contains("being handed over"), "{why}");
        // Slow without players: the old halving.
        assert_eq!(
            scale_out_target(&map, &hot(1, k, 0), spawn),
            ScaleOut::Octant
        );
    }

    /// The spawn point's cell is the one hot cell a shard of its own does
    /// not help with: every new player starts there.
    ///
    /// Mutation this catches: carving the spawn cell like any other hot cell
    /// (the state before: with all of the root's players on the spawn point
    /// the cell went to a new shard that was exactly as loaded, could never
    /// be relieved — a region is never smaller than a cell — and was never
    /// merged back; the cluster had grown by a shard and nobody was relieved).
    #[test]
    fn the_spawn_cell_is_carved_only_while_that_relieves_somebody() {
        let addr: std::net::SocketAddr = "127.0.0.1:1".parse().unwrap();
        let root = NodeRef::new(40, [0, 0, 0]);
        let k = root.center().cell;
        let spawn_cell = NodeRef::new(64, k);
        let map = ShardMap::new(1, vec![ShardRegion::new(root, ShardId::new(1), addr)]);
        let crowd = |shard: u32, players: u32, in_spawn_cell: u32| LoadReport {
            shard: ShardId::new(shard),
            players,
            hot_cell: k,
            hot_cell_players: in_spawn_cell,
            ..Default::default()
        };
        // Everybody is in the spawn cell: a shard of its own changes nothing.
        let ScaleOut::Nothing(why) = scale_out_target(&map, &crowd(1, 9, 9), Some(spawn_cell))
        else {
            panic!("the spawn cell must not be carved with every player in it");
        };
        assert!(
            why.contains("the spawn cell is at capacity")
                && why.contains("cannot be smaller than a cell"),
            "{why}"
        );
        // Three more out in the region: carving the cell takes the crowd off
        // their tick.
        assert_eq!(
            scale_out_target(&map, &crowd(1, 12, 9), Some(spawn_cell)),
            ScaleOut::Cell(spawn_cell)
        );
        // Once it is a region of its own it cannot be cut again, and the
        // answer names it.
        let carved = ShardMap::new(
            2,
            vec![
                ShardRegion::new(root, ShardId::new(1), addr),
                ShardRegion::new(spawn_cell, ShardId::new(2), addr),
            ],
        );
        let ScaleOut::Nothing(why) = scale_out_target(&carved, &crowd(2, 9, 9), Some(spawn_cell))
        else {
            panic!("a cell cannot be cut");
        };
        assert!(why.contains("the spawn cell is at capacity"), "{why}");
        // A map without a spawn point (no scene anchor): a hot cell like any
        // other.
        assert_eq!(
            scale_out_target(&map, &crowd(1, 9, 9), None),
            ScaleOut::Cell(spawn_cell)
        );
    }

    /// Settings that would make the cluster flap, or never act, are refused.
    ///
    /// Mutation this catches: accepting `in_load >= out_load` (a shard is
    /// then hot and cold at once: relieved and merged in turns).
    #[test]
    fn settings_without_a_band_are_refused() {
        assert_eq!(ScaleConfig::default().check(), Ok(()));
        let with = |f: fn(&mut ScaleConfig)| {
            let mut config = ScaleConfig::default();
            f(&mut config);
            config.check()
        };
        assert!(with(|c| c.capacity = 0).is_err());
        // (Mutation: no upper bound — a shard refuses the 101st player while
        // a capacity of 200 calls it half empty.)
        let limit = u32::try_from(space_server::shard::MAX_PLAYERS_PER_SHARD).unwrap();
        assert_eq!(with(|c| c.capacity = 100), Ok(()));
        assert_eq!(limit, 100, "the default capacity is the shard's limit");
        let too_many = with(|c| c.capacity = 101).expect_err("above what a shard takes");
        assert!(too_many.contains("--shard-capacity"), "{too_many}");
        assert!(with(|c| c.in_load = 0.8).is_err());
        assert!(with(|c| c.in_load = 0.9).is_err());
        assert!(with(|c| c.in_load = 0.0).is_err());
        assert!(with(|c| c.out_dwell_ticks = 0).is_err());
        assert!(with(|c| c.drain_timeout_ticks = MAX_DRAIN_TIMEOUT_TICKS + 1).is_err());
        assert_eq!(
            with(|c| c.drain_timeout_ticks = MAX_DRAIN_TIMEOUT_TICKS),
            Ok(())
        );
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
