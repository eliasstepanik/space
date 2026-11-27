//! Adaptive quality: lower the ray budget and the render scale when the GPU
//! misses its share of the frame budget, and raise them again when there is
//! room (`--adaptive auto|off|0..6`).
//!
//! The levels are an ordered ladder, 0 = the preset as chosen, 6 = the
//! cheapest. [`Adaptive::observe`] is fed the measured GPU time of each frame;
//! it averages [`WINDOW`] frames and decides once per window. Going down is
//! quick (it jumps straight to the level its cost model predicts will fit).
//! Going up is careful: one level at a time, each step checked against a
//! measured window at the new level, and only when the level above is
//! predicted to still be well inside the budget. The first step up waits
//! for [`UP_WINDOWS`] calm windows in a row; while the windows stay calm
//! each further step needs only [`UP_AGAIN`], so a view that became cheap
//! is back at level 0 within about six seconds. A window that is not calm,
//! or a step down, ends the quick rise, so the level does not oscillate.

use std::str::FromStr;
use std::time::Duration;

use space_render::{QualityPreset, RayBudget, RenderOptions};

/// One rung of the ladder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Level {
    /// The ray budget of the level.
    pub budget: RayBudget,
    /// The render scale of the level.
    pub scale: f32,
    /// How much more a frame costs one level further up (cheaper) than this
    /// one, as the decisions use it: going down into level `k` divides the
    /// predicted time by `LEVELS[k].step_up_cost`, and going up out of level
    /// `k` multiplies it by it.
    pub step_up_cost: f32,
}

const fn level(budget: RayBudget, scale: f32, step_up_cost: f32) -> Level {
    Level {
        budget,
        scale,
        step_up_cost,
    }
}

/// The ladder, level 0 first.
pub const LEVELS: [Level; 7] = [
    level(RayBudget::Full, 1.00, 1.00),
    level(RayBudget::Half, 1.00, 1.35),
    level(RayBudget::Quarter, 1.00, 1.25),
    level(RayBudget::Quarter, 0.85, 1.40),
    level(RayBudget::Quarter, 0.70, 1.48),
    level(RayBudget::Quarter, 0.60, 1.37),
    level(RayBudget::Quarter, 0.50, 1.44),
];

/// The last level.
const LAST: usize = LEVELS.len() - 1;

/// Frames averaged per decision.
pub const WINDOW: u32 = 30;
/// Frames ignored after a change of level.
pub const SETTLE: u32 = 5;
/// Calm windows in a row before the first step up.
pub const UP_WINDOWS: u32 = 4;
/// Calm windows before each further step up while the rise goes on.
pub const UP_AGAIN: u32 = 1;
/// A mean above this share of the budget steps down.
pub const OVER: f32 = 0.90;
/// A step down aims for a predicted time at or below this share.
pub const FIT: f32 = 0.80;
/// A step up needs the level above predicted at or below this share.
pub const UNDER: f32 = 0.70;
/// The share of the frame time the GPU may use.
pub const GPU_SHARE: f32 = 0.75;

/// What `--adaptive` chose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptiveMode {
    /// Follow the measured GPU time.
    Auto,
    /// Stay on this level.
    Fixed(usize),
}

impl Default for AdaptiveMode {
    fn default() -> Self {
        Self::Fixed(0)
    }
}

impl FromStr for AdaptiveMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "off" => Ok(Self::Fixed(0)),
            _ => match s.parse::<usize>() {
                Ok(n) if n <= LAST && s.bytes().all(|b| b.is_ascii_digit()) => Ok(Self::Fixed(n)),
                _ => Err(format!(
                    "unknown adaptive mode \"{s}\" (expected auto, off or a level 0-6)"
                )),
            },
        }
    }
}

/// The adaptive-quality controller.
#[derive(Clone, Debug)]
pub struct Adaptive {
    mode: AdaptiveMode,
    /// The GPU time one frame may take, seconds.
    budget: f32,
    level: usize,
    last: Option<Duration>,
    sum: f32,
    count: u32,
    settle: u32,
    calm: u32,
    /// Whether the last change was a step up and every window since was calm.
    rising: bool,
}

impl Adaptive {
    /// A controller for `mode` that aims at a GPU share of `frame_target`.
    #[must_use]
    pub fn new(mode: AdaptiveMode, frame_target: Duration) -> Self {
        let level = match mode {
            AdaptiveMode::Fixed(n) => n.min(LAST),
            AdaptiveMode::Auto => 0,
        };
        Self {
            mode,
            budget: frame_target.as_secs_f32() * GPU_SHARE,
            level,
            last: None,
            sum: 0.0,
            count: 0,
            settle: 0,
            calm: 0,
            rising: false,
        }
    }

    /// The current level, 0 to 6.
    #[must_use]
    pub fn level(&self) -> usize {
        self.level
    }

    /// The last GPU time observed.
    #[must_use]
    pub fn gpu_time(&self) -> Option<Duration> {
        self.last
    }

    /// `options` with the ray budget and render scale of the current level;
    /// the `Ci` preset is never touched.
    #[must_use]
    pub fn apply(&self, mut options: RenderOptions) -> RenderOptions {
        if options.preset != QualityPreset::Ci {
            let l = LEVELS[self.level];
            options.ray_budget = l.budget;
            options.render_scale = l.scale;
        }
        options
    }

    /// Takes one frame's GPU time. It is always remembered; it steers the
    /// level only when `adapt` and the mode is [`AdaptiveMode::Auto`].
    pub fn observe(&mut self, gpu: Duration, adapt: bool) {
        self.last = Some(gpu);
        if !adapt || self.mode != AdaptiveMode::Auto {
            return;
        }
        if self.settle > 0 {
            self.settle -= 1;
            return;
        }
        self.sum += gpu.as_secs_f32();
        self.count += 1;
        if self.count < WINDOW {
            return;
        }
        let mean = self.sum / WINDOW as f32;
        self.sum = 0.0;
        self.count = 0;

        if mean > OVER * self.budget && self.level < LAST {
            let mut predicted = mean;
            let mut k = self.level;
            while k < LAST && predicted > FIT * self.budget {
                k += 1;
                predicted /= LEVELS[k].step_up_cost;
            }
            self.level = k;
            self.calm = 0;
            self.rising = false;
            self.settle = SETTLE;
        } else if self.level > 0 && mean * LEVELS[self.level].step_up_cost <= UNDER * self.budget {
            self.calm += 1;
            let needed = if self.rising { UP_AGAIN } else { UP_WINDOWS };
            if self.calm >= needed {
                self.level -= 1;
                self.calm = 0;
                self.rising = true;
                self.settle = SETTLE;
            }
        } else {
            self.calm = 0;
            self.rising = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TARGET: Duration = Duration::from_micros(16_667);

    fn budget() -> f32 {
        TARGET.as_secs_f32() * GPU_SHARE
    }

    /// The synthetic GPU time at `level` for a cost of `base` at level 0.
    fn cost(base: f32, level: usize) -> Duration {
        let divisor: f32 = LEVELS[1..=level].iter().map(|l| l.step_up_cost).product();
        Duration::from_secs_f32(base / divisor)
    }

    fn feed(a: &mut Adaptive, base: f32, frames: u32) -> Vec<(u32, usize)> {
        let mut changes = Vec::new();
        for i in 0..frames {
            let before = a.level();
            a.observe(cost(base, before), true);
            if a.level() != before {
                changes.push((i, a.level()));
            }
        }
        changes
    }

    #[test]
    fn a_cheap_scene_stays_at_level_zero() {
        let mut a = Adaptive::new(AdaptiveMode::Auto, TARGET);
        let changes = feed(&mut a, 0.5 * budget(), 50 * WINDOW);
        assert!(changes.is_empty());
        assert_eq!(a.level(), 0);
    }

    #[test]
    fn an_expensive_scene_settles_and_then_recovers_quickly_once_calm() {
        let mut a = Adaptive::new(AdaptiveMode::Auto, TARGET);
        let heavy = 3.0 * budget();
        feed(&mut a, heavy, 2 * (WINDOW + SETTLE));
        let level = a.level();
        assert!(
            level == 6 || cost(heavy, level).as_secs_f32() <= OVER * budget(),
            "level {level}"
        );
        assert!(level > 0);
        let changes = feed(&mut a, heavy, 50 * WINDOW);
        assert!(changes.is_empty(), "{changes:?}");

        let light = 0.1 * budget();
        let changes = feed(&mut a, light, 100 * WINDOW);
        assert_eq!(a.level(), 0);
        assert_eq!(changes.len(), level);
        let mut previous: Option<u32> = None;
        let mut expected = level;
        for (frame, new) in changes {
            expected -= 1;
            assert_eq!(new, expected, "one level per step");
            match previous {
                None => assert!(frame >= (UP_WINDOWS - 1) * WINDOW, "first step at {frame}"),
                Some(p) => assert_eq!(frame - p, UP_AGAIN * WINDOW + SETTLE, "{p} then {frame}"),
            }
            previous = Some(frame);
        }
    }

    #[test]
    fn a_calm_scene_is_back_at_level_zero_within_six_seconds() {
        let mut a = Adaptive::new(AdaptiveMode::Auto, TARGET);
        feed(&mut a, 10.0 * budget(), 4 * (WINDOW + SETTLE));
        assert_eq!(a.level(), LAST);
        let bound =
            (UP_WINDOWS + 1) * WINDOW + u32::try_from(LAST).unwrap() * (UP_AGAIN * WINDOW + SETTLE);
        assert!(bound <= 360, "{bound} frames is more than 6 s at 60 fps");
        feed(&mut a, 0.1 * budget(), bound);
        assert_eq!(a.level(), 0);
    }

    #[test]
    fn a_step_up_that_does_not_stay_calm_ends_the_quick_rise() {
        let mut a = Adaptive::new(AdaptiveMode::Auto, TARGET);
        feed(&mut a, 3.0 * budget(), 2 * (WINDOW + SETTLE));
        assert_eq!(a.level(), 5);
        // Calm at level 5; at level 4 neither calm nor over the budget.
        let changes = feed(&mut a, 2.0 * budget(), 50 * WINDOW);
        assert_eq!(changes.len(), 1, "{changes:?}");
        assert_eq!(a.level(), 4);
        // The quick rise is over: the next step up waits the full time again.
        let changes = feed(&mut a, 0.1 * budget(), 100 * WINDOW);
        assert_eq!(a.level(), 0);
        assert!(changes[0].0 >= (UP_WINDOWS - 1) * WINDOW, "{changes:?}");
    }

    #[test]
    fn a_fixed_level_stays() {
        let mut a = Adaptive::new(AdaptiveMode::Fixed(3), TARGET);
        assert_eq!(a.level(), 3);
        feed(&mut a, 5.0 * budget(), 10 * WINDOW);
        feed(&mut a, 0.01 * budget(), 10 * WINDOW);
        assert_eq!(a.level(), 3);
    }

    #[test]
    fn without_adapt_the_level_never_changes_but_the_time_is_kept() {
        let mut a = Adaptive::new(AdaptiveMode::Auto, TARGET);
        assert_eq!(a.gpu_time(), None);
        let t = Duration::from_secs_f32(5.0 * budget());
        for _ in 0..10 * WINDOW {
            a.observe(t, false);
        }
        assert_eq!(a.level(), 0);
        assert_eq!(a.gpu_time(), Some(t));
    }

    #[test]
    fn apply_sets_the_level_and_spares_ci() {
        let a = Adaptive::new(AdaptiveMode::Fixed(6), TARGET);
        let ci = RenderOptions::default();
        assert_eq!(ci.preset, QualityPreset::Ci);
        let out = a.apply(ci);
        assert_eq!(out.ray_budget, ci.ray_budget);
        assert_eq!(out.render_scale, ci.render_scale);
        let high = RenderOptions {
            preset: QualityPreset::High,
            ..RenderOptions::default()
        };
        let out = a.apply(high);
        assert_eq!(out.ray_budget, RayBudget::Quarter);
        assert_eq!(out.render_scale, 0.5);
    }

    #[test]
    fn the_modes_parse() {
        use AdaptiveMode::{Auto, Fixed};
        for (text, want) in [
            ("auto", Auto),
            ("off", Fixed(0)),
            ("0", Fixed(0)),
            ("3", Fixed(3)),
            ("6", Fixed(6)),
        ] {
            assert_eq!(text.parse(), Ok(want), "{text}");
        }
        for text in ["7", "9", "fast", "", "-1", "+2", "AUTO"] {
            assert_eq!(
                text.parse::<AdaptiveMode>(),
                Err(format!(
                    "unknown adaptive mode \"{text}\" (expected auto, off or a level 0-6)"
                )),
                "{text}"
            );
        }
        assert_eq!(AdaptiveMode::default(), Fixed(0));
    }
}
