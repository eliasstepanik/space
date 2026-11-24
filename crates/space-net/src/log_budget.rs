//! A budget for log lines that anybody who can reach a port can cause.
//!
//! A server says in its log when it turns a peer away. A peer that can be
//! turned away as often as it likes — a stranger sending the same bad hello a
//! thousand times a second — must not be able to turn that into a thousand
//! lines a second in a log nobody rotates. [`LogBudget`] is the rule every
//! such line goes through: a few lines per window, the rest counted, and the
//! count said with the next line that is written.

use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

/// At most a few lines per window; what is held back is counted.
///
/// Shared between tasks (`&self`); never held across an await.
#[derive(Debug)]
pub struct LogBudget {
    /// Lines that may be written per window.
    per_window: u32,
    /// The length of a window.
    window: Duration,
    state: Mutex<State>,
}

#[derive(Debug, Default)]
struct State {
    /// When the current window started (`None`: no line was ever asked for).
    window_start: Option<Instant>,
    /// Lines written in the current window.
    written: u32,
    /// Lines held back since the last one that was written.
    held_back: u64,
}

impl LogBudget {
    /// A budget of `per_window` lines every `window`.
    #[must_use]
    pub const fn new(per_window: u32, window: Duration) -> Self {
        Self {
            per_window,
            window,
            state: Mutex::new(State {
                window_start: None,
                written: 0,
                held_back: 0,
            }),
        }
    }

    /// Ask to write one line now. `Some(n)`: write it — `n` lines were held
    /// back since the last one written, which the caller says in this one.
    /// `None`: the budget of this window is used up; the line is counted and
    /// not written.
    pub fn admit(&self) -> Option<u64> {
        self.admit_at(Instant::now())
    }

    /// [`Self::admit`] at a given instant (a test that does not want to wait
    /// a window out).
    pub fn admit_at(&self, now: Instant) -> Option<u64> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let fresh = state
            .window_start
            .is_none_or(|start| now.saturating_duration_since(start) >= self.window);
        if fresh {
            state.window_start = Some(now);
            state.written = 0;
        }
        if state.written >= self.per_window {
            state.held_back = state.held_back.saturating_add(1);
            return None;
        }
        state.written += 1;
        Some(std::mem::take(&mut state.held_back))
    }
}

/// `" (N more like it were not written)"`, or nothing for `held_back == 0`:
/// what a line admitted by [`LogBudget::admit`] ends with.
#[must_use]
pub fn held_back_note(held_back: u64) -> String {
    if held_back == 0 {
        String::new()
    } else {
        format!(" ({held_back} more like it were not written here)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mutation this catches: a budget that never refuses (the flood it is
    /// there to stop), one that never opens again (a log that goes silent for
    /// good), or a count that is lost instead of said.
    #[test]
    fn a_few_lines_per_window_and_the_rest_are_counted() {
        let budget = LogBudget::new(3, Duration::from_secs(10));
        let start = Instant::now();
        let at = |ms: u64| start + Duration::from_millis(ms);
        assert_eq!(budget.admit_at(at(0)), Some(0));
        assert_eq!(budget.admit_at(at(1)), Some(0));
        assert_eq!(budget.admit_at(at(2)), Some(0));
        for n in 0..1_000 {
            assert_eq!(budget.admit_at(at(3 + n)), None, "line {n}");
        }
        // The window is not moved by what was refused in it.
        assert_eq!(budget.admit_at(at(9_999)), None);
        // The next window: a line again, and it says what was held back.
        assert_eq!(budget.admit_at(at(10_000)), Some(1_001));
        assert_eq!(budget.admit_at(at(10_001)), Some(0));
        assert_eq!(budget.admit_at(at(10_002)), Some(0));
        assert_eq!(budget.admit_at(at(10_003)), None);
        // A long silence, then one line: it still knows about that one.
        assert_eq!(budget.admit_at(at(600_000)), Some(1));
        assert_eq!(held_back_note(0), "");
        assert!(held_back_note(7).contains("7 more"));
    }
}
