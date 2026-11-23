//! The recent-events log: a bounded ring buffer an operator can read.
//!
//! Pure and synchronous, like the modules it sits next to: an entry is stamped
//! with the orchestrator's **tick**, never a wall clock, so the log is as
//! deterministic as the decisions it records (a front end turns ticks into
//! "12 s ago"). Nothing in the orchestrator ever *reads* the log to decide
//! anything — it exists for the operator view ([`crate::webui`]).
//!
//! The buffer keeps the newest [`EVENT_LOG_CAPACITY`] entries. A message that
//! repeats back to back (a misconfigured shard retrying its hello every second)
//! is counted on the entry it repeats instead of pushing the interesting
//! entries out, and every text is capped at [`MAX_EVENT_TEXT`] bytes, so the
//! log's memory is bounded no matter what arrives over the network.
//!
//! # Entries anybody can cause
//!
//! The control port checks a cluster name and nothing else, so whoever can
//! reach it can make the orchestrator refuse a hello — as often and with as
//! many different texts as it likes. Counting repeats does not help against
//! two texts sent in turn. Such entries ([`EventLog::push_peer_caused`]) are
//! therefore kept in a **ring of their own**, [`PEER_EVENT_CAPACITY`] long:
//! they can push each other out, and never an operator's request, an epoch or
//! a shard's registration. Both rings share one numbering, so the log still
//! reads in the order things happened.

use std::collections::VecDeque;

/// How many entries the log keeps.
pub const EVENT_LOG_CAPACITY: usize = 256;

/// How many entries caused by an unauthenticated peer the log keeps, on top
/// of [`EVENT_LOG_CAPACITY`] (see the module docs).
pub const PEER_EVENT_CAPACITY: usize = 16;

/// Longest text kept for one entry, in bytes (longer texts are cut).
pub const MAX_EVENT_TEXT: usize = 400;

/// What an entry is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    /// A shard registered, was refused, or its control link closed.
    Shard,
    /// An epoch was proposed, committed or aborted.
    Epoch,
    /// The autoscaler decided something.
    Autoscaler,
    /// An operator asked for something through a front end.
    Operator,
    /// The launcher could not start or stop a shard.
    Launcher,
    /// Something about the cluster's scene that no operator asked for (the
    /// scene file could not be used at startup).
    Scene,
}

impl EventKind {
    /// A short lower-case name (`"shard"`, `"epoch"`, …).
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            EventKind::Shard => "shard",
            EventKind::Epoch => "epoch",
            EventKind::Autoscaler => "autoscaler",
            EventKind::Operator => "operator",
            EventKind::Launcher => "launcher",
            EventKind::Scene => "scene",
        }
    }
}

/// How an entry should read at a glance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Something happened.
    Info,
    /// Something completed as intended (a registration, a commit).
    Good,
    /// Something was refused, lost or aborted.
    Warn,
}

impl Severity {
    /// A short lower-case name (`"info"`, `"ok"`, `"warn"`).
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Severity::Info => "info",
            Severity::Good => "ok",
            Severity::Warn => "warn",
        }
    }
}

/// One entry of the log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    /// The entry's number: 1 for the first entry ever logged, never reused.
    pub seq: u64,
    /// The orchestrator tick of the (most recent) occurrence.
    pub tick: u64,
    /// The tick of the first occurrence (equal to `tick` unless it repeated).
    pub first_tick: u64,
    /// How many times it happened back to back (1 = once).
    pub repeats: u32,
    /// What it is about.
    pub kind: EventKind,
    /// How it should read.
    pub severity: Severity,
    /// The message.
    pub text: String,
    /// Whether anybody who can reach the control port could have caused it (a
    /// refused hello). Such entries live in their own small ring.
    pub peer_caused: bool,
}

/// The bounded log. Newest entries win; see the module docs.
#[derive(Debug)]
pub struct EventLog {
    entries: VecDeque<Event>,
    capacity: usize,
    /// The ring of entries an unauthenticated peer can cause.
    peer_caused: VecDeque<Event>,
    /// How often [`EventLog::push_peer_caused`] was called, repeats and
    /// entries long pushed out included.
    peer_caused_total: u64,
    next_seq: u64,
    revision: u64,
}

impl Default for EventLog {
    fn default() -> Self {
        Self::new(EVENT_LOG_CAPACITY)
    }
}

impl EventLog {
    /// An empty log keeping at most `capacity` entries (at least one).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            entries: VecDeque::with_capacity(capacity.min(EVENT_LOG_CAPACITY)),
            capacity,
            peer_caused: VecDeque::with_capacity(PEER_EVENT_CAPACITY),
            peer_caused_total: 0,
            next_seq: 1,
            revision: 0,
        }
    }

    /// Record that `text` happened at `tick`; returns the entry's number.
    ///
    /// An entry identical to the newest one (same kind, severity and text) is
    /// counted on it instead of being added, and that entry's number is
    /// returned.
    pub fn push(
        &mut self,
        tick: u64,
        kind: EventKind,
        severity: Severity,
        text: impl Into<String>,
    ) -> u64 {
        self.push_into(false, tick, kind, severity, text.into())
    }

    /// [`Self::push`] for something anybody who can reach the control port can
    /// cause (a refused hello).
    ///
    /// The entry goes into the small ring kept for such entries
    /// ([`PEER_EVENT_CAPACITY`]) and can only ever push another one of its kind
    /// out — never an entry logged through [`Self::push`]. A repeat is counted
    /// on the newest entry of that ring.
    pub fn push_peer_caused(
        &mut self,
        tick: u64,
        kind: EventKind,
        severity: Severity,
        text: impl Into<String>,
    ) -> u64 {
        self.peer_caused_total += 1;
        self.push_into(true, tick, kind, severity, text.into())
    }

    /// How often something was logged through [`Self::push_peer_caused`] —
    /// every occurrence, also the ones the small ring no longer holds.
    #[must_use]
    pub fn peer_caused_total(&self) -> u64 {
        self.peer_caused_total
    }

    fn push_into(
        &mut self,
        peer_caused: bool,
        tick: u64,
        kind: EventKind,
        severity: Severity,
        text: String,
    ) -> u64 {
        let text = capped(text);
        self.revision += 1;
        let (ring, capacity) = if peer_caused {
            (&mut self.peer_caused, PEER_EVENT_CAPACITY)
        } else {
            (&mut self.entries, self.capacity)
        };
        if let Some(newest) = ring.back_mut() {
            if newest.kind == kind && newest.severity == severity && newest.text == text {
                newest.repeats = newest.repeats.saturating_add(1);
                newest.tick = tick;
                return newest.seq;
            }
        }
        if ring.len() >= capacity {
            ring.pop_front();
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        ring.push_back(Event {
            seq,
            tick,
            first_tick: tick,
            repeats: 1,
            kind,
            severity,
            text,
            peer_caused,
        });
        seq
    }

    /// The entries of both rings, newest first.
    pub fn newest_first(&self) -> impl Iterator<Item = &Event> {
        let mut main = self.entries.iter().rev().peekable();
        let mut peer = self.peer_caused.iter().rev().peekable();
        // Each ring is in number order; merge them, highest number first.
        std::iter::from_fn(move || match (main.peek(), peer.peek()) {
            (Some(a), Some(b)) if a.seq > b.seq => main.next(),
            (Some(_), None) => main.next(),
            _ => peer.next(),
        })
    }

    /// The entry numbered `seq`, if the log still holds it.
    #[must_use]
    pub fn get(&self, seq: u64) -> Option<&Event> {
        self.entries
            .iter()
            .chain(&self.peer_caused)
            .find(|event| event.seq == seq)
    }

    /// How many entries are held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len() + self.peer_caused.len()
    }

    /// Whether nothing has been logged (or everything was pushed out).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.peer_caused.is_empty()
    }

    /// A counter that moves every time anything is logged, repeats included.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.revision
    }
}

/// `text` cut to [`MAX_EVENT_TEXT`] bytes on a character boundary, with an
/// ellipsis when something was removed.
fn capped(mut text: String) -> String {
    if text.len() <= MAX_EVENT_TEXT {
        return text;
    }
    let mut end = MAX_EVENT_TEXT;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
    text.push('…');
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_are_numbered_from_one_and_listed_newest_first() {
        let mut log = EventLog::default();
        assert!(log.is_empty());
        assert_eq!(log.push(3, EventKind::Shard, Severity::Good, "a"), 1);
        assert_eq!(log.push(4, EventKind::Epoch, Severity::Info, "b"), 2);
        let seen: Vec<(u64, &str)> = log.newest_first().map(|e| (e.seq, &*e.text)).collect();
        assert_eq!(seen, vec![(2, "b"), (1, "a")]);
        assert_eq!(log.get(1).map(|e| e.tick), Some(3));
        assert!(log.get(3).is_none());
        assert_eq!(log.len(), 2);
    }

    /// Mutation this catches: an unbounded `Vec`. The log is fed from the
    /// network (a refused hello is an entry), so it must not grow for ever.
    #[test]
    fn the_log_is_a_ring_the_oldest_entry_leaves_first() {
        let mut log = EventLog::new(3);
        for n in 0..10u64 {
            log.push(n, EventKind::Shard, Severity::Info, format!("event {n}"));
        }
        assert_eq!(log.len(), 3);
        let seqs: Vec<u64> = log.newest_first().map(|e| e.seq).collect();
        assert_eq!(seqs, vec![10, 9, 8]);
        assert!(log.get(7).is_none(), "the oldest entries are gone");
        assert_eq!(log.revision(), 10);
    }

    /// Mutation this catches: pushing every repeat. One shard retrying a
    /// refused hello would flush every other entry out of the log.
    #[test]
    fn a_back_to_back_repeat_is_counted_not_added() {
        let mut log = EventLog::new(4);
        let first = log.push(1, EventKind::Shard, Severity::Warn, "hello refused");
        for tick in 2..=50 {
            let again = log.push(tick, EventKind::Shard, Severity::Warn, "hello refused");
            assert_eq!(again, first);
        }
        assert_eq!(log.len(), 1);
        let entry = log.get(first).expect("still there");
        assert_eq!((entry.repeats, entry.first_tick, entry.tick), (50, 1, 50));
        // A different entry in between ends the run.
        log.push(51, EventKind::Epoch, Severity::Info, "other");
        let next = log.push(52, EventKind::Shard, Severity::Warn, "hello refused");
        assert_ne!(next, first);
        assert_eq!(log.len(), 3);
        // The same words with another severity or kind are another entry too.
        let warn = log.push(53, EventKind::Shard, Severity::Info, "hello refused");
        assert_ne!(warn, next);
    }

    /// Mutation this catches: logging a refused hello into the main ring.
    /// Whoever can reach the control port could then erase the operator's and
    /// the epochs' history by sending 256 hellos that take turns between two
    /// cluster names (which also defeats the repeat count).
    #[test]
    fn entries_a_peer_can_cause_never_push_the_others_out() {
        let mut log = EventLog::default();
        let operator = log.push(1, EventKind::Operator, Severity::Info, "split shard 1");
        let commit = log.push(2, EventKind::Epoch, Severity::Good, "epoch 2 committed");
        for n in 0..10_000u64 {
            // Two texts in turn: no two back-to-back entries are equal.
            let text = if n.is_multiple_of(2) {
                "hello refused: a"
            } else {
                "hello refused: b"
            };
            log.push_peer_caused(3 + n, EventKind::Shard, Severity::Warn, text);
        }
        assert!(
            log.get(operator).is_some(),
            "the operator's entry is still there"
        );
        assert!(log.get(commit).is_some(), "and so is the commit");
        assert_eq!(log.len(), 2 + PEER_EVENT_CAPACITY, "the noise is bounded");
        assert_eq!(log.peer_caused_total(), 10_000, "but every one was counted");
        let seen: Vec<&Event> = log.newest_first().collect();
        assert!(
            seen.windows(2).all(|pair| pair[0].seq > pair[1].seq),
            "both rings read as one log, newest first"
        );
        assert!(seen[..PEER_EVENT_CAPACITY].iter().all(|e| e.peer_caused));
        assert_eq!(
            seen[PEER_EVENT_CAPACITY..]
                .iter()
                .map(|e| e.seq)
                .collect::<Vec<_>>(),
            vec![commit, operator]
        );
        assert!(!log.get(operator).expect("held").peer_caused);

        // The main ring keeps working around them: an entry logged later sorts
        // in front, and a repeat in either ring is still counted.
        let later = log.push(
            20_000,
            EventKind::Shard,
            Severity::Good,
            "shard 1 registered",
        );
        assert_eq!(log.newest_first().next().map(|e| e.seq), Some(later));
        let again = log.push(
            20_001,
            EventKind::Shard,
            Severity::Good,
            "shard 1 registered",
        );
        assert_eq!(
            again, later,
            "a peer-caused entry in between does not hide a repeat"
        );
        let noise = log.push_peer_caused(20_002, EventKind::Shard, Severity::Warn, "x");
        assert_eq!(
            log.push_peer_caused(20_003, EventKind::Shard, Severity::Warn, "x"),
            noise
        );
        assert_eq!(log.get(noise).map(|e| e.repeats), Some(2));
    }

    #[test]
    fn a_long_text_is_cut_on_a_character_boundary() {
        let mut log = EventLog::default();
        let long = "é".repeat(MAX_EVENT_TEXT); // two bytes each
        let seq = log.push(1, EventKind::Shard, Severity::Info, long);
        let text = &log.get(seq).expect("held").text;
        assert!(text.len() <= MAX_EVENT_TEXT + '…'.len_utf8());
        assert!(text.ends_with('…'));
        assert!(text.trim_end_matches('…').chars().all(|c| c == 'é'));
        // A short text is kept as it is.
        let seq = log.push(2, EventKind::Shard, Severity::Info, "short");
        assert_eq!(log.get(seq).expect("held").text, "short");
    }

    #[test]
    fn a_zero_capacity_log_still_holds_one_entry() {
        let mut log = EventLog::new(0);
        log.push(1, EventKind::Shard, Severity::Info, "a");
        log.push(2, EventKind::Shard, Severity::Info, "b");
        assert_eq!(log.len(), 1);
        assert_eq!(log.newest_first().next().map(|e| &*e.text), Some("b"));
    }
}
