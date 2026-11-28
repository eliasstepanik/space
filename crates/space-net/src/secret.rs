//! Holding a shared secret, and comparing a presented one with it.
//!
//! Three things in the cluster are gated by a shared secret a peer presents in
//! a message: the orchestrator's web UI (its token), a shard's registration
//! with the orchestrator (the cluster secret in `ShardHello`) and a shard's
//! hello to another shard (the peer token). All three compare with
//! [`constant_time_eq`], so how long a refusal takes says nothing about how
//! much of a guess was right.
//!
//! Whoever holds one of them holds it in a [`Secret`]: a string whose `Debug`
//! never shows it, so that a struct with a secret in it can derive `Debug` —
//! and be printed in a test failure or a log line — without giving it away.

/// A secret that never shows in a `Debug` print (a config, a control plane or
/// a shard's front end is printed in test failures and may end up in a log).
///
/// There is no `Display`: the only way to the text is [`Secret::expose`],
/// which is easy to find in a review.
#[derive(Clone, PartialEq, Eq, Default)]
pub struct Secret(String);

impl Secret {
    /// Wrap `secret`.
    #[must_use]
    pub fn new(secret: impl Into<String>) -> Self {
        Self(secret.into())
    }

    /// The secret itself: to compare with, or to hand to whoever must present
    /// it.
    #[must_use]
    pub fn expose(&self) -> &str {
        &self.0
    }

    /// Whether there is none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl core::fmt::Debug for Secret {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(if self.0.is_empty() {
            "Secret(<none>)"
        } else {
            "Secret(<redacted>)"
        })
    }
}

/// Whether `given` equals `secret`, in time that depends only on `given`'s
/// length — not on where the two differ, and not on the secret.
///
/// An empty secret matches nothing: "no secret configured" must be decided by
/// the caller, never by an empty guess happening to equal it.
#[must_use]
pub fn constant_time_eq(given: &[u8], secret: &[u8]) -> bool {
    if secret.is_empty() {
        return false;
    }
    let mut diff = u8::from(given.len() != secret.len());
    let mut at = 0usize;
    for byte in given {
        // Every byte of the guess is compared, against the secret repeated as
        // often as needed; there is no early exit to time.
        diff |= byte ^ secret.get(at).copied().unwrap_or(0);
        at += 1;
        if at == secret.len() {
            at = 0;
        }
    }
    std::hint::black_box(diff) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_secret_itself_matches() {
        assert!(constant_time_eq(b"correct horse", b"correct horse"));
        for wrong in [
            &b""[..],
            b"correct hors",
            b"correct horsf",
            b"correct horse ",
            b"Correct horse",
        ] {
            assert!(!constant_time_eq(wrong, b"correct horse"), "{wrong:?}");
        }
        // A guess that is the secret repeated is still the wrong length.
        assert!(!constant_time_eq(b"abab", b"ab"));
    }

    #[test]
    fn an_empty_secret_matches_nothing() {
        assert!(!constant_time_eq(b"", b""));
        assert!(!constant_time_eq(b"x", b""));
    }

    /// Mutation this catches: deriving `Debug` on the wrapper (every struct
    /// that holds one would print it).
    #[test]
    fn a_secret_is_never_printed() {
        let secret = Secret::new("do-not-print-this-one");
        assert_eq!(secret.expose(), "do-not-print-this-one");
        assert!(!secret.is_empty());
        let printed = format!("{secret:?} {:?}", Some(&secret));
        assert!(!printed.contains("do-not-print"), "{printed}");
        assert!(printed.contains("<redacted>"), "{printed}");
        assert_eq!(format!("{:?}", Secret::default()), "Secret(<none>)");
        assert!(Secret::default().is_empty());
    }
}
