//! Scratch paths for tests.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A path in the system's temp directory, unique per call:
/// `<temp>/<prefix>-<tag>-<pid>-<nanos>-<n>`.
///
/// `nanos` is the sub-second part of the time since the Unix epoch (0 if the
/// clock is before it) and `n` counts the calls of this process. Nothing is
/// created on disk.
pub fn temp_path(prefix: &str, tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{prefix}-{tag}-{}-{nanos}-{n}", std::process::id()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn a_thousand_calls_give_a_thousand_paths() {
        let paths: HashSet<PathBuf> = (0..1000).map(|_| temp_path("p", "t")).collect();
        assert_eq!(paths.len(), 1000);
    }
}
