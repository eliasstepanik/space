//! The operator's way into the run loop: read a [`Snapshot`], send a command.
//!
//! The orchestrator's event loop owns every decision. A front end (the web UI,
//! [`crate::webui`]) never touches the orchestrator's state; it holds an
//! [`OpsHandle`], which can do exactly two things:
//!
//! * **read** the latest [`Snapshot`] the loop published — a shared, immutable
//!   copy, so reading never waits on the loop and the loop never waits on a
//!   reader;
//! * **ask** for an [`OperatorCommand`]. The request is *offered* to a small
//!   bounded queue (a full queue is an immediate [`OpsError::Busy`], never a
//!   wait), the loop takes it between two beats, runs it through the same
//!   synchronous code path the autoscaler uses
//!   ([`crate::orchestrator::Orchestrator::handle_operator`]) and answers at
//!   once with an [`OperatorReply`]: accepted, or refused with the reason.
//!   Nothing is queued "for later" — a split that cannot start now is refused
//!   now.
//!
//! So a slow, stuck or hostile front end can cost the loop one queue slot and
//! nothing else, and no command can bypass the epoch protocol (H10, H12).

use std::sync::Arc;
use std::time::Duration;

use space_protocol::shard::ShardId;
use space_protocol::{Scene, SceneState};
use tokio::sync::{mpsc, oneshot, watch};

use crate::snapshot::{octant_signs, Snapshot};

/// How many operator requests may wait for the loop at once.
pub const REQUEST_QUEUE: usize = 16;

/// How long a front end waits for the loop's answer before giving up.
///
/// The loop answers within one beat (33 ms) unless it is stalled; this only
/// bounds how long a request can hold a front-end connection.
pub const REPLY_TIMEOUT: Duration = Duration::from_secs(3);

/// Something an operator can ask the orchestrator to do.
///
/// Every variant is carried out by the run loop through the code path that
/// already existed for it; none of them kills a shard, resets state or edits
/// an id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperatorCommand {
    /// Split `shard`'s region: carve one octant out for a new shard, through
    /// the epoch protocol.
    Split {
        /// The shard whose region is split.
        shard: ShardId,
        /// The octant to carve (0..=7; bit 0 = +x, bit 1 = +y, bit 2 = +z), or
        /// `None` for the autoscaler's choice: the first free octant counting
        /// round from the one holding the region's centre.
        octant: Option<u8>,
        /// The map epoch the operator was looking at. When given and the
        /// committed map has moved on, the command is refused rather than
        /// applied to a map the operator has not seen.
        expected_epoch: Option<u64>,
    },
    /// Merge `shard`'s region back into the region around it (its parent
    /// node after an octant split, the nearest enclosing region for a carved
    /// cell), through the epoch protocol. Only a shard whose fresh load report
    /// says it has no players is merged (H13).
    Merge {
        /// The shard whose region is merged away.
        shard: ShardId,
        /// See [`OperatorCommand::Split::expected_epoch`].
        expected_epoch: Option<u64>,
    },
    /// Stop the autoscaler from acting until it is resumed. Reports are still
    /// received and shown; operator splits and merges still work.
    PauseAutoscaler,
    /// Let the autoscaler act again. Every shard needs a fresh, full dwell.
    ResumeAutoscaler,
    /// Send the committed map, with every known shard's current address, to
    /// every shard in it — the same push that follows each commit.
    PushMap,
    /// Make `scene` the cluster's scene: persist it, raise the revision and
    /// tell every connected shard, which tells its clients. Not an epoch
    /// change — it is carried out at once whether or not one is in flight.
    SetScene {
        /// The scene every shard switches to.
        scene: Scene,
    },
}

impl core::fmt::Display for OperatorCommand {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            OperatorCommand::Split {
                shard,
                octant: Some(octant),
                ..
            } => write!(
                f,
                "split {shard}, octant {octant} ({})",
                octant_signs(*octant)
            ),
            OperatorCommand::Split {
                shard,
                octant: None,
                ..
            } => write!(f, "split {shard}, first free octant from the centre"),
            OperatorCommand::Merge { shard, .. } => {
                write!(f, "merge {shard} into the region around it")
            }
            OperatorCommand::PauseAutoscaler => f.write_str("pause the autoscaler"),
            OperatorCommand::ResumeAutoscaler => f.write_str("resume the autoscaler"),
            OperatorCommand::PushMap => f.write_str("re-send the committed map"),
            OperatorCommand::SetScene { scene } => {
                write!(f, "switch every shard to scene {scene}")
            }
        }
    }
}

/// The loop's answer to an [`OperatorCommand`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorReply {
    /// Whether the command was carried out (for a split or merge: whether its
    /// epoch was *proposed* — it can still abort at its deadline).
    pub accepted: bool,
    /// What was started, or why the command was refused.
    pub message: String,
    /// The number of the event-log entry recording this request.
    pub event: u64,
    /// The scene and revision this request made current, when it changed the
    /// scene (`None` for every other command, for a refusal, and for a request
    /// for the scene that was already running). A front end uses it to wait
    /// for the shards' confirmations before it shows the result.
    pub scene_changed: Option<SceneState>,
}

/// Why a request got no [`OperatorReply`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpsError {
    /// Too many requests are already waiting for the loop.
    Busy,
    /// The orchestrator's loop is not running (any more).
    Stopped,
    /// The loop did not answer within [`REPLY_TIMEOUT`]. The command may still
    /// be carried out.
    TimedOut,
}

impl core::fmt::Display for OpsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            OpsError::Busy => "the orchestrator has too many operator requests waiting; try again",
            OpsError::Stopped => "the orchestrator's control loop is not running",
            OpsError::TimedOut => {
                "the orchestrator did not answer in time; reload to see whether the request was applied"
            }
        })
    }
}

impl std::error::Error for OpsError {}

/// A request in the queue: the command and where its answer goes.
#[derive(Debug)]
pub struct OperatorRequest {
    /// What is asked for.
    pub command: OperatorCommand,
    /// Where the loop sends its answer.
    pub reply: oneshot::Sender<OperatorReply>,
}

/// The front end's side: read snapshots, offer commands. Cheap to clone.
#[derive(Debug, Clone)]
pub struct OpsHandle {
    requests: mpsc::Sender<OperatorRequest>,
    state: watch::Receiver<Arc<Snapshot>>,
}

/// The run loop's side: receive commands, publish snapshots.
#[derive(Debug)]
pub struct OpsPort {
    requests: mpsc::Receiver<OperatorRequest>,
    state: watch::Sender<Arc<Snapshot>>,
}

/// A connected handle/port pair whose first snapshot is `initial`.
#[must_use]
pub fn channel(initial: Snapshot) -> (OpsHandle, OpsPort) {
    let (request_tx, request_rx) = mpsc::channel(REQUEST_QUEUE);
    let (state_tx, state_rx) = watch::channel(Arc::new(initial));
    (
        OpsHandle {
            requests: request_tx,
            state: state_rx,
        },
        OpsPort {
            requests: request_rx,
            state: state_tx,
        },
    )
}

impl OpsHandle {
    /// The latest snapshot the loop published.
    ///
    /// Never waits: it clones a shared pointer.
    #[must_use]
    pub fn snapshot(&self) -> Arc<Snapshot> {
        // The borrow is released before this returns: nothing holds the
        // channel's lock while a page is rendered.
        self.state.borrow().clone()
    }

    /// Wait until the loop has published a snapshot `done` is satisfied with,
    /// for at most `limit`, and return the latest snapshot either way.
    ///
    /// This only ever *reads* what the loop publishes of its own accord (after
    /// every logged event, and several times a second regardless): it asks the
    /// loop for nothing, so however many front-end requests wait here, the
    /// loop does not notice.
    pub async fn wait_until(
        &self,
        limit: Duration,
        done: impl Fn(&Snapshot) -> bool,
    ) -> Arc<Snapshot> {
        let mut state = self.state.clone();
        let deadline = tokio::time::Instant::now() + limit;
        loop {
            let latest = state.borrow_and_update().clone();
            if done(&latest) {
                return latest;
            }
            match tokio::time::timeout_at(deadline, state.changed()).await {
                Ok(Ok(())) => {}
                // Out of time, or the loop is gone: what there is, is all
                // there will be.
                _ => return state.borrow().clone(),
            }
        }
    }

    /// Ask the loop to carry out `command` and wait for its answer.
    ///
    /// # Errors
    ///
    /// [`OpsError`] when the request could not be handed over or was not
    /// answered; the loop itself is never made to wait.
    pub async fn request(&self, command: OperatorCommand) -> Result<OperatorReply, OpsError> {
        let (reply, answer) = oneshot::channel();
        self.requests
            .try_send(OperatorRequest { command, reply })
            .map_err(|refused| match refused {
                mpsc::error::TrySendError::Full(_) => OpsError::Busy,
                mpsc::error::TrySendError::Closed(_) => OpsError::Stopped,
            })?;
        match tokio::time::timeout(REPLY_TIMEOUT, answer).await {
            Ok(Ok(reply)) => Ok(reply),
            Ok(Err(_)) => Err(OpsError::Stopped),
            Err(_) => Err(OpsError::TimedOut),
        }
    }
}

impl OpsPort {
    /// The next request, or `None` once every [`OpsHandle`] is gone.
    /// Cancel-safe: a request is either returned or still queued.
    pub async fn next_request(&mut self) -> Option<OperatorRequest> {
        self.requests.recv().await
    }

    /// Make `snapshot` the one every handle reads from now on. Never waits.
    pub fn publish(&self, snapshot: Snapshot) {
        self.state.send_replace(Arc::new(snapshot));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_command_reads_as_what_it_does() {
        let split = OperatorCommand::Split {
            shard: ShardId::new(1),
            octant: Some(3),
            expected_epoch: None,
        };
        assert_eq!(split.to_string(), "split shard 1, octant 3 (+x +y -z)");
        let auto = OperatorCommand::Split {
            shard: ShardId::new(1),
            octant: None,
            expected_epoch: Some(4),
        };
        assert!(auto.to_string().contains("first free octant"));
        let merge = OperatorCommand::Merge {
            shard: ShardId::new(2),
            expected_epoch: None,
        };
        assert_eq!(merge.to_string(), "merge shard 2 into the region around it");
        let scene = OperatorCommand::SetScene {
            scene: Scene::Galaxy,
        };
        assert_eq!(scene.to_string(), "switch every shard to scene galaxy");
    }
}
