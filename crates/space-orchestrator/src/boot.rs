//! The orchestrator's startup wiring, shared by the binary and by tests.
//!
//! [`boot`] opens the persistent id counter, pre-allocates the root owner's id,
//! builds the epoch-1 map and the [`ControlPlane`], and marks that root owner as
//! *unclaimed*: the orchestrator reserves the id (so a restart resumes above it,
//! H9) but launches no shard for it, so the first live shard that asks for the
//! root adopts it ([`ControlPlane::handle_shard_hello`]).

use space_protocol::shard::{ShardMap, ShardRegion};

use crate::config::OrchestratorConfig;
use crate::control::ControlPlane;
use crate::epoch::EpochPublisher;
use crate::ids::PersistentShardIdAllocator;

/// Build the control plane the binary runs.
///
/// The initial map holds one region at the configured root, owned by the first
/// allocated id and carrying the control port as a placeholder endpoint (the
/// adopting shard replaces it with its own address on its hello).
///
/// # Errors
///
/// The id counter file is corrupt, out of range or unreadable.
pub fn boot(config: OrchestratorConfig, peer_token: String) -> Result<ControlPlane, String> {
    let mut ids = PersistentShardIdAllocator::open(&config.state_path, config.first_shard_id)
        .map_err(|e| e.to_string())?;
    let root_owner = ids.allocate().map_err(|e| e.to_string())?;
    let map = ShardMap::new(
        1,
        vec![ShardRegion::new(
            config.root_node(),
            root_owner,
            config.bind,
        )],
    );
    let mut control = ControlPlane::new(config, ids, EpochPublisher::new(map), peer_token);
    control.set_unclaimed_root_owner(root_owner);
    Ok(control)
}
