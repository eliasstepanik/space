//! B5: the per-entity snapshot size the docs quote (~158 B) must be the real
//! encoded size, measured with the pinned bincode config, not an estimate.
//!
//! Run with `--nocapture` to print the breakdown.

use space_protocol::codec::encode_raw;
use space_protocol::{EntityGuid, EntitySnapshot, ShardId};
use space_sim::components::{Owner, ShipControls, ShipState};
use space_sim::wire::encode_ship_state;
use space_world::WorldPos;

/// Build the snapshot a ship with the three replicated components produces.
fn ship_snapshot() -> EntitySnapshot {
    let guid = EntityGuid::new(ShardId(7), 1);
    let state = ShipState::at_rest(WorldPos::from_metres([1000, -2000, 3000]));
    EntitySnapshot::new(
        guid,
        vec![
            (0, encode_ship_state(&state)),
            (1, encode_raw(&ShipControls::default())),
            (2, encode_raw(&Owner(42))),
        ],
    )
}

#[test]
fn per_entity_snapshot_size_is_measured() {
    let state = ShipState::at_rest(WorldPos::from_metres([1000, -2000, 3000]));
    let ship_bytes = encode_ship_state(&state);
    let ctrl_bytes = encode_raw(&ShipControls::default());
    let owner_bytes = encode_raw(&Owner(42));
    let snap = ship_snapshot();

    let measured = snap.encoded_len();
    println!(
        "ship_state={} controls={} owner={} entity encoded_len={}",
        ship_bytes.len(),
        ctrl_bytes.len(),
        owner_bytes.len(),
        measured
    );

    // Component payloads are the hand-rolled wire formats from `space_sim::wire`.
    assert_eq!(ship_bytes.len(), 76, "ShipState wire length");
    // 25 since protocol 6: thrust, torque and the fast-travel flag.
    assert_eq!(ctrl_bytes.len(), 25, "ShipControls wire length");
    assert_eq!(owner_bytes.len(), 8, "Owner wire length");

    // The number docs/plan.md and the Snapshot doc comment quote (~158 B; 159
    // since protocol 6 added the fast-travel flag to the controls), now
    // pinned exactly: if this trips, the real size drifted and the docs must be
    // updated to match.
    assert_eq!(
        measured, 159,
        "per-entity snapshot size drifted; update the docs that quote ~158 B"
    );
}
