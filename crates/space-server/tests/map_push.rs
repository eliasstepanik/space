//! The orchestrator's committed-map push, as a shard consumes it (plan F9).
//!
//! An epoch commit names ownership only, so a shard that learns of a brand-new
//! neighbour through a commit holds an un-dialable sentinel endpoint
//! ([`ShardRegion::UNRESOLVED_ENDPOINT`]) for it. The push
//! (`DirectoryReply`) is what supplies the real address, and
//! [`Shard::peer_endpoint`] is what the transport dials.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use space_protocol::{
    DirectoryReply, EntityGuid, EntitySnapshot, EpochPrepare, HandoffOffer, Message, ShardId,
    ShardMap, ShardRegion,
};
use space_server::net::ShardNet;
use space_server::shard::{Destination, Shard};
use space_server::ShardConfig;
use space_world::{octree, NodeRef};
use tokio::sync::Mutex;

fn addr(port: u16) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)
}

fn root() -> NodeRef {
    NodeRef::new(40, [0, 0, 0])
}

/// Shard 1 (listening on 7001) with a map where shard 2's carved child still has
/// the placeholder endpoint a commit gives it: its parent's own address.
fn shard_with_placeholder() -> Shard {
    let child = octree::children(root()).unwrap()[0];
    let mut shard = Shard::new(ShardConfig::new(ShardId::new(1), addr(7_001), root()));
    shard.set_shard_map(ShardMap::new(
        2,
        vec![
            ShardRegion::new(root(), ShardId::new(1), addr(7_001)),
            ShardRegion::new(child, ShardId::new(2), addr(7_001)),
        ],
    ));
    shard
}

fn push(epoch: u64, child_port: u16) -> Message {
    let child = octree::children(root()).unwrap()[0];
    Message::DirectoryReply(DirectoryReply {
        shard_map: ShardMap::new(
            epoch,
            vec![
                ShardRegion::new(root(), ShardId::new(1), addr(7_001)),
                ShardRegion::new(child, ShardId::new(2), addr(child_port)),
            ],
        ),
        tick: 0,
    })
}

#[test]
fn a_placeholder_endpoint_is_never_offered_as_a_peer_address() {
    let shard = shard_with_placeholder();
    assert_eq!(
        shard.peer_endpoint(ShardId::new(2)),
        None,
        "the child's placeholder is this shard's own address: dialling it loops back"
    );
    assert_eq!(shard.peer_endpoint(ShardId::new(9)), None, "unknown shard");
}

#[test]
fn the_map_push_supplies_the_real_endpoint() {
    let mut shard = shard_with_placeholder();
    shard.orchestrator_message(push(2, 7_002));
    assert_eq!(shard.peer_endpoint(ShardId::new(2)), Some(addr(7_002)));
    assert_eq!(
        shard.epoch(),
        2,
        "a same-epoch push only refreshes endpoints"
    );
}

#[test]
fn a_push_older_than_the_shards_map_is_ignored() {
    let mut shard = shard_with_placeholder();
    shard.orchestrator_message(push(1, 7_002));
    assert_eq!(shard.epoch(), 2);
    assert_eq!(shard.peer_endpoint(ShardId::new(2)), None);
}

#[test]
fn a_newer_push_is_installed_like_a_commit_the_shard_missed() {
    let mut shard = shard_with_placeholder();
    shard.orchestrator_message(push(3, 7_002));
    assert_eq!(shard.epoch(), 3);
    assert_eq!(shard.peer_endpoint(ShardId::new(2)), Some(addr(7_002)));
}

#[test]
fn the_unresolved_sentinel_is_never_offered_as_a_peer_address() {
    let child = octree::children(root()).unwrap()[0];
    let mut shard = Shard::new(ShardConfig::new(ShardId::new(1), addr(7_001), root()));
    shard.set_shard_map(ShardMap::new(
        2,
        vec![
            ShardRegion::new(root(), ShardId::new(1), addr(7_001)),
            ShardRegion::new(child, ShardId::new(2), ShardRegion::UNRESOLVED_ENDPOINT),
        ],
    ));
    assert_eq!(ShardRegion::UNRESOLVED_ENDPOINT.port(), 0);
    assert_eq!(
        shard.peer_endpoint(ShardId::new(2)),
        None,
        "a child that has not registered is un-dialable"
    );
    // The real address arrives with the push and is then used.
    shard.orchestrator_message(push(2, 7_002));
    assert_eq!(shard.peer_endpoint(ShardId::new(2)), Some(addr(7_002)));
}

fn prepare(epoch: u64) -> Message {
    Message::EpochPrepare(EpochPrepare {
        epoch,
        activation_tick: 100,
        changes: vec![],
    })
}

/// Whether the shard's outbox holds an `EpochAck` for `epoch` (draining it).
fn acked(shard: &mut Shard, epoch: u64) -> bool {
    shard
        .take_outbox()
        .iter()
        .any(|o| matches!(&o.message, Message::EpochAck(a) if a.epoch == epoch))
}

/// A map push that installs epoch N must also clear a prepare staged for N: a
/// leftover staged prepare would make the shard refuse every later prepare.
#[test]
fn installing_a_newer_epoch_via_a_push_clears_the_staged_prepare() {
    let mut shard = shard_with_placeholder(); // epoch 2
    shard.orchestrator_message(prepare(3));
    assert!(acked(&mut shard, 3), "prepare 3 is acked");
    assert!(shard.epoch_participant().has_staged());

    // The commit for 3 was missed; the push installs epoch 3 instead.
    shard.orchestrator_message(push(3, 7_002));
    assert_eq!(shard.epoch(), 3);
    assert!(
        !shard.epoch_participant().has_staged(),
        "the staged prepare for 3 is superseded"
    );

    shard.orchestrator_message(prepare(4));
    assert!(
        acked(&mut shard, 4),
        "prepare 4 is not wedged by the old one"
    );
}

/// A shard on `127.0.0.1:0` with its transport serving.
async fn serving_shard(
    id: u32,
    map_for: impl Fn(SocketAddr) -> ShardMap,
) -> (Arc<ShardNet>, Arc<Mutex<Shard>>) {
    let net = Arc::new(ShardNet::bind(addr(0)).expect("bind"));
    let bound = net.local_addr().expect("local addr");
    let mut shard = Shard::new(ShardConfig::new(ShardId::new(id), bound, root()));
    shard.set_shard_map(map_for(bound));
    shard.set_endpoint(bound);
    let shard = Arc::new(Mutex::new(shard));
    tokio::spawn(Arc::clone(&net).serve(Arc::clone(&shard)));
    (net, shard)
}

/// Poll `shard`'s outbox until it holds a `StaleEpoch` (the observable reaction to
/// an offer under the wrong epoch), up to a deadline.
async fn received_stale_epoch(shard: &Arc<Mutex<Shard>>, within: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + within;
    loop {
        let seen = shard
            .lock()
            .await
            .take_outbox()
            .iter()
            .any(|o| matches!(o.message, Message::StaleEpoch(_)));
        if seen {
            return true;
        }
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// A cached peer sender must repoint when a same-epoch map refresh changes the
/// peer's address: the next message goes to the NEW receiver, not the old one.
///
/// Mutation this catches: `peer_sender` reusing the cached sender regardless of
/// its address — the second offer would reach the old receiver.
#[tokio::test(flavor = "multi_thread")]
async fn a_map_refresh_that_changes_a_peers_address_repoints_the_cached_sender() {
    let child = octree::children(root()).unwrap()[0];
    let map_with = |own: SocketAddr, peer_id: u32, peer: SocketAddr, own_id: u32| {
        ShardMap::new(
            5,
            vec![
                ShardRegion::new(root(), ShardId::new(own_id), own),
                ShardRegion::new(child, ShardId::new(peer_id), peer),
            ],
        )
    };
    // Two receivers, both shard 2, both at epoch 5 and both knowing shard 1.
    let (_old_net, old) = serving_shard(2, |own| map_with(own, 1, addr(7_001), 2)).await;
    let (_new_net, new) = serving_shard(2, |own| map_with(own, 1, addr(7_001), 2)).await;
    let old_addr = old.lock().await.shard_map().regions[0].endpoint;
    let new_addr = new.lock().await.shard_map().regions[0].endpoint;
    assert_ne!(old_addr, new_addr);

    // The sender (shard 1) first believes shard 2 lives at the old address.
    let (sender_net, sender) = serving_shard(1, |own| map_with(own, 2, old_addr, 1)).await;

    let offer = |epoch: u64| {
        let guid = EntityGuid::new(ShardId::new(1), 1);
        Message::HandoffOffer(HandoffOffer {
            entity: guid,
            from: ShardId::new(1),
            to: ShardId::new(2),
            // Wrong epoch: the receiver answers with a StaleEpoch we can observe.
            epoch,
            state: Box::new(EntitySnapshot::new(guid, vec![])),
            activation_tick: 10,
        })
    };

    sender
        .lock()
        .await
        .send_to(Destination::Shard(ShardId::new(2)), offer(99));
    sender_net.flush(&sender).await;
    assert!(
        received_stale_epoch(&old, Duration::from_secs(5)).await,
        "the first offer reaches the old receiver"
    );

    // A same-epoch refresh moves shard 2 to the new address.
    let refreshed = map_with(sender_addr(&sender).await, 2, new_addr, 1);
    sender
        .lock()
        .await
        .orchestrator_message(Message::DirectoryReply(DirectoryReply {
            shard_map: refreshed,
            tick: 0,
        }));
    assert_eq!(sender.lock().await.epoch(), 5, "a same-epoch refresh");
    assert_eq!(
        sender.lock().await.peer_endpoint(ShardId::new(2)),
        Some(new_addr)
    );

    sender
        .lock()
        .await
        .send_to(Destination::Shard(ShardId::new(2)), offer(99));
    sender_net.flush(&sender).await;
    assert!(
        received_stale_epoch(&new, Duration::from_secs(5)).await,
        "after the refresh the offer reaches the NEW receiver"
    );
    assert!(
        !received_stale_epoch(&old, Duration::from_millis(300)).await,
        "and the old receiver sees nothing more"
    );
}

async fn sender_addr(shard: &Arc<Mutex<Shard>>) -> SocketAddr {
    shard.lock().await.shard_map().regions[0].endpoint
}
