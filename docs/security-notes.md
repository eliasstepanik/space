# Security notes: known gaps and deferred items

This file records what the cluster does **not** protect against today. It
describes current behaviour, not intent.

## Peer (shard↔shard) authentication: current behaviour

* **Transport.** QUIC with a self-signed certificate (`space-net/src/crypto.rs`)
  and no certificate pinning. A dialler cannot verify which process it reached,
  and an acceptor requests no client certificate.
* **Hello.** `space_protocol::check_hello` checks the protocol version only.
  A peer identifies itself with the plain string `shard:<id>`
  (`space-server/src/net.rs`, `peer_shard_id`); any client can type it.
* **Orchestrator.** `ControlPlane::handle_shard_hello` checks the cluster name
  only. It *issues* a `peer_token` in `ShardWelcome`, but that token is never
  presented on peer connections and never verified anywhere.
* **Shard.** `Shard::peer_message` checks only that the claimed shard id is in
  the shard's own map (and a commit must come from the shard that staged the
  transfer). A party that can reach a shard's port and guesses a valid
  neighbour id can speak the handoff protocol as that neighbour.
* **Control port.** The orchestrator's control endpoint binds to `127.0.0.1` by
  default. That is the only mitigation for the items below.

## Deferred items

1. **`peer_token` enforcement in `Shard::peer_message`.** Shards should present
   the token they were issued, and receivers should verify it before acting on
   a peer message. Today the field is a wire slot with no consumer.
2. **Per-shard hello credentials.** The hello should carry a credential bound to
   one shard id (and be checked by the orchestrator), instead of a bare
   `shard:<id>` string, plus certificate pinning or a separate ALPN for peer
   traffic. A signed handoff ticket would additionally stop a legitimate shard
   forging handoffs for entities it does not own.
3. **Endpoint-table poisoning via claimed shard ids.** The orchestrator records
   the endpoint a `ShardHello` names for the shard id it claims, and pushes
   those endpoints to every shard in a `DirectoryReply`. A party that can reach
   the control port can register under a claimed id and redirect peer traffic
   or client redirects. This is mitigated today only by the loopback-default
   control bind; exposing the control port on a non-loopback address removes
   that mitigation.
<<<<<<< HEAD

## Known limitations (Phase 3 close-out)

* The H13 frame-gap test relies on lockstepped harness ticks; production continuity rests on the separately-tested commit-gate machinery.
* The Dockerfiles and compose files are correct-by-inspection only (no Docker daemon in CI).
* The Phase-4 present path (swapchain) is validated by build + clippy only (no window system in CI).
=======
>>>>>>> refs/remotes/origin/main
