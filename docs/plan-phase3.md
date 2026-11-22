# Phase 3 — orchestrator + autoscaler + Docker (H9–H15)

Derived from `docs/plan.md` (revision 2), the Phase-2 guarantees in
`crates/space-server/src/handoff.rs`, and the spike findings in
`spikes/orch-docker/README.md`. `docs/plan.md` names Phase 3 and its load
metric / ShardId rules but does **not** spell out H9–H15; deriving those seven
guarantees is a deliverable of this plan, in the same form as Phase 2 derived
H1–H8.

Sandbox reality that shapes every decision below (`spikes/ENVIRONMENT.md`,
`spikes/orch-docker/README.md`):

* **No Docker daemon, no `docker` CLI, no `/var/run/docker.sock`.** The spike
  proves the bollard mechanics only as pure-helper unit tests (13 tests) and
  records `detect_docker: Socket not found`. Docker E2E is an **entry
  criterion**, not something this phase can run.
* Everything else runs here: loopback QUIC (quinn), `tokio` timers, in-process
  shards. Phase 2's `RunningShard` harness in
  `crates/space-server/tests/acceptance.rs` already starts a real shard on
  `127.0.0.1:0` and drives it over the real transport, so the whole
  split/merge/rebalance loop is testable in-sandbox.
* 100% Rust first-party. The one documented C-binding exception is `ring`
  (TLS crypto, `crates/space-net/src/crypto.rs`). New deps allowed for this
  phase: `bollard` 0.21.1 + `bollard-stubs` + `serde_json` + `futures-util`
  (all pure Rust; already vendored in the sandbox cargo cache from the spike).
  No new C-binding crate may be added.

---

## 0. Facts the plan is built on (verified by reading the code)

| # | fact | where |
|---|------|-------|
| F1 | `Shard` is a synchronous 30 Hz state machine; `tick()` is pure and deterministic; the quinn front end is the only async part. | `crates/space-server/src/shard.rs` (module docs, `tick`) |
| F2 | `ShardIdAllocator` exists, is monotonic, has no `free`, panics past `u32::MAX`/`ShardId::NONE`, and takes its start value from the caller (`new(first)`) — i.e. it is *ready* to be resumed from a persisted counter. It is **not** referenced by `main.rs`. | `crates/space-server/src/alloc.rs` |
| F3 | `main.rs` takes `--shard <id>` from argv and builds the shard map from `--region` + one `--peer` per neighbour, epoch hardcoded to 1 (`ShardMap::new(1, …)`). This is Phase-2 MAJOR 3: production ids are hand-written, not allocated. | `crates/space-server/src/main.rs` |
| F4 | The epoch protocol is already on the wire and already consumed by handoff: `EpochPrepare/EpochAck/EpochCommit/EpochAbort` are `Message` variants 13–16; handoff offer/accept/commit are epoch-checked, a stale-epoch message answers `StaleEpoch`, an in-flight transfer aborts with `HandoffReason::EpochChanged` the moment the map epoch moves (before *and* after the commit gate). **No shard code currently sends or handles `EpochPrepare`** — grep for it in `crates/space-server` returns nothing. The orchestrator is the missing half. | `space-protocol/src/shard.rs`, `messages.rs`, `shard.rs::handoff_pass`, `on_offer/on_accept/on_commit` |
| F5 | A split is expressed *additively*: `ShardMap` regions may nest and `lookup` keeps the **deepest** match, so "carve a child node out and give it to a new shard" is one appended region; the parent keeps the old owner. A merge is the inverse (drop the child region). | `space-protocol/src/shard.rs` (`ShardRegion` docs, `lookup`) |
| F6 | `LoadReport { shard, players, tick_p95_ms, load }` exists, and its doc fixes the metric and the thresholds: `max(players/100, tick_p95_ms/33.3)`, split at ≥0.8 for 10 s, merge at ≤0.35 for 60 s. `AdminCommand::{ForceSplit,ForceMerge,KillShard,SpawnBots,Status}` and `DirectoryQuery/DirectoryReply` also already exist. **Nothing produces or consumes them yet.** | `space-protocol/src/control.rs` |
| F7 | `Message::VARIANT_COUNT = 23` and the variant order *is* the wire tag; new variants are appended only. `PROTOCOL_VERSION = 2`. | `space-protocol/src/messages.rs`, `codec.rs` |
| F8 | `Redirect` carries a full `ShardMap`, and `Welcome` carries one, so a client always learns the new map from the shard it is already talking to — the orchestrator never has to reach a client. | `messages.rs` (`Redirect`, `Welcome`), `shard.rs::handoff_pass` |
| F9 | `Shard::set_shard_map(map)` is public and is exactly the "shards get the new map" seam; `set_endpoint` patches the shard's own region endpoint after bind. | `shard.rs` |
| F10 | Peer identity is a hello token `shard:<id>` and is only checked against "is this id in my map"; `shard.rs` explicitly states **Phase 3 owns full shard-to-shard authentication**. | `net.rs` (`peer_shard_id`, `PEER_TOKEN_PREFIX`), `shard.rs::peer_message` |
| F11 | The owner token rides a handoff as a private component (`OWNER_TOKEN_COMPONENT = u16::MAX`) because `HandoffOffer` has no token field; full auth (signed ticket) was deferred to Phase 3. | `shard.rs` (`with_owner_token`, `offer_token_of`) |
| F12 | `MAX_PLAYERS_PER_SHARD = 100`; `join` refuses with `RejectReason::Full`, and an offer to a full shard is answered `TargetFull`. `RejectReason::Draining` exists but is never produced. | `shard.rs` |
| F13 | `space_world::octree::children(node)` gives the 8 children in octant-bit order, `parent()` inverts it, `NodeRef::size_m(depth)` is the edge length, `MAX_DEPTH = 64` (one cell = 1024 m). Split geometry is free. | `space-world/src/octree.rs` |
| F14 | `SimWorld::guid_state_hash()` is the sanctioned cross-shard state hash and is what Phase 2's no-desync test compares. | `space-sim/src/sim_world.rs`, `acceptance.rs::handoff_leaves_no_desync` |
| F15 | CI runs `cargo xtask ci` = `fmt --check` + `clippy --workspace --all-targets --locked -D warnings` + `nextest run --workspace --locked`. `xtask e2e` is a `stub()` that exits 2. | `xtask/src/main.rs`, `.github/workflows/ci.yml` |
| F16 | The spike's compose template mounts the socket, uses `group_add: ${DOCKER_GID}`, pins the image by digest, publishes a UDP port from `HOST_PORT_RANGE`, labels containers `space.cluster=<id>` + `space.role=shard`, and reaps orphans by label. Its README lists the Phase-3 fixes: attach shards to the cluster network, reconcile uid 10001 vs 1000, parameterise subnet + port range. | `spikes/orch-docker/{docker-compose.yaml,Dockerfile,README.md}` |
| F17 | `bollard = "0.21.1"` default features are `["http","pipe"]`; `pipe` pulls `hyper-named-pipe`, which is **not** in the sandbox cargo cache (verified: `cargo metadata --offline` fails on it, while `--no-default-features --features http` resolves offline). | `spikes/orch-docker/Cargo.toml`, local cargo registry cache |

F17 has a concrete consequence: the workspace must depend on
`bollard = { version = "0.21.1", default-features = false, features = ["http"] }`
so `cargo … --locked` keeps working offline in this sandbox (unix-socket
connect still works — `hyperlocal` is cached and is what `connect_with_local`
uses; the named-pipe path is Windows-only and unused). If a later host needs
the pipe feature, it is a one-line feature flip, not a design change.

---

## 1. Crates and module breakdown

One new workspace member, plus small additive changes to two existing crates.

### 1.1 `crates/space-orchestrator` (new)

```
crates/space-orchestrator/
  Cargo.toml            # space-{world,protocol,net}, tokio, bollard (opt-in feature)
  src/
    lib.rs              # crate docs: the H9–H15 table (mirrors handoff.rs)
    config.rs           # OrchestratorConfig: cluster id, root region + depth,
                        #   thresholds, port range, state path, launcher choice
    ids.rs              # PersistentShardIdAllocator: wraps space_server::alloc::
                        #   ShardIdAllocator, loads/persists the counter (closes MAJOR 3)
    topology.rs         # the authoritative ShardMap + split/merge geometry
                        #   (child_of, split_plan, merge_plan) over space_world::octree
    epoch.rs            # the two-phase epoch state machine: prepare → acks → commit,
                        #   with timeout → abort; drives F4's messages
    autoscale.rs        # load metric + dwell timers → ScaleAction::{Split,Merge,None}
    launcher/
      mod.rs            # trait Launcher + LaunchSpec/RunningShard/Handle + tests
      local.rs          # in-process + subprocess launchers (sandbox-testable)
      docker.rs         # bollard impl, ported from spikes/orch-docker (feature-gated)
    control.rs          # the orchestrator's QUIC front end: LoadReport in,
                        #   EpochPrepare/Commit/Abort out, DirectoryQuery/Reply,
                        #   AdminCommand/AdminReply
    orchestrator.rs     # Orchestrator: the event loop tying the above together
    main.rs             # `space-orchestrator` binary (arg parsing in main.rs only)
  tests/
    acceptance.rs       # H9–H15 over loopback QUIC (see §5)
```

Notes on the split:

* `topology.rs` is pure and synchronous (like `Shard`): given a map and a shard
  id it returns the new map, the new epoch and the `Vec<EpochChange>`. Every
  H-worth of geometry reasoning is testable without a single socket.
* `epoch.rs` is also pure: it takes the map + the set of shards that must ack
  and produces the next `Message`s and a timer deadline. The async layer only
  moves messages.
* `launcher/mod.rs` is the seam that makes the phase testable: the orchestrator
  never names Docker.
* `control.rs` mirrors `space_server::net::ShardNet` in shape (one task per
  connection, outbox flush) so reviewers see one idiom, not two.
* `#![forbid(unsafe_code)]`, `#![warn(missing_docs, missing_debug_implementations)]`
  as in every other crate.

### 1.2 `crates/space-server` (additive changes only)

| module | change |
|---|---|
| `main.rs` | **`--shard` becomes optional.** New flags `--orchestrator <addr>` and `--claim-region <depth:x:y:z>`. With `--orchestrator`, the shard binds the same fixed default port as standalone (`127.0.0.1:7777`; several orchestrated shards on one host each need their own `--bind` — a restarted shard must come back on the same address to be recognised), asks the orchestrator for an id + region + map, and thereafter follows pushed maps. Without it, today's argv path is unchanged (regression safety for Phase 2's tests and docs). |
| `net.rs` | A control connection to the orchestrator (same `space_net::connect`, hello token `orchestrator`), reusing the persistent-connection + FIFO-sender pattern already used for peers. |
| `shard.rs` | (a) `set_shard_map` gains a sibling `apply_epoch(map)` that also aborts in-flight transfers — already covered by `handoff_pass`'s stale-epoch sweep, so this is mostly a call-order guarantee, not new logic. (b) A `draining: bool` flag: `join` answers `RejectReason::Draining` (F12 — the variant exists and is currently dead). (c) Tick-time instrumentation: a fixed-size ring of per-tick durations + `tick_p95_ms()`, so `LoadReport` can be produced honestly. |
| `alloc.rs` | Docs updated: the counter now lives in the orchestrator; the type stays here (it is where `ShardId` is understood) and `space_orchestrator::ids` composes it with persistence. |

`space-protocol` needs **no** new message variants for the core loop (F4, F6:
`EpochPrepare/Ack/Commit/Abort`, `LoadReport`, `DirectoryQuery/Reply`,
`AdminCommand/Reply` all exist). Two additions are needed and both are
appends, so `VARIANT_COUNT` goes 23 → 25 and `PROTOCOL_VERSION` stays 2 only
if we treat them as additive-compatible; the plan bumps it to **3** because
the shard↔orchestrator handshake is a new required exchange:

* `ShardHello { cluster, shard, endpoint, region }` / `ShardWelcome { shard,
  region, shard_map, peer_token }` — the shard's "give me an id and my map"
  exchange, which is also where the orchestrator hands out the peer
  authentication token (F10/F11). Appended as variants 23 and 24.

If the reviewer prefers zero protocol churn, the fallback is to smuggle the
same exchange through `Hello.player_token` + `Welcome.shard_map` (the shard
sends `hello("claim:<region>")` and reads its id back from the `Welcome`'s
map). **Decision D1: use the two new variants** — smuggling is what F11 had to
do under a freeze, and Phase 3 is not frozen.

---

## 2. The derived guarantees H9–H15

Phase 2's H1–H8 say what a *single* handoff must never do. Phase 3's H9–H15 say
what must hold when **the map itself is what changes** — when a shard is
created or destroyed under the players. Each is stated as a testable property,
each names the mechanism that makes it hold, and each maps to at least one
acceptance test in §5. They are the module-doc table of
`space-orchestrator/src/lib.rs`.

| # | guarantee | mechanism that makes it hold | test |
|---|-----------|------------------------------|------|
| **H9** | **Shard ids are allocated by the orchestrator, are strictly monotonic, and are never reused across a split, a merge, a crash or an orchestrator restart.** No code path in a production launch takes a shard id from a human. | `PersistentShardIdAllocator` owns the counter: `allocate()` delegates to `space_server::alloc::ShardIdAllocator` (F2 — no `free`, panics at the id-space limit) and every allocation is appended to a durable counter file **before** the id is handed to a launcher. A restart constructs the allocator from the persisted value, so the first id after a restart is strictly greater than the last before it. `main.rs --shard` is gone from the orchestrator-launched path (F3). | `t1`, `t2`, `t3` |
| **H10** | **A map change is published through the epoch protocol only: prepare → ack → commit. No shard ever observes a map that is not some committed epoch, and every shard that must act on a change has acknowledged it before it activates.** | `epoch.rs` never mutates the published map in place. It builds `EpochPrepare{epoch+1, activation_tick, changes}`, sends it to every shard named in `changes` **plus** every shard whose neighbour set the change touches, collects `EpochAck`, then sends `EpochCommit`. `activation_tick = max(observer ticks) + EPOCH_LEAD` so every shard switches on its own clock with slack. A missing ack before the deadline sends `EpochAbort` and the published map is byte-identical to what it was. | `t4`, `t5` |
| **H11** | **A split or merge never loses, duplicates or double-steps a player, and never leaves a position owned by two shards or by none.** The map is a total, non-conflicting ownership function before and after every change. | Splits are **additive** (F5): the child node is appended with the new owner and the parent keeps the old one, so `lookup` (deepest match) changes ownership only inside the child. Merges remove the child region, which is the exact inverse. `topology.rs` asserts the invariant on every produced map: for every octant boundary in the affected parent, `lookup` returns the old owner outside the child and the new owner inside, and no two regions of equal depth overlap. Player movement is then Phase 2's job: the new map makes `handoff_pass`'s existing boundary test fire for exactly the ships that are now on the far side, and H1–H8 do the rest. | `t6`, `t7`, `t8` |
| **H12** | **An epoch change cannot strand a handoff.** Every transfer in flight when the epoch moves terminates within `HANDOFF_TIMEOUT_TICKS`, the offering shard keeps simulating the entity, and the guid becomes re-offerable. | This is Phase 2's guarantee 7 + 8, and Phase 3's job is to *not break it* while generating epoch changes far more often than Phase 2 ever did. The orchestrator therefore never bumps the epoch while it has an un-acked prepare outstanding (one epoch change at a time, serialised in `epoch.rs`), and it never bumps twice inside `HANDOFF_TIMEOUT_TICKS` for the same region pair. | `t9` |
| **H13** | **A split or merge is invisible to a connected client: its guid is unchanged, it receives at least one frame per tick across the transition with no gap in tick numbering, and it is never told "nothing happened here".** | Reuses Phase 2's make-before-break: the client is `Redirect`ed with the *new* map (F8) and keeps receiving from the old shard until the activation tick. Phase 3 adds nothing on the client path — which is the point. The measurable form: for the transitioning session, the union of frames it receives over the transition has (a) no tick gap larger than 1, (b) its own ship present in every frame after the first, (c) the same `EntityGuid` throughout, and (d) no `Reject` at any point. | `t10`, `t11` |
| **H14** | **The autoscaler is deterministic: the same sequence of `LoadReport`s produces the same sequence of decisions, and it never flaps.** `load = max(players/100, tick_p95_ms/33.3)`; split at `load ≥ 0.8` sustained 10 s, merge at `load ≤ 0.35` sustained 60 s; a shard that was just split or merged is quiet for a cooldown before it can act again. | `autoscale.rs` is a pure function of `(reports, clock, prior state)` with **no wall clock and no RNG**: the caller supplies the tick, dwell is counted in ticks (`ticks_in(10.0)` = 300, `ticks_in(60.0)` = 1800), and the hysteresis band `0.35 < load < 0.8` clears both timers. One scale action is in flight at a time (shared with H10's serialisation), so a split cannot be followed by a merge of its own child before the split has settled. | `t12`, `t13`, `t14` |
| **H15** | **The orchestrator converges the world back to its declared topology after a shard dies, and never re-uses the dead shard's id or leaves its region unowned.** | The launcher reports liveness (`Launcher::poll`), and a shard that misses `LOAD_REPORT_INTERVAL` beats is declared dead: its region is reassigned to a surviving shard via the epoch protocol (H10), and the replacement — if one is needed — gets a fresh id from H9's counter. On startup the orchestrator reconciles *before* serving: it loads the persisted counter and map, lists what the launcher actually has running, reaps what is not in the map, and adopts what is. | `t15`, `t16` |

Two things deliberately **not** promoted to guarantees, recorded so the
reviewer does not read them as omissions:

* **Full shard-to-shard authentication (F10/F11).** Phase 3 *issues* an
  orchestrator peer token (`ShardWelcome.peer_token`) but does **not** enforce
  it: shards never present it on peer connections and nothing verifies it, so
  `Shard::peer_message` still applies only the bare "id is in my map" test.
  "Anyone can be a peer" is closed only to the extent a claimed id must be in
  the map; "anyone can impersonate a known peer" remains open, as does handoff
  forgery by a shard holding a valid id. Enforcement and a signed handoff
  ticket are deferred; see `docs/security-notes.md`.
  `space-net/src/crypto.rs`'s "pinning real certificates is an orchestrator
  (Phase 3) concern" line remains true: no certificate is pinned.
* **Cross-host scheduling.** The launcher places shards on one Docker host.
  Multi-host placement is a `Launcher` impl detail and is out of scope.

---

## 3. Design of the four moving parts

### 3.1 ShardId allocation (closes Phase-2 MAJOR 3)

```rust
// space-orchestrator/src/ids.rs
pub struct PersistentShardIdAllocator { inner: ShardIdAllocator, sink: Box<dyn CounterSink> }
impl PersistentShardIdAllocator {
    pub fn open(path: &Path, first: u64) -> Result<Self, InitError>; // reads the file if present
    pub fn allocate(&mut self) -> ShardId;   // persist-then-return, never return-then-persist
    pub fn peek(&self) -> u64;
}
```

* **Persist before hand-out**, not after. A crash between "allocated" and
  "persisted" would be the one window in which an id could be reused; writing
  first makes the worst case a skipped id, which is harmless (F2's rule is
  *never reused*, not *contiguous*).
* Format: one line, the decimal next-id, written to `state_path` with
  write-to-temp + rename (atomic on POSIX; on Windows `rename` over an existing
  file needs `MOVEFILE_REPLACE_EXISTING`-equivalent handling, so the writer
  deletes-then-renames and the reader tolerates a missing file by falling back
  to `first`). CI runs Windows, so this is a real requirement, not a detail.
* `first` defaults to 1 and is a config value, so a test can prove "restart
  resumes strictly above the last id" without a long run.
* `main.rs`'s `--shard` stays accepted for the standalone path but the
  orchestrator path never uses it; the acceptance test asserts that a shard
  launched through the launcher receives its id from the orchestrator and that
  the id is not present in the launch spec.

### 3.2 Topology and split geometry (`topology.rs`, pure)

Geometry arithmetic first, because it is easy to get wrong:
`NodeRef::size_m(d) = CELL_SIZE × 2^(64−d)` with `CELL_SIZE = 1024 m`, so
`size_m(64) = 1024 m` (one cell), `size_m(54) = 1 048 576 m = 1024 km`,
`size_m(44) ≈ 1.07 × 10⁹ m`. A child is exactly half its parent's edge.

Constants:
* `SPLIT_MIN_NODE_SIZE_M = 1_048_576` — only a region whose edge is at least
  1024 km (i.e. **depth ≤ 54**) may be split. Below that a region is still ~100×
  the 10 km interest radius, so splitting it changes routing without relieving
  load.
* `SPLIT_MAX_DEPTH = 60` — see the ceiling note below.
* `MERGE_MIN_PLAYERS = 1`.
* `OrchestratorConfig::root_region` must be declared at a depth with split
  headroom (default **depth 40**, ≈ 2.7 × 10¹⁰ m). This matters: Phase 2's tests
  and `main.rs`'s examples use **depth-64** regions (`acceptance.rs:41`,
  `DEPTH = 64`), and `octree::children` returns `None` at depth 64 — a
  depth-64 root can never be split. The orchestrator's own config therefore
  carries its own root depth and does not inherit the test convention.

* `split_plan(map, shard) -> Option<Split>`: pick the octant axis with the most
  players on each side (from the last `LoadReport`s, falling back to octant 0),
  take `children(node)[octant]` (F13) as the carved-out child, allocate one new
  id, and return `EpochChange(vec![(child, new_id)])` + the appended
  `ShardRegion`. Deeper splits recurse into the child on the next round, so a
  hot region fans out one octant at a time rather than 8-wide in one step.
* `merge_plan(map, shard) -> Option<Merge>`: the merge target is the
  **parent** if the shard's node is a strict child in the map (the common case
  after a split), otherwise the lowest-load sibling. The change is
  `EpochChange(vec![(child, parent_owner)])` and the region is *removed*, so
  `lookup` falls back to the parent (F5).
* `validate(&map) -> Result<(), TopologyError>`: no two equal-depth regions
  overlap; every region's shard is in the live set; every node is valid at its
  depth (the same rules `space_protocol::shard::validate_node` enforces on
  decode). Called on every produced map — H11's structural half.

**Ceiling on split depth.** `validate_node` requires, at depth `d ≥ 1`, each
axis index to lie in `[-2^(d−1), 2^(d−1))`. At `d = 64` that interval is
`[-2^63, 2^63)`, which *exactly saturates* `i64` — every `i64` index is accepted
and there is no headroom left for arithmetic on it. `octree::children` computes
child indices as `2·i + bit`, which is safe for a valid depth-63 parent (the
result lands in `[-2^63, 2^63-1]`) but would wrap if ever applied at depth 64.
The orchestrator therefore caps splitting at `SPLIT_MAX_DEPTH = 60` — four levels
of headroom below the addressing limit, and a depth-60 node is
`size_m(60) = 16 384 m` across, already far below any useful shard region.
`validate` rejects any region it produced deeper than that. This
is a defensive bound on new code, not a claim about an existing defect: the
current `validate_node` is correct at every depth, and
`crates/space-protocol/tests/acceptance.rs:1219` already covers `depth >
MAX_DEPTH`. `P3.2` adds one property test that `children()` composed `n` times
from the root always yields nodes that `validate_node` accepts, for `n ≤ 60`.

### 3.3 Epoch publication (`epoch.rs`, pure state machine)

```
Idle --propose(changes)--> Preparing{epoch, activation_tick, awaiting: Set<ShardId>}
Preparing --all acked-----> Committed{epoch}   (published map := new map)
Preparing --deadline------> Aborted           (published map unchanged)
```

* `EPOCH_LEAD_TICKS = 60` (2 s at 30 Hz): `activation_tick` is computed from
  the **maximum** tick reported across the shards that must switch, because the
  two shards run independent loops and drift (the exact reason Phase 2 split
  `activation_tick` from `local_activation_tick`).
* Recipients: every shard named in `changes` **plus** every shard whose region
  is adjacent to a changed node. A shard that is not told the map moved will
  keep offering handoffs under the old epoch and eat `StaleEpoch` answers
  forever; the neighbour set is cheap to compute from the octree.
* Deadline: `EPOCH_ACK_TIMEOUT_TICKS = 90` (3 s). On expiry, `EpochAbort` to
  everyone who acked and **back off**: the same pair is not re-proposed for
  `SCALE_COOLDOWN_TICKS`.
* One outstanding prepare at a time (H10 + H12 + H14 all depend on it).
* The orchestrator also **pushes** the committed map to every shard (not just
  the changed ones) so that `Shard::shard_map()` — and therefore every future
  `Welcome` and `Redirect` — carries the new epoch (F8, F9). A shard that
  missed the push learns the new epoch from the next `Redirect`/`StaleEpoch`
  it exchanges; the directory (`DirectoryQuery`) is the pull fallback.

### 3.4 The launcher seam (`launcher/mod.rs`)

```rust
#[async_trait] // no: hand-roll `-> impl Future` in the trait objects the
               // orchestrator already boxes; avoid a new proc-dependency.
pub trait Launcher: Send + Sync {
    fn launch(&self, spec: LaunchSpec) -> BoxFuture<'_, Result<Handle, LaunchError>>;
    fn poll(&self, handle: &Handle) -> BoxFuture<'_, ShardLiveness>;
    fn terminate(&self, handle: &Handle) -> BoxFuture<'_, Result<(), LaunchError>>;
    fn reap_orphans(&self, expected: &[Handle]) -> BoxFuture<'_, Vec<String>>;
}
```

`LaunchSpec { cluster, shard, region, bind_hint, orchestrator_addr, peer_token,
port_slot }`. `Handle { shard, backend_id, endpoint }`.

Three impls, in increasing fidelity:

1. **`NullLauncher`** (test-only): records the specs it was asked for and
   reports the shard running at a caller-supplied endpoint. Used by the pure
   `topology`/`epoch`/`autoscale` tests, which must not need a socket at all.
2. **`InProcessLauncher`** (the workhorse for H9–H15): spawns a real
   `space_server::shard::Shard` + `ShardNet` on `127.0.0.1:0` inside the test's
   own tokio runtime — literally the `RunningShard` harness from
   `crates/space-server/tests/acceptance.rs`, lifted into `space-server`'s
   public API as `space_server::testutil::spawn_shard(config) -> RunningShard`
   so the orchestrator's tests can use it without duplicating it. Real QUIC,
   real 30 Hz loop, real handoffs; `terminate` aborts the tasks and closes the
   endpoint. **This is what makes the whole split/merge/rebalance loop
   verifiable in-sandbox.**
3. **`DockerLauncher`** (feature `docker`, default-on for the binary): bollard,
   ported from `spikes/orch-docker/src/main.rs` — `shard_labels`,
   `shard_create_body` (image pinned by digest, UDP port published from
   `HOST_PORT_RANGE`, `restart_policy = no`), `pull_image`, `health_check`
   (running **and** published port allocated), `list_shards` by label,
   `remove_container` (force, `-v`), `reap_orphans`. The spike's 13 pure-helper
   tests move with it so the coverage is not lost. `connect_with_defaults()`
   failure is a clean `LaunchError::NoDaemon`, never a panic.

The orchestrator binary picks the impl from config
(`--launcher local|docker`), defaulting to `docker` in the compose file and
`local` in tests.

### 3.5 The control plane (`control.rs`)

The orchestrator binds one QUIC endpoint (same `space_net::crypto` self-signed
identity, `SERVER_NAME = "space.local"`) and accepts two kinds of connection,
distinguished by hello token exactly the way `net.rs` distinguishes clients
from peers:

* `shard:<id>` → a shard. It sends `LoadReport` every
  `LOAD_REPORT_INTERVAL_TICKS = 30` (1 s) and receives epoch messages.
* `orchestrator-admin` → an operator/MCP client. `AdminCommand` in,
  `AdminReply` out.
* anyone else → `DirectoryQuery` answered with `DirectoryReply` (the pull path
  for clients that need a fresh map).

Shard-side, `space_server::net` gains `OrchestratorLink`: one persistent
connection + FIFO sender, reusing `run_peer_sender`'s shape (order matters for
epoch messages for the same reason it matters for handoff messages).

`AdminCommand::SpawnBots` is implemented as a no-op that answers
`Refused("bots arrive with the Phase-7 load harness")` rather than silently
doing nothing — an honest refusal beats a fake `Accepted`. `KillShard` and
`ForceSplit`/`ForceMerge` are implemented (they are the cheapest way to drive
H15 and H11 in a test).

### 3.6 The orchestrator loop (`orchestrator.rs`)

A single `tokio::select!` loop, one task, no locks on the hot path:

```
loop { select! {
    report = load_reports()      => autoscaler.observe(report, tick),
    action = autoscaler.ready()  => epoch.propose(topology::split_plan(...)),
    ack    = control.acks()      => epoch.ack(shard),
    _      = epoch.deadline()    => epoch.timeout(),
    _      = clock.tick()        => { launcher.poll(); control.push_map(); }
    cmd    = control.admin()     => handle_admin(cmd),
}}
```

The loop's clock is the orchestrator's own 30 Hz ticker; every decision is
expressed in ticks, never in `Instant`, so a test can drive it by calling
`step(n)` instead of sleeping (H14's determinism depends on this).

---

## 4. Task sequence

Five coders can work in parallel on the marked tracks; the dependency column is
what blocks. Each task ends with `cargo xtask ci` green (F15) — nothing is
"done" on a red workspace.

| # | task | files | depends on | who |
|---|------|-------|-----------|-----|
| **P3.0** | **Docker E2E entry criterion.** Add a `docker-e2e` CI job (a runner with a daemon) that runs the spike binary and fails on any non-`ok` step; implement `xtask e2e` (currently `stub()`, F15) to run the same check. Until this is green, P3.8 (the `DockerLauncher`) is *written but unproven*. | `.github/workflows/ci.yml`, `xtask/src/main.rs`, `spikes/orch-docker/README.md` | — | coder-A |
| **P3.1** | Protocol additions: `ShardHello`/`ShardWelcome` appended (variants 23, 24), `VARIANT_COUNT` 23→25, `PROTOCOL_VERSION` 2→3, round-trip + tag-density tests extended. | `space-protocol/src/{messages.rs,lib.rs,codec.rs}`, `space-protocol/tests/acceptance.rs` | — | coder-A |
| **P3.2** | `PersistentShardIdAllocator` + `ids.rs` (persist-before-handout, atomic write, restart-resume test). | `space-orchestrator/src/ids.rs` | crate skeleton | coder-B |
| **P3.3** | `topology.rs`: split/merge plans + `validate`, all geometry tests. Pure, no tokio. | `space-orchestrator/src/topology.rs` | crate skeleton | coder-B |
| **P3.4** | `autoscale.rs`: metric, dwell timers, cooldown, one-action-at-a-time; property test that the same report sequence yields the same decisions. Pure. | `space-orchestrator/src/autoscale.rs` | crate skeleton | coder-C |
| **P3.5** | `epoch.rs`: prepare/ack/commit/abort state machine + recipient selection + deadline. Pure. | `space-orchestrator/src/epoch.rs` | P3.3 | coder-C |
| **P3.6** | `space-server` shard-side: tick-time ring + `tick_p95_ms()`, `draining` flag wired to `RejectReason::Draining`, `apply_epoch`, `OrchestratorLink`, `--orchestrator/--claim-region` flags in `main.rs`, `testutil::spawn_shard` extracted from the acceptance harness. | `space-server/src/{shard.rs,net.rs,main.rs,lib.rs,testutil.rs}`, `space-server/tests/acceptance.rs` | P3.1 | coder-D |
| **P3.7** | `launcher/mod.rs` + `launcher/local.rs` (`NullLauncher`, `InProcessLauncher`). | `space-orchestrator/src/launcher/*` | P3.6 | coder-E |
| **P3.8** | `launcher/docker.rs`: port the spike, feature-gated on `bollard` with `default-features = false, features = ["http"]` (F17); the 13 spike tests move here. | `space-orchestrator/src/launcher/docker.rs`, `crates/space-orchestrator/Cargo.toml` | P3.7, P3.0 | coder-E |
| **P3.9** | `control.rs`: the orchestrator's QUIC front end (shard reports in, epoch out, directory, admin). | `space-orchestrator/src/control.rs` | P3.1, P3.6 | coder-D |
| **P3.10** | `orchestrator.rs` + `main.rs` + `config.rs`: the loop, startup reconciliation, arg parsing. | `space-orchestrator/src/{orchestrator.rs,main.rs,config.rs,lib.rs}` | P3.2–P3.5, P3.7, P3.9 | coder-B (integrator) |
| **P3.11** | H9–H15 acceptance suite over loopback QUIC (§5). | `space-orchestrator/tests/acceptance.rs` | P3.10 | coder-C + coder-D |
| **P3.12** | Docker layer: `Dockerfile`s for shard + orchestrator, `deploy/docker-compose.yaml` extending the spike template (cluster network for shards, parameterised subnet + `HOST_PORT_RANGE`, uid reconciled with Phase 8), `docs/plan.md` Phase-3 line updated with the H9–H15 pointer. | `deploy/*`, `Dockerfile.shard`, `Dockerfile.orchestrator`, `docs/plan.md` | P3.8 | coder-A |
| **P3.13** | Peer authentication: orchestrator-issued `peer_token` (issued only; enforcement deferred, see `docs/security-notes.md`); update the F10/F11 doc comments to state what is and is not closed. | `space-server/src/{shard.rs,net.rs}`, `space-orchestrator/src/control.rs` | P3.9, P3.10 | coder-D |

Critical path: **P3.1 → P3.6 → P3.7 → P3.10 → P3.11**. P3.0 gates only the
Docker claim, not the logic.

---

## 5. Acceptance tests

### 5.1 In-sandbox (loopback QUIC, `InProcessLauncher`, runs in `cargo xtask ci`)

Pure/unit (no sockets):

| id | test | proves |
|----|------|--------|
| u1 | `ids_never_reuse_across_splits_and_merges` | H9 |
| u2 | `a_restart_resumes_strictly_above_the_last_allocated_id` | H9 |
| u3 | `the_id_space_limit_panics_rather_than_wrapping` (already in `alloc.rs`; re-asserted through the persistent wrapper) | H9 |
| u4 | `a_split_map_is_valid_and_changes_ownership_only_inside_the_child` | H11 |
| u5 | `a_merge_map_is_the_exact_inverse_of_the_split` | H11 |
| u6 | `no_two_regions_at_equal_depth_overlap_in_any_produced_map` | H11 |
| u7 | `an_unacked_prepare_aborts_and_the_published_map_is_byte_identical` | H10 |
| u8 | `every_shard_that_must_switch_acked_before_the_commit` | H10 |
| u9 | `the_neighbour_of_a_changed_region_is_always_a_prepare_recipient` | H10 |
| u10 | `the_same_report_sequence_yields_the_same_decisions` (property, 1000 random report streams) | H14 |
| u11 | `split_needs_0_8_for_300_ticks_and_merge_needs_0_35_for_1800` | H14 |
| u12 | `the_hysteresis_band_clears_both_timers` and `a_shard_in_cooldown_is_not_rescaled` | H14 |
| u13 | `load_is_the_max_of_the_two_terms` (players/100 vs p95/33.3, both directions) | H14 |
| u14 | `one_scale_action_is_in_flight_at_a_time` | H12, H14 |

Integration (real shards on `127.0.0.1:0`, real quinn, real 30 Hz loops):

| id | test | proves |
|----|------|--------|
| t1 | `the_orchestrator_allocates_ids_and_no_launch_spec_carries_one` | H9 |
| t2 | `a_split_gets_a_strictly_greater_id_than_its_parent` | H9 |
| t3 | `after_an_orchestrator_restart_the_next_id_exceeds_every_id_ever_issued` | H9 |
| t4 | `a_split_publishes_prepare_then_acks_then_commit_in_that_order` | H10 |
| t5 | `a_shard_that_never_acks_gets_an_abort_and_the_world_keeps_the_old_map` | H10 |
| t6 | `a_split_rebalances_players_with_zero_client_visible_discontinuity` — 8 players on one shard, split, assert every player ends on the shard that owns its position, guid unchanged, no tick gap, no `Reject` | H11, H13 |
| t7 | `a_merge_rebalances_players_back_onto_one_shard` — the inverse of t6 | H11, H13 |
| t8 | `after_any_topology_change_lookup_agrees_with_the_simulating_shard_for_every_player_position` | H11 |
| t9 | `an_epoch_bump_mid_handoff_aborts_with_epoch_changed_and_the_guid_is_reofferable` (Phase 2's test, re-run against an orchestrator-generated epoch instead of a hand-written map) | H12 |
| t10 | `a_transitioning_client_receives_a_frame_every_tick_with_no_gap` | H13 |
| t11 | `an_observer_of_a_moving_ship_follows_it_across_a_split_without_losing_it` | H13 |
| t12 | `the_autoscaler_splits_a_hot_shard_end_to_end_over_loopback` — drive `players` up, assert the split happens and the epoch moves | H14, H13 |
| t13 | `the_autoscaler_merges_a_cold_pair_end_to_end` | H14, H13 |
| t14 | `a_flapping_load_does_not_flap_the_topology` | H14 |
| t15 | `a_killed_shards_region_is_reassigned_and_its_id_is_never_reused` (`AdminCommand::KillShard` + `launcher.poll`) | H15 |
| t16 | `startup_reconciliation_reaps_orphans_and_adopts_survivors` | H15 |
| t17 | `a_shard_without_the_orchestrator_token_is_not_treated_as_a_peer` | §3.6/P3.13 |
| t18 | `directory_query_returns_the_current_epoch_map` | H10 |

Determinism rules for all of the above (the Phase-2 lesson: no wall-clock
flakiness): the orchestrator is driven by `step(n)` in tests, dwell timers are
counted in ticks, and any test that must wait on a real socket uses
`wait_for`-style polling with a bounded retry count (the `wait_for_clients`
pattern already in `acceptance.rs`), never a bare `sleep` as the assertion.

### 5.2 Deferred to the Docker-daemon entry criterion (P3.0)

Cannot run here (no daemon, F16/spike README) and must not be claimed as
verified until a daemon job runs them:

* `docker_launcher_create_health_list_remove_reap` — the spike's five-step
  cycle against a real daemon, asserting `all_steps_ok: true`.
* `docker_launcher_launches_a_real_shard_container_and_the_shard_connects_back`
  — the shard image boots, binds its UDP port, registers with the orchestrator.
* `docker_split_starts_a_second_container_and_a_merge_removes_one` — the full
  H9–H15 loop with `DockerLauncher` instead of `InProcessLauncher`.
* `compose_up_brings_up_orchestrator_plus_one_shard` — the `deploy/` template.
* Orphan reaping after `docker kill` of a shard container (H15 on real
  infrastructure).

The CI job runs the same test names with `#[cfg(feature = "docker-e2e")]`
gating (or `#[ignore]` + `nextest --run-ignored only` in the daemon job), so
the deferred tests are *written now* and only their execution is deferred.

---

## 6. Risks and mitigations

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | **No Docker daemon in the sandbox.** The container path can be written and unit-tested but never executed here. | High — an unproven deliverable | The spike already proves bollard 0.21.1 works against a real daemon (`spikes/orch-docker/README.md`, 100/100). Everything except `docker.rs`'s syscall-facing lines is covered by the `Launcher` seam. `P3.0` is an explicit, blocking exit criterion, not an afterthought. |
| R2 | **`hyper-named-pipe` is not in the offline cargo cache**, so bollard's default feature set does not resolve here. | Medium — would block the build | Pin `bollard = { version = "0.21", default-features = false, features = ["http"] }` (verified to resolve offline; the spike's lockfile already carries `bollard-stubs 1.53.1-rc.29.3.1`). Linux-only, which matches the orchestrator's target. **Do not** enable `unix-boot-time` or any feature that pulls the named-pipe backend. |
| R3 | **The orchestrator takes over `ShardId` allocation from `main.rs` argv.** Today `main.rs` reads `--shard <id>` and builds `ShardMap::new(1, …)` from a static TOML; `ShardIdAllocator` exists but is referenced by nothing outside its own unit tests (Phase-2 review, MAJOR 3). | High — this is the defect Phase 3 exists to close | `P3.1` deletes the argv path: `--shard`/`--map` become *bootstrap-only* (a shard may still start standalone for tests), and any id arriving from the orchestrator is checked against the orchestrator's own `used` set before it is handed back. `T1` is the first task precisely so nothing else is built on the old path. The acceptance test for **H9** is the regression test for MAJOR 3. |
| R4 | **Epoch bump invalidates in-flight handoffs** (`RejectReason::EpochChanged`), so a split racing a player transfer drops the transfer. | Medium — a player could be dropped mid-jump | The drop is *safe* (the player stays on the source shard and retries), but it must be *bounded*. `P3.4` adds the `EpochPrepare`/`EpochAck` quiesce step: no shard begins a handoff while an epoch is preparing, and the commit waits for outstanding handoffs to drain (bounded by `HANDOFF_LEAD`). **H13** measures the bound. |
| R5 | **Split geometry is wrong if the world is not uniformly populated.** Carving all eight children of a hot region creates seven empty shards and leaves the eighth still hot — the autoscale does nothing except churn. | Medium — autoscale achieves nothing | Split **one octant at a time**, the octant holding the most players (`split_plan`, using the last `LoadReport`'s player positions), and only for regions of at least 1024 km edge (depth ≤ 54, §3.2). `P3.3`'s property test asserts the eight children tile the parent exactly and that `lookup` flips ownership only inside the carved child. |
| R6 | **Flapping.** A shard at exactly the scale-out threshold creates a child, drops below, and merges it back. | Medium — churn, wasted handoffs | Three independent dampers, all in `autoscale.rs` and all in *ticks* so they are testable: (a) the hysteresis band — split only at `load ≥ 0.8` sustained `SPLIT_DWELL_TICKS = 300` (10 s), merge only at `load ≤ 0.35` sustained `MERGE_DWELL_TICKS = 1800` (60 s); (b) a per-shard `SCALE_COOLDOWN_TICKS = 900` (30 s) after any scale action involving it; (c) a global `MIN_EPOCH_INTERVAL_TICKS = 90` (3 s) between epoch commits, and one scale action in flight at a time. The thresholds are the ones already written into `space_protocol::control::LoadReport`'s doc (F6) — do not invent new numbers. |
| R7 | **`tick_p95` is not currently measured.** `Shard` records no timing at all. | Medium — the load metric is half-missing | `P3.5` adds a 512-sample ring buffer of tick durations to `Shard`, exported in `LoadReport`. It is one `Instant::now()` pair per tick — measurable overhead, and **H15**'s benchmark re-measures it. |
| R8 | **New protocol variants break wire compatibility** with anything built before Phase 3. | Low in-tree, high for released builds | `PROTOCOL_VERSION` 2 → 3 and `VARIANT_COUNT` 23 → 25. `quinn.rs` already rejects a mismatched ALPN/version at the connection level, so the failure is a clean connect error, not a misparse. Note it in the changelog; there is no external consumer yet. |
| R9 | **Two new first-party crates must pass the no-C-crate gate.** | Low | `space-orchestrator` and `space-control` depend only on workspace crates + `clap`/`tracing`/`serde`. `bollard` is pure Rust (hyper + tokio + serde); its only native dependency is the OS socket, which is not a C crate. `xtask lint-deps` needs no exception entry. |
| R10 | **The orchestrator becomes a single point of failure.** | Medium | Accepted for Phase 3: the orchestrator is control-plane only. If it dies, running shards keep simulating and players keep playing — no epoch can commit, so no split happens. `P3.10` restarts it under compose (`restart: unless-stopped`), and a restart re-reads the live shard set before proposing anything. Cross-host scheduling and orchestrator HA are explicitly **out of scope**. |
| R11 | **`SpawnBots` is requested by the brief but there is no bot population to spawn.** | Low | Do not fake it. `AdminCommand::SpawnBots` is accepted by the protocol but the shard answers `AdminResult::Refused("bots arrive with the Phase-7 load harness")`. The autoscaler is tested with real connected clients and with injected `LoadReport`s instead. |
| R12 | **Docker networking assumptions.** The shard must be reachable from the host *and* from peer shards. | Medium | Compose publishes `7870-7899/udp` for host clients; inter-shard traffic rides the compose network over the same port. The shard advertises **two** addresses in `ShardHello` (internal for peers, published for clients) — `P3.6`. The spike's `orchestrator.rs` already handles the published-port mapping and is the reference. |

---

## 7. Out of scope for Phase 3

Stated so nobody starts it by accident: cross-host scheduling and orchestrator
HA (R10); full shard-to-shard authentication (D3); new physics beyond Phase 1;
client-side prediction; world persistence/snapshots; metrics export beyond the
`LoadReport` ring buffer; any Windows story for the orchestrator beyond
"it compiles and its pure modules unit-test" (Docker Desktop is out of the
sandbox anyway — CI's Windows job builds and unit-tests the crate, and that is
all it can do).

## 8. Definition of done

Phase 3 is done when:

1. Every **H9–H15** test in §5.1 exists, is named for its guarantee, and passes
   under `cargo test --workspace --release`.
2. `cargo lint`, `cargo clippy --workspace --all-targets -- -D warnings`,
   `cargo doc`, `cargo lint-deps` and `cargo e2e` (loopback) are all green —
   i.e. the existing CI job passes with the new crates added.
3. `ShardIdAllocator` is wired into production and `main.rs` no longer accepts a
   shard id from argv except as an explicit standalone-bootstrap escape hatch
   (closes Phase-2 MAJOR 3).
4. `deploy/Dockerfile` builds an image that runs a shard, and
   `deploy/docker-compose.yml` brings up orchestrator + N shards on a user-defined
   network — **executed and green on a Docker-equipped runner** (`P3.0`, twice in a row).
5. §6 lists no open risk without a mitigation, and this document is updated to
   record anything that changed during implementation.

## Deferred: bollard DockerLauncher

**Status: deferred, by lead authorization.** The `bollard`-backed `DockerLauncher`
(P3.8) is **not** implemented. `bollard` and its entire hyper/http/futures
dependency tree are absent from the offline cargo cache
(`cargo metadata --offline` → "no matching package named `bollard`"; F17's
`--no-default-features --features http` workaround resolved only inside the
spike's own lockfile) and the sandbox has no network, so it cannot be fetched.
No dependency was added; the default build stays offline-clean.

What landed instead (P3.12, deployment layer + container-path wiring):

* `launcher/docker.rs` — an honest stub. `DockerLauncher` implements `Launcher`;
  `spawn`/`kill`/`poll` return `LaunchError::Unavailable(..)`, and
  `space-orchestrator --launcher docker` fails fast at startup with the same
  message (an error, not a panic).
* `deploy/Dockerfile.shard`, `deploy/Dockerfile.orchestrator`,
  `deploy/docker-compose.yaml` — written, **not executed** (no Docker daemon).
* `space-server --bind <addr:port>` (default `127.0.0.1:0` orchestrated) and
  `--advertise <addr:port>` (reported in `ShardHello`; defaults to the bound
  address); `space-orchestrator --control-addr <addr:port>` (what shards dial;
  defaults to `--bind`).

To finish: with network access add
`bollard = { default-features = false, features = ["http"] }`, port
`spikes/orch-docker/src/main.rs` into `launcher/docker.rs` (spawn runs a
container from `space/shard` with `--orchestrator <control_addr> --advertise
<published addr>`), and run the P3.0 Docker end-to-end entry criterion.
