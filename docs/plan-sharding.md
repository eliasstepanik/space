# Seamless sharding

The plan for: one address for players, four shard areas on the demo area, shards the
orchestrator starts and stops by itself, seamless crossings, and objects of other shards that
stay visible. `AGENT.md` and the two skills stay binding; where this plan changes a rule they
state (a guarantee's wording, a port, a flag), the same change updates them.

## Decisions (owner, 2026-11-23)

1. Players type **one address, the orchestrator's**. It **redirects** them to the shard that owns
   their position; game traffic then flows client <-> shard directly. No proxy.
2. Objects another shard simulates stay **visible and visibly moving**, with the update rate
   dropping by distance, and **interaction across borders stays exact** (push, collide, rest).
   Shards do not simulate far things coarser.
3. The demo area is cut into **four shard areas** everywhere (dev scripts, tests, production)
   through a `minimum shards = 4` setting.
4. Other shards' objects are visible as far as the own shard's: 10 km. A border makes no
   difference to what a player sees.
5. The orchestrator creates, starts, stops and removes shard containers itself through the Docker
   socket, mounted directly. The compose file contains only the orchestrator.
6. A joining player always starts at the scene's spawn point.
7. At most 8 shards (a setting).
8. Built straight through on a feature branch. Nothing is pushed to `main` until the owner says so.

**Hold:** another session is changing the demo area. Nothing in `crates/space-scene/src/demo.rs`,
`crates/space-client/src/demo_area.rs` or `scripts/harness/physics.lua` is touched until the owner
lifts the hold (phase 7).

## What the code does today

Checked in code unless marked otherwise. Written before any of this was built: section 1 is
implemented now, so what is said here about `split_plan`, `validate`, `merge_target`,
`recipients`, `scene_anchor` and `free_spawn_point` describes the code *before* it (see the status
note under section 1 for what they do now). Everything else still stands.

- The map is `ShardMap { epoch, regions }`; a region is an octree node `depth:x:y:z` (depth 64 =
  one 1024 m cell) with an owner and one endpoint. Lookup = the deepest region holding the position.
- The orchestrator's root is depth 40. A split carves one **direct child octant**
  (`topology::split_plan`), only at depth <= 54, and `topology::validate` refuses any region deeper
  than 60. A merge needs the **direct parent** node to be a region (`merge_target`).
  `EpochPublisher::recipients` reaches the new owner, the direct parent's owner and sibling /
  grand-sibling owners only; everyone else learns of a change from a droppable `DirectoryReply`.
- A shard hello without a region means "the root"; a second one is refused and the shard exits.
  A split child is launched first with the id and node the orchestrator chose, and must say hello
  and ack within the 3 s epoch deadline or the split aborts and the child is killed.
- Launchers: `local` runs the shard inside the orchestrator process on `127.0.0.1:0`; `docker` is a
  stub. `Launcher::poll` is never called; nothing launches the root's shard; a shard that loses the
  orchestrator keeps running and never re-dials.
- No shard sends `LoadReport`. Reports from a shard that owns no region are dropped. A merged child
  is killed when the merge commits; merges are refused unless it has no players, because nothing
  hands its entities over.
- There is **no cluster tick**: every shard counts from its own start with a skipping interval.
  Epoch `activation_tick`s are stored and never read; a commit takes effect on arrival.
- Handoff: `HandoffCommit.activation_tick` is the *offered* tick, the state is read at the actual
  commit tick (later when the client connects late), and the receiver sets `input_ack_floor` to the
  offered tick. The receiver installs at its own accept tick + lead.
- Client: dials a shard (default port 7777); `connect` accepts only `Welcome` or `Reject`. At a
  handoff switch `Replica::switch_to` zeroes the ticks and swaps the entity view, `Clock` snaps and
  `OwnShip` is rebuilt (its key holds session and handoff count), inputs in flight are dropped. The
  own ship is found by session id, which is per shard.
- Across shards: `GhostUpdate` carries bodies within 64 m + 3 s of motion of the neighbour's
  territory; a ghost is a kinematic rapier body. "Neighbours" = every other shard in the map. Peers
  are trusted on map membership alone; `peer_token` is issued and never checked.
- Replication: everything replicated is an ECS entity in `SimWorld`; interest 10 km; one frame per
  tick; one `change_base` tick per client; no level of detail. The client restarts an entity's
  history after a 15-tick gap, draws 2 ticks behind and extrapolates 3 at most.
- `scene_anchor(map)` = centre of the shallowest region: for depth < 64 the corner where its eight
  children meet, so borders through the scene lie on x = 0, y = 0 (eye height) and z = 0. A shard
  that does not own the spawn point still spawns a joiner, at its own region's centre.
  `free_spawn_point` hands out a grid reaching 15 m from the spawn point.

## 1. The map

- Regions stay octree nodes; the smallest shard area stays one cell (1024 m). A crowd inside one
  cell cannot be split further.
- **Carve**: a region may be carved out of any ancestor region, not only as a direct child octant.
  `topology` gets `carve_plan(map, node, new_owner)` (the node must lie inside an existing region
  and not be a region already) and the merge removes a carved region and falls back to the
  **nearest ancestor** region's owner. The octant split used by the web UI becomes the special case
  "carve this child octant". `validate` allows depth 64 (`children()` already returns `None` there).
- **Recipients** of an epoch change: every shard in the map, and every owner the change names (the
  new shard of a carve is not in the map yet). With at most 8 shards that is cheap, and it removes
  the reliance on the droppable map push.
- **Anchor rule** (`space-scene::scene_anchor`): a depth-64 shallowest region keeps its cell centre.
  Any other gets the node's centre corner **plus (6 m, 512 m, 4 m)**: half a cell up, so no
  horizontal border lies within 512 m of the scene, with the vertical borders at scene x = -6 and
  z = -4. With `k` the corner's cell index, the four cells that meet there are `[k,k,k]` (holds the
  spawn point), `[k,k,k-1]`, `[k-1,k,k]`, `[k-1,k,k-1]`.
- `free_spawn_point` only hands out points inside the territory of the shard that owns the spawn
  point, so a busy spawn does not put a new ship across a border.
- Tests and examples that pin the old rule move with it: `space-scene/tests/scene.rs`, the fixtures
  with a depth-63 root in `space-client/tests/handoff.rs` and `space-server/tests/physics.rs`, and
  the skill's `63:0:0:0` + `64:1:0:1` recipe (that cell now lies 512 m below the scene).

**Status: built (phase 1, first half).** What it became, and where it differs from the bullets
above:

- `topology::carve_plan(map, node, new_owner) -> Carve { node, new_shard, from, map, change }`,
  `enclosing_region(map, node)` (the nearest ancestor that is a region), `is_inside`,
  `octant_node`; `split_plan` = `carve_plan` of a child octant. The octant split **keeps its size
  floor** (`can_split`: edge >= 1024 km, depth <= 54) — it is what the autoscaler and the web UI
  use, and halving by load below that relieves nothing; `carve_plan` has no floor.
  `SPLIT_MAX_DEPTH` (60) is gone; `MAX_REGION_DEPTH` = 64 is the octree's own limit.
- `Orchestrator::begin_carve(node) -> Result<CarveStarted { node, shard, from, epoch }, String>`
  runs a carve as one epoch with **today's launcher flow** (the new shard is launched, says hello
  with the id and node allocated for it, is handed the prepare, acks). `begin_split` picks the
  octant and calls the same code. Section 2 replaces the launch by "already connected" (done: it
  returns `CarveWanted { node, from }` now, see the status block there).
- A merge works for a carved cell end to end (autoscaler, operator, web UI); the snapshot's
  `RegionView::parent` is now the enclosing region (any number of levels up), `octant` only when
  that is the direct parent, and the page says "carved out of ..., N levels up".
- **Every** shard of the map must now ack a change within the 3 s deadline. A shard that cannot —
  dead, or the reserved root owner nobody has claimed — holds up every map change until section 2's
  supervision replaces it. Before, only the shards next to a change were asked. Left for section 2:
  the operator's pre-check (`split_refusal`) still only asks whether the shard being split holds a
  control link, so with a third shard disconnected a split is accepted, launched and then rolled
  back at the deadline (the event log names who did not ack) instead of being refused up front.
- A merge still needs the shard's own fresh `LoadReport` saying it has no players, and no shard
  sends one yet: the tests that merge a carved cell say so on the shard's behalf.
- The spawn point is a function of the **map** (`scene_anchor`), not of the installed scene: a
  shard nobody gave a scene still puts its players on the point its clients measure from.
- A shard that does **not** own the spawn point still puts a joiner on its own region's centre
  (section 4 turns that into a `JoinRedirect`), and that fallback is *not* restricted to its own
  ground: under a root of several cells the centre is the corner where the cells meet, which
  belongs to cell `[k,k,k]` — if another shard owns that cell, the joiner is that shard's from its
  first tick and is handed over at once. The handoff tests' fixture (`63:0:0:0` + `64:1:1:1`)
  rests on exactly this.
- Fixtures: the `63:0:0:0` + `64:1:1:1` map stays (shard 2 now holds the spawn point's cell: scene
  x > -6, z > -4); `64:1:0:1` (the runner) became that same map with the run started 3 m behind the
  x = -6 border; `64:0:1:1` (the zero-g corner) stays and is now x < -6, z > -4. The skill's recipe
  is `63:0:0:0` + `64:1:1:0` (the bowl end of the gravity room, z < -4).
- What can no longer be staged on the showcase: anything across a **horizontal** border (a crate
  falling through one, pendulums above one swinging over crates below it). That is the point of the
  rule; the mechanisms keep their hand-built tests in `space-server/tests/physics.rs`.
- Noted while re-deriving, for section 7. From `demo.rs`: the floating crate at (-13.2, 0.6, -4.0)
  lies exactly on the z = -4 border. From reading `border::territory_distance` (not from a run): it
  measures to the faces of the deepest region around a point, so beside a *second* carved cell it
  under-reports the distance to a shard whose ground is further away (documented there as "never
  more than the true distance"), and the 2 m hysteresis can then hold a body on the wrong one of
  three shards until the settle rule moves it. From a test run: a player standing half a metre from
  the border in the gravity room drew every body of the room onto its shard for a while, through
  chained islands (rolling balls reach far with their velocity lead).

## 2. Shard lifecycle

One path for every shard, launched or started by hand: **start without a region, get one at hello.**

- A region-less hello gets the root if nobody owns it. Otherwise the orchestrator carves the next
  free **layout cell** for it - in the fixed order `[k,k,k-1]`, `[k-1,k,k]`, `[k-1,k,k-1]`, then the
  four cells of the layer below - allocates its id and runs the carve as an epoch change with the
  shard already connected, so the ack deadline never has to cover a process or container start.
  One change at a time: further hellos wait their turn (the shard retries with back-off; a refused
  hello no longer makes it exit unless the refusal is final, e.g. a wrong cluster or `max-shards`).
- `--min-shards N` (default 0 = today: nothing is launched) and `--max-shards` (default 8). A
  **reconciler** in the run loop launches a shard whenever fewer than `min-shards` are live or
  starting. Root + three layout cells = the four quadrants of the demo area.
- Launchers: `local` (unchanged, for tests), **`process`** (starts the `space-server` binary that
  lies next to the orchestrator's own executable; ports from the host port range), `docker` (8).
- **Supervision**: the run loop polls what it launched. A dead shard's regions go to a replacement
  with a **new id** (ids are never reused; guid serials restart with a process, so reusing the id
  would mint guids twice) through an epoch change, and the orchestrator then raises the **scene
  revision**, so the scene's bodies are spawned afresh instead of twice. Players of the dead shard
  reconnect through the orchestrator. The H15 text is updated to say so.
- A shard that was started by a launcher exits when it has lost the orchestrator for more than a
  few seconds, so nothing is orphaned by an orchestrator restart.
- **Secrets**: the orchestrator creates a cluster secret at start unless given one. A shard hello
  must carry it when the control port is not on loopback, and every shard-to-shard hello carries
  the `peer_token` and is refused without it - shard ports become public in section 4, so map
  membership alone is no longer enough.

**Status: built (phase 1, second half), except the `docker` launcher (section 8).** What it
became, and where it differs from the bullets above:

- **Hello** (`Orchestrator::handle_shard_hello -> Result<ShardWelcome, HelloRefusal>`): a hello
  without id and region is given, in this order, the unclaimed root; the regions of a **dead**
  shard (a replacement goes before everything else); the node a pending request waits for
  (`wanted`: an operator's or the autoscaler's split or carve); the next free layout cell
  (`topology::layout_cells`, `next_layout_cell`). The id is allocated there and the carve or the
  replacement is proposed with the shard's control link already registered. The welcome names the
  region, but the shard owns it only once the map says so (`ShardWelcome` docs).
- **Waiting** is `HelloRefusal::Wait`, sent as the existing `RejectReason::Draining` - no new wire
  variant. The shard (`space-server/src/orchestrated.rs`: `register`, `classify`) asks again after
  250 ms, doubling to 2 s, for as long as it takes - **on a new connection each time**: the
  orchestrator closes a control connection after every refusal, final or not (see "The control
  connection" below). An orchestrator it cannot reach at all is given up on after 10 s;
  `Unauthorized`, `Full` (`max-shards`), `Other` (wrong cluster, a claim that cannot be served, a
  launched shard that is no longer needed) and a protocol mismatch are final. Hello-driven changes
  do **not** wait for `MIN_EPOCH_INTERVAL`: `--min-shards 4` settles in about a second. A "wait"
  that passes by itself (a change in flight, a shard that is not connected yet) is not logged; one
  that does not - the id counter cannot be written, a hello names a live shard's address from
  another host - is, once per reason.
- **The control connection** (`control.rs`, `Inbound`): the cluster secret gates the hello and the
  hello gates everything else. The connection's task reads nothing after the hello until the run
  loop's verdict is in; refused, the reason is delivered (`Conn::reject`: written, finished, given
  up to a second to be read) and the connection closed; welcomed, the connection is admitted as
  **that shard id** for its life, and a `LoadReport`, `EpochAck` or `SceneAck` is passed on only if
  it names it. The run loop takes the sender from the connection, not from the message.
- **A split or carve no longer proposes anything.** `begin_carve(node) -> Result<CarveWanted {
  node, from }, String>` records the node as wanted and the reconciler launches a shard for it;
  the epoch is proposed when that shard says hello. One wanted node at a time; it is given up
  after 60 s (`WANTED_TIMEOUT_TICKS`). The operator's answer reads "region X is wanted for a new
  shard" instead of "epoch N proposed".
- **Live recipients**: a change is refused up front while a shard of the map (or one it names) has
  no control link (`absent_recipient`), with the one exception of the dead shard a replacement
  takes over from (`EpochPublisher::propose_without`).
- **Claims**: a hello that names an id is only confirmed for a node that id owns in the map, or
  the reserved root. An issued id on a node outside the map, an id that is already registered and
  the old "same endpoint gets its id back" restart are all refused or replaced: a shard that
  comes back at its address is a new shard with a new id.
- **Coming back at one's address** (`take_over_in_place`): a hello that advertises the address of
  a live registered shard replaces that shard - but only (a) when it is going to be served: no
  change is in flight (it is told to wait, and nobody is made dead) and everybody who has to ack
  the replacement is connected; (b) when its control connection comes from the IP it advertises
  (from anywhere else it waits until the old shard's own link has closed); and (c) without
  killing a process: if the old shard was a launch, the launch *is* the process saying hello, and
  is matched to the new id. No launch is staged for regions the hello is taking itself.
- **Reconciler** (`reconcile_launches`): target = max(`min_shards`, live owners + dead shards to
  replace + the wanted node), never above `max_shards`; one launch at a time until the root is
  claimed; a failed launch is followed by a pause of 1 s, doubling to 30 s - the count starts
  over only when *a launch's shard comes to own a region*, not because other launched shards are
  healthy; a launch that has not registered after 30 s is stopped. A dead shard is replaced by a
  launch only if it was launched itself or `min_shards > 0` - a shard started by hand is replaced
  by the next one that says hello. **Surplus launches are stopped**: when there are more live and
  starting shards than the target (a shard started by hand took a dead shard's regions first, a
  wanted node was given up or taken), the launches that have not registered are stopped, newest
  first; one whose hello gets there first and finds nothing left is refused for good and stopped
  instead of being given a layout cell. (A shard started by hand that registers with nothing
  missing still gets the next layout cell.) The in-process launcher's shards cannot be told from
  hand-started ones by their hello, so its surplus is found by counting only, and never while a
  shard that registered within the last 5 s is still unmatched. `--min-shards` above what can be
  placed (8: the root and the seven layout cells), above `--max-shards` or above `--max-regions`
  is refused at startup; a merge that would take the cluster to or below `--min-shards` is
  refused.
- **Launchers** are keyed by `LaunchId`, not by shard id (`SpawnSpec { launch, cluster,
  control_addr, host_port, secret }`): the orchestrator learns which shard a launch became from
  the handle (`local`) or from the advertised endpoint (`process`: matched at the hello, by the
  host port the launch was handed). A launch is never matched to a shard that is dead or that
  registered before the launch was staged, and a launch whose shard is gone is removed when its
  process is. `process` (`launcher/process.rs`): `space-server` next to the orchestrator's
  executable or `--shard-exe` (made absolute once and started by exactly that path), one port of
  `--host-port-range` each, `shard-<port>.log` under `--shard-log-dir`, the secret through the
  child's environment, the orchestrator's own variables (`SPACE_ORCHESTRATOR_*`,
  `SPACE_HARNESS_TOKEN`) kept out of it. `launched-shards.pid` lists **exactly the processes
  that are running**: rewritten (temporary file, rename) at every start, stop and observed exit,
  removed when none is left. **The host port range must lie outside the OS's dynamic port range**
  (a shard bound to `127.0.0.1:p` takes the datagrams of a socket another shard got on
  `0.0.0.0:p`); the binary does not check that.
- **Supervision**: a closed control link of a registered shard, a launch whose process is gone
  and the admin `KillShard` all end in `declare_dead`. The replacement's epoch moves every region
  of the dead shard to the new id; when it commits the scene revision is raised once
  (`respawn_scene`; a raise that cannot be saved is tried again every second). Measured on
  real `process` stacks (release build, loopback): 0.3-1.1 s from the kill to a whole cluster on
  the new revision - up to half a second of it is the wait for the launcher's next poll - also
  when the range has exactly as many ports as shards and the replacement is handed the dead
  shard's own port, three times in a row. **A join whose epoch aborts is stopped**
  (link closed, process killed), not kept - whatever aborted it. A shard started by hand whose
  join was aborted therefore **exits and has to be started again**; one the orchestrator
  launched is launched again, after the pause a failed launch earns when the abort was at the
  deadline. (Keeping a healthy joiner when the abort was another shard's death, and giving it
  that shard's regions instead, is not built: what a shard owns is what the committed map says,
  but where it puts a player who joins it directly is still the centre of the region its
  *welcome* named - `shard_physics.rs` - so a joiner cannot simply be handed other regions.)
  **A shard that is connected but does not answer** is counted: three map changes in a row that
  run into their deadline without its ack make it dead, and it is replaced like any other. The
  outcome of an epoch is *queued* for a shard that is then stopped (an aborted joiner, a merged
  shard) before its link is closed in the same beat; whether it still reads it is not guaranteed,
  and it stops either way.
- **A shard without its orchestrator stops** (`RunningShard::wait` returns the reason, the binary
  exits non-zero) as soon as the control link is lost - every orchestrated shard, not only
  launched ones. After a hard kill of the orchestrator that is QUIC's idle timeout, about 11 s.
- **Secrets**: `--cluster-secret` / `--cluster-secret-file` / `SPACE_CLUSTER_SECRET[_FILE]`
  (>= 16 characters). A configured secret is required on every bind, loopback included. Off
  loopback with none configured the orchestrator starts only with `--min-shards >= 1`: it then
  generates one and hands it to what it launches; otherwise it refuses to start. `ShardHello`
  ends with `secret` (`PROTOCOL_VERSION` 7). When a secret is required it is checked before the
  cluster name: without it a hello is told `Unauthorized` and nothing about the cluster. The
  peer token is generated when none is given (`--peer-token[-file]`,
  `SPACE_ORCHESTRATOR_PEER_TOKEN[_FILE]`; a configured one under 16 characters is refused at
  startup - an empty one would switch peer authentication off); a shard that has one refuses a
  peer hello without it (`shard:<id>:<token>`, `net::peer_token_admits`, compared with
  `space_net::constant_time_eq`; counted, logged at most three times per ten seconds). Both
  secrets are held in `space_net::Secret` and `Hello` has a redacting `Debug`: no derived
  `Debug` prints either.
- **Not done here**: the `docker` launcher (signatures adapted only, section 8); players of a
  dead shard are **not** redirected - a client keeps retrying the address it was given until
  section 4's player port exists (and the replacement of a launched shard is handed the *next*
  free port of the range, not the dead one's: such a client does not come back by itself, and
  after the root's shard was replaced the address the dev script printed is stale); a client
  shows the
  old and the new scene's bodies together for about 0.3 s after the revision is raised; an
  orderly exit of the orchestrator on a signal (its shards notice the lost link after QUIC's idle
  timeout, about 11 s, and stop by themselves; the pid file of a killed orchestrator still lists
  them). **Since phase 2 (section 4)** the player port exists: a client that lost its shard dials
  its entry address again and is sent to the replacement, and a shard no longer puts a joiner on
  its region's centre - the two remarks above about that are history.

## 3. Load and autoscaling

- Shards send `LoadReport` once a second: players, measured tick p95, entity count, and the hot
  cell (the depth-64 cell holding most of its players, with that count).
- **Out**: the metric `max(players / capacity, tick_p95 / 33.3 ms)` above 0.8 for 10 s, fewer than
  `max-shards` live, and the hot cell not being the shard's whole load in its only cell: launch a
  shard and carve the hot cell for it.
- **In**: a carved shard beyond `min-shards` whose load together with its fallback owner's stays
  under 0.35 for 60 s is merged. The merge commits first; the child then **drains** - it owns
  nothing, so its ships and bodies leave through ordinary handoffs - and is stopped when it reports
  zero entities or after a timeout. For that: the kill is deferred, reports of a draining shard are
  kept, and peers keep accepting a shard for a while after a map change removed it.
- Capacity, thresholds and dwell times become flags, so a test can make the cluster scale with a
  handful of clients.

## 4. One address

- The orchestrator gets a second listener, the **player port** (`--player-bind`, default port
  7777). It answers a client `Hello` with a new message `JoinRedirect { shard, shard_map }` naming
  the owner of the spawn point, or with a `Reject` ("starting") until `min-shards` are up. The
  control port (7900) is for shards only and is never published.
- The client treats every address alike: `Welcome` = this is a shard (standalone setups and tests
  keep working); `JoinRedirect` = dial that shard, two hops at most. `Hello` keeps its layout so a
  version mismatch stays a readable `Reject`.
- A shard that receives a fresh join without owning the spawn point answers with `JoinRedirect`
  too instead of spawning the player at its region's centre. Adopting a handed-over player is
  untouched.
- `ShardRegion` gets a **client endpoint** beside the internal one. An unspecified IP there
  (`0.0.0.0:port`) means "the host you dialed, this port": the shards run on the orchestrator's
  host, so no public host name is configured anywhere. The client applies this to the join
  redirect and to the handoff `Redirect` alike. The verifier accepts any certificate, so the same
  host on another port works as it is.
- A shard may register with an unspecified IP in its internal endpoint; the orchestrator fills in
  the address the control connection comes from (containers). `Inbound::Hello` carries that address.
- Dev scripts: orchestrator player port 7777, shards from 7870 up. `single` and `two-shards` stay.

**Status: built (phase 2).** What it became, and where it differs from the bullets above:

- **Wire** (`PROTOCOL_VERSION` 7 — 6 is what `main` shipped meanwhile, the merged demo area with
  `SceneClock` at tag 28; see section 9): `Message::JoinRedirect { shard, shard_map }` is tag 29
  (`VARIANT_COUNT` 30); `ShardRegion` ends with `client_endpoint`
  (`ShardRegion::new` sets it to the endpoint; `resolve_client_endpoint(addr, entry_host)` is the
  one rule: port 0 = none, an unspecified IP = the entry's host, anything else as it is;
  `ShardMap::client_endpoint_of`); `ShardHello` carries an optional `client_endpoint` before its
  secret. **Added beyond the plan:** `RejectReason::Starting`, appended, so "the cluster is
  starting" is a reason a client can tell from a refusal and retries on, not a sentence in `Other`.
- **Player port** (`space-orchestrator/src/player.rs`, `--player-bind` /
  `SPACE_ORCHESTRATOR_PLAYER_BIND`, off unless given, no token on any bind, never the control
  port's port): a listener of its own that reads one hello, checks the version, answers and
  closes. It holds no reference to the orchestrator: the run loop (`run_with_player`) publishes
  the current answer after every turn and the listener reads it. 256 hellos in flight at most, 5 s
  for a hello, no per-player state. A first packet from an address that is not validated yet is
  answered with a QUIC retry and takes none of the 256; the answer's write gives up after a second
  (a peer that grants no receive window), and the whole of one connection is bounded from outside
  as well. The map in the answer is `ShardMap::for_players()`: client endpoints only. **"Starting"
  is not "until `min-shards` are up"** as the
  bullet says: the answer is a redirect as soon as the committed map's spawn owner is a connected
  shard with a client endpoint, and "starting" otherwise (no root yet, the owner dead and not yet
  replaced). Waiting for all of `min-shards` would keep players out for no reason - the other
  shards are reached by walking.
- **The control port refuses players** with a sentence ("this is the control port ...; players
  dial the player port") instead of treating their hello as a broken shard hello, and counts
  them; a shard-style hello at the player port is refused the same way.
- **Shards** (`Shard::join -> Result<Welcome, JoinRefusal>`): a fresh join is taken only when
  `free_spawn_point()` has a point, which is only on the spawn point's owner; everywhere else the
  answer is `JoinRedirect` to the owner, or `Reject(Starting)` when the map names no reachable
  owner (the shard itself is named but does not own the ground yet, or the owner has no client
  endpoint). No session is spent on a redirect. **The region-centre fallback is gone**, also on
  the adoption path: a client that said hello for a handoff which then never arrived is, at the
  adopt timeout, sent the same `JoinRedirect` and its connection closed, and the client joins
  afresh there. Adoption itself is untouched (H1-H8).
- **Client** (`link.rs`): `ServerLink::join(entry)` follows at most `MAX_JOIN_REDIRECTS` = 2
  redirects and names the hops when it gives up; the entry address is what is remembered, retried
  and shown, the shard's own address is `shard_address` (harness `/state`, Lua). A handoff
  `Redirect` is dialed at the target's **client** endpoint, resolved against the entry's host.
  **The handoff's hello** (changed by the merge with `main`, `d83fe04`): the offering shard sends
  the client's `Redirect` when the accepting shard's `HandoffAccept` arrives, not with the offer,
  so the transfer is staged before the client can say hello there, and the hello is recognised by
  its token. The client therefore says hello **once** (it used to ask up to 5 times, 120 ms
  apart, while the two went out in the same tick). What is left is a transfer the accepting shard
  dropped again before the hello arrived: it then answers like for a fresh join - a
  `JoinRedirect`/`Starting`, which ends that attempt (the old shard offers again), or, on the
  spawn point's owner, a welcome with a new ship, which the client recognises in the first frame
  and leaves at once. A `Reject` as the last word on a handoff's connection ends it like a
  `JoinRedirect` does, and a connection that has closed is never reused for the next redirect.
  **Only the ship's own player is redirected** (`Shard::redirect_owner`); a bystander keeps seeing
  the ship as its own shard's ghost of it, and a client ignores a redirect for a ship it does not
  fly. A hello without a player token is refused by every shard.
- **Client endpoints in the map**: the orchestrator keeps one per shard beside the endpoint, sets
  it in every region it gives (hello, carve, replacement) and fills an unspecified **internal**
  IP from the control connection's source (the free function `control::as_seen_from`, which the
  run loop applies to a hello when it handles it); a client endpoint's
  unspecified IP is left as it is - that is its meaning. The `process` launcher starts a shard
  with `--client-endpoint 0.0.0.0:<its port>`. `/api/state` has `player_port { bind, answer,
  sends_to_shard, redirected, told_starting, refused_busy }`, `map.spawn_owner` and a
  `client_endpoint` per region and shard; the page shows the same.
- **Dev scripts**: the orchestrated topology is always `--launcher process` behind
  `--player-bind 127.0.0.1:7777` (`-MinShards` default 1); the client is pointed at the player
  port; the scripts wait for `player_port.answer == "redirect"`.
- **Not done here**: shards the `process` launcher starts still **bind loopback**, so a player on
  another machine is sent to `<entry host>:<port>` and reaches nothing - publishing shards is
  section 8's. The compose files are untouched: they run one hand-started shard without a player
  port, and players dial that shard as before. A player who comes back after its shard died
  starts on the spawn point again (nothing of a player is kept across a shard's death). A client
  notices a dead shard through QUIC's idle timeout (about 10 s) before it dials the entry again.
  The player port has no per-source limit, and the spawn point's shard takes 100 players and
  evicts nobody for being idle (both in `docs/security-notes.md`, "Player port").

## 5. One cluster tick, seamless crossing

- `ShardWelcome` carries the cluster's **tick origin** (a wall-clock instant). A shard reads the
  wall clock once to place itself on that timeline and then runs on the monotonic clock: tick `n`
  is due at `origin + n / 30 s`, late ticks are caught up (bounded), and a shard that cannot keep
  up jumps forward and says so in its log and its load report. Standalone shards take
  `--tick-origin` (the dev script gives both the same); tests that step shards by hand stay at 0.
  All shards run on one host (decision 5), so their clocks agree; drift between hosts is not solved.
- **Handoff** (the three-phase exchange and its message layouts stay): the commit's state carries
  the tick it was read at, as a private component beside the owner token. The receiver sets the
  input floor to **that** tick and, installing at a later local tick, brings the entity forward by
  the missing ticks - a ship with the pure ship integrator and the inputs it holds for those ticks,
  a body along its velocity and the local pull. H2, H4 and H5 are reworded to match and their
  tests extended.
- **Client**: with one timeline the clock keeps steering through a switch, the prediction world
  and input history are kept (the own ship is followed by guid, not by session), inputs for ticks
  around the switch go to both shards (each applies only its side of the floor), and other
  entities keep their histories, merged by guid.
- Ships get a small border hysteresis (bodies have 2 m) so hovering on a border is not a handoff
  every second.
- **Test**: a client crosses a border at speed holding thrust; the drawn own ship and the camera
  move without a step beyond centimetres, speed rises without a flat spot, no input is lost. And
  the same while turning, and back across.

**Status: built (phase 3).** What it became, where it differs from the bullets above, and what was
measured. Numbers are from `crates/space-client/tests/seamless.rs` (four in-process shards started
137 ms apart, the client core flown frame by frame at about 120 frames a second, a debug build,
2026-11-24) unless a live stack is named; "before" is the same rig on the code as it was
(`7c26931`).

- **What a crossing looked like before.** Every shard counted from its own start, so at every
  change of shard the client's draw clock **snapped** by the difference of the two shards' ticks
  (-4.3, -8.3, +3.9, +8.5 ticks between shards started 137 ms apart), its prediction was built
  anew from the new shard's first word with no inputs to replay, and the ship as drawn was about
  **two ticks of flight** off — 0.7 to 0.9 m at 13 m/s — which the client then absorbed over a
  third of a second (a visible glide, not a cut; above 2 m it would have been a cut). The
  existing two-shard test saw none of it: it watches the shard's word (0.21 m worst), not what is
  drawn. On the shards' side the ship was installed on "accept tick + lead" of the accepting
  shard's own clock, a tick early or late by the two loops' phase: a tick nobody simulated, now
  and then. Inputs sent between the commit and the switch went to the shard that no longer had
  the ship.
- **Cluster clock** (`space-server/src/clock.rs`): `ShardWelcome` ends with `tick_origin_us`
  (`OrchestratorConfig::tick_origin_us`, `--tick-origin`), every shard runs `clock::run` on
  `TickClock::new(origin)`: tick `n` is due at `origin + n / 30 s`, the wall clock is read once,
  late ticks are made up (8 at most), a longer stall is jumped over (`Shard::skip_to`,
  `ShardStats::skipped`, three log lines a minute). **Deviations:** (a) the default origin is a
  **fixed instant** (`SceneClock::DEFAULT_ORIGIN_US`, 2026-01-01T00:00:00Z) for the orchestrator
  and for a standalone shard alike, so the dev scripts give nothing and two shards started by
  hand agree without a flag; `space-server --tick-origin` exists for whoever wants another
  timeline. Tick numbers are therefore large (8.5e8 today); nothing casts one to `f32`. (b) It is
  **not a second clock beside `SceneClock`**: the cluster clock *is* the scene clock a shard tells
  its clients (`SceneClock::cluster(origin)`), set once and never re-anchored. `main`'s
  wall-clock re-anchoring (`Shard::sync_scene_clock`, 30 ms tolerance) is called by neither
  binary any more; it is left for a shard driven by a loop of its own. (c) "Says so in its load
  report" is not built: no shard sends a load report yet (section 3). Measured: three shards
  started 1.3 s and 0.9 s apart are never more than one tick apart when looked at together
  (`tests/crossing.rs`); on a live four-shard stack every shard's log names the same tick.
- **Handoff** (`handoff.rs`, "The tick of a state"; H2, H4, H5 and H6 reworded). **Deviation:**
  no new private component. A handoff state has carried its scene time since `main`'s orbits
  (`SCENE_TIME_COMPONENT`), and on the cluster clock that *is* the tick; it got a ninth byte,
  "read off a shared clock". The accepting shard reads the tick off its own clock and installs
  the entity **on it**: a commit that is early waits for the tick (the plan only had the late
  case), one that is on time is installed before the next step — the tick after the commit is
  the accepting shard's, none is lost — and one that is late is brought forward: a ship by the
  flight model's free-flight step under the inputs kept for those ticks, a body along its
  velocity. **Deviation:** no "local pull" in the catch-up (a body at rest on a floor would be
  sunk by it), no contacts, and a ship in fast travel only coasts; at most 15 ticks, beyond which
  the two clocks are taken not to agree and the old rule applies (accept tick + lead) — as it
  does between shards stepped by hand, which is why the 41 acceptance tests pass unchanged.
  `ShardStats::caught_up` counts the ticks made up: 0 in every flight of the client tests but the
  fast one over the demo area (11 ticks in all, among some hundred transfers, most of them
  bodies; which entities they were was not looked at). Checked tick for tick against a ship that
  never changed shard, bit for bit: on time, 1/2/4 ticks late, early (`tests/crossing.rs`).
- **Inputs.** The client sends every batch to both shards while a handoff is in flight; the
  accepting shard keeps what it is sent for a staged ship (`staged_inputs`, 45 at most), replays
  the catch-up's ticks from it and files the rest at the install; the input floor is the install
  tick. **Added:** the client now gives **every tick exactly one input** (`link::inputs_for`).
  It used to tag each batch with its clock's estimate, which names a tick twice now and then and
  skips one as often; a skipped tick coasts on the shard, which in fast travel is a fifteenth of
  the speed gone. Measured: a key pressed 0.3 to 1.25 s after the offer — before the commit, at
  it, after the switch — shows in the shard's word after 94 to 105 ms, as one pressed far from
  any border does (74 to 110 ms); in fast travel the prediction's corrections went from up to
  1.0 m to 0.
- **Client** (`link.rs`, "One timeline across the switch"; `predict.rs`): at the switch the
  link checks that the new shard's first tick is within 6 of the tick it expects; then nothing
  but the session changes — tick, sent inputs, `input_ack`, and the history of every entity both
  shards report (merged by guid, `ShipView::continue_from`). The draw clock and the prediction are
  keyed by ship and `Replica::tick_epoch`, which only goes up onto a shard that counts other
  ticks (then everything starts over, as before). Measured over 12 crossings at 11-14 m/s
  through all four cells, both borders both ways: drawn ship off by at most **3.5 mm** from
  where the frame before puts it (before: 0.7-0.9 m), drawn velocity steps 0.12 m/s (the
  thrust's own), clock 0.02 ticks, **correction 0.0000 m**, one handoff per crossing, 1.13 s at
  most from a metre past the border. Beside the showcase (8 crossings through the zero-g
  corner): all 35 bodies in sight before and after are drawn in every frame between, once; the
  largest move of one beyond its motion 3 cm. In fast travel (up to 330 m/s, ten metres a tick):
  0.13 m at the switch frames, 1 ms of flight — what any frame of fast travel shows (99 % under
  1.04 ms): its velocity is not smooth, because the speed limit follows the distance to the
  nearest surface.
- **Ship hysteresis** (`border::SHIP_HYSTERESIS` = 1 m, and the 3 s settle rule bodies have): a
  ship is offered once it is a metre inside the region it entered, measured to the nearest
  ground of the shard that simulates it — exactly, round the corner where four cells meet
  (`near_territory_distance`; `home` used a lower bound that held a body on the wrong one of
  three shards) — or once it has been inside for three seconds. Measured: walking along x = -6
  for 20 s, weaving 0.27 m to either side: **0 handoffs** (before: 13 in 14 legs); standing a
  hair across it: one, after three seconds, and none in the next six.
- **Is the second of lead needed, and does a player notice it?** It is what the client is given
  to open its second connection, and the commit waits for it. Nothing is frozen meanwhile: the
  old shard simulates the ship as before, inputs act as fast as anywhere (above), nothing drawn
  steps. A player sees it in the HUD's shard number, which changes a metre and a second after
  the border, and in nothing else. It was not shortened. What it does cost is that for that
  second the ship is simulated by the shard it has left — 15 m into the neighbour's ground at
  flying speed, hundreds in fast travel — and meets that shard's bodies as ghosts unless the
  island rule has moved them first.
- **On a live stack** (`dev.ps1 -MinShards 4`, release build, two headless clients at 120
  frames a second driven through the harness, `/state` sampled about 140 times a second; the
  raw samples are in `target/dev/shots/phase3/`): 40 crossings of both borders, both ways —
  12 at 10-15 m/s, 16 at walking speed (2 m/s), 12 in fast travel (28-355 m/s). The drawn ship
  was off from where the sample before put it by at most 4.0 mm flying and 3.6 mm walking
  (0.012 and 0.06 of a tick of motion), and by 0.22 m in the median, 0.29 m at most in fast
  travel (0.05 and 0.10 of a tick of motion — fast travel's own unevenness); the draw clock
  stepped by at most 0.057 ticks at a switch; `correction_m` was 0 at every one of the 40;
  from a metre past the border to the new shard number took 1.00-1.30 s (median 1.12); all
  35 bodies were listed throughout. Standing 10 cm across x = -6 for 15 s: one handoff (after
  three seconds), then none; walking along it for 20 s, 0.32 m to either side at most: none.
  One client standing 1.25 m and 9.2 m beside the path of another that crossed ten times
  each: 0 handoffs, the same session, the other drawn in every sample. (In an earlier run the
  two ships collided — the pilot's lane was a metre off — and the one who stood was knocked
  across the border and then handed over, once, for where it was.) A floating crate knocked
  from x = -17 over the border to x = -1 was in every sample of both clients, never moved
  backwards, and the two clients' views of it were never more than 0.1 m apart. All four
  shards' log lines name the same tick at the same moment; no shard jumped a tick; no
  player's transfer was called off; both clients ended with 0 validation messages.
- **Not done here, and what still looks wrong.** Hosts whose clocks differ (decision 5: one
  host). The web UI does not show the tick origin. A player in fast travel takes every body its
  second of motion reaches along (the whole showcase, from a hundred metres up) and they come
  back afterwards: correct, invisible to clients, but tens of body handoffs per fly-by, some
  refused by the crossing rule (section 7): after half an hour of the live run above the root's
  shard had handed 659 entities out and counted 197 body transfers called off. Bodies that have
  been knocked further than the ghost margin from a border (two crates ended up six kilometres
  out) are drawn by the clients of the shard that simulates them and by nobody else, so the
  count of bodies in sight changes at a border by exactly those: that is section 6's far
  replicas, not a crossing. Fast travel's velocity kinks are the flight
  model's, not a border's. Noticed while reading, not verified: `ShipView::apply` treats every
  chunk of a snapshot as the whole interest set, so a snapshot too large for one frame would
  lose its earlier chunks on the client (the demo area's fits one).

## 6. Far replicas and level of detail

- Two tiers across shards: **ghosts** as today (near a border, in the physics world) and **far
  replicas**: `SimWorld` entities with their components but **no rapier body**, so the interest
  grid and the replication path treat them like any entity and physics never sees them.
- Once a second each shard tells the others where its players are (`FarInterest`). A shard sends
  its own ships and bodies that are beyond ghost reach of shard B but within 10 km of one of B's
  players as `FarUpdate`, at a rate by the distance to B's nearest player (10 Hz to 250 m, 5 Hz to
  1 km, 1-2 Hz beyond), with a removal list, a periodic full refresh and a silence rule like the
  ghosts'. The receiver moves a far replica along its velocity between updates.
- One guid is in exactly one state on a shard - own, ghost or far replica - and changes state in
  place, so clients keep its history as they do for own <-> ghost today.
- **Per-client distance steps** for every entity (own, ghost, far): near things every tick, far
  things less often. `ClientReplica` gets a last-sent tick **per entity** in place of the single
  change base, so an entity skipped for a while and then at rest is still brought up to date. This
  is prototyped and tested on its own before far replicas build on it.
- **Client**: the delay an entity is drawn at follows its observed update spacing, the 15-tick
  restart rule and the extrapolation limit scale with it, and interpolation uses velocities, so a
  thing updated five times a second moves smoothly. Far replicas are never copied into prediction.

## 7. Physics across four borders

**Seen on a live four-shard stack (2026-11-24, after the merge with `main`), and solved in
phase 3** (the rules: `border.rs`, "Players' ships"; the tests: `space-server/tests/crossing.rs`,
`space-client/tests/seamless.rs`):

- **A standing player was handed over because another flew by.** Confirmed as the cause: the
  island rule moved a player's ship like any member. Two ships within reach of each other's
  second of motion are one island, its home was that of its lowest guid, and *every* member was
  offered there — so the player whose ship has the higher guid (the one who joined later) went
  wherever the other one's home was. (With the order of joining reversed — the one who stands
  has the lower guid — the same fly-bys handed nobody over who did not cross: 0 handoffs.) Now
  a player's ship
  is never moved by an island: it is offered for its own position only, and in an island's key
  it counts with the shard that simulates it. Measured: a player 1.5 m and 9 m beside the path
  of one who crossed ten times: 0 handoffs, the same session throughout.
- **A crossing beside the showcase's bodies took 3 s instead of 1.** The refusal was "crossing
  island transfer; the lower shard id takes it": as the ship came near, the corner's shard sent
  the crates its second of motion reached to the ship's shard (consolidation), one of them was
  under way at the moment the ship's shard offered the ship *in a group with its bodies*, and the
  accepting shard refused the group's first member — the ship — because it was itself sending a
  member of that island the other way. The whole group was called off and paused for two
  seconds. Deterministic on the way out of the spawn cell (3.11-3.12 s, four times of four), not
  on the way back (1.11-1.14 s). Now the crossing rule is for bodies only, a player's ship
  commits when its own gate opens whatever becomes of the bodies travelling with it, and a
  refused body calls off the group's bodies, not the ship. Measured: 1.10-1.13 s, eight of
  eight. Shards now say which transfer was called off and why (`Shard::recent_aborts`, a few
  log lines per ten seconds).

**Still open for this section** (seen in phase 3's runs): bodies follow a player across a border
and back in numbers (above, section 5's last bullet); two shards still refuse each other's body
groups by the crossing rule now and then (harmless: retried after the pause, no client sees
it).

`space-server/tests/physics.rs` gets a four-shard fixture with the borders through the showcase:
islands whose members have three or four homes, ghost refresh between every pair, hysteresis at
the corner where four cells meet, a player walking through the room across both borders. What
breaks is fixed in `border.rs` / `shard_physics.rs` with the rules written down there.

## 8. Docker launcher and compose

- A hand-written Docker Engine API client (HTTP/1.1 over the Unix socket; `cfg(unix)`, elsewhere
  the launcher reports itself unavailable): ping, pull, create, start, stop, remove, list by
  label, inspect. JSON through `serde_json`, which the workspace already uses.
- `spawn`: create and start a container from the shard image, labelled with cluster, role and
  shard, on the orchestrator's own network (found by inspecting itself), one host UDP port from the
  range mapped to the shard's game port, removed when it exits. `kill`: stop and remove. `poll`:
  inspect. At start the orchestrator removes leftover shard containers of its cluster.
- The shard image is the one built from the orchestrator's own commit (`sha-<short>` tag baked in
  at build time); `release-images.yml` pushes the shard image before the orchestrator's `:latest`
  moves. Shard containers are not Watchtower's business. A registry credential, if the images are
  private, comes from the environment.
- Compose, local and production: the orchestrator only, socket mounted, player port published,
  control port not published, web UI on loopback as now.

## 9. The wire (PROTOCOL_VERSION 6 -> 7)

`ShardRegion` + client endpoint. `LoadReport` + entity count, hot cell. `ShardHello` + secret.
`ShardWelcome` + tick origin. Peer hellos carry the peer token in the existing token field.
Appended variants: `JoinRedirect`, `FarInterest`, `FarUpdate` (`VARIANT_COUNT` 32). The handoff
messages, the epoch messages and `Hello` keep their layouts.

**Status.** This plan was written as "5 -> 6". While the branch was being built `main` shipped a
protocol 6 of its own - the merged demo area: `SceneClock` (tag 28), `PlayerInput::fast`, 25-byte
`ShipControls` - and that 6 stands as released. Everything of this branch is therefore
**protocol 7**, on top of it: built so far are `ShardHello` + optional client endpoint + secret,
`ShardRegion` + client endpoint, `RejectReason::Starting`, the peer token in a shard's hello and
`JoinRedirect` at tag 29 (`VARIANT_COUNT` 30), and since phase 3 `ShardWelcome` + tick origin
(`tick_origin_us`, its last field). `FarInterest`/`FarUpdate` (30, 31) and the load report's new
fields are still to come and stay inside 7 as long as 7 has not been pushed to `main`; once it
has, the next layout change is 8. (The handoff state's private components are not wire layout:
the scene-time component got a ninth byte in phase 3.)

## 10. Phases

Each phase ends with `cargo fmt`, `clippy -D warnings`, the tests of the crates it touched,
`cargo run -p xtask -- ci` with `SPACE_REQUIRE_GPU=1`, a look at a running stack, and the docs it
made untrue brought up to date.

1. **Map and lifecycle.** Carve and ancestor merge, depth 64, all-shard recipients (first, as a
   pure test: carve three cells, merge one); the anchor rule and spawn points; regions at hello;
   `min/max-shards`, the reconciler, the `process` launcher, supervision, secrets; load reports.
   The dev scripts get the four-shard topology as an option, not yet the default.
2. **One address.** Player port, `JoinRedirect`, client endpoints, the address a shard is seen at.
3. **Seamless crossing.** Tick origin, the handoff state's tick and the catch-up, the client's
   clock, prediction and histories across a switch, ship hysteresis.
4. **Physics on four shards.** The fixture, the fixes; then four shards become the dev default.
5. **Far replicas.** The per-entity change base, distance steps, far replicas, sparse-sample
   interpolation on the client.
6. **Docker and autoscaling.** Engine API client, launcher, compose and Dockerfiles, the hot-cell
   and drain policy, the workflow order.
7. **Demo-area content** (after the hold): a far rock with bodies in orbit some 350 m out in
   another quadrant, and the showcase tour on four shards.

## Not solved here

- Shards on several hosts (clock drift, public addresses per host).
- A shard smaller than one cell.
- Players resuming where they left.
- Keeping a dead shard's bodies.
