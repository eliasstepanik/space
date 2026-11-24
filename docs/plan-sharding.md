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
note under section 1 for what they do now). (Note, 2026-11-24: sections 2 to 6 and 8 are built
since, phase 6 being the last. The bullets below about the hello, the launchers, load reports,
merges, the cluster tick, the handoff and the client likewise describe the code *before*; each
section's status block says what it does now.)

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
  sends one yet: the tests that merge a carved cell say so on the shard's behalf. (Since phase 6,
  section 3: shards report, and a merge needs the report but no empty shard — the shard drains
  after the commit.)
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

**Status: built (phase 1, second half), except the `docker` launcher (section 8; built in
phase 6).** What it became, and where it differs from the bullets above:

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
  its region's centre - the two remarks above about that are history. **Since phase 6 (sections 3
  and 8)** the `docker` launcher is built; the orchestrator leaves its loop on SIGTERM or Ctrl-C
  and the launcher stops what it started (`Launcher::shutdown`, the `process` launcher too); and
  a merged shard is no longer stopped at the commit: it drains and is told to stop
  (`ShardRetire`), so "a merged shard" in the supervision bullet above is history as well.

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

**Status: built (phase 6).** What it became, where it differs from the bullets above, and what
was measured. The code: `space-server/src/load.rs` (the report), the orchestrator's
`autoscale.rs` (the decision, a pure function of loads, tick and prior state; `ScaleConfig`) and
`orchestrator.rs` ("Scaling": `shard_loads`, `scale_out`, `begin_drain`, `drain_pass`,
`drain_ended`), `space-server/src/shard.rs` (`report_load`, `load_report`, `note_departing`,
`knows_shard`). Live numbers are from `dev.ps1 -ServersOnly -PortOffset 300` with
`SPACE_ORCHESTRATOR_SHARD_CAPACITY=2`, `_SCALE_OUT_AFTER=5`, `_SCALE_IN_AFTER=15`,
`_SCALE_COOLDOWN=10` (the `process` launcher, release build, two headless clients at 60 frames a
second on the harness, `/state` sampled about 190 times a second, 2026-11-24; raw samples,
frames, state dumps and logs in `target/dev/shots/phase6/` and `target/phase6/`).

- **Load reports.** Every orchestrated shard reports once a second (`Shard::report_load`,
  switched on in `start_orchestrated`; a standalone shard reports nothing): players, entities
  (the ships and bodies it simulates — not ghosts, not far replicas), the tick time's mean, p95
  (nearest rank: of 30 ticks the 29th) and worst over that second, the ticks it has jumped in
  all, and the hot cell with its player count (ties go to the smallest index). **Deviation:**
  the plan named p95, entities and the hot cell; mean, worst and the jumped ticks were added
  (the last is section 5's "says so in its load report"). The tick time is measured around
  `Shard::tick` — the simulation and the building of frames, not the socket writes. A report is
  taken only from the connection that was welcomed as that shard (the rule of section 2), from a
  draining shard too, and is shown per shard in `/api/state` (`load_report`: `players`,
  `entities`, `tick_mean_ms`, `tick_p95_ms`, `tick_max_ms`, `ticks_jumped`, `hot_cell { cell,
  players }`, `load_metric`) and in the page's shard table.
- **Out.** Load = `max(players / capacity, tick p95 / 33.3 ms)` at or above `--scale-out-load`
  (0.8) for `--scale-out-after` (10 s): the shard's hot cell is carved for a new shard
  (`begin_carve`: the node is wanted, a shard is launched, the epoch is proposed when it
  registers), up to `--max-shards` (8); at the limit the refusal names it. Of several hot
  shards the lowest id goes first. **When the hot cell is the shard's whole region** — a
  one-cell shard, in particular the spawn point's cell once it has a shard of its own, where
  every player joins — **nothing is done**, and the event log says so once per cooldown: "...
  a region cannot be smaller than a cell (1024 m), so no new shard can take part of this load".
  **Added:** a report that names no hot cell (a shard that is slow without players) falls back
  to the old octant split, which works for regions of 1024 km and more only.
- **In.** **Deviation: the pair is measured as one shard would be** — the players of both
  against one capacity, the tick p95 of both summed against one budget — at or below
  `--scale-in-load` (0.35) for `--scale-in-after` (60 s). The newest shard (highest id) goes
  first, so a cluster that grew and shrinks ends with the shards it began with. **Never
  merged by the autoscaler:** the shard that owns the spawn point; one of the layout cells
  `--min-shards` stands for (with 4: the three cells beside the spawn point's — the demo area
  keeps its four areas, and what comes and goes are the shards the autoscaler added); anybody
  while the cluster is at `--min-shards`. The layout-cell rule is a decision made while
  building, after a live run in which a quadrant was merged away while the players were out at
  a carved cell. An operator's merge (web UI) is held to `--min-shards` only: it may merge a
  layout cell or the spawn point's shard, and the player port follows the new owner at the
  commit.
- **No flapping.** The band between the two thresholds; a load that crosses back starts its
  dwell over; `--scale-cooldown` (30 s) per shard after any action that involved it — also
  after "nothing can be done", and for the shard that takes a region back; one action per
  out-dwell in the whole cluster; one epoch change at a time (section 2).
- **Settings.** `--shard-capacity` (100; that is also the most a shard takes,
  `MAX_PLAYERS_PER_SHARD`), `--scale-out-load`, `--scale-out-after`, `--scale-in-load`,
  `--scale-in-after`, `--scale-cooldown`, `--drain-timeout` (30 s, 45 at most); each has an
  environment form (`SPACE_ORCHESTRATOR_SHARD_CAPACITY`, `_SCALE_OUT_LOAD`, `_SCALE_OUT_AFTER`,
  `_SCALE_IN_LOAD`, `_SCALE_IN_AFTER`, `_SCALE_COOLDOWN`, `_DRAIN_TIMEOUT`); seconds on the
  command line, ticks inside. Settings without a band (in >= out), a capacity of 0, a dwell of
  0 or a drain over 45 s are refused at startup. What runs is in `/api/state` ->
  `autoscaler.config` and on the page's Autoscaler card, which can still be paused.
- **Drain, then stop: the premise of a merge changed.** A merge used to need `players == 0` and
  the shard was killed at the commit. Now it needs only the shard's fresh report. At the commit
  the shard owns nothing and **drains** (role `draining` in `/api/state` and on the page):
  everything it simulates lies on another shard's ground and is offered there by the ordinary
  handoff, its players redirected as at any crossing. The other shards go on hearing a shard
  that left the map for `FORMER_SHARD_GRACE_TICKS` = 60 s (`Shard::knows_shard`,
  `peer_endpoint`) — which is why the drain timeout is capped at 45 s. The orchestrator keeps
  the reports; when one that arrived **after** the commit says 0 players and 0 entities it sends
  `ShardRetire` (tag 32). The shard prints `space-server: shard N stops: the orchestrator
  retired it (...)`, stops and **exits 0** — the one orderly end of an orchestrated shard. The
  orchestrator logs `shard N is drained after X s (0 players, 0 entities) and is told to stop`
  and `shard N stopped as it was told (exit status 0)` (the status is the launcher's:
  `Launcher::exit_code`, which the `process` and `docker` launchers have), closes the link and
  reaps the process or removes the container. Not drained within `--drain-timeout`: stopped as
  it is, with a warning that names what it still held ("what it held is gone"). Told to stop
  and still there after 5 s (`RETIRE_GRACE_TICKS`): link closed, process stopped. A shard that
  dies while draining is said to have, and nobody is replaced: it owned nothing. A draining
  shard does not count towards `--min-shards`.
- **Measured in the fixture** (`space-server/tests/four_cells.rs`,
  `a_shard_whose_region_is_merged_away_hands_everything_over_and_is_empty`): the diagonal cell's
  shard with 7 scene bodies and a standing player is empty after 32 ticks, 0 transfers called
  off, no resting body moved 1 cm.
- **On the live stack.** Four shards after 1.2 s; idle reports: 0 players, 19/3/6/7 entities,
  tick mean 0.5-0.7 ms, p95 0.6-1.0 ms, load 0.02-0.03. Two clients joined at the spawn point
  and lifted off: 5.2 s after the second was online the root's shard (2 of 2 players in cell
  `64:8388608:8388608:8388608`, load 1.0) was relieved — wanted, launched, registered and
  committed within 2 ticks of each other on the orchestrator's clock (tick 843 -> 845); both
  players were handed over 1.1 s later while moving at 5.7 m/s: drawn step 5.7 mm and 3.6 mm,
  correction 0, 35 bodies listed throughout, 21 of the scene's entities went with the cell.
  Then, once per about 45 s: "split shard 92 was not carried out: 2 of shard 92's 2 player(s)
  are in one cell, and that cell is its whole region...". Both flew 1.9 km out in fast travel:
  5 s after they stopped the root's shard was relieved at cell `64:8388609:...` (a sixth
  process; handed over in place: 5.1 mm, correction 0). Both flew back: 15 s and the cooldown
  later `merge shard 93: epoch 7 proposed`, committed in the same tick, `drained after 0.8 s`,
  `stopped as it was told (exit status 0)` 11 ticks later; its pid left `launched-shards.pid`
  (six pids, then five); nobody was sent to the connect screen; no player's transfer was called
  off. In an earlier run of the same day, before the layout-cell rule, a layout quadrant with 7
  scene bodies was merged instead: `drained after 0.9 s`, and another with 6 bodies `after
  4.3 s`, exit status 0 both. Four flying crossings of x = -6 afterwards at 25-65 m/s: drawn
  step 4.5 / 5.6 / 6.7 / 11.4 mm, corrections 0. `scripts/harness/smoke.lua` passed (a third
  client, exit 0). Both clients ended with 0 validation messages.
- **Tests.** Each names its mutation and each mutation was run (35 of them,
  `target/phase6/mutations.py`; three survived the first run and the tests were tightened until
  none did). `space-server/src/load.rs` (4 unit tests); `tests/four_cells.rs` (the drain above,
  `a_shard_reports_its_load_once_a_second`); `tests/orchestrated.rs`
  (`a_shard_told_to_stop_ends_in_an_orderly_way`); `autoscale.rs` (the decision table, the pair
  load, the cooldown, the hot-cell target, the settings); `orchestrator.rs`
  (`a_merged_shard_drains_and_is_told_to_stop_only_when_it_is_empty`,
  `a_drain_that_does_not_end_is_cut_short`,
  `a_shard_is_merged_on_its_report_and_a_pair_only_when_cold_together`,
  `the_autoscaler_never_removes_the_spawn_points_shard_nor_goes_below_the_minimum`,
  `the_autoscaler_keeps_the_layout_cells_of_min_shards`,
  `the_cluster_grows_at_the_hot_cell_and_shrinks_again_by_draining` — the whole loop on the
  orchestrator's own tick, no sleeps — `a_loaded_shard_is_not_relieved_beyond_max_shards`);
  `main.rs` (the scale flags).
- **Tests whose premise changed.** The merge tests that held "only an empty shard is merged"
  and "the merged shard is killed at the commit": in `orchestrator.rs` they are now
  `a_merged_shard_drains...`, `a_shard_is_merged_on_its_report...` and
  `an_operator_merge_needs_a_report_and_the_shard_drains_after_the_commit`; the two cold-merge
  tests of `tests/split_merge.rs` now play the drain; in `tests/lifecycle.rs`,
  `tests/player_port.rs` and `tests/webui.rs` "no load report" became "the shard's own report,
  not the forged one", because real shards report now.
- **Not done here, and what still looks wrong.**
  - **Carving the hot spawn cell moves the load, it does not split it**: the players are all
    in that cell, so the new one-cell shard takes all of them. And since the spawn point's
    shard is never removed, a cluster that was busy at the spawn point once keeps that fifth
    shard (the root's shard, three layout cells, the spawn cell). A crowd in one cell cannot be
    split (section 1: the smallest area is one cell).
  - While a shard drains it gets no ghosts and no far replicas from the others (it owns no
    ground), so for the second or so of a drain its bodies are simulated without their
    neighbours'.
  - Seen on the live stack: switching fast travel **off at 775 m/s** (the flight model clamps
    to 120 m/s at once) gave a prediction correction of 23 m on that one tick and a drawn step
    of 7-16 m, a second before a change of shard; the change of shard itself had correction 0.
    It is the fast-travel toggle, not a border, and was not looked at on the code before this
    phase (not verified whether it is older).
  - Three body transfers were called off by an epoch change at the start of the cluster (known
    since phase 3, section 5).

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
  `--player-bind 127.0.0.1:7777` (`-MinShards` default 1 — 4 since phase 4, section 7); the
  client is pointed at the player port; the scripts wait for `player_port.answer == "redirect"`.
- **Not done here**: shards the `process` launcher starts still **bind loopback**, so a player on
  another machine is sent to `<entry host>:<port>` and reaches nothing - publishing shards is
  section 8's. The compose files are untouched: they run one hand-started shard without a player
  port, and players dial that shard as before.
  (Since phase 6, section 8: the `docker` launcher's shards bind the wildcard and publish their
  port on the host, and both compose files run the orchestrator alone, with the player port
  published. The `process` launcher's shards still bind loopback.) A player who comes back after
  its shard died
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
  (**Changed by the review, below:** "reads the wall clock once" did not keep two shards of one
  host together — a shard keeps measuring and slewing.)
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
  `TickClock::new(origin)`: tick `n` is due at `origin + n / 30 s`, the wall clock is read once
  (no longer: see the review's status block below),
  late ticks are made up (8 at most), a longer stall is jumped over (`Shard::skip_to`,
  `ShardStats::skipped`, three log lines a minute). **Deviations:** (a) the default origin is a
  **fixed instant** (`SceneClock::DEFAULT_ORIGIN_US`, 2026-01-01T00:00:00Z) for the orchestrator
  and for a standalone shard alike, so the dev scripts give nothing and two shards started by
  hand agree without a flag; `space-server --tick-origin` exists for whoever wants another
  timeline. Tick numbers are therefore large (8.5e8 today); nothing casts one to `f32`. (b) It is
  **not a second clock beside `SceneClock`**: the cluster clock *is* the scene clock a shard tells
  its clients (`SceneClock::cluster(origin)`), set once and never re-anchored. `main`'s
  wall-clock re-anchoring (`Shard::sync_scene_clock`, 30 ms tolerance) was called by neither
  binary any more and by nothing else; it and its test were removed in the review. (c) "Says so in its load
  report" is not built: no shard sends a load report yet (section 3; built in phase 6:
  `LoadReport::skipped`). Measured: three shards
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
  (As built in phase 3 the 15 ticks were checked only when the commit arrived; an entity
  installed after its shard had stalled was brought forward by the whole stall. Fixed in the
  review: the bound holds at the install.)
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
  lose its earlier chunks on the client (the demo area's fits one). (Verified and fixed in the
  review: `Snapshot::more`.)

**Status: reviewed (phase 3b).** An independent review of the phase-3 commit found the following;
each was checked against the code as it stood after phase 4, fixed where it held, and given a
test that fails without the fix. Numbers from a live stack are from `dev.ps1 -ServersOnly
-PortOffset 300` (four shards, release build, two headless clients on the harness, 2026-11-24;
logs and samples in `target/phase3b/` and `target/dev/shots/phase3b/`).

- **The catch-up was bounded when a commit arrived, not when it was installed.** A commit read
  within 15 ticks of the present and installed after its shard had stalled and jumped was brought
  forward by the whole stall: seconds of flight without a contact. Now the bound holds at the
  install (`PendingCommit::catch_up_at`), for ships and bodies: past it the state is installed
  as it is — the entity stood still for the stall like everything else on that shard.
  (`a_commit_installed_after_a_stall_is_not_brought_forward_by_the_stall`: a 90-tick stall with
  the commit in hand; without the bound the ship is 36 m on.)
- **A jump restarted scene time and dropped every input.** `Shard::skip_to` used the prediction
  mirror's `reset_clock`: after a jump the orbiting bodies moved by the whole stall and what
  they carry by one tick — a ship standing on the Moon was left 19 000 km behind it — and
  inputs filed for ticks *after* the jump were thrown away with the rest. `SimWorld::skip_to`
  keeps where scene time stood (the next step carries by all of it) and drops only the inputs
  of the ticks jumped over. So "the world stands still across a jump" is now exact about what
  it means: nothing flies, falls or collides for the skipped ticks; the sky moves on and takes
  what it holds along.
- **The clocks of two shards of one host could differ for good.** Each read the wall clock
  once and counted on the monotonic clock: whatever the two drifted apart by between two
  shards' starts — an NTP step, a suspended host, hours of drift before a replacement — stayed
  as a constant offset between them, and nothing measured it. Now every shard compares its tick
  clock with the wall clock once a second, says what it found (`ShardStats::clock_offset_us`,
  the census line: `... N ticks jumped, M caught up; clock offset +0.3 ms`) and corrects it: up
  to 2 ms is left alone, more is slewed away at 0.5 % at most (never backwards, a tick 0.17 ms
  longer or shorter), more than half a second **behind** is stepped over at once and handled as
  the stall it is. The clocks are read through `clock::TimeSource`, so tests set the time
  (`ManualTime`) instead of sleeping, and the loop without its sleep is `clock::Pacer`.
  **What is still not solved:** hosts whose wall clocks differ; the time a slew takes — 7 s per
  tick of offset, during which a crossing between the two shards is a commit that many ticks
  early or late (waited for, or caught up), so "a replacement started later counts the same
  ticks" is true to a few milliseconds once it has slewed, not by construction; and a wall
  clock stepped *back*, which is only slewed (200 s per second of step) and said in the log.
  On the live stack every shard's line read `clock offset +0.0 ms` for the half hour it ran,
  the replacement's included.
- **A tick origin in the future** made `tick_at` saturate at 0: the shard slept until the
  origin came while it admitted players. Both binaries now refuse such a `--tick-origin`, and a
  shard refuses a welcome that names one (`SceneClock::origin_refusal`), with a sentence that
  says by how much.
- **Inputs could be lost when the commit followed the client's connect within the input
  lead.** The client tags inputs three ticks ahead and sent each once: the ones it had sent
  just before its second connection existed reached only the old shard, and a shard dropped
  whatever a player sent before that player's status had arrived (datagrams are not ordered
  with the stream). Two ends: the link says the inputs for ticks still to come again the moment
  the connection is up (`link::resend_tail`), and the accepting shard keeps what the owner of a
  staged ship sends, status or not. With a connection that comes up as the lead runs out, a
  ship now flies bit for bit as on one shard, and each shard's record of the ticks it applied
  an input on (`SimWorld::applied_inputs`) names every tick once.
  (`inputs_sent_before_a_late_connection_are_said_again_and_kept`,
  `the_inputs_sent_just_before_a_handoff_connection_are_said_again_on_it`.) "Inputs are never
  lost" is narrower than it read: no input that reaches a shard in time is lost *to the
  handoff*; a datagram that is dropped or late is lost as on any tick, and nobody applies the
  inputs of ticks a stalled shard jumped over.
- **A snapshot in several chunks lost all but its last on the client** (seen while reading in
  phase 3, confirmed): each chunk was taken for the whole interest set. `Snapshot` ends with
  `more` (protocol 7): the client collects the names of every chunk and drops what none of them
  named when the chunk that says `more: false` arrives; a shard ends a snapshot whose rest
  despawned with an empty last chunk. (`a_join_snapshot_in_several_chunks_arrives_whole`: nine
  crates and a ship at 600 B a frame.)
- **`drain_inputs` on a ghost: refuted.** A ghost is imported without `ShipControls` and
  without an `Owner`, and every write of `drain_inputs` goes through one of the two: an input
  filed for a ship that has since been handed away does nothing. Pinned
  (`a_ghost_takes_no_input_that_was_filed_for_the_ship_it_replaces`); no code changed.
- **Two players in contact across a border.** Measured first
  (`two_players_pushing_each_other_across_a_border`, against the same on one shard): each ship
  met the other's ghost — a kinematic body at the other's velocity, which pushes and cannot be
  pushed — and was thrown back at the *other's* speed. Two players pushing at each other across
  `x = -6` bounced apart to 2.3 m and back for as long as they pushed (the gap changing by
  0.41 m a tick); on one shard they stand, hull to hull. Fixed in `space-sim/src/rigid.rs`
  (`meeting_velocities`): a **ship's** ghost that is about to touch one ship of this world is
  presented to the solver at the velocity the two share after an inelastic meeting of equal
  masses; the other shard does the same from the same two velocities, so no word between the
  shards is needed. Now: pressed together they stand 0.59 m apart (0.60 on one shard), the gap
  moving 1.6 mm a tick; one pushing the other moves off with it at 17.6 m/s after eight seconds
  where one shard gives 19.7 (they are on two shards, a tick behind each other, until the
  pusher is a metre across). Nothing changes for ghosts of bodies, or for a ghost about to
  touch several ships. On the live stack, two clients steering at each other from a gentle
  contact for six seconds: across the border the gap stayed 0.54-0.63 m in every one of 234
  samples (largest change between two samples 7 cm) while the pair drifted 7.7 m west and the
  pusher was handed over; the same two on one shard: 0.50-1.15 m (they are frictionless
  capsules and slide round each other). Corrections up to 1 cm for both players.
- **The tests were timed against themselves.** `seamless.rs` held whole-flight maxima over
  thousands of frames to limits taken against the test's own `Instant::now()` (5 cm at 15 m/s is
  3.3 ms of scheduling). Its bounds are now taken on the app's own frame instants
  (`DrawnShip::at`) and drawn ticks, with what a long frame explains taken off, **around each
  change of shard** (1.5 s before to 0.7 s after), and pass under a floor that is the size of
  defect each is there to catch or within three times the 99.5th percentile of the same flight
  away from any switch (module docs of the file). The latency test's two ticks of slack are
  now four on a median — a coarse net — and the exact check is the shards' record of applied
  inputs: every tick once, through the real send path, at every switch of every crossing test.
  The three-shard clock tests set the time instead of reading it. The file runs in 2 min 40 s
  instead of six to seven (one loop instead of two, two standers at once, two players in the
  border test, turn-rounds without a full stop); the target of under two minutes was not
  reached: the handoff's second and the three-second settle rule are most of what is left.
  Pinned to two logical CPUs each, two copies at once, with a release build and a live stack
  running beside them: 10 runs of `seamless.rs`, one failure — a premise of the test itself
  (a climb timed by the wall clock ended 20 cm above its upper bound; it now climbs by height)
  — then 10 of 10 for that test; `handoff.rs` 9 of 10 (a look that read the link twice could
  miss a handoff between the two; fixed), then 10 of 10; `crossing.rs` and `orchestrated.rs`
  10 of 10 each.
- **On the live stack** (clients at 30 frames a second this time, `/state` sampled 50-140
  times a second). 20 crossings at 8-16 m/s: the drawn ship was off from where the sample
  before puts it by 3.8 mm at the 99.5th percentile of the samples around a switch (5.5 mm
  away from one), corrections 0, 1.07-1.15 s from a metre past the border (median 1.11) —
  phase 3 measured 4.0 mm, 0 and 1.00-1.30 s. (The largest single step, 0.19 m, was the ship
  running into a floating crate of the zero-g corner 1.5 s before a switch: the shard's word
  changed by 9 m/s in one tick, and the prediction had it to 0.03 mm.) 10 crossings in fast
  travel (20-364 m/s): 0.36 m at the 99.5th percentile around a switch, 0.34 m away from one,
  the largest 0.44 m against 1.19 m elsewhere; corrections 0; 1.09-1.15 s. One shard killed by
  pid and replaced (new id, next port, scene re-issued): 10 crossings to and from the
  replacement, 6.5 mm at the 99.5th percentile (5.5 mm away from a switch), corrections 0,
  1.09-1.13 s. A shard stalled for three seconds in the middle of a handoff into it
  (`SPACE_DEBUG_STALL_DIR`, a hook for exactly this): the old shard committed on time, the
  picture stood still for two seconds, and the ship went on from **where its commit put it** —
  4.7 m behind where the client's prediction had run to, not 30 m ahead of it; the shard logged
  `could not keep up: it was 91 ticks behind ... jumped` and `90 ticks jumped, 0 caught up`.
  At the start of the cluster three body transfers were called off by an epoch change (the
  third and fourth shard were still joining); none after, and none of a player's.
- **The 16 cm correction of phase 4** is what running into a body costs: the client predicts
  its ship against *kinematic copies* of the bodies near it, so the first ticks of a collision
  are predicted against a body that does not give way. Ramming the zero-g corner's crates at
  2-4 m/s: four corrections over 3 cm (7-13 cm), each with a crate 0.4-0.6 m from the ship,
  none in free flight. By design; a dynamic copy would be the fix, and is not planned here.
- **The dev profile's optimised physics** (`rapier3d-f64`, `parry3d-f64` at 3, `space-sim` at
  2) is what the CI's tests use too — the `test` profile inherits `dev`, package overrides
  included. A clean build of `space-sim` with its dependencies: 36 s with them, 23 s with all
  three at 0, on this machine (32 threads, busy with two test runs at the time).
- **Not verified here.** `.config/nextest.toml`'s new `timing` group was not run: nextest is
  not installed on this machine (the `cargo test` fallback ran). The Linux job's second pass
  with `SPACE_SKIP_TIMING=1` was not run either. No wall was put in a stalled ship's way on
  the live stack: that nothing is brought forward through one rests on the test.

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

**Status: built (phase 5).** What it became, where it differs from the bullets above, and what
was measured. The code: `space-protocol/src/far.rs` (range, steps, draw delay, the two messages,
`Origin`), `space-server/src/far.rs` (the far half of a shard), `replicate.rs` (per-entity base and
steps), `space-client/src/{link,bodies}.rs`. Tests: `space-server/tests/far.rs` (eleven, on two
shards sixteen cells wide each) and unit tests beside the code; every one names its mutation and
each mutation was run (`target/phase5/mutate.py`, 33 of them; the first run let eight through —
tests that took their expectation from the function under test, or that a second line of defence
kept green — and the tests were tightened until none was left). Live numbers are from `dev.ps1 -ServersOnly
-PortOffset 300` (four shards, release build, two headless clients at 60 frames a second on the
harness, `/state` sampled 100 to 200 times a second, 2026-11-24; tracks and frames in
`target/dev/shots/phase5/`).

- **Who sends what: shard to shard, relayed by the player's own shard.** A client keeps exactly
  one play connection outside a handoff. Every 10 ticks (not once a second: a third of a second of
  fast travel is already a kilometre) a shard tells every other shard of its map where its players
  are — and where the ships are that are on their way to it, so a player's view is there before
  the player is. Each answers with what it simulates within 10 km of one of them and does not
  already send as a ghost. It rides the peer connections the handoff uses, behind the peer token:
  no new listener, no new credential. The alternative — a client connected to every shard in view
  — was not built: up to eight connections a client, shards that would have to admit players they
  do not serve, and one entity arriving on two connections with only the client to settle which
  counts.
- **Deviation: a far replica is not a `SimWorld` entity.** The plan put it into the world "with
  its components but no rapier body". It is in **no world**: the shard keeps it in the interest
  grid under a key of its own (`Interest::upsert_far`) and in the component log, which is all the
  replication path reads. Nothing can step it, collide with it, hand it over or hash it, by
  construction; `SimWorld` did not change.
- **Deviation: the receiver does not move a far replica between updates.** Dead reckoning a moon
  at one sample a second would put it some two metres off a six-metre orbit and snap it back
  (arithmetic, not a run: this was not built). A sample keeps **the tick
  it is the state of** — the simulating shard's, which is the cluster's — all the way to the
  client (`space_protocol::Origin`, a twelfth replicated component id, 11: what kind of copy this
  is, of which shard, of which tick), and the client interpolates on that timeline.
- **Distance steps** (the plan's numbers, plus a near range): every tick to 100 m, every 3rd to
  250 m (10 Hz), every 6th to 1 km (5 Hz), every 30th to 10 km (1 Hz); coarser only 10 % past a
  bound, finer at the bound (`far::step_after`). One table for both hops and for **every** entity
  — own, ghost and far — per client by the distance from that client's ship, between shards by the
  distance from the receiver's nearest player. `ClientReplica::sent` holds, per entity, the tick it
  was last sent and its step, in place of the one change base: an entity is due when it changed
  since then and its step's spacing has passed, so one that was skipped and then stopped is still
  brought up to date, and one that moves after a rest is sent **at once** (its first sample is the
  start of the motion). Nothing is sent for what did not change, on either hop. Beyond 10 km: not
  sent between shards (a member is dropped 10 % further out), and a client is **told** the removal
  like any entity that leaves its interest — no timeout is involved on that hop.
- **Bounded cost.** A client: at most `sum over the moving entities in view of (one sample x the
  rate of its step)` a second plus 56 B a tick, never more than the frame budget a tick. A peer
  link: the same sum at the step of the receiver's nearest player, plus 12 B per held replica
  every 2 s (the refresh: the keep-alive of a replica at rest) and the interest (16 B + 36 B a
  player, three times a second); a `FarUpdate` carries at most 32 KiB, what does not fit waits a
  tick. `tests/far.rs`: sixty balls 600 m off cost a client under `10 s x (60 x 130 B x 5 + 30 x
  80 B)` and the link under the like; four hundred appearing at once never make an update over
  the bound; a body at rest costs 44 B per refresh and no frame.
- **One entity, one picture: own > ghost > far, in place.** A guid is one of the three on a
  shard. A far update never touches a guid that is in the world; a guid that enters it (a ghost
  update names it, a handoff installs it) stops being a far replica in the same moment, and its
  log entry goes on, so the client gets the difference and keeps the history. A ghost its owner
  withdraws — it left the border margin — becomes a far replica where it stands
  (`Shard::ghost_to_far`), to be confirmed by its owner's next update or dropped 6 ticks later.
  **Found on the way:** an entity replaced by another kind of copy within one tick (a ghost
  installed as the shard's own ship) kept the old copy's components on clients — the log forgot
  the entity and with it what had to be listed as removed; a ship stayed "a ghost" for good.
  `ComponentLog::forget` now keeps what it had as removed for 128 ticks. And a withdrawn ghost of
  a shard no far update had ever come from was dropped by the silence rule at once (found by
  delivering the far updates three ticks late in `from_local_to_ghost_to_far_replica_without_a_gap`;
  the ghost frame that withdraws it now counts as that shard's word).
- **Smooth at low rates** (`bodies.rs`). A thing is drawn further in the past the further it is
  from the own ship (`far::draw_delay`: 2 ticks to 12 m, eased up to a step's spacing + 3.5 ticks
  from 50 m on — 4.5, 6.5, 9.5, 33.5 — continuous, and a function of the **distance alone**, so
  it does not step when a far replica becomes a ghost and then local). Samples more than a tick
  apart are joined by the cubic that leaves and arrives with the **velocities** they carry — a
  six-metre orbit at 0.8 rad/s sampled once a second is followed within 2 cm, where the straight
  line cuts 47 cm off it (`bodies.rs`' test) — unless the velocities do not explain the way between (what a planet carries; a
  bounce): then the straight line. **Found on the way, both on the live stack:** (a) the delay
  was first taken from the distance to the newest *sample*, which steps thirty times a second:
  everything coming nearer or going away juddered, 0.2 m to and fro at 40 m/s; it is taken to
  where the thing is drawn now (`draw_time`). (b) **Ghosts were a tick of motion off, to and
  fro.** A ghost was moved on one tick after every step, whatever tick its update was of; two
  shards step at the same instants, so the update of tick n is sometimes here before the step
  and sometimes after: a player flying past at 22 to 35 m/s leapt 0.75 m ahead and waited there,
  every second or two (0.2 to 0.6 m between two frames). A ghost now knows the tick its state is
  of and is moved on only while that is behind (`GhostEntry::shown`, `border::
  GHOST_TICK_TOLERANCE`); `a_ghost_shows_its_owners_state_of_the_tick_whoever_stepped_first`.
- **Players too.** A far player is a ship with the player model (`Visual(PLAYER)`) and nothing
  else of the player: `FAR_COMPONENTS` is `ShipState`, `Body`, `Visual`, `Springs`,
  `Articulation`, `PartStates` — no `Owner`, no `ShipControls`, no `PosResidual`, nothing private
  — applied by the sender and again by the receiver.
- **Shard death and map changes.** A shard that sent nothing for 3 s, or that the map no longer
  names, loses its far replicas here; scene bodies of another revision are never kept. A message
  to a peer can be dropped (its queue was full, the peer was being dialled): what a refresh names
  and the receiver lacks, it asks for in its next interest and is sent in full.
- **Tests whose premise changed.** `physics.rs`' crossing tests held that a client is told a body
  "left" when its ghost leaves the margin and that a moving body is seen on every tick at any
  distance: the client now keeps it as a far replica, and the watcher was moved to within 100 m
  (or, over QUIC, expects one sample in six at 500 m). `replicate.rs`'
  `a_change_made_while_a_snapshot_is_going_out_is_not_lost` moved the client's **own** ship, which
  is repeated in every frame until the first acknowledgement whatever the bases say: it could not
  fail. It moves another ship of the first chunk now, and fails under its mutation. The client's
  `a_resting_entity_holds_until_it_moves` held that 39 ticks between two samples of a moving
  entity start its history over (the limit was 15 ticks); that is one motion of something a
  kilometre off now, and the restart is at 76.
- **Security**: `docs/security-notes.md`, "Far replicas between shards".
- **Harness**: `/state` has `source` (`local`/`ghost`/`far`), `step` and `sample_age_s` per body
  and per other player (`others`), and `far_drawn`. The census line of a shard has `N far` and
  the traffic.

**Looked at on the live stack:**

- *From the spawn point*, both clients list the 35 bodies of the showcase, 35 distinct: 19
  `local`, 16 `ghost`, none `far` — and 35 on a `-MinShards 1` stack (all `local`). **Nothing of
  the showcase is a far replica**: it lies within 64 m of the borders; and a thing within 64 m of
  *any* face of its cell is a ghost on the root's shard as well (`border::territory_distance` is a
  lower bound around carved cells — a player 194 m inside the west cell but 4 m from the z = -4
  plane was a ghost on the root). Far replicas appear more than 64 m from both planes.
- *A far player*: B standing at `pos` -110 0 80 is `far` for A on the spawn point (133 m, step 1,
  sample age up to 0.17 s). B flying at up to 10.3 m/s (0.172 m a frame): its drawn position on A
  moved at most 0.179 m a frame, and the step changed by at most 14.5 mm from one frame to the
  next. `look2-far-player-first-person.png`: the character, 76 m off, twenty pixels tall.
- *A crate knocked away* (the zero-g corner's, rammed at 6.2 m/s along (-0.8, 0, 0.6)): on A it
  went `ghost` step 0 -> `ghost` step 1 at 110 m -> `far` at `pos` -143 3 70; in every one of
  2603 frames, 35 bodies listed throughout; where it turned far its drawn step changed by 0.8 mm
  (it moved 56 mm a frame), the largest change anywhere 7.4 mm.
- *Far -> ghost -> local.* B flying at 43 m/s (0.72 m a frame) from far inside its shard across
  x = -6 into A's: `far` -> `ghost` -> `local` on A, in all 578 frames; the step changed by 1.8 mm
  at far -> ghost, 9.3 mm at ghost -> local, 46 mm at most anywhere (before the two fixes above:
  0.2 to 0.6 m). And A flying to that crate, 266 m off on the other shard, and into it: `far`
  (step 2) -> `local` at A's own change of shard, in all 3635 frames, never two of it; rammed, it
  went from 0 to 4.6 m/s (the own ship's prediction was corrected by 16 cm, as it is for any
  ramming, section 5).
- *Fast travel out to 14.7 km and back.* Measured sample rates of a moon (a ghost) and a pendulum
  (local), the same for both: 29.0 a second within 100 m, 10.0 at 181 m, 5.0 at 465 m, 1.0 at
  3.7 km and at 6.9 km, nothing beyond. The step changed going out at 110 m, at 276 to 279 m and
  between 1.1 and 1.7 km (at fast-travel speed the place is not resolved better), coming in at
  972 to 980 m, 248 m and 100 m. Going out the 35 bodies were listed to 10.2 km and gone at
  10.4 km (the other player too); coming back 22 were there at 10.0 km, 34 at 9.6 km and the
  last — the one far replica among them — at 8.9 km: at some kilometres a second the far set is
  a third of a second of flight behind.
- *A shard killed* (the one B was on, by pid, its path checked): B gone from A's `others` 0.19 s
  later, the scene re-issued (for one sample the old and the new scene's bodies were listed
  together, 64 — the 0.3 s of section 2), 35 again at 0.49 s. B came back by itself after 12 s on
  the spawn point, flew out again and was `far` on A once, among 35 distinct bodies.
- *Traffic* (census lines, `SPACE_CENSUS_ALWAYS`). To the one client of the root's shard, by its
  distance from the showcase on that flight: 21.7 kB/s within 60 m (everything every tick; with
  two clients in the first minute of a stack, while the bowl's balls still roll, 42 to 50 kB/s
  each), 13.9 kB/s at 80 to 125 m, 11.3 at 130 to 240 m, 8.7 at 300 to 465 m, 6.1 at 1 to 5 km,
  and 5.4 kB/s beyond 10 km with nothing in view — the own ship every tick and the frames
  themselves. Between shards: 48 B/s a link from a shard without players (its empty interest),
  156 B/s a link from one with one player; the most one shard sent to its three peers together
  was 4.6 kB/s, the most one took in 4.2 kB/s — there were never more than two far replicas at
  once.
- *Regression.* Ten crossings of x = -6 at 21 to 33.5 m/s: the drawn ship off the frame before
  by 4.6 mm at the 99.5th percentile around a switch (4.2 mm away from one; 19.6 and 21.5 mm at
  most), corrections 0, 1.11 to 1.16 s from a metre past the border, 35 bodies and the other
  player listed throughout. `smoke.lua` passes. Three body transfers were called off at the start
  of the cluster by an epoch change, before any client (section 5); none after, on any shard; no
  shard logged a tick it could not keep up with or a far message it dropped; every client ended
  with 0 validation messages.

**Not done here, and what still looks wrong.**

- **The demo area has nothing between 64 m and 10 km** (the hold): every far replica looked at
  was a player or a crate knocked out of the showcase, at most 300 m off. Far replicas at
  kilometres — their rates, their bounds, their truthful ticks — rest on `tests/far.rs` (3 km, and
  11.5 km for the cut); the rates measured live at 0.5 to 7 km are of ghosts and local bodies,
  which go through the same per-client steps. Phase 7's far rock is what will show it.
- A player in fast travel outruns the interest (above). A thing in the eased band between 12 and
  50 m that comes nearer or goes away is drawn with its time running up to a tenth faster or
  slower at 40 m/s (the delay changes with the distance); it does not step.
- Orientation between sparse samples is a normalised lerp: something spinning faster than half a
  turn between two samples (a turn a second at a kilometre) is drawn turning the short way round.
- The bounds on what a peer sends are per peer, and one peer token is the whole cluster's
  (`docs/security-notes.md`).
- Not looked at on the live stack: a shard that stalls while it holds far replicas (the 3 s
  silence rule is a test), more than two far replicas at once (tests: 460), a scene switch with
  far replicas in view (test).

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

**Was still open after phase 3, closed in phase 4** (the status block below): bodies followed a
player across a border and back in numbers (section 5's last bullet); two shards refused each
other's body groups by the crossing rule now and then.

`space-server/tests/physics.rs` gets a four-shard fixture with the borders through the showcase:
islands whose members have three or four homes, ghost refresh between every pair, hysteresis at
the corner where four cells meet, a player walking through the room across both borders. What
breaks is fixed in `border.rs` / `shard_physics.rs` with the rules written down there.

**Status: built (phase 4).** The fixture is `space-server/tests/four_cells.rs` (a file of its
own, not `physics.rs`): the orchestrator's four-shard layout as four in-process shards on one
cluster clock, every scenario run on it and on one shard that owns all four cells, compared body
by body and tick by tick. Numbers below are from it (debug build) unless a live stack is named.

- **What was measured before anything was changed** (the same fixture on `f05f5aa`). Idle: one
  transfer called off at tick 11 of every start ("crossing island transfer"), the three balls of
  the bowl handed to and fro in the first second and ending up to 18 cm from where one shard has
  them, everything else bit-identical to one shard. Six passes of a ship 100 m above the showcase
  at 120 m/s: **446 handoffs**, 6 of them the ship's — 73 bodies a pass. Flying through the
  zero-g corner six times: 70 handoffs for 6 crossings. A player walking away from the gravity
  room: the crate stack **sank up to 30 cm into itself and the floor for three ticks and hopped
  back** — on four shards and on one.
- **Cause 1, the churn: what "may touch soon" meant.** An island was every body within
  `|v| x 1 s` of another *in any direction*: a ship at 120 m/s was one island with everything
  within 120 m, behind and beside it too, and since bodies follow a player the whole showcase
  changed shard with every pass and came back. Now two bodies are one island when their bounding
  spheres are within a metre of each other, or come that close within a second **on the way they
  are moving** — the closest approach of their relative motion (`space_sim::IslandQuery`,
  `rigid.rs::may_meet`; no new constant). A body changes shard because it moved, or because
  something that is about to touch it did; not because a player is near. The same six passes: 6
  handoffs, the ship's. What is left: the test is on bounding spheres, so a player who walks past
  a stack within about a metre and a half of its hull still takes it along (and it comes back
  three seconds after the player has gone).
- **Cause 2, transfers called off with nobody near.** (a) At every start: the balls rolling
  down the ramp, *away* from the bowl, were one island with the balls rolling into it across
  `z = -4`, the two shards computed different keys for one tick and offered each other its
  members. Gone with cause 1. (b) In general the two sides' knowledge of each other's island key
  is one tick old. Three rules now keep that from becoming two offers that cross
  (`border.rs`, "Islands and consolidation" and "Players' ships"): an island that touches a
  neighbour's bodies is offered only once its home has been the same for two passes
  (`CONFIRM_TICKS`); an island one of whose bodies is on its way *here* waits for it and goes on
  with it; and nothing follows a player **back** — while a player's ship is on its way to a
  shard (staged there, or a metre into its ground), that shard's bodies ahead of it wait where
  they are. The crossing rule itself (the lower shard id takes it) stays as the last resort; no
  test and no minute of the live run below made it fire. (c) A scene body the spawner puts on a
  neighbour's ground goes there at once and with the others, also when it lies within the 2 m
  hysteresis of the border: the hysteresis is for a body that has crossed one.
- **Cause 3, the sinking stack, was not a border at all** (`space-sim/src/rigid.rs`). When a
  bubble split, the group holding its *smallest guid* kept the solver's world — a player's ship
  has a lower guid than any scene body, so whenever a player left a group of resting bodies it
  was they who were moved into a fresh world; and a body inserted there **asleep** has no
  contacts for a few steps after something wakes it. Now the **largest** group keeps the bubble
  (nothing happens to it at all), and a body is never inserted asleep (a stack that does have to
  move re-settles within 6 mm instead of dropping 30 cm). Two tests in `space-sim/tests/physics.rs`.
- **What "the same as on one shard" is, measured.** Never handed over, or handed over in **free
  flight** (weightless, in orbit, falling; across `x = -6`, `z = -4`, through the corner, along
  a border, at 20 m/s): **bit-identical** at every tick. Handed over **in contact**: the solver's
  warm start stays behind, so the first tick on the new shard is solved cold — a crate sliding
  across the border on the floor stays within 0.7 mm of one shard's and within 0.02 mm of its
  height (no sink, no hop); crates at rest on the planetoid 0.04 mm; the three balls colliding in
  the bowl 9 mm; a stack brought to a passing player's shard and back 8 mm at its top crate; a
  crate a player pushes 2.7 m across the border 1.5 cm, the player 1.1 cm; a collision that is
  under way on the very ticks of a handoff 2 %. A spring whose ends are on two shards (across the
  corner, 20 s, a dozen swings): 0.04 mm, the swing neither grows nor dies. A pendulum and a
  chain swinging over the border do not change shard and are bit-identical. Resting against and
  on a ghost until consolidated (11 ticks): heights within 2 mm.
- **Not corrected, and pinned:** a commit that arrives late is brought forward along the body's
  velocity without gravity, so a body that was *falling* is short of 0.33 m/s per late tick —
  with two ticks it falls up to half a metre behind and lands a tick or two late, in the same
  place (`a_late_commit_does_not_sink_or_hop_a_falling_or_a_sliding_crate`). A body on the floor
  is unaffected. On one host commits were never late in the live run.
- **"A crate was knocked up and forward rather than pushed"** is the collider, not a border: the
  demo area floats that crate at head height and a ship is a capsule, so what strikes the crate
  is the capsule's rounded top, and a ship has no friction. The same flight on one shard sends
  it off the same way, to the bit when the ship has changed shard before the blow
  (`the_floating_crate_on_the_border_is_struck_the_same_on_one_shard`). Left as it is.
- **Dev default.** `scripts/dev.ps1` / `scripts/dev.sh` start **four shards** (`-MinShards` /
  `--min-shards` default 4; 1 = one shard). The orchestrator binary's own default (0) and the
  compose files are unchanged (the compose files until phase 6, section 8).
- **Build profile** (`Cargo.toml`, no dependency added): `rapier3d-f64` and `parry3d-f64` are
  built at opt-level 3 and `space-sim` at 2 in the dev profile. The four-shard tests step minutes
  of physics; 300 ticks of the fixture went from 6.7 s to 0.84 s. The new file's 14 tests take
  6 s together on this machine (25 s one at a time).
- **On a live stack** (`dev.ps1 -ServersOnly -PortOffset 300`, four shards by default, release
  build, two headless clients on the harness; json, frames and outputs in
  `target/dev/shots/phase4/`, 2026-11-24). Start: 16 bodies handed to their shards, **0 called
  off** (2-3 before). Two minutes idle: no counter moved; only the moons, the pendulums, the
  chain, the spring pair and the ramp's balls moved more than a millimetre. Walking into the
  crate stack and on over `z = -4`: the lower three crates crossed still stacked, 4.1 m; a
  second client on the far side's shard saw every crate in every sample, its view of them
  within 15 cm of the first's while they moved (the two reads are not simultaneous) and
  identical at rest. The floating crate on the border struck, the spring pair pushed 30 m
  across `x = -6`, the door swung, a crate pushed past the four-cell corner (0.4 m from the
  line) into the third shard: views within 7-10 cm in motion, medians under 1 cm, no sample
  missing, 0 called off. Ten fast-travel passes at 245-274 m/s 150 m up: 10 handoffs, the
  ship's ten. Five minutes of two clients flying through everything (38 legs, 24 changes of
  shard): 103 handoffs, 79 of them bodies (16 of those the scene being rebuilt just before),
  **0 called off** — against 659 handoffs and 197 called off in the half hour of phase 3. The
  shard that owns the bowl end killed by pid: replaced under a new id on the next port, the
  scene re-issued once, both clients then listing 35 bodies, 35 distinct, all of the new
  revision. `smoke.lua`, `physics.lua` (unchanged; 97 s) and `moon.lua` pass against the
  four-shard stack. In the whole session no shard logged a transfer called off or a tick it
  could not keep up with, and every client ended with 0 validation messages.
- **Not done here, and what still looks wrong.** Bodies knocked further than the ghost margin
  from a border are drawn only by the clients of the shard that simulates them (two crates
  325 m out were missing for a client one shard over): section 6. The reach of an island is
  measured on bounding spheres (above). During the second of a player's handoff the ship meets
  bodies that came within reach only then as body against ghost. The falling body's late commit
  (above). A crossing island transfer is still possible in principle and was seen once on the
  live stack *before* the last of the three rules was written (a ball offered towards a ship
  that was on its way back), not since. The harness clients' prediction was 16 cm off the
  shard's word once in the five minutes of ramming things (`correction_max_m`); where was not
  looked at. (Looked at in the review of phase 3, section 5: a ship that runs into a body is
  corrected, by design — the client predicts against a kinematic copy of the body.)

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

**Status: built (phase 6).** What it became, where it differs from the bullets above, and what
was run. The code: `space-orchestrator/src/launcher/docker.rs` (`--launcher docker`), the two
compose files, `deploy/Dockerfile.orchestrator`, `deploy/Dockerfile.shard`,
`.github/workflows/release-images.yml`.

- **The client.** Hand-rolled HTTP/1.1 over the Engine's unix socket, one connection per
  request, JSON through `serde_json` (already in the workspace): **no HTTP crate, no Docker
  crate**. Linux only: on Windows the orchestrator refuses to start with `--launcher docker`,
  with a sentence that says to use `--launcher process`. The client is generic over its
  transport, so its 19 tests run against a daemon of their own, in memory — on Windows too.
  Every call is bounded (10 s; 15 s a remove; 300 s a pull), and an error is a failed launch
  that feeds the back-off of section 2.
- **Flags** (each with `SPACE_ORCHESTRATOR_...`): `--docker-socket` (`_DOCKER_SOCKET`, default
  `/var/run/docker.sock`), `--shard-image` (`_SHARD_IMAGE`, **no default in the binary**; the
  orchestrator image has it baked in), `--docker-network` (`_DOCKER_NETWORK`),
  `--docker-control-addr` (`_DOCKER_CONTROL_ADDR`), `--docker-publish-ip` (`_DOCKER_PUBLISH_IP`,
  default `0.0.0.0`), `--docker-name-prefix` (`_DOCKER_NAME_PREFIX`, default `space-shard-`;
  a container's name is the prefix and its host port).
- **A shard container**: created and started from the image with `Entrypoint ["space-server"]`
  and `--orchestrator <control> --cluster <id> --bind 0.0.0.0:P --advertise 0.0.0.0:P
  --client-endpoint 0.0.0.0:P`, P one host port of `--host-port-range`, the same number inside
  and outside, `P/udp` published on the publish IP. The cluster secret is in the container's
  **environment** (`SPACE_CLUSTER_SECRET`), never in its command, never logged (errors are
  scrubbed of it). Labels `space.cluster=<cluster>`, `space.role=shard`, `space.launch=<n>`,
  `space.port=<P>` (**deviation:** not the shard's id — a shard is given its id at the hello,
  after the container exists; the launch is matched to the registering shard by address, as
  the `process` launcher's is: container IP and P, the shard advertises `0.0.0.0:P` and the
  orchestrator fills in the IP it sees). Restart policy `no`, `CapDrop: ALL`,
  `no-new-privileges`, not privileged, **no mounts, never the socket**.
- **Deviation: not "removed when it exits".** No `AutoRemove`: an exited container keeps its
  exit status for the orchestrator's log line, and is removed by the orchestrator (`DELETE
  ...?force=true`) — on scale-in, on replacement, as surplus.
- **Deviation: the orchestrator's own IP, not a name, is the default control address** (its
  container's address on the network + the control port's port). The network is found by
  inspecting itself through `$HOSTNAME`; a container on several networks with none named is
  refused. A `name:port` works too: `space-server --orchestrator` now also takes a host name,
  resolved once at start (`the_orchestrator_is_named_by_address_or_by_name`, in `space-server`'s
  `main.rs`; the docker flags have their tests in the orchestrator's).
- **Leftovers are removed, not adopted.** At start every container that carries this cluster's
  shard labels and that an earlier run left is removed: a shard cannot join a new orchestrator
  (it stops when it loses its own, ids are never reused), so a leftover is dead or about to be,
  and holds a host port. A name that is taken is freed only when its container carries both
  labels; a stranger's container of that name is left alone and the launch fails.
- **SIGTERM / Ctrl-C.** The orchestrator leaves its loop and the launcher force-removes
  everything it started under one 20 s bound, then the process exits 0 (compose:
  `stop_grace_period: 30s`). **One new crate in the lockfile for it: `signal-hook-registry`**
  (pure Rust, on `libc`, Unix only), through tokio's `signal` feature — std has no signal API
  and the crate forbids `unsafe`. Should the orchestrator be killed instead, its shards notice
  the lost link after QUIC's idle timeout (about 10 s), exit non-zero and stay as exited
  containers until the next start removes them. The `process` launcher got `exit_code` and
  `shutdown` too; `dev.ps1` / `dev.sh` are unchanged and keep `--launcher process`: Docker is
  not needed for the dev loop.
- **Deviation: no registry credential is handled.** An image that is absent is pulled (`POST
  /images/create`); a private image has to be pulled on the host by the operator (`docker
  login` + `docker pull`) before the orchestrator starts.
- **Compose.** `deploy/docker-compose.yaml` and `deploy/proxy/docker-compose.yaml` hold the
  orchestrator as the whole cluster (and the `client` stream service: opt-in locally, kept in
  production — whether it is needed there is an open question for the owner): the socket
  mounted, `group_add: DOCKER_GID` (the orchestrator is not root in its container; local
  default 0, right for Docker Desktop; required in production), the control port not published,
  the player port 7777/udp published, the shards' ports 7870-7899/udp published by the shard
  containers themselves, `SPACE_CLUSTER_SECRET` required with no default in both, `--launcher
  docker --min-shards 4 --max-shards 8` through the environment, the web UI off in production
  and on loopback behind a token locally, as before. The fixed container IPs are gone. The
  `client` service dials the player port **through the host** (`host.docker.internal`): one
  that dialed the orchestrator's in-network address would be sent on to `<that address>:<shard
  port>`, where no shard listens.
- **Images and the order of a release.** `Dockerfile.orchestrator` takes `ARG
  SPACE_SHARD_IMAGE` into `SPACE_ORCHESTRATOR_SHARD_IMAGE`; `release-images.yml` builds and
  pushes the three `sha-<commit>` images in the matrix job (the orchestrator's with
  `SPACE_SHARD_IMAGE=.../space-shard:sha-<commit>`) and moves `:latest` in a second job,
  `latest`, that needs all three: shard, client, **the orchestrator last**. Watchtower updates
  the orchestrator only: the old container gets SIGTERM and removes its shards, the new one
  removes leftovers and starts four shards from its commit's tag. Players of the old shards
  reconnect through the player port, as after any shard death (about 10 s, back on the spawn
  point).
- **Security**: `docs/security-notes.md`, "The Docker socket" — the orchestrator's container
  is root-equivalent on the host.
- **Executed** on Windows 11 with Docker Desktop 28.5.1 (Linux engine in WSL2), 2026-11-24:
  `deploy/docker-compose.yaml` as it is, with a project name, ports moved by environment (player
  port 8077, shards 8170-8199, web UI 8280), a generated throwaway `SPACE_CLUSTER_SECRET`,
  `DOCKER_GID=0`, lowered autoscaler thresholds (capacity 2, out after 5 s, in after 15 s,
  cooldown 10 s) and local image tags. Compose started the orchestrator alone and it created
  four shard containers (the first registered 0.9 s after the orchestrator started, all four
  owned their regions after 3.5 s); a headless client on the host joined through
  `127.0.0.1:8077` and was sent to a shard container's published port (35 bodies listed, frames
  looked at); four flying crossings between two shard containers at 15-40 m/s: drawn step
  3.1-4.6 mm, corrections 0; `docker kill` of one shard container: noticed and replaced under a
  new id in a new container within about a second (lost at tick 2385, the replacement's epoch
  committed at tick 2416), the scene re-issued once; scale-out: two players in the spawn cell,
  a fifth container, then both in the next cell, a sixth (0.9 s from "wanted" to registered);
  scale-in: the far cell's shard merged, drained in 1.1 s, `exit status 0`, container removed;
  `docker compose stop` (SIGTERM): all five shard containers removed and the orchestrator
  exited 0 in under 5 s; `down -v` left no container, network or volume. The cluster secret
  appears in none of the captured logs. That long scenario ran on images built from a
  throwaway Dockerfile (a single build stage with a cache) with the same runtime stages and the
  same binaries; the images from `deploy/Dockerfile.orchestrator` and `deploy/Dockerfile.shard`
  themselves were built and run through a start, a client's join, a killed orchestrator with
  the leftovers removed at the next start, and `compose down`.
- **Only parsed** (`docker compose config`): `deploy/proxy/docker-compose.yaml`, with
  `.env.example` plus a throwaway secret and `DOCKER_GID`.
- **Not verified.** Anything on a Linux host (a socket group other than 0, a firewall, real
  published ports to another machine); Watchtower replacing the orchestrator; a pull of the
  `sha-` tagged shard image from ghcr; a private registry; the `client` stream service in
  either compose file; `Dockerfile.client` (unchanged); `release-images.yml` (YAML-checked
  with `.github/scripts/check_yaml.py` only).
- **Not done here, and what still looks wrong.** No socket proxy restricts the Engine API to
  the calls the launcher needs; the shard image's name is trusted; shard ports are published on
  every interface (all in `docs/security-notes.md`). **Seen on Docker Desktop:** in the WSL2 VM
  the shards' census lines showed a `clock offset` between 0 and **-84 ms** (the hypervisor's
  time sync steps the VM's wall clock; a process stack on the same machine shows 0.0 ms
  throughout). All shards share that clock, but each slews by itself, so for seconds at a time
  two shards were a tick or two apart: two of the later crossings there showed a drawn step of
  0.40 m and 0.76 m and corrections of 0.05 to 1.2 m (the four measured right after the start:
  3.1-4.6 mm, 0). Not a defect of the launcher: it is section 5's limit — a wall clock that is
  stepped — made visible by that VM. Not verified on a real Linux host, where one kernel clock
  serves every container.

## 9. The wire (PROTOCOL_VERSION 6 -> 7)

`ShardRegion` + client endpoint. `LoadReport` + entity count, hot cell. `ShardHello` + secret.
`ShardWelcome` + tick origin. `Snapshot` + `more` (added by the review of phase 3: a snapshot in
several chunks). Peer hellos carry the peer token in the existing token field.
Appended variants: `JoinRedirect`, `FarInterest`, `FarUpdate`, `ShardRetire` (added in phase 6:
the orderly end of a drained shard; `VARIANT_COUNT` 33). The handoff
messages, the epoch messages and `Hello` keep their layouts.

**Status.** This plan was written as "5 -> 6". While the branch was being built `main` shipped a
protocol 6 of its own - the merged demo area: `SceneClock` (tag 28), `PlayerInput::fast`, 25-byte
`ShipControls` - and that 6 stands as released. Everything of this branch is therefore
**protocol 7**, on top of it: built so far are `ShardHello` + optional client endpoint + secret,
`ShardRegion` + client endpoint, `RejectReason::Starting`, the peer token in a shard's hello and
`JoinRedirect` at tag 29, since phase 3 `ShardWelcome` + tick origin
(`tick_origin_us`, its last field), since its review `Snapshot` + `more` (its last field:
"further chunks of this snapshot follow"), and since phase 5 `FarInterest` (tag 30: `from`, `tick`,
`players`, `missing`) and `FarUpdate` (tag 31: `from`, `tick`, `new`, `changed`, `removed`,
`refresh`, `held`, `more`) — `VARIANT_COUNT` 32 — and one more replicated component id, 11, the
`Origin` a shard adds to what it tells clients about a ghost or a far replica (14 bytes: version,
kind, shard, tick; a client that does not know the id skips it). Since phase 6 `LoadReport` ends
with `entities`, `tick_mean_ms`, `tick_max_ms`, `skipped`, `hot_cell` (`[i64; 3]`) and
`hot_cell_players` (its last field; the message has a `Default`), and `ShardRetire { shard,
reason }` is tag 32 — `VARIANT_COUNT` 33. All of it stays inside 7 as long as 7 has not been
pushed to `main`; once it has, the next layout change is 8. (The handoff state's private components are not wire layout:
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
