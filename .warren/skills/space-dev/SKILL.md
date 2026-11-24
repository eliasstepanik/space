---
name: space-dev
description: Use when developing, testing, packaging or debugging the space engine (the Rust workspace with space-world/ecs/protocol/sim/net/server/orchestrator/render/client/harness/stream/viewer/mcp crates) - building, running CI, GPU/lavapipe tests, wire-protocol or handoff changes, the orchestrator web UI, scenes, Docker/CI files, the release pipeline, or driving a shard/client through the MCP servers. For looking at the running game use the space-visual-test skill.
---

# space-dev

Rust workspace, toolchain pinned in `rust-toolchain.toml` (1.100.0 + rustfmt, clippy,
`wasm32-unknown-unknown`). All crates are 0.1.0. Rules and workflow: `AGENT.md` (binding).
Visual testing: `.warren/skills/space-visual-test/SKILL.md`. Docs: `docs/plan.md` (phases),
`docs/plan-phase3.md`, `docs/plan-sharding.md` (seamless sharding: the map, the anchor, what comes next),
`docs/security-notes.md`, `spikes/ENVIRONMENT.md` (GPU env),
`deploy/README.md` (pipeline, web UI, what is/isn't verified).

**Three facts first** (details in `AGENT.md`): the client is **online only**; the **scene is
server state** owned by the orchestrator; **every push to `main` is a release and a deploy**.

## Layout
| crate | role |
|---|---|
| `space-world` | i64-cell + f32 position model (1024 m cells), octree, deterministic galaxy, LOD selector |
| `space-ecs` | custom archetype ECS |
| `space-protocol` | wire messages, codec, `PROTOCOL_VERSION`, `Message::VARIANT_COUNT`; `scene.rs` = the scene catalogue (`Scene`, `SceneId`, `SceneState`; id 2 `solar` retired: `Scene::RETIRED_IDS`, resolves to the default) and the **scene clock** (`SceneClock`: a shard's tick -> UTC µs mapping, exactly 1e6/30 µs per tick; `SceneClock::cluster(origin)` = the cluster's clock, tick 0 at the tick origin, `DEFAULT_ORIGIN_US` = the origin of whoever is given none); `ShardWelcome::tick_origin_us`; `far.rs` = seeing across shards: the view range (`FAR_RANGE` 10 km), the **distance steps** (`STEPS`: every tick to 100 m, every 3/6/30 ticks to 250 m/1 km/10 km; `step_after` = with hysteresis, coarser only 10 % past a bound), `draw_delay` (how far in the past a client draws something at a distance: 2 ticks to 12 m, a step's spacing + 4 beyond 50 m - 5, 7, 10, 34 ticks - eased, continuous), the shard-to-shard messages `FarInterest`/`FarUpdate` and `Origin` (replicated component id 11: "this copy is a ghost / a far replica of that shard's entity, its state is of that tick") |
| `space-sim` | deterministic simulation, `SimWorld`: ship flight (`physics.rs`), rigid bodies on `rapier3d-f64` in floating-origin **bubbles** (`rigid.rs`: merge/split/re-anchor rules - on a split the largest group keeps the solver world - statics, islands by closest approach, ghosts), components `Body`/`Visual`/`Springs`/`Articulation`/`PartStates`/`PosResidual` (ids 4-10, versioned wire forms), gravity zones + large bodies (`gravity.rs`; a `LargeBody` has a `reach`: the pull fades over its outer quarter and is exactly 0 beyond, `f64::INFINITY` = no limit; `BodyPlace::Orbit` = a fixed orbit of **scene time** (`motion.rs`: `OrbitPath` = sum of circles, phase from `t mod period` in integers); inside its reach an orbiting body **carries** what is near it by its displacement over the tick, weighted like the pull (`GravityField::carry`, step 0 of `rigid.rs`'s tick; no fictitious force); `StaticCollider::motion` = a collider teleported along the same path each tick; `SimWorld::set_scene_clock`/`scene_time`), orbits on rails, prediction (`PredictedShip`, `PredictedPhysics`). Ships: `ShipParams::lift` (1.0) = lift engines, the part of the input that pushes against the local pull gets `lift × |g|` more (`physics::lift_accel`, any direction, nothing in weightless space, so free flight is bit-identical); `Body::ship` has friction 0 (half the floor's by the average rule), so a landed ship can walk. **Fast travel** (`ShipControls::fast`, persistent, from `PlayerInput::fast`): speed limit `clamp((surface distance − clearance) / 1 s, 120 m/s, 1e10 m/s)` (`physics::fast_travel_limit`, distance to the nearest large body, zone or static; clearance `FAST_CLEARANCE` = 2 km while closing in on it, 0 while moving away), velocity follows `input × limit` with a 0.5 s time constant (`fast_velocity_step`); switching it off clamps the ship to 120 m/s at once |
| `space-scene` | the physical definition of every scene, shared by shards and clients: statics (the demo area's stations with the client's numbers, the Kestrel collider read from `kestrel_lod1.glb`), the demo area's physics showcase (the gravity room, whose curbs stop balls but not ships; the zero-g corner; the planetoid at `demo::PLANETOID_CENTER`: a 4 m rock with large-body gravity, 4 m/s² on its surface, a 16 m reach, crates resting on it and three balls in orbit), the demo area's **far content** (`demo.rs`, "Far content": one thing in each of the three cells beside the spawn point's, more than 64 m + 3 s of motion from every face of its cell and inside one update step's band of distances from the spawn point — the drifters at `DRIFT_CENTER` 171 m out, a spring pair and three tumbling crates; the far rock at `FAR_ROCK_CENTER` 706 m out and the giant rock at `GIANT_ROCK_CENTER` 1.23 km out, static balls of 16 m and 40 m with large-body gravity and three moons each; 11 bodies appended to the showcase's 35, so a scene body's index and guid stay what they were; nothing in it touches anything, so its handoff is exact), the demo area's **sky** (`solar.rs`: Earth-centred frame, Earth fixed at `EARTH_CENTER` = 8 Earth radii from the spawn point; Sun, planets and Moon on circular Kepler orbits sped up `SPEEDUP` = 8766x; reaches Earth 6 R, others 10 R; real `GM`; moving ball colliders), gravity, the shape catalogue, the `visual` ids and names, `scene_anchor` (where a scene is in the map: the centre of a one-cell root, else the root's centre corner + `ANCHOR_OFFSET` = 6 m, 512 m up, 4 m) |
| `space-net` | quinn QUIC transport (self-signed rustls, `ring`), hello/handshake (a version mismatch is answered with a `Reject`, not a dropped connection; `join` = a player's hello, answered with a `Welcome` or a `JoinRedirect` (`Joined::Elsewhere`), `connect` = a shard's dial, for which a redirect is an error; `Conn::turn_away` answers a hello with one last message and closes once it was read - the write gives up after `LAST_WORD_TIMEOUT` = 1 s, so a peer without a receive window holds nothing; the **first frame** of a connection is read with a hello's size limit, `MAX_HELLO_PAYLOAD` = 4 KiB, not the general 1 MiB, and a shared secret that has to fit one is at most `MAX_SECRET_LEN` = 1024); `secret.rs` = the one constant-time compare every shared secret uses and `Secret`, the type every holder of one keeps it in (its `Debug` prints `<redacted>`); `log_budget.rs` = `LogBudget`, a few log lines per window for what a stranger can cause |
| `space-server` | authoritative shard (`space-server` bin), `clock.rs` = the **cluster's tick clock** (`TickClock`; `clock::run` = the tick loop of both binaries' shards: tick `n` is due at `origin + n/30 s`; the shard is placed on that timeline by the wall clock at start (an origin in the future is refused) and **kept on it**: every second the tick clock is compared with the wall clock (`ShardStats::clock_offset_us`, in the census log line), an offset over 2 ms is slewed away at 0.5 % at most, one of more than 0.5 s behind is stepped over at once; late ticks are made up — 8 at most — and a longer stall is jumped over, `ShardStats::skipped` (`Shard::skip_to`: nothing moves for the skipped ticks but the orbiting bodies and what they carry; inputs for later ticks are kept); `TimeSource` = the two clocks a `TickClock` reads (`SystemTime`, or `ManualTime` in tests), `Pacer` = the loop without the waiting, which tests drive; `Shard::start_clock` puts a shard on it and makes it the scene clock its clients are told, behind `SceneUpdate` on join and with every re-announcement; the wall-clock re-anchoring it replaced, `Shard::sync_scene_clock`, is gone); `handoff.rs` seamless handoff (ships and unowned bodies, island groups; the state carries its scene time as the private component `SCENE_TIME_COMPONENT` — a ninth byte says "read off a shared clock" — and the receiver carries it to its own clock; **on shared clocks that time is the tick the entity is installed on**, a late commit is brought forward — 15 ticks at most, checked when the commit arrives **and** when the entity is installed (after a stall nothing is brought forward) — `ShardStats::caught_up`, and the inputs a player sends to both shards are kept for it, also the ones that arrive before the player's status (`SimWorld::applied_inputs` = each shard's record of the ticks it applied an input on); a player is sent the `Redirect` only once the peer has accepted, and only the ship's owner is; `Shard::recent_aborts` and budgeted log lines say which transfer was called off and why); `orchestrated.rs` = an orchestrated shard's life: it registers **without id or region** and is given both, asks again while told "not yet" (`Reject`/`Draining`), ends on any other refusal, **stops when it loses the orchestrator** (the process exits non-zero), and stops in an orderly way — **exit 0**, `space-server: shard N stops: the orchestrator retired it (…)` — when it is sent `ShardRetire`, which the orchestrator does once a shard whose region was merged away has handed everything over; `--orchestrator` takes `ip:port` or `name:port` (resolved once at start); `load.rs` = the **load report** an orchestrated shard sends once a second (`LoadWindow`, pure: players, entities = the ships and bodies it simulates, tick time mean / p95 by nearest rank / worst over the second, measured around `Shard::tick`, ticks jumped, the **hot cell** = the depth-64 cell holding most of its players, ties to the smallest index; `Shard::report_load` switches it on, a standalone shard reports nothing); a shard goes on hearing a shard that left the map for `FORMER_SHARD_GRACE_TICKS` = 60 s (`Shard::note_departing`, `knows_shard`, `peer_endpoint`): that is what lets a merged shard hand its entities over; `net.rs` = the front end, incl. the cluster's peer token (`shard:<id>:<token>` in a peer hello, refused without it when the shard holds one); holds and relays the scene (`--scene` standalone only) and simulates it (`shard_physics.rs`: the scene's bodies, spawned once per cluster by the shallowest region's shard; free spawn points, on the shard's own ground only around the scene's spawn point; ghosts); **joining** (`Shard::join` -> `Welcome` or `JoinRefusal`): a fresh join is welcomed only by the shard that owns the spawn point, every other shard answers `JoinRedirect` to it (`Starting` while its address is unknown) - a player whose ship is being handed over (owner token of a staged transfer) is welcomed anywhere; `--client-endpoint` (orchestrated only) = where its players dial it; `border.rs` = the cross-shard physics rules (ghost margin 64 m + 3 s of motion, islands = bounding spheres within 2 x 0.5 m now or within 1 s on the way they are moving, consolidation onto the home of an island's lowest guid; hysteresis: a body moves 2 m into a region or after 3 s in it, a **ship 1 m** (`SHIP_HYSTERESIS`, measured exactly round a corner of four cells: `near_territory_distance`) or after 3 s, an island that was in a group waits until it has wanted the move for 3 s, doubling up to 48 s for a body that keeps coming back; **a player's ship is handed over for where it is only**: no island moves it, in an island's key it counts with the shard that simulates it, the bodies of an island it is the key of travel with it in its group, and it waits for none of them; a player's ship carries `Visual(PLAYER)`, ghosts included); `replicate.rs` = every body in interest, changed components only in deltas (`ComponentLog`; a forgotten entity's components stay as removed for 128 ticks, so a copy replaced within a tick loses what the new one lacks), **at the rate of its distance step per client** (`ClientReplica::sent`: per entity the tick it was last sent and its step - no single change base; due = changed since then and the step's spacing passed, so a thing that moves after a rest is sent at once); `far.rs` (a child module of `shard`, like the physics half) = **far replicas**: what the other shards simulate in view of this shard's players, held in the interest grid and the component log only - never in the simulation - and what this shard sends them of its own (`far_pass`: `FarInterest` every 10 ticks, `FarUpdate`s at the step of the receiver's nearest player, a refresh of names every 2 s, 3 s silence rule); one guid is own, ghost or far here, in that precedence, and changes in place (`ghost_to_far`, `FarState::claimed_by_the_world`); the census line says `N far` and the traffic (`far N B/s out, N B/s in, N B/s to N clients`; `SPACE_CENSUS_ALWAYS` = every 5 s regardless) |
| `space-orchestrator` | epoch-versioned shard map (`topology.rs`: `carve_plan` carves any node inside a region for a new owner, down to one cell; the octant split is that with a size floor; a merge gives a region back to the nearest region around it; `epoch.rs`: every shard of the map is sent and must ack a change), `Orchestrator::begin_carve` (asks for a node; nothing is proposed before a shard registers to take it), the **autoscaler** (`autoscale.rs`, a pure function of loads, tick and prior state; `ScaleConfig`, every number a flag; load = max(players / capacity, tick p95 / 33.3 ms); out: a loaded shard's **hot cell** is carved for a new shard; in: a shard that is cold *together with the one it would merge into* is merged, then **drains** — `orchestrator.rs` "Scaling": `shard_loads`, `scale_out`, `begin_drain`, `drain_pass`, `drain_ended`: the shard owns nothing after the commit, hands over what it holds by ordinary handoffs, role `draining`, and is sent `ShardRetire` when a report from after the commit says 0 players and 0 entities, or stopped as it is after `--drain-timeout`), ShardId counter; the **shard lifecycle** (`orchestrator.rs`: a region is given at the hello - the root, a dead shard's regions, a wanted node, else the next layout cell of `topology::layout_cells` - through an epoch proposed with the shard already connected; `--min-shards`/`--max-shards` and the reconciler, which also stops launches that are no longer needed; supervision: a dead shard - its process gone, its link closed, or three map changes in a row unacknowledged - is replaced under a new id and the scene revision raised); launchers `launcher/{local,process,docker}.rs` (`local` = inside this process, `process` = one `space-server` process per shard, `docker` = one container per shard through the Docker Engine's unix socket: hand-rolled HTTP/1.1, JSON through `serde_json`, no HTTP or Docker crate, Linux only — elsewhere the orchestrator refuses to start with it — tests against an in-memory fake daemon that run everywhere; every launcher has `exit_code` and `shutdown`: on SIGTERM / Ctrl-C the orchestrator leaves its loop and stops what it launched, then exits 0); cluster secret + peer token (`control.rs`, `config.rs`); `control.rs` is also the control port's front end: a connection is read only after its hello was welcomed, and then only about the shard it was welcomed as (a player's hello there is refused in words); **the player port** (`player.rs`, `--player-bind`, off unless given): the one address players type - a hello is answered with `JoinRedirect { owner of the spawn point, published map }` or `Reject(Starting)` and the connection closed; it only reads the answer the run loop publishes (`Orchestrator::player_answer`); each shard's **client endpoint** (what its hello asks for, else its endpoint) goes into the map, an internal endpoint without an IP is filled in from the control connection (`control::as_seen_from`); owns the scene (`scene.rs`, persisted in `<state>.scene`); operator path `ops.rs`/`snapshot.rs`/`events.rs`; built-in web UI `webui/` (hand-rolled HTTP, server-rendered HTML, **no JavaScript**) |
| `space-render` | ash/Vulkan renderer, naga WGSL->SPIR-V in `build.rs`, ray-query GI and shadows, presets, reversed-Z, `mesh.rs` (GLB models, in the TLAS), `texture.rs` (PNG/JPEG decode, procedural textures), `overlay.rs` (2D rects + built-in 5x7 font), `environment.rs`, `swapchain.rs` |
| `space-client` | the client, one core (`app.rs`) with three front ends: window (`window.rs`), `--headless` and `--stream` (`headless.rs`); `connect.rs`/`address.rs` (connect screen), `input.rs`, `ui.rs` (HUD), `scenes.rs` + `demo_area.rs` (what each server scene looks like; the physics showcase's statics drawn from `space-scene`'s definition), `link.rs` (player link: `ServerLink::join` follows up to `MAX_JOIN_REDIRECTS` = 2 join redirects from the **entry** address and keeps it, `shard_address()` = where it plays; scene, shard transitions — **one timeline across the switch**: while a handoff is in flight every input goes to both shards, every tick gets exactly one input (`inputs_for`), and at the switch the tick, the sent inputs and every entity's history carry on, merged by guid; `Replica::tick_epoch` goes up only onto a shard that counts other ticks — replicated bodies merged per protocol 5), `bodies.rs` + `meshes.rs` (bodies drawn from their collision shapes, interpolated between ticks - a thing further off is drawn further in the past (`bodies::draw_time`, by the distance to where it is drawn) and samples more than a tick apart are joined along their velocities (`along`) - springs as coils, joints as pins; `link::Source` = which copy an entity is, a far replica's samples carry the tick its `Origin` names; a far replica is never copied into the prediction), `predict.rs` (the own ship predicted through `PredictedPhysics` with the scene's statics, gravity, nearby bodies, the shard's scene clock and its word on fast travel; the draw clock; both keyed by ship and `tick_epoch`, not by shard or session, so they run on through a shard transition; the shard's word on fast travel is taken **after** the reconcile with the state it came with - before it, leaving fast travel at speed is a tick of fast flight of false correction; what an input that arrives late changes in the frame being drawn is carried in the visual offset, up to 32 m, and an offset over 2 m never decays by more than the picture has come forward since the last frame (`last_drawn`; not speed x wall time: the tick drawn does not advance by exactly the frame's wall time), so the drawn ship is never pulled back; `landed` is the prediction's: the ship touches something or is asleep, and there is a pull); `App::drawn_ship`/`drawn_others`/`drawn_bodies` = what the last frame drew (tests, and the harness state's `drawn_position`/`drawn_velocity`/`drawn_tick`/`others_drawn`/`others`/`far_drawn`, and `source`/`step`/`sample_age_s` per body); the demo area's sky drawn every frame at the own ship's draw tick (`scenes::SkyTime`, `World::sky`, interpolated between ticks), sunlight from the Sun as the 8th light |
| `space-harness` | drive a client without a human: `Control` trait + `Hub`, loopback HTTP API (`http.rs`), sandboxed Lua scripts (`script.rs`, interpreter `lunita`), `examples/fake.rs` |
| `space-stream` | JPEG frame + viewer input wire format (keys, typed text, turns); feature `net` = ws:// server (tokio-tungstenite, no TLS), feature `jpeg` = encoder only |
| `space-viewer` | wasm cdylib browser viewer (canvas + WebSocket); a local dev tool, never deployed |
| `space-mcp` | hand-rolled JSON-RPC MCP core: `--stdio` or `--http addr` (HTTP/1.1 + SSE on tokio) |
| `space-mcp-server` | MCP server booting an in-process shard |
| `space-mcp-client` | MCP server acting as a player + offscreen frame capture |
| `xtask` | `ci`, `lint`, `gpu-setup`, `viewer` (build + serve the browser viewer); `e2e/load/parity/web-smoke/accept` are stubs (exit 2) |

## CI (run before declaring done; network is fine, no `--offline`)
```sh
TMPDIR=/workspace/.warren/tmp RUSTUP_HOME=/workspace/.warren/rustup RUSTUP_TOOLCHAIN=stable \
  cargo run -p xtask -- ci
```
= `cargo fmt --check`; `cargo clippy --workspace --all-targets --locked -- -D warnings`;
`cargo nextest run --workspace --locked` (falls back to `cargo test` without nextest;
test target dir `<target dir>/xtask-test`, honouring `CARGO_TARGET_DIR`). GPU-using packages run
one test at a time: `.config/nextest.toml` group `gpu`, `--test-threads=1` in the fallback.
The **wall-clock-sensitive** tests — space-client's `tests/seamless.rs` and `tests/handoff.rs`: a real
client through real shard loops in real time — have a serial nextest group of their own (`timing`, four
test slots each, decided before `gpu`; `TIMING_TESTS` in xtask); in the fallback they run with their
package, after everything else, one at a time. Their bounds are taken around each change of shard and
compared with the same flight away from one (module docs of `seamless.rs`); do not add a test that
compares a whole-flight maximum with the test's own `Instant::now()`. `SPACE_SKIP_TIMING=1` skips
`seamless.rs` (the CI's second, renderer-only pass over space-client sets it). Tests that need a tick
clock set the time themselves (`space_server::clock::ManualTime`, `Pacer`) instead of sleeping.
Baseline history: Phase 8 394 passed; after the audit 557; after the online-only rework
(web UI, scenes, harness, textures/shadows, connect screen): 950 passed, 0 failed, 3 ignored;
after physics (bodies, ghosts, body handoff, protocol 5) and the renderer fixes (sky occlusion,
ray-traced glass, tinted SSR, fog into the sky): 1081 passed, 0 failed, 4 ignored;
after the merged demo area (orbits, scene clock, fast travel, protocol 6): 1097 passed, 0 failed, 4 ignored;
after the sharding branch was merged with it (map, shard lifecycle, one address, protocol 7): 1235 passed,
0 failed, 4 ignored; after seamless crossing (cluster clock, ship hysteresis, one timeline on the client):
1257 passed, 0 failed, 4 ignored — `space-client/tests/seamless.rs` alone flies for about six minutes
(`cargo test` fallback, no nextest)
after physics on four shards (islands by closest approach, the bubble split, `tests/four_cells.rs`):
1273 passed, 0 failed, 4 ignored, 865 s;
after the review of seamless crossing (clock slew and `ManualTime`, the install bound, `Snapshot::more`,
the resent input tail, ships meeting across a border, the timing tests reworked): 1288 passed, 0 failed,
4 ignored, 539 s — `seamless.rs` now flies for 2 min 40 s;
after far replicas (distance steps, `tests/far.rs`, ghosts stamped with their tick): 1310 passed,
0 failed, 4 ignored, 527 s;
after the shard manager (load reports, autoscaling on the hot cell, drain and retire, the Docker
launcher): 1348 passed, 0 failed, 4 ignored, 551 s;
after the demo area's far content (46 scene bodies, `scripts/harness/sharding.lua`): 1352 passed,
0 failed, 4 ignored, 545 s;
after the review of the far replicas and the shard manager (drains, launcher off the loop, instance
label, loss handling, far cadence, fast travel off): 1378 passed, 0 failed, 4 ignored, 554 s
(Windows, RTX 4080, `SPACE_REQUIRE_GPU=1 SPACE_RENDER_VALIDATION=1`).
Wasm build (also a CI job):
`cargo build --release --locked --target wasm32-unknown-unknown -p space-viewer`.
Lockfile edits need a non-`--locked` cargo command first (e.g. `cargo metadata`).
CI is not the end: `AGENT.md` step 4 (look at it on a running stack) comes after.

## GPU tests
Without a Vulkan device the render tests skip loudly. The userspace stack lives in
`/workspace/.local/vulkan` (`cargo xtask gpu-setup` installs it; needs deb.debian.org).
```sh
P=/workspace/.local/vulkan/usr
export LD_LIBRARY_PATH=$P/lib/x86_64-linux-gnu
# lavapipe (CPU, ray_query yes; exact-pixel tests must pass here):
VK_DRIVER_FILES=$P/share/vulkan/icd.d/lvp_icd.json \
  cargo test -p space-render -p space-client -- --test-threads=1
# real iGPU (RADV GFX1151):
VK_DRIVER_FILES=$P/share/vulkan/icd.d/radeon_icd.json cargo test -p space-render --test gpu -- --nocapture
```
(prefix the CI env vars above for TMPDIR/RUSTUP.) `context_reports_its_device` with
`--nocapture` prints which device ran. `SPACE_RENDER_VALIDATION=1` + `VK_LAYER_PATH=$P/share/vulkan/explicit_layer.d`
requires the Khronos validation layer and asserts zero validation messages (a missing
layer is then a hard failure; the CI GPU step is `continue-on-error: true` as the hosted runner's Mesa is too old).
Do not run GPU tests in parallel (`--test-threads=1`).
`SPACE_REQUIRE_GPU=1` turns the skip into a failure (`space_render::skip_or_require_gpu`): set it on any
machine that is supposed to have a device, otherwise "nothing rendered" is a green run.
On Windows with a ray-tracing GPU and the Vulkan SDK the same tests run natively:
`$env:SPACE_REQUIRE_GPU='1'; $env:SPACE_RENDER_VALIDATION='1'; cargo test -p space-render -p space-client -- --test-threads=1`.
The renderer needs `VK_KHR_acceleration_structure`: lavapipe has it from Mesa 24.1 (Ubuntu 24.04's
Mesa 25 works, verified in WSL; jammy's 23.x does not).

## Renderer notes
- One frame = one command buffer, one submission. `RenderContext` keeps the frame's GPU resources
  (`passes::FrameGpu`, instance buffers, BLAS/TLAS, uploaded meshes and textures) and reuses them while the
  size/feature set is unchanged; `render_offscreen` reads pixels back, `render_to_swapchain` blits into the
  swapchain image and presents (no readback, no wait). Drop a `Swapchain` before its `RenderContext`.
- Depth is **reversed with an infinite far plane** (`scene::mat4_perspective_reversed`: depth = near/distance,
  compare `GREATER`, clear 0). That is what makes solar-system distances work; do not "fix" it back.
- Spheres are tessellated per instance from projected size (`scene::sphere_level_for`, levels 0..3 =
  1/4/16/64 segments per icosahedron edge; the draw's first vertex carries the level). The vertex shader's
  tables and `scene::sphere_mesh` must stay identical - tests compare them, and the rasterised vs
  ray-traced outline.
- The material buffer (rgba16f) carries metallic, roughness and the octahedral shading normal; the post
  pass (SSR/refraction) reads normals from it, not from depth. The HDR target's alpha is **coverage**
  (cleared to 0): the post pass uses it to tell "nothing drawn here".
- **Shadows** are ray-traced (one ray per light per pixel, hard), preset High/Medium, and only traced when
  the scene has a light and the device has ray query (`RenderContext::shadows_active`). Closed solids are
  single-sided for shadow rays; the facing rule is pinned by `shadow_rays_ignore_the_faces_they_leave_a_solid_through`.
- **Models** (`ModelInstance` -> `Arc<MeshData>`): one BLAS per mesh (built once, cached with its buffers and
  textures), one TLAS instance per placement after the sphere/box instances. Keep and reuse the `Arc`: the
  GPU cache is keyed by it, a mesh rebuilt every frame is re-uploaded every frame. Models face **-Z**, +Y up
  (`sample_ship()`, `sample_character()`: `rotation_y(atan2(-d.x, -d.z))` to face `d`).
- **Textures**: base colour only (sRGB RGBA8, full mip chain), `MeshVertex.uv` with (0,0) top-left,
  `MeshMaterial.base_color_texture` indexes `MeshData.textures`. Procedural: `Texture::{checkerboard, uv_grid,
  hull_panels, brushed_metal, hazard_stripes, solid}`. GLB: `TEXCOORD_0` + embedded PNG/JPEG `baseColorTexture`;
  everything else is ignored with a warning.
- **Reflections are screen-space**: a mirror only shows what is on screen (use a mirror floor), weighted by
  the surface's specular colour (a third `rgba8` target), the same tint as the environment reflection where
  the screen has nothing - gold mirrors everything in gold.
- **Glass** (sphere/box instances only) is ray traced with ray query: bent going in and coming out,
  reflected inside up to `glass_bounces` (3 High, 1 Medium), then the frame's colour of what it hits, a rough
  shading if that is off screen, or the environment - never black. Without ray query the old stand-in
  remains (one bend, a lookup 2 m along; `RenderContext::glass_rays_active`).
- **Environment** (`Scene.environment`, default none = frames unchanged): a sky gradient giving ambient
  light, reflections, the refraction fallback and optionally a painted background, into which distance fog
  fades (far geometry melts into the sky; the background itself is not fogged). With ray query, **sky
  rays** (8 High, 4 Medium) occlude the ambient term and the environment reflection per pixel, on a fixed
  4x4 pattern averaged by the post pass (no noise, no flicker): rooms are as dark as their openings. GI
  probes see the environment too (a probe ray that misses takes the sky). `Low`/`Ci` cast no sky rays.
- Surfaces are shaded on the side the camera sees (`side_of`): closed solids are dark inside, open sheets
  have two outsides. Nothing is see-through from behind.
- **Overlay** (`Scene.overlay`): rectangles and 5x7 bitmap text in output pixels, drawn after tone mapping,
  identical in readback and swapchain frames; empty = no pass. The connect screen and HUD are built on it.
- `scene::mat4_inverse` is exercised with turned cameras (`the_inverse_of_a_turned_cameras_view_projection_is_its_inverse`):
  a wrong inverse shows up as misplaced reflections/refractions only when the camera has yaw.
- Golden `GALAXY_BRIGHTNESS` in `tests/gpu.rs` pins LOD selection + star geometry; its comment has the history.
- `cargo run --release -p space-render --example demo_shots -- OUT_DIR [--no-environment]` renders a set of
  reference pictures; open them after any renderer change.

## Invariants - do not break
- **Wire gate**: `PROTOCOL_VERSION` (`space-protocol/src/codec.rs`, now 7: `ShardHello` carries an optional
  client endpoint and, as its last field, the cluster secret, a `Reject`/`Draining` answer to it means "ask
  again", a shard-to-shard `Hello` carries the peer token behind the id in its token field, `ShardRegion`
  carries a `client_endpoint` behind its `endpoint`, `ShardWelcome` ends with the cluster's tick origin
  (`tick_origin_us`), `Snapshot` ends with `more` (a snapshot that needs several frames is one snapshot: the
  client drops what it does not name only when the chunk that says `more: false` arrives), and
  `RejectReason::Starting` and the messages
  `JoinRedirect` (tag 29), `FarInterest` (30), `FarUpdate` (31) and `ShardRetire { shard, reason }` (32:
  orchestrator -> shard, "stop": the orderly end of a drained shard) were appended, `LoadReport` ends with
  `entities`, `tick_mean_ms`, `tick_max_ms`, `skipped`, `hot_cell` (`[i64; 3]`) and `hot_cell_players` (its
  last field; it has a `Default`), and a replicated
  entity may carry component id 11, an `Origin` (`space_protocol::far`; not a component of the
  simulation: a shard adds it to what it tells clients about a ghost or a far replica); 6, as shipped from `main`: `SceneClock` (tag 28) and
  `PlayerInput::fast` / 25-byte `ShipControls`; 5: bodies with component ids 4-10 in
  snapshots/deltas, a delta entry for a known entity lists only changed components - `ShipState` and `Owner`
  always, an empty payload removes one - and `GhostUpdate` is the per-tick batch of border ghosts) is checked in every hello
  (`check_hello`). `Message::VARIANT_COUNT` (33) = one past the highest tag; only APPEND variants,
  bump the count, add the arm to `wire_tag()`; `message_tags_are_dense_and_covered` and
  `unknown_tag_rejected` (acceptance.rs) pin it. Changing existing layouts - or adding a message an
  older peer would receive unasked and fail on - means bumping `PROTOCOL_VERSION` (7 is this branch's and has not shipped: what the sharding work
  still changes stays in 7 until it is pushed to `main`; 6 is released and never changes again).
  `Hello` never changes its layout: that is what keeps a version mismatch a readable `Reject`. Client,
  shard and orchestrator are always built from one commit.
- **One cluster clock** (`docs/plan-sharding.md` §5, `space-server/src/clock.rs`): every shard counts its
  ticks from the cluster's **tick origin** — the orchestrator's (`--tick-origin`, default
  `SceneClock::DEFAULT_ORIGIN_US` = 2026-01-01T00:00:00Z) in the `ShardWelcome`, a standalone shard's own
  `--tick-origin` or the same default — so a tick number is the same instant on every shard. It **is** the
  scene clock a shard tells its clients (`SceneClock::cluster`): one mechanism, and that mapping is never
  re-anchored. An origin in the future is refused (both binaries, and a welcome that names one): there is
  no tick to run. *When* a shard steps follows the wall clock: placed by it at start, measured against it
  every second and slewed (0.5 % at most, never backwards; behind by more than 0.5 s it steps and then
  bursts or jumps) — so a replacement started hours later is on the same ticks to within a few
  milliseconds, not by luck. A shard that falls behind makes up to 8 ticks up at once and jumps over more
  (`ShardStats::skipped`, a log line). **Limits, stated:** all shards of a cluster run on one host; hosts
  whose clocks differ are not handled; an offset takes 7 s per tick to slew away, and until it is gone a
  crossing between the two shards is a commit that many ticks early or late; a wall clock stepped *back*
  is only slewed (200 s per second). What rests on the clock: a handoff installs on the tick its state
  describes (no tick lost, a late commit caught up for at most 15 ticks — H2, H4, H5), and a client keeps
  its clock, prediction, inputs and histories through a switch. Shards that are stepped by hand (tests) or
  run a loop of their own count from their own zero and take the old paths: new code must not assume two
  shards' ticks agree without `Shard::start_clock` on both. Tests that need a clock use
  `clock::ManualTime` and `clock::Pacer`, not sleeps.
- **A player's ship moves for where it is** (`border.rs`, "Players' ships"): never with an island (a
  bystander must not be handed over because somebody flew past), never refused by the crossing rule, never
  called off because a body of its group was. `space-server/tests/crossing.rs` pins each with a mutation.
  Two ships that push at each other across a border therefore stay on two shards, each the other's ghost:
  a **ship's** ghost that is about to touch one ship is presented to the solver at the velocity the two
  share (`space-sim/src/rigid.rs`, `meeting_velocities`), so they stand or move off together as on one
  shard instead of bouncing (`two_players_pushing_each_other_across_a_border`).
- **A body changes shard for a reason** (`docs/plan-sharding.md` §7, `border.rs`): because it moved into
  another region (2 m in, or 3 s), or because something that is **about to touch it** is on another shard -
  an island is what is within a metre (bounding spheres) or comes that close within a second *on the way it
  is moving* (`space_sim::IslandQuery`: closest approach, not "everything within `|v|` x 1 s"). Not because
  a player is near or flies past. Two shards never offer each other members of one island: a
  consolidation waits two passes (`CONFIRM_TICKS`), an island waits for a member that is on its way here,
  and nothing follows a player back to the shard it is leaving. `aborted` stays 0 in every scenario of
  `space-server/tests/four_cells.rs`; each test there names the mutation it catches, and compares four
  shards with one: bit-identical for what is never handed over or handed over in free flight, within a
  millimetre for what is handed over in contact (the solver's warm start does not travel).
- **Seeing across shards** (`docs/plan-sharding.md` §6, `space-server/src/far.rs`,
  `space_protocol::far`): a player sees every ship and body within 10 km whichever shard simulates
  it. Shard to shard on the peer links, relayed by the player's own shard: **a client keeps one play
  connection**, and no listener or credential was added. A **far replica is for seeing only**: it is
  in no world on the shard (interest grid + component log) and in no prediction on the client. **One
  guid, one copy** per shard - own > ghost > far - changing in place, so a client is never told a
  removal for something it can still see and its history goes on; a far message never touches a
  guid that is in the receiver's world. Only `FAR_COMPONENTS` cross (no `Owner`, no `ShipControls`,
  nothing private), cut down on both ends. A far sample keeps **the tick it is the state of** (the
  cluster's; the receiver holds it within 8 ticks of its own) and is not moved between samples; a
  **ghost** shows its owner's state of *this shard's tick* whichever of the two stepped first
  (`GhostEntry::shown` - moving every ghost on a tick after every step made fast ghosts leap a tick
  of motion to and fro). **How often**: the distance steps, for every entity and both hops; nothing
  is sent for what did not change, on either hop; a far replica at rest costs 12 B per 2 s between
  shards. **A far sample is relayed in the frame of the tick it reaches the player's shard**: a
  step's spacing is counted between the *samples'* ticks (`replicate::step_is_due`, `Sent::sample`),
  never from when something was last sent to the client - a re-send must not move the cadence, or
  the stored sample ages in the shard (far moons that stood and leapt). So a far sample is with the
  client a constant 0 to 2 ticks after the tick it is of, and `far::draw_delay` (spacing + 4 ticks)
  covers that; a truly lost frame or a stalled shard is not covered. **A delta entry lists what
  changed since the last frame known to have arrived** (`Sent::base`: acknowledged or aged out),
  removals included, so a lost frame takes no one-off change with it. A shard merged away stays a
  far neighbour for its minute of grace (`far_neighbours`). **Bounded**: a `FarUpdate` <= 32 KiB
  (removals budgeted too), 512 KiB/s, 180 updates/s and 4096 replicas per peer, what does not
  fit waits a tick; refused messages are counted (`ShardStats::far_refused`) and logged through a
  `LogBudget`. `space-server/tests/far.rs` pins each with a mutation. Known limits, stated: in the
  dev layout a thing within 64 m of **any** face of its cell is a ghost on the root's shard too
  (`territory_distance` is a lower bound), so far replicas only appear more than 64 m from both
  border planes (the demo area's far content is placed there: 11 far replicas from the spawn point,
  `space-server/tests/far.rs`' demo-area test); a cell is 1024 m, so from the spawn point no far
  replica of the dev layout is further than some 1.4 km — beyond the three carved cells everything
  is the root's own again; a player in fast travel outruns the 10-tick interest (the far set is up to a
  third of a second of its flight behind).
- **Bubbles** (`space-sim/src/rigid.rs`): on a split the **largest** group keeps the solver's world, and a
  body is never inserted into one asleep. Either the other way round makes resting stacks drop into the
  floor for a few ticks whenever a player walks away from them (`space-sim/tests/physics.rs`,
  `a_resting_stack_...`).
- **Dev-profile opt-levels** (`Cargo.toml`): `rapier3d-f64`, `parry3d-f64` at 3 and `space-sim` at 2 in
  debug builds, for the tests that step minutes of physics. No fast-math anywhere: results are the same at
  every opt-level, and tests compare trajectories bit for bit.
- **One address** (`docs/plan-sharding.md` §4): a new player **always starts on the scene's spawn point**,
  whatever address was dialed. A hello is answered with `Welcome` (a shard that takes the player),
  `JoinRedirect { shard, shard_map }` (not here: the orchestrator's player port, or a shard that does not
  own the spawn point) or `Reject`. The client (`ServerLink::join`) dials the named shard's **client
  endpoint** from that map, at most 2 redirects per attempt; an endpoint without an IP (`0.0.0.0:port`)
  means *the host of the address the player dialed first* (`space_protocol::resolve_client_endpoint`), for a
  join redirect and a handoff `Redirect` alike; port 0 = not known. `ShardRegion::new(node, shard, endpoint)`
  = client endpoint is the endpoint (static maps, `--peer`); `with_client_endpoint` for another. The
  orchestrator decides the map's client endpoints (hello's wish, else the endpoint; the `process` launcher
  passes `--client-endpoint 0.0.0.0:<host port>`). A client keeps the **entry** address and retries *it*,
  never a shard's. Only a player whose ship is being handed over (the owner token of a staged transfer)
  is welcomed by a shard that does not own the spawn point; one that never adopts is sent on like a fresh
  join (`client_timeout_pass`). No region-centre spawn exists any more: do not bring one back.
  **A hello without a player token is refused** by every shard, and "the player of a staged ship" is
  decided by `handoff::Incoming::is_owned_by` only (a ship transfer, a non-empty owner token, constant
  time): a body is staged under the empty token, which must never match anything. **A map sent to a
  player is `ShardMap::for_players()`** (`Welcome`, `JoinRedirect`, the handoff's `Redirect`, the player
  port's answer): every region's internal `endpoint` replaced by its `client_endpoint`; the real map
  stays between shards and the orchestrator. The client never reads `endpoint`.
- **The handoff's redirect**: the client's `Redirect` is sent when the peer's `HandoffAccept` arrives
  (`Shard::on_accept` -> `redirect_owner`), never with the offer - the transfer is staged before the
  client can say hello there - and **only to the ship's own player**: a bystander stays on its shard and
  sees the ship as a ghost. The client (`link.rs`) says hello once, follows a redirect only for the ship
  it flies, takes a `JoinRedirect` **or a `Reject`** on a handoff's connection as the end of that
  attempt, never reuses a connection that has closed, and leaves a shard that gave it a ship of its own
  instead of the one being handed over. `space-client/tests/handoff.rs` flies out of the spawn shard and
  back in and holds every crossing to: handoff within 2 s of the border, no jump over 0.5 m between two
  frames, no step back, one session per player on the spawn shard.
- **The player port** (`space-orchestrator/src/player.rs`, `--player-bind`, off unless given) is the public
  entry, **not** a control surface: no token on any bind, and in exchange no capability - a QUIC retry
  for a first packet whose source is not validated (before a slot is taken), a hello timeout (5 s), the
  published answer (`Orchestrator::player_answer`: the spawn point's owner if it is connected, not dead
  and has a client endpoint, else `Starting`; its map is `for_players()`), written for at most 1 s
  (`space_net::LAST_WORD_TIMEOUT`: a peer that grants no receive window), a close that waits at most 1 s,
  and one bound around the whole connection; at most 256 connections at once (no cap per source, a known
  limit in `docs/security-notes.md`); nothing behind the hello is read; it holds no handle into the run loop, so no shard can
  register and no ack or report can be forged through it; its log lines go through a `LogBudget`. The
  control port answers a hello that is not a shard's (`shard:<id>`) with a readable `Reject`.
- **Scene**: `SceneSet` (orchestrator->shard), `SceneAck` (shard->orchestrator), `SceneUpdate`
  (shard->client, after `Welcome`, on change, and repeated every 5 s). Revisions are monotonic; revision 0
  = "never set, default applies". A scene change is **not** an epoch change and must not touch the epoch
  protocol. Unknown scene ids decode and resolve to the default.
- **The map**: a region is an octree node `depth:x:y:z`; the smallest is one cell (depth 64, 1024 m) and
  `topology::validate` accepts nothing deeper. Every region comes from `topology::carve_plan` (any node inside
  an existing region, for a new owner; `split_plan` = the carve of a direct child octant, kept to regions of
  1024 km or more) and goes through `merge_plan` (back to the **nearest enclosing** region's owner, not
  necessarily the parent node). Plans are pure functions of the map and validated before they are returned
  (H11/H14); the orchestrator's one way in is `begin_carve`/`begin_split`/`begin_merge` -> one epoch, and
  `EpochPublisher::recipients` = every shard of the published map + every owner named in the change (one
  shard that cannot ack holds up every change). Plan and status: `docs/plan-sharding.md`.
- **The anchor**: `space_scene::scene_anchor(map)` is where the scene's spawn point is, for shards and
  clients alike (`link::map_anchor` is the same function): the map's shallowest region; a one-cell root ->
  its centre; any larger root -> its centre corner + `ANCHOR_OFFSET` (6 m, 512 m, 4 m). So under the
  orchestrator's depth-40 root the spawn point lies strictly inside cell `[k,k,k]` (k = the corner's cell
  index), the border planes a carve can draw through a scene are the vertical ones at scene x = -6 and z = -4
  (cells `[k,k,k-1]`, `[k-1,k,k]`, `[k-1,k,k-1]`), and no horizontal border lies within 512 m. Do not move
  the scene back onto the corner: a border at eye height is what the offset is for. `free_spawn_point` hands
  out only points the spawn point's shard owns, and is `None` on every other shard.
- **Handoff freeze**: `space-server/src/handoff.rs` documents guarantees H1-H8 (one simulator per tick,
  bounded time, ...) for ships and for unowned bodies (no client gate, whole component list, island groups
  commit together; `tests/physics.rs` checks the body half; `tests/crossing.rs` checks 2, 4 and 5 on the
  cluster clock, tick for tick against a ship that never changed shard); H9-H15 are the orchestrator's. A settled body is
  on the shard the map names **except** while it is in an island consolidated on another shard or within the
  2 m body hysteresis of a border (`border.rs`). `HandoffOffer` has no token field and space-protocol
  was frozen for handoff - the owner token rides inside the state payload. Don't change the three-phase
  offer/accept/commit exchange without updating the guarantee table and its tests.
- **Viewer wire format** (`space-stream/src/protocol.rs`): message tags 1-4 and `Key` ids 1-15 are
  append-only (`the_wire_ids_are_append_only`); text-entry keys are strokes (act on `down: true`).
- **`#![forbid(unsafe_code)]`** in every library crate except `space-render` (ash FFI) and
  `space-client` (lib; GPU/window glue). Keep new crates forbidding it.
- **No C dependencies**, pure Rust only. Single sanctioned exception: `ring` (via quinn/rustls/rcgen),
  documented in root `Cargo.toml` and `space-net/src/crypto.rs`. Adding any other C/`-sys` dep needs sign-off.
  The Lua interpreter is `lunita` (pure Rust, pinned to an exact version in the root manifest - a young
  single-maintainer crate, so an upgrade is a reviewed change); never `mlua`/`rlua`.
  **A new third-party crate needs a stated reason.** The last one: `space-orchestrator` enables tokio's
  `signal` feature, which puts `signal-hook-registry` into the lockfile (pure Rust, on `libc`, Unix only;
  none on Windows) - the orchestrator has to hear SIGTERM / Ctrl-C to stop and remove the shards it
  started, std has no signal API and the crate forbids `unsafe`. The Docker launcher added none: its HTTP
  is hand-rolled and its JSON is `serde_json`, which the workspace already had (xtask, space-mcp). No
  `bollard`, no HTTP client crate.
- **No JavaScript in the orchestrator web UI**, no JS toolchain anywhere. The viewer is Rust -> wasm.
- **ws:// and http:// only, no TLS** on the stream, MCP, harness and web UI paths. `wss://`/`https://`
  would need `ring`; use a TLS proxy or an SSH tunnel.
- **Control surfaces are closed by default** (`docs/security-notes.md`): the orchestrator web UI
  (`--webui`) and the client harness (`--harness`) are off unless asked for, need no credentials on
  loopback only, and refuse any other bind without a token (>= 16 chars). A tokenless web UI serves
  loopback `Host`s only and refuses proxied requests. Do not weaken either.
- **Shard registration and peers**: the control port takes a `ShardHello` without a secret on a loopback
  bind only; on any other bind (or with `--cluster-secret`/`SPACE_CLUSTER_SECRET` set) the hello must
  carry the cluster secret - checked **before** the cluster name, so a stranger learns nothing - and the
  orchestrator refuses to start off loopback without one (a configured one >= 16 chars, or - with
  `--min-shards` >= 1 - one it generates and hands its shards in their environment, never argv).
  **The hello gates the connection** (`control.rs`, `Inbound`): nothing a connection sends is read before
  the run loop's verdict; refused (for good or "not yet") it is told why and **closed**; welcomed it is
  admitted as one shard id, and a `LoadReport`/`EpochAck`/`SceneAck` is taken only if it names that id -
  the run loop gets the sender from the connection (`handle_*_from`), never from the message. Shards
  present the `ShardWelcome`'s peer token to each other and refuse a peer hello without it; a configured
  one (`--peer-token[-file]`, `SPACE_ORCHESTRATOR_PEER_TOKEN[_FILE]`) must be >= 16 chars (an empty one
  would switch the check off). Constant-time compares (`space_net::constant_time_eq`). **No secret in a
  derived `Debug`**: hold one in `space_net::Secret` (`ControlPlane`, `ShardNet` do), and a message that
  carries one gets a hand-written `Debug` (`Hello` - its token field is the peer token between shards -
  `ShardHello`, `ShardWelcome`; `SpawnSpec`, `OrchestratedOptions`); each has a test that prints it. Log
  lines a stranger can cause (refused hellos, refused peers, misattributed acks) go through a budget
  (`space_net::LogBudget`, the orchestrator's peer-caused event ring): a few per window, the rest counted.
- **Shard lifecycle** (`docs/plan-sharding.md` §2): a shard never names its id or region; **ids are never
  reused**, also not for a shard that restarts in place (it replaces itself under a new id - but only a
  hello that is going to be served, from the IP it advertises, makes the old shard dead, and a launched
  shard's process is kept); a region is owned only once its epoch has committed; one epoch change at a
  time, and none is proposed while a shard of the map holds no control link (every shard of the map must
  ack; three deadlines missed in a row make a connected shard dead). A join whose epoch aborts is stopped.
  **Launches**: matched to their shard by address, never to a dead shard nor to one that registered
  before the launch was staged; the pause after failures (1 s doubling to 30 s) starts over only when a
  launch's shard owns a region; a launch that is no longer needed is stopped, not given a layout cell;
  `--min-shards` <= 8, `--max-shards`, `--max-regions` or the orchestrator does not start; no merge at or
  below `--min-shards`. **A merge ends in a drain, not a kill** (`docs/plan-sharding.md` §3): a merge needs
  the shard's own fresh load report, not an empty shard; at the commit the shard owns nothing, has the role
  `draining` (it does not count towards `--min-shards`, and its reports are still kept) and hands what it
  simulates to the owners of the ground by ordinary handoffs - the other shards hear a shard that left the
  map for `FORMER_SHARD_GRACE_TICKS` = 60 s, which is why `--drain-timeout` is at most 45 s; when a report
  that arrived **after** the commit says 0 players and 0 entities the orchestrator sends `ShardRetire`,
  the shard **exits 0** (every other end of an orchestrated shard is non-zero) and the launcher's
  `exit_code` goes into the event log; not drained in time it is stopped as it is, with a warning naming
  what it held; told to stop and still there after 5 s (`RETIRE_GRACE_TICKS`) its link is closed and its
  process stopped; one that dies while draining is not replaced (it owned nothing) and is taken off the
  list of draining shards at once (`mark_dead`). **No map change while a shard drains**
  (`Orchestrator::drain_refusal`): a draining shard is told of no further epoch and a shard refuses an
  offer of another epoch (`StaleEpoch`), so one more change would strand it with what it holds - none is
  proposed (autoscaler not asked, operator and admin verbs refused with the reason, a hello told to wait,
  no launch staged) until the drain is over, at most `--drain-timeout` + 5 s; a shard that dies meanwhile
  is replaced afterwards, **except the shard being drained into**: then the drain is ended at once (the
  draining shard is stopped as it is) so that the replacement is not held up. **A drain that ends with
  bodies** - the timeout, or the shard died, and its last report since the commit did not say 0 entities
  - raises the scene's revision (`end_drain` -> `respawn_scene`, `RespawnCause::Drained`; left to the
  replacement's commit while a dead shard waits for one), so the scene's bodies are built afresh instead
  of being gone until somebody switches the scene. While it drains it is sent no ghosts (it owns no
  ground) but stays a **far neighbour** both ways for its minute of grace (`far_neighbours`). **The
  launcher is asked and told, never waited for** (`Orchestrator::drive_launcher`): stops and liveness
  questions run in tasks and report back like launches do; one question per launch at a time; a failed
  one, or one open for 5 s, is counted (`launcher_failures`) and written to the event log at most once
  per 10 s. **The player port names a new owner of the spawn point only once that shard has answered a
  `SceneSet` queued behind the commit** (`settle_spawn_owner`): until then it says "starting". The `process` launcher starts the executable by absolute path, keeps
  `SPACE_ORCHESTRATOR_*` and `SPACE_HARNESS_TOKEN` out of the child's environment, and keeps
  `launched-shards.pid` equal to its table of running children. `--host-port-range` must stay out of
  the OS's dynamic port range (a shard binding `127.0.0.1:<p>` takes over the datagrams of a client socket
  on `0.0.0.0:<p>`); nothing checks that for you.
- **Scaling** (`docs/plan-sharding.md` §3, `autoscale.rs`, `orchestrator.rs` "Scaling"): load =
  max(players / capacity, tick p95 / 33.3 ms). **Out**: at or above `--scale-out-load` (0.8) for
  `--scale-out-after` (10 s) the shard's **hot cell** is carved for a new shard (`begin_carve`), lowest id
  first, never beyond `--max-shards`; when the hot cell **is the shard's whole region** (a one-cell shard)
  nothing is done and the event log says so once per cooldown - a region cannot be smaller than a cell.
  **The spawn cell** (`scale_out_target`'s `spawn_cell`): carved only while that relieves somebody - the
  loaded shard has players *outside* it; with every player in it a shard of its own would be exactly as
  loaded, so nothing is done and the log says "the spawn cell is at capacity" once per cooldown. A shard
  that was given the spawn cell is merged back like any carved cell once the pair is cold (the spawn
  point is the surrounding shard's again and the player port follows). A report without a hot cell falls
  back to the octant split (regions >= 1024 km). **In**: a shard *and the shard it would merge into*,
  taken together (players of both against one capacity, tick p95 of both summed against one budget), at
  or below `--scale-in-load` (0.35) for `--scale-in-after` (60 s), highest id first. **The autoscaler
  merges away only a shard the orchestrator launched** (one started by hand - with
  `--min-shards 0` that is every shard but a split's - stays until an operator merges it),
  never one of the layout cells `--min-shards` stands for (with 4: the three cells beside the spawn
  point's), nobody at `--min-shards`, and nobody whose **merge target has no fresh report or no room**:
  no report is not "idle", and a merge - the autoscaler's or an operator's - is refused when the two
  shards hold more than `MAX_PLAYERS_PER_SHARD` (100) players together (`merge_room_refusal`).
  `--shard-capacity` above 100 is refused at startup. An operator's merge is held to `--min-shards` and
  to the report-and-room rule. **No flapping**: the band between the thresholds, a dwell that starts over when the
  load crosses back, `--scale-cooldown` (30 s) per shard after any action that involved it (also after
  "nothing can be done", and for the shard that takes a region back), one action per out-dwell in the
  cluster. Settings without a band (in >= out), capacity 0, a dwell of 0 or a drain over 45 s are refused
  at startup. Timers are ticks, the decision is a pure function: test it with a table of loads, no sleeps
  (`the_cluster_grows_at_the_hot_cell_and_shrinks_again_by_draining`). Stated consequence: a crowd in one
  cell cannot be split (the smallest area is one cell); the cluster no longer grows for it.
- **The Docker launcher** (`launcher/docker.rs`, `docs/security-notes.md` "The Docker socket"): whoever
  holds the socket is root on the host, so: **nothing a player or a shard sends reaches a Docker API
  parameter** (image, command, ports, network, labels come from `DockerOptions`, `SpawnSpec` and the
  daemon's own answers); a shard container gets **no mount, never the socket, no capability**
  (`CapDrop: ALL`), `no-new-privileges`, not privileged, restart policy `no`, **a read-only root** (a shard
  writes no file: keep it that way, or give the launcher a tmpfs for what it writes), `PidsLimit` 1024,
  `Memory` = `MemorySwap` = `--shard-memory` (MiB, default 2048, 0 = none), `User` `10001:10001` by number;
  the cluster secret goes into
  the container's **environment**, never its command, and is scrubbed from every error text - **before**
  the daemon's words are clipped (`Failure::without`, then `clip`), never after; containers are
  removed **only when they carry all three labels** (`space.cluster=<cluster>`, `space.role=shard`,
  `space.instance=<instance>`) - the
  leftovers of an earlier run at start (removed, not adopted: a shard cannot join a new orchestrator), and
  a taken name only then. **The instance, not the cluster's name, is what tells two stacks on one host
  apart** (`--docker-instance` / `SPACE_ORCHESTRATOR_DOCKER_INSTANCE`, else an id made once and kept in
  `<state>.instance`; the local compose file passes the compose project's name; never derived from a
  secret); `space.launch` and `space.port` are for the operator and are not read. The orchestrator finds
  its own container **by name** (`--docker-self` / `SPACE_ORCHESTRATOR_DOCKER_SELF`, set by both compose
  files), then by `$HOSTNAME` - the Engine resolves a container by id, name or id prefix, never by host
  name, and a container Watchtower recreated carries the old one's id as its host name - and inspects
  nothing when `--docker-network` and `--docker-control-addr` are both given. An image the daemon no
  longer has when a container is created (404) is pulled again by that launch; the orchestrator removes
  no image. Every API call has a time limit (10 s, 15 s a remove, 300 s a pull, 20 s for the
  shutdown) and an error is a failed launch; no registry credentials. A shard container is
  `space-server --orchestrator <control> --cluster <id> --bind 0.0.0.0:P --advertise 0.0.0.0:P
  --client-endpoint 0.0.0.0:P`, named `<prefix><P>`, `P/udp` published under the same number: the binary
  must stay on the shard image's PATH under that name. Do not weaken any of it.
- naga must resolve >= 30 (`space-render/build.rs` asserts it). ShardIds are monotonic, never reused.
- Docker: both compose files hold **the orchestrator alone** (socket mounted, `group_add: DOCKER_GID`,
  player port published, control port never, `SPACE_CLUSTER_SECRET` required with no default in both); the
  shard image has to exist locally under `SPACE_SHARD_IMAGE` (`space/shard:dev`). **Executed** (Windows 11,
  Docker Desktop 28.5.1 with the Linux engine in WSL2, 2026-11-24): `deploy/docker-compose.yaml` as it is
  with ports moved by environment - four shard containers created, a client on the host through the
  published player port, crossings between shard containers, a shard container killed and replaced,
  scale-out to a sixth container, scale-in with drain and exit status 0, SIGTERM removing every shard
  container - on images with the same runtime stages and binaries built from a throwaway Dockerfile;
  the images from `Dockerfile.orchestrator` and `Dockerfile.shard` themselves were built and run through a
  start, a client's join, a killed orchestrator with the leftovers removed at the next start, and
  `compose down`. **Only parsed** (`docker compose config`): `deploy/proxy/docker-compose.yaml`. **Not
  executed**: anything on a Linux host (socket group other than 0, firewall, ports to another machine),
  Watchtower replacing the orchestrator, a pull of the `sha-` shard image from ghcr, the `client` stream
  service, `release-images.yml` (YAML-checked only). Seen there and not solved: Docker Desktop's VM steps
  its wall clock, the shards' `clock offset` ran to -84 ms and two crossings stepped by 0.40 m and 0.76 m
  (`deploy/README.md`). Say plainly which files you did and did not execute when you touch them. The
  control port is never published.
- **Pipeline** (details: `deploy/README.md`, `deploy/proxy/README.md`): EVERY push to `main` (no path filter, no
  concurrency group) runs both `release-images.yml`, which pushes
  `ghcr.io/eliasstepanik/space-{shard,orchestrator,client}` as `:sha-<short>` (matrix job; the orchestrator's
  build gets `SPACE_SHARD_IMAGE=…/space-shard:sha-<short>` and starts its shards from exactly that image) and
  then moves `:latest` in a second job, `latest`, that needs all three - shard, client, **the orchestrator
  last**, so a commit's shard image is in the registry before the orchestrator's `:latest` names it; keep
  that order - and
  `release.yml`, which tags the commit with the next patch version (`.github/scripts/release_version.js`) and attaches
  `space-client`, `space-client.exe`, `space-viewer.zip` to a GitHub Release. So a push to `main` IS a release and a
  deploy: never tag a release by hand after a push (a pushed `v*` tag is only for moving the minor/major version).
  The browser viewer is a local/agent testing tool and is not deployed; the proxy host
  runs Watchtower (label opt-in, 60s poll, stack in `/home/proxy/stacks/space`), which replaces the orchestrator only: the old one gets SIGTERM and removes its shard containers, the new one starts them from its own commit's image; `deploy.yml` is the manual SSH path (`redeploy.sh`)
  (secrets `DEPLOY_SSH_HOST/USER/KEY`). ci.yml's `lint` job parses all workflow YAML (`.github/scripts/check_yaml.py`):
  keep heredoc/script bodies out of `run: |` blocks (use `.github/scripts/`) and re-run that check after any workflow edit.

## Phase map
0 risk-gate spikes (`spikes/`); 1 world/ecs/protocol/sim; 2 QUIC netcode, headless multiplayer, handoff H1-H8;
3 orchestrator + autoscaler + H9-H15 (`--launcher docker` deferred then; built in the sharding work's phase 6, `docs/plan-sharding.md` §8); 4 ash renderer core (headless, GPU tests);
5 ray-query GI/SSR/fog/refraction + quality presets (`--quality high|medium|low|ci`);
6 pixel streaming: `space-stream`, `space-client --stream`, wasm `space-viewer` (JPEG only);
7 MCP servers; 8 packaging: Docker images, compose `stream` profile, CI wasm + docker jobs, this skill, v0.1.0.
After v0.1: playable window + GLB + shard transitions; then the online-only rework: connect screen,
server-owned scenes with the demo area, orchestrator web UI, textures/shadows/overlay/environment, the
harness (HTTP + Lua), per-push releases.

## MCP servers (agentic testing)
Both: `--stdio` (default) or `--http 127.0.0.1:PORT`.
- `cargo run -p space-mcp-server -- [--bind 127.0.0.1:0] [--no-tick]`: real shard behind real QUIC at 30 Hz.
  Tools: `shard_status` (reports the QUIC addr), `topology`, `list_ships`, `spawn_ship`, `ship_state`,
  `set_scene` (the shard starts with no scene: an empty world), `list_bodies` (ships, bodies, ghosts), `spawn_body`,
  `despawn_ship`, `advance_ticks` (deterministic stepping; use with `--no-tick`). No split/merge (orchestrator's job).
- `cargo run -p space-mcp-client -- [--out-dir DIR]`: a player. Tools: `connect` (host:port from `shard_status`),
  `disconnect`, `client_status`, `send_input` (thrust, optional duration), `capture_frame` (offscreen render of
  the scene the shard shows, base64 JPEG or file; works on lavapipe).
Typical loop: start server, `shard_status` -> addr; client `connect`; `send_input`; `advance_ticks`;
`ship_state` vs `client_status` to check replication; `capture_frame` to inspect pixels.
For anything a real client shows (connect screen, HUD, scene switches) prefer the harness in
`.warren/skills/space-visual-test/SKILL.md`: it drives the real `space-client`.

## Run the stack by hand
`scripts/dev.ps1` / `scripts/dev.sh`: `-Topology orchestrated|single|two-shards` (default orchestrated:
the orchestrator with its **player port `127.0.0.1:7777`** - what the client is pointed at - its control
port 7900 and web UI `http://127.0.0.1:7980/`; it always starts its shards itself), `-MinShards N`
(orchestrated, **default 4**, 1 to 8: N shards with `--launcher process`, ports from 7870, logs
`target/dev/shard-<port>.log`; 4 = the demo area's four quadrants, 1 = one shard and no border; the script
waits until they own their regions and the player port has a shard to send a client to; the orchestrator
binary's own default stays 0; the compose files run the orchestrator alone with `--launcher docker
--min-shards 4`, the dev scripts never need Docker), `-Scene` (what the SERVERS
show), `-Client window|headless|stream`, `-Harness`, `-Script`, `-ServersOnly`, `-Stop`, `-PortOffset`.
`-Stop` stops the pids in `target/dev/servers.pid` and `launched-shards.pid` (the orchestrator's list of
its running shards) - each only if its executable lies in this checkout's `target/release`.
Orchestrator flags for it: `--player-bind`, `--min-shards` (at most 8), `--max-shards` (8),
`--launcher local|process|docker`, `--shard-exe`, `--shard-log-dir`, `--host-port-range`,
`--cluster-secret[-file]`, `--peer-token[-file]` (env forms in `space-orchestrator`'s `main.rs` docs),
the autoscaler's `--shard-capacity` (100), `--scale-out-load` (0.8), `--scale-out-after` (10 s),
`--scale-in-load` (0.35), `--scale-in-after` (60 s), `--scale-cooldown` (30 s), `--drain-timeout` (30 s, at
most 45; env `SPACE_ORCHESTRATOR_SHARD_CAPACITY`, `_SCALE_OUT_LOAD`, `_SCALE_OUT_AFTER`, `_SCALE_IN_LOAD`,
`_SCALE_IN_AFTER`, `_SCALE_COOLDOWN`, `_DRAIN_TIMEOUT` - set capacity 2, out after 5, in after 15,
cooldown 10 in the environment of `dev.ps1`/`dev.sh` to watch a cluster scale with two clients), the
`docker` launcher's `--docker-socket` (`/var/run/docker.sock`), `--shard-image` (no default in the binary;
baked into the orchestrator image), `--docker-network` (default: the orchestrator container's own),
`--docker-control-addr` (default: its own IP there + the control port), `--docker-publish-ip` (`0.0.0.0`),
`--docker-name-prefix` (`space-shard-`), `--docker-self` (the orchestrator container's own name; default
`$HOSTNAME`), `--docker-instance` (default: an id kept in `<state>.instance`), `--shard-memory` (MiB, 2048),
`--tick-origin <unix µs>` (the cluster's tick 0, refused if it lies in the future; `space-server --tick-origin` for a standalone shard; both
default to the same fixed instant, so there is rarely a reason to give one).
`two-shards` is the manual seamless-handoff setup (shard 1 owns `64:0:0:0`, shard 2 `64:1:0:0`, `--peer` each
other; a client that dials shard 2 is sent to shard 1, which owns the spawn point); the automated versions are
`crates/space-client/tests/handoff.rs` (two shards, the link) and `crates/space-client/tests/seamless.rs` (four
shards, the client core frame by frame: nothing that is drawn may step at a change of shard).
Ports: 7777 what a client connects to (the player port, or the shard in `single`/`two-shards`), 7778 the
second shard of `two-shards`, 7870-7899 shards the orchestrator starts, 7900 orchestrator control, 7980 web UI, 8097
viewer page, 8098 harness API, 8099 stream.
Everything else about running and looking at it: `.warren/skills/space-visual-test/SKILL.md`. Containers: `deploy/README.md`.
