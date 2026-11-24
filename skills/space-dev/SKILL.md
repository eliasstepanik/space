---
name: space-dev
description: Use when developing, testing, packaging or debugging the space engine (the Rust workspace with space-world/ecs/protocol/sim/net/server/orchestrator/render/client/harness/stream/viewer/mcp crates) - building, running CI, GPU/lavapipe tests, wire-protocol or handoff changes, the orchestrator web UI, scenes, Docker/CI files, the release pipeline, or driving a shard/client through the MCP servers. For looking at the running game use the space-visual-test skill.
---

# space-dev

Rust workspace, toolchain pinned in `rust-toolchain.toml` (1.100.0 + rustfmt, clippy,
`wasm32-unknown-unknown`). All crates are 0.1.0. Rules and workflow: `AGENT.md` (binding).
Visual testing: `skills/space-visual-test/SKILL.md`. Docs: `docs/plan.md` (phases),
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
| `space-protocol` | wire messages, codec, `PROTOCOL_VERSION`, `Message::VARIANT_COUNT`; `scene.rs` = the scene catalogue (`Scene`, `SceneId`, `SceneState`; id 2 `solar` retired: `Scene::RETIRED_IDS`, resolves to the default) and the **scene clock** (`SceneClock`: a shard's tick -> UTC µs mapping, exactly 1e6/30 µs per tick) |
| `space-sim` | deterministic simulation, `SimWorld`: ship flight (`physics.rs`), rigid bodies on `rapier3d-f64` in floating-origin **bubbles** (`rigid.rs`: merge/split/re-anchor rules, statics, islands, ghosts), components `Body`/`Visual`/`Springs`/`Articulation`/`PartStates`/`PosResidual` (ids 4-10, versioned wire forms), gravity zones + large bodies (`gravity.rs`; a `LargeBody` has a `reach`: the pull fades over its outer quarter and is exactly 0 beyond, `f64::INFINITY` = no limit; `BodyPlace::Orbit` = a fixed orbit of **scene time** (`motion.rs`: `OrbitPath` = sum of circles, phase from `t mod period` in integers); inside its reach an orbiting body **carries** what is near it by its displacement over the tick, weighted like the pull (`GravityField::carry`, step 0 of `rigid.rs`'s tick; no fictitious force); `StaticCollider::motion` = a collider teleported along the same path each tick; `SimWorld::set_scene_clock`/`scene_time`), orbits on rails, prediction (`PredictedShip`, `PredictedPhysics`). Ships: `ShipParams::lift` (1.0) = lift engines, the part of the input that pushes against the local pull gets `lift × |g|` more (`physics::lift_accel`, any direction, nothing in weightless space, so free flight is bit-identical); `Body::ship` has friction 0 (half the floor's by the average rule), so a landed ship can walk. **Fast travel** (`ShipControls::fast`, persistent, from `PlayerInput::fast`): speed limit `clamp((surface distance − clearance) / 1 s, 120 m/s, 1e10 m/s)` (`physics::fast_travel_limit`, distance to the nearest large body, zone or static; clearance `FAST_CLEARANCE` = 2 km while closing in on it, 0 while moving away), velocity follows `input × limit` with a 0.5 s time constant (`fast_velocity_step`); switching it off clamps the ship to 120 m/s at once |
| `space-scene` | the physical definition of every scene, shared by shards and clients: statics (the demo area's stations with the client's numbers, the Kestrel collider read from `kestrel_lod1.glb`), the demo area's physics showcase (the gravity room, whose curbs stop balls but not ships; the zero-g corner; the planetoid at `demo::PLANETOID_CENTER`: a 4 m rock with large-body gravity, 4 m/s² on its surface, a 16 m reach, crates resting on it and three balls in orbit), the demo area's **sky** (`solar.rs`: Earth-centred frame, Earth fixed at `EARTH_CENTER` = 8 Earth radii from the spawn point; Sun, planets and Moon on circular Kepler orbits sped up `SPEEDUP` = 8766x; reaches Earth 6 R, others 10 R; real `GM`; moving ball colliders), gravity, the shape catalogue, the `visual` ids and names, `scene_anchor` (where a scene is in the map: the centre of a one-cell root, else the root's centre corner + `ANCHOR_OFFSET` = 6 m, 512 m up, 4 m) |
| `space-net` | quinn QUIC transport (self-signed rustls, `ring`), hello/handshake (a version mismatch is answered with a `Reject`, not a dropped connection; `join` = a player's hello, answered with a `Welcome` or a `JoinRedirect` (`Joined::Elsewhere`), `connect` = a shard's dial, for which a redirect is an error; `Conn::turn_away` answers a hello with one last message and closes once it was read - the write gives up after `LAST_WORD_TIMEOUT` = 1 s, so a peer without a receive window holds nothing; the **first frame** of a connection is read with a hello's size limit, `MAX_HELLO_PAYLOAD` = 4 KiB, not the general 1 MiB, and a shared secret that has to fit one is at most `MAX_SECRET_LEN` = 1024); `secret.rs` = the one constant-time compare every shared secret uses and `Secret`, the type every holder of one keeps it in (its `Debug` prints `<redacted>`); `log_budget.rs` = `LogBudget`, a few log lines per window for what a stranger can cause |
| `space-server` | authoritative shard (`space-server` bin), `handoff.rs` seamless handoff (ships and unowned bodies, island groups; the state carries its scene time as the private component `SCENE_TIME_COMPONENT` and the receiver carries it to its own clock; a player is sent the `Redirect` only once the peer has accepted, and only the ship's owner is); the scene clock (`Shard::sync_scene_clock`, called by the run loops with the wall clock after every tick, re-anchors past `config::SCENE_CLOCK_TOLERANCE_US` = 30 ms and sends `SceneClock` to the clients; also behind `SceneUpdate` on join and with every re-announcement); `orchestrated.rs` = an orchestrated shard's life: it registers **without id or region** and is given both, asks again while told "not yet" (`Reject`/`Draining`), ends on any other refusal, and **stops when it loses the orchestrator** (the process exits non-zero); `net.rs` = the front end, incl. the cluster's peer token (`shard:<id>:<token>` in a peer hello, refused without it when the shard holds one); holds and relays the scene (`--scene` standalone only) and simulates it (`shard_physics.rs`: the scene's bodies, spawned once per cluster by the shallowest region's shard; free spawn points, on the shard's own ground only around the scene's spawn point; ghosts); **joining** (`Shard::join` -> `Welcome` or `JoinRefusal`): a fresh join is welcomed only by the shard that owns the spawn point, every other shard answers `JoinRedirect` to it (`Starting` while its address is unknown) - a player whose ship is being handed over (owner token of a staged transfer) is welcomed anywhere; `--client-endpoint` (orchestrated only) = where its players dial it; `border.rs` = the cross-shard physics rules (ghost margin 64 m + 3 s of motion, islands 0.5 m + 1 s of motion, consolidation onto the home of an island's lowest guid; hysteresis: a body moves 2 m into a region or after 3 s in it, an island that was in a group waits until it has wanted the move for 3 s, doubling up to 48 s for a body that keeps coming back; a player's ship carries `Visual(PLAYER)`, ghosts included); `replicate.rs` = every body in interest, changed components only in deltas (`ComponentLog`) |
| `space-orchestrator` | epoch-versioned shard map (`topology.rs`: `carve_plan` carves any node inside a region for a new owner, down to one cell; the octant split is that with a size floor; a merge gives a region back to the nearest region around it; `epoch.rs`: every shard of the map is sent and must ack a change), `Orchestrator::begin_carve` (asks for a node; nothing is proposed before a shard registers to take it), autoscaler, ShardId counter; the **shard lifecycle** (`orchestrator.rs`: a region is given at the hello - the root, a dead shard's regions, a wanted node, else the next layout cell of `topology::layout_cells` - through an epoch proposed with the shard already connected; `--min-shards`/`--max-shards` and the reconciler, which also stops launches that are no longer needed; supervision: a dead shard - its process gone, its link closed, or three map changes in a row unacknowledged - is replaced under a new id and the scene revision raised); launchers `launcher/{local,process,docker}.rs` (`local` = inside this process, `process` = one `space-server` process per shard, `docker` = stub); cluster secret + peer token (`control.rs`, `config.rs`); `control.rs` is also the control port's front end: a connection is read only after its hello was welcomed, and then only about the shard it was welcomed as (a player's hello there is refused in words); **the player port** (`player.rs`, `--player-bind`, off unless given): the one address players type - a hello is answered with `JoinRedirect { owner of the spawn point, published map }` or `Reject(Starting)` and the connection closed; it only reads the answer the run loop publishes (`Orchestrator::player_answer`); each shard's **client endpoint** (what its hello asks for, else its endpoint) goes into the map, an internal endpoint without an IP is filled in from the control connection (`control::as_seen_from`); owns the scene (`scene.rs`, persisted in `<state>.scene`); operator path `ops.rs`/`snapshot.rs`/`events.rs`; built-in web UI `webui/` (hand-rolled HTTP, server-rendered HTML, **no JavaScript**) |
| `space-render` | ash/Vulkan renderer, naga WGSL->SPIR-V in `build.rs`, ray-query GI and shadows, presets, reversed-Z, `mesh.rs` (GLB models, in the TLAS), `texture.rs` (PNG/JPEG decode, procedural textures), `overlay.rs` (2D rects + built-in 5x7 font), `environment.rs`, `swapchain.rs` |
| `space-client` | the client, one core (`app.rs`) with three front ends: window (`window.rs`), `--headless` and `--stream` (`headless.rs`); `connect.rs`/`address.rs` (connect screen), `input.rs`, `ui.rs` (HUD), `scenes.rs` + `demo_area.rs` (what each server scene looks like; the physics showcase's statics drawn from `space-scene`'s definition), `link.rs` (player link: `ServerLink::join` follows up to `MAX_JOIN_REDIRECTS` = 2 join redirects from the **entry** address and keeps it, `shard_address()` = where it plays; scene, shard transitions, replicated bodies merged per protocol 5), `bodies.rs` + `meshes.rs` (bodies drawn from their collision shapes, interpolated between ticks, springs as coils, joints as pins), `predict.rs` (the own ship predicted through `PredictedPhysics` with the scene's statics, gravity, nearby bodies, the shard's scene clock and its word on fast travel; the draw clock); the demo area's sky drawn every frame at the own ship's draw tick (`scenes::SkyTime`, `World::sky`, interpolated between ticks), sunlight from the Sun as the 8th light |
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
Baseline history: Phase 8 394 passed; after the audit 557; after the online-only rework
(web UI, scenes, harness, textures/shadows, connect screen): 950 passed, 0 failed, 3 ignored;
after physics (bodies, ghosts, body handoff, protocol 5) and the renderer fixes (sky occlusion,
ray-traced glass, tinted SSR, fog into the sky): 1081 passed, 0 failed, 4 ignored;
after the merged demo area (orbits, scene clock, fast travel, protocol 6): 1097 passed, 0 failed, 4 ignored;
after the sharding branch was merged with it (map, shard lifecycle, one address, protocol 7): 1235 passed,
0 failed, 4 ignored (`cargo test` fallback, no nextest)
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
  carries a `client_endpoint` behind its `endpoint`, and `RejectReason::Starting` and the message
  `JoinRedirect` (tag 29) were appended; 6, as shipped from `main`: `SceneClock` (tag 28) and
  `PlayerInput::fast` / 25-byte `ShipControls`; 5: bodies with component ids 4-10 in
  snapshots/deltas, a delta entry for a known entity lists only changed components - `ShipState` and `Owner`
  always, an empty payload removes one - and `GhostUpdate` is the per-tick batch of border ghosts) is checked in every hello
  (`check_hello`). `Message::VARIANT_COUNT` (30) = one past the highest tag; only APPEND variants,
  bump the count, add the arm to `wire_tag()`; `message_tags_are_dense_and_covered` and
  `unknown_tag_rejected` (acceptance.rs) pin it. Changing existing layouts - or adding a message an
  older peer would receive unasked and fail on - means bumping `PROTOCOL_VERSION` (7 is this branch's and has not shipped: what the sharding work
  still changes stays in 7 until it is pushed to `main`; 6 is released and never changes again).
  `Hello` never changes its layout: that is what keeps a version mismatch a readable `Reject`. Client,
  shard and orchestrator are always built from one commit.
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
  commit together; `tests/physics.rs` checks the body half); H9-H15 are the orchestrator's. A settled body is
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
  below `--min-shards`. The `process` launcher starts the executable by absolute path, keeps
  `SPACE_ORCHESTRATOR_*` and `SPACE_HARNESS_TOKEN` out of the child's environment, and keeps
  `launched-shards.pid` equal to its table of running children. `--host-port-range` must stay out of
  the OS's dynamic port range (a shard binding `127.0.0.1:<p>` takes over the datagrams of a client socket
  on `0.0.0.0:<p>`); nothing checks that for you.
- naga must resolve >= 30 (`space-render/build.rs` asserts it). ShardIds are monotonic, never reused.
- Docker: `Dockerfile.orchestrator`, `Dockerfile.shard` and the local compose have been built and run
  (Docker 28) - the compose files **not** since they were given the cluster secret (required, no default:
  `${SPACE_CLUSTER_SECRET:?...}` in the local one) and stopped publishing the control port; say plainly
  which files you did and did not execute when you touch them. The control port is never published.
- **Pipeline** (details: `deploy/README.md`, `deploy/proxy/README.md`): EVERY push to `main` (no path filter, no
  concurrency group) runs both `release-images.yml`, which pushes
  `ghcr.io/eliasstepanik/space-{shard,orchestrator,client}` as `:sha-<short>` and then moves `:latest`, and
  `release.yml`, which tags the commit with the next patch version (`.github/scripts/release_version.js`) and attaches
  `space-client`, `space-client.exe`, `space-viewer.zip` to a GitHub Release. So a push to `main` IS a release and a
  deploy: never tag a release by hand after a push (a pushed `v*` tag is only for moving the minor/major version).
  The browser viewer is a local/agent testing tool and is not deployed; the proxy host
  runs Watchtower (label opt-in, 60s poll, stack in `/home/proxy/stacks/space`); `deploy.yml` is the manual SSH path (`redeploy.sh`)
  (secrets `DEPLOY_SSH_HOST/USER/KEY`). ci.yml's `lint` job parses all workflow YAML (`.github/scripts/check_yaml.py`):
  keep heredoc/script bodies out of `run: |` blocks (use `.github/scripts/`) and re-run that check after any workflow edit.

## Phase map
0 risk-gate spikes (`spikes/`); 1 world/ecs/protocol/sim; 2 QUIC netcode, headless multiplayer, handoff H1-H8;
3 orchestrator + autoscaler + H9-H15 (`--launcher docker` deferred); 4 ash renderer core (headless, GPU tests);
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
`skills/space-visual-test/SKILL.md`: it drives the real `space-client`.

## Run the stack by hand
`scripts/dev.ps1` / `scripts/dev.sh`: `-Topology orchestrated|single|two-shards` (default orchestrated:
the orchestrator with its **player port `127.0.0.1:7777`** - what the client is pointed at - its control
port 7900 and web UI `http://127.0.0.1:7980/`; it always starts its shards itself), `-MinShards N`
(orchestrated, default 1: N shards with `--launcher process`, ports from 7870, logs
`target/dev/shard-<port>.log`; 4 = the demo area's four quadrants; the script waits until they own their
regions and the player port has a shard to send a client to), `-Scene` (what the SERVERS
show), `-Client window|headless|stream`, `-Harness`, `-Script`, `-ServersOnly`, `-Stop`, `-PortOffset`.
`-Stop` stops the pids in `target/dev/servers.pid` and `launched-shards.pid` (the orchestrator's list of
its running shards) - each only if its executable lies in this checkout's `target/release`.
Orchestrator flags for it: `--player-bind`, `--min-shards` (at most 8), `--max-shards` (8),
`--launcher local|process|docker`, `--shard-exe`, `--shard-log-dir`, `--host-port-range`,
`--cluster-secret[-file]`, `--peer-token[-file]` (env forms in `space-orchestrator`'s `main.rs` docs).
`two-shards` is the manual seamless-handoff setup (shard 1 owns `64:0:0:0`, shard 2 `64:1:0:0`, `--peer` each
other; a client that dials shard 2 is sent to shard 1, which owns the spawn point); the automated version is
`crates/space-client/tests/handoff.rs`.
Ports: 7777 what a client connects to (the player port, or the shard in `single`/`two-shards`), 7778 the
second shard of `two-shards`, 7870-7899 shards the orchestrator starts, 7900 orchestrator control, 7980 web UI, 8097
viewer page, 8098 harness API, 8099 stream.
Everything else about running and looking at it: `skills/space-visual-test/SKILL.md`. Containers: `deploy/README.md`.
