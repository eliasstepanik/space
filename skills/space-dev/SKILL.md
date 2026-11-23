---
name: space-dev
description: Use when developing, testing, packaging or debugging the space engine (the Rust workspace with space-world/ecs/protocol/sim/net/server/orchestrator/render/client/harness/stream/viewer/mcp crates) - building, running CI, GPU/lavapipe tests, wire-protocol or handoff changes, the orchestrator web UI, scenes, Docker/CI files, the release pipeline, or driving a shard/client through the MCP servers. For looking at the running game use the space-visual-test skill.
---

# space-dev

Rust workspace, toolchain pinned in `rust-toolchain.toml` (1.100.0 + rustfmt, clippy,
`wasm32-unknown-unknown`). All crates are 0.1.0. Rules and workflow: `AGENT.md` (binding).
Visual testing: `skills/space-visual-test/SKILL.md`. Docs: `docs/plan.md` (phases),
`docs/plan-phase3.md`, `docs/security-notes.md`, `spikes/ENVIRONMENT.md` (GPU env),
`deploy/README.md` (pipeline, web UI, what is/isn't verified).

**Three facts first** (details in `AGENT.md`): the client is **online only**; the **scene is
server state** owned by the orchestrator; **every push to `main` is a release and a deploy**.

## Layout
| crate | role |
|---|---|
| `space-world` | i64-cell + f32 position model (1024 m cells), octree, deterministic galaxy, LOD selector |
| `space-ecs` | custom archetype ECS |
| `space-protocol` | wire messages, codec, `PROTOCOL_VERSION`, `Message::VARIANT_COUNT`; `scene.rs` = the scene catalogue (`Scene`, `SceneId`, `SceneState`) |
| `space-sim` | deterministic physics/orbits/prediction, `SimWorld` |
| `space-net` | quinn QUIC transport (self-signed rustls, `ring`), hello/handshake (a version mismatch is answered with a `Reject`, not a dropped connection) |
| `space-server` | authoritative shard (`space-server` bin), `handoff.rs` seamless handoff; holds and relays the scene (`--scene` standalone only) |
| `space-orchestrator` | epoch-versioned shard map, autoscaler, ShardId counter, `--launcher local`; owns the scene (`scene.rs`, persisted in `<state>.scene`); operator path `ops.rs`/`snapshot.rs`/`events.rs`; built-in web UI `webui/` (hand-rolled HTTP, server-rendered HTML, **no JavaScript**) |
| `space-render` | ash/Vulkan renderer, naga WGSL->SPIR-V in `build.rs`, ray-query GI and shadows, presets, reversed-Z, `mesh.rs` (GLB models, in the TLAS), `texture.rs` (PNG/JPEG decode, procedural textures), `overlay.rs` (2D rects + built-in 5x7 font), `environment.rs`, `swapchain.rs` |
| `space-client` | the client, one core (`app.rs`) with three front ends: window (`window.rs`), `--headless` and `--stream` (`headless.rs`); `connect.rs`/`address.rs` (connect screen), `input.rs`, `ui.rs` (HUD), `scenes.rs` + `demo_area.rs` (what each server scene looks like), `link.rs` (player link, scene, shard transitions) |
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
(web UI, scenes, harness, textures/shadows, connect screen): 950 passed, 0 failed, 3 ignored
(Windows, RTX 4080, `SPACE_REQUIRE_GPU=1`).
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
- **Reflections and refraction are screen-space**: a mirror only shows what is on screen (use a mirror
  floor), glass is for sphere/box instances only. A missed lookup falls back to the scene straight behind
  the glass, then to the environment - never to black.
- **Environment** (`Scene.environment`, default none = frames unchanged): a sky gradient giving ambient
  light, the reflection for polished metal when the screen-space trace misses, the last refraction
  fallback and optionally a painted background. The ambient term is unoccluded.
- **Overlay** (`Scene.overlay`): rectangles and 5x7 bitmap text in output pixels, drawn after tone mapping,
  identical in readback and swapchain frames; empty = no pass. The connect screen and HUD are built on it.
- `scene::mat4_inverse` is exercised with turned cameras (`the_inverse_of_a_turned_cameras_view_projection_is_its_inverse`):
  a wrong inverse shows up as misplaced reflections/refractions only when the camera has yaw.
- Golden `GALAXY_BRIGHTNESS` in `tests/gpu.rs` pins LOD selection + star geometry; its comment has the history.
- `cargo run --release -p space-render --example demo_shots -- OUT_DIR [--no-environment]` renders a set of
  reference pictures; open them after any renderer change.

## Invariants - do not break
- **Wire gate**: `PROTOCOL_VERSION` (`space-protocol/src/codec.rs`, now 4) is checked in every hello
  (`check_hello`). `Message::VARIANT_COUNT` (28) = one past the highest tag; only APPEND variants,
  bump the count, add the arm to `wire_tag()`; `message_tags_are_dense_and_covered` and
  `unknown_tag_rejected` (acceptance.rs) pin it. Changing existing layouts - or adding a message an
  older peer would receive unasked and fail on - means bumping `PROTOCOL_VERSION`. Client, shard and
  orchestrator are always built from one commit.
- **Scene**: `SceneSet` (orchestrator->shard), `SceneAck` (shard->orchestrator), `SceneUpdate`
  (shard->client, after `Welcome`, on change, and repeated every 5 s). Revisions are monotonic; revision 0
  = "never set, default applies". A scene change is **not** an epoch change and must not touch the epoch
  protocol. Unknown scene ids decode and resolve to the default.
- **Handoff freeze**: `space-server/src/handoff.rs` documents guarantees H1-H8 (one simulator per tick,
  bounded time, ...); H9-H15 are the orchestrator's. `HandoffOffer` has no token field and space-protocol
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
- naga must resolve >= 30 (`space-render/build.rs` asserts it). ShardIds are monotonic, never reused.
- Docker: `Dockerfile.orchestrator`, `Dockerfile.shard` and the local compose have been built and run
  (Docker 28); say plainly which files you did and did not execute when you touch them.
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
orchestrator + web UI `http://127.0.0.1:7980/` + one shard `127.0.0.1:7777`), `-Scene` (what the SERVERS
show), `-Client window|headless|stream`, `-Harness`, `-Script`, `-ServersOnly`, `-Stop`, `-PortOffset`.
`two-shards` is the manual seamless-handoff setup (shard 1 owns `64:0:0:0`, shard 2 `64:1:0:0`, `--peer` each
other); the automated version is `crates/space-client/tests/handoff.rs`.
Ports: 7777/7778 shards, 7900 orchestrator, 7980 web UI, 8097 viewer page, 8098 harness API, 8099 stream.
Everything else about running and looking at it: `skills/space-visual-test/SKILL.md`. Containers: `deploy/README.md`.
