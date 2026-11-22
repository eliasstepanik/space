---
name: space-dev
description: Use when developing, testing, packaging or debugging the space engine (the Rust workspace with space-world/ecs/protocol/sim/net/server/orchestrator/render/client/stream/viewer/mcp crates) - building, running CI, GPU/lavapipe tests, wire-protocol or handoff changes, Docker/CI files, or driving a shard/client through the MCP servers.
---

# space-dev

Rust workspace, toolchain pinned in `rust-toolchain.toml` (1.100.0 + rustfmt, clippy,
`wasm32-unknown-unknown`). All crates are 0.1.0. Docs: `docs/plan.md` (phases),
`docs/plan-phase3.md`, `docs/security-notes.md`, `spikes/ENVIRONMENT.md` (GPU env),
`deploy/README.md` (what is/isn't verified).

## Layout
| crate | role |
|---|---|
| `space-world` | i64-cell + f32 position model (1024 m cells), octree, deterministic galaxy, LOD selector |
| `space-ecs` | custom archetype ECS |
| `space-protocol` | wire messages, codec, `PROTOCOL_VERSION`, `Message::VARIANT_COUNT` |
| `space-sim` | deterministic physics/orbits/prediction, `SimWorld` |
| `space-net` | quinn QUIC transport (self-signed rustls, `ring`), hello/handshake |
| `space-server` | authoritative shard (`space-server` bin), `handoff.rs` seamless handoff |
| `space-orchestrator` | epoch-versioned shard map, autoscaler, ShardId counter, `--launcher local` |
| `space-render` | ash/Vulkan renderer, naga WGSL->SPIR-V in `build.rs`, ray-query GI, presets |
| `space-client` | native client: offline `--demo/--galaxy`, window, `--stream ADDR [--server ADDR]`, `link.rs` |
| `space-stream` | JPEG frame + input wire format; feature `net` = ws:// server (tokio-tungstenite, no TLS) |
| `space-viewer` | wasm cdylib browser viewer (canvas + WebSocket), `index.html?ws=ws://host:8099` |
| `space-mcp` | hand-rolled JSON-RPC MCP core: `--stdio` or `--http addr` (HTTP/1.1 + SSE on tokio) |
| `space-mcp-server` | MCP server booting an in-process shard |
| `space-mcp-client` | MCP server acting as a player + offscreen frame capture |
| `xtask` | `ci`, `lint`, `gpu-setup`; `e2e/load/parity/web-smoke/accept` are stubs (exit 2) |

## CI (run before declaring done; network is fine, no `--offline`)
```sh
TMPDIR=/workspace/.warren/tmp RUSTUP_HOME=/workspace/.warren/rustup RUSTUP_TOOLCHAIN=stable \
  cargo run -p xtask -- ci
```
= `cargo fmt --check`; `cargo clippy --workspace --all-targets --locked -- -D warnings`;
`cargo nextest run --workspace --locked` (falls back to `cargo test` without nextest;
test target dir `target/xtask-test`). Baseline at Phase 8: 394 passed, 0 failed.
Wasm build (also a CI job):
`cargo build --release --locked --target wasm32-unknown-unknown -p space-viewer`.
Lockfile edits need a non-`--locked` cargo command first (e.g. `cargo metadata`).

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
layer is then a hard failure; the CI GPU step is `continue-on-error: true` as it is unproven on hosted runners).
Do not run GPU tests in parallel (`--test-threads=1`).

## Invariants - do not break
- **Wire gate**: `PROTOCOL_VERSION` (`space-protocol/src/codec.rs`, now 3) is checked in every hello
  (`check_hello`). `Message::VARIANT_COUNT` (25) = one past the highest tag; only APPEND variants,
  bump the count, add the arm to `wire_tag()`; `message_tags_are_dense_and_covered` and
  `unknown_tag_rejected` (acceptance.rs) pin it. Changing existing layouts means bumping `PROTOCOL_VERSION`.
- **Handoff freeze**: `space-server/src/handoff.rs` documents guarantees H1-H8 (one simulator per tick,
  bounded time, ...); H9-H15 are the orchestrator's. `HandoffOffer` has no token field and space-protocol
  was frozen for handoff - the owner token rides inside the state payload. Don't change the three-phase
  offer/accept/commit exchange without updating the guarantee table and its tests.
- **`#![forbid(unsafe_code)]`** in every library crate except `space-render` (ash FFI) and
  `space-client` (lib; GPU/window glue). Keep new crates forbidding it.
- **No C dependencies**, pure Rust only. Single sanctioned exception: `ring` (via quinn/rustls/rcgen),
  documented in root `Cargo.toml` and `space-net/src/crypto.rs`. Adding any other C/`-sys` dep needs sign-off.
- **ws:// only, no TLS** on the stream and MCP paths: `tokio-tungstenite` has default features off
  (`handshake` only). `wss://` would need `ring`; use a TLS proxy. Peer auth gaps are in `docs/security-notes.md`.
- naga must resolve >= 30 (`space-render/build.rs` asserts it). ShardIds are monotonic, never reused.
- Docker files are **not** built in the sandbox (no daemon): correct-by-inspection, say so.
- **Pipeline** (details: `deploy/README.md`, `deploy/proxy/README.md`): merge to `main` -> `release-images.yml`
  pushes `ghcr.io/eliasstepanik/space-{shard,orchestrator,client}` as `:latest` + `:sha-<short>`; tag `v*` ->
  `release.yml` attaches `space-client`, `space-client.exe`, `space-viewer.zip` to a GitHub Release; the proxy host
  runs Watchtower (label opt-in, 60s poll, stack in `/home/proxy/stacks/space`); `deploy.yml` is the manual SSH path (`redeploy.sh`)
  (secrets `DEPLOY_SSH_HOST/USER/KEY`). ci.yml's `lint` job parses all workflow YAML (`.github/scripts/check_yaml.py`):
  keep heredoc/script bodies out of `run: |` blocks (use `.github/scripts/`) and re-run that check after any workflow edit.

## Phase map
0 risk-gate spikes (`spikes/`); 1 world/ecs/protocol/sim; 2 QUIC netcode, headless multiplayer, handoff H1-H8;
3 orchestrator + autoscaler + H9-H15 (`--launcher docker` deferred); 4 ash renderer core (headless, GPU tests);
5 ray-query GI/SSR/fog/refraction + quality presets (`--quality high|medium|low|ci`);
6 pixel streaming: `space-stream`, `space-client --stream`, wasm `space-viewer` (JPEG only);
7 MCP servers; 8 packaging: Docker images, compose `stream` profile, CI wasm + docker jobs, this skill, v0.1.0.

## MCP servers (agentic testing)
Both: `--stdio` (default) or `--http 127.0.0.1:PORT`.
- `cargo run -p space-mcp-server -- [--bind 127.0.0.1:0] [--no-tick]`: real shard behind real QUIC at 30 Hz.
  Tools: `shard_status` (reports the QUIC addr), `topology`, `list_ships`, `spawn_ship`, `ship_state`,
  `despawn_ship`, `advance_ticks` (deterministic stepping; use with `--no-tick`). No split/merge (orchestrator's job).
- `cargo run -p space-mcp-client -- [--out-dir DIR]`: a player. Tools: `connect` (host:port from `shard_status`),
  `disconnect`, `client_status`, `send_input` (thrust, optional duration), `capture_frame` (offscreen render,
  base64 JPEG or file; works on lavapipe).
Typical loop: start server, `shard_status` -> addr; client `connect`; `send_input`; `advance_ticks`;
`ship_state` vs `client_status` to check replication; `capture_frame` to inspect pixels.

## Run the stack by hand
`cargo run -p space-client -- --stream 127.0.0.1:8099 --galaxy` (add `--server HOST:PORT` to play on a shard,
`--frames N` to stop); serve `crates/space-viewer` statically and open `index.html?ws=ws://127.0.0.1:8099`
(build steps: `crates/space-viewer/README.md`). Containers: `deploy/README.md`.
