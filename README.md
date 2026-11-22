# space

A space game engine in Rust: an authoritative, shardable server (QUIC), a
Vulkan renderer and a native client, with a world model that stays exact from
metres to light-hours (`i64` cells + `f32` offsets).

| crate | role |
|---|---|
| `space-world`, `space-ecs`, `space-sim`, `space-protocol` | world/position model, ECS, deterministic simulation, wire format |
| `space-net`, `space-server`, `space-orchestrator` | QUIC transport, the shard, the shard map / autoscaler |
| `space-render`, `space-client` | ash/Vulkan renderer, the native client (window, offline render, pixel stream) |
| `space-stream`, `space-viewer` | JPEG-over-WebSocket streaming and the wasm browser viewer |
| `space-mcp*`, `xtask` | MCP servers for agentic testing, CI driver |

Developer notes (invariants, CI, GPU tests, MCP) live in
[`skills/space-dev/SKILL.md`](skills/space-dev/SKILL.md); deployment in
[`deploy/README.md`](deploy/README.md).

## Run it (Windows)

Needs the Rust toolchain from `rust-toolchain.toml` and a Vulkan GPU with
`VK_KHR_acceleration_structure` (any ray-tracing capable card).

```powershell
.\scripts\dev.ps1                                   # one shard + the window client
.\scripts\dev.ps1 -Topology two-shards              # two neighbouring shards: fly across the border
.\scripts\dev.ps1 -Topology orchestrated            # orchestrator + one shard it places
.\scripts\dev.ps1 -Topology offline -Scene solar    # no server, real-size solar system
```

The script builds the release binaries, starts the server side in the
background (logs in `target\dev\`), runs the client, and stops the servers
when the client window closes. `Get-Help .\scripts\dev.ps1 -Full` lists the
options.

By hand:

```powershell
cargo build --release -p space-client -p space-server
.\target\release\space-server.exe --shard 1 --region 64:0:0:0          # 127.0.0.1:7777
.\target\release\space-client.exe --window --galaxy --server 127.0.0.1:7777
```

### The client

`space-client` with no arguments opens a window on the demo scene.

| | |
|---|---|
| `W` `A` `S` `D` / arrows | move (online: thrust the ship) |
| `Space` / `Shift` | up / down |
| drag with the left mouse button | look around |
| mouse wheel | flying speed (offline) |
| `Esc` or `Q` | quit |

Scenes — a scene flag on its own renders one frame to a PPM file and exits;
add `--window` to fly around it:

| flag | scene |
|---|---|
| `--demo` | a sphere and a cube (the scene the GPU tests assert on) |
| `--galaxy` | the deterministic galaxy through the LOD selector |
| `--materials` | PBR chart: gold, plastic and steel from mirror-smooth to rough, a mirror floor, glass, a lamp |
| `--solar` | the solar system at real size and distance (coloured spheres); you start just off Earth |

Other flags: `--server ADDR` (join a shard as a player), `--glb FILE` (show a
binary-glTF model in front of the spawn point; `--glb-scale S`), `--size WxH`,
`--quality high|medium|low|ci`, `--no-vsync`, `--frames N`, `--out FILE`,
`--stream ADDR` (serve the view to the browser viewer instead of a window).
The title bar shows frame rate, scene, speed or the shard you are on, and — in
the solar system — the distance to the nearest body.

A sample model without any external asset:

```powershell
cargo run -p space-render --example make_test_glb -- ship.glb
.\target\release\space-client.exe --window --solar --glb ship.glb
```

GLB support covers geometry (positions, normals, node hierarchy) and material
*factors* (base colour, metallic, roughness, emissive); textures, skins and
animations are ignored, compressed/quantised geometry is rejected.

### Shard transitions

With `-Topology two-shards` the ship spawns 512 m from the border between two
shards. Fly across (`+X`: hold `D` in the demo scene, `A` in the galaxy scene,
about 15 s at full thrust) and the first shard hands the ship to its
neighbour: the client opens the second connection, the shards commit, and the
title changes to `shard 2 after 1 handoff(s)` without the view jumping.

## Test

```powershell
cargo run -p xtask -- ci        # fmt, clippy -D warnings, every test
$env:SPACE_REQUIRE_GPU = '1'    # make the GPU tests fail instead of skip without a device
$env:SPACE_RENDER_VALIDATION = '1'   # run them under the Khronos validation layer
cargo test -p space-render -p space-client -- --test-threads=1
```
