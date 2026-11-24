# space

A space game engine in Rust: authoritative, shardable servers (QUIC) behind an
orchestrator, a Vulkan renderer and a native client, with a world model that
stays exact from metres to light-hours (`i64` cells + `f32` offsets).

The game is **online only**: the client always plays on a shard, and what it
shows — the *scene* — is chosen on the server side, for everyone at once.

| crate | role |
|---|---|
| `space-world`, `space-ecs`, `space-sim`, `space-protocol` | world/position model, ECS, deterministic simulation (ship flight and rigid-body physics), wire format (incl. the scene catalogue) |
| `space-scene` | what each scene is physically: static colliders, bodies, gravity, shape catalogue, scene anchor |
| `space-net`, `space-server`, `space-orchestrator` | QUIC transport, the shard, the shard map / autoscaler / scene owner with its built-in web UI |
| `space-render`, `space-client` | ash/Vulkan renderer (ray-traced shadows, textures, GLB models, 2D overlay), the client (window, headless, pixel stream) |
| `space-harness` | drive a client without a human: HTTP API for state, frames and input, sandboxed Lua test scripts |
| `space-stream`, `space-viewer` | JPEG-over-WebSocket streaming and the wasm browser viewer (a local development tool) |
| `space-mcp*`, `xtask` | MCP servers for agentic testing; CI driver and `xtask viewer` |

Working on it: [`AGENT.md`](AGENT.md) is the rulebook and the workflow (it
applies to people and to AI agents), [`.warren/skills/space-dev/SKILL.md`](.warren/skills/space-dev/SKILL.md)
has the invariants and CI, [`.warren/skills/space-visual-test/SKILL.md`](.warren/skills/space-visual-test/SKILL.md)
the visual test loop. Deployment: [`deploy/README.md`](deploy/README.md).

## Play

Needs a Vulkan GPU with `VK_KHR_acceleration_structure` (any ray-tracing
capable card; on Linux software Vulkan from Mesa 24.1 or newer also works).

Every push to `main` is released, so the newest prebuilt Windows client is
always
[`space-client.exe`](https://github.com/eliasstepanik/space/releases/latest/download/space-client.exe).
Run it with no arguments: a window opens on the **connect screen**. Type the
address of a shard (`host`, `host:port` or `[ipv6]:port`; the port defaults to
7777) and press Enter. The client and the servers must come from the same
release; a mismatch is refused with a readable message.

| | |
|---|---|
| `W` `A` `S` `D` / arrows | move along the view |
| `Space` / `Shift` | up / down |
| drag with the left mouse button | look around |
| `C` | chase camera / first person |
| `B` | brake assist (stops the ship when no movement key is held) |
| `F` | fast travel: the speed limit grows with the distance to the nearest surface (normal flight near anything: within 120 m, or 2 km of what you are flying at; the Moon is about half a minute away). Switching it off drops the ship to its normal 120 m/s |
| `H` | HUD on / off |
| `Esc` | back to the connect screen; on the connect screen: quit (window only) |

Other players appear as the placeholder character model. The HUD's `pos` is in
metres from the scene's spawn point, which every client derives from the shard
map, so all players agree on where things are.

## Run the whole thing locally

Needs the Rust toolchain from `rust-toolchain.toml`.

```powershell
.\scripts\dev.ps1                          # orchestrator + web UI + one shard + the window client
.\scripts\dev.ps1 -Topology two-shards     # two neighbouring shards: fly across the border
.\scripts\dev.ps1 -Client headless -Script scripts\harness\smoke.lua   # a scripted test run
```

```sh
bash scripts/dev.sh                        # the same on Linux
```

The scripts build the release binaries, start the server side in the
background (logs in `target/dev/`), run the client, and stop the servers when
the client ends. `Get-Help .\scripts\dev.ps1 -Full` (or the header of
`scripts/dev.sh`) lists the options.

By hand:

```powershell
cargo build --release -p space-client -p space-server -p space-orchestrator
.\target\release\space-orchestrator.exe --bind 127.0.0.1:7900 --state orchestrator.id --webui 127.0.0.1:7980
.\target\release\space-server.exe --orchestrator 127.0.0.1:7900 --bind 127.0.0.1:7777
.\target\release\space-client.exe --server 127.0.0.1:7777
```

### Scenes and the orchestrator web UI

Open `http://127.0.0.1:7980/`. The page is part of the orchestrator (server-
rendered, no JavaScript) and shows the cluster, the map, the shards, the
autoscaler and recent events; the same state is JSON at `/api/state`. From it
you can split and merge regions, pause the autoscaler and **switch the scene on
every shard**. Connected clients follow within a second.

| scene | |
|---|---|
| `demo-area` (default) | one place that shows everything the renderer does: textures, a mirror floor, glass, a PBR chart, coloured lights and shadows, models, the character, the IC-4 Kestrel on its apron (solid; its full model and LOD1 proxy are embedded in `space-client`, about 17 MB of its 31 MB) — and the physics: a gravity room (stacks, a ramp, a bowl, a seesaw, spring pendulums, a chain), a zero-g corner (floating crates, a spring pair, a door) and a planetoid with its own gravity, crates on it and balls in orbit. The area rides along with **Earth** (eight Earth radii ahead and above the spawn point): the Moon, the Sun and the planets move around it on their real orbits, sped up 8766× (a year per real hour, the Moon round in 4.5 min), at real size, each a solid ball you can land on, pulling with its real `GM` within a finite reach that ends far from the demo area |
| `materials` | PBR chart: gold, plastic and steel from mirror-smooth to rough |
| `galaxy` | the deterministic galaxy through the LOD selector |

(The old `solar` scene is part of `demo-area` now; its wire id 2 stays
reserved and resolves to the default, so an orchestrator whose saved scene was
`solar` comes back up showing the demo area.)

The planets move by **scene time** (UTC microseconds): every shard maps its
ticks onto the wall clock, re-anchors when it drifts more than 30 ms, and
tells its clients the mapping (`SceneClock`, protocol 6), so all shards and
clients agree on where the Moon is.

`space-orchestrator --scene NAME` sets the scene a new cluster starts with; a
scene chosen in the web UI is remembered next to the orchestrator's state file
and wins on restart. A shard run without an orchestrator takes
`space-server --scene NAME`.

The web UI is off unless `--webui ADDR` is given, needs no login on a loopback
address, and refuses to start on any other address without a token. See
[`deploy/README.md`](deploy/README.md) and [`docs/security-notes.md`](docs/security-notes.md).

### The client

```
space-client [--server ADDR] [--headless | --stream ADDR] [--size WxH]
             [--quality high|medium|low|ci] [--no-vsync] [--frames N] [--fps N]
             [--glb FILE [--glb-scale S]]
             [--harness ADDR] [--script FILE [--script-timeout S]] [--out-dir DIR]
```

- **window** (default), **`--headless`** (no window, renders offscreen) or
  **`--stream ADDR`** (headless, and streams the picture to the browser viewer).
- `--server ADDR` connects at once and keeps reconnecting (after 2, 3, 5, then
  every 8 s) when the shard is not there yet or goes away, until the address is
  edited or Esc is pressed; connection events are logged to stderr. Without it
  the connect screen is shown and waits, in every front end.
- A headless or streamed client never quits on user input (a viewer cannot take
  it down): it ends through the harness, `--frames`, a finished `--script` or a
  signal.
- `--glb FILE` shows a binary-glTF model near the spawn point (geometry, UVs,
  base-colour textures and material factors; only this client sees it).
  `cargo run -p space-render --example make_test_glb -- ship.glb` writes a
  sample (`--character` for the character model).

### Driving a client without a human

For tests and for AI agents: the picture and the controls over HTTP, and Lua
scripts that act as a user.

```sh
space-client --headless --harness 127.0.0.1:8098
curl http://127.0.0.1:8098/                          # the routes
curl http://127.0.0.1:8098/state                     # screen, scene, position, speed, fps, gravity, bodies ...
curl -o frame.png http://127.0.0.1:8098/frame.png    # or frame.jpg (much smaller)
curl -X POST http://127.0.0.1:8098/connect -d '{"address":"127.0.0.1:7777"}'

space-client --headless --server 127.0.0.1:7777 --script scripts/harness/smoke.lua   # exit code = result
space-client --headless --server 127.0.0.1:7777 --script scripts/harness/physics.lua --script-timeout 300
```

The harness is loopback-only without credentials; any other address needs a
bearer token in `SPACE_HARNESS_TOKEN`. Details in
[`.warren/skills/space-visual-test/SKILL.md`](.warren/skills/space-visual-test/SKILL.md).

### In a browser

```sh
space-client --stream 127.0.0.1:8099 --server 127.0.0.1:7777
cargo run -p xtask -- viewer            # builds the wasm page and serves http://127.0.0.1:8097/
```

The viewer page is a local development tool; it is not deployed.

### Shard transitions

With `-Topology two-shards` the ship spawns 512 m from the border between two
shards. Fly across (`+X`: hold `D`, about 15 s) and the first shard hands the
ship to its neighbour: the client opens the second connection, the shards
commit, and the HUD changes to `shard 2  handoffs 1` without the view jumping.

Physics bodies cross the same way, with nobody to redirect: near a border each
shard shows its neighbour's bodies as ghosts, bodies that may touch are moved
onto one shard before they do, and a body that leaves a region is handed to
the next (the rules: `crates/space-server/src/border.rs`).

## Test

```powershell
cargo run -p xtask -- ci        # fmt, clippy -D warnings, every test
$env:SPACE_REQUIRE_GPU = '1'    # make the GPU tests fail instead of skip without a device
$env:SPACE_RENDER_VALIDATION = '1'   # run them under the Khronos validation layer
cargo test -p space-render -p space-client -- --test-threads=1
```

Then look at it: `AGENT.md` requires every visible change to be checked on a
running stack, by looking at real frames.
