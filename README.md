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
applies to people and to AI agents), [`skills/space-dev/SKILL.md`](skills/space-dev/SKILL.md)
has the invariants and CI, [`skills/space-visual-test/SKILL.md`](skills/space-visual-test/SKILL.md)
the visual test loop. Deployment: [`deploy/README.md`](deploy/README.md).

## Play

Needs a Vulkan GPU with `VK_KHR_acceleration_structure` (any ray-tracing
capable card; on Linux software Vulkan from Mesa 24.1 or newer also works).

Every push to `main` is released, so the newest prebuilt Windows client is
always
[`space-client.exe`](https://github.com/eliasstepanik/space/releases/latest/download/space-client.exe).
Run it with no arguments: a window opens on the **connect screen**. Type the
server's address (`host`, `host:port` or `[ipv6]:port`; the port defaults to
7777) and press Enter. That one address is all a player needs: a cluster's is
its orchestrator's **player port**, which sends the client on to the shard a
new player starts on (a single shard's own address works the same way). The
client and the servers must come from the same release; a mismatch is refused
with a readable message.

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
.\scripts\dev.ps1                          # orchestrator (player port, web UI) + the four shards it starts + the window client
.\scripts\dev.ps1 -MinShards 1             # the same on one shard: no borders anywhere
.\scripts\dev.ps1 -Topology two-shards     # two neighbouring shards: fly across the border
.\scripts\dev.ps1 -Client headless -Script scripts\harness\smoke.lua   # a scripted test run
```

```sh
bash scripts/dev.sh                        # the same on Linux
bash scripts/dev.sh --min-shards 1
```

The scripts build the release binaries, start the server side in the
background (logs in `target/dev/`), run the client, and stop the servers when
the client ends. `Get-Help .\scripts\dev.ps1 -Full` (or the header of
`scripts/dev.sh`) lists the options.

The scripts start **four shards** behind the one address (`-MinShards` /
`--min-shards`, 1 to 8): the demo area lies in four shard areas that meet at
`pos` x = -6 / z = -4, and what a player does there — walk across, push a
crate over a border, knock a stack over it — is what it would be on one
shard. That is the scripts' default only: `space-orchestrator` itself starts
no shard unless told to (`--min-shards`, default 0). The compose files run
the orchestrator alone with `--launcher docker --min-shards 4`: it starts its
shards as containers ([`deploy/README.md`](deploy/README.md)). The dev scripts
do not need Docker.

By hand:

```powershell
cargo build --release -p space-client -p space-server -p space-orchestrator
.\target\release\space-orchestrator.exe --bind 127.0.0.1:7900 --player-bind 127.0.0.1:7777 --state orchestrator.id --webui 127.0.0.1:7980
.\target\release\space-server.exe --orchestrator 127.0.0.1:7900 --bind 127.0.0.1:7870
.\target\release\space-client.exe --server 127.0.0.1:7777
```

### One address

A client connects to **one address**: the orchestrator's player port
(`--player-bind`, off unless given; 7777 in the dev scripts). It answers the
client's hello with the shard that owns the scene's spawn point and closes;
the client dials that shard and plays there — game traffic never passes
through the orchestrator. While no shard owns the spawn point yet (the cluster
is starting, or that shard is being replaced) the client is told so and tries
again. A shard that is dialed directly does the same thing when the spawn
point is not its own, so every address leads to the same place: **a new player
always starts on the spawn point.** A client started with `--server` that
loses its shard comes back through that address by itself.

Where a shard's players reach it is part of the shard map: `space-server
--client-endpoint ADDR`, and `0.0.0.0:<port>` there means "the host the player
dialed, this port" — which is what the orchestrator gives the shards it starts,
so no public host name is configured anywhere. The player port is the public
entry and needs no token; the control port (`--bind`) is for shards only.

### Shards

A shard starts without a region and is given one when it registers with the
orchestrator: the first gets the root (the whole world, and the scene's spawn
point), each further one the next *layout cell* beside the spawn point — so
four shards are the four quadrants of the demo area, meeting at `pos` x = -6 /
z = -4 — up to `--max-shards` (default 8). Start more by hand with the same
`space-server --orchestrator ...` line and another `--bind`, or let the
orchestrator do it:

```powershell
.\target\release\space-orchestrator.exe --player-bind 127.0.0.1:7777 --webui 127.0.0.1:7980 --launcher process --min-shards 4
```

`--launcher process` starts the `space-server` lying next to the orchestrator,
one process per shard on a port from `--host-port-range` (default 7870-7899;
keep it out of the operating system's dynamic port range), logging to
`shard-<port>.log` in `--shard-log-dir`; `launched-shards.pid` there lists the
pids of the shards that are running, and nothing else. `--min-shards` can be 8
at most (the root and the seven layout cells) and not more than `--max-shards`
or `--max-regions`. A shard that dies — its process exits, its connection to
the orchestrator closes, or it stops acknowledging map changes — is replaced:
its region goes to a new shard under a new id and the scene's bodies are built
again. The replacement listens on the next free port of the range; a client
that was on the dead shard and was started with `--server <player port>` comes
back through the player port and is sent there (it starts on the spawn point
again: where a player was is not kept). A shard that loses its orchestrator
exits by itself. The web UI and `/api/state` show the player port, which shard
it sends players to, and where each shard's players dial it.

`--launcher docker` does the same with **one container per shard**, through
the Docker Engine's socket (Linux only; on Windows the orchestrator refuses to
start with it and names `--launcher process`): created from `--shard-image`
(the orchestrator image names the shard image of its own commit), named
`space-shard-<port>`, its UDP port published on the host under the same
number, the cluster secret in its environment, no mount, no capability and no
restart policy; removed when it is no longer needed, when the orchestrator is
told to stop (SIGTERM, Ctrl-C), and — what an orchestrator that was killed
left behind — at the next start. Both compose files hold the orchestrator
alone. **Whoever holds the Docker socket is root on the host**: read
[`docs/security-notes.md`](docs/security-notes.md), "The Docker socket",
before mounting it anywhere.

**Scaling.** Every shard reports its load once a second: players, the ships
and bodies it simulates, its tick times, and its *hot cell* — the 1024 m cell
holding most of its players. Load is `max(players / capacity, tick p95 /
33.3 ms)`. A shard at 0.8 or more for 10 s has its hot cell carved for a new
shard, up to `--max-shards`. A shard that, together with the shard it would
merge into, stays at 0.35 or less for 60 s is merged away: its region goes
back, it **drains** — hands its ships and bodies to the shards that own the
ground now, by the ordinary handoff, its players included — and is told to
stop when it reports nothing left (it exits 0), or stopped as it is after
`--drain-timeout`. The autoscaler never removes the shard that owns the spawn
point nor the layout cells `--min-shards` stands for, never goes below
`--min-shards`, and leaves a shard alone for 30 s after any change that
involved it. A crowd inside one cell cannot be split: a region is never
smaller than a cell, and the event log says so. Every number is a flag
(`--shard-capacity`, `--scale-out-load`, `--scale-out-after`,
`--scale-in-load`, `--scale-in-after`, `--scale-cooldown`, `--drain-timeout`,
each with a `SPACE_ORCHESTRATOR_...` variable), so a test can make a cluster
scale with two clients; the rules and what was measured are in
[`docs/plan-sharding.md`](docs/plan-sharding.md) §3.

The orchestrator's control port takes any shard on a loopback address. On any
other address a shard needs the **cluster secret** (`SPACE_CLUSTER_SECRET` on
both, or `--cluster-secret` / `--cluster-secret-file` on the orchestrator), and
the orchestrator refuses to start there without one. A connection whose hello
is refused is closed; one that is welcomed speaks for its own shard only.
Shards of one cluster prove themselves to each other with a **peer token** the
orchestrator hands out: generated at start, or given with
`SPACE_ORCHESTRATOR_PEER_TOKEN` / `--peer-token-file` / `--peer-token` (at
least 16 characters). Neither secret is printed, and a shard the orchestrator
starts does not inherit the orchestrator's own settings
(`SPACE_ORCHESTRATOR_*`, the web UI token among them).

### Scenes and the orchestrator web UI

Open `http://127.0.0.1:7980/`. The page is part of the orchestrator (server-
rendered, no JavaScript) and shows the cluster, the map, the shards with their
load reports, the autoscaler with its settings and recent events; the same
state is JSON at `/api/state`. From it
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
- `--server ADDR` connects at once and keeps reconnecting **to that address**
  (after 2, 3, 5, then every 8 s) when nothing is there yet or the shard goes
  away, until the address is edited or Esc is pressed; connection events are
  logged to stderr. Without it the connect screen is shown and waits, in every
  front end. `ADDR` may be a shard or an orchestrator's player port: the
  client follows up to two redirects to the shard it plays on, and the harness
  `/state` says both (`address`, `shard_address`).
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
curl http://127.0.0.1:8098/state                     # screen, address, shard, shard_address, scene, position, speed, fps, gravity, bodies ...
curl -o frame.png http://127.0.0.1:8098/frame.png    # or frame.jpg (much smaller)
curl -X POST http://127.0.0.1:8098/connect -d '{"address":"127.0.0.1:7777"}'

space-client --headless --server 127.0.0.1:7777 --script scripts/harness/smoke.lua   # exit code = result
space-client --headless --server 127.0.0.1:7777 --script scripts/harness/physics.lua --script-timeout 300
```

The harness is loopback-only without credentials; any other address needs a
bearer token in `SPACE_HARNESS_TOKEN`. Details in
[`skills/space-visual-test/SKILL.md`](skills/space-visual-test/SKILL.md).

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

The shards of a cluster count the same ticks — from the cluster's tick
origin, which the orchestrator hands every shard it welcomes and a standalone
shard takes from `--tick-origin` (both default to one fixed instant; one that
lies in the future is refused) — so a
tick number names the same instant on every shard. The accepting shard
installs the ship on the tick its committed state describes, with the inputs
the client has been sending to both shards since it was redirected, and the
client's clock, prediction and view carry on: on the four shard areas of
`-MinShards 4` the borders at `pos` x = -6 and z = -4 show in the HUD's shard
number (about a second after the ship is a metre across) and in nothing else.
Standing on a border or walking along it is no handoff, and a player is
handed over for where it is, not because another flew past.

Physics bodies cross the same way, with nobody to redirect: near a border each
shard shows its neighbour's bodies as ghosts, bodies that may touch — that are
close, or moving *towards* each other — are moved onto one shard before they
do, and a body that leaves a region is handed to the next (the rules:
`crates/space-server/src/border.rs`). A body changes shard for where it is or
for what it is about to touch, not because a player is near or flies past.
Against the same scene on one shard (`crates/space-server/tests/four_cells.rs`):
a body handed over in free flight goes on bit-identically; one handed over in
contact — sliding, resting in a stack — within a millimetre, because the
contact solver starts afresh on the shard that takes it.

A border makes no difference to what a player **sees** either: every ship and
body within 10 km is in the picture, whichever shard simulates it. What lies
beyond the reach of a neighbour's ghosts (64 m from a border) arrives as a
**far replica**: the shard that simulates it tells the other shards, each
relays it to its own clients — a client keeps one play connection — and
nothing collides with it. How often something's motion is sent falls with its
distance, for far replicas and for everything else: every tick within 100 m,
then 10, 5 and 1 times a second out to 250 m, 1 km and 10 km; what does not
move is not sent. The client draws distant things a little further in the
past and joins their sparse samples along their velocities
(`crates/space-protocol/src/far.rs`, `crates/space-server/src/far.rs`). The
harness says which copy of a thing a client holds: `source` (`local`, `ghost`,
`far`), `step` and `sample_age_s` per body and per other player in `/state`.

## Test

```powershell
cargo run -p xtask -- ci        # fmt, clippy -D warnings, every test
$env:SPACE_REQUIRE_GPU = '1'    # make the GPU tests fail instead of skip without a device
$env:SPACE_RENDER_VALIDATION = '1'   # run them under the Khronos validation layer
cargo test -p space-render -p space-client -- --test-threads=1
```

Then look at it: `AGENT.md` requires every visible change to be checked on a
running stack, by looking at real frames.
