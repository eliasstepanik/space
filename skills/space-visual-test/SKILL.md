---
name: space-visual-test
description: Use whenever a change to the space engine can be seen or operated - renderer, client, scenes, models, camera, HUD, connect screen, shard handoff, orchestrator web UI, scene switching - and whenever asked to test, demo or look at the game. Covers starting the real stack (orchestrator + shards + client), driving a headless client through the harness HTTP API and Lua scripts, using the browser viewer, capturing frames and judging them.
---

# space-visual-test

`AGENT.md` makes this loop mandatory: **what can be seen is verified by looking
at real frames from a real running stack**. Tests passing is not enough.

The client is online only. To see anything you need a shard; to change the
scene you need the orchestrator. So the unit of visual testing is always:

```
orchestrator (player port, web UI) ── shard(s) ── client ── you (harness API / Lua / browser viewer)
```

A client is given **one address**, the orchestrator's player port. The
orchestrator sends it on to the shard that owns the scene's spawn point; from
then on the client talks to shards only.

Everything below was run on Windows 11 (RTX 4080) and on Linux (WSL Ubuntu
24.04, Mesa 25 lavapipe, no GPU, no display).

## 1. Start the stack

```sh
# Linux (and Git Bash)                      # Windows PowerShell
bash scripts/dev.sh --servers-only          .\scripts\dev.ps1 -ServersOnly
bash scripts/dev.sh --stop                  .\scripts\dev.ps1 -Stop
```

That builds the release binaries and starts, in the background:

| what | where |
|---|---|
| orchestrator | **player port `127.0.0.1:7777`** (QUIC) — the address a client connects to; control port `127.0.0.1:7900` (QUIC, shards only); **web UI `http://127.0.0.1:7980/`**, JSON at `/api/state` |
| shard(s) | started by the orchestrator, one `space-server` process each on ports from `7870`; a client is sent to them, it is never pointed at one |
| logs | `target/dev/*.log`, the shards' in `target/dev/shard-<port>.log` |

- The orchestrator always starts its shards itself (`space-orchestrator
  --player-bind … --launcher process --min-shards N`), **four by default**:
  the demo area in four shard areas that meet at HUD `pos` x = -6 / z = -4
  (`--min-shards 1` / `-MinShards 1` is one shard that owns everything — the
  thing to compare with when something looks wrong at a border; up to 8).
  The script waits until every
  shard owns its region and the player port has a shard to send a client to
  (about a second), and says which shard that is. `--stop` stops those shards
  too: it reads their pids from `target/dev/launched-shards.pid` — the
  orchestrator's own list of the shards it has running — and stops each only
  if its executable lies in this checkout's `target/release` (a pid may belong
  to something else by now). At most 8: the orchestrator refuses a
  `--min-shards` it cannot place.
- **Every address leads to the spawn point.** The player port answers a
  client's hello with the shard that owns the spawn point; a shard that is
  dialed directly and does not own it does the same. While no shard owns it
  (the cluster is starting, the root's shard is being replaced) the client is
  told "the cluster is starting" and, started with `--server`, tries again.
- `--topology single` is one shard with no orchestrator (scene from
  `--scene`), on `127.0.0.1:7777` itself; `--topology two-shards` is the
  seamless-handoff setup (fly +X, hold D about 15 s, the HUD goes to
  `shard 2  handoffs 1`).
- `--port-offset N` moves every port, for a second stack or when the defaults
  are taken. `--no-build` reuses the binaries. Both scripts honour
  `CARGO_TARGET_DIR`.
- The orchestrator remembers the scene an operator chose in
  `target/dev/orchestrator.id.scene`. Passing `--scene NAME` forgets it.
- PowerShell: do **not** pipe `dev.ps1 -ServersOnly` into another command; the
  background servers keep the pipe open and the command never returns.

Without the scripts:

```sh
space-orchestrator --bind 127.0.0.1:7900 --player-bind 127.0.0.1:7777 --state orchestrator.id --webui 127.0.0.1:7980
space-server --orchestrator 127.0.0.1:7900 --bind 127.0.0.1:7870
# a second, third ... shard is given the next layout cell beside the spawn point:
space-server --orchestrator 127.0.0.1:7900 --bind 127.0.0.1:7871
space-client --server 127.0.0.1:7777          # the player port; never a shard's address
```

A shard names neither id nor region: the orchestrator gives it both (the first
one the root, and with it the spawn point). Clients connect to the player
port; `--player-bind` is off unless given, and then a client has to dial a
shard — any of them: it is sent to the one that owns the spawn point.

### In containers (Docker)

The dev scripts do not need Docker (they use `--launcher process`). The
compose stack is the other way to run the same cluster: **the orchestrator
alone**, which creates one container per shard through the Docker socket
(`--launcher docker`; a Linux engine — Docker Desktop's counts).

```sh
docker build -f deploy/Dockerfile.shard -t space/shard:dev .       # once, and after every change: the image shards are started from
export SPACE_CLUSTER_SECRET="$(openssl rand -hex 24)"              # required, no default
docker compose -f deploy/docker-compose.yaml up --build -d         # builds and starts the orchestrator only
docker ps                                                          # the orchestrator and four space-shard-<port>
space-client --headless --harness 127.0.0.1:8098 --server 127.0.0.1:7777   # a client on the host: the published player port
docker compose -f deploy/docker-compose.yaml down                  # SIGTERM: the orchestrator removes its shard containers, then exits 0
```

- Without the shard image the launches fail and the orchestrator's log says
  so (`space/shard:dev` is in no registry to pull from).
- Ports already taken, or a second stack: `SPACE_PLAYER_PORT`,
  `SPACE_SHARD_PORTS` (a range, e.g. `8170-8199`), `WEBUI_PORT`, and a compose
  project name (`-p`). The web UI is on the host's loopback behind a token
  (any user name, password `space-local-dev-token` unless `SPACE_WEBUI_TOKEN`
  is set): `curl -u x:space-local-dev-token http://127.0.0.1:7980/api/state`.
- A client on the host dials the **published player port** and is sent on to
  a shard container's published port (`shard_address` in `/state` is
  `127.0.0.1:<a shard port>`). A shard's log is `docker logs
  space-shard-<port>`.
- To watch it scale: `SPACE_SHARD_CAPACITY`, `SPACE_SCALE_OUT_AFTER`,
  `SPACE_SCALE_IN_AFTER`, `SPACE_SCALE_COOLDOWN` (section 4, "Scaling").
- **Known on Docker Desktop:** its VM's wall clock is stepped by the
  hypervisor, the shards' census lines show `clock offset` up to tens of
  milliseconds (-84 ms was seen), and while two shards slew by themselves a
  crossing between them can step by decimetres. A process stack on the same
  machine shows 0.0 ms. Judge crossings on the process stack; on the
  containers say what the clock offsets were.
- The Docker socket makes the orchestrator's container root-equivalent on the
  host (`docs/security-notes.md`, "The Docker socket"): run this on your own
  machine, and take it down when you are done (`down -v` also removes the
  state volume).

## 2. Drive a client

### a) Headless + harness API — the default for agents

No window, no display, works on software Vulkan.

```sh
space-client --headless --harness 127.0.0.1:8098            # starts on the connect screen
curl http://127.0.0.1:8098/                                  # the route list
curl http://127.0.0.1:8098/state                             # screen, address, shard, shard_address, position, speed, fps ...
curl -o frame.png http://127.0.0.1:8098/frame.png            # what the player sees now (or /frame.jpg)
curl -X POST http://127.0.0.1:8098/input -d '[{"text":"7"},{"press":"backspace"},{"press":"enter"}]'
curl -X POST http://127.0.0.1:8098/connect -d '{"address":"127.0.0.1:7777"}'
curl -X POST http://127.0.0.1:8098/input -d '[{"press":"w","ms":1000},{"turn":[25,-5]}]'
curl -X POST http://127.0.0.1:8098/disconnect
curl -X POST http://127.0.0.1:8098/quit
```

`screen` in `/state` is `connect`, `connecting`, `online` or `error` (with the
reason in `error`; `retry_in_s` counts down when a `--server` client is
retrying). `address` is the **entry** address — what was typed or given with
`--server`, usually the orchestrator's player port — and stays that for the
whole session; `shard_address` is the `ip:port` of the shard actually being
played on (`null` while not on one), and `shard` its id: the two change with a
shard transition and when the client comes back after its shard died, the
entry address never does. While playing, `scene` and `scene_revision` say what the shard
shows, `position` is the absolute world position and `scene_position` the same
in metres from the scene's spawn point (what the HUD prints). The physics:
`gravity` (m/s² at the ship: 9.81 in the demo area's gravity room, 0 in
weightless space), `landed` (standing on something in gravity), `bodies` (the
physics bodies drawn, nearest first: `guid`, `visual` — `crate`, `ball`,
`plank`, `door`, `chain`, `pendulum` — `position` in the frame of
`scene_position`, `distance`, `speed`, `parts`, `in_view`, and which copy of
it this is: `source`, `step`, `sample_age_s`, below), `bodies_in_view`,
and `correction_m` / `correction_max_m` (how far the client's prediction of
its own ship was from the shard's word). **What is drawn** is `drawn_position`
(the frame of `scene_position`), `drawn_velocity` and `drawn_tick`: the own
ship as the last frame showed it — predicted ahead of the shard and
interpolated between ticks, so it moves every frame, where `scene_position`
is the shard's word and steps thirty times a second — and the fractional tick
it was drawn at; `others_drawn` counts the other players and ships in the
frame and `others` lists them (`guid`, `player`, `position`, `distance`,
`source`, `step`, `sample_age_s`). Look at these, not at `scene_position`,
for a step or a rubber band (section 4, "Seamless crossing").
**Which copy of a thing the client holds**: `source` is `local` (the shard
being played on simulates it), `ghost` (a neighbour does, within 64 m of a
border: the shard's ghost of it) or `far` (another shard does, further off: a
far replica, relayed for seeing only); `step` is its distance step, 0 to 3
(its motion is sent 30, 10, 5, 1 times a second out to 100 m, 250 m, 1 km,
10 km from the ship — whatever its source); `sample_age_s` is how old its
newest sample is (the step's spacing at most while it moves, a few ticks
more for a far replica; it **grows while a thing is at rest**: nothing is
sent for what does not change); `far_drawn` counts the far replicas in the
frame, bodies and ships. `?wait=1` on a
frame blocks until the next one. `frame.jpg` is small and fine for looking;
`frame.png` is lossless (and large) for pixel-exact checks. Input is
the same vocabulary a person has: keys (`w a s d space shift`, `enter backspace
escape tab delete home end caret_left caret_right`), typed text, view turns in
degrees. While playing, `c` (camera), `b` (brake assist), `f` (fast travel)
and `h` (HUD) are typed: `{"text":"c"}`. `/state` also has `fast` (fast travel
on), `nearest` and `nearest_surface_m` (the solar-system body whose surface is
nearest: `"Earth"`, `"Moon"`, ...; the demo area only).

Then **open the picture** (the Read tool shows PNG/JPEG) and look at it.

### b) Lua scripts — a repeatable user

A script is a test: a failing `assert` makes the client exit 1 with
`file:line: message`.

```sh
space-client --headless --server 127.0.0.1:7777 --script scripts/harness/smoke.lua --out-dir shots
bash scripts/dev.sh --client headless --script scripts/harness/smoke.lua      # stack + client + script, then stop
.\scripts\dev.ps1 -Client headless -Script scripts\harness\smoke.lua
curl -X POST http://127.0.0.1:8098/script --data-binary @my-test.lua           # into a running client; JSON report back
```

`scripts/harness/smoke.lua` is the model: it reaches the connect screen, types
the address, connects, holds W, checks the ship moved, turns the view, takes
screenshots, leaves. `scripts/harness/physics.lua` tours the demo area's
physics showcase and asserts the physics happened (it lands in the gravity
room and lifts off, sets a spring pendulum swinging, pushes a floating crate
and the door, lands on the planetoid and lifts off, watches a moon circle
it); it flies around for a couple of minutes, so give it
`--script-timeout 300`. `scripts/harness/moon.lua` fast-travels to the Moon
and lands on it (about three minutes; `--script-timeout 330`). The API:

```lua
client.type("127.0.0.1:7777")  client.press("enter")
client.wait_until(function() return client.state().screen == "online" end, 10)
client.hold("w", 1.5)          client.key_down("d")  client.key_up("d")
client.turn(30, -10)           -- degrees: yaw left+, pitch up+
local s = client.state()       -- same fields as GET /state (s.scene, s.scene_position, ...);
                               -- s.address = the entry address that was dialed, s.shard_address = the
                               -- shard being played on (nil while not on one), s.shard = its id
client.wait_until(function() return client.state().scene == "galaxy" end, 10)  -- after a scene switch
client.wait(0.5)  client.wait_frames(2)
client.screenshot("after.png") -- below --out-dir; returns the path
local r, g, b = client.pixel(x, y)
client.connect(addr [, timeout])  client.disconnect()  client.quit()
```

The sandbox has `math`, `string`, `table` and `client` only: no `os`, `io`,
`require` or file access; a script is stopped at its time limit
(`--script-timeout`, default 60 s).

### c) Browser viewer — watch and play by hand

```sh
space-client --stream 127.0.0.1:8099            # headless, streams JPEG over ws://; add --harness too if you like
cargo run -p xtask -- viewer                    # builds the wasm page, serves http://127.0.0.1:8097/
```

Open `http://127.0.0.1:8097/`. Click the picture, then use the keyboard and
mouse as in the window client: type the address and press Enter on the connect
screen; W/A/S/D, Space/Shift and mouse-drag while playing. `xtask viewer` needs
the `wasm-bindgen` CLI in the version `Cargo.lock` pins; it prints the exact
`cargo install` line (or pass `--install-bindgen`).

The viewer is a local tool. It is not part of any deployment.

### d) The window

`space-client` with no arguments (or `scripts/dev.*` without `--client`) opens
the window. `--harness` works there too, so a frame of the real window can be
captured without touching the mouse or keyboard.

## 3. Operate the cluster while you watch

Open `http://127.0.0.1:7980/` (no login on loopback). The **Scene** card
switches every shard and client at once: it has one button per scene
("Switch all shards to galaxy", ...; the running one is greyed out as
`(current)`). One click switches. The answer waits up to 1.5 s for the shards
to confirm and then says "All 1 connected shard switched to galaxy (revision
2)", or "Switching ..." if some have not confirmed yet; the page then
refreshes every second until they have. The client's picture changes within a
second or two. The same from a script:

```sh
T=$(curl -s http://127.0.0.1:7980/api/state | python3 -c "import json,sys; print(json.load(sys.stdin)['webui']['form_token'])")
curl -s -D - -o /dev/null -X POST --data "csrf=$T&scene=galaxy" http://127.0.0.1:7980/action/scene | grep -iE "x-space-(result|confirmed)"
```

`X-Space-Result` is `accepted` or `refused`; `X-Space-Confirmed: 1/1` says how
many connected shards run the new revision. A shard that loses its orchestrator
**exits** (within about 10 s when the orchestrator was killed, at once when it
closed the link): after restarting the orchestrator start the shards again —
or run it with `--min-shards`, and it does.

The page also splits and merges regions, pauses the autoscaler and shows which
shards have confirmed the scene. Actions are rate-limited (429 when hurried).
A split carves one octant of a region for a newly launched shard; a region
carved more than one level down (a single cell under the root) is listed under
the region around it as "carved out of ...", and its merge button gives it
back to that region's owner. Every shard of the map must acknowledge a map
change within 3 s, or it is rolled back. A merged shard is not stopped at the
commit: its role becomes `draining`, it hands over what it holds and is told
to stop when it is empty (section 4, "Scaling"). An operator's merge is held
to `--min-shards` only; it may merge a layout cell or the spawn point's shard,
which the autoscaler never does.

**Load and the autoscaler.** Every shard reports once a second; `/api/state`
has it per shard as `load_report` (`players`, `entities` — the ships and
bodies it simulates, not ghosts or far replicas — `tick_mean_ms`,
`tick_p95_ms`, `tick_max_ms`, `ticks_jumped`, `hot_cell { cell, players }`,
`load_metric`) and the page shows it in the shard table. An idle four-shard
demo area reads 0 players, tick mean under a millisecond, load 0.02-0.03.
`autoscaler.config` (and the page's Autoscaler card) says what numbers the
autoscaler runs on; the card's button still pauses it.

## 4. What to look at

Check what your change touches, and one step either side of it.

- **Connect screen:** title, address field with caret, hint line; a wrong
  address gives a red, readable reason and keeps what was typed.
- **Demo area** (the default scene, built to show every renderer feature in one
  place): textured floor; mirror floor with objects standing on it; glass
  sphere and cube with textured panels behind them; the gold/plastic roughness
  chart; red, green and blue lamps casting coloured shadows on a white wall;
  the texture crates (the UV grid must read `A1` top-left, upright, not
  mirrored); the ship; several characters; an emissive cube; stars in a
  black sky. Polished metal reflects a dim ambient gradient; nothing
  should be black. The HUD reads `pos 0.0 0.0 0.0` on arrival. Look from the
  spawn, from close up, from above and from behind.
- **The sky** (demo area): Earth ahead, up to the right, about 14 degrees
  across (it does not move: the area rides along with it); the Moon (it goes
  round Earth in 4.5 minutes: look twice, a minute apart), the Sun (round
  once an hour; the stations' shadows follow it) and the planets as coloured
  spheres. The HUD's last lines say the nearest body (`44597 km from Earth`
  at the spawn point). **Fast travel**: type `f` (HUD `fast travel on`),
  turn towards the Moon (`/state`'s `sky` lists every body's position in
  the `scene_position` frame) and hold W: within about half a minute
  `nearest` = `Moon` and `nearest_surface_m` falls to ~2-3 km (fast travel
  keeps 2 km clear of what it flies at); type `f` again (the ship drops to
  120 m/s) and come down in normal flight (`gravity` about 1.6, then
  `landed`). `scripts/harness/moon.lua` does exactly that
  (`--script-timeout 300`).
- **Physics showcase** (demo area; the shard simulates it): turn right from
  the spawn point for the **gravity room** — a blue mat, cyan edges round the
  volume gravity acts in, a `GRAVITY 9.81 M/S2` sign; inside a crate stack, a
  pyramid, a ramp, the concave bowl with balls in it, a seesaw and a gantry
  with two spring pendulums (steel coils that stretch) and a hanging chain
  (pins at its joints). Turn left for the **zero-g corner** — a violet mat, a
  `ZERO G` sign, floating crates, two crates on a spring, a door on a hinge.
  Up to the left, beyond the floor's corner, the **planetoid**: a cratered
  rock with its own gravity (HUD about `gravity 2 m/s2  landed` standing on
  it — the pull is taken at the eye, 1.7 m over 4 m/s² ground), three crates
  resting on it, three balls in orbit, a `PLANETOID 4 M/S2` sign under it.
  Everything moves smoothly at any frame rate (no 30 Hz steps), sits on what
  it rests on (no gap, no sinking), and a player walking into the gravity room
  lands (HUD `gravity 9.81 m/s2  landed`) and lifts off again with Space.
- **Other scenes:** switch to each one and back.
- **Players:** a second client must see the first one's character.
- **One address:** a client given the player port comes online on the shard
  that owns the spawn point, HUD `pos 0.0 0.0 0.0`; `/state` has the player
  port as `address` and the shard's as `shard_address`. A second client
  pointed straight at **another** shard's port ends up on the same shard,
  beside the first (not in the middle of another cell), and each sees the
  other's character. `/api/state` → `player_port` (`bind`, `answer`:
  `redirect` or `starting`, `sends_to_shard`, counters) and
  `client_endpoint` per region and shard; the page shows the same on its
  Cluster card and in the shard table's "players dial" column
  (`0.0.0.0:<port>` = "the host they dialed").
- **Four shard areas** (the scripts' default). `GET /api/state` (and the page)
  show four regions, each with a connected owner; `fleet` says what the
  orchestrator keeps running and what is starting. From the spawn point
  (`pos 0 0 0`, the root's shard) cross x = -6 and z = -4: the HUD's shard
  number changes about a second after the ship is **a metre past** the
  border (the ship's hysteresis, then the handoff's lead: the second the
  shards give the client to connect), `handoffs` counts up and `/state`'s
  `shard_address` changes with it (`address` does not); nothing may vanish
  or turn black. **Into** the spawn point's cell as out of it; more than
  1.6 s from a metre past the border is a defect, and so is a client log
  line `does not take this client`, a shard log line `the handoff of the
  player's ship ... was called off`, or one `could not keep up` (a shard
  that fell behind the cluster clock). Every shard's census line (every
  5 s when its counts change) ends with `N ticks jumped, M caught up; clock
  offset +0.3 ms`: how far its tick clock was from the wall clock when it
  last looked. More than a few milliseconds for longer than it takes to
  slew (7 s per tick of offset) is a defect; a line `was ... s behind the
  wall clock` or `is ... s ahead of the wall clock` says a host's clock was
  stepped or the host suspended. A second client that only looks on
  keeps its `shard`, `session` and `handoffs` while the first crosses, also
  when the first flies right past it: a player's ship is handed over for
  where it is, not for who is near (`space-server/src/border.rs`, "Players'
  ships"). The bodies a player is about to touch — within a metre or so of
  its hull, or on the line it is moving along within a second — do change
  shard with it and go back three seconds after it has gone (the shards'
  handoff counters run). **Flying past changes nothing**: a pass over the
  showcase, however fast, is the ship's own handoff and no other, and a
  shard's `aborted` counter stays where it is — a log line `the handoff of
  ... was called off` is something to look into, for a body as for a player.
- **Physics on four shards** (`docs/plan-sharding.md` §7; the repeatable
  version is `crates/space-server/tests/four_cells.rs`, against the same
  scene on one shard). Leave the stack alone for two minutes: no shard's
  `handoffs`/`aborted` moves (the census line in `shard-<port>.log` is only
  written when something changed), and in `/state`'s `bodies` only the
  moons, pendulums, chain and spring pair move. Then use it: walk into the
  crate stack and on over z = -4, push the crate that floats on the border
  at `pos` -13.2 0.6 -4.0 (it goes up and forward: it floats at head height
  and a ship's top is round — on one shard too), push the spring pair over
  x = -6, walk away from a stack. Nothing may sink into what it rests on,
  hop, jitter, vanish or be there twice; a second client on the neighbour's
  shard lists the same bodies at the same places (`bodies` is drawn state:
  two clients sampled a few milliseconds apart differ by that much motion,
  and not at all at rest).
- **Seeing across shards** (`docs/plan-sharding.md` §6; the repeatable
  version is `crates/space-server/tests/far.rs`). A border makes no
  difference to what is in `bodies` and `others`: every client lists the
  same 35 bodies of the showcase, each once, whichever shard it is on, and
  the count equals the one of a `-MinShards 1` stack. From the spawn point
  none of them is `far`: the showcase lies within 64 m of the borders, so
  the other shards' part of it arrives as ghosts (and a thing within 64 m
  of *any* face of its cell counts as near the root's ground too —
  `border::territory_distance` is a lower bound). To see far replicas put
  something more than 64 m (plus three seconds of its motion) from both
  border planes: a second client at `pos` -110 0 80 is `far` for one on the
  spawn point (`others[1].source`, `far_drawn` 1); a zero-g crate knocked
  south-west — rise half a metre, ram the one at -17.0 1.1 -1.5 along
  (-0.8, 0, 0.6) — turns from `ghost` into `far` about 140 m out and stays
  in every frame. Sample `others` / `bodies` as fast as `/state` answers
  while the thing moves: its drawn position moves on by its speed times
  the frame time, through the changes of `source` and `step` (a change of
  the step from one frame to the next of more than a few centimetres at
  40 m/s is a defect). Fly away in fast travel: the bodies stay listed to
  10 km and are gone beyond it, and come back on the way in. Kill the
  shard a far player is on (by pid, below): it is gone from `others`
  within about three seconds. The census line of each shard says what
  this costs: `N far` (replicas held), `far N B/s out, N B/s in` (to and
  from the other shards) and `N B/s to N clients`; with
  `SPACE_CENSUS_ALWAYS` set in the stack's environment it is written
  every five seconds whether or not a count changed.
- **Seamless crossing** (`docs/plan-sharding.md` §5): a change of shard
  shows in the HUD's shard number and in nothing else. Sample `/state` as
  fast as it answers while crossing and look at what is **drawn**:
  `drawn_position` moves on by `drawn_velocity` times the time between two
  samples (at 15 m/s a tick is half a metre; a step at the switch of more
  than a few centimetres is a defect), `drawn_tick` runs on (it never
  restarts: every shard counts the cluster's ticks; all four shards' log
  lines name the same tick for the same moment), `correction_m` stays
  where it is in ordinary flight (millimetres), and the bodies and players
  in sight before and after are in every frame between (`bodies`,
  `others_drawn`; a burst of `frame.jpg?wait=1` across the border shows it).
  A key pressed during the second of the handoff acts as fast as anywhere
  else. Standing on a border, or walking along it within a metre, is no
  handoff; stopping just across it is one, three seconds later. The
  repeatable version is `crates/space-client/tests/seamless.rs`. In **fast
  travel** the picture is as good at a border as anywhere else, which is
  not perfect: the speed limit follows the distance to the nearest surface,
  so the velocity changes all the time.
  Stop one shard **by its pid**, never by name:
  `target/dev/launched-shards.pid` lists the pids of exactly the shards the
  orchestrator has running (it rewrites the file whenever one starts, stops
  or is seen to have exited) — and check what a pid is before you stop it
  (`(Get-Process -Id N).Path` / `readlink /proc/N/exe` must be this
  checkout's `target/release/space-server`). Within a second or two the event
  log says `shard N lost: its process exited`, a replacement registers under
  a **new id**, its epoch commits, the scene's revision rises by one — the
  bodies are built again — and the pid file names the replacement in place of
  the dead one. The replacement listens on **another port** (the next free
  one of the range; `/api/state` has it). A client that was on the dead shard
  **comes back by itself** if it was started with `--server <player port>`:
  it notices the loss when the transport's idle timeout runs out (about
  10 s — a killed process says no goodbye), shows `lost the connection`,
  retries the player port after 2 s and is sent to the shard that owns the
  spawn point now, where it starts on the spawn point again (`pos 0 0 0`:
  where a player was is not kept). While the root's shard is being replaced
  the player port answers "the cluster is starting"; afterwards
  `player_port.sends_to_shard` names the replacement.
  On the compose stack the same test is `docker kill space-shard-<port>` —
  only a container of a stack **you** started, and look first (`docker ps`,
  `docker inspect`: it carries `space.cluster=<your cluster>` and
  `space.role=shard`). The replacement is a new container under a new id
  within about a second.
- **Scaling** (`docs/plan-sharding.md` §3). Make the cluster scale with two
  clients: start the stack with the autoscaler's numbers lowered in the
  environment —

  ```sh
  SPACE_ORCHESTRATOR_SHARD_CAPACITY=2 SPACE_ORCHESTRATOR_SCALE_OUT_AFTER=5 \
  SPACE_ORCHESTRATOR_SCALE_IN_AFTER=15 SPACE_ORCHESTRATOR_SCALE_COOLDOWN=10 \
    bash scripts/dev.sh --servers-only
  # PowerShell: $env:SPACE_ORCHESTRATOR_SHARD_CAPACITY = '2' ... ; .\scripts\dev.ps1 -ServersOnly
  ```

  — and connect two headless clients. **Out:** some five seconds after the
  second is online the root's shard reports 2 of 2 players in the spawn
  point's cell (load 1.0) and the event log shows that cell wanted, a shard
  launched, registered and its epoch committed, all within a few ticks;
  `launched-shards.pid` has a fifth pid; a second later both clients are on
  the new shard: **the HUD's shard number changes with no border crossed, and
  nothing else does** (`handoffs` counts, `shard_address` changes, the drawn
  step is millimetres, `correction_m` 0, the same bodies listed). From then on
  the event log says about every 45 s that a split `was not carried out: 2 of
  shard N's 2 player(s) are in one cell, and that cell is its whole region
  ...`: a region cannot be smaller than a cell — expected, not a defect. Fly
  both clients a couple of kilometres out (fast travel) and stop: five
  seconds later the cell they stand in is carved for a sixth shard. **In:**
  fly both back; after 15 s and the cooldown the event log says `merge shard
  N: epoch E proposed`, the shard's role in `/api/state` is `draining` for
  about a second, then `shard N is drained after 0.8 s (0 players, 0
  entities) and is told to stop` and `shard N stopped as it was told (exit
  status 0)`; its log ends with `space-server: shard N stops: the
  orchestrator retired it (...)` and its pid leaves `launched-shards.pid`.
  The autoscaler never merges the spawn point's shard nor the three layout
  cells beside it, so the four areas stay and the spawn cell's shard does too.
  **A defect:** a client on the connect screen at any point; a log line
  `called off` for a player's ship; a drain that runs into its timeout (`...
  what it held is gone`); a shard `told to stop ... and is still there`; an
  exit status other than 0 for a retired shard; a shard merged while the
  cluster is at `--min-shards`. The repeatable versions:
  `the_cluster_grows_at_the_hot_cell_and_shrinks_again_by_draining`
  (`space-orchestrator`) and
  `a_shard_whose_region_is_merged_away_hands_everything_over_and_is_empty`
  (`space-server/tests/four_cells.rs`). Known, seen there: switching fast
  travel **off** at several hundred m/s is a prediction correction of metres
  on that tick (23 m at 775 m/s) — the toggle, not a border.
- **A stalled shard** (a suspended host, a debugger): start the stack with
  `SPACE_DEBUG_STALL_DIR` set to a directory (the orchestrator hands its
  environment to the shards it starts). A file `stall-<shard id>` in it
  holding a number of milliseconds makes that shard stand still that long,
  once, the next time its loop turns (the file is removed; the shard logs
  `stalls for N ms`), and then find itself behind: it logs `could not keep
  up ... jumped` and its census line counts the ticks. Cross into it while
  it stands: the client keeps playing on the shard it was on, the handoff
  completes when the stalled shard is back, and the ship is **where its
  commit put it** — not brought forward by the seconds of the stall
  (`caught up` stays 0 for a stall over half a second). The variable is
  for this test only; unset, the hook is one environment lookup a tick.
- **Handoff:** `--topology two-shards`, fly across (the border is at HUD
  `pos` x = 512); the picture must not jump and the scene must stay where it
  was. Players and bodies within 64 m (plus 3 s of their motion) of a border
  are seen from both sides as the neighbour's *ghosts*, which the physics
  of both sides collides with; further apart they are seen as *far
  replicas*, out to 10 km ("Seeing across shards" above).
- **Where the borders are:** a shard area is at least one cell (1024 m). In
  `two-shards` each shard is one cell and the scene sits in the middle of
  shard 1's, 512 m from every border. Under a root of more than one cell —
  the orchestrator's depth-40 root, or `--region 63:…` by hand — the scene
  sits 6 m, half a cell up and 4 m from the corner where the root's cells
  meet: the only border planes through it are the vertical ones at HUD `pos`
  x = -6 and z = -4, the spawn point is inside one cell, and no border is
  horizontal within 512 m (nothing ever falls or floats across one).
- **Physics across a border:** run two shards so that a border cuts the
  showcase: `space-server --shard 1 --region 63:0:0:0 --peer
  2:127.0.0.1:<p2>:64:1:1:0 --scene demo-area` and `space-server --shard 2
  --region 64:1:1:0 --peer 1:127.0.0.1:<p1>:63:0:0:0 --scene demo-area`
  (shard 2 owns what lies at z < -4 with x > -6: the bowl end of the gravity
  room; `64:0:1:1` instead is x < -6, the zero-g corner). Connect to shard 1
  (or to shard 2, which sends you to shard 1: it owns the spawn point)
  — you arrive on the spawn point, `pos 0.0 0.0 0.0` — walk into the room
  and on past the crate stack towards the bowl, over z = -4 (the HUD goes
  to `shard 2  handoffs 1`), and back. The bodies near the player move with
  it as one group (each shard logs `... N bodies, M ghosts; handoffs ...`
  every 5 s when it changes); nothing may jump, sink or stay on the move
  once the player has gone.
- **In every frame:** nothing black that should be lit, nothing inside-out,
  no z-fighting, sane exposure, the HUD values plausible (`fps`, `shard`,
  `scene`, `speed`).
- **In every run:** the client's last line reports `0 validation messages`
  when `SPACE_RENDER_VALIDATION=1` is set and the validation layer is installed.

## 5. Report

Give the commands, the picture paths and what you saw — including what still
looks wrong. Stop every process you started (`--stop`, `POST /quit`) and say
that you did.

## Pitfalls

- **Never use OS-level key injection or the clipboard** (`SendKeys`,
  `AppActivate`, `xdotool`, `Set-Clipboard`): the keys go to whatever window
  has focus. Use the harness, Lua or the viewer page.
- A harness or web UI bind that is not loopback refuses to start without a
  token (`SPACE_HARNESS_TOKEN`; `SPACE_ORCHESTRATOR_WEBUI_TOKEN`). Keep both on
  loopback for testing.
- Client, shard and orchestrator must be built from the same commit (the hello
  is refused on a protocol-version mismatch, with a readable reason).
- On lavapipe expect about 5-10 frames a second at 640x480; give scripts time.
- Ports already taken: `--port-offset`.
- Point a client at the **player port** (7777), not at the control port
  (7900, shards only — a client that dials it is told so on its connect
  screen) and not at a shard's port from `/api/state` (it works, but after
  that shard is replaced the address is dead; the player port's never is).
