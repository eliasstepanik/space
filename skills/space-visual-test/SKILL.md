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
  --player-bind … --launcher process --min-shards N`), one by default.
  `--min-shards 4` / `-MinShards 4` gives four: the demo area in four shard
  areas that meet at HUD `pos` x = -6 / z = -4. The script waits until every
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
`scene_position`, `distance`, `speed`, `parts`, `in_view`), `bodies_in_view`,
and `correction_m` / `correction_max_m` (how far the client's prediction of
its own ship was from the shard's word). `?wait=1` on a
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
change within 3 s, or it is rolled back.

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
- **Four shard areas:** `--min-shards 4`. `GET /api/state` (and the page)
  show four regions, each with a connected owner; `fleet` says what the
  orchestrator keeps running and what is starting. From the spawn point
  (`pos 0 0 0`, the root's shard) cross x = -6 and z = -4: the HUD's shard
  number changes at each crossing, `handoffs` counts up and `/state`'s
  `shard_address` changes with it (`address` does not); nothing may
  vanish or turn black. A crossing takes about a second from the border to
  the new shard number (the handoff's lead), **into** the spawn point's cell
  as out of it; more than two is a defect, and so is a client log line
  `does not take this client`. Seen on a live four-shard stack and not
  solved: a crossing right beside the showcase's bodies (x = -6 at z about
  0, coming from the spawn point; x = -6 at z about -35 from the bowl's
  side) took about 3 s — the time of one attempt called off, the 2 s pause
  and a second attempt; the shards' `aborted` counters rise with it. The
  ship's physics island takes bodies along there
  (`space-server/src/border.rs`; four borders are `docs/plan-sharding.md`
  §7). A second client that only looks on keeps its `shard`,
  `session` and `handoffs` while the first crosses (only a ship's own player
  is redirected; the HUD's `players` stays what it was) **if it stands
  clear of the mover** — some tens of metres: within reach of the mover's
  motion (0.5 m plus one second of it) the two ships are one physics island,
  which is moved to the mover's shard and back, so the bystander's HUD shard
  number flips and its `handoffs` rises by two per fly-by although it never
  moved. Its picture does not jump and no second ship appears (the shards'
  logs: the ship counts add up to the number of players at every line). Stop one shard **by its pid**, never by name:
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
- **Handoff:** `--topology two-shards`, fly across (the border is at HUD
  `pos` x = 512); the picture must not jump and the scene must stay where it
  was. Players and bodies within 64 m (plus 3 s of their motion) of a border
  are seen from both sides (the neighbour's *ghosts*); further apart a
  client sees its own shard's only.
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
