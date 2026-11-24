# AGENT.md

Rules and workflow for anyone — AI agent or human — changing this repository.
They are binding. The how-to lives in the skills folder; read both skills
before you change code:

- [`skills/space-dev/SKILL.md`](skills/space-dev/SKILL.md) — layout, build, CI,
  GPU tests, the invariants that must not break, the release pipeline.
- [`skills/space-visual-test/SKILL.md`](skills/space-visual-test/SKILL.md) —
  the visual test loop: start the stack, drive a client, look at the frames.

`AGENTS.md` and `CLAUDE.md` only point here.

## What this project is

A space game engine in Rust: authoritative shards behind an orchestrator, a
Vulkan client, a browser viewer. Three facts shape every change:

1. **Online only.** The client always plays on a shard. There is no offline
   mode, no client-side scene choice and no "render a scene to a file" mode.
   Started without `--server` it shows a connect screen and waits for an
   address.
2. **The scene is server state.** The orchestrator owns it (default
   `demo-area`), every shard runs it and tells its clients. It is changed in the
   orchestrator's web UI, for the whole cluster at once — never by a client.
3. **Every push to `main` is a release and a deploy.** CI tags the pushed
   commit with the next version, publishes the client binaries and pushes the
   server images, which the production host pulls within a minute.

## The workflow

Every change goes through these steps, in this order.

1. **Understand.** Read the skill, the crate's module docs and the tests of the
   area you touch. State what you are about to change and why.
2. **Change.** Small, in the crate that owns the behaviour. Keep the invariants
   listed in `skills/space-dev/SKILL.md` (wire gate, handoff freeze, pure Rust,
   `forbid(unsafe_code)`, no TLS in-tree).
3. **Check.** `cargo fmt`, `cargo clippy -D warnings` and the tests of every
   crate you touched; then the whole thing: `cargo run -p xtask -- ci`. On a
   machine with a GPU set `SPACE_REQUIRE_GPU=1`, or "nothing rendered" passes.
4. **Look.** Anything a player, an operator or a viewer can see is verified
   **visually, against a real running stack** — orchestrator, shard(s) and a
   client — by looking at real frames. Passing tests are not a substitute.
   This is not optional and it is not only for renderer work: a new flag, a
   changed message, a web UI field, a camera tweak all end with you looking at
   the result. How: `skills/space-visual-test/SKILL.md`.
5. **Report.** Say what you ran and what you saw, with the commands and the
   pictures. Say plainly what you did not verify and what still looks wrong.
   Never report a check you did not run.
6. **Commit and push only when asked.** A push to `main` ships: see fact 3.
   Never tag a release by hand.

## Building visually

You are expected to *use* the game while you build it.

| I want to… | Use |
|---|---|
| start the whole stack | `scripts/dev.ps1` (Windows) / `scripts/dev.sh` (Linux): the orchestrator (player port, web UI), the **four shards** it starts — the demo area in four shard areas that meet at HUD `pos` x = -6 / z = -4 — and a client pointed at the player port. The orchestrator starts the shards itself (one `space-server` process each, logs in `target/dev/shard-<port>.log`), replaces one that dies, and `-Stop` stops them too |
| run it on one shard, or on more | the same with `-MinShards N` / `--min-shards N` (1 to 8; 1 = no borders anywhere). The client's address stays the player port |
| watch the cluster scale | the dev scripts with `SPACE_ORCHESTRATOR_SHARD_CAPACITY=2`, `SPACE_ORCHESTRATOR_SCALE_OUT_AFTER=5`, `SPACE_ORCHESTRATOR_SCALE_IN_AFTER=15`, `SPACE_ORCHESTRATOR_SCALE_COOLDOWN=10` in the environment and two clients: a loaded shard's hot cell is carved for a new shard, an idle shard is merged away, drains and stops (`skills/space-visual-test/SKILL.md`, "Scaling") |
| run it in containers | `deploy/docker-compose.yaml`: **the orchestrator alone**, which starts its shard containers through the Docker socket (`--launcher docker`, Linux engine only). The dev scripts do not need Docker: they keep `--launcher process` |
| drive a client without a human | the harness: `space-client --headless --harness 127.0.0.1:8098`, then `GET /state`, `GET /frame.png`, `POST /input`, `POST /script` |
| script a user | Lua: `space-client --headless --script scripts/harness/smoke.lua` — the exit code is the script's result |
| watch and play in a browser | `space-client --stream 127.0.0.1:8099` + `cargo run -p xtask -- viewer` → `http://127.0.0.1:8097/` |
| operate the cluster / switch the scene | the orchestrator web UI, `http://127.0.0.1:7980/` (`/api/state` is the same as JSON) |

The headless client runs on software Vulkan (lavapipe, Mesa 24.1 or newer), so
it works in a Linux sandbox with no GPU and no display. Open the frames you
capture and judge them with your own eyes: is the scene lit, are textures the
right way round, do shadows fall where the lights say, is anything black,
inside-out or flickering?

The browser viewer is a **local development tool**. It is not deployed and
must not be added back to the production stack.

## Hard rules

- **Never send OS-level input and never touch the clipboard.** No `SendKeys`,
  `AppActivate`, `xdotool`, `osascript`, `Set-Clipboard` or anything else that
  types into "whatever window has focus". Keys sent that way have gone to the
  wrong window on this project and deleted a source file. A client is driven
  only through the harness API, Lua scripts, or the browser viewer page.
- **Do not rewrite or delete work that is not yours.** No `git stash`,
  `git reset`, `git checkout --`, `git clean` or `rm` on files you did not
  create, unless the owner asked for exactly that. Read-only git is always fine.
- **Stop what you start.** Every server, client and viewer you launch is
  stopped before you finish; say so in your report. Use spare ports (the dev
  scripts take a port offset) when someone else may be using the defaults.
- **No secrets in the repository.** Tokens come from the environment or a file
  outside the tree — the cluster secret of the compose files too: it has no
  default. The compose files ship with the admin UI off in production and do
  not publish the orchestrator's control port.
- **Admin and control surfaces stay closed by default.** The orchestrator web
  UI and the client harness are off unless asked for, need no credentials on
  loopback only, and refuse any other bind without a token. The orchestrator's
  control port follows the same rule for shards: no secret on loopback, and on
  any other bind no start without a cluster secret and no hello without it.
  A control connection is heard only after its hello was welcomed, and then
  only about the shard it was welcomed as. Shards of one cluster present its
  peer token to each other. Do not weaken that, and never print a secret: one
  is held in `space_net::Secret`, and a message that carries one has a `Debug`
  that leaves it out.
- **The Docker socket is the orchestrator's alone.** With `--launcher docker`
  the orchestrator's container holds `/var/run/docker.sock`, which makes it
  root-equivalent on the host (`docs/security-notes.md`, "The Docker
  socket"). A shard container never gets the socket, a mount, a capability
  or a privilege. Nothing a player or a shard sends may reach a Docker API
  call: image, command, ports, network and labels come from the
  orchestrator's configuration and its own counters. Only containers that
  carry this cluster's labels are ever removed. The compose files hold the
  orchestrator and no shard, and keep the web UI off in production.
- **The player port is the public entry, not a control surface.** The
  orchestrator's `--player-bind` (off unless given) needs no token on any
  bind — and must stay unable to do anything: it answers a client's hello
  with the shard to dial (or "starting") and closes, keeps no state per
  player, reads nothing a peer sends behind its hello, and has no way into
  the orchestrator's state. No shard registers through it, and the control
  port does not serve players.
- **Pure Rust.** No C dependencies (`ring` is the one sanctioned exception), no
  JavaScript toolchain; the orchestrator web UI is server-rendered HTML with no
  script at all. A new third-party crate needs a stated reason.
- **The wire format is append-only**, and a change that an older peer cannot
  survive bumps `PROTOCOL_VERSION`. Client, shard and orchestrator are always
  built and shipped from the same commit.
- **Docs move with the code.** A change to a flag, a port, a message or the
  workflow updates the README, the skills and this file in the same change.

## Ports used by the local stack

| port | what |
|---|---|
| 7777/udp | what a client connects to (QUIC): the orchestrator's player port (`--player-bind`), which sends the client on to the shard that owns the spawn point — or the shard itself in `single` / `two-shards`. Published on the host by the compose files |
| 7778/udp | second shard in the two-shard handoff setup |
| 7870-7899/udp | shards the orchestrator starts itself (`--launcher process`, or `--launcher docker`: then each shard container publishes its own port on the host; `--host-port-range`); players are sent to them by the player port and by handoffs, as `0.0.0.0:<port>` = "the host you dialed". Which one owns the spawn point is `player_port.sends_to_shard` / `map.spawn_owner` in `/api/state` — usually the first port, but not after a first launch that failed or once the root's shard has been replaced |
| 7900/udp | orchestrator control port (QUIC): shards only (a player who dials it is told so). No credentials on loopback; any other bind needs the cluster secret. Never published by the compose files |
| 7980/tcp | orchestrator web UI (HTTP) |
| 8097/tcp | browser viewer page (`xtask viewer`) |
| 8098/tcp | client harness API (HTTP) |
| 8099/tcp | client pixel stream (ws://) |
