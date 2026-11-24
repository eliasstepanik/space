# AGENT.md

Rules and workflow for anyone — AI agent or human — changing this repository.
They are binding. The how-to lives in `.warren/skills/` (Warren loads them
as this project's skills); read both skills before you change code:

- [`.warren/skills/space-dev/SKILL.md`](.warren/skills/space-dev/SKILL.md) — layout, build, CI,
  GPU tests, the invariants that must not break, the release pipeline.
- [`.warren/skills/space-visual-test/SKILL.md`](.warren/skills/space-visual-test/SKILL.md) —
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
   listed in `.warren/skills/space-dev/SKILL.md` (wire gate, handoff freeze, pure Rust,
   `forbid(unsafe_code)`, no TLS in-tree).
3. **Check.** `cargo fmt`, `cargo clippy -D warnings` and the tests of every
   crate you touched; then the whole thing: `cargo run -p xtask -- ci`. On a
   machine with a GPU set `SPACE_REQUIRE_GPU=1`, or "nothing rendered" passes.
4. **Look.** Anything a player, an operator or a viewer can see is verified
   **visually, against a real running stack** — orchestrator, shard(s) and a
   client — by looking at real frames. Passing tests are not a substitute.
   This is not optional and it is not only for renderer work: a new flag, a
   changed message, a web UI field, a camera tweak all end with you looking at
   the result. How: `.warren/skills/space-visual-test/SKILL.md`.
5. **Report.** Say what you ran and what you saw, with the commands and the
   pictures. Say plainly what you did not verify and what still looks wrong.
   Never report a check you did not run.
6. **Commit and push only when asked.** A push to `main` ships: see fact 3.
   Never tag a release by hand.

## Building visually

You are expected to *use* the game while you build it.

| I want to… | Use |
|---|---|
| start the whole stack | `scripts/dev.ps1` (Windows) / `scripts/dev.sh` (Linux): orchestrator + web UI + shard + client |
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
  outside the tree. The compose files ship with the admin UI off in production.
- **Admin and control surfaces stay closed by default.** The orchestrator web
  UI and the client harness are off unless asked for, need no credentials on
  loopback only, and refuse any other bind without a token. Do not weaken that.
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
| 7777/udp | first shard (QUIC) — what a client connects to |
| 7778/udp | second shard in the two-shard handoff setup |
| 7900/udp | orchestrator control port (QUIC) |
| 7980/tcp | orchestrator web UI (HTTP) |
| 8097/tcp | browser viewer page (`xtask viewer`) |
| 8098/tcp | client harness API (HTTP) |
| 8099/tcp | client pixel stream (ws://) |
