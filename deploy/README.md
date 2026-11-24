# Deployment

| File | Image | Runs |
|---|---|---|
| `../Dockerfile` | `space/ci` | the full CI (`cargo run -p xtask -- ci`) plus the wasm viewer build, as build steps |
| `Dockerfile.shard` | `space/shard` | `space-server`, orchestrated shard; started by the orchestrator, one container per shard (QUIC/UDP, one port of 7870-7899 each) |
| `Dockerfile.orchestrator` | `space/orchestrator` | `space-orchestrator --launcher docker --min-shards 4 --max-shards 8` (control port QUIC/UDP 7900, player port QUIC/UDP 7777) |
| `Dockerfile.client` | `space/client` | `space-client --stream` (ws://, TCP 8099), optionally `--server <player port>` |
| `docker-compose.yaml` | | **the orchestrator alone**: it starts its shard containers itself; the client behind profile `stream` |
| `proxy/` | `ghcr.io/eliasstepanik/space-*` | production compose + `redeploy.sh` for the proxy host |

All builds use the **repository root** as context.

```sh
docker build -t space/ci .                                         # runs CI
docker build -f deploy/Dockerfile.shard -t space/shard:dev .       # the image the orchestrator starts shards from
export SPACE_CLUSTER_SECRET="$(openssl rand -hex 24)"              # required: no default
docker compose -f deploy/docker-compose.yaml up --build            # the orchestrator; it starts four shards
docker compose -f deploy/docker-compose.yaml --profile stream up --build   # + client
space-client --server 127.0.0.1:7777                               # a client on the host: the player port
```

`up --build` builds the orchestrator image only. The shard image has to exist
under the name the orchestrator starts shards from (`SPACE_SHARD_IMAGE`,
default `space/shard:dev`): build it once, and again after every change. An
image that is not on the host is pulled, and `space/shard:dev` is in no
registry: without the build the launch fails and says so. `docker ps` then
shows the orchestrator and `space-shard-<port>` per shard.

What is published on the host: the **player port** `7777/udp`
(`SPACE_PLAYER_PORT`), the one address players type; the **shards' ports**
`7870-7899/udp` (`SPACE_SHARD_PORTS`), each by the shard container that has it
— a player is sent on to `<the host it dialed>:<that port>`; the web UI on the
host's loopback. The orchestrator's control port, `7900/udp`, is **not
published on the host** by either compose file: only the shards use it, and
they reach it over the compose network. There are no fixed container IPs any
more: the orchestrator finds its own network and address by inspecting its
container.

`docker compose down` (or `stop`) sends the orchestrator SIGTERM: it removes
its shard containers (one bound of 20 s; `stop_grace_period: 30s` gives it the
time) and exits 0. Should it be killed instead, its shards notice within about
ten seconds and exit, and the next start first removes what is left of them
(the containers labelled `space.cluster=<cluster>`, `space.role=shard`).

## Publishing pipeline

Both pipelines run on **every push to `main`**, whatever the commit touched, so
a push to `main` is a release and a deploy. GitHub starts one run per push: of
several commits pushed together, the head is the one that is built.

- **Images**: `.github/workflows/release-images.yml` builds and pushes
  `ghcr.io/eliasstepanik/space-shard`, `space-orchestrator` and `space-client`
  as `:sha-<shortsha>` in its matrix job — the orchestrator's build is given
  `SPACE_SHARD_IMAGE=ghcr.io/eliasstepanik/space-shard:sha-<shortsha>`, the
  shard image of its own commit, which is what it starts its shards from —
  and then moves `:latest` in a **second job, `latest`**, that needs all
  three builds: shard, client, **the orchestrator last**. So the shard image
  of a commit is in the registry before the orchestrator's `:latest` points
  at that commit. (The `space/*:dev` names above are local builds only.)
  `:latest` only moves while the commit is still the head of `main`, so a
  slow build of an older push cannot put it back. The repo owner must make a
  package public after its first push. `ci.yml`'s `docker` job still only
  builds.
- **Release**: `.github/workflows/release.yml` takes the next patch version
  after the highest `vX.Y.Z` tag, tags the pushed commit with it, builds
  `space-client` (`--release --locked`) on Linux and Windows plus the wasm
  viewer, and attaches `space-client`, `space-client.exe` and `space-viewer.zip`
  (`index.html` + `pkg/`) to a GitHub Release with generated notes. Nothing is
  tagged by hand; the newest Windows client is always
  `https://github.com/eliasstepanik/space/releases/latest/download/space-client.exe`.
  Pushing a `v*` tag yourself still releases that tag, and is how the minor or
  major version moves (`git tag v0.2.0 && git push origin v0.2.0`; the next push
  to `main` is then `v0.2.1`). A commit that already carries a `v*` tag is not
  numbered again. The numbering lives in `.github/scripts/release_version.js`.
- **Proxy**: `deploy/proxy/` is the production bundle (compose pulling the ghcr
  images, `.env.example`, `redeploy.sh`). It does not serve the browser viewer:
  that page is a local development and agent-testing tool (`AGENTS.md`). Watchtower on the host (label opt-in,
  60s poll) replaces the orchestrator (and the stream client) on new images;
  the shard containers are not its business — the new orchestrator starts
  them from its own commit's shard image. `.github/workflows/deploy.yml` is the manual
  SSH path (secrets `DEPLOY_SSH_HOST/USER/KEY`, not yet created). See
  `deploy/proxy/README.md`.
- **Workflow lint**: `ci.yml`'s `lint` job parses every workflow and deploy YAML
  (`.github/scripts/check_yaml.py`), because GitHub silently drops a workflow
  with invalid YAML.

## Shards and the cluster secret

A shard registers with neither id nor region and is given both: the first the
root, each further one the next layout cell beside the scene's spawn point (up
to `--max-shards`, default 8). A shard that restarts at the same address is its
own replacement, under a **new id**; one that loses the orchestrator exits.

**The orchestrator starts the shards.** Its image runs `--launcher docker
--min-shards 4 --max-shards 8` (image defaults `SPACE_ORCHESTRATOR_LAUNCHER`,
`_MIN_SHARDS`, `_MAX_SHARDS`; the compose variables `SPACE_MIN_SHARDS`,
`SPACE_MAX_SHARDS`): no shard is listed in either compose file. Through the
Docker socket it creates one container per shard from the shard image
(`SPACE_ORCHESTRATOR_SHARD_IMAGE`: baked into the orchestrator image as the
shard image of its own commit; the local compose file sets `space/shard:dev`),
named `space-shard-<port>`, on its own network, with one host UDP port of
`SPACE_SHARD_PORTS` (7870-7899) published under the same number, the cluster
secret in its environment, no mount, no capability and no restart policy. Four
to begin with — the demo area in four areas; a fifth to an eighth when a shard
is loaded (its hot cell is carved for the new one); a shard whose load stays
low is merged away, hands over what it holds and is stopped when it is empty
(exit status 0, container removed); one that dies is replaced under a new id
in a new container. The autoscaler's numbers are flags and environment
variables (`SPACE_ORCHESTRATOR_SHARD_CAPACITY`, `_SCALE_OUT_LOAD`,
`_SCALE_OUT_AFTER`, `_SCALE_IN_LOAD`, `_SCALE_IN_AFTER`, `_SCALE_COOLDOWN`,
`_DRAIN_TIMEOUT`; the local compose file passes `SPACE_SHARD_CAPACITY`,
`SPACE_SCALE_OUT_AFTER`, `SPACE_SCALE_IN_AFTER`, `SPACE_SCALE_COOLDOWN`
through); the rules are in `docs/plan-sharding.md` §3. `--launcher process`
(one `space-server` process per shard) is for a host without containers and
is what `scripts/dev.*` use: the dev loop does not need Docker. `--launcher
docker` is Linux only; on Windows the orchestrator refuses to start with it.

**Players dial the player port** (`7777/udp`, published by both compose files)
and are sent on to the shard that owns the spawn point, at `<the host they
dialed>:<that shard's port>`. So the shards' ports must be reachable wherever
the player port is: open both in the firewall. A shard started **by hand**
from the shard image still works next to an orchestrator (`ORCHESTRATOR_ADDR`,
which may be `name:port`; `SHARD_ADVERTISE`, default `0.0.0.0:7870` — the
orchestrator fills in the address it sees the container at; optional
`SHARD_CLIENT_ENDPOINT`), and `ORCHESTRATOR_CONTROL_ADDR` on the orchestrator
is only for such shards: the docker launcher finds its own.

**The Docker socket makes the orchestrator's container root-equivalent on the
host.** Whoever can talk to `/var/run/docker.sock` can start a privileged
container and mount the host's disk, non-root user inside or not — the
orchestrator is not root in its container, and `group_add: DOCKER_GID` (the
socket's group: default 0 locally, right for Docker Desktop; on a Linux host
`stat -c %g /var/run/docker.sock`, required in `proxy/`) is what gives it the
socket. What is done about it: nothing a player or a shard sends reaches a
Docker call, shard containers get no socket, no mount and no privilege, only
containers carrying this cluster's labels are ever removed, the control port
is not published and the web UI stays off in production. What is not: no
socket proxy narrows the API, and the shard image's name is trusted. All of
it: `docs/security-notes.md`, "The Docker socket".

In a container the control port binds `0.0.0.0`, and **a control port that is
not on loopback takes a shard's hello only with the cluster secret, and the
orchestrator refuses to start without one**. Both compose files therefore give
the orchestrator `SPACE_CLUSTER_SECRET`, which it hands to the shards it
starts in their environment (never on a command line), and **neither has a
default: the repository holds no secret.** Set at least 16 random characters:

- **local stack** (`docker-compose.yaml`): in your shell
  (`export SPACE_CLUSTER_SECRET="$(openssl rand -hex 24)"`, PowerShell:
  `$env:SPACE_CLUSTER_SECRET = ...`) or in a `deploy/.env` file that you do not
  check in (`SPACE_CLUSTER_SECRET=...`). Without it `docker compose` stops
  before it starts anything and says which variable is missing.
- **`proxy/`**: in `.env` (see `.env.example`); with it empty `docker compose`
  refuses to start the stack there too, and says which variable is missing.

The secret gates the hello, and the hello gates the rest: a connection whose
hello is refused is told why and closed, and one that is welcomed may report
and acknowledge only for the shard it was welcomed as. **The control port is
not published** by either file (it was, on every host interface, before this
change): nothing outside the compose network has any business with it. If you
do publish it for shards on another host, the cluster secret is all that stands
between that port and anybody who can reach it - see
`docs/security-notes.md`.

**A host that runs the proxy stack has to get the new `.env` lines
(`SPACE_CLUSTER_SECRET`, `DOCKER_GID`), the new compose file and the open
firewall ports before it pulls images built from this change** (Watchtower
only swaps the image: with the old compose file the new orchestrator image
would start the docker launcher without a socket and refuse to start — safe,
but down). The steps and their order: `proxy/README.md`, "Before the
shard-manager images arrive".

## Scene and the orchestrator web UI

**The scene is server state.** The orchestrator owns it, every shard runs it and
tells its clients; a client has no say. Catalogue: `demo-area` (default; the
solar system is part of it), `materials`, `galaxy`. The former `solar` scene was
retired: a saved `solar` comes back as `demo-area` at the next revision, and
`--scene solar` is refused. `space-orchestrator --scene NAME` (env
`SPACE_ORCHESTRATOR_SCENE`, compose variable `SPACE_SCENE`) is only the *initial*
scene: one chosen at run time is saved next to the id counter
(`<state file>.scene`) and wins on restart, so keep the state directory on a
volume and delete that file to make `--scene` apply again. A shard run without
an orchestrator takes `space-server --scene NAME`; give hand-wired `--peer`
shards the same one. Client, shard and orchestrator must come from the same
build (wire protocol 7) - the pipelines above build them from one commit.

**The web UI** is part of the `space-orchestrator` binary (server-rendered HTML,
no JavaScript): cluster, map, shards, autoscaler, id allocator, recent events,
and the actions split / merge / pause autoscaler / re-send map / **switch the
scene on all shards**. The same state is JSON at `/api/state`.

- Off unless `--webui ADDR` (env `SPACE_ORCHESTRATOR_WEBUI`). Suggested port 7980.
- A loopback bind needs no credentials - and then serves only loopback `Host`s
  and refuses anything that arrived through a proxy. **Never put a proxy in
  front of a tokenless UI.**
- Any other bind (a container needs `0.0.0.0:7980`) refuses to start without
  `--webui-token` / `SPACE_ORCHESTRATOR_WEBUI_TOKEN` / `..._TOKEN_FILE` of at
  least 16 characters. Log in with any user name and the token as the password.
- Plain HTTP with Basic auth: publish it on loopback and tunnel
  (`ssh -L 7980:127.0.0.1:7980 host`), or put a TLS proxy in front and name the
  public host with `--webui-host name[:port]` (needs a token).
- `--max-regions N` (default 64) caps how far splits can grow the map.
- This compose file switches it on for local use at `http://127.0.0.1:7980/`
  (password `space-local-dev-token` unless `SPACE_WEBUI_TOKEN` is set);
  `proxy/` leaves it off until `.env` sets a bind and a token.

Details and known gaps: `docs/security-notes.md`.

## Streaming client and the wasm viewer

`Dockerfile.client` starts `space-client --stream $STREAM_BIND [--server
$SPACE_SERVER] $CLIENT_ARGS`. Compose sets
`SPACE_SERVER=host.docker.internal:<player port>` (with `extra_hosts:
host-gateway`): the client is a player like any other and dials the player
port **on the host**, because a client that dialed the orchestrator's
in-network address would be sent on to `<that address>:<shard port>`, where no
shard listens. (That path has not been executed: see below.) `CLIENT_ARGS` is
for extra flags such as `--size` (the scene is not a
client flag any more: the shard announces it). It renders on lavapipe (software Vulkan);
bookworm's Mesa 22.3 lavapipe lacks `VK_KHR_ray_query`, so GI is off in that
image (the renderer logs it and carries on). Pass `/dev/dri` through or use
Mesa >= 25 for the ray-query path.

The client's port speaks WebSocket only. There is **no** static-file flag, so the
wasm viewer is served separately: build it (`crates/space-viewer/README.md`:
`cargo build --target wasm32-unknown-unknown`, then `wasm-bindgen --target web
--out-dir crates/space-viewer/pkg`) and serve `crates/space-viewer/` with any
static server, e.g. `python3 -m http.server 8000`. Open
`http://127.0.0.1:8000/index.html?ws=ws://127.0.0.1:8099`; the `ws` query
parameter is the viewer's only configuration (default: port 8099 on the host
that served the page, `ws://<page hostname>:8099`). The viewer shows a status
line and reconnects on its own when the stream goes away.
`ws://` only, no TLS; put a TLS-terminating proxy in front for `wss://`.

## What is and is not verified

State at the online-only rework (wire protocol 4), on Windows 11 with an RTX
4080 SUPER and Docker 28, and on Ubuntu 24.04 in WSL (Mesa 25 lavapipe):

**Run and passing**
- `cargo run -p xtask -- ci` with `SPACE_REQUIRE_GPU=1`: fmt, clippy `-D warnings`
  and every test - **1081 passed, 0 failed, 4 ignored**, with
  `SPACE_RENDER_VALIDATION=1` (zero validation messages).
- `Dockerfile.orchestrator`, `Dockerfile.shard`, `Dockerfile.client` and
  `docker-compose.yaml` **built and run**, including the `stream` profile: the
  shard registers; the web UI answers 401 without the token and 200 with it; a
  scene switch in the web UI reaches the shard and the client container, and
  survives an orchestrator restart; the client container renders on lavapipe
  (`ray_query=true`), connects over the bridge and its stream was watched in a
  browser through `cargo run -p xtask -- viewer`.
- The same stack without containers through `scripts/dev.ps1` (Windows) and
  `scripts/dev.sh` (Linux/WSL, software Vulkan): connect screen, demo area,
  scene switching, a second client, the Lua smoke script, the harness API.

The list above is the state before the compose files were given the cluster
secret and the orchestrator became the shard manager; what it says about the
compose stack (one hand-started shard, the client over the bridge, the
`stream` profile) describes files that no longer exist in that form.

**The shard manager (wire protocol 7): executed** on Windows 11 with Docker
Desktop 28.5.1 (Linux engine in WSL2), 2026-11-24
- `docker-compose.yaml` as it is, with a project name, ports moved by
  environment (player port 8077, shards 8170-8199, web UI 8280), a generated
  throwaway `SPACE_CLUSTER_SECRET`, `DOCKER_GID=0`, lowered autoscaler
  thresholds (capacity 2, out after 5 s, in after 15 s, cooldown 10 s) and
  local image tags.
- Compose started the orchestrator alone and it created four shard containers:
  the first registered 0.9 s after the orchestrator started, all four owned
  their regions after 3.5 s.
- A headless client on the host joined through `127.0.0.1:8077` and was sent
  to a shard container's published port (35 bodies listed, frames looked at).
  Four flying crossings between two shard containers at 15-40 m/s: drawn step
  3.1-4.6 mm, corrections 0.
- `docker kill` of one shard container: noticed and replaced under a new id in
  a new container within about a second (lost at tick 2385, the replacement's
  epoch committed at tick 2416), the scene re-issued once.
- Scale-out: two players in the spawn cell, a fifth container; then both in
  the next cell, a sixth (0.9 s from "wanted" to registered). Scale-in: the far
  cell's shard merged, drained in 1.1 s, `exit status 0`, container removed.
- `docker compose stop` (SIGTERM): all five shard containers removed and the
  orchestrator exited 0 in under 5 s; `down -v` left no container, network or
  volume. The cluster secret appears in none of the captured logs.
- That long scenario ran on images built from a throwaway Dockerfile (a single
  build stage with a cache) with the same runtime stages and the same
  binaries. The images from `Dockerfile.orchestrator` and `Dockerfile.shard`
  themselves were built and run through a start, a client's join, a killed
  orchestrator with the leftovers removed at the next start, and `compose
  down`.

**Only parsed** (`docker compose config`)
- `proxy/docker-compose.yaml`, with `.env.example` plus a throwaway secret and
  `DOCKER_GID`.

**Not run**
- Anything on a Linux host: a socket group other than 0, a firewall, real
  published ports to another machine.
- Watchtower replacing the orchestrator; a pull of the `sha-` tagged shard
  image from ghcr; a private registry.
- The `client` stream service in either compose file since it dials the player
  port through the host; `Dockerfile.client` (unchanged).
- `release-images.yml` with its `latest` job (YAML-checked with
  `.github/scripts/check_yaml.py` only).
- `redeploy.sh`; the web UI behind a real TLS proxy; the root `Dockerfile`
  (CI's `docker` job builds it).
- GPU tests on the hosted CI runner: its Mesa is too old for the renderer, so
  that step is `continue-on-error: true` and fails visibly rather than
  reporting tests that rendered nothing.
- Two players seeing each other across a border between two shard
  containers: nothing is recorded about it for the run above. (Seen live
  outside containers: two players on either side of a border see each other
  as characters.)

**What still looks wrong, seen on Docker Desktop**
- In the WSL2 VM the shards' census lines showed a `clock offset` between 0
  and **-84 ms**: the hypervisor's time sync steps the VM's wall clock (a
  process stack on the same machine shows 0.0 ms throughout). All shards share
  that clock, but each slews by itself, so for seconds at a time two shards
  were a tick or two apart: two of the later crossings there showed a drawn
  step of 0.40 m and 0.76 m and corrections of 0.05 to 1.2 m (the four
  measured right after the start: 3.1-4.6 mm, 0). Not a defect of the Docker
  launcher: it is the cluster clock's stated limit (`docs/plan-sharding.md`
  §5: hosts whose clocks differ, a wall clock that is stepped) made visible by
  that VM. Not verified on a real Linux host, where one kernel clock serves
  every container.
