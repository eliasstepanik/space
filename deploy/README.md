# Deployment

| File | Image | Runs |
|---|---|---|
| `../Dockerfile` | `space/ci` | the full CI (`cargo run -p xtask -- ci`) plus the wasm viewer build, as build steps |
| `Dockerfile.shard` | `space/shard` | `space-server`, orchestrated shard (QUIC/UDP 7870) |
| `Dockerfile.orchestrator` | `space/orchestrator` | `space-orchestrator --launcher local` (QUIC/UDP 7900) |
| `Dockerfile.client` | `space/client` | `space-client --stream` (ws://, TCP 8099), optionally `--server <shard>` |
| `docker-compose.yaml` | | orchestrator + one shard; the client behind profile `stream` |
| `proxy/` | `ghcr.io/eliasstepanik/space-*` | production compose + `redeploy.sh` for the proxy host |

All builds use the **repository root** as context.

```sh
docker build -t space/ci .                                         # runs CI
docker build -f deploy/Dockerfile.shard -t space/shard .
export SPACE_CLUSTER_SECRET="$(openssl rand -hex 24)"              # required: no default
docker compose -f deploy/docker-compose.yaml up --build            # orchestrator + shard-a
docker compose -f deploy/docker-compose.yaml --profile stream up --build   # + client
```

Fixed IPs on the `172.31.0.0/24` bridge (addresses are parsed as socket
addresses, not resolved): orchestrator `.10:7900`, shard-a `.11:7870`, client
`.12:8099`. The orchestrator's control port, `7900/udp`, is **not published on
the host** by either compose file: only the shards use it, and they reach it
over the compose network.

## Publishing pipeline

Both pipelines run on **every push to `main`**, whatever the commit touched, so
a push to `main` is a release and a deploy. GitHub starts one run per push: of
several commits pushed together, the head is the one that is built.

- **Images**: `.github/workflows/release-images.yml` builds and pushes
  `ghcr.io/eliasstepanik/space-shard`, `space-orchestrator` and `space-client`
  as `:sha-<shortsha>`, then moves `:latest` to that image (the `space/*:dev`
  names above are local compose builds only). `:latest` only moves while the
  commit is still the head of `main`, so a slow build of an older push cannot
  put it back. The repo owner must make a package public after its first push.
  `ci.yml`'s `docker` job still only builds.
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
  60s poll) redeploys on new images; `.github/workflows/deploy.yml` is the manual
  SSH path (secrets `DEPLOY_SSH_HOST/USER/KEY`, not yet created). See
  `deploy/proxy/README.md`.
- **Workflow lint**: `ci.yml`'s `lint` job parses every workflow and deploy YAML
  (`.github/scripts/check_yaml.py`), because GitHub silently drops a workflow
  with invalid YAML.

## Shards and the cluster secret

A shard registers with neither id nor region and is given both: the first the
root, each further one the next layout cell beside the scene's spawn point (up
to `--max-shards`, default 8). A shard that restarts at the same address is its
own replacement, under a **new id**; one that loses the orchestrator exits (and
compose's `restart: unless-stopped` starts a fresh one). The orchestrator image
runs `--launcher local`; `--min-shards` / `--launcher process` (the orchestrator
starts the shards itself) are for a host without containers.

**No player port in the containers yet.** The orchestrator's `--player-bind`
(the one address of `scripts/dev.*`, 7777/udp there) is not set by either
compose file, and nothing new is published: players dial `shard-a` directly,
as before. That still works because `shard-a` is given the root and so owns
the scene's spawn point; a shard that does not would now send a new player on
to the one that does, at the address that shard advertises
(`SHARD_ADVERTISE`, a compose-network address a player outside cannot
reach). Putting the player port and the shards' public addresses into the
containers belongs to the Docker launcher work (`docs/plan-sharding.md` §8).

In a container the control port binds `0.0.0.0`, and **a control port that is
not on loopback takes a shard's hello only with the cluster secret, and the
orchestrator refuses to start without one**. Both compose files therefore give
the orchestrator and the shard `SPACE_CLUSTER_SECRET` (read by both binaries,
never passed on a command line), and **neither has a default: the repository
holds no secret.** Set at least 16 random characters:

- **local stack** (`docker-compose.yaml`): in your shell
  (`export SPACE_CLUSTER_SECRET="$(openssl rand -hex 24)"`, PowerShell:
  `$env:SPACE_CLUSTER_SECRET = ...`) or in a `deploy/.env` file that you do not
  check in (`SPACE_CLUSTER_SECRET=...`). Without it `docker compose` stops
  before it starts anything and says which variable is missing.
- **`proxy/`**: in `.env` (see `.env.example`); with it empty the orchestrator
  stays down and says why in its log.

The secret gates the hello, and the hello gates the rest: a connection whose
hello is refused is told why and closed, and one that is welcomed may report
and acknowledge only for the shard it was welcomed as. **The control port is
not published** by either file (it was, on every host interface, before this
change): nothing outside the compose network has any business with it. If you
do publish it for shards on another host, the cluster secret is all that stands
between that port and anybody who can reach it - see
`docs/security-notes.md`.

**A host that runs the proxy stack has to get the `.env` line and the new
compose file before it pulls images built from this change** (Watchtower only
swaps the image: with the old compose file the new orchestrator would refuse to
start).

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
$SPACE_SERVER] $CLIENT_ARGS`. Compose sets `SPACE_SERVER=172.31.0.11:7870`
(shard-a); `CLIENT_ARGS` is for extra flags such as `--size` (the scene is not a
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

**Not run**
- Neither compose file since the cluster secret was added to them (shard
  lifecycle change, wire protocol 7), nor since the control port stopped being
  published and the local file's default secret was removed: the
  `SPACE_CLUSTER_SECRET` lines (`${...:?...}` in the local file) and the
  `ports:` lists are correct by inspection and YAML-checked
  (`.github/scripts/check_yaml.py`) only. The "run and passing" list above is
  the state before those changes.
- `proxy/docker-compose.yaml` on the host (checked with `docker compose config`
  against `.env.example` only) and `redeploy.sh`; the web UI behind a real TLS
  proxy; the root `Dockerfile` (CI's `docker` job builds it).
- GPU tests on the hosted CI runner: its Mesa is too old for the renderer, so
  that step is `continue-on-error: true` and fails visibly rather than
  reporting tests that rendered nothing.
- Cross-shard visibility in containers: players and physics bodies near a
  border are ghosted to the neighbouring shard (seen live with two shards
  outside containers: two players 24 m apart on either side of the border see
  each other as characters), but no container setup with two shards has been
  run.
