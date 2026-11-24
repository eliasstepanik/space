# Proxy deployment bundle

Production stack for the proxy host: **the orchestrator, which starts its
shard containers itself** through the Docker socket, and the ws:// pixel-stream
client, from prebuilt `ghcr.io/eliasstepanik/space-*` images; the orchestrator
is auto-redeployed by Watchtower and brings its shards with it.

| File | Purpose |
|---|---|
| `docker-compose.yaml` | pulls `space-orchestrator` and `space-client`; both carry the Watchtower opt-in label; no shard is listed - the orchestrator creates `space-shard-<port>` containers from `space-shard:sha-<its commit>`; the Docker socket mounted into the orchestrator; `restart: unless-stopped` (except `client`: `restart: "no"`, see Client configuration); orchestrator state in `./state` |
| `.env.example` | copy to `.env`: cluster name, **cluster secret (required)**, **`DOCKER_GID` (required)**, host ports, shard counts, image tag |
| `redeploy.sh` | manual path only (used by `.github/workflows/deploy.yml`): `compose pull`, `up -d`, image prune |

## Before the shard-manager images arrive

A host that runs the earlier compose file (orchestrator + `shard-a`) has to be
prepared **before** the images built from the shard-manager change are pushed,
or at the latest together with them. In this order:

1. **`.env`**: add `DOCKER_GID=$(stat -c %g /var/run/docker.sock)` (the
   number, as a line `DOCKER_GID=<n>`), and keep `SPACE_CLUSTER_SECRET` (at
   least 16 characters). Both are required: without either `docker compose`
   refuses to start the stack and says which is missing.
2. **Firewall**: open UDP 7777 (the player port; `PLAYER_PORT`) and UDP
   7870-7899 (the shards; `SPACE_SHARD_PORTS`). Players now type
   `<host>:7777`, not the old shard port 7870, and are sent on to one of the
   shard ports.
3. **Compose file**: install this directory's `docker-compose.yaml` and run
   `docker compose up -d --remove-orphans`. That removes the old
   `space-shard-a` container; the new shards are named `space-shard-<port>`.
4. **Accept what the socket means**: mounting `/var/run/docker.sock` makes the
   orchestrator's container root-equivalent on this host (below).

Why the order: Watchtower swaps only the image. The **old compose file with
the new orchestrator image** starts the docker launcher without a socket, and
the orchestrator refuses to start - safe, but down. The **new compose file
with the old image** ignores the new variables and runs `--launcher local`
with no shard - up, and empty. So: new compose file and `.env` first, then the
push. To hold a version, pin `SPACE_TAG` to a `sha-` tag.

## The Docker socket

Whoever can talk to `/var/run/docker.sock` can start a privileged container
and mount the host's disk: **the orchestrator's container is root-equivalent
on this host**, although the orchestrator is not root inside it (`group_add:
DOCKER_GID` is what gives it the socket). What is done: nothing a player or a
shard sends reaches a Docker call (image, command, ports, network and labels
come from the orchestrator's configuration and its own launch counter); shard
containers get no socket, no mount, no capability, `no-new-privileges` and no
restart policy; only containers carrying both of this cluster's labels
(`space.cluster`, `space.role=shard`) are ever removed; the cluster secret is
in the shard containers' environment (visible to whoever can `docker inspect`
here, that is to the root-equivalent only); the control port is not published;
the web UI - the one way to operate the orchestrator from outside, and it can
split and merge, which starts and stops containers - is off. What is not done:
no socket proxy narrows the Engine API to the calls needed (a filtering proxy
that allows only containers create/start/inspect/delete/list and images
inspect/create is a recommendation, not something tested here); the shard
image's name is trusted; the shards' ports are published on every interface.
`docs/security-notes.md`, "The Docker socket", has all of it.

## Install (one-time, on the host)

The stack lives in `/home/proxy/stacks/space/` (no sudo needed):

```sh
mkdir -p /home/proxy/stacks/space/state
cp docker-compose.yaml /home/proxy/stacks/space/
cp .env.example /home/proxy/stacks/space/.env     # then edit
chmod a+rwx /home/proxy/stacks/space/state        # orchestrator runs as uid 10001
cd /home/proxy/stacks/space && docker compose up -d
```

`SPACE_CLUSTER_SECRET` in `.env` is **required**: at least 16 random characters
(`openssl rand -hex 24`), written into `.env` on the host and nowhere else.
Inside its container the orchestrator's control port binds the wildcard
address, and a control port that is not on loopback takes a shard's hello only
with the secret. The orchestrator hands it to the shard containers it starts,
in their environment; nobody else needs it. Without it `docker compose`
refuses to start the stack and says why.

`DOCKER_GID` in `.env` is **required** too: the group of the Docker socket on
this host (`stat -c %g /var/run/docker.sock`). The orchestrator does not run
as root in its container and is given that group instead.

**What has to be open in the firewall:** UDP `7777` (`PLAYER_PORT`), the one
address players type, and UDP `7870-7899` (`SPACE_SHARD_PORTS`), one per shard,
each published by the shard container that has it. A player is sent on from
the player port to `<the host it dialed>:<a shard's port>`.

**The control port (`7900/udp`) is not published on the host.** Only the
shards use it, over the compose network; `ORCHESTRATOR_PORT` is gone from
`.env.example` (a line left in an existing `.env` is simply unused). An
installation that was started from an earlier compose file that published it
has the port open on every interface until the stack is recreated from this
one (`docker compose up -d`).

**Shards.** The orchestrator keeps `SPACE_MIN_SHARDS` (4: the demo area in four
areas) running and goes up to `SPACE_MAX_SHARDS` (8) under load; a shard that
dies is replaced, one that is no longer needed hands over what it holds and is
stopped. `docker ps` shows them as `space-shard-<port>`. They are not in the
compose file and `docker compose` commands do not list them; stopping the
orchestrator (`docker compose stop` / `down`: SIGTERM, `stop_grace_period:
30s`) removes them, and a start removes whatever an orchestrator that was
killed left behind (containers labelled `space.cluster=<SPACE_CLUSTER>`,
`space.role=shard`) before it starts new ones.

## How updates reach the host

1. Every push to `main` runs `.github/workflows/release-images.yml`, which
   pushes `ghcr.io/eliasstepanik/space-{shard,orchestrator,client}` tagged
   `sha-<shortsha>` and then, in a second job that needs all three, moves
   `latest`: shard, client, the orchestrator **last**. The orchestrator image
   names the shard image of its own commit
   (`ghcr.io/eliasstepanik/space-shard:sha-<shortsha>`, baked in at build
   time), so that image is in the registry before the orchestrator's `latest`
   points at the commit.
2. **Primary path (Watchtower):** the Watchtower container already running on
   the host polls every 60s. It only touches containers that opt in with the
   label `com.centurylinklabs.watchtower.enable: "true"` (set on `orchestrator`
   and `client`). When the `:latest` digest changes it pulls the new
   image and recreates the container with the same configuration. Nothing to
   install or schedule per stack. **The shard containers are not Watchtower's
   business:** the old orchestrator gets SIGTERM and removes its shards, the
   new one removes leftovers and starts four shards from its own commit's
   shard image (pulled if it is not on the host). Players of the old shards
   reconnect through the player port, as after any shard death: about 10 s,
   and back on the spawn point.
3. **Manual path:** `.github/workflows/deploy.yml` (`workflow_dispatch`, or
   `repository_dispatch` with `event_type: redeploy`) SSHes to the host and runs
   `redeploy.sh`. It is no longer the primary path.

Watchtower only follows the tag the container runs. Rollback: set
`SPACE_TAG=sha-<shortsha>` in `.env` and run `docker compose up -d`; set it back
to `latest` to resume auto-updates. The shards always follow the orchestrator:
it starts them from the shard image of its own commit.

## Scene and the operator web UI

`SPACE_SCENE` in `.env` is the scene the cluster starts with (`demo-area`,
which includes the solar system, `materials`, `galaxy`). Once an operator switches it in the web UI the
choice is saved in `./state/orchestrator.id.scene` and wins over `.env` on every
restart; delete that file to go back to `SPACE_SCENE`.

The web UI (status, split / merge, autoscaler pause, scene switch) is **off**
until `.env` sets both `SPACE_WEBUI_BIND=0.0.0.0:7980` and a `SPACE_WEBUI_TOKEN`
of at least 16 random characters (a bind without a token makes the orchestrator
refuse to start, which takes the cluster down - set both or neither). With the
Docker socket mounted, leave it off unless you need it: it can split and
merge, which starts and stops containers. It is
plain HTTP with Basic auth, so the port is published on the host's loopback
only:

```sh
ssh -L 7980:127.0.0.1:7980 <host>      # then open http://127.0.0.1:7980/
```

Log in with any user name and the token as the password. To serve it under a
public name instead, put a TLS-terminating proxy in front and add
`SPACE_ORCHESTRATOR_WEBUI_HOSTS=<name[:port]>` to the orchestrator's
environment. `docs/security-notes.md` lists what the UI does and does not
protect against.

## No browser viewer on this host

The stack used to include a `viewer` service (stock `nginx:alpine` serving a
hand-copied wasm bundle from `./viewer/space-viewer` on port 8098). It is gone:
the browser viewer is a local development and agent-testing tool (`AGENTS.md`,
`crates/space-viewer/README.md`), not part of the deployment.

On a host that still runs the older compose file: copy this directory's
`docker-compose.yaml` to `/home/proxy/stacks/space/`, run
`docker compose up -d --remove-orphans` there (that stops and removes the
`space-viewer` container), then delete `./viewer/` and the `VIEWER_PORT` line
in `.env`.

## Updating a host that runs the older compose file

**Not "at your convenience" any more.** Images built from the shard-manager
change do **not** start under an older compose file: see "Before the
shard-manager images arrive" at the top - `.env` (`DOCKER_GID`, the cluster
secret), the firewall (UDP 7777 and 7870-7899), then this directory's
`docker-compose.yaml` with `docker compose up -d --remove-orphans`, all before
the push. That run also removes what older files left: the `space-shard-a`
container, the `viewer` container, a `--galaxy` in `CLIENT_ARGS`.

## Client configuration

The client image's ENTRYPOINT is a shell wrapper that runs
`space-client --stream "$STREAM_BIND" [--server "$SPACE_SERVER"] $CLIENT_ARGS`,
so the compose file sets those env vars (`STREAM_BIND=0.0.0.0:8099`,
`SPACE_SERVER=host.docker.internal:<PLAYER_PORT>`, `CLIENT_ARGS=` for extra
flags) instead of a `command:` (which the wrapper would ignore).

The client dials the player port **through the host** (`extra_hosts:
host.docker.internal:host-gateway`): one that dialed the orchestrator's
in-network address would be sent on to `<that address>:<shard port>`, where no
shard listens. **That path has not been executed.** The service was kept in
this file because it is not known whether the pixel stream is needed on this
host - an open question for the owner; remove the service if it is not.

Note: the `client` image renders on software Vulkan (lavapipe from Mesa 25,
which has `VK_KHR_acceleration_structure`), so it needs no GPU. Should no usable
Vulkan device be found it exits 1, and it is set to `restart: "no"` so that it is
not restarted in a loop (each start would open a new backend session). While
running it reconnects on its own when its shard is replaced, and nothing a
viewer sends can make it quit. `orchestrator` stays at
`restart: unless-stopped`; the shard containers have no restart policy (the
orchestrator replaces a shard that dies).

## Network subnet

The compose network uses `172.29.0.0/24`. No container has a fixed address any
more: the orchestrator finds its own network and address by inspecting its
container, and the shards it starts join that network. (The subnet is named
because `172.31.0.0/24`, which `deploy/docker-compose.yaml` once used,
collides with an existing docker network on the proxy host; the local file
has no fixed subnet now.)

## Images must be public

The three ghcr packages (`space-shard`, `space-orchestrator`, `space-client`)
are public, so the host, Watchtower and the orchestrator - which pulls the
shard image through the Docker socket when it is not on the host - pull
anonymously (no `docker login`). The orchestrator handles **no registry
credentials**: were the shard image private, the operator would have to
`docker login` and `docker pull` it on the host before the orchestrator
starts.
New packages pushed with `GITHUB_TOKEN` start private: make them public under
Packages > the package > Package settings > Change visibility.

## GitHub secrets for the manual path

Needed only for `deploy.yml`; add them in Settings > Secrets and variables >
Actions:

- `DEPLOY_SSH_HOST` host name or IP of the proxy
- `DEPLOY_SSH_USER` login user (must be allowed to run `docker`)
- `DEPLOY_SSH_KEY` private key whose public half is in that user's
  `~/.ssh/authorized_keys`

## Verification status

**Only parsed:** this compose file, with `docker compose config` against
`.env.example` plus a throwaway secret and `DOCKER_GID`; scripts are checked
with `bash -n`. It has not been run from the repository (it pulls the
published images and belongs on the host).

**Executed elsewhere:** the same topology in `../docker-compose.yaml` - the
orchestrator alone, creating, replacing, scaling and removing its shard
containers - and images built from `../Dockerfile.orchestrator` and
`../Dockerfile.shard`, on Windows 11 with Docker Desktop 28.5.1 (Linux engine
in WSL2), 2026-11-24: see "What is and is not verified" in `../README.md`.

**Not executed:** anything on a Linux host (a socket group other than 0, a
firewall, real published ports to another machine); Watchtower replacing the
orchestrator; a pull of the `sha-` tagged shard image from ghcr; a private
registry; the `client` stream service; `release-images.yml` (YAML-checked
only).

**What still looks wrong, seen on Docker Desktop:** in the WSL2 VM the shards'
`clock offset` ran between 0 and -84 ms (the hypervisor steps the VM's wall
clock), each shard slews by itself, and for seconds at a time two shards were
a tick or two apart: two crossings showed a drawn step of 0.40 m and 0.76 m
and corrections of 0.05 to 1.2 m (four measured right after the start:
3.1-4.6 mm, 0). It is the cluster clock's stated limit, not the launcher's;
not verified on a real Linux host, where one kernel clock serves every
container. Watch the shards' census lines (`docker logs space-shard-<port>`)
for `clock offset` on this host.
