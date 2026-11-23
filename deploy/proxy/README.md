# Proxy deployment bundle

Production stack for the proxy host: orchestrator + `shard-a` + the ws://
pixel-stream client from prebuilt `ghcr.io/eliasstepanik/space-*` images,
auto-redeployed by Watchtower.

| File | Purpose |
|---|---|
| `docker-compose.yaml` | pulls `space-orchestrator`, `space-shard`, `space-client`; all three carry the Watchtower opt-in label; `restart: unless-stopped` (except `client`: `restart: "no"`, see Client configuration); orchestrator state in `./state` |
| `.env.example` | copy to `.env`: cluster name, host ports, image tag |
| `redeploy.sh` | manual path only (used by `.github/workflows/deploy.yml`): `compose pull`, `up -d`, image prune |

## Install (one-time, on the host)

The stack lives in `/home/proxy/stacks/space/` (no sudo needed):

```sh
mkdir -p /home/proxy/stacks/space/state
cp docker-compose.yaml /home/proxy/stacks/space/
cp .env.example /home/proxy/stacks/space/.env     # then edit
chmod a+rwx /home/proxy/stacks/space/state        # orchestrator runs as uid 10001
cd /home/proxy/stacks/space && docker compose up -d
```

## How updates reach the host

1. Every push to `main` runs `.github/workflows/release-images.yml`, which
   pushes `ghcr.io/eliasstepanik/space-{shard,orchestrator,client}` tagged
   `sha-<shortsha>` and then moves `latest` to that image.
2. **Primary path (Watchtower):** the Watchtower container already running on
   the host polls every 60s. It only touches containers that opt in with the
   label `com.centurylinklabs.watchtower.enable: "true"` (set on `orchestrator`,
   `shard-a` and `client`). When the `:latest` digest changes it pulls the new
   image and recreates the container with the same configuration. Nothing to
   install or schedule per stack.
3. **Manual path:** `.github/workflows/deploy.yml` (`workflow_dispatch`, or
   `repository_dispatch` with `event_type: redeploy`) SSHes to the host and runs
   `redeploy.sh`. It is no longer the primary path.

Watchtower only follows the tag the container runs. Rollback: set
`SPACE_TAG=sha-<shortsha>` in `.env` and run `docker compose up -d`; set it back
to `latest` to resume auto-updates.

## Scene and the operator web UI

`SPACE_SCENE` in `.env` is the scene the cluster starts with (`demo-area`,
which includes the solar system, `materials`, `galaxy`). Once an operator switches it in the web UI the
choice is saved in `./state/orchestrator.id.scene` and wins over `.env` on every
restart; delete that file to go back to `SPACE_SCENE`.

The web UI (status, split / merge, autoscaler pause, scene switch) is **off**
until `.env` sets both `SPACE_WEBUI_BIND=0.0.0.0:7980` and a `SPACE_WEBUI_TOKEN`
of at least 16 random characters (a bind without a token makes the orchestrator
refuse to start, which takes the cluster down - set both or neither). It is
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

Images built from the online-only rework still start under the older compose
file: the client ignores the removed scene flag in `CLIENT_ARGS=--galaxy` (with
a warning in its log) and reconnects on its own, and orchestrator, shard and
client all speak wire protocol 4 once Watchtower has pulled the three of them.
Then, at your convenience, copy this directory's `docker-compose.yaml` and the
new entries of `.env.example` to the host and run
`docker compose up -d --remove-orphans`: that drops `--galaxy`, removes the old
`viewer` container, and makes the scene and the web UI configurable.

## Client configuration

The client image's ENTRYPOINT is a shell wrapper that runs
`space-client --stream "$STREAM_BIND" [--server "$SPACE_SERVER"] $CLIENT_ARGS`,
so the compose file sets those env vars (`STREAM_BIND=0.0.0.0:8099`,
`SPACE_SERVER=172.29.0.11:7870`, `CLIENT_ARGS=` for extra flags) instead of a
`command:` (which the wrapper would ignore).

Note: the `client` image renders on software Vulkan (lavapipe from Mesa 25,
which has `VK_KHR_acceleration_structure`), so it needs no GPU. Should no usable
Vulkan device be found it exits 1, and it is set to `restart: "no"` so that it is
not restarted in a loop (each start would open a new backend session). While
running it reconnects on its own when the shard restarts, and nothing a viewer
sends can make it quit. `orchestrator` and `shard-a` stay at
`restart: unless-stopped`.

## Network subnet

The compose network uses `172.29.0.0/24` (orchestrator `.10`, shard-a `.11`,
client `.12`). `deploy/docker-compose.yaml` uses `172.31.0.0/24`, which collides
with an existing docker network on the proxy host, hence the different subnet
here.

## Images must be public

The three ghcr packages (`space-shard`, `space-orchestrator`, `space-client`)
are public, so both the host and Watchtower pull anonymously (no `docker login`).
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

This compose file is checked with `docker compose config` against
`.env.example`; scripts are checked with `bash -n`. It has not been run from the
repository (it pulls the published images and belongs on the host). The images
themselves, and the same topology in `../docker-compose.yaml`, have been built
and run locally: see "What is and is not verified" in `../README.md`.
