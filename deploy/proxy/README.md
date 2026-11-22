# Proxy deployment bundle

Production stack for the proxy host: orchestrator + `shard-a` + the ws://
pixel-stream client from prebuilt `ghcr.io/eliasstepanik/space-*` images,
auto-redeployed by Watchtower.

| File | Purpose |
|---|---|
| `docker-compose.yaml` | pulls `space-orchestrator`, `space-shard`, `space-client`; all three carry the Watchtower opt-in label; `restart: unless-stopped`; orchestrator state in `./state` |
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

1. A merge to `main` touching `crates/`, `deploy/`, `Dockerfile` or `Cargo.*`
   runs `.github/workflows/release-images.yml`, which pushes
   `ghcr.io/eliasstepanik/space-{shard,orchestrator,client}` tagged `latest` and
   `sha-<shortsha>`.
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

## Client configuration

The client image's ENTRYPOINT is a shell wrapper that runs
`space-client --stream "$STREAM_BIND" [--server "$SPACE_SERVER"] $CLIENT_ARGS`,
so the compose file sets those env vars (`STREAM_BIND=0.0.0.0:8099`,
`SPACE_SERVER=172.29.0.11:7870`, `CLIENT_ARGS=--galaxy`) instead of a
`command:` (which the wrapper would ignore).

Note: the `client` service requires a GPU with `VK_KHR_acceleration_structure`
(ray tracing). On hosts without one (e.g. software Vulkan such as lavapipe) it
exits, and it is set to `restart: "no"` so it is not restarted (a crash loop
would churn a new backend session each time). `orchestrator` and `shard-a` stay
at `restart: unless-stopped`.
=======
>>>>>>> refs/remotes/origin/main

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

Compose and workflows are validated as YAML only; scripts are checked with
`bash -n`. Nothing here has been run in the authoring sandbox (no Docker daemon,
no access to the host).
