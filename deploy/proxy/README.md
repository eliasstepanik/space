# Proxy deployment bundle

Production stack for the proxy host: orchestrator + `shard-a` (+ optional
pixel-stream client) from prebuilt `ghcr.io/eliasstepanik/space-*` images.

| File | Purpose |
|---|---|
| `docker-compose.yaml` | pulls `space-orchestrator`, `space-shard`, `space-client` (profile `stream`); `restart: unless-stopped`; named volume for orchestrator state |
| `.env.example` | copy to `.env`: cluster name, host ports, image tag, profiles |
| `redeploy.sh` | `docker compose pull --quiet && up -d`, then `docker image prune -f`; logs to stdout |

## Install (one-time, on the host)

```sh
sudo mkdir -p /opt/space && sudo cp docker-compose.yaml redeploy.sh /opt/space/
sudo cp .env.example /opt/space/.env     # then edit
sudo chmod +x /opt/space/redeploy.sh
/opt/space/redeploy.sh
```

## How updates reach the host

1. A merge to `main` touching `crates/`, `deploy/`, `Dockerfile` or `Cargo.*`
   runs `.github/workflows/release-images.yml`, which pushes
   `ghcr.io/eliasstepanik/space-{shard,orchestrator,client}` tagged `latest` and
   `sha-<shortsha>`.
2. **Primary path (set up on the host, NOT part of this repo):** a systemd timer
   runs `/opt/space/redeploy.sh` every 5 minutes. `docker compose pull` fetches
   only when a new digest exists and `up -d` recreates only containers whose
   image changed, so a no-op poll is cheap. Needs no inbound access and no
   secrets in GitHub.
3. **Manual path:** `.github/workflows/deploy.yml` (`workflow_dispatch`, or
   `repository_dispatch` with `event_type: redeploy`) SSHes to the host and
   runs `/opt/space/redeploy.sh`.

Rollback: set `SPACE_TAG=sha-<shortsha>` in `.env` and run `redeploy.sh`.

## Make the images public (one-time, repo owner)

Packages pushed with `GITHUB_TOKEN` start private. After the first push, for
each of `space-shard`, `space-orchestrator`, `space-client`: GitHub > your
profile > Packages > the package > Package settings > Change visibility >
Public. Then the host pulls anonymously (no `docker login`). If you keep them
private instead, run `docker login ghcr.io` on the host with a PAT that has
`read:packages`.

## GitHub secrets for the manual path

These do **not exist yet**; add them in Settings > Secrets and variables >
Actions before using `deploy.yml`:

- `DEPLOY_SSH_HOST` host name or IP of the proxy
- `DEPLOY_SSH_USER` login user (must be allowed to run `docker`)
- `DEPLOY_SSH_KEY` private key whose public half is in that user's
  `~/.ssh/authorized_keys`

## Verification status

Compose and workflows are validated as YAML only. Nothing here has been run: the
authoring sandbox has no Docker daemon and no access to the host.
