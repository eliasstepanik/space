# Proxy deployment bundle

Production stack for the proxy host: orchestrator + `shard-a` (+ optional
pixel-stream client) from prebuilt `ghcr.io/eliasstepanik/space-*` images.

| File | Purpose |
|---|---|
| `docker-compose.yaml` | pulls `space-orchestrator`, `space-shard`, `space-client` (profile `stream`); `restart: unless-stopped`; named volume for orchestrator state |
| `.env.example` | copy to `.env`: cluster name, host ports, image tag, profiles |
| `redeploy.sh` | flock-serialized: `compose --profile '*' pull --quiet`, `up -d --remove-orphans`, `docker image prune -f`; logs to stdout |
| `check-updates.sh` | compares remote vs local digests of the 3 images; runs `redeploy.sh` if any differ. Exit 0 no change, 10 redeployed, 20 error |
| `space-redeploy.nix` | NixOS module: oneshot service + 5-minute timer running `check-updates.sh` |

## Install (one-time, on the host)

```sh
sudo mkdir -p /opt/space && sudo cp docker-compose.yaml redeploy.sh check-updates.sh /opt/space/
sudo cp .env.example /opt/space/.env     # then edit
sudo chmod +x /opt/space/redeploy.sh /opt/space/check-updates.sh
/opt/space/redeploy.sh
# then install the poll timer (see below), e.g. on NixOS:
#   cp space-redeploy.nix /etc/nixos/ and add it to `imports`, then nixos-rebuild switch
```

## How updates reach the host

1. A merge to `main` touching `crates/`, `deploy/`, `Dockerfile` or `Cargo.*`
   runs `.github/workflows/release-images.yml`, which pushes
   `ghcr.io/eliasstepanik/space-{shard,orchestrator,client}` tagged `latest` and
   `sha-<shortsha>`.
2. **Primary path (poll timer on the host):** a systemd timer runs
   `/opt/space/check-updates.sh` every 5 minutes. It compares the remote digest
   (`docker buildx imagetools inspect`) of `space-{shard,orchestrator,client}`
   with the local `RepoDigests`; if any differ it runs `redeploy.sh`. Images
   that are not found / denied / 401 / 404 (not yet public) count as "no
   change", quietly. Exit codes: `0` no change, `10` redeployed, `20` error.
   Needs no inbound access and no secrets in GitHub.
3. **Manual path:** `.github/workflows/deploy.yml` (`workflow_dispatch`, or
   `repository_dispatch` with `event_type: redeploy`) SSHes to the host and
   runs `/opt/space/redeploy.sh`. `redeploy.sh` takes a flock on
   `/var/lock/space-redeploy.lock`, so a manual run and the timer never overlap.

### Poll timer unit spec

On the proxy host (NixOS) this is `space-redeploy.nix` (in this directory). The
equivalent systemd units:

```ini
# space-redeploy.service
[Unit]
Description=Check ghcr for new space images and redeploy
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=oneshot
SuccessExitStatus=10
ExecStart=/opt/space/check-updates.sh

# space-redeploy.timer
[Unit]
Description=Poll for new space images every 5 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
```

`SuccessExitStatus=10` keeps "redeployed" from showing up as a failed unit.
Logs: `journalctl -u space-redeploy`.

## Network subnet

The compose network uses `172.29.0.0/24` (orchestrator `.10`, shard-a `.11`,
client `.12`). `deploy/docker-compose.yaml` uses `172.31.0.0/24`, which collides
with an existing docker network on the proxy host, hence the different subnet
here.

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

Compose and workflows are validated as YAML only; the scripts are checked with
`bash -n`. Nothing here has been run in the authoring sandbox (no Docker daemon,
no access to the host).
