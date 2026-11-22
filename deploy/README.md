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
docker compose -f deploy/docker-compose.yaml up --build            # orchestrator + shard-a
docker compose -f deploy/docker-compose.yaml --profile stream up --build   # + client
```

Fixed IPs on the `172.31.0.0/24` bridge (addresses are parsed as socket
addresses, not resolved): orchestrator `.10:7900`, shard-a `.11:7870`, client
`.12:8099`.

## Publishing pipeline

- **Images**: every push to `main` touching `crates/**`, `deploy/**`,
  `Dockerfile` or `Cargo.*` runs `.github/workflows/release-images.yml`, which
  builds and pushes `ghcr.io/eliasstepanik/space-shard`, `space-orchestrator`
  and `space-client` as `:latest` and `:sha-<shortsha>` (the `space/*:dev` names
  above are local compose builds only). The repo owner must make the packages
  public after the first push. `ci.yml`'s `docker` job still only builds.
- **Release**: pushing a `v*` tag runs `.github/workflows/release.yml`: builds
  `space-client` (`--release --locked`) on Linux and Windows plus the wasm viewer,
  and attaches `space-client`, `space-client.exe` and `space-viewer.zip`
  (`index.html` + `pkg/`) to a GitHub Release with generated notes.
- **Proxy**: `deploy/proxy/` is the production bundle (compose pulling the ghcr
  images, `.env.example`, `redeploy.sh`). Watchtower on the host (label opt-in,
  60s poll) redeploys on new images; `.github/workflows/deploy.yml` is the manual
  SSH path (secrets `DEPLOY_SSH_HOST/USER/KEY`, not yet created). See
  `deploy/proxy/README.md`.
- **Workflow lint**: `ci.yml`'s `lint` job parses every workflow and deploy YAML
  (`.github/scripts/check_yaml.py`), because GitHub silently drops a workflow
  with invalid YAML.

## Streaming client and the wasm viewer

`Dockerfile.client` starts `space-client --stream $STREAM_BIND [--server
$SPACE_SERVER] $CLIENT_ARGS`. Compose sets `SPACE_SERVER=172.31.0.11:7870`
(shard-a) and `CLIENT_ARGS=--galaxy`. It renders on lavapipe (software Vulkan);
bookworm's Mesa 22.3 lavapipe lacks `VK_KHR_ray_query`, so GI is off in that
image (the renderer logs it and carries on). Pass `/dev/dri` through or use
Mesa >= 25 for the ray-query path.

The client's port speaks WebSocket only. There is **no** static-file flag, so the
wasm viewer is served separately: build it (`crates/space-viewer/README.md`:
`cargo build --target wasm32-unknown-unknown`, then `wasm-bindgen --target web
--out-dir crates/space-viewer/pkg`) and serve `crates/space-viewer/` with any
static server, e.g. `python3 -m http.server 8000`. Open
`http://127.0.0.1:8000/index.html?ws=ws://127.0.0.1:8099`; the `ws` query
parameter is the viewer's only configuration (default `ws://127.0.0.1:8099`).
`ws://` only, no TLS; put a TLS-terminating proxy in front for `wss://`.

## What is and is not verified

Verified (run in the authoring sandbox): the binaries' flags match the
Dockerfiles' entrypoints (read from the `--help`/parser sources), the workspace
builds with `--locked`, `space-viewer` builds for `wasm32-unknown-unknown`, and
the full `cargo run -p xtask -- ci` passes.

**Not verified: every Dockerfile and the compose file.** The sandbox has no
Docker daemon, so nothing here has been built or run (see
`docs/security-notes.md`). Correct by inspection only; in particular the
`rust:1.100-bookworm` tag, the `apt` package names in `Dockerfile.client`, and
the compose profile have never been exercised. CI's `docker` job
(`.github/workflows/ci.yml`) builds each Dockerfile (no push, no run) and is the
first real check. Container runtime behaviour (client reaching the shard over the
bridge, viewer connecting) has no automated test.

The CI GPU step is `continue-on-error: true` for the same reason: it has not run
on a hosted runner.
