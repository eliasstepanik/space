# Phase-0 spike 0.6 — Docker-as-orchestrator mechanics

**Goal.** Prove the primitives the Phase-3 `space-orchestrator` needs: from
*inside* a container, use the Docker Engine API (via [`bollard`]) to create,
health-check, list (by label), remove and orphan-reap sibling "shard"
containers, emitting a JSON report of every step.

This is a **throwaway spike**, standalone (its `Cargo.toml` has an empty
`[workspace]` table and is **not** a member of the root `space` workspace).

[`bollard`]: https://crates.io/crates/bollard

## Result in this sandbox: NO Docker daemon

This sandbox has **no Docker daemon and no Docker CLI**. Verbatim:

```
$ docker info
sh: 1: docker: not found
$ which docker
(no output; exit 1)
$ ls -la /var/run/docker.sock
ls: cannot access '/var/run/docker.sock': No such file or directory
```

So the end-to-end create→health→list→remove→reap cycle **cannot run here**.
Per the spike brief, the valid fallback is: validate what we can with unit tests
of the pure helpers, and record that the real end-to-end check still needs a
home. **It has not found one yet** — see "Docker E2E: no CI home yet" below; the
E2E run is an **entry criterion for Phase 3**, not something CI already does.

`report-sandbox.json` (produced by `cargo run --release`) documents the blocker
with the exact error text:

```json
{
  "all_steps_ok": false,
  "bollard_stubs": "1.53.1-rc.29.3.1",
  "bollard_version": "0.21.1",
  "cluster_id": "spike-0.6",
  "host_port_range": [46001, 46064],
  "image": "alpine:3.20@sha256:d9e853e87e55526f6b2917df91a2115c36dd7c696a35be12163d44e6e2a4b6bc",
  "shard_udp_port": 46000,
  "spike": "0.6",
  "steps": [
    {
      "detail": { "connected": false },
      "error": "Socket not found: /var/run/docker.sock",
      "ok": false,
      "step": "detect_docker"
    }
  ]
}
```

The binary reaches `Docker::connect_with_defaults()` (which resolves
`DOCKER_HOST`, defaulting to `unix:///var/run/docker.sock`) and fails cleanly at
the socket with `Socket not found: /var/run/docker.sock` — exactly the expected
behaviour with no daemon. It exits 0 with valid JSON (a blocked spike is a
valid, important result, matching the `vkprobe` spike's convention).

## What was validated here (without a daemon)

**Scope, stated precisely:** the **13 unit tests** (`cargo test`) cover **pure,
daemon-independent helpers only** — port-range parsing, the shard label map, the
`ContainerCreateBody` construction, and reading a published host port out of a
`ContainerInspectResponse`. They do **not** exercise the **bollard call
sequence**: no test performs a `ping()`, `create_container`, `start_container`,
`inspect_container`, `list_containers` or `remove_container` call, and no test
verifies that the async step functions call them in the right order or handle
their errors. The bollard client construction is likewise only checked
indirectly (the binary reaches `connect_with_defaults()` and fails cleanly at
the missing socket).

The tests cover:

- `parse_port_range` — parsing the configurable `HOST_PORT_RANGE`
  (`"46001-46064"`), including whitespace, single-port, reversed and garbage
  inputs.
- `shard_labels` — the `space.cluster=<id>` + `space.role=shard` label map.
- `shard_create_body` — the `ContainerCreateBody`: image pinned by digest
  (`alpine:3.20@sha256:…`), `sleep infinity` dummy shard command, exposed UDP
  port, `PortBindings` publishing the container UDP port to a host port from the
  range, labels, and `restart_policy = no`.
- `allocated_host_port` — reading the published host port out of a
  `ContainerInspectResponse` (the health-check's port-allocation assertion),
  including the unbound / no-network-settings cases.
- `Step` JSON shape — the report entry format.

The image digest was resolved live from the Docker Hub registry API
(`docker-content-digest` for `library/alpine:3.20`), not guessed.

## Docker E2E: no CI home yet — ENTRY CRITERION for Phase 3

The end-to-end create→health→list→remove→reap cycle has **never run anywhere**:
this sandbox has no daemon, and **no CI job runs it** (`.github/workflows/ci.yml`
has no Docker-daemon job). So the bollard call sequence is **unverified code**.

Therefore a real Docker E2E is an **ENTRY CRITERION for Phase 3** (before the
Phase-3 orchestrator work is considered started): a CI job must actually execute
this binary against a daemon — either a compose job
(`DOCKER_GID=… docker compose up --build` and assert `all_steps_ok: true`) or an
`xtask e2e` step that runs it — and must fail on any non-`ok` step. Until that
exists, treat every claim in "What CI (with a daemon) will run" below as
**unproven**.

**Since then:** the spike was ported to `crates/space-orchestrator/src/launcher/docker.rs`,
and the CI home is the `docker-e2e` job of `.github/workflows/ci.yml`
(`cargo run -p xtask -- e2e`): it runs the production launcher through
`deploy/docker-compose.yaml` — create, list, remove on shutdown — not this binary,
which still has never been run against a daemon.

## What a daemon host would run (not yet executed anywhere)

On a host with a Docker daemon, the same binary runs the full cycle and prints
all five steps with `"ok": true`:

1. **detect_docker** — `connect_with_defaults()` + `ping()`.
2. **create_shard** — pull `alpine` by digest, `create_container` (UDP port
   published from `HOST_PORT_RANGE`), `start_container`.
3. **health_check** — `inspect_container`: state running **and** the published
   host port allocated.
4. **list_shards** — `list_containers` filtered by
   `label=space.cluster=<id>`, `label=space.role=shard`.
5. **remove_shard** + **reap_orphans** — `remove_container` by id (force, `-v`),
   then remove any labelled shard not in the expected set.

Run it directly against a daemon:

```sh
cd spikes/orch-docker
HOST_PORT_RANGE=46001-46064 SPACE_CLUSTER_ID=spike-0.6 cargo run --release
```

Or via the compose template (which mounts the socket and adds the non-root
user to the docker group):

```sh
DOCKER_GID=$(getent group docker | cut -d: -f3) docker compose up --build
```

## Files

| File | Purpose |
| --- | --- |
| `Cargo.toml` | Standalone package; `bollard = "0.21.1"` (stubs `=1.53.1-rc.29.3.1`). |
| `src/main.rs` | The orchestrator probe + 13 unit tests. |
| `Dockerfile` | Multi-stage build → minimal runtime, non-root `orchestrator` (uid 10001). |
| `docker-compose.yaml` | Phase-8 template: socket mount, `group_add` docker group, explicit `172.31.99.0/24` subnet. |
| `report-sandbox.json` | Sandbox run (documents the no-daemon blocker). |

## Commands used (verbatim results)

```
$ RUSTUP_TOOLCHAIN=stable cargo test
test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

$ RUSTUP_TOOLCHAIN=stable cargo run --release   # -> report-sandbox.json
exit:0   # all_steps_ok=false, detect_docker error "Socket not found: /var/run/docker.sock"
```

## Phase-8 notes (template refinements)

- **Subnet comment mismatch (Phase-8 item).** `docker-compose.yaml` declares an
  explicit `172.31.99.0/24` subnet on `orch_net` and its comment says this
  avoids address-pool exhaustion "when many shard networks are created" — but
  the spike's **shards are created on the host's default bridge** (they are
  siblings created through the socket, not attached to `orch_net`; the compose
  header comment says so itself). The explicit subnet therefore only constrains
  the orchestrator's own network and does nothing for shard address pools. Fix
  the comment or attach shards to the cluster network when Phase 3 moves them
  off the default bridge.
- **uid 10001 vs P8's uid 1000 (Phase-8 item).** The runtime image creates the
  `orchestrator` user at **uid 10001**, while the Phase-8 packaging plan uses
  **uid 1000**. Reconcile before the real image is built (volume ownership and
  the socket-proxy policy both key off the uid).
- The compose `subnet`/`gateway` and `HOST_PORT_RANGE` should be parameterised
  per cluster to avoid address-pool and host-port exhaustion at scale.
- The runtime image should drop the `docker` group in favour of a scoped socket
  proxy (e.g. a TLS-authenticated socket or a policy-enforcing proxy) rather
  than raw socket access.
- Shard containers are siblings on the host bridge; Phase 3 will move them onto
  the cluster network and add the epoch-versioned shard map on top of the
  label-based discovery proven here.
