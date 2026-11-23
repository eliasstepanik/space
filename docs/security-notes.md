# Security notes: known gaps and deferred items

This file records what the cluster does **not** protect against today. It
describes current behaviour, not intent.

## Peer (shard↔shard) authentication: current behaviour

* **Transport.** QUIC with a self-signed certificate (`space-net/src/crypto.rs`)
  and no certificate pinning. A dialler cannot verify which process it reached,
  and an acceptor requests no client certificate.
* **Hello.** `space_protocol::check_hello` checks the protocol version only.
  A peer identifies itself with the plain string `shard:<id>`
  (`space-server/src/net.rs`, `peer_shard_id`); any client can type it.
* **Orchestrator.** `ControlPlane::handle_shard_hello` checks the cluster name
  only. It *issues* a `peer_token` in `ShardWelcome`, but that token is never
  presented on peer connections and never verified anywhere.
* **Shard.** `Shard::peer_message` checks only that the claimed shard id is in
  the shard's own map (and a commit must come from the shard that staged the
  transfer). A party that can reach a shard's port and guesses a valid
  neighbour id can speak the handoff protocol as that neighbour — and send
  `GhostUpdate`s as it (ghosts are kinematic copies the receiver's bodies
  collide with; their component lists are decoded with the same strict,
  total decoders as a handoff's; a ghost never replaces a body the receiver
  simulates, and a removal is honoured only from the shard that sent it).
* **Control port.** The orchestrator's control endpoint binds to `127.0.0.1` by
  default. That is the only mitigation for the items below.

## Deferred items

1. **`peer_token` enforcement in `Shard::peer_message`.** Shards should present
   the token they were issued, and receivers should verify it before acting on
   a peer message. Today the field is a wire slot with no consumer.
2. **Per-shard hello credentials.** The hello should carry a credential bound to
   one shard id (and be checked by the orchestrator), instead of a bare
   `shard:<id>` string, plus certificate pinning or a separate ALPN for peer
   traffic. A signed handoff ticket would additionally stop a legitimate shard
   forging handoffs for entities it does not own.
3. **Endpoint-table poisoning via claimed shard ids.** The orchestrator records
   the endpoint a `ShardHello` names for the shard id it claims, and pushes
   those endpoints to every shard in a `DirectoryReply`. A party that can reach
   the control port can register under a claimed id and redirect peer traffic
   or client redirects. This is mitigated today only by the loopback-default
   control bind; exposing the control port on a non-loopback address removes
   that mitigation.

## Orchestrator web UI (`space-orchestrator --webui`)

An admin surface: it shows cluster state and can split, merge, pause the
autoscaler, re-send the map and switch the scene on every shard. Off by default.

* **Transport and auth.** Plain HTTP (no TLS in-tree, by project rule) with HTTP
  Basic auth: any user name, the token as the password, compared in constant
  time. On the wire the token is readable, so the UI belongs behind an SSH
  tunnel or a TLS-terminating proxy.
* **Tokenless mode is loopback-only by construction.** A loopback bind may run
  without a token; it then serves only loopback `Host`s (`localhost`,
  127.0.0.0/8, `[::1]`) and refuses any request carrying `Forwarded`,
  `X-Forwarded-*`, `X-Real-IP` or `Via`. Any other bind, or any `--webui-host`,
  refuses to start without a token of at least 16 characters.
  **Residual:** a reverse proxy on the same host that adds none of those headers
  and rewrites `Host` to `127.0.0.1` makes a tokenless UI reachable by whoever
  reaches the proxy. Never proxy a tokenless UI. Tokenless loopback also trusts
  every local process, including anything on the host that can be made to issue
  HTTP requests.
* **CSRF / rebinding.** Actions are POST only and need a per-process form token;
  `Origin`, when sent, must match `Host` (or a listed `name[:port]`; the scheme
  cannot be checked behind a TLS proxy) and `Sec-Fetch-Site` must be
  same-origin; `Host` must be an IP literal, `localhost` or a listed name. No
  CORS headers are ever sent. The form token is also in `/api/state` (readable
  only with the credentials) and stays valid until restart.
* **Availability.** Head/body/connection limits and deadlines; one operator
  action at a time, 250 ms apart (429 otherwise); `--max-regions` (default 64)
  bounds how far splits can grow the map. A token holder can still churn the
  256-entry event log at that rate.
* **`SceneAck` is unauthenticated** like the rest of the control port: a peer
  that can reach it can fake "shard N confirmed the scene" (display only; it
  cannot change the scene). Same for register/lost pairs in the event log by a
  peer that knows the cluster name. Both need control-port authentication
  (deferred item 2).
* **Scene delivery to clients** is repeated periodically by the shard, so a
  dropped `SceneUpdate` heals without a reconnect.

## Client harness (`space-client --harness`, `--script`)

Remote control of a client plus script execution, for tests and AI agents. Off
by default.

* **HTTP API.** Plain HTTP. A loopback bind needs no credentials; any other
  bind is refused unless `SPACE_HARNESS_TOKEN` holds a token of at least 16
  printable characters, and then every request needs `Authorization: Bearer`
  (constant-time compare; the token crosses the network in clear, so tunnel
  it). `Host` must be an IP literal, `localhost` or a listed name (DNS
  rebinding); any `Origin` that is not the API's own, and any `Sec-Fetch-Site`
  other than same-origin/none, is refused, and no CORS header is ever sent - a
  web page cannot drive the client. Head, body, connection and time limits as
  in the web UI. Loopback without a token trusts every local process.
* **Lua scripts** arrive from a file (`--script`) or over the API (`POST
  /script`). The interpreter is `lunita`, a pure-Rust Lua VM (a fork of
  `piccolo` hardened for untrusted input), pinned to one exact version. A
  script gets `math`, `string`, `table` and the `client` table only - no `os`,
  `io`, `package`, `debug`, `load` or file access - with a wall-clock limit, a
  memory limit and bounded VM steps, so a runaway script is always stopped.
  `client.screenshot` writes only below the configured output directory
  (relative names, no `..`, no device names, link escapes checked).
  **Residual:** `lunita` is a young, single-maintainer crate that has not been
  audited here beyond a scan for file, network, process and FFI use (none
  found); an upgrade should be a reviewed change. A script can do anything a
  user at the keyboard can: connect anywhere the machine can reach, and fill
  the output directory up to the screenshot cap.
* **Never drive a client with OS-level input** (SendKeys, xdotool, clipboard):
  that is not a harness, it types into whatever has focus. See `AGENT.md`.

## Known limitations (Phase 3 close-out)

* The H13 frame-gap test relies on lockstepped harness ticks; production continuity rests on the separately-tested commit-gate machinery.
* The Dockerfiles and compose files are correct-by-inspection only (no Docker daemon in CI).
* The Phase-4 present path (swapchain) is validated by build + clippy only (no window system in CI).
