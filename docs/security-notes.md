# Security notes: known gaps and deferred items

This file records what the cluster does **not** protect against today. It
describes current behaviour, not intent.

## Shard registration and peers: current behaviour

* **Transport.** QUIC with a self-signed certificate (`space-net/src/crypto.rs`)
  and no certificate pinning. A dialler cannot verify which process it reached,
  and an acceptor requests no client certificate. Nothing below changes that.
* **Registration (shard → orchestrator).** `ShardHello` carries the **cluster
  secret** (protocol 7). `ControlPlane::check_credentials` checks — when a
  secret is required — the secret first, in constant time; a wrong or missing
  one is a final refusal (`Unauthorized`) that says nothing else, in
  particular not what the cluster is called, and is logged without the
  secret. Only then is the cluster name compared. A secret is required when
  one was configured (`--cluster-secret[-file]`, `SPACE_CLUSTER_SECRET[_FILE]`)
  and always when the control port is not on loopback. Off loopback the
  orchestrator refuses to start without a secret a shard could present: a
  configured one of at least 16 characters, or — with `--min-shards` 1 or
  more — one it generates (from the web UI form token's source, see below)
  and hands to the shards it starts **in their environment**, never on a
  command line. On a loopback control port with nothing configured a hello
  needs no secret: that trusts every local process, like the tokenless web UI
  and harness. **"Loopback" is the bind address, not who can reach it:** a
  loopback control port that a same-host UDP forwarder points at, or that a
  host-network container shares, takes hellos from remote peers without a
  secret — configure a cluster secret before doing either.
* **The hello gates the connection.** A control connection's task hands the
  `ShardHello` to the run loop and reads nothing more until the verdict is
  in (`control.rs`, `Inbound`). Refused — for good or "not yet" — the reason
  is delivered and the connection **closed**; a shard that asks again dials a
  new one. Welcomed, the connection is admitted as **one shard id** for its
  life: a `LoadReport`, `EpochAck` or `SceneAck` is passed on only if it names
  that id (others are dropped, counted and — a few times per ten seconds —
  written to standard error), and the run loop takes the sender from the
  connection, never from the message. So nobody who has not been welcomed is
  heard at all, and a welcomed shard speaks for itself only.
* **Peers (shard ↔ shard).** The orchestrator issues a **peer token** in
  `ShardWelcome`: one it generates at start, or a configured one
  (`--peer-token[-file]`, `SPACE_ORCHESTRATOR_PEER_TOKEN[_FILE]`) of at least
  16 characters — a shorter or empty one is refused at startup, because an
  empty token is what a shard takes for "no peer token" and it then lets every
  peer in — and at most 1024 (`space_net::MAX_SECRET_LEN`): it has to fit a
  hello, which is read with a size limit of its own (see "Every listening
  port" below). A shard presents the token in every hello to another shard
  (`shard:<id>:<token>` in the hello's token field) and refuses a peer hello
  without it before reading a peer message (`space-server/src/net.rs`,
  `peer_token_admits`; `Unauthorized`; counted, and written to the shard's log
  at most three times per ten seconds — anybody who reaches the game port can
  cause it). A shard that holds none — standalone shards wired with `--peer` —
  accepts peers on the `shard:<id>` string alone. `Shard::peer_message` still
  checks that the claimed id is in the shard's own map, and a commit must come
  from the shard that staged the transfer.
* **Ids and regions are the orchestrator's.** A hello names neither; a region
  is given through an epoch change, never claimed, and an id is used by one
  process, once (a shard that restarts replaces itself under a new id). A
  hello that advertises the address of a **live** registered shard takes that
  shard's place only if its control connection comes from the IP it
  advertises, and only when the replacement can be carried out at once;
  otherwise it waits, and the live shard is untouched until its own link
  closes.
* **Secrets are not printed.** Both secrets are held in `space_net::Secret`,
  whose `Debug` shows `<redacted>` — so do the structs that hold one and derive
  `Debug`: the orchestrator's config, `ControlPlane` and `Orchestrator`, a
  shard's `ShardNet` and `RunningShard`. The messages that carry one have a
  hand-written `Debug`: `ShardHello`, `ShardWelcome`, and `Hello` — whose token
  field is the peer token between shards (`shard:<id>:<redacted>`) and a
  player's token otherwise, and which an endpoint prints when it is sent one
  it did not expect. The launcher's `SpawnSpec` and a shard's
  `OrchestratedOptions` print whether there is a secret, not the secret. Each
  of those structs and messages has a test that prints it and looks for the
  secret. The event log, the snapshot and `/api/state` hold neither secret (a
  snapshot — which carries the event log — is checked for the cluster secret
  in `tests/lifecycle.rs`). What this does **not** cover: a secret given
  as a command-line flag shows in the process list (use the environment or a
  file), and anything that formats `Secret::expose()` or the fields of a wire
  message by hand.

## Deferred items

1. **The peer token is one secret for the whole cluster.** It says "a shard of
   this cluster", not which one: a shard that holds it — or anybody who learns
   it — can claim a neighbour's id, speak the handoff protocol as that
   neighbour and send `GhostUpdate`s as it (ghosts are kinematic copies the
   receiver's bodies collide with; their component lists are decoded with the
   same strict, total decoders as a handoff's; a ghost never replaces a body
   the receiver simulates, and a removal is honoured only from the shard that
   sent it). A credential bound to one shard id, certificate pinning or a
   separate ALPN for peer traffic, and a signed handoff ticket are not built.
2. **Both secrets cross the network in clear inside QUIC/TLS to an unverified
   peer.** The channel is encrypted but, with no pinning, not authenticated: a
   party that can intercept a shard's first packets to the orchestrator
   (on-path, same L2) can terminate the handshake itself and read the cluster
   secret. Keep the control port and the shard ports on a network you trust,
   or tunnel them.
3. **The secret is as strong as its source.** A configured secret is the
   operator's; a generated one comes from `webui::auth::random_token` (128
   bits built on the standard library's `RandomState` keys — see its docs for
   what that does and does not promise).
4. **A registered shard is believed about itself.** It advertises its own
   address, which the orchestrator publishes to every shard and (through
   redirects) to clients; its load reports, its epoch acks and what it says
   about its scene are taken at its word — a `SceneAck` naming a revision a
   little ahead of the cluster's makes the orchestrator re-issue the scene
   above it (at most once a second, never further than
   `MAX_REVISION_CATCH_UP`), and a shard that acknowledges a map change it did
   not apply lets it commit. A party that holds the cluster secret can register
   and do all of that. What it cannot do any more is speak **for another
   shard**: a report or an ack that names a shard other than the one its
   connection was admitted as is dropped. Two things remain between shards of
   one host: a hello that names a live shard's address is believed when it
   comes from the same IP (every shard on loopback, or several behind one
   address), so one registered party there can take another's regions by
   naming its address; and a late `EpochAck` carrying the number of an epoch
   that was aborted and proposed again still counts (`epoch.rs`, "Known
   limit").
5. **Children of the `process` launcher inherit the orchestrator's
   environment, less what is the orchestrator's own**: every variable named
   `SPACE_ORCHESTRATOR_*` (the web UI token, the peer token, their file forms)
   and `SPACE_HARNESS_TOKEN` are taken out, and the cluster secret is set to
   the one the launch is handed (or removed). Anything else that is in the
   environment the orchestrator was started from — another program's
   credentials, say — still reaches every shard: start it from a clean
   environment. The executable is started by its absolute path (never looked
   up in `PATH`). The pid file and the per-shard logs are written where
   `--shard-log-dir` says, with the process's default permissions; the pid
   file lists the processes that are running, and a stop script must still
   check what a pid is before it stops it (the dev scripts compare the
   process's executable path with the checkout's own).
6. **A map change waits for everybody, and three strikes take a while.** A
   shard that holds its control link but does not acknowledge map changes
   holds each of them up for its 3 s deadline and is only taken for dead —
   and replaced — after three in a row. Whoever can register can therefore
   stall map changes for about nine seconds per registration, each costing a
   shard id.

## Every listening port: the first frame and the last word

What holds for the player port, the control port and every shard's game port
alike (`space-net/src/endpoint.rs`, `framed.rs`):

* **The first frame is read as a hello, with a hello's size limit.** Whoever
  opens a connection has proved nothing yet, and the first frame it sends is
  read with a limit of 4 KiB (`MAX_HELLO_PAYLOAD`; a real hello is about 60
  bytes, 1051 at most between shards with the longest peer token). A header
  that declares more is refused from the header alone. Before, the general
  frame limit applied: eight bytes from a stranger made the receiver allocate
  up to 1 MiB and wait for it until the handshake timeout. A dialer whose
  hello would not fit is told so before it dials. Frames after the hello are
  still read with the 1 MiB limit — from a peer that was welcomed.
* **The last word is written with a time limit.** A `Reject` or a
  `JoinRedirect` that ends a connection (`Conn::turn_away`, `Conn::reject`,
  the version-mismatch answer) is written for at most a second
  (`LAST_WORD_TIMEOUT`) and the close then waits at most another for it to be
  read. A QUIC write waits for the peer's flow-control credit; a peer that
  says hello and grants a receive window of zero could otherwise keep the
  writing task — and whatever it holds — for as long as it keeps its
  connection alive.
* **A hello carries a token, and a player's may not be empty.** A shard
  refuses a client hello without a token (`Shard::join`). A staged handoff
  that has no player — a body, a ship nobody flies — is staged under the
  empty owner token, and "is this the player of that staged ship" is asked
  through `handoff::Incoming::is_owned_by`: a ship transfer, a non-empty
  owner token, compared in constant time. An empty hello token used to equal
  the empty owner token of every body in flight: such a hello was welcomed by
  a shard that does not own the spawn point, and could claim the body.

## Player port (`space-orchestrator --player-bind`)

The one address a player dials. Off unless given. It is **public by design**:
no token on any bind, because all it tells a caller is what a shard would
tell it anyway.

* **What it does.** It reads one `Hello`, checks the protocol version, and
  answers with a `JoinRedirect` — the id of the shard that owns the spawn
  point and the shard map — or with `Reject(Starting)` while no connected
  shard owns it, then closes. It reads nothing behind the hello, keeps no
  state per player, issues no session and no token, and has no way into the
  orchestrator: the run loop publishes the answer (a `watch` value) and the
  listener only reads it (`player.rs`). A hello that claims to be a shard
  (`shard:<id>` in the token field) is refused with a sentence saying which
  port this is. The control port, in turn, refuses a hello that is not a
  shard's with a readable reason and never hands it to the run loop.
* **What it exposes.** The shard map **as a player needs it**
  (`ShardMap::for_players`): every region, its owner's id, and for each shard
  the address players dial (the *client endpoint*). The address shards dial
  each other on (the internal endpoint) is not in it — in a map sent to a
  player that field repeats the client endpoint. The same holds for the map
  in a shard's `Welcome`, in its `JoinRedirect` and in a handoff's
  `Redirect`; shards and the orchestrator exchange the real map among
  themselves only. So anybody who can reach the player port learns how many
  shards there are, which ground each owns and where players reach them — all
  of which a player has to know. Neither secret is in it, nor the cluster's
  name, the scene, loads or the event log. (With the `process` launcher the
  two addresses differ only in the IP today — `127.0.0.1:<port>` against
  `0.0.0.0:<port>` — so the port a shard listens on is known either way.)
* **Limits.** A first packet from an address that has not been validated is
  answered with a QUIC **retry** (a stateless token; no handshake, no slot):
  a forged source address gets one small packet and nothing else, and a real
  client comes back with the token within the same connect. Then: at most 256
  hellos in flight (more are closed at once and counted as `refused_busy` in
  `/api/state`), 5 s for a hello to arrive, at most a second to write the
  answer and a second for it to be read, the whole connection bounded once
  more from outside (`player.rs`, `ANSWER_SLACK`), log lines about bad hellos
  a few per ten seconds.
* **Known limit: no cap per source address.** The 256 slots are shared by
  everybody. One host that keeps opening connections and saying nothing can
  hold all of them — each for the 5 s a hello may take — and while it does,
  other players' joins are refused (`refused_busy`). That delays joins and
  nothing else: shards, the map and players already on a shard do not pass
  through the port. A per-source limit (or a far shorter hello timeout once
  the retry has shown the peer is real and fast) is not built.
* **Known limit: the spawn point's shard is a choke point.** Every new
  player is sent to the one shard that owns the spawn point, a shard takes
  100 players (`MAX_PLAYERS_PER_SHARD`), and a player who never moves is
  never handed to another shard and is **never evicted for being idle** — a
  client that keeps its connection alive keeps its slot. A hundred idle
  clients (or one host that opens a hundred links) fill it; from then on a
  join is answered `Full` there, while the player port keeps sending new
  players to it. Idle eviction, a queue, or spreading newcomers over the
  shards around the spawn point are not built.
* **A shard is believed about its client endpoint** as it is about its
  endpoint (deferred item 4): a registered shard — any local process on a
  secretless loopback control port — can advertise any address, and players
  are then sent there. An unspecified IP (`0.0.0.0:<port>`) is filled in by
  the *client* with the host it dialed, so it can only ever point back at the
  entry's own host.
* **Shards answer the same question.** A shard that does not own the spawn
  point answers a fresh join with the same `JoinRedirect`; a client follows
  at most two in a row and then gives up with a readable reason, so two
  shards that disagree about the map cannot bounce it for ever.
* **Reachability is not solved here.** Shards started by the `process`
  launcher bind loopback and advertise `0.0.0.0:<port>` to players; a player
  on another machine is told to dial `<entry host>:<port>` and reaches
  nothing unless those ports are forwarded. Publishing shards is a later
  phase (`docs/plan-sharding.md`).

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
* **`SceneAck` names its own sender**, like a load report and an epoch ack, and
  is taken only from the control connection that was admitted as that shard:
  what the page shows as "shard N confirmed the scene" is shard N's own word.
  That word is not checked — a registered shard (on a loopback control port
  without a secret: any local process that registers) can claim any scene
  state **for itself**, which changes what the page shows for it and, for a
  revision a little ahead, makes the orchestrator re-issue the scene one
  above it; it cannot choose the scene. See "Shard registration and peers"
  above.
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
