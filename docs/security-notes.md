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

## Cluster certificate pinning

**Threat model.** A man-in-the-middle who can intercept traffic between a
shard and the orchestrator, or between two shards, could impersonate either
side, steal the cluster secret or inject commands. Without pinning, the
accept-anything TLS of `space_net::crypto` cannot tell them apart from the
real peer.

**Mechanism.** The cluster secret seeds a deterministic Ed25519 CA
(PBKDF2-HMAC-SHA256, 100,000 iterations, salt `space-cluster-ca-v1`; see
`space_net::trust`). The CA signs a per-endpoint leaf certificate, with a fresh
key each time; the CA key never leaves the process. The **pin** is the
hex-encoded CA public key (`ClusterPin`). A dialer verifies the server's
certificate against a trust store that holds only that CA, and rejects any
certificate the CA did not issue.

**Pin properties.** The pin is deterministic: the same secret always gives the
same pin, and rotating the secret changes the pin. It is a public key and safe
to publish.

**How to verify.** The orchestrator prints the pin at startup; an operator
compares it out-of-band with the pin they expect. Shards that hold the secret
verify the orchestrator's certificate automatically, before they send
`ShardHello`, and verify a peer shard's certificate before the hello that
carries the peer token. The orchestrator presents the cluster certificate on
its control and player ports whenever a secret is required. A shard with an
empty secret behaves as before (no pinning).

**Current scope.** Implemented: orchestrator certificate and shard-to-shard
verification by shards, and **client-side trust on first use**. A cluster
server presents the CA's self-signed certificate behind its leaf
(`ClusterTrust::ca_certificate`, an intermediate in the TLS chain; no wire
message changed). The client (`space_client::tofu`, which every client dial goes
through: the entry address and every shard a redirect or handoff names) reads
the pin off that certificate, checks that the leaf was issued by that CA, and
the first time it sees a `host:port` stores the pin and logs it at info level
(`trusting <host:port> pin <hex> for the first time; verify it out of band` -
compare it with the pin the orchestrator printed). Every later dial of that
`host:port` must present the same pin; a mismatch is refused with an error
naming the host:port, the expected and the received pin, and `--reset-pins`.
A server without a cluster secret presents no CA: the client connects unpinned
(as before) and stores nothing; a server that has a stored pin and later
presents none is refused.

The store is a JSON object `{"host:port": "<64 hex>"}` at
`<config dir>/space/pins.json` (`$XDG_CONFIG_HOME` or `$HOME/.config`; on
Windows `%APPDATA%`; the environment variable `SPACE_PIN_FILE` overrides the
path). An unreadable file is treated as empty, with a warning. A corrupt one
is first renamed to `pins.json.corrupt` (so the pins in it are not lost), then
treated as empty with a warning; pins are written through a temporary file and
a rename, so a crash does not leave half a file.
`space-client --reset-pins` deletes it and exits 0; `space-client --pin <hex>`
holds every server of that run to the given pin (TOFU skipped, store neither
read nor written, a mismatch fails hard). Limits, stated: TOFU protects against
a man in the middle only from the second connection on (verify the first pin
out of band); the key is the address as dialed, so the same server under a new
address is a first use; two client processes are not locked against each other
when they write the file.

**What an attacker can do.** One who holds the cluster secret can do nothing
new: they already control the cluster. One who does not hold it cannot
impersonate the orchestrator or a shard to a verifying shard.

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
   about its scene are taken at its word — a load report is what the
   autoscaler acts on: a shard that reports itself loaded has its hot cell
   carved for a new shard (a process or a container is started, up to
   `--max-shards`, once per cooldown), one that reports itself idle may be
   merged away, and a draining shard that says "0 players, 0 entities" is told
   to stop; a `SceneAck` naming a revision a
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
  shards around the spawn point are not built. **What the autoscaler does
  about it** (`docs/plan-sharding.md` §3): when that shard's load stays at
  0.8 of its capacity for 10 s, the spawn point's cell is carved for a shard
  of its own — once. That *moves* the crowd (it is all in that cell), it
  does not split it; from then on the hot cell is the shard's whole region, a
  region cannot be smaller than a cell (1024 m), and the autoscaler does
  nothing and says so in the event log once per cooldown. The limit of 100
  stands.
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
* **Reachability depends on the launcher.** Shards started by the `process`
  launcher bind loopback and advertise `0.0.0.0:<port>` to players; a player
  on another machine is told to dial `<entry host>:<port>` and reaches
  nothing unless those ports are forwarded. Shards started by the `docker`
  launcher (the compose files) publish their port on the host — on every
  interface by default (`--docker-publish-ip`): the shards' ports
  (7870-7899/udp) are as public as the player port and have to be open in
  the host's firewall. See "The Docker socket" below.

## The Docker socket (`space-orchestrator --launcher docker`)

With the `docker` launcher the orchestrator creates, starts, stops and removes
one container per shard through the Docker Engine's socket
(`launcher/docker.rs`), and both compose files mount `/var/run/docker.sock`
into the orchestrator's container for it.

**Whoever can talk to that socket can start a privileged container and mount
the host's disk. The orchestrator's container is root-equivalent on the
host** — whether the user inside it is root or not: it is not (uid 10001),
and compose's `group_add: DOCKER_GID` is exactly what gives it the socket.

What is done:

* **Nothing a player or a shard sends reaches a Docker call.** Every parameter
  of every call — image, command, ports, network, labels, names — comes from
  the orchestrator's configuration and its own launch counter, and from the
  daemon's own answers (container ids). A shard's hello, load report or ack
  never names an image, a command, a port or a label. What a peer *can* cause
  is that a call is made at all: a shard that reports load gets a new shard
  started (deferred item 4), up to `--max-shards`.
* **A shard container is given nothing**: no socket, no mount, every
  capability dropped (`CapDrop: ALL`), `no-new-privileges`, not privileged,
  no restart policy. A shard faces players; the orchestrator does not let it
  near the daemon. And it is held in: a **read-only root filesystem** (a
  shard writes no file; its log is its standard output, and nothing is
  mounted for it to write to), at most **1024 processes and threads**
  (`PidsLimit`), a **memory limit** (`--shard-memory`, default 2048 MiB,
  `0` = none; `MemorySwap` is set to the same, so no swap on top), and the
  **user `10001:10001`** set on the container by number — the uid
  `deploy/Dockerfile.shard` runs as, so that an image that was built without
  a `USER` line still does not run a shard as root.
* **Only this orchestrator's containers are removed.** The removal of
  leftovers at start, and the path that frees a container name that is taken
  (the daemon's 409), touch only containers that carry all three of its
  labels: `space.cluster=<cluster>`, `space.role=shard` and
  `space.instance=<instance>`. The instance is what tells two stacks on one
  host apart — the cluster's name does not: both may have been left at the
  default, and each would remove the other's running shards at every start.
  It is an id that is the same at every start of one orchestrator and
  different between two: made once and kept next to the state file
  (`<state>.instance`), or given (`--docker-instance`; the local compose file
  passes the compose project's name). **It is not a secret and is derived
  from none**: it is on every shard container for anyone who can list
  containers to read. A stranger's container of the same name — another
  stack's shard included — is left alone and the launch fails. (The labels
  `space.launch` and `space.port` are for the operator; the launcher never
  reads them.)
* **The cluster secret travels in the container's environment**
  (`SPACE_CLUSTER_SECRET`), never in its command, and is scrubbed from every
  error text the launcher returns — **before** the daemon's words are cut to
  length, so a secret that lies across the cut does not leave its front half
  behind. It is visible to anyone who can `docker inspect` on the host — that
  is, to the root-equivalent only.
* **Every call is bounded** (10 s; 15 s a remove; 300 s a pull; 20 s for
  removing everything at shutdown) and answers are read under size limits —
  **and the run loop waits for none of them**: a launch, a stop and a liveness
  question each run in a task of their own and report back on a later beat
  (`Orchestrator::drive_launcher`). Until the review of phase 6 only the
  launches did: stops and liveness questions were awaited inside the beat,
  one after the other, so a daemon that hung stopped the orchestrator's clock
  for 10 to 15 s per shard — no tick, no epoch deadline, no answer to a hello
  or an operator — and "a failed launch, not a hung orchestrator", which this
  note claimed, was true of launches only. Now a daemon that does not answer
  costs: the launch that fails after its bound; one liveness question per
  shard left open (it is not asked again while open); and a line in the event
  log at most every ten seconds, with the count of what was not written
  (`launcher_failures`). What still waits for the daemon, bounded, outside
  the loop: `prepare` before the loop starts, and the removal of the shards at
  shutdown.
* **The control port stays unpublished**, and **the web UI stays off in
  production** (loopback and a token locally). The web UI is the one way to
  operate the orchestrator from outside, and it can split and merge — that is,
  start and stop containers. Leave it off on a host with the socket mounted
  unless you need it.

What is not done:

* **No socket proxy.** The orchestrator holds the whole Engine API, of which
  it needs a few calls. An operator who wants less can put a filtering proxy
  between the orchestrator and the socket that allows only: containers
  create / start / inspect / delete / list, images inspect / create (the
  pull), and point `--docker-socket` at it. That is a recommendation; no
  such setup was tested here.
* **The shard image's name is trusted.** Whoever can change the orchestrator's
  environment (`SPACE_ORCHESTRATOR_SHARD_IMAGE`) or move the registry tag it
  names runs code as a container on the host. No digest is pinned and no
  signature checked. No registry credential is handled: a private image is
  pulled by the operator before the orchestrator starts. An image that
  disappears from the host while the orchestrator runs is pulled again, by
  that same name, by the launch that misses it.
* **Images are never removed by the orchestrator.** One `space-shard:sha-…`
  image per release stays on the host until the operator removes it
  (`deploy/proxy/README.md` has a command that removes only shard images no
  container uses). Removing them from the orchestrator was looked at and left
  out: "an image of the shard repository with another tag that no container
  uses" is also what a second stack's image looks like while that stack is
  down, and what a locally built image looks like, which no pull brings back.
* **No user namespace, no seccomp profile of its own, no CPU limit.** The
  daemon's defaults apply.
* **Shard ports are published on every interface** by default
  (`--docker-publish-ip 0.0.0.0`), with no per-source limit on any of them
  (as on the player port).
* **A killed orchestrator leaves its shards behind** for about 10 s (QUIC's
  idle timeout), then as exited containers until its next start removes them.
  Stopped with SIGTERM it removes them itself.

## Orchestrator web UI (`space-orchestrator --webui`)

An admin surface: it shows cluster state and can split, merge, pause the
autoscaler, re-send the map and switch the scene on every shard. Off by default.
A split launches a shard and a merge ends in one being stopped: with
`--launcher process` or `docker` whoever operates the page starts and stops
processes or containers (never with parameters of their choosing: "The Docker
socket" above).

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

## Far replicas between shards (`FarInterest`, `FarUpdate`)

Since phase 5 (`docs/plan-sharding.md` §6) a shard tells the other shards of its map where its
players are and is sent what they should see of those shards' entities beyond the border ghosts;
it relays that to its clients. What is new on the wire and what bounds it
(`space-server/src/far.rs`, "What a peer may do"; `space_protocol::far`):

* **No new listener, no new credential.** Both messages ride the peer connections the handoff
  uses: taken only after the peer hello was admitted (the cluster's peer token), only from a shard
  the receiver's map names — or named until a minute ago: a shard whose region was merged away
  is still a far neighbour while it hands over what it simulates (`Shard::knows_shard`, the same
  rule its handoff offers are heard by) — and only as that shard: `from` is checked against the
  connection.
  A client never sends or receives either message; what it gets of a far entity is an ordinary
  replicated entity with an `Origin` component.
* **Bounded on receipt.** A `FarUpdate` over 32 KiB of entities and names, more than 512 KiB of
  updates a second from one shard **or more than 180 updates** (`FAR_UPDATES`: twice what a shard
  sends at its busiest; the byte bound alone let a peer have tens of thousands of empty updates
  a second decoded), more than nine `FarInterest`s a second, an interest naming more
  than 256 players or asking for more than 256 entities again, an entity of more than 16 KiB of
  components, a 4097th far replica of one shard: each is dropped and counted
  (`ShardStats::far_refused`). A refresh's list of names stops at 4096. The log line about a drop
  goes through a `LogBudget` (three per ten seconds, the rest counted). A sender keeps its own
  messages under the bound — removals too, twelve bytes each, budgeted with the entities: 2 700
  at once used to make one message the receiver dropped whole
  (`removals_are_split_over_far_updates_like_everything_else`).
* **A far message cannot write what a shard simulates or holds a ghost of**: a guid that is in the
  receiver's world is skipped (`tests/far.rs`,
  `what_a_peer_sends_is_bounded_and_cannot_overwrite_local_entities`). A far replica is in no
  physics and no handoff, on the shard or on a client (a client's prediction does not copy one).
* **Only what draws an entity crosses** (`FAR_COMPONENTS`: motion, shape, model, springs, joints,
  part poses), cut down by the sender *and* by the receiver. A player's `Owner` (a session id),
  its `ShipControls` and anything private to a handoff — the owner's token lives only there —
  never leave the shard that simulates the ship. What does leave it is **where its players are**:
  every other shard of the cluster learns their positions three times a second. They are shards of
  one cluster on one host; a peer holding the token learns nothing a player near enough would not
  see.
* **A sample's tick is not trusted**: it is held to the receiver's own clock (at most 8 ticks
  behind it, never ahead), so a peer cannot place a sample somewhere else on a client's timeline.
* **What it does not cover.** One peer token for the whole cluster (above): a shard that holds it
  can claim a neighbour's id, and can then show this shard's clients a wrong picture of that
  neighbour's distant entities — a far replica is taken over from another shard by a newer sample.
  It cannot move anything, and the picture is corrected or dropped within three seconds of the
  real shard's next word. The bounds are per peer; eight peers at the bound are 4 MiB a second.

## The census knob (`SPACE_CENSUS_ALWAYS`)

A shard writes its census line every five seconds when its counts changed. With the environment
variable `SPACE_CENSUS_ALWAYS` set (to anything) it writes it every five seconds regardless — for
reading the traffic of a quiet stack off the log (`far N B/s out, N B/s in, N B/s to N clients`).
It changes nothing but that log line; nothing sets it.

## The shard's stall hook (`SPACE_DEBUG_STALL_DIR`)

`space-server` has one debugging hook in its tick loop (`space-server/src/clock.rs`,
`debug_stall`): when the environment variable `SPACE_DEBUG_STALL_DIR` names a directory, a file
`stall-<shard id>` in it holding a number of milliseconds (60 000 at most) makes that shard's loop
stand still that long, once; the file is removed first. It exists to look at what a suspended
shard does (`.warren/skills/space-visual-test/SKILL.md`).

* **Off unless the variable is set**, and nothing sets it: not the compose files, not the dev
  scripts. The orchestrator's `process` launcher passes its own environment on to the shards it
  starts (minus its secrets), so a variable set for the orchestrator reaches them — set it only
  for such a test.
* **It is not a network surface.** Nothing a peer or a player sends reaches it; it reads a file
  name it builds itself from the shard's own id, in a directory the operator named. Whoever can
  write into that directory can stall the shard — as they could stop it. It gives nothing else:
  the file's content is parsed as a number and used for nothing but a sleep.

## Known limitations (Phase 3 close-out)

* The H13 frame-gap test relies on lockstepped harness ticks; production continuity rests on the separately-tested commit-gate machinery.
* The Dockerfiles and compose files were correct-by-inspection only at that time (no Docker daemon in CI). What has been built and run since, and what has only been parsed: `deploy/README.md`, "What is and is not verified".
* The Phase-4 present path (swapchain) is validated by build + clippy only (no window system in CI).
