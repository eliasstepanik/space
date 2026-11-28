#!/usr/bin/env bash
# Local development environment for the space engine (Linux; also Git Bash on
# Windows): builds the binaries, starts the server side in the background and
# runs a client against it. When the client ends, everything this script
# started is stopped. scripts/dev.ps1 is the same thing for PowerShell.
#
# The client is always online; what it shows is the server's choice.
#
#   scripts/dev.sh [options] [-- extra space-client arguments]
#
#   --topology orchestrated|single|two-shards   (default: orchestrated)
#       orchestrated  the orchestrator: its player port on 127.0.0.1:7777 -
#                     the one address a client connects to - its control port
#                     on 127.0.0.1:7900 (shards only) and its web UI on
#                     http://127.0.0.1:7980/. It starts the shards itself
#                     (--min-shards, default 4; below) and sends every client
#                     to the one that owns the spawn point. Change the scene
#                     for every shard and client in the web UI. Needs curl.
#       single        one standalone shard on 127.0.0.1:7777
#       two-shards    two standalone shards that are neighbours along +X
#                     (7777 owns cell 0, 7778 owns cell 1): fly +X (hold D for
#                     about 15 s, until the HUD's pos passes 512) and the HUD
#                     changes to "shard 2  handoffs 1".
#   --scene demo-area|materials|galaxy    (default: demo-area; the solar
#       system is part of demo-area)
#       What the SERVERS show. The orchestrator takes it as its initial scene;
#       a scene chosen later in the web UI is remembered in
#       target/dev/orchestrator.id.scene, and giving --scene forgets that.
#   --client window|headless|stream             (default: window)
#       headless: no window; unless --harness or --script says otherwise the
#       HTTP API is served on 127.0.0.1:8098. stream: headless, streaming to
#       browser viewers on --stream (default 127.0.0.1:8099).
#   --connect-screen     start the client on the connect screen instead of
#                        connecting to 127.0.0.1:7777 at once - the
#                        orchestrator's player port, or the first shard in
#                        single and two-shards (a client that connects at once
#                        also reconnects there by itself when the connection
#                        fails or is lost; from the connect screen nothing is
#                        retried)
#   --harness ADDR       serve the harness HTTP API there
#   --script FILE        run a Lua script against the client; its outcome is
#                        this script's exit code
#   --out-dir DIR        where scripts write screenshots
#                        (default target/dev/harness-out)
#   --stream ADDR        where the stream client serves viewers
#   --size WxH  --quality high|medium|low|ci  --glb FILE  --no-vsync
#   --min-shards N       orchestrated only (default 4, at most 8): the
#                        orchestrator starts and keeps N shards running by
#                        itself (space-orchestrator --launcher process
#                        --min-shards N), each a space-server process on a
#                        port from 7870 up, logging to
#                        target/dev/shard-<port>.log. With the default 4 the
#                        demo area lies in four shard areas that meet at pos
#                        x = -6 / z = -4. A shard that dies is replaced, and a
#                        client that was on it comes back by itself through
#                        the player port. --min-shards 1 is the whole world
#                        on one shard.
#   --port-offset N      add N to every port (a second stack beside the first)
#   --no-build           use the binaries that are already built
#
# The binaries are built with the `quick` profile (Cargo.toml): optimised like
# release, but without LTO and incrementally, so a rebuild after an edit takes
# seconds. They land in target/quick. What ships is built with --release.
#   --servers-only       start the server side and return (stop with --stop)
#   --stop               stop servers a previous --servers-only run left behind
#
# Examples:
#   scripts/dev.sh
#   scripts/dev.sh --topology two-shards
#   scripts/dev.sh --min-shards 1
#   scripts/dev.sh --client headless --script scripts/harness/smoke.lua
#   scripts/dev.sh --client headless        # then: curl http://127.0.0.1:8098/state
#   scripts/dev.sh -- --frames 600 --no-vsync
#
# Server logs go to target/dev/*.log (below CARGO_TARGET_DIR when that is set).

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Where cargo puts its output: target/, unless CARGO_TARGET_DIR says otherwise.
target="${CARGO_TARGET_DIR:-$root/target}"
bin="$target/quick"
dev_dir="$target/dev"
pid_file="$dev_dir/servers.pid"
# The shards an orchestrator started itself (--min-shards): it keeps the pids
# of those that are running here, in the directory it is told to log to (the
# file is rewritten whenever one is started, stopped or seen to have exited,
# and removed when none is left).
launched_pid_file="$dev_dir/launched-shards.pid"
exe=""
case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*) exe=".exe" ;;
esac

topology="orchestrated"
scene="demo-area"
scene_given=0
client="window"
stream=""
harness=""
script=""
out_dir=""
size=""
quality="high"
glb=""
port_offset=0
min_shards=4
min_shards_given=0
connect_screen=0
no_vsync=0
no_build=0
servers_only=0
stop=0
client_args=()

usage() {
    # The comment block at the top of this file, without the shebang.
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

die() {
    echo "dev.sh: $*" >&2
    exit 2
}

need_value() {
    [ "$#" -ge 2 ] || die "$1 needs a value"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --topology) need_value "$@"; topology="$2"; shift 2 ;;
        --scene) need_value "$@"; scene="$2"; scene_given=1; shift 2 ;;
        --client) need_value "$@"; client="$2"; shift 2 ;;
        --stream) need_value "$@"; stream="$2"; shift 2 ;;
        --harness) need_value "$@"; harness="$2"; shift 2 ;;
        --script) need_value "$@"; script="$2"; shift 2 ;;
        --out-dir) need_value "$@"; out_dir="$2"; shift 2 ;;
        --size) need_value "$@"; size="$2"; shift 2 ;;
        --quality) need_value "$@"; quality="$2"; shift 2 ;;
        --glb) need_value "$@"; glb="$2"; shift 2 ;;
        --port-offset) need_value "$@"; port_offset="$2"; shift 2 ;;
        --min-shards) need_value "$@"; min_shards="$2"; min_shards_given=1; shift 2 ;;
        --connect-screen) connect_screen=1; shift ;;
        --no-vsync) no_vsync=1; shift ;;
        --no-build) no_build=1; shift ;;
        --servers-only) servers_only=1; shift ;;
        --stop) stop=1; shift ;;
        -h | --help) usage; exit 0 ;;
        --) shift; client_args+=("$@"); break ;;
        # Anything else goes to space-client verbatim.
        *) client_args+=("$1"); shift ;;
    esac
done

case "$topology" in
    orchestrated | single | two-shards) ;;
    *) die "--topology must be orchestrated, single or two-shards (got '$topology')" ;;
esac
case "$scene" in
    demo-area | materials | galaxy) ;;
    *) die "--scene must be demo-area, materials or galaxy (got '$scene'; the solar system is part of demo-area)" ;;
esac
case "$client" in
    window | headless | stream) ;;
    *) die "--client must be window, headless or stream (got '$client')" ;;
esac
case "$quality" in
    high | medium | low | ci) ;;
    *) die "--quality must be high, medium, low or ci (got '$quality')" ;;
esac
case "$port_offset" in
    '' | *[!0-9]*) die "--port-offset must be a number (got '$port_offset')" ;;
esac
case "$min_shards" in
    # 8 is the orchestrator's own limit: the root's shard and one per layout
    # cell (it refuses to start with more).
    [1-8]) ;;
    *) die "--min-shards must be a number from 1 to 8 (got '$min_shards')" ;;
esac
if [ "$min_shards_given" = 1 ] && [ "$topology" != orchestrated ]; then
    die "--min-shards needs the orchestrated topology: it is the orchestrator that starts the shards"
fi

# The name of the program running as `pid`, or nothing.
process_name() {
    if [ -n "$exe" ]; then
        # Git Bash: ps knows Windows processes only by their Windows pid.
        ps -W 2> /dev/null | awk -v pid="$1" '$4 == pid { n = split($NF, parts, /[\\\/]/); print parts[n] }'
    else
        ps -p "$1" -o comm= 2> /dev/null || true
    fi
}

# The directory the executable running as `pid` lies in, or nothing when this
# shell cannot tell (no /proc, as on macOS and the BSDs; a process of another
# user).
process_dir() {
    local path=""
    if [ -n "$exe" ]; then
        # Git Bash: the last column of ps -W is the executable's full Windows
        # path, which may contain spaces.
        path="$(ps -W 2> /dev/null | awk -v pid="$1" '$4 == pid && match($0, /[A-Za-z]:\\.*$/) { print substr($0, RSTART) }')"
        [ -n "$path" ] || return 0
        path="$(cygpath -u "$path" 2> /dev/null || true)"
    elif [ -e "/proc/$1/exe" ]; then
        path="$(readlink "/proc/$1/exe" 2> /dev/null || true)"
        # A binary that was rebuilt while the process ran.
        path="${path% (deleted)}"
    fi
    [ -n "$path" ] || return 0
    (cd "$(dirname "$path")" 2> /dev/null && pwd -P) || true
}

# is_ours PID NAME: whether the process is one of THIS checkout's binaries.
#
# A pid in a pid file is only a number: the process that had it may be gone,
# and the system may have given the number to anything since - another
# checkout's space-server, for one. So the name is not enough: the executable
# must lie in this checkout's build directory (target/quick).
is_ours() {
    case "$2" in space-*) ;; *) return 1 ;; esac
    local dir ours
    dir="$(process_dir "$1")"
    if [ -z "$dir" ]; then
        # This shell cannot tell where the executable lies (see process_dir):
        # the name is all there is to go by.
        return 0
    fi
    ours="$(cd "$bin" 2> /dev/null && pwd -P || true)"
    if [ -n "$exe" ]; then
        # Windows paths do not distinguish case.
        dir="$(printf '%s' "$dir" | tr '[:upper:]' '[:lower:]')"
        ours="$(printf '%s' "$ours" | tr '[:upper:]' '[:lower:]')"
    fi
    [ -n "$ours" ] && [ "$dir" = "$ours" ]
}

# stop_processes FILE: stop every process listed in FILE (one pid a line) that
# is one of this checkout's binaries, then remove FILE.
stop_processes() {
    local file="$1"
    [ -f "$file" ] || return 0
    local id name
    while read -r id; do
        id="${id%$'\r'}"
        case "$id" in '' | *[!0-9]*) continue ;; esac
        name="$(process_name "$id")"
        [ -n "$name" ] || continue
        if is_ours "$id" "$name"; then
            if [ -n "$exe" ]; then
                taskkill //PID "$id" //F > /dev/null 2>&1 || true
            else
                kill "$id" 2> /dev/null || true
            fi
            echo "stopped $name (pid $id)"
        else
            echo "left pid $id ($name) alone: it is not a binary of this checkout"
        fi
    done < "$file"
    rm -f "$file"
}

stop_servers() {
    # What this script started first - the orchestrator among it, so that it
    # cannot start a replacement for a shard stopped next - then the shards
    # that orchestrator had started. (They would stop by themselves a few
    # seconds after losing it; this does not wait for that.)
    stop_processes "$pid_file"
    stop_processes "$launched_pid_file"
}

# start_server NAME EXE ARGS...: run a server in the background, logging to
# target/dev/NAME.log.
start_server() {
    local name="$1" program="$2"
    shift 2
    local log="$dev_dir/$name.log"
    (cd "$dev_dir" && exec "$bin/$program$exe" "$@" > "$name.out.log" 2> "$log") &
    local id=$!
    if [ -n "$exe" ]; then
        # The Windows pid, which is what ps -W and taskkill know.
        sleep 0.2
        id="$(cat "/proc/$id/winpid" 2> /dev/null || echo "$id")"
    fi
    echo "$id" >> "$pid_file"
    sleep 0.4
    if [ -z "$(process_name "$id")" ]; then
        tail -n 20 "$log" | sed 's/^/  /'
        stop_servers
        echo "dev.sh: $name exited at once; see $log" >&2
        exit 1
    fi
    echo "started $name (pid $id): $program $*"
}

if [ "$stop" = 1 ]; then
    stop_servers
    exit 0
fi

mkdir -p "$dev_dir"
stop_servers

if [ "$no_build" = 0 ]; then
    echo 'building (profile quick: space-client, space-server, space-orchestrator)...'
    (cd "$root" && cargo build --profile quick --locked -p space-client -p space-server -p space-orchestrator)
fi

# Every server address is a literal IPv4 socket address: the servers do not
# resolve host names.
#
# $entry is the address a client connects to: the orchestrator's player port
# in the orchestrated topology (it sends the client on to the shard that owns
# the spawn point), the first shard itself in the other two.
entry="127.0.0.1:$((7777 + port_offset))"
shard2="127.0.0.1:$((7778 + port_offset))"
orchestrator="127.0.0.1:$((7900 + port_offset))"
webui="127.0.0.1:$((7980 + port_offset))"
case "$topology" in
    orchestrated)
        # The state file is relative to the working directory, target/dev. A
        # scene picked in the web UI is kept next to it and wins over --scene;
        # an explicit --scene is the newer word, so it forgets that choice.
        if [ "$scene_given" = 1 ]; then
            rm -f "$dev_dir/orchestrator.id.scene"
        fi
        # The orchestrator starts the shards itself: one space-server process
        # each, on a port of this range, logging to target/dev. Players
        # connect to its player port and are sent on to a shard.
        first_port=$((7870 + port_offset))
        started=$(date +%s)
        start_server orchestrator space-orchestrator \
            --bind "$orchestrator" --player-bind "$entry" \
            --state orchestrator.id \
            --webui "$webui" --scene "$scene" \
            --launcher process --min-shards "$min_shards" \
            --host-port-range "$first_port-$((7899 + port_offset))" \
            --shard-log-dir "$dev_dir"
        # Wait until every shard has its region and the player port has a
        # shard to send a client to.
        ready=0
        tries=0
        while [ "$ready" = 0 ]; do
            sleep 0.25
            state="$(curl -s --max-time 5 "http://$webui/api/state" || true)"
            regions="$(printf '%s' "$state" | grep -o '"nesting":' | wc -l | tr -d ' ')"
            live="$(printf '%s' "$state" | grep -o '"role":"owner","connected":true' | wc -l | tr -d ' ')"
            if [ "$regions" -ge "$min_shards" ] && [ "$live" -ge "$min_shards" ] &&
                printf '%s' "$state" | grep -q '"epoch_in_flight":null' &&
                printf '%s' "$state" | grep -q '"answer":"redirect"'; then
                ready=1
            fi
            tries=$((tries + 1))
            if [ "$ready" = 0 ] && [ "$tries" -ge 240 ]; then
                tail -n 20 "$dev_dir/orchestrator.out.log" | sed 's/^/  /'
                stop_servers
                echo "dev.sh: the orchestrator did not bring up $min_shards shard(s) within 60 s; see $dev_dir/orchestrator.out.log and shard-*.log" >&2
                exit 1
            fi
        done
        sends_to="$(printf '%s' "$state" | sed -n 's/.*"sends_to_shard":\([0-9]*\).*/\1/p')"
        echo "$min_shards shard(s) started by the orchestrator own $regions region(s) after $(($(date +%s) - started)) s; players are sent to shard $sends_to"
        ;;
    single)
        start_server shard-1 space-server \
            --shard 1 --region 64:0:0:0 --bind "$entry" --scene "$scene"
        ;;
    two-shards)
        start_server shard-1 space-server \
            --shard 1 --region 64:0:0:0 --bind "$entry" \
            --peer "2:$shard2:64:1:0:0" --scene "$scene"
        start_server shard-2 space-server \
            --shard 2 --region 64:1:0:0 --bind "$shard2" \
            --peer "1:$entry:64:0:0:0" --scene "$scene"
        ;;
esac

echo
if [ "$topology" = orchestrated ]; then
    echo "  player port    $entry (QUIC; the address to type on the connect screen)"
    echo "                 ($min_shards shard(s) behind it on ports from $((7870 + port_offset)), logs in shard-<port>.log)"
    echo "  web UI         http://$webui/  (scene, shards, players)"
else
    echo "  shard          $entry (QUIC; the address to type on the connect screen)"
fi
if [ "$topology" = two-shards ]; then
    echo "  second shard   $shard2"
fi

if [ "$servers_only" = 1 ]; then
    echo
    echo "servers are running; logs in $dev_dir. Stop them with: scripts/dev.sh --stop"
    exit 0
fi

arguments=(--quality "$quality")
if [ "$connect_screen" = 0 ]; then
    arguments+=(--server "$entry")
fi
if [ -n "$size" ]; then
    arguments+=(--size "$size")
fi
if [ -n "$glb" ]; then
    arguments+=(--glb "$(cd "$(dirname "$glb")" && pwd)/$(basename "$glb")")
fi
if [ "$no_vsync" = 1 ]; then
    arguments+=(--no-vsync)
fi
case "$client" in
    window) ;;
    headless)
        arguments+=(--headless)
        # A headless client nobody can reach is no use to anybody.
        if [ -z "$harness" ] && [ -z "$script" ]; then
            harness="127.0.0.1:$((8098 + port_offset))"
        fi
        ;;
    stream)
        if [ -z "$stream" ]; then
            stream="127.0.0.1:$((8099 + port_offset))"
        fi
        arguments+=(--stream "$stream")
        echo "  stream         ws://$stream  (viewer: crates/space-viewer, index.html?ws=ws://$stream)"
        ;;
esac
if [ -n "$harness" ]; then
    arguments+=(--harness "$harness")
    echo "  harness API    http://$harness/  (GET / lists the routes)"
fi
if [ -n "$script" ] || [ -n "$harness" ]; then
    if [ -z "$out_dir" ]; then
        out_dir="$dev_dir/harness-out"
    fi
    arguments+=(--out-dir "$out_dir")
    echo "  screenshots    $out_dir"
fi
if [ -n "$script" ]; then
    [ -f "$script" ] || { stop_servers; die "--script $script: no such file"; }
    arguments+=(--script "$(cd "$(dirname "$script")" && pwd)/$(basename "$script")")
fi
arguments+=("${client_args[@]+"${client_args[@]}"}")

echo
echo "space-client ${arguments[*]}"
exit_code=0
trap stop_servers EXIT
"$bin/space-client$exe" "${arguments[@]}" || exit_code=$?
exit "$exit_code"
