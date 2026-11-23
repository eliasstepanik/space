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
#       orchestrated  the orchestrator (127.0.0.1:7900, web UI on
#                     http://127.0.0.1:7980/) plus one shard (127.0.0.1:7777)
#                     that registers with it. Change the scene for every shard
#                     and client in the web UI.
#       single        one standalone shard on 127.0.0.1:7777
#       two-shards    two standalone shards that are neighbours along +X
#                     (7777 owns cell 0, 7778 owns cell 1): fly +X (hold D for
#                     about 15 s, until the HUD's pos passes 512) and the HUD
#                     changes to "shard 2  handoffs 1".
#   --scene demo-area|materials|solar|galaxy    (default: demo-area)
#       What the SERVERS show. The orchestrator takes it as its initial scene;
#       a scene chosen later in the web UI is remembered in
#       target/dev/orchestrator.id.scene, and giving --scene forgets that.
#   --client window|headless|stream             (default: window)
#       headless: no window; unless --harness or --script says otherwise the
#       HTTP API is served on 127.0.0.1:8098. stream: headless, streaming to
#       browser viewers on --stream (default 127.0.0.1:8099).
#   --connect-screen     start the client on the connect screen instead of
#                        connecting to the first shard at once (a client that
#                        connects at once also reconnects by itself when the
#                        connection fails or is lost; from the connect screen
#                        nothing is retried)
#   --harness ADDR       serve the harness HTTP API there
#   --script FILE        run a Lua script against the client; its outcome is
#                        this script's exit code
#   --out-dir DIR        where scripts write screenshots
#                        (default target/dev/harness-out)
#   --stream ADDR        where the stream client serves viewers
#   --size WxH  --quality high|medium|low|ci  --glb FILE  --no-vsync
#   --port-offset N      add N to every port (a second stack beside the first)
#   --no-build           use the release binaries that are already built
#   --servers-only       start the server side and return (stop with --stop)
#   --stop               stop servers a previous --servers-only run left behind
#
# Examples:
#   scripts/dev.sh
#   scripts/dev.sh --topology two-shards
#   scripts/dev.sh --client headless --script scripts/harness/smoke.lua
#   scripts/dev.sh --client headless        # then: curl http://127.0.0.1:8098/state
#   scripts/dev.sh -- --frames 600 --no-vsync
#
# Server logs go to target/dev/*.log (below CARGO_TARGET_DIR when that is set).

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Where cargo puts its output: target/, unless CARGO_TARGET_DIR says otherwise.
target="${CARGO_TARGET_DIR:-$root/target}"
bin="$target/release"
dev_dir="$target/dev"
pid_file="$dev_dir/servers.pid"
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
    demo-area | materials | solar | galaxy) ;;
    *) die "--scene must be demo-area, materials, solar or galaxy (got '$scene')" ;;
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

# The name of the program running as `pid`, or nothing.
process_name() {
    if [ -n "$exe" ]; then
        # Git Bash: ps knows Windows processes only by their Windows pid.
        ps -W 2> /dev/null | awk -v pid="$1" '$4 == pid { n = split($NF, parts, /[\\\/]/); print parts[n] }'
    else
        ps -p "$1" -o comm= 2> /dev/null || true
    fi
}

stop_servers() {
    [ -f "$pid_file" ] || return 0
    local id name
    while read -r id; do
        case "$id" in '' | *[!0-9]*) continue ;; esac
        name="$(process_name "$id")"
        # Only ever stop our own binaries: a pid may have been reused.
        case "$name" in
            space-*)
                if [ -n "$exe" ]; then
                    taskkill //PID "$id" //F > /dev/null 2>&1 || true
                else
                    kill "$id" 2> /dev/null || true
                fi
                echo "stopped $name (pid $id)"
                ;;
        esac
    done < "$pid_file"
    rm -f "$pid_file"
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
    echo 'building release binaries (space-client, space-server, space-orchestrator)...'
    (cd "$root" && cargo build --release --locked -p space-client -p space-server -p space-orchestrator)
fi

# Every server address is a literal IPv4 socket address: the servers do not
# resolve host names.
shard1="127.0.0.1:$((7777 + port_offset))"
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
        start_server orchestrator space-orchestrator \
            --bind "$orchestrator" --state orchestrator.id \
            --webui "$webui" --scene "$scene"
        # A fixed bind address, so a restarted shard is recognised as the same one.
        start_server shard-1 space-server --orchestrator "$orchestrator" --bind "$shard1"
        ;;
    single)
        start_server shard-1 space-server \
            --shard 1 --region 64:0:0:0 --bind "$shard1" --scene "$scene"
        ;;
    two-shards)
        start_server shard-1 space-server \
            --shard 1 --region 64:0:0:0 --bind "$shard1" \
            --peer "2:$shard2:64:1:0:0" --scene "$scene"
        start_server shard-2 space-server \
            --shard 2 --region 64:1:0:0 --bind "$shard2" \
            --peer "1:$shard1:64:0:0:0" --scene "$scene"
        ;;
esac

echo
echo "  shard          $shard1 (QUIC; the address to type on the connect screen)"
if [ "$topology" = two-shards ]; then
    echo "  second shard   $shard2"
fi
if [ "$topology" = orchestrated ]; then
    echo "  web UI         http://$webui/  (scene, shards, players)"
fi

if [ "$servers_only" = 1 ]; then
    echo
    echo "servers are running; logs in $dev_dir. Stop them with: scripts/dev.sh --stop"
    exit 0
fi

arguments=(--quality "$quality")
if [ "$connect_screen" = 0 ]; then
    arguments+=(--server "$shard1")
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
