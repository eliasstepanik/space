<#
.SYNOPSIS
    Local development environment for the space engine on Windows: builds the
    binaries, starts the server side in the background and runs a client
    against it. When the client ends, everything this script started is
    stopped. (scripts/dev.sh is the same thing for Linux.)

.DESCRIPTION
    The client is always online; what it shows is the server's choice.

    Topologies (-Topology):

      orchestrated  (default) the orchestrator (127.0.0.1:7900, web UI on
                    http://127.0.0.1:7980/) plus one shard (127.0.0.1:7777)
                    that registers with it and is given the root region.
                    Change the scene for every shard and client in the web UI.
      single        one standalone shard on 127.0.0.1:7777
      two-shards    two standalone shards that are neighbours along +X
                    (127.0.0.1:7777 owns cell 0, 127.0.0.1:7778 owns cell 1).
                    The ship spawns in the middle of the first shard's cell,
                    512 m from the border: fly +X across it (hold D for about
                    15 s, until the HUD's pos passes 512) and the HUD changes
                    to "shard 2  handoffs 1" - a seamless shard transition.

    Scene (-Scene demo-area|materials|galaxy): what the SERVERS show. The
    solar system is part of demo-area (there is no separate solar scene).
    Standalone shards take it on their command line. The orchestrator takes it
    as its initial scene; a scene chosen later in the web UI is remembered in
    target\dev\orchestrator.id.scene, and giving -Scene forgets that choice.

    Client (-Client):

      window        (default) the windowed client
      headless      no window; drive it through the harness. Unless -Harness
                    or -Script says otherwise the HTTP API is served on
                    127.0.0.1:8098 (curl http://127.0.0.1:8098/ lists the routes)
      stream        headless, streaming to browser viewers on -Stream
                    (default 127.0.0.1:8099); see crates\space-viewer\README.md

    The client connects to the first shard at once, and reconnects by itself
    when the connection fails or is lost (until the address is edited or Esc
    is pressed); -ConnectScreen starts it on the connect screen instead (the
    address is prefilled: press Enter), and then nothing is retried.

    Harness (any client): -Harness ADDR serves the control API, -Script FILE
    runs a Lua script and makes its outcome this script's exit code, -OutDir
    DIR is where scripts write screenshots (default target\dev\harness-out).

    -PortOffset N adds N to every port (a second stack beside the first).

    Server logs go to target\dev\*.log.

.EXAMPLE
    .\scripts\dev.ps1
    Orchestrator, one shard and the windowed client in the demo area. Open
    http://127.0.0.1:7980/ to switch the scene while playing.

.EXAMPLE
    .\scripts\dev.ps1 -Topology two-shards
    Fly across the shard border: hold D until the HUD says "shard 2".

.EXAMPLE
    .\scripts\dev.ps1 -Topology single -Scene galaxy -ConnectScreen
    One shard showing the galaxy; the client starts on the connect
    screen with the shard's address in the field.

.EXAMPLE
    .\scripts\dev.ps1 -Client headless -Script scripts\harness\smoke.lua
    A scripted smoke test: connect, fly, turn, take screenshots; the exit code
    is the script's result.

.EXAMPLE
    .\scripts\dev.ps1 -Client headless
    A headless client to drive by hand:
      curl http://127.0.0.1:8098/state
      curl -o frame.png http://127.0.0.1:8098/frame.png

.EXAMPLE
    .\scripts\dev.ps1 --frames 600 --no-vsync
    Extra arguments go to space-client: here, exit after 600 frames, uncapped.
#>
[CmdletBinding()]
param(
    [ValidateSet('orchestrated', 'single', 'two-shards')]
    [string]$Topology = 'orchestrated',
    [ValidateSet('demo-area', 'materials', 'galaxy')]
    [string]$Scene = 'demo-area',
    [ValidateSet('window', 'headless', 'stream')]
    [string]$Client = 'window',
    # Where the stream client serves viewers (ws://).
    [string]$Stream = '',
    # Serve the harness HTTP API on this address.
    [string]$Harness = '',
    # Run this Lua script against the client; its outcome is the exit code.
    [string]$Script = '',
    # Where scripts write screenshots.
    [string]$OutDir = '',
    # Frame/window size; default: the client's own (1280x720 in a window).
    [string]$Size = '',
    [ValidateSet('high', 'medium', 'low', 'ci')]
    [string]$Quality = 'high',
    [string]$Glb = '',
    # Add this to every port.
    [int]$PortOffset = 0,
    # Start the client on the connect screen instead of connecting at once.
    [switch]$ConnectScreen,
    # Present as fast as frames render instead of waiting for vsync.
    [switch]$NoVsync,
    # Use the binaries that are already built (target\quick) instead of rebuilding.
    [switch]$NoBuild,
    # Start the server side and return, leaving it running (stop with -Stop).
    [switch]$ServersOnly,
    # Stop servers a previous -ServersOnly run left behind, then exit.
    [switch]$Stop,
    # Anything else on the command line is passed to space-client verbatim,
    # e.g.  .\scripts\dev.ps1 --frames 600
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ClientArgs = @()
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
# Where cargo puts its output: target\, unless CARGO_TARGET_DIR says otherwise.
$target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $root 'target' }
# The `quick` profile (Cargo.toml): optimised like release, but without LTO and
# incremental, so a rebuild after an edit takes seconds. What ships is built
# with --release.
$bin = Join-Path $target 'quick'
$devDir = Join-Path $target 'dev'
$pidFile = Join-Path $devDir 'servers.pid'

function Stop-DevServers {
    if (-not (Test-Path $pidFile)) { return }
    foreach ($line in Get-Content $pidFile) {
        $id = 0
        if (-not [int]::TryParse($line, [ref]$id)) { continue }
        $process = Get-Process -Id $id -ErrorAction SilentlyContinue
        # Only ever stop our own binaries: a pid may have been reused.
        if ($process -and $process.ProcessName -like 'space-*') {
            Stop-Process -Id $id -Force -Confirm:$false
            Write-Host "stopped $($process.ProcessName) (pid $id)"
        }
    }
    Remove-Item $pidFile -Confirm:$false
}

function Start-DevServer {
    param([string]$Name, [string]$Exe, [string[]]$Arguments)
    $log = Join-Path $devDir "$Name.log"
    # Start-Process joins the list with spaces and does not quote: do it here,
    # or an argument containing a space (a path under "C:\Users\First Last")
    # arrives as two.
    $quoted = $Arguments | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    $process = Start-Process -FilePath (Join-Path $bin $Exe) -ArgumentList $quoted `
        -WorkingDirectory $devDir -WindowStyle Hidden -PassThru `
        -RedirectStandardError $log -RedirectStandardOutput (Join-Path $devDir "$Name.out.log")
    Add-Content -Path $pidFile -Value $process.Id
    Start-Sleep -Milliseconds 400
    if ($process.HasExited) {
        Get-Content $log -Tail 20 | ForEach-Object { Write-Host "  $_" }
        throw "$Name exited at once (exit code $($process.ExitCode)); see $log"
    }
    Write-Host "started $Name (pid $($process.Id)): $Exe $($Arguments -join ' ')"
}

if ($Stop) {
    Stop-DevServers
    return
}

New-Item -ItemType Directory -Force $devDir | Out-Null
Stop-DevServers

if (-not $NoBuild) {
    Write-Host 'building (profile quick: space-client, space-server, space-orchestrator)...'
    Push-Location $root
    try {
        cargo build --profile quick --locked -p space-client -p space-server -p space-orchestrator
        if ($LASTEXITCODE -ne 0) { throw "cargo build failed (exit code $LASTEXITCODE)" }
    } finally {
        Pop-Location
    }
}

# Every server address is a literal IPv4 socket address: the servers do not
# resolve host names.
$shard1 = "127.0.0.1:$(7777 + $PortOffset)"
$shard2 = "127.0.0.1:$(7778 + $PortOffset)"
$orchestrator = "127.0.0.1:$(7900 + $PortOffset)"
$webui = "127.0.0.1:$(7980 + $PortOffset)"
switch ($Topology) {
    'orchestrated' {
        # The state file is relative to the working directory, target\dev. A
        # scene picked in the web UI is kept next to it and wins over --scene;
        # an explicit -Scene is the newer word, so it forgets that choice.
        if ($PSBoundParameters.ContainsKey('Scene')) {
            Remove-Item (Join-Path $devDir 'orchestrator.id.scene') -ErrorAction SilentlyContinue -Confirm:$false
        }
        Start-DevServer 'orchestrator' 'space-orchestrator.exe' @(
            '--bind', $orchestrator, '--state', 'orchestrator.id',
            '--webui', $webui, '--scene', $Scene)
        # A fixed bind address, so a restarted shard is recognised as the same one.
        Start-DevServer 'shard-1' 'space-server.exe' @(
            '--orchestrator', $orchestrator, '--bind', $shard1)
    }
    'single' {
        Start-DevServer 'shard-1' 'space-server.exe' @(
            '--shard', '1', '--region', '64:0:0:0', '--bind', $shard1, '--scene', $Scene)
    }
    'two-shards' {
        Start-DevServer 'shard-1' 'space-server.exe' @(
            '--shard', '1', '--region', '64:0:0:0', '--bind', $shard1,
            '--peer', "2:${shard2}:64:1:0:0", '--scene', $Scene)
        Start-DevServer 'shard-2' 'space-server.exe' @(
            '--shard', '2', '--region', '64:1:0:0', '--bind', $shard2,
            '--peer', "1:${shard1}:64:0:0:0", '--scene', $Scene)
    }
}

Write-Host ''
Write-Host "  shard          $shard1 (QUIC; the address to type on the connect screen)"
if ($Topology -eq 'two-shards') { Write-Host "  second shard   $shard2" }
if ($Topology -eq 'orchestrated') {
    Write-Host "  web UI         http://$webui/  (scene, shards, players)"
}

if ($ServersOnly) {
    Write-Host ''
    Write-Host "servers are running; logs in $devDir. Stop them with: .\scripts\dev.ps1 -Stop"
    return
}

$arguments = @('--quality', $Quality)
if (-not $ConnectScreen) { $arguments += @('--server', $shard1) }
if ($Size) { $arguments += @('--size', $Size) }
if ($Glb) { $arguments += @('--glb', (Resolve-Path $Glb).Path) }
if ($NoVsync) { $arguments += '--no-vsync' }
switch ($Client) {
    'window' { }
    'headless' {
        $arguments += '--headless'
        # A headless client nobody can reach is no use to anybody.
        if (-not $Harness -and -not $Script) { $Harness = "127.0.0.1:$(8098 + $PortOffset)" }
    }
    'stream' {
        if (-not $Stream) { $Stream = "127.0.0.1:$(8099 + $PortOffset)" }
        $arguments += @('--stream', $Stream)
        Write-Host "  stream         ws://$Stream  (viewer: crates\space-viewer, index.html?ws=ws://$Stream)"
    }
}
if ($Harness) {
    $arguments += @('--harness', $Harness)
    Write-Host "  harness API    http://$Harness/  (GET / lists the routes)"
}
if ($Script -or $Harness) {
    if (-not $OutDir) { $OutDir = Join-Path $devDir 'harness-out' }
    $arguments += @('--out-dir', $OutDir)
    Write-Host "  screenshots    $OutDir"
}
if ($Script) { $arguments += @('--script', (Resolve-Path $Script).Path) }
$arguments += $ClientArgs

Write-Host ''
Write-Host "space-client $($arguments -join ' ')"
$exitCode = 0
try {
    # The client logs to stderr. With 'Stop', Windows PowerShell turns the
    # first such line into a terminating error as soon as somebody redirects
    # this script's output (dev.ps1 ... 2>&1 | Tee-Object log).
    $ErrorActionPreference = 'Continue'
    & (Join-Path $bin 'space-client.exe') @arguments
    $exitCode = $LASTEXITCODE
} finally {
    Stop-DevServers
}
exit $exitCode
