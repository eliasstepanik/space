<#
.SYNOPSIS
    Local development environment for the space engine on Windows: builds the
    binaries, starts the server side in the background and runs the windowed
    client against it. Closing the client stops everything this script started.

.DESCRIPTION
    Topologies (-Topology):

      single        one standalone shard on 127.0.0.1:7777 (default)
      two-shards    two standalone shards that are neighbours along +X
                    (127.0.0.1:7777 owns cell 0, 127.0.0.1:7778 owns cell 1).
                    The ship spawns in the middle of the first shard's cell,
                    512 m from the border: fly +X across it and the title bar
                    changes to "shard 2 after 1 handoff(s)" - a seamless shard
                    transition.
      orchestrated  the orchestrator (127.0.0.1:7900) plus one shard that
                    registers with it and is given the root region
      offline       no server: just the client, flying a free camera

    Scenes (-Scene): demo, galaxy, materials (PBR chart), solar (real-size
    solar system). -Glb shows a .glb model in front of the spawn point.

    Server logs go to target\dev\*.log.

.EXAMPLE
    .\scripts\dev.ps1
    One shard, the windowed client connected to it, galaxy scene.

.EXAMPLE
    .\scripts\dev.ps1 -Topology two-shards
    Fly across the shard border. In the galaxy scene the view starts facing
    +Z, so +X is to the LEFT: hold A (about 10 s) to cross into shard 2.

.EXAMPLE
    .\scripts\dev.ps1 -Topology offline -Scene solar
    The real-size solar system; the mouse wheel changes the flying speed.

.EXAMPLE
    .\scripts\dev.ps1 -Scene demo --frames 600 --no-vsync
    Extra arguments go to space-client: here, exit after 600 frames, uncapped.

.EXAMPLE
    .\scripts\dev.ps1 -Topology offline -Scene materials -Glb .\ship.glb
    The PBR material chart with a GLB model in front of it
    (cargo run -p space-render --example make_test_glb -- ship.glb writes one).
#>
[CmdletBinding()]
param(
    [ValidateSet('single', 'two-shards', 'orchestrated', 'offline')]
    [string]$Topology = 'single',
    [ValidateSet('demo', 'galaxy', 'materials', 'solar')]
    [string]$Scene = 'galaxy',
    [string]$Size = '1280x720',
    [ValidateSet('high', 'medium', 'low', 'ci')]
    [string]$Quality = 'high',
    [string]$Glb = '',
    # Present as fast as frames render instead of waiting for vsync.
    [switch]$NoVsync,
    # Use the binaries already in target\release instead of rebuilding.
    [switch]$NoBuild,
    # Start the server side and return, leaving it running (stop with -Stop).
    [switch]$ServersOnly,
    # Stop servers a previous -ServersOnly run left behind, then exit.
    [switch]$Stop,
    # Anything else on the command line is passed to space-client verbatim,
    # e.g.  .\scripts\dev.ps1 -Scene solar --frames 600
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ClientArgs = @()
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$bin = Join-Path $root 'target\release'
$devDir = Join-Path $root 'target\dev'
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
    Write-Host 'building release binaries (space-client, space-server, space-orchestrator)...'
    Push-Location $root
    try {
        cargo build --release --locked -p space-client -p space-server -p space-orchestrator
        if ($LASTEXITCODE -ne 0) { throw "cargo build failed (exit code $LASTEXITCODE)" }
    } finally {
        Pop-Location
    }
}

# Every address is a literal IPv4 socket address: the binaries do not resolve
# host names.
$server = $null
switch ($Topology) {
    'single' {
        Start-DevServer 'shard-1' 'space-server.exe' @(
            '--shard', '1', '--region', '64:0:0:0', '--bind', '127.0.0.1:7777')
        $server = '127.0.0.1:7777'
    }
    'two-shards' {
        Start-DevServer 'shard-1' 'space-server.exe' @(
            '--shard', '1', '--region', '64:0:0:0', '--bind', '127.0.0.1:7777',
            '--peer', '2:127.0.0.1:7778:64:1:0:0')
        Start-DevServer 'shard-2' 'space-server.exe' @(
            '--shard', '2', '--region', '64:1:0:0', '--bind', '127.0.0.1:7778',
            '--peer', '1:127.0.0.1:7777:64:0:0:0')
        $server = '127.0.0.1:7777'
    }
    'orchestrated' {
        # The state file is relative to the working directory, target\dev.
        Start-DevServer 'orchestrator' 'space-orchestrator.exe' @(
            '--bind', '127.0.0.1:7900', '--state', 'orchestrator.id')
        # A fixed bind address, so a restarted shard is recognised as the same one.
        Start-DevServer 'shard-1' 'space-server.exe' @(
            '--orchestrator', '127.0.0.1:7900', '--bind', '127.0.0.1:7777')
        $server = '127.0.0.1:7777'
    }
    'offline' { }
}

if ($ServersOnly) {
    Write-Host "servers are running; logs in $devDir. Stop them with: .\scripts\dev.ps1 -Stop"
    return
}

$arguments = @('--window', "--$Scene", '--size', $Size, '--quality', $Quality)
if ($server) { $arguments += @('--server', $server) }
if ($Glb) { $arguments += @('--glb', (Resolve-Path $Glb).Path) }
if ($NoVsync) { $arguments += '--no-vsync' }
$arguments += $ClientArgs

Write-Host "space-client $($arguments -join ' ')"
$exitCode = 0
try {
    & (Join-Path $bin 'space-client.exe') @arguments
    $exitCode = $LASTEXITCODE
} finally {
    Stop-DevServers
}
exit $exitCode
