# bringup.ps1 - start/stop one MJOLNIR tier server (jayvis_server.exe) and wait for health.
# Runs each tier SOLO: vector/ultra are sequential-only under ~4GB VRAM (RESULTS failure log),
# so never call start for two tiers without stopping the previous one first.
# Usage:
#   powershell -File harness\bringup.ps1 -Tier turbo [-TimeoutSec 240]   # start turbo on 8080, wait health
#   powershell -File harness\bringup.ps1 -Tier turbo -Kill               # stop tier on that port
# Exit codes: 0 healthy (or stopped), 1 unhealthy/not found.
param(
    [Parameter(Mandatory = $true)][ValidateSet("turbo", "vector", "ultra")][string]$Tier,
    [switch]$Kill,
    [int]$TimeoutSec = 300,
    [int]$EvalCtx = 0  # override tier ctx for eval-operating-point runs (uniform hv across tiers); 0 = tier native ctx
)
$ErrorActionPreference = "Continue"
# analysis root (this file lives in <repo>\mjolnir\harness\)
$ANALYSIS = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$EXE = Join-Path $ANALYSIS "build-jve\Release\jayvis_server.exe"
$CORE = Join-Path $ANALYSIS "core"
$TIERS = @{
    turbo  = @{ model = "jayvis-turbo.gguf";                    ctx = "32768";    port = 8080 }
    vector = @{ model = "jayvis-vector-00001-of-00006.gguf";    ctx = "131072";   port = 8081 }
    ultra  = @{ model = "jayvis-ultra-00001-of-00007.gguf";     ctx = "1048576";  port = 8082 }
}
if (-not (Test-Path $EXE)) { Write-Output "[bringup] no engine at $EXE"; exit 1 }
$cfg = $TIERS[$Tier]
$port = $cfg.port
$ownerPid = $null
try {
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($conn) { $ownerPid = $conn.OwningProcess }
} catch { }
if ($Kill) {
    if ($ownerPid) {
        Write-Output "[bringup] stopping $Tier pid=$ownerPid on port $port"
        Stop-Process -Id $ownerPid -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 3
    } else {
        Write-Output "[bringup] $Tier not running (nothing on port $port)"
    }
    exit 0
}
if ($ownerPid) {
    Write-Output "[bringup] port $port already in use (pid=$ownerPid); refusing start"
    exit 1
}
$freeMb = 0
try {
    $freeMb = [int]@((nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -split "`n")[0]
} catch { }
if (-not (Test-Path (Join-Path $CORE $cfg.model))) { Write-Output "[bringup] missing model $($cfg.model) in $CORE"; exit 1 }
$log = Join-Path $PSScriptRoot "tier_$Tier.log"
if ($EvalCtx -gt 0) { $ctxUsed = "$EvalCtx" } else { $ctxUsed = $cfg.ctx }
# Start-Process -ArgumentList re-joins items with spaces WITHOUT quoting, so any
# item containing spaces (model path) gets split. Sidestep: run from core\ and
# pass bare relative filenames (no spaces in our tree).
$argsList = @(
    "-m", $cfg.model,
    "-ngl", "99", "-c", $ctxUsed, "-t", "4", "-tb", "4",
    "--port", $port, "-b", "256", "-ub", "256", "-np", "1",
    "--numa", "isolate", "--prio", "2", "--poll", "50",
    "--defrag-thold", "0.1", "--cache-reuse", "256",
    "-fit", "off", "--flash-attn", "on"
)
Write-Output ("[bringup] start {0} port {1} ctx {2} free_vram={3}MB log={4}" -f $Tier, $port, $ctxUsed, $freeMb, $log)
$proc = Start-Process -FilePath $exe -ArgumentList $argsList -WorkingDirectory $CORE `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru -WindowStyle Hidden
$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $deadline) {
    if ($proc.HasExited) { Write-Output "[bringup] $Tier exited early (code $($proc.ExitCode))"; Get-Content "$log.err" -Tail 20 -ErrorAction SilentlyContinue; exit 1 }
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -eq 200) {
            Start-Sleep -Seconds 2
            $vram = "n/a"
            try { $vram = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) } catch { }
            Write-Output ("[bringup] {0} HEALTHY port {1} vram_used={2}MB pid={3}" -f $Tier, $port, $vram, $proc.Id)
            exit 0
        }
    } catch { }
    Start-Sleep -Seconds 2
}
Write-Output "[bringup] $Tier unhealthy after $TimeoutSec s"
Get-Content $log -Tail 20 -ErrorAction SilentlyContinue
exit 1
