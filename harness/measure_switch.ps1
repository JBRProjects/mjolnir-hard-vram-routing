# measure_switch.ps1 - measure the cost of switching the resident tier.
#
# The evaluation runs one tier at a time (vector/ultra are sequential-only under
# ~4GB VRAM), so a cascade that escalates has to EVICT the current tier and
# load the next one. That eviction+load cost is paid per escalation and is not
# in the latency of a generation, so the off-policy cascade numbers in
# RESULTS.md are optimistic unless it is measured.
#
# This script times, per tier: process launch -> /health 200, at the evaluation
# operating point (-c 2048), and writes the numbers to harness/switch_cost.json
# so the analysis can add the switch cost to the cascade.
#
# Usage:
#   powershell -File harness/measure_switch.ps1
#   powershell -File harness/measure_switch.ps1 -Tiers turbo vector
param(
    [string[]]$Tiers = @("turbo", "vector", "ultra"),
    [int]$EvalCtx = 2048,
    [int]$TimeoutSec = 600
)
$ErrorActionPreference = "Continue"
$ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$OUT = Join-Path $PSScriptRoot "switch_cost.json"
Set-Location $ROOT

$results = @{}
foreach ($tier in $Tiers) {
    $port = @{ turbo = 8080; vector = 8081; ultra = 8082 }[$tier]
    # make sure nothing is resident
    & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "bringup.ps1") -Tier $tier -Kill | Out-Null
    Start-Sleep -Seconds 3

    $t0 = Get-Date
    $proc = Start-Process -FilePath powershell -ArgumentList @(
        "-ExecutionPolicy", "Bypass", "-File", (Join-Path $PSScriptRoot "bringup.ps1"),
        "-Tier", $tier, "-EvalCtx", "$EvalCtx", "-TimeoutSec", "$TimeoutSec"
    ) -PassThru -WindowStyle Hidden
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $ok = $false
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -TimeoutSec 2 -UseBasicParsing
            if ($r.StatusCode -eq 200) { $ok = $true; break }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    $loadSec = ((Get-Date) - $t0).TotalSeconds
    if (-not $ok) { Write-Output "[switch] $tier did not become healthy in time"; continue }
    # let the engine settle (first request also pays a warm-up cost)
    try {
        $body = '{"prompt":"hi","n_predict":1,"temperature":0.0,"stream":false}'
        Invoke-WebRequest -Uri "http://127.0.0.1:$port/completion" -Method POST -Body $body `
            -ContentType "application/json" -TimeoutSec 300 -UseBasicParsing | Out-Null
    } catch { }
    & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "bringup.ps1") -Tier $tier -Kill | Out-Null
    Start-Sleep -Seconds 3
    $results[$tier] = [math]::Round($loadSec, 2)
    Write-Output ("[switch] {0}: load+health {1:N1} s" -f $tier, $loadSec)
}

if ($results.Count -gt 0) {
    $obj = [ordered]@{ eval_ctx = $EvalCtx; measured_at = (Get-Date -Format o); load_seconds = $results }
    $obj | ConvertTo-Json -Depth 4 | Set-Content -Path $OUT -Encoding utf8
    Write-Output "[switch] wrote $OUT"
}
