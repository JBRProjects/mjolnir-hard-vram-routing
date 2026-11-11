# run_cycle2.ps1 - full live oracle cycle for the v2 (metric-fixed) eval set.
# One tier resident at a time (vector/ultra are sequential-only under ~4GB VRAM).
#   1. build eval_prompts/prompts_v2.jsonl (180 distinct items, gold answers + unit tests)
#   2. per tier: bring up -> run the oracle pass -> kill the tier
#   3. merge the per-tier passes into harness/oracle_v2.jsonl
# Usage:
#   powershell -File harness/run_cycle2.ps1                 # full cycle (turbo, vector, ultra)
#   powershell -File harness/run_cycle2.ps1 -Tiers turbo    # subset
#   powershell -File harness/run_cycle2.ps1 -SkipBuild      # reuse existing prompts_v2.jsonl
param(
    [string[]]$Tiers = @("turbo", "vector", "ultra"),
    [switch]$SkipBuild,
    [int]$EvalCtx = 2048,
    [int]$NPredict = 128,
    [int]$Timeout = 600
)
$ErrorActionPreference = "Continue"
$ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PROMPTS = Join-Path $ROOT "eval_prompts\prompts_v2.jsonl"
Set-Location $ROOT

if (-not $SkipBuild) {
    Write-Output "[cycle2] building v2 prompt set"
    python "eval_prompts\build_prompts_v2.py" --out $PROMPTS
    if ($LASTEXITCODE -ne 0) { Write-Output "[cycle2] prompt build failed"; exit 1 }
}

foreach ($tier in $Tiers) {
    $out = Join-Path $ROOT "harness\oracle_v2_$tier.jsonl"
    Write-Output "[cycle2] === $tier === $(Get-Date -Format o)"
    & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "bringup.ps1") -Tier $tier -EvalCtx $EvalCtx
    if ($LASTEXITCODE -ne 0) { Write-Output "[cycle2] $tier bringup failed - skipping tier"; continue }
    python "harness\run_oracle.py" --prompts $PROMPTS --out $out --tiers $tier --n-predict $NPredict --timeout $Timeout
    Write-Output "[cycle2] $tier pass done -> $out"
    & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "bringup.ps1") -Tier $tier -Kill
    Start-Sleep -Seconds 5
}

Write-Output "[cycle2] merging tier passes"
python "harness\merge_tiers.py" --out (Join-Path $ROOT "harness\oracle_v2.jsonl") `
    --tier turbo (Join-Path $ROOT "harness\oracle_v2_turbo.jsonl") `
    --tier vector (Join-Path $ROOT "harness\oracle_v2_vector.jsonl") `
    --tier ultra (Join-Path $ROOT "harness\oracle_v2_ultra.jsonl")
Write-Output "[cycle2] done $(Get-Date -Format o)"
