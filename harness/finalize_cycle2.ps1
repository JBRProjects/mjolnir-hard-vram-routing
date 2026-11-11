# finalize_cycle2.ps1 - merge, score, train and report the cycle-2 live run.
# Run after harness/run_cycle2.ps1 (or after stopping it early - the merge uses
# the prompt ids that every tier file has, so a partial cycle still yields a
# usable three-tier oracle over the completed prompts).
#
# Usage:
#   powershell -File harness/finalize_cycle2.ps1
#   powershell -File harness/finalize_cycle2.ps1 -Judge          # also run the judge validation
param(
    [switch]$Judge,
    [string]$JudgeUrl = "http://127.0.0.1:8082",
    [int]$JudgeMax = 150
)
$ErrorActionPreference = "Continue"
$ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ROOT

$oracle = Join-Path $ROOT "harness\oracle_v2.jsonl"
$scores = Join-Path $ROOT "scoring\scores_v2.jsonl"

Write-Output "[finalize] merging tier passes (intersection of prompt ids)"
python "harness\merge_tiers.py" --out $oracle `
    --tier turbo (Join-Path $ROOT "harness\oracle_v2_turbo.jsonl") `
    --tier vector (Join-Path $ROOT "harness\oracle_v2_vector.jsonl") `
    --tier ultra (Join-Path $ROOT "harness\oracle_v2_ultra.jsonl")

Write-Output "[finalize] scoring (metric v2)"
python "scoring\score.py" --in $oracle --out $scores --metric v2

if ($Judge) {
    Write-Output "[finalize] judge validation against $JudgeUrl"
    python "scoring\score.py" --in $oracle --out (Join-Path $ROOT "scoring\scores_v2_judge.jsonl") `
        --metric v2 --judge-url $JudgeUrl --judge-mode all --judge-max $JudgeMax
}

Write-Output "[finalize] learned router (+ feature ablation)"
python "router\learned.py" --scores $scores --ablation

Write-Output "[finalize] cost-aware router frontier"
python "router\budget_router.py" --scores $scores

Write-Output "[finalize] plot + tables"
python "analysis\make_plot.py" --scores $scores
python "analysis\report.py" --in $oracle --scores $scores --out (Join-Path $ROOT "analysis\report.md")

Write-Output "[finalize] done"
