# DATA — what is in this repo, and how it was produced

The live measurements are committed here as an off-site backup: the raw oracle
completions exist nowhere else (they were produced on one consumer GPU, one
tier resident at a time). Everything in this file is regenerable from the
scripts, but regenerating it means another multi-hour GPU run, so the data is
kept in the repository.

## Machine and operating point

* single NVIDIA T500, 4 GB VRAM class; one tier resident at a time
  (vector/ultra are sequential-only under 4 GB — see SPEC §12 / RESULTS "Failure log");
* uniform evaluation operating point for every tier and every cycle:
  `-c 2048 -b 256 -ub 256 -t 4 -tb 4 -np 1`, temperature 0, `n_predict` 128,
  `cache_prompt` on, ports 8080/8081/8082;
* the tiers are served through `/completion`; they are completion-style models,
  so raw outputs contain the model's own continuation of the conversation
  (`Human: ...`). That noise is trimmed at scoring time, not at collection time
  (the raw text is preserved here).

## Live oracle data (raw completions + measured latency)

| file | cycle | contents |
|---|---|---|
| `harness/oracle.jsonl` | 1 | merged cycle-1 oracle, 180 prompts x 3 tiers (540 runs) |
| `harness/oracle_turbo.jsonl` | 1 | cycle-1 turbo pass (solo-tier file) |
| `harness/oracle_vector.jsonl` | 1 | cycle-1 vector pass |
| `harness/oracle_ultra.jsonl` | 1 | cycle-1 ultra pass |
| `harness/oracle_v2.jsonl` | 2 | merged cycle-2 oracle, 180 distinct prompts x 3 tiers (540 runs, zero errors) |
| `harness/oracle_v2_turbo.jsonl` | 2 | cycle-2 turbo pass |
| `harness/oracle_v2_vector.jsonl` | 2 | cycle-2 vector pass |
| `harness/oracle_v2_ultra.jsonl` | 2 | cycle-2 ultra pass |

Row format: `{id, task, prompt, reference, split, fn, tests, tiers}` where
`tests` holds the unit tests of a code item (args + expected value) and each
tier entry holds `{content, latency_ms, tokens_cached, timings, error}`.
`latency_ms` is wall-clock around the HTTP request, `timings` is the engine's own
report (`predicted_per_second` etc.).

Reproduction: `powershell -File harness/run_cycle2.ps1` (cycle 2),
`harness/run_oracle.py` + `harness/merge_tiers.py` (cycle 1).

## Scores (metric output)

| file | produced by |
|---|---|
| `scoring/scores.jsonl` | `scoring/score.py --in harness/oracle.jsonl --metric legacy` (cycle-1 metric) |
| `scoring/scores_v2.jsonl` | `scoring/score.py --in harness/oracle_v2.jsonl --metric v2` (cycle-2 data, metric v2) |
| `scoring/scores_v2_legacy.jsonl` | cycle-1 completions re-scored under metric v2 with the reconstructed gold (`eval_prompts/legacy_gold.json`) — the metric ablation |
| `scoring/scores_v2_judge.jsonl` | `scoring/score.py ... --judge-url ... --judge-mode all` (judge validation subset) |

Row format: `{id, task, split, prompt, reference, metric, fn, tests, tiers,
oracle, best_quality}`; each tier entry holds `{quality, latency_ms, error,
confidence}` plus the extracted artefacts (`answer` / `code`), the unit-test
detail and the judge verdict when available.

## Router outputs and figures

| file | contents |
|---|---|
| `router/learned_preds.jsonl` | learned router predictions per test id (`{id, pred, oracle}`) |
| `router/budget_curve.json` | cost-aware learned router's quality/cost frontier (`lambda` sweep) |
| `analysis/results.json` | machine-readable snapshot of every published number (see "Verification") |
| `analysis/table.md` | policy comparison, per-task quality, oracle distribution, frontier |
| `analysis/report.md` | the full report regenerated from live data (`analysis/report.py`) |
| `analysis/cost_quality.png` | the cost-vs-quality plot (core figure) |
| `harness/switch_cost.json` | measured tier eviction+load cost per tier (`measure_switch.ps1`) |

## Verification

The published numbers are not hand-copied: they are regenerated from the
committed data and checked in CI.

```
python analysis/verify_results.py            # recompute the analysis tables and
                                             # compare with analysis/results.json
python analysis/verify_results.py --rescore  # also re-run the metric on the
                                             # committed oracle data (~3 min) and
                                             # check it reproduces the committed scores
```

## Engine logs

`harness/tier_{turbo,vector,ultra}.log` are the raw server logs of the live
runs (request bodies, prefill/decode timings, cache hits). They are kept as
evidence for the tier characterization and for the failure log; the cycle-1
`harness/{turbo,vector,ultra}_pass.log` files are the older bring-up logs.

## Not committed

* mock/smoke artifacts (`harness/oracle_smoke*.jsonl`, `scoring/scores_smoke*.jsonl`):
  deterministic synthetic outputs for pipeline validation only — they are NOT
  results and are regenerable with `harness/run_oracle.py --mock`;
* `*.pid`, `*.log.err`: stale process ids and empty stderr logs;
* the engine, the model weights and the proprietary backend (`*.exe`, `*.gguf`,
  `core/`, `build-jve/`, ...) — closed for now, never pushed.
