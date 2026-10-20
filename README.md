# MJØLNIR: Memory-constrained Joint Orchestration of Layered Neural Inference Runtimes under Hard VRAM Budgets

> *Whosoever routes this query, if he be worthy, shall possess the power of Thor - measured, not simulated.*
> Cost-aware per-query routing across heterogeneous local model tiers on fixed-VRAM hardware.

Working paper title: *Cost-Aware Routing Across Heterogeneous Local Model Tiers on Fixed-VRAM Hardware*

## One-line pitch

Given a fixed, small memory budget and locally-hosted models with very different cost/quality/architecture profiles (dense transformer vs hybrid SSM), learn a routing policy `r(x) -> tier` per query and measure how close it gets to the oracle Pareto frontier - using real hardware numbers.

## Tiers (testbed)

| tier | model | arch | ctx |
|------|-------|------|-----|
| turbo | qwen2 3B | dense 36L, matmul-bound 96.7% | 32768 |
| vector | qwen3 8B (6 shards) | dense + YaRN 4x, dequant-bound 64.1% | 131072 |
| ultra | qwen35 9B (7 shards) | hybrid SSM128 + full-attn/4 + MTP1, YaRN 4x | 1048576 |

Single consumer GPU (~4GB VRAM class). Uniform HTTP interface: `GET /health`, `POST /completion`, `POST /v1/chat/completions` (default ports 8080/8081/8082).

## Repo layout

```
eval_prompts/  # 180 prompts (60 factual QA / 60 code / 60 reasoning), 70/30 split
harness/       # run_oracle.py — all tiers x all prompts, latency + output
scoring/       # score.py — correctness primary, judge secondary
router/        # heuristic.py, cascade.py, learned.py
analysis/      # make_plot.py — cost vs quality
SPEC.md        # full spec
RESULTS.md     # numbers as they land
```

## Repro

```bash
python eval_prompts/build_prompts.py
# live collection: one tier resident at a time (4GB VRAM class), uniform eval operating point
powershell -File harness/bringup.ps1 -Tier turbo  -EvalCtx 2048   # wait for HEALTHY
python harness/run_oracle.py --prompts eval_prompts/prompts.jsonl --out harness/oracle_turbo.jsonl  --tiers turbo  --n-predict 128 --timeout 300
powershell -File harness/bringup.ps1 -Tier turbo -Kill
powershell -File harness/bringup.ps1 -Tier vector -EvalCtx 2048
python harness/run_oracle.py --prompts eval_prompts/prompts.jsonl --out harness/oracle_vector.jsonl --tiers vector --n-predict 128 --timeout 600
powershell -File harness/bringup.ps1 -Tier vector -Kill
powershell -File harness/bringup.ps1 -Tier ultra  -EvalCtx 2048
python harness/run_oracle.py --prompts eval_prompts/prompts.jsonl --out harness/oracle_ultra.jsonl  --tiers ultra  --n-predict 128 --timeout 600
powershell -File harness/bringup.ps1 -Tier ultra -Kill
python harness/merge_tiers.py --out harness/oracle.jsonl
python scoring/score.py --in harness/oracle.jsonl --out scoring/scores.jsonl
python router/learned.py
python analysis/make_plot.py
```

No live servers? Exercise the full pipeline with deterministic synthetic
tier outputs (pipeline validation only — NOT for RESULTS.md numbers):

```bash
python harness/run_oracle.py --out harness/oracle_smoke.jsonl --mock
python scoring/score.py --in harness/oracle_smoke.jsonl --out scoring/scores_smoke.jsonl
python router/learned.py --scores scoring/scores_smoke.jsonl
python analysis/make_plot.py --scores scoring/scores_smoke.jsonl
```

Optional judge-based secondary scoring (SPEC §5, needs a tier up as judge,
e.g. ultra): `python scoring/score.py --in harness/oracle.jsonl --judge-url http://127.0.0.1:8082 --judge-max 60`.

See `SPEC.md` for formalization (tiers, budget B, router, oracle upper bound) and `RESULTS.md` for live numbers.

## Status

Oracle collected (540/540 tier-runs successful, zero errors): tier characterization + all four
router policies measured on held-out data. Main caveat recorded in RESULTS.md — the frozen
correctness metric favors turbo's answer formatting, so the oracle mostly selects turbo and the
achievable gap is small; judge-based/secondary quality scoring is the next cycle's priority.

## Note

Just in clear it wasn't obvious - MJOLNIR is a play on words, not an actual hammer. :)
