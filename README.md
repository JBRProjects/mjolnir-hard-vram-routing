# MJØLNIR: Memory-constrained Joint Orchestration of Layered Neural Inference Runtimes under Hard VRAM Budgets

> *Whosoever routes this query, if he be worthy, shall possess the power of Thor - measured, not simulated.*
> Cost-aware per-query routing across heterogeneous local model tiers on fixed-VRAM hardware.

Working paper title: *Cost-Aware Routing Across Heterogeneous Local Model Tiers on Fixed-VRAM Hardware*

## One-line pitch

Given a fixed, small memory budget and locally-hosted models with very different cost/quality/architecture profiles (dense transformer vs hybrid SSM), learn a routing policy `r(x) -> tier` per query and measure how close it gets to the oracle Pareto frontier - using real hardware numbers.

## Tiers (testbed)

| equivalent tier | model | arch | ctx |
|------|-------|------|-----|
| turbo | qwen2 3B | dense 36L, matmul-bound 96.7% | 32768 |
| vector | qwen3 8B (6 shards) | dense + YaRN 4x, dequant-bound 64.1% | 131072 |
| ultra | qwen35 9B (7 shards) | hybrid SSM128 + full-attn/4 + MTP1, YaRN 4x | 1048576 |

Single consumer GPU (~4GB VRAM class). Uniform HTTP interface: `GET /health`, `POST /completion`, `POST /v1/chat/completions` (default ports 8080/8081/8082).

## Repo layout

```
eval_prompts/  # build_prompts.py      -> cycle-1 set (180 rows, 20 distinct items)
               # build_prompts_v2.py   -> cycle-2 set: 180 DISTINCT items
               #                         (60 factual / 60 code / 60 reasoning),
               #                         gold answers for reasoning, unit tests for code
               # build_legacy_gold.py  -> gold + tests for the cycle-1 base items
harness/       # run_oracle.py  - all tiers x all prompts, latency + output
               # run_cycle2.ps1 - scripted live cycle for the v2 prompt set
               # finalize_cycle2.ps1 - merge + score + train + plot + report in one go
scoring/       # score.py      - correctness primary (metric v2), judge secondary
               # extract.py    - format-neutral answer/code extraction + semantic matching
               # exec_tests.py - sandboxed execution of candidate code against unit tests
router/        # heuristic.py, cascade.py (confidence gate + verified gate), learned.py,
               # budget_router.py (cost-aware router, quality/cost frontier)
analysis/      # make_plot.py  - cost vs quality
               # report.py     - regenerates every RESULTS table from live data
               # tier_table.py - per-tier latency/throughput table
tests/         # test_mjolnir.py - unit checks (also run in CI)
SPEC.md        # full spec
RESULTS.md     # numbers as they land
PAPER.md       # working-paper draft (the narrative)
DATA.md        # manifest of the committed live measurements (off-site backup)
```

The live measurements are committed (raw completions, scores, figures, engine
logs) so the only copy of the hardware data is not the local disk - see
`DATA.md` for the manifest and how each file was produced.

## Quality metric (v2)

The cycle-1 metric was partly measuring output *format* (it required a literal
`Answer:` line and scored code by identifier overlap with the reference), which
made the oracle collapse onto turbo. The v2 metric separates content from
surface form:

* `trim_continuation` cuts the model's own conversation continuation
  (`Human: ...`) - the tiers are completion-style models and keep talking.
* `extract_answer` / `extract_code` pull the answer or the program out of any
  surface form (`Answer:`, `**Answer:**`, `\boxed{}`, fenced or bare code).
* `normalize_answer` + `answer_match` compare the extracted answer with the
  gold answer semantically (case, punctuation, articles, units, thousands
  separators, numeric equality).
* code tasks are scored by **executing** the candidate against the task's unit
  tests, not by comparing it with the reference solution.

The cycle-1 metric is still available for comparison: `--metric legacy`.

## Repro

Cycle 2 (current: format-neutral metric, 180 distinct eval items):

```bash
python eval_prompts/build_prompts_v2.py          # -> eval_prompts/prompts_v2.jsonl
# live collection: one tier resident at a time (4GB VRAM class), uniform eval operating point
powershell -File harness/run_cycle2.ps1           # turbo -> vector -> ultra -> merge
python scoring/score.py --in harness/oracle_v2.jsonl --out scoring/scores.jsonl --metric v2
python router/learned.py
python analysis/make_plot.py
```

Cycle 1 (legacy prompt set and metric):

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
python harness/run_oracle.py --prompts eval_prompts/prompts.jsonl --out harness/oracle_ultra.jsonl --tiers ultra  --n-predict 128 --timeout 600
powershell -File harness/bringup.ps1 -Tier ultra -Kill
python harness/merge_tiers.py --out harness/oracle.jsonl
python scoring/score.py --in harness/oracle.jsonl --out scoring/scores.jsonl --metric legacy
```

Re-score the cycle-1 LIVE data under the v2 metric (the gold lookup supplies the
answers and unit tests for the 20 cycle-1 base items):

```bash
python eval_prompts/build_legacy_gold.py
python scoring/score.py --in harness/oracle.jsonl --out scoring/scores_v2_legacy.jsonl --metric v2 --gold eval_prompts/legacy_gold.json
```

No live servers? Exercise the full pipeline with deterministic synthetic
tier outputs (pipeline validation only - NOT for RESULTS.md numbers):

```bash
python harness/run_oracle.py --out harness/oracle_smoke.jsonl --mock
python scoring/score.py --in harness/oracle_smoke.jsonl --out scoring/scores_smoke.jsonl
python router/learned.py --scores scoring/scores_smoke.jsonl
python analysis/make_plot.py --scores scoring/scores_smoke.jsonl
```

Unit checks (no GPU, no servers): `python tests/test_mjolnir.py`.

Optional judge-based secondary scoring (SPEC 5, needs a tier up as judge,
e.g. ultra): `python scoring/score.py --in harness/oracle_v2.jsonl --judge-url http://127.0.0.1:8082 --judge-max 60`.

See `SPEC.md` for formalization (tiers, budget B, router, oracle upper bound) and `RESULTS.md` for live numbers.

## Status

Cycle 2 complete: the quality metric was rebuilt (format-neutral extraction +
execution-based code scoring), the eval set was replaced with 180 distinct items
(cycle 1 had only 20 distinct items repeated with cosmetic variants, which also
leaked across the train/test split), and a fresh live oracle cycle was collected
on the new prompt set (540/540 tier-runs, zero errors, 2026-11-11).

Headline numbers on the held-out split: random routing 0.718, oracle 0.927;
the verifier-gated cascade captures 82.4% of the oracle gap, the cost-aware
learned router reaches 0.890 at 30.7 s (vs the oracle's 0.927 at 25.7 s), the
cheap-feature learned router 35.3%, and the hand-written heuristic is *worse
than random*. Full tables in RESULTS.md.

## Note

Just in clear it wasn't obvious - MJOLNIR is a play on words, not an actual hammer. :)
