# RESULTS — Adaptive Model Routing (fill in as you go)

## Tier characterization (measured, not simulated)

Operating point for the oracle cycle (uniform across tiers): `-c 2048 -b 256 -ub 256 -t 4 -tb 4 -np 1`,
temperature 0, n_predict 128, single NVIDIA T500 4 GB (4 GB VRAM class), one tier resident at a time.
Measured over all 180 prompts.

| tier | model | arch | ctx (eval) | VRAM resident (MB) | avg latency (s) | median (s) | avg tok/s | bottleneck note |
|------|-------|------|------------|--------------------|-----------------|-----------|-----------|-----------------|
| turbo | jayvis-turbo.gguf (qwen2 3B) | dense 36L | 2048 (model 32768) | n/m | 25.4 | 18.6 | 6.24 | GPU matmul-bound (prelim 96.7%); 300 MHz idle between calls |
| vector | jayvis-vector 6 shards (qwen3 8B) | dense + YaRN 4x | 2048 (model 131072) | 684 (GPU) | 36.0 | 28.5 | 4.09 | effectively CPU/IO-bound at eval: 2% GPU util, 300 MHz, weights mostly off-GPU |
| ultra | jayvis-ultra 7 shards (qwen35 9B) | hybrid SSM128 + full-attn/4 + MTP1, YaRN 4x | 2048 (model 1048576) | n/m | 55.4 | 40.6 | 3.06 | high variance (26–409 s); SSM/cache behavior; prefill crash not observed (prompts short) |

Notes:
- Native ctx (32768/131072/1048576) does not fit a 4 GB card concurrently; the uniform eval operating
  point (-c 2048) is required for a fair per-tier cost comparison and is what the numbers above reflect.
  Native-ctx vector measured ~0.5–0.7 tok/s (KV swap) in spot checks — recorded here as evidence, not used for the oracle.
- The vector tier OOMs on forward buffers at `-b 512` under ~3 GB free VRAM; `-b 256` is the stable eval config.
- One ultra request recorded a corrupt 54,352 s latency (clock anomaly); it was re-measured (27.1 s) and replaced.

Ports (default harness): turbo 8080 / vector 8081 / ultra 8082.
Endpoints: `GET /health`, `POST /completion`, `POST /v1/chat/completions`.
Bring-up is scripted: `harness/bringup.ps1 -Tier <tier> [-EvalCtx 2048]` / `-Kill`.

## Oracle
- Eval set: `eval_prompts/prompts.jsonl` — 180 prompts (60 factual QA / 60 code-gen / 60 reasoning), 70/30 train/test split.
- Procedure: run all 3 tiers per prompt (solo-tier passes, one server at a time), record (latency, output) → oracle = best quality-per-cost in hindsight.
- Oracle results: collected 2026-10-18/19. 540/540 tier-runs successful (zero errors).
  - Oracle tier distribution (all 180): turbo 158, vector 5, ultra 17. On the held-out test split (54): turbo 50, vector 3, ultra 1.

## Router comparison (held-out 30%)
| policy | avg cost (ms) | quality (correctness %) | oracle-gap captured (%) |
|--------|---------------|-------------------------|-------------------------|
| random | 39161.7 | 0.574 | 0.0 (floor) |
| heuristic | 39161.7 | 0.593 | 12.5 |
| cascade | 26606.4 | 0.676 | 68.8 |
| learned (logreg/tiny-MLP) | 26606.4 | 0.676 | 68.8 |
| oracle | 27286.1 | 0.722 | 100 (ceiling) |

Learned router details: sklearn LogisticRegression over 7 cheap prompt features, trained on 126 train
rows, oracle-agreement 0.926 on 54 test rows (it converges on "turbo most of the time", matching the
oracle's own 93% turbo rate). Cascade's confidence proxy passes on every turbo output, so it also
collapses to "always turbo". Random and heuristic happen to coincide in cost on this split (verified
as a coincidence in this data, not an evaluation bug).

Oracle-gap capture = `(q_policy − q_random) / (q_oracle − q_random)` on quality over the
held-out test rows (random = floor 0%, oracle = ceiling 100%). A policy CAN score negative
(lose to random routing); reported, not clipped.

### Key finding / caveat (metric bias)
Per-task quality (test split):

| task | turbo | vector | ultra |
|---|---|---|---|
| factual_qa | 0.944 | 0.944 | 0.944 |
| code_gen | 0.583 | 0.417 | 0.361 |
| reasoning | 0.500 | 0.250 | 0.278 |

The frozen correctness metric systematically favors turbo's answer formatting:
- factual QA: exact-substring metric saturates (all tiers 0.944) — no discrimination, so turbo wins on cost.
- code-gen: keyword-overlap rewards matching the reference's identifier names; larger models write valid
  but differently-named implementations and score 0.5 instead of 1.0.
- reasoning: requires a literal final `Answer:` line; turbo complies, vector/ultra mostly do not.

Consequence: the oracle collapses to "always turbo" and the achievable oracle gap is small
(0.676 → 0.722 quality). This is precisely the risk flagged in SPEC §5/§12 ("quality metric is shaky").
Next cycle should add the format-neutral judge-based secondary scoring (implemented, runnable via
`scoring/score.py --judge-url http://127.0.0.1:8082`) and/or an answer-extraction scoring pass before
any strong routing claims are made.

Pipeline notes:
- `scoring/scores.jsonl` carries `prompt` and per-tier `confidence` (cascade proxy) through, so
  routers/analysis evaluate on real prompt text and raw model outputs are not needed downstream.
- Cascade cost is the ACCUMULATED latency of every tier attempted (a failed turbo attempt still
  costs before the vector escalation).
- `router/learned.py` trains the router and writes `router/learned_preds.jsonl`, which
  `analysis/make_plot.py` consumes.
- `harness/run_oracle.py --mock` synthesizes deterministic tier outputs (seeded per id+tier,
  realistic error shapes incl. ultra long-prompt prefill crash + rare vector 503s) for
  pipeline validation only — it produces `*_smoke.jsonl` artifacts and is gitignored;
  RESULTS must come from live runs only.

Core plot: cost vs quality — see `analysis/cost_quality.png` (generated by `analysis/make_plot.py` from `harness/oracle.jsonl` + `scoring/scores.jsonl`).

## Failure log
- Ultra prefill crash on long multi-chunk prompts: isolate/skip, do not take down harness (SPEC §12).
- Vector sequential-only under ~4GB VRAM (60s health timeout, not parallel).

## Repro
```
python eval_prompts/build_prompts.py
python harness/run_oracle.py --prompts eval_prompts/prompts.jsonl --out harness/oracle.jsonl
python scoring/score.py --in harness/oracle.jsonl --out scoring/scores.jsonl
python router/learned.py  # trains on 70%, evaluates on 30%, writes router/learned_preds.jsonl
python analysis/make_plot.py
```
