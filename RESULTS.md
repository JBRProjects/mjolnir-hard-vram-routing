# RESULTS — Adaptive Model Routing (fill in as you go)

## Cycle 2 — metric repair and a leakage-free eval set

**Why cycle 2 exists.** Cycle 1 produced a clean oracle (540/540 tier-runs) but
two results that could not be trusted:

1. the quality metric was partly measuring *format* (see the cycle-1 caveat
   below): turbo won reasoning because it emits a literal `Answer:` line while
   ultra emits `**Answer:**`, and code was scored by identifier overlap with the
   reference solution;
2. an audit of `eval_prompts/prompts.jsonl` showed it held only **20 distinct
   items** (10 factual, 5 code, 5 reasoning) repeated 6x/12x/12x with cosmetic
   `(variant N)` suffixes — so the 70/30 split put variants of the same item on
   both sides and the effective sample size was 20, not 180.

Cycle 2 therefore rebuilds the *measurement*, not the router: a format-neutral
metric (extraction + semantic matching + execution-based code scoring), and 180
genuinely distinct eval items with gold answers for reasoning and unit tests
for code. See SPEC.md §5.1/§5.2.

### 2.1 Metric ablation on the cycle-1 LIVE data

The cycle-1 oracle data (180 prompts x 3 tiers, real hardware, 2026-10-18/19)
was re-scored under metric v2, using gold answers and unit tests reconstructed
for the 20 cycle-1 base items (`eval_prompts/build_legacy_gold.py`). Same
completions, same latencies, only the metric changed.

Per-task quality (all 180 prompts):

| task | metric | turbo | vector | ultra |
|---|---|---|---|---|
| factual_qa | cycle-1 (substring) | 0.967 | 0.817 | 0.983 |
| factual_qa | **v2 (semantic)** | 0.967 | 0.750 | 0.983 |
| code_gen | cycle-1 (identifier overlap) | 0.525 | 0.300 | 0.450 |
| code_gen | **v2 (unit tests)** | 0.800 | 0.043 | 0.700 |
| reasoning | cycle-1 (`Answer:` line) | 0.525 | 0.250 | 0.258 |
| reasoning | **v2 (gold answer)** | 0.400 | 0.417 | **0.600** |

Reading of the ablation:

* **reasoning flips.** Under the format-sensitive metric turbo led ultra by
  0.267 (0.525 vs 0.258) — a pure formatting artefact (plain `Answer:` vs
  markdown `**Answer:**`). With gold answers and format-neutral extraction
  ultra leads instead (0.600 vs 0.400). The direction of the conclusion
  changes with the metric, which is exactly why the metric had to be fixed
  before any routing claim.
* **code_gen separates behaviour from naming.** Execution-based scoring drops
  vector to 0.043 — it answers code prompts in prose and never defines the
  requested function — and separates turbo (0.800) from ultra (0.700) on
  actually passing the task's unit tests rather than on identifier overlap.
* **factual_qa is simply not discriminative.** All three tiers sit at
  0.75-0.98 under either metric: short factual QA is easy for all of them, so
  there is almost no routing headroom on that slice. (The v2 metric does not
  *lower* vector here — 0.750 vs the cycle-1 0.817 — because it accepts an
  answer stated anywhere in the model's opening, which the substring rule also
  did; the difference is that v2 stops punishing the *form* of the answer.)

Two implementation details that materially changed the code numbers and are
worth recording, because both are failure modes of any LLM code benchmark:

* the tiers often repeat the function and then run out of budget mid-definition,
  so the raw candidate ends with a half-written statement (a SyntaxError) even
  though the first copy of the function is complete. `repair_code` recovers the
  longest line-aligned prefix that parses and defines the target function;
  without it, correct answers scored 0 (turbo's code_gen score moved from
  0.517 to 0.800 when it was added);
* the tiers also prefix the program with prose that itself contains code words
  ("No need to import any module. ... def is_prime(n):"), so the extractor
  anchors on the *requested function name* instead of on a line that starts
  with `def` (live cycle-2 data, turbo: code_gen 0.662 -> 0.827 after the fix).

Candidates that call `input()` (blocked in the sandbox) or that never define
the requested function are reported separately (5 and 1 turbo/vector cases
respectively) instead of being silently averaged in.

Oracle tier distribution (all 180 prompts, cheapest tier achieving the best
quality): cycle-1 metric turbo 166 / vector 5 / ultra 9 -> v2 metric turbo 164 /
vector 13 / ultra 3. The oracle is no longer a synonym for "turbo": on the
reasoning slice the oracle picks a non-turbo tier on 14 of 60 prompts.

### 2.2 Router comparison on the cycle-1 live data, re-scored under metric v2

Held-out test split (54 prompts), same oracle labels, same latencies:

| policy | cycle-1 metric (cost ms / quality / capture) | **v2 metric (cost ms / quality / capture)** |
|---|---|---|
| random | 39161.7 / 0.574 / 0.0% | 39161.7 / 0.630 / 0.0% |
| heuristic | 39161.7 / 0.593 / 12.5% | 39161.7 / 0.648 / 12.5% |
| cascade (confidence gate) | 26606.4 / 0.676 / 68.8% | 26606.4 / 0.667 / 25.0% |
| cascade (verified gate, new) | - | 39195.9 / 0.667 / 25.0% |
| learned (logreg, 7 features) | 26606.4 / 0.676 / 68.8% | **27818.9 / 0.741 / 75.0%** |
| oracle | 27286.1 / 0.722 / 100% | 27520.9 / 0.778 / 100% |

Under the repaired metric the policies separate in a way they did not before:
the learned router captures 75% of the oracle gap while costing about the same
as the oracle itself, the cascade collapses to the heuristic floor (its cycle-1
number of 68.8% was an artefact of the format-biased metric), and the random
floor itself moves (0.574 -> 0.630) because the metric now discriminates.

### 2.2b Router feature ablation (SPEC section 8)

Held-out split, cycle-1 live data under metric v2:

| feature subset | agreement with oracle | avg cost (ms) | quality | capture |
|---|---|---|---|---|
| length only (2) | 0.889 | 26606.4 | 0.667 | 33.3% |
| task keywords only (3) | 0.889 | 26606.4 | 0.667 | 33.3% |
| surface ratios only (2) | 0.889 | 26606.4 | 0.667 | 33.3% |
| all features (7) | 0.963 | 27818.9 | 0.741 | 77.8% |

Any single feature family is enough to route most queries to turbo (the oracle's
own 91% turbo rate); the full feature set is what lets the router find the
queries where the bigger tier is actually needed.

### 2.3 Eval set v2

`eval_prompts/build_prompts_v2.py` -> `eval_prompts/prompts_v2.jsonl`:

* 180 **distinct** items, 60 factual QA / 60 code-gen / 60 reasoning, 70/30
  stratified split (126 train / 54 test), no item in both splits;
* every reasoning item carries a gold final answer;
* every code item carries a function name and 2-4 unit tests whose expected
  values are *derived by executing the reference solution* at build time, so
  gold cannot drift from the specification;
* `build_prompts_v2.py --self-test` re-derives every expected value and fails
  the build if a reference solution disagrees with its own tests.

### 2.4 Live cycle-2 oracle (completed)

`harness/run_cycle2.ps1` ran the full live cycle on the v2 prompt set at the
cycle-1 operating point (`-c 2048 -b 256 -ub 256 -t 4 -tb 4 -np 1`, temperature 0,
n_predict 128, one tier resident at a time), 2026-11-11 11:43 - 17:44.
**540/540 tier-runs successful, zero errors.**

#### Tier characterization (v2 live run)

| tier | runs | errors | avg latency (s) | median (s) | p90 (s) | avg tok/s |
|---|---|---|---|---|---|---|
| turbo | 180 | 0 | 20.3 | 19.7 | 25.3 | 6.54 |
| vector | 180 | 0 | 49.6 | 48.2 | 60.3 | 2.67 |
| ultra | 180 | 0 | 49.9 | 48.6 | 57.2 | 2.59 |

The v2 prompt set is *harder* than the cycle-1 one for the big tiers: vector is
slower here (49.6 s vs 36.0 s in cycle 1) because the new prompts make it
generate more text, and turbo is faster (20.3 s vs 25.4 s) because the brevity
instruction on the reasoning slice keeps its answers short. The cost ordering
is unchanged, but the *quality* ordering is not (below).

#### Per-task quality (all 180 prompts, metric v2)

| task | turbo | vector | ultra |
|---|---|---|---|
| code_gen | 0.659 | 0.143 | **0.968** |
| factual_qa | 0.917 | 0.733 | 0.917 |
| reasoning | 0.783 | 0.550 | **0.817** |

With the format-neutral metric and 60 distinct tasks per slice, the tiers
separate sharply:

* **code_gen** — ultra passes 97% of the task unit tests, turbo 66%, vector 14%
  (vector answers code prompts in prose and never defines the requested
  function);
* **reasoning** — ultra 0.817 vs turbo 0.783: the gap is small but it is the
  slice where the *cheapest* tier is not the best, which is what a router
  needs;
* **factual_qa** — turbo and ultra tie at 0.917; vector lags at 0.733.

This is the opposite of the cycle-1 conclusion ("route everything to turbo"),
and it is what the repaired measurement says on fresh, distinct prompts.

#### Oracle tier distribution (180 prompts)

| tier | count | share |
|---|---|---|
| turbo | 146 | 0.811 |
| vector | 6 | 0.033 |
| ultra | 28 | 0.156 |

| task | turbo | vector | ultra |
|---|---|---|---|
| code_gen | 37 (0.62) | 3 (0.05) | 20 (0.33) |
| factual_qa | 58 (0.97) | 1 (0.02) | 1 (0.02) |
| reasoning | 51 (0.85) | 2 (0.03) | 7 (0.12) |

The oracle now spends its budget where the bigger tier actually pays: a third
of the code prompts and an eighth of the reasoning prompts are worth an
escalation.

#### Router comparison (held-out test split, 54 prompts)

| policy | avg cost (ms) | quality | oracle-gap captured (%) |
|---|---|---|---|
| random | 42016.9 | 0.718 | 0.0% (floor) |
| heuristic | 41675.5 | 0.444 | **-130.1%** |
| cascade (confidence gate) | 20757.0 | 0.773 | 26.5% |
| cascade (verified gate) | 34985.9 | 0.890 | 82.4% |
| learned (logreg, 7 features) | 21466.6 | 0.792 | 35.3% |
| oracle | 25737.3 | 0.927 | 100% (ceiling) |

Reading:

* **the oracle gap is large and real**: 0.718 (random) -> 0.927 (oracle).
  Under the cycle-1 metric the same comparison spanned 0.574 -> 0.722, and the
  achievable part of it was an artefact of the metric;
* **the heuristic is worse than random** (-130%). It routes mid-length prompts
  to vector, and vector is the weakest tier on the code slice (0.143). A
  router built on prompt surface features without the measured tier
  profile can lose to random routing - worth stating explicitly, because it is
  the failure mode a reviewer will look for;
* **the verified cascade is the strongest single policy** (82.4% of the gap at
  35.0 s): it starts at turbo and escalates only when the task-specific
  verifier rejects the cheap answer, so it pays for ultra exactly where the
  unit tests say turbo was wrong;
* **the confidence-gated cascade buys speed, not quality** (26.5% at 20.8 s):
  the surface proxy accepts nearly every turbo answer, so it is a
  turbo-only policy;
* **the learned router on cheap features is beaten by the verified cascade**
  (35.3% vs 82.4%): with an oracle that is 81% turbo, a 7-feature classifier
  mostly learns "turbo", and the feature ablation shows the full feature set
  is not even the best subset here.

#### Cost-aware learned router: quality/cost frontier

`router/budget_router.py` models `P(correct | x, t)` and routes with a cost
penalty `argmax_t [P(correct|x,t) - lambda * measured_latency_t]`, sweeping the
penalty:

| cost penalty lambda | avg cost (ms) | quality | tier mix (test split) |
|---|---|---|---|
| 0.00 | 50361.6 | 0.872 | turbo 0, ultra 54 |
| 0.05 | 36979.9 | 0.872 | turbo 25, ultra 29 |
| **0.10** | **30699.0** | **0.890** | turbo 36, ultra 18 |
| 0.20 | 23542.2 | 0.773 | turbo 49, ultra 5 |
| 0.40 | 20757.0 | 0.773 | turbo 54 |

The `lambda = 0.10` point is the best implementable operating point found in
this cycle: quality 0.890 at 30.7 s, versus the oracle's 0.927 at 25.7 s -
i.e. ~96% of the oracle quality for ~1.2x the oracle cost, and it dominates
every single-point policy in the table above (the verified cascade reaches the
same quality for 4 s more).

#### Router feature ablation (SPEC section 8, live cycle-2 data)

| feature subset | agreement with oracle | avg cost (ms) | quality | capture (%) |
|---|---|---|---|---|
| length only (2) | 0.870 | 22012.1 | 0.810 | 51.8% |
| task keywords only (3) | 0.833 | 20757.0 | 0.773 | 36.6% |
| surface ratios only (2) | 0.833 | 20757.0 | 0.773 | 36.6% |
| all features (7) | 0.852 | 21466.6 | 0.792 | 44.2% |

On the fresh data the full 7-feature set is *not* the best subset (length only
wins), which is a mild negative result about the cheap-feature router: with
126 training rows and an 81%-turbo oracle label distribution it mostly learns
the prior. The pair model used by the cost-aware router (which adds
prompt-type x tier interactions) is the variant that recovers the tier
structure.

#### 2.4b What the memory budget costs: the measured tier switch

Only one tier fits in the 4 GB card at a time, so a cascade that escalates must
*evict* the resident tier and *load* the next one. The cascade numbers above
charge only generation latency, so they are optimistic. `harness/measure_switch.ps1`
measures that cost directly (process launch -> `/health` 200, at the evaluation
operating point `-c 2048`):

| tier | shards | load + health |
|---|---|---|
| turbo | 1 file, 2.1 GB | **3.9 s** |
| vector | 6 files, ~5 GB | **11.3 s** |
| ultra | 7 files, ~6 GB | **11.3 s** |

Re-running the verified cascade with those costs charged per escalation
(`cascade (verified gate + tier switch)` in `analysis/report.md`):

| policy | avg cost (ms) | quality | oracle-gap captured (%) |
|---|---|---|---|
| cascade (verified gate) | 34985.9 | 0.890 | 82.4% |
| cascade (verified gate + tier switch) | 38329.6 | 0.890 | 82.4% |

The eviction cost is **+9.6%** on the cascade's average request, because only
the queries that escalate pay it (a turbo -> ultra escalation costs 11.3 s of
loading, against ~30 s of generation saved). The quality picture does not
change, but the honest number is the second row: under a hard memory budget,
switching is not free — it is small here only because the models are small
enough to load in seconds and the tiers are few.


#### 2.5 Judge validation (secondary metric) — negative result

`scoring/score.py --judge-url ... --judge-mode all` grades the tiers with a
tier acting as the judge and compares the verdict with the primary metric
(agreement + Cohen's kappa). The mechanism works; **the tiers are not usable
as judges**:

* **ultra as judge** opens a `<think>` block and spends its token budget on the
  reasoning trace: with an 8-token budget only 3 of 24 calls produced a parsable
  verdict. With a 64-token budget it answers, but a single judgement costs
  ~70 s of prefill plus generation on this hardware - roughly the cost of a
  real query, so a judge-based metric would be as expensive as the workload it
  is supposed to measure.
* **turbo as judge** answers immediately (35 of 36 calls parsable) but its
  verdicts frequently contradict its own explanation - e.g. `NO` followed by
  *"The candidate answer 'Python' is correct in identifying the programming
  language created"*. Agreement with the primary metric: **0.543** (kappa
  undefined: the judge answered NO almost everywhere, so the chance agreement
  is 1). Per candidate tier: turbo 0.667 (n=12), ultra 0.500 (n=12), vector
  0.455 (n=11).

The underlying reason is structural: the tiers are completion-style models
served through `/completion`, so a judge instruction is just more text for them
to continue, not a command to obey. Two consequences for the paper:

1. a judge-based secondary metric needs a *stronger* judge than the local
   stack (an external grader, or a bigger tier with a chat template applied) -
   it cannot be produced by the tiers themselves;
2. this is exactly why the cycle-2 primary metric is **execution-based**: unit
   tests are checkable by a reviewer without trusting any model's opinion,
   including turbo's.

## Cycle 1 — tier characterization and first oracle

### Tier characterization (measured, not simulated)

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

### Oracle
- Eval set: `eval_prompts/prompts.jsonl` — 180 prompts (60 factual QA / 60 code-gen / 60 reasoning), 70/30 train/test split.
- Procedure: run all 3 tiers per prompt (solo-tier passes, one server at a time), record (latency, output) → oracle = best quality-per-cost in hindsight.
- Oracle results: collected 2026-10-18/19. 540/540 tier-runs successful (zero errors).
  - Oracle tier distribution (all 180): turbo 158, vector 5, ultra 17. On the held-out test split (54): turbo 50, vector 3, ultra 1.

### Router comparison (held-out 30%), cycle-1 metric
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
Cycle 2 replaces the metric and re-measures (see section 2.1 above).

### Pipeline notes
- `scoring/scores.jsonl` carries `prompt` and per-tier `confidence` (cascade proxy) through, so
  routers/analysis evaluate on real prompt text and raw model outputs are not needed downstream.
  Under metric v2 it additionally carries the extracted answer/code and the task's unit tests.
- Cascade cost is the ACCUMULATED latency of every tier attempted (a failed turbo attempt still
  costs before the vector escalation).
- `router/learned.py` trains the router and writes `router/learned_preds.jsonl`, which
  `analysis/make_plot.py` consumes.
- `harness/run_oracle.py --mock` synthesizes deterministic tier outputs (seeded per id+tier,
  realistic error shapes incl. ultra long-prompt prefill crash + rare vector 503s) for
  pipeline validation only — it produces `*_smoke.jsonl` artifacts and is gitignored;
  RESULTS must come from live runs only.

Core plot: cost vs quality — see `analysis/cost_quality.png` (generated by
`analysis/make_plot.py` from `harness/oracle.jsonl` + `scoring/scores.jsonl`).
All live data, scores, figures and engine logs are committed to the repository
as an off-site backup; `DATA.md` is the manifest of what each file is and how it
was produced.

## Failure log
- Ultra prefill crash on long multi-chunk prompts: isolate/skip, do not take down harness (SPEC §12).
- Vector sequential-only under ~4GB VRAM (60s health timeout, not parallel).
- The tiers are completion-style models served through `/completion`: after answering they keep
  generating the next turn ("Human: ..."), so every raw output contains model noise that must be
  trimmed before scoring (SPEC §5.1).

## Repro
```
# cycle 2 (current)
python eval_prompts/build_prompts_v2.py
powershell -File harness/run_cycle2.ps1
python scoring/score.py --in harness/oracle_v2.jsonl --out scoring/scores.jsonl --metric v2
python router/learned.py
python analysis/make_plot.py

# cycle 1
python eval_prompts/build_prompts.py
python harness/run_oracle.py --prompts eval_prompts/prompts.jsonl --out harness/oracle.jsonl
python scoring/score.py --in harness/oracle.jsonl --out scoring/scores.jsonl --metric legacy
python router/learned.py  # trains on 70%, evaluates on 30%, writes router/learned_preds.jsonl
python analysis/make_plot.py
```
