# Cost-Aware Routing Across Heterogeneous Local Model Tiers on Fixed-VRAM Hardware

Working paper (draft). Numbers live in `RESULTS.md`; this file is the narrative.
Cycle-2 live numbers are marked `[pending]` until `harness/run_cycle2.ps1` finishes.

## Abstract

Routing frameworks in production sit behind elastic, homogeneous serving stacks:
when a request is hard, more compute is bought. The deployment this paper studies
has no such option. A single consumer GPU with ~4 GB of VRAM must host a small
stack of *architecturally different* local models - a dense 3B transformer, a
8B dense model sharded across six files, and a 9B hybrid SSM/attention model
sharded across seven - and only one of them fits at a time. Under that budget
the question is not "which model is best" but "which model should answer *this*
query, given that a bigger model is both slower and may not fit".

We build MJØLNIR, a per-query router over three such tiers on real hardware,
and we measure it against the oracle that sees every tier's output in hindsight.
The measurement turned out to be the hard part: on our first cycle the quality
metric that defines "correct" was sensitive to output *format* (it required a
literal `Answer:` line and scored programs by identifier overlap with the
reference solution), so the oracle collapsed onto the cheapest tier and the
apparent routing headroom was an artefact of the metric. We therefore rebuild the
measurement - format-neutral answer extraction plus execution-based scoring of
generated programs against per-task unit tests - and replace an eval set that
turned out to contain only 20 distinct items repeated under cosmetic suffixes,
with 180 distinct items whose gold answers and unit tests are generated and
self-checked at build time.

On the first live data set, re-scored under the repaired metric, the conclusion
*changes*: the tier that looked worse than the cheapest one on reasoning turns
out to be the better one (0.550 vs 0.400), and a cheap-feature learned router
captures 70% of the oracle's quality gap while being cheaper than the oracle
itself. We also report what the repair costs: a cascade gated on a surface
confidence signal collapses to "always cheapest", which is the honest reading of
the cycle-1 result.

## 1. Motivation

Almost all efficient-inference work assumes elastic compute. The deployment
motivation for local, personal-scale inference inverts that assumption: the
memory budget is fixed and small, the models are already on disk, and the
interesting question becomes *orchestration* - which of several heterogeneous
models should serve a given query when you cannot have them all resident, and
when the models differ in architecture rather than only in size.

Our testbed is a real instance of that setting: one NVIDIA T500 (4 GB VRAM
class) hosting three tiers whose measured bottlenecks differ fundamentally
(turbo: GPU matmul-bound; vector: dequant/CPU-IO-bound with weights mostly
off-GPU; ultra: hybrid SSM with high variance). This is a setting most routing
papers have to simulate, because they vary only parameter count and because they
can afford elastic serving.

## 2. Testbed

| tier | model | architecture | native ctx | measured at eval ctx 2048 |
|---|---|---|---|---|
| turbo | qwen2 3B (single file) | dense 36L | 32768 | 25.4 s avg, 6.24 tok/s |
| vector | qwen3 8B (6 shards) | dense + YaRN 4x | 131072 | 36.0 s avg, 4.09 tok/s |
| ultra | qwen35 9B (7 shards) | hybrid SSM128 + full-attn/4 + MTP1, YaRN 4x | 1048576 | 55.4 s avg, 3.06 tok/s |

The native contexts do not fit concurrently on a 4 GB card, so the evaluation
uses a uniform operating point (`-c 2048 -b 256 -ub 256 -t 4 -tb 4 -np 1`,
temperature 0, `n_predict` 128) and runs one tier resident at a time. That
concession is itself part of the setting: the memory budget is what forces the
sequencing, and the router's job is to decide which tier to pay for.

Details, including the failure log (ultra's untested long-prompt prefill path,
vector's sequential-only residency), are in RESULTS.md.

## 3. Formalization

*Tiers* `M = {m_1..m_k}` with measured cost `c_i` (here: end-to-end latency,
which folds in both generation throughput and any tier-switch cost) and
per-query quality `q_i(x)`. *Budget* `B`: a hard memory ceiling that in
practice means only a subset of tiers is resident, so tier choice also implies
a residency cost. *Router* `r(x) -> i`, chosen before generation and cheap
relative to generation. *Objective*: maximise quality subject to a latency
budget (the curve in RESULTS.md), equivalently minimise cost subject to a
quality floor. *Oracle*: for each query, the tier that maximises quality, ties
broken by the cheaper tier - the ceiling any router can reach.

## 4. Methods

### 4.1 Quality metric (the part that had to be fixed first)

Quality is task-specific correctness, and the details decide the paper:

* the model's own conversation continuation is trimmed (the tiers are
  completion-style models; after answering they keep generating the next turn);
* the final answer is extracted from any surface form and compared to the gold
  answer semantically;
* generated programs are **executed** against per-task unit tests, so
  correctness is behaviour rather than resemblance to the reference solution.

Section 5.2 of SPEC.md records why the previous metric was not defensible, and
RESULTS.md section 2.1 shows the size of the effect on live data.

### 4.2 Routers

1. *Heuristic* (floor): route on prompt length and code/reasoning keywords.
2. *Cascade*: start at the cheapest tier, escalate on a cheap signal. Two
   gates are reported - a surface confidence proxy, and a task-specific
   verifier (execute the unit tests for code, require a well-formed answer for
   QA). Every attempted tier is paid for.
3. *Learned*: logistic regression over seven cheap prompt features (lengths,
   keyword counts, digit/bracket ratios) trained on the oracle labels of the
   training split. Deliberately tiny: it must not defeat the cost it saves.
4. *Cost-aware learned router*: instead of classifying the oracle tier, model
   `P(correct | x, t)` over (prompt features, tier one-hot, prompt-type x tier
   interactions) and route with a cost penalty, sweeping the penalty to trace
   the quality/cost frontier.
5. *Oracle* (ceiling) and *random* (floor sanity check).

## 5. Results

See RESULTS.md. Headline findings on live hardware:

* the repaired metric reverses the cycle-1 ranking on reasoning (turbo 0.400,
  ultra 0.550) and separates code generation by behaviour rather than naming
  (turbo 0.800, ultra 0.700, vector 0.043 - vector answers code prompts in
  prose);
* the learned router captures 70% of the oracle gap at lower cost than the
  oracle on the cycle-1 data, and the cascade gated on surface confidence
  collapses to the cheapest tier (the cycle-1 "68.8% capture" was a metric
  artefact);
* live cycle-2 numbers on 180 distinct items: `[pending]`.

## 6. Discussion

*Why the metric was the story.* With three tiers whose outputs differ mostly in
style, any metric that touches style will decide the experiment. Our first
cycle produced a clean, publishable-looking table whose main conclusion
("route everything to the cheapest tier") was a property of the metric.

*Where routing still fails.* [pending cycle-2 analysis]

*Memory-eviction robustness.* Only one tier is resident at a time, so tier
switching is a real cost that the latency measurement includes; a future cycle
should add the swap cost explicitly.

## 7. Related work

Model cascades and early-exit networks route between models of increasing cost
using a cheap confidence signal; mixture-of-depths and adaptive-depth
inference vary compute within a model. Our framing differs in two ways: the
action space is *architecturally* heterogeneous (dense vs hybrid SSM) rather
than a size ladder, and the budget is a hard memory ceiling that forces
eviction, so the routing decision is entangled with residency.

## 8. Limitations

* three tiers is a small action space - instantiated at k=3, extensible;
* the eval set is 180 short items; a judge-based secondary metric is
  implemented and used to validate the primary metric on the judged subset;
* the tiers are completion-style models served through `/completion`, so
  outputs are noisier than a chat-tuned serving stack would produce.

## 9. Reproducibility

See README.md. One command per tier for collection (`harness/run_cycle2.ps1`),
then `scoring/score.py`, `router/learned.py`, `analysis/make_plot.py`; unit
checks run without a GPU (`python tests/test_mjolnir.py`) and in CI.
