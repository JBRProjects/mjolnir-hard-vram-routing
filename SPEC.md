# Adaptive Model Routing Under Hard Memory Constraints

**Working title:** *Cost-Aware Routing Across Heterogeneous Local Model Tiers on Fixed-VRAM Hardware*

## 1. One-line pitch

Given a fixed, small memory budget and a set of locally-hosted models with very different cost/quality/architecture profiles (dense transformer vs. hybrid SSM), learn or design a routing policy that decides, per query, which tier to run on — and measure how close that policy gets to the Pareto frontier an oracle could achieve, using real hardware numbers rather than simulated ones.

## 2. Why this is the strongest candidate

Almost all "efficient inference" literature assumes elastic compute: if quality suffers, spin up a bigger GPU. The realistic constraint for personal/edge deployment is the opposite — the memory budget is fixed and small, and the question is how to make the most of a *heterogeneous* stack of models that are already sitting on disk. This is exactly the situation your turbo/vector/ultra tiers create: three real models with measured, divergent cost profiles (turbo matmul-bound, vector dequant-bound, ultra a hybrid SSM/full-attention architecture with its own bottleneck shape), running on a single consumer GPU (~4GB VRAM class). That's a testbed most academic groups have to fake. You already have it, instrumented, with real per-token timing data.

This also avoids the most common failure mode of this kind of project: routing papers that never touch real weights and just simulate a cost model. Yours would be argued from measured numbers on your own inference engine.

## 3. Research question

> Under a fixed memory budget, does a lightweight, learned per-query routing policy over heterogeneous model tiers achieve meaningfully better cost-quality tradeoffs than static or heuristic routing — and how much of the theoretical (oracle) gain does it actually capture?

Secondary question, if time allows: does routing accuracy degrade gracefully or catastrophically as memory pressure forces tier eviction (i.e., not all tiers resident at once)?

## 4. Formalization

- **Tiers:** a set of models `M = {m_1, ..., m_k}` with known/measured cost `c_i` (latency, memory footprint) and unknown-per-query quality `q_i(x)`.
- **Budget:** hard memory ceiling `B` (VRAM/RAM), which may force some tiers to be swapped rather than resident — this is the part standard MoE/cascade literature usually ignores, and where your turbo/vector/ultra footprint differences already give you real numbers to work with.
- **Router:** a function `r(x) -> i` choosing a tier before generation begins (cheap, since it must not dominate the cost it's trying to save).
- **Objective:** minimize expected cost subject to a quality floor, or equivalently maximize quality subject to a latency/memory budget — report both framings, since reviewers will ask for the one you didn't pick.
- **Oracle upper bound:** for each query in your eval set, the best tier *in hindsight* (ran all tiers, picked the best quality-per-cost) — this is your ceiling, and the gap between your router and it is your main result.

## 5. What "quality" means here (the part to nail down first)

This is the single biggest risk to the project, so resolve it before writing any routing code:
- Cheapest defensible option: use a judge model (or your own ultra tier) to score turbo/vector outputs pairwise on a held-out prompt set, producing a relative quality ranking rather than an absolute score.
- Alternative: task-specific correctness where possible (code-execution pass/fail, factual QA with known answers) — stronger evidence, narrower scope.
- Recommendation: pick one small, task-specific correctness benchmark (e.g., a code-gen or factual-QA slice) as your primary metric, and use judge-based scoring only as a secondary/robustness check. Reviewers trust the former far more than the latter.

## 6. Routing policy — three variants to compare (ascending complexity)

1. **Heuristic baseline:** route by prompt length / keyword features (e.g., "looks like code" -> turbo, "looks like reasoning" -> ultra). Cheap, no training, sets the floor.
2. **Cascade baseline:** always start at the cheapest tier; escalate to the next tier only if a cheap confidence signal (e.g., entropy of first-token logits, or a short self-consistency check) suggests low quality. This is the standard model-cascade approach — include it because reviewers will ask why you didn't just do this.
3. **Learned router:** a small classifier (logistic regression or tiny MLP over cheap prompt features — length, embedding from a small local encoder, maybe a few structural features) trained on the oracle labels from your eval set. This is the actual contribution; keep it deliberately small so it stays "cheap enough to not defeat the purpose."

The paper's core plot: cost vs. quality, four points/curves (heuristic, cascade, learned, oracle), ideally with a random-routing baseline too as a sanity floor.

## 7. Using your existing infra as the testbed

- Turbo/vector/ultra already give you three tiers with genuinely different architectures (dense vs. hybrid SSM), not just three sizes of the same model — this is a stronger claim than most routing papers get to make (they usually vary only parameter count).
- Your existing profiling (turbo 96.7% matmul-bound, vector 64.1% dequant-bound) is usable as-is in a "characterizing the tiers" section — this is real preliminary work, not something to redo.
- The known ultra prefill crash on long multi-chunk prompts is worth keeping in your back pocket as a "failure mode of naive routing" anecdote (routing a long prompt to ultra without knowing about its untested prefill path is exactly the kind of blind spot a coverage-blind router has) — don't fix it for this project unless it blocks eval, just note the constraint.
- Your `jayvis_server.exe` HTTP endpoints (`/completion`, `/v1/chat/completions`) mean you already have a uniform interface across tiers — the router just needs to pick which port/process to hit. Minimal engineering lift.

## 8. Minimal experimental design

- **Eval set:** 150-300 prompts, stratified across 2-3 task types (e.g., short factual QA, code generation, open-ended reasoning) — big enough to plot a curve, small enough to hand-label or judge-score in a weekend.
- **Procedure per prompt:** run all three tiers, record latency + quality; this gives you the oracle and the full cost-quality scatter for free.
- **Train/test split:** train router features on ~70%, report the cost-quality curve on the held-out 30%.
- **Report:** the four-curve plot (Section 6), a table of tier characterization stats (your existing profiling data), and a short ablation on router feature choice.

## 9. Repo structure suggestion

```
/eval_prompts/           # the 150-300 prompt set, tagged by task type
/harness/                # calls jayvis_server.exe endpoints, records (tier, latency, output)
/scoring/                # judge-scoring + task-specific correctness scripts
/router/                 # heuristic.py, cascade.py, learned.py
/analysis/               # notebook producing the cost-quality plot + tables
SPEC.md                  # this file
RESULTS.md               # fill in as you go
```

## 10. Write-up shape (workshop-paper length, 4-6 pages)

1. Motivation (fixed-memory personal deployment is understudied vs. elastic-cloud assumption)
2. Testbed description (your three real, architecturally distinct tiers — this section is your strongest asset, be concrete about it)
3. Formalization (Section 4)
4. Methods (three router variants)
5. Results (the four-curve plot + tier characterization table)
6. Discussion: how much oracle gap remains, where the learned router fails, memory-eviction robustness if you get to it
7. Related work: model cascades, speculative decoding, early-exit networks, mixture-of-depths — cite these as adjacent, argue your fixed-hardware framing is the differentiator

## 11. Realistic timeline

- Week 1: eval prompt set + scoring methodology decided and frozen
- Week 2: harness wired to `jayvis_server.exe`, oracle data collected (all tiers x all prompts)
- Week 3: heuristic + cascade baselines, learned router trained
- Week 4: analysis, plots, write-up draft

## 12. Biggest risks

- **Quality metric is shaky** -> mitigated by Section 5's recommendation to anchor on a correctness-based slice, not just judge scores.
- **Only 3 tiers is a small routing space** -> be upfront about this; frame it as "instantiated at k=3, extendable" rather than pretending it's a large action space.
- **Ultra's untested prefill path could crash mid-eval** -> run ultra's oracle pass first and isolate/skip failures rather than letting it take down the harness.
