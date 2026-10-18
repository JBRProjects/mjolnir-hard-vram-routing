"""Cost-quality plot: policies (random, heuristic, cascade, learned, oracle).
Reads scoring/scores.jsonl (+ router/learned_preds.jsonl from router/learned.py)
and evaluates each policy on the held-out test rows:
- routes decide on PROMPT TEXT (scores.jsonl carries `prompt` through)
- cascade walks turbo->vector->ultra by per-tier confidence proxy; cost =
  accumulated latency of every attempted tier (failed attempts still cost)
- capture: (q_policy - q_random) / (q_oracle - q_random) on quality, per the
  RESULTS.md table (random = floor 0%, oracle = ceiling 100%).
Writes analysis/table.md (+ analysis/cost_quality.png if matplotlib available).
Usage: python analysis/make_plot.py   (run after router/learned.py for the learned point)
"""
import json, pathlib, random, sys

BASE = pathlib.Path(__file__).parent.parent
SCORES_DEFAULT = BASE / "scoring" / "scores.jsonl"
TIER_ORDER = ["turbo", "vector", "ultra"]

def load(scores_path):
    return [json.loads(l) for l in scores_path.read_text(encoding="utf-8").splitlines() if l.strip()]

def load_learned_preds():
    pp = BASE / "router" / "learned_preds.jsonl"
    preds = {}
    if pp.exists():
        for l in pp.read_text(encoding="utf-8").splitlines():
            if l.strip():
                j = json.loads(l)
                preds[j["id"]] = j["pred"]
    return preds

def main():
    sys.path.insert(0, str(BASE))
    scores_path = SCORES_DEFAULT
    if "--scores" in sys.argv:
        scores_path = pathlib.Path(sys.argv[sys.argv.index("--scores") + 1])
    if not scores_path.exists():
        print("no scores — run harness + scoring first"); return
    rows = load(scores_path)
    test = [r for r in rows if r.get("split") != "train"] or rows
    from router import heuristic, cascade
    preds = load_learned_preds()
    if preds:
        n_correct = sum(1 for r in test if preds.get(r.get("id")) == r["oracle"])
        print(f"[router] learned: {len(preds)} predictions, oracle-agreement {n_correct/max(1,len(test)):.3f} on test")
    else:
        print("[router] learned: no router/learned_preds.jsonl — learned point skipped (run router/learned.py)")

    def policy_random(r): return TIER_ORDER[int(random.random() * 3) % 3]
    def policy_heuristic(r): return heuristic.route(r.get("prompt", ""))

    def cascade_policy(r):
        tiers = r.get("tiers", {})
        conf = {t: (tiers.get(t) or {}).get("confidence", 0.0) for t in TIER_ORDER}
        chosen = cascade.cascade(conf)
        total = 0.0
        for t in TIER_ORDER:
            total += (tiers.get(t) or {}).get("latency_ms", 0)
            if t == chosen:
                break
        return chosen, total

    def eval_policy(chooser, accumulate_cost):
        costs, qs = [], []
        for r in test:
            if accumulate_cost:
                t, cost = chooser(r)
            else:
                t = chooser(r)
                cost = (r.get("tiers", {}).get(t) or {}).get("latency_ms", 0)
            costs.append(cost)
            qs.append((r.get("tiers", {}).get(t) or {}).get("quality", 0.0))
        return sum(costs) / max(1, len(costs)), sum(qs) / max(1, len(qs))

    random.seed(0)
    pts = {
        "random": eval_policy(policy_random, False),
        "heuristic": eval_policy(policy_heuristic, False),
        "cascade": eval_policy(cascade_policy, True),
        "learned": eval_policy(lambda r: preds.get(r.get("id"), "turbo"), False) if preds else (0.0, 0.0),
        "oracle": eval_policy(lambda r: r["oracle"], False),
    }
    q_floor, q_ceil = pts["random"][1], pts["oracle"][1]
    def capture(k):
        if q_ceil <= q_floor: return 0.0
        if k == "random": return 0.0
        if k == "oracle": return 100.0
        return 100.0 * (pts[k][1] - q_floor) / (q_ceil - q_floor)
    for k, (c, q) in pts.items():
        print(f"{k:12s} cost={c:9.1f}ms quality={q:.3f} capture={capture(k):5.1f}%")
    md = ("| policy | avg cost (ms) | quality | oracle-gap captured (%) |\n|---|---|---|---|\n"
          + "".join(f"| {k} | {c:.1f} | {q:.3f} | {capture(k):.1f}% |\n" for k, (c, q) in pts.items()))
    (BASE / "analysis" / "table.md").write_text(md, encoding="utf-8")
    print("wrote analysis/table.md")
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for k, (c, q) in pts.items():
            ax.scatter([c], [q]); ax.annotate(k, (c, q))
        ax.set_xlabel("avg cost (ms)"); ax.set_ylabel("avg quality")
        ax.set_title("Routing cost vs quality (test split)")
        fig.savefig(BASE / "analysis" / "cost_quality.png", dpi=150)
        print("wrote analysis/cost_quality.png")
    except ImportError:
        print("matplotlib missing — wrote table.md only")

if __name__ == "__main__":
    main()
