"""Cost-quality analysis: policies (random, heuristic, cascade, cascade-verified,
learned, oracle) on the held-out test split.

Reads scoring/scores.jsonl (+ router/learned_preds.jsonl from router/learned.py)
and evaluates each policy:
  * routes decide on PROMPT TEXT (scores.jsonl carries `prompt` through)
  * cascade (confidence gate) walks turbo->vector->ultra by the surface
    confidence proxy; cost = accumulated latency of every attempted tier
  * cascade-verified walks the same ladder but stops at the first tier that
    passes the TASK-SPECIFIC VERIFIER (unit tests for code, answer presence for
    QA/reasoning); cost = accumulated latency of every attempted tier
  * capture: (q_policy - q_random) / (q_oracle - q_random) on quality
    (random = floor 0%, oracle = ceiling 100%)

Writes analysis/table.md (+ analysis/cost_quality.png if matplotlib is available)
Usage: python analysis/make_plot.py [--scores scoring/scores.jsonl] [--split test]
"""
import json, pathlib, random, sys, collections

BASE = pathlib.Path(__file__).parent.parent
SCORES_DEFAULT = BASE / "scoring" / "scores.jsonl"
TIER_ORDER = ["turbo", "vector", "ultra"]


def load(scores_path):
    return [json.loads(l) for l in pathlib.Path(scores_path).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_learned_preds():
    pp = BASE / "router" / "learned_preds.jsonl"
    preds = {}
    if pp.exists():
        for l in pp.read_text(encoding="utf-8").splitlines():
            if l.strip():
                j = json.loads(l)
                preds[j["id"]] = j["pred"]
    return preds


def _artifact(r, tier):
    """The candidate artefact of a tier, as carried by the scores file."""
    e = (r.get("tiers", {}) or {}).get(tier) or {}
    if "code" in e:
        return e.get("code") or ""
    return e.get("answer") or ""


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

    def cascade_verified_policy(r):
        tiers = r.get("tiers", {})
        contents = {t: _artifact(r, t) for t in TIER_ORDER}
        lat = {t: (tiers.get(t) or {}).get("latency_ms", 0) for t in TIER_ORDER}
        chosen, total = cascade.cascade_verified(r.get("task", ""), contents, lat,
                                                 r.get("fn"), r.get("tests"))
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
        "cascade-verified": eval_policy(cascade_verified_policy, True),
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
        print(f"{k:18s} cost={c:9.1f}ms quality={q:.3f} capture={capture(k):5.1f}%")

    md = ("| policy | avg cost (ms) | quality | oracle-gap captured (%) |\n|---|---|---|---|\n"
          + "".join(f"| {k} | {c:.1f} | {q:.3f} | {capture(k):.1f}% |\n" for k, (c, q) in pts.items()))

    # per-task, per-tier quality (the tier characterization table)
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in test:
        for t, v in (r.get("tiers") or {}).items():
            per[r["task"]][t].append(v.get("quality", 0.0))
    md += "\n## Per-task quality on the test split\n\n| task | " + " | ".join(TIER_ORDER) + " |\n|---|" + "---|" * len(TIER_ORDER) + "\n"
    for task in sorted(per):
        md += f"| {task} | " + " | ".join(
            f"{sum(per[task][t])/max(1,len(per[task][t])):.3f}" if per[task].get(t) else "n/a"
            for t in TIER_ORDER) + " |\n"

    # oracle label distribution
    dist = collections.Counter(r.get("oracle") for r in test)
    md += "\n## Oracle tier distribution (test split)\n\n| tier | count | share |\n|---|---|---|\n"
    for t in TIER_ORDER:
        n = dist.get(t, 0)
        md += f"| {t} | {n} | {n/max(1,len(test)):.3f} |\n"

    # cost-aware learned router: quality/cost Pareto curve (SPEC section 4)
    curve = []
    try:
        from router import budget_router
        curve, lat, note = budget_router.curve(rows)
        print(f"[budget] cost-aware learned router ({note})")
    except Exception as e:                       # pragma: no cover - optional
        print(f"[budget] curve unavailable: {e}")
    if curve:
        md += "\n## Cost-aware learned router: quality/cost frontier\n\n"
        md += "| cost penalty lambda | avg cost (ms) | quality | tier mix |\n|---|---|---|---|\n"
        for p in curve:
            mix = ", ".join(f"{t}:{p['mix'][t]}" for t in TIER_ORDER)
            md += f"| {p['lambda']:.2f} | {p['cost_ms']:.1f} | {p['quality']:.3f} | {mix} |\n"
        (BASE / "router" / "budget_curve.json").write_text(json.dumps(curve, indent=1), encoding="utf-8")

    (BASE / "analysis" / "table.md").write_text(md, encoding="utf-8")
    print("wrote analysis/table.md")
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for k, (c, q) in pts.items():
            ax.scatter([c], [q]); ax.annotate(k, (c, q))
        if curve:
            ax.plot([p["cost_ms"] for p in curve], [p["quality"] for p in curve],
                    marker="o", linestyle="--", color="gray", label="cost-aware learned router")
            ax.legend()
        ax.set_xlabel("avg cost (ms)"); ax.set_ylabel("avg quality")
        ax.set_title("Routing cost vs quality (test split)")
        fig.savefig(BASE / "analysis" / "cost_quality.png", dpi=150)
        print("wrote analysis/cost_quality.png")
    except ImportError:
        print("matplotlib missing — wrote table.md only")


if __name__ == "__main__":
    main()
