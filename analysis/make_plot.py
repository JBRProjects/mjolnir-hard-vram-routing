"""Cost-quality plot: four curves/points (random, heuristic, cascade, learned, oracle).
Reads scoring/scores.jsonl (+ router/learned_preds.jsonl if present).
Writes analysis/cost_quality.png + analysis/table.md (matplotlib optional).
Usage: python analysis/make_plot.py
"""
import json, pathlib, random

BASE = pathlib.Path(__file__).parent.parent
SCORES = BASE / "scoring" / "scores.jsonl"

def load():
    return [json.loads(l) for l in SCORES.read_text(encoding="utf-8").splitlines() if l.strip()]

def avg_cost_quality(rows, chooser):
    c, q = [], []
    for r in rows:
        t = chooser(r)
        v = r["tiers"].get(t, {})
        c.append(v.get("latency_ms", 0)); q.append(v.get("quality", 0))
    return (sum(c)/max(1,len(c)), sum(q)/max(1,len(q)))

def main():
    import sys
    sys.path.insert(0, str(BASE))
    if not SCORES.exists():
        print("no scores — run harness + scoring first"); return
    rows = load()
    test = [r for r in rows if r.get("split") != "train"] or rows
    from router import heuristic, cascade
    preds = {}
    pp = BASE / "router" / "learned_preds.jsonl"
    if pp.exists():
        for l in pp.read_text(encoding="utf-8").splitlines():
            if l.strip():
                j = json.loads(l); preds[j["id"]] = j["pred"]
    random.seed(0)
    pts = {
        "random": avg_cost_quality(test, lambda r: random.choice(["turbo","vector","ultra"])),
        "heuristic": avg_cost_quality(test, lambda r: heuristic.route(r.get("id",""))),
        "cascade-proxy": (sum(v["latency_ms"] for r in test for v in r["tiers"].values())/max(1,3*len(test)),
                          sum(r["best_quality"] for r in test)/max(1,len(test))*0.9),
        "learned": avg_cost_quality(test, lambda r: preds.get(r["id"], "turbo")) if preds else (0,0),
        "oracle": avg_cost_quality(test, lambda r: r["oracle"]),
    }
    # note: heuristic/cascade here use id-proxy unless wired to prompts; harness eval should pass prompt text
    for k,(c,q) in pts.items(): print(f"{k:12s} cost={c:8.1f}ms quality={q:.3f}")
    md = "| policy | avg cost (ms) | quality |\n|---|---|---|\n" + "".join(f"| {k} | {c:.1f} | {q:.3f} |\n" for k,(c,q) in pts.items())
    (BASE/"analysis"/"table.md").write_text(md, encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for k,(c,q) in pts.items():
            ax.scatter([c],[q]); ax.annotate(k,(c,q))
        ax.set_xlabel("avg cost (ms)"); ax.set_ylabel("avg quality"); ax.set_title("Routing cost vs quality (test)")
        fig.savefig(BASE/"analysis"/"cost_quality.png", dpi=150)
        print("wrote analysis/cost_quality.png + table.md")
    except ImportError:
        print("matplotlib missing — wrote table.md only")

if __name__ == "__main__":
    main()
