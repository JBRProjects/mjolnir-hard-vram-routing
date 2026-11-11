"""Generate the RESULTS.md tables from a live oracle run (reproducible numbers).

Reads the merged oracle file and the scores file and prints/writes a markdown
block with:
  * tier characterization (runs, errors, latency, throughput)
  * per-task quality per tier
  * oracle tier distribution
  * router comparison (random / heuristic / cascade / cascade-verified / learned / oracle)
  * the cost-aware learned router's quality/cost frontier
  * the router feature ablation

Usage:
  python analysis/report.py --in harness/oracle_v2.jsonl --scores scoring/scores.jsonl \
      [--out analysis/report.md]
"""
import argparse, collections, json, pathlib, statistics, sys

BASE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
TIER_ORDER = ["turbo", "vector", "ultra"]


def load(path):
    return [json.loads(l) for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def pct(vals, q):
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(BASE / "harness" / "oracle_v2.jsonl"))
    ap.add_argument("--scores", default=str(BASE / "scoring" / "scores.jsonl"))
    ap.add_argument("--out", default=str(BASE / "analysis" / "report.md"))
    args = ap.parse_args()
    if not pathlib.Path(args.inp).exists():
        print(f"no oracle file at {args.inp}"); return
    if not pathlib.Path(args.scores).exists():
        print(f"no scores file at {args.scores}"); return
    rows = load(args.scores)
    oracle_rows = load(args.inp)
    by_id = {r["id"]: r for r in oracle_rows}
    test = [r for r in rows if r.get("split") != "train"] or rows
    out = []
    A = out.append

    A(f"## Report: `{pathlib.Path(args.inp).name}` / `{pathlib.Path(args.scores).name}`\n")
    A(f"Prompts: {len(rows)} (test split: {len(test)})\n")

    # tier characterization
    A("\n### Tier characterization\n")
    A("| tier | runs | errors | avg latency (s) | median (s) | p90 (s) | avg tok/s |")
    A("|---|---|---|---|---|---|---|")
    for t in TIER_ORDER:
        lat, tps, errs, runs = [], [], 0, 0
        for r in rows:
            o = by_id.get(r["id"], {})
            e = (o.get("tiers") or {}).get(t)
            if not e:
                continue
            runs += 1
            if e.get("error"):
                errs += 1
                continue
            if e.get("latency_ms"):
                lat.append(e["latency_ms"] / 1000.0)
            tm = e.get("timings") or {}
            if tm.get("predicted_per_second"):
                tps.append(float(tm["predicted_per_second"]))
        avg_tps = statistics.mean(tps) if tps else 0.0
        A(f"| {t} | {runs} | {errs} | {statistics.mean(lat) if lat else 0:.1f} | "
          f"{statistics.median(lat) if lat else 0:.1f} | "
          f"{pct(lat, 0.9):.1f} | {avg_tps:.2f} |")

    # per-task quality
    tasks = sorted({r["task"] for r in rows})
    A("\n### Per-task quality (all prompts)\n")
    A("| task | " + " | ".join(TIER_ORDER) + " |")
    A("|---|" + "---|" * len(TIER_ORDER))
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for t, e in (r.get("tiers") or {}).items():
            per[r["task"]][t].append(e.get("quality", 0.0))
    for task in tasks:
        A(f"| {task} | " + " | ".join(
            f"{sum(per[task][t])/len(per[task][t]):.3f}" if per[task].get(t) else "n/a" for t in TIER_ORDER) + " |")

    # oracle distribution
    A("\n### Oracle tier distribution\n")
    A("| tier | count | share |")
    A("|---|---|---|")
    dist = collections.Counter(r.get("oracle") for r in rows)
    for t in TIER_ORDER:
        n = dist.get(t, 0)
        A(f"| {t} | {n} | {n/max(1,len(rows)):.3f} |")

    # per-task oracle distribution: which tier is worth paying for on which task
    A("\n### Oracle tier distribution per task\n")
    A("| task | " + " | ".join(TIER_ORDER) + " |")
    A("|---|" + "---|" * len(TIER_ORDER))
    per_task_oracle = collections.defaultdict(collections.Counter)
    for r in rows:
        per_task_oracle[r["task"]][r.get("oracle")] += 1
    for task in sorted(per_task_oracle):
        c = per_task_oracle[task]
        tot = sum(c.values())
        A(f"| {task} | " + " | ".join(f"{c.get(t,0)} ({c.get(t,0)/max(1,tot):.2f})" for t in TIER_ORDER) + " |")

    # router comparison
    from router import heuristic, cascade
    preds = {}
    pp = BASE / "router" / "learned_preds.jsonl"
    if pp.exists():
        preds = {json.loads(l)["id"]: json.loads(l)["pred"]
                 for l in pp.read_text(encoding="utf-8").splitlines() if l.strip()}

    def artifact(r, tier):
        e = (r.get("tiers") or {}).get(tier) or {}
        return e.get("code") if "code" in e else (e.get("answer") or "")

    def eval_policy(chooser, acc):
        costs, qs = [], []
        for r in test:
            if acc:
                t, cost = chooser(r)
            else:
                t = chooser(r)
                cost = (r.get("tiers", {}).get(t) or {}).get("latency_ms", 0)
            costs.append(cost)
            qs.append((r.get("tiers", {}).get(t) or {}).get("quality", 0.0))
        return sum(costs) / max(1, len(costs)), sum(qs) / max(1, len(qs))

    def p_cascade(r):
        tiers = r.get("tiers", {})
        conf = {t: (tiers.get(t) or {}).get("confidence", 0.0) for t in TIER_ORDER}
        chosen = cascade.cascade(conf)
        total = 0.0
        for t in TIER_ORDER:
            total += (tiers.get(t) or {}).get("latency_ms", 0)
            if t == chosen:
                break
        return chosen, total

    def p_cascade_verified(r):
        tiers = r.get("tiers", {})
        contents = {t: artifact(r, t) for t in TIER_ORDER}
        lat = {t: (tiers.get(t) or {}).get("latency_ms", 0) for t in TIER_ORDER}
        return cascade.cascade_verified(r.get("task", ""), contents, lat, r.get("fn"), r.get("tests"))

    import random as _r
    _r.seed(0)
    pts = {
        "random": eval_policy(lambda r: TIER_ORDER[int(_r.random() * 3) % 3], False),
        "heuristic": eval_policy(lambda r: heuristic.route(r.get("prompt", "")), False),
        "cascade (confidence gate)": eval_policy(p_cascade, True),
        "cascade (verified gate)": eval_policy(p_cascade_verified, True),
        "learned": eval_policy(lambda r: preds.get(r.get("id"), "turbo"), False) if preds else (0.0, 0.0),
        "oracle": eval_policy(lambda r: r["oracle"], False),
    }
    q_floor, q_ceil = pts["random"][1], pts["oracle"][1]
    A("\n### Router comparison (held-out test split)\n")
    A("| policy | avg cost (ms) | quality | oracle-gap captured (%) |")
    A("|---|---|---|---|")
    for k, (c, q) in pts.items():
        cap = 0.0 if k == "random" else (100.0 if k == "oracle" else
               (100.0 * (q - q_floor) / (q_ceil - q_floor) if q_ceil > q_floor else 0.0))
        A(f"| {k} | {c:.1f} | {q:.3f} | {cap:.1f}% |")

    # budget curve
    try:
        from router import budget_router
        curve, lat, note = budget_router.curve(rows)
        A(f"\n### Cost-aware learned router: quality/cost frontier ({note})\n")
        A("| cost penalty lambda | avg cost (ms) | quality | tier mix |")
        A("|---|---|---|---|")
        for p in curve:
            mix = ", ".join(f"{t}:{p['mix'][t]}" for t in TIER_ORDER)
            A(f"| {p['lambda']:.2f} | {p['cost_ms']:.1f} | {p['quality']:.3f} | {mix} |")
    except Exception as e:
        A(f"\n(budget curve unavailable: {e})\n")

    # ablation
    try:
        from router.learned import ablation
        A("\n### Router feature ablation (held-out split)\n")
        A("| feature subset | agreement | avg cost (ms) | quality | capture (%) |")
        A("|---|---|---|---|---|")
        for row in ablation(args.scores):
            A(f"| {row['subset']} | {row['agreement']:.3f} | {row['cost_ms']:.1f} | "
              f"{row['quality']:.3f} | {row['capture']:.1f}% |")
    except Exception as e:
        A(f"\n(ablation unavailable: {e})\n")

    text = "\n".join(out)
    pathlib.Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
