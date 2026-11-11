"""Tier characterization table from live oracle data (RESULTS.md material).

Reads a merged oracle file and reports, per tier: success/failure counts,
latency distribution, throughput (from the engine's own timings) and the
per-task quality under the metric stored in the scores file.

Usage:
  python analysis/tier_table.py --in harness/oracle_v2.jsonl [--scores scoring/scores.jsonl]
"""
import argparse, json, pathlib, statistics, sys

BASE = pathlib.Path(__file__).parent.parent
TIER_ORDER = ["turbo", "vector", "ultra"]


def load(path):
    return [json.loads(l) for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def pct(vals, q):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(BASE / "harness" / "oracle_v2.jsonl"))
    ap.add_argument("--scores", default="", help="optional scores file for the per-task quality columns")
    args = ap.parse_args()
    if not pathlib.Path(args.inp).exists():
        print(f"no oracle file at {args.inp}"); return
    rows = load(args.inp)
    print(f"prompts: {len(rows)}")

    q = {}
    if args.scores and pathlib.Path(args.scores).exists():
        srows = {r["id"]: r for r in load(args.scores)}
        for r in rows:
            sr = srows.get(r["id"])
            if not sr:
                continue
            for t, e in (sr.get("tiers") or {}).items():
                q.setdefault((t, sr["task"]), []).append(e.get("quality", 0.0))

    tasks = sorted({r["task"] for r in rows})
    print("\n| tier | runs | errors | avg latency (s) | median (s) | p90 (s) | avg tok/s | "
          + " | ".join(f"quality {t}" for t in tasks) + " |")
    print("|---|---|---|---|---|---|---|" + "---|" * len(tasks))
    for t in TIER_ORDER:
        lat, tps, errs, runs = [], [], 0, 0
        for r in rows:
            e = (r.get("tiers") or {}).get(t)
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
        cols = []
        for task in tasks:
            v = q.get((t, task), [])
            cols.append(f"{sum(v)/len(v):.3f}" if v else "n/a")
        print(f"| {t} | {runs} | {errs} | {statistics.mean(lat):.1f} | {statistics.median(lat):.1f} | "
              f"{pct(lat, 0.9):.1f} | {statistics.mean(tps):.2f} | " + " | ".join(cols) + " |")


if __name__ == "__main__":
    main()
