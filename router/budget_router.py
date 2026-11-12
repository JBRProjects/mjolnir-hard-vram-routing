"""Cost-aware learned router + quality/cost Pareto curve (SPEC section 4).

SPEC section 4 asks for both objective framings:

  (a) maximise quality subject to a latency/memory budget,
  (b) minimise expected cost subject to a quality floor.

This module implements the cost-aware version of (a) as a *curve* instead of a
single operating point. Instead of classifying "which tier is the oracle tier"
(a formulation that degenerates when the oracle itself is 87% one tier), it
models the probability that a given tier answers a given query correctly:

    P(correct | x, t)  ~  logistic( w . [phi(x), onehot(t)] )

and then routes with a cost penalty:

    r_lambda(x) = argmax_t  [ P(correct | x, t)  -  lambda * latency_t ]

where `latency_t` is the MEASURED mean latency of tier t (from the oracle data,
not a simulation). Sweeping lambda traces the quality-vs-cost Pareto frontier:
lambda = 0 gives the quality-maximising router, large lambda collapses to the
cheapest tier.

Everything is evaluated OFF-POLICY on the already-collected oracle data: the
reported cost is the measured latency of the tier the router picks and the
reported quality is the measured quality of that tier's output.

Usage:
  python router/budget_router.py --scores scoring/scores.jsonl [--lams 0,0.1,0.3,1]
"""
import argparse, json, pathlib, re, sys

BASE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
from router.learned import featurize

TIER_ORDER = ["turbo", "vector", "ultra"]
CORRECT_AT = 0.5      # a code answer passing half its unit tests counts as correct


def load(scores_path):
    return [json.loads(l) for l in pathlib.Path(scores_path).read_text(encoding="utf-8").splitlines() if l.strip()]


def prompt_type(prompt):
    """Three cheap prompt-type indicators (no labels, no model call).

    The tier advantage is strongly task dependent (ultra wins reasoning, turbo
    wins code), so the pair model needs the interaction between prompt type and
    tier; a plain [phi(x), onehot(t)] model cannot represent it.
    """
    p = (prompt or "").lower()
    code = 1.0 if re.search(r"\b(def|function|class|return|python|code|import)\b", p) else 0.0
    reason = 1.0 if re.search(r"\b(why|explain|prove|reasoning|steps|because|therefore)\b", p) else 0.0
    qa = 1.0 if re.search(r"\b(what|who|when|where|which|capital|how many)\b", p) else 0.0
    return code, reason, qa


def pair_features(prompt, tier):
    """Prompt features + tier one-hot + (prompt-type x tier) interactions."""
    f = list(featurize(prompt))
    onehot = [1.0 if tier == t else 0.0 for t in TIER_ORDER]
    f += onehot
    types = prompt_type(prompt)
    for oh in onehot:
        for tp in types:
            f.append(oh * tp)
    return f


def tier_latency(rows):
    acc = {t: [] for t in TIER_ORDER}
    for r in rows:
        for t in TIER_ORDER:
            e = (r.get("tiers") or {}).get(t) or {}
            if e.get("latency_ms"):
                acc[t].append(e["latency_ms"])
    return {t: (sum(v) / len(v) if v else float("inf")) for t, v in acc.items()}


def fit_pair_model(scores):
    """Train P(correct | x, t) on the train split."""
    X, y = [], []
    for r in scores:
        if r.get("split") != "train":
            continue
        for t in TIER_ORDER:
            e = (r.get("tiers") or {}).get(t) or {}
            if e.get("error"):
                continue
            X.append(pair_features(r.get("prompt", ""), t))
            y.append(1 if e.get("quality", 0.0) >= CORRECT_AT else 0)
    if not X:
        raise SystemExit("no train rows")
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        def prob(prompt, tier):
            return float(clf.predict_proba([pair_features(prompt, tier)])[0][1])
        return prob, f"sklearn-logreg (n={len(X)})"
    except ImportError:
        base = sum(y) / len(y)
        def prob(prompt, tier):
            return base
        return prob, f"constant({base:.2f}) (sklearn missing)"


def evaluate_on(rows, prob, lat, lams):
    """Evaluate the cost-aware rule on an arbitrary set of rows."""
    unit = min(v for v in lat.values() if v != float("inf")) or 1.0
    out = []
    for lam in lams:
        costs, qs, picks = [], [], []
        for r in rows:
            best_t, best_v = None, None
            for t in TIER_ORDER:
                v = prob(r.get("prompt", ""), t) - lam * (lat.get(t, 0.0) / unit)
                if best_v is None or v > best_v:
                    best_t, best_v = t, v
            e = (r.get("tiers") or {}).get(best_t) or {}
            costs.append(e.get("latency_ms", 0))
            qs.append(e.get("quality", 0.0))
            picks.append(best_t)
        out.append({"lambda": lam,
                    "cost_ms": sum(costs) / max(1, len(costs)),
                    "quality": sum(qs) / max(1, len(qs)),
                    "mix": {t: picks.count(t) for t in TIER_ORDER}})
    return out


def curve(scores, lams=None):
    train = [r for r in scores if r.get("split") == "train"]
    test = [r for r in scores if r.get("split") != "train"] or scores
    prob, note = fit_pair_model(scores)
    lat = tier_latency(train or scores)
    if lams is None:
        lams = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    return evaluate_on(test, prob, lat, lams), lat, note


def select_lambda(scores, budget_factor=1.2, lams=None):
    """Pick the cost penalty on the TRAIN split for a latency budget.

    The budget is `budget_factor` times the average cost of the oracle on the
    training split: "as much as the hindsight-optimal router, plus a margin".
    Among the penalties whose TRAIN cost stays inside the budget, the one with
    the highest train quality is chosen; the penalty is then applied to the
    held-out split. (The budget and the penalty are therefore never chosen on
    the data they are reported on.)

    Returns (lambda, budget_ms, train_cost_ms, train_quality, note).
    """
    train = [r for r in scores if r.get("split") == "train"] or scores
    prob, note = fit_pair_model(scores)
    lat = tier_latency(train)
    if lams is None:
        lams = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    oracle_cost = sum((r.get("tiers", {}).get(r["oracle"]) or {}).get("latency_ms", 0)
                      for r in train) / max(1, len(train))
    budget = budget_factor * oracle_cost
    tr_pts = evaluate_on(train, prob, lat, lams)
    inside = [p for p in tr_pts if p["cost_ms"] <= budget]
    best = max(inside, key=lambda p: (p["quality"], -p["cost_ms"])) if inside else \
        max(tr_pts, key=lambda p: p["quality"])
    return best["lambda"], budget, best["cost_ms"], best["quality"], note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(BASE / "scoring" / "scores.jsonl"))
    ap.add_argument("--lams", default="", help="comma separated cost penalties")
    ap.add_argument("--out", default=str(BASE / "router" / "budget_curve.json"))
    args = ap.parse_args()
    sp = pathlib.Path(args.scores)
    if not sp.exists():
        print("no scores - run harness + scoring first"); return
    lams = [float(x) for x in args.lams.split(",") if x.strip()] if args.lams else None
    scores = load(sp)
    pts, lat, note = curve(scores, lams)
    print(f"cost-aware learned router ({note}); measured mean latency per tier (ms): "
          + ", ".join(f"{t}={lat[t]:.0f}" for t in TIER_ORDER))
    for p in pts:
        mix = " ".join(f"{t}:{p['mix'][t]}" for t in TIER_ORDER)
        print(f"  lambda={p['lambda']:5.2f} -> cost={p['cost_ms']:9.1f}ms quality={p['quality']:.3f}  [{mix}]")
    pathlib.Path(args.out).write_text(json.dumps(pts, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
