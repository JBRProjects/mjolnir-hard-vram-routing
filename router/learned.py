"""Learned router (SPEC section 6.3): tiny classifier over cheap prompt features.

Features (all <1ms, no model call): char len, word len, code-keyword count,
reason-keyword count, question-word count, digit ratio, bracket ratio.
Model: sklearn LogisticRegression if available, else pure-python majority fallback.
Trains on oracle labels from scoring/scores.jsonl (split==train), evals on test,
writes router/learned_preds.jsonl consumed by analysis/make_plot.py.

`--ablation` additionally reports the router's behaviour for several cheap
feature subsets (SPEC section 8 asks for a feature-choice ablation): oracle
agreement, average cost, average quality and captured oracle gap per subset.
Usage: python router/learned.py [--scores scoring/scores.jsonl] [--ablation]
"""
import json, re, pathlib, sys, argparse

BASE = pathlib.Path(__file__).parent.parent
PROMPTS = BASE / "eval_prompts" / "prompts.jsonl"

FEATURE_NAMES = ["char_len", "word_len", "code_kw", "reason_kw", "question_kw",
                 "digit_ratio", "bracket_ratio"]

# subsets used by the ablation (indices into FEATURE_NAMES)
ABLATION_SETS = {
    "length only": [0, 1],
    "task keywords only": [2, 3, 4],
    "surface ratios only": [5, 6],
    "all features": [0, 1, 2, 3, 4, 5, 6],
}


def featurize(prompt: str):
    p = prompt or ""
    pl = p.lower()
    code_kw = len(re.findall(r"(def |class |import |;|\{|\}|```|return|for |while )", pl))
    reason_kw = len(re.findall(r"\b(why|prove|explain|steps|reason|therefore|because|if|then)\b", pl))
    q_kw = len(re.findall(r"\b(what|who|when|where|which|how many)\b", pl))
    digits = sum(c.isdigit() for c in p) / max(1, len(p))
    brackets = sum(c in "()[]{}" for c in p) / max(1, len(p))
    return [len(p), len(p.split()), code_kw, reason_kw, q_kw, digits, brackets]


def load(scores_path):
    scores = [json.loads(l) for l in scores_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    # scores.jsonl carries `prompt` through (score.py); fall back to prompts.jsonl for older files
    prompt_by_id = {}
    if any(not r.get("prompt") for r in scores) and PROMPTS.exists():
        prompt_by_id = {json.loads(l)["id"]: json.loads(l)["prompt"]
                        for l in PROMPTS.read_text(encoding="utf-8").splitlines() if l.strip()}
    Xtr, ytr, Xte, yte, ids_te = [], [], [], [], []
    for s in scores:
        f = featurize(s.get("prompt") or prompt_by_id.get(s["id"], ""))
        if s.get("split") == "train":
            Xtr.append(f); ytr.append(s["oracle"])
        else:
            Xte.append(f); yte.append(s["oracle"]); ids_te.append(s["id"])
    return Xtr, ytr, Xte, yte, ids_te


def _fit_predict(Xtr, ytr, Xte):
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=500)
        clf.fit(Xtr, ytr)
        return list(clf.predict(Xte)), "sklearn-logreg"
    except ImportError:
        from collections import Counter
        maj = Counter(ytr).most_common(1)[0][0]
        return [maj] * len(Xte), f"majority('{maj}') (sklearn missing)"


def ablation(scores_path):
    """Feature-subset ablation on the held-out split."""
    scores = [json.loads(l) for l in pathlib.Path(scores_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    test = [r for r in scores if r.get("split") != "train"] or scores
    # floor/ceiling follow analysis/make_plot.py: random = uniform over tiers,
    # oracle = the oracle tier per row.
    q_floor = 0.0
    for r in test:
        tiers = r.get("tiers", {})
        q_floor += sum((tiers.get(t) or {}).get("quality", 0.0) for t in ("turbo", "vector", "ultra")) / 3.0
    q_floor /= max(1, len(test))
    q_ceil = sum((r.get("tiers", {}).get(r["oracle"]) or {}).get("quality", 0.0) for r in test) / max(1, len(test))
    rows = []
    for name, idx in ABLATION_SETS.items():
        Xtr, ytr, Xte, yte, ids = [], [], [], [], []
        for s in scores:
            f = featurize(s.get("prompt", ""))
            sel = [f[i] for i in idx]
            if s.get("split") == "train":
                Xtr.append(sel); ytr.append(s["oracle"])
            else:
                Xte.append(sel); yte.append(s["oracle"]); ids.append(s["id"])
        if not Xtr or not Xte:
            continue
        pred, note = _fit_predict(Xtr, ytr, Xte)
        acc = sum(a == b for a, b in zip(pred, yte)) / max(1, len(yte))
        by_id = dict(zip(ids, pred))
        costs, qs = [], []
        for r in test:
            t = by_id.get(r["id"], "turbo")
            e = r.get("tiers", {}).get(t) or {}
            costs.append(e.get("latency_ms", 0))
            qs.append(e.get("quality", 0.0))
        cost = sum(costs) / max(1, len(costs))
        q = sum(qs) / max(1, len(qs))
        capture = 100.0 * (q - q_floor) / (q_ceil - q_floor) if q_ceil > q_floor else 0.0
        rows.append({"subset": name, "n_features": len(idx), "agreement": acc,
                     "cost_ms": cost, "quality": q, "capture": capture})
        print(f"  {name:22s} feats={len(idx)} agree={acc:.3f} cost={cost:8.1f}ms "
              f"quality={q:.3f} capture={capture:5.1f}%")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(BASE / "scoring" / "scores.jsonl"))
    ap.add_argument("--ablation", action="store_true", help="report the feature-subset ablation")
    args = ap.parse_args()
    sp = pathlib.Path(args.scores)
    if not sp.exists():
        print("no scores — run harness + scoring first"); return
    Xtr, ytr, Xte, yte, ids = load(sp)
    if not Xtr:
        print("no training data — oracle labels missing?"); return
    pred, note = _fit_predict(Xtr, ytr, Xte)
    acc = sum(a == b for a, b in zip(pred, yte)) / max(1, len(yte))
    print(f"learned ({note}): trained on {len(Xtr)}, test {len(Xte)}, acc={acc:.3f}")
    out = pathlib.Path(__file__).parent / "learned_preds.jsonl"
    out.write_text("\n".join(json.dumps({"id": i, "pred": p, "oracle": o}) for i, p, o in zip(ids, pred, yte)), encoding="utf-8")
    print(f"wrote {out}")
    if args.ablation:
        print("feature ablation (held-out split):")
        ablation(sp)


if __name__ == "__main__":
    main()
