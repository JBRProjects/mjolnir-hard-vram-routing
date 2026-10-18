"""Learned router (SPEC section 6.3): tiny classifier over cheap prompt features.
Features (all <1ms, no model call): char len, word len, code-keyword count,
reason-keyword count, question-word count, digit ratio, bracket ratio.
Model: sklearn LogisticRegression if available, else pure-python majority fallback.
Trains on oracle labels from scoring/scores.jsonl (split==train), evals on test,
writes router/learned_preds.jsonl consumed by analysis/make_plot.py.
Usage: python router/learned.py [--scores scoring/scores.jsonl]
"""
import json, re, pathlib, sys, argparse

BASE = pathlib.Path(__file__).parent.parent
PROMPTS = BASE / "eval_prompts" / "prompts.jsonl"

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(BASE / "scoring" / "scores.jsonl"))
    args = ap.parse_args()
    sp = pathlib.Path(args.scores)
    if not sp.exists():
        print("no scores — run harness + scoring first"); return
    Xtr, ytr, Xte, yte, ids = load(sp)
    if not Xtr:
        print("no training data — oracle labels missing?"); return
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=500)
        clf.fit(Xtr, ytr)
        pred = list(clf.predict(Xte)) if Xte else []
        note = "sklearn-logreg"
    except ImportError:
        from collections import Counter
        maj = Counter(ytr).most_common(1)[0][0]
        pred = [maj] * len(Xte)
        note = f"majority('{maj}') (sklearn missing)"
    acc = sum(a == b for a, b in zip(pred, yte)) / max(1, len(yte))
    print(f"learned ({note}): trained on {len(Xtr)}, test {len(Xte)}, acc={acc:.3f}")
    out = pathlib.Path(__file__).parent / "learned_preds.jsonl"
    out.write_text("\n".join(json.dumps({"id": i, "pred": p, "oracle": o}) for i, p, o in zip(ids, pred, yte)), encoding="utf-8")
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
