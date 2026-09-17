"""Learned router (SPEC section 6.3): tiny classifier over cheap prompt features.
Features (all <1ms, no model call): char len, word len, code-keyword count,
reason-keyword count, question-word count, digit ratio, bracket ratio.
Model: sklearn LogisticRegression if available, else pure-python majority fallback.
Trains on oracle labels from scoring/scores.jsonl (split==train), evals on test.
Usage: python router/learned.py [--scores scoring/scores.jsonl]
"""
import json, re, pathlib, sys

PROMPTS = pathlib.Path(__file__).parent.parent / "eval_prompts" / "prompts.jsonl"

def featurize(prompt: str):
    p = prompt or ""
    pl = p.lower()
    code_kw = len(re.findall(r"(def |class |import |;|\{|\}|```|return|for |while )", pl))
    reason_kw = len(re.findall(r"\b(why|prove|explain|steps|reason|therefore|because|if|then)\b", pl))
    q_kw = len(re.findall(r"\b(what|who|when|where|which|how many)\b", pl))
    digits = sum(c.isdigit() for c in p) / max(1, len(p))
    brackets = sum(c in "()[]{}" for c in p) / max(1, len(p))
    return [len(p), len(p.split()), code_kw, reason_kw, q_kw, digits, brackets]

def load():
    sp = pathlib.Path(sys.argv[sys.argv.index("--scores") + 1]) if "--scores" in sys.argv else pathlib.Path(__file__).parent.parent / "scoring" / "scores.jsonl"
    scores = {json.loads(l)["id"]: json.loads(l) for l in sp.read_text(encoding="utf-8").splitlines() if l.strip()}
    prompts = {json.loads(l)["id"]: json.loads(l) for l in PROMPTS.read_text(encoding="utf-8").splitlines() if l.strip()} if PROMPTS.exists() else {}
    Xtr, ytr, Xte, yte, ids_te = [], [], [], [], []
    for pid, s in scores.items():
        pr = prompts.get(pid, {}).get("prompt", pid)
        f = featurize(pr)
        if s.get("split") == "train":
            Xtr.append(f); ytr.append(s["oracle"])
        else:
            Xte.append(f); yte.append(s["oracle"]); ids_te.append(pid)
    return Xtr, ytr, Xte, yte, ids_te

def main():
    Xtr, ytr, Xte, yte, ids = load()
    if not Xtr:
        print("no training data — run harness + scoring first"); return
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=500)
        clf.fit(Xtr, ytr)
        pred = list(clf.predict(Xte)) if Xte else []
        print(f"learned: trained on {len(Xtr)}, test {len(Xte)}, acc={sum(a==b for a,b in zip(pred,yte))/max(1,len(yte)):.3f}")
    except ImportError:
        from collections import Counter
        maj = Counter(ytr).most_common(1)[0][0]
        pred = [maj] * len(Xte)
        print(f"sklearn missing — majority fallback '{maj}' acc={sum(a==b for a,b in zip(pred,yte))/max(1,len(yte)):.3f}")
    out = pathlib.Path(__file__).parent / "learned_preds.jsonl"
    out.write_text("\n".join(json.dumps({"id": i, "pred": p, "oracle": o}) for i, p, o in zip(ids, pred, yte)), encoding="utf-8")
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
