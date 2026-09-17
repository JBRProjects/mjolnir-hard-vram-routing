"""Scoring: primary = task-specific correctness; secondary = judge placeholder.
- factual_qa: normalized exact-substring match of reference in output.
- code_gen: python syntax check (ast.parse) + reference keyword overlap; execution pass/fail left to human review (see notes).
- reasoning: keyword 'Answer:' presence + reference match if reference non-empty else length/non-empty heuristic (weak — judge needed for real paper).
- judge: stub — set scores.judge = None unless --judge-tier URL provided (uses ultra tier as judge per SPEC section 5).
Usage: python scoring/score.py --in harness/oracle.jsonl --out scoring/scores.jsonl
"""
import argparse, json, pathlib, ast, re

def norm(s): return re.sub(r"\s+", " ", (s or "").strip().lower())

def score_factual(output, reference):
    if not reference: return 0.0
    return 1.0 if norm(reference) in norm(output) else 0.0

def score_code(output, reference):
    # extract code block if fenced
    m = re.search(r"```(?:python)?\s*(.*?)```", output, re.S)
    code = m.group(1) if m else output
    pts = 0.0
    try:
        ast.parse(code)
        pts += 0.5
    except Exception:
        pass
    # keyword overlap with reference (function name etc.)
    ref_words = set(re.findall(r"[a-z_]+", norm(reference))) - {"def", "return", "for", "in", "if"}
    out_words = set(re.findall(r"[a-z_]+", norm(code)))
    if ref_words and len(ref_words & out_words) / max(1, len(ref_words)) >= 0.5:
        pts += 0.5
    return pts

def score_reasoning(output, reference):
    if reference and norm(reference) in norm(output):
        return 1.0
    # weak heuristic: has final Answer line and non-trivial length
    has_answer = bool(re.search(r"^answer\s*:", output.strip().lower(), re.M))
    return 0.75 if has_answer and len(output.split()) > 20 else (0.25 if len(output.split()) > 5 else 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(pathlib.Path(__file__).parent.parent / "harness" / "oracle.jsonl"))
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "scores.jsonl"))
    args = ap.parse_args()
    rows = [json.loads(l) for l in pathlib.Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for r in rows:
        scored = {"id": r["id"], "task": r["task"], "split": r.get("split", ""), "tiers": {}}
        for tier, t in r.get("tiers", {}).items():
            content = t.get("content", "")
            err = t.get("error")
            if err:
                q = 0.0
            elif r["task"] == "factual_qa":
                q = score_factual(content, r.get("reference", ""))
            elif r["task"] == "code_gen":
                q = score_code(content, r.get("reference", ""))
            else:
                q = score_reasoning(content, r.get("reference", ""))
            scored["tiers"][tier] = {"quality": q, "latency_ms": t.get("latency_ms", 0), "error": err}
        # oracle label: cheapest tier achieving max quality (ties -> turbo > vector > ultra)
        order = ["turbo", "vector", "ultra"]
        best_q = max((v["quality"] for v in scored["tiers"].values()), default=0.0)
        oracle = next((t for t in order if scored["tiers"].get(t, {}).get("quality", -1) == best_q), "turbo")
        scored["oracle"] = oracle
        scored["best_quality"] = best_q
        out.append(scored)
    pathlib.Path(args.out).write_text("\n".join(json.dumps(o) for o in out), encoding="utf-8")
    print(f"wrote {args.out} ({len(out)} prompts)")

if __name__ == "__main__":
    main()
