"""Scoring: primary = task-specific correctness; secondary = judge (ultra as judge, optional).
- factual_qa: normalized exact-substring match of reference in output.
- code_gen: python syntax check (ast.parse) + reference keyword overlap; execution pass/fail left to human review (see notes).
- reasoning: keyword 'Answer:' presence + reference match if reference non-empty else length/non-empty heuristic (weak — judge needed for real paper).
- judge: optional (--judge-url, e.g. http://127.0.0.1:8082) — ultra tier judged YES/NO correctness
  for prompts where tiers disagree on the primary metric (SPEC section 5). Connection failures
  degrade to judge=null and never abort scoring.
Also passes `prompt` through to output rows and records cascade `confidence` per tier so
router/analysis code can evaluate policies without re-reading raw model outputs.
Usage: python scoring/score.py --in harness/oracle.jsonl --out scoring/scores.jsonl
       python scoring/score.py --in harness/oracle.jsonl --out scoring/scores.jsonl --judge-url http://127.0.0.1:8082 --judge-max 60
"""
import argparse, json, pathlib, ast, re, sys
BASE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
from router.cascade import confidence_proxy

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

JUDGE_TMPL = ("You are a strict grader. Question: {q}\nExpected answer: {ref}\n"
              "Candidate answer: {cand}\nDoes the candidate answer correctly and completely "
              "relative to the expected answer? Reply with exactly YES or NO.")

def judge_once(url, prompt_text, timeout=60):
    """One ultra-as-judge call. Returns 'yes'/'no', or None on any failure."""
    try:
        import requests
    except ImportError:
        return None
    try:
        r = requests.post(url.rstrip("/") + "/completion", json={
            "prompt": prompt_text, "n_predict": 4, "temperature": 0.0,
            "stream": False, "cache_prompt": False,
        }, timeout=timeout)
        if r.status_code != 200:
            return None
        reply = r.json().get("content", "").strip().upper()
        if reply.startswith("YES"): return "yes"
        if reply.startswith("NO"): return "no"
        return None
    except Exception as e:
        print(f"[judge] call failed: {e}", file=sys.stderr)
        return None

def run_judge(rows, url, max_calls):
    """Judge score in {0.0,1.0} for tiers on prompts where the primary metric
    disagrees (any tier at 0 next to any tier above 0). factual_qa only."""
    from collections import defaultdict
    n = 0
    out = defaultdict(dict)   # id -> {tier: 0.0|1.0}
    for r in rows:
        if n >= max_calls: break
        if r.get("task") != "factual_qa": continue
        tiers = r.get("tiers", [])
        tops = [t for t, v in tiers.items() if v.get("quality", 0) > 0]
        zeros = [t for t, v in tiers.items() if v.get("quality", 0) == 0 and not v.get("error")]
        if not (tops and zeros):
            continue
        for t in zeros:
            content = tiers[t].get("content", "")
            if not content.strip(): continue
            reply = judge_once(url, JUDGE_TMPL.format(q=r.get("prompt", ""), ref=r.get("reference", ""), cand=content))
            judged = 1.0 if reply == "yes" else 0.0
            out[r["id"]][t] = judged
            n += 1
            if n >= max_calls: break
    return dict(out), n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(BASE / "harness" / "oracle.jsonl"))
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "scores.jsonl"))
    ap.add_argument("--judge-url", default="", help="tier base URL to use as judge (e.g. http://127.0.0.1:8082); empty = disabled")
    ap.add_argument("--judge-max", type=int, default=60, help="cap on judge calls (latency budget)")
    args = ap.parse_args()
    rows = [json.loads(l) for l in pathlib.Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    judged = {}
    if args.judge_url:
        judged, n_calls = run_judge(rows, args.judge_url, args.judge_max)
        print(f"[judge] {n_calls} calls completed ({len(judged)} prompts judged)")
    out = []
    for r in rows:
        scored = {"id": r["id"], "task": r["task"], "split": r.get("split", ""),
                  "prompt": r.get("prompt", ""), "tiers": {}}
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
            entry = {"quality": q, "latency_ms": t.get("latency_ms", 0), "error": err,
                     "confidence": 0.0 if err else round(confidence_proxy(content), 4)}
            # judge overrides only where recorded (secondary metric; primary kept untouched)
            if args.judge_url and r["id"] in judged and tier in judged[r["id"]]:
                entry["judge"] = judged[r["id"]][tier]
            scored["tiers"][tier] = entry
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
