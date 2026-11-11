"""Scoring: task-specific correctness (primary) + judge (ultra as judge, optional).

METRIC v2 (default) — format-neutral, capability-revealing (cycle 2):
  * factual_qa: extract the model's final answer (scoring/extract.py), compare it
    to the gold answer semantically (normalization + numeric equality + short
    containment). No longer "does the reference string appear anywhere".
  * code_gen: EXECUTE the candidate code and run the task's unit tests
    (scoring/exec_tests.py). Correctness is measured by behaviour, not by
    identifier overlap with the reference.
  * reasoning: extract the final answer and compare it to the gold answer.
  * every task: the model's own conversation continuation ("Human: ...") is
    trimmed first (scoring/extract.py:trim_continuation).

METRIC legacy (`--metric legacy`) keeps the frozen cycle-1 metric so the two
metrics can be compared on the same oracle data (see RESULTS.md cycle 2).

Optional judge scoring (--judge-url) grades the tiers that DISAGREE under the
primary metric with ultra acting as the judge; connection failures degrade to
judge=null and never abort scoring.

Usage:
  python scoring/score.py --in harness/oracle.jsonl --out scoring/scores.jsonl
  python scoring/score.py --in harness/oracle.jsonl --out scoring/scores.jsonl \
      --judge-url http://127.0.0.1:8082 --judge-max 60
"""
import argparse, json, pathlib, ast, re, sys

BASE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
from router.cascade import confidence_proxy
from scoring.extract import (answer_match, extract_answer, extract_code,
                              normalize_answer, repair_code, trim_continuation)
from scoring.exec_tests import score_code_exec


# --------------------------------------------------------------- legacy (v1)
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


# ------------------------------------------------------------------- metric v2
def _gold_in_text(gold: str, text: str) -> bool:
    """Lenient fallback: the gold answer appears in the trimmed output.

    Only used for short, non-numeric golds (a stray "2" anywhere in an answer
    would otherwise be a false positive).
    """
    if not gold or not text:
        return False
    gt = [t for t in normalize_answer(gold).split() if t]
    if not gt or len(gt) > 3:
        return False
    if any(t.replace(".", "", 1).isdigit() for t in gt):
        return False
    tt = set(normalize_answer(text).split())
    return all(t in tt for t in gt)


def v2_factual(output, reference):
    if not reference:
        return 0.0
    text = trim_continuation(output)
    ans = extract_answer(text)
    if answer_match(ans, reference):
        return 1.0
    if _gold_in_text(reference, text):
        return 1.0
    return 0.0


def v2_reasoning(output, reference):
    if not reference:
        return 0.0
    text = trim_continuation(output)
    ans = extract_answer(text)
    if answer_match(ans, reference):
        return 1.0
    if _gold_in_text(reference, text):
        return 1.0
    return 0.0


def v2_code(output, reference, fn=None, tests=None):
    """Execution-based correctness. Falls back to the legacy metric when the
    task carries no unit tests (e.g. cycle-1 oracle data)."""
    if not tests:
        return score_code(output, reference), None
    code = repair_code(extract_code(output), fn)
    if not code.strip():
        return 0.0, {"error": "no code extracted"}
    from scoring.exec_tests import run_tests
    r = run_tests(code, fn, tests)
    score = r["passed"] / r["total"] if r.get("total") else 0.0
    detail = {"tests_passed": f"{r['passed']}/{r['total']}"}
    if r.get("error"):
        detail["error"] = r["error"]
    return score, detail


# ------------------------------------------------------------------ judge (v2)
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


def run_judge(rows, url, max_calls, tasks=None):
    """Judge score in {0.0,1.0} for tiers on prompts where the primary metric
    disagrees (any tier above 0 next to any tier at 0)."""
    from collections import defaultdict
    n = 0
    out = defaultdict(dict)   # id -> {tier: 0.0|1.0}
    for r in rows:
        if n >= max_calls: break
        if tasks and r.get("task") not in tasks: continue
        tiers = r.get("tiers", [])
        tops = [t for t, v in tiers.items() if v.get("quality", 0) > 0]
        zeros = [t for t, v in tiers.items() if v.get("quality", 0) == 0 and not v.get("error")]
        if not (tops and zeros):
            continue
        for t in zeros:
            content = tiers[t].get("content", "")
            if not content.strip(): continue
            reply = judge_once(url, JUDGE_TMPL.format(q=r.get("prompt", ""), ref=r.get("reference", ""),
                                                      cand=trim_continuation(content)[:1500]))
            judged = 1.0 if reply == "yes" else 0.0
            out[r["id"]][t] = judged
            n += 1
            if n >= max_calls: break
    return dict(out), n


def apply_gold(rows, gold_path):
    """Attach gold answers / unit tests to rows by longest-prefix match.

    Used to re-score cycle-1 data (20 distinct base items, no gold answers for
    reasoning, no unit tests for code) with the metric v2 scorer.
    """
    gold = json.loads(pathlib.Path(gold_path).read_text(encoding="utf-8"))
    keys = sorted(gold.keys(), key=len, reverse=True)
    n = 0
    for r in rows:
        p = (r.get("prompt") or "").strip()
        for k in keys:
            if p.startswith(k):
                g = gold[k]
                r["reference"] = g.get("reference", r.get("reference", ""))
                if g.get("fn"): r["fn"] = g["fn"]
                if g.get("tests"): r["tests"] = g["tests"]
                n += 1
                break
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(BASE / "harness" / "oracle.jsonl"))
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "scores.jsonl"))
    ap.add_argument("--metric", choices=["v2", "legacy"], default="v2",
                    help="v2 = format-neutral execution/semantic metric (default); legacy = frozen cycle-1 metric")
    ap.add_argument("--gold", default="", help="gold lookup (see eval_prompts/build_legacy_gold.py) applied by longest-prefix match")
    ap.add_argument("--judge-url", default="", help="tier base URL to use as judge (e.g. http://127.0.0.1:8082); empty = disabled")
    ap.add_argument("--judge-max", type=int, default=60, help="cap on judge calls (latency budget)")
    ap.add_argument("--judge-tasks", nargs="*", default=None, help="restrict judging to these task types")
    args = ap.parse_args()
    rows = [json.loads(l) for l in pathlib.Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.gold:
        n = apply_gold(rows, args.gold)
        print(f"[gold] attached gold to {n}/{len(rows)} rows from {args.gold}")
    judged = {}
    if args.judge_url:
        judged, n_calls = run_judge(rows, args.judge_url, args.judge_max, args.judge_tasks)
        print(f"[judge] {n_calls} calls completed ({len(judged)} prompts judged)")
    out = []
    for r in rows:
        scored = {"id": r["id"], "task": r["task"], "split": r.get("split", ""),
                  "prompt": r.get("prompt", ""), "metric": args.metric,
                  "fn": r.get("fn"), "tests": r.get("tests"), "tiers": {}}
        for tier, t in r.get("tiers", {}).items():
            content = t.get("content", "")
            err = t.get("error")
            detail = None
            if err:
                q = 0.0
            elif args.metric == "legacy":
                if r["task"] == "factual_qa":
                    q = score_factual(content, r.get("reference", ""))
                elif r["task"] == "code_gen":
                    q = score_code(content, r.get("reference", ""))
                else:
                    q = score_reasoning(content, r.get("reference", ""))
            else:
                if r["task"] == "factual_qa":
                    q = v2_factual(content, r.get("reference", ""))
                elif r["task"] == "code_gen":
                    q, detail = v2_code(content, r.get("reference", ""), r.get("fn"), r.get("tests"))
                else:
                    q = v2_reasoning(content, r.get("reference", ""))
            entry = {"quality": q, "latency_ms": t.get("latency_ms", 0), "error": err,
                     "confidence": 0.0 if err else round(confidence_proxy(content), 4)}
            if detail:
                entry["detail"] = detail
            if args.metric != "legacy":
                # carry the extracted artefacts so the routers/analysis can run the
                # verifier (unit tests for code, answer presence for QA) without
                # re-reading the raw oracle file.
                if r["task"] == "code_gen":
                    entry["code"] = extract_code(content)
                else:
                    entry["answer"] = extract_answer(content)[:400]
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
    print(f"wrote {args.out} ({len(out)} prompts, metric={args.metric})")


if __name__ == "__main__":
    main()
