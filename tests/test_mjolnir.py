"""Unit tests for MJOLNIR (run: python tests/test_mjolnir.py).

Covers the pieces that the paper's numbers depend on:
  * format-neutral extraction (trimming, answer extraction, normalization, matching)
  * execution-based code scoring (correct / wrong / prose / crash)
  * the v2 scorer end-to-end on synthetic rows
  * the routers (heuristic, confidence cascade, verified cascade)
  * the eval-set builder invariants (60 items per task, gold soundness, no leakage)
No third-party dependencies: runnable in CI without pytest.
"""
import json, pathlib, subprocess, sys, tempfile

BASE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from scoring.extract import (answer_match, extract_answer, extract_code,
                              normalize_answer, repair_code, trim_continuation)
from scoring.exec_tests import run_tests, score_code_exec
from scoring.score import v2_code, v2_factual, v2_reasoning
from router.heuristic import route
from router.cascade import cascade, cascade_verified, confidence_proxy, gate

FAILS = []


def check(name, cond, extra=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {extra}")
        FAILS.append(name)


def test_trim():
    print("trim_continuation")
    check("cuts Human: turn", trim_continuation(" Paris.Human: next question") == "Paris.")
    check("cuts newline Human:", trim_continuation("a\nHuman: b") == "a")
    check("keeps answer text", trim_continuation("Answer: 42") == "Answer: 42")
    check("cuts chat template end", trim_continuation("hi<|im_end|>tail") == "hi")
    check("lowercase q: is kept", "q: value" in trim_continuation("q: value"))


def test_extract_answer():
    print("extract_answer")
    check("bold answer", answer_match(extract_answer("**Answer:** 150 km"), "150"))
    check("plain answer", answer_match(extract_answer("Answer: 0.05"), "0.05"))
    check("boxed", answer_match(extract_answer("so \\boxed{72}"), "72"))
    check("the answer is", answer_match(extract_answer("The answer is 391."), "391"))
    check("last line fallback", answer_match(extract_answer("blah\nParis"), "Paris"))
    check("currency ignored", answer_match(extract_answer("Answer: $0.05"), "0.05"))
    check("thousands separator", answer_match(extract_answer("Answer: 2,598,960"), "2598960"))
    check("wrong answer rejected", not answer_match(extract_answer("Answer: 12"), "150"))
    check("normalize strips articles", normalize_answer("The Paris.") == "paris")


def test_extract_code():
    print("extract_code")
    code = "text\n```python\ndef f(x):\n    return x\n```\nmore"
    check("fenced block", extract_code(code).startswith("def f(x):"))
    check("bare code", extract_code("sure:\ndef f(x):\n    return x").startswith("def f(x):"))
    check("prose -> empty", extract_code("Write a function that adds numbers.") == "")


def test_exec():
    print("exec_tests")
    tests = [{"args": [7], "expected": True}, {"args": [9], "expected": False}]
    good = "def is_prime(n):\n    return n in (2,3,5,7,11,13)"
    check("correct code scores 1.0", score_code_exec(good, "is_prime", tests) == 1.0)
    half = "def is_prime(n):\n    return True"
    check("partially correct scores 0.5", score_code_exec(half, "is_prime", tests) == 0.5)
    check("prose scores 0.0", score_code_exec("no code here", "is_prime", tests) == 0.0)
    check("missing function scores 0.0", score_code_exec("def other():\n    pass", "is_prime", tests) == 0.0)
    r = run_tests("def is_prime(n):\n    raise ValueError('x')", "is_prime", tests)
    check("raising candidate is a failed test, not a crash", r["error"] is None and r["passed"] == 0)
    r = run_tests("import os\ndef is_prime(n):\n    return True", "is_prime", tests)
    check("forbidden import is blocked", r["error"] is not None and r["passed"] == 0)


def test_repair_code():
    print("repair_code")
    truncated = ("def is_prime(n):\n    if n < 2:\n        return False\n"
                 "    return True\n"
                 "def is_prime(n):\n    if n <= 1:\n        ret")   # cut mid-statement
    fixed = repair_code(truncated, "is_prime")
    check("truncated repetition is repaired", "return True" in fixed and fixed.count("def is_prime") == 1)
    good = "def f(x):\n    return x"
    check("valid code is unchanged", repair_code(good, "f") == good)
    broken = "def f(x):\n    retur"
    check("unrepairable code is returned as is", repair_code(broken, "f") == broken)


def test_scorer_v2():
    print("scorer v2")
    check("factual match", v2_factual("The capital of France is Paris.", "Paris") == 1.0)
    check("factual miss", v2_factual("The capital of France is Lyon.", "Paris") == 0.0)
    check("factual ignores continuation", v2_factual(" Paris.Human: garbage", "Paris") == 1.0)
    check("reasoning bold answer", v2_reasoning("reasoning\n**Answer:** 150", "150") == 1.0)
    check("reasoning miss", v2_reasoning("reasoning\n**Answer:** 99", "150") == 0.0)
    q, detail = v2_code("```python\ndef add(a, b):\n    return a + b\n```", "", "add",
                        [{"args": [1, 2], "expected": 3}])
    check("code via unit tests", q == 1.0)
    q, _ = v2_code("no code", "", "add", [{"args": [1, 2], "expected": 3}])
    check("code without code scores 0", q == 0.0)
    q, _ = v2_code("def add(a, b):\n    return a - b", "", "add",
                   [{"args": [1, 2], "expected": 3}])
    check("wrong code scores 0", q == 0.0)


def test_routers():
    print("routers")
    check("heuristic code -> turbo", route("Write Python factorial(n). Output code only.") in ("turbo", "vector"))
    check("heuristic reason -> ultra", route("Why is the sky blue? Explain the steps.") == "ultra")
    check("confidence proxy empty", confidence_proxy("") == 0.0)
    check("confidence cascade picks turbo", cascade({"turbo": 0.9}) == "turbo")
    check("confidence cascade falls back", cascade({"turbo": 0.0, "vector": 0.0}) == "ultra")
    # verified gate: code task with unit tests
    good = {"turbo": "```python\ndef add(a, b):\n    return a + b\n```"}
    bad = {"turbo": "I cannot help with that.", "vector": "Still no code."}
    lat = {"turbo": 1000.0, "vector": 2000.0, "ultra": 3000.0}
    tier, cost = cascade_verified("code_gen", good, lat, "add", [{"args": [1, 2], "expected": 3}])
    check("verified cascade accepts turbo", tier == "turbo" and cost == 1000.0)
    tier, cost = cascade_verified("code_gen", bad, lat, "add", [{"args": [1, 2], "expected": 3}])
    check("verified cascade escalates and pays for every attempt",
          tier == "ultra" and cost == 6000.0, f"{tier} {cost}")
    check("gate rejects prose for code", not gate("code_gen", "no code", "add", [{"args": [1], "expected": 1}]))
    check("gate accepts an answer for QA", gate("factual_qa", "Answer: Paris"))


def test_prompt_set():
    print("eval set")
    spec = subprocess.run([sys.executable, str(BASE / "eval_prompts" / "build_prompts_v2.py"), "--self-test"],
                          capture_output=True, text=True)
    check("v2 builder self-test passes", spec.returncode == 0, spec.stdout + spec.stderr)
    rows = [json.loads(l) for l in (BASE / "eval_prompts" / "prompts_v2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    check("180 items", len(rows) == 180, str(len(rows)))
    tasks = {t: sum(1 for r in rows if r["task"] == t) for t in ("factual_qa", "code_gen", "reasoning")}
    check("60 items per task", all(v == 60 for v in tasks.values()), str(tasks))
    check("prompts are distinct", len({r["prompt"] for r in rows}) == 180)
    check("every reasoning item has a gold answer",
          all(r["reference"] for r in rows if r["task"] == "reasoning"))
    check("every code item has unit tests",
          all(r.get("tests") and r.get("fn") for r in rows if r["task"] == "code_gen"))
    splits = {r["id"] for r in rows}
    check("ids unique", len(splits) == 180)


def main():
    for t in (test_trim, test_extract_answer, test_extract_code, test_repair_code,
              test_exec, test_scorer_v2, test_routers, test_prompt_set):
        t()
    if FAILS:
        print(f"\n{len(FAILS)} FAILURES: {FAILS}")
        sys.exit(1)
    print("\nall unit checks OK")


if __name__ == "__main__":
    main()
