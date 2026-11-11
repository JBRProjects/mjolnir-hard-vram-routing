"""Build eval_prompts/legacy_gold.json — gold data for the CYCLE-1 prompt set.

Cycle 1's prompt set (eval_prompts/prompts.jsonl) has only 20 distinct base
items (10 factual, 5 code, 5 reasoning) repeated with cosmetic "(variant N)"
suffixes, and its reasoning rows carry NO gold answer. To re-score the cycle-1
LIVE oracle data with the format-neutral metric v2, the gold answers and the
unit tests for those 20 base items are needed.

This script emits a lookup keyed by the BASE prompt text; the scorer matches a
row's prompt against the longest key that prefixes it (so "(variant 3: ...)"
suffixes still resolve).

Writes: eval_prompts/legacy_gold.json
Usage: python eval_prompts/build_legacy_gold.py
"""
import json, pathlib, importlib.util

BASE = pathlib.Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("bp2", BASE / "eval_prompts" / "build_prompts_v2.py")
bp2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp2)

OUT = BASE / "eval_prompts" / "legacy_gold.json"

# legacy base prompt -> gold answer (factual / reasoning)
LEGACY_SHORT = {
    # factual
    "What is the capital of France?": "Paris",
    "What is 17 * 23?": "391",
    "Which element has chemical symbol Fe?": "Iron",
    "In which year did Apollo 11 land on the Moon?": "1969",
    "What does CPU stand for?": "Central Processing Unit",
    "What is the largest planet in the solar system?": "Jupiter",
    "Who wrote '1984'?": "George Orwell",
    "What is the boiling point of water at sea level in Celsius?": "100",
    "Which protocol is used to send email (SMTP/FTP)?": "SMTP",
    "What data structure uses LIFO order?": "Stack",
    # reasoning
    "A train travels 60 km/h for 2.5 hours. How far does it go? Show steps.": "150",
    "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops Lazzies? Explain briefly.": "Yes",
    "You have 3 boxes labeled Apples, Oranges, Mixed \u2014 all mislabeled. You may pick one fruit from one box. How do you relabel all boxes? Explain.": "Mixed",
    "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much is the ball? Explain.": "0.05",
    "If you flip two fair coins, what is P(at least one head)? Show reasoning.": "0.75",
}

# legacy base prompt -> function name (code tasks; unit tests come from the v2 bank)
LEGACY_CODE = {
    "Write a Python function add(a, b) returning a+b. Output code only.": "add",
    "Write Python to reverse a string s without [::-1]. Output code only.": "reverse",
    "Write Python factorial(n) iteratively. Output code only.": "factorial",
    "Write Python is_prime(n) returning True/False. Output code only.": "is_prime",
    "Write Python fibonacci(n) returning first n numbers as list. Output code only.": "fibonacci",
}


def main():
    gold = {}
    for prompt, ans in LEGACY_SHORT.items():
        gold[prompt] = {"reference": ans}
    by_fn = {fn: (prompt, ref, cases) for prompt, fn, ref, cases in bp2.CODE}
    for prompt, fn in LEGACY_CODE.items():
        assert fn in by_fn, f"no v2 code entry for {fn}"
        _, ref, cases = by_fn[fn]
        vals = bp2._run_ref(ref, fn, cases)
        gold[prompt] = {"reference": ref, "fn": fn,
                        "tests": [{"args": list(c), "expected": v} for c, v in zip(cases, vals)]}
    OUT.write_text(json.dumps(gold, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({len(gold)} legacy base items: "
          f"{len(LEGACY_SHORT)} short-answer + {len(LEGACY_CODE)} code)")


if __name__ == "__main__":
    main()
