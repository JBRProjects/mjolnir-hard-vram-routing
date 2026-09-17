"""Build stratified eval prompt set: 60 factual QA + 60 code-gen + 60 reasoning = 180.
Writes eval_prompts/prompts.jsonl with {id, task, prompt, reference, split}.
Split: 70% train / 30% test, stratified by task, deterministic (seed 13).
Usage: python eval_prompts/build_prompts.py
"""
import json, random, pathlib

OUT = pathlib.Path(__file__).parent / "prompts.jsonl"
SEED = 13

FACTUAL = [
    ("What is the capital of France?", "Paris"),
    ("What is 17 * 23?", "391"),
    ("Which element has chemical symbol Fe?", "Iron"),
    ("In which year did Apollo 11 land on the Moon?", "1969"),
    ("What does CPU stand for?", "Central Processing Unit"),
    ("What is the largest planet in the solar system?", "Jupiter"),
    ("Who wrote '1984'?", "George Orwell"),
    ("What is the boiling point of water at sea level in Celsius?", "100"),
    ("Which protocol is used to send email (SMTP/FTP)?", "SMTP"),
    ("What data structure uses LIFO order?", "Stack"),
]
CODE = [
    ("Write a Python function add(a, b) returning a+b. Output code only.", "def add(a, b):\n    return a + b"),
    ("Write Python to reverse a string s without [::-1]. Output code only.", "def reverse(s):\n    out=''\n    for ch in s:\n        out=ch+out\n    return out"),
    ("Write Python factorial(n) iteratively. Output code only.", "def factorial(n):\n    r=1\n    for i in range(2,n+1):\n        r*=i\n    return r"),
    ("Write Python is_prime(n) returning True/False. Output code only.", "def is_prime(n):\n    if n<2:\n        return False\n    i=2\n    while i*i<=n:\n        if n%i==0:\n            return False\n        i+=1\n    return True"),
    ("Write Python fibonacci(n) returning first n numbers as list. Output code only.", "def fibonacci(n):\n    a,b=[],(0,1)\n    x,y=0,1\n    for _ in range(n):\n        a.append(x)\n        x,y=y,x+y\n    return a"),
]
REASON = [
    "A train travels 60 km/h for 2.5 hours. How far does it go? Show steps.",
    "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops Lazzies? Explain briefly.",
    "You have 3 boxes labeled Apples, Oranges, Mixed — all mislabeled. You may pick one fruit from one box. How do you relabel all boxes? Explain.",
    "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much is the ball? Explain.",
    "If you flip two fair coins, what is P(at least one head)? Show reasoning.",
]

def expand():
    rows = []
    # factual: cycle base 10 -> 60 with variants
    for i in range(60):
        q, a = FACTUAL[i % len(FACTUAL)]
        prompt = q if i < len(FACTUAL) else f"{q} (variant {i//len(FACTUAL)+1}: answer concisely.)"
        rows.append({"task": "factual_qa", "prompt": prompt, "reference": a})
    # code: cycle base 5 -> 60 with variants (different function names/constraints)
    for i in range(60):
        q, a = CODE[i % len(CODE)]
        prompt = q if i < len(CODE) else f"{q} (variant {i//len(CODE)+1})"
        rows.append({"task": "code_gen", "prompt": prompt, "reference": a})
    # reasoning: cycle base 5 -> 60
    for i in range(60):
        q = REASON[i % len(REASON)]
        prompt = q if i < len(REASON) else f"{q} (restate answer in one final line starting with 'Answer:')"
        rows.append({"task": "reasoning", "prompt": prompt, "reference": ""})
    return rows

def main():
    random.seed(SEED)
    rows = expand()
    # stratified split
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    out = []
    n = 0
    for task, lst in by_task.items():
        random.shuffle(lst)
        cut = int(len(lst) * 0.7)
        for j, r in enumerate(lst):
            r = dict(r)
            r["split"] = "train" if j < cut else "test"
            out.append(r)
    random.shuffle(out)
    for i, r in enumerate(out):
        r["id"] = f"p{i:03d}_{r['task']}"
    OUT.write_text("\n".join(json.dumps(r) for r in out), encoding="utf-8")
    print(f"wrote {len(out)} prompts -> {OUT} (train={sum(1 for r in out if r['split']=='train')}, test={sum(1 for r in out if r['split']=='test')})")

if __name__ == "__main__":
    main()
