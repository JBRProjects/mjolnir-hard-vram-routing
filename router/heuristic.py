"""Heuristic baseline (SPEC section 6.1): no training, cheap prompt features only.
Rules (floor to beat):
- looks like code (def/class/import/```/';'/brackets) -> turbo (fast dense, per SPEC example)
- long prompt (>1500 chars) -> avoid ultra prefill risk -> vector
- reasoning keywords (why/prove/steps/explain/if-then) -> ultra
- default -> turbo
Returns tier string. Keep inference <1ms.
"""
import re

CODE_RE = re.compile(r"(def |class |import |#include|```|;|\(\)|=>|\{|\})")
REASON_RE = re.compile(r"\b(why|prove|explain|steps|reason|if .+ then|therefore|because)\b", re.I)

def route(prompt: str) -> str:
    p = prompt or ""
    if len(p) > 1500:
        return "vector"  # dodge ultra long-prefill crash (SPEC section 7)
    if CODE_RE.search(p):
        return "turbo"
    if REASON_RE.search(p):
        return "ultra"
    if len(p.split()) < 12:
        return "turbo"  # short factual -> cheapest
    return "vector"

def route_batch(prompts):
    return [route(p) for p in prompts]
