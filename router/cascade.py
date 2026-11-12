"""Cascade baseline (SPEC section 6.2): start cheap, escalate on low confidence.

Two cascade variants live here:

1. `confidence_proxy` / `cascade` — the cycle-1 heuristic gate. It only looks at
   surface properties of the output (length, repetition, emptiness) because the
   /completion endpoint exposes no logits. Measured on cycle-1 data this gate
   passes for essentially every turbo answer, so the cascade collapsed to
   "always turbo" (RESULTS.md, router comparison table).

2. `gate` / `cascade_verified` — the cycle-2 gate: a TASK-SPECIFIC VERIFIER.
   For code tasks the verifier is the task's own unit test suite, executed
   (scoring/exec_tests.py); the answer is accepted only if the candidate code
   passes the tests. For QA/reasoning the verifier is the presence of a
   well-formed final answer (scoring/extract.py), since there is no executable
   check available without the gold answer.

The verifier is cheap (a local subprocess, tens of ms) next to a generation
(tens of seconds), so it does not defeat the purpose of the cascade.

Both variants evaluate OFF-POLICY on already-collected oracle outputs: every
tier that is *attempted* costs its latency, and the cascade stops at the first
tier whose verifier passes. `ultra` is a forced fallback.
"""
import sys, pathlib

BASE = pathlib.Path(__file__).parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))


def confidence_proxy(content: str) -> float:
    if not content or not content.strip():
        return 0.0
    toks = content.split()
    if len(toks) < 5:
        return 0.3
    uniq = len(set(toks)) / max(1, len(toks))
    if uniq < 0.25:  # degenerate repetition
        return 0.2
    return 0.9


def cascade(confidences: dict) -> str:
    """confidences: {tier: confidence (0..1)} for tiers in cost order.
    Returns first tier that passes the confidence threshold; ultra is a
    forced fallback (nothing cheaper left to escalate to)."""
    for tier in ("turbo", "vector"):
        if confidences.get(tier, 0.0) >= 0.5:
            return tier
    return "ultra"


def cascade_cost(content_by_tier: dict, latency_by_tier: dict, threshold: float = 0.5):
    """Off-policy cascade (confidence gate): walk tiers cheap->expensive,
    accumulate the latency of every tier actually attempted (a failed attempt
    still costs), and stop at the first tier whose output passes the proxy.
    Returns (final_tier, total_cost_ms)."""
    total = 0.0
    for tier in ("turbo", "vector", "ultra"):
        total += latency_by_tier.get(tier, 0.0)
        if confidence_proxy(content_by_tier.get(tier, "")) >= threshold or tier == "ultra":
            return tier, total
    return "ultra", total


# ------------------------------------------------------- verified gate (v2)

def gate(task: str, content: str, fn=None, tests=None) -> bool:
    """Task-specific verifier for a single candidate answer.

    code_gen : the candidate must define the function and pass the task's unit
               tests (execution, not naming).
    else     : the candidate must contain a well-formed final answer.
    """
    if content is None:
        return False
    from scoring.extract import extract_answer, extract_code, repair_code
    if task == "code_gen":
        if not tests:
            return bool(extract_code(content, fn).strip())
        from scoring.exec_tests import run_tests
        code = repair_code(extract_code(content, fn), fn)
        if not code.strip():
            return False
        r = run_tests(code, fn, tests)
        return r["passed"] == r["total"] and r["total"] > 0
    return bool(extract_answer(content).strip())


def cascade_verified(task: str, content_by_tier: dict, latency_by_tier: dict, fn=None, tests=None,
                     switch_ms: dict = None):
    """Off-policy cascade with the task-specific verifier.

    Walks turbo -> vector -> ultra, pays for every attempted tier, and stops at
    the first tier the verifier accepts. Returns (final_tier, total_cost_ms).

    `switch_ms` optionally maps a tier to the measured cost of making it
    resident (evict the current tier + load the next one; see
    harness/measure_switch.ps1). Only one tier fits in VRAM, so every
    escalation really does pay that price - without it the cascade numbers are
    optimistic.
    """
    switch_ms = switch_ms or {}
    total = 0.0
    first = True
    for tier in ("turbo", "vector", "ultra"):
        if not first:
            total += switch_ms.get(tier, 0.0)
        first = False
        total += latency_by_tier.get(tier, 0.0)
        if tier == "ultra":
            return tier, total
        if gate(task, content_by_tier.get(tier, ""), fn, tests):
            return tier, total
    return "ultra", total


def load_switch_costs(path=None) -> dict:
    """Measured tier load/eviction costs (ms) from harness/switch_cost.json."""
    import json, pathlib
    if path is None:
        path = pathlib.Path(__file__).parent.parent / "harness" / "switch_cost.json"
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    try:
        # utf-8-sig: the file is written by PowerShell, which may add a BOM
        data = json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    loads = data.get("load_seconds") or {}
    return {k: float(v) * 1000.0 for k, v in loads.items()}
