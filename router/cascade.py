"""Cascade baseline (SPEC section 6.2): start cheap, escalate on low confidence.
Since real first-token entropy needs logits access (not in /completion),
use a cheap proxy: output length / repetition / empty as confidence.
- run turbo; if quality proxy passes -> stop
- else run vector; if passes -> stop
- else ultra
This file implements the *policy decision* given already-collected oracle outputs
(off-policy eval), where each attempted tier adds its latency to total cost.
For on-policy use, wire to live /completion calls in run_oracle order.
"""
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
    """Off-policy cascade: walk tiers cheap->expensive, accumulate the latency
    of every tier actually attempted (a failed attempt still costs), and stop
    at the first tier whose output passes the confidence proxy.
    Returns (final_tier, total_cost_ms)."""
    total = 0.0
    for tier in ("turbo", "vector", "ultra"):
        total += latency_by_tier.get(tier, 0.0)
        if confidence_proxy(content_by_tier.get(tier, "")) >= threshold or tier == "ultra":
            return tier, total
    return "ultra", total  # unreachable: ultra is a forced stop above
