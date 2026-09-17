"""Cascade baseline (SPEC section 6.2): start cheap, escalate on low confidence.
Since real first-token entropy needs logits access (not in /completion),
use a cheap proxy: output length / repetition / empty as confidence.
- run turbo; if quality proxy passes -> stop
- else run vector; if passes -> stop
- else ultra
This file implements the *policy decision* given already-collected oracle outputs
(off-policy eval). For on-policy use, wire to live /completion calls in run_oracle order.
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

def cascade(scored_tiers: dict, threshold: float = 0.5) -> str:
    """scored_tiers: {tier: content}. Returns chosen tier."""
    for tier in ("turbo", "vector", "ultra"):
        c = scored_tiers.get(tier, "")
        if tier == "ultra":
            return "ultra"
        if confidence_proxy(c) >= threshold:
            return tier
    return "ultra"
