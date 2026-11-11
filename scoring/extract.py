"""Format-neutral answer extraction and matching (MJOLNIR quality metric v2).

Why this module exists
----------------------
Cycle 1 (RESULTS.md, "Key finding / caveat (metric bias)") showed that the frozen
correctness metric was partly measuring OUTPUT FORMAT rather than CORRECTNESS:

  * reasoning scored "has a literal `Answer:` line"; turbo emits `Answer: ...`
    while ultra emits `**Answer:** ...`, so turbo won on style;
  * code-gen was scored by identifier overlap with the reference, so a correct
    implementation with different names scored 0.5;
  * factual QA was scored by substring match over the WHOLE output.

Metric v2 fixes this by separating "what the model said" from "how it said it":

  trim_continuation()  cut the model-generated conversation continuation
                       ("Human: ...", "<|im_end|>", "### Instruction") that the
                       base-style tiers append after answering;
  extract_code()       pull the code out of a fenced block or a code-ish span;
  extract_answer()     pull the final answer out of the many surface forms
                       ("Answer:", "**Answer:**", "\\boxed{}", "The answer is",
                       last line, ...);
  normalize_answer()   case/punctuation/article/number normalization;
  answer_match()       semantic comparison of the extracted answer to the gold.

Everything here is pure-python, no model calls, no dependencies: it is part of
the scoring path, not the router, so it must stay deterministic and cheap.
"""
import re

# ------------------------------------------------------------------ trimming

# Turn markers the tiers emit when they continue the prompt as if it were a
# conversation. Everything from the first marker on is model noise.
# The chat-style markers are matched case-insensitively; "Q:" only in upper case
# so that python dict keys / variables named q are not mistaken for a turn.
_TURN_MARKERS_CI = [
    r"Human\s*:", r"User\s*:", r"Assistant\s*:", r"System\s*:", r"Prompt\s*:",
    r"###\s*(?:Instruction|Question|Response|Human|User)",
    r"<\|im_end\|>", r"<\|im_start\|>", r"<\|endoftext\|>", r"</s>",
]
_TURN_MARKERS_CS = [r"Q\s*:"]
_TURN_RE_CI = re.compile("|".join(_TURN_MARKERS_CI))
_TURN_RE_CS = re.compile("|".join(_TURN_MARKERS_CS))


def trim_continuation(text: str) -> str:
    """Drop the model's own continuation of the conversation.

    The tiers are completion-style models served through /completion, so after
    answering they keep generating the next turn ("Human: ..."). That noise is
    not part of the answer and used to defeat substring/line based scoring.
    """
    if not text:
        return ""
    cut = len(text)
    for rx in (_TURN_RE_CI, _TURN_RE_CS):
        m = rx.search(text)
        if m and m.start() < cut:
            cut = m.start()
    if cut < len(text):
        text = text[:cut]
    return text.strip()


# -------------------------------------------------------------- code extract

_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\n?(.*?)(?:```|$)", re.S)


def extract_code(text: str) -> str:
    """Return the candidate program from a model answer.

    Order: fenced block -> first `def`/`import`/`class` span. Prose around the
    code is dropped so that scoring measures behaviour, not formatting.
    """
    text = trim_continuation(text)
    if not text:
        return ""
    blocks = _FENCE_RE.findall(text)
    if blocks:
        # prefer the first block that actually defines something
        for b in blocks:
            if re.search(r"^\s*(def |class |import |from )", b, re.M):
                return b.strip()
        return blocks[0].strip()
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*(def |class |import |from )", line):
            start = i
            break
    if start is None:
        return ""
    out = []
    for line in lines[start:]:
        if re.match(r"^(Human|User|Assistant)\s*:", line):
            break
        out.append(line)
    return "\n".join(out).strip()


# ------------------------------------------------------------ answer extract

_ANSWER_PATTERNS = [
    re.compile(r"(?:\*\*|__|#{1,6}\s*)?(?:final\s+)?answer\s*(?:\*\*|__)?\s*[:\-]\s*(.+)", re.I),
    re.compile(r"\\boxed\{([^{}]*)\}"),
    re.compile(r"the\s+(?:final\s+)?answer\s+is\s*:?\s*(.+)", re.I),
    re.compile(r"^\s*####\s*(.+)", re.M),
]


def _clean_span(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\**\s*", "", s)
    s = re.sub(r"\s*\**$", "", s)
    s = s.strip().strip("`*").strip()
    s = re.sub(r"^[:\-–—\s]+", "", s)
    s = s.strip().strip("`*_").strip()
    # drop a trailing period but keep decimals ("3.15")
    s = re.sub(r"\.\s*$", "", s)
    return s.strip()


def extract_answer(text: str) -> str:
    """Extract the model's final answer from an arbitrary surface form."""
    text = trim_continuation(text)
    if not text:
        return ""
    # explicit answer markers: take the LAST occurrence (models restate)
    best = None
    for pat in _ANSWER_PATTERNS:
        matches = pat.findall(text)
        if matches:
            cand = _clean_span(matches[-1] if isinstance(matches[-1], str) else matches[-1][0])
            if cand:
                best = cand
    if best:
        # a marked answer may still carry trailing prose -> keep the first clause
        return _first_clause(best)
    # no marker: fall back to the last non-empty line
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    last = _clean_span(lines[-1])
    if len(last) > 120:
        last = _clean_span(last[-120:])
    return _first_clause(last)


def _first_clause(s: str) -> str:
    """Trim a marked answer down to the answer itself (first clause/sentence)."""
    s = s.split("\n")[0].strip()
    m = re.match(r"^(.{1,80}?)(?:[.;]\s|$)", s)
    if m:
        s = m.group(1)
    return s.strip()


# ------------------------------------------------------------- normalization

_ARTICLES = re.compile(r"\b(the|a|an)\b")
_NUM_COMMA = re.compile(r"(?<=\d),(?=\d\d\d\b)")


def normalize_answer(s: str) -> str:
    """Casefold, strip markdown/punctuation/articles, canonicalise numbers."""
    if not s:
        return ""
    s = s.lower()
    s = _NUM_COMMA.sub("", s)          # 2,598,960 -> 2598960
    s = s.replace("\\(", "").replace("\\)", "")
    s = re.sub(r"[*_`~#>]+", " ", s)
    s = re.sub(r"[“”‘’«»]", "", s)
    s = re.sub(r"[$€£¥]", " ", s)      # currency symbols are not part of the value
    s = re.sub(r"[^\w\s.+/\-:%]", " ", s)   # keep word chars, digits, . + / - : %
    s = re.sub(r"\s+", " ", s).strip()
    s = _ARTICLES.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(" .-")
    return s


def _tokens(s: str):
    return [t for t in normalize_answer(s).split() if t]


def _num_tokens(s: str):
    """Numeric-looking tokens, normalized (150.0 -> 150)."""
    out = []
    for t in _tokens(s):
        t = t.replace(",", "")
        try:
            f = float(t)
            if f == int(f):
                out.append(str(int(f)))
            else:
                out.append(("%.6f" % f).rstrip("0").rstrip("."))
        except ValueError:
            pass
    return out


def answer_match(pred: str, gold: str) -> bool:
    """True when the extracted answer agrees with the gold answer.

    Matching is deliberately semantic rather than literal:
      * exact match after normalization;
      * numeric equality (so "150", "150.0", "150 km" vs "150 kilometres" work);
      * gold contained in the prediction when the gold is short (the model may
        add a unit or a short justification).
    """
    if not pred or not gold:
        return False
    p, g = normalize_answer(pred), normalize_answer(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    pn, gn = _num_tokens(p), _num_tokens(g)
    if gn and pn and pn[0] == gn[0]:
        return True
    gt = _tokens(gold)
    pt = _tokens(pred)
    if gt and pt:
        if all(t in pt for t in gt):
            return True
        if len(gt) == 1 and len(pt) <= 3 and gt[0] in pt:
            return True
    return False
