"""Verdict probabilities from token logprobs (Method 3, ported from m3-m6).

The judge answers with ``FAITHFULNESS: PASS|FAIL`` and ``RELEVANCE: PASS|FAIL``.
Instead of trusting the sampled text alone, we read P(PASS) from the token
logprobs at the two verdict positions: softmax over the PASS/FAIL pair when
both sides are visible in top logprobs, direct token probability when only
one side is visible.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_right

_WORDS = ("PASS", "FAIL")
_AXES = ("FAITHFULNESS", "RELEVANCE")

# tokens: [{"token": str, "logprob": float, "top": {token: logprob}}]
Token = dict


def _anchor_bounds(tokens: list[Token], axis: str) -> tuple[int, int] | None:
    """Character and token boundary after the last ``AXIS:`` anchor."""
    text = "".join(str(token["token"]) for token in tokens)
    matches = list(re.finditer(rf"\b{axis}\s*:\**", text, re.IGNORECASE))
    if not matches:
        return None
    match = matches[-1]
    token_ends: list[int] = []
    total = 0
    for token in tokens:
        total += len(str(token["token"]))
        token_ends.append(total)
    return match.start(), bisect_right(token_ends, match.end())


def _verdict_position_between(tokens: list[Token], start: int, stop: int) -> int | None:
    """First whole or BPE-merged PASS/FAIL inside one anchored axis section."""
    i = start
    while i < stop:
        t = tokens[i]["token"].strip().upper()
        if t in _WORDS:
            return i
        for w in _WORDS:
            if t and t != w and w.startswith(t):
                acc, j = t, i + 1
                while j < stop and acc != w and w.startswith(acc):
                    piece = tokens[j]["token"].strip().upper()
                    if not piece:
                        break  # a whitespace subtoken breaks the merge
                    acc += piece
                    j += 1
                if acc == w:
                    return i
        i += 1
    return None


def _verdict_positions(tokens: list[Token]) -> list[int]:
    """Verdict positions inside the two named axis sections.

    PASS/FAIL mentions in free-form analysis are ignored. Missing anchors leave
    the corresponding axis unresolved so the caller can use the text fallback.
    """
    bounds = {axis: _anchor_bounds(tokens, axis) for axis in _AXES}
    out: list[int] = []
    for index, axis in enumerate(_AXES):
        bound = bounds[axis]
        if bound is None:
            continue
        start = bound[1]
        stop = len(tokens)
        if index + 1 < len(_AXES):
            next_bound = bounds[_AXES[index + 1]]
            if next_bound is None or next_bound[0] <= bound[0]:
                continue
            text_length = 0
            stop = len(tokens)
            for token_index, token in enumerate(tokens):
                if text_length >= next_bound[0]:
                    stop = token_index
                    break
                text_length += len(str(token["token"]))
        position = _verdict_position_between(tokens, start, stop)
        if position is not None:
            out.append(position)
    return out


def _side_logprob(top: dict[str, float], word: str) -> float | None:
    """Logprob of one side at a position: exact token matches beat prefixes.

    Prefixes are used only when the whole token is absent from top — otherwise
    prefix mass (which includes other continuations) would inflate the side.
    One-character prefixes are accepted only for ``P`` and ``F`` because local
    tokenizers may split PASS/FAIL after the first letter.
    """
    exact: list[float] = []
    prefix: list[float] = []
    for t, v in top.items():
        s = t.strip().upper()
        if s == word:
            exact.append(v)
        elif (len(s) >= 2 or s == word[0]) and word.startswith(s):
            prefix.append(v)
    lps = exact or prefix
    if not lps:
        return None
    m = max(lps)
    return m + math.log(sum(math.exp(v - m) for v in lps))


def _pass_prob(tok: Token) -> float:
    """P(PASS) at a verdict position.

    A one-sided top-logprobs entry already contains an absolute log-probability,
    so it must be exponentiated rather than passed through a sigmoid.
    """
    top = tok["top"] or {tok["token"]: tok["logprob"]}
    lp_pass, lp_fail = _side_logprob(top, "PASS"), _side_logprob(top, "FAIL")
    if lp_pass is None and lp_fail is None:
        return 0.5
    if lp_pass is None:
        return 1.0 - math.exp(lp_fail)
    if lp_fail is None:
        return math.exp(lp_pass)
    m = max(lp_pass, lp_fail)
    e_p, e_f = math.exp(lp_pass - m), math.exp(lp_fail - m)
    return e_p / (e_p + e_f)


def extract_verdict_probs(tokens: list[Token]) -> tuple[float, float] | None:
    """(p_faith, p_rel) from anchored verdict positions; None if either is absent."""
    pos = _verdict_positions(tokens)
    if len(pos) < 2:
        return None
    return _pass_prob(tokens[pos[0]]), _pass_prob(tokens[pos[1]])
