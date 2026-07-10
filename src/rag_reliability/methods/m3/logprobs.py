"""Verdict probabilities from token logprobs (Method 3, ported from m3-m6).

The judge answers with ``FAITHFULNESS: PASS|FAIL`` and ``RELEVANCE: PASS|FAIL``.
Instead of trusting the sampled text alone, we read P(PASS) from the token
logprobs at the two verdict positions: softmax over the PASS/FAIL pair when
both sides are visible in top logprobs, sigmoid when only one side is.
"""

from __future__ import annotations

import math

_WORDS = ("PASS", "FAIL")

# tokens: [{"token": str, "logprob": float, "top": {token: logprob}}]
Token = dict


def _verdict_positions(tokens: list[Token]) -> list[int]:
    """Start positions of PASS/FAIL: a whole token or a run of merged BPE subtokens."""
    out: list[int] = []
    i, n = 0, len(tokens)
    while i < n:
        t = tokens[i]["token"].strip().upper()
        if t in _WORDS:
            out.append(i)
            i += 1
            continue
        matched = False
        for w in _WORDS:
            if t and t != w and w.startswith(t):
                acc, j = t, i + 1
                while j < n and acc != w and w.startswith(acc):
                    piece = tokens[j]["token"].strip().upper()
                    if not piece:
                        break  # a whitespace subtoken breaks the merge
                    acc += piece
                    j += 1
                if acc == w:
                    out.append(i)
                    i = j
                    matched = True
                    break
        if not matched:
            i += 1
    return out


def _side_logprob(top: dict[str, float], word: str) -> float | None:
    """Logprob of one side at a position: exact token matches beat prefixes.

    Prefixes (>= 2 chars) are used only when the whole token is absent from
    top — otherwise prefix mass (which includes other continuations) would
    inflate the side.
    """
    exact: list[float] = []
    prefix: list[float] = []
    for t, v in top.items():
        s = t.strip().upper()
        if s == word:
            exact.append(v)
        elif len(s) >= 2 and word.startswith(s):
            prefix.append(v)
    lps = exact or prefix
    if not lps:
        return None
    m = max(lps)
    return m + math.log(sum(math.exp(v - m) for v in lps))


def _pass_prob(tok: Token) -> float:
    """P(PASS) at a verdict position: pair softmax; one side -> sigmoid; none -> 0.5."""
    top = tok["top"] or {tok["token"]: tok["logprob"]}
    lp_pass, lp_fail = _side_logprob(top, "PASS"), _side_logprob(top, "FAIL")
    if lp_pass is None and lp_fail is None:
        return 0.5
    if lp_pass is None:
        return 1.0 - 1.0 / (1.0 + math.exp(-lp_fail))
    if lp_fail is None:
        return 1.0 / (1.0 + math.exp(-lp_pass))
    m = max(lp_pass, lp_fail)
    e_p, e_f = math.exp(lp_pass - m), math.exp(lp_fail - m)
    return e_p / (e_p + e_f)


def extract_verdict_probs(tokens: list[Token]) -> tuple[float, float] | None:
    """(p_faith, p_rel) from the 1st and 2nd verdict positions; None if fewer than 2."""
    pos = _verdict_positions(tokens)
    if len(pos) < 2:
        return None
    return _pass_prob(tokens[pos[0]]), _pass_prob(tokens[pos[1]])
