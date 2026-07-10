"""Verdict-probability extraction scenarios ported from m3-m6, incl. BPE merging."""

import math

import pytest

from rag_reliability.methods.m3.logprobs import (
    _pass_prob,
    _verdict_positions,
    extract_verdict_probs,
)


def tok(token: str, logprob: float = -0.1, top: dict | None = None) -> dict:
    return {"token": token, "logprob": logprob, "top": top or {}}


def test_whole_tokens_both_axes() -> None:
    tokens = [
        tok("FAITHFULNESS"),
        tok(":"),
        tok(" PASS", top={" PASS": -0.1, " FAIL": -2.4}),
        tok("\n"),
        tok("RELEVANCE"),
        tok(":"),
        tok(" FAIL", top={" FAIL": -0.2, " PASS": -1.7}),
    ]
    res = extract_verdict_probs(tokens)
    assert res is not None
    p_f, p_r = res
    assert p_f == pytest.approx(1 / (1 + math.exp(-2.3)), abs=1e-6)
    assert p_r == pytest.approx(1 - 1 / (1 + math.exp(-1.5)), abs=1e-6)


def test_bpe_subtokens_merged() -> None:
    tokens = [
        tok("FAITH"),
        tok("FULNESS"),
        tok(":"),
        tok(" PA", top={" PA": -0.2, " FA": -1.8}),
        tok("SS"),
        tok("\nRELEVANCE"),
        tok(":"),
        tok(" FA", top={" FA": -0.3, " PA": -1.4}),
        tok("IL"),
    ]
    assert _verdict_positions(tokens) == [3, 7]
    res = extract_verdict_probs(tokens)
    assert res is not None
    p_f, p_r = res
    assert p_f == pytest.approx(1 / (1 + math.exp(-1.6)), abs=1e-6)
    assert p_r == pytest.approx(1 - 1 / (1 + math.exp(-1.1)), abs=1e-6)


def test_label_tokens_not_matched_as_verdicts() -> None:
    tokens = [
        tok("FA"),
        tok("ITHFULNESS"),
        tok(":"),
        tok(" PASS"),
        tok(" RELEVANCE"),
        tok(":"),
        tok(" FAIL"),
    ]
    assert _verdict_positions(tokens) == [3, 6]


def test_one_side_visible_sigmoid() -> None:
    p = _pass_prob(tok(" PASS", logprob=-0.1, top={" PASS": -0.1}))
    assert p == pytest.approx(1 / (1 + math.exp(0.1)), abs=1e-6)


def test_exact_token_preferred_over_prefix() -> None:
    with_prefix = tok(" PASS", top={" PASS": -0.1, " PA": -0.5, " FAIL": -2.4})
    without_prefix = tok(" PASS", top={" PASS": -0.1, " FAIL": -2.4})
    assert _pass_prob(with_prefix) == pytest.approx(_pass_prob(without_prefix), abs=1e-9)


def test_whitespace_subtoken_breaks_merge() -> None:
    tokens = [tok("PA"), tok(" "), tok("SS"), tok(" FAIL")]
    assert _verdict_positions(tokens) == [3]


def test_fewer_than_two_positions_gives_none() -> None:
    assert extract_verdict_probs([tok("nothing"), tok(" PASS")]) is None
