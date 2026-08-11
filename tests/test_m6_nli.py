"""Тесты оконного NLI на мок-токенайзере и мок-forward: весов и GPU не нужно.

Проверяется требуемое поведение (окна покрывают премису, пара берётся из одного
окна, нормировка идёт по {entail, contra}), а не то, что получилось у текущей
реализации: три из четырёх свойств здесь — это починки известных багов.
"""

from __future__ import annotations

import math

import pytest

from rag_reliability.methods.m6.nli import (
    SPECIAL_TOKENS_MARGIN,
    aggregate_windows,
    score_pairs,
    split_tokens,
    two_class_probabilities,
    window_premise,
)


class FakeTokenizer:
    """Пословный токенайзер: id токена — его индекс в тексте, decode склеивает."""

    def __call__(self, *texts: str, add_special_tokens: bool = True, **kwargs: object):  # noqa: ARG002
        if len(texts) == 1:
            return {"input_ids": texts[0].split()}
        return {"input_ids": [list(pair) for pair in zip(*(text.split() for text in texts))]}

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


def _tokens(text: str) -> list[str]:
    return text.split()


# --------------------------------------------------------------------------- #
# split_tokens
# --------------------------------------------------------------------------- #


def test_short_sequence_stays_one_window() -> None:
    tokens = list("abc")
    assert split_tokens(tokens, budget=10, overlap=3) == [tokens]


@pytest.mark.parametrize("n_tokens", [11, 37, 100, 257])
@pytest.mark.parametrize(("budget", "overlap"), [(10, 3), (16, 8), (32, 0)])
def test_windows_cover_every_token(n_tokens: int, budget: int, overlap: int) -> None:
    """Ни один токен премисы не теряется — иначе окно молча режет условие."""
    tokens = list(range(n_tokens))
    windows = split_tokens(tokens, budget=budget, overlap=overlap)

    assert all(len(window) <= budget for window in windows)
    assert set().union(*(set(window) for window in windows)) == set(tokens)
    assert windows[-1][-1] == tokens[-1]


def test_windows_overlap_by_requested_amount() -> None:
    windows = split_tokens(list(range(30)), budget=10, overlap=4)
    for left, right in zip(windows, windows[1:]):
        assert set(left[-4:]) & set(right), "соседние окна должны перекрываться"


# --------------------------------------------------------------------------- #
# window_premise: бюджет от фактической длины гипотезы, без огрызка
# --------------------------------------------------------------------------- #


def test_long_premise_is_windowed_not_truncated() -> None:
    """Баг (в): премиса резалась до 32 токенов, результат зависел от длины гипотезы."""
    premise = " ".join(f"p{i}" for i in range(300))
    hypothesis = "короткая гипотеза"

    windows = window_premise(
        premise, hypothesis, tokenizer=FakeTokenizer(), max_length=64, overlap=8
    )

    assert len(windows) > 1
    covered = " ".join(windows).split()
    assert set(covered) == set(premise.split())


def test_budget_follows_actual_hypothesis_length() -> None:
    """Длинная гипотеза оставляет меньше места премисе — значит больше окон, не огрызок."""
    premise = " ".join(f"p{i}" for i in range(300))
    tokenizer = FakeTokenizer()

    short = window_premise(premise, "h", tokenizer=tokenizer, max_length=64, overlap=0)
    long = window_premise(
        premise, " ".join(f"h{i}" for i in range(30)), tokenizer=tokenizer, max_length=64, overlap=0
    )

    assert len(long) > len(short)
    for windows in (short, long):
        assert set(" ".join(windows).split()) == set(premise.split())


def test_window_plus_hypothesis_fits_max_length() -> None:
    premise = " ".join(f"p{i}" for i in range(200))
    hypothesis = " ".join(f"h{i}" for i in range(12))
    max_length = 64

    windows = window_premise(
        premise, hypothesis, tokenizer=FakeTokenizer(), max_length=max_length, overlap=4
    )

    for window in windows:
        total = len(_tokens(window)) + len(_tokens(hypothesis)) + SPECIAL_TOKENS_MARGIN
        assert total <= max_length


def test_hypothesis_longer_than_max_length_is_truncated_and_premise_survives() -> None:
    """Единственный случай усечения — гипотеза сама не влезает; премиса всё равно в окнах."""
    premise = " ".join(f"p{i}" for i in range(100))
    hypothesis = " ".join(f"h{i}" for i in range(200))

    windows = window_premise(
        premise, hypothesis, tokenizer=FakeTokenizer(), max_length=32, overlap=4
    )

    assert windows
    assert set(" ".join(windows).split()) == set(premise.split())


# --------------------------------------------------------------------------- #
# aggregate_windows: пара из одного окна
# --------------------------------------------------------------------------- #


def test_pair_comes_from_a_single_window() -> None:
    """Баг (г): независимые max по entail и contra давали сумму > 1."""
    scores = [
        {"entail": 0.10, "contra": 0.90},
        {"entail": 0.80, "contra": 0.20},
        {"entail": 0.30, "contra": 0.70},
    ]

    aggregated = aggregate_windows(scores, groups=[0, 0, 0], n_pairs=1)

    assert aggregated == [{"entail": 0.80, "contra": 0.20}]


def test_aggregate_keeps_pair_order_and_length() -> None:
    scores = [
        {"entail": 0.1, "contra": 0.9},
        {"entail": 0.7, "contra": 0.3},
        {"entail": 0.4, "contra": 0.6},
    ]

    aggregated = aggregate_windows(scores, groups=[0, 1, 1], n_pairs=2)

    assert aggregated == [{"entail": 0.1, "contra": 0.9}, {"entail": 0.7, "contra": 0.3}]


def test_aggregated_probabilities_never_exceed_one() -> None:
    scores = [{"entail": e, "contra": 1.0 - e} for e in (0.05, 0.55, 0.95, 0.35)]

    aggregated = aggregate_windows(scores, groups=[0, 0, 0, 0], n_pairs=1)

    assert aggregated[0]["entail"] + aggregated[0]["contra"] == pytest.approx(1.0)


def test_aggregate_rejects_pair_without_windows() -> None:
    with pytest.raises(ValueError, match="no window scores"):
        aggregate_windows([{"entail": 0.5, "contra": 0.5}], groups=[0], n_pairs=2)


# --------------------------------------------------------------------------- #
# Нормировка по {entail, contra}
# --------------------------------------------------------------------------- #


def test_two_class_normalization_drops_neutral() -> None:
    """Баг (д): трёхклассовый softmax отдавал массу neutral, медиана contra = 0.0099."""
    logits = [2.0, 5.0, 1.0]  # entail, neutral, contra

    probabilities = two_class_probabilities(logits, entail_index=0, contra_index=2)

    expected_entail = math.exp(2.0) / (math.exp(2.0) + math.exp(1.0))
    assert probabilities["entail"] == pytest.approx(expected_entail)
    assert probabilities["contra"] == pytest.approx(1.0 - expected_entail)
    assert probabilities["entail"] + probabilities["contra"] == pytest.approx(1.0)


def test_two_class_normalization_is_monotone_in_entail_logit() -> None:
    previous = -1.0
    for entail_logit in (-3.0, -1.0, 0.0, 1.0, 4.0):
        value = two_class_probabilities(
            [entail_logit, 3.0, 0.5], entail_index=0, contra_index=2
        )["entail"]
        assert value > previous
        previous = value


def test_two_class_normalization_survives_large_logits() -> None:
    probabilities = two_class_probabilities([900.0, 0.0, 800.0], entail_index=0, contra_index=2)

    assert math.isfinite(probabilities["entail"])
    assert probabilities["entail"] == pytest.approx(1.0 / (1.0 + math.exp(-100.0)))


# --------------------------------------------------------------------------- #
# score_pairs: сквозной путь на мок-forward
# --------------------------------------------------------------------------- #


def _forward_fn(calls: list[list[tuple[str, str]]]):
    """Мок-модель: логиты зависят от совпадения слов окна и гипотезы."""

    def forward(batch: list[tuple[str, str]]) -> list[list[float]]:
        calls.append(batch)
        logits = []
        for premise, hypothesis in batch:
            shared = len(set(premise.split()) & set(hypothesis.split()))
            logits.append([float(shared), 0.5, 1.0 - float(shared)])
        return logits

    return forward


def test_score_pairs_returns_one_row_per_pair_in_order() -> None:
    pairs = [
        (" ".join(f"p{i}" for i in range(100)), "p99"),
        ("p0 p1 p2", "zzz"),
    ]
    calls: list[list[tuple[str, str]]] = []

    scored = score_pairs(
        pairs,
        tokenizer=FakeTokenizer(),
        forward_fn=_forward_fn(calls),
        entail_index=0,
        contra_index=2,
        max_length=32,
        overlap=4,
        batch_size=8,
    )

    assert len(scored) == len(pairs)
    for row in scored:
        assert row["entail"] + row["contra"] == pytest.approx(1.0)
    # первая пара подкреплена только хвостовым окном — оконная нарезка его нашла
    assert scored[0]["entail"] > scored[1]["entail"]


def test_score_pairs_batches_windows_not_pairs() -> None:
    pairs = [(" ".join(f"p{i}" for i in range(200)), "p1")]
    calls: list[list[tuple[str, str]]] = []

    score_pairs(
        pairs,
        tokenizer=FakeTokenizer(),
        forward_fn=_forward_fn(calls),
        entail_index=0,
        contra_index=2,
        max_length=32,
        overlap=4,
        batch_size=4,
    )

    assert sum(len(batch) for batch in calls) > 1
    assert all(len(batch) <= 4 for batch in calls)


def test_score_pairs_on_empty_input() -> None:
    assert score_pairs(
        [],
        tokenizer=FakeTokenizer(),
        forward_fn=_forward_fn([]),
        entail_index=0,
        contra_index=2,
        max_length=32,
        overlap=4,
        batch_size=4,
    ) == []
