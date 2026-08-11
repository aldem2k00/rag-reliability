# tests/test_surface_features.py
"""Свойства поверхностных фич: инварианты, а не снимок текущих чисел."""

from __future__ import annotations

from rag_reliability.methods.surface.features import (
    FEATURE_KEYS,
    feature_vector,
    ngram_overlap,
    split_chunks,
    surface_features,
)
from rag_reliability.schema import RagSample


def _sample(question: str = "вопрос", context: str = "контекст", answer: str = "ответ") -> RagSample:
    return RagSample(
        id="s", question=question, context=context, answer=answer, faithfulness=1, relevance=1
    )


def test_overlap_is_one_when_answer_is_copied_from_context() -> None:
    text = "обслуживание карты стоит 150 рублей в месяц"
    assert ngram_overlap(text, f"вводная часть {text} и хвост", n=2) == 1.0


def test_overlap_is_zero_for_disjoint_texts() -> None:
    assert ngram_overlap("альфа бета гамма", "дельта эпсилон дзета", n=2) == 0.0


def test_overlap_falls_back_to_unigrams_for_short_answers() -> None:
    """Один токен короче биграммы — иначе фича молча обнулялась бы на коротких ответах."""
    assert ngram_overlap("да", "да, конечно", n=2) == 1.0


def test_empty_answer_has_zero_overlap() -> None:
    assert ngram_overlap("", "любой текст", n=2) == 0.0


def test_overlap_is_bounded() -> None:
    for answer, context in [("а б в", "а б в г"), ("", "x"), ("длинный ответ", "")]:
        assert 0.0 <= ngram_overlap(answer, context, n=2) <= 1.0


def test_digit_match_is_one_when_answer_has_no_numbers() -> None:
    """Нет чисел — нечему противоречить; ноль читался бы как «числа не подтверждены»."""
    assert surface_features(_sample(answer="ответ без цифр"))["digit_match_ratio"] == 1.0


def test_digit_match_detects_an_unsupported_number() -> None:
    supported = surface_features(_sample(context="стоит 150 рублей", answer="стоит 150 рублей"))
    invented = surface_features(_sample(context="стоит 150 рублей", answer="стоит 300 рублей"))
    assert supported["digit_match_ratio"] == 1.0
    assert invented["digit_match_ratio"] == 0.0


def test_digit_match_normalizes_decimal_separator() -> None:
    features = surface_features(_sample(context="ставка 7.5 процента", answer="ставка 7,5"))
    assert features["digit_match_ratio"] == 1.0


def test_chunks_are_recovered_from_markers() -> None:
    context = "[CHUNK 1]\nпервый\n[CHUNK 2]\nвторой\n[CHUNK 3]\nтретий"
    assert len(split_chunks(context)) == 3
    assert surface_features(_sample(context=context))["n_chunks"] == 3.0


def test_context_without_markers_counts_as_one_chunk() -> None:
    assert split_chunks("сплошной текст") == ["сплошной текст"]
    assert surface_features(_sample(context="сплошной текст"))["n_chunks"] == 1.0


def test_empty_context_has_no_chunks() -> None:
    assert split_chunks("") == []


def test_navigation_ratio_reacts_to_links() -> None:
    plain = surface_features(_sample(answer="комиссия составляет сто рублей"))
    navigational = surface_features(_sample(answer="перейдите по ссылке https://example.test"))
    assert plain["url_or_nav_ratio"] == 0.0
    assert navigational["url_or_nav_ratio"] > 0.0


def test_feature_vector_follows_the_declared_order() -> None:
    sample = _sample(context="[CHUNK 1]\nтекст 150", answer="текст 150")
    features = surface_features(sample)
    assert feature_vector(sample) == [features[key] for key in FEATURE_KEYS]


def test_features_are_finite_on_degenerate_input() -> None:
    for sample in [_sample("", "", ""), _sample("q", "", "a"), _sample("", "c", "")]:
        assert all(value == value and abs(value) != float("inf") for value in feature_vector(sample))
