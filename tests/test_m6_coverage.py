"""Тесты детектора пропущенных условий. NLI не участвует вовсе."""

from __future__ import annotations

import inspect
import math

import pytest

from rag_reliability.methods.m6.coverage import (
    COVERAGE_KEYS,
    coverage_features,
    find_conditions,
)

CHUNK_CONDITIONS = (
    "Перевод возможен, если сумма не менее 100 руб. "
    "Зачисление происходит в течение 5 дней. Комиссия — 0.5%."
)
CHUNK_IRRELEVANT = "Кредитная карта выпускается не позднее 30 дней при условии дохода от 20000 руб."


# --------------------------------------------------------------------------- #
# Извлечение условий
# --------------------------------------------------------------------------- #


def test_finds_numeric_and_lexical_conditions() -> None:
    found = {condition.text for condition in find_conditions(CHUNK_CONDITIONS)}

    assert any("не менее" in text for text in found)
    assert any("в течение" in text for text in found)
    assert any("0.5" in text for text in found)
    assert any("100" in text and "руб" in text for text in found)


def test_condition_without_matches_yields_nothing() -> None:
    assert find_conditions("Обычный текст без ограничений и чисел.") == []


# --------------------------------------------------------------------------- #
# Покрытие
# --------------------------------------------------------------------------- #


def test_full_coverage_when_answer_repeats_every_condition() -> None:
    features = coverage_features(
        CHUNK_CONDITIONS,
        [CHUNK_CONDITIONS, CHUNK_IRRELEVANT],
        source_chunk_ids={0},
    )

    assert features["m6.cond_coverage"] == pytest.approx(1.0)
    assert features["m6.digit_coverage"] == pytest.approx(1.0)
    assert features["m6.uncovered_max"] == pytest.approx(0.0)
    assert features["m6.n_conditions"] > 0


def test_missed_condition_lowers_coverage_and_raises_uncovered_max() -> None:
    """Ровно тот класс ошибок, который entailment невидит: сказанное верно, но неполно."""
    complete = coverage_features(
        "Сумма не менее 100 руб., срок в течение 5 дней, комиссия 0.5%.",
        [CHUNK_CONDITIONS],
        source_chunk_ids={0},
    )
    incomplete = coverage_features(
        "Комиссия 0.5%.",
        [CHUNK_CONDITIONS],
        source_chunk_ids={0},
    )

    assert incomplete["m6.cond_coverage"] < complete["m6.cond_coverage"]
    assert incomplete["m6.digit_coverage"] < complete["m6.digit_coverage"]
    assert incomplete["m6.uncovered_max"] > complete["m6.uncovered_max"]


def test_coverage_is_limited_to_source_chunks() -> None:
    """Считать по всем 5–8 чанкам бессмысленно: там условия не по вопросу."""
    answer = "Сумма не менее 100 руб., зачисление в течение 5 дней, комиссия 0.5%."

    only_source = coverage_features(
        answer, [CHUNK_CONDITIONS, CHUNK_IRRELEVANT], source_chunk_ids={0}
    )
    both_chunks = coverage_features(
        answer, [CHUNK_CONDITIONS, CHUNK_IRRELEVANT], source_chunk_ids={0, 1}
    )

    assert only_source["m6.cond_coverage"] == pytest.approx(1.0)
    assert both_chunks["m6.n_conditions"] > only_source["m6.n_conditions"]
    assert both_chunks["m6.cond_coverage"] < only_source["m6.cond_coverage"]


def test_unknown_source_chunk_id_raises() -> None:
    with pytest.raises(IndexError, match="source chunk id"):
        coverage_features("ответ", [CHUNK_CONDITIONS], source_chunk_ids={0, 7})


def test_empty_source_set_raises_instead_of_perfect_coverage() -> None:
    """Пустое множество источников не имеет права выглядеть полным покрытием."""
    with pytest.raises(ValueError, match="source_chunk_ids"):
        coverage_features("ответ", [CHUNK_CONDITIONS], source_chunk_ids=set())


def test_chunk_without_conditions_gives_neutral_features() -> None:
    features = coverage_features(
        "Ответ без условий.", ["Текст без ограничений."], source_chunk_ids={0}
    )

    assert features["m6.n_conditions"] == pytest.approx(0.0)
    # нечего покрывать — это не «всё пропущено» и не «всё покрыто»
    assert features["m6.cond_coverage"] == pytest.approx(1.0)
    assert features["m6.uncovered_max"] == pytest.approx(0.0)


def test_returns_exactly_four_declared_finite_features() -> None:
    features = coverage_features("Ответ.", [CHUNK_CONDITIONS], source_chunk_ids={0})

    assert set(features) == set(COVERAGE_KEYS)
    assert len(COVERAGE_KEYS) == 4
    assert all(math.isfinite(value) for value in features.values())


def test_number_formats_are_normalized() -> None:
    """«100 000 ₽» в чанке и «100000 ₽» в ответе — одно и то же число."""
    features = coverage_features(
        "Лимит 100000 рублей в сутки.",
        ["Перевод до 100 000 ₽ в сутки."],
        source_chunk_ids={0},
    )

    assert features["m6.digit_coverage"] == pytest.approx(1.0)


def test_decimal_comma_and_dot_are_the_same_number() -> None:
    features = coverage_features(
        "Ставка 0.5% годовых.", ["Комиссия составляет 0,5% от суммы."], source_chunk_ids={0}
    )

    assert features["m6.digit_coverage"] == pytest.approx(1.0)


def test_coverage_cannot_call_nli_by_construction() -> None:
    """Критерий приёмки: coverage переиспользует матрицу grounding, своих вызовов нет."""
    parameters = inspect.signature(coverage_features).parameters

    assert not [name for name in parameters if "nli" in name.lower()]
    assert set(parameters) == {"answer", "chunks", "source_chunk_ids"}
