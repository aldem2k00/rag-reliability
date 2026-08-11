"""Метрика GEPA (balanced accuracy) и стоп-правило H5. Без dspy."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from rag_reliability.evaluation.bootstrap import PairedResult
from rag_reliability.methods.m3.gepa import (
    H5_REJECTED,
    H5_SUPPORTED,
    H5_UNTESTED,
    balanced_accuracy,
    class_weights,
    example_score,
    gepa_metric,
    h5_decision,
    h5_verdict,
)


_SPEC = importlib.util.spec_from_file_location(
    "run_gepa", Path(__file__).parents[1] / "scripts" / "run_gepa.py"
)
assert _SPEC is not None and _SPEC.loader is not None
run_gepa = importlib.util.module_from_spec(_SPEC)
sys.modules["run_gepa"] = run_gepa
_SPEC.loader.exec_module(run_gepa)


def _labels(n_positive: int, n_negative: int) -> list[int]:
    return [1] * n_positive + [0] * n_negative


# --------------------------------------------------------------------------- #
# Свойства метрики
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("split", [(72, 28), (95, 5), (50, 50), (5, 95)])
@pytest.mark.parametrize("constant", [0, 1])
def test_constant_predictor_scores_half_at_any_imbalance(
    split: tuple[int, int], constant: int
) -> None:
    """Главное свойство: «всегда PASS» не может выиграть эволюцию.

    Прежняя accuracy давала константному PASS 0.72 на базовой ставке корпуса,
    и кандидаты выигрывали, сдвигая судью в этот режим.
    """
    golds = _labels(*split)
    assert gepa_metric(golds, [constant] * len(golds)) == pytest.approx(0.5)


def test_recovering_negative_recall_strictly_raises_the_score() -> None:
    """Улучшение recall редкого класса обязано двигать метрику вверх."""
    golds = _labels(72, 28)
    always_pass = [1] * 100
    one_negative_found = [1] * 72 + [0] + [1] * 27

    assert gepa_metric(golds, one_negative_found) > gepa_metric(golds, always_pass)


def test_negative_recall_is_worth_more_than_positive_recall_at_72_28() -> None:
    """При 72/28 одно верное решение по редкому классу весит больше, чем по частому."""
    golds = _labels(72, 28)
    one_negative = [1] * 72 + [0] + [1] * 27
    lose_one_positive = [0] + [1] * 71 + [1] * 28

    assert gepa_metric(golds, one_negative) - 0.5 > 0.5 - gepa_metric(golds, lose_one_positive)


def test_perfect_and_inverted_predictors() -> None:
    golds = _labels(72, 28)
    assert gepa_metric(golds, golds) == pytest.approx(1.0)
    assert gepa_metric(golds, [1 - gold for gold in golds]) == pytest.approx(0.0)


def test_unparsed_verdict_counts_as_a_miss() -> None:
    """``None`` — вердикта в выводе нет. Сломанный формат обязан стоить скора,
    иначе GEPA свободно эволюционирует промпт, который инференс не разберёт."""
    golds = _labels(2, 2)
    assert gepa_metric(golds, [1, 1, 0, 0]) == pytest.approx(1.0)
    assert gepa_metric(golds, [1, None, 0, 0]) == pytest.approx(0.75)


def test_missing_class_is_an_error_not_a_silent_zero() -> None:
    with pytest.raises(ValueError, match="Class 0 is absent"):
        gepa_metric([1, 1, 1], [1, 1, 1])


def test_length_mismatch_is_an_error() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        gepa_metric([1, 0], [1])


# --------------------------------------------------------------------------- #
# По-примерный скор для DSPy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("split", [(72, 28), (95, 5), (50, 50), (1, 9)])
def test_mean_example_score_equals_balanced_accuracy(split: tuple[int, int]) -> None:
    """GEPA усредняет по-примерные скоры; среднее обязано СОВПАДАТЬ с метрикой.

    Иначе оптимизируется одна величина, а отчитывается похожая другая — ровно
    тот класс расхождения, из-за которого метрику и пришлось переделывать.
    """
    golds = _labels(*split)
    rng = np.random.default_rng(0)
    preds = rng.integers(0, 2, size=len(golds)).tolist()

    weights = class_weights(golds)
    per_example = [
        example_score(gold, pred, weights) for gold, pred in zip(golds, preds, strict=True)
    ]
    assert float(np.mean(per_example)) == pytest.approx(balanced_accuracy(golds, preds))


def test_class_weights_are_inverse_frequency() -> None:
    weights = class_weights(_labels(72, 28))
    assert weights[1] == pytest.approx(100 / (2 * 72))
    assert weights[0] == pytest.approx(100 / (2 * 28))


def test_normalized_weights_stay_in_the_unit_interval_and_keep_the_ordering() -> None:
    """При 72/28 сырой вес редкого класса равен 1.79. Если GEPA клипует скор в
    [0, 1], награда за редкий класс срезается — а это ровно то свойство, ради
    которого метрику и меняли."""
    golds = _labels(72, 28)
    raw = class_weights(golds)
    normalized = class_weights(golds, normalize=True)

    assert max(raw.values()) > 1.0
    assert all(0.0 < value <= 1.0 for value in normalized.values())
    assert normalized[0] / normalized[1] == pytest.approx(raw[0] / raw[1])

    rng = np.random.default_rng(1)
    for _ in range(5):
        preds = rng.integers(0, 2, size=len(golds)).tolist()
        mean = float(
            np.mean(
                [example_score(g, p, normalized) for g, p in zip(golds, preds, strict=True)]
            )
        )
        assert mean == pytest.approx(balanced_accuracy(golds, preds) / max(raw.values()))


def test_example_score_rejects_unknown_gold_label() -> None:
    with pytest.raises(ValueError, match="No class weight"):
        example_score(7, 7, class_weights(_labels(1, 1)))


# --------------------------------------------------------------------------- #
# Стоп-правило H5
# --------------------------------------------------------------------------- #


def _paired(delta: float, lo: float, hi: float, p: float = 0.5) -> PairedResult:
    return PairedResult(delta=delta, ci95=(lo, hi), p=p, significant=p < 0.05)


def test_the_historical_run_must_not_be_rejected() -> None:
    """Регресс на прошлый ложный отказ.

    Эволюция была остановлена по Δ = +0.0216 при 95% ДИ [−0.037, +0.100].
    Верхняя граница выше нуля, значит H5 осталась НЕПРОВЕРЕННОЙ, а не
    опровергнутой. Этот тест существует, чтобы отказ не повторился.
    """
    verdict = h5_decision(_paired(0.0216, -0.037, 0.100), n_seeds=1, n_cases=300)

    assert verdict.status == H5_UNTESTED
    assert not verdict.rejected
    assert "не проверена" in verdict.conclusion


def test_positive_point_estimate_alone_never_rejects() -> None:
    """Даже отрицательная точечная оценка не отвергает, пока ДИ накрывает ноль."""
    verdict = h5_decision(_paired(-0.02, -0.09, 0.05), n_seeds=3, n_cases=300)
    assert verdict.status == H5_UNTESTED
    assert not verdict.rejected


def test_rejected_only_when_the_upper_bound_is_below_zero() -> None:
    verdict = h5_decision(_paired(-0.06, -0.11, -0.01), n_seeds=3, n_cases=300)
    assert verdict.status == H5_REJECTED
    assert verdict.rejected
    assert "верхняя граница" in verdict.conclusion


def test_upper_bound_exactly_zero_does_not_reject() -> None:
    """Граница строгая: ДИ, касающийся нуля, не является доказательством вреда."""
    assert h5_decision(_paired(-0.05, -0.10, 0.0), n_seeds=3, n_cases=300).status == H5_UNTESTED


def test_supported_when_the_lower_bound_is_above_zero() -> None:
    verdict = h5_decision(_paired(0.06, 0.01, 0.11), n_seeds=3, n_cases=300)
    assert verdict.status == H5_SUPPORTED
    assert not verdict.rejected


# --------------------------------------------------------------------------- #
# Парный бутстрэп по кейсам и сидам
# --------------------------------------------------------------------------- #


def test_identical_runs_give_zero_delta_and_no_rejection() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    runs = [rng.integers(0, 2, size=200) for _ in range(3)]

    verdict = h5_verdict(y, runs, list(runs), B=200, seed=0)

    assert verdict.delta == pytest.approx(0.0)
    assert verdict.ci95 == (0.0, 0.0)
    assert verdict.status == H5_UNTESTED
    assert verdict.n_seeds == 3
    assert verdict.n_cases == 200


def test_a_clearly_worse_markers_arm_is_rejected() -> None:
    """Реально худший markers: ДИ целиком ниже нуля — вот это и есть отказ."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=400)
    plain = [y.copy() for _ in range(3)]
    markers = []
    for seed in range(3):
        noisy = y.copy()
        flip = np.random.default_rng(seed).choice(400, size=140, replace=False)
        noisy[flip] = 1 - noisy[flip]
        markers.append(noisy)

    verdict = h5_verdict(y, markers, plain, B=300, seed=0)

    assert verdict.ci95[1] < 0.0
    assert verdict.status == H5_REJECTED


def test_unpaired_arms_are_an_error() -> None:
    y = [0, 1, 0, 1]
    with pytest.raises(ValueError, match="paired by seed"):
        h5_verdict(y, [[0, 1, 0, 1], [0, 1, 0, 1]], [[0, 1, 0, 1]], B=10)


def test_case_count_mismatch_is_an_error() -> None:
    with pytest.raises(ValueError, match="disagree with y"):
        h5_verdict([0, 1, 0, 1], [[0, 1, 0]], [[0, 1, 0, 1]], B=10)


# --------------------------------------------------------------------------- #
# CLI: аргументы и таблица H5
# --------------------------------------------------------------------------- #


def test_evolve_requires_an_axis_and_a_variant() -> None:
    """Оси эволюционируют раздельно — прогон без оси не имеет смысла."""
    with pytest.raises(SystemExit):
        run_gepa.parse_args(["--mode", "evolve", "--data", "d.jsonl"])


def test_evolve_defaults_to_the_reworked_protocol() -> None:
    args = run_gepa.parse_args(
        ["--mode", "evolve", "--data", "d.jsonl", "--axis", "faithfulness", "--variant", "plain"]
    )
    assert args.auto == "medium"  # был light
    assert args.pareto_size == 300  # был val_size 30
    assert args.train_size == 300


def test_h5_arms_must_be_paired_by_seed() -> None:
    with pytest.raises(SystemExit):
        run_gepa.parse_args(
            ["--mode", "h5", "--data", "d.jsonl", "--h5-markers", "a.jsonl", "--h5-plain", "b.jsonl",
             "--h5-plain", "c.jsonl"]
        )


def test_h5_table_states_the_verdict_and_the_stop_rule() -> None:
    verdict = h5_decision(_paired(0.0216, -0.037, 0.100), n_seeds=3, n_cases=1486)
    table = run_gepa.h5_table(
        verdict,
        [
            ("markers", 0, Path("results/markers_s0/scores.jsonl"), 0.6412),
            ("plain", 0, Path("results/plain_s0/scores.jsonl"), 0.6196),
        ],
    )
    assert "| markers | 0 | `results/markers_s0/scores.jsonl` | 0.6412 |" in table
    assert "Δ = **+0.0216**" in table
    assert "95% ДИ = [-0.0370, +0.1000]" in table
    assert H5_UNTESTED in table
    assert "верхней границе 95% ДИ ниже нуля" in table
