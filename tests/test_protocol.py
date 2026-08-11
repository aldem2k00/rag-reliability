"""Инварианты протокола оценки: изоляция фолда, учёт исключённых, безопасность выражений."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from rag_reliability.evaluation.protocol import (
    CVResult,
    LeakageError,
    _check_fold_isolation,
    build_report,
    check_corpus_hash,
    compare_runs,
    compile_score_expr,
    default_score_fn,
    evaluate_cv,
    evaluate_cv_labeled,
    faith_score_fn,
    fit_threshold,
    grid_macro_f1,
    has_axis_scores,
    load_folds,
    mean_macro_f1_over_repeats,
    metric_with_ci,
    rel_score_fn,
    threshold_fit_apply,
)
from rag_reliability.schema import Prediction, RagSample
from rag_reliability.splits import sha256_file
from rag_reliability.thresholds import macro_f1_binary, unit_interval_grid

CHEAP = {"bootstrap_b": 200, "null_trials": 20}
CHEAP_CLI = ["--bootstrap-B", "200", "--null-trials", "20"]


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "evaluate_cv_script", Path(__file__).parents[1] / "scripts" / "evaluate_cv.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_cv_script"] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


# --------------------------------------------------------------------------- #
# Фикстуры
# --------------------------------------------------------------------------- #


def make_sample(index: int, faithfulness: int, relevance: int) -> RagSample:
    return RagSample(
        id=f"case_{index:04d}",
        question="вопрос",
        context="контекст",
        answer="ответ",
        faithfulness=faithfulness,
        relevance=relevance,
    )


def make_prediction(sample_id: str, p_faith: float, p_rel: float) -> Prediction:
    return Prediction(
        id=sample_id,
        faithfulness_pred=0,
        relevance_pred=0,
        scores={"m3.p_faith": p_faith, "m3.p_rel": p_rel},
    )


def make_folds(
    assignment: dict[str, list[int]],
    *,
    n_folds: int = 5,
    n_repeats: int = 1,
    sha256: str = "0" * 64,
    n: int | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "corpus": {"path": "data/corpus.jsonl", "sha256": sha256, "n": n or len(assignment)},
        "config": {"n_folds": n_folds, "n_repeats": n_repeats},
        "assignment": assignment,
    }


def round_robin(ids: list[str], n_folds: int = 5, n_repeats: int = 1) -> dict[str, list[int]]:
    return {
        sample_id: [(index + repeat) % n_folds for repeat in range(n_repeats)]
        for index, sample_id in enumerate(ids)
    }


def separable_corpus(n: int = 100) -> tuple[list[RagSample], list[Prediction]]:
    """Скор ``p_faith * p_rel`` растёт вместе с меткой: задача решаемая, но не тривиальная."""
    rng = np.random.default_rng(0)
    samples: list[RagSample] = []
    predictions: list[Prediction] = []
    for index in range(n):
        label = int(index % 3 != 0)
        score = float(np.clip(rng.normal(0.75 if label else 0.35, 0.12), 0.01, 0.99))
        samples.append(make_sample(index, label, label))
        predictions.append(make_prediction(f"case_{index:04d}", score, 0.9))
    return samples, predictions


# --------------------------------------------------------------------------- #
# Подбор порога
# --------------------------------------------------------------------------- #


def test_grid_macro_f1_matches_reference_loop() -> None:
    """Векторизация обязана совпадать с поэлементной формулой метрики."""
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, size=60)
    scores = rng.random(60)
    grid = unit_interval_grid(0.05)

    vectorized = grid_macro_f1(y, scores, grid)
    reference = [macro_f1_binary(y, (scores >= t).astype(int)) for t in grid]

    assert vectorized == pytest.approx(reference, abs=1e-12)


def test_fit_threshold_prefers_the_lowest_of_equally_good_thresholds() -> None:
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    fit = fit_threshold(y, scores, grid_step=0.01)

    assert fit.train_f1 == pytest.approx(1.0)
    assert fit.threshold == pytest.approx(0.21, abs=1e-9)


def test_fit_threshold_rejects_an_empty_training_fold() -> None:
    with pytest.raises(ValueError, match="empty training fold"):
        fit_threshold(np.array([], dtype=int), np.array([]), grid_step=0.01)


# --------------------------------------------------------------------------- #
# Утечка порога
# --------------------------------------------------------------------------- #


def test_threshold_applied_to_a_fold_is_fitted_without_that_fold() -> None:
    """Фолд 0 размечен наоборот: его собственный оптимум резко отличается.

    Если бы порог подбирался с участием фолда 0, он сдвинулся бы к его оптимуму;
    проверяем точное совпадение с порогом, подобранным по остальным фолдам.
    """
    samples: list[RagSample] = []
    predictions: list[Prediction] = []
    assignment: dict[str, list[int]] = {}
    for index in range(100):
        fold = index % 5
        sample_id = f"case_{index:04d}"
        assignment[sample_id] = [fold]
        if fold == 0:
            # Инвертированный фолд: высокие скоры — негативы.
            score = 0.9 if index % 2 else 0.1
            label = 0 if index % 2 else 1
        else:
            score = 0.9 if index % 2 else 0.1
            label = 1 if index % 2 else 0
        samples.append(make_sample(index, label, 1))
        predictions.append(make_prediction(sample_id, score, 1.0))

    folds = make_folds(assignment, n_folds=5, n_repeats=1)
    result = evaluate_cv(samples, predictions, folds)

    index_by_id = {sample.id: position for position, sample in enumerate(samples)}
    y = np.array([sample.reliable for sample in samples])
    scores = np.array([default_score_fn(prediction) for prediction in predictions])
    outside = np.array([index_by_id[key] for key, folds_ in assignment.items() if folds_[0] != 0])
    inside = np.array([index_by_id[key] for key, folds_ in assignment.items() if folds_[0] == 0])

    expected = fit_threshold(y[outside], scores[outside], 0.01)
    own = fit_threshold(y[inside], scores[inside], 0.01)

    assert result.thresholds[0] == pytest.approx(expected.threshold)
    assert result.thresholds[0] != pytest.approx(own.threshold)


def test_oof_predictions_of_a_fold_use_only_that_folds_threshold() -> None:
    samples, predictions = separable_corpus(60)
    ids = [sample.id for sample in samples]
    folds = make_folds(round_robin(ids, n_folds=3), n_folds=3)

    result = evaluate_cv(samples, predictions, folds)

    scores = np.array([default_score_fn(prediction) for prediction in predictions])
    fold_of = np.array([folds["assignment"][sample_id][0] for sample_id in ids])
    for fold, threshold in enumerate(result.thresholds):
        mask = fold_of == fold
        assert np.array_equal(
            result.oof_pred_by_repeat[mask, 0], (scores[mask] >= threshold).astype(int)
        )


def test_out_of_fold_metric_is_below_the_in_sample_optimum() -> None:
    """Оптимизм in-sample порога — то, ради чего протокол переписан."""
    samples, predictions = separable_corpus(150)
    ids = [sample.id for sample in samples]
    folds = make_folds(round_robin(ids, n_repeats=3), n_repeats=3)

    result = evaluate_cv(samples, predictions, folds)

    scores = np.array([default_score_fn(prediction) for prediction in predictions])
    in_sample = fit_threshold(result.y, scores, 0.01)
    assert mean_macro_f1_over_repeats(result.y, result.oof_pred_by_repeat) <= in_sample.train_f1


def test_fold_isolation_failure_is_raised_not_warned() -> None:
    samples, predictions = separable_corpus(20)
    ids = [sample.id for sample in samples]
    assignment = round_robin(ids, n_folds=2)
    folds = make_folds(assignment, n_folds=2)
    # Ломаем протокол: один и тот же кейс объявлен в обоих фолдах сразу нельзя,
    # поэтому эмулируем вырождение — все кейсы в одном фолде.
    folds["assignment"] = {sample_id: [0] for sample_id in ids}

    with pytest.raises(ValueError, match="one fold"):
        evaluate_cv(samples, predictions, folds)


# --------------------------------------------------------------------------- #
# fit_fn
# --------------------------------------------------------------------------- #


def test_fit_fn_sees_only_training_indices() -> None:
    samples, predictions = separable_corpus(50)
    ids = [sample.id for sample in samples]
    assignment = round_robin(ids)
    folds = make_folds(assignment)
    position = {sample_id: index for index, sample_id in enumerate(ids)}
    seen: list[set[int]] = []

    def fit_fn(train_predictions, y_train):
        seen.append({position[prediction.id] for prediction in train_predictions})
        assert len(y_train) == len(train_predictions)
        return lambda batch: np.array([default_score_fn(p) for p in batch])

    result = evaluate_cv(samples, predictions, folds, score_fn=default_score_fn, fit_fn=fit_fn)

    assert len(seen) == 5
    for fold, train_indices in enumerate(seen):
        held_out = {position[key] for key, value in assignment.items() if value[0] == fold}
        assert train_indices.isdisjoint(held_out)
        assert train_indices | held_out == set(range(len(ids)))
    assert len(result.thresholds) == 5


def test_fit_fn_returning_a_non_callable_is_rejected() -> None:
    samples, predictions = separable_corpus(20)
    folds = make_folds(round_robin([sample.id for sample in samples], n_folds=2), n_folds=2)

    with pytest.raises(TypeError, match="callable model"):
        evaluate_cv(samples, predictions, folds, fit_fn=lambda batch, y: "not a model")


def test_score_fn_with_a_second_required_argument_is_rejected() -> None:
    """Метки нельзя передать в score_fn: у него ровно один аргумент."""
    samples, predictions = separable_corpus(20)
    folds = make_folds(round_robin([sample.id for sample in samples], n_folds=2), n_folds=2)

    with pytest.raises(ValueError, match="exactly one argument"):
        evaluate_cv(samples, predictions, folds, score_fn=lambda pred, y: 0.5)


# --------------------------------------------------------------------------- #
# Исключённые кейсы и целостность входа
# --------------------------------------------------------------------------- #


def test_cases_missing_from_the_assignment_are_excluded_and_counted() -> None:
    samples, predictions = separable_corpus(30)
    ids = [sample.id for sample in samples]
    assignment = round_robin(ids[:25])
    folds = make_folds(assignment, n=30)

    result = evaluate_cv(samples, predictions, folds)

    assert result.n_excluded == 5
    assert len(result.y) == 25
    assert set(result.ids) == set(ids[:25])


def test_missing_prediction_is_an_error_not_a_default() -> None:
    samples, predictions = separable_corpus(20)
    folds = make_folds(round_robin([sample.id for sample in samples], n_folds=2), n_folds=2)

    with pytest.raises(ValueError, match="Missing predictions"):
        evaluate_cv(samples, predictions[:-1], folds)


def test_prediction_without_the_expected_score_key_raises() -> None:
    samples, _ = separable_corpus(10)
    predictions = [
        Prediction(id=sample.id, faithfulness_pred=0, relevance_pred=0, scores={"m3.p_faith": 0.5})
        for sample in samples
    ]
    folds = make_folds(round_robin([sample.id for sample in samples], n_folds=2), n_folds=2)

    with pytest.raises(ValueError, match="p_faith"):
        evaluate_cv(samples, predictions, folds)


def test_ambiguous_method_prefix_raises_instead_of_guessing() -> None:
    prediction = Prediction(
        id="case_0000",
        faithfulness_pred=0,
        relevance_pred=0,
        scores={"m3.p_faith": 0.5, "m3.p_rel": 0.5, "surf.p_faith": 0.1, "surf.p_rel": 0.2},
    )
    with pytest.raises(ValueError, match="several method prefixes"):
        default_score_fn(prediction)


def test_scores_outside_the_unit_interval_are_rejected() -> None:
    samples, _ = separable_corpus(10)
    predictions = [make_prediction(sample.id, 2.0, 1.0) for sample in samples]
    folds = make_folds(round_robin([sample.id for sample in samples], n_folds=2), n_folds=2)

    with pytest.raises(ValueError, match="unit interval"):
        evaluate_cv(samples, predictions, folds)


def test_folds_built_on_another_corpus_fail_loudly() -> None:
    samples, predictions = separable_corpus(10)
    folds = make_folds({"other_0001": [0], "other_0002": [1]}, n_folds=2)

    with pytest.raises(ValueError, match="different corpus"):
        evaluate_cv(samples, predictions, folds)


# --------------------------------------------------------------------------- #
# folds.json
# --------------------------------------------------------------------------- #


def test_corpus_sha256_mismatch_is_an_error(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"id": "case_0000"}\n', encoding="utf-8")
    folds = make_folds({"case_0000": [0]}, sha256="f" * 64)

    with pytest.raises(ValueError, match="sha256 mismatch"):
        check_corpus_hash(folds, corpus)

    folds["corpus"]["sha256"] = sha256_file(corpus)
    assert check_corpus_hash(folds, corpus) == sha256_file(corpus)


def test_load_folds_rejects_a_foreign_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "folds.json"
    payload = make_folds({"case_0000": [0]})
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_folds(path)


# --------------------------------------------------------------------------- #
# --score-expr
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "m3.p_faith.__class__",
        "abs(m3.p_faith)",
        "[p for p in m3.p_faith]",
        "m3.p_faith if m3.p_rel else 0",
        "open('/etc/passwd')",
        "m3.p_faith ** 2",
        "lambda: 1",
    ],
)
def test_score_expression_rejects_anything_but_arithmetic(expression: str) -> None:
    with pytest.raises(ValueError):
        compile_score_expr(expression)


def test_score_expression_evaluates_arithmetic_over_score_keys() -> None:
    expression = compile_score_expr("(m3.p_faith * m3.p_rel + 0.5 * surf.p_faith) / 2")
    prediction = Prediction(
        id="case_0000",
        faithfulness_pred=0,
        relevance_pred=0,
        scores={"m3.p_faith": 0.8, "m3.p_rel": 0.5, "surf.p_faith": 0.2},
    )

    assert expression(prediction) == pytest.approx((0.8 * 0.5 + 0.5 * 0.2) / 2)
    assert set(expression.keys) == {"m3.p_faith", "m3.p_rel", "surf.p_faith"}


def test_score_expression_reports_a_missing_key_by_case_id() -> None:
    expression = compile_score_expr("m6.entropy")
    prediction = make_prediction("case_0042", 0.5, 0.5)

    with pytest.raises(ValueError, match="case_0042"):
        expression(prediction)


# --------------------------------------------------------------------------- #
# Отчёт
# --------------------------------------------------------------------------- #


def _report(samples, predictions, folds, **kwargs):
    primary = evaluate_cv(samples, predictions, folds)
    axes = {
        "faithfulness_f1_macro": evaluate_cv_labeled(
            samples,
            predictions,
            folds,
            score_fn=faith_score_fn,
            label_fn=lambda sample: sample.faithfulness,
        ),
        "relevance_f1_macro": evaluate_cv_labeled(
            samples,
            predictions,
            folds,
            score_fn=rel_score_fn,
            label_fn=lambda sample: sample.relevance,
        ),
    }
    return build_report(
        method="test",
        variant="synthetic",
        primary=primary,
        axes=axes,
        predictions=predictions,
        protocol={"folds": "memory", "n_evaluated": len(primary.ids)},
        **{**CHEAP, **kwargs},
    )


def test_primary_is_one_macro_f1_over_the_glued_oof_decision() -> None:
    """Карточка §1: метрика считается ОДИН раз по склеенным OOF-решениям."""
    samples, predictions = separable_corpus(80)
    folds = make_folds(round_robin([s.id for s in samples], n_repeats=3), n_repeats=3)
    result = evaluate_cv(samples, predictions, folds)

    metric = metric_with_ci(result, **CHEAP)

    assert metric.value == pytest.approx(macro_f1_binary(result.y, result.oof_pred))
    assert metric.ci95[0] <= metric.value <= metric.ci95[1]
    assert metric.null_percentile is not None


def test_primary_is_not_the_average_of_per_repeat_metrics() -> None:
    """Когда повторы спорят о кейсах, две величины расходятся — и печатать надо склеенную.

    Каждый повтор ловит свою треть негативов и в одиночку выглядит прилично
    (0.625), но негатив, который поймал ровно один повтор из трёх, голосованием
    объявляется надёжным. Склеенное решение вырождается в константу (0.333) —
    именно это и должно попасть в отчёт, потому что именно так метод решает.
    """
    n_cases, n_repeats = 30, 3
    y = np.array([1] * 15 + [0] * 15)
    by_repeat = np.ones((n_cases, n_repeats), dtype=int)
    for repeat in range(n_repeats):
        block = slice(15 + repeat * 5, 20 + repeat * 5)
        by_repeat[block, repeat] = 0

    result = CVResult(
        oof_scores=np.full(n_cases, 0.5),
        oof_pred=(by_repeat.mean(axis=1) >= 0.5).astype(int),
        y=y,
        ids=[f"case_{index:04d}" for index in range(n_cases)],
        per_repeat_f1=[macro_f1_binary(y, by_repeat[:, r]) for r in range(n_repeats)],
        thresholds=[0.5] * (n_repeats * 3),
        n_excluded=0,
        oof_pred_by_repeat=by_repeat,
        fold_matrix=np.array([[index % 3] * n_repeats for index in range(n_cases)]),
    )

    glued = macro_f1_binary(result.y, result.oof_pred)
    averaged = mean_macro_f1_over_repeats(result.y, result.oof_pred_by_repeat)

    assert glued == pytest.approx(1 / 3, abs=1e-6)
    assert averaged == pytest.approx(0.625, abs=1e-6)
    assert metric_with_ci(result, **CHEAP).value == pytest.approx(glued)


def test_null_calibration_refuses_to_stand_in_for_a_fitted_model() -> None:
    """С fit_fn шум обязан проходить ту же процедуру, включая обучение модели."""
    samples, predictions = separable_corpus(40)
    folds = make_folds(round_robin([s.id for s in samples], n_folds=4), n_folds=4)

    def fit_fn(train_predictions, y_train):
        return lambda batch: np.array([default_score_fn(p) for p in batch])

    result = evaluate_cv(samples, predictions, folds, fit_fn=fit_fn)
    assert result.used_fit_fn is True

    with pytest.raises(ValueError, match="fit_apply_fn"):
        metric_with_ci(result, **CHEAP)

    metric = metric_with_ci(result, fit_apply_fn=threshold_fit_apply, **CHEAP)
    assert metric.null_percentile is not None


def test_constant_scores_do_not_clear_the_noise_floor() -> None:
    samples = [make_sample(index, int(index % 3 != 0), 1) for index in range(80)]
    predictions = [make_prediction(sample.id, 0.5, 0.5) for sample in samples]
    folds = make_folds(round_robin([sample.id for sample in samples], n_repeats=2), n_repeats=2)

    metric = metric_with_ci(evaluate_cv(samples, predictions, folds), **CHEAP)

    assert metric.above_noise is False


def test_report_is_deterministic_across_runs() -> None:
    samples, predictions = separable_corpus(80)
    folds = make_folds(round_robin([sample.id for sample in samples], n_repeats=2), n_repeats=2)

    first = _report(samples, predictions, folds)
    second = _report(samples, predictions, folds)

    assert first.model_dump() == second.model_dump()


def test_report_carries_thresholds_and_axis_intervals() -> None:
    samples, predictions = separable_corpus(80)
    folds = make_folds(round_robin([sample.id for sample in samples], n_repeats=2), n_repeats=2)

    report = _report(samples, predictions, folds)

    assert len(report.diagnostics["thresholds_per_fold"]) == 10
    assert set(report.axes) == {"faithfulness_f1_macro", "relevance_f1_macro"}
    for metric in report.axes.values():
        assert metric.ci95[0] <= metric.value <= metric.ci95[1]


def test_comparison_against_a_disjoint_run_is_refused() -> None:
    samples, predictions = separable_corpus(40)
    folds = make_folds(round_robin([sample.id for sample in samples], n_folds=4), n_folds=4)
    primary = evaluate_cv(samples, predictions, folds)
    other = CVResult(
        oof_scores=primary.oof_scores,
        oof_pred=primary.oof_pred,
        y=primary.y,
        ids=[f"foreign_{index}" for index in range(len(primary.ids))],
        per_repeat_f1=primary.per_repeat_f1,
        thresholds=primary.thresholds,
        n_excluded=0,
        oof_pred_by_repeat=primary.oof_pred_by_repeat,
        fold_matrix=primary.fold_matrix,
    )

    with pytest.raises(ValueError, match="no evaluated cases"):
        compare_runs(primary, other, "foreign", bootstrap_b=50)


def test_comparison_of_a_run_with_itself_has_zero_delta() -> None:
    samples, predictions = separable_corpus(40)
    folds = make_folds(round_robin([sample.id for sample in samples], n_folds=4), n_folds=4)
    primary = evaluate_cv(samples, predictions, folds)

    comparison = compare_runs(primary, primary, "self", bootstrap_b=100)

    assert comparison["delta"] == pytest.approx(0.0)
    assert comparison["significant"] is False
    assert comparison["n_common"] == len(primary.ids)


def test_leakage_error_is_a_value_error_subclass() -> None:
    assert issubclass(LeakageError, ValueError)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def write_corpus(path: Path, samples: list[RagSample]) -> Path:
    path.write_text(
        "".join(sample.model_dump_json() + "\n" for sample in samples), encoding="utf-8"
    )
    return path


def write_scores(path: Path, predictions: list[Prediction]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"id": p.id, "scores": p.scores}, ensure_ascii=False) + "\n"
            for p in predictions
        ),
        encoding="utf-8",
    )
    return path


def cli_fixture(tmp_path: Path, n: int = 80, n_repeats: int = 2) -> dict[str, Path]:
    samples, predictions = separable_corpus(n)
    corpus = write_corpus(tmp_path / "corpus.jsonl", samples)
    scores = write_scores(tmp_path / "predictions" / "alfa" / "m3" / "zero_shot" / "scores.jsonl",
                          predictions)
    folds = make_folds(
        round_robin([sample.id for sample in samples], n_repeats=n_repeats),
        n_repeats=n_repeats,
        sha256=sha256_file(corpus),
    )
    folds_path = tmp_path / "folds.json"
    folds_path.write_text(json.dumps(folds), encoding="utf-8")
    return {"corpus": corpus, "scores": scores, "folds": folds_path, "output": tmp_path / "r.json"}


def test_cli_writes_a_valid_report(tmp_path: Path) -> None:
    paths = cli_fixture(tmp_path)

    assert (
        cli.main(
            [
                "--data", str(paths["corpus"]),
                "--folds", str(paths["folds"]),
                "--scores", str(paths["scores"]),
                "--score-expr", "m3.p_faith * m3.p_rel",
                "--output", str(paths["output"]),
                *CHEAP_CLI,
            ]
        )
        == 0
    )

    payload = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert payload["method"] == "m3" and payload["variant"] == "zero_shot"
    assert payload["primary"]["ci95"][0] <= payload["primary"]["value"]
    assert payload["primary"]["null_percentile"] is not None
    assert payload["protocol"]["n_evaluated"] == 80
    assert payload["protocol"]["corpus_sha256"] == sha256_file(paths["corpus"])


def test_cli_is_deterministic(tmp_path: Path) -> None:
    """Два прогона с тем же folds.json дают идентичный report.json."""
    paths = cli_fixture(tmp_path)
    argv = [
        "--data", str(paths["corpus"]),
        "--folds", str(paths["folds"]),
        "--scores", str(paths["scores"]),
        "--output", str(paths["output"]),
        *CHEAP_CLI,
    ]

    cli.main(argv)
    first = paths["output"].read_text(encoding="utf-8")
    cli.main(argv)

    assert paths["output"].read_text(encoding="utf-8") == first


def test_cli_refuses_a_corpus_that_does_not_match_the_folds(tmp_path: Path) -> None:
    paths = cli_fixture(tmp_path)
    folds = json.loads(paths["folds"].read_text(encoding="utf-8"))
    folds["corpus"]["sha256"] = "f" * 64
    paths["folds"].write_text(json.dumps(folds), encoding="utf-8")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        cli.main(
            [
                "--data", str(paths["corpus"]),
                "--folds", str(paths["folds"]),
                "--scores", str(paths["scores"]),
                "--output", str(paths["output"]),
                *CHEAP_CLI,
            ]
        )


def test_cli_refuses_a_partial_artifact(tmp_path: Path) -> None:
    """Спека §2.2: частичный артефакт не участвует в CV до полного прогона."""
    paths = cli_fixture(tmp_path)
    rows = paths["scores"].read_text(encoding="utf-8").splitlines()[:60]
    paths["scores"].write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="is partial"):
        cli.main(
            [
                "--data", str(paths["corpus"]),
                "--folds", str(paths["folds"]),
                "--scores", str(paths["scores"]),
                "--output", str(paths["output"]),
                *CHEAP_CLI,
            ]
        )

    assert not paths["output"].exists()


def test_cli_refuses_a_partial_comparison_artifact(tmp_path: Path) -> None:
    """Отказ обязан распространяться и на --compare, иначе Δ считается по подвыборке."""
    paths = cli_fixture(tmp_path)
    rival = write_scores(
        tmp_path / "predictions" / "alfa" / "baselines" / "surface" / "scores.jsonl",
        [
            Prediction(
                id=f"case_{index:04d}",
                faithfulness_pred=0,
                relevance_pred=0,
                scores={"surf.p_faith": 0.5, "surf.p_rel": 0.5},
            )
            for index in range(50)
        ],
    )

    with pytest.raises(ValueError, match="is partial"):
        cli.main(
            [
                "--data", str(paths["corpus"]),
                "--folds", str(paths["folds"]),
                "--scores", str(paths["scores"]),
                "--compare", str(rival),
                "--output", str(paths["output"]),
                *CHEAP_CLI,
            ]
        )


def test_cli_counts_cases_absent_from_the_assignment_against_the_whole_corpus(
    tmp_path: Path,
) -> None:
    """n_excluded — про корпус, а не про покрытую часть: иначе он занижен."""
    paths = cli_fixture(tmp_path, n=80)
    folds = json.loads(paths["folds"].read_text(encoding="utf-8"))
    folds["assignment"] = {
        key: value for key, value in folds["assignment"].items() if key < "case_0060"
    }
    paths["folds"].write_text(json.dumps(folds), encoding="utf-8")

    assert (
        cli.main(
            [
                "--data", str(paths["corpus"]),
                "--folds", str(paths["folds"]),
                "--scores", str(paths["scores"]),
                "--output", str(paths["output"]),
                *CHEAP_CLI,
            ]
        )
        == 0
    )

    protocol = json.loads(paths["output"].read_text(encoding="utf-8"))["protocol"]
    assert protocol["n_corpus"] == 80
    assert protocol["n_evaluated"] == 60
    assert protocol["n_excluded"] == 20


def test_cli_rejects_an_injected_score_expression(tmp_path: Path) -> None:
    paths = cli_fixture(tmp_path)

    with pytest.raises(ValueError, match="Call"):
        cli.main(
            [
                "--data", str(paths["corpus"]),
                "--folds", str(paths["folds"]),
                "--scores", str(paths["scores"]),
                "--score-expr", "__import__('os').system('id')",
                "--output", str(paths["output"]),
                *CHEAP_CLI,
            ]
        )


def test_cli_records_a_paired_comparison(tmp_path: Path) -> None:
    paths = cli_fixture(tmp_path)
    rival = write_scores(
        tmp_path / "predictions" / "alfa" / "baselines" / "surface" / "scores.jsonl",
        [
            Prediction(
                id=f"case_{index:04d}",
                faithfulness_pred=0,
                relevance_pred=0,
                scores={"surf.p_faith": 0.5, "surf.p_rel": 0.5},
            )
            for index in range(80)
        ],
    )

    cli.main(
        [
            "--data", str(paths["corpus"]),
            "--folds", str(paths["folds"]),
            "--scores", str(paths["scores"]),
            "--compare", str(rival),
            "--output", str(paths["output"]),
            *CHEAP_CLI,
        ]
    )

    payload = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert len(payload["comparisons"]) == 1
    comparison = payload["comparisons"][0]
    assert comparison["vs"] == "baselines/surface"
    assert comparison["n_common"] == 80
    assert set(comparison) >= {"delta", "ci95", "p", "significant"}


def test_cli_infers_method_and_variant_from_the_artifact_path(tmp_path: Path) -> None:
    path = tmp_path / "predictions" / "alfa" / "m6" / "base" / "scores.jsonl"
    assert cli.infer_method_variant(path) == ("m6", "base")


def test_cli_reads_pre_a3_artifacts_without_a_scores_block(tmp_path: Path) -> None:
    path = tmp_path / "val.jsonl"
    path.write_text('{"id": "case_0000", "p_faith": 0.7, "p_rel": 0.4}\n', encoding="utf-8")

    [prediction] = cli.load_scores(path)

    assert prediction.scores == {"legacy.p_faith": 0.7, "legacy.p_rel": 0.4}


def test_cli_rejects_a_scores_row_without_probabilities(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    path.write_text('{"id": "case_0000", "meta": {}}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="neither 'scores' nor"):
        cli.load_scores(path)


def test_cli_reports_an_artifact_from_another_corpus(tmp_path: Path) -> None:
    paths = cli_fixture(tmp_path, n=20)
    foreign = write_scores(
        tmp_path / "predictions" / "alfa" / "m3" / "foreign" / "scores.jsonl",
        [make_prediction(f"organizer_{index:06d}", 0.5, 0.5) for index in range(20)],
    )

    with pytest.raises(ValueError, match="another corpus"):
        cli.main(
            [
                "--data", str(paths["corpus"]),
                "--folds", str(paths["folds"]),
                "--scores", str(foreign),
                "--output", str(paths["output"]),
                *CHEAP_CLI,
            ]
        )


@pytest.mark.parametrize(
    ("train", "test", "message"),
    [
        ([0, 1, 2], [2, 3], "both the fitting"),
        ([], [0, 1], "empty training"),
        ([0, 1], [], "empty held-out"),
        ([0], [1], "!= 4 evaluated"),
    ],
)
def test_fold_isolation_guard_rejects_every_broken_partition(
    train: list[int], test: list[int], message: str
) -> None:
    """Проверка живёт в рантайме: она стережёт будущие правки цикла, а не текущую."""
    with pytest.raises(LeakageError, match=message):
        _check_fold_isolation(np.array(train, dtype=int), np.array(test, dtype=int), 4, 0, 0)


# --------------------------------------------------------------------------- #
# Артефакты с одной осью
# --------------------------------------------------------------------------- #


def single_axis_fixture(tmp_path: Path, n: int = 80) -> dict[str, Path]:
    """Корпус и артефакт пофрагментной верификации: только ось faithfulness.

    Ключи те же, что пишет ``methods/m3/perchunk.py``: пары ``p_faith``/``p_rel``
    в нём нет вовсе, потому что промпт relevance чанков не получает.
    """
    samples, joint = separable_corpus(n)
    predictions = [
        Prediction(
            id=prediction.id,
            faithfulness_pred=0,
            relevance_pred=0,
            scores={
                "m3.max_chunk_score": prediction.scores["m3.p_faith"],
                "m3.mean_chunk_score": prediction.scores["m3.p_faith"] * 0.9,
                "m3.chunk_disagreement": 0.1,
                "m3.n_supporting": 2.0,
                "m3.argmax_chunk": 0.0,
            },
        )
        for prediction in joint
    ]
    corpus = write_corpus(tmp_path / "corpus.jsonl", samples)
    scores = write_scores(
        tmp_path / "predictions" / "alfa" / "m3_judge" / "perchunk" / "scores.jsonl", predictions
    )
    folds_path = tmp_path / "folds.json"
    folds_path.write_text(
        json.dumps(
            make_folds(
                round_robin([sample.id for sample in samples], n_repeats=2),
                n_repeats=2,
                sha256=sha256_file(corpus),
            )
        ),
        encoding="utf-8",
    )
    return {"corpus": corpus, "scores": scores, "folds": folds_path, "output": tmp_path / "r.json"}


def test_has_axis_scores_distinguishes_single_axis_artifacts() -> None:
    assert has_axis_scores(make_prediction("case_0000", 0.7, 0.8))
    assert not has_axis_scores(
        Prediction(
            id="case_0000",
            faithfulness_pred=0,
            relevance_pred=0,
            scores={"m3.max_chunk_score": 0.7, "m3.chunk_disagreement": 0.1},
        )
    )


def test_cli_scores_a_single_axis_artifact_and_omits_axis_diagnostics(tmp_path: Path) -> None:
    """Пофрагментный прогон оценивается, а поосевые F1 не выдумываются.

    До этого CLI падал на поиске пары ``p_faith``/``p_rel``, и артефакт метода,
    у которого одна ось, нельзя было оценить вообще — при том что первичная
    метрика считается по явному ``--score-expr`` и осей не требует.
    """
    paths = single_axis_fixture(tmp_path)

    assert (
        cli.main(
            [
                "--data", str(paths["corpus"]),
                "--folds", str(paths["folds"]),
                "--scores", str(paths["scores"]),
                "--score-expr", "m3.max_chunk_score",
                "--output", str(paths["output"]),
                *CHEAP_CLI,
            ]
        )
        == 0
    )
    report = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert report["axes"] == {}, "ось, которой в артефакте нет, не должна появляться в отчёте"
    assert report["primary"]["value"] > 0.0


def test_cli_keeps_the_axis_that_was_given_explicitly(tmp_path: Path) -> None:
    """Явный --faith-expr считается и для однооосевого артефакта; relevance — нет."""
    paths = single_axis_fixture(tmp_path)

    cli.main(
        [
            "--data", str(paths["corpus"]),
            "--folds", str(paths["folds"]),
            "--scores", str(paths["scores"]),
            "--score-expr", "m3.max_chunk_score",
            "--faith-expr", "m3.max_chunk_score",
            "--output", str(paths["output"]),
            *CHEAP_CLI,
        ]
    )
    report = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert set(report["axes"]) == {"faithfulness_f1_macro"}
