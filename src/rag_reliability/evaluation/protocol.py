"""Оценка по фолдам с вложенным подбором порога.

Единичный holdout с порогом, подобранным на том же val, на котором отчитываются,
даёт систематический оптимизм +0.035…+0.053 и sd = 0.032 по жребию сплита —
больше, чем любая разница между методами проекта. Здесь порог подбирается внутри
каждого фолда на train-части и применяется к held-out; OOF-решения склеиваются,
метрика считается один раз по всем кейсам.

Инварианты изоляции фолда проверяются на каждом фолде каждого повтора в рантайме,
а не только в тестах: тест проверяет реализацию в момент написания, проверка в
коде — каждый прогон, включая будущие ``fit_fn`` и ``score_fn``.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rag_reliability.evaluation.bootstrap import bootstrap_ci, exact_mcnemar, paired_bootstrap
from rag_reliability.evaluation.nullcal import FitApplyFn, null_calibration, percentile_of
from rag_reliability.metrics import degenerate_rate, operational_metrics
from rag_reliability.schema import EvaluationReport, MetricWithCI, Prediction, RagSample
from rag_reliability.splits import sha256_file
from rag_reliability.thresholds import macro_f1_binary, unit_interval_grid

#: Скор, обученный на train-части, обязан быть оптимальным на ней же с точностью
#: до арифметики float; больший зазор означает, что порог пришёл не из train.
OPTIMALITY_TOLERANCE = 1e-12

SCHEMA_VERSION = 1

ScoreFn = Callable[[Prediction], float]
#: ``fit_fn(train_predictions, y_train)`` -> модель, отображающая предсказания в скоры.
FitFn = Callable[[Sequence[Prediction], np.ndarray], Callable[[Sequence[Prediction]], np.ndarray]]


class LeakageError(ValueError):
    """Нарушен инвариант изоляции: порог или модель могли видеть held-out кейсы."""


@dataclass(frozen=True)
class CVResult:
    """Результат кросс-валидации с вложенным подбором порога.

    Первичная метрика считается ОДИН раз по склеенному ``oof_pred`` (карточка B1
    §1). ``per_repeat_f1`` — диагностика разброса по жребию, а не первичное
    число: величины не эквивалентны и на вырожденных данных расходятся на
    десятые доли.

    ``oof_pred_by_repeat``, ``fold_matrix`` и ``used_fit_fn`` — расширение
    относительно карточки: первые два нужны для диагностик и нулевой калибровки,
    третий — чтобы ``metric_with_ci`` не подсунул калибровку от чужой процедуры.
    """

    oof_scores: np.ndarray
    oof_pred: np.ndarray
    y: np.ndarray
    ids: list[str]
    per_repeat_f1: list[float]
    thresholds: list[float]
    n_excluded: int
    oof_pred_by_repeat: np.ndarray
    fold_matrix: np.ndarray
    used_fit_fn: bool = False


# --------------------------------------------------------------------------- #
# Подбор одномерного порога
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ThresholdFitResult:
    """Порог и достигнутое им macro-F1 на данных, по которым он подобран."""

    threshold: float
    train_f1: float


def grid_macro_f1(y: np.ndarray, scores: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """macro-F1 для каждого порога сетки за один проход.

    Поэлементный цикл по сетке внутри нулевой калибровки — это 12 500 подборов на
    прогон; векторизация переводит калибровку из минут в секунды. Формула та же,
    что в :func:`rag_reliability.thresholds.macro_f1_binary`, включая нулевой
    вклад пустого класса.
    """
    positive = y.astype(bool)
    hits = scores[None, :] >= grid[:, None]

    tp_1 = np.count_nonzero(hits & positive, axis=1)
    fp_1 = np.count_nonzero(hits & ~positive, axis=1)
    fn_1 = np.count_nonzero(~hits & positive, axis=1)
    tp_0 = np.count_nonzero(~hits & ~positive, axis=1)

    # Для класса 0 ошибки те же, что для класса 1, только зеркально:
    # fp_0 = ~hits & positive = fn_1, а fn_0 = hits & ~positive = fp_1.
    denominator_1 = 2 * tp_1 + fp_1 + fn_1
    denominator_0 = 2 * tp_0 + fn_1 + fp_1

    f1_1 = np.divide(2 * tp_1, denominator_1, out=np.zeros(len(grid)), where=denominator_1 > 0)
    f1_0 = np.divide(2 * tp_0, denominator_0, out=np.zeros(len(grid)), where=denominator_0 > 0)
    return (f1_0 + f1_1) / 2.0


def fit_threshold(
    y_train: np.ndarray, scores_train: np.ndarray, grid_step: float = 0.01
) -> ThresholdFitResult:
    """Порог, максимизирующий macro-F1 на переданных данных.

    Tie-break — как в ``thresholds.fit_thresholds``: восходящий скан, замена
    только на строгом улучшении, то есть побеждает наименьший порог.
    """
    if len(y_train) == 0:
        raise ValueError("Cannot fit a threshold on an empty training fold")
    grid = unit_interval_grid(grid_step)
    scores = np.asarray(scores_train, dtype=float)
    values = grid_macro_f1(np.asarray(y_train, dtype=int), scores, grid)
    best = int(np.argmax(values))  # первый максимум — наименьший порог
    return ThresholdFitResult(threshold=float(grid[best]), train_f1=float(values[best]))


def mean_macro_f1_over_repeats(y: Any, pred_by_repeat: Any) -> float:
    """Диагностика: среднее OOF macro-F1 по повторам.

    Не первичная метрика. Первичная считается один раз по склеенному ``oof_pred``
    (карточка B1 §1), и эти две величины не эквивалентны: усреднение по повторам
    маскирует случаи, где повторы расходятся в решениях по одному кейсу.
    """
    y_array = np.asarray(y, dtype=int)
    matrix = np.asarray(pred_by_repeat)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    return float(
        np.mean([macro_f1_binary(y_array, matrix[:, r].astype(int)) for r in range(matrix.shape[1])])
    )


# --------------------------------------------------------------------------- #
# Скоры
# --------------------------------------------------------------------------- #


def _score_prefixes(prediction: Prediction) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    for key, value in prediction.scores.items():
        prefix, _, signal = key.partition(".")
        grouped.setdefault(prefix, {})[signal] = value
    return grouped


def _axis_prefixes(prediction: Prediction) -> list[str]:
    """Префиксы методов, у которых в scores есть обе оси."""
    return sorted(
        prefix
        for prefix, signals in _score_prefixes(prediction).items()
        if "p_faith" in signals and "p_rel" in signals
    )


def _resolve_axis_keys(prediction: Prediction) -> tuple[str, str]:
    """Ключи ``<method>.p_faith`` и ``<method>.p_rel`` единственного метода в scores."""
    candidates = _axis_prefixes(prediction)
    if not candidates:
        raise ValueError(
            f"Prediction {prediction.id!r} has no '<method>.p_faith' + '<method>.p_rel' pair "
            f"in scores (keys: {sorted(prediction.scores)[:8]}); pass --score-expr explicitly"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Prediction {prediction.id!r} carries several method prefixes {candidates}; "
            "the reliability score is ambiguous, pass --score-expr explicitly"
        )
    return f"{candidates[0]}.p_faith", f"{candidates[0]}.p_rel"


def has_axis_scores(prediction: Prediction) -> bool:
    """Есть ли в артефакте пара ``p_faith``/``p_rel`` ровно одного метода.

    Не всякий метод даёт обе оси: пофрагментная верификация судит только
    faithfulness (ось relevance чанков не получает по построению) и выдаёт
    ``m3.max_chunk_score`` и соседние фичи. Поосевая диагностика для такого
    артефакта не определена, и вызывающий отличает «оси не посчитаны» от
    «оси посчитаны и вышли нулевыми» до того, как поймает исключение.
    """
    return len(_axis_prefixes(prediction)) == 1


def _require_score(prediction: Prediction, key: str) -> float:
    if key not in prediction.scores:
        raise ValueError(
            f"Prediction {prediction.id!r} has no score {key!r} "
            f"(keys: {sorted(prediction.scores)[:8]})"
        )
    return float(prediction.scores[key])


def default_score_fn(pred: Prediction) -> float:
    """p_reliable = p_faith * p_rel. Один порог вместо двух.

    Две оси дают квадратичное число кандидатов при линейном приросте
    выразительности, а ось relevance у судьи имеет AUC 0.497 — второй порог
    подбирается по шуму.
    """
    faith_key, rel_key = _resolve_axis_keys(pred)
    return _require_score(pred, faith_key) * _require_score(pred, rel_key)


def faith_score_fn(pred: Prediction) -> float:
    """Диагностическая ось faithfulness со своим порогом."""
    faith_key, _ = _resolve_axis_keys(pred)
    return _require_score(pred, faith_key)


def rel_score_fn(pred: Prediction) -> float:
    """Диагностическая ось relevance со своим порогом.

    Опубликованный ``f1_macro_rel`` считался при ``t_rel = 0.01``, то есть был
    метрикой константного предсказателя; здесь порог оси подбирается отдельно.
    """
    _, rel_key = _resolve_axis_keys(pred)
    return _require_score(pred, rel_key)


# --------------------------------------------------------------------------- #
# Безопасное выражение по ключам scores
# --------------------------------------------------------------------------- #

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Name,
    ast.Constant,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.USub,
    ast.UAdd,
)

_KEY_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.")

#: ``<method>.<signal>`` из контракта HANDOFF §7.1; сегмент не начинается с ``_``.
#: Всё остальное с точкой (``m3.p_faith.__class__``) ключом не считается и уходит
#: в разбор как обращение к атрибуту — где его отвергает белый список узлов.
_KEY_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+")


@dataclass(frozen=True)
class ScoreExpression:
    """Выражение по ключам ``Prediction.scores``, разобранное без ``eval``.

    ``eval`` с любым словарём имён остаётся исполнением произвольного кода:
    строка выражения приходит из аргумента командной строки, а отчёты собираются
    автоматикой. Поэтому дерево обходится вручную и допускает только арифметику.
    """

    source: str
    keys: tuple[str, ...]
    _tree: ast.Expression
    _names: Mapping[str, str]

    def __call__(self, pred: Prediction) -> float:
        values = {name: _require_score(pred, key) for name, key in self._names.items()}
        return float(_eval_node(self._tree.body, values, self.source))


def _tokenize_keys(expression: str) -> tuple[str, dict[str, str]]:
    """Заменить ``m3.p_faith`` на плейсхолдер: точка в имени — не атрибут."""
    names: dict[str, str] = {}
    rendered: list[str] = []
    index = 0
    while index < len(expression):
        char = expression[index]
        if char not in _KEY_CHARS:
            rendered.append(char)
            index += 1
            continue
        start = index
        while index < len(expression) and expression[index] in _KEY_CHARS:
            index += 1
        token = expression[start:index]
        if not _KEY_PATTERN.fullmatch(token):
            # Число или что-то, что ключом не является: пусть разбирает ast.parse
            # и отвергает белый список узлов.
            rendered.append(token)
            continue
        name = f"_key{len(names)}"
        names[name] = token
        rendered.append(name)
    return "".join(rendered), names


def compile_score_expr(expression: str) -> ScoreExpression:
    """Разобрать выражение вида ``m3.p_faith * m3.p_rel``.

    Разрешены только имена ключей, числа и ``+ - * / ( )``. Вызовы, атрибуты,
    индексация, comprehension и служебные имена отвергаются.
    """
    if not expression.strip():
        raise ValueError("Score expression is empty")
    rendered, names = _tokenize_keys(expression)
    try:
        tree = ast.parse(rendered, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Cannot parse score expression {expression!r}: {exc}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"Score expression {expression!r} uses {type(node).__name__}, "
                "only score keys, numbers and + - * / ( ) are allowed"
            )
        if isinstance(node, ast.Constant) and not isinstance(node.value, int | float):
            raise ValueError(
                f"Score expression {expression!r} contains a non-numeric constant "
                f"{node.value!r}"
            )
        if isinstance(node, ast.Name) and node.id not in names:
            raise ValueError(
                f"Score expression {expression!r} references {node.id!r}, which is not a "
                "score key of the form '<method>.<signal>'"
            )
    return ScoreExpression(
        source=expression,
        keys=tuple(names[name] for name in sorted(names)),
        _tree=tree,
        _names=names,
    )


def _eval_node(node: ast.AST, values: Mapping[str, float], source: str) -> float:
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, values, source)
        return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, values, source)
        right = _eval_node(node.right, values, source)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0.0:
            raise ValueError(f"Score expression {source!r} divides by zero")
        return left / right
    raise ValueError(f"Score expression {source!r} contains an unsupported node")


# --------------------------------------------------------------------------- #
# folds.json
# --------------------------------------------------------------------------- #


def load_folds(path: str | Path) -> dict[str, Any]:
    """Прочитать ``folds.json`` и проверить структуру, а не только наличие файла."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("schema_version", "corpus", "config", "assignment"):
        if key not in payload:
            raise ValueError(f"folds file {path} has no {key!r} section")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"folds file {path} has schema_version {payload['schema_version']!r}, "
            f"expected {SCHEMA_VERSION}"
        )
    for key in ("n_folds", "n_repeats"):
        if key not in payload["config"]:
            raise ValueError(f"folds file {path} has no config.{key}")
    for key in ("sha256", "n"):
        if key not in payload["corpus"]:
            raise ValueError(f"folds file {path} has no corpus.{key}")
    return payload


def check_corpus_hash(folds: Mapping[str, Any], corpus_path: str | Path) -> str:
    """Ошибка, а не предупреждение: числа с разных корпусов несравнимы."""
    actual = sha256_file(corpus_path)
    recorded = folds["corpus"]["sha256"]
    if actual != recorded:
        raise ValueError(
            f"Corpus sha256 mismatch: folds.json was built on {recorded}, "
            f"{corpus_path} is {actual}. Numbers from different corpora are not comparable."
        )
    return actual


# --------------------------------------------------------------------------- #
# Ядро
# --------------------------------------------------------------------------- #


def _align_predictions(
    samples: Sequence[RagSample], predictions: Sequence[Prediction]
) -> dict[str, Prediction]:
    by_id: dict[str, Prediction] = {}
    for prediction in predictions:
        if prediction.id in by_id:
            raise ValueError(f"Duplicate prediction id: {prediction.id!r}")
        by_id[prediction.id] = prediction
    missing = [sample.id for sample in samples if sample.id not in by_id]
    if missing:
        raise ValueError(
            f"Missing predictions for {len(missing)} sample(s): {missing[:5]}"
        )
    return by_id


def _check_score_fn(score_fn: ScoreFn) -> None:
    """score_fn обязан принимать ровно один аргумент — предсказание.

    Так метки структурно недостижимы: y в него просто нечем передать.
    """
    try:
        signature = inspect.signature(score_fn)
    except (TypeError, ValueError):  # builtins без интроспекции
        return
    required = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(required) != 1:
        raise ValueError(
            f"score_fn must take exactly one argument (the prediction), got {len(required)} "
            f"required parameters: {[p.name for p in required]}. Labels must stay unreachable."
        )


def _apply_score_fn(score_fn: ScoreFn, predictions: Sequence[Prediction]) -> np.ndarray:
    _check_score_fn(score_fn)
    scores = np.array([float(score_fn(prediction)) for prediction in predictions], dtype=float)
    _check_scores(scores, [prediction.id for prediction in predictions])
    return scores


def _check_scores(scores: np.ndarray, ids: Sequence[str]) -> None:
    bad = [
        ids[index]
        for index in np.flatnonzero(~np.isfinite(scores) | (scores < 0.0) | (scores > 1.0))
    ]
    if bad:
        raise ValueError(
            f"{len(bad)} score(s) outside the unit interval or non-finite: {bad[:5]}. "
            "The threshold grid is [0, 1]; rescale the score before evaluating."
        )


def _fold_matrix(
    folds: Mapping[str, Any], ids: Sequence[str], n_folds: int, n_repeats: int
) -> np.ndarray:
    assignment = folds["assignment"]
    matrix = np.array([assignment[sample_id] for sample_id in ids], dtype=int)
    if matrix.ndim != 2 or matrix.shape[1] != n_repeats:
        raise ValueError(
            f"folds.assignment must give {n_repeats} repeats per case, got shape {matrix.shape}"
        )
    if matrix.size and (matrix.min() < 0 or matrix.max() >= n_folds):
        raise ValueError(
            f"folds.assignment has fold numbers outside [0, {n_folds}): "
            f"[{matrix.min()}, {matrix.max()}]"
        )
    for repeat in range(n_repeats):
        if len(np.unique(matrix[:, repeat])) < 2:
            raise ValueError(
                f"Repeat {repeat} puts every evaluated case into one fold; "
                "there is no held-out part to report on"
            )
    return matrix


def _check_fold_isolation(
    train_idx: np.ndarray, test_idx: np.ndarray, n_cases: int, repeat: int, fold: int
) -> None:
    """Порог фолда k обязан быть подобран без единого кейса из фолда k."""
    overlap = np.intersect1d(train_idx, test_idx)
    if overlap.size:
        raise LeakageError(
            f"repeat {repeat} fold {fold}: {overlap.size} case(s) are in both the fitting and "
            f"the held-out part, e.g. indices {overlap[:5].tolist()}"
        )
    if train_idx.size == 0:
        raise LeakageError(f"repeat {repeat} fold {fold}: empty training part")
    if test_idx.size == 0:
        raise LeakageError(f"repeat {repeat} fold {fold}: empty held-out part")
    if train_idx.size + test_idx.size != n_cases:
        raise LeakageError(
            f"repeat {repeat} fold {fold}: train ({train_idx.size}) + test ({test_idx.size}) "
            f"!= {n_cases} evaluated cases"
        )


def _check_threshold_is_train_optimal(
    threshold: float,
    y_train: np.ndarray,
    scores_train: np.ndarray,
    grid_step: float,
    repeat: int,
    fold: int,
) -> None:
    """Применённый порог обязан быть оптимальным на train-части.

    Проверка ловит подмену подборщика: порог, подсмотренный на held-out, почти
    никогда не оптимален на train, и прогон падает вместо тихого оптимизма.
    """
    grid = unit_interval_grid(grid_step)
    values = grid_macro_f1(y_train, scores_train, grid)
    achieved = macro_f1_binary(y_train, (scores_train >= threshold).astype(int))
    if achieved < float(np.max(values)) - OPTIMALITY_TOLERANCE:
        raise LeakageError(
            f"repeat {repeat} fold {fold}: applied threshold {threshold:.4f} scores "
            f"{achieved:.6f} on the training part, below the best {float(np.max(values)):.6f} "
            "reachable there — it was not fitted on the training part alone"
        )


def _fit_model(
    fit_fn: FitFn,
    train_predictions: Sequence[Prediction],
    y_train: np.ndarray,
    test_predictions: Sequence[Prediction],
    repeat: int,
    fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = fit_fn(train_predictions, y_train)
    if not callable(model):
        raise TypeError(
            f"fit_fn must return a callable model, got {type(model).__name__} "
            f"at repeat {repeat} fold {fold}"
        )
    train_scores = np.asarray(model(train_predictions), dtype=float)
    test_scores = np.asarray(model(test_predictions), dtype=float)
    for name, scores, expected in (
        ("train", train_scores, len(train_predictions)),
        ("held-out", test_scores, len(test_predictions)),
    ):
        if scores.shape != (expected,):
            raise ValueError(
                f"repeat {repeat} fold {fold}: model returned {scores.shape} scores for "
                f"{expected} {name} case(s)"
            )
    _check_scores(train_scores, [p.id for p in train_predictions])
    _check_scores(test_scores, [p.id for p in test_predictions])
    return train_scores, test_scores


def _reliable_label(sample: RagSample) -> int:
    return sample.reliable


def evaluate_cv(
    samples: Sequence[RagSample],
    predictions: Sequence[Prediction],
    folds: Mapping[str, Any],
    *,
    score_fn: ScoreFn = default_score_fn,
    fit_fn: FitFn | None = None,
    grid_step: float = 0.01,
) -> CVResult:
    """Оценка по фолдам с вложенным подбором порога (контракт HANDOFF §7.3)."""
    return evaluate_cv_labeled(
        samples,
        predictions,
        folds,
        score_fn=score_fn,
        fit_fn=fit_fn,
        grid_step=grid_step,
        label_fn=_reliable_label,
    )


def evaluate_cv_labeled(
    samples: Sequence[RagSample],
    predictions: Sequence[Prediction],
    folds: Mapping[str, Any],
    *,
    score_fn: ScoreFn = default_score_fn,
    fit_fn: FitFn | None = None,
    grid_step: float = 0.01,
    label_fn: Callable[[RagSample], int] = _reliable_label,
) -> CVResult:
    """Оценка по фолдам с вложенным подбором порога.

    Для каждого повтора r и фолда k:
        train_idx = все кейсы, кроме фолда k
        test_idx  = фолд k
        если fit_fn задан: model = fit_fn(X[train], y[train])
        порог подбирается на train_idx, применяется к test_idx
    OOF-предсказания склеиваются; метрика считается ОДИН раз по всем кейсам.

    Кейсы, отсутствующие в ``folds['assignment']`` (oversized-группы), исключаются
    и учитываются в ``n_excluded`` — не молча.

    ``label_fn`` существует ради диагностических осей: у оси faithfulness свой
    порог, и подбирать его надо под метку оси, а не под ``reliable``. Публичный
    контракт ``evaluate_cv`` от этого не меняется.
    """
    if not samples:
        raise ValueError("Cannot evaluate an empty corpus")
    by_id = _align_predictions(samples, predictions)
    assignment = folds["assignment"]

    evaluated = [sample for sample in samples if sample.id in assignment]
    n_excluded = len(samples) - len(evaluated)
    if not evaluated:
        raise ValueError(
            f"None of the {len(samples)} sample(s) appear in folds.assignment "
            f"({len(assignment)} ids); the folds were built on a different corpus"
        )

    ids = [sample.id for sample in evaluated]
    ordered_predictions = [by_id[sample_id] for sample_id in ids]
    y = np.array([label_fn(sample) for sample in evaluated], dtype=int)
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("label_fn must return binary labels 0 and 1")
    n_folds = int(folds["config"]["n_folds"])
    n_repeats = int(folds["config"]["n_repeats"])
    matrix = _fold_matrix(folds, ids, n_folds, n_repeats)

    base_scores = _apply_score_fn(score_fn, ordered_predictions)

    oof_pred_by_repeat = np.zeros((len(ids), n_repeats), dtype=int)
    oof_scores_by_repeat = np.zeros((len(ids), n_repeats), dtype=float)
    thresholds: list[float] = []

    for repeat in range(n_repeats):
        repeat_folds = matrix[:, repeat]
        for fold in sorted(np.unique(repeat_folds).tolist()):
            test_idx = np.flatnonzero(repeat_folds == fold)
            train_idx = np.flatnonzero(repeat_folds != fold)
            _check_fold_isolation(train_idx, test_idx, len(ids), repeat, fold)

            y_train = y[train_idx]  # fancy indexing копирует: срез train изолирован
            if fit_fn is None:
                train_scores = base_scores[train_idx]
                test_scores = base_scores[test_idx]
            else:
                train_scores, test_scores = _fit_model(
                    fit_fn,
                    [ordered_predictions[index] for index in train_idx],
                    y_train,
                    [ordered_predictions[index] for index in test_idx],
                    repeat,
                    fold,
                )

            fit = fit_threshold(y_train, train_scores, grid_step)
            _check_threshold_is_train_optimal(
                fit.threshold, y_train, train_scores, grid_step, repeat, fold
            )
            thresholds.append(fit.threshold)
            oof_scores_by_repeat[test_idx, repeat] = test_scores
            oof_pred_by_repeat[test_idx, repeat] = (test_scores >= fit.threshold).astype(int)

    per_repeat_f1 = [
        macro_f1_binary(y, oof_pred_by_repeat[:, repeat]) for repeat in range(n_repeats)
    ]
    return CVResult(
        oof_scores=oof_scores_by_repeat.mean(axis=1),
        oof_pred=(oof_pred_by_repeat.mean(axis=1) >= 0.5).astype(int),
        y=y,
        ids=ids,
        per_repeat_f1=per_repeat_f1,
        thresholds=thresholds,
        n_excluded=n_excluded,
        oof_pred_by_repeat=oof_pred_by_repeat,
        fold_matrix=matrix,
        used_fit_fn=fit_fn is not None,
    )


def threshold_fit_apply(
    y_train: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    grid_step: float,
) -> np.ndarray:
    """Процедура фолда в форме, которую принимает ``nullcal.null_calibration``.

    Нулевая калибровка обязана прогонять чистый шум через ту же процедуру, что и
    реальные скоры; иначе перцентиль описывает чужой протокол.
    """
    fit = fit_threshold(np.asarray(y_train, dtype=int), np.asarray(train_scores, float), grid_step)
    return (np.asarray(test_scores, dtype=float) >= fit.threshold).astype(int)


# --------------------------------------------------------------------------- #
# Сборка отчёта
# --------------------------------------------------------------------------- #


def metric_with_ci(
    result: CVResult,
    *,
    bootstrap_b: int = 10_000,
    null_trials: int = 500,
    grid_step: float = 0.01,
    seed: int = 0,
    fit_apply_fn: FitApplyFn | None = None,
) -> MetricWithCI:
    """Точечная оценка, 95% ДИ и перцентиль нулевого распределения.

    Точка и ДИ считаются по склеенному ``oof_pred`` — по тому же вектору решений,
    который метод выдал бы в проде, и ровно один раз по всем кейсам.

    ``fit_apply_fn`` обязателен, если ``evaluate_cv`` работал с ``fit_fn``: шум
    должен проходить через ту же процедуру, что реальные скоры, а обучение модели
    подбором порога не воспроизводится. Подставлять ``threshold_fit_apply`` за
    вызывающего нельзя — перцентиль описывал бы более простую процедуру, чем та,
    что дала число, то есть завышал бы уверенность.

    Остаточное расхождение, которое нельзя устранить, не меняя ``nullcal``:
    ``null_calibration`` усредняет OOF macro-F1 по повторам, а первичная метрика
    склеивает решения повторов голосованием. Процедура уровня фолда — подбор
    порога на train и применение к held-out — совпадает; различается только
    способ свести повторы. Это записано в ``diagnostics.null_aggregation``.
    """
    if result.used_fit_fn and fit_apply_fn is None:
        raise ValueError(
            "This CVResult was produced with a fit_fn, so the noise floor must fit the same "
            "model on random data; pass fit_apply_fn reproducing it. Calibrating a stack or "
            "an encoder against a bare threshold search understates the noise floor."
        )
    boot = bootstrap_ci(result.y, result.oof_pred, macro_f1_binary, B=bootstrap_b, seed=seed)
    null = null_calibration(
        result.y,
        result.fold_matrix,
        n_trials=null_trials,
        grid_step=grid_step,
        seed=seed,
        fit_apply_fn=fit_apply_fn or threshold_fit_apply,
    )
    percentile = percentile_of(boot.point, null)
    return MetricWithCI(
        value=boot.point,
        ci95=(boot.lo, boot.hi),
        null_percentile=percentile,
        above_noise=percentile > 95.0,
    )


def compare_runs(
    primary: CVResult,
    other: CVResult,
    name: str,
    *,
    bootstrap_b: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Парный бутстрэп против другого прогона на общих кейсах.

    Пересечение считается явно: сравнивать прогоны с разным покрытием как
    парные — самый дешёвый способ получить значимость из разного состава кейсов.
    """
    common = [sample_id for sample_id in primary.ids if sample_id in set(other.ids)]
    if not common:
        raise ValueError(f"Comparison {name!r} shares no evaluated cases with the primary run")
    primary_index = {sample_id: index for index, sample_id in enumerate(primary.ids)}
    other_index = {sample_id: index for index, sample_id in enumerate(other.ids)}
    left = np.array([primary_index[sample_id] for sample_id in common])
    right = np.array([other_index[sample_id] for sample_id in common])

    y = primary.y[left]
    if not np.array_equal(y, other.y[right]):
        raise ValueError(f"Comparison {name!r} disagrees with the primary run on gold labels")

    paired = paired_bootstrap(
        y,
        primary.oof_pred[left],
        other.oof_pred[right],
        macro_f1_binary,
        B=bootstrap_b,
        seed=seed,
    )
    mcnemar = exact_mcnemar(y, primary.oof_pred[left], other.oof_pred[right])
    return {
        "vs": name,
        "n_common": len(common),
        "delta": paired.delta,
        "ci95": list(paired.ci95),
        "p": paired.p,
        "significant": paired.significant,
        "mcnemar": {"b": mcnemar.b, "c": mcnemar.c, "p": mcnemar.p},
    }


def build_report(
    *,
    method: str,
    variant: str,
    primary: CVResult,
    axes: Mapping[str, CVResult],
    predictions: Sequence[Prediction],
    protocol: Mapping[str, Any],
    comparisons: Sequence[Mapping[str, Any]] = (),
    bootstrap_b: int = 10_000,
    null_trials: int = 500,
    grid_step: float = 0.01,
    seed: int = 0,
    fit_apply_fn: FitApplyFn | None = None,
) -> EvaluationReport:
    """Собрать ``EvaluationReport``: число без ДИ и перцентиля шума не публикуется."""
    primary_metric = metric_with_ci(
        primary,
        bootstrap_b=bootstrap_b,
        null_trials=null_trials,
        grid_step=grid_step,
        seed=seed,
        fit_apply_fn=fit_apply_fn,
    )
    axis_metrics = {
        name: metric_with_ci(
            result,
            bootstrap_b=bootstrap_b,
            null_trials=null_trials,
            grid_step=grid_step,
            seed=seed,
        )
        for name, result in axes.items()
    }

    threshold = float(np.median(primary.thresholds))
    operational = operational_metrics(primary.y, primary.oof_scores, threshold)

    faith = axes.get("faithfulness_f1_macro")
    relevance = axes.get("relevance_f1_macro")
    if faith is not None and relevance is not None:
        decisions = [
            Prediction(
                id=sample_id,
                faithfulness_pred=int(faith_pred),
                relevance_pred=int(rel_pred),
            )
            for sample_id, faith_pred, rel_pred in zip(
                primary.ids, faith.oof_pred, relevance.oof_pred, strict=True
            )
        ]
        degenerate = degenerate_rate(decisions)
    else:
        degenerate = degenerate_rate(
            [
                Prediction(id=sample_id, faithfulness_pred=int(pred), relevance_pred=int(pred))
                for sample_id, pred in zip(primary.ids, primary.oof_pred, strict=True)
            ]
        )

    invalid = [prediction for prediction in predictions if prediction.invalid_output]
    diagnostics = {
        "degenerate": degenerate,
        "invalid_output_rate": len(invalid) / len(predictions) if predictions else 0.0,
        "thresholds_per_fold": [float(value) for value in primary.thresholds],
        "threshold_sd": float(np.std(primary.thresholds)),
        "operating_threshold": threshold,
        "per_repeat_f1": [float(value) for value in primary.per_repeat_f1],
        "mean_per_repeat_f1": mean_macro_f1_over_repeats(primary.y, primary.oof_pred_by_repeat),
        "sd_across_repeats": float(np.std(primary.per_repeat_f1)),
        # Процедура уровня фолда у шума и у метрики одна; повторы шум сводит
        # усреднением, метрика — голосованием. Расхождение видно из соседнего
        # mean_per_repeat_f1 и не прячется за общим числом.
        "null_aggregation": "mean of per-repeat OOF macro-F1 (nullcal contract)",
        "primary_aggregation": "single macro-F1 over the glued OOF decision",
        "bootstrap_B": bootstrap_b,
        "null_trials": null_trials,
        "seed": seed,
        "grid_step": grid_step,
    }

    return EvaluationReport(
        method=method,
        variant=variant,
        protocol=dict(protocol),
        primary=primary_metric,
        axes=axis_metrics,
        operational=operational,
        diagnostics=diagnostics,
        comparisons=[dict(comparison) for comparison in comparisons],
    )


# --------------------------------------------------------------------------- #
# Режим совместимости: единичный holdout и две сетки порогов
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LegacySplitMetrics:
    """Опубликованный набор чисел одного сплита."""

    f1_macro_reliable: float
    f1_macro_faith: float
    f1_macro_rel: float
    n: int


@dataclass(frozen=True)
class LegacyHoldoutResult:
    """Воспроизведение старой схемы: порог с val, отчёт на val и test."""

    t_faith: float
    t_rel: float
    val: LegacySplitMetrics
    test: LegacySplitMetrics


def _legacy_axis_scores(
    predictions: Sequence[Prediction], faith_fn: ScoreFn, rel_fn: ScoreFn
) -> tuple[np.ndarray, np.ndarray]:
    faith = np.array([float(faith_fn(p)) for p in predictions], dtype=float)
    rel = np.array([float(rel_fn(p)) for p in predictions], dtype=float)
    return faith, rel


def _legacy_metrics(
    samples: Sequence[RagSample],
    faith: np.ndarray,
    rel: np.ndarray,
    t_faith: float,
    t_rel: float,
) -> LegacySplitMetrics:
    y_faith = np.array([s.faithfulness for s in samples], dtype=int)
    y_rel = np.array([s.relevance for s in samples], dtype=int)
    faith_pred = (faith >= t_faith).astype(int)
    rel_pred = (rel >= t_rel).astype(int)
    return LegacySplitMetrics(
        f1_macro_reliable=macro_f1_binary(y_faith & y_rel, faith_pred & rel_pred),
        f1_macro_faith=macro_f1_binary(y_faith, faith_pred),
        f1_macro_rel=macro_f1_binary(y_rel, rel_pred),
        n=len(samples),
    )


def evaluate_legacy_holdout(
    val_samples: Sequence[RagSample],
    val_predictions: Sequence[Prediction],
    test_samples: Sequence[RagSample],
    test_predictions: Sequence[Prediction],
    *,
    faith_fn: ScoreFn = faith_score_fn,
    rel_fn: ScoreFn = rel_score_fn,
    grid_step: float = 0.01,
) -> LegacyHoldoutResult:
    """Старый протокол: сетка ``t_faith × t_rel`` на val, отчёт на том же val и на test.

    Существует только ради регресс-теста: он показывает, что новый контур изменил
    протокол, а не арифметику. В отчётах не использовать — порог здесь подобран на
    данных, на которых печатается число.
    """
    val_by_id = _align_predictions(val_samples, val_predictions)
    test_by_id = _align_predictions(test_samples, test_predictions)
    val_ordered = [val_by_id[s.id] for s in val_samples]
    test_ordered = [test_by_id[s.id] for s in test_samples]

    val_faith, val_rel = _legacy_axis_scores(val_ordered, faith_fn, rel_fn)
    test_faith, test_rel = _legacy_axis_scores(test_ordered, faith_fn, rel_fn)

    y_val = np.array([s.reliable for s in val_samples], dtype=int)
    grid = unit_interval_grid(grid_step)
    faith_hits = val_faith[None, :] >= grid[:, None]
    rel_hits = val_rel[None, :] >= grid[:, None]
    best_score, best_faith, best_rel = -1.0, 0.0, 0.0
    for i, t_faith in enumerate(grid):
        for j, t_rel in enumerate(grid):
            score = macro_f1_binary(y_val, (faith_hits[i] & rel_hits[j]).astype(int))
            if score > best_score:
                best_score, best_faith, best_rel = score, float(t_faith), float(t_rel)

    return LegacyHoldoutResult(
        t_faith=best_faith,
        t_rel=best_rel,
        val=_legacy_metrics(val_samples, val_faith, val_rel, best_faith, best_rel),
        test=_legacy_metrics(test_samples, test_faith, test_rel, best_faith, best_rel),
    )
