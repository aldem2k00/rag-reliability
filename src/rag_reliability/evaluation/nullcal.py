"""Empirical noise floor for fold-fitted decision procedures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from rag_reliability.thresholds import macro_f1_binary, unit_interval_grid

FitApplyFn = Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]


@dataclass(frozen=True)
class NullResult:
    """Summary and raw trials from a null-calibration simulation."""

    mean: float
    p50: float
    p90: float
    p95: float
    p99: float
    max: float
    values: tuple[float, ...]


def _validated_inputs(y: Any, folds: Any) -> tuple[np.ndarray, np.ndarray]:
    y_array = np.asarray(y)
    if y_array.ndim != 1 or len(y_array) == 0:
        raise ValueError("y must be a non-empty one-dimensional array")
    if not np.all(np.isin(y_array, (0, 1))):
        raise ValueError("y must contain only binary labels 0 and 1")
    y_array = y_array.astype(int, copy=False)

    fold_array = np.asarray(folds)
    if fold_array.ndim == 1:
        fold_array = fold_array[:, None]
    if fold_array.ndim != 2 or fold_array.shape[0] != len(y_array):
        raise ValueError(
            "folds must have shape (n_cases,) or (n_cases, n_repeats), "
            f"got {fold_array.shape} for {len(y_array)} cases"
        )
    for repeat in range(fold_array.shape[1]):
        if len(np.unique(fold_array[:, repeat])) < 2:
            raise ValueError(f"Repeat {repeat} must contain at least two folds")
    return y_array, fold_array


def _default_fit_apply(
    y_train: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    grid_step: float,
) -> np.ndarray:
    grid = unit_interval_grid(grid_step)
    best_score = -1.0
    best_threshold = 0.0
    for threshold in grid:
        metric = macro_f1_binary(y_train, (train_scores >= threshold).astype(int))
        if metric > best_score:
            best_score = metric
            best_threshold = float(threshold)
    return (test_scores >= best_threshold).astype(int)


def _validate_trials(n_trials: int) -> None:
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials <= 0:
        raise ValueError(f"n_trials must be a positive integer, got {n_trials!r}")


def null_calibration(
    y: Any,
    folds: Any,
    *,
    n_trials: int = 500,
    grid_step: float = 0.01,
    seed: int = 0,
    fit_apply_fn: FitApplyFn | None = None,
) -> NullResult:
    """Measure what the same fold-fitted procedure extracts from random scores.

    ``fit_apply_fn`` receives ``(y_train, train_scores, test_scores, grid_step)``
    and must return one binary decision per held-out score.
    """
    _validate_trials(n_trials)
    y_array, fold_array = _validated_inputs(y, folds)
    procedure = fit_apply_fn or _default_fit_apply
    rng = np.random.default_rng(seed)
    trial_metrics = np.empty(n_trials, dtype=float)

    for trial in range(n_trials):
        scores = rng.random(len(y_array))
        repeat_metrics = np.empty(fold_array.shape[1], dtype=float)
        for repeat in range(fold_array.shape[1]):
            repeat_folds = fold_array[:, repeat]
            oof_predictions = np.empty(len(y_array), dtype=int)
            for fold in np.unique(repeat_folds):
                test_mask = repeat_folds == fold
                train_mask = ~test_mask
                predictions = np.asarray(
                    procedure(
                        y_array[train_mask],
                        scores[train_mask],
                        scores[test_mask],
                        grid_step,
                    )
                )
                if predictions.shape != (int(np.sum(test_mask)),):
                    raise ValueError(
                        "fit_apply_fn must return one prediction per held-out case, "
                        f"got shape {predictions.shape} for {int(np.sum(test_mask))} cases"
                    )
                if not np.all(np.isin(predictions, (0, 1))):
                    raise ValueError("fit_apply_fn must return only binary predictions 0 and 1")
                oof_predictions[test_mask] = predictions.astype(int, copy=False)
            repeat_metrics[repeat] = macro_f1_binary(y_array, oof_predictions)
        trial_metrics[trial] = float(np.mean(repeat_metrics))

    p50, p90, p95, p99 = np.percentile(trial_metrics, [50, 90, 95, 99])
    return NullResult(
        mean=float(np.mean(trial_metrics)),
        p50=float(p50),
        p90=float(p90),
        p95=float(p95),
        p99=float(p99),
        max=float(np.max(trial_metrics)),
        values=tuple(float(value) for value in trial_metrics),
    )


def percentile_of(value: float, null_result: NullResult) -> float:
    """Return the empirical percentile on a 0–100 reporting scale."""
    if not np.isfinite(value):
        raise ValueError(f"value must be finite, got {value!r}")
    if not null_result.values:
        raise ValueError("null_result contains no trial values")
    return 100.0 * float(np.mean(np.asarray(null_result.values) <= value))
