"""Case-level uncertainty intervals and paired statistical tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import comb
from statistics import NormalDist
from typing import Any

import numpy as np

MetricFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate with a percentile 95% confidence interval."""

    point: float
    lo: float
    hi: float


@dataclass(frozen=True)
class PairedResult:
    """Paired metric difference with uncertainty and a two-sided p-value."""

    delta: float
    ci95: tuple[float, float]
    p: float
    significant: bool


@dataclass(frozen=True)
class McNemarResult:
    """Discordant correctness counts and exact McNemar p-value."""

    b: int
    c: int
    p: float


def _paired_arrays(y: Any, *predictions: Any) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(values) for values in (y, *predictions))
    if any(array.ndim == 0 for array in arrays):
        raise ValueError("y and predictions must contain one value per case")
    n_cases = len(arrays[0])
    if n_cases == 0:
        raise ValueError("Cannot evaluate an empty set of cases")
    lengths = [len(array) for array in arrays]
    if any(length != n_cases for length in lengths[1:]):
        raise ValueError(f"y and predictions must have equal lengths, got {lengths}")
    return arrays


def _validate_replicates(B: int) -> None:
    if isinstance(B, bool) or not isinstance(B, int) or B <= 0:
        raise ValueError(f"B must be a positive integer, got {B!r}")


def _metric_value(metric_fn: MetricFn, y: np.ndarray, pred: np.ndarray) -> float:
    value = float(metric_fn(y, pred))
    if not np.isfinite(value):
        raise ValueError(f"metric_fn returned a non-finite value: {value!r}")
    return value


def bootstrap_ci(
    y: Any,
    pred: Any,
    metric_fn: MetricFn,
    *,
    B: int = 10_000,
    seed: int = 0,
) -> BootstrapResult:
    """Estimate a percentile interval by resampling whole cases."""
    _validate_replicates(B)
    y_array, pred_array = _paired_arrays(y, pred)
    rng = np.random.default_rng(seed)
    estimates = np.empty(B, dtype=float)
    for replicate in range(B):
        indices = rng.integers(0, len(y_array), size=len(y_array))
        estimates[replicate] = _metric_value(
            metric_fn, y_array[indices], pred_array[indices]
        )
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return BootstrapResult(
        point=_metric_value(metric_fn, y_array, pred_array),
        lo=float(lo),
        hi=float(hi),
    )


def paired_bootstrap(
    y: Any,
    pred_a: Any,
    pred_b: Any,
    metric_fn: MetricFn,
    *,
    B: int = 10_000,
    seed: int = 0,
) -> PairedResult:
    """Compare methods on identical resampled cases to preserve pairing."""
    _validate_replicates(B)
    y_array, a_array, b_array = _paired_arrays(y, pred_a, pred_b)
    rng = np.random.default_rng(seed)
    deltas = np.empty(B, dtype=float)
    for replicate in range(B):
        indices = rng.integers(0, len(y_array), size=len(y_array))
        sampled_y = y_array[indices]
        deltas[replicate] = _metric_value(
            metric_fn, sampled_y, a_array[indices]
        ) - _metric_value(metric_fn, sampled_y, b_array[indices])

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    if np.all(deltas == 0.0):
        p_value = 1.0
    else:
        p_value = min(
            1.0,
            2.0 * min(float(np.mean(deltas <= 0.0)), float(np.mean(deltas >= 0.0))),
        )
    delta = _metric_value(metric_fn, y_array, a_array) - _metric_value(
        metric_fn, y_array, b_array
    )
    return PairedResult(
        delta=delta,
        ci95=(float(lo), float(hi)),
        p=p_value,
        significant=p_value < 0.05,
    )


def exact_mcnemar(y: Any, pred_a: Any, pred_b: Any) -> McNemarResult:
    """Use the exact binomial tail because discordant counts can be small."""
    y_array, a_array, b_array = _paired_arrays(y, pred_a, pred_b)
    a_correct = a_array == y_array
    b_correct = b_array == y_array
    b_count = int(np.sum(a_correct & ~b_correct))
    c_count = int(np.sum(~a_correct & b_correct))
    discordant = b_count + c_count
    if discordant == 0:
        p_value = 1.0
    else:
        lower_tail = sum(comb(discordant, i) for i in range(min(b_count, c_count) + 1))
        p_value = min(1.0, 2.0 * lower_tail / (2**discordant))
    return McNemarResult(b=b_count, c=c_count, p=p_value)


def wilson_ci(k: int, n: int, *, alpha: float = 0.05) -> tuple[float, float]:
    """Bound a binomial proportion without the zero-width Wald failure."""
    if isinstance(k, bool) or isinstance(n, bool) or not isinstance(k, int) or not isinstance(n, int):
        raise TypeError(f"k and n must be integers, got k={k!r}, n={n!r}")
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"Expected 0 <= k <= n with n > 0, got k={k}, n={n}")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")

    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    proportion = k / n
    denominator = 1.0 + z**2 / n
    centre = (proportion + z**2 / (2.0 * n)) / denominator
    margin = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / n + z**2 / (4.0 * n**2))
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)
