"""Val-fitted threshold search over probability-bearing predictions.

Grid search (default step 0.01) for (t_faith, t_rel) maximizing macro-F1 of
reliable = (p_faith >= t_faith) AND (p_rel >= t_rel). Fit on the validation
split only; apply unchanged elsewhere. Deterministic tie-break: ascending scan,
replace only on strict improvement, so the lowest (t_faith, t_rel) wins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rag_reliability.schema import Prediction, RagSample


@dataclass(frozen=True)
class ThresholdFit:
    t_faith: float
    t_rel: float
    val_reliable_f1_macro: float
    grid_step: float


def unit_interval_grid(grid_step: float) -> np.ndarray:
    """Return ascending thresholds in [0, 1], always including the endpoint."""
    if not np.isfinite(grid_step) or grid_step <= 0:
        raise ValueError("grid_step must be a positive finite number")
    return np.append(np.arange(0.0, 1.0, grid_step), 1.0)


def extract_probs(predictions: list[Prediction]) -> tuple[np.ndarray, np.ndarray]:
    """Return (p_faith, p_rel) arrays; raise naming ids without probabilities."""
    missing = [
        p.id for p in predictions if p.faithfulness_prob is None or p.relevance_prob is None
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} prediction(s) without probabilities "
            f"(need faithfulness_prob/relevance_prob): {missing[:5]}"
        )
    p_faith = np.array([p.faithfulness_prob for p in predictions], dtype=float)
    p_rel = np.array([p.relevance_prob for p in predictions], dtype=float)
    return p_faith, p_rel


def macro_f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 over classes {0, 1}; empty class contributes 0 (sklearn zero_division=0)."""
    scores = []
    for cls in (0, 1):
        tp = int(np.sum((y_pred == cls) & (y_true == cls)))
        fp = int(np.sum((y_pred == cls) & (y_true != cls)))
        fn = int(np.sum((y_pred != cls) & (y_true == cls)))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(scores))


def _ordered_predictions(
    samples: list[RagSample], predictions: list[Prediction]
) -> list[Prediction]:
    by_id = {p.id: p for p in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("Duplicate prediction ids")
    missing = [s.id for s in samples if s.id not in by_id]
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} sample(s): {missing[:5]}")
    return [by_id[s.id] for s in samples]


def fit_thresholds(
    samples: list[RagSample], predictions: list[Prediction], grid_step: float = 0.01
) -> ThresholdFit:
    ordered = _ordered_predictions(samples, predictions)
    p_faith, p_rel = extract_probs(ordered)
    y_reliable = np.array([s.reliable for s in samples], dtype=int)
    grid = unit_interval_grid(grid_step)
    faith_hits = p_faith[None, :] >= grid[:, None]  # (G, n)
    rel_hits = p_rel[None, :] >= grid[:, None]
    best_score, best_tf, best_tr = -1.0, 0.0, 0.0
    for i, t_faith in enumerate(grid):
        fh = faith_hits[i]
        for j, t_rel in enumerate(grid):
            y_pred = (fh & rel_hits[j]).astype(int)
            score = macro_f1_binary(y_reliable, y_pred)
            if score > best_score:
                best_score, best_tf, best_tr = score, float(t_faith), float(t_rel)
    return ThresholdFit(best_tf, best_tr, best_score, grid_step)


def apply_thresholds(
    predictions: list[Prediction], t_faith: float, t_rel: float
) -> list[Prediction]:
    """New Predictions with binary fields recomputed from probs; everything else kept."""
    p_faith, p_rel = extract_probs(predictions)
    return [
        pred.model_copy(
            update={
                "faithfulness_pred": int(pf >= t_faith),
                "relevance_pred": int(pr >= t_rel),
            }
        )
        for pred, pf, pr in zip(predictions, p_faith, p_rel, strict=True)
    ]
