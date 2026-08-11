"""Validation-fitted threshold ablations for Method 6 feature rows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from rag_reliability.schema import RagSample
from rag_reliability.thresholds import macro_f1_binary, unit_interval_grid

_CONTRADICTION_KEY = "selfcheck_contra_mean"
_ENTROPY_KEY = "semantic_entropy"
_RELEVANCE_KEY = "cos_q_a"

def _ordered_feature_rows(
    samples: list[RagSample], features_by_id: Mapping[str, Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """Return feature rows in sample order, rejecting samples without features."""
    missing = [sample.id for sample in samples if sample.id not in features_by_id]
    if missing:
        raise ValueError(f"Missing Method 6 features for {len(missing)} sample(s): {missing[:5]}")
    return [features_by_id[sample.id] for sample in samples]


def _feature_arrays(
    samples: list[RagSample],
    features_by_id: Mapping[str, Mapping[str, Any]],
    *,
    use_entropy: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Extract ordered feature and gold-label vectors using prediction defaults."""
    rows = _ordered_feature_rows(samples, features_by_id)
    contradiction = np.array([float(row.get(_CONTRADICTION_KEY, 0.0)) for row in rows])
    relevance = np.array([float(row.get(_RELEVANCE_KEY, 1.0)) for row in rows])
    entropy = (
        np.array([float(row.get(_ENTROPY_KEY, 0.0)) for row in rows]) if use_entropy else None
    )
    reliable = np.array([sample.reliable for sample in samples], dtype=int)
    return contradiction, entropy, relevance, reliable


def fit_feature_thresholds(
    samples: list[RagSample],
    features_by_id: Mapping[str, Mapping[str, Any]],
    *,
    use_entropy: bool,
    grid_step: float = 0.01,
) -> dict[str, float | None]:
    """Fit Method 6 reliability thresholds on validation samples only.

    Thresholds are scanned in ascending order and are updated only on a strict
    score improvement, which makes ties resolve to the earliest grid point.
    """
    contradiction, entropy, relevance, y_reliable = _feature_arrays(
        samples, features_by_id, use_entropy=use_entropy
    )
    grid = unit_interval_grid(grid_step)
    contra_hits = contradiction[None, :] <= grid[:, None]
    relevance_hits = relevance[None, :] >= grid[:, None]
    entropy_grid = np.unique(np.append(entropy, np.inf)) if use_entropy else np.array([np.inf])
    entropy_hits = (
        entropy[None, :] <= entropy_grid[:, None]
        if entropy is not None
        else np.ones((1, len(samples)), dtype=bool)
    )

    best_score = -1.0
    best_contra = 0.0
    best_entropy: float | None = None
    best_relevance = 0.0
    for contra_index, t_contra in enumerate(grid):
        for entropy_index, t_entropy in enumerate(entropy_grid):
            contra_entropy_hits = contra_hits[contra_index] & entropy_hits[entropy_index]
            for relevance_index, t_rel in enumerate(grid):
                y_pred = (contra_entropy_hits & relevance_hits[relevance_index]).astype(int)
                score = macro_f1_binary(y_reliable, y_pred)
                if score > best_score:
                    best_score = score
                    best_contra = float(t_contra)
                    best_entropy = float(t_entropy) if use_entropy else None
                    best_relevance = float(t_rel)

    return {
        "t_contra": best_contra,
        "t_entropy": best_entropy,
        "t_rel": best_relevance,
        "val_reliable_f1_macro": best_score,
    }


def score_feature_thresholds(
    samples: list[RagSample],
    features_by_id: Mapping[str, Mapping[str, Any]],
    fit: Mapping[str, float | None],
) -> float:
    """Score validation-fitted thresholds unchanged on another labeled split."""
    required = ("t_contra", "t_entropy", "t_rel")
    missing = [key for key in required if key not in fit]
    if missing:
        raise ValueError(f"Threshold fit missing required key(s): {missing}")

    use_entropy = fit["t_entropy"] is not None
    contradiction, entropy, relevance, y_reliable = _feature_arrays(
        samples, features_by_id, use_entropy=use_entropy
    )
    t_contra = fit["t_contra"]
    t_rel = fit["t_rel"]
    if t_contra is None or t_rel is None:
        raise ValueError("Threshold fit requires numeric t_contra and t_rel")

    y_pred = (contradiction <= t_contra) & (relevance >= t_rel)
    if entropy is not None:
        t_entropy = fit["t_entropy"]
        if t_entropy is None:
            raise ValueError("Entropy threshold is required when entropy is enabled")
        y_pred &= entropy <= t_entropy
    return macro_f1_binary(y_reliable, y_pred.astype(int))
