"""Evaluation metrics: reliable/faithfulness/relevance F1-macro + invalid rate."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import log2

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

from rag_reliability.formatting import resolve_marker
from rag_reliability.schema import EvaluationResult, Prediction, RagSample


def _marker_metrics(
    samples: list[RagSample], pred_by_id: dict[str, Prediction]
) -> tuple[float, dict[str, float], dict[str, dict[str, int]]]:
    """Per-marker F1 and gold->pred confusion counts.

    Gold markers fall back like training targets (none/unknown); predictions
    without a marker (parse fallback) count as "unknown".
    """
    gold = [resolve_marker(s) for s in samples]
    pred = [pred_by_id[s.id].marker_pred or "unknown" for s in samples]

    labels = sorted(set(gold) | set(pred))
    per_class = f1_score(gold, pred, labels=labels, average=None, zero_division=0)
    macro = float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0))

    confusion: dict[str, dict[str, int]] = {}
    for g, p in zip(gold, pred, strict=True):
        row = confusion.setdefault(g, {})
        row[p] = row.get(p, 0) + 1

    return macro, {label: float(v) for label, v in zip(labels, per_class, strict=True)}, confusion


def evaluate_predictions(
    samples: list[RagSample], predictions: list[Prediction]
) -> EvaluationResult:
    """Join predictions to samples by id and compute macro-F1 metrics.

    reliable = faithfulness AND relevance, on both gold and predicted sides.
    Invalid outputs keep their conservative (0, 0) predictions and are counted
    in invalid_output_rate.
    """
    if not samples:
        raise ValueError("No samples to evaluate")

    pred_by_id: dict[str, Prediction] = {}
    for p in predictions:
        if p.id in pred_by_id:
            raise ValueError(f"Duplicate prediction id: {p.id!r}")
        pred_by_id[p.id] = p
    missing = [s.id for s in samples if s.id not in pred_by_id]
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} sample(s): {missing[:5]}...")

    faithfulness_true, faithfulness_pred = [], []
    relevance_true, relevance_pred = [], []
    reliable_true, reliable_pred = [], []
    invalid_count = 0

    for sample in samples:
        pred = pred_by_id[sample.id]
        faithfulness_true.append(sample.faithfulness)
        relevance_true.append(sample.relevance)
        reliable_true.append(sample.reliable)
        faithfulness_pred.append(pred.faithfulness_pred)
        relevance_pred.append(pred.relevance_pred)
        reliable_pred.append(pred.reliable_pred)
        if pred.invalid_output:
            invalid_count += 1

    def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    marker_macro: float | None = None
    marker_per_class: dict[str, float] | None = None
    marker_confusion: dict[str, dict[str, int]] | None = None
    if any(p.marker_pred is not None for p in pred_by_id.values()):
        marker_macro, marker_per_class, marker_confusion = _marker_metrics(samples, pred_by_id)

    total = len(samples)
    return EvaluationResult(
        reliable_f1_macro=macro_f1(reliable_true, reliable_pred),
        faithfulness_f1_macro=macro_f1(faithfulness_true, faithfulness_pred),
        relevance_f1_macro=macro_f1(relevance_true, relevance_pred),
        invalid_output_rate=invalid_count / total,
        total=total,
        invalid_count=invalid_count,
        marker_f1_macro=marker_macro,
        marker_per_class_f1=marker_per_class,
        marker_confusion=marker_confusion,
    )


def operational_metrics(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    threshold: float,
) -> dict[str, object]:
    """Evaluate a deployment gate where the positive event is an unreliable answer."""
    y_array = np.asarray(y_true)
    score_array = np.asarray(scores, dtype=float)
    if y_array.ndim != 1 or score_array.ndim != 1 or len(y_array) == 0:
        raise ValueError("y_true and scores must be non-empty one-dimensional arrays")
    if len(y_array) != len(score_array):
        raise ValueError(
            f"y_true and scores must have equal lengths, got {len(y_array)} and {len(score_array)}"
        )
    if not np.all(np.isin(y_array, (0, 1))):
        raise ValueError("y_true must contain only binary reliable labels 0 and 1")
    if not np.all(np.isfinite(score_array)) or not np.all((0.0 <= score_array) & (score_array <= 1.0)):
        raise ValueError("scores must contain only finite probabilities in [0, 1]")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be a finite probability in [0, 1], got {threshold!r}")

    unreliable_true = 1 - y_array.astype(int, copy=False)
    if len(np.unique(unreliable_true)) != 2:
        raise ValueError("operational metrics require both reliable and unreliable cases")
    risk_scores = 1.0 - score_array
    flagged = (score_array < threshold).astype(int)

    base_rate = float(np.mean(unreliable_true))
    pr_auc = float(average_precision_score(unreliable_true, risk_scores))
    precision, recall, _ = precision_recall_curve(unreliable_true, risk_scores)
    target_recalls = {
        target: float(np.max(recall[precision >= target], initial=0.0))
        for target in (0.5, 0.6, 0.7)
    }

    tn = int(np.sum((unreliable_true == 0) & (flagged == 0)))
    fp = int(np.sum((unreliable_true == 0) & (flagged == 1)))
    fn = int(np.sum((unreliable_true == 1) & (flagged == 0)))
    tp = int(np.sum((unreliable_true == 1) & (flagged == 1)))
    return {
        "roc_auc": float(roc_auc_score(unreliable_true, risk_scores)),
        "pr_auc_unreliable": pr_auc,
        "base_rate_unreliable": base_rate,
        "lift": pr_auc / base_rate,
        "recall_at_precision": target_recalls,
        "flagged_share": float(np.mean(flagged)),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def degenerate_rate(predictions: Sequence[Prediction]) -> dict[str, float | bool]:
    """Detect collapse even when every constant output is syntactically valid."""
    if not predictions:
        raise ValueError("Cannot diagnose an empty prediction set")
    counts = Counter(
        (prediction.faithfulness_pred, prediction.relevance_pred)
        for prediction in predictions
    )
    total = len(predictions)
    shares = [count / total for count in counts.values()]
    const_share = max(shares)
    entropy = -sum(share * log2(share) for share in shares)
    return {
        "const_share": const_share,
        "output_entropy": entropy,
        "is_degenerate": const_share > 0.98,
    }
