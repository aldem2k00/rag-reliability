#!/usr/bin/env python
"""Evaluate a predictions jsonl against a labeled dataset.

Example:
    python scripts/evaluate.py \
        --data data/dummy.jsonl \
        --predictions results/dummy_predictions.jsonl \
        --output results/dummy_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_reliability.dataset import load_jsonl
from rag_reliability.metrics import evaluate_predictions
from rag_reliability.schema import Prediction, RagSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dummy.jsonl", help="Labeled dataset (jsonl)")
    parser.add_argument("--predictions", required=True, help="Predictions file (jsonl)")
    parser.add_argument("--output", default="results/metrics.json", help="Where to save metrics")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples")
    parser.add_argument("--val-data", default=None, help="Val dataset to fit thresholds on (jsonl)")
    parser.add_argument(
        "--val-predictions", default=None, help="Val predictions with probabilities (jsonl)"
    )
    parser.add_argument("--grid-step", type=float, default=0.01)
    args = parser.parse_args()
    if (args.val_data is None) != (args.val_predictions is None):
        parser.error("--val-data and --val-predictions must be given together")
    return args


def apply_limit(items: list, limit: int | None) -> list:
    return items if limit is None else items[:limit]


def load_predictions(path: str | Path) -> list[Prediction]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path.resolve()}")
    predictions: list[Prediction] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                predictions.append(Prediction.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"Invalid prediction at {path}:{line_no}: {exc}") from exc
    return predictions


def evaluate_with_thresholds(
    samples: list[RagSample],
    predictions: list[Prediction],
    val_samples: list[RagSample],
    val_predictions: list[Prediction],
    grid_step: float = 0.01,
) -> dict[str, object]:
    """Fit thresholds on validation data and evaluate tuned and default predictions."""
    from dataclasses import asdict

    from rag_reliability.thresholds import apply_thresholds, fit_thresholds

    fit = fit_thresholds(val_samples, val_predictions, grid_step=grid_step)
    tuned = evaluate_predictions(samples, apply_thresholds(predictions, fit.t_faith, fit.t_rel))
    default = evaluate_predictions(samples, predictions)
    return {
        "mode": "threshold_fit",
        "thresholds": asdict(fit),
        "tuned": tuned.model_dump(exclude_none=True),
        "binary_default": default.model_dump(exclude_none=True),
    }


def main() -> None:
    args = parse_args()
    samples = apply_limit(load_jsonl(args.data), args.limit)
    predictions = load_predictions(args.predictions)

    if args.val_data is not None:
        val_samples = load_jsonl(args.val_data)
        val_predictions = load_predictions(args.val_predictions)
        payload = evaluate_with_thresholds(
            samples, predictions, val_samples, val_predictions, args.grid_step
        )
    else:
        payload = evaluate_predictions(samples, predictions).model_dump(exclude_none=True)
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
