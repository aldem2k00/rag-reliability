#!/usr/bin/env python
"""Compare validation-fitted Method 6 thresholds with and without entropy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag_reliability.dataset import load_jsonl
from rag_reliability.methods.m6.ablation import fit_feature_thresholds, score_feature_thresholds
from rag_reliability.methods.m6.predict import load_features
from rag_reliability.schema import RagSample


def parse_args() -> argparse.Namespace:
    """Parse split-specific feature and dataset paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--val-features", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--test-features", required=True)
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _result_for_setting(
    *,
    use_entropy: bool,
    val_samples: list[RagSample],
    val_features: dict[str, dict[str, Any]],
    test_samples: list[RagSample],
    test_features: dict[str, dict[str, Any]],
    grid_step: float,
) -> dict[str, Any]:
    """Fit on validation data and score exactly those thresholds on test data."""
    fit = fit_feature_thresholds(
        val_samples,
        val_features,
        use_entropy=use_entropy,
        grid_step=grid_step,
    )
    return {
        "fit": fit,
        "test_reliable_f1_macro": score_feature_thresholds(test_samples, test_features, fit),
    }


def main() -> None:
    """Write and print the entropy comparison JSON."""
    args = parse_args()
    val_samples = load_jsonl(args.val_data)
    test_samples = load_jsonl(args.test_data)
    val_features = load_features(args.val_features)
    test_features = load_features(args.test_features)
    result = {
        "with_entropy": _result_for_setting(
            use_entropy=True,
            val_samples=val_samples,
            val_features=val_features,
            test_samples=test_samples,
            test_features=test_features,
            grid_step=args.grid_step,
        ),
        "contradiction_only": _result_for_setting(
            use_entropy=False,
            val_samples=val_samples,
            val_features=val_features,
            test_samples=test_samples,
            test_features=test_features,
            grid_step=args.grid_step,
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
