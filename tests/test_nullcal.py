"""Tests for calibration against threshold-fitted random scores."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from rag_reliability.evaluation.nullcal import null_calibration, percentile_of

ROOT = Path(__file__).resolve().parents[1]


def _legacy_test_y() -> np.ndarray:
    with (
        ZipFile(ROOT / "from_organizators/data/data.zip") as archive,
        archive.open("data.csv") as raw,
    ):
        rows = list(csv.DictReader(line.decode("utf-8-sig") for line in raw))

    labels: dict[str, int] = {}
    for row in rows:
        digest = hashlib.sha1(
            (row["full_dialog"] + "\0" + row["answer"]).encode("utf-8")
        ).hexdigest()
        sample_id = f"alfa_{digest[:12]}"
        faith = int(row["binary_faithfulness"].strip().lower() in {"true", "1"})
        relevance = int(row["binary_relevancy"].strip().lower() in {"true", "1"})
        labels[sample_id] = faith & relevance

    artifact = ROOT / "predictions/alfa/baselines/surface/test.jsonl"
    ids = [
        json.loads(line)["id"]
        for line in artifact.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return np.array([labels[sample_id] for sample_id in ids], dtype=int)


def _balanced_folds(y: np.ndarray, n_folds: int) -> np.ndarray:
    folds = np.empty(len(y), dtype=int)
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        folds[indices] = np.arange(len(indices)) % n_folds
    return folds


def test_random_labels_p95_is_close_to_chance_theory() -> None:
    rng = np.random.default_rng(18)
    y = rng.binomial(1, 0.5, size=400)
    folds = _balanced_folds(y, 5)

    result = null_calibration(y, folds, n_trials=500, seed=23)

    theoretical_p95 = 0.5 + 1.645 * 0.5 / np.sqrt(len(y))
    assert result.p95 == pytest.approx(theoretical_p95, abs=0.03)


def test_percentile_of_is_monotonic() -> None:
    y = np.array([0, 1] * 50)
    folds = _balanced_folds(y, 2)
    result = null_calibration(y, folds, n_trials=100, seed=4)

    observed = [percentile_of(value, result) for value in (0.3, 0.5, 0.7)]

    assert observed == sorted(observed)
    assert observed[0] <= observed[-1]


def test_fit_apply_extension_receives_train_only_and_controls_predictions() -> None:
    y = np.array([0, 0, 1, 1, 0, 1])
    folds = np.array([0, 1, 0, 1, 0, 1])
    seen_train_sizes: list[int] = []

    def always_unreliable(
        y_train: np.ndarray,
        train_scores: np.ndarray,
        test_scores: np.ndarray,
        grid_step: float,
    ) -> np.ndarray:
        seen_train_sizes.append(len(y_train))
        assert len(train_scores) == len(y_train)
        assert grid_step == 0.01
        return np.zeros(len(test_scores), dtype=int)

    result = null_calibration(
        y, folds, n_trials=3, seed=9, fit_apply_fn=always_unreliable
    )

    assert seen_train_sizes == [3] * 6
    assert len(set(result.values)) == 1


def test_legacy_test_labels_have_expected_null_p95() -> None:
    y = _legacy_test_y()
    folds = _balanced_folds(y, 2)

    result = null_calibration(y, folds, n_trials=500, grid_step=0.01, seed=0)

    assert result.p95 == pytest.approx(0.549, abs=0.02)
