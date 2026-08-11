"""Tests for bootstrap intervals and paired statistical tests."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from sklearn.metrics import f1_score

from rag_reliability.evaluation.bootstrap import (
    bootstrap_ci,
    exact_mcnemar,
    paired_bootstrap,
    wilson_ci,
)

ROOT = Path(__file__).resolve().parents[1]


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _legacy_test_labels() -> dict[str, int]:
    """Recreate stable content ids used by the committed legacy artifacts."""
    archive_path = ROOT / "from_organizators/data/data.zip"
    with ZipFile(archive_path) as archive, archive.open("data.csv") as raw:
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
    return labels


def _legacy_predictions(path: str) -> tuple[list[str], np.ndarray]:
    prediction_path = ROOT / path
    rows = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = json.loads(
        prediction_path.with_name("report_test.json").read_text(encoding="utf-8")
    )
    pred = np.array(
        [
            row["p_faith"] >= report["t_faith"] and row["p_rel"] >= report["t_rel"]
            for row in rows
        ],
        dtype=int,
    )
    return [row["id"] for row in rows], pred


def test_bootstrap_ci_contains_point_and_narrows_with_more_cases() -> None:
    rng = np.random.default_rng(42)
    small = rng.binomial(1, 0.7, size=40)
    large = rng.binomial(1, 0.7, size=800)

    small_result = bootstrap_ci(
        np.zeros_like(small), small, lambda _y, pred: float(np.mean(pred)), B=4_000, seed=7
    )
    large_result = bootstrap_ci(
        np.zeros_like(large), large, lambda _y, pred: float(np.mean(pred)), B=4_000, seed=7
    )

    assert small_result.lo <= small_result.point <= small_result.hi
    assert large_result.lo <= large_result.point <= large_result.hi
    assert large_result.hi - large_result.lo < small_result.hi - small_result.lo


def test_paired_bootstrap_identical_predictions_has_unit_p_value() -> None:
    y = np.array([0, 0, 1, 1, 1])
    pred = np.array([0, 1, 1, 0, 1])

    result = paired_bootstrap(y, pred, pred, _macro_f1, B=1_000, seed=4)

    assert result.delta == 0.0
    assert result.ci95 == (0.0, 0.0)
    assert result.p == 1.0
    assert result.significant is False


def test_bootstrap_is_deterministic_for_fixed_seed() -> None:
    y = np.array([0, 0, 1, 1, 1, 0])
    pred = np.array([0, 1, 1, 1, 0, 0])

    first = bootstrap_ci(y, pred, _macro_f1, B=2_000, seed=91)
    second = bootstrap_ci(y, pred, _macro_f1, B=2_000, seed=91)

    assert first == second


def test_exact_mcnemar_matches_known_discordant_table() -> None:
    y = np.zeros(58, dtype=int)
    pred_a = np.concatenate([np.zeros(31, dtype=int), np.ones(27, dtype=int)])
    pred_b = np.concatenate([np.ones(31, dtype=int), np.zeros(27, dtype=int)])

    result = exact_mcnemar(y, pred_a, pred_b)

    assert (result.b, result.c) == (31, 27)
    assert result.p == pytest.approx(0.6940040941)


def test_wilson_interval_at_zero_is_bounded_and_nonzero_width() -> None:
    lo, hi = wilson_ci(0, 10)

    assert 0.0 <= lo < hi <= 1.0


def test_surface_vs_m3_zero_shot_regression() -> None:
    labels = _legacy_test_labels()
    surface_ids, surface = _legacy_predictions(
        "predictions/alfa/baselines/surface/test.jsonl"
    )
    m3_ids, m3 = _legacy_predictions(
        "predictions/alfa/m3/zero_shot/test.jsonl"
    )
    assert surface_ids == m3_ids
    y = np.array([labels[sample_id] for sample_id in surface_ids], dtype=int)

    result = paired_bootstrap(y, surface, m3, _macro_f1, B=10_000, seed=0)

    assert result.delta == pytest.approx(0.0141, abs=0.005)
    assert result.ci95 == pytest.approx((-0.076, 0.105), abs=0.01)
    assert result.p == pytest.approx(0.765, abs=0.05)
    assert result.significant is False
