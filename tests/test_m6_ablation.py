"""Validation-fitted Method 6 feature-threshold ablation tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rag_reliability.methods.m6.ablation import (
    fit_feature_thresholds,
    score_feature_thresholds,
)
from rag_reliability.schema import RagSample
from rag_reliability.thresholds import macro_f1_binary


def make_sample(sample_id: str, reliable: int) -> RagSample:
    """Make a sample whose reliable label has the requested value."""
    return RagSample(
        id=sample_id,
        question="q",
        context="c",
        answer="a",
        faithfulness=reliable,
        relevance=reliable,
    )


def test_fit_without_entropy_ignores_entropy_column() -> None:
    samples = [make_sample("reliable", 1), make_sample("unreliable", 0)]
    features = {
        "reliable": {"selfcheck_contra_mean": 0.1, "semantic_entropy": 99.0, "cos_q_a": 0.9},
        "unreliable": {"selfcheck_contra_mean": 0.9, "semantic_entropy": 0.0, "cos_q_a": 0.9},
    }

    fit = fit_feature_thresholds(samples, features, use_entropy=False)

    assert fit["t_entropy"] is None
    assert fit["val_reliable_f1_macro"] == pytest.approx(1.0)


def test_fit_with_entropy_grid_over_observed_values() -> None:
    samples = [make_sample(f"s{i}", int(i == 0)) for i in range(3)]
    features = {
        "s0": {"selfcheck_contra_mean": 0.1, "semantic_entropy": 0.0, "cos_q_a": 0.9},
        "s1": {"selfcheck_contra_mean": 0.1, "semantic_entropy": 0.69, "cos_q_a": 0.9},
        "s2": {"selfcheck_contra_mean": 0.1, "semantic_entropy": 1.09, "cos_q_a": 0.9},
    }

    fit = fit_feature_thresholds(samples, features, use_entropy=True)

    assert fit["t_entropy"] in {0.0, 0.69, 1.09, math.inf}


def test_score_applies_val_thresholds_to_other_set() -> None:
    val_samples = [make_sample("v0", 1), make_sample("v1", 0)]
    val_features = {
        "v0": {"selfcheck_contra_mean": 0.2, "semantic_entropy": 0.1, "cos_q_a": 0.8},
        "v1": {"selfcheck_contra_mean": 0.8, "semantic_entropy": 0.1, "cos_q_a": 0.8},
    }
    test_samples = [make_sample("t0", 1), make_sample("t1", 0), make_sample("t2", 0)]
    test_features = {
        "t0": {"selfcheck_contra_mean": 0.2, "semantic_entropy": 0.2, "cos_q_a": 0.8},
        "t1": {"selfcheck_contra_mean": 0.8, "semantic_entropy": 0.2, "cos_q_a": 0.8},
        "t2": {"selfcheck_contra_mean": 0.1, "semantic_entropy": 0.2, "cos_q_a": 0.1},
    }

    fit = fit_feature_thresholds(val_samples, val_features, use_entropy=True)
    actual = score_feature_thresholds(test_samples, test_features, fit)
    expected_pred = np.array(
        [
            int(
                row["selfcheck_contra_mean"] <= fit["t_contra"]
                and row["semantic_entropy"] <= fit["t_entropy"]
                and row["cos_q_a"] >= fit["t_rel"]
            )
            for row in test_features.values()
        ]
    )
    expected = macro_f1_binary(np.array([sample.reliable for sample in test_samples]), expected_pred)

    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("grid_step", [0.3, 1.5])
def test_fit_grid_is_bounded_and_includes_unit_endpoint(grid_step: float) -> None:
    samples = [
        make_sample("reliable", 1),
        make_sample("near_relevant", 0),
        make_sample("unreliable", 0),
    ]
    features = {
        "reliable": {"selfcheck_contra_mean": 1.0, "cos_q_a": 1.0},
        "near_relevant": {"selfcheck_contra_mean": 1.0, "cos_q_a": 0.95},
        "unreliable": {"selfcheck_contra_mean": 0.0, "cos_q_a": 0.0},
    }

    fit = fit_feature_thresholds(samples, features, use_entropy=False, grid_step=grid_step)

    assert fit["t_contra"] == pytest.approx(1.0)
    assert fit["t_rel"] == pytest.approx(1.0)
    assert fit["val_reliable_f1_macro"] == pytest.approx(1.0)
    assert 0.0 <= fit["t_contra"] <= 1.0
    assert 0.0 <= fit["t_rel"] <= 1.0
