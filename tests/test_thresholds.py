import numpy as np
import pytest
from sklearn.metrics import f1_score

from rag_reliability.schema import Prediction, RagSample
from rag_reliability.thresholds import (
    apply_thresholds,
    extract_probs,
    fit_thresholds,
    macro_f1_binary,
)


def make_sample(i: int, faith: int, rel: int) -> RagSample:
    return RagSample(
        id=f"s{i}", question="q", context="c", answer="a", faithfulness=faith, relevance=rel
    )


def make_pred(i: int, p_f: float, p_r: float) -> Prediction:
    return Prediction(
        id=f"s{i}",
        faithfulness_pred=0,
        relevance_pred=0,
        faithfulness_prob=p_f,
        relevance_prob=p_r,
        prob_method="logprobs",
    )


def test_macro_f1_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_pred = rng.integers(0, 2, size=200)
    expected = f1_score(y_true, y_pred, average="macro", zero_division=0)
    assert macro_f1_binary(y_true, y_pred) == pytest.approx(expected)


def test_fit_recovers_separating_thresholds() -> None:
    # Reliable exactly when p_faith >= 0.3 and p_rel >= 0.6: perfectly separable.
    samples, preds = [], []
    grid = [0.1, 0.2, 0.35, 0.5, 0.65, 0.9]
    i = 0
    for pf in grid:
        for pr in grid:
            samples.append(make_sample(i, int(pf >= 0.3), int(pr >= 0.6)))
            preds.append(make_pred(i, pf, pr))
            i += 1
    fit = fit_thresholds(samples, preds)
    assert fit.val_reliable_f1_macro == pytest.approx(1.0)
    assert 0.2 < fit.t_faith <= 0.35
    assert 0.5 < fit.t_rel <= 0.65


def test_fit_tie_break_deterministic() -> None:
    samples = [make_sample(i, 1, 1) for i in range(4)]
    preds = [make_pred(i, 0.5, 0.5) for i in range(4)]
    fit_a = fit_thresholds(samples, preds)
    fit_b = fit_thresholds(samples, preds)
    assert (fit_a.t_faith, fit_a.t_rel) == (fit_b.t_faith, fit_b.t_rel) == (0.0, 0.0)


@pytest.mark.parametrize("grid_step", [0.3, 1.5])
def test_fit_grid_is_bounded_and_includes_unit_endpoint(grid_step: float) -> None:
    samples = [
        make_sample(0, 1, 1),
        make_sample(1, 0, 0),
        make_sample(2, 0, 0),
    ]
    preds = [
        make_pred(0, 1.0, 1.0),
        make_pred(1, 1.0, 0.95),
        make_pred(2, 0.0, 0.0),
    ]

    fit = fit_thresholds(samples, preds, grid_step=grid_step)

    assert fit.t_rel == pytest.approx(1.0)
    assert fit.val_reliable_f1_macro == pytest.approx(1.0)
    assert 0.0 <= fit.t_faith <= 1.0
    assert 0.0 <= fit.t_rel <= 1.0


@pytest.mark.parametrize("grid_step", [0.0, -0.1, np.inf, np.nan])
def test_fit_rejects_invalid_grid_step(grid_step: float) -> None:
    samples = [make_sample(0, 1, 1)]
    preds = [make_pred(0, 1.0, 1.0)]

    with pytest.raises(ValueError, match="grid_step must be a positive finite number"):
        fit_thresholds(samples, preds, grid_step=grid_step)


def test_extract_probs_raises_on_missing() -> None:
    good = make_pred(0, 0.7, 0.7)
    bad = Prediction(id="s1", faithfulness_pred=1, relevance_pred=1)
    with pytest.raises(ValueError, match="s1"):
        extract_probs([good, bad])


def test_fit_raises_on_missing_and_duplicate_ids() -> None:
    samples = [make_sample(0, 1, 1), make_sample(1, 0, 0)]
    with pytest.raises(ValueError, match="s1"):
        fit_thresholds(samples, [make_pred(0, 0.5, 0.5)])
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        fit_thresholds(samples, [make_pred(0, 0.5, 0.5), make_pred(0, 0.5, 0.5)])


def test_apply_thresholds_recomputes_binary_keeps_probs() -> None:
    preds = [make_pred(0, 0.7, 0.2)]
    out = apply_thresholds(preds, t_faith=0.6, t_rel=0.3)
    assert out[0].faithfulness_pred == 1
    assert out[0].relevance_pred == 0
    assert out[0].faithfulness_prob == pytest.approx(0.7)
    assert out[0].prob_method == "logprobs"
