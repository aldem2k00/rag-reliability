"""Тесты стэкинга разнородных сигналов."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from rag_reliability.dataset import load_jsonl
from rag_reliability.evaluation.protocol import evaluate_cv
from rag_reliability.schema import Prediction, RagSample
from rag_reliability.stacking.collect import collect_features
from rag_reliability.stacking.stack import (
    FEATURE_SET_V1,
    fit_stack,
    make_prediction_fit_fn,
    select_features_by_ci,
)


def _write_scores(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_collect_features_preserves_sample_order(tmp_path: Path) -> None:
    surface = tmp_path / "surface.jsonl"
    m3 = tmp_path / "m3.jsonl"
    _write_scores(
        surface,
        [
            {"id": "b", "scores": {"surf.p_faith": 0.8, "surf.p_rel": 0.7}},
            {"id": "a", "scores": {"surf.p_faith": 0.2, "surf.p_rel": 0.3}},
        ],
    )
    _write_scores(
        m3,
        [
            {"id": "a", "scores": {"m3.p_faith": 0.4}},
            {"id": "b", "scores": {"m3.p_faith": 0.9}},
        ],
    )

    matrix, names = collect_features(
        {"surf": surface, "m3": m3},
        ["a", "b"],
        ["surf.p_faith", "surf.p_rel", "m3.p_faith"],
    )

    assert names == ["surf.p_faith", "surf.p_rel", "m3.p_faith"]
    np.testing.assert_allclose(matrix, [[0.2, 0.3, 0.4], [0.8, 0.7, 0.9]])


def test_collect_features_lists_missing_ids(tmp_path: Path) -> None:
    surface = tmp_path / "surface.jsonl"
    _write_scores(
        surface,
        [{"id": "a", "scores": {"surf.p_faith": 0.2}}],
    )

    with pytest.raises(ValueError, match=r"Missing 2 sample id\(s\).*b.*c"):
        collect_features(
            {"surf": surface},
            ["a", "b", "c"],
            ["surf.p_faith"],
        )


def test_collect_features_reports_missing_key(tmp_path: Path) -> None:
    surface = tmp_path / "surface.jsonl"
    _write_scores(
        surface,
        [
            {"id": "a", "scores": {"surf.p_faith": 0.2}},
            {"id": "b", "scores": {"surf.p_rel": 0.7}},
        ],
    )

    with pytest.raises(ValueError, match=r"Missing feature 'surf\.p_faith'.*b"):
        collect_features(
            {"surf": surface},
            ["a", "b"],
            ["surf.p_faith"],
        )


def test_collect_features_skips_absent_optional_source(tmp_path: Path) -> None:
    surface = tmp_path / "surface.jsonl"
    _write_scores(
        surface,
        [{"id": "a", "scores": {"surf.p_faith": 0.2}}],
    )

    matrix, names = collect_features(
        {"surf": surface, "enc": tmp_path / "missing.jsonl"},
        ["a"],
        ["surf.p_faith", "enc.logit"],
        required=False,
    )

    assert names == ["surf.p_faith"]
    np.testing.assert_allclose(matrix, [[0.2]])


def test_collect_features_rejects_duplicate_ids(tmp_path: Path) -> None:
    surface = tmp_path / "surface.jsonl"
    _write_scores(
        surface,
        [
            {"id": "a", "scores": {"surf.p_faith": 0.2}},
            {"id": "a", "scores": {"surf.p_faith": 0.3}},
        ],
    )

    with pytest.raises(ValueError, match="Duplicate id 'a'"):
        collect_features({"surf": surface}, ["a"], ["surf.p_faith"])


def test_fit_stack_auto_chooses_platt_below_500_and_isotonic_from_500() -> None:
    rng = np.random.default_rng(7)
    small_x = rng.normal(size=(60, 3))
    small_y = np.tile([0, 1], 30)
    large_x = rng.normal(size=(500, 3))
    large_y = np.tile([0, 1], 250)

    small = fit_stack(small_x, small_y, model="logreg", calibrate="auto", seed=4)
    large = fit_stack(large_x, large_y, model="logreg", calibrate="auto", seed=4)

    assert small.calibration_method == "sigmoid"
    assert large.calibration_method == "isotonic"
    assert small.predict_proba(small_x).shape == (60, 2)
    assert large.predict_proba(large_x).shape == (500, 2)


def test_prediction_fit_fn_receives_only_training_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rag_reliability.stacking.stack as stack_module

    seen_fit_rows: list[set[int]] = []

    class _RecordingModel:
        def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
            positive = np.full(len(matrix), 0.6)
            return np.column_stack([1.0 - positive, positive])

    def recording_fit(
        matrix: np.ndarray,
        labels: np.ndarray,
        *,
        model: str,
        calibrate: str,
        seed: int,
    ) -> _RecordingModel:
        del labels, model, calibrate, seed
        seen_fit_rows.append(set(matrix[:, 0].astype(int).tolist()))
        return _RecordingModel()

    monkeypatch.setattr(stack_module, "fit_stack", recording_fit)
    samples = [
        RagSample(
            id=f"s{index}",
            question="q",
            context="c",
            answer="a",
            faithfulness=index % 2,
            relevance=1,
        )
        for index in range(6)
    ]
    predictions = [
        Prediction(
            id=f"s{index}",
            faithfulness_pred=0,
            relevance_pred=0,
            scores={"surf.row": float(index)},
        )
        for index in range(6)
    ]
    folds = {
        "config": {"n_folds": 2, "n_repeats": 1},
        "assignment": {f"s{index}": [index % 2] for index in range(6)},
    }

    evaluate_cv(
        samples,
        predictions,
        folds,
        score_fn=lambda prediction: 0.5,
        fit_fn=make_prediction_fit_fn(["surf.row"]),
    )

    assert seen_fit_rows == [{1, 3, 5}, {0, 2, 4}]


def test_select_features_by_ci_uses_strictly_positive_lower_bound() -> None:
    selected = select_features_by_ci(
        ["surf.p_faith", "surf.p_rel"],
        {
            "m3.p_faith": (0.002, 0.051),
            "m3.p_rel": (-0.008, 0.013),
            "enc.logit": (0.0, 0.040),
        },
    )

    assert selected == ["surf.p_faith", "surf.p_rel", "m3.p_faith"]
    assert "m3.p_rel" not in FEATURE_SET_V1


def test_run_stack_cli_exposes_required_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "run_stack.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    for option in ("--data", "--folds", "--sources", "--features", "--model", "--output"):
        assert option in completed.stdout


@pytest.mark.slow
def test_real_artifact_regression_reproduces_historical_stack() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = {
        "surf": root / "predictions/alfa/baselines/surface_e5/scores.jsonl",
        "m3": root / "predictions/alfa/m3/zero_shot/scores.jsonl",
    }
    source_ids: list[set[str]] = []
    for path in sources.values():
        source_ids.append(
            {
                str(json.loads(line)["id"])
                for line in path.read_text(encoding="utf-8").splitlines()
            }
        )
    common_ids = set.intersection(*source_ids)
    samples = [
        sample
        for sample in load_jsonl(root / "data/alfa.jsonl")
        if sample.id in common_ids
    ]
    feature_keys = ["surf.p_faith", "surf.p_rel", "m3.p_faith"]
    matrix, names = collect_features(
        sources,
        [sample.id for sample in samples],
        feature_keys,
    )
    predictions = [
        Prediction(
            id=sample.id,
            faithfulness_pred=0,
            relevance_pred=0,
            scores=dict(zip(names, row, strict=True)),
        )
        for sample, row in zip(samples, matrix, strict=True)
    ]
    labels = np.asarray([sample.reliable for sample in samples], dtype=int)
    assignment = {sample.id: [-1] * 5 for sample in samples}
    splitter = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=0,
    )
    for split_index, (_, test_indices) in enumerate(splitter.split(matrix, labels)):
        repeat, fold = divmod(split_index, 5)
        for index in test_indices:
            assignment[samples[index].id][repeat] = fold

    result = evaluate_cv(
        samples,
        predictions,
        {
            "config": {"n_folds": 5, "n_repeats": 5},
            "assignment": assignment,
        },
        score_fn=lambda prediction: 0.5,
        fit_fn=make_prediction_fit_fn(feature_keys, seed=0),
    )

    assert len(result.ids) == 446
    assert float(np.mean(result.per_repeat_f1)) == pytest.approx(0.6085, abs=0.02)
    assert roc_auc_score(result.y, result.oof_scores) == pytest.approx(0.6277, abs=0.02)
