"""End-to-end tests for the resumable Method 6 pipeline."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rag_reliability.dataset import load_jsonl
from rag_reliability.metrics import evaluate_predictions
from rag_reliability.schema import Prediction

REPO = Path(__file__).resolve().parents[1]


def _write_data(tmp_path: Path) -> Path:
    rows = [
        {
            "id": f"s{i}",
            "question": "вопрос?",
            "context": "контекст.",
            "answer": "ответ.",
            "faithfulness": i % 2,
            "relevance": 1,
        }
        for i in range(4)
    ]
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    return path


def _run_pipeline(tmp_path: Path, data: Path, extra: tuple[str, ...] = ()) -> Path:
    output = tmp_path / "preds.jsonl"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_m6_pipeline.py",
            "--data",
            str(data),
            "--samples-dir",
            str(tmp_path / "samples"),
            "--features",
            str(tmp_path / "features.jsonl"),
            "--output",
            str(output),
            "--backend",
            "dummy",
            "--features-backend",
            "dummy",
            "--n-samples",
            "3",
            *extra,
        ],
        check=True,
        cwd=REPO,
    )
    return output


def _write_features(path: Path, sample_ids: list[str]) -> None:
    rows = [
        {
            "id": sample_id,
            "selfcheck_contra_mean": 0.1,
            "semantic_entropy": 0.2,
            "cos_q_a": 0.9,
            "seeded": True,
        }
        for sample_id in sample_ids
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_pipeline_dummy_end_to_end(tmp_path: Path) -> None:
    data = _write_data(tmp_path)

    output = _run_pipeline(tmp_path, data)

    lines = output.read_text(encoding="utf-8").strip().splitlines()
    predictions = [Prediction.model_validate_json(line) for line in lines]
    assert len(predictions) == 4
    assert all(prediction.faithfulness_prob is not None for prediction in predictions)
    result = evaluate_predictions(load_jsonl(data), predictions)
    assert result.total == 4


def test_pipeline_reuses_existing_features(tmp_path: Path) -> None:
    data = _write_data(tmp_path)
    _run_pipeline(tmp_path, data)
    features = tmp_path / "features.jsonl"
    mtime = features.stat().st_mtime_ns

    _run_pipeline(tmp_path, data)

    assert features.stat().st_mtime_ns == mtime


def test_pipeline_complete_features_skip_sample_cache_creation(tmp_path: Path) -> None:
    data = _write_data(tmp_path)
    _write_features(tmp_path / "features.jsonl", [f"s{i}" for i in range(4)])

    output = _run_pipeline(tmp_path, data)

    assert output.exists()
    assert not (tmp_path / "samples").exists()


def test_pipeline_partial_features_sample_only_missing_ids(tmp_path: Path) -> None:
    data = _write_data(tmp_path)
    features_path = tmp_path / "features.jsonl"
    _write_features(features_path, ["s0", "s1"])

    _run_pipeline(tmp_path, data)

    assert sorted(path.stem for path in (tmp_path / "samples").glob("*.json")) == ["s2", "s3"]
    features = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in features_path.read_text(encoding="utf-8").splitlines()
        )
    }
    assert set(features) == {"s0", "s1", "s2", "s3"}
    assert features["s0"]["seeded"] is True


def test_pipeline_refresh_features_samples_all_ids(tmp_path: Path) -> None:
    data = _write_data(tmp_path)
    features_path = tmp_path / "features.jsonl"
    _write_features(features_path, [f"s{i}" for i in range(4)])

    _run_pipeline(tmp_path, data, ("--refresh-features",))

    assert sorted(path.stem for path in (tmp_path / "samples").glob("*.json")) == [
        "s0",
        "s1",
        "s2",
        "s3",
    ]
    assert all(
        "seeded" not in json.loads(line)
        for line in features_path.read_text(encoding="utf-8").splitlines()
    )


@pytest.mark.parametrize("seed_features", [False, True])
def test_pipeline_empty_refresh_writes_empty_features(
    tmp_path: Path, seed_features: bool
) -> None:
    data = _write_data(tmp_path)
    features_path = tmp_path / "features.jsonl"
    if seed_features:
        _write_features(features_path, ["stale"])

    _run_pipeline(tmp_path, data, ("--limit", "0", "--refresh-features"))

    assert features_path.exists()
    assert features_path.read_text(encoding="utf-8") == ""
    assert not (tmp_path / "samples").exists()
