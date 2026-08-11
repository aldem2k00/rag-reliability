"""Tests for evaluate.py helper behavior."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rag_reliability.metrics import evaluate_predictions
from rag_reliability.schema import Prediction, RagSample

_SPEC = importlib.util.spec_from_file_location(
    "evaluate_script",
    Path(__file__).parents[1] / "scripts" / "evaluate.py",
)
assert _SPEC is not None
evaluate_script = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["evaluate_script"] = evaluate_script
_SPEC.loader.exec_module(evaluate_script)


def test_apply_limit_keeps_first_n_samples() -> None:
    assert evaluate_script.apply_limit([1, 2, 3], None) == [1, 2, 3]
    assert evaluate_script.apply_limit([1, 2, 3], 2) == [1, 2]


def _write_split(
    tmp_path: Path, name: str, reliable_prob: float = 0.9, unreliable_prob: float = 0.1
) -> tuple[Path, Path]:
    samples = [
        {
            "id": f"{name}-reliable",
            "question": "q",
            "context": "c",
            "answer": "a",
            "faithfulness": 1,
            "relevance": 1,
        },
        {
            "id": f"{name}-unfaithful",
            "question": "q",
            "context": "c",
            "answer": "a",
            "faithfulness": 0,
            "relevance": 1,
        },
        {
            "id": f"{name}-irrelevant",
            "question": "q",
            "context": "c",
            "answer": "a",
            "faithfulness": 1,
            "relevance": 0,
        },
        {
            "id": f"{name}-unreliable",
            "question": "q",
            "context": "c",
            "answer": "a",
            "faithfulness": 0,
            "relevance": 0,
        },
    ]
    predictions = [
        {
            "id": f"{name}-reliable",
            "faithfulness_pred": 0,
            "relevance_pred": 0,
            "faithfulness_prob": reliable_prob,
            "relevance_prob": reliable_prob,
        },
        {
            "id": f"{name}-unfaithful",
            "faithfulness_pred": 0,
            "relevance_pred": 0,
            "faithfulness_prob": unreliable_prob,
            "relevance_prob": reliable_prob,
        },
        {
            "id": f"{name}-irrelevant",
            "faithfulness_pred": 0,
            "relevance_pred": 0,
            "faithfulness_prob": reliable_prob,
            "relevance_prob": unreliable_prob,
        },
        {
            "id": f"{name}-unreliable",
            "faithfulness_pred": 0,
            "relevance_pred": 0,
            "faithfulness_prob": unreliable_prob,
            "relevance_prob": unreliable_prob,
        },
    ]
    data_path = tmp_path / f"{name}_data.jsonl"
    predictions_path = tmp_path / f"{name}_predictions.jsonl"
    data_path.write_text("\n".join(json.dumps(row) for row in samples) + "\n", encoding="utf-8")
    predictions_path.write_text(
        "\n".join(json.dumps(row) for row in predictions) + "\n", encoding="utf-8"
    )
    return data_path, predictions_path


def _run_evaluate(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/evaluate.py", *args],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
        check=False,
    )


def test_threshold_mode_report_structure(tmp_path: Path) -> None:
    val_data, val_predictions = _write_split(tmp_path, "val", reliable_prob=0.5, unreliable_prob=0.49)
    test_data, test_predictions = _write_split(
        tmp_path, "test", reliable_prob=0.2, unreliable_prob=0.19
    )
    output = tmp_path / "report.json"

    result = _run_evaluate(
        "--data",
        str(test_data),
        "--predictions",
        str(test_predictions),
        "--val-data",
        str(val_data),
        "--val-predictions",
        str(val_predictions),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["mode"] == "threshold_fit"
    assert set(payload["thresholds"]) == {
        "t_faith",
        "t_rel",
        "val_reliable_f1_macro",
        "grid_step",
    }
    assert "reliable_f1_macro" in payload["tuned"]
    assert "reliable_f1_macro" in payload["binary_default"]
    assert payload["thresholds"]["t_faith"] == 0.5
    assert payload["thresholds"]["t_rel"] == 0.5
    assert payload["tuned"]["reliable_f1_macro"] == pytest.approx(3 / 7)


def test_threshold_mode_requires_both_val_args(tmp_path: Path) -> None:
    test_data, test_predictions = _write_split(tmp_path, "test")
    output = tmp_path / "report.json"

    result = _run_evaluate(
        "--data",
        str(test_data),
        "--predictions",
        str(test_predictions),
        "--val-data",
        str(test_data),
        "--output",
        str(output),
    )

    assert result.returncode != 0
    assert "--val-data and --val-predictions must be given together" in result.stderr


def test_plain_mode_output_unchanged(tmp_path: Path) -> None:
    data_path, predictions_path = _write_split(tmp_path, "plain")
    output = tmp_path / "report.json"

    result = _run_evaluate(
        "--data",
        str(data_path),
        "--predictions",
        str(predictions_path),
        "--output",
        str(output),
    )

    expected = evaluate_predictions(
        [RagSample.model_validate_json(line) for line in data_path.read_text().splitlines()],
        [Prediction.model_validate_json(line) for line in predictions_path.read_text().splitlines()],
    ).model_dump(exclude_none=True)
    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8") == json.dumps(expected, indent=2, ensure_ascii=False) + "\n"
