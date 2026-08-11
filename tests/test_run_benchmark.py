"""Tests for the unified benchmark runner."""

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_benchmark",
    Path(__file__).parents[1] / "scripts" / "run_benchmark.py",
)
assert _SPEC is not None
run_benchmark = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["run_benchmark"] = run_benchmark
_SPEC.loader.exec_module(run_benchmark)


def test_build_method_commands_for_dummy_and_evaluation(tmp_path: Path) -> None:
    run = run_benchmark.build_method_run(
        method="dummy_marker",
        data=Path("data/dummy.jsonl"),
        output_dir=tmp_path,
        python="python",
    )

    assert run.name == "dummy_marker"
    assert run.predictions_path == tmp_path / "dummy_marker" / "predictions.jsonl"
    assert run.metrics_path == tmp_path / "dummy_marker" / "metrics.json"
    assert run.run_command == [
        "python",
        "scripts/run_prompt_baseline.py",
        "--data",
        "data/dummy.jsonl",
        "--output",
        str(tmp_path / "dummy_marker" / "predictions.jsonl"),
        "--mode",
        "marker",
        "--backend",
        "dummy",
        "--dummy-strategy",
        "keyword",
    ]
    assert run.evaluate_command[-4:] == [
        "--predictions",
        str(tmp_path / "dummy_marker" / "predictions.jsonl"),
        "--output",
        str(tmp_path / "dummy_marker" / "metrics.json"),
    ]


def test_build_method_commands_for_encoder_export_predictions(tmp_path: Path) -> None:
    run = run_benchmark.build_method_run(
        method="encoder",
        data=Path("data/organizers.jsonl"),
        output_dir=tmp_path,
        python="python",
    )

    assert run.run_command[:3] == ["python", "scripts/train_encoder_baseline.py", "--data"]
    assert "--predictions-output" in run.run_command
    assert str(tmp_path / "encoder" / "predictions.jsonl") in run.run_command
    assert run.evaluate_command[0:2] == ["python", "scripts/evaluate.py"]


def test_build_method_commands_for_lora_and_lettucedetect(tmp_path: Path) -> None:
    lora = run_benchmark.build_method_run(
        method="lora_direct",
        data=Path("data/dummy.jsonl"),
        output_dir=tmp_path,
        python="python",
    )
    lettuce = run_benchmark.build_method_run(
        method="lettucedetect",
        data=Path("data/dummy.jsonl"),
        output_dir=tmp_path,
        python="python",
    )

    assert lora.run_command[1:4] == ["scripts/infer.py", "--data", "data/dummy.jsonl"]
    assert "--adapter-path" in lora.run_command
    assert lettuce.run_command[1:4] == [
        "scripts/infer_lettucedetect.py",
        "--data",
        "data/dummy.jsonl",
    ]
    assert "--model" in lettuce.run_command


def test_build_method_commands_for_m3_and_m6(tmp_path: Path) -> None:
    m3 = run_benchmark.build_method_run(
        method="m3_zero_shot",
        data=Path("data/dummy.jsonl"),
        output_dir=tmp_path,
        python="python",
        m3_backend="dummy",
    )
    m6 = run_benchmark.build_method_run(
        method="m6_selfcheck",
        data=Path("data/dummy.jsonl"),
        output_dir=tmp_path,
        python="python",
        m6_features="results/m6/features.jsonl",
        m6_backend="dummy",
        m6_samples_dir="results/m6/test-samples",
        m6_n_samples=3,
    )

    assert m3.run_command[0:2] == ["python", "scripts/run_m3.py"]
    assert "--backend" in m3.run_command
    assert "dummy" in m3.run_command
    assert m6.run_command[0:2] == ["python", "scripts/run_m6_pipeline.py"]
    assert "--samples-dir" in m6.run_command
    assert "results/m6/test-samples" in m6.run_command
    assert "--features" in m6.run_command
    assert "results/m6/features.jsonl" in m6.run_command
    assert "--backend" in m6.run_command
    assert "--n-samples" in m6.run_command
    assert "3" in m6.run_command


def test_build_method_commands_for_m3_few_shot_and_gepa(tmp_path: Path) -> None:
    few_shot = run_benchmark.build_method_run(
        method="m3_few_shot",
        data=Path("data/dummy.jsonl"),
        output_dir=tmp_path,
        python="python",
        m3_examples="configs/few_shot.yaml",
    )
    gepa = run_benchmark.build_method_run(
        method="m3_gepa",
        data=Path("data/dummy.jsonl"),
        output_dir=tmp_path,
        python="python",
        m3_prompt_file="artifacts/m3_optimized_prompt.txt",
    )

    assert few_shot.run_command[0:2] == ["python", "scripts/run_m3.py"]
    assert "--mode" in few_shot.run_command
    assert "few_shot" in few_shot.run_command
    assert "--examples" in few_shot.run_command
    assert "configs/few_shot.yaml" in few_shot.run_command
    assert "--mode" in gepa.run_command
    assert "gepa" in gepa.run_command
    assert "--prompt-file" in gepa.run_command
    assert "artifacts/m3_optimized_prompt.txt" in gepa.run_command


def test_build_method_command_for_m3_openai(tmp_path: Path) -> None:
    run = run_benchmark.build_method_run(
        method="m3_openai",
        data=Path("data/dummy.jsonl"),
        output_dir=tmp_path,
        python="python",
        model="remote-judge",
        m3_api_base="https://example.test/v1",
        m3_cache_dir="results/m3/cache",
    )

    assert run.run_command[0:2] == ["python", "scripts/run_m3.py"]
    assert "--backend" in run.run_command
    assert "openai" in run.run_command
    assert "--api-base" in run.run_command
    assert "https://example.test/v1" in run.run_command
    assert "--cache-dir" in run.run_command
    assert "results/m3/cache" in run.run_command


def test_build_method_command_for_m3_openai_judge(tmp_path: Path) -> None:
    run = run_benchmark.build_method_run(
        method="m3_openai_judge",
        data=Path("data/dummy.jsonl"),
        output_dir=tmp_path,
        python="python",
        model="remote-judge",
        m3_api_base="https://example.test/v1",
        m3_cache_dir="results/m3/cache",
        m3_concurrency=8,
    )

    assert run.run_command[0:2] == ["python", "scripts/run_m3.py"]
    assert "openai_judge" in run.run_command
    assert "--api-base" in run.run_command
    assert "--concurrency" in run.run_command
    assert "8" in run.run_command


def test_build_method_command_for_independent(tmp_path: Path) -> None:
    run = run_benchmark.build_method_run(
        method="independent",
        data=Path("data/dummy.jsonl"),
        output_dir=tmp_path,
        python="python",
    )
    assert run.run_command[0:2] == ["python", "scripts/run_independent.py"]
    assert run.predictions_path == tmp_path / "independent" / "predictions.jsonl"
    assert "--faithfulness-threshold" in run.run_command


def test_build_method_run_passes_limit_to_runner_and_evaluator(tmp_path: Path) -> None:
    run = run_benchmark.build_method_run(
        method="m3_zero_shot",
        data=Path("data/organizers.jsonl"),
        output_dir=tmp_path,
        python="python",
        m3_backend="dummy",
        limit=25,
    )

    assert "--limit" in run.run_command
    assert "25" in run.run_command
    assert "--limit" in run.evaluate_command
    assert "25" in run.evaluate_command


def test_m3_backend_accepts_openai_judge(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--m3-backend", "openai_judge"])

    args = run_benchmark.parse_args()

    assert args.m3_backend == "openai_judge"


def test_m6_pipeline_cli_options(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_benchmark.py",
            "--m6-backend",
            "openai",
            "--m6-samples-dir",
            "cache/m6",
            "--m6-n-samples",
            "7",
        ],
    )

    args = run_benchmark.parse_args()

    assert args.m6_backend == "openai"
    assert args.m6_samples_dir == "cache/m6"
    assert args.m6_n_samples == 7
