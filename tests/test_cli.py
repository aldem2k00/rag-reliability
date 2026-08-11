from typer.testing import CliRunner

import rag_reliability.cli as cli
from rag_reliability.cli import app

runner = CliRunner()


def test_global_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("run", "benchmark", "eval", "serve", "list-methods"):
        assert command in result.output


def test_benchmark_help_documents_methods_option() -> None:
    result = runner.invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "--methods" in result.output


def test_list_methods_prints_all_methods() -> None:
    result = runner.invoke(app, ["list-methods"])
    assert result.exit_code == 0
    assert "independent" in result.output
    assert "m3_openai_judge" in result.output


def test_run_rejects_unknown_method() -> None:
    result = runner.invoke(app, ["run", "--method", "nope", "--data", "data/dummy.jsonl"])
    assert result.exit_code != 0


def test_run_rejects_multiple_methods() -> None:
    result = runner.invoke(app, ["run", "--method", "all", "--data", "data/dummy.jsonl"])
    assert result.exit_code != 0


def test_eval_forwards_threshold_options(monkeypatch, tmp_path) -> None:
    commands = []

    def fake_run(command, check):
        commands.append((command, check))

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    result = runner.invoke(
        app,
        [
            "eval",
            "--data",
            str(tmp_path / "test.jsonl"),
            "--predictions",
            str(tmp_path / "test_predictions.jsonl"),
            "--output",
            str(tmp_path / "report.json"),
            "--val-data",
            str(tmp_path / "val.jsonl"),
            "--val-predictions",
            str(tmp_path / "val_predictions.jsonl"),
            "--grid-step",
            "0.2",
        ],
    )

    assert result.exit_code == 0
    assert commands == [
        (
            [
                cli.sys.executable,
                "scripts/evaluate.py",
                "--data",
                str(tmp_path / "test.jsonl"),
                "--predictions",
                str(tmp_path / "test_predictions.jsonl"),
                "--output",
                str(tmp_path / "report.json"),
                "--val-data",
                str(tmp_path / "val.jsonl"),
                "--val-predictions",
                str(tmp_path / "val_predictions.jsonl"),
                "--grid-step",
                "0.2",
            ],
            True,
        )
    ]


def test_eval_rejects_val_data_without_val_predictions(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli.subprocess, "run", lambda *_args, **_kwargs: None)
    result = runner.invoke(
        app,
        [
            "eval",
            "--data",
            str(tmp_path / "test.jsonl"),
            "--predictions",
            str(tmp_path / "test_predictions.jsonl"),
            "--output",
            str(tmp_path / "report.json"),
            "--val-data",
            str(tmp_path / "val.jsonl"),
        ],
    )

    assert result.exit_code != 0
    assert "--val-data and --val-predictions must be given together" in result.output


def test_eval_rejects_val_predictions_without_val_data(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli.subprocess, "run", lambda *_args, **_kwargs: None)
    result = runner.invoke(
        app,
        [
            "eval",
            "--data",
            str(tmp_path / "test.jsonl"),
            "--predictions",
            str(tmp_path / "test_predictions.jsonl"),
            "--output",
            str(tmp_path / "report.json"),
            "--val-predictions",
            str(tmp_path / "val_predictions.jsonl"),
        ],
    )

    assert result.exit_code != 0
    assert "--val-data and --val-predictions must be given together" in result.output
