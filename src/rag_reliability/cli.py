"""rag-judge: one CLI over the reliability method registry."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import typer

from rag_reliability.methods import registry

app = typer.Typer(help="Run and evaluate RAG answer-reliability methods.", no_args_is_help=True)


def _execute(methods: list[str], data: Path, output_dir: Path, overrides: dict) -> dict:
    import run_benchmark  # reuse MethodRun + subprocess plumbing

    summary: dict = {}
    for name in methods:
        method_run = run_benchmark.build_method_run(
            method=name, data=data, output_dir=output_dir, python=sys.executable, **overrides
        )
        method_run.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        run_benchmark.run_command(method_run.run_command)
        run_benchmark.run_command(method_run.evaluate_command)
        summary[name] = {
            "predictions": str(method_run.predictions_path),
            "metrics": str(method_run.metrics_path),
        }
    (output_dir).mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


@app.command(help="Run a single method end-to-end (predict + evaluate).")
def run(
    method: str = typer.Option(..., help="Method name. See `rag-judge list-methods`."),
    data: Path = typer.Option(Path("data/dummy.jsonl"), help="Input RagSample JSONL."),
    output_dir: Path = typer.Option(Path("results/run"), help="Directory for predictions + metrics."),
    model: str = typer.Option("mlx-community/Qwen2.5-1.5B-Instruct-4bit", help="MLX model id."),
    limit: int | None = typer.Option(None, help="Run only the first N samples."),
) -> None:
    try:
        registry.get(method)
    except KeyError as exc:
        raise typer.BadParameter(
            f"Unknown method {method!r}; available: {', '.join(registry.all_method_names())}"
        ) from exc
    _execute([method], data, output_dir, {"model": model, "limit": limit})


@app.command(help="Run several methods through the shared predictions -> metrics contract.")
def benchmark(
    methods: str = typer.Option("dummy_direct,dummy_marker", help="Comma list or 'all'."),
    data: Path = typer.Option(Path("data/dummy.jsonl"), help="Input RagSample JSONL."),
    output_dir: Path = typer.Option(Path("results/benchmark"), help="Directory for all runs."),
    model: str = typer.Option("mlx-community/Qwen2.5-1.5B-Instruct-4bit", help="MLX model id."),
    limit: int | None = typer.Option(None, help="Run only the first N samples."),
) -> None:
    try:
        names = registry.resolve_names(methods)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    summary = _execute(names, data, output_dir, {"model": model, "limit": limit})
    typer.echo(f"Saved benchmark summary for {len(summary)} method(s) to {output_dir / 'summary.json'}")


@app.command(help="Score a predictions JSONL against gold labels.")
def eval(  # noqa: A001 - CLI verb
    data: Path = typer.Option(..., help="Input RagSample JSONL with gold labels."),
    predictions: Path = typer.Option(..., help="Predictions JSONL to score."),
    output: Path = typer.Option(..., help="Where to write metrics.json."),
    limit: int | None = typer.Option(None, help="Score only the first N samples."),
    val_data: Path | None = typer.Option(None, help="Val dataset to fit thresholds on."),
    val_predictions: Path | None = typer.Option(None, help="Val predictions with probs."),
    grid_step: float = typer.Option(0.01, help="Threshold grid step."),
) -> None:
    if (val_data is None) != (val_predictions is None):
        raise typer.BadParameter("--val-data and --val-predictions must be given together")
    command = [
        sys.executable, "scripts/evaluate.py",
        "--data", str(data), "--predictions", str(predictions), "--output", str(output),
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if val_data is not None and val_predictions is not None:
        command.extend(
            [
                "--val-data",
                str(val_data),
                "--val-predictions",
                str(val_predictions),
                "--grid-step",
                str(grid_step),
            ]
        )
    subprocess.run(command, check=True)


@app.command(help="Launch the local Gradio demo UI.")
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(7860, help="Bind port."),
) -> None:
    subprocess.run(
        [sys.executable, "scripts/serve_demo.py", "--host", host, "--port", str(port)], check=True
    )


@app.command("list-methods", help="Print every registered method with family and requirements.")
def list_methods() -> None:
    for spec in registry.METHODS.values():
        requires = ", ".join(spec.requires) if spec.requires else "-"
        demo = "demo" if spec.demo_runner else "batch-only"
        scope = "corpus-wide" if spec.corpus_wide else "split-only"
        scores = ", ".join(spec.score_keys) if spec.score_keys else "-"
        typer.echo(
            f"{spec.name:16} {spec.family:14} {demo:10} {scope:12} "
            f"scores: {scores:44} requires: {requires}"
        )


if __name__ == "__main__":
    app()
