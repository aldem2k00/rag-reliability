#!/usr/bin/env python
"""Run supported reliability methods through one predictions -> metrics contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from rag_reliability.methods import registry

METHODS = registry.all_method_names()


@dataclass(frozen=True)
class MethodRun:
    name: str
    predictions_path: Path
    metrics_path: Path
    run_command: list[str]
    evaluate_command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dummy.jsonl", help="Input RagSample JSONL")
    parser.add_argument("--output-dir", default="results/benchmark", help="Where to write runs")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N samples")
    parser.add_argument(
        "--methods",
        default="dummy_direct,dummy_marker",
        help=f"Comma-separated methods. Available: {', '.join(METHODS)}",
    )
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--direct-adapter-path", default="results/adapters_direct")
    parser.add_argument("--marker-adapter-path", default="results/adapters_marker")
    parser.add_argument("--lettucedetect-model", default="results/lettucedetect/classifier.joblib")
    parser.add_argument("--encoder-model", default="deepvk/RuModernBERT-base")
    parser.add_argument("--encoder-output-dir", default=None)
    parser.add_argument("--encoder-max-length", type=int, default=512)
    parser.add_argument("--encoder-batch-size", type=int, default=4)
    parser.add_argument("--encoder-epochs", type=float, default=3)
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-5)
    parser.add_argument("--encoder-pos-weight-mode", choices=["balanced", "none"], default="none")
    parser.add_argument(
        "--m3-backend", choices=["dummy", "mlx", "openai", "openai_judge"], default="mlx"
    )
    parser.add_argument("--m3-max-tokens", type=int, default=400)
    parser.add_argument("--m3-max-context-chars", type=int, default=None)
    parser.add_argument("--m3-examples", default="configs/few_shot.yaml")
    parser.add_argument("--m3-prompt-file", default="configs/m3_gepa_prompt.txt")
    parser.add_argument("--m3-api-base", default="http://localhost:8000/v1")
    parser.add_argument("--m3-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--m3-cache-dir", default="results/m3/cache")
    parser.add_argument("--m3-concurrency", type=int, default=1)
    parser.add_argument("--m6-features", default="results/m6/features.jsonl")
    parser.add_argument("--m6-backend", choices=["dummy", "mlx", "openai"], default="dummy")
    parser.add_argument("--m6-samples-dir", default="results/m6/samples")
    parser.add_argument("--m6-n-samples", type=int, default=5)
    parser.add_argument("--m6-contradiction-threshold", type=float, default=0.5)
    parser.add_argument("--m6-entropy-threshold", type=float, default=1.0)
    parser.add_argument("--m6-relevance-threshold", type=float, default=0.25)
    return parser.parse_args()


def parse_methods(raw_methods: str) -> list[str]:
    return registry.resolve_names(raw_methods)


def build_evaluate_command(
    python: str,
    data: Path,
    predictions_path: Path,
    metrics_path: Path,
    limit: int | None = None,
) -> list[str]:
    command = [
        python,
        "scripts/evaluate.py",
        "--data",
        str(data),
        "--predictions",
        str(predictions_path),
        "--output",
        str(metrics_path),
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def build_method_run(  # noqa: PLR0913
    method: str,
    data: Path,
    output_dir: Path,
    python: str = sys.executable,
    model: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    max_tokens: int = 64,
    direct_adapter_path: str = "results/adapters_direct",
    marker_adapter_path: str = "results/adapters_marker",
    lettucedetect_model: str = "results/lettucedetect/classifier.joblib",
    encoder_model: str = "deepvk/RuModernBERT-base",
    encoder_output_dir: str | None = None,
    encoder_max_length: int = 512,
    encoder_batch_size: int = 4,
    encoder_epochs: float = 3,
    encoder_learning_rate: float = 2e-5,
    encoder_pos_weight_mode: str = "none",
    m3_backend: str = "mlx",
    m3_max_tokens: int = 400,
    m3_max_context_chars: int | None = None,
    m3_examples: str = "configs/few_shot.yaml",
    m3_prompt_file: str = "configs/m3_gepa_prompt.txt",
    m3_api_base: str = "http://localhost:8000/v1",
    m3_api_key_env: str = "OPENAI_API_KEY",
    m3_cache_dir: str = "results/m3/cache",
    m3_concurrency: int = 1,
    m6_features: str = "results/m6/features.jsonl",
    m6_backend: str = "dummy",
    m6_samples_dir: str = "results/m6/samples",
    m6_n_samples: int = 5,
    m6_contradiction_threshold: float = 0.5,
    m6_entropy_threshold: float = 1.0,
    m6_relevance_threshold: float = 0.25,
    limit: int | None = None,
) -> MethodRun:
    spec = registry.get(method)
    run_dir = output_dir / method
    predictions_path = run_dir / "predictions.jsonl"
    metrics_path = run_dir / "metrics.json"
    ctx = registry.CommandContext(
        data=data,
        run_dir=run_dir,
        predictions_path=predictions_path,
        python=python,
        model=model,
        max_tokens=max_tokens,
        direct_adapter_path=direct_adapter_path,
        marker_adapter_path=marker_adapter_path,
        lettucedetect_model=lettucedetect_model,
        encoder_model=encoder_model,
        encoder_output_dir=encoder_output_dir,
        encoder_max_length=encoder_max_length,
        encoder_batch_size=encoder_batch_size,
        encoder_epochs=encoder_epochs,
        encoder_learning_rate=encoder_learning_rate,
        encoder_pos_weight_mode=encoder_pos_weight_mode,
        m3_backend=m3_backend,
        m3_max_tokens=m3_max_tokens,
        m3_max_context_chars=m3_max_context_chars,
        m3_examples=m3_examples,
        m3_prompt_file=m3_prompt_file,
        m3_api_base=m3_api_base,
        m3_api_key_env=m3_api_key_env,
        m3_cache_dir=m3_cache_dir,
        m3_concurrency=m3_concurrency,
        m6_features=m6_features,
        m6_backend=m6_backend,
        m6_samples_dir=m6_samples_dir,
        m6_n_samples=m6_n_samples,
        m6_contradiction_threshold=m6_contradiction_threshold,
        m6_entropy_threshold=m6_entropy_threshold,
        m6_relevance_threshold=m6_relevance_threshold,
        limit=limit,
    )
    return MethodRun(
        name=spec.name,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        run_command=spec.build_command(ctx),
        evaluate_command=build_evaluate_command(python, data, predictions_path, metrics_path, limit),
    )


def run_command(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    data = Path(args.data)
    output_dir = Path(args.output_dir)
    runs = [
        build_method_run(
            method=method,
            data=data,
            output_dir=output_dir,
            model=args.model,
            max_tokens=args.max_tokens,
            direct_adapter_path=args.direct_adapter_path,
            marker_adapter_path=args.marker_adapter_path,
            lettucedetect_model=args.lettucedetect_model,
            encoder_model=args.encoder_model,
            encoder_output_dir=args.encoder_output_dir,
            encoder_max_length=args.encoder_max_length,
            encoder_batch_size=args.encoder_batch_size,
            encoder_epochs=args.encoder_epochs,
            encoder_learning_rate=args.encoder_learning_rate,
            encoder_pos_weight_mode=args.encoder_pos_weight_mode,
            m3_backend=args.m3_backend,
            m3_max_tokens=args.m3_max_tokens,
            m3_max_context_chars=args.m3_max_context_chars,
            m3_examples=args.m3_examples,
            m3_prompt_file=args.m3_prompt_file,
            m3_api_base=args.m3_api_base,
            m3_api_key_env=args.m3_api_key_env,
            m3_cache_dir=args.m3_cache_dir,
            m3_concurrency=args.m3_concurrency,
            m6_features=args.m6_features,
            m6_backend=args.m6_backend,
            m6_samples_dir=args.m6_samples_dir,
            m6_n_samples=args.m6_n_samples,
            m6_contradiction_threshold=args.m6_contradiction_threshold,
            m6_entropy_threshold=args.m6_entropy_threshold,
            m6_relevance_threshold=args.m6_relevance_threshold,
            limit=args.limit,
        )
        for method in parse_methods(args.methods)
    ]

    summary = {}
    for method_run in runs:
        method_run.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        run_command(method_run.run_command)
        run_command(method_run.evaluate_command)
        summary[method_run.name] = {
            "predictions": str(method_run.predictions_path),
            "metrics": str(method_run.metrics_path),
        }

    summary_path = output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved benchmark summary to {summary_path}")


if __name__ == "__main__":
    main()
