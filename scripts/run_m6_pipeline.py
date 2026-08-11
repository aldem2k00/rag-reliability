#!/usr/bin/env python
"""Method 6 end-to-end: answer samples -> NLI/entropy features -> predictions.

Resumable: the per-id sample cache is topped up, feature rows are computed only
for ids missing from --features (or all of them with --refresh-features). The
dummy backends make the whole pipeline runnable offline (CI smoke path).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_m6_features as m6_features  # noqa: E402
import prepare_m6_samples as m6_samples  # noqa: E402

from rag_reliability.dataset import load_jsonl, save_jsonl  # noqa: E402
from rag_reliability.methods.m6.predict import (  # noqa: E402
    load_features,
    predictions_from_feature_rows,
)
from rag_reliability.schema import RagSample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dummy.jsonl")
    parser.add_argument("--samples-dir", default="results/m6/samples")
    parser.add_argument("--features", default="results/m6/features.jsonl")
    parser.add_argument("--output", default="results/m6_selfcheck_predictions.jsonl")
    parser.add_argument("--backend", choices=["dummy", "mlx", "openai"], default="dummy")
    parser.add_argument(
        "--features-backend",
        choices=["real", "dummy"],
        default=None,
        help="Defaults to dummy when --backend dummy, else real",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-context-chars", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--nli-model",
        default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    )
    parser.add_argument("--embed-model", default="intfloat/multilingual-e5-large")
    parser.add_argument("--entail-threshold", type=float, default=0.5)
    parser.add_argument("--use-n-samples", type=int, default=None)
    parser.add_argument("--refresh-features", action="store_true")
    parser.add_argument("--contradiction-threshold", type=float, default=0.5)
    parser.add_argument("--entropy-threshold", type=float, default=1.0)
    parser.add_argument("--relevance-threshold", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-meta", default=None)
    return parser.parse_args()


def ensure_samples(samples: list[RagSample], args: argparse.Namespace) -> None:
    if args.backend == "dummy":
        generate_fn, batch_fn = m6_samples.dummy_generate, None
    elif args.backend == "mlx":
        from rag_reliability.mlx_backend import make_generate_fn  # noqa: PLC0415

        generate_fn, batch_fn = make_generate_fn(args.model, args.max_tokens), None
    else:
        generate_fn = None
        batch_fn = m6_samples.make_openai_batch_fn(
            model=args.model,
            api_base=args.api_base,
            api_key=os.environ.get(args.api_key_env, ""),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    m6_samples.build_sample_cache(
        samples,
        output_dir=args.samples_dir,
        generate_fn=generate_fn,
        generate_batch_fn=batch_fn,
        n_samples=args.n_samples,
        max_context_chars=args.max_context_chars,
    )


def _feature_work(
    samples: list[RagSample], args: argparse.Namespace
) -> tuple[list[RagSample], dict[str, dict[str, Any]]]:
    features_path = Path(args.features)
    existing_features: dict[str, dict[str, Any]] = {}
    if features_path.exists() and not args.refresh_features:
        existing_features = load_features(features_path)
    pending = (
        samples
        if args.refresh_features
        else [sample for sample in samples if sample.id not in existing_features]
    )
    return pending, existing_features


def ensure_features(
    samples: list[RagSample],
    args: argparse.Namespace,
    *,
    work: tuple[list[RagSample], dict[str, dict[str, Any]]] | None = None,
) -> None:
    if work is None:
        work = _feature_work(samples, args)
    pending, existing_features = work
    if not pending:
        if args.refresh_features:
            save_jsonl([], args.features)
        return

    features_backend = args.features_backend or (
        "dummy" if args.backend == "dummy" else "real"
    )
    if features_backend == "dummy":
        from rag_reliability.methods.m6.dummy import (  # noqa: PLC0415
            DummyEmbedder,
            DummyNLI,
        )

        nli, embedder = DummyNLI(), DummyEmbedder()
    else:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            from rag_reliability.methods.m6.nli import NLIScorer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError('Install Method 6 deps with: uv pip install -e ".[m6]"') from exc
        nli, embedder = NLIScorer(args.nli_model), SentenceTransformer(args.embed_model)

    rows = m6_features.build_feature_rows(
        pending,
        samples_dir=args.samples_dir,
        nli=nli,
        embedder=embedder,
        entail_threshold=args.entail_threshold,
        use_n_samples=args.use_n_samples,
    )
    save_jsonl([*existing_features.values(), *rows], args.features)


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.data)
    if args.limit is not None:
        samples = samples[: args.limit]
    work = _feature_work(samples, args)
    pending, _ = work
    if pending:
        ensure_samples(pending, args)
    ensure_features(samples, args, work=work)
    predictions = predictions_from_feature_rows(
        samples,
        load_features(args.features),
        contradiction_threshold=args.contradiction_threshold,
        entropy_threshold=args.entropy_threshold,
        relevance_threshold=args.relevance_threshold,
    )
    save_jsonl(predictions, args.output)
    if args.run_meta:
        from rag_reliability.run_meta import write_run_meta  # noqa: PLC0415

        write_run_meta(args.run_meta, args)
    print(f"Wrote {len(predictions)} Method 6 predictions to {args.output}")


if __name__ == "__main__":
    main()
