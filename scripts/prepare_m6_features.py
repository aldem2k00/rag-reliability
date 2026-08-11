#!/usr/bin/env python
"""Prepare Method 6 SelfCheck-style features from generated answer samples."""

from __future__ import annotations

import argparse

from tqdm import tqdm

from rag_reliability.dataset import load_jsonl, save_jsonl
from rag_reliability.methods.m6.features import build_feature_row, load_sample_cache
from rag_reliability.schema import RagSample


def build_feature_rows(
    samples: list[RagSample],
    *,
    samples_dir,
    nli,
    embedder,
    entail_threshold: float,
    use_n_samples: int | None = None,
) -> list[dict]:
    rows = []
    for sample in tqdm(samples, desc="m6/features"):
        generated_samples = load_sample_cache(samples_dir, sample.id)
        if use_n_samples is not None:
            if len(generated_samples) < use_n_samples:
                raise ValueError(
                    f"{sample.id}: cache has {len(generated_samples)} samples, "
                    f"need {use_n_samples}"
                )
            generated_samples = generated_samples[:use_n_samples]
        rows.append(
            build_feature_row(
                sample,
                generated_samples,
                nli=nli,
                embedder=embedder,
                entail_threshold=entail_threshold,
            )
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dummy.jsonl")
    parser.add_argument("--samples-dir", default="results/m6/samples")
    parser.add_argument("--output", default="results/m6/features.jsonl")
    parser.add_argument(
        "--nli-model",
        default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    )
    parser.add_argument("--embed-model", default="intfloat/multilingual-e5-large")
    parser.add_argument("--entail-threshold", type=float, default=0.5)
    parser.add_argument("--use-n-samples", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.data)
    if args.limit is not None:
        samples = samples[: args.limit]

    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    from rag_reliability.methods.m6.nli import NLIScorer  # noqa: PLC0415

    nli = NLIScorer(args.nli_model)
    embedder = SentenceTransformer(args.embed_model)
    rows = build_feature_rows(
        samples,
        samples_dir=args.samples_dir,
        nli=nli,
        embedder=embedder,
        entail_threshold=args.entail_threshold,
        use_n_samples=args.use_n_samples,
    )
    save_jsonl(rows, args.output)
    print(f"Wrote {len(rows)} Method 6 feature rows to {args.output}")


if __name__ == "__main__":
    main()
