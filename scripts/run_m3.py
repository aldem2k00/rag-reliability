#!/usr/bin/env python
"""Run Method 3 prompt judge through the shared predictions contract."""

from __future__ import annotations

import argparse
import os

from tqdm import tqdm

from rag_reliability.dataset import load_jsonl, save_jsonl
from rag_reliability.dummy_model import STRATEGIES, DummyPredictor
from rag_reliability.methods.m3 import build_system_prompt, build_user_prompt, parse_m3_prediction
from rag_reliability.mlx_backend import make_generate_fn
from rag_reliability.parsing import parse_prediction
from rag_reliability.schema import Prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dummy.jsonl")
    parser.add_argument("--output", default="results/m3_zero_shot_predictions.jsonl")
    parser.add_argument("--mode", choices=["zero_shot", "few_shot", "gepa"], default="zero_shot")
    parser.add_argument("--examples", default=None, help="YAML examples for --mode few_shot")
    parser.add_argument("--prompt-file", default=None, help="Prompt text for --mode gepa")
    parser.add_argument(
        "--backend", choices=["dummy", "mlx", "openai", "openai_judge"], default="mlx"
    )
    parser.add_argument("--dummy-strategy", choices=list(STRATEGIES), default="always_reliable")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--max-context-chars", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--profile",
        choices=["local", "cloud"],
        default="local",
        help="cloud enables the data-leak guard (openai_judge backend)",
    )
    parser.add_argument(
        "--allow-real-data",
        action="store_true",
        help="explicit data-owner opt-in to send real samples with --profile cloud",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="parallel requests for the openai_judge backend (1 = sync client)",
    )
    return parser.parse_args()


def _prob_prediction(sample_id: str, p_faith: float, p_rel: float, meta: dict) -> Prediction:
    """Probabilities -> binary contract fields at 0.5, probabilities kept alongside."""
    return Prediction(
        id=sample_id,
        faithfulness_pred=int(p_faith >= 0.5),
        relevance_pred=int(p_rel >= 0.5),
        raw_output=str(meta.get("raw", "")),
        invalid_output=meta.get("method") == "default",
        faithfulness_prob=p_faith,
        relevance_prob=p_rel,
        prob_method=str(meta.get("method", "")),
    )


def run_openai_judge(args: argparse.Namespace, samples: list) -> list[Prediction]:
    """Logprob-probability judge over an OpenAI-compatible endpoint (sync or async)."""
    from rag_reliability.methods.m3.judge_client import (  # noqa: PLC0415
        AsyncJudgeClient,
        JudgeClient,
    )

    system_prompt = build_system_prompt(
        args.mode, examples_path=args.examples, prompt_file=args.prompt_file
    )
    common = {
        "model": args.model,
        "api_base": args.api_base,
        "api_key": os.environ.get(args.api_key_env, ""),
        "profile": args.profile,
        "allow_real": args.allow_real_data,
    }
    items = [(s, build_user_prompt(s, args.max_context_chars)) for s in samples]

    if args.concurrency > 1:
        import asyncio  # noqa: PLC0415

        client = AsyncJudgeClient(
            cache_dir=args.cache_dir, concurrency=args.concurrency, **common
        )
        results = asyncio.run(client.judge_many(system_prompt, items, max_tokens=args.max_tokens))
    else:
        sync_client = JudgeClient(cache_dir=args.cache_dir, **common)
        results = [
            sync_client.judge(system_prompt, user, sample=s, max_tokens=args.max_tokens)
            for s, user in tqdm(items, desc="m3/openai_judge")
        ]
    return [
        _prob_prediction(sample.id, p_f, p_r, meta)
        for (sample, _), (p_f, p_r, meta) in zip(items, results, strict=True)
    ]


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.data)
    if args.limit is not None:
        samples = samples[: args.limit]

    if args.backend == "openai_judge":
        predictions = run_openai_judge(args, samples)
        save_jsonl(predictions, args.output)
        invalid = sum(prediction.invalid_output for prediction in predictions)
        print(f"Wrote {len(predictions)} predictions to {args.output} (invalid outputs: {invalid})")
        return

    if args.backend == "dummy":
        predictor = DummyPredictor(strategy=args.dummy_strategy, mode="direct")
        generate_fn = None
        chat_client = None
    elif args.backend == "openai":
        from rag_reliability.methods.m3.openai_client import CachedChatClient  # noqa: PLC0415

        predictor = None
        generate_fn = None
        chat_client = CachedChatClient(
            model=args.model,
            api_base=args.api_base,
            api_key=os.environ.get(args.api_key_env, ""),
            cache_dir=args.cache_dir,
        )
    else:
        predictor = None
        generate_fn = make_generate_fn(args.model, args.max_tokens)
        chat_client = None

    system_prompt = build_system_prompt(
        args.mode,
        examples_path=args.examples,
        prompt_file=args.prompt_file,
    )

    predictions: list[Prediction] = []
    for sample in tqdm(samples, desc=f"m3/{args.backend}"):
        if predictor is not None:
            raw_output = predictor.predict(sample)
            prediction = parse_prediction(raw_output, sample.id)
        else:
            user_prompt = build_user_prompt(sample, args.max_context_chars)
            if chat_client is not None:
                raw_output = chat_client.chat(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=args.max_tokens,
                )
            else:
                prompt = f"{system_prompt}\n\n{user_prompt}"
                raw_output = generate_fn(prompt)
            prediction = parse_m3_prediction(raw_output, sample.id)
        predictions.append(prediction)

    save_jsonl(predictions, args.output)
    invalid = sum(prediction.invalid_output for prediction in predictions)
    print(f"Wrote {len(predictions)} predictions to {args.output} (invalid outputs: {invalid})")


if __name__ == "__main__":
    main()
