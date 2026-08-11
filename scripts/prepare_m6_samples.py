#!/usr/bin/env python
"""Generate Method 6 answer samples into a per-sample JSON cache.

MLX generation is greedy and therefore yields identical samples. Use the
``openai`` backend for real Method 6 sampling with nonzero temperature.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from rag_reliability.dataset import load_jsonl
from rag_reliability.methods.m3.judge_client import LLMClient
from rag_reliability.mlx_backend import make_generate_fn
from rag_reliability.schema import RagSample

BOT_SYSTEM = (
    "Ты — ассистент банка для корпоративных клиентов. Отвечай на вопрос клиента, "
    "используя только предоставленные фрагменты документации. Если ответа в "
    "фрагментах нет, скажи об этом."
)
BOT_USER = "Фрагменты документации:\n{context}\n\nВопрос клиента: {question}\n\nОтвет:"


def build_bot_prompt(sample: RagSample, max_context_chars: int | None = None) -> str:
    context = sample.context
    if max_context_chars is not None and len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n[контекст усечён]"
    return f"{BOT_SYSTEM}\n\n{BOT_USER.format(context=context, question=sample.question)}"


def need_samples(cache_file: Path, target: int) -> tuple[int, list[str]]:
    existing: list[str] = []
    if cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        raw_samples = payload.get("samples") or []
        if not isinstance(raw_samples, list) or not all(
            isinstance(sample, str) for sample in raw_samples
        ):
            raise ValueError(f"Invalid Method 6 sample cache at {cache_file}")
        existing = raw_samples
    return max(0, target - len(existing)), existing


def write_sample_cache(output_dir: str | Path, sample_id: str, samples: list[str]) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_file = output_path / f"{sample_id}.json"
    payload = json.dumps({"id": sample_id, "samples": samples}, ensure_ascii=False)
    tmp = cache_file.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(cache_file)


def make_openai_batch_fn(
    *,
    model: str,
    api_base: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Callable[[str, int], list[str]]:
    """Sample k answers per prompt via one n=k request (retry + backfill in LLMClient)."""
    client = LLMClient(model=model, api_base=api_base, api_key=api_key)

    def generate_batch(prompt: str, k: int) -> list[str]:
        choices = client.chat(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            n=k,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        return [choice["text"] for choice in choices]

    return generate_batch


def build_sample_cache(
    samples: list[RagSample],
    *,
    output_dir: str | Path,
    generate_fn: Callable[[str], str] | None = None,
    generate_batch_fn: Callable[[str, int], list[str]] | None = None,
    n_samples: int,
    max_context_chars: int | None = None,
) -> None:
    if (generate_fn is None) == (generate_batch_fn is None):
        raise ValueError("provide exactly one of generate_fn / generate_batch_fn")
    output_path = Path(output_dir)
    for sample in tqdm(samples, desc="m6/samples"):
        cache_file = output_path / f"{sample.id}.json"
        needed, existing = need_samples(cache_file, n_samples)
        if needed == 0:
            continue
        prompt = build_bot_prompt(sample, max_context_chars=max_context_chars)
        if generate_batch_fn is not None:
            generated = generate_batch_fn(prompt, needed)
        else:
            generated = [generate_fn(prompt) for _ in range(needed)]
        write_sample_cache(output_path, sample.id, existing + generated)


def dummy_generate(prompt: str) -> str:
    return prompt.rsplit("Ответ:", maxsplit=1)[-1].strip() or "dummy answer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dummy.jsonl")
    parser.add_argument("--output-dir", default="results/m6/samples")
    parser.add_argument("--backend", choices=["dummy", "mlx", "openai"], default="mlx")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-context-chars", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.data)
    if args.limit is not None:
        samples = samples[: args.limit]

    generate_batch_fn = None
    if args.backend == "dummy":
        generate_fn = dummy_generate
    elif args.backend == "mlx":
        generate_fn = make_generate_fn(args.model, args.max_tokens)
    else:
        generate_fn = None
        generate_batch_fn = make_openai_batch_fn(
            model=args.model,
            api_base=args.api_base,
            api_key=os.environ.get(args.api_key_env, ""),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    build_sample_cache(
        samples,
        output_dir=args.output_dir,
        generate_fn=generate_fn,
        generate_batch_fn=generate_batch_fn,
        n_samples=args.n_samples,
        max_context_chars=args.max_context_chars,
    )
    print(f"Prepared Method 6 samples for {len(samples)} case(s) under {args.output_dir}")


if __name__ == "__main__":
    main()
