#!/usr/bin/env python
"""Единая точка инференса: метод -> scores.jsonl по всему корпусу.

Метод производит скоры по всем кейсам; сплит применяется на этапе оценки
(`scripts/evaluate_cv.py`), а не инференса. Поэтому здесь нет ни --split, ни
val/test: один прогон -> один артефакт, который переживает смену протокола.

    python scripts/score.py --method m3_openai_judge --variant zero_shot \\
        --data data/organizers.jsonl \\
        --output predictions/alfa/m3/zero_shot/scores.jsonl [--limit N] [--resume]

Прогон идемпотентен и пригоден к прерыванию: строки пишутся по мере счёта,
--resume дочитывает уже посчитанные id и досчитывает остальные.
"""

from __future__ import annotations

import argparse
import datetime
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import yaml
from tqdm import tqdm

from rag_reliability.dataset import load_jsonl
from rag_reliability.methods import registry
from rag_reliability.run_meta import git_state
from rag_reliability.schema import Prediction, RagSample

DEFAULT_FLUSH_EVERY = 20


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, help=f"One of: {', '.join(registry.METHODS)}")
    parser.add_argument("--variant", required=True, help="Run label, e.g. zero_shot")
    parser.add_argument("--data", required=True, help="Corpus JSONL (RagSample records)")
    parser.add_argument("--output", required=True, help="Where to write scores.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Smoke run over the first N cases")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run: keep scored ids, compute the rest",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=DEFAULT_FLUSH_EVERY,
        help="Flush the output file every N rows (interrupted runs stay readable)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Сколько кейсов считать одновременно. >1 имеет смысл только для методов, "
            "упирающихся в сеть (судья на vLLM); локальные модели он не ускорит"
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Recorded in run.yaml")
    parser.add_argument("--run-yaml", default=None, help="Default: run.yaml next to --output")

    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--direct-adapter-path", default="results/adapters_direct")
    parser.add_argument("--marker-adapter-path", default="results/adapters_marker")
    parser.add_argument("--lettucedetect-model", default="results/lettucedetect/classifier.joblib")
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
    parser.add_argument(
        "--m3-concurrency",
        type=int,
        default=1,
        help=(
            "Параллельные запросы внутри одного кейса. Работает там, где кейс "
            "порождает несколько вызовов (--method m3_perchunk: по вызову на чанк)"
        ),
    )
    parser.add_argument("--m3-dummy-strategy", default="always_reliable")
    parser.add_argument("--independent-faithfulness-threshold", type=float, default=0.20)
    parser.add_argument("--independent-relevance-threshold", type=float, default=0.10)
    return parser.parse_args(argv)


def build_context(args: argparse.Namespace) -> registry.CommandContext:
    output = Path(args.output)
    return registry.CommandContext(
        data=Path(args.data),
        run_dir=output.parent,
        predictions_path=output,
        model=args.model,
        max_tokens=args.max_tokens,
        direct_adapter_path=args.direct_adapter_path,
        marker_adapter_path=args.marker_adapter_path,
        lettucedetect_model=args.lettucedetect_model,
        m3_backend=args.m3_backend,
        m3_max_tokens=args.m3_max_tokens,
        m3_max_context_chars=args.m3_max_context_chars,
        m3_examples=args.m3_examples,
        m3_prompt_file=args.m3_prompt_file,
        m3_api_base=args.m3_api_base,
        m3_api_key_env=args.m3_api_key_env,
        m3_cache_dir=args.m3_cache_dir,
        m3_concurrency=args.m3_concurrency,
        m3_dummy_strategy=args.m3_dummy_strategy,
        independent_faithfulness_threshold=args.independent_faithfulness_threshold,
        independent_relevance_threshold=args.independent_relevance_threshold,
        limit=args.limit,
    )


def scored_ids(path: str | Path) -> list[str]:
    """id уже посчитанных кейсов; хвост-огрызок после SIGKILL отбрасывается.

    Прогоны на DataSphere обрываются в произвольный момент, в том числе на
    середине строки. Считать такую строку валидной нельзя: кейс будет молча
    пропущен и не пересчитан.
    """
    path = Path(path)
    if not path.exists():
        return []
    ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.endswith("\n") or not line.strip():
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if not isinstance(row, dict) or "id" not in row:
                break
            ids.append(str(row["id"]))
    return ids


def _truncate_after(path: Path, n_rows: int) -> None:
    """Оставить ровно n_rows целых строк, отрезав недописанный хвост."""
    with path.open(encoding="utf-8") as handle:
        kept = [next(handle) for _ in range(n_rows)]
    path.write_text("".join(kept), encoding="utf-8")


def _scored_in_order(
    samples: Sequence[RagSample], scorer: registry.Scorer, workers: int
) -> Iterator[Prediction]:
    """Предсказания строго в порядке корпуса, считанные в ``workers`` потоков.

    Порядок обязателен: строки пишутся по мере готовности, и ``--resume``
    дочитывает файл сверху. Если порядок «как посчиталось», то обрыв оставит
    дыру в середине, а перезапуск её не заметит.

    Потоки, а не процессы: узкое место здесь — ожидание ответа vLLM, а скореры
    держат сетевой клиент, который через границу процесса не переживёт.
    """
    if workers <= 1:
        yield from (scorer(sample) for sample in samples)
        return
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(scorer, samples)


def score_samples(
    samples: list[RagSample],
    scorer: registry.Scorer,
    output: str | Path,
    *,
    resume: bool = False,
    flush_every: int = DEFAULT_FLUSH_EVERY,
    progress: bool = True,
    workers: int = 1,
) -> int:
    """Посчитать кейсы и дописать их в output. Возвращает число строк в файле."""
    if flush_every < 1:
        raise ValueError(f"--flush-every must be >= 1, got {flush_every}")
    if workers < 1:
        raise ValueError(f"--workers must be >= 1, got {workers}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if resume:
        already = scored_ids(output)
        done = set(already)
        if output.exists():
            _truncate_after(output, len(already))
    elif output.exists():
        output.unlink()

    pending = [sample for sample in samples if sample.id not in done]
    n_written = len(done)
    mode = "a" if resume else "w"
    with output.open(mode, encoding="utf-8") as handle:
        iterator = tqdm(
            zip(pending, _scored_in_order(pending, scorer, workers), strict=True),
            desc="score",
            total=len(pending),
            disable=not progress,
        )
        for index, (sample, prediction) in enumerate(iterator, start=1):
            if prediction.id != sample.id:
                raise ValueError(
                    f"Scorer returned prediction id {prediction.id!r} for sample {sample.id!r}"
                )
            handle.write(json.dumps(prediction.model_dump(), ensure_ascii=False) + "\n")
            n_written += 1
            if index % flush_every == 0:
                handle.flush()
    return n_written


def write_run_yaml(
    path: str | Path,
    args: argparse.Namespace,
    spec: registry.MethodSpec,
    *,
    n: int,
    partial: bool,
) -> None:
    """Провенанс прогона рядом с артефактом: аргументы, git, seed, контракт метода."""
    payload = {
        "method": {
            "name": spec.name,
            "version": registry.contract_version(spec),
            "family": spec.family,
            "mode": spec.mode,
            "score_keys": list(spec.score_keys),
            "default_score_expr": spec.default_score_expr,
            "corpus_wide": spec.corpus_wide,
        },
        "variant": args.variant,
        "config": {key: str(value) for key, value in sorted(vars(args).items())},
        "git": dict(git_state()),
        "seed": args.seed,
        "n": n,
        "partial": partial,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    spec = registry.get(args.method)
    ctx = build_context(args)

    samples: list[RagSample] = load_jsonl(args.data)
    n_corpus = len(samples)
    if args.limit is not None:
        samples = samples[: args.limit]
    partial = args.limit is not None or len(samples) < n_corpus

    scorer = registry.build_scorer(args.method, ctx)
    n = score_samples(
        samples,
        scorer,
        args.output,
        resume=args.resume,
        flush_every=args.flush_every,
        workers=args.workers,
    )

    run_yaml = Path(args.run_yaml) if args.run_yaml else Path(args.output).parent / "run.yaml"
    write_run_yaml(run_yaml, args, spec, n=n, partial=partial)

    registry.validate_scores_file(args.output, spec, expected_n=len(samples))
    print(f"Wrote {n} scored case(s) to {args.output} (partial={partial}); meta: {run_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
