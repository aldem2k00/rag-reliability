"""Материализация и валидация канонического group-aware разбиения корпуса.

Два режима::

    python scripts/prepare_splits.py --data data/organizers.jsonl \
        --output data/splits/folds.json
    python scripts/prepare_splits.py --check --folds data/splits/folds.json \
        --data data/organizers.jsonl

Скрипт больше не вызывает ``dataset.split_samples``: стратифицированное
разбиение без учёта групп протекает (24.9% строк делят вопрос клиента с train),
и под ним 1-NN-запоминатель обгоняет все методы проекта. Разбиение теперь
единственное и живёт в ``data/splits/folds.json``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rag_reliability.dataset import load_jsonl
from rag_reliability.schema import RagSample
from rag_reliability.splits import (
    DEFAULT_N_FOLDS,
    DEFAULT_N_REPEATS,
    DEFAULT_NEAR_DUP_THRESHOLD,
    DEFAULT_SEED,
    FoldConfig,
    assign_folds,
    build_groups,
    check_folds,
    write_folds,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Разобрать опции генерации и валидации фолдов."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/organizers.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/splits/folds.json"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="валидировать существующий folds.json вместо генерации",
    )
    parser.add_argument("--folds", type=Path, default=Path("data/splits/folds.json"))
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--n-repeats", type=int, default=DEFAULT_N_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--near-dup-threshold", type=float, default=DEFAULT_NEAR_DUP_THRESHOLD)
    parser.add_argument(
        "--no-chunk-key",
        dest="use_chunk_key",
        action="store_false",
        help="не склеивать кейсы по общему chunk_1",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke-прогон на первых N кейсах; артефакт получается неканоническим",
    )
    return parser.parse_args(argv)


def _generate(args: argparse.Namespace, samples: list[RagSample]) -> int:
    config = FoldConfig(
        corpus_path=args.data,
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
        seed=args.seed,
        near_dup_threshold=args.near_dup_threshold,
        use_chunk_key=args.use_chunk_key,
    )
    groups = build_groups(
        samples,
        near_dup_threshold=config.near_dup_threshold,
        use_chunk_key=config.use_chunk_key,
    )
    assignment = assign_folds(
        samples,
        groups,
        n_folds=config.n_folds,
        n_repeats=config.n_repeats,
        seed=config.seed,
    )
    write_folds(args.output, samples, groups, assignment, config)

    report = check_folds(args.output, samples, corpus_path=args.data)
    stats = report["recomputed_stats"]
    print(f"folds -> {args.output}")
    print(f"  cases              {len(samples)} ({len(assignment)} assigned)")
    print(f"  groups             {stats['n_groups']} (largest {stats['largest_group']})")
    print(f"  oversized groups   {stats['oversized_groups'] or '-'}")
    print(f"  excluded cases     {stats['excluded_ids']}")
    print(f"  pos rate global    {stats['pos_rate_global']:.4f}")
    print(f"  leak_check         {stats['leak_check']}")
    _print_findings(report)
    return 0 if report["passed"] else 1


def _validate(args: argparse.Namespace, samples: list[RagSample]) -> int:
    report = check_folds(args.folds, samples, corpus_path=args.data)
    stats = report["recomputed_stats"]
    print(f"check {args.folds}: {'PASS' if report['passed'] else 'FAIL'}")
    if stats is not None:
        print(f"  groups             {stats['n_groups']} (largest {stats['largest_group']})")
        print(f"  excluded cases     {stats['excluded_ids']}")
        print(f"  pos rate global    {stats['pos_rate_global']:.4f}")
        for repeat, rates in enumerate(stats["pos_rate_by_fold"]):
            formatted = ", ".join(f"{rate:.4f}" for rate in rates)
            print(
                f"  pos rate repeat {repeat}  [{formatted}]  sizes {stats['size_by_fold'][repeat]}"
            )
        print(f"  leak_check         {stats['leak_check']}")
    _print_findings(report)
    return 0 if report["passed"] else 1


def _print_findings(report: dict) -> None:
    for warning in report["warnings"]:
        print(f"  WARNING  {warning}")
    for error in report["errors"]:
        print(f"  ERROR    {error}")


def main(argv: Sequence[str] | None = None) -> int:
    """Сгенерировать или проверить ``folds.json``; 1 — провал блокирующей проверки."""
    args = parse_args(argv)
    samples = load_jsonl(args.data)
    if args.limit is not None:
        samples = samples[: args.limit]
    if args.check:
        return _validate(args, samples)
    return _generate(args, samples)


if __name__ == "__main__":
    raise SystemExit(main())
