#!/usr/bin/env python
"""OOF-обучение энкодера надёжности: артефакт scores.jsonl по всему оцениваемому корпусу.

    python scripts/train_encoder_baseline.py --variant len8192_lr2e-5 \\
        --max-length 8192 --learning-rate 2e-5 --batch-size 1 --grad-accum 8

Скрипт — тонкая обёртка: вся логика в ``rag_reliability.methods.encoder``.
Собственного сплита здесь нет и быть не может — разбиение приходит из
``folds.json``.

Артефакт покрывает корпус целиком. 1480 кейсов, которым ``folds.json`` дал
фолд, предсказаны честным out-of-fold; оставшиеся 753 — это ровно одна
oversized-группа, целиком отсутствующая в train-части любого фолда, и их
скорит среднее по моделям фолдов. Источник скора помечен в ``prob_method`` и
посчитан в ``run.yaml``; в метрики эти строки не попадают — ``evaluate_cv``
отбрасывает их по отсутствию фолда.

Оценка — отдельным шагом; порог здесь не подбирается:

    python scripts/evaluate_cv.py --data data/alfa.jsonl \\
        --folds data/splits/folds_alfa.json \\
        --score-expr "enc.prob" --faith-expr "enc.prob" --rel-expr "enc.prob" \\
        --scores predictions/alfa/encoder/<variant>/scores.jsonl \\
        --output predictions/alfa/encoder/<variant>/report.json

Оси задаются явно и одним выражением: энкодер предсказывает ``reliable``
целиком, отдельного сигнала на faithfulness и relevance у него нет.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import score  # noqa: E402

from rag_reliability.dataset import load_jsonl  # noqa: E402
from rag_reliability.methods import registry  # noqa: E402
from rag_reliability.methods.encoder.predict import (  # noqa: E402
    checkpoint_meta,
    logits_to_predictions,
    write_scores,
)
from rag_reliability.methods.encoder.train import (  # noqa: E402
    POS_WEIGHT_MODES,
    FoldTrainer,
    TrainConfig,
    train_oof_detailed,
)
from rag_reliability.methods.surface.oof import corpus_sha256, load_folds  # noqa: E402

METHOD = "encoder"
DEFAULT_OUTPUT_ROOT = Path("predictions/alfa/encoder")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="default", help="Метка прогона, напр. len8192_lr2e-5")
    # Канонический корпус волны 2 — alfa.jsonl (2233) с folds_alfa.json; на нём
    # прогнаны surface/majority, и только числа с одного folds.json сравнимы.
    parser.add_argument("--data", default="data/alfa.jsonl", help="Корпус RagSample jsonl")
    parser.add_argument("--folds", default="data/splits/folds_alfa.json")
    parser.add_argument(
        "--repeat", type=int, default=0, help="Номер повтора folds.json; обучается ровно один"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Диагностика прогона (json). По умолчанию — рядом с scores.jsonl",
    )
    parser.add_argument(
        "--predictions-output",
        default=None,
        help="scores.jsonl. По умолчанию predictions/alfa/encoder/<variant>/scores.jsonl",
    )
    parser.add_argument("--run-yaml", default=None, help="По умолчанию run.yaml рядом с артефактом")
    parser.add_argument("--model", default="deepvk/RuModernBERT-base")
    parser.add_argument("--output-dir", default="results/encoder_checkpoints")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--pos-weight-mode", choices=list(POS_WEIGHT_MODES), default="none")
    parser.add_argument("--limit", type=int, default=None, help="Смоук по первым N кейсам")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        model=args.model,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        pos_weight_mode=args.pos_weight_mode,
        weight_decay=args.weight_decay,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """scores.jsonl, run.yaml и файл диагностики — все три рядом, если не заданы."""
    scores_path = (
        Path(args.predictions_output)
        if args.predictions_output
        else DEFAULT_OUTPUT_ROOT / args.variant / "scores.jsonl"
    )
    run_yaml = Path(args.run_yaml) if args.run_yaml else scores_path.parent / "run.yaml"
    diagnostics = (
        Path(args.output) if args.output else scores_path.parent / "encoder_diagnostics.json"
    )
    return scores_path, run_yaml, diagnostics


def _append_encoder_meta(
    run_yaml: Path,
    *,
    config: TrainConfig,
    diagnostics: dict,
    checkpoints: list[dict],
    coverage: dict,
) -> None:
    """Гиперпараметры, вердикт о схлопывании и покрытие — типами, а не строками.

    ``score.write_run_yaml`` кладёт argparse-аргументы строками; сравнивать по
    такому файлу конфигурации нельзя, а ``collapsed`` обязан читаться однозначно:
    именно на схлопнувшемся прогоне был сделан прежний вывод про 1024 токена.
    """
    import yaml  # noqa: PLC0415

    payload = yaml.safe_load(run_yaml.read_text(encoding="utf-8"))
    payload["encoder"] = {
        "n_repeats": 1,
        "repeat": diagnostics["repeat"],
        "collapsed": diagnostics["collapsed"],
        "collapse_reason": diagnostics["collapse_reason"],
        "collapsed_folds": diagnostics["collapsed_folds"],
        "const_share": diagnostics["const_share"],
        "output_entropy": diagnostics["output_entropy"],
        "model": config.model,
        "max_length": config.max_length,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "batch_size": config.batch_size,
        "grad_accum": config.grad_accum,
        "epochs": config.epochs,
        "pos_weight_mode": config.pos_weight_mode,
        "weight_decay": config.weight_decay,
        "seed": config.seed,
        "checkpoints": checkpoints,
        "epoch_log": diagnostics["epochs"],
    }
    payload["coverage"] = coverage
    run_yaml.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None, *, train_fold: FoldTrainer | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    spec = registry.get(METHOD)
    config = build_config(args)
    scores_path, run_yaml, diagnostics_path = resolve_paths(args)

    samples = load_jsonl(args.data)
    n_corpus = len(samples)
    if args.limit is not None:
        samples = samples[: args.limit]

    folds = load_folds(args.folds)
    result = train_oof_detailed(
        samples, folds, config, repeat=args.repeat, train_fold=train_fold
    )
    diagnostics = result.diagnostics()
    n_written = write_scores(
        logits_to_predictions(result.logits, ensemble_ids=result.ensemble_ids), scores_path
    )

    # Артефакт покрывает корпус целиком; partial бывает только под --limit.
    partial = n_written < n_corpus
    score.write_run_yaml(run_yaml, args, spec, n=n_written, partial=partial)
    _append_encoder_meta(
        run_yaml,
        config=config,
        diagnostics=diagnostics,
        checkpoints=checkpoint_meta(result.checkpoints),
        coverage={
            "corpus_n": n_corpus,
            "scored_n": n_written,
            "excluded_n": n_corpus - n_written,
            "oof_n": diagnostics["n_oof"],
            "fold_ensemble_n": diagnostics["n_ensemble"],
            "reason": "cases outside the folds (one oversized group) are scored by the mean "
            "of the fold models: that group is absent from every training part, so no leak",
            "corpus_sha256": corpus_sha256(args.data),
        },
    )

    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    registry.validate_scores_file(scores_path, spec, expected_n=n_written)

    print(
        f"Wrote {n_written} OOF logit(s) of {n_corpus} corpus case(s) to {scores_path}; "
        f"collapsed={diagnostics['collapsed']} const_share={diagnostics['const_share']:.4f}; "
        f"meta: {run_yaml}"
    )
    if diagnostics["collapsed"]:
        print(
            f"WARNING: прогон схлопнулся ({diagnostics['collapse_reason']}, "
            f"фолды {diagnostics['collapsed_folds']}) и исключается из выбора "
            "лучшей конфигурации"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
