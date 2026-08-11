#!/usr/bin/env python
"""Fine-tuning судьи Метода 3 на одном фолде ``folds.json``.

    python scripts/train_ft_judge.py \\
        --data data/alfa.jsonl --folds data/splits/folds_alfa.json --fold 0 \\
        --model Qwen/Qwen2.5-7B-Instruct --mode direct \\
        --predictions-output predictions/alfa/ft_judge/direct_fold0/scores.jsonl

Скрипт — тонкая обёртка: вся логика в ``rag_reliability.methods.ft_judge``.
Собственного сплита здесь нет и быть не может — фолд приходит из ``folds.json``,
``split_samples`` не вызывается.

Один вызов обучает один фолд и скорит его held-out часть. Пять фолдов — пять
заданий; склейка OOF по всем фолдам делается отдельно, из пяти артефактов.

``--smoke-only`` собирает обучающие примеры, проверяет симметрию формата и
печатает баланс классов, не загружая модель: на этом шаге ловится и кривой
промпт, и перекос 72/28, из-за которого прошлый прогон схлопнулся.

Оценка — отдельным шагом; порог здесь не подбирается:

    python scripts/evaluate_cv.py --data data/alfa.jsonl \\
        --folds data/splits/folds_alfa.json \\
        --scores predictions/alfa/ft_judge/direct_fold0/scores.jsonl \\
        --output predictions/alfa/ft_judge/direct_fold0/report.json
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
from rag_reliability.methods.ft_judge.data import (  # noqa: E402
    MODES,
    build_examples,
    class_balance,
    compute_pos_weight,
)
from rag_reliability.methods.ft_judge.predict import (  # noqa: E402
    probs_to_predictions,
    write_scores,
)
from rag_reliability.methods.ft_judge.train import (  # noqa: E402
    POS_WEIGHT_MODES,
    SAVE_STRATEGIES,
    TUNINGS,
    FoldTrainer,
    FtConfig,
    train_one_fold,
)
from rag_reliability.methods.surface.oof import corpus_sha256, load_folds  # noqa: E402

METHOD = "ft_judge"
DEFAULT_OUTPUT_ROOT = Path("predictions/alfa/ft_judge")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/alfa.jsonl", help="Корпус RagSample jsonl")
    parser.add_argument("--folds", default="data/splits/folds_alfa.json")
    parser.add_argument("--fold", type=int, required=True, help="Номер обучаемого фолда")
    parser.add_argument("--repeat", type=int, default=0, help="Номер повтора folds.json")
    parser.add_argument(
        "--variant",
        default=None,
        help="Метка прогона. По умолчанию <mode>_fold<N>",
    )

    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--mode", choices=list(MODES), default="direct")
    parser.add_argument("--tuning", choices=list(TUNINGS), default="lora")
    parser.add_argument("--lora-r", type=int, default=256)
    parser.add_argument("--lora-alpha", type=int, default=512)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="all-linear")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--pos-weight-mode", choices=list(POS_WEIGHT_MODES), default="balanced")
    parser.add_argument(
        "--oversample-negatives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Доложить FAIL-примеры до паритета с PASS (дисбаланс 72/28 уже ронял прогон)",
    )
    parser.add_argument("--save-strategy", choices=list(SAVE_STRATEGIES), default="epoch")
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompts-dir", default=None, help="По умолчанию configs/prompts")
    parser.add_argument(
        "--allow-small-gpu",
        action="store_true",
        help="Снять ассерт VRAM >= 70 GB. Только для смоука, не для отчётных прогонов",
    )
    parser.add_argument("--push-to-hub", default=None, help="Репозиторий HF Hub для адаптера")

    parser.add_argument("--output-dir", default=None, help="Куда класть чекпоинты")
    parser.add_argument("--predictions-output", default=None, help="scores.jsonl")
    parser.add_argument("--diagnostics-output", default=None, help="ft_diagnostics.json")
    parser.add_argument("--run-yaml", default=None, help="По умолчанию run.yaml рядом с артефактом")
    parser.add_argument("--limit", type=int, default=None, help="Смоук по первым N кейсам")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Продолжить с последнего сохранённого чекпоинта эпохи",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Собрать примеры и проверить формат, не загружая модель",
    )
    return parser.parse_args(argv)


def variant_name(args: argparse.Namespace) -> str:
    return args.variant or f"{args.mode}_fold{args.fold}"


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """scores.jsonl, run.yaml, диагностика и каталог чекпоинтов."""
    variant = variant_name(args)
    scores_path = (
        Path(args.predictions_output)
        if args.predictions_output
        else DEFAULT_OUTPUT_ROOT / variant / "scores.jsonl"
    )
    run_yaml = Path(args.run_yaml) if args.run_yaml else scores_path.parent / "run.yaml"
    diagnostics = (
        Path(args.diagnostics_output)
        if args.diagnostics_output
        else scores_path.parent / "ft_diagnostics.json"
    )
    checkpoints = Path(args.output_dir) if args.output_dir else Path("results/ft_judge") / variant
    return scores_path, run_yaml, diagnostics, checkpoints


def build_config(args: argparse.Namespace, output_dir: Path) -> FtConfig:
    return FtConfig(
        model=args.model,
        mode=args.mode,
        tuning=args.tuning,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        epochs=args.epochs,
        max_length=args.max_length,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        weight_decay=args.weight_decay,
        pos_weight_mode=args.pos_weight_mode,
        oversample_negatives=args.oversample_negatives,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        output_dir=str(output_dir),
        prompts_dir=args.prompts_dir,
        allow_small_gpu=args.allow_small_gpu,
        push_to_hub=args.push_to_hub,
    )


def run_smoke(args: argparse.Namespace, config: FtConfig) -> int:
    """Формат и баланс классов до загрузки весов.

    Симметрия формата проверяется внутри ``build_examples``: обучающее
    завершение обязано читаться тем же парсером, что и ответ модели. Прогон,
    в котором это не так, теряет логпробы вердикта и выглядит успешным.
    """
    samples = load_jsonl(args.data)
    if args.limit is not None:
        samples = samples[: args.limit]
    examples = build_examples(samples, mode=config.mode, prompts_dir=config.prompts_dir)
    balance = class_balance(examples)
    pos_weight = compute_pos_weight(
        [example.label for example in examples], config.pos_weight_mode
    )
    first = examples[0]
    print(f"Cases: {len(samples)}; examples: {len(examples)} ({len(examples) // max(len(samples), 1)} per case)")
    for axis, counts in sorted(balance.items()):
        total = counts["pass"] + counts["fail"]
        share = counts["fail"] / total if total else 0.0
        print(f"  {axis}: PASS={counts['pass']} FAIL={counts['fail']} (FAIL share {share:.1%})")
    print(f"pos_weight ({config.pos_weight_mode}) = {pos_weight:.4f}")
    print(f"Format symmetry: OK for all {len(examples)} example(s)")
    print(f"--- sample prompt ({first.axis}, {first.sample_id}) ---")
    print(first.system[:400])
    print("...")
    print(first.user[:400])
    print(f"--- target completion ---\n{first.completion}")
    return 0


def main(argv: Sequence[str] | None = None, *, train_fold: FoldTrainer | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    scores_path, run_yaml, diagnostics_path, checkpoint_dir = resolve_paths(args)
    config = build_config(args, checkpoint_dir)

    if args.smoke_only:
        return run_smoke(args, config)

    spec = registry.get(METHOD)
    samples = load_jsonl(args.data)
    n_corpus = len(samples)
    if args.limit is not None:
        samples = samples[: args.limit]

    folds = load_folds(args.folds)
    result = train_one_fold(
        samples,
        folds,
        config,
        fold=args.fold,
        repeat=args.repeat,
        resume=args.resume,
        train_fold=train_fold,
    )
    diagnostics = result.diagnostics()
    n_written = write_scores(probs_to_predictions(result.probs), scores_path)

    # Артефакт покрывает held-out часть одного фолда — по построению это часть
    # корпуса, а не весь корпус.
    score.write_run_yaml(run_yaml, args, spec, n=n_written, partial=True)
    _append_ft_meta(
        run_yaml,
        config=config,
        diagnostics=diagnostics,
        coverage={
            "corpus_n": n_corpus,
            "scored_n": n_written,
            "fold": args.fold,
            "repeat": args.repeat,
            "n_folds": folds.n_folds,
            "reason": "one fold per run: the artifact covers this fold's held-out part; "
            "the full OOF is the concatenation of all folds",
            "corpus_sha256": corpus_sha256(args.data),
        },
    )

    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    registry.validate_scores_file(scores_path, spec, expected_n=n_written)

    print(
        f"Wrote {n_written} held-out case(s) of fold {args.fold} to {scores_path}; "
        f"collapsed={diagnostics['collapsed']} const_share={diagnostics['const_share']:.4f}; "
        f"meta: {run_yaml}"
    )
    if diagnostics["collapsed"]:
        print(
            f"WARNING: прогон схлопнулся ({diagnostics['collapse_reason']}) и исключается "
            "из выбора лучшей конфигурации"
        )
    return 0


def _append_ft_meta(
    run_yaml: Path, *, config: FtConfig, diagnostics: dict, coverage: dict
) -> None:
    """Гиперпараметры и вердикт о схлопывании — типами, а не строками.

    ``score.write_run_yaml`` кладёт argparse-аргументы строками; сравнивать по
    такому файлу конфигурации нельзя, а ``collapsed`` обязан читаться однозначно.
    """
    import yaml  # noqa: PLC0415

    payload = yaml.safe_load(run_yaml.read_text(encoding="utf-8"))
    payload["ft_judge"] = {
        "collapsed": diagnostics["collapsed"],
        "collapse_reason": diagnostics["collapse_reason"],
        "const_share": diagnostics["const_share"],
        "output_entropy": diagnostics["output_entropy"],
        "pos_weight": diagnostics["pos_weight"],
        "class_balance": diagnostics["class_balance"],
        "n_train_examples": diagnostics["n_train"],
        "checkpoint": diagnostics["checkpoint"],
        "model": config.model,
        "mode": config.mode,
        "tuning": config.tuning,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_target_modules": config.lora_target_modules,
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "max_length": config.max_length,
        "batch_size": config.batch_size,
        "grad_accum": config.grad_accum,
        "pos_weight_mode": config.pos_weight_mode,
        "oversample_negatives": config.oversample_negatives,
        "seed": config.seed,
        "epoch_log": diagnostics["epochs"],
    }
    payload["coverage"] = coverage
    run_yaml.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
