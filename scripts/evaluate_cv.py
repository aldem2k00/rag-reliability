#!/usr/bin/env python
"""Единая точка оценки: 5×5 CV с вложенным подбором порога и отчёт с ДИ.

    python scripts/evaluate_cv.py \\
      --data data/alfa.jsonl \\
      --folds data/splits/folds_alfa.json \\
      --scores predictions/alfa/m3/zero_shot/scores.jsonl \\
      --score-expr "m3.p_faith * m3.p_rel" \\
      --compare predictions/alfa/baselines/surface/scores.jsonl \\
      --output predictions/alfa/m3/zero_shot/report.json

Протокол перестаёт быть свойством каждого скрипта: разбиение читается из
``folds.json``, порог подбирается внутри фолда, а число без 95% ДИ и перцентиля
шума не собирается — это валидирует схема ``EvaluationReport``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rag_reliability.dataset import load_jsonl
from rag_reliability.evaluation.protocol import (
    CVResult,
    ScoreFn,
    build_report,
    check_corpus_hash,
    compare_runs,
    compile_score_expr,
    default_score_fn,
    evaluate_cv,
    evaluate_cv_labeled,
    evaluate_legacy_holdout,
    faith_score_fn,
    has_axis_scores,
    load_folds,
    rel_score_fn,
)
from rag_reliability.schema import Prediction, RagSample


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="корпус с золотыми метками (jsonl)")
    parser.add_argument("--folds", type=Path, help="data/splits/folds.json; обязателен в CV-режиме")
    parser.add_argument("--scores", type=Path, required=True, help="scores.jsonl оцениваемого прогона")
    parser.add_argument(
        "--score-expr",
        default=None,
        help="выражение по ключам scores, например \"m3.p_faith * m3.p_rel\"; "
        "по умолчанию p_faith * p_rel единственного метода в артефакте",
    )
    parser.add_argument("--faith-expr", default=None, help="ось faithfulness (диагностика)")
    parser.add_argument("--rel-expr", default=None, help="ось relevance (диагностика)")
    parser.add_argument(
        "--compare",
        type=Path,
        action="append",
        default=[],
        help="scores.jsonl другого прогона для парного бутстрэпа; можно несколько раз",
    )
    parser.add_argument("--output", type=Path, required=True, help="куда записать report.json")
    parser.add_argument("--method", default=None, help="по умолчанию выводится из пути --scores")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--bootstrap-B", dest="bootstrap_b", type=int, default=10_000)
    parser.add_argument("--null-trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--legacy-holdout",
        action="store_true",
        help="воспроизвести старую схему (единичный val/test + сетка t_faith × t_rel); "
        "только для регресс-теста, в отчётах не использовать",
    )
    parser.add_argument("--legacy-val", type=Path, default=None, help="ids val-сплита (jsonl)")
    parser.add_argument("--legacy-test", type=Path, default=None, help="ids test-сплита (jsonl)")
    args = parser.parse_args(argv)

    if args.legacy_holdout:
        if args.legacy_val is None or args.legacy_test is None:
            parser.error("--legacy-holdout requires --legacy-val and --legacy-test")
    elif args.folds is None:
        parser.error("--folds is required unless --legacy-holdout is given")
    return args


# --------------------------------------------------------------------------- #
# Загрузка артефактов
# --------------------------------------------------------------------------- #


def load_scores(path: str | Path) -> list[Prediction]:
    """Прочитать ``scores.jsonl``.

    Бинарные поля ``*_pred`` в артефакте не хранятся и здесь не нужны: решение
    порождает порог фолда. Отсутствие скоров — ошибка, а не пустой словарь:
    молчаливый дефолт уже трактовал битую строку как идеально надёжный кейс.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scores file not found: {path.resolve()}")
    predictions: list[Prediction] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row:
                raise ValueError(f"Missing 'id' at {path}:{line_number}")
            if "scores" in row:
                scores = {str(key): float(value) for key, value in row["scores"].items()}
            elif "p_faith" in row and "p_rel" in row:
                # Артефакты до A3: вероятности лежали в корне без префикса метода.
                scores = {"legacy.p_faith": float(row["p_faith"]), "legacy.p_rel": float(row["p_rel"])}
            else:
                raise ValueError(
                    f"Prediction at {path}:{line_number} has neither 'scores' nor 'p_faith'/'p_rel'"
                )
            predictions.append(
                Prediction(
                    id=str(row["id"]),
                    faithfulness_pred=0,
                    relevance_pred=0,
                    scores=scores,
                    invalid_output=bool(row["invalid_output"]) if "invalid_output" in row else False,
                )
            )
    if not predictions:
        raise ValueError(f"Scores file {path} is empty")
    return predictions


def read_ids(path: str | Path) -> list[str]:
    """Список id в порядке файла — так задаётся членство в историческом сплите."""
    ids: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row:
                raise ValueError(f"Missing 'id' at {path}:{line_number}")
            ids.append(str(row["id"]))
    return ids


def infer_method_variant(scores_path: Path) -> tuple[str, str]:
    """``predictions/alfa/m3/zero_shot/scores.jsonl`` -> ``("m3", "zero_shot")``."""
    parts = scores_path.resolve().parts
    if len(parts) < 3:
        raise ValueError(f"Cannot infer method/variant from {scores_path}; pass --method/--variant")
    return parts[-3], parts[-2]


def require_full_coverage(
    samples: Sequence[RagSample], predictions: Sequence[Prediction], scores_path: Path
) -> list[Prediction]:
    """Частичный артефакт в CV не участвует до полного прогона (спека §2.2).

    Оценить покрытую часть и напечатать число выглядит безобиднее, чем есть:
    состав кейсов становится свойством того, докуда дошёл прогон, а метрики
    разных методов перестают быть сравнимыми между собой. Поэтому отсутствие
    предсказания — отказ, а не подвыборка.
    """
    by_id = {prediction.id: prediction for prediction in predictions}
    if len(by_id) != len(predictions):
        raise ValueError(f"Duplicate prediction ids in {scores_path}")
    missing = [sample.id for sample in samples if sample.id not in by_id]
    if not missing:
        return [by_id[sample.id] for sample in samples]

    if len(missing) == len(samples):
        raise ValueError(
            f"None of the {len(samples)} corpus ids appear in {scores_path} "
            f"(scores start with {sorted(by_id)[:3]}); the artifact belongs to another corpus"
        )
    raise ValueError(
        f"{scores_path} is partial: {len(by_id)} of {len(samples)} corpus case(s) scored, "
        f"{len(missing)} missing, e.g. {missing[:5]}. Partial artifacts do not take part in CV "
        "until the full run exists (docs/specs/10_PHASE0 §2.2); rerun the method over the whole "
        "corpus and evaluate again."
    )


def _score_fn(expression: str | None, fallback: ScoreFn) -> ScoreFn:
    return compile_score_expr(expression) if expression else fallback


def _faithfulness_label(sample: RagSample) -> int:
    return sample.faithfulness


def _relevance_label(sample: RagSample) -> int:
    return sample.relevance


# --------------------------------------------------------------------------- #
# Режимы
# --------------------------------------------------------------------------- #


def run_cv(args: argparse.Namespace) -> int:
    samples = load_jsonl(args.data)
    folds = load_folds(args.folds)
    corpus_sha = check_corpus_hash(folds, args.data)

    predictions = require_full_coverage(samples, load_scores(args.scores), args.scores)
    score_fn = _score_fn(args.score_expr, default_score_fn)

    primary = evaluate_cv(
        samples,
        predictions,
        folds,
        score_fn=score_fn,
        grid_step=args.grid_step,
    )
    # Поосевая диагностика опускается, когда метод не даёт пары p_faith/p_rel и
    # выражение оси не задано явно: пофрагментная верификация судит только
    # faithfulness. Раньше такой артефакт вообще нельзя было оценить — CLI падал
    # на поиске ключей осей, хотя первичная метрика считалась по --score-expr.
    # Пустой axes допускается схемой EvaluationReport; выдумывать ось из чужого
    # сигнала нельзя — это дало бы правдоподобное число, ничего не измеряющее.
    axes = {}
    axes_available = has_axis_scores(predictions[0])
    for name, expression, fallback, label_fn in (
        ("faithfulness_f1_macro", args.faith_expr, faith_score_fn, _faithfulness_label),
        ("relevance_f1_macro", args.rel_expr, rel_score_fn, _relevance_label),
    ):
        if expression is None and not axes_available:
            print(
                f"{name}: пропущено — в артефакте нет пары '<метод>.p_faith'/'.p_rel', "
                f"а выражение оси не задано (ключи: {sorted(predictions[0].scores)[:5]})"
            )
            continue
        axes[name] = evaluate_cv_labeled(
            samples,
            predictions,
            folds,
            score_fn=_score_fn(expression, fallback),
            grid_step=args.grid_step,
            label_fn=label_fn,
        )

    comparisons = [
        compare_runs(
            primary,
            _comparison_result(args, samples, path, folds),
            _comparison_name(path),
            bootstrap_b=args.bootstrap_b,
            seed=args.seed,
        )
        for path in args.compare
    ]

    method, variant = args.method, args.variant
    if method is None or variant is None:
        inferred = infer_method_variant(args.scores)
        method = method or inferred[0]
        variant = variant or inferred[1]

    report = build_report(
        method=method,
        variant=variant,
        primary=primary,
        axes=axes,
        predictions=predictions,
        protocol={
            "folds": str(args.folds).replace("\\", "/"),
            "folds_sha256": _sha256_of(args.folds),
            "corpus": str(args.data).replace("\\", "/"),
            "corpus_sha256": corpus_sha,
            "n_folds": folds["config"]["n_folds"],
            "n_repeats": folds["config"]["n_repeats"],
            "n_corpus": len(samples),
            "n_evaluated": len(primary.ids),
            "n_excluded": primary.n_excluded,
            "score_expr": args.score_expr or "p_faith * p_rel",
            "scores": str(args.scores).replace("\\", "/"),
        },
        comparisons=comparisons,
        bootstrap_b=args.bootstrap_b,
        null_trials=args.null_trials,
        grid_step=args.grid_step,
        seed=args.seed,
    )
    _write(args.output, report.model_dump())
    _print_summary(report)
    return 0


def _comparison_result(
    args: argparse.Namespace, samples: Sequence[RagSample], path: Path, folds: dict
) -> CVResult:
    """Прогон сравнения оценивается тем же протоколом: иначе Δ мерит разницу протоколов."""
    return evaluate_cv(
        samples,
        require_full_coverage(samples, load_scores(path), path),
        folds,
        score_fn=default_score_fn,
        grid_step=args.grid_step,
    )


def _comparison_name(path: Path) -> str:
    parts = path.resolve().parts
    return "/".join(parts[-3:-1]) if len(parts) >= 3 else path.stem


def _sha256_of(path: Path) -> str:
    from rag_reliability.splits import sha256_file

    return sha256_file(path)


def run_legacy(args: argparse.Namespace) -> int:
    """Старый протокол на исторических сплитах — вход регресс-теста, не отчёт."""
    samples = {sample.id: sample for sample in load_jsonl(args.data)}
    predictions = {prediction.id: prediction for prediction in load_scores(args.scores)}

    def split(path: Path) -> tuple[list[RagSample], list[Prediction]]:
        ids = read_ids(path)
        missing = [sample_id for sample_id in ids if sample_id not in samples]
        if missing:
            raise ValueError(
                f"{len(missing)} id(s) from {path} are absent from {args.data}: {missing[:5]}"
            )
        absent = [sample_id for sample_id in ids if sample_id not in predictions]
        if absent:
            raise ValueError(f"{len(absent)} id(s) from {path} have no scores: {absent[:5]}")
        return [samples[i] for i in ids], [predictions[i] for i in ids]

    val_samples, val_predictions = split(args.legacy_val)
    test_samples, test_predictions = split(args.legacy_test)

    result = evaluate_legacy_holdout(
        val_samples,
        val_predictions,
        test_samples,
        test_predictions,
        faith_fn=_score_fn(args.faith_expr, faith_score_fn),
        rel_fn=_score_fn(args.rel_expr, rel_score_fn),
        grid_step=args.grid_step,
    )
    payload = {
        "mode": "legacy_holdout",
        "t_faith": result.t_faith,
        "t_rel": result.t_rel,
        "val": vars(result.val),
        "test": vars(result.test),
    }
    _write(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )


def _print_summary(report) -> None:
    primary = report.primary
    print(f"{report.method}/{report.variant}: macro-F1(reliable) OOF")
    print(
        f"  {primary.value:.4f}  95% CI [{primary.ci95[0]:.4f}, {primary.ci95[1]:.4f}]"
        f"  (±{(primary.ci95[1] - primary.ci95[0]) / 2:.4f})"
    )
    print(
        f"  null percentile {primary.null_percentile:.1f}  "
        f"above_noise={primary.above_noise}"
    )
    print(
        f"  n_corpus {report.protocol['n_corpus']}  "
        f"n_evaluated {report.protocol['n_evaluated']}  "
        f"n_excluded {report.protocol['n_excluded']} (absent from folds.assignment)"
    )
    for comparison in report.comparisons:
        print(
            f"  vs {comparison['vs']}: Δ {comparison['delta']:+.4f} "
            f"[{comparison['ci95'][0]:+.4f}, {comparison['ci95'][1]:+.4f}] p={comparison['p']:.3f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_legacy(args) if args.legacy_holdout else run_cv(args)


if __name__ == "__main__":
    raise SystemExit(main())
