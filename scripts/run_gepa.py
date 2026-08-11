#!/usr/bin/env python
"""Эволюция инструкции судьи Метода 3 через GEPA (DSPy) и вердикт по H5.

Два режима.

``--mode evolve`` — эволюция промпта ОДНОЙ оси::

    python scripts/run_gepa.py --mode evolve \\
      --data data/organizers.jsonl --folds data/splits/folds.json \\
      --axis faithfulness --variant markers --seed 0 --auto medium \\
      --reflection-model Qwen/Qwen2.5-72B-Instruct

Сид эволюции — system-промпт оси из ``configs/prompts/{axis}.yaml``, то есть
ровно тот текст, который читает инференс. Результат — ``.txt``, который
``scripts/run_m3.py --mode gepa --prompt-file-{axis}`` подаёт обратно без
преобразований. Данные — train-часть фолда из ``folds.json``: D_pareto = 300 со
стратификацией по метке оси, held-out в оптимизацию не попадает.

``--mode h5`` — вердикт по гипотезе H5 (маркерный feedback против обычного) по
уже посчитанным scores.jsonl обоих плеч::

    python scripts/run_gepa.py --mode h5 \\
      --data data/organizers.jsonl --folds data/splits/folds.json \\
      --h5-markers .../markers_s0/scores.jsonl --h5-markers .../markers_s1/scores.jsonl \\
      --h5-plain .../plain_s0/scores.jsonl --h5-plain .../plain_s1/scores.jsonl \\
      --output-dir results/gepa

Стоп-правило: H5 отвергается ТОЛЬКО если верхняя граница 95% ДИ приращения
(markers − plain) ниже нуля. Точечная оценка сама по себе ничего не решает —
именно так был получен прошлый ложный отказ.

Режим ``evolve`` требует extra "gepa": ``uv pip install -e ".[gepa]"``.
Режим ``h5`` и ``--dry-run`` работают без dspy.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rag_reliability.dataset import load_jsonl
from rag_reliability.evaluation.protocol import (
    check_corpus_hash,
    compile_score_expr,
    default_score_fn,
    evaluate_cv,
    load_folds,
)
from rag_reliability.methods.m3.axes import AXES
from rag_reliability.methods.m3.gepa import (
    DEFAULT_PARETO_SIZE,
    DEFAULT_TRAIN_SIZE,
    H5Verdict,
    axis_gold,
    build_examples,
    build_optimization_sets,
    build_program,
    class_weights,
    extract_instruction,
    h5_verdict,
    load_marker_gloss,
    load_seed_instruction,
    load_stopwords,
    make_metric,
    prompt_defects,
    serialize_detailed,
)
from rag_reliability.methods.m3.gepa_report import render_report
from rag_reliability.run_meta import git_state
from rag_reliability.schema import Prediction
from rag_reliability.thresholds import macro_f1_binary

DEFAULT_FOLDS = Path("data/splits/folds.json")
DEFAULT_OUTPUT_DIR = Path("results/gepa")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=["evolve", "h5"], default="evolve")
    parser.add_argument("--data", type=Path, required=True, help="корпус с золотыми метками")
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="smoke-прогон на первых N кейсах")

    evolve = parser.add_argument_group("evolve")
    evolve.add_argument("--axis", choices=list(AXES), default=None)
    evolve.add_argument("--variant", choices=["markers", "plain"], default=None)
    evolve.add_argument("--repeat", type=int, default=0, help="повтор folds.json")
    evolve.add_argument("--fold", type=int, default=0, help="held-out фолд; в оптимизацию не идёт")
    evolve.add_argument("--pareto-size", type=int, default=DEFAULT_PARETO_SIZE)
    evolve.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    evolve.add_argument("--auto", choices=["light", "medium", "heavy"], default="medium")
    evolve.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    evolve.add_argument("--api-base", default="http://localhost:8000/v1")
    evolve.add_argument("--api-key-env", default="OPENAI_API_KEY")
    evolve.add_argument("--max-tokens", type=int, default=600)
    evolve.add_argument("--reflection-model", default=None, help="по умолчанию --model")
    evolve.add_argument("--reflection-api-base", default=None, help="по умолчанию --api-base")
    evolve.add_argument("--reflection-api-key-env", default=None, help="по умолчанию --api-key-env")
    evolve.add_argument("--reflection-max-tokens", type=int, default=8000)
    evolve.add_argument("--markers-gloss", type=Path, default=Path("configs/markers.yaml"))
    evolve.add_argument("--prompts-dir", type=Path, default=None, help="по умолчанию configs/prompts")
    evolve.add_argument("--max-context-chars", type=int, default=None)
    evolve.add_argument("--stopwords-file", type=Path, default=None, help="стоп-слова доменов")
    evolve.add_argument(
        "--fail-on-defects",
        action="store_true",
        help="ненулевой выход, если в промпте посторонние домены или нет якоря вердикта "
        "(артефакты прогона всё равно сохраняются)",
    )
    evolve.add_argument(
        "--dry-run",
        action="store_true",
        help="собрать наборы и проверить сид, не вызывая LLM (не требует dspy)",
    )

    h5 = parser.add_argument_group("h5")
    h5.add_argument("--h5-markers", type=Path, action="append", default=[], help="scores.jsonl")
    h5.add_argument("--h5-plain", type=Path, action="append", default=[], help="scores.jsonl")
    h5.add_argument("--score-expr", default=None, help='по умолчанию "<method>.p_faith * .p_rel"')
    h5.add_argument("--bootstrap-B", dest="bootstrap_b", type=int, default=10_000)

    args = parser.parse_args(argv)
    if args.mode == "evolve":
        if args.axis is None or args.variant is None:
            parser.error("--mode evolve requires --axis and --variant")
    else:
        if not args.h5_markers or not args.h5_plain:
            parser.error("--mode h5 requires --h5-markers and --h5-plain")
        if len(args.h5_markers) != len(args.h5_plain):
            parser.error(
                f"H5 arms must be paired by seed: {len(args.h5_markers)} markers run(s) "
                f"vs {len(args.h5_plain)} plain run(s)"
            )
    return args


# --------------------------------------------------------------------------- #
# Общее
# --------------------------------------------------------------------------- #


def run_meta(args: argparse.Namespace, extra: dict[str, Any]) -> dict[str, Any]:
    """Конфиг + git-хэш + seed рядом с артефактом (правило проекта)."""
    state = git_state()
    return {
        "args": {key: str(value) for key, value in sorted(vars(args).items())},
        "git_hash": state["hash"],
        "git_dirty": state["dirty"],
        "git_changed": state["changed"],
        "seed": args.seed,
        **extra,
    }


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def load_scores(path: str | Path) -> list[Prediction]:
    """Прочитать ``scores.jsonl``.

    Повторяет контракт ``scripts/evaluate_cv.load_scores``; вынести его в пакет
    D2 не может — файл принадлежит другой задаче (см. «Требуется от других»).
    Отсутствие скоров — ошибка, а не пустой словарь.
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
            if "scores" not in row:
                raise ValueError(f"Prediction at {path}:{line_number} has no 'scores'")
            predictions.append(
                Prediction(
                    id=str(row["id"]),
                    faithfulness_pred=0,
                    relevance_pred=0,
                    scores={str(key): float(value) for key, value in row["scores"].items()},
                )
            )
    if not predictions:
        raise ValueError(f"Scores file {path} is empty")
    return predictions


# --------------------------------------------------------------------------- #
# Режим evolve
# --------------------------------------------------------------------------- #


def _artifact_suffix(args: argparse.Namespace) -> str:
    """``{axis}_{variant}_seed{seed}``.

    Ось в имени обязательна: прогоны faithfulness и relevance идут раздельно и
    иначе затирали бы друг друга. Префиксы ``m3_gepa_{prompt,stats,report}``
    сохранены от прошлой схемы имён, чтобы ``scripts/gepa_report.py`` (файл
    чужой задачи) продолжал класть отчёт туда же, куда клал раньше.
    """
    return f"{args.axis}_{args.variant}_seed{args.seed}"


def _report_defects(defects: Sequence[str], where: str, fail: bool) -> None:
    if not defects:
        return
    lines = "\n".join(f"  - {defect}" for defect in defects)
    message = f"Санитария промпта ({where}):\n{lines}"
    if fail:
        raise ValueError(message)
    print(f"ПРЕДУПРЕЖДЕНИЕ. {message}")


def run_evolve(args: argparse.Namespace) -> None:  # noqa: PLR0915
    samples = load_jsonl(args.data)
    if args.limit is not None:
        samples = samples[: args.limit]
    folds = load_folds(args.folds)
    check_corpus_hash(folds, args.data)

    sets = build_optimization_sets(
        samples,
        folds,
        axis=args.axis,
        repeat=args.repeat,
        fold=args.fold,
        pareto_size=args.pareto_size,
        train_size=args.train_size,
        seed=args.seed,
    )
    seed_instruction = load_seed_instruction(args.axis, prompts_dir=args.prompts_dir)
    terms = load_stopwords(args.stopwords_file)
    _report_defects(
        prompt_defects(seed_instruction, args.axis, terms=terms), "сид", args.fail_on_defects
    )

    # Веса классов считаются по D_pareto: именно на нём GEPA ранжирует кандидатов.
    # normalize=True держит по-примерный скор в [0, 1] — порядок кандидатов от
    # деления на константу не меняется, а среднее остаётся пропорциональным
    # balanced accuracy.
    pareto_golds = [axis_gold(sample, args.axis) for sample in sets.pareto]
    weights = class_weights(pareto_golds, normalize=True)

    print(
        f"ось {args.axis}, вариант {args.variant}, seed {args.seed}\n"
        f"фолд: repeat {args.repeat}, fold {args.fold}\n"
        f"train {len(sets.train)}, D_pareto {len(sets.pareto)} "
        f"(позитивов {sum(pareto_golds)}), held-out {len(sets.held_out)} — не используется"
    )

    output_dir = Path(args.output_dir)
    suffix = _artifact_suffix(args)
    if args.dry_run:
        write_yaml(
            output_dir / f"run_gepa_{suffix}.yaml",
            run_meta(
                args,
                {
                    "dry_run": True,
                    "train_size": len(sets.train),
                    "pareto_size": len(sets.pareto),
                    "held_out_size": len(sets.held_out),
                    "pareto_positive_rate": sum(pareto_golds) / len(pareto_golds),
                },
            ),
        )
        print(
            "dry-run: наборы собраны, LLM не вызывалась\n"
            f"run.yaml: {output_dir / f'run_gepa_{suffix}.yaml'}"
        )
        return

    try:
        import dspy  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError('Install GEPA deps with: uv pip install -e ".[gepa]"') from exc

    api_key = os.environ.get(args.api_key_env, "")
    reflection_key = os.environ.get(args.reflection_api_key_env or args.api_key_env, "")
    task_lm = dspy.LM(
        f"openai/{args.model}",
        api_base=args.api_base,
        api_key=api_key,
        temperature=0.0,
        max_tokens=args.max_tokens,
    )
    reflection_lm = dspy.LM(
        f"openai/{args.reflection_model or args.model}",
        api_base=args.reflection_api_base or args.api_base,
        api_key=reflection_key,
        temperature=1.0,
        max_tokens=args.reflection_max_tokens,
    )
    dspy.configure(lm=task_lm)

    program = build_program(args.axis, seed_instruction)
    metric = make_metric(
        args.axis,
        use_markers=args.variant == "markers",
        gloss=load_marker_gloss(args.markers_gloss),
        weights=weights,
    )
    trainset = build_examples(
        sets.train, args.axis, prompts_dir=args.prompts_dir, max_context_chars=args.max_context_chars
    )
    valset = build_examples(
        sets.pareto, args.axis, prompts_dir=args.prompts_dir, max_context_chars=args.max_context_chars
    )

    gepa = dspy.GEPA(
        metric=metric,
        auto=args.auto,
        reflection_lm=reflection_lm,
        track_stats=True,
        seed=args.seed,
    )
    optimized = gepa.compile(program, trainset=trainset, valset=valset)

    instruction = extract_instruction(optimized)
    defects = prompt_defects(instruction, args.axis, terms=terms)

    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / f"m3_gepa_prompt_{suffix}.txt"
    prompt_path.write_text(instruction, encoding="utf-8")
    optimized.save(str(output_dir / f"m3_gepa_program_{suffix}.json"))

    stats = {
        "axis": args.axis,
        "variant": args.variant,
        "seed": args.seed,
        "metric": "balanced accuracy",
        "auto": args.auto,
        "repeat": args.repeat,
        "fold": args.fold,
        "train_size": len(trainset),
        "pareto_size": len(valset),
        "held_out_size": len(sets.held_out),
        "use_marker_feedback": args.variant == "markers",
        "task_model": args.model,
        "reflection_model": args.reflection_model or args.model,
        "task_lm_calls": len(getattr(task_lm, "history", []) or []),
        "reflection_lm_calls": len(getattr(reflection_lm, "history", []) or []),
        "git_hash": git_state()["hash"],
        "prompt_defects": defects,
        "seed_instruction": seed_instruction,
        "best_instruction": instruction,
        "detailed_results": serialize_detailed(getattr(optimized, "detailed_results", None)),
    }
    stats_path = output_dir / f"m3_gepa_stats_{suffix}.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    report_path = output_dir / f"m3_gepa_report_{suffix}.md"
    report_path.write_text(render_report(stats), encoding="utf-8")
    # run.yaml повторяет ключевые поля stats: конфиг прогона должен читаться без
    # разбора многомегабайтного json с текстами кандидатов.
    recorded = (
        "axis", "variant", "metric", "auto", "repeat", "fold", "train_size",
        "pareto_size", "held_out_size", "task_model", "reflection_model", "prompt_defects",
    )
    write_yaml(
        output_dir / f"run_gepa_{suffix}.yaml",
        run_meta(args, {key: stats[key] for key in recorded}),
    )

    print(
        f"prompt: {prompt_path}\nstats: {stats_path}\nreport: {report_path}\n"
        f"вызовы LM: task={stats['task_lm_calls']}, reflection={stats['reflection_lm_calls']}\n"
        f"инференс: scripts/run_m3.py --mode gepa --prompt-file-{args.axis} {prompt_path}"
    )
    # Претензии к результату разбираются ПОСЛЕ записи: прогон стоил часов GPU, и
    # терять его из-за стоп-слова в промпте нельзя. --fail-on-defects делает
    # выход ненулевым, чтобы автоматика не подхватила промпт молча.
    _report_defects(defects, "эволюционировавшая инструкция", args.fail_on_defects)


# --------------------------------------------------------------------------- #
# Режим h5
# --------------------------------------------------------------------------- #


def _oof_decisions(
    samples, folds, path: Path, score_fn
) -> tuple[list[str], np.ndarray, np.ndarray]:
    result = evaluate_cv(samples, load_scores(path), folds, score_fn=score_fn)
    return result.ids, result.oof_pred, result.y


def h5_table(verdict: H5Verdict, per_run: Sequence[tuple[str, int, Path, float]]) -> str:
    """Markdown-таблица H5 для PR: шесть прогонов, приращение, вердикт."""
    lines = [
        "# H5 — маркерный feedback против обычного",
        "",
        f"Кейсов в сравнении: {verdict.n_cases}. Пар (сидов): {verdict.n_seeds}.",
        "",
        "| вариант | сид | scores.jsonl | macro-F1 (OOF) |",
        "|---|---|---|---|",
    ]
    for variant, index, path, score in per_run:
        lines.append(f"| {variant} | {index} | `{path}` | {score:.4f} |")
    lines += [
        "",
        "## Парный бутстрэп markers − plain (по кейсам и по сидам)",
        "",
        f"- Δ = **{verdict.delta:+.4f}**",
        f"- 95% ДИ = [{verdict.ci95[0]:+.4f}, {verdict.ci95[1]:+.4f}]",
        f"- p = {verdict.p:.4f}",
        "",
        "## Вердикт",
        "",
        f"**{verdict.status}** — {verdict.conclusion}",
        "",
        "Стоп-правило: H5 отвергается только при верхней границе 95% ДИ ниже нуля.",
        "",
    ]
    return "\n".join(lines)


def run_h5(args: argparse.Namespace) -> None:
    samples = load_jsonl(args.data)
    if args.limit is not None:
        samples = samples[: args.limit]
    folds = load_folds(args.folds)
    check_corpus_hash(folds, args.data)
    score_fn = compile_score_expr(args.score_expr) if args.score_expr else default_score_fn

    reference_ids: list[str] | None = None
    reference_y: np.ndarray | None = None
    arms: dict[str, list[np.ndarray]] = {"markers": [], "plain": []}
    per_run: list[tuple[str, int, Path, float]] = []

    for variant, paths in (("markers", args.h5_markers), ("plain", args.h5_plain)):
        for index, path in enumerate(paths):
            ids, oof_pred, y = _oof_decisions(samples, folds, path, score_fn)
            if reference_ids is None:
                reference_ids, reference_y = ids, y
            elif ids != reference_ids:
                # Разное покрытие кейсов — самый дешёвый способ получить
                # «значимость» из разного состава, а не из разницы методов.
                raise ValueError(
                    f"{path} covers a different set of cases than {args.h5_markers[0]}; "
                    "paired comparison requires identical coverage"
                )
            arms[variant].append(oof_pred)
            per_run.append((variant, index, path, macro_f1_binary(y, oof_pred)))

    if reference_y is None:
        raise ValueError("No H5 runs were loaded")
    verdict = h5_verdict(
        reference_y, arms["markers"], arms["plain"], B=args.bootstrap_b, seed=args.seed
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "h5_markers_vs_plain.md"
    table_path.write_text(h5_table(verdict, per_run), encoding="utf-8")
    json_path = output_dir / "h5_markers_vs_plain.json"
    json_path.write_text(
        json.dumps(
            {
                "delta": verdict.delta,
                "ci95": list(verdict.ci95),
                "p": verdict.p,
                "n_seeds": verdict.n_seeds,
                "n_cases": verdict.n_cases,
                "status": verdict.status,
                "conclusion": verdict.conclusion,
                "runs": [
                    {"variant": variant, "seed": index, "scores": str(path), "macro_f1": score}
                    for variant, index, path, score in per_run
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_yaml(output_dir / "run_h5.yaml", run_meta(args, {"status": verdict.status}))
    print(verdict.conclusion)
    print(f"таблица: {table_path}\njson: {json_path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "evolve":
        run_evolve(args)
    else:
        run_h5(args)


if __name__ == "__main__":
    main()
