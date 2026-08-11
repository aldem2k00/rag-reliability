#!/usr/bin/env python
"""Метод 6 (grounding): скоры опоры ответа на чанки по всему корпусу.

Прогон идёт через ``scripts/score.py``: инкрементальная запись, ``--resume``,
``run.yaml`` и валидация артефакта — общие с остальными методами, дублировать их
здесь нечего.

    python scripts/score_m6_grounding.py \\
        --data data/alfa.jsonl \\
        --output predictions/alfa/m6_grounding/base/scores.jsonl [--limit N] [--resume]

Первый в истории ветки ROC-AUC считается тем же скриптом по готовому артефакту:

    python scripts/score_m6_grounding.py --auc-only \\
        --data data/alfa.jsonl --scores predictions/alfa/m6_grounding/base/scores.jsonl \\
        --auc-report predictions/alfa/m6_grounding/base/auc.json

Метод не бинаризует: ``faithfulness_pred``/``relevance_pred`` остаются нулями,
порог подбирает ``scripts/evaluate_cv.py`` внутри train-части фолда.

Метод ``m6_grounding`` не зарегистрирован в ``methods/registry.py``: реестр
принадлежит задаче B2. Контракт объявлен здесь и проверяется той же
``registry.validate_scores_file`` (см. раздел «Требуется от других» в PR).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as score_cli  # noqa: E402

from rag_reliability.dataset import load_jsonl  # noqa: E402
from rag_reliability.methods import registry  # noqa: E402
from rag_reliability.methods.m6.coverage import COVERAGE_KEYS, coverage_features  # noqa: E402
from rag_reliability.methods.m6.grounding import GROUNDING_KEYS, compute_grounding  # noqa: E402
from rag_reliability.methods.surface.features import split_chunks  # noqa: E402
from rag_reliability.schema import Prediction, RagSample  # noqa: E402

DEFAULT_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

SCORE_KEYS: tuple[str, ...] = (*GROUNDING_KEYS, *COVERAGE_KEYS)


def _unregistered(_: registry.CommandContext) -> list[str]:
    raise NotImplementedError(
        "m6_grounding is scored by scripts/score_m6_grounding.py; "
        "registering it in methods/registry.py belongs to task B2"
    )


#: Контракт артефакта в терминах реестра — без записи в сам реестр.
SPEC = registry.MethodSpec(
    name="m6_grounding",
    label="Method 6 — NLI grounding + coverage",
    family="m6",
    mode=None,
    build_command=_unregistered,
    demo_runner=None,
    requires=("NLI model",),
    score_keys=SCORE_KEYS,
    default_score_expr="m6.min_entail",
    corpus_wide=True,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Corpus JSONL (RagSample records)")
    parser.add_argument("--output", default=None, help="Where to write scores.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Smoke run over the first N cases")
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Случайная подвыборка N кейсов (сид фиксирован --seed); для CPU-бюджета",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Перемешать порядок кейсов сидом: любой префикс прерванного прогона "
        "остаётся несмещённой выборкой корпуса",
    )
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted run")
    parser.add_argument("--flush-every", type=int, default=score_cli.DEFAULT_FLUSH_EVERY)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", default="base", help="Run label, recorded in run.yaml")
    parser.add_argument("--run-yaml", default=None, help="Default: run.yaml next to --output")

    parser.add_argument("--backend", choices=["real", "dummy"], default="real")
    parser.add_argument("--nli-model", default=DEFAULT_NLI_MODEL)
    parser.add_argument("--device", default=None, help="cpu / cuda / mps; auto by default")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--entail-threshold", type=float, default=0.5)

    parser.add_argument("--auc-report", default=None, help="Куда записать ROC-AUC с ДИ")
    parser.add_argument(
        "--auc-only",
        action="store_true",
        help="Не считать скоры: только ROC-AUC по готовому --scores",
    )
    parser.add_argument("--scores", default=None, help="Готовый scores.jsonl для --auc-only")
    parser.add_argument("--bootstrap-B", dest="bootstrap_b", type=int, default=10_000)
    args = parser.parse_args(argv)

    if args.auc_only:
        if not args.scores or not args.auc_report:
            parser.error("--auc-only requires --scores and --auc-report")
    elif not args.output:
        parser.error("--output is required unless --auc-only is given")
    return args


def build_nli(args: argparse.Namespace):
    """Реальный NLI-скорер или детерминированная заглушка для смоука."""
    if args.backend == "dummy":
        from rag_reliability.methods.m6.dummy import DummyNLI  # noqa: PLC0415

        return DummyNLI()
    try:
        from rag_reliability.methods.m6.nli import NLIScorer  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise ImportError('Install Method 6 deps with: uv pip install -e ".[m6]"') from exc
    return NLIScorer(
        args.nli_model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        overlap=args.overlap,
    )


def make_scorer(nli, *, entail_threshold: float) -> registry.Scorer:
    """Скорер одного кейса: 8 фич grounding + 4 фичи coverage на той же матрице."""

    def score_one(sample: RagSample) -> Prediction:
        chunks = split_chunks(sample.context)
        if not chunks:
            raise ValueError(
                f"sample {sample.id!r} has no context chunks; grounding cannot be computed"
            )
        grounding = compute_grounding(
            sample.answer, chunks, nli, entail_threshold=entail_threshold
        )
        scores = {
            **grounding.features,
            **coverage_features(
                sample.answer, chunks, source_chunk_ids=grounding.source_chunk_ids
            ),
        }
        return Prediction(
            id=sample.id,
            faithfulness_pred=0,
            relevance_pred=0,
            invalid_output=False,
            scores=scores,
        )

    return score_one


def select_samples(samples: list[RagSample], args: argparse.Namespace) -> list[RagSample]:
    """Подвыборка и/или перетасовка. Отбор идёт ДО --limit и фиксируется сидом."""
    if args.shuffle or args.subsample is not None:
        rng = random.Random(args.seed)
        samples = list(samples)
        rng.shuffle(samples)
    if args.subsample is not None:
        samples = samples[: args.subsample]
    if args.limit is not None:
        samples = samples[: args.limit]
    return samples


# --------------------------------------------------------------------------- #
# ROC-AUC с ДИ
# --------------------------------------------------------------------------- #


def auc_with_ci(labels, values, *, B: int, seed: int) -> dict[str, float]:
    """ROC-AUC и перцентильный бутстрэп-ДИ по кейсам.

    Число без интервала в этом репозитории не собирается (HANDOFF §8 П3), а на
    выборке в сотни кейсов ширина ДИ у AUC — единственное, что отличает сигнал
    от совпадения.
    """
    import numpy as np  # noqa: PLC0415
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    from rag_reliability.evaluation.bootstrap import bootstrap_ci  # noqa: PLC0415

    labels = np.asarray(labels, dtype=int)
    values = np.asarray(values, dtype=float)
    if len(set(labels.tolist())) < 2:
        raise ValueError("ROC-AUC needs both classes present in the evaluated subset")

    def metric(y, x):  # оба класса могут исчезнуть в реплике бутстрэпа
        if len(set(y.tolist())) < 2:
            return 0.5
        return float(roc_auc_score(y, x))

    result = bootstrap_ci(labels, values, metric, B=B, seed=seed)
    return {"auc": result.point, "ci95_lo": result.lo, "ci95_hi": result.hi}


def build_auc_report(
    samples: list[RagSample], scores_path: str | Path, *, B: int, seed: int
) -> dict:
    """AUC каждой фичи против faithfulness и reliable + счётчик кейсов."""
    rows = {}
    with Path(scores_path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["id"])] = row["scores"]

    evaluated = [sample for sample in samples if sample.id in rows]
    if not evaluated:
        raise ValueError(f"No overlap between {scores_path} and the corpus")

    report: dict = {
        "n": len(evaluated),
        "n_corpus": len(samples),
        "bootstrap_B": B,
        "seed": seed,
        "features": {},
    }
    targets = {
        "faithfulness": [sample.faithfulness for sample in evaluated],
        "reliable": [sample.reliable for sample in evaluated],
    }
    for key in SCORE_KEYS:
        missing = [sample.id for sample in evaluated if key not in rows[sample.id]]
        if missing:
            raise KeyError(
                f"score key {key!r} missing for {len(missing)} case(s): {missing[:5]}"
            )
        values = [rows[sample.id][key] for sample in evaluated]
        if len(set(values)) == 1:
            report["features"][key] = {"constant": True, "value": values[0]}
            continue
        report["features"][key] = {
            target: auc_with_ci(labels, values, B=B, seed=seed)
            for target, labels in targets.items()
        }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    samples: list[RagSample] = load_jsonl(args.data)
    n_corpus = len(samples)

    if not args.auc_only:
        samples = select_samples(samples, args)
        nli = build_nli(args)
        n = score_cli.score_samples(
            samples,
            make_scorer(nli, entail_threshold=args.entail_threshold),
            args.output,
            resume=args.resume,
            flush_every=args.flush_every,
        )
        # Стоимость ветки — часть её обоснования (90k пар против 404k у SelfCheck),
        # поэтому счётчики едут в run.yaml вместе с конфигом.
        args.nli_pairs = getattr(nli, "n_pairs", None)
        args.nli_windows = getattr(nli, "n_windows", None)
        # Фактические device и dtype: на CUDA скорер берёт fp16, на CPU fp32, и
        # скоры от этого отличаются. Артефакт без этой пометки нельзя сравнивать
        # с артефактом другого прогона — и нельзя дописывать через --resume.
        args.resolved_device = getattr(nli, "device", None)
        args.resolved_dtype = str(getattr(getattr(nli, "model", None), "dtype", None))
        run_yaml = Path(args.run_yaml) if args.run_yaml else Path(args.output).parent / "run.yaml"
        score_cli.write_run_yaml(
            run_yaml, args, SPEC, n=n, partial=len(samples) < n_corpus
        )
        registry.validate_scores_file(args.output, SPEC, expected_n=len(samples))
        print(f"Wrote {n} scored case(s) to {args.output}; meta: {run_yaml}")

    if args.auc_report:
        report = build_auc_report(
            load_jsonl(args.data),
            args.scores or args.output,
            B=args.bootstrap_b,
            seed=args.seed,
        )
        Path(args.auc_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.auc_report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Wrote ROC-AUC report for {report['n']} case(s) to {args.auc_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
