#!/usr/bin/env python
"""Не-LLM бейзлайны surface/majority: OOF-скоры по всему оцениваемому корпусу.

    python scripts/run_surface_baseline.py --variant surface \\
        --data data/organizers.jsonl --folds data/splits/folds.json \\
        --output predictions/alfa/baselines/surface/scores.jsonl

Артефакт короче корпуса намеренно: folds.json исключает oversized-группы, а
предсказать кейс out-of-fold, не имея для него фолда, нечем. Число строк
равно размеру оцениваемого корпуса и записывается в run.yaml.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import score  # noqa: E402

from rag_reliability.dataset import load_jsonl  # noqa: E402
from rag_reliability.methods import registry  # noqa: E402
from rag_reliability.methods.surface.features import FEATURE_KEYS, feature_vector  # noqa: E402
from rag_reliability.methods.surface.oof import (  # noqa: E402
    VARIANTS,
    corpus_sha256,
    evaluable_samples,
    load_folds,
    oof_probabilities,
)
from rag_reliability.schema import Prediction, RagSample  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=list(VARIANTS), default="surface")
    parser.add_argument("--data", default="data/organizers.jsonl")
    parser.add_argument("--folds", default="data/splits/folds.json")
    parser.add_argument("--output", required=True, help="Where to write scores.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Smoke run over the first N cases")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-yaml", default=None, help="Default: run.yaml next to --output")
    return parser.parse_args(argv)


def build_predictions(
    samples: list[RagSample],
    folds_path: str,
    *,
    variant: str,
    seed: int,
) -> list[Prediction]:
    """Фичи по всем кейсам + OOF-головы; бинарное решение не принимается."""
    folds = load_folds(folds_path)
    evaluable = evaluable_samples(samples, folds)
    if not evaluable:
        raise ValueError(
            f"None of the {len(samples)} sample(s) appear in {folds_path}; "
            "wrong corpus or wrong folds file"
        )

    features = np.asarray([feature_vector(sample) for sample in evaluable], dtype=float)
    probabilities = oof_probabilities(
        evaluable, features, folds, variant=variant, seed=seed
    )

    predictions: list[Prediction] = []
    for position, sample in enumerate(evaluable):
        p_faith, p_rel = probabilities[sample.id]
        scores: dict[str, float] = {}
        if variant == "surface":
            # majority — константа базовой ставки, поверхностные фичи к ней
            # отношения не имеют и в артефакте были бы шумом.
            scores = {
                f"surf.{key}": float(features[position, index])
                for index, key in enumerate(FEATURE_KEYS)
            }
        scores["surf.p_faith"] = p_faith
        scores["surf.p_rel"] = p_rel
        predictions.append(
            Prediction(
                id=sample.id,
                faithfulness_pred=0,
                relevance_pred=0,
                faithfulness_prob=p_faith,
                relevance_prob=p_rel,
                prob_method=f"surface_oof_{variant}",
                scores=scores,
            )
        )
    return predictions


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    method = "surface" if args.variant == "surface" else "majority"
    spec = registry.get(method)

    samples = load_jsonl(args.data)
    n_corpus = len(samples)
    if args.limit is not None:
        samples = samples[: args.limit]

    predictions = build_predictions(
        samples, args.folds, variant=args.variant, seed=args.seed
    )
    by_id = {prediction.id: prediction for prediction in predictions}
    scored = [sample for sample in samples if sample.id in by_id]

    score.score_samples(
        scored,
        lambda sample: by_id[sample.id],
        args.output,
        progress=False,
    )

    run_yaml = Path(args.run_yaml) if args.run_yaml else Path(args.output).parent / "run.yaml"
    # partial=True всегда: oversized-группы вне фолдов, полного покрытия корпуса не бывает.
    score.write_run_yaml(run_yaml, args, spec, n=len(predictions), partial=True)
    _append_coverage(run_yaml, n_corpus=n_corpus, n_scored=len(predictions), data=args.data)

    registry.validate_scores_file(args.output, spec, expected_n=len(predictions))
    print(
        f"Wrote {len(predictions)} OOF score(s) of {n_corpus} corpus case(s) "
        f"to {args.output}; meta: {run_yaml}"
    )
    return 0


def _append_coverage(run_yaml: Path, *, n_corpus: int, n_scored: int, data: str) -> None:
    """Покрытие пишется явно: 'partial' без числа читается как сбой прогона."""
    import yaml

    payload = yaml.safe_load(run_yaml.read_text(encoding="utf-8"))
    payload["coverage"] = {
        "corpus_n": n_corpus,
        "scored_n": n_scored,
        "excluded_n": n_corpus - n_scored,
        "reason": "cases outside data/splits/folds.json (oversized groups) cannot be scored OOF",
        "corpus_sha256": corpus_sha256(data),
    }
    run_yaml.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
