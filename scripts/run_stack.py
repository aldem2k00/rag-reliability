#!/usr/bin/env python3
"""OOF-стэкинг готовых scores.jsonl с абляциями и CI-отбором фич."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from rag_reliability.dataset import load_jsonl
from rag_reliability.evaluation.bootstrap import bootstrap_ci, paired_bootstrap
from rag_reliability.evaluation.protocol import (
    CVResult,
    evaluate_cv,
    fit_threshold,
    load_folds,
    metric_with_ci,
)
from rag_reliability.run_meta import git_state
from rag_reliability.schema import Prediction, RagSample
from rag_reliability.splits import sha256_file
from rag_reliability.stacking.collect import collect_features
from rag_reliability.stacking.stack import (
    FEATURE_SET_V1,
    make_prediction_fit_fn,
    select_features_by_ci,
)
from rag_reliability.thresholds import macro_f1_binary

_BASE_FEATURES = ("surf.p_faith", "surf.p_rel")
_M3_REL = "m3.p_rel"


def _parse_sources(values: Sequence[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(
                f"Source must have NAME=PATH form, got {value!r}"
            )
        if name in sources:
            raise ValueError(f"Duplicate source name {name!r}")
        sources[name] = Path(raw_path)
    return sources


def _parse_features(value: str) -> list[str]:
    features = [item.strip() for item in value.split(",") if item.strip()]
    if not features:
        raise ValueError("--features must contain at least one feature key")
    if len(features) != len(set(features)):
        raise ValueError("--features contains duplicates")
    return features


def _read_score_rows(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload: dict[str, Any] = json.loads(line)
                sample_id = str(payload["id"])
                scores = payload["scores"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid score row at {path}:{line_no}: {exc}") from exc
            if not isinstance(scores, dict):
                raise ValueError(
                    f"Invalid score row at {path}:{line_no}: 'scores' must be an object"
                )
            if sample_id in rows:
                raise ValueError(f"Duplicate id {sample_id!r} in {path}")
            rows[sample_id] = scores
    if not rows:
        raise ValueError(f"Score source is empty: {path}")
    return rows


def _available_features(
    sources: dict[str, Path],
    requested: Sequence[str],
) -> tuple[list[str], dict[str, dict[str, float]]]:
    active: list[str] = []
    rows_by_source: dict[str, dict[str, float]] = {}
    for feature in requested:
        source = feature.partition(".")[0]
        if source not in sources or not sources[source].is_file():
            continue
        if source not in rows_by_source:
            rows_by_source[source] = _read_score_rows(sources[source])
        active.append(feature)
    if not active:
        raise ValueError("None of the requested features has an available source")
    return active, rows_by_source


def _common_samples(
    samples: Sequence[RagSample],
    active_features: Sequence[str],
    rows_by_source: dict[str, dict[str, float]],
) -> list[RagSample]:
    active_sources = {feature.partition(".")[0] for feature in active_features}
    common_ids: set[str] | None = None
    for source in sorted(active_sources):
        ids = set(rows_by_source[source])
        common_ids = ids if common_ids is None else common_ids & ids
    if common_ids is None:
        raise ValueError("Cannot form a cohort without active score sources")
    cohort = [sample for sample in samples if sample.id in common_ids]
    if not cohort:
        raise ValueError("Available score sources share no ids with the corpus")
    return cohort


def _constant_score(prediction: Prediction) -> float:
    del prediction
    return 0.5


def _predictions_from_matrix(
    samples: Sequence[RagSample],
    matrix: np.ndarray,
    names: Sequence[str],
) -> list[Prediction]:
    return [
        Prediction(
            id=sample.id,
            faithfulness_pred=0,
            relevance_pred=0,
            scores=dict(zip(names, row, strict=True)),
        )
        for sample, row in zip(samples, matrix, strict=True)
    ]


def _evaluate_configuration(
    samples: Sequence[RagSample],
    predictions: Sequence[Prediction],
    folds: dict[str, Any],
    features: Sequence[str],
    *,
    model: str,
    calibrate: str,
    seed: int,
    grid_step: float,
) -> CVResult:
    return evaluate_cv(
        samples,
        predictions,
        folds,
        score_fn=_constant_score,
        fit_fn=make_prediction_fit_fn(
            features,
            model=model,
            calibrate=calibrate,
            seed=seed,
        ),
        grid_step=grid_step,
    )


def _bidirectional_noise_fit_apply(
    y_train: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    grid_step: float,
) -> np.ndarray:
    """Нулевая модель учитывает оба направления случайного ранжирования.

    Для одномерного шума logreg + Platt монотонны: после подбора порога класс
    решений совпадает с порогом по шуму либо по его инверсии. Это сохраняет
    сложность нулевой процедуры без тысяч вложенных GridSearchCV.
    """
    train = np.asarray(train_scores, dtype=float)
    test = np.asarray(test_scores, dtype=float)
    direct = fit_threshold(y_train, train, grid_step)
    inverse = fit_threshold(y_train, 1.0 - train, grid_step)
    if inverse.train_f1 > direct.train_f1:
        return ((1.0 - test) >= inverse.threshold).astype(int)
    return (test >= direct.threshold).astype(int)


def _metric_bundle(
    result: CVResult,
    *,
    bootstrap_b: int,
    seed: int,
) -> dict[str, Any]:
    f1 = bootstrap_ci(
        result.y,
        result.oof_pred,
        macro_f1_binary,
        B=bootstrap_b,
        seed=seed,
    )
    auc = bootstrap_ci(
        result.y,
        result.oof_scores,
        lambda y, scores: float(roc_auc_score(y, scores)),
        B=bootstrap_b,
        seed=seed,
    )
    return {
        "macro_f1": {"value": f1.point, "ci95": [f1.lo, f1.hi]},
        "roc_auc": {"value": auc.point, "ci95": [auc.lo, auc.hi]},
        "sd_across_repeats": float(np.std(result.per_repeat_f1)),
        "n": len(result.ids),
    }


def _delta_to_base(
    result: CVResult,
    base: CVResult,
    *,
    bootstrap_b: int,
    seed: int,
) -> dict[str, Any]:
    if result.ids != base.ids or not np.array_equal(result.y, base.y):
        raise ValueError("Ablation configurations must use the same ordered cohort")
    paired = paired_bootstrap(
        result.y,
        result.oof_pred,
        base.oof_pred,
        macro_f1_binary,
        B=bootstrap_b,
        seed=seed,
    )
    return {
        "value": paired.delta,
        "ci95": [paired.ci95[0], paired.ci95[1]],
        "p": paired.p,
    }


def _configuration_plan(
    active_features: Sequence[str],
    diagnostic_m3_rel: bool,
) -> list[tuple[str, tuple[str, ...], bool]]:
    base = tuple(_BASE_FEATURES)
    standard_candidates = [
        feature for feature in FEATURE_SET_V1 if feature not in base
    ]
    custom_candidates = [
        feature
        for feature in active_features
        if feature not in base and feature not in standard_candidates
    ]
    candidates = [*standard_candidates, *custom_candidates]
    plan: list[tuple[str, tuple[str, ...], bool]] = [
        ("surface (base)", base, True)
    ]
    plan.extend(
        (f"+ {feature}", (*base, feature), feature in active_features)
        for feature in candidates
    )
    all_features = tuple(active_features)
    plan.append(("all", all_features, True))
    if "m3.p_faith" in all_features:
        plan.append(
            (
                "all - m3.p_faith",
                tuple(feature for feature in all_features if feature != "m3.p_faith"),
                True,
            )
        )
    if diagnostic_m3_rel:
        plan.append(("+ m3.p_rel (diagnostic)", (*base, _M3_REL), True))
    return plan


def _write_scores(path: Path, result: CVResult, features: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for sample_id, score in zip(result.ids, result.oof_scores, strict=True):
            row = {
                "id": sample_id,
                "scores": {"stack.p_reliable": float(score)},
                "meta": {"features": list(features), "kind": "oof_mean"},
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_run_yaml(
    path: Path,
    args: argparse.Namespace,
    *,
    requested_features: Sequence[str],
    active_features: Sequence[str],
    final_features: Sequence[str],
    sources: dict[str, Path],
) -> None:
    state = git_state()
    payload = {
        "profile": "local",
        "seed": args.seed,
        "model": args.model,
        "calibrate": args.calibrate,
        "data": str(args.data),
        "folds": str(args.folds),
        "sources": {name: str(source) for name, source in sorted(sources.items())},
        "requested_features": list(requested_features),
        "available_features": list(active_features),
        "final_features": list(final_features),
        "git": state,
    }
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    sources = _parse_sources(args.sources)
    requested = _parse_features(args.features)
    active, rows_by_source = _available_features(sources, requested)
    for base_feature in _BASE_FEATURES:
        if base_feature not in active:
            raise ValueError(
                f"Required surface baseline feature {base_feature!r} is unavailable"
            )

    samples = _common_samples(load_jsonl(args.data), active, rows_by_source)
    diagnostic_m3_rel = False
    if "m3" in rows_by_source:
        diagnostic_m3_rel = all(
            _M3_REL in rows_by_source["m3"][sample.id]
            for sample in samples
        )
    collected_features = [*active]
    if diagnostic_m3_rel and _M3_REL not in collected_features:
        collected_features.append(_M3_REL)

    matrix, names = collect_features(
        sources,
        [sample.id for sample in samples],
        collected_features,
    )
    predictions = _predictions_from_matrix(samples, matrix, names)
    folds = load_folds(args.folds)

    result_cache: dict[tuple[str, ...], CVResult] = {}
    rows: list[dict[str, Any]] = []
    result_by_name: dict[str, CVResult] = {}
    plan = _configuration_plan(active, diagnostic_m3_rel)
    for configuration, features, available in plan:
        if not available:
            rows.append(
                {
                    "configuration": configuration,
                    "features": list(features),
                    "status": "unavailable",
                    "reason": "source artifact is absent",
                    "macro_f1": None,
                    "roc_auc": None,
                    "sd_across_repeats": None,
                    "n": None,
                    "delta_to_base": None,
                }
            )
            continue
        if features not in result_cache:
            result_cache[features] = _evaluate_configuration(
                samples,
                predictions,
                folds,
                features,
                model=args.model,
                calibrate=args.calibrate,
                seed=args.seed,
                grid_step=args.grid_step,
            )
        result = result_cache[features]
        result_by_name[configuration] = result
        row = {
            "configuration": configuration,
            "features": list(features),
            "status": "evaluated",
            **_metric_bundle(result, bootstrap_b=args.bootstrap_b, seed=args.seed),
        }
        rows.append(row)

    base = result_by_name["surface (base)"]
    increment_ci95: dict[str, tuple[float, float]] = {}
    for row in rows:
        configuration = str(row["configuration"])
        if row["status"] == "unavailable":
            continue
        if configuration == "surface (base)":
            row["delta_to_base"] = None
            continue
        result = result_by_name[configuration]
        delta = _delta_to_base(
            result,
            base,
            bootstrap_b=args.bootstrap_b,
            seed=args.seed,
        )
        row["delta_to_base"] = delta
        if configuration.startswith("+ ") and "(diagnostic)" not in configuration:
            feature = configuration.removeprefix("+ ")
            interval = delta["ci95"]
            increment_ci95[feature] = (float(interval[0]), float(interval[1]))

    final_features = select_features_by_ci(_BASE_FEATURES, increment_ci95)
    final_key = tuple(final_features)
    if final_key not in result_cache:
        result_cache[final_key] = _evaluate_configuration(
            samples,
            predictions,
            folds,
            final_features,
            model=args.model,
            calibrate=args.calibrate,
            seed=args.seed,
            grid_step=args.grid_step,
        )
    final = result_cache[final_key]
    primary = metric_with_ci(
        final,
        bootstrap_b=args.bootstrap_b,
        null_trials=args.null_trials,
        grid_step=args.grid_step,
        seed=args.seed,
        fit_apply_fn=_bidirectional_noise_fit_apply,
    )
    auc = _metric_bundle(
        final,
        bootstrap_b=args.bootstrap_b,
        seed=args.seed,
    )["roc_auc"]

    m3_rel_row = next(
        (
            row
            for row in rows
            if row["configuration"] == "+ m3.p_rel (diagnostic)"
        ),
        None,
    )
    if m3_rel_row is None:
        m3_rel_contributes: bool | None = None
    else:
        m3_rel_delta = m3_rel_row["delta_to_base"]
        m3_rel_contributes = float(m3_rel_delta["ci95"][0]) > 0.0

    report = {
        "schema_version": 1,
        "method": "stack",
        "variant": args.variant,
        "protocol": {
            "data": str(args.data),
            "data_sha256": sha256_file(args.data),
            "folds": str(args.folds),
            "folds_sha256": sha256_file(args.folds),
            "n_folds": int(folds["config"]["n_folds"]),
            "n_repeats": int(folds["config"]["n_repeats"]),
            "grid_step": args.grid_step,
        },
        "primary": {
            "value": primary.value,
            "ci95": list(primary.ci95),
            "null_percentile": primary.null_percentile,
            "above_noise": primary.above_noise,
        },
        "axes": {"roc_auc": auc},
        "diagnostics": {
            "per_repeat_f1": [float(value) for value in final.per_repeat_f1],
            "sd_across_repeats": float(np.std(final.per_repeat_f1)),
            "n_evaluated": len(final.ids),
            "n_excluded_by_folds": final.n_excluded,
            "bootstrap_B": args.bootstrap_b,
            "null_trials": args.null_trials,
        },
        "ablation": rows,
        "selection_rule": "include iff lower bound of paired 95% CI for macro-F1 increment > 0",
        "final_features": final_features,
        "m3_p_rel_contributes": m3_rel_contributes,
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _write_scores(output / "scores.jsonl", final, final_features)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_run_yaml(
        output / "run.yaml",
        args,
        requested_features=requested,
        active_features=active,
        final_features=final_features,
        sources=sources,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", required=True, metavar="NAME=PATH")
    parser.add_argument("--features", required=True, help="Comma-separated score keys")
    parser.add_argument("--model", choices=("logreg", "hgb"), default="logreg")
    parser.add_argument(
        "--calibrate",
        choices=("auto", "platt", "sigmoid", "isotonic"),
        default="auto",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", default="v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--bootstrap-b", type=int, default=10_000)
    parser.add_argument("--null-trials", type=int, default=500)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
