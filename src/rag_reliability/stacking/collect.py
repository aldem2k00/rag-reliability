"""Строгая сборка матрицы признаков из артефактов scores.jsonl."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_source(path: Path) -> dict[str, dict[str, float]]:
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
    return rows


def collect_features(
    scores_paths: dict[str, Path],
    sample_ids: list[str],
    feature_keys: list[str],
    *,
    required: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Собрать фичи в порядке корпуса, не маскируя битые или неполные строки.

    При ``required=False`` пропускается только целиком отсутствующий источник.
    Частично заполненный существующий артефакт остаётся ошибкой: иначе состав
    когорты незаметно зависел бы от конкретной фичи.
    """
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_ids contains duplicates")
    if len(feature_keys) != len(set(feature_keys)):
        raise ValueError("feature_keys contains duplicates")

    active: list[tuple[str, str, Path]] = []
    for feature_key in feature_keys:
        if "." not in feature_key:
            raise ValueError(
                f"Feature key must have '<source>.<signal>' form, got {feature_key!r}"
            )
        source = feature_key.partition(".")[0]
        if source not in scores_paths:
            if required:
                raise ValueError(
                    f"No score source {source!r} for required feature {feature_key!r}"
                )
            continue
        path = Path(scores_paths[source])
        if not path.is_file():
            if required:
                raise FileNotFoundError(
                    f"Score source {source!r} for feature {feature_key!r} does not exist: {path}"
                )
            continue
        active.append((feature_key, source, path))

    if not active:
        raise ValueError("No requested feature has an available score source")

    rows_by_source = {
        source: _load_source(path)
        for _, source, path in active
    }
    for source, rows in rows_by_source.items():
        missing_ids = [sample_id for sample_id in sample_ids if sample_id not in rows]
        if missing_ids:
            raise ValueError(
                f"Missing {len(missing_ids)} sample id(s) in source {source!r}: "
                f"{missing_ids[:5]}"
            )

    columns: list[list[float]] = []
    names: list[str] = []
    for feature_key, source, _ in active:
        rows = rows_by_source[source]
        missing_keys = [
            sample_id
            for sample_id in sample_ids
            if feature_key not in rows[sample_id]
        ]
        if missing_keys:
            raise ValueError(
                f"Missing feature {feature_key!r} for {len(missing_keys)} sample(s) "
                f"in source {source!r}: {missing_keys[:5]}"
            )
        try:
            column = [float(rows[sample_id][feature_key]) for sample_id in sample_ids]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Feature {feature_key!r} in source {source!r} contains a non-numeric value"
            ) from exc
        bad = [
            sample_ids[index]
            for index, value in enumerate(column)
            if not np.isfinite(value)
        ]
        if bad:
            raise ValueError(
                f"Feature {feature_key!r} contains {len(bad)} non-finite value(s): {bad[:5]}"
            )
        columns.append(column)
        names.append(feature_key)

    return np.asarray(columns, dtype=float).T, names
