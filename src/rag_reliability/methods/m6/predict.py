"""Строки фич Метода 6 -> Prediction со скорами. Без порогов и без бинаризации.

Раньше здесь стояли три хардкод-порога, причём ``contradiction_threshold = 0.5``
при медиане фичи 0.0099 — условие было истинно почти всегда, и faithfulness
определялся одной энтропией. Порог — дело протокола оценки
(``evaluation/protocol.py``), который подбирает его на train-части фолда;
метод отдаёт скоры, а бинарные поля остаются нулями (см. B2, ``scores_only``).

Фичи читаются строго: отсутствие ключа — исключение. Молчаливый дефолт
``features.get("selfcheck_contra_mean", 0.0)`` трактовал битую строку признаков
как идеально надёжный кейс.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from rag_reliability.schema import Prediction, RagSample

#: Фичи, без которых строка не является строкой Метода 6. Совпадают с
#: ``score_keys`` метода ``m6_selfcheck`` в реестре.
REQUIRED_FEATURE_KEYS: dict[str, str] = {
    "selfcheck_contra_mean": "m6.contra_mean",
    "semantic_entropy": "m6.entropy",
    "cos_q_a": "m6.cos_q_a",
}

#: Фичи, которые считались, но не использовались ни в одном решающем правиле
#: (карточка C4 §5). Считать их бесплатно — они переиспользуют ту же NLI-матрицу
#: и кластеризацию, — поэтому они не выбрасываются, а попадают в артефакт и
#: доступны стэкеру. Строка, где их нет (старый артефакт), остаётся валидной.
OPTIONAL_FEATURE_KEYS: dict[str, str] = {
    "selfcheck_contra_max": "m6.contra_max",
    "n_clusters": "m6.n_clusters",
    "answer_in_top_cluster": "m6.answer_in_top_cluster",
}


def load_features(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Method 6 features file not found: {path.resolve()}")
    features: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid features JSON at {path}:{line_no}: {exc}") from exc
            if "id" not in row:
                raise ValueError(f"Feature row at {path}:{line_no} has no id")
            features[str(row["id"])] = row
    return features


def _numeric(features: dict[str, Any], key: str, sample_id: str) -> float:
    try:
        value = features[key]
    except KeyError as exc:
        raise KeyError(
            f"Method 6 feature {key!r} is missing for sample {sample_id!r}; "
            f"present keys: {sorted(features)[:8]}"
        ) from exc
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(
            f"Method 6 feature {key!r} for sample {sample_id!r} must be a number, "
            f"got {type(value).__name__}"
        )
    return float(value)


def feature_scores(features: dict[str, Any], sample_id: str) -> dict[str, float]:
    """Скоры ``m6.*`` из строки фич. Обязательные ключи — строго, без дефолтов."""
    scores = {
        score_key: _numeric(features, feature_key, sample_id)
        for feature_key, score_key in REQUIRED_FEATURE_KEYS.items()
    }
    for feature_key, score_key in OPTIONAL_FEATURE_KEYS.items():
        if feature_key in features:
            scores[score_key] = _numeric(features, feature_key, sample_id)
    return scores


def _warn_deprecated_thresholds(**thresholds: float | None) -> None:
    passed = sorted(name for name, value in thresholds.items() if value is not None)
    if passed:
        warnings.warn(
            f"Method 6 no longer binarizes; threshold argument(s) {passed} are ignored. "
            "Thresholds are fitted per fold by evaluation/protocol.py. "
            "Remove them from the caller (see PR for task C4).",
            DeprecationWarning,
            stacklevel=3,
        )


def prediction_from_features(
    sample: RagSample,
    features: dict[str, Any],
    *,
    contradiction_threshold: float | None = None,
    entropy_threshold: float | None = None,
    relevance_threshold: float | None = None,
) -> Prediction:
    """Строка фич -> Prediction со скорами; бинарные поля остаются нулями.

    Пороговые аргументы сохранены только ради вызывающих скриптов вне владения
    задачи C4 (``scripts/run_m6_pipeline.py``, ``scripts/run_m6_selfcheck.py``):
    они игнорируются и предупреждают. Удаление флагов — за владельцами скриптов.
    """
    _warn_deprecated_thresholds(
        contradiction_threshold=contradiction_threshold,
        entropy_threshold=entropy_threshold,
        relevance_threshold=relevance_threshold,
    )
    scores = feature_scores(features, sample.id)
    contradiction = scores["m6.contra_mean"]
    cosine = scores["m6.cos_q_a"]
    return Prediction(
        id=sample.id,
        faithfulness_pred=0,
        relevance_pred=0,
        raw_output=json.dumps(features, ensure_ascii=False),
        invalid_output=False,
        # Вероятности остаются как свидетельство метода (монотонные преобразования
        # тех же фич), но решения из них больше не выводятся.
        faithfulness_prob=max(0.0, min(1.0, 1.0 - contradiction)),
        relevance_prob=max(0.0, min(1.0, cosine)),
        prob_method="m6_features",
        scores=scores,
    )


def predictions_from_feature_rows(
    samples: list[RagSample],
    features_by_id: dict[str, dict[str, Any]],
    *,
    contradiction_threshold: float | None = None,
    entropy_threshold: float | None = None,
    relevance_threshold: float | None = None,
) -> list[Prediction]:
    missing = [sample.id for sample in samples if sample.id not in features_by_id]
    if missing:
        raise ValueError(f"Missing Method 6 features for {len(missing)} sample(s): {missing[:5]}")
    _warn_deprecated_thresholds(
        contradiction_threshold=contradiction_threshold,
        entropy_threshold=entropy_threshold,
        relevance_threshold=relevance_threshold,
    )
    return [prediction_from_features(sample, features_by_id[sample.id]) for sample in samples]
