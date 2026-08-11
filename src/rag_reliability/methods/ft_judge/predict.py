"""Артефакт дообученного судьи: ``scores.jsonl`` общего контракта.

Метод отдаёт вероятности, а не решение. ``faithfulness_pred``/``relevance_pred``
остаются нулями намеренно: бинаризация — дело протокола (порог подбирается
внутри train-части фолда в ``evaluate_cv``). Ключи ``m3.p_faith``/``m3.p_rel``
те же, что у промптового судьи: это одно семейство методов, и стэкер должен
уметь подставить одно вместо другого, не меняя выражение скора.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rag_reliability.dataset import save_jsonl
from rag_reliability.schema import Prediction

FAITH_KEY = "m3.p_faith"
REL_KEY = "m3.p_rel"
SCORE_KEYS: tuple[str, ...] = (FAITH_KEY, REL_KEY)
SCORE_EXPR = "m3.p_faith * m3.p_rel"
PROB_METHOD = "ft_judge_logprobs"

#: Порог только для диагностики схлопывания. Отчётный порог подбирает протокол.
DECISION_THRESHOLD = 0.5


@dataclass(frozen=True)
class AxisProbs:
    """Вероятности обеих осей одного кейса и то, откуда они взялись."""

    p_faith: float
    p_rel: float
    method: str = PROB_METHOD
    marker: str | None = None

    def __post_init__(self) -> None:
        for name in ("p_faith", "p_rel"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a probability in [0, 1], got {value!r}")


def decisions_from_probs(probs: Mapping[str, AxisProbs]) -> list[Prediction]:
    """Бинарные решения ради диагностики схлопывания.

    Артефакт метода решений не содержит — их принимает протокол. Но
    ``degenerate_rate`` по определению смотрит на решения, и на артефактных
    нулях он показал бы схлопывание у любого прогона. Порог здесь
    диагностический, а не отчётный.
    """
    if not probs:
        raise ValueError("Cannot diagnose an empty set of probabilities")
    return [
        Prediction(
            id=sample_id,
            faithfulness_pred=int(item.p_faith >= DECISION_THRESHOLD),
            relevance_pred=int(item.p_rel >= DECISION_THRESHOLD),
        )
        for sample_id, item in probs.items()
    ]


def probs_to_predictions(probs: Mapping[str, AxisProbs]) -> list[Prediction]:
    """Вероятности -> строки артефакта. Порядок словаря сохраняется."""
    if not probs:
        raise ValueError("Cannot build an artifact from an empty set of probabilities")
    return [
        Prediction(
            id=sample_id,
            faithfulness_pred=0,
            relevance_pred=0,
            marker_pred=item.marker,
            faithfulness_prob=float(item.p_faith),
            relevance_prob=float(item.p_rel),
            prob_method=item.method,
            scores={FAITH_KEY: float(item.p_faith), REL_KEY: float(item.p_rel)},
        )
        for sample_id, item in probs.items()
    ]


def write_scores(predictions: Sequence[Prediction], path: str | Path) -> int:
    """Записать ``scores.jsonl`` и вернуть число строк."""
    save_jsonl(predictions, path)
    return len(predictions)
